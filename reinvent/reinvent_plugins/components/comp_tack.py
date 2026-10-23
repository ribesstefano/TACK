"""REINVENT4 scoring component backed by a `tack serve` endpoint.

Installed by putting this file's *grandparent* directory on ``PYTHONPATH`` — nothing is
copied into REINVENT's own install. REINVENT discovers scoring components through native
namespace packages: it scans every ``reinvent_plugins/components`` directory it finds
across ``sys.path``, imports the files whose names start with ``comp_`` (which is why
this file is ``comp_tack.py`` and ``tack_client.py`` is not picked up as a component),
and registers the classes tagged ``@add_tag("__component")``.

That mechanism depends on ``reinvent_plugins`` and ``components`` having **no**
``__init__.py``: an init file turns the namespace package into a regular one and REINVENT's
own directory then stops merging with this one, so the component silently disappears. Do
not add one.

See ``README.md`` two directories up for the install and configuration steps.

Why a server and not REINVENT's own external-process component: that component spawns a
fresh process per scoring call, which would re-import rdkit and re-deserialise the
ensemble thousands of times over a run. Here the model stays warm and each call is a
localhost round trip.

The component is intentionally thin — every HTTP concern lives in
:class:`tack_client.TackClient`, which has no REINVENT imports and can be run on its own
to check a server before wiring it into a config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import numpy as np

# `add_tag` and `component_results` live in REINVENT's own installed
# `reinvent_plugins/components`, a different sys.path entry than this file. The relative
# form still reaches them: both directories contribute to the same namespace package, so
# `.add_tag` resolves across the merge. `tack_client` sits next to this file and resolves
# on this side of it.
from .add_tag import add_tag
from .component_results import ComponentResults
from .client_tack import TackClient, TackServerError

logger = logging.getLogger(__name__)


def _per_endpoint(values: List, n: int, default):
    """Spread a `Parameters` field across `n` endpoints, filling in a default.

    REINVENT collects each endpoint's ``params`` into parallel lists on the dataclass,
    one entry per endpoint. A key omitted from the TOML arrives as a short list or as
    ``None`` entries, so every field is resolved through here rather than indexed
    directly.
    """
    out = []
    for i in range(n):
        value = values[i] if i < len(values) else None
        out.append(default if value is None else value)
    return out


@add_tag("__parameters")
@dataclass
class Parameters:
    """Per-endpoint parameters, as REINVENT assembles them from the TOML config.

    Every field is a list with one entry per endpoint. All are optional: the defaults
    assume a server on ``127.0.0.1:8777`` started with its own ``--context``, which is
    the arrangement the README recommends.
    """

    url: List[str] = field(default_factory=list)
    port_file: List[str] = field(default_factory=list)
    context: List[Union[Dict, str, None]] = field(default_factory=list)
    task: List[str] = field(default_factory=list)
    key: List[str] = field(default_factory=list)
    timeout: List[float] = field(default_factory=list)
    max_batch: List[int] = field(default_factory=list)
    fail_on_error: List[bool] = field(default_factory=list)
    ready_timeout: List[float] = field(default_factory=list)


@add_tag("__component")
class TackDegradation:
    """Scores molecules with a TACK fusion ensemble served over HTTP.

    Every endpoint is served from **one** request per population. The server returns all
    the tasks it loaded in a single response, so an objective reading `dmax`, `pdc50` and
    `activity` — and any of their uncertainty keys — costs one round trip and one
    featurisation, not one per endpoint. Adding an endpoint is then free at run time; it
    only selects a different slice of a response already in hand.

    Because of that, every endpoint shares one server connection: `url`/`port_file`,
    `context`, `timeout` and `max_batch` describe the component, not the endpoint, and
    are taken from whichever endpoint states them. Endpoints that disagree on a server
    would need separate requests and so defeat the point; that is refused rather than
    silently honoured. `task` and `key` are what legitimately vary per endpoint.
    """

    def __init__(self, params: Parameters):
        # The endpoint count is whatever the longest configured list implies; a config
        # that sets no params at all still describes one endpoint.
        n = max((len(getattr(params, f.name)) for f in fields(params)), default=0) or 1
        self.tasks = _per_endpoint(params.task, n, "dmax")
        # None lets the client pick the task's own point prediction (`mean` or `proba`),
        # so a config need not know whether the task is regression or classification.
        self.keys = _per_endpoint(params.key, n, None)

        url = _one_value(params.url, "url")
        port_file = _one_value(params.port_file, "port_file")
        context = _parse_context(_one_value(params.context, "context"))
        timeout = _one_value(params.timeout, "timeout") or 300.0
        max_batch = _one_value(params.max_batch, "max_batch") or 256
        # 60s rather than the client's own 300s default: here the usual cause of no
        # answer is a wrong URL in the config, and a long wait just delays the report.
        ready_timeout = _one_value(params.ready_timeout, "ready_timeout") or 60.0
        fail_on_error = _one_value(params.fail_on_error, "fail_on_error")
        self.fail_on_error = True if fail_on_error is None else bool(fail_on_error)

        common = dict(context=context, timeout=timeout, max_batch=max_batch)
        if port_file:
            self.client = TackClient.from_port_file(port_file, **common)
        else:
            self.client = TackClient(url or "http://127.0.0.1:8777", **common)

        # Fail here rather than at the first scoring call: a config pointing at a server
        # that was never started should stop the run before it samples.
        info = self.client.wait_until_ready(timeout=ready_timeout)
        served = info.get("tasks", {})
        missing = sorted({t for t in self.tasks if t not in served})
        if missing:
            raise TackServerError(
                f"{self.client.url} serves {sorted(served)}, so it cannot answer for "
                f"{missing}. Restart it without --task, which loads all three tasks."
            )
        for task, key in zip(self.tasks, self.keys):
            logger.info(
                f"TackDegradation endpoint -> {self.client.url} (task={task}, "
                f"key={key or 'default'}, n_models={served[task].get('n_models')})"
            )
        self.number_of_endpoints = n

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """Score one population, returning one array per endpoint from one request.

        An unparseable SMILES becomes ``nan``, REINVENT's own convention for "no score
        for this molecule", rather than a number the optimiser would chase.
        """
        try:
            response = self.client.predict(smilies)
            series = [self.client.select(response, task, key)
                      for task, key in zip(self.tasks, self.keys)]
        except TackServerError:
            if self.fail_on_error:
                raise
            # Opt-in degradation: keep a long run alive across a server restart by
            # scoring the batch as unavailable. Off by default, because silently
            # returning all-nan for every remaining step wastes the run instead.
            logger.exception(
                f"{self.client.url} failed; scoring {len(smilies)} molecules as nan "
                "(fail_on_error = false)."
            )
            series = [[None] * len(smilies) for _ in self.tasks]
        return ComponentResults(
            [np.array([np.nan if v is None else float(v) for v in values], dtype=float)
             for values in series]
        )


def _one_value(values: List, name: str):
    """Collapse a per-endpoint list into the single value this component can use.

    Args:
        values: The field as REINVENT assembled it, one entry per endpoint.
        name: Field name, for the error message.

    Returns:
        The one distinct non-``None`` value, or ``None`` if no endpoint set it.

    Raises:
        TackServerError: If endpoints set conflicting values. Honouring both would mean
            one request per endpoint, which is exactly what this component exists to
            avoid, so it is better to say so than to quietly become slower.
    """
    distinct = {v for v in values if v is not None and v != ""}
    if len(distinct) > 1:
        raise TackServerError(
            f"Endpoints disagree on {name!r} ({sorted(map(str, distinct))}). It applies "
            "to the whole component, since all endpoints share one request. Set it on "
            "one endpoint, or use a second TackDegradation component for the other "
            "server."
        )
    return distinct.pop() if distinct else None


def _parse_context(context: Union[Dict, str, None]) -> Optional[Dict]:
    """Accept a context as a TOML table, a JSON string, or a path to a JSON file.

    TOML's inline tables cover the common case, but a long POI sequence is far easier to
    keep in a file than inline in a config, so a path is accepted too.
    """
    if context is None or isinstance(context, dict):
        return context
    import json
    from pathlib import Path

    text = context if context.lstrip().startswith("{") else Path(context).read_text()
    return json.loads(text)
