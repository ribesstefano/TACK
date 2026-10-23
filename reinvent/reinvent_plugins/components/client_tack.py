"""Minimal client for a `tack serve` endpoint.

Standard library only, on purpose: this file is meant to be copied into a REINVENT (or
any other) environment, and the entire reason the server exists is to avoid installing
anything of TACK's beside those dependencies. Adding a `requests` import here would
undercut that for no benefit — one POST of JSON is `urllib`-sized work.

Its name does not start with ``comp_``, so REINVENT's plugin scan ignores it; it is
imported by ``comp_tack.py`` beside it rather than registered as a component of its own.

Self-contained and free of REINVENT imports, so it can be exercised on its own, which is
the quickest way to check a server before involving REINVENT at all:

    python tack_client.py --url http://127.0.0.1:8777 "CCO" "c1ccccc1"
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_URL = "http://127.0.0.1:8777"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_BATCH = 256


class TackServerError(RuntimeError):
    """The server could not be reached, or answered with an error."""


class TackClient:
    """Scores molecules against a running `tack serve` process.

    Args:
        url: Base URL of the server.
        context: Experimental context to send with every request (keys ``poi_seq``,
            ``e3_seq``, ``cell_id``, ``assay``, ``assay_time``). Omit when the server was
            started with ``--context``, which is the usual arrangement.
        task: Which task's predictions to read — ``"dmax"``, ``"pdc50"`` or
            ``"activity"``. The server returns every task it loaded in one response, so
            this selects rather than requests.
        key: Which of that task's output keys to read, e.g. ``"mean"``/``"std"`` for a
            regression task or ``"proba"`` for ``activity``. ``None`` takes the natural
            point prediction — ``mean`` where present, else ``proba`` — so the caller
            need not know which kind of task it asked for.
        timeout: Per-request timeout in seconds. Generous by default: a first request
            pays for featurising a whole population with a cold descriptor cache.
        retries: Extra attempts after a connection-level failure. Covers a server
            restart mid-run; it does not retry a 4xx, which would only fail again.
        max_batch: Split larger populations into chunks of this size, so one oversized
            request cannot hit a proxy or body limit.
    """

    def __init__(self, url: str = DEFAULT_URL, context: Optional[Dict] = None,
                 task: str = "dmax", key: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT, retries: int = 2,
                 max_batch: int = DEFAULT_MAX_BATCH):
        self.url = url.rstrip("/")
        self.context = context
        self.task = task
        self.key = key
        self.timeout = timeout
        self.retries = retries
        self.max_batch = max_batch

    @classmethod
    def from_port_file(cls, path: str, host: str = "127.0.0.1", **kwargs) -> "TackClient":
        """Build a client for a server started with ``--port 0 --port-file PATH``."""
        port = Path(path).read_text().strip()
        return cls(url=f"http://{host}:{port}", **kwargs)

    # ------------------------------------------------------------------ transport

    def _request(self, path: str, payload: Optional[Dict] = None) -> Dict:
        """One JSON round trip, retried on connection-level failures only."""
        url = f"{self.url}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                # The server answered: a retry would produce the same reply. Surface the
                # body, which carries FastAPI's validation detail.
                detail = e.read().decode(errors="replace")[:2000]
                raise TackServerError(f"{url} returned HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise TackServerError(
            f"Could not reach {url} after {self.retries + 1} attempts ({last}). "
            "Is `tack serve` running, and pointed at the same host/port?"
        ) from last

    # -------------------------------------------------------------------- queries

    def health(self) -> bool:
        """Whether the server answers its liveness probe."""
        try:
            return self._request("/health").get("status") == "ok"
        except TackServerError:
            return False

    def info(self) -> Dict:
        """What the server is serving (task, model count, calibration state)."""
        return self._request("/info")

    def wait_until_ready(self, timeout: float = 300.0, interval: float = 2.0) -> Dict:
        """Block until the server answers, then return its ``/info``.

        Startup is not instant — the ensemble may be downloaded from the Hub and the
        context tables read from cache — so poll rather than guessing at a `sleep`.

        Raises:
            TackServerError: If the server does not answer within `timeout`.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health():
                return self.info()
            time.sleep(interval)
        raise TackServerError(
            f"{self.url} did not become ready within {timeout:.0f}s. Start it with "
            "`tack serve --repo-id ... --task ...` and check its log."
        )

    def select(self, response: Dict, task: str,
               key: Optional[str] = None) -> List[Optional[float]]:
        """Pull one task and output key out of a full response.

        Separate from :meth:`predict` so a caller that wants several values — every task,
        or a prediction and its uncertainty — reads them out of **one** response instead
        of requesting the same batch again per value.

        Args:
            response: Decoded body from :meth:`predict`.
            task: Task to read: ``"dmax"``, ``"pdc50"`` or ``"activity"``.
            key: Output key within that task. ``None`` takes the task's point prediction
                (``mean``, or ``proba`` for a classification task).

        Returns:
            The chosen list of values, one per molecule.

        Raises:
            TackServerError: If the server did not load that task, or the task has no
                such output key — both configuration mistakes worth naming precisely,
                since the alternative is a ``KeyError`` from inside a scoring loop.
        """
        predictions = response.get("predictions", {})
        if task not in predictions:
            raise TackServerError(
                f"Server did not return task {task!r}; it serves {sorted(predictions)}. "
                f"Start it with --task {task}, or point this client at a task it loaded."
            )
        task_out = predictions[task]
        if key is None:
            key = "mean" if "mean" in task_out else "proba"
        if key not in task_out:
            raise TackServerError(
                f"Task {task!r} has no output key {key!r}; it returns {sorted(task_out)}."
            )
        return task_out[key]

    def predict(self, smiles: Sequence[str]) -> Dict:
        """Score molecules once and return the server's whole response.

        Every loaded task comes back from a single request, so a caller needing several
        tasks or several keys pays for one round trip and one featurisation rather than
        one per value. Chunks larger than ``max_batch`` are merged back into a single
        response, so the result is shaped as if one request had carried the whole batch.

        Args:
            smiles: SMILES strings, in the order the values should come back.

        Returns:
            ``{"tasks": [...], "predictions": {task: {key: [...]}}, "n_invalid": int}``,
            with every list as long as `smiles` and ``None`` where a SMILES did not parse.

        Raises:
            TackServerError: On an unreachable server, an HTTP error, or a response whose
                value lists do not line up with the molecules sent.
        """
        smiles = list(smiles)
        if not smiles:
            return {"tasks": [], "predictions": {}, "n_invalid": 0}

        merged: Optional[Dict] = None
        for start in range(0, len(smiles), self.max_batch):
            chunk = smiles[start:start + self.max_batch]
            payload: Dict = {"smiles": chunk}
            if self.context is not None:
                payload["context"] = self.context
            response = self._request("/predict", payload)
            self._check_alignment(response, len(chunk))
            if merged is None:
                merged = response
                continue
            for task, task_out in response.get("predictions", {}).items():
                for key, values in task_out.items():
                    merged["predictions"][task][key].extend(values)
            merged["n_invalid"] = merged.get("n_invalid", 0) + response.get("n_invalid", 0)
        return merged

    def _check_alignment(self, response: Dict, n: int) -> None:
        """Refuse a response whose lists do not match the molecules sent.

        Alignment is the one invariant a generative loop cannot recover from: it needs
        one value per proposed molecule and has no way to tell which one went missing.
        """
        for task, task_out in response.get("predictions", {}).items():
            for key, values in task_out.items():
                if len(values) != n:
                    raise TackServerError(
                        f"Server returned {len(values)} values for {n} molecules "
                        f"({task}.{key})."
                    )

    def score(self, smiles: Sequence[str]) -> List[Optional[float]]:
        """Score molecules against this client's own `task`/`key`, one value per input.

        A convenience over :meth:`predict` plus :meth:`select` for a caller that wants a
        single series. Reading several series this way would re-request the batch each
        time, so prefer `predict` + `select` for that.

        Returns:
            A list as long as `smiles`, holding ``None`` wherever the server could not
            parse the SMILES. Callers that need a float array should map ``None`` to
            ``nan`` themselves — the distinction between "unparseable" and "scored 0"
            matters too much to collapse here.
        """
        smiles = list(smiles)
        if not smiles:
            return []
        return self.select(self.predict(smiles), self.task, self.key)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Score SMILES from the command line, to check a server end to end."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="tack_client", description="Score SMILES against a running `tack serve`.")
    parser.add_argument("smiles", nargs="*", help="SMILES to score.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL}).")
    parser.add_argument("--port-file", default=None,
                        help="Read the port from this file instead of --url.")
    parser.add_argument("--context", default=None,
                        help="Context as a JSON object, if the server has no default.")
    parser.add_argument("--task", default="dmax", help="Task to read (default: dmax).")
    parser.add_argument("--key", default=None,
                        help="Output key to read (default: the task's point prediction).")
    parser.add_argument("--info", action="store_true", help="Print /info and exit.")
    args = parser.parse_args(argv)

    context = json.loads(args.context) if args.context else None
    kwargs = {"context": context, "task": args.task, "key": args.key}
    client = (TackClient.from_port_file(args.port_file, **kwargs)
              if args.port_file else TackClient(args.url, **kwargs))

    if args.info or not args.smiles:
        print(json.dumps(client.info(), indent=2))
        return 0
    for smile, score in zip(args.smiles, client.score(args.smiles)):
        print(f"{smile}\t{'invalid' if score is None else f'{score:.4f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
