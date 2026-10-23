# Guidelines: serving a PROTAC scorer to REINVENT4

**Audience: Claude Code, working in a different repository.** You are building an HTTP
server that scores PROTAC SMILES with one regression value, plus a REINVENT4 scoring
plugin that calls it. This file records what was learned building the same thing for
TACK (see `README.md`, `reinvent_plugins/`, `tackai/serve/` in this repo as the worked
reference) so you do not rediscover it.

Read the whole file before writing code. Most of the traps below fail **silently** or
fail on *every request for the life of the server*, not at the point of the mistake.

## The target

A model that maps SMILES → one float. Simpler than TACK's case, which also took an
experimental context (protein, cell line, assay) and served three tasks. **What drops
out:** context encoding, context memoisation, a `/predict/records` endpoint for
per-row contexts, protein-embedding-space flags, and per-task response nesting. **What
stays:** everything in "Server rules" and "Component rules" below.

If the model does report an uncertainty, serve it as its own key — REINVENT can log it
as a zero-weight endpoint, which is genuinely useful (see "Weights" below).

## Why a server at all

Two reasons, in order of importance:

1. **Dependency isolation.** REINVENT4 pins its own `torch` and `rdkit`. If your model
   needs different versions, co-installing is at best fragile. Concretely: `torch` and
   `xgboost` each bundle a `libomp` and **segfault** when both run multi-threaded in one
   process. A separate process makes the collision impossible.
2. **A warm model.** REINVENT ships an external-process scoring component that could
   just invoke your CLI, but it spawns a process per scoring call — re-importing rdkit
   and re-deserialising the model thousands of times. A server pays that once.

If your model shares REINVENT's exact dependency versions and loads in milliseconds, a
server may be unnecessary. Check before building one.

## Build order

```
yourrepo/
├── yourpkg/serve/           # server: app.py + schemas.py, imported lazily
└── reinvent/
    ├── scoring.toml
    └── reinvent_plugins/            # NO __init__.py
        └── components/              # NO __init__.py
            ├── comp_yourmodel.py    # the component; MUST start with comp_
            └── client.py            # stdlib-only HTTP client, NOT a comp_ file
```

1. Server (`yourpkg/serve/`), with a `<cli> serve` subcommand.
2. Stdlib-only client, runnable standalone as a CLI — this is what you debug with.
3. The `comp_*.py` component wrapping the client.
4. The TOML.
5. Tests for the server; a namespace-package stub for the component.

## Server rules

- **Gate the dependency.** Put `fastapi`/`uvicorn` in an optional extra (`[serve]`) and
  import them inside the subcommand, turning `ImportError` into a message naming the
  extra. Keep `--help` working without it.
- **Set `OMP_NUM_THREADS=1` in the entry point** *before* importing xgboost or torch, if
  both could end up in the process. After the import it is too late.
- **One worker, and a lock around prediction.** Each uvicorn worker holds its own model
  copy and its own featuriser cache, so replicating workers multiplies memory and throws
  away cache hits. Define endpoints `def` (not `async def`) so starlette runs them in a
  threadpool, and take a `threading.Lock` around predict — concurrent batches otherwise
  race on any memo cache the featuriser keeps. Scale with batch size, not concurrency.
- **Validate everything knowable at load time, at load time.** The single worst bug in
  the TACK version: the design matrix was the wrong width, which surfaced as an xgboost
  `Feature shape mismatch` 500 on *every request for the server's whole life*. It was
  knowable before listening. Compare the feature count the model was fitted on
  (`n_features_in_`) against what you build, and refuse to start, with a message naming
  the likely cause and the flag that fixes it. Same for a default context, a missing
  artifact, a task the checkpoint does not hold.
- **An invalid SMILES is data, not an error.** Return `null` in place and keep the list
  length. A 500 kills a run at step 1400 over one malformed string. Also map non-finite
  floats to `null`: `NaN` is not valid JSON.
- **Return the model's own output shape.** Do not rename `mean`→`score` or fold extra
  keys into an `extra` bag. A client that knows the Python API should know the payload,
  and a key added to the model should appear without a server change.
- **Bind `127.0.0.1`.** There is no authentication. On a shared node `0.0.0.0` exposes
  inference to everyone on the network.
- **Offer `--port 0` with `--port-file PATH`.** Bind the socket yourself, read the real
  port, write it, then hand the socket to `uvicorn.Server.run(sockets=[sock])`. This is
  what makes several jobs per node workable. Remove the file on shutdown.
- **`/health` (cheap), `/info` (what is loaded), `POST /predict`.** `/info` should report
  enough to diagnose a mismatch: model source, feature count, version, output keys.
- **Close pools in a lifespan handler**, or worker processes leak across restarts.
- **Give `Dict[...]` response fields an explicit example.** Swagger renders
  `additionalProp1`/`"string"` placeholders for a mapping, because it has no field names
  to show. Set `model_config = ConfigDict(json_schema_extra={"example": {...}})` with real
  values. Make the request example work unmodified so "Try it out" at `/docs` succeeds.

## Component rules

These come from REINVENT's plugin mechanism. Breaking any of the first three makes the
component **silently absent** — no error naming the cause.

- **The file must be named `comp_*.py`.** REINVENT imports only those; any other name is
  skipped. Keep the client under a non-`comp_` name so it is *not* registered, and import
  it from the component.
- **No `__init__.py`** in `reinvent_plugins/` or `components/`. They are native namespace
  packages; that is what merges your directory with REINVENT's own. Add an init file and
  yours shadows REINVENT's, giving
  `ModuleNotFoundError: No module named 'reinvent_plugins.components.add_tag'`.
- **`PYTHONPATH` points at the directory *containing* `reinvent_plugins`** — not at
  `components/`. Nothing is copied into REINVENT's install.
- **Relative imports work across the merge.** `from .add_tag import add_tag` resolves to
  REINVENT's own file on a different `sys.path` entry. No try/except fallback needed.
- **Tag with `@add_tag("__component")`**; at most one `@add_tag("__parameters")` dataclass
  per file. Several component classes per file are allowed.
- **`Parameters` fields are parallel lists, one entry per endpoint.** A key omitted from
  the TOML arrives as a short list or `None` entries, so resolve every field through a
  helper that fills a default, rather than indexing. Endpoint count = longest list, or 1.
- **`__call__(self, smilies) -> ComponentResults([arr, ...])`**, one `np.array` per
  endpoint, `dtype=float`, `nan` for unscorable molecules (REINVENT's own convention).

### One request for every endpoint

Do not create a client per endpoint and call it per endpoint — that is N requests per
population, each recomputing the same featurisation. Instead: **one** request, then slice
the response per endpoint. Split the client into `predict(smiles)` (full response) and
`select(response, key)` (one series), and have `__call__` call `predict` once.

This makes connection settings (`url`, `port_file`, `timeout`, `max_batch`) properties of
the **component**, not the endpoint, while `key` varies per endpoint. Take each from
whichever endpoint sets it, and **refuse conflicting values with a clear error** — quietly
honouring two URLs means a request each, undoing the whole design. Document that a second
server needs a second component.

### Client rules

- **Standard library only.** The point is to add nothing to REINVENT's environment; one
  JSON POST is `urllib`-sized work.
- **Retry connection failures, never HTTP 4xx.** A 4xx will fail identically; surface its
  body, which carries FastAPI's validation detail.
- **Check alignment and raise if it breaks.** One value per molecule is the invariant a
  generative loop cannot recover from — it cannot tell which molecule lost its score.
- **Chunk above `max_batch` and merge back** into one response shaped like a single reply.
  Note: chunking is not bit-reproducible — xgboost blocks prediction by row count, so
  values can differ in the last ULP (~1e-16). Harmless, worth documenting.
- **`wait_until_ready()` polling `/health`, called at component construction.** A config
  pointing at a server that was never started must stop the run *before* it samples, not
  thousands of steps in. Use a short timeout (~60 s) here: the usual cause is a wrong URL.
- **`fail_on_error` defaults to `true`.** The `false` path (score the batch `nan`) keeps a
  run alive across a server restart, but if the server stays down it silently wastes every
  remaining step. Opt-in only.

## Weights and transforms in the TOML

- Give the regression endpoint a `transform` — that is where the objective actually lives.
  `sigmoid` with `low`/`high` bracketing indifference→good is the usual shape.
- If you serve an uncertainty key, add it as a **zero-weight** endpoint. Logging it is
  valuable: a calibrated model's rising uncertainty is the earliest sign a run has drifted
  out of the training domain. A *positive* weight would teach the optimiser to propose
  molecules near the training set rather than good ones. Say so in a comment.
- Add property bounds (MW, SlogP) matching the chemistry the model was trained on. Without
  them a generative run drifts to molecules the model has no basis for scoring — for
  PROTACs, roughly MW 600–1200.
- `geometric_mean` aggregation stops a molecule buying a good total on one component while
  failing another.

## Verify without a REINVENT install

You almost certainly cannot run REINVENT. You can still verify your half:

1. **Two-directory namespace stub.** Create `stub/reinvent_plugins/components/` with
   minimal `add_tag.py` and `component_results.py`, **no `__init__.py`**, on a *separate*
   path entry from yours. Then `PYTHONPATH=stub:yourdir` and run the check REINVENT's docs
   prescribe: `from reinvent_plugins.components import comp_yourmodel`. Print
   `reinvent_plugins.components.__path__` to confirm both entries merged.
2. **Confirm the negative.** Add an `__init__.py`, see the `ModuleNotFoundError`, remove
   it. Do not assert this from memory.
3. **Simulate discovery.** `pkgutil.walk_packages` over the namespace path, import
   `comp_*`, collect classes with `__component`, assert yours is found.
4. **Count HTTP calls.** Monkeypatch the client's request method and assert N endpoints
   produce **one** `/predict` call.
5. **Drive from the real TOML.** Parse it with `tomllib`, build `Parameters` the way
   REINVENT does (parallel lists), assert no `params.*` key is unknown to the dataclass,
   and run the component against a live server.
6. **Run your README's commands verbatim**, from the directory it tells the reader to be
   in. Doc drift is as real as code drift.
7. Clean up `__pycache__` afterwards — stale `__init__` bytecode in a namespace package is
   confusing to find later.

## Be explicit about what is unverified

The REINVENT-side TOML keys (`learning_strategy`, `transform.*` coefficients,
`termination`, built-in components) cannot be validated without a REINVENT install. Say so
in the README, in a form the reader cannot miss, and mark the transform windows as
placeholders for the project's own thresholds. Do not present a schema written from
documentation as if it had been run.
