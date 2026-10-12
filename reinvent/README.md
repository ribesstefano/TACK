# Scoring REINVENT molecules with a TACK fusion ensemble

This directory is a REINVENT4 **scoring plugin** that scores molecules with a TACK
fusion ensemble **without installing `tackai` into the REINVENT environment**:

```
reinvent/                                   <- put THIS directory on PYTHONPATH
├── staged_learning_tack.toml               <- ready-to-edit REINVENT config
├── example_context.json
└── reinvent_plugins/                       <- no __init__.py (native namespace package)
    └── components/                         <- no __init__.py
        ├── comp_tack.py                    <- the component; must start with comp_
        └── tack_client.py                  <- plain HTTP client, not a component
```

| File | Where it runs | Dependencies |
|---|---|---|
| `tack_client.py` | REINVENT environment | standard library only |
| `comp_tack.py` | REINVENT environment | `numpy` + REINVENT's own plugin interfaces |

The model itself stays in a separate TACK environment behind `tack serve`. Nothing here
imports `tackai`.

Three constraints come from REINVENT's plugin mechanism, and breaking any of them makes
the component silently fail to appear:

- The component file **must** be named `comp_*.py`. REINVENT imports only those; any
  other name is skipped. `tack_client.py` is deliberately not a `comp_` file, so it is
  imported by `comp_tack.py` rather than registered as a component.
- There must be **no `__init__.py`** in `reinvent_plugins` or `components`. They are
  native namespace packages, which is what lets this directory merge with REINVENT's own
  `reinvent_plugins/components`. Add an init file and this directory shadows REINVENT's
  instead, so `from .add_tag import add_tag` fails with `ModuleNotFoundError`.
- The directory that goes on `PYTHONPATH` is the one *containing* `reinvent_plugins`,
  i.e. `reinvent/`, not `reinvent/reinvent_plugins/components/`.

## Why a server

REINVENT4 pins its own `torch` and `rdkit`. Installing `tackai` beside them is not just
untidy: `torch` and `xgboost` each bundle a copy of `libomp`, and the two segfault when
both run multi-threaded in one process.
A separate process makes that collision impossible.

REINVENT already ships an external-process scoring component that could call
`tack predict` directly, but it spawns a fresh process per scoring call — re-importing
rdkit and re-deserialising the ensemble thousands of times over a run. Here the model is
loaded once and each call is a localhost round trip.

## 1. Start the server (TACK environment)

Install the extra once:

```bash
cd /path/to/TACK
uv pip install -e ".[serve]"     # fastapi + uvicorn on top of the base deps; no torch
```

Then serve one ensemble:

```bash
tack serve \
    --repo-id ailab-bio/TACK-Fast-Ensemble \
    --context reinvent/example_context.json \
    --port 8777
```

That loads **all three tasks** — `dmax`, `pdc50` and `activity` — and every request
returns all of them. Each task's ensemble comes from the subfolder of the same name, so
the task name *is* the subfolder; there is no separate flag that could disagree with it.
Pass `--task dmax` (repeatable) to load fewer.

Serving three tasks is cheaper than three servers, but not free. On 128 freshly proposed
molecules against the published ensembles: featurising costs ~710 ms and dominates, while
predicting rises from ~60 ms for one task to ~195 ms for three — about **18% more per
request** than a single task. If your scoring function only ever reads `dmax`, say
`--task dmax`.

Use `--checkpoints-dir DIR` instead of `--repo-id` for a local ensemble. `tack serve`
sets `OMP_NUM_THREADS=1` itself before importing xgboost, so no wrapper is needed.

The context embedding tables come from `ailab-bio/TACK-Context-Encoding` by default,
because the published ensembles and the published tables are a matched pair: the tables'
manifest records the protein embedding space the models were trained in, and that space
fixes how wide the POI and E3 blocks are.

Interactive API docs are at <http://127.0.0.1:8777/docs> and the OpenAPI schema at
`/openapi.json`, so you can exercise the endpoint from a browser before wiring up
REINVENT.

### The context file

The experimental context is fixed for a whole generative run, so give it to the server
once and let REINVENT post only SMILES:

```json
{
  "poi_seq": "MAGEGDQQDAAHNMGNHLPLLPADSEDEDDEIEMEVEDQ...",
  "e3_seq": "MADEEAGGTERMEISAELPQTPQRLASWWDQQVDFYTAFL...",
  "cell_id": "CVCL_0001",
  "assay": "western blot",
  "assay_time": 24.0
}
```

Keys match the fields of a fusion record. To see what the cached tables actually know:

```python
from tackai.fusion.data import FusionData
d = FusionData()
d.help()                                        # overview
d.help(block="cell_id", return_keys=True)       # the cell lines the cache holds
d.help(block="assay", return_keys=True)         # the closed assay vocabulary
```

Missing-key behaviour is deliberately asymmetric: a blank cell line gets the "unknown"
vector, an absent POI accession raises. A context the encoder rejects fails at **startup**
rather than on the first request.

> **Keep the context file out of the ensemble directory.** `XGBStackedEnsemble` treats
> every `.json` in its directory except `config.json` as a model file, so a stray
> `context.json` there makes loading fail with an xgboost stack trace.

### Ports on a shared node

Let the OS pick a port and publish it, so several jobs need no port bookkeeping:

```bash
tack serve --repo-id ... --task dmax --port 0 --port-file /tmp/tack-$SLURM_JOB_ID.port
```

The client then reads that file (`port_file` in the TOML below). The file is removed on
shutdown. A Unix socket works too, via `--uds /tmp/tack.sock`.

Keep the default `--host 127.0.0.1`. The endpoint has **no authentication**; on a shared
machine `--host 0.0.0.0` would expose it to every other user on the network.

## 2. Install the plugin (REINVENT environment)

Nothing is copied into REINVENT's install. Put this directory on `PYTHONPATH` and
REINVENT picks the component up on its own:

```bash
export PYTHONPATH=/path/to/TACK/reinvent:$PYTHONPATH
```

Add it to the shell profile or the job script that launches REINVENT so it survives a
new session. Then verify the import, which is the check REINVENT's own docs prescribe —
its log reports an import failure, so confirming it directly is quicker than reading the
log:

```bash
python -c "from reinvent_plugins.components import comp_tack; print('ok')"
```

If that raises `ModuleNotFoundError: No module named 'reinvent_plugins.components.add_tag'`,
an `__init__.py` has crept into `reinvent_plugins/` or `components/` — delete it.

Check the server is reachable from that environment before touching any config:

```bash
python reinvent_plugins/components/tack_client.py --url http://127.0.0.1:8777 --info
python reinvent_plugins/components/tack_client.py --url http://127.0.0.1:8777 \
    "CCO" "c1ccccc1" "bad-smiles"
```

`tack_client.py` is standard-library only and imports nothing from REINVENT, so it runs
anywhere — including from the TACK environment itself.

## 3. Configure the scoring function

[`staged_learning_tack.toml`](staged_learning_tack.toml) is a complete staged-learning
config. Copy it, point `prior_file`/`agent_file` at your own prior, and adjust the
transforms to your project's thresholds.

Its TACK component declares four endpoints, all answered by **one** request per
population:

```toml
[[stage.scoring.component]]
[stage.scoring.component.TackDegradation]

[[stage.scoring.component.TackDegradation.endpoint]]
name = "Dmax"
weight = 3.0
params.url = "http://127.0.0.1:8777"
params.task = "dmax"
transform.type = "sigmoid"
transform.low = 0.50
transform.high = 0.90
transform.k = 0.5

[[stage.scoring.component.TackDegradation.endpoint]]
name = "pDC50"
weight = 2.0
params.task = "pdc50"
transform.type = "sigmoid"
transform.low = 6.0
transform.high = 9.0
transform.k = 0.5

[[stage.scoring.component.TackDegradation.endpoint]]
name = "P(active)"
weight = 1.0
params.task = "activity"          # a probability is already a score: no transform

[[stage.scoring.component.TackDegradation.endpoint]]
name = "Dmax std"
weight = 0.0                      # reported, not optimised
params.task = "dmax"
params.key = "std"
```

### One request for every endpoint

The server returns every task it loaded in a single response, and the component makes
**one** `/predict` call per population and then slices that reply per endpoint. Four
endpoints cost one round trip and one featurisation, so adding an endpoint is free at run
time. Verified by counting HTTP calls while driving the component from this exact TOML:
4 endpoints, 1 call.

That is why `url`, `port_file`, `context`, `timeout` and `max_batch` configure the
**component**, not the endpoint: set them on one endpoint (the first is the natural
place). Endpoints that disagree on one are refused with an explicit error rather than
silently falling back to a request each — which would defeat the whole arrangement. Only
`task` and `key` legitimately vary per endpoint. To score against two different servers,
declare a second `TackDegradation` component.

### Weights and transforms

`geometric_mean` aggregation means a near-zero on any weighted component sinks the
product, so a molecule cannot earn a good total by being potent while degrading nothing.
The weights above say degradation matters most, then potency, then the classifier.

The uncertainty endpoint carries `weight = 0.0` deliberately. Logging it is valuable —
the published ensembles are calibrated, so rising `std` is the earliest sign a run has
drifted into chemistry the training set never covered. But giving it a positive weight
would teach the optimiser to propose molecules *near the training set* rather than good
ones. Filter on uncertainty after the run instead.

### Parameters

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:8777` | Server base URL. |
| `port_file` | — | Read the port from this file; overrides `url`. |
| `context` | — | Context as a TOML table, a JSON string, or a path to a JSON file. Only needed when the server has no `--context`. Component-wide. |
| `task` | `dmax` | Which task's predictions to read: `dmax`, `pdc50` or `activity`. The server must have loaded it. **Per endpoint.** |
| `key` | the task's point prediction | Which of the task's output keys to read — `mean`, `std`, `std_noise`, `std_disagreement`, `lower`/`upper` when calibrated; `proba` and the entropy terms for `activity`. Left unset it takes `mean`, or `proba` for a classification task. **Per endpoint.** |
| `timeout` | `300.0` | Per-request timeout, seconds. Component-wide. |
| `max_batch` | `256` | Split larger populations into chunks, merged back into one response. Component-wide. |
| `ready_timeout` | `60.0` | How long to wait for the server at startup. Component-wide. |
| `fail_on_error` | `true` | `false` scores a batch as `nan` instead of aborting the run when the server is unreachable. Component-wide. |

Only `task` and `key` vary per endpoint; everything else describes the shared connection
and is taken from whichever endpoint sets it.

## 4. Run it

```bash
# TACK environment, one terminal
tack serve --repo-id ailab-bio/TACK-Fast-Ensemble \
    --context /path/to/TACK/reinvent/example_context.json --port 8777

# REINVENT environment, another terminal
export PYTHONPATH=/path/to/TACK/reinvent:$PYTHONPATH
reinvent -l staged_learning_tack.log staged_learning_tack.toml
```

The server logs one `POST /predict` line per step. If you see several per step, the
endpoints disagree on `url`/`port_file` and each is talking to its own connection.

> **What is and is not verified here.** The component, the client, the one-call
> behaviour, the plugin discovery and the `params.*` names were all exercised against a
> live `tack serve` and a faithful namespace-package layout. The REINVENT-side keys —
> `learning_strategy`, `transform.*` coefficients, `termination`, the `MolecularWeight`
> and `SlogP` blocks — are written from REINVENT4's documented schema but have **not**
> been run through a real REINVENT install, since that environment is not available here.
> Treat them as a starting point to check against your REINVENT version, and the
> `transform` windows as placeholders for your project's own thresholds.

## The response shape

`POST /predict` returns each loaded task's entry exactly as
`XGBStackedEnsemble.predict` produces it — nothing is renamed or dropped, so a client
that knows the Python API already knows this payload:

```json
{
  "tasks": ["dmax", "pdc50", "activity"],
  "predictions": {
    "dmax":     {"mean": [0.936], "std": [0.109], "std_noise": [0.097],
                 "std_disagreement": [0.049], "lower": [0.807], "upper": [1.066]},
    "pdc50":    {"mean": [8.881], "std": [0.854], "std_noise": [0.740],
                 "std_disagreement": [0.426], "lower": [6.856], "upper": [10.906]},
    "activity": {"proba": [0.872], "entropy_total": [0.382],
                 "entropy_aleatoric": [0.154], "entropy_epistemic": [0.156]}
  },
  "n_invalid": 0
}
```

`lower` and `upper` appear only for a calibrated regression ensemble (the published ones
are). `GET /info` lists each task's `output_keys` up front, so a client can check what it
will get without scoring anything.

## Behaviour worth knowing

**Invalid SMILES become `nan`, never an error.** A molecule RDKit cannot parse gets
`nan` in place, which is REINVENT's own convention for "no score". The returned array is
always the same length as the population, in the same order — a generative loop cannot
recover from a silently shortened array, so the client raises if the server ever breaks
that alignment.

**The component fails at construction, not mid-run.** It polls `/health` when REINVENT
builds the scoring function, so a config pointing at a server that was never started
stops the run before it samples rather than thousands of steps in.

**`fail_on_error = false` is opt-in for a reason.** It keeps a long run alive across a
server restart, but if the server stays down it quietly scores every remaining step as
`nan` — wasting the run rather than failing it. Leave it `true` unless you are
deliberately trading that off.

**One request per population, not per molecule — and not per endpoint.** The server
vectorises over rows, so a 256-molecule request costs barely more than a 1-molecule one.
The component sends the whole population in one call and reads every endpoint out of that
one response.

**`max_batch` is not bit-reproducible.** A population split into chunks can differ from an
unsplit one in the last floating-point digit (measured max difference 8.9e-16), because
xgboost blocks its prediction differently by row count. That is ~14 orders of magnitude
below the ensembles' own uncertainty, so it matters only if you are diffing outputs
exactly.

**The server handles one request at a time.** Requests are serialised by a lock, since
concurrent batches would race on the featuriser's memo cache. That is the intended
single-client arrangement — scale with batch size, not with concurrent clients.

## Troubleshooting

| Symptom | Cause |
|---|---|
| REINVENT never mentions `TackDegradation`, and an unknown-component error names it | The plugin was not discovered. Check `PYTHONPATH` points at `reinvent/` (not at `components/`), and that the file is still named `comp_tack.py`. |
| `ModuleNotFoundError: No module named 'reinvent_plugins.components.add_tag'` | An `__init__.py` exists in `reinvent_plugins/` or `components/`, so this directory shadows REINVENT's instead of merging with it. Delete it. |
| `Could not reach .../predict ... Is tack serve running?` | Server not started, or a different host/port than the config names. |
| `did not become ready within 60s` | Server still loading (a first `--repo-id` run downloads from the Hub), or the URL is wrong. Raise `ready_timeout` or check the server log. |
| HTTP 422 `No context given and the server was started without a default` | Pass `--context` to the server, or `params.context` in the TOML. |
| HTTP 422 `Could not encode context` | A POI/E3/cell key the cached tables do not hold. Check with `FusionData.help(block=...)`. |
| HTTP 422 on a context you believe is right | A typo'd key. Unknown fields are rejected rather than silently defaulted — the fields are `poi_seq`, `e3_seq`, `cell_id`, `assay`, `assay_time`. |
| xgboost stack trace at startup | A stray `.json` in the ensemble directory (see the context-file warning above). |
| `The loaded models were fitted on N columns but this server builds M` | The protein embedding space does not match the one the models were trained in. Drop `--local-context` to use the published tables, or pass `--protein-space combined`. |
| `Feature shape mismatch, expected: 1400, got 1354` from xgboost | The same cause, from a server started before this check existed. Restart it. |
| Every score is `nan` | Either the SMILES genuinely do not parse, or `fail_on_error = false` is masking an unreachable server — check the REINVENT log for the warning. |
