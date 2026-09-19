# Loading a Pretrained Ensemble

[`EnsemblePredictor.from_pretrained`][tackai.ensemble_predictor.EnsemblePredictor.from_pretrained]
is the single entry point for loading an ensemble, whether it lives on the
Hugging Face Hub or in a local directory.

```python
from tackai.ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles", subfolder="dmax_caruana",
)
```

## How dispatch works

`from_pretrained` decides what `model_id` means by checking the filesystem
first:

- **If `model_id` is an existing local path**, it is loaded directly — no
  network access, equivalent to the old `from_directory` (still available as
  a deprecated alias).
- **Otherwise**, `model_id` is treated as a Hugging Face Hub repo id. The
  checkpoints are fetched with `huggingface_hub.snapshot_download`, honoring
  `HF_HOME`/`HF_HUB_CACHE` exactly like `datasets`/`transformers` downloads.

## Available Hub ensembles

[`ailab-bio/TACK-ensembles`](https://huggingface.co/ailab-bio/TACK-ensembles)
has one subfolder per task × selection-strategy combination:

| `subfolder` | Task | Selection strategy |
|---|---|---|
| `dmax_caruana` | Dmax (%) | [Caruana greedy forward selection](https://github.com/ribesstefano/TACK) across architectures |
| `dmax_best_arch` | Dmax (%) | Fold-models from whichever single architecture/data-config scored best overall |
| `dc50_caruana` | DC50 (nM) | Caruana greedy forward selection across architectures |
| `dc50_best_arch` | DC50 (nM) | Fold-models from whichever single architecture/data-config scored best overall |
| `bin_caruana` | Binary activity | Caruana greedy forward selection across architectures |
| `bin_best_arch` | Binary activity | Fold-models from whichever single architecture/data-config scored best overall |

Caruana selection typically mixes architectures/feature sets and tends to
have better-calibrated ensemble diversity; `best_arch` is a simpler
single-architecture ensemble. Both are valid `EnsemblePredictor`s — pick
based on your own validation if you're unsure.

## What gets downloaded

Two kinds of assets live on the Hub, mirroring the Zenodo `ensembles.zip` /
`cache.zip` split described in the repository README:

1. **Model checkpoints** (from `model_id`/`subfolder`) — per-fold model
   files (`.ckpt` for Lightning/MLP, `.json`/`.ubj` for XGBoost) plus the
   paired `DegradationComplexDataModule` state (`*_hparams.yaml` /
   `*_state.pt`, fitted scalers/encoders) and an `ensemble_weights_*.json`
   with the per-model Caruana/uniform weights.
2. **Shared cache assets** (from `cache_repo_id`, default
   [`ailab-bio/TACK-cache`](https://huggingface.co/datasets/ailab-bio/TACK-cache)) —
   precomputed protein (POI/E3) and cell-line embeddings, Morgan fingerprint
   caches, RDKit descriptors, and (only if a loaded datamodule actually needs
   it, e.g. `cell_features: description`) the ~370 MB Cellosaurus lookup
   tables. `from_pretrained` inspects the downloaded `*_hparams.yaml` files
   first and only fetches the cache files those specific models require.

After downloading, `from_pretrained` points the `TACKAI_CACHE` environment
variable at the downloaded cache snapshot for the rest of the process —
**overriding** any `TACKAI_CACHE` already set (e.g. via `.env`), since that
value would otherwise silently shadow the freshly downloaded, known-complete
snapshot. Pass `cache_repo_id=None` to opt out and keep your own
`TACKAI_CACHE` in charge.

## Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `model_id` | — | Local directory, or a Hub repo id such as `"ailab-bio/TACK-ensembles"`. |
| `subfolder` | `None` | Subdirectory within the Hub repo for one ensemble (e.g. `"dmax_caruana"`). Ignored for local paths — point `model_id` at the checkpoints directory directly instead. |
| `revision` | `None` | Hub revision (branch/tag/commit) for `model_id`. |
| `cache_repo_id` | `"ailab-bio/TACK-cache"` | Dataset repo with the shared cache assets. `None` skips downloading cache assets entirely. |
| `cache_revision` | `None` | Hub revision for `cache_repo_id`. |
| `token` | `None` | Hub auth token for private repos; defaults to whatever `huggingface_hub` resolves from the environment/CLI login. |
| `weights_file` | `None` | Explicit `ensemble_weights_*.json`. If omitted and exactly one such file exists alongside the checkpoints, it's used automatically. |
| `device` | `"cpu"` | `"cpu"` or `"cuda"`. |
| `n_jobs` | `None` | XGBoost thread count (`None` = all cores). |
| `pattern` | `None` | Optional regex to filter which model files get loaded. |
| `hparam_overrides` | `None` | Hparam key/value pairs applied to every datamodule before instantiation — e.g. to point at a different `poi_embeddings_file`. Caller-supplied values win over anything `from_pretrained` infers. |
| `lazy_loading` | `True` | See [Lazy vs. eager loading](#lazy-vs-eager-loading) below. |

## Lazy vs. eager loading

By default (`lazy_loading=True`), models and their datamodules are loaded
from disk **one at a time** during `predict()` and freed immediately after —
peak memory stays proportional to a single model rather than the whole
ensemble (ensembles can have dozens of fold-models). Set
`lazy_loading=False` to load everything upfront if you're doing many
back-to-back prediction calls and have the RAM to spare.

```python
predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles",
    subfolder="dmax_caruana",
    lazy_loading=False,  # load every fold-model into memory once, up front
)
```

## Offline / compute-node workflow

Some HPC clusters (this project targets one — Berzelius) block outbound
internet from compute nodes. `from_pretrained` also accepts a local
directory, so the Zenodo-archive-based workflow keeps working unchanged:

```python
predictor = EnsemblePredictor.from_pretrained(
    "ensembles/dmax_caruana_ensemble/checkpoints",
    weights_file="ensembles/dmax_caruana_ensemble/ensemble_weights_dmax_caruana_ensemble.json",
)
```

Either run `from_pretrained(...)` once from a login node to populate the
local Hugging Face cache (subsequent calls from compute nodes reuse it), or
download `cache.zip`/`ensembles.zip` from
[Zenodo](https://doi.org/10.5281/zenodo.15691822) and set `TACKAI_CACHE` by
hand — see the repository README's "Pre-trained Models & Cache Files"
section.

## Inspecting what loaded

```python
info = predictor.get_model_info()
print(info["available_tasks"])   # e.g. ['dmax']
print(info["n_models"])          # number of fold-models in the ensemble
print(info["weights"])           # {model_name: ensemble_weight}
print(predictor)                 # EnsemblePredictor(n_models=..., tasks=[...], device='cpu', mode='lazy')
```

See the [API reference][tackai.ensemble_predictor.EnsemblePredictor] for the
full method list.
