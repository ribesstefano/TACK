# TACK

**A statistical evaluation of degradation activity on a novel TArgeting Chimeras Knowledge dataset.**

[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ailab-bio/TACK)
[![Ensembles](https://img.shields.io/badge/🤗%20HuggingFace-Ensembles-yellow)](https://huggingface.co/ailab-bio/TACK-ensembles)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/ribesstefano/TACK/blob/main/LICENSE)

TACK combines PROTAC degradation-activity data from multiple sources (TPDdb,
PROTAC-DB, PROTACpedia) into a curated dataset, published on the Hugging Face
Hub as [`ailab-bio/TACK`](https://huggingface.co/datasets/ailab-bio/TACK), together
with pre-trained ensemble models for predicting **Dmax**, **DC50**, and
**binary degradation activity**.

## Scope of this site

This site documents the **Python API for *running* the pre-trained
models** — i.e. inference, not training — through the parts of `tackai` that
work directly with the Hugging Face Hub:

- [`EnsemblePredictor.from_pretrained`][tackai.ensemble_predictor.EnsemblePredictor.from_pretrained] —
  load a pre-trained ensemble straight from
  [`ailab-bio/TACK-ensembles`](https://huggingface.co/ailab-bio/TACK-ensembles),
  with its shared cache assets pulled automatically from
  [`ailab-bio/TACK-cache`](https://huggingface.co/datasets/ailab-bio/TACK-cache).
- [`EnsemblePredictor.predict`][tackai.ensemble_predictor.EnsemblePredictor.predict] /
  [`predict_dataframe`][tackai.ensemble_predictor.EnsemblePredictor.predict_dataframe] —
  weighted ensemble inference with uncertainty quantification.
- [`SampleInput`][tackai.ensemble_predictor.SampleInput] — the input container for a single sample.
- [`transform_context`][tackai.ensemble_predictor.EnsemblePredictor.transform_context] —
  the fast path for screening many compounds against one fixed biological context.
- Loading the `ailab-bio/TACK` dataset itself via 🤗 `datasets`.

For training new models (Hydra configs, nested cross-validation, Optuna
tuning), data curation, and evaluation/ranking tooling, see the main
[repository README](https://github.com/ribesstefano/TACK) and
[`CLAUDE.md`](https://github.com/ribesstefano/TACK/blob/main/CLAUDE.md) —
those are out of scope here.

## Where to start

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)** — install `tackai`; no
  manual cache/model download needed for the Hugging Face path.
- **[Quickstart](getting-started/quickstart.md)** — load an ensemble and get
  your first prediction in a few lines.
- **[Tutorial Notebook](tutorial/ensemble_predictor_tutorial.ipynb)** — the
  full worked walkthrough, rendered from
  [`notebooks/ensemble_predictor_tutorial.ipynb`](https://github.com/ribesstefano/TACK/blob/main/notebooks/ensemble_predictor_tutorial.ipynb).
- **[API Reference](api/ensemble-predictor.md)** — generated from the
  docstrings in `tackai/ensemble_predictor.py`.

</div>

## Why Hugging Face Hub?

`EnsemblePredictor.from_pretrained` accepts either a local checkpoints
directory or a Hub repo id. Pointed at the Hub, it downloads the model
checkpoints *and* the shared embedding/lookup caches it depends on, and wires
up `TACKAI_CACHE` for you — no Zenodo archive to unzip, no environment
variable to set by hand:

```python
from tackai.ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles", subfolder="dmax_caruana",
)
```

See [Loading a Pretrained Ensemble](guide/loading-models.md) for the full
picture, including the local-directory alternative for offline/compute-node
use.
