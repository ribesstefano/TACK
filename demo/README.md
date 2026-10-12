---
title: TACK Demo
emoji: 🧬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: Work in progress — PROTAC degradation predictions (Dmax, pDC50, activity)
---

# TACK — fast PROTAC degradation predictions

> ⚠️ **Work in progress.** An unreviewed preview of models trained on an unpublished,
> extended curation — *not* the models from the TACK paper. Not for clinical, diagnostic,
> regulatory or safety-critical use. The same disclaimer sits at the top of the app.

Score a PROTAC in a specific experimental context — target protein, E3 ligase, cell line,
assay and treatment time — and get **Dmax**, **pDC50** and **P(active)**, each with a
calibrated uncertainty. CPU only, milliseconds per molecule, no `torch`.

* **Code:** [github.com/ribesstefano/TACK](https://github.com/ribesstefano/TACK)
* **Models:** [`ailab-bio/TACK-Fast-Ensemble`](https://huggingface.co/ailab-bio/TACK-Fast-Ensemble)
* **Context tables:** [`ailab-bio/TACK-Context-Encoding`](https://huggingface.co/datasets/ailab-bio/TACK-Context-Encoding)
* **Tutorial:** [`notebooks/fast_TACK_models_tutorial.ipynb`](https://github.com/ribesstefano/TACK/blob/main/notebooks/fast_TACK_models_tutorial.ipynb)

## Developing this Space

The Space's source of truth is [`demo/`](https://github.com/ribesstefano/TACK/tree/main/demo)
in the TACK repository; `.github/workflows/deploy-demo.yml` mirrors that directory here on
every push to `main` that touches it. Edit the Space directly only to debug — the next
deployment overwrites it.

```bash
uv pip install -e . && uv pip install -r demo/requirements.txt
OMP_NUM_THREADS=1 python demo/app.py            # http://127.0.0.1:7860
OMP_NUM_THREADS=1 python demo/build_vocab.py    # refresh vocab.json after a table release
```

`vocab.json` is what makes the target / E3 / cell-line dropdowns readable: the published
context tables are keyed by full amino-acid sequences and Cellosaurus accessions, and
`build_vocab.py` joins those keys against the curated dataset's gene symbols and cell-line
names, keeping only the pairs that exist on both sides.

The workflow deploys to `ailab-bio/TACK-Demo` by default; set the `HF_SPACE_ID` repository
variable to point it somewhere else, and an `HF_TOKEN` repository secret (a write token) for
it to authenticate with.
