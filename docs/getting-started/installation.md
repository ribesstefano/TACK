# Installation

`tackai` is not (yet) published on PyPI — install it from a clone of the
repository. The project uses [`uv`](https://docs.astral.sh/uv/) for
environment management, but any `pip`-compatible tool works the same way.

```bash
git clone https://github.com/ribesstefano/TACK.git
cd TACK

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

That's it — `huggingface-hub` is a core dependency, so
[`EnsemblePredictor.from_pretrained`][tackai.ensemble_predictor.EnsemblePredictor.from_pretrained]
works right after install with no extra packages.

!!! note "No cache/model download needed for the Hugging Face path"
    The rest of the project's README describes downloading `cache.zip` and
    `ensembles.zip` from Zenodo and pointing the `TACKAI_CACHE` environment
    variable at them. That manual setup is **only needed for the
    local-directory workflow**. If you load ensembles with
    `from_pretrained("ailab-bio/TACK-ensembles", ...)`, both the model
    checkpoints and the shared cache assets are downloaded automatically the
    first time you call it — see
    [Loading a Pretrained Ensemble](../guide/loading-models.md).

## Requirements

- Python ≥ 3.11 (the project develops against 3.13).
- Outbound network access to `huggingface.co` the first time you load a
  given ensemble — after that, downloads are cached locally (standard 🤗 Hub
  cache, respects `HF_HOME`).

!!! warning "No outbound internet from a compute node?"
    Some HPC clusters (this project targets one — Berzelius) block outbound
    internet from compute nodes. Either run `from_pretrained(...)` once from
    a login node to populate the local Hugging Face cache, or fall back to
    the Zenodo-archive-based local-directory workflow described in the
    [repository README](https://github.com/ribesstefano/TACK#-pre-trained-models--cache-files).

## GPU

Inference works fine on CPU — the models are small and the bottleneck is
feature encoding, not the forward pass. Pass `device="cuda"` to
`from_pretrained` if you do want GPU inference and have a CUDA-enabled
PyTorch installed; see the main README's GPU install instructions.

## Verifying the install

```python
from tackai.ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles", subfolder="dmax_caruana",
)
print(predictor)
# EnsemblePredictor(n_models=..., tasks=['dmax'], device='cpu', mode='lazy')
```

Continue to the [Quickstart](quickstart.md) for a full prediction example.
