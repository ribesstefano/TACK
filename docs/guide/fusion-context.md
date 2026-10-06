# Fusion Context Cache (Experimental)

The fusion surrogates in `tackai.fusion` (notebook-driven, under active development — see
`notebooks/fusion_comparison.ipynb`) read their biological context (cell line, POI, E3
ligase, assay) from a handful of small, PCA-reduced embedding tables cached in
`TACKAI_CACHE`. `FusionData.from_pretrained` downloads them for you instead of requiring the
notebooks that produce them:

```python
from tackai.fusion import FusionData

data = FusionData.from_pretrained()   # ailab-bio/TACK-fusion-context by default
ctx = data.encode_context({
    "poi_seq": "MAGEG...", "e3_seq": "MEPVR...",
    "cell_id": "CVCL_0031", "assay": "western blot", "assay_time": 24.0,
})
```

This installs the tables into `TACKAI_CACHE` (honoring `HF_HOME` for the download itself) and
returns a `FusionData` with no training table — it is ready for `encode`, `encode_context` and
`assemble`, but not for `X`, `groups` or `target()` (those need `FusionData.from_csv` and the
curated CSVs).

Passing a local directory instead of a repo id (one written by
`scripts/publish_fusion_context.py`'s `stage` phase) skips the network entirely:

```python
data = FusionData.from_pretrained("/path/to/staged/fusion_context")
```

If a file already in your cache differs from the one being installed — most often because you
locally refitted a PCA — `from_pretrained` raises rather than silently mixing the two. Pass
`force_download=True` to replace it deliberately.
