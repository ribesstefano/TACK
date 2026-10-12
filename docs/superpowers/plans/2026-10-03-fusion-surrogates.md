# Fast fusion surrogates (M4 + M7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote M4 (GP with cross-block product kernels) and M7 (XGBoost) from `notebooks/fusion_comparison.ipynb` into a maintained `tackai.fusion` subpackage, running on the pre-reduced context embeddings with no PCA inside the pipeline, with a shared data handler and a generic cached-context ensemble.

**Architecture:** Seven focused modules under `tackai/fusion/`. Block-wise preprocessing fitted inside each fold (`blocks.py`), context embeddings read from five cached npz tables (`context.py`), molecular features computed on the fly bit-exactly with `MolEmbedding` (`features.py`), one `FusionData` class serving both training and inference (`data.py`), a plain-torch exact GP with cached per-block squared distances (`gp.py`), the two estimators behind one `fit`/`predict` interface (`models.py`), and a generic ensemble whose members all consume the same design matrix (`ensemble.py`).

**Tech Stack:** numpy, pandas, scikit-learn, torch (plain, no gpytorch), xgboost, rdkit, sentence-transformers (assay fallback only), joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-fusion-surrogates-design.md`

## Global Constraints

- Python ≥ 3.11; packages already declared in `pyproject.toml`. **Do not add `gpytorch`** — the GP is plain torch.
- macOS: `OMP_NUM_THREADS=1` must be set before importing torch/xgboost in the same process (two libomp copies segfault). `test/conftest.py` sets it; the notebook's first cell sets it.
- **No PCA anywhere in the pipeline.** Context arrives already reduced from the cache; molecule blocks stay raw.
- Molecule blocks get **no scaler**: raw values divided by `sqrt(width)` only (user decision).
- `BLOCK_ORDER = ["fingerprint", "descriptors", "e3", "cell", "poi", "assay", "assay_time"]` — the notebook's order, kept so cached blocks and column indices line up.
- Dims: fingerprint 1024 (Morgan r=16, `includeChirality=True`), descriptors 217 (all `Descriptors._descList`, invalid → `-1`), poi 51, e3 7, cell 47, assay 8, assay_time 1. Total 1355.
- Exact cache filenames (from `protein_pooling_meta.json` / `context_embeddings_meta.json`):
  - `cell_embeddings_model=sentence-transformer_pooling=mean_pca47.npz`
  - `protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18_pooling=lse_window=1022_block=poi_pca51.npz`
  - `protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18_pooling=lse_window=1022_block=combined_pca52.npz`
  - `protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca7.npz`
  - `assay_embeddings_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz`
  - PCA side files: same stem + `_model.npz` for cell/protein; `assay_pca_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz` for assay.
- Tasks: `dmax` (regression, [0,1]), `pdc50` (regression, `9 - log10(DC50 nM)`), `activity` (binary). `DMAX_THR = 0.80`, `PDC50_THR = 6.0`.
- Docstrings: summary line, blank line, `Args:` / `Returns:` for non-trivial public methods (CLAUDE.md).
- Every test runs without the real `TACKAI_CACHE`; real-artifact tests are marked `requires_cache`.

## Review Focus

1. **Unseen POI/E3 sequence at inference** → must raise `KeyError` naming the block, never silently substitute a vector. (Task 3)
2. **Unparseable SMILES in an inference batch** → the batch must not raise; that row gets `ok=False` and a NaN prediction. (Tasks 2, 7)
3. **Constant target in a training fold** (`y.std() == 0`) → no division by zero; predictions equal the constant. (Task 6)
4. **A context column constant in the training fold** (e.g. a single cell line) → `StandardScaler` must not emit `inf`/NaN. (Task 1)
5. **Empty input** (`predict([])`) → returns shape `(0,)`, does not raise. (Task 7)

---

### Task 1: Package skeleton and `BlockPreprocessor`

**Files:**
- Create: `tackai/fusion/__init__.py`, `tackai/fusion/blocks.py`
- Create: `test/conftest.py`, `test/test_fusion_blocks.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BLOCK_ORDER: list[str]`, `DENSE_BLOCKS = ["fingerprint","descriptors","poi","e3","cell","assay"]`, `SMALL_BLOCKS = ["assay_time"]`, `MOL_BLOCKS = ["fingerprint","descriptors"]`, `BLOCK_DIMS: dict[str,int]`
  - `block_index(dims: dict[str,int]) -> dict[str, np.ndarray]` — column indices per block, in `BLOCK_ORDER`.
  - `class BlockPreprocessor(blocks: dict[str,np.ndarray] | None = None, scale_blocks: Sequence[str] | None = None)` with `fit(X, y=None) -> self`, `transform(X) -> dict[str, np.ndarray]`, `transform_blocks(X, only: Sequence[str]) -> dict`, `concat(Z, names=None) -> np.ndarray`, attributes `dims_: dict[str,int]`, `blocks_`.
  - Default `scale_blocks` = dense blocks **minus** `MOL_BLOCKS` (i.e. `poi, e3, cell, assay`).

- [ ] **Step 1: Write `test/conftest.py`** (shared by all later tasks)

```python
"""Shared fixtures: a synthetic TACKAI_CACHE with the real filenames, and small tables."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # must precede torch/xgboost imports

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CELL_FILE = "cell_embeddings_model=sentence-transformer_pooling=mean_pca47.npz"
POI_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
            "_pooling=lse_window=1022_block=poi_pca51.npz")
COMBINED_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=18"
                 "_pooling=lse_window=1022_block=combined_pca52.npz")
E3_FILE = ("protein_embeddings_esm_model=facebook-esm2_t30_150M_UR50D_layer=30"
           "_pooling=mean_rm_pc2_window=1022_block=e3_pca7.npz")
ASSAY_FILE = "assay_embeddings_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz"
ASSAY_PCA_FILE = "assay_pca_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz"
NOT_FOUND = "Unknown cell line."

SEQS = {  # short stand-ins; the real tables are keyed by full sequences
    "poi": ["MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQ", "MPRRAENWDEAEVGAEEAGVEEYGPEEDGGEESGAEE",
            "MHKTASQRLFPGPSYQNIKSIMEDSTILSDWTNSNK", "MSAEVIHQVEEALDTDEKEMLLFLCRDVAIDVVPPN"],
    "e3": ["MEPVRRSSRLSAQKQQQQQQQQAEDEEMEVEDQDSK", "MAAGSIEPVRRSSRLSAQKQQQQQAEDEEMEVEDQ"],
}
CELLS = ["CVCL_0031", "CVCL_0395", "CVCL_0291", "CVCL_0062"]
ASSAYS = ["western blot", "htrf", "hibit", "capillary immunoassay", "elisa", "unknown",
          "nanoluc reporter", "flow cytometry", "in-cell immunoassay", "msd",
          "high-content imaging", "fluorescence imaging", "dot blot", "viability assay",
          "incell hunter", "elisa + western blot", "in-cell immunoassay + western blot",
          "mass spectrometry", "immunoprecipitation", "reporter assay"]
SMILES = [
    "CC(C)(C)c1ccc(cc1)C(=O)NC1CCN(CC1)C(=O)c1ccccc1",
    "O=C(Nc1ccc(cc1)S(=O)(=O)N)C1CCN(CC1)Cc1ccccc1",
    "COc1ccc(cc1)C(=O)N1CCN(CC1)c1ncccn1",
    "CC(=O)Nc1ccc(cc1)C(=O)N1CCC(CC1)Oc1ccccc1",
    "Clc1ccc(cc1)C(=O)NC1CCN(CC1)C(=O)OC(C)(C)C",
    "OC(=O)C1CCN(CC1)C(=O)c1ccc(cc1)N1CCOCC1",
    "CN1CCN(CC1)c1ccc(cc1)NC(=O)c1ccc(Cl)cc1",
    "CCOC(=O)c1ccc(cc1)N1CCN(CC1)C(=O)c1ccccc1",
]


def _write(path, keys, dim, seed):
    rng = np.random.default_rng(seed)
    np.savez(path, **{k: rng.normal(size=dim).astype(np.float32) for k in keys})


def _write_pca_model(path, full_dim, out_dim, seed):
    rng = np.random.default_rng(seed)
    np.savez(path, mean_=rng.normal(size=full_dim).astype(np.float32),
             components_=rng.normal(size=(out_dim, full_dim)).astype(np.float32),
             explained_variance_ratio_=np.full(out_dim, 1.0 / out_dim, dtype=np.float32))


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A TACKAI_CACHE holding the five npz tables under their real names. Returns the Path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    _write(cache / CELL_FILE, CELLS + [NOT_FOUND], 47, 1)
    _write(cache / POI_FILE, SEQS["poi"], 51, 2)
    _write(cache / COMBINED_FILE, SEQS["poi"] + SEQS["e3"], 52, 3)
    _write(cache / E3_FILE, SEQS["e3"], 7, 4)
    _write(cache / ASSAY_FILE, ASSAYS, 8, 5)
    _write_pca_model(cache / CELL_FILE.replace(".npz", "_model.npz"), 768, 47, 6)
    _write_pca_model(cache / POI_FILE.replace(".npz", "_model.npz"), 640, 51, 7)
    _write_pca_model(cache / E3_FILE.replace(".npz", "_model.npz"), 640, 7, 8)
    _write_pca_model(cache / ASSAY_PCA_FILE, 768, 8, 9)
    monkeypatch.setenv("TACKAI_CACHE", str(cache))
    return cache


@pytest.fixture
def tiny_records():
    """Eight inference-shaped records covering repeated contexts and molecules."""
    return [
        {"smiles": SMILES[i], "poi_seq": SEQS["poi"][i % 4], "e3_seq": SEQS["e3"][i % 2],
         "cell_id": CELLS[i % 4], "assay": ASSAYS[i % 6], "assay_time": 24.0 + i}
        for i in range(8)
    ]


@pytest.fixture
def tiny_csv(tmp_path):
    """A development-shaped CSV with the columns build_dev_table reads."""
    rows = []
    for i in range(24):
        rows.append({
            "SMILES": SMILES[i % len(SMILES)],
            "Recruiter": ["CRBN", "VHL"][i % 2],
            "Recruiter_Sequence": SEQS["e3"][i % 2],
            "Degradation_Target_Uniprot": f"P{i % 4:05d}",
            "Degradation_Target_Sequence": SEQS["poi"][i % 4],
            "Cell_Line_ID": CELLS[i % 4],
            "Cell_Line": "HeLa",
            "Assay": ASSAYS[i % 6],
            "DC50": [10.0, 100.0, 1000.0, np.nan][i % 4],
            "DC50_units": "nM",
            "DC50_h": 24.0,
            "Dmax_h": 24.0,
            "Dmax": [95.0, 60.0, np.nan, 80.0][i % 4],
        })
    path = tmp_path / "dev.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
```

Also add the marker to `pyproject.toml` `[tool.pytest.ini_options]`: `markers = ["requires_cache: needs a populated TACKAI_CACHE"]`.

- [ ] **Step 2: Write the failing test** `test/test_fusion_blocks.py`

```python
import numpy as np
import pytest

from tackai.fusion.blocks import (BLOCK_DIMS, BLOCK_ORDER, DENSE_BLOCKS, MOL_BLOCKS,
                                  SMALL_BLOCKS, BlockPreprocessor, block_index)


def make_X(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())), dtype=np.float64)
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(0, 1, (n, 217))
    X[:, idx["descriptors"][0]] = rng.normal(0, 1, n) * 1e18   # Ipc-scale column
    for b in ("poi", "e3", "cell", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([12.0, 24.0, np.nan], (n, 1))
    return X, idx


def test_block_index_is_contiguous_and_in_block_order():
    idx = block_index(BLOCK_DIMS)
    assert list(idx) == BLOCK_ORDER
    flat = np.concatenate([idx[b] for b in BLOCK_ORDER])
    assert np.array_equal(flat, np.arange(sum(BLOCK_DIMS.values())))
    assert sum(BLOCK_DIMS.values()) == 1355


def test_molecule_blocks_are_not_scaled_only_divided_by_sqrt_width():
    X, idx = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.allclose(Z["fingerprint"], X[:, idx["fingerprint"]] / np.sqrt(1024))
    assert np.allclose(Z["descriptors"], X[:, idx["descriptors"]] / np.sqrt(217))


def test_no_pca_dims_are_preserved():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    assert pre.dims_ == BLOCK_DIMS


def test_context_blocks_are_standardised():
    X, _ = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.allclose(Z["cell"].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Z["cell"].std(axis=0) * np.sqrt(47), 1, atol=1e-6)


def test_constant_context_column_does_not_produce_inf(): # Review Focus 4
    X, idx = make_X()
    X[:, idx["cell"][3]] = 7.0            # one cell line only -> zero variance
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["cell"]).all()
    assert np.allclose(Z["cell"][:, 3], 0.0)


def test_fitted_on_training_rows_only():
    """Statistics come from the fit rows only: changing the held-out rows cannot move them."""
    X, _ = make_X(n=60)
    tr, te = np.arange(30), np.arange(30, 60)
    pre = BlockPreprocessor().fit(X[tr])
    baseline = pre.transform(X[te])["cell"]

    X_shifted = X.copy()
    X_shifted[te] += 100.0
    pre_same_train = BlockPreprocessor().fit(X_shifted[tr])
    assert np.allclose(pre_same_train.transform(X[te])["cell"], baseline, rtol=1e-10, atol=1e-12)

    # and the transform of the training rows is unaffected by which rows it is applied with
    assert np.allclose(pre.transform(X)["cell"][:30], pre.transform(X[tr])["cell"])


def test_assay_time_is_median_imputed_and_standardised():
    X, idx = make_X()
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["assay_time"]).all()


def test_nan_in_dense_block_is_mean_imputed():
    X, idx = make_X()
    X[0, idx["poi"][0]] = np.nan
    Z = BlockPreprocessor().fit(X).transform(X)
    assert np.isfinite(Z["poi"]).all()


def test_transform_blocks_subset_matches_full_transform():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    full = pre.transform(X)
    part = pre.transform_blocks(X, only=["cell", "poi"])
    assert set(part) == {"cell", "poi"}
    for b in part:
        assert np.allclose(part[b], full[b])


def test_concat_follows_block_order():
    X, _ = make_X()
    pre = BlockPreprocessor().fit(X)
    A = pre.concat(pre.transform(X))
    assert A.shape == (40, 1355)
```

- [ ] **Step 3: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_blocks.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion'`

- [ ] **Step 4: Implement `tackai/fusion/blocks.py`**

Port `BlockPreprocessor` from the notebook's `preproc` cell with PCA removed:

- `BLOCK_DIMS = {"fingerprint": 1024, "descriptors": 217, "e3": 7, "cell": 47, "poi": 51, "assay": 8, "assay_time": 1}`.
- `block_index(dims)` walks `BLOCK_ORDER` accumulating `np.arange`.
- `fit`: per block — dense blocks get `SimpleImputer(strategy="mean", keep_empty_features=True)`; blocks in `scale_blocks` additionally get `StandardScaler`; `width = sqrt(n_cols)` for every dense block. `assay_time` gets median imputation + `StandardScaler`, `width = 1`. Guard zero variance: after fitting the scaler, replace `scale_ == 0` with `1.0` so constant columns map to `0.0` instead of `inf`.
- `transform`: impute → (scale) → `/ width`; returns `dict[str, np.ndarray]`.
- `transform_blocks(X, only)`: same, restricted; `X` may be the full-width matrix.
- `concat(Z, names=None)`: `np.concatenate` in `BLOCK_ORDER`.
- `tackai/fusion/__init__.py`: export the names added so far; extend it in later tasks.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest test/test_fusion_blocks.py -q`
Expected: PASS, 10 passed

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/__init__.py tackai/fusion/blocks.py test/conftest.py test/test_fusion_blocks.py pyproject.toml
git commit -m "feat(fusion): block layout and fold-internal BlockPreprocessor without PCA"
```

---

### Task 2: `MolFeaturizer` — on-the-fly molecular features

**Files:**
- Create: `tackai/fusion/features.py`, `test/test_fusion_features.py`

**Interfaces:**
- Consumes: `BLOCK_DIMS` from `tackai.fusion.blocks`.
- Produces: `class MolFeaturizer(radius=16, fp_size=1024, share_ipc=True, use_cache=True, n_workers=1)` with
  `featurize(smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning
  `(fp (n,1024) float32, desc (n,217) float32, ok (n,) bool)`, `stats: dict`, `close()`,
  and `DESCRIPTOR_NAMES: list[str]` at module level.

- [ ] **Step 1: Write the failing test** `test/test_fusion_features.py`

```python
import numpy as np
import pytest

from tackai.fusion.features import DESCRIPTOR_NAMES, MolFeaturizer
from test.conftest import SMILES


def test_shapes_and_dtypes():
    fp, desc, ok = MolFeaturizer().featurize(SMILES)
    assert fp.shape == (len(SMILES), 1024) and fp.dtype == np.float32
    assert desc.shape == (len(SMILES), 217) and desc.dtype == np.float32
    assert ok.all() and len(DESCRIPTOR_NAMES) == 217


def test_fingerprint_is_binary():
    fp, _, _ = MolFeaturizer().featurize(SMILES)
    assert set(np.unique(fp)) <= {0.0, 1.0}


def test_invalid_smiles_does_not_raise_and_is_flagged():  # Review Focus 2
    fp, desc, ok = MolFeaturizer().featurize(["not_a_molecule", SMILES[0], ""])
    assert list(ok) == [False, True, False]
    assert np.all(fp[0] == 0) and np.all(desc[0] == 0)


def test_empty_input_returns_empty_arrays():  # Review Focus 5
    fp, desc, ok = MolFeaturizer().featurize([])
    assert fp.shape == (0, 1024) and desc.shape == (0, 217) and ok.shape == (0,)


def test_duplicates_are_computed_once_and_cached_across_calls():
    f = MolFeaturizer()
    f.featurize([SMILES[0], SMILES[0], SMILES[1]])
    assert f.stats["computed"] == 2
    f.featurize([SMILES[0], SMILES[1]])
    assert f.stats["computed"] == 2 and f.stats["cache_hits"] >= 2


def test_rows_are_in_input_order_with_repeats():
    fp, _, _ = MolFeaturizer().featurize([SMILES[1], SMILES[0], SMILES[1]])
    assert np.array_equal(fp[0], fp[2]) and not np.array_equal(fp[0], fp[1])


def test_descriptor_sentinel_replaces_non_finite_and_huge_values():
    _, desc, _ = MolFeaturizer().featurize(SMILES)
    assert np.isfinite(desc).all()
    assert desc.max() <= 1e20


def test_bit_exact_with_tackai_mol_embedding(tmp_path):
    """The whole point: these features must equal what the training cache holds."""
    from tackai.data.embeddings.mol_embeddings import MolEmbedding
    fp_ref = MolEmbedding(embeddings_type="fingerprint", radius=16, fp_size=1024,
                          cache_dir=str(tmp_path)).transform(SMILES[:3])
    de_ref = MolEmbedding(embeddings_type="rdkit_descriptors",
                          cache_dir=str(tmp_path)).transform(SMILES[:3])
    fp, desc, ok = MolFeaturizer().featurize(SMILES[:3])
    assert ok.all()
    for i, s in enumerate(SMILES[:3]):
        assert np.array_equal(fp[i], fp_ref[s])
        assert np.allclose(desc[i], de_ref[s], rtol=0, atol=0)


def test_process_pool_gives_the_same_values():
    a, b, _ = MolFeaturizer(n_workers=1).featurize(SMILES)
    f = MolFeaturizer(n_workers=2)
    c, d, _ = f.featurize(SMILES)
    f.close()
    assert np.array_equal(a, c) and np.allclose(b, d)
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_features.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.features'`

- [ ] **Step 3: Implement `tackai/fusion/features.py`**

Port `MolFeaturizer` and `_featurize_chunk` from `notebooks/ensemble_speed_rl.ipynb` (tag `featurizer`), which is already asserted bit-exact there. Keep:
- `rdFingerprintGenerator.GetMorganGenerator(radius=16, fpSize=1024, includeChirality=True)`;
- all `Descriptors._descList`, with `Ipc`/`AvgIpc` sharing one characteristic polynomial via `Graphs.CharacteristicPolynomial` + `entropy.InfoEntropy`, guarded by `mol._adjMat` caching;
- sentinel: `val is None or nan or inf or |val| > 1e20` → `-1`;
- dedup via `dict.fromkeys`, memo dict, interleaved chunks for the loky pool, `stats` counters,
- invalid SMILES → zeros + `ok=False` (do not raise).
`DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_features.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/features.py test/test_fusion_features.py
git commit -m "feat(fusion): on-the-fly MolFeaturizer, bit-exact with MolEmbedding"
```

---

### Task 3: `ContextEncoder` — the five cached npz tables

**Files:**
- Create: `tackai/fusion/context.py`, `test/test_fusion_context.py`

**Interfaces:**
- Consumes: `tackai.data.utils.get_cache_dir`.
- Produces:
  - `normalize_assay(value) -> str` (ported, open vocabulary)
  - `CONTEXT_FILES: dict[str, dict[str, str]]` — block → `{"table":, "pca_model":}` filenames; blocks `poi`, `e3`, `cell`, `assay`, `combined`.
  - `class ContextEncoder(cache_dir=None, protein_space="per_block")` with
    `encode(block, keys) -> np.ndarray`, `dim(block) -> int`,
    `register_sequence(block, sequence, embedding) -> None`,
    `encode_context(record) -> dict[str, np.ndarray]`, `NOT_FOUND = "Unknown cell line."`.
  - `protein_space="combined"` makes `encode("poi", ...)` and `encode("e3", ...)` read the combined table.

- [ ] **Step 1: Write the failing test** `test/test_fusion_context.py`

```python
import numpy as np
import pytest

from tackai.fusion.context import ContextEncoder, normalize_assay
from test.conftest import ASSAYS, CELLS, SEQS


def test_normalize_assay_ported_rules():
    assert normalize_assay("Western Blot") == "western blot"
    assert normalize_assay("WESTERN BLOT / ELISA") == "elisa + western blot"
    assert normalize_assay("Simple Western") == "capillary immunoassay"
    assert normalize_assay(None) == "unknown"
    assert normalize_assay("") == "unknown"
    assert normalize_assay(float("nan")) == "unknown"
    assert normalize_assay("HiBiT assay") == "hibit"
    assert normalize_assay("some novel readout") == "some novel readout"


def test_dims_match_the_cached_tables(fake_cache):
    enc = ContextEncoder()
    assert (enc.dim("poi"), enc.dim("e3"), enc.dim("cell"), enc.dim("assay")) == (51, 7, 47, 8)


def test_sequences_are_looked_up_by_stripped_string(fake_cache):
    enc = ContextEncoder()
    a = enc.encode("poi", [SEQS["poi"][0]])
    b = enc.encode("poi", ["  " + SEQS["poi"][0] + "\n"])
    assert a.shape == (1, 51) and np.array_equal(a, b)


def test_unseen_sequence_raises_keyerror_naming_the_block(fake_cache):  # Review Focus 1
    enc = ContextEncoder()
    with pytest.raises(KeyError, match="poi"):
        enc.encode("poi", ["MKKKWWWNOTINTABLE"])
    with pytest.raises(KeyError, match="e3"):
        enc.encode("e3", ["MKKKWWWNOTINTABLE"])


def test_missing_cell_line_gets_the_not_found_vector(fake_cache):
    enc = ContextEncoder()
    expected = enc.encode("cell", [ContextEncoder.NOT_FOUND])
    for missing in (None, "", "   ", np.nan):
        assert np.array_equal(enc.encode("cell", [missing]), expected)


def test_unknown_cell_accession_raises(fake_cache):
    with pytest.raises(KeyError, match="cell"):
        ContextEncoder().encode("cell", ["CVCL_9999"])


def test_known_assay_comes_from_the_table(fake_cache):
    enc = ContextEncoder()
    assert enc.encode("assay", ["Western blot"]).shape == (1, 8)
    assert np.array_equal(enc.encode("assay", ["Western blot"]), enc.encode("assay", ["western blot"]))


def test_unseen_assay_is_embedded_on_the_fly_and_projected(fake_cache, monkeypatch):
    """A canonical type absent from the table uses the sentence-transformer + cached PCA."""
    calls = []

    def fake_encode_texts(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 768), dtype=np.float32)

    enc = ContextEncoder()
    monkeypatch.setattr(enc, "_encode_texts", fake_encode_texts)
    out = enc.encode("assay", ["a brand new readout"])
    assert out.shape == (1, 8) and np.isfinite(out).all()
    assert calls == [["a brand new readout"]]


def test_register_sequence_projects_a_full_embedding(fake_cache):
    enc = ContextEncoder()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=640).astype(np.float32)
    enc.register_sequence("poi", "MNEWSEQ", emb)
    out = enc.encode("poi", ["MNEWSEQ"])
    side = np.load(enc.cache_dir / enc.files("poi")["pca_model"])
    assert np.allclose(out[0], (emb - side["mean_"]) @ side["components_"].T, rtol=1e-5, atol=1e-5)


def test_combined_space_is_selectable(fake_cache):
    enc = ContextEncoder(protein_space="combined")
    assert enc.dim("poi") == 52 and enc.dim("e3") == 52
    assert enc.encode("e3", [SEQS["e3"][0]]).shape == (1, 52)


def test_encode_context_returns_every_context_block(fake_cache):
    enc = ContextEncoder()
    row = enc.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                              "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0})
    assert set(row) == {"poi", "e3", "cell", "assay", "assay_time"}
    assert row["poi"].shape == (1, 51) and row["assay_time"].shape == (1, 1)


def test_missing_assay_time_is_nan_not_an_error(fake_cache):
    row = ContextEncoder().encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                                           "cell_id": CELLS[0], "assay": "htrf"})
    assert np.isnan(row["assay_time"]).all()


def test_tables_are_loaded_once(fake_cache):
    enc = ContextEncoder()
    enc.encode("poi", [SEQS["poi"][0]])
    before = enc._tables["poi"]
    enc.encode("poi", [SEQS["poi"][1]])
    assert enc._tables["poi"] is before
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_context.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.context'`

- [ ] **Step 3: Implement `tackai/fusion/context.py`**

- Port `normalize_assay` verbatim from `notebooks/assay_embeddings.ipynb` (cell 7), replacing `pd.isna(value)` with a guard that also treats `None` and blank strings as missing.
- `CONTEXT_FILES` holds the exact filenames from Global Constraints.
- Lazy `_tables: dict[str, dict[str, np.ndarray]]` loaded with `dict(np.load(path))`; `files(block)` resolves `protein_space`.
- `encode(block, keys)`: strip string keys; `cell` missing → `NOT_FOUND`; `assay` → `normalize_assay` then table, with misses batched through `_encode_texts` + the cached PCA (`mean_`, `components_`); `poi`/`e3` missing or unseen → `KeyError(f"{block}: sequence not in the cached table: {seq[:30]}…")`.
- `_encode_texts(texts)`: lazily construct `sentence_transformers.SentenceTransformer("sentence-transformers/all-mpnet-base-v1")` and `encode`. Only reached for unseen assays, so the import stays lazy.
- `register_sequence(block, seq, emb)`: project with the block's `_model.npz` and insert into the loaded table.
- `encode_context(record)`: read keys `poi_seq`, `e3_seq`, `cell_id`, `assay`, `assay_time` (accept `cell_key` as an alias for `cell_id`); `assay_time` missing → `np.nan`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_context.py -q`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/context.py test/test_fusion_context.py
git commit -m "feat(fusion): ContextEncoder over the cached PCA-reduced context embeddings"
```

---

### Task 4: `FusionData` — the data handler

**Files:**
- Create: `tackai/fusion/data.py`, `test/test_fusion_data.py`

**Interfaces:**
- Consumes: `BLOCK_ORDER`, `BLOCK_DIMS`, `block_index` (Task 1); `MolFeaturizer` (Task 2); `ContextEncoder` (Task 3).
- Produces:
  - `make_targets(dmax_pct, dc50_nM) -> dict[str, np.ndarray]` with keys `dmax`, `pdc50`, `activity`; `DMAX_THR=0.80`, `PDC50_THR=6.0`, `TASKS=("dmax","pdc50","activity")`, `TASK_TYPES={"dmax":"regression","pdc50":"regression","activity":"binary"}`.
  - `scaffold_groups(smiles) -> (np.ndarray, dict)`
  - `class FusionData` with classmethod `from_csv(files, *, encoder=None, featurizer=None, cache=True) -> FusionData`;
    properties `X`, `groups`, `smiles`, `table`, `blocks`;
    methods `target(task)`, `task_rows(task) -> (idx, X, y, groups)`, `splits(task, n_repeats=5, n_folds=5, seeds=None)`,
    `encode(records) -> np.ndarray`, `encode_context(record) -> np.ndarray` (1×n_context_cols),
    `assemble(context_row, smiles) -> np.ndarray`, `context_columns -> np.ndarray`, `mol_columns -> np.ndarray`.

- [ ] **Step 1: Write the failing test** `test/test_fusion_data.py`

```python
import numpy as np
import pytest

from tackai.fusion.blocks import BLOCK_DIMS, block_index
from tackai.fusion.data import (DMAX_THR, PDC50_THR, FusionData, make_targets,
                                scaffold_groups)
from test.conftest import CELLS, SEQS, SMILES


def test_make_targets_scales_dmax_and_logs_dc50():
    t = make_targets([95.0, 60.0, np.nan], [10.0, 1000.0, 100.0])
    assert np.allclose(t["dmax"][:2], [0.95, 0.60])
    assert np.isnan(t["dmax"][2])
    assert np.allclose(t["pdc50"], [8.0, 6.0, 7.0])


def test_make_targets_non_positive_dc50_is_undefined():
    t = make_targets([50.0, 50.0], [0.0, -5.0])
    assert np.isnan(t["pdc50"]).all()


def test_activity_needs_both_or_a_decisive_one():
    t = make_targets([95.0, 95.0, 60.0, np.nan, np.nan],
                     [10.0, 10000.0, np.nan, 10.0, 10000.0])
    assert t["activity"][0] == 1.0        # dmax > .8 and pdc50 > 6
    assert t["activity"][1] == 0.0        # pdc50 = 5 -> inactive
    assert t["activity"][2] == 0.0        # dmax .6 < .8 alone is decisive
    assert np.isnan(t["activity"][3])     # pdc50 = 8 alone cannot decide active
    assert t["activity"][4] == 0.0        # pdc50 = 5 alone is decisive


def test_scaffold_groups_share_a_group_for_one_scaffold():
    g, info = scaffold_groups([SMILES[0], SMILES[0], SMILES[1]])
    assert g[0] == g[1]
    assert info["n_failed_rows"] == 0


def test_acyclic_molecule_gets_its_own_group():
    g, info = scaffold_groups(["CCCC", "CCCCC", SMILES[0]])
    assert g[0] != g[1] and info["n_failed_rows"] == 2


def test_from_csv_builds_the_design_matrix(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.X.shape == (24, sum(BLOCK_DIMS.values()))
    assert data.X.dtype == np.float32
    assert len(data.groups) == 24 and len(data.smiles) == 24


def test_blocks_land_in_their_columns(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    idx = block_index(BLOCK_DIMS)
    fp = data.X[:, idx["fingerprint"]]
    assert set(np.unique(fp)) <= {0.0, 1.0}
    assert np.isfinite(data.X[:, idx["cell"]]).all()


def test_task_rows_drops_undefined_targets(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    idx, X, y, g = data.task_rows("dmax")
    assert np.isfinite(y).all() and len(idx) == len(y) == len(X) == len(g)
    assert len(y) < 24     # the fixture has NaN Dmax rows


def test_splits_never_split_a_scaffold_group(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    _, _, y, g = data.task_rows("pdc50")
    splits = data.splits("pdc50", n_repeats=2, n_folds=3)
    assert len(splits) == 2 and len(splits[0]) == 3
    for repeat in splits:
        covered = np.concatenate([te for _, te in repeat])
        assert sorted(covered) == list(range(len(y)))
        for tr, te in repeat:
            assert not set(g[tr]) & set(g[te])
            assert not set(tr) & set(te)


def test_encode_reproduces_the_training_row_for_identical_inputs(fake_cache, tiny_csv):
    """The inference path must agree with the training path, with mol features on the fly."""
    data = FusionData.from_csv([tiny_csv], cache=False)
    row = data.table.iloc[0]
    X_inf = data.encode([{"smiles": row["smiles"], "poi_seq": row["poi_seq"],
                          "e3_seq": row["e3_seq"], "cell_id": row["cell_key"],
                          "assay": row["assay_raw"], "assay_time": row["assay_time"]}])
    assert np.allclose(X_inf[0], data.X[0], rtol=0, atol=0, equal_nan=True)


def test_assemble_broadcasts_one_context_over_many_smiles(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    ctx = data.encode_context({"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                               "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0})
    X = data.assemble(ctx, SMILES[:4])
    assert X.shape == (4, sum(BLOCK_DIMS.values()))
    assert np.array_equal(X[:, data.context_columns], np.repeat(ctx, 4, axis=0))
    direct = data.encode([{"smiles": s, "poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0],
                           "cell_id": CELLS[0], "assay": "western blot", "assay_time": 24.0}
                          for s in SMILES[:4]])
    assert np.allclose(X, direct, equal_nan=True)


def test_encode_accepts_a_dataframe(fake_cache, tiny_csv, tiny_records):
    import pandas as pd
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert np.allclose(data.encode(pd.DataFrame(tiny_records)), data.encode(tiny_records),
                       equal_nan=True)


def test_encode_empty_returns_empty_matrix(fake_cache, tiny_csv):  # Review Focus 5
    data = FusionData.from_csv([tiny_csv], cache=False)
    assert data.encode([]).shape == (0, sum(BLOCK_DIMS.values()))


def test_context_blocks_are_cached_between_constructions(fake_cache, tiny_csv):
    first = FusionData.from_csv([tiny_csv], cache=True)
    second = FusionData.from_csv([tiny_csv], cache=True)
    assert np.allclose(first.X, second.X, equal_nan=True)
    assert (fake_cache / "fusion_blocks").is_dir()
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_data.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.data'`

- [ ] **Step 3: Implement `tackai/fusion/data.py`**

- `make_targets`, `scaffold_groups`, `_strat_labels`, `make_splits`: port verbatim from the notebook (tags `targets`, `scaffolds`, `splits`), with module constants replacing notebook globals (`N_STRAT_BINS = 5`, `REPEAT_SEEDS = [1000 + r for r in range(5)]`).
- `build_table(files)`: port `build_dev_table` (tag `dev_table`) — harmonise both CSV layouts, `_first_valid` for cell keys, `assay_time = mean(DC50_h, Dmax_h)`, drop rows without SMILES, fill a missing POI sequence from the most frequent sequence of the same UniProt. Column names in the returned frame: `smiles`, `e3_raw`, `e3_seq`, `poi_uniprot`, `poi_seq`, `cell_key`, `assay_raw`, `assay_time`, `dmax_pct`, `dc50_nM`, `source`. Read `Recruiter_Sequence` for `e3_seq` (the E3 block is now a sequence embedding, not a one-hot of `Recruiter`).
- `from_csv`: build the table, compute the content hash (`sha1` over the context columns plus vocabularies/model names/featuriser settings — never a label), look for `TACKAI_CACHE/fusion_blocks/<hash>/{block}.npy`, else compute: `MolFeaturizer.featurize` for the molecule blocks and `ContextEncoder.encode` per context block, then save.
- `X` is the `np.float32` concatenation in `BLOCK_ORDER`; `context_columns` / `mol_columns` are the column indices of the context / molecule blocks.
- `encode(records)`: accept a list of dicts or a DataFrame; featurise SMILES on the fly; encode context per row; assemble in `BLOCK_ORDER`.
- `encode_context(record)` → `(1, len(context_columns))`; `assemble(ctx, smiles)` featurises the SMILES, tiles `ctx` and writes both into one matrix.
- `splits(task, ...)` caches per task in memory keyed by `(task, n_repeats, n_folds, seeds)`; no pickle files (that was notebook bookkeeping).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_data.py -q`
Expected: PASS, 14 passed

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/data.py test/test_fusion_data.py
git commit -m "feat(fusion): FusionData handler for training and on-the-fly inference"
```

---

### Task 5: `AdditiveProductGP` — plain-torch exact GP with cached distances

**Files:**
- Create: `tackai/fusion/gp.py`, `test/test_fusion_gp.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on a `dict[str, np.ndarray]` of processed blocks).
- Produces: `class AdditiveProductGP(dims: dict[str,int], *, interactions=("mol*poi","mol*cell","poi*cell"), ard_blocks=("descriptors",), rbf_blocks=None, linear_blocks=("assay_time",), jitter=1e-6, dtype="float64")` with
  `fit(Z, y, *, n_restarts=3, n_iter=60, lr=0.1, seed=0, max_hyper_points=1200, state=None) -> dict` (returns the fitted hyper-parameter state),
  `load_state(state, Z, y) -> None`, `predict(Z, return_std=False) -> np.ndarray | tuple`,
  `kernel_report() -> dict[str, float]`.
  `MOL_KERNEL_BLOCKS = ("fingerprint", "descriptors")`.

- [ ] **Step 1: Write the failing test** `test/test_fusion_gp.py`

```python
import numpy as np
import pytest

from tackai.fusion.gp import AdditiveProductGP


def toy(n=60, seed=0):
    """Two blocks with a genuine product interaction, plus a linear small block."""
    rng = np.random.default_rng(seed)
    Z = {"fingerprint": rng.normal(size=(n, 4)), "descriptors": rng.normal(size=(n, 3)),
         "poi": rng.normal(size=(n, 2)), "cell": rng.normal(size=(n, 2)),
         "assay_time": rng.normal(size=(n, 1))}
    y = (np.sin(Z["fingerprint"][:, 0]) + Z["poi"][:, 0] * Z["fingerprint"][:, 1]
         + 0.5 * Z["assay_time"][:, 0] + 0.01 * rng.normal(size=n))
    return Z, y


def dims_of(Z):
    return {b: a.shape[1] for b, a in Z.items()}


def test_fit_then_predict_recovers_training_signal():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=40, seed=0)
    pred = gp.predict(Z)
    assert pred.shape == (60,)
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_predict_returns_std_and_it_is_smaller_on_training_points():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=40, seed=0)
    _, std_train = gp.predict(Z, return_std=True)
    far = {b: a + 50.0 for b, a in Z.items()}
    _, std_far = gp.predict(far, return_std=True)
    assert (std_train > 0).all()
    assert std_far.mean() > std_train.mean() * 2


def test_variance_matches_the_textbook_formula():
    """var = k** - ks' (K+sI)^-1 ks, computed independently from the fitted kernel."""
    Z, y = toy(n=40)
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    _, std = gp.predict(Z, return_std=True)
    K = gp.kernel_matrix(Z, Z)                     # noise-free covariance
    Kn = K + np.eye(len(y)) * gp.noise_
    naive = np.sqrt(np.clip(np.diag(K) - np.einsum("ij,jk,ki->i", K, np.linalg.inv(Kn), K), 0, None))
    assert np.allclose(std, naive, rtol=1e-5, atol=1e-7)


def test_cached_distances_equal_a_naive_recomputation():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    state = gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    cached = gp.predict(Z)
    fresh = AdditiveProductGP(dims_of(Z))
    fresh.load_state(state, Z, y)
    assert np.allclose(cached, fresh.predict(Z), rtol=1e-10, atol=1e-12)


def test_ard_block_learns_one_lengthscale_per_column():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z), ard_blocks=("descriptors",))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    report = gp.kernel_report()
    assert np.shape(report["lengthscale"]["descriptors"]) == (3,)
    assert np.shape(report["lengthscale"]["fingerprint"]) == ()


def test_ard_absorbs_a_wildly_scaled_column():
    """A column 1e18 larger must not swamp the kernel when ARD is on."""
    Z, y = toy()
    Z["descriptors"][:, 0] *= 1e18
    gp = AdditiveProductGP(dims_of(Z), ard_blocks=("descriptors",))
    gp.fit(Z, y, n_restarts=1, n_iter=60, seed=0)
    assert np.corrcoef(gp.predict(Z), y)[0, 1] > 0.85


def test_product_kernels_can_be_disabled():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z), interactions=())
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    assert "prod:mol*poi" not in gp.kernel_report()["weight"]


def test_interactions_are_configurable_including_mol_x_e3():
    Z, y = toy()
    Z["e3"] = Z["cell"].copy()
    gp = AdditiveProductGP(dims_of(Z), interactions=("mol*e3",))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    assert "prod:mol*e3" in gp.kernel_report()["weight"]


def test_fit_is_reproducible_for_a_seed():
    Z, y = toy()
    a = AdditiveProductGP(dims_of(Z)); a.fit(Z, y, n_restarts=2, n_iter=20, seed=7)
    b = AdditiveProductGP(dims_of(Z)); b.fit(Z, y, n_restarts=2, n_iter=20, seed=7)
    assert np.allclose(a.predict(Z), b.predict(Z), rtol=1e-10, atol=1e-12)


def test_hyper_fit_subsamples_but_conditions_on_all_rows():
    Z, y = toy(n=80)
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0, max_hyper_points=30)
    assert gp.n_train_ == 80 and gp.n_hyper_ == 30


def test_cholesky_recovers_from_an_ill_conditioned_kernel():
    Z, y = toy(n=30)
    Z["fingerprint"][5] = Z["fingerprint"][4]      # duplicate row
    Z["descriptors"][5] = Z["descriptors"][4]
    Z["poi"][5] = Z["poi"][4]; Z["cell"][5] = Z["cell"][4]
    Z["assay_time"][5] = Z["assay_time"][4]
    gp = AdditiveProductGP(dims_of(Z), jitter=1e-10)
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert np.isfinite(gp.predict(Z)).all()
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_gp.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.gp'`

- [ ] **Step 3: Implement `tackai/fusion/gp.py`**

Exact GP, `float64`, all in plain torch:

- **Parameters** (all unconstrained, `exp`-transformed on use): `log_ls[b]` per RBF block — scalar for isotropic blocks, vector of length `dims[b]` for ARD blocks; `log_scale[term]` for each additive term (`rbf:<b>`, `linear:<b>`, `prod:<a>*<b>`); `log_noise`; `mean_const`.
- **Distance caching.** For every isotropic RBF block, precompute `D2[b] = |a|² + |b|² − 2ab'` once per `(train, train)` and once per `predict` call, and reuse it across restarts and Adam steps — only the lengthscale changes. For ARD blocks, scale first (`X / exp(log_ls[b])`) and recompute `D2` each step (one matmul; 217 columns is milliseconds). This is the whole performance point of the module: comment it as such.
- **Kernel assembly.** `K_b = exp(-D2_b / 2)` after scaling; `mol = K_fingerprint + K_descriptors`; a product term `a*b` is `exp(log_scale) * K_a * K_b` where `K_mol` is that sum; `linear:assay_time` is `exp(log_scale) * (Xa @ Xb.T)`. `kernel_matrix(Za, Zb)` assembles the noise-free covariance from the current parameters and is used by the variance test.
- **Fit.** Subsample `max_hyper_points` rows (sorted `rng.choice`) for the marginal-likelihood optimisation; `n_restarts` restarts seeded `seed*1000+r`, initialising `log_ls ~ N(0, 0.4)`, `log_scale ~ N(0,0.4) − log(n_terms)`, `log_noise = log(0.5)`; Adam(`lr`) for `n_iter` steps on the exact negative log marginal likelihood via `torch.linalg.cholesky`; keep the restart with the lowest final loss. Raise `RuntimeError("GP marginal-likelihood optimisation diverged in every restart")` if all restarts are non-finite.
- **Conditioning.** After the hyper-fit, build `K` on **all** training rows, Cholesky with escalating jitter (`jitter`, ×10 up to 1e-2; raise if it still fails), store `alpha = K⁻¹(y − mean)` and the factor. `n_train_`, `n_hyper_`, `noise_` are attributes.
- **Predict.** `mean = mean_const + Ks @ alpha`; with `return_std`, `var = diag(K**) − ‖L⁻¹Ks‖²` clipped at 0, `std = sqrt(var)`. Noise-free (epistemic) variance, matching the test.
- **`load_state(state, Z, y)`** re-conditions on given data without re-fitting hyper-parameters — this is how ensemble members reuse a tuned kernel and how the cached-distance test compares paths.
- **`kernel_report()`** returns `{"lengthscale": {...}, "weight": {term: share}, "noise": float}`, shares computed as mean diagonal contribution over at most 400 rows (as the notebook's `_report` did).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_gp.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/gp.py test/test_fusion_gp.py
git commit -m "feat(fusion): plain-torch exact GP with cached block distances and exact variance"
```

---

### Task 6: `GPInteraction` (M4) and `XGBoostFusion` (M7)

**Files:**
- Create: `tackai/fusion/models.py`, `test/test_fusion_models.py`

**Interfaces:**
- Consumes: `BlockPreprocessor`, `block_index`, `BLOCK_DIMS` (Task 1); `AdditiveProductGP` (Task 5); `TASK_TYPES` (Task 4).
- Produces:
  - `class FusionEstimator(task_type="regression", blocks=None, random_state=0)` — `fit(X, y, groups=None)`, `predict(X, return_std=False)`, `supports_std` (property, `False`), `inner_group_splits(groups, n_splits=3)` module function.
  - `class GPInteraction(FusionEstimator)` — extra args `interactions`, `ard_blocks=("descriptors",)`, `n_restarts=3`, `n_iter=60`, `lr=0.1`, `max_hyper_points=1200`; `supports_std = True`; attributes `kernel_report_`.
  - `class XGBoostFusion(FusionEstimator)` — extra args `grid=None`, `n_estimators=400`, `learning_rate=0.05`, `n_jobs=1`; `native_binary = True`; attribute `n_trees_`.

- [ ] **Step 1: Write the failing test** `test/test_fusion_models.py`

```python
import numpy as np
import pytest

from tackai.fusion.blocks import BLOCK_DIMS, block_index
from tackai.fusion.models import GPInteraction, XGBoostFusion


def synth(n=90, seed=0, binary=False):
    """A design matrix in the real block layout with a learnable signal."""
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())), dtype=np.float64)
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(size=(n, 217))
    X[:, idx["descriptors"][0]] *= 1e18                      # Ipc-scale column
    for b in ("poi", "e3", "cell", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([6.0, 24.0], (n, 1))
    signal = (X[:, idx["fingerprint"][:8]].sum(axis=1)
              + 2.0 * X[:, idx["poi"][0]] * X[:, idx["fingerprint"][0]]
              + X[:, idx["cell"][0]])
    groups = rng.integers(0, 12, n)
    if binary:
        return X, (signal > np.median(signal)).astype(float), groups
    return X, signal + 0.05 * rng.normal(size=n), groups


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_fit_returns_self_and_predict_has_the_right_shape(cls):
    X, y, g = synth()
    est = cls(task_type="regression", random_state=0)
    assert est.fit(X, y, g) is est
    pred = est.predict(X)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_predictions_are_in_original_target_units(cls):
    X, y, g = synth()
    y = y * 100.0 + 500.0
    est = cls(random_state=0).fit(X, y, g)
    pred = est.predict(X)
    assert abs(pred.mean() - y.mean()) < 0.5 * y.std()


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_learns_better_than_predicting_the_mean(cls):
    X, y, g = synth(n=120)
    tr = np.arange(90); te = np.arange(90, 120)
    est = cls(random_state=0).fit(X[tr], y[tr], g[tr])
    pred = est.predict(X[te])
    assert np.mean((pred - y[te]) ** 2) < np.mean((y[tr].mean() - y[te]) ** 2)


def test_gp_reports_std_in_original_units():
    X, y, g = synth()
    est = GPInteraction(random_state=0).fit(X, y * 100.0, g)
    mean, std = est.predict(X, return_std=True)
    assert est.supports_std and std.shape == mean.shape and (std > 0).all()
    far = X.copy(); far[:, block_index(BLOCK_DIMS)["poi"]] += 40.0
    _, std_far = est.predict(far, return_std=True)
    assert std_far.mean() > std.mean()


def test_xgboost_refuses_std():
    X, y, g = synth()
    est = XGBoostFusion(random_state=0).fit(X, y, g)
    assert est.supports_std is False
    with pytest.raises(NotImplementedError):
        est.predict(X, return_std=True)


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_binary_task_returns_probabilities(cls):
    X, y, g = synth(binary=True)
    est = cls(task_type="binary", random_state=0).fit(X, y, g)
    p = est.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.7


def test_gp_binary_std_is_a_probability_interval():
    X, y, g = synth(binary=True)
    est = GPInteraction(task_type="binary", random_state=0).fit(X, y, g)
    p, std = est.predict(X, return_std=True)
    assert ((p >= 0) & (p <= 1)).all() and (std >= 0).all() and (std <= 1).all()


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_constant_target_does_not_divide_by_zero(cls):  # Review Focus 3
    X, _, g = synth()
    y = np.full(len(X), 3.5)
    est = cls(random_state=0).fit(X, y, g)
    pred = est.predict(X)
    assert np.isfinite(pred).all() and np.allclose(pred, 3.5, atol=1e-3)


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_preprocessing_never_sees_the_test_rows(cls):
    X, y, g = synth(n=120)
    tr, te = np.arange(90), np.arange(90, 120)
    est = cls(random_state=0).fit(X[tr], y[tr], g[tr])
    a = est.predict(X[te])
    shifted = X.copy()
    shifted[te] = shifted[te] + 3.0        # changing test rows cannot change the fit
    b = est.predict(shifted[te])
    assert not np.allclose(a, b)           # predictions move
    est2 = cls(random_state=0).fit(X[tr], y[tr], g[tr])
    assert np.allclose(est2.predict(X[te]), a, rtol=1e-8, atol=1e-10)   # the fit did not


def test_xgboost_sees_raw_molecule_columns():
    """Trees must receive unscaled Morgan bits: the design matrix reaches them undistorted."""
    X, y, g = synth()
    est = XGBoostFusion(random_state=0).fit(X, y, g)
    Z = est.pre_.transform(X)
    idx = block_index(BLOCK_DIMS)
    assert np.allclose(Z["fingerprint"] * np.sqrt(1024), X[:, idx["fingerprint"]])


def test_xgboost_selects_a_grid_configuration():
    X, y, g = synth()
    est = XGBoostFusion(random_state=0).fit(X, y, g)
    assert set(est.hyper_) == {"max_depth", "reg_lambda"}
    assert est.n_trees_ >= 1


def test_gp_exposes_its_kernel_report():
    X, y, g = synth()
    est = GPInteraction(random_state=0).fit(X, y, g)
    rep = est.kernel_report_
    assert "prod:mol*poi" in rep["weight"] and rep["noise"] > 0


@pytest.mark.parametrize("cls", [GPInteraction, XGBoostFusion])
def test_same_seed_same_predictions(cls):
    X, y, g = synth()
    a = cls(random_state=3).fit(X, y, g).predict(X)
    b = cls(random_state=3).fit(X, y, g).predict(X)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10)
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_models.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.models'`

- [ ] **Step 3: Implement `tackai/fusion/models.py`**

Port `FusionBase` (tag `base_estimator`) with PCA removed:

- `fit(X, y, groups=None)`: `y_mean_`, `y_std_ = y.std() or 1.0` (the `or` guards the constant-target case); fit a `BlockPreprocessor` on the training rows only; standardise `y`; call `_fit_model`; for `task_type="binary"` and not `native_binary`, fit `LogisticRegression(C=1e4)` on scaffold-grouped inner out-of-fold scores (`inner_group_splits`, `GroupKFold(3)`) with the frozen hyper-parameters.
- `predict(X, return_std=False)`: for regression, rescale by `y_std_`/`y_mean_` (std by `y_std_` only); for binary, probabilities (`native_binary` → clip to [0,1]; otherwise Platt). With `return_std` on a binary Platt model, return `(platt(s+σ) − platt(s−σ))/2`. Base `supports_std` is `False` and `predict(..., return_std=True)` raises `NotImplementedError` unless the subclass supports it.
- `GPInteraction._fit_model`: `AdditiveProductGP(self.pre_.dims_, interactions=..., ard_blocks=...)`, `fit(Z, ys, ...)`; store `kernel_report_`. `_predict_model(model, Z, return_std)` delegates. Reuse of a tuned kernel for the inner Platt folds goes through `load_state`.
- `XGBoostFusion`: port `M7XGB` (tag `m7`) **minus** the PCA preprocessor override — `_make_preprocessor` is now the default `BlockPreprocessor` (molecule blocks raw, context standardised). Keep `GRID`, `GroupShuffleSplit(0.2)` early stopping, inner-CV grid choice, `XGBClassifier`/`XGBRegressor` by `task_type`, `n_trees_ = best_iteration + 1`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_models.py -q`
Expected: PASS, 20 passed

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/models.py test/test_fusion_models.py
git commit -m "feat(fusion): GPInteraction (M4) and XGBoostFusion (M7) estimators"
```

---

### Task 7: `FusionEnsemble` — generic members, cached context, `from_pretrained`

**Files:**
- Create: `tackai/fusion/ensemble.py`, `test/test_fusion_ensemble.py`
- Modify: `tackai/fusion/__init__.py` (export the full public surface)

**Interfaces:**
- Consumes: `FusionData` (Task 4), `GPInteraction` / `XGBoostFusion` (Task 6), `BLOCK_ORDER`/`block_index` (Task 1).
- Produces:
  - `@dataclass FusionPrediction` — `mean`, `std`, `member_predictions: dict[str, np.ndarray]`, `member_std`, `weights`, `task`, `label_name`, `ci_lower_95`, `ci_upper_95`, `ok`; `to_frame() -> pd.DataFrame`, `summary() -> str`.
  - `@dataclass FusionContext` — `columns: np.ndarray`, `values: np.ndarray`, `per_member: list[dict[str, np.ndarray]]`, `source: dict`.
  - `class FusionEnsemble(members, data, task, weights=None)` with classmethods
    `fit(factory, data, task, *, splits=None, n_members=5, n_repeats=5, n_folds=5, n_jobs=1, verbose=False) -> FusionEnsemble`,
    `from_pretrained(path_or_repo_id, *, task=None, revision=None, token=None, data=None) -> FusionEnsemble`;
    methods `save(path)`, `transform_context(record) -> FusionContext`, `predict(samples, context=None, return_individual=True, return_std=True) -> FusionPrediction`, `available_tasks`.

- [ ] **Step 1: Write the failing test** `test/test_fusion_ensemble.py`

```python
import numpy as np
import pytest

from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble, FusionPrediction
from tackai.fusion.models import GPInteraction, XGBoostFusion
from test.conftest import CELLS, SEQS, SMILES

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


def test_fit_produces_the_requested_number_of_members(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)
    assert len(ens.members) == 3 and ens.task == "pdc50"
    assert sum(ens.weights.values()) == pytest.approx(1.0)


def test_predict_is_the_weighted_mean_of_the_members(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    pred = ens.predict([{"smiles": SMILES[0], **CTX}])
    assert isinstance(pred, FusionPrediction)
    members = np.array([p for p in pred.member_predictions.values()])
    expected = np.average(members, axis=0, weights=list(ens.weights.values()))
    assert np.allclose(pred.mean, expected)


def test_context_path_equals_the_ordinary_path_exactly(data):
    """The whole point of the context cache: identical numbers, less work."""
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    ctx = ens.transform_context(CTX)
    fast = ens.predict(SMILES[:4], context=ctx)
    slow = ens.predict([{"smiles": s, **CTX} for s in SMILES[:4]])
    assert np.allclose(fast.mean, slow.mean, rtol=0, atol=0)
    assert np.allclose(fast.std, slow.std, rtol=0, atol=0)


def test_std_combines_member_variance_and_member_spread(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    means = np.array(list(pred.member_predictions.values()))
    varis = np.array(list(pred.member_std.values())) ** 2
    w = np.array(list(ens.weights.values()))[:, None]
    expected = np.sqrt((w * varis).sum(0) + (w * (means - pred.mean) ** 2).sum(0))
    assert np.allclose(pred.std, expected)


def test_std_falls_back_to_member_spread_without_predictive_variance(data):
    ens = FusionEnsemble.fit(fast_xgb, data, task="pdc50", n_members=3, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    means = np.array(list(pred.member_predictions.values()))
    w = np.array(list(ens.weights.values()))[:, None]
    assert np.allclose(pred.std, np.sqrt((w * (means - pred.mean) ** 2).sum(0)))


def test_confidence_interval_brackets_the_mean(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    assert (pred.ci_lower_95 <= pred.mean).all() and (pred.mean <= pred.ci_upper_95).all()


def test_members_may_mix_estimator_classes(data):
    """Generic over members: same input for all of them."""
    ens = FusionEnsemble.fit([fast_gp, fast_xgb, fast_gp], data, task="pdc50", n_folds=3)
    assert len(ens.members) == 3
    pred = ens.predict(SMILES[:2], context=ens.transform_context(CTX))
    assert pred.mean.shape == (2,) and np.isfinite(pred.mean).all()


def test_binary_task_predicts_probabilities(data):
    ens = FusionEnsemble.fit(fast_xgb, data, task="activity", n_members=2, n_folds=3)
    pred = ens.predict(SMILES[:3], context=ens.transform_context(CTX))
    assert ((pred.mean >= 0) & (pred.mean <= 1)).all()
    assert pred.label_name


def test_save_and_from_pretrained_round_trip(data, tmp_path):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    before = ens.predict(SMILES[:3], context=ens.transform_context(CTX)).mean
    ens.save(tmp_path / "ens")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "ens")
    assert loaded.task == "pdc50" and len(loaded.members) == 2
    after = loaded.predict(SMILES[:3], context=loaded.transform_context(CTX)).mean
    assert np.allclose(before, after, rtol=0, atol=0)


def test_manifest_records_the_block_layout(data, tmp_path):
    import json
    FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3).save(tmp_path / "ens")
    manifest = json.loads((tmp_path / "ens" / "manifest.json").read_text())
    assert manifest["task"] == "pdc50"
    assert manifest["block_dims"]["fingerprint"] == 1024
    assert manifest["n_members"] == 1 and "tackai" in manifest["versions"]


def test_invalid_smiles_yields_nan_not_an_exception(data):  # Review Focus 2
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    pred = ens.predict(["not_a_molecule", SMILES[0]], context=ens.transform_context(CTX))
    assert list(pred.ok) == [False, True]
    assert np.isnan(pred.mean[0]) and np.isfinite(pred.mean[1])


def test_empty_batch_returns_empty_arrays(data):  # Review Focus 5
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    pred = ens.predict([], context=ens.transform_context(CTX))
    assert pred.mean.shape == (0,) and pred.std.shape == (0,)


def test_unseen_sequence_in_a_context_raises(data):  # Review Focus 1
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=1, n_folds=3)
    with pytest.raises(KeyError, match="poi"):
        ens.transform_context({**CTX, "poi_seq": "MKKKWWWNOTINTABLE"})


def test_to_frame_has_one_row_per_molecule(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    df = ens.predict(SMILES[:3], context=ens.transform_context(CTX)).to_frame()
    assert len(df) == 3 and {"prediction", "std", "ci_lower_95", "ci_upper_95"} <= set(df.columns)
```

- [ ] **Step 2: Run it to watch it fail**

Run: `uv run pytest test/test_fusion_ensemble.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tackai.fusion.ensemble'`

- [ ] **Step 3: Implement `tackai/fusion/ensemble.py`**

- `fit(factory, data, task, ...)`: `factory` is a callable or a list of callables (one per member); take the task's rows and splits, fit member `k` on `splits[r][f]`'s training indices walking repeats then folds, stop at `n_members`, pass `task_type=TASK_TYPES[task]` and `random_state=k`. `n_jobs > 1` uses `joblib.Parallel` with the loky backend (keep `OMP_NUM_THREADS=1`). Equal weights by default; `weights` accepts a dict or array.
- `transform_context(record)`: encode the context once through `data.encode_context`, then per member store `member.pre_.transform_blocks(full_row, only=context_blocks)` so a batch only transforms molecule columns. Record the source dict for `FusionPrediction`.
- `predict(samples, context=None, ...)`:
  - with `context`: `samples` is SMILES string(s) → `data` featurises them on the fly, each member's `A` is assembled from its cached context blocks plus the freshly transformed molecule blocks, then the member's `_predict_model` runs.
  - without: `samples` is dicts/DataFrame → `data.encode(...)` then each member's ordinary `predict`.
  - Rows whose SMILES failed to parse get `NaN` and `ok=False` and are excluded from the member calls.
  - Aggregate with the law of total variance; members lacking `supports_std` contribute zero variance. CI = `mean ± 1.96·std`.
- `save(path)`: `joblib.dump` per member (`member_00.joblib`, …) plus `manifest.json` with `task`, `n_members`, `weights`, `block_dims`, the encoder settings (`protein_space`, featuriser radius/size, cache filenames) and `versions` (tackai, numpy, torch, xgboost, sklearn).
- `from_pretrained(path_or_repo_id, ...)`: local directory → read the manifest and members; otherwise `huggingface_hub.snapshot_download(repo_id, revision=..., token=...)` first (lazy import, mirroring `EnsemblePredictor`). Rebuild a `FusionData` encoder-only instance from the manifest when `data` is not supplied — inference needs the encoder and the block layout, not the training table.
- `FusionPrediction.label_name` from `{"dmax": "Dmax (fraction)", "pdc50": "pDC50", "activity": "P(active)"}`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_fusion_ensemble.py -q`
Expected: PASS, 14 passed

- [ ] **Step 5: Run the whole fusion suite**

Run: `uv run pytest test/ -q`
Expected: all fusion tests pass; `test/test_ensemble.py` may still fail for the pre-existing missing-`ensembles/` reason (not ours).

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/ensemble.py tackai/fusion/__init__.py test/test_fusion_ensemble.py
git commit -m "feat(fusion): generic FusionEnsemble with cached context and from_pretrained"
```

---

### Task 8: Real-cache tests and the training / inference notebook

**Files:**
- Create: `test/test_fusion_real_cache.py`, `notebooks/fusion_surrogates.ipynb`
- Modify: `docs/` index if it lists notebooks (check `mkdocs.yml`)

**Interfaces:**
- Consumes: the whole public surface of `tackai.fusion`.
- Produces: no code other modules import.

- [ ] **Step 1: Write the real-cache tests** `test/test_fusion_real_cache.py`

```python
"""Opt-in checks against the real TACKAI_CACHE: filenames, keys and dimensions."""
import os
from pathlib import Path

import numpy as np
import pytest

from tackai.data.utils import get_cache_dir
from tackai.fusion.context import CONTEXT_FILES, ContextEncoder

pytestmark = pytest.mark.requires_cache

CACHE = Path(get_cache_dir())


def missing():
    return [f["table"] for f in CONTEXT_FILES.values() if not (CACHE / f["table"]).exists()]


@pytest.mark.skipif(bool(missing()), reason="TACKAI_CACHE lacks the context embedding tables")
def test_real_tables_have_the_documented_dimensions():
    enc = ContextEncoder()
    assert enc.dim("cell") == 47
    assert enc.dim("poi") == 51
    assert enc.dim("e3") == 7
    assert enc.dim("assay") == 8
    assert ContextEncoder(protein_space="combined").dim("poi") == 52


@pytest.mark.skipif(bool(missing()), reason="TACKAI_CACHE lacks the context embedding tables")
def test_real_cache_has_the_unknown_cell_line_key():
    enc = ContextEncoder()
    assert enc.encode("cell", [None]).shape == (1, 47)


@pytest.mark.skipif(bool(missing()), reason="TACKAI_CACHE lacks the context embedding tables")
def test_real_assay_table_covers_the_open_vocabulary():
    enc = ContextEncoder()
    for raw in ["Western blot", "HiBiT", "HTRF", "simple western", None]:
        assert enc.encode("assay", [raw]).shape == (1, 8)


@pytest.mark.skipif(not (Path("data/yaochen/TACKv2.csv")).exists(), reason="dev CSV not present")
def test_development_table_encodes_end_to_end():
    from tackai.fusion.data import FusionData
    data = FusionData.from_csv(["data/yaochen/TACKv2.csv"])
    assert data.X.shape[1] == 1355
    assert np.isfinite(data.X[:, data.context_columns]).all()
    idx, X, y, g = data.task_rows("dmax")
    assert len(y) > 100 and np.isfinite(y).all()
```

- [ ] **Step 2: Run them**

Run: `uv run pytest test/test_fusion_real_cache.py -q -m requires_cache`
Expected: PASS (4 passed) on this machine, since the cache and CSVs are present.

- [ ] **Step 3: Write the notebook** `notebooks/fusion_surrogates.ipynb`

Build it as a percent-format python file in the scratchpad and convert with `jupytext`/`nbformat`, then execute with `nbconvert`. Sections:

1. **Setup** — `OMP_NUM_THREADS=1` before imports, `load_dotenv`, print `TACKAI_CACHE` and library versions.
2. **Data** — `FusionData.from_csv([...AutoTPDplus-PROTAC-training.csv, ...TACKv2.csv])`; print the block layout, dims, row counts per task, scaffold-group statistics.
3. **A single model** — fit `GPInteraction` and `XGBoostFusion` on one scaffold-grouped fold of `pdc50`; print R² and the GP's `kernel_report()` (lengthscales and per-term weights); scatter of predicted vs observed with GP error bars.
4. **Ensembles** — `FusionEnsemble.fit(GPInteraction, data, task=...)` with 5 members for each of the three tasks; report per-member and ensemble out-of-fold scores (R², Spearman, ROC-AUC for activity); `save()` each to `ensembles/fusion_<task>/`.
5. **Inference** — `FusionEnsemble.from_pretrained("ensembles/fusion_pdc50")`; score a handful of SMILES in one context with `transform_context`, show `to_frame()` with uncertainty; demonstrate that an unseen POI sequence raises `KeyError` and that an invalid SMILES yields `NaN` with `ok=False`.
6. **Screening throughput** — time `transform_context` once plus `predict` over 512 SMILES, report molecules/second split into featurisation and scoring, and compare the context path against the ordinary path (same numbers, less time).
7. **Conclusions** — what to use for an RL loop, with the member-count trade-off.

Run it to completion:

```bash
cd notebooks && OMP_NUM_THREADS=1 ../.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --output fusion_surrogates.ipynb fusion_surrogates.ipynb
```
Expected: executes with no errors; every section prints its numbers and the figures are inline.

- [ ] **Step 4: Commit**

```bash
git add test/test_fusion_real_cache.py notebooks/fusion_surrogates.ipynb
git commit -m "test(fusion): real-cache checks; docs: training and inference notebook"
```

---

## Notes for the executor

- `test/` has no `__init__.py`; `from test.conftest import SMILES` works because `pyproject.toml` sets `rootdir`-relative imports. If it does not, add `test/__init__.py` in Task 1 and ledger the ruling.
- The notebook in Task 8 is the only place that needs the real cache and the real CSVs. Everything before it runs hermetically.
- Fitting a GP member on the full `dmax` table takes minutes; the notebook is the first place that happens. Keep `n_restarts`/`n_iter` at the defaults there and small in tests.
