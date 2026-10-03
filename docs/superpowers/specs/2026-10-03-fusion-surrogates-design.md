# Fast fusion surrogates (M4 GP + M7 XGBoost) as a tackai subpackage — design

**Date:** 2026-10-03
**Branch:** `fast-surrogates`
**Status:** approved (design discussion 2026-10-03)

## Goal

Promote the two winning fusion methods of `notebooks/fusion_comparison.ipynb` — **M4**
(additive-kernel GP with cross-block product kernels) and **M7** (regularised gradient-boosted
trees) — from notebook cells to maintained classes in the `tackai` package, running on the
pre-reduced context embeddings selected by the three embedding notebooks, with **no PCA applied
inside the pipeline**. Add the data handler they share and a generic ensemble that keeps the
`from_pretrained` / `predict` API of `tackai.ensemble_predictor.EnsemblePredictor`.

## Why

M4 and M7 are the only two methods that credibly beat ridge regression on all three tasks
(dmax R² .474 / .417, pDC50 .579 / .551, activity AUC .871 / .867 over 25 scaffold-grouped folds).
They live in notebook cells, so nothing outside the notebook can use them. The target consumer is
a reinforcement-learning-style loop: a fixed biological context, a stream of new molecules,
hundreds of thousands of scores — which is why context caching and on-the-fly molecular
featurisation are requirements rather than optimisations.

## Decisions taken during the design discussion

| Decision | Choice | Alternative rejected |
|---|---|---|
| Protein blocks | POI from `poi_pca51`, E3 from `e3_pca7` (per-block spaces) | `combined_pca52` for both — kept selectable, not default |
| Molecule scaling | **No scaler** on fingerprint or descriptors; raw, divided by `sqrt(width)` | StandardScaler ÷ √width (blows up rare Morgan bits) |
| Tasks | `dmax`, `pdc50`, `activity` — the notebook's three | repo aliases (`dc50`, `bin`) |
| Test dependencies | Synthetic npz fixtures in `tmp_path` + opt-in `requires_cache` tests | requiring the real cache; hermetic-only |
| GP backend | Plain-torch exact GP with cached per-block distances | GPyTorch port (recomputes distances every step; undeclared dependency) |
| Unseen POI/E3 sequence | Raise `KeyError` — confirmed acceptable by the user | on-the-fly ESM (impossible: `mean_rm_pc2` is table-fitted) |

## Block layout

No PCA is applied anywhere in the pipeline: the context arrives already reduced from the cache,
and the molecule blocks stay raw.

| block | dim | source | preprocessing | kernel role (M4) |
|---|---|---|---|---|
| `fingerprint` | 1024 | on the fly, Morgan r=16, `includeChirality=True` | none, ÷ √1024 | RBF (isotropic, cached distances) |
| `descriptors` | 217 | on the fly, all of `Descriptors._descList`, invalid → `-1` | mean-impute, ÷ √217 | RBF with **ARD** |
| `poi` | 51 | `protein_embeddings_..._layer=18_pooling=lse_window=1022_block=poi_pca51.npz` | mean-impute, standardise, ÷ √51 | RBF |
| `e3` | 7 | `protein_embeddings_..._layer=30_pooling=mean_rm_pc2_window=1022_block=e3_pca7.npz` | mean-impute, standardise, ÷ √7 | RBF |
| `cell` | 47 | `cell_embeddings_model=sentence-transformer_pooling=mean_pca47.npz` | mean-impute, standardise, ÷ √47 | RBF |
| `assay` | 8 | `assay_embeddings_vocab=open20_model=all-mpnet-base-v1_pooling=mean_pca8.npz` | mean-impute, standardise, ÷ √8 | RBF |
| `assay_time` | 1 | `mean(DC50_h, Dmax_h)` | median-impute, standardise | linear |

`combined_pca52` is registered as a selectable alternative protein space (`protein_space="combined"`).

**Why ARD on descriptors.** `Ipc` is in tackai's descriptor set and is clamped only at `1e20`,
while `FractionCSP3` is ~0.3. With no scaler (the user's choice) a single lengthscale cannot fit
both, so the descriptor block gets one lengthscale per column, learned by marginal likelihood.
Isotropic blocks get cached squared-distance matrices; ARD blocks recompute `X/l` distances each
step (217 columns, one matmul, milliseconds) because per-column distances cannot be cached in
`N×N` memory.

**Interactions.** The three benchmarked product kernels are the default: `mol*poi`, `mol*cell`,
`poi*cell`, where the molecular kernel is `RBF_fingerprint + RBF_descriptors`. `mol*e3` is newly
expressible (E3 is an embedding now, not a one-hot) and is available via `interactions=` but off by
default, so M4 stays comparable to the measured numbers.

E3 and assay stop being one-hot blocks, so the GP's `small_linear` term now covers `assay_time` alone.

## Components

```
tackai/fusion/
  __init__.py     public exports
  blocks.py       BLOCK_ORDER, DENSE_BLOCKS, SMALL_BLOCKS, BlockPreprocessor
  context.py      ContextEncoder — the five npz tables, lookups, PCA-projection fallback
  features.py     MolFeaturizer — on-the-fly Morgan + RDKit, dedup + memo cache
  data.py         FusionData — the data handler (training and inference)
  gp.py           AdditiveProductGP — plain-torch exact GP, cached distances, exact variance
  models.py       FusionEstimator, GPInteraction (M4), XGBoostFusion (M7)
  ensemble.py     FusionEnsemble, FusionContext, FusionPrediction
```

### `ContextEncoder` (context.py)

Loads the five npz tables lazily from `TACKAI_CACHE` and encodes context keys to rows.

- `encode(block, keys) -> (n, dim) float32`
- **cell**: missing/empty → the cache's `"Unknown cell line."` vector (a real key in the npz);
  an accession absent from the table → `KeyError`.
- **assay**: free text canonised by the ported `normalize_assay` (open vocabulary); a canonical type
  absent from the table is embedded on the fly with `all-mpnet-base-v1` and projected through
  `assay_pca_vocab=open20_...npz` (`mean_`, `components_`).
- **poi / e3**: looked up by stripped sequence; unseen sequence → `KeyError` naming the block and
  the first 30 residues. This cannot be fixed on the fly (`lse` is not in `ProteinEmbedding`;
  `mean_rm_pc2` is fitted on a whole vocabulary). Escape hatch:
  `register_sequence(block, seq, embedding_640d)` projects a user-supplied full embedding through
  the cached PCA.
- `register_sequence` and the assay fallback both use the `_model.npz` side files.

### `MolFeaturizer` (features.py)

On-the-fly molecular features, **bit-exact with `tackai.data.embeddings.mol_embeddings.MolEmbedding`**
(asserted by a test): Morgan r=16/1024 `includeChirality=True`; all 217 `Descriptors._descList`
values with `None`/NaN/inf/`|v|>1e20` → `-1`; `Ipc` and `AvgIpc` share one characteristic
polynomial. De-duplicates within a call, memoises across calls, optional process pool.

- `featurize(smiles) -> (fp (n,1024), desc (n,217), ok (n,) bool)`; an unparseable SMILES yields
  zero rows and `ok=False` rather than raising.

### `FusionData` (data.py) — the data handler

One class for both training and inference.

```python
data = FusionData.from_csv([...])          # defaults to the two dev CSVs
data.X, data.groups, data.smiles           # design matrix, Murcko scaffold ids
data.target("dmax")                        # make_targets: dmax / pdc50 / activity
data.task_rows("dmax")                     # finite-target view: (idx, X, y, groups)
data.splits("dmax", n_repeats=5, n_folds=5)  # StratifiedGroupKFold, no group split
data.encode(records)                       # inference: dicts/DataFrame -> design matrix
data.encode_context(record)                # one context row, reused across many SMILES
data.assemble(ctx_row, smiles)             # mol blocks on the fly + context broadcast
```

Targets follow the notebook's `make_targets` exactly, including the partial-information rules for
`activity`. Context blocks of a training table are cached under
`TACKAI_CACHE/fusion_blocks/<content-hash>/` (the hash covers SMILES, context keys, vocabularies,
model names and featuriser settings — never a label). Molecular features at **inference** are always
computed on the fly.

### Models (models.py, gp.py)

`FusionEstimator` is the shared harness: fit-time-only preprocessing, target standardisation,
binary handling. Subclasses implement `_fit_model` / `_predict_model`.

```python
est.fit(X, y, groups=None) -> self
est.predict(X) -> (n,)                        # original target units
est.predict(X, return_std=True) -> (mean, std)  # GP only; XGBoost raises NotImplementedError
est.supports_std -> bool
```

- `GPInteraction` (M4): `interactions`, `ard_blocks=("descriptors",)`, `n_restarts=3`, `n_iter=60`,
  `lr=0.1`, `max_hyper_points=1200`. Hyper-parameters are fitted on at most 1200 rows, then the exact
  GP uses all training rows (as in the notebook). Posterior variance is exact and returned in
  original units. For `task_type="binary"` the latent score is Platt-calibrated on scaffold-grouped
  inner out-of-fold scores, and `return_std` reports `(platt(s+σ) − platt(s−σ))/2`.
- `XGBoostFusion` (M7): native binary (`XGBClassifier`), `GRID` over `max_depth ∈ {3,5}` ×
  `reg_lambda ∈ {5,20}` chosen by scaffold-grouped inner CV, `n_estimators=400`, `lr=0.05`,
  early stopping on a grouped 20% split. Molecule blocks reach the trees raw.
- Fold-internal: a constant training target (`y_std == 0`) and constant context columns must not
  produce division by zero.

### `FusionEnsemble` (ensemble.py)

Generic over members: every member consumes the **same** design matrix, so members may mix
`GPInteraction` and `XGBoostFusion`.

```python
ens = FusionEnsemble.fit(factory, data, task="dmax", n_members=5, n_jobs=1)
ens.save(dir)
ens = FusionEnsemble.from_pretrained(dir_or_hf_repo_id, task="dmax")
ctx = ens.transform_context({"poi_seq": ..., "cell_id": "CVCL_0031", "e3_seq": ..., "assay": "western blot", "assay_time": 24})
pred = ens.predict(smiles_list, context=ctx)      # fast path
pred = ens.predict(records)                        # ordinary path
pred.mean, pred.std, pred.member_predictions, pred.ci_lower_95, pred.ci_upper_95
```

- Members are fitted on the task's scaffold-grouped CV splits (default 5 members, since the speed
  notebook found 5 ≈ 25 in accuracy at a quarter of the cost; `n_members` up to 25).
- `transform_context` precomputes, **per member**, the processed context columns, so a batch only
  pushes the molecule columns through each member's preprocessor. Context-path and ordinary-path
  predictions must agree exactly.
- Uncertainty combines both sources by the law of total variance:
  `std = sqrt(mean(member_var) + var(member_means))`, degrading to member spread alone when no
  member supports predictive variance.
- Persistence: one joblib file per member plus `manifest.json` (block layout and dims, task,
  weights, `FusionData` encoder settings, library versions). `from_pretrained` accepts a local
  directory or a Hugging Face repo id (`snapshot_download`), mirroring `EnsemblePredictor`.

## Testing

TDD, tests before implementation, in `test/`. A `fake_cache` fixture writes tiny npz tables using
the **real filenames, keys and dimensions** into `tmp_path` and points `TACKAI_CACHE` there, so the
whole suite runs on a fresh checkout in seconds. Tests marked `requires_cache` exercise the real
artifacts and skip when absent.

Behaviours that must be covered: no-leakage preprocessing (fit on train rows only), no PCA anywhere,
scaffold-group split integrity, `encode(records)` reproducing the training row for identical inputs,
cached-distance GP equal to a naive recomputation, exact GP variance against the textbook formula,
context-path equal to ordinary-path ensemble prediction, save/`from_pretrained` round-trip, and
bit-exactness of `MolFeaturizer` against `MolEmbedding`.

## Out of scope

- M1, M2, M3, M5, M6 — not promoted.
- The notebook's statistics machinery (RM-ANOVA, Nadeau-Bengio, Friedman) stays in the notebook.
- Kernel-folding of context-only GP terms (the speed notebook's numpy export engine): the per-member
  context precompute gets most of the win at a fraction of the complexity.
- Adding `lse` / `mean_rm_pc2` pooling to `ProteinEmbedding` (notebook §8.1), which would be the
  prerequisite for on-the-fly POI embeddings.
- Hydra configs and `tack` CLI subcommands for the new classes.
