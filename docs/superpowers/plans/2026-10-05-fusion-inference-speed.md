# Fusion Ensemble Inference Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `FusionEnsemble` inference faster (float32 default, no redundant work), make every member see one canonical biological context, and prove both with an executed benchmark notebook that compares against numbers captured before the change.

**Architecture:** Task 1 captures the *before* numbers from the unchanged package. Tasks 2-8 change `tackai/fusion/{blocks,gp,models,data,ensemble}.py` in dependency order: the preprocessor first (no standardisation, float32 apply, consensus), then the GP (float32, float64 promotion, train-side cache), then the estimators and ensemble that consume them. Task 9 builds and executes the notebook that overlays before and after.

**Tech Stack:** Python 3.13 (`.venv`), numpy 2.5, torch 2.11, scikit-learn, xgboost, RDKit, pytest, nbformat / nbconvert.

**Spec:** `docs/superpowers/specs/2026-10-05-fusion-inference-speed-design.md` (amended by the "Plan amendments" section at its end).

## Global Constraints

- Run Python and tests as `.venv/bin/python` and `.venv/bin/python -m pytest`. `OMP_NUM_THREADS=1` must be set before torch/xgboost import (the `.venv` has two libomp copies and segfaults otherwise); `test/conftest.py` already does it, and so must any script or notebook.
- Default dtype is **float32** in `AdditiveProductGP` and the preprocessor output, *including the fit and the Cholesky*. A failed float32 factorisation is retried in float64 and the promotion is recorded.
- Preprocessor statistics are **fitted in float64 and applied in float32** (a raw `Ipc` near 1e18 would lose its imputer mean in float32).
- **No block is standardised anywhere**, `assay_time` included. `÷ sqrt(width)` stays for every dense block.
- Jitter ladder: float32 tries 1e-5, 1e-4, 1e-3 and then promotes; float64 keeps 1e-6 up to 1e-2.
- The canonical context is the **consensus** (mean of the members' fitted statistics) of the members' context preprocessors, used on all three prediction paths (cached context, records, `predict_matrix`).
- The superseded behaviour stays reachable: `FusionEnsemble(shared_context=False)`, `dtype="float64"`, `BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS)`.
- Redundancy removals (R1-R3, R5, R6, R7) are gated on **bit-exact** equality with the unoptimised formula at the same dtype. Only float32, shared context and no-standardisation may change values.
- Accuracy is measured and reported, never a pass/fail gate (user, 2026-10-05: degradation is accepted, the ensembles are retrained later with hyper-parameter optimisation).
- Where an existing test asserts a tolerance tighter than float32 can give, pin `dtype="float64"` in that test and add a float32 sibling at a stated tolerance. Never loosen the original assertion.
- `tackai/fusion/data.py` carries one uncommitted docstring edit and `tackai/fusion/features.py` an uncommitted import cleanup that predate this plan. Never `git add -A` / `git add .`; stage named files only, and do not touch `features.py`. Task 6 commits `data.py` whole and says so in its message.
- Float32 sibling tolerances written in this plan are estimates from error analysis (a float32 solve with condition number ~1e5 loses ~1e-2 relative accuracy). If a sibling fails, measure the actual error, set the tolerance to about 3x that, and ledger a `Ruling:` with the observed number. An error above ~1e-1 is a defect, not a tolerance.
- float64 promotion refactorises the **float32-assembled** kernel in float64. It removes the factorisation's own rounding error but not the rounding already in the stored kernel. If the notebook reports promotions, the follow-up is to reassemble the kernel in float64; that is out of scope here and is recorded in the spec's Risks.
- Commit trailer, on every commit: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Review Focus

Input classes the spec implies and the per-task tests would otherwise miss; each has a test in the named task.

1. **Artifacts pickled before this change** (float64 GP, preprocessor without `dtype`, no `_train_side_`, no `promoted_`) must load, score and give the same numbers as before: Tasks 2, 5, 8.
2. **NaN molecular features and invalid SMILES under the shared molecular transform**: sharing must switch off when a feature is NaN, and invalid rows must stay NaN without disturbing valid ones: Task 6.
3. **float32 overflow and cancellation**: descriptor columns at the featuriser's 1e20 limit and near-duplicate molecules must give a finite, positive-definite kernel: Task 3.
4. **The preprocessor must never write into the caller's array**: it now uses in-place operations, and `predict_matrix(X)` hands it the user's array: Task 2.
5. **A context with a missing `assay_time`, mixed GP/XGBoost members, and members that disagree about layout or scaling**: Task 7.

---

### Task 1: Benchmark harness and the "before" numbers

**Files:**
- Create: `notebooks/ensemble_inference_bench.py`
- Create: `test/test_inference_bench.py`
- Create (by running the harness): `notebooks/ensemble_inference_speed_results/baseline_{scaling,breakdown,redundancy,setup,featurise,heldout}.csv`, `baseline_predictions.npz`, `baseline_context.json`

**Interfaces:**
- Consumes: `FusionEnsemble.{from_pretrained,transform_context,predict,predict_matrix}`, `FusionData.{from_csv,encode,task_rows,splits}`, `MolFeaturizer.featurize` — all as they exist now.
- Produces (used by Task 9): module `ensemble_inference_bench` with
  - `stamp() -> {"commit": str, "dirty": bool}`
  - `timeit(fn, repeat=7, warmup=1) -> float` (median seconds)
  - `context_record(data) -> dict`, `smiles_pool(data, size) -> list[str]`
  - `sweep_paths(ens, record, pool, batch_sizes=BATCH_SIZES, repeat=5) -> DataFrame[path,batch,seconds,us_per_mol,mol_per_s]` over `PATHS = ("context+std","context","records","matrix")`
  - `StageTimer(ens)` context manager with `.seconds: dict[stage -> exclusive seconds]`
  - `stage_breakdown(ens, record, smiles, repeat=5) -> DataFrame[path,stage,seconds,share,batch]` for paths `context`, `context_nostd`, `records`
  - `redundancy_costs(ens, record, smiles, repeat=15) -> DataFrame[item,what,seconds,share_of_predict,batch,members]` with items `R1`..`R7`
  - `heldout_score(ens, data, task, n_folds=5) -> float` (R² or ROC-AUC on the held-out fold of split 0)
  - `capture_baseline(ensembles, data, out_dir, *, label="baseline", batch_sizes=BATCH_SIZES, pool_size=1024, breakdown_batch=256, repeat=5, heldout_folds=5) -> None`
  - `main(label="baseline", out_dir=None) -> None` (command line: `python notebooks/ensemble_inference_bench.py [label]`)

- [ ] **Step 1: Write the failing tests**

Create `test/test_inference_bench.py`:

```python
"""The benchmark harness itself: it must measure what it says, on old and new code alike."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import ensemble_inference_bench as bench  # noqa: E402

from tackai.fusion.data import FusionData  # noqa: E402
from tackai.fusion.ensemble import FusionEnsemble  # noqa: E402
from tackai.fusion.models import GPInteraction  # noqa: E402


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


@pytest.fixture
def small(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    return data, ens


def test_context_record_is_encodable(small):
    data, ens = small
    ens.transform_context(bench.context_record(data))           # must not raise


def test_sweep_paths_reports_every_path_at_every_affordable_batch(small):
    data, ens = small
    pool = bench.smiles_pool(data, 8)
    frame = bench.sweep_paths(ens, bench.context_record(data), pool,
                              batch_sizes=(1, 4, 64), repeat=1)
    assert set(frame["path"]) == set(bench.PATHS)
    assert sorted(frame["batch"].unique()) == [1, 4]            # 64 > len(pool) is skipped
    assert (frame["seconds"] > 0).all()
    assert np.allclose(frame["mol_per_s"] * frame["seconds"], frame["batch"])


def test_stage_timer_attributes_time_and_restores_the_methods(small):
    data, ens = small
    record, pool = bench.context_record(data), bench.smiles_pool(data, 4)
    ctx = ens.transform_context(record)
    pre = ens.members[0].pre_
    with bench.StageTimer(ens) as timer:
        ens.predict(pool, context=ctx)
    assert {"featurise", "transform", "model", "aggregate"} <= set(timer.seconds)
    assert all(v >= 0 for v in timer.seconds.values())
    assert "transform_blocks" not in vars(pre) and "featurize" not in vars(data.featurizer)


def test_stage_breakdown_accounts_for_the_whole_wall_time(small):
    data, ens = small
    frame = bench.stage_breakdown(ens, bench.context_record(data),
                                  bench.smiles_pool(data, 8), repeat=2)
    assert set(frame["path"]) == {"context", "context_nostd", "records"}
    for path, part in frame.groupby("path"):
        assert 0.98 <= part["share"].sum() <= 1.02, path


def test_redundancy_costs_cover_every_item(small):
    data, ens = small
    frame = bench.redundancy_costs(ens, bench.context_record(data),
                                   bench.smiles_pool(data, 8), repeat=2)
    assert list(frame["item"]) == ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    assert (frame["seconds"] >= 0).all() and (frame["share_of_predict"] >= 0).all()


def test_capture_baseline_writes_stamped_files(small, tmp_path):
    data, ens = small
    bench.capture_baseline({"pdc50": ens}, data, tmp_path, pool_size=8, batch_sizes=(1, 4),
                           breakdown_batch=4, repeat=1, heldout_folds=3)
    for name in ("scaling", "breakdown", "redundancy", "setup", "featurise", "heldout"):
        frame = pd.read_csv(tmp_path / f"baseline_{name}.csv")
        assert len(frame) > 0 and frame["commit"].astype(str).str.len().gt(0).all(), name
    assert (pd.read_csv(tmp_path / "baseline_scaling.csv")["task"] == "pdc50").all()
    assert (tmp_path / "baseline_predictions.npz").exists()
    assert (tmp_path / "baseline_context.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest test/test_inference_bench.py -q 2>&1 | tail -5`
Expected: collection error, `ModuleNotFoundError: No module named 'ensemble_inference_bench'`.

- [ ] **Step 3: Write the harness**

Create `notebooks/ensemble_inference_bench.py`:

```python
"""Timing helpers for ``notebooks/ensemble_inference_speed.ipynb``.

Everything here goes through the public surface of :class:`FusionEnsemble` plus a few
long-lived internals, so one module measures the package both before and after the speed
work. Results are stamped with the commit they were measured at.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")     # torch + xgboost libomp clash on macOS

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, roc_auc_score

from tackai.fusion.data import TASK_TYPES, FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.features import MolFeaturizer

REPO = Path(__file__).resolve().parent.parent
DEV_FILES = [REPO / "data/yaochen/AutoTPDplus-PROTAC-training.csv",
             REPO / "data/yaochen/TACKv2.csv"]
RESULTS = REPO / "notebooks" / "ensemble_inference_speed_results"
BATCH_SIZES = (1, 4, 16, 64, 256, 1024)
PATHS = ("context+std", "context", "records", "matrix")


def stamp() -> Dict[str, object]:
    """The commit the package was at when a number was measured, and whether it was dirty."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "tackai"))}


def timeit(fn: Callable[[], object], repeat: int = 7, warmup: int = 1) -> float:
    """Median wall-clock seconds of ``fn`` over ``repeat`` calls, after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def context_record(data: FusionData) -> dict:
    """An encodable experimental context, from the first row that has an assay time."""
    table = data.table
    row = table[table["assay_time"].notna()].iloc[0]
    return {"poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"], "cell_id": row["cell_key"],
            "assay": row["assay_raw"], "assay_time": float(row["assay_time"])}


def smiles_pool(data: FusionData, size: int) -> List[str]:
    """The first ``size`` distinct development SMILES."""
    return list(dict.fromkeys(data.table["smiles"]))[:size]


def cold_featurise_seconds(smiles: Sequence[str], n: int = 32) -> float:
    """Seconds per molecule to featurise ``n`` molecules on a featuriser with an empty cache."""
    fresh = MolFeaturizer()
    batch = list(smiles)[:n]
    t0 = time.perf_counter()
    fresh.featurize(batch)
    return (time.perf_counter() - t0) / len(batch)


def sweep_paths(ens: FusionEnsemble, record: dict, pool: Sequence[str],
                batch_sizes: Sequence[int] = BATCH_SIZES, repeat: int = 5) -> pd.DataFrame:
    """Latency of each prediction path at several batch sizes, with a warm featuriser.

    Featurisation is warmed first because it is measured on its own
    (:func:`cold_featurise_seconds`); what is timed here is everything after it.

    Args:
        ens: The ensemble to time.
        record: An experimental context.
        pool: Distinct SMILES to draw batches from.
        batch_sizes: Batch sizes; those larger than the pool are skipped.
        repeat: Timed repetitions per cell (median reported).

    Returns:
        One row per path and batch size.
    """
    pool = list(pool)
    ctx = ens.transform_context(record)
    ens.data.featurizer.featurize(pool)
    records = [{"smiles": s, **record} for s in pool]
    X_pool = ens.data.encode(records)
    out = []
    for n in [b for b in batch_sizes if b <= len(pool)]:
        batch, rows = pool[:n], records[:n]
        calls = {
            "context+std": lambda: ens.predict(batch, context=ctx, return_individual=False),
            "context": lambda: ens.predict(batch, context=ctx, return_individual=False,
                                           return_std=False),
            "records": lambda: ens.predict(rows, return_individual=False),
            "matrix": lambda: ens.predict_matrix(X_pool[:n], return_individual=False),
        }
        for name, call in calls.items():
            seconds = timeit(call, repeat=repeat)
            out.append({"path": name, "batch": n, "seconds": seconds,
                        "us_per_mol": seconds / n * 1e6, "mol_per_s": n / seconds})
    return pd.DataFrame(out)


class StageTimer:
    """Exclusive wall time per pipeline stage, by wrapping long-lived methods of one ensemble.

    A call nested inside another timed call is charged to its own stage and paused out of its
    parent's, so the stages add up to the time spent inside wrapped code without double
    counting. Wrappers are removed on exit.
    """

    def __init__(self, ens: FusionEnsemble):
        self.ens = ens
        self.seconds: Dict[str, float] = {}
        self._stack: list = []
        self._undo: list = []

    def _wrap(self, obj, name: str, stage: str) -> None:
        fn = getattr(obj, name, None)
        if fn is None:
            return

        def timed(*args, **kwargs):
            now = time.perf_counter()
            if self._stack:
                parent = self._stack[-1]
                self.seconds[parent[0]] = self.seconds.get(parent[0], 0.0) + now - parent[1]
            self._stack.append([stage, now])
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.perf_counter()
                top = self._stack.pop()
                self.seconds[stage] = self.seconds.get(stage, 0.0) + end - top[1]
                if self._stack:
                    self._stack[-1][1] = end

        self._undo.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, timed)

    def __enter__(self) -> "StageTimer":
        ens = self.ens
        self._wrap(ens.data, "encode", "encode")
        self._wrap(ens.data.featurizer, "featurize", "featurise")
        self._wrap(ens, "_aggregate", "aggregate")
        for member in ens.members:
            for name in ("transform", "transform_blocks", "transform_block"):
                self._wrap(member.pre_, name, "transform")
            self._wrap(member, "_predict_model", "model")
            model = getattr(member, "model_", None)
            if model is not None:
                for name in ("predict_in_context", "predict"):
                    self._wrap(model, name, "model")
        return self

    def __exit__(self, *exc) -> None:
        for obj, name, had, original in reversed(self._undo):
            if had:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def stage_breakdown(ens: FusionEnsemble, record: dict, smiles: Sequence[str],
                    repeat: int = 5) -> pd.DataFrame:
    """Where one batch's time goes, for the cached-context path (with and without std) and
    the records path. ``other`` is whatever the wrapped stages do not account for."""
    smiles = list(smiles)
    ctx = ens.transform_context(record)
    records = [{"smiles": s, **record} for s in smiles]
    ens.data.featurizer.featurize(smiles)
    calls = {
        "context": lambda: ens.predict(smiles, context=ctx, return_individual=False),
        "context_nostd": lambda: ens.predict(smiles, context=ctx, return_individual=False,
                                             return_std=False),
        "records": lambda: ens.predict(records, return_individual=False),
    }
    rows = []
    for path, call in calls.items():
        call()                                              # warm up outside the timer
        total, wall = {}, 0.0
        for _ in range(repeat):
            with StageTimer(ens) as timer:
                t0 = time.perf_counter()
                call()
                wall += time.perf_counter() - t0
            for stage, seconds in timer.seconds.items():
                total[stage] = total.get(stage, 0.0) + seconds
        wall /= repeat
        parts = {stage: seconds / repeat for stage, seconds in total.items()}
        parts["other"] = max(wall - sum(parts.values()), 0.0)
        norm = sum(parts.values())
        for stage, seconds in parts.items():
            rows.append({"path": path, "stage": stage, "seconds": seconds,
                         "share": seconds / norm, "batch": len(smiles)})
    return pd.DataFrame(rows)


def redundancy_costs(ens: FusionEnsemble, record: dict, smiles: Sequence[str],
                     repeat: int = 15) -> pd.DataFrame:
    """What each redundancy found in the inference path costs, measured in isolation.

    Each item replays just the redundant operation at the real shapes, so the number is what
    removing it can save at most.

    Args:
        ens: An ensemble of GP members.
        record: An experimental context.
        smiles: The batch.
        repeat: Timed repetitions per item.

    Returns:
        One row per item R1..R7.
    """
    smiles = list(smiles)
    n, members = len(smiles), len(ens.members)
    data, ctx = ens.data, ens.transform_context(record)
    fp, desc, ok = data.featurizer.featurize(smiles)
    valid = np.flatnonzero(ok)
    total = timeit(lambda: ens.predict(smiles, context=ctx, return_individual=False), repeat)
    rows = []

    def add(item: str, what: str, seconds: float) -> None:
        seconds = max(float(seconds), 0.0)
        rows.append({"item": item, "what": what, "seconds": seconds,
                     "share_of_predict": seconds / total, "batch": n, "members": members})

    def r1():
        wide = np.zeros((n, data.n_columns))
        wide[:, data.index["fingerprint"]] = fp
        wide[:, data.index["descriptors"]] = desc
        for _ in range(members):
            wide[valid]
    add("R1", "full-width float64 scratch row, plus one fancy-index copy per member",
        timeit(r1, repeat))

    pre = ens.members[0].pre_
    fp64, desc64 = fp.astype(np.float64), desc.astype(np.float64)

    def four_pass():
        for block, arr in (("fingerprint", fp64), ("descriptors", desc64)):
            imp, sc, width = pre.steps_[block]
            Z = imp.transform(arr)
            if sc is not None:
                Z = sc.transform(Z)
            np.ascontiguousarray(Z / width)

    def two_pass():
        for block, arr in (("fingerprint", fp), ("descriptors", desc)):
            Z = arr.astype(np.float32)
            Z /= np.float32(pre.steps_[block][2])
    lean = timeit(two_pass, repeat)
    add("R2", "imputer + scaler + divide + contiguous copy, against one copy and an in-place "
              "divide, per member", members * (timeit(four_pass, repeat) - lean))

    gp = getattr(ens.members[0], "model_", None)
    if hasattr(gp, "Z_train_"):
        def r3():
            with torch.no_grad():
                for _ in range(members):
                    train = gp.Z_train_["fingerprint"]
                    (train * train).sum(1)
                    scaled = gp.Z_train_["descriptors"] / gp._lengthscale("descriptors",
                                                                          gp.params_)
                    (scaled * scaled).sum(1)
        add("R3", "train-side row norms and the scaled ARD block, recomputed per batch "
                  "and member", timeit(r3, repeat))

        def matmul(dtype):
            def run():
                with torch.no_grad():
                    for width in (1024, 217):
                        a = torch.randn(n, width, dtype=dtype)
                        b = torch.randn(gp.n_train_, width, dtype=dtype)
                        a @ b.T
            return run
        add("R4", "the two molecular cross-kernel matmuls in float64 rather than float32",
            members * (timeit(matmul(torch.float64), repeat)
                       - timeit(matmul(torch.float32), repeat)))

        weight = torch.rand(gp.n_train_, dtype=torch.float64)

        def r5():
            for _ in range(members * 2):
                bool(torch.any(weight != 0))
        add("R5", "torch.any(weight != 0) per molecular block, member and batch",
            timeit(r5, repeat))
    else:
        for item, what in (("R3", "needs a GP member"), ("R4", "needs a GP member"),
                           ("R5", "needs a GP member")):
            add(item, what, 0.0)

    add("R6", "a second featurize() call just to recover ok",
        timeit(lambda: data.featurizer.featurize(smiles), repeat))
    add("R7", "the molecular transform repeated for members that share it",
        (members - 1) * lean)
    return pd.DataFrame(rows)


def heldout_score(ens: FusionEnsemble, data: FusionData, task: str, n_folds: int = 5) -> float:
    """R² (ROC-AUC for the binary task) on the held-out fold of split 0.

    The saved ensembles are fitted on the pool of that split only, so these rows are unseen.
    """
    _, X, y, _ = data.task_rows(task)
    _, holdout = data.splits(task, n_repeats=1, n_folds=n_folds)[0][0]
    mean = ens.predict_matrix(X[holdout], return_individual=False).mean
    if TASK_TYPES[task] == "binary":
        return float(roc_auc_score(y[holdout], mean))
    return float(r2_score(y[holdout], mean))


def capture_baseline(ensembles: Dict[str, FusionEnsemble], data: FusionData, out_dir,
                     *, label: str = "baseline", batch_sizes: Sequence[int] = BATCH_SIZES,
                     pool_size: int = 1024, breakdown_batch: int = 256, repeat: int = 5,
                     heldout_folds: int = 5) -> None:
    """Measure every ensemble and write ``<label>_*.csv`` plus its predictions.

    Args:
        ensembles: Mapping of task to ensemble.
        data: Development data (supplies the context and the SMILES pool).
        out_dir: Directory to write into (created if needed).
        label: Filename prefix (``baseline`` for the unchanged package, ``after`` later).
        batch_sizes: Batch sizes of the scaling sweep.
        pool_size: Number of distinct SMILES to draw batches from.
        breakdown_batch: Batch size of the stage breakdown and redundancy items.
        repeat: Timed repetitions.
        heldout_folds: Folds of the split whose held-out fold scores the ensemble.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = stamp()
    record = context_record(data)
    pool = smiles_pool(data, pool_size)
    batch = pool[:min(breakdown_batch, len(pool))]
    (out / f"{label}_context.json").write_text(json.dumps(record, indent=1))

    frames: Dict[str, list] = {k: [] for k in
                               ("scaling", "breakdown", "redundancy", "setup", "heldout")}
    arrays = {"smiles": np.array(pool[:256])}
    for task, ens in ensembles.items():
        def tag(frame):
            return frame.assign(task=task, **meta)
        frames["scaling"].append(tag(sweep_paths(ens, record, pool, batch_sizes, repeat)))
        frames["breakdown"].append(tag(stage_breakdown(ens, record, batch, repeat)))
        frames["redundancy"].append(tag(redundancy_costs(ens, record, batch)))
        gp = getattr(ens.members[0], "model_", None)
        frames["setup"].append(tag(pd.DataFrame([{
            "context_setup_s": timeit(lambda: ens.transform_context(record), repeat),
            "members": len(ens.members), "n_train": int(getattr(gp, "n_train_", 0))}])))
        frames["heldout"].append(tag(pd.DataFrame([{
            "score": heldout_score(ens, data, task, heldout_folds),
            "metric": "ROC-AUC" if TASK_TYPES[task] == "binary" else "R2"}])))
        pred = ens.predict(pool[:256], context=ens.transform_context(record))
        arrays[f"{task}_mean"], arrays[f"{task}_std"] = pred.mean, pred.std
        arrays[f"{task}_members"] = np.array(list(pred.member_predictions.values()))

    for name, parts in frames.items():
        pd.concat(parts, ignore_index=True).to_csv(out / f"{label}_{name}.csv", index=False)
    pd.DataFrame([{"featurise_cold_s_per_mol": cold_featurise_seconds(pool), **meta}]).to_csv(
        out / f"{label}_featurise.csv", index=False)
    np.savez(out / f"{label}_predictions.npz", **arrays)


def main(label: str = "baseline", out_dir: Optional[str] = None,
         tasks: Sequence[str] = ("dmax", "pdc50", "activity")) -> None:
    """Load the saved ensembles from ``ensembles/`` and capture them under ``label``."""
    data = FusionData.from_csv(DEV_FILES)
    ensembles = {t: FusionEnsemble.from_pretrained(REPO / "ensembles" / f"fusion_{t}", data=data)
                 for t in tasks}
    capture_baseline(ensembles, data, out_dir or RESULTS, label=label)
    print(f"wrote {label}_* to {out_dir or RESULTS} at {stamp()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest test/test_inference_bench.py -q 2>&1 | tail -8`
Expected: `6 passed`. If `test_stage_timer...` fails on `"transform" ` missing from `timer.seconds`, the context path is not calling `transform_blocks` on the member preprocessor — read `FusionEnsemble._predict_with_context` before changing the test.

- [ ] **Step 5: Commit the harness**

```bash
git add notebooks/ensemble_inference_bench.py test/test_inference_bench.py
git commit -m "test(fusion): benchmark harness for the ensemble inference path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Capture the baseline on the UNCHANGED package**

This must happen before Task 2 touches `tackai/`. Confirm first: `git diff --stat HEAD -- tackai/fusion/blocks.py tackai/fusion/gp.py tackai/fusion/ensemble.py tackai/fusion/models.py` prints nothing.

Run: `OMP_NUM_THREADS=1 .venv/bin/python notebooks/ensemble_inference_bench.py baseline 2>&1 | tail -5`
Expected: `wrote baseline_* to .../ensemble_inference_speed_results at {'commit': '<sha>', 'dirty': True}` (`dirty` is True because of the two pre-existing uncommitted edits named in Global Constraints; that is expected). Takes several minutes.

Then sanity-check the numbers:
Run: `.venv/bin/python -c "import pandas as pd; r='notebooks/ensemble_inference_speed_results/'; s=pd.read_csv(r+'baseline_scaling.csv'); print(s[(s.batch==256)].pivot(index='task',columns='path',values='us_per_mol').round(1)); print(pd.read_csv(r+'baseline_heldout.csv')[['task','metric','score']]); print(pd.read_csv(r+'baseline_redundancy.csv').query('task==\"dmax\"')[['item','seconds','share_of_predict']])"`
Expected: a µs/mol table with all four paths for the three tasks, held-out scores matching the published fusion notebook's (dmax and pdc50 R² clearly above 0, activity ROC-AUC above 0.5), and R1-R7 rows.

- [ ] **Step 7: Commit the baseline**

```bash
git add notebooks/ensemble_inference_speed_results/
git commit -m "docs(fusion): inference timings of the unchanged ensemble code

Captured before any speed change so the notebook can overlay before and after.
Marked dirty only by two unrelated uncommitted edits in data.py / features.py.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 2: Preprocessor — no standardisation, float32 apply, fused transform, consensus

**Files:**
- Modify: `tackai/fusion/blocks.py` (constants, `BlockPreprocessor`)
- Modify: `test/test_fusion_blocks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 4, 6, 7, 8):
  - `LEGACY_SCALE_BLOCKS = ("e3", "cell", "poi", "assay", "assay_time")`
  - `BlockPreprocessor(blocks=None, scale_blocks=None, dtype="float32")`; `scale_blocks=None` now means *nothing is standardised*
  - `BlockPreprocessor.transform_block(block: str, Xb: ndarray) -> ndarray` — one block from an array holding only that block's columns; always returns a fresh C-contiguous array of the preprocessor's dtype
  - `BlockPreprocessor.transform_blocks(X, only)` and `.transform(X)` — same signatures as before
  - `BlockPreprocessor.transform_signature(block) -> tuple` — equal for two preprocessors exactly when `transform_block` agrees on NaN-free input
  - `BlockPreprocessor.consensus(preprocessors, only=None) -> BlockPreprocessor` (classmethod) — raises `ValueError` naming the block on any disagreement of columns, width or scaler presence
  - `pre.dtype` (str); a preprocessor unpickled without the attribute behaves as `"float64"`

- [ ] **Step 1: Update the tests that encode the old behaviour**

In `test/test_fusion_blocks.py`, change the import to include `LEGACY_SCALE_BLOCKS`:

```python
from tackai.fusion.blocks import (BLOCK_DIMS, BLOCK_ORDER, DENSE_BLOCKS, LEGACY_SCALE_BLOCKS,
                                  MOL_BLOCKS, SMALL_BLOCKS, BlockPreprocessor, block_index)
```

Replace `test_context_blocks_are_standardised` with the two tests below:

```python
def test_no_block_is_standardised_by_default():
    """The context arrives PCA-reduced and the planned tree members ignore scaling."""
    X, idx = make_X()
    Z = BlockPreprocessor(dtype="float64").fit(X).transform(X)
    for block in ("e3", "cell", "poi", "assay"):
        assert np.array_equal(Z[block], X[:, idx[block]] / np.sqrt(BLOCK_DIMS[block])), block


def test_scale_blocks_restores_the_legacy_standardisation():
    X, _ = make_X()
    Z = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X).transform(X)
    assert np.allclose(Z["cell"].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z["cell"].std(axis=0) * np.sqrt(47), 1, atol=1e-4)
```

In `test_constant_context_column_does_not_produce_inf` and `test_fitted_on_training_rows_only`, build the preprocessor with `BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64")` (every `BlockPreprocessor()` inside those two tests) so they still exercise a *fitted* statistic. Keep every assertion as is. Add one assertion to the constant-column test's end: `assert np.isfinite(BlockPreprocessor().fit(X).transform(X)["cell"]).all()`.

Replace `test_assay_time_is_median_imputed_and_standardised` with:

```python
def test_assay_time_is_median_imputed_and_not_standardised():
    X, idx = make_X()
    Z = BlockPreprocessor(dtype="float64").fit(X).transform(X)
    assert np.isfinite(Z["assay_time"]).all()
    observed = X[:, idx["assay_time"]][~np.isnan(X[:, idx["assay_time"]])]
    assert set(np.unique(Z["assay_time"])) <= {12.0, 24.0, float(np.median(observed))}
```

Append the new tests:

```python
def four_pass_reference(pre, X, only):
    """The transform as it was before the fused pass: imputer, scaler, divide, copy."""
    X = np.asarray(X, dtype=np.float64)
    out = {}
    for b in only:
        imp, sc, width = pre.steps_[b]
        Z = imp.transform(X[:, pre.blocks_[b]])
        if sc is not None:
            Z = sc.transform(Z)
        out[b] = np.ascontiguousarray(Z / width)
    return out


def make_X_with_gaps():
    X, idx = make_X()
    X[3, idx["poi"][2]] = np.nan
    X[5, idx["cell"][0]] = np.nan
    X[7, idx["e3"][1]] = np.nan
    X[9, idx["descriptors"][4]] = np.nan
    return X, idx


@pytest.mark.parametrize("scale_blocks", [None, LEGACY_SCALE_BLOCKS])
def test_fused_transform_is_bit_identical_to_the_four_pass_reference(scale_blocks):
    """Removing the redundant passes must not move a single bit (at float64)."""
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=scale_blocks, dtype="float64").fit(X)
    Z, ref = pre.transform(X), four_pass_reference(pre, X, pre.blocks_)
    for block in ref:
        assert np.array_equal(Z[block], ref[block]), block


def test_float32_output_tracks_the_float64_reference():
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor().fit(X)
    Z, ref = pre.transform(X), four_pass_reference(pre, X, pre.blocks_)
    for block in ref:
        assert Z[block].dtype == np.float32 and Z[block].flags.c_contiguous, block
        assert np.allclose(Z[block], ref[block], rtol=1e-6, atol=1e-30), block


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_transform_never_writes_into_the_callers_array(dtype):  # Review Focus 4
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype=dtype).fit(X)
    for arr in (X.copy(), X.astype(np.float32)):
        before = arr.copy()
        pre.transform(arr)
        pre.transform_blocks(arr, only=["cell", "fingerprint"])
        assert np.array_equal(arr, before, equal_nan=True)


def test_statistics_are_fitted_in_float64():
    X, idx = make_X()
    col = idx["descriptors"][0]                       # the Ipc-scale column, ~1e18
    X[0, col] = np.nan
    pre = BlockPreprocessor().fit(X.astype(np.float32))
    stats = pre.steps_["descriptors"][0].statistics_
    assert stats.dtype == np.float64
    assert stats[0] == pytest.approx(np.nanmean(X.astype(np.float32).astype(np.float64)[:, col]))


def test_a_preprocessor_pickled_before_dtype_existed_behaves_as_float64():  # Review Focus 1
    X, _ = make_X()
    old = BlockPreprocessor().fit(X)
    del old.dtype                                      # an artifact written by the old code
    assert old.transform(X)["cell"].dtype == np.float64


def test_non_contiguous_blocks_transform_identically():
    X, idx = make_X_with_gaps()
    reversed_layout = {b: idx[b][::-1].copy() for b in idx}
    straight = BlockPreprocessor(dtype="float64").fit(X)
    flipped = BlockPreprocessor(blocks=reversed_layout, dtype="float64").fit(X)
    assert isinstance(straight._columns("cell"), slice)
    assert isinstance(flipped._columns("cell"), np.ndarray)
    Zs, Zf = straight.transform(X), flipped.transform(X)
    for block in Zs:
        assert np.array_equal(Zs[block][:, ::-1], Zf[block]), block


def two_halves(**kw):
    X, _ = make_X(n=60)
    return X, BlockPreprocessor(**kw).fit(X[:30]), BlockPreprocessor(**kw).fit(X[30:])


def test_consensus_averages_the_fitted_statistics():
    _, a, b = two_halves(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64")
    merged = BlockPreprocessor.consensus([a, b], only=["cell", "assay_time"])
    for block in ("cell", "assay_time"):
        (ia, sa, _), (ib, sb, _) = a.steps_[block], b.steps_[block]
        imp, sc, _ = merged.steps_[block]
        assert np.allclose(imp.statistics_, (ia.statistics_ + ib.statistics_) / 2)
        assert np.allclose(sc.mean_, (sa.mean_ + sb.mean_) / 2)
        assert np.allclose(sc.scale_, (sa.scale_ + sb.scale_) / 2)
    assert not np.allclose(a.steps_["cell"][1].mean_, merged.steps_["cell"][1].mean_)


def test_consensus_does_not_depend_on_member_order():
    X, a, b = two_halves(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64")
    ab = BlockPreprocessor.consensus([a, b]).transform(X)
    ba = BlockPreprocessor.consensus([b, a]).transform(X)
    for block in ab:
        assert np.array_equal(ab[block], ba[block]), block


def test_consensus_of_a_single_preprocessor_is_that_preprocessor():
    X, _ = make_X_with_gaps()
    pre = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS, dtype="float64").fit(X)
    Z, Zc = pre.transform(X), BlockPreprocessor.consensus([pre]).transform(X)
    for block in Z:
        assert np.array_equal(Z[block], Zc[block]), block


def test_consensus_rejects_members_that_disagree_on_scaling():  # Review Focus 5
    X, _ = make_X(n=60)
    scaled = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X)
    plain = BlockPreprocessor().fit(X)
    with pytest.raises(ValueError, match="cell"):
        BlockPreprocessor.consensus([scaled, plain], only=["cell"])


def test_consensus_rejects_members_with_different_columns():  # Review Focus 5
    X, idx = make_X(n=60)
    shuffled = {b: idx[b][::-1].copy() for b in idx}
    with pytest.raises(ValueError, match="poi"):
        BlockPreprocessor.consensus([BlockPreprocessor().fit(X),
                                     BlockPreprocessor(blocks=shuffled).fit(X)], only=["poi"])


def test_transform_signature_ignores_the_imputer_but_not_the_scaler():
    _, a, b = two_halves()
    assert a.steps_["descriptors"][0].statistics_.tolist() != \
        b.steps_["descriptors"][0].statistics_.tolist()
    for block in ("fingerprint", "descriptors"):
        assert a.transform_signature(block) == b.transform_signature(block)
    _, sa, sb = two_halves(scale_blocks=("descriptors",))
    assert sa.transform_signature("descriptors") != sb.transform_signature("descriptors")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest test/test_fusion_blocks.py -q 2>&1 | tail -15`
Expected: collection fails with `ImportError: cannot import name 'LEGACY_SCALE_BLOCKS'`.

- [ ] **Step 3: Implement**

In `tackai/fusion/blocks.py`: add `import copy` at the top; add the constant after `MOL_BLOCKS`:

```python
#: What the first version of the pipeline standardised. Kept so the superseded behaviour stays
#: reachable (and testable) through ``scale_blocks``; nothing uses it by default.
LEGACY_SCALE_BLOCKS = ("e3", "cell", "poi", "assay", "assay_time")
```

Update the module docstring's last sentence to: `"...and the molecular blocks stay raw, so trees split on individual Morgan bits and the GP sees untransformed features. No block is standardised either: the context is already centred by its PCA."`

Replace the whole `BlockPreprocessor` class with:

```python
class BlockPreprocessor(BaseEstimator):
    """Per-block imputation and ``1/sqrt(width)`` scaling, fitted on the rows given to :meth:`fit`.

    Nothing is standardised by default. The context blocks arrive PCA-reduced, hence already
    centred; the molecular blocks must stay raw so a tree can split on an individual Morgan
    bit and a GP kernel sees untransformed descriptors; and ``assay_time`` feeds a linear
    kernel that learns its own scale. Dense blocks are mean-imputed and divided by
    ``sqrt(width)`` so no block dominates a shared-lengthscale kernel by width alone;
    ``assay_time`` is median-imputed. Imputation statistics are fitted in float64 (a raw
    ``Ipc`` near 1e18 would lose its mean in float32) and applied in ``dtype``.

    Args:
        blocks: Mapping of block name to column indices (default: contiguous
            :data:`BLOCK_DIMS` layout).
        scale_blocks: Blocks to standardise (default: none). :data:`LEGACY_SCALE_BLOCKS` is
            the set the first version used.
        dtype: Floating-point type of :meth:`transform`'s output (``"float32"`` or
            ``"float64"``). A preprocessor unpickled without this attribute behaves as
            ``"float64"``, which is what it was fitted under.
    """

    def __init__(self, blocks: Optional[Dict[str, np.ndarray]] = None,
                 scale_blocks: Optional[Sequence[str]] = None, dtype: str = "float32"):
        self.blocks = blocks
        self.scale_blocks = scale_blocks
        self.dtype = dtype

    def _dtype(self) -> np.dtype:
        """The output dtype; float64 for an object pickled before ``dtype`` existed."""
        return np.dtype(getattr(self, "dtype", "float64"))

    def fit(self, X, y=None) -> "BlockPreprocessor":
        """Fit the per-block imputers (and any requested scalers) on ``X``.

        Args:
            X: Design matrix of shape ``(n_rows, n_columns)``.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        self.blocks_ = {b: np.asarray(i) for b, i in
                        (self.blocks or block_index(BLOCK_DIMS)).items()}
        scaled = self._selected_scale_blocks()
        self.steps_, self.dims_ = {}, {}
        for b, idx in self.blocks_.items():
            Xb = X[:, idx]
            dense = b in DENSE_BLOCKS
            imp = SimpleImputer(strategy="mean" if dense else "median",
                                keep_empty_features=True).fit(Xb)
            sc = self._fit_scaler(imp.transform(Xb)) if b in scaled else None
            self.steps_[b] = (imp, sc, np.sqrt(Xb.shape[1]) if dense else 1.0)
            self.dims_[b] = Xb.shape[1]
        return self

    @staticmethod
    def _fit_scaler(Z) -> StandardScaler:
        """Standardiser whose zero-variance columns map to 0.0 instead of +/-inf."""
        sc = StandardScaler().fit(Z)
        sc.scale_ = np.where(sc.scale_ == 0, 1.0, sc.scale_)
        return sc

    def _selected_scale_blocks(self) -> set:
        """The blocks to standardise; unknown names are an error."""
        chosen = set(() if self.scale_blocks is None else self.scale_blocks)
        unknown = chosen - set(DENSE_BLOCKS) - set(SMALL_BLOCKS)
        if unknown:
            raise ValueError(f"scale_blocks must be among {DENSE_BLOCKS + SMALL_BLOCKS}, "
                             f"got {sorted(unknown)}")
        return chosen

    def _columns(self, block: str):
        """Where a block lives in the design matrix: a slice (a view) when it is a run."""
        idx = self.blocks_[block]
        if len(idx) and np.all(np.diff(idx) == 1):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return idx

    def transform_block(self, block: str, Xb) -> np.ndarray:
        """Process one block from an array holding only that block's columns.

        One copy, then in-place imputation and division. The copy is never skipped: the
        result must not alias the caller's rows, and it must be C-contiguous because it goes
        straight into BLAS matmuls, which are not bit-reproducible across memory layouts.

        Args:
            block: Block name.
            Xb: Array of shape ``(n_rows, width)``.

        Returns:
            Fresh C-contiguous array of this preprocessor's dtype.
        """
        imp, sc, width = self.steps_[block]
        dtype = self._dtype()
        Z = np.array(Xb, dtype=dtype, order="C")
        missing = np.isnan(Z)
        if missing.any():
            rows, cols = np.nonzero(missing)
            Z[rows, cols] = imp.statistics_.astype(dtype)[cols]
        if sc is not None:
            Z -= sc.mean_.astype(dtype)
            Z /= sc.scale_.astype(dtype)
        Z /= dtype.type(width)
        return Z

    def transform(self, X) -> Dict[str, np.ndarray]:
        """Processed blocks of ``X``.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.

        Returns:
            Mapping of block name to its processed array.
        """
        return self.transform_blocks(X, only=self.blocks_)

    def transform_blocks(self, X, only: Sequence[str]) -> Dict[str, np.ndarray]:
        """Processed values of a subset of the blocks.

        Args:
            X: Design matrix with the columns this preprocessor was fitted on.
            only: Names of the blocks to transform.

        Returns:
            Mapping of the requested block names to their processed arrays.
        """
        X = np.asarray(X)
        return {b: self.transform_block(b, X[:, self._columns(b)]) for b in only}

    def transform_signature(self, block: str) -> tuple:
        """Everything that makes :meth:`transform_block` member-specific on NaN-free input.

        The imputer is left out on purpose: it only acts on NaN. Two preprocessors with equal
        signatures produce identical output for a NaN-free block, so the ensemble can
        transform it once for both.

        Args:
            block: Block name.

        Returns:
            A hashable tuple.
        """
        _, sc, width = self.steps_[block]
        scaler = None if sc is None else (sc.mean_.tobytes(), sc.scale_.tobytes())
        return (str(self._dtype()), float(width), scaler)

    @classmethod
    def consensus(cls, preprocessors: Sequence["BlockPreprocessor"],
                  only: Optional[Sequence[str]] = None) -> "BlockPreprocessor":
        """One preprocessor whose statistics are the mean of several fitted ones.

        Members fitted on different folds disagree about the statistics of the *same* context.
        The consensus gives every member one answer, independent of which rows it was fitted
        on. Blocks outside ``only`` keep the first preprocessor's steps and should not be used.

        Args:
            preprocessors: Fitted preprocessors, all with the same layout.
            only: Blocks to average (default: all).

        Returns:
            A new fitted preprocessor; the inputs are not modified.

        Raises:
            ValueError: If a block's columns, width or scaler presence differ.
        """
        first = preprocessors[0]
        merged = copy.copy(first)
        merged.steps_ = dict(first.steps_)
        for block in (list(only) if only is not None else list(first.blocks_)):
            imp0, sc0, width0 = first.steps_[block]
            for pre in preprocessors[1:]:
                imp, sc, width = pre.steps_[block]
                if not np.array_equal(pre.blocks_[block], first.blocks_[block]):
                    raise ValueError(f"block {block!r}: members read different columns of the "
                                     "design matrix, so there is no common context to share")
                if width != width0 or (sc is None) != (sc0 is None):
                    raise ValueError(f"block {block!r}: members disagree on its width or on "
                                     "whether it is standardised; refit them with one setting")
            imp_c = copy.deepcopy(imp0)
            imp_c.statistics_ = np.mean([p.steps_[block][0].statistics_
                                         for p in preprocessors], axis=0)
            sc_c = None
            if sc0 is not None:
                sc_c = copy.deepcopy(sc0)
                sc_c.mean_ = np.mean([p.steps_[block][1].mean_ for p in preprocessors], axis=0)
                sc_c.scale_ = np.mean([p.steps_[block][1].scale_ for p in preprocessors],
                                      axis=0)
                sc_c.var_ = sc_c.scale_ ** 2
            merged.steps_[block] = (imp_c, sc_c, width0)
        return merged

    def concat(self, Z: Dict[str, np.ndarray], names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatenate processed blocks in :data:`BLOCK_ORDER`.

        Args:
            Z: Mapping of block name to processed array.
            names: Blocks to concatenate (default: those present in ``Z``, in block order).

        Returns:
            One array of shape ``(n_rows, sum of widths)``.
        """
        names = list(names or [b for b in BLOCK_ORDER if b in Z])
        return np.concatenate([Z[b] for b in names], axis=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest test/test_fusion_blocks.py -q 2>&1 | tail -15`
Expected: all pass. If `test_fused_transform_is_bit_identical...` fails by one ulp for the scaled variant, the cause is operand order in the standardisation: sklearn does `X -= mean_` then `X /= scale_`; keep exactly that order (do not fold them into one multiply).

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/blocks.py test/test_fusion_blocks.py
git commit -m "perf(fusion): no standardisation, float32 apply, fused block transform, consensus

The context blocks arrive PCA-reduced and the tree members ignore scaling, so no
block is standardised by default (LEGACY_SCALE_BLOCKS restores the old set).
Statistics are fitted in float64 and applied in float32. transform_block does one
copy plus in-place work instead of four passes and is bit-identical to the old
transform at float64. consensus() averages members' fitted statistics.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Note for the executor: other fusion test files may be red until Task 4; that is expected. This task's gate is `test/test_fusion_blocks.py` only.

---

### Task 3: GP — float32 default, dtype-scaled jitter, float64 promotion, `astype`

**Files:**
- Modify: `tackai/fusion/gp.py`
- Modify: `test/test_fusion_gp.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Tasks 4, 5, 8):
  - `AdditiveProductGP(dims, *, ..., jitter: Optional[float] = None, dtype=torch.float32)`
  - `gp._factor(K) -> (L | None, promoted: bool)`; `gp._cholesky(K) -> L` (raises `RuntimeError` matching `"positive definite"`)
  - after `load_state`: `gp.promoted_: bool`, `gp.chol_` (float64 when promoted), `gp.alpha_` (always `gp.dtype`)
  - lifetime counters `gp.factorisations_`, `gp.promotions_`
  - `gp.astype(dtype, jitter=None) -> gp` — re-conditions the fitted GP in another precision, keeping its hyper-parameters

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fusion_gp.py`:

```python
import torch


def test_default_dtype_is_float32():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert gp.dtype is torch.float32
    assert gp.Z_train_["fingerprint"].dtype == torch.float32
    assert gp.alpha_.dtype == torch.float32
    assert gp.predict(Z).dtype == np.float32


def test_float64_is_still_available():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z), dtype="float64")
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert gp.dtype is torch.float64 and gp.predict(Z).dtype == np.float64
    assert gp.promoted_ is False


def test_jitter_ladder_depends_on_the_dtype():
    dims = {"fingerprint": 2, "descriptors": 2, "assay_time": 1}
    assert AdditiveProductGP(dims)._base_jitter() == 1e-5
    assert AdditiveProductGP(dims, dtype="float64")._base_jitter() == 1e-6
    assert AdditiveProductGP(dims, jitter=3e-7)._base_jitter() == 3e-7


def test_float32_factorisation_is_promoted_to_float64_when_it_must(monkeypatch):
    """Control flow of the fallback, made deterministic: float32 factorisations are refused."""
    real = torch.linalg.cholesky

    def float32_cannot(A, *args, **kwargs):
        if A.dtype == torch.float32:
            raise torch.linalg.LinAlgError("forced for the test")
        return real(A, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, "cholesky", float32_cannot)

    gp = AdditiveProductGP({"fingerprint": 2, "descriptors": 2, "assay_time": 1})
    L, promoted = gp._factor(torch.eye(10) * 2.0)
    assert promoted is True and L.dtype == torch.float64
    assert gp.promotions_ == 1 and gp.factorisations_ == 1


def test_a_kernel_that_float64_cannot_factorise_either_is_refused():
    gp = AdditiveProductGP({"fingerprint": 2, "descriptors": 2, "assay_time": 1})
    indefinite = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    L, promoted = gp._factor(indefinite)
    assert L is None and promoted is False
    with pytest.raises(RuntimeError, match="positive definite"):
        gp._cholesky(indefinite)


def test_a_well_conditioned_float32_kernel_is_not_promoted():
    gp = AdditiveProductGP({"fingerprint": 2, "descriptors": 2, "assay_time": 1})
    L, promoted = gp._factor(torch.eye(10) * 2.0)
    assert promoted is False and L.dtype == torch.float32


def test_prediction_works_with_a_promoted_float64_factor():
    Z, y = toy()
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    mean, std = gp.predict(Z, return_std=True)
    gp.chol_, gp.promoted_ = gp.chol_.double(), True
    mean2, std2 = gp.predict(Z, return_std=True)
    assert np.allclose(mean, mean2, atol=1e-6) and np.allclose(std, std2, atol=1e-4)


def test_astype_reconditions_in_the_new_precision():
    Z, y = toy(n=80)
    gp = AdditiveProductGP(dims_of(Z), dtype="float64")
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    mean64, std64 = gp.predict(Z, return_std=True)
    assert gp.astype("float32") is gp
    assert gp.dtype is torch.float32 and gp.Z_train_["poi"].dtype == torch.float32
    assert gp.params_["mean"].dtype == torch.float32
    mean32, std32 = gp.predict(Z, return_std=True)
    assert np.allclose(mean32, mean64, rtol=1e-2, atol=1e-2)
    assert np.allclose(std32, std64, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("scale", [1e9, 1e18, 1e20])
def test_float32_survives_descriptor_columns_up_to_the_featuriser_limit(scale):  # Review Focus 3
    """Ipc-like columns reach 1e20; squared they approach float32's 3.4e38 ceiling."""
    Z, y = toy(n=60)
    Z["descriptors"] = Z["descriptors"] * scale
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    mean, std = gp.predict(Z, return_std=True)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    K = gp.kernel_matrix(Z, Z)
    assert np.isfinite(K).all()
    eig = np.linalg.eigvalsh(0.5 * (K + K.T).astype(np.float64))
    assert eig.min() > -1e-4 * eig.max(), "kernel is indefinite"


def test_float32_survives_near_duplicate_molecules():  # Review Focus 3
    Z, y = toy(n=60)
    for b in Z:
        Z[b][1] = Z[b][0] + 1e-7
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=10, seed=0)
    assert np.isfinite(gp.predict(Z)).all()
```

Pin the old float64 tolerances (do not loosen them): in `test/test_fusion_gp.py` construct with `dtype="float64"` in `test_cached_distances_equal_a_naive_recomputation`, `test_fit_is_reproducible_for_a_seed`, `test_rbf_self_covariance_is_one_at_any_column_scale`, `test_folded_context_matches_the_full_kernel`, `test_folded_context_matches_the_full_kernel_for_any_interactions` (every `AdditiveProductGP(...)` call and the `fresh` one inside the first). Then add float32 siblings for the two folded-context tests at `rtol=1e-3, atol=1e-3` on the mean and `atol=2e-3` on the std, and for the self-covariance test at `atol=1e-4`:

```python
def test_folded_context_matches_the_full_kernel_in_float32():
    Z, y = toy(n=80)
    Z["e3"] = np.random.default_rng(1).normal(size=(80, 3))
    gp = AdditiveProductGP(dims_of(Z))
    gp.fit(Z, y, n_restarts=1, n_iter=20, seed=0)
    context = {b: Z[b][:1] for b in ("e3", "poi", "cell", "assay_time")}
    mols = {b: Z[b][:12] for b in ("fingerprint", "descriptors")}
    fold = gp.fold_context(context)
    folded_mean, folded_std = gp.predict_in_context(mols, fold, return_std=True)
    full = {**{b: np.repeat(context[b], 12, axis=0) for b in context}, **mols}
    mean, std = gp.predict(full, return_std=True)
    assert np.allclose(folded_mean, mean, rtol=1e-3, atol=1e-3)
    assert np.allclose(folded_std, std, atol=2e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest test/test_fusion_gp.py -q 2>&1 | tail -15`
Expected: the new tests fail (`gp.dtype is torch.float32` False; `_base_jitter` / `_factor` / `astype` missing) and the pinned ones error on `dtype="float64"` only if the argument is rejected — it is accepted today, so those still pass.

- [ ] **Step 3: Implement**

In `tackai/fusion/gp.py`:

(a) After `NOISE_FLOOR`, add:

```python
#: Where each precision's jitter ladder starts and the largest value it tries before giving up
#: (float32 then promotes to float64, whose ladder is the original). float32's epsilon is 1.2e-7,
#: so the float64 starting jitter of 1e-6 would sit below the rounding noise of a unit-scale kernel.
JITTER_START = {torch.float32: 1e-5, torch.float64: 1e-6}
JITTER_CEILING = {torch.float32: 1e-3, torch.float64: 1e-2}


def _ladder(start: float, ceiling: float):
    """Jitter values ``start, 10*start, ...`` up to ``ceiling``; ``0 * 10`` is 0, so restart at 1e-8."""
    jitter = start
    while jitter <= ceiling * (1 + 1e-9):
        yield jitter
        jitter = max(jitter * 10, 1e-8)


def _try_cholesky(K: torch.Tensor, start: float, ceiling: float):
    """Cholesky factor of ``K + jitter*I`` for the first jitter that works, else None."""
    eye = torch.eye(K.shape[0], dtype=K.dtype)
    for jitter in _ladder(start, ceiling):
        try:
            return torch.linalg.cholesky(K + eye * jitter)
        except Exception:
            continue
    return None
```

(b) In `__init__`: change the signature to `jitter: Optional[float] = None, dtype: Union[str, torch.dtype] = torch.float32`; update the Args docs (`jitter`: "Starting diagonal jitter (default: 1e-5 in float32, 1e-6 in float64)"; `dtype`: "Torch dtype, float32 by default; a factorisation float32 cannot do is retried in float64"); after `self.dtype = ...` add:

```python
        self.promoted_ = False
        self.promotions_ = 0
        self.factorisations_ = 0
```

and add the method:

```python
    def _base_jitter(self, dtype: Optional[torch.dtype] = None) -> float:
        """Starting jitter: the user's, else the precision's default."""
        return self.jitter if self.jitter is not None else JITTER_START[dtype or self.dtype]
```

(c) Replace `_cholesky` with:

```python
    def _factor(self, K: torch.Tensor):
        """Cholesky factor with escalating jitter, promoting float32 to float64 if it must.

        Args:
            K: Noisy kernel matrix.

        Returns:
            ``(L, promoted)``; ``L`` is None when even float64 cannot factorise ``K``.
        """
        self.factorisations_ = getattr(self, "factorisations_", 0) + 1
        L = _try_cholesky(K, self._base_jitter(K.dtype), JITTER_CEILING[K.dtype])
        if L is not None or K.dtype == torch.float64:
            return L, False
        L = _try_cholesky(K.to(torch.float64), self._base_jitter(torch.float64),
                          JITTER_CEILING[torch.float64])
        if L is not None:
            self.promotions_ = getattr(self, "promotions_", 0) + 1
        return L, L is not None

    def _cholesky(self, K: torch.Tensor) -> torch.Tensor:
        """Cholesky factor of the noisy kernel; raises if even float64 cannot factorise it."""
        L, promoted = self._factor(K)
        if L is None:
            raise RuntimeError("GP kernel is not positive definite even with "
                               f"{JITTER_CEILING[torch.float64]:g} jitter in float64")
        self.promoted_ = promoted
        return L
```

(d) In `_neg_log_mll`, replace everything after `K = ...` with:

```python
        L, _ = self._factor(K)
        if L is None:
            return torch.tensor(float("inf"), dtype=self.dtype)
        resid = (y - params["mean"]).unsqueeze(1).to(L.dtype)
        alpha = torch.cholesky_solve(resid, L)
        return (0.5 * (resid * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * np.log(2.0 * np.pi))
```

(e) In `load_state`, replace the last three lines of the `with torch.no_grad():` block with:

```python
            self.chol_ = self._cholesky(K)
            resid = (self.y_train_ - self.params_["mean"]).unsqueeze(1).to(self.chol_.dtype)
            self.alpha_ = torch.cholesky_solve(resid, self.chol_).to(self.dtype)
```

(f) In `predict` and `predict_in_context`, change the triangular solve to `v = torch.linalg.solve_triangular(self.chol_, cross.T.to(self.chol_.dtype), upper=False)`.

(g) Add after `load_state`:

```python
    def astype(self, dtype: Union[str, torch.dtype], jitter: Optional[float] = None):
        """Re-condition this fitted GP in another precision, keeping its hyper-parameters.

        The kernel is rebuilt and refactorised in ``dtype`` (with float64 promotion if that
        precision cannot do it), so ``promoted_`` reports whether the saved hyper-parameters
        survive the cast.

        Args:
            dtype: Target torch dtype (or its name).
            jitter: Starting jitter (default: the new precision's own).

        Returns:
            self
        """
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        state = self.state_
        cast = {**state,
                "params": {k: v.to(dtype) for k, v in state["params"].items()},
                "ls_scale": {k: v.to(dtype) for k, v in state.get("ls_scale", {}).items()}}
        Z = {b: a.double().numpy() for b, a in self.Z_train_.items()}
        y = self.y_train_.double().numpy()
        self.dtype, self.jitter = dtype, jitter
        self.load_state(cast, Z, y)
        return self
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest test/test_fusion_gp.py -q 2>&1 | tail -15`
Expected: all pass. Known candidates if one fails: `test_noise_has_a_floor` (float32 rounding of `1e-3 + exp(.)`: compare with `>= 1e-3 - 1e-9`, a rounding allowance, not a loosening of the floor), and `test_variance_matches_the_textbook_formula` (rtol 1e-5: pin `dtype="float64"` and add a float32 sibling at `rtol=1e-3`).

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/gp.py test/test_fusion_gp.py
git commit -m "feat(fusion): float32 GP with float64 promotion and astype

float32 is the default in the fit and the Cholesky. A factorisation float32 cannot
do is retried in float64 and recorded (promoted_, promotions_). The jitter ladder
scales with the precision. astype() re-conditions a fitted GP in another precision
so artifacts saved in float64 can be served in float32.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Note for the executor: ensemble and model tests may be red until Task 4. This task's gate is `test/test_fusion_gp.py`.

---

### Task 4: Estimators carry the dtype; the whole fusion suite is green again

**Files:**
- Modify: `tackai/fusion/models.py`
- Create: `test/test_fusion_dtype.py`
- Modify: `test/test_fusion_ensemble.py`, and any other fusion test the full run shows red for a reason covered by the rule in step 5

**Interfaces:**
- Consumes: `BlockPreprocessor(dtype=...)` (Task 2), `AdditiveProductGP(dtype=...)` (Task 3).
- Produces (used by Tasks 6-9): `FusionEstimator(..., dtype="float32")`, `GPInteraction(..., dtype="float32")`, `XGBoostFusion(..., dtype="float32")`; `est.dtype` is passed to the preprocessor and the GP; a fitted member exposes `est.pre_.dtype` and `est.model_.dtype`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_dtype.py`:

```python
"""One dtype knob on the estimators reaches the preprocessor and the GP."""
import numpy as np
import pytest
import torch

from tackai.fusion.blocks import BLOCK_DIMS, block_index
from tackai.fusion.models import GPInteraction, XGBoostFusion


def design(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = block_index(BLOCK_DIMS)
    X = np.zeros((n, sum(BLOCK_DIMS.values())))
    X[:, idx["fingerprint"]] = rng.integers(0, 2, (n, 1024))
    X[:, idx["descriptors"]] = rng.normal(size=(n, 217))
    for b in ("e3", "cell", "poi", "assay"):
        X[:, idx[b]] = rng.normal(size=(n, BLOCK_DIMS[b]))
    X[:, idx["assay_time"]] = rng.choice([12.0, 24.0], (n, 1))
    y = X[:, idx["poi"][0]] + 0.1 * rng.normal(size=n)
    return X, y


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def test_estimators_default_to_float32():
    assert GPInteraction().dtype == "float32" and XGBoostFusion().dtype == "float32"


def test_gp_member_is_float32_end_to_end():
    X, y = design()
    est = fast_gp().fit(X, y)
    assert est.pre_.dtype == "float32" and est.model_.dtype is torch.float32
    assert est.predict(X).dtype in (np.float32, np.float64) and np.isfinite(est.predict(X)).all()


def test_gp_member_can_be_float64_end_to_end():
    X, y = design()
    est = fast_gp(dtype="float64").fit(X, y)
    assert est.pre_.dtype == "float64" and est.model_.dtype is torch.float64


def test_xgboost_member_preprocesses_in_the_requested_dtype():
    X, y = design()
    est = XGBoostFusion(n_estimators=10, grid=[{"max_depth": 3, "reg_lambda": 5.0}]).fit(X, y)
    assert est.pre_.dtype == "float32"
    assert est.pre_.transform(X)["fingerprint"].dtype == np.float32
    assert np.isfinite(est.predict(X)).all()


def test_dtype_survives_sklearn_clone():
    from sklearn.base import clone
    assert clone(fast_gp(dtype="float64")).dtype == "float64"
```

In `test/test_fusion_ensemble.py`, change `test_context_path_agrees_with_the_ordinary_path` to build with `fast_gp(dtype="float64")` (keep its 1e-10 assertions), and add its float32 sibling after it:

```python
def test_context_path_agrees_with_the_ordinary_path_in_float32(data):
    ens = FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
    ctx = ens.transform_context(CTX)
    fast = ens.predict(SMILES[:4], context=ctx)
    slow = ens.predict([{"smiles": s, **CTX} for s in SMILES[:4]])
    assert np.allclose(fast.mean, slow.mean, rtol=1e-3, atol=1e-3)
    assert np.allclose(fast.std, slow.std, atol=2e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_fusion_dtype.py -q 2>&1 | tail -10`
Expected: failures — `GPInteraction().dtype` raises `AttributeError`.

- [ ] **Step 3: Implement**

In `tackai/fusion/models.py`:
- `FusionEstimator.__init__(self, task_type="regression", blocks=None, random_state=0, dtype: str = "float32")` storing `self.dtype = dtype`; add to the class docstring `dtype: Floating-point type of the processed blocks and, for a GP, of the model.`
- `_make_preprocessor` returns `BlockPreprocessor(blocks=self.blocks, dtype=self.dtype)`.
- `GPInteraction.__init__`: append `dtype: str = "float32"` as the last parameter, pass `dtype=dtype` to `super().__init__`, document it; in `_fit_model` add `dtype=self.dtype` to the `AdditiveProductGP(...)` call.
- `XGBoostFusion.__init__`: append `dtype: str = "float32"`, pass to `super().__init__`, document it.

- [ ] **Step 4: Run to verify the new tests pass**

Run: `.venv/bin/python -m pytest test/test_fusion_dtype.py test/test_fusion_ensemble.py -q 2>&1 | tail -15`
Expected: `test_fusion_dtype.py` passes; list any failures in `test_fusion_ensemble.py` for step 5.

- [ ] **Step 5: Restore the whole fusion suite**

Run: `.venv/bin/python -m pytest test/test_fusion_blocks.py test/test_fusion_context.py test/test_fusion_data.py test/test_fusion_dtype.py test/test_fusion_ensemble.py test/test_fusion_features.py test/test_fusion_gp.py test/test_fusion_models.py test/test_inference_bench.py -q -m "not requires_cache" 2>&1 | tail -25`

For every failure apply exactly one rule and record which in the commit message:
1. The assertion's tolerance is tighter than float32 allows (about 1e-6 relative): pin `dtype="float64"` in that test, keep the assertion, add a float32 sibling at a stated looser tolerance.
2. The assertion encodes standardised context values or a float64 dtype that the spec changed: update it to the new contract, with the reason in the commit message.
3. Anything else: stop and use superpowers:systematic-debugging; do not touch the test.

Expected end state: `0 failed`. Also confirm the baseline captured in Task 1 is untouched: `git status --short notebooks/ensemble_inference_speed_results/` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add tackai/fusion/models.py test/test_fusion_dtype.py test/test_fusion_ensemble.py
# plus, by name, every other test file edited under the step 5 rule
git commit -m "feat(fusion): dtype knob on the estimators, suite green on float32 defaults

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: GP train-side cache and one-time block weights (R3, R5), bit-exact

**Files:**
- Modify: `tackai/fusion/gp.py`
- Create: `test/test_fusion_gp_cache.py`

**Interfaces:**
- Consumes: Task 3's `load_state`, `astype`, `_factor`.
- Produces: `gp._train_side() -> {"sq": {block: Tensor}, "ard": {block: Tensor}}` built at `load_state` (and lazily for a GP unpickled without it); `fold_context(...)` returns an extra key `"active_blocks": list[str]`; `_sq_dists(A, B, same=False, b_sq=None)`, `_cache_distances(Za, Zb, same=False, train=None)`, `_rbf(..., train=None)`, `_assemble(..., train=None)` — the `train=None` default is the unchanged uncached formula and is what the tests use as the reference.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_gp_cache.py`:

```python
"""The train-side cache and the one-time block weights change nothing but the work done."""
import numpy as np
import pytest
import torch

from tackai.fusion.gp import MOL_KERNEL_BLOCKS, AdditiveProductGP

from test_fusion_gp import dims_of, toy


def uncached_predict(gp, Z, return_std=False):
    """predict() as it was before the cache: every train-side term recomputed per call."""
    with torch.no_grad():
        Zt = gp._to_tensor(Z)
        cross = gp._assemble(gp.params_, Zt, gp.Z_train_, gp._cache_distances(Zt, gp.Z_train_))
        mean = (gp.params_["mean"] + (cross @ gp.alpha_).squeeze(1)).numpy()
        if not return_std:
            return mean
        v = torch.linalg.solve_triangular(gp.chol_, cross.T.to(gp.chol_.dtype), upper=False)
        var = gp._diag(gp.params_, Zt) - (v * v).sum(0)
        return mean, torch.sqrt(var.clamp_min(0.0)).numpy()


def uncached_predict_in_context(gp, mol_blocks, fold, return_std=False):
    """predict_in_context() as it was: train-side terms recomputed, every block weight re-tested."""
    with torch.no_grad():
        Zm = {b: torch.as_tensor(np.asarray(mol_blocks[b]), dtype=gp.dtype)
              for b in MOL_KERNEL_BLOCKS}
        cached = {b: gp._sq_dists(Zm[b], gp.Z_train_[b])
                  for b in MOL_KERNEL_BLOCKS if b not in gp.ard_blocks}
        rbf = {b: gp._rbf(b, gp.params_, Zm, gp.Z_train_, cached) for b in MOL_KERNEL_BLOCKS}
        scaled = None
        for b in MOL_KERNEL_BLOCKS:
            term = torch.exp(gp.params_[f"sc:rbf:{b}"]) * rbf[b]
            scaled = term if scaled is None else scaled + term
        mol = sum(rbf[b] for b in MOL_KERNEL_BLOCKS)
        cross = scaled + mol * fold["mol_weight"][None, :] + fold["const"][None, :]
        for block, weight in fold.get("block_weight", {}).items():
            if bool(torch.any(weight != 0)):
                cross = cross + rbf[block] * weight[None, :]
        mean = (gp.params_["mean"] + (cross @ gp.alpha_).squeeze(1)).numpy()
        if not return_std:
            return mean
        v = torch.linalg.solve_triangular(gp.chol_, cross.T.to(gp.chol_.dtype), upper=False)
        var = fold["prior_var"] - (v * v).sum(0)
        return mean, torch.sqrt(var.clamp_min(0.0)).numpy()


def fitted(dtype, interactions=None, n=80):
    Z, y = toy(n=n)
    Z["e3"] = np.random.default_rng(1).normal(size=(n, 3))
    kw = {} if interactions is None else {"interactions": interactions}
    gp = AdditiveProductGP(dims_of(Z), dtype=dtype, **kw)
    gp.fit(Z, y, n_restarts=1, n_iter=15, seed=0)
    return gp, Z


def context_of(Z):
    return {b: Z[b][:1] for b in ("e3", "poi", "cell", "assay_time")}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cached_predict_is_bit_identical_to_the_uncached_formula(dtype):
    gp, Z = fitted(dtype)
    q = {b: a[:23] for b, a in Z.items()}
    for return_std in (False, True):
        got, ref = gp.predict(q, return_std=return_std), uncached_predict(gp, q, return_std)
        pairs = zip(got, ref) if return_std else [(got, ref)]
        for g, r in pairs:
            assert np.array_equal(g, r)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("interactions", [None, ("fingerprint*poi", "descriptors*cell", "poi*cell")])
def test_cached_predict_in_context_is_bit_identical_to_the_uncached_formula(dtype, interactions):
    gp, Z = fitted(dtype, interactions)
    fold = gp.fold_context(context_of(Z))
    mols = {b: Z[b][:17] for b in MOL_KERNEL_BLOCKS}
    got = gp.predict_in_context(mols, fold, return_std=True)
    ref = uncached_predict_in_context(gp, mols, fold, return_std=True)
    assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


def test_the_train_side_cache_is_built_once_per_conditioning(monkeypatch):
    gp, Z = fitted("float32")
    calls = []
    original = AdditiveProductGP._build_train_side
    monkeypatch.setattr(AdditiveProductGP, "_build_train_side",
                        lambda self: calls.append(1) or original(self))
    q = {b: a[:5] for b, a in Z.items()}
    for _ in range(3):
        gp.predict(q, return_std=True)
    assert calls == []                                 # served from the cache built at load_state
    gp.astype("float64")                               # a new conditioning rebuilds it
    assert calls == [1]


def test_a_gp_pickled_before_the_cache_existed_still_predicts_the_same():  # Review Focus 1
    gp, Z = fitted("float32")
    q = {b: a[:9] for b, a in Z.items()}
    before = gp.predict(q, return_std=True)
    del gp._train_side_                                # what an old artifact looks like
    after = gp.predict(q, return_std=True)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_fold_context_resolves_the_active_block_weights_once():
    gp, Z = fitted("float32")                          # default interactions are all "mol*..."
    assert gp.fold_context(context_of(Z))["active_blocks"] == []
    gp2, Z2 = fitted("float32", interactions=("fingerprint*poi",))
    assert gp2.fold_context(context_of(Z2))["active_blocks"] == ["fingerprint"]


def test_a_fold_without_active_blocks_still_works():
    gp, Z = fitted("float32", interactions=("fingerprint*poi", "descriptors*cell"))
    fold = gp.fold_context(context_of(Z))
    mols = {b: Z[b][:6] for b in MOL_KERNEL_BLOCKS}
    expected = gp.predict_in_context(mols, fold)
    fold.pop("active_blocks")
    assert np.array_equal(gp.predict_in_context(mols, fold), expected)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_fusion_gp_cache.py -q 2>&1 | tail -12`
Expected: `test_the_train_side_cache...` and the `active_blocks` / `_train_side_` tests fail (`_build_train_side` and the key do not exist); the bit-identical tests already pass (the cache is not there yet) — they are the gate that must still pass after Step 3.

- [ ] **Step 3: Implement**

In `tackai/fusion/gp.py`:

(a) `_sq_dists` gains `b_sq`:

```python
    @staticmethod
    def _sq_dists(A: torch.Tensor, B: torch.Tensor, same: bool = False,
                  b_sq: Optional[torch.Tensor] = None) -> torch.Tensor:
```
Add to its Args: `b_sq: Precomputed ``(B * B).sum(1)``; identical to computing it here.` and change the first body line to:

```python
        b2 = (B * B).sum(1) if b_sq is None else b_sq
        d2 = (A * A).sum(1)[:, None] + b2[None, :] - 2.0 * (A @ B.T)
```

(b) `_cache_distances(self, Za, Zb, same=False, train=None)`:

```python
        return {b: self._sq_dists(Za[b], Zb[b], same=same,
                                  b_sq=None if train is None else train["sq"][b])
                for b in self.rbf_blocks if b not in self.ard_blocks}
```

(c) `_rbf(self, block, params, Za, Zb, cached, same=False, train=None)` with body:

```python
        ls = self._lengthscale(block, params)
        if block in self.ard_blocks:
            if train is None:
                right, right_sq = Zb[block] / ls, None
            else:
                right, right_sq = train["ard"][block], train["sq"][block]
            d2 = self._sq_dists(Za[block] / ls, right, same=same, b_sq=right_sq)
            return torch.exp(-0.5 * d2)
        return torch.exp(-0.5 * cached[block] / (ls[0] ** 2))
```

(d) `_assemble(self, params, Za, Zb, cached, same=False, train=None)`: pass `train=train` to the `self._rbf(...)` call in its first line.

(e) Add after `load_state`'s body (and call it at the end of `load_state`'s `with` block as `self._train_side_ = self._build_train_side()`):

```python
    def _build_train_side(self) -> dict:
        """Quantities of the training rows that no query batch changes.

        The hyper-parameters are frozen once the GP is conditioned, so the row norms of the
        isotropic blocks and the lengthscale-scaled ARD blocks are the same for every batch
        and every member call. They are exactly the values ``_sq_dists`` would recompute.
        """
        with torch.no_grad():
            sq, ard = {}, {}
            for b in self.rbf_blocks:
                if b in self.ard_blocks:
                    scaled = self.Z_train_[b] / self._lengthscale(b, self.params_)
                    ard[b], sq[b] = scaled, (scaled * scaled).sum(1)
                else:
                    sq[b] = (self.Z_train_[b] * self.Z_train_[b]).sum(1)
            return {"sq": sq, "ard": ard}

    def _train_side(self) -> dict:
        """The train-side cache; rebuilt lazily for a GP pickled before it existed."""
        if self.__dict__.get("_train_side_") is None:
            self._train_side_ = self._build_train_side()
        return self._train_side_
```

(f) `predict`: `train = self._train_side()`; use `self._cache_distances(Zt, self.Z_train_, train=train)` and pass `train=train` to `_assemble`.

(g) `fold_context`: `train = self._train_side()` after `with torch.no_grad():`; compute `cached` with `b_sq=train["sq"][b]` in the `_sq_dists` call and pass `train=train` to the `_rbf` call; before returning add `active = [b for b in MOL_KERNEL_BLOCKS if bool(torch.any(block_weight[b] != 0))]` and return `{..., "active_blocks": active}`.

(h) `predict_in_context`: `train = self._train_side()`; `cached = {b: self._sq_dists(Zm[b], self.Z_train_[b], b_sq=train["sq"][b]) for b in MOL_KERNEL_BLOCKS if b not in self.ard_blocks}`; `rbf` with `train=train`; replace the `for block, weight in fold.get("block_weight", {}).items():` loop with:

```python
            active = fold.get("active_blocks")
            if active is None:                  # a fold built before the weights were resolved once
                active = [b for b, w in fold.get("block_weight", {}).items()
                          if bool(torch.any(w != 0))]
            for block in active:
                cross = cross + rbf[block] * fold["block_weight"][block][None, :]
```

- [ ] **Step 4: Run to verify**

Run: `.venv/bin/python -m pytest test/test_fusion_gp_cache.py test/test_fusion_gp.py -q 2>&1 | tail -12`
Expected: all pass, including the `array_equal` tests (bit-exact). A one-ulp mismatch means a train-side value was computed with different operands than `_sq_dists` uses — compare `(scaled * scaled).sum(1)` with the line it replaces before changing any test.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/gp.py test/test_fusion_gp_cache.py
git commit -m "perf(fusion): cache the GP's train-side kernel terms, resolve block weights once

Bit-identical to the unoptimised formula at both dtypes. The train rows' squared
norms and the lengthscale-scaled ARD block were recomputed on every batch for
every member although the hyper-parameters are frozen after conditioning.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Ensemble hoisting and the shared molecular transform (R1, R6, R7), bit-exact

**Files:**
- Modify: `tackai/fusion/data.py` (`encode`)
- Modify: `tackai/fusion/ensemble.py` (`_predict_with_context`, `_predict_from_records`, new `_molecular_blocks`)
- Create: `test/test_fusion_ensemble_speed.py`

`data.py` carries an uncommitted one-line docstring edit that predates this plan. Commit the file whole and say so in the message (Global Constraints).

**Interfaces:**
- Consumes: `BlockPreprocessor.transform_block`, `.transform_signature` (Task 2); `gp.predict_in_context` (Task 5).
- Produces: `FusionData.encode(records, return_ok=False)` — unchanged return by default; with `return_ok=True` it returns `(X, ok)`. `FusionEnsemble._molecular_blocks(fp, desc) -> list[dict]` — one `{"fingerprint", "descriptors"}` dict per member, sharing one computed result between members whose `transform_signature`s are equal, and only when `fp`/`desc` contain no NaN.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_ensemble_speed.py`:

```python
"""The context path does its shared work once, and changes no returned bit by doing so."""
import numpy as np
import pytest

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.blocks import BlockPreprocessor
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


@pytest.fixture
def ens(data):
    return FusionEnsemble.fit(fast_gp, data, task="pdc50", n_members=3, n_folds=3)


def count_calls(monkeypatch, cls, name):
    calls = []
    original = getattr(cls, name)

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, name, spy)
    return calls


def unhoisted_context_predict(ens, smiles, ctx):
    """The context path as it was: a full-width scratch row and one [valid] copy per member."""
    n = len(smiles)
    fp, desc, ok = ens.data.featurizer.featurize(smiles)
    valid = np.flatnonzero(ok)
    mol_row = np.zeros((n, ens.data.n_columns), dtype=np.float64)
    mol_row[:, ens.data.index["fingerprint"]] = fp
    mol_row[:, ens.data.index["descriptors"]] = desc
    per_member = []
    for member, fold in zip(ens.members, ctx.folds):
        mean, std = np.full(n, np.nan), np.zeros(n)
        Z = dict(member.pre_.transform_blocks(mol_row[valid],
                                              only=["fingerprint", "descriptors"]))
        mean[valid], std[valid] = ens._folded_scores(member, Z, fold, True)
        per_member.append((mean, std))
    return ens._aggregate(per_member, ok, smiles, True)


def test_context_path_is_bit_identical_to_the_unhoisted_reference(ens):
    ctx = ens.transform_context(CTX)
    smiles = SMILES[:4] + ["not a molecule"] + SMILES[4:6]
    got, ref = ens.predict(smiles, context=ctx), unhoisted_context_predict(ens, smiles, ctx)
    assert np.array_equal(got.mean, ref.mean, equal_nan=True)
    assert np.array_equal(got.std, ref.std, equal_nan=True)
    assert got.ok.tolist() == ref.ok.tolist() == [True] * 4 + [False] + [True] * 2


def test_invalid_neighbours_do_not_disturb_the_valid_rows(ens):  # Review Focus 2
    ctx = ens.transform_context(CTX)
    clean = ens.predict(SMILES[:4], context=ctx)
    mixed = ens.predict([SMILES[0], "bad", SMILES[1], SMILES[2], "worse", SMILES[3]], context=ctx)
    keep = mixed.ok
    assert np.allclose(mixed.mean[keep], clean.mean, rtol=1e-4, atol=1e-4)
    assert np.isnan(mixed.mean[~keep]).all()


def test_an_all_invalid_batch_is_all_nan_and_not_an_error(ens):  # Review Focus 2
    pred = ens.predict(["bad", "worse"], context=ens.transform_context(CTX))
    assert np.isnan(pred.mean).all() and not pred.ok.any()


def test_members_with_identical_molecular_transforms_share_one_result(ens, monkeypatch):
    ctx = ens.transform_context(CTX)
    ens.predict(SMILES[:4], context=ctx)                       # warm the featuriser memo
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_block")
    ens.predict(SMILES[:4], context=ctx)
    assert len(calls) == 2, "fingerprint and descriptors should be transformed once for all members"


def test_a_nan_feature_switches_sharing_off(ens, monkeypatch):  # Review Focus 2
    """With no NaN the imputer is a no-op and sharing is exact; with one it is not."""
    ctx = ens.transform_context(CTX)
    real = ens.data.featurizer.featurize

    def with_nan(smiles):
        fp, desc, ok = real(smiles)
        desc = desc.copy()
        desc[0, 3] = np.nan
        return fp, desc, ok
    monkeypatch.setattr(ens.data.featurizer, "featurize", with_nan)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_block")
    pred = ens.predict(SMILES[:4], context=ctx)
    assert len(calls) == 2 * len(ens.members)
    assert np.isfinite(pred.mean).all()


def test_the_context_path_never_builds_a_full_width_row(ens, monkeypatch):
    ctx = ens.transform_context(CTX)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_blocks")
    ens.predict(SMILES[:4], context=ctx)
    assert calls == []


def test_the_records_path_featurises_once(ens, monkeypatch):
    records = [{"smiles": s, **CTX} for s in SMILES[:4]]
    ens.predict(records)
    calls, real = [], ens.data.featurizer.featurize
    monkeypatch.setattr(ens.data.featurizer, "featurize", lambda s: calls.append(1) or real(s))
    ens.predict(records)
    assert len(calls) == 1


def test_encode_can_also_return_which_rows_parsed(data):
    records = [{"smiles": s, **CTX} for s in (SMILES[0], "not a molecule")]
    X = data.encode(records)
    assert isinstance(X, np.ndarray)
    X2, ok = data.encode(records, return_ok=True)
    assert np.array_equal(X, X2) and ok.tolist() == [True, False]
    X0, ok0 = data.encode([], return_ok=True)
    assert X0.shape == (0, data.n_columns) and ok0.shape == (0,)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_fusion_ensemble_speed.py -q 2>&1 | tail -15`
Expected: the sharing, full-width-row, featurise-once and `return_ok` tests fail; the bit-identical, invalid-neighbour and all-invalid tests pass already (they pin behaviour that must survive the change).

- [ ] **Step 3: Implement**

In `tackai/fusion/data.py`, replace `encode` with:

```python
    def encode(self, records: Union[Sequence[dict], pd.DataFrame], return_ok: bool = False):
        """Encode inference records into rows of the design matrix.

        Molecular features are computed on the fly; context blocks are looked up in the
        cached tables.

        Args:
            records: Dicts (or a DataFrame) with ``smiles``, ``poi_seq``, ``e3_seq``,
                ``cell_id``, ``assay`` and optionally ``assay_time``.
            return_ok: Also return which rows' SMILES RDKit could parse, so a caller that
                needs it does not featurise a second time to find out.

        Returns:
            Array of shape ``(len(records), n_columns)``, ``float32``; with ``return_ok``,
            ``(array, ok)``.
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        records = list(records)
        if not records:
            empty = np.empty((0, self.n_columns), dtype=np.float32)
            return (empty, np.empty(0, dtype=bool)) if return_ok else empty

        fp, desc, ok = self.featurizer.featurize([r.get("smiles") for r in records])
        blocks = {"fingerprint": fp, "descriptors": desc}
        blocks["e3"] = self.encoder.encode("e3", [r.get("e3_seq") for r in records])
        blocks["cell"] = self.encoder.encode(
            "cell", [r.get("cell_id", r.get("cell_key")) for r in records])
        blocks["poi"] = self.encoder.encode("poi", [r.get("poi_seq") for r in records])
        blocks["assay"] = self.encoder.encode("assay", [r.get("assay") for r in records])
        blocks["assay_time"] = np.array(
            [[np.nan if r.get("assay_time") is None else float(r["assay_time"])] for r in records],
            dtype=np.float32)
        X = np.concatenate([blocks[b] for b in BLOCK_ORDER], axis=1).astype(np.float32)
        return (X, ok) if return_ok else X
```

In `tackai/fusion/ensemble.py`, replace `_predict_with_context` and `_predict_from_records` with the versions below and add `_molecular_blocks` after `_folded_scores`:

```python
    def _predict_with_context(self, smiles: List[str], context: FusionContext,
                              return_std: bool):
        """Fast path: featurise the molecules, reuse each member's transformed context."""
        n = len(smiles)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        fp, desc, ok = self.data.featurizer.featurize(smiles)
        valid = np.flatnonzero(ok)
        if len(valid) < n:                       # copy only when something is invalid
            fp, desc = fp[valid], desc[valid]
        molecular = self._molecular_blocks(fp, desc) if len(valid) else None

        folds = context.folds or [None] * len(self.members)
        per_member = []
        for k, (member, ctx_blocks, fold) in enumerate(zip(self.members, context.per_member,
                                                           folds)):
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                Z = dict(molecular[k])           # a copy: the molecular dict may be shared
                if fold is not None:
                    mean[valid], std[valid] = self._folded_scores(member, Z, fold, return_std)
                else:
                    for block, row in ctx_blocks.items():
                        Z[block] = np.repeat(row, len(valid), axis=0)
                    mean[valid], std[valid] = self._member_scores(member, Z, return_std)
            per_member.append((mean, std))
        return ok, per_member

    def _molecular_blocks(self, fp: np.ndarray, desc: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Each member's processed molecular blocks, computed once per distinct transform.

        Members whose preprocessors treat a molecular block identically (no scaler, same
        width and dtype) get the same array. That is only exact on NaN-free input, where the
        per-member imputer does nothing; the featuriser never emits NaN (it uses a sentinel),
        but if one appears every member transforms for itself. The arrays are shared between
        members, so nothing downstream may write into them.

        Args:
            fp: Fingerprints of the valid molecules.
            desc: Descriptors of the valid molecules.

        Returns:
            One ``{"fingerprint", "descriptors"}`` dict per member.
        """
        raw = {"fingerprint": fp, "descriptors": desc}
        shareable = not (np.isnan(fp).any() or np.isnan(desc).any())
        done, out = {}, []
        for member in self.members:
            pre = member.pre_
            key = tuple(pre.transform_signature(b) for b in raw) if shareable else None
            if key is None or key not in done:
                Z = {b: pre.transform_block(b, arr) for b, arr in raw.items()}
                if key is not None:
                    done[key] = Z
            else:
                Z = done[key]
            out.append(Z)
        return out
```

```python
    def _predict_from_records(self, records: List[dict], return_std: bool):
        """Ordinary path: encode each record in full, then score it with every member."""
        n = len(records)
        if n == 0:
            return np.zeros(0, dtype=bool), [(np.empty(0), np.empty(0)) for _ in self.members]

        X, ok = self.data.encode(records, return_ok=True)
        valid = np.flatnonzero(ok)
        Xv = X if len(valid) == n else X[valid]
        per_member = []
        for member in self.members:
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                Z = member.pre_.transform(Xv)
                mean[valid], std[valid] = self._member_scores(member, Z, return_std)
            per_member.append((mean, std))
        return ok, per_member
```

- [ ] **Step 4: Run to verify**

Run: `.venv/bin/python -m pytest test/test_fusion_ensemble_speed.py test/test_fusion_ensemble.py test/test_fusion_data.py -q 2>&1 | tail -12`
Expected: all pass, including the bit-identical test.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/data.py tackai/fusion/ensemble.py test/test_fusion_ensemble_speed.py
git commit -m "perf(fusion): hoist the valid-row copy, featurise once, share the molecular transform

Bit-identical to the unhoisted context path. data.py also carries an unrelated
one-line docstring edit that was already in the working tree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: One shared biological context for every member

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Create: `test/test_fusion_shared_context.py`

**Interfaces:**
- Consumes: `BlockPreprocessor.consensus`, `.transform_blocks`, `.transform_signature` (Task 2); Task 6's `_predict_with_context` / `_predict_from_records`.
- Produces (used by Tasks 8, 9): `FusionEnsemble(members, data, task, weights=None, shared_context: bool = True)`; `ens.shared_context: bool`; `ens.context_pre_: BlockPreprocessor | None` (the consensus, `None` when sharing is off); `FusionEnsemble.from_pretrained(..., shared_context=True)`; `ens._iter_member_blocks(X)` (generator of per-member block dicts); with sharing on, `transform_context(...).per_member` is a list in which **every element is the same dict object**.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_shared_context.py`:

```python
"""Every member scores a context through one transform, whatever fold it was fitted on."""
import numpy as np
import pytest

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.blocks import LEGACY_SCALE_BLOCKS, BlockPreprocessor
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction, XGBoostFusion

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
TASK = "pdc50"


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


@pytest.fixture
def fitted(data):
    return FusionEnsemble.fit(fast_gp, data, task=TASK, n_members=3, n_folds=3)


def make_members_disagree(members, data, order=None):
    """Give each member a legacy-scaled preprocessor fitted on a different part of the rows,
    as members fitted on different folds have. Predictions become meaningless; the context
    each member sees is what is under test."""
    _, X, _, _ = data.task_rows(TASK)
    parts = np.array_split(np.arange(len(X)), len(members))
    for k, member in enumerate(members):
        rows = parts[k if order is None else order[k]]
        member.pre_ = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS,
                                        dtype=member.dtype).fit(X[rows])
    return members


def count_calls(monkeypatch, cls, name):
    calls, original = [], getattr(cls, name)

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, name, spy)
    return calls


def test_every_member_is_handed_the_same_context(fitted, data):
    members = make_members_disagree(fitted.members, data)
    shared = FusionEnsemble(members, data, TASK, shared_context=True)
    own = FusionEnsemble(members, data, TASK, shared_context=False)

    ctx = shared.transform_context(CTX)
    assert all(blocks is ctx.per_member[0] for blocks in ctx.per_member)

    per_member = own.transform_context(CTX).per_member          # the defect this removes
    assert not np.allclose(per_member[0]["cell"], per_member[1]["cell"])


def test_the_shared_flag_off_keeps_per_member_contexts(fitted, data):
    ens = FusionEnsemble(fitted.members, data, TASK, shared_context=False)
    assert ens.context_pre_ is None and ens.shared_context is False


def test_the_context_is_transformed_once_not_once_per_member(fitted, data, monkeypatch):
    shared = FusionEnsemble(fitted.members, data, TASK, shared_context=True)
    own = FusionEnsemble(fitted.members, data, TASK, shared_context=False)
    calls = count_calls(monkeypatch, BlockPreprocessor, "transform_blocks")
    shared.transform_context(CTX)
    assert len(calls) == 1
    calls.clear()
    own.transform_context(CTX)
    assert len(calls) == len(own.members)


def test_all_three_paths_use_the_shared_context(fitted, data):
    members = make_members_disagree(fitted.members, data)
    ens = FusionEnsemble(members, data, TASK, shared_context=True)
    records = [{"smiles": s, **CTX} for s in SMILES[:4]]
    from_records = ens.predict(records)
    from_matrix = ens.predict_matrix(data.encode(records))
    from_context = ens.predict(SMILES[:4], context=ens.transform_context(CTX))
    assert np.array_equal(from_records.mean, from_matrix.mean)
    assert np.allclose(from_records.mean, from_context.mean, rtol=1e-3, atol=1e-3)
    unshared = FusionEnsemble(members, data, TASK, shared_context=False).predict(records)
    assert not np.allclose(from_records.mean, unshared.mean)     # the toggle does something


def test_the_context_does_not_depend_on_which_member_saw_which_rows(fitted, data):
    members = fitted.members
    make_members_disagree(members, data, order=[0, 1, 2])
    a = FusionEnsemble(members, data, TASK).transform_context(CTX).per_member[0]
    a = {b: arr.copy() for b, arr in a.items()}
    make_members_disagree(members, data, order=[2, 0, 1])
    b = FusionEnsemble(members, data, TASK).transform_context(CTX).per_member[0]
    for block in a:
        assert np.allclose(a[block], b[block], rtol=1e-5, atol=1e-6), block


def test_members_that_disagree_on_scaling_are_refused_by_name(fitted, data):  # Review Focus 5
    _, X, _, _ = data.task_rows(TASK)
    fitted.members[1].pre_ = BlockPreprocessor(scale_blocks=LEGACY_SCALE_BLOCKS).fit(X)
    with pytest.raises(ValueError, match="e3"):
        FusionEnsemble(fitted.members, data, TASK)


def test_a_missing_assay_time_is_imputed_from_the_consensus(fitted):  # Review Focus 5
    record = {k: v for k, v in CTX.items() if k != "assay_time"}
    ctx = fitted.transform_context(record)
    assert np.isfinite(ctx.per_member[0]["assay_time"]).all()
    assert np.isfinite(fitted.predict(SMILES[:3], context=ctx).mean).all()


def test_gp_and_xgboost_members_share_the_context(data):  # Review Focus 5
    ens = FusionEnsemble.fit([fast_gp, fast_xgb], data, task=TASK, n_folds=3)
    ctx = ens.transform_context(CTX)
    assert ctx.per_member[0] is ctx.per_member[1]
    assert ctx.folds[0] is not None and ctx.folds[1] is None
    pred = ens.predict(SMILES[:4], context=ctx)
    assert np.isfinite(pred.mean).all()
    assert np.allclose(pred.mean, np.average(list(pred.member_predictions.values()), axis=0,
                                             weights=list(ens.weights.values())))


def test_scaled_float64_members_from_the_old_code_still_load_and_score(data):  # Review Focus 1
    ens = FusionEnsemble.fit(lambda **kw: fast_gp(dtype="float64", **kw), data, task=TASK,
                             n_members=2, n_folds=3)
    make_members_disagree(ens.members, data)
    for member in ens.members:
        del member.pre_.dtype                                    # written before dtype existed
    old = FusionEnsemble(ens.members, data, TASK)
    assert old.context_pre_.steps_["cell"][1] is not None        # a scaling consensus
    records = [{"smiles": s, **CTX} for s in SMILES[:3]]
    assert np.isfinite(old.predict(records).mean).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_fusion_shared_context.py -q 2>&1 | tail -15`
Expected: failures with `TypeError: __init__() got an unexpected keyword argument 'shared_context'`.

- [ ] **Step 3: Implement**

In `tackai/fusion/ensemble.py`:

(a) Add `from tackai.fusion.blocks import BLOCK_ORDER, BlockPreprocessor`.

(b) `FusionEnsemble.__init__` gains `shared_context: bool = True` (document it: "Score every member through the consensus of their context preprocessors, so the members cannot disagree about the context whatever they were fitted on. ``False`` restores each member's own context transform."), and after `self.set_weights(weights)`:

```python
        self.shared_context = shared_context
        self.context_pre_ = self._build_context_pre() if shared_context else None
```

(c) Add:

```python
    @property
    def _context_blocks(self) -> List[str]:
        """The context blocks, in design-matrix order."""
        return [b for b in BLOCK_ORDER if b in CONTEXT_BLOCKS]

    def _build_context_pre(self) -> BlockPreprocessor:
        """The one context transform every member uses: the consensus of their own.

        Raises:
            ValueError: If the members disagree about a context block's columns, width or
                whether it is standardised (they were not fitted with one setting).
        """
        return BlockPreprocessor.consensus([m.pre_ for m in self.members],
                                           only=self._context_blocks)

    def _iter_member_blocks(self, X: np.ndarray):
        """Each member's processed blocks for the rows ``X``, one member at a time.

        With a shared context the context blocks are transformed once and handed to every
        member; the molecular blocks always go through the member's own preprocessor, since
        they were fitted on raw molecular columns. A generator, so a large batch never holds
        every member's copy at once.
        """
        if self.context_pre_ is None:
            for member in self.members:
                yield member.pre_.transform(X)
            return
        shared = self.context_pre_.transform_blocks(X, only=self._context_blocks)
        molecular = [b for b in BLOCK_ORDER if b not in CONTEXT_BLOCKS]
        for member in self.members:
            yield {**member.pre_.transform_blocks(X, only=molecular), **shared}
```

(d) `transform_context`: replace the `context_blocks = ...` / `per_member = ...` lines with:

```python
        context_blocks = self._context_blocks
        if self.context_pre_ is not None:
            shared = self.context_pre_.transform_blocks(row, only=context_blocks)
            per_member = [shared for _ in self.members]
        else:
            per_member = [m.pre_.transform_blocks(row, only=context_blocks)
                          for m in self.members]
```

(e) `_predict_from_records` — replace the member loop with:

```python
        blocks = (self._iter_member_blocks(Xv) if len(valid)
                  else iter([None] * len(self.members)))
        per_member = []
        for member, Z in zip(self.members, blocks):
            mean, std = np.full(n, np.nan), np.zeros(n)
            if len(valid):
                mean[valid], std[valid] = self._member_scores(member, Z, return_std)
            per_member.append((mean, std))
        return ok, per_member
```

(f) `predict_matrix` — replace the member loop with:

```python
        blocks = self._iter_member_blocks(X) if n else iter([None] * len(self.members))
        per_member = []
        for member, Z in zip(self.members, blocks):
            if n == 0:
                per_member.append((np.empty(0), np.empty(0)))
            else:
                per_member.append(self._member_scores(member, Z, return_std))
        return self._aggregate(per_member, ok, [None] * n, return_individual)
```

(g) `from_pretrained(..., shared_context: bool = True)` (document it) and construct with `cls(members, data, manifest["task"], weights=manifest.get("weights"), shared_context=shared_context)`.

- [ ] **Step 4: Run to verify**

Run: `.venv/bin/python -m pytest test/test_fusion_shared_context.py test/test_fusion_ensemble.py test/test_fusion_ensemble_speed.py -q 2>&1 | tail -12`
Expected: all pass. If `test_missing_assay_time_is_imputed_from_the_consensus` fails inside `encode_context`, read it first: it already encodes a missing time as NaN (`tackai/fusion/context.py`, `encode_context`), so a failure is in the consensus imputer statistic, not in encoding.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_shared_context.py
git commit -m "feat(fusion): one shared biological context for every ensemble member

Members fitted on different folds disagreed by up to 20% on the same context's
scaler statistics and by 6 hours on the assay-time imputation. The ensemble now
builds the consensus of its members' context preprocessors and uses it on the
cached-context, records and matrix paths. shared_context=False restores the old
per-member behaviour.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Serve saved ensembles in another precision

**Files:**
- Modify: `tackai/fusion/ensemble.py`
- Create: `test/test_fusion_ensemble_astype.py`

**Interfaces:**
- Consumes: `AdditiveProductGP.astype`, `.promoted_` (Task 3); `_build_context_pre` (Task 7).
- Produces (used by Task 9): `FusionEnsemble.astype(dtype) -> self` (in place); `FusionEnsemble.promoted -> list[bool]`; `FusionEnsemble.from_pretrained(..., dtype: Optional[str] = None)` — `None` keeps the stored precision; the manifest gains `"dtype"`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_fusion_ensemble_astype.py`:

```python
"""Saved float64 ensembles can be served in float32 without refitting."""
from functools import partial

import numpy as np
import pytest
import torch

from fusion_fixtures import CELLS, SEQS, SMILES
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble
from tackai.fusion.models import GPInteraction, XGBoostFusion

CTX = {"poi_seq": SEQS["poi"][0], "e3_seq": SEQS["e3"][0], "cell_id": CELLS[0],
       "assay": "western blot", "assay_time": 24.0}
RECORDS = [{"smiles": s, **CTX} for s in SMILES[:4]]


@pytest.fixture
def data(fake_cache, tiny_csv):
    return FusionData.from_csv([tiny_csv], cache=False)


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


def fast_xgb(**kw):
    return XGBoostFusion(n_estimators=20, grid=[{"max_depth": 3, "reg_lambda": 5.0}], **kw)


@pytest.fixture
def ens64(data):
    return FusionEnsemble.fit(partial(fast_gp, dtype="float64"), data, task="pdc50",
                              n_members=2, n_folds=3)


def test_astype_casts_members_preprocessors_and_the_shared_context(ens64):
    before = ens64.predict(RECORDS)
    assert ens64.astype("float32") is ens64
    for member in ens64.members:
        assert member.model_.dtype is torch.float32 and member.pre_.dtype == "float32"
        assert member.dtype == "float32"
    assert ens64.context_pre_.dtype == "float32"
    after = ens64.predict(RECORDS)
    assert np.allclose(before.mean, after.mean, rtol=1e-2, atol=1e-2)


def test_promoted_reports_one_flag_per_member(ens64):
    ens64.astype("float32")
    assert len(ens64.promoted) == len(ens64.members)
    assert all(isinstance(flag, bool) for flag in ens64.promoted)


def test_from_pretrained_can_serve_in_another_precision(ens64, tmp_path):
    before = ens64.predict(RECORDS).mean
    ens64.save(tmp_path / "ens")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "ens", dtype="float32")
    assert all(m.model_.dtype is torch.float32 for m in loaded.members)
    assert np.allclose(loaded.predict(RECORDS).mean, before, rtol=1e-2, atol=1e-2)
    kept = FusionEnsemble.from_pretrained(tmp_path / "ens")           # default: as stored
    assert all(m.model_.dtype is torch.float64 for m in kept.members)


def test_the_manifest_records_the_precision(ens64, tmp_path):
    import json
    ens64.save(tmp_path / "ens")
    assert json.loads((tmp_path / "ens" / "manifest.json").read_text())["dtype"] == "float64"


def test_astype_leaves_xgboost_members_working(data):
    ens = FusionEnsemble.fit([fast_gp, fast_xgb], data, task="pdc50", n_folds=3)
    ens.astype("float64")
    assert np.isfinite(ens.predict(RECORDS).mean).all()
    assert ens.promoted == [False, False]


def test_an_artifact_written_before_this_change_loads_and_scores_identically(ens64, tmp_path):
    """Review Focus 1: no preprocessor dtype, no train-side cache, no promoted_ flag."""
    before = ens64.predict(RECORDS)
    for member in ens64.members:
        del member.pre_.dtype
        del member.model_._train_side_
        del member.model_.promoted_
        del member.dtype
    ens64.save(tmp_path / "old")
    loaded = FusionEnsemble.from_pretrained(tmp_path / "old")
    after = loaded.predict(RECORDS)
    assert np.array_equal(before.mean, after.mean) and np.array_equal(before.std, after.std)
    assert loaded.promoted == [False, False]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_fusion_ensemble_astype.py -q 2>&1 | tail -12`
Expected: failures — `AttributeError: 'FusionEnsemble' object has no attribute 'astype'`.

- [ ] **Step 3: Implement**

In `tackai/fusion/ensemble.py`:

(a) Add after `available_tasks`:

```python
    def astype(self, dtype: str) -> "FusionEnsemble":
        """Serve every member in another precision, in place and without refitting.

        GP members are re-conditioned in ``dtype`` with their saved hyper-parameters; a
        factorisation that precision cannot do is promoted to float64 and shows up in
        :attr:`promoted`. Preprocessors switch their output dtype; tree members only change
        the dtype their inputs are prepared in. The shared context is rebuilt.

        Args:
            dtype: ``"float32"`` or ``"float64"``.

        Returns:
            self
        """
        name = str(dtype).replace("torch.", "")
        for member in self.members:
            member.dtype = name
            member.pre_.dtype = name
            model = getattr(member, "model_", None)
            if hasattr(model, "astype"):
                model.astype(name)
        if self.context_pre_ is not None:
            self.context_pre_ = self._build_context_pre()
        return self

    @property
    def promoted(self) -> List[bool]:
        """Per member: whether its GP's factorisation had to be promoted to float64."""
        return [bool(getattr(getattr(m, "model_", None), "promoted_", False))
                for m in self.members]
```

(b) `from_pretrained(..., dtype: Optional[str] = None, shared_context: bool = True)`: document `dtype` ("Cast every member to this precision after loading; ``None`` keeps the precision it was saved in"), and replace the final `return cls(...)` with:

```python
        ens = cls(members, data, manifest["task"], weights=manifest.get("weights"),
                  shared_context=shared_context)
        return ens.astype(dtype) if dtype is not None else ens
```

(c) `save`: add to the manifest dict `"dtype": getattr(self.members[0], "dtype", "float64"),` next to `"protein_space"`.

- [ ] **Step 4: Run to verify, then the whole fusion suite**

Run: `.venv/bin/python -m pytest test/test_fusion_ensemble_astype.py -q 2>&1 | tail -10`
Expected: all pass.

Run: `.venv/bin/python -m pytest test/test_fusion_blocks.py test/test_fusion_context.py test/test_fusion_data.py test/test_fusion_dtype.py test/test_fusion_ensemble.py test/test_fusion_ensemble_astype.py test/test_fusion_ensemble_speed.py test/test_fusion_features.py test/test_fusion_gp.py test/test_fusion_gp_cache.py test/test_fusion_models.py test/test_fusion_shared_context.py test/test_inference_bench.py -q -m "not requires_cache" 2>&1 | tail -6`
Expected: `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add tackai/fusion/ensemble.py test/test_fusion_ensemble_astype.py
git commit -m "feat(fusion): serve saved ensembles in another precision

astype() re-conditions GP members with their saved hyper-parameters and reports
float64 promotions; from_pretrained(dtype=...) does it on load. Artifacts written
before the dtype, train-side cache and promoted_ attributes existed still load and
score bit-identically.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The notebook — measure, compare, plot, conclude

**Files:**
- Create: `notebooks/ensemble_inference_speed.ipynb` (built by a generator kept outside the repo, then executed)
- Create (by executing it): `notebooks/ensemble_inference_speed_results/{after_*.csv, after_predictions.npz, after_context.json, *.png}`

**Interfaces:**
- Consumes: everything above, and Task 1's `baseline_*` files.
- Produces: the executed notebook and its figures; nothing imports from it.

- [ ] **Step 1: Write the notebook generator (scratch, not committed)**

Set `SCRATCH=/private/tmp/claude-501/-Users-ribes-phd-TACK/e894295a-dd38-4009-bff8-7bf918010cd5/scratchpad` (the session scratchpad; any scratch directory works). Create `$SCRATCH/build_inference_nb.py`:

```python
import nbformat as nbf

CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


md('''
# Fusion ensemble inference: where the time goes, and what the speed work bought

`FusionEnsemble.predict(smiles, context=ctx)` is the call a screening or reinforcement-learning
loop makes hundreds of thousands of times: one fixed biological context, a stream of molecules.
This notebook measures that path and compares it with numbers captured from the **unchanged
package** (`baseline_*.csv`, stamped with the commit they came from).

What changed in the package: float32 throughout (with a float64 retry for a factorisation float32
cannot do), no standardisation of any block, one canonical context shared by every member, and
six redundancies removed from the inference path (R1-R7 below).

Accuracy is **measured and reported, not gated**: the ensembles on disk were fitted under the old
numerics and will be retrained with hyper-parameter optimisation, so a degradation is expected.
''')

code('''
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # torch + xgboost libomp clash on macOS

import copy
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "notebooks"))
import ensemble_inference_bench as bench
from tackai.fusion.data import FusionData
from tackai.fusion.ensemble import FusionEnsemble

RESULTS = REPO / "notebooks" / "ensemble_inference_speed_results"
ENSEMBLES = REPO / "ensembles"
TASKS = [t for t in os.environ.get("INFER_TASKS", "dmax,pdc50,activity").split(",") if t]
POOL, REPEAT = 1024, int(os.environ.get("INFER_REPEAT", "5"))
PATH_COLOR = {"context+std": "#2a6fdb", "context": "#4aa3c7", "records": "#d9822b",
              "matrix": "#6b9a3a"}
STAGES = ["encode", "featurise", "transform", "model", "aggregate", "other"]
STAGE_COLOR = dict(zip(STAGES, ["#a07fd1", "#d9822b", "#4aa3c7", "#2a6fdb", "#6b9a3a", "#c9cbcf"]))

NEED = ["scaling", "breakdown", "redundancy", "setup", "featurise", "heldout"]
missing = [f"baseline_{n}.csv" for n in NEED if not (RESULTS / f"baseline_{n}.csv").exists()]
if not (RESULTS / "baseline_predictions.npz").exists():
    missing.append("baseline_predictions.npz")
assert not missing, f"capture the baseline at the pre-change commit first: {missing}"
print(f"torch {torch.__version__} | numpy {np.__version__} | tasks {TASKS} | repeat {REPEAT}")
print("package now at", bench.stamp())
''')

md('''
## 1. Before: the unchanged code

Everything in this section is read from `baseline_*.csv`, written by `ensemble_inference_bench.py`
before any of the speed work landed.
''')

code('''
def read(label, name):
    frame = pd.read_csv(RESULTS / f"{label}_{name}.csv")
    return frame[frame["task"].isin(TASKS)] if "task" in frame else frame


base = {name: read("baseline", name) for name in NEED}
print("baseline measured at", base["scaling"][["commit", "dirty"]].drop_duplicates()
      .to_dict("records"))
print(f"cold featurisation: {base['featurise'].iloc[0, 0] * 1e3:.1f} ms per molecule "
      "(not part of any number below: the featuriser memoises)")
print("\\nmicroseconds per molecule at batch 256")
display(base["scaling"].query("batch == 256")
        .pivot(index="task", columns="path", values="us_per_mol").round(1))
print("context set-up (ms) and training rows per task")
display(base["setup"].assign(context_setup_ms=lambda f: f["context_setup_s"] * 1e3)
        [["task", "members", "n_train", "context_setup_ms"]].round(2))
''')

code('''
def plot_scaling(series, fname, title):
    """series: (label, frame, linestyle) triples; one panel per task, one line per path."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4.3), sharey=True,
                             squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        for label, frame, style in series:
            for path, g in frame[frame["task"] == task].groupby("path"):
                ax.plot(g["batch"], g["us_per_mol"], style, color=PATH_COLOR[path], marker="o",
                        ms=3, label=f"{path} ({label})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(task)
        ax.set_xlabel("batch size")
    axes[0][0].set_ylabel("microseconds per molecule (lower is better)")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(RESULTS / fname, dpi=130)
    plt.show()


plot_scaling([("before", base["scaling"], "-")], "01_before_scaling.png",
             "Before: cost per molecule against batch size")
''')

code('''
def plot_breakdown(frames, task, fname, title):
    """Stacked horizontal bars of exclusive time per stage; frames: label -> breakdown frame."""
    rows = [(label, path) for label in frames for path in ("context", "context_nostd", "records")]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6))
    for y, (label, path) in enumerate(rows):
        part = frames[label].query("task == @task and path == @path").set_index("stage")["seconds"]
        left = 0.0
        for stage in STAGES:
            value = part.get(stage, 0.0) * 1e3
            ax.barh(y, value, left=left, color=STAGE_COLOR[stage],
                    label=stage if y == 0 else None)
            left += value
        ax.text(left, y, f"  {left:.1f} ms", va="center", fontsize=8)
    names = {"context": "context + std", "context_nostd": "context, no std", "records": "records"}
    ax.set_yticks(range(len(rows)), [f"{names[p]} ({l})" for l, p in rows])
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds per batch of 256")
    ax.set_title(title)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130, bbox_inches="tight")
    plt.show()


plot_breakdown({"before": base["breakdown"]}, TASKS[0], "02_before_breakdown.png",
               f"Before: where one batch of 256 goes ({TASKS[0]})")
''')

code('''
red = base["redundancy"][base["redundancy"]["task"] == TASKS[0]]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.barh(red["item"], red["share_of_predict"] * 100, color="#d9822b")
for y, (share, what) in enumerate(zip(red["share_of_predict"] * 100, red["what"])):
    ax.text(share, y, f"  {share:.1f}%  {what}", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, max(red["share_of_predict"] * 100) * 2.6)
ax.set_xlabel(f"% of one batch of 256 ({TASKS[0]}, cached-context path with std)")
ax.set_title("Before: what each redundancy costs, replayed in isolation")
fig.tight_layout()
fig.savefig(RESULTS / "03_before_redundancy.png", dpi=130)
plt.show()
display(red[["item", "what", "seconds", "share_of_predict"]].round(5))
''')

md('''
## 2. After: four configurations, so each change gets its own number

The saved ensembles are float64 with per-member context preprocessors. They are served here as

| | precision | context |
|---|---|---|
| **A** | float64 | per member (`shared_context=False`) — the old numerics on the new code: *redundancy removal only* |
| **B** | float64 | shared |
| **C** | float32 | per member |
| **D** | float32 | shared — the new defaults |

A and B run first; `astype("float32")` then re-conditions the same members in place for C and D.
''')

code('''
data = FusionData.from_csv(bench.DEV_FILES)
record = bench.context_record(data)
pool = bench.smiles_pool(data, POOL)
base_pred = np.load(RESULTS / "baseline_predictions.npz")
assert list(base_pred["smiles"]) == pool[:256], "baseline pool differs from this run's pool"

CONFIGS = {"A": "f64 / per-member ctx", "B": "f64 / shared ctx",
           "C": "f32 / per-member ctx", "D": "f32 / shared ctx"}
final, sweeps, shifts, heldout, promoted = {}, [], [], [], []

for task in TASKS:
    stored = FusionEnsemble.from_pretrained(ENSEMBLES / f"fusion_{task}", data=data,
                                            shared_context=False)
    members = stored.members
    for name, cast, shared in (("A", None, False), ("B", None, True),
                               ("C", "float32", False), ("D", None, True)):
        if cast:
            FusionEnsemble(members, data, task, shared_context=False).astype(cast)
        ens = FusionEnsemble(members, data, task, shared_context=shared)
        sweeps.append(bench.sweep_paths(ens, record, pool, repeat=REPEAT)
                      .assign(task=task, config=name))
        pred = ens.predict(pool[:256], context=ens.transform_context(record))
        b_mean, b_std = base_pred[f"{task}_mean"], base_pred[f"{task}_std"]
        spread = base_pred[f"{task}_members"].std(axis=0).mean()
        shifts.append({"task": task, "config": name,
                       "corr": np.corrcoef(b_mean, pred.mean)[0, 1],
                       "mean_abs_shift": np.abs(pred.mean - b_mean).mean(),
                       "max_abs_shift": np.abs(pred.mean - b_mean).max(),
                       "baseline_member_spread": spread,
                       "std_ratio": float(np.median(pred.std / np.maximum(b_std, 1e-12)))})
        heldout.append({"task": task, "config": name,
                        "score": bench.heldout_score(ens, data, task)})
        promoted.append({"task": task, "config": name, "promoted_members": sum(ens.promoted),
                         "members": len(ens.members)})
        print(f"{task} {name} ({CONFIGS[name]}) done")
    final[task] = ens                       # config D
    del stored
    gc.collect()

sweeps = pd.concat(sweeps, ignore_index=True)
sweeps.to_csv(RESULTS / "configs_scaling.csv", index=False)
''')

code('''
# The same capture the baseline used, run on the new defaults (config D), so the overlay
# compares like with like.
bench.capture_baseline(final, data, RESULTS, label="after", pool_size=POOL, repeat=REPEAT)
after = {name: read("after", name) for name in NEED}
print("after measured at", after["scaling"][["commit", "dirty"]].drop_duplicates()
      .to_dict("records"))
plot_scaling([("before", base["scaling"], "--"), ("after", after["scaling"], "-")],
             "04_before_after_scaling.png", "Cost per molecule against batch size: before (dashed) "
             "and after (solid)")
''')

code('''
def speedups(batch):
    key = ["task", "path"]
    b = base["scaling"].query("batch == @batch").set_index(key)["us_per_mol"]
    a = after["scaling"].query("batch == @batch").set_index(key)["us_per_mol"]
    return (b / a).rename("speedup").reset_index()


batches = [16, 256, 1024]
fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 3.8), sharey=True)
for ax, batch in zip(axes, batches):
    table = speedups(batch).pivot(index="task", columns="path", values="speedup")
    table = table.reindex(TASKS)[list(PATH_COLOR)]
    table.plot.bar(ax=ax, color=list(PATH_COLOR.values()), width=0.8, legend=batch == batches[-1])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_title(f"batch {batch}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
axes[0].set_ylabel("speedup against the unchanged code (x)")
fig.suptitle("Speedup per path")
fig.tight_layout()
fig.savefig(RESULTS / "05_speedup.png", dpi=130)
plt.show()
display(speedups(256).pivot(index="task", columns="path", values="speedup").round(2))
''')

code('''
# Attribution at batch 256: unchanged code, then A -> B -> C -> D on the new code.
rows = []
for path in ("context+std", "records"):
    for task in TASKS:
        row = {"path": path, "task": task,
               "before": base["scaling"].query("task == @task and path == @path and batch == 256")
               ["us_per_mol"].iloc[0]}
        for name in CONFIGS:
            row[name] = sweeps.query("task == @task and path == @path and batch == 256 "
                                     "and config == @name")["us_per_mol"].iloc[0]
        rows.append(row)
attr = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, path in zip(axes, ("context+std", "records")):
    part = attr[attr["path"] == path].set_index("task").reindex(TASKS)
    part[["before", "A", "B", "C", "D"]].plot.bar(
        ax=ax, color=["#8a8d91", "#c9cbcf", "#9fc0f2", "#f0b987", "#2a6fdb"], width=0.8)
    ax.set_title(f"{path}, batch 256")
    ax.set_ylabel("microseconds per molecule")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
axes[0].legend(["unchanged code"] + [f"{k}: {v}" for k, v in CONFIGS.items()], fontsize=8,
               frameon=False)
fig.suptitle("Which change bought what")
fig.tight_layout()
fig.savefig(RESULTS / "06_attribution.png", dpi=130)
plt.show()
display(attr.round(1))
''')

code('''
plot_breakdown({"before": base["breakdown"], "after": after["breakdown"]}, TASKS[0],
               "07_breakdown_before_after.png",
               f"Where one batch of 256 goes, before and after ({TASKS[0]})")
''')

code('''
import cProfile
import pstats

ens_p = final[TASKS[0]]
ctx_p = ens_p.transform_context(record)
ens_p.predict(pool[:256], context=ctx_p)                          # warm
prof = cProfile.Profile()
prof.enable()
for _ in range(10):
    ens_p.predict(pool[:256], context=ctx_p, return_individual=False)
prof.disable()
rows = [{"function": f"{Path(f).name}:{line}({name})", "calls per batch": nc / 10,
         "own ms per batch": tt * 1e3 / 10, "cumulative ms per batch": ct * 1e3 / 10}
        for (f, line, name), (cc, nc, tt, ct, callers) in pstats.Stats(prof).stats.items()]
print(f"where the new code spends a batch of 256 ({TASKS[0]}, cached context, with std)")
display(pd.DataFrame(rows).sort_values("own ms per batch", ascending=False).head(12).round(2))
''')

md('''
## 3. Twenty-five members

Five members is what `FusionEnsemble.fit` produces by default; the expected production size is
about 25. Inference cost depends on the member count, the training-set size and the block widths,
not on parameter values, so the fitted `dmax` members are **cloned** up to 25 (deep copies, so the
memory footprint is real). This measures cost only: a cloned member adds no accuracy.
''')

code('''
source = final["dmax"] if "dmax" in final else next(iter(final.values()))
task_m = source.task
clones = [copy.deepcopy(source.members[i % len(source.members)]) for i in range(25)]
batch = pool[:256]
rows = []
for m in (1, 2, 5, 10, 15, 20, 25):
    ens_m = FusionEnsemble(clones[:m], data, task_m, shared_context=True)
    ctx = ens_m.transform_context(record)
    ens_m.predict(batch, context=ctx)                            # warm
    rows.append({
        "members": m,
        "setup_ms": bench.timeit(lambda: ens_m.transform_context(record), REPEAT) * 1e3,
        "batch256_std_ms": bench.timeit(
            lambda: ens_m.predict(batch, context=ctx, return_individual=False), REPEAT) * 1e3,
        "batch256_nostd_ms": bench.timeit(
            lambda: ens_m.predict(batch, context=ctx, return_individual=False,
                                  return_std=False), REPEAT) * 1e3,
        "batch1_std_ms": bench.timeit(
            lambda: ens_m.predict(batch[:1], context=ctx, return_individual=False), REPEAT) * 1e3})
members_frame = pd.DataFrame(rows)
members_frame.to_csv(RESULTS / "members_scaling.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for col, label, color in (("batch256_std_ms", "batch 256, with std", "#2a6fdb"),
                          ("batch256_nostd_ms", "batch 256, no std", "#4aa3c7"),
                          ("batch1_std_ms", "batch 1, with std", "#d9822b")):
    axes[0].plot(members_frame["members"], members_frame[col], marker="o", color=color,
                 label=label)
axes[0].set_xlabel("ensemble members")
axes[0].set_ylabel("milliseconds per call")
axes[0].set_title(f"Cost against ensemble size ({task_m})")
axes[0].legend(frameon=False)
axes[1].plot(members_frame["members"], members_frame["setup_ms"], marker="o", color="#6b9a3a")
axes[1].set_xlabel("ensemble members")
axes[1].set_ylabel("milliseconds")
axes[1].set_title("One-off context set-up (transform_context)")
fig.tight_layout()
fig.savefig(RESULTS / "08_members.png", dpi=130)
plt.show()
slope = np.polyfit(members_frame["members"], members_frame["batch256_std_ms"], 1)[0]
display(members_frame.round(2))
print(f"marginal cost of one more member at batch 256 with std: {slope:.1f} ms "
      f"({slope / 256 * 1e3:.0f} microseconds per molecule)")
del clones
gc.collect()
''')

md('''
## 4. What it cost in accuracy (reported, not gated)

Two views. First, how far each configuration's predictions move from the **unchanged code's** on
the same 256 molecules in one context, set against the spread of the baseline's own members.
Second, the score on the fold held out from fitting (the saved ensembles were fitted on the pool of
split 0 only), which says whether that movement is harmful.
''')

code('''
shift_frame = pd.DataFrame(shifts)
score_frame = pd.DataFrame(heldout).pivot(index="task", columns="config", values="score")
score_frame.insert(0, "before", base["heldout"].set_index("task")["score"])
metric = base["heldout"].set_index("task")["metric"]
score_frame.insert(0, "metric", metric)
shift_frame.to_csv(RESULTS / "accuracy_shift.csv", index=False)
score_frame.to_csv(RESULTS / "accuracy_heldout.csv")
print("movement of the predictions against the unchanged code (256 molecules, one context)")
display(shift_frame.round(4))
print("held-out score (R2, or ROC-AUC for activity): unchanged code, then configurations A-D")
display(score_frame.round(4))
print("members whose float32 factorisation had to be promoted to float64")
display(pd.DataFrame(promoted).pivot(index="task", columns="config", values="promoted_members"))

fig, axes = plt.subplots(1, len(TASKS), figsize=(4.6 * len(TASKS), 4.2), squeeze=False)
for ax, task in zip(axes[0], TASKS):
    b = base_pred[f"{task}_mean"]
    d = final[task].predict(pool[:256], context=final[task].transform_context(record)).mean
    ax.scatter(b, d, s=8, alpha=0.5, color="#2a6fdb")
    lo, hi = min(b.min(), d.min()), max(b.max(), d.max())
    ax.plot([lo, hi], [lo, hi], "k", lw=0.8)
    ax.set_title(task)
    ax.set_xlabel("unchanged code")
axes[0][0].set_ylabel("new defaults (D)")
fig.suptitle("Predictions, new defaults against unchanged code")
fig.tight_layout()
fig.savefig(RESULTS / "09_prediction_shift.png", dpi=130)
plt.show()
''')

md('''
## 5. Conclusions

Generated from the numbers above, so they cannot drift from the measurements.
''')

code('''
def speed(path, batch=256):
    s = speedups(batch).query("path == @path").set_index("task")["speedup"]
    return s.min(), s.max()


lo, hi = speed("context+std")
lo_n, hi_n = speed("context")
lo_r, hi_r = speed("records")
lo_m, hi_m = speed("matrix")
print("Speed (batch 256, against the unchanged code, across tasks)")
print(f"  - cached context, with std:  {lo:.1f}-{hi:.1f}x")
print(f"  - cached context, no std:    {lo_n:.1f}-{hi_n:.1f}x")
print(f"  - records path:              {lo_r:.1f}-{hi_r:.1f}x")
print(f"  - predict_matrix:            {lo_m:.1f}-{hi_m:.1f}x")

d256 = after["scaling"].query("batch == 256 and path == 'context+std'").set_index("task")
print("\\nThroughput of the new defaults, cached context with std, batch 256")
for task in TASKS:
    print(f"  - {task}: {d256.loc[task, 'mol_per_s']:.0f} molecules/s "
          f"({d256.loc[task, 'us_per_mol']:.0f} microseconds each) with "
          f"{int(after['setup'].set_index('task').loc[task, 'members'])} members")

per_member_us = slope / 256 * 1e3
five = members_frame.query("members == 5")["batch256_std_ms"].iloc[0]
twenty_five = members_frame.query("members == 25")["batch256_std_ms"].iloc[0]
print(f"\\nEnsemble size ({task_m}): 5 members {five:.0f} ms, 25 members {twenty_five:.0f} ms "
      f"per 256 molecules, i.e. {twenty_five / five:.1f}x for 5x the members; "
      f"+{per_member_us:.0f} microseconds per molecule per extra member")

feat = base["featurise"].iloc[0, 0] * 1e3
print(f"\\nFeaturising a molecule that has never been seen costs {feat:.1f} ms, against "
      f"{d256['us_per_mol'].min() / 1e3:.2f}-{d256['us_per_mol'].max() / 1e3:.2f} ms to score it: "
      "for novel molecules the featuriser, not the model, is the bottleneck, and the memo cache "
      "only helps molecules proposed twice.")

worst = shift_frame.query("config == 'D'").set_index("task")
print("\\nAccuracy (new defaults against the unchanged code; reported, not gated)")
for task in TASKS:
    before_score, after_score = score_frame.loc[task, "before"], score_frame.loc[task, "D"]
    print(f"  - {task}: held-out {score_frame.loc[task, 'metric']} {before_score:.3f} -> "
          f"{after_score:.3f}; predictions correlate {worst.loc[task, 'corr']:.4f}, mean shift "
          f"{worst.loc[task, 'mean_abs_shift']:.4f} against a member spread of "
          f"{worst.loc[task, 'baseline_member_spread']:.4f}")
print("\\nThe saved ensembles were fitted with standardised context and float64; they are served "
      "here under different numerics, so these scores understate what a refit would give.")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {"kernelspec": {"display_name": "tack-venv", "language": "python",
                                 "name": "tack-venv"}}
nbf.write(nb, "/Users/ribes/phd/TACK/notebooks/ensemble_inference_speed.ipynb")
print("wrote", len(CELLS), "cells")
```

- [ ] **Step 2: Build the notebook and register a scratch kernel**

Run:
```bash
SCRATCH=/private/tmp/claude-501/-Users-ribes-phd-TACK/e894295a-dd38-4009-bff8-7bf918010cd5/scratchpad
mkdir -p $SCRATCH/jupyter/kernels/tack-venv
printf '%s' '{"argv": ["/Users/ribes/phd/TACK/.venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "tack-venv", "language": "python", "env": {"OMP_NUM_THREADS": "1"}}' > $SCRATCH/jupyter/kernels/tack-venv/kernel.json
.venv/bin/python $SCRATCH/build_inference_nb.py
```
Expected: `wrote 20 cells` and no traceback. The kernelspec lives in the scratchpad, selected through `JUPYTER_PATH`, so nothing is installed outside the repo.

- [ ] **Step 3: Smoke-test on one task before the long run**

Run: `cd notebooks && JUPYTER_PATH=$SCRATCH/jupyter INFER_TASKS=dmax INFER_REPEAT=2 ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=tack-venv --stdout ensemble_inference_speed.ipynb > /dev/null 2>$SCRATCH/smoke.err; echo exit $?; tail -5 $SCRATCH/smoke.err`
Expected: `exit 0`. A failure here is a notebook bug (wrong column name, wrong path): fix the generator, rebuild, rerun. Do not proceed to the long run on a failing smoke test. `--stdout` leaves the notebook file untouched; the smoke run does overwrite `after_*` and figure files in the results directory, and the real run in Step 4 overwrites them again.

- [ ] **Step 4: Execute the real run (background)**

Run in the background (it takes tens of minutes): `cd notebooks && JUPYTER_PATH=$SCRATCH/jupyter ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=tack-venv --output ensemble_inference_speed.ipynb ensemble_inference_speed.ipynb`
Do not set `MPLBACKEND=Agg`: with it the executed notebook loses its inline figures.
Expected: exit 0 when the background job reports back.

- [ ] **Step 5: Verify the executed notebook**

Run:
```bash
.venv/bin/python - <<'EOF'
import json, pathlib
nb = json.load(open("notebooks/ensemble_inference_speed.ipynb"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errors, errors[0]["ename"] + ": " + errors[0]["evalue"]
assert all(c["execution_count"] for c in code), "a code cell did not run"
figs = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"{len(code)} code cells ran, {figs} inline figures")
res = pathlib.Path("notebooks/ensemble_inference_speed_results")
for name in ["after_scaling.csv", "after_predictions.npz", "configs_scaling.csv",
             "members_scaling.csv", "accuracy_shift.csv", "accuracy_heldout.csv",
             "05_speedup.png", "06_attribution.png", "08_members.png"]:
    assert (res / name).exists(), name
print("all result files present")
EOF
```
Expected: no assertion error; `figs` at least 8.

Then look at the figures: read `notebooks/ensemble_inference_speed_results/05_speedup.png`, `06_attribution.png`, `08_members.png` and `09_prediction_shift.png` with the image reader, and check for clipped titles, overlapping legends, and bars with no values. The previous speed notebook had a clipped title and an overlapping legend; fix the generator and re-execute rather than shipping them. Read the Conclusions cell's output and check each number against the table above it.

- [ ] **Step 6: Commit the notebook and its results**

```bash
git add notebooks/ensemble_inference_speed.ipynb notebooks/ensemble_inference_speed_results/
git commit -m "docs(fusion): executed notebook comparing ensemble inference before and after

Overlays the pre-change timings on float32 / shared-context numbers, attributes
the gain to each change, times 1-25 members, and reports the accuracy movement.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Full verification before finishing**

Run: `.venv/bin/python -m pytest test/test_fusion_blocks.py test/test_fusion_context.py test/test_fusion_data.py test/test_fusion_dtype.py test/test_fusion_ensemble.py test/test_fusion_ensemble_astype.py test/test_fusion_ensemble_speed.py test/test_fusion_features.py test/test_fusion_gp.py test/test_fusion_gp_cache.py test/test_fusion_models.py test/test_fusion_shared_context.py test/test_inference_bench.py -q -m "not requires_cache" 2>&1 | tail -6`
Expected: `0 failed`. (`test/test_ensemble.py` fails for an unrelated pre-existing reason: `ensembles/dc50_ensemble/` is not on disk; it is not part of this suite.)
