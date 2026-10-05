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
