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
from fusion_fixtures import build_ensemble


def fast_gp(**kw):
    return GPInteraction(n_restarts=1, n_iter=5, max_hyper_points=40, **kw)


@pytest.fixture
def small(fake_cache, tiny_csv):
    data = FusionData.from_csv([tiny_csv], cache=False)
    ens = build_ensemble(fast_gp, data, task="pdc50", n_members=2, n_folds=3)
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
