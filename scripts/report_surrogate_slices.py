"""Report paired development errors by observed biological/source metadata.

No confirmation/test predictions or labels are read. Slice metrics are descriptive
and retain group counts; small slices must not be treated as reliable rankings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .benchmark_surrogates import scores
    from .summarize_surrogate_benchmarks import paired_interval, paired_view
except ImportError:
    from benchmark_surrogates import scores
    from summarize_surrogate_benchmarks import paired_interval, paired_view
from tackai.fusion.research import sha256


METADATA = ["source", "publication", "e3_raw", "Degradation_Target", "assay_raw",
            "cell_key", "assay_time", "partition", "source_partition"]


def slice_report(candidate, baseline, metadata, task, view):
    """Compare aligned rows, retaining small/single-class slices without inference."""
    pred, base = paired_view(candidate, baseline, view)
    selected = metadata.iloc[pred.index].copy()
    if not selected.partition.eq("search").all():
        raise ValueError("Development report contains non-search rows")
    if view == "source_holdout" and not selected.source_partition.eq("eval").all():
        raise ValueError("Source report contains rows outside source evaluation")
    selected.index = pred.index
    selected["time_bin"] = pd.cut(pd.to_numeric(selected.assay_time, errors="coerce"),
                                   [-np.inf, 6, 24, 48, np.inf],
                                   labels=["<=6h", ">6-24h", ">24-48h", ">48h"])
    results = []
    fields = ["source", "publication", "e3_raw", "Degradation_Target", "assay_raw", "cell_key", "time_bin"]
    selections = [("all", "all", pred.index)]
    for field in fields:
        for value, group in selected.groupby(selected[field].astype("string").fillna("(missing)")):
            selections.append((field, str(value), group.index))
    if task in ("dmax", "pdc50"):
        threshold = .8 if task == "dmax" else 6.
        selections.extend(("target_region", label, pred.index[mask]) for label, mask in
                          [(f"y>{threshold}", pred.y > threshold), (f"y<={threshold}", pred.y <= threshold)])
    for field, value, indices in selections:
        if not len(indices):
            continue
        candidate_scores = scores(pred.loc[indices, "y"].to_numpy(),
                                  pred.loc[indices, "prediction"].to_numpy(), task == "activity") if len(indices) > 1 else None
        base_scores = scores(base.loc[indices, "y"].to_numpy(),
                             base.loc[indices, "prediction"].to_numpy(), task == "activity") if len(indices) > 1 else None
        groups = pred.loc[indices, "group"].nunique()
        publications = selected.loc[indices, "publication"].nunique()
        row = dict(field=field, value=value, n=len(indices), chemical_groups=int(groups),
                            publications=int(publications),
                            small_slice=bool(len(indices) < 30 or groups < 5 or
                                            (view == "source_holdout" and publications < 5)),
                            candidate=candidate_scores, reference=base_scores)
        if field == "e3_raw" and not row["small_slice"]:
            units = (selected.loc[indices, "publication"] if view == "source_holdout"
                     else pred.loc[indices, "group"])
            if units.isna().any():
                raise ValueError("Cannot bootstrap missing group identities")
            row["improvement_ci"] = paired_interval(
                pred.loc[indices, "y"].to_numpy(), base.loc[indices, "prediction"].to_numpy(),
                pred.loc[indices, "prediction"].to_numpy(), units.to_numpy(), task == "activity", 1000)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configs, predictions = [], []
    for directory in (args.candidate, args.reference):
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"Incomplete run: {directory}")
        configs.append(json.loads((directory / "config.json").read_text()))
        predictions.append(pd.read_csv(directory / "predictions.csv"))
    for key in ("snapshot_hash", "task"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Incompatible {key}")
    if configs[0].get("eval_policy", "exact") != configs[1].get("eval_policy", "exact"):
        raise ValueError("Incompatible evaluation policies")
    snapshot = Path(configs[0]["snapshot"])
    if sha256(snapshot / "manifest.json") != configs[0]["snapshot_hash"]:
        raise ValueError("Snapshot manifest changed")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if sha256(snapshot / "rows.csv") != manifest["artifact_hashes"]["rows.csv"]:
        raise ValueError("Snapshot metadata changed")
    metadata = pd.read_csv(snapshot / "rows.csv", usecols=METADATA)
    report = {view: slice_report(*predictions, metadata, configs[0]["task"], view)
              for view in ("cv", "source_holdout")}
    report.update(candidate=str(args.candidate), reference=str(args.reference),
                  snapshot_hash=configs[0]["snapshot_hash"], task=configs[0]["task"],
                  note="Development slices; E3 intervals use 1000 paired group draws without multiple-comparison correction.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
