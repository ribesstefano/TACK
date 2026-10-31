"""Audit training-label attrition by E3/source using only research search rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.research import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    for filename in ("rows.csv", "labels.npz"):
        if sha256(args.snapshot / filename) != manifest["artifact_hashes"][filename]:
            raise ValueError(f"Snapshot changed: {filename}")
    table = pd.read_csv(args.snapshot / "rows.csv", usecols=[
        "partition", "source_partition", "e3_raw", "source", "publication",
        "Dmax_h", "DC50_h", "Dmax_operator", "DC50_operator"])
    table = table.loc[table.partition.eq("search")].copy()
    with np.load(args.snapshot / "labels.npz", allow_pickle=False) as data:
        labels = {name: data[name][table.index] for name in data.files
                  if name.endswith(("_activity", "_dmax", "_pdc50"))}
    table.reset_index(drop=True, inplace=True)
    reports = []
    for fields in (["e3_raw"], ["source_partition", "e3_raw"], ["source", "e3_raw"]):
        for values, rows in table.groupby(fields, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            index = rows.index.to_numpy()
            record = {"slice": {field: str(value) for field, value in zip(fields, values)},
                      "rows": len(index), "publications": int(rows.publication.nunique()), "labels": {}}
            for name, array in labels.items():
                y = array[index]
                observed = np.isfinite(y)
                record["labels"][name] = {"observed": int(observed.sum())}
                if name.endswith("_activity"):
                    record["labels"][name].update(positive=int((y == 1).sum()),
                        prevalence=float(np.mean(y[observed])) if observed.any() else None)
            compat, exact = labels["compat_activity"][index], labels["exact_activity"][index]
            lost = (compat == 1) & ~np.isfinite(exact)
            times = rows[["Dmax_h", "DC50_h"]].to_numpy(float)
            known_times = np.isfinite(times).all(axis=1)
            mismatched = known_times & ~np.isclose(times[:, 0], times[:, 1])
            record["lost_compat_positive"] = int(lost.sum())
            record["lost_positive_missing_time"] = int((lost & ~known_times).sum())
            record["lost_positive_different_times"] = int((lost & mismatched).sum())
            record["lost_positive_missing_exact_endpoint"] = int((lost & (
                ~np.isfinite(labels["exact_dmax"][index]) |
                ~np.isfinite(labels["exact_pdc50"][index]))).sum())
            record["observed_activity_disagreements"] = int((np.isfinite(compat) & np.isfinite(exact)
                                                            & (compat != exact)).sum())
            reports.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"snapshot_hash": sha256(args.snapshot / "manifest.json"),
        "scope": "search rows only; reasons can overlap; no confirmation/test analysis", "slices": reports},
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
