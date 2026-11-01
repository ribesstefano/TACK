"""Freeze one explicitly chosen completed grouped run into a candidate recipe.

This tool never ranks models or reads confirmation/test data. The caller must
select the run using development evidence before creating the recipe.
"""
import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_from_run(directory, *, weights="uniform", l2=.1, alpha=.1):
    """Extract one set of nested-tuned parameters per shared search fold."""
    directory = Path(directory)
    if not (directory / "COMPLETE").is_file():
        raise ValueError("Cannot freeze an incomplete benchmark run")
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    if config.get("model") not in {"xgb", "current_xgb", "extratrees", "catboost"}:
        raise ValueError("Candidate fit tool cannot reproduce this model family")
    if config.get("ablation") != "all" or config.get("policy") not in {"exact", "compat"}:
        raise ValueError("Only complete-feature exact or compatibility training is supported")
    if config.get("eval_policy") != "exact" or config.get("trials", 0) < 1:
        raise ValueError("A tuned run with exact evaluation labels is required")
    repeats, folds = config["repeats"], config["folds"]
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(folds, int) or folds < 2:
        raise ValueError("Invalid repeat/fold schedule")
    expected = {f"repeat{r}_fold{f}": (r, f) for r in range(repeats) for f in range(folds)}
    seen, members, source_count = set(), [], 0
    for row in metrics:
        name = row.get("split")
        if name == "source_holdout":
            source_count += 1
            continue
        if name not in expected or name in seen or not isinstance(row.get("params"), dict):
            raise ValueError(f"Unexpected, duplicate or invalid benchmark fold: {name}")
        seen.add(name)
        repeat, fold = expected[name]
        members.append({"family": config["model"], "repeat": repeat, "fold": fold,
                        "seed": 20261030 + repeat * folds + fold, "params": row["params"]})
    if seen != set(expected) or source_count != 1:
        raise ValueError("The completed benchmark is missing one or more search folds")
    members.sort(key=lambda item: (item["repeat"], item["fold"]))
    return {"schema_version": 1, "snapshot_hash": config["snapshot_hash"],
            "task": config["task"], "feature": config["feature"],
            "label_policy": config["policy"],
            "cv": {"repeats": repeats, "folds": folds,
                   "seeds": [1000 + r for r in range(repeats)]},
            "members": members, "stacking": {"weights": weights, "l2": l2},
            "calibration": {"alpha": alpha}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["uniform", "fit"], default="uniform")
    parser.add_argument("--l2", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.1)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or not 0 <= args.l2 < float("inf"):
        parser.error("alpha must lie between zero and one and l2 must be finite and nonnegative")
    recipe = recipe_from_run(args.run, weights=args.weights, l2=args.l2, alpha=args.alpha)
    snapshot = Path(json.loads((args.run / "config.json").read_text())["snapshot"])
    config = json.loads((args.run / "config.json").read_text())
    if file_hash(snapshot / "manifest.json") != config["snapshot_hash"]:
        raise ValueError("Frozen snapshot does not match benchmark provenance")
    receipt = {"source_run": str(args.run), "snapshot_hash": config["snapshot_hash"],
               "config_sha256": file_hash(args.run / "config.json"),
               "metrics_sha256": file_hash(args.run / "metrics.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite recipe or provenance")
    with args.output.open("x") as stream:
        json.dump(recipe, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
