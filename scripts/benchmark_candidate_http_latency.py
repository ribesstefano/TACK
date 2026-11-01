"""Measure a provisional candidate through the real loopback HTTP service.

Only search SMILES and stacking labels are read. A temporary calibration is
fitted on disjoint stacking groups solely to exercise the server's full output
shape; it is not a final calibrated artifact or a model-selection result.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.research import read_npz_rows, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--calibration-unit", choices=["row", "group_max"], default="row")
    parser.add_argument("--tackos-source", type=Path,
                        help="Optional TACKOS checkout for timing its HTTP predictor")
    args = parser.parse_args()
    if args.output.exists() or args.batch < 1 or args.repeats < 1:
        raise ValueError("Need a fresh output and positive batch/repeats")
    manifest_path = args.snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = CandidateArtifact.from_pretrained(args.bundle)
    if artifact.recipe.get("fit_stage") != "pre_calibration" or artifact.task != "regression":
        raise ValueError("Need a provisional uncalibrated regression bundle")
    if artifact.context_contract.get("snapshot_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Bundle and snapshot differ")
    feature = artifact.recipe["feature"]
    for name in ("rows.csv", "partition.npy", "groups.npy", "labels.npz", f"{feature}.npy"):
        if sha256(args.snapshot / name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    roles = np.load(args.snapshot / "partition.npy", allow_pickle=False)
    groups = np.load(args.snapshot / "groups.npy", allow_pickle=False)
    X = np.load(args.snapshot / f"{feature}.npy", mmap_mode="r", allow_pickle=False)
    stack = np.flatnonzero(roles == "stacking")
    y = read_npz_rows(args.snapshot / "labels.npz",
                      f"{artifact.recipe['label_policy']}_{artifact.recipe['task']}", stack)
    valid = np.isfinite(y)
    stack, y = stack[valid], y[valid]
    train, cal = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=20261101).split(
        stack, groups=groups[stack]))
    if set(groups[stack[train]]) & set(groups[stack[cal]]):
        raise ValueError("Latency-only scale/calibration groups overlap")
    artifact.fit_residual_scale(X[stack[train]], y[train])
    artifact.calibrate(X[stack[cal]], y[cal], alpha=.1, unit=args.calibration_unit,
                       groups=groups[stack[cal]] if args.calibration_unit == "group_max" else None)
    artifact.recipe["fit_stage"] = "latency_only_not_for_deployment"

    smiles, context, seen = [], None, set()
    with (args.snapshot / "rows.csv").open(newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if roles[i] != "search":
                continue
            if context is None and row["poi_seq"] and row["e3_seq"]:
                context = {"poi_seq": row["poi_seq"], "e3_seq": row["e3_seq"],
                           "cell_id": row["cell_key"] or None, "assay": row["assay_raw"] or None,
                           "assay_time": float(row["assay_time"]) if row["assay_time"] else None}
            value = row["canonical_smiles"]
            if value not in seen and len(smiles) < args.batch:
                smiles.append(value)
                seen.add(value)
            if len(smiles) >= args.batch and context is not None:
                break
    if len(smiles) != args.batch or context is None:
        raise ValueError("Insufficient search molecules or context for HTTP latency benchmark")
    task = artifact.recipe["task"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tack-latency-") as temp:
        temp = Path(temp)
        artifact.save_pretrained(temp / "models" / task)
        context_path, port_path, log_path = temp / "context.json", temp / "port", temp / "server.log"
        context_path.write_text(json.dumps(context))
        command = [sys.executable, "-m", "tackai.cli", "serve", "--candidate-dir", str(temp / "models"),
                   "--task", task, "--local-context", "--protein-space", "combined",
                   "--context", str(context_path), "--port", "0", "--port-file", str(port_path),
                   "--log-level", "warning"]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["TACKAI_CACHE"] = str((args.snapshot / "context").resolve())
        started = time.perf_counter()
        with log_path.open("w+") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while not port_path.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.1)
                if not port_path.exists():
                    log.flush()
                    raise RuntimeError(f"Latency server did not bind: {log_path.read_text()}")
                base = f"http://127.0.0.1:{int(port_path.read_text())}"
                while True:
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as response:
                            if json.load(response).get("status") == "ok":
                                break
                    except urllib.error.URLError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.flush()
                            raise RuntimeError(f"Latency server did not start: {log_path.read_text()}")
                        time.sleep(.1)
                startup_seconds = time.perf_counter() - started
                payload = json.dumps({"smiles": smiles}).encode()

                def score():
                    request = urllib.request.Request(base + "/predict", data=payload, method="POST",
                                                     headers={"Content-Type": "application/json"})
                    begin = time.perf_counter()
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result = json.load(response)
                    elapsed = (time.perf_counter() - begin) * 1000
                    if result["n_invalid"] or len(result["predictions"][task]["mean"]) != args.batch:
                        raise ValueError("HTTP latency response has invalid or misaligned predictions")
                    return elapsed

                cold = score()
                warm = [score() for _ in range(args.repeats)]
                tackos_warm = None
                if args.tackos_source is not None:
                    sys.path.insert(0, str(args.tackos_source.resolve()))
                    from tackos.models.tack_http_predictor import TACKHTTPPredictor

                    predictor = TACKHTTPPredictor(base, tasks=[task], max_batch=args.batch)
                    tackos_warm = []
                    for _ in range(args.repeats):
                        begin = time.perf_counter()
                        values = predictor.predict(smiles)
                        tackos_warm.append((time.perf_counter() - begin) * 1000)
                        if any(len(v) != args.batch or not np.isfinite(v).all() for v in values.values()):
                            raise ValueError("TACKOS predictor returned invalid or misaligned outputs")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if port_path.exists():
            raise RuntimeError("Server did not clean up its port file")
    result = {"snapshot_sha256": sha256(manifest_path),
              "base_model_sha256": sha256(args.bundle / "candidate.joblib"),
              "task": task, "batch": args.batch, "repeats": args.repeats,
              "temporary_calibration_unit": args.calibration_unit,
              "startup_seconds": startup_seconds, "cold_request_ms": cold,
              "warm_request_ms": {"median": float(np.median(warm)), "p95": float(np.quantile(warm, .95))},
              "note": "Local HTTP with temporary stacking-only calibration; not a final model or RL run."}
    if tackos_warm is not None:
        result["tackos_warm_predict_ms"] = {"median": float(np.median(tackos_warm)),
                                             "p95": float(np.quantile(tackos_warm, .95))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
