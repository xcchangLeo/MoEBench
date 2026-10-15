#!/usr/bin/env python3
"""Compute same-host train-fold mean baselines from audited OOF reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    per_run = {str(row["run_json"]): row for row in report["per_run"]}
    folds = report["fold_manifest"]
    if len(per_run) != 5 or len(folds) != 5:
        raise ValueError(f"Expected five runs/folds in {path}")

    predictions = []
    seen_heldout: set[str] = set()
    for fold in folds:
        heldout = [str(value) for value in fold["heldout_runs"]]
        train = [str(value) for value in fold["train_runs"]]
        if len(heldout) != 1 or len(train) != 4:
            raise ValueError(f"Expected one held-out and four train runs in {path}")
        heldout_path = heldout[0]
        if heldout_path in seen_heldout or heldout_path not in per_run:
            raise ValueError(f"Invalid held-out run membership in {path}: {heldout_path}")
        if set(train) != set(per_run) - {heldout_path}:
            raise ValueError(f"Train split does not match other four runs in {path}")
        seen_heldout.add(heldout_path)
        target = float(per_run[heldout_path]["ground_truth_suite"])
        prediction = sum(float(per_run[run]["ground_truth_suite"]) for run in train) / len(train)
        pipeline_error = float(per_run[heldout_path]["comparison"]["suite_relative_error"]) * 100
        predictions.append({
            "fold": int(fold["fold"]),
            "heldout_run": heldout_path,
            "training_runs": train,
            "train_mean_prediction": prediction,
            "ground_truth_suite": target,
            "train_mean_relative_error_percent": abs(prediction - target) / max(abs(target), 1e-12) * 100,
            "pipeline_relative_error_percent": pipeline_error,
        })
    if seen_heldout != set(per_run):
        raise ValueError(f"Not every run is held out once in {path}")
    return {
        "machine": report["machine"],
        "benchmark": report.get("benchmark", "phoronix" if report.get("pts_suite") else "unixbench"),
        "pts_suite": report.get("pts_suite"),
        "n_runs": len(predictions),
        "mean_train_mean_error_percent": sum(
            row["train_mean_relative_error_percent"] for row in predictions
        ) / len(predictions),
        "mean_pipeline_error_percent": sum(
            row["pipeline_relative_error_percent"] for row in predictions
        ) / len(predictions),
        "fresh_measurement_seconds": 0.0,
        "per_run": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Audited OOF JSON reports with fold manifests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    inputs = []
    for path in args.input:
        resolved = path.resolve()
        rows.append(analyze(resolved))
        inputs.append({"path": str(resolved), "sha256": sha256(resolved)})
    output = {
        "schema": "moebench.oof_train_mean_baseline.v1",
        "protocol": "For each held-out run, predict the suite target by the arithmetic mean of the other four training-run suite targets from the same host and OOF fold; no current-run context or workload measurement is used.",
        "inputs": inputs,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{
        "machine": row["machine"], "benchmark": row["benchmark"],
        "mean_train_mean_error_percent": row["mean_train_mean_error_percent"],
        "mean_pipeline_error_percent": row["mean_pipeline_error_percent"],
    } for row in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
