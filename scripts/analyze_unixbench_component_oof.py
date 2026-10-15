#!/usr/bin/env python3
"""Summarize component errors in the archived crossfit UnixBench OOF results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from paper_reconstruct_cv_extras import _load_unixbench_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(dataset_root: Path, oof_dir: Path) -> dict:
    rows, test_ids = _load_unixbench_rows(dataset_root, "*/run-*.json")
    truth = {str(Path(row["path"]).resolve()): row for row in rows}
    files = sorted(oof_dir.glob("*crossfit_reconstructor_oof.json"))
    if len(files) != 5:
        raise ValueError(f"expected five host reports, found {len(files)}")

    all_component_errors = []
    all_suite_errors = []
    seen_runs = set()
    hosts = {}
    sources = []
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") != "unixbench" or len(report.get("per_run", [])) != 5:
            raise ValueError(f"unexpected report scope: {path}")
        host = report["machine"]
        if host in hosts:
            raise ValueError(f"duplicate host: {host}")
        run_errors = []
        suite_errors = []
        for prediction in report["per_run"]:
            run_path = str(Path(prediction["run_json"]).resolve())
            if run_path in seen_runs or run_path not in truth:
                raise ValueError(f"duplicate or missing reference run: {run_path}")
            seen_runs.add(run_path)
            predicted = prediction["reconstruction"]["predicted_subtest"]
            if set(predicted) != set(test_ids):
                raise ValueError(f"incomplete component predictions: {run_path}")
            actual = truth[run_path]["y"][:-1]
            estimates = np.asarray([predicted[test_id] for test_id in test_ids], dtype=float)
            if not np.all(np.isfinite(estimates)) or np.any(actual <= 0):
                raise ValueError(f"non-finite prediction or invalid target: {run_path}")
            component_errors = np.abs(estimates - actual) / actual * 100.0
            suite_error = float(prediction["comparison"]["suite_relative_error"]) * 100.0
            run_errors.append({
                "run_json": run_path,
                "reference_sha256": sha256(Path(run_path)),
                "mean_component_ape_percent": float(np.mean(component_errors)),
                "max_component_ape_percent": float(np.max(component_errors)),
                "suite_ape_percent": suite_error,
            })
            all_component_errors.extend(component_errors.tolist())
            all_suite_errors.append(suite_error)
            suite_errors.append(suite_error)
        hosts[host] = {
            "n_runs": len(run_errors),
            "mean_component_ape_percent": float(np.mean([r["mean_component_ape_percent"] for r in run_errors])),
            "mean_suite_ape_percent": float(np.mean(suite_errors)),
            "runs": run_errors,
        }
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})

    return {
        "protocol": "archived outer run-held-out crossfit UnixBench predictions",
        "n_hosts": len(hosts),
        "n_runs": len(seen_runs),
        "n_component_predictions": len(all_component_errors),
        "mean_component_ape_percent": float(np.mean(all_component_errors)),
        "median_component_ape_percent": float(np.median(all_component_errors)),
        "p95_component_ape_percent": float(np.percentile(all_component_errors, 95)),
        "max_component_ape_percent": float(np.max(all_component_errors)),
        "mean_suite_ape_percent": float(np.mean(all_suite_errors)),
        "hosts": hosts,
        "source_reports": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset_root, args.oof_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n_runs", "n_component_predictions", "mean_component_ape_percent",
        "p95_component_ape_percent", "mean_suite_ape_percent")}, sort_keys=True))


if __name__ == "__main__":
    main()
