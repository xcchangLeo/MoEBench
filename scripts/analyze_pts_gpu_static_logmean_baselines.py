#!/usr/bin/env python3
"""Re-evaluate existing static-feature PTS-GPU proxies on the OOF log-mean target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("wang_dnn", "wang_lr", "tousi_rf", "tousi_dt", "tousi_mlp", "tousi_en")


def _load_baseline_module() -> Any:
    path = ROOT / "scripts" / "baseline_execution_free.py"
    spec = importlib.util.spec_from_file_location("baseline_execution_free", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="aces-System-Product-Name")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_baseline_module()
    from moebench.phoronix.training_data import extract_targets_from_pts_dataset
    from scripts.reconstruct_train_eval import pts_suite_target_from_indices

    paths = baseline._collect_paths(
        args.dataset_root.resolve(), suite_key="pts_gpu", machine=args.machine
    )
    rows, test_ids = baseline._build_rows(
        paths,
        suite_key="pts_gpu",
        xi_mode="static_hw_only",
    )
    if len(rows) != 5:
        raise RuntimeError(f"Expected five audited M1 GPU runs, found {len(rows)}")

    target_by_path: dict[str, float] = {}
    for row in rows:
        ds = baseline._load_run(Path(row["path"]))
        targets = extract_targets_from_pts_dataset(ds, tuple(test_ids))
        if targets is None:
            raise RuntimeError(f"Missing PTS-GPU target values in {row['path']}")
        indices, _ = targets
        row["y"] = pts_suite_target_from_indices(indices, "logmean")
        target_by_path[row["path"]] = float(row["y"])

    results = []
    for method in args.methods:
        result = baseline._run_method(
            method,
            rows,
            train_kwargs=baseline._method_train_kwargs(
                method, hidden=128, epochs=300, lr=1e-3
            ),
            rf_estimators=args.rf_estimators,
            seed=args.seed,
        )
        results.append(result)

    run_records = []
    for row in rows:
        path = Path(row["path"])
        run_records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "target_logmean": target_by_path[str(path)],
            }
        )
    probe_path = args.dataset_root / "models" / args.machine / "probe_dataset_pts_gpu.json"
    result = {
        "schema": "moebench.pts_gpu.static_logmean_baselines.v1",
        "machine": args.machine,
        "suite": "pts/nvidia-gpu-compute",
        "target": "log-mean of ten primary PTS-GPU profile values, matching the audited hybrid OOF target",
        "feature_scope": "static_hw_only, using the same XiVectorizer path as existing execution-free proxies",
        "split_protocol": "leave-one-run-out over five runs from one collection session",
        "seed": args.seed,
        "rf_estimators": args.rf_estimators,
        "probe_dataset_sha256": _sha256(probe_path),
        "runs": run_records,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "method": row["method"],
                    "mean_error_percent": row["mean_suite_rel_err_pct"],
                    "median_feature_cost_seconds": row["median_xi_wall_s"],
                }
                for row in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
