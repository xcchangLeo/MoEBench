#!/usr/bin/env python3
"""Re-evaluate static-feature PTS-CPU proxies on each host's hybrid OOF target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MACHINES = (
    "aces-System-Product-Name",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp16krl0yc7euw7sb6slZ",
    "iZbp1acaw5wdllhz47922rZ",
    "iZbp1glgt48i9a8d49embxZ",
)
METHODS = ("wang_dnn", "tousi_mlp")


def _load_baseline_module() -> Any:
    path = ROOT / "scripts" / "baseline_execution_free.py"
    spec = importlib.util.spec_from_file_location("baseline_execution_free", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_oof(machine: str, oof_root: Path) -> dict[str, Any]:
    path = oof_root / f"round-1-oof-pts-cpu-{machine}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("pts_suite_target") != "logmean":
        raise RuntimeError(f"Expected logmean OOF target in {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--oof-root", type=Path, default=ROOT / "agent_workflow" / "rounds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_baseline_module()
    from moebench.phoronix.training_data import extract_targets_from_pts_dataset
    from scripts.reconstruct_train_eval import pts_suite_target_from_indices

    host_results = []
    for machine in MACHINES:
        oof = _load_oof(machine, args.oof_root)
        paths = baseline._collect_paths(
            args.dataset_root.resolve(), suite_key="pts_cpu", machine=machine
        )
        rows, test_ids = baseline._build_rows(
            paths, suite_key="pts_cpu", xi_mode="static_hw_only"
        )
        if len(rows) != 5:
            raise RuntimeError(f"{machine}: expected five runs, found {len(rows)}")

        row_by_path = {str(Path(row["path"]).resolve()): row for row in rows}
        oof_runs = {str(Path(row["path"]).resolve()): row for row in oof["input_artifacts"]["runs"]}
        if set(row_by_path) != set(oof_runs):
            raise RuntimeError(f"{machine}: baseline and hybrid OOF run sets differ")

        for path_key, row in row_by_path.items():
            ds = baseline._load_run(Path(path_key))
            targets = extract_targets_from_pts_dataset(ds, tuple(test_ids))
            if targets is None:
                raise RuntimeError(f"Missing PTS-CPU targets in {path_key}")
            indices, _ = targets
            row["y"] = pts_suite_target_from_indices(indices, "logmean")
            if _sha256(Path(path_key)) != oof_runs[path_key]["sha256"]:
                raise RuntimeError(f"{machine}: input hash differs from hybrid OOF: {path_key}")

        oof_truth = {
            str(Path(row["run_json"]).resolve()): float(row["ground_truth_suite"])
            for row in oof["per_run"]
        }
        for path_key, row in row_by_path.items():
            if not np.isclose(row["y"], oof_truth[path_key], rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"{machine}: target mismatch for {path_key}")

        folds = []
        for manifest in oof["fold_manifest"]:
            train_paths = [str(Path(value).resolve()) for value in manifest["train_runs"]]
            test_paths = [str(Path(value).resolve()) for value in manifest["heldout_runs"]]
            folds.append(
                (
                    [next(i for i, row in enumerate(rows) if str(Path(row["path"]).resolve()) == value) for value in train_paths],
                    [next(i for i, row in enumerate(rows) if str(Path(row["path"]).resolve()) == value) for value in test_paths],
                )
            )

        method_results = []
        for method in METHODS:
            per_run = []
            for fold_i, (train_idx, test_idx) in enumerate(folds):
                x_train = np.asarray([rows[i]["x"] for i in train_idx], dtype=np.float64)
                y_train = np.asarray([rows[i]["y"] for i in train_idx], dtype=np.float64)
                x_test = np.asarray([rows[i]["x"] for i in test_idx], dtype=np.float64)
                y_pred = baseline._predict_fold(
                    method,
                    x_train,
                    y_train,
                    x_test,
                    train_kwargs=baseline._method_train_kwargs(method, hidden=128, epochs=300, lr=1e-3),
                    rf_estimators=args.rf_estimators,
                    seed=args.seed + fold_i,
                )
                for local_i, row_i in enumerate(test_idx):
                    truth = float(rows[row_i]["y"])
                    prediction = float(y_pred[local_i])
                    per_run.append(
                        {
                            "path": str(Path(rows[row_i]["path"]).resolve()),
                            "fold": fold_i,
                            "target_logmean": truth,
                            "prediction": prediction,
                            "relative_error_percent": 100.0 * abs(truth - prediction) / max(abs(truth), 1e-9),
                            "xi_wall_s": float(rows[row_i]["xi_wall_s"]),
                        }
                    )
            method_results.append(
                {
                    "method": method,
                    "mean_error_percent": float(np.mean([r["relative_error_percent"] for r in per_run])),
                    "median_xi_wall_s": float(np.median([r["xi_wall_s"] for r in per_run])),
                    "per_run": per_run,
                }
            )

        probe_path = args.dataset_root / "models" / machine / "probe_dataset_pts_cpu.json"
        host_results.append(
            {
                "machine": machine,
                "oof_artifact": str((args.oof_root / f"round-1-oof-pts-cpu-{machine}.json").resolve()),
                "target": "log-mean of the host-specific canonical PTS-CPU profile intersection",
                "split_protocol": "exact five-fold run split from hybrid OOF fold_manifest",
                "probe_dataset_sha256": _sha256(probe_path),
                "runs": [
                    {"path": path_key, "sha256": _sha256(Path(path_key)), "target_logmean": float(row["y"])}
                    for path_key, row in row_by_path.items()
                ],
                "results": method_results,
            }
        )
        print(
            machine,
            ", ".join(
                f"{row['method']}={row['mean_error_percent']:.4f}%"
                for row in method_results
            ),
            file=sys.stderr,
        )

    result = {
        "schema": "moebench.pts_cpu.static_logmean_baselines.v1",
        "suite": "pts/cpu",
        "feature_scope": "static_hw_only, using existing execution-free proxy implementations",
        "split_protocol": "five-run, one-session leave-one-run-out; exact hybrid OOF folds and fold seeds",
        "seed": args.seed,
        "rf_estimators": args.rf_estimators,
        "hosts": host_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "machine": host["machine"],
                    "results": [
                        {"method": row["method"], "mean_error_percent": row["mean_error_percent"]}
                        for row in host["results"]
                    ],
                }
                for host in host_results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
