#!/usr/bin/env python3
"""Leakage-controlled run-level OOF evaluation of the PTS hybrid pipeline.

Each fold fits the router, probe estimator, and reconstructor using only the
training runs. Held-out runs are evaluated with their stored probe telemetry;
probe labels from held-out runs are never passed to a fold model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cv_helpers():
    path = ROOT / "scripts" / "paper_reconstruct_cv_extras.py"
    spec = importlib.util.spec_from_file_location("paper_reconstruct_cv_extras", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CV helpers: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_key(value: str | Path) -> tuple[str, str]:
    path = Path(value)
    return path.parent.name, path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machine", required=True)
    parser.add_argument(
        "--pts-suite",
        required=True,
        choices=("cpu", "pts/nvidia-gpu-compute"),
        help="PTS suite selector used by the dataset loader",
    )
    parser.add_argument("--probe-dataset", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-duration-s", type=float, default=4.0)
    parser.add_argument("--xi-overhead-s", type=float, default=3.0)
    parser.add_argument("--train-aug", type=int, default=20)
    parser.add_argument("--train-k-min", type=int, default=2)
    parser.add_argument("--train-k-max", type=int, default=6)
    parser.add_argument("--reconstructor", choices=("lightgbm", "xgboost"), default="lightgbm")
    parser.add_argument("--lgbm-estimators", type=int, default=30)
    parser.add_argument("--xgb-estimators", type=int, default=100)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from moebench.hybrid.eval import evaluate_single_run_offline
    from moebench.probe.model_train import train_probe_bundle
    from moebench.router.feature_vectorizer import XiVectorizer

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Restrict the canonical profile intersection to this host before loading
    # rows. Intersecting all hosts first can leave only profiles common to
    # every machine and silently change the suite cardinality.
    session_tag = "cpu" if args.pts_suite == "cpu" else "pts_nvidia-gpu-compute"
    glob_pattern = f"{args.machine}_{session_tag}_*/run-*.json"
    rows, test_ids = cv._load_phoronix_rows(
        args.dataset_root.resolve(),
        glob_pattern,
        args.pts_suite,
        "logmean",
    )
    rows = [row for row in rows if args.machine in Path(row["path"]).parent.name]
    if len(rows) < args.folds or args.folds < 2:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")
    if args.train_k_max > len(test_ids):
        args.train_k_max = len(test_ids)
    if args.train_k_min > args.train_k_max:
        raise SystemExit("train-k-min exceeds the number of PTS profiles")
    if args.top_k > len(test_ids):
        raise SystemExit("top-k exceeds the number of PTS profiles")

    probe_path = args.probe_dataset or (
        args.dataset_root / "models" / args.machine / (
            "probe_dataset_pts_cpu.json"
            if args.pts_suite == "cpu"
            else "probe_dataset_pts_gpu.json"
        )
    )
    probe_dataset = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe_dataset.get("benchmark") != "phoronix":
        raise SystemExit(f"Unexpected probe dataset benchmark: {probe_dataset.get('benchmark')}")
    if str(probe_dataset.get("pts_suite")) != args.pts_suite:
        raise SystemExit(
            f"Probe dataset suite {probe_dataset.get('pts_suite')!r} does not match {args.pts_suite!r}"
        )

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
    xi_vec = XiVectorizer()
    reports: list[dict[str, Any] | None] = [None] * len(rows)
    fold_manifest: list[dict[str, Any]] = []
    probe_index = {
        _run_key(sample["source_run"]) + (str(sample["test_id"]),): {
            "source_run": sample["source_run"],
            "test_id": sample["test_id"],
            "probe": sample.get("probe") or {},
        }
        for sample in probe_dataset.get("samples", [])
        if sample.get("source_run") and sample.get("test_id")
    }

    for fold_id, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_indices = [i for i in range(len(rows)) if i not in val_set]
        train_rows = [rows[i] for i in train_indices]
        train_keys = {_run_key(row["path"]) for row in train_rows}
        fold_probe_dataset = dict(probe_dataset)
        fold_probe_dataset["samples"] = [
            sample
            for sample in probe_dataset.get("samples", [])
            if sample.get("source_run") and _run_key(sample["source_run"]) in train_keys
        ]
        if not fold_probe_dataset["samples"]:
            raise RuntimeError(f"Fold {fold_id}: no training probe samples matched training runs")

        router = cv._train_fold_router(train_rows, test_ids, xi_vec, seed=args.seed + fold_id)
        probe_bundle = train_probe_bundle(fold_probe_dataset, model_type="lightgbm")

        train_rng = np.random.RandomState(args.seed + fold_id * 997)
        x_train, y_train = cv.rte.build_augmented_train_matrix_pts(
            train_rows,
            xi_vec,
            test_ids,
            train_rng,
            args.train_aug,
            args.train_k_min,
            args.train_k_max,
            False,
        )
        recon_bundle = cv.rte.sklearn_export_bundle(
            args.reconstructor,
            x_train,
            y_train,
            auto_install=False,
            lgbm_estimators=args.lgbm_estimators,
            xgb_estimators=args.xgb_estimators,
            log1p_partial_index=False,
            test_ids=test_ids,
            with_uncertainty=False,
            benchmark="phoronix",
        )
        recon_bundle["pts_suite"] = args.pts_suite

        for row_index in val_indices:
            row_meta = rows[row_index]
            run_path = Path(row_meta["path"])
            reports[row_index] = evaluate_single_run_offline(
                ds=row_meta["ds"],
                run_path=run_path,
                router_meta=router,
                recon_bundle=recon_bundle,
                probe_bundle=probe_bundle,
                probe_index=probe_index,
                top_k=args.top_k,
                probe_duration_s=args.probe_duration_s,
                xi_overhead_s=args.xi_overhead_s,
                benchmark="phoronix",
                recon_test_ids=test_ids,
            )

        fold_manifest.append(
            {
                "fold": fold_id,
                "train_runs": [rows[i]["path"] for i in train_indices],
                "heldout_runs": [rows[i]["path"] for i in val_indices],
                "probe_training_samples": len(fold_probe_dataset["samples"]),
                "router_fit_scope": router.get("router_fit_scope"),
                "router_fit_runs": router.get("router_fit_n_runs"),
            }
        )
        print(
            f"fold {fold_id + 1}/{len(folds)} complete "
            f"({len(train_rows)} train, {len(val_indices)} held out)",
            file=sys.stderr,
        )

    valid = [row for row in reports if row is not None and not row.get("skipped")]
    errors = [float(row["comparison"]["suite_relative_error"]) for row in valid]
    probe_times = [float(row["timing_seconds"]["probe_selected_subtests"]) for row in valid]
    hybrid_times = [float(row["timing_seconds"]["hybrid_wall_estimate"]) for row in valid]
    report = {
        "schema": "moebench.hybrid.run_level_oof.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "randomly shuffled run-level folds; router, probe, and reconstructor fit only on training runs",
        "machine": args.machine,
        "benchmark": "phoronix",
        "pts_suite": args.pts_suite,
        "pts_suite_target": "logmean",
        "folds": args.folds,
        "seed": args.seed,
        "top_k": args.top_k,
        "probe_duration_s": args.probe_duration_s,
        "xi_overhead_s": args.xi_overhead_s,
        "reconstructor": args.reconstructor,
        "reconstructor_estimators": args.lgbm_estimators if args.reconstructor == "lightgbm" else args.xgb_estimators,
        "probe_model": "lightgbm",
        "router_model": "fold-local MLP",
        "input_artifacts": {
            "probe_dataset": str(probe_path.resolve()),
            "probe_dataset_sha256": _sha256(probe_path),
            "runs": [
                {"path": row["path"], "sha256": _sha256(Path(row["path"]))}
                for row in rows
            ],
        },
        "fold_manifest": fold_manifest,
        "summary": {
            "n_runs": len(valid),
            "mean_relative_error_percent": float(np.mean(errors) * 100) if errors else None,
            "median_relative_error_percent": float(np.median(errors) * 100) if errors else None,
            "p95_relative_error_percent": float(np.percentile(errors, 95) * 100) if errors else None,
            "max_relative_error_percent": float(np.max(errors) * 100) if errors else None,
            "mean_probe_wall_s": float(np.mean(probe_times)) if probe_times else None,
            "mean_hybrid_wall_estimate_s": float(np.mean(hybrid_times)) if hybrid_times else None,
        },
        "per_run": reports,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
