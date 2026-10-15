#!/usr/bin/env python3
"""Leakage-controlled run-level OOF evaluation of the UnixBench hybrid pipeline."""

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
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from scripts.run_corrected_oof_unixbench import (
        _build_crossfitted_reconstructor_training_matrix,
        _crossfit_training_probe_predictions,
    )
    from moebench.hybrid.eval import evaluate_single_run_offline
    from moebench.probe.model_train import train_probe_bundle
    from moebench.router.feature_vectorizer import XiVectorizer

    from sklearn.multioutput import MultiOutputRegressor  # noqa: F401
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    rows, test_ids = cv._load_unixbench_rows(
        args.dataset_root.resolve(), f"*/run-*.json"
    )
    rows = [row for row in rows if args.machine in Path(row["path"]).parent.name]
    if len(rows) < args.folds or args.folds < 2:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")

    probe_path = args.probe_dataset or (
        args.dataset_root / "models" / args.machine / "probe_dataset_unixbench.json"
    )
    probe_dataset = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe_dataset.get("benchmark") != "unixbench":
        raise SystemExit(f"Unexpected probe dataset benchmark: {probe_dataset.get('benchmark')}")

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
    xi_vec = XiVectorizer()
    reports: list[dict[str, Any] | None] = [None] * len(rows)
    fold_manifest: list[dict[str, Any]] = []
    probe_samples = {
        _run_key(sample["source_run"]) + (str(sample["test_id"]),): sample
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
            sample for sample in probe_dataset.get("samples", [])
            if sample.get("source_run") and _run_key(sample["source_run"]) in train_keys
        ]
        if not fold_probe_dataset["samples"]:
            raise RuntimeError(f"Fold {fold_id}: no training probe samples matched training runs")

        router = cv._train_fold_router(
            train_rows, test_ids, xi_vec, seed=args.seed + fold_id
        )
        probe_bundle = train_probe_bundle(fold_probe_dataset, model_type="lightgbm")

        crossfit_predictions, inner_probe_folds = _crossfit_training_probe_predictions(
            train_rows,
            probe_dataset,
            test_ids,
            seed=args.seed + fold_id * 1009,
        )
        x_train, y_train = _build_crossfitted_reconstructor_training_matrix(
            train_rows,
            crossfit_predictions,
            probe_samples,
            xi_vec,
            test_ids,
            seed=args.seed + fold_id * 997,
            train_aug=args.train_aug,
            train_k_min=args.train_k_min,
            train_k_max=args.train_k_max,
        )
        recon_bundle = cv.rte.sklearn_export_bundle(
            args.reconstructor,
            x_train,
            y_train,
            auto_install=False,
            lgbm_estimators=200,
            xgb_estimators=200,
            log1p_partial_index=False,
            test_ids=test_ids,
            with_uncertainty=False,
            benchmark="unixbench",
        )
        fold_probe_index = {
            _run_key(sample["source_run"]) + (str(sample["test_id"]),): {
                "source_run": sample["source_run"],
                "test_id": sample["test_id"],
                "probe": sample.get("probe") or {},
            }
            for sample in probe_dataset.get("samples", [])
            if sample.get("source_run") and sample.get("test_id")
        }
        for row_index in val_indices:
            row_meta = rows[row_index]
            run_path = Path(row_meta["path"])
            ds = row_meta["ds"]
            per_run = evaluate_single_run_offline(
                ds=ds,
                run_path=run_path,
                router_meta=router,
                recon_bundle=recon_bundle,
                probe_bundle=probe_bundle,
                probe_index=fold_probe_index,
                top_k=args.top_k,
                probe_duration_s=args.probe_duration_s,
                xi_overhead_s=args.xi_overhead_s,
                benchmark="unixbench",
                recon_test_ids=test_ids,
            )
            reports[row_index] = per_run

        fold_manifest.append({
            "fold": fold_id,
            "train_runs": [rows[i]["path"] for i in train_indices],
            "heldout_runs": [rows[i]["path"] for i in val_indices],
            "probe_training_samples": len(fold_probe_dataset["samples"]),
            "router_fit_scope": router.get("router_fit_scope"),
            "router_fit_runs": router.get("router_fit_n_runs"),
            "reconstructor_fit_scope": "training runs only; partial scores are inner-OOF probe predictions and probe wall times",
            "inner_probe_folds": inner_probe_folds,
        })
        print(f"fold {fold_id + 1}/{len(folds)} complete ({len(train_rows)} train, {len(val_indices)} held out)", file=sys.stderr)

    valid = [row for row in reports if row is not None and not row.get("skipped")]
    errors = [float(row["comparison"]["suite_relative_error"]) for row in valid]
    times = [float(row["timing_seconds"]["hybrid_wall_estimate"]) for row in valid]
    report = {
        "schema": "moebench.hybrid.run_level_oof.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "randomly shuffled run-level folds; router and outer probe fit only on outer training runs; reconstructor train partial scores are inner-OOF probe predictions, evaluated with the outer-fold probe model",
        "machine": args.machine,
        "benchmark": "unixbench",
        "folds": args.folds,
        "seed": args.seed,
        "top_k": args.top_k,
        "probe_duration_s": args.probe_duration_s,
        "xi_overhead_s": args.xi_overhead_s,
        "reconstructor": args.reconstructor,
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
            "mean_hybrid_wall_estimate_s": float(np.mean(times)) if times else None,
            "mean_probe_wall_s": float(np.mean([
                row["timing_seconds"]["probe_selected_subtests"] for row in valid
            ])) if valid else None,
        },
        "per_run": reports,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
