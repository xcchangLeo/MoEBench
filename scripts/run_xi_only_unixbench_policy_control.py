#!/usr/bin/env python3
"""Compare the UnixBench policy reports with a fold-matched xi-only control."""

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

POLICIES = ("learned", "fixed_cpu_mix", "fixed_io_mix", "fixed_first_k")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-aug", type=int, default=20)
    parser.add_argument("--train-k-min", type=int, default=2)
    parser.add_argument("--train-k-max", type=int, default=6)
    parser.add_argument("--reconstructor", choices=("lightgbm", "xgboost"), default="lightgbm")
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from moebench.reconstruct.inference import predict_from_partial
    from moebench.router.feature_vectorizer import XiVectorizer

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    rows, test_ids = cv._load_unixbench_rows(args.dataset_root.resolve(), "*/run-*.json")
    rows = [row for row in rows if args.machine in Path(row["path"]).parent.name]
    if len(rows) < args.folds or args.folds < 2:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")

    report_paths = {
        policy: args.policy_dir / f"{args.machine}__{policy}.json"
        for policy in POLICIES
    }
    reports = {policy: json.loads(path.read_text(encoding="utf-8")) for policy, path in report_paths.items()}
    for policy, report in reports.items():
        report_runs = [row.get("run_json") if row else None for row in report["per_run"]]
        data_runs = [row["path"] for row in rows]
        if report_runs != data_runs:
            raise SystemExit(f"Run ordering mismatch in {report_paths[policy]}")

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
    xi_vec = XiVectorizer()
    xi_dim = len(xi_vec.feature_names)
    xi_only_reports: list[dict[str, Any] | None] = [None] * len(rows)
    fold_manifest: list[dict[str, Any]] = []

    for fold_id, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_indices = [i for i in range(len(rows)) if i not in val_set]
        train_rows = [rows[i] for i in train_indices]
        train_rng = np.random.RandomState(args.seed + fold_id * 997)
        x_train, y_train = cv.rte.build_augmented_train_matrix(
            train_rows,
            xi_vec,
            test_ids,
            train_rng,
            args.train_aug,
            args.train_k_min,
            args.train_k_max,
            False,
        )
        x_train[:, xi_dim:] = 0.0
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

        for row_index in val_indices:
            meta = rows[row_index]
            xi = meta["ds"].get("xi") or {}
            prediction = predict_from_partial(recon_bundle, xi, [])
            predicted = float(prediction["suite_index"])
            actual = float(meta["y"][-1])
            xi_only_reports[row_index] = {
                "run_json": str(meta["path"]),
                "session": Path(meta["path"]).parent.name,
                "prediction": predicted,
                "ground_truth": actual,
                "suite_relative_error": abs(predicted - actual) / max(abs(actual), 1e-12),
            }

        fold_manifest.append(
            {
                "fold": fold_id,
                "train_runs": [rows[i]["path"] for i in train_indices],
                "heldout_runs": [rows[i]["path"] for i in val_indices],
                "train_augmented_rows": int(x_train.shape[0]),
                "xi_feature_count": xi_dim,
                "probe_feature_count_zeroed": int(x_train.shape[1] - xi_dim),
                "train_input_treatment": "random-subset feature rows generated as in policy OOF, then all probe mask/index/time columns zeroed",
                "heldout_input_treatment": "all probe mask/index/time columns zeroed",
            }
        )
        print(f"fold {fold_id + 1}/{len(folds)} complete", file=sys.stderr)

    errors = [r["suite_relative_error"] for r in xi_only_reports if r is not None]
    out = {
        "schema": "moebench.hybrid.unixbench_xi_only_control.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine,
        "benchmark": "unixbench",
        "purpose": "fold-matched xi-only control for the four policy OOF reports",
        "split_protocol": "same seeded random run-level folds as policy OOF",
        "model_protocol": "same labels, augmentation, reconstructor type, and training parameters; all probe mask/index/time input columns zeroed for train and evaluation",
        "folds": args.folds,
        "seed": args.seed,
        "top_k": args.top_k,
        "train_aug": args.train_aug,
        "train_k_range": [args.train_k_min, args.train_k_max],
        "reconstructor": args.reconstructor,
        "xi_feature_count": xi_dim,
        "probe_feature_count": 3 * len(test_ids),
        "source_policy_reports": {
            policy: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for policy, path in report_paths.items()
        },
        "fold_manifest": fold_manifest,
        "summary": {
            "n_runs": len(errors),
            "mean_relative_error_percent": float(np.mean(errors) * 100) if errors else None,
            "median_relative_error_percent": float(np.median(errors) * 100) if errors else None,
            "p95_relative_error_percent": float(np.percentile(errors, 95) * 100) if errors else None,
            "max_relative_error_percent": float(np.max(errors) * 100) if errors else None,
        },
        "per_run": xi_only_reports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.machine}__xi_only_control.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "summary": out["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
