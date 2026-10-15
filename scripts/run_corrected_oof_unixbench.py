#!/usr/bin/env python3
"""Cross-fitted probe-input training diagnostic for UnixBench reconstruction."""

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


def _crossfit_training_probe_predictions(
    train_rows: list[dict[str, Any]],
    probe_dataset: dict[str, Any],
    test_ids: list[str],
    *,
    seed: int,
    model_type: str = "lightgbm",
) -> tuple[dict[tuple[str, str], dict[str, float]], list[dict[str, Any]]]:
    from moebench.probe.inference import predict_subtest
    from moebench.probe.model_train import train_probe_bundle

    def run_key(value: str | Path) -> tuple[str, str]:
        path = Path(value)
        return path.parent.name, path.name

    row_keys = {run_key(row["path"]): row for row in train_rows}
    keys = list(row_keys)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    inner_folds = [keys[i : i + 1] for i in range(len(keys))]
    samples_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for sample in probe_dataset.get("samples", []):
        if sample.get("source_run") and sample.get("test_id"):
            samples_by_key.setdefault(run_key(sample["source_run"]), {})[
                str(sample["test_id"])
            ] = sample

    predictions: dict[tuple[str, str], dict[str, float]] = {}
    manifest = []
    for inner_id, heldout_keys in enumerate(inner_folds):
        heldout = set(heldout_keys)
        fit_keys = set(keys) - heldout
        fold_dataset = dict(probe_dataset)
        fold_dataset["samples"] = [
            sample
            for sample in probe_dataset.get("samples", [])
            if sample.get("source_run") and run_key(sample["source_run"]) in fit_keys
        ]
        if not fold_dataset["samples"]:
            raise RuntimeError(f"Inner fold {inner_id}: no probe training samples")
        bundle = train_probe_bundle(fold_dataset, model_type=model_type)
        for key in heldout_keys:
            by_test = samples_by_key.get(key, {})
            missing = [tid for tid in test_ids if tid not in by_test]
            if missing:
                raise RuntimeError(f"Missing OOF probe samples for {key}: {missing}")
            predictions[key] = {
                tid: predict_subtest(bundle, by_test[tid].get("probe") or {}, tid)
                for tid in test_ids
            }
        manifest.append(
            {
                "inner_fold": inner_id,
                "probe_fit_runs": [row_keys[key]["path"] for key in keys if key in fit_keys],
                "probe_prediction_runs": [row_keys[key]["path"] for key in heldout_keys],
                "probe_fit_samples": len(fold_dataset["samples"]),
            }
        )
    return predictions, manifest


def _build_crossfitted_reconstructor_training_matrix(
    train_rows: list[dict[str, Any]],
    probe_predictions: dict[tuple[str, str], dict[str, float]],
    probe_samples: dict[tuple[str, str, str], dict[str, Any]],
    xi_vec: Any,
    test_ids: list[str],
    *,
    seed: int,
    train_aug: int,
    train_k_min: int,
    train_k_max: int,
    benchmark: str = "unixbench",
    log1p_value: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if benchmark == "phoronix":
        from moebench.phoronix.training_data import build_partial_feature_row_from_pts_executed
    else:
        from moebench.reconstruct.data import build_partial_feature_row_from_executed_tests

    def run_key(value: str | Path) -> tuple[str, str]:
        path = Path(value)
        return path.parent.name, path.name

    rng = np.random.RandomState(seed)
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for meta in train_rows:
        key = run_key(meta["path"])
        run_predictions = probe_predictions[key]
        by_tid = {tid: probe_samples[key + (tid,)] for tid in test_ids}
        for _ in range(train_aug):
            k = rng.randint(train_k_min, train_k_max + 1)
            selected = rng.choice(test_ids, size=k, replace=False).tolist()
            executed = []
            for tid in selected:
                sample = by_tid[tid]
                probe = sample.get("probe") or {}
                wall = probe.get("wall_s") or probe.get("duration_s") or 4.0
                score = float(run_predictions[tid])
                if benchmark == "phoronix":
                    executed.append({"test_id": tid, "value": score, "time_s": float(wall)})
                else:
                    executed.append({"test_id": tid, "score": score, "index_detail": {"index": score}, "time_s": float(wall)})
            if benchmark == "phoronix":
                row = build_partial_feature_row_from_pts_executed(
                    meta["ds"].get("xi") or {}, executed, test_ids=tuple(test_ids),
                    xi_vectorizer=xi_vec, log1p_value=log1p_value,
                )
            else:
                row = build_partial_feature_row_from_executed_tests(
                    meta["ds"].get("xi") or {},
                    executed,
                    test_ids=tuple(test_ids),
                    xi_vectorizer=xi_vec,
                    log1p_index=log1p_value,
                )
            if row is None:
                raise RuntimeError(f"Could not build cross-fitted feature row for {meta['path']}")
            x_rows.append(row)
            y_rows.append(meta["y"].tolist())
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


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
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from moebench.hybrid.eval import evaluate_single_run_offline
    from moebench.probe.model_train import train_probe_bundle
    from moebench.router.feature_vectorizer import XiVectorizer

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    rows, test_ids = cv._load_unixbench_rows(args.dataset_root.resolve(), "*/run-*.json")
    rows = [row for row in rows if args.machine in Path(row["path"]).parent.name]
    if len(rows) < args.folds or args.folds < 2:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")
    probe_path = args.probe_dataset or (
        args.dataset_root / "models" / args.machine / "probe_dataset_unixbench.json"
    )
    probe_dataset = json.loads(probe_path.read_text(encoding="utf-8"))
    probe_samples = {
        (Path(sample["source_run"]).parent.name, Path(sample["source_run"]).name, str(sample["test_id"])): sample
        for sample in probe_dataset.get("samples", [])
        if sample.get("source_run") and sample.get("test_id")
    }

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
    xi_vec = XiVectorizer()
    reports: list[dict[str, Any] | None] = [None] * len(rows)
    fold_manifest = []

    for fold_id, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_indices = [i for i in range(len(rows)) if i not in val_set]
        train_rows = [rows[i] for i in train_indices]
        train_keys = {
            (Path(row["path"]).parent.name, Path(row["path"]).name)
            for row in train_rows
        }
        outer_probe_dataset = dict(probe_dataset)
        outer_probe_dataset["samples"] = [
            sample
            for sample in probe_dataset.get("samples", [])
            if sample.get("source_run")
            and (Path(sample["source_run"]).parent.name, Path(sample["source_run"]).name) in train_keys
        ]
        outer_probe_bundle = train_probe_bundle(outer_probe_dataset, model_type="lightgbm")
        crossfit_predictions, inner_manifest = _crossfit_training_probe_predictions(
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
        router = cv._train_fold_router(train_rows, test_ids, xi_vec, seed=args.seed + fold_id)
        for row_index in val_indices:
            row_meta = rows[row_index]
            run_path = Path(row_meta["path"])
            reports[row_index] = evaluate_single_run_offline(
                ds=row_meta["ds"],
                run_path=run_path,
                router_meta=router,
                recon_bundle=recon_bundle,
                probe_bundle=outer_probe_bundle,
                probe_index=probe_samples,
                top_k=args.top_k,
                probe_duration_s=args.probe_duration_s,
                xi_overhead_s=args.xi_overhead_s,
                benchmark="unixbench",
                recon_test_ids=test_ids,
            )

        fold_manifest.append(
            {
                "fold": fold_id,
                "train_runs": [row["path"] for row in train_rows],
                "heldout_runs": [rows[i]["path"] for i in val_indices],
                "outer_probe_fit_scope": "all outer training runs only",
                "outer_probe_training_samples": len(outer_probe_dataset["samples"]),
                "reconstructor_train_rows": int(x_train.shape[0]),
                "reconstructor_train_features": int(x_train.shape[1]),
                "reconstructor_probe_inputs": "four-fold inner OOF probe predictions; target run excluded from each probe fit",
                "inner_probe_folds": inner_manifest,
            }
        )
        print(f"fold {fold_id + 1}/{len(folds)} complete", file=sys.stderr)

    valid = [row for row in reports if row is not None and not row.get("skipped")]
    errors = [float(row["comparison"]["suite_relative_error"]) for row in valid]
    report = {
        "schema": "moebench.hybrid.unixbench_crossfit_reconstructor_oof.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine,
        "benchmark": "unixbench",
        "purpose": "diagnostic corrected for mismatch between oracle partial-score reconstructor training and probe-predicted held-out inference",
        "protocol": "outer run-level OOF; each outer training run receives reconstructor features from an inner probe model that excluded that run; outer held-out inference uses a probe model trained on all outer training runs",
        "folds": args.folds,
        "seed": args.seed,
        "top_k": args.top_k,
        "probe_duration_s": args.probe_duration_s,
        "xi_overhead_s": args.xi_overhead_s,
        "train_aug": args.train_aug,
        "train_k_range": [args.train_k_min, args.train_k_max],
        "reconstructor": args.reconstructor,
        "probe_model": "fold-local LightGBM",
        "input_artifacts": {
            "probe_dataset": str(probe_path.resolve()),
            "probe_dataset_sha256": _sha256(probe_path),
            "runs": [{"path": row["path"], "sha256": _sha256(Path(row["path"]))} for row in rows],
        },
        "fold_manifest": fold_manifest,
        "summary": {
            "n_runs": len(valid),
            "mean_relative_error_percent": float(np.mean(errors) * 100) if errors else None,
            "median_relative_error_percent": float(np.median(errors) * 100) if errors else None,
            "p95_relative_error_percent": float(np.percentile(errors, 95) * 100) if errors else None,
            "max_relative_error_percent": float(np.max(errors) * 100) if errors else None,
        },
        "per_run": reports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.machine}__crossfit_reconstructor_oof.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
