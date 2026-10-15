#!/usr/bin/env python3
"""Audit probe-feature use in the fold-local UnixBench reconstruction heads."""

from __future__ import annotations

import argparse
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


def _load_cv_helpers() -> Any:
    path = ROOT / "scripts" / "paper_reconstruct_cv_extras.py"
    spec = importlib.util.spec_from_file_location("paper_reconstruct_cv_extras", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CV helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_head_importance(estimator: Any) -> dict[str, Any]:
    split = np.asarray(estimator.feature_importances_, dtype=float)
    gain = np.asarray(estimator.booster_.feature_importance(importance_type="gain"), dtype=float)
    return {"split": split, "gain": gain}


def _shares(values: np.ndarray, xi_dim: int) -> tuple[float, float]:
    total = float(np.sum(values))
    if total <= 0:
        return 0.0, 0.0
    return float(np.sum(values[:xi_dim]) / total), float(np.sum(values[xi_dim:]) / total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machines", nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-aug", type=int, default=20)
    parser.add_argument("--train-k-min", type=int, default=2)
    parser.add_argument("--train-k-max", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cv = _load_cv_helpers()
    from moebench.router.feature_vectorizer import XiVectorizer

    all_folds: list[dict[str, Any]] = []
    for machine in args.machines:
        rows, test_ids = cv._load_unixbench_rows(args.dataset_root.resolve(), "*/run-*.json")
        rows = [row for row in rows if machine in Path(row["path"]).parent.name]
        if len(rows) != args.folds:
            raise RuntimeError(f"{machine}: expected {args.folds} rows, found {len(rows)}")

        rng = np.random.RandomState(args.seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        fold_indices = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
        xi_vec = XiVectorizer()
        xi_dim = len(xi_vec.feature_names)

        for fold_id, val_indices in enumerate(fold_indices):
            val_set = set(val_indices)
            train_rows = [rows[i] for i in range(len(rows)) if i not in val_set]
            fold_rng = np.random.RandomState(args.seed + fold_id * 997)
            x_train, y_train = cv.rte.build_augmented_train_matrix(
                train_rows,
                xi_vec,
                test_ids,
                fold_rng,
                args.train_aug,
                args.train_k_min,
                args.train_k_max,
                False,
            )
            from sklearn.multioutput import MultiOutputRegressor

            base = cv.rte._sklearn_base_estimator(
                "lightgbm",
                auto_install=False,
                lgbm_estimators=200,
                xgb_estimators=200,
            )
            base.set_params(n_jobs=1)
            multioutput = MultiOutputRegressor(base, n_jobs=1).fit(x_train, y_train)
            estimators = multioutput.estimators_
            suite = _one_head_importance(estimators[-1])
            suite_split_xi, suite_split_probe = _shares(suite["split"], xi_dim)
            suite_gain_xi, suite_gain_probe = _shares(suite["gain"], xi_dim)

            component_split_probe = []
            component_gain_probe = []
            for estimator in estimators[:-1]:
                head = _one_head_importance(estimator)
                _, split_probe = _shares(head["split"], xi_dim)
                _, gain_probe = _shares(head["gain"], xi_dim)
                component_split_probe.append(split_probe)
                component_gain_probe.append(gain_probe)

            all_folds.append(
                {
                    "machine": machine,
                    "fold": fold_id,
                    "train_runs": [Path(row["path"]).name for row in train_rows],
                    "heldout_runs": [Path(rows[i]["path"]).name for i in val_indices],
                    "augmented_rows": int(x_train.shape[0]),
                    "feature_count": int(x_train.shape[1]),
                    "xi_feature_count": xi_dim,
                    "probe_feature_count": int(x_train.shape[1] - xi_dim),
                    "suite_head": {
                        "probe_split_fraction": suite_split_probe,
                        "xi_split_fraction": suite_split_xi,
                        "probe_gain_fraction": suite_gain_probe,
                        "xi_gain_fraction": suite_gain_xi,
                        "n_probe_splits": int(np.sum(suite["split"][xi_dim:])),
                        "n_xi_splits": int(np.sum(suite["split"][:xi_dim])),
                    },
                    "component_heads": {
                        "mean_probe_split_fraction": float(np.mean(component_split_probe)),
                        "median_probe_split_fraction": float(np.median(component_split_probe)),
                        "mean_probe_gain_fraction": float(np.mean(component_gain_probe)),
                        "median_probe_gain_fraction": float(np.median(component_gain_probe)),
                        "heads_with_probe_gain": int(sum(value > 0 for value in component_gain_probe)),
                    },
                }
            )
        print(f"{machine}: {args.folds} folds complete", file=sys.stderr)

    def summarize(key: str, field: str) -> dict[str, Any]:
        values = [float(row[key][field]) for row in all_folds]
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "zero_count": int(sum(value == 0 for value in values)),
            "n": len(values),
        }

    summary = {
        "n_folds": len(all_folds),
        "suite_probe_split_fraction": summarize("suite_head", "probe_split_fraction"),
        "suite_probe_gain_fraction": summarize("suite_head", "probe_gain_fraction"),
        "component_mean_probe_gain_fraction": summarize("component_heads", "mean_probe_gain_fraction"),
        "component_median_probe_gain_fraction": summarize("component_heads", "median_probe_gain_fraction"),
        "component_heads_with_any_probe_gain": int(
            sum(row["component_heads"]["heads_with_probe_gain"] for row in all_folds)
        ),
        "component_head_count": 12 * len(all_folds),
    }
    result = {
        "schema": "moebench.unixbench.reconstructor_probe_sensitivity.v1",
        "scope": "five-fold run-level OOF reconstruction training across supplied machine identities",
        "training_protocol": {
            "seed": args.seed,
            "folds": args.folds,
            "train_aug": args.train_aug,
            "train_k_range": [args.train_k_min, args.train_k_max],
            "reconstructor": "200-estimator LightGBM MultiOutputRegressor",
            "feature_partition": "host context xi versus all per-component mask/index/time probe features",
        },
        "summary": summary,
        "folds": all_folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
