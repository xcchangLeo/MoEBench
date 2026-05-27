"""Train probe regressors (shared or per-profile for PTS)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from moebench.probe.training_data import (
    build_training_matrix,
    inverse_probe_label,
    probe_estimator_mode,
    probe_label_transform,
    transform_probe_label,
)
from moebench.probe.vectorizer import ProbeVectorizer


def _regressor_base(model_type: str, *, n_rows: int):
    min_child = max(2, min(8, n_rows // 8)) if n_rows > 0 else 2
    n_est = 120 if n_rows < 120 else 200
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=0.05,
            num_leaves=8 if n_rows < 120 else 31,
            min_child_samples=min_child,
        )
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=4 if n_rows < 120 else 6,
        min_child_weight=min_child if n_rows < 120 else 1,
    )


def train_probe_bundle(
    probe_dataset: dict[str, Any],
    *,
    model_type: str = "lightgbm",
    suite_aggregate: str = "geomean_index",
) -> dict[str, Any]:
    """Return a pickle-ready bundle (schema ``moebench.probe.model.v1``)."""
    from sklearn.multioutput import MultiOutputRegressor

    import numpy as np

    benchmark = str(probe_dataset.get("benchmark", "unixbench"))
    label_tf = probe_label_transform(benchmark)
    est_mode = probe_estimator_mode(benchmark)
    test_ids = list(probe_dataset.get("test_ids") or [])
    samples = list(probe_dataset.get("samples") or [])

    if est_mode == "per_test":
        vec = ProbeVectorizer()
        feat_names = list(vec.feature_names)
        estimators: dict[str, Any] = {}
        train_rows = 0
        for tid in test_ids:
            rows = [s for s in samples if str(s.get("test_id")) == tid]
            if len(rows) < 3:
                continue
            Xi = np.asarray(
                [vec.transform(s.get("probe") or {}) for s in rows],
                dtype=np.float64,
            )
            yi = np.asarray(
                [[transform_probe_label(float(s["label_value"]), label_tf)] for s in rows],
                dtype=np.float64,
            )
            base = _regressor_base(model_type, n_rows=len(rows))
            est = MultiOutputRegressor(base)
            est.fit(Xi, yi)
            estimators[tid] = est
            train_rows += len(rows)
        if len(estimators) < max(3, len(test_ids) // 2):
            raise ValueError(
                f"per_test training: only {len(estimators)} profiles had enough samples "
                f"(need ≥3 rows each). Collect more full runs with probe_collect.py."
            )
        return {
            "schema": "moebench.probe.model.v1",
            "model_type": model_type,
            "benchmark": benchmark,
            "pts_suite": probe_dataset.get("pts_suite"),
            "machine": probe_dataset.get("machine"),
            "test_ids": test_ids,
            "probe_duration_s": float(probe_dataset.get("probe_duration_s", 4.0)),
            "probe_mode": probe_dataset.get("probe_mode", "micro"),
            "estimator_mode": "per_test",
            "estimators": estimators,
            "estimator": None,
            "include_test_onehot": False,
            "feature_names": feat_names,
            "label_transform": label_tf,
            "suite_aggregate": suite_aggregate,
            "train_rows": train_rows,
        }

    X, y, _, feat_names = build_training_matrix(
        probe_dataset,
        include_test_onehot=True,
        label_transform=label_tf,
    )
    if len(X) < 4:
        raise ValueError(f"Need more probe samples for training, got {len(X)}")

    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    base = _regressor_base(model_type, n_rows=len(X))
    est = MultiOutputRegressor(base)
    est.fit(x_arr, y_arr)

    return {
        "schema": "moebench.probe.model.v1",
        "model_type": model_type,
        "benchmark": benchmark,
        "pts_suite": probe_dataset.get("pts_suite"),
        "machine": probe_dataset.get("machine"),
        "test_ids": test_ids,
        "probe_duration_s": float(probe_dataset.get("probe_duration_s", 4.0)),
        "probe_mode": probe_dataset.get("probe_mode", "micro"),
        "estimator_mode": "shared",
        "estimator": est,
        "estimators": None,
        "include_test_onehot": True,
        "feature_names": feat_names,
        "label_transform": label_tf,
        "suite_aggregate": suite_aggregate,
        "train_rows": len(X),
    }
