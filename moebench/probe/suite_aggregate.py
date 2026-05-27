"""Aggregate per-subtest index predictions into suite-level score."""

from __future__ import annotations

import math
from typing import Any


def aggregate_suite_index(
    subtest_predictions: dict[str, float],
    *,
    mode: str = "geomean_index",
) -> float:
    """
    Combine subtest index predictions into one suite score.

    ``geomean_index``: exp(mean(log1p(index))) - 1  (robust to scale)
    ``mean_index``: arithmetic mean
    """
    vals = [max(0.0, float(v)) for v in subtest_predictions.values() if v is not None]
    if not vals:
        return 0.0
    if mode == "mean_index":
        return float(sum(vals) / len(vals))
    if mode == "geomean_index":
        return float(math.expm1(sum(math.log1p(v) for v in vals) / len(vals)))
    raise ValueError(f"unknown suite aggregate mode: {mode!r}")


def suite_error_report(
    predicted_suite: float,
    ground_truth_suite: float,
) -> dict[str, Any]:
    err = abs(predicted_suite - ground_truth_suite)
    rel = err / max(abs(ground_truth_suite), 1e-9)
    return {
        "predicted_suite": predicted_suite,
        "ground_truth_suite": ground_truth_suite,
        "abs_error": err,
        "relative_error": rel,
    }
