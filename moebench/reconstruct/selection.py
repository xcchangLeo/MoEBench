"""Active subtest selection using reconstruction uncertainty (information-gain proxy)."""

from __future__ import annotations

from typing import Any


def merge_executed_tests(
    existing: list[dict[str, Any]], new_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge by ``test_id``; later entries overwrite."""
    by: dict[str, dict[str, Any]] = {}
    for e in existing + new_entries:
        tid = e.get("test_id")
        if not tid or e.get("missing"):
            continue
        by[str(tid)] = e
    return list(by.values())


def pick_next_subtest_max_uncertainty(
    uncertainty_subtest: dict[str, float],
    executed_test_ids: set[str],
    candidate_ids: tuple[str, ...],
) -> str | None:
    """Choose an unexecuted subtest with largest predicted σ (proxy for information gain)."""
    best: str | None = None
    best_u = -1.0
    for tid in candidate_ids:
        if tid in executed_test_ids:
            continue
        u = float(uncertainty_subtest.get(tid, 0.0))
        if u > best_u:
            best_u = u
            best = tid
    return best


__all__ = ["merge_executed_tests", "pick_next_subtest_max_uncertainty"]
