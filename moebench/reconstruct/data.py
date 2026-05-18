"""Build feature rows and targets for UnixBench score reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from moebench.router.feature_vectorizer import XiVectorizer
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES

_TI_PARALLEL_KEY = str(UNIXBENCH_PARALLEL_COPIES)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def preferred_run_block(yi: dict[str, Any] | None) -> dict[str, Any] | None:
    """Prefer single-copy run (parallel_copies==1); fallback to smallest numeric copy."""
    if not yi:
        return None
    runs = yi.get("runs") or []
    if not runs:
        return None

    def sort_key(rb: dict[str, Any]) -> tuple[int, int]:
        pc = rb.get("parallel_copies")
        if pc == UNIXBENCH_PARALLEL_COPIES:
            return (0, 0)
        if isinstance(pc, int):
            return (1, pc)
        if pc is None:
            return (2, 10**9)
        return (3, 10**9)

    return sorted(runs, key=sort_key)[0]


def extract_test_index_from_block(run_block: dict[str, Any], test_id: str) -> float | None:
    tests = run_block.get("tests") or {}
    tinfo = tests.get(test_id)
    if not tinfo:
        return None
    idx_detail = tinfo.get("index_detail") or {}
    r = _safe_float(idx_detail.get("index"))
    if r is not None:
        return r
    return _safe_float(tinfo.get("score"))


def extract_targets_from_dataset(
    ds: dict[str, Any],
    *,
    test_ids: tuple[str, ...] = INDEX_SUITE_TEST_IDS,
) -> tuple[list[float], float] | None:
    """Return (per-test index values in test_ids order, system_benchmarks_index_score)."""
    rb = preferred_run_block(ds.get("yi") or {})
    if not rb:
        return None
    suite = _safe_float(rb.get("system_benchmarks_index_score"))
    if suite is None:
        return None
    ys: list[float] = []
    for tid in test_ids:
        v = extract_test_index_from_block(rb, tid)
        if v is None:
            return None
        ys.append(float(v))
    return ys, float(suite)


def _ti_for_test(
    ds: dict[str, Any],
    test_id: str,
    *,
    parallel_key: str = _TI_PARALLEL_KEY,
) -> float | None:
    by_test = (ds.get("ti") or {}).get("by_test_id") or {}
    entry = by_test.get(test_id) or {}
    return _safe_float(entry.get(parallel_key))


def full_suite_wall_seconds(
    ds: dict[str, Any],
    *,
    test_ids: tuple[str, ...] = INDEX_SUITE_TEST_IDS,
    parallel_key: str = _TI_PARALLEL_KEY,
) -> float | None:
    total = 0.0
    for tid in test_ids:
        t = _ti_for_test(ds, tid, parallel_key=parallel_key)
        if t is None:
            return None
        total += t
    return total


def partial_wall_seconds(
    ds: dict[str, Any],
    executed: Iterable[str],
    *,
    parallel_key: str = _TI_PARALLEL_KEY,
) -> float | None:
    total = 0.0
    for tid in executed:
        t = _ti_for_test(ds, tid, parallel_key=parallel_key)
        if t is None:
            return None
        total += t
    return total


def build_partial_feature_row(
    ds: dict[str, Any],
    executed_test_ids: set[str],
    *,
    test_ids: tuple[str, ...] = INDEX_SUITE_TEST_IDS,
    xi_vectorizer: XiVectorizer | None = None,
    parallel_key: str = _TI_PARALLEL_KEY,
    log1p_index: bool = False,
) -> list[float] | None:
    """
    Features: xi numeric vector, then for each test in fixed order:
      [mask, index_if_run_else_0, time_s_if_run_else_0]
    """
    vec = xi_vectorizer or XiVectorizer()
    xi = ds.get("xi") or {}
    xi_part = vec.transform(xi)
    rb = preferred_run_block(ds.get("yi") or {})
    if not rb:
        return None
    triplet: list[float] = []
    for tid in test_ids:
        if tid in executed_test_ids:
            idx = extract_test_index_from_block(rb, tid)
            if idx is None:
                return None
            if log1p_index:
                idx = math.log1p(max(0.0, idx))
            t = _ti_for_test(ds, tid, parallel_key=parallel_key)
            if t is None:
                return None
            triplet.extend([1.0, float(idx), float(t)])
        else:
            triplet.extend([0.0, 0.0, 0.0])
    return list(xi_part) + triplet


def build_partial_feature_row_from_executed_tests(
    xi: dict[str, Any],
    executed_tests: list[dict[str, Any]],
    *,
    test_ids: tuple[str, ...] = INDEX_SUITE_TEST_IDS,
    xi_vectorizer: XiVectorizer | None = None,
    log1p_index: bool = False,
) -> list[float] | None:
    """
    Same layout as ``build_partial_feature_row``: xi vector + (mask, index, time).
    ``executed_tests`` uses the same shape as router JSON ``executed.executed_tests``.
    """
    vec = xi_vectorizer or XiVectorizer()
    xi_part = vec.transform(xi)

    by_tid: dict[str, dict[str, Any]] = {}
    for e in executed_tests:
        tid = e.get("test_id")
        if not tid or e.get("missing"):
            continue
        by_tid[str(tid)] = e

    triplet: list[float] = []
    for tid in test_ids:
        if tid in by_tid:
            tinfo = by_tid[tid]
            idx_detail = tinfo.get("index_detail") or {}
            idx = _safe_float(idx_detail.get("index"))
            if idx is None:
                idx = _safe_float(tinfo.get("score"))
            if idx is None:
                return None
            if log1p_index:
                idx = math.log1p(max(0.0, float(idx)))
            t = _safe_float(tinfo.get("time_s"))
            if t is None:
                return None
            triplet.extend([1.0, float(idx), float(t)])
        else:
            triplet.extend([0.0, 0.0, 0.0])
    return list(xi_part) + triplet


def collect_unixbench_run_paths(
    dataset_root: str | Path,
    *,
    glob_pattern: str = "*/run-*.json",
) -> list[Path]:
    root = Path(dataset_root).resolve()
    paths = sorted(root.glob(glob_pattern))
    if not paths and (root / "dataset").is_dir():
        paths = sorted((root / "dataset").glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No run JSON matched {root / glob_pattern}")
    out: list[Path] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                head = json.load(f)
        except Exception:
            continue
        if head.get("schema") == "moebench.unixbench_router.run.v1":
            continue
        if head.get("schema") != "moebench.unixbench.dataset.v1":
            continue
        out.append(p)
    return out
