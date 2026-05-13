"""PTS dataset helpers for router / reconstruct training (cpu-style suite rows)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import math

import numpy as np


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def collect_phoronix_run_paths(
    dataset_root: str | Path,
    *,
    glob_pattern: str = "*/run-*.json",
    exclude_session_names: frozenset[str] | None = None,
    pts_suite: str | None = None,
) -> list[Path]:
    """Load PTS ``moebench.phoronix.dataset.v1`` run JSONs (excludes UnixBench sessions).

    When ``pts_suite`` is set (must match ``yi.suite``, e.g. ``pts/nvidia-gpu-compute``),
    only sessions collected for that PTS suite are returned so CPU and GPU runs can
    coexist under the same ``dataset-root``.
    """
    root = Path(dataset_root).resolve()
    paths = sorted(root.glob(glob_pattern))
    if not paths and (root / "dataset").is_dir():
        paths = sorted((root / "dataset").glob(glob_pattern))
    exclude = exclude_session_names or frozenset()
    out: list[Path] = []
    for p in paths:
        if p.parent.name in exclude:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                head = json.load(f)
        except Exception:
            continue
        if head.get("schema") != "moebench.phoronix.dataset.v1":
            continue
        if pts_suite is not None:
            yi = head.get("yi") or {}
            if yi.get("suite") != pts_suite:
                continue
        out.append(p)
    if not out:
        hint = f" (yi.suite == {pts_suite!r})" if pts_suite else ""
        raise FileNotFoundError(
            f"No PTS run JSON matched {root}/{glob_pattern} (schema moebench.phoronix.dataset.v1){hint}"
        )
    return out


def expert_test_ids_from_dataset(ds: dict[str, Any]) -> list[str]:
    """Ordered unique ``test_id`` values (older datasets may list the same profile more than once)."""
    ex = ds.get("experts") or []
    seen: set[str] = set()
    out: list[str] = []
    for e in ex:
        tid = str(e.get("test_id") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def primary_value_from_export(export: dict[str, Any], test_id: str) -> float | None:
    """
    Representative primary ``value`` for profile ``test_id`` in PTS JSON export.

    PTS may emit multiple ``results`` blocks with the same ``identifier`` (e.g. hashcat
    benchmarks like MD5/NTLM/SHA1). Returning the first seen value is order-dependent and
    causes unstable labels across runs. We aggregate all available values for the profile
    with log-mean (geometric-like mean in log1p space), robust to large scale differences.
    """
    vals: list[float] = []
    for _h, robj in (export.get("results") or {}).items():
        tid = str(robj.get("identifier") or _h)
        if tid != test_id:
            continue
        for buf in (robj.get("results") or {}).values():
            v = _safe_float(buf.get("value"))
            if v is not None:
                vals.append(float(v))
    if not vals:
        return None
    return float(math.expm1(sum(math.log1p(max(0.0, v)) for v in vals) / len(vals)))


def primary_time_from_pts_export(export: dict[str, Any], test_id: str) -> float | None:
    """Sum ``test_run_times`` in ``pts_export`` for profile ``test_id`` (all duplicate result blocks)."""
    total = 0.0
    for _h, robj in (export.get("results") or {}).items():
        tid = str(robj.get("identifier") or _h)
        if tid != test_id:
            continue
        for buf in (robj.get("results") or {}).values():
            trt = buf.get("test_run_times")
            if isinstance(trt, list) and trt:
                total += sum(float(x) for x in trt)
    return total if total > 0 else None


def time_seconds_for_profile(ds: dict[str, Any], test_id: str) -> float | None:
    by_test = (ds.get("ti") or {}).get("by_test_id") or {}
    ent = by_test.get(test_id) or {}
    t = _safe_float(ent.get("time_s_total"))
    if t is not None:
        return t
    return _safe_float(ent.get("execution_cost"))


def extract_targets_from_pts_dataset(
    ds: dict[str, Any],
    test_ids: tuple[str, ...],
) -> tuple[list[float], float] | None:
    """Per-profile primary values + suite mean (last target)."""
    export = (ds.get("yi") or {}).get("pts_export") or {}
    if not export:
        return None
    vals: list[float] = []
    for tid in test_ids:
        v = primary_value_from_export(export, tid)
        if v is None:
            return None
        vals.append(float(v))
    suite_mean = float(sum(vals) / max(len(vals), 1))
    return vals, suite_mean


def full_suite_wall_seconds_pts(
    ds: dict[str, Any],
    *,
    test_ids: tuple[str, ...],
) -> float | None:
    total = 0.0
    for tid in test_ids:
        t = time_seconds_for_profile(ds, tid)
        if t is None:
            return None
        total += t
    return total


def partial_wall_seconds_pts(
    ds: dict[str, Any],
    executed: list[str],
    *,
    test_ids: tuple[str, ...],
) -> float | None:
    total = 0.0
    for tid in executed:
        if tid not in test_ids:
            continue
        t = time_seconds_for_profile(ds, tid)
        if t is None:
            return None
        total += t
    return total


def build_partial_feature_row_pts(
    ds: dict[str, Any],
    executed_test_ids: set[str],
    *,
    test_ids: tuple[str, ...],
    xi_vectorizer: Any = None,
    log1p_value: bool = False,
) -> list[float] | None:
    """xi || (mask, value, time) * len(test_ids) for PTS primary values."""
    from moebench.router.feature_vectorizer import XiVectorizer

    vec = xi_vectorizer or XiVectorizer()
    xi = ds.get("xi") or {}
    xi_part = vec.transform(xi)
    export = (ds.get("yi") or {}).get("pts_export") or {}
    if not export:
        return None
    triplet: list[float] = []
    for tid in test_ids:
        if tid in executed_test_ids:
            v = primary_value_from_export(export, tid)
            if v is None:
                return None
            if log1p_value:
                v = math.log1p(max(0.0, float(v)))
            t = time_seconds_for_profile(ds, tid)
            if t is None:
                return None
            triplet.extend([1.0, float(v), float(t)])
        else:
            triplet.extend([0.0, 0.0, 0.0])
    return list(xi_part) + triplet


def build_partial_feature_row_from_pts_executed(
    xi: dict[str, Any],
    executed: list[dict[str, Any]],
    *,
    test_ids: tuple[str, ...],
    xi_vectorizer: Any = None,
    log1p_value: bool = False,
) -> list[float] | None:
    """``executed`` entries: ``test_id``, ``value`` (primary), ``time_s``."""
    from moebench.router.feature_vectorizer import XiVectorizer

    vec = xi_vectorizer or XiVectorizer()
    xi_part = vec.transform(xi)
    by_tid: dict[str, dict[str, Any]] = {}
    for e in executed:
        tid = e.get("test_id")
        if tid:
            by_tid[str(tid)] = e
    triplet: list[float] = []
    for tid in test_ids:
        if tid in by_tid:
            e = by_tid[tid]
            v = _safe_float(e.get("value"))
            if v is None:
                return None
            if log1p_value:
                v = math.log1p(max(0.0, float(v)))
            t = _safe_float(e.get("time_s"))
            if t is None:
                return None
            triplet.extend([1.0, float(v), float(t)])
        else:
            triplet.extend([0.0, 0.0, 0.0])
    return list(xi_part) + triplet


def canonical_test_ids_from_runs(paths: list[Path]) -> tuple[str, ...]:
    """Ordered profile ids present in every run (primary value non-null)."""
    with open(paths[0], encoding="utf-8") as f:
        ds0 = json.load(f)
    base_ids = expert_test_ids_from_dataset(ds0)
    common: set[str] | None = None
    for p in paths:
        with open(p, encoding="utf-8") as f:
            ds = json.load(f)
        export = (ds.get("yi") or {}).get("pts_export") or {}
        ok = {t for t in base_ids if primary_value_from_export(export, t) is not None}
        common = ok if common is None else (common & ok)
    if not common:
        raise RuntimeError("No profile has values in all runs; cannot build canonical suite.")
    ordered = [t for t in base_ids if t in common]
    if len(ordered) < 2:
        raise RuntimeError("Need at least 2 PTS experts after intersection")
    return tuple(ordered)


def build_augmented_train_matrix_pts(
    rows_meta: list[dict[str, Any]],
    vec: Any,
    test_ids: list[str],
    rng: np.random.RandomState,
    train_aug: int,
    train_k_min: int,
    train_k_max: int,
    log1p_value: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Random partial subsets of PTS profiles for reconstruction training."""
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for i in range(len(rows_meta)):
        meta = rows_meta[i]
        ds = meta["ds"]
        for _ in range(train_aug):
            k = rng.randint(train_k_min, train_k_max + 1)
            ex = set(rng.choice(test_ids, size=k, replace=False).tolist())
            row = build_partial_feature_row_pts(
                ds,
                ex,
                test_ids=tuple(test_ids),
                xi_vectorizer=vec,
                log1p_value=log1p_value,
            )
            if row is None:
                continue
            x_rows.append(row)
            y_rows.append(meta["y"].tolist())
    if not x_rows:
        raise RuntimeError("No training rows for PTS reconstruction model")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)
