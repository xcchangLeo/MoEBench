"""Load MoEBench UnixBench router training data from dataset JSONs.

We treat each run JSON as one "query" (system xi). Each expert is one item.
For each (query, item) we build:
  - features: xi vector + one-hot expert id
  - label: relevance score derived from yi (expert index on full suite)
  - cost/time: derived from ti (for logging; not used for ranking loss by default)
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from moebench.router.feature_vectorizer import XiVectorizer


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def choose_y_block(yi: dict[str, Any]) -> dict[str, Any] | None:
    runs = yi.get("runs") or []
    if not runs:
        return None
    # Prefer parallel_copies == 32 (common in your ti keys); else smallest numeric; else first.
    def key(rb: dict[str, Any]) -> tuple[int, int]:
        pc = rb.get("parallel_copies")
        if pc == 32:
            return (0, 0)
        if isinstance(pc, int):
            return (1, pc)
        if pc is None:
            return (2, 10**9)
        return (3, 10**9)

    return sorted(runs, key=key)[0]


def extract_relevance(run_block: dict[str, Any], test_id: str) -> float | None:
    tests = run_block.get("tests") or {}
    tinfo = tests.get(test_id)
    if not tinfo:
        return None
    idx_detail = tinfo.get("index_detail") or {}
    # Prefer dimensionless "index" if present.
    r = _safe_float(idx_detail.get("index"))
    if r is not None:
        return r
    # fallback to raw score
    return _safe_float(tinfo.get("score"))


def extract_runtime_seconds(ds: dict[str, Any], test_id: str, parallel_copies: Any) -> float | None:
    by_test = (ds.get("ti") or {}).get("by_test_id") or {}
    test_entry = by_test.get(test_id) or {}
    if parallel_copies is not None:
        key = str(parallel_copies)
        return _safe_float(test_entry.get(key))
    # fallback: choose first available time key
    for k, v in test_entry.items():
        rt = _safe_float(v)
        if rt is not None:
            return rt
    return None


@dataclass
class RouterDataset:
    feature_names: list[str]
    expert_ids: list[str]
    expert_test_ids: list[str]
    # X rows: (#queries * #experts) x (#xi_features + #experts_onehot)
    X: list[list[float]]
    y: list[float]
    # group: number of items per query (always len(experts))
    group: list[int]
    query_ids: list[str]
    meta: dict[str, Any]


def load_unixbench_dataset_for_router(
    dataset_root: str | Path,
    *,
    glob_pattern: str = "*/run-*.json",
    xi_vectorizer: XiVectorizer | None = None,
) -> RouterDataset:
    root = Path(dataset_root).resolve()
    files = sorted(root.glob(glob_pattern))
    if not files:
        # allow passing dataset_root as repo-root (with dataset/ inside)
        files = sorted(Path(dataset_root).resolve().parent.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No run json files matched: {root} / {glob_pattern}")

    vec = xi_vectorizer or XiVectorizer()
    sample = json.load(open(files[0], "r", encoding="utf-8"))
    experts = sample.get("experts") or []
    if not experts:
        raise RuntimeError("No experts field found in dataset json")
    expert_ids = [e["expert_id"] for e in experts]
    expert_test_ids = [e["test_id"] for e in experts]
    n_experts = len(expert_ids)
    if n_experts < 2:
        raise RuntimeError("Need at least 2 experts for router training")

    xi_dim = len(vec.feature_names)

    X: list[list[float]] = []
    y: list[float] = []
    group: list[int] = []
    query_ids: list[str] = []

    for fp in files:
        ds = json.load(open(fp, "r", encoding="utf-8"))
        xi = ds.get("xi") or {}
        xi_vec = vec.transform(xi)  # length xi_dim
        run_block = choose_y_block(ds.get("yi") or {})
        if not run_block:
            continue
        pc = run_block.get("parallel_copies")
        suite_idx = _safe_float(run_block.get("system_benchmarks_index_score"))
        qid = fp.parent.name
        query_ids.append(qid)

        # build items for each expert
        for ei, tid in enumerate(expert_test_ids):
            label = extract_relevance(run_block, tid)
            if label is None:
                # still include item, but label=0.0 (can harm training; but keep stable)
                label = 0.0
            onehot = [0.0] * n_experts
            onehot[ei] = 1.0
            X.append(list(xi_vec) + onehot)
            y.append(float(label))
        group.append(n_experts)

    feature_names = vec.feature_names + [f"expert_onehot_{eid}" for eid in expert_ids]
    meta = {
        "num_files": len(files),
        "num_queries": len(group),
        "num_experts": n_experts,
        "xi_dim": xi_dim,
        "suite_relevance_source": "yi.tests[test_id].index_detail.index (fallback score)",
    }
    return RouterDataset(
        feature_names=feature_names,
        expert_ids=expert_ids,
        expert_test_ids=expert_test_ids,
        X=X,
        y=y,
        group=group,
        query_ids=query_ids,
        meta=meta,
    )


def load_phoronix_dataset_for_router(
    dataset_root: str | Path,
    *,
    glob_pattern: str = "*/run-*.json",
    exclude_session_names: frozenset[str] | None = None,
    pts_suite: str | None = None,
    xi_vectorizer: XiVectorizer | None = None,
) -> RouterDataset:
    """PTS runs: relevance = primary result ``value`` per profile (``yi.pts_export``)."""
    from moebench.phoronix.training_data import (
        collect_phoronix_run_paths,
        expert_test_ids_from_dataset,
        primary_value_from_export,
    )

    root = Path(dataset_root).resolve()
    files = collect_phoronix_run_paths(
        root,
        glob_pattern=glob_pattern,
        exclude_session_names=exclude_session_names or frozenset(),
        pts_suite=pts_suite,
    )

    vec = xi_vectorizer or XiVectorizer()
    sample = json.load(open(files[0], "r", encoding="utf-8"))
    base_ids = expert_test_ids_from_dataset(sample)
    common: set[str] | None = None
    for fp in files:
        ds = json.load(open(fp, "r", encoding="utf-8"))
        export = (ds.get("yi") or {}).get("pts_export") or {}
        ok = {t for t in base_ids if primary_value_from_export(export, t) is not None}
        common = ok if common is None else (common & ok)
    if not common:
        raise RuntimeError("No PTS profile has primary values in every run; check pts_export.")
    ordered = [t for t in base_ids if t in common]
    tid_to_eid = {e["test_id"]: e["expert_id"] for e in (sample.get("experts") or [])}
    expert_test_ids = ordered
    expert_ids = [tid_to_eid[t] for t in ordered]
    n_experts = len(expert_ids)
    if n_experts < 2:
        raise RuntimeError("Need at least 2 experts for router training")

    xi_dim = len(vec.feature_names)
    X: list[list[float]] = []
    y: list[float] = []
    group: list[int] = []
    query_ids: list[str] = []

    for fp in files:
        ds = json.load(open(fp, "r", encoding="utf-8"))
        xi = ds.get("xi") or {}
        xi_vec = vec.transform(xi)
        export = (ds.get("yi") or {}).get("pts_export") or {}
        qid = fp.parent.name
        query_ids.append(qid)

        for ei, tid in enumerate(expert_test_ids):
            label = primary_value_from_export(export, tid) if export else None
            if label is None:
                label = 0.0
            onehot = [0.0] * n_experts
            onehot[ei] = 1.0
            X.append(list(xi_vec) + onehot)
            y.append(float(label))
        group.append(n_experts)

    feature_names = vec.feature_names + [f"expert_onehot_{eid}" for eid in expert_ids]
    meta = {
        "benchmark": "phoronix",
        "pts_suite": pts_suite,
        "num_files": len(files),
        "num_queries": len(group),
        "num_experts": n_experts,
        "xi_dim": xi_dim,
        "suite_relevance_source": "yi.pts_export primary buffer value per profile",
    }
    return RouterDataset(
        feature_names=feature_names,
        expert_ids=expert_ids,
        expert_test_ids=expert_test_ids,
        X=X,
        y=y,
        group=group,
        query_ids=query_ids,
        meta=meta,
    )

