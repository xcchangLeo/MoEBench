"""Build probe training rows from full-run datasets + live probes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from moebench.dataset_machines import resolve_glob_for_machine, resolve_training_machine
from moebench.probe.collector import ProbeMode, collect_subtest_probe
from moebench.probe.vectorizer import ProbeVectorizer
from moebench.reconstruct.data import (
    collect_unixbench_run_paths,
    extract_test_index_from_block,
    preferred_run_block,
)
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

SCHEMA_PROBE_DATASET = "moebench.probe.dataset.v1"


def probe_label_transform(benchmark: str) -> str:
    """PTS primary values span many orders of magnitude → train in log1p space."""
    return "log1p" if benchmark == "phoronix" else "none"


def probe_estimator_mode(benchmark: str) -> str:
    """One regressor per PTS profile; shared + one-hot for UnixBench."""
    return "per_test" if benchmark == "phoronix" else "shared"


def transform_probe_label(value: float, transform: str) -> float:
    if transform == "log1p":
        return float(math.log1p(max(0.0, value)))
    return float(value)


def inverse_probe_label(value: float, transform: str) -> float:
    if transform == "log1p":
        return float(math.expm1(float(value)))
    return max(0.0, float(value))


def _suite_logmean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(math.expm1(sum(math.log1p(max(0.0, v)) for v in values) / len(values)))


def label_index_from_unixbench_run(ds: dict[str, Any], test_id: str) -> float | None:
    rb = preferred_run_block(ds.get("yi") or {})
    if not rb:
        return None
    return extract_test_index_from_block(rb, test_id)


def label_suite_from_unixbench_run(ds: dict[str, Any]) -> float | None:
    rb = preferred_run_block(ds.get("yi") or {})
    if not rb:
        return None
    try:
        return float(rb.get("system_benchmarks_index_score"))
    except (TypeError, ValueError):
        return None


def label_value_from_pts_run(ds: dict[str, Any], test_id: str) -> float | None:
    from moebench.phoronix.training_data import primary_value_from_export

    export = (ds.get("yi") or {}).get("pts_export") or {}
    return primary_value_from_export(export, test_id)


def label_suite_from_pts_run(ds: dict[str, Any], test_ids: list[str]) -> float | None:
    vals: list[float] = []
    for tid in test_ids:
        v = label_value_from_pts_run(ds, tid)
        if v is None:
            return None
        vals.append(float(v))
    return _suite_logmean(vals)


def collect_probe_dataset(
    *,
    benchmark: str,
    dataset_root: str | Path,
    machine: str | None = None,
    glob_pattern: str | None = None,
    pts_suite: str | None = None,
    probe_duration_s: float = 4.0,
    enable_ebpf: bool = True,
    probe_mode: ProbeMode = "micro",
    live_probe: bool = True,
    test_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build probe dataset for unixbench or phoronix (PTS cpu / gpu via ``pts_suite``)."""
    m = resolve_training_machine(machine)
    glo = resolve_glob_for_machine(
        benchmark=benchmark,
        machine=m,
        glob_pattern=glob_pattern,
        pts_suite=pts_suite,
    )
    vec = ProbeVectorizer()
    samples: list[dict[str, Any]] = []

    tids: list[str] = []
    if benchmark == "unixbench":
        paths = collect_unixbench_run_paths(Path(dataset_root), glob_pattern=glo)
        tids = list(test_ids or INDEX_SUITE_TEST_IDS)
        for path in paths:
            with open(path, encoding="utf-8") as f:
                ds = json.load(f)
            suite = label_suite_from_unixbench_run(ds)
            for tid in tids:
                lab = label_index_from_unixbench_run(ds, tid)
                if lab is None:
                    continue
                probe = (
                    collect_subtest_probe(
                        tid,
                        duration_s=probe_duration_s,
                        enable_ebpf=enable_ebpf,
                        benchmark="unixbench",
                        probe_mode=probe_mode,
                    )
                    if live_probe
                    else {"test_id": tid, "synthetic": True}
                )
                samples.append(
                    _sample_row(path, tid, float(lab), suite, probe, vec, label_kind="index")
                )
    elif benchmark == "phoronix":
        from moebench.phoronix.training_data import (
            canonical_test_ids_from_runs,
            collect_phoronix_run_paths,
        )

        if not pts_suite:
            raise ValueError("pts_suite required for phoronix probe dataset (e.g. cpu, pts/nvidia-gpu-compute)")
        paths = collect_phoronix_run_paths(
            Path(dataset_root),
            glob_pattern=glo,
            pts_suite=pts_suite,
        )
        tids = list(test_ids) if test_ids else list(canonical_test_ids_from_runs(paths))
        experts_title: dict[str, str] = {}
        if paths:
            with open(paths[0], encoding="utf-8") as f:
                head = json.load(f)
            for ex in head.get("experts") or []:
                tid = str(ex.get("test_id") or "")
                if tid:
                    experts_title[tid] = str(ex.get("title") or "")

        for path in paths:
            with open(path, encoding="utf-8") as f:
                ds = json.load(f)
            suite = label_suite_from_pts_run(ds, tids)
            for tid in tids:
                lab = label_value_from_pts_run(ds, tid)
                if lab is None:
                    continue
                title = experts_title.get(tid)
                probe = (
                    collect_subtest_probe(
                        tid,
                        duration_s=probe_duration_s,
                        enable_ebpf=enable_ebpf,
                        benchmark="phoronix",
                        probe_mode=probe_mode,
                        pts_title=title,
                    )
                    if live_probe
                    else {"test_id": tid, "synthetic": True}
                )
                samples.append(
                    _sample_row(path, tid, float(lab), suite, probe, vec, label_kind="primary_value")
                )
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")

    return {
        "schema": SCHEMA_PROBE_DATASET,
        "benchmark": benchmark,
        "pts_suite": pts_suite,
        "machine": m,
        "probe_duration_s": float(probe_duration_s),
        "probe_mode": probe_mode,
        "enable_ebpf": enable_ebpf,
        "label_transform": probe_label_transform(benchmark),
        "estimator_mode": probe_estimator_mode(benchmark),
        "test_ids": tids,
        "num_samples": len(samples),
        "samples": samples,
    }


def _sample_row(
    path: Path,
    tid: str,
    label: float,
    suite: float | None,
    probe: dict[str, Any],
    vec: ProbeVectorizer,
    *,
    label_kind: str,
) -> dict[str, Any]:
    return {
        "test_id": tid,
        "source_run": str(path),
        "session": path.parent.name,
        "label_kind": label_kind,
        "label_value": float(label),
        "label_index": float(label) if label_kind == "index" else None,
        "label_suite": float(suite) if suite is not None else None,
        "probe": probe,
        "probe_vector": vec.transform(probe),
    }


def collect_probe_dataset_unixbench(**kwargs: Any) -> dict[str, Any]:
    kwargs.pop("benchmark", None)
    return collect_probe_dataset(benchmark="unixbench", **kwargs)


def build_training_matrix(
    probe_dataset: dict[str, Any],
    *,
    include_test_onehot: bool = True,
    label_transform: str | None = None,
) -> tuple[list[list[float]], list[float], list[str], list[str]]:
    """X, y (label), test_ids per row, feature_names."""
    label_tf = label_transform or str(
        probe_dataset.get("label_transform") or probe_label_transform(str(probe_dataset.get("benchmark", "unixbench")))
    )
    test_ids = list(probe_dataset.get("test_ids") or INDEX_SUITE_TEST_IDS)
    tid_to_i = {t: i for i, t in enumerate(test_ids)}
    n = len(test_ids)
    base_names = ProbeVectorizer().feature_names
    feat_names = list(base_names)
    if include_test_onehot:
        feat_names.extend([f"test_onehot_{t}" for t in test_ids])

    X: list[list[float]] = []
    y: list[float] = []
    row_tids: list[str] = []

    for s in probe_dataset.get("samples") or []:
        tid = str(s.get("test_id"))
        if tid not in tid_to_i:
            continue
        pv = s.get("probe_vector")
        if not pv:
            pv = ProbeVectorizer().transform(s.get("probe") or {})
        row = list(pv)
        if include_test_onehot:
            oh = [0.0] * n
            oh[tid_to_i[tid]] = 1.0
            row.extend(oh)
        lab = s.get("label_value")
        if lab is None:
            lab = s.get("label_index")
        if lab is None:
            continue
        X.append(row)
        y.append(transform_probe_label(float(lab), label_tf))
        row_tids.append(tid)

    return X, y, row_tids, feat_names
