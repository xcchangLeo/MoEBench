"""Load probe models and predict subtests + suite."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

from moebench.probe.collector import collect_subtest_probe
from moebench.probe.suite_aggregate import aggregate_suite_index
from moebench.probe.training_data import inverse_probe_label
from moebench.probe.vectorizer import ProbeVectorizer

SCHEMA_PROBE_MODEL = "moebench.probe.model.v1"


def load_probe_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with open(p, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or blob.get("schema") != SCHEMA_PROBE_MODEL:
        raise ValueError(f"Not a probe model bundle: {p}")
    return blob


def _feature_row(
    probe: dict[str, Any],
    test_id: str,
    bundle: dict[str, Any],
) -> list[float]:
    vec = ProbeVectorizer()
    row = list(vec.transform(probe))
    if bundle.get("include_test_onehot", True):
        test_ids = list(bundle.get("test_ids") or [])
        oh = [0.0] * len(test_ids)
        if test_id in test_ids:
            oh[test_ids.index(test_id)] = 1.0
        row.extend(oh)
    return row


def _predict_raw(est: Any, row: list[float]) -> float:
    import numpy as np

    x = np.asarray([row], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        pred = est.predict(x)
    return float(np.asarray(pred).ravel()[0])


def predict_subtest(
    bundle: dict[str, Any],
    probe: dict[str, Any],
    test_id: str,
) -> float:
    label_tf = str(bundle.get("label_transform") or "none")
    est_mode = str(bundle.get("estimator_mode") or "shared")

    if est_mode == "per_test":
        estimators = bundle.get("estimators") or {}
        est = estimators.get(test_id)
        if est is None:
            raise ValueError(f"no per-test estimator for {test_id!r}")
        row = list(ProbeVectorizer().transform(probe))
    else:
        est = bundle.get("estimator")
        if est is None:
            raise ValueError("bundle missing estimator")
        row = _feature_row(probe, test_id, bundle)

    raw = _predict_raw(est, row)
    return inverse_probe_label(raw, label_tf)


def predict_suite_from_probes(
    bundle: dict[str, Any],
    *,
    test_ids: list[str] | None = None,
    duration_s: float | None = None,
    enable_ebpf: bool = True,
    aggregate_mode: str | None = None,
) -> dict[str, Any]:
    """
    Run a short probe per subtest, predict each index, aggregate suite score.
    """
    tids = list(test_ids or bundle.get("test_ids") or [])
    dur = float(duration_s if duration_s is not None else bundle.get("probe_duration_s", 4.0))
    agg = aggregate_mode or bundle.get("suite_aggregate", "geomean_index")

    sub_preds: dict[str, float] = {}
    probes: dict[str, Any] = {}
    mode = str(bundle.get("probe_mode", "micro"))
    benchmark = str(bundle.get("benchmark", "unixbench"))
    for tid in tids:
        probe = collect_subtest_probe(
            tid,
            duration_s=dur,
            enable_ebpf=enable_ebpf,
            benchmark=benchmark,
            probe_mode=mode,
        )
        probes[tid] = probe
        sub_preds[tid] = predict_subtest(bundle, probe, tid)

    suite = aggregate_suite_index(sub_preds, mode=agg)
    return {
        "subtest_index": sub_preds,
        "suite_index": suite,
        "probes": probes,
        "probe_duration_s": dur,
        "suite_aggregate": agg,
    }
