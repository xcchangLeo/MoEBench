#!/usr/bin/env python3
"""Matched run-level OOF comparisons for learned and fixed UnixBench subsets.

Each fold fits one router, probe estimator, and reconstructor on training runs.
All policies share those fold models and receive probe-model predictions for
their selected subtests. Held-out full-run scores are used only for scoring.
"""

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

# OOF training is small and must remain reproducible on hosts where other
# jobs occupy the GPUs. Set this before importing torch or router modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLICY_IDS = (
    "learned",
    "dynamic_context",
    "static_context",
    "fixed_cpu_mix",
    "fixed_io_mix",
    "fixed_first_k",
    "random_top_k",
    "learned_corr_weighted",
    "learned_probe_quality",
    "learned_anchor",
    "learned_anchor_pool",
    "probe_anchor_top_k",
    "train_assoc_top_k",
    "train_assoc_diverse_top_k",
    "train_probe_reliable_top_k",
)
FIXED_CPU_IDS = ("dhry2reg", "whetstone-double", "syscall")
FIXED_IO_IDS = ("fstime", "fsbuffer", "fsdisk")
DYNAMIC_FEATURE_IDS = {
    "warmup_s",
    "proc_cpu_utilization_ratio",
    "proc_iowait_ratio",
    "vm_page_faults_per_sec",
    "vm_major_faults_per_sec",
    "vm_minor_faults_per_sec",
    "loadavg_1m",
    "loadavg_5m",
    "loadavg_15m",
    "runnable_tasks",
    "total_tasks",
    "last_pid",
    "memory_copy_gib_s",
    "memory_copy_elapsed_s",
    "perf_ipc",
    "perf_cycles_per_sec",
    "perf_instructions_per_sec",
    "perf_cache_misses_per_sec",
    "perf_branch_misses_per_sec",
    "perf_context_switches_per_sec",
    "perf_page_faults_per_sec",
    "gpu0_utilization_gpu_pct",
    "gpu0_utilization_memory_pct",
    "gpu0_memory_free_mib",
    "gpu0_memory_used_mib",
    "gpu_min_memory_free_mib",
    "gpu0_power_draw_w",
    "gpu0_temperature_gpu_c",
    "gpu0_clock_current_sm_mhz",
    "gpu0_clock_current_memory_mhz",
}


class _MaskedXiVectorizer:
    def __init__(self, base: Any, mask: list[float]) -> None:
        self.base = base
        self.mask = mask
        self.feature_names = list(base.feature_names)

    def transform(self, xi: dict[str, Any]) -> list[float]:
        return [float(value) * mask for value, mask in zip(self.base.transform(xi), self.mask)]


def _context_mask(feature_names: list[str], scope: str) -> list[float]:
    if scope == "dynamic":
        return [1.0 if name in DYNAMIC_FEATURE_IDS else 0.0 for name in feature_names]
    if scope == "static":
        return [0.0 if name in DYNAMIC_FEATURE_IDS else 1.0 for name in feature_names]
    return [1.0] * len(feature_names)


def _load_cv_helpers():
    path = ROOT / "scripts" / "paper_reconstruct_cv_extras.py"
    spec = importlib.util.spec_from_file_location("paper_reconstruct_cv_extras", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CV helpers: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_corrected_oof_helpers():
    path = ROOT / "scripts" / "run_corrected_oof_unixbench.py"
    spec = importlib.util.spec_from_file_location("run_corrected_oof_unixbench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load corrected OOF helpers: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_key(value: str | Path) -> tuple[str, str]:
    path = Path(value)
    return path.parent.name, path.name


def _select(policy: str, xi: dict[str, Any], test_ids: list[str], k: int,
            router: dict[str, Any],
            fold_selected_test_ids: list[str] | None = None) -> list[str]:
    if policy in (
        "learned",
        "dynamic_context",
        "static_context",
        "learned_corr_weighted",
        "learned_probe_quality",
        "learned_anchor",
    ):
        from moebench.hybrid.eval import router_select_test_ids

        return router_select_test_ids(router, xi, top_k=k)[1]
    if policy == "learned_anchor_pool":
        if fold_selected_test_ids is None:
            raise ValueError("learned_anchor_pool requires a training-fold candidate pool")
        from moebench.router.inference import predict_expert_scores

        scores, _, _, expert_test_ids = predict_expert_scores(router, xi)
        score_by_test = dict(zip(expert_test_ids, scores))
        return sorted(
            fold_selected_test_ids,
            key=lambda tid: (-float(score_by_test.get(tid, -float("inf"))), test_ids.index(tid)),
        )[:k]
    if policy == "fixed_cpu_mix":
        preferred = [tid for tid in FIXED_CPU_IDS if tid in test_ids]
    elif policy == "fixed_io_mix":
        preferred = [tid for tid in FIXED_IO_IDS if tid in test_ids]
    elif policy == "fixed_first_k":
        preferred = []
    elif policy == "random_top_k":
        # The caller supplies a fold/run-stable random ordering through the
        # optional selection override. No labels, timings, or probe features
        # are consulted by this baseline.
        if fold_selected_test_ids is None:
            raise ValueError("random_top_k requires a seeded selection")
        return list(fold_selected_test_ids[:k])
    elif policy in ("train_assoc_top_k", "train_assoc_diverse_top_k"):
        if fold_selected_test_ids is None:
            raise ValueError(f"{policy} requires a training-fold selection")
        return list(fold_selected_test_ids[:k])
    elif policy == "train_probe_reliable_top_k":
        if fold_selected_test_ids is None:
            raise ValueError(f"{policy} requires a training-fold selection")
        return list(fold_selected_test_ids[:k])
    elif policy == "probe_anchor_top_k":
        if fold_selected_test_ids is None:
            raise ValueError("probe_anchor_top_k requires probe-derived selection")
        return list(fold_selected_test_ids[:k])
    else:
        raise ValueError(f"Unknown subset policy {policy!r}")
    ordered = preferred + [tid for tid in test_ids if tid not in preferred]
    return ordered[:k]


def _train_assoc_selection(
    train_rows: list[dict[str, Any]], test_ids: list[str], top_k: int
) -> dict[str, Any]:
    """Select a fixed subset by absolute Spearman association with suite index."""
    from scipy.stats import spearmanr

    targets = np.asarray([float(row["y"][-1]) for row in train_rows], dtype=np.float64)
    ranked: list[tuple[float, int, str]] = []
    correlations: dict[str, float | None] = {}
    for index, test_id in enumerate(test_ids):
        values = np.asarray([float(row["y"][index]) for row in train_rows], dtype=np.float64)
        rho = float(spearmanr(values, targets).statistic)
        if not np.isfinite(rho):
            correlations[test_id] = None
            rho = 0.0
        else:
            correlations[test_id] = rho
        ranked.append((-abs(rho), index, test_id))
    ranked.sort()
    selected = [test_id for _, _, test_id in ranked[:top_k]]
    return {
        "selected_test_ids": selected,
        "training_fold_spearman_rho": correlations,
        "selection_method": "absolute Spearman correlation between each component index and official suite index across outer-training runs; descending absolute value, test-id order breaks ties",
        "n_training_runs": len(train_rows),
    }


def _train_assoc_diverse_selection(
    train_rows: list[dict[str, Any]], test_ids: list[str], top_k: int
) -> dict[str, Any]:
    """Select predictive, nonredundant experts using training-fold ranks only."""
    from itertools import combinations
    from scipy.stats import spearmanr

    targets = np.asarray([float(row["y"][-1]) for row in train_rows], dtype=np.float64)
    values = np.asarray(
        [[float(row["y"][j]) for row in train_rows] for j in range(len(test_ids))],
        dtype=np.float64,
    )

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        value = float(spearmanr(a, b).statistic)
        return value if np.isfinite(value) else 0.0

    suite_rho = [rho(values[j], targets) for j in range(len(test_ids))]
    pair_rho = np.eye(len(test_ids), dtype=np.float64)
    for i in range(len(test_ids)):
        for j in range(i + 1, len(test_ids)):
            pair_rho[i, j] = pair_rho[j, i] = rho(values[i], values[j])

    best_subset: tuple[int, ...] | None = None
    best_objective = -float("inf")
    for subset in combinations(range(len(test_ids)), top_k):
        relevance = float(np.mean([abs(suite_rho[i]) for i in subset]))
        pairs = [abs(pair_rho[i, j]) for i, j in combinations(subset, 2)]
        redundancy = float(np.mean(pairs)) if pairs else 0.0
        objective = relevance - redundancy
        if objective > best_objective:
            best_subset = subset
            best_objective = objective

    if best_subset is None:
        raise RuntimeError("No candidate expert subset was evaluated")
    pair_values = [abs(pair_rho[i, j]) for i, j in combinations(best_subset, 2)]
    return {
        "selected_test_ids": [test_ids[i] for i in best_subset],
        "training_fold_spearman_rho_with_suite": {
            test_ids[i]: suite_rho[i] for i in range(len(test_ids))
        },
        "mean_abs_suite_association": float(np.mean([abs(suite_rho[i]) for i in best_subset])),
        "mean_abs_pairwise_redundancy": float(np.mean(pair_values)) if pair_values else 0.0,
        "objective": float(best_objective),
        "selection_method": (
            "exhaustive Top-K maximizing mean absolute training-fold Spearman "
            "association with suite index minus mean absolute pairwise expert association"
        ),
        "n_training_runs": len(train_rows),
    }


def _train_fold_corr_router(
    train_rows: list[dict[str, Any]],
    test_ids: list[str],
    xi_vec: Any,
    *,
    seed: int,
    probe_reliability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Train an xi-conditioned router against score rank weighted by suite association."""
    from scipy.stats import rankdata, spearmanr
    import torch
    from moebench.router.dataset_loader import RouterDataset
    from moebench.router.neural_routers import train_pointwise_mlp

    torch.manual_seed(seed)
    n_experts = len(test_ids)
    xi_dim = len(xi_vec.feature_names)
    targets = np.asarray([float(row["y"][-1]) for row in train_rows], dtype=np.float64)
    expert_values = np.asarray(
        [[float(row["y"][j]) for row in train_rows] for j in range(n_experts)],
        dtype=np.float64,
    )
    assoc = np.zeros(n_experts, dtype=np.float64)
    for j in range(n_experts):
        rho = float(spearmanr(expert_values[j], targets).statistic)
        assoc[j] = abs(rho) if np.isfinite(rho) else 0.0

    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for row in train_rows:
        xi = xi_vec.transform(row["ds"].get("xi") or {})
        component_scores = np.asarray(row["y"][:-1], dtype=np.float64)
        ranked = (rankdata(component_scores, method="average") - 1.0) / max(n_experts - 1, 1)
        for expert_index in range(n_experts):
            onehot = [0.0] * n_experts
            onehot[expert_index] = 1.0
            x_rows.append(list(xi) + onehot)
            reliability = float((probe_reliability or {}).get(test_ids[expert_index], 1.0))
            y_rows.append(float(ranked[expert_index] * assoc[expert_index] * reliability))

    bundle = train_pointwise_mlp(
        RouterDataset(
            feature_names=list(xi_vec.feature_names)
            + [f"expert_onehot_e_{i + 1:03d}" for i in range(n_experts)],
            expert_ids=[f"e_{i + 1:03d}" for i in range(n_experts)],
            expert_test_ids=list(test_ids),
            X=x_rows,
            y=y_rows,
            group=[n_experts] * len(train_rows),
            query_ids=[str(row["path"]) for row in train_rows],
            meta={"xi_dim": xi_dim, "num_queries": len(train_rows)},
        ),
        label_transform="none",
        hidden=64,
        epochs=200,
        lr=1e-3,
    )
    bundle.update(
        {
            "router_fit_scope": "training_fold_only",
            "router_fit_n_runs": len(train_rows),
            "router_fit_paths": [str(row["path"]) for row in train_rows],
            "relevance_target": "within-run normalized expert rank times absolute training-fold Spearman association with suite index",
            "training_fold_suite_association": {
                test_ids[i]: float(assoc[i]) for i in range(n_experts)
            },
            "training_fold_probe_reliability": {
                tid: float((probe_reliability or {}).get(tid, 1.0)) for tid in test_ids
            },
        }
    )
    return bundle


def _train_fold_anchor_router(
    train_rows: list[dict[str, Any]],
    test_ids: list[str],
    xi_vec: Any,
    *,
    seed: int,
    probe_reliability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Train a router for components representative of the suite geometric mean."""
    import torch
    from moebench.router.dataset_loader import RouterDataset
    from moebench.router.neural_routers import train_pointwise_mlp

    torch.manual_seed(seed)
    n_experts = len(test_ids)
    xi_dim = len(xi_vec.feature_names)
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for row in train_rows:
        xi = xi_vec.transform(row["ds"].get("xi") or {})
        suite = max(float(row["y"][-1]), 1e-9)
        for expert_index, tid in enumerate(test_ids):
            onehot = [0.0] * n_experts
            onehot[expert_index] = 1.0
            component = max(float(row["y"][expert_index]), 1e-9)
            closeness = -abs(float(np.log(component / suite)))
            reliability = float((probe_reliability or {}).get(tid, 1.0))
            x_rows.append(list(xi) + onehot)
            y_rows.append(closeness * reliability)
    bundle = train_pointwise_mlp(
        RouterDataset(
            feature_names=list(xi_vec.feature_names)
            + [f"expert_onehot_e_{i + 1:03d}" for i in range(n_experts)],
            expert_ids=[f"e_{i + 1:03d}" for i in range(n_experts)],
            expert_test_ids=list(test_ids),
            X=x_rows,
            y=y_rows,
            group=[n_experts] * len(train_rows),
            query_ids=[str(row["path"]) for row in train_rows],
            meta={"xi_dim": xi_dim, "num_queries": len(train_rows)},
        ),
        label_transform="none",
        hidden=64,
        epochs=200,
        lr=1e-3,
    )
    bundle.update({
        "router_fit_scope": "training_fold_only",
        "router_fit_n_runs": len(train_rows),
        "router_fit_paths": [str(row["path"]) for row in train_rows],
        "relevance_target": "negative absolute log component-to-suite ratio times inner-OOF probe reliability",
    })
    return bundle


def _probe_reliability_weights(
    train_rows: list[dict[str, Any]],
    train_probe_predictions: dict[tuple[str, str], dict[str, float]],
    test_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Rank experts by inner-OOF relative prediction error; smaller is more reliable."""
    by_test: dict[str, list[float]] = {tid: [] for tid in test_ids}
    for row in train_rows:
        predictions = train_probe_predictions[_run_key(row["path"])]
        for index, tid in enumerate(test_ids):
            truth = float(row["y"][index])
            pred = float(predictions[tid])
            by_test[tid].append(abs(pred - truth) / max(abs(truth), 1e-9))
    errors = {tid: float(np.median(values)) for tid, values in by_test.items()}
    ordered = sorted(test_ids, key=lambda tid: (errors[tid], tid))
    n_experts = len(ordered)
    weights = {
        tid: float((n_experts - rank) / n_experts)
        for rank, tid in enumerate(ordered)
    }
    return weights, errors


def _probe_prediction_calibration_factor(
    train_rows: list[dict[str, Any]],
    train_probe_predictions: dict[tuple[str, str], dict[str, float]],
    test_ids: list[str],
) -> float:
    """Robust global multiplicative calibration from inner-OOF probe predictions."""
    log_ratios: list[float] = []
    for row in train_rows:
        key = _run_key(row["path"])
        predictions = train_probe_predictions[key]
        for expert_index, test_id in enumerate(test_ids):
            truth = float(row["y"][expert_index])
            estimate = float(predictions[test_id])
            if truth > 0.0 and estimate > 0.0:
                log_ratios.append(float(np.log(truth / estimate)))
    if not log_ratios:
        return 1.0
    return float(np.exp(np.median(np.asarray(log_ratios, dtype=np.float64))))


def _probe_prediction_calibration_factors(
    train_rows: list[dict[str, Any]],
    train_probe_predictions: dict[tuple[str, str], dict[str, float]],
    test_ids: list[str],
) -> dict[str, float]:
    """Estimate a robust multiplicative correction separately per probe target."""
    ratios: dict[str, list[float]] = {tid: [] for tid in test_ids}
    for row in train_rows:
        predictions = train_probe_predictions[_run_key(row["path"])]
        for index, tid in enumerate(test_ids):
            truth = float(row["y"][index])
            estimate = float(predictions[tid])
            if truth > 0.0 and estimate > 0.0:
                ratios[tid].append(float(np.log(truth / estimate)))
    return {
        tid: float(np.exp(np.median(values))) if values else 1.0
        for tid, values in ratios.items()
    }


def _evaluate_policy_run(
    *,
    policy: str,
    ds: dict[str, Any],
    run_path: Path,
    router: dict[str, Any],
    probe_bundle: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_index: dict[tuple[str, str, str], dict[str, Any]],
    test_ids: list[str],
    top_k: int,
    probe_duration_s: float,
    xi_overhead_s: float,
    fold_selected_test_ids: list[str] | None = None,
    random_selection: list[str] | None = None,
    partial_source: str = "probe_predictions",
) -> dict[str, Any]:
    from moebench.hybrid.eval import _suite_errors, probe_predictions_to_executed_tests
    from moebench.probe.inference import predict_subtest
    from moebench.probe.training_data import (
        label_index_from_unixbench_run,
        label_suite_from_unixbench_run,
    )
    from moebench.reconstruct.inference import predict_from_partial
    from moebench.reconstruct.data import full_suite_wall_seconds

    xi = ds.get("xi") or {}
    locator = _run_key(run_path)
    probe_anchor_selection: list[str] | None = None
    all_probe_wall_s = 0.0
    if policy == "probe_anchor_top_k":
        from moebench.probe.inference import predict_subtest

        predicted_scores: dict[str, float] = {}
        for tid in test_ids:
            sample = probe_index.get(locator + (tid,))
            if sample is None:
                continue
            probe = sample.get("probe") or {}
            predicted_scores[tid] = max(float(predict_subtest(probe_bundle, probe, tid)), 1e-9)
            all_probe_wall_s += float(probe.get("wall_s") or probe.get("duration_s") or probe_duration_s)
        if len(predicted_scores) < top_k:
            return {
                "run_json": str(run_path),
                "skipped": True,
                "reason": "insufficient_probe_samples_for_probe_anchor_selection",
            }
        log_suite_anchor = float(np.mean([np.log(value) for value in predicted_scores.values()]))
        probe_anchor_selection = sorted(
            predicted_scores,
            key=lambda tid: (abs(float(np.log(predicted_scores[tid])) - log_suite_anchor), test_ids.index(tid)),
        )[:top_k]
    selected = _select(
        policy,
        xi,
        test_ids,
        top_k,
        router,
        probe_anchor_selection
        if policy == "probe_anchor_top_k"
        else random_selection if policy == "random_top_k" else fold_selected_test_ids,
    )
    sub_preds: dict[str, float] = {}
    probe_walls: dict[str, float] = {}
    if partial_source == "heldout_observed":
        from moebench.reconstruct.data import _ti_for_test

        for tid in selected:
            truth = label_index_from_unixbench_run(ds, tid)
            wall = _ti_for_test(ds, tid)
            if truth is None or wall is None:
                return {"run_json": str(run_path), "skipped": True, "reason": "missing_observed_partial"}
            sub_preds[tid] = float(truth)
            probe_walls[tid] = float(wall)
        executed = [
            {
                "test_id": tid,
                "time_s": probe_walls[tid],
                "index_detail": {"index": sub_preds[tid]},
            }
            for tid in selected
        ]
    else:
        for tid in selected:
            sample = probe_index.get(locator + (tid,))
            if sample is None:
                return {
                    "run_json": str(run_path),
                    "skipped": True,
                    "reason": "missing_probe_sample",
                    "missing_test_id": tid,
                }
            probe = sample.get("probe") or {}
            sub_preds[tid] = predict_subtest(probe_bundle, probe, tid)
            wall = probe.get("wall_s") or probe.get("duration_s")
            if wall is not None:
                probe_walls[tid] = float(wall)
        executed = probe_predictions_to_executed_tests(
            selected,
            sub_preds,
            probe_duration_s,
            benchmark="unixbench",
            probe_wall_by_tid=probe_walls,
        )
    prediction = predict_from_partial(recon_bundle, xi, executed)
    actual = label_suite_from_unixbench_run(ds)
    if actual is None:
        return {"run_json": str(run_path), "skipped": True, "reason": "missing_suite_label"}

    predicted = float(prediction["suite_index"])
    xi_only_prediction = predict_from_partial(recon_bundle, xi, [])
    xi_only_predicted = float(xi_only_prediction["suite_index"])
    actual_subtests = {
        tid: float(label_index_from_unixbench_run(ds, tid))
        for tid in selected
        if label_index_from_unixbench_run(ds, tid) is not None
    }
    probe_component_relative_errors = {
        tid: abs(float(sub_preds[tid]) - truth) / max(abs(truth), 1e-9)
        for tid, truth in actual_subtests.items()
    }
    probe_wall = sum(probe_walls.get(tid, probe_duration_s) for tid in selected)
    selection_probe_wall = all_probe_wall_s if policy == "probe_anchor_top_k" else probe_wall
    xi_elapsed = float(xi_overhead_s)
    paired_manifest = run_path.parent / "paired_capture_manifest.json"
    if paired_manifest.is_file():
        try:
            xi_elapsed = float(json.loads(paired_manifest.read_text(encoding="utf-8")).get("xi_elapsed_s", xi_overhead_s))
        except (OSError, ValueError, TypeError):
            pass
    full_suite_wall = full_suite_wall_seconds(ds)
    random_subset_errors: list[float] = []
    from itertools import combinations
    from moebench.reconstruct.data import _ti_for_test

    for subset in combinations(test_ids, top_k):
        if partial_source == "heldout_observed":
            subset_executed = [
                {
                    "test_id": tid,
                    "time_s": float(_ti_for_test(ds, tid) or probe_duration_s),
                    "index_detail": {
                        "index": float(label_index_from_unixbench_run(ds, tid) or 0.0)
                    },
                }
                for tid in subset
            ]
        else:
            subset_pred = {}
            subset_walls = {}
            for tid in subset:
                sample = probe_index.get(locator + (tid,))
                if sample is None:
                    continue
                probe = sample.get("probe") or {}
                subset_pred[tid] = predict_subtest(probe_bundle, probe, tid)
                subset_walls[tid] = float(
                    probe.get("wall_s") or probe.get("duration_s") or probe_duration_s
                )
            subset_executed = probe_predictions_to_executed_tests(
                list(subset),
                subset_pred,
                probe_duration_s,
                benchmark="unixbench",
                probe_wall_by_tid=subset_walls,
            )
        subset_prediction = float(predict_from_partial(recon_bundle, xi, subset_executed)["suite_index"])
        random_subset_errors.append(
            abs(subset_prediction - float(actual)) / max(abs(float(actual)), 1e-9)
        )
    return {
        "run_json": str(run_path),
        "session": run_path.parent.name,
        "policy": policy,
        "selection": {
            "selected_test_ids": selected,
            "selection_inputs": (
                "fresh per-subtest micro-probe predictions; select components nearest their predicted geometric mean"
                if policy == "probe_anchor_top_k"
                else
                "pre-run xi, including fresh dynamic context and static host descriptors"
                if policy == "learned"
                else "pre-run dynamic xi only; static features zero-masked"
                if policy == "dynamic_context"
                else "pre-run static xi only; dynamic features zero-masked"
                if policy == "static_context"
                else "outer-training-run component labels and official suite index"
                if policy in ("train_assoc_top_k", "train_assoc_diverse_top_k")
                else "pre-run xi; router relevance includes training-fold expert-to-suite association and probe reliability"
                if policy == "learned_probe_quality"
                else "pre-run xi; router relevance includes training-fold expert-to-suite association"
                if policy == "learned_corr_weighted"
                else "Xi-conditioned anchor router restricted to a training-fold association-and-diversity candidate pool"
                if policy == "learned_anchor_pool"
                else "fixed test-id list"
            ),
        },
        "probe": {
            "predicted_subtest": sub_preds,
            "heldout_actual_subtest": actual_subtests,
            "heldout_component_relative_error": probe_component_relative_errors,
            "probe_duration_budget_s": probe_duration_s,
            "historical_probe_wall_sum_s": probe_wall,
            "selection_probe_wall_sum_s": selection_probe_wall,
            "all_subtest_probe_count": len(test_ids) if policy == "probe_anchor_top_k" else top_k,
            "partial_source": partial_source,
        },
        "prediction": predicted,
        "ground_truth": float(actual),
        "comparison": _suite_errors(predicted, float(actual)),
        "xi_only_prediction": xi_only_predicted,
        "xi_only_comparison": _suite_errors(xi_only_predicted, float(actual)),
        "all_top_k_subset_relative_errors": random_subset_errors,
        "timing_estimate_seconds": {
            "xi_collection_estimate": xi_elapsed,
            "selected_probe_wall_sum": probe_wall,
            "selected_partial_execution_wall_s": probe_wall,
            "selection_probe_wall_s": selection_probe_wall,
            "hybrid_wall_estimate": xi_elapsed + selection_probe_wall,
            "full_suite_wall_from_run": full_suite_wall,
            "cost_basis": "measured xi and selected partial wall time where available; xi fallback is configured estimate",
        },
    }


def _parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(POLICY_IDS))
    if not policies or unknown:
        raise ValueError(f"Choose one or more of {', '.join(POLICY_IDS)}; unknown={unknown}")
    if len(policies) != len(set(policies)):
        raise ValueError("Each policy may appear only once")
    return policies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--glob-pattern", default="*/run-*.json", help="Run JSON glob before machine filtering")
    parser.add_argument("--sessions", default="", help="Comma-separated session directory names to include")
    parser.add_argument("--holdout-session", default="", help="Evaluate one whole session while training on all other selected sessions")
    parser.add_argument("--probe-dataset", type=Path, default=None)
    parser.add_argument("--policies", default=",".join(POLICY_IDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-duration-s", type=float, default=4.0)
    parser.add_argument(
        "--partial-source",
        choices=("probe_predictions", "heldout_observed"),
        default="probe_predictions",
        help="Replay predicted micro-probes or exact scores from selected subtests",
    )
    parser.add_argument("--xi-overhead-s", type=float, default=3.0)
    parser.add_argument("--train-aug", type=int, default=20)
    parser.add_argument("--train-k-min", type=int, default=2)
    parser.add_argument("--train-k-max", type=int, default=6)
    parser.add_argument("--probe-model", choices=("lightgbm", "ridge"), default="lightgbm")
    parser.add_argument("--reconstructor", choices=("lightgbm", "lightgbm_small", "xgboost", "ridge"), default="lightgbm")
    parser.add_argument(
        "--suite-output-mode",
        choices=(
            "direct",
            "observed_plus_imputed_geomean",
            "observed_plus_training_geomean",
            "observed_plus_scaled_imputed_geomean",
            "probe_plus_imputed_geomean",
            "reconstructed_geomean",
            "probe_scaled_geomean",
        ),
        default="direct",
    )
    parser.add_argument("--log1p-partial-index", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    policies = _parse_policies(args.policies)

    cv = _load_cv_helpers()
    corrected_oof = _load_corrected_oof_helpers()
    from moebench.probe.model_train import train_probe_bundle
    from moebench.router.feature_vectorizer import XiVectorizer

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    rows, test_ids = cv._load_unixbench_rows(args.dataset_root.resolve(), args.glob_pattern)
    rows = [row for row in rows if args.machine in Path(row["path"]).parent.name]
    if args.sessions.strip():
        wanted_sessions = {s.strip() for s in args.sessions.split(",") if s.strip()}
        rows = [row for row in rows if Path(row["path"]).parent.name in wanted_sessions]
    if len(rows) < 2 or args.folds < 2:
        raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")
    if not 1 <= args.top_k <= len(test_ids):
        raise SystemExit(f"top-k must be in [1, {len(test_ids)}]")

    probe_path = args.probe_dataset or (
        args.dataset_root / "models" / args.machine / "probe_dataset_unixbench.json"
    )
    probe_dataset = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe_dataset.get("benchmark") != "unixbench":
        raise SystemExit(f"Unexpected probe dataset benchmark: {probe_dataset.get('benchmark')}")

    if args.holdout_session:
        folds = [
            [i for i, row in enumerate(rows) if Path(row["path"]).parent.name == args.holdout_session]
        ]
        if not folds[0]:
            raise SystemExit(f"No run rows found for held-out session {args.holdout_session!r}")
        if len(folds[0]) == len(rows):
            raise SystemExit("Held-out session leaves no training runs")
        split_protocol = "whole-session holdout; held-out session is excluded from router/probe/reconstructor fitting"
    else:
        if len(rows) < args.folds:
            raise SystemExit(f"Need >= folds run records; found {len(rows)} for {args.machine}")
        rng = np.random.RandomState(args.seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        folds = [list(map(int, part)) for part in np.array_split(indices, args.folds)]
        split_protocol = "same seeded random run-level folds as run_hybrid_oof_unixbench.py"
    xi_vec = XiVectorizer()
    dynamic_mask = _context_mask(list(xi_vec.feature_names), "dynamic")
    static_mask = _context_mask(list(xi_vec.feature_names), "static")
    policy_reports: dict[str, list[dict[str, Any] | None]] = {
        policy: [None] * len(rows) for policy in policies
    }
    fold_manifest: list[dict[str, Any]] = []
    fold_selections: dict[int, dict[str, dict[str, Any]]] = {}
    probe_index = {
        _run_key(sample["source_run"]) + (str(sample["test_id"]),): sample
        for sample in probe_dataset.get("samples", [])
        if sample.get("source_run") and sample.get("test_id")
    }

    for fold_id, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_indices = [i for i in range(len(rows)) if i not in val_set]
        train_rows = [rows[i] for i in train_indices]
        fold_selections[fold_id] = {}
        if "train_assoc_top_k" in policies:
            fold_selections[fold_id]["train_assoc_top_k"] = _train_assoc_selection(
                train_rows, test_ids, args.top_k
            )
        if "train_assoc_diverse_top_k" in policies:
            fold_selections[fold_id]["train_assoc_diverse_top_k"] = _train_assoc_diverse_selection(
                train_rows, test_ids, args.top_k
            )
        if "learned_anchor_pool" in policies:
            candidate_k = min(len(test_ids), max(args.top_k * 2, args.top_k + 3))
            candidate_pool = _train_assoc_diverse_selection(
                train_rows, test_ids, candidate_k
            )
            candidate_pool["selection_method"] += "; used as candidate pool for Xi-conditioned anchor ranking"
            candidate_pool["pool_size"] = candidate_k
            fold_selections[fold_id]["learned_anchor_pool"] = candidate_pool
        if "train_probe_reliable_top_k" in policies:
            # Filled after inner-OOF probe predictions are available below.
            fold_selections[fold_id]["train_probe_reliable_top_k"] = {
                "selected_test_ids": list(test_ids[:args.top_k]),
                "selection_method": "placeholder_until_inner_oof_probe_reliability",
            }
        train_keys = {_run_key(row["path"]) for row in train_rows}
        fold_probe_dataset = dict(probe_dataset)
        fold_probe_dataset["samples"] = [
            sample
            for sample in probe_dataset.get("samples", [])
            if sample.get("source_run") and _run_key(sample["source_run"]) in train_keys
        ]
        if not fold_probe_dataset["samples"]:
            raise RuntimeError(f"Fold {fold_id}: no training probe samples matched training runs")

        routers = {"learned": cv._train_fold_router(
            train_rows, test_ids, xi_vec, seed=args.seed + fold_id
        )}
        for policy, mask in (("dynamic_context", dynamic_mask), ("static_context", static_mask)):
            if policy in policies:
                router = cv._train_fold_router(
                    train_rows,
                    test_ids,
                    _MaskedXiVectorizer(xi_vec, mask),
                    seed=args.seed + fold_id,
                )
                router["xi_feature_mask"] = list(mask)
                router["context_scope"] = policy
                routers[policy] = router
        probe_bundle = train_probe_bundle(fold_probe_dataset, model_type=args.probe_model)
        train_probe_predictions, inner_probe_manifest = corrected_oof._crossfit_training_probe_predictions(
            train_rows,
            fold_probe_dataset,
            test_ids,
            seed=args.seed + fold_id * 997,
            model_type=args.probe_model,
        )
        if "train_probe_reliable_top_k" in policies:
            reliability, reliability_errors = _probe_reliability_weights(
                train_rows, train_probe_predictions, test_ids
            )
            ordered = sorted(test_ids, key=lambda tid: (-reliability.get(tid, 0.0), tid))
            fold_selections[fold_id]["train_probe_reliable_top_k"] = {
                "selected_test_ids": ordered[: args.top_k],
                "selection_method": "lowest inner-OOF relative probe error",
                "inner_oof_probe_error_median_relative": reliability_errors,
            }
        if "learned_corr_weighted" in policies:
            routers["learned_corr_weighted"] = _train_fold_corr_router(
                train_rows, test_ids, xi_vec, seed=args.seed + fold_id
            )
        if "learned_probe_quality" in policies:
            reliability, probe_error_by_test = _probe_reliability_weights(
                train_rows, train_probe_predictions, test_ids
            )
            routers["learned_probe_quality"] = _train_fold_corr_router(
                train_rows,
                test_ids,
                xi_vec,
                seed=args.seed + fold_id,
                probe_reliability=reliability,
            )
            routers["learned_probe_quality"]["training_fold_probe_error_median_relative"] = (
                probe_error_by_test
            )
        if "learned_anchor" in policies or "learned_anchor_pool" in policies:
            reliability, probe_error_by_test = _probe_reliability_weights(
                train_rows, train_probe_predictions, test_ids
            )
            routers["learned_anchor"] = _train_fold_anchor_router(
                train_rows,
                test_ids,
                xi_vec,
                seed=args.seed + fold_id,
                probe_reliability=reliability,
            )
            routers["learned_anchor"]["training_fold_probe_error_median_relative"] = (
                probe_error_by_test
            )
            if "learned_anchor_pool" in policies:
                routers["learned_anchor_pool"] = routers["learned_anchor"]
        x_train, y_train = corrected_oof._build_crossfitted_reconstructor_training_matrix(
            train_rows,
            train_probe_predictions,
            probe_index,
            xi_vec,
            test_ids,
            seed=args.seed + fold_id * 997,
            train_aug=args.train_aug,
            train_k_min=args.train_k_min,
            train_k_max=args.train_k_max,
            log1p_value=args.log1p_partial_index,
        )
        calibration_factor = _probe_prediction_calibration_factor(
            train_rows,
            train_probe_predictions,
            test_ids,
        )
        calibration_factors = _probe_prediction_calibration_factors(
            train_rows,
            train_probe_predictions,
            test_ids,
        )
        recon_bundle = cv.rte.sklearn_export_bundle(
            args.reconstructor,
            x_train,
            y_train,
            auto_install=False,
            lgbm_estimators=200,
            xgb_estimators=200,
            log1p_partial_index=args.log1p_partial_index,
            test_ids=test_ids,
            with_uncertainty=False,
            benchmark="unixbench",
            suite_output_mode=args.suite_output_mode,
        )
        recon_bundle["probe_prediction_calibration_factor"] = calibration_factor
        recon_bundle["probe_prediction_calibration_factors"] = calibration_factors
        recon_bundle["training_component_log_means"] = {
            tid: float(
                np.mean(
                    np.log(
                        np.maximum(
                            np.asarray([float(row["y"][i]) for row in train_rows], dtype=np.float64),
                            1e-9,
                        )
                    )
                )
            )
            for i, tid in enumerate(test_ids)
        }

        for row_index in val_indices:
            row_meta = rows[row_index]
            for policy in policies:
                random_selection = None
                if policy == "random_top_k":
                    seed_bytes = f"{args.seed}:{fold_id}:{row_meta['path']}".encode("utf-8")
                    seed_int = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
                    random_rng = np.random.RandomState(seed_int % (2**32 - 1))
                    random_selection = list(test_ids)
                    random_rng.shuffle(random_selection)
                policy_reports[policy][row_index] = _evaluate_policy_run(
                    policy=policy,
                    ds=row_meta["ds"],
                    run_path=Path(row_meta["path"]),
                    router=routers.get(policy, routers["learned"]),
                    probe_bundle=probe_bundle,
                    recon_bundle=recon_bundle,
                    probe_index=probe_index,
                    test_ids=test_ids,
                    top_k=args.top_k,
                    probe_duration_s=args.probe_duration_s,
                    xi_overhead_s=args.xi_overhead_s,
                    fold_selected_test_ids=(
                        fold_selections[fold_id][policy]["selected_test_ids"]
                        if policy in fold_selections[fold_id] else None
                    ),
                    random_selection=random_selection,
                    partial_source=args.partial_source,
                )

        fold_manifest.append(
            {
                "fold": fold_id,
                "train_runs": [rows[i]["path"] for i in train_indices],
                "heldout_runs": [rows[i]["path"] for i in val_indices],
                "probe_training_samples": len(fold_probe_dataset["samples"]),
                "router_fit_scope": "training_runs_only",
                "router_fit_runs": len(train_rows),
                "probe_fit_scope": "training_run_samples_only",
                "reconstructor_probe_inputs": "inner-run-held-out probe-model predictions",
                "probe_prediction_calibration_factor": calibration_factor,
                "probe_prediction_calibration_factors": calibration_factors,
                "reconstructor_probe_fit_manifest": inner_probe_manifest,
                "reconstructor_fit_scope": "training_runs_only",
                "training_fold_selections": fold_selections.get(fold_id),
            }
        )
        print(
            f"fold {fold_id + 1}/{len(folds)} complete "
            f"({len(train_rows)} train, {len(val_indices)} held out)",
            file=sys.stderr,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for policy in policies:
        per_run = policy_reports[policy]
        valid = [row for row in per_run if row is not None and not row.get("skipped")]
        errors = [float(row["comparison"]["suite_relative_error"]) for row in valid]
        xi_only_errors = [float(row["xi_only_comparison"]["suite_relative_error"]) for row in valid]
        random_pool = [
            float(value)
            for row in valid
            for value in row.get("all_top_k_subset_relative_errors", [])
        ]
        random_ranks = [
            100.0
            * sum(
                float(candidate) <= float(row["comparison"]["suite_relative_error"])
                for candidate in row.get("all_top_k_subset_relative_errors", [])
            )
            / max(len(row.get("all_top_k_subset_relative_errors", [])), 1)
            for row in valid
        ]
        selected_probe_errors = [
            error
            for row in valid
            for error in row["probe"]["heldout_component_relative_error"].values()
        ]
        probe_walls = [float(row["timing_estimate_seconds"]["selected_probe_wall_sum"]) for row in valid]
        hybrid_walls = [float(row["timing_estimate_seconds"]["hybrid_wall_estimate"]) for row in valid]
        report = {
            "schema": "moebench.hybrid.unixbench_policy_oof.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "machine": args.machine,
            "benchmark": "unixbench",
            "policy": policy,
            "split_protocol": split_protocol,
            "model_protocol": "fold-local router, probe, and reconstructor; all policies share the same models within each fold",
            "selection_uses": (
                "training-fitted router and held-out xi for learned; fixed component IDs for fixed policies; "
                "outer-training-run labels only for association policies; no held-out score labels or ti timings"
            ),
            "folds": args.folds,
            "seed": args.seed,
            "top_k": args.top_k,
            "probe_duration_s": args.probe_duration_s,
            "partial_source": args.partial_source,
            "xi_overhead_s": args.xi_overhead_s,
            "train_aug": args.train_aug,
            "train_k_range": [args.train_k_min, args.train_k_max],
            "reconstructor": args.reconstructor,
            "probe_model": args.probe_model,
            "router_model": (
                "fold-local MLP with suite-representative anchor target and inner-OOF probe reliability"
                if policy == "learned_anchor"
                else "fold-local anchor MLP restricted to a training-fold association-and-diversity candidate pool"
                if policy == "learned_anchor_pool"
                else "fold-local MLP with suite-association and inner-OOF probe-reliability weights"
                if policy == "learned_probe_quality"
                else "fold-local MLP with suite-association-weighted relevance"
                if policy == "learned_corr_weighted"
                else "fold-local MLP"
            ),
            "suite_output_mode": args.suite_output_mode,
            "log1p_partial_index": args.log1p_partial_index,
            "selected_test_ids_by_policy": {
                "fixed_cpu_mix": list(FIXED_CPU_IDS[:args.top_k]),
                "fixed_io_mix": list(FIXED_IO_IDS[:args.top_k]),
                "fixed_first_k": test_ids[:args.top_k],
                "random_top_k": "per-run seeded random subset; no xi, score, or timing inputs",
                "train_assoc_top_k": "per-outer-fold Top-K by absolute Spearman association with suite index",
                "train_assoc_diverse_top_k": (
                    "per-outer-fold exhaustive Top-K maximizing suite association minus "
                    "pairwise redundancy, fit on outer-training run labels"
                ),
                "train_probe_reliable_top_k": (
                    "training-fold Top-K by inner-OOF probe reliability; no held-out labels"
                ),
                "learned": "per-run Top-K from xi",
                "dynamic_context": "per-run Top-K from the dynamic subset of xi; static features zero-masked",
                "static_context": "per-run Top-K from the static subset of xi; dynamic features zero-masked",
                "learned_corr_weighted": (
                    "per-run Top-K from xi; router relevance weights within-run expert rank "
                    "by absolute training-fold expert-to-suite Spearman association"
                ),
                "learned_probe_quality": (
                    "per-run Top-K from full xi; relevance is within-run expert rank times "
                    "training-fold suite association times inner-OOF probe-reliability rank"
                ),
                "learned_anchor": (
                    "per-run Top-K from a fold-local predictor of component-to-suite log closeness, "
                    "weighted by inner-OOF probe reliability"
                ),
                "learned_anchor_pool": (
                    "per-run Top-K from a fold-local suite-representative anchor router, restricted "
                    "to the outer-training association-and-diversity candidate pool"
                ),
                "probe_anchor_top_k": (
                    "per-run Top-K nearest the predicted suite geometric mean; selection uses fresh "
                    "micro-probe predictions for every component"
                ),
            }[policy],
            "input_artifacts": {
                "probe_dataset": str(probe_path.resolve()),
                "probe_dataset_sha256": _sha256(probe_path),
                "runs": [
                    {"path": row["path"], "sha256": _sha256(Path(row["path"]))}
                    for row in rows
                ],
            },
            "fold_manifest": fold_manifest,
            "summary": {
                "n_runs": len(valid),
                "mean_relative_error_percent": float(np.mean(errors) * 100) if errors else None,
                "xi_only_mean_relative_error_percent": (
                    float(np.mean(xi_only_errors) * 100) if xi_only_errors else None
                ),
                "selected_probe_score_mape_percent": (
                    float(np.mean(selected_probe_errors) * 100) if selected_probe_errors else None
                ),
                "mean_error_reduction_vs_xi_only_percentage_points": (
                    float((np.mean(xi_only_errors) - np.mean(errors)) * 100)
                    if errors and xi_only_errors else None
                ),
                "median_relative_error_percent": float(np.median(errors) * 100) if errors else None,
                "p95_relative_error_percent": float(np.percentile(errors, 95) * 100) if errors else None,
                "max_relative_error_percent": float(np.max(errors) * 100) if errors else None,
                "exhaustive_random_subset_mean_error_percent": (
                    float(np.mean(random_pool) * 100) if random_pool else None
                ),
                "exhaustive_random_subset_median_error_percent": (
                    float(np.median(random_pool) * 100) if random_pool else None
                ),
                "mean_selected_error_percentile_within_random_subsets": (
                    float(np.mean(random_ranks)) if random_ranks else None
                ),
                "mean_probe_wall_s": float(np.mean(probe_walls)) if probe_walls else None,
                "mean_hybrid_wall_estimate_s": float(np.mean(hybrid_walls)) if hybrid_walls else None,
            },
            "per_run": per_run,
        }
        report_path = args.out_dir / f"{args.machine}__{policy}.json"
        manifest_path = args.out_dir / f"{args.machine}__{policy}.manifest.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "schema": "moebench.hybrid.unixbench_policy_manifest.v1",
            "report": str(report_path.resolve()),
            "policy": policy,
            "machine": args.machine,
            "folds": args.folds,
            "seed": args.seed,
            "top_k": args.top_k,
            "split_protocol": report["split_protocol"],
            "selection_uses": report["selection_uses"],
            "input_artifacts": report["input_artifacts"],
            "fold_manifest": fold_manifest,
            "summary": report["summary"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results[policy] = report["summary"]

    index_path = args.out_dir / f"{args.machine}__index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": "moebench.hybrid.unixbench_policy_oof_index.v1",
                "machine": args.machine,
                "policies": {
                    policy: {
                        "report": f"{args.machine}__{policy}.json",
                        "manifest": f"{args.machine}__{policy}.manifest.json",
                        "summary": results[policy],
                    }
                    for policy in policies
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
