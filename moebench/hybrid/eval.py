"""Hybrid Route A+B evaluation: router Top-K → probe subtest scores → reconstructor suite."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.probe.training_data import label_suite_from_pts_run, label_suite_from_unixbench_run
from moebench.reconstruct.data import full_suite_wall_seconds
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs


def _run_locator(path: str | Path) -> tuple[str, str]:
    """Stable (session_dir, run_filename) key across machines / clone paths."""
    p = Path(path)
    return (p.parent.name, p.name)


def load_router_meta(model_fp: Path) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        with open(model_fp, "rb") as f:
            return pickle.load(f)
    import torch

    try:
        return torch.load(model_fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_fp, map_location="cpu")


def index_probe_dataset(probe_dataset: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Map (session_dir, run_filename, test_id) → probe sample."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sample in probe_dataset.get("samples") or []:
        src = sample.get("source_run")
        tid = sample.get("test_id")
        if not src or not tid:
            continue
        session, run_name = _run_locator(src)
        out[(session, run_name, str(tid))] = sample
    return out


def router_select_test_ids(
    router_meta: dict[str, Any],
    xi: dict[str, Any],
    *,
    top_k: int | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    k = int(top_k if top_k is not None else router_meta.get("top_k", 3))
    selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, k)
    detail = {
        "top_k": k,
        "selected_experts": selected_experts,
        "selected_test_ids": selected_test_ids,
        "scores": dict(zip(expert_ids, scores)),
        "probabilities": dict(zip(expert_ids, probs)),
    }
    return selected_experts, selected_test_ids, detail


def probe_predictions_to_executed_tests(
    selected_test_ids: list[str],
    sub_preds: dict[str, float],
    probe_duration_s: float,
    *,
    benchmark: str,
    probe_wall_by_tid: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    walls = probe_wall_by_tid or {}
    for tid in selected_test_ids:
        if tid not in sub_preds:
            continue
        score = float(sub_preds[tid])
        t = float(walls.get(tid, probe_duration_s))
        if benchmark == "phoronix":
            executed.append({"test_id": tid, "value": score, "time_s": t})
        else:
            executed.append(
                {
                    "test_id": tid,
                    "score": score,
                    "index_detail": {"index": score},
                    "time_s": t,
                }
            )
    return executed


def _ground_truth_suite(
    ds: dict[str, Any],
    *,
    benchmark: str,
    test_ids: list[str],
) -> float | None:
    if benchmark == "phoronix":
        return label_suite_from_pts_run(ds, test_ids)
    return label_suite_from_unixbench_run(ds)


def _suite_errors(predicted: float, actual: float) -> dict[str, float]:
    err = abs(float(predicted) - float(actual))
    rel = err / max(abs(float(actual)), 1e-9)
    return {"suite_absolute_error": err, "suite_relative_error": rel}


def evaluate_single_run_offline(
    *,
    ds: dict[str, Any],
    run_path: Path,
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    probe_index: dict[tuple[str, str, str], dict[str, Any]],
    top_k: int | None,
    probe_duration_s: float,
    xi_overhead_s: float,
    benchmark: str,
    recon_test_ids: list[str],
) -> dict[str, Any] | None:
    xi = ds.get("xi")
    if not isinstance(xi, dict):
        return None

    _, selected_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=top_k)
    if not selected_ids:
        return None

    sub_preds: dict[str, float] = {}
    probe_walls: dict[str, float] = {}
    missing_probe: list[str] = []
    session, run_name = _run_locator(run_path)
    for tid in selected_ids:
        sample = probe_index.get((session, run_name, tid))
        if sample is None:
            missing_probe.append(tid)
            continue
        probe = sample.get("probe") or {}
        sub_preds[tid] = predict_subtest(probe_bundle, probe, tid)
        wall = probe.get("wall_s") or probe.get("duration_s")
        if wall is not None:
            probe_walls[tid] = float(wall)

    if missing_probe:
        return {
            "run_json": str(run_path),
            "skipped": True,
            "reason": "missing_probe_samples",
            "missing_test_ids": missing_probe,
            "router": router_detail,
        }

    executed = probe_predictions_to_executed_tests(
        selected_ids,
        sub_preds,
        probe_duration_s,
        benchmark=benchmark,
        probe_wall_by_tid=probe_walls,
    )
    if not executed:
        return None

    pred = predict_from_partial(recon_bundle, xi, executed)
    gt = _ground_truth_suite(ds, benchmark=benchmark, test_ids=recon_test_ids)
    if gt is None:
        return None

    full_t = full_suite_wall_seconds(ds, test_ids=tuple(recon_test_ids)) if benchmark == "unixbench" else None
    if benchmark == "phoronix":
        from moebench.phoronix.training_data import full_suite_wall_seconds_pts

        full_t = full_suite_wall_seconds_pts(ds, test_ids=tuple(recon_test_ids))
    partial_t = sum(probe_walls.get(tid, probe_duration_s) for tid in selected_ids)
    hybrid_t = float(xi_overhead_s) + partial_t

    comp = _suite_errors(float(pred["suite_index"]), float(gt))
    saved = (float(full_t) - partial_t) if full_t is not None else None
    ratio = (float(full_t) / partial_t) if full_t and partial_t > 0 else None

    return {
        "run_json": str(run_path),
        "session": run_path.parent.name,
        "router": router_detail,
        "probe": {
            "predicted_subtest": sub_preds,
            "probe_duration_s": probe_duration_s,
            "probe_wall_s_selected_sum": partial_t,
        },
        "reconstruction": {
            "predicted_suite": float(pred["suite_index"]),
            "predicted_subtest": pred.get("subtest_index"),
        },
        "ground_truth_suite": float(gt),
        "timing_seconds": {
            "xi_collection_estimate": float(xi_overhead_s),
            "probe_selected_subtests": partial_t,
            "hybrid_wall_estimate": hybrid_t,
            "full_suite_from_dataset": full_t,
        },
        "comparison": {
            **comp,
            "benchmark_time_saved_seconds_vs_full": saved,
            "benchmark_time_ratio_full_over_hybrid_probe": ratio,
            "fraction_saved_vs_full": ((saved / full_t) if saved is not None and full_t else None),
        },
    }


def evaluate_hybrid_offline(
    *,
    run_paths: list[Path],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    probe_dataset: dict[str, Any],
    top_k: int | None = None,
    probe_duration_s: float | None = None,
    xi_overhead_s: float = 3.0,
) -> dict[str, Any]:
    benchmark = str(recon_bundle.get("benchmark") or probe_bundle.get("benchmark") or "unixbench")
    probe_duration_s = float(
        probe_duration_s
        if probe_duration_s is not None
        else probe_bundle.get("probe_duration_s")
        or probe_dataset.get("probe_duration_s")
        or 4.0
    )
    recon_test_ids = list(recon_bundle.get("test_ids") or probe_bundle.get("test_ids") or [])
    probe_index = index_probe_dataset(probe_dataset)

    per_run: list[dict[str, Any]] = []
    for rp in run_paths:
        with open(rp, encoding="utf-8") as f:
            ds = json.load(f)
        row = evaluate_single_run_offline(
            ds=ds,
            run_path=rp,
            router_meta=router_meta,
            recon_bundle=recon_bundle,
            probe_bundle=probe_bundle,
            probe_index=probe_index,
            top_k=top_k,
            probe_duration_s=probe_duration_s,
            xi_overhead_s=xi_overhead_s,
            benchmark=benchmark,
            recon_test_ids=recon_test_ids,
        )
        if row is not None:
            per_run.append(row)

    valid = [r for r in per_run if not r.get("skipped")]
    rel_errs = [float(r["comparison"]["suite_relative_error"]) for r in valid]
    saved_fracs = [
        float(r["comparison"]["fraction_saved_vs_full"])
        for r in valid
        if r.get("comparison", {}).get("fraction_saved_vs_full") is not None
    ]

    def _mean(xs: list[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    aggregate = {
        "num_runs_total": len(per_run),
        "num_runs_evaluated": len(valid),
        "mean_suite_relative_error": _mean(rel_errs),
        "mean_fraction_saved_vs_full": _mean(saved_fracs),
    }

    return {
        "schema": "moebench.experiment.router_probe_reconstruct.v1",
        "mode": "offline",
        "benchmark": benchmark,
        "pts_suite": recon_bundle.get("pts_suite") or probe_bundle.get("pts_suite"),
        "probe_duration_s": probe_duration_s,
        "xi_overhead_s": float(xi_overhead_s),
        "aggregate": aggregate,
        "per_run": per_run,
    }


def evaluate_hybrid_online(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    ground_truth_ds: dict[str, Any],
    ground_truth_run: Path,
    top_k: int | None = None,
    probe_duration_s: float | None = None,
    probe_mode: str | None = None,
    enable_ebpf: bool = True,
    xi_wall_s: float = 0.0,
) -> dict[str, Any]:
    from moebench.probe.collector import collect_subtest_probe

    benchmark = str(recon_bundle.get("benchmark") or probe_bundle.get("benchmark") or "unixbench")
    probe_duration_s = float(
        probe_duration_s
        if probe_duration_s is not None
        else probe_bundle.get("probe_duration_s", 4.0)
    )
    mode = probe_mode or probe_bundle.get("probe_mode", "micro")
    recon_test_ids = list(recon_bundle.get("test_ids") or probe_bundle.get("test_ids") or [])

    _, selected_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=top_k)
    sub_preds: dict[str, float] = {}
    probes: dict[str, Any] = {}
    probe_walls: dict[str, float] = {}
    for tid in selected_ids:
        probe = collect_subtest_probe(
            tid,
            duration_s=probe_duration_s,
            enable_ebpf=enable_ebpf,
            benchmark=benchmark,
            probe_mode=mode,
            pts_title=None,
        )
        probes[tid] = probe
        sub_preds[tid] = predict_subtest(probe_bundle, probe, tid)
        probe_walls[tid] = float(probe.get("wall_s") or probe_duration_s)

    executed = probe_predictions_to_executed_tests(
        selected_ids,
        sub_preds,
        probe_duration_s,
        benchmark=benchmark,
        probe_wall_by_tid=probe_walls,
    )
    pred = predict_from_partial(recon_bundle, xi, executed)
    gt = _ground_truth_suite(ground_truth_ds, benchmark=benchmark, test_ids=recon_test_ids)
    if gt is None:
        raise ValueError(f"Could not read ground-truth suite from {ground_truth_run}")

    if benchmark == "phoronix":
        from moebench.phoronix.training_data import full_suite_wall_seconds_pts

        full_t = full_suite_wall_seconds_pts(ground_truth_ds, test_ids=tuple(recon_test_ids))
    else:
        full_t = full_suite_wall_seconds(ground_truth_ds, test_ids=tuple(recon_test_ids))

    partial_t = sum(probe_walls.values())
    comp = _suite_errors(float(pred["suite_index"]), float(gt))
    saved = (float(full_t) - partial_t) if full_t is not None else None

    return {
        "schema": "moebench.experiment.router_probe_reconstruct.v1",
        "mode": "online",
        "benchmark": benchmark,
        "pts_suite": recon_bundle.get("pts_suite") or probe_bundle.get("pts_suite"),
        "ground_truth_run": str(ground_truth_run),
        "router": router_detail,
        "probe": {
            "predicted_subtest": sub_preds,
            "probe_duration_s": probe_duration_s,
            "probe_mode": mode,
            "probe_wall_s_selected_sum": partial_t,
        },
        "reconstruction": {
            "predicted_suite": float(pred["suite_index"]),
            "predicted_subtest": pred.get("subtest_index"),
        },
        "ground_truth_suite": float(gt),
        "timing_seconds": {
            "xi_collection": float(xi_wall_s),
            "probe_selected_subtests": partial_t,
            "hybrid_wall": float(xi_wall_s) + partial_t,
            "full_suite_from_dataset": full_t,
        },
        "comparison": {
            **comp,
            "benchmark_time_saved_seconds_vs_full": saved,
            "fraction_saved_vs_full": ((saved / full_t) if saved is not None and full_t else None),
        },
        "scores": {
            "predicted_full_suite_benchmarks_index": float(pred["suite_index"]),
            "actual_full_suite_benchmarks_index": float(gt),
        },
    }
