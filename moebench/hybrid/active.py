"""Uncertainty-guided active refinement for Hybrid (router → probe → reconstruct)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from moebench.hybrid.eval import (
    index_probe_dataset,
    index_probe_dataset_by_session,
    probe_predictions_to_executed_tests,
    router_select_test_ids,
    _ground_truth_suite,
    _suite_errors,
)
from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import predict_subtest
from moebench.reconstruct.data import full_suite_wall_seconds
from moebench.reconstruct.inference import bundle_has_uncertainty, predict_from_partial
from moebench.reconstruct.selection import pick_next_subtest_max_uncertainty
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS


def _should_stop_active(
    pred: dict[str, Any],
    *,
    stop_sigma_suite: float | None,
    stop_min_confidence: float | None,
) -> bool:
    if stop_sigma_suite is not None and float(pred["uncertainty_suite"]) <= float(stop_sigma_suite):
        return True
    if stop_min_confidence is not None and float(pred["suite_confidence"]) >= float(stop_min_confidence):
        return True
    return False


def _probe_one_subtest_online(
    tid: str,
    *,
    probe_bundle: dict[str, Any],
    benchmark: str,
    probe_duration_s: float,
    probe_mode: str,
    enable_ebpf: bool,
) -> tuple[float, float, dict[str, Any]]:
    probe = collect_subtest_probe(
        tid,
        duration_s=probe_duration_s,
        enable_ebpf=enable_ebpf,
        benchmark=benchmark,
        probe_mode=probe_mode,
        pts_title=None,
    )
    score = float(predict_subtest(probe_bundle, probe, tid))
    wall = float(probe.get("wall_s") or probe_duration_s)
    return score, wall, probe


def _probe_one_subtest_offline(
    tid: str,
    *,
    probe_bundle: dict[str, Any],
    probe_index: dict[tuple[str, str, str], dict[str, Any]] | None,
    probe_index_session: dict[tuple[str, str], dict[str, Any]] | None,
    session: str,
    run_name: str,
    probe_duration_s: float,
) -> tuple[float, float, dict[str, Any]] | None:
    sample = None
    if probe_index is not None:
        sample = probe_index.get((session, run_name, tid))
    if sample is None and probe_index_session is not None:
        sample = probe_index_session.get((session, tid))
    if sample is None:
        return None
    probe = sample.get("probe") or {}
    score = float(predict_subtest(probe_bundle, probe, tid))
    wall = float(probe.get("wall_s") or probe.get("duration_s") or probe_duration_s)
    return score, wall, probe


def _build_executed_from_state(
    probed_ids: list[str],
    sub_preds: dict[str, float],
    probe_walls: dict[str, float],
    *,
    benchmark: str,
    probe_duration_s: float,
) -> list[dict[str, Any]]:
    return probe_predictions_to_executed_tests(
        probed_ids,
        sub_preds,
        probe_duration_s,
        benchmark=benchmark,
        probe_wall_by_tid=probe_walls,
    )


def _snapshot_hybrid_row(
    *,
    pred: dict[str, Any],
    ground_truth: float,
    probed_ids: list[str],
    probe_wall_s: float,
    xi_wall_s: float,
    label: str,
    extra_test_ids: list[str] | None = None,
) -> dict[str, Any]:
    comp = _suite_errors(float(pred["suite_index"]), float(ground_truth))
    extra = extra_test_ids or []
    return {
        "label": label,
        "n_probed": len(probed_ids),
        "probed_test_ids": list(probed_ids),
        "extra_test_ids": list(extra),
        "n_extra_probed": len(extra),
        "predicted_suite": float(pred["suite_index"]),
        "predicted_subtest": pred.get("subtest_index"),
        "uncertainty_suite": float(pred.get("uncertainty_suite", 0.0)),
        "suite_confidence": float(pred.get("suite_confidence", 0.0)),
        "uncertainty_subtest": pred.get("uncertainty_subtest"),
        "probe_wall_s": float(probe_wall_s),
        "hybrid_wall_s": float(xi_wall_s) + float(probe_wall_s),
        "comparison": comp,
    }


def evaluate_hybrid_fixed_k(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    ground_truth: float,
    top_k: int,
    probe_duration_s: float,
    probe_mode: str,
    enable_ebpf: bool,
    benchmark: str,
    recon_test_ids: list[str],
    xi_wall_s: float,
    probe_index: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    probe_index_session: dict[tuple[str, str], dict[str, Any]] | None = None,
    session: str | None = None,
    run_name: str | None = None,
    online: bool = True,
    return_uncertainty: bool = False,
) -> dict[str, Any]:
    """Fixed Top-K hybrid: router → probe K subtests → reconstruct (optional σ)."""
    _, selected_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=top_k)
    sub_preds: dict[str, float] = {}
    probe_walls: dict[str, float] = {}
    missing: list[str] = []

    for tid in selected_ids:
        if online:
            score, wall, _ = _probe_one_subtest_online(
                tid,
                probe_bundle=probe_bundle,
                benchmark=benchmark,
                probe_duration_s=probe_duration_s,
                probe_mode=probe_mode,
                enable_ebpf=enable_ebpf,
            )
        else:
            assert probe_index is not None and session and run_name
            row = _probe_one_subtest_offline(
                tid,
                probe_bundle=probe_bundle,
                probe_index=probe_index,
                probe_index_session=probe_index_session,
                session=session,
                run_name=run_name,
                probe_duration_s=probe_duration_s,
            )
            if row is None:
                missing.append(tid)
                continue
            score, wall, _ = row
        sub_preds[tid] = score
        probe_walls[tid] = wall

    if missing:
        return {
            "skipped": True,
            "reason": "missing_probe_samples",
            "missing_test_ids": missing,
            "router": router_detail,
            "top_k": top_k,
        }

    executed = _build_executed_from_state(
        selected_ids,
        sub_preds,
        probe_walls,
        benchmark=benchmark,
        probe_duration_s=probe_duration_s,
    )
    pred = predict_from_partial(
        recon_bundle,
        xi,
        executed,
        return_uncertainty=return_uncertainty and bundle_has_uncertainty(recon_bundle),
    )
    probe_wall = sum(probe_walls.values())
    row = _snapshot_hybrid_row(
        pred=pred,
        ground_truth=ground_truth,
        probed_ids=selected_ids,
        probe_wall_s=probe_wall,
        xi_wall_s=xi_wall_s,
        label=f"fixed_k{top_k}",
    )
    row["router"] = router_detail
    row["top_k"] = top_k
    return row


def evaluate_hybrid_active_refinement(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    ground_truth: float,
    initial_top_k: int,
    active_max_extra: int,
    stop_sigma_suite: float | None,
    stop_min_confidence: float | None,
    probe_duration_s: float,
    probe_mode: str,
    enable_ebpf: bool,
    benchmark: str,
    recon_test_ids: list[str],
    xi_wall_s: float,
    probe_index: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    probe_index_session: dict[tuple[str, str], dict[str, Any]] | None = None,
    session: str | None = None,
    run_name: str | None = None,
    online: bool = True,
) -> dict[str, Any]:
    """Hybrid with σ-guided probe refinement after an initial router Top-K batch."""
    if not bundle_has_uncertainty(recon_bundle):
        raise ValueError("Reconstruction bundle must export uncertainty (v2 schema).")

    _, selected_ids, router_detail = router_select_test_ids(
        router_meta, xi, top_k=initial_top_k
    )
    sub_preds: dict[str, float] = {}
    probe_walls: dict[str, float] = {}
    probed_order: list[str] = []
    missing: list[str] = []

    def _probe_tid(tid: str) -> bool:
        if online:
            score, wall, _ = _probe_one_subtest_online(
                tid,
                probe_bundle=probe_bundle,
                benchmark=benchmark,
                probe_duration_s=probe_duration_s,
                probe_mode=probe_mode,
                enable_ebpf=enable_ebpf,
            )
        else:
            assert probe_index is not None and session and run_name
            row = _probe_one_subtest_offline(
                tid,
                probe_bundle=probe_bundle,
                probe_index=probe_index,
                probe_index_session=probe_index_session,
                session=session,
                run_name=run_name,
                probe_duration_s=probe_duration_s,
            )
            if row is None:
                missing.append(tid)
                return False
            score, wall, _ = row
        sub_preds[tid] = score
        probe_walls[tid] = wall
        if tid not in probed_order:
            probed_order.append(tid)
        return True

    for tid in selected_ids:
        _probe_tid(tid)

    initial_probe_wall = sum(probe_walls[tid] for tid in selected_ids if tid in probe_walls)

    if missing:
        return {
            "skipped": True,
            "reason": "missing_probe_samples",
            "missing_test_ids": missing,
            "router": router_detail,
        }

    probed_set = set(probed_order)
    executed = _build_executed_from_state(
        probed_order,
        sub_preds,
        probe_walls,
        benchmark=benchmark,
        probe_duration_s=probe_duration_s,
    )
    cur = predict_from_partial(recon_bundle, xi, executed, return_uncertainty=True)
    initial_row = _snapshot_hybrid_row(
        pred=cur,
        ground_truth=ground_truth,
        probed_ids=list(probed_order),
        probe_wall_s=initial_probe_wall,
        xi_wall_s=xi_wall_s,
        label="initial",
    )

    rounds: list[dict[str, Any]] = []
    extra_wall_s = 0.0
    extra_ids: list[str] = []

    for step in range(int(active_max_extra)):
        if _should_stop_active(
            cur,
            stop_sigma_suite=stop_sigma_suite,
            stop_min_confidence=stop_min_confidence,
        ):
            break
        nxt = pick_next_subtest_max_uncertainty(
            cur["uncertainty_subtest"],
            probed_set,
            tuple(recon_test_ids or INDEX_SUITE_TEST_IDS),
        )
        if nxt is None:
            break
        sigma_before = float(cur["uncertainty_suite"])
        if not _probe_tid(nxt):
            break
        extra_wall_s += float(probe_walls[nxt])
        probed_set.add(nxt)
        extra_ids.append(nxt)
        executed = _build_executed_from_state(
            probed_order,
            sub_preds,
            probe_walls,
            benchmark=benchmark,
            probe_duration_s=probe_duration_s,
        )
        cur = predict_from_partial(recon_bundle, xi, executed, return_uncertainty=True)
        rounds.append(
            {
                "step": step,
                "added_test_id": nxt,
                "uncertainty_subtest_chosen": float(cur["uncertainty_subtest"].get(nxt, 0.0)),
                "uncertainty_suite_before": sigma_before,
                "uncertainty_suite_after": float(cur["uncertainty_suite"]),
                "suite_predicted_after": float(cur["suite_index"]),
                "probe_wall_s_added": float(probe_walls[nxt]),
            }
        )

    total_probe_wall = sum(probe_walls[tid] for tid in probed_order)
    final_row = _snapshot_hybrid_row(
        pred=cur,
        ground_truth=ground_truth,
        probed_ids=list(probed_order),
        probe_wall_s=total_probe_wall,
        xi_wall_s=xi_wall_s,
        label="final",
        extra_test_ids=extra_ids,
    )

    return {
        "router": router_detail,
        "initial_top_k": int(initial_top_k),
        "active_max_extra": int(active_max_extra),
        "stop_sigma_suite": stop_sigma_suite,
        "stop_min_confidence": stop_min_confidence,
        "initial": initial_row,
        "final": final_row,
        "rounds": rounds,
        "extra_subtests_count": len(extra_ids),
        "extra_probe_wall_s": float(extra_wall_s),
        "timing_seconds": {
            "xi_collection": float(xi_wall_s),
            "initial_probe_wall_s": float(initial_probe_wall),
            "extra_probe_wall_s": float(extra_wall_s),
            "total_probe_wall_s": float(total_probe_wall),
            "hybrid_wall_initial": float(xi_wall_s) + float(initial_probe_wall),
            "hybrid_wall_final": float(xi_wall_s) + float(total_probe_wall),
        },
    }


def compare_active_vs_fixed(
    *,
    fixed_k_row: dict[str, Any],
    active: dict[str, Any],
) -> dict[str, Any]:
    """Delta metrics: active final vs fixed-K hybrid baseline."""
    gt = None
    init = active["initial"]["comparison"]
    fin = active["final"]["comparison"]
    fix = fixed_k_row["comparison"]
    err_init = float(init["suite_relative_error"])
    err_fin = float(fin["suite_relative_error"])
    err_fix = float(fix["suite_relative_error"])
    return {
        "fixed_k_top_k": int(fixed_k_row.get("top_k", 0)),
        "initial_vs_fixed_k_relative_error_pp": (err_init - err_fix) * 100.0,
        "final_vs_fixed_k_relative_error_pp": (err_fin - err_fix) * 100.0,
        "error_reduction_initial_to_final_abs": float(init["suite_absolute_error"])
        - float(fin["suite_absolute_error"]),
        "error_reduction_initial_to_final_rel_pp": (err_init - err_fin) * 100.0,
        "extra_probe_wall_s_vs_fixed_k": float(active["timing_seconds"]["total_probe_wall_s"])
        - float(fixed_k_row["probe_wall_s"]),
        "extra_hybrid_wall_s_vs_fixed_k": float(active["timing_seconds"]["hybrid_wall_final"])
        - float(fixed_k_row["hybrid_wall_s"]),
        "active_extra_subtests": int(active["extra_subtests_count"]),
    }


def evaluate_hybrid_active_experiment(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    probe_bundle: dict[str, Any],
    ground_truth_ds: dict[str, Any],
    ground_truth_run: Path | None,
    initial_top_k: int = 3,
    active_max_extra: int = 3,
    stop_sigma_suite: float | None = None,
    stop_min_confidence: float | None = None,
    probe_duration_s: float | None = None,
    probe_mode: str | None = None,
    enable_ebpf: bool = True,
    xi_wall_s: float = 0.0,
    online: bool = True,
    probe_dataset: dict[str, Any] | None = None,
    fixed_k_compare: list[int] | None = None,
) -> dict[str, Any]:
    benchmark = str(recon_bundle.get("benchmark") or probe_bundle.get("benchmark") or "unixbench")
    probe_duration_s = float(
        probe_duration_s
        if probe_duration_s is not None
        else probe_bundle.get("probe_duration_s", 4.0)
    )
    mode = str(probe_mode or probe_bundle.get("probe_mode") or "micro")
    recon_test_ids = list(recon_bundle.get("test_ids") or probe_bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    gt = _ground_truth_suite(ground_truth_ds, benchmark=benchmark, test_ids=recon_test_ids)
    if gt is None:
        raise ValueError("Could not read ground-truth suite score")

    if benchmark == "phoronix":
        from moebench.phoronix.training_data import full_suite_wall_seconds_pts

        full_t = full_suite_wall_seconds_pts(ground_truth_ds, test_ids=tuple(recon_test_ids))
    else:
        full_t = full_suite_wall_seconds(ground_truth_ds, test_ids=tuple(recon_test_ids))

    probe_index = index_probe_dataset(probe_dataset) if probe_dataset else None
    probe_index_session = (
        index_probe_dataset_by_session(probe_dataset) if probe_dataset else None
    )
    session = run_name = None
    if ground_truth_run is not None:
        session = ground_truth_run.parent.name
        run_name = ground_truth_run.name

    fixed_k = int(initial_top_k)
    active = evaluate_hybrid_active_refinement(
        xi=xi,
        router_meta=router_meta,
        recon_bundle=recon_bundle,
        probe_bundle=probe_bundle,
        ground_truth=float(gt),
        initial_top_k=fixed_k,
        active_max_extra=int(active_max_extra),
        stop_sigma_suite=stop_sigma_suite,
        stop_min_confidence=stop_min_confidence,
        probe_duration_s=probe_duration_s,
        probe_mode=mode,
        enable_ebpf=enable_ebpf,
        benchmark=benchmark,
        recon_test_ids=recon_test_ids,
        xi_wall_s=xi_wall_s,
        probe_index=probe_index,
        probe_index_session=probe_index_session,
        session=session,
        run_name=run_name,
        online=online,
    )
    if active.get("skipped"):
        return {"skipped": True, "reason": active.get("reason"), "active_refinement": active}

    fixed_row = copy.deepcopy(active["initial"])
    fixed_row["label"] = f"fixed_k{fixed_k}"
    fixed_row["top_k"] = fixed_k
    fixed_row["router"] = active["router"]

    comparison = compare_active_vs_fixed(fixed_k_row=fixed_row, active=active)

    extra_compare: dict[str, Any] = {}
    final_n = int(active["final"]["n_probed"])
    for k in fixed_k_compare or []:
        if k <= fixed_k or k == fixed_k:
            continue
        row = evaluate_hybrid_fixed_k(
            xi=xi,
            router_meta=router_meta,
            recon_bundle=recon_bundle,
            probe_bundle=probe_bundle,
            ground_truth=float(gt),
            top_k=int(k),
            probe_duration_s=probe_duration_s,
            probe_mode=mode,
            enable_ebpf=enable_ebpf,
            benchmark=benchmark,
            recon_test_ids=recon_test_ids,
            xi_wall_s=xi_wall_s,
            probe_index=probe_index,
            probe_index_session=probe_index_session,
            session=session,
            run_name=run_name,
            online=online,
        )
        if not row.get("skipped"):
            extra_compare[f"fixed_k{k}"] = row

    return {
        "schema": "moebench.experiment.hybrid_active_refinement.v1",
        "benchmark": benchmark,
        "mode": "online" if online else "offline",
        "ground_truth_run": str(ground_truth_run) if ground_truth_run else None,
        "ground_truth_suite": float(gt),
        "timing_seconds": {"full_suite_from_dataset": full_t, "xi_collection": float(xi_wall_s)},
        "probe_duration_s": probe_duration_s,
        "probe_mode": mode,
        "initial_top_k": fixed_k,
        "active_max_extra": int(active_max_extra),
        "fixed_k_hybrid": fixed_row,
        "active_refinement": active,
        "comparison": comparison,
        "fixed_k_baselines": extra_compare,
    }
