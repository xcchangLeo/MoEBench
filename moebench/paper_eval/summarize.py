"""Summarize offline paper supplementary experiment JSON reports."""

from __future__ import annotations

import math
from typing import Any


def _f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


# Secondary weight on unsaved wall-time fraction; accuracy (error) term dominates ranking.
BALANCED_SCORE_TIME_WEIGHT = 0.15


def _combo_metrics(combo: dict[str, Any]) -> dict[str, float | None]:
    oof = combo.get("oof_metrics") or {}
    ts = combo.get("time_savings") or {}
    return {
        "mae_suite": _f(oof.get("mae_suite_index")),
        "rmse_suite": _f(oof.get("rmse_suite_index")),
        "spearman_suite": _f(oof.get("spearman_suite")),
        "rel_err_suite": _f(oof.get("mean_suite_relative_error")),
        "time_saved_frac": _f(ts.get("mean_fraction_wall_time_saved_vs_full_suite")),
    }


def balanced_score(
    *,
    rel_err_suite: float | None = None,
    mae_suite: float | None = None,
    time_saved_frac: float | None = None,
    err_ref: float | None = None,
    mae_ref: float | None = None,
    time_weight: float = BALANCED_SCORE_TIME_WEIGHT,
) -> float:
    """Lower is better. Normalized suite error is primary; unsaved time fraction is secondary."""
    if rel_err_suite is not None:
        err = rel_err_suite
        ref = err_ref if err_ref and err_ref > 0 else max(err, 1e-12)
    else:
        mae = mae_suite if mae_suite is not None else 1e18
        ref = mae_ref if mae_ref and mae_ref > 0 else max(mae, 1.0)
        err = mae
    err_term = err / ref
    save = time_saved_frac if time_saved_frac is not None else 0.0
    save_term = 1.0 - max(0.0, min(1.0, save))
    return err_term + float(time_weight) * save_term


def summarize_topk_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rank K values per suite (policy should be ``router``, xi ``full``)."""
    out: dict[str, Any] = {"experiment": "topk_sweep", "suites": []}
    for block in report.get("suite_results") or []:
        sk = block.get("suite_key")
        combos = [
            c
            for c in block.get("combinations") or []
            if c.get("policy") == "router" and c.get("xi_ablation") == "full"
        ]
        by_k: dict[int, dict[str, Any]] = {}
        for c in combos:
            k = int(c.get("eval_partial_k", 0))
            by_k[k] = c
        if not by_k:
            combos = block.get("combinations") or []
            for c in combos:
                k = int(c.get("eval_partial_k", 0))
                by_k[k] = c
        rows: list[dict[str, Any]] = []
        rel_list = [_f((c.get("oof_metrics") or {}).get("mean_suite_relative_error")) for c in by_k.values()]
        mae_list = [_f((c.get("oof_metrics") or {}).get("mae_suite_index")) for c in by_k.values()]
        rel_ref = min(x for x in rel_list if x is not None) if any(x is not None for x in rel_list) else None
        mae_ref = min(x for x in mae_list if x is not None) if any(x is not None for x in mae_list) else None
        for k in sorted(by_k):
            c = by_k[k]
            m = _combo_metrics(c)
            score = balanced_score(
                rel_err_suite=m["rel_err_suite"],
                mae_suite=m["mae_suite"],
                time_saved_frac=m["time_saved_frac"],
                err_ref=rel_ref,
                mae_ref=mae_ref,
            )
            rows.append(
                {
                    "k": k,
                    "mean_suite_relative_error": m["rel_err_suite"],
                    "mae_suite_index": m["mae_suite"],
                    "rmse_suite_index": m["rmse_suite"],
                    "spearman_suite": m["spearman_suite"],
                    "mean_wall_time_saved_fraction": m["time_saved_frac"],
                    "balanced_score": score,
                }
            )
        rows.sort(key=lambda r: float(r["balanced_score"]))
        recommended = rows[0] if rows else None
        out["suites"].append(
            {
                "suite_key": sk,
                "n_samples": block.get("n_samples"),
                "ranking_by_balanced_score": rows,
                "recommended_k": recommended["k"] if recommended else None,
                "recommended": recommended,
            }
        )
    return out


def summarize_policy_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rank routing / subset policies at fixed K (xi should be ``full``)."""
    out: dict[str, Any] = {"experiment": "routing_policy_compare", "suites": []}
    for block in report.get("suite_results") or []:
        combos = [c for c in block.get("combinations") or [] if c.get("xi_ablation") == "full"]
        rows: list[dict[str, Any]] = []
        rel_list = [_f((c.get("oof_metrics") or {}).get("mean_suite_relative_error")) for c in combos]
        mae_list = [_f((c.get("oof_metrics") or {}).get("mae_suite_index")) for c in combos]
        rel_ref = min(x for x in rel_list if x is not None) if any(x is not None for x in rel_list) else None
        mae_ref = min(x for x in mae_list if x is not None) if any(x is not None for x in mae_list) else None
        for c in combos:
            m = _combo_metrics(c)
            score = balanced_score(
                rel_err_suite=m["rel_err_suite"],
                mae_suite=m["mae_suite"],
                time_saved_frac=m["time_saved_frac"],
                err_ref=rel_ref,
                mae_ref=mae_ref,
            )
            rows.append(
                {
                    "policy": c.get("policy"),
                    "eval_partial_k": c.get("eval_partial_k"),
                    "mean_suite_relative_error": m["rel_err_suite"],
                    "mae_suite_index": m["mae_suite"],
                    "rmse_suite_index": m["rmse_suite"],
                    "spearman_suite": m["spearman_suite"],
                    "mean_wall_time_saved_fraction": m["time_saved_frac"],
                    "balanced_score": score,
                }
            )
        rows.sort(key=lambda r: float(r["balanced_score"]))
        out["suites"].append(
            {
                "suite_key": block.get("suite_key"),
                "eval_partial_k": rows[0].get("eval_partial_k") if rows else None,
                "ranking_by_balanced_score": rows,
                "recommended_policy": rows[0]["policy"] if rows else None,
            }
        )
    return out


_XI_MODE_LABELS = {
    "full": "完整 xi（基线）",
    "static_hw_only": "仅静态硬件相关维度",
    "no_perf_pmu": "去掉 PMU/perf 动态计数",
    "no_dynamic_proc": "去掉 /proc 动态与负载代理",
    "no_gpu": "去掉 GPU/OpenCL 维度",
}


def summarize_xi_ablation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rank xi ablation modes by suite relative-error increase vs ``full`` (router policy, fixed K)."""
    out: dict[str, Any] = {"experiment": "xi_ablation", "suites": []}
    for block in report.get("suite_results") or []:
        combos = [c for c in block.get("combinations") or [] if c.get("policy") == "router"]
        full_rel: float | None = None
        full_mae: float | None = None
        for c in combos:
            if c.get("xi_ablation") == "full":
                oof = c.get("oof_metrics") or {}
                full_rel = _f(oof.get("mean_suite_relative_error"))
                full_mae = _f(oof.get("mae_suite_index"))
                break
        rows: list[dict[str, Any]] = []
        for c in combos:
            mode = str(c.get("xi_ablation", ""))
            m = _combo_metrics(c)
            rel = m["rel_err_suite"]
            delta_rel = (rel - full_rel) if (rel is not None and full_rel is not None) else None
            delta_rel_pp = (delta_rel * 100.0) if delta_rel is not None else None
            mae = m["mae_suite"]
            delta_mae = (mae - full_mae) if (mae is not None and full_mae is not None) else None
            rows.append(
                {
                    "xi_ablation": mode,
                    "label_zh": _XI_MODE_LABELS.get(mode, mode),
                    "mean_suite_relative_error": rel,
                    "relative_error_increase_vs_full": delta_rel,
                    "relative_error_increase_vs_full_pp": delta_rel_pp,
                    "mae_suite_index": mae,
                    "mae_increase_vs_full": delta_mae,
                    "mean_wall_time_saved_fraction": m["time_saved_frac"],
                }
            )
        # Importance: larger relative-error increase when removed => more important block kept in full
        imp = [r for r in rows if r["xi_ablation"] != "full"]
        imp.sort(
            key=lambda r: (
                -(r["relative_error_increase_vs_full"] if r["relative_error_increase_vs_full"] is not None else -1e18),
            ),
        )
        for rank, r in enumerate(imp, start=1):
            r["importance_rank"] = rank
        rows.sort(key=lambda r: r["xi_ablation"])
        out["suites"].append(
            {
                "suite_key": block.get("suite_key"),
                "baseline_mean_suite_relative_error_full_xi": full_rel,
                "baseline_mae_suite_full_xi": full_mae,
                "eval_partial_k": combos[0].get("eval_partial_k") if combos else None,
                "by_mode": rows,
                "importance_ranking": imp,
                "interpretation": (
                    "importance_ranking 按「去掉该组特征后 suite 相对误差上升幅度」排序；"
                    "上升越大（ΔRel. err. 越大），说明该特征组越重要。"
                ),
            }
        )
    return out
