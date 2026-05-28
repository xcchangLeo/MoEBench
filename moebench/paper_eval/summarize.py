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


def _combo_metrics(combo: dict[str, Any]) -> dict[str, float | None]:
    oof = combo.get("oof_metrics") or {}
    ts = combo.get("time_savings") or {}
    return {
        "mae_suite": _f(oof.get("mae_suite_index")),
        "rmse_suite": _f(oof.get("rmse_suite_index")),
        "spearman_suite": _f(oof.get("spearman_suite")),
        "time_saved_frac": _f(ts.get("mean_fraction_wall_time_saved_vs_full_suite")),
    }


def balanced_score(
    *,
    mae_suite: float | None,
    time_saved_frac: float | None,
    mae_ref: float | None = None,
) -> float:
    """Lower is better. Favors low suite MAE and high wall-time savings."""
    mae = mae_suite if mae_suite is not None else 1e18
    ref = mae_ref if mae_ref and mae_ref > 0 else max(mae, 1.0)
    err_term = mae / ref
    save = time_saved_frac if time_saved_frac is not None else 0.0
    save_term = 1.0 - max(0.0, min(1.0, save))
    return err_term + save_term


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
        mae_list = [_f((c.get("oof_metrics") or {}).get("mae_suite_index")) for c in by_k.values()]
        mae_ref = min(x for x in mae_list if x is not None) if any(x is not None for x in mae_list) else None
        for k in sorted(by_k):
            c = by_k[k]
            m = _combo_metrics(c)
            score = balanced_score(mae_suite=m["mae_suite"], time_saved_frac=m["time_saved_frac"], mae_ref=mae_ref)
            rows.append(
                {
                    "k": k,
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
        mae_list = [_f((c.get("oof_metrics") or {}).get("mae_suite_index")) for c in combos]
        mae_ref = min(x for x in mae_list if x is not None) if any(x is not None for x in mae_list) else None
        for c in combos:
            m = _combo_metrics(c)
            score = balanced_score(mae_suite=m["mae_suite"], time_saved_frac=m["time_saved_frac"], mae_ref=mae_ref)
            rows.append(
                {
                    "policy": c.get("policy"),
                    "eval_partial_k": c.get("eval_partial_k"),
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
    """Rank xi ablation modes by suite MAE increase vs ``full`` (router policy, fixed K)."""
    out: dict[str, Any] = {"experiment": "xi_ablation", "suites": []}
    for block in report.get("suite_results") or []:
        combos = [c for c in block.get("combinations") or [] if c.get("policy") == "router"]
        full_mae: float | None = None
        for c in combos:
            if c.get("xi_ablation") == "full":
                full_mae = _f((c.get("oof_metrics") or {}).get("mae_suite_index"))
                break
        rows: list[dict[str, Any]] = []
        for c in combos:
            mode = str(c.get("xi_ablation", ""))
            m = _combo_metrics(c)
            mae = m["mae_suite"]
            delta = (mae - full_mae) if (mae is not None and full_mae is not None) else None
            rel_delta = (delta / full_mae) if (delta is not None and full_mae and full_mae > 0) else None
            rows.append(
                {
                    "xi_ablation": mode,
                    "label_zh": _XI_MODE_LABELS.get(mode, mode),
                    "mae_suite_index": mae,
                    "mae_increase_vs_full": delta,
                    "relative_mae_increase_vs_full": rel_delta,
                    "spearman_suite": m["spearman_suite"],
                    "mean_wall_time_saved_fraction": m["time_saved_frac"],
                }
            )
        # Importance: larger MAE increase when removed => more important block kept in full
        imp = [r for r in rows if r["xi_ablation"] != "full"]
        imp.sort(
            key=lambda r: (
                -(r["mae_increase_vs_full"] if r["mae_increase_vs_full"] is not None else -1e18),
                -(r["relative_mae_increase_vs_full"] or 0),
            ),
        )
        rows.sort(key=lambda r: r["xi_ablation"])
        out["suites"].append(
            {
                "suite_key": block.get("suite_key"),
                "baseline_mae_suite_full_xi": full_mae,
                "eval_partial_k": combos[0].get("eval_partial_k") if combos else None,
                "by_mode": rows,
                "importance_ranking": imp,
                "interpretation": (
                    "importance_ranking 按「去掉该组特征后 MAE 上升幅度」排序；"
                    "上升越大，说明该特征组对重建越重要。"
                ),
            }
        )
    return out
