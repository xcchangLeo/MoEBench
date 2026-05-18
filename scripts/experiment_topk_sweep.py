#!/usr/bin/env python3
"""Top-K sweep for fixed router/reconstruction models.

Runs:
1) one xi collection
2) one full UnixBench run (shared ground truth)
3) partial+reconstruct for each K in --k-values

Designed for: GNN router + XGBoost reconstruction (can override paths).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench import collect_all
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES
from moebench.unixbench.report_parser import parse_executed_tests_from_report, parse_report_text, pick_preferred_run_block


def _load_helpers() -> Any:
    p = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full.py"
    spec = importlib.util.spec_from_file_location("_ervf", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


_ervf = _load_helpers()
load_router_meta = _ervf.load_router_meta
resolve_existing_file = _ervf.resolve_existing_file


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _run_ub(unixbench_root: Path, result_dir: Path, base_name: str, copies: int, test_ids: list[str]) -> Path:
    run_script = unixbench_root / "Run"
    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)
    cmd = ["perl", str(run_script), "-c", str(copies)] + test_ids
    print("+", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(unixbench_root), env=env)
    if rc != 0:
        raise RuntimeError(f"UnixBench Run failed rc={rc}")
    return result_dir / base_name


def _recommend(rows: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    if objective == "error":
        return sorted(
            rows,
            key=lambda r: (
                float(r.get("suite_absolute_error") or 1e99),
                -float(r.get("benchmark_time_saved_seconds_vs_full") or -1e99),
            ),
        )[0]
    if objective == "time":
        return sorted(
            rows,
            key=lambda r: (
                -float(r.get("benchmark_time_saved_seconds_vs_full") or -1e99),
                float(r.get("suite_absolute_error") or 1e99),
            ),
        )[0]
    # balanced
    def score(r: dict[str, Any]) -> float:
        e = float(r.get("suite_relative_error") or 1e9)
        # higher save is better; convert to "cost"
        save = float(r.get("benchmark_time_saved_seconds_vs_full") or 0.0)
        full = float(r.get("full_unixbench") or 1.0)
        save_cost = 1.0 - max(0.0, min(1.0, save / max(full, 1e-9)))
        return e + save_cost

    return sorted(rows, key=score)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sudo", action="store_true", help="Re-run with sudo -E")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--router-model", type=str, default="dataset/router_models/20260402T045114Z/router_gnn.pt")
    ap.add_argument("--reconstruct-model", type=str, default="dataset/models/reconstruct_xgb_v2.pkl")
    ap.add_argument("--k-values", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--objective", type=str, choices=("error", "time", "balanced"), default="balanced")
    ap.add_argument(
        "--copies",
        type=int,
        default=0,
        help=f"UnixBench -c; 0 = {UNIXBENCH_PARALLEL_COPIES} (single-copy only)",
    )
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--unixbench-root", type=str, default=None)
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("-o", "--output", type=str, default="")
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    if args.sudo and os.geteuid() != 0:
        fwd = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + fwd
        raise SystemExit(subprocess.call(cmd))

    try:
        k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    except ValueError:
        print("--k-values must be comma-separated integers", file=sys.stderr)
        return 2
    if not k_values:
        print("No k values provided", file=sys.stderr)
        return 2
    max_k = len(INDEX_SUITE_TEST_IDS)
    for k in k_values:
        if k <= 0 or k > max_k:
            print(f"Invalid k={k}; valid range is 1..{max_k}", file=sys.stderr)
            return 2

    repo_root = REPO_ROOT
    router_fp = resolve_existing_file(args.router_model, repo_root, kind="Router model")
    reconstruct_fp = resolve_existing_file(args.reconstruct_model, repo_root, kind="Reconstruction model")
    unixbench_root = Path(args.unixbench_root).resolve() if args.unixbench_root else repo_root / "byte-unixbench" / "UnixBench"
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    session_tag = args.session
    if not session_tag:
        host = os.uname().nodename.split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_tag = f"{host}_{ts}"
        session_tag = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in session_tag)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = Path(args.dataset_root).resolve() / "experiments" / f"topk_sweep_{session_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else out_dir / "topk_sweep.json"

    copies = args.copies if args.copies and args.copies > 0 else UNIXBENCH_PARALLEL_COPIES

    t0 = time.perf_counter()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    t_xi = time.perf_counter() - t0

    router_meta = load_router_meta(router_fp, args.auto_install)
    recon_bundle = load_reconstruction_bundle(reconstruct_fp)
    scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)

    # one shared full run as ground truth baseline
    full_base = f"moebench_topk_full_{session_tag}_{stamp}".replace(":", "-")
    t_f0 = time.perf_counter()
    full_report = _run_ub(unixbench_root, result_dir, full_base, copies, list(INDEX_SUITE_TEST_IDS))
    t_full = time.perf_counter() - t_f0
    parsed_full = parse_report_text(full_report.read_text(encoding="utf-8", errors="replace"))
    full_run = pick_preferred_run_block(parsed_full)
    actual_suite_f = _safe_float(full_run.get("system_benchmarks_index_score"))
    if actual_suite_f is None:
        print("Failed to parse full suite index from baseline run", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for k in k_values:
        selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, k)
        part_base = f"moebench_topk_partial_k{k}_{session_tag}_{stamp}".replace(":", "-")
        t_p0 = time.perf_counter()
        partial_report = _run_ub(unixbench_root, result_dir, part_base, copies, selected_test_ids)
        t_partial = time.perf_counter() - t_p0

        executed_tests, suite_partial_only = parse_executed_tests_from_report(
            partial_report.read_text(encoding="utf-8", errors="replace"),
            selected_test_ids,
        )
        t_r0 = time.perf_counter()
        pred = predict_from_partial(recon_bundle, xi, executed_tests)
        t_re = time.perf_counter() - t_r0
        pred_suite = float(pred["suite_index"])
        suite_err = abs(pred_suite - actual_suite_f)
        suite_rel = suite_err / max(abs(actual_suite_f), 1e-9)

        rows.append(
            {
                "k": k,
                "selected_experts": selected_experts,
                "selected_test_ids": selected_test_ids,
                "predicted_full_suite_benchmarks_index": pred_suite,
                "actual_full_suite_benchmarks_index": actual_suite_f,
                "suite_absolute_error": suite_err,
                "suite_relative_error": suite_rel,
                "partial_run_composite_index_selected_tests_only": suite_partial_only,
                "partial_unixbench": t_partial,
                "reconstruct_predict": t_re,
                "partial_plus_predict": t_partial + t_re,
                "full_unixbench": t_full,
                "benchmark_time_saved_seconds_vs_full": t_full - t_partial,
                "benchmark_time_ratio_full_over_partial": (t_full / t_partial) if t_partial > 0 else None,
                "artifacts": {
                    "partial_report": str(partial_report),
                },
            }
        )

    best = _recommend(rows, args.objective)

    result = {
        "schema": "moebench.experiment.topk_sweep.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_tag": session_tag,
        "objective": args.objective,
        "k_values": k_values,
        "router_model": str(router_fp),
        "router_model_type": router_meta.get("model_type"),
        "reconstruct_model": str(reconstruct_fp),
        "copies_parallel": copies,
        "timing_seconds": {
            "xi_collection": t_xi,
            "full_unixbench_shared": t_full,
            "note": "Each K has its own partial_unixbench/reconstruct_predict; full_unixbench is shared baseline.",
        },
        "router_scores_shared": {
            "scores": dict(zip(expert_ids, scores)),
            "probabilities": dict(zip(expert_ids, probs)),
        },
        "artifacts": {
            "full_report": str(full_report),
        },
        "per_k": rows,
        "recommended": {
            "k": int(best["k"]),
            "reason_objective": args.objective,
            "suite_absolute_error": best["suite_absolute_error"],
            "suite_relative_error": best["suite_relative_error"],
            "benchmark_time_saved_seconds_vs_full": best["benchmark_time_saved_seconds_vs_full"],
            "selected_test_ids": best["selected_test_ids"],
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
