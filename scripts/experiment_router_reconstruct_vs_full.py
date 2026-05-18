#!/usr/bin/env python3
"""Full experiment: xi → router + partial UnixBench → reconstruct predicted suite → full UnixBench.

Outputs wall times (xi, partial benchmark, full benchmark), predicted vs actual suite
(System Benchmarks Index), and error / time-saved summary as JSON (+ print).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES
from moebench.unixbench.report_parser import parse_report_text, parse_executed_tests_from_report


def _maybe_auto_install(module_name: str, auto_install: bool) -> None:
    if not auto_install:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", module_name])


def load_router_meta(model_fp: Path, auto_install: bool) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        try:
            with open(model_fp, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "unknown"
            if missing == "lightgbm":
                _maybe_auto_install("lightgbm", auto_install)
                with open(model_fp, "rb") as f:
                    return pickle.load(f)
            raise
    import torch

    try:
        return torch.load(model_fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_fp, map_location="cpu")


def _pick_parsed_run(parsed: dict[str, Any]) -> dict[str, Any]:
    from moebench.unixbench.report_parser import pick_preferred_run_block

    return pick_preferred_run_block(parsed)


def parse_executed_from_report(
    report_txt: str, selected_test_ids: list[str]
) -> tuple[list[dict[str, Any]], float | None]:
    executed, suite_f = parse_executed_tests_from_report(report_txt, selected_test_ids)
    return executed, suite_f


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def resolve_existing_file(user_path: str, repo_root: Path, *, kind: str) -> Path:
    """Try absolute path, then cwd-relative, then path under repo root (MoEBench 根目录)."""
    raw = Path(user_path).expanduser()
    tried: list[str] = []
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(repo_root / raw)
    for c in candidates:
        r = c.resolve()
        tried.append(str(r))
        if r.is_file():
            return r
    hint = ""
    if "router" in kind.lower():
        hint = (
            f"\nExpected after training, e.g.:\n"
            f"  {repo_root / 'dataset' / 'unixbench_router' / 'router_model.pkl'}"
        )
    elif "reconstruct" in kind.lower():
        hint = (
            f"\nExport once with:\n"
            f"  python3 scripts/reconstruct_train_eval.py --dataset-root dataset --skip-cv \\\n"
            f"    --export-model {repo_root / 'dataset' / 'models' / 'reconstruct_lgbm.pkl'} --model-type lightgbm"
        )
    raise FileNotFoundError(f"{kind} not found: {user_path!r}\nTried:\n  " + "\n  ".join(tried) + hint)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sudo", action="store_true", help="Re-run with sudo -E (delegates via exec)")
    ap.add_argument(
        "--router-model",
        type=str,
        default="dataset/unixbench_router/router_model.pkl",
        help="Router checkpoint (.pkl / .pt). Default: under repo dataset/unixbench_router/",
    )
    ap.add_argument(
        "--reconstruct-model",
        type=str,
        default="dataset/models/reconstruct_lgbm.pkl",
        help="Reconstruction bundle from reconstruct_train_eval --export-model",
    )
    ap.add_argument("--top-k", type=int, default=None)
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
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("-o", "--output", type=str, default="", help="Write experiment JSON (default under dataset/experiments/...)")
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    if args.sudo and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + forwarded
        raise SystemExit(subprocess.call(cmd))

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

    out_dir = Path(args.dataset_root).resolve() / "experiments" / session_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else out_dir / "experiment_router_reconstruct_vs_full.json"

    stamps = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- 1) xi
    t0 = time.perf_counter()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    t_xi = time.perf_counter() - t0

    # ---- 2) router + partial UnixBench
    router_meta = load_router_meta(router_fp, args.auto_install)
    stored_k = int(router_meta.get("top_k", 3))
    top_k = int(args.top_k) if args.top_k is not None else stored_k

    scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, top_k)

    copies = args.copies if args.copies and args.copies > 0 else UNIXBENCH_PARALLEL_COPIES
    run_script = unixbench_root / "Run"

    ub_partial_base = f"moebench_exp_partial_{session_tag}_{stamps}".replace(":", "-")
    env_partial = os.environ.copy()
    env_partial["UB_OUTPUT_FILE_NAME"] = ub_partial_base
    env_partial["UB_RESULTDIR"] = str(result_dir)
    cmd_partial = ["perl", str(run_script), "-c", str(copies)] + selected_test_ids

    t1 = time.perf_counter()
    rc_p = subprocess.call(cmd_partial, cwd=str(unixbench_root), env=env_partial)
    t_partial = time.perf_counter() - t1
    if rc_p != 0:
        print(f"Partial UnixBench failed with rc={rc_p}", file=sys.stderr)
        return rc_p

    report_path_partial = result_dir / ub_partial_base
    report_txt_partial = report_path_partial.read_text(encoding="utf-8", errors="replace")
    executed_tests, suite_partial_only = parse_executed_from_report(report_txt_partial, selected_test_ids)

    # ---- 3) reconstruct
    t_re0 = time.perf_counter()
    recon_bundle = load_reconstruction_bundle(reconstruct_fp)
    pred = predict_from_partial(recon_bundle, xi, executed_tests)
    t_re = time.perf_counter() - t_re0
    predicted_suite = pred["suite_index"]

    # ---- 4) full UnixBench
    ub_full_base = f"moebench_exp_full_{session_tag}_{stamps}".replace(":", "-")
    env_full = os.environ.copy()
    env_full["UB_OUTPUT_FILE_NAME"] = ub_full_base
    env_full["UB_RESULTDIR"] = str(result_dir)
    cmd_full = ["perl", str(run_script), "-c", str(copies)]

    t2 = time.perf_counter()
    rc_f = subprocess.call(cmd_full, cwd=str(unixbench_root), env=env_full)
    t_full = time.perf_counter() - t2
    if rc_f != 0:
        print(f"Full UnixBench failed with rc={rc_f}", file=sys.stderr)
        return rc_f

    report_path_full = result_dir / ub_full_base
    report_txt_full = report_path_full.read_text(encoding="utf-8", errors="replace")
    parsed_full = parse_report_text(report_txt_full)
    full_run = _pick_parsed_run(parsed_full)
    actual_suite = full_run.get("system_benchmarks_index_score")
    actual_suite_f = _safe_float(actual_suite)

    # ---- report
    suite_err = (
        abs(float(predicted_suite) - float(actual_suite_f))
        if actual_suite_f is not None
        else None
    )
    suite_rel = (
        suite_err / max(abs(float(actual_suite_f)), 1e-9) if actual_suite_f is not None and suite_err is not None else None
    )

    result = {
        "schema": "moebench.experiment.router_reconstruct_vs_full.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_tag": session_tag,
        "router_model": str(router_fp),
        "router_model_type": router_meta.get("model_type"),
        "reconstruct_model": str(reconstruct_fp),
        "unixbench_root": str(unixbench_root),
        "copies_parallel": copies,
        "router": {
            "selected_experts": selected_experts,
            "selected_test_ids": selected_test_ids,
            "top_k": top_k,
            "scores": dict(zip(expert_ids, scores)),
            "probabilities": dict(zip(expert_ids, probs)),
        },
        "timing_seconds": {
            "xi_collection": t_xi,
            "partial_unixbench": t_partial,
            "reconstruct_predict": t_re,
            "full_unixbench": t_full,
            "partial_plus_predict": t_partial + t_re,
            "note": "Compare partial_unixbench vs full_unixbench for benchmark-only savings; "
            "xi_collection is shared once at start.",
        },
        "scores": {
            "predicted_full_suite_benchmarks_index": float(predicted_suite),
            "actual_full_suite_benchmarks_index": actual_suite_f,
            "partial_run_composite_index_selected_tests_only": suite_partial_only,
            "partial_composite_note": "UnixBench composite when running only selected tests; not equal to full-suite index.",
        },
        "reconstruction": {
            "predicted_subtest_index": pred["subtest_index"],
        },
        "comparison": {
            "suite_absolute_error": suite_err,
            "suite_relative_error": suite_rel,
            "benchmark_time_ratio_full_over_partial": (t_full / t_partial) if t_partial > 0 else None,
            "benchmark_time_saved_seconds_vs_full": (t_full - t_partial),
            "moe_benchmark_wall_seconds": t_partial,
            "full_benchmark_wall_seconds": t_full,
        },
        "artifacts": {
            "partial_report": str(report_path_partial),
            "full_report": str(report_path_full),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
