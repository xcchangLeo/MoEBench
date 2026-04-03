#!/usr/bin/env python3
"""Compare LightGBM / XGBoost / MLP reconstruction (v2 + σ) with GNN router + active subtests.

1) Collect xi once; GNN router selects Top-K; one shared partial UnixBench.
2) One shared full UnixBench for ground-truth suite index.
3) For each reconstruction model: copy initial partial results, optionally run extra
   subtests (max σ among unrun tests) until σ_suite is low or budget exhausted; re-predict.
4) Report per-model: initial vs final suite error, extra wall time, active rounds.

Requires v2 reconstruction bundles (default export: ``reconstruct_train_eval.py`` without
``--no-uncertainty``). Train GNN router separately: ``router_train.py --model-type gnn_expert``.
"""

from __future__ import annotations

import argparse
import copy
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
from moebench.reconstruct.inference import (
    bundle_has_uncertainty,
    load_reconstruction_bundle,
    predict_from_partial,
)
from moebench.reconstruct.selection import merge_executed_tests, pick_next_subtest_max_uncertainty
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS
from moebench.unixbench.report_parser import parse_executed_tests_from_report, parse_report_text
from moebench.unixbench.report_parser import pick_preferred_run_block


def _load_experiment_helpers() -> Any:
    p = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full.py"
    spec = importlib.util.spec_from_file_location("_ervf", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


_ervf = _load_experiment_helpers()
resolve_existing_file = _ervf.resolve_existing_file
load_router_meta = _ervf.load_router_meta


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def run_subprocess_ub(
    run_script: Path,
    result_dir: Path,
    base_name: str,
    copies: int,
    test_ids: list[str],
) -> Path:
    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)
    cmd = ["perl", str(run_script), "-c", str(copies)] + test_ids
    print("+", " ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(run_script.parent), env=env)
    if rc != 0:
        raise RuntimeError(f"UnixBench Run failed rc={rc}")
    return result_dir / base_name


def train_default_reconstruct_exports(
    dataset_root: Path,
    *,
    auto_install: bool,
    log1p: bool,
    mlp_epochs: int,
    mlp_hidden: int,
    lgbm_estimators: int,
    xgb_estimators: int,
) -> None:
    models = Path(dataset_root) / "models"
    models.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reconstruct_train_eval.py"),
        "--dataset-root",
        str(dataset_root),
        "--skip-cv",
        "--train-aug",
        "10",
        "--train-k-min",
        "2",
        "--train-k-max",
        "6",
        "--lgbm-estimators",
        str(lgbm_estimators),
        "--xgb-estimators",
        str(xgb_estimators),
        "--mlp-epochs",
        str(mlp_epochs),
        "--mlp-hidden",
        str(mlp_hidden),
    ]
    if log1p:
        common.append("--log1p-partial-index")
    if auto_install:
        common.append("--auto-install")
    jobs = [
        ("lightgbm", models / "reconstruct_lgbm_v2.pkl"),
        ("xgboost", models / "reconstruct_xgb_v2.pkl"),
        ("mlp", models / "reconstruct_mlp_v2.pt"),
    ]
    for mt, outp in jobs:
        cmd = common + ["--model-type", mt, "--export-model", str(outp)]
        print("+", " ".join(cmd), file=sys.stderr)
        subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def active_pipeline_for_model(
    *,
    label: str,
    bundle: dict[str, Any],
    xi: dict[str, Any],
    executed_seed: list[dict[str, Any]],
    unixbench_root: Path,
    result_dir: Path,
    copies: int,
    session_tag: str,
    stamps: str,
    actual_suite: float,
    active_max_extra: int,
    stop_sigma_suite: float | None,
    stop_min_confidence: float | None,
) -> dict[str, Any]:
    if not bundle_has_uncertainty(bundle):
        raise ValueError(f"Reconstruction bundle '{label}' has no uncertainty export (need v2 / --uncertainty).")

    run_script = unixbench_root / "Run"
    executed = copy.deepcopy(executed_seed)
    executed_ids = {str(e["test_id"]) for e in executed if e.get("test_id") and not e.get("missing")}

    pred = predict_from_partial(bundle, xi, executed, return_uncertainty=True)
    suite0 = float(pred["suite_index"])
    err0 = abs(suite0 - actual_suite)

    extra_wall_s = 0.0
    rounds: list[dict[str, Any]] = []

    def should_stop(p: dict[str, Any]) -> bool:
        if stop_sigma_suite is not None and float(p["uncertainty_suite"]) <= stop_sigma_suite:
            return True
        if stop_min_confidence is not None and float(p["suite_confidence"]) >= stop_min_confidence:
            return True
        return False

    cur = pred
    for step in range(active_max_extra):
        if should_stop(cur):
            break
        nxt = pick_next_subtest_max_uncertainty(
            cur["uncertainty_subtest"], executed_ids, INDEX_SUITE_TEST_IDS
        )
        if nxt is None:
            break
        base = f"moebench_act_{label}_{session_tag}_{stamps}_s{step}".replace(":", "-")
        t0 = time.perf_counter()
        run_subprocess_ub(run_script, result_dir, base, copies, [nxt])
        extra_wall_s += time.perf_counter() - t0
        report_txt = (result_dir / base).read_text(encoding="utf-8", errors="replace")
        new_ex, _ = parse_executed_tests_from_report(report_txt, [nxt])
        executed = merge_executed_tests(executed, new_ex)
        executed_ids = {str(e["test_id"]) for e in executed if e.get("test_id") and not e.get("missing")}
        cur = predict_from_partial(bundle, xi, executed, return_uncertainty=True)
        rounds.append(
            {
                "step": step,
                "added_test_id": nxt,
                "suite_predicted": float(cur["suite_index"]),
                "uncertainty_suite": float(cur["uncertainty_suite"]),
                "suite_confidence": float(cur["suite_confidence"]),
            }
        )

    suite_f = float(cur["suite_index"])
    err_f = abs(suite_f - actual_suite)
    return {
        "label": label,
        "suite_predicted_initial": suite0,
        "suite_predicted_final": suite_f,
        "suite_absolute_error_initial": err0,
        "suite_absolute_error_final": err_f,
        "suite_error_reduction": err0 - err_f,
        "extra_subtests_wall_seconds": extra_wall_s,
        "active_rounds": rounds,
        "final_uncertainty_suite": float(cur["uncertainty_suite"]),
        "final_suite_confidence": float(cur["suite_confidence"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sudo", action="store_true", help="Re-exec via sudo -E (xi + perf)")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--router-model", type=str, default="dataset/unixbench_router/router_gnn.pt")
    ap.add_argument("--reconstruct-lightgbm", type=str, default="dataset/models/reconstruct_lgbm_v2.pkl")
    ap.add_argument("--reconstruct-xgboost", type=str, default="dataset/models/reconstruct_xgb_v2.pkl")
    ap.add_argument("--reconstruct-mlp", type=str, default="dataset/models/reconstruct_mlp_v2.pt")
    ap.add_argument(
        "--train-reconstruct-models",
        action="store_true",
        help="Export three v2 models before running (by default runs pip install for missing lightgbm/xgboost/torch)",
    )
    ap.add_argument("--unixbench-root", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--copies", type=int, default=0, help="UnixBench -c; 0 = min(32, cpu_count)")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("-o", "--output", type=str, default="", help="Write comparison JSON path")
    ap.add_argument("--auto-install", action="store_true", help="Also pass to router/model loads where supported")
    ap.add_argument(
        "--no-auto-install",
        action="store_true",
        help="With --train-reconstruct-models, do not pass --auto-install to reconstruct_train_eval (fail if xgboost etc. missing)",
    )
    ap.add_argument("--log1p-partial-index", action="store_true", help="Passed to --train-reconstruct-models only")
    ap.add_argument("--active-max-extra-tests", type=int, default=5)
    ap.add_argument(
        "--active-stop-sigma-suite",
        type=float,
        default=None,
        help="Stop adding subtests when predicted σ_suite <= this (omit to use only max-extra / confidence)",
    )
    ap.add_argument(
        "--active-stop-min-confidence",
        type=float,
        default=1.0,
        help="Stop when suite_confidence >= this (1/(1+σ_suite)); default 1.0 = disabled; try 0.999 to stop once confident",
    )
    ap.add_argument("--mlp-epochs-train", type=int, default=400)
    ap.add_argument("--mlp-hidden-train", type=int, default=128)
    ap.add_argument("--lgbm-estimators-train", type=int, default=200)
    ap.add_argument("--xgb-estimators-train", type=int, default=200)
    args = ap.parse_args()

    if args.sudo and os.geteuid() != 0:
        fwd = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + fwd
        raise SystemExit(subprocess.call(cmd))

    if args.active_stop_min_confidence >= 1.0:
        stop_conf = None
    else:
        stop_conf = float(args.active_stop_min_confidence)

    ds_root = Path(args.dataset_root).resolve()
    if args.train_reconstruct_models:
        train_default_reconstruct_exports(
            ds_root,
            auto_install=not args.no_auto_install,
            log1p=args.log1p_partial_index,
            mlp_epochs=args.mlp_epochs_train,
            mlp_hidden=args.mlp_hidden_train,
            lgbm_estimators=args.lgbm_estimators_train,
            xgb_estimators=args.xgb_estimators_train,
        )

    repo_root = REPO_ROOT
    router_fp = resolve_existing_file(args.router_model, repo_root, kind="Router (GNN)")
    paths = {
        "lightgbm": resolve_existing_file(args.reconstruct_lightgbm, repo_root, kind="Reconstruct LightGBM"),
        "xgboost": resolve_existing_file(args.reconstruct_xgboost, repo_root, kind="Reconstruct XGBoost"),
        "mlp": resolve_existing_file(args.reconstruct_mlp, repo_root, kind="Reconstruct MLP"),
    }
    bundles = {k: load_reconstruction_bundle(v) for k, v in paths.items()}
    for k, b in bundles.items():
        if not bundle_has_uncertainty(b):
            print(
                f"ERROR: {k} bundle at {paths[k]} has no uncertainty. "
                f"Re-export with: python3 scripts/reconstruct_train_eval.py --skip-cv --model-type {k} "
                f"--export-model ... (omit --no-uncertainty)",
                file=sys.stderr,
            )
            return 2

    unixbench_root = Path(args.unixbench_root).resolve() if args.unixbench_root else repo_root / "byte-unixbench" / "UnixBench"
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    session_tag = args.session
    if not session_tag:
        host = os.uname().nodename.split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_tag = f"{host}_{ts}"
        session_tag = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in session_tag)

    stamps = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ds_root / "experiments" / f"reconstruct_active_{session_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else out_dir / "reconstruct_active_three.json"

    cpu_count = os.cpu_count() or 1
    copies = args.copies if args.copies and args.copies > 0 else min(32, int(cpu_count))
    run_script = unixbench_root / "Run"

    t_xi0 = time.perf_counter()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    t_xi = time.perf_counter() - t_xi0

    router_meta = load_router_meta(router_fp, args.auto_install)
    stored_k = int(router_meta.get("top_k", 3))
    top_k = int(args.top_k) if args.top_k is not None else stored_k
    scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, top_k)

    ub_shared = f"moebench_active_shared_partial_{session_tag}_{stamps}".replace(":", "-")
    t_p0 = time.perf_counter()
    run_subprocess_ub(run_script, result_dir, ub_shared, copies, selected_test_ids)
    t_partial = time.perf_counter() - t_p0
    report_partial = (result_dir / ub_shared).read_text(encoding="utf-8", errors="replace")
    executed_seed, suite_partial_comp = parse_executed_tests_from_report(report_partial, selected_test_ids)

    ub_full = f"moebench_active_shared_full_{session_tag}_{stamps}".replace(":", "-")
    t_f0 = time.perf_counter()
    run_subprocess_ub(run_script, result_dir, ub_full, copies, list(INDEX_SUITE_TEST_IDS))
    t_full = time.perf_counter() - t_f0
    report_full_txt = (result_dir / ub_full).read_text(encoding="utf-8", errors="replace")
    parsed_full = parse_report_text(report_full_txt)
    full_run = pick_preferred_run_block(parsed_full)
    actual_suite = _safe_float(full_run.get("system_benchmarks_index_score"))
    if actual_suite is None:
        print("Could not parse actual suite index from full run", file=sys.stderr)
        return 2

    per_model: dict[str, Any] = {}
    for key in ("lightgbm", "xgboost", "mlp"):
        per_model[key] = active_pipeline_for_model(
            label=key,
            bundle=bundles[key],
            xi=xi,
            executed_seed=executed_seed,
            unixbench_root=unixbench_root,
            result_dir=result_dir,
            copies=copies,
            session_tag=session_tag,
            stamps=stamps,
            actual_suite=float(actual_suite),
            active_max_extra=int(args.active_max_extra_tests),
            stop_sigma_suite=float(args.active_stop_sigma_suite) if args.active_stop_sigma_suite is not None else None,
            stop_min_confidence=stop_conf,
        )

    result = {
        "schema": "moebench.experiment.reconstruct_active_three.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_tag": session_tag,
        "router_model": str(router_fp),
        "router_model_type": router_meta.get("model_type"),
        "reconstruct_models": {k: str(paths[k]) for k in paths},
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
            "shared_partial_unixbench": t_partial,
            "shared_full_unixbench": t_full,
            "note": "Per-model extra subtest times are in per_model.*.extra_subtests_wall_seconds",
        },
        "scores": {
            "actual_full_suite_benchmarks_index": float(actual_suite),
            "partial_run_composite_index_selected_tests_only": suite_partial_comp,
        },
        "per_model": per_model,
        "artifacts": {
            "partial_report": str(result_dir / ub_shared),
            "full_report": str(result_dir / ub_full),
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
