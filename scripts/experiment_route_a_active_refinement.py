#!/usr/bin/env python3
"""Route A active refinement: router Top-K + real UnixBench subtests + v2 recon (σ).

Unlike Hybrid (probe-predicted partial scores), Route A runs selected subtests to
completion and feeds **measured** indices into the reconstructor, so σ-guided
补跑 can materially change the partial observation vector.

Default checkpoints: H1 ``router_recon_grid_unixbench_aces-System-Product-Name_*``.
Ground truth defaults to ``aces-System-Product-Name_20260524T102558Z/run-05.json``
(use ``--run-full-ground-truth`` for a live full-suite run instead).

Reports initial vs final suite error, extra subtest wall time, and fixed Top-K
Route~A baselines (K=3/5/6) on the same live ``xi``.
"""

from __future__ import annotations

import argparse
import copy
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

from moebench.ml_venv import ensure_ml_interpreter

ensure_ml_interpreter(
    need_modules=["numpy", "sklearn", "lightgbm"],
    auto_install="--auto-install" in sys.argv,
    label="route_a_active_refinement",
)

from moebench import collect_all
from moebench.dataset_machines import ensure_machine_output_dir, machine_experiments_dir, resolve_training_machine
from moebench.probe.training_data import label_suite_from_unixbench_run
from moebench.reconstruct.inference import (
    bundle_has_uncertainty,
    load_reconstruction_bundle,
    predict_from_partial,
)
from moebench.reconstruct.selection import merge_executed_tests, pick_next_subtest_max_uncertainty
from moebench.hybrid.eval import router_select_test_ids
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES
from moebench.unixbench.report_parser import (
    parse_executed_tests_from_report,
    parse_report_text,
    pick_preferred_run_block,
)

SCHEMA = "moebench.experiment.route_a_active_refinement.v1"

H1_GRID_DIR = (
    REPO_ROOT
    / "dataset/experiments/router_recon_grid_unixbench_aces-System-Product-Name_20260524T140119Z/trained_models"
)
DEFAULT_GT_RUN = REPO_ROOT / "dataset/aces-System-Product-Name_20260524T102558Z/run-05.json"


def _load_router_meta(model_fp: Path, auto_install: bool) -> dict[str, Any]:
    from scripts.experiment_router_reconstruct_vs_full import load_router_meta

    return load_router_meta(model_fp, auto_install)


def _suite_errors(predicted: float, actual: float) -> dict[str, float]:
    err = abs(float(predicted) - float(actual))
    rel = err / max(abs(float(actual)), 1e-9)
    return {"suite_absolute_error": err, "suite_relative_error": rel}


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
        raise RuntimeError(f"UnixBench Run failed rc={rc} for {test_ids}")
    return result_dir / base_name


def _train_reconstruct_v2(
    *,
    dataset_root: Path,
    machine: str,
    export_path: Path,
    model_type: str,
    auto_install: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/reconstruct_train_eval.py"),
        "--benchmark",
        "unixbench",
        "--machine",
        machine,
        "--dataset-root",
        str(dataset_root),
        "--model-type",
        model_type,
        "--skip-cv",
        "--export-model",
        str(export_path),
        "--train-aug",
        "10",
        "--train-k-min",
        "2",
        "--train-k-max",
        "6",
        "--log1p-partial-index",
    ]
    if auto_install:
        cmd.append("--auto-install")
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def _snapshot_row(
    *,
    label: str,
    pred: dict[str, Any],
    ground_truth: float,
    executed: list[dict[str, Any]],
    partial_wall_s: float,
    xi_wall_s: float,
    router_detail: dict[str, Any] | None = None,
    top_k: int | None = None,
    extra_test_ids: list[str] | None = None,
) -> dict[str, Any]:
    probed = [str(e["test_id"]) for e in executed if e.get("test_id") and not e.get("missing")]
    comp = _suite_errors(float(pred["suite_index"]), ground_truth)
    row: dict[str, Any] = {
        "label": label,
        "top_k": top_k,
        "n_executed": len(probed),
        "executed_test_ids": probed,
        "extra_test_ids": list(extra_test_ids or []),
        "predicted_suite": float(pred["suite_index"]),
        "predicted_subtest": pred.get("subtest_index"),
        "partial_unixbench_wall_s": float(partial_wall_s),
        "route_a_wall_s": float(xi_wall_s) + float(partial_wall_s),
        "comparison": comp,
    }
    if "uncertainty_suite" in pred:
        row["uncertainty_suite"] = float(pred["uncertainty_suite"])
        row["suite_confidence"] = float(pred.get("suite_confidence", 0.0))
        row["uncertainty_subtest"] = pred.get("uncertainty_subtest")
    if router_detail is not None:
        row["router"] = router_detail
    return row


def evaluate_fixed_k_route_a(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    ground_truth: float,
    top_k: int,
    unixbench_root: Path,
    result_dir: Path,
    copies: int,
    xi_wall_s: float,
    session_tag: str,
    stamps: str,
    label_prefix: str = "fixed",
) -> dict[str, Any]:
    _, selected_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=top_k)

    run_script = unixbench_root / "Run"
    base = f"moebench_ra_{label_prefix}_k{top_k}_{session_tag}_{stamps}".replace(":", "-")
    t0 = time.perf_counter()
    run_subprocess_ub(run_script, result_dir, base, copies, selected_ids)
    partial_wall = time.perf_counter() - t0
    report_txt = (result_dir / base).read_text(encoding="utf-8", errors="replace")
    executed, _ = parse_executed_tests_from_report(report_txt, selected_ids)
    pred = predict_from_partial(recon_bundle, xi, executed, return_uncertainty=False)
    row = _snapshot_row(
        label=f"{label_prefix}_k{top_k}",
        pred=pred,
        ground_truth=ground_truth,
        executed=executed,
        partial_wall_s=partial_wall,
        xi_wall_s=xi_wall_s,
        router_detail=router_detail,
        top_k=top_k,
    )
    row["partial_report"] = str(result_dir / base)
    return row


def evaluate_active_route_a(
    *,
    xi: dict[str, Any],
    router_meta: dict[str, Any],
    recon_bundle: dict[str, Any],
    ground_truth: float,
    initial_top_k: int,
    active_max_extra: int,
    stop_sigma_suite: float | None,
    stop_min_confidence: float | None,
    unixbench_root: Path,
    result_dir: Path,
    copies: int,
    xi_wall_s: float,
    session_tag: str,
    stamps: str,
) -> dict[str, Any]:
    if not bundle_has_uncertainty(recon_bundle):
        raise ValueError("Reconstruction bundle must be v2 (uncertainty export).")

    _, selected_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=initial_top_k)

    run_script = unixbench_root / "Run"
    base0 = f"moebench_ra_active_k{initial_top_k}_{session_tag}_{stamps}".replace(":", "-")
    t0 = time.perf_counter()
    run_subprocess_ub(run_script, result_dir, base0, copies, selected_ids)
    initial_partial_wall = time.perf_counter() - t0
    report0 = (result_dir / base0).read_text(encoding="utf-8", errors="replace")
    executed = parse_executed_tests_from_report(report0, selected_ids)[0]
    executed_ids = {str(e["test_id"]) for e in executed if e.get("test_id") and not e.get("missing")}

    cur = predict_from_partial(recon_bundle, xi, executed, return_uncertainty=True)
    initial_row = _snapshot_row(
        label="initial",
        pred=cur,
        ground_truth=ground_truth,
        executed=executed,
        partial_wall_s=initial_partial_wall,
        xi_wall_s=xi_wall_s,
        router_detail=router_detail,
        top_k=initial_top_k,
    )

    def _should_stop(p: dict[str, Any]) -> bool:
        if stop_sigma_suite is not None and float(p["uncertainty_suite"]) <= stop_sigma_suite:
            return True
        if stop_min_confidence is not None and float(p["suite_confidence"]) >= stop_min_confidence:
            return True
        return False

    extra_wall = 0.0
    extra_ids: list[str] = []
    rounds: list[dict[str, Any]] = []

    for step in range(int(active_max_extra)):
        if _should_stop(cur):
            break
        nxt = pick_next_subtest_max_uncertainty(
            cur["uncertainty_subtest"], executed_ids, INDEX_SUITE_TEST_IDS
        )
        if nxt is None:
            break
        sigma_before = float(cur["uncertainty_suite"])
        sigma_chosen = float(cur["uncertainty_subtest"].get(nxt, 0.0))
        base = f"moebench_ra_active_extra_s{step}_{session_tag}_{stamps}".replace(":", "-")
        t1 = time.perf_counter()
        run_subprocess_ub(run_script, result_dir, base, copies, [nxt])
        step_wall = time.perf_counter() - t1
        extra_wall += step_wall
        report_txt = (result_dir / base).read_text(encoding="utf-8", errors="replace")
        new_ex, _ = parse_executed_tests_from_report(report_txt, [nxt])
        executed = merge_executed_tests(executed, new_ex)
        executed_ids = {str(e["test_id"]) for e in executed if e.get("test_id") and not e.get("missing")}
        cur = predict_from_partial(recon_bundle, xi, executed, return_uncertainty=True)
        extra_ids.append(nxt)
        rounds.append(
            {
                "step": step,
                "added_test_id": nxt,
                "uncertainty_subtest_chosen": sigma_chosen,
                "uncertainty_suite_before": sigma_before,
                "uncertainty_suite_after": float(cur["uncertainty_suite"]),
                "suite_predicted_after": float(cur["suite_index"]),
                "partial_wall_s_added": step_wall,
            }
        )

    total_partial = initial_partial_wall + extra_wall
    final_row = _snapshot_row(
        label="final",
        pred=cur,
        ground_truth=ground_truth,
        executed=executed,
        partial_wall_s=total_partial,
        xi_wall_s=xi_wall_s,
        top_k=initial_top_k,
        extra_test_ids=extra_ids,
    )

    return {
        "router": router_detail,
        "initial_top_k": int(initial_top_k),
        "active_max_extra": int(active_max_extra),
        "initial": initial_row,
        "final": final_row,
        "rounds": rounds,
        "extra_subtests_count": len(extra_ids),
        "extra_partial_wall_s": float(extra_wall),
        "partial_report_initial": str(result_dir / base0),
    }


def _compare_active_vs_fixed(fixed_row: dict[str, Any], active: dict[str, Any]) -> dict[str, Any]:
    err_i = float(active["initial"]["comparison"]["suite_relative_error"])
    err_f = float(active["final"]["comparison"]["suite_relative_error"])
    err_fix = float(fixed_row["comparison"]["suite_relative_error"])
    return {
        "fixed_k_top_k": int(fixed_row.get("top_k", 0)),
        "initial_vs_fixed_k_relative_error_pp": (err_i - err_fix) * 100.0,
        "final_vs_fixed_k_relative_error_pp": (err_f - err_fix) * 100.0,
        "error_reduction_initial_to_final_abs": float(active["initial"]["comparison"]["suite_absolute_error"])
        - float(active["final"]["comparison"]["suite_absolute_error"]),
        "error_reduction_initial_to_final_rel_pp": (err_i - err_f) * 100.0,
        "extra_partial_wall_s_vs_fixed_k": float(active["final"]["partial_unixbench_wall_s"])
        - float(fixed_row["partial_unixbench_wall_s"]),
        "extra_route_a_wall_s_vs_fixed_k": float(active["final"]["route_a_wall_s"])
        - float(fixed_row["route_a_wall_s"]),
        "active_extra_subtests": int(active["extra_subtests_count"]),
    }


def _parse_k_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _ground_truth_from_run(path: Path) -> tuple[float, str]:
    with open(path, encoding="utf-8") as f:
        ds = json.load(f)
    gt = label_suite_from_unixbench_run(ds)
    if gt is None:
        raise ValueError(f"Could not read suite index from {path}")
    return float(gt), str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sudo",
        action="store_true",
        help="Re-exec via sudo -E using this Python (run from conda: python script.py --sudo; not sudo python3)",
    )
    ap.add_argument("--machine", type=str, default="aces-System-Product-Name")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--grid-dir", type=str, default=str(H1_GRID_DIR))
    ap.add_argument("--router-model", type=str, default="")
    ap.add_argument("--reconstruct-model", type=str, default="")
    ap.add_argument("--reconstruct-model-type", type=str, choices=("lightgbm", "xgboost"), default="lightgbm")
    ap.add_argument("--train-reconstruct-v2", action="store_true")
    ap.add_argument("--ground-truth-run", type=str, default=str(DEFAULT_GT_RUN))
    ap.add_argument("--run-full-ground-truth", action="store_true")
    ap.add_argument("--initial-top-k", type=int, default=3)
    ap.add_argument("--active-max-extra-tests", type=int, default=3)
    ap.add_argument("--active-stop-sigma-suite", type=float, default=None)
    ap.add_argument("--active-stop-min-confidence", type=float, default=1.0)
    ap.add_argument("--fixed-k-compare", type=str, default="5,6")
    ap.add_argument("--skip-fixed-k", action="store_true", help="Only run active path (reuse initial as fixed K=3)")
    ap.add_argument("--unixbench-root", type=str, default="")
    ap.add_argument("--copies", type=int, default=0)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    if args.sudo and os.geteuid() != 0:
        fwd = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + fwd
        raise SystemExit(subprocess.call(cmd))

    stop_conf = None if args.active_stop_min_confidence >= 1.0 else float(args.active_stop_min_confidence)
    machine = resolve_training_machine(args.machine)
    ds_root = Path(args.dataset_root).resolve()
    grid_dir = Path(args.grid_dir).resolve()
    ext = ".pkl"
    recon_default = grid_dir / f"recon_{'lgbm' if args.reconstruct_model_type == 'lightgbm' else 'xgb'}_v2{ext}"
    router_fp = Path(args.router_model).resolve() if args.router_model.strip() else grid_dir / "router_mlp.pt"
    recon_fp = Path(args.reconstruct_model).resolve() if args.reconstruct_model.strip() else recon_default

    if args.train_reconstruct_v2 or not recon_fp.is_file():
        recon_fp.parent.mkdir(parents=True, exist_ok=True)
        _train_reconstruct_v2(
            dataset_root=ds_root,
            machine=machine,
            export_path=recon_fp,
            model_type=args.reconstruct_model_type,
            auto_install=args.auto_install,
        )

    if not router_fp.is_file():
        print(f"Router not found: {router_fp}", file=sys.stderr)
        return 2
    if not recon_fp.is_file():
        print(f"Reconstruct model not found: {recon_fp}", file=sys.stderr)
        return 2

    recon_bundle = load_reconstruction_bundle(recon_fp)
    if not bundle_has_uncertainty(recon_bundle):
        print(f"ERROR: {recon_fp} has no uncertainty; use --train-reconstruct-v2", file=sys.stderr)
        return 2

    router_meta = _load_router_meta(router_fp, args.auto_install)
    unixbench_root = (
        Path(args.unixbench_root).resolve()
        if args.unixbench_root.strip()
        else REPO_ROOT / "byte-unixbench" / "UnixBench"
    )
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    copies = args.copies if args.copies > 0 else UNIXBENCH_PARALLEL_COPIES

    host = os.uname().nodename.split(".")[0]
    session_tag = "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in f"{host}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    stamps = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    t_xi0 = time.perf_counter()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    t_xi = time.perf_counter() - t_xi0

    gt_path = Path(args.ground_truth_run).resolve()
    if args.run_full_ground_truth:
        run_script = unixbench_root / "Run"
        ub_full = f"moebench_ra_gt_full_{session_tag}_{stamps}".replace(":", "-")
        t_f0 = time.perf_counter()
        run_subprocess_ub(run_script, result_dir, ub_full, copies, list(INDEX_SUITE_TEST_IDS))
        t_full = time.perf_counter() - t_f0
        report_full_txt = (result_dir / ub_full).read_text(encoding="utf-8", errors="replace")
        full_run = pick_preferred_run_block(parse_report_text(report_full_txt))
        gt = float(full_run.get("system_benchmarks_index_score"))
        gt_path_str = str(result_dir / ub_full)
        full_wall_s = t_full
    else:
        if not gt_path.is_file():
            print(f"Ground-truth run not found: {gt_path}", file=sys.stderr)
            return 2
        gt, gt_path_str = _ground_truth_from_run(gt_path)
        full_wall_s = None

    active = evaluate_active_route_a(
        xi=xi,
        router_meta=router_meta,
        recon_bundle=recon_bundle,
        ground_truth=gt,
        initial_top_k=int(args.initial_top_k),
        active_max_extra=int(args.active_max_extra_tests),
        stop_sigma_suite=args.active_stop_sigma_suite,
        stop_min_confidence=stop_conf,
        unixbench_root=unixbench_root,
        result_dir=result_dir,
        copies=copies,
        xi_wall_s=t_xi,
        session_tag=session_tag,
        stamps=stamps,
    )

    fixed_k3 = copy.deepcopy(active["initial"])
    fixed_k3["label"] = "fixed_k3"

    fixed_baselines: dict[str, Any] = {"fixed_k3": fixed_k3}
    if not args.skip_fixed_k:
        for k in _parse_k_list(args.fixed_k_compare):
            if k <= int(args.initial_top_k):
                continue
            fixed_baselines[f"fixed_k{k}"] = evaluate_fixed_k_route_a(
                xi=xi,
                router_meta=router_meta,
                recon_bundle=recon_bundle,
                ground_truth=gt,
                top_k=k,
                unixbench_root=unixbench_root,
                result_dir=result_dir,
                copies=copies,
                xi_wall_s=t_xi,
                session_tag=session_tag,
                stamps=stamps,
            )

    comparison = _compare_active_vs_fixed(fixed_k3, active)
    final_k = int(active["final"]["n_executed"])
    fk = f"fixed_k{final_k}"
    if fk in fixed_baselines and fk != "fixed_k3":
        fin_err = float(active["final"]["comparison"]["suite_relative_error"])
        fix_err = float(fixed_baselines[fk]["comparison"]["suite_relative_error"])
        comparison[f"final_vs_{fk}_relative_error_pp"] = (fin_err - fix_err) * 100.0
        comparison[f"extra_route_a_wall_s_vs_{fk}"] = (
            float(active["final"]["route_a_wall_s"]) - float(fixed_baselines[fk]["route_a_wall_s"])
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "mode": "route_a_online",
        "benchmark": "unixbench",
        "ground_truth_run": gt_path_str,
        "ground_truth_suite": gt,
        "timing_seconds": {
            "xi_collection": t_xi,
            "full_suite_unixbench": full_wall_s,
        },
        "initial_top_k": int(args.initial_top_k),
        "active_max_extra": int(args.active_max_extra_tests),
        "router_model": str(router_fp),
        "reconstruct_model": str(recon_fp),
        "copies_parallel": copies,
        "fixed_k_hybrid_note": "Route A uses real UnixBench subtest scores (not probe predictions).",
        "fixed_k_route_a": fixed_k3,
        "active_refinement": active,
        "comparison": comparison,
        "fixed_k_baselines": {k: v for k, v in fixed_baselines.items() if k != "fixed_k3"},
    }

    if args.output.strip():
        out = Path(args.output).resolve()
    else:
        out = machine_experiments_dir(ds_root, machine) / "route_a_active_refinement_unixbench.json"

    ensure_machine_output_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = {
        "output": str(out),
        "ground_truth_suite": gt,
        "fixed_k3_rel_err_pct": fixed_k3["comparison"]["suite_relative_error"] * 100.0,
        "active_initial_rel_err_pct": active["initial"]["comparison"]["suite_relative_error"] * 100.0,
        "active_final_rel_err_pct": active["final"]["comparison"]["suite_relative_error"] * 100.0,
        "extra_subtests": active["extra_subtests_count"],
        "extra_partial_wall_s": active["extra_partial_wall_s"],
        "comparison": comparison,
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
