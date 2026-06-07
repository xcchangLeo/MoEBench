#!/usr/bin/env python3
"""Hybrid active refinement: σ-guided extra micro-probes after router Top-K.

Compares uncertainty-guided refinement (initial Top-K=3, up to N extra probes via max-σ)
against fixed-K Hybrid on H1 UnixBench (or other machine/suite).

Reports initial vs final suite error, extra subtest count / wall time, and deltas vs
fixed Top-K Hybrid. Requires a v2 reconstruction bundle with uncertainty export.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    ensure_machine_output_dir,
    find_latest_router_recon_models_dir,
    find_probe_checkpoint,
    machine_experiments_dir,
    machine_models_dir,
    resolve_checkpoint_file,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.ml_venv import ensure_ml_interpreter
from moebench.reconstruct.data import collect_unixbench_run_paths
from moebench.reconstruct.inference import bundle_has_uncertainty, load_reconstruction_bundle

SCHEMA = "moebench.experiment.hybrid_active_refinement.v1"


def _early_ml_modules() -> list[str]:
    return ["numpy", "sklearn", "lightgbm"]


ensure_ml_interpreter(
    need_modules=_early_ml_modules(),
    auto_install="--auto-install" in sys.argv,
    label="hybrid_active_refinement",
)

from moebench.hybrid.active import evaluate_hybrid_active_experiment, evaluate_hybrid_fixed_k
from moebench.hybrid.eval import index_probe_dataset, index_probe_dataset_by_session, load_router_meta
from moebench.probe.inference import load_probe_bundle


def _latest_run(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def _train_reconstruct_v2(
    *,
    dataset_root: Path,
    machine: str,
    model_type: str,
    export_path: Path,
    auto_install: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reconstruct_train_eval.py"),
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


def _default_paths(
    dataset_root: Path,
    machine: str,
    reconstruct_model_type: str,
) -> dict[str, Path]:
    models_dir = machine_models_dir(dataset_root, machine)
    grid_dir = find_latest_router_recon_models_dir(
        dataset_root, machine=machine, benchmark="unixbench"
    )
    if grid_dir is None:
        raise FileNotFoundError(
            f"No router_recon_grid trained_models for {machine!r}. "
            f"Run scripts/run_router_reconstruct_model_grid.py first."
        )
    ext = ".pt" if reconstruct_model_type == "mlp" else ".pkl"
    recon_v2 = models_dir / f"reconstruct_{reconstruct_model_type}_v2{ext}"
    probe = find_probe_checkpoint(dataset_root, machine=machine, filename="probe_unixbench_lgbm.pkl")
    if probe is None:
        probe = models_dir / "probe_unixbench_lgbm.pkl"
    return {
        "router": grid_dir / "router_lgbm.pkl",
        "reconstruct": recon_v2,
        "probe": probe,
        "probe_dataset": models_dir / "probe_dataset_unixbench.json",
        "grid_dir": grid_dir,
    }


def _parse_fixed_k_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", type=str, default="", help="Host slug (default: local hostname)")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--router-model", type=str, default="", help="Default: latest grid router_lgbm.pkl")
    ap.add_argument("--reconstruct-model", type=str, default="", help="Default: models/<machine>/reconstruct_lgbm_v2.pkl")
    ap.add_argument("--reconstruct-model-type", type=str, choices=("lightgbm", "xgboost", "mlp"), default="lightgbm")
    ap.add_argument("--probe-model", type=str, default="")
    ap.add_argument("--probe-dataset", type=str, default="", help="Required for --offline replay")
    ap.add_argument("--initial-top-k", type=int, default=3)
    ap.add_argument("--active-max-extra-tests", type=int, default=3, help="Max extra probes after Top-K (2–3 typical)")
    ap.add_argument("--active-stop-sigma-suite", type=float, default=None, help="Stop when σ_suite <= threshold")
    ap.add_argument(
        "--active-stop-min-confidence",
        type=float,
        default=1.0,
        help="Stop when suite_confidence >= value; default 1.0 disables",
    )
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument("--probe-mode", type=str, choices=("micro", "real"), default="micro")
    ap.add_argument("--top-k", type=int, default=None, dest="initial_top_k_legacy", help=argparse.SUPPRESS)
    ap.add_argument(
        "--fixed-k-compare",
        type=str,
        default="",
        help="Comma-separated extra fixed-K baselines, e.g. 5,6 (same probe budget, router-selected)",
    )
    ap.add_argument("--auto-fixed-k-match", action="store_true", help="Also compare fixed K = initial + extras used")
    ap.add_argument("--train-reconstruct-v2", action="store_true", help="Export v2 recon with uncertainty before eval")
    ap.add_argument("--offline", action="store_true", help="Replay probes from probe_dataset on latest GT run")
    ap.add_argument("--online", action="store_true", help="Live xi + probes (default when --offline omitted)")
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    if args.initial_top_k_legacy is not None:
        args.initial_top_k = args.initial_top_k_legacy

    stop_conf = None if args.active_stop_min_confidence >= 1.0 else float(args.active_stop_min_confidence)
    offline = bool(args.offline and not args.online)
    online = not offline

    machine = resolve_training_machine(args.machine or None)
    ds_root = Path(args.dataset_root).resolve()
    defaults = _default_paths(ds_root, machine, args.reconstruct_model_type)

    router_fp = Path(args.router_model).resolve() if args.router_model.strip() else defaults["router"]
    recon_fp = Path(args.reconstruct_model).resolve() if args.reconstruct_model.strip() else defaults["reconstruct"]
    probe_fp = Path(args.probe_model).resolve() if args.probe_model.strip() else defaults["probe"]
    probe_ds_fp = Path(args.probe_dataset).resolve() if args.probe_dataset.strip() else defaults["probe_dataset"]

    if args.train_reconstruct_v2 or not recon_fp.is_file():
        recon_fp.parent.mkdir(parents=True, exist_ok=True)
        _train_reconstruct_v2(
            dataset_root=ds_root,
            machine=machine,
            model_type=args.reconstruct_model_type,
            export_path=recon_fp,
            auto_install=args.auto_install,
        )

    router_fp = resolve_checkpoint_file(str(router_fp), REPO_ROOT, kind="Router", fallback=defaults["router"])
    recon_fp = resolve_checkpoint_file(str(recon_fp), REPO_ROOT, kind="Reconstruct", fallback=recon_fp)
    probe_fp = resolve_checkpoint_file(str(probe_fp), REPO_ROOT, kind="Probe", fallback=defaults["probe"])

    recon_bundle = load_reconstruction_bundle(recon_fp)
    if not bundle_has_uncertainty(recon_bundle):
        print(
            f"ERROR: {recon_fp} has no uncertainty (need v2). "
            f"Re-run with --train-reconstruct-v2 or export without --no-uncertainty.",
            file=sys.stderr,
        )
        return 2

    router_meta = load_router_meta(router_fp)
    probe_bundle = load_probe_bundle(probe_fp)

    glo = resolve_glob_for_machine(benchmark="unixbench", machine=machine, glob_pattern=None, pts_suite=None)
    run_paths = collect_unixbench_run_paths(ds_root, glob_pattern=glo)
    if not run_paths:
        print(f"No UnixBench runs for machine {machine!r}", file=sys.stderr)
        return 2
    gt_path = _latest_run(run_paths)
    with open(gt_path, encoding="utf-8") as f:
        gt_ds = json.load(f)

    probe_dataset: dict[str, Any] | None = None
    if offline:
        if not probe_ds_fp.is_file():
            print(f"Probe dataset required for --offline: {probe_ds_fp}", file=sys.stderr)
            return 2
        with open(probe_ds_fp, encoding="utf-8") as f:
            probe_dataset = json.load(f)
        xi = gt_ds.get("xi")
        if not isinstance(xi, dict):
            print(f"No xi in ground-truth run {gt_path}", file=sys.stderr)
            return 2
        xi_wall_s = 0.0
    else:
        from moebench import collect_all

        t0 = time.perf_counter()
        xi = collect_all(enable_ebpf=not args.no_ebpf)
        xi_wall_s = time.perf_counter() - t0

    fixed_k_compare = _parse_fixed_k_list(args.fixed_k_compare)

    report = evaluate_hybrid_active_experiment(
        xi=xi,
        router_meta=router_meta,
        recon_bundle=recon_bundle,
        probe_bundle=probe_bundle,
        ground_truth_ds=gt_ds,
        ground_truth_run=gt_path,
        initial_top_k=int(args.initial_top_k),
        active_max_extra=int(args.active_max_extra_tests),
        stop_sigma_suite=args.active_stop_sigma_suite,
        stop_min_confidence=stop_conf,
        probe_duration_s=args.probe_duration_s,
        probe_mode=args.probe_mode,
        enable_ebpf=not args.no_ebpf,
        xi_wall_s=xi_wall_s,
        online=online,
        probe_dataset=probe_dataset,
        fixed_k_compare=fixed_k_compare or None,
    )

    if report.get("skipped"):
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2

    if args.auto_fixed_k_match:
        final_k = int(report["active_refinement"]["final"]["n_probed"])
        if final_k > int(args.initial_top_k) and final_k not in fixed_k_compare:
            row = evaluate_hybrid_fixed_k(
                xi=xi,
                router_meta=router_meta,
                recon_bundle=recon_bundle,
                probe_bundle=probe_bundle,
                ground_truth=float(report["ground_truth_suite"]),
                top_k=final_k,
                probe_duration_s=float(
                    args.probe_duration_s or probe_bundle.get("probe_duration_s", 4.0)
                ),
                probe_mode=args.probe_mode,
                enable_ebpf=not args.no_ebpf,
                benchmark="unixbench",
                recon_test_ids=list(recon_bundle.get("test_ids") or []),
                xi_wall_s=xi_wall_s,
                probe_index=index_probe_dataset(probe_dataset) if offline and probe_dataset else None,
                probe_index_session=index_probe_dataset_by_session(probe_dataset)
                if offline and probe_dataset
                else None,
                session=gt_path.parent.name if offline else None,
                run_name=gt_path.name if offline else None,
                online=online,
            )
            if not row.get("skipped"):
                report.setdefault("fixed_k_baselines", {})[f"fixed_k{final_k}"] = row
                fin_err = float(report["active_refinement"]["final"]["comparison"]["suite_relative_error"])
                fix_err = float(row["comparison"]["suite_relative_error"])
                report.setdefault("comparison", {})[
                    f"final_vs_fixed_k{final_k}_relative_error_pp"
                ] = (fin_err - fix_err) * 100.0
                report["comparison"][f"extra_hybrid_wall_s_vs_fixed_k{final_k}"] = (
                    float(report["active_refinement"]["timing_seconds"]["hybrid_wall_final"])
                    - float(row["hybrid_wall_s"])
                )

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["machine"] = machine
    report["router_model"] = str(router_fp)
    report["reconstruct_model"] = str(recon_fp)
    report["probe_model"] = str(probe_fp)
    if offline:
        report["probe_dataset"] = str(probe_ds_fp)

    if args.output.strip():
        out = Path(args.output).resolve()
    else:
        out = (
            machine_experiments_dir(ds_root, machine)
            / "hybrid_active_refinement_unixbench.json"
        )

    ensure_machine_output_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = {
        "output": str(out),
        "ground_truth_suite": report["ground_truth_suite"],
        "fixed_k3_rel_err_pct": report["fixed_k_hybrid"]["comparison"]["suite_relative_error"] * 100.0,
        "active_initial_rel_err_pct": report["active_refinement"]["initial"]["comparison"]["suite_relative_error"] * 100.0,
        "active_final_rel_err_pct": report["active_refinement"]["final"]["comparison"]["suite_relative_error"] * 100.0,
        "extra_subtests": report["active_refinement"]["extra_subtests_count"],
        "extra_probe_wall_s": report["active_refinement"]["extra_probe_wall_s"],
        "comparison": report["comparison"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
