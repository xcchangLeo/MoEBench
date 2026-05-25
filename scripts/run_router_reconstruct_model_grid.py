#!/usr/bin/env python3
"""Per-suite pipeline: train 3 routers + 3 reconstructors, then 3×3 end-to-end experiments.

Supports **UnixBench** (``experiment_router_reconstruct_vs_full.py``) and **PTS**
(``experiment_router_reconstruct_vs_full_pts.py``). Use one invocation per benchmark;
run UnixBench / PTS CPU / PTS GPU **separately** as needed.

Glob defaults match data collection (``moebench.dataset_globs``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    list_machines_in_dataset,
    local_host_slug,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.phoronix.pipeline import safe_session_tag

ROUTER_TRAIN = REPO_ROOT / "scripts" / "router_train.py"
RECON_TRAIN = REPO_ROOT / "scripts" / "reconstruct_train_eval.py"
EXPERIMENT_UB = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full.py"
EXPERIMENT_PTS = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full_pts.py"
ML_VENV_PY = REPO_ROOT / ".venv-moebench-router" / "bin" / "python3"
INSTALL_ML_DEPS = REPO_ROOT / "scripts" / "install_ml_python_deps.sh"

ROUTER_SPECS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "router_lgbm.pkl"),
    ("mlp", "router_mlp.pt"),
    ("gnn_expert", "router_gnn.pt"),
)

RECON_SPECS: tuple[tuple[str, str], ...] = (
    ("xgboost", "recon_xgb.pkl"),
    ("lightgbm", "recon_lgbm.pkl"),
    ("mlp", "recon_mlp.pt"),
)


def _preferred_python() -> str:
    venv_py = REPO_ROOT / ".venv-moebench-router" / "bin" / "python3"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


ROUTER_VENV = REPO_ROOT / ".venv-moebench-router"


def _python_import_ok(py: str, module: str) -> bool:
    return (
        subprocess.run([py, "-c", f"import {module}"], capture_output=True).returncode == 0
    )


def _resolve_python(*, require_ml: bool = False) -> str:
    if ML_VENV_PY.is_file() and (not require_ml or _python_import_ok(str(ML_VENV_PY), "numpy")):
        return str(ML_VENV_PY)
    for name in ("python", "python3"):
        cand = ROUTER_VENV / "bin" / name
        if cand.is_file() and (not require_ml or _python_import_ok(str(cand), "numpy")):
            return str(cand)
    if not require_ml or _python_import_ok(sys.executable, "numpy"):
        return sys.executable
    return sys.executable


def _ensure_ml_python(auto_install: bool) -> str:
    py = _resolve_python(require_ml=True)
    if _python_import_ok(py, "numpy"):
        if py != sys.executable:
            print(f"[grid] using project venv python: {py}", file=sys.stderr)
        return py

    if not auto_install:
        raise SystemExit(
            "Missing Python ML deps (numpy). Run:\n"
            "  bash scripts/install_ml_python_deps.sh --use-venv\n"
            f"Then re-run with: {ML_VENV_PY} scripts/run_router_reconstruct_model_grid.py ..."
        )

    if INSTALL_ML_DEPS.is_file():
        install_args = [str(INSTALL_ML_DEPS)]
        if not os.environ.get("CONDA_PREFIX"):
            install_args.append("--use-venv")
        print(f"[grid] bootstrapping ML deps via {' '.join(install_args)}", file=sys.stderr)
        subprocess.check_call(["bash", *install_args])
    else:
        raise SystemExit(f"Missing install script: {INSTALL_ML_DEPS}")

    py = _resolve_python(require_ml=True)
    if not _python_import_ok(py, "numpy"):
        raise SystemExit("ML dependency install finished but numpy is still unavailable.")
    print(f"[grid] using python: {py}", file=sys.stderr)
    return py

def _summarize_unixbench(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    comp = d.get("comparison") or {}
    scores = d.get("scores") or {}
    timing = d.get("timing_seconds") or {}
    return {
        "experiment_json": str(path),
        "router_model_type": d.get("router_model_type"),
        "reconstruct_model": d.get("reconstruct_model"),
        "predicted_suite": scores.get("predicted_full_suite_benchmarks_index"),
        "actual_suite": scores.get("actual_full_suite_benchmarks_index"),
        "suite_abs_err": comp.get("suite_absolute_error"),
        "suite_rel_err": comp.get("suite_relative_error"),
        "partial_benchmark_s": timing.get("partial_unixbench"),
        "full_benchmark_s": timing.get("full_unixbench"),
        "xi_s": timing.get("xi_collection"),
        "time_saved_benchmark_s": comp.get("benchmark_time_saved_seconds_vs_full"),
    }


def _summarize_pts(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sm = d.get("suite_mean") or {}
    ts = d.get("times_s") or {}
    return {
        "experiment_json": str(path),
        "router_model_path": d.get("router_model"),
        "reconstruct_model_path": d.get("reconstruct_model"),
        "predicted_suite": sm.get("predicted_from_partial"),
        "actual_suite": sm.get("ground_truth_full"),
        "suite_abs_err": sm.get("abs_error"),
        "suite_rel_err": sm.get("relative_error"),
        "partial_benchmark_s": ts.get("pts_partial"),
        "full_benchmark_s": ts.get("pts_full"),
        "xi_s": ts.get("xi"),
        "time_saved_benchmark_s": (
            float(ts.get("pts_full", 0)) - float(ts.get("pts_partial", 0))
            if ts.get("pts_full") is not None and ts.get("pts_partial") is not None
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument(
        "--benchmark",
        type=str,
        choices=("unixbench", "phoronix"),
        default="unixbench",
        help="Which end-to-end experiment driver to use after training",
    )
    ap.add_argument(
        "--pts-suite",
        type=str,
        default="",
        help="Required for --benchmark phoronix: training filter / yi.suite (e.g. cpu, pts/nvidia-gpu-compute)",
    )
    ap.add_argument(
        "--suite-full",
        type=str,
        default="",
        help="PTS full baseline passed to phoronix-test-suite (default: same as --pts-suite)",
    )
    ap.add_argument("--pts-mode", type=str, default="batch-run", choices=("run", "batch-run"))
    ap.add_argument("--sudo-for-xi", action="store_true", help="Forward to PTS experiment (xi only)")
    ap.add_argument(
        "--out-parent",
        type=str,
        default="",
        help="Output directory root (default under <dataset-root>/experiments/...)",
    )
    ap.add_argument(
        "--glob-pattern",
        type=str,
        default="",
        help="Training glob (default: auto from benchmark + --pts-suite)",
    )
    ap.add_argument(
        "--machine",
        type=str,
        default="",
        help="Train and evaluate using only this host's sessions (default: current hostname)",
    )
    ap.add_argument(
        "--all-machines",
        action="store_true",
        help="Run the full pipeline once per machine found under dataset-root (implies --stage all)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stage",
        type=str,
        choices=("all", "routers", "reconstructors", "grid"),
        default="all",
        help="all: train routers + reconstructors + 9 experiments; routers/reconstructors/grid: run one phase only "
        "(use same --out-parent between phases when not using default timestamp dir).",
    )
    ap.add_argument("--sudo", action="store_true", help="UnixBench: forward --sudo to experiment script")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--mlp-epochs-router", type=int, default=200)
    ap.add_argument("--mlp-epochs-recon", type=int, default=400)
    ap.add_argument("--mlp-hidden-router", type=int, default=64)
    ap.add_argument("--mlp-hidden-recon", type=int, default=128)
    ap.add_argument("--gnn-emb-dim", type=int, default=12)
    ap.add_argument("--train-aug", type=int, default=20)
    ap.add_argument("--train-k-min", type=int, default=2)
    ap.add_argument("--train-k-max", type=int, default=6)
    ap.add_argument("--eval-partial-k", type=int, default=3)
    ap.add_argument(
        "--pts-suite-target",
        type=str,
        choices=("arithmetic_mean", "logmean"),
        default="logmean",
        help="PTS reconstruction export only",
    )
    ap.add_argument(
        "--experiment-extra",
        type=str,
        default="",
        help="Extra argv token(s) for the experiment script (quoted; rarely needed)",
    )
    args = ap.parse_args()
    python_exe = _ensure_ml_python(args.auto_install)

    if args.all_machines and args.stage != "all":
        print("--all-machines requires --stage all (one output tree per machine).", file=sys.stderr)
        return 2

    if args.benchmark == "phoronix" and not args.pts_suite.strip():
        print("--benchmark phoronix requires --pts-suite", file=sys.stderr)
        return 2

    if args.stage in ("reconstructors", "grid") and not args.out_parent.strip() and not args.all_machines:
        print(
            "--stage reconstructors|grid requires --out-parent=<same directory> from the first stage "
            "(or run --stage all / --stage routers with optional --out-parent).",
            file=sys.stderr,
        )
        return 2

    ds_root = Path(args.dataset_root).resolve()
    pts_suite = args.pts_suite.strip() if args.benchmark == "phoronix" else None

    if args.all_machines:
        machines = list_machines_in_dataset(ds_root, benchmark=args.benchmark, pts_suite=pts_suite)
        if not machines:
            print(
                f"No sessions found for benchmark={args.benchmark!r} under {ds_root}",
                file=sys.stderr,
            )
            return 2
    else:
        machines = [resolve_training_machine(args.machine or None)]

    batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    last_code = 0
    for machine in machines:
        code = _run_for_machine(
            args,
            machine=machine,
            ds_root=ds_root,
            batch_stamp=batch_stamp,
            python_exe=python_exe,
        )
        if code != 0:
            last_code = code
    return last_code


def _run_for_machine(
    args: argparse.Namespace,
    *,
    machine: str,
    ds_root: Path,
    batch_stamp: str,
    python_exe: str,
) -> int:
    if args.benchmark == "phoronix" and not args.pts_suite.strip():
        return 2

    if args.stage in ("reconstructors", "grid") and not args.out_parent.strip():
        print(
            "--stage reconstructors|grid requires --out-parent=<same directory> from the first stage "
            "(or run --stage all / --stage routers with optional --out-parent).",
            file=sys.stderr,
        )
        return 2

    suite_full = (args.suite_full or args.pts_suite).strip() if args.benchmark == "phoronix" else ""
    pts_suite = args.pts_suite.strip() if args.benchmark == "phoronix" else None

    glob_eff = resolve_glob_for_machine(
        benchmark=args.benchmark,
        machine=machine,
        glob_pattern=args.glob_pattern or None,
        pts_suite=pts_suite,
    )

    if args.out_parent.strip():
        base_out = Path(args.out_parent).resolve()
        out_dir = base_out / machine if args.all_machines else base_out
    else:
        if args.benchmark == "unixbench":
            sub = f"router_recon_grid_unixbench_{machine}_{batch_stamp}"
        else:
            tok = safe_session_tag(str(pts_suite or "").replace("/", "_"))
            sub = f"router_recon_grid_{tok}_{machine}_{batch_stamp}"
        out_dir = (ds_root / "experiments" / sub).resolve()

    print(f"[grid] machine={machine!r} glob={glob_eff!r} out={out_dir}", file=sys.stderr)

    py = python_exe
    models_dir = out_dir / "trained_models"

    do_routers = args.stage in ("all", "routers")
    do_recon = args.stage in ("all", "reconstructors")
    do_grid = args.stage in ("all", "grid")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if do_routers or do_recon:
            models_dir.mkdir(parents=True, exist_ok=True)

    train_router_base = [
        py,
        str(ROUTER_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        args.benchmark,
        "--glob-pattern",
        glob_eff,
        "--machine",
        machine,
        "--top-k",
        str(args.top_k),
        "--mlp-epochs",
        str(args.mlp_epochs_router),
        "--mlp-hidden",
        str(args.mlp_hidden_router),
        "--gnn-emb-dim",
        str(args.gnn_emb_dim),
    ]
    if args.benchmark == "phoronix":
        train_router_base.extend(["--pts-suite", pts_suite or ""])
    if args.auto_install:
        train_router_base.append("--auto-install")

    train_k_max = int(args.train_k_max)

    train_recon_base = [
        py,
        str(RECON_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        args.benchmark,
        "--glob-pattern",
        glob_eff,
        "--machine",
        machine,
        "--skip-cv",
        "--no-uncertainty",
        "--train-aug",
        str(args.train_aug),
        "--train-k-min",
        str(args.train_k_min),
        "--train-k-max",
        str(train_k_max),
        "--eval-partial-k",
        str(args.eval_partial_k),
        "--mlp-epochs",
        str(args.mlp_epochs_recon),
        "--mlp-hidden",
        str(args.mlp_hidden_recon),
    ]
    if args.benchmark == "phoronix":
        train_recon_base.extend(
            [
                "--pts-suite",
                pts_suite or "",
                "--pts-suite-target",
                args.pts_suite_target,
            ]
        )
    if args.auto_install:
        train_recon_base.append("--auto-install")

    if do_routers:
        for mt, fname in ROUTER_SPECS:
            out_path = models_dir / fname
            cmd = [
                *train_router_base,
                "--model-type",
                mt,
                "--model-out",
                str(out_path),
            ]
            _run(cmd, dry_run=args.dry_run)
    if do_recon:
        for mt, fname in RECON_SPECS:
            out_path = models_dir / fname
            cmd = [
                *train_recon_base,
                "--model-type",
                mt,
                "--export-model",
                str(out_path),
            ]
            _run(cmd, dry_run=args.dry_run)

    exp_tokens = args.experiment_extra.split() if args.experiment_extra.strip() else []
    if args.benchmark == "unixbench" and args.sudo:
        exp_tokens.append("--sudo")
    if args.benchmark == "phoronix" and args.sudo_for_xi:
        exp_tokens.append("--sudo-for-xi")

    summarize = _summarize_unixbench if args.benchmark == "unixbench" else _summarize_pts
    exp_script = EXPERIMENT_UB if args.benchmark == "unixbench" else EXPERIMENT_PTS

    rows: list[dict[str, Any]] = []
    if do_grid:
        for r_mt, r_fn in ROUTER_SPECS:
            for c_mt, c_fn in RECON_SPECS:
                router_ckpt = models_dir / r_fn
                recon_ckpt = models_dir / c_fn
                tag = f"exp_{r_mt}__{c_mt}"
                exp_out = out_dir / f"{tag}.json"
                if not router_ckpt.is_file() and not args.dry_run:
                    print(f"Missing router checkpoint: {router_ckpt}", file=sys.stderr)
                    return 2
                if not recon_ckpt.is_file() and not args.dry_run:
                    print(f"Missing reconstruct checkpoint: {recon_ckpt}", file=sys.stderr)
                    return 2
                cmd = [
                    py,
                    str(exp_script),
                    "--router-model",
                    str(router_ckpt),
                    "--reconstruct-model",
                    str(recon_ckpt),
                    "--dataset-root",
                    str(ds_root),
                    "-o",
                    str(exp_out),
                ]
                if args.benchmark == "phoronix":
                    cmd.extend(
                        [
                            "--pts-mode",
                            args.pts_mode,
                            "--suite-full",
                            suite_full,
                        ]
                    )
                cmd.extend(exp_tokens)
                _run(cmd, dry_run=args.dry_run)
                if not args.dry_run:
                    row = {
                        "router_type": r_mt,
                        "reconstruct_type": c_mt,
                        **summarize(exp_out),
                    }
                    rows.append(row)

    def sort_key(r: dict[str, Any]) -> tuple[float, float]:
        err = r.get("suite_abs_err")
        te = r.get("suite_rel_err")
        err_v = float(err) if err is not None else float("inf")
        rel_v = float(te) if te is not None else float("inf")
        return (err_v, rel_v)

    ranked = sorted(rows, key=sort_key) if rows else []
    report: dict[str, Any] = {
        "schema": "moebench.experiment.router_recon_grid_summary.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(ds_root),
        "machine": machine,
        "local_host_slug": local_host_slug(),
        "benchmark": args.benchmark,
        "stage": args.stage,
        "glob_pattern": glob_eff,
        "pts_suite": pts_suite,
        "pts_suite_full": suite_full if args.benchmark == "phoronix" else None,
        "pts_mode": args.pts_mode if args.benchmark == "phoronix" else None,
        "models_dir": str(models_dir),
        "comparison_dir": str(out_dir),
        "routers": [m for m, _ in ROUTER_SPECS],
        "reconstructors": [m for m, _ in RECON_SPECS],
        "ranking_note": "Sorted by suite_abs_err then suite_rel_err (lower is better).",
        "per_combination": rows,
        "ranked": ranked,
    }
    summary_path = out_dir / "grid_summary.json"
    if do_grid and not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    if do_grid:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not args.dry_run:
            print(f"\nWrote summary: {summary_path}", file=sys.stderr)
    elif not args.dry_run:
        print(f"Stage {args.stage} finished; artifacts under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
