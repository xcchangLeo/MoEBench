#!/usr/bin/env python3
"""Train router × reconstruction model pairs and run end-to-end UnixBench experiments.

Trains **3 routers** (LightGBM, MLP, GNN / ``gnn_expert``) and **3 reconstructors**
(XGBoost, LightGBM, MLP), then runs ``experiment_router_reconstruct_vs_full.py`` for each
of the **9** combinations and writes a ranked summary JSON.

Glob defaults match data collection (see ``moebench.dataset_globs``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_globs import resolve_glob_pattern

ROUTER_TRAIN = REPO_ROOT / "scripts" / "router_train.py"
RECON_TRAIN = REPO_ROOT / "scripts" / "reconstruct_train_eval.py"
EXPERIMENT = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full.py"

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


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _summarize_experiment(path: Path) -> dict[str, Any]:
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
        "partial_ub_s": timing.get("partial_unixbench"),
        "full_ub_s": timing.get("full_unixbench"),
        "xi_s": timing.get("xi_collection"),
        "time_saved_ub_s": comp.get("benchmark_time_saved_seconds_vs_full"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument(
        "--out-parent",
        type=str,
        default="",
        help="Output directory (default: <dataset-root>/experiments/router_recon_grid_<UTC>/)",
    )
    ap.add_argument(
        "--glob-pattern",
        type=str,
        default="",
        help="Router + reconstruct training glob (default: */run-*.json = UnixBench collection layout)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Only run 9 experiments (expect checkpoints under --out-parent)")
    ap.add_argument("--sudo", action="store_true", help="Forward to experiment_router_reconstruct_vs_full.py")
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
    ap.add_argument("--experiment-extra", type=str, default="", help="Extra args for experiment script (one shell token)")
    args = ap.parse_args()

    if args.skip_train and not args.out_parent.strip():
        print(
            "--skip-train requires --out-parent=<existing grid dir> with trained_models/ inside.",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ds_root = Path(args.dataset_root).resolve()
    if args.out_parent.strip():
        out_dir = Path(args.out_parent).resolve()
    else:
        out_dir = (ds_root / "experiments" / f"router_recon_grid_{stamp}").resolve()
    glob_eff = resolve_glob_pattern(
        benchmark="unixbench",
        glob_pattern=args.glob_pattern or None,
        pts_suite=None,
    )

    py = sys.executable
    models_dir = out_dir / "trained_models"
    if not args.dry_run and not args.skip_train:
        models_dir.mkdir(parents=True, exist_ok=True)

    train_router_base = [
        py,
        str(ROUTER_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        "unixbench",
        "--glob-pattern",
        glob_eff,
        "--top-k",
        str(args.top_k),
        "--mlp-epochs",
        str(args.mlp_epochs_router),
        "--mlp-hidden",
        str(args.mlp_hidden_router),
        "--gnn-emb-dim",
        str(args.gnn_emb_dim),
    ]
    if args.auto_install:
        train_router_base.append("--auto-install")

    train_recon_base = [
        py,
        str(RECON_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        "unixbench",
        "--glob-pattern",
        glob_eff,
        "--skip-cv",
        "--no-uncertainty",
        "--train-aug",
        str(args.train_aug),
        "--train-k-min",
        str(args.train_k_min),
        "--train-k-max",
        str(args.train_k_max),
        "--eval-partial-k",
        str(args.eval_partial_k),
        "--mlp-epochs",
        str(args.mlp_epochs_recon),
        "--mlp-hidden",
        str(args.mlp_hidden_recon),
    ]
    if args.auto_install:
        train_recon_base.append("--auto-install")

    if not args.skip_train:
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
    if args.sudo:
        exp_tokens.append("--sudo")

    rows: list[dict[str, Any]] = []
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
                str(EXPERIMENT),
                "--router-model",
                str(router_ckpt),
                "--reconstruct-model",
                str(recon_ckpt),
                "--dataset-root",
                str(ds_root),
                "-o",
                str(exp_out),
            ]
            cmd.extend(exp_tokens)
            _run(cmd, dry_run=args.dry_run)
            if not args.dry_run:
                row = {
                    "router_type": r_mt,
                    "reconstruct_type": c_mt,
                    **_summarize_experiment(exp_out),
                }
                rows.append(row)

    def sort_key(r: dict[str, Any]) -> tuple[float, float]:
        err = r.get("suite_abs_err")
        te = r.get("suite_rel_err")
        err_v = float(err) if err is not None else float("inf")
        rel_v = float(te) if te is not None else float("inf")
        return (err_v, rel_v)

    ranked = sorted(rows, key=sort_key) if rows else []
    report = {
        "schema": "moebench.experiment.router_recon_grid_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(ds_root),
        "glob_pattern": glob_eff,
        "benchmark": "unixbench",
        "models_dir": str(models_dir),
        "comparison_dir": str(out_dir),
        "routers": [m for m, _ in ROUTER_SPECS],
        "reconstructors": [m for m, _ in RECON_SPECS],
        "ranking_note": "Sorted by suite_absolute_error then suite_relative_error (lower is better).",
        "per_combination": rows,
        "ranked": ranked,
    }
    summary_path = out_dir / "grid_summary.json"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"\nWrote summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
