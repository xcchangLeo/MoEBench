#!/usr/bin/env python3
"""Train multiple router architectures (optional) and run Router+Reconstruction vs Full for each.

Produces a single comparison JSON summarizing timing and suite-score errors per model.
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
ROUTER_TRAIN = REPO_ROOT / "scripts" / "router_train.py"
EXPERIMENT = REPO_ROOT / "scripts" / "experiment_router_reconstruct_vs_full.py"

DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "router_lgbm.pkl"),
    ("mlp", "router_mlp.pt"),
    ("subset_sel", "router_subset.pt"),
    ("gnn_expert", "router_gnn.pt"),
)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _make_ablation_out_dir(
    ds_root: Path, rel_parent: str, stamp: str, *, dry_run: bool
) -> Path:
    """Create <dataset-root>/<rel_parent>/router_ablation_<stamp>/.

    If that path is not writable (e.g. experiments/ was created by sudo as root),
    fall back to <dataset-root>/ablation_runs/router_ablation_<stamp>/.
    """
    rel_parent = rel_parent.strip() or "experiments"
    out_dir = (ds_root / rel_parent / f"router_ablation_{stamp}").resolve()
    if dry_run:
        return out_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    except PermissionError:
        alt = (ds_root / "ablation_runs" / f"router_ablation_{stamp}").resolve()
        print(
            f"Warning: cannot create output under {out_dir.parent} (permission denied); "
            f"using {alt.parent} instead. Fix with: sudo chown -R $USER:$USER {out_dir.parent}",
            file=sys.stderr,
        )
        alt.mkdir(parents=True, exist_ok=True)
        return alt


def _summarize_experiment(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    comp = d.get("comparison") or {}
    scores = d.get("scores") or {}
    timing = d.get("timing_seconds") or {}
    return {
        "experiment_json": str(path),
        "router_model_type": d.get("router_model_type"),
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
        epilog=(
            "Examples (copy as ONE line, or keep each \\ at end of line with NO spaces after it):\n"
            "  %(prog)s --dataset-root dataset --reconstruct-model dataset/models/reconstruct_lgbm.pkl --sudo --auto-install\n"
            "Do not put `--experiment-extra` on its own line without a value; use `--sudo` instead of `--experiment-extra --sudo`.\n"
            "If mkdir fails under dataset/experiments (often root-owned after sudo runs), use `--experiments-parent ablation_runs` or fix ownership."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument(
        "--models-dir",
        type=str,
        default="",
        help="Where to store trained checkpoints (default: <dataset-root>/router_models/<stamp>/)",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="lightgbm,mlp,subset_sel,gnn_expert",
        help="Comma-separated subset of: lightgbm,mlp,subset_sel,gnn_expert",
    )
    ap.add_argument("--skip-train", action="store_true", help="Only run experiments (expect checkpoints to exist)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--reconstruct-model", type=str, default="dataset/models/reconstruct_lgbm.pkl")
    ap.add_argument("--train-extra", type=str, default="", help="Extra args passed to router_train.py (quoted string)")
    ap.add_argument(
        "--sudo",
        action="store_true",
        help="Forward --sudo to experiment_router_reconstruct_vs_full.py (avoids --experiment-extra parsing issues).",
    )
    ap.add_argument(
        "--experiment-extra",
        type=str,
        default="",
        help='Whitespace-separated extra args for the experiment script. Example: --experiment-extra="--copies 16" '
        "(must be one argv value; do not write `--experiment-extra --sudo` without quoting).",
    )
    ap.add_argument(
        "--experiments-parent",
        type=str,
        default="experiments",
        help="Subdirectory of --dataset-root for ablation outputs (default: experiments). "
        "If that directory is not writable, the script falls back to ablation_runs/.",
    )
    ap.add_argument("--glob-pattern", type=str, default="*/run-*.json")
    ap.add_argument("--mlp-epochs", type=int, default=200)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--gnn-emb-dim", type=int, default=12)
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ds_root = Path(args.dataset_root).resolve()
    if args.models_dir:
        models_dir = Path(args.models_dir).resolve()
    else:
        models_dir = (ds_root / "router_models" / stamp).resolve()
    out_dir = _make_ablation_out_dir(
        ds_root, args.experiments_parent, stamp, dry_run=args.dry_run
    )

    wanted = {x.strip() for x in args.models.split(",") if x.strip()}
    specs = [(mt, fn) for mt, fn in DEFAULT_MODELS if mt in wanted]
    if not specs:
        print("No valid models in --models", file=sys.stderr)
        return 2

    if not args.dry_run:
        models_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = args.train_extra.split() if args.train_extra.strip() else []
    exp_tokens = args.experiment_extra.split() if args.experiment_extra.strip() else []
    if args.sudo:
        exp_tokens.append("--sudo")

    py = sys.executable

    if not args.skip_train:
        for mt, fname in specs:
            out_path = models_dir / fname
            cmd = [
                py,
                str(ROUTER_TRAIN),
                "--dataset-root",
                str(ds_root),
                "--glob-pattern",
                args.glob_pattern,
                "--model-type",
                mt,
                "--model-out",
                str(out_path),
                "--top-k",
                str(args.top_k),
                "--mlp-epochs",
                str(args.mlp_epochs),
                "--mlp-hidden",
                str(args.mlp_hidden),
                "--gnn-emb-dim",
                str(args.gnn_emb_dim),
            ]
            if args.auto_install:
                cmd.append("--auto-install")
            cmd.extend(train_tokens)
            _run(cmd, dry_run=args.dry_run)

    rows: list[dict[str, Any]] = []
    for mt, fname in specs:
        ckpt = models_dir / fname
        if not ckpt.is_file() and not args.dry_run:
            print(f"Missing checkpoint (train first): {ckpt}", file=sys.stderr)
            return 2
        exp_out = out_dir / f"experiment_{mt}.json"
        cmd = [
            py,
            str(EXPERIMENT),
            "--router-model",
            str(ckpt),
            "--reconstruct-model",
            args.reconstruct_model,
            "-o",
            str(exp_out),
        ]
        cmd.extend(exp_tokens)
        _run(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            rows.append({"model_type": mt, **_summarize_experiment(exp_out)})

    report = {
        "schema": "moebench.experiment.router_ablation_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_dir": str(models_dir),
        "comparison_dir": str(out_dir),
        "models": [m for m, _ in specs],
        "per_model": rows,
    }
    summary_path = out_dir / "ablation_summary.json"
    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"\nWrote summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
