#!/usr/bin/env python3
"""Train a router for expert subset selection.

Training objectives:
  - lightgbm: LGBMRanker (lambdarank) on xi||one-hot expert, per-query groups.
  - mlp: pointwise MSE on relevance for xi||one-hot (same featurization as before).
  - subset_sel: xi -> logits over experts; soft cross-entropy to normalized relevance per system.
  - gnn_expert: simple message-passing on fixed expert graph; same listwise soft CE.

Outputs:
  - model checkpoint (.pkl for lightgbm, .pt for torch models)
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import resolve_glob_for_machine, resolve_training_machine


def _ensure_import(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as e:
        raise ImportError(f"Missing dependency '{module_name}'.") from e


def _maybe_auto_install(auto_install: bool, pkgs: list[str]) -> None:
    if not auto_install:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
    subprocess.check_call(cmd)


def _bootstrap_router_deps(auto_install: bool, model_type: str) -> None:
    pkgs = ["numpy", "scikit-learn"]
    if model_type == "lightgbm":
        pkgs.append("lightgbm")
    else:
        pkgs.append("torch")
    try:
        __import__("numpy")
        __import__("lightgbm" if model_type == "lightgbm" else "torch")
    except ImportError:
        if not auto_install:
            raise
        _maybe_auto_install(True, pkgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default="dataset", help="Path to dataset folder (with */run-*.json)")
    ap.add_argument(
        "--benchmark",
        type=str,
        choices=("unixbench", "phoronix"),
        default="unixbench",
        help="Training data: UnixBench runs or PTS (moebench.phoronix.dataset.v1)",
    )
    ap.add_argument(
        "--glob-pattern",
        type=str,
        default="",
        help="Glob under dataset-root (default: auto from benchmark + --pts-suite; see moebench.dataset_globs)",
    )
    ap.add_argument("--model-out", type=str, required=True, help="Where to save model checkpoint")
    ap.add_argument(
        "--model-type",
        type=str,
        choices=("lightgbm", "mlp", "subset_sel", "gnn_expert"),
        default="lightgbm",
    )
    ap.add_argument("--label-transform", type=str, choices=("none", "log1p"), default="log1p")
    ap.add_argument("--auto-install", action="store_true", help="If deps missing, attempt pip install")
    ap.add_argument(
        "--pts-suite",
        type=str,
        default=None,
        metavar="ID",
        help="With --benchmark phoronix: only use runs where yi.suite matches (e.g. pts/nvidia-gpu-compute)",
    )
    ap.add_argument(
        "--machine",
        type=str,
        default="",
        help="Train only on sessions from this host slug (default: current hostname; see moebench.dataset_machines)",
    )

    ap.add_argument("--top-k", type=int, default=3, help="For runtime selection output; stored in model")
    ap.add_argument("--lgbm-estimators", type=int, default=300)
    ap.add_argument("--lgbm-lr", type=float, default=0.05)
    ap.add_argument("--lgbm-leaves", type=int, default=31)
    ap.add_argument("--lgbm-min-child-samples", type=int, default=20)

    ap.add_argument("--mlp-hidden", type=int, default=64, help="Hidden size for mlp / subset_sel / gnn MLP layers")
    ap.add_argument("--gnn-emb-dim", type=int, default=12, help="Expert embedding dim for gnn_expert")
    ap.add_argument("--mlp-epochs", type=int, default=200)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    args = ap.parse_args()
    _bootstrap_router_deps(args.auto_install, args.model_type)

    from moebench.router.dataset_loader import (
        load_phoronix_dataset_for_router,
        load_unixbench_dataset_for_router,
    )
    from moebench.router.neural_routers import (
        train_expert_gnn,
        train_pointwise_mlp,
        train_subset_selection_router,
    )

    if args.benchmark == "phoronix" and not args.pts_suite:
        print(
            "phoronix training requires --pts-suite (e.g. cpu or pts/nvidia-gpu-compute) "
            "so experts and default session globs match collected data.",
            file=sys.stderr,
        )
        return 2

    machine = resolve_training_machine(args.machine or None)
    glob_eff = resolve_glob_for_machine(
        benchmark=args.benchmark,
        machine=machine,
        glob_pattern=args.glob_pattern or None,
        pts_suite=args.pts_suite,
    )
    print(f"[router_train] machine={machine!r} glob={glob_eff!r}", file=sys.stderr)

    if args.benchmark == "phoronix":
        ds = load_phoronix_dataset_for_router(
            args.dataset_root,
            glob_pattern=glob_eff,
            pts_suite=args.pts_suite,
            xi_vectorizer=None,
        )
    else:
        ds = load_unixbench_dataset_for_router(
            args.dataset_root,
            glob_pattern=glob_eff,
            xi_vectorizer=None,
        )

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == "lightgbm":
        y = list(ds.y)
        if args.label_transform == "log1p":
            y = [math.log1p(max(0.0, float(v))) for v in y]
        try:
            _ensure_import("lightgbm")
            import numpy as np  # noqa: F401
        except ImportError:
            _maybe_auto_install(args.auto_install, ["numpy", "scikit-learn", "lightgbm"])
        import numpy as np
        import lightgbm as lgb

        X = np.asarray(ds.X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        group = ds.group

        y_int = np.zeros_like(y_arr, dtype=np.int32)
        start = 0
        for g in group:
            labels = y_arr[start : start + g]
            order = np.argsort(labels)
            ranks = np.empty(g, dtype=np.int32)
            ranks[order] = np.arange(g, dtype=np.int32)
            y_int[start : start + g] = ranks
            start += g

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=args.lgbm_estimators,
            learning_rate=args.lgbm_lr,
            num_leaves=args.lgbm_leaves,
            min_child_samples=args.lgbm_min_child_samples,
            subsample=0.9,
            colsample_bytree=0.9,
            metric="ndcg",
        )
        ranker.fit(X, y_int, group=group)

        model_obj: dict[str, Any] = {
            "schema": "moebench.router.model.v1",
            "model_type": "lightgbm",
            "benchmark": args.benchmark,
            "feature_names": ds.feature_names,
            "expert_ids": ds.expert_ids,
            "expert_test_ids": ds.expert_test_ids,
            "top_k": args.top_k,
            "label_transform": args.label_transform,
            "label_discretization": "per-group-rank-integers-0..g-1",
            "ranker": ranker,
            "machine": machine,
        }
        if args.benchmark == "phoronix" and args.pts_suite:
            model_obj["pts_suite"] = args.pts_suite
        with open(out, "wb") as f:
            pickle.dump(model_obj, f)
        print(f"Wrote model: {out}")
        return 0

    # Torch models
    try:
        _ensure_import("torch")
        import numpy as np  # noqa: F401
    except ImportError:
        _maybe_auto_install(args.auto_install, ["numpy", "scikit-learn", "torch"])

    import torch
    from moebench.router.neural_routers import (
        train_expert_gnn,
        train_pointwise_mlp,
        train_subset_selection_router,
    )

    if args.model_type == "mlp":
        bundle = train_pointwise_mlp(
            ds,
            label_transform=args.label_transform,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
        )
    elif args.model_type == "subset_sel":
        bundle = train_subset_selection_router(
            ds,
            label_transform=args.label_transform,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
        )
    elif args.model_type == "gnn_expert":
        bundle = train_expert_gnn(
            ds,
            label_transform=args.label_transform,
            hidden=args.mlp_hidden,
            emb_dim=args.gnn_emb_dim,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
        )
    else:
        raise RuntimeError(f"Unhandled model_type: {args.model_type}")

    bundle["top_k"] = args.top_k
    bundle["label_transform"] = args.label_transform
    bundle["benchmark"] = args.benchmark
    bundle["machine"] = machine
    if args.benchmark == "phoronix" and args.pts_suite:
        bundle["pts_suite"] = args.pts_suite

    if out.suffix in (".pkl", ".pickle"):
        out = out.with_suffix(".pt")
        print(f"Note: torch router saved as {out} (.pt)", file=sys.stderr)

    try:
        torch.save(bundle, out, pickle_protocol=4)
    except TypeError:
        torch.save(bundle, out)
    print(f"Wrote model: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
