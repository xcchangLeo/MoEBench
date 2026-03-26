#!/usr/bin/env python3
"""Train a router for expert subset selection.

Training objective (ranking):
  - For each system (xi) we create items = experts (e_001..e_N).
  - Label y_i = relevance score derived from yi for that expert.
  - LightGBM Ranker trains to rank experts within each system query group.

MLP alternative:
  - Pointwise regression on relevance score with expert one-hot appended to xi vector.
  - Inference uses softmax across experts to output selection probabilities.

Outputs:
  - model checkpoint file (pickle or torch.save)
"""

from __future__ import annotations

import argparse
import json
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

from moebench.router.dataset_loader import load_unixbench_dataset_for_router


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


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default="dataset", help="Path to dataset folder (with */run-*.json)")
    ap.add_argument("--glob-pattern", type=str, default="*/run-*.json", help="Glob under dataset-root")
    ap.add_argument("--model-out", type=str, required=True, help="Where to save model checkpoint")
    ap.add_argument("--model-type", type=str, choices=("lightgbm", "mlp"), default="lightgbm")
    ap.add_argument("--label-transform", type=str, choices=("none", "log1p"), default="log1p")
    ap.add_argument("--auto-install", action="store_true", help="If deps missing, attempt pip install")

    # training params
    ap.add_argument("--top-k", type=int, default=3, help="For runtime selection output; stored in model")
    ap.add_argument("--lgbm-estimators", type=int, default=300)
    ap.add_argument("--lgbm-lr", type=float, default=0.05)
    ap.add_argument("--lgbm-leaves", type=int, default=31)
    ap.add_argument("--lgbm-min-child-samples", type=int, default=20)

    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--mlp-epochs", type=int, default=200)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    args = ap.parse_args()

    # Load dataset
    ds = load_unixbench_dataset_for_router(
        args.dataset_root,
        glob_pattern=args.glob_pattern,
        xi_vectorizer=None,
    )

    # Transform labels
    y = list(ds.y)
    if args.label_transform == "log1p":
        y = [math.log1p(max(0.0, float(v))) for v in y]

    # Ensure ML deps
    if args.model_type == "lightgbm":
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

        # LightGBM lambdarank expects integer labels (gain).
        # We discretize per-query by ranking labels within each group:
        # smallest label -> 0, largest label -> group_size-1.
        y_int = np.zeros_like(y_arr, dtype=np.int32)
        start = 0
        for g in group:
            labels = y_arr[start : start + g]
            order = np.argsort(labels)  # ascending
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

        model_obj = {
            "schema": "moebench.router.model.v1",
            "model_type": "lightgbm",
            "feature_names": ds.feature_names,
            "expert_ids": ds.expert_ids,
            "expert_test_ids": ds.expert_test_ids,
            "top_k": args.top_k,
            "label_transform": args.label_transform,
            "label_discretization": "per-group-rank-integers-0..g-1",
            "ranker": ranker,
        }
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(model_obj, f)
        print(f"Wrote model: {out}")
        return 0

    # MLP
    try:
        _ensure_import("torch")
        import numpy as np  # noqa: F401
    except ImportError:
        _maybe_auto_install(args.auto_install, ["numpy", "scikit-learn", "torch"])

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    X = np.asarray(ds.X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    device = torch.device("cpu")
    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y_arr).to(device).view(-1, 1)

    in_dim = x_t.shape[1]
    net = nn.Sequential(
        nn.Linear(in_dim, args.mlp_hidden),
        nn.ReLU(),
        nn.Linear(args.mlp_hidden, args.mlp_hidden),
        nn.ReLU(),
        nn.Linear(args.mlp_hidden, 1),
    ).to(device)

    opt = optim.Adam(net.parameters(), lr=args.mlp_lr)
    loss_fn = nn.MSELoss()

    net.train()
    for epoch in range(1, args.mlp_epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = net(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        if epoch % max(1, args.mlp_epochs // 10) == 0:
            print(f"epoch {epoch}/{args.mlp_epochs} loss={float(loss.item()):.6f}")

    model_obj = {
        "schema": "moebench.router.model.v1",
        "model_type": "mlp",
        "feature_names": ds.feature_names,
        "expert_ids": ds.expert_ids,
        "expert_test_ids": ds.expert_test_ids,
        "top_k": args.top_k,
        "label_transform": args.label_transform,
        "xi_feature_dim": len(ds.feature_names) - len(ds.expert_ids),
        "state_dict": net.state_dict(),
        "mlp_hidden": args.mlp_hidden,
    }
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_obj, out)
    print(f"Wrote model: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

