"""Shared router inference: map xi vector to per-expert scores for Top-K selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from moebench.router.feature_vectorizer import XiVectorizer


def softmax_list(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def predict_expert_scores(router_meta: dict[str, Any], xi: dict[str, Any]) -> tuple[list[float], list[float], list[str], list[str]]:
    """
    Returns (raw_scores, probabilities, expert_ids, expert_test_ids).
    """
    vec = XiVectorizer()
    xi_vec = vec.transform(xi)
    feature_mask = router_meta.get("xi_feature_mask")
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=np.float32)
        if mask.shape != (len(xi_vec),):
            raise ValueError(
                f"xi_feature_mask has shape {mask.shape}, expected {(len(xi_vec),)}"
            )
        xi_vec = (np.asarray(xi_vec, dtype=np.float32) * mask).tolist()
    xi_dim = len(vec.feature_names)
    expert_ids = router_meta["expert_ids"]
    expert_test_ids = router_meta["expert_test_ids"]
    n_experts = len(expert_ids)
    model_type = router_meta.get("model_type")

    scores: list[float] = []
    if model_type == "lightgbm":
        X_rows = []
        for ei in range(n_experts):
            onehot = [0.0] * n_experts
            onehot[ei] = 1.0
            X_rows.append(list(xi_vec) + onehot)
        X_np = np.asarray(X_rows, dtype=np.float32)
        if router_meta.get("input_mean") is not None:
            mean = np.asarray(router_meta["input_mean"], dtype=np.float32)
            scale = np.asarray(router_meta.get("input_scale"), dtype=np.float32)
            if mean.shape != (X_np.shape[1],) or scale.shape != (X_np.shape[1],):
                raise ValueError("router input normalization shape does not match feature rows")
            X_np = (X_np - mean) / np.maximum(scale, 1e-8)
        ranker = router_meta["ranker"]
        scores = [float(x) for x in ranker.predict(X_np)]
    elif model_type == "mlp":
        import torch
        import torch.nn as nn

        X_rows = []
        for ei in range(n_experts):
            onehot = [0.0] * n_experts
            onehot[ei] = 1.0
            X_rows.append(list(xi_vec) + onehot)
        X_np = np.asarray(X_rows, dtype=np.float32)
        if router_meta.get("input_mean") is not None:
            mean = np.asarray(router_meta["input_mean"], dtype=np.float32)
            scale = np.asarray(router_meta.get("input_scale"), dtype=np.float32)
            if mean.shape != (X_np.shape[1],) or scale.shape != (X_np.shape[1],):
                raise ValueError("router input normalization shape does not match feature rows")
            X_np = (X_np - mean) / np.maximum(scale, 1e-8)
        in_dim = xi_dim + n_experts
        hidden = int(router_meta.get("mlp_hidden", 64))
        net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        net.load_state_dict(router_meta["state_dict"])
        net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_np)
            out = net(x_t).view(-1).tolist()
        scores = [float(v) for v in out]
    elif model_type == "subset_sel":
        import torch

        from moebench.router.neural_routers import SubsetSelectionRouter

        hidden = int(router_meta.get("subset_hidden", 64))
        net = SubsetSelectionRouter(xi_dim, n_experts, hidden)
        net.load_state_dict(router_meta["state_dict"])
        net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(xi_vec, dtype=np.float32)).unsqueeze(0)
            logits = net(x_t).view(-1)
        scores = [float(x) for x in logits.tolist()]
    elif model_type == "gnn_expert":
        import torch

        from moebench.router.neural_routers import SimpleExpertGNN

        hidden = int(router_meta.get("gnn_hidden", 64))
        emb_dim = int(router_meta.get("gnn_emb_dim", 12))
        net = SimpleExpertGNN(xi_dim, n_experts, hidden, emb_dim=emb_dim)
        net.load_state_dict(router_meta["state_dict"])
        net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(xi_vec, dtype=np.float32)).unsqueeze(0)
            logits = net(x_t).view(-1)
        scores = [float(x) for x in logits.tolist()]
    else:
        raise RuntimeError(f"Unknown router model_type: {model_type}")

    probs = softmax_list(scores)
    return scores, probs, expert_ids, expert_test_ids


def select_top_k_from_probs(
    probs: list[float],
    expert_ids: list[str],
    expert_test_ids: list[str],
    top_k: int,
) -> tuple[list[str], list[str]]:
    n = len(expert_ids)
    top_k = max(1, min(int(top_k), n))
    ranked_idx = sorted(range(n), key=lambda i: probs[i], reverse=True)[:top_k]
    return [expert_ids[i] for i in ranked_idx], [expert_test_ids[i] for i in ranked_idx]
