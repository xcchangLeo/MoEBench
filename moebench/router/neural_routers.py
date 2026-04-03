"""PyTorch routers: pointwise MLP (xi+onehot), subset-selection (xi→logits), simple expert GNN."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moebench.router.dataset_loader import RouterDataset


def apply_label_transform(y: float, label_transform: str) -> float:
    if label_transform == "log1p":
        return math.log1p(max(0.0, float(y)))
    return float(y)


def dataset_to_group_tensors(
    ds: RouterDataset,
    *,
    label_transform: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return X_grouped (Q, n, feat_dim), y_grouped (Q, n), xi_dim, n_experts."""
    n_experts = len(ds.expert_ids)
    xi_dim = ds.meta["xi_dim"]
    X = np.asarray(ds.X, dtype=np.float32)
    y_list = [apply_label_transform(v, label_transform) for v in ds.y]
    y = np.asarray(y_list, dtype=np.float32)
    q = len(ds.group)
    if X.shape[0] != q * n_experts:
        raise RuntimeError("Unexpected router dataset shape")
    Xg = X.reshape(q, n_experts, -1)
    yg = y.reshape(q, n_experts)
    return Xg, yg, xi_dim, n_experts


def train_pointwise_mlp(
    ds: RouterDataset,
    *,
    label_transform: str,
    hidden: int,
    epochs: int,
    lr: float,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Same as legacy router MLP: xi || onehot -> scalar relevance."""
    dev = device or torch.device("cpu")
    Xg, yg, xi_dim, n_experts = dataset_to_group_tensors(ds, label_transform=label_transform)
    q, n, feat = Xg.shape
    if feat != xi_dim + n_experts:
        raise RuntimeError("Feature dim mismatch")
    x_t = torch.from_numpy(Xg.reshape(q * n, feat)).to(dev)
    y_t = torch.from_numpy(yg.reshape(q * n)).to(dev).view(-1, 1)

    net = nn.Sequential(
        nn.Linear(feat, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = net(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"  [mlp] epoch {epoch}/{epochs} loss={float(loss.item()):.6f}")

    return {
        "schema": "moebench.router.model.v1",
        "model_type": "mlp",
        "feature_names": ds.feature_names,
        "expert_ids": ds.expert_ids,
        "expert_test_ids": ds.expert_test_ids,
        "state_dict": net.state_dict(),
        "mlp_hidden": hidden,
        "xi_feature_dim": xi_dim,
        "label_transform": label_transform,
    }


class SubsetSelectionRouter(nn.Module):
    """Map system features to one logit per expert; train with soft cross-entropy to relevance distribution."""

    def __init__(self, xi_dim: int, n_experts: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xi_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """xi: (B, xi_dim) -> logits (B, n_experts)."""
        return self.net(xi)


def train_subset_selection_router(
    ds: RouterDataset,
    *,
    label_transform: str,
    hidden: int,
    epochs: int,
    lr: float,
    device: torch.device | None = None,
) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    Xg, yg, xi_dim, n_experts = dataset_to_group_tensors(ds, label_transform=label_transform)
    q = Xg.shape[0]
    net = SubsetSelectionRouter(xi_dim, n_experts, hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for i in range(q):
            xi = torch.from_numpy(Xg[i, 0, :xi_dim]).unsqueeze(0).to(dev)
            logits = net(xi).view(-1)
            rel = torch.from_numpy(yg[i]).to(dev).clamp_min(0.0)
            rel = rel / (rel.sum() + 1e-8)
            loss_acc = loss_acc - (rel * F.log_softmax(logits, dim=-1)).sum()
        loss = loss_acc / q
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"  [subset_sel] epoch {epoch}/{epochs} loss={float(loss.item()):.6f}")

    return {
        "schema": "moebench.router.model.v1",
        "model_type": "subset_sel",
        "feature_names": ds.feature_names,
        "expert_ids": ds.expert_ids,
        "expert_test_ids": ds.expert_test_ids,
        "state_dict": net.state_dict(),
        "subset_hidden": hidden,
        "xi_feature_dim": xi_dim,
        "label_transform": label_transform,
    }


class SimpleExpertGNN(nn.Module):
    """Fixed undirected expert graph; node features = learned expert embedding || projected xi."""

    def __init__(self, xi_dim: int, n_experts: int, hidden: int, emb_dim: int = 12) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.hidden = hidden
        self.expert_emb = nn.Embedding(n_experts, emb_dim)
        self.proj_xi = nn.Linear(xi_dim, hidden)
        d_in = emb_dim + hidden
        self.lin1 = nn.Linear(d_in, hidden)
        self.msg = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        adj = torch.ones(n_experts, n_experts) + torch.eye(n_experts)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        a_norm = adj / deg
        self.register_buffer("A_norm", a_norm)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """xi: (B, xi_dim) -> logits (B, n_experts)."""
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        b, _xd = xi.shape
        xh = self.proj_xi(xi)  # (b, hidden)
        emb = self.expert_emb.weight.unsqueeze(0).expand(b, self.n_experts, -1)
        xh_n = xh.unsqueeze(1).expand(b, self.n_experts, self.hidden)
        h = torch.cat([emb, xh_n], dim=-1)
        h = F.relu(self.lin1(h))
        for _ in range(2):
            agg = torch.matmul(self.A_norm, h)
            h = F.relu(self.msg(agg)) + h
        logits = self.out(h).squeeze(-1)
        return logits


def train_expert_gnn(
    ds: RouterDataset,
    *,
    label_transform: str,
    hidden: int,
    emb_dim: int,
    epochs: int,
    lr: float,
    device: torch.device | None = None,
) -> dict[str, Any]:
    dev = device or torch.device("cpu")
    Xg, yg, xi_dim, n_experts = dataset_to_group_tensors(ds, label_transform=label_transform)
    q = Xg.shape[0]
    net = SimpleExpertGNN(xi_dim, n_experts, hidden, emb_dim=emb_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for i in range(q):
            xi = torch.from_numpy(Xg[i, 0, :xi_dim]).unsqueeze(0).to(dev)
            logits = net(xi).view(-1)
            rel = torch.from_numpy(yg[i]).to(dev).clamp_min(0.0)
            rel = rel / (rel.sum() + 1e-8)
            loss_acc = loss_acc - (rel * F.log_softmax(logits, dim=-1)).sum()
        loss = loss_acc / q
        loss.backward()
        opt.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"  [gnn_expert] epoch {epoch}/{epochs} loss={float(loss.item()):.6f}")

    return {
        "schema": "moebench.router.model.v1",
        "model_type": "gnn_expert",
        "feature_names": ds.feature_names,
        "expert_ids": ds.expert_ids,
        "expert_test_ids": ds.expert_test_ids,
        "state_dict": net.state_dict(),
        "gnn_hidden": hidden,
        "gnn_emb_dim": emb_dim,
        "xi_feature_dim": xi_dim,
        "label_transform": label_transform,
    }


