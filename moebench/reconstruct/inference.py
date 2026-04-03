"""Load a saved reconstruction model and predict full-suite scores."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from moebench.reconstruct.data import build_partial_feature_row_from_executed_tests
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

SCHEMA_V1 = "moebench.reconstruct.model.v1"
SCHEMA_V2 = "moebench.reconstruct.model.v2"


def _sklearn_multioutput_row(pred: Any) -> np.ndarray:
    """``predict(X)`` may be (n_samples, n_out) or (n_out,) for n_samples==1; never index [0] on 1D."""
    a = np.asarray(pred, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a[0]
    raise ValueError(f"Unexpected multi-output predict shape: {a.shape}")


def load_reconstruction_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    blob: Any
    if p.suffix in (".pt", ".pth"):
        import torch

        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(p, map_location="cpu")
    else:
        with open(p, "rb") as f:
            blob = pickle.load(f)
    if not isinstance(blob, dict):
        raise ValueError(f"Invalid reconstruction bundle: {p}")
    sch = blob.get("schema")
    if sch not in (SCHEMA_V1, SCHEMA_V2):
        raise ValueError(f"Not a MoEBench reconstruction bundle v1/v2: {p} (schema={sch!r})")
    return blob


def bundle_has_uncertainty(bundle: dict[str, Any]) -> bool:
    return (
        bundle.get("schema") == SCHEMA_V2
        or bool(bundle.get("uncertainty_estimator"))
        or bool(bundle.get("heteroscedastic"))
    )


def _mlp_forward_mean_only(
    x: np.ndarray,
    *,
    in_dim: int,
    hidden: int,
    out_dim: int,
    state_dict: dict[str, Any],
) -> np.ndarray:
    import torch
    import torch.nn as nn

    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )
    net.load_state_dict(state_dict)
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(x.astype(np.float32))).numpy()[0]


def _mlp_forward_heteroscedastic(
    x: np.ndarray,
    *,
    in_dim: int,
    hidden: int,
    out_dim: int,
    state_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn

    class Het(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.head_mean = nn.Linear(hidden, out_dim)
            self.head_logvar = nn.Linear(hidden, out_dim)

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.body(z)
            return self.head_mean(h), self.head_logvar(h)

    net = Het()
    net.load_state_dict(state_dict)
    net.eval()
    with torch.no_grad():
        xm = torch.from_numpy(x.astype(np.float32))
        # Ensure batch dimension so indexing [0] returns a vector.
        # When x is shape (in_dim,), torch Linear outputs (out_dim,) and [0] would be scalar.
        if xm.dim() == 1:
            xm = xm.unsqueeze(0)
        mean_t, logvar_t = net(xm)
        mean = mean_t.numpy()[0]
        lv = np.clip(logvar_t.numpy()[0], -10.0, 10.0)
        sigma = np.exp(0.5 * lv)
    return mean.astype(np.float64), sigma.astype(np.float64)


def predict_from_partial(
    bundle: dict[str, Any],
    xi: dict[str, Any],
    executed_tests: list[dict[str, Any]],
    *,
    return_uncertainty: bool = False,
) -> dict[str, Any]:
    """Return predicted subtest indices (ordered) and suite Benchmarks Index.

    If ``return_uncertainty`` is True, the bundle must provide uncertainty (v2 trees:
    ``uncertainty_estimator``; v2 MLP: heteroscedastic heads). Missing uncertainty raises.
    """
    log1p = bool(bundle.get("log1p_partial_index", False))
    row = build_partial_feature_row_from_executed_tests(
        xi,
        executed_tests,
        test_ids=tuple(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS),
        log1p_index=log1p,
    )
    if row is None:
        raise ValueError("Could not build reconstruction feature row (missing index/time?).")

    x = np.asarray([row], dtype=np.float64)
    mt = bundle.get("model_type")
    tids = list(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    n_out = len(tids) + 1

    sigma_vec: np.ndarray | None = None

    if mt in ("lightgbm", "xgboost"):
        est = bundle.get("estimator")
        if est is None:
            raise ValueError("Bundle missing 'estimator'")
        pred = _sklearn_multioutput_row(est.predict(x))
        if return_uncertainty:
            unc = bundle.get("uncertainty_estimator")
            if unc is None:
                raise ValueError("Bundle has no uncertainty_estimator (need v2 export with --uncertainty)")
            sigma_vec = _sklearn_multioutput_row(unc.predict(x))
    elif mt == "mlp":
        in_dim = int(bundle["in_dim"])
        hidden = int(bundle["mlp_hidden"])
        out_dim = int(bundle["out_dim"])
        sd = bundle["state_dict"]
        if bundle.get("heteroscedastic"):
            pred, sig = _mlp_forward_heteroscedastic(
                x, in_dim=in_dim, hidden=hidden, out_dim=out_dim, state_dict=sd
            )
            pred = pred.astype(np.float64)
            sigma_vec = sig.astype(np.float64) if return_uncertainty else None
        else:
            pred = _mlp_forward_mean_only(x, in_dim=in_dim, hidden=hidden, out_dim=out_dim, state_dict=sd)
            if return_uncertainty:
                raise ValueError("MLP bundle is v1 (homoscedastic); re-export with --uncertainty for v2")
    else:
        raise ValueError(f"Unknown model_type in bundle: {mt}")

    if len(pred) != n_out:
        raise ValueError(f"Prediction dim {len(pred)} != len(test_ids)+1 ({n_out})")

    sub: dict[str, float] = {}
    for i, tid in enumerate(tids):
        sub[tid] = float(pred[i])

    out: dict[str, Any] = {
        "subtest_index": sub,
        "suite_index": float(pred[-1]),
    }
    if return_uncertainty:
        assert sigma_vec is not None
        if len(sigma_vec) != n_out:
            raise ValueError("Uncertainty length mismatch")
        unc_sub: dict[str, float] = {}
        for i, tid in enumerate(tids):
            unc_sub[tid] = float(max(sigma_vec[i], 1e-9))
        sigma_suite = float(max(sigma_vec[-1], 1e-9))
        out["uncertainty_subtest"] = unc_sub
        out["uncertainty_suite"] = sigma_suite
        out["suite_confidence"] = float(1.0 / (1.0 + sigma_suite))
    return out
