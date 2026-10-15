"""Load a saved reconstruction model and predict full-suite scores."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from moebench.reconstruct.data import build_partial_feature_row_from_executed_tests
from moebench.probe.suite_aggregate import aggregate_suite_index
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

try:
    from moebench.phoronix.training_data import build_partial_feature_row_from_pts_executed
except ImportError:
    build_partial_feature_row_from_pts_executed = None  # type: ignore[misc, assignment]

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


def _normalize_sequential_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Training wraps ``nn.Sequential`` in ``self.net``; inference uses a bare Sequential."""
    if not state_dict:
        return state_dict
    if any(str(k).startswith("net.") for k in state_dict):
        return {str(k)[4:]: v for k, v in state_dict.items() if str(k).startswith("net.")}
    return state_dict


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
    net.load_state_dict(_normalize_sequential_state_dict(state_dict))
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
    tids = tuple(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    if bundle.get("benchmark") == "phoronix":
        if build_partial_feature_row_from_pts_executed is None:
            raise RuntimeError("PTS reconstruct helpers unavailable")
        row = build_partial_feature_row_from_pts_executed(
            xi,
            executed_tests,
            test_ids=tids,
            log1p_value=log1p,
        )
    else:
        row = build_partial_feature_row_from_executed_tests(
            xi,
            executed_tests,
            test_ids=tids,
            log1p_index=log1p,
        )
    if row is None:
        raise ValueError("Could not build reconstruction feature row (missing index/time?).")

    x = np.asarray([row], dtype=np.float64)
    mt = bundle.get("model_type")
    tids = list(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    n_out = len(tids) + 1

    sigma_vec: np.ndarray | None = None

    if mt in ("lightgbm", "lightgbm_small", "xgboost", "ridge"):
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

    suite_output_mode = str(bundle.get("suite_output_mode", "direct"))
    if suite_output_mode in (
        "observed_plus_imputed_geomean",
        "probe_plus_imputed_geomean",
    ):
        if bundle.get("benchmark", "unixbench") != "unixbench":
            raise ValueError(f"{suite_output_mode} is currently defined for UnixBench only")
        observed: dict[str, float] = {}
        for item in executed_tests:
            tid = str(item.get("test_id") or "")
            if not tid or item.get("missing"):
                continue
            index_detail = item.get("index_detail") or {}
            value = index_detail.get("index")
            if value is None:
                value = item.get("score")
            if value is not None:
                # Exact partial benchmark scores need no calibration. Micro
                # probes predict scores and use only training-fold factors.
                if suite_output_mode == "probe_plus_imputed_geomean":
                    factors = bundle.get("probe_prediction_calibration_factors") or {}
                    factor = factors.get(
                        tid, bundle.get("probe_prediction_calibration_factor", 1.0)
                    )
                else:
                    factor = 1.0
                observed[tid] = float(value) * float(factor)
        all_components = {
            tid: observed.get(tid, float(pred[i]))
            for i, tid in enumerate(tids)
        }
        suite_index = aggregate_suite_index(all_components, mode="geomean_raw_index")
    elif suite_output_mode == "observed_plus_training_geomean":
        if bundle.get("benchmark", "unixbench") != "unixbench":
            raise ValueError("observed_plus_training_geomean is currently defined for UnixBench only")
        observed: dict[str, float] = {}
        for item in executed_tests:
            tid = str(item.get("test_id") or "")
            if not tid or item.get("missing"):
                continue
            index_detail = item.get("index_detail") or {}
            value = index_detail.get("index")
            if value is None:
                value = item.get("score")
            if value is not None:
                observed[tid] = max(float(value), 1e-9)
        training_means = bundle.get("training_component_log_means") or {}
        all_components = {
            tid: observed.get(
                tid,
                float(np.exp(float(training_means.get(tid, np.log(max(float(pred[i]), 1e-9)))))),
            )
            for i, tid in enumerate(tids)
        }
        suite_index = aggregate_suite_index(all_components, mode="geomean_raw_index")
    elif suite_output_mode == "observed_plus_scaled_imputed_geomean":
        if bundle.get("benchmark", "unixbench") != "unixbench":
            raise ValueError("observed_plus_scaled_imputed_geomean is currently defined for UnixBench only")
        observed: dict[str, float] = {}
        ratios: list[float] = []
        for item in executed_tests:
            tid = str(item.get("test_id") or "")
            if not tid or item.get("missing") or tid not in tids:
                continue
            index_detail = item.get("index_detail") or {}
            value = index_detail.get("index")
            if value is None:
                value = item.get("score")
            if value is None:
                continue
            exact = max(float(value), 1e-9)
            observed[tid] = exact
            base = max(float(pred[tids.index(tid)]), 1e-9)
            ratios.append(exact / base)
        scale = float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else 1.0
        all_components = {
            tid: observed.get(tid, max(float(pred[i]) * scale, 1e-9))
            for i, tid in enumerate(tids)
        }
        suite_index = aggregate_suite_index(all_components, mode="geomean_raw_index")
    elif suite_output_mode == "reconstructed_geomean":
        if bundle.get("benchmark", "unixbench") != "unixbench":
            raise ValueError("reconstructed_geomean is currently defined for UnixBench only")
        suite_index = aggregate_suite_index(
            {tid: float(pred[i]) for i, tid in enumerate(tids)},
            mode="geomean_raw_index",
        )
    elif suite_output_mode == "probe_scaled_geomean":
        if bundle.get("benchmark", "unixbench") != "unixbench":
            raise ValueError("probe_scaled_geomean is currently defined for UnixBench only")
        ratios: list[float] = []
        for item in executed_tests:
            tid = str(item.get("test_id") or "")
            if tid not in tids or item.get("missing"):
                continue
            index_detail = item.get("index_detail") or {}
            value = index_detail.get("index")
            if value is None:
                value = item.get("score")
            base = float(pred[tids.index(tid)])
            if value is not None and float(value) > 0.0 and base > 0.0:
                ratios.append(float(value) / base)
        scale = float(np.exp(np.median(np.log(ratios)))) if ratios else 1.0
        suite_index = aggregate_suite_index(
            {tid: float(pred[i]) * scale for i, tid in enumerate(tids)},
            mode="geomean_raw_index",
        )
    elif suite_output_mode == "direct":
        suite_index = float(pred[-1])
    else:
        raise ValueError(f"Unknown suite_output_mode: {suite_output_mode!r}")

    out: dict[str, Any] = {
        "subtest_index": sub,
        "suite_index": suite_index,
        "suite_index_direct_head": float(pred[-1]),
        "suite_output_mode": suite_output_mode,
    }
    if suite_output_mode == "probe_scaled_geomean":
        out["probe_scale_anchor_count"] = len(ratios)
        out["probe_scale_factor"] = scale
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
