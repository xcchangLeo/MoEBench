"""Load a saved reconstruction model and predict full-suite scores."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from moebench.reconstruct.data import build_partial_feature_row_from_executed_tests
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

SCHEMA_V1 = "moebench.reconstruct.model.v1"  # keep in sync with scripts/reconstruct_train_eval.py


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
    if not isinstance(blob, dict) or blob.get("schema") != SCHEMA_V1:
        raise ValueError(f"Not a MoEBench reconstruction bundle v1: {p}")
    return blob


def predict_from_partial(
    bundle: dict[str, Any],
    xi: dict[str, Any],
    executed_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return predicted subtest indices (ordered) and suite Benchmarks Index."""
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

    if mt in ("lightgbm", "xgboost"):
        est = bundle.get("estimator")
        if est is None:
            raise ValueError("Bundle missing 'estimator'")
        pred = est.predict(x)[0]
    elif mt == "mlp":
        import torch
        import torch.nn as nn

        in_dim = int(bundle["in_dim"])
        hidden = int(bundle["mlp_hidden"])
        out_dim = int(bundle["out_dim"])
        sd = bundle["state_dict"]
        net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        net.load_state_dict(sd)
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(x.astype(np.float32))).numpy()[0]
    else:
        raise ValueError(f"Unknown model_type in bundle: {mt}")

    tids = list(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    if len(pred) != len(tids) + 1:
        raise ValueError(f"Prediction dim {len(pred)} != len(test_ids)+1 ({len(tids)+1})")

    sub: dict[str, float] = {}
    for i, tid in enumerate(tids):
        sub[tid] = float(pred[i])
    return {
        "subtest_index": sub,
        "suite_index": float(pred[-1]),
    }
