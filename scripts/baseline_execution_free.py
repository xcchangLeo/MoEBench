#!/usr/bin/env python3
"""Execution-free performance prediction baselines (Wang 2019 / Tousi 2022 proxies).

Predict suite score from xi only (no partial benchmark execution), using
leave-one-session-out CV on each host's historical runs. Outputs JSON for
Table~3-style aggregation (Err.% and Time(s) per host).

Supported model variants (use ``--methods all`` to run every proxy):

* **Wang et al. (2019):** ``wang_dnn`` (2-layer MLP), ``wang_lr`` (linear regression)
* **Tousi et al. (2022):** ``tousi_rf``, ``tousi_dt``, ``tousi_mlp``, ``tousi_en``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    glob_for_machine,
    machine_from_session_tag,
    resolve_training_machine,
)
from moebench.paper_eval.xi_ablation import AblatedXiVectorizer
from moebench.phoronix.training_data import (
    canonical_test_ids_from_runs,
    collect_phoronix_run_paths,
    extract_targets_from_pts_dataset,
)
from moebench.reconstruct.data import (
    collect_unixbench_run_paths,
    extract_targets_from_dataset,
)
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

SUITE_CONFIG = {
    "unixbench": {
        "benchmark": "unixbench",
        "pts_suite": None,
        "label": "UnixBench",
    },
    "pts_cpu": {
        "benchmark": "phoronix",
        "pts_suite": "cpu",
        "label": "PTS-CPU",
    },
    "pts_gpu": {
        "benchmark": "phoronix",
        "pts_suite": "pts/nvidia-gpu-compute",
        "label": "PTS-GPU",
    },
}

# model_family drives the training backend
METHOD_META: dict[str, dict[str, Any]] = {
    "wang_dnn": {
        "paper": "wang2019",
        "paper_label": "Wang et al. (2019)",
        "table_label": "Wang-style (DNN)",
        "model_family": "mlp",
        "hidden": 128,
        "hidden_layers": 2,
        "epochs": 300,
        "lr": 1e-3,
    },
    "wang_lr": {
        "paper": "wang2019",
        "paper_label": "Wang et al. (2019)",
        "table_label": "Wang-style (LR)",
        "model_family": "linear",
    },
    "tousi_rf": {
        "paper": "tousi2022",
        "paper_label": "Tousi et al. (2022)",
        "table_label": "Tousi-style (RF)",
        "model_family": "random_forest",
    },
    "tousi_dt": {
        "paper": "tousi2022",
        "paper_label": "Tousi et al. (2022)",
        "table_label": "Tousi-style (DT)",
        "model_family": "decision_tree",
    },
    "tousi_mlp": {
        "paper": "tousi2022",
        "paper_label": "Tousi et al. (2022)",
        "table_label": "Tousi-style (MLP)",
        "model_family": "mlp",
        "hidden": 64,
        "hidden_layers": 1,
        "epochs": 150,
        "lr": 1e-3,
    },
    "tousi_en": {
        "paper": "tousi2022",
        "paper_label": "Tousi et al. (2022)",
        "table_label": "Tousi-style (Elastic-Net)",
        "model_family": "elastic_net",
    },
}

# Backward-compatible alias
METHOD_ALIASES = {
    "wang_mlp": "wang_dnn",
}

ALL_METHODS = list(METHOD_META.keys())
WANG_METHODS = [m for m, meta in METHOD_META.items() if meta["paper"] == "wang2019"]
TOUSI_METHODS = [m for m, meta in METHOD_META.items() if meta["paper"] == "tousi2022"]


def _resolve_method(name: str) -> str:
    key = name.strip()
    return METHOD_ALIASES.get(key, key)


def _load_run(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _suite_score(ds: dict[str, Any], *, suite_key: str, test_ids: tuple[str, ...]) -> float | None:
    if suite_key == "unixbench":
        out = extract_targets_from_dataset(ds, test_ids=test_ids)
    else:
        out = extract_targets_from_pts_dataset(ds, test_ids)
    if out is None:
        return None
    return float(out[1])


def _xi_wall_estimate(ds: dict[str, Any], *, default_s: float) -> float:
    xi = ds.get("xi") or {}
    dyn = xi.get("dynamic") or {}
    warmup = float(dyn.get("warmup_s") or 0.0)
    mem = dyn.get("memory_copy") or {}
    mem_elapsed = float(mem.get("memory_copy_elapsed_s") or 0.0)
    proc_s = 0.5
    est = warmup + mem_elapsed + proc_s
    return est if est > 0 else default_s


def _build_rows(
    paths: list[Path],
    *,
    suite_key: str,
    xi_mode: str,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if suite_key == "unixbench":
        test_ids = INDEX_SUITE_TEST_IDS
    else:
        test_ids = tuple(canonical_test_ids_from_runs(paths))

    vec = AblatedXiVectorizer(xi_mode)
    rows: list[dict[str, Any]] = []
    pts_suite = SUITE_CONFIG[suite_key]["pts_suite"]

    for path in paths:
        ds = _load_run(path)
        xi = ds.get("xi")
        if not xi:
            continue
        y = _suite_score(ds, suite_key=suite_key, test_ids=test_ids)
        if y is None:
            continue
        rows.append(
            {
                "path": str(path),
                "session": path.parent.name,
                "machine": machine_from_session_tag(path.parent.name, pts_suite=pts_suite),
                "x": vec.transform(xi),
                "y": float(y),
                "xi_wall_s": _xi_wall_estimate(ds, default_s=3.0),
            }
        )
    if not rows:
        raise RuntimeError("No usable runs with xi + suite score")
    return rows, test_ids


def _cv_folds(rows: list[dict[str, Any]]) -> tuple[list[tuple[list[int], list[int]]], str]:
    sess_map: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        sess_map.setdefault(row["session"], []).append(i)

    if len(sess_map) >= 2:
        folds = []
        for test_idx in sess_map.values():
            test_set = set(test_idx)
            train_idx = [i for i in range(len(rows)) if i not in test_set]
            folds.append((train_idx, test_idx))
        return folds, "leave_one_session_out"

    if len(rows) >= 2:
        warnings.warn(
            f"Only one session ({next(iter(sess_map))}); using leave-one-run-out. "
            "For paper Table 3, merge all 5 collection sessions per host.",
            stacklevel=2,
        )
        return [([i for i in range(len(rows)) if i != j], [j]) for j in range(len(rows))], "leave_one_run_out"

    raise RuntimeError("Need at least 2 runs (ideally 5 sessions x 5 runs) for cross-validation")


def _fit_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden: int,
    hidden_layers: int,
    epochs: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cpu")
    x_t = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)
    xv = torch.from_numpy(x_test.astype(np.float32)).to(device)

    n_in = x_train.shape[1]
    layers: list[nn.Module] = []
    width_in = n_in
    for _ in range(max(1, hidden_layers)):
        layers.extend([nn.Linear(width_in, hidden), nn.ReLU()])
        width_in = hidden
    layers.append(nn.Linear(width_in, 1))

    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(xv).cpu().numpy().reshape(-1)
    return out.astype(np.float64)


def _fit_predict_sklearn(
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    rf_estimators: int,
) -> np.ndarray:
    if family == "linear":
        from sklearn.linear_model import LinearRegression

        est = LinearRegression()
    elif family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        est = RandomForestRegressor(n_estimators=rf_estimators, random_state=seed, n_jobs=-1)
    elif family == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor

        est = DecisionTreeRegressor(random_state=seed)
    elif family == "elastic_net":
        from sklearn.linear_model import ElasticNet

        est = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=5000)
    else:
        raise ValueError(f"unknown sklearn family: {family!r}")

    est.fit(x_train, y_train)
    return est.predict(x_test).astype(np.float64)


def _method_train_kwargs(method: str, *, hidden: int, epochs: int, lr: float) -> dict[str, Any]:
    meta = METHOD_META[method]
    if meta["model_family"] != "mlp":
        return {}
    if method == "wang_dnn":
        return {
            "hidden": hidden,
            "hidden_layers": meta["hidden_layers"],
            "epochs": epochs,
            "lr": lr,
        }
    return {
        "hidden": meta["hidden"],
        "hidden_layers": meta["hidden_layers"],
        "epochs": meta["epochs"],
        "lr": meta["lr"],
    }


def _predict_fold(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    train_kwargs: dict[str, Any],
    rf_estimators: int,
    seed: int,
) -> np.ndarray:
    meta = METHOD_META[method]
    family = meta["model_family"]
    if family == "mlp":
        return _fit_predict_mlp(
            x_train,
            y_train,
            x_test,
            hidden=int(train_kwargs["hidden"]),
            hidden_layers=int(train_kwargs["hidden_layers"]),
            epochs=int(train_kwargs["epochs"]),
            lr=float(train_kwargs["lr"]),
            seed=seed,
        )
    return _fit_predict_sklearn(
        family,
        x_train,
        y_train,
        x_test,
        seed=seed,
        rf_estimators=rf_estimators,
    )


def _run_method(
    method: str,
    rows: list[dict[str, Any]],
    *,
    train_kwargs: dict[str, Any],
    rf_estimators: int,
    seed: int,
) -> dict[str, Any]:
    meta = METHOD_META[method]
    folds, cv_mode = _cv_folds(rows)
    per_run: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if not train_idx:
            continue
        x_train = np.array([rows[i]["x"] for i in train_idx], dtype=np.float64)
        y_train = np.array([rows[i]["y"] for i in train_idx], dtype=np.float64)
        x_test = np.array([rows[i]["x"] for i in test_idx], dtype=np.float64)
        y_test = np.array([rows[i]["y"] for i in test_idx], dtype=np.float64)

        y_pred = _predict_fold(
            method,
            x_train,
            y_train,
            x_test,
            train_kwargs=train_kwargs,
            rf_estimators=rf_estimators,
            seed=seed + fold_i,
        )

        for j, idx in enumerate(test_idx):
            yt = float(y_test[j])
            yp = float(y_pred[j])
            rel = abs(yt - yp) / max(abs(yt), 1e-9)
            row = rows[idx]
            per_run.append(
                {
                    "path": row["path"],
                    "session": row["session"],
                    "y_true": yt,
                    "y_pred": yp,
                    "suite_rel_err": rel,
                    "xi_wall_s": row["xi_wall_s"],
                    "fold": fold_i,
                }
            )

    rels = [r["suite_rel_err"] for r in per_run]
    walls = [r["xi_wall_s"] for r in per_run]
    out_meta = {
        k: v
        for k, v in METHOD_META[method].items()
        if k not in ("model_family", "hidden", "hidden_layers", "epochs", "lr")
    }
    if meta["model_family"] == "mlp":
        out_meta["train_config"] = train_kwargs
    return {
        "method": method,
        "model_family": meta["model_family"],
        **out_meta,
        "mean_suite_rel_err": float(np.mean(rels)),
        "mean_suite_rel_err_pct": float(100.0 * np.mean(rels)),
        "median_xi_wall_s": float(statistics.median(walls)),
        "n_runs": len(per_run),
        "n_folds": len(folds),
        "cv_mode": cv_mode,
        "per_run": per_run,
    }


def _collect_paths(dataset_root: Path, *, suite_key: str, machine: str) -> list[Path]:
    cfg = SUITE_CONFIG[suite_key]
    glob_pat = glob_for_machine(
        benchmark=cfg["benchmark"],
        machine=machine,
        pts_suite=cfg["pts_suite"],
    )
    if cfg["benchmark"] == "unixbench":
        return collect_unixbench_run_paths(dataset_root, glob_pattern=glob_pat)
    return collect_phoronix_run_paths(
        dataset_root,
        glob_pattern=glob_pat,
        pts_suite=cfg["pts_suite"],
    )


def _parse_methods(raw: str) -> list[str]:
    text = raw.strip().lower()
    if text == "all":
        return list(ALL_METHODS)
    if text == "wang":
        return list(WANG_METHODS)
    if text == "tousi":
        return list(TOUSI_METHODS)
    out: list[str] = []
    for part in raw.split(","):
        m = _resolve_method(part)
        if m not in METHOD_META:
            raise SystemExit(f"Unknown method {part!r}; choose from {ALL_METHODS} or presets: all, wang, tousi")
        if m not in out:
            out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(SUITE_CONFIG))
    ap.add_argument("--machine", default="", help="Host slug (default: local hostname)")
    ap.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    ap.add_argument(
        "--xi-mode",
        default="static_hw_only",
        choices=["static_hw_only", "full"],
    )
    ap.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method ids, or presets: all, wang, tousi "
        f"({', '.join(ALL_METHODS)})",
    )
    ap.add_argument("--hidden", type=int, default=128, help="wang_dnn hidden size override")
    ap.add_argument("--epochs", type=int, default=300, help="wang_dnn epochs override")
    ap.add_argument("--lr", type=float, default=1e-3, help="wang_dnn learning rate override")
    ap.add_argument("--rf-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    machine = resolve_training_machine(args.machine or None)
    cfg = SUITE_CONFIG[args.suite]
    methods = _parse_methods(args.methods)
    paths = _collect_paths(args.dataset_root, suite_key=args.suite, machine=machine)
    rows, test_ids = _build_rows(paths, suite_key=args.suite, xi_mode=args.xi_mode)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (
        args.dataset_root / "experiments" / f"exec_free_{args.suite}_{machine}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method in methods:
        res = _run_method(
            method,
            rows,
            train_kwargs=_method_train_kwargs(method, hidden=args.hidden, epochs=args.epochs, lr=args.lr),
            rf_estimators=args.rf_estimators,
            seed=args.seed,
        )
        results.append(res)
        with open(out_dir / f"{method}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    cv_mode = results[0]["cv_mode"] if results else "unknown"
    ranked = sorted(results, key=lambda r: r["mean_suite_rel_err_pct"], reverse=True)
    summary = {
        "schema": "moebench.experiment.exec_free_baselines.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "suite_label": cfg["label"],
        "machine": machine,
        "xi_mode": args.xi_mode,
        "cv_mode": cv_mode,
        "n_runs": len(rows),
        "n_sessions": len({r["session"] for r in rows}),
        "test_ids": list(test_ids),
        "methods_run": methods,
        "methods": results,
        "ranked_by_err_pct_desc": [
            {
                "method": r["method"],
                "table_label": r["table_label"],
                "paper": r["paper"],
                "err_pct": r["mean_suite_rel_err_pct"],
                "time_s": r["median_xi_wall_s"],
            }
            for r in ranked
        ],
        "table3_cells": {
            r["method"]: {"err_pct": r["mean_suite_rel_err_pct"], "time_s": r["median_xi_wall_s"]}
            for r in results
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["ranked_by_err_pct_desc"], indent=2, ensure_ascii=False))
    print(f"\nWrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
