#!/usr/bin/env python3
"""Train + cross-validate UnixBench reconstruction models.

Input: system features (xi) + partial subtest results (executed subset: index + wall time).
Target: full 12 subtest index scores + system Benchmarks Index (suite total).

Simulates MoE partial runs using held-out random subsets; compares predicted vs full-run
ground truth and reports time saved vs full ti totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_globs import resolve_glob_pattern
from moebench.phoronix.training_data import (
    build_augmented_train_matrix_pts,
    canonical_test_ids_from_runs,
    collect_phoronix_run_paths,
    extract_targets_from_pts_dataset,
    full_suite_wall_seconds_pts,
)
from moebench.reconstruct.data import (
    build_partial_feature_row,
    collect_unixbench_run_paths,
    extract_targets_from_dataset,
    full_suite_wall_seconds,
    partial_wall_seconds,
)
from moebench.reconstruct.inference import SCHEMA_V1, SCHEMA_V2
from moebench.router.feature_vectorizer import XiVectorizer
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS


def _ensure_import(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as e:
        raise ImportError(f"Missing dependency: {name}") from e


def _maybe_auto_install(flag: bool, pkgs: list[str]) -> None:
    if not flag:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    c = np.corrcoef(ra.astype(float), rb.astype(float))[0, 1]
    return float(c) if not math.isnan(float(c)) else 0.0


def kendall_tau_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b style count without tie correction (fine for continuous scores)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                conc += 1
            else:
                disc += 1
    den = conc + disc
    return (conc - disc) / den if den else float("nan")


def stable_seed(parts: tuple[Any, ...]) -> int:
    h = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) % (2**31)


def pts_suite_target_from_indices(indices: list[float], mode: str) -> float:
    """Aggregate PTS per-profile primary values into one suite-level target."""
    vals = [float(v) for v in indices]
    if not vals:
        return 0.0
    if mode == "logmean":
        return float(math.expm1(sum(math.log1p(max(0.0, v)) for v in vals) / len(vals)))
    return float(sum(vals) / len(vals))


def _sklearn_base_estimator(
    model_name: str,
    *,
    auto_install: bool,
    lgbm_estimators: int,
    xgb_estimators: int,
    n_estimators_override: int | None = None,
) -> Any:
    ne_l = int(n_estimators_override) if n_estimators_override is not None else int(lgbm_estimators)
    ne_x = int(n_estimators_override) if n_estimators_override is not None else int(xgb_estimators)
    if model_name == "lightgbm":
        _maybe_auto_install(auto_install, ["lightgbm"])
        _ensure_import("lightgbm")
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=ne_l,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
        )
    if model_name == "xgboost":
        _maybe_auto_install(auto_install, ["xgboost"])
        _ensure_import("xgboost")
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=ne_x,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            tree_method="hist",
        )
    raise ValueError(model_name)


def fit_sklearn_multioutput(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    auto_install: bool,
    lgbm_estimators: int,
    xgb_estimators: int,
    n_estimators_override: int | None = None,
) -> Any:
    _ensure_import("sklearn")
    from sklearn.multioutput import MultiOutputRegressor

    base = _sklearn_base_estimator(
        model_name,
        auto_install=auto_install,
        lgbm_estimators=lgbm_estimators,
        xgb_estimators=xgb_estimators,
        n_estimators_override=n_estimators_override,
    )
    mor = MultiOutputRegressor(base)
    mor.fit(x_train, y_train)
    return mor


def fit_sklearn_uncertainty_estimator(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mor_main: Any,
    *,
    auto_install: bool,
    lgbm_estimators: int,
    xgb_estimators: int,
) -> Any:
    """Predict per-target expected |residual| (proxy σ) for active sampling / confidence."""
    pred = mor_main.predict(x_train)
    resid = np.abs(y_train.astype(np.float64) - pred.astype(np.float64))
    resid = np.maximum(resid, 1e-6)
    n_unc = max(64, (lgbm_estimators if model_name == "lightgbm" else xgb_estimators) // 2)
    return fit_sklearn_multioutput(
        model_name,
        x_train,
        resid,
        auto_install=auto_install,
        lgbm_estimators=lgbm_estimators,
        xgb_estimators=xgb_estimators,
        n_estimators_override=n_unc,
    )


def fit_predict_sklearn_multioutput(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    auto_install: bool,
    lgbm_estimators: int,
    xgb_estimators: int,
) -> np.ndarray:
    mor = fit_sklearn_multioutput(
        model_name,
        x_train,
        y_train,
        auto_install=auto_install,
        lgbm_estimators=lgbm_estimators,
        xgb_estimators=xgb_estimators,
        n_estimators_override=None,
    )
    return mor.predict(x_val).astype(np.float64)


def build_augmented_train_matrix(
    rows_meta: list[dict[str, Any]],
    vec: XiVectorizer,
    test_ids: list[str],
    rng: np.random.RandomState,
    train_aug: int,
    train_k_min: int,
    train_k_max: int,
    log1p_partial_index: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for i in range(len(rows_meta)):
        meta = rows_meta[i]
        ds = meta["ds"]
        for _ in range(train_aug):
            k = rng.randint(train_k_min, train_k_max + 1)
            ex = set(rng.choice(test_ids, size=k, replace=False).tolist())
            row = build_partial_feature_row(
                ds,
                ex,
                xi_vectorizer=vec,
                log1p_index=log1p_partial_index,
            )
            if row is None:
                continue
            x_rows.append(row)
            y_rows.append(meta["y"].tolist())
    if not x_rows:
        raise RuntimeError("No training rows for reconstruction model")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def train_mlp_export_bundle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    auto_install: bool,
    log1p_partial_index: bool,
    test_ids: list[str],
    benchmark: str = "unixbench",
) -> dict[str, Any]:
    _maybe_auto_install(auto_install, ["torch"])
    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    x_t = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    n_in = x_train.shape[1]
    n_out = y_train.shape[1]

    class MExport(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_out),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.net(z)

    model = MExport().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
    return {
        "schema": SCHEMA_V1,
        "model_type": "mlp",
        "state_dict": model.state_dict(),
        "mlp_hidden": hidden,
        "in_dim": n_in,
        "out_dim": n_out,
        "log1p_partial_index": log1p_partial_index,
        "test_ids": list(test_ids),
        "benchmark": benchmark,
    }


def train_mlp_heteroscedastic_export_bundle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    auto_install: bool,
    log1p_partial_index: bool,
    test_ids: list[str],
    benchmark: str = "unixbench",
) -> dict[str, Any]:
    """MLP with Gaussian NLL (mean + log-variance heads) for per-target uncertainty."""
    _maybe_auto_install(auto_install, ["torch"])
    import math

    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    x_t = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    n_in = x_train.shape[1]
    n_out = y_train.shape[1]
    log_2pi = math.log(2.0 * math.pi)

    class Het(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.head_mean = nn.Linear(hidden, n_out)
            self.head_logvar = nn.Linear(hidden, n_out)

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.body(z)
            return self.head_mean(h), self.head_logvar(h)

    model = Het().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mean, logvar = model(x_t)
        lv = logvar.clamp(-10.0, 10.0)
        inv = torch.exp(-lv)
        nll = 0.5 * torch.mean(inv * (y_t - mean) ** 2 + lv + log_2pi)
        nll.backward()
        opt.step()

    return {
        "schema": SCHEMA_V2,
        "model_type": "mlp",
        "heteroscedastic": True,
        "state_dict": model.state_dict(),
        "mlp_hidden": hidden,
        "in_dim": n_in,
        "out_dim": n_out,
        "log1p_partial_index": log1p_partial_index,
        "test_ids": list(test_ids),
        "benchmark": benchmark,
    }


def save_reconstruction_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle.get("model_type") == "mlp":
        import torch

        torch.save(bundle, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(bundle, f)


def sklearn_export_bundle(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    auto_install: bool,
    lgbm_estimators: int,
    xgb_estimators: int,
    log1p_partial_index: bool,
    test_ids: list[str],
    with_uncertainty: bool = True,
    benchmark: str = "unixbench",
) -> dict[str, Any]:
    mor = fit_sklearn_multioutput(
        model_name,
        x_train,
        y_train,
        auto_install=auto_install,
        lgbm_estimators=lgbm_estimators,
        xgb_estimators=xgb_estimators,
        n_estimators_override=None,
    )
    schema = SCHEMA_V2 if with_uncertainty else SCHEMA_V1
    bundle: dict[str, Any] = {
        "schema": schema,
        "model_type": model_name,
        "estimator": mor,
        "log1p_partial_index": log1p_partial_index,
        "test_ids": list(test_ids),
        "out_dim": int(y_train.shape[1]),
        "benchmark": benchmark,
    }
    if with_uncertainty:
        bundle["uncertainty_estimator"] = fit_sklearn_uncertainty_estimator(
            model_name,
            x_train,
            y_train,
            mor,
            auto_install=auto_install,
            lgbm_estimators=lgbm_estimators,
            xgb_estimators=xgb_estimators,
        )
    return bundle


def fit_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    auto_install: bool,
) -> np.ndarray:
    _maybe_auto_install(auto_install, ["torch"])
    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    x_t = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    xv = torch.from_numpy(x_val.astype(np.float32)).to(device)

    n_in = x_train.shape[1]
    n_out = y_train.shape[1]

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_out),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.net(z)

    model = M().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(xv).cpu().numpy()
    return out.astype(np.float64)


def aggregate_err(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """y shape (n, 13): first 12 tests, last suite."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae_tests = float(np.mean(np.abs(err[:, :-1])))
    rmse_tests = float(np.sqrt(np.mean(err[:, :-1] ** 2)))
    suite_t = y_true[:, -1]
    suite_p = y_pred[:, -1]
    mae_s = float(np.mean(np.abs(suite_t - suite_p)))
    rmse_s = float(np.sqrt(np.mean((suite_t - suite_p) ** 2)))
    return {
        "mae_all_targets": mae,
        "rmse_all_targets": rmse,
        "mae_subtest_index_only": mae_tests,
        "rmse_subtest_index_only": rmse_tests,
        "mae_suite_index": mae_s,
        "rmse_suite_index": rmse_s,
        "spearman_suite": spearman_rho(suite_t, suite_p),
        "kendall_tau_suite": kendall_tau_simple(suite_t, suite_p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default="dataset", help="Folder containing session subdirs")
    ap.add_argument(
        "--benchmark",
        type=str,
        choices=("unixbench", "phoronix"),
        default="unixbench",
        help="Dataset schema: unixbench (default) or phoronix (PTS cpu-style runs)",
    )
    ap.add_argument(
        "--glob-pattern",
        type=str,
        default="",
        help="Glob under dataset-root (default: auto from benchmark + --pts-suite; see moebench.dataset_globs)",
    )
    ap.add_argument("--model-type", type=str, choices=("xgboost", "lightgbm", "mlp"), default="lightgbm")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-partial-k", type=int, default=3, help="How many subtests are 'executed' at eval")
    ap.add_argument(
        "--pts-suite-target",
        type=str,
        choices=("arithmetic_mean", "logmean"),
        default="logmean",
        help="With --benchmark phoronix: suite target aggregation mode (default: logmean)",
    )
    ap.add_argument("--train-aug", type=int, default=10, help="Random partial subsets per training sample")
    ap.add_argument("--train-k-min", type=int, default=2)
    ap.add_argument("--train-k-max", type=int, default=6)
    ap.add_argument("--log1p-partial-index", action="store_true", help="Log1p partial index features")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-epochs", type=int, default=400)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--lgbm-estimators", type=int, default=200)
    ap.add_argument("--xgb-estimators", type=int, default=200)
    ap.add_argument("--report-json", type=str, default="", help="Write full report JSON to this path")
    ap.add_argument(
        "--include-per-sample",
        action="store_true",
        help="Include suite true/pred/err for each sample in the report JSON",
    )
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument(
        "--export-model",
        type=str,
        default="",
        help="After training, save reconstruction model for inference (use .pkl for trees, .pt for mlp)",
    )
    ap.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Export v1 (no σ head): homoscedastic MLP or trees without residual-calibrator",
    )
    ap.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation; only fit on all samples and write --export-model",
    )
    ap.add_argument(
        "--pts-suite",
        type=str,
        default=None,
        metavar="ID",
        help="With --benchmark phoronix: only use runs where yi.suite matches (e.g. pts/nvidia-gpu-compute)",
    )
    args = ap.parse_args()
    with_uncertainty = not args.no_uncertainty

    if args.benchmark == "phoronix" and not args.pts_suite:
        print(
            "phoronix reconstruction requires --pts-suite (e.g. cpu or pts/nvidia-gpu-compute).",
            file=sys.stderr,
        )
        return 2

    glob_eff = resolve_glob_pattern(
        benchmark=args.benchmark,
        glob_pattern=args.glob_pattern or None,
        pts_suite=args.pts_suite,
    )

    if args.benchmark == "phoronix":
        if not args.skip_cv or not args.export_model:
            print(
                "PTS mode: use --skip-cv --export-model <path> (few sessions; CV optional later).",
                file=sys.stderr,
            )
            return 2
        paths_pts = collect_phoronix_run_paths(
            Path(args.dataset_root),
            glob_pattern=glob_eff,
            pts_suite=args.pts_suite,
        )
        test_ids = list(canonical_test_ids_from_runs(paths_pts))
        n_test = len(test_ids)
        train_k_max = int(args.train_k_max)
        eval_partial_k = int(args.eval_partial_k)
        if train_k_max > n_test:
            print(
                f"[PTS] train-k-max={train_k_max} exceeds num_profiles={n_test}; clamping to {n_test}.",
                file=sys.stderr,
            )
            train_k_max = n_test
        if eval_partial_k > n_test:
            print(
                f"[PTS] eval-partial-k={eval_partial_k} exceeds num_profiles={n_test}; clamping to {n_test}.",
                file=sys.stderr,
            )
            eval_partial_k = n_test
        if args.train_k_min > train_k_max:
            print(
                f"[PTS] train-k-min={args.train_k_min} > effective train-k-max={train_k_max}; "
                f"set --train-k-min <= {train_k_max}.",
                file=sys.stderr,
            )
            return 2
        vec = XiVectorizer()
        rows_meta: list[dict[str, Any]] = []
        for p in paths_pts:
            with open(p, encoding="utf-8") as f:
                ds = json.load(f)
            t_full = full_suite_wall_seconds_pts(ds, test_ids=tuple(test_ids))
            tgt = extract_targets_from_pts_dataset(ds, tuple(test_ids))
            if t_full is None or tgt is None:
                continue
            indices, _suite_mean_raw = tgt
            suite_mean = pts_suite_target_from_indices(indices, args.pts_suite_target)
            rows_meta.append(
                {
                    "path": str(p),
                    "ds": ds,
                    "y": np.asarray(indices + [suite_mean], dtype=np.float64),
                    "t_full": float(t_full),
                }
            )
        if not rows_meta:
            print("No valid PTS samples after parsing", file=sys.stderr)
            return 2
        rng = np.random.RandomState(args.seed)
        xt, yt = build_augmented_train_matrix_pts(
            rows_meta,
            vec,
            test_ids,
            rng,
            args.train_aug,
            args.train_k_min,
            train_k_max,
            args.log1p_partial_index,
        )
        outp = Path(args.export_model)
        bm = "phoronix"
        if args.model_type == "mlp":
            if with_uncertainty:
                bundle = train_mlp_heteroscedastic_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                    benchmark=bm,
                )
            else:
                bundle = train_mlp_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                    benchmark=bm,
                )
        else:
            bundle = sklearn_export_bundle(
                args.model_type,
                xt,
                yt,
                auto_install=args.auto_install,
                lgbm_estimators=args.lgbm_estimators,
                xgb_estimators=args.xgb_estimators,
                log1p_partial_index=args.log1p_partial_index,
                test_ids=test_ids,
                with_uncertainty=with_uncertainty,
                benchmark=bm,
            )
        if args.pts_suite:
            bundle["pts_suite"] = args.pts_suite
        bundle["pts_suite_target"] = args.pts_suite_target
        save_reconstruction_bundle(outp, bundle)
        print(
            json.dumps(
                {
                    "export_only": True,
                    "benchmark": "phoronix",
                    "pts_suite": args.pts_suite,
                    "pts_suite_target": args.pts_suite_target,
                    "export_model": str(outp.resolve()),
                    "train_rows": int(len(xt)),
                    "model_type": args.model_type,
                    "num_profiles": len(test_ids),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    test_ids = list(INDEX_SUITE_TEST_IDS)
    n_test = len(test_ids)
    if args.train_k_max > n_test or (not args.skip_cv and args.eval_partial_k > n_test):
        print("train-k-max / eval-partial-k must be <= number of subtests", file=sys.stderr)
        return 2
    if args.train_k_min > args.train_k_max:
        print("train-k-min must be <= train-k-max", file=sys.stderr)
        return 2
    if args.skip_cv and not args.export_model:
        print("--skip-cv requires --export-model", file=sys.stderr)
        return 2

    paths = collect_unixbench_run_paths(Path(args.dataset_root), glob_pattern=glob_eff)
    records: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            records.append(json.load(f))
    if not args.skip_cv and len(records) < args.folds:
        print(f"Need at least {args.folds} samples, got {len(records)}", file=sys.stderr)
        return 2

    vec = XiVectorizer()
    rows_meta: list[dict[str, Any]] = []
    for p, ds in zip(paths, records):
        t_full = full_suite_wall_seconds(ds)
        tgt = extract_targets_from_dataset(ds)
        if t_full is None or tgt is None:
            continue
        indices, suite = tgt
        rows_meta.append(
            {
                "path": str(p),
                "ds": ds,
                "y": np.asarray(indices + [suite], dtype=np.float64),
                "t_full": float(t_full),
            }
        )

    if args.skip_cv:
        if not rows_meta:
            print("No valid samples after parsing", file=sys.stderr)
            return 2
        rng = np.random.RandomState(args.seed)
        xt, yt = build_augmented_train_matrix(
            rows_meta,
            vec,
            test_ids,
            rng,
            args.train_aug,
            args.train_k_min,
            args.train_k_max,
            args.log1p_partial_index,
        )
        outp = Path(args.export_model)
        if args.model_type == "mlp":
            if with_uncertainty:
                bundle = train_mlp_heteroscedastic_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                )
            else:
                bundle = train_mlp_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                )
        else:
            bundle = sklearn_export_bundle(
                args.model_type,
                xt,
                yt,
                auto_install=args.auto_install,
                lgbm_estimators=args.lgbm_estimators,
                xgb_estimators=args.xgb_estimators,
                log1p_partial_index=args.log1p_partial_index,
                test_ids=test_ids,
                with_uncertainty=with_uncertainty,
            )
        save_reconstruction_bundle(outp, bundle)
        summary = {
            "export_only": True,
            "export_model": str(outp.resolve()),
            "train_rows": int(len(xt)),
            "model_type": args.model_type,
            "uncertainty": with_uncertainty,
            "schema": bundle.get("schema"),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if len(rows_meta) < args.folds:
        print(f"Not enough valid samples after parsing: {len(rows_meta)}", file=sys.stderr)
        return 2

    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(rows_meta))
    rng.shuffle(idxs)
    folds = np.array_split(idxs, args.folds)

    oof_pred = np.zeros((len(rows_meta), n_test + 1), dtype=np.float64)
    oof_true = np.stack([rows_meta[i]["y"] for i in range(len(rows_meta))], axis=0)
    oof_mask = np.zeros(len(rows_meta), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    all_time_saved_ratio: list[float] = []

    for fi, val_idx in enumerate(folds):
        val_idx = [int(i) for i in val_idx]
        train_idx = [i for i in range(len(rows_meta)) if i not in set(val_idx)]

        x_rows: list[list[float]] = []
        y_rows: list[list[float]] = []

        for i in train_idx:
            meta = rows_meta[i]
            ds = meta["ds"]
            for _ in range(args.train_aug):
                k = rng.randint(args.train_k_min, args.train_k_max + 1)
                ex = set(rng.choice(test_ids, size=k, replace=False).tolist())
                row = build_partial_feature_row(
                    ds,
                    ex,
                    xi_vectorizer=vec,
                    log1p_index=args.log1p_partial_index,
                )
                if row is None:
                    continue
                x_rows.append(row)
                y_rows.append(meta["y"].tolist())

        if not x_rows:
            print("No training rows built", file=sys.stderr)
            return 2

        x_train = np.asarray(x_rows, dtype=np.float64)
        y_train = np.asarray(y_rows, dtype=np.float64)

        val_blocks: list[tuple[int, list[float], set[str]]] = []
        for t_i in val_idx:
            meta = rows_meta[t_i]
            ds = meta["ds"]
            ex_seed = stable_seed((rows_meta[t_i]["path"], fi, args.seed, "eval"))
            ex_rng = np.random.RandomState(ex_seed)
            ex = set(ex_rng.choice(test_ids, size=args.eval_partial_k, replace=False).tolist())
            row = build_partial_feature_row(
                ds,
                ex,
                xi_vectorizer=vec,
                log1p_index=args.log1p_partial_index,
            )
            if row is None:
                print(f"skip val row {meta['path']}", file=sys.stderr)
                continue
            val_blocks.append((t_i, row, ex))

        if not val_blocks:
            print("No validation rows", file=sys.stderr)
            return 2

        x_val = np.asarray([b[1] for b in val_blocks], dtype=np.float64)

        if args.model_type == "mlp":
            pred_val = fit_predict_mlp(
                x_train,
                y_train,
                x_val,
                hidden=args.mlp_hidden,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                auto_install=args.auto_install,
            )
        else:
            pred_val = fit_predict_sklearn_multioutput(
                args.model_type,
                x_train,
                y_train,
                x_val,
                auto_install=args.auto_install,
                lgbm_estimators=args.lgbm_estimators,
                xgb_estimators=args.xgb_estimators,
            )

        for j, (t_i, _, ex) in enumerate(val_blocks):
            oof_pred[t_i] = pred_val[j]
            oof_mask[t_i] = True
            ds = rows_meta[t_i]["ds"]
            t_part = partial_wall_seconds(ds, ex)
            t_full = rows_meta[t_i]["t_full"]
            if t_part is not None and t_full > 0:
                all_time_saved_ratio.append((t_full - t_part) / t_full)

        y_val_true = np.stack([rows_meta[t_i]["y"] for t_i, _, _ in val_blocks], axis=0)
        fr = aggregate_err(y_val_true, pred_val)
        fr["fold"] = fi
        fr["n_val"] = int(pred_val.shape[0])
        fold_reports.append(fr)

    if not oof_mask.all():
        print(f"warning: OOF missing for {int(np.sum(~oof_mask))} samples", file=sys.stderr)
    overall = aggregate_err(oof_true[oof_mask], oof_pred[oof_mask])

    subtest_spear: list[float] = []
    subtest_kend: list[float] = []
    for c in range(n_test):
        subtest_spear.append(spearman_rho(oof_true[oof_mask, c], oof_pred[oof_mask, c]))
        subtest_kend.append(kendall_tau_simple(oof_true[oof_mask, c], oof_pred[oof_mask, c]))

    report: dict[str, Any] = {
        "model_type": args.model_type,
        "n_samples": len(rows_meta),
        "cv_folds": args.folds,
        "eval_partial_k": args.eval_partial_k,
        "train_aug_per_sample": args.train_aug,
        "train_k_range": [args.train_k_min, args.train_k_max],
        "log1p_partial_index": args.log1p_partial_index,
        "oof_metrics": overall
        | {
            "mean_spearman_subtest_index_across_samples": float(np.nanmean(subtest_spear)),
            "mean_kendall_tau_subtest_index_across_samples": float(np.nanmean(subtest_kend)),
            "per_subtest_spearman_with_true_index": {test_ids[i]: subtest_spear[i] for i in range(n_test)},
            "per_subtest_kendall_tau_with_true_index": {test_ids[i]: subtest_kend[i] for i in range(n_test)},
        },
        "per_fold_metrics": fold_reports,
        "time_savings": {
            "mean_fraction_wall_time_saved_vs_full_suite": float(np.mean(all_time_saved_ratio))
            if all_time_saved_ratio
            else None,
            "description": "Mean (T_full - T_partial) / T_full over validation rows; "
            "T from ti.by_test_id for parallel key 1 (single-copy).",
        },
        "targets_layout": {
            "y_dim": n_test + 1,
            "first_dims_subtest_index_order": test_ids,
            "last_dim": "system_benchmarks_index_score",
        },
    }

    if args.include_per_sample:
        ps: list[dict[str, Any]] = []
        for i in range(len(rows_meta)):
            if not oof_mask[i]:
                continue
            yt = float(oof_true[i, -1])
            yp = float(oof_pred[i, -1])
            ps.append(
                {
                    "sample_path": rows_meta[i]["path"],
                    "suite_index_true": yt,
                    "suite_index_predicted": yp,
                    "suite_abs_error": abs(yt - yp),
                    "suite_relative_error": abs(yt - yp) / max(abs(yt), 1e-9),
                }
            )
        report["per_sample_suite"] = ps

    if args.export_model:
        rng_exp = np.random.RandomState(args.seed + 99)
        xt, yt = build_augmented_train_matrix(
            rows_meta,
            vec,
            test_ids,
            rng_exp,
            args.train_aug,
            args.train_k_min,
            args.train_k_max,
            args.log1p_partial_index,
        )
        outp = Path(args.export_model)
        if args.model_type == "mlp" or outp.suffix in (".pt", ".pth"):
            if with_uncertainty:
                bundle = train_mlp_heteroscedastic_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                )
            else:
                bundle = train_mlp_export_bundle(
                    xt,
                    yt,
                    hidden=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    auto_install=args.auto_install,
                    log1p_partial_index=args.log1p_partial_index,
                    test_ids=test_ids,
                )
            if outp.suffix not in (".pt", ".pth"):
                outp = outp.with_suffix(".pt")
        else:
            bundle = sklearn_export_bundle(
                args.model_type,
                xt,
                yt,
                auto_install=args.auto_install,
                lgbm_estimators=args.lgbm_estimators,
                xgb_estimators=args.xgb_estimators,
                log1p_partial_index=args.log1p_partial_index,
                test_ids=test_ids,
                with_uncertainty=with_uncertainty,
            )
        save_reconstruction_bundle(outp, bundle)
        report["exported_model"] = str(outp.resolve())
        report["export_train_rows"] = int(len(xt))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json:
        outp = Path(args.report_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
