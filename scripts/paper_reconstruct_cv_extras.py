#!/usr/bin/env python3
"""Paper-oriented reconstruction CV: baselines, xi ablations, LOSO vs random folds.

Supports **UnixBench**, **PTS CPU** (`yi.suite == cpu`), and **PTS GPU**
(`yi.suite == pts/nvidia-gpu-compute`) offline cross-validation on existing dataset JSONs.

Training uses random partial subsets (same idea as ``reconstruct_train_eval.py``); evaluation
uses explicit subset policies. Use ``--suites`` to run one or more benchmarks; default runs all three.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_reconstruct_train_eval_module():
    path = REPO_ROOT / "scripts" / "reconstruct_train_eval.py"
    spec = importlib.util.spec_from_file_location("moebench_reconstruct_train_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rte = _load_reconstruct_train_eval_module()

from moebench.paper_eval.subset_policies import select_eval_subset
from moebench.paper_eval.xi_ablation import AblatedXiVectorizer
from moebench.phoronix.training_data import (
    build_augmented_train_matrix_pts,
    build_partial_feature_row_pts,
    canonical_test_ids_from_runs,
    collect_phoronix_run_paths,
    extract_targets_from_pts_dataset,
    full_suite_wall_seconds_pts,
    partial_wall_seconds_pts,
    time_seconds_for_profile,
)
from moebench.reconstruct.data import (
    build_partial_feature_row,
    collect_unixbench_run_paths,
    extract_targets_from_dataset,
    full_suite_wall_seconds,
    partial_wall_seconds,
)
from moebench.router.feature_vectorizer import XiVectorizer
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS


def _load_router_meta(model_fp: Path, auto_install: bool) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        try:
            with open(model_fp, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "unknown"
            if missing == "lightgbm" and auto_install:
                import subprocess

                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "lightgbm"])
                with open(model_fp, "rb") as f:
                    return pickle.load(f)
            raise
    import torch

    try:
        return torch.load(model_fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_fp, map_location="cpu")


def _resolve_model_path(p: str) -> Path:
    raw = Path(p).expanduser()
    if raw.is_file():
        return raw.resolve()
    cand = (REPO_ROOT / p).resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Router model not found: {p}")


def _median_bucket_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    med = float(np.median(y_true))
    t_side = y_true >= med
    p_side = y_pred >= med
    return float(np.mean(t_side == p_side))


def _build_folds(
    rows_meta: list[dict[str, Any]],
    *,
    cv_mode: str,
    folds: int,
    seed: int,
) -> list[list[int]]:
    if cv_mode == "random_fold":
        rng = np.random.RandomState(seed)
        idxs = np.arange(len(rows_meta))
        rng.shuffle(idxs)
        parts = np.array_split(idxs, folds)
        return [[int(i) for i in part] for part in parts]
    if cv_mode == "leave_one_session_out":
        sess_map: dict[str, list[int]] = {}
        for i, m in enumerate(rows_meta):
            parent = Path(m["path"]).parent.name
            sess_map.setdefault(parent, []).append(i)
        sessions = sorted(sess_map.keys())
        return [sess_map[s] for s in sessions]
    raise ValueError("cv_mode must be random_fold or leave_one_session_out")


def run_one_combo_unixbench(
    *,
    rows_meta: list[dict[str, Any]],
    test_ids: list[str],
    xi_vec: Any,
    policy: str,
    router_meta: dict[str, Any] | None,
    eval_partial_k: int,
    model_type: str,
    train_aug: int,
    train_k_min: int,
    train_k_max: int,
    log1p_partial_index: bool,
    mlp_hidden: int,
    mlp_epochs: int,
    mlp_lr: float,
    lgbm_estimators: int,
    xgb_estimators: int,
    auto_install: bool,
    cv_mode: str,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    fold_indices = _build_folds(rows_meta, cv_mode=cv_mode, folds=folds, seed=seed)

    oof_pred = np.zeros((len(rows_meta), len(test_ids) + 1), dtype=np.float64)
    oof_true = np.stack([rows_meta[i]["y"] for i in range(len(rows_meta))], axis=0)
    oof_mask = np.zeros(len(rows_meta), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    all_time_saved_ratio: list[float] = []

    for fi, val_idx in enumerate(fold_indices):
        val_idx = [int(i) for i in val_idx]
        train_idx = [i for i in range(len(rows_meta)) if i not in set(val_idx)]

        rng = np.random.RandomState(seed + fi * 997)
        x_train, y_train = rte.build_augmented_train_matrix(
            [rows_meta[i] for i in train_idx],
            xi_vec,
            test_ids,
            rng,
            train_aug,
            train_k_min,
            train_k_max,
            log1p_partial_index,
        )

        val_blocks: list[tuple[int, list[float], set[str]]] = []
        for t_i in val_idx:
            meta = rows_meta[t_i]
            ds = meta["ds"]
            seed_parts = (meta["path"], fi, seed, policy, eval_partial_k)
            ex = select_eval_subset(
                policy,
                test_ids=test_ids,
                k=eval_partial_k,
                ds=ds,
                rng=rng,
                seed_parts=seed_parts,
                router_meta=router_meta,
            )
            row = build_partial_feature_row(
                ds,
                ex,
                xi_vectorizer=xi_vec,
                log1p_index=log1p_partial_index,
            )
            if row is None:
                continue
            val_blocks.append((t_i, row, ex))

        if not val_blocks:
            raise RuntimeError(f"Fold {fi}: no validation rows")

        x_val = np.asarray([b[1] for b in val_blocks], dtype=np.float64)

        if model_type == "mlp":
            pred_val = rte.fit_predict_mlp(
                x_train,
                y_train,
                x_val,
                hidden=mlp_hidden,
                epochs=mlp_epochs,
                lr=mlp_lr,
                auto_install=auto_install,
            )
        else:
            pred_val = rte.fit_predict_sklearn_multioutput(
                model_type,
                x_train,
                y_train,
                x_val,
                auto_install=auto_install,
                lgbm_estimators=lgbm_estimators,
                xgb_estimators=xgb_estimators,
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
        fr = rte.aggregate_err(y_val_true, pred_val)
        suite_t = y_val_true[:, -1]
        suite_p = pred_val[:, -1]
        fr["fold"] = fi
        fr["n_val"] = int(pred_val.shape[0])
        fr["median_bucket_accuracy_suite"] = _median_bucket_accuracy(suite_t, suite_p)
        ratios_fold: list[float] = []
        for t_i, _, ex in val_blocks:
            tf = rows_meta[t_i]["t_full"]
            tp = partial_wall_seconds(rows_meta[t_i]["ds"], ex)
            if tp is not None and tf > 0:
                ratios_fold.append((tf - tp) / tf)
        fr["mean_eval_wall_saved_fraction"] = float(np.mean(ratios_fold)) if ratios_fold else float("nan")
        fold_reports.append(fr)

    overall = rte.aggregate_err(oof_true[oof_mask], oof_pred[oof_mask])
    suite_t_all = oof_true[oof_mask, -1]
    suite_p_all = oof_pred[oof_mask, -1]
    overall["median_bucket_accuracy_suite"] = _median_bucket_accuracy(suite_t_all, suite_p_all)

    return {
        "benchmark": "unixbench",
        "eval_partial_k": int(eval_partial_k),
        "policy": policy,
        "oof_metrics": overall,
        "per_fold_metrics": fold_reports,
        "time_savings": {
            "mean_fraction_wall_time_saved_vs_full_suite": float(np.mean(all_time_saved_ratio))
            if all_time_saved_ratio
            else None,
            "description": "Eval subset; UnixBench ti.by_test_id parallel key 32.",
        },
        "n_oof": int(np.sum(oof_mask)),
    }


def run_one_combo_phoronix(
    *,
    rows_meta: list[dict[str, Any]],
    test_ids: list[str],
    xi_vec: Any,
    policy: str,
    router_meta: dict[str, Any] | None,
    eval_partial_k: int,
    model_type: str,
    train_aug: int,
    train_k_min: int,
    train_k_max: int,
    log1p_partial_value: bool,
    mlp_hidden: int,
    mlp_epochs: int,
    mlp_lr: float,
    lgbm_estimators: int,
    xgb_estimators: int,
    auto_install: bool,
    cv_mode: str,
    folds: int,
    seed: int,
    pts_wall_fn: Callable[[dict[str, Any], str], float | None],
) -> dict[str, Any]:
    fold_indices = _build_folds(rows_meta, cv_mode=cv_mode, folds=folds, seed=seed)
    n_out = len(test_ids) + 1
    oof_pred = np.zeros((len(rows_meta), n_out), dtype=np.float64)
    oof_true = np.stack([rows_meta[i]["y"] for i in range(len(rows_meta))], axis=0)
    oof_mask = np.zeros(len(rows_meta), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    all_time_saved_ratio: list[float] = []
    tid_tup = tuple(test_ids)

    for fi, val_idx in enumerate(fold_indices):
        val_idx = [int(i) for i in val_idx]
        train_idx = [i for i in range(len(rows_meta)) if i not in set(val_idx)]

        rng = np.random.RandomState(seed + fi * 997)
        x_train, y_train = build_augmented_train_matrix_pts(
            [rows_meta[i] for i in train_idx],
            xi_vec,
            test_ids,
            rng,
            train_aug,
            train_k_min,
            train_k_max,
            log1p_partial_value,
        )

        val_blocks: list[tuple[int, list[float], set[str]]] = []
        for t_i in val_idx:
            meta = rows_meta[t_i]
            ds = meta["ds"]
            seed_parts = (meta["path"], fi, seed, policy, eval_partial_k)
            ex = select_eval_subset(
                policy,
                test_ids=test_ids,
                k=eval_partial_k,
                ds=ds,
                rng=rng,
                seed_parts=seed_parts,
                router_meta=router_meta,
                profile_wall_seconds=pts_wall_fn,
            )
            row = build_partial_feature_row_pts(
                ds,
                ex,
                test_ids=tid_tup,
                xi_vectorizer=xi_vec,
                log1p_value=log1p_partial_value,
            )
            if row is None:
                continue
            val_blocks.append((t_i, row, ex))

        if not val_blocks:
            raise RuntimeError(f"Fold {fi}: no validation rows")

        x_val = np.asarray([b[1] for b in val_blocks], dtype=np.float64)

        if model_type == "mlp":
            pred_val = rte.fit_predict_mlp(
                x_train,
                y_train,
                x_val,
                hidden=mlp_hidden,
                epochs=mlp_epochs,
                lr=mlp_lr,
                auto_install=auto_install,
            )
        else:
            pred_val = rte.fit_predict_sklearn_multioutput(
                model_type,
                x_train,
                y_train,
                x_val,
                auto_install=auto_install,
                lgbm_estimators=lgbm_estimators,
                xgb_estimators=xgb_estimators,
            )

        for j, (t_i, _, ex) in enumerate(val_blocks):
            oof_pred[t_i] = pred_val[j]
            oof_mask[t_i] = True
            ds = rows_meta[t_i]["ds"]
            t_part = partial_wall_seconds_pts(ds, list(ex), test_ids=tid_tup)
            t_full = rows_meta[t_i]["t_full"]
            if t_part is not None and t_full > 0:
                all_time_saved_ratio.append((t_full - t_part) / t_full)

        y_val_true = np.stack([rows_meta[t_i]["y"] for t_i, _, _ in val_blocks], axis=0)
        fr = rte.aggregate_err(y_val_true, pred_val)
        suite_t = y_val_true[:, -1]
        suite_p = pred_val[:, -1]
        fr["fold"] = fi
        fr["n_val"] = int(pred_val.shape[0])
        fr["median_bucket_accuracy_suite"] = _median_bucket_accuracy(suite_t, suite_p)
        ratios_fold: list[float] = []
        for t_i, _, ex in val_blocks:
            tf = rows_meta[t_i]["t_full"]
            tp = partial_wall_seconds_pts(rows_meta[t_i]["ds"], list(ex), test_ids=tid_tup)
            if tp is not None and tf > 0:
                ratios_fold.append((tf - tp) / tf)
        fr["mean_eval_wall_saved_fraction"] = float(np.mean(ratios_fold)) if ratios_fold else float("nan")
        fold_reports.append(fr)

    overall = rte.aggregate_err(oof_true[oof_mask], oof_pred[oof_mask])
    suite_t_all = oof_true[oof_mask, -1]
    suite_p_all = oof_pred[oof_mask, -1]
    overall["median_bucket_accuracy_suite"] = _median_bucket_accuracy(suite_t_all, suite_p_all)

    return {
        "benchmark": "phoronix",
        "eval_partial_k": int(eval_partial_k),
        "policy": policy,
        "oof_metrics": overall,
        "per_fold_metrics": fold_reports,
        "time_savings": {
            "mean_fraction_wall_time_saved_vs_full_suite": float(np.mean(all_time_saved_ratio))
            if all_time_saved_ratio
            else None,
            "description": "Eval subset; PTS ti.by_test_id.time_s_total (or execution_cost).",
        },
        "n_oof": int(np.sum(oof_mask)),
    }


def _load_unixbench_rows(
    dataset_root: Path,
    glob_pattern: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = collect_unixbench_run_paths(dataset_root, glob_pattern=glob_pattern)
    rows_meta: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            ds = json.load(f)
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
    return rows_meta, list(INDEX_SUITE_TEST_IDS)


def _load_phoronix_rows(
    dataset_root: Path,
    glob_pattern: str,
    pts_suite: str,
    pts_suite_target: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    paths = collect_phoronix_run_paths(
        dataset_root,
        glob_pattern=glob_pattern,
        pts_suite=pts_suite,
    )
    test_ids = list(canonical_test_ids_from_runs(paths))
    rows_meta: list[dict[str, Any]] = []
    tup = tuple(test_ids)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            ds = json.load(f)
        t_full = full_suite_wall_seconds_pts(ds, test_ids=tup)
        tgt = extract_targets_from_pts_dataset(ds, tup)
        if t_full is None or tgt is None:
            continue
        indices, _arith = tgt
        suite_agg = rte.pts_suite_target_from_indices(indices, pts_suite_target)
        rows_meta.append(
            {
                "path": str(p),
                "ds": ds,
                "y": np.asarray(indices + [suite_agg], dtype=np.float64),
                "t_full": float(t_full),
            }
        )
    return rows_meta, test_ids


def _prepare_cv_mode(
    rows_meta: list[dict[str, Any]],
    cv_mode: str,
    folds: int,
    label: str,
) -> tuple[str, int]:
    sessions = sorted({Path(m["path"]).parent.name for m in rows_meta})
    eff_folds = folds
    eff_cv = cv_mode
    if cv_mode == "leave_one_session_out" and len(sessions) < 2:
        print(
            f"[{label}] leave_one_session_out needs >= 2 session dirs; "
            f"got {len(sessions)}. Falling back to random_fold.",
            file=sys.stderr,
        )
        eff_cv = "random_fold"
        if len(rows_meta) < eff_folds:
            eff_folds = max(2, min(eff_folds, len(rows_meta)))
    if eff_cv == "random_fold" and len(rows_meta) < eff_folds:
        print(f"[{label}] random_fold: reducing folds from {eff_folds} to {len(rows_meta)}", file=sys.stderr)
        eff_folds = max(2, len(rows_meta))
    return eff_cv, eff_folds


def run_suite_block(
    *,
    suite_key: str,
    pts_suite: str | None,
    glob_pattern: str,
    dataset_root: Path,
    k_list: list[int],
    xi_modes: list[str],
    pol_list: list[str],
    router_meta: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if suite_key == "unixbench":
        rows_meta, test_ids = _load_unixbench_rows(dataset_root, glob_pattern)
    elif suite_key in ("phoronix_cpu", "phoronix_gpu"):
        ps = pts_suite or ("cpu" if suite_key == "phoronix_cpu" else "pts/nvidia-gpu-compute")
        rows_meta, test_ids = _load_phoronix_rows(dataset_root, glob_pattern, ps, args.pts_suite_target)
    else:
        raise ValueError(suite_key)

    if len(rows_meta) < 2:
        raise RuntimeError(f"[{suite_key}] need >= 2 valid samples (glob={glob_pattern!r})")

    eff_cv, eff_folds = _prepare_cv_mode(rows_meta, args.cv_mode, args.folds, suite_key)
    sessions = sorted({Path(m["path"]).parent.name for m in rows_meta})

    eff_train_k_max = min(int(args.train_k_max), len(test_ids))
    eff_train_k_min = min(int(args.train_k_min), eff_train_k_max)

    combinations: list[dict[str, Any]] = []
    pts_wall_fn = time_seconds_for_profile

    for eval_k in k_list:
        if eval_k > len(test_ids) or eval_k < 1:
            raise ValueError(f"K={eval_k} invalid for {suite_key} (num_profiles={len(test_ids)})")

        for xi_mode in xi_modes:
            xi_vec: Any = XiVectorizer() if xi_mode == "full" else AblatedXiVectorizer(xi_mode)
            base_names = XiVectorizer().feature_names
            if xi_vec.feature_names != base_names:
                raise RuntimeError("AblatedXiVectorizer feature_names mismatch")

            for policy in pol_list:
                if suite_key == "unixbench":
                    block = run_one_combo_unixbench(
                        rows_meta=rows_meta,
                        test_ids=test_ids,
                        xi_vec=xi_vec,
                        policy=policy,
                        router_meta=router_meta,
                        eval_partial_k=eval_k,
                        model_type=args.model_type,
                        train_aug=args.train_aug,
                        train_k_min=eff_train_k_min,
                        train_k_max=eff_train_k_max,
                        log1p_partial_index=args.log1p_partial_index,
                        mlp_hidden=args.mlp_hidden,
                        mlp_epochs=args.mlp_epochs,
                        mlp_lr=args.mlp_lr,
                        lgbm_estimators=args.lgbm_estimators,
                        xgb_estimators=args.xgb_estimators,
                        auto_install=args.auto_install,
                        cv_mode=eff_cv,
                        folds=eff_folds,
                        seed=int(args.seed),
                    )
                else:
                    block = run_one_combo_phoronix(
                        rows_meta=rows_meta,
                        test_ids=test_ids,
                        xi_vec=xi_vec,
                        policy=policy,
                        router_meta=router_meta,
                        eval_partial_k=eval_k,
                        model_type=args.model_type,
                        train_aug=args.train_aug,
                        train_k_min=eff_train_k_min,
                        train_k_max=eff_train_k_max,
                        log1p_partial_value=args.log1p_partial_value,
                        mlp_hidden=args.mlp_hidden,
                        mlp_epochs=args.mlp_epochs,
                        mlp_lr=args.mlp_lr,
                        lgbm_estimators=args.lgbm_estimators,
                        xgb_estimators=args.xgb_estimators,
                        auto_install=args.auto_install,
                        cv_mode=eff_cv,
                        folds=eff_folds,
                        seed=int(args.seed),
                        pts_wall_fn=pts_wall_fn,
                    )
                combinations.append({"suite_key": suite_key, "xi_ablation": xi_mode, **block})

    return {
        "suite_key": suite_key,
        "pts_suite_filter": pts_suite,
        "glob_pattern": glob_pattern,
        "cv_mode_effective": eff_cv,
        "folds_effective": eff_folds,
        "sessions": sessions,
        "n_samples": len(rows_meta),
        "num_profiles_or_subtests": len(test_ids),
        "profile_ids_head": test_ids[: min(8, len(test_ids))],
        "combinations": combinations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument(
        "--suites",
        type=str,
        default="unixbench,phoronix_cpu,phoronix_gpu",
        help="Comma-separated: unixbench, phoronix_cpu, phoronix_gpu",
    )
    ap.add_argument("--glob-unixbench", type=str, default="*/run-*.json")
    ap.add_argument("--glob-pts-cpu", type=str, default="aces-*/run-*.json")
    ap.add_argument("--glob-pts-gpu", type=str, default="*pts_nvidia-gpu-compute*/run-*.json")
    ap.add_argument(
        "--cv-mode",
        type=str,
        choices=("random_fold", "leave_one_session_out"),
        default="leave_one_session_out",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-partial-k", type=int, default=3)
    ap.add_argument(
        "--k-sweep",
        type=str,
        default="",
        help="Comma-separated K values; overrides --eval-partial-k when set.",
    )
    ap.add_argument(
        "--policies",
        type=str,
        default="random,fixed_first_k,fixed_cpu_mix,greedy_slowest,greedy_fastest",
    )
    ap.add_argument(
        "--xi-ablations",
        type=str,
        default="full",
    )
    ap.add_argument(
        "--pts-suite-target",
        type=str,
        choices=("arithmetic_mean", "logmean"),
        default="logmean",
        help="PTS suite scalar target (last dim); matches reconstruct_train_eval export.",
    )
    ap.add_argument(
        "--router-model",
        type=str,
        default="",
        help="Alias for --router-model-unixbench (backward compatible).",
    )
    ap.add_argument("--router-model-unixbench", type=str, default="")
    ap.add_argument("--router-model-pts-cpu", type=str, default="")
    ap.add_argument("--router-model-pts-gpu", type=str, default="")
    ap.add_argument("--model-type", type=str, choices=("lightgbm", "xgboost", "mlp"), default="lightgbm")
    ap.add_argument("--train-aug", type=int, default=10)
    ap.add_argument("--train-k-min", type=int, default=2)
    ap.add_argument("--train-k-max", type=int, default=6)
    ap.add_argument("--log1p-partial-index", action="store_true", help="UnixBench only: log1p partial index")
    ap.add_argument("--log1p-partial-value", action="store_true", help="PTS only: log1p primary value")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-epochs", type=int, default=400)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--lgbm-estimators", type=int, default=200)
    ap.add_argument("--xgb-estimators", type=int, default=200)
    ap.add_argument("--report-json", type=str, default="")
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    if args.router_model and not args.router_model_unixbench:
        args.router_model_unixbench = args.router_model

    dataset_root = Path(args.dataset_root).resolve()

    k_list: list[int]
    if args.k_sweep.strip():
        k_list = [int(x.strip()) for x in args.k_sweep.split(",") if x.strip()]
        if not k_list:
            print("--k-sweep had no integers", file=sys.stderr)
            return 2
    else:
        k_list = [int(args.eval_partial_k)]

    suite_keys = [s.strip() for s in args.suites.split(",") if s.strip()]
    valid_sk = {"unixbench", "phoronix_cpu", "phoronix_gpu"}
    for sk in suite_keys:
        if sk not in valid_sk:
            print(f"Unknown suite {sk!r}; choose from {sorted(valid_sk)}", file=sys.stderr)
            return 2

    pol_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    xi_modes = [x.strip() for x in args.xi_ablations.split(",") if x.strip()]

    if "router" in pol_list:
        missing: list[str] = []
        if "unixbench" in suite_keys and not args.router_model_unixbench:
            missing.append("unixbench: --router-model-unixbench")
        if "phoronix_cpu" in suite_keys and not args.router_model_pts_cpu:
            missing.append("phoronix_cpu: --router-model-pts-cpu")
        if "phoronix_gpu" in suite_keys and not args.router_model_pts_gpu:
            missing.append("phoronix_gpu: --router-model-pts-gpu")
        if missing:
            print(
                "Policies include 'router' but router checkpoints are missing:\n  "
                + "\n  ".join(missing),
                file=sys.stderr,
            )
            return 2

    def _meta_for(sk: str) -> dict[str, Any] | None:
        if "router" not in pol_list:
            return None
        path_map = {
            "unixbench": args.router_model_unixbench,
            "phoronix_cpu": args.router_model_pts_cpu,
            "phoronix_gpu": args.router_model_pts_gpu,
        }
        p = path_map.get(sk, "")
        if not p:
            return None
        return _load_router_meta(_resolve_model_path(p), args.auto_install)

    suite_results: list[dict[str, Any]] = []
    errors: list[str] = []

    spec_map: list[tuple[str, str | None, str]] = []
    if "unixbench" in suite_keys:
        spec_map.append(("unixbench", None, args.glob_unixbench))
    if "phoronix_cpu" in suite_keys:
        spec_map.append(("phoronix_cpu", "cpu", args.glob_pts_cpu))
    if "phoronix_gpu" in suite_keys:
        spec_map.append(("phoronix_gpu", "pts/nvidia-gpu-compute", args.glob_pts_gpu))

    for suite_key, pts_suite, glob_pat in spec_map:
        try:
            block = run_suite_block(
                suite_key=suite_key,
                pts_suite=pts_suite,
                glob_pattern=glob_pat,
                dataset_root=dataset_root,
                k_list=k_list,
                xi_modes=xi_modes,
                pol_list=pol_list,
                router_meta=_meta_for(suite_key),
                args=args,
            )
            suite_results.append(block)
        except (FileNotFoundError, RuntimeError) as e:
            errors.append(f"{suite_key}: {e}")

    if not suite_results:
        print("No suite produced results:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    out: dict[str, Any] = {
        "schema": "moebench.paper_reconstruct_cv_extras.v2",
        "dataset_root": str(dataset_root),
        "suites_requested": suite_keys,
        "eval_partial_k_list": k_list,
        "train_k_range_requested": [args.train_k_min, args.train_k_max],
        "pts_suite_target": args.pts_suite_target,
        "model_type": args.model_type,
        "policies": pol_list,
        "xi_ablations": xi_modes,
        "suite_results": suite_results,
    }
    if errors:
        out["suite_errors"] = errors

    txt = json.dumps(out, indent=2, ensure_ascii=False)
    print(txt)
    if args.report_json:
        outp = Path(args.report_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
