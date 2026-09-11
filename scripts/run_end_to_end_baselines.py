#!/usr/bin/env python3
"""Evaluate simple end-to-end UnixBench alternatives with training-only statistics.

The implementation intentionally uses fixed, low-capacity estimators.  It is
an offline replay: selected partial scores are read from historical complete
runs, while selection and reconstruction are performed as if only those
scores had been observed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moebench.dataset_machines import glob_for_machine
from moebench.reconstruct.data import (
    _ti_for_test,
    collect_unixbench_run_paths,
    extract_targets_from_dataset,
)
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

METHODS = (
    "correlation_reconstruction",
    "matrix_completion",
    "adaptive_probing",
)
K_DEFAULT = 3
FOLD_SEED = 42
MATRIX_RANK = 2
MATRIX_RIDGE = 1e-2


def _load_rows(dataset_root: Path, machine: str) -> list[dict[str, Any]]:
    pattern = glob_for_machine(benchmark="unixbench", machine=machine)
    rows: list[dict[str, Any]] = []
    for path in collect_unixbench_run_paths(dataset_root, glob_pattern=pattern):
        ds = json.loads(path.read_text(encoding="utf-8"))
        target = extract_targets_from_dataset(ds, test_ids=INDEX_SUITE_TEST_IDS)
        if target is None:
            continue
        values, suite = target
        times = {}
        for tid in INDEX_SUITE_TEST_IDS:
            value = _ti_for_test(ds, tid, parallel_key="1")
            if value is None or value <= 0:
                raise RuntimeError(f"Missing positive timing for {tid} in {path}")
            times[tid] = float(value)
        rows.append(
            {
                "path": str(path),
                "session": path.parent.name,
                "ds": ds,
                "values": np.asarray(values, dtype=float),
                "suite": float(suite),
                "times": times,
                "full_time_s": float(sum(times.values())),
            }
        )
    if len(rows) < 2:
        raise RuntimeError(f"Need at least two usable UnixBench runs for {machine}")
    return rows


def _build_splits(
    n: int,
    *,
    split_mode: str = "kfold",
    split_seed: int = FOLD_SEED,
    folds: int = 5,
    test_size: float = 0.2,
) -> list[tuple[int, list[int], list[int]]]:
    """Build deterministic run-level splits.

    The ``kfold`` branch mirrors reconstruct_train_eval.py: shuffle the
    sample indices with the split seed and use np.array_split.  The
    ``train_test`` branch is a single held-out test split using the same
    shuffled-index convention.  No values are inspected here.
    """
    if n < 2:
        raise ValueError("need at least two runs for a train/test split")
    if split_mode not in {"kfold", "train_test"}:
        raise ValueError(f"unknown split mode: {split_mode}")
    rng = np.random.RandomState(split_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    if split_mode == "train_test":
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        n_test = max(1, int(math.ceil(n * test_size)))
        n_test = min(n - 1, n_test)
        parts = [indices[:n_test]]
    else:
        if folds < 2 or folds > n:
            raise ValueError(f"folds must be between 2 and n={n}")
        parts = np.array_split(indices, folds)
    out: list[tuple[int, list[int], list[int]]] = []
    for split_id, part in enumerate(parts):
        test = [int(x) for x in part]
        train_set = set(test)
        train = [i for i in range(n) if i not in train_set]
        if train:
            out.append((split_id, train, test))
    return out


def _folds(n: int) -> list[tuple[list[int], list[int]]]:
    """Backward-compatible alias for the main method's five-fold split."""
    return [(train, test) for _sid, train, test in _build_splits(n)]


def _stats(train: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.stack([r["values"] for r in train])
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=0)
    std = np.where(std > 1e-9, std, 1.0)
    z = (matrix - mean) / std
    if len(train) < 2:
        corr = np.eye(matrix.shape[1])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
    return mean, std, corr


def _conditional_prediction(
    observed: np.ndarray,
    observed_ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation reconstruction and conditional residual std."""
    n = len(mean)
    pred = mean.copy()
    residual = std.copy()
    if not observed_ids:
        return pred, residual
    obs = np.asarray(observed_ids, dtype=int)
    z_obs = (observed - mean[obs]) / std[obs]
    block = corr[np.ix_(obs, obs)] + MATRIX_RIDGE * np.eye(len(obs))
    inv = np.linalg.pinv(block)
    for j in range(n):
        if j in observed_ids:
            pred[j] = observed[observed_ids.index(j)]
            residual[j] = 0.0
            continue
        cross = corr[j, obs]
        weights = cross @ inv
        pred[j] = mean[j] + std[j] * float(weights @ z_obs)
        var_ratio = 1.0 - float(weights @ corr[obs, j])
        residual[j] = std[j] * math.sqrt(max(0.0, var_ratio))
    return pred, residual


def _correlation_reconstruction(
    values: np.ndarray,
    train: list[dict[str, Any]],
    k: int,
) -> tuple[list[int], np.ndarray]:
    mean, std, corr = _stats(train)
    selected = list(range(min(k, len(values))))
    pred, _ = _conditional_prediction(values[selected], selected, mean, std, corr)
    return selected, pred


def _matrix_completion(
    values: np.ndarray,
    train: list[dict[str, Any]],
    k: int,
) -> tuple[list[int], np.ndarray]:
    mean, std, _ = _stats(train)
    z_train = np.stack([(r["values"] - mean) / std for r in train])
    _u, _s, vt = np.linalg.svd(z_train, full_matrices=False)
    rank = min(MATRIX_RANK, vt.shape[0], vt.shape[1])
    basis = vt[:rank]
    selected = list(range(min(k, len(values))))
    obs = np.asarray(selected, dtype=int)
    z_obs = (values[obs] - mean[obs]) / std[obs]
    design = basis[:, obs].T
    lhs = design.T @ design + MATRIX_RIDGE * np.eye(rank)
    coef = np.linalg.solve(lhs, design.T @ z_obs)
    pred = mean + std * (coef @ basis)
    pred[obs] = values[obs]
    return selected, pred


def _adaptive_selection_trace(
    values: np.ndarray,
    train: list[dict[str, Any]],
    k: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    mean, std, corr = _stats(train)
    selected = [0]
    trace: list[dict[str, Any]] = [
        {
            "round": 1,
            "selected_index": 0,
            "selected_test_id": INDEX_SUITE_TEST_IDS[0],
            "observed_indices": [0],
            "observed_test_ids": [INDEX_SUITE_TEST_IDS[0]],
            "selection_basis": "fixed_first_probe",
        }
    ]
    while len(selected) < min(k, len(values)):
        observed = values[selected]
        pred, residual = _conditional_prediction(observed, selected, mean, std, corr)
        candidates = [i for i in range(len(values)) if i not in selected]
        scored = {
            i: abs(float((pred[i] - mean[i]) / std[i])) * float(residual[i])
            for i in candidates
        }
        choice = max(
            candidates,
            key=lambda i: (scored[i], -i),
        )
        selected.append(choice)
        trace.append(
            {
                "round": len(selected),
                "selected_index": choice,
                "selected_test_id": INDEX_SUITE_TEST_IDS[choice],
                "observed_indices": list(selected),
                "observed_test_ids": [INDEX_SUITE_TEST_IDS[i] for i in selected],
                "candidate_utility": {
                    INDEX_SUITE_TEST_IDS[i]: float(scored[i]) for i in candidates
                },
                "selection_basis": "observed_partial_scores_and_training_statistics",
            }
        )
    return selected, trace


def _adaptive_selection(
    values: np.ndarray,
    train: list[dict[str, Any]],
    k: int,
) -> list[int]:
    return _adaptive_selection_trace(values, train, k)[0]


def _predict(
    method: str,
    values: np.ndarray,
    train: list[dict[str, Any]],
    k: int,
) -> tuple[list[int], np.ndarray, list[dict[str, Any]]]:
    if method == "correlation_reconstruction":
        selected, pred = _correlation_reconstruction(values, train, k)
        return selected, pred, []
    if method == "matrix_completion":
        selected, pred = _matrix_completion(values, train, k)
        return selected, pred, []
    if method == "adaptive_probing":
        mean, std, corr = _stats(train)
        selected, trace = _adaptive_selection_trace(values, train, k)
        pred, _ = _conditional_prediction(values[selected], selected, mean, std, corr)
        return selected, pred, trace
    raise ValueError(method)


def _suite_from_values(values: np.ndarray) -> float:
    clipped = np.maximum(values, 1e-9)
    return float(math.exp(float(np.mean(np.log(clipped)))))


def _evaluate_method(
    method: str,
    rows: list[dict[str, Any]],
    k: int,
    *,
    split_mode: str = "kfold",
    split_seed: int = FOLD_SEED,
    folds: int = 5,
    test_size: float = 0.2,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    splits = _build_splits(
        len(rows),
        split_mode=split_mode,
        split_seed=split_seed,
        folds=folds,
        test_size=test_size,
    )
    for split_id, train_idx, test_idx in splits:
        train = [rows[i] for i in train_idx]
        train_paths = [r["path"] for r in train]
        for test_i in test_idx:
            row = rows[test_i]
            selected, pred_values, selection_trace = _predict(method, row["values"], train, k)
            selected_ids = [INDEX_SUITE_TEST_IDS[i] for i in selected]
            predicted_suite = _suite_from_values(pred_values)
            actual_suite = row["suite"]
            abs_error = abs(predicted_suite - actual_suite)
            rel_error = abs_error / max(abs(actual_suite), 1e-9)
            partial_time = sum(row["times"][tid] for tid in selected_ids)
            per_run.append(
                {
                    "run_id": Path(row["path"]).name,
                    "run_path": row["path"],
                    "split_id": split_id,
                    "split_mode": split_mode,
                    "training_run_indices": train_idx,
                    "test_run_indices": test_idx,
                    "training_run_paths": train_paths,
                    "selected_test_ids": selected_ids,
                    "selection_trace": selection_trace,
                    "partial_observed_values": {
                        tid: float(row["values"][i]) for i, tid in zip(selected, selected_ids)
                    },
                    "reconstructed_test_values": {
                        tid: float(pred_values[i]) for i, tid in enumerate(INDEX_SUITE_TEST_IDS)
                    },
                    "predicted_full_suite": predicted_suite,
                    "actual_full_suite": actual_suite,
                    "absolute_error": abs_error,
                    "relative_error": rel_error,
                    "partial_wall_time_s": partial_time,
                    "full_suite_wall_time_s": row["full_time_s"],
                    "estimated_time_saved_fraction": (row["full_time_s"] - partial_time) / row["full_time_s"],
                    "leakage_flags": [],
                }
            )
    errors = np.asarray([r["relative_error"] for r in per_run], dtype=float)
    abs_errors = np.asarray([r["absolute_error"] for r in per_run], dtype=float)
    return {
        "method": method,
        "kind": "independent_end_to_end_alternative",
        "k": k,
        "n_test_predictions": len(per_run),
        "mean_relative_error": float(np.mean(errors)),
        "mean_relative_error_pct": float(np.mean(errors) * 100.0),
        "median_relative_error_pct": float(np.median(errors) * 100.0),
        "p90_relative_error_pct": float(np.percentile(errors, 90) * 100.0),
        "p95_relative_error_pct": float(np.percentile(errors, 95) * 100.0),
        "max_relative_error_pct": float(np.max(errors) * 100.0),
        "mae_suite_index": float(np.mean(abs_errors)),
        "rmse_suite_index": float(np.sqrt(np.mean(abs_errors**2))),
        "mean_time_saved_fraction": float(
            np.mean([r["estimated_time_saved_fraction"] for r in per_run])
        ),
        "protocol": {
            "split_mode": split_mode,
            "split_seed": split_seed,
            "folds": folds if split_mode == "kfold" else None,
            "test_size": test_size if split_mode == "train_test" else None,
            "selection_fit": "training_split_only",
            "reconstruction_fit": "training_split_only",
            "partial_input": "historical_run_replay",
            "validation_unobserved_values_used": False,
            "test_timing_used_for_selection": False,
            "matrix_rank": MATRIX_RANK if method == "matrix_completion" else None,
            "matrix_ridge": MATRIX_RIDGE if method in ("matrix_completion", "adaptive_probing", "correlation_reconstruction") else None,
        },
        "per_run": per_run,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 Alternative End-to-End Baselines",
        "",
        f"Offline replay on M1 (`aces-System-Product-Name`), UnixBench, `K={report['k']}`.",
        f"Split: `{report['protocol']['split_mode']}`; "
        f"all statistics are fit on the training split only; no test-driven tuning.",
        "",
        "| Method | Mean error (%) | Median (%) | P95 (%) | Max (%) | MAE | RMSE | Saved (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["methods"]:
        lines.append(
            f"| {r['method']} | {r['mean_relative_error_pct']:.3f} | "
            f"{r['median_relative_error_pct']:.3f} | {r['p95_relative_error_pct']:.3f} | "
            f"{r['max_relative_error_pct']:.3f} | {r['mae_suite_index']:.3f} | "
            f"{r['rmse_suite_index']:.3f} | {r['mean_time_saved_fraction'] * 100:.3f} |"
        )
    lines.extend(
        [
            "",
            "Selection controls and BenchScout results are reported separately.",
            "The timing column is estimated from historical `ti` fields, not an online 5-second probe.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("unixbench",), default="unixbench")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="42,7,21", help="Accepted for matrix compatibility; deterministic methods use fixed fold seed.")
    ap.add_argument(
        "--split-mode",
        choices=("kfold", "train_test"),
        default="kfold",
        help="kfold mirrors the main method; train_test evaluates one held-out test split.",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=FOLD_SEED)
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out fraction for --split-mode train_test.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        ap.error(f"unknown method(s): {', '.join(unknown)}")
    if args.k < 1 or args.k > len(INDEX_SUITE_TEST_IDS):
        ap.error(f"--k must be in [1, {len(INDEX_SUITE_TEST_IDS)}]")
    rows = _load_rows(args.dataset_root.resolve(), args.machine)
    try:
        splits = _build_splits(
            len(rows),
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            folds=args.folds,
            test_size=args.test_size,
        )
    except ValueError as exc:
        ap.error(str(exc))
    reports = [
        _evaluate_method(
            method,
            rows,
            args.k,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            folds=args.folds,
            test_size=args.test_size,
        )
        for method in methods
    ]
    report = {
        "schema": "moebench.end_to_end_alternative_baselines.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "machine": args.machine,
        "dataset_root": str(args.dataset_root.resolve()),
        "n_runs": len(rows),
        "n_sessions": len({r["session"] for r in rows}),
        "k": args.k,
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "folds": args.folds if args.split_mode == "kfold" else None,
        "test_size": args.test_size if args.split_mode == "train_test" else None,
        "splits": [
            {
                "split_id": split_id,
                "training_run_indices": train_idx,
                "test_run_indices": test_idx,
                "training_run_paths": [rows[i]["path"] for i in train_idx],
                "test_run_paths": [rows[i]["path"] for i in test_idx],
            }
            for split_id, train_idx, test_idx in splits
        ],
        "requested_seeds": [int(x) for x in args.seeds.split(",") if x.strip()],
        "methods": reports,
        "protocol": {
            "evaluation": "offline_end_to_end_replay",
            "full_suite_ground_truth": "historical UnixBench system_benchmarks_index_score",
            "partial_budget": f"K={args.k} observed subtests",
            "split_mode": args.split_mode,
            "split_seed": args.split_seed,
            "training_statistics": "training_split_only",
            "validation_unobserved_values_used": False,
            "online_probe_duration_s": None,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "moebench.end_to_end_alternative_baselines_manifest.v1",
                "created_at_utc": report["created_at_utc"],
                "command": " ".join(__import__("sys").argv),
                "methods": methods,
                "k": args.k,
                "machine": args.machine,
                "split_mode": args.split_mode,
                "split_seed": args.split_seed,
                "folds": args.folds if args.split_mode == "kfold" else None,
                "test_size": args.test_size if args.split_mode == "train_test" else None,
                "splits": report["splits"],
                "run_paths": [r["path"] for r in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "per_run_results.json").write_text(
        json.dumps(
            {method["method"]: method["per_run"] for method in reports},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "baseline_report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
