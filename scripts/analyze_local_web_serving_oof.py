#!/usr/bin/env python3
"""Audit local wrk logs and compare run-held-out HTTP profile predictors."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np


CONNECTIONS = [20, 100, 200, 500, 1000]
K = 3
REQUESTS_RE = re.compile(r"^Requests/sec:\s*([0-9.]+)\s*$", re.MULTILINE)
ELAPSED_RE = re.compile(r"^elapsed_real_seconds=([0-9.]+)\s*$", re.MULTILINE)
P99_RE = re.compile(r"^\s*99%\s+([0-9.]+)(us|ms|s)\s*$", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_log(path: Path, run_id: int, connections: int) -> dict[str, float | str]:
    text = path.read_text(encoding="utf-8")
    requests = REQUESTS_RE.findall(text)
    elapsed = ELAPSED_RE.findall(text)
    p99 = P99_RE.findall(text)
    if len(requests) != 1 or len(elapsed) != 1 or len(p99) != 1:
        raise ValueError(f"Expected one throughput, p99 latency, and elapsed-time field in {path}")
    socket_errors = re.findall(r"Socket errors:.*", text)
    if socket_errors and any(int(value) for value in re.findall(r"\b\d+\b", socket_errors[0])):
        raise ValueError(f"Nonzero socket error in {path}: {socket_errors[0]}")
    if "Non-2xx or 3xx responses" in text:
        raise ValueError(f"Non-success HTTP responses in {path}")
    p99_value, p99_unit = p99[0]
    p99_us = float(p99_value) * {"us": 1.0, "ms": 1000.0, "s": 1_000_000.0}[p99_unit]
    return {
        "run_id": f"run-{run_id}",
        "connections": connections,
        "requests_per_second": float(requests[0]),
        "p99_latency_microseconds": p99_us,
        "elapsed_seconds": float(elapsed[0]),
        "raw_log": str(path),
        "raw_log_sha256": sha256(path),
    }


def correlation_stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0, ddof=0)
    std = np.where(std > 1e-9, std, 1.0)
    z = (train - mean) / std
    corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return mean, std, corr


def conditional_prediction(
    observed: np.ndarray,
    selected: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = mean.copy()
    residual = std.copy()
    if not selected:
        return pred, residual
    obs = np.asarray(selected, dtype=int)
    z_obs = (observed[obs] - mean[obs]) / std[obs]
    block = corr[np.ix_(obs, obs)] + 0.01 * np.eye(len(obs))
    inverse = np.linalg.pinv(block)
    for index in range(len(mean)):
        if index in selected:
            pred[index] = observed[index]
            residual[index] = 0.0
            continue
        weights = corr[index, obs] @ inverse
        pred[index] = mean[index] + std[index] * float(weights @ z_obs)
        variance_ratio = 1.0 - float(weights @ corr[obs, index])
        residual[index] = std[index] * np.sqrt(max(0.0, variance_ratio))
    return pred, residual


def adaptive_selection(values: np.ndarray, train: np.ndarray) -> tuple[list[int], list[dict[str, Any]]]:
    mean, std, corr = correlation_stats(train)
    selected = [0]
    trace: list[dict[str, Any]] = [{"selected_connection": CONNECTIONS[0], "basis": "fixed_first_profile"}]
    while len(selected) < K:
        pred, residual = conditional_prediction(values, selected, mean, std, corr)
        candidates = [index for index in range(len(values)) if index not in selected]
        utility = {
            index: abs(float((pred[index] - mean[index]) / std[index])) * float(residual[index])
            for index in candidates
        }
        chosen = max(candidates, key=lambda index: (utility[index], -index))
        selected.append(chosen)
        trace.append({
            "selected_connection": CONNECTIONS[chosen],
            "basis": "observed_profiles_and_training_fold_statistics",
            "candidate_utility": {str(CONNECTIONS[index]): value for index, value in utility.items()},
        })
    return selected, trace


def matrix_completion(values: np.ndarray, train: np.ndarray, selected: list[int]) -> np.ndarray:
    mean, std, _ = correlation_stats(train)
    normalized = (train - mean) / std
    _u, _s, vt = np.linalg.svd(normalized, full_matrices=False)
    rank = min(2, vt.shape[0], vt.shape[1])
    basis = vt[:rank]
    obs = np.asarray(selected, dtype=int)
    z_observed = (values[obs] - mean[obs]) / std[obs]
    design = basis[:, obs].T
    coefficients = np.linalg.solve(design.T @ design + 0.01 * np.eye(rank), design.T @ z_observed)
    pred = mean + std * (coefficients @ basis)
    pred[obs] = values[obs]
    return pred


def suite_score(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-9)))))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = np.asarray([row["relative_error_percent"] for row in rows], dtype=float)
    saved = np.asarray([row["time_saved_percent"] for row in rows], dtype=float)
    return {
        "n_runs": int(len(rows)),
        "mean_relative_error_percent": float(np.mean(errors)),
        "median_relative_error_percent": float(np.median(errors)),
        "p95_relative_error_percent": float(np.percentile(errors, 95)),
        "max_relative_error_percent": float(np.max(errors)),
        "mean_time_saved_percent": float(np.mean(saved)),
    }


def evaluate_target(
    values: np.ndarray,
    durations: np.ndarray,
    folds: list[list[int]],
    seed: int,
    target_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    names = [
        "fixed_first_k_mean_profile",
        "random_k_correlation",
        "adaptive_correlation",
        "fixed_first_k_correlation",
        "fixed_first_k_matrix_completion_rank2",
        "nested_train_only_best_fixed_k_correlation",
    ]
    method_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    subset_rows: list[dict[str, Any]] = []
    fold_manifest: list[dict[str, Any]] = []
    all_subsets = list(itertools.combinations(range(len(CONNECTIONS)), K))
    for fold_id, test_indices in enumerate(folds):
        test_index = test_indices[0]
        train_indices = [index for index in range(len(values)) if index != test_index]
        train = values[train_indices]
        test = values[test_index]
        mean, std, corr = correlation_stats(train)
        fixed = list(range(K))

        # Select a fixed profile set using only leave-one-run-out predictions within this outer train fold.
        inner_errors = []
        for subset in all_subsets:
            errors = []
            for inner_test_index in train_indices:
                inner_train_indices = [index for index in train_indices if index != inner_test_index]
                inner_train = values[inner_train_indices]
                inner_test = values[inner_test_index]
                inner_mean, inner_std, inner_corr = correlation_stats(inner_train)
                prediction, _ = conditional_prediction(inner_test, list(subset), inner_mean, inner_std, inner_corr)
                predicted_score = suite_score(prediction)
                actual_score = suite_score(inner_test)
                errors.append(abs(predicted_score - actual_score) / max(abs(actual_score), 1e-9) * 100)
            inner_errors.append({"subset": list(subset), "mean_inner_relative_error_percent": float(np.mean(errors))})
        best_subset = min(inner_errors, key=lambda row: (row["mean_inner_relative_error_percent"], row["subset"]))["subset"]

        rng = np.random.RandomState(seed + fold_id + 1)
        random_selected = sorted(map(int, rng.choice(len(CONNECTIONS), size=K, replace=False)))
        adaptive_selected, adaptive_trace = adaptive_selection(test, train)
        method_selections = {
            "fixed_first_k_mean_profile": (fixed, []),
            "random_k_correlation": (random_selected, []),
            "adaptive_correlation": (adaptive_selected, adaptive_trace),
            "fixed_first_k_correlation": (fixed, []),
            "fixed_first_k_matrix_completion_rank2": (fixed, []),
            "nested_train_only_best_fixed_k_correlation": (list(best_subset), []),
        }
        fold_manifest.append({
            "fold": fold_id,
            "train_runs": [f"run-{index + 1}" for index in train_indices],
            "heldout_run": f"run-{test_index + 1}",
            "nested_selector_inner_runs": [f"run-{index + 1}" for index in train_indices],
            "nested_selector_chosen_connections": [CONNECTIONS[index] for index in best_subset],
            "nested_selector_inner_mean_errors_percent": [
                {"connections": [CONNECTIONS[index] for index in row["subset"]],
                 "mean_relative_error_percent": row["mean_inner_relative_error_percent"]}
                for row in inner_errors
            ],
        })

        def score_subset(subset: tuple[int, ...] | list[int]) -> dict[str, Any]:
            prediction, _ = conditional_prediction(test, list(subset), mean, std, corr)
            predicted_score = suite_score(prediction)
            actual_score = suite_score(test)
            selected_cost = float(np.sum(durations[test_index, list(subset)]))
            full_cost = float(np.sum(durations[test_index]))
            return {
                "run": f"run-{test_index + 1}",
                "selected_connections": [CONNECTIONS[index] for index in subset],
                "predicted_suite_geomean": predicted_score,
                "actual_suite_geomean": actual_score,
                "relative_error_percent": abs(predicted_score - actual_score) / max(abs(actual_score), 1e-9) * 100,
                "selected_profile_elapsed_seconds": selected_cost,
                "full_suite_elapsed_seconds": full_cost,
                "time_saved_percent": (full_cost - selected_cost) / full_cost * 100,
            }

        for subset in all_subsets:
            subset_rows.append({"fold": fold_id, "heldout_run": f"run-{test_index + 1}",
                                "connections": [CONNECTIONS[index] for index in subset], **score_subset(subset)})
        for method, (selected, trace) in method_selections.items():
            if method == "fixed_first_k_mean_profile":
                prediction = mean.copy()
                prediction[selected] = test[selected]
            elif method == "fixed_first_k_matrix_completion_rank2":
                prediction = matrix_completion(test, train, selected)
            else:
                prediction, _residual = conditional_prediction(test, selected, mean, std, corr)
            predicted_suite = suite_score(prediction)
            actual_suite = suite_score(test)
            selected_cost = float(np.sum(durations[test_index, selected]))
            full_cost = float(np.sum(durations[test_index]))
            method_rows[method].append({
                "target": target_name,
                "run": f"run-{test_index + 1}",
                "selected_connections": [CONNECTIONS[index] for index in selected],
                "selection_trace": trace,
                "predicted_profile_values": {str(connection): float(prediction[index]) for index, connection in enumerate(CONNECTIONS)},
                "predicted_suite_geomean": predicted_suite,
                "actual_suite_geomean": actual_suite,
                "relative_error_percent": abs(predicted_suite - actual_suite) / max(abs(actual_suite), 1e-9) * 100,
                "selected_profile_elapsed_seconds": selected_cost,
                "full_suite_elapsed_seconds": full_cost,
                "time_saved_percent": (full_cost - selected_cost) / full_cost * 100,
            })
    return ([{"method": name, "k": K, "summary": summarize(rows), "per_run": rows} for name, rows in method_rows.items()],
            fold_manifest, subset_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    logs: dict[tuple[int, int], dict[str, Any]] = {}
    for run_id in range(1, 6):
        for connections in CONNECTIONS:
            path = input_dir / f"run-{run_id}-connections-{connections}.log"
            if not path.is_file():
                raise SystemExit(f"Missing raw run log: {path}")
            logs[(run_id, connections)] = parse_log(path, run_id, connections)

    throughput_values = np.asarray([
        [logs[(run_id, connections)]["requests_per_second"] for connections in CONNECTIONS]
        for run_id in range(1, 6)
    ], dtype=float)
    p99_values = np.asarray([
        [logs[(run_id, connections)]["p99_latency_microseconds"] for connections in CONNECTIONS]
        for run_id in range(1, 6)
    ], dtype=float)
    durations = np.asarray([
        [logs[(run_id, connections)]["elapsed_seconds"] for connections in CONNECTIONS]
        for run_id in range(1, 6)
    ], dtype=float)
    order = np.arange(len(throughput_values))
    np.random.RandomState(args.fold_seed).shuffle(order)
    folds = [list(map(int, part)) for part in np.array_split(order, 5)]
    throughput_methods, fold_manifest, throughput_subsets = evaluate_target(
        throughput_values, durations, folds, args.fold_seed, "requests_per_second"
    )
    p99_methods, p99_fold_manifest, p99_subsets = evaluate_target(
        p99_values, durations, folds, args.fold_seed, "p99_latency_microseconds"
    )
    artifact_files = [
        input_dir / "host_context.txt",
        input_dir / "nginx.conf",
        input_dir / "nginx-error.log",
        input_dir / "nginx-config-test.log",
        input_dir / "http-smoke-check.txt",
        input_dir / "run-order.txt",
        input_dir / "www" / "index.html",
        input_dir / "run_wr_repeats.sh",
        Path(__file__).resolve(),
    ]
    artifact_files += [Path(row["raw_log"]) for row in logs.values()]
    output = {
        "schema": "moebench.web_http_local_oof.v1",
        "protocol": "five-run repeated local HTTP serving workload; five-fold run-level OOF; training statistics fit only on four training runs; nested fixed-subset selection uses inner leave-one-run-out predictions within outer training runs; K=3 of five concurrency profiles",
        "benchmark": {
            "server": "isolated Nginx static-content instance",
            "target": "http://127.0.0.1:18080/",
            "connections": CONNECTIONS,
            "wrk_threads": 4,
            "duration_per_profile_seconds": 20,
            "suite_scores": {
                "requests_per_second": "geometric mean of throughput across five concurrency profiles",
                "p99_latency_microseconds": "geometric mean of p99 latency in microseconds across five concurrency profiles; analyzed as a separate positive-valued target",
            },
            "cost": "measured elapsed real time of selected wrk processes; excludes context collection and model fit",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "fold_seed": args.fold_seed,
        "fold_manifest": fold_manifest,
        "raw_runs": [logs[(run_id, connections)] for run_id in range(1, 6) for connections in CONNECTIONS],
        "targets": {
            "requests_per_second": {"methods": throughput_methods, "all_fixed_k3_subsets_per_fold": throughput_subsets},
            "p99_latency_microseconds": {"fold_manifest": p99_fold_manifest, "methods": p99_methods,
                                          "all_fixed_k3_subsets_per_fold": p99_subsets},
        },
        "artifacts": [{"path": str(path), "sha256": sha256(path)} for path in artifact_files],
        "limitations": [
            "One host and one short collection period; repeated runs are not independent sessions.",
            "Localhost static content does not represent a production Web stack, remote client network, database, cache, or realistic user workload.",
            "The reported time saved is an offline replay of selected profile durations and excludes context collection, model fitting, and orchestration overhead.",
            "The nested best-fixed selector is selected by inner leave-one-run-out error on the four outer training runs, then evaluated on the untouched outer held-out run; its five outer test scores are still highly uncertain.",
            "Enumerated fixed-subset results on outer held-out runs are diagnostic comparisons, not an unbiased estimate of the best subset because every subset is reported and a reader could select the best after seeing those scores.",
            "This pilot is evidence of an HTTP-serving workload only; it does not establish generalization across hosts or deployment conditions.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "requests_per_second": [{"method": method["method"], **method["summary"]} for method in throughput_methods],
        "p99_latency_microseconds": [{"method": method["method"], **method["summary"]} for method in p99_methods],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
