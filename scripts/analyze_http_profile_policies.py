#!/usr/bin/env python3
"""Analyze the round-8 repeated local HTTP profile study."""

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


CONFIGS = {"workers1": 1, "workers4": 4}
CONNECTIONS = [20, 40, 60, 80, 100]
SESSIONS = [1, 2, 3, 4]
K = 3
P99_THRESHOLD_US = 1000.0
REQUESTS_RE = re.compile(r"^Requests/sec:\s*([0-9.]+)\s*$", re.MULTILINE)
P99_RE = re.compile(r"^\s*99%\s+([0-9.]+)(us|ms|s)\s*$", re.MULTILINE)
ELAPSED_RE = re.compile(r"^elapsed_real_seconds=([0-9.]+)\s*$", re.MULTILINE)
EXIT_RE = re.compile(r"^exit_code=(\d+)\s*$", re.MULTILINE)
ORDER_RE = re.compile(r"^session=(\d+) config=(workers\d+) workers=(\d+) phase=(probe|full) order=([0-9,]+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run_index(path: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["config"]), int(row["session"]), int(row["connections"]), str(row["phase"]))
        if key in rows:
            raise ValueError(f"Duplicate measurement at {path}:{line_number}: {key}")
        if int(row["exit_code"]) != 0:
            raise ValueError(f"Failed measurement at {path}:{line_number}: {key}")
        if int(row["nginx_pid"]) <= 0 or int(row["end_epoch_ns"]) <= int(row["start_epoch_ns"]):
            raise ValueError(f"Invalid server PID or timestamps at {path}:{line_number}: {key}")
        rows[key] = row
    return rows


def audit_run_order(path: Path, run_index: dict[tuple[str, int, int, str], dict[str, Any]]) -> dict[str, Any]:
    parsed = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = ORDER_RE.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Malformed run order at {path}:{line_number}: {line}")
        session, config, workers, phase, levels = match.groups()
        session_id = int(session)
        order = [int(value) for value in levels.split(",")]
        if config not in CONFIGS or CONFIGS[config] != int(workers):
            raise ValueError(f"Config/worker mismatch at {path}:{line_number}")
        if sorted(order) != sorted(CONNECTIONS):
            raise ValueError(f"Load order is not a permutation of {CONNECTIONS} at {path}:{line_number}")
        parsed.append({"session": session_id, "config": config, "workers": int(workers), "phase": phase,
                       "declared_connections_order": order})
    expected_pairs = {(config, session, phase) for config in CONFIGS for session in SESSIONS
                      for phase in ("probe", "full")}
    actual_pairs = {(row["config"], row["session"], row["phase"]) for row in parsed}
    if len(parsed) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError(f"run-order.txt must contain one row for each config/session/phase; got {actual_pairs}")

    actual_order = []
    for row in parsed:
        config, session, phase = row["config"], row["session"], row["phase"]
        ordered = sorted(CONNECTIONS, key=lambda connection: int(
            run_index[(config, session, connection, phase)]["start_epoch_ns"]))
        if ordered != row["declared_connections_order"]:
            raise ValueError(f"Declared load order disagrees with measurement timestamps: {config} session {session}")
        pids = {int(run_index[(config, session, connection, phase)]["nginx_pid"])
                for connection in CONNECTIONS}
        if len(pids) != 1:
            raise ValueError(f"Expected one Nginx instance per phase: {config}/{session}/{phase}: {pids}")
        actual_order.append({**row, "nginx_pid": next(iter(pids)),
                             "timestamp_verified_connections_order": ordered})
    config_orders_by_session = {}
    for session in SESSIONS:
        declared_config_order = [row["config"] for row in parsed
                                 if row["session"] == session and row["phase"] == "probe"]
        rows = sorted((row for row in parsed if row["session"] == session and row["phase"] == "probe"),
                      key=lambda row: min(int(run_index[(row["config"], session, c, "probe")]["start_epoch_ns"])
                                          for c in CONNECTIONS))
        actual_config_order = [row["config"] for row in rows]
        if actual_config_order != declared_config_order or len(set(actual_config_order)) != 2:
            raise ValueError(f"Config order disagrees with run timestamps in session {session}")
        config_orders_by_session[str(session)] = actual_config_order
    phase_order_by_pair = {}
    for config in CONFIGS:
        for session in SESSIONS:
            probe_times = [int(run_index[(config, session, connection, "probe")]["start_epoch_ns"])
                           for connection in CONNECTIONS]
            probe_ends = [int(run_index[(config, session, connection, "probe")]["end_epoch_ns"])
                          for connection in CONNECTIONS]
            full_times = [int(run_index[(config, session, connection, "full")]["start_epoch_ns"])
                          for connection in CONNECTIONS]
            probe_pid = {int(run_index[(config, session, connection, "probe")]["nginx_pid"])
                         for connection in CONNECTIONS}
            full_pid = {int(run_index[(config, session, connection, "full")]["nginx_pid"])
                        for connection in CONNECTIONS}
            if max(probe_ends) >= min(full_times):
                raise ValueError(f"All probes must precede all full references: {config}/{session}")
            if len(probe_pid) != 1 or len(full_pid) != 1 or probe_pid == full_pid:
                raise ValueError(f"Expected separate Nginx instances across phases: {config}/{session}")
            phase_order_by_pair[f"{config}/session-{session}"] = {
                "probe_start_order_connections": [connection for _, connection in sorted(
                    zip(probe_times, CONNECTIONS))],
                "full_start_order_connections": [connection for _, connection in sorted(
                    zip(full_times, CONNECTIONS))],
                "probe_nginx_pid": next(iter(probe_pid)),
                "full_nginx_pid": next(iter(full_pid)),
                "all_probes_completed_before_full_phase": True,
                "separate_nginx_instances": True,
            }
    return {"declared_and_timestamp_verified_orders": actual_order,
            "timestamp_verified_config_order_by_session": config_orders_by_session,
            "timestamp_verified_phase_order_by_config_session": phase_order_by_pair,
            "load_order_matches_timestamps": True,
            "probes_precede_full_references_with_restart": True}


def parse_log(path: Path) -> dict[str, float | str]:
    text = path.read_text(encoding="utf-8")
    requests, p99, elapsed, exit_code = (
        REQUESTS_RE.findall(text), P99_RE.findall(text), ELAPSED_RE.findall(text), EXIT_RE.findall(text)
    )
    if not (len(requests) == len(p99) == len(elapsed) == len(exit_code) == 1):
        raise ValueError(f"Expected one throughput, p99, elapsed, and exit field in {path}")
    if exit_code[0] != "0":
        raise ValueError(f"Nonzero runner exit code in {path}: {exit_code[0]}")
    p99_value, p99_unit = p99[0]
    p99_us = float(p99_value) * {"us": 1.0, "ms": 1000.0, "s": 1_000_000.0}[p99_unit]
    socket_errors = re.findall(r"Socket errors:.*", text)
    if socket_errors and any(int(value) for value in re.findall(r"\b\d+\b", socket_errors[0])):
        raise ValueError(f"Nonzero socket error in {path}: {socket_errors[0]}")
    if "Non-2xx or 3xx responses" in text:
        raise ValueError(f"Non-success HTTP responses in {path}")
    return {
        "requests_per_second": float(requests[0]),
        "p99_latency_microseconds": p99_us,
        "wrk_elapsed_seconds": float(elapsed[0]),
        "raw_log": str(path),
        "raw_log_sha256": sha256(path),
    }


def stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    standardized = (train - mean) / std
    covariance = standardized.T @ standardized
    norm = np.sqrt(np.diag(covariance))
    denominator = np.outer(norm, norm)
    corr = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                     where=denominator > 1e-9)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return mean, std, corr


def predict(values: np.ndarray, selected: list[int], train: np.ndarray) -> np.ndarray:
    mean, std, corr = stats(train)
    prediction = mean.copy()
    if selected:
        obs = np.asarray(selected, dtype=int)
        observed_z = (values[obs] - mean[obs]) / std[obs]
        inverse = np.linalg.pinv(corr[np.ix_(obs, obs)] + 0.01 * np.eye(len(obs)))
        for index in range(len(mean)):
            if index in selected:
                prediction[index] = values[index]
            else:
                weights = corr[index, obs] @ inverse
                prediction[index] = mean[index] + std[index] * float(weights @ observed_z)
    return np.maximum(prediction, 1e-9)


def relative_errors(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - actual) / np.maximum(np.abs(actual), 1e-9) * 100
    return {
        "mean_absolute_relative_error_percent": float(np.mean(error)),
        "p95_absolute_relative_error_percent": float(np.percentile(error, 95)),
        "max_absolute_relative_error_percent": float(np.max(error)),
    }


def suite_score(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-9)))))


def choose_subset(train_full: np.ndarray, train_probe: np.ndarray) -> tuple[list[int], list[dict[str, Any]]]:
    subsets = list(itertools.combinations(range(len(CONNECTIONS)), K))
    scores = []
    for subset in subsets:
        fold_scores = []
        for heldout in range(len(train_full)):
            inner_train = np.delete(train_full, heldout, axis=0)
            prediction = predict(train_probe[heldout], list(subset), inner_train)
            fold_scores.append(float(relative_errors(prediction, train_full[heldout])[
                "mean_absolute_relative_error_percent"]))
        scores.append({"connections": [CONNECTIONS[i] for i in subset],
                       "mean_inner_curve_error_percent": float(np.mean(fold_scores))})
    winner = min(range(len(subsets)), key=lambda i: (scores[i]["mean_inner_curve_error_percent"], subsets[i]))
    return list(subsets[winner]), scores


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {"n": int(len(array)), "mean": float(np.mean(array)),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    index_path = root / "runs.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"Missing run manifest: {index_path}")
    run_index = parse_run_index(index_path)
    expected = {(config, session, connection, phase)
                for config in CONFIGS for session in SESSIONS for connection in CONNECTIONS
                for phase in ("full", "probe")}
    missing = expected - set(run_index)
    extra = set(run_index) - expected
    if missing or extra:
        raise SystemExit(f"Run manifest incomplete or unexpected: missing={sorted(missing)}, extra={sorted(extra)}")

    measurements: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for key, row in run_index.items():
        raw = Path(row["raw_log"])
        if not raw.is_absolute():
            raw = (Path.cwd() / raw).resolve()
        parsed = parse_log(raw)
        expected_duration = 20.0 if key[3] == "full" else 4.0
        lower, upper = ((15.0, 25.0) if key[3] == "full" else (2.5, 8.0))
        if not lower <= float(parsed["wrk_elapsed_seconds"]) <= upper:
            raise ValueError(f"Unexpected {key[3]} duration for {key}: {parsed['wrk_elapsed_seconds']}s")
        parsed.update({"config": key[0], "workers": CONFIGS[key[0]], "session": key[1],
                       "connections": key[2], "phase": key[3],
                       "planned_duration_seconds": expected_duration,
                       "runner_elapsed_wall_seconds": float(row["elapsed_wall_seconds"]),
                       "raw_log": str(raw)})
        measurements[key] = parsed

    order_audit = audit_run_order(root / "run-order.txt", run_index)
    metric_names = ["requests_per_second", "p99_latency_microseconds"]
    probe_fidelity: dict[str, Any] = {}
    for metric in metric_names:
        by_config = {}
        for config in CONFIGS:
            rows = []
            for session in SESSIONS:
                for connection in CONNECTIONS:
                    full = float(measurements[(config, session, connection, "full")][metric])
                    probe = float(measurements[(config, session, connection, "probe")][metric])
                    signed = (probe - full) / max(abs(full), 1e-9) * 100
                    rows.append({"session": session, "connections": connection,
                                 "full_reference": full, "short_probe": probe,
                                 "signed_relative_difference_percent": signed,
                                 "absolute_relative_difference_percent": abs(signed)})
            by_config[config] = {
                "signed_relative_difference_percent": summarize([
                    row["signed_relative_difference_percent"] for row in rows
                ]),
                "absolute_relative_difference_percent": summarize([
                    row["absolute_relative_difference_percent"] for row in rows
                ]),
                "by_session_and_load": rows,
            }
        probe_fidelity[metric] = by_config

    target_summaries: dict[str, Any] = {}
    fold_details: list[dict[str, Any]] = []
    config_policy_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        metric: {"probe_based": [], "train_history_baseline": []} for metric in metric_names
    }
    curve_rows: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metric_names}
    selection_rows: list[dict[str, Any]] = []
    for fold, heldout_session in enumerate(SESSIONS):
        train_sessions = [session for session in SESSIONS if session != heldout_session]
        policy_row: dict[str, Any] = {"fold": fold, "heldout_session": heldout_session,
                                      "train_sessions": train_sessions, "targets": {}}
        for metric in metric_names:
            config_data: dict[str, dict[str, Any]] = {}
            for config in CONFIGS:
                train_full = np.asarray([
                    [measurements[(config, session, connection, "full")][metric] for connection in CONNECTIONS]
                    for session in train_sessions
                ])
                train_probe = np.asarray([
                    [measurements[(config, session, connection, "probe")][metric] for connection in CONNECTIONS]
                    for session in train_sessions
                ])
                actual_full = np.asarray([
                    measurements[(config, heldout_session, connection, "full")][metric]
                    for connection in CONNECTIONS
                ])
                heldout_probe = np.asarray([
                    measurements[(config, heldout_session, connection, "probe")][metric]
                    for connection in CONNECTIONS
                ])
                chosen, inner_scores = choose_subset(train_full, train_probe)
                predictions = {}
                subsets = {"fixed_first_k_short_probe": list(range(K)),
                           "fixed_middle_k3_short_probe": [1, 2, 3],
                           "nested_train_only_k3_short_probe": chosen,
                           "train_history_mean_no_probe": []}
                for method, subset in subsets.items():
                    prediction = (np.mean(train_full, axis=0) if not subset else
                                  predict(heldout_probe, subset, train_full))
                    predictions[method] = prediction
                    selected_runner_seconds = sum(
                        float(measurements[(config, heldout_session, CONNECTIONS[index], "probe")]
                              ["runner_elapsed_wall_seconds"])
                        for index in subset
                    )
                    curve_rows[metric].append({
                        "fold": fold, "heldout_session": heldout_session, "config": config,
                        "target": metric, "method": method,
                        "observed_probe_values": [float(heldout_probe[index]) for index in subset],
                        "selected_connections": [CONNECTIONS[index] for index in subset],
                        "probe_count": len(subset),
                        "nominal_probe_seconds": float(4 * len(subset)),
                        "selected_probe_runner_wall_seconds": selected_runner_seconds,
                        "predicted_values": [float(value) for value in prediction],
                        "actual_full_reference_values": [float(value) for value in actual_full],
                        **relative_errors(prediction, actual_full),
                    })
                random_rows = []
                for random_subset in itertools.combinations(range(len(CONNECTIONS)), K):
                    prediction = predict(heldout_probe, list(random_subset), train_full)
                    selected_runner_seconds = sum(
                        float(measurements[(config, heldout_session, CONNECTIONS[index], "probe")]
                              ["runner_elapsed_wall_seconds"])
                        for index in random_subset
                    )
                    row = {
                        "fold": fold, "heldout_session": heldout_session, "config": config,
                        "target": metric, "method": "uniform_random_k3_short_probe",
                        "random_subset_connections": [CONNECTIONS[index] for index in random_subset],
                        "probe_count": K, "nominal_probe_seconds": float(4 * K),
                        "selected_probe_runner_wall_seconds": selected_runner_seconds,
                        **relative_errors(prediction, actual_full),
                    }
                    curve_rows[metric].append(row)
                    random_rows.append(row)
                random_by_profile = {
                    "mean_absolute_relative_error_percent": float(np.mean([
                        row["mean_absolute_relative_error_percent"] for row in random_rows
                    ])),
                    "p95_absolute_relative_error_percent": float(np.mean([
                        row["p95_absolute_relative_error_percent"] for row in random_rows
                    ])),
                    "selected_probe_runner_wall_seconds": float(np.mean([
                        row["selected_probe_runner_wall_seconds"] for row in random_rows
                    ])),
                }
                curve_rows[metric].append({
                    "fold": fold, "heldout_session": heldout_session, "config": config,
                    "target": metric, "method": "uniform_random_k3_expected_per_profile",
                    "probe_count": K, "nominal_probe_seconds": float(4 * K),
                    **random_by_profile,
                })
                probe_costs = [measurements[(config, heldout_session, CONNECTIONS[index], "probe")]
                               for index in chosen]
                selection_rows.append({
                    "fold": fold, "heldout_session": heldout_session, "config": config,
                    "target": metric, "selected_connections": [CONNECTIONS[index] for index in chosen],
                    "nested_inner_scores": inner_scores,
                    "selected_probe_wall_seconds": float(sum(row["runner_elapsed_wall_seconds"]
                                                               for row in probe_costs)),
                    "selected_probe_process_seconds": float(sum(row["wrk_elapsed_seconds"]
                                                                  for row in probe_costs)),
                })
                config_data[config] = {
                    "train_full": train_full, "train_probe": train_probe,
                    "full": actual_full, "chosen_subset": chosen,
                    "chosen_prediction": predictions["nested_train_only_k3_short_probe"],
                    "training_suite_score": suite_score(np.mean(train_full, axis=0)),
                    "actual_suite_score": suite_score(actual_full),
                    "predicted_suite_score": suite_score(predictions["nested_train_only_k3_short_probe"]),
                }

            if metric == "requests_per_second":
                preferred = max
                regret = lambda actual, selected: (max(actual.values()) / max(selected, 1e-9) - 1.0) * 100
            else:
                preferred = min
                regret = lambda actual, selected: (selected / max(min(actual.values()), 1e-9) - 1.0) * 100
            actual_scores = {config: config_data[config]["actual_suite_score"] for config in CONFIGS}
            predicted_scores = {config: config_data[config]["predicted_suite_score"] for config in CONFIGS}
            training_scores = {config: config_data[config]["training_suite_score"] for config in CONFIGS}
            probe_selected = preferred(CONFIGS, key=lambda config: (predicted_scores[config], config))
            train_selected = preferred(CONFIGS, key=lambda config: (training_scores[config], config))
            actual_preferred = preferred(CONFIGS, key=lambda config: (actual_scores[config], config))
            probe_result = {
                "predicted_suite_scores_by_config_from_short_probes": predicted_scores,
                "heldout_full_reference_suite_scores_by_config": actual_scores,
                "selected_config_from_short_probes": probe_selected,
                "heldout_preferred_config": actual_preferred,
                "ranking_correct": probe_selected == actual_preferred,
                "relative_selection_regret_percent": regret(actual_scores, actual_scores[probe_selected]),
                "selection_requires_probes_on_both_configurations": True,
            }
            train_result = {
                "training_suite_scores_by_config": training_scores,
                "selected_config_from_training_history": train_selected,
                "heldout_full_reference_suite_scores_by_config": actual_scores,
                "heldout_preferred_config": actual_preferred,
                "ranking_correct": train_selected == actual_preferred,
                "relative_selection_regret_percent": regret(actual_scores, actual_scores[train_selected]),
            }
            policy_row["targets"][metric] = {
                "short_probe_policy": probe_result,
                "train_history_baseline": train_result,
            }
            config_policy_rows[metric]["probe_based"].append(probe_result)
            config_policy_rows[metric]["train_history_baseline"].append(train_result)
            selection_rows[-2]["used_for_probe_based_configuration_decision"] = True
            selection_rows[-1]["used_for_probe_based_configuration_decision"] = True
            costs_for_both = [row for row in selection_rows[-2:]]
            policy_row["targets"][metric]["short_probe_policy"]["selected_probe_wall_seconds_both_configs"] = float(
                sum(row["selected_probe_wall_seconds"] for row in costs_for_both))
        fold_details.append(policy_row)

    for metric in metric_names:
        methods = {}
        for method in ("fixed_first_k_short_probe", "fixed_middle_k3_short_probe",
                       "nested_train_only_k3_short_probe", "train_history_mean_no_probe",
                       "uniform_random_k3_expected_per_profile"):
            rows = [row for row in curve_rows[metric] if row["method"] == method]
            errors = [row["mean_absolute_relative_error_percent"] for row in rows]
            p95_errors = [row["p95_absolute_relative_error_percent"] for row in rows]
            methods[method] = {"mean_curve_error_percent": summarize(errors),
                               "mean_within_curve_p95_error_percent": summarize(p95_errors),
                               "mean_selected_probe_runner_wall_seconds": summarize([
                                   row["selected_probe_runner_wall_seconds"] for row in rows
                               ]),
                               "nominal_probe_seconds_per_profile": float(
                                   np.mean([row["nominal_probe_seconds"] for row in rows])
                               ),
                               "per_fold_config": rows}
        target_summaries[metric] = {
            "methods": methods,
            "configuration_policy": {
                policy_name: {
                    "ranking_accuracy_percent": summarize([100.0 if row["ranking_correct"] else 0.0
                                                            for row in rows]),
                    "relative_selection_regret_percent": summarize([
                        row["relative_selection_regret_percent"] for row in rows
                    ]),
                    "per_fold": rows,
                }
                for policy_name, rows in config_policy_rows[metric].items()
            },
        }

    selected_policy_cost_rows = []
    for metric in metric_names:
        for fold, policy in enumerate(fold_details):
            per_config_selections = [row for row in selection_rows
                                     if row["fold"] == fold and row["target"] == metric]
            if {row["config"] for row in per_config_selections} != set(CONFIGS):
                raise ValueError(f"Expected selected short-probe profiles for both configs, fold={fold}, {metric}")
            selected_policy_cost_rows.append({
                "fold": fold,
                "heldout_session": policy["heldout_session"],
                "target": metric,
                "selected_config_after_both_configs_probed": policy["targets"][metric]
                    ["short_probe_policy"]["selected_config_from_short_probes"],
                "probed_configs_and_selected_connections": [
                    {"config": row["config"], "connections": row["selected_connections"]}
                    for row in per_config_selections
                ],
                "selected_probe_wall_seconds_both_configs": float(sum(
                    row["selected_probe_wall_seconds"] for row in per_config_selections)),
                "selected_probe_process_seconds_both_configs": float(sum(
                    row["selected_probe_process_seconds"] for row in per_config_selections)),
            })

    threshold_counts = {}
    for config in CONFIGS:
        threshold_counts[config] = {}
        for phase in ("full", "probe"):
            violating = []
            for session in SESSIONS:
                for connection in CONNECTIONS:
                    value = measurements[(config, session, connection, phase)]["p99_latency_microseconds"]
                    if value > P99_THRESHOLD_US:
                        violating.append({"session": session, "connections": connection,
                                          "p99_latency_microseconds": value})
            threshold_counts[config][phase] = {
                "violations": len(violating), "opportunities": len(SESSIONS) * len(CONNECTIONS),
                "threshold_microseconds": P99_THRESHOLD_US, "details": violating,
            }
        confusion = {name: 0 for name in ("true_positive", "false_positive", "false_negative", "true_negative")}
        for session in SESSIONS:
            for connection in CONNECTIONS:
                full_violation = (measurements[(config, session, connection, "full")]
                                  ["p99_latency_microseconds"] > P99_THRESHOLD_US)
                probe_violation = (measurements[(config, session, connection, "probe")]
                                   ["p99_latency_microseconds"] > P99_THRESHOLD_US)
                label = ("true_positive" if probe_violation and full_violation else
                         "false_positive" if probe_violation else
                         "false_negative" if full_violation else "true_negative")
                confusion[label] += 1
        threshold_counts[config]["probe_vs_full_confusion"] = {
            **confusion,
            "opportunities": len(SESSIONS) * len(CONNECTIONS),
            "threshold_microseconds": P99_THRESHOLD_US,
        }

    artifact_paths = [root.parent / "README.md", root.parent / "nginx.template.conf",
                      root.parent / "www" / "index.html", root.parent / "run_http_profile_study.sh",
                      index_path, root / "run-order.txt",
                      root / "host-context.txt", Path(__file__).resolve()]
    artifact_paths += sorted(root.glob("*-probe.conf")) + sorted(root.glob("*-full.conf"))
    artifact_paths += sorted(root.glob("*-probe-nginx-test.log")) + sorted(root.glob("*-full-nginx-test.log"))
    artifact_paths += sorted(root.glob("*-probe-nginx-error.log")) + sorted(root.glob("*-full-nginx-error.log"))
    artifact_paths += [Path(row["raw_log"]) for row in measurements.values()]
    missing_artifacts = [str(path) for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError("Missing analysis provenance artifacts: " + ", ".join(missing_artifacts))
    output = {
        "schema": "moebench.http_profile_policy_analysis.v2",
        "protocol": "four paired repeated blocks; per config/session five 4 s probes run first in randomized order on one fresh Nginx instance, followed after stop/restart by five 20 s full references in separately randomized order; leave-one-session-out nested K=3 selection uses training-block short probes and predicts held-out full-reference profiles from held-out short probes; configuration choice ranks predictions for both configurations",
        "benchmark": {"configs": CONFIGS, "concurrency_levels": CONNECTIONS, "sessions": SESSIONS,
                      "full_reference_seconds": 20, "probe_seconds": 4, "probe_budget_profiles": K,
                      "p99_illustrative_threshold_microseconds": P99_THRESHOLD_US,
                      "suite_score": "geometric mean across five full-reference profile values"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "run_order_audit": order_audit,
        "short_probe_fidelity": {
            "description": "Paired 4 s short-probe values compared with the corresponding 20 s full-reference values; reported separately for throughput and wrk-reported p99.",
            "by_target": probe_fidelity,
        },
        "folds": fold_details,
        "curve_predictions": target_summaries,
        "selected_probe_cost": {
            "description": "Configuration choice requires probing both candidate configurations. Cost sums the K=3 selected 4 s probes on each candidate and excludes context collection, fitting, and orchestration.",
            "probe_based_configuration_comparison": {
                "per_fold_target": selected_policy_cost_rows,
                "runner_wall_seconds_both_configs": summarize([row["selected_probe_wall_seconds_both_configs"]
                                                   for row in selected_policy_cost_rows]),
                "wrk_process_seconds_both_configs": summarize([row["selected_probe_process_seconds_both_configs"]
                                                   for row in selected_policy_cost_rows]),
            },
            "per_fold_config_target": selection_rows,
            "runner_wall_seconds": summarize([row["selected_probe_wall_seconds"] for row in selection_rows]),
            "wrk_process_seconds": summarize([row["selected_probe_process_seconds"] for row in selection_rows]),
        },
        "p99_threshold_counts": threshold_counts,
        "raw_measurements": list(measurements.values()),
        "artifacts": [{"path": str(path), "sha256": sha256(path)} for path in artifact_paths],
        "limitations": [
            "Four repeated blocks on one host and one collection period are descriptive, not independent host or deployment sessions.",
            "The workload is loopback static content and does not represent CloudSuite, a production web stack, or remote clients.",
            "Configuration selection and profile selection are evaluated on a very small sample; the ranking results are not evidence of a stable winner.",
            "Short probes use a 4 s wrk run; p99 fidelity can differ from the 20 s full-reference runs, especially under low request counts.",
            "Probe-first phase ordering and separate Nginx master PIDs remove direct same-load full-run conditioning, but do not control all host drift or cache effects between phases.",
            "The 1 ms p99 threshold is illustrative for this laboratory workload and is not a production SLO; probe-phase threshold counts are diagnostics only.",
            "Selected probe cost excludes context collection, model fitting, and orchestration overhead.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({metric: {
        "nested_short_probe_curve_error_percent": target_summaries[metric]["methods"]
            ["nested_train_only_k3_short_probe"]["mean_curve_error_percent"],
        "probe_based_configuration_ranking": target_summaries[metric]["configuration_policy"]
            ["probe_based"]["ranking_accuracy_percent"],
        "probe_based_selection_regret_percent": target_summaries[metric]["configuration_policy"]
            ["probe_based"]["relative_selection_regret_percent"],
    } for metric in metric_names} | {"p99_threshold_counts": threshold_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
