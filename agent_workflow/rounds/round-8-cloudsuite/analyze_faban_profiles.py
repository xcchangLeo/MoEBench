#!/usr/bin/env python3
"""Analyze restarted CloudSuite profiles using held-out session folds."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


LOADS = (20, 40, 60, 80, 100)
CONFIGS = ("fpm4", "fpm8")
PHASES = ("probe", "full")
K = 3


def load_cells(root: Path, sessions: tuple[int, ...]) -> dict[tuple[str, int, str, int], dict]:
    rows = {}
    for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        key = (run["config"], int(run["session"]), run["phase"], int(run["scale"]))
        if key in rows:
            raise ValueError(f"duplicate cell {key}")
        if key[0] not in CONFIGS or key[2] not in PHASES or key[3] not in LOADS:
            raise ValueError(f"unexpected cell {key}")
        if int(run["exit_code"]) != 0 or int(run["client_log_protocol_error"]) != 0:
            raise ValueError(f"Faban failed for {key}")
        validation = json.loads(Path(run["summary_validation_json"]).read_text(encoding="utf-8"))
        if not validation.get("summary_valid"):
            raise ValueError(f"invalid summary for {key}: {validation.get('reason')}")
        rows[key] = {**run, "validation": validation}
    expected = set(itertools.product(CONFIGS, sessions, PHASES, LOADS))
    if set(rows) != expected:
        raise ValueError(f"missing={sorted(expected - set(rows))}; extra={sorted(set(rows) - expected)}")
    return rows


def profile_stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    z = (train - mean) / std
    gram = z.T @ z
    norm = np.sqrt(np.diag(gram))
    corr = np.divide(gram, np.outer(norm, norm), out=np.zeros_like(gram),
                     where=np.outer(norm, norm) > 1e-9)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return mean, std, corr


def predict(probe: np.ndarray, subset: tuple[int, ...], train: np.ndarray) -> np.ndarray:
    mean, std, corr = profile_stats(train)
    selected = np.asarray(subset, dtype=int)
    observed = (probe[selected] - mean[selected]) / std[selected]
    inverse = np.linalg.pinv(corr[np.ix_(selected, selected)] + 0.01 * np.eye(len(selected)))
    result = mean.copy()
    for index in range(len(mean)):
        if index in subset:
            result[index] = probe[index]
        else:
            result[index] = mean[index] + std[index] * float(corr[index, selected] @ inverse @ observed)
    return np.maximum(result, 1e-9)


def predict_linear_interpolation(probe: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    """Interpolate observed loads and linearly extrapolate beyond their endpoints."""
    selected = np.sort(np.asarray(subset, dtype=int))
    x = np.asarray(LOADS, dtype=float)
    observed_x = x[selected]
    observed_y = probe[selected]
    result = np.interp(x, observed_x, observed_y)
    left_slope = (observed_y[1] - observed_y[0]) / (observed_x[1] - observed_x[0])
    right_slope = (observed_y[-1] - observed_y[-2]) / (observed_x[-1] - observed_x[-2])
    left = x < observed_x[0]
    right = x > observed_x[-1]
    result[left] = observed_y[0] + left_slope * (x[left] - observed_x[0])
    result[right] = observed_y[-1] + right_slope * (x[right] - observed_x[-1])
    result[selected] = observed_y
    return np.maximum(result, 1e-9)


def predict_history_scale(probe: np.ndarray, subset: tuple[int, ...], history: np.ndarray) -> np.ndarray:
    """Scale the training-fold mean curve by least squares on observed probe loads."""
    selected = np.asarray(subset, dtype=int)
    baseline = np.asarray(history, dtype=float)
    denominator = float(baseline[selected] @ baseline[selected])
    scale = float(probe[selected] @ baseline[selected]) / denominator if denominator > 1e-18 else 1.0
    return np.maximum(scale * baseline, 1e-9)


def adaptive_subset(probe: np.ndarray, train: np.ndarray) -> tuple[int, ...]:
    mean, std, corr = profile_stats(train)
    selected = [0]
    while len(selected) < K:
        observed = np.asarray(selected, dtype=int)
        z_observed = (probe[observed] - mean[observed]) / std[observed]
        inverse = np.linalg.pinv(corr[np.ix_(observed, observed)] + 0.01 * np.eye(len(observed)))
        utility = {}
        for index in range(len(LOADS)):
            if index in selected:
                continue
            weights = corr[index, observed] @ inverse
            predicted_z = float(weights @ z_observed)
            residual = std[index] * np.sqrt(max(0.0, 1.0 - float(weights @ corr[observed, index])))
            utility[index] = abs(predicted_z) * float(residual)
        selected.append(max(utility, key=lambda index: (utility[index], -index)))
    return tuple(selected)


def curve_error(prediction: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual) / np.maximum(actual, 1e-9)) * 100)


def geometric_mean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-9)))))


def analyze(root: Path, sessions: tuple[int, ...]) -> dict:
    if len(sessions) < 4:
        raise ValueError("confirmatory analysis requires four restarted sessions")
    rows = load_cells(root, sessions)
    order_records = {}
    order_pattern = re.compile(
        r"session=(\d+) config=(fpm4|fpm8) php_fpm_max_children=(\d+) "
        r"phase=(probe|full) steady_seconds=(\d+) load_order=([0-9,]+)"
    )
    for line in (root / "randomized-order.txt").read_text(encoding="utf-8").splitlines():
        match = order_pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid order record: {line}")
        session, config, children, phase, steady, loads = match.groups()
        key = (config, int(session), phase)
        declared = tuple(int(load) for load in loads.split(","))
        actual = tuple(sorted(LOADS, key=lambda load: int(rows[(config, int(session), phase, load)][
            "start_epoch_ns"])))
        if key in order_records or int(children) != int(config[-1]) or \
           int(steady) != (60 if phase == "probe" else 120) or \
           sorted(declared) != list(LOADS) or declared != actual:
            raise ValueError(f"declared order differs from actual cells for {key}")
        order_records[key] = declared
    if set(order_records) != set(itertools.product(CONFIGS, sessions, PHASES)):
        raise ValueError("missing or extra randomized-order records")
    phase_rows = {}
    for line in (root / "phase-times.jsonl").read_text(encoding="utf-8").splitlines():
        phase = json.loads(line)
        key = (phase["config"], int(phase["session"]), phase["phase"])
        if key in phase_rows or phase["failed"]:
            raise ValueError(f"duplicate or failed phase {key}")
        if not int(phase["setup_start_epoch_ns"]) < int(phase["setup_end_epoch_ns"]) < int(phase["phase_end_epoch_ns"]):
            raise ValueError(f"invalid phase timestamps {key}")
        phase_rows[key] = phase
    expected_phases = set(itertools.product(CONFIGS, sessions, PHASES))
    if set(phase_rows) != expected_phases:
        raise ValueError(f"phase timing records differ: {sorted(expected_phases - set(phase_rows))}")
    cell_times = {}
    for line in (root / "cell-times.jsonl").read_text(encoding="utf-8").splitlines():
        cell = json.loads(line)
        key = (cell["config"], int(cell["session"]), cell["phase"], int(cell["scale"]))
        if key in cell_times or cell["failed"]:
            raise ValueError(f"duplicate or failed cell timing {key}")
        if not int(cell["setup_start_epoch_ns"]) < int(cell["setup_end_epoch_ns"]) < int(
            cell["cell_end_epoch_ns"]
        ):
            raise ValueError(f"invalid cell timestamps {key}")
        cell_times[key] = cell
    if set(cell_times) != set(rows):
        raise ValueError(f"cell timing records differ: {sorted(set(rows) - set(cell_times))}")
    for config, session in itertools.product(CONFIGS, sessions):
        if int(phase_rows[(config, session, "probe")]["phase_end_epoch_ns"]) >= int(
            phase_rows[(config, session, "full")]["setup_start_epoch_ns"]
        ):
            raise ValueError(f"probe/full phases overlap for {config}, session {session}")
    quality = defaultdict(int)
    operation_slo_violations = defaultdict(int)
    p99_availability = defaultdict(lambda: defaultdict(int))
    for cell in rows.values():
        validation = cell["validation"]
        quality["total_cells"] += 1
        quality["mix_valid_cells"] += bool(validation["mix_valid"])
        quality["throughput_usable_cells"] += bool(validation["throughput_usable"])
        quality["latency_slo_passed_cells"] += bool(validation["latency_slo_passed"])
        quality["operation_failures"] += int(validation["operation_failures"])
        for operation in validation["response_times"]:
            name = operation["name"]
            operation_slo_violations[name] += operation["slo_passed"] is False
            status = ("censored" if operation.get("p99_censored") else
                      "exact" if "p99_ms" in operation else "missing")
            p99_availability[name][status] += 1
    quality["latency_slo_violations_by_operation"] = dict(operation_slo_violations)
    quality["p99_availability_by_operation"] = {
        name: dict(counts) for name, counts in p99_availability.items()
    }
    unusable = [list(key) for key, cell in rows.items() if not cell["validation"]["throughput_usable"]]
    if unusable:
        return {"quality": dict(quality), "unusable_cells": unusable,
                "analysis_complete": False, "reason": "throughput curve has invalid cells"}

    curves = {(config, session, phase): np.asarray([
        rows[(config, session, phase, load)]["validation"]["throughput_ops_per_sec"]
        for load in LOADS], dtype=float)
        for config, session, phase in itertools.product(CONFIGS, sessions, PHASES)}
    subsets = tuple(itertools.combinations(range(len(LOADS)), K))
    fixed = {
        "fixed_first_3": (0, 1, 2),
        "fixed_middle_3": (1, 2, 3),
        "fixed_evenly_spaced_3": (0, 2, 4),
    }
    predictions = {}
    random_predictions = {}
    random_baseline_predictions = {}
    fold_rows = []
    analysis_overheads = []
    regimes = ("same_config", "cold_start_config")
    for regime, config, heldout in itertools.product(regimes, CONFIGS, sessions):
        fit_start = time.perf_counter()
        source_config = config if regime == "same_config" else next(c for c in CONFIGS if c != config)
        train_sessions = tuple(s for s in sessions if s != heldout)
        train_full = np.asarray([curves[(source_config, s, "full")] for s in train_sessions])
        inner_scores = {}
        for subset in subsets:
            errors = []
            for inner_index, inner_session in enumerate(train_sessions):
                inner_train = np.delete(train_full, inner_index, axis=0)
                inner_probe = curves[(source_config, inner_session, "probe")]
                errors.append(curve_error(predict(inner_probe, subset, inner_train),
                                          curves[(source_config, inner_session, "full")]))
            inner_scores[subset] = float(np.mean(errors))
        nested = min(subsets, key=lambda subset: (inner_scores[subset], subset))
        actual = curves[(config, heldout, "full")]
        probe = curves[(config, heldout, "probe")]
        methods = {"nested_train_only_3": nested,
                   "adaptive_correlation_3": adaptive_subset(probe, train_full), **fixed}
        for name, subset in methods.items():
            predicted = predict(probe, subset, train_full)
            predictions[(regime, name, config, heldout)] = predicted
            elapsed = sum(float(rows[(config, heldout, "probe", LOADS[i])]["elapsed_seconds"])
                          for i in subset)
            cell_cost = sum(float(cell_times[(config, heldout, "probe", LOADS[i])]["cell_total_seconds"])
                            for i in subset)
            fold_rows.append({"regime": regime, "method": name, "config": config,
                              "source_config": source_config, "heldout_session": heldout,
                              "train_sessions": train_sessions, "selected_loads": [LOADS[i] for i in subset],
                              "curve_error_percent": curve_error(predicted, actual),
                              "measured_probe_runner_seconds": elapsed,
                              "reconstructed_selected_cell_seconds": cell_cost,
                              "predicted_throughput": predicted.tolist(),
                              "reference_throughput": actual.tolist()})
            policy_name = {
                "nested_train_only_3": "nested",
                "adaptive_correlation_3": "adaptive_correlation",
                "fixed_first_3": "fixed_first",
                "fixed_middle_3": "fixed_middle",
                "fixed_evenly_spaced_3": "fixed_evenly_spaced",
            }[name]
            paired_baselines = {
                f"{policy_name}_linear_interp_3": predict_linear_interpolation(probe, subset),
                f"{policy_name}_history_scale_3": predict_history_scale(
                    probe, subset, train_full.mean(axis=0)
                ),
            }
            for baseline_name, baseline_prediction in paired_baselines.items():
                predictions[(regime, baseline_name, config, heldout)] = baseline_prediction
                fold_rows.append({"regime": regime, "method": baseline_name, "config": config,
                                  "source_config": source_config, "heldout_session": heldout,
                                  "train_sessions": train_sessions, "selected_loads": [LOADS[i] for i in subset],
                                  "curve_error_percent": curve_error(baseline_prediction, actual),
                                  "measured_probe_runner_seconds": elapsed,
                                  "reconstructed_selected_cell_seconds": cell_cost,
                                  "predicted_throughput": baseline_prediction.tolist(),
                                  "reference_throughput": actual.tolist()})
        random_curves = [predict(probe, subset, train_full) for subset in subsets]
        random_predictions[(regime, config, heldout)] = random_curves
        fold_rows.append({"regime": regime, "method": "uniform_random_3", "config": config,
                          "source_config": source_config,
                          "heldout_session": heldout, "train_sessions": train_sessions,
                          "selected_loads": "uniform over all ten three-load subsets",
                          "curve_error_percent": float(np.mean([
                              curve_error(curve, actual) for curve in random_curves])),
                          "measured_probe_runner_seconds": float(np.mean([
                              sum(float(rows[(config, heldout, "probe", LOADS[i])]["elapsed_seconds"])
                                  for i in subset) for subset in subsets])),
                          "reconstructed_selected_cell_seconds": float(np.mean([
                              sum(float(cell_times[(config, heldout, "probe", LOADS[i])]["cell_total_seconds"])
                                  for i in subset) for subset in subsets])),
                          "reference_throughput": actual.tolist()})
        mean = train_full.mean(axis=0)
        random_baselines = {
            "uniform_random_linear_interp_3": [
                predict_linear_interpolation(probe, subset) for subset in subsets
            ],
            "uniform_random_history_scale_3": [
                predict_history_scale(probe, subset, mean) for subset in subsets
            ],
        }
        for baseline_name, baseline_curves in random_baselines.items():
            random_baseline_predictions[(regime, baseline_name, config, heldout)] = baseline_curves
            fold_rows.append({"regime": regime, "method": baseline_name, "config": config,
                              "source_config": source_config, "heldout_session": heldout,
                              "train_sessions": train_sessions,
                              "selected_loads": "uniform over all ten three-load subsets",
                              "curve_error_percent": float(np.mean([
                                  curve_error(curve, actual) for curve in baseline_curves])),
                              "measured_probe_runner_seconds": float(np.mean([
                                  sum(float(rows[(config, heldout, "probe", LOADS[i])]["elapsed_seconds"])
                                      for i in subset) for subset in subsets])),
                              "reconstructed_selected_cell_seconds": float(np.mean([
                                  sum(float(cell_times[(config, heldout, "probe", LOADS[i])]["cell_total_seconds"])
                                      for i in subset) for subset in subsets])),
                              "reference_throughput": actual.tolist()})
        predictions[(regime, "history_only", config, heldout)] = mean
        fold_rows.append({"regime": regime, "method": "history_only", "config": config,
                          "source_config": source_config, "heldout_session": heldout,
                          "train_sessions": train_sessions, "selected_loads": [],
                          "curve_error_percent": curve_error(mean, actual),
                          "measured_probe_runner_seconds": 0,
                          "reconstructed_selected_cell_seconds": 0,
                          "predicted_throughput": mean.tolist(),
                          "reference_throughput": actual.tolist()})
        analysis_overheads.append({"regime": regime, "config": config, "heldout_session": heldout,
                                   "fit_select_and_predict_seconds": time.perf_counter() - fit_start})

    summary = {}
    for regime, method in itertools.product(regimes, sorted({row["method"] for row in fold_rows})):
        selected = [row for row in fold_rows if row["regime"] == regime and row["method"] == method]
        errors = [row["curve_error_percent"] for row in selected]
        costs = [row["measured_probe_runner_seconds"] for row in selected]
        cell_costs = [row["reconstructed_selected_cell_seconds"] for row in selected]
        decisions = []
        for session in sessions:
            actual_scores = {config: geometric_mean(curves[(config, session, "full")]) for config in CONFIGS}
            winner = max(CONFIGS, key=lambda config: actual_scores[config])
            if method.startswith("uniform_random_"):
                if method == "uniform_random_3":
                    curve_pools = [random_predictions[(regime, config, session)] for config in CONFIGS]
                else:
                    curve_pools = [random_baseline_predictions[(regime, method, config, session)]
                                   for config in CONFIGS]
                pairs = itertools.product(*curve_pools)
                chosen_configs = [max(CONFIGS, key=lambda config: geometric_mean(pair[CONFIGS.index(config)]))
                                  for pair in pairs]
                wrong = float(np.mean([choice != winner for choice in chosen_configs]))
                regret = float(np.mean([(actual_scores[winner] - actual_scores[choice]) /
                                        actual_scores[winner] * 100 for choice in chosen_configs]))
                decisions.append({"session": session, "reference_winner": winner,
                                  "random_subset_pairs": len(chosen_configs),
                                  "wrong_choice_probability": wrong,
                                  "expected_throughput_regret_percent": regret})
            else:
                predicted_scores = {config: geometric_mean(predictions[(regime, method, config, session)])
                                    for config in CONFIGS}
                chosen = max(CONFIGS, key=lambda config: predicted_scores[config])
                decisions.append({"session": session, "reference_winner": winner,
                                  "predicted_choice": chosen,
                                  "wrong_choice_probability": float(chosen != winner),
                                  "expected_throughput_regret_percent":
                                  (actual_scores[winner] - actual_scores[chosen]) / actual_scores[winner] * 100})
        summary[f"{regime}/{method}"] = {"mean_curve_error_percent": float(np.mean(errors)),
                           "mean_probe_runner_seconds_per_config": float(np.mean(costs)),
                           "mean_selected_cell_seconds_per_config": float(np.mean(cell_costs)),
                           "expected_wrong_choices": float(sum(row["wrong_choice_probability"] for row in decisions)),
                           "mean_throughput_regret_percent": float(np.mean([
                               row["expected_throughput_regret_percent"] for row in decisions])),
                           "decisions": decisions}
    reference_costs = [float(rows[(config, session, "full", load)]["elapsed_seconds"])
                       for config, session, load in itertools.product(CONFIGS, sessions, LOADS)]
    reference_phase = [float(phase_rows[(config, session, "full")]["phase_total_seconds"])
                       for config, session in itertools.product(CONFIGS, sessions)]
    reference_cell = [sum(float(cell_times[(config, session, "full", load)]["cell_total_seconds"])
                          for load in LOADS)
                      for config, session in itertools.product(CONFIGS, sessions)]
    probe_setup = [float(phase_rows[(config, session, "probe")]["setup_seconds"])
                   for config, session in itertools.product(CONFIGS, sessions)]
    return {"analysis_complete": True, "quality": dict(quality), "sessions": sessions,
            "loads": LOADS, "reference_runner_seconds_per_config": float(np.mean(reference_costs) * len(LOADS)),
            "reference_phase_seconds_per_config": float(np.mean(reference_phase)),
            "reference_cell_seconds_per_config": float(np.mean(reference_cell)),
            "all_five_probe_setup_seconds_per_config": float(np.mean(probe_setup)),
            "selected_cost_is_reconstructed_from_measured_cells": True,
            "declared_order_matches_timestamps": True,
            "summary": summary, "fold_rows": fold_rows,
            "offline_analysis_overheads": analysis_overheads}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_dir, (1, 2, 3, 4))
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("analysis_complete", "quality")}, sort_keys=True))
    if result["analysis_complete"]:
        for name, row in result["summary"].items():
            print(f"{name}: {row['mean_curve_error_percent']:.3f}% error; "
                  f"{row['expected_wrong_choices']:.2f}/4 expected wrong choices; "
                  f"{row['mean_probe_runner_seconds_per_config']:.1f} s/config")


if __name__ == "__main__":
    main()
