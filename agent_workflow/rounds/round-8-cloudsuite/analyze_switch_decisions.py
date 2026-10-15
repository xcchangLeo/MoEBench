#!/usr/bin/env python3
"""Exploratory incumbent-to-new-configuration decisions from held-out curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONFIGS = ("fpm4", "fpm8")
METHODS = (
    "history_only",
    "nested_train_only_3",
    "adaptive_correlation_3",
    "fixed_evenly_spaced_3",
    "nested_linear_interp_3",
    "adaptive_correlation_linear_interp_3",
    "fixed_evenly_spaced_linear_interp_3",
    "nested_history_scale_3",
    "adaptive_correlation_history_scale_3",
    "fixed_evenly_spaced_history_scale_3",
)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("throughput curves must contain positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze(source: dict) -> dict:
    primary = source["primary_analysis"]
    sessions = tuple(source["primary_sessions"])
    if not primary["analysis_complete"] or len(sessions) != 4:
        raise ValueError("requires the completed four-session primary analysis")

    rows = {}
    for row in primary["fold_rows"]:
        key = (row["regime"], row["method"], row["config"], row["heldout_session"])
        if key in rows:
            raise ValueError(f"duplicate fold row: {key}")
        rows[key] = row

    decisions = []
    for target in CONFIGS:
        incumbent = next(config for config in CONFIGS if config != target)
        for session in sessions:
            incumbent_row = rows[("same_config", "history_only", incumbent, session)]
            incumbent_prediction = geometric_mean(incumbent_row["predicted_throughput"])
            incumbent_actual = geometric_mean(incumbent_row["reference_throughput"])
            for method in METHODS:
                target_row = rows[("cold_start_config", method, target, session)]
                if target_row["source_config"] != incumbent:
                    raise ValueError("cold-start source differs from incumbent")
                if target_row["train_sessions"] != incumbent_row["train_sessions"]:
                    raise ValueError("target and incumbent use different training sessions")
                target_prediction = geometric_mean(target_row["predicted_throughput"])
                target_actual = geometric_mean(target_row["reference_throughput"])
                reference_winner = target if target_actual > incumbent_actual else incumbent
                predicted_winner = target if target_prediction > incumbent_prediction else incumbent
                actual = {target: target_actual, incumbent: incumbent_actual}
                regret = (actual[reference_winner] - actual[predicted_winner]) / actual[reference_winner] * 100
                decisions.append({
                    "method": method,
                    "new_config": target,
                    "incumbent_config": incumbent,
                    "heldout_session": session,
                    "train_sessions": target_row["train_sessions"],
                    "reference_winner": reference_winner,
                    "predicted_winner": predicted_winner,
                    "wrong_choice": predicted_winner != reference_winner,
                    "throughput_regret_percent": regret,
                    "new_config_curve_error_percent": target_row["curve_error_percent"],
                    "new_config_probe_runner_seconds": target_row["measured_probe_runner_seconds"],
                    "new_config_selected_cell_seconds": target_row["reconstructed_selected_cell_seconds"],
                    "new_config_predicted_geomean": target_prediction,
                    "incumbent_predicted_geomean": incumbent_prediction,
                    "new_config_reference_geomean": target_actual,
                    "incumbent_reference_geomean": incumbent_actual,
                })

    summary = {}
    for method in METHODS:
        method_rows = [row for row in decisions if row["method"] == method]
        by_target = {}
        for target in CONFIGS:
            target_rows = [row for row in method_rows if row["new_config"] == target]
            by_target[target] = {
                "wrong_choices": sum(row["wrong_choice"] for row in target_rows),
                "decisions": len(target_rows),
                "mean_regret_percent": sum(row["throughput_regret_percent"] for row in target_rows) / len(target_rows),
            }
        summary[method] = {
            "wrong_choices": sum(row["wrong_choice"] for row in method_rows),
            "decisions": len(method_rows),
            "mean_regret_percent": sum(row["throughput_regret_percent"] for row in method_rows) / len(method_rows),
            "by_new_config": by_target,
        }
    return {
        "schema": "moebench.cloudsuite.exploratory_switch_decision.v1",
        "scope": "Post hoc decision analysis; not a predeclared primary endpoint. Each direction treats the other configuration as the known incumbent and estimates the new configuration with three short probes or a zero-probe history transfer.",
        "incumbent_rule": "Predict the known incumbent from its other three full sessions; choose the new configuration only when its predicted geometric-mean throughput is strictly higher.",
        "reference_rule": "Choose the higher geometric-mean throughput from both configurations' full curves in the same held-out session. Latency SLO is not an optimization target.",
        "source_analysis": "Four-session strict primary analysis in the input JSON; no failed session 3 data are used.",
        "sessions": list(sessions),
        "summary": summary,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input_json.read_text(encoding="utf-8"))
    result = analyze(source)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method, row in result["summary"].items():
        print(f"{method}: {row['wrong_choices']}/{row['decisions']} wrong, {row['mean_regret_percent']:.3f}% mean regret")


if __name__ == "__main__":
    main()
