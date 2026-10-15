#!/usr/bin/env python3
"""One-load history scaling controls for CloudSuite cold-start profiling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONFIGS = ("fpm4", "fpm8")
LOADS = (20, 40, 60, 80, 100)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def curve_mape(prediction: list[float], reference: list[float]) -> float:
    return sum(abs(p - y) / y for p, y in zip(prediction, reference, strict=True)) / len(reference) * 100


def analyze(source: dict) -> dict:
    sessions = tuple(source["primary_sessions"])
    if not source["primary_analysis"]["analysis_complete"] or len(sessions) != 4:
        raise ValueError("requires a complete four-session primary analysis")

    run_rows = {}
    for manifest in source["input_manifests"]:
        for line in Path(manifest["path"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["config"], row["session"], row["phase"], row["scale"])
            if key in run_rows:
                raise ValueError(f"duplicate run cell: {key}")
            run_rows[key] = row

    history_rows = {}
    for row in source["primary_analysis"]["fold_rows"]:
        if row["regime"] == "same_config" and row["method"] == "history_only":
            key = (row["config"], row["heldout_session"])
            if key in history_rows:
                raise ValueError(f"duplicate history row: {key}")
            history_rows[key] = row

    decisions = []
    for target in CONFIGS:
        incumbent = next(config for config in CONFIGS if config != target)
        for session in sessions:
            target_reference = history_rows[(target, session)]["reference_throughput"]
            incumbent_row = history_rows[(incumbent, session)]
            source_curve = incumbent_row["predicted_throughput"]
            incumbent_reference = incumbent_row["reference_throughput"]
            reference_scores = {
                target: geometric_mean(target_reference),
                incumbent: geometric_mean(incumbent_reference),
            }
            reference_winner = max(CONFIGS, key=lambda config: reference_scores[config])
            incumbent_prediction = geometric_mean(source_curve)
            for index, load in enumerate(LOADS):
                cell = run_rows[(target, session, "probe", load)]
                validation = json.loads(Path(cell["summary_validation_json"]).read_text(encoding="utf-8"))
                if not validation["throughput_usable"] or validation["operation_failures"]:
                    raise ValueError(f"invalid target probe: {target} session {session} load {load}")
                probe = float(validation["throughput_ops_per_sec"])
                ratio = probe / source_curve[index]
                prediction = [ratio * value for value in source_curve]
                predicted_winner = target if geometric_mean(prediction) > incumbent_prediction else incumbent
                regret = (reference_scores[reference_winner] - reference_scores[predicted_winner]) / reference_scores[reference_winner] * 100
                decisions.append({
                    "new_config": target,
                    "incumbent_config": incumbent,
                    "heldout_session": session,
                    "load": load,
                    "probe_throughput": probe,
                    "predicted_curve": prediction,
                    "reference_curve": target_reference,
                    "curve_error_percent": curve_mape(prediction, target_reference),
                    "reference_winner": reference_winner,
                    "predicted_winner": predicted_winner,
                    "wrong_choice": predicted_winner != reference_winner,
                    "throughput_regret_percent": regret,
                    "probe_runner_seconds": float(cell["elapsed_seconds"]),
                })

    summary = {}
    for load in LOADS:
        selected = [row for row in decisions if row["load"] == load]
        summary[str(load)] = {
            "mean_curve_error_percent": sum(row["curve_error_percent"] for row in selected) / len(selected),
            "wrong_choices": sum(row["wrong_choice"] for row in selected),
            "decisions": len(selected),
            "mean_throughput_regret_percent": sum(row["throughput_regret_percent"] for row in selected) / len(selected),
            "mean_probe_runner_seconds": sum(row["probe_runner_seconds"] for row in selected) / len(selected),
        }
    return {
        "schema": "moebench.cloudsuite.one_probe_history_scaling.v1",
        "scope": "Post hoc exploratory controls on the strict primary sessions; all five fixed single-load choices are reported.",
        "method": "For each new configuration, scale the known incumbent's training-session mean curve by the ratio of one new-configuration short probe to the incumbent history at that same offered load. Keep the incumbent on a tie. No target full reference is used in prediction.",
        "sessions": list(sessions),
        "summary": summary,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input_json.read_text(encoding="utf-8")))
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for load, row in result["summary"].items():
        print(f"load {load}: {row['mean_curve_error_percent']:.3f}% MAPE; "
              f"{row['wrong_choices']}/{row['decisions']} wrong; "
              f"{row['mean_probe_runner_seconds']:.1f} s runner")


if __name__ == "__main__":
    main()
