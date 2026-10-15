#!/usr/bin/env python3
"""Summarize locked CloudSuite fold errors at the independent-session level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

T_975_DF3 = 3.182446


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict[str, Any]], loads: list[int]) -> dict[str, Any]:
    sessions = sorted({int(row["heldout_session"]) for row in rows})
    per_session = []
    for session in sessions:
        session_rows = [row for row in rows if int(row["heldout_session"]) == session]
        all_errors = []
        unselected_errors = []
        for row in session_rows:
            predicted = row["predicted_throughput"]
            reference = row["reference_throughput"]
            selected = set(row.get("selected_loads", []))
            for index, (estimate, actual) in enumerate(zip(predicted, reference)):
                error = abs(float(estimate) - float(actual)) / max(float(actual), 1e-9) * 100
                all_errors.append(error)
                if loads[index] not in selected:
                    unselected_errors.append(error)
        per_session.append({
            "session": session,
            "profile_count": len(session_rows),
            "curve_mape_percent": sum(all_errors) / len(all_errors),
            "curve_mape_cells": len(all_errors),
            "unselected_load_mape_percent": (
                sum(unselected_errors) / len(unselected_errors) if unselected_errors else None
            ),
            "unselected_load_cells": len(unselected_errors),
        })

    values = [float(row["curve_mape_percent"]) for row in per_session]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half_width = T_975_DF3 * sd / math.sqrt(len(values)) if len(values) == 4 else None
    return {
        "heldout_sessions": sessions,
        "n_independent_sessions": len(sessions),
        "mean_session_mape_percent": mean,
        "session_sd_percent": sd,
        "descriptive_95pct_t_interval": (
            [mean - half_width, mean + half_width] if half_width is not None else None
        ),
        "per_session": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name(
        "final_supplemented_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(
        "session_uncertainty_audit.json"))
    args = parser.parse_args()
    input_path = args.input.resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    analysis = data["primary_analysis"]
    fold_rows = analysis["fold_rows"]
    selected = {
        "same_config": ["history_only", "adaptive_correlation_3"],
        "cold_start_config": [
            "history_only", "adaptive_correlation_3",
            "adaptive_correlation_linear_interp_3",
            "fixed_evenly_spaced_linear_interp_3",
        ],
    }
    summaries = {}
    for regime, methods in selected.items():
        summaries[regime] = {}
        for method in methods:
            rows = [row for row in fold_rows
                    if row["regime"] == regime and row["method"] == method]
            if not rows:
                raise ValueError(f"missing fold rows for {regime}/{method}")
            summaries[regime][method] = summarize(rows, analysis["loads"])

    output = {
        "schema": "moebench.cloudsuite.session_uncertainty_audit.v1",
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "aggregation": "Compute curve MAPE within each held-out session across that session's profiles and offered loads; sessions are the independent units.",
        "unselected_load_metric": "Within each held-out session, compute MAPE only over offered-load cells absent from the method's selected_loads; history-only has no selected loads and therefore uses all loads.",
        "interval_note": "The 95% Student-t interval uses df=3 and is descriptive only; n=4 sessions does not support a general superiority claim.",
        "primary_sessions": analysis["sessions"],
        "loads": analysis["loads"],
        "results": summaries,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
