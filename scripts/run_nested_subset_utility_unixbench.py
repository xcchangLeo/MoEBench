#!/usr/bin/env python3
"""Nested session-holdout selection of Top-K subsets by suite reconstruction error."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-report", type=Path, required=True)
    parser.add_argument("--probe-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-model", choices=("lightgbm", "ridge"), default="lightgbm")
    parser.add_argument("--reconstructor", choices=("lightgbm", "lightgbm_small", "xgboost", "ridge"), default="lightgbm")
    parser.add_argument("--suite-output-mode", default="probe_plus_imputed_geomean")
    parser.add_argument("--seed", type=int, default=1977)
    args = parser.parse_args()

    outer_report_path = args.outer_report.resolve()
    probe_dataset_path = args.probe_dataset.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outer_report = read_json(outer_report_path)
    probe_dataset = read_json(probe_dataset_path)
    sessions = [item["session"] for item in outer_report["sessions"]]
    test_ids = list(probe_dataset["test_ids"])
    if not 1 <= args.top_k <= len(test_ids):
        raise SystemExit(f"top-k must be in [1, {len(test_ids)}]")
    subsets = list(itertools.combinations(test_ids, args.top_k))
    cv_script = ROOT / "scripts" / "run_hybrid_oof_unixbench_policies.py"
    fold_results: dict[str, Any] = {}

    for outer_index, outer_heldout in enumerate(sessions):
        outer_train = [session for session in sessions if session != outer_heldout]
        subset_errors: list[list[float]] = []
        for inner_index, inner_heldout in enumerate(outer_train):
            inner_sessions = [session for session in outer_train if session != inner_heldout]
            inner_dir = out_dir / f"outer_{outer_heldout}" / f"inner_{inner_heldout}"
            inner_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(cv_script),
                "--machine",
                args.machine,
                "--sessions",
                ",".join(outer_train),
                "--holdout-session",
                inner_heldout,
                "--probe-dataset",
                str(probe_dataset_path),
                "--policies",
                "random_top_k",
                "--top-k",
                str(args.top_k),
                "--probe-model",
                args.probe_model,
                "--reconstructor",
                args.reconstructor,
                "--suite-output-mode",
                args.suite_output_mode,
                "--seed",
                str(args.seed + outer_index * 100 + inner_index),
                "--out-dir",
                str(inner_dir),
            ]
            log_path = inner_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"Inner holdout failed; see {log_path}")
            inner_report = read_json(inner_dir / f"{args.machine}__random_top_k.json")
            valid_runs = [row for row in inner_report.get("per_run", []) if row]
            if len(valid_runs) != 1:
                raise RuntimeError(f"Expected one inner held-out run for {inner_heldout}")
            errors = valid_runs[0].get("all_top_k_subset_relative_errors") or []
            if len(errors) != len(subsets):
                raise RuntimeError(
                    f"Subset error count mismatch for {inner_heldout}: {len(errors)} != {len(subsets)}"
                )
            subset_errors.append([float(value) for value in errors])
            print(
                f"Completed nested inner holdout for outer {outer_index + 1}/{len(sessions)}: "
                f"{inner_heldout}",
                file=sys.stderr,
            )

        mean_inner_errors = [
            sum(row[index] for row in subset_errors) / len(subset_errors)
            for index in range(len(subsets))
        ]
        selected_index = min(range(len(subsets)), key=lambda index: (mean_inner_errors[index], index))
        selected_subset = list(subsets[selected_index])

        outer_dir = outer_report_path.parent / f"holdout_{outer_heldout}"
        outer_random_path = outer_dir / f"{args.machine}__random_top_k.json"
        outer_random_report = read_json(outer_random_path)
        outer_runs = [row for row in outer_random_report.get("per_run", []) if row]
        if len(outer_runs) != 1:
            raise RuntimeError(f"Expected one outer held-out run for {outer_heldout}")
        outer_errors = outer_runs[0].get("all_top_k_subset_relative_errors") or []
        if len(outer_errors) != len(subsets):
            raise RuntimeError(f"Outer subset error count mismatch for {outer_heldout}")
        policy_summary = outer_report["per_session_policy_summary"][outer_heldout]
        fold_results[outer_heldout] = {
            "selected_test_ids": selected_subset,
            "inner_cv_mean_relative_error_percent": mean_inner_errors[selected_index] * 100.0,
            "outer_heldout_relative_error_percent": float(outer_errors[selected_index]) * 100.0,
            "outer_all_subsets_random_mean_error_percent": (
                sum(float(value) for value in outer_errors) / len(outer_errors) * 100.0
            ),
            "outer_learned_router_error_percent": policy_summary.get("learned", {}).get(
                "mean_relative_error_percent"
            ),
            "outer_random_single_draw_error_percent": policy_summary.get("random_top_k", {}).get(
                "mean_relative_error_percent"
            ),
            "num_inner_sessions": len(outer_train),
        }

    nested_errors = [item["outer_heldout_relative_error_percent"] for item in fold_results.values()]
    random_means = [item["outer_all_subsets_random_mean_error_percent"] for item in fold_results.values()]
    learned_errors = [
        float(item["outer_learned_router_error_percent"])
        for item in fold_results.values()
        if item["outer_learned_router_error_percent"] is not None
    ]
    report = {
        "schema": "moebench.unixbench.nested_subset_utility_holdout.v1",
        "machine": args.machine,
        "split_protocol": (
            "outer leave-one-session-out; subset utility optimized by inner leave-one-session-out "
            "over the outer training sessions only"
        ),
        "selection_objective": "minimum inner-fold relative suite-index reconstruction error",
        "top_k": args.top_k,
        "probe_model": args.probe_model,
        "reconstructor": args.reconstructor,
        "suite_output_mode": args.suite_output_mode,
        "outer_sessions": sessions,
        "num_candidate_subsets": len(subsets),
        "folds": fold_results,
        "mean_outer_relative_error_percent": sum(nested_errors) / len(nested_errors),
        "mean_outer_all_subsets_random_error_percent": sum(random_means) / len(random_means),
        "mean_outer_learned_router_error_percent": sum(learned_errors) / len(learned_errors),
    }
    report_path = out_dir / "nested_subset_utility_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "mean_outer_relative_error_percent": report["mean_outer_relative_error_percent"],
                "mean_outer_all_subsets_random_error_percent": report[
                    "mean_outer_all_subsets_random_error_percent"
                ],
                "mean_outer_learned_router_error_percent": report[
                    "mean_outer_learned_router_error_percent"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
