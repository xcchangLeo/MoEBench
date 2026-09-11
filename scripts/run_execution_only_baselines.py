#!/usr/bin/env python3
"""Evaluate UnixBench baselines that use no training data.

Each method estimates the full UnixBench index directly from the actual
partial subtest scores observed in the same run.  Historical complete runs are
used only as offline replay inputs and for final error accounting; no other
run is used to fit statistics, select a subset, or train a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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

K_DEFAULT = 3
DEFAULT_RANDOM_SEEDS = (42, 7, 21, 100, 123)
CATEGORY_BALANCED_IDS = ("dhry2reg", "fsdisk", "syscall")


def _load_rows(dataset_root: Path, machine: str) -> list[dict[str, Any]]:
    pattern = glob_for_machine(benchmark="unixbench", machine=machine)
    rows: list[dict[str, Any]] = []
    for path in collect_unixbench_run_paths(dataset_root, glob_pattern=pattern):
        ds = json.loads(path.read_text(encoding="utf-8"))
        target = extract_targets_from_dataset(ds, test_ids=INDEX_SUITE_TEST_IDS)
        if target is None:
            continue
        values, suite = target
        times: dict[str, float] = {}
        for tid in INDEX_SUITE_TEST_IDS:
            value = _ti_for_test(ds, tid, parallel_key="1")
            if value is None or value <= 0:
                raise RuntimeError(f"Missing positive timing for {tid} in {path}")
            times[tid] = float(value)
        rows.append(
            {
                "path": str(path),
                "session": path.parent.name,
                "values": np.asarray(values, dtype=float),
                "suite": float(suite),
                "times": times,
                "full_time_s": float(sum(times.values())),
            }
        )
    if not rows:
        raise RuntimeError(f"No usable UnixBench runs found for {machine}")
    return rows


def _geometric_mean(values: np.ndarray) -> float:
    if len(values) == 0 or np.any(values <= 0):
        raise ValueError("geometric mean requires non-empty positive values")
    return float(math.exp(float(np.mean(np.log(values)))))


def _fixed_first_ids(k: int) -> list[str]:
    return list(INDEX_SUITE_TEST_IDS[:k])


def _random_ids(seed: int, k: int) -> list[str]:
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(INDEX_SUITE_TEST_IDS), size=k, replace=False)
    return [INDEX_SUITE_TEST_IDS[int(i)] for i in selected]


def _evaluate_selection(
    method: str,
    rows: list[dict[str, Any]],
    selected_ids_by_run: list[list[str]],
    *,
    selection_rule: str,
    seed: int | None = None,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    index_by_id = {tid: i for i, tid in enumerate(INDEX_SUITE_TEST_IDS)}
    for row, selected_ids in zip(rows, selected_ids_by_run):
        selected_indices = [index_by_id[tid] for tid in selected_ids]
        partial_values = row["values"][selected_indices]
        predicted = _geometric_mean(partial_values)
        actual = row["suite"]
        absolute_error = abs(predicted - actual)
        relative_error = absolute_error / max(abs(actual), 1e-9)
        partial_time = float(sum(row["times"][tid] for tid in selected_ids))
        per_run.append(
            {
                "run_id": Path(row["path"]).name,
                "run_path": row["path"],
                "session": row["session"],
                "selected_test_ids": selected_ids,
                "partial_observed_values": {
                    tid: float(row["values"][index_by_id[tid]])
                    for tid in selected_ids
                },
                "predicted_full_suite": predicted,
                "actual_full_suite": actual,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "partial_wall_time_s": partial_time,
                "full_suite_wall_time_s": row["full_time_s"],
                "estimated_time_saved_fraction": (
                    (row["full_time_s"] - partial_time) / row["full_time_s"]
                ),
                "training_run_paths": [],
                "leakage_flags": [],
            }
        )

    errors = np.asarray([r["relative_error"] for r in per_run], dtype=float)
    abs_errors = np.asarray([r["absolute_error"] for r in per_run], dtype=float)
    return {
        "method": method,
        "kind": "execution_only",
        "n_runs": len(per_run),
        "k": len(selected_ids_by_run[0]),
        "mean_relative_error_pct": float(np.mean(errors) * 100.0),
        "std_relative_error_pct": float(np.std(errors, ddof=0) * 100.0),
        "median_relative_error_pct": float(np.median(errors) * 100.0),
        "p95_relative_error_pct": float(np.percentile(errors, 95) * 100.0),
        "max_relative_error_pct": float(np.max(errors) * 100.0),
        "mae_suite_index": float(np.mean(abs_errors)),
        "rmse_suite_index": float(np.sqrt(np.mean(abs_errors**2))),
        "mean_time_saved_fraction": float(
            np.mean([r["estimated_time_saved_fraction"] for r in per_run])
        ),
        "selection_rule": selection_rule,
        "seed": seed,
        "training_data_used": False,
        "reconstruction": "geometric_mean_of_observed_partial_scores",
        "per_run": per_run,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 Execution-Only UnixBench Baselines",
        "",
        "These methods use no training data, model, historical mean, or "
        "cross-run statistic.",
        "They directly compute a suite estimate from actual observed partial "
        "subtest scores.",
        "",
        "| Method | K | Mean error (%) | Std (%) | Median (%) | P95 (%) | Max (%) | Saved (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["methods"]:
        lines.append(
            f"| {result['method']} | {result['k']} | "
            f"{result['mean_relative_error_pct']:.3f} | "
            f"{result['std_relative_error_pct']:.3f} | "
            f"{result['median_relative_error_pct']:.3f} | "
            f"{result['p95_relative_error_pct']:.3f} | "
            f"{result['max_relative_error_pct']:.3f} | "
            f"{result['mean_time_saved_fraction'] * 100:.3f} |"
        )
    lines.extend(
        [
            "",
            "Evaluation is an offline replay of historical complete runs. "
            "The complete run is used only for final error accounting.",
            "The timing column is estimated from recorded per-subtest wall times; "
            "it is not a live online probe measurement.",
            "",
            "Random subset results are reported per fixed seed and are not "
            "selected using test error.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument(
        "--random-seeds",
        default=",".join(str(x) for x in DEFAULT_RANDOM_SEEDS),
        help="Fixed seeds for random subset controls.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.k < 1 or args.k > len(INDEX_SUITE_TEST_IDS):
        ap.error(f"--k must be in [1, {len(INDEX_SUITE_TEST_IDS)}]")
    if args.k != len(CATEGORY_BALANCED_IDS):
        ap.error(
            f"category-balanced baseline is defined for K={len(CATEGORY_BALANCED_IDS)}"
        )
    try:
        seeds = [int(x.strip()) for x in args.random_seeds.split(",") if x.strip()]
    except ValueError as exc:
        ap.error(f"invalid --random-seeds: {exc}")
    if not seeds:
        ap.error("at least one random seed is required")

    rows = _load_rows(args.dataset_root.resolve(), args.machine)
    fixed_ids = _fixed_first_ids(args.k)
    category_ids = list(CATEGORY_BALANCED_IDS)
    methods: list[dict[str, Any]] = []
    methods.append(
        _evaluate_selection(
            "fixed_partial_geomean",
            rows,
            [fixed_ids for _ in rows],
            selection_rule="fixed first K subtests",
        )
    )
    methods.append(
        _evaluate_selection(
            "category_balanced_partial_geomean",
            rows,
            [category_ids for _ in rows],
            selection_rule="fixed CPU/IO/syscall subtests",
        )
    )
    random_results: list[dict[str, Any]] = []
    for seed in seeds:
        ids = _random_ids(seed, args.k)
        result = _evaluate_selection(
            f"random_partial_geomean_seed_{seed}",
            rows,
            [ids for _ in rows],
            selection_rule="uniform random subset with fixed seed",
            seed=seed,
        )
        result["selected_test_ids"] = ids
        random_results.append(result)
    methods.extend(random_results)

    random_mean_errors = np.asarray(
        [r["mean_relative_error_pct"] for r in random_results], dtype=float
    )
    report = {
        "schema": "moebench.execution_only_baselines.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine,
        "suite": "unixbench",
        "dataset_root": str(args.dataset_root.resolve()),
        "n_runs": len(rows),
        "k": args.k,
        "random_seeds": seeds,
        "methods": methods,
        "random_seed_summary": {
            "mean_of_seed_mean_errors_pct": float(np.mean(random_mean_errors)),
            "std_of_seed_mean_errors_pct": float(np.std(random_mean_errors, ddof=0)),
            "min_seed_mean_error_pct": float(np.min(random_mean_errors)),
            "max_seed_mean_error_pct": float(np.max(random_mean_errors)),
        },
        "protocol": {
            "evaluation": "offline_execution_only_replay",
            "training_data_used": False,
            "cross_run_statistics_used": False,
            "selection_uses_test_error": False,
            "reconstruction": "geometric_mean_of_observed_partial_scores",
            "full_suite_ground_truth": "historical UnixBench system_benchmarks_index_score",
            "online_probe": False,
        },
        "run_paths": [r["path"] for r in rows],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "moebench.execution_only_baselines_manifest.v1",
                "created_at_utc": report["created_at_utc"],
                "command": " ".join(sys.argv),
                "machine": args.machine,
                "suite": "unixbench",
                "k": args.k,
                "random_seeds": seeds,
                "training_data_used": False,
                "methods": [
                    "fixed_partial_geomean",
                    "category_balanced_partial_geomean",
                    *[f"random_partial_geomean_seed_{s}" for s in seeds],
                ],
                "run_paths": report["run_paths"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "baseline_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    (args.out_dir / "per_run_results.json").write_text(
        json.dumps(
            {result["method"]: result["per_run"] for result in methods},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
