#!/usr/bin/env python3
"""Create one auditable baseline report for the local M1/H1 host."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moebench.phoronix.training_data import collect_phoronix_run_paths
from moebench.reconstruct.data import collect_unixbench_run_paths


M1 = "aces-System-Product-Name"
SUITES = {
    "unixbench": ("aces-System-Product-Name_20260524T102558Z/run-*.json", None),
    "phoronix_cpu": ("aces-System-Product-Name_cpu_20260520T033403Z/run-*.json", "cpu"),
    "phoronix_gpu": (
        "aces-System-Product-Name_pts_nvidia-gpu-compute_20260522T041557Z/run-*.json",
        "pts/nvidia-gpu-compute",
    ),
}


def _policy_rows(report_dir: Path, suite: str) -> list[dict[str, Any]]:
    path = report_dir / f"policy_{suite}_{M1}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    block = (data.get("suite_results") or [{}])[0]
    rows = []
    for combo in block.get("combinations") or []:
        oof = combo.get("oof_metrics") or {}
        dist = oof.get("suite_relative_error_distribution") or {}
        time = combo.get("time_savings") or {}
        rows.append(
            {
                "suite": suite,
                "method": combo.get("policy"),
                "kind": "subset",
                "seed": data.get("seed"),
                "source_dir": str(report_dir),
                "cv_mode": block.get("cv_mode_effective"),
                "session_holdout_available": block.get("session_holdout_available"),
                "n_runs": block.get("n_samples"),
                "mean_error_pct": 100 * oof.get("mean_suite_relative_error", 0.0),
                "median_error_pct": 100 * dist.get("median", 0.0),
                "p95_error_pct": 100 * dist.get("p95", 0.0),
                "max_error_pct": 100 * dist.get("max", 0.0),
                "mae_suite_index": oof.get("mae_suite_index"),
                "rmse_suite_index": oof.get("rmse_suite_index"),
                "saved_pct": 100
                * time.get("mean_fraction_wall_time_saved_vs_full_suite", 0.0),
            }
        )
    return rows


def _full_time(suite: str) -> dict[str, Any]:
    pattern, pts_suite = SUITES[suite]
    root = REPO / "dataset"
    if suite == "unixbench":
        paths = collect_unixbench_run_paths(root, glob_pattern=pattern)
    else:
        paths = collect_phoronix_run_paths(
            root, glob_pattern=pattern, pts_suite=pts_suite
        )
    values = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        by_test = (data.get("ti") or {}).get("by_test_id") or {}
        total = 0.0
        for value in by_test.values():
            if isinstance(value, dict) and value.get("time_s_total") is not None:
                total += float(value["time_s_total"])
            elif isinstance(value, dict):
                total += sum(
                    float(v) for v in value.values() if isinstance(v, (int, float))
                )
        if total > 0:
            values.append(total)
    return {
        "suite": suite,
        "method": "full_suite",
        "kind": "reference",
        "n_runs": len(values),
        "mean_full_time_s": statistics.mean(values) if values else None,
        "run_times_s": values,
        "mean_error_pct": 0.0,
        "saved_pct": 0.0,
    }


def _execution_free(
    report_root: Path | None = None,
    suites: set[str] | None = None,
) -> list[dict[str, Any]]:
    if report_root is not None:
        report_root = report_root.resolve()
    if report_root is not None and report_root.is_dir():
        out: list[dict[str, Any]] = []
        for path in sorted(report_root.glob("execution_free_seed*/unixbench/*/*/summary.json")):
            if suites is not None and "unixbench" not in suites:
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            for method in data.get("methods") or []:
                out.append(
                    {
                        "suite": "unixbench",
                        "method": method.get("method"),
                        "kind": "execution_free",
                        "seed": int(path.parts[-5].removeprefix("execution_free_seed")),
                        "n_runs": data.get("n_runs"),
                        "mean_error_pct": method.get("mean_suite_rel_err_pct"),
                        "saved_pct": None,
                        "time_s": method.get("median_xi_wall_s"),
                        "source": str(path.resolve().relative_to(REPO)),
                    }
                )
        if out:
            return out
    path = REPO / "dataset/experiments/exec_free_table3_wang_tousi_mlp_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    mapping = {"unixbench": "unixbench", "pts_cpu": "phoronix_cpu", "pts_gpu": "phoronix_gpu"}
    for source_suite, suite in mapping.items():
        row = data["suites"][source_suite]["by_host"].get("M1")
        if not row:
            continue
        for method in ("wang_dnn", "tousi_mlp"):
            out.append(
                {
                    "suite": suite,
                    "method": method,
                    "kind": "execution_free",
                    "n_runs": 5,
                    "mean_error_pct": row[method]["err_pct"],
                    "saved_pct": None,
                    "time_s": row[method]["time_s"],
                "source": str(path.relative_to(REPO)),
                }
            )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["method"]), []).append(row)
    summaries = []
    for (suite, method), values in sorted(grouped.items()):
        numeric = lambda name: [
            float(v[name]) for v in values if v.get(name) is not None
        ]
        mean_errors = numeric("mean_error_pct")
        summaries.append(
            {
                "suite": suite,
                "method": method,
                "kind": values[0]["kind"],
                "seeds_or_reports": len(values),
                "mean_error_pct": statistics.mean(mean_errors) if mean_errors else None,
                "std_error_pct": statistics.stdev(mean_errors)
                if len(mean_errors) > 1
                else 0.0
                if mean_errors
                else None,
                "mean_saved_pct": statistics.mean(numeric("saved_pct"))
                if numeric("saved_pct")
                else None,
                "reports": values,
            }
        )
    return {
        "schema": "moebench.m1_baseline_report.v1",
        "host": "M1",
        "hostname_slug": M1,
        "note": "Strict end-to-end offline replay: fold-local router/reconstructor; "
        "run-level five-fold CV within one session; no session holdout.",
        "protocol": {
            "end_to_end": True,
            "router_fit": "training_fold_only",
            "reconstructor_fit": "training_fold_only",
            "greedy_order_fit": "training_fold_only",
            "test_run_timing_used_for_selection": False,
            "partial_inputs": "historical_run_replay",
        },
        "rows": rows,
        "summaries": summaries,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 Baseline Report",
        "",
        "Host: M1 (`aces-System-Product-Name`, 32U128G).",
        "Subset policies use strict end-to-end offline replay with run-level five-fold CV "
        "within the canonical single session.",
        "No row is a session-level temporal holdout.",
        "",
        "| Suite | Method | Kind | Reports | Mean error (%) | Std (%) | Saved (%) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        err = "n/a" if row["mean_error_pct"] is None else f"{row['mean_error_pct']:.3f}"
        std = "n/a" if row["std_error_pct"] is None else f"{row['std_error_pct']:.3f}"
        saved = "n/a" if row["mean_saved_pct"] is None else f"{row['mean_saved_pct']:.3f}"
        lines.append(
            f"| {row['suite']} | {row['method']} | {row['kind']} | "
            f"{row['seeds_or_reports']} | {err} | {std} | {saved} |"
        )
    lines.extend(["", "Full-suite wall time is recorded in the JSON reference rows."])
    missing = report.get("missing_reports") or []
    if missing:
        lines.extend(
            [
                "",
                "Incomplete or not-requested reports (not treated as zero or failed measurements):",
            ]
        )
        for item in missing:
            lines.append(
                f"- `{item['source_dir']}`: `{item['suite']}` "
                f"({item['status']})"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--suites",
        default="unixbench,phoronix_cpu,phoronix_gpu",
        help="Comma-separated suites to include.",
    )
    parser.add_argument(
        "--execution-free-root",
        type=Path,
        default=None,
        help="Root containing execution_free_seed*/.../summary.json outputs.",
    )
    args = parser.parse_args()
    selected_suites = {x.strip() for x in args.suites.split(",") if x.strip()}
    unknown_suites = selected_suites - set(SUITES)
    if unknown_suites:
        parser.error(f"unknown suite(s): {', '.join(sorted(unknown_suites))}")
    rows: list[dict[str, Any]] = []
    missing_reports: list[dict[str, str]] = []
    for directory in args.policy_dirs:
        for suite in sorted(selected_suites):
            suite_rows = _policy_rows(directory, suite)
            expected = directory / f"policy_{suite}_{M1}.json"
            if not expected.is_file():
                missing_reports.append(
                    {
                        "source_dir": str(directory),
                        "suite": suite,
                        "expected_report": str(expected),
                        "status": "not_completed_or_not_requested",
                    }
                )
            seed_from_name = None
            name = directory.name.lower()
            if "seed7" in name:
                seed_from_name = 7
            elif "seed21" in name:
                seed_from_name = 21
            elif name == "paper_reviewer_baselines":
                seed_from_name = 42
            for row in suite_rows:
                row["seed"] = row["seed"] or seed_from_name
            rows.extend(suite_rows)
    for suite in sorted(selected_suites):
        rows.append(_full_time(suite))
    rows.extend(_execution_free(args.execution_free_root, selected_suites))
    report = summarize(rows)
    report["missing_reports"] = missing_reports
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.json} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
