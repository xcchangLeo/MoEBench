#!/usr/bin/env python3
"""Summarize reviewer baseline reports across fixed random seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def stdev(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0 if xs else None


def metric(combo: dict[str, Any], name: str) -> float | None:
    oof = combo.get("oof_metrics") or {}
    dist = oof.get("suite_relative_error_distribution") or {}
    ts = combo.get("time_savings") or {}
    aliases = {
        "mean_error": oof.get("mean_suite_relative_error"),
        "median_error": dist.get("median"),
        "p95_error": dist.get("p95"),
        "max_error": dist.get("max"),
        "saved_fraction": ts.get("mean_fraction_wall_time_saved_vs_full_suite"),
    }
    value = aliases[name]
    return float(value) if value is not None and math.isfinite(float(value)) else None


def collect(input_dirs: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for directory in input_dirs:
        for path in sorted(directory.glob("policy_*.json")):
            if path.name.endswith("_merged.json") or path.name.endswith("_summary.json"):
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            blocks = report.get("suite_results") or [
                {
                    "suite_key": report.get("suite_key"),
                    "combinations": report.get("combinations") or [],
                }
            ]
            for block in blocks:
                suite = str(block.get("suite_key"))
                for combo in block.get("combinations") or []:
                    policy = str(combo.get("policy"))
                    row = {
                        "source": str(path),
                        "suite": suite,
                        "policy": policy,
                        "n_samples": report.get("n_samples"),
                        "cv_mode_effective": report.get("cv_mode_effective"),
                        "session_holdout_available": report.get("session_holdout_available"),
                    }
                    for name in ("mean_error", "median_error", "p95_error", "max_error", "saved_fraction"):
                        row[name] = metric(combo, name)
                    rows.setdefault((suite, policy), []).append(row)
    return rows


def summarize(rows: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    result = []
    for (suite, policy), values in sorted(rows.items()):
        stats: dict[str, Any] = {
            "suite": suite,
            "policy": policy,
            "n_reports": len(values),
            "n_hosts": len({Path(v["source"]).stem.rsplit("_", 1)[-1] for v in values}),
            "cv_modes": sorted({v["cv_mode_effective"] for v in values}),
            "session_holdout_available": any(
                bool(v["session_holdout_available"]) for v in values
            ),
        }
        for name in ("mean_error", "median_error", "p95_error", "max_error", "saved_fraction"):
            xs = [v[name] for v in values if v[name] is not None]
            stats[name] = {"mean": mean(xs), "std": stdev(xs), "n": len(xs)}
        result.append(stats)
    return {
        "schema": "moebench.reviewer_baseline_summary.v1",
        "input_note": "Per-host run-level five-fold reports from the available fixed-seed run; one session per host-suite pair.",
        "results": result,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reviewer Baseline Summary",
        "",
        "All values are means across available host reports. The available completed run used fixed seed 42.",
        "Errors are relative errors; saved is the fraction of full-suite wall time saved.",
        "The effective split is run-level five-fold CV because each host-suite pair has one session.",
        "",
        "| Suite | Policy | Reports | Mean error (%) | Std (%) | Median (%) | P95 (%) | Max (%) | Saved (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        def pct(name: str, digits: int = 3) -> str:
            value = row[name]["mean"]
            return "n/a" if value is None else f"{100 * value:.{digits}f}"

        def pct_std(name: str, digits: int = 3) -> str:
            value = row[name]["std"]
            return "n/a" if value is None else f"{100 * value:.{digits}f}"

        lines.append(
            f"| {row['suite']} | {row['policy']} | {row['n_reports']} | "
            f"{pct('mean_error')} | {pct_std('mean_error')} | "
            f"{pct('median_error')} | {pct('p95_error')} | {pct('max_error')} | "
            f"{pct('saved_fraction')} |"
        )
    lines.extend(
        [
            "",
            "No row is a session-level temporal holdout. Correlation metadata is computed from the training fold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(collect(args.input_dirs))
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.markdown)
    print(f"wrote {args.json} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
