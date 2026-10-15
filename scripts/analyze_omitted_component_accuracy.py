#!/usr/bin/env python3
"""Audit OOF reconstruction error only on unmeasured suite components."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unixbench_targets(run: dict[str, Any]) -> dict[str, float]:
    from moebench.reconstruct.data import extract_test_index_from_block, preferred_run_block

    block = preferred_run_block(run.get("yi") or {})
    if not block:
        raise ValueError("UnixBench run has no preferred result block")
    targets = {}
    for test_id in block.get("tests", {}):
        value = extract_test_index_from_block(block, test_id)
        if value is not None:
            targets[str(test_id)] = float(value)
    return targets


def pts_targets(run: dict[str, Any], test_ids: list[str]) -> dict[str, float]:
    from moebench.phoronix.training_data import primary_value_from_export

    export = (run.get("yi") or {}).get("pts_export") or {}
    targets = {}
    for test_id in test_ids:
        value = primary_value_from_export(export, test_id)
        if value is not None:
            targets[str(test_id)] = float(value)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "agent_workflow" / "rounds",
        help="Directory containing round-1 run-level OOF JSON files.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "agent_workflow" / "rounds" / "round-13-missing-component-audit" / "omitted_component_accuracy.json",
    )
    args = parser.parse_args()

    reports = []
    for report_path in sorted(args.input_dir.glob("round-1-*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != "moebench.hybrid.run_level_oof.v1":
            continue
        reports.append((report_path, report))
    if not reports:
        raise SystemExit(f"No run-level OOF reports found under {args.input_dir}")

    output_runs: list[dict[str, Any]] = []
    summary_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    input_manifest = []
    for report_path, report in reports:
        report_fold = {}
        for fold in report.get("fold_manifest", []):
            for run_path in fold.get("heldout_runs", []):
                report_fold[str(Path(run_path).resolve())] = fold.get("fold")
        input_manifest.append({
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
            "machine": report.get("machine"),
            "benchmark": report.get("benchmark"),
            "pts_suite": report.get("pts_suite"),
            "n_runs": len(report.get("per_run", [])),
        })
        for record in report.get("per_run", []):
            if record is None or record.get("skipped"):
                raise ValueError(f"Skipped/missing held-out record in {report_path}")
            run_path = Path(record["run_json"]).resolve()
            run_key = str(run_path)
            if run_key not in report_fold:
                raise ValueError(f"Run is not listed as held out by its fold manifest: {run_path}")
            run = json.loads(run_path.read_text(encoding="utf-8"))
            predicted = record.get("reconstruction", {}).get("predicted_subtest") or {}
            selected = set(record.get("router", {}).get("selected_test_ids") or [])
            if report.get("benchmark") == "unixbench":
                targets = unixbench_targets(run)
                suite = "unixbench"
            elif report.get("benchmark") == "phoronix":
                targets = pts_targets(run, list(predicted))
                suite = f"pts-{report.get('pts_suite', 'unknown')}"
            else:
                raise ValueError(f"Unsupported benchmark in {report_path}: {report.get('benchmark')}")

            predicted_ids = set(map(str, predicted))
            target_ids = set(targets)
            if predicted_ids != target_ids:
                raise ValueError(
                    f"Prediction/target ID mismatch for {run_path}: "
                    f"missing_predictions={sorted(target_ids - predicted_ids)}, "
                    f"extra_predictions={sorted(predicted_ids - target_ids)}"
                )
            if not selected <= target_ids:
                raise ValueError(f"Selected IDs absent from target profile for {run_path}")
            omitted_ids = sorted(target_ids - selected)
            if not omitted_ids:
                raise ValueError(f"No omitted coordinates to score for {run_path}")

            component_errors = []
            for test_id in omitted_ids:
                target = targets[test_id]
                estimate = float(predicted[test_id])
                if not math.isfinite(target) or not math.isfinite(estimate):
                    raise ValueError(f"Non-finite target or estimate at {run_path}:{test_id}")
                component_errors.append(abs(estimate - target) / max(abs(target), 1e-9) * 100)
            run_summary = {
                "machine": report.get("machine"),
                "suite": suite,
                "run_json": str(run_path),
                "run_sha256": sha256(run_path),
                "fold": report_fold[run_key],
                "n_profile_components": len(target_ids),
                "n_selected": len(selected),
                "selected_test_ids": sorted(selected),
                "n_omitted": len(omitted_ids),
                "omitted_test_ids": omitted_ids,
                "omitted_component_mape_percent": mean(component_errors),
            }
            output_runs.append(run_summary)
            summary_rows[(str(report.get("machine")), suite)].append(run_summary)

    summaries = []
    for (machine, suite), rows in sorted(summary_rows.items()):
        values = [row["omitted_component_mape_percent"] for row in rows]
        summaries.append({
            "machine": machine,
            "suite": suite,
            "n_heldout_runs": len(rows),
            "n_omitted_component_predictions": sum(row["n_omitted"] for row in rows),
            "mean_per_run_omitted_component_mape_percent": mean(values),
            "median_per_run_omitted_component_mape_percent": median(values),
            "per_run_omitted_component_mape_percent": values,
        })

    aggregate_by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        aggregate_by_suite[row["suite"]].append(row)
    suite_summaries = []
    for suite, rows in sorted(aggregate_by_suite.items()):
        per_host = [row["mean_per_run_omitted_component_mape_percent"] for row in rows]
        suite_summaries.append({
            "suite": suite,
            "n_hosts": len(rows),
            "n_heldout_runs": sum(row["n_heldout_runs"] for row in rows),
            "n_omitted_component_predictions": sum(row["n_omitted_component_predictions"] for row in rows),
            "mean_host_omitted_component_mape_percent": mean(per_host),
            "median_host_omitted_component_mape_percent": median(per_host),
        })

    result = {
        "schema": "moebench.omitted_component_accuracy_audit.v1",
        "metric": "For each held-out run, compute MAPE only over component IDs not selected by the router; report per-run means, then equal-weight host means by suite.",
        "target_join": "Each per_run.run_json is resolved to its exact full-run JSON; the run must be declared held out in the OOF report fold_manifest. Predicted and target component ID sets must match exactly.",
        "target_definition": {
            "unixbench": "official per-subtest Index score from the preferred single-copy result block",
            "pts": "primary value from Phoronix Test Suite export; PTS suite-level log-mean proxy is not used in this component metric",
        },
        "inputs": input_manifest,
        "suite_summaries_equal_weight_host_means": suite_summaries,
        "host_suite_summaries": summaries,
        "per_run": output_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "suite_summaries": suite_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
