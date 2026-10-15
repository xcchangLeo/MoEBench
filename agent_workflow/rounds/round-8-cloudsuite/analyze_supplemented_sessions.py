#!/usr/bin/env python3
"""Select complete CloudSuite sessions, run strict OOF, and isolate failure sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
import tempfile
from pathlib import Path
from typing import Any

LOADS = (20, 40, 60, 80, 100)
CONFIGS = ("fpm4", "fpm8")
PHASES = ("probe", "full")
EXPECTED_CELLS = set(itertools.product(CONFIGS, PHASES, LOADS))
ROUND_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "round8_analyze_faban_profiles", ROUND_DIR / "analyze_faban_profiles.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load analyze_faban_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(roots: list[Path]) -> tuple[dict[tuple[str, int, str, int], dict], dict[int, list[dict]]]:
    records: dict[tuple[str, int, str, int], dict] = {}
    by_session: dict[int, list[dict]] = {}
    for root in roots:
        manifest = root / "runs.jsonl"
        if not manifest.is_file():
            raise ValueError(f"missing run manifest: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["config"]), int(row["session"]), str(row["phase"]), int(row["scale"]))
            if key[0] not in CONFIGS or key[2] not in PHASES or key[3] not in LOADS:
                raise ValueError(f"unexpected run cell: {key}")
            if key in records:
                raise ValueError(f"duplicate session cell across inputs: {key}")
            row["_source_root"] = str(root.resolve())
            records[key] = row
            by_session.setdefault(key[1], []).append(row)
    if not by_session:
        raise ValueError("no run records found")
    return records, by_session


def cell_validity(row: dict) -> tuple[bool, list[str], dict | None]:
    reasons: list[str] = []
    if int(row.get("exit_code", 1)) != 0:
        reasons.append(f"client_exit_code={row.get('exit_code')}")
    if int(row.get("client_log_protocol_error", 1)) != 0:
        reasons.append("client_protocol_error")
    validation_path = Path(row.get("summary_validation_json", ""))
    if not validation_path.is_file():
        return False, reasons + ["missing_summary_validation"], None
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("summary_valid"):
        reasons.append(f"invalid_summary:{validation.get('reason', 'unknown')}")
    if not validation.get("mix_valid"):
        reasons.append("operation_mix_invalid")
    if int(validation.get("operation_failures", 0)) != 0:
        failures = [
            {"operation": op.get("name"), "failures": int(op.get("failures", 0))}
            for op in validation.get("operations", []) if int(op.get("failures", 0)) > 0
        ]
        reasons.append(f"transaction_failures={validation['operation_failures']}:{failures}")
    if not validation.get("throughput_usable"):
        reasons.append("throughput_not_usable_under_zero_failure_rule")
    return not reasons, reasons, validation


def session_audit(by_session: dict[int, list[dict]]) -> list[dict[str, Any]]:
    audit = []
    for session, rows in by_session.items():
        by_key = {(r["config"], r["phase"], int(r["scale"])): r for r in rows}
        present = set(by_key)
        reasons = []
        for missing in sorted(EXPECTED_CELLS - present):
            reasons.append({"cell": list(missing), "reasons": ["missing_cell"]})
        failed_cells = []
        for key in sorted(present):
            valid, cell_reasons, validation = cell_validity(by_key[key])
            if not valid:
                failed_cells.append({
                    "cell": list(key), "reasons": cell_reasons,
                    "operation_failures": (validation or {}).get("operation_failures"),
                    "operation_failures_by_name": [
                        {"operation": op.get("name"), "failures": int(op.get("failures", 0))}
                        for op in (validation or {}).get("operations", [])
                        if int(op.get("failures", 0)) > 0
                    ],
                })
        reasons.extend(failed_cells)
        starts = [int(r.get("start_epoch_ns", 0)) for r in rows if int(r.get("start_epoch_ns", 0)) > 0]
        audit.append({
            "session": session,
            "first_start_epoch_ns": min(starts) if starts else None,
            "complete_cell_count": len(present),
            "expected_cell_count": len(EXPECTED_CELLS),
            "invalid_cell_count": len(reasons),
            "transaction_failure_cell_count": sum(
                bool(cell.get("operation_failures")) for cell in failed_cells
            ),
            "valid_under_predeclared_rule": present == EXPECTED_CELLS and not reasons,
            "invalid_cells": reasons,
        })
    return sorted(audit, key=lambda row: (
        row["first_start_epoch_ns"] is None,
        row["first_start_epoch_ns"] if row["first_start_epoch_ns"] is not None else 0,
        row["session"],
    ))


def _filter_lines(path: Path, session_ids: set[int], jsonl: bool) -> list[str]:
    if not path.is_file():
        raise ValueError(f"missing analysis manifest: {path}")
    selected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if jsonl:
            if int(json.loads(line)["session"]) in session_ids:
                selected.append(line)
        else:
            match = re.search(r"(?:^|\s)session=(\d+)(?:\s|$)", line)
            if match is None:
                raise ValueError(f"cannot parse session from order row: {line}")
            if int(match.group(1)) in session_ids:
                selected.append(line)
    return selected


def write_analysis_view(
    records: dict[tuple[str, int, str, int], dict],
    roots: list[Path],
    session_ids: tuple[int, ...],
    destination: Path,
    *,
    include_failed_target_session: int | None = None,
) -> None:
    selected_sessions = set(session_ids)
    selected_records = [row for key, row in records.items() if key[1] in selected_sessions]
    expected = set(itertools.product(CONFIGS, session_ids, PHASES, LOADS))
    selected_keys = {
        (row["config"], int(row["session"]), row["phase"], int(row["scale"]))
        for row in selected_records
    }
    if selected_keys != expected:
        raise ValueError(f"analysis view incomplete: missing={sorted(expected - selected_keys)}")
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(selected_records, key=lambda x: (
            int(x["session"]), x["config"], x["phase"], int(x["scale"])
        )):
            output_row = {key: value for key, value in row.items() if not key.startswith("_")}
            if include_failed_target_session == int(row["session"]):
                source_validation = Path(row["summary_validation_json"])
                validation = json.loads(source_validation.read_text(encoding="utf-8"))
                validation["throughput_usable"] = True
                copied = destination / f"sensitivity-{row['config']}-session{row['session']}-{row['phase']}-scale{row['scale']}-validation.json"
                copied.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
                output_row["summary_validation_json"] = str(copied.resolve())
            stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    for filename, is_jsonl in (
        ("randomized-order.txt", False),
        ("phase-times.jsonl", True),
        ("cell-times.jsonl", True),
    ):
        lines = []
        for root in roots:
            lines.extend(_filter_lines(root / filename, selected_sessions, is_jsonl))
        (destination / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def one_failure_sensitivity_targets(
    records: dict[tuple[str, int, str, int], dict], primary_sessions: set[int]
) -> list[dict[str, Any]]:
    targets = []
    for session in sorted({key[1] for key in records} - primary_sessions):
        rows = [row for key, row in records.items() if key[1] == session]
        by_key = {(r["config"], r["phase"], int(r["scale"])): r for r in rows}
        if set(by_key) != EXPECTED_CELLS:
            continue
        bad = []
        for key, row in by_key.items():
            valid, reasons, validation = cell_validity(row)
            if not valid:
                bad.append((key, reasons, validation))
        if len(bad) != 1:
            continue
        key, reasons, validation = bad[0]
        if validation is None or int(validation.get("operation_failures", 0)) != 1:
            continue
        failed_ops = [
            {"operation": op.get("name"), "failures": int(op.get("failures", 0))}
            for op in validation.get("operations", []) if int(op.get("failures", 0)) > 0
        ]
        if failed_ops != [{"operation": "Register", "failures": 1}]:
            continue
        other_invalid = []
        for other_key, other_row in by_key.items():
            if other_key == key:
                continue
            valid, other_reasons, _ = cell_validity(other_row)
            if not valid:
                other_invalid.append({"cell": list(other_key), "reasons": other_reasons})
        if other_invalid:
            continue
        targets.append({
            "session": session,
            "cell": list(key),
            "operation_failures": 1,
            "operation_successes": int(validation.get("operation_successes", 0)),
            "attempted_operations": int(validation.get("operation_successes", 0))
                + int(validation.get("operation_failures", 0)),
            "operation_failures_by_name": failed_ops,
            "http_status_context": "HTTP 400, as recorded in incident_session3_register.md and retained client log",
            "predeclared_exclusion_reasons": reasons,
        })
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True,
                        help="Original 80-cell directory followed by independent supplemental result dirs")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true",
                        help="Write the candidate-session quality audit without running model analysis")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.input_dir]
    records, by_session = read_inputs(roots)
    audit = session_audit(by_session)
    eligible = [row for row in audit if row["valid_under_predeclared_rule"]]
    if args.audit_only:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "schema": "moebench.cloudsuite.session_quality_audit.v1",
            "analysis_complete": False,
            "reason": "audit_only_no_model_analysis",
            "required_primary_session_count": 4,
            "eligible_session_ids": [row["session"] for row in eligible],
            "remaining_eligible_sessions_needed": max(0, 4 - len(eligible)),
            "selection_rule": "Among observed session IDs, select by earliest first-cell start timestamp the first four sessions for which both configs have all five probe and five full-reference cells, each structurally valid, mix-valid, zero transaction failures, and throughput-usable. No prediction error is inspected during selection.",
            "cell_validity_rule": "summary_valid AND mix_valid AND operation_failures == 0 AND throughput_usable AND client exit code 0 AND no client protocol error; latency SLO outcomes do not decide session eligibility.",
            "input_manifests": [
                {"path": str(root / "runs.jsonl"), "sha256": sha256(root / "runs.jsonl")}
                for root in roots
            ],
            "candidate_session_audit_in_chronological_order": audit,
        }
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "analysis_complete": False,
            "eligible_session_ids": output["eligible_session_ids"],
            "remaining_eligible_sessions_needed": output["remaining_eligible_sessions_needed"],
            "output": str(output_path),
        }, indent=2))
        return
    if len(eligible) < 4:
        raise SystemExit(f"Need four complete zero-failure paired sessions; found {len(eligible)}")
    primary = eligible[:4]
    primary_ids = tuple(row["session"] for row in primary)
    analyzer = load_analyzer()
    output_path = args.output_json.resolve()

    with tempfile.TemporaryDirectory(prefix="cloudsuite-analysis-") as temporary:
        primary_root = Path(temporary) / "primary"
        write_analysis_view(records, roots, primary_ids, primary_root)
        primary_result = analyzer.analyze(primary_root, primary_ids)
        if not primary_result.get("analysis_complete"):
            raise SystemExit(f"strict primary analysis failed: {primary_result.get('reason')}")

        sensitivity_targets = one_failure_sensitivity_targets(records, set(primary_ids))
        sensitivities = []
        for target in sensitivity_targets:
            session = int(target["session"])
            sensitivity_ids = primary_ids + (session,)
            sensitivity_root = Path(temporary) / f"sensitivity-{session}"
            write_analysis_view(
                records, roots, sensitivity_ids, sensitivity_root,
                include_failed_target_session=session,
            )
            result = analyzer.analyze(sensitivity_root, sensitivity_ids)
            if not result.get("analysis_complete"):
                raise SystemExit(f"sensitivity analysis failed for session {session}: {result.get('reason')}")
            target_rows = [row for row in result["fold_rows"] if row["heldout_session"] == session]
            target_decisions = {}
            for summary_key, summary in result["summary"].items():
                decision = next((row for row in summary["decisions"] if row["session"] == session), None)
                if decision is not None:
                    target_decisions[summary_key] = decision
            sensitivities.append({
                **target,
                "analysis_scope": "post hoc target-session holdout; train only on the four strict primary sessions; report only target fold",
                "included_throughput_from_failed_cell": True,
                "curve_rows": target_rows,
                "configuration_decisions": target_decisions,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema": "moebench.cloudsuite.supplemented_sessions_analysis.v1",
        "selection_rule": "Among observed session IDs, select by earliest first-cell start timestamp the first four sessions for which both configs have all five probe and five full-reference cells, each structurally valid, mix-valid, zero transaction failures, and throughput-usable. No prediction error is inspected during selection.",
        "cell_validity_rule": "summary_valid AND mix_valid AND operation_failures == 0 AND throughput_usable AND client exit code 0 AND no client protocol error; latency SLO outcomes do not decide session eligibility.",
        "input_manifests": [
            {"path": str(root / "runs.jsonl"), "sha256": sha256(root / "runs.jsonl")}
            for root in roots
        ],
        "candidate_session_audit_in_chronological_order": audit,
        "primary_sessions": list(primary_ids),
        "primary_analysis": primary_result,
        "failed_session_sensitivity": {
            "rule": "For each complete excluded session with exactly one otherwise-valid cell containing exactly one Register transaction failure, evaluate that session once as held out, train only on the four strict primary sessions, include the recorded failed-cell throughput, and do not add the sensitivity fold to the strict primary estimate.",
            "sessions": sensitivities,
        },
        "excluded_session_3_reason": next((row for row in audit if row["session"] == 3), None),
        "preservation": "Input run manifests and cell artifacts are read-only; the sensitivity override is written only into a temporary analysis view and never changes source validation JSON.",
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary_sessions": list(primary_ids),
        "sensitivity_sessions": [row["session"] for row in sensitivities],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
