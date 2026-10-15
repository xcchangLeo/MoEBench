#!/usr/bin/env python3
"""Score locked fpm8 throughput predictions and retain per-cell p99 details."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

LOADS = (20, 40, 60, 80, 100)
SESSIONS = (101, 102, 103, 104)
METHODS = ("correlation", "history_scale", "linear_interpolation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def curve_mape(prediction: list[float], actual: list[float]) -> float:
    if len(prediction) != 5 or len(actual) != 5 or any(not math.isfinite(x) or x <= 0 for x in actual):
        raise ValueError("invalid five-load curve")
    if any(not math.isfinite(x) or x <= 0 for x in prediction):
        raise ValueError("invalid predicted throughput")
    return sum(abs(p - y) / y for p, y in zip(prediction, actual)) * 20.0


def p99_cell(validation: dict) -> dict:
    rows = validation.get("response_times", [])
    row = next((item for item in rows if item.get("name") == "BrowsetoElgg"), None)
    if row is None:
        return {"p99_ms": None, "p99_lower_bound_ms": None, "p99_censored": None,
                "successful_samples": None, "availability": "missing"}
    exact = row.get("p99_ms")
    lower = row.get("p99_lower_bound_ms")
    censored = row.get("p99_censored")
    return {"p99_ms": exact, "p99_lower_bound_ms": lower, "p99_censored": censored,
            "successful_samples": row.get("samples"),
            "availability": "censored" if censored else ("exact" if exact is not None else "missing")}


def verify_target_cell(result_dir: Path, validation_path: Path, expected_validation_hash: str | None = None,
                       expected_xml_hash: str | None = None) -> dict:
    if not validation_path.is_file():
        raise ValueError(f"missing validation: {validation_path.name}")
    if expected_validation_hash and sha256(validation_path) != expected_validation_hash:
        raise ValueError(f"validation hash mismatch: {validation_path.name}")
    validation = load_json(validation_path)
    summary = Path(validation.get("summary_xml", ""))
    if not summary.is_file() or not summary.is_relative_to(result_dir.resolve()):
        raise ValueError(f"summary XML missing or outside result directory: {validation_path.name}")
    recorded_hash = result_dir / f"{validation_path.name.removesuffix('-validation.json')}-summary.sha256"
    if not recorded_hash.is_file() or not recorded_hash.read_text(encoding="utf-8").split():
        raise ValueError(f"missing summary hash record: {recorded_hash.name}")
    actual_hash = sha256(summary)
    expected_recorded = recorded_hash.read_text(encoding="utf-8").split()[0]
    if actual_hash != expected_recorded or (expected_xml_hash and actual_hash != expected_xml_hash):
        raise ValueError(f"summary XML hash mismatch: {validation_path.name}")
    return validation


def score(result_dir: Path, manifest: Path) -> dict:
    frozen = load_json(manifest)
    errors = []
    sessions = []
    descriptive = []
    locked_predictions = {}
    phase_path = result_dir / "phase-times.jsonl"
    phase_rows = {}
    if phase_path.is_file():
        for line in phase_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["session"]), row["phase"])
            if key in phase_rows:
                errors.append(f"duplicate phase timing: {key}")
            phase_rows[key] = row
    else:
        errors.append("missing phase timing record")
    timing_path = result_dir / "cell-times.jsonl"
    cell_times = {}
    if timing_path.is_file():
        for line in timing_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["session"]), row["phase"], int(row["load"]))
            if key in cell_times:
                errors.append(f"duplicate cell timing: {key}")
            cell_times[key] = row
    else:
        errors.append("missing cell timing record")
    for session in SESSIONS:
        phases = [phase_rows.get((session, phase)) for phase in ("probe", "prediction_lock", "full")]
        if any(row is None for row in phases):
            errors.append(f"missing probe, prediction lock, or full phase timing for session {session}")
        else:
            probe_phase, lock_phase, full_phase = phases
            if not (int(probe_phase["start_epoch_ns"]) < int(probe_phase["end_epoch_ns"])
                    < int(lock_phase["start_epoch_ns"]) < int(lock_phase["end_epoch_ns"])
                    < int(full_phase["start_epoch_ns"]) < int(full_phase["end_epoch_ns"])):
                errors.append(f"probe/prediction/reference phase order invalid for session {session}")
        prediction_path = result_dir / f"fpm8-session{session}-prediction.json"
        prediction_sum = result_dir / f"fpm8-session{session}-prediction.json.sha256"
        provenance_path = result_dir / f"fpm8-session{session}-prediction-provenance.json"
        provenance_sum = result_dir / f"fpm8-session{session}-prediction-provenance.json.sha256"
        required_paths = (prediction_path, prediction_sum, provenance_path, provenance_sum)
        if any(not path.is_file() for path in required_paths):
            errors.append(f"prediction lock incomplete for session {session}")
            continue
        expected_prediction = prediction_sum.read_text(encoding="utf-8").split()[0]
        expected_provenance = provenance_sum.read_text(encoding="utf-8").split()[0]
        lock_phase = phase_rows.get((session, "prediction_lock"))
        if lock_phase and (lock_phase.get("prediction_sha256") != expected_prediction or
                           lock_phase.get("provenance_sha256") != expected_provenance or
                           lock_phase.get("status") != "locked"):
            errors.append(f"phase timing and prediction lock disagree for session {session}")
            continue
        if sha256(prediction_path) != expected_prediction or sha256(provenance_path) != expected_provenance:
            errors.append(f"prediction lock hash mismatch for session {session}")
            continue
        prediction, provenance = load_json(prediction_path), load_json(provenance_path)
        if prediction.get("manifest_sha256") != sha256(manifest) or prediction.get("target_session") != session or \
           provenance.get("manifest_sha256") != sha256(manifest) or provenance.get("target_session") != session or \
           provenance.get("prediction_sha256") != expected_prediction or \
           provenance.get("model_id") != prediction.get("model_id"):
            errors.append(f"prediction provenance mismatch for session {session}")
            continue
        prediction_probes = {int(row["scale"]): row for row in prediction.get("target_probe_cells", [])}
        provenance_probes = {int(row["scale"]): row for row in provenance.get("probe_cells", [])}
        selected = set(frozen["selected_loads"])
        if set(prediction_probes) != selected or set(provenance_probes) != selected or any(
            prediction_probes[load].get("validation_sha256") != provenance_probes[load].get("validation_sha256") or
            prediction_probes[load].get("summary_sha256") != provenance_probes[load].get("summary_sha256")
            for load in selected
        ):
            errors.append(f"probe hashes disagree between prediction and provenance for session {session}")
            continue
        locked_predictions[session] = prediction
    if len(locked_predictions) != len(SESSIONS):
        return {"analysis_complete": False, "primary_endpoint": "five-load throughput curve MAPE",
                "sessions": [], "summary": None, "browsetoelgg_p99_by_cell": [],
                "errors": errors,
                "interpretation_note": "Scoring starts only after all four session predictions and provenance hashes are locked."}
    for session in SESSIONS:
        session_errors = False
        prediction = locked_predictions[session]
        probes = prediction.get("target_probe_cells", [])
        if len(probes) != 3 or {int(row["scale"]) for row in probes} != set(frozen["selected_loads"]):
            errors.append(f"prediction probe provenance incomplete for session {session}")
            continue
        probe_hashes = {int(row["scale"]): row for row in probes}
        for load in frozen["selected_loads"]:
            probe_path = result_dir / f"fpm8-session{session}-probe-scale{load}-validation.json"
            try:
                validation = verify_target_cell(result_dir, probe_path,
                                                probe_hashes[load]["validation_sha256"],
                                                probe_hashes[load]["summary_sha256"])
                descriptive.append({"session": session, "phase": "probe", "load": load,
                                    **p99_cell(validation)})
            except (KeyError, ValueError) as exc:
                errors.append(f"invalid locked probe cell session={session}, load={load}: {exc}")
                session_errors = True
        actual = []
        for load in LOADS:
            full_path = result_dir / f"fpm8-session{session}-full-scale{load}-validation.json"
            if not full_path.is_file():
                errors.append(f"missing full-reference validation: {full_path.name}")
                session_errors = True
                continue
            try:
                cell = verify_target_cell(result_dir, full_path)
            except ValueError as exc:
                errors.append(f"invalid full-reference cell session={session}, load={load}: {exc}")
                session_errors = True
                continue
            descriptive.append({"session": session, "phase": "full", "load": load, **p99_cell(cell)})
            if not cell.get("summary_valid") or not cell.get("throughput_usable"):
                errors.append(f"unusable full-reference cell: session={session}, load={load}")
                session_errors = True
                continue
            actual.append(float(cell["throughput_ops_per_sec"]))
        if session_errors or len(actual) != 5:
            continue
        pred_curve = prediction.get("predicted_throughput_ops_per_sec", [])
        history_curve = prediction.get("history_only_throughput_ops_per_sec", [])
        comparators = prediction.get("same_budget_comparators_ops_per_sec", {})
        if set(comparators) != set(METHODS) or \
           comparators.get(frozen["predictor_method"]) != pred_curve:
            errors.append(f"same-budget predictions are missing or disagree for session {session}")
            continue
        try:
            probe_error = curve_mape(pred_curve, actual)
            history_error = curve_mape(history_curve, actual)
            comparator_errors = {method: curve_mape(comparators[method], actual)
                                 for method in METHODS}
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid prediction curve for session {session}: {exc}")
            continue
        selected = frozen["selected_loads"]
        cost_keys = [(session, "probe", load) for load in selected] + [
            (session, "full", load) for load in LOADS]
        if any(key not in cell_times for key in cost_keys):
            errors.append(f"missing selected-probe or full-reference cost for session {session}")
            continue
        if any(cell_times[key].get("status") != "complete" for key in cost_keys):
            errors.append(f"failed selected-probe or full-reference cell for session {session}")
            continue
        costs = {}
        for phase, loads in (("probe", selected), ("full", LOADS)):
            costs[phase] = {
                "client_execution_seconds": sum(float(cell_times[(session, phase, load)]["runner_seconds"])
                                                for load in loads),
                "setup_seconds": sum(float(cell_times[(session, phase, load)]["setup_seconds"])
                                     for load in loads),
                "cell_total_seconds": sum(float(cell_times[(session, phase, load)]["cell_total_seconds"])
                                          for load in loads),
            }
        sessions.append({"session": session, "probe_mape_percent": probe_error,
                         "history_only_mape_percent": history_error,
                         "same_budget_comparator_mape_percent": comparator_errors,
                         "paired_difference_probe_minus_history_percentage_points": probe_error - history_error,
                         "predicted_throughput_ops_per_sec": pred_curve,
                         "history_only_throughput_ops_per_sec": history_curve,
                         "reference_throughput_ops_per_sec": actual,
                         "prediction_lock_seconds": float(phase_rows[(session, "prediction_lock")]["elapsed_seconds"]),
                         "measured_cost_seconds": costs})
    complete = len(sessions) == 4 and not errors
    differences = [row["paired_difference_probe_minus_history_percentage_points"] for row in sessions]
    summary = None
    if complete:
        summary = {"independent_unit": "target session", "n_sessions": 4,
                   "probe_mean_mape_percent": statistics.mean(row["probe_mape_percent"] for row in sessions),
                   "history_only_mean_mape_percent": statistics.mean(row["history_only_mape_percent"] for row in sessions),
                   "same_budget_comparator_mean_mape_percent": {
                       method: statistics.mean(row["same_budget_comparator_mape_percent"][method]
                                               for row in sessions)
                       for method in METHODS},
                   "mean_prediction_lock_seconds": statistics.mean(
                       row["prediction_lock_seconds"] for row in sessions),
                   "mean_measured_cost_seconds": {
                       phase: {key: statistics.mean(row["measured_cost_seconds"][phase][key]
                                                    for row in sessions)
                               for key in ("client_execution_seconds", "setup_seconds", "cell_total_seconds")}
                       for phase in ("probe", "full")},
                   "mean_paired_difference_percentage_points": statistics.mean(differences),
                   "sd_paired_difference_percentage_points": statistics.stdev(differences),
                   "paired_differences_percentage_points": differences}
    return {"analysis_complete": complete, "primary_endpoint": "five-load throughput curve MAPE",
            "sessions": sessions, "summary": summary, "browsetoelgg_p99_by_cell": descriptive,
            "errors": errors,
            "interpretation_note": "Four sessions are the independent units; p99 is descriptive only, and censored values remain lower bounds."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.result_dir, args.manifest)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_complete": result["analysis_complete"],
                      "errors": result["errors"]}, sort_keys=True))
    return 0 if result["analysis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
