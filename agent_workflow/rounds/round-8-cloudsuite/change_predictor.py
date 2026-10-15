#!/usr/bin/env python3
"""Freeze an fpm4-only curve predictor and lock predictions before fpm8 references."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from analyze_faban_profiles import (
    LOADS,
    curve_error,
    predict,
    predict_history_scale,
    predict_linear_interpolation,
)

SOURCE_SESSIONS = (1, 2, 3, 4)
TARGET_SESSIONS = (101, 102, 103, 104)
METHODS = ("correlation", "history_scale", "linear_interpolation")
MODEL_VERSION = "fpm4-looso-curve-v1"
PREPROCESSING_VERSION = "faban-throughput-ops-per-sec-v1"
HYPERPARAMETERS = {
    "correlation_diagonal_regularization": 0.01,
    "correlation_standard_deviation_floor": 1e-9,
    "prediction_floor_ops_per_sec": 1e-9,
    "history_scale_fit": "least_squares_through_origin",
    "linear_interpolation_endpoint_rule": "linear_extrapolation",
    "subset_size": 3,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"locked output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o444)


def source_records(root: Path) -> tuple[np.ndarray, list[dict]]:
    curves = []
    records = []
    for session in SOURCE_SESSIONS:
        values = []
        for scale in LOADS:
            key = f"fpm4-session{session}-full-scale{scale}"
            validation_path = root / f"{key}-validation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if not validation.get("summary_valid") or not validation.get("mix_valid") or \
               int(validation.get("operation_failures", -1)) != 0 or \
               not validation.get("throughput_usable"):
                raise ValueError(f"invalid source cell: {key}")
            xml_path = Path(validation["summary_xml"])
            if not xml_path.is_file() or not xml_path.is_relative_to(root.resolve()):
                raise ValueError(f"source XML missing or outside source root: {key}")
            values.append(float(validation["throughput_ops_per_sec"]))
            records.append({"session": session, "scale": scale,
                            "validation_path": str(validation_path.resolve()),
                            "validation_sha256": digest(validation_path),
                            "summary_path": str(xml_path.resolve()),
                            "summary_sha256": digest(xml_path)})
        curves.append(values)
    return np.asarray(curves, dtype=float), records


def estimate(method: str, probe: np.ndarray, subset: tuple[int, ...],
             train: np.ndarray) -> np.ndarray:
    if method == "correlation":
        return predict(probe, subset, train)
    if method == "history_scale":
        return predict_history_scale(probe, subset, train.mean(axis=0))
    if method == "linear_interpolation":
        return predict_linear_interpolation(probe, subset)
    raise ValueError(f"unsupported predictor: {method}")


def freeze(args: argparse.Namespace) -> None:
    root = args.source_root.resolve()
    full_curves, sources = source_records(root)
    probe_curves = []
    probe_sources = []
    for session in SOURCE_SESSIONS:
        values = []
        for scale in LOADS:
            key = f"fpm4-session{session}-probe-scale{scale}"
            path = root / f"{key}-validation.json"
            validation = json.loads(path.read_text(encoding="utf-8"))
            if not validation.get("summary_valid") or not validation.get("mix_valid") or \
               int(validation.get("operation_failures", -1)) != 0 or \
               not validation.get("throughput_usable"):
                raise ValueError(f"invalid source probe: {key}")
            xml_path = Path(validation["summary_xml"])
            if not xml_path.is_file() or not xml_path.is_relative_to(root):
                raise ValueError(f"source probe XML missing or outside source root: {key}")
            values.append(float(validation["throughput_ops_per_sec"]))
            probe_sources.append({"session": session, "scale": scale,
                                  "validation_path": str(path.resolve()),
                                  "validation_sha256": digest(path),
                                  "summary_path": str(xml_path.resolve()),
                                  "summary_sha256": digest(xml_path)})
        probe_curves.append(values)
    probe_curves = np.asarray(probe_curves, dtype=float)
    candidates = []
    for method, subset in itertools.product(METHODS, itertools.combinations(range(5), 3)):
        fold_errors = []
        for heldout in range(len(SOURCE_SESSIONS)):
            training = np.delete(full_curves, heldout, axis=0)
            estimated = estimate(method, probe_curves[heldout], subset, training)
            fold_errors.append(curve_error(estimated, full_curves[heldout]))
        candidates.append({"method": method, "loads": [LOADS[i] for i in subset],
                           "fpm4_session_errors_percent": fold_errors,
                           "fpm4_mean_error_percent": float(np.mean(fold_errors))})
    chosen = min(candidates, key=lambda row: (row["fpm4_mean_error_percent"],
                                              row["method"], row["loads"]))
    script_path = Path(__file__).resolve()
    analyzer_path = script_path.with_name("analyze_faban_profiles.py")
    manifest = {"protocol": "fpm4-source-to-new-fpm8-target-v1",
                "model_id": f"{MODEL_VERSION}/{chosen['method']}/"
                            + "-".join(str(load) for load in chosen["loads"]),
                "model_version": MODEL_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "hyperparameters": HYPERPARAMETERS,
                "source_root": str(root), "source_sessions": list(SOURCE_SESSIONS),
                "source_full_cells": sources, "source_probe_cells": probe_sources,
                "target_config": "fpm8", "target_sessions": list(TARGET_SESSIONS),
                "probe_seconds": 60, "reference_seconds": 120,
                "load_levels": list(LOADS), "selected_loads": chosen["loads"],
                "predictor_method": chosen["method"],
                "selection_rule": "minimum mean fpm4 leave-one-session-out curve MAPE over 30 candidates",
                "selection_scores": candidates,
                "predictor_source_sha256": digest(script_path),
                "analyzer_source_sha256": digest(analyzer_path),
                "numpy_version": np.__version__, "randomization_seed": 20271013}
    atomic_json(args.output_manifest, manifest)
    print(json.dumps({"manifest": str(args.output_manifest), "sha256": digest(args.output_manifest),
                      "selected_loads": chosen["loads"], "method": chosen["method"]}))


def verify_source(manifest: dict) -> np.ndarray:
    if digest(Path(__file__).resolve()) != manifest["predictor_source_sha256"]:
        raise ValueError("predictor source changed after freeze")
    if digest(Path(__file__).with_name("analyze_faban_profiles.py")) != manifest["analyzer_source_sha256"]:
        raise ValueError("analyzer source changed after freeze")
    if np.__version__ != manifest["numpy_version"]:
        raise ValueError("numpy version differs from frozen environment")
    for record in manifest["source_full_cells"] + manifest["source_probe_cells"]:
        if digest(Path(record["validation_path"])) != record["validation_sha256"] or \
           digest(Path(record["summary_path"])) != record["summary_sha256"]:
            raise ValueError(f"source cell changed after freeze: {record['session']}/{record['scale']}")
        validation = json.loads(Path(record["validation_path"]).read_text(encoding="utf-8"))
        if not validation.get("summary_valid") or not validation.get("mix_valid") or \
           int(validation.get("operation_failures", -1)) != 0 or \
           not validation.get("throughput_usable"):
            raise ValueError(f"source cell is not strict-valid: {record['session']}/{record['scale']}")
    curves = []
    for session in manifest["source_sessions"]:
        curve = []
        for scale in LOADS:
            record = next(row for row in manifest["source_full_cells"]
                          if row["session"] == session and row["scale"] == scale)
            validation = json.loads(Path(record["validation_path"]).read_text(encoding="utf-8"))
            curve.append(float(validation["throughput_ops_per_sec"]))
        curves.append(curve)
    return np.asarray(curves, dtype=float)


def source_keys(records: list[dict]) -> set[tuple[int, int]]:
    keys = [(int(row["session"]), int(row["scale"])) for row in records]
    expected = set(itertools.product(SOURCE_SESSIONS, LOADS))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("source records do not cover exactly four sessions and five loads")
    return set(keys)


def check_manifest(args: argparse.Namespace) -> None:
    if digest(args.manifest) != args.expected_sha256:
        raise ValueError("frozen manifest digest mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "fpm4-source-to-new-fpm8-target-v1" or \
       manifest.get("model_version") != MODEL_VERSION or \
       manifest.get("preprocessing_version") != PREPROCESSING_VERSION or \
       manifest.get("hyperparameters") != HYPERPARAMETERS or \
       manifest.get("target_config") != "fpm8" or \
       manifest.get("target_sessions") != list(TARGET_SESSIONS) or \
       manifest.get("source_sessions") != list(SOURCE_SESSIONS) or \
       manifest.get("probe_seconds") != 60 or manifest.get("reference_seconds") != 120 or \
       manifest.get("load_levels") != list(LOADS) or \
       manifest.get("predictor_method") not in METHODS or \
       len(manifest.get("selected_loads", [])) != 3 or \
       len(set(manifest["selected_loads"])) != 3 or \
       any(load not in LOADS for load in manifest["selected_loads"]):
        raise ValueError("invalid frozen target protocol or subset")
    expected_model_id = f"{MODEL_VERSION}/{manifest['predictor_method']}/" + "-".join(
        str(load) for load in manifest["selected_loads"])
    if manifest.get("model_id") != expected_model_id:
        raise ValueError("frozen model ID does not match method and selected loads")
    if any("fpm8" in row["validation_path"] or "fpm8" in row["summary_path"]
           for row in manifest["source_full_cells"] + manifest["source_probe_cells"]):
        raise ValueError("fpm8 target observations appear in source manifest")
    source_keys(manifest["source_full_cells"])
    source_keys(manifest["source_probe_cells"])
    verify_source(manifest)
    print(" ".join(str(load) for load in manifest["selected_loads"]))


def predict_target(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol"] != "fpm4-source-to-new-fpm8-target-v1" or \
       args.session not in manifest["target_sessions"]:
        raise ValueError("manifest protocol or target session mismatch")
    source = verify_source(manifest)
    subset = tuple(LOADS.index(load) for load in manifest["selected_loads"])
    probe = np.full(len(LOADS), np.nan)
    probe_records = []
    if len(args.probe_validation) != 3:
        raise ValueError("exactly three target probe records are required")
    for path in args.probe_validation:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("summary_valid") or not data.get("mix_valid") or \
           int(data.get("operation_failures", -1)) != 0 or \
           not data.get("throughput_usable"):
            raise ValueError(f"invalid target probe: {path}")
        # The scale is taken from the runner's cell name, not a result-dependent choice.
        name = path.name
        prefix = f"fpm8-session{args.session}-probe-scale"
        if not name.startswith(prefix) or not name.endswith("-validation.json"):
            raise ValueError(f"target probe identity mismatch: {name}")
        scale = int(name[len(prefix):-len("-validation.json")])
        if scale not in manifest["selected_loads"] or not np.isnan(probe[LOADS.index(scale)]):
            raise ValueError(f"unexpected or repeated target load: {scale}")
        probe[LOADS.index(scale)] = float(data["throughput_ops_per_sec"])
        xml_path = Path(data["summary_xml"])
        if not xml_path.is_file():
            raise ValueError(f"missing target probe XML: {xml_path}")
        probe_records.append({"scale": scale, "validation_path": str(path.resolve()),
                              "validation_sha256": digest(path),
                              "summary_path": str(xml_path.resolve()),
                              "summary_sha256": digest(xml_path)})
    if sorted(row["scale"] for row in probe_records) != sorted(manifest["selected_loads"]):
        raise ValueError("target probe set differs from frozen subset")
    estimate_curve = estimate(manifest["predictor_method"], probe, subset, source)
    history_curve = source.mean(axis=0)
    same_budget_curves = {method: estimate(method, probe, subset, source).tolist()
                          for method in METHODS}
    output = {"protocol": manifest["protocol"], "target_session": args.session,
              "target_config": "fpm8", "manifest_path": str(manifest_path),
              "manifest_sha256": digest(manifest_path),
              "model_id": manifest["model_id"],
              "selected_loads": manifest["selected_loads"],
              "predictor_method": manifest["predictor_method"],
              "target_probe_cells": probe_records,
              "predicted_throughput_ops_per_sec": estimate_curve.tolist(),
              "same_budget_comparators_ops_per_sec": same_budget_curves,
              "history_only_throughput_ops_per_sec": history_curve.tolist()}
    atomic_json(args.output_prediction, output)
    print(json.dumps({"prediction": str(args.output_prediction),
                      "sha256": digest(args.output_prediction)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("--source-root", type=Path, required=True)
    lock.add_argument("--output-manifest", type=Path, required=True)
    lock.set_defaults(function=freeze)
    check = commands.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--expected-sha256", required=True)
    check.set_defaults(function=check_manifest)
    target = commands.add_parser("predict")
    target.add_argument("--manifest", type=Path, required=True)
    target.add_argument("--session", type=int, required=True)
    target.add_argument("--probe-validation", type=Path, action="append", required=True)
    target.add_argument("--output-prediction", type=Path, required=True)
    target.set_defaults(function=predict_target)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
