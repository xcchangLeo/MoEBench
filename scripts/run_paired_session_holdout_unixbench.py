#!/usr/bin/env python3
"""Run leave-one-session-out policy evaluation on verified paired UnixBench data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_sessions(dataset_root: Path, machine: str) -> list[dict[str, Any]]:
    sessions = []
    for manifest_path in sorted(dataset_root.glob("*/paired_capture_manifest.json")):
        session = manifest_path.parent.name
        if not session.startswith(machine + "_"):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("probe_timing") != "before_full_run":
            continue
        if manifest.get("capture_status") not in (None, "complete"):
            continue
        run_record = manifest.get("run") or {}
        run_path = Path(str(run_record.get("path") or ""))
        probe_record = manifest.get("probe_dataset") or {}
        probe_path = Path(str(probe_record.get("path") or ""))
        if not run_path.is_file() or not probe_path.is_file():
            continue
        if run_record.get("sha256") and sha256(run_path) != run_record["sha256"]:
            raise ValueError(f"Run checksum mismatch: {run_path}")
        if probe_record.get("sha256") and sha256(probe_path) != probe_record["sha256"]:
            raise ValueError(f"Probe checksum mismatch: {probe_path}")
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        test_ids = list(probe.get("test_ids") or [])
        samples = list(probe.get("samples") or [])
        source_runs = {str(row.get("source_run")) for row in samples}
        sample_ids = {str(row.get("test_id")) for row in samples}
        if len(samples) != len(test_ids) or sample_ids != set(test_ids):
            raise ValueError(f"Incomplete probe labels: {probe_path}")
        if source_runs != {str(run_path)}:
            raise ValueError(f"Probe labels do not point to the paired run: {probe_path}")
        sessions.append(
            {
                "session": session,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "run_path": str(run_path.resolve()),
                "probe_path": str(probe_path.resolve()),
                "suite_index": float(
                    json.loads(run_path.read_text(encoding="utf-8"))["yi"]["runs"][0]["system_benchmarks_index_score"]
                ),
                "load_before": manifest.get("load_before"),
                "xi_elapsed_s": manifest.get("xi_elapsed_s"),
                "probe_wall_sum_s": manifest.get("probe_wall_sum_s"),
                "full_run_elapsed_s": manifest.get("full_run_elapsed_s"),
            }
        )
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--sessions", default="", help="Comma-separated paired sessions to include")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-model", choices=("lightgbm", "ridge"), default="lightgbm")
    parser.add_argument("--reconstructor", choices=("lightgbm", "lightgbm_small", "xgboost", "ridge"), default="lightgbm")
    parser.add_argument(
        "--partial-source",
        choices=("probe_predictions", "heldout_observed"),
        default="probe_predictions",
    )
    parser.add_argument(
        "--policies",
        default="learned,learned_anchor,learned_corr_weighted,learned_probe_quality,probe_anchor_top_k,train_assoc_top_k,train_assoc_diverse_top_k,random_top_k,fixed_first_k",
    )
    parser.add_argument(
        "--suite-output-mode",
        choices=(
            "direct",
            "observed_plus_imputed_geomean",
            "observed_plus_training_geomean",
            "observed_plus_scaled_imputed_geomean",
            "probe_plus_imputed_geomean",
            "reconstructed_geomean",
            "probe_scaled_geomean",
        ),
        default=None,
    )
    parser.add_argument("--log1p-partial-index", action="store_true")
    args = parser.parse_args()

    if args.suite_output_mode is None:
        args.suite_output_mode = (
            "observed_plus_imputed_geomean"
            if args.partial_source == "heldout_observed"
            else "probe_plus_imputed_geomean"
        )
    if (
        args.partial_source == "heldout_observed"
        and args.suite_output_mode == "probe_plus_imputed_geomean"
    ):
        raise SystemExit(
            f"{args.suite_output_mode} calibrates predicted probe values with training-only "
            "with heldout_observed exact subtest scores"
        )

    dataset_root = args.dataset_root.resolve()
    out_dir = args.output_dir.resolve()
    sessions = complete_sessions(dataset_root, args.machine)
    if args.sessions.strip():
        requested = {item.strip() for item in args.sessions.split(",") if item.strip()}
        available = {item["session"] for item in sessions}
        missing = sorted(requested - available)
        if missing:
            raise SystemExit(f"Requested paired sessions are unavailable: {missing}")
        sessions = [item for item in sessions if item["session"] in requested]
    if len(sessions) < args.min_sessions:
        raise SystemExit(
            f"Need at least {args.min_sessions} verified paired sessions for session holdout; found {len(sessions)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dataset = out_dir / "paired_probe_dataset.json"
    builder = ROOT / "scripts" / "build_unixbench_paired_probe_set.py"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--paired-only",
            "--paired-root",
            str(dataset_root),
            "--output",
            str(probe_dataset),
        ],
        check=True,
    )

    session_names = [item["session"] for item in sessions]
    run_script = ROOT / "scripts" / "run_hybrid_oof_unixbench_policies.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for fold_index, heldout in enumerate(session_names):
        fold_dir = out_dir / f"holdout_{heldout}"
        log_path = fold_dir / "run.log"
        fold_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(run_script),
            "--dataset-root",
            str(dataset_root),
            "--machine",
            args.machine,
            "--sessions",
            ",".join(session_names),
            "--holdout-session",
            heldout,
            "--probe-dataset",
            str(probe_dataset),
            "--policies",
            args.policies,
            "--probe-model",
            args.probe_model,
            "--reconstructor",
            args.reconstructor,
            "--seed",
            str(args.seed + fold_index),
            "--top-k",
            str(args.top_k),
            "--suite-output-mode",
            args.suite_output_mode,
            "--partial-source",
            args.partial_source,
            "--out-dir",
            str(fold_dir),
        ]
        if args.log1p_partial_index:
            command.append("--log1p-partial-index")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Holdout failed for {heldout}; see {log_path}")
        print(f"Completed paired session holdout {fold_index + 1}/{len(sessions)}: {heldout}", file=sys.stderr)

    policy_errors: dict[str, list[float]] = {}
    per_session: dict[str, dict[str, Any]] = {}
    for item in sessions:
        fold_dir = out_dir / f"holdout_{item['session']}"
        index_path = fold_dir / f"{args.machine}__index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        per_session[item["session"]] = {}
        for policy, details in index.get("policies", {}).items():
            summary = details["summary"]
            per_session[item["session"]][policy] = summary
            if summary.get("mean_relative_error_percent") is not None:
                policy_errors.setdefault(policy, []).append(float(summary["mean_relative_error_percent"]))

    report = {
        "schema": "moebench.unixbench.paired_session_holdout.v1",
        "machine": args.machine,
        "split_protocol": "leave-one-complete-pre-run-paired-session-out; all training rows and probe labels come from other verified paired sessions",
        "top_k": args.top_k,
        "suite_output_mode": args.suite_output_mode,
        "probe_model": args.probe_model,
        "reconstructor": args.reconstructor,
        "partial_source": args.partial_source,
        "log1p_partial_index": args.log1p_partial_index,
        "policy_ids": [item.strip() for item in args.policies.split(",") if item.strip()],
        "sessions": sessions,
        "per_session_policy_summary": per_session,
        "mean_session_holdout_error_percent": {
            policy: sum(values) / len(values) for policy, values in policy_errors.items()
        },
        "paired_probe_dataset": {"path": str(probe_dataset), "sha256": sha256(probe_dataset)},
    }
    report_path = out_dir / "paired_session_holdout_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "mean_session_holdout_error_percent": report["mean_session_holdout_error_percent"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
