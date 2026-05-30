#!/usr/bin/env python3
"""End-to-end probe experiment: short probes per subtest → predict suite vs ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    ensure_machine_output_dir,
    machine_experiments_dir,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.ml_venv import ensure_ml_interpreter


def _ml_modules_for_probe(model_path: str) -> list[str]:
    name = Path(model_path).name.lower()
    backend = "xgboost" if "xgb" in name else "lightgbm"
    return ["numpy", "sklearn", backend]


def _early_ml_modules() -> list[str]:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--probe-model" and i + 1 < len(argv):
            return _ml_modules_for_probe(argv[i + 1])
        if a.startswith("--probe-model="):
            return _ml_modules_for_probe(a.split("=", 1)[1])
    return ["numpy", "sklearn", "lightgbm"]


ensure_ml_interpreter(
    need_modules=_early_ml_modules(),
    auto_install="--auto-install" in sys.argv,
    label="probe_experiment",
)

from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.probe.suite_aggregate import aggregate_suite_index, suite_error_report
from moebench.probe.training_data import label_suite_from_pts_run, label_suite_from_unixbench_run
from moebench.reconstruct.data import collect_unixbench_run_paths
from moebench.phoronix.training_data import canonical_test_ids_from_runs, collect_phoronix_run_paths


def _ground_truth_unixbench(dataset_root: Path, machine: str) -> tuple[float | None, str | None]:
    glo = resolve_glob_for_machine(benchmark="unixbench", machine=machine, glob_pattern=None, pts_suite=None)
    paths = collect_unixbench_run_paths(dataset_root, glob_pattern=glo)
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        ds = json.load(f)
    return label_suite_from_unixbench_run(ds), str(latest)


def _ground_truth_pts(
    dataset_root: Path,
    machine: str,
    pts_suite: str,
) -> tuple[float | None, str | None, list[str]]:
    glo = resolve_glob_for_machine(
        benchmark="phoronix",
        machine=machine,
        glob_pattern=None,
        pts_suite=pts_suite,
    )
    paths = collect_phoronix_run_paths(dataset_root, glob_pattern=glo, pts_suite=pts_suite)
    if not paths:
        return None, None, []
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        ds = json.load(f)
    tids = list(canonical_test_ids_from_runs([latest]))
    suite = label_suite_from_pts_run(ds, tids)
    return suite, str(latest), tids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-model", type=str, required=True, help="Trained probe bundle .pkl")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument(
        "--probe-mode",
        type=str,
        choices=("micro", "real"),
        default="",
        help="Override bundle default (micro or real subtest run)",
    )
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--auto-install", action="store_true", help="Bootstrap project ML venv if deps missing")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    machine = resolve_training_machine(args.machine or None)
    bundle = load_probe_bundle(args.probe_model)
    benchmark = str(bundle.get("benchmark", "unixbench"))
    pts_suite = bundle.get("pts_suite")
    tids = list(bundle.get("test_ids") or [])
    dur = float(args.probe_duration_s if args.probe_duration_s is not None else bundle.get("probe_duration_s", 4.0))
    mode = args.probe_mode or bundle.get("probe_mode", "micro")
    agg = str(bundle.get("suite_aggregate", "geomean_index"))

    sub_preds: dict[str, float] = {}
    probes: dict[str, Any] = {}
    for tid in tids:
        probe = collect_subtest_probe(
            tid,
            duration_s=dur,
            enable_ebpf=not args.no_ebpf,
            benchmark=benchmark,
            probe_mode=mode,
            pts_title=None,
        )
        probes[tid] = probe
        sub_preds[tid] = predict_subtest(bundle, probe, tid)

    suite_pred = aggregate_suite_index(sub_preds, mode=agg)

    gt_suite: float | None = None
    gt_path: str | None = None
    if benchmark == "unixbench":
        gt_suite, gt_path = _ground_truth_unixbench(Path(args.dataset_root), machine)
    else:
        if not pts_suite:
            print("PTS bundle missing pts_suite", file=sys.stderr)
            return 2
        gt_suite, gt_path, _ = _ground_truth_pts(Path(args.dataset_root), machine, str(pts_suite))

    report: dict = {
        "schema": "moebench.probe.experiment.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "benchmark": benchmark,
        "pts_suite": pts_suite,
        "probe_model": str(Path(args.probe_model).resolve()),
        "probe_duration_s": dur,
        "probe_mode": mode,
        "suite_aggregate": agg,
        "predicted_subtest": sub_preds,
        "predicted_suite": suite_pred,
        "ground_truth_run": gt_path,
        "ground_truth_suite": gt_suite,
        "estimated_probe_wall_s": dur * len(tids),
    }
    if gt_suite is not None:
        report["suite_comparison"] = suite_error_report(float(suite_pred), float(gt_suite))

    if args.output.strip():
        out = Path(args.output).resolve()
    else:
        tok = "unixbench" if benchmark == "unixbench" else safe_session_tag(str(pts_suite).replace("/", "_"))
        out = (
            machine_experiments_dir(args.dataset_root, machine)
            / f"probe_{tok}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )

    ensure_machine_output_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


def safe_session_tag(tag: str) -> str:
    from moebench.phoronix.pipeline import safe_session_tag as _s

    return _s(tag)


if __name__ == "__main__":
    raise SystemExit(main())
