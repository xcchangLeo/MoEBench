#!/usr/bin/env python3
"""Compare probe-based suite prediction with eBPF enabled vs disabled (telemetry degradation).

Runs the same trained probe model twice on the local host:
  1) ``enable_ebpf=True``  (may require root / CAP_BPF for bpftrace)
  2) ``enable_ebpf=False`` (zero-masks eBPF dimensions; pipeline must not crash)

Outputs a side-by-side JSON report suitable for paper tables.
"""

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
    label="probe_ebpf_ablation",
)

from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.probe.suite_aggregate import aggregate_suite_index, suite_error_report
from moebench.probe.training_data import label_suite_from_pts_run, label_suite_from_unixbench_run
from moebench.phoronix.training_data import canonical_test_ids_from_runs, collect_phoronix_run_paths
from moebench.reconstruct.data import collect_unixbench_run_paths

SCHEMA = "moebench.experiment.probe_ebpf_ablation.v1"


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
) -> tuple[float | None, str | None]:
    glo = resolve_glob_for_machine(
        benchmark="phoronix",
        machine=machine,
        glob_pattern=None,
        pts_suite=pts_suite,
    )
    paths = collect_phoronix_run_paths(dataset_root, glob_pattern=glo, pts_suite=pts_suite)
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        ds = json.load(f)
    tids = list(canonical_test_ids_from_runs([latest]))
    return label_suite_from_pts_run(ds, tids), str(latest)


def _default_probe_model(dataset_root: Path, machine: str, benchmark: str, pts_suite: str | None) -> Path:
    models_dir = dataset_root / "models" / machine
    if benchmark == "unixbench":
        for name in ("probe_unixbench_lgbm.pkl", "probe_unixbench_xgb.pkl"):
            p = models_dir / name
            if p.is_file():
                return p
    else:
        stem = "probe_pts_cpu" if pts_suite == "cpu" else "probe_pts_gpu"
        for suffix in ("_lgbm.pkl", "_xgb.pkl"):
            p = models_dir / f"{stem}{suffix}"
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"No probe model under {models_dir}. Train first, e.g.:\n"
        f"  python3 scripts/probe_train.py \\\n"
        f"    --probe-dataset {models_dir}/probe_dataset_unixbench.json \\\n"
        f"    --model-out {models_dir}/probe_unixbench_lgbm.pkl --auto-install"
    )


def _summarize_ebpf(probes: dict[str, Any]) -> dict[str, Any]:
    n = len(probes)
    if n == 0:
        return {"n_subtests": 0, "ebpf_available_count": 0, "ebpf_available_fraction": 0.0}
    avail = 0
    sched_rates: list[float] = []
    syscall_rates: list[float] = []
    reasons: dict[str, int] = {}
    for probe in probes.values():
        ebpf = probe.get("ebpf") or {}
        if ebpf.get("available"):
            avail += 1
        else:
            reason = str(ebpf.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        sched_rates.append(float(ebpf.get("sched_switch_per_s") or 0.0))
        syscall_rates.append(float(ebpf.get("syscall_enter_per_s") or 0.0))
    return {
        "n_subtests": n,
        "ebpf_available_count": avail,
        "ebpf_available_fraction": avail / n,
        "mean_sched_switch_per_s": sum(sched_rates) / n,
        "mean_syscall_enter_per_s": sum(syscall_rates) / n,
        "unavailable_reasons": reasons,
    }


def _run_condition(
    *,
    label: str,
    enable_ebpf: bool,
    bundle: dict[str, Any],
    tids: list[str],
    dur: float,
    mode: str,
    benchmark: str,
    pts_title: str | None,
    agg: str,
    gt_suite: float | None,
) -> dict[str, Any]:
    sub_preds: dict[str, float] = {}
    probes: dict[str, Any] = {}
    errors: list[str] = []

    for tid in tids:
        try:
            probe = collect_subtest_probe(
                tid,
                duration_s=dur,
                enable_ebpf=enable_ebpf,
                benchmark=benchmark,
                probe_mode=mode,
                pts_title=pts_title,
            )
            probes[tid] = probe
            sub_preds[tid] = predict_subtest(bundle, probe, tid)
        except Exception as e:
            errors.append(f"{tid}: {e!s}")

    wall_s = sum(float(p.get("wall_s") or dur) for p in probes.values())
    suite_pred = aggregate_suite_index(sub_preds, mode=agg) if sub_preds else None

    out: dict[str, Any] = {
        "condition": label,
        "enable_ebpf": enable_ebpf,
        "n_subtests_requested": len(tids),
        "n_subtests_completed": len(probes),
        "n_subtests_predicted": len(sub_preds),
        "probe_wall_s_sum": wall_s,
        "predicted_subtest": sub_preds,
        "predicted_suite": suite_pred,
        "telemetry": _summarize_ebpf(probes),
        "errors": errors,
        "pipeline_ok": len(errors) == 0 and len(sub_preds) == len(tids),
    }
    if gt_suite is not None and suite_pred is not None:
        out["suite_comparison"] = suite_error_report(float(suite_pred), float(gt_suite))
    return out


def _compare_conditions(on: dict[str, Any], off: dict[str, Any]) -> dict[str, Any]:
    def _rel_err(row: dict[str, Any]) -> float | None:
        sc = row.get("suite_comparison") or {}
        v = sc.get("relative_error")
        if v is None:
            v = sc.get("suite_relative_error")
        return float(v) if v is not None else None

    def _rel_err_pct(row: dict[str, Any]) -> float | None:
        v = _rel_err(row)
        return (v * 100.0) if v is not None else None

    err_on = _rel_err(on)
    err_off = _rel_err(off)
    delta_pp: float | None = None
    if err_on is not None and err_off is not None:
        delta_pp = (err_off - err_on) * 100.0

    return {
        "suite_relative_error_ebpf_on": err_on,
        "suite_relative_error_ebpf_off": err_off,
        "suite_relative_error_pct_ebpf_on": _rel_err_pct(on),
        "suite_relative_error_pct_ebpf_off": _rel_err_pct(off),
        "delta_relative_error_pp_off_minus_on": delta_pp,
        "abs_error_ebpf_on": (on.get("suite_comparison") or {}).get("abs_error"),
        "abs_error_ebpf_off": (off.get("suite_comparison") or {}).get("abs_error"),
        "predicted_suite_ebpf_on": on.get("predicted_suite"),
        "predicted_suite_ebpf_off": off.get("predicted_suite"),
        "probe_wall_s_ebpf_on": on.get("probe_wall_s_sum"),
        "probe_wall_s_ebpf_off": off.get("probe_wall_s_sum"),
        "ebpf_available_fraction_on": (on.get("telemetry") or {}).get("ebpf_available_fraction"),
        "ebpf_available_fraction_off": (off.get("telemetry") or {}).get("ebpf_available_fraction"),
        "pipeline_ok_ebpf_on": on.get("pipeline_ok"),
        "pipeline_ok_ebpf_off": off.get("pipeline_ok"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-model", type=str, default="", help="Trained probe .pkl (default: auto under dataset/models/<machine>/)")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="", help="Host slug (default: local)")
    ap.add_argument(
        "--benchmark",
        type=str,
        default="",
        choices=("", "unixbench", "phoronix"),
        help="Override bundle benchmark (default: from model)",
    )
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument("--probe-mode", type=str, default="", choices=("", "micro", "real"))
    ap.add_argument(
        "--conditions",
        type=str,
        default="on,off",
        help="Comma-separated: on (eBPF enabled), off (--no-ebpf equivalent)",
    )
    ap.add_argument("--skip-on", action="store_true", help="Skip eBPF-on condition (reuse prior JSON via --reuse-on)")
    ap.add_argument("--skip-off", action="store_true", help="Skip eBPF-off condition")
    ap.add_argument("--reuse-on", type=str, default="", help="Reuse ebpf_on block from prior ablation JSON")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    machine = resolve_training_machine(args.machine or None)
    ds_root = Path(args.dataset_root).resolve()

    if args.probe_model.strip():
        probe_path = Path(args.probe_model).resolve()
    else:
        probe_path = _default_probe_model(ds_root, machine, "unixbench", None)

    if not probe_path.is_file():
        print(f"Probe model not found: {probe_path}", file=sys.stderr)
        return 2

    bundle = load_probe_bundle(probe_path)
    benchmark = args.benchmark or str(bundle.get("benchmark", "unixbench"))
    pts_suite = bundle.get("pts_suite")
    tids = list(bundle.get("test_ids") or [])
    dur = float(args.probe_duration_s if args.probe_duration_s is not None else bundle.get("probe_duration_s", 4.0))
    mode = args.probe_mode or str(bundle.get("probe_mode", "micro"))
    agg = str(bundle.get("suite_aggregate", "geomean_index"))

    gt_suite: float | None = None
    gt_path: str | None = None
    if benchmark == "unixbench":
        gt_suite, gt_path = _ground_truth_unixbench(ds_root, machine)
    else:
        if not pts_suite:
            print("PTS bundle missing pts_suite", file=sys.stderr)
            return 2
        gt_suite, gt_path = _ground_truth_pts(ds_root, machine, str(pts_suite))

    if gt_suite is None:
        print("No ground-truth full run found for this machine/benchmark", file=sys.stderr)
        return 2

    conds = {c.strip().lower() for c in args.conditions.split(",") if c.strip()}
    results: dict[str, Any] = {}

    if args.reuse_on.strip():
        with open(args.reuse_on, encoding="utf-8") as f:
            prior = json.load(f)
        block = (prior.get("conditions") or {}).get("ebpf_on")
        if not block:
            print("--reuse-on JSON missing conditions.ebpf_on", file=sys.stderr)
            return 2
        results["ebpf_on"] = block
    elif "on" in conds and not args.skip_on:
        print("=== Condition: eBPF ON ===", file=sys.stderr)
        results["ebpf_on"] = _run_condition(
            label="ebpf_on",
            enable_ebpf=True,
            bundle=bundle,
            tids=tids,
            dur=dur,
            mode=mode,
            benchmark=benchmark,
            pts_title=None,
            agg=agg,
            gt_suite=gt_suite,
        )

    if "off" in conds and not args.skip_off:
        print("=== Condition: eBPF OFF ===", file=sys.stderr)
        results["ebpf_off"] = _run_condition(
            label="ebpf_off",
            enable_ebpf=False,
            bundle=bundle,
            tids=tids,
            dur=dur,
            mode=mode,
            benchmark=benchmark,
            pts_title=None,
            agg=agg,
            gt_suite=gt_suite,
        )

    if "ebpf_on" not in results or "ebpf_off" not in results:
        print("Need both ebpf_on and ebpf_off results for comparison", file=sys.stderr)
        return 2

    comparison = _compare_conditions(results["ebpf_on"], results["ebpf_off"])

    tok = "unixbench" if benchmark == "unixbench" else str(pts_suite or "pts").replace("/", "_")
    if args.output.strip():
        out_path = Path(args.output).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = machine_experiments_dir(str(ds_root), machine) / f"probe_ebpf_ablation_{tok}_{stamp}.json"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "benchmark": benchmark,
        "pts_suite": pts_suite,
        "probe_model": str(probe_path),
        "probe_duration_s": dur,
        "probe_mode": mode,
        "ground_truth_run": gt_path,
        "ground_truth_suite": gt_suite,
        "conditions": results,
        "comparison": comparison,
    }

    ensure_machine_output_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({"output": str(out_path), "comparison": comparison}, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
