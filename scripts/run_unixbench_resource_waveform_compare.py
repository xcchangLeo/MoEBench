#!/usr/bin/env python3
"""Capture CPU / memory waveforms for UnixBench: full vs route A vs route B."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench import collect_all
from moebench.dataset_machines import machine_experiments_dir, resolve_training_machine
from moebench.monitoring.plot_waveforms import plot_waveform_grid, plot_waveform_overlay
from moebench.monitoring.resource_monitor import ResourceMonitor, trace_dict
from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES

SCHEMA_COMPARE = "moebench.experiment.unixbench_resource_waveforms.v1"


def _load_router_meta(model_fp: Path) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        with open(model_fp, "rb") as f:
            return pickle.load(f)
    import torch

    try:
        return torch.load(model_fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_fp, map_location="cpu")


def _run_ub(
    unixbench_root: Path,
    result_dir: Path,
    base_name: str,
    test_ids: list[str] | None,
    copies: int,
) -> float:
    run_script = unixbench_root / "Run"
    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)
    cmd = ["perl", str(run_script), "-c", str(copies)]
    if test_ids:
        cmd.extend(test_ids)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(unixbench_root), env=env)
    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"UnixBench Run failed rc={rc}")
    return wall


def capture_full(
    *,
    unixbench_root: Path,
    result_dir: Path,
    session_tag: str,
    copies: int,
    interval_s: float,
    test_ids: list[str] | None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"moebench_wave_full_{session_tag}_{stamp}".replace(":", "-")
    mon = ResourceMonitor(interval_s=interval_s)

    def _work() -> None:
        _run_ub(unixbench_root, result_dir, base, test_ids, copies)

    tr = mon.run(_work)
    tr["label"] = "Full UnixBench"
    tr["mode"] = "full"
    tr["unixbench_report"] = str(result_dir / base)
    tr["test_ids"] = test_ids or list(INDEX_SUITE_TEST_IDS)
    return tr


def capture_route_a(
    *,
    unixbench_root: Path,
    result_dir: Path,
    session_tag: str,
    router_model: Path,
    copies: int,
    interval_s: float,
    top_k: int | None,
    skip_xi: bool,
    warmup_s: float,
    test_ids: list[str] | None,
) -> dict[str, Any]:
    router_meta = _load_router_meta(router_model)
    k = int(top_k if top_k is not None else router_meta.get("top_k", 3))

    xi: dict[str, Any] = {}
    if not skip_xi:
        xi = collect_all(warmup_s=warmup_s, proc_sample_s=0.5, enable_ebpf=False, mem_mb=64)

    _scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    _selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, k)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"moebench_wave_route_a_{session_tag}_{stamp}".replace(":", "-")
    mon = ResourceMonitor(interval_s=interval_s)

    def _work() -> None:
        _run_ub(unixbench_root, result_dir, base, selected_test_ids, copies)

    tr = mon.run(_work)
    tr["label"] = f"Route A (Top-{k} partial UB)"
    tr["mode"] = "route_a"
    tr["router_model"] = str(router_model.resolve())
    tr["top_k"] = k
    tr["selected_test_ids"] = selected_test_ids
    tr["unixbench_report"] = str(result_dir / base)
    if test_ids:
        tr["note"] = "quick mode: router still used; partial list may differ from test_ids filter"
    return tr


def capture_route_b(
    *,
    probe_model: Path,
    interval_s: float,
    probe_duration_s: float | None,
    probe_mode: str,
    enable_ebpf: bool,
) -> dict[str, Any]:
    bundle = load_probe_bundle(probe_model)
    tids = list(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    dur = float(probe_duration_s if probe_duration_s is not None else bundle.get("probe_duration_s", 4.0))
    mode = probe_mode or str(bundle.get("probe_mode", "micro"))
    benchmark = str(bundle.get("benchmark", "unixbench"))

    mon = ResourceMonitor(interval_s=interval_s)
    mon.start()
    phase_markers: list[dict[str, Any]] = []
    try:
        for tid in tids:
            phase_markers.append({"name": tid, "t_rel_s": mon.elapsed_s()})
            collect_subtest_probe(
                tid,
                duration_s=dur,
                enable_ebpf=enable_ebpf,
                benchmark=benchmark,
                probe_mode=mode,
            )
    finally:
        tr = mon.stop()
    tr["label"] = f"Route B (probe {mode}, {dur:.0f}s/subtest)"
    tr["mode"] = "route_b"
    tr["probe_model"] = str(probe_model.resolve())
    tr["probe_duration_s"] = dur
    tr["probe_mode"] = mode
    tr["test_ids"] = tids
    tr["phase_markers"] = phase_markers
    return tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        type=str,
        default="full,route_a,route_b",
        help="Comma-separated: full, route_a, route_b",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--unixbench-root", type=str, default="")
    ap.add_argument("--interval-s", type=float, default=0.5, help="Sample interval (seconds)")
    ap.add_argument("--copies", type=int, default=0, help=f"UnixBench -c; 0 = {UNIXBENCH_PARALLEL_COPIES}")
    ap.add_argument("--router-model", type=str, default="", help="Route A router checkpoint")
    ap.add_argument("--probe-model", type=str, default="", help="Route B probe .pkl")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--skip-xi", action="store_true", help="Route A: skip collect_all before routing")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument("--probe-mode", type=str, default="", choices=("", "micro", "real"))
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only run dhry2reg+whetstone-double (fast smoke; not for paper figures)",
    )
    ap.add_argument("-o", "--output-dir", type=str, default="")
    ap.add_argument("--auto-install", action="store_true", help="pip install matplotlib if missing")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--plot-from",
        type=str,
        default="",
        help="Only plot from existing resource_waveforms.json (skip capture)",
    )
    args = ap.parse_args()

    if args.plot_from.strip():
        with open(args.plot_from, encoding="utf-8") as f:
            report = json.load(f)
        traces = list(report.get("traces") or [])
        out_dir = Path(args.plot_from).resolve().parent
        if not traces:
            print("No traces in report", file=sys.stderr)
            return 2
        grid_png = plot_waveform_grid(
            traces,
            out_path=out_dir / "resource_waveforms_grid.png",
            title="UnixBench CPU / Memory (Full vs Route A vs Route B)",
            auto_install=args.auto_install,
        )
        overlay_png = plot_waveform_overlay(
            traces,
            out_path=out_dir / "resource_waveforms_overlay.png",
            title="UnixBench resource usage overlay",
            auto_install=args.auto_install,
        )
        print(json.dumps({"grid": str(grid_png), "overlay": str(overlay_png)}, indent=2))
        return 0

    machine = resolve_training_machine(args.machine or None)
    repo = REPO_ROOT
    unixbench_root = Path(args.unixbench_root).resolve() if args.unixbench_root else repo / "byte-unixbench" / "UnixBench"
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    copies = args.copies if args.copies > 0 else UNIXBENCH_PARALLEL_COPIES

    if args.output_dir.strip():
        out_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = machine_experiments_dir(args.dataset_root, machine) / f"ub_resource_waveforms_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    host = os.uname().nodename.split(".")[0]
    session_tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in machine or host)

    quick_ids = ["dhry2reg", "whetstone-double"] if args.quick else None
    if args.quick:
        print("WARNING: --quick uses only 2 subtests; omit for paper-quality full waveforms.", file=sys.stderr)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    traces: list[dict[str, Any]] = []

    for mode in modes:
        print(f"=== Capturing {mode} ===", file=sys.stderr)
        if mode == "full":
            tr = capture_full(
                unixbench_root=unixbench_root,
                result_dir=result_dir,
                session_tag=session_tag,
                copies=copies,
                interval_s=args.interval_s,
                test_ids=quick_ids,
            )
        elif mode == "route_a":
            if not args.router_model.strip():
                print("--router-model required for route_a", file=sys.stderr)
                return 2
            tr = capture_route_a(
                unixbench_root=unixbench_root,
                result_dir=result_dir,
                session_tag=session_tag,
                router_model=Path(args.router_model).resolve(),
                copies=copies,
                interval_s=args.interval_s,
                top_k=args.top_k,
                skip_xi=args.skip_xi,
                warmup_s=args.warmup_s,
                test_ids=quick_ids,
            )
        elif mode == "route_b":
            if not args.probe_model.strip():
                print("--probe-model required for route_b", file=sys.stderr)
                return 2
            tr = capture_route_b(
                probe_model=Path(args.probe_model).resolve(),
                interval_s=args.interval_s,
                probe_duration_s=args.probe_duration_s,
                probe_mode=args.probe_mode,
                enable_ebpf=not args.no_ebpf,
            )
        else:
            print(f"Unknown mode {mode!r}", file=sys.stderr)
            return 2
        trace_path = out_dir / f"trace_{mode}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(tr, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {trace_path}", file=sys.stderr)
        traces.append(tr)

    report: dict[str, Any] = {
        "schema": SCHEMA_COMPARE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "interval_s": args.interval_s,
        "quick_mode": args.quick,
        "traces": traces,
    }
    report_path = out_dir / "resource_waveforms.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.no_plot and traces:
        grid_png = plot_waveform_grid(
            traces,
            out_path=out_dir / "resource_waveforms_grid.png",
            title="UnixBench CPU / Memory (Full vs Route A vs Route B)",
            auto_install=args.auto_install,
        )
        overlay_png = plot_waveform_overlay(
            traces,
            out_path=out_dir / "resource_waveforms_overlay.png",
            title="UnixBench resource usage overlay",
            auto_install=args.auto_install,
        )
        report["plots"] = {"grid": str(grid_png), "overlay": str(overlay_png)}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {grid_png}", file=sys.stderr)
        print(f"Wrote {overlay_png}", file=sys.stderr)

    print(json.dumps({"report": str(report_path), "n_traces": len(traces)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
