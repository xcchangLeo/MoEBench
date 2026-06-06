#!/usr/bin/env python3
"""Capture CPU / memory waveforms for UnixBench: full vs router vs probe vs BenchScout."""

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
from moebench.dataset_machines import machine_config_label, machine_experiments_dir, resolve_training_machine
from moebench.ml_venv import ensure_ml_interpreter
from moebench.monitoring.plot_waveforms import (
    MODE_DISPLAY_LABELS,
    plot_waveform_compare_pair,
    plot_waveform_grid,
    plot_waveform_overlay,
)
from moebench.monitoring.resource_monitor import ResourceMonitor, trace_dict
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES

SCHEMA_COMPARE = "moebench.experiment.unixbench_resource_waveforms.v2"

MODE_ALIASES = {
    "router_only": "route_a",
    "probe_only": "route_b",
    "hybrid": "benchscout",
    "moebench": "benchscout",
    "benchscout": "benchscout",
}

DEFAULT_MODES = "full,route_a,route_b,benchscout"
DEFAULT_PAPER_WAVEFORM_DIR = REPO_ROOT / "paper" / "waveforms"


def _cli_flag(flag: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1].strip()
        if a.startswith(flag + "="):
            return a.split("=", 1)[1].strip()
    return ""


def _normalize_mode(mode: str) -> str:
    m = mode.strip()
    return MODE_ALIASES.get(m, m)


def _ml_modules_for_args(args: argparse.Namespace) -> list[str]:
    mods: list[str] = ["numpy"]
    modes = {_normalize_mode(m) for m in args.modes.split(",") if m.strip()}
    if "route_a" in modes or "benchscout" in modes:
        router = args.router_model.strip()
        if router:
            suf = Path(router).suffix.lower()
            if suf in (".pkl", ".pickle", ".dat"):
                mods.append("lightgbm")
            elif suf == ".pt":
                mods.append("torch")
    if "route_b" in modes or "benchscout" in modes:
        probe = args.probe_model.strip()
        if probe:
            name = Path(probe).name.lower()
            mods.append("xgboost" if "xgb" in name else "lightgbm")
    if "benchscout" in modes:
        recon = args.recon_model.strip()
        if recon:
            name = Path(recon).name.lower()
            if "xgb" in name:
                mods.append("xgboost")
            elif recon:
                mods.append("lightgbm")
    return list(dict.fromkeys(mods))


def _early_ml_modules() -> list[str]:
    class _Args:
        modes = _cli_flag("--modes") or DEFAULT_MODES
        router_model = _cli_flag("--router-model")
        probe_model = _cli_flag("--probe-model")
        recon_model = _cli_flag("--recon-model")

    return _ml_modules_for_args(_Args())  # type: ignore[arg-type]


ensure_ml_interpreter(
    need_modules=_early_ml_modules(),
    auto_install="--auto-install" in sys.argv,
    label="waveform",
)

from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs
from moebench.hybrid.eval import probe_predictions_to_executed_tests, router_select_test_ids


def _load_router_meta(model_fp: Path, *, auto_install: bool = False) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        try:
            with open(model_fp, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else ""
            if missing and auto_install:
                from moebench.pip_install import ensure_importable

                ensure_importable(missing, auto_install=True)
                with open(model_fp, "rb") as f:
                    return pickle.load(f)
            raise
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
    tr["label"] = MODE_DISPLAY_LABELS["full"]
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
    router_meta = _load_router_meta(router_model, auto_install=False)
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
    tr["label"] = MODE_DISPLAY_LABELS["route_a"]
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
    tr["label"] = MODE_DISPLAY_LABELS["route_b"]
    tr["mode"] = "route_b"
    tr["probe_model"] = str(probe_model.resolve())
    tr["probe_duration_s"] = dur
    tr["probe_mode"] = mode
    tr["test_ids"] = tids
    tr["phase_markers"] = phase_markers
    return tr


def capture_benchscout(
    *,
    router_model: Path,
    recon_model: Path,
    probe_model: Path,
    copies: int,
    interval_s: float,
    top_k: int | None,
    skip_xi: bool,
    warmup_s: float,
    probe_duration_s: float | None,
    probe_mode: str,
    enable_ebpf: bool,
) -> dict[str, Any]:
    """BenchScout: xi → router Top-K → probe selected subtests → recon (full hybrid path)."""
    del copies  # hybrid uses probes, not partial UnixBench
    router_meta = _load_router_meta(router_model, auto_install=False)
    recon_bundle = load_reconstruction_bundle(recon_model)
    probe_bundle = load_probe_bundle(probe_model)
    k = int(top_k if top_k is not None else router_meta.get("top_k", 3))
    dur = float(probe_duration_s if probe_duration_s is not None else probe_bundle.get("probe_duration_s", 4.0))
    mode = probe_mode or str(probe_bundle.get("probe_mode", "micro"))
    benchmark = str(probe_bundle.get("benchmark") or recon_bundle.get("benchmark") or "unixbench")

    mon = ResourceMonitor(interval_s=interval_s)
    mon.start()
    phase_markers: list[dict[str, Any]] = []
    router_detail: dict[str, Any] = {}
    selected_test_ids: list[str] = []
    predicted_suite: float | None = None

    try:
        xi: dict[str, Any] = {}
        if not skip_xi:
            phase_markers.append({"name": "xi", "t_rel_s": mon.elapsed_s()})
            xi = collect_all(warmup_s=warmup_s, proc_sample_s=0.5, enable_ebpf=False, mem_mb=64)

        phase_markers.append({"name": "router", "t_rel_s": mon.elapsed_s()})
        _, selected_test_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=k)

        sub_preds: dict[str, float] = {}
        probe_walls: dict[str, float] = {}
        for tid in selected_test_ids:
            phase_markers.append({"name": tid, "t_rel_s": mon.elapsed_s()})
            probe = collect_subtest_probe(
                tid,
                duration_s=dur,
                enable_ebpf=enable_ebpf,
                benchmark=benchmark,
                probe_mode=mode,
            )
            sub_preds[tid] = predict_subtest(probe_bundle, probe, tid)
            probe_walls[tid] = float(probe.get("wall_s") or dur)

        phase_markers.append({"name": "recon", "t_rel_s": mon.elapsed_s()})
        executed = probe_predictions_to_executed_tests(
            selected_test_ids,
            sub_preds,
            dur,
            benchmark=benchmark,
            probe_wall_by_tid=probe_walls,
        )
        pred = predict_from_partial(recon_bundle, xi, executed)
        predicted_suite = float(pred["suite_index"])
    finally:
        tr = mon.stop()

    tr["label"] = MODE_DISPLAY_LABELS["benchscout"]
    tr["mode"] = "benchscout"
    tr["router_model"] = str(router_model.resolve())
    tr["recon_model"] = str(recon_model.resolve())
    tr["probe_model"] = str(probe_model.resolve())
    tr["top_k"] = k
    tr["selected_test_ids"] = selected_test_ids
    tr["router"] = router_detail
    tr["probe_duration_s"] = dur
    tr["probe_mode"] = mode
    tr["predicted_suite_index"] = predicted_suite
    tr["phase_markers"] = phase_markers
    return tr


def _plot_title_suffix(machine: str) -> str:
    label = machine_config_label(machine)
    return f" ({label})" if label else ""


def _paper_waveform_basenames(machine: str) -> tuple[str, str]:
    stem = f"{machine}_waveform"
    return f"{stem}_cpu.png", f"{stem}_memory.png"


def _emit_plots(
    traces: list[dict[str, Any]],
    *,
    out_dir: Path,
    machine: str,
    auto_install: bool,
    paper_dir: Path | None = None,
) -> dict[str, str]:
    suffix = _plot_title_suffix(machine)
    title_cpu = f"UnixBench CPU usage{suffix}"
    title_mem = f"UnixBench memory usage{suffix}"

    pair = plot_waveform_compare_pair(
        traces,
        out_dir=out_dir,
        title_cpu=title_cpu,
        title_mem=title_mem,
        auto_install=auto_install,
    )
    grid_png = plot_waveform_grid(
        traces,
        out_path=out_dir / "resource_waveforms_grid.png",
        title=f"UnixBench CPU / Memory{suffix}",
        auto_install=auto_install,
    )
    overlay_png = plot_waveform_overlay(
        traces,
        out_path=out_dir / "resource_waveforms_overlay.png",
        title=f"UnixBench resource usage overlay{suffix}",
        auto_install=auto_install,
    )

    plots: dict[str, str] = {
        "cpu": str(pair["cpu"]),
        "memory": str(pair["memory"]),
        "grid": str(grid_png),
        "overlay": str(overlay_png),
    }

    if paper_dir is not None and machine:
        paper_dir.mkdir(parents=True, exist_ok=True)
        cpu_name, mem_name = _paper_waveform_basenames(machine)
        paper_pair = plot_waveform_compare_pair(
            traces,
            out_dir=paper_dir,
            title_cpu=title_cpu,
            title_mem=title_mem,
            auto_install=auto_install,
            cpu_basename=cpu_name,
            memory_basename=mem_name,
        )
        plots["paper_cpu"] = str(paper_pair["cpu"])
        plots["paper_memory"] = str(paper_pair["memory"])

    return plots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        type=str,
        default=DEFAULT_MODES,
        help="Comma-separated: full, route_a|router_only, route_b|probe_only, benchscout|hybrid",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--unixbench-root", type=str, default="")
    ap.add_argument("--interval-s", type=float, default=0.5, help="Sample interval (seconds)")
    ap.add_argument("--copies", type=int, default=0, help=f"UnixBench -c; 0 = {UNIXBENCH_PARALLEL_COPIES}")
    ap.add_argument("--router-model", type=str, default="", help="Router checkpoint (route_a / benchscout)")
    ap.add_argument("--recon-model", type=str, default="", help="Reconstructor bundle (benchscout)")
    ap.add_argument("--probe-model", type=str, default="", help="Probe bundle (route_b / benchscout)")
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
    ap.add_argument("--auto-install", action="store_true", help="Auto-install matplotlib / ML deps if missing")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--paper-dir",
        type=str,
        default=str(DEFAULT_PAPER_WAVEFORM_DIR),
        help="Write paper figures as {machine}_waveform_cpu.png / _memory.png (empty to skip)",
    )
    ap.add_argument(
        "--plot-from",
        type=str,
        default="",
        help="Only plot from existing resource_waveforms.json (skip capture)",
    )
    args = ap.parse_args()

    paper_dir: Path | None = None
    if args.paper_dir.strip():
        paper_dir = Path(args.paper_dir).resolve()
        if not paper_dir.is_absolute():
            paper_dir = (REPO_ROOT / paper_dir).resolve()

    if args.plot_from.strip():
        with open(args.plot_from, encoding="utf-8") as f:
            report = json.load(f)
        traces = list(report.get("traces") or [])
        out_dir = Path(args.plot_from).resolve().parent
        machine = str(report.get("machine") or resolve_training_machine(args.machine or None))
        if not traces:
            print("No traces in report", file=sys.stderr)
            return 2
        plots = _emit_plots(
            traces,
            out_dir=out_dir,
            machine=machine,
            auto_install=args.auto_install,
            paper_dir=paper_dir,
        )
        print(json.dumps(plots, indent=2))
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

    modes = [_normalize_mode(m) for m in args.modes.split(",") if m.strip()]
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
        elif mode == "benchscout":
            if not args.router_model.strip():
                print("--router-model required for benchscout", file=sys.stderr)
                return 2
            if not args.recon_model.strip():
                print("--recon-model required for benchscout", file=sys.stderr)
                return 2
            if not args.probe_model.strip():
                print("--probe-model required for benchscout", file=sys.stderr)
                return 2
            tr = capture_benchscout(
                router_model=Path(args.router_model).resolve(),
                recon_model=Path(args.recon_model).resolve(),
                probe_model=Path(args.probe_model).resolve(),
                copies=copies,
                interval_s=args.interval_s,
                top_k=args.top_k,
                skip_xi=args.skip_xi,
                warmup_s=args.warmup_s,
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
        plots = _emit_plots(
            traces,
            out_dir=out_dir,
            machine=machine,
            auto_install=args.auto_install,
            paper_dir=paper_dir,
        )
        report["plots"] = plots
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        for key, path in plots.items():
            print(f"Wrote {path}", file=sys.stderr)

    print(json.dumps({"report": str(report_path), "n_traces": len(traces)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
