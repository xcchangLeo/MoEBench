#!/usr/bin/env python3
"""Capture CPU / memory waveforms for UnixBench: full vs router vs probe vs BenchScout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    find_probe_checkpoint,
    find_router_recon_checkpoint,
    machine_config_label,
    machine_experiments_dir,
    machine_models_dir,
    resolve_checkpoint_file,
    resolve_training_machine,
)
from moebench.ml_venv import ensure_ml_interpreter

SCHEMA_COMPARE = "moebench.experiment.unixbench_resource_waveforms.v3"

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


def _merge_traces_from_out_dir(
    traces: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Include trace_<mode>.json from out_dir when a mode is missing from traces."""
    by_mode: dict[str, dict[str, Any]] = {}
    for tr in traces:
        mode = str(tr.get("mode") or "")
        if mode:
            by_mode[mode] = tr

    for mode in MODE_ORDER:
        if mode in by_mode:
            continue
        trace_path = out_dir / f"trace_{mode}.json"
        if not trace_path.is_file():
            continue
        with open(trace_path, encoding="utf-8") as f:
            by_mode[mode] = json.load(f)
        print(f"[waveform] merged existing {trace_path.name}", file=sys.stderr)

    order = {m: i for i, m in enumerate(MODE_ORDER)}
    return sorted(by_mode.values(), key=lambda t: order.get(str(t.get("mode", "")), 99))


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

    mods = _ml_modules_for_args(_Args())  # type: ignore[arg-type]
    if "--no-plot" not in sys.argv:
        mods.append("matplotlib")
    return list(dict.fromkeys(mods))


ensure_ml_interpreter(
    need_modules=_early_ml_modules(),
    auto_install="--auto-install" in sys.argv,
    label="waveform",
)

from moebench.monitoring.plot_waveforms import (
    MODE_DISPLAY_LABELS,
    MODE_ORDER,
    plot_waveform_compare_pair,
    plot_waveform_grid,
    plot_waveform_overlay,
)
from moebench.monitoring.waveform_capture import (
    capture_benchscout,
    capture_full,
    capture_route_a,
    capture_route_b,
    default_copies,
)
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS


def _resolve_waveform_models(
    args: argparse.Namespace,
    *,
    repo: Path,
    machine: str,
    modes: list[str],
) -> tuple[Path | None, Path | None, Path | None]:
    ds_root = Path(args.dataset_root).resolve()
    if not ds_root.is_absolute():
        ds_root = (repo / ds_root).resolve()

    router_fp: Path | None = None
    recon_fp: Path | None = None
    probe_fp: Path | None = None

    if "route_a" in modes or "benchscout" in modes:
        router_name = Path(args.router_model.strip()).name if args.router_model.strip() else "router_lgbm.pkl"
        router_fallback = find_router_recon_checkpoint(
            ds_root,
            machine=machine,
            benchmark="unixbench",
            filename=router_name,
        )
        if args.router_model.strip():
            router_fp = resolve_checkpoint_file(
                args.router_model,
                repo,
                kind="Router model",
                fallback=router_fallback,
            )
        elif router_fallback is not None:
            router_fp = router_fallback
        else:
            raise FileNotFoundError(
                "Router model required for route_a/benchscout. Pass --router-model or train via "
                "scripts/run_router_reconstruct_model_grid.py."
            )
        print(f"[waveform] router model: {router_fp}", file=sys.stderr)

    if "benchscout" in modes:
        recon_name = Path(args.recon_model.strip()).name if args.recon_model.strip() else "recon_lgbm.pkl"
        recon_fallback = find_router_recon_checkpoint(
            ds_root,
            machine=machine,
            benchmark="unixbench",
            filename=recon_name,
        )
        if args.recon_model.strip():
            recon_fp = resolve_checkpoint_file(
                args.recon_model,
                repo,
                kind="Reconstruction model",
                fallback=recon_fallback,
            )
        elif recon_fallback is not None:
            recon_fp = recon_fallback
        else:
            raise FileNotFoundError(
                "Reconstruction model required for benchscout. Pass --recon-model or train via "
                "scripts/run_router_reconstruct_model_grid.py."
            )
        print(f"[waveform] recon model: {recon_fp}", file=sys.stderr)

    if "route_b" in modes or "benchscout" in modes:
        probe_name = Path(args.probe_model.strip()).name if args.probe_model.strip() else "probe_unixbench_lgbm.pkl"
        probe_fallback = find_probe_checkpoint(ds_root, machine=machine, filename=probe_name)
        if not probe_fallback and not args.probe_model.strip():
            default_probe = machine_models_dir(ds_root, machine) / probe_name
            probe_fallback = default_probe if default_probe.is_file() else None
        if args.probe_model.strip():
            probe_fp = resolve_checkpoint_file(
                args.probe_model,
                repo,
                kind="Probe model",
                fallback=probe_fallback,
            )
        elif probe_fallback is not None:
            probe_fp = probe_fallback
        else:
            default_probe = machine_models_dir(ds_root, machine) / probe_name
            raise FileNotFoundError(
                f"Probe model required for route_b/benchscout. Pass --probe-model or train with:\n"
                f"  python3 scripts/probe_train.py --probe-dataset {ds_root / 'models' / machine / 'probe_dataset_unixbench.json'} "
                f"--model-out {default_probe}"
            )
        print(f"[waveform] probe model: {probe_fp}", file=sys.stderr)

    return router_fp, recon_fp, probe_fp


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
        traces = _merge_traces_from_out_dir(traces, out_dir)
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
        report["traces"] = traces
        report["plots"] = plots
        with open(args.plot_from, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(plots, indent=2))
        return 0

    machine = resolve_training_machine(args.machine or None)
    repo = REPO_ROOT
    unixbench_root = Path(args.unixbench_root).resolve() if args.unixbench_root else repo / "byte-unixbench" / "UnixBench"
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    copies = default_copies(args.copies)

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
    try:
        router_fp, recon_fp, probe_fp = _resolve_waveform_models(
            args,
            repo=repo,
            machine=machine,
            modes=modes,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

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
            assert router_fp is not None
            tr = capture_route_a(
                unixbench_root=unixbench_root,
                result_dir=result_dir,
                session_tag=session_tag,
                router_model=router_fp,
                copies=copies,
                interval_s=args.interval_s,
                top_k=args.top_k,
                skip_xi=args.skip_xi,
                warmup_s=args.warmup_s,
                test_ids=quick_ids,
            )
        elif mode == "route_b":
            assert probe_fp is not None
            tr = capture_route_b(
                probe_model=probe_fp,
                interval_s=args.interval_s,
                probe_duration_s=args.probe_duration_s,
                probe_mode=args.probe_mode or "micro",
            )
        elif mode == "benchscout":
            assert router_fp is not None and recon_fp is not None and probe_fp is not None
            tr = capture_benchscout(
                router_model=router_fp,
                recon_model=recon_fp,
                probe_model=probe_fp,
                interval_s=args.interval_s,
                top_k=args.top_k,
                skip_xi=args.skip_xi,
                warmup_s=args.warmup_s,
                probe_duration_s=args.probe_duration_s,
                probe_mode=args.probe_mode or "micro",
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

    traces = _merge_traces_from_out_dir(traces, out_dir)

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
