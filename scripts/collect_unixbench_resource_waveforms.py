#!/usr/bin/env python3
"""Collect UnixBench CPU/memory waveform data (JSON only).

Four modes (paper labels):
  full        — complete ``perl Run -c 1``
  route_a     — router only: partial UnixBench on router Top-K
  route_b     — probe only: micro-workloads on all 12 subtests
  benchscout  — router Top-K + micro-probes + reconstruction

Outputs per-mode ``trace_<mode>.json`` plus ``resource_waveforms.json`` under
``--output-dir``. Plotting is intentionally omitted; use
``run_unixbench_resource_waveform_compare.py --plot-from`` after review.
"""

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
    machine_experiments_dir,
    machine_models_dir,
    resolve_checkpoint_file,
    resolve_training_machine,
)
from moebench.ml_venv import ensure_ml_interpreter
from moebench.monitoring.plot_waveforms import MODE_ORDER
from moebench.monitoring.waveform_capture import (
    capture_benchscout,
    capture_full,
    capture_route_a,
    capture_route_b,
    default_copies,
)

SCHEMA_COMPARE = "moebench.experiment.unixbench_resource_waveforms.v3"

MODE_ALIASES = {
    "router_only": "route_a",
    "probe_only": "route_b",
    "hybrid": "benchscout",
    "moebench": "benchscout",
    "benchscout": "benchscout",
}

DEFAULT_MODES = "full,route_a,route_b,benchscout"


def _cli_flag(flag: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1].strip()
        if a.startswith(flag + "="):
            return a.split("=", 1)[1].strip()
    return ""


def _normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode.strip(), mode.strip())


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
    label="waveform-collect",
)


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
        print(f"[collect] router model: {router_fp}", file=sys.stderr)

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
        print(f"[collect] recon model: {recon_fp}", file=sys.stderr)

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
        print(f"[collect] probe model: {probe_fp}", file=sys.stderr)

    return router_fp, recon_fp, probe_fp


def _merge_traces(out_dir: Path) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        trace_path = out_dir / f"trace_{mode}.json"
        if trace_path.is_file():
            with open(trace_path, encoding="utf-8") as f:
                by_mode[mode] = json.load(f)
    return [by_mode[m] for m in MODE_ORDER if m in by_mode]


def _write_report(out_dir: Path, report: dict[str, Any]) -> Path:
    report_path = out_dir / "resource_waveforms.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report_path


def _print_summary(traces: list[dict[str, Any]]) -> None:
    print("\n=== Trace summary ===", file=sys.stderr)
    for tr in traces:
        mode = tr.get("mode", "?")
        summary = tr.get("summary") or {}
        print(
            f"  {mode:12s}  wall={float(tr.get('wall_s') or 0):7.1f}s  "
            f"cpu_mean={summary.get('cpu_pct_mean') or 0:5.1f}%  "
            f"mem_delta_mean={summary.get('mem_delta_pct_mean') or 0:5.2f}%  "
            f"mem_used_mean={summary.get('mem_used_pct_mean') or 0:5.1f}%",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        type=str,
        default=DEFAULT_MODES,
        help="Comma-separated: full, route_a|router_only, route_b|probe_only, benchscout",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--unixbench-root", type=str, default="")
    ap.add_argument("--interval-s", type=float, default=0.5, help="Sample interval (seconds)")
    ap.add_argument("--copies", type=int, default=0, help="UnixBench -c; 0 = single copy")
    ap.add_argument("--router-model", type=str, default="")
    ap.add_argument("--recon-model", type=str, default="")
    ap.add_argument("--probe-model", type=str, default="")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--skip-xi", action="store_true", help="Skip collect_all before routing")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument("--probe-mode", type=str, default="micro", choices=("micro", "real"))
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only run dhry2reg+whetstone-double for full/route_a (smoke test, not for paper)",
    )
    ap.add_argument("-o", "--output-dir", type=str, default="")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip modes whose trace_<mode>.json already exists in output-dir",
    )
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

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
        print("WARNING: --quick uses only 2 subtests; omit for paper-quality waveforms.", file=sys.stderr)

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

    for mode in modes:
        trace_path = out_dir / f"trace_{mode}.json"
        if args.resume and trace_path.is_file():
            print(f"=== Skipping {mode} (exists: {trace_path.name}) ===", file=sys.stderr)
            continue

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
                probe_mode=args.probe_mode,
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
                probe_mode=args.probe_mode,
            )
        else:
            print(f"Unknown mode {mode!r}", file=sys.stderr)
            return 2

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(tr, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {trace_path}", file=sys.stderr)

    traces = _merge_traces(out_dir)
    report: dict[str, Any] = {
        "schema": SCHEMA_COMPARE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "interval_s": args.interval_s,
        "quick_mode": args.quick,
        "measurement_notes": {
            "cpu": "system-wide /proc/stat aggregate",
            "memory_absolute": "MemTotal - MemAvailable",
            "memory_delta": "baseline MemAvailable at t=0 minus current MemAvailable",
            "probe_trace_excludes": ["ebpf", "proc_snapshot"],
            "xi_outside_trace": "route_a and benchscout collect xi before monitoring",
        },
        "traces": traces,
    }
    report_path = _write_report(out_dir, report)
    _print_summary(traces)
    print(json.dumps({"report": str(report_path), "n_traces": len(traces), "output_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
