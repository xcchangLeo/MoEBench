#!/usr/bin/env python3
"""Hybrid experiment: xi → router Top-K → probe (Route B) → reconstruct → compare vs dataset full run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    ensure_machine_output_dir,
    machine_experiments_dir,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.hybrid.eval import evaluate_hybrid_offline, evaluate_hybrid_online, load_router_meta
from moebench.probe.inference import load_probe_bundle
from moebench.reconstruct.data import collect_unixbench_run_paths
from moebench.reconstruct.inference import load_reconstruction_bundle
from moebench.phoronix.training_data import collect_phoronix_run_paths


def _latest_run(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_probe_dataset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--router-model", type=str, required=True)
    ap.add_argument("--reconstruct-model", type=str, required=True)
    ap.add_argument("--probe-model", type=str, required=True)
    ap.add_argument("--probe-dataset", type=str, default="", help="Required for --offline")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--benchmark", choices=("unixbench", "phoronix"), default="unixbench")
    ap.add_argument("--pts-suite", type=str, default="cpu")
    ap.add_argument("--glob-pattern", type=str, default="")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--probe-duration-s", type=float, default=None)
    ap.add_argument("--probe-mode", type=str, choices=("micro", "real"), default="")
    ap.add_argument("--xi-overhead-s", type=float, default=3.0, help="Offline: estimated xi collection time")
    ap.add_argument("--offline", action="store_true", help="Replay all historical runs (default if set)")
    ap.add_argument("--online", action="store_true", help="Live xi + probe on router-selected subset only")
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    if args.online and args.offline:
        print("Choose only one of --online or --offline", file=sys.stderr)
        return 2
    offline = args.offline or not args.online

    machine = resolve_training_machine(args.machine or None)
    ds_root = Path(args.dataset_root).resolve()
    router_meta = load_router_meta(Path(args.router_model).resolve())
    recon_bundle = load_reconstruction_bundle(args.reconstruct_model)
    probe_bundle = load_probe_bundle(args.probe_model)

    pts_suite = args.pts_suite.strip() if args.benchmark == "phoronix" else None
    glo = resolve_glob_for_machine(
        benchmark=args.benchmark,
        machine=machine,
        glob_pattern=args.glob_pattern or None,
        pts_suite=pts_suite,
    )

    if offline:
        probe_ds_path = Path(args.probe_dataset).resolve() if args.probe_dataset else None
        if probe_ds_path is None:
            tok = "unixbench" if args.benchmark == "unixbench" else f"pts_{pts_suite.replace('/', '_')}"
            probe_ds_path = ds_root / "models" / machine / f"probe_dataset_{tok}.json"
        if not probe_ds_path.is_file():
            print(f"Probe dataset not found: {probe_ds_path}", file=sys.stderr)
            return 2
        probe_dataset = _load_probe_dataset(probe_ds_path)
        if args.benchmark == "unixbench":
            run_paths = collect_unixbench_run_paths(ds_root, glob_pattern=glo)
        else:
            run_paths = collect_phoronix_run_paths(ds_root, glob_pattern=glo, pts_suite=pts_suite)
        report = evaluate_hybrid_offline(
            run_paths=run_paths,
            router_meta=router_meta,
            recon_bundle=recon_bundle,
            probe_bundle=probe_bundle,
            probe_dataset=probe_dataset,
            top_k=args.top_k,
            probe_duration_s=args.probe_duration_s,
            xi_overhead_s=args.xi_overhead_s,
        )
    else:
        from moebench import collect_all

        t0 = time.perf_counter()
        xi = collect_all(enable_ebpf=not args.no_ebpf)
        t_xi = time.perf_counter() - t0
        if args.benchmark == "unixbench":
            paths = collect_unixbench_run_paths(ds_root, glob_pattern=glo)
        else:
            paths = collect_phoronix_run_paths(ds_root, glob_pattern=glo, pts_suite=pts_suite)
        if not paths:
            print("No ground-truth runs in dataset", file=sys.stderr)
            return 2
        gt_path = _latest_run(paths)
        gt_ds = json.load(open(gt_path, encoding="utf-8"))
        report = evaluate_hybrid_online(
            xi=xi,
            router_meta=router_meta,
            recon_bundle=recon_bundle,
            probe_bundle=probe_bundle,
            ground_truth_ds=gt_ds,
            ground_truth_run=gt_path,
            top_k=args.top_k,
            probe_duration_s=args.probe_duration_s,
            probe_mode=args.probe_mode or None,
            enable_ebpf=not args.no_ebpf,
            xi_wall_s=t_xi,
        )

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["machine"] = machine
    report["router_model"] = str(Path(args.router_model).resolve())
    report["reconstruct_model"] = str(Path(args.reconstruct_model).resolve())
    report["probe_model"] = str(Path(args.probe_model).resolve())
    report["probe_backend"] = str(probe_bundle.get("model_type") or "")
    report["dataset_root"] = str(ds_root)
    report["glob_pattern"] = glo

    if args.output.strip():
        out = Path(args.output).resolve()
    else:
        tok = "unixbench" if args.benchmark == "unixbench" else args.pts_suite.replace("/", "_")
        out = machine_experiments_dir(ds_root, machine) / f"hybrid_{tok}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    ensure_machine_output_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
