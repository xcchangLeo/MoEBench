#!/usr/bin/env python3
"""Collect short eBPF probe dataset (labels from full runs on this machine)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import ensure_machine_output_dir, machine_models_dir, resolve_training_machine
from moebench.ml_venv import ensure_ml_interpreter
from moebench.phoronix.pipeline import safe_session_tag

ensure_ml_interpreter(
    need_modules=["numpy"],
    auto_install="--auto-install" in sys.argv,
    label="probe_collect",
)

from moebench.probe.training_data import collect_probe_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--benchmark",
        type=str,
        choices=("unixbench", "phoronix"),
        default="unixbench",
    )
    ap.add_argument(
        "--pts-suite",
        type=str,
        default="",
        help="Required for phoronix: cpu or pts/nvidia-gpu-compute",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="", help="Host slug (default: current hostname)")
    ap.add_argument("--probe-duration-s", type=float, default=4.0, help="Seconds per subtest (3–5 recommended)")
    ap.add_argument(
        "--probe-mode",
        type=str,
        choices=("micro", "real"),
        default="micro",
        help="micro: category workload; real: timeout-wrapped real UnixBench/PTS subtest",
    )
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument(
        "--auto-install",
        action="store_true",
        help="Bootstrap project ML venv (scripts/install_ml_python_deps.sh) if numpy is missing",
    )
    ap.add_argument("-o", "--output", type=str, default="", help="Output JSON path")
    ap.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Limit number of source run JSONs (0 = all on this machine)",
    )
    args = ap.parse_args()

    if args.benchmark == "phoronix" and not args.pts_suite.strip():
        print("--benchmark phoronix requires --pts-suite", file=sys.stderr)
        return 2

    machine = resolve_training_machine(args.machine or None)
    pts_suite = args.pts_suite.strip() or None
    ds = collect_probe_dataset(
        benchmark=args.benchmark,
        dataset_root=args.dataset_root,
        machine=machine,
        pts_suite=pts_suite,
        probe_duration_s=args.probe_duration_s,
        enable_ebpf=not args.no_ebpf,
        probe_mode=args.probe_mode,
        live_probe=True,
    )
    if args.max_runs > 0:
        by_run: dict[str, list] = {}
        for s in ds.get("samples") or []:
            by_run.setdefault(str(s.get("source_run")), []).append(s)
        keys = sorted(by_run.keys())[: args.max_runs]
        kept: list = []
        for k in keys:
            kept.extend(by_run[k])
        ds["samples"] = kept
        ds["num_samples"] = len(kept)

    if args.output.strip():
        out = Path(args.output).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if args.benchmark == "unixbench":
            fname = f"probe_dataset_unixbench_{stamp}.json"
        else:
            tok = safe_session_tag(pts_suite or "pts").replace("/", "_")
            fname = f"probe_dataset_{tok}_{stamp}.json"
        out = machine_models_dir(args.dataset_root, machine) / fname

    ensure_machine_output_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "wrote": str(out),
                "benchmark": args.benchmark,
                "pts_suite": pts_suite,
                "machine": machine,
                "probe_mode": args.probe_mode,
                "num_samples": ds.get("num_samples"),
                "probe_duration_s": ds.get("probe_duration_s"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
