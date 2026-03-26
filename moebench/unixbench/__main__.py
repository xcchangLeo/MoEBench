"""CLI: collect xi + run UnixBench + write dataset JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS
from moebench.unixbench.pipeline import (
    default_dataset_root,
    default_session_tag,
    expert_catalog_only,
    run_unixbench_batch,
    run_unixbench_dataset,
    safe_session_tag,
)


def main() -> int:
    if "--" in sys.argv:
        i = sys.argv.index("--")
        forward = sys.argv[i + 1 :]
        sys.argv = sys.argv[:i]
    else:
        forward = []

    p = argparse.ArgumentParser(
        description="MoEBench UnixBench dataset: features (xi) + full Run (yi, ti) + expert metadata",
    )
    p.add_argument(
        "--sudo",
        action="store_true",
        help="Re-run this command with sudo -E for privileged feature collection",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="Single-run only: write dataset JSON to this path (default: dataset/<session>/run-01.json)",
    )
    p.add_argument(
        "-n",
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of benchmark rounds (default: 1). For N>1, writes dataset/<session>/run-01.json … run-NN.json",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=f"Root directory for all datasets (default: {default_dataset_root()})",
    )
    p.add_argument(
        "--session",
        type=str,
        default=None,
        help="Subfolder name under dataset-root (default: hostname + UTC timestamp)",
    )
    p.add_argument(
        "--reuse-xi",
        action="store_true",
        help="After round 1, reuse the same xi for rounds 2..N (default: collect xi every round)",
    )
    p.add_argument(
        "--unixbench-root",
        type=str,
        default=None,
        help="Path to UnixBench directory (default: byte-unixbench/UnixBench under MoEBench root)",
    )
    p.add_argument("--no-features", action="store_true", help="Skip moebench static+dynamic xi collection")
    p.add_argument("--warmup-s", type=float, default=3.0)
    p.add_argument("--proc-sample-s", type=float, default=0.5)
    p.add_argument("--mem-mb", type=int, default=64)
    p.add_argument("--no-ebpf", action="store_true")
    p.add_argument(
        "--catalog-only",
        action="store_true",
        help="Only write expert catalog E (no benchmark run, no xi)",
    )
    args = p.parse_args()

    if args.sudo and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, "-m", "moebench.unixbench"] + forwarded
        raise SystemExit(subprocess.call(cmd))

    ds_root = Path(args.dataset_root).resolve() if args.dataset_root else default_dataset_root()
    session_resolved = safe_session_tag(args.session or default_session_tag())

    if args.catalog_only:
        out = Path(args.output) if args.output else ds_root / "unixbench-expert-catalog.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "moebench.unixbench.expert_catalog.v1",
            "suite_test_ids": list(INDEX_SUITE_TEST_IDS),
            "experts": expert_catalog_only(),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out}", file=sys.stderr)
        return 0

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2

    if args.runs > 1 and args.output:
        print("Do not use -o with --runs > 1; outputs go under dataset/<session>/run-NN.json", file=sys.stderr)
        return 2

    if args.runs > 1:
        run_unixbench_batch(
            num_rounds=args.runs,
            dataset_root=ds_root,
            session_tag=session_resolved,
            reuse_xi=args.reuse_xi,
            unixbench_root=args.unixbench_root,
            collect_features=not args.no_features,
            warmup_s=args.warmup_s,
            proc_sample_s=args.proc_sample_s,
            mem_mb=args.mem_mb,
            enable_ebpf=not args.no_ebpf,
            run_args=forward if forward else None,
        )
        print(f"Session directory: {ds_root / session_resolved}", file=sys.stderr)
        return 0

    out_path = Path(args.output) if args.output else ds_root / session_resolved / "run-01.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_unixbench_dataset(
        unixbench_root=args.unixbench_root,
        output_json=out_path,
        collect_features=not args.no_features,
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        mem_mb=args.mem_mb,
        enable_ebpf=not args.no_ebpf,
        run_args=forward if forward else None,
        round_index=1,
        total_rounds=1,
        session_tag=session_resolved,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
