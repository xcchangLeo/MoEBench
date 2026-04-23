"""CLI: collect xi + run Phoronix Test Suite + write dataset JSON."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from moebench.phoronix.pipeline import (
    DEFAULT_PTS_SMOKE_SUITE,
    default_dataset_root,
    default_pts_install_root,
    default_session_tag,
    run_pts_batch,
    run_pts_dataset,
    safe_session_tag,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="MoEBench PTS dataset: features (xi) + phoronix-test-suite run (yi, ti)",
    )
    p.add_argument(
        "--sudo",
        action="store_true",
        help="Re-run this command with sudo -E for privileged feature collection",
    )
    p.add_argument(
        "--pts-stream",
        action="store_true",
        help=(
            "Capture PTS output and re-print it (enables batch-run setup notice detection). "
            "Default: inherit your real terminal for --pts-mode run (like typing "
            "phoronix-test-suite run …); stream for batch-run/batch-benchmark."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="Write dataset JSON (default: dataset/<session>/run-01.json)",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=f"Root for datasets (default: {default_dataset_root()})",
    )
    p.add_argument(
        "--session",
        type=str,
        default=None,
        help="Subfolder under dataset-root (default: hostname + UTC timestamp)",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="cpu",
        help="PTS suite/test name to pass to phoronix-test-suite (default: cpu)",
    )
    p.add_argument(
        "--pts-smoke",
        action="store_true",
        help=(
            "Quick pipeline check: run a single lightweight PTS test instead of the full "
            f"--suite value (default test: {DEFAULT_PTS_SMOKE_SUITE}). Overrides --suite. "
            "Install once: phoronix-test-suite install <suite>."
        ),
    )
    p.add_argument(
        "--pts-smoke-suite",
        type=str,
        default=DEFAULT_PTS_SMOKE_SUITE,
        metavar="ID",
        help=f"PTS test profile for --pts-smoke (default: {DEFAULT_PTS_SMOKE_SUITE})",
    )
    p.add_argument(
        "--pts-mode",
        type=str,
        choices=("batch-run", "run", "batch-benchmark"),
        default="batch-run",
        help="PTS subcommand (default: batch-run, non-interactive defaults; use `run` for `phoronix-test-suite run <suite>`)",
    )
    p.add_argument(
        "--pts-bin",
        type=str,
        default=None,
        help="Path to phoronix-test-suite executable (default: PATH or <repo>/phoronix-test-suite/phoronix-test-suite)",
    )
    p.add_argument(
        "--pts-root",
        type=str,
        default=None,
        help="MoEBench repo subdir containing phoronix-test-suite (default: <repo>/phoronix-test-suite)",
    )
    p.add_argument(
        "--result-name",
        type=str,
        default=None,
        help="TEST_RESULTS_NAME (single round only; default: moebench_pts_<host>_<UTC>)",
    )
    p.add_argument(
        "-n",
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of rounds / dataset files (default: 1). Writes run-01.json … run-NN.json + manifest.json",
    )
    p.add_argument(
        "--reuse-xi",
        action="store_true",
        help="After round 1, reuse the same xi for rounds 2..N (default: collect xi every round)",
    )
    p.add_argument("--no-features", action="store_true", help="Skip xi collection")
    p.add_argument("--warmup-s", type=float, default=3.0)
    p.add_argument("--proc-sample-s", type=float, default=0.5)
    p.add_argument("--mem-mb", type=int, default=64)
    p.add_argument("--no-ebpf", action="store_true")
    p.add_argument(
        "pts_extra",
        nargs="*",
        help="Extra args appended after suite (e.g. additional test IDs)",
    )
    args = p.parse_args()

    suite = args.pts_smoke_suite if args.pts_smoke else args.suite
    if args.pts_smoke:
        print(
            f"--pts-smoke: using single test {suite!r} (override with --pts-smoke-suite)",
            file=sys.stderr,
        )

    if args.sudo and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, "-m", "moebench.phoronix"] + forwarded
        raise SystemExit(subprocess.call(cmd))

    ds_root = Path(args.dataset_root).resolve() if args.dataset_root else default_dataset_root()
    session_resolved = safe_session_tag(args.session or default_session_tag(suite))
    out_path = Path(args.output) if args.output else ds_root / session_resolved / "run-01.json"

    pts_root = Path(args.pts_root).resolve() if args.pts_root else default_pts_install_root()

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2
    if args.runs > 1 and args.output:
        print("Do not use -o with --runs > 1; outputs go under dataset/<session>/run-NN.json", file=sys.stderr)
        return 2
    if args.runs > 1 and args.result_name:
        print("Do not use --result-name with --runs > 1; each round uses a unique TEST_RESULTS_NAME", file=sys.stderr)
        return 2

    pts_inherit_stdio: bool | None
    if args.pts_stream:
        pts_inherit_stdio = False
    else:
        pts_inherit_stdio = args.pts_mode == "run"

    if args.runs > 1:
        run_pts_batch(
            num_rounds=args.runs,
            dataset_root=ds_root,
            session_tag=session_resolved,
            reuse_xi=args.reuse_xi,
            pts_bin=args.pts_bin,
            pts_root=pts_root if pts_root.is_dir() else None,
            suite=suite,
            pts_mode=args.pts_mode,
            pts_extra_args=list(args.pts_extra) if args.pts_extra else None,
            collect_features=not args.no_features,
            warmup_s=args.warmup_s,
            proc_sample_s=args.proc_sample_s,
            mem_mb=args.mem_mb,
            enable_ebpf=not args.no_ebpf,
            pts_inherit_stdio=pts_inherit_stdio,
        )
        print(f"Session directory: {ds_root / session_resolved}", file=sys.stderr)
        return 0

    run_pts_dataset(
        pts_bin=args.pts_bin,
        pts_root=pts_root if pts_root.is_dir() else None,
        suite=suite,
        pts_mode=args.pts_mode,
        pts_extra_args=list(args.pts_extra) if args.pts_extra else None,
        output_json=out_path,
        collect_features=not args.no_features,
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        mem_mb=args.mem_mb,
        enable_ebpf=not args.no_ebpf,
        session_tag=session_resolved,
        round_index=1,
        total_rounds=1,
        result_name=args.result_name,
        pts_inherit_stdio=pts_inherit_stdio,
    )
    print(f"Session directory: {out_path.parent}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
