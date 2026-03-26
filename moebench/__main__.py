"""CLI: dump static/dynamic features as JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from moebench import __version__, collect_all, collect_dynamic, collect_static


def _unified_payload(mode: str, raw: dict) -> dict:
    """Merge采集结果与元数据，便于单文件存档与后续训练管线。"""
    meta = {
        "moebench_version": __version__,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
    }
    if mode == "all":
        return {
            "meta": meta,
            "static": raw.get("static"),
            "dynamic": raw.get("dynamic"),
        }
    if mode == "static":
        return {"meta": meta, "static": raw}
    return {"meta": meta, "dynamic": raw}


def main() -> int:
    p = argparse.ArgumentParser(description="MoEBench system feature collection")
    p.add_argument(
        "--sudo",
        action="store_true",
        help="Re-run this command with sudo -E (recommended when perf/bpftrace needs elevated permissions)",
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("all", "static", "dynamic"),
        help="Which bundle to collect (default: all)",
    )
    p.add_argument("--warmup-s", type=float, default=3.0, help="Warmup duration for dynamic features")
    p.add_argument("--proc-sample-s", type=float, default=0.5, help="/proc delta window (seconds)")
    p.add_argument("--mem-mb", type=int, default=64, help="Warmup working set size (MiB)")
    p.add_argument("--no-ebpf", action="store_true", help="Skip bpftrace probe")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (0 for compact; default 2 for readability)")
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write unified, pretty-printed JSON to this file (UTF-8). If set, stdout stays quiet unless --print",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Also print the same JSON to stdout (only meaningful with -o)",
    )
    p.add_argument(
        "--envelope",
        action="store_true",
        help="Print unified JSON (with meta) to stdout even without -o",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Write only raw collector output (no meta envelope); for -o or stdout",
    )
    args = p.parse_args()

    if args.sudo and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, "-m", "moebench"] + forwarded
        raise SystemExit(subprocess.call(cmd))

    if args.mode == "static":
        raw = collect_static()
    elif args.mode == "dynamic":
        raw = collect_dynamic(
            warmup_s=args.warmup_s,
            proc_sample_s=args.proc_sample_s,
            enable_ebpf=not args.no_ebpf,
            mem_mb=args.mem_mb,
        )
    else:
        raw = collect_all(
            warmup_s=args.warmup_s,
            proc_sample_s=args.proc_sample_s,
            enable_ebpf=not args.no_ebpf,
            mem_mb=args.mem_mb,
        )

    use_envelope = (bool(args.output) or args.envelope) and not args.raw
    if use_envelope:
        out_obj: dict = _unified_payload(args.mode, raw)
    else:
        out_obj = raw

    indent = None if args.indent == 0 else args.indent
    text = json.dumps(out_obj, indent=indent, ensure_ascii=False) + "\n"

    if args.output:
        path = args.output
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {path}", file=sys.stderr)

    if not args.output or args.print:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
