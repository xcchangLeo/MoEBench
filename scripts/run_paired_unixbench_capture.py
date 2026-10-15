#!/usr/bin/env python3
"""Run one full UnixBench session and collect probes paired to its exact label run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moebench.probe.training_data import collect_probe_dataset
from moebench.dataset_machines import local_host_slug
from moebench.probe.collector import collect_subtest_probe
from moebench.probe.vectorizer import ProbeVectorizer
from moebench.collector import collect_all
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS
from moebench.unixbench.pipeline import run_unixbench_dataset, safe_session_tag


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_snapshot() -> dict[str, float | int]:
    cpus = os.cpu_count() or 1
    load1, load5, load15 = os.getloadavg()
    return {
        "cpu_count": cpus,
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load1_per_cpu": load1 / cpus,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    ap.add_argument("--session", default="", help="Unique output session name")
    ap.add_argument("--unixbench-root", type=Path, default=None)
    ap.add_argument("--probe-duration-s", type=float, default=4.0)
    ap.add_argument("--probe-mode", choices=("micro", "real"), default="micro")
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--max-load-per-cpu", type=float, default=0.8)
    ap.add_argument("--allow-high-load", action="store_true")
    ap.add_argument("unixbench_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    before = load_snapshot()
    if before["load1_per_cpu"] > args.max_load_per_cpu and not args.allow_high_load:
        raise SystemExit(
            f"Refusing noisy capture: load1/cpu={before['load1_per_cpu']:.2f} exceeds "
            f"{args.max_load_per_cpu:.2f}. Retry when idle or explicitly pass --allow-high-load."
        )

    dataset_root = args.dataset_root.resolve()
    session = safe_session_tag(args.session or f"{local_host_slug()}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    session_dir = dataset_root / session
    session_dir.mkdir(parents=True, exist_ok=False)
    run_path = session_dir / "run-01.json"
    run_args = args.unixbench_args
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]

    # Capture the context once before any probe workload and reuse it for the
    # paired full run. This keeps the router input aligned with the probe state.
    xi_started_at = datetime.now(timezone.utc).isoformat()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    xi_finished_at = datetime.now(timezone.utc).isoformat()
    xi_elapsed_s = (datetime.fromisoformat(xi_finished_at) - datetime.fromisoformat(xi_started_at)).total_seconds()
    probe_started_at = xi_finished_at
    probes = {
        tid: collect_subtest_probe(
            tid,
            duration_s=args.probe_duration_s,
            enable_ebpf=not args.no_ebpf,
            benchmark="unixbench",
            probe_mode=args.probe_mode,
            unixbench_root=args.unixbench_root,
        )
        for tid in INDEX_SUITE_TEST_IDS
    }
    load_after_probes = load_snapshot()
    probe_finished_at = datetime.now(timezone.utc).isoformat()

    pre_run_path = session_dir / "pre_run_capture.json"
    pre_run_capture = {
        "schema": "moebench.unixbench.pre_run_capture.v1",
        "session": session,
        "probe_timing": "before_full_run",
        "xi_timing": "before_probe_and_full_run",
        "xi": xi,
        "probes": probes,
        "xi_started_at_utc": xi_started_at,
        "xi_finished_at_utc": xi_finished_at,
        "probe_started_at_utc": probe_started_at,
        "probe_finished_at_utc": probe_finished_at,
        "load_before": before,
        "load_after_probes": load_after_probes,
    }
    write_json(pre_run_path, pre_run_capture)
    manifest_path = session_dir / "paired_capture_manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "moebench.unixbench.paired_capture.v1",
            "session": session,
            "machine": platform.node(),
            "capture_status": "probes_complete_full_run_pending",
            "protocol": "fresh xi + probes-before-full-run + complete UnixBench label run; probe labels restricted to exact run path",
            "probe_mode": args.probe_mode,
            "probe_timing": "before_full_run",
            "xi_timing": "before_probe_and_full_run",
            "xi_started_at_utc": xi_started_at,
            "xi_finished_at_utc": xi_finished_at,
            "probe_started_at_utc": probe_started_at,
            "probe_finished_at_utc": probe_finished_at,
            "xi_elapsed_s": xi_elapsed_s,
            "probe_duration_s": args.probe_duration_s,
            "probe_wall_sum_s": float(sum(float(p.get("wall_s") or 0.0) for p in probes.values())),
            "enable_ebpf": not args.no_ebpf,
            "load_before": before,
            "load_after_probes": load_after_probes,
            "pre_run_capture": {"path": str(pre_run_path.resolve()), "sha256": sha256(pre_run_path)},
        },
    )

    full_run_started_at = datetime.now(timezone.utc).isoformat()
    full_run_started_clock = time.perf_counter()
    run_unixbench_dataset(
        unixbench_root=args.unixbench_root,
        output_json=run_path,
        collect_features=False,
        xi_override=xi,
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        mem_mb=args.mem_mb,
        enable_ebpf=not args.no_ebpf,
        run_args=run_args or None,
        round_index=1,
        total_rounds=1,
        session_tag=session,
    )
    full_run_finished_at = datetime.now(timezone.utc).isoformat()
    full_run_elapsed_s = time.perf_counter() - full_run_started_clock
    probe_dataset = collect_probe_dataset(
        benchmark="unixbench",
        dataset_root=dataset_root,
        machine=local_host_slug(),
        probe_duration_s=args.probe_duration_s,
        enable_ebpf=not args.no_ebpf,
        probe_mode=args.probe_mode,
        live_probe=False,
        source_runs=[run_path],
    )
    vectorizer = ProbeVectorizer()
    for sample in probe_dataset["samples"]:
        probe = probes[sample["test_id"]]
        sample["probe"] = probe
        sample["probe_vector"] = vectorizer.transform(probe)
    probe_dataset["collection_started_at_utc"] = probe_started_at
    probe_dataset["collection_finished_at_utc"] = probe_finished_at
    probe_dataset["probe_timing"] = "before_full_run"
    probe_path = session_dir / "probe_dataset_unixbench.json"
    probe_path.write_text(json.dumps(probe_dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "schema": "moebench.unixbench.paired_capture.v1",
        "session": session,
        "machine": platform.node(),
        "capture_status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fresh xi + probes-before-full-run + complete UnixBench label run; probe labels restricted to exact run path",
        "probe_mode": args.probe_mode,
        "probe_timing": "before_full_run",
        "xi_timing": "before_probe_and_full_run",
        "xi_started_at_utc": xi_started_at,
        "xi_finished_at_utc": xi_finished_at,
        "probe_started_at_utc": probe_started_at,
        "probe_finished_at_utc": probe_finished_at,
        "full_run_started_at_utc": full_run_started_at,
        "full_run_finished_at_utc": full_run_finished_at,
        "full_run_elapsed_s": full_run_elapsed_s,
        "xi_elapsed_s": xi_elapsed_s,
        "probe_duration_s": args.probe_duration_s,
        "probe_wall_sum_s": float(sum(float(p.get("wall_s") or 0.0) for p in probes.values())),
        "enable_ebpf": not args.no_ebpf,
        "load_before": before,
        "load_after_probes": load_after_probes,
        "load_after_run": load_snapshot(),
        "run": {"path": str(run_path.resolve()), "sha256": sha256(run_path)},
        "probe_dataset": {"path": str(probe_path.resolve()), "sha256": sha256(probe_path)},
        "pre_run_capture": {"path": str(pre_run_path.resolve()), "sha256": sha256(pre_run_path)},
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"session_dir": str(session_dir), "run": str(run_path), "probes": str(probe_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
