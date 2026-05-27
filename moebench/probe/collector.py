"""Collect one short probe window per subtest (workload + eBPF + /proc)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

from moebench.probe.ebpf_features import collect_ebpf_window
from moebench.probe.real_runner import run_pts_profile_timed, run_unixbench_subtest_timed
from moebench.probe.workloads import (
    category_for_pts_test,
    category_for_unixbench_test,
    run_category_workload,
)

ProbeMode = Literal["micro", "real"]


def collect_subtest_probe(
    test_id: str,
    *,
    duration_s: float = 4.0,
    enable_ebpf: bool = True,
    mem_mb: int = 64,
    benchmark: str = "unixbench",
    category: str | None = None,
    probe_mode: ProbeMode = "micro",
    pts_exe: str | None = None,
    unixbench_root: Path | None = None,
    pts_title: str | None = None,
) -> dict[str, Any]:
    """
    Collect probe features for one subtest/profile.

    ``probe_mode=micro``: category micro-workload + eBPF (fast, approximate stress).
    ``probe_mode=real``: run real ``perl Run -c 1 <test>`` or ``phoronix-test-suite run <profile>``
    under ``timeout`` while eBPF samples — **real binary**, but often **no valid score** if
    the official test needs longer than ``duration_s`` (labels still from full ``dataset/`` runs).
    """
    duration_s = max(1.0, min(30.0, float(duration_s)))
    if category is None:
        if benchmark == "unixbench":
            category = category_for_unixbench_test(test_id)
        else:
            category = category_for_pts_test(test_id, pts_title)

    ebpf_result: dict[str, Any] | None = None

    def _ebpf_thread() -> None:
        nonlocal ebpf_result
        try:
            if enable_ebpf:
                ebpf_result = collect_ebpf_window(duration_s)
            else:
                ebpf_result = {"available": False, "reason": "disabled"}
        except Exception as e:
            ebpf_result = {"available": False, "reason": str(e)}

    real_run: dict[str, Any] | None = None
    workload: dict[str, Any] | None = None

    t_ebpf = threading.Thread(target=_ebpf_thread, daemon=True)
    t0 = time.perf_counter()
    t_ebpf.start()

    if probe_mode == "real":
        if benchmark == "unixbench":
            real_run = run_unixbench_subtest_timed(
                test_id,
                duration_s=duration_s,
                unixbench_root=unixbench_root,
            )
        elif benchmark == "phoronix":
            if not pts_exe:
                from moebench.phoronix.pipeline import _which_pts, default_pts_install_root

                root = default_pts_install_root()
                pts_exe = _which_pts(None, root if root.is_dir() else None)
            real_run = run_pts_profile_timed(
                test_id,
                duration_s=duration_s,
                pts_exe=pts_exe,
            )
        else:
            raise ValueError(f"unknown benchmark: {benchmark!r}")
        workload = {
            "category": category,
            "duration_s": duration_s,
            "returncode": real_run.get("returncode"),
            "mode": "real",
        }
    else:
        workload = run_category_workload(category, duration_s, mem_mb=mem_mb)
        workload["mode"] = "micro"

    proc_snap = _proc_snapshot(min(0.5, duration_s * 0.25))
    t_ebpf.join(timeout=duration_s + 15.0)
    wall_s = time.perf_counter() - t0

    return {
        "test_id": test_id,
        "benchmark": benchmark,
        "category": category,
        "probe_mode": probe_mode,
        "duration_s": duration_s,
        "wall_s": wall_s,
        "workload": workload,
        "real_run": real_run,
        "ebpf": ebpf_result,
        "proc": proc_snap,
    }


def _proc_snapshot(interval_s: float) -> dict[str, Any]:
    from moebench import proc_metrics

    return {
        "cpu_utilization": proc_metrics.sample_cpu_utilization(interval_s),
        "vmstat_faults": proc_metrics.sample_vmstat_faults(interval_s),
        "loadavg": proc_metrics.read_loadavg(),
    }
