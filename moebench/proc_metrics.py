"""Read CPU, memory, and I/O proxies from /proc and /sys (no perf required)."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CpuStatDelta:
    """Per-CPU jiffies from /proc/stat (first line is aggregate)."""

    user: float
    nice: float
    system: float
    idle: float
    iowait: float
    irq: float
    softirq: float
    steal: float
    guest: float
    guest_nice: float


def _parse_cpu_line(line: str) -> CpuStatDelta | None:
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    # cpu user nice system idle iowait irq softirq steal guest guest_nice
    vals = [float(x) for x in parts[1:11]]
    while len(vals) < 10:
        vals.append(0.0)
    return CpuStatDelta(
        user=vals[0],
        nice=vals[1],
        system=vals[2],
        idle=vals[3],
        iowait=vals[4],
        irq=vals[5],
        softirq=vals[6],
        steal=vals[7],
        guest=vals[8],
        guest_nice=vals[9],
    )


def read_proc_stat_cpu() -> CpuStatDelta | None:
    with open("/proc/stat", encoding="utf-8") as f:
        for line in f:
            c = _parse_cpu_line(line)
            if c is not None:
                return c
    return None


def read_loadavg() -> dict[str, Any]:
    with open("/proc/loadavg", encoding="utf-8") as f:
        line = f.read().strip()
    parts = line.split()
    # load1 load5 load15 running/total last_pid
    if len(parts) >= 5:
        run_parts = parts[3].split("/")
        return {
            "loadavg_1m": float(parts[0]),
            "loadavg_5m": float(parts[1]),
            "loadavg_15m": float(parts[2]),
            "runnable_tasks": int(run_parts[0]) if len(run_parts) > 1 else None,
            "total_tasks": int(run_parts[1]) if len(run_parts) > 1 else None,
            "last_pid": int(parts[4]),
        }
    return {}


def sample_cpu_utilization(interval_s: float = 0.5) -> dict[str, float | None]:
    """CPU utilization and iowait fraction from /proc/stat deltas."""
    a = read_proc_stat_cpu()
    if a is None:
        return {}
    time.sleep(interval_s)
    b = read_proc_stat_cpu()
    if b is None:
        return {}

    def total(c: CpuStatDelta) -> float:
        return (
            c.user
            + c.nice
            + c.system
            + c.idle
            + c.iowait
            + c.irq
            + c.softirq
            + c.steal
            + c.guest
            + c.guest_nice
        )

    da = total(b) - total(a)
    if da <= 0:
        return {"cpu_utilization_ratio": None, "iowait_ratio": None}

    idle_d = b.idle - a.idle
    iowait_d = b.iowait - a.iowait
    used = da - idle_d
    return {
        "cpu_utilization_ratio": max(0.0, min(1.0, used / da)),
        "iowait_ratio": max(0.0, min(1.0, iowait_d / da)),
    }


def read_vmstat_keys(keys: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/vmstat", encoding="utf-8") as f:
        for line in f:
            if " " not in line:
                continue
            k, v = line.split(None, 1)
            if k in keys:
                out[k] = int(v.strip())
    return out


def sample_vmstat_faults(interval_s: float = 0.5) -> dict[str, float | None]:
    """Major/minor page faults per second from /proc/vmstat."""
    keys = {"pgfault", "pgmajfault"}
    a = read_vmstat_keys(keys)
    time.sleep(interval_s)
    b = read_vmstat_keys(keys)
    if not a or not b:
        return {}
    dt = interval_s
    minor_a = a.get("pgfault", 0) - a.get("pgmajfault", 0)
    minor_b = b.get("pgfault", 0) - b.get("pgmajfault", 0)
    return {
        "page_faults_per_sec": (b.get("pgfault", 0) - a.get("pgfault", 0)) / dt,
        "major_faults_per_sec": (b.get("pgmajfault", 0) - a.get("pgmajfault", 0)) / dt,
        "minor_faults_per_sec": (minor_b - minor_a) / dt,
    }


def read_diskstats() -> dict[str, dict[str, int]]:
    """Per-device stats from /proc/diskstats (kernel doc: field indices)."""
    # https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
    out: dict[str, dict[str, int]] = {}
    with open("/proc/diskstats", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 14:
                continue
            dev = p[2]
            out[dev] = {
                "reads_completed": int(p[3]),
                "reads_merged": int(p[4]),
                "sectors_read": int(p[5]),
                "read_ms": int(p[6]),
                "writes_completed": int(p[7]),
                "writes_merged": int(p[8]),
                "sectors_written": int(p[9]),
                "write_ms": int(p[10]),
                "in_flight": int(p[11]),
                "io_ms": int(p[12]),
                "weighted_io_ms": int(p[13]),
            }
    return out


def io_latency_proxy(interval_s: float = 0.5) -> dict[str, Any]:
    """
    Rough I/O latency proxy: delta weighted_io_ms / delta completed_ios per disk.
    Not comparable across kernels if drivers change accounting.
    """
    a = read_diskstats()
    time.sleep(interval_s)
    b = read_diskstats()
    per_disk: dict[str, dict[str, float | None]] = {}
    for dev in set(a) & set(b):
        ra, rb = a[dev], b[dev]
        rio = rb["reads_completed"] - ra["reads_completed"]
        wio = rb["writes_completed"] - ra["writes_completed"]
        ios = rio + wio
        wms = rb["weighted_io_ms"] - ra["weighted_io_ms"]
        per_disk[dev] = {
            "ios_per_sec": ios / interval_s,
            "weighted_io_ms_per_sec": wms / interval_s,
            "ms_per_io": (wms / ios) if ios > 0 else None,
        }
    return {"per_disk": per_disk, "interval_s": interval_s}


def memory_bandwidth_proxy_mb_s(duration_s: float = 0.25, chunk_mb: int = 64) -> dict[str, float | None]:
    """
    User-space memcpy bandwidth (GB/s proxy). Not DRAM theoretical peak; stable relative feature.
    """
    try:
        import array
    except ImportError:
        return {"memory_copy_gib_s": None, "error": "array module missing"}

    n = chunk_mb * 1024 * 1024
    buf_a = array.array("B", (i % 256 for i in range(n)))
    buf_b = array.array("B", [0] * n)
    t0 = time.perf_counter()
    end = t0 + duration_s
    moves = 0
    while time.perf_counter() < end:
        buf_b[:] = buf_a
        moves += 1
    elapsed = time.perf_counter() - t0
    gib = (n * moves) / (1024**3)
    return {
        "memory_copy_gib_s": gib / elapsed if elapsed > 0 else None,
        "memory_copy_iterations": moves,
        "memory_copy_elapsed_s": elapsed,
    }
