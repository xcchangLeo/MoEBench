"""Time-series CPU / memory sampling during benchmark workloads."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from moebench.proc_metrics import CpuStatDelta, read_proc_stat_cpu

SCHEMA_RESOURCE_TRACE = "moebench.resource_trace.v1"


@dataclass
class ResourceSample:
    t_rel_s: float
    cpu_pct: float
    mem_used_pct: float
    mem_used_mib: float
    mem_avail_mib: float
    mem_total_mib: float

    def to_dict(self) -> dict[str, float]:
        return {
            "t_rel_s": self.t_rel_s,
            "cpu_pct": self.cpu_pct,
            "mem_used_pct": self.mem_used_pct,
            "mem_used_mib": self.mem_used_mib,
            "mem_avail_mib": self.mem_avail_mib,
            "mem_total_mib": self.mem_total_mib,
        }


def _cpu_total(c: CpuStatDelta) -> float:
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


def _cpu_used_ratio(prev: CpuStatDelta, curr: CpuStatDelta) -> float:
    dt = _cpu_total(curr) - _cpu_total(prev)
    if dt <= 0:
        return 0.0
    idle_d = curr.idle - prev.idle
    used = dt - idle_d
    return max(0.0, min(100.0, 100.0 * used / dt))


def _read_mem_mib() -> tuple[float, float, float]:
    """Return (total_mib, avail_mib, used_mib)."""
    total_kb = avail_kb = None
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
    if total_kb is None or avail_kb is None:
        return 0.0, 0.0, 0.0
    total_mib = total_kb / 1024.0
    avail_mib = avail_kb / 1024.0
    used_mib = max(0.0, total_mib - avail_mib)
    return total_mib, avail_mib, used_mib


class ResourceMonitor:
    """Background sampler; attach while a workload runs."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = max(0.1, float(interval_s))
        self._samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None
        self._prev_cpu: CpuStatDelta | None = None

    def elapsed_s(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._prev_cpu = read_proc_stat_cpu()
        self._thread = threading.Thread(target=self._loop, name="moebench-resource-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._t0 is not None
        while not self._stop.wait(self.interval_s):
            curr = read_proc_stat_cpu()
            total_mib, avail_mib, used_mib = _read_mem_mib()
            cpu_pct = 0.0
            if self._prev_cpu is not None and curr is not None:
                cpu_pct = _cpu_used_ratio(self._prev_cpu, curr)
            if curr is not None:
                self._prev_cpu = curr
            used_pct = (100.0 * used_mib / total_mib) if total_mib > 0 else 0.0
            self._samples.append(
                ResourceSample(
                    t_rel_s=time.perf_counter() - self._t0,
                    cpu_pct=cpu_pct,
                    mem_used_pct=used_pct,
                    mem_used_mib=used_mib,
                    mem_avail_mib=avail_mib,
                    mem_total_mib=total_mib,
                )
            )

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
        wall_s = (self._samples[-1].t_rel_s if self._samples else 0.0) if self._t0 else 0.0
        return trace_dict(
            samples=self._samples,
            label="",
            wall_s=wall_s,
            interval_s=self.interval_s,
        )

    def run(self, fn: Callable[[], Any]) -> dict[str, Any]:
        self.start()
        try:
            fn()
        finally:
            return self.stop()


def trace_dict(
    *,
    samples: list[ResourceSample],
    label: str,
    wall_s: float,
    interval_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cpu_vals = [s.cpu_pct for s in samples]
    mem_vals = [s.mem_used_pct for s in samples]
    out: dict[str, Any] = {
        "schema": SCHEMA_RESOURCE_TRACE,
        "label": label,
        "interval_s": interval_s,
        "wall_s": wall_s,
        "n_samples": len(samples),
        "samples": [s.to_dict() for s in samples],
        "summary": {
            "cpu_pct_mean": _mean(cpu_vals),
            "cpu_pct_max": max(cpu_vals) if cpu_vals else None,
            "mem_used_pct_mean": _mean(mem_vals),
            "mem_used_pct_max": max(mem_vals) if mem_vals else None,
            "mem_used_mib_mean": _mean([s.mem_used_mib for s in samples]),
        },
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        out.update(extra)
    return out


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))
