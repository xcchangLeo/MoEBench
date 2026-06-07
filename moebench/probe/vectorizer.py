"""Vectorize probe dicts for ML (eBPF + proc + metadata)."""

from __future__ import annotations

from typing import Any

_PROBE_FEATURE_NAMES: list[str] = [
    "probe_duration_s",
    "wall_s",
    "workload_rc",
    "ebpf_sched_switch_per_s",
    "ebpf_syscall_per_s",
    "ebpf_available",
    "proc_cpu_user",
    "proc_cpu_system",
    "proc_cpu_idle",
    "proc_minor_faults",
    "proc_major_faults",
    "cat_cpu",
    "cat_memory",
    "cat_io",
    "cat_thread",
    "cat_syscall",
    "cat_gpu",
    "real_run",
    "real_timed_out",
]


def probe_feature_names() -> list[str]:
    return list(_PROBE_FEATURE_NAMES)


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


class ProbeVectorizer:
    """Fixed-order float vector from ``collect_subtest_probe`` output."""

    def __init__(self) -> None:
        self.feature_names = probe_feature_names()

    def transform(self, probe: dict[str, Any]) -> list[float]:
        ebpf = probe.get("ebpf") or {}
        proc = probe.get("proc") or {}
        cpu = proc.get("cpu_utilization") or {}
        faults = proc.get("vmstat_faults") or {}
        wl = probe.get("workload") or {}
        cat = str(probe.get("category") or "CPU").lower()

        cats = {c: 0.0 for c in ("cpu", "memory", "io", "thread", "syscall", "gpu")}
        if cat in cats:
            cats[cat] = 1.0

        cpu_ratio = _safe_float(cpu.get("cpu_utilization_ratio"))
        if cpu_ratio == 0.0:
            cpu_ratio = _safe_float(cpu.get("user_pct")) + _safe_float(cpu.get("system_pct"))
        iowait = _safe_float(cpu.get("iowait_ratio"))
        minor_faults = _safe_float(faults.get("minor_faults_per_sec"))
        if minor_faults == 0.0:
            minor_faults = _safe_float(faults.get("minor_faults"))
        major_faults = _safe_float(faults.get("major_faults_per_sec"))
        if major_faults == 0.0:
            major_faults = _safe_float(faults.get("major_faults"))

        ebpf_ok = 1.0 if ebpf.get("available") else 0.0
        real = probe.get("real_run") or {}
        return [
            _safe_float(probe.get("duration_s")),
            _safe_float(probe.get("wall_s")),
            _safe_float(wl.get("returncode")),
            _safe_float(ebpf.get("sched_switch_per_s")),
            _safe_float(ebpf.get("syscall_enter_per_s")),
            ebpf_ok,
            cpu_ratio,
            iowait,
            max(0.0, 1.0 - cpu_ratio - iowait),
            minor_faults,
            major_faults,
            cats["cpu"],
            cats["memory"],
            cats["io"],
            cats["thread"],
            cats["syscall"],
            cats["gpu"],
            1.0 if real.get("runner") else 0.0,
            1.0 if real.get("timed_out") else 0.0,
        ]
