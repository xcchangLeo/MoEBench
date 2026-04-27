"""Feature vectorization for xi dict (static + dynamic) -> numeric vector.

This module intentionally avoids numpy so it can run in minimal environments.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\\.[0-9]+)?)\\s*([KMGTP]?)(?:i?B)?\\s*$", re.IGNORECASE)


def _parse_size_to_kib(s: str | None) -> float | None:
    if not s:
        return None
    s = str(s).strip()
    m = _SIZE_RE.match(s)
    if not m:
        # sometimes sizes look like "48K" or "896 KiB"
        m2 = re.search(r"([0-9]+(?:\\.[0-9]+)?)\\s*(KiB|KB|MiB|MB|GiB|GB|TiB|TB)", s, re.IGNORECASE)
        if not m2:
            return None
        val = float(m2.group(1))
        unit = m2.group(2).lower()
        if unit in ("kib", "kb"):
            return val
        if unit in ("mib", "mb"):
            return val * 1024.0
        if unit in ("gib", "gb"):
            return val * 1024.0 * 1024.0
        if unit in ("tib", "tb"):
            return val * 1024.0 * 1024.0 * 1024.0
        return None

    val = float(m.group(1))
    unit = m.group(2).upper()
    # unit: '' means bytes
    if unit == "":
        return val / 1024.0
    if unit == "K":
        return val
    if unit == "M":
        return val * 1024.0
    if unit == "G":
        return val * 1024.0 * 1024.0
    if unit == "T":
        return val * 1024.0 * 1024.0 * 1024.0
    if unit == "P":
        return val * 1024.0 * 1024.0 * 1024.0 * 1024.0
    return None


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _hash_to_unit_float(s: str, salt: str = "") -> float:
    """Deterministic mapping from arbitrary string -> [0,1)."""
    h = hashlib.sha256((salt + s).encode("utf-8", errors="replace")).digest()
    # take first 8 bytes
    v = int.from_bytes(h[:8], byteorder="little", signed=False)
    return (v % 10_000_000) / 10_000_000.0


def _extract_memtotal_kb(meminfo_text: str | None) -> float | None:
    if not meminfo_text:
        return None
    # e.g. "MemTotal:       131900000 kB"
    m = re.search(r"^MemTotal:\\s*([0-9]+)\\s*kB\\s*$", meminfo_text, re.MULTILINE)
    if not m:
        m2 = re.search(r"MemTotal:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(kB|KB)", meminfo_text, re.IGNORECASE)
        if m2:
            return float(m2.group(1))
        return None
    return float(m.group(1))


def _extract_num_cpus_from_cpuinfo(cpuinfo_text: str | None) -> int | None:
    if not cpuinfo_text:
        return None
    # count lines "processor : <id>"
    ids = re.findall(r"^processor\\s*:\\s*(\\d+)\\s*$", cpuinfo_text, flags=re.MULTILINE)
    if not ids:
        return None
    # ids may have gaps; use max+1
    maxi = max(int(x) for x in ids)
    return maxi + 1


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _gpu_list_sorted(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def idx(g: dict[str, Any]) -> int:
        try:
            return int(g.get("index", 0))
        except Exception:
            return 0

    return sorted(gpus, key=idx)


def _compute_mode_code(s: str | None) -> float:
    """Stable numeric encoding for ``nvidia-smi`` compute mode."""
    sl = (s or "").strip().lower()
    if not sl:
        return 0.0
    if "prohibited" in sl:
        return 3.0
    if "exclusive thread" in sl or "exclusive_thread" in sl:
        return 2.0
    if "exclusive process" in sl or "exclusive_process" in sl:
        return 1.0
    return 0.0


class XiVectorizer:
    """Convert xi dict to numeric vector with fixed feature ordering."""

    def __init__(self) -> None:
        # Fixed order; keep names stable for saving models.
        self.feature_names = [
            # dynamic / warmup
            "warmup_s",
            "proc_cpu_utilization_ratio",
            "proc_iowait_ratio",
            "vm_page_faults_per_sec",
            "vm_major_faults_per_sec",
            "vm_minor_faults_per_sec",
            "loadavg_1m",
            "loadavg_5m",
            "loadavg_15m",
            "runnable_tasks",
            "total_tasks",
            "last_pid",
            "memory_copy_gib_s",
            "memory_copy_elapsed_s",
            # perf derived (may be absent)
            "perf_ipc",
            "perf_cycles_per_sec",
            "perf_instructions_per_sec",
            "perf_cache_misses_per_sec",
            "perf_branch_misses_per_sec",
            "perf_context_switches_per_sec",
            "perf_page_faults_per_sec",
            # static sysctl
            "perf_event_paranoid",
            "sched_latency_ns",
            "sched_min_granularity_ns",
            "sched_wakeup_granularity_ns",
            "sched_child_runs_first",
            "sched_autogroup_enabled",
            "sched_tunable_scaling",
            "numa_balancing",
            # static cache (from cpu0 cache hierarchy)
            "l1d_total_kib",
            "l1i_total_kib",
            "l2_total_kib",
            "l3_total_kib",
            "cache_line_size_avg",
            # static numa total mem
            "numa_mem_total_mb",
            # static block rotational ratio proxy
            "rotational_devices_ratio",
            # cpufreq governor categorical proxy
            "cpufreq_governor_hash_mean",
            "memtotal_kb",
            "num_cpus",
            # GPU static + dynamic (nvidia-smi + clinfo; absent -> 0)
            "gpu_nvidia_available",
            "gpu_device_count",
            "gpu0_memory_total_mib",
            "gpu0_pcie_gen_max",
            "gpu0_pcie_gen_current",
            "gpu0_pcie_width_max",
            "gpu0_pcie_width_current",
            "gpu0_clock_max_sm_mhz",
            "gpu0_clock_max_memory_mhz",
            "gpu0_power_limit_w",
            "gpu0_persistence_enabled",
            "gpu0_compute_mode_code",
            "gpu_driver_version_hash",
            "opencl_available",
            "opencl_platform_count",
            "opencl_device_count",
            "gpu0_utilization_gpu_pct",
            "gpu0_utilization_memory_pct",
            "gpu0_memory_free_mib",
            "gpu0_memory_used_mib",
            "gpu_min_memory_free_mib",
            "gpu0_power_draw_w",
            "gpu0_temperature_gpu_c",
            "gpu0_clock_current_sm_mhz",
            "gpu0_clock_current_memory_mhz",
        ]

    def transform(self, xi: dict[str, Any]) -> list[float]:
        static = (xi or {}).get("static") or {}
        dynamic = (xi or {}).get("dynamic") or {}

        # dynamic
        warmup_s = _safe_float(dynamic.get("warmup_s")) or 0.0
        proc = dynamic.get("proc") or {}
        cpu_util = proc.get("cpu_utilization") or {}
        cpu_util_ratio = _safe_float(cpu_util.get("cpu_utilization_ratio")) or 0.0
        iowait_ratio = _safe_float(cpu_util.get("iowait_ratio")) or 0.0
        vm = proc.get("vmstat_faults") or proc.get("vmstat_faults_sample") or {}
        pf = _safe_float(vm.get("page_faults_per_sec")) or 0.0
        maj = _safe_float(vm.get("major_faults_per_sec")) or 0.0
        minf = _safe_float(vm.get("minor_faults_per_sec")) or 0.0
        loadavg = proc.get("loadavg") or {}
        l1 = _safe_float(loadavg.get("loadavg_1m")) or 0.0
        l5 = _safe_float(loadavg.get("loadavg_5m")) or 0.0
        l15 = _safe_float(loadavg.get("loadavg_15m")) or 0.0
        runnable = _safe_float(loadavg.get("runnable_tasks")) or 0.0
        total_tasks = _safe_float(loadavg.get("total_tasks")) or 0.0
        last_pid = _safe_float(loadavg.get("last_pid")) or 0.0
        mbw = proc.get("memory_bandwidth_proxy") or {}
        mbw_gib_s = _safe_float(mbw.get("memory_copy_gib_s")) or 0.0
        mbw_elapsed = _safe_float(mbw.get("memory_copy_elapsed_s")) or 0.0

        perf = dynamic.get("perf") or {}
        perf_derived = (perf or {}).get("derived") or {}
        perf_ipc = _safe_float(perf_derived.get("ipc")) or 0.0
        perf_cycles = _safe_float(perf_derived.get("cycles_per_sec")) or 0.0
        perf_insts = _safe_float(perf_derived.get("instructions_per_sec")) or 0.0
        perf_cache_misses = _safe_float(perf_derived.get("cache_misses_per_sec")) or 0.0
        perf_branch_misses = _safe_float(perf_derived.get("branch_misses_per_sec")) or 0.0
        perf_ctx_switches = _safe_float(perf_derived.get("context_switches_per_sec")) or 0.0
        perf_pf = _safe_float(perf_derived.get("page_faults_per_sec")) or 0.0

        # static
        perf_event_paranoid = _safe_float(static.get("scheduler_sysctl", {}).get("perf_event_paranoid_file")) or _safe_float(
            (static.get("scheduler_sysctl", {}) or {}).get("kernel.perf_event_paranoid")
        ) or 0.0

        sysctl = static.get("scheduler_sysctl", {}).get("sysctl") or static.get("scheduler_sysctl", {}).get("sysctl", {})
        sched_latency = _safe_float(sysctl.get("kernel.sched_latency_ns")) or 0.0
        sched_min_gran = _safe_float(sysctl.get("kernel.sched_min_granularity_ns")) or 0.0
        sched_wakeup = _safe_float(sysctl.get("kernel.sched_wakeup_granularity_ns")) or 0.0
        sched_child_runs_first = _safe_float(sysctl.get("kernel.sched_child_runs_first")) or 0.0
        sched_autogroup_enabled = _safe_float(sysctl.get("kernel.sched_autogroup_enabled")) or 0.0
        sched_tunable_scaling = _safe_float(sysctl.get("kernel.sched_tunable_scaling")) or 0.0
        numa_balancing = _safe_float(sysctl.get("kernel.numa_balancing")) or 0.0

        # cache hierarchy from cpu0 cache
        caches = static.get("cache_hierarchy", {}).get("cpu0_cache_indices") or []
        l1d_total_kib = 0.0
        l1i_total_kib = 0.0
        l2_total_kib = 0.0
        l3_total_kib = 0.0
        line_sizes_kib: list[float] = []
        for c in caches:
            level = c.get("level")
            ctype = str(c.get("type") or "")
            size_kib = _parse_size_to_kib(c.get("size"))
            if size_kib is None:
                continue
            if level == "1" or level == 1:
                if ctype.lower().startswith("data") or "data" in ctype.lower():
                    l1d_total_kib += size_kib
                elif ctype.lower().startswith("instruction") or "instruction" in ctype.lower():
                    l1i_total_kib += size_kib
            if level == "2" or level == 2:
                l2_total_kib += size_kib
            if level == "3" or level == 3:
                l3_total_kib += size_kib
            # coherency line size
            cls = _safe_float(c.get("coherency_line_size"))
            if cls is not None and cls > 0:
                # cls is bytes; convert to kib for comparability (or keep bytes; we store kib -> line size avg)
                line_sizes_kib.append(cls / 1024.0)
        cache_line_size_avg = _mean(line_sizes_kib) or 0.0

        # NUMA mem total: parse each node meminfo for MemTotal
        nodes = static.get("numa", {}).get("sysfs_nodes") or []
        mem_total_mb = 0.0
        for n in nodes:
            memtxt = n.get("meminfo")
            if not memtxt:
                continue
            # expect: "Node 0 MemTotal: 128563 MB" or similar
            m = re.search(r"MemTotal:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*([KMGTP]?B?)", memtxt, re.IGNORECASE)
            if not m:
                continue
            val = float(m.group(1))
            unit = (m.group(2) or "").upper()
            unit = unit.replace("B", "")
            if unit == "K":
                mem_total_mb += val / 1024.0
            elif unit == "M" or unit == "":
                mem_total_mb += val
            elif unit == "G":
                mem_total_mb += val * 1024.0
            elif unit == "T":
                mem_total_mb += val * 1024.0 * 1024.0
            elif unit == "P":
                mem_total_mb += val * 1024.0 * 1024.0 * 1024.0

        # rotational devices ratio proxy from sysfs_rotational
        rotational = static.get("block_devices", {}).get("sysfs_rotational") or {}
        rot_vals = list(rotational.values())
        rot_total = len(rot_vals)
        rot_count = 0
        for v in rot_vals:
            fv = _safe_float(v)
            if fv is not None and int(fv) == 1:
                rot_count += 1
        rotational_devices_ratio = (rot_count / rot_total) if rot_total > 0 else 0.0

        # cpufreq governor hash mean
        governors = static.get("cpufreq", {}).get("scaling_governor_per_cpu") or {}
        gov_hashes: list[float] = []
        for _, gv in governors.items():
            if gv is None:
                continue
            gv_s = str(gv)
            gov_hashes.append(_hash_to_unit_float(gv_s, salt="cpufreq"))
        cpufreq_governor_hash_mean = _mean(gov_hashes) or 0.0

        meminfo_text = static.get("memory", {}).get("meminfo") or ""
        memtotal_kb = _extract_memtotal_kb(meminfo_text) or 0.0
        cpuinfo_text = static.get("cpuinfo", {}).get("text") or ""
        num_cpus = float(_extract_num_cpus_from_cpuinfo(cpuinfo_text) or 0)

        gpu_s = static.get("gpu") or {}
        gpu_d = dynamic.get("gpu") or {}
        nv_s = gpu_s.get("nvidia") or {}
        nv_d = gpu_d.get("nvidia") or {}
        oc = gpu_s.get("opencl") or {}

        gpus_s = _gpu_list_sorted(list(nv_s.get("gpus") or []))
        gpus_d = _gpu_list_sorted(list(nv_d.get("gpus") or []))

        gpu_nvidia_available = 1.0 if nv_s.get("available") else 0.0
        gpu_device_count = float(len(gpus_s))

        gs0 = gpus_s[0] if gpus_s else {}
        gd0: dict[str, Any] = {}
        if gpus_d:
            want_idx = gs0.get("index")
            if want_idx is not None:
                for cand in gpus_d:
                    try:
                        if int(cand.get("index")) == int(want_idx):
                            gd0 = cand
                            break
                    except (TypeError, ValueError):
                        continue
            if not gd0:
                gd0 = gpus_d[0]

        gpu0_memory_total_mib = _safe_float(gs0.get("memory_total_mib")) or 0.0
        gpu0_pcie_gen_max = _safe_float(gs0.get("pcie_gen_max")) or 0.0
        gpu0_pcie_gen_current = _safe_float(gs0.get("pcie_gen_current")) or 0.0
        gpu0_pcie_width_max = _safe_float(gs0.get("pcie_width_max")) or 0.0
        gpu0_pcie_width_current = _safe_float(gs0.get("pcie_width_current")) or 0.0
        gpu0_clock_max_sm_mhz = _safe_float(gs0.get("clock_max_sm_mhz")) or 0.0
        gpu0_clock_max_memory_mhz = _safe_float(gs0.get("clock_max_memory_mhz")) or 0.0
        gpu0_power_limit_w = _safe_float(gs0.get("power_limit_w")) or 0.0
        pers = str(gs0.get("persistence_mode") or "")
        gpu0_persistence_enabled = 1.0 if "enabled" in pers.lower() else 0.0
        gpu0_compute_mode_code = _compute_mode_code(str(gs0.get("compute_mode")))
        drv = str(gs0.get("driver_version") or "")
        gpu_driver_version_hash = _hash_to_unit_float(drv, salt="gpu_driver")

        opencl_available = 1.0 if oc.get("available") else 0.0
        opencl_platform_count = float(oc.get("platform_count") or 0)
        opencl_device_count = float(oc.get("device_count") or 0)

        gpu0_utilization_gpu_pct = _safe_float(gd0.get("utilization_gpu_pct")) or 0.0
        gpu0_utilization_memory_pct = _safe_float(gd0.get("utilization_mem_pct")) or 0.0
        gpu0_memory_free_mib = _safe_float(gd0.get("memory_free_mib")) or 0.0
        gpu0_memory_used_mib = _safe_float(gd0.get("memory_used_mib")) or 0.0

        free_vals: list[float] = []
        for g in gpus_d:
            mf = _safe_float(g.get("memory_free_mib"))
            if mf is not None:
                free_vals.append(mf)
        gpu_min_memory_free_mib = float(min(free_vals)) if free_vals else 0.0

        gpu0_power_draw_w = _safe_float(gd0.get("power_draw_w")) or 0.0
        gpu0_temperature_gpu_c = _safe_float(gd0.get("temperature_gpu_c")) or 0.0
        gpu0_clock_current_sm_mhz = _safe_float(gd0.get("clock_current_sm_mhz")) or 0.0
        gpu0_clock_current_memory_mhz = _safe_float(gd0.get("clock_current_memory_mhz")) or 0.0

        # pack in fixed order
        vec = [
            warmup_s,
            cpu_util_ratio,
            iowait_ratio,
            pf,
            maj,
            minf,
            l1,
            l5,
            l15,
            runnable,
            total_tasks,
            last_pid,
            mbw_gib_s,
            mbw_elapsed,
            perf_ipc,
            perf_cycles,
            perf_insts,
            perf_cache_misses,
            perf_branch_misses,
            perf_ctx_switches,
            perf_pf,
            perf_event_paranoid,
            sched_latency,
            sched_min_gran,
            sched_wakeup,
            sched_child_runs_first,
            sched_autogroup_enabled,
            sched_tunable_scaling,
            numa_balancing,
            l1d_total_kib,
            l1i_total_kib,
            l2_total_kib,
            l3_total_kib,
            cache_line_size_avg,
            mem_total_mb,
            rotational_devices_ratio,
            cpufreq_governor_hash_mean,
            memtotal_kb,
            num_cpus,
            gpu_nvidia_available,
            gpu_device_count,
            gpu0_memory_total_mib,
            gpu0_pcie_gen_max,
            gpu0_pcie_gen_current,
            gpu0_pcie_width_max,
            gpu0_pcie_width_current,
            gpu0_clock_max_sm_mhz,
            gpu0_clock_max_memory_mhz,
            gpu0_power_limit_w,
            gpu0_persistence_enabled,
            gpu0_compute_mode_code,
            gpu_driver_version_hash,
            opencl_available,
            opencl_platform_count,
            opencl_device_count,
            gpu0_utilization_gpu_pct,
            gpu0_utilization_memory_pct,
            gpu0_memory_free_mib,
            gpu0_memory_used_mib,
            gpu_min_memory_free_mib,
            gpu0_power_draw_w,
            gpu0_temperature_gpu_c,
            gpu0_clock_current_sm_mhz,
            gpu0_clock_current_memory_mhz,
        ]
        # Ensure stable length
        if len(vec) != len(self.feature_names):
            raise RuntimeError(f"Vector length mismatch: {len(vec)} vs {len(self.feature_names)}")
        return [float(x) for x in vec]

