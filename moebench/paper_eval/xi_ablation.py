"""Zero-out portions of the xi numeric vector for ablation studies."""

from __future__ import annotations

from typing import Any

from moebench.router.feature_vectorizer import XiVectorizer


def _keep_mask_for_mode(feature_names: list[str], mode: str) -> list[bool]:
    if mode == "full":
        return [True] * len(feature_names)
    if mode == "static_hw_only":
        keep_exact = {
            "perf_event_paranoid",
            "sched_latency_ns",
            "sched_min_granularity_ns",
            "sched_wakeup_granularity_ns",
            "sched_child_runs_first",
            "sched_autogroup_enabled",
            "sched_tunable_scaling",
            "numa_balancing",
            "l1d_total_kib",
            "l1i_total_kib",
            "l2_total_kib",
            "l3_total_kib",
            "cache_line_size_avg",
            "numa_mem_total_mb",
            "rotational_devices_ratio",
            "cpufreq_governor_hash_mean",
            "memtotal_kb",
            "num_cpus",
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
        }
        return [n in keep_exact for n in feature_names]
    if mode == "no_perf_pmu":
        # Zero PMU-derived counters (dynamic perf.*); keep sysctl perf_event_paranoid.
        return [(n == "perf_event_paranoid") or (not n.startswith("perf_")) for n in feature_names]
    if mode == "no_dynamic_proc":
        dyn = {
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
        }
        gpu_dyn = {
            "gpu0_utilization_gpu_pct",
            "gpu0_utilization_memory_pct",
            "gpu0_memory_free_mib",
            "gpu0_memory_used_mib",
            "gpu_min_memory_free_mib",
            "gpu0_power_draw_w",
            "gpu0_temperature_gpu_c",
            "gpu0_clock_current_sm_mhz",
            "gpu0_clock_current_memory_mhz",
        }
        return [n not in dyn and n not in gpu_dyn for n in feature_names]
    if mode == "no_gpu":
        return [not (n.startswith("gpu") or n.startswith("opencl")) for n in feature_names]
    raise ValueError(
        f"Unknown xi ablation mode {mode!r}; choose from: "
        "full, static_hw_only, no_perf_pmu, no_dynamic_proc, no_gpu"
    )


def ablate_xi_vector(vec: list[float], feature_names: list[str], mode: str) -> list[float]:
    if len(vec) != len(feature_names):
        raise ValueError("vec length must match feature_names")
    keep = _keep_mask_for_mode(feature_names, mode)
    return [float(v) if keep[i] else 0.0 for i, v in enumerate(vec)]


class AblatedXiVectorizer:
    """Wraps XiVectorizer and applies an ablation mode on transform()."""

    def __init__(self, mode: str, base: XiVectorizer | None = None) -> None:
        self._base = base or XiVectorizer()
        self.mode = mode

    @property
    def feature_names(self) -> list[str]:
        return list(self._base.feature_names)

    def transform(self, xi: dict[str, Any]) -> list[float]:
        v = self._base.transform(xi)
        return ablate_xi_vector(v, self.feature_names, self.mode)
