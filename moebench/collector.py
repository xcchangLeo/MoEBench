"""High-level API: static + dynamic feature bundles."""

from __future__ import annotations

from typing import Any

from moebench.dynamic_features import collect_dynamic
from moebench.static_features import collect_static


def collect_all(
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    enable_ebpf: bool = True,
    mem_mb: int = 64,
) -> dict[str, Any]:
    """
    Collect static system description and dynamic warmup/perf snapshot.

    The temporary warmup script is deleted inside ``collect_dynamic``.
    """
    static = collect_static()
    dynamic = collect_dynamic(
        warmup_s=warmup_s,
        proc_sample_s=proc_sample_s,
        enable_ebpf=enable_ebpf,
        mem_mb=mem_mb,
    )

    return {
        "static": static,
        "dynamic": dynamic,
    }


__all__ = ["collect_all", "collect_static", "collect_dynamic"]
