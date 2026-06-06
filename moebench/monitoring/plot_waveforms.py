"""Plot CPU / memory waveforms from resource traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

MODE_ORDER = ("full", "route_a", "route_b", "benchscout")

MODE_DISPLAY_LABELS: dict[str, str] = {
    "full": "Full run",
    "route_a": "router only",
    "route_b": "probe only",
    "benchscout": "BenchScout",
}

PALETTE_FOUR = ["#2563eb", "#16a34a", "#ea580c", "#9333ea"]


def trace_display_label(tr: dict[str, Any]) -> str:
    mode = str(tr.get("mode") or "")
    if mode in MODE_DISPLAY_LABELS:
        return MODE_DISPLAY_LABELS[mode]
    return str(tr.get("label") or mode or "trace")


def sort_traces_for_plot(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    return sorted(traces, key=lambda t: order.get(str(t.get("mode", "")), 99))


def _ensure_matplotlib(auto_install: bool):
    from moebench.pip_install import ensure_importable

    ensure_importable("matplotlib", auto_install=auto_install)
    import matplotlib.pyplot as plt

    return plt


def plot_waveform_grid(
    traces: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str = "UnixBench resource usage",
    auto_install: bool = False,
) -> Path:
    """One row CPU, one row memory; one column per trace (full / route A / route B)."""
    plt = _ensure_matplotlib(auto_install)
    n = len(traces)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6), squeeze=False, sharex="col")

    colors_cpu = "#2563eb"
    colors_mem = "#dc2626"

    for col, tr in enumerate(traces):
        label = trace_display_label(tr)
        samples = tr.get("samples") or []
        if not samples:
            continue
        t = [float(s["t_rel_s"]) for s in samples]
        cpu = [float(s["cpu_pct"]) for s in samples]
        mem = [float(s["mem_used_pct"]) for s in samples]

        ax_cpu = axes[0, col]
        ax_mem = axes[1, col]
        ax_cpu.plot(t, cpu, color=colors_cpu, linewidth=1.2)
        ax_cpu.set_ylim(0, 100)
        ax_cpu.set_ylabel("CPU (%)")
        ax_cpu.set_title(label)
        ax_cpu.grid(True, alpha=0.3)
        wall = tr.get("wall_s")
        if wall is not None:
            ax_cpu.text(
                0.02,
                0.95,
                f"wall {float(wall):.0f}s",
                transform=ax_cpu.transAxes,
                va="top",
                fontsize=9,
            )

        ax_mem.plot(t, mem, color=colors_mem, linewidth=1.2)
        ax_mem.set_ylim(0, 100)
        ax_mem.set_ylabel("Memory used (%)")
        ax_mem.set_xlabel("Time (s)")
        ax_mem.grid(True, alpha=0.3)

        for ev in tr.get("phase_markers") or []:
            t_mark = float(ev.get("t_rel_s", 0))
            name = str(ev.get("name", ""))
            for ax in (ax_cpu, ax_mem):
                ax.axvline(t_mark, color="#6b7280", linestyle="--", alpha=0.35, linewidth=0.8)
            ax_cpu.text(t_mark, 2, name, rotation=90, fontsize=7, alpha=0.7)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_waveform_overlay(
    traces: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str = "UnixBench resource usage (overlay)",
    auto_install: bool = False,
) -> Path:
    """Overlay traces on shared axes (each starts at t=0)."""
    plt = _ensure_matplotlib(auto_install)
    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    traces = sort_traces_for_plot(traces)

    for i, tr in enumerate(traces):
        label = trace_display_label(tr)
        color = PALETTE_FOUR[i % len(PALETTE_FOUR)]
        samples = tr.get("samples") or []
        if not samples:
            continue
        t = [float(s["t_rel_s"]) for s in samples]
        cpu = [float(s["cpu_pct"]) for s in samples]
        mem = [float(s["mem_used_pct"]) for s in samples]
        ax_cpu.plot(t, cpu, label=label, color=color, linewidth=1.2)
        ax_mem.plot(t, mem, label=label, color=color, linewidth=1.2)

    ax_cpu.set_ylabel("CPU (%)")
    ax_cpu.set_ylim(0, 100)
    ax_cpu.legend(loc="upper right")
    ax_cpu.grid(True, alpha=0.3)
    ax_mem.set_ylabel("Memory used (%)")
    ax_mem.set_xlabel("Time (s)")
    ax_mem.set_ylim(0, 100)
    ax_mem.legend(loc="upper right")
    ax_mem.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_waveform_single_metric(
    traces: list[dict[str, Any]],
    *,
    metric_key: Literal["cpu_pct", "mem_used_pct"],
    ylabel: str,
    title: str,
    out_path: Path,
    auto_install: bool = False,
    save_pdf: bool = True,
) -> Path:
    """One figure with all traces overlaid for a single metric."""
    plt = _ensure_matplotlib(auto_install)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    traces = sort_traces_for_plot(traces)

    for i, tr in enumerate(traces):
        label = trace_display_label(tr)
        color = PALETTE_FOUR[i % len(PALETTE_FOUR)]
        samples = tr.get("samples") or []
        if not samples:
            continue
        t = [float(s["t_rel_s"]) for s in samples]
        y = [float(s[metric_key]) for s in samples]
        ax.plot(t, y, label=label, color=color, linewidth=1.2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_waveform_compare_pair(
    traces: list[dict[str, Any]],
    *,
    out_dir: Path,
    title_cpu: str = "UnixBench CPU usage",
    title_mem: str = "UnixBench memory usage",
    auto_install: bool = False,
    save_pdf: bool = True,
    cpu_basename: str = "resource_waveforms_cpu.png",
    memory_basename: str = "resource_waveforms_memory.png",
) -> dict[str, Path]:
    """Separate CPU and memory overlay figures (paper-style four-curve comparison)."""
    out_dir = Path(out_dir)
    cpu_path = _plot_waveform_single_metric(
        traces,
        metric_key="cpu_pct",
        ylabel="CPU (%)",
        title=title_cpu,
        out_path=out_dir / cpu_basename,
        auto_install=auto_install,
        save_pdf=save_pdf,
    )
    mem_path = _plot_waveform_single_metric(
        traces,
        metric_key="mem_used_pct",
        ylabel="Memory used (%)",
        title=title_mem,
        out_path=out_dir / memory_basename,
        auto_install=auto_install,
        save_pdf=save_pdf,
    )
    return {"cpu": cpu_path, "memory": mem_path}
