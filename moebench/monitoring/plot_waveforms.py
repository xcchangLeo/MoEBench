"""Plot CPU / memory waveforms from resource traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _ensure_matplotlib(auto_install: bool):
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        if auto_install:
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib.pyplot as plt  # noqa: F401
        else:
            raise
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
        label = str(tr.get("label") or f"mode_{col}")
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
    palette = ["#2563eb", "#16a34a", "#ea580c", "#9333ea"]

    for i, tr in enumerate(traces):
        label = str(tr.get("label") or f"trace_{i}")
        color = palette[i % len(palette)]
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
