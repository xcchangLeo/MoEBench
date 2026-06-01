#!/usr/bin/env python3
"""
Motivation figure: UnixBench iterative subtest performance convergence curves.

For Dhrystone (``dhry2reg``) and Whetstone (``whetstone-double``), run the official
UnixBench binaries with increasing measurement windows (1..10 s, step 1 s). At each
window length ``t``, the reported cumulative operation rate approximates the rate
observed at time ``t`` within the default 10 s UnixBench pass.

Outputs (default under ``paper/``):
  - unixbench_convergence_curves.pdf
  - unixbench_convergence_curves.png
  - unixbench_convergence_curves_meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moebench.unixbench.pipeline import _default_unixbench_root

_COUNT_RE = re.compile(r"COUNT\|([0-9.]+)")
_TIME_RE = re.compile(r"TIME\|([0-9.]+)")

# UnixBench system index defaults (Run).
DHRY_OFFICIAL_TIMEOUT_S = 10
WHET_OFFICIAL_TIMEOUT_S = 10
PROBE_CUTOFF_S = 4.0


def _parse_count_output(text: str) -> tuple[float | None, float | None]:
    m = _COUNT_RE.search(text)
    count = float(m.group(1)) if m else None
    t = _TIME_RE.search(text)
    elapsed = float(t.group(1)) if t else None
    return count, elapsed


def run_dhry2reg(ub_root: Path, duration_s: int) -> dict:
    bin_path = ub_root / "pgms" / "dhry2reg"
    p = subprocess.run(
        [str(bin_path), str(duration_s)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ub_root),
    )
    text = (p.stdout or "") + (p.stderr or "")
    count, _elapsed = _parse_count_output(text)
    rate = (count / duration_s) if count is not None else None
    return {
        "test_id": "dhry2reg",
        "duration_s": duration_s,
        "count": count,
        "rate_lps": rate,
        "returncode": p.returncode,
    }


def ensure_whetstone_duration_binary(ub_root: Path, cache_dir: Path) -> Path:
    """Build whetstone-double with optional argv[1] duration (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_bin = cache_dir / "whetstone-double-duration"
    src_orig = ub_root / "src" / "whets.c"
    stamp = cache_dir / "whets.duration.patch.stamp"
    if out_bin.is_file() and stamp.is_file() and stamp.stat().st_mtime >= src_orig.stat().st_mtime:
        return out_bin

    text = src_orig.read_text(encoding="utf-8")
    anchor = "#ifdef UNIXBENCH\n    int duration = 10;"
    patch = (
        "#ifdef UNIXBENCH\n"
        "    int duration = 10;\n"
        "    if (argc > 1) { duration = atoi(argv[1]); if (duration < 1) duration = 1; }"
    )
    if anchor not in text:
        raise RuntimeError(f"Cannot patch {src_orig} for duration CLI")
    patched = text.replace(anchor, patch, 1)

    work = Path(tempfile.mkdtemp(prefix="moebench_whet_"))
    try:
        shutil.copytree(ub_root / "src", work / "src")
        (work / "src" / "whets.c").write_text(patched, encoding="utf-8")
        subprocess.check_call(
            [
                "gcc",
                "-O3",
                "-DDP",
                "-DGTODay",
                "-DUNIXBENCH",
                str(work / "src" / "whets.c"),
                "-o",
                str(out_bin),
                "-lm",
            ]
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    stamp.write_text("ok\n", encoding="utf-8")
    return out_bin


def run_whetstone(ub_root: Path, whet_bin: Path, duration_s: int) -> dict:
    p = subprocess.run(
        [str(whet_bin), str(duration_s)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ub_root),
    )
    text = (p.stdout or "") + (p.stderr or "")
    mwips, elapsed = _parse_count_output(text)
    return {
        "test_id": "whetstone-double",
        "duration_s": duration_s,
        "mwips": mwips,
        "elapsed_s": elapsed,
        "returncode": p.returncode,
    }


def sample_convergence(
    ub_root: Path,
    *,
    whet_bin: Path,
    time_points: list[int],
    repeats: int,
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"dhry2reg": [], "whetstone-double": []}
    for t in time_points:
        for rep in range(repeats):
            d = run_dhry2reg(ub_root, t)
            d["repeat"] = rep
            out["dhry2reg"].append(d)
            w = run_whetstone(ub_root, whet_bin, t)
            w["repeat"] = rep
            out["whetstone-double"].append(w)
    return out


def _aggregate_series(records: list[dict], *, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_t: dict[int, list[float]] = {}
    for rec in records:
        y = rec.get(y_key)
        if y is None:
            continue
        by_t.setdefault(int(rec["duration_s"]), []).append(float(y))
    times = sorted(by_t)
    means = np.array([float(np.mean(by_t[t])) for t in times])
    stds = np.array([float(np.std(by_t[t])) if len(by_t[t]) > 1 else 0.0 for t in times])
    return np.array(times, dtype=float), means, stds


def plot_convergence(
    series: dict[str, list[dict]],
    *,
    out_base: Path,
    probe_cutoff_s: float,
    official_timeout_s: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=150, sharex=True)

    panels = [
        (
            axes[0],
            "dhry2reg",
            "rate_lps",
            "Dhrystone 2 (register variables)",
            "Throughput (Mlps)",
            1e6,
        ),
        (
            axes[1],
            "whetstone-double",
            "mwips",
            "Double-Precision Whetstone",
            "Throughput (kMWIPS)",
            1e3,
        ),
    ]

    for ax, test_id, y_key, title, ylabel, scale in panels:
        times, means, stds = _aggregate_series(series[test_id], y_key=y_key)
        y_plot = means / scale
        y_err = stds / scale
        ax.plot(times, y_plot, marker="o", linewidth=1.8, markersize=4.5, color="#1f77b4")
        if np.any(y_err > 0):
            ax.fill_between(times, y_plot - y_err, y_plot + y_err, alpha=0.18, color="#1f77b4")

        ax.axvline(
            probe_cutoff_s,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Probing Cutoff",
        )
        ax.axvline(
            official_timeout_s,
            color="#444444",
            linestyle="--",
            linewidth=1.5,
            label="Official Timeout",
        )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Execution time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xlim(left=0, right=official_timeout_s + 0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "UnixBench iterative subtests: cumulative performance metric vs. elapsed time",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf}", file=sys.stderr)
    print(f"Wrote {png}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unixbench-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "paper")
    ap.add_argument(
        "--out-base",
        type=Path,
        default=None,
        help="Basename without extension (default: paper/unixbench_convergence_curves)",
    )
    ap.add_argument("--sample-step-s", type=int, default=1, help="Sample every N seconds (default: 1)")
    ap.add_argument("--max-time-s", type=int, default=10, help="Official timeout / max x (default: 10)")
    ap.add_argument("--probe-cutoff-s", type=float, default=PROBE_CUTOFF_S)
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per time point (default: 3)")
    ap.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "dataset" / ".motivation_cache")
    args = ap.parse_args()

    ub_root = (args.unixbench_root or _default_unixbench_root()).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = (args.out_base or out_dir / "unixbench_convergence_curves").resolve()

    step = max(1, int(args.sample_step_s))
    max_t = max(1, int(args.max_time_s))
    time_points = list(range(step, max_t + 1, step))

    whet_bin = ensure_whetstone_duration_binary(ub_root, args.cache_dir.resolve())
    series = sample_convergence(ub_root, whet_bin=whet_bin, time_points=time_points, repeats=max(1, args.repeats))

    meta = {
        "schema": "moebench.paper.unixbench_convergence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unixbench_root": str(ub_root),
        "method": (
            "Increasing-duration official UnixBench binaries; at each window t, "
            "cumulative rate = dhry COUNT/t or whetstone MWIPS (COUNT field)."
        ),
        "sample_step_s": step,
        "max_time_s": max_t,
        "probe_cutoff_s": float(args.probe_cutoff_s),
        "official_timeout_s": float(max_t),
        "repeats_per_time_point": int(args.repeats),
        "time_points_s": time_points,
        "series": series,
    }
    meta_path = out_base.with_name(out_base.name + "_meta").with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {meta_path}", file=sys.stderr)

    plot_convergence(
        series,
        out_base=out_base,
        probe_cutoff_s=float(args.probe_cutoff_s),
        official_timeout_s=float(max_t),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
