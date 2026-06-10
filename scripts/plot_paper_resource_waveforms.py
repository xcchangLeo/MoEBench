#!/usr/bin/env python3
"""Plot paper 2×3 UnixBench resource waveform figure from v3 JSON data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import machine_config_label, machine_experiments_dir
from moebench.monitoring.plot_waveforms import (
    PAPER_INSET_ZOOM_END_S,
    plot_waveform_compare_pair,
    plot_waveform_paper_2x3_grid,
)

DEFAULT_MACHINES = (
    "aces-System-Product-Name",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp16krl0yc7euw7sb6slZ",
)

WAVEFORM_SUBDIR = "ub_resource_waveforms_v3"
DEFAULT_OUT = REPO_ROOT / "paper" / "images" / "resource_waveforms_2x3.png"
DEFAULT_COMBINE_OUT = REPO_ROOT / "paper" / "images" / "resource_waveforms_2x3_combine.png"

CONFIG_BASENAME = {
    "aces-System-Product-Name": "32U128G",
    "iZbp15n87643uk1sqjrdvdZ": "4U8G",
    "iZbp16krl0yc7euw7sb6slZ": "4U16G",
}


def _load_traces(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)
    traces = list(report.get("traces") or [])
    out_dir = json_path.parent
    by_mode: dict[str, dict] = {str(t.get("mode") or ""): t for t in traces if t.get("mode")}
    for mode in ("full", "route_a", "route_b", "benchscout"):
        if mode in by_mode:
            continue
        sidecar = out_dir / f"trace_{mode}.json"
        if sidecar.is_file():
            with open(sidecar, encoding="utf-8") as f:
                by_mode[mode] = json.load(f)
    order = {"full": 0, "route_a": 1, "route_b": 2, "benchscout": 3}
    return [by_mode[m] for m in sorted(by_mode, key=lambda k: order.get(k, 99))]


def _json_path(dataset_root: Path, machine: str) -> Path:
    return machine_experiments_dir(dataset_root, machine) / WAVEFORM_SUBDIR / "resource_waveforms.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--machines",
        type=str,
        default=",".join(DEFAULT_MACHINES),
        help="Comma-separated hostname slugs (columns left→right)",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--also-per-machine",
        action="store_true",
        help="Also write waveform_{config}_cpu/memory.png under paper/images",
    )
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument(
        "--xlim-max",
        type=float,
        default=0.0,
        help="Shared x-axis upper bound in seconds (0 = full trace, e.g. full-run ~1680s)",
    )
    ap.add_argument(
        "--combine-inset",
        action="store_true",
        help="Add 12s magnifier inset per panel; default output resource_waveforms_2x3_combine",
    )
    ap.add_argument(
        "--inset-zoom-end",
        type=float,
        default=PAPER_INSET_ZOOM_END_S,
        help="Inset x-axis upper bound in seconds (default: BenchScout wall ~12s)",
    )
    args = ap.parse_args()

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    dataset_root = Path(args.dataset_root).resolve()
    out_path = Path(args.output).resolve()
    if args.combine_inset and args.output == str(DEFAULT_OUT):
        out_path = DEFAULT_COMBINE_OUT.resolve()
    images_dir = out_path.parent

    columns: list[dict] = []
    for machine in machines:
        jp = _json_path(dataset_root, machine)
        if not jp.is_file():
            print(f"Missing {jp}", file=sys.stderr)
            return 2
        label = machine_config_label(machine) or CONFIG_BASENAME.get(machine) or machine
        columns.append({"label": label, "traces": _load_traces(jp)})

    saved = plot_waveform_paper_2x3_grid(
        columns,
        out_path=out_path,
        auto_install=args.auto_install,
        xlim_max=args.xlim_max if args.xlim_max > 0 else None,
        inset_zoom_end_s=args.inset_zoom_end if args.combine_inset else None,
    )
    print(f"Wrote {saved}")
    pdf = saved.with_suffix(".pdf")
    if pdf.is_file():
        print(f"Wrote {pdf}")

    if args.also_per_machine:
        for machine, col in zip(machines, columns):
            stem = CONFIG_BASENAME.get(machine) or col["label"]
            pair = plot_waveform_compare_pair(
                col["traces"],
                out_dir=images_dir,
                title_cpu=f"{stem}, CPU",
                title_mem=f"{stem}, Memory",
                auto_install=args.auto_install,
                cpu_basename=f"waveform_{stem}_cpu.png",
                memory_basename=f"waveform_{stem}_memory.png",
            )
            print(f"Wrote {pair['cpu']}")
            print(f"Wrote {pair['memory']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
