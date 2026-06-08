#!/usr/bin/env python3
"""Merge per-machine UnixBench resource waveform panels into one 2×3 paper figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import machine_config_label, machine_experiments_dir
from moebench.monitoring.plot_waveforms import plot_waveform_multimachine_grid

DEFAULT_MACHINES = (
    "aces-System-Product-Name",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp1acaw5wdllhz47922rZ",
)

DEFAULT_WAVEFORM_DIR = REPO_ROOT / "paper" / "waveforms"
DEFAULT_OUT = REPO_ROOT / "paper" / "images" / "resource_waveforms_2x3.png"


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


def _column_spec(
    machine: str,
    *,
    dataset_root: Path,
    waveform_dir: Path,
    prefer_png: bool,
) -> dict:
    label = machine_config_label(machine) or machine
    cpu_image = waveform_dir / f"{machine}_waveform_cpu.png"
    mem_image = waveform_dir / f"{machine}_waveform_memory.png"
    has_png = cpu_image.is_file() and mem_image.is_file()

    json_path = (
        machine_experiments_dir(dataset_root, machine) / "ub_resource_waveforms" / "resource_waveforms.json"
    )

    if prefer_png and has_png:
        return {
            "label": label,
            "cpu_image": cpu_image,
            "memory_image": mem_image,
        }

    if json_path.is_file():
        return {"label": label, "traces": _load_traces(json_path)}

    if has_png:
        return {
            "label": label,
            "cpu_image": cpu_image,
            "memory_image": mem_image,
        }

    raise FileNotFoundError(
        f"No waveform PNG pair or JSON for {machine!r} "
        f"(expected {cpu_image} / {mem_image} or {json_path})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--machines",
        type=str,
        default=",".join(DEFAULT_MACHINES),
        help="Comma-separated hostname slugs (columns left→right)",
    )
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument(
        "--waveform-dir",
        type=str,
        default=str(DEFAULT_WAVEFORM_DIR),
        help="Fallback PNG directory ({machine}_waveform_cpu/memory.png)",
    )
    ap.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--prefer-json",
        action="store_true",
        help="Plot from resource_waveforms.json when available (default: merge existing PNGs)",
    )
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    dataset_root = Path(args.dataset_root).resolve()
    waveform_dir = Path(args.waveform_dir).resolve()
    out_path = Path(args.output).resolve()

    columns = [
        _column_spec(
            m,
            dataset_root=dataset_root,
            waveform_dir=waveform_dir,
            prefer_png=not args.prefer_json,
        )
        for m in machines
    ]
    saved = plot_waveform_multimachine_grid(
        columns,
        out_path=out_path,
        auto_install=args.auto_install,
        crop_image_title_frac=0.0 if not args.prefer_json else 0.11,
    )
    print(f"Wrote {saved}")
    pdf = saved.with_suffix(".pdf")
    if pdf.is_file():
        print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
