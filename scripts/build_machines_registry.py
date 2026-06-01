#!/usr/bin/env python3
"""Build dataset/machines_registry.json from collected session run JSONs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_machine_from_dirname(name: str) -> str:
    m = re.match(r"^(aces-System-Product-Name|iZbp[a-zA-Z0-9]+)", name)
    return m.group(1) if m else name.split("_", 1)[0]


def parse_cpus(lscpu_text: str, cpuinfo_text: str = "") -> int | None:
    for pat in (
        r"^CPU\(s\):\s+(\d+)",
        r"^CPU:\s+(\d+)",
        r"^CPU 数：\s+(\d+)",
    ):
        m = re.search(pat, lscpu_text, re.M)
        if m:
            return int(m.group(1))
    if cpuinfo_text:
        n = len(re.findall(r"^processor\t:", cpuinfo_text, re.M))
        if n:
            return n
    return None


def parse_mem_gib(meminfo: str) -> float | None:
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
    if not m:
        return None
    return round(int(m.group(1)) / 1024 / 1024, 2)


def has_gpu(static: dict) -> bool:
    gpu = static.get("gpu") or {}
    nvidia = gpu.get("nvidia") or {}
    if isinstance(nvidia, dict) and nvidia.get("available"):
        return True
    for key in ("nvidia_smi", "opencl"):
        block = gpu.get(key) or {}
        if isinstance(block, dict) and (block.get("text") or "").strip():
            return True
    return False


def nominal_mem_gib(measured: float) -> int:
    if measured < 12:
        return 8
    if measured < 24:
        return 16
    return 128


def paper_host_id(vcpus: int, mem_nominal_gib: int) -> str:
    mapping = {
        (32, 128): "H1",
        (2, 8): "H2",
        (4, 8): "H3",
        (4, 16): "H4",
        (8, 8): "H5",
    }
    return mapping.get((vcpus, mem_nominal_gib), "?")


def suite_from_session_dir(name: str) -> str:
    if "_cpu_" in name:
        return "pts-cpu"
    if "gpu" in name.lower() or "nvidia" in name.lower():
        return "pts-gpu"
    return "unixbench"


def build_registry(dataset_root: Path) -> dict:
    machines: dict[str, dict] = {}

    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir() or p.name in ("experiments", "models", "paper_supplementary"):
            continue
        machine = parse_machine_from_dirname(p.name)
        run_files = sorted(rf for rf in p.glob("run-*.json") if "_pts_raw" not in rf.name)
        if not run_files:
            continue

        with open(run_files[0], encoding="utf-8") as f:
            d = json.load(f)
        static = d.get("xi", {}).get("static", {})
        vcpus = parse_cpus(
            static.get("lscpu", {}).get("text", ""),
            static.get("cpuinfo", {}).get("text", ""),
        )
        mem_gib = parse_mem_gib(static.get("memory", {}).get("meminfo", ""))
        gpu = has_gpu(static)

        rec = machines.setdefault(
            machine,
            {
                "hostname_slug": machine,
                "vcpus": vcpus,
                "mem_total_gib_measured": mem_gib,
                "gpu": gpu,
                "session_dirs": [],
                "suites": set(),
            },
        )
        rec["session_dirs"].append(p.name)
        rec["suites"].add(suite_from_session_dir(p.name))
        if vcpus is not None:
            rec["vcpus"] = vcpus
        if mem_gib is not None:
            rec["mem_total_gib_measured"] = mem_gib
        rec["gpu"] = rec["gpu"] or gpu

    entries = []
    for machine, rec in sorted(machines.items()):
        vcpus = rec["vcpus"] or 0
        mem_measured = rec["mem_total_gib_measured"] or 0.0
        mem_nominal = nominal_mem_gib(mem_measured)
        entries.append(
            {
                "hostname_slug": machine,
                "paper_host_id": paper_host_id(vcpus, mem_nominal),
                "config_label": f"{vcpus}U{mem_nominal}G",
                "vcpus": vcpus,
                "mem_nominal_gib": mem_nominal,
                "mem_total_gib_measured": mem_measured,
                "gpu": rec["gpu"],
                "suites": sorted(rec["suites"]),
                "session_dirs": sorted(rec["session_dirs"]),
                "models_dir": f"dataset/models/{machine}/",
                "experiments_dir": f"dataset/experiments/{machine}/",
                "has_models_dir": (dataset_root / "models" / machine).exists(),
                "has_experiments_dir": (dataset_root / "experiments" / machine).exists(),
            }
        )

    return {
        "schema": "moebench.machines_registry.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "dataset session run-01.json xi.static",
        "machines": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=_REPO_ROOT / "dataset",
        help="MoEBench dataset root (default: repo dataset/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <dataset-root>/machines_registry.json)",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    out = (args.out or dataset_root / "machines_registry.json").resolve()
    registry = build_registry(dataset_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {out} ({len(registry['machines'])} machines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
