"""GPU telemetry for xi: NVIDIA (nvidia-smi) + brief OpenCL inventory (clinfo).

Designed for PTS GPU/OpenCL suites (e.g. ``pts/nvidia-gpu-compute``): complements CPU-centric
``static_features`` / ``dynamic_features`` with discrete-GPU identity, clocks, PCIe, VRAM,
and instantaneous utilization snapshot.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
from typing import Any


def _run_cmd(
    cmd: list[str],
    *,
    timeout: float = 25.0,
) -> tuple[str, int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (p.stdout or ""), p.returncode, (p.stderr or "")
    except FileNotFoundError:
        return "", 127, "command not found"
    except subprocess.TimeoutExpired:
        return "", 124, "timeout"


def _strip_units(cell: str) -> str:
    s = str(cell).strip().strip('"')
    # "24564 MiB" -> "24564", "2130 MHz" -> "2130"
    s = re.sub(r"\s*(MiB|MB|MHz|W)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _parse_float(cell: str | None) -> float | None:
    if cell is None:
        return None
    s = _strip_units(cell)
    if not s or s.lower() in ("not supported", "[n/a]", "n/a", "unknown"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_csv_rows(stdout: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse nvidia-smi CSV with header row."""
    raw = stdout.strip()
    if not raw:
        return [], []
    f = io.StringIO(raw)
    rdr = csv.reader(f)
    rows = list(rdr)
    if not rows:
        return [], []
    header = [h.strip() for h in rows[0]]
    out: list[dict[str, str]] = []
    for row in rows[1:]:
        if not row or all(not c.strip() for c in row):
            continue
        # pad row to header length
        while len(row) < len(header):
            row.append("")
        rec = {header[i]: row[i] for i in range(len(header))}
        out.append(rec)
    return header, out


def collect_opencl_summary() -> dict[str, Any]:
    """Lightweight OpenCL inventory via ``clinfo -l`` (no dependency on JSON output)."""
    out, code, err = _run_cmd(["clinfo", "-l"], timeout=15.0)
    if code != 0 or not out.strip():
        return {
            "available": False,
            "returncode": code,
            "stderr": err.strip() or None,
            "text": None,
            "platform_count": None,
            "device_count": None,
        }
    platform_count = len(re.findall(r"^\s*Platform\s+#", out, re.MULTILINE))
    device_count = len(re.findall(r"^\s*(?:\+--|`--)\s*Device\s+#", out, re.MULTILINE))
    return {
        "available": True,
        "returncode": code,
        "stderr": err.strip() or None,
        "text": out.strip(),
        "platform_count": platform_count,
        "device_count": device_count,
    }


_NVIDIA_STATIC_FIELDS = (
    "index,name,uuid,driver_version,vbios_version,"
    "memory.total,"
    "pcie.link.gen.max,pcie.link.gen.current,"
    "pcie.link.width.max,pcie.link.width.current,"
    "clocks.max.graphics,clocks.max.sm,clocks.max.memory,"
    "enforced.power.limit,power.default_limit,"
    "persistence_mode,compute_mode"
)


_NVIDIA_DYNAMIC_FIELDS = (
    "index,memory.total,memory.free,memory.used,"
    "utilization.gpu,utilization.memory,"
    "temperature.gpu,temperature.memory,"
    "power.draw,"
    "clocks.current.graphics,clocks.current.sm,clocks.current.memory,"
    "pcie.link.gen.current,pcie.link.width.current,"
    "encoder.stats.sessionCount,fan.speed"
)


def _normalize_csv_header(key: str) -> str:
    """Strip CSV units/brackets so ``memory.total [MiB]`` -> ``memory.total``."""
    k = key.strip().strip('"')
    k = re.sub(r"\s*\[[^\]]*\]", "", k)
    return k.strip()


def _record_by_normalized_keys(rec: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in rec.items():
        nk = _normalize_csv_header(k)
        out[nk.lower()] = v
    return out


def _query_nvidia(fields: str) -> dict[str, Any]:
    stdout, code, stderr = _run_cmd(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
        timeout=30.0,
    )
    header, rows = _parse_csv_rows(stdout)
    gpus: list[dict[str, Any]] = []
    for rec in rows:
        r = _record_by_normalized_keys(rec)

        def g(key: str) -> str:
            return (r.get(key.lower()) or "").strip()

        idx = _parse_float(g("index"))
        gpus.append(
            {
                "index": int(idx) if idx is not None else 0,
                "raw": rec,
                "name": g("name"),
                "uuid": g("uuid"),
                "driver_version": g("driver_version"),
                "vbios_version": g("vbios_version"),
                "memory_total_mib": _parse_float(g("memory.total")),
                "pcie_gen_max": _parse_float(g("pcie.link.gen.max")),
                "pcie_gen_current": _parse_float(g("pcie.link.gen.current")),
                "pcie_width_max": _parse_float(g("pcie.link.width.max")),
                "pcie_width_current": _parse_float(g("pcie.link.width.current")),
                "clock_max_graphics_mhz": _parse_float(g("clocks.max.graphics")),
                "clock_max_sm_mhz": _parse_float(g("clocks.max.sm")),
                "clock_max_memory_mhz": _parse_float(g("clocks.max.memory")),
                "power_limit_w": _parse_float(g("enforced.power.limit")),
                "power_default_limit_w": _parse_float(g("power.default_limit")),
                "persistence_mode": g("persistence_mode"),
                "compute_mode": g("compute_mode"),
                "memory_free_mib": _parse_float(g("memory.free")),
                "memory_used_mib": _parse_float(g("memory.used")),
                "utilization_gpu_pct": _parse_float(g("utilization.gpu")),
                "utilization_mem_pct": _parse_float(g("utilization.memory")),
                "temperature_gpu_c": _parse_float(g("temperature.gpu")),
                "temperature_memory_c": _parse_float(g("temperature.memory")),
                "power_draw_w": _parse_float(g("power.draw")),
                "clock_current_graphics_mhz": _parse_float(g("clocks.current.graphics")),
                "clock_current_sm_mhz": _parse_float(g("clocks.current.sm")),
                "clock_current_memory_mhz": _parse_float(g("clocks.current.memory")),
                "encoder_sessions": _parse_float(g("encoder.stats.sessioncount")),
                "fan_speed_pct": _parse_float(g("fan.speed")),
            }
        )
    return {
        "available": code == 0 and len(gpus) > 0,
        "returncode": code,
        "stderr": stderr.strip()[:4000] or None,
        "gpus": gpus,
        "csv_headers": header,
    }


def collect_gpu_static() -> dict[str, Any]:
    """NVIDIA inventory (per-GPU) + OpenCL device/plaform counts. Best-effort if tools missing."""
    nvidia = _query_nvidia(_NVIDIA_STATIC_FIELDS)
    opencl = collect_opencl_summary()
    return {"nvidia": nvidia, "opencl": opencl}


def collect_gpu_dynamic() -> dict[str, Any]:
    """Snapshot under load-experiment conditions: utilization, clocks, VRAM use, thermals."""
    return {"nvidia": _query_nvidia(_NVIDIA_DYNAMIC_FIELDS)}


__all__ = ["collect_gpu_static", "collect_gpu_dynamic", "collect_opencl_summary"]
