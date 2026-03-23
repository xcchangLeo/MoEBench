"""Parse `perf stat` text output into structured counters."""

from __future__ import annotations

import re
from typing import Any


def parse_perf_stat(stdout: str, stderr: str) -> dict[str, Any]:
    """
    Aggregate perf stat output (multiplexed or not).
    Lines look like: 1,234,567      cycles (66.12%)  or  with <not supported>
    """
    text = stdout + "\n" + stderr
    counters: dict[str, dict[str, Any]] = {}
    # value unit name optional remainder
    pat = re.compile(
        r"^\s*([\d,]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?\s*$",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        raw_val, unit, name, rest = m.groups()
        val_s = raw_val.replace(",", "")
        try:
            val = float(val_s) if "." in val_s else int(val_s)
        except ValueError:
            val = raw_val
        entry = {"value": val, "unit": unit}
        if rest:
            entry["note"] = rest.strip()
        counters[name] = entry

    # Some perf versions print "not counted" / "not supported"
    not_supported = "not supported" in text.lower() or "not counted" in text.lower()
    return {"counters": counters, "perf_not_supported_hint": not_supported}


def derive_ipc_and_rates(counters: dict[str, dict[str, Any]], duration_s: float) -> dict[str, float | None]:
    """IPC and per-second rates when instructions/cycles are present."""
    def get_val(name: str) -> float | None:
        c = counters.get(name)
        if not c:
            return None
        v = c.get("value")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    inst = get_val("instructions")
    cyc = get_val("cycles")
    ipc = (inst / cyc) if inst is not None and cyc and cyc > 0 else None

    def per_sec(name: str) -> float | None:
        v = get_val(name)
        if v is None or duration_s <= 0:
            return None
        return v / duration_s

    return {
        "ipc": ipc,
        "instructions_per_sec": per_sec("instructions"),
        "cycles_per_sec": per_sec("cycles"),
        "context_switches_per_sec": per_sec("context-switches"),
        "cpu_migrations_per_sec": per_sec("cpu-migrations"),
        "page_faults_per_sec": per_sec("page-faults"),
        "cache_references_per_sec": per_sec("cache-references"),
        "cache_misses_per_sec": per_sec("cache-misses"),
        "branch_misses_per_sec": per_sec("branch-misses"),
    }
