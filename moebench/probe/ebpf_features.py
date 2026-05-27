"""eBPF probes (bpftrace) during a timed observation window."""

from __future__ import annotations

import re
import subprocess
import textwrap
from typing import Any


def collect_ebpf_window(duration_s: float) -> dict[str, Any]:
    """
    Count sched_switch and syscall tracepoints over ``duration_s`` seconds.

    Requires bpftrace in PATH (often root). Returns rates (counts / duration).
    """
    duration_s = max(1.0, float(duration_s))
    which = subprocess.run(["which", "bpftrace"], capture_output=True, text=True, check=False)
    if which.returncode != 0 or not (which.stdout or "").strip():
        return {"available": False, "reason": "bpftrace not in PATH"}

    sec = int(max(1, round(duration_s)))
    prog = textwrap.dedent(
        f"""
        tracepoint:sched:sched_switch {{ @sw++; }}
        tracepoint:raw_syscalls:sys_enter {{ @sc++; }}
        interval:s:{sec} {{
            printf("sched_switch_count %llu\\n", @sw);
            printf("syscall_enter_count %llu\\n", @sc);
            exit();
        }}
        """
    )
    p = subprocess.run(
        ["bpftrace", "-e", prog],
        capture_output=True,
        text=True,
        timeout=duration_s + 8.0,
        check=False,
    )
    out = p.stdout or ""
    sw = _parse_count(out, "sched_switch_count")
    sc = _parse_count(out, "syscall_enter_count")
    return {
        "available": True,
        "returncode": p.returncode,
        "duration_s": duration_s,
        "sched_switch_count": sw,
        "syscall_enter_count": sc,
        "sched_switch_per_s": (float(sw) / duration_s) if sw is not None else None,
        "syscall_enter_per_s": (float(sc) / duration_s) if sc is not None else None,
        "stderr": (p.stderr or "")[:4000] if p.stderr else None,
    }


def _parse_count(text: str, key: str) -> int | None:
    m = re.search(rf"{re.escape(key)}\s+(\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
