"""Dynamic features: short warmup + perf stat (primary) + /proc + optional eBPF."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from moebench.perf_parser import derive_ipc_and_rates, parse_perf_stat

# PMU / software events (kernel-dependent; perf will error on unknown names).
DEFAULT_PERF_EVENTS = ",".join(
    [
        "cycles",
        "instructions",
        "cache-references",
        "cache-misses",
        "branch-misses",
        "branch-instructions",
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "major-faults",
        "minor-faults",
    ]
)

REDUCED_PERF_EVENTS = ",".join(
    [
        "task-clock",
        "context-switches",
        "page-faults",
        "major-faults",
        "minor-faults",
    ]
)


def _workload_script_path(warmup_s: float, mem_mb: int) -> str:
    code = textwrap.dedent(
        f"""
        import array, math, time
        end = time.time() + {warmup_s!r}
        n = {mem_mb!r} * 1024 * 1024
        a = array.array("B", (i % 256 for i in range(n)))
        b = array.array("B", [0] * n)
        i = 0
        while time.time() < end:
            b[:] = a
            x = 0.0
            for j in range(5000):
                x += math.sqrt(float(j + 1))
            i += 1
        """
    )
    fd, path = tempfile.mkstemp(suffix="_moebench_warmup.py", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _run_perf_json(
    warmup_s: float,
    events: str,
    workload_path: str,
) -> dict[str, Any]:
    cmd = [
        "perf",
        "stat",
        "-j",
        "-e",
        events,
        "--",
        sys.executable,
        workload_path,
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(30.0, warmup_s + 15.0),
        check=False,
    )
    counters: dict[str, dict[str, Any]] = {}
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = obj.get("event")
        if not ev:
            continue
        cv = obj.get("counter-value")
        unit = obj.get("unit") or ""
        if cv is not None:
            try:
                val = float(str(cv).replace(",", ""))
            except ValueError:
                val = cv
            counters[ev] = {"value": val, "unit": unit}
    parsed = {"counters": counters, "perf_not_supported_hint": False}
    derived = derive_ipc_and_rates(counters, float(warmup_s))
    return {
        "command": cmd,
        "returncode": p.returncode,
        "stderr": (p.stderr or "")[:8000],
        "stdout_lines": len((p.stdout or "").splitlines()),
        "parsed": parsed,
        "derived": derived,
        "duration_s": warmup_s,
    }


def _run_perf_text_fallback(warmup_s: float, workload_path: str) -> dict[str, Any]:
    cmd = ["perf", "stat", "-e", REDUCED_PERF_EVENTS, "--", sys.executable, workload_path]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(30.0, warmup_s + 15.0),
        check=False,
    )
    parsed = parse_perf_stat(p.stdout or "", p.stderr or "")
    derived = derive_ipc_and_rates(parsed["counters"], float(warmup_s))
    return {
        "command": cmd,
        "returncode": p.returncode,
        "stderr": (p.stderr or "")[:8000],
        "parsed": parsed,
        "derived": derived,
        "duration_s": warmup_s,
    }


def _proc_fallback(interval_s: float) -> dict[str, Any]:
    from moebench import proc_metrics

    cpu = proc_metrics.sample_cpu_utilization(interval_s)
    faults = proc_metrics.sample_vmstat_faults(interval_s)
    load = proc_metrics.read_loadavg()
    io = proc_metrics.io_latency_proxy(interval_s)
    mbw = proc_metrics.memory_bandwidth_proxy_mb_s(duration_s=min(0.5, interval_s))
    return {
        "source": "proc_fallback",
        "cpu_utilization": cpu,
        "vmstat_faults": faults,
        "loadavg": load,
        "io_proxy": io,
        "memory_bandwidth_proxy": mbw,
    }


def _bpftrace_sched_switch(duration_s: float) -> dict[str, Any]:
    """Optional: count sched_switch tracepoints (often needs root)."""
    which = subprocess.run(["which", "bpftrace"], capture_output=True, text=True, check=False)
    if which.returncode != 0 or not (which.stdout or "").strip():
        return {"available": False, "reason": "bpftrace not in PATH"}

    # Global hit count; interval probe prints and exits (needs privileges on many systems).
    prog = textwrap.dedent(
        f"""
        tracepoint:sched:sched_switch {{
            @c++;
        }}
        interval:s:{int(max(1, duration_s))} {{
            printf("sched_switch_count %llu\\n", @c);
            exit();
        }}
        """
    )
    p = subprocess.run(
        ["bpftrace", "-e", prog],
        capture_output=True,
        text=True,
        timeout=duration_s + 5.0,
        check=False,
    )
    out = p.stdout or ""
    m = re.search(r"sched_switch_count\s+(\d+)", out)
    count = int(m.group(1)) if m else None
    return {
        "available": True,
        "returncode": p.returncode,
        "sched_switch_samples": count,
        "stderr": (p.stderr or "")[:4000] if p.stderr else None,
    }


def collect_dynamic(
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    enable_ebpf: bool = True,
    mem_mb: int = 64,
) -> dict[str, Any]:
    """
    Run a short CPU+memory warmup, attach perf stat to the workload process.

    If perf is blocked (e.g. kernel.perf_event_paranoid), falls back to /proc-based metrics.

    Parameters
    ----------
    warmup_s : float
        Workload duration (and perf aggregation window).
    proc_sample_s : float
        Interval for /proc delta metrics (fallback or supplement).
    enable_ebpf : bool
        If True, try bpftrace sched_switch counter (best-effort).
    mem_mb : int
        Working set size for the in-process memcpy loop.
    """
    warmup_s = float(warmup_s)
    proc_sample_s = float(proc_sample_s)
    path = _workload_script_path(warmup_s, mem_mb)

    result: dict[str, Any] = {
        "warmup_s": warmup_s,
        "workload_script": path,
        "perf": None,
        "perf_degraded": None,
        "proc": None,
        "ebpf": None,
    }

    perf_res = _run_perf_json(warmup_s, DEFAULT_PERF_EVENTS, path)
    if perf_res["returncode"] != 0 or not perf_res["parsed"]["counters"]:
        perf_res2 = _run_perf_json(warmup_s, REDUCED_PERF_EVENTS, path)
        if perf_res2["parsed"]["counters"]:
            perf_res = perf_res2
        else:
            text_try = _run_perf_text_fallback(warmup_s, path)
            if text_try["parsed"]["counters"]:
                perf_res = text_try
            else:
                perf_res = text_try

    stderr_combined = perf_res.get("stderr") or ""
    paranoid_hint = "perf_event_paranoid" in stderr_combined or "limited" in stderr_combined.lower()

    if not perf_res["parsed"]["counters"]:
        result["perf_degraded"] = {
            "reason": "perf produced no counters (check CAP_PERFMON or kernel.perf_event_paranoid)",
            "hint_paranoid": paranoid_hint,
            "stderr_excerpt": stderr_combined[:4000] or None,
        }
        # perf may refuse to spawn the workload; run the same script so dynamic load exists.
        try:
            subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=max(30.0, warmup_s + 10.0),
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
        result["proc"] = _proc_fallback(proc_sample_s)
    else:
        result["perf"] = perf_res
        from moebench import proc_metrics

        # Keys must match ``proc_fallback`` and ``XiVectorizer``: training data is mostly from
        # degraded (no perf) runs; using ``vmstat_faults_sample`` left page-fault features at 0
        # whenever perf succeeded (common under sudo), hurting router/reconstruct parity.
        result["proc"] = {
            "source": "supplement_after_warmup",
            "cpu_utilization": proc_metrics.sample_cpu_utilization(proc_sample_s),
            "loadavg": proc_metrics.read_loadavg(),
            "vmstat_faults": proc_metrics.sample_vmstat_faults(proc_sample_s),
            "memory_bandwidth_proxy": proc_metrics.memory_bandwidth_proxy_mb_s(
                duration_s=min(0.5, float(proc_sample_s))
            ),
        }

    if enable_ebpf:
        result["ebpf"] = _bpftrace_sched_switch(min(2.0, max(1.0, warmup_s / 2)))

    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass
    result["workload_script"] = None

    return result
