"""Static system features: CPU topology, memory, storage, kernel, compiler, power/sched."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from moebench.gpu_features import collect_gpu_static


def _run_text(cmd: list[str], timeout: int = 30) -> tuple[str, int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.stdout or "", p.returncode, p.stderr or ""
    except FileNotFoundError:
        return "", 127, "command not found"
    except subprocess.TimeoutExpired:
        return "", 124, "timeout"


def _read_file(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except OSError:
        return ""


def collect_lscpu() -> dict[str, Any]:
    out, code, err = _run_text(["lscpu"])
    return {"text": out.strip(), "returncode": code, "stderr": err.strip() or None}


def collect_cpuinfo() -> dict[str, Any]:
    return {"text": _read_file("/proc/cpuinfo").strip()}


def collect_cache_hierarchy() -> dict[str, Any]:
    """Per cache index under cpu0 (representative)."""
    caches: list[dict[str, Any]] = []
    base = "/sys/devices/system/cpu/cpu0/cache"
    for idx_path in sorted(glob.glob(f"{base}/index*")):
        entry: dict[str, Any] = {"path": idx_path}
        for name in ("level", "type", "size", "ways_of_associativity", "number_of_sets", "coherency_line_size"):
            fp = os.path.join(idx_path, name)
            entry[name] = _read_file(fp).strip() or None
        shared = os.path.join(idx_path, "shared_cpu_map")
        entry["shared_cpu_map"] = _read_file(shared).strip() or None
        caches.append(entry)
    return {"cpu0_cache_indices": caches}


def collect_numa_topology() -> dict[str, Any]:
    out: dict[str, Any] = {}
    nu, code, err = _run_text(["numactl", "--hardware"])
    out["numactl_hardware"] = {"text": nu.strip(), "returncode": code, "stderr": err.strip() or None}

    nodes: list[dict[str, Any]] = []
    for node_dir in sorted(glob.glob("/sys/devices/system/node/node*")):
        if not os.path.isdir(node_dir):
            continue
        m = re.match(r".*/node(\d+)$", node_dir)
        nid = int(m.group(1)) if m else -1
        mem_total = _read_file(os.path.join(node_dir, "meminfo")).strip()
        cpulist = _read_file(os.path.join(node_dir, "cpulist")).strip()
        distance = _read_file(os.path.join(node_dir, "distance")).strip()
        nodes.append({"id": nid, "meminfo": mem_total, "cpulist": cpulist, "distance": distance})
    out["sysfs_nodes"] = nodes
    return out


def collect_memory() -> dict[str, Any]:
    return {"meminfo": _read_file("/proc/meminfo").strip()}


def collect_block_devices() -> dict[str, Any]:
    out: dict[str, Any] = {}
    jout, code, _ = _run_text(["lsblk", "-J", "-o", "NAME,ROTA,TYPE,MODEL,TRAN,SIZE,FSTYPE,MOUNTPOINTS"])
    if code == 0 and jout.strip():
        try:
            out["lsblk_json"] = json.loads(jout)
        except json.JSONDecodeError:
            out["lsblk_json"] = None
    txt, code2, _ = _run_text(["lsblk", "-o", "NAME,ROTA,TYPE,MODEL,TRAN,SIZE"])
    out["lsblk_text"] = txt.strip()
    out["lsblk_returncode"] = code2

    rotational: dict[str, str] = {}
    for path in glob.glob("/sys/block/*/queue/rotational"):
        dev = path.split("/sys/block/")[1].split("/queue")[0]
        rotational[dev] = _read_file(path).strip()
    out["sysfs_rotational"] = rotational
    return out


def collect_filesystem_root() -> dict[str, Any]:
    out: dict[str, Any] = {}
    jout, code, _ = _run_text(["findmnt", "-J", "/"])
    if code == 0 and jout.strip():
        try:
            out["findmnt_root_json"] = json.loads(jout)
        except json.JSONDecodeError:
            out["findmnt_root_json"] = None
    txt, code2, _ = _run_text(["findmnt", "/", "-o", "SOURCE,FSTYPE,OPTIONS"])
    out["findmnt_text"] = txt.strip()
    out["findmnt_returncode"] = code2
    return out


def collect_kernel() -> dict[str, Any]:
    u, code, err = _run_text(["uname", "-a"])
    return {
        "uname_a": u.strip(),
        "returncode": code,
        "stderr": err.strip() or None,
        "os_release": _read_file("/etc/os-release").strip() or None,
    }


def collect_compilers() -> dict[str, Any]:
    compilers: dict[str, Any] = {}
    for name in ("gcc", "g++", "clang", "clang++"):
        path = shutil.which(name)
        if not path:
            continue
        ver, code, err = _run_text([name, "--version"])
        compilers[name] = {
            "path": path,
            "version_text": ver.strip(),
            "returncode": code,
            "stderr": err.strip() or None,
        }
    return compilers


def collect_cpufreq_governors() -> dict[str, Any]:
    gov: dict[str, str] = {}
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")):
        cpu_m = re.search(r"cpu(\d+)", path)
        label = cpu_m.group(0) if cpu_m else path
        gov[label] = _read_file(path).strip()
    return {"scaling_governor_per_cpu": gov}


def collect_sched_sysctl() -> dict[str, Any]:
    """Common scheduler-related tunables (best-effort)."""
    keys = [
        "kernel.sched_latency_ns",
        "kernel.sched_min_granularity_ns",
        "kernel.sched_wakeup_granularity_ns",
        "kernel.sched_child_runs_first",
        "kernel.sched_autogroup_enabled",
        "kernel.sched_tunable_scaling",
        "kernel.numa_balancing",
        "kernel.perf_event_paranoid",
    ]
    out: dict[str, str] = {}
    for k in keys:
        p = _run_text(["sysctl", "-n", k])
        if p[1] == 0 and p[0].strip():
            out[k] = p[0].strip()
    paranoid_file = _read_file("/proc/sys/kernel/perf_event_paranoid").strip()
    return {"sysctl": out, "perf_event_paranoid_file": paranoid_file or None}


def collect_static() -> dict[str, Any]:
    return {
        "lscpu": collect_lscpu(),
        "cpuinfo": collect_cpuinfo(),
        "cache_hierarchy": collect_cache_hierarchy(),
        "numa": collect_numa_topology(),
        "memory": collect_memory(),
        "block_devices": collect_block_devices(),
        "filesystem_root": collect_filesystem_root(),
        "kernel": collect_kernel(),
        "compilers": collect_compilers(),
        "cpufreq": collect_cpufreq_governors(),
        "scheduler_sysctl": collect_sched_sysctl(),
        "gpu": collect_gpu_static(),
    }
