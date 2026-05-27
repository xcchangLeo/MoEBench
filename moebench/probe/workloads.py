"""Category-aligned micro-workloads for fixed-duration subtest probes."""

from __future__ import annotations

import array
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


from moebench.unixbench.experts import category_for_test as category_for_unixbench_test


def category_for_pts_test(test_id: str, title: str | None = None) -> str:
    from moebench.phoronix.experts import infer_pts_category

    return infer_pts_category(test_id, title)


def _workload_script(category: str, duration_s: float, mem_mb: int) -> str:
    dur = float(duration_s)
    mb = int(mem_mb)
    cat = category.lower()
    if cat == "cpu":
        body = textwrap.dedent(
            f"""
            import math, time
            end = time.time() + {dur!r}
            x = 0.0
            while time.time() < end:
                for j in range(8000):
                    x += math.sqrt(float(j + 1))
            """
        )
    elif cat == "memory":
        body = textwrap.dedent(
            f"""
            import array, time
            end = time.time() + {dur!r}
            n = {mb!r} * 1024 * 1024
            a = array.array("B", (i % 256 for i in range(n)))
            b = array.array("B", [0] * n)
            while time.time() < end:
                b[:] = a
            """
        )
    elif cat == "io":
        body = textwrap.dedent(
            f"""
            import os, tempfile, time
            end = time.time() + {dur!r}
            fd, path = tempfile.mkstemp(prefix="moe_probe_io_")
            os.close(fd)
            chunk = os.urandom(65536)
            try:
                while time.time() < end:
                    with open(path, "wb") as f:
                        for _ in range(32):
                            f.write(chunk)
                    with open(path, "rb") as f:
                        while f.read(65536):
                            pass
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            """
        )
    elif cat == "syscall":
        body = textwrap.dedent(
            f"""
            import os, time
            end = time.time() + {dur!r}
            while time.time() < end:
                for _ in range(50000):
                    os.getpid()
            """
        )
    elif cat == "thread":
        body = textwrap.dedent(
            f"""
            import subprocess, time
            end = time.time() + {dur!r}
            while time.time() < end:
                subprocess.run(["true"], check=False, capture_output=True)
            """
        )
    elif cat == "gpu":
        body = textwrap.dedent(
            f"""
            import array, math, time
            end = time.time() + {dur!r}
            n = min({mb!r}, 16) * 1024 * 1024
            a = array.array("f", (0.1 * (i % 997) for i in range(n // 4)))
            while time.time() < end:
                s = 0.0
                for j in range(min(5000, len(a))):
                    s += math.sqrt(abs(a[j]) + 1.0)
            """
        )
    else:
        body = textwrap.dedent(
            f"""
            import math, time
            end = time.time() + {dur!r}
            while time.time() < end:
                sum(math.sqrt(float(i)) for i in range(1, 5000))
            """
        )
    return textwrap.dedent(
        f"""
        {body}
        """
    ).strip() + "\n"


def run_category_workload(
    category: str,
    duration_s: float,
    *,
    mem_mb: int = 64,
) -> dict[str, Any]:
    """Run a micro-workload for ``duration_s`` seconds (subprocess)."""
    script = _workload_script(category, duration_s, mem_mb)
    fd, path = tempfile.mkstemp(suffix="_moebench_probe_wl.py", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(script)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=max(10.0, duration_s + 5.0),
            check=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "category": category,
            "duration_s": float(duration_s),
            "elapsed_s": float(elapsed),
            "returncode": p.returncode,
            "stderr_excerpt": (p.stderr or "")[:2000] if p.stderr else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "category": category,
            "duration_s": float(duration_s),
            "elapsed_s": float(duration_s),
            "returncode": -9,
            "stderr_excerpt": "timeout",
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
