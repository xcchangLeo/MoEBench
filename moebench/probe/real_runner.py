"""Run real UnixBench / PTS subtests for a bounded wall time (then SIGTERM)."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES
from moebench.unixbench.pipeline import _default_unixbench_root
from moebench.unixbench.report_parser import parse_report_text, pick_preferred_run_block


def run_unixbench_subtest_timed(
    test_id: str,
    *,
    duration_s: float,
    unixbench_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run ``perl Run -c 1 <test_id>`` under ``timeout``.

    UnixBench subtests are designed to run to completion (often 10–30s+ per test).
    After ``duration_s`` the process is killed: you get **real workload + eBPF** for
  that window, but usually **no valid official score** in the report (labels still
    come from full runs in ``dataset/``).
    """
    root = (unixbench_root or _default_unixbench_root()).resolve()
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    base_name = f"moebench_probe_ub_{test_id}_{stamp}".replace(":", "-")[:200]
    report_path = result_dir / base_name

    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)

    run_script = root / "Run"
    timeout_s = max(2.0, float(duration_s)) + 3.0
    cmd = [
        "timeout",
        f"{timeout_s:.1f}",
        "perl",
        str(run_script),
        "-c",
        str(UNIXBENCH_PARALLEL_COPIES),
        test_id,
    ]
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    timed_out = p.returncode == 124

    partial: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            txt = report_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_report_text(txt)
            rb = pick_preferred_run_block(parsed)
            tinfo = (rb.get("tests") or {}).get(test_id) if rb else None
            if tinfo:
                partial = {
                    "score": tinfo.get("score"),
                    "time_s": tinfo.get("time_s"),
                    "index": (tinfo.get("index_detail") or {}).get("index"),
                }
        except OSError:
            pass

    return {
        "runner": "unixbench",
        "test_id": test_id,
        "command": cmd,
        "duration_limit_s": float(duration_s),
        "elapsed_s": elapsed,
        "returncode": p.returncode,
        "timed_out": timed_out,
        "report_path": str(report_path) if report_path.is_file() else None,
        "partial_result": partial,
        "stderr_excerpt": (p.stderr or "")[:3000] if p.stderr else None,
    }


def run_pts_profile_timed(
    test_id: str,
    *,
    duration_s: float,
    pts_exe: str,
    result_basename: str | None = None,
) -> dict[str, Any]:
    """Run ``phoronix-test-suite run <profile>`` under ``timeout``."""
    from moebench.phoronix.pipeline import (
        _export_result_json,
        _pts_argv_as_installing_user,
        pts_clean_save_name,
        pts_subprocess_env,
        safe_session_tag,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_name = result_basename or f"moebench_probe_pts_{stamp}"
    name = pts_clean_save_name(safe_session_tag(raw_name))
    env = pts_subprocess_env()
    env["TEST_RESULTS_NAME"] = name
    env["TEST_RESULTS_IDENTIFIER"] = name
    env["TEST_RESULTS_DESCRIPTION"] = f"MoEBench probe {test_id}"

    timeout_s = max(2.0, float(duration_s)) + 15.0
    cmd = _pts_argv_as_installing_user(pts_exe, ["run", test_id])
    cmd = ["timeout", f"{timeout_s:.1f}", *cmd]

    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    timed_out = p.returncode == 124

    partial: dict[str, Any] | None = None
    export_path: str | None = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            raw_json = Path(tf.name)
        export = _export_result_json(pts_exe, name, raw_json)
        from moebench.phoronix.training_data import primary_value_from_export

        v = primary_value_from_export(export, test_id)
        if v is not None:
            partial = {"primary_value": float(v)}
        export_path = str(raw_json)
    except Exception:
        export_path = None

    return {
        "runner": "phoronix",
        "test_id": test_id,
        "command": cmd,
        "duration_limit_s": float(duration_s),
        "elapsed_s": elapsed,
        "returncode": p.returncode,
        "timed_out": timed_out,
        "pts_result_name": name,
        "export_path": export_path,
        "partial_result": partial,
        "stderr_excerpt": (p.stderr or "")[:3000] if p.stderr else None,
    }
