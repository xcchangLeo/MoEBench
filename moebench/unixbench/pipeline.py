"""Run feature collection + full UnixBench `Run`, emit dataset JSON."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moebench.collector import collect_all
from moebench.unixbench.experts import (
    INDEX_SUITE_TEST_IDS,
    UNIXBENCH_PARALLEL_COPIES,
    build_expert_catalog,
    expert_template,
)
from moebench.unixbench.report_parser import build_ti_from_runs, parse_report_text


def _moebench_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_unixbench_root() -> Path:
    return _moebench_repo_root() / "byte-unixbench" / "UnixBench"


def default_dataset_root() -> Path:
    return _moebench_repo_root() / "dataset"


def host_slug() -> str:
    try:
        h = socket.gethostname()
    except OSError:
        h = "unknown-host"
    h = h.strip() or "unknown-host"
    safe = re.sub(r"[^\w.\-]+", "_", h, flags=re.UNICODE)
    safe = re.sub(r"_+", "_", safe).strip("._-")[:64]
    return safe or "host"


def safe_session_tag(tag: str) -> str:
    t = tag.strip() or "session"
    t = re.sub(r"[^\w.\-]+", "_", t, flags=re.UNICODE)
    t = re.sub(r"_+", "_", t).strip("._-")
    return (t[:120] if t else "session")


def default_session_tag() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{host_slug()}_{stamp}"


def resolve_unixbench_run_args(run_args: list[str] | None) -> list[str]:
    """Ensure ``perl Run`` uses single-copy mode unless caller passed ``-c`` after ``--``."""
    if run_args:
        return list(run_args)
    return ["-c", str(UNIXBENCH_PARALLEL_COPIES)]


def _merge_experts_observed(
    catalog: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach observed metrics from run with smallest parallel_copies (usually 1)."""
    if not runs:
        return catalog
    best = min(
        (r for r in runs if r.get("parallel_copies") is not None),
        key=lambda r: (r.get("parallel_copies") or 1_000_000),
        default=runs[0],
    )
    tests = best.get("tests") or {}
    out: list[dict[str, Any]] = []
    for ex in catalog:
        tid = ex["test_id"]
        obs = tests.get(tid)
        row = dict(ex)
        if obs:
            row["observed"] = {
                "parallel_copies": best.get("parallel_copies"),
                "score": obs.get("score"),
                "score_unit": obs.get("score_unit"),
                "time_s": obs.get("time_s"),
                "pass_samples": obs.get("pass_samples"),
                "index_detail": obs.get("index_detail"),
            }
            row["execution_cost"] = obs.get("time_s")
        else:
            row["observed"] = None
        out.append(row)
    return out


def run_unixbench_dataset(
    *,
    unixbench_root: Path | str | None = None,
    output_json: Path | str | None = None,
    collect_features: bool = True,
    xi_override: dict[str, Any] | None = None,
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    mem_mb: int = 64,
    enable_ebpf: bool = True,
    run_args: list[str] | None = None,
    perl_exe: str = "perl",
    ub_output_basename: str | None = None,
    round_index: int | None = None,
    total_rounds: int | None = None,
    session_tag: str | None = None,
) -> dict[str, Any]:
    """
    1) Collect xi (static + dynamic features), unless ``xi_override`` is set.
    2) Run `perl Run -c 1` (single parallel copy) in UnixBench directory with inherited stdio.
    3) Parse report file; build yi (scores + index) and ti (per-test times).

    Uses UB_OUTPUT_FILE_NAME + UB_RESULTDIR so the report path is known.
    """
    root = Path(unixbench_root) if unixbench_root else _default_unixbench_root()
    root = root.resolve()
    run_script = root / "Run"
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    if ub_output_basename:
        base_name = ub_output_basename
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        base_name = f"moebench_ub_{stamp}"
    report_path = result_dir / base_name

    xi: dict[str, Any] | None = None
    if xi_override is not None:
        xi = xi_override
    elif collect_features:
        xi = collect_all(
            warmup_s=warmup_s,
            proc_sample_s=proc_sample_s,
            enable_ebpf=enable_ebpf,
            mem_mb=mem_mb,
        )

    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)

    cmd = [perl_exe, str(run_script)]
    cmd.extend(resolve_unixbench_run_args(run_args))

    rc = subprocess.call(cmd, cwd=str(root), env=env)
    if rc != 0:
        raise RuntimeError(f"UnixBench Run exited with code {rc}")

    report_txt = report_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_report_text(report_txt)
    runs = parsed.get("runs") or []

    yi = {
        "suite": "index",
        "unixbench_version_guess": _read_unixbench_version(report_txt),
        "runs": runs,
    }
    ti = build_ti_from_runs(runs)

    catalog = build_expert_catalog(INDEX_SUITE_TEST_IDS)
    experts = _merge_experts_observed(catalog, runs)

    dataset: dict[str, Any] = {
        "schema": "moebench.unixbench.dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "xi": xi,
        "yi": yi,
        "ti": ti,
        "experts": experts,
        "session": {
            "tag": session_tag,
            "round_index": round_index,
            "total_rounds": total_rounds,
            "xi_reused_from_previous_round": xi_override is not None,
        },
        "unixbench": {
            "root": str(root),
            "command": cmd,
            "returncode": rc,
            "result_files": {
                "report": str(report_path),
                "log": str(report_path) + ".log",
                "html": str(report_path) + ".html",
            },
            "env": {"UB_OUTPUT_FILE_NAME": base_name, "UB_RESULTDIR": str(result_dir)},
        },
        "notes": {
            "D": "Each sample is (xi, yi, ti); aggregate historical mean/variance/correlations offline.",
            "weights": "suite_contribution_weight / correlation_with / hardware_stability left null until aggregated.",
        },
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out_path}", file=sys.stderr)

    return dataset


def _read_unixbench_version(report_txt: str) -> str | None:
    m = __import__("re").search(r"BYTE UNIX Benchmarks \(Version\s+([0-9.]+)\)", report_txt)
    return m.group(1) if m else None


def expert_catalog_only() -> list[dict[str, Any]]:
    """Export E with placeholders (no benchmark run)."""
    return [expert_template(tid, i + 1) for i, tid in enumerate(INDEX_SUITE_TEST_IDS)]


def run_unixbench_batch(
    *,
    num_rounds: int,
    dataset_root: Path | str | None = None,
    session_tag: str | None = None,
    reuse_xi: bool = False,
    unixbench_root: Path | str | None = None,
    collect_features: bool = True,
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    mem_mb: int = 64,
    enable_ebpf: bool = True,
    run_args: list[str] | None = None,
    perl_exe: str = "perl",
) -> dict[str, Any]:
    """
    Run ``num_rounds`` full UnixBench pipelines, writing:

        ``{dataset_root}/{session_tag}/run-01.json`` … ``run-NN.json``

    and ``manifest.json``. By default each round collects xi; set ``reuse_xi=True``
    to reuse xi from round 1 on later rounds (faster, same xi for all runs).
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    root_ds = Path(dataset_root) if dataset_root else default_dataset_root()
    root_ds = root_ds.resolve()
    tag = safe_session_tag(session_tag or default_session_tag())
    session_dir = root_ds / tag
    session_dir.mkdir(parents=True, exist_ok=True)

    xi_cached: dict[str, Any] | None = None
    written: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()

    for i in range(1, num_rounds + 1):
        use_xi: dict[str, Any] | None = None
        do_collect = collect_features
        if reuse_xi and i > 1:
            use_xi = xi_cached
            do_collect = False
        elif not collect_features:
            do_collect = False

        ub_base = f"moebench_ub_{tag}_r{i:02d}_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}"
        out_path = session_dir / f"run-{i:02d}.json"
        ds = run_unixbench_dataset(
            unixbench_root=unixbench_root,
            output_json=out_path,
            collect_features=do_collect and use_xi is None,
            xi_override=use_xi,
            warmup_s=warmup_s,
            proc_sample_s=proc_sample_s,
            mem_mb=mem_mb,
            enable_ebpf=enable_ebpf,
            run_args=run_args,
            perl_exe=perl_exe,
            ub_output_basename=ub_base,
            round_index=i,
            total_rounds=num_rounds,
            session_tag=tag,
        )
        if xi_cached is None and ds.get("xi") is not None:
            xi_cached = ds["xi"]
        written.append(
            {
                "round": i,
                "json": str(out_path),
                "created_at_utc": ds.get("created_at_utc"),
                "unixbench_report": ds.get("unixbench", {}).get("result_files", {}).get("report"),
            }
        )

    manifest: dict[str, Any] = {
        "schema": "moebench.unixbench.batch_manifest.v1",
        "session_tag": tag,
        "session_dir": str(session_dir),
        "num_rounds": num_rounds,
        "reuse_xi": reuse_xi,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": written,
    }
    man_path = session_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote batch manifest {man_path}", file=sys.stderr)

    return manifest
