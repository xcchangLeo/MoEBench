#!/usr/bin/env python3
"""
Analyze PTS dataset sessions: per-profile primary values, correlations, redundancy (cpu / gpu suites).

Pure Python + stdlib JSON. Outputs under dataset/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_globs import resolve_glob_pattern
from moebench.phoronix.experts import expert_template_pts
from moebench.phoronix.pipeline import safe_session_tag
from moebench.phoronix.training_data import (
    collect_phoronix_run_paths,
    expert_test_ids_from_dataset,
    primary_value_from_export,
    time_seconds_for_profile,
)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = vx = vy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _session_host(name: str) -> str:
    return name.split("_", 1)[0] or name


def _expert_meta_from_sample(sample: dict[str, Any], test_id: str, expert_id: str) -> dict[str, Any]:
    """Merge dataset row with ``expert_template_pts`` so outputs match UnixBench-style fields even for older runs."""
    raw: dict[str, Any] = {}
    for e in sample.get("experts") or []:
        if str(e.get("test_id")) == test_id:
            raw = {k: v for k, v in e.items() if k != "observed"}
            break
    try:
        idx = int(str(expert_id).split("_", 1)[1])
    except (IndexError, ValueError):
        idx = 1
    tmpl = expert_template_pts(
        test_id,
        idx,
        title=raw.get("title"),
        default_suite=str(raw.get("phoronix_default_suite") or "cpu"),
    )
    return {**tmpl, **raw}


@dataclass
class ProfVec:
    test_id: str
    expert_id: str
    values: list[float]
    times_s: list[float]
    suite_means: list[float]
    sessions: list[str]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument(
        "--glob-pattern",
        type=str,
        default="",
        help="Glob under dataset-root (default: auto from --pts-suite; see moebench.dataset_globs)",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--redundancy-threshold", type=float, default=0.9)
    ap.add_argument(
        "--pts-suite",
        type=str,
        default=None,
        metavar="ID",
        help="Only use runs where yi.suite matches (e.g. pts/nvidia-gpu-compute); output filenames include this tag",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo / "dataset"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_root

    glob_eff = resolve_glob_pattern(
        benchmark="phoronix",
        glob_pattern=args.glob_pattern or None,
        pts_suite=args.pts_suite,
    )

    paths = collect_phoronix_run_paths(
        dataset_root,
        glob_pattern=glob_eff,
        pts_suite=args.pts_suite,
    )
    sample = json.load(open(paths[0], encoding="utf-8"))
    base_ids = expert_test_ids_from_dataset(sample)
    # Intersect profiles that have a primary value in every run (some runs may omit a test).
    common: set[str] | None = None
    for p in paths:
        ds = json.load(open(p, encoding="utf-8"))
        export = (ds.get("yi") or {}).get("pts_export") or {}
        ok = {t for t in base_ids if primary_value_from_export(export, t) is not None}
        common = ok if common is None else (common & ok)
    if not common:
        raise SystemExit("No profile has a primary value in every run; check pts_export completeness.")
    test_ids = [t for t in base_ids if t in common]
    tid_to_eid: dict[str, str] = {}
    for e in sample.get("experts") or []:
        tid = str(e.get("test_id") or "")
        eid = str(e.get("expert_id") or "")
        if tid and eid and tid not in tid_to_eid:
            tid_to_eid[tid] = eid
    expert_ids = [tid_to_eid.get(t) or f"e_{i:03d}" for i, t in enumerate(test_ids)]
    m = len(test_ids)
    pv = [
        ProfVec(
            test_id=test_ids[i],
            expert_id=expert_ids[i] if i < len(expert_ids) else f"e_{i:03d}",
            values=[],
            times_s=[],
            suite_means=[],
            sessions=[],
        )
        for i in range(m)
    ]

    for p in paths:
        ds = json.load(open(p, encoding="utf-8"))
        export = (ds.get("yi") or {}).get("pts_export") or {}
        if not export:
            continue
        vals_run: list[float] = []
        for tid in test_ids:
            v = primary_value_from_export(export, tid)
            assert v is not None  # common set guarantees
            vals_run.append(float(v))
        if len(vals_run) != len(test_ids):
            continue
        suite_mean = float(sum(vals_run) / len(vals_run))
        sess = p.parent.name
        host = _session_host(sess)
        for i, tid in enumerate(test_ids):
            pv[i].values.append(vals_run[i])
            pv[i].suite_means.append(suite_mean)
            pv[i].sessions.append(sess)
            t = time_seconds_for_profile(ds, tid)
            pv[i].times_s.append(float(t) if t is not None else float("nan"))

    if len(pv[0].values) < 2:
        raise SystemExit("Too few aligned PTS runs for analysis")

    corr = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                corr[i][j] = 1.0
            else:
                r = _pearson(pv[i].values, pv[j].values)
                corr[i][j] = r if r is not None else 0.0

    thr = float(args.redundancy_threshold)
    redundant_pairs: list[tuple[str, str, float]] = []
    for i in range(m):
        for j in range(i + 1, m):
            r = corr[i][j]
            if abs(r) >= thr:
                redundant_pairs.append((pv[i].test_id, pv[j].test_id, r))

    now = datetime.now(timezone.utc).isoformat()
    experts_out: list[dict[str, Any]] = []
    for i in range(m):
        meta = _expert_meta_from_sample(sample, pv[i].test_id, pv[i].expert_id)
        n_t = sum(1 for x in pv[i].times_s if not math.isnan(x))
        time_mean = (
            sum(x for x in pv[i].times_s if not math.isnan(x)) / n_t if n_t else float("nan")
        )
        row: dict[str, Any] = {
            **meta,
            "expert_id": pv[i].expert_id,
            "test_id": pv[i].test_id,
            "historical_runtime_mean_s": time_mean if not math.isnan(time_mean) else None,
            "value_mean": sum(pv[i].values) / len(pv[i].values),
            "time_mean_s": time_mean,
        }
        experts_out.append(row)

    model = {
        "schema": "moebench.phoronix.expert_model_global.v2",
        "created_at_utc": now,
        "dataset_root": str(dataset_root),
        "glob_pattern": glob_eff,
        "pts_suite": args.pts_suite,
        "num_runs": len(pv[0].values),
        "experts": experts_out,
        "correlation_matrix": {"order": [pv[i].expert_id for i in range(m)], "values": corr},
    }
    out_stem = (
        f"phoronix_{safe_session_tag(args.pts_suite.replace('/', '_'))}_expert_model_global"
        if args.pts_suite
        else "phoronix_expert_model_global"
    )
    out_json = out_dir / f"{out_stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
        f.write("\n")

    csv_path = out_dir / (
        f"phoronix_{safe_session_tag(args.pts_suite.replace('/', '_'))}_correlation_matrix.csv"
        if args.pts_suite
        else "phoronix_expert_correlation_matrix.csv"
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [pv[i].test_id for i in range(m)])
        for i in range(m):
            w.writerow([pv[i].test_id] + [f"{corr[i][j]:.6f}" for j in range(m)])

    rp = out_dir / (
        f"phoronix_{safe_session_tag(args.pts_suite.replace('/', '_'))}_redundant_pairs.csv"
        if args.pts_suite
        else "phoronix_expert_redundant_pairs.csv"
    )
    with open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_id_a", "test_id_b", "pearson_r"])
        for a, b, r in redundant_pairs:
            w.writerow([a, b, f"{r:.6f}"])

    print(f"Wrote {out_json}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
