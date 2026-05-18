#!/usr/bin/env python3
"""
Analyze UnixBench dataset sessions to model each expert and validate redundancy.

No external scientific deps required (pure Python).

Outputs (under ./dataset by default):
  - unixbench_expert_model_global.json
  - unixbench_expert_correlation_matrix.csv
  - unixbench_expert_clustering.json
  - unixbench_expert_redundant_pairs.csv
  - unixbench_expert_catalog_modeled.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = 0.0
    vx = 0.0
    vy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    if vx <= 0.0 or vy <= 0.0:
        return None
    # Population covariance/variance is fine for correlation.
    return cov / math.sqrt(vx * vy)


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _variance(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return v


def _stdev(xs: list[float]) -> float | None:
    v = _variance(xs)
    if v is None:
        return None
    return math.sqrt(v)


def _cv(xs: list[float]) -> float | None:
    m = _mean(xs)
    s = _stdev(xs)
    if m is None or s is None or m == 0.0:
        return None
    return s / m


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _session_host(session_dir_name: str) -> str:
    # Expected: hostname_UTCtimestamp or custom tag. Best-effort.
    return session_dir_name.split("_", 1)[0] or session_dir_name


def _select_block_for_vectors(ds: dict[str, Any]) -> dict[str, Any] | None:
    runs = ds.get("yi", {}).get("runs") or []
    if not runs:
        return None
    from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES

    # Prefer single-copy block (MoEBench ``Run -c 1``).
    def key(rb: dict[str, Any]) -> tuple[int, int, int]:
        pc = rb.get("parallel_copies")
        if pc == UNIXBENCH_PARALLEL_COPIES:
            return (0, 0, 0)
        if isinstance(pc, int):
            return (1, pc, 0)
        if pc is None:
            return (2, 0, 0)
        # unknown types
        return (3, 0, 0)

    return sorted(runs, key=key)[0]


def _choose_ti_parallel_copy(ds: dict[str, Any], test_id: str) -> str | None:
    by_test = ds.get("ti", {}).get("by_test_id", {}) or {}
    copies = (by_test.get(test_id) or {}).keys()
    if not copies:
        return None
    from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES

    pk = str(UNIXBENCH_PARALLEL_COPIES)
    if pk in copies:
        return pk
    numeric = []
    for c in copies:
        try:
            numeric.append((int(c), c))
        except Exception:
            pass
    if numeric:
        numeric.sort(key=lambda t: t[0])
        return numeric[0][1]
    return sorted(list(copies))[0]


@dataclass
class ExpertVectors:
    test_id: str
    expert_id: str
    scores: list[float]
    runtimes_s: list[float]
    suite_scores: list[float]
    session_tags: list[str]
    host_slugs: list[str]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None, help="Default: ./dataset under repo")
    ap.add_argument("--out-dir", type=str, default=None, help="Default: same as dataset-root")
    ap.add_argument("--redundancy-threshold", type=float, default=0.9, help="|corr| threshold for redundant pairs")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / "dataset"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_root

    run_paths = []
    for p in dataset_root.glob("*/run-*.json"):
        if p.is_file():
            run_paths.append(p)
    run_paths.sort()
    if not run_paths:
        raise SystemExit(f"No run json found under {dataset_root}")

    # Load one file to infer expert catalog shape.
    sample_ds = json.load(open(run_paths[0], "r", encoding="utf-8"))
    experts = sample_ds.get("experts") or []
    test_ids = [ex["test_id"] for ex in experts]
    expert_ids = [ex["expert_id"] for ex in experts]
    if not experts or len(test_ids) < 2:
        raise SystemExit("Cannot infer experts from dataset json")

    # Vectors indexed by expert index.
    ev = [
        ExpertVectors(
            test_id=test_ids[i],
            expert_id=expert_ids[i],
            scores=[],
            runtimes_s=[],
            suite_scores=[],
            session_tags=[],
            host_slugs=[],
        )
        for i in range(len(experts))
    ]

    # Aggregate across runs.
    for rp in run_paths:
        ds = json.load(open(rp, "r", encoding="utf-8"))
        rb = _select_block_for_vectors(ds)
        if not rb:
            continue
        suite_score = _safe_float(rb.get("system_benchmarks_index_score"))
        if suite_score is None:
            # Skip runs with no suite score.
            continue

        ti_by_test = (ds.get("ti", {}).get("by_test_id", {}) or {})
        session_dir = rp.parent.name
        host = _session_host(session_dir)

        for i, tid in enumerate(test_ids):
            # score
            tinfo = (rb.get("tests", {}) or {}).get(tid)
            score = _safe_float(tinfo.get("score") if tinfo else None)
            if score is None:
                continue

            # runtime
            pc = _choose_ti_parallel_copy(ds, tid)
            runtime = None
            if pc is not None:
                runtime = _safe_float(ti_by_test.get(tid, {}).get(pc))
            if runtime is None:
                # fallback to experts[i].observed.execution_cost
                obs = (ds.get("experts") or [])
                if i < len(obs):
                    runtime = _safe_float((obs[i].get("execution_cost") if obs[i] else None))
            if runtime is None:
                continue

            ev[i].scores.append(score)
            ev[i].runtimes_s.append(runtime)
            ev[i].suite_scores.append(suite_score)
            ev[i].session_tags.append(session_dir)
            ev[i].host_slugs.append(host)

    # Sanity check
    num_samples = len(ev[0].scores)
    if num_samples < 5:
        raise SystemExit(f"Too few samples inferred: {num_samples}. Check report_parser / ti keys.")

    # Build correlation matrix between expert scores.
    m = len(ev)
    corr = [[None for _ in range(m)] for __ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                corr[i][j] = 1.0
                continue
            # Align by run order is implicit if we appended in same iteration.
            # Since we may have skipped some points per expert, we recompute with shared length:
            # For simplicity we compute on min length (still consistent if skips are rare).
            n = min(len(ev[i].scores), len(ev[j].scores))
            xs = ev[i].scores[:n]
            ys = ev[j].scores[:n]
            r = _pearson_corr(xs, ys)
            corr[i][j] = r

    # Suite contribution weights based on corr(expert_score, suite_score).
    suite_weights = []
    for i in range(m):
        n = min(len(ev[i].scores), len(ev[i].suite_scores))
        r = _pearson_corr(ev[i].scores[:n], ev[i].suite_scores[:n])
        w = abs(r) if r is not None else 0.0
        suite_weights.append(w)
    s = sum(suite_weights)
    if s == 0.0:
        suite_weights_norm = [0.0 for _ in suite_weights]
    else:
        suite_weights_norm = [w / s for w in suite_weights]

    # Cluster via greedy agglomeration with average similarity = abs(corr).
    clusters: list[list[int]] = [[i] for i in range(m)]
    # Keep merge history
    merge_steps = []

    def avg_similarity(ci: list[int], cj: list[int]) -> float:
        best = 0.0
        cnt = 0
        acc = 0.0
        for a in ci:
            for b in cj:
                r = corr[a][b]
                if r is None:
                    continue
                acc += abs(r)
                cnt += 1
        if cnt == 0:
            return 0.0
        return acc / cnt

    while len(clusters) > 1:
        best_pair = None
        best_score = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sc = avg_similarity(clusters[i], clusters[j])
                if sc > best_score:
                    best_score = sc
                    best_pair = (i, j)
        i, j = best_pair  # type: ignore[misc]
        merged = clusters[i] + clusters[j]
        merge_steps.append(
            {
                "merge_of": [clusters[i], clusters[j]],
                "avg_abs_corr": best_score,
                "merged_cluster": merged,
            }
        )
        # remove higher index first
        for idx in sorted([i, j], reverse=True):
            clusters.pop(idx)
        clusters.append(merged)

    # Use redundancy threshold to make connected components based on |corr| > threshold.
    # This gives an easy "cluster" view for validation.
    thr = float(args.redundancy_threshold)
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(m):
        for j in range(i + 1, m):
            r = corr[i][j]
            if r is not None and abs(r) >= thr:
                union(i, j)

    comp: dict[int, list[int]] = {}
    for i in range(m):
        root = find(i)
        comp.setdefault(root, []).append(i)
    redundancy_clusters = [sorted(v) for v in comp.values()]
    redundancy_clusters.sort(key=lambda cl: (-len(cl), cl[0]))

    # Write outputs
    now = datetime.now(timezone.utc).isoformat()
    model = {
        "schema": "moebench.unixbench.expert_model_global.v1",
        "created_at_utc": now,
        "dataset_root": str(dataset_root),
        "num_run_json": len(run_paths),
        "experts": [],
        "suite_contribution_weights_raw_abs_corr": {
            ev[i].expert_id: (None if _pearson_corr(ev[i].scores, ev[i].suite_scores) is None else abs(_pearson_corr(ev[i].scores, ev[i].suite_scores)))  # type: ignore[arg-type]
            for i in range(m)
        },
        "suite_contribution_weight_normalized": {
            ev[i].expert_id: suite_weights_norm[i] for i in range(m)
        },
        "correlation_matrix": {
            "order": [ev[i].expert_id for i in range(m)],
            "values": corr,
        },
        "clustering": {
            "greedy_hierarchical_merge_steps": merge_steps,
            "redundancy_threshold": thr,
            "redundancy_connected_components": [
                {
                    "cluster_indices": cl,
                    "expert_ids": [ev[k].expert_id for k in cl],
                }
                for cl in redundancy_clusters
            ],
        },
    }

    redundant_pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            r = corr[i][j]
            if r is None:
                continue
            if abs(r) >= thr:
                redundant_pairs.append(
                    {
                        "expert_a": ev[i].expert_id,
                        "expert_b": ev[j].expert_id,
                        "test_id_a": ev[i].test_id,
                        "test_id_b": ev[j].test_id,
                        "pearson_corr": r,
                        "abs_corr": abs(r),
                    }
                )

    # Per-expert stats
    for i in range(m):
        runt = ev[i].runtimes_s
        sc = ev[i].scores
        cv_sc = _cv(sc)
        cv_rt = _cv(runt)
        model["experts"].append(
            {
                "expert_id": ev[i].expert_id,
                "test_id": ev[i].test_id,
                "num_samples": len(sc),
                "historical_runtime_mean_s": _mean(runt),
                "historical_runtime_variance_s": _variance(runt),
                "execution_cost_mean_s": _mean(runt),
                "execution_cost_variance_s": _variance(runt),
                "score_mean": _mean(sc),
                "score_variance": _variance(sc),
                # Within-host stability proxy (single hardware available).
                "within_host_score_cv": cv_sc,
                "within_host_runtime_cv": cv_rt,
                "hardware_stability_proxy": None if cv_sc is None else 1.0 / (cv_sc + 1e-9),
                "suite_contribution_weight": suite_weights_norm[i],
                "correlation_with": {
                    ev[j].expert_id: corr[i][j] for j in range(m) if j != i and corr[i][j] is not None
                },
            }
        )

    # Correlation matrix CSV
    order_ids = [ev[i].expert_id for i in range(m)]
    corr_csv_path = out_dir / "unixbench_expert_correlation_matrix.csv"
    with open(corr_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert_id"] + order_ids)
        for i in range(m):
            row = [order_ids[i]]
            for j in range(m):
                v = corr[i][j]
                row.append("" if v is None else f"{v:.6f}")
            w.writerow(row)

    # Lightweight HTML heatmap (no scientific libs).
    # Color map: -1..1 -> blue..white..red.
    html_path = out_dir / "unixbench_expert_correlation_heatmap.html"
    order_labels = order_ids

    def color_for(v: float) -> str:
        # v in [-1,1]
        v = max(-1.0, min(1.0, v))
        # Normalize to [0,1] for intensity around 0.
        if v >= 0:
            # white -> red
            t = v
            r = 255
            g = int(255 * (1.0 - t))
            b = g
        else:
            # white -> blue
            t = -v
            b = 255
            g = int(255 * (1.0 - t))
            r = g
        return f"rgb({r},{g},{b})"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>UnixBench Expert Correlation Heatmap</title>"
            "<style>"
            "body{font-family:system-ui,Arial,sans-serif; padding:16px;}"
            "table{border-collapse:collapse; margin-top:10px;}"
            "th,td{border:1px solid #ddd; padding:6px 10px; text-align:center;}"
            "th{background:#f6f6f6; position:sticky; top:0; z-index:1;}"
            ".small{font-size:12px; color:#555; margin-top:6px;}"
            "</style></head><body>"
        )
        f.write("<h2>UnixBench Expert Correlation Heatmap</h2>")
        f.write("<div class='small'>Pearson correlation of expert score vectors across runs. "
                "Rows/columns follow expert_id order from unixbench_expert_correlation_matrix.csv.</div>")
        f.write("<table>")
        f.write("<tr><th></th>" + "".join(f"<th>{eid}</th>" for eid in order_labels) + "</tr>")
        for i in range(m):
            f.write(f"<tr><th>{order_labels[i]}</th>")
            for j in range(m):
                v = corr[i][j]
                if v is None:
                    f.write("<td>NA</td>")
                    continue
                col = color_for(float(v))
                f.write(f"<td style='background:{col}' title='{v:.6f}'>{v:.2f}</td>")
            f.write("</tr>")
        f.write("</table>")
        f.write("</body></html>")

    model_json_path = out_dir / "unixbench_expert_model_global.json"
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Redundant pairs CSV
    pairs_path = out_dir / "unixbench_expert_redundant_pairs.csv"
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["expert_a", "expert_b", "test_id_a", "test_id_b", "pearson_corr", "abs_corr"],
        )
        w.writeheader()
        for row in sorted(redundant_pairs, key=lambda r: -r["abs_corr"]):  # type: ignore[index]
            w.writerow(row)

    # Modeled expert catalog: reuse template but fill learned fields.
    # We don't overwrite each run; this file serves as learned E.
    catalog = sample_ds.get("experts") or []
    id_to_stats = {ex["expert_id"]: ex for ex in model["experts"]}
    for ex in catalog:
        sid = ex.get("expert_id")
        if sid in id_to_stats:
            ex.update(id_to_stats[sid])
    catalog_path = out_dir / "unixbench_expert_catalog_modeled.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "moebench.unixbench.expert_catalog_modeled.v1",
                "created_at_utc": now,
                "dataset_root": str(dataset_root),
                "experts": catalog,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    clustering_path = out_dir / "unixbench_expert_clustering.json"
    with open(clustering_path, "w", encoding="utf-8") as f:
        json.dump(model["clustering"], f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote expert model: {model_json_path}")
    print(f"Wrote correlation matrix CSV: {corr_csv_path}")
    print(f"Wrote correlation heatmap HTML: {html_path}")
    print(f"Wrote redundancy pairs CSV: {pairs_path}")
    print(f"Wrote modeled expert catalog JSON: {catalog_path}")
    print(f"Wrote clustering: {clustering_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

