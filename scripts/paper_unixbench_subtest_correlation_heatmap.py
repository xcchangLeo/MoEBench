#!/usr/bin/env python3
"""
Motivation figure: 12×12 UnixBench subtest score correlation heatmap.

Pools full-suite UnixBench runs from all machines in ``dataset/``, builds a
samples × subtests table, and plots Pearson or Spearman correlation.

Outputs (default under ``paper/``):
  - unixbench_subtest_correlation_heatmap.pdf
  - unixbench_subtest_correlation_heatmap.png
  - unixbench_subtest_correlation_matrix.csv
  - unixbench_subtest_correlation_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moebench.dataset_machines import load_machines_registry, machine_from_session_tag
from moebench.reconstruct.data import collect_unixbench_run_paths
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES


def _select_run_block(ds: dict) -> dict | None:
    runs = ds.get("yi", {}).get("runs") or []
    if not runs:
        return None

    def key(rb: dict) -> tuple[int, int]:
        pc = rb.get("parallel_copies")
        if pc == UNIXBENCH_PARALLEL_COPIES:
            return (0, 0)
        if isinstance(pc, int):
            return (1, pc)
        return (2, 0)

    return sorted(runs, key=key)[0]


def _subtest_index_score(tinfo: dict | None) -> float | None:
    if not tinfo:
        return None
    idx = (tinfo.get("index_detail") or {}).get("index")
    if idx is not None:
        try:
            return float(idx)
        except (TypeError, ValueError):
            pass
    try:
        return float(tinfo.get("score"))
    except (TypeError, ValueError):
        return None


def build_score_table(run_paths: list[Path]) -> tuple[np.ndarray, list[dict]]:
    """Return matrix shape (n_samples, n_subtests) and per-row metadata."""
    test_ids = list(INDEX_SUITE_TEST_IDS)
    rows: list[list[float]] = []
    meta: list[dict] = []

    try:
        reg = load_machines_registry()
        by_slug = reg.get("by_slug", {})
    except OSError:
        by_slug = {}

    for rp in run_paths:
        with open(rp, encoding="utf-8") as f:
            ds = json.load(f)
        rb = _select_run_block(ds)
        if not rb:
            continue
        tests = rb.get("tests") or {}
        values: list[float] = []
        for tid in test_ids:
            v = _subtest_index_score(tests.get(tid))
            if v is None:
                break
            values.append(v)
        if len(values) != len(test_ids):
            continue

        session = rp.parent.name
        host = machine_from_session_tag(session)
        host_info = by_slug.get(host, {})
        rows.append(values)
        meta.append(
            {
                "run_json": str(rp),
                "session_tag": session,
                "hostname_slug": host,
                "paper_host_id": host_info.get("paper_host_id"),
                "config_label": host_info.get("config_label"),
                "suite_index_score": rb.get("system_benchmarks_index_score"),
            }
        )

    if not rows:
        raise SystemExit("No complete UnixBench runs with all 12 subtest scores.")
    return np.asarray(rows, dtype=float), meta


def correlation_matrix(x: np.ndarray, method: str) -> np.ndarray:
    n = x.shape[1]
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            xi = x[:, i]
            xj = x[:, j]
            if method == "pearson":
                r, _ = pearsonr(xi, xj)
            elif method == "spearman":
                r, _ = spearmanr(xi, xj)
            else:
                raise ValueError(method)
            corr[i, j] = corr[j, i] = float(r)
    return corr


def plot_heatmap(
    corr: np.ndarray,
    labels: list[str],
    *,
    method: str,
    n_samples: int,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.capitalize()} correlation", fontsize=9)

    method_title = "Pearson" if method == "pearson" else "Spearman"
    ax.set_title(
        f"UnixBench subtest score correlation ({method_title}, n={n_samples} full runs)",
        fontsize=10,
        pad=12,
    )
    fig.tight_layout()

    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf}", file=sys.stderr)
    print(f"Wrote {png}", file=sys.stderr)


def write_csv(corr: np.ndarray, labels: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_id"] + labels)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + [f"{corr[i, j]:.6f}" for j in range(len(labels))])
    print(f"Wrote {path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=_REPO_ROOT / "dataset")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "paper")
    ap.add_argument(
        "--method",
        choices=("pearson", "spearman"),
        default="spearman",
        help="Correlation method (default: spearman, robust across heterogeneous hosts)",
    )
    ap.add_argument(
        "--out-base",
        type=Path,
        default=None,
        help="Output basename without extension (default: <out-dir>/unixbench_subtest_correlation_heatmap)",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = (args.out_base or out_dir / "unixbench_subtest_correlation_heatmap").resolve()

    run_paths = collect_unixbench_run_paths(dataset_root)
    x, meta = build_score_table(run_paths)
    labels = list(INDEX_SUITE_TEST_IDS)
    corr = correlation_matrix(x, args.method)

    write_csv(corr, labels, out_base.with_name(out_base.name + "_matrix").with_suffix(".csv"))

    meta_path = out_base.with_name(out_base.name + "_meta").with_suffix(".json")
    off_diag = corr[~np.eye(len(labels), dtype=bool)]
    summary = {
        "schema": "moebench.paper.unixbench_subtest_correlation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "score_field": "index_detail.index (fallback: score)",
        "n_samples": int(x.shape[0]),
        "n_subtests": len(labels),
        "test_ids": labels,
        "mean_off_diagonal_corr": float(np.mean(off_diag)),
        "median_off_diagonal_corr": float(np.median(off_diag)),
        "min_off_diagonal_corr": float(np.min(off_diag)),
        "max_off_diagonal_corr": float(np.max(off_diag)),
        "hosts": sorted({m.get("config_label") or m.get("hostname_slug") for m in meta}),
        "runs": meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {meta_path}", file=sys.stderr)

    plot_heatmap(corr, labels, method=args.method, n_samples=x.shape[0], out_base=out_base)

    print(
        f"Correlation summary ({args.method}): "
        f"mean={summary['mean_off_diagonal_corr']:.3f}, "
        f"median={summary['median_off_diagonal_corr']:.3f}, "
        f"range=[{summary['min_off_diagonal_corr']:.3f}, {summary['max_off_diagonal_corr']:.3f}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
