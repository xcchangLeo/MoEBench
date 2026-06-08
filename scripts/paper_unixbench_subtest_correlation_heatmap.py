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

Combined Pearson + Spearman figure (``--combined``):
  - paper/images/unixbench_subtest_correlation_combined.pdf
  - paper/images/unixbench_subtest_correlation_combined.png
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
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


def _short_labels(labels: list[str]) -> list[str]:
    short = {
        "dhry2reg": "dhry2",
        "whetstone-double": "whet",
        "fsbuffer": "fsbuf",
        "context1": "ctx1",
        "shell1": "sh1",
        "shell8": "sh8",
    }
    return [short.get(x, x) for x in labels]


def _strong_corr_cmap() -> LinearSegmentedColormap:
    colors = [
        "#053061",
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
        "#67001F",
    ]
    return LinearSegmentedColormap.from_list("corr_strong", colors, N=256)


def _corr_color_norm(corr: np.ndarray) -> Normalize:
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    vmin = max(0.0, float(np.floor(float(np.min(off)) * 10) / 10) - 0.05)
    return Normalize(vmin=vmin, vmax=1.0)


def load_matrix_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    data = [[float(x) for x in row[1:]] for row in rows[1:]]
    return np.asarray(data, dtype=float), labels


def plot_heatmap_panel(
    ax,
    corr: np.ndarray,
    labels: list[str],
    *,
    method: str,
    n_samples: int,
    cmap,
    norm: Normalize,
) -> None:
    tick = _short_labels(labels)
    n = len(labels)
    ax.imshow(corr, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick, fontsize=9)
    method_title = "Pearson" if method == "pearson" else "Spearman"
    ax.set_title(f"{method_title} (n={n_samples})", fontsize=11, pad=8)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            t = (val - norm.vmin) / (norm.vmax - norm.vmin)
            color = "white" if t > 0.62 or t < 0.18 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=color)


def plot_combined_heatmap(
    pearson: np.ndarray,
    spearman: np.ndarray,
    labels: list[str],
    *,
    n_samples: int,
    out_base: Path,
    layout: str = "horizontal",
) -> None:
    """Render Pearson + Spearman panels.

    ``horizontal``: side-by-side panels sized for a single paper column.
    ``vertical``: stacked panels for a single paper column.
    ``horizontal-wide``: side-by-side panels for a full-page (two-column) span.
    """
    cmap = _strong_corr_cmap()
    norm = _corr_color_norm(pearson)
    if layout == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05), dpi=150)
        fig.subplots_adjust(left=0.13, right=0.86, bottom=0.28, top=0.82, wspace=0.42)
        tick_fs, ann_fs, title_fs = 5.5, 4.2, 7.0
    elif layout == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 7.2), dpi=150)
        fig.subplots_adjust(left=0.22, right=0.92, bottom=0.06, top=0.94, hspace=0.38)
        tick_fs, ann_fs, title_fs = 7.5, 6.0, 9.5
    elif layout == "horizontal-wide":
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=150)
        fig.subplots_adjust(left=0.07, right=0.88, bottom=0.18, top=0.88, wspace=0.22)
        tick_fs, ann_fs, title_fs = 10.0, 7.5, 15.0
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    axes_flat = np.atleast_1d(axes).ravel()
    for ax, mat, method in zip(axes_flat, (pearson, spearman), ("pearson", "spearman")):
        tick = _short_labels(labels)
        n = len(labels)
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(tick, fontsize=tick_fs)
        method_title = "Pearson" if method == "pearson" else "Spearman"
        ax.set_title(f"{method_title} (n={n_samples})", fontsize=title_fs, pad=4)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                t = (val - norm.vmin) / (norm.vmax - norm.vmin)
                color = "white" if t > 0.62 or t < 0.18 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=ann_fs, color=color)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes_flat.tolist(),
        fraction=0.04 if layout == "horizontal" else 0.035,
        pad=0.02,
    )
    cbar.set_label("Correlation coefficient", fontsize=7.5 if layout == "horizontal" else 9)
    cbar.ax.tick_params(labelsize=7 if layout == "horizontal" else 8)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf}", file=sys.stderr)
    print(f"Wrote {png}", file=sys.stderr)


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
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Plot Pearson + Spearman from existing *_matrix.csv under --out-dir",
    )
    ap.add_argument(
        "--combined-layout",
        choices=("horizontal", "vertical", "horizontal-wide"),
        default="horizontal-wide",
        help="Panel arrangement for --combined (default: horizontal-wide, two-column paper figure)",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=_REPO_ROOT / "paper" / "images",
        help="Output directory for --combined figure",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.combined:
        pearson_csv = out_dir / "unixbench_subtest_correlation_heatmap_matrix.csv"
        spearman_csv = out_dir / "unixbench_subtest_correlation_heatmap_spearman_matrix.csv"
        pearson_meta_path = out_dir / "unixbench_subtest_correlation_heatmap_meta.json"
        if not pearson_csv.is_file() or not spearman_csv.is_file():
            raise SystemExit(f"Missing CSV under {out_dir}; run pearson and spearman exports first.")
        pearson, labels = load_matrix_csv(pearson_csv)
        spearman, labels2 = load_matrix_csv(spearman_csv)
        if labels != labels2:
            raise SystemExit("Pearson/Spearman label order mismatch.")
        n_samples = 25
        if pearson_meta_path.is_file():
            with open(pearson_meta_path, encoding="utf-8") as f:
                n_samples = int(json.load(f).get("n_samples", n_samples))
        images_dir = args.images_dir.resolve()
        images_dir.mkdir(parents=True, exist_ok=True)
        plot_combined_heatmap(
            pearson,
            spearman,
            labels,
            n_samples=n_samples,
            out_base=images_dir / "unixbench_subtest_correlation_combined",
            layout=args.combined_layout,
        )
        return 0

    dataset_root = args.dataset_root.resolve()
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
