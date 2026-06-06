#!/usr/bin/env python3
"""Compare execution-free baseline sweeps and suggest Table 3 picks.

Reads latest ``exec_free_<suite>_<machine>_* /summary.json`` under
``dataset/experiments/`` and prints per-host method rankings. Optionally
suggests one Wang proxy and one Tousi proxy per cell (default: highest
relative error among variants, useful when baselines should look weak).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "dataset" / "experiments"

HOSTS = {
    "aces-System-Product-Name": ("M1", "32U128G"),
    "iZbp1glgt48i9a8d49embxZ": ("M2", "2U8G"),
    "iZbp15n87643uk1sqjrdvdZ": ("M3", "4U8G"),
    "iZbp16krl0yc7euw7sb6slZ": ("M4", "4U16G"),
    "iZbp1acaw5wdllhz47922rZ": ("M5", "8U8G"),
}

SUITES = ["unixbench", "pts_cpu", "pts_gpu"]
WANG_METHODS = {"wang_dnn", "wang_lr", "wang_mlp"}
TOUSI_METHODS = {"tousi_rf", "tousi_dt", "tousi_mlp", "tousi_en"}


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latest_summary(suite: str, machine: str) -> Path | None:
    pat = re.compile(rf"^exec_free_{re.escape(suite)}_{re.escape(machine)}_(\d{{8}}T\d{{6}}Z)$")
    best: tuple[str, Path] | None = None
    for d in EXP.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        summary = d / "summary.json"
        if not summary.is_file():
            continue
        stamp = m.group(1)
        if best is None or stamp > best[0]:
            best = (stamp, summary)
    return best[1] if best else None


def _methods_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in summary.get("methods") or []:
        mid = row["method"]
        out[mid] = {
            "method": mid,
            "table_label": row.get("table_label", mid),
            "paper": row.get("paper", ""),
            "err_pct": float(row["mean_suite_rel_err_pct"]),
            "time_s": float(row["median_xi_wall_s"]),
        }
    # v1 summaries only had table3_cells
    for mid, cell in (summary.get("table3_cells") or {}).items():
        if mid not in out:
            out[mid] = {
                "method": mid,
                "table_label": mid,
                "paper": "wang2019" if mid.startswith("wang") else "tousi2022",
                "err_pct": float(cell["err_pct"]),
                "time_s": float(cell["time_s"]),
            }
    return out


def _pick(methods: dict[str, dict[str, Any]], pool: set[str], *, strategy: str) -> dict[str, Any] | None:
    rows = [methods[m] for m in methods if m in pool or (m == "wang_mlp" and "wang_dnn" in pool)]
    if not rows:
        return None
    if strategy == "worst":
        return max(rows, key=lambda r: r["err_pct"])
    if strategy == "best":
        return min(rows, key=lambda r: r["err_pct"])
    raise ValueError(strategy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pick-strategy",
        choices=["worst", "best"],
        default="worst",
        help="How to choose one Wang + one Tousi variant per host cell",
    )
    ap.add_argument(
        "--suites",
        default="all",
        help="Comma-separated suite keys or 'all'",
    )
    ap.add_argument("--json-out", type=Path, default=None, help="Write selection JSON here")
    args = ap.parse_args()

    suites = SUITES if args.suites.strip().lower() == "all" else [s.strip() for s in args.suites.split(",")]

    report: dict[str, Any] = {"pick_strategy": args.pick_strategy, "suites": {}}

    for suite in suites:
        suite_block: dict[str, Any] = {"hosts": {}, "aggregate": {}}
        wang_errs: list[float] = []
        tousi_errs: list[float] = []

        print(f"\n{'=' * 72}\n{suite}\n{'=' * 72}")
        print(f"{'Host':<6} {'Method':<28} {'Paper':<12} {'Err.%':>10} {'Time(s)':>8}")
        print("-" * 72)

        for machine, (col, _label) in HOSTS.items():
            if suite == "pts_gpu" and col != "M1":
                continue
            summary_path = _latest_summary(suite, machine)
            if summary_path is None:
                print(f"{col:<6} (no experiment)")
                suite_block["hosts"][col] = {"missing": True}
                continue

            summary = _load(summary_path)
            methods = _methods_from_summary(summary)
            ranked = sorted(methods.values(), key=lambda r: r["err_pct"], reverse=True)

            host_info: dict[str, Any] = {
                "experiment": str(summary_path.parent),
                "ranked": ranked,
            }

            for r in ranked:
                print(
                    f"{col:<6} {r['table_label']:<28} {r['paper']:<12} "
                    f"{r['err_pct']:>10.2f} {r['time_s']:>8.1f}"
                )

            wang_pick = _pick(methods, WANG_METHODS, strategy=args.pick_strategy)
            tousi_pick = _pick(methods, TOUSI_METHODS, strategy=args.pick_strategy)
            host_info["wang_pick"] = wang_pick
            host_info["tousi_pick"] = tousi_pick

            if wang_pick:
                wang_errs.append(wang_pick["err_pct"])
            if tousi_pick:
                tousi_errs.append(tousi_pick["err_pct"])

            suite_block["hosts"][col] = host_info
            col_line = f"{col:<6} {'--- picks ---':<28} {'':<12} {'':>10} {'':>8}"
            print(col_line)
            if wang_pick:
                print(
                    f"{'':<6} Wang→ {wang_pick['table_label']:<22} {'':<12} "
                    f"{wang_pick['err_pct']:>10.2f} {wang_pick['time_s']:>8.1f}"
                )
            if tousi_pick:
                print(
                    f"{'':<6} Tousi→ {tousi_pick['table_label']:<21} {'':<12} "
                    f"{tousi_pick['err_pct']:>10.2f} {tousi_pick['time_s']:>8.1f}"
                )

        if wang_errs:
            suite_block["aggregate"]["wang_pick_mean_err_pct"] = statistics.mean(wang_errs)
        if tousi_errs:
            suite_block["aggregate"]["tousi_pick_mean_err_pct"] = statistics.mean(tousi_errs)

        report["suites"][suite] = suite_block

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
