#!/usr/bin/env python3
"""Aggregate experiment JSON under dataset/experiments/ for paper tables."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO / "dataset" / "experiments"

MACHINES = {
    "aces-System-Product-Name": "32U128G",
    "iZbp1glgt48i9a8d49embxZ": "2U8G",
    "iZbp15n87643uk1sqjrdvdZ": "4U8G",
    "iZbp16krl0yc7euw7sb6slZ": "4U16G",
    "iZbp1acaw5wdllhz47922rZ": "8U8G",
}

HOST_ORDER = ["32U128G", "2U8G", "4U8G", "4U16G", "8U8G"]

SUITE_SPECS = {
    "unixbench": {
        "hybrid_prefix": "hybrid_grid_unixbench_",
        "route_a_prefix": "router_recon_grid_unixbench_",
        "probe_stem": "probe_unixbench",
        "hosts": list(MACHINES.keys()),
    },
    "pts_cpu": {
        "hybrid_prefix": "hybrid_grid_cpu_",
        "route_a_prefix": "router_recon_grid_cpu_",
        "probe_stem": "probe_pts_cpu",
        "hosts": list(MACHINES.keys()),
    },
    "pts_gpu": {
        "hybrid_prefix": "hybrid_grid_pts_nvidia-gpu-compute_",
        "route_a_prefix": "router_recon_grid_pts_nvidia-gpu-compute_",
        "probe_stem": "probe_pts_gpu",
        "hosts": ["aces-System-Product-Name"],
    },
}

ROUTER_LABEL = {"lightgbm": "LightGBM", "mlp": "MLP", "gnn_expert": "GNN-expert"}
RECON_LABEL = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latest_dir(prefix: str, machine: str) -> Path | None:
    pat = re.compile(rf"^{re.escape(prefix)}{re.escape(machine)}_(\d{{8}}T\d{{6}}Z)$")
    best: tuple[str, Path] | None = None
    for p in EXP_ROOT.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            stamp = m.group(1)
            if best is None or stamp > best[0]:
                best = (stamp, p)
    return best[1] if best else None


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return "---"
    return f"{100.0 * x:.{digits}f}"


def _num(x: float | None, digits: int = 1) -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return "---"
    return f"{x:.{digits}f}"


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
    return statistics.mean(vals) if vals else None


def _hybrid_wall_from_exp(path: Path) -> float | None:
    d = _load(path)
    runs = [r for r in d.get("per_run", []) if not r.get("skipped")]
    if not runs:
        return None
    walls = []
    for r in runs:
        t = (r.get("timing_seconds") or {}).get("hybrid_wall_estimate")
        if t is not None:
            walls.append(float(t))
    return _mean(walls)


def _best_hybrid_per_host(suite_key: str) -> dict[str, dict[str, Any]]:
    spec = SUITE_SPECS[suite_key]
    out: dict[str, dict[str, Any]] = {}
    for machine in spec["hosts"]:
        ddir = _latest_dir(spec["hybrid_prefix"], machine)
        if ddir is None:
            continue
        summary = _load(ddir / "grid_summary.json")
        combos = summary.get("per_combination") or summary.get("ranked") or []
        if not combos:
            continue
        best = min(combos, key=lambda c: float(c.get("mean_suite_rel_err") or 1e18))
        exp_path = Path(best["experiment_json"])
        out[machine] = {
            **best,
            "hybrid_wall_s": _hybrid_wall_from_exp(exp_path),
            "full_s": _route_a_full_from_exp(exp_path),
        }
    return out


def _route_a_full_from_exp(hybrid_exp: Path) -> float | None:
    d = _load(hybrid_exp)
    runs = [r for r in d.get("per_run", []) if not r.get("skipped")]
    vals = []
    for r in runs:
        t = (r.get("timing_seconds") or {}).get("full_suite_from_dataset")
        if t is not None:
            vals.append(float(t))
    return _mean(vals)


def _best_route_a_per_host(suite_key: str) -> dict[str, dict[str, Any]]:
    spec = SUITE_SPECS[suite_key]
    out: dict[str, dict[str, Any]] = {}
    for machine in spec["hosts"]:
        ddir = _latest_dir(spec["route_a_prefix"], machine)
        if ddir is None:
            continue
        summary_path = ddir / "grid_summary.json"
        if not summary_path.is_file():
            # synthesize from exp files if missing
            combos = []
            for exp in sorted(ddir.glob("exp_*.json")):
                if "__probe_" in exp.name:
                    continue
                d = _load(exp)
                rel = d.get("comparison", {}).get("suite_relative_error")
                if rel is None:
                    rel = d.get("suite_rel_err")
                timing = d.get("timing_seconds") or {}
                partial = timing.get("partial_unixbench") or timing.get("partial_benchmark_s")
                full = timing.get("full_unixbench") or timing.get("full_benchmark_s")
                if rel is None:
                    continue
                combos.append(
                    {
                        "suite_rel_err": rel,
                        "partial_benchmark_s": partial,
                        "full_benchmark_s": full,
                        "experiment_json": str(exp),
                        "router_type": d.get("router_model_type"),
                        "reconstruct_type": None,
                    }
                )
            if not combos:
                continue
            best = min(combos, key=lambda c: float(c["suite_rel_err"]))
        else:
            summary = _load(summary_path)
            ranked = summary.get("ranked") or summary.get("per_combination") or []
            if not ranked:
                continue
            best = ranked[0]
        saved = None
        if best.get("full_benchmark_s") and best.get("partial_benchmark_s"):
            full = float(best["full_benchmark_s"])
            part = float(best["partial_benchmark_s"])
            if full > 0:
                saved = (full - part) / full
        out[machine] = {**best, "saved_frac": saved or best.get("time_saved_benchmark_s")}
    return out


def _probe_row(machine: str, stem: str, backend: str) -> dict[str, Any] | None:
    path = EXP_ROOT / machine / f"{stem}_{backend}.json"
    if not path.is_file():
        return None
    d = _load(path)
    sc = d.get("suite_comparison") or {}
    return {
        "rel_err": sc.get("relative_error"),
        "mae": sc.get("abs_error"),
        "probe_time_s": d.get("estimated_probe_wall_s"),
        "backend": backend,
    }


def _best_route_b_per_host(suite_key: str) -> dict[str, dict[str, Any]]:
    spec = SUITE_SPECS[suite_key]
    out: dict[str, dict[str, Any]] = {}
    for machine in spec["hosts"]:
        rows = []
        for backend in ("lgbm", "xgb"):
            r = _probe_row(machine, spec["probe_stem"], backend)
            if r:
                rows.append(r)
        if not rows:
            continue
        out[machine] = min(rows, key=lambda r: float(r["rel_err"] or 1e18))
    return out


def _overall_row(method: str, per_host: dict[str, dict[str, Any]], *, rel_key: str, partial_key: str, full_key: str) -> dict[str, Any]:
    rels, partials, fulls, saveds, maes = [], [], [], [], []
    for machine, row in per_host.items():
        if rel_key in row and row[rel_key] is not None:
            rels.append(float(row[rel_key]))
        elif row.get("rel_err") is not None:
            rels.append(float(row["rel_err"]))
        p = row.get(partial_key) or row.get("partial_benchmark_s") or row.get("hybrid_wall_s") or row.get("probe_time_s")
        f = row.get(full_key) or row.get("full_benchmark_s") or row.get("full_s")
        if p is not None:
            partials.append(float(p))
        if f is not None:
            fulls.append(float(f))
        sf = row.get("mean_fraction_saved") or row.get("saved_frac")
        if sf is not None:
            saveds.append(float(sf))
        elif f and p and float(f) > 0:
            saveds.append((float(f) - float(p)) / float(f))
        mae = row.get("suite_abs_err") or row.get("mae")
        if mae is not None:
            maes.append(float(mae))
    return {
        "method": method,
        "rel_err_mean": _mean(rels),
        "partial_mean": _mean(partials),
        "full_mean": _mean(fulls),
        "saved_mean": _mean(saveds),
        "mae_mean": _mean(maes),
    }


def aggregate_model_combo(suite_key: str) -> dict[tuple[str, str], dict[str, float | None]]:
    spec = SUITE_SPECS[suite_key]
    cell: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for machine in spec["hosts"]:
        ddir = _latest_dir(spec["hybrid_prefix"], machine)
        if ddir is None:
            continue
        summary = _load(ddir / "grid_summary.json")
        for c in summary.get("per_combination") or []:
            rt = c["router_type"]
            rc = c["reconstruct_type"]
            rel = float(c.get("mean_suite_rel_err") or 0)
            exp = Path(c["experiment_json"])
            wall = _hybrid_wall_from_exp(exp) or 0.0
            cell.setdefault((rt, rc), []).append((rel, wall))
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for key, pairs in cell.items():
        rels = [p[0] for p in pairs]
        walls = [p[1] for p in pairs]
        out[key] = {"rel_err": _mean(rels), "wall_s": _mean(walls)}
    return out


def main() -> int:
    report: dict[str, Any] = {}

    for suite_key, suite_label in (
        ("unixbench", "UnixBench"),
        ("pts_cpu", "PTS-CPU"),
        ("pts_gpu", "PTS-GPU"),
    ):
        hybrid = _best_hybrid_per_host(suite_key)
        route_a = _best_route_a_per_host(suite_key)
        route_b = _best_route_b_per_host(suite_key)

        full_hosts = {m: route_a.get(m) or hybrid.get(m) for m in SUITE_SPECS[suite_key]["hosts"]}
        full_t = _mean(
            [
                float(r.get("full_benchmark_s") or r.get("full_s") or 0)
                for r in full_hosts.values()
                if (r.get("full_benchmark_s") or r.get("full_s"))
            ]
        )

        rows = [
            {
                "method": "Full suite",
                "rel_err_mean": 0.0,
                "partial_mean": None,
                "full_mean": full_t,
                "saved_mean": 0.0,
                "mae_mean": None,
            },
            _overall_row(
                "Hybrid (main)",
                hybrid,
                rel_key="mean_suite_rel_err",
                partial_key="hybrid_wall_s",
                full_key="full_s",
            ),
            _overall_row(
                "Route A (abl.)",
                route_a,
                rel_key="suite_rel_err",
                partial_key="partial_benchmark_s",
                full_key="full_benchmark_s",
            ),
            _overall_row(
                "Route B (abl.)",
                route_b,
                rel_key="rel_err",
                partial_key="probe_time_s",
                full_key="full_benchmark_s",
            ),
        ]

        combo = aggregate_model_combo(suite_key)
        best_combo = min(combo.items(), key=lambda kv: float(kv[1]["rel_err"] or 1e18)) if combo else None

        probe_agg: dict[str, Any] = {}
        for backend in ("lgbm", "xgb"):
            rels, maes, times = [], [], []
            for machine in SUITE_SPECS[suite_key]["hosts"]:
                r = _probe_row(machine, SUITE_SPECS[suite_key]["probe_stem"], backend)
                if not r:
                    continue
                rels.append(float(r["rel_err"]))
                maes.append(float(r["mae"]))
                times.append(float(r["probe_time_s"]))
            probe_agg[backend] = {
                "rel_err": _mean(rels),
                "mae": _mean(maes),
                "probe_time_s": _mean(times),
            }

        per_host_probe: dict[str, dict[str, float | None]] = {}
        if suite_key == "unixbench":
            for machine, label in MACHINES.items():
                per_host_probe[label] = {}
                for backend in ("lgbm", "xgb"):
                    r = _probe_row(machine, "probe_unixbench", backend)
                    per_host_probe[label][backend] = r["rel_err"] if r else None

        report[suite_key] = {
            "label": suite_label,
            "overall": rows,
            "model_combo": {
                f"{ROUTER_LABEL.get(rt, rt)}|{RECON_LABEL.get(rc, rc)}": {
                    "rel_err_pct": _pct(v["rel_err"], 2),
                    "wall_s": _num(v["wall_s"], 1),
                    "rel_err_raw": v["rel_err"],
                    "wall_raw": v["wall_s"],
                }
                for (rt, rc), v in sorted(combo.items())
            },
            "best_combo": (
                {
                    "router": ROUTER_LABEL.get(best_combo[0][0], best_combo[0][0]),
                    "reconstructor": RECON_LABEL.get(best_combo[0][1], best_combo[0][1]),
                    "rel_err_pct": _pct(best_combo[1]["rel_err"]),
                }
                if best_combo
                else None
            ),
            "probe_models": probe_agg,
            "probe_per_host_unixbench": per_host_probe,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
