#!/usr/bin/env python3
"""Run paper supplementary CV (Top-K / policy / xi ablation) per machine and aggregate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moebench.dataset_machines import glob_for_machine, list_machines_in_dataset
from moebench.paper_eval.summarize import (
    balanced_score,
    summarize_policy_report,
    summarize_topk_report,
    summarize_xi_ablation_report,
)
from moebench.phoronix.training_data import canonical_test_ids_from_runs, collect_phoronix_run_paths
from moebench.reconstruct.data import collect_unixbench_run_paths
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS

EXP = REPO / "dataset" / "experiments"
OUT_DIR = REPO / "dataset" / "paper_supplementary"

MACHINES_UB_CPU = [
    "aces-System-Product-Name",
    "iZbp1glgt48i9a8d49embxZ",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp16krl0yc7euw7sb6slZ",
    "iZbp1acaw5wdllhz47922rZ",
]

SUITE_CFG = {
    "unixbench": {
        "suite_key": "unixbench",
        "benchmark": "unixbench",
        "pts_suite": None,
        "grid_prefix": "router_recon_grid_unixbench_",
        "router_file": "router_lgbm.pkl",
        "model_type": "xgboost",
        "log_flag": "--log1p-partial-index",
        "train_k_max": 6,
        "k_sweep": "1,2,3,4,5,6",
        "hosts": MACHINES_UB_CPU,
    },
    "phoronix_cpu": {
        "suite_key": "phoronix_cpu",
        "benchmark": "phoronix",
        "pts_suite": "cpu",
        "grid_prefix": "router_recon_grid_cpu_",
        "router_file": "router_mlp.pt",
        "model_type": "lightgbm",
        "log_flag": "--log1p-partial-value",
        "train_k_max": 12,
        "k_sweep": "1,2,3,4,5",
        "hosts": MACHINES_UB_CPU,
    },
    "phoronix_gpu": {
        "suite_key": "phoronix_gpu",
        "benchmark": "phoronix",
        "pts_suite": "pts/nvidia-gpu-compute",
        "grid_prefix": "router_recon_grid_pts_nvidia-gpu-compute_",
        "router_file": "router_lgbm.pkl",
        "model_type": "lightgbm",
        "log_flag": "--log1p-partial-value",
        "train_k_max": 12,
        "k_sweep": "1,2,3,4,5",
        "hosts": ["aces-System-Product-Name"],
    },
}


def _num_components(machine: str, cfg: dict[str, Any]) -> int:
    ds_root = REPO / "dataset"
    glob_pat = glob_for_machine(
        benchmark=cfg["benchmark"],
        machine=machine,
        pts_suite=cfg["pts_suite"],
    )
    if cfg["suite_key"] == "unixbench":
        paths = collect_unixbench_run_paths(ds_root, glob_pattern=glob_pat)
        if not paths:
            return len(INDEX_SUITE_TEST_IDS)
        return len(INDEX_SUITE_TEST_IDS)
    paths = collect_phoronix_run_paths(
        ds_root, glob_pattern=glob_pat, pts_suite=cfg["pts_suite"]
    )
    if not paths:
        return 1
    return max(1, len(canonical_test_ids_from_runs(paths)))


def _cap_k_sweep(k_sweep: str, max_k: int) -> str:
    vals = [int(x.strip()) for x in k_sweep.split(",") if x.strip()]
    kept = [k for k in vals if 0 < k <= max_k]
    if not kept:
        kept = [min(max_k, 3)]
    return ",".join(str(k) for k in kept)


def _cap_eval_k(eval_k: int, max_k: int) -> int:
    return max(1, min(int(eval_k), max_k))


def _latest_grid(prefix: str, machine: str) -> Path | None:
    pat = re.compile(rf"^{re.escape(prefix)}{re.escape(machine)}_(\d{{8}}T\d{{6}}Z)$")
    best: tuple[str, Path] | None = None
    for p in EXP.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m and (best is None or m.group(1) > best[0]):
            best = (m.group(1), p)
    return best[1] if best else None


def _router_path(machine: str, cfg: dict[str, Any]) -> Path:
    d = _latest_grid(cfg["grid_prefix"], machine)
    if d is None:
        raise FileNotFoundError(f"No grid for {machine} prefix={cfg['grid_prefix']}")
    p = d / "trained_models" / cfg["router_file"]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def _run_cv(
    *,
    machine: str,
    cfg: dict[str, Any],
    mode: str,
    k_sweep: str,
    eval_k: int,
    policies: str,
    xi_ablations: str,
    out_json: Path,
    auto_install: bool,
    skip_existing: bool,
) -> None:
    if skip_existing and out_json.is_file():
        print(f"skip existing {out_json}", flush=True)
        return
    glob_pat = glob_for_machine(
        benchmark=cfg["benchmark"],
        machine=machine,
        pts_suite=cfg["pts_suite"],
    )
    max_k = _num_components(machine, cfg)
    k_sweep_eff = _cap_k_sweep(k_sweep, max_k) if k_sweep else k_sweep
    eval_k_eff = _cap_eval_k(eval_k, max_k)
    router = _router_path(machine, cfg)
    sk = cfg["suite_key"]
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "paper_reconstruct_cv_extras.py"),
        "--dataset-root",
        str(REPO / "dataset"),
        "--suites",
        sk,
        "--cv-mode",
        "leave_one_session_out",
        "--seed",
        "42",
        "--model-type",
        cfg["model_type"],
        cfg["log_flag"],
        "--train-aug",
        "10",
        "--train-k-max",
        str(cfg["train_k_max"]),
        "--xi-ablations",
        xi_ablations,
        "--report-json",
        str(out_json),
    ]
    if auto_install:
        cmd.append("--auto-install")
    if sk == "unixbench":
        cmd.extend(["--glob-unixbench", glob_pat, "--router-model-unixbench", str(router)])
    elif sk == "phoronix_cpu":
        cmd.extend(["--glob-pts-cpu", glob_pat, "--router-model-pts-cpu", str(router)])
    else:
        cmd.extend(["--glob-pts-gpu", glob_pat, "--router-model-pts-gpu", str(router)])

    if mode == "topk":
        cmd.extend(["--k-sweep", k_sweep_eff, "--policies", "router"])
    elif mode == "policy":
        cmd.extend(["--eval-partial-k", str(eval_k_eff), "--policies", policies])
    else:
        cmd.extend(["--eval-partial-k", str(eval_k_eff), "--policies", "router"])

    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _mean_or_none(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and v == v]
    return statistics.mean(xs) if xs else None


def _combo_key(c: dict[str, Any]) -> tuple[Any, ...]:
    return (
        c.get("suite_key"),
        c.get("eval_partial_k"),
        c.get("policy"),
        c.get("xi_ablation"),
    )


def _metrics_from_combo(c: dict[str, Any]) -> dict[str, float | None]:
    oof = c.get("oof_metrics") or {}
    ts = c.get("time_savings") or {}
    return {
        "mae_suite_index": oof.get("mae_suite_index"),
        "rmse_suite_index": oof.get("rmse_suite_index"),
        "spearman_suite": oof.get("spearman_suite"),
        "mean_wall_time_saved_fraction": ts.get("mean_fraction_wall_time_saved_vs_full_suite"),
    }


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    """Merge per-machine reports by averaging combo metrics."""
    buckets: dict[tuple[Any, ...], list[dict[str, float | None]]] = {}
    suite_meta: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        for block in rep.get("suite_results") or []:
            sk = block.get("suite_key")
            suite_meta.setdefault(sk, {"suite_key": sk, "n_samples": []})
            suite_meta[sk]["n_samples"].append(block.get("n_samples"))
            for c in block.get("combinations") or []:
                c = dict(c)
                c["suite_key"] = sk
                buckets.setdefault(_combo_key(c), []).append(_metrics_from_combo(c))

    merged_combos: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items(), key=lambda x: x[0]):
        sk, ek, pol, xi = key
        mae = _mean_or_none([r["mae_suite_index"] for r in rows])
        rmse = _mean_or_none([r["rmse_suite_index"] for r in rows])
        sp = _mean_or_none([r["spearman_suite"] for r in rows])
        saved = _mean_or_none([r["mean_wall_time_saved_fraction"] for r in rows])
        merged_combos.append(
            {
                "suite_key": sk,
                "eval_partial_k": ek,
                "policy": pol,
                "xi_ablation": xi,
                "oof_metrics": {
                    "mae_suite_index": mae,
                    "rmse_suite_index": rmse,
                    "spearman_suite": sp,
                },
                "time_savings": {"mean_fraction_wall_time_saved_vs_full_suite": saved},
                "n_machines": len(rows),
            }
        )

    by_suite: dict[str, list[dict[str, Any]]] = {}
    for c in merged_combos:
        by_suite.setdefault(str(c["suite_key"]), []).append(c)

    suite_results = []
    for sk, combos in by_suite.items():
        meta = suite_meta.get(sk, {})
        suite_results.append(
            {
                "suite_key": sk,
                "n_samples_mean": _mean_or_none(meta.get("n_samples", [])),
                "combinations": combos,
            }
        )
    return {"schema": "moebench.paper_supplementary.merged.v1", "suite_results": suite_results}


def run_mode(mode: str, *, eval_k: int, auto_install: bool, skip_existing: bool = True) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_machine: list[Path] = []

    for suite_name, cfg in SUITE_CFG.items():
        if mode == "topk":
            k_sweep, policies, xi = cfg["k_sweep"], "router", "full"
        elif mode == "policy":
            k_sweep, policies, xi = "", (
                "random,fixed_first_k,fixed_cpu_mix,fixed_io_mix,greedy_slowest,greedy_fastest,router"
            ), "full"
        else:
            k_sweep, policies, xi = "", "router", "full,static_hw_only,no_perf_pmu,no_dynamic_proc,no_gpu"

        for machine in cfg["hosts"]:
            out = OUT_DIR / f"{mode}_{suite_name}_{machine}.json"
            _run_cv(
                machine=machine,
                cfg=cfg,
                mode=mode,
                k_sweep=k_sweep,
                eval_k=eval_k,
                policies=policies,
                xi_ablations=xi,
                out_json=out,
                auto_install=auto_install,
                skip_existing=skip_existing,
            )
            per_machine.append(out)

    merged_path = OUT_DIR / f"{mode}_three_suites_merged.json"
    summary_path = OUT_DIR / f"{mode}_three_suites_summary.json"
    merged = merge_reports(per_machine)
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if mode == "topk":
        summary = summarize_topk_report(merged)
    elif mode == "policy":
        summary = summarize_policy_report(merged)
    else:
        summary = summarize_xi_ablation_report(merged)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {merged_path} and {summary_path}", flush=True)
    return merged_path, summary_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("topk", "policy", "xi_ablation", "all"),
        default="all",
    )
    ap.add_argument("--eval-k", type=int, default=3, help="Fixed K for policy/xi (updated after topk if needed)")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    modes = ["topk", "policy", "xi_ablation"] if args.mode == "all" else [args.mode]
    eval_k = args.eval_k
    skip_existing = not args.no_skip_existing
    for mode in modes:
        _, summary_path = run_mode(
            mode, eval_k=eval_k, auto_install=args.auto_install, skip_existing=skip_existing
        )
        if mode == "topk":
            topk = json.loads(summary_path.read_text(encoding="utf-8"))
            rec = {s["suite_key"]: s.get("recommended_k") for s in topk.get("suites", [])}
            print(f"Recommended K from topk: {rec}", flush=True)
            # Use UB recommended K for policy if only one eval_k globally; keep 3 as default min
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
