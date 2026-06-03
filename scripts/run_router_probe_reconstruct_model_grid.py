#!/usr/bin/env python3
"""Main MoEBench pipeline (Route A + B): train router + recon + probe, then 3×2×3 hybrid grid.

Hybrid flow per evaluation run:
  xi → router Top-K → probe on selected subtests only → reconstructor → suite score
Ground-truth full suite and full-suite wall time come from collected dataset runs
(no live full-suite execution).

Full main-experiment grid: 3 routers × 2 probe regressors (LightGBM, XGBoost) × 3 reconstructors = 18 combos.

Per-machine only: pass ``--machine <hostname_slug>`` or ``--all-machines``.
Original Route-A-only and Route-B-only grids remain ablation baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    find_latest_router_recon_models_dir,
    list_machines_in_dataset,
    local_host_slug,
    machine_models_dir,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.phoronix.pipeline import safe_session_tag

ROUTER_TRAIN = REPO_ROOT / "scripts" / "router_train.py"
RECON_TRAIN = REPO_ROOT / "scripts" / "reconstruct_train_eval.py"
PROBE_COLLECT = REPO_ROOT / "scripts" / "probe_collect.py"
PROBE_TRAIN = REPO_ROOT / "scripts" / "probe_train.py"
HYBRID_EXP = REPO_ROOT / "scripts" / "experiment_router_probe_reconstruct.py"
ML_VENV_PY = REPO_ROOT / ".venv-moebench-router" / "bin" / "python3"
INSTALL_ML_DEPS = REPO_ROOT / "scripts" / "install_ml_python_deps.sh"

ROUTER_SPECS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "router_lgbm.pkl"),
    ("mlp", "router_mlp.pt"),
    ("gnn_expert", "router_gnn.pt"),
)

RECON_SPECS: tuple[tuple[str, str], ...] = (
    ("xgboost", "recon_xgb.pkl"),
    ("lightgbm", "recon_lgbm.pkl"),
    ("mlp", "recon_mlp.pt"),
)

PROBE_SPECS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "lgbm"),
    ("xgboost", "xgb"),
)

_PROBE_MODEL_SUFFIX = {name: suffix for name, suffix in PROBE_SPECS}


def _probe_backends_for_args(probe_backend: str) -> list[str]:
    if probe_backend == "all":
        return [name for name, _ in PROBE_SPECS]
    return [probe_backend]


def _run(cmd: list[str], *, dry_run: bool, check: bool = True) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def _experiment_json_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("schema") == "moebench.experiment.router_probe_reconstruct.v1"
    except Exception:
        return False


def _python_import_ok(py: str, module: str) -> bool:
    return subprocess.run([py, "-c", f"import {module}"], capture_output=True).returncode == 0


def _resolve_python(*, require_ml: bool = False) -> str:
    if ML_VENV_PY.is_file() and (not require_ml or _python_import_ok(str(ML_VENV_PY), "numpy")):
        return str(ML_VENV_PY)
    if not require_ml or _python_import_ok(sys.executable, "numpy"):
        return sys.executable
    return sys.executable


def _ensure_ml_python(auto_install: bool) -> str:
    py = _resolve_python(require_ml=True)
    if _python_import_ok(py, "numpy"):
        return py
    if auto_install and INSTALL_ML_DEPS.is_file():
        install_args = [str(INSTALL_ML_DEPS)]
        if not os.environ.get("CONDA_PREFIX"):
            install_args.append("--use-venv")
        subprocess.check_call(["bash", *install_args])
        py = _resolve_python(require_ml=True)
    if not _python_import_ok(py, "numpy"):
        raise SystemExit("Missing numpy/sklearn; run scripts/install_ml_python_deps.sh --use-venv")
    return py


def _pts_probe_token(pts_suite: str | None) -> str:
    """Canonical probe artifact basename token (matches run_probe_three_suites.sh)."""
    s = (pts_suite or "cpu").strip().lower()
    if s in ("cpu", "pts/cpu"):
        return "pts_cpu"
    if "gpu" in s or "nvidia" in s:
        return "pts_gpu"
    return safe_session_tag(s.replace("/", "_"))


def _probe_paths(
    *,
    models_dir: Path,
    benchmark: str,
    pts_suite: str | None,
    probe_backend: str,
) -> tuple[Path, Path]:
    suffix = _PROBE_MODEL_SUFFIX.get(probe_backend, probe_backend)
    if benchmark == "unixbench":
        ds_name = "probe_dataset_unixbench.json"
        model_name = f"probe_unixbench_{suffix}.pkl"
    else:
        tok = _pts_probe_token(pts_suite)
        ds_name = f"probe_dataset_{tok}.json"
        model_name = f"probe_{tok}_{suffix}.pkl"
    return models_dir / ds_name, models_dir / model_name


def _summarize_hybrid(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    agg = d.get("aggregate") or {}
    timing = d.get("timing_seconds") or {}
    if not timing and d.get("per_run"):
        valid = [r for r in d["per_run"] if not r.get("skipped")]
        if valid:
            timing = valid[0].get("timing_seconds") or {}
    comp = d.get("comparison") or {}
    if not comp and d.get("per_run"):
        rels = [
            r["comparison"]["suite_relative_error"]
            for r in d["per_run"]
            if not r.get("skipped") and r.get("comparison")
        ]
        if rels:
            comp = {"suite_relative_error": sum(rels) / len(rels)}
    return {
        "experiment_json": str(path),
        "mode": d.get("mode"),
        "mean_suite_rel_err": agg.get("mean_suite_relative_error") or comp.get("suite_relative_error"),
        "mean_fraction_saved": agg.get("mean_fraction_saved_vs_full"),
        "num_runs_evaluated": agg.get("num_runs_evaluated"),
        "probe_model": d.get("probe_model"),
        "probe_backend": d.get("probe_backend"),
        "router_model": d.get("router_model"),
        "reconstruct_model": d.get("reconstruct_model"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--benchmark", choices=("unixbench", "phoronix"), default="unixbench")
    ap.add_argument("--pts-suite", type=str, default="cpu")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--all-machines", action="store_true")
    ap.add_argument("--glob-pattern", type=str, default="")
    ap.add_argument("--out-parent", type=str, default="")
    ap.add_argument(
        "--models-dir",
        type=str,
        default="",
        help="Reuse existing router/recon checkpoints (e.g. prior router_recon_grid/trained_models/)",
    )
    ap.add_argument(
        "--stage",
        choices=("all", "routers", "reconstructors", "probe", "grid"),
        default="grid",
        help="Default grid: offline 18-combo eval only (reuse existing checkpoints). Use --train for full pipeline.",
    )
    ap.add_argument(
        "--train",
        action="store_true",
        help="Train routers + reconstructors + probe (if missing), then run grid (= --stage all).",
    )
    ap.add_argument(
        "--probe-backend",
        choices=("lightgbm", "xgboost", "all"),
        default="all",
        help="Probe regressor(s) in grid: lightgbm, xgboost, or all (default: both → 18 combos)",
    )
    ap.add_argument("--probe-duration-s", type=float, default=4.0)
    ap.add_argument("--probe-mode", choices=("micro", "real"), default="micro")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--xi-overhead-s", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--mlp-epochs-router", type=int, default=200)
    ap.add_argument("--mlp-epochs-recon", type=int, default=400)
    ap.add_argument("--mlp-hidden-router", type=int, default=64)
    ap.add_argument("--mlp-hidden-recon", type=int, default=128)
    ap.add_argument("--gnn-emb-dim", type=int, default=12)
    ap.add_argument("--train-aug", type=int, default=20)
    ap.add_argument("--train-k-min", type=int, default=2)
    ap.add_argument("--train-k-max", type=int, default=6)
    ap.add_argument("--eval-partial-k", type=int, default=3)
    ap.add_argument("--pts-suite-target", choices=("arithmetic_mean", "logmean"), default="logmean")
    args = ap.parse_args()
    if args.train:
        args.stage = "all"

    python_exe = _ensure_ml_python(args.auto_install)
    ds_root = Path(args.dataset_root).resolve()
    pts_suite = args.pts_suite.strip() if args.benchmark == "phoronix" else None

    if args.all_machines:
        machines = list_machines_in_dataset(ds_root, benchmark=args.benchmark, pts_suite=pts_suite)
    else:
        machines = [resolve_training_machine(args.machine or None)]

    if not machines:
        print("No machines found", file=sys.stderr)
        return 2

    batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    last_code = 0
    for machine in machines:
        code = _run_for_machine(args, machine=machine, ds_root=ds_root, batch_stamp=batch_stamp, python_exe=python_exe)
        if code != 0:
            last_code = code
    return last_code


def _run_for_machine(
    args: argparse.Namespace,
    *,
    machine: str,
    ds_root: Path,
    batch_stamp: str,
    python_exe: str,
) -> int:
    pts_suite = args.pts_suite.strip() if args.benchmark == "phoronix" else None
    glob_eff = resolve_glob_for_machine(
        benchmark=args.benchmark,
        machine=machine,
        glob_pattern=args.glob_pattern or None,
        pts_suite=pts_suite,
    )

    if args.out_parent.strip():
        out_dir = Path(args.out_parent).resolve()
        if args.all_machines:
            out_dir = out_dir / machine
    else:
        if args.benchmark == "unixbench":
            sub = f"hybrid_grid_unixbench_{machine}_{batch_stamp}"
        else:
            tok = safe_session_tag(str(pts_suite or "").replace("/", "_"))
            sub = f"hybrid_grid_{tok}_{machine}_{batch_stamp}"
        out_dir = (ds_root / "experiments" / sub).resolve()

    models_dir = out_dir / "trained_models"
    models_dir_source = "hybrid_out/trained_models"
    router_recon_reused = False
    if args.models_dir.strip():
        models_dir = Path(args.models_dir).resolve()
        models_dir_source = str(models_dir)
        router_recon_reused = True
    elif args.stage in ("grid", "probe"):
        found = find_latest_router_recon_models_dir(
            ds_root, machine=machine, benchmark=args.benchmark, pts_suite=pts_suite
        )
        if found is not None:
            models_dir = found
            models_dir_source = f"reuse:{found}"
            router_recon_reused = True
    machine_models = machine_models_dir(ds_root, machine)
    probe_backends = _probe_backends_for_args(args.probe_backend)

    print(
        f"[hybrid-grid] machine={machine!r} out={out_dir} "
        f"stage={args.stage} router_recon={models_dir_source} probe_backends={probe_backends}",
        file=sys.stderr,
    )

    do_routers = args.stage in ("all", "routers") and not router_recon_reused
    do_recon = args.stage in ("all", "reconstructors") and not router_recon_reused
    do_probe = args.stage in ("all", "probe")
    do_grid = args.stage in ("all", "grid")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if (do_routers or do_recon):
            models_dir.mkdir(parents=True, exist_ok=True)

    py = python_exe
    train_router_base = [
        py,
        str(ROUTER_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        args.benchmark,
        "--glob-pattern",
        glob_eff,
        "--machine",
        machine,
        "--top-k",
        str(args.top_k),
        "--mlp-epochs",
        str(args.mlp_epochs_router),
        "--mlp-hidden",
        str(args.mlp_hidden_router),
        "--gnn-emb-dim",
        str(args.gnn_emb_dim),
    ]
    if args.benchmark == "phoronix":
        train_router_base.extend(["--pts-suite", pts_suite or ""])
    if args.auto_install:
        train_router_base.append("--auto-install")

    train_recon_base = [
        py,
        str(RECON_TRAIN),
        "--dataset-root",
        str(ds_root),
        "--benchmark",
        args.benchmark,
        "--glob-pattern",
        glob_eff,
        "--machine",
        machine,
        "--skip-cv",
        "--no-uncertainty",
        "--train-aug",
        str(args.train_aug),
        "--train-k-min",
        str(args.train_k_min),
        "--train-k-max",
        str(args.train_k_max),
        "--eval-partial-k",
        str(args.eval_partial_k),
        "--mlp-epochs",
        str(args.mlp_epochs_recon),
        "--mlp-hidden",
        str(args.mlp_hidden_recon),
    ]
    if args.benchmark == "phoronix":
        train_recon_base.extend(
            ["--pts-suite", pts_suite or "", "--pts-suite-target", args.pts_suite_target]
        )
    if args.auto_install:
        train_recon_base.append("--auto-install")

    if do_routers:
        for mt, fname in ROUTER_SPECS:
            _run(
                [*train_router_base, "--model-type", mt, "--model-out", str(models_dir / fname)],
                dry_run=args.dry_run,
            )
    if do_recon:
        for mt, fname in RECON_SPECS:
            _run(
                [
                    *train_recon_base,
                    "--model-type",
                    mt,
                    "--export-model",
                    str(models_dir / fname),
                ],
                dry_run=args.dry_run,
            )

    if do_probe:
        machine_models.mkdir(parents=True, exist_ok=True)
        probe_ds_path, _ = _probe_paths(
            models_dir=machine_models,
            benchmark=args.benchmark,
            pts_suite=pts_suite,
            probe_backend=probe_backends[0],
        )
        if not probe_ds_path.is_file() and not args.dry_run:
            collect_cmd = [
                py,
                str(PROBE_COLLECT),
                "--dataset-root",
                str(ds_root),
                "--machine",
                machine,
                "--benchmark",
                args.benchmark,
                "--probe-duration-s",
                str(args.probe_duration_s),
                "--probe-mode",
                args.probe_mode,
                "-o",
                str(probe_ds_path),
            ]
            if args.benchmark == "phoronix":
                collect_cmd.extend(["--pts-suite", pts_suite or ""])
            if args.auto_install:
                collect_cmd.append("--auto-install")
            _run(collect_cmd, dry_run=args.dry_run)
        if not probe_ds_path.is_file() and not args.dry_run:
            print(f"Missing probe dataset: {probe_ds_path}", file=sys.stderr)
            return 2
        for probe_backend in probe_backends:
            _, probe_model_path = _probe_paths(
                models_dir=machine_models,
                benchmark=args.benchmark,
                pts_suite=pts_suite,
                probe_backend=probe_backend,
            )
            if probe_model_path.is_file() or args.dry_run:
                continue
            train_cmd = [
                py,
                str(PROBE_TRAIN),
                "--probe-dataset",
                str(probe_ds_path),
                "--model-type",
                probe_backend,
                "--model-out",
                str(probe_model_path),
            ]
            if args.auto_install:
                train_cmd.append("--auto-install")
            _run(train_cmd, dry_run=args.dry_run)

    if do_grid and not args.dry_run and router_recon_reused is False and args.stage == "grid":
        print(
            "No reusable router/recon checkpoints found. Run Route-A grid first, or pass "
            "--models-dir <router_recon_grid.../trained_models>, or use --train.",
            file=sys.stderr,
        )
        return 2

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if do_grid:
        probe_ds_path, _ = _probe_paths(
            models_dir=machine_models,
            benchmark=args.benchmark,
            pts_suite=pts_suite,
            probe_backend=probe_backends[0],
        )
        if not probe_ds_path.is_file() and not args.dry_run:
            print(f"Probe dataset required for offline hybrid grid: {probe_ds_path}", file=sys.stderr)
            return 2

        for probe_backend in probe_backends:
            probe_suffix = _PROBE_MODEL_SUFFIX[probe_backend]
            _, probe_model_path = _probe_paths(
                models_dir=machine_models,
                benchmark=args.benchmark,
                pts_suite=pts_suite,
                probe_backend=probe_backend,
            )
            if not probe_model_path.is_file() and not args.dry_run:
                print(f"Probe model required: {probe_model_path}", file=sys.stderr)
                return 2

            for r_mt, r_fn in ROUTER_SPECS:
                for c_mt, c_fn in RECON_SPECS:
                    tag = f"exp_{r_mt}__{c_mt}__probe_{probe_suffix}"
                    exp_out = out_dir / f"{tag}.json"
                    router_ckpt = models_dir / r_fn
                    recon_ckpt = models_dir / c_fn
                    if not args.dry_run:
                        if not router_ckpt.is_file() or not recon_ckpt.is_file():
                            print(f"Missing checkpoints for {tag}", file=sys.stderr)
                            return 2
                    if (
                        not args.dry_run
                        and not args.force_rerun
                        and _experiment_json_ok(exp_out)
                    ):
                        rows.append(
                            {
                                "router_type": r_mt,
                                "reconstruct_type": c_mt,
                                "probe_type": probe_backend,
                                "skipped": True,
                                **_summarize_hybrid(exp_out),
                            }
                        )
                        continue
                    cmd = [
                        py,
                        str(HYBRID_EXP),
                        "--offline",
                        "--router-model",
                        str(router_ckpt),
                        "--reconstruct-model",
                        str(recon_ckpt),
                        "--probe-model",
                        str(probe_model_path),
                        "--probe-dataset",
                        str(probe_ds_path),
                        "--dataset-root",
                        str(ds_root),
                        "--machine",
                        machine,
                        "--benchmark",
                        args.benchmark,
                        "--glob-pattern",
                        glob_eff,
                        "--top-k",
                        str(args.top_k),
                        "--probe-duration-s",
                        str(args.probe_duration_s),
                        "--xi-overhead-s",
                        str(args.xi_overhead_s),
                        "-o",
                        str(exp_out),
                    ]
                    if args.benchmark == "phoronix":
                        cmd.extend(["--pts-suite", pts_suite or ""])
                    rc = _run(cmd, dry_run=args.dry_run, check=False)
                    if rc != 0:
                        failures.append(f"{tag} (exit {rc})")
                        continue
                    if not args.dry_run and _experiment_json_ok(exp_out):
                        row = {
                            "router_type": r_mt,
                            "reconstruct_type": c_mt,
                            "probe_type": probe_backend,
                            **_summarize_hybrid(exp_out),
                        }
                        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda r: (
            float(r.get("mean_suite_rel_err") or 1e9),
        ),
    )
    report = {
        "schema": "moebench.experiment.hybrid_grid_summary.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "router_topk + probe_subtests + reconstructor",
        "grid_shape": f"{len(ROUTER_SPECS)} routers × {len(probe_backends)} probe × {len(RECON_SPECS)} recon = {len(ROUTER_SPECS) * len(probe_backends) * len(RECON_SPECS)} combinations",
        "dataset_root": str(ds_root),
        "machine": machine,
        "local_host_slug": local_host_slug(),
        "benchmark": args.benchmark,
        "pts_suite": pts_suite,
        "glob_pattern": glob_eff,
        "stage": args.stage,
        "probe_backend": args.probe_backend,
        "probe_backends": probe_backends,
        "probe_dataset": str(probe_ds_path) if do_grid else "",
        "models_dir": str(models_dir),
        "models_dir_source": models_dir_source,
        "router_recon_reused": router_recon_reused,
        "probe_models_dir": str(machine_models),
        "comparison_dir": str(out_dir),
        "routers": [m for m, _ in ROUTER_SPECS],
        "probe_regressors": probe_backends,
        "reconstructors": [m for m, _ in RECON_SPECS],
        "ranking_note": "Sorted by mean_suite_rel_err (lower is better).",
        "grid_failures": failures,
        "per_combination": rows,
        "ranked": ranked,
    }
    if do_grid and not args.dry_run:
        summary_path = out_dir / "grid_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Wrote summary: {summary_path}", file=sys.stderr)
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
