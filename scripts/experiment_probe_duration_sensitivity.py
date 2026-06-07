#!/usr/bin/env python3
"""Probe duration sensitivity: train and evaluate Route~B at multiple probe budgets (e.g. 2/4/8 s).

For each duration ``D``:
  1) collect ``probe_dataset_*_{D}s.json`` from historical full runs (unless skipped),
  2) train ``probe_unixbench_{D}s_lgbm.pkl`` (unless skipped),
  3) online micro-probe all components at ``D`` seconds and predict suite vs ground truth.

Outputs one comparison JSON for paper tables / Pareto plots.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import (
    ensure_machine_output_dir,
    machine_experiments_dir,
    machine_models_dir,
    resolve_glob_for_machine,
    resolve_training_machine,
)
from moebench.ml_venv import ensure_ml_interpreter
from moebench.pip_install import ensure_importable


def _early_ml_modules() -> list[str]:
    mt = "lightgbm"
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--model-type" and i + 1 < len(argv):
            return ["numpy", "sklearn", argv[i + 1]]
        if a.startswith("--model-type="):
            return ["numpy", "sklearn", a.split("=", 1)[1]]
    return ["numpy", "sklearn", mt]


ensure_ml_interpreter(
    need_modules=_early_ml_modules(),
    auto_install="--auto-install" in sys.argv,
    label="probe_duration_sensitivity",
)

from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.probe.model_train import train_probe_bundle
from moebench.probe.suite_aggregate import aggregate_suite_index, suite_error_report
from moebench.probe.training_data import (
    collect_probe_dataset,
    label_suite_from_pts_run,
    label_suite_from_unixbench_run,
)
from moebench.phoronix.training_data import canonical_test_ids_from_runs, collect_phoronix_run_paths
from moebench.reconstruct.data import collect_unixbench_run_paths

SCHEMA = "moebench.experiment.probe_duration_sensitivity.v1"


def _parse_durations(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        d = float(part)
        if d < 1.0 or d > 30.0:
            raise ValueError(f"probe duration {d} out of supported range [1, 30]")
        out.append(d)
    if not out:
        raise ValueError("no durations parsed")
    return out


def _dur_tag(d: float) -> str:
    if abs(d - round(d)) < 1e-9:
        return str(int(round(d)))
    return str(d).replace(".", "p")


def _paths(
    models_dir: Path,
    benchmark: str,
    pts_suite: str | None,
    duration_s: float,
    model_type: str,
) -> tuple[Path, Path]:
    tag = _dur_tag(duration_s)
    if benchmark == "unixbench":
        ds = models_dir / f"probe_dataset_unixbench_{tag}s.json"
        model = models_dir / f"probe_unixbench_{tag}s_{model_type}.pkl"
    else:
        tok = (pts_suite or "pts").replace("/", "_")
        ds = models_dir / f"probe_dataset_{tok}_{tag}s.json"
        model = models_dir / f"probe_{tok}_{tag}s_{model_type}.pkl"
    return ds, model


def _ground_truth_unixbench(dataset_root: Path, machine: str) -> tuple[float | None, str | None]:
    glo = resolve_glob_for_machine(benchmark="unixbench", machine=machine, glob_pattern=None, pts_suite=None)
    paths = collect_unixbench_run_paths(dataset_root, glob_pattern=glo)
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        ds = json.load(f)
    return label_suite_from_unixbench_run(ds), str(latest)


def _ground_truth_pts(dataset_root: Path, machine: str, pts_suite: str) -> tuple[float | None, str | None]:
    glo = resolve_glob_for_machine(
        benchmark="phoronix",
        machine=machine,
        glob_pattern=None,
        pts_suite=pts_suite,
    )
    paths = collect_phoronix_run_paths(dataset_root, glob_pattern=glo, pts_suite=pts_suite)
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        ds = json.load(f)
    tids = list(canonical_test_ids_from_runs([latest]))
    return label_suite_from_pts_run(ds, tids), str(latest)


def _collect_dataset(
    *,
    benchmark: str,
    dataset_root: str,
    machine: str,
    pts_suite: str | None,
    duration_s: float,
    probe_mode: str,
    enable_ebpf: bool,
    out_path: Path,
) -> Path:
    ds = collect_probe_dataset(
        benchmark=benchmark,
        dataset_root=dataset_root,
        machine=machine,
        pts_suite=pts_suite,
        probe_duration_s=duration_s,
        enable_ebpf=enable_ebpf,
        probe_mode=probe_mode,
        live_probe=True,
    )
    ensure_machine_output_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote probe dataset {out_path} ({ds.get('num_samples')} samples)", file=sys.stderr)
    return out_path


def _train_model(
    *,
    dataset_path: Path,
    model_path: Path,
    model_type: str,
    auto_install: bool,
) -> Path:
    ensure_importable(model_type, auto_install=auto_install)
    with open(dataset_path, encoding="utf-8") as f:
        ds = json.load(f)
    bundle = train_probe_bundle(ds, model_type=model_type)
    ensure_machine_output_dir(model_path)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote probe model {model_path}", file=sys.stderr)
    return model_path


def _evaluate_online(
    *,
    bundle: dict[str, Any],
    model_path: Path,
    duration_s: float,
    probe_mode: str,
    benchmark: str,
    enable_ebpf: bool,
    gt_suite: float | None,
) -> dict[str, Any]:
    tids = list(bundle.get("test_ids") or [])
    agg = str(bundle.get("suite_aggregate", "geomean_index"))
    sub_preds: dict[str, float] = {}
    probe_walls: list[float] = []
    errors: list[str] = []

    for tid in tids:
        try:
            probe = collect_subtest_probe(
                tid,
                duration_s=duration_s,
                enable_ebpf=enable_ebpf,
                benchmark=benchmark,
                probe_mode=probe_mode,
            )
            sub_preds[tid] = predict_subtest(bundle, probe, tid)
            probe_walls.append(float(probe.get("wall_s") or duration_s))
        except Exception as e:
            errors.append(f"{tid}: {e!s}")

    suite_pred = aggregate_suite_index(sub_preds, mode=agg) if sub_preds else None
    wall_sum = float(sum(probe_walls))

    row: dict[str, Any] = {
        "probe_duration_s": duration_s,
        "probe_model": str(model_path.resolve()),
        "n_subtests_requested": len(tids),
        "n_subtests_predicted": len(sub_preds),
        "probe_wall_s_sum": wall_sum,
        "predicted_subtest": sub_preds,
        "predicted_suite": suite_pred,
        "errors": errors,
        "pipeline_ok": len(errors) == 0 and len(sub_preds) == len(tids),
    }
    if gt_suite is not None and suite_pred is not None:
        row["suite_comparison"] = suite_error_report(float(suite_pred), float(gt_suite))
    return row


def _rel_err_pct(row: dict[str, Any]) -> float | None:
    sc = row.get("suite_comparison") or {}
    v = sc.get("relative_error")
    return float(v) * 100.0 if v is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--machine", type=str, default="")
    ap.add_argument("--benchmark", type=str, choices=("unixbench", "phoronix"), default="unixbench")
    ap.add_argument("--pts-suite", type=str, default="")
    ap.add_argument("--durations", type=str, default="2,4,8", help="Comma-separated probe budgets in seconds")
    ap.add_argument("--probe-mode", type=str, choices=("micro", "real"), default="micro")
    ap.add_argument("--model-type", type=str, choices=("lightgbm", "xgboost"), default="lightgbm")
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--skip-collect", action="store_true", help="Reuse existing probe_dataset_*_{D}s.json")
    ap.add_argument("--skip-train", action="store_true", help="Reuse existing probe_*_{D}s_*.pkl")
    ap.add_argument("--eval-only", action="store_true", help="Skip collect and train (implies both skips)")
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()

    if args.eval_only:
        args.skip_collect = True
        args.skip_train = True

    machine = resolve_training_machine(args.machine or None)
    ds_root = Path(args.dataset_root).resolve()
    models_dir = machine_models_dir(str(ds_root), machine)
    models_dir.mkdir(parents=True, exist_ok=True)
    pts_suite = args.pts_suite.strip() or None
    if args.benchmark == "phoronix" and not pts_suite:
        print("--benchmark phoronix requires --pts-suite", file=sys.stderr)
        return 2

    durations = _parse_durations(args.durations)
    enable_ebpf = not args.no_ebpf

    if args.benchmark == "unixbench":
        gt_suite, gt_path = _ground_truth_unixbench(ds_root, machine)
    else:
        gt_suite, gt_path = _ground_truth_pts(ds_root, machine, str(pts_suite))
    if gt_suite is None:
        print("No ground-truth full run found", file=sys.stderr)
        return 2

    per_duration: list[dict[str, Any]] = []

    for duration_s in durations:
        ds_path, model_path = _paths(models_dir, args.benchmark, pts_suite, duration_s, args.model_type)
        print(f"=== Duration {duration_s}s ===", file=sys.stderr)

        if not args.skip_collect:
            if ds_path.is_file():
                print(f"Reusing existing dataset {ds_path}", file=sys.stderr)
            else:
                _collect_dataset(
                    benchmark=args.benchmark,
                    dataset_root=str(ds_root),
                    machine=machine,
                    pts_suite=pts_suite,
                    duration_s=duration_s,
                    probe_mode=args.probe_mode,
                    enable_ebpf=enable_ebpf,
                    out_path=ds_path,
                )
        elif not ds_path.is_file():
            print(f"Missing dataset {ds_path} (drop --skip-collect or --eval-only)", file=sys.stderr)
            return 2

        if not args.skip_train:
            _train_model(
                dataset_path=ds_path,
                model_path=model_path,
                model_type=args.model_type,
                auto_install=args.auto_install,
            )
        elif not model_path.is_file():
            print(f"Missing model {model_path} (drop --skip-train or --eval-only)", file=sys.stderr)
            return 2

        bundle = load_probe_bundle(model_path)
        row = _evaluate_online(
            bundle=bundle,
            model_path=model_path,
            duration_s=duration_s,
            probe_mode=args.probe_mode,
            benchmark=args.benchmark,
            enable_ebpf=enable_ebpf,
            gt_suite=gt_suite,
        )
        row["probe_dataset"] = str(ds_path.resolve())
        per_duration.append(row)

    comparison_rows = []
    for row in per_duration:
        comparison_rows.append(
            {
                "probe_duration_s": row["probe_duration_s"],
                "suite_relative_error_pct": _rel_err_pct(row),
                "probe_wall_s_sum": row["probe_wall_s_sum"],
                "predicted_suite": row.get("predicted_suite"),
                "pipeline_ok": row.get("pipeline_ok"),
            }
        )

    best_err = min(
        (r for r in comparison_rows if r.get("suite_relative_error_pct") is not None),
        key=lambda r: float(r["suite_relative_error_pct"]),
        default=None,
    )
    best_time = min(comparison_rows, key=lambda r: float(r["probe_wall_s_sum"]))

    if args.output.strip():
        out_path = Path(args.output).resolve()
    else:
        tok = "unixbench" if args.benchmark == "unixbench" else (pts_suite or "pts").replace("/", "_")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = machine_experiments_dir(str(ds_root), machine) / f"probe_duration_sensitivity_{tok}_{stamp}.json"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine,
        "benchmark": args.benchmark,
        "pts_suite": pts_suite,
        "probe_mode": args.probe_mode,
        "model_type": args.model_type,
        "enable_ebpf": enable_ebpf,
        "durations_s": durations,
        "ground_truth_run": gt_path,
        "ground_truth_suite": gt_suite,
        "per_duration": per_duration,
        "comparison": {
            "rows": comparison_rows,
            "lowest_relative_error_duration_s": best_err["probe_duration_s"] if best_err else None,
            "lowest_wall_time_duration_s": best_time["probe_duration_s"],
        },
    }

    ensure_machine_output_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({"output": str(out_path), "comparison": report["comparison"]}, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
