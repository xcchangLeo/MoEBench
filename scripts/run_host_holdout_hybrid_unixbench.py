#!/usr/bin/env python3
"""Offline whole-host holdout of the UnixBench router/probe/reconstructor path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOSTS = (
    "aces-System-Product-Name",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp16krl0yc7euw7sb6slZ",
    "iZbp1acaw5wdllhz47922rZ",
    "iZbp1glgt48i9a8d49embxZ",
)


def load_cv_module() -> Any:
    path = ROOT / "scripts" / "paper_reconstruct_cv_extras.py"
    spec = importlib.util.spec_from_file_location("paper_reconstruct_cv_extras", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy_module() -> Any:
    path = ROOT / "scripts" / "run_hybrid_oof_unixbench_policies.py"
    spec = importlib.util.spec_from_file_location("run_hybrid_oof_unixbench_policies", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(path: str | Path) -> tuple[str, str]:
    source = Path(path)
    return source.parent.name, source.name


def host_of_session(session: str) -> str:
    for host in HOSTS:
        if session.startswith(host + "_202605"):
            return host
    raise ValueError(f"Unexpected UnixBench session: {session}")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([row["relative_error_percent"] for row in records], dtype=float)
    component_errors = [row.get("selected_probe_mape_percent") for row in records]
    component_errors = [value for value in component_errors if value is not None]
    probe_wall = np.asarray([row["selected_probe_wall_seconds_from_archive"] for row in records], dtype=float)
    full_wall = np.asarray([row["full_suite_wall_seconds_from_run"] for row in records], dtype=float)
    summary = {
        "n_runs": len(records),
        "mean_error_percent": float(errors.mean()),
        "median_error_percent": float(np.median(errors)),
        "max_error_percent": float(errors.max()),
        "mean_selected_probe_mape_percent": float(np.mean(component_errors)) if component_errors else None,
        "mean_selected_probe_wall_seconds": float(probe_wall.mean()),
        "mean_full_suite_wall_seconds": float(full_wall.mean()),
        "mean_probe_only_fraction_of_full_wall": float(np.mean(probe_wall / full_wall)),
    }
    return summary


def probe_calibration_factors(
    train_rows: list[dict[str, Any]],
    crossfit: dict[tuple[str, str], dict[str, float]],
    test_ids: list[str],
) -> dict[str, float]:
    """Fit per-component multiplicative calibration using training rows only."""
    log_ratios: dict[str, list[float]] = {tid: [] for tid in test_ids}
    for row in train_rows:
        predictions = crossfit[key(row["path"])]
        for index, tid in enumerate(test_ids):
            truth = float(row["y"][index])
            estimate = float(predictions[tid])
            if truth > 0.0 and estimate > 0.0:
                log_ratios[tid].append(float(np.log(truth / estimate)))
    return {
        tid: float(np.exp(np.median(values))) if values else 1.0
        for tid, values in log_ratios.items()
    }


def correlation_top_k(
    train_rows: list[dict[str, Any]], test_ids: list[str], k: int
) -> tuple[list[str], dict[str, float | None]]:
    """Choose experts by absolute training-fold Spearman association with suite score."""
    from scipy.stats import spearmanr

    targets = np.asarray([float(row["y"][-1]) for row in train_rows], dtype=np.float64)
    ranked: list[tuple[float, int, str]] = []
    correlations: dict[str, float | None] = {}
    for index, tid in enumerate(test_ids):
        values = np.asarray([float(row["y"][index]) for row in train_rows], dtype=np.float64)
        rho = float(spearmanr(values, targets).statistic)
        correlations[tid] = rho if np.isfinite(rho) else None
        ranked.append((-(abs(rho) if np.isfinite(rho) else 0.0), index, tid))
    ranked.sort()
    return [tid for _, _, tid in ranked[:k]], correlations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suite-output-mode",
        choices=("direct", "probe_plus_imputed_geomean"),
        default="probe_plus_imputed_geomean",
        help="How the suite estimate consumes selected probe predictions",
    )
    args = parser.parse_args()

    import torch
    from moebench.hybrid.eval import router_select_test_ids, probe_predictions_to_executed_tests
    from moebench.probe.inference import predict_subtest
    from moebench.probe.model_train import train_probe_bundle
    from moebench.reconstruct.data import full_suite_wall_seconds
    from moebench.reconstruct.inference import predict_from_partial
    from moebench.router.feature_vectorizer import XiVectorizer
    from scripts.run_corrected_oof_unixbench import _build_crossfitted_reconstructor_training_matrix

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv = load_cv_module()
    policy_helpers = load_policy_module()
    root = args.dataset_root.resolve()
    all_rows, test_ids = cv._load_unixbench_rows(root, "*/run-*.json")
    rows = [row for row in all_rows if any(key(row["path"])[0].startswith(host + "_202605") for host in HOSTS)]
    if len(rows) != 25:
        raise ValueError(f"Expected 25 canonical May UnixBench runs, found {len(rows)}")

    probe_paths = [root / "models" / host / "probe_dataset_unixbench.json" for host in HOSTS]
    datasets = [json.loads(path.read_text(encoding="utf-8")) for path in probe_paths]
    probe_dataset = dict(datasets[0])
    probe_dataset["samples"] = [sample for dataset in datasets for sample in dataset["samples"]]
    if len(probe_dataset["samples"]) != 25 * len(test_ids):
        raise ValueError("Expected one probe sample per run and UnixBench component")
    samples = {key(sample["source_run"]) + (str(sample["test_id"]),): sample for sample in probe_dataset["samples"]}
    if len(samples) != len(probe_dataset["samples"]):
        raise ValueError("Duplicate probe sample key")
    xi_vec = XiVectorizer()
    folds = []
    for fold_index, heldout in enumerate(HOSTS):
        train = [row for row in rows if host_of_session(key(row["path"])[0]) != heldout]
        test = [row for row in rows if host_of_session(key(row["path"])[0]) == heldout]
        source_hosts = [host for host in HOSTS if host != heldout]
        train_keys = {key(row["path"]) for row in train}
        outer_dataset = dict(probe_dataset)
        outer_dataset["samples"] = [sample for sample in probe_dataset["samples"] if key(sample["source_run"]) in train_keys]
        outer_probe = train_probe_bundle(outer_dataset, model_type="lightgbm")

        # Inner whole-host probe predictions give the reconstructor realistic
        # training inputs without exposing a source host to its own probe fit.
        crossfit: dict[tuple[str, str], dict[str, float]] = {}
        inner_manifest = []
        for inner_host in source_hosts:
            inner_fit_hosts = set(source_hosts) - {inner_host}
            inner_dataset = dict(probe_dataset)
            inner_dataset["samples"] = [
                sample for sample in probe_dataset["samples"]
                if host_of_session(key(sample["source_run"])[0]) in inner_fit_hosts
            ]
            inner_probe = train_probe_bundle(inner_dataset, model_type="lightgbm")
            inner_rows = [row for row in train if host_of_session(key(row["path"])[0]) == inner_host]
            for row in inner_rows:
                run_key = key(row["path"])
                crossfit[run_key] = {
                    tid: predict_subtest(inner_probe, samples[run_key + (tid,)]["probe"], tid)
                    for tid in test_ids
                }
            inner_manifest.append({"heldout_source_host": inner_host, "probe_fit_hosts": sorted(inner_fit_hosts)})

        x_train, y_train = _build_crossfitted_reconstructor_training_matrix(
            train, crossfit, samples, xi_vec, test_ids,
            seed=args.seed + fold_index * 997, train_aug=20, train_k_min=2, train_k_max=6,
        )
        oracle_scores = {
            key(row["path"]): {tid: float(row["y"][i]) for i, tid in enumerate(test_ids)}
            for row in train
        }
        x_oracle, y_oracle = _build_crossfitted_reconstructor_training_matrix(
            train, oracle_scores, samples, xi_vec, test_ids,
            seed=args.seed + fold_index * 997, train_aug=20, train_k_min=2, train_k_max=6,
        )
        recon = cv.rte.sklearn_export_bundle(
            "lightgbm", x_train, y_train, auto_install=False,
            lgbm_estimators=200, xgb_estimators=200,
            log1p_partial_index=False, test_ids=test_ids,
            with_uncertainty=False, benchmark="unixbench",
            suite_output_mode=args.suite_output_mode,
        )
        calibration_factors = probe_calibration_factors(train, crossfit, test_ids)
        recon["probe_prediction_calibration_factors"] = calibration_factors
        oracle_recon = cv.rte.sklearn_export_bundle(
            "lightgbm", x_oracle, y_oracle, auto_install=False,
            lgbm_estimators=200, xgb_estimators=200,
            log1p_partial_index=False, test_ids=test_ids,
            with_uncertainty=False, benchmark="unixbench",
            suite_output_mode="observed_plus_imputed_geomean",
        )
        router = cv._train_fold_router(train, test_ids, xi_vec, seed=args.seed + fold_index)
        probe_reliability, probe_error_by_test = policy_helpers._probe_reliability_weights(
            train, crossfit, test_ids
        )
        anchor_router = policy_helpers._train_fold_anchor_router(
            train, test_ids, xi_vec, seed=args.seed + fold_index, probe_reliability=probe_reliability
        )
        corr_router = policy_helpers._train_fold_corr_router(
            train, test_ids, xi_vec, seed=args.seed + fold_index, probe_reliability=probe_reliability
        )
        source_scores = np.asarray([row["y"][:-1] for row in train])
        score_order = np.argsort(source_scores.mean(axis=0))[-3:]
        score_ranked = [test_ids[i] for i in sorted(score_order)]
        corr_ids, corr_values = correlation_top_k(train, test_ids, 3)
        rng = np.random.default_rng(args.seed + fold_index)
        random_ids = [test_ids[i] for i in sorted(rng.choice(len(test_ids), 3, replace=False))]
        results: dict[str, list[dict[str, Any]]] = {
            name: [] for name in (
                "learned_router", "learned_anchor_router", "learned_corr_reliability_router",
                "score_ranked_fixed", "correlation_top3_fixed",
                "fixed_first_three", "random_fixed",
            )
        }
        oracle_results: dict[str, list[dict[str, Any]]] = {name: [] for name in results}
        for row in test:
            run_key = key(row["path"])
            xi = row["ds"].get("xi") or {}
            full_suite_wall = float(full_suite_wall_seconds(row["ds"]))
            _, learned_ids, detail = router_select_test_ids(router, xi, top_k=3)
            _, anchor_ids, anchor_detail = router_select_test_ids(anchor_router, xi, top_k=3)
            _, corr_ids_learned, corr_detail = router_select_test_ids(corr_router, xi, top_k=3)
            policies = {
                "learned_router": learned_ids,
                "learned_anchor_router": anchor_ids,
                "learned_corr_reliability_router": corr_ids_learned,
                "score_ranked_fixed": score_ranked,
                "correlation_top3_fixed": corr_ids,
                "fixed_first_three": test_ids[:3],
                "random_fixed": random_ids,
            }
            for name, selected in policies.items():
                predictions = {tid: predict_subtest(outer_probe, samples[run_key + (tid,)]["probe"], tid) for tid in selected}
                walls = {tid: float(samples[run_key + (tid,)]["probe"].get("wall_s") or 4.0) for tid in selected}
                executed = probe_predictions_to_executed_tests(
                    selected, predictions, 4.0, benchmark="unixbench", probe_wall_by_tid=walls
                )
                result = predict_from_partial(recon, xi, executed)
                truth = float(row["y"][-1])
                prediction = float(result["suite_index"])
                actual_components = {tid: float(row["y"][test_ids.index(tid)]) for tid in selected}
                component_mape = float(np.mean([
                    abs(predictions[tid] - actual_components[tid]) / max(abs(actual_components[tid]), 1e-9) * 100.0
                    for tid in selected
                ]))
                results[name].append({
                    "run": row["path"],
                    "selected_test_ids": selected,
                    "predicted_selected_scores": predictions,
                    "predicted_suite": prediction,
                    "actual_suite": truth,
                    "relative_error_percent": abs(prediction - truth) / truth * 100,
                    "selected_probe_mape_percent": component_mape,
                    "suite_output_mode": result["suite_output_mode"],
                    "selected_probe_wall_seconds_from_archive": float(sum(walls.values())),
                    "full_suite_wall_seconds_from_run": full_suite_wall,
                    "probe_only_fraction_of_full_wall": float(sum(walls.values())) / max(full_suite_wall, 1e-9),
                    "router_scores": (
                        detail["scores"] if name == "learned_router"
                        else anchor_detail["scores"] if name == "learned_anchor_router"
                        else corr_detail["scores"] if name == "learned_corr_reliability_router"
                        else None
                    ),
                    "training_fold_component_suite_spearman": corr_values if name == "correlation_top3_fixed" else None,
                })
                oracle_values = {
                    tid: float(row["y"][test_ids.index(tid)]) for tid in selected
                }
                oracle_executed = probe_predictions_to_executed_tests(
                    selected, oracle_values, 4.0, benchmark="unixbench", probe_wall_by_tid=walls
                )
                oracle_result = predict_from_partial(oracle_recon, xi, oracle_executed)
                oracle_results[name].append({
                    "run": row["path"],
                    "selected_test_ids": selected,
                    "predicted_suite": float(oracle_result["suite_index"]),
                    "actual_suite": truth,
                    "relative_error_percent": abs(float(oracle_result["suite_index"]) - truth) / truth * 100.0,
                    "suite_output_mode": oracle_result["suite_output_mode"],
                    "selected_probe_wall_seconds_from_archive": float(sum(walls.values())),
                    "full_suite_wall_seconds_from_run": full_suite_wall,
                    "label_access": "heldout exact component labels; diagnostic oracle, not an online result",
                })
        folds.append({
            "heldout_host": heldout,
            "training_hosts": source_hosts,
            "training_runs": [row["path"] for row in train],
            "test_runs": [row["path"] for row in test],
            "inner_probe_folds": inner_manifest,
            "reconstructor_train_rows": int(len(x_train)),
            "oracle_reconstructor_train_rows": int(len(x_oracle)),
            "probe_calibration_factors": calibration_factors,
            "training_fold_probe_error_by_test": probe_error_by_test,
            "training_fold_probe_reliability": probe_reliability,
            "correlation_top3_ids": corr_ids,
            "correlation_top3_training_fold_spearman": corr_values,
            "policies": {name: {"summary": summarize(records), "per_run": records} for name, records in results.items()},
            "observed_score_oracle": {
                name: {"summary": summarize(records), "per_run": records}
                for name, records in oracle_results.items()
            },
        })
        print(f"Completed host fold {fold_index + 1}/5: {heldout}", file=sys.stderr)

    report = {
        "schema": "moebench.hybrid.whole_host_holdout_offline.v1",
        "protocol": "Outer whole-host holdout. Router/probe/reconstructor trained only on four source hosts. Reconstructor source inputs are probe predictions from inner whole-host probe fits. Test probe telemetry is replayed from archived records keyed to each source run; legacy probe timestamps do not establish that probes preceded their matching full runs. No fresh online target-host workload or paired timing is claimed.",
        "k": 3,
        "suite_output_mode": args.suite_output_mode,
        "policy_definitions": {
            "learned_router": "Pointwise MLP trained to predict raw component scores from xi and component identity.",
            "learned_anchor_router": "Pointwise MLP trained on component-to-suite log-scale closeness, weighted by training-fold probe reliability.",
            "learned_corr_reliability_router": "Pointwise MLP trained on within-run component rank weighted by training-fold absolute Spearman association with suite score and probe reliability.",
            "score_ranked_fixed": "Fixed Top-3 by mean component score in the outer training hosts.",
            "correlation_top3_fixed": "Fixed Top-3 by absolute Spearman association between each component and suite score in the outer training hosts.",
            "fixed_first_three": "The first three benchmark components in canonical test-id order.",
            "random_fixed": "One seed-fixed random Top-3 per outer host fold.",
        },
        "probe_prediction_calibration_factors_by_training_fold": [
            {"heldout_host": fold["heldout_host"], "factors": fold["probe_calibration_factors"]}
            for fold in folds
        ],
        "input_probe_datasets": [{"path": str(path), "sha256": sha256(path)} for path in probe_paths],
        "input_runs": [{"path": row["path"], "sha256": sha256(Path(row["path"]))} for row in rows],
        "folds": folds,
        "across_host_mean_error_percent": {
            name: float(np.mean([fold["policies"][name]["summary"]["mean_error_percent"] for fold in folds]))
            for name in folds[0]["policies"]
        },
        "observed_score_oracle_across_host_mean_error_percent": {
            name: float(np.mean([
                fold["observed_score_oracle"][name]["summary"]["mean_error_percent"] for fold in folds
            ]))
            for name in folds[0]["observed_score_oracle"]
        },
        "limitations": [
            "Legacy archived probe timestamps are unavailable, so pre-run ordering and paired collection are not independently verified; this is not an online target-host timing experiment.",
            "Five hosts with one five-run session each. Source-host folds are not independent sessions.",
            "The primary learned_router uses component-score supervision. Anchor and correlation-plus-probe-reliability supervision are compared, but none is trained directly on held-out reconstruction error.",
            "Reported probe-only wall fractions exclude xi collection and other orchestration overhead; archived probe wall times may not match a live pre-run workflow.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["across_host_mean_error_percent"], indent=2))


if __name__ == "__main__":
    main()
