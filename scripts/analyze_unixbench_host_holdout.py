#!/usr/bin/env python3
"""Host-held-out diagnostic for sparse UnixBench profile reconstruction.

This uses full-run component scores as oracle observations. It is an
information-value diagnostic, not a timed BenchScout pipeline evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from moebench.reconstruct.data import extract_targets_from_dataset
from moebench.router.feature_vectorizer import XiVectorizer
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS


HOSTS = (
    "aces-System-Product-Name",
    "iZbp15n87643uk1sqjrdvdZ",
    "iZbp16krl0yc7euw7sb6slZ",
    "iZbp1acaw5wdllhz47922rZ",
    "iZbp1glgt48i9a8d49embxZ",
)
K = 3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path) -> list[dict]:
    rows = []
    vectorizer = XiVectorizer()
    for host in HOSTS:
        sessions = sorted(root.glob(f"{host}_20*/run-*.json"))
        if len(sessions) != 5 or len({p.parent for p in sessions}) != 1:
            raise ValueError(f"Expected one five-run UnixBench session for {host}; found {sessions}")
        for path in sessions:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != "moebench.unixbench.dataset.v1":
                raise ValueError(f"Unexpected record schema: {path}")
            target = extract_targets_from_dataset(data)
            if target is None:
                raise ValueError(f"Missing complete UnixBench targets: {path}")
            component, suite = target
            if any(value <= 0 for value in component) or suite <= 0:
                raise ValueError(f"Nonpositive score: {path}")
            rows.append({
                "host": host,
                "path": str(path.resolve()),
                "sha256": digest(path),
                "components": component,
                "suite": suite,
                "context": vectorizer.transform(data.get("xi") or {}),
            })
    return rows


def fit_predict(train: np.ndarray, test: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    """Predict log suite score from observed log component indices."""
    x_train = train[:, subset]
    x_test = test[:, subset]
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    model = Ridge(alpha=1.0)
    model.fit((x_train - center) / scale, train[:, -1])
    return model.predict((x_test - center) / scale)


def mean_error(train: np.ndarray, test: np.ndarray, subset: tuple[int, ...]) -> float:
    prediction = np.exp(fit_predict(train, test, subset))
    actual = np.exp(test[:, -1])
    return float(np.mean(np.abs(prediction - actual) / actual) * 100.0)


def context_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    model = Ridge(alpha=alpha)
    model.fit((x_train - center) / scale, y_train)
    return model.predict((x_test - center) / scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.dataset_root.resolve())
    all_values = np.log(np.asarray([row["components"] + [row["suite"]] for row in rows]))
    all_context = np.asarray([row["context"] for row in rows], dtype=float)
    groups = np.asarray([row["host"] for row in rows])
    subsets = list(itertools.combinations(range(len(INDEX_SUITE_TEST_IDS)), K))
    results = []
    for heldout in HOSTS:
        train_indices = np.flatnonzero(groups != heldout)
        test_indices = np.flatnonzero(groups == heldout)
        train_hosts = [host for host in HOSTS if host != heldout]
        train = all_values[train_indices]
        test = all_values[test_indices]
        inner_errors = []
        for subset in subsets:
            host_errors = []
            for validation_host in train_hosts:
                inner_train = all_values[(groups != heldout) & (groups != validation_host)]
                inner_test = all_values[groups == validation_host]
                host_errors.append(mean_error(inner_train, inner_test, subset))
            inner_errors.append(float(np.mean(host_errors)))
        nested = subsets[int(np.argmin(inner_errors))]
        score_ranked = tuple(sorted(np.argsort(train[:, :-1].mean(axis=0))[-K:].tolist()))
        random_generator = np.random.default_rng(42)
        random_subsets = [tuple(sorted(random_generator.choice(12, K, replace=False).tolist())) for _ in range(100)]
        policies = {
            "nested_target_loss": nested,
            "score_ranked": score_ranked,
            "fixed_first_three": (0, 1, 2),
        }
        per_policy = {}
        for name, subset in policies.items():
            predicted = np.exp(fit_predict(train, test, subset))
            truth = np.exp(test[:, -1])
            per_policy[name] = {
                "subset": [INDEX_SUITE_TEST_IDS[i] for i in subset],
                "inner_host_mean_error_percent": inner_errors[subsets.index(subset)],
                "mean_error_percent": float(np.mean(np.abs(predicted - truth) / truth) * 100),
                "per_run_error_percent": (np.abs(predicted - truth) / truth * 100).tolist(),
            }
        random_errors = [mean_error(train, test, subset) for subset in random_subsets]
        per_policy["random_three"] = {
            "seed": 42,
            "draws": 100,
            "mean_error_percent": float(np.mean(random_errors)),
            "median_error_percent": float(np.median(random_errors)),
            "p05_p95_error_percent": np.percentile(random_errors, [5, 95]).tolist(),
        }
        # A constant predictor is the honest no-target-measurement control.
        no_probe_prediction = math.exp(float(train[:, -1].mean()))
        per_policy["training_mean_no_probe"] = {
            "mean_error_percent": float(np.mean(np.abs(no_probe_prediction - np.exp(test[:, -1])) / np.exp(test[:, -1])) * 100),
        }
        alphas = (1.0, 10.0, 100.0, 1000.0)
        alpha_losses = []
        for alpha in alphas:
            host_errors = []
            for validation_host in train_hosts:
                inner_fit = np.flatnonzero((groups != heldout) & (groups != validation_host))
                inner_test = np.flatnonzero(groups == validation_host)
                predicted = np.exp(context_predict(
                    all_context[inner_fit], all_values[inner_fit, -1],
                    all_context[inner_test], alpha,
                ))
                actual = np.exp(all_values[inner_test, -1])
                host_errors.append(float(np.mean(np.abs(predicted - actual) / actual) * 100))
            alpha_losses.append(float(np.mean(host_errors)))
        context_alpha = alphas[int(np.argmin(alpha_losses))]
        context_predicted = np.exp(context_predict(
            all_context[train_indices], train[:, -1], all_context[test_indices], context_alpha,
        ))
        context_actual = np.exp(test[:, -1])
        per_policy["context_only_ridge"] = {
            "selected_alpha": context_alpha,
            "training_host_cv_error_by_alpha_percent": dict(zip(map(str, alphas), alpha_losses)),
            "mean_error_percent": float(np.mean(np.abs(context_predicted - context_actual) / context_actual) * 100),
            "per_run_error_percent": (np.abs(context_predicted - context_actual) / context_actual * 100).tolist(),
        }
        results.append({
            "heldout_host": heldout,
            "training_hosts": train_hosts,
            "training_runs": [rows[i]["path"] for i in train_indices],
            "test_runs": [rows[i]["path"] for i in test_indices],
            "policies": per_policy,
        })
    report = {
        "schema": "moebench.unixbench.host_holdout_oracle_components.v1",
        "protocol": "Five outer host-held-out folds; nested selection evaluates each of 220 three-component subsets by leave-one-training-host-out loss. True full-run component scores are observed on the held-out host; this is not a timed or probe-predicted pipeline evaluation.",
        "component_ids": list(INDEX_SUITE_TEST_IDS),
        "input_artifacts": [{key: row[key] for key in ("host", "path", "sha256")} for row in rows],
        "folds": results,
        "across_host_mean_error_percent": {
            name: float(np.mean([fold["policies"][name]["mean_error_percent"] for fold in results]))
            for name in results[0]["policies"]
        },
        "limitations": [
            "Five host identities; one five-run session per host.",
            "Oracle full-run component scores are supplied at test time. This does not establish 15-second probe fidelity or wall-time savings.",
            "Selected subset optimizes only suite score error, not component-profile error, measurement cost, or Web-serving decisions.",
            "Context-only Ridge uses the raw Xi feature vector, standardized on source hosts, with regularization selected by inner whole-host cross-validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["across_host_mean_error_percent"], indent=2))


if __name__ == "__main__":
    main()
