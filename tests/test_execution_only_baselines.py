from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_execution_only_baselines.py"
spec = importlib.util.spec_from_file_location("execution_only_baselines", MODULE_PATH)
assert spec and spec.loader
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


def _rows() -> list[dict]:
    values = np.asarray(
        [
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
            1100.0,
            1200.0,
        ]
    )
    return [
        {
            "path": "run-01.json",
            "session": "session",
            "values": values,
            "suite": baselines._geometric_mean(values),
            "times": {tid: 1.0 for tid in baselines.INDEX_SUITE_TEST_IDS},
            "full_time_s": 12.0,
        }
    ]


def test_execution_only_estimator_uses_only_selected_values():
    rows = _rows()
    ids = baselines._fixed_first_ids(3)
    result = baselines._evaluate_selection(
        "fixed_partial_geomean",
        rows,
        [ids],
        selection_rule="fixed first K subtests",
    )
    run = result["per_run"][0]
    expected = baselines._geometric_mean(rows[0]["values"][:3])
    assert run["predicted_full_suite"] == expected
    assert result["training_data_used"] is False
    assert run["training_run_paths"] == []


def test_random_selection_is_deterministic_and_has_no_duplicates():
    first = baselines._random_ids(42, 3)
    second = baselines._random_ids(42, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3


def test_execution_only_result_reports_actual_partial_values_and_time():
    rows = _rows()
    ids = list(baselines.CATEGORY_BALANCED_IDS)
    result = baselines._evaluate_selection(
        "category_balanced_partial_geomean",
        rows,
        [ids],
        selection_rule="fixed CPU/IO/syscall subtests",
    )
    run = result["per_run"][0]
    assert set(run["partial_observed_values"]) == set(ids)
    assert run["partial_wall_time_s"] == 3.0
    assert run["leakage_flags"] == []
