from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_end_to_end_baselines.py"
spec = importlib.util.spec_from_file_location("end_to_end_baselines", MODULE_PATH)
assert spec and spec.loader
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


def _rows() -> list[dict]:
    rows = []
    for i in range(5):
        values = np.asarray(
            [
                100 + i,
                200 + 2 * i,
                300 - i,
                400 + i,
                500 + 3 * i,
                600 - 2 * i,
                700 + i,
                800,
                900 + i,
                1000 - i,
                1100 + 2 * i,
                1200 - i,
            ],
            dtype=float,
        )
        rows.append({"values": values, "path": f"run-{i}.json"})
    return rows


def test_correlation_and_matrix_completion_keep_only_observed_values_exact():
    rows = _rows()
    selected_c, predicted_c = baselines._correlation_reconstruction(rows[0]["values"], rows[1:], 3)
    selected_m, predicted_m = baselines._matrix_completion(rows[0]["values"], rows[1:], 3)
    assert selected_c == [0, 1, 2]
    assert selected_m == [0, 1, 2]
    np.testing.assert_array_equal(predicted_c[:3], rows[0]["values"][:3])
    np.testing.assert_array_equal(predicted_m[:3], rows[0]["values"][:3])


def test_simple_end_to_end_baselines_keep_observed_values_exact():
    rows = _rows()
    for method in (
        baselines._mean_profile,
        baselines._global_scale,
        baselines._nearest_neighbor,
        baselines._linear_regression,
    ):
        selected, predicted = method(rows[0]["values"], rows[1:], 3)
        assert selected == [0, 1, 2]
        np.testing.assert_array_equal(predicted[:3], rows[0]["values"][:3])


def test_knn_reconstruction_uses_fixed_neighbor_count():
    rows = _rows()
    selected, predicted = baselines._nearest_neighbor(
        rows[0]["values"],
        rows[1:],
        3,
        neighbors=baselines.KNN_NEIGHBORS,
    )
    assert selected == [0, 1, 2]
    assert predicted.shape == (12,)


def test_adaptive_trace_has_fixed_budget_and_no_duplicate_indices():
    rows = _rows()
    selected, trace = baselines._adaptive_selection_trace(rows[0]["values"], rows[1:], 3)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert trace[0]["selection_basis"] == "fixed_first_probe"
    assert trace[1]["selection_basis"] == "observed_partial_scores_and_training_statistics"
    assert trace[-1]["observed_indices"] == selected


def test_fold_split_covers_each_run_once():
    folds = baselines._folds(5)
    held_out = [i for _train, test in folds for i in test]
    assert sorted(held_out) == list(range(5))


def test_kfold_split_matches_main_method_convention():
    got = baselines._build_splits(5, split_mode="kfold", split_seed=42, folds=5)
    rng = np.random.RandomState(42)
    indices = np.arange(5)
    rng.shuffle(indices)
    expected = []
    for part in np.array_split(indices, 5):
        test = [int(x) for x in part]
        expected.append(
            (
                [i for i in range(5) if i not in set(test)],
                test,
            )
        )
    assert [(train, test) for _sid, train, test in got] == expected


def test_train_test_split_is_disjoint_and_holds_out_expected_count():
    splits = baselines._build_splits(
        5,
        split_mode="train_test",
        split_seed=42,
        test_size=0.2,
    )
    assert len(splits) == 1
    _split_id, train, test = splits[0]
    assert len(train) == 4
    assert len(test) == 1
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(5))
