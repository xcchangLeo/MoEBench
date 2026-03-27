"""Reconstruct full UnixBench results from partial subtest runs + system features (xi)."""

from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial

from moebench.reconstruct.data import (
    build_partial_feature_row,
    build_partial_feature_row_from_executed_tests,
    collect_unixbench_run_paths,
    extract_targets_from_dataset,
    extract_test_index_from_block,
    full_suite_wall_seconds,
    partial_wall_seconds,
    preferred_run_block,
)

__all__ = [
    "load_reconstruction_bundle",
    "predict_from_partial",
    "build_partial_feature_row",
    "build_partial_feature_row_from_executed_tests",
    "collect_unixbench_run_paths",
    "extract_targets_from_dataset",
    "extract_test_index_from_block",
    "full_suite_wall_seconds",
    "partial_wall_seconds",
    "preferred_run_block",
]
