# M1 Execution-Only UnixBench Baselines

These methods use no training data, model, historical mean, or cross-run statistic.
They directly compute a suite estimate from actual observed partial subtest scores.

| Method | K | Mean error (%) | Std (%) | Median (%) | P95 (%) | Max (%) | Saved (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_partial_geomean | 3 | 18.537 | 0.213 | 18.489 | 18.846 | 18.899 | 84.391 |
| category_balanced_partial_geomean | 3 | 61.585 | 0.629 | 61.855 | 62.211 | 62.216 | 84.372 |
| random_partial_geomean_seed_42 | 3 | 26.086 | 0.376 | 26.217 | 26.435 | 26.472 | 74.995 |
| random_partial_geomean_seed_7 | 3 | 52.332 | 0.081 | 52.379 | 52.414 | 52.419 | 68.763 |
| random_partial_geomean_seed_21 | 3 | 31.888 | 0.419 | 31.892 | 32.362 | 32.434 | 84.372 |
| random_partial_geomean_seed_100 | 3 | 12.157 | 0.101 | 12.177 | 12.260 | 12.271 | 90.623 |
| random_partial_geomean_seed_123 | 3 | 66.295 | 0.627 | 66.314 | 67.004 | 67.013 | 78.121 |

Evaluation is an offline replay of historical complete runs. The complete run is used only for final error accounting.
The timing column is estimated from recorded per-subtest wall times; it is not a live online probe measurement.

Random subset results are reported per fixed seed and are not selected using test error.
