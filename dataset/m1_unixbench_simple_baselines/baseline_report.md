# M1 Alternative End-to-End Baselines

Offline replay on M1 (`aces-System-Product-Name`), UnixBench, `K=3`.
Split: `kfold`; all statistics are fit on the training split only; no test-driven tuning.

| Method | Mean error (%) | Median (%) | P95 (%) | Max (%) | MAE | RMSE | Saved (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean_profile | 0.171 | 0.085 | 0.356 | 0.373 | 5.448 | 6.920 | 84.391 |
| global_scale | 0.280 | 0.275 | 0.514 | 0.554 | 8.925 | 10.418 | 84.391 |
| nearest_neighbor | 0.256 | 0.282 | 0.337 | 0.337 | 8.172 | 8.541 | 84.391 |
| knn_reconstruction | 0.226 | 0.162 | 0.395 | 0.413 | 7.200 | 8.158 | 84.391 |
| linear_regression | 0.528 | 0.568 | 0.806 | 0.827 | 16.834 | 19.051 | 84.391 |

Selection controls and BenchScout results are reported separately.
The timing column is estimated from historical `ti` fields, not an online 5-second probe.
