# M1 Alternative End-to-End Baselines

Offline replay on M1 (`aces-System-Product-Name`), UnixBench, `K=3`.
Split: `kfold`; all statistics are fit on the training split only; no test-driven tuning.

| Method | Mean error (%) | Median (%) | P95 (%) | Max (%) | MAE | RMSE | Saved (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| correlation_reconstruction | 0.770 | 0.886 | 0.928 | 0.937 | 24.554 | 25.134 | 84.391 |
| matrix_completion | 0.503 | 0.561 | 0.795 | 0.852 | 16.027 | 17.776 | 84.391 |
| adaptive_probing | 0.072 | 0.008 | 0.259 | 0.318 | 2.280 | 4.542 | 70.620 |

Selection controls and BenchScout results are reported separately.
The timing column is estimated from historical `ti` fields, not an online 5-second probe.
