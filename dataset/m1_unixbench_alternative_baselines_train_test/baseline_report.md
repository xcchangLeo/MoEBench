# M1 Alternative End-to-End Baselines

Offline replay on M1 (`aces-System-Product-Name`), UnixBench, `K=3`.
Split: `train_test`; all statistics are fit on the training split only; no test-driven tuning.

| Method | Mean error (%) | Median (%) | P95 (%) | Max (%) | MAE | RMSE | Saved (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| correlation_reconstruction | 0.611 | 0.611 | 0.611 | 0.611 | 19.417 | 19.417 | 84.401 |
| matrix_completion | 0.561 | 0.561 | 0.561 | 0.561 | 17.843 | 17.843 | 84.401 |
| adaptive_probing | 0.008 | 0.008 | 0.008 | 0.008 | 0.251 | 0.251 | 68.740 |

Selection controls and BenchScout results are reported separately.
The timing column is estimated from historical `ti` fields, not an online 5-second probe.
