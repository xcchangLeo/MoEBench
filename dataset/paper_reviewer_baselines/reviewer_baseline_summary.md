# Reviewer Baseline Summary

All values are means across available host reports. The available completed run used fixed seed 42.
Errors are relative errors; saved is the fraction of full-suite wall time saved.
The effective split is run-level five-fold CV because each host-suite pair has one session.

| Suite | Policy | Reports | Mean error (%) | Std (%) | Median (%) | P95 (%) | Max (%) | Saved (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phoronix_cpu | correlation | 5 | 0.568 | 0.668 | 0.305 | 1.285 | 1.393 | 38.357 |
| phoronix_cpu | fixed_cpu_mix | 5 | 0.556 | 0.674 | 0.274 | 1.280 | 1.389 | 31.440 |
| phoronix_cpu | fixed_first_k | 5 | 0.556 | 0.674 | 0.274 | 1.280 | 1.389 | 31.440 |
| phoronix_cpu | fixed_io_mix | 5 | 0.556 | 0.674 | 0.274 | 1.280 | 1.389 | 31.440 |
| phoronix_cpu | greedy_fastest | 5 | 0.555 | 0.636 | 0.267 | 1.263 | 1.363 | 84.169 |
| phoronix_cpu | greedy_slowest | 5 | 0.556 | 0.648 | 0.267 | 1.267 | 1.368 | 10.103 |
| phoronix_cpu | random | 5 | 0.575 | 0.669 | 0.331 | 1.302 | 1.420 | 53.213 |
| phoronix_cpu | router | 5 | 0.559 | 0.673 | 0.279 | 1.279 | 1.387 | 58.245 |
| phoronix_gpu | correlation | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 39.777 |
| phoronix_gpu | fixed_cpu_mix | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 96.189 |
| phoronix_gpu | fixed_first_k | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 96.189 |
| phoronix_gpu | fixed_io_mix | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 96.189 |
| phoronix_gpu | greedy_fastest | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 99.608 |
| phoronix_gpu | greedy_slowest | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 9.814 |
| phoronix_gpu | random | 1 | 0.918 | 0.000 | 0.440 | 1.884 | 2.044 | 69.894 |
| phoronix_gpu | router | 1 | 0.918 | 0.000 | 0.439 | 1.884 | 2.042 | 96.189 |
| unixbench | correlation | 5 | 0.470 | 0.479 | 0.267 | 1.076 | 1.212 | 74.743 |
| unixbench | fixed_cpu_mix | 5 | 0.469 | 0.480 | 0.269 | 1.066 | 1.199 | 90.636 |
| unixbench | fixed_first_k | 5 | 0.470 | 0.479 | 0.265 | 1.080 | 1.216 | 84.390 |
| unixbench | fixed_io_mix | 5 | 0.471 | 0.479 | 0.268 | 1.078 | 1.215 | 71.870 |
| unixbench | greedy_fastest | 5 | 0.469 | 0.480 | 0.266 | 1.075 | 1.210 | 90.640 |
| unixbench | greedy_slowest | 5 | 0.471 | 0.479 | 0.273 | 1.073 | 1.208 | 53.116 |
| unixbench | random | 5 | 0.481 | 0.520 | 0.258 | 1.095 | 1.214 | 77.000 |
| unixbench | router | 5 | 0.471 | 0.479 | 0.267 | 1.084 | 1.223 | 74.626 |

No row is a session-level temporal holdout. Correlation metadata is computed from the training fold.
