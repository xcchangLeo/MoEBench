# M1 Baseline Report

Host: M1 (`aces-System-Product-Name`, 32U128G).
Subset policies use run-level five-fold CV within the canonical single session.
No row is a session-level temporal holdout.

| Suite | Method | Kind | Reports | Mean error (%) | Std (%) | Saved (%) |
|---|---|---|---:|---:|---:|---:|
| phoronix_cpu | correlation | subset | 1 | 1.762 | 0.000 | 73.022 |
| phoronix_cpu | fixed_cpu_mix | subset | 1 | 1.762 | 0.000 | 70.830 |
| phoronix_cpu | fixed_first_k | subset | 1 | 1.762 | 0.000 | 70.830 |
| phoronix_cpu | fixed_io_mix | subset | 1 | 1.762 | 0.000 | 70.830 |
| phoronix_cpu | full_suite | reference | 1 | 0.000 | 0.000 | 0.000 |
| phoronix_cpu | greedy_fastest | subset | 1 | 1.691 | 0.000 | 98.845 |
| phoronix_cpu | greedy_slowest | subset | 1 | 1.714 | 0.000 | 21.534 |
| phoronix_cpu | random | subset | 1 | 1.770 | 0.000 | 73.749 |
| phoronix_cpu | router | subset | 1 | 1.762 | 0.000 | 89.122 |
| phoronix_cpu | tousi_mlp | execution_free | 1 | 99.967 | 0.000 | n/a |
| phoronix_cpu | wang_dnn | execution_free | 1 | 85.535 | 0.000 | n/a |
| phoronix_gpu | correlation | subset | 1 | 0.918 | 0.000 | 39.777 |
| phoronix_gpu | fixed_cpu_mix | subset | 1 | 0.918 | 0.000 | 96.189 |
| phoronix_gpu | fixed_first_k | subset | 1 | 0.918 | 0.000 | 96.189 |
| phoronix_gpu | fixed_io_mix | subset | 1 | 0.918 | 0.000 | 96.189 |
| phoronix_gpu | full_suite | reference | 1 | 0.000 | 0.000 | 0.000 |
| phoronix_gpu | greedy_fastest | subset | 1 | 0.918 | 0.000 | 99.608 |
| phoronix_gpu | greedy_slowest | subset | 1 | 0.918 | 0.000 | 9.814 |
| phoronix_gpu | random | subset | 1 | 0.918 | 0.000 | 69.894 |
| phoronix_gpu | router | subset | 1 | 0.918 | 0.000 | 96.189 |
| phoronix_gpu | tousi_mlp | execution_free | 1 | 98.850 | 0.000 | n/a |
| phoronix_gpu | wang_dnn | execution_free | 1 | 0.256 | 0.000 | n/a |
| unixbench | correlation | subset | 1 | 0.253 | 0.000 | 71.876 |
| unixbench | fixed_cpu_mix | subset | 1 | 0.258 | 0.000 | 90.623 |
| unixbench | fixed_first_k | subset | 1 | 0.244 | 0.000 | 84.391 |
| unixbench | fixed_io_mix | subset | 1 | 0.256 | 0.000 | 71.870 |
| unixbench | full_suite | reference | 1 | 0.000 | 0.000 | 0.000 |
| unixbench | greedy_fastest | subset | 1 | 0.258 | 0.000 | 90.623 |
| unixbench | greedy_slowest | subset | 1 | 0.249 | 0.000 | 53.116 |
| unixbench | random | subset | 3 | 0.236 | 0.003 | 70.626 |
| unixbench | router | subset | 1 | 0.243 | 0.000 | 78.140 |
| unixbench | tousi_mlp | execution_free | 1 | 0.172 | 0.000 | n/a |
| unixbench | wang_dnn | execution_free | 1 | 0.209 | 0.000 | n/a |

Full-suite wall time is recorded in the JSON reference rows.

Incomplete or not-requested reports (not treated as zero or failed measurements):
- `dataset/m1_reviewer_random_seed7`: `phoronix_cpu` (not_completed_or_not_requested)
- `dataset/m1_reviewer_random_seed7`: `phoronix_gpu` (not_completed_or_not_requested)
- `dataset/m1_reviewer_random_seed21`: `phoronix_cpu` (not_completed_or_not_requested)
- `dataset/m1_reviewer_random_seed21`: `phoronix_gpu` (not_completed_or_not_requested)
