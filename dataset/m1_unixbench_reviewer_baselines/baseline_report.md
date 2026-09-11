# M1 Baseline Report

Host: M1 (`aces-System-Product-Name`, 32U128G).
Subset policies use run-level five-fold CV within the canonical single session.
No row is a session-level temporal holdout.

| Suite | Method | Kind | Reports | Mean error (%) | Std (%) | Saved (%) |
|---|---|---|---:|---:|---:|---:|
| unixbench | correlation | subset | 3 | 0.253 | 0.000 | 71.876 |
| unixbench | fixed_cpu_mix | subset | 3 | 0.258 | 0.000 | 90.623 |
| unixbench | fixed_first_k | subset | 3 | 0.244 | 0.000 | 84.391 |
| unixbench | fixed_io_mix | subset | 3 | 0.256 | 0.000 | 71.870 |
| unixbench | full_suite | reference | 1 | 0.000 | 0.000 | 0.000 |
| unixbench | greedy_fastest | subset | 3 | 0.258 | 0.000 | 90.623 |
| unixbench | greedy_slowest | subset | 3 | 0.249 | 0.000 | 53.116 |
| unixbench | random | subset | 3 | 0.236 | 0.003 | 70.626 |
| unixbench | router | subset | 3 | 0.243 | 0.000 | 78.140 |
| unixbench | tousi_mlp | execution_free | 3 | 0.200 | 0.025 | n/a |
| unixbench | wang_dnn | execution_free | 3 | 0.209 | 0.000 | n/a |

Full-suite wall time is recorded in the JSON reference rows.
