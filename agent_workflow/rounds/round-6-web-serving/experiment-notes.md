# Experiment Notes

## PTS `pts/nginx` attempt

Command attempted: `phoronix-test-suite batch-benchmark pts/nginx`.

PTS identified `pts/nginx-3.1.0`, estimated 53 seconds to install and 340 seconds to run, then failed while downloading the upstream `wrk-4.2.0.tar.gz` source from GitHub. The system's installed `wrk` is version 4.1.0 and nginx is 1.24.0. No PTS Nginx benchmark started. The batch interface also reported that batch mode was not configured. The selected local workload below uses the installed binaries directly and is labeled as a local HTTP pilot, not a PTS result.

## Isolated runner startup attempts

The first runner invocation stopped while collecting environment information because the Debian `wrk` binary rejects `--version`. A second invocation confirmed it also rejects `-v`; both print usage and return nonzero. The runner was changed to record the package version with `dpkg-query`. Neither failed invocation started Nginx or sent benchmark traffic.

The final runner validated `nginx.conf` with `nginx -t`, started a dedicated Nginx process on `127.0.0.1:18080`, and passed an HTTP 200 smoke check before load generation. It did not stop or modify the existing Nginx service. The runner's trap sent `SIGQUIT` only to the PID created for this isolated configuration.

## Workload and analysis

The captured suite consists of five concurrency profiles (20, 100, 200, 500, and 1000 connections), each using four `wrk` threads for 20 seconds. Five full repeated suites completed in 8 minutes 21 seconds including Nginx startup and inter-profile overhead. Each repetition uses a deterministic pseudorandom profile order. `run-order.txt`, `host_context.txt`, `nginx-config-test.log`, `http-smoke-check.txt`, `nginx.conf`, the static response, and all 25 raw `wrk` logs are retained. Every log contains exactly one parsed throughput and elapsed-time value; none reports socket errors or non-2xx/3xx responses. Across five repeats, throughput coefficient of variation by concurrency level ranges from 0.51% to 1.12%.

`results.json` evaluates five-fold run-level holdout with K=3 on the five repeated suite runs. All predictor fitting is limited to four training runs in each fold. The suite target is a geometric mean of requests/second across concurrency levels, defined for this pilot and not an official benchmark score. Profile costs use measured elapsed time for the selected `wrk` processes, excluding context collection, model fitting, and orchestration. The randomized, fixed, adaptive, correlation, and rank-2 matrix-completion controls all use the same folds and K.

| Method | Mean error (%) | P95 (%) | Mean measured time saved (%) |
|---|---:|---:|---:|
| Fixed first K, training mean fill | 0.1964 | 0.3559 | 40.00 |
| Random K, correlation reconstruction | 0.6354 | 1.7189 | 40.03 |
| Adaptive selection, correlation reconstruction | 0.1557 | 0.4477 | 39.97 |
| Fixed first K, correlation reconstruction | 0.2849 | 0.4721 | 40.00 |
| Fixed first K, rank-2 matrix completion | 0.3203 | 0.5293 | 40.00 |

The adaptive method is slightly better than fixed-first-K correlation on this five-run pilot, while a simple mean-fill fixed baseline is close. Given only five runs from one observation period and the low-complexity static payload, this difference is descriptive and does not establish a robust adaptive-probing advantage. Throughput remains approximately 463k--484k requests/s across the five concurrency settings; this lightweight profile is not a stress or saturation test.

This is one machine, one short observation period, a localhost client, and a small static response. It is a reproducible HTTP-serving workload pilot, not evidence of deployment-scale Web performance or cross-host/session robustness. It cannot validate the paper's reported PTS-GPU comparison or the supplied M6/T4 figures.
