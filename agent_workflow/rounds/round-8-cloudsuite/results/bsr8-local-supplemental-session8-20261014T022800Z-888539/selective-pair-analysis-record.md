# Session 8 Selective Pair Analysis

This is the second exploratory single-host FPM4 pair under the frozen fixed-even policy. The runner was invoked for session 8 with `CONFIG_SPECS=fpm4:4`, `PROBE_LOADS='20 60 100'`, and `FULL_LOADS='20 40 60 80 100'`. The runner hash, Docker version, image identifiers, randomized load orders, per-cell validation JSON, phase totals, and raw Faban summaries are retained in this directory.

All eight FPM4 cells are valid and have zero operation failures. Probe throughputs at loads 20/60/100 are 16.517/26.183/23.283 ops/s. Same-session full-reference throughputs at 20/40/60/80/100 are 17.167/25.358/21.808/23.842/21.825 ops/s. Linear interpolation predicts 21.350 and 24.733 ops/s at the unprobed loads 40 and 80. Their mean absolute percentage error against the full reference is 9.7714%; the mean over all five full-reference loads is 10.0142%.

Summing `cell_total_seconds` in `cell-times.jsonl` gives 475.056 s for the three probe cells and 1,086.334 s for the five full-reference cells. The phase-level totals in `phase-times.jsonl` are 475.066 and 1,086.350 s; the paper reports cell-total sums for consistency with session 6. Probe/full planned steady durations were 60/120 s, and the probe phase ran first. Treat this as n=1 exploratory joint accuracy/cost evidence; it does not isolate probe-count effects or rule out temporal drift.
