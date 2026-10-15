# Round 2 Change Record

## Reviewer items addressed

- Kept the single current abstract and its supported headline numbers: 0.21% median UnixBench relative error and 15.1 s mean selected-probe time. The abstract also reports mean, P95, maximum, and the 18.1 s budget including estimated context collection.
- Corrected time-column bolding in the CPU tables and tie bolding in the GPU table.
- Clarified in both primary table captions and the runtime section that historical full-suite wall times and nominal OOF budgets are from separate measurements and do not form paired speedups.
- Added exact single-run online UnixBench trace timings for M1/M3/M4 (full suite 1,678.9--1,701.7 s; BenchScout trace 12.05 s), explicitly identifying that context collection is excluded and that these traces are separate from OOF accuracy. Source paths and SHA-256 values are in `live_timing_provenance.json`.
- Audited the available raw records: the supplied GPU table reports M1 / RTX 3090 Ti and is not an M6/T4 result. M6/T4 remains pending raw runs and manifests.

## Not resolved in this round

- Independent session/time holdout and nested or locked model selection require new experimental data.
- Historical full-suite table cells still lack a per-cell run ID, aggregation rule, and elapsed-time provenance. UnixBench OOF `full_suite_from_dataset` sums per-test timing fields rather than measuring suite elapsed time; PTS timing sums profile costs and omits unrecorded end-to-end overhead.
- The original submission archive does not include the full raw data, checkpoints, split manifests, and runnable artifact needed to reproduce the headline tables.
- The baseline comparison remains limited; no newly audited strong baseline was added.

The modifier made no unsupported or fabricated result claims. The original submission archive was not overwritten.
