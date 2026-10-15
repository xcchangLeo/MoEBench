# Calibration Stop Record

The frozen 12-cell buffer-pool/SLO calibration was stopped after interruption at the user's direction. It is incomplete and is not evidence for manuscript claims.

- Cells 1-8 have complete runner records in `cells.csv`; all passed structural validation with zero operation failures, and all failed the native latency SLO.
- Cell 9 completed the client load phase, but the runner stopped before result validation and before writing a `cells.csv` row. Its raw client output, container logs, inspection data, and diff are preserved when available. It is not a validated observation.
- Cells 10-12 were not run.
- No result was imputed, rerun, or used in the manuscript.
- All containers and the Docker network with prefix `bsr8bpcal-20261014T003808Z-453545` were removed. The host sampler PID recorded earlier was already absent at cleanup; no matching sampler or runner remains.

The partial records remain available for audit only. They must not be described as a completed calibration or used to infer a feasible configuration.
