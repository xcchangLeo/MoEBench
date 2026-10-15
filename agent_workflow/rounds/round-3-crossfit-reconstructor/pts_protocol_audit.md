# PTS Cross-Fitted Reconstructor Audit

The PTS runner has the same original contract mismatch as UnixBench: `build_augmented_train_matrix_pts` builds reconstructor training rows from measured `pts_export` profile values and full-run profile times, while held-out inference uses probe-model predictions and probe sample wall times.

I ran the corrected protocol on M1 (`aces-System-Product-Name`) for both available PTS suites. Each outer fold fits the probe model on the four outer training runs; each of those four runs receives reconstructor training features from an inner leave-one-run-out probe model. Held-out rows use the outer-fold probe model. PTS feature rows use `value` and `time_s` triplets with the same probe wall-time fallback (`wall_s`, then `duration_s`, then 4 seconds) used by inference. The paper-defined `logmean` suite target and existing PTS estimator counts are unchanged.

| Suite | Mean error (%) | Median (%) | P95 (%) | Max (%) | Original-to-corrected prediction delta |
|---|---:|---:|---:|---:|---:|
| PTS-CPU | 1.738691 | 0.868968 | 4.133219 | 4.435523 | 0 for all 5 runs |
| PTS-GPU | 0.953391 | 0.469875 | 1.905783 | 1.968496 | 0 for all 5 runs |

The corrected metrics and every per-run suite prediction are exactly equal to the original PTS OOF reports. This indicates that, for these M1 suites and estimators, replacing oracle training profile values with cross-fitted probe predictions did not affect the suite output. It does not remove the contract mismatch as a methodological issue and should not be generalized beyond this corpus.

Corrected reports: `m1_pts_cpu_crossfit.json` and `m1_pts_gpu_crossfit.json`. Reproduction script: `scripts/run_corrected_oof_phoronix.py`. The canonical UnixBench runner has already been promoted to inner-OOF probe-feature training; the PTS runner remains unchanged pending broader protocol review.
