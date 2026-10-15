# Cross-Fitted Reconstructor Input Audit

## Contract Finding

The original UnixBench OOF runner trains the reconstructor with `scripts/reconstruct_train_eval.py::build_augmented_train_matrix`. That helper calls `moebench.reconstruct.data::build_partial_feature_row`, which extracts observed subtest scores from the training run's full `yi` block and full-run subtest times. Held-out evaluation instead uses probe-model predictions from `predict_subtest`, plus timing fields from the collected probe sample. This is a train/evaluation feature mismatch (oracle partial scores and full-run times in reconstructor training versus estimated scores and probe times at evaluation), rather than held-out-label leakage into fold fitting.

## Corrected Diagnostic Protocol

For each outer five-fold split (seed 42), the four outer training runs are split into four inner folds. Each inner probe LightGBM is fit only on the other three runs and predicts all subtests for its held-out training run. The reconstructor's 20 random partial rows per training run use those cross-fitted probe predictions and probe timing fields. The outer held-out run uses the outer-fold probe model fit on all four training runs, with the original fold-local router and the same LightGBM reconstructor configuration. Thus, both reconstructor training and evaluation use predicted subtest scores and probe times, while each training run is excluded from the probe model that creates its reconstructor features.

Timing feature parity was checked in code: both paths use each selected run/test sample's `probe.wall_s`, falling back to `probe.duration_s`, then 4 seconds. The corrected training rows use the same rule directly; held-out evaluation uses the standard `evaluate_single_run_offline` path.

## Results

| Host | n | Mean error (%) | Median (%) | P95 (%) | Max (%) | Max per-run prediction delta vs headline |
|---|---:|---:|---:|---:|---:|---:|
| M1 `aces-System-Product-Name` | 5 | 0.180512 | 0.190717 | 0.296985 | 0.313247 | 0 |
| M2 `iZbp1glgt48i9a8d49embxZ` | 5 | 0.286465 | 0.297843 | 0.428370 | 0.457021 | 0 |
| M3 `iZbp15n87643uk1sqjrdvdZ` | 5 | 0.343874 | 0.435346 | 0.621549 | 0.663015 | 0 |
| M4 `iZbp16krl0yc7euw7sb6slZ` | 5 | 0.194445 | 0.196983 | 0.210190 | 0.210306 | 0 |
| M5 `iZbp1acaw5wdllhz47922rZ` | 5 | 1.946944 | 0.298579 | 4.652531 | 4.715528 | 0 |

Across the same 25 runs, the original and corrected suite predictions match exactly on every run. Aggregate mean error is 0.590448%, median is 0.210306%, `np.percentile(errors, 95)` with NumPy's default linear interpolation is 3.653039%, and maximum is 4.715528% for both. Therefore, this corrected diagnostic does not change the reported 0.21% median / 0.59% mean headline numerically. The contract mismatch remains a methodological issue; the observed invariance shows that the fitted suite head's output was unaffected by replacing oracle training partial scores with cross-fitted probe predictions in this corpus. Do not generalize this numerical invariance to other data or model settings.

## Artifact Hashes

Each corrected report includes SHA-256 hashes for its probe dataset and input run JSON files. The hashes below identify the corrected report and corresponding original learned-policy report:

| Host | Original report SHA-256 | Corrected report SHA-256 |
|---|---|---|
| M1 `aces-System-Product-Name` | `8eb1e987fd6d11fefec2c9151a7f06fefed34429bc2b53261d347f76c5c6192b` | `78e2a7a73af0cca755dccf498377335e4d48ebf403650f5c2ea83e540d40fd87` |
| M2 `iZbp1glgt48i9a8d49embxZ` | `fa8d9f387951a783fe2c38434517e3480fc5c1bf5180bf261c4e6dd9b6aac8c3` | `ecc532d140238a90d420f0d91983d4020718bb77c97fa8b5f56c03f0f6d00fbf` |
| M3 `iZbp15n87643uk1sqjrdvdZ` | `48208139dbe7f709474a55363ef794847b648cd83ba8cf04f40e9753a28f2fec` | `d3443abda4a4acd6811377717caf9d5feb15ac5c22aa026de2e078bfe8ad1895` |
| M4 `iZbp16krl0yc7euw7sb6slZ` | `7433af57c88b19980a32cf3763660a66729a57bb08751d58f75cc3099815669b` | `336ed1ec91b3baaacf8ca623e3a396ee44d06accdef6e1ad1aec89456040e4b5` |
| M5 `iZbp1acaw5wdllhz47922rZ` | `493eace1ca7ab58ced3fab2b6e1b93c4bccce531ca5c4463d48abaf230968a3e` | `e62be27ad2aa2154bd9d9e38dd3468ae18808f27627050b71cc018950653b5b9` |

Reproduction script: `scripts/run_corrected_oof_unixbench.py`. Corrected per-host reports: `*_crossfit_reconstructor_oof.json`. Original reports are under `agent_workflow/rounds/round-2-unixbench-policies/`.
