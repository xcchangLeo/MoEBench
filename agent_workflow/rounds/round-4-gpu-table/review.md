# Round 5 Independent WWW Review

## Recommendation

**Reject (2/5), confidence 5/5.** The M1/M6 table update fixes the earlier hardware-label inconsistency and the source, PDF, and submission ZIP are now synchronized. However, the table still presents protocol-unspecified supplied numbers as an “Overall effectiveness” result, so the GPU evidence is not independently auditable and cannot support the comparative claims made by the table.

## Artifact and Table Checks

- `work_submission/main.tex`, `work_submission/main.pdf`, and the root submission ZIP now have matching source/PDF content and hashes. The PDF is 12-page Letter output and contains the M1 RTX 3090 Ti and M6 NVIDIA T4 table.
- The manuscript correctly states that the M1/M6 values are supplied, distinct from the repository's M1 run-held-out OOF diagnostic (0.95% at a nominal 18.0 s), and not pooled with that diagnostic. It also correctly omits CPU/RAM specifications and run-count metadata for M6 rather than inventing them.
- The supplied table reports M1/M6 BenchScout errors of 0.18%/0.14%, 15.0 s times, proxy errors of 0.26%/5.13% and 98.85%/67.14%, and full-suite times of 27,018.0/12,951.0 s. The repository independently contains only the M1 OOF result (mean error 0.9534%, nominal 18.035 s) and no M6 run, split, or model artifact.

## Blocking Findings

1. **The supplied GPU table is not reproducible or independently auditable.** The manuscript says the aggregation and evaluation protocol is unspecified, but gives no source identifier, raw run records, sample count, session count, split manifest, model/checkpoint configuration, target definition, per-run predictions, or timing log for either the supplied M1/M6 comparison. The M6 rows therefore cannot be verified from the submission bundle or repository.

2. **The table's presentation overstates the status of the supplied values.** Its caption calls this “Overall effectiveness,” says that hosts with “available GPU measurements” are reported, and bolds BenchScout as the best method. Those labels read as validated experimental results even though the text admits that the supplied protocol is unknown. The table should be explicitly marked as an external/unverified comparison or removed from the main results until the provenance is supplied; bolding and comparative language should not be used for unknown-protocol values.

3. **M1 has two materially different BenchScout results with no reconciliation.** The supplied table reports 0.18% at 15.0 s, whereas the independently auditable five-run M1 OOF artifact reports 0.9534% at 18.035 s. Saying that the protocols differ is not enough to determine which result is the paper's primary estimate: the manuscript does not state the supplied split, training data, probe duration, context-cost inclusion, reconstructor/router versions, or whether its target is the same paper-defined PTS-GPU log-mean proxy.

4. **Timing semantics are ambiguous in the GPU table.** The generic “Time (s)” column places 15.0 s beside full-suite values without saying whether these are selected-probe time, context-inclusive budget, end-to-end wall time, or profile-time sums. The auditable M1 OOF record's 27,018.48 s is the maximum of five sums of per-profile timing fields, not an end-to-end suite wall time. The table's 27,018.0 and 12,951.0 s values cannot be compared as speedup evidence without their provenance and elapsed-time definition.

5. **The M6 proxy rows are especially unsupported.** M6 has no reported CPU/RAM metadata, raw run, or split in the repository, yet the table reports static-feature proxy errors and a 3.5 s feature cost. It is impossible to establish what features were available, how the proxy was trained, or whether its M6 values use the same target and protocol as M1. These cells should be `N/A` unless their artifacts are provided.

## Remaining WWW-Level Concerns

- The UnixBench evidence remains a one-session, five-run-per-host exploratory evaluation with no independent session, temporal, configuration, or cross-host holdout.
- Learned and fixed routing policies produce identical suite predictions on all 25 audited runs, and the xi-only control is essentially tied; the paper does not demonstrate a routing accuracy benefit. The nested crossfit diagnostic corrects the training-feature mismatch but leaves all held-out predictions unchanged.
- Baselines remain local static-feature proxies, and stronger available proxy variants are not consistently included in the primary comparison.
- CPU timing cells still lack complete per-cell provenance; the submission bundle lacks raw data, checkpoints, complete split manifests, and a portable table-generation path.

## Evidence Needed for Reconsideration

- Add the supplied M1/M6 source artifacts: raw or stable-access run records, host metadata, target aggregation definition, per-run predictions, train/validation/test membership, model/configuration checkpoints, probe and context timings, and commands that regenerate every table cell.
- Reconcile the supplied M1 result with the auditable 0.9534%/18.035 s OOF result, or remove one of them from the main comparison.
- Label supplied values as non-audited external comparisons and remove bold/best claims until the protocol is documented.
- Provide an independent session/time/configuration holdout and replicated routing/probe sensitivity controls.
