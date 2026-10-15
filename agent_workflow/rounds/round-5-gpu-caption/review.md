# Round 6 Independent WWW Review

## Recommendation

**Reject (2/5), confidence 5/5.** The revised GPU caption now describes the M1/M6 figures as reported values, discloses missing run-level records and unspecified aggregation/timing boundaries, and disclaims paired speedups. This is a material presentation improvement, but it does not make the GPU comparison reproducible or resolve the study's central evaluation and contribution problems. The score remains unchanged.

## Artifact and GPU Table Audit

- `work_submission/main.tex`, `work_submission/main.pdf`, and `BenchScout_WWW2026_submission.zip` are synchronized. The source and archived source hashes match; the PDF and archived PDF hashes match. The PDF is 12 pages, Letter size.
- The current GPU caption reads “Reported PTS-GPU comparison on M1 and M6.” It states that values are reported as provided, that run-level records and definitions for error aggregation and timing boundaries are absent, that the comparison is not independently reproducible from the submission bundle, and that the times are not used to claim paired speedups.
- M1/M6 labels are RTX 3090 Ti / NVIDIA T4. Values remain M1/M6 BenchScout error 0.18%/0.14%, time 15.0/15.0 s; Wang-style proxy error 0.26%/5.13%; Tousi-style MLP proxy error 98.85%/67.14%; and full-suite time 27,018.0/12,951.0 s. BenchScout times are no longer bolded. The lowest reported error per host remains bolded, which is a literal ranking of supplied values, not independently validated evidence.
- The paper separately reports the repository's M1 run-held-out OOF result of 0.9534% mean error at a nominal 18.035 s budget, and explicitly says not to pool it with the supplied table. The repository's audited M1 artifact covers five runs in one session; it does not contain M6 raw runs or a split/model record for the supplied M6 result.

## Remaining Blocking Findings

1. **The GPU values remain unverifiable.** The caption appropriately admits this limitation, but the submission still does not provide source identifiers, raw runs, sample/session counts, split membership, target aggregation inputs, per-run predictions, model/checkpoint configuration, or timing logs for the supplied M1/M6 cells. M6's reported proxy errors and BenchScout result cannot be independently checked.

2. **M1's two results remain unreconciled.** The supplied comparison gives 0.18% at 15.0 s, while the independently auditable OOF result gives 0.9534% at 18.035 s. Separate reporting and a no-pooling caveat are appropriate, but absent the supplied protocol it remains impossible to establish which result represents the primary method under the stated evaluation, or whether the suite target, training data, selected probes, context cost, and evaluation unit match.

3. **The GPU time cells remain uninterpretable as a common cost metric.** The caption now says timing boundaries are unspecified and disclaims paired speedups, which prevents the strongest earlier overclaim. Still, “Time (s)” places values of 3.5, 15.0, 12,951, and 27,018 in one column without identifying whether they are feature collection, selected-probe time, context-inclusive budget, end-to-end elapsed time, or sums of profile fields. The auditable M1 OOF full-suite value of 27,018.48 s is the maximum of five per-profile timing sums, not suite wall time.

4. **M6 remains outside the audited corpus.** The manuscript correctly marks its CPU/RAM and run count as unavailable, but reports M6 proxy and method errors without an artifact-backed feature set, training corpus, target definition, or split. The caption makes their supplied status clear; it cannot substitute for evidence needed to use these cells as experimental results.

5. **The main study still does not support its routing contribution or broad generalization claims.** Each audited host-suite pair has one session and five runs. Run-level folds do not establish temporal, session, configuration, or cross-host generalization. In UnixBench, learned and fixed policies produce identical suite predictions on all 25 runs, and the xi-only control is nearly tied. The nested crossfit diagnostic fixes the reconstructor feature-construction mismatch but leaves predictions exactly unchanged.

6. **Baseline and reproducibility gaps remain.** The execution-free baselines are local static-feature proxies, not reproductions, and alternative Tousi-inspired variants available in the repository are stronger than the selected MLP on GPU. The submission ZIP still lacks raw data, checkpoints, full split manifests, and a complete portable table-generation path. CPU table timing cells also remain incompletely traceable.

## Evidence Needed for Reconsideration

- Supply the raw/stable-access M1 and M6 records, evaluation protocol, host metadata, split membership, target calculation, per-run predictions, checkpoint/configuration, and timing logs; reconcile the two M1 errors.
- Define each table timing quantity and report repeated, paired online elapsed-time measurements before making runtime or speedup claims.
- Evaluate the pipeline with independent session/time/configuration holdouts and controls that test probe and routing value against xi-only reconstruction and fixed subsets.
- Compare against competitive, precisely specified baselines under the same protocol and provide a portable artifact reproduction path.
