# Round 1 Independent Review

## Recommendation

Reject in the current form; major revision is required before a WWW-level accept recommendation.

## Evidence

The leakage-controlled OOF runner (`scripts/run_hybrid_oof_unixbench.py`) fits the router, probe model, and reconstructor using only four training runs in each fold. Held-out runs are used only for probe-feature replay and final scoring. Across five hosts and 25 UnixBench held-out runs:

| Statistic | Result |
|---|---:|
| Mean suite relative error | 0.590% |
| Median suite relative error | 0.210% |
| P95 suite relative error | 3.653% |
| Maximum suite relative error | 4.716% |
| Mean selected-probe time | 15.08 s |
| Mean estimated end-to-end budget (including 3 s context collection) | 18.08 s |

The independent reviewer found no explicit train/test label leakage in this runner. The evaluation is nevertheless one-session-per-host run-level random folding, with one held-out run per fold. It cannot establish new-session, temporal, or cross-host transfer. The 0.21% value is therefore a median, not a mean; the 15.08 s value is probe time, not end-to-end time. Two M5 runs have 4.40% and 4.72% errors around a sharp ground-truth state change and must remain visible in the paper.

The same fold-local protocol was run for all PTS suites with a fixed 30-tree LightGBM reconstructor. PTS-CPU host means were 1.739% (M1), 0.194% (M2), 0.255% (M3), 0.297% (M4), and 0.192% (M5); PTS-GPU on M1 was 0.953%. Their nominal end-to-end budgets were 18.03--18.05 s. These results are reported separately from the UnixBench median and do not support a cross-suite 0.2% claim.

## Required Revision

Keep the contribution framed as a practical sparse accelerator for repeated benchmark execution, but report the median together with mean and tail errors, define the timing components, and avoid presenting the current corpus as independent generalization evidence. M6/T4 must not be represented without raw run records and manifests.

## Audit Corrections and Remaining Findings

The reviewer note about a duplicate abstract was based on an earlier snapshot. The current `work_submission/main.tex` contains one abstract paragraph, and its current PDF must be regenerated from that source before submission. The current CPU table previously bolded only the lowest errors and omitted the tied-best 3.5 s execution-free times; the current edit now bolds those ties. In the GPU table, Wang has the best error (0.26%), and Wang and Tousi tie for the lowest execution-free time (3.5 s); both timing values are now bold.

The full-suite timing discrepancy remains unresolved. The OOF reports derive `full_suite_from_dataset` by summing recorded per-test `time_s` values (UnixBench) or `time_s_total`/`execution_cost` values (PTS). The primary tables retain earlier full-suite baselines from separate experiment artifacts, including UnixBench `full_unixbench` wall timings. These values therefore do not constitute paired wall-clock comparisons with the OOF probe budgets. The manuscript must not use them to claim measured OOF speedups unless the collection and timing protocols are reconciled.

The M1 GPU table contains the values 0.26% (Wang), 0.95% (BenchScout), and 18.0 s (BenchScout) for an RTX 3090 Ti. It is not an M6/T4 result. The current raw records and manifests still lack an M6/T4 host run, so that experiment remains pending.

The independent recommendation remains Reject/current major revision because the evaluation has only one session per host-suite pair, model selection reuses the same five-run M1 folds, full-suite time provenance is not aligned with OOF results, and the submission archive lacks the complete runnable artifacts needed to reproduce the tables.
