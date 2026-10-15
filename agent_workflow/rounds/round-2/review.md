# Round 3 Independent Review

## Recommendation

**Reject (2/5), confidence 4/5.** The latest edits improve labeling and timing disclosure but do not resolve the study's central validity and generalization limits. Textual clarification alone does not change the score.

## Verified Changes

- `work_submission/main.tex` now has a single abstract. It reports UnixBench median error 0.21%, mean error 0.59%, a nominal 18.1 s budget, and explicitly says evaluation uses one collection session per host and does not measure independent-session generalization.
- Baselines are labeled as local Wang-style DNN and Tousi-style MLP proxies, with explicit language that they are not reproductions of the cited methods.
- The main results section now reports the matched learned-versus-fixed policy null result, the xi-only comparison, and a representative M5 suite-head sensitivity diagnosis.
- The GPU table is identified as M1 / NVIDIA RTX 3090 Ti. Its OOF row is numerically consistent to rounding: BenchScout mean error is 0.9534% and nominal budget 18.035 s; Wang DNN is 0.2559% and Tousi MLP is 98.8503%.
- GPU reported cost is now explicitly distinguished from end-to-end wall time: 27,018.0 s is the maximum of five per-profile timing sums, with a reported mean of 13,774.752 s. The table correctly warns against treating these quantities as paired speedups.
- Current PDF is compiled from the revised source (12 pages, Letter). The ACM reference-format warning is gone; a hyperref PDF-string warning remains.

## Blocking Findings

1. **No independent generalization test.** Each host-suite pair has one session and five runs. Random run-level folds with one held-out run per fold cannot establish session, temporal, or cross-host generalization. Model and Top-K choices reuse this small evidence without a locked test or nested selection.

2. **Policy OOF provides no evidence that learned routing improves suite accuracy.** The round-2 policy artifacts use distinct selections and fold-local probe/reconstruction models, but all four policies produce bitwise-identical per-run suite predictions for all five hosts. The manuscript now reports this null result. The fold-matched xi-only controls differ from policy predictions on only 3 to 5 of 5 runs per host, and aggregate error changes are small and inconsistent: policy vs xi-only mean error is 0.181% vs 0.179% (M1), 0.286% vs 0.280% (M2), 0.344% vs 0.342% (M3), 0.194% vs 0.196% (M4), and 1.947% vs 2.043% (M5). Across the 25 runs, means are 0.590% for the policy pipeline and 0.608% for xi-only; on M1, xi-only is slightly better. These five-run-per-host controls show no measurable accuracy advantage among routing policies and cannot establish a probe benefit. The manuscript also reports a representative M5 fold where all probe features have zero split gain; the separate audit confirms that 10% probe score/time perturbations leave that fold's suite output unchanged. This candid disclosure improves the paper's accuracy, but the negative result weakens the central learned-routing contribution. See `round-2-unixbench-policies/policy_audit_summary.json`, `prediction_sensitivity.md`, and `xi-only-control/`.

3. **Baseline comparison remains weak.** The table now correctly identifies the Wang-style DNN and Tousi-style MLP as local proxies rather than reproductions of the cited methods. This fixes attribution risk but leaves them as limited baselines. The same available GPU baseline artifact reports other Tousi-inspired variants (RF, decision tree, elastic net) near 0.26% error, versus 98.85% for the selected MLP proxy. The paper now disclaims family-level conclusions, but does not compare with stronger historical-data or dynamic-context approaches under an independent test. All execution-free baselines use leave-one-run-out within one session on a known host with static features, not a test of new-host prediction.

4. **Primary table timing remains partly unauditable.** The GPU table now correctly labels its reported-cost semantics. In contrast, primary CPU full-suite cells still lack per-cell run identifiers, aggregation statistics, and a consistent elapsed-time definition. The policy/OOF artifacts' UnixBench `full_suite_from_dataset` sums per-test `ti` values (320 s on M1), not total suite elapsed time; PTS sums profile costs and omits setup and end-to-end overhead. The three live traces are one run per host, exclude context collection from BenchScout time, and are separate from accuracy evaluation. They are useful case demonstrations, not statistical timing estimates or accuracy-latency pairs.

5. **Reproduction remains incomplete.** The submission ZIP lacks raw runs, model checkpoints, full split manifests, and a complete runnable table-generation path. Absolute machine-specific paths remain in experiment JSON. PTS remains a paper-defined log-mean proxy rather than an official PTS aggregate.

## Evidence Needed for Reconsideration

- Independent session/time/configuration holdout, with model and Top-K selection completed before the locked test.
- Replicated sensitivity diagnostics across folds and independent sessions, showing whether probes and routing add measurable predictive value over an xi-only reconstructor and fixed subsets.
- Traceable per-cell timing sources with one elapsed-time definition; keep profile-time sums distinct from end-to-end wall time and report repeated timing runs.
- Competitive baselines evaluated under the same split and precisely named proxy/model variants.
- Portable artifact bundle with raw data or stable access, checkpoints/configuration, split membership, environment, and table-generation commands.
