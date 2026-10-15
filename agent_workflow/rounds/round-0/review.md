# Round 0 Independent Review

This is an independent review of `BenchScout_WWW2026_submission.zip` and the repository's experiment artifacts. The review extracted the submission to a temporary directory and did not modify the manuscript or code. Full report: [`review_round0.md`](../../../review_round0.md).

## Decision

**Reject (2/5), confidence 4/5.** The current evidence does not establish the reported accuracy under an independent test protocol, nor does it support all host coverage claims. Editorial changes alone cannot resolve these issues.

## P0 blockers

1. **Evaluation split and sample size are inadequate for the generalization claim.** `scripts/audit_paper_protocol.py` finds 55 raw runs, 5 host identities, 11 host-suite pairs, and only one raw session with five runs per pair. It also finds no machine-readable train/test manifest for the headline numbers. Supplementary artifacts use `random_fold`, `n_samples=5`, and one validation sample per fold. Independent session holdout is not demonstrated.

2. **Headline results are not reproducible from the submitted package.** The ZIP contains manuscript and figures but not raw data, split manifests, per-fold predictions, or scripts. Repository experiment summaries point to author-machine absolute paths such as `/home/cxc/MoEBench/...`; `scripts/paper_aggregate_tables.py` could not load a referenced result in this environment. PTS's `log-mean` over selected profile values is an author-defined proxy; the paper must not conflate it with an official PTS aggregate score.

3. **M6/T4 coverage is unsupported by the available audit.** The manuscript lists six hosts and a 12,951-second PTS-GPU full-run time for M6, but `dataset/MACHINES.md` and the raw-data audit show only five host identities, with no M6/T4 data record. Provide its manifest/raw run and clarify its role, or remove the M6 claim and datum.

## Required evidence to reconsider

- A locked test set from independent sessions or newly collected configurations, with a sample-level split manifest and per-fold predictions.
- A clean, portable reproduction package that regenerates the paper tables from pinned code, data, configuration, and aggregation definitions.
- Clear separation of official UnixBench index results from author-defined PTS profile aggregates.
- Raw M6/T4 evidence if that host remains in the paper.

## Initial scores

Novelty 2/5; technical quality 2/5; experimental quality 1/5; clarity 3/5; reproducibility 1/5.

See the full report for P1/P2 findings, WWW formatting checks, and evidence references.
