# Independent WWW Review: Pre-CloudSuite

## Recommendation

**Reject (2/5), confidence 5/5.** The paper is candid about limitations and the WWW 2027 Web Infrastructure and Agentic Systems track is a plausible fit. However, the current evidence establishes neither reliable value for Web-serving operators nor an advantage from the learned sparse-selection pipeline. A successful CloudSuite run may add one useful workload result; by itself, it will not resolve the independence, baseline, and contribution gaps below.

This is an independent assessment. No changes to the manuscript, experiment code, or results were made for this review.

## WWW Fit and Framing

The CFP calls for explicit Web relevance on the first page and says that using a Web artifact alone is insufficient ([`www要求.md:67`](../../../www要求.md:67)). The selected track explicitly includes Web performance measurement and characterization, virtualization and resource management in Web infrastructure, and empirical systems evaluation ([`www要求.md:276`](../../../www要求.md:276), [`www要求.md:282`](../../../www要求.md:282)). The subject therefore fits the track in principle.

The present framing opens with Web-serving operations and bounded HTTP profiles, but then makes generic UnixBench/PTS benchmark reconstruction the dominant evaluated contribution ([`main.tex:204`](../../../work_submission/main.tex:204), [`main.tex:286`](../../../work_submission/main.tex:286), [`main.tex:293`](../../../work_submission/main.tex:293)). The title and track citation cannot substitute for a demonstrated Web-systems result. The local Nginx study is explicitly loopback, one host, one period, and not the complete BenchScout pipeline ([`main.tex:817`](../../../work_submission/main.tex:817), [`main.tex:823`](../../../work_submission/main.tex:823)).

An ambitious and defensible Web-systems framing remains possible: define the problem as **budget-constrained characterization of Web-service behavior across offered load**, where the operator needs throughput and tail-latency curves to compare configurations or enforce an SLO. Make the scientific question whether sparse, target-system measurements can preserve these operational conclusions under a fixed measurement budget. Treat UnixBench/PTS as controlled evidence about the generic measurement components, not as a substitute for Web-service validation. State this concrete Web problem and its implications on page one.

## Blocking Findings

1. **No demonstrated decision value in the current HTTP pilot.** The eight held-out configuration-block profiles report 2.12% throughput curve error for nested K=3 selection versus 2.53% for fixed-first-three, but p99 curve error is worse (29.63% versus 27.81%). The nested method has lower throughput error in only two of four block averages. Probe-based and training-history-only comparisons select the same configuration in every block ([`main.tex:821`](../../../work_submission/main.tex:821), [`main.tex:823`](../../../work_submission/main.tex:823), [`main.tex:835`](../../../work_submission/main.tex:835)). This is feasibility evidence for loopback profile estimation, not evidence that probing improves an operator decision or tail-latency estimate.

2. **The central learned-routing claim is contradicted by the reported controls.** All four routing policies give exactly the same UnixBench suite prediction on all 25 held-out runs; context-only error is comparable to the complete pipeline (0.61% versus 0.59%), and one representative fold gives all probe features zero split gain ([`main.tex:750`](../../../work_submission/main.tex:750)). The whole-host replay reports 15.07% mean error and 40.44% worst-host mean, with near-identical predictions across four policies ([`main.tex:752`](../../../work_submission/main.tex:752), [`main.tex:763`](../../../work_submission/main.tex:763)). These results do not establish that learned context routing or observed probes improve reconstruction.

3. **Current evaluation does not establish session or deployment generalization.** The abstract reports five-fold run-held-out results over 25 UnixBench runs, while the paper states there is one five-run collection session per host-suite pair ([`main.tex:144`](../../../work_submission/main.tex:144), [`main.tex:293`](../../../work_submission/main.tex:293)). Run-level folds do not represent new sessions, changing workload conditions, or new Web deployments. The later same-host checkpoint diagnostic has n=1 and 6.69% error ([`main.tex:771`](../../../work_submission/main.tex:771)); it is a useful negative diagnostic, not a generalization estimate.

4. **The timing comparisons are not paired end-to-end measurements.** The manuscript states that primary partial costs are budget estimates, while full-suite values come from separate historical runs ([`main.tex:724`](../../../work_submission/main.tex:724), [`main.tex:725`](../../../work_submission/main.tex:725)). In the HTTP experiment, 24.45 seconds is load-generator runner time for six selected probes, omitting server setup, fitting, context collection, and orchestration ([`main.tex:823`](../../../work_submission/main.tex:823)). Thus the current results do not quantify total operator-facing savings under a matched protocol.

5. **The independent-host result is an oracle diagnostic, not an end-to-end transfer result.** The held-out host receives full-run component scores, so the 4.49% score-ranked result and other oracle comparisons do not validate short-probe prediction or the full pipeline ([`README.md:18`](../round-8-host-holdout/README.md:18), [`main.tex:1134`](../../../work_submission/main.tex:1134)). The existing interim review correctly treats it as limited information-value evidence and maintains rejection ([`reviewer_interim.md:3`](../round-8-host-holdout/reviewer_interim.md:3), [`reviewer_interim.md:20`](../round-8-host-holdout/reviewer_interim.md:20)).

6. **CloudSuite currently contributes no valid workload result.** The seeded database passed validation, but the latest attempt status is `web_not_ready config=fpm4 session=1 phase=probe`; the run manifest is empty, so no Faban measurement was accepted ([`status.txt:1`](results/bsr8-local-20261013T055546Z-439181/status.txt:1), [`runs.jsonl:1`](results/bsr8-local-20261013T055546Z-439181/runs.jsonl:1)). A seeded database or a passing setup check is not Web performance evidence. A future smoke must include a real Faban summary that passes the independent XML validator; a single smoke cell would still be a pilot, not independent validation.

7. **Some manuscript results remain explicitly non-reproducible.** The manuscript acknowledges that the package lacks the raw M6 GPU record and protocol for the supplied GPU table, and that complete provenance is missing for some historical values ([`main.tex:1110`](../../../work_submission/main.tex:1110), [`main.tex:1151`](../../../work_submission/main.tex:1151)). Keep such entries provenance-only or remove them from evidence-bearing comparisons until raw records, split definitions, predictions, and timing semantics can be audited.

## Falsifiable Acceptance Bar

The following is a concrete bar for a regular WWW Research Track acceptance recommendation on the stated Web-systems contribution. These criteria are deliberately testable; passing only the first criterion would support a pilot, not acceptance.

1. **Independent Web evidence:** Evaluate the full pipeline on at least two materially different Web-serving workloads, including one multi-tier workload such as CloudSuite and a second workload with a distinct application or traffic profile. Collect at least three independently restarted sessions per configuration on at least three host identities. Keep all model, subset, and threshold choices outside the final test sessions. If compute constraints prevent this scale, narrow the target claim and explain the resulting limit, but do not present the study as deployment-general.

2. **Same-budget baselines:** On each locked test session, compare the full sparse pipeline against (a) the best fixed K-load subset selected using training data only, (b) a context/history-only predictor, and (c) full measurement. Use the same training data and cost accounting for every method. Publish per-session predictions and raw measurements. Aggregate uncertainty by host/session rather than treating load points as independent samples.

3. **Operational outcomes:** Pre-register one primary profile metric and one decision metric. A reasonable target is mean throughput-curve error at most 5% and p99-curve error at most 10% at K no greater than 3 of 5 load levels on the locked set, with confidence intervals reported. For configuration choice, report SLO classification errors or regret against full profiles; the sparse method must improve over both fixed-subset and history-only controls, with a paired 95% interval excluding no improvement. If it does not, the paper must describe the null result and not claim decision benefit.

4. **Measured savings:** Report repeated paired end-to-end elapsed time from beginning of context collection through returned estimate, including service setup where the deployment workflow requires it, policy/model overhead, probe execution, and orchestration. Compare against full profiling under randomized order. A speedup claim requires lower paired elapsed time with uncertainty and no violation of the pre-registered accuracy/decision criterion; estimates from nominal budgets do not qualify.

5. **Reproduction:** Provide an anonymous artifact path that regenerates the primary tables from raw or stably accessible records, locked split manifests, configs/checkpoints, environment versions, and exact commands. Preserve an explicit distinction between official benchmark scores and paper-defined aggregates.

The numerical error targets above are reviewer-proposed practical thresholds for the paper's stated sparse-profile goal, not requirements quoted from the CFP. They can be replaced before data collection with a justified application-specific tolerance, but cannot be selected after observing the final test results.

## Prior Review Context

The prior independent review reached the same broad conclusion: the oracle holdout did not validate the learned router, short probes, end-to-end timing, or Web performance, and called for independent Web sessions, matched baselines, tail-latency or decision metrics, and full-pipeline online cost and accuracy ([`reviewer_interim.md:20`](../round-8-host-holdout/reviewer_interim.md:20)). Earlier review evidence also identifies the routing null result, weakly matched baselines, timing provenance, and artifact reproduction gaps ([`review.md:18`](../round-2/review.md:18), [`review.md:34`](../round-2/review.md:34)). Editorial caveats improved scientific integrity, but do not change those experimental findings.

## Score

| Dimension | Score |
|---|---:|
| Novelty / contribution | 2/5 |
| Technical quality | 2/5 |
| Experimental quality | 1/5 |
| Clarity | 4/5 |
| Reproducibility | 3/5 |
| Overall | 2/5, Reject |

The clarity score reflects unusually direct disclosure of negative results and limitations. It does not raise the scientific score because the main contribution still lacks demonstrated advantage on Web workloads.

## Reassessment Trigger

Reassess CloudSuite separately after there is a completed, valid workload result with raw Faban XML, validator output, nonzero completed operations, zero or explicitly analyzed errors, and recorded image/configuration provenance. Treat that result as one pilot observation. It changes this recommendation only if it contributes to the independent, matched, repeated Web evaluation defined above.
