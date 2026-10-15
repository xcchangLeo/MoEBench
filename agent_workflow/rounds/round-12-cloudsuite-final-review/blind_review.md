# Independent PDF-Only Review, Round 13

The reviewer read only the anonymous 11-page `work_submission/main.pdf` built
after the four-session CloudSuite analysis. Recommendation: **Weak Reject
(2/5)**. The reviewer found the paper anonymous, in ACM two-column format,
within the 12-page cap, and self-contained in the first eight pages.

## Acceptance blockers

1. **Operational Web benefit is unproven.** Table 5 (p. 8) shows that
   same-configuration history beats adaptive correlation (5.26% versus 6.40%
   MAPE). Fixed one-load controls make zero wrong choices in the eight
   exploratory throughput decisions, and no CloudSuite cell meets the latency
   SLO (p. 6). The cold-start 7.21% versus 39.16% curve result has not been
   shown to improve an SLO-feasible choice. Run prospective multi-configuration
   tuning with feasible SLOs and compare zero-, one-, and three-probe policies
   at matched cost.
2. **Independent Web evidence is narrow.** CloudSuite uses one host, two FPM
   settings, five loads, and four strict restarted sessions per setting (p. 6).
   Nginx also uses one host and favors history. Report session uncertainty and
   error specifically on unobserved loads; extend to more configurations,
   workloads, hosts, and days where possible.
3. **The suite pipeline does not establish its mechanism or generalization.**
   Learned and fixed policies make identical UnixBench predictions on all 25
   within-session folds (p. 6), probe features have zero split gain in 22/25
   suite-output heads, and each fold has only four training runs. The single
   later-run diagnostic has 6.69% error (p. 10). Prospective session-held-out
   paired trials and matched fixed/history/context-only controls are needed.
4. **The Web method's novelty and coherence remain limited.** It uses a
   separate correlation/interpolation implementation rather than the named
   suite router/probe estimator/reconstructor (pp. 1, 4, 6). Adaptive and
   fixed evenly spaced interpolation have close cold-start error (7.21% versus
   7.91%, p. 8), with a reversed ordering in one transfer direction. A
   reusable method or robust methodological insight needs broader evidence.
5. **Cost evidence is reconstructed.** CloudSuite selective time sums measured
   cells from the full grid; it is not a timed selective online run (p. 6).
   Suite 18.1 s costs are estimates compared with historical full runs.
   Measure paired online workflow time including setup, context, load
   generation, and probe overhead.

The reviewer considered the Web fit and transparent presentation strengths,
but found the positive result limited to cold-start throughput-curve fidelity
on one host and two configurations. Revision alone did not raise the score.
