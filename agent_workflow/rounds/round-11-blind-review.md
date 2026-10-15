# Independent PDF-Only Review, Round 11

Source: anonymous `work_submission/main.pdf` compiled on 2026-10-13 before the
four-session CloudSuite result. The reviewer used a fresh context and read only
the PDF. Score: **2/5, Reject**. Revision alone did not raise the score.

## Acceptance Blockers

1. The visible Web evidence is one-host loopback static Nginx. History-only is
   more accurate than three-load probing (throughput MAPE 1.40% vs 2.12%; p99
   MAPE 14.02% vs 29.63%), and both choose the same configuration. A
   multi-tier, session-held-out configuration study with decision regret is
   needed.
2. The learned router ties three fixed policies on all 25 UnixBench test runs;
   22/25 suite-output heads do not use probe features. The router ranks high
   component scores rather than directly optimizing reconstruction error.
   Claims of an adaptive-routing gain need matched evidence or a revised
   method and objective.
3. System-suite data have one five-run session per host-suite, with four
   training runs per fold. A later temporal diagnostic and offline host
   transfer are substantially weaker. Independent session and host evidence
   is needed for generalization claims.
4. Historical full-run times and nominal probe budgets are not paired
   end-to-end timings. Wang/Tousi rows are local proxies rather than published
   implementations. Add measured Web pipeline cost and equal-budget curve
   fitting/adaptive selection baselines.
5. PTS log-mean is a paper-defined proxy, the M1 online GPU case overlaps
   model selection, and M6/T4 lacks raw provenance. Keep these results
   separate from independent efficacy claims.

The reviewer also found an unresolved appendix reference on PDF page 6; this
was fixed in the next TeX build. The current planned CloudSuite matrix and
frozen fpm8 target-session experiment address blockers 1 and 4, but their
results must be audited before any claim changes. The other blockers require
new evidence or narrower claims; they cannot be repaired by presentation.
