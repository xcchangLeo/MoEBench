# Independent WWW Review, Round 8 Interim

Decision: **Reject (2/5)**, pending CloudSuite evidence.

The whole-host holdout split and nested selection are auditable: the five
outer folds exclude every run from the test host during model fitting and
subset selection. The added context-only Ridge control is selected using only
source hosts. The score-ranked oracle policy reaches 4.49% mean error across
five held-out hosts, while context-only reaches 16.26%; this comparison is
limited to full-run component values supplied at test time. It cannot validate
the learned context router, short probe prediction, end-to-end timing, or Web
performance. The target-loss-selected subset is unstable (8.08% mean), and
the score-ranked policy ranges from 0.98% to 10.25% across test hosts.

The manuscript now distinguishes the oracle result from the 15-second
pipeline and moves user-supplied GPU figures into provenance-only appendix
material. The updated submission archive matches the manuscript source/PDF,
and a separate artifact bundle includes code and historical data. These
changes improve integrity and reproducibility but do not establish a novel
learned-routing advantage or external Web-serving performance. A favorable
decision still requires independent Web workloads/sessions, matched strong
baselines, tail-latency or configuration-decision metrics, and full-pipeline
online cost and accuracy.
