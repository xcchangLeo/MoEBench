# CloudSuite Web Serving Analysis Contract

This contract is set before the four-session matrix. The one-session pilot is
used to check protocol viability and is excluded from confirmatory estimates.

## Workload and split

- One physical host runs the official CloudSuite Web Serving Faban, Web, DB,
  and Memcached images on a private Docker bridge. This is a local multi-tier
  experiment, not cross-host validation.
- Compare PHP-FPM `max_children=4` and `8` at Faban scales 20, 40, 60, 80,
  and 100. Independently restore the same seeded DB and cache for every
  config/session/load/phase cell, using 60-second probes and 120-second
  references. The one-session pilot used phase-level restores and is excluded.
- Collect four independently restarted sessions per configuration. Randomize
  configuration and load order and record the realized order. For each held-out
  session, fit all profile statistics and subset choices using the other three
  sessions only.

## Cell validity

- Require a complete single Faban `summary.xml`, positive totalOps and
  throughput, per-operation success/failure counts summing to totalOps, a
  canonical application URL, and no client protocol error in the log.
- Report operation-mix validity at Faban's unchanged absolute tolerance of
  0.0200. Use aggregate throughput for the primary curve only when the mix
  passes and no transaction fails. Retain other cells as explicit invalid or
  availability outcomes, with the reason and denominator shown.
- Treat Faban latency-limit failures as measured SLO outcomes. Do not convert
  `benchSummary passed=false` into a protocol failure when the operation mix
  and transaction protocol are valid.

## Comparisons

- Primary endpoint: mean absolute relative error of the five-load throughput
  curve per held-out config/session, then the mean over those profiles. Use
  $K=3$ 60-second probe loads per configuration.
- Secondary cold-start configuration endpoint: hold out every complete target
  configuration, train only on the other PHP-FPM configuration's complete
  curves from the other three sessions, and use only the target's three short
  probes at evaluation. Compare with the other configuration's history mean
  and the same fixed/random subsets. This is within-host configuration
  transfer, not unseen-host generalization; report all eight held-out profiles.
- Compare a training-only nested subset selector; fixed first loads (20, 40,
  60), fixed middle loads (40, 60, 80), and fixed evenly spaced loads (20, 60,
  100); an adaptive correlation selector that starts at load 20 and chooses
  the next two loads from training correlations and observed short probes;
  uniform random selection among all ten $K=3$ load subsets; and a zero-probe
  history-only predictor. For each fixed subset, report correlation
  reconstruction plus paired linear-interpolation and history-scale variants
  on the same three short observations. Report the random subset's exact
  expected error over all ten choices per profile; for a two-configuration
  decision, average over the 100 independent subset pairs rather than
  averaging predictions from more than three probes.
  Every probe-based method receives the same 60-second observations and uses
  the same training sessions; no held-out full references guide selection.
- Secondary endpoint: select the higher-throughput PHP-FPM configuration by
  the geometric mean of the predicted five-load curve and compare with the
  choice from full references. Report wrong-choice count and throughput regret
  against the full-reference winner. Report the history-only decision too.
- Report measured runner wall time and per-cell setup time for the selected
  probes and full references, with model-fitting time separately accounted
  for. Selected-subset cost reconstructed from measured cells is not an
  independently timed online selective execution; do not call nominal
  steady-state seconds an end-to-end speedup.
- Report transaction failures, Faban operation-mix pass rate, and per-operation
  latency SLO violations. Per-operation p95/p99 values are descriptive with
  sample counts; sparse short-window p99 is not a primary endpoint.

Aggregate uncertainty by held-out session. Present all four session outcomes
and do not infer independent host or deployment generalization from this run.
