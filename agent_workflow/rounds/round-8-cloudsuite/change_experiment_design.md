# Post-CloudSuite Change Experiment Design

## Question

On this host and CloudSuite Web Serving workload, does a bounded target-host
profile after changing PHP-FPM `pm.max_children` from 4 to 8 estimate the
resulting offered-load curve better than a zero-probe forecast from the
host's prior fpm4 history?

This is a single-host configuration-transfer pilot. It does not test
cross-host or production generalization.

## Data Split and Lock

- Use only the four completed fpm4 sessions from the current CloudSuite run
  as source history and model-calibration data.
- Do not use any fpm8 result from the current run for model fitting, subset
  selection, threshold choice, or reporting the locked-test estimate.
- Before collecting target data, save a manifest containing the fpm4 source
  run hashes, software and container image digests, preprocessing, model
  versions, hyperparameters, selection rule, selected load levels, and
  prediction code hash.
- Select three of the five Faban scales (20, 40, 60, 80, 100) and all model
  settings using session-held-out validation within the four fpm4 sessions.
  Freeze these choices before any new fpm8 measurement. Training and
  validation folds must hold out entire sessions, including both probe and
  full-reference phases.
- After the lock, collect four new fpm8 sessions with globally unique IDs
  101--104. For each session, first
  write and hash the probe-based five-load prediction; only then collect the
  full-reference cells used as test labels. Keep this probe-then-reference
  phase order fixed in every target session. Do not update the model or
  selection between sessions.

The existing fpm8 rows are excluded from the test because each was collected
in a session that also contributed fpm4 measurements. The new target sessions
provide a clean temporal boundary from the source history.

## Paired Comparison

For each new fpm8 session, compare two predictions against that session's
full-reference profile at the same five offered-load scales:

- **History-only:** At each scale, predict the `BrowsetoElgg` operation p99 and
  aggregate throughput by the arithmetic mean of that scale's full-reference
  results from the four fpm4 source sessions. This uses no target-session
  measurement.
- **Bounded probes:** Use the locked three-scale selection and the short Faban
  results from that fpm8 session to estimate its five-point
  aggregate-throughput curve. Use only selected probe outputs and the frozen
  fpm4-trained models. `BrowsetoElgg` p99 is recorded descriptively, not
  predicted as a primary curve.

Before each target full-reference phase, lock three throughput predictions
from the same selected probe cells: correlation reconstruction, linear
interpolation, and least-squares scaling of the fpm4 history curve. The
fpm4-only leave-one-session-out selection chooses one of these as the primary
bounded-probe predictor. Report the other two as same-budget comparators,
regardless of their target-session performance. The zero-probe history mean
remains the paired primary control.

Both methods are scored on the same fpm8 full-reference cells. Record exact
runner elapsed time, container setup time, probe duration, and model overhead
separately; do not infer end-to-end savings from nominal steady-state budgets.

## Endpoint and Protocol

Throughput-curve MAPE is the primary endpoint, computed per session across the
five offered-load levels:

\[
E_T(s,m)=\frac{100}{5}\sum_{l=1}^{5}
\frac{|\widehat{T}_{s,m,l}-T_{s,l}|}{T_{s,l}}.
\]

The paired effect is \(E_T(s,\text{probes})-E_T(s,\text{history})\), reported
for each of the four target sessions and as their mean. The independent unit is
the session, not the 20 load-level cells. With four sessions, report the
small-sample uncertainty descriptively and avoid a general superiority claim.

Latency is a descriptive secondary outcome restricted to the named
`BrowsetoElgg` operation. Existing fpm4 full-reference cells show about 98--147
successful `BrowsetoElgg` operations per cell; other operations generally have
about 50--125. At these counts, an empirical p99 is effectively determined by
one or two upper-tail observations and is too unstable for a primary endpoint
or a strong SLO claim. Report its p99 and successful-sample count for every
probe and full-reference cell, without averaging it with other operations'
percentiles. This is not an overall request-mixture p99.

Extract only `responseTimes/operation[@name="BrowsetoElgg"]`'s `p99` from each
Faban summary. A leading `>` marks a right-censored lower bound and is not an
exact p99; the percentile `limit` attribute records an SLO threshold and does
not itself indicate censoring. Missing, malformed, or censored values remain
missing/censored in the report: do not substitute the SLO limit or another
operation, or pass a censored bound to a numeric model as an exact value.
Report Faban operation-mix validity, failed operations, p99 availability and
censoring, and actual elapsed time alongside the primary throughput endpoint.
Configuration choice or SLO classification is not an endpoint in this
experiment.

Reuse the existing CloudSuite image and seeded database, bridge isolation,
Faban validator, table/schema checks, raw XML retention, and per-cell
manifests. Reset the web, database, and Memcached services between cells as
the current runner does. The phase order is fixed: run the probe phase first,
lock its prediction, then run the full-reference phase in every target
session. Randomize load order independently within each phase. Capture host
load and container/image metadata for every session.
Invalid XML, client protocol failures, and operation errors are retained and
handled using criteria fixed before target collection, not removed based on
their effect on the estimate.

## Manuscript Footprint

If results are valid, report this as a separate configuration-transfer
scenario in the existing HTTP-serving subsection. Use one compact table with
the history and probe methods, primary throughput error, descriptive
`BrowsetoElgg` p99 values and sample counts, and measured cost. Keep the
static-content Nginx result as a distinct workload or move its detailed policy
table to the appendix if needed. Do not merge the two workloads into one
average, and do not claim the one-host result meets a multi-host WWW generality
bar.
