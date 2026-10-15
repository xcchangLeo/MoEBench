# CloudSuite Session-Held-Out Analysis Audit

The locked four-session result is in `final_supplemented_analysis.json`, produced
by `analyze_supplemented_sessions.py` from the original 80-cell matrix and the
independent session-5 supplement. The predefined validity rule selects
sessions 1, 2, 4, and 5. Session 3 has one failed Register transaction among
3,999 attempted operations in its fpm8/full/load-40 cell; it is excluded from
the strict primary analysis and retained as a separate sensitivity fold.
All 80 primary cells pass operation-mix validation and have zero transaction
failures. None passes the configured Faban latency SLO.

## Predefined Curve Results

Each held-out profile has five offered-load points. Three-point policies use
the same 60-second probe cells and are trained only on other sessions.

| Policy | Same-config MAPE | Cold-start MAPE | Measured selected-cell total |
| --- | ---: | ---: | ---: |
| History mean, K=0 | 5.26% | 39.16% | 0 s |
| Nested correlation, K=3 | 6.04% | 19.61% | 476 s |
| Adaptive correlation, K=3 | 6.40% | 17.35% | 466 s |
| Adaptive plus linear interpolation, K=3 | 7.61% | 7.21% | 466 s |
| Fixed evenly spaced plus linear interpolation, K=3 | 7.91% | 7.91% | 478 s |

The five-point full-reference total is 1,100.4 s per configuration, measured
as the sum of complete cell times. Three-point totals are reconstructed from
the individually measured selected cells, not timed as a separate online
selective workflow. Cold-start error for adaptive plus linear interpolation
is 9.74% with fpm4 as the new configuration and 4.68% with fpm8 as new;
history transfer yields 48.18% and 30.14%, respectively. Same-configuration
history remains more accurate than the tested adaptive correlation policy.

## Exploratory Decision Controls

`analyze_switch_decisions.py` reads the locked primary output and evaluates a
post hoc incumbent-to-new-configuration choice using geometric-mean throughput.
The old configuration's predictor is its other three full sessions; the new
configuration's predictor is its cold-start curve. History transfer chooses
incorrectly in 4/8 directional held-out decisions, with 15.665% average
throughput regret. Several three-point strategies choose correctly in 8/8.

`analyze_one_probe_baseline.py` adds all five fixed one-probe load choices,
scaling the incumbent history curve by one new-configuration short measurement.
Every one-probe choice also makes 0/8 wrong decisions, at mean curve errors
of 33.76%, 17.45%, 18.27%, 15.97%, and 18.64% for loads 20 through 100.
This decision is easy because fpm8 wins every full-reference session; the
three-point result supports better curve recovery than these one-point
controls, not a unique decision benefit. These two decision analyses were
added after the primary curve protocol and must be labeled exploratory.
Neither supports an SLO-feasible configuration claim because 0/80 primary
CloudSuite cells pass the configured latency SLO.
