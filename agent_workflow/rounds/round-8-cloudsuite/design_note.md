# CloudSuite Study Scope

The CloudSuite analysis is a session-held-out sparse throughput-curve
reconstruction study. Each profile has five offered-load points; three
60-second target-session probes are used to estimate the corresponding
five-point 120-second throughput reference curve. Training uses only the
other sessions' complete curves (or, in the cold-start configuration analysis,
the other configuration's complete curves from the other training sessions).

This experiment does not implement or validate BenchScout's full three-stage
pipeline. It has no learned context-conditioned router, no probe-telemetry to
official-score estimator, and no masked reconstructor trained on
probe-imputed scores. Its correlation selector and conditional curve predictor
are experiment-specific sparse reconstruction methods. The 60-second Faban
throughput at a load is already the measured target metric at that load, so
the probe-estimator stage collapses to a short-window versus reference-window
calibration problem rather than the UnixBench micro-probe score prediction
problem.

The two simple Web baselines use exactly the selected three probe loads for
each selection policy: piecewise-linear interpolation with linear endpoint
extrapolation, and a least-squares multiplicative calibration of the
training-session mean curve. The random-subset baselines average predictions
over all ten three-load subsets. All conclusions are limited to four restarted
sessions on one host; they do not establish cross-host or deployment
generalization.
