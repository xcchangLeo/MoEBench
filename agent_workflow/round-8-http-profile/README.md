# Round 8 Multi-configuration HTTP Profile Study

This local, loopback-only study compares two Nginx configurations that differ
only in `worker_processes` (1 versus 4). Four repeated profile blocks per
configuration measure 20, 40, 60, 80, and 100 concurrent connections. For each
configuration/block, the runner starts a fresh Nginx instance and runs all five
4 s probes in randomized load order. It stops and restarts Nginx, then measures
all five 20 s full references in a separately randomized order. Configuration
order is randomized within each block. The manifest records each phase's order,
timestamps, and Nginx master PID.

The separate 4 s probe samples support an offline fixed-budget policy
comparison with K=3 load levels. Nested profile selection is evaluated using
short probes from training blocks, then the held-out 4 s probe values predict
the held-out 20 s full-reference profile. Configuration ranking compares the
predicted profiles for both configurations, so its measured probe cost includes
three selected loads on each configuration. A train-history-only configuration
choice is reported separately. These are local policies, not a trained
host-context router or the full BenchScout pipeline. The study is same-host,
static-content, loopback traffic; repeated blocks do not create independent
host or deployment sessions. Reported probe time excludes context collection,
policy fitting, and orchestration.

An illustrative 1 ms p99 threshold is recorded before analysis to count
per-load tail-latency violations. This is a laboratory threshold for this
loopback workload, not a production SLO. The analysis also reports full-profile
throughput and p99 errors, per-load curve error, probe-to-reference fidelity,
configuration ranking accuracy and regret, measured probe cost on both
configurations, and p99 threshold counts. The analyzer verifies one order row
per config/session/phase, checks that each order matches timestamps, confirms
all five probes completed before the full-reference phase, and verifies that
the phases used separate Nginx instances.

The initial phase-adjacent attempt was interrupted after four indexed
measurements when we identified that each probe followed its full reference.
An additional full-reference log completed but was interrupted before its
manifest entry. All five records are preserved in
`results-invalid-preflight-phase-order/` and excluded from the corrected study
and analysis.

Run with `bash agent_workflow/round-8-http-profile/run_http_profile_study.sh`.
Analyze with `python scripts/analyze_http_profile_policies.py --input-dir
agent_workflow/round-8-http-profile/results --output-json
agent_workflow/round-8-http-profile/results/policy-analysis.json`.

The corrected run completed all 80 measurements with zero runner failures.
Nested K=3 selection has 2.12% mean throughput curve error across the eight
held-out configuration-block profiles, versus 2.53% for fixed first-three;
nested error is lower in two of the four block-level averages. P99 curve
errors are 29.63% and 27.81%, respectively. Short-probe per-load throughput
error averages 2.38% for workers1 and 2.37% for workers4, while p99 error
averages 30.85% and 36.53%. Both the probe-based and training-history-only
configuration choices select workers4 in all four blocks, so the ranking
result does not show a probe advantage. The 1 ms p99 confusion counts use all
20 paired load cells per configuration, including loads outside the selected
K=3 subsets; they are a dense-probe diagnostic, not sparse-policy results.
The measurement is one host and one collection period, and repeated blocks
are not independent deployment sessions.
