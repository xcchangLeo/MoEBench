# BenchScout research artifact

This bundle accompanies the anonymous WWW manuscript. It contains the code,
archived run records, saved models, experiment logs, and split/output JSON used
for the auditable results. It is a research snapshot, not a claim that every
number in the manuscript can be regenerated from the files here.

## Environment

The analyses in this snapshot were run with Python 3.13 on Linux. Relevant
installed package versions at packaging time: NumPy 2.3.4, SciPy 1.16.3,
scikit-learn 1.7.2, LightGBM 4.6.0, PyTorch 2.13.0, pandas 3.0.5, and psutil
7.2.2. Some older saved checkpoints were created with scikit-learn 1.8.0;
loading them under 1.7.2 emits compatibility warnings. `xgboost` is not
installed in this snapshot's analysis environment and is needed only for
scripts that explicitly select its model family. Docker and `wrk` are required
only to rerun workload collection, not to reanalyze the archived logs.

## Reproduce archived analyses

From the extracted bundle root, with the package versions above installed:

```sh
PYTHONPATH=. python scripts/analyze_local_web_serving_oof.py \
  --input-dir agent_workflow/rounds/round-6-web-serving \
  --fold-seed 42 \
  --output-json /tmp/benchscout_http_reanalysis.json

PYTHONPATH=. python scripts/analyze_http_profile_policies.py \
  --input-dir agent_workflow/round-8-http-profile/results \
  --output-json /tmp/benchscout_http_policy_analysis.json

PYTHONPATH=. python scripts/analyze_unixbench_host_holdout.py \
  --dataset-root dataset \
  --output /tmp/benchscout_host_holdout_reanalysis.json

PYTHONPATH=. python scripts/analyze_unixbench_component_oof.py \
  --dataset-root dataset \
  --oof-dir agent_workflow/rounds/round-3-crossfit-reconstructor \
  --output-json /tmp/benchscout_component_oof.json

PYTHONPATH=. python agent_workflow/rounds/round-8-cloudsuite/analyze_supplemented_sessions.py \
  --input-dir agent_workflow/rounds/round-8-cloudsuite/results/bsr8-local-20261013T080042Z-877852 \
              agent_workflow/rounds/round-8-cloudsuite/results/bsr8-local-supplemental-session5-20261013T121407Z-2008275 \
  --output-json /tmp/benchscout_cloudsuite_profiles.json

PYTHONPATH=. python agent_workflow/rounds/round-8-cloudsuite/analyze_switch_decisions.py \
  --input-json agent_workflow/rounds/round-8-cloudsuite/final_supplemented_analysis.json \
  --output-json /tmp/benchscout_cloudsuite_switch_decisions.json

PYTHONPATH=. python agent_workflow/rounds/round-8-cloudsuite/analyze_one_probe_baseline.py \
  --input-json agent_workflow/rounds/round-8-cloudsuite/final_supplemented_analysis.json \
  --output-json /tmp/benchscout_cloudsuite_one_probe.json
```

The round-6 HTTP analysis consumes 25 raw `wrk` logs and is retained as an
archived diagnostic. The round-8 CloudSuite study uses an original 80-cell
matrix and an independent fifth session. A predeclared all-cells-valid rule
selects sessions 1, 2, 4, and 5; session 3 has one failed Register transaction
and appears only in a separate sensitivity analysis. The 80 primary cells have
valid operation mixes and no transaction failures, but none meets the
configured latency SLO. Strict session-held-out analysis reports 5.26%
same-configuration MAPE for history-only and 6.40% for adaptive correlation.
For cross-configuration cold-start, adaptive correlation with linear
interpolation has 7.21% average MAPE across both transfer directions, versus
39.16% for history-only; fixed evenly spaced probes with interpolation give
7.91%. The three selected cells have 466 s measured total cell time per
configuration, compared with 1,100 s for five full-reference cells. These
totals are reconstructed from separately measured runs, not from a separately
timed online selective workflow. Post hoc one-probe controls have 0 wrong
choices in eight throughput-only decisions but 15.96--33.76% curve MAPE; they
do not establish a unique decision advantage for adaptive selection. The
separate static-Nginx study uses 80 raw logs from four repeated blocks and two
worker configurations; its mean held-out throughput curve error is 2.12% for
nested selection and 2.53% for fixed first-three, while corresponding p99
errors are 29.63% and 27.81%. The host holdout analysis consumes 25
UnixBench raw records and records every outer training/test host split. See
the respective `README.md` and `audit.md` files under `agent_workflow/rounds/`
for protocols and expected summaries. The existing JSON outputs contain
input file hashes; paths embedded in some archived reports refer to the
original collection machine and should be resolved relative to this extracted
bundle.

`SHA256SUMS.txt` lists relative paths and SHA-256 values for the included
code, data, and archived experiment files. From the extracted bundle root,
`sha256sum -c SHA256SUMS.txt` verifies file integrity. The CloudSuite analysis
command applies only to the original 80-cell matrix plus the independent
session-5 directory; do not combine them with earlier smoke, calibration, or
one-session pilot directories. The anonymous bundle includes raw Faban XML and validation records after the run
has been audited. Its JSON files use redacted paths; resolve those paths
relative to the extracted bundle before rerunning a script that reads them.

To repeat CloudSuite workload collection, obtain the upstream Elgg database
archive from `http://datasets.epfl.ch/cloudsuite/ELGG_DB.tar.gz` and verify
SHA-256 `7ca3c78076089f96a8f514e5a588b746e2d61e930373df45d28f9b948502307c`.
The 472 MB database archive and Docker images are not redistributed in this
bundle. `agent_workflow/rounds/round-8-cloudsuite/bootstrap_local_seed.sh`
creates the local seeded image after validating the archive and 18-table
schema; the collection scripts then require that image and the upstream
CloudSuite web, Memcached, and Faban client images. Image digests and Docker
metadata are retained with the original run.

## Evidence boundaries

- The historical CPU/UnixBench/PTS corpus has one five-run session for each
  available host-suite pair; run-level folds do not estimate session transfer.
- The current HTTP result uses static, loopback Nginx traffic on one host; four
  repeated blocks do not estimate between-session variation and the study does
  not execute the complete router/probe/reconstructor pipeline. Tail-latency
  errors are high, and the training-history baseline already chooses the same
  worker configuration as probe-based ranking.
- The multi-tier CloudSuite profile study is also limited to one host and two
  PHP-FPM settings. Its strict four-session analysis evaluates workload-specific
  curve selectors and estimators, not the complete suite pipeline. Same-setting
  history is a strong baseline; cross-setting cold-start favors target
  measurements for curve fidelity, with results varying by transfer direction.
  All cells fail the configured latency SLO. One-probe and switch-decision
  controls are post hoc and cannot support a claim of unique adaptive decision
  value.
- The host holdout analysis observes oracle full-run component scores on each
  target host; it is not a timed short-probe experiment.
- The M1 GPU table's 0.18% value is traceable to one online JSON report, but
  its `run-05` target overlaps the five-run corpus used to train/select the
  model grid. The report records 15.035 s of selected probes plus 7.930 s of
  context collection (22.965 s total); it is not an independent held-out
  result. The separate M6/T4 values still lack raw runs and a hardware
  manifest. Both are kept separate from the audited M1 five-fold result.
- Some original legacy tables and checkpoints lack a complete command/split
  manifest. Their reproduction cannot be guaranteed from this snapshot.

Neither the non-independent M1 online report nor the unverified M6/T4 values
should be pooled with the archived M1 GPU crossfit diagnostic (0.95% mean
error) or the local HTTP pilot.
