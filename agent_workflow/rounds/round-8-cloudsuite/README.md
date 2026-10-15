# Round 8 CloudSuite Web-serving pilot

This is an isolated multi-configuration/multi-session pilot using the
CloudSuite Web Serving image set. It compares PHP-FPM `max_children=4` and
`max_children=8` at Faban scales 20, 40, 60, 80, and 100. Each configuration
has four complete five-load sessions; load order is randomized within each
session, and configuration order is randomized across sessions. The calibrated
protocol uses a 5 s ramp-up and ramp-down, with 60 s steady probes and 120 s
steady full references. These are workload-specific budgets, separate from
the UnixBench 15-second result.

All containers use a dedicated Docker bridge with round-8 names. The web and
database services have no published host ports. Memcached is capped at 512 MB
instead of the image default of 64 GB. The runner removes only containers and
the network it creates. Raw Faban stdout, summaries, service logs, Docker
metadata, and a run manifest are retained in `results/`.

The original `run_cloudsuite_profile.sh` keeps the upstream database entrypoint
and its download behavior. The local-seed path is isolated in
`bootstrap_local_seed.sh` and `run_cloudsuite_local_seed.sh`: the bootstrap
validates the archived seed hash, restores it in a disposable `bsr8-*`
container, checks authenticated SQL access, the exact 18-table ELGG schema,
and nonzero entity and metadata rows, stops MySQL cleanly, and commits a
round-8-only image. The local runner has a bounded two-cell `--smoke` mode
(scales 20 and 40 in one fresh 30-second stage), a `--calibrate` mode (two
fresh scale-20 stages at 60 and 120 seconds), a one-session `--pilot` matrix,
and a four-session `--full` matrix. The
calibration pair sets the workload duration based on Faban mix validity before
any full matrix is run. It creates only round-8
containers and a private bridge, requires exactly one Faban `summary.xml`, and
records XML structure, operation mix, transaction failures, and latency SLO
outcomes separately. A mix- or SLO-failing cell remains in the raw record;
only a structurally invalid result or client protocol error stops the run.
Each Faban invocation has a 300-second timeout. The runner verifies the ELGG
table count for each newly started database and recreates the database and
Memcached before each configuration/block. No workload result is claimed until
the seeded database and both smoke cells have been independently checked.

A completed result is a controlled pilot, not production Web evidence; client
and server share one host, and the test matrix is small.
