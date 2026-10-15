# Local Web-Serving Pilot

This pilot uses an isolated Nginx process bound to loopback port 18080 and the system `wrk` binary. It does not stop or modify the system Nginx service. The configured suite has five concurrency profiles (20, 100, 200, 500, and 1000 connections), with five complete repeated suites. Each profile runs for 20 seconds with four `wrk` threads. Profile order is pseudorandomized independently for each repetition using recorded deterministic seeds.

Run from the repository root:

```sh
bash agent_workflow/rounds/round-6-web-serving/run_wr_repeats.sh
```

The script stores host metadata, the HTTP smoke check, Nginx config validation output, randomized run order, and one complete `wrk` log per run/profile in this directory. Each log includes UTC timestamps, the exact command, throughput, latency distribution, and `/usr/bin/time` elapsed real time. The planned load duration is 5 profiles x 20 seconds x 5 repetitions = 500 seconds, plus startup and between-profile overhead.

Analyze the completed logs with the run-held-out predictor comparison:

```sh
python scripts/analyze_local_web_serving_oof.py \
  --input-dir agent_workflow/rounds/round-6-web-serving \
  --fold-seed 42 \
  --output-json agent_workflow/rounds/round-6-web-serving/results.json
```

The JSON contains all five run folds, per-run estimates and timing costs, model-selection traces, and SHA-256 hashes for the raw logs and run configuration. Baselines use the same five folds and K=3. Adaptive selection uses only the held-out run's observed profiles and training-fold statistics; it does not inspect held-out unobserved scores or durations to select probes.

This is a local static-content HTTP-serving pilot, not CloudSuite or PTS `pts/nginx`. Its results can establish workload feasibility and provide a small multivariate dataset. They do not by themselves establish generalization to public Web workloads, other server configurations, hosts, or session conditions.
