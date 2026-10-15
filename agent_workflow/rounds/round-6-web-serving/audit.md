# Local HTTP Pilot: Analysis Audit

## Reproduction

From the repository root, regenerate the complete analysis with:

```sh
python scripts/analyze_local_web_serving_oof.py \
  --input-dir agent_workflow/rounds/round-6-web-serving \
  --fold-seed 42 \
  --output-json agent_workflow/rounds/round-6-web-serving/results.json
```

The run parsed all 25 raw `wrk` logs, including each reported p99 latency. The JSON retains the raw log SHA-256 values, environment, fold membership, predictor outputs, selection traces, and input-artifact hashes. At the time of this audit, SHA-256 is:

| Artifact | SHA-256 |
|---|---|
| `scripts/analyze_local_web_serving_oof.py` | `b46ddf3733cc08317a4f0123df447675ef6815964bc67fe4a15438a983ef689e` |
| `results.json` | `7a004892386e1d61dcaefd463e48718bd35aafeccb29d9bf8537ac014edc0b18` |

## Split and Selection Audit

There are five outer folds, each holding out one whole repeated suite run and fitting correlation statistics on the other four. The seeded fold manifest in `results.json` is shared by the throughput and p99 targets. In each outer fold, the nested fixed-subset method scores each of the ten K=3 profile sets by leave-one-run-out prediction using only those four outer-training runs; it selects the lowest mean inner error, then evaluates that selected set once on the untouched outer run. The outer held-out target values are used for scoring only, not for choosing this nested set.

The adaptive method uses observed scores from the held-out run to choose later profiles, as an online adaptive policy would. Its trace records those observed-profile choices; its correlation and scale statistics come from the outer training runs. The selector does not inspect unobserved held-out values or held-out durations. Profile duration is summed only after selection to report retrospective cost.

The analysis also reports all ten fixed K=3 subsets on every outer held-out run (50 records per target). These rows are useful for seeing sensitivity to the selected profiles. They are not an unbiased estimate of the best subset: selecting a winner after inspecting these held-out comparisons would use test outcomes. Use the nested selector for a train-only selected fixed-subset comparison.

## Results and Scope

Five-run OOF summary (mean absolute relative error of the profile geometric mean):

| Target and method | Mean error | P95 error | Mean estimated time saved |
|---|---:|---:|---:|
| Throughput, adaptive correlation | 0.1557% | 0.4477% | 39.97% |
| Throughput, nested train-only fixed K=3 | 0.1763% | 0.5190% | 39.99% |
| Throughput, fixed-first-K mean fill | 0.1964% | 0.3559% | 40.00% |
| Throughput, fixed-first-K correlation | 0.2849% | 0.4721% | 40.00% |
| Throughput, rank-2 completion | 0.3203% | 0.5293% | 40.00% |
| Throughput, random K=3 correlation | 0.6354% | 1.7189% | 40.03% |
| p99 latency, adaptive correlation | 4.6029% | 11.1978% | 39.95% |
| p99 latency, nested train-only fixed K=3 | 3.8816% | 5.6436% | 39.98% |
| p99 latency, fixed-first-K mean fill | 2.0339% | 4.2201% | 40.00% |

The near-equal saved-time figures follow from measuring three of five nearly equal 20-second profiles. They are offline sums of the selected historical `wrk` process durations, not live end-to-end savings; context collection, model fitting, and orchestration are excluded. The small five-run sample from one collection period is descriptive and cannot establish a stable winner. The p99 target uses the `99%` values printed by `wrk`, converted to microseconds, then scores the geometric mean across profiles. The large gap between p99 and throughput errors shows that throughput results do not imply tail-latency accuracy.

This result is only a localhost static-content HTTP pilot. It is not a PTS result, CloudSuite result, production workload, or cross-host evaluation. The PTS `pts/nginx` attempt did not start because its upstream `wrk` download failed. Do not use this pilot to validate the paper's M1 GPU result or supplied M6/T4 figures.

## Separation From M1 GPU Audit

This artifact analyzes five local HTTP runs and reports throughput and p99 latency only. The separate M1 PTS-GPU offline crossfit result is in `../round-6-m1-gpu-audit/m1_pts_gpu_crossfit_rerun.json`, with a five-run mean error of 0.953391%; it is based on archived GPU benchmark runs, not these HTTP logs. The HTTP figures must not be described as supporting, reproducing, or replacing that M1 result. The supplied 0.1761% M1 online record is also a distinct result, but its protocol is not held-out: its grid was trained on all five M1 runs and its probe data includes samples from run-05, the run used for that reported ground truth. It must not be conflated with the audited M1 OOF result or with this HTTP pilot.
