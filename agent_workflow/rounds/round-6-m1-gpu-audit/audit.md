# Round 6: M1 GPU and Baseline Audit

## M1 PTS-GPU crossfit rerun

This is an offline re-evaluation of the five archived M1 PTS-GPU full runs. It is not a new collection session and does not reproduce the separately supplied M1 value of 0.18% at 15 seconds.

Command:

```sh
python scripts/run_corrected_oof_phoronix.py \
  --machine aces-System-Product-Name \
  --pts-suite pts/nvidia-gpu-compute \
  --folds 5 --seed 42 --top-k 3 \
  --probe-duration-s 4 --xi-overhead-s 3 \
  --train-aug 20 --train-k-min 2 --train-k-max 6 \
  --reconstructor lightgbm --lgbm-estimators 30 \
  --report-json agent_workflow/rounds/round-6-m1-gpu-audit/m1_pts_gpu_crossfit_rerun.json
```

Environment observed during the rerun: Linux x86_64; Python 3.13; PyTorch 2.13.0+cu130; CUDA available; two NVIDIA GeForce RTX 3090 Ti GPUs; NVIDIA driver 580.178.04. The rerun used archived benchmark records and probe samples, not a live PTS execution.

The report records SHA-256 hashes for the five run JSON files and probe dataset, all outer and inner fold memberships, per-run predictions, and estimated probe costs. It matches `round-3-crossfit-reconstructor/m1_pts_gpu_crossfit.json` exactly on fold manifests and all five run errors:

| Metric | Rerun |
|---|---:|
| Runs / sessions | 5 / 1 |
| Mean relative error | 0.953391% |
| Median relative error | 0.469875% |
| P95 relative error | 1.905783% |
| Maximum relative error | 1.968496% |
| Nominal evaluation budget | 18.035 s |

The one-session, five-run corpus supports only exploratory run-held-out analysis. It does not establish temporal, configuration, or cross-host generalization. M6/T4 has no raw run, split, model, or timing artifact in the repository, so no local rerun can verify its supplied value.

## Existing alternative baselines

`dataset/m1_unixbench_alternative_baselines/baseline_report.json` evaluates M1 UnixBench with five run-level folds, one held-out run per fold, and K=3. Its manifest and report say selection and reconstruction statistics are fit on training runs only, no test timing drives selection, and held-out partial scores are replayed from historical runs.

| Method | Mean relative error | Mean estimated time saved |
|---|---:|---:|
| Correlation reconstruction | 0.7701% | 84.39% |
| Matrix completion, rank 2 | 0.5028% | 84.39% |
| Adaptive probing | 0.0716% | 70.62% |

Adaptive probing starts with a fixed first test. Later selections use only the held-out run's already observed scores and correlation/scale statistics fit on that fold's training runs. The implementation does not use unobserved held-out scores or held-out test durations to select probes. Its selection utility is not time-aware. Reported elapsed-cost savings sum the selected tests' recorded full-run durations, so this is a same-K comparison, not an equal-wall-time comparison; the strongest result should be presented as an accuracy/cost tradeoff and not omitted.

The separate M1 UnixBench control report also has five-run OOF random, fixed, greedy, and router-only rows at K=3. For seed 42, random has 0.2361% mean error / 74.37% saved; fixed-first-K 0.2437% / 84.39%; fixed-CPU mix 0.2583% / 90.62%; and router-only 0.2520% / 76.89%. These remain within-session estimates with only five held-out runs.

On PTS-GPU, the repository has random, fixed, greedy, correlation, and router controls (five runs, one session). Most methods report 0.9181% mean error while estimated time saved ranges from 9.81% to 99.61%, confirming that equal K does not imply equal duration. The repository has no adaptive-probing or matrix-completion PTS-GPU result. These control results do not explain the separate supplied 0.18% M1 GPU number.

## Web-serving benchmark feasibility

The local Phoronix Test Suite lists `pts/cloudsuite-ws` (CloudSuite Web Serving), `pts/nginx`, and `pts/apache`. PTS reports CloudSuite Web Serving as a Docker-based workload using a web server, Memcached, and MySQL, with an estimated 493-second run time, 75-second installation, and 700 MB environment. The local host has Docker 29.7.2, nginx 1.24.0, wrk 4.1.0, 897 GB free disk, and an active nginx service. The PTS CloudSuite profile is not installed.

PTS reports `pts/nginx` as an HTTPS nginx workload driven by wrk, with an estimated 340-second run and 53-second installation. This is the smaller first candidate. A useful method evaluation would still require an explicitly defined set of load/concurrency profiles and multiple full-run repetitions to train and test the reconstructor; a single benchmark invocation would only establish feasibility, not validate BenchScout on a Web workload. Five CloudSuite runs would take at least about 41 minutes at the displayed estimate, plus image setup/pulls and analysis; the profile's reported upstream history also notes a deprecated revision, so its installed v1.1.0 behavior needs validation before relying on it.

No Web workload was run in this audit. The available host and PTS profiles satisfy the initial software/disk prerequisites, but a portable, repeated workload protocol and result parser are not yet present in the repository.
