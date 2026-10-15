# Round 10: Router and PTS-GPU Diagnostics

This note records two post-hoc diagnostics used to qualify the paper's claims.
Neither is independent-session validation.

## UnixBench Reconstructor Feature Use

Command:

```bash
python scripts/analyze_unixbench_reconstructor_sensitivity.py \
  --machines aces-System-Product-Name \
    iZbp15n87643uk1sqjrdvdZ iZbp16krl0yc7euw7sb6slZ \
    iZbp1acaw5wdllhz47922rZ iZbp1glgt48i9a8d49embxZ \
  --folds 5 --seed 42 --train-aug 20 --train-k-min 2 --train-k-max 6 \
  --out agent_workflow/rounds/round-10-router-diagnostics/unixbench-probe-importance.json
```

The analysis re-fits the fold-local 200-estimator LightGBM multi-output
reconstructor under the five-host, five-fold UnixBench run-held-out protocol.
Each fold has four unique training runs expanded to 80 augmented rows; this
does not increase the number of independent runs. Feature importance separates
host context from per-component probe mask, index, and duration inputs.

Across 25 suite-output heads, 22 have no probe-feature split; probe-feature
gain is zero in those same 22 heads. Across 300 component-output heads, 13 have
nonzero probe-feature gain. This supports the narrow explanation that probe
features seldom influence these fitted heads under the current training
protocol. It does not establish that probes are generally uninformative, nor
does it establish causality for every identical policy prediction.

## PTS-GPU Static Proxies on the Log-Mean Target

Command:

```bash
python scripts/analyze_pts_gpu_static_logmean_baselines.py \
  --machine aces-System-Product-Name --seed 42 --rf-estimators 200 \
  --out agent_workflow/rounds/round-10-router-diagnostics/pts-gpu-static-logmean-baselines.json
```

The five input runs are from one M1 PTS-GPU collection session:

`dataset/aces-System-Product-Name_pts_nvidia-gpu-compute_20260522T041557Z/`

The proxy targets are recomputed as the log-mean of the ten primary profile
values, matching `agent_workflow/rounds/round-1-oof-pts-gpu-aces-System-Product-Name.json`.
Both analyses use leave-one-run-out folds over the same five runs. The saved
diagnostic includes SHA-256 digests for all five run JSON files and the probe
dataset; its target labels match the hybrid OOF ground-truth labels run by run.
It reports 0.7987% mean error for the Wang-style local DNN proxy, 0.7993% for
the fold-local history mean, and 0.9534% for BenchScout. These are point
estimates from five runs in one session, not evidence of general superiority.
The DNN proxy takes an estimated 3.5 s of new feature collection; BenchScout's
estimated budget is 18.0 s. These costs exclude training and end-to-end serving.

An earlier proxy summary,
`dataset/experiments/exec_free_pts_gpu_aces-System-Product-Name_20260605T054845Z/summary.json`,
uses an arithmetic-mean target. It is not directly comparable with the
log-mean target: its run labels are on a different scale (about 3.4 million
versus about 236). The user-supplied M1/M6 comparison is also kept separate
because its aggregation and timing protocol are unavailable.

## PTS-CPU Static Proxies on the Log-Mean Target

Command:

```bash
python scripts/analyze_pts_cpu_static_logmean_baselines.py \
  --out agent_workflow/rounds/round-10-router-diagnostics/pts-cpu-static-logmean-baselines.json
```

The earlier PTS-CPU proxy results used the default arithmetic-mean target from
`baseline_execution_free.py`, while the hybrid OOF artifacts use `logmean`.
The rerun above recomputes the two proxies shown in the main table (Wang-style
DNN and Tousi-style MLP) using each host's canonical profile intersection,
the exact five-run fold manifest and fold seeds stored in the matching hybrid
OOF artifact, and the same log-mean labels. It verifies run hashes, probe
dataset hashes, and target values against each hybrid OOF record before
fitting. The resulting mean errors (%) are:

| Host | Wang-style DNN | Tousi-style MLP |
| --- | ---: | ---: |
| M1 | 1.91 | 1.88 |
| M2 | 0.26 | 99.42 |
| M3 | 68.53 | 99.93 |
| M4 | 0.22 | 97.57 |
| M5 | 0.17 | 98.77 |

This resolves the target mismatch for the displayed comparison but does not
make the five-run, one-session estimates stable or generalizable. The very
large Tousi-style errors on four hosts remain observations about these local
proxy implementations and this small training sample, not about the cited
methods in general.

## M1 System-Context Ablation Table

The restored Appendix table is transcribed from these artifacts:

- `dataset/paper_supplementary/xi_ablation_unixbench_aces-System-Product-Name.json`
- `dataset/paper_supplementary/xi_ablation_phoronix_cpu_aces-System-Product-Name.json`
- `dataset/paper_supplementary/xi_ablation_phoronix_gpu_aces-System-Product-Name.json`

The reports use LightGBM, fixed router policy with `K=3`, the `full`,
`static_hw_only`, `no_perf_pmu`, `no_dynamic_proc`, and `no_gpu` context
settings, and five run-level folds over the same M1 sessions. The requested
leave-one-session-out mode falls back to random folds because each suite has
one session. PTS targets are `logmean`. This is a router-only reconstruction
control: held-out partial inputs are measured component values from the
archived full run, not scores predicted by the short-probe model. The table
therefore isolates context ablations in the reconstructor path but does not
measure the complete pipeline.

The pattern is not uniformly favorable to retaining all context. Removing
static or dynamic context raises PTS-CPU error by about 0.28 percentage points,
whereas removing GPU features lowers PTS-GPU error by about 0.10 points. Every
UnixBench effect is under 0.01 points. The scope remains five runs from one
session on M1.

## Interpretation Boundary

The matched UnixBench policy result and the GPU proxy comparison both come
from small, run-held-out samples within a single collection session per
host-suite. They qualify the reported results and do not resolve the need for
independent-session or cold-start configuration-transfer evaluation. A
CloudSuite cold-start study, if completed, should be reported as a separate
scenario from same-configuration history-based warm-start results.
