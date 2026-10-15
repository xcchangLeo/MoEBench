# M1 External-Session UnixBench Diagnostic

## Protocol

On 2026-10-12, collected one complete UnixBench 6.0.0 run on M1 using the repository collector. The raw record is `dataset/m1_20261012_session/run-01.json`; it contains all 12 component scores and the full-suite index. This session was collected after the archived five-run M1 corpus and was stored separately from the training data.

We then ran `scripts/experiment_router_probe_reconstruct.py` in online mode against that full-run record. Router, reconstructor, and probe checkpoints were created on 2026-06-10 or earlier and cannot contain the October record. The online evaluation collected fresh context, selected three subtests, measured each with a four-second real probe, predicted the partial scores, and reconstructed the suite index. It used `--no-ebpf`; model selection was not retuned on the October target.

Reproduction command for the online stage:

```sh
MOEBENCH_ML_BOOTSTRAPPED=1 python scripts/experiment_router_probe_reconstruct.py \
  --router-model dataset/experiments/hybrid_grid_unixbench_aces-System-Product-Name_20260610T115239Z/trained_models/router_lgbm.pkl \
  --reconstruct-model dataset/experiments/hybrid_grid_unixbench_aces-System-Product-Name_20260610T115239Z/trained_models/recon_lgbm.pkl \
  --probe-model dataset/models/aces-System-Product-Name/probe_unixbench_lgbm.pkl \
  --dataset-root agent_workflow/rounds/round-7-m1-session/dataset \
  --machine aces-System-Product-Name \
  --glob-pattern 'm1_20261012_session/run-*.json' \
  --top-k 3 --probe-duration-s 4 --probe-mode real --online --no-ebpf \
  --output agent_workflow/rounds/round-7-m1-session/hybrid_online.json
```

## Result

The complete-run target was 3418.5 and the pipeline predicted 3189.654, for 6.6943% relative error. Router scores for all 12 components were identical, so the Top-3 selection fell back to the first three listed components (`dhry2reg`, `whetstone-double`, and `execl`); this run does not demonstrate learned routing. Fresh context collection took 8.699 s and the three real probes took 3.012 s, for 11.711 s combined online collection and inference-path time.

This is one same-host, one-session temporal test. It provides no estimate of error variability and is not a Web-serving result. It indicates that the historical M1 run-held-out accuracy does not transfer to this new full UnixBench observation under the saved checkpoints. The script's `full_suite_from_dataset` value (319.9 s) sums per-component durations recorded in the UnixBench report; it is not a measured end-to-end wall time and must not be used to claim a speedup.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `dataset/m1_20261012_session/run-01.json` | `ae62d8bbb3a3be966321da3e35e3d6edbf9921d7a2bf73da2d34d6f6ffa17092` |
| `hybrid_online.json` | `58d65c891e7e27f1c747c6347ec5b1e4b9f2235e620c9325a60d571c127a7bc3` |
| Router checkpoint | `e90a641c7585450fe11804e6cfd051f52224843c80aecaedcc73b94786ced39f` |
| Reconstructor checkpoint | `4439d0814f7fea0f3e04e9b813ed46808565b7fb87832e9e27cad8b72ba4016b` |
| Probe checkpoint | `e3306271c36ed3e47a27f0a50e5924ac43d7da990e543fa1bcefd89d5f10b658` |

The checkpoints use scikit-learn objects saved under version 1.8.0 and were loaded under 1.7.2, which emitted compatibility warnings. This adds uncertainty to the external diagnostic. It is useful as a reproducible failure case, not as a definitive estimate of cross-session performance.
