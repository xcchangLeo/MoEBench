#!/usr/bin/env bash
# Run offline paper CV for UnixBench + PTS CPU + PTS GPU in one JSON report.
# Adjust globs if your dataset layout differs.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-dataset/paper_cv_three_suites.json}"

python3 scripts/paper_reconstruct_cv_extras.py \
  --dataset-root "${DATASET_ROOT:-dataset}" \
  --suites unixbench,phoronix_cpu,phoronix_gpu \
  --glob-unixbench "${GLOB_UNIXBENCH:-*/run-*.json}" \
  --glob-pts-cpu "${GLOB_PTS_CPU:-aces-*/run-*.json}" \
  --glob-pts-gpu "${GLOB_PTS_GPU:-*pts_nvidia-gpu-compute*/run-*.json}" \
  --cv-mode "${CV_MODE:-leave_one_session_out}" \
  --folds "${FOLDS:-5}" \
  --seed "${SEED:-42}" \
  --eval-partial-k "${EVAL_K:-3}" \
  --policies "${POLICIES:-random,fixed_first_k,fixed_cpu_mix,greedy_slowest,greedy_fastest}" \
  --xi-ablations "${XI_ABLATIONS:-full}" \
  --pts-suite-target "${PTS_SUITE_TARGET:-logmean}" \
  --model-type "${MODEL_TYPE:-lightgbm}" \
  --train-aug "${TRAIN_AUG:-10}" \
  --train-k-min "${TRAIN_K_MIN:-2}" \
  --train-k-max "${TRAIN_K_MAX:-6}" \
  --lgbm-estimators "${LGBM_EST:-200}" \
  ${ROUTER_UNIXBENCH:+--router-model-unixbench "$ROUTER_UNIXBENCH"} \
  ${ROUTER_PTS_CPU:+--router-model-pts-cpu "$ROUTER_PTS_CPU"} \
  ${ROUTER_PTS_GPU:+--router-model-pts-gpu "$ROUTER_PTS_GPU"} \
  --report-json "$OUT"

echo "Wrote $OUT"
