#!/usr/bin/env bash
# Paper supplementary (2/3): compare subset-selection / routing policies at fixed K.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-dataset/paper_supplementary/policy_compare_three_suites.json}"
SUMMARY="${SUMMARY:-dataset/paper_supplementary/policy_compare_three_suites_summary.json}"
EVAL_K="${EVAL_K:-3}"
POLICIES="${POLICIES:-random,fixed_first_k,fixed_cpu_mix,fixed_io_mix,greedy_slowest,greedy_fastest,router}"
MODEL_TYPE="${MODEL_TYPE:-lightgbm}"

if [[ -z "${ROUTER_UNIXBENCH:-}" || -z "${ROUTER_PTS_CPU:-}" || -z "${ROUTER_PTS_GPU:-}" ]]; then
  echo "Set ROUTER_UNIXBENCH, ROUTER_PTS_CPU, ROUTER_PTS_GPU (needed when POLICIES includes router)." >&2
  exit 1
fi

python3 scripts/paper_reconstruct_cv_extras.py \
  --dataset-root "${DATASET_ROOT:-dataset}" \
  --suites unixbench,phoronix_cpu,phoronix_gpu \
  --glob-unixbench "${GLOB_UNIXBENCH:-*/run-*.json}" \
  --glob-pts-cpu "${GLOB_PTS_CPU:-*_cpu_*/run-*.json}" \
  --glob-pts-gpu "${GLOB_PTS_GPU:-*_pts_nvidia-gpu-compute_*/run-*.json}" \
  --cv-mode "${CV_MODE:-leave_one_session_out}" \
  --folds "${FOLDS:-5}" \
  --seed "${SEED:-42}" \
  --eval-partial-k "$EVAL_K" \
  --policies "$POLICIES" \
  --xi-ablations full \
  --model-type "$MODEL_TYPE" \
  --log1p-partial-value \
  --train-aug "${TRAIN_AUG:-10}" \
  --train-k-max "${TRAIN_K_MAX:-12}" \
  --router-model-unixbench "$ROUTER_UNIXBENCH" \
  --router-model-pts-cpu "$ROUTER_PTS_CPU" \
  --router-model-pts-gpu "$ROUTER_PTS_GPU" \
  --report-json "$OUT" \
  ${AUTO_INSTALL:+--auto-install}

python3 scripts/paper_summarize_supplementary.py --mode policy --input "$OUT" -o "$SUMMARY"
echo "Wrote $OUT and $SUMMARY"
