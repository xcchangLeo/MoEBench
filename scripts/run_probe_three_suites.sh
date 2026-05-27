#!/usr/bin/env bash
# Collect + train + experiment for UnixBench, PTS CPU, PTS GPU (per-machine data).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DURATION="${PROBE_DURATION_S:-4}"
MODE="${PROBE_MODE:-micro}"
MAX_RUNS="${MAX_RUNS:-3}"
DS="${DATASET_ROOT:-dataset}"

if [[ -z "${MACHINE:-}" ]]; then
  MACHINE="$(python3 -c 'from moebench.dataset_machines import local_host_slug; print(local_host_slug())')"
fi

EXP_DIR="${EXP_DIR:-$DS/experiments/$MACHINE}"
MODEL_DIR="${MODEL_DIR:-$DS/models/$MACHINE}"
mkdir -p "$EXP_DIR" "$MODEL_DIR"

machine_args=(--machine "$MACHINE")

echo "Machine slug: $MACHINE"
echo "Experiments: $EXP_DIR"
echo "Models:      $MODEL_DIR"

echo "=== UnixBench probe ==="
python3 scripts/probe_collect.py \
  --benchmark unixbench \
  --dataset-root "$DS" \
  --probe-duration-s "$DURATION" \
  --probe-mode "$MODE" \
  --max-runs "$MAX_RUNS" \
  "${machine_args[@]}" \
  -o "$MODEL_DIR/probe_dataset_unixbench.json"

python3 scripts/probe_train.py \
  --probe-dataset "$MODEL_DIR/probe_dataset_unixbench.json" \
  --model-out "$MODEL_DIR/probe_unixbench_lgbm.pkl" \
  --auto-install

python3 scripts/probe_experiment.py \
  --probe-model "$MODEL_DIR/probe_unixbench_lgbm.pkl" \
  --dataset-root "$DS" \
  --probe-mode "$MODE" \
  "${machine_args[@]}" \
  -o "$EXP_DIR/probe_unixbench_lgbm.json"

echo "=== PTS CPU probe ==="
python3 scripts/probe_collect.py \
  --benchmark phoronix \
  --pts-suite cpu \
  --dataset-root "$DS" \
  --probe-duration-s "$DURATION" \
  --probe-mode "$MODE" \
  --max-runs "$MAX_RUNS" \
  "${machine_args[@]}" \
  -o "$MODEL_DIR/probe_dataset_pts_cpu.json"

python3 scripts/probe_train.py \
  --probe-dataset "$MODEL_DIR/probe_dataset_pts_cpu.json" \
  --model-out "$MODEL_DIR/probe_pts_cpu_lgbm.pkl" \
  --auto-install

python3 scripts/probe_experiment.py \
  --probe-model "$MODEL_DIR/probe_pts_cpu_lgbm.pkl" \
  --dataset-root "$DS" \
  --probe-mode "$MODE" \
  "${machine_args[@]}" \
  -o "$EXP_DIR/probe_pts_cpu_lgbm.json"

echo "=== PTS GPU probe ==="
python3 scripts/probe_collect.py \
  --benchmark phoronix \
  --pts-suite pts/nvidia-gpu-compute \
  --dataset-root "$DS" \
  --probe-duration-s "$DURATION" \
  --probe-mode "$MODE" \
  --max-runs "$MAX_RUNS" \
  "${machine_args[@]}" \
  -o "$MODEL_DIR/probe_dataset_pts_gpu.json"

python3 scripts/probe_train.py \
  --probe-dataset "$MODEL_DIR/probe_dataset_pts_gpu.json" \
  --model-out "$MODEL_DIR/probe_pts_gpu_lgbm.pkl" \
  --auto-install

python3 scripts/probe_experiment.py \
  --probe-model "$MODEL_DIR/probe_pts_gpu_lgbm.pkl" \
  --dataset-root "$DS" \
  --probe-mode "$MODE" \
  "${machine_args[@]}" \
  -o "$EXP_DIR/probe_pts_gpu_lgbm.json"

echo "Done. Models under $MODEL_DIR, experiment JSONs under $EXP_DIR"
