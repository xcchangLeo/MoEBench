#!/usr/bin/env bash
# Wrapper: train multiple routers + run full MoE experiment per model (see README).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_router_model_ablation.py "$@"
