#!/usr/bin/env bash
set -Eeuo pipefail

ROUND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROBE_LOADS="20 60 100"
export FULL_LOADS="20 40 60 80 100"
printf 'session=6\nprobe_loads=%s\nfull_loads=%s\n' "${PROBE_LOADS}" "${FULL_LOADS}"
exec "${ROUND_DIR}/run_cloudsuite_supplemental_session.sh" --full 6
