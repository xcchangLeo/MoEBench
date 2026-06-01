#!/usr/bin/env bash
# Patch Phoronix Test Suite pts/x265-1.5.0 for CMake 4.x (CMP0025/CMP0054 OLD removed;
# cmake_minimum_required < 3.5 no longer supported).
set -euo pipefail

MARKER="# moebench: cmake4-x265-patch"

patch_one() {
  local install_sh="$1"
  [[ -f "$install_sh" ]] || return 0
  if grep -qF "$MARKER" "$install_sh"; then
    echo "Already patched: $install_sh"
    return 0
  fi
  if ! grep -q 'tar -xf x265_4.1.tar.gz' "$install_sh"; then
    echo "Skip (unexpected layout): $install_sh" >&2
    return 0
  fi
  local tmp
  tmp="$(mktemp)"
  awk -v marker="$MARKER" '
    { print }
    /tar -xf x265_4\.1\.tar\.gz/ && !done {
      print marker
      print "sed -i \\"
      print "  -e \"s/cmake_policy(SET CMP0025 OLD)/cmake_policy(SET CMP0025 NEW)/\" \\"
      print "  -e \"s/cmake_policy(SET CMP0054 OLD)/cmake_policy(SET CMP0054 NEW)/\" \\"
      print "  -e \"s/cmake_minimum_required (VERSION 2.8.8)/cmake_minimum_required(VERSION 3.5)/\" \\"
      print "  x265_4.1/source/CMakeLists.txt"
      done = 1
    }
  ' "$install_sh" > "$tmp"
  chmod --reference="$install_sh" "$tmp" 2>/dev/null || chmod 755 "$tmp"
  mv "$tmp" "$install_sh"
  echo "Patched: $install_sh"
}

ROOTS=(
  /var/lib/phoronix-test-suite/test-profiles
  /usr/share/phoronix-test-suite/ob-cache/test-profiles
)
if [[ -n "${MOEBENCH_ROOT:-}" ]]; then
  ROOTS+=("${MOEBENCH_ROOT}/phoronix-test-suite/ob-cache/test-profiles")
elif [[ -f "$(dirname "$0")/../phoronix-test-suite/phoronix-test-suite" ]]; then
  ROOTS+=("$(cd "$(dirname "$0")/.." && pwd)/phoronix-test-suite/ob-cache/test-profiles")
fi

for root in "${ROOTS[@]}"; do
  patch_one "${root}/pts/x265-1.5.0/install.sh"
done

echo "x265 CMake patch applied. Re-run: phoronix-test-suite install pts/x265-1.5.0"
