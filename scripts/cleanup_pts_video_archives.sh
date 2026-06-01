#!/usr/bin/env bash
# Remove Bosphorus .7z archives after .y4m exists (or symlink) under each PTS install dir.
set -euo pipefail

removed=0
while IFS= read -r -d '' z; do
  dir="$(dirname "$z")"
  base="$(basename "$z" .7z)"
  y4m=""
  case "$base" in
    Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M) y4m="$dir/Bosphorus_3840x2160.y4m" ;;
    Bosphorus_1920x1080_120fps_420_8bit_YUV_Y4M) y4m="$dir/Bosphorus_1920x1080_120fps_420_8bit_YUV.y4m" ;;
    *) continue ;;
  esac
  if [[ -e "$y4m" ]]; then
    echo "rm $z"
    rm -f "$z"
    removed=$((removed + 1))
  fi
done < <(find /var/lib/phoronix-test-suite/installed-tests -name 'Bosphorus_*.7z' -print0 2>/dev/null)

echo "Removed ${removed} archive(s)."
df -h /
