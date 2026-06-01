#!/usr/bin/env bash
# Avoid triplicate Bosphorus Y4M assets (x264/x265/kvazaar) on small disks.
set -euo pipefail

MARKER="# moebench: bosphorus-symlink-from-x264"
X264_INSTALLED="/var/lib/phoronix-test-suite/installed-tests/pts/x264-2.7.0"

SYMLINK_BLOCK='cd ~
# moebench: bosphorus-symlink-from-x264
SHARED="/var/lib/phoronix-test-suite/installed-tests/pts/x264-2.7.0"
if [ -f "$SHARED/Bosphorus_3840x2160.y4m" ]; then
  ln -sf "$SHARED/Bosphorus_3840x2160.y4m" .
  ln -sf "$SHARED/Bosphorus_1920x1080_120fps_420_8bit_YUV.y4m" .
  rm -f Bosphorus_*.7z 2>/dev/null || true
else
  7z x Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M.7z -aoa
  7z x Bosphorus_1920x1080_120fps_420_8bit_YUV_Y4M.7z -aoa
fi'

link_videos_into() {
  local dir="$1"
  [[ -d "$X264_INSTALLED" ]] || return 0
  [[ -f "$X264_INSTALLED/Bosphorus_3840x2160.y4m" ]] || return 0
  mkdir -p "$dir"
  rm -f "$dir"/Bosphorus_3840x2160.y4m "$dir"/Bosphorus_1920x1080_120fps_420_8bit_YUV.y4m 2>/dev/null || true
  ln -sf "$X264_INSTALLED/Bosphorus_3840x2160.y4m" "$dir/"
  ln -sf "$X264_INSTALLED/Bosphorus_1920x1080_120fps_420_8bit_YUV.y4m" "$dir/"
  find "$dir" -maxdepth 1 -name 'Bosphorus_*.7z' -delete 2>/dev/null || true
}

patch_kvazaar_install() {
  local install_sh="$1"
  [[ -f "$install_sh" ]] || return 0
  if grep -qF "$MARKER" "$install_sh"; then
    echo "Already patched: $install_sh"
    return 0
  fi
  python3 - "$install_sh" "$SYMLINK_BLOCK" <<'PY'
import sys
path, block = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
old = "cd ~\n7z x Bosphorus_3840x2160_120fps_420_8bit_YUV_Y4M.7z -aoa\n7z x Bosphorus_1920x1080_120fps_420_8bit_YUV_Y4M.7z -aoa"
if old not in text:
    print(f"Skip (unexpected layout): {path}", file=sys.stderr)
    sys.exit(0)
open(path, "w", encoding="utf-8").write(text.replace(old, block, 1))
print(f"Patched: {path}")
PY
}

for root in \
  /var/lib/phoronix-test-suite/test-profiles \
  /usr/share/phoronix-test-suite/ob-cache/test-profiles \
  "$(cd "$(dirname "$0")/.." && pwd)/phoronix-test-suite/ob-cache/test-profiles"; do
  patch_kvazaar_install "${root}/pts/kvazaar-1.2.0/install.sh"
done

link_videos_into "/var/lib/phoronix-test-suite/installed-tests/pts/x265-1.5.0"
link_videos_into "/var/lib/phoronix-test-suite/installed-tests/pts/kvazaar-1.2.0"

echo "Bosphorus symlink patch done."
