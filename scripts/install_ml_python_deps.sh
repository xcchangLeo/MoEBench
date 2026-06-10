#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install_ml_python_deps.sh [options]

Options:
  --no-torch          Skip torch install
  --no-lightgbm       Skip lightgbm install
  --use-venv          Force install into local venv (ignores conda detection)
  -h, --help
EOF
}

WITH_TORCH=1
WITH_LIGHTGBM=1
USE_VENV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-torch) WITH_TORCH=0 ;;
    --no-lightgbm) WITH_LIGHTGBM=0 ;;
    --use-venv) USE_VENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

need_cmd python3

ensure_venv_capable() {
  if python3 -c "import ensurepip" >/dev/null 2>&1; then
    return 0
  fi
  local pyver
  pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo "Python's ensurepip is not available; \`python3 -m venv\` will fail." >&2
  echo "On Debian/Ubuntu install the venv package, for example:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y python3-venv" >&2
  echo "If that does not match your interpreter, try:" >&2
  echo "  sudo apt-get install -y python${pyver}-venv" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-moebench-ml}"

if [[ "${USE_VENV}" -eq 0 && -n "${CONDA_PREFIX:-}" ]]; then
  echo "Conda detected: ${CONDA_PREFIX}"
  echo "Installing into active conda env via: $(command -v python3)"
  python3 -m pip install --upgrade pip setuptools wheel
  pkgs=(numpy)
  if [[ "${WITH_LIGHTGBM}" -eq 1 ]]; then
    pkgs+=(lightgbm)
  fi
  if [[ "${WITH_TORCH}" -eq 1 ]]; then
    pkgs+=(torch)
  fi
  pkgs+=(scikit-learn matplotlib)
  python3 -m pip install --upgrade "${pkgs[@]}"
  echo "Done."
  echo "Next, run with: $(command -v python3)"
  return 0 2>/dev/null || exit 0
fi

if [[ -d "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" && ! -x "${VENV_DIR}/bin/python3" ]]; then
  echo "Removing incomplete venv at ${VENV_DIR}" >&2
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  ensure_venv_capable
  echo "Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "Installing into venv: ${VENV_DIR}"
echo "Upgrading pip in venv..."
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel

pkgs=(numpy)
if [[ "${WITH_LIGHTGBM}" -eq 1 ]]; then
  pkgs+=(lightgbm)
fi
if [[ "${WITH_TORCH}" -eq 1 ]]; then
  pkgs+=(torch)
fi
pkgs+=(scikit-learn matplotlib)

echo "Installing into venv: ${pkgs[*]}"
"${VENV_PIP}" install --upgrade "${pkgs[@]}"

echo "Done."
echo "Next, run with: ${VENV_DIR}/bin/python3"

