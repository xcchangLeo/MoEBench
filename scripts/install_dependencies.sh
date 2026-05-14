#!/usr/bin/env bash
set -euo pipefail

# Install runtime/build dependencies for MoEBench + UnixBench.
# Supports: Debian/Ubuntu, RHEL/Fedora, Arch, openSUSE.

WITH_BPFTRACE=1
BUILD_UNIXBENCH=1

usage() {
  cat <<'EOF'
Usage:
  scripts/install_dependencies.sh [options]

Options:
  --no-bpftrace        Do not install bpftrace package
  --no-build           Do not run make in byte-unixbench/UnixBench
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bpftrace) WITH_BPFTRACE=0 ;;
    --no-build) BUILD_UNIXBENCH=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

need_cmd uname
need_cmd grep

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
else
  echo "Cannot detect distro: /etc/os-release not found" >&2
  exit 1
fi

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Please run as root or install sudo." >&2
    exit 1
  fi
fi

install_debian() {
  local perf_kernel_pkg="linux-tools-$(uname -r)"
  local pkgs=(
    python3
    python3-venv
    perl
    gcc
    g++
    make
    libc6-dev
    numactl
    util-linux
    procps
    linux-tools-common
    linux-tools-generic
    "${perf_kernel_pkg}"
  )
  if [[ "${WITH_BPFTRACE}" -eq 1 ]]; then
    pkgs+=(bpftrace)
  fi
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y "${pkgs[@]}" || {
    echo "Kernel-specific perf package unavailable, retrying without ${perf_kernel_pkg}" >&2
    local pkgs_no_kernel=("${pkgs[@]/${perf_kernel_pkg}/}")
    ${SUDO} apt-get install -y "${pkgs_no_kernel[@]}"
  }
}

install_fedora_rhel() {
  local mgr
  if command -v dnf >/dev/null 2>&1; then
    mgr="dnf"
  elif command -v yum >/dev/null 2>&1; then
    mgr="yum"
  else
    echo "Neither dnf nor yum found." >&2
    exit 1
  fi

  local pkgs=(
    python3
    perl
    gcc
    gcc-c++
    make
    glibc-devel
    numactl
    util-linux
    procps-ng
    perf
  )
  if [[ "${WITH_BPFTRACE}" -eq 1 ]]; then
    pkgs+=(bpftrace)
  fi
  ${SUDO} "${mgr}" install -y "${pkgs[@]}"
}

install_arch() {
  local pkgs=(
    python
    perl
    gcc
    make
    glibc
    numactl
    util-linux
    procps-ng
    perf
  )
  if [[ "${WITH_BPFTRACE}" -eq 1 ]]; then
    pkgs+=(bpftrace)
  fi
  ${SUDO} pacman -Sy --needed --noconfirm "${pkgs[@]}"
}

install_opensuse() {
  local pkgs=(
    python3
    perl
    gcc
    gcc-c++
    make
    glibc-devel
    numactl
    util-linux
    procps
    perf
  )
  if [[ "${WITH_BPFTRACE}" -eq 1 ]]; then
    pkgs+=(bpftrace)
  fi
  ${SUDO} zypper --non-interactive install "${pkgs[@]}"
}

echo "Detected distro: ${ID:-unknown} (${PRETTY_NAME:-unknown})"

case "${ID:-}" in
  ubuntu|debian)
    install_debian
    ;;
  fedora|rhel|rocky|almalinux|centos)
    install_fedora_rhel
    ;;
  arch|manjaro)
    install_arch
    ;;
  opensuse*|sles)
    install_opensuse
    ;;
  *)
    echo "Unsupported distro ID: ${ID:-unknown}" >&2
    echo "Install dependencies manually: python3, perl, gcc, g++, make, numactl, util-linux, procps, perf, bpftrace(optional)." >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIXBENCH_DIR="${ROOT_DIR}/byte-unixbench/UnixBench"

if [[ "${BUILD_UNIXBENCH}" -eq 1 ]]; then
  if [[ -f "${UNIXBENCH_DIR}/Makefile" ]]; then
    echo "Building UnixBench binaries..."
    make -C "${UNIXBENCH_DIR}"
  else
    echo "Skip build: ${UNIXBENCH_DIR}/Makefile not found" >&2
  fi
fi

echo "Dependency installation completed."
echo "Tip: verify perf permission via /proc/sys/kernel/perf_event_paranoid"
