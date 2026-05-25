"""Best-effort pip installs for MoEBench scripts (--auto-install)."""

from __future__ import annotations

import subprocess
import sys


def pip_install(packages: list[str]) -> None:
    base = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    try:
        subprocess.check_call(base)
    except subprocess.CalledProcessError:
        subprocess.check_call([*base, "--break-system-packages"])


def ensure_importable(module: str, pip_name: str | None = None, *, auto_install: bool) -> None:
    pip_name = pip_name or module
    try:
        __import__(module)
    except ImportError:
        if not auto_install:
            raise
        pip_install([pip_name])
