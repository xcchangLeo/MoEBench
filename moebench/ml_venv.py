"""Re-exec MoEBench scripts with project ML venv when system Python lacks deps."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_VENV_PY = REPO_ROOT / ".venv-moebench-router" / "bin" / "python3"
INSTALL_ML_DEPS = REPO_ROOT / "scripts" / "install_ml_python_deps.sh"


def import_ok(module: str, py: str | None = None) -> bool:
    interpreter = py or sys.executable
    return subprocess.run([interpreter, "-c", f"import {module}"], capture_output=True).returncode == 0


def ensure_ml_interpreter(*, need_modules: list[str], auto_install: bool, label: str = "moebench") -> None:
    """Switch to project venv before importing ML packages (avoids PEP 668 on system Python)."""
    if os.environ.get("MOEBENCH_ML_BOOTSTRAPPED") == "1" or not need_modules:
        return
    missing = [m for m in need_modules if not import_ok(m)]
    if not missing:
        return

    venv_py = str(ML_VENV_PY)
    if ML_VENV_PY.is_file() and import_ok("numpy", venv_py):
        print(f"[{label}] re-exec with project venv: {venv_py}", file=sys.stderr)
        os.environ["MOEBENCH_ML_BOOTSTRAPPED"] = "1"
        os.execv(venv_py, [venv_py, *sys.argv])

    if auto_install and INSTALL_ML_DEPS.is_file():
        install_args = ["bash", str(INSTALL_ML_DEPS), "--no-torch"]
        if not os.environ.get("CONDA_PREFIX"):
            install_args.append("--use-venv")
        print(f"[{label}] bootstrapping ML deps: {' '.join(install_args)}", file=sys.stderr)
        subprocess.check_call(install_args)
        if ML_VENV_PY.is_file() and import_ok("numpy", venv_py):
            os.environ["MOEBENCH_ML_BOOTSTRAPPED"] = "1"
            os.execv(venv_py, [venv_py, *sys.argv])

    raise SystemExit(
        "Missing Python modules: "
        + ", ".join(missing)
        + ".\nRun:\n  bash scripts/install_ml_python_deps.sh --no-torch --use-venv\nThen:\n  "
        + f"{ML_VENV_PY} {' '.join(sys.argv)}"
    )
