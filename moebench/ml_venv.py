"""Re-exec MoEBench scripts with project ML venv when system Python lacks deps."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_VENV_NAMES = (".venv-moebench-ml", ".venv-moebench-router")
ML_VENV_PY = REPO_ROOT / ML_VENV_NAMES[0] / "bin" / "python3"
INSTALL_ML_DEPS = REPO_ROOT / "scripts" / "install_ml_python_deps.sh"


def import_ok(module: str, py: str | None = None) -> bool:
    interpreter = py or sys.executable
    return subprocess.run([interpreter, "-c", f"import {module}"], capture_output=True).returncode == 0


def _home_for_ml_search() -> Path | None:
    for key in ("SUDO_USER", "USER"):
        user = os.environ.get(key, "").strip()
        if user and user != "root":
            home = Path(f"/home/{user}")
            if home.is_dir():
                return home
    home = Path.home()
    return home if home.is_dir() else None


def _ml_interpreter_candidates() -> list[Path]:
    """Interpreters to try when ``sys.executable`` lacks numpy/sklearn/lightgbm."""
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        key = str(p.resolve()) if p.is_file() else str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        add(Path(conda_prefix) / "bin" / "python3")

    home = _home_for_ml_search()
    if home is not None:
        for rel in (
            "dev/miniconda3/bin/python3",
            "miniconda3/bin/python3",
            "anaconda3/bin/python3",
        ):
            add(home / rel)

    explicit = os.environ.get("MOEBENCH_ML_PYTHON", "").strip()
    if explicit:
        add(Path(explicit))

    for name in ML_VENV_NAMES:
        add(REPO_ROOT / name / "bin" / "python3")
    return out


def _pick_working_interpreter(need_modules: list[str]) -> Path | None:
    for candidate in _ml_interpreter_candidates():
        if not candidate.is_file():
            continue
        py = str(candidate)
        if all(import_ok(m, py) for m in need_modules):
            return candidate
    return None


def ensure_ml_interpreter(*, need_modules: list[str], auto_install: bool, label: str = "moebench") -> None:
    """Switch to conda/project venv before importing ML packages (avoids PEP 668 on system Python)."""
    if os.environ.get("MOEBENCH_ML_BOOTSTRAPPED") == "1" or not need_modules:
        return
    missing = [m for m in need_modules if not import_ok(m)]
    if not missing:
        return

    alt = _pick_working_interpreter(need_modules)
    if alt is not None:
        py = str(alt)
        print(f"[{label}] re-exec with ML interpreter: {py}", file=sys.stderr)
        os.environ["MOEBENCH_ML_BOOTSTRAPPED"] = "1"
        os.execv(py, [py, *sys.argv])

    if auto_install and INSTALL_ML_DEPS.is_file():
        install_args = ["bash", str(INSTALL_ML_DEPS), "--no-torch"]
        if not os.environ.get("CONDA_PREFIX"):
            install_args.append("--use-venv")
        print(f"[{label}] bootstrapping ML deps: {' '.join(install_args)}", file=sys.stderr)
        subprocess.check_call(install_args)
        alt = _pick_working_interpreter(need_modules)
        if alt is not None:
            py = str(alt)
            os.environ["MOEBENCH_ML_BOOTSTRAPPED"] = "1"
            os.execv(py, [py, *sys.argv])

    sudo_hint = ""
    if os.geteuid() == 0:
        sudo_hint = (
            "\n\nYou ran under sudo with system Python (/usr/bin/python3). Prefer:\n"
            "  conda activate <env>\n"
            "  python scripts/... --sudo\n"
            "or:\n"
            "  sudo -E /path/to/miniconda3/bin/python3 scripts/...\n"
            "Do not use bare `sudo python3` — it ignores your conda env."
        )

    raise SystemExit(
        "Missing Python modules: "
        + ", ".join(missing)
        + ".\nRun:\n  bash scripts/install_ml_python_deps.sh --no-torch --use-venv\nThen:\n  "
        + f"{ML_VENV_PY} {' '.join(sys.argv)}"
        + sudo_hint
    )
