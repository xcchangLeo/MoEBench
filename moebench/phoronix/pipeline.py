"""Run collect_all + Phoronix Test Suite, emit dataset JSON (xi, yi, ti)."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moebench.collector import collect_all
from moebench.phoronix.parse import build_experts_from_pts_json, extract_ti_from_pts_json


def moebench_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_root() -> Path:
    return moebench_repo_root() / "dataset"


def default_pts_install_root() -> Path:
    """Typical clone path: ``<MoEBench>/phoronix-test-suite/phoronix-test-suite``."""
    return moebench_repo_root() / "phoronix-test-suite"


def host_slug() -> str:
    try:
        h = socket.gethostname()
    except OSError:
        h = "unknown-host"
    h = h.strip() or "unknown-host"
    safe = re.sub(r"[^\w.\-]+", "_", h, flags=re.UNICODE)
    safe = re.sub(r"_+", "_", safe).strip("._-")[:64]
    return safe or "host"


def safe_session_tag(tag: str) -> str:
    t = tag.strip() or "session"
    t = re.sub(r"[^\w.\-]+", "_", t, flags=re.UNICODE)
    t = re.sub(r"_+", "_", t).strip("._-")
    return (t[:120] if t else "session")


# Characters allowed in pts_test_run_manager::clean_save_name keep_in_string mask
# (LETTER | NUMERIC | CHAR_DASH only — underscores are stripped).
_PTS_SAVE_NAME_ALLOWED = frozenset(string.ascii_letters + string.digits + "-")


def pts_clean_save_name(name: str) -> str:
    """
    Mirror ``pts_test_run_manager::clean_save_name($name, is_new_save=true)``.

    PTS writes results under ``~/.phoronix-test-suite/test-results/<this-id>/``.
    On Linux the path is case-sensitive; ``result-file-to-json`` must use the same
    string PTS derived after cleaning (lowercase, no underscores, etc.).
    Skips PHP ``swap_variables`` (MoEBench names do not use PTS template variables).
    """
    s = name.strip().replace(" ", "-")
    s = "".join(c for c in s if c in _PTS_SAVE_NAME_ALLOWED)
    out: list[str] = []
    prev: str | None = None
    for c in s:
        if c == "-" and prev == "-":
            continue
        out.append(c)
        prev = c
    s = "".join(out).lower()
    if len(s) > 126:
        s = s[:126]
    if not s:
        s = f"moebench-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return s


def default_session_tag(pts_suite: str | None = None) -> str:
    """
    Default dataset folder name under ``dataset/<session>/``.

    When ``pts_suite`` is set (e.g. ``pts/nvidia-gpu-compute``), it is embedded in the
    tag so CPU / GPU / other PTS runs are easy to tell apart without ``--session``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = host_slug()
    if pts_suite:
        token = safe_session_tag(pts_suite.replace("/", "_"))
        return f"{host}_{token}_{stamp}"
    return f"{host}_{stamp}"


# Single fast PTS profile for ``--pts-smoke`` (install once: ``phoronix-test-suite install pts/ctx-clock``).
DEFAULT_PTS_SMOKE_SUITE = "pts/ctx-clock"


def _which_pts(pts_bin: str | None, pts_root: Path | None) -> str:
    if pts_bin:
        p = Path(pts_bin)
        if p.is_file():
            return str(p.resolve())
        return pts_bin
    root = Path(pts_root) if pts_root else default_pts_install_root()
    cand = root / "phoronix-test-suite"
    if cand.is_file():
        return str(cand.resolve())
    return "phoronix-test-suite"


_BATCH_MODE_NOT_CONFIGURED = re.compile(
    r"batch mode must first be configured",
    re.IGNORECASE,
)


def _pts_argv_as_installing_user(pts_exe: str, pts_subargs: list[str]) -> list[str]:
    """
    If MoEBench is running as root after ``sudo``, run PTS as ``SUDO_USER`` so it uses
    that user's ``~/.phoronix-test-suite``.  Pair with :func:`pts_subprocess_env` so
    ``HOME`` in the subprocess environment is not left as ``/root`` (PTS consults
    both the real uid and ``$HOME`` when resolving install paths).
    """
    if os.geteuid() != 0:
        return [pts_exe, *pts_subargs]
    su = os.environ.get("SUDO_USER")
    if not su:
        return [pts_exe, *pts_subargs]
    try:
        import pwd

        pwd.getpwnam(su)
    except (ImportError, KeyError):
        return [pts_exe, *pts_subargs]
    return ["sudo", "-u", su, "-E", pts_exe, *pts_subargs]


def _sudo_target_home() -> str | None:
    if os.geteuid() != 0:
        return None
    su = os.environ.get("SUDO_USER")
    if not su:
        return None
    try:
        import pwd

        return pwd.getpwnam(su).pw_dir
    except (ImportError, KeyError):
        return None


def pts_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """
    Environment for PTS subprocesses when MoEBench runs as root after ``sudo``.

    Passing ``env=...`` to ``subprocess`` replaces the whole environment. Root
    sessions often retain ``HOME=/root``, so Phoronix looks for tests under
    ``~root/.phoronix-test-suite`` and reports profiles as not installed even when
    ``sudo -u <SUDO_USER>`` is used.  Copy the base env and set ``HOME`` / ``USER`` /
    ``LOGNAME`` to the invoking login user (``SUDO_USER``) so PTS matches a normal
    ``phoronix-test-suite install`` as your user.
    """
    env = dict(base if base is not None else os.environ)
    if os.geteuid() != 0:
        return env
    su = os.environ.get("SUDO_USER")
    if not su:
        return env
    try:
        import pwd

        pw = pwd.getpwnam(su)
    except (ImportError, KeyError):
        return env
    env["HOME"] = pw.pw_dir
    env["USER"] = su
    env["LOGNAME"] = su
    return env


def _mkdir_chown_output_dir_for_sudo_user(dir_path: Path) -> None:
    """
    MoEBench often runs as root via ``sudo`` while ``result-file-to-json`` runs as
    ``SUDO_USER``. Directories created by root (``mkdir``) are not writable by that
    user, so ``file_put_contents`` to ``dataset/.../run-01_pts_raw.json`` fails with
    permission denied unless we hand ownership to the invoking user.
    """
    d = Path(dir_path).resolve()
    d.mkdir(parents=True, exist_ok=True)
    if os.geteuid() != 0:
        return
    su = os.environ.get("SUDO_USER")
    if not su:
        return
    try:
        import pwd

        pw = pwd.getpwnam(su)
    except KeyError:
        return
    try:
        os.chown(d, pw.pw_uid, pw.pw_gid, follow_symlinks=False)
    except OSError as exc:
        print(f"Warning: could not chown {d} to {su}: {exc}", file=sys.stderr)


def _pts_argv_result_file_to_json(
    pts_exe: str,
    save_id: str,
    *,
    output_dir: str,
    output_file: str,
) -> list[str]:
    """
    ``result-file-to-json`` must see ``OUTPUT_FILE`` / ``OUTPUT_DIR``. Many sudoers
    use ``env_reset`` so ``sudo -u user -E`` drops custom variables; inline ``env``
    assignments survive into the PTS PHP process.
    """
    save_id = save_id.strip()
    if os.geteuid() != 0:
        return [pts_exe, "result-file-to-json", save_id]
    su = os.environ.get("SUDO_USER")
    if not su:
        return [pts_exe, "result-file-to-json", save_id]
    try:
        import pwd

        pw = pwd.getpwnam(su)
    except (ImportError, KeyError):
        return [pts_exe, "result-file-to-json", save_id]
    env_bin = shutil.which("env") or "/usr/bin/env"
    path = os.environ.get("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
    assigns = [
        f"HOME={pw.pw_dir}",
        f"OUTPUT_DIR={output_dir}",
        f"OUTPUT_FILE={output_file}",
        f"PATH={path}",
    ]
    return ["sudo", "-u", su, env_bin, *assigns, pts_exe, "result-file-to-json", save_id]


def _run_pts_command_streaming(cmd: list[str], env: dict[str, str]) -> tuple[int, bool]:
    """
    Run PTS with merged stdout/stderr streamed to inherited stdout (live progress).
    Returns (returncode, batch_mode_not_configured) where the latter is True if PTS
    printed the standard batch-setup notice (batch-run exits 0 but runs no tests).
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    batch_notice = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if _BATCH_MODE_NOT_CONFIGURED.search(line):
                batch_notice = True
    finally:
        if proc.stdout:
            proc.stdout.close()
    rc = proc.wait()
    return rc, batch_notice


def _export_result_json(pts_exe: str, result_file_name: str, out_path: Path) -> dict[str, Any]:
    """Call ``result-file-to-json``; prefer OUTPUT_DIR/OUTPUT_FILE, else stdout JSON."""
    # PTS ``save_output_handler`` treats OUTPUT_FILE as the full path when set; a bare
    # basename is written under the process CWD, not OUTPUT_DIR (see pts_client.php).
    out_path = Path(out_path).resolve()
    _mkdir_chown_output_dir_for_sudo_user(out_path.parent)
    save_id = result_file_name.strip()
    od = str(out_path.parent)
    of = str(out_path)
    cmd = _pts_argv_result_file_to_json(pts_exe, save_id, output_dir=od, output_file=of)
    env = pts_subprocess_env()
    env["OUTPUT_DIR"] = od
    env["OUTPUT_FILE"] = of
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    last_rc = p.returncode
    last_stdout = p.stdout or ""
    last_stderr = p.stderr or ""

    def _try_read_json_file(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    parsed = _try_read_json_file(out_path)
    if parsed is not None:
        return parsed
    stray = Path.cwd() / out_path.name
    if stray.is_file() and stray.resolve() != out_path:
        parsed = _try_read_json_file(stray)
        if parsed is not None:
            try:
                stray.replace(out_path)
            except OSError:
                pass
            return parsed
    home = _sudo_target_home() or str(Path.home())
    legacy_home_json = Path(home) / f"{save_id}.json"
    parsed = _try_read_json_file(legacy_home_json)
    if parsed is not None:
        try:
            legacy_home_json.replace(out_path)
        except OSError:
            pass
        return parsed

    out = last_stdout.strip().lstrip("\ufeff")
    if p.returncode == 0 and out.startswith("{"):
        return json.loads(out)
    if p.returncode == 0 and "{" in out:
        i = out.find("{")
        try:
            return json.loads(out[i:])
        except json.JSONDecodeError:
            pass

    err = last_stderr[:4000]
    hint = (
        " If you used batch-run, configure once: phoronix-test-suite batch-setup "
        "(sets PhoronixTestSuite/Options/BatchMode/Configured). "
        "Or use pts_mode=run / --pts-mode run instead of batch-run."
    )
    preview = (last_stdout or "")[:2000]
    raise RuntimeError(
        f"Could not read PTS JSON for result file {save_id!r} (rc={last_rc}). "
        f"stderr={err!r}. stdout_preview={preview!r}.{hint}"
    )


def run_pts_dataset(
    *,
    pts_bin: str | None = None,
    pts_root: Path | None = None,
    suite: str = "cpu",
    pts_mode: str = "batch-run",
    pts_extra_args: list[str] | None = None,
    output_json: Path | str | None = None,
    collect_features: bool = True,
    xi_override: dict[str, Any] | None = None,
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    mem_mb: int = 64,
    enable_ebpf: bool = True,
    session_tag: str | None = None,
    round_index: int | None = None,
    total_rounds: int | None = None,
    result_name: str | None = None,
    pts_inherit_stdio: bool | None = None,
) -> dict[str, Any]:
    """
    1) Collect xi via ``collect_all`` unless ``xi_override`` is set.
    2) Run ``phoronix-test-suite <pts_mode> <suite> ...`` (inherited stdio when
       ``pts_inherit_stdio`` is True; otherwise capture+replay for batch-run notices).
    3) Export ``result-file-to-json`` using the **PTS save identifier** (after
       ``clean_save_name``: lowercase, no underscores — same as on-disk folder name).

    Use ``pts_mode=batch-run`` for non-interactive defaults; use ``run`` only if batch-setup allows it.

    When the process is root after ``sudo``, PTS is executed as ``SUDO_USER`` so tests
    installed under the normal user's Phoronix directory are visible.
    """
    pts_exe = _which_pts(pts_bin, pts_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proposed = result_name or f"moebench_pts_{host_slug()}_{stamp}"
    proposed = safe_session_tag(proposed)
    result_file_name = pts_clean_save_name(proposed)

    xi: dict[str, Any] | None = None
    if xi_override is not None:
        xi = xi_override
    elif collect_features:
        xi = collect_all(
            warmup_s=warmup_s,
            proc_sample_s=proc_sample_s,
            enable_ebpf=enable_ebpf,
            mem_mb=mem_mb,
        )

    env = pts_subprocess_env()
    env["TEST_RESULTS_NAME"] = result_file_name
    env["TEST_RESULTS_IDENTIFIER"] = result_file_name
    desc = f"MoEBench PTS {suite} ({pts_mode})"
    if round_index is not None and total_rounds is not None:
        desc += f" round {round_index}/{total_rounds}"
    env["TEST_RESULTS_DESCRIPTION"] = desc

    subargs = [pts_mode, suite]
    if pts_extra_args:
        subargs.extend(pts_extra_args)
    cmd = _pts_argv_as_installing_user(pts_exe, subargs)

    inherit = pts_inherit_stdio if pts_inherit_stdio is not None else pts_mode == "run"
    if os.geteuid() == 0 and not os.environ.get("SUDO_USER"):
        print(
            "Warning: running as root without SUDO_USER; Phoronix uses root's test install "
            "tree (~root/.phoronix-test-suite). Installs under your login user are ignored.",
            file=sys.stderr,
        )

    print("+", " ".join(cmd), file=sys.stderr)
    if inherit:
        rc = subprocess.call(cmd, env=env)
        batch_not_configured = False
    else:
        rc, batch_not_configured = _run_pts_command_streaming(cmd, env)
    if batch_not_configured:
        raise RuntimeError(
            "Phoronix batch-run did not execute any benchmarks: batch mode is not configured.\n"
            "Run once (interactive): phoronix-test-suite batch-setup\n"
            "Then re-run MoEBench, or use non-batch mode: --pts-mode run"
        )
    if rc != 0:
        raise RuntimeError(f"Phoronix Test Suite exited with code {rc}")

    if output_json:
        op = Path(output_json)
        export_path = op.with_name(f"{op.stem}_pts_raw.json")
    else:
        export_path = moebench_repo_root() / "dataset" / ".pts_cache" / f"{result_file_name}.json"
    pts_export = _export_result_json(pts_exe, result_file_name, export_path)

    yi: dict[str, Any] = {
        "suite": suite,
        "pts_mode": pts_mode,
        "pts_export": pts_export,
    }
    ti = extract_ti_from_pts_json(pts_export)
    experts = build_experts_from_pts_json(pts_export, default_suite=suite)

    dataset: dict[str, Any] = {
        "schema": "moebench.phoronix.dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "xi": xi,
        "yi": yi,
        "ti": ti,
        "experts": experts,
        "session": {
            "tag": session_tag,
            "round_index": round_index,
            "total_rounds": total_rounds,
            "xi_reused_from_previous_round": xi_override is not None,
        },
        "phoronix": {
            "pts_executable": pts_exe,
            "command": cmd,
            "pts_inherit_stdio": inherit,
            "pts_run_via_sudo_user": os.geteuid() == 0 and bool(os.environ.get("SUDO_USER")),
            "returncode": rc,
            "result_file_name": result_file_name,
            "pts_name_before_clean_save": proposed,
            "env": {
                "TEST_RESULTS_NAME": result_file_name,
                "TEST_RESULTS_IDENTIFIER": result_file_name,
            },
            "raw_json_export_path": str(export_path),
        },
        "notes": {
            "D": "yi.pts_export is PTS result-file-to-json; ti.by_test_id keyed by PTS profile identifier.",
        },
    }

    if output_json:
        out_path = Path(output_json)
        _mkdir_chown_output_dir_for_sudo_user(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out_path}", file=sys.stderr)

    return dataset


def run_pts_batch(
    *,
    num_rounds: int,
    dataset_root: Path | str | None = None,
    session_tag: str | None = None,
    reuse_xi: bool = False,
    pts_bin: str | None = None,
    pts_root: Path | None = None,
    suite: str = "cpu",
    pts_mode: str = "batch-run",
    pts_extra_args: list[str] | None = None,
    collect_features: bool = True,
    warmup_s: float = 3.0,
    proc_sample_s: float = 0.5,
    mem_mb: int = 64,
    enable_ebpf: bool = True,
    pts_inherit_stdio: bool | None = None,
) -> dict[str, Any]:
    """
    Run ``num_rounds`` PTS pipelines, writing:

        ``{dataset_root}/{session_tag}/run-01.json`` … ``run-NN.json``

    and ``manifest.json``. Each round uses a distinct ``TEST_RESULTS_NAME`` so PTS
    does not overwrite. By default every round collects **xi**; set ``reuse_xi=True``
    to reuse xi from round 1 on rounds 2..N.
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    root_ds = Path(dataset_root) if dataset_root else default_dataset_root()
    root_ds = root_ds.resolve()
    tag = safe_session_tag(session_tag or default_session_tag(suite))
    session_dir = root_ds / tag
    session_dir.mkdir(parents=True, exist_ok=True)

    xi_cached: dict[str, Any] | None = None
    written: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()

    for i in range(1, num_rounds + 1):
        use_xi: dict[str, Any] | None = None
        if reuse_xi and i > 1:
            use_xi = xi_cached

        micro = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        result_name = f"moebench_pts_{tag}_r{i:02d}_{micro}"
        result_name = safe_session_tag(result_name)

        out_path = session_dir / f"run-{i:02d}.json"
        ds = run_pts_dataset(
            pts_bin=pts_bin,
            pts_root=pts_root,
            suite=suite,
            pts_mode=pts_mode,
            pts_extra_args=pts_extra_args,
            output_json=out_path,
            collect_features=collect_features,
            xi_override=use_xi,
            warmup_s=warmup_s,
            proc_sample_s=proc_sample_s,
            mem_mb=mem_mb,
            enable_ebpf=enable_ebpf,
            session_tag=tag,
            round_index=i,
            total_rounds=num_rounds,
            result_name=result_name,
            pts_inherit_stdio=pts_inherit_stdio,
        )
        if xi_cached is None and ds.get("xi") is not None:
            xi_cached = ds["xi"]
        written.append(
            {
                "round": i,
                "json": str(out_path),
                "created_at_utc": ds.get("created_at_utc"),
                "pts_result_file_name": (ds.get("phoronix") or {}).get("result_file_name"),
                "pts_raw_export": (ds.get("phoronix") or {}).get("raw_json_export_path"),
            }
        )

    manifest: dict[str, Any] = {
        "schema": "moebench.phoronix.batch_manifest.v1",
        "session_tag": tag,
        "session_dir": str(session_dir),
        "num_rounds": num_rounds,
        "reuse_xi": reuse_xi,
        "suite": suite,
        "pts_mode": pts_mode,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": written,
    }
    man_path = session_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote batch manifest {man_path}", file=sys.stderr)

    return manifest
