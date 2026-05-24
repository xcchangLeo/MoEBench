"""Per-machine dataset filtering (train / evaluate on one host's sessions only)."""

from __future__ import annotations

import re
import socket
from pathlib import Path

from moebench.dataset_globs import (
    GLOB_UNIXBENCH_RUNS,
    glob_for_pts_collected_sessions,
    pts_session_token_from_suite,
    resolve_glob_pattern,
)

_SESSION_UTC_RE = re.compile(r"^(.+)_(\d{8}T\d{6}Z)$")


def local_host_slug() -> str:
    """Same slug as UnixBench / PTS collection default session prefix."""
    try:
        h = socket.gethostname()
    except OSError:
        h = "unknown-host"
    h = h.strip() or "unknown-host"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in h)
    return safe or "host"


def machine_from_session_tag(session_tag: str, *, pts_suite: str | None = None) -> str:
    """
    Parse machine slug from a session directory name.

    UnixBench: ``<host>_<UTC>``
    PTS: ``<host>_<suite_token>_<UTC>``
    """
    tag = session_tag.strip()
    m = _SESSION_UTC_RE.match(tag)
    if not m:
        return tag.split("_", 1)[0] or tag
    prefix = m.group(1)
    if pts_suite:
        token = pts_session_token_from_suite(pts_suite)
        suffix = f"_{token}"
        if prefix.endswith(suffix):
            return prefix[: -len(suffix)] or prefix
    return prefix


_SESSION_UTC_GLOB = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z"


def glob_for_machine(
    *,
    benchmark: str,
    machine: str,
    pts_suite: str | None = None,
) -> str:
    host = machine.strip()
    if not host:
        raise ValueError("machine slug must be non-empty")
    if benchmark == "unixbench":
        # ``<host>_<UTC>/run-*.json`` — exclude ``<host>_cpu_<UTC>`` PTS dirs.
        return f"{host}_{_SESSION_UTC_GLOB}/run-*.json"
    if benchmark == "phoronix":
        token = pts_session_token_from_suite(pts_suite or "cpu")
        return f"{host}_{token}_{_SESSION_UTC_GLOB}/run-*.json"
    raise ValueError(f"unknown benchmark: {benchmark!r}")


def resolve_glob_for_machine(
    *,
    benchmark: str,
    machine: str | None,
    glob_pattern: str | None,
    pts_suite: str | None,
) -> str:
    """Machine-scoped glob unless the caller supplied an explicit ``glob_pattern``."""
    if glob_pattern and glob_pattern.strip():
        return glob_pattern.strip()
    if machine and machine.strip():
        return glob_for_machine(benchmark=benchmark, machine=machine.strip(), pts_suite=pts_suite)
    return resolve_glob_pattern(
        benchmark=benchmark,
        glob_pattern=None,
        pts_suite=pts_suite,
    )


def list_machines_in_dataset(
    dataset_root: str | Path,
    *,
    benchmark: str,
    pts_suite: str | None = None,
) -> list[str]:
    """Discover machine slugs that have at least one matching run JSON for ``benchmark``."""
    root = Path(dataset_root).resolve()
    if benchmark == "unixbench":
        from moebench.reconstruct.data import collect_unixbench_run_paths

        paths = collect_unixbench_run_paths(root, glob_pattern=GLOB_UNIXBENCH_RUNS)
    elif benchmark == "phoronix":
        from moebench.phoronix.training_data import collect_phoronix_run_paths

        pattern = glob_for_pts_collected_sessions(pts_suite or "cpu")
        paths = collect_phoronix_run_paths(root, glob_pattern=pattern, pts_suite=pts_suite)
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")

    seen: set[str] = set()
    for p in paths:
        seen.add(machine_from_session_tag(p.parent.name, pts_suite=pts_suite))
    return sorted(seen)


def resolve_training_machine(explicit: str | None) -> str:
    """Default training scope is the local host (each machine trains on its own data)."""
    if explicit and explicit.strip():
        return explicit.strip()
    return local_host_slug()
