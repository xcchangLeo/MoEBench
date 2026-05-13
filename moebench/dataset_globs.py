"""Glob patterns aligned with MoEBench data collection session directory names.

UnixBench writes ``dataset/<host>_<UTCstamp>/run-NN.json`` (see ``unixbench.pipeline.default_session_tag``).

PTS writes ``dataset/<host>_<suite_token>_<UTCstamp>/run-NN.json`` where ``suite_token`` is
``pts_suite.replace('/', '_')`` passed through ``safe_session_tag`` (see ``phoronix.pipeline.default_session_tag``).
"""

from __future__ import annotations

from moebench.phoronix.pipeline import safe_session_tag

# Under ``dataset-root``; UnixBench loader filters by schema ``moebench.unixbench.dataset.v1``.
GLOB_UNIXBENCH_RUNS = "*/run-*.json"


def pts_session_token_from_suite(pts_suite: str) -> str:
    return safe_session_tag(str(pts_suite).replace("/", "_"))


def glob_for_pts_collected_sessions(pts_suite: str) -> str:
    """Match only session dirs produced by default PTS collection for ``pts_suite``."""
    token = pts_session_token_from_suite(pts_suite)
    return f"*_{token}_*/run-*.json"


def resolve_glob_pattern(
    *,
    benchmark: str,
    glob_pattern: str | None,
    pts_suite: str | None,
) -> str:
    """
    Return an effective glob under ``dataset-root``.

    If ``glob_pattern`` is non-empty, it wins. Otherwise:
    - ``unixbench`` → ``GLOB_UNIXBENCH_RUNS`` (``*/run-*.json``).
    - ``phoronix`` → suite-specific glob when ``pts_suite`` is set, else ``*/run-*.json``.
    """
    if glob_pattern and glob_pattern.strip():
        return glob_pattern.strip()
    if benchmark == "unixbench":
        return GLOB_UNIXBENCH_RUNS
    if benchmark == "phoronix":
        if pts_suite:
            return glob_for_pts_collected_sessions(pts_suite)
        return "*/run-*.json"
    raise ValueError(f"unknown benchmark: {benchmark!r}")
