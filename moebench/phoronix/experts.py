"""Phoronix profile experts: metadata template aligned with ``moebench.unixbench.experts``."""

from __future__ import annotations

from typing import Any


def infer_pts_category(test_id: str, title: str | None) -> str:
    """Coarse category (CPU / memory / IO / compile / encode / crypto / …)."""
    blob = f"{title or ''} {test_id}".lower()
    if any(x in blob for x in ("compile", "build-linux", "kernel", "gcc", "llvm")):
        return "compile"
    if any(x in blob for x in ("encode", "x264", "x265", "kvazaar", "vp9", "av1")):
        return "encode"
    if any(x in blob for x in ("openssl", "encrypt", "crypto", "gnupg")):
        return "crypto"
    if any(x in blob for x in ("iozone", "fio", "sqlite", "postgres", "redis")):
        return "IO"
    if any(x in blob for x in ("memory", "ramspeed", "stream")):
        return "memory"
    return "CPU"


def expert_template_pts(
    test_id: str,
    expert_index: int,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """
    Static expert metadata (same field names / roles as ``unixbench.experts.expert_template``).

    - historical_runtime_* / suite_contribution_weight / correlation_with / hardware_stability:
      filled or estimated from aggregated runs offline.
    - execution_cost: proxy = wall time (seconds) summed from PTS buffers for this profile.
    """
    disp = title if title else test_id
    cat = infer_pts_category(test_id, title)
    return {
        "expert_id": f"e_{expert_index:03d}",
        "test_id": test_id,
        "title": disp,
        "category": cat,
        "phoronix_profile_identifier": test_id,
        "phoronix_default_suite": "cpu",
        "historical_runtime_mean_s": None,
        "historical_runtime_variance": None,
        "suite_contribution_weight": None,
        "correlation_with": {},
        "hardware_stability": None,
        "execution_cost": None,
        "notes": "Weights/correlations/stability to be estimated from aggregated dataset D.",
    }


def sort_profile_identifiers(ids: list[str]) -> list[str]:
    """Stable, human-friendly order: version suffixes sort naturally where possible."""
    return sorted(ids, key=lambda s: (s.lower(), s))
