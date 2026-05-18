"""UnixBench sub-tests as experts: categories and metadata templates."""

from __future__ import annotations

from typing import Any

# MoEBench always runs UnixBench with a single parallel copy (``perl Run -c 1``).
# UnixBench's default without ``-c`` also runs N-copy blocks when CPU count > 1.
UNIXBENCH_PARALLEL_COPIES: int = 1

# Default `perl Run` uses the "index" suite (system index score).
INDEX_SUITE_TEST_IDS: tuple[str, ...] = (
    "dhry2reg",
    "whetstone-double",
    "execl",
    "fstime",
    "fsbuffer",
    "fsdisk",
    "pipe",
    "context1",
    "spawn",
    "syscall",
    "shell1",
    "shell8",
)

# Research-facing coarse category (CPU / memory / IO / syscall / thread / network).
ExpertCategory = str

# Human-readable titles (must match UnixBench Run $testParams logmsg where applicable).
_TEST_TITLES: dict[str, str] = {
    "dhry2reg": "Dhrystone 2 using register variables",
    "whetstone-double": "Double-Precision Whetstone",
    "execl": "Execl Throughput",
    "fstime": "File Copy 1024 bufsize 2000 maxblocks",
    "fsbuffer": "File Copy 256 bufsize 500 maxblocks",
    "fsdisk": "File Copy 4096 bufsize 8000 maxblocks",
    "pipe": "Pipe Throughput",
    "context1": "Pipe-based Context Switching",
    "spawn": "Process Creation",
    "syscall": "System Call Overhead",
    "shell1": "Shell Scripts (1 concurrent)",
    "shell8": "Shell Scripts (8 concurrent)",
}

# MoE / modeling category (user taxonomy: CPU / memory / IO / syscall / thread / network).
_CATEGORY: dict[str, ExpertCategory] = {
    "dhry2reg": "CPU",
    "whetstone-double": "CPU",
    "execl": "thread",
    "fstime": "IO",
    "fsbuffer": "IO",
    "fsdisk": "IO",
    "pipe": "thread",
    "context1": "thread",
    "spawn": "thread",
    "syscall": "syscall",
    "shell1": "thread",
    "shell8": "thread",
}


def expert_template(test_id: str, expert_index: int) -> dict[str, Any]:
    """
    Static expert metadata. Fields to be filled or updated across dataset builds:

    - historical_runtime_mean / historical_runtime_variance: from multiple runs
    - suite_contribution_weight: e.g. sensitivity of composite index to this test
    - correlation_with: Pearson / Kendall vs other experts (offline)
    - hardware_stability: e.g. coefficient of variation across machines (offline)
    - execution_cost: proxy = mean wall time (seconds) on reference runs
    """
    title = _TEST_TITLES.get(test_id, test_id)
    cat = _CATEGORY.get(test_id, "CPU")
    return {
        "expert_id": f"e_{expert_index:03d}",
        "test_id": test_id,
        "title": title,
        "category": cat,
        "unixbench_default_suite": "index",
        "historical_runtime_mean_s": None,
        "historical_runtime_variance": None,
        "suite_contribution_weight": None,
        "correlation_with": {},
        "hardware_stability": None,
        "execution_cost": None,
        "notes": "Weights/correlations/stability to be estimated from aggregated dataset D.",
    }


def build_expert_catalog(test_ids: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    ids = list(test_ids) if test_ids is not None else list(INDEX_SUITE_TEST_IDS)
    return [expert_template(tid, i + 1) for i, tid in enumerate(ids)]
