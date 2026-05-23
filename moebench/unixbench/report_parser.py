"""Parse UnixBench plain-text report produced by `Run` (summarizeRun / logResults)."""

from __future__ import annotations

import re
from typing import Any

from moebench.unixbench.experts import _TEST_TITLES


# Dhrystone 2 using register variables             12345.0 lps   (10.5 s, 9 samples)
_SCORE_LINE = re.compile(
    r"^(.+?)\s+(\d+(?:\.\d+)?)\s+(\S+)\s+\((\d+(?:\.\d+)?) s, (\d+) samples\)\s*$"
)

def _parse_index_row(line: str) -> tuple[str, float, float, float] | None:
    """Parse index table row: <title> baseline result index (last three tokens are floats)."""
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        idx = float(parts[-1])
        res = float(parts[-2])
        base = float(parts[-3])
    except ValueError:
        return None
    title = " ".join(parts[:-3]).strip()
    return title, base, res, idx


def _title_to_test_id(title: str) -> str | None:
    t = title.strip()
    for tid, msg in _TEST_TITLES.items():
        if msg.strip() == t:
            return tid
    return None


def _split_benchmark_blocks(text: str) -> list[str]:
    parts = re.split(r"(?=Benchmark Run:)", text, flags=re.MULTILINE)
    blocks = [p.strip() for p in parts if "Benchmark Run:" in p]
    if not blocks and "Benchmark Run:" in text:
        return [text.strip()]
    return blocks


def parse_report_text(report_text: str) -> dict[str, Any]:
    blocks = _split_benchmark_blocks(report_text)
    runs: list[dict[str, Any]] = []
    for raw in blocks:
        run = _parse_single_block(raw)
        if run:
            runs.append(run)
    return {"runs": runs}


def _parse_single_block(block: str) -> dict[str, Any] | None:
    if "Benchmark Run:" not in block:
        return None

    m_copies = re.search(r"running\s+(\d+)\s+parallel\s+copies", block)
    parallel_copies = int(m_copies.group(1)) if m_copies else None

    scores_by_title: dict[str, dict[str, Any]] = {}
    for line in block.splitlines():
        m = _SCORE_LINE.match(line.rstrip())
        if not m:
            continue
        title, score, unit, t_s, samples = m.groups()
        title = title.strip()
        scores_by_title[title] = {
            "score": float(score),
            "score_unit": unit,
            "time_s": float(t_s),
            "pass_samples": int(samples),
        }

    index_by_title: dict[str, dict[str, float]] = {}
    in_index = False
    for line in block.splitlines():
        if "System Benchmarks Index Values" in line or "System Benchmarks Partial Index" in line:
            in_index = True
            continue
        if in_index:
            if not line.strip():
                continue
            if "Index Score" in line and "System Benchmarks" in line:
                break
            if "BASELINE" in line and "RESULT" in line:
                continue
            parsed = _parse_index_row(line.rstrip())
            if parsed:
                title, base, res, idx = parsed
                index_by_title[title] = {
                    "baseline_score": base,
                    "result_score": res,
                    "index": idx,
                }

    cat_index: float | None = None
    for line in block.splitlines():
        if "System Benchmarks Index Score" in line:
            parts = line.split()
            if parts:
                try:
                    cat_index = float(parts[-1])
                except ValueError:
                    pass
            if cat_index is not None:
                break

    tests: dict[str, Any] = {}
    for title, sc in scores_by_title.items():
        tid = _title_to_test_id(title)
        if tid is None:
            tid = f"unmapped:{title[:48]}"
        entry = {"title": title, **sc}
        if title in index_by_title:
            entry["index_detail"] = index_by_title[title]
        tests[tid] = entry

    return {
        "parallel_copies": parallel_copies,
        "tests": tests,
        "system_benchmarks_index_score": cat_index,
        "index_rows": index_by_title,
    }


def build_ti_from_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """ti: per-subtest wall time (seconds) keyed by test_id and parallel_copies."""
    from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES

    by_test: dict[str, dict[str, float]] = {}
    for run in runs:
        copies = run.get("parallel_copies")
        key = str(copies if copies is not None else UNIXBENCH_PARALLEL_COPIES)
        for tid, tinfo in run.get("tests", {}).items():
            if tid.startswith("unmapped:"):
                continue
            if tid not in by_test:
                by_test[tid] = {}
            by_test[tid][key] = float(tinfo.get("time_s", 0.0))
    return {"by_test_id": by_test, "unit": "seconds", "description": "Wall time per sub-benchmark from UnixBench Run report"}


def pick_preferred_run_block(parsed: dict[str, Any]) -> dict[str, Any]:
    """Prefer MoEBench single-copy block (parallel_copies==1), else smallest int."""
    from moebench.unixbench.experts import UNIXBENCH_PARALLEL_COPIES

    runs = parsed.get("runs") or []

    def key(rb: dict[str, Any]) -> tuple[int, int]:
        pc = rb.get("parallel_copies")
        if pc == UNIXBENCH_PARALLEL_COPIES:
            return (0, 0)
        if isinstance(pc, int):
            return (1, pc)
        if pc is None:
            return (2, 10**9)
        return (3, 10**9)

    return sorted(runs, key=key)[0] if runs else {}


def parse_executed_tests_from_report(
    report_txt: str, selected_test_ids: list[str]
) -> tuple[list[dict[str, Any]], float | None]:
    """Build ``executed_tests`` list + composite suite index for selected subtests."""
    parsed = parse_report_text(report_txt)
    parsed_run = pick_preferred_run_block(parsed)
    tests_map = parsed_run.get("tests") or {}
    executed: list[dict[str, Any]] = []
    for tid in selected_test_ids:
        tinfo = tests_map.get(tid)
        if not tinfo:
            executed.append({"test_id": tid, "missing": True})
            continue
        executed.append(
            {
                "test_id": tid,
                "title": tinfo.get("title"),
                "score": tinfo.get("score"),
                "score_unit": tinfo.get("score_unit"),
                "time_s": tinfo.get("time_s"),
                "pass_samples": tinfo.get("pass_samples"),
                "index_detail": tinfo.get("index_detail"),
            }
        )
    suite = parsed_run.get("system_benchmarks_index_score")
    try:
        suite_f = float(suite) if suite is not None else None
    except (TypeError, ValueError):
        suite_f = None
    return executed, suite_f
