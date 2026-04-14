"""Extract yi/ti-friendly structures from PTS ``result-file-to-json`` output."""

from __future__ import annotations

from typing import Any

from moebench.phoronix.experts import expert_template_pts, sort_profile_identifiers


def _group_pts_results(export: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Group raw PTS ``results`` entries by profile ``identifier`` (one logical expert per profile)."""
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for _h, robj in sorted((export.get("results") or {}).items(), key=lambda x: x[0]):
        tid = str(robj.get("identifier") or _h)
        groups.setdefault(tid, []).append((_h, robj))
    return groups


def extract_ti_from_pts_json(export: dict[str, Any]) -> dict[str, Any]:
    """
    Build ``ti`` compatible with UnixBench-style ``by_test_id`` where possible.

    Per-test times come from buffer ``test_run_times`` (seconds per run), summed per profile.
    Multiple ``results`` blocks with the same ``identifier`` are merged (summed times).
    """
    groups = _group_pts_results(export)
    by_test: dict[str, dict[str, Any]] = {}
    for tid in sort_profile_identifiers(list(groups.keys())):
        rows = groups[tid]
        total_time = 0.0
        run_lists: list[list[float]] = []
        title = None
        for _h, robj in rows:
            title = title or robj.get("title")
            buffers = robj.get("results") or {}
            for _rid in sorted(buffers.keys()):
                buf = buffers[_rid]
                trt = buf.get("test_run_times")
                if isinstance(trt, list) and trt:
                    vals = [float(x) for x in trt]
                    run_lists.append(vals)
                    total_time += sum(vals)
        entry: dict[str, Any] = {"identifier": tid, "title": title}
        if total_time > 0:
            entry["time_s_total"] = total_time
        if run_lists:
            entry["test_run_times_per_buffer"] = run_lists
        if not entry.get("time_s_total") and rows:
            robj0 = rows[0][1]
            buffers = robj0.get("results") or {}
            if len(buffers) == 1:
                b0 = next(iter(buffers.values()))
                trt = b0.get("test_run_times")
                if isinstance(trt, list) and trt:
                    entry["time_s_total"] = float(sum(float(x) for x in trt))
        by_test[tid] = entry

    return {
        "by_test_id": by_test,
        "unit": "seconds",
        "description": "Per-profile times from PTS JSON buffers.test_run_times (summed per profile).",
    }


def build_experts_from_pts_json(export: dict[str, Any]) -> list[dict[str, Any]]:
    """One expert row per unique PTS profile ``identifier``, metadata aligned with UnixBench experts."""
    groups = _group_pts_results(export)
    out: list[dict[str, Any]] = []
    for idx, tid in enumerate(sort_profile_identifiers(list(groups.keys())), start=1):
        rows = groups[tid]
        title = None
        total_cost = 0.0
        observed: dict[str, Any] | None = None
        for _h, robj in rows:
            title = title or robj.get("title")
            buffers = robj.get("results") or {}
            for _rid in sorted(buffers.keys()):
                buf = buffers[_rid]
                trt = buf.get("test_run_times")
                if isinstance(trt, list) and trt:
                    total_cost += sum(float(x) for x in trt)
        for _h, robj in rows:
            buffers = robj.get("results") or {}
            for _rid in sorted(buffers.keys()):
                buf = buffers[_rid]
                trt = buf.get("test_run_times")
                observed = {
                    "value": buf.get("value"),
                    "scale": robj.get("scale"),
                    "raw_values": buf.get("raw_values"),
                    "test_run_times": trt,
                }
                break
            if observed is not None:
                break
        base = expert_template_pts(tid, idx, title=title)
        if observed is not None:
            base["observed"] = observed
        base["execution_cost"] = total_cost if total_cost > 0 else None
        out.append(base)
    return out
