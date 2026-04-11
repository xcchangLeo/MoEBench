"""Extract yi/ti-friendly structures from PTS ``result-file-to-json`` output."""

from __future__ import annotations

from typing import Any


def extract_ti_from_pts_json(export: dict[str, Any]) -> dict[str, Any]:
    """
    Build ``ti`` compatible with UnixBench-style ``by_test_id`` where possible.

    Per-test times come from buffer ``test_run_times`` (seconds per run), summed per profile.
    """
    by_test: dict[str, dict[str, Any]] = {}
    for _h, robj in (export.get("results") or {}).items():
        tid = str(robj.get("identifier") or _h)
        buffers = robj.get("results") or {}
        total_time = 0.0
        run_lists: list[list[float]] = []
        for _rid, buf in buffers.items():
            trt = buf.get("test_run_times")
            if isinstance(trt, list) and trt:
                vals = [float(x) for x in trt]
                run_lists.append(vals)
                total_time += sum(vals)
        entry: dict[str, Any] = {"identifier": tid, "title": robj.get("title")}
        if total_time > 0:
            entry["time_s_total"] = total_time
        if run_lists:
            entry["test_run_times_per_buffer"] = run_lists
        if len(buffers) == 1 and not entry.get("time_s_total"):
            # single buffer, value-only
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
    """Lightweight expert list: one row per PTS result profile."""
    out: list[dict[str, Any]] = []
    idx = 0
    for _h, robj in (export.get("results") or {}).items():
        tid = str(robj.get("identifier") or _h)
        idx += 1
        eid = f"e_{idx:03d}"
        buffers = robj.get("results") or {}
        observed: dict[str, Any] | None = None
        if buffers:
            first = next(iter(buffers.values()))
            observed = {
                "value": first.get("value"),
                "scale": robj.get("scale"),
                "raw_values": first.get("raw_values"),
                "test_run_times": first.get("test_run_times"),
            }
        cost = None
        if observed and observed.get("test_run_times"):
            cost = sum(float(x) for x in observed["test_run_times"])
        out.append(
            {
                "expert_id": eid,
                "test_id": tid,
                "title": robj.get("title"),
                "observed": observed,
                "execution_cost": cost,
            }
        )
    return out
