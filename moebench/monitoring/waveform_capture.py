"""UnixBench resource waveform capture for four execution modes.

Measurement policy (v3):
- **full** / **route_a**: monitor only ``perl Run`` subprocess work (xi routing is outside the trace).
- **route_b** / **benchscout**: monitor only micro-workload execution + light ML inference;
  eBPF feature collection and ``/proc`` probe snapshots are excluded from the trace window.
- **benchscout**: ``collect_all`` (xi) runs before the monitor, matching route_a.
- Memory samples include ``mem_delta_*`` fields (drop in MemAvailable vs. t=0 baseline).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moebench import collect_all
from moebench.hybrid.eval import probe_predictions_to_executed_tests, router_select_test_ids
from moebench.monitoring.plot_waveforms import MODE_DISPLAY_LABELS
from moebench.monitoring.resource_monitor import ResourceMonitor
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.probe.workloads import category_for_unixbench_test, run_category_workload
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS, UNIXBENCH_PARALLEL_COPIES

WAVEFORM_PROBE_MEM_MB = 64


def load_router_meta(model_fp: Path, *, auto_install: bool = False) -> dict[str, Any]:
    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        try:
            with open(model_fp, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else ""
            if missing and auto_install:
                from moebench.pip_install import ensure_importable

                ensure_importable(missing, auto_install=True)
                with open(model_fp, "rb") as f:
                    return pickle.load(f)
            raise
    import torch

    try:
        return torch.load(model_fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_fp, map_location="cpu")


def run_unixbench(
    unixbench_root: Path,
    result_dir: Path,
    base_name: str,
    test_ids: list[str] | None,
    copies: int,
) -> float:
    run_script = unixbench_root / "Run"
    env = os.environ.copy()
    env["UB_OUTPUT_FILE_NAME"] = base_name
    env["UB_RESULTDIR"] = str(result_dir)
    cmd = ["perl", str(run_script), "-c", str(copies)]
    if test_ids:
        cmd.extend(test_ids)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(unixbench_root), env=env)
    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"UnixBench Run failed rc={rc}")
    return wall


def run_waveform_micro_probes(
    test_ids: list[str],
    *,
    duration_s: float,
    mem_mb: int = WAVEFORM_PROBE_MEM_MB,
    phase_markers: list[dict[str, Any]] | None = None,
    mon: ResourceMonitor | None = None,
) -> list[dict[str, Any]]:
    """Run category micro-workloads only (no eBPF / proc snapshots)."""
    markers = phase_markers if phase_markers is not None else []
    for tid in test_ids:
        if mon is not None:
            markers.append({"name": tid, "t_rel_s": mon.elapsed_s()})
        category = category_for_unixbench_test(tid)
        run_category_workload(category, duration_s, mem_mb=mem_mb)
    return markers


def capture_full(
    *,
    unixbench_root: Path,
    result_dir: Path,
    session_tag: str,
    copies: int,
    interval_s: float,
    test_ids: list[str] | None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"moebench_wave_full_{session_tag}_{stamp}".replace(":", "-")
    mon = ResourceMonitor(interval_s=interval_s)

    def _work() -> None:
        run_unixbench(unixbench_root, result_dir, base, test_ids, copies)

    tr = mon.run(_work)
    tr["label"] = MODE_DISPLAY_LABELS["full"]
    tr["mode"] = "full"
    tr["monitored_scope"] = "unixbench_subprocess"
    tr["unixbench_report"] = str(result_dir / base)
    tr["test_ids"] = test_ids or list(INDEX_SUITE_TEST_IDS)
    return tr


def capture_route_a(
    *,
    unixbench_root: Path,
    result_dir: Path,
    session_tag: str,
    router_model: Path,
    copies: int,
    interval_s: float,
    top_k: int | None,
    skip_xi: bool,
    warmup_s: float,
    test_ids: list[str] | None,
) -> dict[str, Any]:
    router_meta = load_router_meta(router_model)
    k = int(top_k if top_k is not None else router_meta.get("top_k", 3))

    xi: dict[str, Any] = {}
    if not skip_xi:
        xi = collect_all(warmup_s=warmup_s, proc_sample_s=0.5, enable_ebpf=False, mem_mb=64)

    _scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    _selected_experts, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, k)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"moebench_wave_route_a_{session_tag}_{stamp}".replace(":", "-")
    mon = ResourceMonitor(interval_s=interval_s)

    def _work() -> None:
        run_unixbench(unixbench_root, result_dir, base, selected_test_ids, copies)

    tr = mon.run(_work)
    tr["label"] = MODE_DISPLAY_LABELS["route_a"]
    tr["mode"] = "route_a"
    tr["monitored_scope"] = "unixbench_subprocess"
    tr["router_model"] = str(router_model.resolve())
    tr["top_k"] = k
    tr["selected_test_ids"] = selected_test_ids
    tr["unixbench_report"] = str(result_dir / base)
    tr["xi_collected_outside_trace"] = not skip_xi
    if test_ids:
        tr["note"] = "quick mode: router still used; partial list may differ from test_ids filter"
    return tr


def capture_route_b(
    *,
    probe_model: Path,
    interval_s: float,
    probe_duration_s: float | None,
    probe_mode: str,
) -> dict[str, Any]:
    bundle = load_probe_bundle(probe_model)
    tids = list(bundle.get("test_ids") or INDEX_SUITE_TEST_IDS)
    dur = float(probe_duration_s if probe_duration_s is not None else bundle.get("probe_duration_s", 4.0))
    mode = probe_mode or str(bundle.get("probe_mode", "micro"))
    if mode != "micro":
        raise ValueError(
            f"waveform capture requires probe_mode=micro (got {mode!r}); "
            "real subtests would inflate resource usage vs. full run"
        )

    mon = ResourceMonitor(interval_s=interval_s)
    mon.start()
    phase_markers: list[dict[str, Any]] = []
    try:
        phase_markers = run_waveform_micro_probes(
            tids,
            duration_s=dur,
            phase_markers=phase_markers,
            mon=mon,
        )
    finally:
        tr = mon.stop()

    tr["label"] = MODE_DISPLAY_LABELS["route_b"]
    tr["mode"] = "route_b"
    tr["monitored_scope"] = "micro_workloads_only"
    tr["probe_model"] = str(probe_model.resolve())
    tr["probe_duration_s"] = dur
    tr["probe_mode"] = mode
    tr["test_ids"] = tids
    tr["phase_markers"] = phase_markers
    tr["excluded_from_trace"] = ["ebpf", "proc_snapshot"]
    return tr


def capture_benchscout(
    *,
    router_model: Path,
    recon_model: Path,
    probe_model: Path,
    interval_s: float,
    top_k: int | None,
    skip_xi: bool,
    warmup_s: float,
    probe_duration_s: float | None,
    probe_mode: str,
) -> dict[str, Any]:
    router_meta = load_router_meta(router_model)
    recon_bundle = load_reconstruction_bundle(recon_model)
    probe_bundle = load_probe_bundle(probe_model)
    k = int(top_k if top_k is not None else router_meta.get("top_k", 3))
    dur = float(probe_duration_s if probe_duration_s is not None else probe_bundle.get("probe_duration_s", 4.0))
    mode = probe_mode or str(probe_bundle.get("probe_mode", "micro"))
    benchmark = str(probe_bundle.get("benchmark") or recon_bundle.get("benchmark") or "unixbench")
    if mode != "micro":
        raise ValueError(
            f"waveform capture requires probe_mode=micro (got {mode!r}); "
            "real subtests would inflate resource usage vs. full run"
        )

    xi: dict[str, Any] = {}
    if not skip_xi:
        xi = collect_all(warmup_s=warmup_s, proc_sample_s=0.5, enable_ebpf=False, mem_mb=64)

    mon = ResourceMonitor(interval_s=interval_s)
    mon.start()
    phase_markers: list[dict[str, Any]] = []
    router_detail: dict[str, Any] = {}
    selected_test_ids: list[str] = []
    predicted_suite: float | None = None

    try:
        phase_markers.append({"name": "router", "t_rel_s": mon.elapsed_s()})
        _, selected_test_ids, router_detail = router_select_test_ids(router_meta, xi, top_k=k)

        sub_preds: dict[str, float] = {}
        probe_walls: dict[str, float] = {}
        for tid in selected_test_ids:
            phase_markers.append({"name": tid, "t_rel_s": mon.elapsed_s()})
            category = category_for_unixbench_test(tid)
            run_category_workload(category, dur, mem_mb=WAVEFORM_PROBE_MEM_MB)
            stub_probe = {
                "test_id": tid,
                "benchmark": benchmark,
                "category": category,
                "probe_mode": mode,
                "duration_s": dur,
                "wall_s": dur,
                "workload": {"category": category, "duration_s": dur, "mode": "micro"},
                "ebpf": {"available": False, "reason": "waveform_capture"},
                "proc": {},
            }
            sub_preds[tid] = predict_subtest(probe_bundle, stub_probe, tid)
            probe_walls[tid] = float(dur)

        phase_markers.append({"name": "recon", "t_rel_s": mon.elapsed_s()})
        executed = probe_predictions_to_executed_tests(
            selected_test_ids,
            sub_preds,
            dur,
            benchmark=benchmark,
            probe_wall_by_tid=probe_walls,
        )
        pred = predict_from_partial(recon_bundle, xi, executed)
        predicted_suite = float(pred["suite_index"])
    finally:
        tr = mon.stop()

    tr["label"] = MODE_DISPLAY_LABELS["benchscout"]
    tr["mode"] = "benchscout"
    tr["monitored_scope"] = "router_inference_micro_workloads_recon"
    tr["router_model"] = str(router_model.resolve())
    tr["recon_model"] = str(recon_model.resolve())
    tr["probe_model"] = str(probe_model.resolve())
    tr["top_k"] = k
    tr["selected_test_ids"] = selected_test_ids
    tr["router"] = router_detail
    tr["probe_duration_s"] = dur
    tr["probe_mode"] = mode
    tr["predicted_suite_index"] = predicted_suite
    tr["phase_markers"] = phase_markers
    tr["xi_collected_outside_trace"] = not skip_xi
    tr["excluded_from_trace"] = ["ebpf", "proc_snapshot", "xi_collection"]
    return tr


def default_copies(copies: int) -> int:
    return copies if copies > 0 else UNIXBENCH_PARALLEL_COPIES
