#!/usr/bin/env python3
"""Run live M1 probes for learned and fixed UnixBench Top-3 policies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moebench import collect_all
from moebench.dataset_machines import find_latest_router_recon_models_dir
from moebench.hybrid.eval import (
    probe_predictions_to_executed_tests,
    router_select_test_ids,
)
from moebench.probe.collector import collect_subtest_probe
from moebench.probe.inference import load_probe_bundle, predict_subtest
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.unixbench.experts import INDEX_SUITE_TEST_IDS
from moebench.unixbench.pipeline import run_unixbench_dataset


POLICY_IDS = ("learned", "fixed_first_k", "fixed_cpu_mix", "fixed_io_mix")
FIXED_CPU_IDS = ("dhry2reg", "whetstone-double", "syscall")
FIXED_IO_IDS = ("fstime", "fsbuffer", "fsdisk")


def _select(policy: str, xi: dict[str, Any], router: dict[str, Any], k: int) -> list[str]:
    if policy == "learned":
        return router_select_test_ids(router, xi, top_k=k)[1]
    preferred = {
        "fixed_first_k": (),
        "fixed_cpu_mix": FIXED_CPU_IDS,
        "fixed_io_mix": FIXED_IO_IDS,
    }[policy]
    ordered = [tid for tid in preferred if tid in INDEX_SUITE_TEST_IDS]
    ordered.extend(tid for tid in INDEX_SUITE_TEST_IDS if tid not in ordered)
    return ordered[:k]


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_average() -> list[float]:
    return [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]


def _run_policy(
    policy: str,
    *,
    xi: dict[str, Any],
    router: dict[str, Any],
    recon: dict[str, Any],
    probe_bundle: dict[str, Any],
    top_k: int,
    duration_s: float,
    enable_ebpf: bool,
    unixbench_root: Path,
) -> dict[str, Any]:
    selected = _select(policy, xi, router, top_k)
    probes: dict[str, Any] = {}
    predictions: dict[str, float] = {}
    walls: dict[str, float] = {}
    load_before = _load_average()
    started = time.perf_counter()
    for test_id in selected:
        probe = collect_subtest_probe(
            test_id,
            duration_s=duration_s,
            enable_ebpf=enable_ebpf,
            benchmark="unixbench",
            probe_mode="micro",
            unixbench_root=unixbench_root,
        )
        probes[test_id] = probe
        predictions[test_id] = predict_subtest(probe_bundle, probe, test_id)
        walls[test_id] = float(probe.get("wall_s") or duration_s)
    executed = probe_predictions_to_executed_tests(
        selected,
        predictions,
        duration_s,
        benchmark="unixbench",
        probe_wall_by_tid=walls,
    )
    inferred = predict_from_partial(recon, xi, executed)
    elapsed = time.perf_counter() - started
    return {
        "policy": policy,
        "selected_test_ids": selected,
        "probe_predictions": predictions,
        "suite_prediction": float(inferred["suite_index"]),
        "probe_process_wall_sum_s": sum(walls.values()),
        "probe_phase_elapsed_s": elapsed,
        "probes": probes,
        "load_average_before": load_before,
        "load_average_after": _load_average(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="aces-System-Product-Name")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--unixbench-root", type=Path, default=Path("/home/zyf/code/byte-unixbench/UnixBench"))
    parser.add_argument("--router-model", type=Path, default=None)
    parser.add_argument("--reconstruct-model", type=Path, default=None)
    parser.add_argument("--probe-model", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probe-duration-s", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--enable-ebpf", action="store_true", help="Enable eBPF; off by default to match the probe training corpus")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    models_dir = dataset_root / "models" / args.machine
    grid_dir = find_latest_router_recon_models_dir(
        dataset_root, machine=args.machine, benchmark="unixbench"
    )
    if grid_dir is None:
        raise FileNotFoundError(f"No UnixBench router/reconstructor model grid for {args.machine}")
    router_path = (args.router_model or grid_dir / "router_mlp.pt").resolve()
    recon_path = (args.reconstruct_model or models_dir / "reconstruct_lightgbm_v2.pkl").resolve()
    probe_path = (args.probe_model or models_dir / "probe_unixbench_4s_lightgbm.pkl").resolve()
    for path in (router_path, recon_path, probe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.unixbench_root / "Run").is_file():
        raise FileNotFoundError(args.unixbench_root / "Run")

    router = __import__("moebench.hybrid.eval", fromlist=["load_router_meta"]).load_router_meta(router_path)
    recon = load_reconstruction_bundle(recon_path)
    probe_bundle = load_probe_bundle(probe_path)
    if recon.get("benchmark") != "unixbench" or probe_bundle.get("benchmark") != "unixbench":
        raise ValueError("The selected checkpoints must be UnixBench models")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or dataset_root / "experiments" / args.machine / f"online_policy_comparison_{stamp}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_path = output.with_name(output.stem + "_full_reference.json")
    report: dict[str, Any] = {
        "schema": "moebench.experiment.online_routing_policy_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": args.machine,
        "protocol": "One live full UnixBench reference, followed by randomized repeated online micro-probe policy runs. Each repetition shares one freshly collected context across policies.",
        "host_runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "load_average_at_start": _load_average(),
        },
        "policies": list(POLICY_IDS),
        "top_k": args.top_k,
        "probe_duration_s": args.probe_duration_s,
        "probe_mode": "micro",
        "enable_ebpf": args.enable_ebpf,
        "models": {"router": str(router_path), "reconstructor": str(recon_path), "probe": str(probe_path)},
        "unixbench_root": str(args.unixbench_root.resolve()),
        "full_reference": {"path": str(reference_path), "status": "pending"},
        "repetitions": [],
    }
    _atomic_write(output, report)

    print(f"Running one full UnixBench reference; output: {output}", flush=True)
    full_load_before = _load_average()
    report["full_reference"]["load_average_before"] = full_load_before
    report["full_reference"]["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    _atomic_write(output, report)
    full_started = time.perf_counter()
    ground_truth_ds = run_unixbench_dataset(
        unixbench_root=args.unixbench_root,
        output_json=reference_path,
        collect_features=False,
        enable_ebpf=args.enable_ebpf,
        run_args=["-c", "1"],
        session_tag=f"online_policy_{stamp}",
    )
    full_elapsed = time.perf_counter() - full_started
    from moebench.probe.training_data import label_suite_from_unixbench_run

    ground_truth = label_suite_from_unixbench_run(ground_truth_ds)
    if ground_truth is None:
        raise RuntimeError("Could not extract suite index from the live full-suite run")
    report["full_reference"] = {
        "path": str(reference_path),
        "suite_index": float(ground_truth),
        "elapsed_s": full_elapsed,
        "load_average_before": full_load_before,
        "load_average_after": _load_average(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "status": "complete",
    }
    _atomic_write(output, report)
    print(f"Full-suite reference={ground_truth:.4f}, elapsed={full_elapsed:.1f}s", flush=True)

    rng = random.Random(args.seed)
    for repetition in range(args.repetitions):
        context_started = time.perf_counter()
        xi = collect_all(enable_ebpf=args.enable_ebpf)
        context_elapsed = time.perf_counter() - context_started
        order = list(POLICY_IDS)
        rng.shuffle(order)
        round_result: dict[str, Any] = {
            "repetition": repetition + 1,
            "policy_order": order,
            "context_collection_s": context_elapsed,
            "load_average_before_probes": _load_average(),
            "results": [],
        }
        report["repetitions"].append(round_result)
        _atomic_write(output, report)
        for policy in order:
            print(f"Rep {repetition + 1}/{args.repetitions}: {policy}", flush=True)
            row = _run_policy(
                policy,
                xi=xi,
                router=router,
                recon=recon,
                probe_bundle=probe_bundle,
                top_k=args.top_k,
                duration_s=args.probe_duration_s,
                enable_ebpf=args.enable_ebpf,
                unixbench_root=args.unixbench_root,
            )
            row["context_collection_s"] = context_elapsed
            row["suite_absolute_error"] = abs(row["suite_prediction"] - float(ground_truth))
            row["suite_relative_error_percent"] = (
                row["suite_absolute_error"] / max(abs(float(ground_truth)), 1e-9) * 100.0
            )
            round_result["results"].append(row)
            _atomic_write(output, report)
        round_result["load_average_after_probes"] = _load_average()
        _atomic_write(output, report)

    report["summary"] = {
        policy: {
            "n": sum(row["policy"] == policy for rep in report["repetitions"] for row in rep["results"]),
            "mean_suite_relative_error_percent": sum(
                row["suite_relative_error_percent"]
                for rep in report["repetitions"] for row in rep["results"]
                if row["policy"] == policy
            ) / args.repetitions,
            "mean_probe_phase_elapsed_s": sum(
                row["probe_phase_elapsed_s"]
                for rep in report["repetitions"] for row in rep["results"]
                if row["policy"] == policy
            ) / args.repetitions,
        }
        for policy in POLICY_IDS
    }
    _atomic_write(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
