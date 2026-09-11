#!/usr/bin/env python3
"""Plan or run the reviewer baseline matrix for selected hosts.

This is the single entry point for the reviewer baselines.  It keeps the
execution-based subset policies and execution-free predictors in separate
artifacts, while using the same host, suite, K, and seed selection.

The default mode is ``--dry-run`` so inspecting a command matrix never starts
an expensive benchmark reconstruction job.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moebench.dataset_machines import glob_for_machine

PYTHON = sys.executable
POLICIES = (
    "full_suite",
    "fixed_first_k",
    "fixed_cpu_mix",
    "fixed_io_mix",
    "random",
    "greedy_slowest",
    "greedy_fastest",
    "correlation",
    "router",
    "wang_dnn",
    "tousi_mlp",
)
EXECUTION_POLICIES = POLICIES[:9]
EXECUTION_FREE = POLICIES[9:]


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _suite_args(suite: str) -> list[str]:
    if suite == "unixbench":
        return ["--suites", "unixbench", "--log1p-partial-index"]
    if suite == "phoronix_cpu":
        return ["--suites", "phoronix_cpu", "--log1p-partial-value"]
    if suite == "phoronix_gpu":
        return ["--suites", "phoronix_gpu", "--log1p-partial-value"]
    raise ValueError(suite)


def _machine_glob(suite: str, machine: str) -> str:
    if suite == "unixbench":
        return glob_for_machine(benchmark="unixbench", machine=machine)
    pts_suite = "cpu" if suite == "phoronix_cpu" else "pts/nvidia-gpu-compute"
    return glob_for_machine(benchmark="phoronix", machine=machine, pts_suite=pts_suite)


def build_commands(
    *,
    suites: list[str],
    machines: list[str],
    policies: list[str],
    k: int,
    seeds: list[int],
    dataset_root: Path,
    out_root: Path,
    router_models: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown baseline(s): {', '.join(unknown)}")
    commands: list[dict[str, object]] = []
    for seed in seeds:
        exec_policies = [p for p in policies if p in EXECUTION_POLICIES and p != "full_suite"]
        if exec_policies:
            out_dir = out_root / f"policy_seed{seed}"
            for machine in machines:
                for suite in suites:
                    cmd = [
                        PYTHON,
                        str(REPO / "scripts" / "paper_reconstruct_cv_extras.py"),
                        "--dataset-root", str(dataset_root),
                        *_suite_args(suite),
                        f"--glob-{'unixbench' if suite == 'unixbench' else 'pts-cpu' if suite == 'phoronix_cpu' else 'pts-gpu'}",
                        _machine_glob(suite, machine),
                        "--cv-mode", "leave_one_session_out",
                        "--seed", str(seed),
                        "--eval-partial-k", str(k),
                        "--policies", ",".join(exec_policies),
                        "--match-train-k-to-eval",
                        "--train-with-eval-policy",
                        "--report-json",
                        str(out_dir / f"policy_{suite}_{machine}.json"),
                    ]
                    if "router" in exec_policies:
                        model_arg = {
                            "unixbench": "--router-model-unixbench",
                            "phoronix_cpu": "--router-model-pts-cpu",
                            "phoronix_gpu": "--router-model-pts-gpu",
                        }[suite]
                        model_path = (router_models or {}).get(suite, "")
                        if not model_path:
                            raise ValueError(
                                f"router baseline requested for {suite}; "
                                f"provide --router-model-{suite.replace('phoronix_', 'pts-')}"
                            )
                        cmd.extend([model_arg, model_path])
                    commands.append({
                        "kind": "execution_based",
                        "suite": suite,
                        "machine": machine,
                        "seed": seed,
                        "output": str(out_dir),
                        "argv": [str(x) for x in cmd],
                    })

        free = [p for p in policies if p in EXECUTION_FREE]
        for method in free:
            for suite in suites:
                free_suite = {"phoronix_cpu": "pts_cpu", "phoronix_gpu": "pts_gpu"}.get(suite, suite)
                for machine in machines:
                    out_dir = out_root / f"execution_free_seed{seed}" / suite / machine / method
                    cmd = [
                        PYTHON,
                        str(REPO / "scripts" / "baseline_execution_free.py"),
                        "--suite", free_suite,
                        "--machine", machine,
                        "--dataset-root", str(dataset_root),
                        "--methods", method,
                        "--seed", str(seed),
                        "--out-dir", str(out_dir),
                    ]
                    commands.append({
                        "kind": "execution_free",
                        "suite": suite,
                        "machine": machine,
                        "seed": seed,
                        "output": str(out_dir),
                        "argv": [str(x) for x in cmd],
                    })
    return commands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", default="unixbench,phoronix_cpu,phoronix_gpu")
    ap.add_argument("--machines", required=True, help="Comma-separated host slugs")
    ap.add_argument("--baselines", default=",".join(POLICIES))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--dataset-root", type=Path, default=REPO / "dataset")
    ap.add_argument("--out-root", type=Path, default=REPO / "dataset/reviewer_baselines")
    ap.add_argument("--router-model-unixbench", default="")
    ap.add_argument("--router-model-pts-cpu", default="")
    ap.add_argument("--router-model-pts-gpu", default="")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--run", action="store_true", help="Execute commands after writing the manifest")
    args = ap.parse_args()

    suites = _csv(args.suites)
    machines = _csv(args.machines)
    policies = _csv(args.baselines)
    seeds = [int(x) for x in _csv(args.seeds)]
    commands = build_commands(
        suites=suites,
        machines=machines,
        policies=policies,
        k=args.k,
        seeds=seeds,
        dataset_root=args.dataset_root.resolve(),
        out_root=args.out_root.resolve(),
        router_models={
            "unixbench": args.router_model_unixbench,
            "phoronix_cpu": args.router_model_pts_cpu,
            "phoronix_gpu": args.router_model_pts_gpu,
        },
    )
    manifest = args.manifest or (args.out_root / "reviewer_baseline_manifest.json")
    payload = {
        "schema": "moebench.reviewer_baseline_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "suites": suites,
        "machines": machines,
        "baselines": policies,
        "k": args.k,
        "seeds": seeds,
        "commands": commands,
        "full_suite_note": "full_suite is a historical full-run reference; it is not re-executed by this runner.",
        "is_dry_run": not args.run,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest}")
    if not args.run:
        for item in commands:
            print(shlex.join(item["argv"]))  # type: ignore[arg-type]
        return 0
    for item in commands:
        subprocess.run(item["argv"], check=True)  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
