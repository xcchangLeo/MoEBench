#!/usr/bin/env python3
"""PTS experiment: xi → router Top-K partial run → reconstruct suite mean → full suite baseline.

Runs **without sudo** except optional ``--sudo-for-xi`` (re-execs only for feature collection).
If you run the whole script under ``sudo`` (e.g. for xi), PTS is still executed as
``SUDO_USER`` with ``HOME`` set to that user's home so installed tests under
``~/.phoronix-test-suite`` are found (same as ``moebench.phoronix`` pipeline).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench import collect_all
from moebench.dataset_machines import latest_pts_run_path_for_machine, resolve_training_machine
from moebench.phoronix.pipeline import (
    _export_result_json,
    _pts_argv_as_installing_user,
    _which_pts,
    default_pts_install_root,
    pts_clean_save_name,
    pts_subprocess_env,
    pts_warn_if_low_nvidia_vram_for_opencl_suite,
    safe_session_tag,
)
from moebench.phoronix.training_data import (
    full_suite_wall_seconds_pts,
    primary_time_from_pts_export,
    primary_value_from_export,
)
from moebench.reconstruct.inference import load_reconstruction_bundle, predict_from_partial
from moebench.router.inference import predict_expert_scores, select_top_k_from_probs


def _suite_experiment_token(suite_full: str) -> str:
    """Filesystem-safe token from e.g. ``pts/nvidia-gpu-compute``."""
    return safe_session_tag(str(suite_full).replace("/", "_"))


def _load_router(path: Path, auto_install: bool) -> dict[str, Any]:
    import pickle

    if path.suffix in (".pkl", ".pickle"):
        return pickle.load(open(path, "rb"))
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _maybe_pip(mod: str, auto_install: bool) -> None:
    if auto_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", mod])


def executed_rows_from_export(export: dict[str, Any], test_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for tid in test_ids:
        v = primary_value_from_export(export, tid)
        ts = primary_time_from_pts_export(export, tid)
        if v is None or ts is None:
            missing.append(tid)
            continue
        out.append({"test_id": tid, "value": float(v), "time_s": float(ts)})
    return out, missing


def _profile_family(test_id: str) -> str:
    """
    Normalize ``pts/<name>-<ver>`` into family key ``pts/<name>``.

    This lets us align reconstruction-model test ids against newer/older profile
    versions present in a full export (e.g. ``pts/openssl-3.6.0`` vs ``pts/openssl-4.0.0``).
    """
    tid = str(test_id or "")
    if not tid:
        return tid
    if "/" in tid:
        pfx, rest = tid.split("/", 1)
    else:
        pfx, rest = "", tid
    if "-" not in rest:
        return tid
    family = rest.rsplit("-", 1)[0]
    return f"{pfx}/{family}" if pfx else family


def _available_ids_from_export(export: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for h, robj in (export.get("results") or {}).items():
        tid = str(robj.get("identifier") or h)
        if tid and tid not in ids:
            ids.append(tid)
    return ids


def _align_recon_ids_to_export(recon_test_ids: list[str], export: dict[str, Any]) -> tuple[list[str], dict[str, str], list[str]]:
    """
    Return aligned ids for target extraction:
    - exact id when present
    - otherwise same family id in export when unique
    """
    available = _available_ids_from_export(export)
    avail_set = set(available)
    fam_to_ids: dict[str, list[str]] = {}
    for tid in available:
        fam_to_ids.setdefault(_profile_family(tid), []).append(tid)

    aligned: list[str] = []
    mapped: dict[str, str] = {}
    missing: list[str] = []
    for tid in recon_test_ids:
        if tid in avail_set:
            aligned.append(tid)
            mapped[tid] = tid
            continue
        fam = _profile_family(tid)
        cands = fam_to_ids.get(fam) or []
        if len(cands) == 1:
            aligned.append(cands[0])
            mapped[tid] = cands[0]
        else:
            missing.append(tid)
    return aligned, mapped, missing


def _suite_mean_from_export_ids(
    export: dict[str, Any],
    test_ids: list[str],
    *,
    suite_target_mode: str = "arithmetic_mean",
) -> tuple[float | None, list[str], list[str]]:
    """
    Compute suite mean on available profiles only.

    Returns (mean_or_none, used_ids, missing_ids).
    """
    vals: list[float] = []
    used: list[str] = []
    missing: list[str] = []
    for tid in test_ids:
        v = primary_value_from_export(export, tid)
        if v is None:
            missing.append(tid)
            continue
        vals.append(float(v))
        used.append(tid)
    if not vals:
        return None, used, missing
    if suite_target_mode == "logmean":
        mean = float(math.expm1(sum(math.log1p(max(0.0, v)) for v in vals) / len(vals)))
        return mean, used, missing
    return float(sum(vals) / len(vals)), used, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sudo-for-xi",
        action="store_true",
        help="Re-run script with sudo -E **only** for collect_all (then drops root for PTS)",
    )
    ap.add_argument("--router-model", type=str, default="dataset/pts_router/router_gnn.pt")
    ap.add_argument("--reconstruct-model", type=str, default="dataset/pts_models/reconstruct_xgb.pkl")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--pts-bin", type=str, default=None)
    ap.add_argument("--pts-root", type=str, default=None)
    ap.add_argument("--pts-mode", type=str, default="run", choices=("run", "batch-run"))
    ap.add_argument(
        "--suite-full",
        type=str,
        default="cpu",
        help="Full baseline suite id for PTS (e.g. cpu, pts/nvidia-gpu-compute)",
    )
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--no-ebpf", action="store_true", default=True, help="Skip eBPF (default on; no root)")
    ap.add_argument("--dataset-root", type=str, default="dataset")
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("-o", "--output", type=str, default="")
    ap.add_argument(
        "--min-partial-tests",
        type=int,
        default=1,
        help="Minimum successful partial tests required to run reconstruction (default: 1)",
    )
    ap.add_argument(
        "--min-ground-truth-tests",
        type=int,
        default=2,
        help="Minimum number of full-export profiles required for ground-truth suite mean (default: 2)",
    )
    ap.add_argument(
        "--ground-truth-from-dataset",
        action="store_true",
        help="Use collected full-suite PTS export from dataset/ as ground truth (skip live full run)",
    )
    ap.add_argument(
        "--machine",
        type=str,
        default="",
        help="With --ground-truth-from-dataset: host slug for session filter (default: current hostname)",
    )
    ap.add_argument(
        "--ground-truth-run",
        type=str,
        default="",
        help="With --ground-truth-from-dataset: explicit run JSON path (overrides --machine lookup)",
    )
    ap.add_argument("--auto-install", action="store_true")
    args = ap.parse_args()

    if args.sudo_for_xi and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo-for-xi"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + forwarded
        return subprocess.call(cmd)

    repo = REPO_ROOT
    router_fp = (repo / args.router_model).resolve() if not Path(args.router_model).is_absolute() else Path(args.router_model)
    recon_fp = (repo / args.reconstruct_model).resolve() if not Path(args.reconstruct_model).is_absolute() else Path(args.reconstruct_model)
    if not router_fp.is_file():
        print(f"Router not found: {router_fp}", file=sys.stderr)
        return 2
    if not recon_fp.is_file():
        print(f"Reconstruction bundle not found: {recon_fp}", file=sys.stderr)
        return 2

    pts_root = Path(args.pts_root).resolve() if args.pts_root else default_pts_install_root()
    pts_exe = _which_pts(args.pts_bin, pts_root if pts_root.is_dir() else None)

    session = args.session or safe_session_tag(
        f"{os.uname().nodename.split('.')[0]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    suite_tok = _suite_experiment_token(args.suite_full)
    out_dir = Path(args.dataset_root).resolve() / "experiments" / f"pts_{suite_tok}_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output) if args.output else out_dir / "experiment_pts_router_reconstruct_vs_full.json"

    # ---- xi (no root after optional sudo re-exec)
    t0 = time.perf_counter()
    xi = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=False,
        mem_mb=args.mem_mb,
    )
    t_xi = time.perf_counter() - t0

    router_meta = _load_router(router_fp, args.auto_install)
    if router_meta.get("benchmark") != "phoronix":
        print("Warning: router meta benchmark is not 'phoronix'; expert ids may mismatch.", file=sys.stderr)
    r_ps = router_meta.get("pts_suite")
    if r_ps and r_ps != args.suite_full:
        print(
            f"Warning: router pts_suite={r_ps!r} differs from this run --suite-full={args.suite_full!r}.",
            file=sys.stderr,
        )
    top_k = int(args.top_k) if args.top_k is not None else int(router_meta.get("top_k", 3))
    scores, probs, expert_ids, expert_test_ids = predict_expert_scores(router_meta, xi)
    _, selected_test_ids = select_top_k_from_probs(probs, expert_ids, expert_test_ids, top_k)

    recon = load_reconstruction_bundle(recon_fp)
    b_ps = recon.get("pts_suite")
    if b_ps and b_ps != args.suite_full:
        print(
            f"Warning: reconstruction bundle pts_suite={b_ps!r} differs from --suite-full={args.suite_full!r}.",
            file=sys.stderr,
        )
    test_ids = list(recon.get("test_ids") or [])
    suite_target_mode = str(recon.get("pts_suite_target") or "arithmetic_mean")
    if not test_ids:
        print("Reconstruction bundle missing test_ids", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_partial = f"moebench_pts_exp_partial_{session}_{stamp}"
    name_partial = pts_clean_save_name(safe_session_tag(raw_partial))
    env_p = pts_subprocess_env()
    env_p["TEST_RESULTS_NAME"] = name_partial
    env_p["TEST_RESULTS_IDENTIFIER"] = name_partial
    env_p["TEST_RESULTS_DESCRIPTION"] = f"MoEBench partial Top-{top_k}"

    pts_warn_if_low_nvidia_vram_for_opencl_suite(args.suite_full)
    cmd_partial = _pts_argv_as_installing_user(pts_exe, [args.pts_mode, *selected_test_ids])
    print("+", " ".join(cmd_partial), file=sys.stderr)
    t1 = time.perf_counter()
    rc = subprocess.call(cmd_partial, env=env_p)
    t_partial = time.perf_counter() - t1
    if rc != 0:
        print(f"Partial PTS failed rc={rc}", file=sys.stderr)
        return rc

    raw_path = out_dir / "partial_export_pts.json"
    partial_export = _export_result_json(pts_exe, name_partial, raw_path)

    executed, missing_partial = executed_rows_from_export(partial_export, list(selected_test_ids))
    if missing_partial:
        print(
            "Warning: missing value/time for some selected partial tests: "
            + ", ".join(missing_partial),
            file=sys.stderr,
        )
    if len(executed) < int(args.min_partial_tests):
        print(
            "Too few successful partial tests for reconstruction "
            f"(got={len(executed)}, min-required={args.min_partial_tests}).",
            file=sys.stderr,
        )
        return 2
    pred = predict_from_partial(recon, xi, executed, return_uncertainty=False)

    ground_truth_source = "live_full_run"
    ground_truth_run_path: str | None = None
    if args.ground_truth_from_dataset:
        machine = resolve_training_machine(args.machine or None)
        if args.ground_truth_run.strip():
            gt_path = Path(args.ground_truth_run).resolve()
        else:
            gt_path = latest_pts_run_path_for_machine(
                Path(args.dataset_root),
                machine=machine,
                pts_suite=args.suite_full,
            )
        with open(gt_path, encoding="utf-8") as f:
            ds_gt = json.load(f)
        full_export = (ds_gt.get("yi") or {}).get("pts_export") or {}
        if not full_export:
            print(f"Ground-truth run missing yi.pts_export: {gt_path}", file=sys.stderr)
            return 2
        t_full_val = full_suite_wall_seconds_pts(ds_gt, test_ids=tuple(test_ids))
        t_full = float(t_full_val) if t_full_val is not None else 0.0
        rc2 = 0
        name_full = f"dataset:{gt_path.name}"
        full_path = gt_path
        ground_truth_source = "dataset_collection"
        ground_truth_run_path = str(gt_path)
        print(
            f"[ground-truth] using collected run {gt_path} (skip live full PTS; t_full≈{t_full:.1f}s)",
            file=sys.stderr,
        )
    else:
        raw_full = f"moebench_pts_exp_full_{session}_{stamp}"
        name_full = pts_clean_save_name(safe_session_tag(raw_full))
        env_f = pts_subprocess_env()
        env_f["TEST_RESULTS_NAME"] = name_full
        env_f["TEST_RESULTS_IDENTIFIER"] = name_full
        env_f["TEST_RESULTS_DESCRIPTION"] = "MoEBench full cpu baseline"
        cmd_full = _pts_argv_as_installing_user(pts_exe, [args.pts_mode, args.suite_full])
        print("+", " ".join(cmd_full), file=sys.stderr)
        t2 = time.perf_counter()
        rc2 = subprocess.call(cmd_full, env=env_f)
        t_full = time.perf_counter() - t2
        if rc2 != 0:
            print(f"Full PTS failed rc={rc2}", file=sys.stderr)
            return rc2

        full_path = out_dir / "full_export_pts.json"
        full_export = _export_result_json(pts_exe, name_full, full_path)

    aligned_ids, id_map, missing_ids = _align_recon_ids_to_export(test_ids, full_export)
    if missing_ids:
        print(
            "Warning: some reconstruction test_ids are absent from full export and cannot be family-mapped: "
            + ", ".join(missing_ids),
            file=sys.stderr,
        )
    if len(aligned_ids) < 2:
        print(
            "Could not extract targets from full export: fewer than 2 aligned test ids after matching.",
            file=sys.stderr,
        )
        return 2

    suite_true, used_ids, missing_after_align = _suite_mean_from_export_ids(
        full_export,
        aligned_ids,
        suite_target_mode=suite_target_mode,
    )
    if suite_true is None or len(used_ids) < int(args.min_ground_truth_tests):
        print(
            "Could not extract reliable ground truth from full export after alignment "
            f"(used={len(used_ids)}, min-required={args.min_ground_truth_tests}). "
            "check full_export_pts.json completeness.",
            file=sys.stderr,
        )
        return 2
    suite_pred = float(pred["suite_index"])
    err = abs(suite_pred - suite_true)
    rel = err / max(abs(suite_true), 1e-9)

    report: dict[str, Any] = {
        "schema": "moebench.phoronix.experiment_router_reconstruct.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "pts_suite_full": args.suite_full,
        "experiment_dir_token": suite_tok,
        "router_model": str(router_fp),
        "reconstruct_model": str(recon_fp),
        "top_k": top_k,
        "selected_test_ids": list(selected_test_ids),
        "times_s": {
            "xi": t_xi,
            "pts_partial": t_partial,
            "pts_full": t_full,
            "pts_partial_plus_full": t_partial + t_full,
        },
        "suite_mean": {"predicted_from_partial": suite_pred, "ground_truth_full": suite_true, "abs_error": err, "relative_error": rel},
        "suite_target_mode": suite_target_mode,
        "predicted_subtests": pred.get("subtest_index"),
        "pts_result_ids": {"partial": name_partial, "full": name_full},
        "ground_truth_source": ground_truth_source,
        "ground_truth_run_path": ground_truth_run_path,
        "target_alignment": {
            "requested_test_ids": list(test_ids),
            "aligned_test_ids_for_ground_truth": list(aligned_ids),
            "used_test_ids_for_ground_truth": list(used_ids),
            "missing_after_alignment_in_full_export": list(missing_after_align),
            "ground_truth_coverage_ratio": (len(used_ids) / max(len(aligned_ids), 1)),
            "id_map_requested_to_aligned": id_map,
            "missing_unaligned_test_ids": missing_ids,
        },
        "partial_execution": {
            "selected_test_ids": list(selected_test_ids),
            "executed_success_test_ids": [str(e.get("test_id")) for e in executed],
            "missing_in_partial_export": missing_partial,
            "partial_execution_coverage_ratio": (len(executed) / max(len(selected_test_ids), 1)),
        },
        "notes": "With --sudo-for-xi, only xi used root; PTS uses SUDO_USER. If the whole script runs as root, PTS still delegates to SUDO_USER with correct HOME.",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
