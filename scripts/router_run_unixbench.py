#!/usr/bin/env python3
"""Run UnixBench with router-selected expert subset.

Pipeline:
  1) Collect xi (static + dynamic) via moebench.collect_all()
  2) Load trained router model checkpoint
  3) Predict score per expert and convert to probabilities (softmax)
  4) Select Top-K experts and execute UnixBench subset via `perl Run <tests...>`
  5) Parse the generated report and save a unified JSON output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
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
from moebench.router.feature_vectorizer import XiVectorizer
from moebench.unixbench.report_parser import parse_report_text


def _ensure_module(module_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError as e:
        raise ImportError(f"Missing dependency '{module_name}' in current Python environment.") from e


def _maybe_auto_install(module_name: str, auto_install: bool) -> None:
    if not auto_install:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", module_name])


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sudo", action="store_true", help="Re-run this command with sudo -E")
    ap.add_argument("--model", type=str, required=True, help="Path to router model checkpoint")
    ap.add_argument("--top-k", type=int, default=None, help="Override top-k stored in model")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--proc-sample-s", type=float, default=0.5)
    ap.add_argument("--mem-mb", type=int, default=64)
    ap.add_argument("--no-ebpf", action="store_true")
    ap.add_argument("--dataset-root", type=str, default="dataset", help="Root for router run JSON output")
    ap.add_argument("--session", type=str, default=None, help="Subfolder tag under dataset-root")
    ap.add_argument("--unixbench-root", type=str, default=None, help="UnixBench directory; default under repo")
    ap.add_argument("--label-transform", type=str, default=None, help="If needed, override label_transform from model")
    ap.add_argument("--enable-forward-stdout", action="store_true", help="(default) keep UnixBench stdout visible")
    ap.add_argument("--auto-install", action="store_true", help="Auto-install missing Python deps in current env (e.g., lightgbm)")
    ap.add_argument(
        "--copies",
        type=int,
        default=0,
        help="UnixBench -c copies value. Default 0 means auto: min(32, os.cpu_count()). (Speeds up by running only one block.)",
    )
    args = ap.parse_args()

    if args.sudo and os.geteuid() != 0:
        forwarded = [a for a in sys.argv[1:] if a != "--sudo"]
        cmd = ["sudo", "-E", sys.executable, str(Path(__file__).resolve())] + forwarded
        raise SystemExit(subprocess.call(cmd))

    repo_root = Path(__file__).resolve().parents[1]
    unixbench_root = Path(args.unixbench_root).resolve() if args.unixbench_root else repo_root / "byte-unixbench" / "UnixBench"
    unixbench_root = unixbench_root.resolve()
    result_dir = unixbench_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    session_tag = args.session
    if not session_tag:
        host = os.uname().nodename.split(".")[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_tag = f"{host}_{ts}"
        # sanitize
        session_tag = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in session_tag)

    ds_out_root = Path(args.dataset_root).resolve() / "unixbench_router" / session_tag
    ds_out_root.mkdir(parents=True, exist_ok=True)

    out_path = ds_out_root / f"run-router.json"
    # avoid overwrite
    if out_path.exists():
        out_path = ds_out_root / f"run-router-{int(time.time())}.json"

    # 1) collect xi
    xi_collect = collect_all(
        warmup_s=args.warmup_s,
        proc_sample_s=args.proc_sample_s,
        enable_ebpf=not args.no_ebpf,
        mem_mb=args.mem_mb,
    )
    xi = xi_collect  # keep full structure (static+dynamic)

    # 2) load model
    model_fp = Path(args.model)
    model_type = None
    router_meta: dict[str, Any] = {}

    if os.geteuid() == 0 and not os.environ.get("CONDA_PREFIX"):
        print(
            "Warning: running as root outside conda env. If model was trained in conda, "
            "prefer `conda activate <env> && python scripts/router_run_unixbench.py ...`.",
            file=sys.stderr,
        )

    if model_fp.suffix in (".pkl", ".pickle", ".dat"):
        try:
            with open(model_fp, "rb") as f:
                router_meta = pickle.load(f)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "unknown"
            if missing == "lightgbm":
                try:
                    _maybe_auto_install("lightgbm", args.auto_install)
                    with open(model_fp, "rb") as f:
                        router_meta = pickle.load(f)
                except Exception:
                    raise RuntimeError(
                        "Model checkpoint requires lightgbm, but current python env cannot import it.\n"
                        "Fix: activate your miniconda env and run:\n"
                        "  conda activate <env>\n"
                        "  python scripts/router_run_unixbench.py --model ...\n"
                        "Or rerun with --auto-install in a writable env."
                    ) from e
            else:
                raise
        model_type = router_meta.get("model_type")
    else:
        # allow torch checkpoints
        try:
            import torch

            router_meta = torch.load(model_fp, map_location="cpu")
            model_type = router_meta.get("model_type")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {model_fp}. {e}") from e

    expert_ids = router_meta["expert_ids"]
    expert_test_ids = router_meta["expert_test_ids"]
    stored_top_k = int(router_meta.get("top_k", 3))
    top_k = int(args.top_k) if args.top_k is not None else stored_top_k
    top_k = max(1, min(top_k, len(expert_ids)))

    label_transform = args.label_transform or router_meta.get("label_transform") or "none"

    # 3) vectorize xi
    vec = XiVectorizer()
    xi_vec = vec.transform(xi)
    xi_dim = len(vec.feature_names)
    n_experts = len(expert_ids)

    X_rows = []
    for ei in range(n_experts):
        onehot = [0.0] * n_experts
        onehot[ei] = 1.0
        X_rows.append(list(xi_vec) + onehot)

    # 4) predict per expert
    scores: list[float] = []
    if model_type == "lightgbm":
        import numpy as np

        X_np = np.asarray(X_rows, dtype=np.float32)
        ranker = router_meta["ranker"]
        preds = ranker.predict(X_np)
        scores = [float(x) for x in preds]
    elif model_type == "mlp":
        import numpy as np
        import torch
        import torch.nn as nn

        in_dim = xi_dim + n_experts
        hidden = int(router_meta.get("mlp_hidden", 64))
        net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        net.load_state_dict(router_meta["state_dict"])
        net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(X_rows, dtype=np.float32))
            out = net(x_t).view(-1).tolist()
        scores = [float(v) for v in out]
    else:
        raise RuntimeError(f"Unknown model_type: {model_type}")

    probs = _softmax(scores)

    # 5) select Top-K
    ranked_idx = sorted(range(n_experts), key=lambda i: probs[i], reverse=True)[:top_k]
    selected_experts = [expert_ids[i] for i in ranked_idx]
    selected_test_ids = [expert_test_ids[i] for i in ranked_idx]

    # 6) run UnixBench subset; keep terminal output visible
    # Fix UB report name for parsing
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ub_base_name = f"moebench_router_{session_tag}_{stamp}".replace(":", "-")
    ub_env = os.environ.copy()
    ub_env["UB_OUTPUT_FILE_NAME"] = ub_base_name
    ub_env["UB_RESULTDIR"] = str(result_dir)

    run_script = unixbench_root / "Run"
    cpu_count = os.cpu_count() or 1
    copies = args.copies if args.copies and args.copies > 0 else min(32, int(cpu_count))
    cmd = ["perl", str(run_script), "-c", str(copies)] + selected_test_ids
    rc = subprocess.call(cmd, cwd=str(unixbench_root), env=ub_env)
    if rc != 0:
        raise RuntimeError(f"UnixBench Run failed with rc={rc}")

    report_path = result_dir / ub_base_name
    report_txt = report_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_report_text(report_txt)
    runs = parsed.get("runs") or []
    # Prefer numeric parallel copies block (often 32 on your machines).
    def key(rb: dict[str, Any]) -> tuple[int, int]:
        pc = rb.get("parallel_copies")
        if pc == 32:
            return (0, 0)
        if isinstance(pc, int):
            return (1, pc)
        if pc is None:
            return (2, 10**9)
        return (3, 10**9)

    parsed_run = sorted(runs, key=key)[0] if runs else {}

    executed_tests: list[dict[str, Any]] = []
    tests_map = (parsed_run.get("tests") or {}) if parsed_run else {}
    for tid in selected_test_ids:
        tinfo = tests_map.get(tid)
        if not tinfo:
            executed_tests.append({"test_id": tid, "missing": True})
            continue
        executed_tests.append(
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

    suite_index = parsed_run.get("system_benchmarks_index_score")

    out_obj = {
        "schema": "moebench.unixbench_router.run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": {"tag": session_tag, "top_k": top_k, "model_path": str(model_fp)},
        "xi": xi_collect,
        "router_prediction": {
            "model_type": model_type,
            "label_transform": label_transform,
            "expert_ids": expert_ids,
            "expert_test_ids": expert_test_ids,
            "scores": dict(zip(expert_ids, scores)),
            "probabilities": dict(zip(expert_ids, probs)),
            "selected_experts": selected_experts,
            "selected_test_ids": selected_test_ids,
        },
        "unixbench": {
            "command": cmd,
            "returncode": rc,
            "result_files": {
                "report": str(report_path),
                "log": str(report_path) + ".log",
                "html": str(report_path) + ".html",
            },
            "ub_env": {"UB_OUTPUT_FILE_NAME": ub_base_name, "UB_RESULTDIR": str(result_dir)},
        },
        "executed": {
            "suite_index_score_for_partial_set": suite_index,
            "executed_tests": executed_tests,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote router run JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

