#!/usr/bin/env python3
"""Online hybrid prediction after sysctl-tuned UnixBench run."""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moebench import collect_all
from moebench.hybrid.eval import evaluate_hybrid_online, load_router_meta
from moebench.probe.inference import load_probe_bundle
from moebench.reconstruct.inference import load_reconstruction_bundle

ROOT = REPO
GT_PATH = ROOT / "dataset/experiments/aces-System-Product-Name/tuned_sysctl_unixbench_20260611.json"
MODELS = ROOT / "dataset/experiments/hybrid_grid_unixbench_aces-System-Product-Name_20260610T115239Z/trained_models"


def main() -> int:
    router = load_router_meta(MODELS / "router_lgbm.pkl")
    recon = load_reconstruction_bundle(MODELS / "recon_lgbm.pkl")
    probe = load_probe_bundle(ROOT / "dataset/models/aces-System-Product-Name/probe_unixbench_lgbm.pkl")

    gt_ds = json.load(open(GT_PATH, encoding="utf-8"))
    t0 = time.perf_counter()
    xi = collect_all(enable_ebpf=True)
    t_xi = time.perf_counter() - t0

    report = evaluate_hybrid_online(
        xi=xi,
        router_meta=router,
        recon_bundle=recon,
        probe_bundle=probe,
        ground_truth_ds=gt_ds,
        ground_truth_run=GT_PATH,
        enable_ebpf=True,
        xi_wall_s=t_xi,
    )
    report["sysctl_tuning"] = {
        "kernel.sched_autogroup_enabled": "0",
        "vm.swappiness": "1",
        "vm.dirty_ratio": "5",
        "vm.dirty_background_ratio": "2",
    }
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["unixbench_run_json"] = str(GT_PATH)

    out = ROOT / "dataset/experiments/aces-System-Product-Name/tuned_sysctl_comparison_20260611.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    actual = float(report["ground_truth_suite"])
    pred = float(report["reconstruction"]["predicted_suite"])
    err = abs(pred - actual)
    rel = err / max(abs(actual), 1e-9)
    print("\n========== COMPARISON ==========")
    print(f"Actual UnixBench score : {actual:.1f}")
    print(f"MoEBench predicted     : {pred:.4f}")
    print(f"Absolute error         : {err:.4f}")
    print(f"Relative error         : {rel * 100:.4f}%")
    print(f"Router Top-3           : {report['router']['selected_test_ids']}")
    print(f"Hybrid wall time       : {report['timing_seconds']['hybrid_wall']:.1f}s")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
