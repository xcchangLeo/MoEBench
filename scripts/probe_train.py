#!/usr/bin/env python3
"""Train probe → subtest index model from probe_dataset.json."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.dataset_machines import ensure_machine_output_dir
from moebench.probe.model_train import train_probe_bundle


def _maybe_pip(auto: bool, pkg: str) -> None:
    if auto:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dataset", type=str, required=True, help="probe_dataset.json from probe_collect.py")
    ap.add_argument("--model-out", type=str, required=True, help="Output .pkl bundle")
    ap.add_argument(
        "--model-type",
        type=str,
        choices=("lightgbm", "xgboost"),
        default="lightgbm",
    )
    ap.add_argument("--auto-install", action="store_true")
    ap.add_argument(
        "--suite-aggregate",
        type=str,
        choices=("geomean_index", "mean_index"),
        default="geomean_index",
    )
    args = ap.parse_args()

    with open(args.probe_dataset, encoding="utf-8") as f:
        ds = json.load(f)

    if args.model_type == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            _maybe_pip(args.auto_install, "lightgbm")
    elif args.model_type == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            _maybe_pip(args.auto_install, "xgboost")

    try:
        bundle = train_probe_bundle(
            ds,
            model_type=args.model_type,
            suite_aggregate=args.suite_aggregate,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    out = ensure_machine_output_dir(Path(args.model_out))
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "train_rows": bundle.get("train_rows"),
                "estimator_mode": bundle.get("estimator_mode"),
                "label_transform": bundle.get("label_transform"),
                "features": len(bundle.get("feature_names") or []),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
