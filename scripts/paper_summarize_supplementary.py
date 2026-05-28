#!/usr/bin/env python3
"""Summarize paper supplementary JSON (Top-K / policy / xi ablation) into ranking tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moebench.paper_eval.summarize import (
    summarize_policy_report,
    summarize_topk_report,
    summarize_xi_ablation_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        type=str,
        choices=("topk", "policy", "xi_ablation"),
        required=True,
    )
    ap.add_argument("--input", type=str, required=True, help="paper_*.json from paper_reconstruct_cv_extras.py")
    ap.add_argument("-o", "--output", type=str, default="", help="Write summary JSON (default: stdout only)")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        report = json.load(f)

    if args.mode == "topk":
        summary = summarize_topk_report(report)
    elif args.mode == "policy":
        summary = summarize_policy_report(report)
    else:
        summary = summarize_xi_ablation_report(report)

    txt = json.dumps(summary, indent=2, ensure_ascii=False)
    print(txt)
    if args.output.strip():
        outp = Path(args.output).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(txt + "\n", encoding="utf-8")
        print(f"Wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
