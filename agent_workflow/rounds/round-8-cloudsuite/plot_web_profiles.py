#!/usr/bin/env python3
"""Draw full-reference curves from the strict CloudSuite primary sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ("fpm4", "fpm8")
LOADS = (20, 40, 60, 80, 100)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    sessions = tuple(int(session) for session in analysis["primary_sessions"])
    if len(sessions) != 4 or len(set(sessions)) != 4:
        raise ValueError("expected four distinct primary sessions")
    if not analysis["primary_analysis"].get("analysis_complete"):
        raise ValueError("strict primary analysis is incomplete")
    manifests = []
    for entry in analysis["input_manifests"]:
        manifest = Path(entry["path"])
        actual_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual_sha256 != entry["sha256"]:
            raise ValueError(f"input manifest changed after analysis: {manifest}")
        manifests.append(manifest)
    rows = {}
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["phase"] != "full" or int(record["session"]) not in sessions:
                continue
            key = (record["config"], int(record["session"]), int(record["scale"]))
            if key in rows:
                raise ValueError(f"duplicate full-reference cell: {key}")
            validation = json.loads(Path(record["summary_validation_json"]).read_text(encoding="utf-8"))
            if (
                int(record.get("exit_code", 1)) != 0
                or int(record.get("client_log_protocol_error", 1)) != 0
                or not validation.get("summary_valid")
                or not validation.get("mix_valid")
                or int(validation.get("operation_failures", 0)) != 0
                or not validation.get("throughput_usable")
            ):
                raise ValueError(f"unusable full-reference cell: {key}")
            rows[key] = float(validation["throughput_ops_per_sec"])
    expected = {(config, session, load) for config in CONFIGS
                for session in sessions for load in LOADS}
    if set(rows) != expected:
        raise ValueError(f"full-reference cells differ: missing={expected - set(rows)}")

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    colors = {"fpm4": "#1f6f78", "fpm8": "#bb5a30"}
    markers = {"fpm4": "o", "fpm8": "s"}
    for config in CONFIGS:
        values = np.asarray([[rows[(config, session, load)] for load in LOADS]
                             for session in sessions], dtype=float)
        mean = values.mean(axis=0)
        label = "4 PHP workers" if config == "fpm4" else "8 PHP workers"
        ax.plot(LOADS, mean, label=label, color=colors[config],
                marker=markers[config], markersize=3.2, linewidth=1.55)
        ax.fill_between(LOADS, values.min(axis=0), values.max(axis=0),
                        color=colors[config], alpha=0.12, linewidth=0)
    ax.set_xlabel("Offered-load scale")
    ax.set_ylabel("Throughput (ops/s)")
    ax.set_xticks(LOADS)
    ax.set_xlim(17, 103)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=1)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    fig.tight_layout(pad=0.35)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
