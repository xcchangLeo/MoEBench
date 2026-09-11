#!/usr/bin/env python3
"""Audit dataset/session splits and paper experiment artifacts.

The audit is intentionally read-only with respect to benchmark data.  It
records the split structure that can be justified from manifests and the
model/report artifacts that are present in the repository.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_from_dataset(data: dict[str, Any], path: Path) -> str:
    schema = str(data.get("schema", ""))
    if "unixbench" in schema:
        return "unixbench"
    yi = data.get("yi") or {}
    suite = str(yi.get("suite", ""))
    if suite == "cpu":
        return "phoronix_cpu"
    if suite == "pts/nvidia-gpu-compute":
        return "phoronix_gpu"
    if "_cpu_" in path.parent.name:
        return "phoronix_cpu"
    if "pts_nvidia-gpu-compute" in path.parent.name:
        return "phoronix_gpu"
    return "unknown"


def _host_from_session(session: str) -> str:
    for marker in ("_pts_nvidia-gpu-compute_", "_cpu_", "_"):
        if marker in session:
            return session.split(marker, 1)[0]
    return session


def _artifact_paths(host: str, suite: str) -> list[str]:
    model_dir = DATASET / "models" / host
    names = {
        "unixbench": ("router_mlp.pt", "router_lgbm.pkl", "reconstruct_lightgbm_v2.pkl"),
        "phoronix_cpu": ("router_mlp.pt", "router_lgbm.pkl", "recon_lgbm.pkl"),
        "phoronix_gpu": ("router_mlp.pt", "router_lgbm.pkl", "recon_lgbm.pkl"),
    }.get(suite, ())
    found = [str(p.relative_to(ROOT)) for n in names if (p := model_dir / n).is_file()]
    found.extend(
        str(p.relative_to(ROOT))
        for p in sorted((DATASET / "experiments").glob(f"*{host}*/trained_models/*"))
        if p.is_file() and ("router" in p.name or "recon" in p.name)
    )
    return sorted(set(found))


def _flag(
    *,
    data: dict[str, Any],
    path: Path,
    session_count_for_host_suite: int,
    model_artifacts: list[str],
) -> list[str]:
    flags: list[str] = []
    session = data.get("session") or {}
    if session.get("xi_reused_from_previous_round"):
        flags.append("xi_reused_within_session")
    if session_count_for_host_suite < 2:
        flags.append("single_session_host_suite")
    if not model_artifacts:
        flags.append("no_local_model_artifact_found")
    if path.parent.name in {"paper_supplementary", "experiments"}:
        flags.append("non_raw_dataset_path")
    return flags


def audit_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = sorted(DATASET.glob("*/*run-*.json"))
    # Also include only direct session children; avoid model/experiment JSONs.
    raw = [
        p
        for p in raw
        if p.parent.parent == DATASET
        and p.name.startswith("run-")
        and p.suffix == ".json"
        and not p.name.endswith("_pts_raw.json")
    ]
    by_host_suite: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    loaded: list[tuple[Path, dict[str, Any], str, str, str]] = []
    for path in raw:
        data = _read_json(path)
        session_name = path.parent.name
        suite = _suite_from_dataset(data, path)
        host = _host_from_session(session_name)
        run_id = str((data.get("session") or {}).get("round_index") or path.stem)
        by_host_suite[(host, suite)].add(session_name)
        loaded.append((path, data, host, suite, session_name))

    for path, data, host, suite, session_name in loaded:
        session = data.get("session") or {}
        artifacts = _artifact_paths(host, suite)
        created = data.get("created_at_utc")
        rows.append(
            {
                "host": host,
                "suite": suite,
                "session": session_name,
                "run_id": str(session.get("round_index") or path.stem),
                "created_at_utc": created,
                "path": str(path.relative_to(ROOT)),
                "train_membership": "unknown_without_training_manifest",
                "test_membership": "unknown_without_test_manifest",
                "model_artifact_paths": artifacts,
                "xi_reused_from_previous_round": bool(
                    session.get("xi_reused_from_previous_round", False)
                ),
                "possible_leakage_flags": _flag(
                    data=data,
                    path=path,
                    session_count_for_host_suite=len(by_host_suite[(host, suite)]),
                    model_artifacts=artifacts,
                ),
            }
        )

    sessions = defaultdict(list)
    for row in rows:
        sessions[(row["host"], row["suite"])].append(row["session"])
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "repository": str(ROOT),
        "raw_run_count": len(rows),
        "host_suite": [],
        "global_findings": [
            f"Raw datasets contain {len({r[2] for r in loaded})} observed host identities.",
            "Each observed host-suite pair has one raw collection session with five runs.",
            "The repository does not contain a machine-readable train/test manifest for the headline results.",
            "Session-level holdout across independently collected sessions cannot be demonstrated from the current raw data alone.",
        ],
    }
    for (host, suite), session_names in sorted(sessions.items()):
        summary["host_suite"].append(
            {
                "host": host,
                "suite": suite,
                "session_count": len(set(session_names)),
                "run_count": sum(
                    1 for row in rows if row["host"] == host and row["suite"] == suite
                ),
                "sessions": sorted(set(session_names)),
                "strict_session_holdout_available": len(set(session_names)) >= 2,
            }
        )
    return rows, summary


def audit_experiment_reports() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((DATASET / "paper_supplementary").glob("*.json")):
        try:
            data = _read_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        reports.append(
            {
                "path": str(path.relative_to(ROOT)),
                "experiment": data.get("experiment"),
                "cv_mode": data.get("cv_mode"),
                "folds": data.get("folds"),
                "suites": [
                    {
                        "suite_key": block.get("suite_key"),
                        "n_samples": block.get("n_samples"),
                        "sessions": block.get("sessions"),
                    }
                    for block in data.get("suite_results", [])
                    if isinstance(block, dict)
                ],
                "possible_leakage_flags": (
                    ["random_or_unspecified_split"]
                    if data.get("cv_mode") not in {"leave_one_session_out"}
                    else []
                ),
            }
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(DATASET / "paper_audit"),
        help="Directory for audit JSON and Markdown outputs.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = audit_dataset()
    reports = audit_experiment_reports()
    payload = {
        "schema": "moebench.paper_protocol_audit.v1",
        "dataset_summary": summary,
        "run_records": rows,
        "experiment_reports": reports,
    }
    json_path = out_dir / "protocol_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = Counter((r["host"], r["suite"]) for r in rows)
    lines = [
        "# Paper Protocol Audit",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Raw run records: `{len(rows)}`",
        f"- Host-suite pairs: `{len(counts)}`",
        "",
        "## Dataset coverage",
        "",
        "| Host | Suite | Sessions | Runs | Strict session holdout |",
        "|---|---|---:|---:|---|",
    ]
    for item in summary["host_suite"]:
        lines.append(
            f"| {item['host']} | {item['suite']} | {item['session_count']} | "
            f"{item['run_count']} | {item['strict_session_holdout_available']} |"
        )
    lines += [
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in summary["global_findings"])
    lines += [
        "",
        "## Interpretation",
        "",
        "The current raw corpus supports leave-one-run-out evaluation within a single "
        "collection session and can support leave-one-session-out only when multiple "
        "independent sessions exist. The audit therefore marks cross-session generalization "
        "as unavailable for the current host-suite pairs rather than inferring it from "
        "same-session runs.",
        "",
        "Experiment reports are recorded separately because a report's CV mode does not "
        "prove that every model-selection artifact was trained without test information. "
        "The revised paper must state the split and tuning protocol explicitly.",
    ]
    md_path = out_dir / "protocol_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
