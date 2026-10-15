#!/usr/bin/env python3
"""Build a UnixBench probe dataset from legacy and/or verified pre-run sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def run_key(value: str) -> tuple[str, str]:
    p = Path(value)
    return p.parent.name, p.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", type=Path, default=None)
    ap.add_argument("--paired-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--paired-only",
        action="store_true",
        help="Exclude legacy probe records whose pre-run ordering is not verified",
    )
    args = ap.parse_args()

    if not args.paired_only and args.legacy is None:
        ap.error("--legacy is required unless --paired-only is set")
    legacy = json.loads(args.legacy.read_text(encoding="utf-8")) if args.legacy else None
    samples = [] if args.paired_only or legacy is None else list(legacy.get("samples") or [])
    source_keys = {
        run_key(str(s.get("source_run"))) + (str(s.get("test_id")),)
        for s in samples
        if s.get("source_run") and s.get("test_id")
    }
    paired_sessions: list[str] = []
    excluded_sessions: list[str] = []
    paired_template: dict | None = None
    for path in sorted(args.paired_root.glob("*/paired_capture_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("probe_timing") != "before_full_run"
            or manifest.get("capture_status") not in (None, "complete")
        ):
            excluded_sessions.append(path.parent.name)
            continue
        probe_path = path.parent / "probe_dataset_unixbench.json"
        if not probe_path.is_file():
            continue
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        run_record = manifest.get("run") or {}
        run_path = Path(str(run_record.get("path") or ""))
        if not run_path.is_file():
            raise FileNotFoundError(f"Paired run file is missing for session {path.parent.name}: {run_path}")
        expected_sha = run_record.get("sha256")
        if expected_sha and sha256(run_path) != expected_sha:
            raise ValueError(f"Run SHA256 mismatch for session {path.parent.name}")
        probe_samples = list(probe.get("samples") or [])
        expected_key = run_key(run_path)
        found_keys = {run_key(str(sample.get("source_run"))) for sample in probe_samples if sample.get("source_run")}
        found_ids = {str(sample.get("test_id")) for sample in probe_samples if sample.get("test_id")}
        expected_ids = set(probe.get("test_ids") or [])
        if found_keys != {expected_key} or found_ids != expected_ids or len(probe_samples) != len(expected_ids):
            raise ValueError(
                f"Incomplete or mismatched probe rows for {path.parent.name}: "
                f"runs={found_keys}, ids={len(found_ids)}/{len(expected_ids)}, rows={len(probe_samples)}"
            )
        if paired_template is None:
            paired_template = dict(probe)
            paired_template["samples"] = []
        for sample in probe_samples:
            key = run_key(str(sample.get("source_run"))) + (str(sample.get("test_id")),)
            if key not in source_keys:
                samples.append(sample)
                source_keys.add(key)
        paired_sessions.append(path.parent.name)

    if not paired_sessions:
        raise ValueError(f"No complete before_full_run probe sessions found under {args.paired_root}")
    out = dict(paired_template if args.paired_only or legacy is None else legacy)
    out["samples"] = samples
    out["num_samples"] = len(samples)
    out["source_runs"] = sorted({str(s.get("source_run")) for s in samples if s.get("source_run")})
    out["composition"] = {
        "legacy_probe_dataset": str(args.legacy.resolve()) if args.legacy else None,
        "paired_only": bool(args.paired_only),
        "paired_pre_run_sessions": paired_sessions,
        "excluded_post_run_sessions": excluded_sessions,
        "selection_rule": (
            "complete sessions explicitly marked probe_timing=before_full_run"
            if args.paired_only
            else "legacy rows plus complete sessions explicitly marked probe_timing=before_full_run"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_samples": len(samples), "paired_sessions": paired_sessions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
