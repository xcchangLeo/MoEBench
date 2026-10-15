#!/usr/bin/env python3
"""Create a path-redacted, checksummed research artifact ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path


INCLUDE = (
    "scripts",
    "moebench",
    "dataset",
    "agent_workflow/rounds/round-3-crossfit-reconstructor",
    "agent_workflow/rounds/round-6-web-serving",
    "agent_workflow/rounds/round-7-m1-session",
    "agent_workflow/rounds/round-8-host-holdout",
    "agent_workflow/round-8-http-profile",
    "agent_workflow/rounds/round-8-cloudsuite",
    "agent_workflow/rounds/round-10-router-diagnostics",
    "agent_workflow/rounds/round-12-component-profile.json",
)
TEXT_EXTENSIONS = {".py", ".sh", ".md", ".txt", ".log", ".csv", ".yaml", ".yml", ".xml", ".tsv", ".conf", ".jsonl", ".html"}
HOME_PATH = re.compile(r"/home/[^/\s\"']+")
BINARY_HOME_PATH = re.compile(rb"/home/[^/\s\"']+")


def redact(value: str, root: Path) -> str:
    value = value.replace(str(root), ".")
    return HOME_PATH.sub("/home/anonymous", value)


def redact_json(value: object, root: Path) -> object:
    if isinstance(value, str):
        return redact(value, root)
    if isinstance(value, list):
        return [redact_json(item, root) for item in value]
    if isinstance(value, dict):
        return {redact(key, root): redact_json(item, root) for key, item in value.items()}
    return value


def eligible(relative: Path) -> bool:
    return (
        "__pycache__" not in relative.parts
        and "paper_audit" not in relative.parts
        and not (
            relative.parts[:3] == ("agent_workflow", "rounds", "round-8-cloudsuite")
            and "seed" in relative.parts
        )
        and relative.suffix != ".pyc"
        and relative.suffix != ".aria2"
    )


def stage_file(source: Path, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(json.dumps(redact_json(data, root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif source.suffix in TEXT_EXTENSIONS:
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            destination.write_bytes(source.read_bytes())
        else:
            destination.write_text(redact(content, root), encoding="utf-8")
    else:
        content = source.read_bytes()
        if BINARY_HOME_PATH.search(content):
            raise ValueError(f"Binary artifact contains an unredacted local path: {source}")
        destination.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="benchscout-anonymous-") as temp:
        stage = Path(temp)
        sources = [root / name for name in INCLUDE if (root / name).exists()]
        sources.append(root / "ARTIFACT_README.md")
        count = 0
        for source in sources:
            paths = (source,) if source.is_file() else source.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if not eligible(relative):
                    continue
                stage_file(path, stage / relative, root)
                count += 1

        (stage / "ANONYMIZATION.md").write_text(
            "# Artifact path redaction\n\n"
            "This review artifact replaces the source machine's absolute home and "
            "repository paths with anonymous or relative paths. Metrics and labels "
            "were not changed. Some archived JSON reports retain SHA-256 values "
            "of the private pre-redaction inputs; those values will not match the "
            "redacted copy. Use the generated SHA256SUMS.txt to verify this ZIP, "
            "and rerun the documented analyses to regenerate reports against the "
            "redacted files.\n",
            encoding="utf-8",
        )
        entries = sorted(path for path in stage.rglob("*") if path.is_file())
        checksums = []
        for path in entries:
            content = path.read_bytes()
            if b"/home/zyf" in content:
                raise ValueError(f"Unredacted user path remains: {path.relative_to(stage)}")
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {path.relative_to(stage)}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage))
    print(f"Wrote {output} with {count} source files")


if __name__ == "__main__":
    main()
