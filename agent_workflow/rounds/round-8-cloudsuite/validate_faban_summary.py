#!/usr/bin/env python3
"""Validate Faban structure; keep mix, failures, and latency SLO distinct."""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def number(parent: ET.Element, tag: str) -> float:
    value = parent.findtext(tag)
    if value is None:
        raise ValueError(f"missing {tag}")
    try:
        result = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"non-numeric {tag}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {tag}")
    return result


def count(parent: ET.Element, tag: str) -> int:
    value = number(parent, tag)
    if value < 0 or value != int(value):
        raise ValueError(f"invalid {tag} count")
    return int(value)


def boolean(parent: ET.Element, tag: str) -> bool:
    value = (parent.findtext(tag) or "").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid {tag} boolean")
    return value == "true"


def inspect(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    if root.tag != "benchResults":
        raise ValueError("unexpected Faban root")
    bench, driver = root.find("benchSummary"), root.find("driverSummary")
    if bench is None or driver is None:
        raise ValueError("missing benchSummary or driverSummary")
    bench_passed, driver_passed = boolean(bench, "passed"), boolean(driver, "passed")
    metric = driver.find("metric")
    if metric is None or metric.get("unit", "").lower() not in {"ops/sec", "operations/sec"}:
        raise ValueError("missing ops/sec metric")
    throughput, completed = number(driver, "metric"), count(driver, "totalOps")
    if throughput <= 0 or completed <= 0:
        raise ValueError("nonpositive throughput or totalOps")

    mix = driver.find("mix")
    if mix is None or mix.get("allowedDeviation") is None:
        raise ValueError("missing operation mix or tolerance")
    tolerance = float(mix.attrib["allowedDeviation"])
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("invalid operation-mix tolerance")
    operations = []
    names = set()
    for node in mix.findall("operation"):
        name = node.get("name", "")
        if not name or name in names:
            raise ValueError("missing or duplicate operation name")
        names.add(name)
        observed, required = number(node, "mix"), number(node, "requiredMix")
        if not 0 <= observed <= 1 or not 0 <= required <= 1:
            raise ValueError(f"invalid mix fraction for {name}")
        operations.append({
            "name": name,
            "successes": count(node, "successes"),
            "failures": count(node, "failures"),
            "observed_mix": observed,
            "required_mix": required,
            "mix_passed": boolean(node, "passed"),
        })
    if not operations:
        raise ValueError("missing operation rows")
    successes = sum(row["successes"] for row in operations)
    failures = sum(row["failures"] for row in operations)
    if successes != completed:
        raise ValueError(f"totalOps={completed} but successes={successes}")
    if abs(sum(row["observed_mix"] for row in operations) - 1) > 0.01:
        raise ValueError("operation mix does not sum to one")
    mix_valid = all(row["mix_passed"] for row in operations)

    response = driver.find("responseTimes")
    if response is None or response.get("unit") != "milliseconds":
        raise ValueError("missing responseTimes in milliseconds")
    response_rows = []
    samples = {row["name"]: row["successes"] for row in operations}
    for node in response.findall("operation"):
        name = node.get("name", "")
        if name not in names:
            raise ValueError(f"unknown response-time operation {name}")
        row = {"name": name, "samples": samples[name]}
        for percentile in node.findall("percentile"):
            nth = percentile.get("nth")
            if nth not in {"95", "99"}:
                continue
            if percentile.text and percentile.text.strip():
                raw = percentile.text.strip()
                censored = raw.startswith(">")
                value = float(raw[1:].strip() if censored else raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid p{nth} for {name}")
                row[f"p{nth}_{'lower_bound_ms' if censored else 'ms'}"] = value
                row[f"p{nth}_censored"] = censored
            if percentile.get("limit") is not None:
                limit = float(percentile.attrib["limit"])
                if not math.isfinite(limit) or limit < 0:
                    raise ValueError(f"invalid p{nth} limit for {name}")
                row[f"p{nth}_limit_ms"] = limit
        status = (node.findtext("passed") or "").strip().lower()
        if status not in {"", "true", "false"}:
            raise ValueError(f"invalid latency status for {name}")
        row["slo_passed"] = None if not status else status == "true"
        if samples[name] and any(
            f"p{nth}_ms" not in row and f"p{nth}_lower_bound_ms" not in row
            for nth in (95, 99)
        ):
            raise ValueError(f"missing latency percentile for sampled {name}")
        response_rows.append(row)
    if {row["name"] for row in response_rows} != names:
        raise ValueError("response-time and mix operation sets differ")
    violations = [row["name"] for row in response_rows if row["slo_passed"] is False]
    return {
        "valid": True,
        "summary_valid": True,
        "mix_valid": mix_valid,
        "throughput_usable": mix_valid and failures == 0,
        "bench_passed": bench_passed,
        "driver_passed": driver_passed,
        "latency_slo_passed": not violations,
        "latency_slo_violations": violations,
        "summary_xml": str(path),
        "throughput_ops_per_sec": throughput,
        "completed_operations": completed,
        "operation_successes": successes,
        "operation_failures": failures,
        "transaction_failure_rate": failures / (successes + failures),
        "operation_mix_tolerance": tolerance,
        "operations": operations,
        "response_times": response_rows,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} SUMMARY_XML OUTPUT_JSON", file=sys.stderr)
        return 2
    source, destination = map(Path, sys.argv[1:])
    try:
        result = inspect(source)
    except (ET.ParseError, OSError, ValueError) as exc:
        result = {"valid": False, "summary_valid": False, "summary_xml": str(source), "reason": str(exc)}
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["valid"]:
        print(result["reason"], file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "summary_valid", "mix_valid", "throughput_usable", "completed_operations",
        "operation_failures", "latency_slo_passed",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
