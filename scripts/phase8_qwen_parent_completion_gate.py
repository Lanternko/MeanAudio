#!/usr/bin/env python3
"""Fail closed until the parent Qwen queue has fully and successfully exited."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parent_completion_passed(report: Path, manifest: Path, *, queue_active: bool) -> bool:
    if queue_active or not report.is_file() or not manifest.is_file():
        return False
    try:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    step = manifest_payload.get("steps", {}).get("paired_final_report", {})
    return (
        report_payload.get("status") == "passed"
        and step.get("status") == "passed"
        and step.get("exit_code") == 0
    )


def queue_is_active() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "phase8_qwen_probe_queue.py --execute"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    return 0 if parent_completion_passed(
        args.report, args.manifest, queue_active=queue_is_active()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
