#!/usr/bin/env python3
"""Write the predeclared final control-vs-Qwen report without cherry-picking."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise RuntimeError(f"metrics file missing: {path}")
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(raw.strip())
    if set(values) != {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
        raise RuntimeError(f"incomplete/nonstandard metrics: {path}: {values}")
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite metrics: {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--qwen-metrics", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--qwen-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--paired-bootstrap", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    control = parse_metrics(args.control_metrics)
    qwen = parse_metrics(args.qwen_metrics)
    control_audit = json.loads(args.control_audit.read_text(encoding="utf-8"))
    qwen_audit = json.loads(args.qwen_audit.read_text(encoding="utf-8"))
    paired = json.loads(args.paired_bootstrap.read_text(encoding="utf-8"))
    if control_audit.get("status") != "passed" or qwen_audit.get("status") != "passed":
        raise RuntimeError("final report is gated by both passed arm audits")
    delta = {key: qwen[key] - control[key] for key in control}
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "primary_metric": "paired MusicCaps CLAP; final it620000 only",
        "control": control,
        "qwen": qwen,
        "delta_qwen_minus_control": delta,
        "paired_per_prompt_clap": paired,
        "stretch_target": contract.get("metrics", {}).get("stretch_target_clap", 0.190),
        "historical_stretch_target": contract.get("metrics", {}).get("historical_stretch_target_clap", 0.1998),
        "selection_rule": "predeclared final checkpoint at it620000; no best-checkpoint selection",
        "thresholds_are_not_retrain_gates": True,
        "audit_paths": {"control": str(args.control_audit), "qwen": str(args.qwen_audit)},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.json_out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
