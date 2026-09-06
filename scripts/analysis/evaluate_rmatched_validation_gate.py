#!/usr/bin/env python3
"""Apply preregistered paired-CI and Jamendo non-inferiority gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paired = json.loads(args.paired.read_text())
    dual = json.loads(args.dual.read_text())
    jamendo = dual.get("benchmarks", {}).get("jamendo_seed42_2048", {})
    metrics = jamendo.get("metrics", {})
    ci = paired.get("delta_ci95", [])
    checks = {
        "paired_n_5521": paired.get("n") == 5521,
        "paired_point_delta_ge_0.005": isinstance(paired.get("delta"), (int, float)) and paired["delta"] >= 0.005,
        "paired_ci95_lower_gt_zero": len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci) and ci[0] > 0,
        "jamendo_n_2048": jamendo.get("n") == 2048,
        "jamendo_clap_ge_0.1936": isinstance(metrics.get("clap_score"), (int, float)) and metrics["clap_score"] >= 0.1936,
        "jamendo_aes_ce_ge_6.064": isinstance(metrics.get("aes_CE"), (int, float)) and metrics["aes_CE"] >= 6.064,
        "dual_report_passed": dual.get("status") == "passed",
    }
    valid = all(isinstance(value, bool) for value in checks.values())
    verdict = "pass" if valid and all(checks.values()) else "fail" if valid else "invalid"
    payload = {
        "schema_version": 1, "completed_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "promote_seed27182818": verdict == "pass", "checks": checks,
        "thresholds": {"paired_delta": 0.005, "paired_ci95_lower": 0.0,
                       "jamendo_clap": 0.1936, "jamendo_aes_CE": 6.064},
        "descriptive_only": ["musiccaps_peav", "musiccaps_t2a_R@10", "jamendo_peav", "jamendo_t2a_R@10"],
        "paired": {"path": str(args.paired), "sha256": sha(args.paired)},
        "dual": {"path": str(args.dual), "sha256": sha(args.dual)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
