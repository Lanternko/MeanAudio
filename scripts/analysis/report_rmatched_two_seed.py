#!/usr/bin/env python3
"""Write an explicitly two-seed descriptive variability report."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-a", type=Path, required=True)
    parser.add_argument("--seed-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.seed_a, args.seed_b)]
    benchmarks = {}
    keys = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
    for benchmark in ("musiccaps", "jamendo_seed42_2048"):
        rows = [report["benchmarks"][benchmark]["metrics"] for report in reports]
        values = {}
        for key in keys:
            pair = [float(row[key]) for row in rows]
            if not all(math.isfinite(value) for value in pair):
                raise SystemExit(f"[FAIL] nonfinite {benchmark}/{key}")
            values[key] = {"seed14159265": pair[0], "seed27182818": pair[1],
                           "mean": sum(pair) / 2, "range": abs(pair[1] - pair[0])}
        benchmarks[benchmark] = values
    payload = {"schema_version": 1, "status": "passed",
               "completed_at": datetime.now(timezone.utc).isoformat(),
               "claim_scope": "descriptive two-seed variability only; not a population confidence claim",
               "seeds": [14159265, 27182818], "benchmarks": benchmarks,
               "source_reports": [str(args.seed_a), str(args.seed_b)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
