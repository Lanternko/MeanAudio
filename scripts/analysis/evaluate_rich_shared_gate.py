#!/usr/bin/env python3
"""Evaluate the preregistered R-Shared quarter -> R-Matched full promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in METRIC_KEYS:
            try:
                values[key] = float(raw)
            except ValueError:
                pass
    return values


def read_matched(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = report.get("global", {}).get("no_q", {})
    return {key: float(values[key]) for key in METRIC_KEYS if key in values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-report", type=Path, required=True)
    parser.add_argument("--shared-metrics", type=Path, required=True)
    parser.add_argument("--shared-model", type=Path, required=True)
    parser.add_argument("--shared-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    args = parser.parse_args()

    shared = read_metrics(args.shared_metrics)
    matched = read_matched(args.matched_report)
    audit = json.loads(args.shared_audit.read_text(encoding="utf-8"))
    complete_keys = set(shared) == set(METRIC_KEYS) and set(matched) == set(METRIC_KEYS)
    finite = complete_keys and all(math.isfinite(value) for value in (*shared.values(), *matched.values()))
    valid = (
        finite
        and audit.get("status") == "passed"
        and audit.get("completed_rows") == 251599
        and args.shared_model.is_file()
    )
    deltas = {
        key: matched[key] - shared[key]
        for key in METRIC_KEYS
        if key in matched and key in shared
    }
    checks = {
        "r_shared_valid_completion": valid,
        "matched_minus_shared_clap_at_least_0.005": valid and deltas["clap_score"] >= 0.005,
        "matched_minus_shared_ce_at_least_minus_0.06": valid and deltas["aes_CE"] >= -0.06,
    }
    promote = all(checks.values())
    gate = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if promote else "not_promoted",
        "promote_r_matched_full": promote,
        "contrast": "R-Matched quarter minus R-Shared quarter",
        "thresholds": {"clap_delta_min": 0.005, "ce_delta_min": -0.06},
        "checks": checks,
        "r_matched_quarter": matched,
        "r_shared_quarter": shared,
        "deltas": deltas,
        "note": "A low but finite R-Shared result is a scientific outcome, not a technical collapse veto.",
    }
    atomic_json(args.gate_report, gate)
    final = {
        "schema_version": 1,
        "completed_at": gate["completed_at"],
        "status": "quarter_complete_full_pending" if promote else "completed_not_promoted",
        "experiment": "rich_shared_quarter_then_conditional_rich_matched_full",
        "gate": gate,
        "artifacts": {
            "contract": {"path": str(args.contract), "sha256": sha256_file(args.contract)},
            "r_shared_model": {"path": str(args.shared_model), "sha256": sha256_file(args.shared_model)},
            "r_shared_npz_audit": str(args.shared_audit),
            "r_matched_quarter_report": str(args.matched_report),
        },
    }
    atomic_json(args.final_report, final)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
