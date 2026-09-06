#!/usr/bin/env python3
"""Write one fixed-milestone paired report for the Qwen caption-dose chain."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from phase8_qwen_paired_report import parse_metrics


def validate_paired_bootstrap(
    paired: dict[str, Any],
    *,
    expected_n: int = 5521,
    expected_seed: int = 14159265,
    expected_replicates: int = 10_000,
    expected_tsv: str | None = None,
    expected_baseline_dir: str | None = None,
    expected_treatment_dir: str | None = None,
    expected_id_sha256: str | None = None,
) -> None:
    required = {
        "n", "mean_delta_treatment_minus_baseline", "ci95_low", "ci95_high",
        "baseline_mean", "treatment_mean", "paired_id_sha256", "bootstrap_seed",
        "bootstrap_replicates", "tsv", "baseline_dir", "treatment_dir",
    }
    missing = sorted(required - set(paired))
    numeric_keys = (
        "mean_delta_treatment_minus_baseline", "ci95_low", "ci95_high",
        "baseline_mean", "treatment_mean",
    )
    bad_numeric = [
        key for key in numeric_keys
        if isinstance(paired.get(key), bool)
        or not isinstance(paired.get(key), (int, float))
        or not math.isfinite(float(paired.get(key)))
    ]
    digest = paired.get("paired_id_sha256")
    path_drift = {
        key: (paired.get(key), expected)
        for key, expected in {
            "tsv": expected_tsv,
            "baseline_dir": expected_baseline_dir,
            "treatment_dir": expected_treatment_dir,
        }.items()
        if expected is not None and paired.get(key) != expected
    }
    invalid = (
        missing
        or type(paired.get("n")) is not int
        or paired.get("n") != expected_n
        or type(paired.get("bootstrap_seed")) is not int
        or paired.get("bootstrap_seed") != expected_seed
        or type(paired.get("bootstrap_replicates")) is not int
        or paired.get("bootstrap_replicates") != expected_replicates
        or bad_numeric
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or (expected_id_sha256 is not None and digest != expected_id_sha256)
        or path_drift
    )
    if not invalid:
        delta = float(paired["mean_delta_treatment_minus_baseline"])
        invalid = (
            float(paired["ci95_low"]) > float(paired["ci95_high"])
            or abs(
                float(paired["treatment_mean"])
                - float(paired["baseline_mean"])
                - delta
            ) > 1e-6
        )
    if invalid:
        raise RuntimeError(
            "invalid paired bootstrap: "
            f"missing={missing}, n={paired.get('n')}, bad_numeric={bad_numeric}, "
            f"path_drift={path_drift}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--qwen-metrics", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--qwen-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--paired-bootstrap", type=Path, required=True)
    parser.add_argument("--iteration", type=int, choices=(650000, 700000), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    control = parse_metrics(args.control_metrics)
    qwen = parse_metrics(args.qwen_metrics)
    control_audit = json.loads(args.control_audit.read_text(encoding="utf-8"))
    qwen_audit = json.loads(args.qwen_audit.read_text(encoding="utf-8"))
    paired = json.loads(args.paired_bootstrap.read_text(encoding="utf-8"))
    if control_audit.get("status") != "passed" or qwen_audit.get("status") != "passed":
        raise RuntimeError("dose report is gated by both passed arm audits")
    evaluation = contract["evaluation"]
    control_audio = str(Path(control_audit["checkpoint"]).parent / "musiccaps_eval/audio")
    qwen_audio = str(Path(qwen_audit["checkpoint"]).parent / "musiccaps_eval/audio")
    validate_paired_bootstrap(
        paired,
        expected_n=evaluation["num_samples"],
        expected_seed=evaluation["paired_bootstrap_seed"],
        expected_replicates=evaluation["paired_bootstrap_replicates"],
        expected_tsv=evaluation["tsv"],
        expected_baseline_dir=control_audio,
        expected_treatment_dir=qwen_audio,
        expected_id_sha256=evaluation["tsv_id_sha256"],
    )
    delta = {key: qwen[key] - control[key] for key in control}
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "iteration": args.iteration,
        "primary_metric": f"paired MusicCaps CLAP; fixed it{args.iteration}",
        "control": control,
        "qwen": qwen,
        "delta_qwen_minus_control": delta,
        "paired_per_prompt_clap": paired,
        "stretch_target": contract.get("metrics", {}).get("stretch_target_clap", 0.19),
        "historical_stretch_target": contract.get("metrics", {}).get("historical_stretch_target_clap", 0.1998),
        "selection_rule": f"predeclared it{args.iteration}; no best-checkpoint selection",
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
