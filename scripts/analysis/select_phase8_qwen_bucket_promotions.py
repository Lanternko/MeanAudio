#!/usr/bin/env python3
"""Validate pilot reports and select noninferior arms for quarter promotion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


KS = (2, 3, 5, 10)
STRATEGIES = ("balanced", "fixed")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--musiccaps", type=Path, required=True)
    parser.add_argument("--pilot-prompts", type=Path, required=True)
    parser.add_argument("--grid-manifest", type=Path, required=True)
    parser.add_argument("--holdout-prompts", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--cap", type=int, default=4)
    args = parser.parse_args()
    grid = json.loads(args.grid_manifest.read_text())
    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    full_fields, full_rows = read_tsv(args.musiccaps)
    pilot_fields, pilot_rows = read_tsv(args.pilot_prompts)
    if full_fields != ["id", "caption"] or pilot_fields != full_fields:
        raise SystemExit("[FAIL] MusicCaps schemas must be exactly id/caption")
    full_by_id = {row["id"]: row for row in full_rows}
    pilot_ids = {row["id"] for row in pilot_rows}
    if (
        len(full_by_id) != 5521
        or len(pilot_ids) != 512
        or len(pilot_ids) != len(pilot_rows)
        or not pilot_ids <= set(full_by_id)
    ):
        raise SystemExit("[FAIL] MusicCaps full/pilot set invariant failed")
    holdout_rows = [row for row in full_rows if row["id"] not in pilot_ids]
    if len(holdout_rows) != 5009:
        raise SystemExit("[FAIL] holdout must be exactly 5009 rows")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=full_fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(holdout_rows)
    text = buffer.getvalue()
    if args.holdout_prompts.exists():
        if args.holdout_prompts.read_text(encoding="utf-8") != text:
            raise SystemExit("[FAIL] existing holdout TSV drift")
    else:
        args.holdout_prompts.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.holdout_prompts.with_suffix(".tsv.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, args.holdout_prompts)

    arms = []
    for k in KS:
        for strategy in STRATEGIES:
            path = (
                args.reports_dir
                / f"phase8_qwen_bucket_pilot_k{k}_{strategy}_FINAL_METRICS.json"
            )
            if not path.is_file():
                raise SystemExit(f"[FAIL] missing pilot report: {path}")
            report = json.loads(path.read_text())
            audit_path = Path(report.get("training_audit", ""))
            audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
            contract_path = Path(audit.get("contract", ""))
            contract = (
                json.loads(contract_path.read_text())
                if contract_path.is_file()
                else {}
            )
            try:
                s1 = float(report["stage1"]["high_q9"]["clap_score"])
                global_score = float(report["global"]["high_q9"]["clap_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"[FAIL] malformed pilot report: {path}") from exc
            if (
                report.get("status") != "passed"
                or report.get("experiment")
                != f"phase8_qwen_bucket_pilot_k{k}_{strategy}"
                or report.get("scale") != "pilot"
                or report.get("k") != k
                or report.get("strategy") != strategy
                or report.get("prompts", {}).get("rows") != 512
                or report.get("prompts", {}).get("path")
                != str(args.pilot_prompts)
                or report.get("prompts", {}).get("sha256")
                != sha(args.pilot_prompts)
                or report.get("stage1", {}).get("protocol")
                != "MusicCaps 512; FluxAudio FM25 CFG4.5"
                or report.get("global", {}).get("protocol")
                != "MusicCaps 512; MeanFlow1 CFG0.5"
                or audit.get("status") != "passed"
                or audit.get("scale") != "pilot"
                or audit.get("k") != k
                or audit.get("strategy") != strategy
                or audit.get("q_conditioning") is not True
                or audit.get("stage1_iteration") != 25000
                or audit.get("stage2_iteration") != 37500
                or not contract_path.is_file()
                or contract.get("train_tsv_sha256")
                != grid["outputs"][f"k{k}_{strategy}"]["sha256"]
                or set(report.get("models", {})) != {"stage1", "global"}
                or any(
                    not Path(model.get("path", "")).is_file()
                    or model.get("sha256") != sha(Path(model["path"]))
                    for model in report.get("models", {}).values()
                )
                or report.get("training_contract", {}).get("path")
                != str(contract_path)
                or report.get("training_contract", {}).get("sha256")
                != sha(contract_path)
                or not math.isfinite(s1)
                or not math.isfinite(global_score)
            ):
                raise SystemExit(f"[FAIL] ineligible/invalid pilot arm: {path}")
            arms.append(
                {
                    "k": k,
                    "strategy": strategy,
                    "stage1_clap": s1,
                    "global_clap": global_score,
                    "report": str(path),
                }
            )

    best_s1 = max(arm["stage1_clap"] for arm in arms)
    best_global = max(arm["global_clap"] for arm in arms)
    for arm in arms:
        arm["within_stage1_margin"] = arm["stage1_clap"] >= best_s1 - args.margin
        arm["within_global_margin"] = arm["global_clap"] >= best_global - args.margin
        arm["eligible"] = arm["within_stage1_margin"] and arm["within_global_margin"]
        arm["selection_loss"] = (
            best_s1 - arm["stage1_clap"] + best_global - arm["global_clap"]
        )
    eligible = [arm for arm in arms if arm["eligible"]]
    for arm in eligible:
        arm["pareto_noninferior"] = not any(
            other is not arm
            and other["stage1_clap"] >= arm["stage1_clap"]
            and other["global_clap"] >= arm["global_clap"]
            and (
                other["stage1_clap"] > arm["stage1_clap"]
                or other["global_clap"] > arm["global_clap"]
            )
            for other in eligible
        )
    selected = sorted(
        (arm for arm in eligible if arm["pareto_noninferior"]),
        key=lambda arm: (arm["selection_loss"], arm["k"], arm["strategy"]),
    )[: args.cap]
    # Preserve an eligible representative of each distribution strategy.
    for strategy in STRATEGIES:
        if any(arm["strategy"] == strategy for arm in selected):
            continue
        candidates = sorted(
            (arm for arm in eligible if arm["strategy"] == strategy),
            key=lambda arm: (arm["selection_loss"], arm["k"]),
        )
        if not candidates:
            continue
        candidate = candidates[0]
        if len(selected) < args.cap:
            selected.append(candidate)
        else:
            replaceable = [
                arm
                for arm in selected
                if sum(x["strategy"] == arm["strategy"] for x in selected) > 1
            ]
            if replaceable:
                worst = max(replaceable, key=lambda arm: arm["selection_loss"])
                selected[selected.index(worst)] = candidate
    selected = sorted(
        selected, key=lambda arm: (arm["selection_loss"], arm["k"], arm["strategy"])
    )
    if not selected:
        raise SystemExit("[FAIL] no arm passed both stage-specific margins")

    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "experiment": "phase8_qwen_bucket_pilot_gate",
        "gate": {
            "stage1": "FM25 CFG4.5 high-q9 CLAP within 0.005 of stage1 best",
            "global": "MF1 CFG0.5 high-q9 CLAP within 0.005 of global best",
            "both_required": True,
            "q9_minus_supported_low_is_diagnostic_only": True,
            "selection": "Pareto/noninferior eligible set; cap 4; eligible strategy coverage",
            "margin": args.margin,
            "cap": args.cap,
            "aggregate_fallback": (
                "aggregate CLAP used because evaluator metrics.txt does not retain "
                "per-prompt scores; paired bootstrap deferred"
            ),
        },
        "best": {"stage1_clap": best_s1, "global_clap": best_global},
        "arms": arms,
        "selected": [
            {"k": arm["k"], "strategy": arm["strategy"]} for arm in selected
        ],
        "holdout": {
            "path": str(args.holdout_prompts),
            "rows": len(holdout_rows),
            "disjoint_from_pilot": True,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
