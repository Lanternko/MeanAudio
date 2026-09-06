#!/usr/bin/env python3
"""Build a read-only retrospective evidence bundle for the nine CFG0 evals.

This does not upgrade historical runs to HARN-canonical status. It records the
surviving evidence needed to compare their metrics without rerunning the GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
EXECUTED_WRAPPER = Path(
    "/home/kojiek/research/meanaudio_training/caption10s_pipeline/eval_musiccaps_mf25.sh"
)
CANONICAL_WRAPPER = ROOT / "scripts/caption10s_pipeline/eval_musiccaps_mf25.sh"
CONTRACT = ROOT / "docs/experiments/caption2p0_quarter_cfg0_rerun_contract.json"
DEFAULT_JSON = (
    ROOT / "docs/experiments/results/phase8/cfg0_retrospective_evidence_2026_08_22.json"
)
DEFAULT_MD = (
    ROOT / "docs/experiments/results/phase8/cfg0_retrospective_evidence_2026_08_22.md"
)
PROTOCOL = "MusicCaps 5521; MeanFlow 25; CFG 0; seed 42; NoMask; full precision"
METRIC_KEYS = ("clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ")
EXPECTED_LABELS = {
    "phase8_qwen_caption10s_multisent_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_bestof3_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_fair013_k3_quarter_musiccaps_mf25_cfg0_q9",
    "phase8_qwen_caption2p0_fair013_worstof3_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_qwen3cap_k3_quarter_musiccaps_mf25_cfg0_q9",
    "phase8_qwen_caption2p0_slot1_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_slot2_noq_quarter_musiccaps_mf25_cfg0_noq",
    "phase8_qwen_caption2p0_worstof3_noq_quarter_musiccaps_mf25_cfg0_noq",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    info = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": info.st_size,
        "mtime": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def read_eval_args(log_text: str) -> str:
    for line in log_text.splitlines():
        if "Eval args:" in line:
            return re.sub(r"\x1b\[[0-9;]*m", "", line).split("Eval args:", 1)[1].strip()
    raise ValueError("missing Eval args line")


def audio_ids_from_log(log_text: str) -> list[str]:
    ids = []
    for raw_line in log_text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)
        if "Audio saved to " not in line:
            continue
        path = line.split("Audio saved to ", 1)[1].strip()
        ids.append(Path(path).stem)
    return ids


def musiccaps_ids() -> list[str]:
    with TSV.open(encoding="utf-8", newline="") as handle:
        return [row["id"] for row in csv.DictReader(handle, delimiter="\t")]


def build_cell(report_path: Path, preregistered: dict[str, dict[str, object]]) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    label = report["label"]
    if label not in EXPECTED_LABELS:
        raise ValueError(f"unexpected CFG0 report: {label}")
    metrics_path = Path(report["metrics_path"])
    checkpoint_path = Path(report["checkpoint"])
    log_path = LOG_ROOT / f"{label}_eval.log"
    for path in (metrics_path, checkpoint_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = report.get("metrics")
    if report.get("status") != "passed" or report.get("protocol") != PROTOCOL:
        raise ValueError(f"report status/protocol mismatch: {label}")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_KEYS):
        raise ValueError(f"metric mismatch: {label}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    saved_ids = audio_ids_from_log(log_text)
    saved = len(saved_ids)
    loaded = log_text.count("載入 5521 筆 test records")
    eval_args = read_eval_args(log_text)
    expected_ids = musiccaps_ids()
    args_match = all(fragment in eval_args for fragment in (
        "'cfg_strength': 0.0", "'num_steps': 25", "'seed': 42",
        "'no_text_attention_mask': True", "'full_precision': True",
        "'use_meanflow': True", f"'model_path': '{checkpoint_path}'",
        f"'tsv': '{TSV}'",
    ))
    conditioning_match = (
        "'no_q': True" in eval_args
        if label.endswith("_noq")
        else "'no_q': False" in eval_args and "'quality_level': 9" in eval_args
    )
    checkpoint_hash = sha256(checkpoint_path)
    prior = preregistered.get(label)
    preregistration = {
        "available": prior is not None,
        "checkpoint_sha256_matches": (
            checkpoint_hash == prior["checkpoint_sha256"] if prior else None
        ),
    }
    checks = {
        "report_passed": True,
        "reported_protocol_exact": True,
        "five_finite_metrics_present": all(
            isinstance(metrics[key], (int, float)) and math.isfinite(metrics[key])
            for key in METRIC_KEYS
        ),
        "eval_args_match_reported_protocol": args_match,
        "eval_args_match_conditioning": conditioning_match,
        "audio_saved_log_count": saved,
        "audio_saved_log_count_is_5521": saved == 5521,
        "audio_saved_log_unique_id_count": len(set(saved_ids)),
        "audio_saved_log_unique_ids_are_5521": len(set(saved_ids)) == 5521,
        "audio_saved_log_ids_match_musiccaps_tsv": set(saved_ids) == set(expected_ids),
        "metric_evaluator_loaded_5521_count": loaded,
        "metric_evaluator_loaded_5521": loaded >= 1,
        "checkpoint_exists": True,
        "musiccaps_tsv_current_sha256_matches_preregistered": (
            sha256(TSV) == "de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f"
        ),
    }
    comparable = all(
        value is True
        for key, value in checks.items()
        if key not in {"audio_saved_log_count", "metric_evaluator_loaded_5521_count"}
        and key != "audio_saved_log_unique_id_count"
    )
    return {
        "label": label,
        "completed_at": report["completed_at"],
        "classification": (
            "retrospective_operationally_complete_comparable"
            if comparable
            else "retrospective_evidence_incomplete"
        ),
        "conditioning": "no_q" if label.endswith("_noq") else "q9",
        "metrics": {key: metrics[key] for key in METRIC_KEYS},
        "reported_protocol": report["protocol"],
        "eval_args_log_line": eval_args,
        "checks": checks,
        "preregistration": preregistration,
        "artifacts": {
            "report": artifact(report_path),
            "metrics": artifact(metrics_path),
            "eval_log": artifact(log_path),
            "checkpoint": artifact(checkpoint_path),
        },
        "known_limitations": [
            "generated FLAC files were deleted by the executed wrapper after metric evaluation",
            "no launch-time executable hash was persisted",
            "the surviving report was not emitted by the strict contract-bound HARN",
        ],
    }


def render_markdown(bundle: dict[str, object], json_path: Path) -> str:
    cells = bundle["cells"]
    lines = [
        "# CFG0 retrospective evidence and comparison",
        "",
        f"Generated: `{bundle['generated_at']}`",
        "",
        "## Disposition",
        "",
        "All nine evaluations are classified as `retrospective_operationally_complete_comparable`.",
        "No GPU rerun is required for the comparison table below. This bundle preserves the",
        "surviving report, metrics, log, checkpoint, and current input hashes. It does not",
        "claim that the historical runs passed the later strict HARN acceptance flow.",
        "",
        "## Metrics",
        "",
        "| Rank | Label | CLAP | AES-CE | AES-CU | AES-PC | AES-PQ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, cell in enumerate(sorted(cells, key=lambda item: item["metrics"]["clap_score"], reverse=True), 1):
        metrics = cell["metrics"]
        lines.append(
            f"| {rank} | `{cell['label']}` | {metrics['clap_score']:.4f} | "
            f"{metrics['aes_CE']:.4f} | {metrics['aes_CU']:.4f} | "
            f"{metrics['aes_PC']:.4f} | {metrics['aes_PQ']:.4f} |"
        )
    lines.extend([
        "",
        "## Evidence boundary",
        "",
        "For every cell, its log contains exactly 5,521 unique `Audio saved` IDs; that ID set",
        "exactly matches the MusicCaps TSV, and the metric evaluator loaded 5,521 records.",
        "The logged CFG, steps, seed, conditioning, NoMask, and precision also match across all",
        "nine cells. The five finite metrics,",
        "checkpoint, metrics file, report, and complete eval log survive and are SHA-256 bound",
        "in the JSON bundle.",
        "",
        "The generated FLAC files do not survive, and the executed wrapper was not hash-bound",
        "at launch. These are provenance limitations, not evidence that the computation failed.",
        "This record therefore supports within-table comparison while retaining the historical",
        "execution caveat.",
        "",
        f"Machine-readable evidence: `{json_path}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    preregistered = {cell["label"]: cell for cell in contract["cells"]}
    reports = sorted(LOG_ROOT.glob("*cfg0*_REPORT.json"))
    labels = {json.loads(path.read_text(encoding="utf-8"))["label"] for path in reports}
    if labels != EXPECTED_LABELS or len(reports) != 9:
        raise SystemExit(f"expected exact nine CFG0 reports; found {len(reports)} labels={sorted(labels)}")

    cells = [build_cell(path, preregistered) for path in reports]
    if not all(cell["classification"] == "retrospective_operationally_complete_comparable" for cell in cells):
        raise SystemExit("one or more CFG0 cells lack comparison evidence")
    bundle = {
        "schema_version": 1,
        "document_kind": "cfg0_retrospective_evidence_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disposition": "operationally_complete_comparable_no_gpu_rerun_requested",
        "scope": {
            "cell_count": 9,
            "protocol": PROTOCOL,
            "comparison_allowed": True,
            "canonical_harn_completion_claimed": False,
        },
        "shared_inputs": {
            "musiccaps_tsv": artifact(TSV),
            "executed_wrapper_observed_after_run": artifact(EXECUTED_WRAPPER),
            "strict_wrapper_not_executed_by_these_runs": artifact(CANONICAL_WRAPPER),
            "four_cell_preregistration_contract": artifact(CONTRACT),
        },
        "cells": sorted(cells, key=lambda item: item["label"]),
    }
    encoded = json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(bundle, args.json_out)
    if args.check:
        existing = json.loads(args.json_out.read_text(encoding="utf-8"))
        for payload in (bundle, existing):
            payload.pop("generated_at", None)
        if bundle != existing:
            raise SystemExit("retrospective evidence bundle drift")
        print(f"CFG0_RETROSPECTIVE_EVIDENCE_OK cells={len(cells)}")
        return
    atomic_write(args.json_out, encoded)
    atomic_write(args.md_out, markdown)
    print(f"CFG0_RETROSPECTIVE_EVIDENCE_WRITTEN cells={len(cells)} json={args.json_out} md={args.md_out}")


if __name__ == "__main__":
    main()
