#!/usr/bin/env python3
"""Fail-closed contract and completion audit for Phase-8 S2-only Q arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
SOURCE_S1 = (
    ROOT
    / "exps/phase8_catalog_matched_noq_stage1_400000"
    / "phase8_catalog_matched_noq_stage1_400000_ckpt_last.pth"
)
MUSICCAPS = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        if key.strip() in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key.strip()] = float(raw.strip())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--q-mode", choices=["real", "shuffled"], required=True)
    parser.add_argument("--phase", choices=["auto", "train", "final"], default="auto")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    issues: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    contract_path = LOG_ROOT / f"{args.prefix}_contract.json"
    if not contract_path.is_file():
        issues.append(f"missing launch contract: {contract_path}")
        contract: dict[str, Any] = {}
    else:
        contract = json.loads(contract_path.read_text())

    expected_contract = {
        "prefix": args.prefix,
        "q_mode": args.q_mode,
        "source_s1_checkpoint": str(SOURCE_S1),
        "source_s1_iteration": 400000,
        "stage1_use_q_conditioning": False,
        "stage2_use_q_conditioning": True,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "stage2_final_iteration": 600000,
        "eval_primary_q": 9,
        "eval_secondary_q": 6,
        "eval_tsv": str(MUSICCAPS),
    }
    drift = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    checks["contract_drift"] = drift
    if drift:
        issues.append(f"launch contract drift: {drift}")

    if not SOURCE_S1.is_file():
        issues.append(f"source S1 checkpoint missing: {SOURCE_S1}")
    elif contract.get("source_s1_sha256"):
        current = sha256(SOURCE_S1)
        checks["source_s1_sha256"] = current
        if current != contract["source_s1_sha256"]:
            issues.append("source S1 checkpoint hash changed after launch")

    train_tsv = Path(contract.get("train_tsv", "/missing"))
    if not train_tsv.is_file():
        issues.append(f"training TSV missing: {train_tsv}")
    elif contract.get("train_tsv_sha256"):
        current = sha256(train_tsv)
        checks["train_tsv_sha256"] = current
        if current != contract["train_tsv_sha256"]:
            issues.append("training TSV hash changed after launch")

    critical_hashes = contract.get("critical_file_sha256") or {}
    changed_sources = []
    for rel, expected_hash in critical_hashes.items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected_hash:
            changed_sources.append(rel)
    checks["critical_sources_unchanged"] = not changed_sources
    if changed_sources:
        issues.append(f"critical source drift: {changed_sources}")

    exp_id = f"{args.prefix}_stage2_200000"
    exp_dir = ROOT / "exps" / exp_id
    configs = sorted(exp_dir.glob("train-*-hydra/config.yaml"))
    checks["hydra_configs"] = [str(path) for path in configs]
    require_training = args.phase in {"train", "final"}
    if not configs and require_training:
        issues.append(f"S2 Hydra config missing: {exp_dir}")
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        expected_cfg = {
            "model": "meanaudio_s",
            "num_iterations": 600000,
            "use_q_conditioning": True,
            "use_text_attention_mask": False,
            "multi_cap": False,
            "batch_size": 8,
            "accumulation_steps": 1,
            "learning_rate": 1e-4,
        }
        for key, value in expected_cfg.items():
            if cfg.get(key) != value:
                issues.append(f"Hydra drift {path}: {key}={cfg.get(key)!r}, expected {value!r}")
        data_cfg = cfg.get("data", {})
        for split in ("AudioCaps_npz", "AudioCaps_val_npz"):
            observed = (data_cfg.get(split) or {}).get("tsv")
            if observed != str(train_tsv):
                issues.append(f"Hydra drift {path}: {split}.tsv={observed!r}")

    ckpt_path = exp_dir / f"{exp_id}_ckpt_last.pth"
    ema_path = exp_dir / f"{exp_id}_ema_final.pth"
    checks["checkpoint"] = str(ckpt_path)
    checks["ema"] = str(ema_path)
    if ckpt_path.is_file() and args.phase == "final":
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checks["checkpoint_iteration"] = state.get("it")
        if state.get("it") != 600000:
            issues.append(f"final checkpoint iteration={state.get('it')}, expected 600000")
    if args.phase == "final" and not ema_path.is_file():
        issues.append(f"final EMA missing: {ema_path}")

    metric_payload: dict[str, Any] = {}
    for q in (9, 6):
        label = f"q{q}"
        output = ROOT / "eval_output" / f"{exp_id}_musiccaps_{label}"
        metrics_path = ROOT / "eval_output/metrics" / f"{exp_id}_musiccaps_{label}" / "metrics.txt"
        eval_log = LOG_ROOT / f"{exp_id}_musiccaps_{label}_eval.log"
        checks[f"{label}_metrics"] = str(metrics_path)
        if metrics_path.is_file():
            values = read_metrics(metrics_path)
            metric_payload[label] = values
            required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
            if set(values) != required or not all(math.isfinite(v) for v in values.values()):
                issues.append(f"invalid/non-finite {label} metrics: {values}")
            count = len(list((output / "audio").glob("*.flac")))
            checks[f"{label}_audio_count"] = count
            if args.phase == "final" and count != 5521:
                issues.append(f"{label} audio count={count}, expected 5521")
        elif args.phase == "final":
            issues.append(f"missing {label} metrics: {metrics_path}")

        if eval_log.is_file():
            clean = re.sub(r"\x1b\[[0-9;]*m", "", eval_log.read_text(errors="replace"))
            args_line = next((line for line in clean.splitlines() if "Eval args:" in line), "")
            if args_line and not (
                "'no_q': False" in args_line
                and f"'quality_level': {q}" in args_line
                and "'no_text_attention_mask': True" in args_line
                and "'num_steps': 1" in args_line
                and "'cfg_strength': 0.5" in args_line
            ):
                issues.append(f"{label} eval argument drift: {args_line[:1000]}")

    payload = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "prefix": args.prefix,
        "q_mode": args.q_mode,
        "phase": args.phase,
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
        "metrics": metric_payload,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.json_out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
