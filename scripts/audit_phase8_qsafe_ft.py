#!/usr/bin/env python3
"""Fail-closed audit for one Phase-8 Q-safe fine-tuning arm."""

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
SOURCE_ID = "phase8_catalog_matched_noq_stage2_200000"
SOURCE = ROOT / "exps" / SOURCE_ID / f"{SOURCE_ID}_ckpt_last.pth"
MUSICCAPS = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                out[key.strip()] = float(value.strip())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--q-mode", choices=["real", "shuffled"], required=True)
    p.add_argument("--phase", choices=["train", "final"], default="train")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    issues: list[str] = []
    checks: dict[str, Any] = {}
    contract_path = LOG_ROOT / f"{args.prefix}_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    if not contract:
        issues.append(f"missing contract: {contract_path}")
    expected = {
        "prefix": args.prefix, "q_mode": args.q_mode,
        "source_checkpoint": str(SOURCE), "source_iteration": 600000,
        "initialization": "copy_q10_exactly_to_q0_through_q9",
        "use_q_conditioning": True, "use_text_attention_mask": False,
        "multi_cap": False, "fine_tune_iterations": 100000,
        "final_iteration": 700000, "learning_rate": 3e-5,
        "batch_size": 8, "accumulation_steps": 1, "seed": 14159265,
        "eval_tsv": str(MUSICCAPS), "eval_primary_q": 9, "eval_secondary_q": 6,
    }
    drift = {k: {"actual": contract.get(k), "expected": v} for k, v in expected.items() if contract.get(k) != v}
    checks["contract_drift"] = drift
    if drift:
        issues.append(f"contract drift: {drift}")

    if not SOURCE.is_file():
        issues.append(f"missing source: {SOURCE}")
    elif contract.get("source_checkpoint_sha256") != sha256(SOURCE):
        issues.append("source checkpoint hash drift")

    init_path = Path(contract.get("init_manifest", "/missing"))
    init = json.loads(init_path.read_text()) if init_path.is_file() else {}
    checks["init_manifest"] = str(init_path)
    if init.get("initialization") != "copy_q10_exactly_to_q0_through_q9":
        issues.append("wrong/missing Q-safe initialization manifest")
    audited = init.get("audited_tensors") or {}
    if len(audited) != 3 or not all(v.get("exactly_equal_after") is True for v in audited.values()):
        issues.append("Q-safe init did not prove exact q0..9=q10 for weights and two EMA tracks")

    train_tsv = Path(contract.get("train_tsv", "/missing"))
    if not train_tsv.is_file() or contract.get("train_tsv_sha256") != sha256(train_tsv):
        issues.append("training TSV missing/hash drift")
    changed = []
    for rel, expected_hash in (contract.get("critical_file_sha256") or {}).items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected_hash:
            changed.append(rel)
    checks["critical_sources_unchanged"] = not changed
    if changed:
        issues.append(f"critical source drift: {changed}")

    exp_id = f"{args.prefix}_stage2_ft100000"
    exp_dir = ROOT / "exps" / exp_id
    configs = sorted(exp_dir.glob("train-*-hydra/config.yaml"))
    checks["hydra_configs"] = [str(x) for x in configs]
    if not configs:
        issues.append("missing Hydra config")
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        expected_cfg = {
            "model": "meanaudio_s", "num_iterations": 700000,
            "use_q_conditioning": True, "use_text_attention_mask": False,
            "multi_cap": False, "batch_size": 8, "accumulation_steps": 1,
            "learning_rate": 3e-5, "seed": 14159265,
        }
        for key, value in expected_cfg.items():
            if cfg.get(key) != value:
                issues.append(f"Hydra drift {key}={cfg.get(key)!r}, expected={value!r}")
        for split in ("AudioCaps_npz", "AudioCaps_val_npz"):
            if ((cfg.get("data") or {}).get(split) or {}).get("tsv") != str(train_tsv):
                issues.append(f"Hydra TSV drift: {split}")

    ckpt = exp_dir / f"{exp_id}_ckpt_last.pth"
    ema = exp_dir / f"{exp_id}_ema_final.pth"
    if args.phase == "final":
        if not ckpt.is_file():
            issues.append("missing final checkpoint")
        else:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            checks["checkpoint_iteration"] = state.get("it")
            if state.get("it") != 700000:
                issues.append(f"checkpoint it={state.get('it')}, expected=700000")
        if not ema.is_file():
            issues.append("missing final EMA")

    result: dict[str, Any] = {}
    for q in (9, 6):
        label = f"q{q}"
        output = ROOT / "eval_output" / f"{exp_id}_musiccaps_{label}"
        metric_path = ROOT / "eval_output/metrics" / f"{exp_id}_musiccaps_{label}" / "metrics.txt"
        values = metrics(metric_path)
        result[label] = values
        if args.phase == "final":
            if set(values) != {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"} or not all(math.isfinite(v) for v in values.values()):
                issues.append(f"invalid {label} metrics: {values}")
            count = len(list((output / "audio").glob("*.flac")))
            checks[f"{label}_audio_count"] = count
            if count != 5521:
                issues.append(f"{label} audio count={count}")
        log_path = LOG_ROOT / f"{exp_id}_musiccaps_{label}_eval.log"
        if log_path.is_file():
            clean = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace"))
            line = next((x for x in clean.splitlines() if "Eval args:" in x), "")
            if not all(token in line for token in (f"'quality_level': {q}", "'no_q': False", "'no_text_attention_mask': True", "'cfg_strength': 0.5", "'num_steps': 1")):
                issues.append(f"{label} eval args drift: {line[:800]}")

    payload = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "prefix": args.prefix, "q_mode": args.q_mode, "phase": args.phase,
        "status": "passed" if not issues else "failed",
        "issues": issues, "checks": checks, "metrics": result,
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
