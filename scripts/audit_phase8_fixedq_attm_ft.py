#!/usr/bin/env python3
"""Fail-closed audit for one Phase-8 Fixed-Q / matched-NoQ fine-tuning arm."""

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
EXPECTED_AUDIO = 5521


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--arm", choices=["noq", "fixedq9"], required=True)
    parser.add_argument("--phase", choices=["train", "final"], default="train")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    issues: list[str] = []
    checks: dict[str, Any] = {}
    contract_path = LOG_ROOT / f"{args.prefix}_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    if not contract:
        issues.append(f"missing contract: {contract_path}")

    use_q = args.arm == "fixedq9"
    init_label = (
        "copy_q10_exactly_to_q0_through_q9"
        if args.arm == "fixedq9"
        else "preserve_q_embed_matched_optimizer_reset"
    )
    eval_mode = "quality_level_9" if args.arm == "fixedq9" else "no_q"
    expected = {
        "prefix": args.prefix,
        "arm": args.arm,
        "source_checkpoint": str(SOURCE),
        "source_iteration": 600000,
        "initialization": init_label,
        "use_q_conditioning": use_q,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "fine_tune_iterations": 100000,
        "final_iteration": 700000,
        "learning_rate": 3e-5,
        "batch_size": 8,
        "accumulation_steps": 1,
        "seed": 14159265,
        "eval_tsv": str(MUSICCAPS),
        "eval_mode": eval_mode,
    }
    drift = {
        k: {"actual": contract.get(k), "expected": v}
        for k, v in expected.items()
        if contract.get(k) != v
    }
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
    if init.get("initialization") != init_label or init.get("mode") != args.arm:
        issues.append("wrong/missing init manifest for arm")
    if not init.get("optimizer_reset") or not init.get("scheduler_reset"):
        issues.append("init did not reset optimizer/scheduler")
    audited = init.get("audited_tensors") or {}
    if len(audited) != 3:
        issues.append(
            f"init audited tensor count={len(audited)}, expected 3 "
            "(weights + 2 EMA tracks)"
        )
    if args.arm == "fixedq9":
        if not all(v.get("exactly_equal_after") is True for v in audited.values()):
            issues.append("fixedq9 init did not prove exact q0..9=q10")
    else:
        if any(v.get("mutated") is True for v in audited.values()):
            issues.append("noq init mutated q_embed")

    train_tsv = Path(contract.get("train_tsv", "/missing"))
    if not train_tsv.is_file() or contract.get("train_tsv_sha256") != sha256(train_tsv):
        issues.append("training TSV missing/hash drift")

    changed: list[str] = []
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
            "model": "meanaudio_s",
            "num_iterations": 700000,
            "use_q_conditioning": use_q,
            "use_text_attention_mask": False,
            "multi_cap": False,
            "batch_size": 8,
            "accumulation_steps": 1,
            "learning_rate": 3e-5,
            "seed": 14159265,
        }
        for key, value in expected_cfg.items():
            if cfg.get(key) != value:
                issues.append(
                    f"Hydra drift {key}={cfg.get(key)!r}, expected={value!r}"
                )
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

    eval_label = "q9" if args.arm == "fixedq9" else "noq"
    output = ROOT / "eval_output" / f"{exp_id}_musiccaps_{eval_label}"
    metric_path = (
        ROOT / "eval_output/metrics" / f"{exp_id}_musiccaps_{eval_label}" / "metrics.txt"
    )
    values = metrics(metric_path)
    result = {eval_label: values}
    if args.phase == "final":
        required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
        if set(values) != required or not all(math.isfinite(v) for v in values.values()):
            issues.append(f"invalid {eval_label} metrics: {values}")
        count = len(list((output / "audio").glob("*.flac")))
        checks[f"{eval_label}_audio_count"] = count
        if count != EXPECTED_AUDIO:
            issues.append(f"{eval_label} audio count={count}, expected={EXPECTED_AUDIO}")

    log_path = LOG_ROOT / f"{exp_id}_musiccaps_{eval_label}_eval.log"
    if args.phase == "final" and not log_path.is_file():
        issues.append(f"missing eval log: {log_path}")
    if log_path.is_file():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace"))
        line = next((x for x in clean.splitlines() if "Eval args:" in x), "")
        common = (
            "'no_text_attention_mask': True",
            "'cfg_strength': 0.5",
            "'num_steps': 1",
        )
        if args.arm == "fixedq9":
            need = common + ("'quality_level': 9", "'no_q': False")
        else:
            need = common + ("'no_q': True",)
        if not all(token in line for token in need):
            issues.append(f"{eval_label} eval args drift: {line[:800]}")
        # Guard against wrong Q flag on either arm.
        if args.arm == "fixedq9" and "'no_q': True" in line:
            issues.append("fixedq9 eval used --no_q (forbidden)")
        if args.arm == "noq" and "'no_q': False" in line and "quality_level" in line:
            # no_q must be True; quality_level default is ignored only when no_q.
            if "'no_q': True" not in line:
                issues.append("noq eval did not set --no_q")

    payload = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "prefix": args.prefix,
        "arm": args.arm,
        "phase": args.phase,
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "checks": checks,
        "metrics": result,
        "interpretation_preregistered": {
            "primary_checkpoint_iteration": 700000,
            "restoration_target_clap": 0.1900,
            "fixedq_benefit_requires_ci95_lb_gt_0": True,
            "do_not_stop_on_loss_plateau_near_0p98": True,
        },
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
