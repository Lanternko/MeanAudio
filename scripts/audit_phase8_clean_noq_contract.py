#!/usr/bin/env python3
"""Fail-closed runtime audit for the Phase-8 catalog-matched clean-NoQ run.

The experiment contract is deliberately stage-specific:
  S1 use_q_conditioning=false -> q=None -> FluxAudio q=10
  S2 use_q_conditioning=false -> q=None -> MeanAudio q=10
  eval --no_q + --no_text_attention_mask

This script is safe to run repeatedly while training is active.  Missing future
stages are reported as pending in ``auto`` mode, while an explicitly required
phase fails if its artifacts are absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
DEFAULT_PREFIX = "phase8_catalog_matched_noq"
DEFAULT_TRAIN_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv"
)
DEFAULT_CACHE = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")
DEFAULT_NPZ = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
DEFAULT_EVAL_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def clean_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def add_issue(issues: list[str], condition: bool, detail: str) -> None:
    if not condition:
        issues.append(detail)


def check_static_q_routing(issues: list[str], checks: dict[str, Any]) -> None:
    runners = {
        "S1": ROOT / "meanaudio/runner_flowmatching.py",
        "S2": ROOT / "meanaudio/runner_meanflow.py",
    }
    token = (
        "q = data['q_level'].cuda(non_blocking=True) "
        "if ('q_level' in data and use_q) else None"
    )
    for stage, path in runners.items():
        text = path.read_text()
        ok = "use_q = self.cfg.get('use_q_conditioning', True)" in text and token in text
        checks[f"{stage.lower()}_runner_noq_routes_none"] = ok
        add_issue(issues, ok, f"{stage} runner no longer routes disabled Q to q=None: {path}")

    networks = (ROOT / "meanaudio/model/networks.py").read_text()
    flux_section = networks[networks.find("class FluxAudio") : networks.find("class MeanAudio")]
    mean_section = networks[networks.find("class MeanAudio") :]
    q10 = re.compile(r"if q is None:\s+q = torch\.full\([^\n]+, 10,", re.MULTILINE)
    flux_ok = bool(q10.search(flux_section))
    mean_ok = bool(q10.search(mean_section))
    checks["fluxaudio_none_maps_q10"] = flux_ok
    checks["meanaudio_none_maps_q10"] = mean_ok
    add_issue(issues, flux_ok, "FluxAudio q=None no longer maps to q=10")
    add_issue(issues, mean_ok, "MeanAudio q=None no longer maps to q=10")


def check_data_gate(
    issues: list[str], checks: dict[str, Any], expected_rows: int, train_tsv: Path,
    cache: Path, npz_dir: Path,
) -> None:
    validation_path = npz_dir / "FULL_VALIDATION.json"
    gate_path = npz_dir / "FULL_GATE_PASSED.json"
    required = [train_tsv, cache, npz_dir / "MANIFEST.tsv", validation_path, gate_path]
    missing = [str(path) for path in required if not path.is_file()]
    checks["data_inputs_present"] = not missing
    if missing:
        issues.append(f"missing data contract inputs: {missing}")
        return

    try:
        validation = json.loads(validation_path.read_text())
        gate = json.loads(gate_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"cannot parse cache gate sentinels: {exc}")
        return
    structural_ok = (
        validation.get("status") == "passed"
        and validation.get("expected_rows") == expected_rows
    )
    semantic_ok = (
        gate.get("status") == "passed"
        and gate.get("decoded_samples") == 512
        and float(gate.get("decoded_cache_clap", -math.inf))
        >= float(gate.get("minimum_clap", math.inf))
        and gate.get("validation_report_sha256") == sha256(validation_path)
    )
    checks["structural_cache_gate"] = structural_ok
    checks["semantic_cache_gate"] = semantic_ok
    add_issue(issues, structural_ok, "structural cache gate is not valid/passed")
    add_issue(issues, semantic_ok, "semantic cache gate is not valid/passed or hash drifted")


def check_launch_contract(
    issues: list[str], warnings: list[str], checks: dict[str, Any], prefix: str,
    args: argparse.Namespace,
) -> None:
    path = LOG_ROOT / f"{prefix}_contract.json"
    checks["launch_contract_path"] = str(path)
    if not path.is_file():
        message = f"launch contract missing (run may predate contract hardening): {path}"
        if args.require_launch_contract:
            issues.append(message)
        else:
            warnings.append(message)
        checks["launch_contract"] = "missing"
        return
    try:
        contract = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"invalid launch contract: {exc}")
        return
    expected = {
        "prefix": prefix,
        "stage1_use_q_conditioning": False,
        "stage2_use_q_conditioning": False,
        "use_text_attention_mask": False,
        "eval_q_mode": "no_q",
        "multi_cap": False,
        "train_tsv": str(args.train_tsv),
        "gt_cache": str(args.gt_cache),
        "npz_dir": str(args.npz_dir),
        "eval_tsv": str(args.eval_tsv),
        "expected_rows": args.expected_rows,
        "stage1_iterations": args.s1_iterations,
        "stage2_additional_iterations": args.s2_iterations,
        "stage2_final_iteration": args.s1_iterations + args.s2_iterations,
    }
    drift = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    checks["launch_contract"] = "passed" if not drift else "drifted"
    if drift:
        issues.append(f"launch contract drift: {drift}")

    regime = contract.get("regime")
    if regime != "clean_noq":
        message = (
            f"launch contract regime={regime!r}, expected 'clean_noq'; "
            "resolved S1/S2/eval fields are audited separately"
        )
        if args.require_launch_contract:
            issues.append(message)
        else:
            warnings.append(message)

    hashes = contract.get("critical_file_sha256") or {}
    stable_critical = {
        "scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh",
        "meanaudio/runner_flowmatching.py",
        "meanaudio/runner_meanflow.py",
        "meanaudio/model/networks.py",
    }
    changed = []
    for rel, expected_hash in hashes.items():
        if rel not in stable_critical:
            continue
        if regime == "custom" and rel == (
            "scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
        ):
            continue
        path_obj = ROOT / rel
        if not path_obj.is_file() or sha256(path_obj) != expected_hash:
            changed.append(rel)
    checks["critical_source_hashes_unchanged"] = not changed
    if changed:
        issues.append(f"critical source changed since launch: {changed}")


def expected_stage_config(args: argparse.Namespace, stage: int) -> dict[str, Any]:
    return {
        "model": "fluxaudio_s" if stage == 1 else "meanaudio_s",
        "num_iterations": (
            args.s1_iterations if stage == 1 else args.s1_iterations + args.s2_iterations
        ),
        "use_q_conditioning": False,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "use_rope": False,
        "batch_size": 8,
        "accumulation_steps": 1,
        "learning_rate": 1e-4,
    }


def check_stage_configs(
    issues: list[str], checks: dict[str, Any], args: argparse.Namespace, stage: int,
    required: bool,
) -> bool:
    iterations = args.s1_iterations if stage == 1 else args.s2_iterations
    exp_id = f"{args.prefix}_stage{stage}_{iterations}"
    exp_dir = ROOT / "exps" / exp_id
    configs = sorted(exp_dir.glob("train-*-hydra/config.yaml"))
    checks[f"s{stage}_hydra_configs"] = [str(path) for path in configs]
    if not configs:
        if required:
            issues.append(f"S{stage} Hydra config missing: {exp_dir}")
        return False

    expected = expected_stage_config(args, stage)
    for path in configs:
        try:
            cfg = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError) as exc:
            issues.append(f"cannot parse {path}: {exc}")
            continue
        drift = {
            key: {"actual": cfg.get(key), "expected": value}
            for key, value in expected.items()
            if cfg.get(key) != value
        }
        expected_schedule = [320000, 360000] if stage == 1 else [999999, 999999]
        if cfg.get("lr_schedule_steps") != expected_schedule:
            drift["lr_schedule_steps"] = {
                "actual": cfg.get("lr_schedule_steps"),
                "expected": expected_schedule,
            }
        for split in ("AudioCaps_npz", "AudioCaps_val_npz"):
            data_cfg = (cfg.get("data") or {}).get(split) or {}
            expected_data = {
                "tsv": str(args.train_tsv),
                "gt_cache": str(args.gt_cache),
                "npz_dir": str(args.npz_dir),
            }
            for key, value in expected_data.items():
                if data_cfg.get(key) != value:
                    drift[f"data.{split}.{key}"] = {
                        "actual": data_cfg.get(key), "expected": value
                    }
        if drift:
            issues.append(f"S{stage} Hydra drift in {path}: {drift}")
    checks[f"s{stage}_runtime_contract"] = not any(
        issue.startswith(f"S{stage} Hydra") for issue in issues
    )
    return True


def check_eval(
    issues: list[str], checks: dict[str, Any], args: argparse.Namespace, required: bool,
) -> bool:
    exp_s2 = f"{args.prefix}_stage2_{args.s2_iterations}"
    log = LOG_ROOT / f"{exp_s2}_musiccaps_eval.log"
    checks["eval_log"] = str(log)
    if not log.is_file():
        if required:
            issues.append(f"eval log missing: {log}")
        return False
    text = clean_ansi(log.read_text(errors="replace")[:512_000])
    args_line = next((line for line in text.splitlines() if "Eval args:" in line), "")
    required_tokens = [
        "'no_q': True",
        "'no_text_attention_mask': True",
        f"'model_path': '{ROOT}/exps/{exp_s2}/{exp_s2}_ema_final.pth'",
        f"'tsv': '{args.eval_tsv}'",
        "'use_meanflow': True",
        "'cfg_strength': 0.5",
        "'num_steps': 1",
    ]
    missing = [token for token in required_tokens if token not in args_line]
    checks["eval_noq_contract"] = not missing
    if missing:
        issues.append(f"eval argument drift; missing tokens {missing}; line={args_line[:500]}")
    return True


def check_final_outputs(
    issues: list[str], checks: dict[str, Any], args: argparse.Namespace,
) -> None:
    exp_s1 = f"{args.prefix}_stage1_{args.s1_iterations}"
    exp_s2 = f"{args.prefix}_stage2_{args.s2_iterations}"
    s1 = ROOT / "exps" / exp_s1 / f"{exp_s1}_ckpt_last.pth"
    s2 = ROOT / "exps" / exp_s2 / f"{exp_s2}_ckpt_last.pth"
    ema = ROOT / "exps" / exp_s2 / f"{exp_s2}_ema_final.pth"
    for label, path in (("S1 ckpt", s1), ("S2 ckpt", s2), ("S2 EMA", ema)):
        checks[label] = str(path)
        if not path.is_file():
            issues.append(f"missing {label}: {path}")

    audio_dir = ROOT / "eval_output" / f"{exp_s2}_musiccaps" / "audio"
    metrics = ROOT / "eval_output/metrics" / f"{exp_s2}_musiccaps" / "metrics.txt"
    try:
        with args.eval_tsv.open() as handle:
            expected_audio = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        issues.append(f"cannot count eval TSV: {exc}")
        expected_audio = None
    actual_audio = len(list(audio_dir.glob("*.flac"))) if audio_dir.is_dir() else 0
    checks["eval_audio"] = {"actual": actual_audio, "expected": expected_audio}
    if expected_audio is None or actual_audio != expected_audio:
        issues.append(f"generated audio count={actual_audio}, expected={expected_audio}")

    if not metrics.is_file():
        issues.append(f"metrics missing: {metrics}")
        return
    metric_text = metrics.read_text()
    match = re.search(r"^clap_score:\s*([-+0-9.eE]+)$", metric_text, re.MULTILINE)
    if not match:
        issues.append("metrics.txt lacks clap_score")
        return
    clap = float(match.group(1))
    checks["clap_score"] = clap
    if not math.isfinite(clap):
        issues.append(f"CLAP is non-finite: {clap}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--s1-iterations", type=int, default=400000)
    parser.add_argument("--s2-iterations", type=int, default=200000)
    parser.add_argument("--expected-rows", type=int, default=251599)
    parser.add_argument("--train-tsv", type=Path, default=DEFAULT_TRAIN_TSV)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--eval-tsv", type=Path, default=DEFAULT_EVAL_TSV)
    parser.add_argument(
        "--phase", choices=("preflight", "auto", "s1", "s2", "eval", "final"),
        default="auto",
    )
    parser.add_argument("--require-launch-contract", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    issues: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    check_static_q_routing(issues, checks)
    check_data_gate(
        issues, checks, args.expected_rows, args.train_tsv, args.gt_cache, args.npz_dir
    )
    check_launch_contract(issues, warnings, checks, args.prefix, args)

    if args.phase != "preflight":
        require_s1 = args.phase in ("s1", "s2", "eval", "final")
        require_s2 = args.phase in ("s2", "eval", "final")
        s1_present = check_stage_configs(issues, checks, args, 1, require_s1)
        s2_present = check_stage_configs(issues, checks, args, 2, require_s2)
        require_eval = args.phase in ("eval", "final")
        eval_present = check_eval(issues, checks, args, require_eval)
        checks["observed_phase"] = (
            "eval" if eval_present else "s2" if s2_present else "s1" if s1_present else "pending"
        )
        if args.phase == "final":
            check_final_outputs(issues, checks, args)

    payload = {
        "status": "passed" if not issues else "failed",
        "audited_at": now_iso(),
        "phase_requested": args.phase,
        "prefix": args.prefix,
        "contract": "S1 NoQ + S2 NoQ + eval --no_q + NoMask",
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
    }
    if args.json_out:
        atomic_json(args.json_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
