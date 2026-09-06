#!/usr/bin/env python3
"""Fail-closed audit for one fixed Phase-8 Qwen caption-dose arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from phase8_qwen_dose_provenance import validate_nested_cache_provenance


REQUIRED_METRICS = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
REPO = Path(__file__).resolve().parents[2]
TORCHRUN = "/home/kojiek/venvs/dac/bin/torchrun"
EVALUATOR = "/home/kojiek/research/meanaudio_eval/phase4_eval.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in REQUIRED_METRICS:
            values[key] = float(raw.strip())
    return values


def tsv_id_sha256(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        ids = [str(row["id"]) for row in csv.DictReader(handle, delimiter="\t")]
    return len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest()


def first_nonfinite(root: Any, label: str) -> str | None:
    stack = [(label, root)]
    while stack:
        current_label, value = stack.pop()
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return current_label
        if isinstance(value, dict):
            stack.extend((f"{current_label}.{key}", child) for key, child in value.items())
        elif isinstance(value, (list, tuple)):
            stack.extend((f"{current_label}[{index}]", child) for index, child in enumerate(value))
    return None


def audit(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[str] = []
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    milestone = next(
        (item for item in contract.get("milestones", []) if item.get("final_iteration") == args.expected_iteration),
        None,
    )
    if milestone is None or milestone.get("source_iteration") != args.source_iteration:
        issues.append("contract milestone/source iteration drift")

    try:
        source_state = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        if source_state.get("it") != args.source_iteration:
            issues.append(f"source it={source_state.get('it')}, expected={args.source_iteration}")
        source_sha256 = sha256_file(args.source_checkpoint)
        del source_state
    except Exception as exc:
        source_sha256 = None
        issues.append(f"invalid source checkpoint: {exc!r}")

    checkpoint = args.run_dir / f"{args.exp_id}_ckpt_last.pth"
    ema = args.run_dir / f"{args.exp_id}_ema_final.pth"
    checkpoint_sha256 = None
    ema_sha256 = None
    if not checkpoint.is_file():
        issues.append(f"missing checkpoint: {checkpoint}")
    else:
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state.get("it") != args.expected_iteration:
                issues.append(f"checkpoint it={state.get('it')}, expected={args.expected_iteration}")
            for root_name in ("weights", "ema", "optimizer", "scheduler"):
                if root_name not in state:
                    issues.append(f"checkpoint missing {root_name}")
                    continue
                bad = first_nonfinite(state[root_name], root_name)
                if bad:
                    issues.append(f"non-finite {bad}")
            del state
            checkpoint_sha256 = sha256_file(checkpoint)
        except Exception as exc:
            issues.append(f"invalid checkpoint: {exc!r}")
    if not ema.is_file():
        issues.append(f"missing final EMA: {ema}")
    else:
        ema_sha256 = sha256_file(ema)

    metric_values = parse_metrics(args.metrics)
    if set(metric_values) != REQUIRED_METRICS or not all(math.isfinite(v) for v in metric_values.values()):
        issues.append(f"invalid metrics: {metric_values}")

    audio_count = 0
    if args.audio_tsv.is_file() and args.audio_dir.is_dir():
        with args.audio_tsv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        audio_count = sum((args.audio_dir / f"{row['id']}.flac").is_file() for row in rows)
    if audio_count != args.expected_audio_count:
        issues.append(f"audio manifest count={audio_count}, expected={args.expected_audio_count}")

    expected_training = {
        "batch_size": 8,
        "accumulation_steps": 1,
        "learning_rate": 1e-5,
        "seed": 14159265,
        "use_q_conditioning": False,
        "use_text_attention_mask": False,
        "use_rope": False,
        "multi_cap": False,
    }
    drift = {
        key: (contract.get("training", {}).get(key), expected)
        for key, expected in expected_training.items()
        if contract.get("training", {}).get(key) != expected
    }
    if drift:
        issues.append(f"contract training drift: {drift}")

    arm_cfg = contract["arms"][args.arm]
    data_files = {
        "tsv_sha256": Path(arm_cfg["tsv"]),
        "cache_list_sha256": Path(arm_cfg["cache_list"]),
        "cache_manifest_sha256": args.cache_manifest,
    }
    for key, path in data_files.items():
        if not path.is_file() or sha256_file(path) != arm_cfg.get(key):
            issues.append(f"input provenance drift for {key}: {path}")
    try:
        row_count, id_hash = tsv_id_sha256(Path(arm_cfg["tsv"]))
        if row_count != 251_599 or id_hash != arm_cfg.get("tsv_id_sha256"):
            issues.append(f"training TSV row/order drift: rows={row_count}, id_sha256={id_hash}")
        eval_rows, eval_id_hash = tsv_id_sha256(args.audio_tsv)
        evaluation = contract["evaluation"]
        if (
            evaluation.get("tsv") != str(args.audio_tsv)
            or evaluation.get("tsv_sha256") != sha256_file(args.audio_tsv)
            or evaluation.get("tsv_id_sha256") != eval_id_hash
            or eval_rows != args.expected_audio_count
        ):
            issues.append("evaluation TSV provenance drift")
    except Exception as exc:
        issues.append(f"invalid TSV provenance: {exc!r}")

    predecessor: dict[str, Any] = {}
    try:
        predecessor = json.loads(args.source_audit.read_text(encoding="utf-8"))
        if predecessor.get("status") != "passed":
            issues.append("source arm audit is not passed")
        if args.source_iteration != 620_000 and predecessor.get("checkpoint") != str(args.source_checkpoint):
            issues.append("source audit checkpoint path does not match continuation source")
    except Exception as exc:
        issues.append(f"invalid source audit: {exc!r}")
    try:
        cache = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
        if cache.get("status") != "passed":
            issues.append("cache manifest/gate is not passed")
        validate_nested_cache_provenance(args.arm, Path(arm_cfg["npz_dir"]), cache)
    except Exception as exc:
        issues.append(f"invalid cache manifest: {exc!r}")

    try:
        execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
        if execution.get("contract_sha256") != sha256_file(args.contract):
            issues.append("execution manifest contract hash drift")
        if args.source_iteration == 620_000:
            expected_source_sha = execution.get("preflight", {}).get(
                "source_checkpoint_sha256", {}
            ).get(args.arm)
        else:
            expected_source_sha = predecessor.get("checkpoint_sha256")
        if expected_source_sha != source_sha256:
            issues.append(
                f"source checkpoint hash is not bound: expected={expected_source_sha}, actual={source_sha256}"
            )

        expected_train = [
            TORCHRUN, "--standalone", "--nproc_per_node=1", str(REPO / "train.py"),
            "data=meanaudio", "model=meanaudio_s",
            f"exp_id={args.exp_id}",
            f"num_iterations={args.expected_iteration}",
            "lr_schedule=step", "lr_schedule_steps=[999999,999999]",
            "batch_size=8", "learning_rate=1e-5", "linear_warmup_steps=1000",
            "seed=14159265", "num_workers=4", "save_weights_interval=10000",
            "save_checkpoint_interval=10000", "val_interval=999999",
            "eval_interval=999999", "save_eval_interval=999999",
            "+accumulation_steps=1", "+use_rope=False", "+use_q_conditioning=false",
            "+use_text_attention_mask=false", "+use_wandb=false", "++multi_cap=false",
            f"hydra.run.dir={args.run_dir}",
            f"++data.AudioCaps_npz.tsv={arm_cfg['tsv']}",
            f"++data.AudioCaps_npz.npz_dir={arm_cfg['npz_dir']}",
            f"++data.AudioCaps_npz.gt_cache={arm_cfg['cache_list']}",
            f"++data.AudioCaps_val_npz.tsv={arm_cfg['tsv']}",
            f"++data.AudioCaps_val_npz.npz_dir={arm_cfg['npz_dir']}",
            f"++data.AudioCaps_val_npz.gt_cache={arm_cfg['cache_list']}",
            f"checkpoint={args.source_checkpoint}",
        ]
        expected_eval = [
            sys.executable, str(REPO / "eval.py"), "--variant", "meanaudio_s",
            "--model_path", str(ema), "--output", str(args.audio_dir),
            "--tsv", str(args.audio_tsv), "--use_meanflow", "--num_steps", "1",
            "--encoder_name", "t5_clap", "--text_c_dim", "512", "--cfg_strength", "0.5",
            "--no_q", "--full_precision", "--no_text_attention_mask",
        ]
        expected_metrics = [
            sys.executable, EVALUATOR, "--gen_dir", str(args.audio_dir),
            "--tsv", str(args.audio_tsv), "--exp_name", args.exp_id,
            "--out_dir", str(args.metrics.parent.parent), "--num_samples", "5521",
        ]
        expected_commands = {
            args.train_step: expected_train,
            args.eval_step: expected_eval,
            args.metrics_step: expected_metrics,
        }
        for step_name, expected_command in expected_commands.items():
            step = execution.get("steps", {}).get(step_name, {})
            if (
                step.get("status") != "passed"
                or step.get("exit_code") != 0
                or step.get("command") != expected_command
            ):
                issues.append(f"exact execution provenance invalid for {step_name}")
    except Exception as exc:
        issues.append(f"invalid execution manifest: {exc!r}")

    result = {
        "schema_version": 1,
        "status": "passed" if not issues else "failed",
        "arm": args.arm,
        "exp_id": args.exp_id,
        "source_iteration": args.source_iteration,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "final_iteration": args.expected_iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "ema": str(ema),
        "ema_sha256": ema_sha256,
        "metrics": metric_values,
        "audio_manifest_count": audio_count,
        "issues": issues,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.json_out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "qwen"), required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--train-step", required=True)
    parser.add_argument("--eval-step", required=True)
    parser.add_argument("--metrics-step", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--audio-tsv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--expected-audio-count", type=int, default=5521)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
