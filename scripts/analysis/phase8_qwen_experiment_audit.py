#!/usr/bin/env python3
"""Audit one final 20k arm and fail closed on config/checkpoint drift."""

from __future__ import annotations

import argparse
import json
import math
import csv
from pathlib import Path
from typing import Any

import torch


def metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(raw.strip())
    return values


def audit(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[str] = []
    run_dir = args.run_dir
    ckpt = run_dir / f"{args.exp_id}_ckpt_last.pth"
    ema = run_dir / f"{args.exp_id}_ema_final.pth"
    if not ckpt.is_file():
        issues.append(f"missing checkpoint: {ckpt}")
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state.get("it") != 620_000:
            issues.append(f"checkpoint it={state.get('it')}, expected 620000")
        if state.get("weights") is None or state.get("ema") is None:
            issues.append("checkpoint missing weights/EMA")
        for root_name in ("weights", "ema", "optimizer"):
            root = state.get(root_name, {})
            stack = [(root_name, root)]
            while stack:
                label, value = stack.pop()
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    issues.append(f"non-finite {label}")
                    break
                if isinstance(value, dict):
                    stack.extend((f"{label}.{key}", child) for key, child in value.items())
    if not ema.is_file():
        issues.append(f"missing final EMA: {ema}")
    metric_values = metrics(args.metrics)
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if set(metric_values) != required or not all(math.isfinite(value) for value in metric_values.values()):
        issues.append(f"invalid metrics: {metric_values}")
    audio_count = 0
    if args.audio_tsv.is_file() and args.audio_dir.is_dir():
        with args.audio_tsv.open(encoding="utf-8", newline="") as handle:
            eval_rows = list(csv.DictReader(handle, delimiter="\t"))
        audio_count = sum((args.audio_dir / f"{row['id']}.flac").is_file() for row in eval_rows)
    if audio_count != args.expected_audio_count:
        issues.append(f"audio manifest count={audio_count}, expected={args.expected_audio_count}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = {
        "seed": 14159265,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "final_iteration": 620000,
        "use_q_conditioning": False,
        "use_text_attention_mask": False,
        "multi_cap": False,
    }
    drift = {key: (contract.get("training", {}).get(key), value) for key, value in expected.items() if contract.get("training", {}).get(key) != value}
    if drift:
        issues.append(f"contract drift: {drift}")
    try:
        init_manifest = json.loads(args.init_manifest.read_text(encoding="utf-8"))
        declared_source = contract["inputs"]["source_checkpoint_sha256"]
        if init_manifest.get("source_sha256") != declared_source:
            issues.append("initializer source hash drift")
        if not all(init_manifest.get(key) is True for key in (
            "weights_preserved", "ema_preserved", "optimizer_reset", "scheduler_reset"
        )):
            issues.append("initializer tensor/reset audit did not pass")
    except Exception as exc:
        issues.append(f"invalid initializer manifest: {exc!r}")
    try:
        cache_manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
        if cache_manifest.get("status") != "passed":
            issues.append("cache manifest/gate is not passed")
    except Exception as exc:
        issues.append(f"invalid cache manifest/gate: {exc!r}")
    try:
        execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
        step = execution.get("steps", {}).get(f"{args.arm}_train", {})
        launched = [str(value) for value in step.get("command", [])]
        required_tokens = {
            "/home/kojiek/venvs/dac/bin/torchrun",
            "--standalone",
            "--nproc_per_node=1",
            f"exp_id={args.exp_id}",
            "num_iterations=620000",
            "batch_size=8",
            "learning_rate=1e-5",
            "seed=14159265",
            "+use_q_conditioning=false",
            "+use_text_attention_mask=false",
            "++multi_cap=false",
        }
        missing_tokens = sorted(required_tokens - set(launched))
        if step.get("status") != "passed" or missing_tokens:
            issues.append(f"actual train launch manifest invalid; missing={missing_tokens}")
    except Exception as exc:
        issues.append(f"invalid execution manifest: {exc!r}")
    result = {
        "schema_version": 1,
        "status": "passed" if not issues else "failed",
        "arm": args.arm,
        "exp_id": args.exp_id,
        "checkpoint": str(ckpt),
        "ema": str(ema),
        "metrics": metric_values,
        "audio_manifest_count": audio_count,
        "issues": issues,
    }
    if args.json_out:
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
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--audio-tsv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--expected-audio-count", type=int, default=5521)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--init-manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
