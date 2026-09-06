#!/usr/bin/env python3
"""Fail-closed verifier for a cooperative training-pause checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


REQUIRED = {
    "it", "weights", "optimizer", "scheduler", "ema",
    "trainer_rng_state", "torch_rng_state", "cuda_rng_state_all",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int)
    args = parser.parse_args()

    ack = json.loads(args.ack.read_text())
    if ack.get("status") != "paused" or Path(str(ack.get("checkpoint"))) != args.checkpoint:
        raise SystemExit("pause acknowledgment does not bind the checkpoint")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size != ack.get("checkpoint_bytes"):
        raise SystemExit("checkpoint size does not match pause acknowledgment")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing = sorted(REQUIRED.difference(checkpoint))
    if missing:
        raise SystemExit(f"checkpoint lacks resume state: {missing}")
    iteration = int(checkpoint["it"])
    if iteration != int(ack.get("iteration", -1)):
        raise SystemExit("checkpoint iteration does not match pause acknowledgment")
    if args.expected_iteration is not None and iteration != args.expected_iteration:
        raise SystemExit(f"iteration {iteration} != expected {args.expected_iteration}")
    if not isinstance(checkpoint["trainer_rng_state"], torch.Tensor):
        raise SystemExit("trainer RNG state is not a tensor")
    atomic_json(args.report, {
        "status": "passed",
        "iteration": iteration,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "required_keys": sorted(REQUIRED),
    })
    print(f"[PASS] cooperative pause checkpoint iteration={iteration}")


if __name__ == "__main__":
    main()
