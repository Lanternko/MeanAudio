#!/usr/bin/env python3
"""Migrate an accepted 100k S1 prefix into a 400k Full continuation."""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import tempfile
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def migrate(checkpoint: dict, *, seed: int, source_sha256: str) -> dict:
    required = {"it", "weights", "optimizer", "scheduler", "ema"}
    missing = sorted(required.difference(checkpoint))
    if missing or int(checkpoint.get("it", -1)) != 100000:
        raise ValueError(f"invalid 100k S1 checkpoint: missing={missing} it={checkpoint.get('it')}")
    scheduler = checkpoint["scheduler"]
    children = scheduler.get("_schedulers")
    if not isinstance(children, list) or len(children) < 2:
        raise ValueError("unexpected scheduler structure")
    step_scheduler = children[1]
    old = step_scheduler.get("milestones")
    if old != collections.Counter({999999: 2}):
        raise ValueError(f"unexpected quarter milestones: {old}")
    step_scheduler["milestones"] = collections.Counter({320000: 1, 360000: 1})

    torch.manual_seed(seed)
    trainer_rng = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    trainer_rng.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint["trainer_rng_state"] = trainer_rng.get_state()
    checkpoint["torch_rng_state"] = torch.get_rng_state()
    checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    checkpoint["resume_provenance"] = {
        "kind": "accepted_quarter_s1_100k_to_full_400k_continuation",
        "source_checkpoint_sha256": source_sha256,
        "rng_boundary_iteration": 100000,
        "rng_restart_seed": seed,
        "old_lr_milestones": [999999, 999999],
        "new_lr_milestones": [320000, 360000],
        "claim_boundary": "not an uninterrupted-from-zero full run",
    }
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    actual = sha256(args.source)
    if actual != args.expected_source_sha256:
        raise SystemExit(f"source checkpoint hash mismatch: {actual}")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    migrated = migrate(checkpoint, seed=args.seed, source_sha256=actual)
    atomic_save(migrated, args.output)
    verified = torch.load(args.output, map_location="cpu", weights_only=True)
    if verified.get("resume_provenance") != migrated["resume_provenance"]:
        raise SystemExit("migrated checkpoint verification failed")
    print(f"[PASS] migrated 100k S1 prefix to Full continuation: {args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
