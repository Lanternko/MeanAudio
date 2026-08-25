#!/usr/bin/env python3
"""No-GPU, read-only preflight for the four post-K5 queue candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
QROOT = Path("/home/kojiek/gpu_queue")
CONTRACTS = {
    "true_random_quarter": ROOT / "docs/experiments/caption2p0_true_random_quarter_cfg0_contract.json",
    "fair013_best_full": ROOT / "docs/experiments/caption2p0_fair013_best_full_cfg0_contract.json",
    "slot2_full": ROOT / "docs/experiments/caption2p0_slot2_full_cfg0_contract.json",
    "fair013_k3_full": ROOT / "docs/experiments/caption2p0_fair013_k3_full_cfg0_contract.json",
}
PROTECTED = {
    "010_s2q_k3.sh": "2abd26e5c696ba43d959109b5ec7f245f7ccda1f4f1a988ce3f15e15a49f1d59",
    "020_s2q_k5.sh": "e9191395d97763149e2f0810c9e527a297d4adb44f08abab3db6d871d94ba691",
}
LEGACY_TRUE_RANDOM = QROOT / "p1/held/020_true_random.sh"
LEGACY_TRUE_RANDOM_SHA = "da383e6a39db0ebf8dcea3d76fd79254b304dcc31c8a2957c363c69344212989"
ALLOWED_INPUT_ROOTS = (
    ROOT,
    Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline"),
    Path("/home/kojiek/exps_nvme"),
    Path("/mnt/HDD/kojiek/phase4_jamendo_data"),
)


QUEUE_STATES = ("pending", "running", "done", "failed", "held")


def locate_queue_script(name: str) -> Path:
    """Find a queue script wherever the host has moved it to.

    Jobs migrate pending -> running -> done/failed as the queue advances, so
    pinning a protected script to one directory makes the guard fail on normal
    progression instead of on real byte drift.
    """
    found = [QROOT / f"p2/{state}" / name for state in QUEUE_STATES
             if (QROOT / f"p2/{state}" / name).is_file()]
    if len(found) != 1:
        raise ValueError(f"expected {name} in exactly one p2 queue state, found {len(found)}")
    return found[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, *, roots: tuple[Path, ...] = ALLOWED_INPUT_ROOTS) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"non-regular input: {path}")
    resolved = path.resolve(strict=True)
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"input outside allowlisted roots: {resolved}")


def verify_bound_file(record: dict, label: str) -> Path:
    path = Path(record["path"])
    require_regular(path)
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{label} hash drift: {actual}")
    return path


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def check_duplicate_true_random() -> None:
    require_regular(LEGACY_TRUE_RANDOM, roots=(QROOT,))
    if sha256(LEGACY_TRUE_RANDOM) != LEGACY_TRUE_RANDOM_SHA:
        raise ValueError("legacy true-random launcher drift")
    for state in ("pending", "running"):
        if any((QROOT / f"p1/{state}").glob("*true_random*.sh")):
            raise ValueError(f"legacy true-random is {state}")
    old = json.loads((ROOT / "docs/experiments/caption2p0_k3_true_vs_fake_random_quarter_contract.json").read_text())
    if old.get("launch_allowed") is True:
        raise ValueError("legacy true-random contract became launch-authorized")


def validate(candidate: str, require_launchable: bool) -> dict:
    contract_path = CONTRACTS[candidate]
    require_regular(contract_path)
    contract = json.loads(contract_path.read_text())
    if contract.get("queue_name", "")[:3] not in {"021", "022", "023", "024"}:
        raise ValueError("queue name is not registered three-digit order")
    if require_launchable:
        if contract.get("launch_allowed") is not True:
            raise ValueError("launch_allowed is not true")
        if (contract.get("corpus_gate") or {}).get("status") != "passed":
            raise ValueError("current full-corpus gate is not passed")
        for key in ("launcher_sha256", "action_sha256"):
            if str((contract.get("bindings") or {}).get(key, "")).startswith("pending"):
                raise ValueError(f"binding not frozen: {key}")
    for name, expected in PROTECTED.items():
        path = locate_queue_script(name)
        require_regular(path, roots=(QROOT,))
        if sha256(path) != expected:
            raise ValueError(f"protected queue drift: {path}")
    if candidate == "true_random_quarter":
        check_duplicate_true_random()
    sources = contract.get("sources") or {}
    for key, record in sources.items():
        if isinstance(record, dict) and {"path", "sha256"}.issubset(record):
            path = verify_bound_file(record, key)
            if record.get("rows") is not None:
                rows = csv_rows(path) if path.suffix == ".tsv" else sum(1 for line in path.open() if line.strip())
                if rows != int(record["rows"]):
                    raise ValueError(f"{key} row mismatch: {rows}")
    resume = contract["resume"]
    checkpoint = Path(resume["checkpoint"])
    require_regular(checkpoint)
    if checkpoint.stat().st_size != int(resume["checkpoint_bytes"]):
        raise ValueError("resume checkpoint size drift")
    if sha256(checkpoint) != resume["checkpoint_sha256"]:
        raise ValueError("resume checkpoint hash drift")
    return {"candidate": candidate, "status": "passed", "launchable": contract.get("launch_allowed") is True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--require-launchable", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.candidate, args.require_launchable), sort_keys=True))


if __name__ == "__main__":
    main()
