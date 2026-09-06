#!/usr/bin/env python3
"""Validated eval-only recovery runner for failed canonical CFG0 queue entries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
QUEUE_ROOT = Path("/home/kojiek/gpu_queue/p2")
WRAPPER = ROOT / "scripts/caption10s_pipeline/eval_musiccaps_mf25.sh"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("launch_allowed") is not True or value.get("recovery_kind") != "eval_only":
        raise ValueError("contract is not an authorized eval-only recovery")
    cells = value.get("cells") or []
    if len(cells) != 1 or cells[0].get("conditioning") != "no_q":
        raise ValueError("exactly one NoQ recovery cell is required")
    protocol = value.get("fixed_protocol") or {}
    required = {
        "rows": 5521, "solver": "MeanFlow", "steps": 25, "cfg_strength": 0,
        "seed": 42, "mask": "NoMask", "precision": "full",
    }
    drift = {key: (protocol.get(key), expected) for key, expected in required.items()
             if protocol.get(key) != expected}
    if drift:
        raise ValueError(f"canonical protocol drift: {drift}")
    cell = cells[0]
    label = str(cell.get("label") or "")
    if not label.endswith("_mf25_cfg0_noq"):
        raise ValueError("NoQ label suffix drift")
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),):
        target = Path(str(cell[path_key]))
        if not target.is_file() or target.is_symlink() or digest(target) != cell[hash_key]:
            raise ValueError(f"{path_key} identity drift")
    tsv = Path(str(protocol["tsv"]))
    if not tsv.is_file() or tsv.is_symlink() or digest(tsv) != protocol["tsv_sha256"]:
        raise ValueError("TSV identity drift")
    bindings = value.get("bindings") or {}
    if digest(WRAPPER) != bindings.get("wrapper_sha256"):
        raise ValueError("registered CFG0 wrapper drift")
    return value


def dependency_gate(contract: dict) -> None:
    sys.path.insert(0, "/home/kojiek/gpu_queue")
    from lib_scheduler import terminal_notification_evidence_ok

    # The dependency's receipt must be validated against the dependency's own
    # contract. lib_scheduler.discover_contract() prefers an ambient
    # GPU_QUEUE_CONTRACT, which here is *this* job's contract, so drop it for
    # the duration of the check and let the dependency launcher pin its own.
    inherited = os.environ.pop("GPU_QUEUE_CONTRACT", None)
    try:
        for name in contract.get("ordering_dependencies") or []:
            matches = [QUEUE_ROOT / state / name for state in ("done", "failed", "interrupted")]
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise ValueError(f"ordering dependency is not terminal: {name}")
            if not terminal_notification_evidence_ok(matches[0]):
                raise ValueError(f"ordering dependency notification is not delivered: {name}")
    finally:
        if inherited is not None:
            os.environ["GPU_QUEUE_CONTRACT"] = inherited


def storage_gate(contract: dict) -> int:
    runtime = contract["runtime_storage"]
    root = Path(runtime["output_root"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    hard = int(runtime["hard_stop_free_bytes"])
    if free < hard:
        raise ValueError(f"storage hard stop: free_bytes={free} hard_stop={hard}")
    return free


def cleanup_partial_audio(contract: dict) -> None:
    cell = contract["cells"][0]
    report = Path(cell["report"])
    label = cell["label"]
    runtime = contract["runtime_storage"]
    output_root = Path(runtime["output_root"]).resolve()
    audio = output_root / label / "audio"
    metrics = Path(runtime["metrics_root"]) / label / "metrics.txt"
    if report.is_file():
        return
    if metrics.exists():
        raise ValueError("metrics exist without an atomic final report")
    if not audio.exists():
        return
    if audio.is_symlink() or not audio.is_dir() or audio.resolve().parent.parent != output_root:
        raise ValueError("unsafe partial audio directory")
    tsv = Path(contract["fixed_protocol"]["tsv"])
    with tsv.open(encoding="utf-8", newline="") as handle:
        expected = {f"{row['id']}.flac" for row in csv.DictReader(handle, delimiter="\t")}
    entries = list(audio.iterdir())
    for path in entries:
        info = path.lstat()
        if (path.name not in expected or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid() or info.st_nlink != 1):
            raise ValueError(f"unsafe partial artifact: {path}")
    for path in entries:
        path.unlink()
    audio.rmdir()
    print(f"RECOVERY_CLEANUP partial_flacs={len(entries)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    contract_path = Path(os.environ["GPU_QUEUE_CONTRACT"]).resolve()
    contract = load_contract(contract_path)
    dependency_gate(contract)
    free = storage_gate(contract)
    if args.preflight:
        print(json.dumps({"status": "passed", "free_bytes": free}, sort_keys=True))
        return 0
    cleanup_partial_audio(contract)
    cell = contract["cells"][0]
    env = os.environ.copy()
    env.update(CFG0_CONTRACT=str(contract_path), CFG0_ARM=cell["cell_id"])
    return subprocess.run(
        [str(WRAPPER), cell["label"], cell["checkpoint"], "--no_q"], env=env,
    ).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError) as exc:
        print(f"HOLD cfg0 recovery preflight: {exc}", file=sys.stderr)
        raise SystemExit(2)
