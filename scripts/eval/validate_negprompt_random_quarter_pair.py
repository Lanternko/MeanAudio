#!/usr/bin/env python3
"""No-GPU preflight for the 013 random-quarter CFG-1.5+negative evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


CONTRACT = Path(
    "/home/kojiek/MeanAudio/docs/experiments/"
    "caption2p0_random_quarter_neg_cfg1p5_contract.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("launch_allowed") is not True:
        raise SystemExit("[HOLD] launch_allowed is not true")
    protocol = contract["protocol"]
    expected = {
        "dataset": "MusicCaps",
        "rows": 5521,
        "solver": "MeanFlow",
        "steps": 25,
        "cfg_strength": 1.5,
        "seed": 42,
        "mask": "NoMask",
        "precision": "full",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise SystemExit(f"[HOLD] protocol {key} mismatch")
    for record in contract["inputs"]:
        path = Path(record["path"])
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"[HOLD] missing/non-regular input: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise SystemExit(f"[HOLD] size mismatch: {path}")
        if sha256(path) != record["sha256"]:
            raise SystemExit(f"[HOLD] hash mismatch: {path}")
    if "--structural-only" not in sys.argv[1:]:
        queue_root = Path("/home/kojiek/gpu_queue/p2")
        for dependency in contract.get("ordering_dependencies") or []:
            terminal = [
                queue_root / state / dependency
                for state in ("done", "failed")
                if (queue_root / state / dependency).is_file()
            ]
            if len(terminal) != 1:
                raise SystemExit(f"[HOLD] ordering dependency is not terminal: {dependency}")
    stats = os.statvfs("/home/kojiek/nvme_experiment_artifacts/meanaudio")
    free = stats.f_bavail * stats.f_frsize
    if free < int(contract["storage"]["hard_stop_free_bytes"]):
        raise SystemExit(f"[HOLD] storage hard stop: free_bytes={free}")
    print(json.dumps({"status": "passed", "free_bytes": free}, sort_keys=True))


if __name__ == "__main__":
    main()
