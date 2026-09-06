#!/usr/bin/env python3
"""No-GPU preflight for c2p0 slot0 CFG-2.5/4.0 full-5521 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


CONTRACT = Path("/home/kojiek/MeanAudio/docs/experiments/c2p0_slot0_neg_cfg2p5_cfg4p0_full5521_contract.json")
HARN_BUNDLE = Path("/home/kojiek/MeanAudio/docs/experiments/harn/c2p0_slot0_neg_cfg_full5521")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    specification = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if specification.get("launch_allowed") is not True:
        raise SystemExit("[HOLD] launch is not authorized")
    protocol = specification["protocol"]
    expected = {
        "classification": "secondary_noncanonical", "dataset": "MusicCaps", "rows": 5521,
        "solver": "MeanFlow", "steps": 25, "cfg_strengths": [2.5, 4.0], "seed": 42,
        "mask": "NoMask", "precision": "full", "conditioning": "NoQ",
        "negative_prompt": "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise SystemExit(f"[HOLD] protocol mismatch: {key}")
    for record in specification["inputs"]:
        path = Path(record["path"])
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"[HOLD] missing/non-regular input: {path}")
        if path.stat().st_size != int(record["bytes"]) or digest(path) != record["sha256"]:
            raise SystemExit(f"[HOLD] input identity mismatch: {path}")
    if "--structural-only" not in sys.argv[1:]:
        sys.path.insert(0, "/home/kojiek/MeanAudio/scripts/experiment_harness")
        from secondary_cfg_sweep_queue_guest import require_028_terminal
        try:
            require_028_terminal()
        except ValueError as exc:
            raise SystemExit(f"[HOLD] ordering dependency invalid: {exc}") from exc
    stats = os.statvfs(specification["storage"]["path"])
    free = stats.f_bavail * stats.f_frsize
    if free < int(specification["storage"]["hard_stop_free_bytes"]):
        raise SystemExit(f"[HOLD] storage hard stop: free_bytes={free}")
    schema_check = subprocess.run([
        "/usr/bin/python3", "/home/kojiek/MeanAudio/scripts/validate_experiment_harness_documents.py",
        "--contract", str(HARN_BUNDLE / "contract.json"),
        "--preflight", str(HARN_BUNDLE / "preflight.json"),
        "--ledger", str(HARN_BUNDLE / "ledger.json"),
        "--queue", str(HARN_BUNDLE / "queue.json"),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if schema_check.returncode != 0:
        raise SystemExit("[HOLD] harn-schema-v1 bundle validation failed")
    print(json.dumps({"status": "passed", "free_bytes": free}, sort_keys=True))


if __name__ == "__main__":
    main()
