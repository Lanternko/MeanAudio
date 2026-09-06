#!/usr/bin/env python3
"""No-GPU immutable preflight for the 033 single-negative CFG3 ablation."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/kojiek/MeanAudio")
CONTRACT = ROOT / "docs/experiments/single_negprompt_cfg3_ablation_20260831_contract.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    spec = json.loads(CONTRACT.read_text())
    expected = {
        "dataset": "MusicCaps seeded subset", "rows": 1024, "subset_seed": 20260830,
        "solver": "MeanFlow", "steps": 25, "cfg_strength": 3.0,
        "generation_seed": 42, "mask": "NoMask", "precision": "full", "conditioning": "NoQ",
    }
    if spec.get("launch_allowed") is not True or any(spec["protocol"].get(k) != v for k, v in expected.items()):
        raise SystemExit("[HOLD] authorization/protocol mismatch")
    expected_cells = {
        "none": None, "low_quality": "low quality", "noisy": "noisy",
        "distorted": "distorted", "muffled": "muffled",
        "poor_fidelity": "poor fidelity", "hiss": "hiss", "lo_fi": "lo-fi",
        "amateur": "amateur", "genre": "genre",
        "fidelity_short": "low quality, noisy",
    }
    if spec["protocol"].get("negative_cells") != expected_cells:
        raise SystemExit("[HOLD] negative cell registry mismatch")
    for item in spec["inputs"]:
        path = Path(item["path"])
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise SystemExit(f"[HOLD] input identity mismatch: {path}")
    free = os.statvfs(spec["storage"]["path"]).f_bavail * os.statvfs(spec["storage"]["path"]).f_frsize
    if "--structural-only" not in sys.argv and free < spec["storage"]["hard_stop_free_bytes"]:
        raise SystemExit(f"[HOLD] storage hard stop: free_bytes={free}")
    print(json.dumps({"status": "passed", "free_bytes": free}))


if __name__ == "__main__":
    main()
