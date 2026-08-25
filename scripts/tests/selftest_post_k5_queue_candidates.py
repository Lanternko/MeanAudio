#!/usr/bin/env python3
"""Static no-launch checks for staged 000-999 queue candidates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
STAGED = ROOT / "scripts/queue_candidates"
CONTRACTS = [
    ROOT / "docs/experiments/caption2p0_true_random_quarter_cfg0_contract.json",
    ROOT / "docs/experiments/caption2p0_fair013_best_full_cfg0_contract.json",
    ROOT / "docs/experiments/caption2p0_slot2_full_cfg0_contract.json",
    ROOT / "docs/experiments/caption2p0_fair013_k3_full_cfg0_contract.json",
]
NAMES = [
    "021_true_random_quarter.sh",
    "022_fair013_best_full.sh",
    "023_slot2_full.sh",
    "024_fair013_k3_full.sh",
]
QROOT = Path("/home/kojiek/gpu_queue")
QUEUE_STATES = ("pending", "running", "done", "failed", "held")
PROTECTED = {
    "010_s2q_k3.sh": "2abd26e5c696ba43d959109b5ec7f245f7ccda1f4f1a988ce3f15e15a49f1d59",
    "020_s2q_k5.sh": "e9191395d97763149e2f0810c9e527a297d4adb44f08abab3db6d871d94ba691",
}


def locate_queue_script(name: str) -> Path:
    """Jobs migrate pending -> running -> done/failed; find the script wherever it is now."""
    found = [QROOT / f"p2/{state}" / name for state in QUEUE_STATES
             if (QROOT / f"p2/{state}" / name).is_file()]
    assert len(found) == 1, f"expected {name} in exactly one p2 queue state, found {len(found)}"
    return found[0]
# All four were authorized and installed into p2/pending on 2026-08-25.
PENDING = Path("/home/kojiek/gpu_queue/p2/pending")
INSTALLED = {name: PENDING / name for name in (
    "021_true_random_quarter.sh",
    "022_fair013_best_full.sh",
    "023_slot2_full.sh",
    "024_fair013_k3_full.sh",
)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert NAMES == sorted(NAMES)
    assert all(re.fullmatch(r"[0-9]{3}_[a-z0-9_]+\.sh", name) for name in NAMES)
    for name, expected in PROTECTED.items():
        assert digest(locate_queue_script(name)) == expected
    for name, contract_path in zip(NAMES, CONTRACTS):
        launcher = STAGED / name
        assert launcher.is_file() and not launcher.is_symlink()
        text = launcher.read_text()
        assert "eval " not in text and "source " not in text
        assert "discord_webhook" not in text and "http://" not in text and "https://" not in text
        contract = json.loads(contract_path.read_text())
        assert contract["queue_name"] == name
        assert contract["queue_role"] == "p2"
        if name in INSTALLED:
            # Authorized: contract must be launchable and the installed copy byte-identical.
            assert contract["launch_allowed"] is True
            assert contract["corpus_gate"]["status"] == "passed"
            installed = INSTALLED[name]
            assert installed.is_file() and not installed.is_symlink()
            assert digest(installed) == contract["bindings"]["launcher_sha256"]
            assert digest(launcher) == contract["bindings"]["launcher_sha256"]
            assert digest(Path(contract["bindings"]["action"])) == contract["bindings"]["action_sha256"]
        else:
            assert contract["launch_allowed"] is False
            assert contract["corpus_gate"]["status"] != "passed"
            assert not (Path("/home/kojiek/gpu_queue/p2/pending") / name).exists()
        assert contract["fixed_protocol"] == {
            "tsv": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
            "tsv_sha256": "de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f",
            "rows": 5521, "solver": "MeanFlow", "steps": 25, "cfg_strength": 0,
            "seed": 42, "mask": "NoMask", "precision": "full",
        }
        assert contract["commands"]["run"][0] == "/bin/bash"
        assert len(contract["commands"]["run"]) == 2
        assert contract["commands"]["run"][1].startswith(str(ROOT / "scripts/training_pipelines/"))
        serialized = json.dumps(contract)
        assert "webhook" not in serialized.lower() and "http://" not in serialized and "https://" not in serialized
    print("[SELFTEST OK] 021/022/023/024 installed and byte-bound to their contracts")


if __name__ == "__main__":
    main()
