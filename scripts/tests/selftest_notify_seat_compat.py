#!/usr/bin/env python3
"""No-network regression tests for legacy and receipt-aware seat notifications."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
HELPER = ROOT / "scripts/experiment_harness/notification_receipts.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(
    queue: Path, experiment: str, status: str = "start", summary: str = "seat",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(NOTIFIER), "--status", status,
            "--experiment", experiment, "--summary", summary, "--exit-code", "0",
        ],
        env={**os.environ, "GPU_QUEUE_ROOT": str(queue), "MEANAUDIO_NOTIFY_DRY_RUN": "true"},
        text=True, capture_output=True,
    )


def main() -> None:
    temp = Path(tempfile.mkdtemp(prefix="seat-compat-test-"))
    queue = temp / "queue"
    pending = queue / "p2/pending"
    pending.mkdir(parents=True)

    legacy_contract = temp / "legacy.contract.json"
    legacy_contract.write_text(json.dumps({
        "experiment_id": "legacy-026", "run_id": "legacy-026-run", "launch_allowed": True,
    }))
    legacy = pending / "026_fake_random_full.sh"
    legacy.write_text(f"#!/bin/bash\n# GPU_QUEUE_CONTRACT={legacy_contract}\n")
    legacy.chmod(0o700)
    result = invoke(queue, "p2-seat-026_fake_random_full")
    assert result.returncode == 0, result.stderr
    assert "STARTED" in result.stdout
    assert "QUEUE HELD" not in result.stdout
    assert "**Experiment:** `026_fake_random_full`" in result.stdout
    assert "Not a hold, not a stop" in result.stdout
    assert "**GPU:**" in result.stdout
    assert "starting, not stopping" in result.stdout
    assert "durable seat_attempt" not in result.stdout
    assert not (queue / "notification_receipts/legacy-026").exists()

    # In-memory hosts still pass --status held for seating until restart.
    held_seat = invoke(queue, "p2-seat-026_fake_random_full", status="held")
    assert held_seat.returncode == 0, held_seat.stderr
    assert "STARTED" in held_seat.stdout
    assert "QUEUE HELD" not in held_seat.stdout

    real_hold = invoke(
        queue, "p2-held-026_fake_random_full", status="held",
        summary="preflight rc=1: input identity mismatch: foo.py",
    )
    assert real_hold.returncode == 0, real_hold.stderr
    assert "QUEUE HELD" in real_hold.stdout
    assert "STARTED" not in real_hold.stdout
    assert "**Experiment:** `026_fake_random_full`" in real_hold.stdout
    assert "Harness pin/hash mismatch" in real_hold.stdout
    assert "not storage, not a GPU race" in real_hold.stdout

    handoff = invoke(
        queue,
        "single-negprompt-cfg3-ablation-20260831:run-20260831-033-single-negprompt-cfg3-ablation:queue_handoff",
        status="held",
        summary="Starting 033. Preflight passed.",
    )
    assert handoff.returncode == 0, handoff.stderr
    assert "STARTED" in handoff.stdout
    assert "QUEUE HELD" not in handoff.stdout
    assert "**Experiment:** `033_single_negprompt_cfg3_ablation`" in handoff.stdout
    assert "queue handoff" in handoff.stdout
    assert "starting, not stopping" in handoff.stdout

    opt_contract = temp / "opt.contract.json"
    opt_contract.write_text(json.dumps({
        "experiment_id": "opt-028", "run_id": "opt-028-run", "launch_allowed": True,
        "notification_receipts": {
            "required": True,
            "root": str(queue / "notification_receipts"),
            "notifier": str(NOTIFIER), "notifier_sha256": sha(NOTIFIER),
            "helper": str(HELPER), "helper_sha256": sha(HELPER),
        },
    }))
    opt = pending / "028_random_quarter_neg_cfg1p5.sh"
    opt.write_text(f"#!/bin/bash\n# GPU_QUEUE_CONTRACT={opt_contract}\n")
    opt.chmod(0o700)
    first = invoke(queue, "p2-seat-028_random_quarter_neg_cfg1p5")
    second = invoke(queue, "p2-seat-028_random_quarter_neg_cfg1p5")
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    receipt = queue / "notification_receipts/opt-028/seat_attempt.json"
    record = json.loads(receipt.read_text())
    assert record["delivery_state"] == "delivered"
    assert "durable seat_attempt" in first.stdout and "durable seat_attempt" in second.stdout
    print(f"PASS legacy/opt-in seat compatibility {temp}")


if __name__ == "__main__":
    main()
