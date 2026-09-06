#!/usr/bin/env python3
"""No-network state-transition tests for storage-gate monitoring semantics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "scripts/experiment_harness/storage_gate_monitor.py"
OPERATIONAL_NOTIFIER = ROOT / "scripts/notify_operational_event_webhook.py"


def main() -> None:
    dry_env = {**os.environ, "MEANAUDIO_NOTIFY_DRY_RUN": "true"}
    advisory_title = subprocess.run([
        sys.executable, str(OPERATIONAL_NOTIFIER), "--status", "advisory",
        "--experiment", "test", "--summary", "gate only",
    ], env=dry_env, text=True, capture_output=True, check=True).stdout
    recovered_title = subprocess.run([
        sys.executable, str(OPERATIONAL_NOTIFIER), "--status", "recovered",
        "--experiment", "test", "--summary", "cleared",
    ], env=dry_env, text=True, capture_output=True, check=True).stdout
    assert "STORAGE GATE BLOCKED" in advisory_title and "QUEUE HELD" not in advisory_title
    assert "STORAGE GATE CLEARED" in recovered_title and "COMPLETED" not in recovered_title

    temp = Path(tempfile.mkdtemp(prefix="storage-monitor-test-"))
    queue_state = temp / "p2.state"
    queue_state.write_text("running 026_fake_random_full\n")
    running = temp / "p2.running.json"
    current_start = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
    running.write_text(json.dumps({"job_id": "026_fake_random_full", "pid": os.getpid(),
                                   "start_time": current_start}))
    launcher = temp / "026_fake_random_full.sh"
    launcher.write_text("#!/bin/bash\n")
    notifier = temp / "notifier.py"
    notifier.write_text("#!/usr/bin/env python3\nprint('ok')\n")
    notifier.chmod(0o700)
    helper = ROOT / "scripts/experiment_harness/notification_receipts.py"
    contract = temp / "contract.json"
    contract.write_text(json.dumps({
        "experiment_id": "storage-monitor-test", "run_id": "run-026",
        "monitoring": {
            "filesystem_path": str(temp), "hard_stop_free_bytes": 100,
            "queue_state_path": str(queue_state), "job_id": "026_fake_random_full",
            "running_metadata_path": str(running), "expected_pid": os.getpid(),
            "expected_start_time": current_start,
            "held_launcher": str(temp / "held/026_fake_random_full.sh"),
            "blocked_action": "canonical evaluation", "running_action": "Stage 2 training",
            "initial_condition": "clear", "initial_generation": 0,
        },
        "notification_receipts": {"root": str(temp / "receipts"),
                                  "operational_notifier": str(notifier),
                                  "operational_notifier_sha256": __import__("hashlib").sha256(notifier.read_bytes()).hexdigest(),
                                  "helper": str(helper),
                                  "helper_sha256": __import__("hashlib").sha256(helper.read_bytes()).hexdigest(),
                                  "base_notifier": str(notifier),
                                  "base_notifier_sha256": __import__("hashlib").sha256(notifier.read_bytes()).hexdigest()},
        "bindings": {
            "original_contract": str(launcher),
            "original_contract_sha256": __import__("hashlib").sha256(launcher.read_bytes()).hexdigest(),
            "noq_hotfix_amendment": str(launcher),
            "noq_hotfix_amendment_sha256": __import__("hashlib").sha256(launcher.read_bytes()).hexdigest(),
            "launcher": str(launcher),
            "launcher_sha256": __import__("hashlib").sha256(launcher.read_bytes()).hexdigest(),
            "watcher": str(MONITOR),
            "watcher_sha256": __import__("hashlib").sha256(MONITOR.read_bytes()).hexdigest(),
        },
    }))
    state = temp / "state/state.json"
    env = {**os.environ, "MEANAUDIO_MONITOR_TEST_MODE": "true"}

    def run(free: int) -> dict:
        completed = subprocess.run([
            sys.executable, str(MONITOR), "--contract", str(contract),
            "--launcher", str(launcher), "--state", str(state), "--free-bytes", str(free),
        ], env=env, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
        return json.loads(state.read_text())

    first = run(90)
    assert first["condition"] == "advisory" and first["generation"] == 1
    assert first["events"][-1]["status"] == "advisory"
    assert "canonical evaluation is blocked" in first["events"][-1]["summary"]
    duplicate = run(80)
    assert len(duplicate["events"]) == 1
    (temp / "held").mkdir()
    (temp / "held/026_fake_random_full.sh").write_text("#!/bin/bash\n")
    held = run(70)
    assert held["condition"] == "held" and held["events"][-1]["status"] == "held"
    cleared = run(120)
    assert cleared["condition"] == "clear" and cleared["events"][-1]["status"] == "recovered"
    assert len(cleared["events"]) == 3
    print(f"PASS storage gate monitor semantics {temp}")


if __name__ == "__main__":
    main()
