#!/usr/bin/env python3
"""No-GPU tests for resource-controller ordering and predecessor receipts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/kojiek/gpu_queue")
sys.path.insert(0, str(ROOT / "scripts/experiment_harness"))
import secondary_queue_controller as controller_module  # noqa: E402
from notification_receipts import deliver_required, sha256_file  # noqa: E402
from secondary_cfg_sweep_queue_guest import require_028_terminal  # noqa: E402
from secondary_queue_controller import Controller  # noqa: E402


def main() -> None:
    temp = Path(tempfile.mkdtemp(prefix="secondary-controller-test-"))
    receipt_root = temp / "receipts"
    notifier = temp / "notifier.py"
    notifier.write_text("#!/usr/bin/env python3\nprint('ok')\n")
    notifier.chmod(0o700)
    launcher = temp / "029_test.sh"
    launcher.write_text("#!/bin/bash\n")
    launcher.chmod(0o700)
    contract = temp / "029.contract.json"
    contract.write_text(json.dumps({
        "experiment_id": "controller-test", "run_id": "controller-run",
        "notification_receipts": {
            "required": True, "root": str(receipt_root),
            "notifier": str(notifier), "notifier_sha256": sha256_file(notifier),
            "helper": str(ROOT / "scripts/experiment_harness/notification_receipts.py"),
            "helper_sha256": sha256_file(ROOT / "scripts/experiment_harness/notification_receipts.py"),
        },
    }))
    old_notifier, old_python = controller_module.NOTIFIER, controller_module.PYTHON
    controller_module.NOTIFIER, controller_module.PYTHON = notifier, Path(sys.executable)
    try:
        controller = Controller(launcher, contract, temp / "state")
        controller.record("mutable_preflight_passed")
        controller.pre_child_notifications("test")
        start_receipt = json.loads((receipt_root / "controller-test" / "start.json").read_text())
        handoff_receipt = json.loads((receipt_root / "controller-test" / "queue_handoff.json").read_text())
        assert start_receipt["status"] == "start", start_receipt
        assert handoff_receipt["status"] == "start", handoff_receipt
        controller.terminal("held", "held", "held", "test held")
    finally:
        controller_module.NOTIFIER, controller_module.PYTHON = old_notifier, old_python
    ledger = json.loads((temp / "state/controller_ledger.json").read_text())
    events = [entry["event"] for entry in ledger["events"]]
    expected = [
        "resource_owner_recorded", "mutable_preflight_passed",
        "notification_queue_handoff_delivered", "notification_start_delivered",
        "evaluation_child_launch_authorized", "notification_held_delivered",
        "terminal_notification_delivered", "terminal_json_committed",
    ]
    positions = [events.index(event) for event in expected]
    assert positions == sorted(positions), events
    terminal = json.loads((temp / "029_test.terminal.json").read_text())
    assert terminal["status"] == "held"
    assert terminal["notification_receipt"]["event"] == "held"

    # A predecessor file alone is insufficient: authentic terminal receipt is mandatory.
    qroot = temp / "queue"
    done = qroot / "p2/done"
    done.mkdir(parents=True)
    predecessor = done / "028_random_quarter_neg_cfg1p5.sh"
    predecessor.write_text("#!/bin/bash\n")
    predecessor.chmod(0o700)
    (done / "028_random_quarter_neg_cfg1p5.terminal.json").write_text(json.dumps({"status": "completed"}))
    try:
        require_028_terminal(qroot, contract)
    except ValueError:
        pass
    else:
        raise AssertionError("receipt-less predecessor accepted")

    predecessor_contract = temp / "028.contract.json"
    predecessor_contract.write_text(json.dumps({
        "experiment_id": "predecessor-test", "run_id": "predecessor-run",
        "notification_receipts": {
            "required": True, "root": str(receipt_root),
            "notifier": str(notifier), "notifier_sha256": sha256_file(notifier),
            "helper": str(ROOT / "scripts/experiment_harness/notification_receipts.py"),
            "helper_sha256": sha256_file(ROOT / "scripts/experiment_harness/notification_receipts.py"),
        },
    }))
    receipt = deliver_required(
        contract_path=predecessor_contract, launcher_path=predecessor,
        event="success", status="success", summary="done",
        idempotency_key="predecessor-test:success", notifier=notifier,
        python=Path(sys.executable), root=receipt_root,
    )
    (done / "028_random_quarter_neg_cfg1p5.terminal.json").write_text(json.dumps({
        "status": "completed",
        "notification_receipt": {"path": str(receipt), "event": "success", "status": "success"},
    }))
    evidence = require_028_terminal(qroot, predecessor_contract)
    assert evidence["status"] == "completed"
    failed = qroot / "p2/failed"
    failed.mkdir(parents=True)
    (failed / predecessor.name).write_bytes(predecessor.read_bytes())
    try:
        require_028_terminal(qroot, predecessor_contract)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate terminal directories accepted")
    print(f"PASS secondary queue notifications {temp}")


if __name__ == "__main__":
    main()
