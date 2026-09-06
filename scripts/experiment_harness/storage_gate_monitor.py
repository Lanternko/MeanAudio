#!/usr/bin/env python3
"""Deterministic storage-gate monitor with accurate queue-state semantics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from notification_receipts import (
    atomic_secure_json,
    deliver_required,
    secure_read_json,
    sha256_file,
    utc_now,
)


PYTHON = Path("/home/kojiek/venvs/dac/bin/python")


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_id", "run_id", "monitoring", "notification_receipts"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("monitoring contract is incomplete")
    bindings = value.get("bindings") or {}
    receipts = value["notification_receipts"]
    pairs = [
        (bindings, "original_contract"), (bindings, "noq_hotfix_amendment"),
        (bindings, "launcher"), (bindings, "watcher"),
        (receipts, "operational_notifier"), (receipts, "helper"),
        (receipts, "base_notifier"),
    ]
    for section, key in pairs:
        target = Path(section[key])
        expected = section[f"{key}_sha256"]
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"monitoring binding mismatch: {key}")
    return value


def free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def process_start_time(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return fields[21]


def queue_observation(config: dict[str, Any]) -> tuple[str, bool]:
    state_path = Path(config["queue_state_path"])
    state = state_path.read_text(encoding="utf-8").strip() if state_path.is_file() else "missing"
    expected = str(config["job_id"])
    actual_held = state == f"held {expected}"
    if state == f"running {expected}":
        running = json.loads(Path(config["running_metadata_path"]).read_text(encoding="utf-8"))
        pid = int(running.get("pid", -1))
        if (running.get("job_id") != expected
                or pid != int(config["expected_pid"])
                or str(running.get("start_time")) != str(config["expected_start_time"])
                or process_start_time(pid) != str(config["expected_start_time"])):
            raise RuntimeError("live queue process identity mismatch")
    held_launcher = Path(config["held_launcher"])
    if held_launcher.is_file():
        actual_held = True
    return state, actual_held


def desired_condition(free: int, hard: int, actual_held: bool) -> str:
    if free >= hard:
        return "clear"
    return "held" if actual_held else "advisory"


def event_for(old: str, new: str, generation: int) -> tuple[str, str]:
    if new == "clear":
        return f"disk_hard_stop_cleared_{generation}", "recovered"
    if new == "held":
        return f"disk_hard_stop_held_{generation}", "held"
    return f"disk_hard_stop_advisory_{generation}", "advisory"


def summary_for(new: str, free: int, hard: int, queue_state: str,
                config: dict[str, Any]) -> str:
    common = f"free_bytes={free}, hard_stop_free_bytes={hard}, queue_state={queue_state}."
    if new == "clear":
        return common + " Storage gate cleared; this does not change the queue state or restart any process."
    if new == "held":
        return common + " Queue is verifiably held; operator action is required before resumption."
    blocked = str(config.get("blocked_action", "the next contract-gated expansion"))
    running = str(config.get("running_action", "the current process"))
    return common + f" {blocked} is blocked, but the queue is not held; {running} remains running."


def initial_state(contract: dict[str, Any]) -> dict[str, Any]:
    monitoring = contract["monitoring"]
    return {
        "document_kind": "storage_gate_monitor_state_v1",
        "experiment_id": contract["experiment_id"],
        "run_id": contract["run_id"],
        "generation": int(monitoring.get("initial_generation", 0)),
        "condition": monitoring.get("initial_condition", "clear"),
        "pending": None,
        "observed_at": utc_now(),
        "events": [],
    }


def monitor_once(contract_path: Path, launcher: Path, state_path: Path,
                 injected_free: int | None = None) -> dict[str, Any]:
    contract = load_contract(contract_path)
    config = contract["monitoring"]
    notifier = Path(contract["notification_receipts"]["operational_notifier"])
    receipt_root = Path(contract["notification_receipts"]["root"])
    hard = int(config["hard_stop_free_bytes"])
    measured = injected_free if injected_free is not None else free_bytes(Path(config["filesystem_path"]))
    queue_state, actual_held = queue_observation(config)
    expected_states = {f"running {config['job_id']}", f"held {config['job_id']}"}
    active = queue_state in expected_states
    desired = desired_condition(measured, hard, actual_held)

    state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state_path.parent, 0o700)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        state = secure_read_json(state_path) if state_path.exists() else initial_state(contract)
        if state.get("experiment_id") != contract["experiment_id"] or state.get("run_id") != contract["run_id"]:
            raise ValueError("monitor state identity mismatch")
        pending = state.get("pending")
        if pending is None and active and desired != state["condition"]:
            generation = int(state["generation"])
            if state["condition"] == "clear" and desired != "clear":
                generation += 1
            event, status = event_for(state["condition"], desired, generation)
            pending = {
                "event": event,
                "status": status,
                "summary": summary_for(desired, measured, hard, queue_state, config),
                "target_condition": desired,
                "generation": generation,
            }
            state.update({"pending": pending, "observed_at": utc_now(), "last_free_bytes": measured,
                          "last_queue_state": queue_state})
            atomic_secure_json(state_path, state)
        if pending is not None:
            key = f"{contract['experiment_id']}:{contract['run_id']}:{pending['event']}"
            receipt = deliver_required(
                contract_path=contract_path,
                launcher_path=launcher,
                event=pending["event"],
                status=pending["status"],
                summary=pending["summary"],
                idempotency_key=key,
                notifier=notifier,
                python=PYTHON,
                root=receipt_root,
            )
            state["events"].append({"at": utc_now(), **pending, "receipt": str(receipt)})
            state.update({"condition": pending["target_condition"], "generation": pending["generation"],
                          "pending": None})
        state.update({"observed_at": utc_now(), "last_free_bytes": measured,
                      "hard_stop_free_bytes": hard, "last_queue_state": queue_state,
                      "active": active})
        atomic_secure_json(state_path, state)
        return state
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--free-bytes", type=int, help="test-only injected observation")
    args = parser.parse_args()
    if args.free_bytes is not None and os.environ.get("MEANAUDIO_MONITOR_TEST_MODE") != "true":
        raise SystemExit("--free-bytes requires MEANAUDIO_MONITOR_TEST_MODE=true")
    while True:
        state = monitor_once(args.contract, args.launcher, args.state, args.free_bytes)
        print(json.dumps({key: state.get(key) for key in (
            "condition", "generation", "last_free_bytes", "last_queue_state", "observed_at", "active"
        )}, sort_keys=True), flush=True)
        if not args.watch or not state["active"]:
            return 0
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
