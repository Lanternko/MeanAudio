#!/usr/bin/env python3
"""Validate one exact SOL-bound Phase-8 repair resume marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def relevant_arm(status: dict[str, Any]) -> dict[str, Any]:
    arms = status.get("arms")
    if not isinstance(arms, list):
        return {}
    active = status.get("active_arm")
    for arm in arms:
        if isinstance(arm, dict) and arm.get("key") == active:
            return arm
    for arm in arms:
        if (
            isinstance(arm, dict)
            and arm.get("state") in {"active", "stalled_or_transition"}
        ):
            return arm
    return {}


def incident_fingerprint(status: dict[str, Any], contract: Path) -> str:
    if status.get("status") != "hard_incident" or not status.get(
        "hard_incidents"
    ):
        raise ValueError("current watcher status is not a hard incident")
    arm = relevant_arm(status)
    signal = {
        "watcher": status.get("watcher"),
        "status": status.get("status"),
        "active_arm": status.get("active_arm"),
        "first_incomplete": (status.get("queue") or {}).get(
            "first_incomplete"
        ),
        "hard_incidents": sorted(
            (
                item
                for item in status.get("hard_incidents", [])
                if isinstance(item, dict)
            ),
            key=lambda item: (
                str(item.get("code")), str(item.get("detail")),
            ),
        ),
        "arm": {
            key: arm.get(key)
            for key in (
                "key",
                "state",
                "phase",
                "latest_iteration",
                "latest_metrics",
                "grad_health",
                "hard_log_errors",
            )
            if key in arm
        },
        "contract_sha256": sha256(contract),
    }
    return hashlib.sha256(canonical(signal)).hexdigest()


def validate(
    marker_path: Path,
    state_path: Path,
    status_path: Path,
    contract_path: Path,
) -> str:
    marker = read_json(marker_path)
    state = read_json(state_path)
    status = read_json(status_path)
    fingerprint = marker.get("incident_fingerprint")
    if not (
        marker.get("schema_version") == 1
        and marker.get("resume_authorized") is True
        and marker.get("consumed") is False
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and float(marker.get("expires_epoch", 0)) > time.time()
    ):
        raise ValueError("marker structure, consumption, or expiry is invalid")
    if incident_fingerprint(status, contract_path) != fingerprint:
        raise ValueError("marker does not match the current incident")

    incidents = state.get("incidents")
    entry = incidents.get(fingerprint) if isinstance(incidents, dict) else None
    if not isinstance(entry, dict) or entry.get(
        "state"
    ) != "awaiting_forward_progress":
        raise ValueError("controller state is not awaiting forward progress")
    approval = entry.get("approval")
    if not isinstance(approval, dict):
        raise ValueError("controller approval binding is missing")
    for key in (
        "incident_fingerprint",
        "reviewed_commit",
        "reviewed_diff_sha256",
        "sol_verdict_sha256",
    ):
        if marker.get(key) != approval.get(key):
            raise ValueError(f"marker/controller mismatch: {key}")

    sol_path = Path(str(approval.get("sol_verdict_path", "")))
    if not sol_path.is_file():
        raise ValueError("bound SOL verdict is missing")
    if sha256(sol_path) != marker.get("sol_verdict_sha256"):
        raise ValueError("SOL verdict digest mismatch")
    sol = read_json(sol_path)
    if not (
        sol.get("decision") == "approve"
        and sol.get("execution_authorized") is True
        and sol.get("incident_fingerprint") == fingerprint
        and sol.get("reviewed_commit") == marker.get("reviewed_commit")
        and sol.get("reviewed_diff_sha256")
        == marker.get("reviewed_diff_sha256")
        and sol.get("approved_command") == approval.get("approved_command")
        and sol.get("rollback_command") == approval.get("rollback_command")
    ):
        raise ValueError("SOL verdict content does not match approval")
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    try:
        fingerprint = validate(
            args.marker, args.state, args.status, args.contract,
        )
    except Exception as exc:
        print(f"resume marker rejected: {exc}")
        return 1
    print(f"resume marker valid: {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
