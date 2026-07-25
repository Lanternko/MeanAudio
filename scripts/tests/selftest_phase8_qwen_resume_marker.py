#!/usr/bin/env python3
"""CPU-only tamper tests for the Phase-8 resume marker validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts/validate_phase8_qwen_resume_marker.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("resume_validator", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    validator = load_validator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contract = root / "contract.md"
        contract.write_text("immutable contract\n")
        status = {
            "watcher": "phase8_qwen_bucket_quarter_backlog",
            "status": "hard_incident",
            "active_arm": "noq",
            "queue": {"first_incomplete": "noq"},
            "hard_incidents": [
                {"severity": "hard", "code": "synthetic", "detail": "x"}
            ],
            "arms": [{
                "key": "noq",
                "state": "stalled_or_transition",
                "phase": "stage1_training",
                "latest_iteration": 10,
                "latest_metrics": {"loss": 1.0},
                "grad_health": {"unhealthy": False},
                "hard_log_errors": ["synthetic"],
            }],
        }
        status_path = root / "status.json"
        write(status_path, status)
        fp = validator.incident_fingerprint(status, contract)
        commit, diff = "a" * 40, "b" * 64
        sol = {
            "decision": "approve",
            "execution_authorized": True,
            "incident_fingerprint": fp,
            "reviewed_commit": commit,
            "reviewed_diff_sha256": diff,
            "approved_command": "python repair.py",
            "rollback_command": "python rollback.py",
        }
        sol_path = root / "sol.json"
        write(sol_path, sol)
        sol_hash = hashlib.sha256(sol_path.read_bytes()).hexdigest()
        approval = {
            **sol,
            "sol_verdict_path": str(sol_path),
            "sol_verdict_sha256": sol_hash,
        }
        state_path = root / "state.json"
        write(state_path, {
            "incidents": {
                fp: {
                    "state": "awaiting_forward_progress",
                    "approval": approval,
                }
            }
        })
        marker_path = root / "marker.json"
        marker = {
            "schema_version": 1,
            "resume_authorized": True,
            "incident_fingerprint": fp,
            "reviewed_commit": commit,
            "reviewed_diff_sha256": diff,
            "sol_verdict_sha256": sol_hash,
            "expires_epoch": time.time() + 600,
            "consumed": False,
        }
        write(marker_path, marker)
        assert validator.validate(
            marker_path, state_path, status_path, contract,
        ) == fp

        for field, bad in (
            ("incident_fingerprint", "c" * 64),
            ("reviewed_commit", "d" * 40),
            ("reviewed_diff_sha256", "e" * 64),
            ("sol_verdict_sha256", "f" * 64),
            ("consumed", True),
        ):
            tampered = dict(marker)
            tampered[field] = bad
            write(marker_path, tampered)
            try:
                validator.validate(
                    marker_path, state_path, status_path, contract,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"tampered marker accepted: {field}")

    print("phase8 qwen resume marker self-test: passed")


if __name__ == "__main__":
    main()
