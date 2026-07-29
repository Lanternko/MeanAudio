#!/usr/bin/env python3
"""CPU-only dry-run/stub tests for the quarter backlog repair controller."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts/repair_phase8_qwen_bucket_incident_with_agents.sh"


def write_status(
    path: Path,
    *,
    incident: bool,
    detail: str = "stale_active_log",
    processes: list[str] | None = None,
    arm_state: str = "active",
    final_metrics_valid: bool = False,
) -> None:
    payload = {
        "schema_version": 1,
        "watcher": "phase8_qwen_bucket_quarter_backlog",
        "status": "hard_incident" if incident else "healthy",
        "active_arm": "k2_balanced" if incident else None,
        "queue": {"first_incomplete": "k2_balanced"},
        "hard_incidents": ([{"severity": "hard", "code": detail, "detail": "synthetic"}] if incident else []),
        "arms": ([{
            "key": "k2_balanced",
            "state": arm_state,
            "phase": "stage1_training",
            "latest_iteration": 1000,
            "latest_metrics": {"loss": 1.0},
            "grad_health": {"unhealthy": False},
            "active_log": None,
            "final_metrics": {"valid": final_metrics_valid},
        }] if incident else []),
        "processes": processes or [],
        "tmux": [],
        "gpu": {"status": "ok"},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def run(*args: str) -> tuple[dict, Path]:
    proc = subprocess.run(["bash", str(CONTROLLER), *args], text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines, proc.stdout
    return json.loads(lines[-1]), Path(args[args.index("--state-dir") + 1])


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        status = base / "status.json"
        state = base / "state"

        write_status(status, incident=False)
        healthy, _ = run("--once", "--status", str(status), "--state-dir", str(state), "--dry-run")
        assert healthy["llm_calls"] == 0

        write_status(status, incident=True)
        planned, _ = run("--once", "--status", str(status), "--state-dir", str(state), "--stub")
        assert planned["status"] == "approved"
        assert planned["llm_calls"] == 2
        assert "approved_command" in planned
        state_payload = json.loads((state / "state.json").read_text())
        fp = planned["fingerprint"]
        assert state_payload["incidents"][fp]["luna_calls"] == 1
        assert state_payload["incidents"][fp]["sol_calls"] == 1
        evidence = state / "incidents" / fp / "evidence.json"
        assert evidence.stat().st_size < 20 * 1024

        suppressed, _ = run("--once", "--status", str(status), "--state-dir", str(state), "--stub")
        assert suppressed["status"] in {"unchanged", "suppressed", "approved"}
        assert suppressed.get("llm_calls", 0) == 0

        # A controller restart never retries a model call that was already
        # reserved. It closes the interrupted transaction for manual review.
        interrupted_state = base / "interrupted-state"
        interrupted_state.mkdir()
        interrupted_fp = "a" * 64
        (interrupted_state / "state.json").write_text(json.dumps({
            "schema_version": 1,
            "last_observation_signature": None,
            "cooldown_until": 0.0,
            "incidents": {
                interrupted_fp: {
                    "state": "repair_in_progress",
                    "luna_calls": 1,
                    "sol_calls": 0,
                }
            },
        }) + "\n")
        write_status(status, incident=False)
        restarted, _ = run(
            "--once", "--status", str(status),
            "--state-dir", str(interrupted_state), "--dry-run",
        )
        assert restarted["llm_calls"] == 0
        interrupted_payload = json.loads(
            (interrupted_state / "state.json").read_text()
        )
        interrupted_entry = interrupted_payload["incidents"][interrupted_fp]
        assert interrupted_entry["state"] == "failed_manual"
        assert interrupted_entry["luna_calls"] == 1
        assert interrupted_entry["sol_calls"] == 0

        # The only executable stub command is harmless and requires the same
        # exact fresh revision authorization path as a real run.  Use a new
        # state directory so this is a real approval/execution path, not an
        # unchanged-observation shortcut.
        write_status(status, incident=True)
        exec_state = base / "exec-state"
        executed, _ = run("--once", "--status", str(status), "--state-dir", str(exec_state), "--stub", "--execute-approved")
        assert executed["status"] == "awaiting_forward_progress"
        assert executed["execution"]["returncode"] == 0

        # Orchestration processes are not live train/eval blockers.  The
        # controller must not mistake its supervisor, watcher, or own process
        # for a running sequence when deciding whether to execute approval.
        write_status(
            status,
            incident=True,
            detail="orchestration_only",
            processes=[
                "100 bash supervise_phase8_qwen_bucket_quarter_backlog.sh",
                "101 python monitor_phase8_qwen_bucket_quarter_backlog.py",
                "102 bash repair_phase8_qwen_bucket_incident_with_agents.sh",
            ],
        )
        orchestration_only, _ = run(
            "--once", "--status", str(status), "--state-dir", str(base / "exec-state-2"),
            "--stub", "--execute-approved",
        )
        assert orchestration_only["status"] == "awaiting_forward_progress"
        assert orchestration_only["execution"]["returncode"] == 0

        # A live trainer defers the already-approved repair without spending
        # another model call.  Once the trainer is absent, the same approval
        # executes and enters forward-progress validation.
        pending_state = base / "pending-state"
        write_status(
            status,
            incident=True,
            detail="pending_train",
            processes=["200 torchrun train.py exp_id=phase8_qwen_bucket"],
        )
        pending, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(pending_state), "--stub", "--execute-approved",
        )
        assert pending["status"] == "pending_approved"
        assert pending["llm_calls"] == 2
        write_status(status, incident=True, detail="pending_train")
        resumed, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(pending_state), "--stub", "--execute-approved",
        )
        assert resumed["status"] == "awaiting_forward_progress"
        assert resumed["llm_calls"] == 0

        # A failed approved command runs the approved rollback and records a
        # derived failure fingerprint instead of blindly retrying.
        failure_state = base / "failure-state"
        write_status(status, incident=True, detail="command_failure")
        failed, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(failure_state), "--stub", "--stub-fail-command",
            "--execute-approved",
        )
        assert failed["status"] == "failed"
        assert failed["execution"]["returncode"] == 7
        assert failed["execution"]["rollback"]["returncode"] == 0
        assert failed["failed_fingerprint"] != failed["fingerprint"]

        # The controller closes only after deterministic forward progress.
        progress_state = base / "progress-state"
        write_status(status, incident=True, detail="progress")
        awaiting, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(progress_state), "--stub", "--execute-approved",
        )
        assert awaiting["status"] == "awaiting_forward_progress"
        payload = json.loads(status.read_text())
        payload["status"] = "healthy"
        payload["hard_incidents"] = []
        payload["active_arm"] = "k2_balanced"
        payload["arms"][0]["latest_iteration"] = 1001
        status.write_text(json.dumps(payload) + "\n")
        progressed, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(progress_state), "--stub", "--execute-approved",
        )
        assert progressed["status"] == "repair_complete"
        assert progressed["forward_progress"]["progressed"] is True

        # A stale training iteration must never close a metrics incident while
        # the hard incident remains. Completion requires a healthy observation
        # and valid final metrics (or genuine post-repair active work).
        metrics_state = base / "metrics-state"
        write_status(
            status,
            incident=True,
            detail="missing_metrics",
            arm_state="stalled_or_transition",
        )
        metrics_awaiting, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(metrics_state), "--stub", "--execute-approved",
        )
        assert metrics_awaiting["status"] == "awaiting_forward_progress"
        payload = json.loads(status.read_text())
        payload["arms"][0]["latest_iteration"] = 1001
        status.write_text(json.dumps(payload) + "\n")
        still_blocked, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(metrics_state), "--stub", "--execute-approved",
        )
        assert still_blocked["status"] == "awaiting_forward_progress"
        assert still_blocked["forward_progress"]["progressed"] is False
        payload["status"] = "healthy"
        payload["hard_incidents"] = []
        payload["arms"][0]["state"] = "complete"
        payload["arms"][0]["final_metrics"] = {"valid": True}
        status.write_text(json.dumps(payload) + "\n")
        metrics_complete, _ = run(
            "--once", "--status", str(status), "--state-dir",
            str(metrics_state), "--stub", "--execute-approved",
        )
        assert metrics_complete["status"] == "repair_complete"
        assert (
            metrics_complete["forward_progress"]["final_metrics_became_valid"]
            is True
        )

    print("phase8 qwen bucket repair controller self-test: passed")


if __name__ == "__main__":
    main()
