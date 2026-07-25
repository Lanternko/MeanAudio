#!/usr/bin/env bash
set -euo pipefail

# Event-driven repair controller.  The implementation is intentionally kept in
# this file so the controller can be installed as one atomic, reviewable file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Preserve this script path as argv[0] so the durable supervisor can identify
# the exec'd Python controller without mistaking it for a dead child.
exec -a "$0" python3 - "$ROOT_DIR" "$@" <<'PY'
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(sys.argv[1]).resolve()
DEFAULT_STATUS = Path("/home/kojiek/logs/phase8_qwen_bucket_quarter_backlog_monitor/status.json")
DEFAULT_ALERT = Path("/home/kojiek/logs/phase8_qwen_bucket_quarter_backlog_monitor/ALERT.json")
DEFAULT_STATE = Path("/home/kojiek/logs/phase8_qwen_bucket_quarter_backlog_repair")
DEFAULT_RESUME_MARKER = DEFAULT_STATE / "RESUME_AUTHORIZED.json"
EVIDENCE_LIMIT = 19 * 1024
DEFAULT_FRESH_SECONDS = 600
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_COMMAND_OUTPUT_BYTES = 8192
DEFAULT_FORWARD_PROGRESS_TIMEOUT_SECONDS = 900


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, max_bytes: int = 256 * 1024) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def text_tail(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def load_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if not state or state.get("schema_version") != 1:
        return {"schema_version": 1, "last_observation_signature": None, "incidents": {}, "cooldown_until": 0.0}
    state.setdefault("incidents", {})
    state.setdefault("cooldown_until", 0.0)
    return state


def audit(state_dir: Path, state: dict[str, Any], event: str, **fields: Any) -> None:
    record = {"at": utc_now(), "event": event, **fields}
    state["audit_tail"] = (state.get("audit_tail", []) + [record])[-32:]
    state["audit_seq"] = int(state.get("audit_seq", 0)) + 1
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"seq": state["audit_seq"], **record}, sort_keys=True)[:12000] + "\n")


def observation_signature(path: Path) -> str | None:
    return file_sha256(path)


def relevant_arm(status: dict[str, Any]) -> dict[str, Any] | None:
    arms = status.get("arms")
    if not isinstance(arms, list):
        return None
    active = status.get("active_arm")
    for arm in arms:
        if isinstance(arm, dict) and arm.get("key") == active:
            return arm
    for arm in arms:
        if isinstance(arm, dict) and arm.get("state") in {"active", "stalled_or_transition"}:
            return arm
    return None


def incident_fingerprint(status: dict[str, Any], contract: Path | None) -> str | None:
    if status.get("status") != "hard_incident" or not status.get("hard_incidents"):
        return None
    arm = relevant_arm(status) or {}
    signal = {
        "watcher": status.get("watcher"),
        "status": status.get("status"),
        "active_arm": status.get("active_arm"),
        "first_incomplete": (status.get("queue") or {}).get("first_incomplete"),
        "hard_incidents": sorted(
            (x for x in status.get("hard_incidents", []) if isinstance(x, dict)),
            key=lambda x: (str(x.get("code")), str(x.get("detail"))),
        ),
        "arm": {
            key: arm.get(key)
            for key in ("key", "state", "phase", "latest_iteration", "latest_metrics", "grad_health", "hard_log_errors")
            if key in arm
        },
        "contract_sha256": file_sha256(contract) if contract else None,
    }
    return digest(signal)


def bounded_evidence(status: dict[str, Any], status_path: Path, contract: Path | None) -> dict[str, Any]:
    arm = relevant_arm(status) or {}
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "status_path": str(status_path),
        "status_sha256": file_sha256(status_path),
        "contract_path": str(contract) if contract else None,
        "contract_sha256": file_sha256(contract) if contract else None,
        "status": status.get("status"),
        "watcher": status.get("watcher"),
        "active_arm": status.get("active_arm"),
        "queue": status.get("queue"),
        "handoff": status.get("handoff"),
        "hard_incidents": [
            {"severity": x.get("severity"), "code": x.get("code"), "detail": str(x.get("detail", ""))[:700]}
            for x in status.get("hard_incidents", [])[:8]
            if isinstance(x, dict)
        ],
        "arm": {
            key: arm.get(key)
            for key in ("key", "state", "phase", "active_log", "latest_iteration", "latest_metrics", "grad_health", "hard_log_errors", "checkpoint")
            if key in arm
        },
        "gpu": status.get("gpu"),
        "processes": [str(x)[:500] for x in status.get("processes", [])[-20:]],
        "tmux": [str(x)[:200] for x in status.get("tmux", [])[-20:]],
    }
    log_path = arm.get("active_log")
    if isinstance(log_path, str):
        lines = text_tail(Path(log_path)).splitlines()
        evidence["relevant_log_lines"] = [line[-500:] for line in lines[-100:]]
    while len(canonical(evidence)) >= EVIDENCE_LIMIT:
        if evidence.get("relevant_log_lines"):
            evidence["relevant_log_lines"] = evidence["relevant_log_lines"][-40:]
        elif evidence.get("processes"):
            evidence["processes"] = evidence["processes"][-8:]
        elif evidence.get("tmux"):
            evidence["tmux"] = evidence["tmux"][-8:]
        elif evidence.get("arm", {}).get("hard_log_errors"):
            evidence["arm"]["hard_log_errors"] = evidence["arm"]["hard_log_errors"][-4:]
        else:
            evidence["bounded_evidence_note"] = "fields reduced to stay below 19 KiB"
            break
    if len(canonical(evidence)) >= EVIDENCE_LIMIT:
        # This fallback remains valid JSON and is intentionally conservative.
        evidence = {
            "schema_version": 1,
            "captured_at": evidence["captured_at"],
            "status": evidence.get("status"),
            "hard_incidents": evidence.get("hard_incidents", [])[:4],
            "bounded_evidence_note": "full evidence exceeded local bound",
        }
    return evidence


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    rendered = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    if len(rendered) >= 20 * 1024:
        raise RuntimeError("evidence must be smaller than 20 KiB")
    atomic_json(path, evidence)


def ensure_isolated_worktree(root: Path, worktree: Path, branch: str, base: str) -> None:
    root_real = root.resolve()
    worktree_parent = worktree.parent.resolve()
    worktree_real = worktree.resolve(strict=False)
    if worktree_real == root_real or root_real in worktree_real.parents:
        raise RuntimeError("repair worktree must be outside the live checkout")
    if worktree_parent == root_real or root_real in worktree_parent.parents:
        raise RuntimeError("repair worktree parent must be outside the live checkout")
    if worktree.exists():
        raise RuntimeError(f"refusing to reuse existing repair worktree: {worktree}")
    result = run(["git", "-C", str(root), "worktree", "add", "-b", branch, str(worktree), base], timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or "git worktree add failed")


def git_revision(worktree: Path, args: list[str]) -> str:
    result = run(["git", "-C", str(worktree), *args])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or "git command failed")
    return result.stdout.strip()


def validate_luna_report(report: dict[str, Any], fp: str, worktree: Path, branch: str, base: str) -> dict[str, Any]:
    if report.get("incident_fingerprint") != fp:
        raise RuntimeError("Luna report fingerprint mismatch")
    if Path(str(report.get("worktree", ""))).resolve() != worktree.resolve():
        raise RuntimeError("Luna report worktree mismatch")
    if report.get("branch") != branch:
        raise RuntimeError("Luna report branch mismatch")
    head = git_revision(worktree, ["rev-parse", "HEAD"])
    if report.get("repair_commit") != head or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("Luna report does not name the current exact commit")
    if git_revision(worktree, ["status", "--porcelain"]):
        raise RuntimeError("Luna worktree is dirty; exact commit cannot be reviewed")
    changed = [x for x in git_revision(worktree, ["diff", "--name-only", f"{base}..{head}"]).splitlines() if x]
    if not changed:
        raise RuntimeError("Luna produced no committed change")
    revision_range = f"{base}..{head}"
    check = run(["git", "-C", str(worktree), "diff", "--check", revision_range])
    if check.returncode != 0:
        raise RuntimeError("Luna diff --check failed")
    diff_hash = hashlib.sha256(run(["git", "-C", str(worktree), "diff", revision_range], timeout=60).stdout.encode()).hexdigest()
    reported_hash = report.get("diff_sha256")
    if reported_hash not in (None, diff_hash):
        raise RuntimeError("Luna report diff digest mismatch")
    report["validated_head"] = head
    report["base_commit"] = base
    report["changed_files"] = changed
    report["diff_sha256"] = diff_hash
    return report


def safe_command(command: str, *, stub: bool = False) -> None:
    if not isinstance(command, str) or not command.strip() or "\n" in command or "\r" in command:
        raise RuntimeError("approved command is empty or multiline")
    if re.search(r"(?:;|&&|\|\||[|><`]|\$\(|sudo|systemctl|modprobe|rmmod|reboot|shutdown|pkill|kill\b|renice)", command, re.I):
        raise RuntimeError("approved command contains a forbidden operator or host action")
    if re.search(
        r"(?:torchrun|deepspeed|accelerate\s+launch|(?:^|[ /])(?:train|eval|phase4_eval)\.py(?:$|\s)|"
        r"sequence_phase8_qwen_bucket)",
        command,
        re.I,
    ):
        raise RuntimeError("approved command may not launch training or evaluation")
    tokens = shlex.split(command)
    if not tokens:
        raise RuntimeError("approved command has no executable")
    executable = next((token for token in tokens if "=" not in token or token.startswith("/")), tokens[0])
    allowed = {"bash", "python", "python3", "printf", "true"}
    if stub:
        if executable not in allowed:
            raise RuntimeError("stub command is outside the test allowlist")
    elif executable not in {"bash", "python", "python3"}:
        raise RuntimeError("live execution only permits an approved shell/python command")


def run_bounded_command(command: str, *, timeout_seconds: int, output_bytes: int, cwd: Path) -> dict[str, Any]:
    """Run one approved process with a short wall-clock and bounded evidence."""
    started = time.time()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=5)
        def tail(handle: Any) -> str:
            handle.flush()
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - output_bytes))
            return handle.read().decode("utf-8", errors="replace")
        return {
            "returncode": int(returncode),
            "timed_out": timed_out,
            "duration_seconds": round(time.time() - started, 3),
            "stdout": tail(stdout_file),
            "stderr": tail(stderr_file),
        }


def checkpoint_baseline(status: dict[str, Any]) -> dict[str, Any]:
    arm = relevant_arm(status) or {}
    checkpoint = arm.get("checkpoint") if isinstance(arm.get("checkpoint"), dict) else {}
    paths = [checkpoint.get("stage1_path"), checkpoint.get("stage2_path")]
    mtimes = []
    for raw in paths:
        if isinstance(raw, str):
            try:
                mtimes.append(Path(raw).stat().st_mtime)
            except OSError:
                pass
    return {
        "captured_at": time.time(),
        "arm": arm.get("key"),
        "active_arm": status.get("active_arm"),
        "latest_iteration": arm.get("latest_iteration"),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "checkpoint_paths": paths,
        "checkpoint_mtime": max(mtimes, default=0.0),
    }


def forward_progress(status: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    arm = relevant_arm(status) or {}
    active = (
        status.get("active_arm") == baseline.get("arm")
        and arm.get("key") == baseline.get("arm")
        and arm.get("state") in {"active", "stalled_or_transition"}
    )
    iteration = arm.get("latest_iteration")
    checkpoint = arm.get("checkpoint") if isinstance(arm.get("checkpoint"), dict) else {}
    try:
        iteration_increased = (
            iteration is not None
            and (
                (
                    baseline.get("latest_iteration") is None
                    and int(iteration) > 0
                )
                or (
                    baseline.get("latest_iteration") is not None
                    and int(iteration) > int(baseline["latest_iteration"])
                )
            )
        )
    except (TypeError, ValueError):
        iteration_increased = False
    mtimes = []
    for raw in checkpoint.get("stage1_path"), checkpoint.get("stage2_path"):
        if isinstance(raw, str):
            try:
                mtimes.append(Path(raw).stat().st_mtime)
            except OSError:
                pass
    checkpoint_updated = max(mtimes, default=0.0) > float(baseline.get("checkpoint_mtime", 0.0))
    progressed = bool(
        arm.get("state") == "complete"
        or (active and (iteration_increased or checkpoint_updated))
    )
    return {
        "active": active,
        "iteration": iteration,
        "iteration_increased": iteration_increased,
        "checkpoint_updated": checkpoint_updated,
        "progressed": progressed,
    }


def derived_failure_fingerprint(fp: str, kind: str, detail: dict[str, Any]) -> str:
    return digest({"source_fingerprint": fp, "kind": kind, "detail": detail})


def archive_current_alert(args: argparse.Namespace, fp: str, incident_dir: Path) -> dict[str, Any]:
    alert_path = args.alert
    if not alert_path.is_file():
        return {"archived": False, "reason": "no_current_alert"}
    alert = read_json(alert_path)
    if alert is None or incident_fingerprint(alert, args.contract) != fp:
        return {"archived": False, "reason": "alert_fingerprint_mismatch"}
    archive = incident_dir / "watcher_ALERT.json"
    try:
        # The rename is atomic and only happens after parsing/binding the
        # current alert to this exact incident fingerprint.
        os.replace(alert_path, archive)
    except OSError as exc:
        return {"archived": False, "reason": f"archive_failed:{exc}"}
    return {"archived": True, "path": str(archive)}


def write_resume_marker(
    args: argparse.Namespace,
    fp: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    approval = entry.get("approval") or {}
    payload = {
        "schema_version": 1,
        "resume_authorized": True,
        "incident_fingerprint": fp,
        "reviewed_commit": approval.get("reviewed_commit"),
        "reviewed_diff_sha256": approval.get("reviewed_diff_sha256"),
        "sol_verdict_sha256": approval.get("sol_verdict_sha256"),
        "created_at": utc_now(),
        "expires_epoch": time.time() + args.forward_progress_timeout_seconds,
        "consumed": False,
    }
    atomic_json(args.resume_marker, payload)
    return {"created": True, "path": str(args.resume_marker), **payload}


def revoke_resume_marker(
    args: argparse.Namespace,
    incident_dir: Path,
    reason: str,
) -> dict[str, Any]:
    if not args.resume_marker.is_file():
        return {"revoked": False, "reason": "marker_absent"}
    destination = incident_dir / f"RESUME_AUTHORIZED.{reason}.json"
    try:
        os.replace(args.resume_marker, destination)
    except OSError as exc:
        return {"revoked": False, "reason": f"revoke_failed:{exc}"}
    return {"revoked": True, "path": str(destination), "reason": reason}


def restore_archived_alert(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    archive_info = entry.get("alert_archive")
    if not isinstance(archive_info, dict) or not archive_info.get("archived"):
        return {"restored": False, "reason": "no_archived_alert"}
    archive = Path(str(archive_info.get("path", "")))
    if args.alert.exists():
        return {"restored": False, "reason": "current_alert_already_exists"}
    if not archive.is_file():
        return {"restored": False, "reason": "archive_missing"}
    try:
        os.replace(archive, args.alert)
    except OSError as exc:
        return {"restored": False, "reason": f"restore_failed:{exc}"}
    return {"restored": True, "path": str(args.alert)}


def bind_approval(sol: dict[str, Any], sol_path: Path, fp: str, commit: str, diff_hash: str) -> dict[str, Any]:
    return {
        "incident_fingerprint": fp,
        "reviewed_commit": commit,
        "reviewed_diff_sha256": diff_hash,
        "approved_command": sol["approved_command"],
        "rollback_command": sol.get("rollback_command"),
        "sol_verdict_path": str(sol_path),
        "sol_verdict_sha256": file_sha256(sol_path),
        "fresh_at_approval": True,
        "approved_at": utc_now(),
    }


def validate_bound_approval(entry: dict[str, Any], fp: str, *, stub: bool) -> tuple[str, str]:
    approval = entry.get("approval")
    if not isinstance(approval, dict):
        raise RuntimeError("missing persisted SOL approval")
    if (
        approval.get("incident_fingerprint") != fp
        or not isinstance(approval.get("reviewed_commit"), str)
        or not isinstance(approval.get("reviewed_diff_sha256"), str)
        or approval.get("fresh_at_approval") is not True
    ):
        raise RuntimeError("persisted approval binding mismatch")
    sol_path = Path(str(approval.get("sol_verdict_path", "")))
    sol = read_json(sol_path)
    if sol is None or file_sha256(sol_path) != approval.get("sol_verdict_sha256"):
        raise RuntimeError("SOL verdict changed after approval")
    if (
        sol.get("decision") != "approve"
        or sol.get("execution_authorized") is not True
        or sol.get("incident_fingerprint") != fp
        or sol.get("reviewed_commit") != approval.get("reviewed_commit")
        or sol.get("reviewed_diff_sha256") != approval.get("reviewed_diff_sha256")
        or sol.get("approved_command") != approval.get("approved_command")
    ):
        raise RuntimeError("SOL exact commit/diff/fingerprint binding mismatch")
    command = approval.get("approved_command")
    rollback = approval.get("rollback_command")
    safe_command(command, stub=stub)
    safe_command(rollback, stub=stub)
    return command, rollback


def mark_failed(state: dict[str, Any], args: argparse.Namespace, fp: str, entry: dict[str, Any], kind: str, detail: dict[str, Any]) -> str:
    failed_fp = derived_failure_fingerprint(fp, kind, detail)
    entry.update({"state": "failed", "failed_fingerprint": failed_fp, "failure_kind": kind, "failure": detail})
    state.setdefault("failed_fingerprints", {})[failed_fp] = {
        "source_fingerprint": fp,
        "kind": kind,
        "created_at": utc_now(),
        "detail": detail,
    }
    state["cooldown_until"] = 0.0
    audit(args.state_dir, state, "incident_failed", fingerprint=fp, failed_fingerprint=failed_fp, kind=kind, detail=detail, llm_calls=0)
    return failed_fp


def execute_approval(args: argparse.Namespace, state: dict[str, Any], fp: str, entry: dict[str, Any], status: dict[str, Any], incident_dir: Path) -> dict[str, Any]:
    command, rollback = validate_bound_approval(entry, fp, stub=args.stub)
    result = run_bounded_command(
        command,
        timeout_seconds=args.command_timeout_seconds,
        output_bytes=args.command_output_bytes,
        cwd=args.root,
    )
    result["approved_command"] = command
    if result["returncode"] != 0 or result["timed_out"]:
        rollback_result = run_bounded_command(
            rollback,
            timeout_seconds=args.command_timeout_seconds,
            output_bytes=args.command_output_bytes,
            cwd=args.root,
        )
        result["rollback"] = rollback_result
        failed_fp = mark_failed(
            state,
            args,
            fp,
            entry,
            "approved_command_failed",
            {"command": result, "rollback": rollback_result},
        )
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "failed", "fingerprint": fp, "failed_fingerprint": failed_fp, "execution": result, "llm_calls": 0}
    resume_marker = write_resume_marker(args, fp, entry)
    baseline = checkpoint_baseline(status)
    deadline = time.time() + args.forward_progress_timeout_seconds
    entry.update({
        "state": "awaiting_forward_progress",
        "baseline": baseline,
        "forward_progress_deadline": deadline,
        "resume_marker": resume_marker,
    })
    audit(args.state_dir, state, "approved_command_succeeded", fingerprint=fp, execution=result, resume_marker=resume_marker, baseline=baseline, deadline=deadline, llm_calls=0)
    atomic_json(args.state_dir / "state.json", state)
    return {"status": "awaiting_forward_progress", "fingerprint": fp, "execution": result, "resume_marker": resume_marker, "llm_calls": 0}


def fake_luna(fp: str, worktree: Path, branch: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "incident_fingerprint": fp,
        "decision": "repair",
        "summary": "stub isolated minimal repair",
        "worktree": str(worktree),
        "branch": branch,
        "repair_commit": "2" * 40,
        "base_commit": "1" * 40,
        "diff_sha256": "3" * 64,
        "changed_files": ["stub/repair.py"],
        "tests": [{"command": "stub-test", "status": "passed"}],
        "contract_preserved": True,
        "proposed_command": "printf stub-approved-command",
        "rollback_command": "printf stub-rollback",
        "execution_authorized": False,
    }


def fake_sol(
    fp: str, commit: str, diff_hash: str, *, fail_command: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision": "approve",
        "execution_authorized": True,
        "incident_fingerprint": fp,
        "reviewed_commit": commit,
        "reviewed_diff_sha256": diff_hash,
        "summary": "stub exact revision approval",
        "findings": [],
        "approved_command": (
            "python -c 'raise SystemExit(7)'"
            if fail_command else "printf stub-approved-command"
        ),
        "rollback_command": "printf stub-rollback",
        "issued_at": utc_now(),
    }


def invoke_model(model: str, prompt: Path, schema: Path, worktree: Path, output: Path, transcript: Path, codex_bin: str, timeout_seconds: int) -> None:
    command = [
        "timeout", f"{timeout_seconds}s", codex_bin, "exec", "--ephemeral", "--model", model,
        "-c", 'approval_policy="never"', "--sandbox", "workspace-write" if model.endswith("luna") else "read-only",
        "--cd", str(worktree), "--output-schema", str(schema), "--output-last-message", str(output), "--json", "-",
    ]
    with prompt.open("rb") as prompt_handle, transcript.open("wb") as transcript_handle:
        result = subprocess.run(command, stdin=prompt_handle, stdout=transcript_handle, stderr=subprocess.STDOUT, timeout=timeout_seconds + 15, check=False)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"{model} invocation failed with exit={result.returncode}")


def is_live_train_or_eval_process(line: Any) -> bool:
    """Return true only for an actual training/evaluation command.

    The status process list can include the supervisor, watcher, and this
    controller.  Those are orchestration processes and must not block an
    approved repair command; only their train/eval children are blockers.
    """
    text = str(line or "")
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "supervise_phase8_qwen_bucket_quarter_backlog.sh",
            "monitor_phase8_qwen_bucket_quarter_backlog.py",
            "repair_phase8_qwen_bucket_incident_with_agents.sh",
        )
    ):
        return False
    return bool(
        re.search(r"(?:^|\s)(?:torchrun|accelerate|deepspeed)(?:\s|$)", lowered)
        or re.search(r"(?:^|\s)(?:python(?:3)?\s+)?(?:train|eval|phase4_eval)\.py(?:\s|$)", lowered)
    )


def live_train_or_eval_processes(status: dict[str, Any]) -> list[str]:
    lines: list[Any] = list(status.get("processes") or [])
    sequence = status.get("sequence")
    if isinstance(sequence, dict):
        lines.extend(sequence.get("processes") or [])
    for arm in status.get("arms") or []:
        if isinstance(arm, dict):
            lines.extend(arm.get("processes") or [])
    return [
        str(line) for line in lines if is_live_train_or_eval_process(line)
    ]


def process_once(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    status_path = args.status.resolve()
    signature = observation_signature(status_path)
    if signature is None:
        audit(args.state_dir, state, "status_unavailable", status_path=str(status_path), llm_calls=0)
        state["last_observation_signature"] = None
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "status_unavailable", "llm_calls": 0}
    status = read_json(status_path)
    state["last_observation_signature"] = signature
    if status is None:
        audit(args.state_dir, state, "invalid_status", status_path=str(status_path), llm_calls=0)
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "invalid_status", "llm_calls": 0}

    # Pending approvals and post-resume validation are local state-machine
    # transitions.  They must continue even when status.json is unchanged and
    # must never trigger a second Luna/SOL call.
    for pending_fp, pending_entry in list(state.get("incidents", {}).items()):
        pending_state = pending_entry.get("state")
        incident_dir = args.state_dir / "incidents" / pending_fp
        if pending_state == "awaiting_forward_progress":
            progress = forward_progress(
                status, pending_entry.get("baseline") or {},
            )
            if progress["progressed"]:
                marker = revoke_resume_marker(
                    args, incident_dir, "progress_confirmed",
                )
                pending_entry.update({
                    "state": "repair_complete",
                    "forward_progress": progress,
                    "resume_marker_final": marker,
                    "completed_at": utc_now(),
                })
                audit(
                    args.state_dir, state, "forward_progress_confirmed",
                    fingerprint=pending_fp, progress=progress, llm_calls=0,
                )
                atomic_json(args.state_dir / "state.json", state)
                return {
                    "status": "repair_complete",
                    "fingerprint": pending_fp,
                    "forward_progress": progress,
                    "llm_calls": 0,
                }
            if time.time() > float(
                pending_entry.get("forward_progress_deadline", 0.0)
            ):
                live = live_train_or_eval_processes(status)
                if live:
                    audit(
                        args.state_dir, state,
                        "rollback_waiting_for_train_exit",
                        fingerprint=pending_fp,
                        live_train_or_eval=live[:8],
                        llm_calls=0,
                    )
                    atomic_json(args.state_dir / "state.json", state)
                    return {
                        "status": "rollback_pending",
                        "fingerprint": pending_fp,
                        "live_train_or_eval": live[:8],
                        "llm_calls": 0,
                    }
                _, rollback = validate_bound_approval(
                    pending_entry, pending_fp, stub=args.stub,
                )
                rollback_result = run_bounded_command(
                    rollback,
                    timeout_seconds=args.command_timeout_seconds,
                    output_bytes=args.command_output_bytes,
                    cwd=args.root,
                )
                marker = revoke_resume_marker(
                    args, incident_dir, "progress_timeout",
                )
                failed_fp = mark_failed(
                    state, args, pending_fp, pending_entry,
                    "forward_progress_timeout",
                    {
                        "progress": progress,
                        "rollback": rollback_result,
                        "resume_marker": marker,
                    },
                )
                atomic_json(args.state_dir / "state.json", state)
                return {
                    "status": "failed",
                    "fingerprint": pending_fp,
                    "failed_fingerprint": failed_fp,
                    "rollback": rollback_result,
                    "resume_marker": marker,
                    "llm_calls": 0,
                }
            return {
                "status": "awaiting_forward_progress",
                "fingerprint": pending_fp,
                "forward_progress": progress,
                "llm_calls": 0,
            }
        if pending_state == "pending_approved":
            current_fp = incident_fingerprint(status, args.contract)
            if current_fp != pending_fp:
                return {
                    "status": "pending_incident_changed",
                    "fingerprint": pending_fp,
                    "current_fingerprint": current_fp,
                    "llm_calls": 0,
                }
            live = live_train_or_eval_processes(status)
            if live:
                return {
                    "status": "pending_approved",
                    "fingerprint": pending_fp,
                    "live_train_or_eval": live[:8],
                    "llm_calls": 0,
                }
            if not args.execute_approved:
                approval = pending_entry.get("approval") or {}
                return {
                    "status": "approved",
                    "fingerprint": pending_fp,
                    "approved_command": approval.get("approved_command"),
                    "llm_calls": 0,
                }
            return execute_approval(
                args, state, pending_fp, pending_entry, status, incident_dir,
            )

    if signature == state.get("last_processed_observation_signature"):
        return {"status": "unchanged", "llm_calls": 0}
    state["last_processed_observation_signature"] = signature

    fp = incident_fingerprint(status, args.contract)
    evidence = bounded_evidence(status, status_path, args.contract)
    if fp is None:
        audit(args.state_dir, state, "healthy_or_nonincident", status=status.get("status"), llm_calls=0)
        atomic_json(args.state_dir / "state.json", state)
        return {"status": status.get("status"), "llm_calls": 0}

    # A failed repair creates a materially new incident identity even if the
    # deterministic watcher still reports the original root-cause fingerprint.
    source_entry = state.get("incidents", {}).get(fp)
    if (
        isinstance(source_entry, dict)
        and source_entry.get("state") == "failed"
        and isinstance(source_entry.get("failed_fingerprint"), str)
    ):
        evidence["source_incident_fingerprint"] = fp
        evidence["repair_failure"] = source_entry.get("failure")
        fp = source_entry["failed_fingerprint"]

    incident_dir = args.state_dir / "incidents" / fp
    incident_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = incident_dir / "evidence.json"
    write_evidence(evidence_path, evidence)
    entry = state["incidents"].setdefault(fp, {"luna_calls": 0, "sol_calls": 0, "state": "new"})
    now = time.time()
    if entry.get("luna_calls", 0) >= 1:
        audit(args.state_dir, state, "incident_suppressed", fingerprint=fp, reason="fingerprint_already_called", llm_calls=0)
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "suppressed", "fingerprint": fp, "llm_calls": 0}
    if now < float(state.get("cooldown_until", 0.0)):
        audit(args.state_dir, state, "incident_suppressed", fingerprint=fp, reason="cooldown", llm_calls=0)
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "cooldown", "fingerprint": fp, "llm_calls": 0}
    if args.dry_run:
        entry.update({"state": "dry_run", "evidence": str(evidence_path)})
        state["cooldown_until"] = now + args.cooldown_seconds
        audit(args.state_dir, state, "dry_run_incident", fingerprint=fp, evidence=str(evidence_path), planned_luna_calls=1, planned_sol_calls=1)
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "dry_run", "fingerprint": fp, "llm_calls": 0, "planned_llm_calls": 2}

    slug = fp[:12]
    worktree = (args.worktree_root / f"phase8-qwen-bucket-repair-{slug}").resolve()
    branch = f"codex/phase8-qwen-bucket-repair/{slug}"
    if args.stub:
        luna = fake_luna(fp, worktree, branch)
        entry["luna_calls"] = 1
        entry["state"] = "luna_stubbed"
        atomic_json(incident_dir / "luna_report.json", luna)
        commit = luna["repair_commit"]
        diff_hash = luna["diff_sha256"]
    else:
        base = git_revision(args.root, ["rev-parse", "HEAD"])
        ensure_isolated_worktree(args.root, worktree, branch, base)
        prompt = incident_dir / "luna_prompt.md"
        prompt.write_text(args.luna_prompt.read_text(encoding="utf-8") + f"\n\nIncident fingerprint: {fp}\nEvidence: {evidence_path}\nWorktree: {worktree}\nBranch: {branch}\n", encoding="utf-8")
        invoke_model("gpt-5.6-luna", prompt, args.report_schema, worktree, incident_dir / "luna_report.json", incident_dir / "luna_transcript.log", args.codex_bin, args.model_timeout)
        luna = read_json(incident_dir / "luna_report.json")
        if luna is None:
            raise RuntimeError("Luna report is not valid JSON")
        luna = validate_luna_report(luna, fp, worktree, branch, base)
        atomic_json(incident_dir / "luna_report.json", luna)
        entry["luna_calls"] = 1
        entry["state"] = "luna_completed"
        commit = luna["validated_head"]
        diff_hash = luna["diff_sha256"]
    state["cooldown_until"] = now + args.cooldown_seconds
    audit(args.state_dir, state, "luna_called", fingerprint=fp, worktree=str(worktree), branch=branch, commit=commit, llm_calls=1)
    # Persist the one-call budget before SOL starts.  A SOL timeout or malformed
    # verdict must never cause a second Luna implementation call on retry.
    atomic_json(args.state_dir / "state.json", state)

    sol_path = incident_dir / "sol_verdict.json"
    if args.stub:
        sol = fake_sol(
            fp, commit, diff_hash, fail_command=args.stub_fail_command,
        )
        atomic_json(sol_path, sol)
    else:
        sol_prompt = incident_dir / "sol_prompt.md"
        sol_prompt.write_text(args.sol_prompt.read_text(encoding="utf-8") + f"\n\nIncident fingerprint: {fp}\nEvidence: {evidence_path}\nLuna report: {incident_dir / 'luna_report.json'}\nReviewed worktree: {worktree}\nReviewed branch: {branch}\nReviewed commit: {commit}\nReviewed diff SHA-256: {diff_hash}\n", encoding="utf-8")
        invoke_model("gpt-5.6-sol", sol_prompt, args.sol_schema, worktree, sol_path, incident_dir / "sol_transcript.log", args.codex_bin, args.model_timeout)
        sol = read_json(sol_path)
        if sol is None:
            raise RuntimeError("SOL verdict is not valid JSON")
    entry["sol_calls"] = 1
    fresh = sol_path.is_file() and 0 <= time.time() - sol_path.stat().st_mtime <= args.fresh_seconds
    authorized = bool(
        sol.get("decision") == "approve"
        and sol.get("execution_authorized") is True
        and sol.get("incident_fingerprint") == fp
        and sol.get("reviewed_commit") == commit
        and sol.get("reviewed_diff_sha256") == diff_hash
        and fresh
        and isinstance(sol.get("approved_command"), str)
        and isinstance(sol.get("rollback_command"), str)
    )
    entry.update({
        "state": "pending_approved" if authorized else "reviewed_not_authorized",
        "fresh": fresh,
        "commit": commit,
        "diff_sha256": diff_hash,
        "approved": authorized,
        "rollback_command": sol.get("rollback_command"),
    })
    if authorized:
        entry["approval"] = bind_approval(sol, sol_path, fp, commit, diff_hash)
    atomic_json(args.state_dir / "state.json", state)
    audit(args.state_dir, state, "sol_reviewed", fingerprint=fp, commit=commit, diff_sha256=diff_hash, decision=sol.get("decision"), revision_match=sol.get("reviewed_commit") == commit, fresh=fresh, execution_authorized=authorized, llm_calls=1)
    if not authorized:
        atomic_json(args.state_dir / "state.json", state)
        return {"status": "not_authorized", "fingerprint": fp, "llm_calls": 2}
    command = sol["approved_command"]
    print(f"APPROVED_COMMAND={command}")
    live = live_train_or_eval_processes(status)
    if live or not args.execute_approved:
        audit(
            args.state_dir, state, "approved_command_pending",
            fingerprint=fp, command=command, live_train_or_eval=live[:8],
            llm_calls=0,
        )
        atomic_json(args.state_dir / "state.json", state)
        return {
            "status": "pending_approved" if live else "approved",
            "fingerprint": fp,
            "approved_command": command,
            "live_train_or_eval": live[:8],
            "llm_calls": 2,
        }
    executed = execute_approval(
        args, state, fp, entry, status, incident_dir,
    )
    executed["llm_calls"] = 2
    return executed


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-driven Phase-8 Qwen bucket incident repair controller")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--stub-fail-command", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--alert", type=Path, default=DEFAULT_ALERT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--resume-marker", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worktree-root", type=Path, default=Path("/home/kojiek/codex-worktrees"))
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "docs/experiments/phase8_qwen_bucket_quarter_backlog_2026_07_26.md",
    )
    parser.add_argument("--cooldown-seconds", type=int, default=900)
    parser.add_argument("--fresh-seconds", type=int, default=DEFAULT_FRESH_SECONDS)
    parser.add_argument("--model-timeout", type=int, default=900)
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--command-output-bytes",
        type=int,
        default=DEFAULT_COMMAND_OUTPUT_BYTES,
    )
    parser.add_argument(
        "--forward-progress-timeout-seconds",
        type=int,
        default=DEFAULT_FORWARD_PROGRESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "/home/kojiek/.local/bin/codex"))
    parser.add_argument("--report-schema", type=Path, default=ROOT / "scripts/phase8_qwen_bucket_repair_report.schema.json")
    parser.add_argument("--sol-schema", type=Path, default=ROOT / "scripts/phase8_qwen_bucket_sol_repair_verdict.schema.json")
    parser.add_argument("--luna-prompt", type=Path, default=ROOT / "docs/experiments/phase8_qwen_bucket_repair_luna_prompt.md")
    parser.add_argument("--sol-prompt", type=Path, default=ROOT / "docs/experiments/phase8_qwen_bucket_repair_sol_prompt.md")
    args = parser.parse_args(sys.argv[2:])
    if (
        args.interval_seconds < 1
        or args.cooldown_seconds < 0
        or args.fresh_seconds < 1
        or args.command_timeout_seconds < 1
        or args.command_output_bytes < 256
        or args.forward_progress_timeout_seconds < 1
    ):
        raise SystemExit(
            "interval/fresh/command/forward-progress must be positive; "
            "cooldown >=0; command-output-bytes >=256"
        )
    if args.execute_approved and args.dry_run:
        raise SystemExit("--execute-approved requires a non-dry-run mode")
    args.state_dir = args.state_dir.resolve()
    args.resume_marker = (
        args.resume_marker.resolve()
        if args.resume_marker
        else args.state_dir / "RESUME_AUTHORIZED.json"
    )
    args.root = args.root.resolve()
    args.worktree_root = args.worktree_root.resolve()
    args.status = args.status.resolve()
    args.alert = args.alert.resolve()
    if args.contract:
        args.contract = args.contract.resolve()
    lock_path = args.state_dir / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another repair controller is already running")
        state = load_state(args.state_dir / "state.json")
        while True:
            try:
                result = process_once(args, state)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:  # fail closed and preserve audit context
                audit(args.state_dir, state, "controller_error", error=str(exc)[:2000], llm_calls=0)
                atomic_json(args.state_dir / "state.json", state)
                print(json.dumps({"status": "error", "error": str(exc)[:2000], "llm_calls": 0}, sort_keys=True), flush=True)
                if not args.loop:
                    return 1
            if not args.loop or args.dry_run:
                return 0
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
PY
