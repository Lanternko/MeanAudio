#!/usr/bin/env python3
"""Low-overhead health watcher for the NoQ-S1-to-Q-S2 bucket queue.

Routine polls are deliberately local-only: process/session state, a bounded log
tail, completed reports, and free disk.  A new hard incident is fingerprinted
once, persisted, and optionally sent to Discord.  It never consumes a GPU,
starts experiments, or invokes a model; a repair decision is deliberately
reserved for a newly fingerprinted hard incident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "phase8_qwen_s2q_from_noq_watcher"
STATUS_PATH = STATE_DIR / "status.json"
ALERT_PATH = STATE_DIR / "ALERT.json"
INCIDENT_PATH = STATE_DIR / "INCIDENT.json"
QUEUE_LOG = LOG_ROOT / "phase8_qwen_s2q_from_noq_sequence.launch.log"
TMUX_SESSION = "phase8_s2q_grid"
ARMS = (
    "k3_balanced",
    "k2_balanced",
    "k2_fixed",
    "k3_fixed",
    "k5_balanced",
    "k5_fixed",
)
HARD_MARKERS = (
    "CUDA out of memory",
    "ChildFailedError",
    "ProcessExitedException",
    "NCCL",
    "Segmentation fault",
    "Traceback (most recent call last)",
    "unbound variable",
    "[FAIL]",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def tail_lines(path: Path, limit: int = 100) -> list[str]:
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64 * 1024))
        return handle.read().decode("utf-8", errors="replace").splitlines()[-limit:]


def command_lines(command: list[str]) -> list[str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def tmux_running() -> bool:
    return TMUX_SESSION in command_lines(["tmux", "list-sessions", "-F", "#{session_name}"])


def queue_processes() -> list[str]:
    lines = command_lines(["ps", "-eo", "pid,etime,pcpu,pmem,args", "--no-headers"])
    # This watcher owns only the completed quarter-scale backlog.  Do not
    # fingerprint unrelated full-scale continuations whose names share the
    # generic S2Q prefix; doing so caused false stale incidents and repeated
    # Discord alerts while a valid full run was training.
    return [
        line[:1000]
        for line in lines
        if "phase8_qwen_s2q_from_noq_quarter_" in line
    ][:20]


def report_state(arm: str) -> dict[str, Any]:
    prefix = f"phase8_qwen_s2q_from_noq_quarter_{arm}"
    report = LOG_ROOT / f"{prefix}_FINAL_METRICS.json"
    audit = LOG_ROOT / f"{prefix}_FINAL_TRAIN_AUDIT.json"
    result: dict[str, Any] = {"arm": arm, "report": str(report), "audit": str(audit)}
    for key, path in (("report", report), ("audit", audit)):
        if not path.is_file():
            result[f"{key}_status"] = "missing"
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result[f"{key}_status"] = payload.get("status", "invalid")
        except (OSError, json.JSONDecodeError):
            result[f"{key}_status"] = "invalid"
    result["complete"] = result["report_status"] == "passed"
    return result


def fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def notify(incident: dict[str, Any]) -> None:
    notifier = ROOT / "scripts/notify_experiment_webhook.py"
    subprocess.run(
        [
            "python", str(notifier), "--status", "failure",
            "--experiment", "phase8_qwen_s2q_from_noq_queue",
            "--exit-code", "1", "--started-epoch", str(int(time.time())),
            "--summary", f"watcher hard incident: {'; '.join(incident['issues'])[:900]}",
            "--log", str(QUEUE_LOG),
        ],
        cwd=ROOT, check=False, timeout=30,
    )


def check(*, send_notification: bool) -> int:
    now = datetime.now(timezone.utc)
    arms = [report_state(arm) for arm in ARMS]
    incomplete = [arm["arm"] for arm in arms if not arm["complete"]]
    processes = queue_processes()
    work_processes = [
        process for process in processes
        if any(token in process for token in ("train.py", "eval.py", "phase4_eval.py"))
    ]
    session = tmux_running()
    queue_tail = tail_lines(QUEUE_LOG)
    issues = [marker for marker in HARD_MARKERS if any(marker.lower() in line.lower() for line in queue_tail)]
    if incomplete and not processes and not session:
        issues.append("queue exited before all reports completed")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 50:
        issues.append(f"workspace free space below 50 GiB ({free_gib:.1f} GiB)")

    active_logs = sorted(
        LOG_ROOT.glob("phase8_qwen_s2q_from_noq_quarter_*_stage2_50000*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    active_log = active_logs[-1] if active_logs else None
    if work_processes and active_log and now.timestamp() - active_log.stat().st_mtime > 1_200:
        issues.append(f"active queue with stale log ({now.timestamp() - active_log.stat().st_mtime:.0f}s)")

    status = "hard_incident" if issues else ("complete" if not incomplete else "healthy")
    payload = {
        "schema_version": 1,
        "watcher": "phase8_qwen_s2q_queue",
        "updated_at": now.isoformat(),
        "status": status,
        "queue_session": TMUX_SESSION,
        "queue_running": session,
        "processes": processes,
        "work_processes": work_processes,
        "active_log": str(active_log) if active_log else None,
        "queue_log": str(QUEUE_LOG),
        "queue_log_tail": queue_tail[-100:],
        "arms": arms,
        "first_incomplete": incomplete[0] if incomplete else None,
        "workspace_free_gib": round(free_gib, 1),
        "hard_incidents": [{"code": "s2q_queue", "detail": issue} for issue in issues],
    }
    atomic_json(STATUS_PATH, payload)

    if not issues:
        ALERT_PATH.unlink(missing_ok=True)
        print(f"status={status} incomplete={len(incomplete)} queue_running={session} issues=0")
        return 0

    incident = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "fingerprint": fingerprint({"issues": issues, "first_incomplete": payload["first_incomplete"]}),
        "issues": issues,
        "status_path": str(STATUS_PATH),
        "evidence": {key: payload[key] for key in ("processes", "active_log", "queue_log_tail", "arms")},
        "repair_policy": "one bounded investigation only for this new fingerprint; no routine model calls",
    }
    previous = json.loads(INCIDENT_PATH.read_text()) if INCIDENT_PATH.is_file() else {}
    is_new = previous.get("fingerprint") != incident["fingerprint"]
    atomic_json(ALERT_PATH, incident)
    atomic_json(INCIDENT_PATH, incident)
    if send_notification and is_new:
        notify(incident)
    print(f"status=hard_incident incomplete={len(incomplete)} queue_running={session} issues={len(issues)} new={is_new}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    if args.once == args.loop or args.interval_seconds < 30:
        raise SystemExit("choose exactly one of --once/--loop; interval must be >=30 seconds")
    while True:
        check(send_notification=args.notify)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
