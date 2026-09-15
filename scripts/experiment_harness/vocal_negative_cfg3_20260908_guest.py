#!/usr/bin/env python3
"""Resource-owning queue controller for the 044 single-negative CFG3 ablation."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kojiek/gpu_queue")
sys.path.insert(0, "/home/kojiek/MeanAudio/scripts/experiment_harness")
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json  # noqa: E402
from notification_receipts import validate_delivered_receipt  # noqa: E402
from secondary_queue_controller import Controller  # noqa: E402

STATE = Path("/home/kojiek/logs/vocal_negative_20260908_harn")
INTERRUPTED = False


def handler(_signum, _frame):
    global INTERRUPTED
    INTERRUPTED = True


def digest(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()



def progress(contract):
    reports = sum(Path(item["path"]).is_file() for item in contract["reports"])
    audio_root = Path(contract["storage"]["transient_root"])
    flacs = len(list(audio_root.glob("*/*.flac"))) if audio_root.is_dir() else 0
    freshness = max([p.stat().st_mtime_ns for p in audio_root.glob("*/*.flac")] + [0]) if audio_root.is_dir() else 0
    return reports, flacs, freshness


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def main() -> int:
    signal.signal(signal.SIGTERM, handler); signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGHUP, handler)
    script = Path(os.environ.get("GPU_QUEUE_JOB_SCRIPT") or sys.argv[0]).resolve()
    accepted, reason = accept_guest(script)
    if not accepted:
        atomic_json(script.with_name(script.stem + ".terminal.json"), {"status": "held", "written_at": now(), "reason": reason})
        return 2
    for _ in range(50):
        seat = read_json(Path("/home/kojiek/gpu_queue/p2.running.json")) or {}
        if seat.get("pid") == os.getpid():
            break
        time.sleep(.1)
    if seat.get("job_id") != script.stem or seat.get("run_id") != os.environ.get("P2_RUN_ID") or seat.get("pid") != os.getpid() or seat.get("start_time") != pid_start_time(os.getpid()):
        raise SystemExit("HOLD: exact P2 process ownership required")
    controller = Controller(script, Path(os.environ["GPU_QUEUE_CONTRACT"]), STATE)
    contract = controller.contract
    try:
        prior = {"ordering": "P2 append-only scheduler; no scientific dependency"}
        checked = subprocess.run(contract["commands"]["preflight"])
        if checked.returncode:
            raise ValueError(f"preflight rc={checked.returncode}")
    except ValueError as exc:
        controller.terminal("held", "held", "held", f"044 held: {exc}", reason=str(exc)); return 2
    controller.record("mutable_preflight_passed", {"predecessor": prior})
    controller.notify("preflight_pass", "start", "044 immutable inputs, protocol, and storage gate passed.")
    controller.pre_child_notifications("044 CFG3 fidelity8 + vocals/singing/choir MusicCaps-5521 ablation")
    child = subprocess.Popen(contract["commands"]["run"], start_new_session=True)
    controller.record("evaluation_child_started", {"pid": child.pid, "start_time": pid_start_time(child.pid)})
    last, changed = progress(contract), time.monotonic()
    control = Path(os.environ.get("P2_CONTROL_DIR") or "")
    warned = False
    while child.poll() is None:
        fs = os.statvfs(contract["storage"]["path"])
        free = fs.f_bavail * fs.f_frsize
        if free < contract["storage"]["hard_stop_free_bytes"]:
            stop(child)
            controller.terminal("held", "held", "held", f"044 disk hard stop: free={free}; resume same cell when capacity clears.", reason="storage")
            return 2
        if free < contract["storage"]["warning_free_bytes"] and not warned:
            controller.notify("disk_warning", "held", f"Storage advisory: free={free}; current evaluation remains running.")
            warned = True
        request = read_json(control / "pause.request.json") if control.is_dir() else None
        if request:
            stop(child)
            resume = Path(contract["resume"]["autoresume"])
            atomic_json(resume, {"document_kind": "single_negprompt_cfg3_resume_v1", "written_at": now(), "progress": progress(contract)})
            atomic_json(control / "pause.ack.json", {"run_id": request["run_id"], "job_id": request["job_id"],
                "request_id": request["request_id"], "checkpoint": str(resume), "checkpoint_bytes": resume.stat().st_size,
                "iteration": progress(contract)[0], "pid": os.getpid(), "start_time": pid_start_time(os.getpid())})
            controller.terminal("paused", "interrupted", "interrupted", "044 paused by P1; completed reports retained.")
            return PAUSE_EXIT
        if INTERRUPTED:
            stop(child); controller.terminal("interrupted", "interrupted", "interrupted", "044 controller interrupted."); return 130
        current = progress(contract)
        if current != last: last, changed = current, time.monotonic()
        elif time.monotonic() - changed > contract["watcher"]["stall_seconds"]:
            stop(child); controller.terminal("held", "held", "held", "044 stalled.", reason="stall"); return 2
        time.sleep(2)
    if child.returncode:
        controller.terminal("failed", "failure", "failure", f"044 child failed rc={child.returncode}.", rc=child.returncode)
        return child.returncode
    if subprocess.run(contract["commands"]["postflight"]).returncode:
        controller.terminal("held", "held", "held", "044 postflight report audit failed.", reason="postflight"); return 2
    evidence = [{"path": item["path"], "sha256": digest(Path(item["path"]))} for item in contract["reports"]]
    controller.terminal("completed", "success", "success", "044 completed all five CFG3 fidelity8 vocal-negative cells.",
        evidence={"ema": contract["completion_evidence"]["ema"], "cfg0_report": contract["completion_evidence"]["cfg0_report"], "secondary_reports": evidence})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
