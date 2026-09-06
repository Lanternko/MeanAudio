#!/usr/bin/env python3
"""Resource-owning queue guest for one-cell canonical CFG0 recovery evals."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kojiek/gpu_queue")
sys.path.insert(0, "/home/kojiek/MeanAudio/scripts/eval")
sys.path.insert(0, "/home/kojiek/MeanAudio/scripts/experiment_harness")
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json
from secondary_queue_controller import Controller
from validate_caption2p0_cfg0_report import validate


INTERRUPTED = False
STALL_SECONDS = 1800


def mark_signal(_signum: int, _frame: object) -> None:
    global INTERRUPTED
    INTERRUPTED = True


def job_script() -> Path:
    return Path(os.environ.get("GPU_QUEUE_JOB_SCRIPT") or sys.argv[0]).resolve()


def unnotified_hold(script: Path, reason: str, **extra: object) -> None:
    atomic_json(script.with_name(script.stem + ".terminal.json"), {
        "status": "held", "written_at": now(), "reason": reason, **extra,
    })


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def progress(contract: dict) -> tuple[int, int, int]:
    cell = contract["cells"][0]
    label = cell["label"]
    audio = Path(contract["runtime_storage"]["output_root"]) / label / "audio"
    report = Path(cell["report"])
    log = Path("/home/kojiek/logs") / f"{label}_eval.log"
    flacs = len(list(audio.glob("*.flac"))) if audio.is_dir() else 0
    freshness = log.stat().st_mtime_ns if log.is_file() else 0
    return int(report.is_file()), flacs, freshness


def free_bytes(contract: dict) -> int:
    root = Path(contract["runtime_storage"]["output_root"])
    return os.statvfs(root).f_bavail * os.statvfs(root).f_frsize


def pause_ack(contract: dict, request: dict) -> None:
    resume = Path(contract["resume"]["progress_checkpoint"])
    snapshot = progress(contract)
    atomic_json(resume, {"document_kind": "cfg0_eval_recovery_resume_v1", "written_at": now(), "progress": snapshot})
    atomic_json(Path(os.environ["P2_CONTROL_DIR"]) / "pause.ack.json", {
        "run_id": request["run_id"], "job_id": request["job_id"], "request_id": request["request_id"],
        "checkpoint": str(resume), "checkpoint_bytes": resume.stat().st_size,
        "iteration": snapshot[1], "pid": os.getpid(), "start_time": pid_start_time(os.getpid()),
    })


def main() -> int:
    signal.signal(signal.SIGTERM, mark_signal)
    signal.signal(signal.SIGINT, mark_signal)
    script = job_script()
    accepted, reason = accept_guest(script)
    if not accepted:
        unnotified_hold(script, reason)
        return 2
    contract_path = Path(os.environ["GPU_QUEUE_CONTRACT"]).resolve()
    state_root = Path("/home/kojiek/logs/cfg0_eval_recovery_harn") / script.stem
    controller = Controller(script, contract_path, state_root)
    contract = controller.contract
    controller.record("mutable_preflight_started")
    preflight = subprocess.run(list(contract["commands"]["preflight"]), env=os.environ.copy())
    if preflight.returncode != 0:
        try:
            controller.terminal("held", "held", "held", f"{script.stem} mutable preflight held rc={preflight.returncode}.", reason="preflight held")
        except RuntimeError as exc:
            unnotified_hold(script, str(exc))
        return 2
    controller.record("mutable_preflight_passed", {"free_bytes": free_bytes(contract)})
    try:
        controller.pre_child_notifications(f"{script.stem} canonical CFG0 eval-only recovery")
    except RuntimeError as exc:
        unnotified_hold(script, str(exc))
        return 2
    child = subprocess.Popen(list(contract["commands"]["run"]), env=os.environ.copy(), start_new_session=True)
    controller.record("evaluation_child_started", {"pid": child.pid, "start_time": pid_start_time(child.pid)})
    control = Path(os.environ.get("P2_CONTROL_DIR") or "")
    last, progressed, warning_sent = progress(contract), time.monotonic(), False
    while child.poll() is None:
        request = read_json(control / "pause.request.json") if control.is_dir() else None
        if request:
            stop_child(child)
            pause_ack(contract, request)
            try:
                controller.terminal("paused", "interrupted", "interrupted", f"{script.stem} paused; partial FLACs will be safely regenerated.")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc)); return 2
            return PAUSE_EXIT
        if INTERRUPTED:
            stop_child(child)
            try:
                controller.terminal("interrupted", "interrupted", "interrupted", f"{script.stem} controller interrupted.")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc)); return 2
            return 130
        free = free_bytes(contract)
        hard = int(contract["runtime_storage"]["hard_stop_free_bytes"])
        warning = int(contract["runtime_storage"]["warning_free_bytes"])
        if free < hard:
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", f"{script.stem} disk hard stop: free_bytes={free}.", reason="disk hard stop")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc))
            return 2
        if free < warning and not warning_sent:
            try:
                controller.notify("disk_warning", "held", f"{script.stem} disk warning: free_bytes={free}.")
            except RuntimeError as exc:
                stop_child(child); unnotified_hold(script, str(exc)); return 2
            warning_sent = True
        current = progress(contract)
        if current != last:
            last, progressed = current, time.monotonic()
        elif time.monotonic() - progressed >= STALL_SECONDS:
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", f"{script.stem} stalled without output progress.", reason="stall")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc))
            return 2
        time.sleep(2)
    if child.returncode != 0:
        try:
            controller.terminal("failed", "failure", "failure", f"{script.stem} evaluator failed rc={child.returncode}.", rc=child.returncode)
        except RuntimeError as exc:
            unnotified_hold(script, str(exc), child_rc=child.returncode); return 2
        return int(child.returncode)
    try:
        cell = contract["cells"][0]
        validate(contract_path, cell["cell_id"], Path(cell["report"]))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        try:
            controller.terminal("held", "held", "held", f"{script.stem} post-run audit held: {exc}", reason=str(exc))
        except RuntimeError as notify_exc:
            unnotified_hold(script, str(notify_exc))
        return 2
    try:
        controller.terminal("completed", "success", "success", f"{script.stem} canonical CFG0 recovery completed.", evidence={
            "ema": cell["checkpoint"], "cfg0_report": cell["report"],
        })
    except RuntimeError as exc:
        unnotified_hold(script, str(exc)); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
