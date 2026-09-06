#!/usr/bin/env python3
"""Resource-owning preflight controller for 028 random-quarter evaluation."""

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
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json, validate_cfg0_report  # noqa: E402
from secondary_queue_controller import Controller  # noqa: E402

STATE_ROOT = Path("/home/kojiek/logs/caption2p0_random_quarter_neg_cfg1p5_harn")
STALL_SECONDS = 1800
INTERRUPTED = False


def _signal(_signum: int, _frame: object) -> None:
    global INTERRUPTED
    INTERRUPTED = True


def job_script() -> Path:
    return Path(os.environ.get("GPU_QUEUE_JOB_SCRIPT") or sys.argv[0]).resolve()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_secondary_reports(contract: dict) -> list[dict]:
    expected_negative = str(contract["protocol"]["negative_prompt"])
    evidence = []
    for spec in contract["secondary_reports"]:
        path = Path(spec["path"])
        rec = read_json(path)
        if not rec or rec.get("label") != spec["label"] or rec.get("exp_id") != spec["exp_id"]:
            raise ValueError(f"report identity mismatch: {path}")
        if float(rec.get("cfg_strength", -1)) != 1.5 or rec.get("negative_prompt") != expected_negative:
            raise ValueError(f"report protocol mismatch: {path}")
        full = (rec.get("aggregates") or {}).get("full") or {}
        if int(full.get("n") or 0) != 5521:
            raise ValueError(f"report row count mismatch: {path}")
        if any(not isinstance(full.get(key), (int, float)) for key in ("clap", "CE", "CU", "PC", "PQ")):
            raise ValueError(f"report metrics incomplete: {path}")
        evidence.append({"path": str(path), "sha256": digest(path)})
    if len(evidence) != 2:
        raise ValueError("exactly two secondary reports are required")
    return evidence


def progress(contract: dict) -> tuple[int, int, int]:
    reports = sum(Path(spec["path"]).is_file() for spec in contract["secondary_reports"])
    flacs, freshness = 0, 0
    for spec in contract["secondary_reports"]:
        directory = Path(spec["path"]).parent / "_audio" / spec["label"]
        if directory.is_dir():
            flacs += len(list(directory.glob("*.flac")))
            freshness = max(freshness, directory.stat().st_mtime_ns)
    return reports, flacs, freshness


def free_bytes(contract: dict) -> int:
    stats = os.statvfs(Path(contract["storage"]["path"]))
    return stats.f_bavail * stats.f_frsize


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def pause_ack(contract: dict, request: dict) -> None:
    checkpoint = Path(contract["resume"]["autoresume"])
    snapshot = progress(contract)
    atomic_json(checkpoint, {"document_kind": "secondary_eval_resume_v1", "written_at": now(), "progress": snapshot})
    atomic_json(Path(os.environ["P2_CONTROL_DIR"]) / "pause.ack.json", {
        "run_id": request["run_id"], "job_id": request["job_id"], "request_id": request["request_id"],
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "iteration": snapshot[1], "pid": os.getpid(), "start_time": pid_start_time(os.getpid()),
    })


def unnotified_hold(script: Path, reason: str, **extra: object) -> None:
    atomic_json(script.with_name(script.stem + ".terminal.json"), {
        "status": "held", "written_at": now(), "reason": reason, **extra,
    })


def main() -> int:
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    script = job_script()
    accepted, reason = accept_guest(script)
    if not accepted:
        unnotified_hold(script, reason)
        return 2
    controller = Controller(script, Path(os.environ["GPU_QUEUE_CONTRACT"]), STATE_ROOT)
    contract = controller.contract
    controller.record("mutable_preflight_started")
    checked = subprocess.run(list(contract["commands"]["preflight"]), env=os.environ.copy())
    if checked.returncode != 0:
        try:
            controller.terminal("held", "held", "held", f"028 mutable preflight failed rc={checked.returncode}.", reason="preflight failed")
        except RuntimeError as exc:
            unnotified_hold(script, str(exc))
        return 2
    free = free_bytes(contract)
    if free < int(contract["storage"]["hard_stop_free_bytes"]):
        try:
            controller.terminal("held", "held", "held", f"028 disk hard stop before child: free_bytes={free}.", reason="disk hard stop")
        except RuntimeError as exc:
            unnotified_hold(script, str(exc))
        return 2
    controller.record("mutable_preflight_passed", {"free_bytes": free})
    try:
        controller.pre_child_notifications("028 CFG1.5 negative-prompt MusicCaps-5521")
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
            stop_child(child); pause_ack(contract, request)
            try:
                controller.terminal("paused", "interrupted", "interrupted", "028 paused by P1; partial phase will resume safely.")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc)); return 2
            return PAUSE_EXIT
        if INTERRUPTED:
            stop_child(child)
            try:
                controller.terminal("interrupted", "interrupted", "interrupted", "028 controller received a termination signal.")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc)); return 2
            return 130
        free = free_bytes(contract)
        hard, warning = int(contract["storage"]["hard_stop_free_bytes"]), int(contract["storage"]["warning_free_bytes"])
        if free < hard:
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", f"028 disk hard stop: free_bytes={free}.", reason="disk hard stop")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc))
            return 2
        if free < warning and not warning_sent:
            try:
                controller.notify("disk_warning", "held", f"028 disk warning: free_bytes={free}.")
            except RuntimeError as exc:
                stop_child(child); unnotified_hold(script, str(exc)); return 2
            warning_sent = True
        current = progress(contract)
        if current != last:
            last, progressed = current, time.monotonic()
        elif time.monotonic() - progressed >= STALL_SECONDS:
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", "028 stalled without file progress.", reason="stall")
            except RuntimeError as exc:
                unnotified_hold(script, str(exc))
            return 2
        time.sleep(2)
    if child.returncode != 0:
        try:
            controller.terminal("failed", "failure", "failure", f"028 evaluation child failed rc={child.returncode}.", rc=child.returncode)
        except RuntimeError as exc:
            unnotified_hold(script, str(exc), child_rc=child.returncode); return 2
        return 128 + (-int(child.returncode)) if child.returncode < 0 else int(child.returncode)
    try:
        secondary = validate_secondary_reports(contract)
        compat = contract["completion_evidence"]
        ema, cfg0 = Path(compat["ema"]), Path(compat["cfg0_report"])
        reason = validate_cfg0_report(cfg0, ema, rows=5521)
        if reason:
            raise ValueError(reason)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        try:
            controller.terminal("held", "held", "held", f"028 post-run audit held: {exc}", reason=str(exc))
        except RuntimeError as notify_exc:
            unnotified_hold(script, str(notify_exc))
        return 2
    try:
        controller.terminal("completed", "success", "success", "028 completed; both MusicCaps-5521 reports passed provenance audit.", evidence={
            "ema": str(ema), "cfg0_report": str(cfg0), "secondary_reports": secondary,
        })
    except RuntimeError as exc:
        unnotified_hold(script, str(exc)); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
