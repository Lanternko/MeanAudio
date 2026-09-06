#!/usr/bin/env python3
"""Queue guest for the c2p0 slot0 CFG-2.5/4.0 MusicCaps secondary sweep."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kojiek/gpu_queue")
sys.path.insert(0, "/home/kojiek/MeanAudio/scripts/experiment_harness")
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json, validate_cfg0_report
from notification_receipts import validate_delivered_receipt
from secondary_queue_controller import Controller


STATE_ROOT = Path("/home/kojiek/logs/c2p0_slot0_neg_cfg2p5_cfg4p0_full5521_harn")
EVENT_LEDGER = STATE_ROOT / "event_ledger.json"
NOTIFIER = Path("/home/kojiek/MeanAudio/scripts/notify_experiment_webhook.py")
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")
STALL_SECONDS = 1800
INTERRUPTED = False


def _signal(_signum: int, _frame: object) -> None:
    global INTERRUPTED
    INTERRUPTED = True


def job_script() -> Path:
    return Path(os.environ.get("GPU_QUEUE_JOB_SCRIPT") or sys.argv[0]).resolve()


def terminal(status: str, **extra: object) -> None:
    script = job_script()
    atomic_json(script.with_name(script.stem + ".terminal.json"), {"status": status, "written_at": now(), **extra})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def contract() -> dict:
    return json.loads(Path(os.environ["GPU_QUEUE_CONTRACT"]).read_text(encoding="utf-8"))


def ensure_state() -> None:
    STATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(STATE_ROOT, 0o700)
    if not EVENT_LEDGER.exists():
        atomic_json(EVENT_LEDGER, {"document_kind": "secondary_eval_event_ledger_v1", "events": {}})
        os.chmod(EVENT_LEDGER, 0o600)
    record = read_json(EVENT_LEDGER)
    if (not isinstance(record, dict) or record.get("document_kind") != "secondary_eval_event_ledger_v1"
            or not isinstance(record.get("events"), dict)):
        raise RuntimeError("event ledger missing or invalid")
    if (EVENT_LEDGER.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError("event ledger mode is not 0600")


def update_event(key: str, **values: object) -> dict:
    ledger = read_json(EVENT_LEDGER)
    if (not isinstance(ledger, dict) or ledger.get("document_kind") != "secondary_eval_event_ledger_v1"
            or not isinstance(ledger.get("events"), dict)):
        raise RuntimeError("event ledger invalid")
    events = ledger["events"]
    event = events.setdefault(key, {"idempotency_key": key, "created_at": now()})
    event.update(values)
    atomic_json(EVENT_LEDGER, ledger)
    os.chmod(EVENT_LEDGER, 0o600)
    return event


def notify_once(key: str, status: str, summary: str) -> None:
    ledger = read_json(EVENT_LEDGER)
    if not isinstance(ledger, dict) or not isinstance(ledger.get("events"), dict):
        raise RuntimeError("event ledger invalid")
    event = ledger["events"].get(key) or {}
    if event.get("delivery") == "delivered":
        return
    if event.get("delivery") == "attempted":
        raise RuntimeError(f"notification delivery ambiguous after restart: {key}")
    update_event(key, delivery="attempted", status=status, summary=summary, attempted_at=now())
    completed = subprocess.run([
        str(PYTHON), str(NOTIFIER), "--status", status, "--experiment", key,
        "--summary", summary, "--exit-code", "0",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if completed.returncode != 0:
        update_event(key, delivery="failed", returncode=completed.returncode, failed_at=now())
        raise RuntimeError(f"required notification failed: {key}")
    update_event(key, delivery="delivered", delivered_at=now())


def free_bytes(specification: dict) -> int:
    stats = os.statvfs(specification["storage"]["path"])
    return stats.f_bavail * stats.f_frsize


def storage_verdict(free: int, warning: int, hard: int) -> str:
    if free < hard:
        return "hard_stop"
    if free < warning:
        return "warning"
    return "pass"


def is_stalled(current: tuple[int, int, int], previous: tuple[int, int, int],
               monotonic_now: float, last_progress_at: float, threshold: int = STALL_SECONDS) -> bool:
    return current == previous and monotonic_now - last_progress_at >= threshold


def progress_snapshot(specification: dict) -> tuple[int, int, int]:
    reports = sum(Path(spec["path"]).is_file() for spec in specification["secondary_reports"])
    flacs = 0
    for spec in specification["secondary_reports"]:
        tag = "cfg" + str(spec["cfg_strength"]).replace(".", "p")
        directory = Path(spec["path"]).parent / "_audio" / f"c2p0_slot0_full_noq_{tag}_neg"
        if directory.is_dir():
            flacs += len(list(directory.glob("*.flac")))
    log = Path("/home/kojiek/logs/c2p0_slot0_neg_cfg2p5_cfg4p0_full5521.log")
    freshness = log.stat().st_mtime_ns if log.is_file() else 0
    return reports, flacs, freshness


def stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def validate_reports(specification: dict) -> list[dict]:
    expected_negative = specification["protocol"]["negative_prompt"]
    inputs = {record["kind"]: record for record in specification["inputs"]}
    required = {
        "checkpoint": "checkpoint",
        "evaluation_tsv": "evaluation_tsv",
        "clap_checkpoint": "clap_checkpoint",
        "scorer": "scorer_reference",
        "eval_entrypoint": "evaluation_entrypoint",
        "eval_utils": "evaluation_utils",
        "paired_cfg0_reference": "paired_cfg0_per_clip_reference",
    }
    expected_identities = {
        report_key: {"path": inputs[input_kind]["path"], "sha256": inputs[input_kind]["sha256"]}
        for report_key, input_kind in required.items()
    }
    import csv
    with Path(inputs["evaluation_tsv"]["path"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_ids = {row["id"] for row in rows}
    if len(rows) != 5521 or len(expected_ids) != 5521:
        raise ValueError("registered MusicCaps TSV identity/count mismatch")
    evidence = []
    for spec in specification["secondary_reports"]:
        path = Path(spec["path"])
        record = read_json(path)
        if not record:
            raise ValueError(f"missing or invalid report: {path}")
        if record.get("label") != spec["label"] or record.get("exp_id") != spec["exp_id"]:
            raise ValueError(f"report identity mismatch: {path}")
        if float(record.get("cfg_strength", -1)) != float(spec["cfg_strength"]):
            raise ValueError(f"report CFG mismatch: {path}")
        if record.get("negative_prompt") != expected_negative:
            raise ValueError(f"report negative prompt mismatch: {path}")
        expected_protocol = {
            "classification": "secondary_noncanonical", "dataset": "MusicCaps", "rows": 5521,
            "solver": "MeanFlow", "steps": 25, "cfg_strength": float(spec["cfg_strength"]),
            "negative_prompt": expected_negative, "seed": 42, "mask": "NoMask",
            "precision": "full", "conditioning": "NoQ", "encoder_name": "t5_clap",
            "text_c_dim": 512,
        }
        if record.get("protocol") != expected_protocol:
            raise ValueError(f"report protocol mismatch: {path}")
        if record.get("input_identities") != expected_identities:
            raise ValueError(f"report input identity mismatch: {path}")
        cfg_text = str(spec["cfg_strength"])
        tag = "cfg" + cfg_text.replace(".", "p")
        directory = path.parent / "_audio" / f"c2p0_slot0_full_noq_{tag}_neg"
        expected_argv = [
            "/home/kojiek/venvs/dac/bin/python", inputs["evaluation_entrypoint"]["path"],
            "--variant", "meanaudio_s", "--model_path", inputs["checkpoint"]["path"],
            "--output", str(directory), "--tsv", inputs["evaluation_tsv"]["path"],
            "--use_meanflow", "--num_steps", "25", "--cfg_strength", cfg_text,
            "--negative_prompt", expected_negative, "--no_text_attention_mask",
            "--encoder_name", "t5_clap", "--text_c_dim", "512", "--seed", "42",
            "--full_precision", "--no_q",
        ]
        if record.get("generation_argv") != expected_argv:
            raise ValueError(f"report argv mismatch: {path}")
        full = (record.get("aggregates") or {}).get("full") or {}
        if int(full.get("n") or 0) != 5521:
            raise ValueError(f"report row count mismatch: {path}")
        for key in ("clap", "CE", "CU", "PC", "PQ"):
            if not isinstance(full.get(key), (int, float)) or not math.isfinite(float(full[key])):
                raise ValueError(f"report missing metric {key}: {path}")
        per_clip = record.get("per_clip") or {}
        if set(per_clip) != expected_ids:
            raise ValueError(f"per-clip count mismatch: {path}")
        for item, metrics in per_clip.items():
            if not isinstance(metrics, dict) or any(
                not isinstance(metrics.get(key), (int, float)) or not math.isfinite(float(metrics[key]))
                for key in ("clap", "CE", "CU", "PC", "PQ")
            ):
                raise ValueError(f"non-finite per-clip metric: {item}")
        evidence.append({"path": str(path), "sha256": digest(path), "cfg_strength": spec["cfg_strength"]})
    if len(evidence) != 2:
        raise ValueError("exactly two secondary reports are required")
    return evidence


def pause_ack(specification: dict, request: dict) -> None:
    checkpoint = Path(specification["resume"]["autoresume"])
    counts = {}
    for spec in specification["secondary_reports"]:
        tag = "cfg" + str(spec["cfg_strength"]).replace(".", "p")
        directory = Path(spec["path"]).parent / "_audio" / f"c2p0_slot0_full_noq_{tag}_neg"
        counts[tag] = {
            "report_complete": Path(spec["path"]).is_file(),
            "generated_flacs": len(list(directory.glob("*.flac"))) if directory.is_dir() else 0,
        }
    atomic_json(checkpoint, {"document_kind": "secondary_eval_resume_v1", "written_at": now(), "counts": counts})
    ack = {
        "run_id": request["run_id"], "job_id": request["job_id"], "request_id": request["request_id"],
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "iteration": sum(value["generated_flacs"] for value in counts.values()),
        "pid": os.getpid(), "start_time": pid_start_time(os.getpid()),
    }
    atomic_json(Path(os.environ["P2_CONTROL_DIR"]) / "pause.ack.json", ack)


def require_028_terminal(root: Path | None = None, contract_path: Path | None = None) -> dict:
    root = root or Path(os.environ.get("GPU_QUEUE_ROOT", "/home/kojiek/gpu_queue"))
    name = "028_random_quarter_neg_cfg1p5.sh"
    matches = [root / "p2" / state / name for state in ("done", "failed", "interrupted")]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"028 must exist in exactly one terminal directory, found={len(matches)}")
    launcher = matches[0]
    evidence_path = launcher.with_name(launcher.stem + ".terminal.json")
    evidence = read_json(evidence_path)
    if not evidence:
        raise ValueError("028 terminal evidence missing")
    status_map = {
        "completed": ("success", "success"),
        "failed": ("failure", "failure"),
        "interrupted": ("interrupted", "interrupted"),
    }
    expected = status_map.get(str(evidence.get("status") or ""))
    receipt = evidence.get("notification_receipt")
    if expected is None or not isinstance(receipt, dict):
        raise ValueError("028 terminal receipt reference invalid")
    event, status = expected
    path = Path(str(receipt.get("path") or ""))
    contract_path = contract_path or Path("/home/kojiek/MeanAudio/docs/experiments/caption2p0_random_quarter_neg_cfg1p5_contract.json")
    ok, reason = validate_delivered_receipt(
        path, contract_path=contract_path, launcher_path=launcher, event=event, status=status,
    )
    if not ok:
        raise ValueError(f"028 terminal receipt invalid: {reason}")
    return {"launcher": str(launcher), "terminal": str(evidence_path), "receipt": str(path), "status": evidence["status"]}


def main() -> int:
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    script = job_script()
    accepted, reason = accept_guest(script)
    if not accepted:
        terminal("held", reason=reason)
        return 2
    controller = Controller(script, Path(os.environ["GPU_QUEUE_CONTRACT"]), STATE_ROOT)
    specification = controller.contract
    controller.record("mutable_preflight_started")
    try:
        predecessor = require_028_terminal()
    except ValueError as error:
        try:
            controller.terminal("held", "held", "held", f"029 predecessor gate held: {error}", reason=str(error))
        except RuntimeError as notify_error:
            terminal("held", reason=str(notify_error))
        return 2
    preflight = list(specification["commands"]["preflight"])
    if subprocess.run(preflight, env=os.environ.copy()).returncode != 0:
        try:
            controller.terminal("held", "held", "held", "029 mutable preflight failed.", reason="preflight failed")
        except RuntimeError as error:
            terminal("held", reason=str(error))
        return 2
    free = free_bytes(specification)
    if free < int(specification["storage"]["hard_stop_free_bytes"]):
        try:
            controller.terminal("held", "held", "held", f"029 disk hard stop before child: free_bytes={free}.", reason="disk hard stop")
        except RuntimeError as error:
            terminal("held", reason=str(error))
        return 2
    controller.record("mutable_preflight_passed", {"predecessor": predecessor, "free_bytes": free})
    try:
        controller.pre_child_notifications("029 MusicCaps-5521 CFG 2.5 then CFG 4.0")
    except RuntimeError as error:
        terminal("held", reason=str(error))
        return 2
    child = subprocess.Popen(list(specification["commands"]["run"]), env=os.environ.copy(), start_new_session=True)
    controller.record("evaluation_child_started", {"pid": child.pid, "start_time": pid_start_time(child.pid)})
    control = Path(os.environ.get("P2_CONTROL_DIR") or "")
    last_progress = progress_snapshot(specification)
    last_progress_at = time.monotonic()
    warning_sent = False
    while child.poll() is None:
        request = read_json(control / "pause.request.json") if control.is_dir() else None
        if request:
            stop_child(child)
            pause_ack(specification, request)
            try:
                controller.terminal("paused", "interrupted", "interrupted", "029 paused by P1; partial phase restarts from seed 42.")
            except RuntimeError as error:
                terminal("held", reason=str(error))
                return 2
            return PAUSE_EXIT
        if INTERRUPTED:
            stop_child(child)
            try:
                controller.terminal("interrupted", "interrupted", "interrupted", "029 controller received a termination signal.")
            except RuntimeError as error:
                terminal("held", reason=str(error)); return 2
            return 130
        free = free_bytes(specification)
        hard = int(specification["storage"]["hard_stop_free_bytes"])
        warning = int(specification["storage"]["warning_free_bytes"])
        disk_state = storage_verdict(free, warning, hard)
        if disk_state == "hard_stop":
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", f"029 disk hard stop: free_bytes={free}, hard_floor={hard}.", reason="disk hard stop")
            except RuntimeError as error:
                terminal("held", reason=str(error))
            return 2
        if disk_state == "warning" and not warning_sent:
            try:
                controller.notify("disk_warning", "held", f"029 disk warning: free_bytes={free}, warning_floor={warning}.")
            except RuntimeError as error:
                stop_child(child)
                terminal("held", reason=str(error))
                return 2
            warning_sent = True
        current = progress_snapshot(specification)
        if current != last_progress:
            last_progress, last_progress_at = current, time.monotonic()
        elif is_stalled(current, last_progress, time.monotonic(), last_progress_at):
            stop_child(child)
            try:
                controller.terminal("held", "held", "held", f"029 stalled for {STALL_SECONDS} seconds.", reason="phase stall")
            except RuntimeError as error:
                terminal("held", reason=str(error))
            return 2
        time.sleep(2)
    if child.returncode != 0:
        try:
            controller.terminal("failed", "failure", "failure", f"029 evaluation child failed rc={child.returncode}.", rc=child.returncode)
        except RuntimeError as error:
            terminal("held", reason=str(error), child_rc=child.returncode)
            return 2
        return 128 + (-int(child.returncode)) if child.returncode < 0 else int(child.returncode)
    try:
        reports = validate_reports(specification)
        anchor = specification["completion_evidence"]
        reason = validate_cfg0_report(Path(anchor["cfg0_report"]), Path(anchor["ema"]), rows=5521)
        if reason:
            raise ValueError(f"CFG0 compatibility anchor invalid: {reason}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        try:
            controller.terminal("held", "held", "held", f"029 post-run audit held: {error}", reason=str(error))
        except RuntimeError as notify_error:
            terminal("held", reason=str(notify_error))
        return 2
    try:
        controller.terminal("completed", "success", "success", "029 completed; CFG 2.5 and CFG 4.0 reports passed provenance audit.", evidence={
            "ema": str(Path(anchor["ema"])), "cfg0_report": str(Path(anchor["cfg0_report"])),
            "secondary_reports": reports,
        })
    except RuntimeError as error:
        terminal("held", reason=str(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
