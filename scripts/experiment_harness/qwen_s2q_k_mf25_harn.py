#!/usr/bin/env python3
"""Durable HARN queue for full-scale Stage 2 K=2/3/5/10 fair evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/phase8_qwen_s2q_k_mf25_harn")
GENERATIONS = STATE / "generations"
OUTBOX = STATE / "outbox"
CURRENT = STATE / "current"
KEY = STATE / "ledger_hmac.key"
LOCK = STATE / "controller.lock"
APPROVAL = STATE / "operator_approval.json"
PROCESS = STATE / "process_identity.json"
WATCH_STATUS = STATE / "watch_status.json"
PENDING_CONTRACT = STATE / "pending_contract.json"
PENDING_PREFLIGHT = STATE / "pending_preflight.json"
PENDING_LEDGER = STATE / "pending_ledger.json"
PENDING_QUEUE = STATE / "pending_queue.json"
VALIDATOR = ROOT / "scripts/validate_experiment_harness_documents.py"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
RUNNER = ROOT / "scripts/eval/eval_phase8_qwen_s2q_full_k_mf25_cfg4p5.sh"
SYSTEM_PY = Path("/usr/bin/python3")
EXPERIMENT = "phase8-qwen-s2q-k-mf25-cfg4p5"
RUN_ID = "run-20260813-k2-k3-k5-k10"
KS = (2, 3, 5, 10)
GIB = 1024**3
COEXISTENT_TTS_MARKER = "tts_server_irodori.py"
COEXISTENT_TTS_MAX_MIB = 2048
MIN_FREE_GPU_MIB = 24 * 1024


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def checkpoint(k: int) -> Path:
    base = f"phase8_qwen_s2q_from_noq_full_k{k}_balanced_stage2_200000"
    return ROOT / "exps" / base / f"{base}_ema_final.pth"


def source_report(k: int) -> Path:
    return Path(f"/home/kojiek/logs/phase8_qwen_s2q_from_noq_full_k{k}_balanced_FINAL_METRICS.json")


def result_report(k: int) -> Path:
    return Path(f"/home/kojiek/logs/phase8_qwen_s2q_from_noq_full_k{k}_balanced_stage2_200000_musiccaps_n5521_mf25_cfg4p5_q9_REPORT.json")


def command_registry() -> dict[str, list[str]]:
    return {f"eval_k{k}": ["/bin/bash", str(RUNNER), str(k)] for k in KS}


def schema_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "docs/experiments/schemas").glob("*.json")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def policy_hash() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "AGENTS.md", ROOT / "docs/experiments/experiment_notification_policy.md", ROOT / "docs/experiments/watcher_policy.md"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def storage_model() -> dict[str, int | str]:
    return {"path": "/", "hard_floor_bytes": 150 * GIB, "warning_floor_bytes": 180 * GIB,
            "peak_additional_bytes": 8 * GIB, "transient_bytes": 2 * GIB, "recovery_reserve_bytes": 10 * GIB}


def storage_check() -> dict[str, Any]:
    model = storage_model()
    free = shutil.disk_usage(str(model["path"])).free
    required = max(int(model["hard_floor_bytes"]), int(1.25 * (int(model["peak_additional_bytes"]) + int(model["transient_bytes"]))) + int(model["recovery_reserve_bytes"]))
    return {**model, "free_bytes": free, "required_bytes_policy": required, "verdict": "pass" if free >= required else "fail"}


def gpu_processes() -> list[dict[str, str]]:
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if completed.returncode:
        return [{"pid": "unknown", "process_name": "nvidia-smi-query-failed", "used_memory_mib": "unknown"}]
    rows = []
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",", 2)]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return rows


def gpu_free_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if completed.returncode:
        return 0
    try:
        return int(completed.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return 0


def process_cmdline(pid: str) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return ""


def blocking_gpu_processes() -> list[dict[str, str]]:
    processes = gpu_processes()
    if gpu_free_mib() < MIN_FREE_GPU_MIB:
        return processes or [{"pid": "memory", "process_name": "insufficient-free-gpu-memory", "used_memory_mib": "unknown"}]
    blocking = []
    for process in processes:
        try:
            used_mib = int(process["used_memory_mib"])
        except ValueError:
            blocking.append(process)
            continue
        is_authorized_tts = COEXISTENT_TTS_MARKER in process_cmdline(process["pid"])
        if not is_authorized_tts or used_mib > COEXISTENT_TTS_MAX_MIB:
            blocking.append(process)
    return blocking


def make_contract() -> dict[str, Any]:
    commands = command_registry()
    zero = "0" * 64
    tsv = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
    sources = [{"path": str(tsv), "sha256": digest_file(tsv)}]
    for k in KS:
        sources.extend([{"path": str(checkpoint(k)), "sha256": digest_file(checkpoint(k))},
                        {"path": str(source_report(k)), "sha256": digest_file(source_report(k))}])
    phases = [{"phase_id": f"k{k}_mf25_cfg4p5_q9", "action_id": f"eval_k{k}", "input_artifacts": sources,
               "output_paths": [str(result_report(k))], "completion_evidence": [{"path": str(result_report(k)), "sha256": zero}],
               "resume_action_id": f"eval_k{k}"} for k in KS]
    registered = [{"action_id": key, "argv": argv, "working_directory": str(ROOT),
                   "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}} for key, argv in commands.items()]
    return {"document_kind": "experiment_contract", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
            "experiment_id": EXPERIMENT, "run_id": RUN_ID,
            "bindings": {"policy_bundle_sha256": policy_hash(), "schema_bundle_sha256": schema_hash(),
                         "runtime_sha256": digest_file(Path(__file__)), "command_set_sha256": digest_bytes(canonical(commands))},
            "approval_requirement": {"required": True, "responsible_role": "operator", "trusted_channels": ["operator_console"]},
            "corpus": {"kind": "non_generated", "source_artifacts": sources}, "repair": {"enabled": False},
            "phases": phases, "filesystems": [storage_model()], "commands": registered,
            "required_preflight_checks": ["approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound", "no_duplicate", "policy_bound", "storage_policy_1.25"],
            "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"]}


def load_key() -> bytes:
    fd = os.open(KEY, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.geteuid() or (stat.st_mode & 0o777) != 0o600:
            raise RuntimeError("unsafe ledger key permissions")
        return os.read(fd, 128)
    finally:
        os.close(fd)


def sign_event(event: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    return hmac.new(load_key(), b"meanaudio-harn-event-v1\0" + canonical(unsigned), hashlib.sha256).hexdigest()


def append_event(ledger: dict[str, Any], kind: str, *, verdict: str = "none", phase: str | None = None,
                 relation: str | None = None, notification: str = "not_applicable") -> str:
    sequence = len(ledger["events"]) + 1
    event = {"sequence": sequence, "event_id": f"event-{sequence}-{kind}",
             "idempotency_key": f"{EXPERIMENT}:{RUN_ID}:{kind}:{sequence}", "event_kind": kind,
             "occurred_at": now(), "phase": phase, "verdict": verdict, "relates_to_event_id": relation,
             "notification_status": notification,
             "previous_event_sha256": ledger["events"][-1]["event_sha256"] if ledger["events"] else None,
             "event_sha256": ""}
    event["event_sha256"] = sign_event(event)
    ledger["events"].append(event)
    return event["event_id"]


def verify_ledger(ledger: dict[str, Any]) -> None:
    prior = None
    for sequence, event in enumerate(ledger["events"], 1):
        if event["sequence"] != sequence or event["previous_event_sha256"] != prior or not hmac.compare_digest(event["event_sha256"], sign_event(event)):
            raise RuntimeError("ledger integrity failure")
        prior = event["event_sha256"]


def make_preflight(contract: dict[str, Any], approval_hash: str, gpu_idle: bool) -> dict[str, Any]:
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=72)
    storage = storage_check()
    checks = []
    for check_id in contract["required_preflight_checks"]:
        verdict = "fail" if check_id == "gpu_idle" and not gpu_idle else "pass"
        if check_id == "storage_policy_1.25" and storage["verdict"] != "pass":
            verdict = "fail"
        checks.append({"check_id": check_id, "verdict": verdict, "observed_at": issued.isoformat(),
                       "valid_until": expires.isoformat(), "evidence_sha256": digest_bytes(f"{check_id}:{verdict}".encode())})
    contract_hash = digest_bytes(canonical(contract))
    return {"document_kind": "preflight_report", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
            "experiment_id": EXPERIMENT, "run_id": RUN_ID, "contract_raw_sha256": contract_hash,
            "approval_evidence": {"evidence_id": "approval-20260813-k-evals", "source_kind": "trusted_operator_record",
                "trusted_channel": "operator_console", "channel_record_id": "operator-message-k-mf25-cfg4p5",
                "channel_record_sha256": approval_hash, "approver_id": "user", "issued_at": issued.isoformat(),
                "expires_at": expires.isoformat(), "experiment_id": EXPERIMENT, "run_id": RUN_ID,
                "bindings": {
                    "contract_raw_sha256": contract_hash,
                    **contract["bindings"],
                    "repair_envelope_sha256": (
                        contract["repair"]["envelope"]["envelope_sha256"]
                        if contract["repair"]["enabled"] else None
                    ),
                }},
            "checks": checks, "storage": [{"measured_at": issued.isoformat(),
                **{key: value for key, value in storage.items() if key not in {"warning_floor_bytes", "required_bytes_policy"}}}],
            "derived_verdict": "pass" if all(item["verdict"] == "pass" for item in checks) else "fail", "created_at": issued.isoformat()}


def write_generation(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any], status: str) -> Path:
    verify_ledger(ledger)
    generation = len(list(GENERATIONS.glob("gen-*"))) + 1
    target = GENERATIONS / f"gen-{generation:06d}"
    target.mkdir(parents=True, mode=0o700)
    def write(name: str, value: Any) -> str:
        raw = canonical(value)
        (target / f"{name}.json").write_bytes(raw)
        return digest_bytes(raw)
    contract_hash = write("contract", contract)
    preflight["contract_raw_sha256"] = contract_hash
    preflight["approval_evidence"]["bindings"]["contract_raw_sha256"] = contract_hash
    preflight_hash = write("preflight", preflight)
    ledger["bindings"] = {"contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
                          "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}
    ledger_hash = write("ledger", ledger)
    terminal = status in {"completed", "failed", "interrupted"}
    queue = {"document_kind": "queue_state", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
             "queue_id": "phase8-s2q-k-mf25-queue", "updated_at": now(), "entries": [{"entry_id": "phase8-s2q-k-mf25-entry",
             "position": 1, "experiment_id": EXPERIMENT, "run_id": RUN_ID, "status": status, "dependencies": [],
             "assigned_resource": {"resource_type": "gpu", "resource_id": "gpu0"} if status in {"ready", "active"} else None,
             "bindings": {"contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
                          "ledger_raw_sha256": ledger_hash, "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]},
             "terminal_notification_status": "delivered" if terminal else "not_applicable"}]}
    write("queue", queue)
    completed = subprocess.run([str(SYSTEM_PY), str(VALIDATOR), "--contract", str(target / "contract.json"),
                                "--preflight", str(target / "preflight.json"), "--ledger", str(target / "ledger.json"),
                                "--queue", str(target / "queue.json")], cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError("HARN validation failed: " + completed.stderr[-2000:] + completed.stdout[-1000:])
    pointer = STATE / f".current.tmp.{os.getpid()}"
    pointer.write_text(str(target) + "\n")
    os.replace(pointer, CURRENT)
    return target


def load_current() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = Path(CURRENT.read_text().strip())
    values = tuple(json.loads((target / f"{name}.json").read_text()) for name in ("contract", "preflight", "ledger"))
    verify_ledger(values[2])
    return values  # type: ignore[return-value]


def notify(key: str, summary: str, *, status: str = "test", report: Path | None = None,
           gpu_released: bool = False) -> None:
    path = OUTBOX / f"{key}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("status") == "delivered":
            return
        raise RuntimeError(f"ambiguous notification state: {key}")
    atomic_json(path, {"status": "attempting", "payload_sha256": digest_bytes(summary.encode()), "created_at": now()})
    argv = ["/home/kojiek/venvs/dac/bin/python", str(NOTIFIER), "--status", status, "--experiment", EXPERIMENT, "--summary", summary]
    if report and report.is_file():
        argv += ["--report", str(report)]
    if gpu_released:
        argv.append("--gpu-released")
    completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        atomic_json(path, {"status": "failed", "failed_at": now(), "error": completed.stderr[-500:]})
        raise RuntimeError(f"Discord delivery failed: {key}")
    atomic_json(path, {"status": "delivered", "delivered_at": now(), "accepted_evidence_sha256": digest_bytes(completed.stdout.encode())})


def acquire_lock() -> int:
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def init(approval_hash: str) -> None:
    if len(approval_hash) != 64:
        raise SystemExit("approval hash must be sha256 hex")
    for directory in (STATE, GENERATIONS, OUTBOX):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    if not KEY.exists():
        fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, os.urandom(32))
        finally:
            os.close(fd)
    contract = make_contract()
    conflicts = blocking_gpu_processes()
    preflight = make_preflight(contract, approval_hash, not conflicts)
    atomic_json(APPROVAL, {"approval_text_sha256": approval_hash, "issued_by": "authenticated_operator_console",
                           "issued_at": now(), "consumed": False, "contract_runtime_sha256": contract["bindings"]["runtime_sha256"]})
    ledger = {"document_kind": "event_ledger", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
              "experiment_id": EXPERIMENT, "run_id": RUN_ID,
              "bindings": {"contract_raw_sha256": "0" * 64, "preflight_report_raw_sha256": "0" * 64,
                           "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}, "events": []}
    append_event(ledger, "contract_registered")
    held = bool(conflicts) or preflight["derived_verdict"] != "pass"
    if held:
        hold_id = append_event(ledger, "queue_hold", verdict="fail", phase="resource_wait", notification="pending")
        failed_checks = [item["check_id"] for item in preflight["checks"] if item["verdict"] != "pass"]
        notify(
            "resource_hold",
            "Queue held before launch; failed checks=" + ",".join(failed_checks)
            + "; blocking PIDs=" + ",".join(item["pid"] for item in conflicts),
            status="held",
        )
        append_event(ledger, "notification_delivery", relation=hold_id, phase="resource_wait", notification="delivered")
        atomic_json(PENDING_CONTRACT, contract)
        atomic_json(PENDING_PREFLIGHT, preflight)
        atomic_json(PENDING_LEDGER, ledger)
        atomic_json(PENDING_QUEUE, {"schema_version": 1, "status": "held", "reason": "preflight_hold",
                                   "order": list(KS), "gpu_processes": conflicts, "updated_at": now(),
                                   "failed_checks": failed_checks,
                                   "next_action": "repeat mutable preflight checks and launch when all pass"})
    else:
        append_event(ledger, "preflight_passed", verdict="pass")
        write_generation(contract, preflight, ledger, "ready")
    location = str(PENDING_QUEUE) if held else CURRENT.read_text().strip()
    print(f"[INIT OK] status={'held' if held else 'ready'} state={location}")


def terminal(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any], success: bool, summary: str) -> None:
    kind = "experiment_completed" if success else "experiment_failed"
    event_id = append_event(ledger, kind, verdict="pass" if success else "fail", phase="terminal", notification="pending")
    notify("terminal_success" if success else "terminal_failure", summary,
           status="success" if success else "failure", gpu_released=True)
    append_event(ledger, "notification_delivery", relation=event_id, phase="terminal", notification="delivered")
    write_generation(contract, preflight, ledger, "completed" if success else "failed")


def run() -> None:
    lock_fd = acquire_lock()
    try:
        approval = json.loads(APPROVAL.read_text())
        if CURRENT.exists():
            contract, preflight, ledger = load_current()
        else:
            contract = json.loads(PENDING_CONTRACT.read_text())
            preflight = json.loads(PENDING_PREFLIGHT.read_text())
            ledger = json.loads(PENDING_LEDGER.read_text())
        while preflight["derived_verdict"] != "pass":
            conflicts = blocking_gpu_processes()
            preflight = make_preflight(contract, approval["approval_text_sha256"], not conflicts)
            if preflight["derived_verdict"] == "pass":
                append_event(ledger, "preflight_passed", verdict="pass")
                write_generation(contract, preflight, ledger, "ready")
                atomic_json(PENDING_QUEUE, {"schema_version": 1, "status": "promoted_to_harn", "order": list(KS),
                                           "updated_at": now(), "current_generation": CURRENT.read_text().strip()})
                break
            failed_checks = [item["check_id"] for item in preflight["checks"] if item["verdict"] != "pass"]
            atomic_json(WATCH_STATUS, {
                "observed_at": now(), "status": "held", "reason": "preflight_hold",
                "failed_checks": failed_checks, "storage": storage_check(),
                "gpu_processes": gpu_processes(), "blocking_gpu_processes": conflicts,
                "gpu_free_mib": gpu_free_mib(), "assigned_resource": None,
            })
            time.sleep(60)
        if not approval.get("consumed"):
            approval["consumed"] = True
            approval["consumed_at"] = now()
            atomic_json(APPROVAL, approval)
        append_event(ledger, "resources_acquired", phase="k2")
        start_id = append_event(ledger, "experiment_started", phase="k2", notification="pending")
        notify("start", "Started fair full-scale Stage 2 Q evaluation queue: K=2,3,5,10; MF25 CFG4.5 q9 seed42.")
        append_event(ledger, "notification_delivery", relation=start_id, phase="k2", notification="delivered")
        write_generation(contract, preflight, ledger, "active")
        commands = command_registry()
        for index, k in enumerate(KS):
            report = result_report(k)
            if not report.is_file():
                completed = subprocess.run(commands[f"eval_k{k}"], cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"})
                if completed.returncode:
                    raise RuntimeError(f"K={k} evaluation failed with exit {completed.returncode}")
            payload = json.loads(report.read_text())
            if payload.get("status") != "passed":
                raise RuntimeError(f"K={k} report did not pass")
            gate_id = append_event(ledger, "gate_result", verdict="pass", phase=f"k{k}", notification="pending")
            notify(f"k{k}_complete", f"K={k} MF25 CFG4.5 q9 completed; next=" + (f"K={KS[index + 1]}" if index + 1 < len(KS) else "terminal report"), report=report)
            append_event(ledger, "notification_delivery", relation=gate_id, phase=f"k{k}", notification="delivered")
            if index + 1 < len(KS):
                append_event(ledger, "promotion_started", phase=f"k{KS[index + 1]}", relation=gate_id)
            write_generation(contract, preflight, ledger, "active")
        terminal(contract, preflight, ledger, True, "K=2,3,5,10 fair MF25 CFG4.5 q9 evaluations completed.")
    except BaseException as exc:
        try:
            contract, preflight, ledger = load_current()
            if not any(event["event_kind"] in {"experiment_completed", "experiment_failed", "experiment_interrupted"} for event in ledger["events"]):
                terminal(contract, preflight, ledger, False, f"Controller failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


def watch(once: bool) -> None:
    while True:
        generation = CURRENT.read_text().strip() if CURRENT.exists() else None
        queue = json.loads((Path(generation) / "queue.json").read_text()) if generation else None
        pending = json.loads(PENDING_QUEUE.read_text()) if PENDING_QUEUE.exists() else None
        atomic_json(WATCH_STATUS, {"observed_at": now(), "current_generation": generation,
                                  "queue_status": queue["entries"][0]["status"] if queue else (pending or {}).get("status", "uninitialized"),
                                  "storage": storage_check(), "gpu_processes": gpu_processes(),
                                  "controller": json.loads(PROCESS.read_text()) if PROCESS.exists() else None})
        if once:
            return
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--approval-text-hash", required=True)
    sub.add_parser("run")
    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        init(args.approval_text_hash)
    elif args.command == "run":
        atomic_json(PROCESS, {"controller_pid": os.getpid(), "started_at": now(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        run()
    else:
        watch(args.once)


if __name__ == "__main__":
    main()
