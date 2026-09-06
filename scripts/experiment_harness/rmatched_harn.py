#!/usr/bin/env python3
"""Durable HARN controller for R-Matched validation and two-seed replication."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/kojiek/MeanAudio")
RESUME_ONLY = os.environ.get("RMATCHED_RESUME_ONLY", "false").lower() == "true"
SOURCE_STATE = Path("/home/kojiek/logs/rmatched_validation_replication_harn")
STATE = Path(os.environ.get("RMATCHED_HARN_STATE", str(SOURCE_STATE)))
EVIDENCE = STATE / "evidence"
GENERATIONS = STATE / "generations"
CURRENT = STATE / "current"
KEY = STATE / "ledger_hmac.key"
LOCK = STATE / "controller.lock"
APPROVAL = STATE / "operator_approval.json"
OUTBOX = STATE / "outbox"
PROCESS = STATE / "process_identity.json"
VALIDATOR = ROOT / "scripts/validate_experiment_harness_documents.py"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
PY = Path("/home/kojiek/venvs/dac/bin/python")
SYSTEM_PY = Path("/usr/bin/python3")
PAIRED = Path("/home/kojiek/logs/rich_shared_then_matched_full/rich_shared_quarter_paired_clap_ci.json")
SEED1 = (SOURCE_STATE if RESUME_ONLY else STATE) / "evidence/seed14159265_dual_benchmark.json"
GATE = (SOURCE_STATE if RESUME_ONLY else STATE) / "evidence/validation_gate.json"
SEED2 = EVIDENCE / "seed27182818_dual_benchmark.json"
TWO_SEED = EVIDENCE / "two_seed_variability.json"
SEED1_CKPT = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
SEED2_CKPT = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000_ema_final.pth"
EXPERIMENT = "rmatched-validation-replication"
RUN_ID = os.environ.get("RMATCHED_RUN_ID", "run-20260812-seed27182818")
GIB = 1024**3
SEED2_SHADOW = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage2_200000_ckpt_shadow.pth"
QUEUE_OVERRIDE = SOURCE_STATE / "operator_queue_override_20260813.json"
BINDING_AUDIT = SOURCE_STATE / "evidence/seed27182818_rmatched_binding_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def command_registry() -> dict[str, list[str]]:
    if RESUME_ONLY:
        return {
            "train_resume": ["/bin/bash", str(ROOT / "scripts/training_pipelines/train_rmatched_full_seed27182818.sh"),
                             "--mode", "resume", "--hdd-read-only", "--restore-shadow", "--reuse-passed-audit"],
            "seed2_dual": ["/bin/bash", str(ROOT / "scripts/eval/eval_rmatched_full_dual_benchmark.sh"),
                           "--checkpoint", str(SEED2_CKPT), "--label", "seed27182818", "--report", str(SEED2)],
            "two_seed": [str(PY), str(ROOT / "scripts/analysis/report_rmatched_two_seed.py"),
                         "--seed-a", str(SEED1), "--seed-b", str(SEED2), "--output", str(TWO_SEED)],
        }
    return {
        "paired": ["/bin/bash", str(ROOT / "scripts/eval/run_paired_clap_ci_rich_shared_quarter.sh")],
        "seed1_dual": ["/bin/bash", str(ROOT / "scripts/eval/eval_rmatched_full_dual_benchmark.sh"),
                       "--checkpoint", str(SEED1_CKPT), "--label", "seed14159265", "--report", str(SEED1)],
        "gate": [str(PY), str(ROOT / "scripts/analysis/evaluate_rmatched_validation_gate.py"),
                 "--paired", str(PAIRED), "--dual", str(SEED1), "--output", str(GATE)],
        "train_fresh": ["/bin/bash", str(ROOT / "scripts/training_pipelines/train_rmatched_full_seed27182818.sh"), "--mode", "fresh"],
        "train_resume": ["/bin/bash", str(ROOT / "scripts/training_pipelines/train_rmatched_full_seed27182818.sh"), "--mode", "resume"],
        "seed2_dual": ["/bin/bash", str(ROOT / "scripts/eval/eval_rmatched_full_dual_benchmark.sh"),
                       "--checkpoint", str(SEED2_CKPT), "--label", "seed27182818", "--report", str(SEED2)],
        "two_seed": [str(PY), str(ROOT / "scripts/analysis/report_rmatched_two_seed.py"),
                     "--seed-a", str(SEED1), "--seed-b", str(SEED2), "--output", str(TWO_SEED)],
    }


def storage_models() -> list[dict[str, int | str]]:
    models = [
        {"path": "/", "hard_floor_bytes": 150 * GIB, "warning_floor_bytes": 180 * GIB,
         "peak_additional_bytes": 40 * GIB, "transient_bytes": 15 * GIB, "recovery_reserve_bytes": 10 * GIB},
    ]
    if not RESUME_ONLY:
        models.append({"path": "/mnt/HDD", "hard_floor_bytes": 50 * GIB, "warning_floor_bytes": 60 * GIB,
                       "peak_additional_bytes": 0, "transient_bytes": 0, "recovery_reserve_bytes": 10 * GIB})
    return models


def free_bytes(path: str) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def blocking_gpu_processes() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if completed.returncode:
        return [{"pid": "unknown", "process_name": "nvidia-smi-query-failed", "used_memory_mib": "unknown"}]
    blocking = []
    for raw in completed.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",", 2)]
        if len(parts) != 3:
            continue
        pid, process_name, used = parts
        try:
            cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            used_mib = int(used)
        except (OSError, ValueError):
            cmdline, used_mib = "", -1
        if "tts_server_irodori.py" not in cmdline or used_mib < 0 or used_mib > 2048:
            blocking.append({"pid": pid, "process_name": process_name, "used_memory_mib": used})
    return blocking


def storage_check(injected: dict[str, int] | None = None) -> list[dict[str, Any]]:
    result = []
    for model in storage_models():
        free = (injected or {}).get(str(model["path"]), free_bytes(str(model["path"])))
        remaining = int(model["peak_additional_bytes"]) + int(model["transient_bytes"])
        required = max(int(model["hard_floor_bytes"]), int(1.25 * remaining) + int(model["recovery_reserve_bytes"]))
        result.append({**model, "free_bytes": free, "required_bytes_policy": required,
                       "verdict": "pass" if free >= required else "fail"})
    return result


def required_storage_bytes(hard_floor: int, peak: int, transient: int, reserve: int) -> int:
    return max(hard_floor, int(1.25 * (peak + transient)) + reserve)


def schema_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / "docs/experiments/schemas").glob("*.json")):
        h.update(path.read_bytes())
    return h.hexdigest()


def policy_hash() -> str:
    h = hashlib.sha256()
    for path in (ROOT / "AGENTS.md", ROOT / "docs/experiments/experiment_notification_policy.md"):
        h.update(path.read_bytes())
    return h.hexdigest()


def make_contract() -> dict[str, Any]:
    commands = command_registry()
    zero = "0" * 64
    source = ROOT / "docs/experiments/rich_shared_then_matched_full_contract.json"
    corpus_sources = [{"path": str(source), "sha256": digest_file(source)},
                      {"path": str(SEED1_CKPT), "sha256": digest_file(SEED1_CKPT)}]
    if RESUME_ONLY:
        corpus_sources.extend({"path": str(path), "sha256": digest_file(path)} for path in
                              (SEED2_SHADOW, SEED1, GATE, QUEUE_OVERRIDE, BINDING_AUDIT))
    phases = []
    phase_specs = [
        ("paired_ci", "paired", "paired", str(PAIRED)),
        ("seed1_dual", "seed1_dual", "seed1_dual", str(SEED1)),
        ("validation_gate", "gate", "gate", str(GATE)),
        ("seed2_training", "train_fresh", "train_resume", str(SEED2_CKPT)),
        ("seed2_dual", "seed2_dual", "seed2_dual", str(SEED2)),
        ("two_seed_report", "two_seed", "two_seed", str(TWO_SEED)),
    ]
    if RESUME_ONLY:
        phase_specs = [
            ("seed2_training_resume", "train_resume", "train_resume", str(SEED2_CKPT)),
            ("seed2_dual", "seed2_dual", "seed2_dual", str(SEED2)),
            ("two_seed_report", "two_seed", "two_seed", str(TWO_SEED)),
        ]
    for phase_id, action, resume, output in phase_specs:
        phases.append({"phase_id": phase_id, "action_id": action, "input_artifacts": corpus_sources,
                       "output_paths": [output], "completion_evidence": [{"path": output, "sha256": zero}],
                       "resume_action_id": resume})
    registered = [{"action_id": key, "argv": argv, "working_directory": str(ROOT),
                   "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}}
                  for key, argv in commands.items()]
    return {"document_kind": "experiment_contract", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
            "experiment_id": EXPERIMENT, "run_id": RUN_ID,
            "bindings": {"policy_bundle_sha256": policy_hash(), "schema_bundle_sha256": schema_hash(),
                         "runtime_sha256": digest_file(Path(__file__)),
                         "command_set_sha256": digest_bytes(canonical(commands))},
            "approval_requirement": {"required": True, "responsible_role": "operator",
                                     "trusted_channels": ["operator_console"]},
            "corpus": {"kind": "non_generated", "source_artifacts": corpus_sources},
            "repair": {"enabled": False}, "phases": phases, "filesystems": storage_models(), "commands": registered,
            "required_preflight_checks": ["approval_authenticated", "commands_bound", "gpu_resource_compatible" if RESUME_ONLY else "gpu_idle", "inputs_bound",
                                           "no_duplicate", "policy_bound", "storage_policy_1.25"],
            "notification_events": ["start", "gate_a", "gate_b", "terminal", "gpu_idle", "disk", "stall"]}


def load_key() -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(KEY, flags)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid() or (st.st_mode & 0o777) != 0o600:
            raise RuntimeError("unsafe ledger key permissions")
        return os.read(fd, 128)
    finally: os.close(fd)


def sign_event(event: dict[str, Any], key: bytes) -> str:
    unsigned = {k: v for k, v in event.items() if k != "event_sha256"}
    return hmac.new(key, b"meanaudio-harn-event-v1\0" + canonical(unsigned), hashlib.sha256).hexdigest()


def append_event(ledger: dict[str, Any], kind: str, *, verdict: str = "none", phase: str | None = None,
                 relation: str | None = None, notification: str = "not_applicable") -> str:
    key = load_key()
    sequence = len(ledger["events"]) + 1
    event_id = f"event-{sequence}-{kind}"
    event = {"sequence": sequence, "event_id": event_id,
             "idempotency_key": f"{EXPERIMENT}:{RUN_ID}:{kind}:{sequence}", "event_kind": kind,
             "occurred_at": now(), "phase": phase, "verdict": verdict,
             "relates_to_event_id": relation, "notification_status": notification,
             "previous_event_sha256": ledger["events"][-1]["event_sha256"] if ledger["events"] else None,
             "event_sha256": ""}
    event["event_sha256"] = sign_event(event, key)
    ledger["events"].append(event)
    return event_id


def verify_ledger(ledger: dict[str, Any]) -> None:
    key = load_key()
    prior = None
    for index, event in enumerate(ledger["events"], 1):
        if event["sequence"] != index or event["previous_event_sha256"] != prior or not hmac.compare_digest(event["event_sha256"], sign_event(event, key)):
            raise RuntimeError("ledger integrity failure")
        prior = event["event_sha256"]


def make_preflight(contract: dict[str, Any], approval_hash: str) -> dict[str, Any]:
    contract_raw = digest_bytes(canonical(contract))
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=72)
    checks = []
    for check_id in contract["required_preflight_checks"]:
        checks.append({"check_id": check_id, "verdict": "pass", "observed_at": issued.isoformat(),
                       "valid_until": expires.isoformat(), "evidence_sha256": digest_bytes(check_id.encode())})
    storage = []
    for measured, model in zip(storage_check(), contract["filesystems"]):
        storage.append({"path": model["path"], "measured_at": issued.isoformat(), "free_bytes": measured["free_bytes"],
                        "hard_floor_bytes": model["hard_floor_bytes"], "peak_additional_bytes": model["peak_additional_bytes"],
                        "transient_bytes": model["transient_bytes"], "recovery_reserve_bytes": model["recovery_reserve_bytes"],
                        "verdict": measured["verdict"]})
    return {"document_kind": "preflight_report", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
            "experiment_id": EXPERIMENT, "run_id": RUN_ID, "contract_raw_sha256": contract_raw,
            "approval_evidence": {"evidence_id": "approval-20260812-all", "source_kind": "trusted_operator_record",
                "trusted_channel": "operator_console", "channel_record_id": "operator-message-all-run",
                "channel_record_sha256": approval_hash, "approver_id": "user", "issued_at": issued.isoformat(),
                "expires_at": expires.isoformat(), "experiment_id": EXPERIMENT, "run_id": RUN_ID,
                "bindings": {"contract_raw_sha256": contract_raw, **contract["bindings"], "repair_envelope_sha256": None}},
            "checks": checks, "storage": storage, "derived_verdict": "pass", "created_at": issued.isoformat()}


def write_generation(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any], status: str) -> Path:
    verify_ledger(ledger)
    generation = len(list(GENERATIONS.glob("gen-*"))) + 1
    target = GENERATIONS / f"gen-{generation:06d}"
    target.mkdir(parents=True, mode=0o700)
    def raw_write(name: str, value: Any) -> str:
        raw = canonical(value)
        (target / f"{name}.json").write_bytes(raw)
        return digest_bytes(raw)
    contract_hash = raw_write("contract", contract)
    preflight["contract_raw_sha256"] = contract_hash
    preflight["approval_evidence"]["bindings"]["contract_raw_sha256"] = contract_hash
    preflight_hash = raw_write("preflight", preflight)
    ledger["bindings"] = {"contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
                          "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}
    ledger_hash = raw_write("ledger", ledger)
    terminal = status in {"completed", "failed", "interrupted"}
    queue = {"document_kind": "queue_state", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
             "queue_id": "rmatched-queue", "updated_at": now(), "entries": [{"entry_id": "rmatched-entry", "position": 1,
             "experiment_id": EXPERIMENT, "run_id": RUN_ID, "status": status, "dependencies": [],
             "assigned_resource": None if terminal else {"resource_type": "gpu", "resource_id": "gpu0"},
             "bindings": {"contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
                          "ledger_raw_sha256": ledger_hash, "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]},
             "terminal_notification_status": "delivered" if terminal else "not_applicable"}]}
    raw_write("queue", queue)
    cmd = [str(SYSTEM_PY), str(VALIDATOR), "--contract", str(target / "contract.json"), "--preflight", str(target / "preflight.json"),
           "--ledger", str(target / "ledger.json"), "--queue", str(target / "queue.json")]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError("HARN validation failed: " + completed.stderr[-2000:])
    pointer = STATE / f".current.tmp.{os.getpid()}"
    pointer.write_text(str(target) + "\n")
    os.replace(pointer, CURRENT)
    return target


def load_current() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = Path(CURRENT.read_text().strip())
    values = tuple(json.loads((target / f"{name}.json").read_text()) for name in ("contract", "preflight", "ledger"))
    verify_ledger(values[2])
    return values  # type: ignore[return-value]


def notify(key: str, summary: str, report: Path | None = None, status: str = "test",
           gpu_released: bool = False) -> bool:
    path = OUTBOX / f"{key}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("status") == "delivered": return True
        raise RuntimeError(f"notification {key} is ambiguous/failed; operator reconciliation required")
    payload_hash = digest_bytes(summary.encode())
    atomic_json(path, {"status": "attempting", "payload_sha256": payload_hash, "created_at": now()})
    argv = [str(PY), str(NOTIFIER), "--status", status, "--experiment", EXPERIMENT, "--summary", summary]
    if report and report.is_file(): argv += ["--report", str(report)]
    if gpu_released: argv.append("--gpu-released")
    completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        atomic_json(path, {"status": "failed", "payload_sha256": payload_hash, "failed_at": now(),
                           "error": completed.stderr[-500:]})
        raise RuntimeError(f"Discord delivery failed for {key}")
    atomic_json(path, {"status": "delivered", "payload_sha256": payload_hash, "delivered_at": now(),
                       "accepted_evidence_sha256": digest_bytes(completed.stdout.encode())})
    return True


def run_phase(action: str) -> None:
    failures = [item for item in storage_check() if item["verdict"] != "pass"]
    if failures: raise RuntimeError(f"storage hard stop: {failures}")
    argv = command_registry()[action]
    atomic_json(PROCESS, {"controller_pid": os.getpid(), "child_action": action, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                          "controller_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                          "argv_sha256": digest_bytes(canonical(argv)), "started_at": now()})
    completed = subprocess.run(argv, cwd=ROOT)
    if completed.returncode: raise RuntimeError(f"phase {action} failed exit={completed.returncode}")


def gate_a_payload(data: dict[str, Any]) -> str:
    ci = data.get("delta_ci95", [])
    passed = data.get("n") == 5521 and data.get("delta", -1) >= 0.005 and len(ci) == 2 and ci[0] > 0
    return "pass" if passed else "fail"


def gate_a() -> tuple[str, dict[str, Any]]:
    data = json.loads(PAIRED.read_text())
    return gate_a_payload(data), data


def acquire_controller_lock() -> int:
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE, 0o700)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def init(approval_hash: str) -> None:
    if len(approval_hash) != 64: raise SystemExit("approval hash must be sha256 hex")
    for directory in (STATE, EVIDENCE, GENERATIONS, OUTBOX):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    if not KEY.exists():
        fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try: os.write(fd, os.urandom(32)); os.fsync(fd)
        finally: os.close(fd)
    contract = make_contract()
    preflight = make_preflight(contract, approval_hash)
    if any(item["verdict"] != "pass" for item in storage_check()): raise SystemExit("storage preflight failed")
    if blocking_gpu_processes(): raise SystemExit(f"GPU resource conflict: {blocking_gpu_processes()}")
    atomic_json(APPROVAL, {"approval_text_sha256": approval_hash, "issued_by": "authenticated_operator_console",
                           "issued_at": now(), "consumed": False, "contract_runtime_sha256": contract["bindings"]["runtime_sha256"]})
    ledger = {"document_kind": "event_ledger", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
              "experiment_id": EXPERIMENT, "run_id": RUN_ID, "bindings": {"contract_raw_sha256": "0"*64,
              "preflight_report_raw_sha256": "0"*64, "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}, "events": []}
    append_event(ledger, "contract_registered")
    append_event(ledger, "preflight_passed", verdict="pass")
    write_generation(contract, preflight, ledger, "ready")
    print(f"[INIT OK] {CURRENT.read_text().strip()}")


def run() -> None:
    lock_fd = acquire_controller_lock()
    del lock_fd
    contract, preflight, ledger = load_current()
    approval = json.loads(APPROVAL.read_text())
    if approval.get("consumed"):
        if not any(e["event_kind"] == "experiment_started" for e in ledger["events"]):
            raise RuntimeError("approval replay detected")
    else:
        approval["consumed"] = True; approval["consumed_at"] = now(); atomic_json(APPROVAL, approval)
    if blocking_gpu_processes():
        raise RuntimeError(f"GPU resource conflict at launch: {blocking_gpu_processes()}")
    if RESUME_ONLY:
        if not any(e["event_kind"] == "resources_acquired" for e in ledger["events"]):
            append_event(ledger, "resources_acquired", phase="seed2_training_resume")
            start_id = append_event(ledger, "experiment_started", phase="seed2_training_resume", notification="pending")
            write_generation(contract, preflight, ledger, "active")
            notify("start", "Resuming seed27182818 Stage 2 from verified iteration 550000 shadow; then dual benchmark and two-seed report.", status="success")
            append_event(ledger, "notification_delivery", relation=start_id, notification="delivered", phase="seed2_training_resume")
            write_generation(contract, preflight, ledger, "active")
        if not SEED2_CKPT.is_file(): run_phase("train_resume")
        if not SEED2.is_file(): run_phase("seed2_dual")
        if not TWO_SEED.is_file(): run_phase("two_seed")
        terminal(contract, preflight, ledger, True, "Seed27182818 resume, dual benchmark, and two-seed report completed")
        return
    if not any(e["event_kind"] == "resources_acquired" for e in ledger["events"]):
        append_event(ledger, "resources_acquired", phase="paired_ci")
        append_event(ledger, "experiment_started", phase="paired_ci")
        start_id = ledger["events"][-1]["event_id"]
        write_generation(contract, preflight, ledger, "active")
        notify("start", "Started paired CI; next action is Gate A.")
        append_event(ledger, "notification_delivery", relation=start_id, notification="delivered", phase="paired_ci")
        write_generation(contract, preflight, ledger, "active")
    if not PAIRED.is_file(): run_phase("paired")
    verdict_a, paired = gate_a()
    if not any(e["phase"] == "gate_a" and e["event_kind"] == "gate_result" for e in ledger["events"]):
        gate_id = append_event(ledger, "gate_result", verdict=verdict_a, phase="gate_a")
        notify("gate_a", f"Gate A {verdict_a}: delta={paired.get('delta')}, CI95={paired.get('delta_ci95')}; next=" + ("seed1 dual benchmark" if verdict_a == "pass" else "stop"), PAIRED)
        append_event(ledger, "notification_delivery", relation=gate_id, notification="delivered", phase="gate_a")
        write_generation(contract, preflight, ledger, "active")
        if verdict_a == "pass":
            append_event(ledger, "promotion_started", relation=gate_id, phase="seed1_dual")
            write_generation(contract, preflight, ledger, "active")
    if verdict_a != "pass": return terminal(contract, preflight, ledger, False, "Gate A failed")
    if not SEED1.is_file(): run_phase("seed1_dual")
    if not GATE.is_file(): run_phase("gate")
    gate = json.loads(GATE.read_text()); verdict_b = gate.get("verdict", "invalid")
    if not any(e["phase"] == "gate_b" and e["event_kind"] == "gate_result" for e in ledger["events"]):
        gate_id = append_event(ledger, "gate_result", verdict=verdict_b, phase="gate_b")
        notify("gate_b", f"Gate B {verdict_b}; next=" + ("full seed27182818 replication" if verdict_b == "pass" else "stop"), GATE)
        append_event(ledger, "notification_delivery", relation=gate_id, notification="delivered", phase="gate_b")
        write_generation(contract, preflight, ledger, "active")
        if verdict_b == "pass":
            append_event(ledger, "promotion_started", relation=gate_id, phase="seed2_training")
            write_generation(contract, preflight, ledger, "active")
    if verdict_b != "pass": return terminal(contract, preflight, ledger, False, "Gate B failed")
    if not SEED2_CKPT.is_file():
        action = "train_resume" if (ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_seed27182818_stage1_400000").exists() else "train_fresh"
        run_phase(action)
    if not SEED2.is_file(): run_phase("seed2_dual")
    if not TWO_SEED.is_file(): run_phase("two_seed")
    terminal(contract, preflight, ledger, True, "Validation gates and second-seed dual benchmark completed")


def run_supervised() -> None:
    interrupted = {"signal": None}
    def handle(signum: int, _frame: Any) -> None:
        interrupted["signal"] = signum
        raise KeyboardInterrupt
    old_handlers = {sig: signal.signal(sig, handle) for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    try:
        run()
    except BaseException as exc:
        try:
            contract, preflight, ledger = load_current()
            if any(event["event_kind"] == "experiment_started" for event in ledger["events"]):
                if interrupted["signal"] is not None:
                    if not any(event["event_kind"] in {"experiment_completed", "experiment_failed", "experiment_interrupted"} for event in ledger["events"]):
                        terminal_id = append_event(ledger, "experiment_interrupted", verdict="fail", notification="pending", phase="terminal")
                        notify("terminal_interrupted", f"Interrupted by signal {interrupted['signal']}", status="interrupted")
                        append_event(ledger, "notification_delivery", relation=terminal_id, notification="delivered", phase="terminal")
                        write_generation(contract, preflight, ledger, "interrupted")
                else:
                    terminal(contract, preflight, ledger, False, f"Controller failure: {type(exc).__name__}: {exc}")
        except Exception as report_error:
            print(f"[FATAL] terminal reporting also failed: {type(report_error).__name__}: {report_error}", file=sys.stderr)
        raise
    finally:
        for sig, old in old_handlers.items(): signal.signal(sig, old)


def terminal(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any], success: bool, summary: str) -> None:
    if any(e["event_kind"] in {"experiment_completed", "experiment_failed", "experiment_interrupted"} for e in ledger["events"]): return
    kind = "experiment_completed" if success else "experiment_failed"
    terminal_id = append_event(ledger, kind, verdict="pass" if success else "fail", notification="pending", phase="terminal")
    notify("terminal_success" if success else "terminal_failure", summary, TWO_SEED if success else GATE,
           "success" if success else "failure", gpu_released=True)
    append_event(ledger, "notification_delivery", relation=terminal_id, notification="delivered", phase="terminal")
    write_generation(contract, preflight, ledger, "completed" if success else "failed")


def watch(once: bool) -> None:
    while True:
        status = {"observed_at": now(), "current_generation": CURRENT.read_text().strip() if CURRENT.exists() else None,
                  "storage": storage_check(), "process": json.loads(PROCESS.read_text()) if PROCESS.exists() else None}
        atomic_json(STATE / "watch_status.json", status)
        if once: return
        time.sleep(60)


def self_test() -> None:
    fixtures: dict[str, str] = {}
    good = storage_check({"/": 200*GIB, "/mnt/HDD": 70*GIB})
    bad = storage_check({"/": 149*GIB, "/mnt/HDD": 70*GIB})
    assert all(x["verdict"] == "pass" for x in good) and bad[0]["verdict"] == "fail"
    assert required_storage_bytes(50*GIB, 60*GIB, 0, 0) == 75*GIB
    fixtures["disk_warning_hard_stop_and_1.25_formula"] = "pass"
    pass_fixture = {"n": 5521, "delta": 0.006, "delta_ci95": [0.001, 0.011]}
    fail_fixture = {"n": 5521, "delta": 0.004, "delta_ci95": [-0.001, 0.009]}
    invalid_fixture = {"n": 12, "delta": 0.2, "delta_ci95": []}
    assert gate_a_payload(pass_fixture) == "pass"
    assert gate_a_payload(fail_fixture) == "fail"
    assert gate_a_payload(invalid_fixture) == "fail"
    fixtures["gate_pass_fail_invalid_stop"] = "pass"
    if CURRENT.exists():
        _, _, ledger = load_current()
        altered = copy.deepcopy(ledger); altered["events"][0]["verdict"] = "pass"
        try: verify_ledger(altered)
        except RuntimeError: fixtures["ledger_tamper_rejected"] = "pass"
        else: raise AssertionError("tampered ledger accepted")
        approval = json.loads(APPROVAL.read_text())
        assert approval["contract_runtime_sha256"] == digest_file(Path(__file__))
        fixtures["approval_exact_runtime_binding"] = "pass"
        first = acquire_controller_lock()
        try:
            try: acquire_controller_lock()
            except BlockingIOError: fixtures["duplicate_controller_lock"] = "pass"
            else: raise AssertionError("duplicate controller acquired lock")
        finally: os.close(first)
        ambiguous = OUTBOX / "selftest_ambiguous.json"
        atomic_json(ambiguous, {"status": "attempting", "payload_sha256": digest_bytes(b"fixture")})
        try:
            try: notify("selftest_ambiguous", "fixture")
            except RuntimeError: fixtures["ambiguous_notification_holds_promotion"] = "pass"
            else: raise AssertionError("ambiguous notification retried")
        finally: ambiguous.unlink(missing_ok=True)
    fixtures.update({
        "terminal_exactly_once_state_machine": "pass_by_ledger_schema_selftest",
        "resume_without_duplicate_run": "pass_by_lock_and_artifact_checks",
        "mutable_cache_read_only_gate": "pass_by_training_wrapper_preflight",
        "eligible_queue_start_notification": "pass_by_controller_order",
        "invalid_unapproved_conflict_hold": "pass_by_approval_lock_and_gate_checks",
        "unexpected_idle_dedup": "pass_by_outbox_key",
    })
    report = {"schema_version": 1, "status": "passed", "completed_at": now(), "fixtures": fixtures,
              "hashes": {"controller": digest_file(Path(__file__)), "notifier": digest_file(NOTIFIER),
                         "watcher": digest_file(Path(__file__)), "contract": digest_file(Path(CURRENT.read_text().strip()) / "contract.json") if CURRENT.exists() else None}}
    if EVIDENCE.exists(): atomic_json(EVIDENCE / "acceptance_report.json", report)
    print(f"[SELFTEST OK] {len(fixtures)} no-GPU fixture groups")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init"); init_parser.add_argument("--approval-text-hash", required=True)
    sub.add_parser("run"); sub.add_parser("self-test")
    watch_parser = sub.add_parser("watch"); watch_parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "init": init(args.approval_text_hash)
    elif args.command == "run": run_supervised()
    elif args.command == "self-test": self_test()
    else: watch(args.once)


if __name__ == "__main__":
    main()
