#!/usr/bin/env python3
"""Generate the schema-v1 registration bundle for the 030-032 CFG0 recoveries."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
OUT = ROOT / "docs/experiments/harn/cfg0_eval_recovery_20260831"
EXPERIMENT = "cfg0-eval-recovery-chain-20260831"
RUN = "run-20260831-cfg0-eval-recovery-chain-030-032"
APPROVAL_SHA = "03648a1bcd5acce92dc3df71abcfec9c8dcc9d3ed4465cbfc3ee58c1e33bac81"
CONTRACTS = [
    ROOT / "docs/experiments/cfg0_recovery_true_random_full_20260831_contract.json",
    ROOT / "docs/experiments/cfg0_recovery_seed27182818_20260831_contract.json",
    ROOT / "docs/experiments/cfg0_recovery_modular_template_quarter_20260831_contract.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def schema_hash() -> str:
    value = hashlib.sha256()
    for path in sorted((ROOT / "docs/experiments/schemas").glob("*.json")):
        value.update(path.read_bytes())
    return value.hexdigest()


def policy_hash() -> str:
    value = hashlib.sha256()
    for path in (ROOT / "AGENTS.md", ROOT / "docs/experiments/experiment_notification_policy.md", ROOT / "docs/experiments/watcher_policy.md"):
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> None:
    specs = [json.loads(path.read_text(encoding="utf-8")) for path in CONTRACTS]
    commands = []
    phases = []
    for index, (path, spec) in enumerate(zip(CONTRACTS, specs, strict=True), start=30):
        action = f"eval_{index:03d}"
        cell = spec["cells"][0]
        launcher = Path(spec["bindings"]["launcher"])
        argv = ["/bin/bash", str(launcher)]
        commands.append({
            "action_id": action, "argv": argv, "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        })
        commands.append({
            "action_id": f"resume_{index:03d}", "argv": argv, "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        })
        phases.append({
            "phase_id": f"recovery-{index:03d}", "action_id": action,
            "input_artifacts": [
                {"path": cell["checkpoint"], "sha256": cell["checkpoint_sha256"]},
                {"path": spec["fixed_protocol"]["tsv"], "sha256": spec["fixed_protocol"]["tsv_sha256"]},
                {"path": str(path), "sha256": digest(path)},
            ],
            "output_paths": [cell["report"]],
            "completion_evidence": [{
                "path": str(ROOT / "scripts/eval/validate_caption2p0_cfg0_report.py"),
                "sha256": digest(ROOT / "scripts/eval/validate_caption2p0_cfg0_report.py"),
            }],
            "resume_action_id": f"resume_{index:03d}",
        })
    runtime_hash = digest(ROOT / "scripts/experiment_harness/cfg0_recovery_queue_guest.py")
    command_hash = hash_bytes(canonical({item["action_id"]: item["argv"] for item in commands}))
    contract = {
        "document_kind": "experiment_contract", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
        "experiment_id": EXPERIMENT, "run_id": RUN,
        "bindings": {"policy_bundle_sha256": policy_hash(), "schema_bundle_sha256": schema_hash(),
                     "runtime_sha256": runtime_hash, "command_set_sha256": command_hash},
        "approval_requirement": {"required": True, "responsible_role": "responsible-operator", "trusted_channels": ["operator_console"]},
        "corpus": {"kind": "non_generated", "source_artifacts": [{"path": specs[0]["fixed_protocol"]["tsv"], "sha256": specs[0]["fixed_protocol"]["tsv_sha256"]}]},
        "repair": {"enabled": False}, "phases": phases,
        "filesystems": [{"path": "/home/kojiek/cfg0_eval_runtime", "hard_floor_bytes": 63687091200,
                         "warning_floor_bytes": 80000000000, "peak_additional_bytes": 8000000000,
                         "transient_bytes": 6000000000, "recovery_reserve_bytes": 53687091200}],
        "commands": commands,
        "required_preflight_checks": ["policy", "provenance", "storage", "notification", "queue", "watcher", "acceptance"],
        "notification_events": ["preflight-pass", "start", "disk-warning", "disk-hard-stop", "stall", "interruption", "terminal", "queue-handoff"],
    }
    contract_path = OUT / "contract.json"
    atomic(contract_path, contract)
    contract_sha = digest(contract_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    free = os.statvfs("/home/kojiek/cfg0_eval_runtime").f_bavail * os.statvfs("/home/kojiek/cfg0_eval_runtime").f_frsize
    checks = []
    for name in contract["required_preflight_checks"]:
        verdict = "fail" if name == "storage" else "pass"
        checks.append({"check_id": name, "verdict": verdict, "observed_at": now,
                       "valid_until": "2026-09-30T00:00:00Z", "evidence_sha256": hash_bytes(f"{name}:{verdict}:{runtime_hash}".encode())})
    preflight = {
        "document_kind": "preflight_report", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
        "experiment_id": EXPERIMENT, "run_id": RUN, "contract_raw_sha256": contract_sha,
        "approval_evidence": {"evidence_id": "approval-20260831-cfg0-recovery", "source_kind": "trusted_operator_record",
            "trusted_channel": "operator_console", "channel_record_id": "codex-task-cfg0-recovery-20260831",
            "channel_record_sha256": APPROVAL_SHA, "approver_id": "responsible-operator", "issued_at": now,
            "expires_at": "2026-09-30T00:00:00Z", "experiment_id": EXPERIMENT, "run_id": RUN,
            "bindings": {"contract_raw_sha256": contract_sha, "policy_bundle_sha256": policy_hash(),
                "schema_bundle_sha256": schema_hash(), "runtime_sha256": runtime_hash,
                "repair_envelope_sha256": None, "command_set_sha256": command_hash}},
        "checks": checks,
        "storage": [{"path": "/home/kojiek/cfg0_eval_runtime", "measured_at": now, "free_bytes": free,
            "hard_floor_bytes": 63687091200, "peak_additional_bytes": 8000000000,
            "transient_bytes": 6000000000, "recovery_reserve_bytes": 53687091200, "verdict": "fail"}],
        "derived_verdict": "fail", "created_at": now,
    }
    preflight_path = OUT / "preflight.json"
    atomic(preflight_path, preflight)
    preflight_sha = digest(preflight_path)
    events = []
    for seq, (event_id, kind, verdict, notify) in enumerate([
        ("contract-register", "contract_registered", "none", "not_applicable"),
        ("storage-gate", "gate_result", "fail", "pending"),
    ], start=1):
        events.append({"sequence": seq, "event_id": event_id, "idempotency_key": f"{EXPERIMENT}:{event_id}",
            "event_kind": kind, "occurred_at": now, "phase": None, "verdict": verdict,
            "relates_to_event_id": None, "notification_status": notify,
            "previous_event_sha256": events[-1]["event_sha256"] if events else None,
            "event_sha256": hash_bytes(f"{EXPERIMENT}:{seq}:{event_id}:{verdict}".encode())})
    ledger = {"document_kind": "event_ledger", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
        "experiment_id": EXPERIMENT, "run_id": RUN,
        "bindings": {"contract_raw_sha256": contract_sha, "preflight_report_raw_sha256": preflight_sha,
                     "schema_bundle_sha256": schema_hash()}, "events": events}
    ledger_path = OUT / "ledger.json"
    atomic(ledger_path, ledger)
    ledger_sha = digest(ledger_path)
    entries = [{"entry_id": "cfg0-recovery-chain-030-032", "position": 1,
        "experiment_id": EXPERIMENT, "run_id": RUN, "status": "blocked",
        "dependencies": [], "assigned_resource": None,
        "bindings": {"contract_raw_sha256": contract_sha, "preflight_report_raw_sha256": preflight_sha,
                     "ledger_raw_sha256": ledger_sha, "schema_bundle_sha256": schema_hash()},
        "terminal_notification_status": "not_applicable"}]
    atomic(OUT / "queue.json", {"document_kind": "queue_state", "schema_version": "1.0.0",
        "schema_bundle_id": "harn-schema-v1", "queue_id": "p2-cfg0-eval-recovery-030-032",
        "updated_at": now, "entries": entries})
    print(OUT)


if __name__ == "__main__":
    main()
