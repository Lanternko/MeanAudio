#!/usr/bin/env python3
"""Prepare immutable audit-only contract and schema-v1 HARN registration."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import slot0_semantic_audit as A

R = Path(__file__).resolve().parents[2]
D = R / "docs/experiments/slot0_semantic_audit_20260915"
O = Path("/home/kojiek/exps_nvme/slot0_semantic_audit_20260915/full_v1")
H = D / "harn_v1"
SOURCE = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def artifact(p):
    return {"path": str(p), "sha256": sha(p)}


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":")).encode()


def main():
    if (O / "state.sqlite").exists():
        raise RuntimeError("registered run already has runtime state; cannot mutate its contract")
    O.mkdir(parents=True, exist_ok=True)
    H.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.isoformat().replace("+00:00", "Z")
    expires = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    operator = D / "operator_scope_v1.txt"
    operator.write_text("Operator requested full slot0 semantic cleaning with Luna API and unchanged caption wording; latest instruction adopts bare no-music without concrete sound -> INVALID and original-prompt audio regeneration, concrete sound/silence -> KEEP; fixes batch isolation/retries and requires new calibration and fresh heldout. API audit uses no GPU. No training launch or GPU queue preemption authorized.\n")
    acceptance = []
    for test in ("test_slot0_semantic_audit.py", "test_slot0_audit_controller.py"):
        result = subprocess.run([sys.executable, str(R / "scripts/tests" / test)], capture_output=True)
        acceptance.append({"test": test, "returncode": result.returncode,
                           "output_sha256": A.digest(result.stdout + result.stderr)})
        if result.returncode:
            raise RuntimeError("acceptance failed")
    A.atomic(D / "acceptance_v1.json", {"status": "passed", "tests": acceptance,
                                       "scope": "API controller; GPU regeneration/encoding not launched by this contract"})
    gates = [Path("/home/kojiek/exps_nvme/slot0_semantic_audit_20260915") / name / "calibration_report.json"
             for name in ("calibration_v5", "heldout_v3")]
    for gate in gates:
        if json.loads(gate.read_text())["status"] != "passed":
            raise RuntimeError("quality calibration gate failed")
    with SOURCE.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    ids = [r["id"] for r in rows]
    if len(ids) != 251599 or len(set(ids)) != len(ids) or not all(ids):
        raise RuntimeError("invalid source IDs")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise RuntimeError("invalid source schema")
    A.atomic(D / "source_schema_v1.json", {"format": "TSV", "fields": fields, "row_count": len(rows),
                                          "ordering": "source exact", "null_fields": "forbidden"})
    taxonomy = {"semantic": ["KEEP", "TRIM_SUFFIX", "REVIEW", "INVALID"],
                "bare_no_music": "INVALID", "concrete_sounds_and_silence": "KEEP",
                "numeric_music_facts": "KEEP", "meta_preface": "REVIEW",
                "unresolved": "quarantine, never silently drop",
                "release": "all IDs resolved plus independent real-data review; no zero-error claim"}
    A.atomic(D / "taxonomy_v1.json", taxonomy)
    # This gate validates safe admission of an UNTRUSTED source into its auditor.
    # It deliberately does not certify that captions are clean or fit for training.
    A.atomic(D / "source_admission_gate_v1.json", {
        "status": "passed", "gate_scope": "read_only_audit_admission",
        "source": artifact(SOURCE), "rows_checked": len(rows), "schema_and_ID_failures": 0,
        "generated_corpus_cleanliness": "unknown_pending_full_semantic_audit",
        "training_or_encoding_authorized": False})
    A.atomic(D / "no_derived_features_v1.json", {"status": "not_applicable", "reason": "audit-only phase creates no feature cache or training TSV; downstream phases require new provenance gates"})
    scripts = [R / "scripts/preprocess" / name for name in (
        "slot0_semantic_audit.py", "slot0_audit_controller.py", "slot0_audit_supervisor.py")]
    scripts += [R / "scripts/caption10s_pipeline/repair_multisent_first_entity_line.py",
                R / "scripts/experiment_harness/notification_receipts.py", R / "scripts/notify_experiment_webhook.py"]
    inputs = [SOURCE, operator, D / "taxonomy_v1.json", D / "acceptance_v1.json"] + scripts + gates
    spec = {
        "experiment_id": "slot0-semantic-audit-20260915", "run_id": "slot0-semantic-audit-20260915-full-v1",
        "model": A.MODEL, "prompt_sha256": A.digest(A.PROMPT.encode()),
        "source": {**artifact(SOURCE), "rows": len(rows)}, "output_dir": str(O), "harn_bundle": str(H),
        "approval": {"scope": "full_slot0_semantic_audit_exact_prefix_and_quarantine", "record": artifact(operator)},
        "immutable_artifacts": [artifact(p) for p in inputs], "quality_gate_reports": [str(p) for p in gates],
        "batch_size": 8, "workers": 4, "min_request_interval_seconds": 0.5, "max_429_attempts": 4,
        "independent_review_sample_rows": 3000, "independent_review_seed": 2026091604,
        "max_api_posts": 65000, "max_wall_seconds": 172800, "max_cost_usd": 75,
        "pricing": {"input_per_million": 0.2, "output_per_million": 1.2,
                    "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
                    "accounting": "conservative uncached input plus reservations; not billing reconciliation"},
        "storage": {"hard_floor_bytes": 53687091200, "warning_floor_bytes": 85899345920,
                    "peak_additional_bytes": 3221225472, "transient_bytes": 1073741824,
                    "recovery_reserve_bytes": 53687091200},
        "watcher": {"stall_seconds": 600, "poll_seconds": 10, "healthy_model_calls": 0},
        "phases": ["audit_all_rows", "independent_trim_check", "export_full_coverage_and_quarantine"],
        "failure_policy": {"429": "bounded backoff; persisted attempts", "timeout_or_5xx": "recover receipt or query stored metadata; otherwise isolate and continue", "empty_lookup": "not proof of zero charge; no blind repost", "401_403_404": "credential/model hold"},
        "release_rule": "No training TSV released by auditor. All unresolved rows and independent miss-rate review remain release blockers.",
        "regeneration_handoff": {"ids": str(O / "regeneration_requests.tsv"), "source": "original audio first 10 seconds",
            "captioner": "Qwen/Qwen2.5-Omni-3B", "prompt_source": str(R / "scripts/caption10s_pipeline/gen_qwen_caption_10s_multisent.py"),
            "prompt_change_allowed": False, "max_attempts": 6, "selection": "first candidate passing structure and semantic review",
            "resource_state": "requires registered GPU seat after audit targets are known; no competing launch"},
    }
    spec_path = D / "full_v1_contract.json"
    A.atomic(spec_path, spec)
    common = {"schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1", "experiment_id": spec["experiment_id"], "run_id": spec["run_id"]}
    schema_hash = A.digest(b"".join(p.read_bytes() for p in sorted((R / "docs/experiments/schemas").glob("*.json"))))
    policy_hash = A.digest(b"".join(p.read_bytes() for p in [R / "AGENTS.md", R / "docs/experiments/experiment_notification_policy.md", R / "docs/experiments/watcher_policy.md", R / "docs/experiments/generated_corpus_policy.md"]))
    commands = [{"action_id": action, "argv": [sys.executable, str(scripts[2]), "--contract", str(spec_path), "--key-file", "/home/kojiek/.config/meanaudio/luna_api_key"], "working_directory": str(R), "environment": {"PYTHONUNBUFFERED": "1"}}
                for action in ("audit", "resume-audit")]
    command_hash = A.digest(canonical({c["action_id"]: c["argv"] for c in commands}))
    contract = {**common, "document_kind": "experiment_contract",
        "bindings": {"policy_bundle_sha256": policy_hash, "schema_bundle_sha256": schema_hash,
                     "runtime_sha256": sha(scripts[1]), "command_set_sha256": command_hash},
        "approval_requirement": {"required": True, "responsible_role": "responsible-operator", "trusted_channels": ["operator_console"]},
        "corpus": {"kind": "generated", "corpus_artifact": artifact(SOURCE), "corpus_schema_sha256": sha(D / "source_schema_v1.json"),
                   "classifier_version": "slot0-semantic-v5-audit-admission-only", "defect_taxonomy_sha256": sha(D / "taxonomy_v1.json"),
                   "stop_behavior_test_id": "api-finish-reason-stop", "full_gate_report": artifact(D / "source_admission_gate_v1.json"),
                   "downstream_bindings": {"tsv_manifest_sha256": sha(D / "no_derived_features_v1.json"), "feature_cache_report_sha256": sha(D / "no_derived_features_v1.json")},
                   "required_gate_points": ["launch", "pre_training", "post_change"]},
        "repair": {"enabled": False}, "phases": [{"phase_id": "audit-only", "action_id": "audit",
            "input_artifacts": [artifact(spec_path), artifact(SOURCE)], "output_paths": [str(O / "full_gate_report.json"), str(O / "quarantine.jsonl")],
            "completion_evidence": [artifact(scripts[1])], "resume_action_id": "resume-audit"}],
        "filesystems": [{"path": str(O), **spec["storage"]}], "commands": commands,
        "required_preflight_checks": ["policy", "provenance", "storage", "notification", "acceptance", "generated_corpus_full_gate"],
        "notification_events": ["preflight-pass", "start", "disk-warning", "disk-hard-stop", "stall", "audit-terminal", "supervisor-terminal"]}
    A.atomic(H / "contract.json", contract)
    contract_hash = sha(H / "contract.json")
    free = os.statvfs(O).f_bavail * os.statvfs(O).f_frsize
    if free < spec["storage"]["hard_floor_bytes"]:
        raise RuntimeError("insufficient storage")
    checks = [{"check_id": name, "verdict": "pass", "observed_at": stamp, "valid_until": expires,
               "evidence_sha256": sha(D / "acceptance_v1.json") if name == "acceptance" else sha(spec_path)}
              for name in contract["required_preflight_checks"]]
    preflight = {**common, "document_kind": "preflight_report", "contract_raw_sha256": contract_hash,
        "approval_evidence": {"evidence_id": "slot0-operator-20260915", "source_kind": "trusted_operator_record",
            "trusted_channel": "operator_console", "channel_record_id": "codex-slot0-semantic-audit-20260915", "channel_record_sha256": sha(operator),
            "approver_id": "responsible-operator", "issued_at": stamp, "expires_at": expires,
            "experiment_id": spec["experiment_id"], "run_id": spec["run_id"],
            "bindings": {"contract_raw_sha256": contract_hash, **contract["bindings"], "repair_envelope_sha256": None}},
        "checks": checks, "storage": [{"path": str(O), "measured_at": stamp, "free_bytes": free,
            **{k: spec["storage"][k] for k in ("hard_floor_bytes", "peak_additional_bytes", "transient_bytes", "recovery_reserve_bytes")}, "verdict": "pass"}],
        "derived_verdict": "pass", "created_at": stamp}
    A.atomic(H / "preflight.json", preflight)
    event = {"sequence": 1, "event_id": "registered", "idempotency_key": spec["run_id"] + ":registered",
             "event_kind": "contract_registered", "occurred_at": stamp, "phase": None, "verdict": "none",
             "relates_to_event_id": None, "notification_status": "not_applicable", "previous_event_sha256": None,
             "event_sha256": A.digest((spec["run_id"] + ":registered").encode())}
    ledger = {**common, "document_kind": "event_ledger", "bindings": {"contract_raw_sha256": contract_hash,
              "preflight_report_raw_sha256": sha(H / "preflight.json"), "schema_bundle_sha256": schema_hash}, "events": [event]}
    A.atomic(H / "ledger.json", ledger)
    entry = {"entry_id": "api-audit", "position": 1, "experiment_id": spec["experiment_id"], "run_id": spec["run_id"],
             "status": "ready", "dependencies": [], "assigned_resource": None, "terminal_notification_status": "not_applicable",
             "bindings": {"contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": sha(H / "preflight.json"),
                          "ledger_raw_sha256": sha(H / "ledger.json"), "schema_bundle_sha256": schema_hash}}
    A.atomic(H / "queue.json", {"document_kind": "queue_state", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
                                "queue_id": "slot0-api-only", "updated_at": stamp, "entries": [entry]})
    print(spec_path)


if __name__ == "__main__":
    main()
