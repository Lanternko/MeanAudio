#!/usr/bin/env python3
"""Durable HARN controller for Caption 2.0 full-scale K=5 S2Q."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/caption2p0_k5_s2q_harn")
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")
PREREG = ROOT / "docs/experiments/caption2p0_full_k5_s2q_contract.json"
AUDITOR = ROOT / "scripts/preprocess/audit_caption_npz_binding.py"
TRAINER = ROOT / "scripts/training_pipelines/execute_phase8_qwen_bucket_s2q_from_noq.sh"
EVALUATOR = ROOT / "scripts/eval/eval_phase8_qwen_s2q_full_k_mf25_cfg4p5.sh"
ACTION_RUNNER = ROOT / "scripts/training_pipelines/run_caption2p0_k5_s2q_action.sh"
TSV = ROOT / "data/phase8_caption2p0_k5_balanced_train.tsv"
ASSIGNMENT = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_meansim_k5_balanced.tsv")
CACHE_LIST = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt")
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_qwen_official_matched_npz")
SOURCE_CONTRACT = Path("/home/kojiek/logs/phase8_qwen_caption10s_multisent_noq_full_contract.json")
STRICT_GATE = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/caption10s_multisent_strict_gate.json")
CORPUS = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/captions_full_251599_10s_multisent.jsonl")
TSV_MANIFEST = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/caption10s_multisent_train_tsv.manifest.json")
GENERATED_POLICY = ROOT / "docs/experiments/generated_corpus_policy.md"
REEXTRACT_REPORT = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/reextract_multisent.DONE.json")
AUDIT = Path("/home/kojiek/logs/phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_BINDING_AUDIT.json")
DIAGNOSTIC = Path("/home/kojiek/logs/phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_FINAL_METRICS.json")
FAIR = Path("/home/kojiek/logs/phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000_musiccaps_n5521_mf25_cfg4p5_q9_REPORT.json")
CHECKPOINT = ROOT / "exps/phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000/phase8_qwen_caption2p0_s2q_from_noq_full_k5_balanced_stage2_200000_ema_final.pth"


def require_active_preregistration() -> None:
    status = json.loads(PREREG.read_text()).get("status")
    if status != "preregistered_held_for_dependencies":
        harn.atomic_json(STATE / "supersession_hold.json", {
            "status": "held", "reason": "HOLD_CFG0_RECONTRACT",
            "preregistration_status": status, "observed_at": harn.now(),
        })
        raise RuntimeError(f"HOLD_CFG0_RECONTRACT preregistration status={status}")


def command_registry() -> dict[str, list[str]]:
    audit = [
        str(PYTHON), str(AUDITOR), "--tsv", str(TSV), "--cache-list", str(CACHE_LIST),
        "--npz-dir", str(NPZ_DIR), "--report", str(AUDIT), "--expected-rows", "251599",
        "--workers", "8",
    ]
    train_fresh = ["/bin/bash", str(ACTION_RUNNER), "train-fresh"]
    train_resume = ["/bin/bash", str(ACTION_RUNNER), "train-resume"]
    fair = ["/bin/bash", str(ACTION_RUNNER), "fair-eval"]
    return {"binding_audit": audit, "train_fresh": train_fresh, "train_resume": train_resume, "fair_eval": fair}


def configure() -> None:
    harn.STATE = STATE
    harn.GENERATIONS = STATE / "generations"
    harn.OUTBOX = STATE / "outbox"
    harn.CURRENT = STATE / "current"
    harn.KEY = STATE / "ledger_hmac.key"
    harn.LOCK = STATE / "controller.lock"
    harn.APPROVAL = STATE / "operator_approval.json"
    harn.PROCESS = STATE / "process_identity.json"
    harn.WATCH_STATUS = STATE / "watch_status.json"
    harn.PENDING_CONTRACT = STATE / "pending_contract.json"
    harn.PENDING_PREFLIGHT = STATE / "pending_preflight.json"
    harn.PENDING_LEDGER = STATE / "pending_ledger.json"
    harn.PENDING_QUEUE = STATE / "pending_queue.json"
    harn.RUNNER = TRAINER
    harn.EXPERIMENT = "phase8-caption2p0-full-k5-s2q"
    harn.RUN_ID = "run-20260814-caption2p0-full-k5-s2q"
    harn.KS = (5,)
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.storage_model = storage_model
    harn.run = run


def storage_model() -> dict[str, int | str]:
    return {
        "path": "/", "hard_floor_bytes": 150 * harn.GIB,
        "warning_floor_bytes": 180 * harn.GIB,
        "peak_additional_bytes": 40 * harn.GIB,
        "transient_bytes": 15 * harn.GIB,
        "recovery_reserve_bytes": 10 * harn.GIB,
    }


def make_contract() -> dict[str, Any]:
    commands = command_registry()
    zero = "0" * 64
    source_paths = (PREREG, AUDITOR, TRAINER, EVALUATOR, ACTION_RUNNER, TSV, ASSIGNMENT, CACHE_LIST,
                    SOURCE_CONTRACT, STRICT_GATE, CORPUS, TSV_MANIFEST, GENERATED_POLICY,
                    REEXTRACT_REPORT)
    sources = [{"path": str(path), "sha256": harn.digest_file(path)} for path in source_paths]
    phases = [
        {"phase_id": "binding_audit", "action_id": "binding_audit", "resume_action_id": "binding_audit",
         "input_artifacts": sources, "output_paths": [str(AUDIT)],
         "completion_evidence": [{"path": str(AUDIT), "sha256": zero}]},
        {"phase_id": "train_and_diagnostic", "action_id": "train_fresh", "resume_action_id": "train_resume",
         "input_artifacts": sources, "output_paths": [str(CHECKPOINT), str(DIAGNOSTIC)],
         "completion_evidence": [{"path": str(DIAGNOSTIC), "sha256": zero}]},
        {"phase_id": "fair_eval", "action_id": "fair_eval", "resume_action_id": "fair_eval",
         "input_artifacts": sources, "output_paths": [str(FAIR)],
         "completion_evidence": [{"path": str(FAIR), "sha256": zero}]},
    ]
    registered = [
        {"action_id": action, "argv": argv, "working_directory": str(ROOT),
         "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}}
        for action, argv in commands.items()
    ]
    return {
        "document_kind": "experiment_contract", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "bindings": {"policy_bundle_sha256": harn.policy_hash(), "schema_bundle_sha256": harn.schema_hash(),
                     "runtime_sha256": harn.digest_file(Path(__file__)),
                     "command_set_sha256": harn.digest_bytes(harn.canonical(commands))},
        "approval_requirement": {"required": True, "responsible_role": "operator", "trusted_channels": ["operator_console"]},
        "corpus": {
            "kind": "generated",
            "corpus_artifact": {"path": str(CORPUS), "sha256": harn.digest_file(CORPUS)},
            "corpus_schema_sha256": harn.digest_file(TSV_MANIFEST),
            "classifier_version": "caption10s-strict-gate-v1",
            "defect_taxonomy_sha256": harn.digest_file(GENERATED_POLICY),
            "stop_behavior_test_id": "caption10s-multisent-stop-clean-v1",
            "full_gate_report": {"path": str(STRICT_GATE), "sha256": harn.digest_file(STRICT_GATE)},
            "downstream_bindings": {
                "tsv_manifest_sha256": harn.digest_file(TSV_MANIFEST),
                "feature_cache_report_sha256": harn.digest_file(REEXTRACT_REPORT),
            },
            "required_gate_points": ["launch", "pre_training", "post_change"],
        },
        "repair": {"enabled": False}, "phases": phases,
        "filesystems": [storage_model()],
        "commands": registered,
        "required_preflight_checks": ["approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
                                      "no_duplicate", "policy_bound", "storage_policy_1.25",
                                      "generated_corpus_full_gate"],
        "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def validate_audit() -> None:
    payload = json.loads(AUDIT.read_text())
    if payload.get("status") != "passed" or payload.get("rows_checked") != 251599:
        raise RuntimeError("Caption 2.0 K=5 binding audit did not pass")


def validate_generated_gate() -> None:
    gate = json.loads(STRICT_GATE.read_text())
    manifest = json.loads(TSV_MANIFEST.read_text())
    reextract = json.loads(REEXTRACT_REPORT.read_text())
    if gate.get("status") != "passed" or gate.get("corpus", {}).get("sha256") != harn.digest_file(CORPUS):
        raise RuntimeError("Caption 2.0 strict full-corpus gate is invalid")
    if manifest.get("status") != "passed" or manifest.get("rows") != 251599:
        raise RuntimeError("Caption 2.0 TSV manifest is invalid")
    if reextract.get("status") != "passed":
        raise RuntimeError("Caption 2.0 feature re-extraction report is invalid")


def validate_diagnostic() -> None:
    payload = json.loads(DIAGNOSTIC.read_text())
    if payload.get("status") != "passed" or payload.get("k") != 5 or payload.get("strategy") != "balanced":
        raise RuntimeError("Caption 2.0 K=5 diagnostic report did not pass")
    if payload.get("model", {}).get("sha256") != harn.digest_file(CHECKPOINT):
        raise RuntimeError("Caption 2.0 K=5 checkpoint binding mismatch")


def validate_fair() -> None:
    payload = json.loads(FAIR.read_text())
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if payload.get("status") != "passed" or set(payload.get("metrics", {})) != required:
        raise RuntimeError("Caption 2.0 K=5 fair report is incomplete")
    if payload.get("provenance", {}).get("checkpoint_sha256") != harn.digest_file(CHECKPOINT):
        raise RuntimeError("Caption 2.0 K=5 fair report checkpoint mismatch")


def run_action(argv: list[str], label: str, extra_env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        argv, cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1", **(extra_env or {})},
    )
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}")


def gate(ledger: dict[str, Any], contract: dict[str, Any], preflight: dict[str, Any], phase: str, summary: str, report: Path) -> None:
    event = harn.append_event(ledger, "gate_result", verdict="pass", phase=phase, notification="pending")
    harn.notify(f"{phase}_complete", summary, report=report)
    harn.append_event(ledger, "notification_delivery", relation=event, phase=phase, notification="delivered")
    harn.append_event(ledger, "promotion_started", relation=event, phase=phase)
    harn.write_generation(contract, preflight, ledger, "active")


def load_after_resource_wait() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    approval = json.loads(harn.APPROVAL.read_text())
    if harn.CURRENT.exists():
        contract, preflight, ledger = harn.load_current()
    else:
        contract = json.loads(harn.PENDING_CONTRACT.read_text())
        preflight = json.loads(harn.PENDING_PREFLIGHT.read_text())
        ledger = json.loads(harn.PENDING_LEDGER.read_text())
    while preflight["derived_verdict"] != "pass":
        conflicts = harn.blocking_gpu_processes()
        preflight = harn.make_preflight(contract, approval["approval_text_sha256"], not conflicts)
        if preflight["derived_verdict"] == "pass":
            harn.append_event(ledger, "preflight_passed", verdict="pass")
            harn.write_generation(contract, preflight, ledger, "ready")
            harn.atomic_json(harn.PENDING_QUEUE, {
                "schema_version": 1, "status": "promoted_to_harn", "order": [5],
                "updated_at": harn.now(), "current_generation": harn.CURRENT.read_text().strip(),
            })
            break
        failed_checks = [item["check_id"] for item in preflight["checks"] if item["verdict"] != "pass"]
        harn.atomic_json(harn.WATCH_STATUS, {
            "observed_at": harn.now(), "status": "held", "reason": "preflight_hold",
            "failed_checks": failed_checks, "storage": harn.storage_check(),
            "gpu_processes": harn.gpu_processes(), "blocking_gpu_processes": conflicts,
            "gpu_free_mib": harn.gpu_free_mib(), "assigned_resource": None,
        })
        time.sleep(60)
    return contract, preflight, ledger


def run() -> None:
    require_active_preregistration()
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = load_after_resource_wait()
        validate_generated_gate()
        approval = json.loads(harn.APPROVAL.read_text())
        approval.update({"consumed": True, "consumed_at": harn.now()})
        harn.atomic_json(harn.APPROVAL, approval)
        harn.append_event(ledger, "resources_acquired", phase="binding_audit")
        start = harn.append_event(ledger, "experiment_started", phase="binding_audit", notification="pending")
        harn.notify("start", "Started Caption 2.0 full-scale K=5 S2Q chain: audit, training, diagnostic, and MF25 fair eval.")
        harn.append_event(ledger, "notification_delivery", relation=start, phase="binding_audit", notification="delivered")
        harn.write_generation(contract, preflight, ledger, "active")
        commands = command_registry()
        run_action(commands["binding_audit"], "binding audit")
        validate_audit()
        gate(ledger, contract, preflight, "binding_audit", "Caption 2.0 K=5 fresh binding audit passed; promoting training.", AUDIT)
        checked = subprocess.run(
            ["/bin/bash", str(ACTION_RUNNER), "train-preflight"], cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        )
        if checked.returncode:
            raise RuntimeError(f"training preflight failed with exit {checked.returncode}")
        if not DIAGNOSTIC.is_file():
            run_action(commands["train_fresh"], "K=5 training and diagnostic evaluation")
        validate_diagnostic()
        gate(ledger, contract, preflight, "train_and_diagnostic", "Caption 2.0 K=5 training and diagnostic evaluation passed; promoting fair eval.", DIAGNOSTIC)
        if not FAIR.is_file():
            run_action(commands["fair_eval"], "K=5 MF25 fair evaluation")
        validate_fair()
        event = harn.append_event(ledger, "gate_result", verdict="pass", phase="fair_eval", notification="pending")
        harn.notify("fair_eval_complete", "Caption 2.0 full-scale K=5 S2Q MF25 fair evaluation completed.", report=FAIR)
        harn.append_event(ledger, "notification_delivery", relation=event, phase="fair_eval", notification="delivered")
        harn.write_generation(contract, preflight, ledger, "active")
        harn.terminal(contract, preflight, ledger, True, "Caption 2.0 full-scale K=5 S2Q chain completed.")
    except BaseException as exc:
        try:
            contract, preflight, ledger = harn.load_current()
            terminal_kinds = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
            if not any(item["event_kind"] in terminal_kinds for item in ledger["events"]):
                harn.terminal(contract, preflight, ledger, False, f"Controller failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


configure()


if __name__ == "__main__":
    harn.main()
