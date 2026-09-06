#!/usr/bin/env python3
"""Durable repair-and-resume HARN for the failed R-Matched matrix run."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/rmatched_matrix_repair_harn")
RUNNER = ROOT / "scripts/eval/eval_rmatched_s1_s2_steps_cfg_matrix.sh"
REPAIR = ROOT / "scripts/repair_rmatched_matrix_corrupt_flac.py"
PREREG = ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_repair_contract.json"
ORIGINAL_PREREG = ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_contract.json"
ORIGINAL_HARN = ROOT / "scripts/experiment_harness/rmatched_matrix_harn.py"
FAILED_GENERATION = Path("/home/kojiek/logs/rmatched_matrix_harn/generations/gen-000003")
FAILED_LEDGER = FAILED_GENERATION / "ledger.json"
FAILED_QUEUE = FAILED_GENERATION / "queue.json"
S1 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage1_400000/phase8_qwen_caption10s_multisent_noq_full_stage1_400000_ema_final.pth"
S2 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
REPAIR_REPORT = STATE / "repair_report.json"
REPORT = Path("/home/kojiek/logs/rmatched_s1_s2_steps_cfg_matrix_seed14159265_REPORT.json")
EXPECTED_CELLS = {
    "s1_fm1_cfg0p5", "s1_fm1_cfg4p5", "s1_fm25_cfg0p5", "s1_fm25_cfg4p5",
    "s2_mf1_cfg0p5", "s2_mf1_cfg4p5", "s2_mf25_cfg0p5", "s2_mf25_cfg4p5",
}
PYTHON = Path("/home/kojiek/venvs/dac/bin/python")


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
    harn.RUNNER = RUNNER
    harn.EXPERIMENT = "rmatched-s1-s2-steps-cfg-matrix-repair"
    harn.RUN_ID = "run-20260814-seed14159265-musiccaps-repair1"
    harn.KS = (0,)
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.storage_model = storage_model
    harn.run = run


def command_registry() -> dict[str, list[str]]:
    python = str(PYTHON)
    return {
        "audit_repair": [python, str(REPAIR), "audit"],
        "apply_repair": [python, str(REPAIR), "apply"],
        "rollback_repair": [python, str(REPAIR), "rollback"],
        "resume_matrix": ["/bin/bash", str(RUNNER)],
    }


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
    source_paths = (
        S1, S2, TSV, PREREG, ORIGINAL_PREREG, ORIGINAL_HARN, RUNNER, REPAIR,
        FAILED_LEDGER, FAILED_QUEUE, ROOT / "eval.py", ROOT / "meanaudio/eval_utils.py",
    )
    sources = [{"path": str(path), "sha256": harn.digest_file(path)} for path in source_paths]
    envelope = {
        "envelope_sha256": harn.digest_file(PREREG),
        "writable_paths": [
            str(STATE),
            str(ROOT / "eval_output/rmatched_s1_s2_steps_cfg_matrix_seed14159265/s2_mf25_cfg0p5/audio/5xIBQGMjiX4_30.flac"),
            str(ROOT / "eval_output/metrics/rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5/metrics.txt"),
        ],
        "test_action_ids": ["audit_repair"],
        "apply_action_id": "apply_repair",
        "rollback_action_id": "rollback_repair",
        "resume_action_id": "resume_matrix",
        "allowed_process_identities": ["rmatched_matrix_repair_controller"],
        "reviewer_roles": ["responsible_operator"],
        "budgets": {
            "max_model_calls": 0, "max_wall_seconds": 1800,
            "max_transient_retries": 0, "max_cost_units": 708,
        },
        "operator_required_conditions": [
            "exact_target_hash", "exact_rng_replay", "single_file_mutation", "full_audio_integrity",
        ],
    }
    return {
        "document_kind": "experiment_contract",
        "schema_version": "1.0.0",
        "schema_bundle_id": "harn-schema-v1",
        "experiment_id": harn.EXPERIMENT,
        "run_id": harn.RUN_ID,
        "bindings": {
            "policy_bundle_sha256": harn.policy_hash(),
            "schema_bundle_sha256": harn.schema_hash(),
            "runtime_sha256": harn.digest_file(Path(__file__)),
            "command_set_sha256": harn.digest_bytes(harn.canonical(commands)),
        },
        "approval_requirement": {
            "required": True, "responsible_role": "operator", "trusted_channels": ["operator_console"],
        },
        "corpus": {"kind": "non_generated", "source_artifacts": sources},
        "repair": {"enabled": True, "envelope": envelope},
        "phases": [
            {
                "phase_id": "repair_audio", "action_id": "apply_repair",
                "resume_action_id": "apply_repair", "input_artifacts": sources,
                "output_paths": [str(REPAIR_REPORT)],
                "completion_evidence": [{"path": str(REPAIR_REPORT), "sha256": zero}],
            },
            {
                "phase_id": "resume_matrix", "action_id": "resume_matrix",
                "resume_action_id": "resume_matrix", "input_artifacts": sources,
                "output_paths": [str(REPORT)],
                "completion_evidence": [{"path": str(REPORT), "sha256": zero}],
            },
        ],
        "filesystems": [storage_model()],
        "commands": [
            {
                "action_id": action_id, "argv": argv, "working_directory": str(ROOT),
                "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
            }
            for action_id, argv in commands.items()
        ],
        "required_preflight_checks": [
            "approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
            "no_duplicate", "policy_bound", "storage_policy_1.25",
        ],
        "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def validate_repair() -> None:
    payload = json.loads(REPAIR_REPORT.read_text())
    if payload.get("status") != "passed" or payload.get("changed_audio_files") != ["5xIBQGMjiX4_30.flac"]:
        raise RuntimeError("registered single-file repair did not validate")
    if payload.get("prefix_replay_hash_matches") != 707:
        raise RuntimeError("RNG replay equivalence evidence is incomplete")


def validate_matrix() -> None:
    payload = json.loads(REPORT.read_text())
    results = payload.get("results", {})
    if payload.get("status") != "passed" or set(results) != EXPECTED_CELLS:
        raise RuntimeError("matrix report is incomplete")
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    for cell in EXPECTED_CELLS:
        if set(results[cell].get("metrics", {})) != required or results[cell].get("peav", {}).get("n_pairs") != 5521:
            raise RuntimeError(f"matrix evidence is incomplete for {cell}")
    if payload.get("checkpoints", {}).get("stage1", {}).get("sha256") != harn.digest_file(S1):
        raise RuntimeError("matrix Stage 1 checkpoint binding mismatch")
    if payload.get("checkpoints", {}).get("stage2", {}).get("sha256") != harn.digest_file(S2):
        raise RuntimeError("matrix Stage 2 checkpoint binding mismatch")


def load_after_resource_wait() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    approval = json.loads(harn.APPROVAL.read_text())
    if harn.CURRENT.exists():
        contract, preflight, ledger = harn.load_current()
    else:
        contract = json.loads(harn.PENDING_CONTRACT.read_text())
        preflight = json.loads(harn.PENDING_PREFLIGHT.read_text())
        ledger = json.loads(harn.PENDING_LEDGER.read_text())
    while harn.blocking_gpu_processes():
        harn.atomic_json(harn.WATCH_STATUS, {
            "observed_at": harn.now(), "status": "held", "reason": "resource_conflict",
            "gpu_processes": harn.gpu_processes(), "blocking_gpu_processes": harn.blocking_gpu_processes(),
            "gpu_free_mib": harn.gpu_free_mib(), "assigned_resource": None,
        })
        time.sleep(60)
    if preflight["derived_verdict"] != "pass":
        preflight = harn.make_preflight(contract, approval["approval_text_sha256"], True)
        harn.append_event(ledger, "preflight_passed", verdict="pass")
        harn.write_generation(contract, preflight, ledger, "ready")
    return contract, preflight, ledger


def run_action(argv: list[str], label: str) -> None:
    completed = subprocess.run(
        argv, cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
    )
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}")


def run() -> None:
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = load_after_resource_wait()
        approval = json.loads(harn.APPROVAL.read_text())
        approval.update({"consumed": True, "consumed_at": harn.now()})
        harn.atomic_json(harn.APPROVAL, approval)
        harn.append_event(ledger, "resources_acquired", phase="repair_audio")
        started = harn.append_event(ledger, "experiment_started", phase="repair_audio", notification="pending")
        harn.notify("start", "Started exact RNG-replay repair for the failed R-Matched matrix.")
        harn.append_event(ledger, "notification_delivery", relation=started, phase="repair_audio", notification="delivered")
        harn.write_generation(contract, preflight, ledger, "active")
        commands = command_registry()
        run_action(commands["audit_repair"], "repair audit")
        run_action(commands["apply_repair"], "exact RNG-replay repair")
        validate_repair()
        gate = harn.append_event(ledger, "gate_result", verdict="pass", phase="repair_audio", notification="pending")
        harn.notify("repair_complete", "Exact one-file repair passed; resuming the preregistered matrix.", report=REPAIR_REPORT)
        harn.append_event(ledger, "notification_delivery", relation=gate, phase="repair_audio", notification="delivered")
        harn.append_event(ledger, "promotion_started", relation=gate, phase="resume_matrix")
        harn.write_generation(contract, preflight, ledger, "active")
        if not REPORT.is_file():
            run_action(commands["resume_matrix"], "matrix continuation")
        validate_matrix()
        matrix_gate = harn.append_event(ledger, "gate_result", verdict="pass", phase="resume_matrix", notification="pending")
        harn.notify("matrix_complete", "Repaired R-Matched Stage/steps/CFG matrix completed.", report=REPORT)
        harn.append_event(ledger, "notification_delivery", relation=matrix_gate, phase="resume_matrix", notification="delivered")
        harn.write_generation(contract, preflight, ledger, "active")
        harn.terminal(contract, preflight, ledger, True, "Matrix repair and continuation completed; queue handoff is next.")
    except BaseException as exc:
        try:
            if harn.CURRENT.exists():
                contract, preflight, ledger = harn.load_current()
                terminal_kinds = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
                if not any(event["event_kind"] in terminal_kinds for event in ledger["events"]):
                    harn.terminal(contract, preflight, ledger, False, f"Repair controller failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


configure()


if __name__ == "__main__":
    harn.main()
