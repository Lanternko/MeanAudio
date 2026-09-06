#!/usr/bin/env python3
"""Durable HARN controller for the R-Matched Stage/steps/CFG matrix."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/rmatched_matrix_harn")
RUNNER = ROOT / "scripts/eval/eval_rmatched_s1_s2_steps_cfg_matrix.sh"
PREREG = ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_contract.json"
S1 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage1_400000/phase8_qwen_caption10s_multisent_noq_full_stage1_400000_ema_final.pth"
S2 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
REPORT = Path("/home/kojiek/logs/rmatched_s1_s2_steps_cfg_matrix_seed14159265_REPORT.json")
EXPECTED_CELLS = {
    "s1_fm1_cfg0p5", "s1_fm1_cfg4p5", "s1_fm25_cfg0p5", "s1_fm25_cfg4p5",
    "s2_mf1_cfg0p5", "s2_mf1_cfg4p5", "s2_mf25_cfg0p5", "s2_mf25_cfg4p5",
}


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
    harn.EXPERIMENT = "rmatched-s1-s2-steps-cfg-matrix"
    harn.RUN_ID = "run-20260814-seed14159265-musiccaps"
    harn.KS = (0,)
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.storage_model = storage_model
    harn.run = run


def command_registry() -> dict[str, list[str]]:
    return {"eval_matrix": ["/bin/bash", str(RUNNER)]}


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
    sources = [
        {"path": str(path), "sha256": harn.digest_file(path)}
        for path in (S1, S2, TSV, PREREG, RUNNER)
    ]
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
            "required": True,
            "responsible_role": "operator",
            "trusted_channels": ["operator_console"],
        },
        "corpus": {"kind": "non_generated", "source_artifacts": sources},
        "repair": {"enabled": False},
        "phases": [{
            "phase_id": "stage_steps_cfg_matrix",
            "action_id": "eval_matrix",
            "input_artifacts": sources,
            "output_paths": [str(REPORT)],
            "completion_evidence": [{"path": str(REPORT), "sha256": zero}],
            "resume_action_id": "eval_matrix",
        }],
        "filesystems": [storage_model()],
        "commands": [{
            "action_id": "eval_matrix",
            "argv": commands["eval_matrix"],
            "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        }],
        "required_preflight_checks": [
            "approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
            "no_duplicate", "policy_bound", "storage_policy_1.25",
        ],
        "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def validate_result() -> None:
    payload = json.loads(REPORT.read_text())
    results = payload.get("results", {})
    if payload.get("status") != "passed" or set(results) != EXPECTED_CELLS:
        raise RuntimeError("matrix report is incomplete")
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    for cell in EXPECTED_CELLS:
        if set(results[cell].get("metrics", {})) != required:
            raise RuntimeError(f"matrix metrics are incomplete for {cell}")
        if results[cell].get("peav", {}).get("n_pairs") != 5521:
            raise RuntimeError(f"matrix PE-AV is incomplete for {cell}")
    checkpoints = payload.get("checkpoints", {})
    if checkpoints.get("stage1", {}).get("sha256") != harn.digest_file(S1):
        raise RuntimeError("matrix Stage 1 checkpoint binding mismatch")
    if checkpoints.get("stage2", {}).get("sha256") != harn.digest_file(S2):
        raise RuntimeError("matrix Stage 2 checkpoint binding mismatch")


def run() -> None:
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = harn.load_current()
        if harn.blocking_gpu_processes():
            raise RuntimeError(f"GPU resource conflict at launch: {harn.blocking_gpu_processes()}")
        checked = subprocess.run(["/bin/bash", str(RUNNER), "--preflight-only"], cwd=ROOT)
        if checked.returncode:
            raise RuntimeError(f"matrix preflight failed with exit {checked.returncode}")
        approval = json.loads(harn.APPROVAL.read_text())
        if not approval.get("consumed"):
            approval["consumed"] = True
            approval["consumed_at"] = harn.now()
            harn.atomic_json(harn.APPROVAL, approval)
        harn.append_event(ledger, "resources_acquired", phase="matrix")
        start_id = harn.append_event(ledger, "experiment_started", phase="matrix", notification="pending")
        harn.notify("start", "Started R-Matched Stage 1/Stage 2 steps by CFG matrix evaluation.")
        harn.append_event(
            ledger, "notification_delivery", relation=start_id,
            phase="matrix", notification="delivered",
        )
        harn.write_generation(contract, preflight, ledger, "active")
        if not REPORT.is_file():
            completed = subprocess.run(
                command_registry()["eval_matrix"], cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
            )
            if completed.returncode:
                raise RuntimeError(f"matrix evaluation failed with exit {completed.returncode}")
        validate_result()
        gate_id = harn.append_event(
            ledger, "gate_result", verdict="pass", phase="matrix", notification="pending"
        )
        harn.notify("matrix_complete", "R-Matched Stage/steps/CFG matrix completed.", report=REPORT)
        harn.append_event(
            ledger, "notification_delivery", relation=gate_id,
            phase="matrix", notification="delivered",
        )
        harn.write_generation(contract, preflight, ledger, "active")
        harn.terminal(contract, preflight, ledger, True, "R-Matched Stage/steps/CFG matrix completed.")
    except BaseException as exc:
        try:
            contract, preflight, ledger = harn.load_current()
            terminal_kinds = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
            if not any(event["event_kind"] in terminal_kinds for event in ledger["events"]):
                harn.terminal(contract, preflight, ledger, False, f"Controller failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


configure()


if __name__ == "__main__":
    harn.main()
