#!/usr/bin/env python3
"""Durable HARN controller for the historical full-track Qwen MF25 evaluation."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/historical_qwen_fulltrack_mf25_harn")
RUNNER = ROOT / "scripts/eval/eval_historical_qwen_fulltrack_noq_s2_mf25_cfg4p5.sh"
CHECKPOINT = ROOT / "exps/phase8_qwen_official_noq_full_stage2_200000/phase8_qwen_official_noq_full_stage2_200000_ema_final.pth"
SOURCE_REPORT = Path("/home/kojiek/logs/phase8_qwen_official_noq_full_FINAL_METRICS.json")
RESULT_REPORT = Path("/home/kojiek/logs/phase8_qwen_official_noq_full_stage2_200000_musiccaps_mf25_cfg4p5_noq_REPORT.json")
PREREG = ROOT / "docs/experiments/historical_qwen_fulltrack_noq_s2_mf25_contract.json"
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")


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
    harn.EXPERIMENT = "historical-qwen-fulltrack-noq-s2-mf25-cfg4p5"
    harn.RUN_ID = "run-20260814-historical-qwen-fulltrack-noq-s2-mf25"
    harn.KS = (0,)
    harn.make_contract = make_contract
    harn.command_registry = command_registry
    harn.run = run


def command_registry() -> dict[str, list[str]]:
    return {"eval_historical_mf25": ["/bin/bash", str(RUNNER)]}


def make_contract() -> dict[str, Any]:
    commands = command_registry()
    zero = "0" * 64
    sources = [
        {"path": str(path), "sha256": harn.digest_file(path)}
        for path in (CHECKPOINT, SOURCE_REPORT, PREREG, TSV, RUNNER)
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
            "phase_id": "historical_fulltrack_noq_s2_mf25_cfg4p5",
            "action_id": "eval_historical_mf25",
            "input_artifacts": sources,
            "output_paths": [str(RESULT_REPORT)],
            "completion_evidence": [{"path": str(RESULT_REPORT), "sha256": zero}],
            "resume_action_id": "eval_historical_mf25",
        }],
        "filesystems": [harn.storage_model()],
        "commands": [{
            "action_id": "eval_historical_mf25",
            "argv": commands["eval_historical_mf25"],
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
    payload = json.loads(RESULT_REPORT.read_text())
    metrics = payload.get("metrics", {})
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if payload.get("status") != "passed" or set(metrics) != required:
        raise RuntimeError("historical MF25 result report is incomplete")
    if payload.get("checkpoint", {}).get("sha256") != harn.digest_file(CHECKPOINT):
        raise RuntimeError("historical MF25 result checkpoint binding mismatch")


def run() -> None:
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = harn.load_current()
        if harn.blocking_gpu_processes():
            raise RuntimeError(f"GPU resource conflict at launch: {harn.blocking_gpu_processes()}")
        checked = subprocess.run(["/bin/bash", str(RUNNER), "--preflight-only"], cwd=ROOT)
        if checked.returncode:
            raise RuntimeError(f"registered evaluator preflight failed with exit {checked.returncode}")
        approval = json.loads(harn.APPROVAL.read_text())
        if not approval.get("consumed"):
            approval["consumed"] = True
            approval["consumed_at"] = harn.now()
            harn.atomic_json(harn.APPROVAL, approval)
        harn.append_event(ledger, "resources_acquired", phase="historical_mf25")
        start_id = harn.append_event(
            ledger, "experiment_started", phase="historical_mf25", notification="pending"
        )
        harn.notify("start", "Started historical full-track one-caption Qwen NoQ Stage 2 MF25 CFG4.5 evaluation.")
        harn.append_event(
            ledger, "notification_delivery", relation=start_id,
            phase="historical_mf25", notification="delivered",
        )
        harn.write_generation(contract, preflight, ledger, "active")
        if not RESULT_REPORT.is_file():
            completed = subprocess.run(
                command_registry()["eval_historical_mf25"], cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
            )
            if completed.returncode:
                raise RuntimeError(f"historical MF25 evaluation failed with exit {completed.returncode}")
        validate_result()
        gate_id = harn.append_event(
            ledger, "gate_result", verdict="pass", phase="historical_mf25", notification="pending"
        )
        harn.notify("metric_complete", "Historical full-track Qwen MF25 CFG4.5 metrics completed.", report=RESULT_REPORT)
        harn.append_event(
            ledger, "notification_delivery", relation=gate_id,
            phase="historical_mf25", notification="delivered",
        )
        harn.write_generation(contract, preflight, ledger, "active")
        harn.terminal(contract, preflight, ledger, True, "Historical full-track Qwen MF25 CFG4.5 evaluation completed.")
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
