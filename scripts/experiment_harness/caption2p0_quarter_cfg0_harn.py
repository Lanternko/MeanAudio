#!/usr/bin/env python3
"""Durable held/queued HARN for four Caption 2.0 quarter CFG0 evaluations."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn

ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/caption2p0_quarter_cfg0_harn")
PREREG = ROOT / "docs/experiments/caption2p0_quarter_cfg0_rerun_contract.json"
POLICY = ROOT / "docs/experiments/evaluation_policy.md"
RUNNER = ROOT / "scripts/eval/eval_caption2p0_quarter_cfg0.sh"
WRAPPER = ROOT / "scripts/caption10s_pipeline/eval_musiccaps_mf25.sh"
VALIDATOR = ROOT / "scripts/validate_experiment_harness_documents.py"
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
METRIC_EVALUATOR = Path("/home/kojiek/research/meanaudio_eval/phase4_eval.py")
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
STRICT_VALIDATOR = ROOT / "scripts/eval/validate_caption2p0_cfg0_report.py"
COMPLETION_VALIDATOR = ROOT / "scripts/eval/validate_caption2p0_cfg0_bundle.py"
PATH_VALIDATOR = ROOT / "scripts/eval/validate_cfg0_output_path.py"
BASE_HARN = ROOT / "scripts/experiment_harness/qwen_s2q_k_mf25_harn.py"
EVAL_ENTRY = ROOT / "eval.py"
EVAL_UTILS = ROOT / "meanaudio/eval_utils.py"
NETWORKS = ROOT / "meanaudio/model/networks.py"
COMPAT_ENV = ROOT / "scripts/runtime/phase8_nvidia_compat_env.sh"
DAC_PYTHON = Path("/home/kojiek/venvs/dac/bin/python")
ARMS = ("caption2p0", "bestof3", "worstof3", "qwen3cap_k3_q9")
CELL_SECONDS = 90 * 60
TOTAL_SECONDS = 360 * 60
BASE_MAKE_PREFLIGHT = harn.make_preflight
CHILD_PATH = "/home/kojiek/venvs/dac/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"


def prereg() -> dict[str, Any]:
    return json.loads(PREREG.read_text())


def cell(arm: str) -> dict[str, Any]:
    matches = [item for item in prereg()["cells"] if item["cell_id"] == arm]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate arm: {arm}")
    return matches[0]


def command_registry() -> dict[str, list[str]]:
    return {f"eval_{arm}": ["/bin/bash", str(RUNNER), arm] for arm in ARMS}


def storage_model() -> dict[str, int | str]:
    return {
        "path": "/", "hard_floor_bytes": 150 * harn.GIB,
        "warning_floor_bytes": 180 * harn.GIB,
        "peak_additional_bytes": 8 * harn.GIB,
        "transient_bytes": 8 * harn.GIB,
        "recovery_reserve_bytes": 10 * harn.GIB,
    }


def make_contract() -> dict[str, Any]:
    commands = command_registry()
    zero = "0" * 64
    registration = prereg()
    fixed = registration["fixed_protocol"]
    if fixed["num_steps"] != 25 or fixed["cfg_strength"] != 0 or fixed["expected_rows"] != 5521:
        raise RuntimeError("preregistered canonical protocol drift")
    if harn.digest_file(TSV) != fixed["tsv_sha256"]:
        raise RuntimeError("MusicCaps TSV drift from preregistration")
    checkpoints = []
    for arm in ARMS:
        item = cell(arm)
        checkpoint = Path(item["checkpoint"])
        if harn.digest_file(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint drift from preregistration: {arm}")
        checkpoints.append(checkpoint)
    sources_paths = [
        PREREG, POLICY, RUNNER, WRAPPER, STRICT_VALIDATOR, COMPLETION_VALIDATOR, PATH_VALIDATOR,
        VALIDATOR, NOTIFIER,
        BASE_HARN, EVAL_ENTRY, EVAL_UTILS, NETWORKS, DAC_PYTHON, TSV,
        METRIC_EVALUATOR, *checkpoints,
    ]
    if COMPAT_ENV.is_file():
        sources_paths.append(COMPAT_ENV)
    sources = [{"path": str(path), "sha256": harn.digest_file(path)} for path in sources_paths]
    phases = []
    for arm in ARMS:
        item = cell(arm)
        phases.append({
            "phase_id": arm, "action_id": f"eval_{arm}", "resume_action_id": f"eval_{arm}",
            "input_artifacts": sources, "output_paths": [item["report"]],
            "completion_evidence": [{"path": item["report"], "sha256": zero}],
        })
    registered = [{
        "action_id": action, "argv": argv, "working_directory": str(ROOT),
        "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
    } for action, argv in commands.items()]
    return {
        "document_kind": "experiment_contract", "schema_version": "1.0.0",
        "schema_bundle_id": "harn-schema-v1",
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "bindings": {
            "policy_bundle_sha256": harn.policy_hash(), "schema_bundle_sha256": harn.schema_hash(),
            "runtime_sha256": harn.digest_file(Path(__file__)),
            "command_set_sha256": harn.digest_bytes(harn.canonical(commands)),
        },
        "approval_requirement": {
            "required": True, "responsible_role": "operator", "trusted_channels": ["operator_console"],
        },
        "corpus": {"kind": "non_generated", "source_artifacts": sources},
        "repair": {"enabled": False}, "phases": phases,
        "filesystems": [storage_model()], "commands": registered,
        "required_preflight_checks": [
            "approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
            "no_duplicate", "policy_bound", "storage_policy_1.25",
        ],
        "notification_events": ["hold", "start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def strict_report_ok(arm: str) -> bool:
    report = Path(cell(arm)["report"])
    if not report.is_file():
        return False
    completed = subprocess.run(
        [str(DAC_PYTHON), str(STRICT_VALIDATOR), "--contract", str(PREREG),
         "--arm", arm, "--report", str(report)],
        cwd=ROOT, text=True, capture_output=True,
    )
    return completed.returncode == 0


def make_preflight(contract: dict[str, Any], approval_hash: str, gpu_idle: bool) -> dict[str, Any]:
    report = BASE_MAKE_PREFLIGHT(contract, approval_hash, gpu_idle)
    commands = command_registry()
    source_map = {
        item["path"]: item["sha256"]
        for item in contract["corpus"]["source_artifacts"]
    }
    current_sources = all(Path(path).is_file() and harn.digest_file(Path(path)) == digest
                          for path, digest in source_map.items())
    command_bound = (
        contract["bindings"]["command_set_sha256"] == harn.digest_bytes(harn.canonical(commands))
        and [item["argv"] for item in contract["commands"]] == list(commands.values())
    )
    p = prereg()
    policy_bound = (
        p["fixed_protocol"]["num_steps"] == 25
        and p["fixed_protocol"]["cfg_strength"] == 0
        and p["fixed_protocol"]["expected_rows"] == 5521
        and "--num_steps 25 --cfg_strength 0" in WRAPPER.read_text()
    )
    duplicate_ok = True
    for arm in ARMS:
        item = cell(arm)
        metrics = Path(p["runtime_storage"]["metrics_root"]) / item["label"] / "metrics.txt"
        report_path = Path(item["report"])
        if report_path.exists() and not strict_report_ok(arm):
            duplicate_ok = False
        if metrics.exists() and not report_path.exists():
            duplicate_ok = False
    verdicts = {
        "approval_authenticated": len(approval_hash) == 64 and all(c in "0123456789abcdef" for c in approval_hash),
        "commands_bound": command_bound,
        "gpu_idle": gpu_idle,
        "inputs_bound": current_sources,
        "no_duplicate": duplicate_ok,
        "policy_bound": policy_bound,
        "storage_policy_1.25": harn.storage_check()["verdict"] == "pass",
    }
    for check in report["checks"]:
        check["verdict"] = "pass" if verdicts.get(check["check_id"], False) else "fail"
        check["evidence_sha256"] = harn.digest_bytes(
            f"{check['check_id']}:{check['verdict']}:{contract['bindings']['runtime_sha256']}".encode()
        )
    report["derived_verdict"] = "pass" if all(x["verdict"] == "pass" for x in report["checks"]) else "fail"
    return report


def validate_report(arm: str) -> Path:
    path = Path(cell(arm)["report"])
    if not strict_report_ok(arm):
        raise RuntimeError(f"strict report validation failed for {arm}")
    return path


def resource_gate(ledger: dict[str, Any], arm: str, run_started: float) -> None:
    elapsed = time.monotonic() - run_started
    if elapsed >= TOTAL_SECONDS:
        raise RuntimeError("total 360 GPU-minute budget exhausted")
    storage = harn.storage_check()
    free = int(storage["free_bytes"])
    if free < 150 * harn.GIB:
        event = harn.append_event(ledger, "disk_hard_stop", verdict="fail", phase=arm, notification="pending")
        harn.notify(f"disk_hard_stop_{arm}", f"CFG0 rerun hard stop before {arm}: free_bytes={free}", status="held")
        harn.append_event(ledger, "notification_delivery", relation=event, phase=arm, notification="delivered")
        raise RuntimeError("root filesystem below 150 GiB hard floor")
    if free < 180 * harn.GIB:
        event = harn.append_event(ledger, "disk_warning", verdict="fail", phase=arm, notification="pending")
        harn.notify(f"disk_warning_{arm}", f"CFG0 rerun warning before {arm}: free_bytes={free}", status="held")
        harn.append_event(ledger, "notification_delivery", relation=event, phase=arm, notification="delivered")


def run_bounded(argv: list[str], arm: str, remaining_seconds: float) -> None:
    timeout = min(CELL_SECONDS, max(1, int(remaining_seconds)))
    child_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HOME": "/home/kojiek",
        "LANG": "C.UTF-8",
        "PATH": CHILD_PATH,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        argv, cwd=ROOT, start_new_session=True,
        env=child_env,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(f"{arm} exceeded registered {timeout}-second budget")
    if returncode:
        raise RuntimeError(f"{arm} failed with exit {returncode}")


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
                "schema_version": 1, "status": "promoted_to_harn", "order": list(ARMS),
                "updated_at": harn.now(), "current_generation": harn.CURRENT.read_text().strip(),
            })
            break
        harn.atomic_json(harn.WATCH_STATUS, {
            "observed_at": harn.now(), "status": "held", "reason": "preflight_hold",
            "failed_checks": [x["check_id"] for x in preflight["checks"] if x["verdict"] != "pass"],
            "storage": harn.storage_check(), "gpu_processes": harn.gpu_processes(),
            "blocking_gpu_processes": conflicts, "gpu_free_mib": harn.gpu_free_mib(),
            "assigned_resource": None,
        })
        time.sleep(60)
    return contract, preflight, ledger


def run() -> None:
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = load_after_resource_wait()
        approval = json.loads(harn.APPROVAL.read_text())
        if not approval.get("consumed"):
            approval.update({"consumed": True, "consumed_at": harn.now()})
            harn.atomic_json(harn.APPROVAL, approval)
        harn.append_event(ledger, "resources_acquired", phase=ARMS[0])
        start = harn.append_event(ledger, "experiment_started", phase=ARMS[0], notification="pending")
        harn.notify("start", "Started four-cell Caption 2.0 quarter MusicCaps MF25 CFG0 rerun.")
        harn.append_event(ledger, "notification_delivery", relation=start, phase=ARMS[0], notification="delivered")
        harn.write_generation(contract, preflight, ledger, "active")
        commands = command_registry()
        run_started = time.monotonic()
        for index, arm in enumerate(ARMS):
            resource_gate(ledger, arm, run_started)
            report = Path(cell(arm)["report"])
            if not report.is_file():
                run_bounded(commands[f"eval_{arm}"], arm, TOTAL_SECONDS - (time.monotonic() - run_started))
            validate_report(arm)
            gate = harn.append_event(ledger, "gate_result", verdict="pass", phase=arm, notification="pending")
            harn.notify(f"{arm}_complete", f"{arm} MusicCaps MF25 CFG0 completed.", report=report)
            harn.append_event(ledger, "notification_delivery", relation=gate, phase=arm, notification="delivered")
            if index + 1 < len(ARMS):
                harn.append_event(ledger, "promotion_started", relation=gate, phase=ARMS[index + 1])
            harn.write_generation(contract, preflight, ledger, "active")
        harn.terminal(contract, preflight, ledger, True, "Four Caption 2.0 quarter MF25 CFG0 evaluations completed.")
    except BaseException as exc:
        try:
            contract, preflight, ledger = harn.load_current()
            terminal = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
            if not any(event["event_kind"] in terminal for event in ledger["events"]):
                harn.terminal(contract, preflight, ledger, False, f"Controller failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


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
    harn.EXPERIMENT = "phase8-caption2p0-quarter-mf25-cfg0-rerun"
    harn.RUN_ID = "run-20260820-caption2p0-quarter-mf25-cfg0-v1"
    harn.KS = ARMS
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.make_preflight = make_preflight
    harn.storage_model = storage_model
    harn.run = run


configure()

if __name__ == "__main__":
    harn.main()
