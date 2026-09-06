#!/usr/bin/env python3
"""Apply or roll back the reviewed Phase-8 NVMe evaluation-output repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
STATE_DIR = Path("/home/kojiek/logs/phase8_qwen_bucket_quarter_backlog")
CONFIG = STATE_DIR / "eval_output_root"
REPAIR_STATE = STATE_DIR / "eval_output_root_repair.json"
LOCAL_OUTPUT = ROOT / "eval_output_local"
CONTRACT = Path(
    "/home/kojiek/logs/"
    "phase8_qwen_bucket_quarter_k5_balanced_contract.json"
)
EXECUTE_SCRIPT = (
    ROOT / "scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh"
)
SEQUENCE_SCRIPT = (
    ROOT
    / "scripts/training_pipelines/"
    "sequence_phase8_qwen_bucket_quarter_backlog.sh"
)
CRITICAL_KEY = (
    "scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_contract(payload: dict[str, Any]) -> dict[str, str]:
    if (
        payload.get("experiment")
        != "phase8_qwen_bucket_quarter_k5_balanced"
        or payload.get("k") != 5
        or payload.get("strategy") != "balanced"
        or payload.get("scale") != "quarter"
        or payload.get("seed") != 14159265
        or payload.get("stage1_updates") != 100000
        or payload.get("stage2_updates") != 50000
        or payload.get("stage2_final_iteration") != 150000
    ):
        raise ValueError("K5 scientific contract identity changed")
    critical = payload.get("critical_file_sha256")
    if not isinstance(critical, dict):
        raise ValueError("critical_file_sha256 is missing")
    current = critical.get(CRITICAL_KEY)
    if not isinstance(current, str) or len(current) != 64:
        raise ValueError("execute-script contract hash is invalid")
    return critical


def validate_preflight() -> None:
    environment = os.environ.copy()
    environment.update({
        "K": "5",
        "STRATEGY": "balanced",
        "SCALE": "quarter",
        "REUSE": "none",
        "VALIDATE_ONLY": "true",
        "EXPERIMENT_RUN_MODE": "resume",
        "EVAL_OUTPUT_ROOT": str(LOCAL_OUTPUT),
    })
    result = subprocess.run(
        ["bash", str(EXECUTE_SCRIPT)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        evidence = (result.stdout + "\n" + result.stderr)[-4000:]
        raise RuntimeError(f"K5 read-only preflight failed: {evidence}")


def apply_repair(execute_sha: str, sequence_sha: str) -> None:
    if sha256(EXECUTE_SCRIPT) != execute_sha:
        raise ValueError("live execute script does not match reviewed hash")
    if sha256(SEQUENCE_SCRIPT) != sequence_sha:
        raise ValueError("live sequence script does not match reviewed hash")
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = read_json(CONTRACT)
    critical = validate_contract(contract)
    previous_hash = critical[CRITICAL_KEY]

    if REPAIR_STATE.is_file():
        state = read_json(REPAIR_STATE)
        if (
            state.get("status") == "applied"
            and state.get("execute_sha256") == execute_sha
            and state.get("sequence_sha256") == sequence_sha
            and CONFIG.read_text(encoding="utf-8").strip()
            == str(LOCAL_OUTPUT)
            and critical[CRITICAL_KEY] == execute_sha
        ):
            validate_preflight()
            print(json.dumps({"status": "already_applied"}))
            return
        raise ValueError("a different eval-output repair state already exists")

    state = {
        "schema_version": 1,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute_sha256": execute_sha,
        "sequence_sha256": sequence_sha,
        "previous_contract_execute_sha256": previous_hash,
        "configured_output_root": str(LOCAL_OUTPUT),
    }
    atomic_json(REPAIR_STATE, state)
    try:
        critical[CRITICAL_KEY] = execute_sha
        atomic_json(CONTRACT, contract)
        atomic_text(CONFIG, f"{LOCAL_OUTPUT}\n")
        validate_preflight()
        state["status"] = "applied"
        state["applied_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(REPAIR_STATE, state)
    except Exception:
        critical[CRITICAL_KEY] = previous_hash
        atomic_json(CONTRACT, contract)
        CONFIG.unlink(missing_ok=True)
        state["status"] = "apply_failed_rolled_back"
        state["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(REPAIR_STATE, state)
        raise
    print(json.dumps({"status": "applied", "output_root": str(LOCAL_OUTPUT)}))


def rollback_repair() -> None:
    state = read_json(REPAIR_STATE)
    if state.get("status") in {"rolled_back", "apply_failed_rolled_back"}:
        print(json.dumps({"status": "already_rolled_back"}))
        return
    if state.get("status") != "applied":
        raise ValueError("repair is not in applied state")
    contract = read_json(CONTRACT)
    critical = validate_contract(contract)
    applied_hash = state.get("execute_sha256")
    previous_hash = state.get("previous_contract_execute_sha256")
    if critical[CRITICAL_KEY] != applied_hash:
        raise ValueError("contract hash drift prevents rollback")
    if not isinstance(previous_hash, str) or len(previous_hash) != 64:
        raise ValueError("rollback contract hash is invalid")
    if CONFIG.is_file() and CONFIG.read_text(encoding="utf-8").strip() != str(
        LOCAL_OUTPUT
    ):
        raise ValueError("eval-output config drift prevents rollback")
    critical[CRITICAL_KEY] = previous_hash
    atomic_json(CONTRACT, contract)
    CONFIG.unlink(missing_ok=True)
    state["status"] = "rolled_back"
    state["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(REPAIR_STATE, state)
    print(json.dumps({"status": "rolled_back"}))


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument("--execute-sha")
    parser.add_argument("--sequence-sha")
    args = parser.parse_args()
    if args.apply:
        if not args.execute_sha or not args.sequence_sha:
            parser.error("--apply requires --execute-sha and --sequence-sha")
        apply_repair(args.execute_sha, args.sequence_sha)
    else:
        rollback_repair()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
