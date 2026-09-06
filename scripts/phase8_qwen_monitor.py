#!/usr/bin/env python3
"""Read-only three-hour monitor for the official-Qwen matched queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path("/home/kojiek/logs/phase8_qwen_official_matched_monitor")
DEFAULT_RUN_ROOT = Path("/home/kojiek/exps_nvme/phase8_qwen_official_matched")
DEFAULT_CONTRACT = ROOT / "docs/experiments/phase8_qwen_official_matched_contract.json"
ARMS = {
    "control": "phase8_qwen_official_matched_control_20k",
    "qwen": "phase8_qwen_official_matched_qwen_20k",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processes() -> list[str]:
    result = subprocess.run(
        [
            "pgrep", "-af",
            "phase8_qwen_probe_queue|phase8_qwen_full_npz|phase8_qwen_probe_eval|"
            "phase8_qwen_official_matched_(control|qwen)_20k",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    own = str(os.getpid())
    return [
        line for line in result.stdout.splitlines()
        if own not in line and "pgrep -af" not in line
    ]


def tail(path: Path, limit: int = 128 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def checkpoint_snapshot(path: Path, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    stat = path.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if (
        previous
        and previous.get("status") == "passed"
        and previous.get("file_signature") == signature
    ):
        return {**previous, "cached_unchanged": True}
    if time.time_ns() - stat.st_mtime_ns < 120 * 1_000_000_000:
        return {
            "status": "updating",
            "path": str(path),
            "file_signature": signature,
            "reason": "checkpoint has not been stable for 120 seconds",
        }
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        iteration = state.get("it")
        finite = True
        nonfinite_key = None
        for root_name in ("weights", "optimizer", "ema"):
            value = state.get(root_name)
            stack: list[tuple[str, Any]] = [(root_name, value)]
            while stack:
                label, child = stack.pop()
                if torch.is_tensor(child):
                    if not torch.isfinite(child).all():
                        finite = False
                        nonfinite_key = label
                        break
                elif isinstance(child, dict):
                    stack.extend((f"{label}.{key}", sub) for key, sub in child.items())
            if not finite:
                break
        return {
            "status": "passed" if finite else "nonfinite",
            "path": str(path),
            "iteration": iteration,
            "finite": finite,
            "nonfinite_key": nonfinite_key,
            "sha256": sha256(path),
            "file_signature": signature,
            "cached_unchanged": False,
        }
    except Exception as exc:  # corruption, partial write, or incompatible pickle
        return {"status": "corrupt", "path": str(path), "error": repr(exc)}


def snapshot(args: argparse.Namespace, previous: dict[str, Any] | None) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    contract_hash = sha256(args.contract) if args.contract.is_file() else None
    if contract_hash is None:
        issues.append(f"missing contract: {args.contract}")
    elif previous and previous.get("contract_sha256") not in (None, contract_hash):
        issues.append("contract hash changed")

    active_lines = processes()
    for arm, exp_id in ARMS.items():
        launches = [line for line in active_lines if "torchrun" in line and exp_id in line]
        if len(launches) > 1:
            issues.append(f"duplicate {arm} torchrun launches: {launches}")
    root_free = shutil.disk_usage(Path("/")).free
    hdd_free = shutil.disk_usage(Path("/mnt/HDD")).free
    if root_free < 50 * 1024**3 or hdd_free < 50 * 1024**3:
        issues.append(f"disk floor violated root={root_free}, hdd={hdd_free}")
    elif root_free < 60 * 1024**3 or hdd_free < 60 * 1024**3:
        warnings.append(f"disk pressure root={root_free}, hdd={hdd_free}")

    arms: dict[str, Any] = {}
    for arm, exp_id in ARMS.items():
        run_dir = args.run_root / arm
        ckpt = run_dir / f"{exp_id}_ckpt_last.pth"
        log_path = args.log_root / "phase8_qwen_official_matched_queue.log"
        previous_ckpt = previous.get("arms", {}).get(arm, {}).get("checkpoint") if previous else None
        ckpt_info = checkpoint_snapshot(ckpt, previous_ckpt)
        telemetry = tail(log_path)
        telemetry_nonfinite = bool(re.search(r"\b(?:nan|inf|infinity)\b", telemetry, flags=re.I))
        if ckpt_info.get("status") in {"corrupt", "nonfinite"}:
            issues.append(f"{arm} checkpoint {ckpt_info['status']}")
        elif telemetry_nonfinite and ckpt_info.get("status") == "passed":
            # A single AMP telemetry NaN is a warning if the latest checkpoint
            # is finite.  It is not a stop/retrain signal by itself.
            warnings.append(f"{arm} logged non-finite telemetry; latest checkpoint is finite")
        arms[arm] = {
            "exp_id": exp_id,
            "run_dir": str(run_dir),
            "checkpoint": ckpt_info,
            "log": str(log_path),
            "telemetry_nonfinite": telemetry_nonfinite,
            "active_process": any(exp_id in line for line in active_lines),
        }

    final_report = args.run_root / "paired_final_report.json"
    completed = False
    if final_report.is_file():
        try:
            completed = json.loads(final_report.read_text(encoding="utf-8")).get("status") == "passed"
        except Exception:
            completed = False
    if args.expect_active and not active_lines and not completed:
        issues.append("expected an active queue/cache/train/eval process, but none is present")
    state = {
        "schema_version": 1,
        "heartbeat_utc": now(),
        "contract_sha256": contract_hash,
        "active_process_lines": active_lines,
        "arms": arms,
        "root_free_bytes": root_free,
        "hdd_free_bytes": hdd_free,
        "completed": completed,
        "issues": issues,
        "warnings": warnings,
        "status": "failed" if issues else "warning" if warnings else "passed",
        "governance": {
            "read_only_monitor": True,
            "persistent_nonfinite_only": True,
            "transient_telemetry_nan_is_warning": True,
            "repair_proposals_require_codex_review": True,
            "stop_change_relaunch_requires_sol_high_approval": True,
            "monitor_never_stops_or_relaunches_processes": True,
        },
    }
    if issues:
        proposal = args.repo_root / "proposals" / "phase8_qwen_monitor_latest.json"
        atomic_json(
            proposal,
            {
                "kind": "phase8_qwen_repair_proposal",
                "created_utc": state["heartbeat_utc"],
                "status": "proposal_only",
                "requires_codex_review": True,
                "requires_sol_high_approval_for_stop_change_relaunch": True,
                "issues": issues,
                "suggested_action": "pause and review evidence; do not change or relaunch automatically",
            },
        )
        state["repair_proposal"] = str(proposal)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration-hours", type=float, default=3.0)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=Path("/home/kojiek/logs"))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--expect-active", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.interval_seconds > 60:
        raise SystemExit("interval must be 1..60 seconds")
    state_path = args.state_dir / "state.json"
    previous: dict[str, Any] | None = None
    deadline = time.monotonic() + max(0.0, args.duration_hours) * 3600
    while True:
        state = snapshot(args, previous)
        atomic_json(state_path, state)
        print(json.dumps(state, indent=2, sort_keys=True), flush=True)
        previous = state
        if args.once or time.monotonic() >= deadline:
            return 0 if state["status"] != "failed" else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
