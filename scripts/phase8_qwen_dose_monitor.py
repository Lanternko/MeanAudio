#!/usr/bin/env python3
"""Read-only deterministic monitor for the Phase-8 Qwen caption-dose chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_STATE = Path("/home/kojiek/logs/phase8_qwen_dose_monitor")
DEFAULT_RUN = Path("/home/kojiek/exps_nvme/phase8_qwen_official_matched_dose")
LIVE_REPO = Path(__file__).resolve().parents[1]
IMPLEMENTATION_RELATIVE_PATHS = (
    "docs/experiments/phase8_qwen_dose_contract.json",
    "docs/experiments/phase8_qwen_dose_luna_prompt.md",
    "docs/experiments/phase8_qwen_dose_sol_approval_prompt.md",
    "docs/experiments/phase8_qwen_dose_sol_incident_prompt.md",
    "scripts/phase8_qwen_dose_queue.py",
    "scripts/analysis/phase8_qwen_dose_audit.py",
    "scripts/analysis/phase8_qwen_dose_paired_report.py",
    "scripts/analysis/phase8_qwen_dose_provenance.py",
    "scripts/analysis/phase8_q_clap_bootstrap.py",
    "scripts/phase8_qwen_dose_monitor.py",
    "scripts/phase8_qwen_dose_approval.schema.json",
    "scripts/phase8_qwen_sol_verdict.schema.json",
    "scripts/phase8_qwen_parent_completion_gate.py",
    "scripts/phase8_qwen_dose_luna_loop.sh",
    "scripts/schedule_phase8_qwen_dose_after_20k.sh",
    "scripts/tests/selftest_phase8_qwen_dose.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_payload(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def active_lines() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "phase8_qwen_dose_queue|phase8_qwen_dose_(control|qwen)"],
        capture_output=True, text=True, check=False,
    )
    own = str(os.getpid())
    return [
        line for line in result.stdout.splitlines()
        if own not in line and "pgrep -af" not in line and "phase8_qwen_dose_monitor" not in line
    ]


def checkpoint(path: Path, target: int) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    if datetime.now().timestamp() - path.stat().st_mtime < 120:
        return {"path": str(path), "status": "writing_or_recent"}
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        iteration = state.get("it")
        finite = True
        for root_name in ("weights", "ema", "optimizer"):
            stack = [state.get(root_name, {})]
            while stack:
                value = stack.pop()
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    finite = False
                    break
                if isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    stack.extend(value)
        return {
            "path": str(path), "status": "passed" if finite else "failed",
            "iteration": iteration, "expected_iteration": target, "finite": finite,
        }
    except Exception as exc:
        return {"path": str(path), "status": "failed", "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", required=True)
    parser.add_argument("--expect-active", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lines = active_lines()
    manifest = json_payload(args.state_dir / "execution_manifest.json") or {}
    running_steps = [
        name for name, value in manifest.get("steps", {}).items()
        if value.get("status") == "running"
    ]
    complete = (json_payload(args.run_root / "100k/paired_final_report.json") or {}).get("status") == "passed"
    root_free = shutil.disk_usage(Path("/")).free
    hdd_free = shutil.disk_usage(Path("/mnt/HDD")).free
    issues: list[str] = []
    if root_free < 50 * 1024**3 or hdd_free < 50 * 1024**3:
        issues.append(f"disk floor violated root={root_free}, hdd={hdd_free}")
    if args.expect_active and not complete and not lines:
        issues.append("dose queue expected active but no process is present")
    if len([line for line in lines if "torchrun" in line]) > 1:
        issues.append("duplicate dose torchrun processes")
    contract_hash = sha256_file(args.contract)
    if manifest and manifest.get("contract_sha256") != contract_hash:
        issues.append("execution manifest contract hash drift")
    if manifest:
        current_implementation = {
            relative: sha256_file(LIVE_REPO / relative)
            for relative in IMPLEMENTATION_RELATIVE_PATHS
        }
        if manifest.get("implementation_sha256") != current_implementation:
            issues.append("execution implementation hash drift")
    snapshots: dict[str, Any] = {}
    for label, target in (("50k", 650000), ("100k", 700000)):
        snapshots[label] = {}
        for arm in ("control", "qwen"):
            eid = f"phase8_qwen_dose_{arm}_{label}"
            path = args.run_root / label / arm / f"{eid}_ckpt_last.pth"
            snapshots[label][arm] = checkpoint(path, target)
            snap = snapshots[label][arm]
            if snap.get("status") == "failed":
                issues.append(f"invalid checkpoint {label}/{arm}: {snap.get('error', 'non-finite')}")
            if snap.get("status") == "passed" and snap.get("iteration") != target:
                issues.append(f"checkpoint iteration drift {label}/{arm}={snap.get('iteration')}")
    state = {
        "schema_version": 1,
        "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not issues else "failed",
        "completed": complete,
        "contract_sha256": contract_hash,
        "active_process_lines": lines,
        "running_steps": running_steps,
        "checkpoints": snapshots,
        "root_free_bytes": root_free,
        "hdd_free_bytes": hdd_free,
        "issues": issues,
        "governance": {
            "read_only_monitor": True,
            "metric_thresholds_are_not_retrain_gates": True,
            "repair_requires_codex_review": True,
            "stop_change_relaunch_requires_sol_high": True,
        },
    }
    state_path = args.state_dir / "state.json"
    temp = state_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, state_path)
    if issues:
        proposal = args.repo_root / "proposals/phase8_qwen_dose_monitor_latest.json"
        proposal.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "phase8_qwen_dose_repair_proposal",
            "status": "proposal_only", "created_utc": state["heartbeat_utc"],
            "issues": issues, "requires_codex_review": True,
            "requires_sol_high_for_stop_change_relaunch": True,
        }
        proposal.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state["repair_proposal"] = str(proposal)
        temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, state_path)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
