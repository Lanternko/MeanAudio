#!/usr/bin/env python3
"""Read-only monitor for the sequential Real-Q / Shuffled-Q S2 experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "phase8_s2_q_ablation_monitor"
STATUS_FILE = STATE_DIR / "status.json"
ALERT_FILE = STATE_DIR / "ALERT.json"
TMUX = "p8_s2_q_ablation"
ARMS = [
    ("real", "phase8_catalog_matched_s2_realq"),
    ("shuffled", "phase8_catalog_matched_s2_shuffledq"),
]
ITER_RE = re.compile(r"\bit\s+(\d+):")
LOSS_RE = re.compile(r"loss:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I)
GRAD_RE = re.compile(r"grad_norm:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I)
LR_RE = re.compile(r"lr:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I)


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def tail(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def numeric(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            if key.strip() in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                values[key.strip()] = float(raw.strip())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    del args

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    processes = run(["pgrep", "-af", "phase8_catalog_matched_s2_(realq|shuffledq)"])
    tmux_names = run(["tmux", "list-sessions", "-F", "#{session_name}"]).splitlines()

    arm_state: dict[str, Any] = {}
    active_mode: str | None = None
    active_prefix: str | None = None
    active_log: Path | None = None
    for mode, prefix in ARMS:
        exp_id = f"{prefix}_stage2_200000"
        train_log = LOG_ROOT / f"{exp_id}.log"
        metrics = {
            f"q{q}": parse_metrics(
                ROOT / "eval_output/metrics" / f"{exp_id}_musiccaps_q{q}" / "metrics.txt"
            )
            for q in (9, 6)
        }
        final_audit = STATE_DIR / f"{prefix}_FINAL_AUDIT.json"
        audit = json.loads(final_audit.read_text()) if final_audit.is_file() else None
        arm_state[mode] = {
            "prefix": prefix,
            "metrics": metrics,
            "final_audit": audit,
            "complete": bool(audit and audit.get("status") == "passed"),
        }
        if prefix in processes:
            active_mode, active_prefix, active_log = mode, prefix, train_log

    both_complete = all(arm_state[mode]["complete"] for mode, _ in ARMS)
    if both_complete:
        phase = "complete"
    elif active_mode:
        exp_id = f"{active_prefix}_stage2_200000"
        q9_log = LOG_ROOT / f"{exp_id}_musiccaps_q9_eval.log"
        q6_log = LOG_ROOT / f"{exp_id}_musiccaps_q6_eval.log"
        if q6_log.is_file() and not arm_state[active_mode]["metrics"]["q6"]:
            phase = f"{active_mode}_eval_q6"
            active_log = q6_log
        elif q9_log.is_file() and not arm_state[active_mode]["metrics"]["q9"]:
            phase = f"{active_mode}_eval_q9"
            active_log = q9_log
        else:
            phase = f"{active_mode}_s2_training"
    elif arm_state["real"]["complete"]:
        phase = "between_arms_or_shuffled_starting"
    else:
        phase = "queued_or_starting"

    latest: dict[str, Any] = {}
    log_age = None
    issues: list[str] = []
    review: list[str] = []
    grad_health = {
        "nonfinite_trailing": 0,
        "nonfinite_recent_20": 0,
        "nonfinite_recent_100": 0,
    }
    if active_log and active_log.is_file():
        text = re.sub(r"\x1b\[[0-9;]*m", "", tail(active_log))
        log_age = max(0.0, now.timestamp() - active_log.stat().st_mtime)
        records = []
        for line in text.replace("\r", "\n").splitlines():
            match = ITER_RE.search(line)
            if not match:
                continue
            loss = numeric(LOSS_RE.search(line).group(1) if LOSS_RE.search(line) else None)
            grad = numeric(GRAD_RE.search(line).group(1) if GRAD_RE.search(line) else None)
            lr = numeric(LR_RE.search(line).group(1) if LR_RE.search(line) else None)
            records.append({"iteration": int(match.group(1)), "loss": loss, "grad_norm": grad, "lr": lr})
        if records:
            latest = records[-1]
            recent100 = records[-100:]
            nonfinite = [not math.isfinite(r["grad_norm"]) for r in recent100 if r["grad_norm"] is not None]
            trailing = 0
            for bad in reversed(nonfinite):
                if not bad:
                    break
                trailing += 1
            grad_health = {
                "nonfinite_trailing": trailing,
                "nonfinite_recent_20": sum(nonfinite[-20:]),
                "nonfinite_recent_100": sum(nonfinite),
            }
            if trailing >= 2 or sum(nonfinite[-20:]) >= 3 or sum(nonfinite) >= 10:
                issues.append(f"persistent/dense non-finite grad_norm: {grad_health}")
            elif any(nonfinite):
                review.append(f"isolated recovered AMP grad overflow: {grad_health}")
            for key in ("loss", "lr"):
                value = latest.get(key)
                if value is not None and not math.isfinite(value):
                    issues.append(f"non-finite latest {key}: {value}")
        hard_patterns = [
            r"CUDA out of memory", r"ChildFailedError", r"segmentation fault",
            r"Traceback \(most recent call last\)", r"NCCL.*(?:error|failed)",
        ]
        found = [pattern for pattern in hard_patterns if re.search(pattern, text, re.I)]
        if found:
            issues.append(f"hard runtime signature(s): {found}")
        if processes and log_age is not None and log_age > 1200:
            issues.append(f"active process but log stale for {log_age:.0f}s")

    disk_free = shutil.disk_usage("/").free / 1024**3
    if disk_free < 50:
        issues.append(f"root free disk below 50 GiB: {disk_free:.1f}")

    gpu_raw = run([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpu: dict[str, float] = {}
    if gpu_raw:
        try:
            util, used, total, temp = [float(part.strip()) for part in gpu_raw.splitlines()[0].split(",")]
            gpu = {"util_pct": util, "mem_used_mib": used, "mem_total_mib": total, "temp_c": temp}
        except ValueError:
            pass

    # Contracts are checked cheaply here. Full hashes/configs are checked by each final audit.
    contract_status: dict[str, str] = {}
    for mode, prefix in ARMS:
        path = LOG_ROOT / f"{prefix}_contract.json"
        if not path.is_file():
            contract_status[mode] = "pending"
            continue
        try:
            contract = json.loads(path.read_text())
            expected = {
                "prefix": prefix,
                "q_mode": mode,
                "stage1_use_q_conditioning": False,
                "stage2_use_q_conditioning": True,
                "use_text_attention_mask": False,
                "multi_cap": False,
                "stage2_final_iteration": 600000,
                "eval_primary_q": 9,
                "eval_secondary_q": 6,
            }
            drift = [key for key, value in expected.items() if contract.get(key) != value]
            contract_status[mode] = "passed" if not drift else "failed"
            if drift:
                issues.append(f"{mode} contract drift: {drift}")
        except Exception as exc:
            contract_status[mode] = "failed"
            issues.append(f"cannot parse {mode} contract: {exc}")

    if not both_complete and not processes and TMUX not in tmux_names:
        sequence_log = LOG_ROOT / "phase8_s2_q_ablation_sequence.log"
        age = now.timestamp() - sequence_log.stat().st_mtime if sequence_log.is_file() else math.inf
        if age > 1200:
            issues.append("experiment incomplete but tmux/process are absent")

    status = "incident" if issues else ("review" if review else "healthy")
    progress = None
    if latest.get("iteration") is not None:
        progress = round(100 * (latest["iteration"] - 400000) / 200000, 3)
    payload = {
        "updated_at": now.isoformat(),
        "experiment": "phase8_s2_q_ablation_sequence",
        "phase": phase,
        "status": status,
        "active_mode": active_mode,
        "active_prefix": active_prefix,
        "active_log": str(active_log) if active_log else None,
        "latest": latest,
        "target_iteration": 600000 if active_mode else None,
        "stage2_progress_pct": progress,
        "log_age_sec": log_age,
        "gpu": gpu,
        "disk_root_free_gb": round(disk_free, 1),
        "contracts": contract_status,
        "grad_health": grad_health,
        "issues": issues,
        "review": review,
        "arms": arm_state,
        "tmux": tmux_names,
        "processes": processes.splitlines() if processes else [],
        "quality_targets": {
            "baseline_noq": 0.1888,
            "meaningful_gain": "Real-Q q9 >= 0.1938 (+0.005)",
            "historical_best": "Real-Q q9 >= 0.1998",
            "signal_useful": "Real-Q must beat both NoQ and Shuffled-Q",
        },
    }
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATUS_FILE)
    if issues:
        alert = {
            "created_at": now.isoformat(),
            "phase": phase,
            "issues": issues,
            "status_file": str(STATUS_FILE),
            "stop_authorized": False,
        }
        tmp = ALERT_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(alert, indent=2, sort_keys=True) + "\n")
        tmp.replace(ALERT_FILE)
    elif ALERT_FILE.exists():
        ALERT_FILE.unlink()
    print(
        f"status={status} phase={phase} mode={active_mode} "
        f"it={latest.get('iteration')}/600000 progress={progress}% "
        f"loss={latest.get('loss')} grad={latest.get('grad_norm')} "
        f"gpu={gpu.get('util_pct')}% disk={disk_free:.1f}G issues={len(issues)}"
    )
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
