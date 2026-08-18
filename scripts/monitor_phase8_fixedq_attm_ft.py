#!/usr/bin/env python3
"""Read-only monitor for the Phase-8 Fixed-Q / matched-NoQ sequence.

Distinguishes:
  - review-only: single/isolated AMP grad NaN that immediately recovers
  - hard incident: persistent/dense nonfinite grad, nonfinite loss/LR,
    OOM/NCCL/segfault/traceback, stale process/log (two-observation for
    stale/GPU/process), disk <50 GB.

Does not kill, start, or mutate training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/kojiek/MeanAudio")
LOG = Path("/home/kojiek/logs")
STATE = LOG / "phase8_fixedq_attm_monitor"
STATUS = STATE / "status.json"
ALERT = STATE / "ALERT.json"
TMUX = "p8_fixedq_attm"
# Order: fixedq9 first, then matched-noq.
ARMS = [
    ("fixedq9", "phase8_fixedq9_prior_ft100k"),
    ("noq", "phase8_matched_noq_ft100k"),
]
ITER = re.compile(r"\bit\s+(\d+):")
LOSS = re.compile(
    r"loss:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I
)
GRAD = re.compile(
    r"grad_norm:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I
)
LR_RE = re.compile(r"\blr:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))", re.I)
HARD_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"ChildFailedError",
        r"NCCL.*(?:error|failed)",
        r"segmentation fault",
        r"Traceback \(most recent call last\)",
        r"ProcessExitedException",
    )
]


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def tail(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        n = handle.tell()
        handle.seek(max(0, n - limit))
        return handle.read().decode(errors="replace")


def number(match: re.Match[str] | None) -> float | None:
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def read_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                    out[key.strip()] = float(value)
    return out


def load_prior_observations() -> dict[str, Any]:
    if not STATUS.is_file():
        return {}
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    prior = load_prior_observations()
    prior_observations = set((prior.get("issues") or []) + (prior.get("review") or []))

    processes = run(
        [
            "pgrep",
            "-af",
            r"phase8_(fixedq9_prior|matched_noq)_ft100k|train_pipeline_phase8_fixedq_attm|sequence_phase8_fixedq_attm",
        ]
    )
    tmux = run(["tmux", "list-sessions", "-F", "#{session_name}"]).splitlines()

    arms: dict[str, Any] = {}
    active_mode = None
    active_prefix = None
    active_log: Path | None = None
    for mode, prefix in ARMS:
        exp = f"{prefix}_stage2_ft100000"
        audit_path = STATE / f"{prefix}_FINAL_AUDIT.json"
        audited = json.loads(audit_path.read_text()) if audit_path.is_file() else None
        eval_label = "q9" if mode == "fixedq9" else "noq"
        metrics_path = (
            ROOT
            / "eval_output/metrics"
            / f"{exp}_musiccaps_{eval_label}"
            / "metrics.txt"
        )
        arms[mode] = {
            "prefix": prefix,
            "complete": bool(audited and audited.get("status") == "passed"),
            "final_audit": audited,
            "metrics": {eval_label: read_metrics(metrics_path)},
        }
        if prefix in processes or exp in processes:
            active_mode, active_prefix = mode, prefix
            active_log = LOG / f"{exp}.log"

    final = STATE / "FINAL_COMPARISON.json"
    if final.is_file():
        phase = "complete"
    elif active_mode:
        exp = f"{active_prefix}_stage2_ft100000"
        eval_label = "q9" if active_mode == "fixedq9" else "noq"
        eval_log = LOG / f"{exp}_musiccaps_{eval_label}_eval.log"
        if eval_log.is_file() and not arms[active_mode]["metrics"].get(eval_label):
            phase = f"{active_mode}_eval"
            active_log = eval_log
        else:
            phase = f"{active_mode}_training"
    elif all(arms[m]["complete"] for m, _ in ARMS):
        phase = "paired_bootstrap_or_finalizing"
    elif arms["fixedq9"]["complete"]:
        phase = "between_arms_awaiting_noq"
    else:
        phase = "queued_or_starting"

    latest: dict[str, Any] = {}
    issues: list[str] = []
    review: list[str] = []
    log_age = None
    grad_health = {
        "nonfinite_trailing": 0,
        "nonfinite_recent_20": 0,
        "nonfinite_recent_100": 0,
        "isolated_amp_overflow": False,
    }

    if active_log and active_log.is_file():
        text = re.sub(r"\x1b\[[0-9;]*m", "", tail(active_log))
        log_age = max(0.0, now.timestamp() - active_log.stat().st_mtime)
        records: list[dict[str, Any]] = []
        for line in text.replace("\r", "\n").splitlines():
            match = ITER.search(line)
            if match:
                records.append(
                    {
                        "iteration": int(match.group(1)),
                        "loss": number(LOSS.search(line)),
                        "grad_norm": number(GRAD.search(line)),
                        "lr": number(LR_RE.search(line)),
                    }
                )
        if records:
            latest = records[-1]
            recent = records[-100:]
            bad = [
                not math.isfinite(x["grad_norm"])
                for x in recent
                if x["grad_norm"] is not None
            ]
            trailing = 0
            for value in reversed(bad):
                if not value:
                    break
                trailing += 1
            recent_20 = sum(bad[-20:]) if bad else 0
            recent_100 = sum(bad)
            grad_health = {
                "nonfinite_trailing": trailing,
                "nonfinite_recent_20": recent_20,
                "nonfinite_recent_100": recent_100,
                "isolated_amp_overflow": bool(
                    recent_100 > 0
                    and trailing < 2
                    and recent_20 < 3
                    and recent_100 < 10
                ),
            }
            if trailing >= 2 or recent_20 >= 3 or recent_100 >= 10:
                issues.append(f"persistent/dense nonfinite grad: {grad_health}")
            elif any(bad):
                review.append(f"isolated recovered AMP overflow: {grad_health}")

            loss = latest.get("loss")
            lr = latest.get("lr")
            if loss is not None and not math.isfinite(loss):
                issues.append("nonfinite latest loss")
            if lr is not None and not math.isfinite(lr):
                issues.append("nonfinite latest lr")

            # Repeated extreme loss / grad (need consecutive hits).
            extreme_loss = 0
            extreme_grad = 0
            for rec in reversed(records[-10:]):
                if rec["loss"] is not None and math.isfinite(rec["loss"]) and rec["loss"] > 5:
                    extreme_loss += 1
                else:
                    break
            for rec in reversed(records[-10:]):
                g = rec["grad_norm"]
                if g is not None and math.isfinite(g) and g > 100:
                    extreme_grad += 1
                else:
                    break
            if extreme_loss >= 3:
                issues.append(f"repeated loss>5 count={extreme_loss}")
            if extreme_grad >= 3:
                issues.append(f"repeated grad_norm>100 count={extreme_grad}")

        found = [p.pattern for p in HARD_PATTERNS if p.search(text)]
        if found:
            issues.append(f"hard runtime signatures: {found}")

        stale_msg = None
        if processes and log_age is not None and log_age > 1200:
            stale_msg = f"active process log stale {log_age:.0f}s"
        # Stale incidents require two observations.
        for msg in (stale_msg,):
            if not msg:
                continue
            if msg in prior_observations or any(
                msg.split()[0] in prev for prev in prior_observations
            ):
                issues.append(msg)
            else:
                review.append(f"first observation (need reconfirm): {msg}")

    # A dead queue must be detected even when no active log exists (for example,
    # a failure in the instant transition between the two arms).
    missing_msg = None
    if phase != "complete" and TMUX not in tmux and not processes:
        missing_msg = "incomplete but queue/training tmux/process absent"
    if missing_msg:
        if missing_msg in prior_observations or any(
            "queue/training tmux/process absent" in prev
            for prev in prior_observations
        ):
            issues.append(missing_msg)
        else:
            review.append(f"first observation (need reconfirm): {missing_msg}")

    root_free = shutil.disk_usage("/").free / 1024**3
    hdd_free = shutil.disk_usage("/mnt/HDD").free / 1024**3
    if root_free < 50:
        issues.append(f"root free below 50 GiB: {root_free:.1f}")
    if hdd_free < 50:
        issues.append(f"HDD free below 50 GiB: {hdd_free:.1f}")

    gpu: dict[str, float] = {}
    raw = run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw:
        try:
            util, used, total, temp = [
                float(x.strip()) for x in raw.splitlines()[0].split(",")
            ]
            gpu = {
                "util_pct": util,
                "mem_used_mib": used,
                "mem_total_mib": total,
                "temp_c": temp,
            }
        except ValueError:
            pass

    comparison = json.loads(final.read_text()) if final.is_file() else None
    status = "incident" if issues else ("review" if review else "healthy")
    progress = None
    if latest.get("iteration") is not None:
        progress = round(100 * (latest["iteration"] - 600000) / 100000, 3)

    payload = {
        "updated_at": now.isoformat(),
        "experiment": "phase8_fixedq_attm_sequence",
        "status": status,
        "phase": phase,
        "active_mode": active_mode,
        "active_prefix": active_prefix,
        "latest": latest,
        "fine_tune_progress_pct": progress,
        "target_iteration": 700000 if active_mode else None,
        "issues": issues,
        "review": review,
        "grad_health": grad_health,
        "log_age_sec": log_age,
        "gpu": gpu,
        "root_free_gb": round(root_free, 1),
        "hdd_free_gb": round(hdd_free, 1),
        "tmux": tmux,
        "processes": processes.splitlines() if processes else [],
        "arms": arms,
        "final_comparison": comparison,
        "targets": {
            "baseline_noq": 0.1888,
            "restoration": "FixedQ9 q9 CLAP >= 0.1900",
            "fixedq_benefit": "paired bootstrap CI95 FixedQ-NoQ lower bound > 0",
            "primary_checkpoint": 700000,
            "no_cherrypick": True,
            "loss_plateau_near_0p98_not_failure": True,
        },
        "policy": {
            "transient_single_amp_grad_nan": "review_only",
            "persistent_nonfinite_grad": "incident",
            "stale_or_process_requires_two_observations": True,
            "do_not_launch_third_training_arm": True,
        },
    }
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATUS)
    if issues:
        alert = {
            "created_at": now.isoformat(),
            "phase": phase,
            "issues": issues,
            "stop_authorized": False,
        }
        tmp = ALERT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(alert, indent=2) + "\n")
        tmp.replace(ALERT)
    elif ALERT.exists():
        ALERT.unlink()

    print(
        f"status={status} phase={phase} mode={active_mode} "
        f"it={latest.get('iteration')}/700000 progress={progress}% "
        f"loss={latest.get('loss')} grad={latest.get('grad_norm')} "
        f"gpu={gpu.get('util_pct')}% root={root_free:.1f}G hdd={hdd_free:.1f}G "
        f"issues={len(issues)} review={len(review)}"
    )
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
