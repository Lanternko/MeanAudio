#!/usr/bin/env python3
"""Unified progress watcher for the MeanAudio experiment chain.

Tracks:
  1) phase8_legacy_repro (S2 → MusicCaps eval → strict audit)
  2) schedule → phase8_catalog_matched_noq (S1 → S2 → eval)

Emits:
  - JSON status: ~/logs/meanaudio_chain_watch/status.json
  - Human log:   ~/logs/meanaudio_chain_watch/watch.log
  - stdout: ALERT / PHASE / HEARTBEAT lines only (for external monitors)

Does not kill training (unlike the fail-closed legacy monitor). This is a
read-only observer for progress, loss health, and unexpected conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "meanaudio_chain_watch"
STATUS_FILE = STATE_DIR / "status.json"
WATCH_LOG = STATE_DIR / "watch.log"
ALERT_FILE = STATE_DIR / "ALERT.json"

LEGACY_GUARD = LOG_ROOT / "phase8_legacy_repro_guard"
LEGACY_STATE = LEGACY_GUARD / "state.json"
LEGACY_AUDIT = LEGACY_GUARD / "final_audit_loop_status.json"
LEGACY_FINAL_AUDIT = LEGACY_GUARD / "FINAL_AUDIT.json"
LEGACY_ALERT = LEGACY_GUARD / "ALERT.json"
NEXT_EXP = LEGACY_GUARD / "next_experiment_catalog_matched_noq.json"
SCHEDULE_LOG = LOG_ROOT / "schedule_catalog_matched_noq_after_legacy.log"

# (name, log path, target_iter or None, stage label)
TRACKED_LOGS = [
    ("legacy_s2", LOG_ROOT / "phase8_legacy_repro_stage2_200000.log", 600_000, "legacy/S2"),
    ("legacy_s2_eval", LOG_ROOT / "phase8_legacy_repro_stage2_200000_musiccaps_eval.log", None, "legacy/eval"),
    ("legacy_pipeline", LOG_ROOT / "phase8_legacy_repro_pipeline.log", None, "legacy/pipeline"),
    ("noq_gate", LOG_ROOT / "phase8_catalog_matched_noq_medium_gate.log", 200, "noq/gate"),
    ("noq_pipeline", LOG_ROOT / "phase8_catalog_matched_noq_pipeline.log", None, "noq/pipeline"),
    ("noq_s1", LOG_ROOT / "phase8_catalog_matched_noq_stage1_400000.log", 400_000, "noq/S1"),
    ("noq_s2", LOG_ROOT / "phase8_catalog_matched_noq_stage2_200000.log", 600_000, "noq/S2"),
    ("noq_s2_eval", LOG_ROOT / "phase8_catalog_matched_noq_stage2_200000_musiccaps_eval.log", None, "noq/eval"),
]

GATE_SENTINEL = Path("/home/kojiek/logs/phase8_legacy_repro_guard/noq_medium_gate_PASSED.json")

ITER_RE = re.compile(r"\bit\s+(\d+):")
LOSS_RE = re.compile(r"\bloss:\s*([^,\s]+)", re.IGNORECASE)
GRAD_RE = re.compile(r"\bgrad_norm:\s*([^,\s]+)", re.IGNORECASE)
AVG_TIME_RE = re.compile(r"avg_time:([0-9.]+)")
REMAINING_RE = re.compile(r"remaining:([^,]+)")
ETA_RE = re.compile(r"eta:([^,]+)")
LR_RE = re.compile(r"\blr:\s*([^,\s]+)", re.IGNORECASE)

HARD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"Error occurred at iteration",
        r"ProcessExitedException",
        r"ChildFailedError",
        r"NCCL.*(?:error|failed)",
        r"Segmentation fault",
        r"RuntimeError",
        r"Traceback \(most recent call last\)",
        r"Killed",
        r"\[FAIL\]",
        r"NaN",
        r"nan",
        r"inf\b",
    )
]

# Expected healthy loss band for MeanFlow S2 (historical ~0.98–1.0)
LOSS_LO = 0.5
LOSS_HI = 2.5
GRAD_HI = 100.0
# S2 historical plateau ~0.986; flag large spikes relative to short baseline
LOSS_SPIKE_ABS = 0.15
STALE_TRAIN_SEC = 900  # 15 min no new train log line while process claims alive
STALE_EVAL_SEC = 1800
HEARTBEAT_EVERY_SEC = 900  # 15 min
MIN_FREE_ROOT_GB = 20
MIN_FREE_HDD_GB = 10


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def tail_text(path: Path, size: int = 256 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None


def parse_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_train_metrics(text: str) -> dict[str, Any]:
    """Parse last training progress line from log tail."""
    lines = [ln for ln in text.splitlines() if "it " in ln and "loss:" in ln.lower()]
    if not lines:
        return {}
    line = lines[-1]
    it_m = ITER_RE.search(line)
    loss_m = LOSS_RE.search(line)
    grad_m = GRAD_RE.search(line)
    avg_m = AVG_TIME_RE.search(line)
    rem_m = REMAINING_RE.search(line)
    eta_m = ETA_RE.search(line)
    lr_m = LR_RE.search(line)
    return {
        "line": line.strip()[-240:],
        "iteration": int(it_m.group(1)) if it_m else None,
        "loss": parse_float(loss_m.group(1) if loss_m else None),
        "grad_norm": parse_float(grad_m.group(1) if grad_m else None),
        "avg_time_s": parse_float(avg_m.group(1) if avg_m else None),
        "remaining": rem_m.group(1).strip() if rem_m else None,
        "eta": eta_m.group(1).strip() if eta_m else None,
        "lr": parse_float(lr_m.group(1) if lr_m else None),
    }


def scan_hard_errors(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in HARD_PATTERNS:
        match = pattern.search(text)
        if match:
            # grab surrounding line
            for line in text.splitlines()[::-1]:
                if pattern.search(line):
                    hits.append(line.strip()[:300])
                    break
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:8]


def pgrep_patterns(patterns: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,cmd", "--no-headers"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    lines = []
    for line in result.stdout.splitlines():
        for pat in patterns:
            if pat in line and "watch_meanaudio_chain" not in line:
                lines.append(line.strip()[:220])
                break
    return lines[:12]


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"error": result.stderr.strip() or "nvidia-smi failed"}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return {"raw": result.stdout.strip()}
        return {
            "util_pct": float(parts[0]),
            "mem_used_mib": float(parts[1]),
            "mem_total_mib": float(parts[2]),
            "temp_c": float(parts[3]),
        }
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def disk_free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def tmux_sessions() -> list[str]:
    try:
        result = subprocess.run(
            ["tmux", "ls", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    except OSError:
        return []


def classify_active_phase(
    processes: list[str],
    logs: dict[str, Any],
    legacy_state: dict[str, Any],
    next_exp: dict[str, Any],
) -> str:
    joined = "\n".join(processes)
    if "phase8_legacy_repro" in joined:
        return "legacy_training"
    if "phase8_catalog_matched_noq_medium_gate" in joined or "catalog_matched_noq_medium_gate" in joined:
        return "noq_medium_gate"
    if "phase8_catalog_matched_noq" in joined:
        # distinguish S1 vs S2 via exp_id in cmd
        if "stage1" in joined:
            return "noq_s1_training"
        if "stage2" in joined:
            return "noq_s2_training"
        return "noq_training"
    nx = next_exp.get("status", "")
    if nx == "GATE_RUNNING":
        return "noq_medium_gate"
    if nx == "GATE_FAILED":
        return "noq_gate_failed"
    if "eval.py" in joined or "phase4_eval" in joined:
        if "legacy" in joined or legacy_state.get("phase") == "TRAINING":
            return "legacy_or_eval"
        return "eval"
    guard = legacy_state.get("phase", "UNKNOWN")
    if guard == "DONE":
        audit = read_json(LEGACY_AUDIT).get("phase", "")
        if audit in ("WAITING", "AUDITING"):
            return "legacy_auditing"
        if audit == "PASSED" or read_json(LEGACY_FINAL_AUDIT).get("status") == "passed":
            nx = next_exp.get("status", "")
            if nx in ("WAITING", "LAUNCHING"):
                return "handoff_to_noq"
            if nx == "RUNNING":
                return "noq_starting"
            if nx == "BLOCKED":
                return "handoff_blocked"
            return "legacy_done"
        if audit == "FAILED":
            return "legacy_audit_failed"
    if guard == "FAILED":
        return "legacy_failed"
    if guard == "TRAINING":
        # process gone but state not updated yet
        return "legacy_training_stale_or_transition"
    return f"idle_or_unknown(guard={guard})"


def assess_health(
    phase: str,
    metrics: dict[str, Any],
    log_errors: list[str],
    gpu: dict[str, Any],
    processes: list[str],
    log_age_sec: float | None,
    prev_loss: float | None,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    for err in log_errors:
        severity = "hard"
        if re.search(r"\bnan\b|\binf\b", err, re.I) and "loss" not in err.lower():
            # common false positives in stack traces; still report as hard if Error present
            pass
        issues.append({"severity": severity, "code": "log_error", "detail": err})

    loss = metrics.get("loss")
    grad = metrics.get("grad_norm")
    if loss is not None:
        if loss < LOSS_LO or loss > LOSS_HI:
            issues.append(
                {
                    "severity": "hard",
                    "code": "loss_out_of_band",
                    "detail": f"loss={loss:.6f} outside [{LOSS_LO},{LOSS_HI}]",
                }
            )
        if prev_loss is not None and abs(loss - prev_loss) > LOSS_SPIKE_ABS:
            issues.append(
                {
                    "severity": "soft",
                    "code": "loss_spike",
                    "detail": f"loss jumped {prev_loss:.6f} → {loss:.6f} (Δ={loss-prev_loss:+.6f})",
                }
            )
    if grad is not None and grad > GRAD_HI:
        issues.append(
            {
                "severity": "hard",
                "code": "grad_explosion",
                "detail": f"grad_norm={grad:.4f} > {GRAD_HI}",
            }
        )

    if phase.endswith("training") or phase in (
        "legacy_training",
        "noq_s1_training",
        "noq_s2_training",
        "noq_training",
    ):
        if not processes:
            issues.append(
                {
                    "severity": "hard",
                    "code": "process_missing",
                    "detail": f"phase={phase} but no matching train process",
                }
            )
        if log_age_sec is not None and log_age_sec > STALE_TRAIN_SEC:
            issues.append(
                {
                    "severity": "hard",
                    "code": "train_log_stale",
                    "detail": f"train log not updated for {log_age_sec:.0f}s (>{STALE_TRAIN_SEC}s)",
                }
            )
        util = gpu.get("util_pct")
        mem = gpu.get("mem_used_mib")
        if isinstance(util, (int, float)) and isinstance(mem, (int, float)):
            if util < 5 and mem < 2000:
                issues.append(
                    {
                        "severity": "hard",
                        "code": "gpu_idle_during_train",
                        "detail": f"GPU util={util}% mem={mem}MiB while training expected",
                    }
                )
            elif util < 15 and mem > 8000:
                issues.append(
                    {
                        "severity": "soft",
                        "code": "gpu_util_low",
                        "detail": f"GPU util={util}% (mem={mem}MiB) — possible stall/data bottleneck",
                    }
                )

    free_root = disk_free_gb("/")
    free_hdd = disk_free_gb("/mnt/HDD")
    if free_root < MIN_FREE_ROOT_GB:
        issues.append(
            {
                "severity": "hard",
                "code": "disk_root_low",
                "detail": f"root free {free_root:.1f}G < {MIN_FREE_ROOT_GB}G",
            }
        )
    if free_hdd < MIN_FREE_HDD_GB:
        issues.append(
            {
                "severity": "soft",
                "code": "disk_hdd_low",
                "detail": f"HDD free {free_hdd:.1f}G < {MIN_FREE_HDD_GB}G",
            }
        )

    if LEGACY_ALERT.exists() and LEGACY_ALERT.stat().st_size > 2:
        issues.append(
            {
                "severity": "hard",
                "code": "legacy_guard_alert",
                "detail": f"legacy ALERT present: {LEGACY_ALERT}",
            }
        )

    return issues


def append_watch_log(line: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a") as handle:
        handle.write(line + "\n")


def emit(kind: str, message: str, also_stdout: bool = True) -> None:
    line = f"[{now_iso()}] {kind}: {message}"
    append_watch_log(line)
    if also_stdout:
        print(line, flush=True)


def collect_snapshot(prev: dict[str, Any] | None) -> dict[str, Any]:
    processes = pgrep_patterns(
        [
            "phase8_legacy_repro",
            "phase8_catalog_matched_noq",
            "catalog_matched_noq_medium_gate",
            "train.py",
            "eval.py",
            "phase4_eval",
            "monitor_phase8_legacy",
            "schedule_catalog_matched_noq",
            "wait_and_audit_phase8",
            "run_phase8_legacy_guarded",
        ]
    )
    gpu = gpu_snapshot()
    legacy_state = read_json(LEGACY_STATE)
    next_exp = read_json(NEXT_EXP)
    audit = read_json(LEGACY_AUDIT)
    final_audit = read_json(LEGACY_FINAL_AUDIT)

    log_info: dict[str, Any] = {}
    active_metrics: dict[str, Any] = {}
    active_log_key = None
    active_log_age = None
    all_errors: list[str] = []

    for key, path, target, label in TRACKED_LOGS:
        mtime = file_mtime(path)
        age = (time.time() - mtime) if mtime else None
        text = tail_text(path) if path.exists() else ""
        metrics = parse_train_metrics(text) if text else {}
        errors = scan_hard_errors(text) if text else []
        progress = None
        if metrics.get("iteration") is not None and target:
            progress = round(100.0 * metrics["iteration"] / target, 2)
        entry = {
            "path": str(path),
            "exists": path.exists(),
            "mtime": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None,
            "age_sec": round(age, 1) if age is not None else None,
            "target_iter": target,
            "progress_pct": progress,
            "metrics": metrics,
            "errors": errors,
            "label": label,
        }
        log_info[key] = entry
        all_errors.extend(errors)
        # Prefer freshest training log with metrics
        if metrics.get("iteration") is not None and age is not None:
            if active_log_age is None or age < active_log_age:
                active_log_age = age
                active_log_key = key
                active_metrics = metrics

    phase = classify_active_phase(processes, log_info, legacy_state, next_exp)

    # Prefer phase-specific log age for staleness
    log_age_for_health = active_log_age
    if phase == "legacy_training" and log_info["legacy_s2"]["age_sec"] is not None:
        log_age_for_health = log_info["legacy_s2"]["age_sec"]
        active_metrics = log_info["legacy_s2"]["metrics"] or active_metrics
    elif phase == "noq_s1_training" and log_info["noq_s1"]["age_sec"] is not None:
        log_age_for_health = log_info["noq_s1"]["age_sec"]
        active_metrics = log_info["noq_s1"]["metrics"] or active_metrics
    elif phase in ("noq_s2_training", "noq_training") and log_info["noq_s2"]["age_sec"] is not None:
        log_age_for_health = log_info["noq_s2"]["age_sec"]
        active_metrics = log_info["noq_s2"]["metrics"] or active_metrics

    prev_loss = None
    if prev and isinstance(prev.get("active_metrics"), dict):
        prev_loss = prev["active_metrics"].get("loss")

    issues = assess_health(
        phase=phase,
        metrics=active_metrics,
        log_errors=all_errors,
        gpu=gpu,
        processes=processes,
        log_age_sec=log_age_for_health,
        prev_loss=prev_loss,
    )

    # Progress summary string
    it = active_metrics.get("iteration")
    loss = active_metrics.get("loss")
    grad = active_metrics.get("grad_norm")
    rem = active_metrics.get("remaining")
    if it is not None and active_log_key:
        target = next((t for k, _, t, _ in TRACKED_LOGS if k == active_log_key), None)
        pct = f"{100*it/target:.1f}%" if target else "?"
        progress_str = (
            f"{active_log_key} it={it}/{target or '?'} ({pct}) "
            f"loss={loss} grad={grad} rem={rem}"
        )
    else:
        progress_str = f"no active train metrics (phase={phase})"

    snapshot = {
        "updated_at": now_iso(),
        "phase": phase,
        "progress": progress_str,
        "active_log": active_log_key,
        "active_metrics": active_metrics,
        "gpu": gpu,
        "disk": {
            "root_free_gb": round(disk_free_gb("/"), 1),
            "hdd_free_gb": round(disk_free_gb("/mnt/HDD"), 1),
        },
        "tmux": tmux_sessions(),
        "legacy_guard": {
            "phase": legacy_state.get("phase"),
            "detail": legacy_state.get("detail"),
            "updated_at": legacy_state.get("updated_at"),
        },
        "legacy_audit": {
            "phase": audit.get("phase"),
            "detail": audit.get("detail"),
            "final_status": final_audit.get("status"),
            "final_issues": final_audit.get("issues"),
        },
        "next_experiment": {
            "status": next_exp.get("status"),
            "detail": next_exp.get("detail"),
            "experiment": next_exp.get("experiment"),
            "flow": next_exp.get("flow"),
        },
        "noq_medium_gate": read_json(GATE_SENTINEL),
        "processes": processes,
        "logs": {
            k: {
                "exists": v["exists"],
                "age_sec": v["age_sec"],
                "progress_pct": v["progress_pct"],
                "iteration": (v["metrics"] or {}).get("iteration"),
                "loss": (v["metrics"] or {}).get("loss"),
                "grad_norm": (v["metrics"] or {}).get("grad_norm"),
                "remaining": (v["metrics"] or {}).get("remaining"),
                "errors": v["errors"],
            }
            for k, v in log_info.items()
        },
        "issues": issues,
        "healthy": not any(i["severity"] == "hard" for i in issues),
    }
    return snapshot


def format_heartbeat(snap: dict[str, Any]) -> str:
    gpu = snap.get("gpu") or {}
    m = snap.get("active_metrics") or {}
    return (
        f"phase={snap['phase']} | {snap['progress']} | "
        f"gpu={gpu.get('util_pct', '?')}%/{gpu.get('mem_used_mib', '?')}MiB "
        f"T={gpu.get('temp_c', '?')}C | "
        f"disk_root={snap['disk']['root_free_gb']}G hdd={snap['disk']['hdd_free_gb']}G | "
        f"guard={snap['legacy_guard'].get('phase')} "
        f"audit={snap['legacy_audit'].get('phase')} "
        f"next={snap['next_experiment'].get('status')} | "
        f"healthy={snap['healthy']} issues={len(snap['issues'])} | "
        f"loss={m.get('loss')} grad={m.get('grad_norm')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=HEARTBEAT_EVERY_SEC,
        help="Stdout heartbeat interval seconds",
    )
    parser.add_argument("--once", action="store_true", help="Single snapshot then exit")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    emit("START", f"chain watcher interval={args.interval}s heartbeat={args.heartbeat_every}s")

    prev: dict[str, Any] | None = None
    last_phase: str | None = None
    last_heartbeat = 0.0
    last_hard_fp: str | None = None

    while True:
        snap = collect_snapshot(prev)
        atomic_json(STATUS_FILE, snap)

        # Always append compact line to watch.log (not all to stdout)
        hb = format_heartbeat(snap)
        append_watch_log(f"[{snap['updated_at']}] STATUS: {hb}")

        phase = snap["phase"]
        if phase != last_phase:
            emit("PHASE", f"{last_phase} → {phase} | {snap['progress']}")
            last_phase = phase

        hard = [i for i in snap["issues"] if i["severity"] == "hard"]
        soft = [i for i in snap["issues"] if i["severity"] == "soft"]
        if hard:
            fp = "|".join(sorted(i["code"] + ":" + i["detail"][:80] for i in hard))
            if fp != last_hard_fp:
                atomic_json(
                    ALERT_FILE,
                    {
                        "updated_at": now_iso(),
                        "phase": phase,
                        "issues": hard,
                        "progress": snap["progress"],
                    },
                )
                for issue in hard:
                    emit("ALERT", f"[{issue['code']}] {issue['detail']}")
                last_hard_fp = fp
        else:
            if ALERT_FILE.exists():
                try:
                    ALERT_FILE.unlink()
                except OSError:
                    pass
            last_hard_fp = None

        if soft:
            for issue in soft:
                # soft only to watch.log + occasional stdout if new
                append_watch_log(
                    f"[{snap['updated_at']}] SOFT: [{issue['code']}] {issue['detail']}"
                )

        now_t = time.time()
        if now_t - last_heartbeat >= args.heartbeat_every:
            emit("HEARTBEAT", hb)
            last_heartbeat = now_t

        prev = snap
        if args.once:
            # always print one full heartbeat on --once
            emit("HEARTBEAT", hb)
            raise SystemExit(0 if snap["healthy"] else 2)

        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
