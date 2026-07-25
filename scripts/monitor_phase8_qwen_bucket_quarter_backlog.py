#!/usr/bin/env python3
"""Read-only health watcher for the Phase-8 Qwen quarter backlog.

The watcher observes processes, tmux sessions, logs, checkpoints by name, and
final JSON artifacts.  It never starts, stops, signals, or resumes work.  Its
only runtime writes are atomic status/alert files below ``STATE_DIR``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "phase8_qwen_bucket_quarter_backlog_monitor"
STATUS_FILE = STATE_DIR / "status.json"
ALERT_FILE = STATE_DIR / "ALERT.json"

ITER_RE = re.compile(r"\bit\s+(\d+):", re.I)
FIELD_RES = {
    "loss": re.compile(r"\bloss:\s*([^,\s]+)", re.I),
    "grad_norm": re.compile(r"\bgrad_norm:\s*([^,\s]+)", re.I),
    "lr": re.compile(r"\blr:\s*([^,\s]+)", re.I),
}
HARD_ERROR_RES = [
    re.compile(pattern, re.I)
    for pattern in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"ProcessExitedException",
        r"ChildFailedError",
        r"NCCL.*(?:error|failed)",
        r"Segmentation fault",
        r"Traceback \(most recent call last\)",
        r"(?:^|\s)Killed(?:\s|$)",
        r"\[FAIL\]",
    )
]


class ArmSpec(NamedTuple):
    key: str
    lane: str
    prefix: str
    label: str
    metrics_name: str
    audit_name: str
    historical_reuse: bool


def queue_specs() -> tuple[ArmSpec, ...]:
    """Return the immutable, scheduled main-then-backup queue."""
    return (
        ArmSpec(
            "noq", "main", "phase8_qwen_bucket_quarter_noq",
            "official-Qwen matched No-Q",
            "phase8_qwen_bucket_quarter_noq_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_noq_FINAL_TRAIN_AUDIT.json",
            False,
        ),
        ArmSpec(
            "k2_balanced", "main", "phase8_qwen_bucket_quarter_k2_balanced",
            "k2 balanced",
            "phase8_qwen_bucket_quarter_k2_balanced_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k2_balanced_FINAL_TRAIN_AUDIT.json",
            True,
        ),
        ArmSpec(
            "k5_balanced", "main", "phase8_qwen_bucket_quarter_k5_balanced",
            "k5 balanced",
            "phase8_qwen_bucket_quarter_k5_balanced_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k5_balanced_FINAL_TRAIN_AUDIT.json",
            False,
        ),
        ArmSpec(
            "k10_balanced", "main", "phase8_qwen_bucket_quarter_k10_balanced",
            "k10 balanced",
            "phase8_qwen_bucket_quarter_k10_balanced_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k10_balanced_FINAL_TRAIN_AUDIT.json",
            False,
        ),
        ArmSpec(
            "k3_balanced", "backup", "phase8_qwen_bucket_quarter_k3_balanced",
            "k3 balanced",
            "phase8_qwen_bucket_quarter_k3_balanced_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k3_balanced_FINAL_TRAIN_AUDIT.json",
            False,
        ),
        ArmSpec(
            "k5_fixed", "backup", "phase8_qwen_bucket_quarter_k5_fixed",
            "k5 fixed",
            "phase8_qwen_bucket_quarter_k5_fixed_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k5_fixed_FINAL_TRAIN_AUDIT.json",
            False,
        ),
        ArmSpec(
            "k10_fixed", "backup", "phase8_qwen_bucket_quarter_k10_fixed",
            "k10 fixed",
            "phase8_qwen_bucket_quarter_k10_fixed_FINAL_METRICS.json",
            "phase8_qwen_bucket_quarter_k10_fixed_FINAL_TRAIN_AUDIT.json",
            True,
        ),
    )


def now_iso(now: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if now is None else now, tz=timezone.utc
    ).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def tail_text(path: Path, size: int = 2 * 1024 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def token_is_nonfinite(token: str | None) -> bool:
    if token is None:
        return False
    try:
        return not math.isfinite(float(token))
    except ValueError:
        return True


def progress_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        match = ITER_RE.search(raw_line)
        if not match or "loss:" not in raw_line.lower():
            continue
        row: dict[str, Any] = {
            "iteration": int(match.group(1)),
            "line": re.sub(r"\x1b\[[0-9;]*m", "", raw_line)[-500:],
        }
        for field, pattern in FIELD_RES.items():
            found = pattern.search(raw_line)
            token = found.group(1) if found else None
            row[f"{field}_token"] = token
            row[field] = parse_float(token)
        rows.append(row)
    return rows


def grad_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [token_is_nonfinite(row.get("grad_norm_token")) for row in rows]
    trailing = 0
    for bad in reversed(flags):
        if not bad:
            break
        trailing += 1
    bad_rows = [row for row, bad in zip(rows, flags) if bad]
    recent_20 = sum(flags[-20:])
    recent_100 = sum(flags[-100:])
    return {
        "nonfinite_recent_20": recent_20,
        "nonfinite_recent_100": recent_100,
        "nonfinite_trailing": trailing,
        "latest_nonfinite_iteration": (
            bad_rows[-1]["iteration"] if bad_rows else None
        ),
        "unhealthy": trailing >= 2 or recent_20 >= 3 or recent_100 >= 10,
    }


def hard_log_errors(text: str) -> list[str]:
    hits: list[str] = []
    lines = text.splitlines()
    for pattern in HARD_ERROR_RES:
        for line in reversed(lines):
            if "Error in extra logging" in line:
                continue
            if pattern.search(line):
                hits.append(
                    re.sub(r"\x1b\[[0-9;]*m", "", line).strip()[-500:]
                )
                break
    return list(dict.fromkeys(hits))


def process_snapshot() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,etime,pcpu,pmem,cmd", "--no-headers"],
        capture_output=True, text=True, check=False,
    )
    needles = (
        "phase8_qwen_bucket",
        "sequence_phase8_qwen_bucket", "quarter_backlog",
    )
    return [
        line.strip()[:1600]
        for line in result.stdout.splitlines()
        if any(needle in line for needle in needles)
        and "monitor_phase8_qwen_bucket_quarter_backlog.py" not in line
    ][:60]


def tmux_sessions() -> list[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True, text=True, check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "gpu_query_error": str(exc)}
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return {
            "status": "error",
            "gpu_query_error": detail[-1000:] or f"exit code {result.returncode}",
        }
    rows = []
    try:
        for index, line in enumerate(result.stdout.splitlines()):
            if not line.strip():
                continue
            values = [float(value.strip()) for value in line.split(",")]
            if len(values) != 4:
                raise ValueError(f"expected 4 columns, got {len(values)}")
            rows.append({
                "index": index,
                "util_pct": values[0],
                "mem_used_mib": values[1],
                "mem_total_mib": values[2],
                "temp_c": values[3],
            })
    except ValueError as exc:
        return {
            "status": "error",
            "gpu_query_error": f"invalid nvidia-smi output: {exc}",
        }
    if not rows:
        return {
            "status": "error",
            "gpu_query_error": "nvidia-smi returned no GPU rows",
        }
    return {"status": "ok", "gpus": rows}


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "top-level JSON is not an object"
    return payload, None


def nonfinite_json_paths(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        hits.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            hits.extend(nonfinite_json_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(nonfinite_json_paths(child, f"{path}[{index}]"))
    return hits


def arm_paths(spec: ArmSpec, root: Path, log_root: Path) -> dict[str, Any]:
    s1_exp = f"{spec.prefix}_stage1_100000"
    s2_exp = f"{spec.prefix}_stage2_50000"
    return {
        "metrics": log_root / spec.metrics_name,
        "audit": log_root / spec.audit_name,
        "wrapper": log_root / f"{spec.prefix}_wrapper.log",
        "stage1_log": log_root / f"{s1_exp}.log",
        "stage2_log": log_root / f"{s2_exp}.log",
        "stage1_checkpoint": root / "exps" / s1_exp / f"{s1_exp}_ckpt_last.pth",
        "stage2_checkpoint": root / "exps" / s2_exp / f"{s2_exp}_ckpt_last.pth",
    }


def arm_processes(spec: ArmSpec, processes: list[str]) -> list[str]:
    return [line for line in processes if spec.prefix in line]


def active_phase(spec: ArmSpec, processes: list[str]) -> str | None:
    joined = "\n".join(arm_processes(spec, processes))
    if not joined:
        return None
    if f"{spec.prefix}_stage2_50000" in joined:
        return "stage2_training"
    if f"{spec.prefix}_stage1_100000" in joined:
        return "stage1_training"
    if "eval.py" in joined or "phase4_eval.py" in joined:
        return "evaluation"
    return "wrapper_or_preflight"


def latest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def inspect_arm(
    spec: ArmSpec,
    root: Path,
    log_root: Path,
    processes: list[str],
    now: float,
) -> dict[str, Any]:
    paths = arm_paths(spec, root, log_root)
    phase = active_phase(spec, processes)
    if phase == "stage1_training":
        active_log = paths["stage1_log"]
    elif phase == "stage2_training":
        active_log = paths["stage2_log"]
    elif phase in {"evaluation", "wrapper_or_preflight"}:
        active_log = latest_existing([
            paths["wrapper"], paths["stage2_log"], paths["stage1_log"],
        ])
    else:
        active_log = latest_existing([
            paths["wrapper"], paths["stage2_log"], paths["stage1_log"],
        ])
    text = tail_text(active_log) if active_log else ""
    rows = progress_rows(text)[-100:]
    latest = rows[-1] if rows else {}
    health = grad_health(rows)
    infrastructure_driver_mismatch = (
        "Driver/library version mismatch" in text
        and "nvmlInit_v2() failed" in text
    )

    metrics, metrics_error = read_json(paths["metrics"])
    audit_path = paths["audit"]
    if metrics and isinstance(metrics.get("training_audit"), str):
        reported_audit = Path(metrics["training_audit"])
        # Historical-reuse reports canonically point to the already audited
        # legacy quarter checkpoint.  That is a valid fast completion path;
        # it does not create a duplicate bucket-prefixed training audit.
        if reported_audit.is_absolute():
            audit_path = reported_audit
        else:
            metrics_error = (
                (metrics_error + "; ") if metrics_error else ""
            ) + "training_audit must be an absolute path"
    audit, audit_error = read_json(audit_path)
    metrics_nonfinite = nonfinite_json_paths(metrics) if metrics else []
    audit_nonfinite = nonfinite_json_paths(audit) if audit else []
    expected_k = (
        int(spec.key.split("_", 1)[0][1:]) if spec.key != "noq" else None
    )
    expected_strategy = (
        spec.key.split("_", 1)[1] if spec.key != "noq" else None
    )
    metrics_identity_ok = bool(
        metrics
        and metrics.get("experiment") == spec.prefix
        and (
            (
                spec.key == "noq"
                and metrics.get("scale") == "quarter"
                and metrics.get("arm") == "noq"
                and metrics.get("matched_bucket_arm") == "k2_balanced"
            )
            or (
                spec.key != "noq"
                and metrics.get("scale") == "quarter"
                and metrics.get("k") == expected_k
                and metrics.get("strategy") == expected_strategy
                and bool(metrics.get("historical_checkpoint_reused"))
                == spec.historical_reuse
            )
        )
    )
    metrics_ok = bool(
        metrics
        and not metrics_error
        and not metrics_nonfinite
        and metrics.get("status") == "passed"
        and metrics_identity_ok
    )
    if spec.key == "noq":
        audit_identity_ok = bool(
            audit
            and audit.get("scale") == "quarter"
            and audit.get("arm") == "noq"
            and audit.get("matched_bucket_arm") == "k2_balanced"
        )
    elif spec.historical_reuse:
        audit_identity_ok = bool(
            audit
            and audit.get("arm")
            == ("halfq" if spec.key == "k2_balanced" else "fullq")
        )
    else:
        audit_identity_ok = bool(
            audit
            and audit.get("scale") == "quarter"
            and audit.get("k") == expected_k
            and audit.get("strategy") == expected_strategy
        )
    if spec.key == "noq":
        audit_q_ok = bool(
            audit
            and audit.get("stage1_use_q_conditioning") is False
            and audit.get("stage2_use_q_conditioning") is False
        )
    else:
        # New bucket audits expose one shared q_conditioning flag, while the
        # historical quarter audits expose one flag per training stage.
        audit_q_ok = bool(
            audit
            and (
                audit.get("q_conditioning") is True
                or (
                    audit.get("stage1_use_q_conditioning") is True
                    and audit.get("stage2_use_q_conditioning") is True
                )
            )
        )
    audit_ok = bool(
        audit
        and not audit_error
        and not audit_nonfinite
        and audit.get("status") == "passed"
        and audit.get("stage1_iteration") == 100000
        and audit.get("stage2_iteration") == 150000
        and audit_q_ok
        and audit_identity_ok
    )
    proc = arm_processes(spec, processes)
    artifact_started = any(
        path.exists() for path in (
            paths["audit"], paths["metrics"], paths["wrapper"],
            paths["stage1_log"], paths["stage2_log"],
            paths["stage1_checkpoint"], paths["stage2_checkpoint"],
        )
    )
    if metrics_ok and audit_ok:
        state = "complete"
    elif proc:
        state = "active"
    elif artifact_started:
        state = "stalled_or_transition"
    else:
        state = "pending"
    age = (
        max(0.0, now - active_log.stat().st_mtime)
        if active_log and active_log.exists() else None
    )
    completion_mtime = None
    completion_files = [
        path for path in (paths["metrics"], paths["audit"]) if path.is_file()
    ]
    if completion_files:
        completion_mtime = max(path.stat().st_mtime for path in completion_files)

    checkpoint_iteration = None
    checkpoint_source = None
    if audit_ok:
        checkpoint_iteration = audit["stage2_iteration"]
        checkpoint_source = "FINAL_TRAIN_AUDIT"
    elif paths["stage2_checkpoint"].is_file():
        latest_it = latest.get("iteration")
        if spec.key == "noq" and latest_it is not None:
            checkpoint_iteration = max(100000, (latest_it // 25000) * 25000)
        elif latest_it is not None and latest_it >= 150000:
            checkpoint_iteration = 150000
        else:
            # Newly migrated S2 starts at the completed S1 iteration.
            checkpoint_iteration = 100000
        checkpoint_source = "checkpoint_presence_and_save_schedule"
    elif paths["stage1_checkpoint"].is_file():
        latest_it = latest.get("iteration")
        if spec.key == "noq" and latest_it is not None:
            checkpoint_iteration = min(100000, (latest_it // 25000) * 25000)
        else:
            checkpoint_iteration = 100000
        checkpoint_source = "checkpoint_presence_and_save_schedule"

    return {
        "key": spec.key,
        "lane": spec.lane,
        "label": spec.label,
        "historical_reuse": spec.historical_reuse,
        "prefix": spec.prefix,
        "state": state,
        "phase": phase,
        "processes": proc,
        "active_log": str(active_log) if active_log else None,
        "log_age_sec": round(age, 1) if age is not None else None,
        "latest_iteration": latest.get("iteration"),
        "latest_metrics": {
            key: latest.get(key) for key in ("loss", "grad_norm", "lr")
        },
        "checkpoint": {
            "iteration": checkpoint_iteration,
            "source": checkpoint_source,
            "stage1_path": str(paths["stage1_checkpoint"]),
            "stage1_exists": paths["stage1_checkpoint"].is_file(),
            "stage2_path": str(paths["stage2_checkpoint"]),
            "stage2_exists": paths["stage2_checkpoint"].is_file(),
        },
        "grad_health": health,
        # A kernel/userspace NVIDIA mismatch is recoverable after the host
        # reloads the installed driver.  The supervisor waits for that repair,
        # so do not turn its known NCCL startup failure into a permanent alert.
        "hard_log_errors": (
            [] if infrastructure_driver_mismatch else hard_log_errors(text)
        ),
        "infrastructure_driver_mismatch": infrastructure_driver_mismatch,
        "final_metrics": {
            "path": str(paths["metrics"]),
            "exists": paths["metrics"].is_file(),
            "valid": metrics_ok,
            "identity_valid": metrics_identity_ok,
            "parse_error": metrics_error,
            "nonfinite_paths": metrics_nonfinite,
        },
        "final_train_audit": {
            "path": str(audit_path),
            "canonical_bucket_path": str(paths["audit"]),
            "referenced_by_final_metrics": audit_path != paths["audit"],
            "exists": audit_path.is_file(),
            "valid": audit_ok,
            "identity_valid": audit_identity_ok,
            "parse_error": audit_error,
            "status": audit.get("status") if audit else None,
            "stage1_iteration": audit.get("stage1_iteration") if audit else None,
            "stage2_iteration": audit.get("stage2_iteration") if audit else None,
            "nonfinite_paths": audit_nonfinite,
        },
        "completion_mode": (
            "historical_reuse_report_validation"
            if metrics_ok and audit_ok and spec.historical_reuse
            else "trained_arm_report_validation"
            if metrics_ok and audit_ok else None
        ),
        "_completion_mtime": completion_mtime,
    }


def sequence_observation(
    processes: list[str], tmux: list[str],
) -> dict[str, Any]:
    process_hits = [
        line for line in processes
        if "sequence_phase8_qwen_bucket" in line
        or "quarter_backlog" in line
    ]
    tmux_hits = [
        name for name in tmux
        if (
            ("phase8" in name.lower() and "quarter" in name.lower())
            or "qwen_bucket" in name.lower()
            or "quarter_backlog" in name.lower()
        )
    ]
    return {
        "detected": bool(process_hits or tmux_hits),
        "processes": process_hits,
        "tmux_sessions": tmux_hits,
    }


def collect(
    *,
    root: Path = ROOT,
    log_root: Path = LOG_ROOT,
    stale_seconds: int = 1200,
    transition_grace_seconds: int = 900,
    processes: list[str] | None = None,
    tmux: list[str] | None = None,
    gpu: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    now_value = time.time() if now is None else now
    process_lines = process_snapshot() if processes is None else processes
    tmux_names = tmux_sessions() if tmux is None else tmux
    gpu_data = gpu_snapshot() if gpu is None else gpu
    specs = queue_specs()
    arms = [
        inspect_arm(spec, root, log_root, process_lines, now_value)
        for spec in specs
    ]
    issues: list[dict[str, str]] = []

    def issue(severity: str, code: str, detail: str) -> None:
        item = {"severity": severity, "code": code, "detail": detail}
        if item not in issues:
            issues.append(item)

    active_indices = [
        index for index, arm in enumerate(arms) if arm["state"] == "active"
    ]
    if len(active_indices) > 1:
        issue(
            "hard", "multiple_active_arms",
            f"active={[arms[index]['key'] for index in active_indices]}",
        )

    first_incomplete = next(
        (index for index, arm in enumerate(arms) if arm["state"] != "complete"),
        len(arms),
    )
    for index in active_indices:
        if index > first_incomplete:
            issue(
                "hard", "queue_order_violation",
                f"{arms[index]['key']} active before {arms[first_incomplete]['key']}",
            )

    for arm in arms:
        if arm["infrastructure_driver_mismatch"]:
            issue(
                "transient", "nvidia_driver_library_mismatch",
                f"{arm['key']}: waiting for matching NVIDIA kernel/userspace",
            )
        for hit in arm["hard_log_errors"]:
            issue("hard", "hard_log_error", f"{arm['key']}: {hit}")
        final_metrics = arm["final_metrics"]
        final_audit = arm["final_train_audit"]
        if final_metrics["exists"] and not final_metrics["valid"]:
            issue(
                "hard", "invalid_final_metrics",
                f"{arm['key']}: parse={final_metrics['parse_error']} "
                f"nonfinite={final_metrics['nonfinite_paths']}",
            )
        if (
            (final_audit["exists"] and not final_audit["valid"])
            or (final_metrics["exists"] and not final_audit["exists"])
        ):
            issue(
                "hard", "invalid_final_train_audit",
                f"{arm['key']}: status={final_audit['status']} "
                f"S1={final_audit['stage1_iteration']} "
                f"S2={final_audit['stage2_iteration']} "
                f"parse={final_audit['parse_error']}",
            )
        latest = arm["latest_metrics"]
        for field in ("loss", "lr"):
            token = None
            if arm["active_log"]:
                rows = progress_rows(tail_text(Path(arm["active_log"])))[-100:]
                for row in rows:
                    token = row.get(f"{field}_token")
                    if token_is_nonfinite(token):
                        issue(
                            "hard", "nonfinite_metric",
                            f"{arm['key']}: it={row['iteration']} "
                            f"{field}={token}",
                        )
        health = arm["grad_health"]
        if health["unhealthy"]:
            issue(
                "hard", "persistent_nonfinite_grad",
                f"{arm['key']}: trailing={health['nonfinite_trailing']} "
                f"recent20={health['nonfinite_recent_20']} "
                f"recent100={health['nonfinite_recent_100']}",
            )
        elif health["nonfinite_recent_100"]:
            issue(
                "transient", "transient_amp_grad_overflow",
                f"{arm['key']}: recovered after it="
                f"{health['latest_nonfinite_iteration']}",
            )
        if arm["state"] == "active":
            age = arm["log_age_sec"]
            if arm["active_log"] is None:
                issue(
                    "transient", "active_log_not_yet_visible",
                    f"{arm['key']}: process exists but no log is visible",
                )
            elif age is not None and age > stale_seconds:
                issue(
                    "hard", "stale_active_log",
                    f"{arm['key']}: age={age:.0f}s > {stale_seconds}s",
                )

    sequence = sequence_observation(process_lines, tmux_names)
    any_backlog_started = any(
        arm["state"] != "pending" for arm in arms[1:]
    )
    if (
        first_incomplete < len(arms)
        and arms[first_incomplete]["state"] == "stalled_or_transition"
        and not active_indices
        and not sequence["detected"]
    ):
        age = arms[first_incomplete]["log_age_sec"]
        if age is not None and age > stale_seconds:
            issue(
                "hard", "incomplete_arm_process_missing",
                f"{arms[first_incomplete]['key']}: no process/tmux; "
                f"log age={age:.0f}s",
            )
        else:
            issue(
                "transient", "arm_transition",
                f"{arms[first_incomplete]['key']}: artifacts visible; "
                "process/tmux not currently observed",
            )

    if gpu_data.get("status") != "ok":
        issue(
            "transient", "gpu_query_error",
            str(gpu_data.get("gpu_query_error") or "unknown GPU query failure"),
        )
    elif active_indices:
        arm = arms[active_indices[0]]
        age = arm["log_age_sec"]
        gpus = gpu_data.get("gpus") or []
        if (
            age is not None and age > 300 and gpus
            and all(
                item.get("util_pct", 100) < 5
                and item.get("mem_used_mib", 100000) < 2000
                for item in gpus
            )
        ):
            issue(
                "hard", "gpu_idle_with_stale_progress",
                f"{arm['key']}: GPU idle and log age={age:.0f}s",
            )

    if first_incomplete == len(arms):
        handoff = {
            "state": "queue_complete",
            "from": arms[-1]["key"],
            "expected_next": None,
            "connected": True,
        }
    else:
        previous = arms[first_incomplete - 1] if first_incomplete else None
        current = arms[first_incomplete]
        if current["state"] == "active":
            handoff_state = "connected"
            connected = True
        elif previous is None:
            handoff_state = "awaiting_first_arm"
            connected = None
        elif not any_backlog_started and first_incomplete == 1:
            handoff_state = "backlog_not_started"
            connected = None
        else:
            completed_at = previous.get("_completion_mtime")
            transition_age = (
                max(0.0, now_value - completed_at)
                if completed_at is not None else None
            )
            if sequence["detected"]:
                handoff_state = "sequence_waiting_or_transitioning"
                connected = True
            elif (
                transition_age is not None
                and transition_age <= transition_grace_seconds
            ):
                handoff_state = "within_transition_grace"
                connected = None
                issue(
                    "transient", "next_arm_transition",
                    f"{previous['key']} -> {current['key']}; "
                    f"age={transition_age:.0f}s",
                )
            else:
                handoff_state = "disconnected"
                connected = False
                issue(
                    "hard", "next_arm_not_connected",
                    f"{previous['key']} complete but {current['key']} "
                    "has no process/tmux",
                )
        handoff = {
            "state": handoff_state,
            "from": previous["key"] if previous else None,
            "expected_next": current["key"],
            "connected": connected,
        }

    for arm in arms:
        arm.pop("_completion_mtime", None)
    hard = [item for item in issues if item["severity"] == "hard"]
    transient = [
        item for item in issues if item["severity"] == "transient"
    ]
    return {
        "schema_version": 1,
        "updated_at": now_iso(now_value),
        "watcher": "phase8_qwen_bucket_quarter_backlog",
        "read_only": True,
        "status": (
            "hard_incident" if hard
            else "transient_or_nonfatal" if transient
            else "healthy"
        ),
        "queue": {
            "main": ["noq", "k2_balanced", "k5_balanced", "k10_balanced"],
            "backup": ["k3_balanced", "k5_fixed", "k10_fixed"],
            "first_incomplete": (
                arms[first_incomplete]["key"]
                if first_incomplete < len(arms) else None
            ),
        },
        "sequence": sequence,
        "active_arm": (
            arms[active_indices[0]]["key"] if len(active_indices) == 1 else None
        ),
        "arms": arms,
        "handoff": handoff,
        "gpu": gpu_data,
        "processes": process_lines,
        "tmux": tmux_names,
        "issues": issues,
        "hard_incidents": hard,
        "transient_nonfatal": transient,
    }


def compact(snapshot: dict[str, Any]) -> str:
    return (
        f"status={snapshot['status']} active={snapshot['active_arm']} "
        f"next={snapshot['queue']['first_incomplete']} "
        f"handoff={snapshot['handoff']['state']} "
        f"hard={len(snapshot['hard_incidents'])} "
        f"transient={len(snapshot['transient_nonfatal'])}"
    )


def self_test() -> None:
    rows = progress_rows(
        "\n".join(
            f"it {index}: grad_norm:{grad}, loss:1.0, lr:0.0001"
            for index, grad in enumerate(("1.0", "nan", "1.2"), 1)
        )
    )
    health = grad_health(rows)
    assert health["nonfinite_recent_100"] == 1
    assert health["nonfinite_trailing"] == 0
    assert not health["unhealthy"]
    rows = progress_rows(
        "it 1: grad_norm:1.0, loss:1.0, lr:0.1\n"
        "it 2: grad_norm:nan, loss:1.0, lr:0.1\n"
        "it 3: grad_norm:inf, loss:1.0, lr:0.1\n"
    )
    assert grad_health(rows)["unhealthy"]
    assert token_is_nonfinite("nan")
    assert token_is_nonfinite("not-a-number")
    assert not token_is_nonfinite("1.0")
    assert [arm.key for arm in queue_specs()] == [
        "noq", "k2_balanced", "k5_balanced", "k10_balanced",
        "k3_balanced", "k5_fixed", "k10_fixed",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="collect once and exit")
    mode.add_argument("--loop", action="store_true", help="collect continuously")
    mode.add_argument(
        "--self-test", action="store_true", help="run CPU-only parser tests",
    )
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--stale-seconds", type=int, default=1200)
    parser.add_argument("--transition-grace-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("monitor self-test: passed")
        return 0
    # No explicit mode is a compatibility alias for --loop.  The durable
    # supervisor predates the explicit switch and invokes this script with no
    # arguments; supporting it does not itself launch the watcher.
    if not args.once and not args.loop:
        args.loop = True
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = collect(
            stale_seconds=args.stale_seconds,
            transition_grace_seconds=args.transition_grace_seconds,
        )
        atomic_json(STATUS_FILE, snapshot)
        if snapshot["hard_incidents"]:
            atomic_json(ALERT_FILE, snapshot)
        elif ALERT_FILE.exists():
            ALERT_FILE.unlink()
        print(compact(snapshot), flush=True)
        for item in snapshot["issues"]:
            print(
                f"[{item['severity'].upper()}] "
                f"{item['code']}: {item['detail']}",
                flush=True,
            )
        if args.once:
            return 2 if snapshot["hard_incidents"] else 0
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    sys.exit(main())
