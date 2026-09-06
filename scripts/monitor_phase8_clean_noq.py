#!/usr/bin/env python3
"""Focused, read-only monitor for phase8_catalog_matched_noq.

It never kills, resumes, or edits a training process.  A non-zero ``--once``
exit marks an incident that needs Codex SOL adjudication; it is deliberately
not permission for the supervising loop to stop training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "phase8_catalog_matched_noq_monitor"
STATUS_FILE = STATE_DIR / "status.json"
ALERT_FILE = STATE_DIR / "ALERT.json"
AUDIT_FILE = STATE_DIR / "contract_audit.json"
PREFIX = "phase8_catalog_matched_noq"
S1_LOG = LOG_ROOT / f"{PREFIX}_stage1_400000.log"
S2_LOG = LOG_ROOT / f"{PREFIX}_stage2_200000.log"
EVAL_LOG = LOG_ROOT / f"{PREFIX}_stage2_200000_musiccaps_eval.log"
METRICS = (
    ROOT / "eval_output/metrics"
    / f"{PREFIX}_stage2_200000_musiccaps"
    / "metrics.txt"
)

ITER_RE = re.compile(r"\bit\s+(\d+):")
FLOAT_FIELDS = {
    "loss": re.compile(r"\bloss:\s*([^,\s]+)", re.I),
    "grad_norm": re.compile(r"\bgrad_norm:\s*([^,\s]+)", re.I),
    "lr": re.compile(r"\blr:\s*([^,\s]+)", re.I),
    "avg_time_s": re.compile(r"\bavg_time:([^,\s]+)", re.I),
}
HARD_ERROR_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"Error occurred at iteration",
        r"ProcessExitedException",
        r"ChildFailedError",
        r"NCCL.*(?:error|failed)",
        r"Segmentation fault",
        r"Traceback \(most recent call last\)",
        r"(?:^|\s)Killed(?:\s|$)",
        r"\[FAIL\]",
    )
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def finite_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def progress_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if "loss:" not in line.lower() or not ITER_RE.search(line):
            continue
        row: dict[str, Any] = {
            "iteration": int(ITER_RE.search(line).group(1)),
            "line": re.sub(r"\x1b\[[0-9;]*m", "", line)[-500:],
        }
        for key, pattern in FLOAT_FIELDS.items():
            match = pattern.search(line)
            token = match.group(1) if match else None
            row[f"{key}_token"] = token
            row[key] = finite_float(token)
        rows.append(row)
    return rows


def token_is_nonfinite(token: str | None) -> bool:
    """Return true only when a present metric token is invalid/non-finite."""
    if token is None:
        return False
    try:
        return not math.isfinite(float(token))
    except ValueError:
        return True


def grad_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify AMP gradient overflows without treating one as corruption.

    GradScaler skips an optimizer update when unscaled gradients are
    non-finite.  An isolated logged ``grad_norm:nan`` followed by finite
    gradients is therefore a review event, not a stop condition.  Persistent
    or dense events remain fail-closed and require Codex SOL adjudication.
    """
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
        "nonfinite_total_in_tail": sum(flags),
        "nonfinite_recent_20": recent_20,
        "nonfinite_recent_100": recent_100,
        "nonfinite_trailing": trailing,
        "latest_nonfinite_iteration": (
            bad_rows[-1].get("iteration") if bad_rows else None
        ),
        "unhealthy": trailing >= 2 or recent_20 >= 3 or recent_100 >= 10,
    }


def self_test() -> None:
    def rows(tokens: list[str]) -> list[dict[str, Any]]:
        return [
            {"iteration": index * 50, "grad_norm_token": token}
            for index, token in enumerate(tokens, 1)
        ]

    isolated = grad_health(rows(["1.8"] * 50 + ["nan"] + ["1.9"] * 49))
    assert isolated["nonfinite_recent_100"] == 1
    assert isolated["nonfinite_trailing"] == 0
    assert not isolated["unhealthy"]

    trailing = grad_health(rows(["1.8"] * 18 + ["nan", "inf"]))
    assert trailing["nonfinite_trailing"] == 2
    assert trailing["unhealthy"]

    dense_20 = grad_health(rows(["1.8"] * 17 + ["nan", "1.8", "nan", "1.8", "nan"]))
    assert dense_20["nonfinite_recent_20"] == 3
    assert dense_20["unhealthy"]

    dense_100 = grad_health(rows((["nan"] + ["1.8"] * 9) * 10))
    assert dense_100["nonfinite_recent_100"] == 10
    assert dense_100["unhealthy"]


def process_snapshot() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,etime,pcpu,pmem,cmd", "--no-headers"],
        capture_output=True, text=True, check=False,
    )
    return [
        line.strip()[:1000]
        for line in result.stdout.splitlines()
        if PREFIX in line
        and "monitor_phase8_clean_noq.py" not in line
        and "audit_phase8_clean_noq_contract.py" not in line
        and "adjudicate_phase8_stop_with_codex.sh" not in line
        and "codex_sol_adjudication" not in line
        and "codex_sol_verdict" not in line
    ][:30]


def tmux_sessions() -> list[str]:
    result = subprocess.run(
        ["tmux", "ls", "-F", "#{session_name}"],
        capture_output=True, text=True, check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=False, timeout=10,
        )
        values = [float(value.strip()) for value in result.stdout.strip().split(",")]
        if len(values) != 4:
            return {"error": result.stderr.strip() or result.stdout.strip()}
        return dict(zip(("util_pct", "mem_used_mib", "mem_total_mib", "temp_c"), values))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def contract_audit() -> tuple[dict[str, Any], int]:
    command = [
        "/home/kojiek/venvs/dac/bin/python",
        str(ROOT / "scripts/audit_phase8_clean_noq_contract.py"),
        "--phase", "auto", "--json-out", str(AUDIT_FILE),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "failed",
            "issues": [f"contract auditor produced invalid JSON: {result.stderr[-1000:]}"],
        }
    return payload, result.returncode


def hard_log_errors(text: str) -> list[str]:
    lines = text.splitlines()
    hits: list[str] = []
    for pattern in HARD_ERROR_PATTERNS:
        for line in reversed(lines):
            # Known non-fatal visualization warning; it does not affect training.
            if "Error in extra logging" in line:
                continue
            if pattern.search(line):
                hits.append(re.sub(r"\x1b\[[0-9;]*m", "", line).strip()[:500])
                break
    return list(dict.fromkeys(hits))


def phase_and_log(processes: list[str]) -> tuple[str, Path | None, int | None]:
    joined = "\n".join(processes)
    if "phase4_eval.py" in joined or " eval.py " in joined:
        return "eval", EVAL_LOG, None
    if f"{PREFIX}_stage2_200000" in joined:
        return "s2_training", S2_LOG, 600000
    if f"{PREFIX}_stage1_400000" in joined:
        return "s1_training", S1_LOG, 400000
    if METRICS.is_file():
        return "complete", EVAL_LOG, None
    if EVAL_LOG.is_file():
        return "eval_or_stopped", EVAL_LOG, None
    if S2_LOG.is_file():
        return "s2_stopped_or_transition", S2_LOG, 600000
    if S1_LOG.is_file():
        return "s1_stopped_or_transition", S1_LOG, 400000
    return "pending", None, None


def collect(stale_seconds: int, min_free_gb: int) -> dict[str, Any]:
    processes = process_snapshot()
    phase, active_log, target = phase_and_log(processes)
    text = tail_text(active_log) if active_log else ""
    rows = progress_rows(text)
    recent = rows[-100:]
    latest = rows[-1] if rows else {}
    age = time.time() - active_log.stat().st_mtime if active_log and active_log.exists() else None
    issues: list[dict[str, str]] = []

    def issue(severity: str, code: str, detail: str) -> None:
        issues.append({"severity": severity, "code": code, "detail": detail})

    audit, audit_rc = contract_audit()
    if audit_rc != 0 or audit.get("status") != "passed":
        issue("hard", "contract_drift", "; ".join(audit.get("issues") or ["audit failed"]))

    for hit in hard_log_errors(text):
        issue("hard", "hard_log_error", hit)

    # A non-finite forward loss or learning rate is always an incident.
    # Gradients are handled separately because AMP GradScaler safely skips an
    # occasional overflow and normally recovers on the next optimizer step.
    for row in recent:
        for field in ("loss", "lr"):
            if token_is_nonfinite(row.get(f"{field}_token")):
                issue(
                    "hard", "nonfinite_metric",
                    f"it={row['iteration']} {field}={row.get(f'{field}_token')}",
                )
    gradient_health = grad_health(recent)
    if gradient_health["unhealthy"]:
        issue(
            "hard", "persistent_nonfinite_grad",
            "AMP grad overflow is persistent/dense: "
            f"latest_it={gradient_health['latest_nonfinite_iteration']} "
            f"trailing={gradient_health['nonfinite_trailing']} "
            f"recent20={gradient_health['nonfinite_recent_20']} "
            f"recent100={gradient_health['nonfinite_recent_100']}",
        )
    elif gradient_health["nonfinite_recent_100"]:
        issue(
            "review", "transient_amp_grad_overflow",
            "isolated AMP grad overflow recovered; optimizer step was skipped: "
            f"latest_it={gradient_health['latest_nonfinite_iteration']} "
            f"trailing={gradient_health['nonfinite_trailing']} "
            f"recent20={gradient_health['nonfinite_recent_20']} "
            f"recent100={gradient_health['nonfinite_recent_100']}",
        )
    if recent:
        losses = [row["loss"] for row in recent[-3:] if row.get("loss") is not None]
        grads = [row["grad_norm"] for row in recent[-3:] if row.get("grad_norm") is not None]
        if len(losses) == 3 and all(value > 5.0 for value in losses):
            issue("hard", "loss_explosion", f"last three losses={losses}")
        elif latest.get("loss") is not None and not (0.5 <= latest["loss"] <= 2.5):
            issue("review", "loss_outside_reference_band", f"loss={latest['loss']:.6f}")
        if len(grads) == 3 and all(value > 100.0 for value in grads):
            issue("hard", "grad_explosion", f"last three grad_norm={grads}")
        elif latest.get("grad_norm") is not None and latest["grad_norm"] > 50.0:
            issue("review", "high_grad", f"grad_norm={latest['grad_norm']:.6f}")

    active_training = phase in ("s1_training", "s2_training")
    if active_training and age is not None and age > stale_seconds:
        issue("hard", "stale_training_log", f"{active_log} age={age:.0f}s > {stale_seconds}s")
    if phase in ("s1_stopped_or_transition", "s2_stopped_or_transition"):
        # Brief transitions are normal; only fail once the last log is stale.
        if age is not None and age > stale_seconds:
            issue("hard", "process_missing", f"phase={phase}, no process, log age={age:.0f}s")

    root_free = shutil.disk_usage("/").free / (1024**3)
    if root_free < min_free_gb:
        issue("hard", "disk_low", f"root free={root_free:.1f}G < {min_free_gb}G")

    clap = None
    quality_gate = "pending"
    if METRICS.is_file():
        match = re.search(r"^clap_score:\s*([-+0-9.eE]+)$", METRICS.read_text(), re.M)
        if not match:
            issue("hard", "metrics_invalid", "metrics.txt has no clap_score")
        else:
            clap = finite_float(match.group(1))
            if clap is None:
                issue("hard", "metrics_nonfinite", f"clap_score={match.group(1)}")
            elif clap >= 0.18:
                quality_gate = "target_met"
            elif clap >= 0.17:
                quality_gate = "partial_recovery"
                issue("review", "clap_partial_recovery", f"CLAP={clap:.4f}; target >=0.18")
            elif clap >= 0.15:
                quality_gate = "hypothesis_not_supported"
                issue("review", "clap_no_recovery", f"CLAP={clap:.4f}; target >=0.18")
            else:
                quality_gate = "failed"
                issue("hard", "clap_collapse", f"CLAP={clap:.4f} < 0.15")

    gpu = gpu_snapshot()
    if active_training and age is not None and age > 300:
        util, mem = gpu.get("util_pct"), gpu.get("mem_used_mib")
        if isinstance(util, float) and isinstance(mem, float) and util < 5 and mem < 2000:
            issue("hard", "gpu_idle", f"GPU util={util}% mem={mem}MiB and log is not fresh")

    iteration = latest.get("iteration")
    progress_pct = round(100 * iteration / target, 3) if iteration is not None and target else None
    hard = [item for item in issues if item["severity"] == "hard"]
    return {
        "status": "healthy" if not hard else "failed",
        "updated_at": now_iso(),
        "experiment": PREFIX,
        "experiment_contract": "S1 NoQ + S2 NoQ + eval --no_q + NoMask",
        "phase": phase,
        "active_log": str(active_log) if active_log else None,
        "log_age_sec": round(age, 1) if age is not None else None,
        "target_iteration": target,
        "latest": latest,
        "grad_health": gradient_health,
        "progress_pct": progress_pct,
        "processes": processes,
        "tmux": tmux_sessions(),
        "gpu": gpu,
        "disk_root_free_gb": round(root_free, 1),
        "contract_audit": {
            "status": audit.get("status"),
            "issues": audit.get("issues"),
            "warnings": audit.get("warnings"),
            "observed_phase": (audit.get("checks") or {}).get("observed_phase"),
            "report": str(AUDIT_FILE),
        },
        "clap_score": clap,
        "quality_gate": quality_gate,
        "quality_target": (
            "CLAP >=0.18 target; 0.17-0.18 partial; "
            "0.15-0.17 no recovery; <0.15 collapse"
        ),
        "issues": issues,
    }


def compact(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest") or {}
    gpu = snapshot.get("gpu") or {}
    return (
        f"status={snapshot['status']} phase={snapshot['phase']} "
        f"it={latest.get('iteration')}/{snapshot.get('target_iteration')} "
        f"progress={snapshot.get('progress_pct')}% loss={latest.get('loss')} "
        f"grad={latest.get('grad_norm')} lr={latest.get('lr')} "
        f"log_age={snapshot.get('log_age_sec')}s "
        f"gpu={gpu.get('util_pct')}%/{gpu.get('mem_used_mib')}MiB "
        f"disk={snapshot.get('disk_root_free_gb')}G "
        f"contract={snapshot['contract_audit']['status']} "
        f"quality={snapshot['quality_gate']} issues={len(snapshot['issues'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--stale-seconds", type=int, default=1200)
    parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("monitor self-test: passed")
        return 0
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = collect(args.stale_seconds, args.min_free_gb)
        atomic_json(STATUS_FILE, snapshot)
        hard = [item for item in snapshot["issues"] if item["severity"] == "hard"]
        if hard:
            atomic_json(ALERT_FILE, snapshot)
        elif ALERT_FILE.exists():
            ALERT_FILE.unlink()
        print(compact(snapshot), flush=True)
        for item in snapshot["issues"]:
            print(f"[{item['severity'].upper()}] {item['code']}: {item['detail']}", flush=True)
        if args.once:
            return 0 if not hard else 2
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    sys.exit(main())
