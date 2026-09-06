#!/usr/bin/env python3
"""External guardian for the live R-Shared -> conditional R-Matched queue.

This deliberately does not edit or control the running sequence.  It records
health while the sequence owns its lock.  Once that lock is released, it
restores the shared mutable NPZ cache to R-Matched and, when possible, runs the
paired quarter CLAP bootstrap from the two surviving checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/kojiek/MeanAudio")
STATE = Path("/home/kojiek/logs/rich_shared_then_matched_full")
SEQUENCE_LOG = STATE / "sequence.log"
LOCK = STATE / "sequence.lock"
STATUS = STATE / "guardian_status.json"
WATCH_LOG = STATE / "guardian.log"
RESTORE = ROOT / "scripts/restore_matched_binding_after_rich_shared.sh"
PAIRED = ROOT / "scripts/eval/run_paired_clap_ci_rich_shared_quarter.sh"
PAIRED_REPORT = STATE / "rich_shared_quarter_paired_clap_ci.json"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
SHARED_MODEL = (
    ROOT
    / "exps/phase8_qwen_rich_shared_noq_quarter_stage2_50000/"
    "phase8_qwen_rich_shared_noq_quarter_stage2_50000_ema_final.pth"
)
MATCHED_MODEL = (
    ROOT
    / "exps/phase8_qwen_caption10s_multisent_noq_quarter_stage2_50000/"
    "phase8_qwen_caption10s_multisent_noq_quarter_stage2_50000_ema_final.pth"
)

FATAL = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"loss:\s*(?:nan|inf)\b",
        r"^Traceback \(most recent call last\):",
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"ChildFailedError",
        r"Segmentation fault",
        r"\[FAIL\]",
    )
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_log(message: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")
    print(message, flush=True)


def sequence_lock_held() -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,etimes,pcpu,pmem,cmd", "--no-headers"],
        capture_output=True,
        text=True,
        check=False,
    )
    needles = ("sequence_rich_shared", "reextract_text_inplace", "torchrun", "train.py", "eval.py")
    return [
        line.strip()[:360]
        for line in result.stdout.splitlines()
        if any(needle in line for needle in needles)
        and "watch_rich_shared_then_matched_full" not in line
    ][:20]


def gpu_snapshot() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"error": result.stderr.strip() or "nvidia-smi failed"}
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "util_pct": float(values[0]),
        "memory_used_mib": float(values[1]),
        "memory_total_mib": float(values[2]),
        "temperature_c": float(values[3]),
    }


def tail(path: Path, size: int = 512 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def fatal_hits(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in FATAL):
            hits.append(line.strip()[-400:])
    return list(dict.fromkeys(hits))[-10:]


TQDM_RATE = re.compile(r"(\d+\.?\d*)(s/it|it/s)\]")


def slowest_rate_s_per_it(text: str) -> float | None:
    """Seconds per iteration from the most recent tqdm progress line.

    Catches the failure mode the staleness check cannot see: the job is still
    emitting progress lines, so the log never looks stale, but throughput has
    collapsed.  On 2026-08-10 a concurrent `bfs` scan of /mnt/HDD/kojiek from
    another session took the exFAT volume lock and drove the text rebind from
    1.05 s/it to 292 s/it (ETA 38 min -> 186 h) without tripping any alert.
    """
    for line in reversed(text.replace("\r", "\n").splitlines()):
        match = TQDM_RATE.search(line)
        if match:
            value = float(match.group(1))
            return value if match.group(2) == "s/it" else (1.0 / value if value else None)
    return None


def phase(processes: list[str], text: str) -> str:
    joined = "\n".join(processes)
    if "reextract_text_inplace" in joined:
        return "text_rebind"
    if "torchrun" in joined or "train.py" in joined:
        return "training"
    if "eval.py" in joined:
        return "evaluation"
    if "audit_caption_npz_binding" in joined:
        return "npz_audit"
    if "[PROMOTE]" in text:
        return "post_gate"
    return "sequence_or_preflight"


def notify(status: str, summary: str) -> None:
    subprocess.run(
        [
            str(NOTIFIER),
            "--status", status,
            "--experiment", "rich_shared_guardian",
            "--exit-code", "0" if status == "success" else "1",
            "--summary", summary,
            "--log", str(WATCH_LOG),
        ],
        check=False,
    )


def gpu_has_compute_process() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def post_sequence() -> int:
    append_log("sequence lock released; enforcing R-Matched cache restoration")
    while gpu_has_compute_process():
        append_log("cache restore waiting: GPU has another compute process")
        time.sleep(300)
    restored = subprocess.run([str(RESTORE)], check=False)
    if restored.returncode != 0:
        append_log(f"restore failed exit={restored.returncode}")
        notify("failure", f"R-Matched cache restore failed (exit {restored.returncode})")
        return restored.returncode
    append_log("R-Matched cache binding is restored/verified")

    if not (MATCHED_MODEL.is_file() and SHARED_MODEL.is_file()):
        append_log("paired CI skipped: one or both quarter EMA checkpoints are absent")
        notify("failure", "sequence ended before both quarter checkpoints existed; cache restored")
        return 0
    while gpu_has_compute_process():
        append_log("paired CI waiting: GPU has another compute process")
        time.sleep(300)
    paired = subprocess.run([str(PAIRED)], check=False)
    if paired.returncode != 0:
        append_log(f"paired CI failed exit={paired.returncode}")
        notify("failure", f"paired CLAP CI failed (exit {paired.returncode}); cache restored")
        return paired.returncode
    append_log(f"paired CI complete: {PAIRED_REPORT}")
    notify("success", "cache restored and R-Matched vs R-Shared paired CLAP CI completed")
    return 0


def snapshot() -> tuple[dict, bool]:
    held = sequence_lock_held()
    processes = process_lines()
    text = tail(SEQUENCE_LOG)
    hits = fatal_hits(text)
    age = None
    if SEQUENCE_LOG.exists():
        age = max(0.0, time.time() - SEQUENCE_LOG.stat().st_mtime)
    payload = {
        "schema_version": 1,
        "checked_at": now(),
        "sequence_lock_held": held,
        "phase": phase(processes, text),
        "processes": processes,
        "gpu": gpu_snapshot(),
        "sequence_log_age_seconds": age,
        "rate_s_per_it": slowest_rate_s_per_it(text),
        "fatal_hits": hits,
        "disk_free_gib": {
            "root": shutil.disk_usage("/").free / 2**30,
            "hdd": shutil.disk_usage("/mnt/HDD").free / 2**30,
        },
    }
    atomic_json(STATUS, payload)
    return payload, held


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    alerted: set[str] = set()
    while True:
        status, held = snapshot()
        append_log(
            f"phase={status['phase']} lock={held} gpu={status['gpu']} "
            f"log_age={status['sequence_log_age_seconds']} fatal={len(status['fatal_hits'])}"
        )
        for hit in status["fatal_hits"]:
            if hit not in alerted:
                alerted.add(hit)
                notify("failure", f"guardian detected fatal log line: {hit[:220]}")
        if args.once:
            break
        if not held:
            raise SystemExit(post_sequence())
        # Rebind scanning and the full-corpus audit both walk all 251,599 NPZ
        # off a full exFAT HDD and emit nothing for ~25-30 minutes; training and
        # eval should emit frequently. Alert conservatively without killing the
        # sequence, and re-arm once the log advances so one benign quiet stretch
        # does not suppress stall detection for the rest of the run.
        age = status["sequence_log_age_seconds"]
        quiet_phases = {"text_rebind", "npz_audit", "sequence_or_preflight"}
        stale_limit = 3600 if status["phase"] in quiet_phases else 1200
        if age is not None and age <= stale_limit:
            alerted.discard("stale")
        elif age is not None and "stale" not in alerted:
            alerted.add("stale")
            notify("failure", f"guardian sees stale {status['phase']} log for {int(age)} seconds")
        # Throughput collapse without staleness: progress lines keep arriving,
        # so only the rate exposes it.  Nominal is ~1.05 s/it for the rebind.
        rate = status["rate_s_per_it"]
        if rate is not None and status["phase"] in {"text_rebind", "evaluation"}:
            if rate <= 10.0:
                alerted.discard("slow")
            elif "slow" not in alerted:
                alerted.add("slow")
                notify(
                    "failure",
                    f"guardian sees {status['phase']} throughput collapse: {rate:.1f} s/it "
                    "(nominal ~1.05); check for a concurrent /mnt/HDD scan holding the exFAT lock",
                )
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
