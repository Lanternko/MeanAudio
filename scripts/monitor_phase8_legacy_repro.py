#!/usr/bin/env python3
"""Fail-closed watcher for the guarded Phase-8 legacy reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path("/home/kojiek/MeanAudio")
LOG_ROOT = Path("/home/kojiek/logs")
STATE_DIR = LOG_ROOT / "phase8_legacy_repro_guard"
STATE_FILE = STATE_DIR / "state.json"
PID_FILE = STATE_DIR / "pipeline.pid"
STATUS_FILE = STATE_DIR / "watcher_status.json"
ALERT_FILE = STATE_DIR / "ALERT.json"
VALIDATION = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz/FULL_VALIDATION.json")
GATE = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz/FULL_GATE_PASSED.json")
TRAIN_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv")
CACHE = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")
NPZ_DIR = Path("/mnt/HDD/kojiek/phase8_legacy_matched_npz")
PREFIX = "phase8_legacy_repro"
S1_EXP = f"{PREFIX}_stage1_400000"
S2_EXP = f"{PREFIX}_stage2_200000"
S1_LOG = LOG_ROOT / f"{S1_EXP}.log"
S2_LOG = LOG_ROOT / f"{S2_EXP}.log"
MASTER_LOG = LOG_ROOT / "phase8_legacy_repro_guarded.log"
PIPELINE_LOG = LOG_ROOT / "phase8_legacy_repro_pipeline.log"

LOSS_RE = re.compile(r"\bloss:\s*([^,\s]+)", re.IGNORECASE)
GRAD_RE = re.compile(r"\bgrad_norm:\s*([^,\s]+)", re.IGNORECASE)
ITER_RE = re.compile(r"\bit\s+(\d+):")
HARD_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"CUDA out of memory",
        r"Error occurred at iteration",
        r"ProcessExitedException",
        r"ChildFailedError",
        r"NCCL.*(?:error|failed)",
        r"Segmentation fault",
        r"\[FAIL\]",
    )
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=1200)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail_text(path: Path, size: int = 512 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_metadata_hashes() -> list[str]:
    errors: list[str] = []
    report = read_json(VALIDATION)
    if report.get("status") != "passed":
        return [f"missing/invalid validation report: {VALIDATION}"]
    hashes = report.get("sha256")
    paths = report.get("paths")
    if not isinstance(hashes, dict) or not isinstance(paths, dict):
        return ["validation report lacks hashes/paths"]
    for key in ("cache", "catalog", "manifest", "output_tsv"):
        path_value = paths.get(key)
        expected = hashes.get(key)
        if not isinstance(path_value, str) or not isinstance(expected, str):
            errors.append(f"validation report lacks {key} hash/path")
            continue
        path = Path(path_value)
        if not path.is_file():
            errors.append(f"validated metadata disappeared: {path}")
        elif sha256_file(path) != expected:
            errors.append(f"validated metadata hash changed: {path}")
    if read_json(GATE).get("status") != "passed":
        errors.append(f"missing/invalid semantic gate: {GATE}")
    return errors


def latest_hydra_config(exp: str) -> Path | None:
    paths = sorted((ROOT / "exps" / exp).glob("train-*-hydra/config.yaml"))
    return paths[-1] if paths else None


def verify_config(path: Path, stage: int) -> list[str]:
    errors: list[str] = []
    try:
        cfg = yaml.safe_load(path.read_text())
    except Exception as exc:  # watcher must report malformed configs clearly
        return [f"cannot parse Hydra config {path}: {exc}"]
    train = cfg.get("data", {}).get("AudioCaps_npz", {})
    expected = {
        "tsv": str(TRAIN_TSV),
        "gt_cache": str(CACHE),
        "npz_dir": str(NPZ_DIR),
    }
    for key, value in expected.items():
        if train.get(key) != value:
            errors.append(f"Stage {stage} config {key}={train.get(key)!r}, expected {value!r}")
    scalar_expected = {
        "use_q_conditioning": True,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "model": "fluxaudio_s" if stage == 1 else "meanaudio_s",
        "num_iterations": 400_000 if stage == 1 else 600_000,
    }
    for key, value in scalar_expected.items():
        if cfg.get(key) != value:
            errors.append(f"Stage {stage} config {key}={cfg.get(key)!r}, expected {value!r}")
    return errors


def parse_numeric(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def grad_health(tokens: list[str]) -> dict[str, object]:
    """Summarize AMP gradient-overflow health from logged grad norms.

    GradScaler is expected to skip an occasional optimizer step when an
    unscaled gradient is non-finite.  Treating one such event as weight
    corruption is a false positive: the following step normally recovers and
    the parameters are unchanged.  Consecutive or dense events are different
    and remain fail-closed.
    """
    values = [parse_numeric(token) for token in tokens]
    bad = [value is None or not math.isfinite(value) or value < 0 for value in values]
    trailing = 0
    for is_bad in reversed(bad):
        if not is_bad:
            break
        trailing += 1
    recent_20 = sum(bad[-20:])
    recent_100 = sum(bad[-100:])
    return {
        "nonfinite_total_in_tail": sum(bad),
        "nonfinite_recent_20": recent_20,
        "nonfinite_recent_100": recent_100,
        "nonfinite_trailing": trailing,
        "unhealthy": trailing >= 2 or recent_20 >= 3 or recent_100 >= 10,
    }


def inspect_logs(since_epoch: float = 0.0) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    metrics: dict[str, object] = {}
    for label, path in (("s1", S1_LOG), ("s2", S2_LOG), ("pipeline", PIPELINE_LOG)):
        if path.exists() and path.stat().st_mtime < since_epoch:
            continue
        text = tail_text(path)
        if not text:
            continue
        for pattern in HARD_PATTERNS:
            match = pattern.search(text)
            if match:
                errors.append(f"{label} log hard failure: {match.group(0)}")
        losses = LOSS_RE.findall(text)
        grads = GRAD_RE.findall(text)
        iterations = [int(value) for value in ITER_RE.findall(text)]
        if iterations:
            metrics[f"{label}_latest_iteration"] = iterations[-1]
        if losses:
            value = parse_numeric(losses[-1])
            metrics[f"{label}_latest_loss"] = losses[-1]
            if value is None or not math.isfinite(value) or value < 0 or value > 1000:
                errors.append(f"{label} invalid loss: {losses[-1]}")
        if grads:
            value = parse_numeric(grads[-1])
            metrics[f"{label}_latest_grad_norm"] = grads[-1]
            health = grad_health(grads)
            for key, health_value in health.items():
                if key != "unhealthy":
                    metrics[f"{label}_grad_{key}"] = health_value
            if value is not None and math.isfinite(value) and value > 1_000_000:
                errors.append(f"{label} invalid grad_norm: {grads[-1]}")
            elif health["unhealthy"]:
                errors.append(
                    f"{label} persistent/dense non-finite grad_norm: "
                    f"trailing={health['nonfinite_trailing']}, "
                    f"recent20={health['nonfinite_recent_20']}, "
                    f"recent100={health['nonfinite_recent_100']}"
                )
    return errors, metrics


def gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def newest_log_age() -> float | None:
    mtimes = [
        path.stat().st_mtime
        for path in (MASTER_LOG, PIPELINE_LOG, S1_LOG, S2_LOG)
        if path.exists()
    ]
    return time.time() - max(mtimes) if mtimes else None


def terminate_pipeline(pid: int | None) -> None:
    if not process_alive(pid):
        return
    assert pid is not None
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def alert(errors: list[str], status: dict[str, object], pid: int | None) -> None:
    payload = {**status, "alerted_at": now(), "errors": errors}
    atomic_json(ALERT_FILE, payload)
    terminate_pipeline(pid)
    print(f"[ALERT] {'; '.join(errors)}", flush=True)


def self_test() -> None:
    normal = "it 100: grad_norm:9.63, loss:0.994, lr:0.0001"
    bad_loss = "it 100: grad_norm:9.63, loss:nan, lr:0.0001"
    harmless = "Error in extra logging: Could not load libtorchcodec"
    assert not any(pattern.search(normal) for pattern in HARD_PATTERNS)
    assert not any(pattern.search(harmless) for pattern in HARD_PATTERNS)
    value = parse_numeric(LOSS_RE.findall(normal)[-1])
    assert value is not None and math.isfinite(value)
    value = parse_numeric(LOSS_RE.findall(bad_loss)[-1])
    assert value is not None and not math.isfinite(value)
    assert not grad_health(["2.5", "nan", "2.6"])["unhealthy"]
    assert grad_health(["2.5", "nan", "nan"])["unhealthy"]
    assert grad_health(["nan", "2.5", "nan", "2.5", "nan"])["unhealthy"]
    assert any(pattern.search("CUDA out of memory") for pattern in HARD_PATTERNS)
    print("[PASS] watcher self-test")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.interval <= 0 or args.stale_seconds <= 0:
        raise SystemExit("Watcher interval/stale timeout must be positive")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ALERT_FILE.unlink(missing_ok=True)
    dead_cycles = 0
    cycle = 0
    while True:
        state = read_json(STATE_FILE)
        phase = str(state.get("phase", "UNKNOWN"))
        pid = read_pid()
        alive = process_alive(pid)
        phase_started_epoch = float(state.get("phase_started_epoch", 0.0) or 0.0)
        errors, metrics = inspect_logs(
            phase_started_epoch if phase == "TRAINING" else 0.0
        )

        if phase == "TRAINING":
            if alive:
                dead_cycles = 0
            else:
                dead_cycles += 1
                if dead_cycles >= 2:
                    errors.append("pipeline process died while state remained TRAINING")

            if cycle % 30 == 0:
                errors.extend(verify_metadata_hashes())
            for stage, exp in ((1, S1_EXP), (2, S2_EXP)):
                config = latest_hydra_config(exp)
                if config is not None:
                    errors.extend(verify_config(config, stage))
            age = newest_log_age()
            if age is not None and age > args.stale_seconds:
                errors.append(f"all pipeline/training logs stale for {age:.0f}s")

        status: dict[str, object] = {
            "checked_at": now(),
            "cycle": cycle,
            "phase": phase,
            "pipeline_pid": pid,
            "pipeline_alive": alive,
            "gpu": gpu_snapshot(),
            "log_age_seconds": newest_log_age(),
            "metrics": metrics,
            "errors": errors,
        }
        atomic_json(STATUS_FILE, status)
        print(
            f"[watcher] phase={phase} alive={alive} gpu={status['gpu']} "
            f"metrics={metrics}",
            flush=True,
        )
        if errors:
            alert(errors, status, pid)
            raise SystemExit(2)
        if phase == "DONE":
            print("[watcher] guarded run completed", flush=True)
            return
        if phase == "FAILED":
            raise SystemExit(2)
        cycle += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
