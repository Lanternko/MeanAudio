#!/usr/bin/env python3
"""Synthetic, CPU-only tests for the Phase-8 quarter backlog watcher."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WATCHER = ROOT / "scripts" / "monitor_phase8_qwen_bucket_quarter_backlog.py"


def load_watcher():
    spec = importlib.util.spec_from_file_location("quarter_backlog_watcher", WATCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def complete_arm(mod, root: Path, logs: Path, key: str) -> None:
    spec = next(item for item in mod.queue_specs() if item.key == key)
    paths = mod.arm_paths(spec, root, logs)
    audit_path = paths["audit"]
    if spec.historical_reuse:
        audit_path = logs / f"historical_{key}_FINAL_TRAIN_AUDIT.json"
    audit = {
        "status": "passed",
        "scale": "quarter",
        "stage1_iteration": 100000,
        "stage2_iteration": 150000,
    }
    if key == "noq":
        audit.update({
            "arm": "noq",
            "matched_bucket_arm": "k2_balanced",
            "stage1_use_q_conditioning": False,
            "stage2_use_q_conditioning": False,
        })
    elif spec.historical_reuse:
        audit.update({
            "arm": "halfq" if key == "k2_balanced" else "fullq",
            "stage1_use_q_conditioning": True,
            "stage2_use_q_conditioning": True,
        })
    else:
        k_raw, strategy = key.split("_", 1)
        audit.update({
            "k": int(k_raw[1:]),
            "strategy": strategy,
            "q_conditioning": True,
        })
    write_json(audit_path, audit)
    metrics = {
        "status": "passed",
        "experiment": spec.prefix,
        "scale": "quarter",
        "clap_score": 0.2,
    }
    if key == "noq":
        metrics.update({"arm": "noq", "matched_bucket_arm": "k2_balanced"})
    else:
        k_raw, strategy = key.split("_", 1)
        metrics.update({
            "k": int(k_raw[1:]),
            "strategy": strategy,
            "historical_checkpoint_reused": spec.historical_reuse,
            "training_audit": str(audit_path),
        })
    write_json(
        paths["metrics"],
        metrics,
    )


def gpu_ok() -> dict:
    return {
        "status": "ok",
        "gpus": [{
            "index": 0, "util_pct": 80.0, "mem_used_mib": 12000.0,
            "mem_total_mib": 24000.0, "temp_c": 60.0,
        }],
    }


def test_queue_and_self_test(mod) -> None:
    mod.self_test()
    assert [item.key for item in mod.queue_specs()] == [
        "noq", "k2_balanced", "k5_balanced", "k10_balanced",
        "k3_balanced", "k5_fixed", "k10_fixed",
    ]
    assert [item.lane for item in mod.queue_specs()] == [
        "main", "main", "main", "main", "backup", "backup", "backup",
    ]


def test_historical_reuse_is_normal_completion(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        complete_arm(mod, root, logs, "noq")
        complete_arm(mod, root, logs, "k2_balanced")
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[], tmux=[],
            gpu=gpu_ok(), now=2_000_000_000,
            transition_grace_seconds=10**10,
        )
        arm = next(x for x in snapshot["arms"] if x["key"] == "k2_balanced")
        assert arm["state"] == "complete"
        assert arm["completion_mode"] == "historical_reuse_report_validation"
        assert arm["final_train_audit"]["referenced_by_final_metrics"] is True
        assert not any(
            x["code"] == "invalid_final_train_audit"
            for x in snapshot["issues"]
        )


def test_pending_backlog_is_nonfatal(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        complete_arm(mod, base / "repo", base / "logs", "noq")
        snapshot = mod.collect(
            root=base / "repo", log_root=base / "logs",
            processes=[], tmux=[], gpu=gpu_ok(), now=2_000_000_000,
        )
        assert snapshot["status"] == "healthy"
        assert snapshot["queue"]["first_incomplete"] == "k2_balanced"
        assert snapshot["handoff"]["state"] == "backlog_not_started"


def test_active_arm_progress_and_handoff(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        complete_arm(mod, root, logs, "noq")
        spec = next(x for x in mod.queue_specs() if x.key == "k2_balanced")
        paths = mod.arm_paths(spec, root, logs)
        paths["stage1_log"].parent.mkdir(parents=True, exist_ok=True)
        paths["stage1_log"].write_text(
            "it 1250: grad_norm:1.2, loss:0.98, lr:0.0001\n",
            encoding="utf-8",
        )
        now = paths["stage1_log"].stat().st_mtime + 10
        process = (
            "123 00:10 99 2 torchrun train.py "
            "exp_id=phase8_qwen_bucket_quarter_k2_balanced_stage1_100000"
        )
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[process], tmux=[],
            gpu=gpu_ok(), now=now,
        )
        assert snapshot["status"] == "healthy"
        assert snapshot["active_arm"] == "k2_balanced"
        assert snapshot["handoff"]["state"] == "connected"
        arm = next(x for x in snapshot["arms"] if x["key"] == "k2_balanced")
        assert arm["latest_iteration"] == 1250
        assert arm["latest_metrics"]["loss"] == 0.98


def test_nonfinite_severity(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        complete_arm(mod, root, logs, "noq")
        spec = next(x for x in mod.queue_specs() if x.key == "k2_balanced")
        paths = mod.arm_paths(spec, root, logs)
        paths["stage1_log"].parent.mkdir(parents=True, exist_ok=True)
        process = f"123 train.py exp_id={spec.prefix}_stage1_100000"

        paths["stage1_log"].write_text(
            "it 50: grad_norm:nan, loss:1.0, lr:0.0001\n"
            "it 100: grad_norm:1.1, loss:1.0, lr:0.0001\n",
            encoding="utf-8",
        )
        now = paths["stage1_log"].stat().st_mtime + 1
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[process], tmux=[],
            gpu=gpu_ok(), now=now,
        )
        assert snapshot["status"] == "transient_or_nonfatal"
        assert any(
            x["code"] == "transient_amp_grad_overflow"
            for x in snapshot["transient_nonfatal"]
        )

        paths["stage1_log"].write_text(
            "it 50: grad_norm:nan, loss:1.0, lr:0.0001\n"
            "it 100: grad_norm:inf, loss:1.0, lr:0.0001\n",
            encoding="utf-8",
        )
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[process], tmux=[],
            gpu=gpu_ok(), now=paths["stage1_log"].stat().st_mtime + 1,
        )
        assert snapshot["status"] == "hard_incident"
        assert any(
            x["code"] == "persistent_nonfinite_grad"
            for x in snapshot["hard_incidents"]
        )

        paths["stage1_log"].write_text(
            "it 150: grad_norm:1.0, loss:nan, lr:0.0001\n",
            encoding="utf-8",
        )
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[process], tmux=[],
            gpu=gpu_ok(), now=paths["stage1_log"].stat().st_mtime + 1,
        )
        assert any(
            x["code"] == "nonfinite_metric" for x in snapshot["hard_incidents"]
        )


def test_gpu_error_never_becomes_idle(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        complete_arm(mod, root, logs, "noq")
        spec = next(x for x in mod.queue_specs() if x.key == "k2_balanced")
        paths = mod.arm_paths(spec, root, logs)
        paths["stage1_log"].parent.mkdir(parents=True, exist_ok=True)
        paths["stage1_log"].write_text(
            "it 50: grad_norm:1.0, loss:1.0, lr:0.0001\n",
            encoding="utf-8",
        )
        process = f"123 train.py exp_id={spec.prefix}_stage1_100000"
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[process], tmux=[],
            gpu={"status": "error", "gpu_query_error": "driver unavailable"},
            now=paths["stage1_log"].stat().st_mtime + 400,
            stale_seconds=1200,
        )
        codes = {x["code"] for x in snapshot["issues"]}
        assert "gpu_query_error" in codes
        assert "gpu_idle_with_stale_progress" not in codes
        assert snapshot["status"] == "transient_or_nonfatal"


def test_driver_library_mismatch_is_recoverable(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        spec = next(x for x in mod.queue_specs() if x.key == "noq")
        paths = mod.arm_paths(spec, root, logs)
        paths["wrapper"].parent.mkdir(parents=True, exist_ok=True)
        paths["wrapper"].write_text(
            "ncclSystemError: external library call failed\n"
            "nvmlInit_v2() failed: Driver/library version mismatch\n",
            encoding="utf-8",
        )
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[], tmux=[],
            gpu={"status": "error", "gpu_query_error":
                 "Driver/library version mismatch"},
            now=paths["wrapper"].stat().st_mtime + 1,
        )
        assert snapshot["status"] == "transient_or_nonfatal"
        assert not snapshot["hard_incidents"]
        assert any(
            x["code"] == "nvidia_driver_library_mismatch"
            for x in snapshot["transient_nonfatal"]
        )


def test_disconnected_next_arm_is_hard(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        complete_arm(mod, root, logs, "noq")
        complete_arm(mod, root, logs, "k2_balanced")
        # Make the completed artifacts old enough to exceed handoff grace.
        old = 1_000_000_000
        spec = next(x for x in mod.queue_specs() if x.key == "k2_balanced")
        paths = mod.arm_paths(spec, root, logs)
        report = json.loads(paths["metrics"].read_text())
        os.utime(Path(report["training_audit"]), (old, old))
        os.utime(paths["metrics"], (old, old))
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[], tmux=[], gpu=gpu_ok(),
            now=old + 3600, transition_grace_seconds=900,
        )
        assert snapshot["handoff"]["state"] == "disconnected"
        assert any(
            x["code"] == "next_arm_not_connected"
            for x in snapshot["hard_incidents"]
        )


def test_invalid_final_json_and_atomic_write(mod) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root, logs = base / "repo", base / "logs"
        spec = mod.queue_specs()[0]
        paths = mod.arm_paths(spec, root, logs)
        paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
        paths["metrics"].write_text("{broken", encoding="utf-8")
        snapshot = mod.collect(
            root=root, log_root=logs, processes=[], tmux=[], gpu=gpu_ok(),
        )
        assert any(
            x["code"] == "invalid_final_metrics"
            for x in snapshot["hard_incidents"]
        )
        status = base / "state" / "status.json"
        mod.atomic_json(status, snapshot)
        assert json.loads(status.read_text())["status"] == "hard_incident"
        assert not list(status.parent.glob("*.tmp.*"))


def test_cli_self_test_and_syntax() -> None:
    for command in (
        [sys.executable, "-m", "py_compile", str(WATCHER), str(Path(__file__))],
        [sys.executable, str(WATCHER), "--self-test"],
    ):
        proc = subprocess.run(
            command, text=True, capture_output=True, check=False,
        )
        assert proc.returncode == 0, (
            f"command failed: {command}\n{proc.stdout}\n{proc.stderr}"
        )


def main() -> None:
    mod = load_watcher()
    tests = [
        ("test_queue_and_self_test", lambda: test_queue_and_self_test(mod)),
        (
            "test_pending_backlog_is_nonfatal",
            lambda: test_pending_backlog_is_nonfatal(mod),
        ),
        (
            "test_historical_reuse_is_normal_completion",
            lambda: test_historical_reuse_is_normal_completion(mod),
        ),
        (
            "test_active_arm_progress_and_handoff",
            lambda: test_active_arm_progress_and_handoff(mod),
        ),
        ("test_nonfinite_severity", lambda: test_nonfinite_severity(mod)),
        (
            "test_gpu_error_never_becomes_idle",
            lambda: test_gpu_error_never_becomes_idle(mod),
        ),
        (
            "test_driver_library_mismatch_is_recoverable",
            lambda: test_driver_library_mismatch_is_recoverable(mod),
        ),
        (
            "test_disconnected_next_arm_is_hard",
            lambda: test_disconnected_next_arm_is_hard(mod),
        ),
        (
            "test_invalid_final_json_and_atomic_write",
            lambda: test_invalid_final_json_and_atomic_write(mod),
        ),
        ("test_cli_self_test_and_syntax", test_cli_self_test_and_syntax),
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"[PASS] {name}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}")
    if failures:
        raise SystemExit(f"{failures} self-test(s) failed")
    print(f"[OK] all {len(tests)} self-tests passed")


if __name__ == "__main__":
    main()
