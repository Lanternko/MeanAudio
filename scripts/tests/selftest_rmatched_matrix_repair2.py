#!/usr/bin/env python3
"""Crash/restart fixtures for the Matrix repair2 filesystem transaction."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/repair_rmatched_matrix_corrupt_flac_v2.py"
PHASES = (
    "staged_copy_verified",
    "audio_quarantined",
    "metrics_quarantined",
    "replacement_installed",
    "verified_complete",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("repair2", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repair2 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, fixture: Path) -> None:
    audio = fixture / "audio"
    metrics_dir = fixture / "metrics"
    state = fixture / "state"
    source = fixture / "source"
    module.STATE = state
    module.NVME_EVAL = fixture
    module.AUDIO = audio
    module.TARGET = audio / "target.flac"
    module.METRICS = metrics_dir / "metrics.txt"
    module.SOURCE_MANIFEST = source / "manifest.json"
    module.SOURCE_REPLAY = source / "replay"
    module.EVIDENCE = state / "evidence.json"
    module.REPORT = state / "report.json"
    module.QUARANTINED_TARGET = audio / ".target.quarantine.flac"
    module.QUARANTINED_METRICS = metrics_dir / ".metrics.quarantine.txt"
    module.STAGED_TARGET = audio / ".target.staged.flac"
    module.TRANSACTION = audio / ".target.transaction.json"
    module.HARN_KEY = state / "ledger_hmac.key"
    module.APPROVAL_STATE = state / "operator_approval.json"
    module.HARN_LOCK = state / "controller.lock"
    module.QUEUE_KEY = state / "queue_hmac.key"
    module.TARGET_ID = "target"
    module.TARGET_INDEX = 0
    module.EXPECTED_COUNT = 1


def canonical(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sign(payload: dict, key: bytes, domain: bytes) -> dict:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {**unsigned, "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()}


def start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().split()[21]


def prepare_authority(fixture: Path, *, action: str = "apply", record_status: str = "approved",
                      record_run: str | None = None, approval_state: str = "reserved",
                      capability_signed: bool = True) -> Path:
    module = load_module()
    configure(module, fixture)
    harn_key = hashlib.sha256(b"fixture-harn-key").digest()
    queue_key = hashlib.sha256(b"fixture-queue-key").digest()
    module.HARN_KEY.write_bytes(harn_key)
    module.QUEUE_KEY.write_bytes(queue_key)
    os.chmod(module.HARN_KEY, 0o600)
    os.chmod(module.QUEUE_KEY, 0o600)
    run_id = record_run or module.RUN_ID
    record = sign({
        "document_kind": "exact_operator_approval", "schema_version": 1,
        "status": record_status, "experiment_id": module.EXPERIMENT_ID, "run_id": run_id,
        "channel_record_sha256": "a" * 64, "contract_sha256": "b" * 64,
        "controller_sha256": "c" * 64, "queue_entry_binding_sha256": "d" * 64,
    }, queue_key, b"meanaudio-queue-approval-v1\0")
    record_path = module.STATE / "exact_approval.json"
    module.atomic_json(record_path, record)
    parent_pid = os.getpid()
    approval = sign({
        "document_kind": "repair2_approval_state", "schema_version": 1,
        "experiment_id": module.EXPERIMENT_ID, "run_id": module.RUN_ID,
        "state": approval_state, "approval_record_path": str(record_path),
        "approval_record_sha256": sha256(record_path), "reservation": {
            "controller_pid": parent_pid, "controller_start_ticks": start_ticks(parent_pid),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
        "channel_record_sha256": "a" * 64,
    }, harn_key, b"meanaudio-repair2-approval-state-v1\0")
    module.atomic_json(module.APPROVAL_STATE, approval)
    module.HARN_LOCK.touch(mode=0o600)
    bindings = [
        {"path": str(path), "sha256": sha256(path)}
        for path in (MODULE_PATH, module.EVIDENCE, module.SOURCE_MANIFEST)
    ]
    capability = {
        "document_kind": "repair2_write_capability", "schema_version": 1,
        "experiment_id": module.EXPERIMENT_ID, "run_id": module.RUN_ID,
        "action": action, "status": "authorized", "parent_pid": parent_pid,
        "parent_start_ticks": start_ticks(parent_pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "executable_sha256": sha256(MODULE_PATH), "harn_lock": str(module.HARN_LOCK),
        "approval_state_sha256": sha256(module.APPROVAL_STATE),
        "approval_record_sha256": sha256(record_path), "bindings": bindings,
        "writable_paths": [str(module.AUDIO), str(module.METRICS.parent), str(module.TRANSACTION.parent)],
    }
    if capability_signed:
        capability = sign(capability, harn_key, b"meanaudio-repair2-capability-v1\0")
    else:
        capability["integrity"] = "0" * 64
    capability_path = module.STATE / "capability.json"
    module.atomic_json(capability_path, capability)
    return capability_path


def invoke_child(fixture: Path, capability: Path | None, fault_phase: str | None = None) -> int:
    module = load_module()
    configure(module, fixture)
    lock_fd = os.open(module.HARN_LOCK, os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        env = {key: value for key, value in os.environ.items() if key != module.CAPABILITY_ENV}
        if capability is not None:
            env[module.CAPABILITY_ENV] = str(capability)
        argv = [sys.executable, "-X", "pycache_prefix=/dev/null", "-B", __file__, "--child", str(fixture)]
        if fault_phase:
            argv += ["--fault-phase", fault_phase]
        return subprocess.run(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    finally:
        os.close(lock_fd)


def initialize(fixture: Path) -> None:
    # Fixture generation alone needs third-party audio packages.  The child
    # authorization/repair path stays stdlib-only under the exact system Python.
    import numpy as np
    import soundfile as sf

    module = load_module()
    configure(module, fixture)
    module.AUDIO.mkdir(parents=True)
    module.METRICS.parent.mkdir(parents=True)
    module.STATE.mkdir(parents=True)
    module.SOURCE_REPLAY.mkdir(parents=True)
    original = np.zeros(module.EXPECTED_FRAMES, dtype=np.float32)
    replacement = np.full(module.EXPECTED_FRAMES, 0.125, dtype=np.float32)
    sf.write(module.TARGET, original, 16000, format="FLAC")
    sf.write(module.SOURCE_REPLAY / module.TARGET.name, replacement, 16000, format="FLAC")
    module.METRICS.write_text("invalid metrics\n", encoding="utf-8")
    module.ORIGINAL_SHA256 = sha256(module.TARGET)
    module.METRICS_SHA256 = sha256(module.METRICS)
    replacement_hash = sha256(module.SOURCE_REPLAY / module.TARGET.name)
    module.atomic_json(module.SOURCE_MANIFEST, {
        "status": "passed", "audio_sha256": {module.TARGET.name: module.ORIGINAL_SHA256},
    })
    module.atomic_json(module.EVIDENCE, {
        "schema_version": 1, "status": "passed", "prefix_replay_hash_matches": 0,
        "replacement": str(module.SOURCE_REPLAY / module.TARGET.name),
        "replacement_sha256": replacement_hash,
    })
    evidence_hash = sha256(module.EVIDENCE)
    module.atomic_json(module.TRANSACTION, {
        "schema_version": 1, "transaction_id": "fixture", "phase": "fixture_initialized",
    })
    (fixture / "fixture.json").write_text(json.dumps({
        "original_sha256": module.ORIGINAL_SHA256,
        "metrics_sha256": module.METRICS_SHA256,
        "replacement_sha256": replacement_hash,
        "evidence_sha256": evidence_hash,
    }))


def child_apply(fixture: Path, fault_phase: str | None = None) -> None:
    module = load_module()
    configure(module, fixture)
    values = json.loads((fixture / "fixture.json").read_text())
    module.ORIGINAL_SHA256 = values["original_sha256"]
    module.METRICS_SHA256 = values["metrics_sha256"]
    module.REPLACEMENT_SHA256 = values["replacement_sha256"]
    module.EVIDENCE_SHA256 = values["evidence_sha256"]
    module.FAULT_AFTER_PHASE = fault_phase
    module.apply_repair()


def assert_complete(fixture: Path) -> None:
    module = load_module()
    configure(module, fixture)
    values = json.loads((fixture / "fixture.json").read_text())
    assert sha256(module.TARGET) == values["replacement_sha256"]
    assert sha256(module.QUARANTINED_TARGET) == values["original_sha256"]
    assert sha256(module.QUARANTINED_METRICS) == values["metrics_sha256"]
    report = json.loads(module.REPORT.read_text())
    assert report["status"] == "passed"
    assert report["changed_audio_files"] == [module.TARGET.name]


def assert_initial_metrics_fail_closed(root: Path) -> None:
    for case in ("missing", "drifted"):
        fixture = root / f"metrics-{case}"
        initialize(fixture)
        module = load_module()
        configure(module, fixture)
        values = json.loads((fixture / "fixture.json").read_text())
        module.ORIGINAL_SHA256 = values["original_sha256"]
        module.METRICS_SHA256 = values["metrics_sha256"]
        module.REPLACEMENT_SHA256 = values["replacement_sha256"]
        module.EVIDENCE_SHA256 = values["evidence_sha256"]
        module.TRANSACTION.unlink()
        if case == "missing":
            module.METRICS.unlink()
        else:
            module.METRICS.write_text("drifted metrics\n", encoding="utf-8")
        before = sha256(module.TARGET)
        try:
            module.audit()
        except RuntimeError as exc:
            assert "metrics" in str(exc)
        else:
            raise AssertionError(f"{case} metrics passed initial audit")
        assert sha256(module.TARGET) == before
        assert not module.QUARANTINED_TARGET.exists()
        assert not module.QUARANTINED_METRICS.exists()
        assert not module.STAGED_TARGET.exists()


def fixture_snapshot(fixture: Path) -> dict[str, str]:
    return {
        str(path.relative_to(fixture)): sha256(path)
        for path in sorted(fixture.rglob("*")) if path.is_file()
    }


def assert_authorization_fail_closed(root: Path) -> None:
    cases = (
        ("absent", {}, None),
        ("arbitrary", {"capability_signed": False}, "capability"),
        ("rejected", {"record_status": "rejected"}, "capability"),
        ("wrong-run", {"record_run": "wrong-run"}, "capability"),
        ("consumed", {"approval_state": "consumed"}, "capability"),
    )
    for name, options, mode in cases:
        fixture = root / f"auth-{name}"
        initialize(fixture)
        capability = prepare_authority(fixture, **options)
        before = fixture_snapshot(fixture)
        result = invoke_child(fixture, capability if mode else None)
        if result == 0:
            raise AssertionError(f"unauthorized {name} Repair2 child was accepted")
        after = fixture_snapshot(fixture)
        if after != before:
            raise AssertionError(f"unauthorized {name} Repair2 child mutated fixture: {set(before) ^ set(after)}")


def assert_dangerous_environment_ignored(root: Path) -> None:
    fixture = root / "dangerous-env"
    initialize(fixture)
    capability = prepare_authority(fixture)
    previous = os.environ.get("MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE")
    os.environ["MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE"] = "audio_quarantined"
    try:
        result = invoke_child(fixture, capability)
    finally:
        if previous is None:
            os.environ.pop("MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE", None)
        else:
            os.environ["MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE"] = previous
    if result:
        raise AssertionError(f"untrusted inherited fault injector affected Repair2: {result}")
    assert_complete(fixture)


def run_parent() -> None:
    with tempfile.TemporaryDirectory(prefix="matrix-repair2-selftest-") as raw:
        root = Path(raw)
        for phase in PHASES:
            fixture = root / phase
            initialize(fixture)
            capability = prepare_authority(fixture)
            failed = invoke_child(fixture, capability, phase)
            if failed != 97:
                raise RuntimeError(f"fault injection did not stop after {phase}: {failed}")
            capability = prepare_authority(fixture)
            resumed = invoke_child(fixture, capability)
            if resumed:
                raise RuntimeError(f"resume failed after {phase}: {resumed}")
            assert_complete(fixture)
        assert_initial_metrics_fail_closed(root)
        assert_authorization_fail_closed(root)
        assert_dangerous_environment_ignored(root)
    print(f"[PASS] repair2 crash/restart phases={len(PHASES)} auth_abuse=5 env_scrub=1 metrics_fail_closed=2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--fault-phase")
    args = parser.parse_args()
    if args.child:
        child_apply(args.child, args.fault_phase)
    else:
        run_parent()


if __name__ == "__main__":
    main()
