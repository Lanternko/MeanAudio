#!/usr/bin/env python3
"""Cross-filesystem-safe installation of the verified Matrix RNG replay."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
NVME_EVAL = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/eval_output")
STATE = Path("/home/kojiek/logs/rmatched_matrix_repair2_harn")
SOURCE_STATE = Path("/home/kojiek/logs/rmatched_matrix_repair_harn")
TARGET_ID = "5xIBQGMjiX4_30"
TARGET_INDEX = 707
EXPECTED_COUNT = 5521
EXPECTED_FRAMES = 159744
ORIGINAL_SHA256 = "c3472953aa061d327979ab97cc736dcf772e6d98a4148c815b28495a865d947e"
AUDIO = NVME_EVAL / "rmatched_s1_s2_steps_cfg_matrix_seed14159265/s2_mf25_cfg0p5/audio"
TARGET = AUDIO / f"{TARGET_ID}.flac"
METRICS = NVME_EVAL / "metrics/rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5/metrics.txt"
SOURCE_MANIFEST = SOURCE_STATE / "repair_pre_manifest.json"
SOURCE_REPLAY = SOURCE_STATE / "rng_replay/audio"
EVIDENCE = STATE / "verified_rng_replay_evidence.json"
REPORT = STATE / "repair_report.json"
QUARANTINED_TARGET = AUDIO / f".{TARGET_ID}.{ORIGINAL_SHA256}.repair-quarantine.flac"
QUARANTINED_METRICS = METRICS.parent / ".invalid_metrics.repair-quarantine.txt"
STAGED_TARGET = AUDIO / f".{TARGET_ID}.repair2-staged.flac"
TRANSACTION = NVME_EVAL / f".repair2_state/{TARGET_ID}.repair2-transaction.json"
METRICS_SHA256 = "a1296812a3966c21bd1e5a60f4edeae4fc10c0ed45bb5ed72f6491a384398399"
REPAIR1_APPROVAL_SHA256 = "6d93c976c3319491d291d8b829ddd958116b5a01782ae4839ee2b6b737e96a77"
REPLACEMENT_SHA256 = "8f50c8e574bd9cf1e638a4e34218dd2276accf73ea0503942aedc22a39e672f6"
EVIDENCE_SHA256 = "2de64da6521ee514ebaac2095402b39ae0e79305a0a0bfd1c7ed2579e37f7d12"
EXPERIMENT_ID = "rmatched-s1-s2-steps-cfg-matrix-repair2"
RUN_ID = "run-20260814-seed14159265-musiccaps-repair2"
HARN_KEY = STATE / "ledger_hmac.key"
APPROVAL_STATE = STATE / "operator_approval.json"
HARN_LOCK = STATE / "controller.lock"
QUEUE_KEY = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue_hmac.key")
CAPABILITY_ENV = "MEANAUDIO_REPAIR2_CAPABILITY"
# Tests set this module variable in an isolated fixture.  Production execution
# never accepts the former inherited environment-variable fault injector.
FAULT_AFTER_PHASE: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _load_private_key(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.geteuid() or (stat.st_mode & 0o777) != 0o600:
            raise RuntimeError(f"unsafe authentication key: {path}")
        key = os.read(fd, 128)
    finally:
        os.close(fd)
    if len(key) < 32:
        raise RuntimeError(f"invalid authentication key: {path}")
    return key


def _verify_mac(payload: dict[str, Any], key: bytes, domain: bytes) -> None:
    supplied = payload.get("integrity")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise RuntimeError("authenticated document signature is missing")
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated document signature is invalid")


def process_start_ticks(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError) as exc:
        raise RuntimeError("authorized parent identity is unavailable") from exc


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _parent_holds_lock(parent_pid: int, lock_path: Path) -> bool:
    lock_stat = lock_path.stat()
    parent_has_descriptor = False
    fd_root = Path(f"/proc/{parent_pid}/fd")
    try:
        descriptors = list(fd_root.iterdir())
    except OSError:
        return False
    for descriptor in descriptors:
        try:
            stat = descriptor.stat()
        except OSError:
            continue
        if (stat.st_dev, stat.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
            parent_has_descriptor = True
            break
    if not parent_has_descriptor:
        return False
    fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def authorize_write(action: str) -> dict[str, Any]:
    """Claim an exact, one-use HARN-issued capability before any write."""
    raw_path = os.environ.get(CAPABILITY_ENV, "")
    if not raw_path:
        raise RuntimeError("repair2 write capability is missing")
    capability_path = Path(raw_path)
    capability = json.loads(capability_path.read_text())
    harn_key = _load_private_key(HARN_KEY)
    _verify_mac(capability, harn_key, b"meanaudio-repair2-capability-v1\0")
    parent_pid = os.getppid()
    required = {
        "document_kind": "repair2_write_capability",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "action": action,
        "status": "authorized",
        "parent_pid": parent_pid,
        "parent_start_ticks": process_start_ticks(parent_pid),
        "boot_id": boot_id(),
        "executable_sha256": sha256(Path(__file__)),
        "harn_lock": str(HARN_LOCK),
        "writable_paths": [str(AUDIO), str(METRICS.parent), str(TRANSACTION.parent)],
    }
    mismatches = {key: (expected, capability.get(key)) for key, expected in required.items()
                  if capability.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"repair2 capability binding mismatch: {mismatches}")
    legacy = (ROOT / "eval_output").resolve()
    for raw in required["writable_paths"]:
        resolved = Path(raw).resolve(strict=False)
        if resolved == legacy or resolved.is_relative_to(legacy):
            raise RuntimeError("legacy HDD eval_output leaked into Repair2 write capability")
        if not resolved.is_relative_to(NVME_EVAL.resolve(strict=False)):
            raise RuntimeError("Repair2 write capability escaped the NVMe final tree")
    if not _parent_holds_lock(parent_pid, HARN_LOCK):
        raise RuntimeError("authorized Repair2 HARN lock is not held by the parent")

    approval = json.loads(APPROVAL_STATE.read_text())
    _verify_mac(approval, harn_key, b"meanaudio-repair2-approval-state-v1\0")
    if approval.get("experiment_id") != EXPERIMENT_ID or approval.get("run_id") != RUN_ID:
        raise RuntimeError("repair2 approval state is bound to a different run")
    if approval.get("state") != "reserved":
        raise RuntimeError("repair2 approval is not reserved and unconsumed")
    reservation = approval.get("reservation", {})
    if (reservation.get("controller_pid") != parent_pid
            or reservation.get("controller_start_ticks") != process_start_ticks(parent_pid)
            or reservation.get("boot_id") != boot_id()):
        raise RuntimeError("repair2 approval reservation belongs to a different controller")
    approval_sha = sha256(APPROVAL_STATE)
    if capability.get("approval_state_sha256") != approval_sha:
        raise RuntimeError("repair2 capability approval-state binding drift")

    record_path = Path(str(approval.get("approval_record_path", "")))
    if not record_path.is_file() or sha256(record_path) != approval.get("approval_record_sha256"):
        raise RuntimeError("exact repair2 approval record is missing or drifted")
    record = json.loads(record_path.read_text())
    _verify_mac(record, _load_private_key(QUEUE_KEY), b"meanaudio-queue-approval-v1\0")
    if (record.get("document_kind") != "exact_operator_approval"
            or record.get("status") != "approved"
            or record.get("experiment_id") != EXPERIMENT_ID
            or record.get("run_id") != RUN_ID
            or record.get("channel_record_sha256") == REPAIR1_APPROVAL_SHA256):
        raise RuntimeError("exact repair2 approval record is rejected or wrong-run")
    if capability.get("approval_record_sha256") != sha256(record_path):
        raise RuntimeError("repair2 capability approval-record binding drift")

    bindings = capability.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise RuntimeError("repair2 capability has no immutable bindings")
    for binding in bindings:
        path = Path(str(binding.get("path", "")))
        expected = binding.get("sha256")
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise RuntimeError(f"repair2 preregistered binding drift: {path}")

    # The claim is itself an authenticated state transition.  It occurs only
    # after every trust-boundary check, so rejected direct CLI calls make no
    # filesystem mutation at all.
    capability["status"] = "claimed"
    capability["claimed_by_pid"] = os.getpid()
    capability["claimed_at"] = now()
    unsigned = {key: value for key, value in capability.items() if key != "integrity"}
    capability["integrity"] = hmac.new(
        harn_key, b"meanaudio-repair2-capability-v1\0" + canonical(unsigned), hashlib.sha256,
    ).hexdigest()
    atomic_json(capability_path, capability)
    return capability


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some exFAT/FUSE implementations do not support directory fsync.
        pass
    finally:
        os.close(fd)


def record_phase(phase: str, **evidence: Any) -> None:
    prior = json.loads(TRANSACTION.read_text()) if TRANSACTION.is_file() else {}
    atomic_json(TRANSACTION, {
        **prior, "schema_version": 1, "transaction_id": "rmatched-matrix-repair2",
        "phase": phase, "updated_at": now(), **evidence,
    })
    if FAULT_AFTER_PHASE == phase:
        os._exit(97)


def validate_audio(path: Path) -> None:
    # FLAC stores integer PCM, so decoded samples cannot encode NaN/Inf.  Parse
    # the mandatory STREAMINFO block directly to keep the privileged repair
    # child stdlib-only and independent of unverified numpy/libsndfile code.
    with path.open("rb") as handle:
        if handle.read(4) != b"fLaC":
            raise RuntimeError("replacement is not a FLAC stream")
        header = handle.read(4)
        if len(header) != 4 or (header[0] & 0x7F) != 0:
            raise RuntimeError("replacement has no leading FLAC STREAMINFO block")
        length = int.from_bytes(header[1:4])
        streaminfo = handle.read(length)
    if length != 34 or len(streaminfo) != 34:
        raise RuntimeError("replacement FLAC STREAMINFO is invalid")
    packed = int.from_bytes(streaminfo[10:18])
    sample_rate = (packed >> 44) & ((1 << 20) - 1)
    channels = ((packed >> 41) & 0x7) + 1
    frames = packed & ((1 << 36) - 1)
    shape = (frames, sample_rate, channels)
    if shape != (EXPECTED_FRAMES, 16000, 1):
        raise RuntimeError(f"invalid replacement shape: {shape}")


def audit() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("status") != "passed" or manifest.get("audio_sha256", {}).get(TARGET.name) != ORIGINAL_SHA256:
        raise RuntimeError("repair1 source manifest is invalid")
    if sha256(TARGET) != ORIGINAL_SHA256:
        raise RuntimeError("original corrupt target hash drift")
    if not METRICS.is_file() or sha256(METRICS) != METRICS_SHA256:
        raise RuntimeError("registered invalid metrics are missing or drifted")
    unexpected = [
        str(path) for path in (QUARANTINED_TARGET, QUARANTINED_METRICS, STAGED_TARGET, TRANSACTION)
        if path.exists()
    ]
    if unexpected:
        raise RuntimeError(f"repair2 transaction paths already exist before mutation: {unexpected}")
    replay_files = sorted(SOURCE_REPLAY.glob("*.flac"))
    if len(replay_files) != TARGET_INDEX + 1:
        raise RuntimeError(f"verified replay count drift: {len(replay_files)}")
    mismatches = []
    target_seen = False
    prefix_matches = 0
    for replay in replay_files:
        if replay.name == TARGET.name:
            target_seen = True
            continue
        existing_hash = manifest["audio_sha256"].get(replay.name)
        replay_hash = sha256(replay)
        if existing_hash != replay_hash:
            mismatches.append({"name": replay.name, "existing": existing_hash, "replay": replay_hash})
            if len(mismatches) >= 10:
                break
        prefix_matches += 1
    if mismatches or prefix_matches != TARGET_INDEX or not target_seen:
        raise RuntimeError(f"RNG replay evidence mismatch: count={prefix_matches} mismatches={mismatches}")
    replacement = SOURCE_REPLAY / TARGET.name
    validate_audio(replacement)
    if sha256(replacement) != REPLACEMENT_SHA256:
        raise RuntimeError("registered replacement hash drift")
    payload = {
        "schema_version": 1, "status": "passed", "created_at": now(),
        "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_replay": str(SOURCE_REPLAY), "prefix_replay_hash_matches": prefix_matches,
        "replacement": str(replacement), "replacement_sha256": sha256(replacement),
        "original_target": str(TARGET), "original_sha256": ORIGINAL_SHA256,
    }
    if EVIDENCE.is_file():
        existing = json.loads(EVIDENCE.read_text())
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        expected = {key: value for key, value in payload.items() if key != "created_at"}
        if comparable != expected:
            raise RuntimeError("registered RNG replay evidence drift")
        if sha256(EVIDENCE) != EVIDENCE_SHA256:
            raise RuntimeError("registered RNG replay evidence hash drift")
        return existing
    raise RuntimeError("registered RNG replay evidence is missing")


def copy_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    fsync_directory(destination.parent)


def file_hash(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def reconcile_filesystem(evidence: dict[str, Any]) -> str:
    replacement_hash = evidence["replacement_sha256"]
    observed = {
        "target": file_hash(TARGET),
        "staged": file_hash(STAGED_TARGET),
        "quarantined_target": file_hash(QUARANTINED_TARGET),
        "metrics": file_hash(METRICS),
        "quarantined_metrics": file_hash(QUARANTINED_METRICS),
    }
    original_pair = (
        observed["target"] == ORIGINAL_SHA256
        and observed["quarantined_target"] is None
        and observed["metrics"] == METRICS_SHA256
        and observed["quarantined_metrics"] is None
    )
    audio_quarantined = (
        observed["target"] is None
        and observed["quarantined_target"] == ORIGINAL_SHA256
        and observed["staged"] == replacement_hash
        and observed["metrics"] == METRICS_SHA256
        and observed["quarantined_metrics"] is None
    )
    metrics_quarantined = (
        observed["target"] is None
        and observed["quarantined_target"] == ORIGINAL_SHA256
        and observed["staged"] == replacement_hash
        and observed["metrics"] is None
        and observed["quarantined_metrics"] == METRICS_SHA256
    )
    installed = (
        observed["target"] == replacement_hash
        and observed["quarantined_target"] == ORIGINAL_SHA256
        and observed["staged"] is None
        and observed["metrics"] is None
        and observed["quarantined_metrics"] == METRICS_SHA256
    )
    if installed:
        phase = "replacement_installed"
    elif metrics_quarantined:
        phase = "metrics_quarantined"
    elif audio_quarantined:
        phase = "audio_quarantined"
    elif original_pair:
        phase = "original_intact"
    else:
        raise RuntimeError(f"ambiguous repair2 filesystem state: {observed}")
    record_phase(phase, observed=observed)
    return phase


def apply_repair() -> None:
    authorize_write("apply")
    if REPORT.is_file() and json.loads(REPORT.read_text()).get("status") == "passed":
        validate_audio(TARGET)
        return
    evidence = json.loads(EVIDENCE.read_text()) if TRANSACTION.is_file() else audit()
    if evidence.get("status") != "passed" or evidence.get("prefix_replay_hash_matches") != TARGET_INDEX:
        raise RuntimeError("registered RNG replay evidence is invalid")
    if sha256(EVIDENCE) != EVIDENCE_SHA256 or evidence.get("replacement_sha256") != REPLACEMENT_SHA256:
        raise RuntimeError("registered RNG replay evidence binding is invalid")
    replacement = Path(evidence["replacement"])
    phase = reconcile_filesystem(evidence)
    if phase == "original_intact":
        if STAGED_TARGET.exists():
            if sha256(STAGED_TARGET) != evidence["replacement_sha256"]:
                raise RuntimeError("stale staged replacement hash mismatch")
        else:
            copy_fsync(replacement, STAGED_TARGET)
        validate_audio(STAGED_TARGET)
        if sha256(STAGED_TARGET) != evidence["replacement_sha256"]:
            raise RuntimeError("cross-filesystem staged copy hash mismatch")
        record_phase("staged_copy_verified", staged_sha256=sha256(STAGED_TARGET))
        os.replace(TARGET, QUARANTINED_TARGET)
        fsync_directory(AUDIO)
        record_phase("audio_quarantined", quarantined_target_sha256=sha256(QUARANTINED_TARGET))
        phase = "audio_quarantined"
    if phase == "audio_quarantined":
        if METRICS.is_file():
            if sha256(METRICS) != METRICS_SHA256:
                raise RuntimeError("invalid prior metrics hash drift")
            os.replace(METRICS, QUARANTINED_METRICS)
            fsync_directory(METRICS.parent)
        record_phase("metrics_quarantined", quarantined_metrics_sha256=file_hash(QUARANTINED_METRICS))
        phase = "metrics_quarantined"
    if phase == "metrics_quarantined":
        os.replace(STAGED_TARGET, TARGET)
        fsync_directory(AUDIO)
        record_phase("replacement_installed", replacement_sha256=sha256(TARGET))
    validate_audio(TARGET)
    if sha256(QUARANTINED_TARGET) != ORIGINAL_SHA256 or sha256(TARGET) != evidence["replacement_sha256"]:
        raise RuntimeError("post-install repair hashes failed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    current = {path.name: sha256(path) for path in sorted(AUDIO.glob("*.flac"))}
    changed = [name for name, old in manifest["audio_sha256"].items() if current.get(name) != old]
    if changed != [TARGET.name]:
        raise RuntimeError(f"repair2 mutation boundary failed: {changed}")
    record_phase("verified_complete", changed_audio_files=changed)
    approval_sha256 = json.loads(APPROVAL_STATE.read_text()).get("channel_record_sha256", "")
    if len(approval_sha256) != 64 or any(character not in "0123456789abcdef" for character in approval_sha256):
        raise RuntimeError("repair2 exact approval sha256 is missing or invalid")
    if approval_sha256 == REPAIR1_APPROVAL_SHA256:
        raise RuntimeError("repair1 authorization cannot authorize repair2")
    atomic_json(REPORT, {
        "schema_version": 1, "status": "passed", "completed_at": now(),
        "operator_authorization_sha256": approval_sha256,
        "rng_replay_evidence": str(EVIDENCE), "rng_replay_evidence_sha256": sha256(EVIDENCE),
        "prefix_replay_hash_matches": TARGET_INDEX,
        "original_sha256": ORIGINAL_SHA256, "replacement_sha256": sha256(TARGET),
        "changed_audio_files": changed, "quarantined_target": str(QUARANTINED_TARGET),
        "invalidated_metrics": str(QUARANTINED_METRICS) if QUARANTINED_METRICS.exists() else None,
        "transaction": str(TRANSACTION), "transaction_sha256": sha256(TRANSACTION),
    })


def rollback() -> None:
    authorize_write("rollback")
    report = json.loads(REPORT.read_text())
    if report.get("status") != "passed" or not QUARANTINED_TARGET.is_file():
        raise RuntimeError("no completed repair2 is available to roll back")
    replacement_backup = AUDIO / f".{TARGET.name}.repair2-replacement.flac"
    os.replace(TARGET, replacement_backup)
    os.replace(QUARANTINED_TARGET, TARGET)
    if QUARANTINED_METRICS.exists():
        os.replace(QUARANTINED_METRICS, METRICS)
    if sha256(TARGET) != ORIGINAL_SHA256:
        raise RuntimeError("repair2 rollback hash failed")
    fsync_directory(AUDIO)
    fsync_directory(METRICS.parent)
    record_phase("rolled_back", replacement_backup=str(replacement_backup))
    report.update({"status": "rolled_back", "rolled_back_at": now(), "replacement_backup": str(replacement_backup)})
    atomic_json(REPORT, report)


def reconcile() -> str:
    authorize_write("reconcile")
    if not TRANSACTION.is_file():
        raise RuntimeError("repair2 transaction has not started")
    return reconcile_filesystem(json.loads(EVIDENCE.read_text()))


def validate_report(expected_approval_sha256: str | None = None) -> dict[str, Any]:
    report = json.loads(REPORT.read_text())
    expected = {
        "status": "passed",
        "prefix_replay_hash_matches": TARGET_INDEX,
        "original_sha256": ORIGINAL_SHA256,
        "replacement_sha256": REPLACEMENT_SHA256,
        "changed_audio_files": [TARGET.name],
        "rng_replay_evidence_sha256": EVIDENCE_SHA256,
    }
    mismatches = {key: {"expected": value, "observed": report.get(key)} for key, value in expected.items()
                  if report.get(key) != value}
    approval = report.get("operator_authorization_sha256", "")
    if (len(approval) != 64 or any(character not in "0123456789abcdef" for character in approval)
            or approval == REPAIR1_APPROVAL_SHA256):
        mismatches["operator_authorization_sha256"] = {"expected": "new 64-hex repair2 approval", "observed": approval}
    if expected_approval_sha256 and approval != expected_approval_sha256:
        mismatches["operator_authorization_binding"] = {"expected": expected_approval_sha256, "observed": approval}
    if not EVIDENCE.is_file() or sha256(EVIDENCE) != EVIDENCE_SHA256:
        mismatches["rng_replay_evidence_file"] = {"expected": EVIDENCE_SHA256, "observed": file_hash(EVIDENCE)}
    if not TARGET.is_file() or file_hash(TARGET) != REPLACEMENT_SHA256:
        mismatches["installed_target"] = {"expected": REPLACEMENT_SHA256, "observed": file_hash(TARGET)}
    if not QUARANTINED_TARGET.is_file() or file_hash(QUARANTINED_TARGET) != ORIGINAL_SHA256:
        mismatches["quarantined_target"] = {"expected": ORIGINAL_SHA256, "observed": file_hash(QUARANTINED_TARGET)}
    if not QUARANTINED_METRICS.is_file() or file_hash(QUARANTINED_METRICS) != METRICS_SHA256:
        mismatches["quarantined_metrics"] = {"expected": METRICS_SHA256, "observed": file_hash(QUARANTINED_METRICS)}
    transaction_sha256 = report.get("transaction_sha256")
    if not TRANSACTION.is_file() or file_hash(TRANSACTION) != transaction_sha256:
        mismatches["transaction"] = {"expected": transaction_sha256, "observed": file_hash(TRANSACTION)}
    if mismatches:
        raise RuntimeError(f"repair2 completion evidence invalid: {mismatches}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "apply", "rollback", "reconcile", "validate-report"))
    parser.add_argument("--expected-approval-sha256")
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps(audit(), indent=2, sort_keys=True))
    elif args.action == "apply":
        apply_repair()
        print(REPORT.read_text())
    elif args.action == "rollback":
        rollback()
    elif args.action == "reconcile":
        print(reconcile())
    else:
        print(json.dumps(validate_report(args.expected_approval_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
