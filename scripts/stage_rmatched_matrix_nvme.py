#!/usr/bin/env python3
"""Capability-gated, crash-resumable HDD-to-NVMe R-Matched staging."""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
EXPERIMENT_ID = "rmatched-s1-s2-steps-cfg-matrix-nvme-stage"
RUN_ID = "run-20260815-rmatched-matrix-nvme-stage1"
SOURCE_MANIFEST = ROOT / "docs/experiments/rmatched_matrix_nvme_stage_source_manifest.json"
LEGACY_EVAL_OUTPUT = ROOT / "eval_output"
NVME_ROOT = Path("/home/kojiek/nvme_experiment_artifacts")
NVME_PARENT = NVME_ROOT / "meanaudio"
FINAL = NVME_PARENT / "eval_output"
STAGING = NVME_PARENT / f".eval_output.stage-{RUN_ID}"
STATE = Path("/home/kojiek/logs/rmatched_matrix_nvme_stage_harn")
JOURNAL = STATE / "transaction.json"
REPORT = STATE / "stage_report.json"
HARN_KEY = STATE / "ledger_hmac.key"
APPROVAL_STATE = STATE / "operator_approval.json"
HARN_LOCK = STATE / "controller.lock"
CAPABILITY_ENV = "MEANAUDIO_NVME_STAGE_CAPABILITY"
CAPABILITY_DOMAIN = b"meanaudio-nvme-stage-capability-v1\0"
ALLOW_TEST_FAULTS = False
FAULT_AFTER: str | None = None


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"nonregular or hardlinked file rejected: {path}")
        digest = hashlib.sha256()
        while block := os.read(fd, 8 << 20):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(fd)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temp, path)
    fsync_directory(path.parent)


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"source/destination root missing, non-directory, or symlinked: {root}")
    digest = hashlib.sha256()
    files: list[tuple[Path, str]] = []
    total = 0
    for directory, directories, names in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        names.sort()
        base = Path(directory)
        for name in directories:
            child = base / name
            if child.is_symlink():
                raise RuntimeError(f"recursive symlink rejected: {child}")
            info = child.stat(follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"non-directory tree entry rejected: {child}")
        for name in names:
            path = base / name
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError(f"nonregular or hardlinked tree entry rejected: {path}")
            file_hash = sha256_file(path)
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode() + b"\0" + str(info.st_size).encode() + b"\0" + file_hash.encode() + b"\n")
            files.append((path, relative))
            total += info.st_size
    return {"file_count": len(files), "total_bytes": total, "tree_sha256": digest.hexdigest(), "files": files}


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("document_kind") != "rmatched_nvme_stage_source_manifest" or manifest.get("schema_version") != 1:
        raise RuntimeError("invalid staging source manifest")
    return manifest


def verify_sources() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_manifest()
    observed: dict[str, dict[str, Any]] = {}
    count = total = 0
    for descriptor in manifest["roots"]:
        snapshot = tree_snapshot(Path(descriptor["source"]))
        for key in ("file_count", "total_bytes", "tree_sha256"):
            if snapshot[key] != descriptor[key]:
                raise RuntimeError(f"HDD source drift: {descriptor['role']} {key}")
        observed[descriptor["role"]] = snapshot
        count += snapshot["file_count"]
        total += snapshot["total_bytes"]
    if {"file_count": count, "total_bytes": total} != manifest["totals"]:
        raise RuntimeError("HDD source totals drift")
    return manifest, observed


def verify_destination(root: Path, manifest: dict[str, Any]) -> None:
    expected_top = {Path(item["destination_relative"]).parts[0] for item in manifest["roots"]}
    actual_top = {path.name for path in root.iterdir()}
    if actual_top != expected_top:
        raise RuntimeError("destination contains missing or unexpected top-level entries")
    for descriptor in manifest["roots"]:
        snapshot = tree_snapshot(root / descriptor["destination_relative"])
        for key in ("file_count", "total_bytes", "tree_sha256"):
            if snapshot[key] != descriptor[key]:
                raise RuntimeError(f"destination verification failed: {descriptor['role']} {key}")


def load_key() -> bytes:
    key = HARN_KEY.read_bytes()
    info = HARN_KEY.stat(follow_symlinks=False)
    if len(key) < 32 or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("unsafe staging authentication key")
    return key


def verify_signed(payload: dict[str, Any], domain: bytes) -> None:
    supplied = payload.get("integrity")
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    expected = hmac.new(load_key(), domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("staging approval-state signature invalid")


def process_start(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().split()[21]


def authorize_write(action: str) -> dict[str, Any]:
    raw = os.environ.get(CAPABILITY_ENV, "")
    if not raw:
        raise RuntimeError("NVMe staging write capability is missing")
    capability_path = Path(raw)
    payload = json.loads(capability_path.read_text())
    supplied = payload.pop("integrity", None)
    expected_sig = hmac.new(load_key(), CAPABILITY_DOMAIN + canonical(payload), hashlib.sha256).hexdigest()
    approval = json.loads(APPROVAL_STATE.read_text())
    verify_signed(approval, b"meanaudio-nvme-stage-approval-v1\0")
    reservation = approval.get("reservation", {})
    if (approval.get("state") != "reserved" or approval.get("experiment_id") != EXPERIMENT_ID
            or approval.get("run_id") != RUN_ID or reservation.get("controller_pid") != os.getppid()
            or reservation.get("controller_start_ticks") != process_start(os.getppid())
            or reservation.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()):
        raise RuntimeError("staging approval is not an exact parent reservation")
    expected = {
        "document_kind": "nvme_stage_write_capability", "status": "authorized",
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "action": action,
        "executable_sha256": sha256_file(Path(__file__)),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "approval_state_sha256": sha256_file(APPROVAL_STATE),
        "approval_record_sha256": approval.get("approval_record_sha256"),
        "parent_pid": os.getppid(), "parent_start_ticks": process_start(os.getppid()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "harn_lock": str(HARN_LOCK),
        "writable_paths": [str(NVME_ROOT), str(NVME_PARENT), str(STAGING), str(FINAL), str(JOURNAL), str(REPORT)],
    }
    if not hmac.compare_digest(str(supplied), expected_sig) or any(payload.get(k) != v for k, v in expected.items()):
        raise RuntimeError("NVMe staging capability binding mismatch")
    lock_fd = os.open(HARN_LOCK, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        else:
            raise RuntimeError("NVMe staging HARN lock is not held by parent")
    finally:
        os.close(lock_fd)
    source_resolved = [Path(item["source"]).resolve() for item in load_manifest()["roots"]]
    legacy_resolved = LEGACY_EVAL_OUTPUT.resolve()
    for writable in expected["writable_paths"]:
        candidate = Path(writable)
        resolved_target = candidate.resolve(strict=False)
        if any(resolved_target == source or resolved_target.is_relative_to(source) for source in source_resolved):
            raise RuntimeError("HDD source path leaked into writable capability")
        if resolved_target == legacy_resolved or resolved_target.is_relative_to(legacy_resolved):
            raise RuntimeError("write-capable path resolves through legacy HDD eval_output")
        if not (resolved_target == NVME_ROOT.resolve(strict=False)
                or resolved_target.is_relative_to(NVME_ROOT.resolve(strict=False))
                or resolved_target == STATE.resolve(strict=False)
                or resolved_target.is_relative_to(STATE.resolve(strict=False))):
            raise RuntimeError("writable capability escaped NVMe/log roots")
    claimed = capability_path.with_suffix(capability_path.suffix + ".claimed")
    os.replace(capability_path, claimed)
    fsync_directory(claimed.parent)
    return payload


def record_phase(phase: str, **extra: Any) -> None:
    prior = json.loads(JOURNAL.read_text()) if JOURNAL.is_file() else {}
    atomic_json(JOURNAL, {**prior, "document_kind": "nvme_stage_transaction", "schema_version": 1,
                          "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "phase": phase, **extra})
    if ALLOW_TEST_FAULTS and FAULT_AFTER == phase:
        raise ChildProcessError(f"injected NVMe staging crash after {phase}")


def ensure_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"unsafe destination directory: {path}")
        return
    path.mkdir(mode=0o700)
    fsync_directory(path.parent)


def copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != sha256_file(source) or destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"partial destination file drift: {destination}")
        return
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        info = partial.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"unsafe partial destination: {partial}")
        partial.unlink()
        fsync_directory(partial.parent)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"unsafe source during copy: {source}")
        while block := os.read(source_fd, 8 << 20):
            os.write(destination_fd, block)
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    os.replace(partial, destination)
    fsync_directory(destination.parent)


def write_report(manifest: dict[str, Any]) -> dict[str, Any]:
    verify_destination(FINAL, manifest)
    verify_sources()
    report = {
        "document_kind": "rmatched_nvme_stage_report", "schema_version": 1, "status": "passed",
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "destination": str(FINAL),
        "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_file_count": manifest["totals"]["file_count"], "source_total_bytes": manifest["totals"]["total_bytes"],
        "hdd_sources_unchanged": True,
    }
    atomic_json(REPORT, report)
    record_phase("completed", report_sha256=sha256_file(REPORT))
    return report


def stage(action: str) -> dict[str, Any]:
    authorize_write(action)
    manifest, source_snapshots = verify_sources()
    if FINAL.exists():
        verify_destination(FINAL, manifest)
        return write_report(manifest)
    ensure_directory(NVME_ROOT)
    ensure_directory(NVME_PARENT)
    if STAGING.exists() and (not STAGING.is_dir() or STAGING.is_symlink()):
        raise RuntimeError("unsafe existing staging path")
    if not STAGING.exists():
        ensure_directory(STAGING)
    if STAGING.stat().st_dev != NVME_PARENT.stat().st_dev:
        raise RuntimeError("staging and final parent are not on the same filesystem")
    record_phase("copy_started")
    copied = 0
    for descriptor in manifest["roots"]:
        destination_root = STAGING / descriptor["destination_relative"]
        for source, relative in source_snapshots[descriptor["role"]]["files"]:
            destination = destination_root / relative
            chain: list[Path] = []
            parent = destination.parent
            while parent != STAGING and not parent.exists():
                chain.append(parent)
                parent = parent.parent
            for directory in reversed(chain):
                ensure_directory(directory)
            copy_file(source, destination)
            copied += 1
            if ALLOW_TEST_FAULTS and FAULT_AFTER == "during_copy" and copied == 1:
                raise ChildProcessError("injected NVMe staging crash during copy")
    verify_destination(STAGING, manifest)
    record_phase("staged_verified", copied_files=copied)
    os.replace(STAGING, FINAL)
    fsync_directory(NVME_PARENT)
    record_phase("final_renamed")
    return write_report(manifest)


def validate_report() -> None:
    manifest = load_manifest()
    payload = json.loads(REPORT.read_text())
    if (payload.get("document_kind") != "rmatched_nvme_stage_report" or payload.get("status") != "passed"
            or payload.get("experiment_id") != EXPERIMENT_ID or payload.get("run_id") != RUN_ID
            or payload.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST)
            or payload.get("hdd_sources_unchanged") is not True):
        raise RuntimeError("invalid NVMe stage completion report")
    verify_sources()
    verify_destination(FINAL, manifest)
    print("[VALID] exact NVMe stage report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "stage", "reconcile", "validate-report"))
    args = parser.parse_args()
    if args.action == "audit":
        manifest, _ = verify_sources()
        print(json.dumps({"status": "passed", "totals": manifest["totals"]}, sort_keys=True))
    elif args.action == "validate-report":
        validate_report()
    else:
        stage(args.action)


if __name__ == "__main__":
    main()
