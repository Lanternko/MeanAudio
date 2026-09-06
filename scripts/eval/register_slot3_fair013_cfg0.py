#!/usr/bin/env python3
"""Zero-GPU registration coordinator for the active fair013 slot3 chain.

This module intentionally contains no evaluation imports or launch path.  The
production CLI validates the exact live caller and registers one of three
legacy call tuples as a protected CFG0 contract.  Test code imports the pure
validation and store helpers with isolated paths.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path("/home/kojiek/MeanAudio")
TEMPLATE = ROOT / "docs/experiments/slot3_fair013_cfg0_eval_template.json"
ORDER = ROOT / "docs/experiments/slot3_cfg0_priority_order.pending.json"
MANIFEST = ROOT / "docs/experiments/slot3_cfg0_registration_runtime_manifest.json"
APPROVAL = ROOT / "docs/experiments/slot3_cfg0_interceptor_activation_approval.json"
PLAN = ROOT / "docs/experiments/slot3_cfg0_priority_interception_registration_v1.md"
SECURITY_REVIEW = ROOT / "docs/reviews/slot3_cfg0_priority_interception_registration_v1_security_review.json"
COORDINATOR = ROOT / "scripts/eval/register_slot3_fair013_cfg0.py"
VALIDATOR = ROOT / "scripts/eval/validate_slot3_fair013_cfg0_registration.py"
INSTALLER = ROOT / "scripts/eval/install_slot3_cfg0_interceptor.py"
HOOK = ROOT / "scripts/eval/slot3_cfg0_interceptor_hook.sh"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
PYTHON = Path("/usr/bin/python3.12")
BASH = Path("/usr/bin/bash")
ENV = Path("/usr/bin/env")
LIVE_EVALUATOR = Path("/home/kojiek/research/meanaudio_training/caption10s_pipeline/eval_musiccaps_mf25.sh")
TOP_QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
RUNTIME = Path("/home/kojiek/slot3_cfg0_interceptor")
RUNTIME_TRUST_ANCHOR = Path("/home/kojiek")
KEY_FILE = RUNTIME / "registration_hmac.key"
STATE_FILE = RUNTIME / "state.json"
LOCK_FILE = RUNTIME / "coordinator.lock"
STATE_HISTORY = RUNTIME / "state_history"
RESOLVED = RUNTIME / "resolved_contracts"
OUTBOX = RUNTIME / "notification_outbox"
ROLLBACK_RECORD = RUNTIME / "rollback.json"
ZERO_MAC = "0" * 64
STATE_DOMAIN = b"meanaudio-slot3-cfg0-registration-state-v1\0"
CONTRACT_DOMAIN = b"meanaudio-slot3-cfg0-resolved-contract-v1\0"
NOTIFICATION_DOMAIN = b"meanaudio-slot3-cfg0-notification-v1\0"
ROLLBACK_DOMAIN = b"meanaudio-slot3-cfg0-rollback-v1\0"
FIXED_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}


class RegistrationError(RuntimeError):
    """A fail-closed registration refusal."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RegistrationError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(fd, min(1024 * 1024, info.st_size - offset), offset)
            if not chunk:
                raise RegistrationError(f"short read: {path}")
            digest.update(chunk)
            offset += len(chunk)
        if os.fstat(fd) != info:
            raise RegistrationError(f"file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def read_regular_bytes(path: Path, *, expected_uid: int | None = None, expected_mode: int | None = None) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegistrationError(f"unsafe regular file: {path}")
        if expected_uid is not None and before.st_uid != expected_uid:
            raise RegistrationError(f"wrong file owner: {path}")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise RegistrationError(f"wrong file mode: {path}")
        if before.st_size > 16 * 1024 * 1024:
            raise RegistrationError(f"file exceeds bounded read limit: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RegistrationError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RegistrationError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def secure_file_identity(path: Path, *, expected_uid: int | None = None) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegistrationError(f"unsafe regular-file identity: {path}")
        if expected_uid is not None and before.st_uid != expected_uid:
            raise RegistrationError(f"wrong file owner: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
            if not chunk:
                raise RegistrationError(f"short read: {path}")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RegistrationError(f"file changed while hashing: {path}")
        return {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "device": before.st_dev,
            "inode": before.st_ino,
            "owner_uid": before.st_uid,
            "owner_gid": before.st_gid,
            "mode": f"{stat.S_IMODE(before.st_mode):04o}",
            "link_count": before.st_nlink,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
    finally:
        os.close(fd)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistrationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistrationError(f"JSON object required: {path}")
    return value


def _check_owned_mode(path: Path, expected_mode: int, *, directory: bool) -> os.stat_result:
    info = path.lstat()
    kind_ok = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if path.is_symlink() or not kind_ok or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != expected_mode:
        kind = "directory" if directory else "file"
        raise RegistrationError(f"unsafe {kind} owner/mode: {path}")
    return info


def validate_runtime_ancestry(root: Path, trust_anchor: Path) -> None:
    root = root.absolute()
    trust_anchor = trust_anchor.absolute()
    try:
        relative_parent = root.parent.relative_to(trust_anchor)
    except ValueError as exc:
        raise RegistrationError(f"runtime root escapes trust anchor: {root}") from exc
    current = trust_anchor
    components = [trust_anchor]
    for part in relative_parent.parts:
        current = current / part
        components.append(current)
    for component in components:
        resolved = component.resolve(strict=True)
        info = component.lstat()
        if (resolved != component or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid() or info.st_mode & 0o022):
            raise RegistrationError(f"unsafe writable or symlinked runtime ancestor: {component}")


def prepare_runtime(root: Path, *, trust_anchor: Path = RUNTIME_TRUST_ANCHOR) -> None:
    validate_runtime_ancestry(root, trust_anchor)
    if root.exists() or root.is_symlink():
        _check_owned_mode(root, 0o700, directory=True)
    else:
        root.mkdir(mode=0o700, parents=False)
        _check_owned_mode(root, 0o700, directory=True)
    for child in (root / "state_history", root / "resolved_contracts", root / "notification_outbox"):
        if child.exists() or child.is_symlink():
            _check_owned_mode(child, 0o700, directory=True)
        else:
            child.mkdir(mode=0o700)
            _check_owned_mode(child, 0o700, directory=True)


def atomic_bytes(path: Path, value: bytes, mode: int = 0o600) -> None:
    _check_owned_mode(path.parent, 0o700, directory=True)
    name = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    temp = path.parent / name
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        view = memoryview(value)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    _check_owned_mode(path, mode, directory=False)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def create_or_load_key(path: Path) -> bytes:
    if not path.exists() and not path.is_symlink():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            key = os.urandom(32)
            if os.write(fd, key) != len(key):
                raise RegistrationError("short HMAC key write")
            os.fsync(fd)
        finally:
            os.close(fd)
    _check_owned_mode(path, 0o600, directory=False)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        key = os.read(fd, 33)
    finally:
        os.close(fd)
    if len(key) != 32:
        raise RegistrationError("HMAC key must contain exactly 32 bytes")
    return key


def signed(value: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    if "integrity" in value:
        raise RegistrationError("cannot sign a value that already has integrity")
    return {**value, "integrity": hmac.new(key, domain + canonical(value), hashlib.sha256).hexdigest()}


def verify_signed(value: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    unsigned = dict(value)
    supplied = unsigned.pop("integrity", None)
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RegistrationError("HMAC validation failed")
    return unsigned


def runtime_bytes(root: Path) -> int:
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RegistrationError(f"symlink in runtime root: {path}")
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise RegistrationError(f"unsafe runtime object: {path}")
            total += info.st_size
    return total


def read_proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = raw.rfind(")")
    if end < 0:
        raise RegistrationError("invalid proc stat")
    fields = raw[end + 2 :].split()
    return int(fields[19])


def proc_parent_pid(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = raw.rfind(")")
    fields = raw[end + 2 :].split()
    return int(fields[1])


def read_proc_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8") for part in raw.rstrip(b"\0").split(b"\0")]


def hash_proc_fd(pid: int, fd_number: int) -> tuple[dict[str, Any], str]:
    path = Path(f"/proc/{pid}/fd/{fd_number}")
    # /proc/<pid>/fd/N is necessarily a kernel magic link.  Do not request
    # O_NOFOLLOW here: validate the opened descriptor's regular-file type and
    # exact device/inode/owner/hash against the preregistered caller instead.
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RegistrationError("caller script fd is not regular")
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(fd, min(1024 * 1024, info.st_size - offset), offset)
            if not chunk:
                raise RegistrationError("short caller script read")
            digest.update(chunk)
            offset += len(chunk)
        return ({"device": info.st_dev, "inode": info.st_ino, "owner_uid": info.st_uid}, digest.hexdigest())
    finally:
        os.close(fd)


def observe_caller(caller_pid: int, hook_pid: int) -> dict[str, Any]:
    if caller_pid <= 1 or hook_pid <= 1:
        raise RegistrationError("invalid caller or hook PID")
    if os.getppid() != hook_pid or proc_parent_pid(hook_pid) != caller_pid:
        raise RegistrationError("hook/caller process ancestry mismatch")
    script, script_hash = hash_proc_fd(caller_pid, 255)
    hook_script, hook_script_hash = hash_proc_fd(hook_pid, 255)
    return {
        "pid": caller_pid,
        "uid": Path(f"/proc/{caller_pid}").stat().st_uid,
        "start_ticks": read_proc_start_ticks(caller_pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        "executable": os.readlink(f"/proc/{caller_pid}/exe"),
        "argv": read_proc_cmdline(caller_pid),
        "script_fd": 255,
        "script_device": script["device"],
        "script_inode": script["inode"],
        "script_owner_uid": script["owner_uid"],
        "script_sha256": script_hash,
        "hook_pid": hook_pid,
        "hook_uid": Path(f"/proc/{hook_pid}").stat().st_uid,
        "hook_parent_pid": proc_parent_pid(hook_pid),
        "hook_executable": os.readlink(f"/proc/{hook_pid}/exe"),
        "hook_script_fd": 255,
        "hook_script_sha256": hook_script_hash,
        "hook_script_inode": hook_script["inode"],
    }


def validate_caller(observed: dict[str, Any], expected: dict[str, Any], replacement_sha256: str) -> None:
    for field in ("pid", "uid", "start_ticks", "boot_id", "executable", "argv", "script_fd", "script_device", "script_inode", "script_sha256"):
        if observed.get(field) != expected.get(field):
            raise RegistrationError(f"caller fingerprint mismatch: {field}")
    if observed.get("script_owner_uid") != expected["uid"]:
        raise RegistrationError("caller script owner mismatch")
    if observed.get("hook_uid") != expected["uid"] or observed.get("hook_parent_pid") != expected["pid"]:
        raise RegistrationError("hook parent identity mismatch")
    if observed.get("hook_executable") != "/usr/bin/bash":
        raise RegistrationError("hook executable mismatch")
    if observed.get("hook_script_sha256") != replacement_sha256:
        raise RegistrationError("hook script descriptor hash mismatch")


def arm_for_argv(template: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    for arm in template["arms"]:
        expected = [arm["legacy_label"], arm["checkpoint"], *arm["legacy_conditioning_argv"]]
        if argv == expected:
            return arm
    raise RegistrationError("argv does not match an exact registered legacy migration tuple")


def validate_template(template: dict[str, Any]) -> None:
    protocol = template.get("scientific_protocol", {})
    required = {
        "benchmark": "MusicCaps", "expected_rows": 5521, "expected_unique_ids": 5521,
        "solver": "MeanFlow", "num_steps": 25, "cfg_strength": 0,
        "generation_seed": 42, "no_text_attention_mask": True, "full_precision": True,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise RegistrationError(f"template protocol mismatch: {key}")
    arms = template.get("arms")
    if not isinstance(arms, list) or [arm.get("sequence") for arm in arms] != [1, 2, 3]:
        raise RegistrationError("template requires exactly three ordered arms")
    if [arm.get("arm_id") for arm in arms] != ["fair013_k3_q9", "fair013_best_noq", "fair013_worst_noq"]:
        raise RegistrationError("unexpected arm ordering")
    labels = [arm.get("canonical_label") for arm in arms]
    if len(labels) != len(set(labels)):
        raise RegistrationError("canonical labels collide")
    for arm in arms:
        if "_cfg0_" not in arm["canonical_label"] or "cfg4p5" in arm["canonical_label"]:
            raise RegistrationError("canonical label is not literal CFG0")
        if "_cfg4p5_" not in arm["legacy_label"]:
            raise RegistrationError("legacy alias does not encode CFG4.5")


def validate_security_receipt(receipt: dict[str, Any], current_plan_sha256: str) -> None:
    expected = {
        "document_kind": "pilotfish_security_readiness_review",
        "readiness_unit_id": "slot3-cfg0-priority-interception-registration-v1",
        "verdict": "READY",
        "reviewed_plan_sha256": current_plan_sha256,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise RegistrationError(f"security receipt semantic mismatch: {field}")


def initial_state(key: bytes, template_sha256: str, manifest_sha256: str, approval_sha256: str) -> dict[str, Any]:
    return signed({
        "schema_version": 1,
        "document_kind": "slot3_cfg0_registration_state",
        "sequence": 0,
        "prior_state_mac": ZERO_MAC,
        "next_arm_index": 0,
        "accepted_arms": [],
        "template_sha256": template_sha256,
        "runtime_manifest_sha256": manifest_sha256,
        "activation_approval_sha256": approval_sha256,
        "status": "awaiting_fair013_k3_q9",
    }, key, STATE_DOMAIN)


def load_state(root: Path, key: bytes, template_sha256: str, manifest_sha256: str, approval_sha256: str) -> dict[str, Any]:
    path = root / "state.json"
    history = root / "state_history"
    if not path.exists() and not path.is_symlink():
        state = initial_state(key, template_sha256, manifest_sha256, approval_sha256)
        atomic_json(history / "0000.json", state)
        atomic_json(path, state)
        return state
    _check_owned_mode(path, 0o600, directory=False)
    state = load_json(path)
    unsigned = verify_signed(state, key, STATE_DOMAIN)
    sequence = unsigned.get("sequence")
    if not isinstance(sequence, int) or sequence < 0 or sequence > 3:
        raise RegistrationError("invalid state sequence")
    prior = ZERO_MAC
    for index in range(sequence + 1):
        history_path = history / f"{index:04d}.json"
        _check_owned_mode(history_path, 0o600, directory=False)
        historic = load_json(history_path)
        historic_unsigned = verify_signed(historic, key, STATE_DOMAIN)
        if historic_unsigned.get("sequence") != index or historic_unsigned.get("prior_state_mac") != prior:
            raise RegistrationError("state HMAC chain is invalid")
        prior = historic["integrity"]
    if historic != state:
        raise RegistrationError("current state does not equal history tail")
    if (unsigned.get("template_sha256") != template_sha256
            or unsigned.get("runtime_manifest_sha256") != manifest_sha256
            or unsigned.get("activation_approval_sha256") != approval_sha256):
        raise RegistrationError("state binding drift")
    if unsigned.get("next_arm_index") != sequence or len(unsigned.get("accepted_arms", [])) != sequence:
        raise RegistrationError("state sequence/order mismatch")
    return state


def advance_state(root: Path, state: dict[str, Any], arm_id: str, contract_sha256: str, key: bytes) -> dict[str, Any]:
    unsigned = verify_signed(state, key, STATE_DOMAIN)
    sequence = unsigned["sequence"] + 1
    next_state = signed({
        **{key_name: value for key_name, value in unsigned.items() if key_name not in {"sequence", "prior_state_mac", "next_arm_index", "accepted_arms", "status"}},
        "sequence": sequence,
        "prior_state_mac": state["integrity"],
        "next_arm_index": sequence,
        "accepted_arms": [*unsigned["accepted_arms"], {"arm_id": arm_id, "resolved_contract_sha256": contract_sha256}],
        "status": "registration_complete" if sequence == 3 else f"awaiting_sequence_{sequence + 1}",
    }, key, STATE_DOMAIN)
    history_path = root / "state_history" / f"{sequence:04d}.json"
    if history_path.exists() or history_path.is_symlink():
        _check_owned_mode(history_path, 0o600, directory=False)
        observed = load_json(history_path)
        verify_signed(observed, key, STATE_DOMAIN)
        if observed != next_state:
            raise RegistrationError("next state history collision")
    else:
        atomic_json(history_path, next_state)
    atomic_json(root / "state.json", next_state)
    return next_state


def build_resolved_contract(
    template: dict[str, Any], arm: dict[str, Any], checkpoint_identity: dict[str, Any],
    caller: dict[str, Any], template_sha256: str, manifest_sha256: str, approval_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_kind": "slot3_fair013_cfg0_resolved_contract",
        "readiness_unit_id": template["readiness_unit_id"],
        "sequence": arm["sequence"],
        "arm_id": arm["arm_id"],
        "status": "registered_held_for_later_gpu_runtime_approval",
        "legacy_migration_alias": {
            "label": arm["legacy_label"],
            "conditioning_argv": arm["legacy_conditioning_argv"],
            "historical_implementation_invoked": False,
        },
        "canonical_label": arm["canonical_label"],
        "conditioning": arm["canonical_conditioning"],
        "checkpoint_identity": checkpoint_identity,
        "protocol": template["scientific_protocol"],
        "caller_fingerprint": caller,
        "template_sha256": template_sha256,
        "runtime_manifest_sha256": manifest_sha256,
        "activation_approval_sha256": approval_sha256,
        "gpu_launch_authorized": False,
        "authenticated_top_queue_mutation_authorized": False,
    }


def stable_caller_fingerprint(caller: dict[str, Any]) -> dict[str, Any]:
    """Drop only the ephemeral hook PID while retaining its trusted identity."""
    return {key: value for key, value in caller.items() if key != "hook_pid"}


def ensure_exact_or_create(path: Path, payload: dict[str, Any], key: bytes, domain: bytes) -> tuple[dict[str, Any], str]:
    expected = signed(payload, key, domain)
    if path.exists() or path.is_symlink():
        _check_owned_mode(path, 0o600, directory=False)
        observed = load_json(path)
        verify_signed(observed, key, domain)
        if observed != expected:
            raise RegistrationError(f"duplicate drift: {path}")
    else:
        atomic_json(path, expected)
    return expected, sha256_bytes(canonical(expected))


def notification_payload(arm: dict[str, Any], contract_sha256: str, template_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_kind": "slot3_cfg0_registration_notification",
        "sequence": arm["sequence"],
        "arm_id": arm["arm_id"],
        "status": "held",
        "event_id": f"slot3-cfg0-registration-{arm['sequence']}-{contract_sha256[:16]}",
        "resolved_contract_sha256": contract_sha256,
        "template_sha256": template_sha256,
        "runtime_manifest_sha256": manifest_sha256,
        "summary": f"Registered {arm['arm_id']} as MusicCaps 5521 / MeanFlow 25 / CFG 0; GPU launch remains held.",
    }


def default_notify(payload: dict[str, Any]) -> None:
    completed = subprocess.run(
        [str(PYTHON), str(NOTIFIER), "--status", "held", "--experiment", "slot3-cfg0-priority-interception-registration-v1", "--summary", payload["summary"], "--repo", str(ROOT)],
        env=FIXED_ENV,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RegistrationError(f"notifier failure rc={completed.returncode}: {completed.stderr[-300:]}")


@dataclass(frozen=True)
class StorePaths:
    root: Path
    key: Path
    state: Path
    lock: Path
    history: Path
    resolved: Path
    outbox: Path
    trust_anchor: Path

    @classmethod
    def under(cls, root: Path, *, trust_anchor: Path | None = None) -> "StorePaths":
        if trust_anchor is None:
            trust_anchor = RUNTIME_TRUST_ANCHOR if root.absolute() == RUNTIME else root.parent
        return cls(
            root, root / "registration_hmac.key", root / "state.json", root / "coordinator.lock",
            root / "state_history", root / "resolved_contracts", root / "notification_outbox",
            trust_anchor,
        )


def register_once(
    *, template: dict[str, Any], argv: list[str], caller: dict[str, Any], paths: StorePaths,
    manifest_sha256: str, approval_sha256: str, notify: Callable[[dict[str, Any]], None],
    checkpoint_identity: dict[str, Any] | None = None,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    validate_template(template)
    arm = arm_for_argv(template, argv)
    template_sha256 = sha256_bytes(canonical(template))
    prepare_runtime(paths.root, trust_anchor=paths.trust_anchor)
    if runtime_bytes(paths.root) > template["interceptor"]["maximum_runtime_bytes"]:
        raise RegistrationError("registration runtime byte budget exceeded")
    key = create_or_load_key(paths.key)
    inject = fault if fault is not None else lambda _point: None
    lock_fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        _check_owned_mode(paths.lock, 0o600, directory=False)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        inject("before_state")
        state = load_state(paths.root, key, template_sha256, manifest_sha256, approval_sha256)
        unsigned_state = verify_signed(state, key, STATE_DOMAIN)
        index = arm["sequence"] - 1
        if index > unsigned_state["next_arm_index"]:
            raise RegistrationError("arm order violation")
        if checkpoint_identity is None:
            checkpoint_identity = secure_file_identity(Path(arm["checkpoint"]), expected_uid=os.geteuid())
        if checkpoint_identity.get("path") != arm["checkpoint"]:
            raise RegistrationError("checkpoint path mismatch")
        resolved_payload = build_resolved_contract(
            template, arm, checkpoint_identity, stable_caller_fingerprint(caller),
            template_sha256, manifest_sha256, approval_sha256,
        )
        contract_path = paths.root / arm["resolved_contract"]
        contract, contract_sha256 = ensure_exact_or_create(contract_path, resolved_payload, key, CONTRACT_DOMAIN)
        event_payload = notification_payload(arm, contract_sha256, template_sha256, manifest_sha256)
        event_path = paths.outbox / f"{arm['sequence']:02d}_{arm['arm_id']}.json"
        delivered_payload = {**event_payload, "delivery_status": "delivered"}
        delivered = signed(delivered_payload, key, NOTIFICATION_DOMAIN)
        if index < unsigned_state["next_arm_index"]:
            existing = load_json(event_path)
            verify_signed(existing, key, NOTIFICATION_DOMAIN)
            if existing != delivered:
                raise RegistrationError("idempotent replay notification drift")
            accepted = unsigned_state["accepted_arms"][index]
            if accepted != {"arm_id": arm["arm_id"], "resolved_contract_sha256": contract_sha256}:
                raise RegistrationError("idempotent replay state drift")
            return {"status": "idempotent", "arm_id": arm["arm_id"], "contract_sha256": contract_sha256}
        attempting = signed({**event_payload, "delivery_status": "attempting"}, key, NOTIFICATION_DOMAIN)
        if event_path.exists() or event_path.is_symlink():
            _check_owned_mode(event_path, 0o600, directory=False)
            existing = load_json(event_path)
            unsigned_existing = verify_signed(existing, key, NOTIFICATION_DOMAIN)
            attempting_payload = {**event_payload, "delivery_status": "attempting"}
            ambiguous_payload = {
                **event_payload,
                "delivery_status": "delivery_ambiguous",
                "automatic_resend_allowed": False,
                "reconciliation_status": "held_pending_separate_approval_bound_reconciliation",
            }
            if unsigned_existing == attempting_payload:
                atomic_json(event_path, signed(ambiguous_payload, key, NOTIFICATION_DOMAIN))
                raise RegistrationError("delivery_ambiguous: recovered attempting event; automatic resend forbidden")
            if unsigned_existing == ambiguous_payload:
                raise RegistrationError("delivery_ambiguous: separate approval-bound reconciliation required")
            if unsigned_existing != delivered_payload:
                raise RegistrationError("notification replay drift")
            advance_state(paths.root, state, arm["arm_id"], contract_sha256, key)
            return {"status": "accepted", "arm_id": arm["arm_id"], "contract_sha256": contract_sha256}
        else:
            atomic_json(event_path, attempting)
        inject("after_attempting_before_send")
        notify(event_payload)
        inject("after_notifier_return_before_delivered")
        atomic_json(event_path, delivered)
        inject("after_delivered_before_state")
        advance_state(paths.root, state, arm["arm_id"], contract_sha256, key)
        if runtime_bytes(paths.root) > template["interceptor"]["maximum_runtime_bytes"]:
            raise RegistrationError("registration runtime byte budget exceeded after write")
        return {"status": "accepted", "arm_id": arm["arm_id"], "contract_sha256": contract_sha256}
    finally:
        os.close(lock_fd)


def outer_loop(attempt: Callable[[], int], *, pause: Callable[[], None], max_attempts: int | None = None) -> int:
    attempts = 0
    while True:
        attempts += 1
        try:
            rc = attempt()
        except BaseException:
            rc = 125
        if rc == 0:
            return 0
        if max_attempts is not None and attempts >= max_attempts:
            return rc
        pause()


def runtime_manifest_payload() -> dict[str, Any]:
    sources = [HOOK, COORDINATOR, VALIDATOR, INSTALLER, NOTIFIER, PYTHON, BASH, ENV, TEMPLATE, ORDER, PLAN, SECURITY_REVIEW]
    return {
        "schema_version": 1,
        "document_kind": "slot3_cfg0_registration_runtime_manifest",
        "readiness_unit_id": "slot3-cfg0-priority-interception-registration-v1",
        "scope": "registration_only_zero_gpu_zero_live_queue_mutation",
        "sources": [{"path": str(path), "sha256": sha256_path(path)} for path in sources],
        "fixed_child_environment": FIXED_ENV,
        "hmac_domains_hex": {
            "state": STATE_DOMAIN.hex(), "resolved_contract": CONTRACT_DOMAIN.hex(),
            "notification": NOTIFICATION_DOMAIN.hex(), "rollback": ROLLBACK_DOMAIN.hex(),
        },
        "allowed_writes": [str(RUNTIME), str(LIVE_EVALUATOR)],
        "live_path_write_constraint": "installer_only_exact_live_evaluator_basename_atomic_replacement",
        "forbidden_execution_paths": [str(ROOT / "eval.py"), "/home/kojiek/research/meanaudio_eval/phase4_eval.py", str(LIVE_EVALUATOR)],
        "runtime_root": str(RUNTIME),
        "runtime_maximum_bytes": 16777216,
        "notifier": {"path": str(NOTIFIER), "secret_source": "local_mode_0600_default_file", "secret_in_argv_or_artifacts": False},
    }


def verify_production_bindings(template: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    plan_sha256 = sha256_path(PLAN)
    security_receipt = load_json(SECURITY_REVIEW)
    validate_security_receipt(security_receipt, plan_sha256)
    manifest = load_json(MANIFEST)
    generated = runtime_manifest_payload()
    if manifest != generated:
        raise RegistrationError("registration runtime manifest drift")
    manifest_sha256 = sha256_path(MANIFEST)
    approval = load_json(APPROVAL)
    if approval.get("approval_status") != "approved":
        raise RegistrationError("interceptor activation approval is not approved")
    unsigned_approval = dict(approval)
    supplied_approval = unsigned_approval.pop("operator_approval_sha256", None)
    expected_approval = sha256_bytes(b"meanaudio-slot3-cfg0-activation-approval-v1\0" + canonical(unsigned_approval))
    if not isinstance(supplied_approval, str) or not hmac.compare_digest(supplied_approval, expected_approval):
        raise RegistrationError("activation approval digest is invalid")
    expected_fields = {
        "approved_plan_sha256": plan_sha256,
        "security_review_sha256": sha256_path(SECURITY_REVIEW),
        "runtime_manifest_sha256": manifest_sha256,
        "replacement_hook_sha256": sha256_path(HOOK),
        "historical_preimage_sha256": template["interceptor"]["historical_preimage_sha256"],
        "template_file_sha256": sha256_path(TEMPLATE),
        "priority_order_file_sha256": sha256_path(ORDER),
        "operator_instruction_sha256": sha256_bytes("插在 slot3 之後。然後確保 slot3 相關的都用 cfg0".encode("utf-8")),
    }
    for field, value in expected_fields.items():
        if approval.get(field) != value:
            raise RegistrationError(f"activation approval binding drift: {field}")
    if approval.get("caller") != template["caller"] or approval.get("mappings") != template["arms"]:
        raise RegistrationError("activation approval caller/mapping drift")
    if approval.get("allowed_writes") != [str(RUNTIME), str(LIVE_EVALUATOR)]:
        raise RegistrationError("activation approval write scope drift")
    if approval.get("notification_behavior") != {
        "event": "held", "one_delivered_idempotent_event_required_per_exact_registration": True,
        "notifier": str(NOTIFIER), "webhook_secret_in_argv_or_artifacts": False,
        "failure_returns_to_outer_hook_loop": True,
    }:
        raise RegistrationError("activation approval notification behavior drift")
    if approval.get("boundary") != {"gpu_minutes": 0, "live_queue_mutation": False, "historical_evaluator_execution": False}:
        raise RegistrationError("activation approval boundary drift")
    if sha256_path(LIVE_EVALUATOR) != approval.get("replacement_hook_sha256"):
        raise RegistrationError("live evaluator is not the approved interceptor")
    return manifest, manifest_sha256, sha256_path(APPROVAL)


def record_rollback(hook_pid: int, caller_pid: int) -> None:
    template = load_json(TEMPLATE)
    approval = load_json(APPROVAL)
    if approval.get("approval_status") != "approved":
        raise RegistrationError("rollback record refused without approved activation")
    observed = observe_caller(caller_pid, hook_pid)
    validate_caller(observed, template["caller"], approval["rollback_failclosed_sha256"])
    if sha256_path(LIVE_EVALUATOR) != approval["rollback_failclosed_sha256"]:
        raise RegistrationError("live rollback stub hash mismatch")
    prepare_runtime(RUNTIME)
    key = create_or_load_key(KEY_FILE)
    value = signed({
        "schema_version": 1,
        "document_kind": "slot3_cfg0_interceptor_rollback_record",
        "status": "CFG0_INTERCEPTOR_ROLLED_BACK",
        "hook_pid": hook_pid,
        "caller_pid": caller_pid,
        "template_sha256": sha256_path(TEMPLATE),
        "recorded_epoch": int(time.time()),
        "gpu_launch_attempted": False,
    }, key, ROLLBACK_DOMAIN)
    atomic_json(ROLLBACK_RECORD, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-runtime-manifest", action="store_true")
    parser.add_argument("--record-rollback", action="store_true")
    parser.add_argument("--hook-pid", type=int)
    parser.add_argument("--caller-pid", type=int)
    parser.add_argument("legacy_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.print_runtime_manifest:
        print(json.dumps(runtime_manifest_payload(), sort_keys=True, indent=2))
        return 0
    if args.hook_pid is None or args.caller_pid is None:
        raise RegistrationError("hook and caller PIDs are required")
    if args.record_rollback:
        record_rollback(args.hook_pid, args.caller_pid)
        return 0
    template = load_json(TEMPLATE)
    validate_template(template)
    _, manifest_sha256, approval_sha256 = verify_production_bindings(template)
    approval = load_json(APPROVAL)
    observed = observe_caller(args.caller_pid, args.hook_pid)
    validate_caller(observed, template["caller"], approval["replacement_hook_sha256"])
    argv = args.legacy_argv[1:] if args.legacy_argv[:1] == ["--"] else args.legacy_argv
    result = register_once(
        template=template, argv=argv, caller=observed, paths=StorePaths.under(RUNTIME),
        manifest_sha256=manifest_sha256, approval_sha256=approval_sha256, notify=default_notify,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RegistrationError, subprocess.SubprocessError) as exc:
        print(f"SLOT3_CFG0_REGISTRATION_HOLD {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(125)
