#!/usr/bin/env python3
"""Fail-closed dual-authority recovery gate for the one-shot NVMe chain.

The scientific execution authority and the recovery authority deliberately use
different HMAC domains even though the currently accepted AUTH-ROOT model uses
one local signer key.  Callers must satisfy both roles; neither role may stand
in for the other.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


PLAN_FINGERPRINT = "a765708d51f72627b40308b1871737ea531f5ff01104bf8dbc68e065be06d9b6"
EXECUTION_AUTHORITY_SHA256 = "c6b74e4e13b7af2915cc2496ffda47e3c1c60eb9d05bba4bf401e705cc766f9a"
RECOVERY_AUTHORITY_SHA256 = "983aaa0998ff409fafc7f2f0b293a1b87f8008dd92187054f7f7e2a5140f3893"
RECOVERY_AUTHORITY_BYTES = 1541
PRIOR_QUEUE_SHA256 = "fdd5eee312e0d51c2824e0af745a544df8d443860dadf9df722e3f400d1266d8"
ARCHIVE_MANIFEST_SHA256 = "6b5b881696dc1f77923228be3b4d676ead1ceedbb1c870331e1157705fc66973"
BLOCKER_FINGERPRINT = "b3ca8b3ec2402543"

QUEUE_DOMAIN = b"meanaudio-operator-queue-v1\0"
APPROVAL_DOMAIN = b"meanaudio-queue-approval-v1\0"
RECOVERY_DOMAIN = b"meanaudio-dual-authority-recovery-v3\0"
ARCHIVE_DOMAIN = b"meanaudio-recovery-archive-v1\0"

EXPERIMENTS = (
    ("rmatched-s1-s2-steps-cfg-matrix-nvme-stage", "run-20260815-rmatched-matrix-nvme-stage1"),
    ("rmatched-s1-s2-steps-cfg-matrix-repair2", "run-20260814-seed14159265-musiccaps-repair2"),
    ("rmatched-s1-s2-steps-cfg-matrix-continuation", "run-20260814-seed14159265-musiccaps-continuation1"),
)
QUEUE_LAYOUT = (
    (EXPERIMENTS[0][0], EXPERIMENTS[0][1], 4, (), ()),
    (EXPERIMENTS[1][0], EXPERIMENTS[1][1], 5, (EXPERIMENTS[0][0],), (EXPERIMENTS[0][0],)),
    (EXPERIMENTS[2][0], EXPERIMENTS[2][1], 6, (EXPERIMENTS[1][0],), (EXPERIMENTS[1][0],)),
)
BASELINE_ONLY_BINDINGS = {
    "03b3cc4bc54150e4fb8c698e70f8fbb365e56181cbe33b80150f2e3becebd178",
    "8b7a11d0bd713c45688821b2260c523df0062743888b7c1fbe0a4d738fcac5d7",
    "b3fa27f2e48de8e695580d1ed5a3c75b792fa3112e3777fa2e6551afc0b3c777",
}

# Ordered and closed.  Reordering, omission, duplication, or adding an
# unreviewed revocation is a schema violation.  The superseded continuation
# approval is unconditional; a role cannot reinterpret it as reusable.
ORDERED_CLOSED_REVOCATIONS: tuple[tuple[str, str], ...] = (
    ("prior_signed_queue", PRIOR_QUEUE_SHA256),
    ("old_stage_contract", "751fc6a07939bd78c3071152128acb4ce22f380acce3fd2c6e86125640b5bea8"),
    ("old_stage_harn", "4dd868b748dd9f2b21776a1944d701f213f7610c932d4c0c686665f9c38d36ed"),
    ("old_repair2_contract", "35911bfa1dc6fcef14749838c344563c82f5967a52d906e595d95f87fbc9db1e"),
    ("old_repair2_harn", "d02e2f49083aacd6c1d89867440ffb247bf53ad9199c14a6247e62e8f65b4d01"),
    ("old_repair2_runtime_manifest", "78f3c0d2338ad9579d245f3ff0c93bed3038b1ca14f0158143ec2ebfb0da2fcc"),
    ("old_continuation_contract", "4f964855d67a27b8f5af59d2bbf9cec783f9dab01e05ee81bcffecef1dd80418"),
    ("old_continuation_harn", "e884a48a2cceb62b78e402c134a37a3c38c20ab8d9cb9460042415c20aaf0a69"),
    ("old_continuation_entry_binding", "b3fa27f2e48de8e695580d1ed5a3c75b792fa3112e3777fa2e6551afc0b3c777"),
    ("old_continuation_runtime_manifest", "23a1ff7d62edd8fee0b9f91eed416168a3eacb0774183024786b331cff5f634c"),
    ("old_top_controller", "b36068f0ae738502b5066d5c5de33909c6bdc6dd985b79c5e078c890f9be838f"),
    ("old_stage_entry_binding", "b4f20ab7932d927c527b0b718c8ced74d0150a023044767afeb0fd45b9d2890d"),
    ("old_repair2_entry_binding", "31feac13d7003dc809e5b55e1f020fff4e83f45935173ba3cba6c2d0a90681b5"),
    ("old_stage_exact_approval", "1203155ed2391b0678709b8bcaa9eb011094187325ab435914a623bbd28f9ec9"),
    ("old_repair2_exact_approval", "1c144a39313f2c8a297c553556ef8f06c82247734a034826c8fa65d67cd0752f"),
    ("old_continuation_exact_approval", "da6c64aaad27044bc0e0163dabab090b5dba43f37aab5796e5fbd1b921927daf"),
    ("old_stage_approval_state", "c50798abde8245d6de6adfd67879d58ce00f84196cd3699cd112e3bc936f1f17"),
    ("old_stage_authorization", "16499f9b8c45fb29d27f796db3efec089749dfa1f7f6015f730ed26738af397b"),
    ("failed_gen_000001_contract", "77b98e4b8dec04257493ae026f72dd71a2ebb03943b05f154ba8b73296780434"),
    ("failed_gen_000001_preflight", "61bcec074e11cb0f011eb74cbd68dbd05476b70b3a91c9158ff2772bf4f119a5"),
    ("failed_gen_000001_ledger", "b50adfea53fcd42da31cc84d1b67b63997eafc410016c0d8e90139f82a2e0961"),
    ("failed_gen_000001_queue", "e73e67e01540e2e47d51e4648bba0c16db24e8fac156d340d3b7a47fd79425da"),
    ("failed_gen_000002_contract", "77b98e4b8dec04257493ae026f72dd71a2ebb03943b05f154ba8b73296780434"),
    ("failed_gen_000002_preflight", "9db639928ec2725a03cbbbd0363537100ab0c97d4d904d08ed0ac91a8a3c873a"),
    ("failed_gen_000002_ledger", "a65a1656fa77ef52a87b73ad2a495458807ae90e4c7a155f0e4632bb64c72e35"),
    ("failed_gen_000002_queue", "f01fd2b87f1962e9ab33b852b20a759acbe8b00ac67a0d1555a4fa9e2b7f7350"),
    ("failed_stage_controller_lock", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
)

RETAINED_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("queue_local_signer", "/home/kojiek/logs/meanaudio_operator_backlog/queue_hmac.key",
     "39c029d383857d604c17af15b51b5e8753fca3a0bd56d336c6b8f72420efff80", "AUTH_ROOT_local_signer"),
    ("stage_child_ledger", "/home/kojiek/logs/rmatched_matrix_nvme_stage_harn/ledger_hmac.key",
     "b706d7fc214ebdab416f62142f01af22b781e9810db74afdb426050163f56eea", "retained_exact_binding_key"),
    ("repair2_child_ledger", "/home/kojiek/logs/rmatched_matrix_repair2_harn/ledger_hmac.key",
     "cc408f540d802b86227a52223a67018ad04d73528ee86a8884d63d88ca253fb5", "retained_child_ledger_key"),
    ("continuation_child_ledger", "/home/kojiek/logs/rmatched_matrix_continuation_harn/ledger_hmac.key",
     "d9d584bf2da0b9d3c9907e72a01326ad8694fe3739e65bc617f000023a7b4c8a", "retained_child_ledger_key"),
)

EXCLUSIVE_LOCKS: tuple[tuple[str, str], ...] = (
    ("top_controller", "/home/kojiek/logs/meanaudio_operator_queue_controller/controller.lock"),
    ("queue_mutation", "/home/kojiek/logs/meanaudio_operator_backlog/queue_mutation.lock"),
    ("stage_controller", "/home/kojiek/logs/rmatched_matrix_nvme_stage_harn/controller.lock"),
    ("repair2_controller", "/home/kojiek/logs/rmatched_matrix_repair2_harn/controller.lock"),
    ("continuation_controller", "/home/kojiek/logs/rmatched_matrix_continuation_harn/controller.lock"),
)
LOCK_FD_ENV = "MEANAUDIO_DUAL_AUTH_LOCK_FDS"


class GateResult:
    __slots__ = ("queue_sha256", "approval_sha256", "recovery_record_sha256",
                 "recovery_channel_sha256", "execution_channel_sha256", "entry_binding_sha256")

    def __init__(self, queue_sha256: str, approval_sha256: str, recovery_record_sha256: str,
                 recovery_channel_sha256: str, execution_channel_sha256: str,
                 entry_binding_sha256: str) -> None:
        self.queue_sha256 = queue_sha256
        self.approval_sha256 = approval_sha256
        self.recovery_record_sha256 = recovery_record_sha256
        self.recovery_channel_sha256 = recovery_channel_sha256
        self.execution_channel_sha256 = execution_channel_sha256
        self.entry_binding_sha256 = entry_binding_sha256


class GuardLease:
    """Ordered lock ownership retained across verification and the guarded action."""

    __slots__ = ("_fds", "_owned", "_roots", "_uid", "_queue_path", "_queue_key_path", "last_result")

    def __init__(self, fds: Mapping[str, int], *, owned: bool,
                 roots: tuple[Path, ...], uid: int, queue_path: Path, queue_key_path: Path) -> None:
        self._fds = dict(fds)
        self._owned = owned
        self._roots = roots
        self._uid = uid
        self._queue_path = queue_path
        self._queue_key_path = queue_key_path
        self.last_result: GateResult | None = None

    def lock_fd(self, role: str) -> int:
        try:
            return self._fds[role]
        except KeyError as exc:
            raise RuntimeError(f"guard lease lacks required lock role: {role}") from exc

    def duplicate_lock_fds(self) -> tuple[dict[str, int], tuple[int, ...]]:
        duplicates: dict[str, int] = {}
        try:
            for role, _path in EXCLUSIVE_LOCKS:
                duplicates[role] = os.dup(self.lock_fd(role))
            return duplicates, tuple(duplicates.values())
        except BaseException:
            for fd in duplicates.values():
                os.close(fd)
            raise

    def reverify(self, experiment_id: str, approval_record_path: Path | None = None) -> GateResult:
        self.last_result = _verify_gate_held(
            queue_path=self._queue_path, queue_key_path=self._queue_key_path,
            experiment_id=experiment_id, approval_record_path=approval_record_path,
            confinement_roots=self._roots, expected_uid=self._uid, held_locks=self._fds,
        )
        return self.last_result

    def verify_proposed(self, queue_payload: dict[str, Any], experiment_id: str,
                        approval_record_path: Path | None = None) -> GateResult:
        self.last_result = _verify_gate_held(
            queue_path=self._queue_path, queue_key_path=self._queue_key_path,
            experiment_id=experiment_id, approval_record_path=approval_record_path,
            confinement_roots=self._roots, expected_uid=self._uid, held_locks=self._fds,
            proposed_queue=queue_payload,
        )
        return self.last_result

    def close(self) -> None:
        if not self._owned:
            return
        for role, _path in reversed(EXCLUSIVE_LOCKS):
            fd = self._fds.pop(role, None)
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def entry_binding(entry: dict[str, Any]) -> str:
    runtime = entry.get("runtime", {})
    descriptor: dict[str, Any] = {
        "experiment_id": entry.get("experiment_id"), "run_id": entry.get("run_id"),
        "contract": entry.get("contract"), "contract_sha256": entry.get("contract_sha256"),
        "dependencies": entry.get("dependencies", []),
        "ordering_dependencies": entry.get("ordering_dependencies", []),
        "controller": runtime.get("controller"), "controller_sha256": runtime.get("controller_sha256"),
        "status_source": runtime.get("status_source"),
        "pending_status_source": runtime.get("pending_status_source"),
    }
    for key in ("bundle_validator", "completion_validator", "approval_cli", "import_bindings",
                "child_ledger_key", "transitive_runtime"):
        if key in runtime:
            descriptor[key] = runtime[key]
    return hashlib.sha256(canonical(descriptor)).hexdigest()


def _verify_mac(payload: dict[str, Any], key: bytes, domain: bytes, label: str) -> None:
    supplied = payload.get("integrity")
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError(f"{label} HMAC is invalid")


def sign_for_fixture(payload: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    """Test/candidate helper. Production callers still require the private key gate."""
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {**unsigned, "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()}


def _relative_to_one(path: Path, roots: Iterable[Path]) -> tuple[Path, Path]:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"recovery path is not canonical absolute: {path}")
    for root in roots:
        absolute_root = Path(os.path.abspath(root))
        absolute_path = Path(os.path.abspath(path))
        try:
            return absolute_root, absolute_path.relative_to(absolute_root)
        except ValueError:
            continue
    raise RuntimeError(f"recovery path escaped confinement roots: {path}")


def _open_beneath(path: Path, roots: Iterable[Path]) -> int:
    root, relative = _relative_to_one(path, roots)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    current_fd = root_fd
    try:
        parts = relative.parts
        if not parts:
            raise RuntimeError(f"recovery descriptor names a directory: {path}")
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        return os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def read_bound_file(path: Path, *, roots: Iterable[Path], expected_sha256: str | None,
                    allowed_modes: set[int], expected_uid: int, label: str) -> tuple[bytes, str]:
    fd = _open_beneath(path, roots)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != expected_uid or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) not in allowed_modes):
            raise RuntimeError(f"unsafe {label} descriptor: {path}")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        offset = 0
        while True:
            block = os.pread(fd, 8 << 20, offset)
            if not block:
                break
            blocks.append(block)
            digest.update(block)
            offset += len(block)
        after = os.fstat(fd)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink", "st_mode", "st_uid")
        if any(getattr(before, name) != getattr(after, name) for name in identity):
            raise RuntimeError(f"{label} changed while held by descriptor: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(observed, expected_sha256):
            raise RuntimeError(f"{label} hash drift: {path}")
        return b"".join(blocks), observed
    finally:
        os.close(fd)


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not exact UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _expected_revocations() -> list[dict[str, str]]:
    return [{"role": role, "sha256": digest} for role, digest in ORDERED_CLOSED_REVOCATIONS]


def _expected_keys() -> list[dict[str, str]]:
    return [{"role": role, "path": path, "sha256": digest, "classification": classification}
            for role, path, digest, classification in RETAINED_KEYS]


def _expected_locks() -> list[dict[str, str]]:
    return [{"role": role, "path": path, "sha256": hashlib.sha256(b"").hexdigest(),
             "classification": "exclusive_flock_no_create"} for role, path in EXCLUSIVE_LOCKS]


def _verify_archive(record: dict[str, Any], key: bytes, roots: tuple[Path, ...], uid: int) -> None:
    descriptor = record.get("archive_manifest")
    if not isinstance(descriptor, dict) or descriptor.get("sha256") != ARCHIVE_MANIFEST_SHA256:
        raise RuntimeError("recovery archive manifest descriptor is missing or unreviewed")
    path = Path(str(descriptor.get("path", "")))
    raw, observed = read_bound_file(path, roots=roots, expected_sha256=ARCHIVE_MANIFEST_SHA256,
                                    allowed_modes={0o600}, expected_uid=uid, label="recovery archive manifest")
    manifest = _parse_json(raw, "recovery archive manifest")
    _verify_mac(manifest, key, ARCHIVE_DOMAIN, "recovery archive manifest")
    if (manifest.get("document_kind") != "authenticated_failed_init_recovery_archive"
            or manifest.get("blocker_fingerprint") != BLOCKER_FINGERPRINT
            or manifest.get("prior_live_queue_sha256") != PRIOR_QUEUE_SHA256
            or manifest.get("recovery_channel_record", {}).get("sha256") != RECOVERY_AUTHORITY_SHA256
            or observed != descriptor.get("sha256")):
        raise RuntimeError("recovery archive manifest identity mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or hashlib.sha256(canonical(entries)).hexdigest() != manifest.get("entries_tree_sha256"):
        raise RuntimeError("recovery archive manifest entries tree is invalid")
    archive_root = path.parent
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError("recovery archive entry schema is invalid")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            raise RuntimeError("recovery archive entry path is unsafe or duplicated")
        seen.add(item["path"])
        entry_path = archive_root / relative
        entry_raw, entry_hash = read_bound_file(
            entry_path, roots=roots, expected_sha256=str(item.get("sha256", "")),
            allowed_modes={0o600}, expected_uid=uid, label="recovery archive entry",
        )
        if (entry_hash != item.get("sha256") or len(entry_raw) != item.get("size")
                or item.get("mode") != "0o600" or item.get("uid") != uid or item.get("nlink") != 1):
            raise RuntimeError("recovery archive entry metadata drift")


def _verify_retained_keys(record: dict[str, Any], roots: tuple[Path, ...], uid: int) -> None:
    if record.get("retained_keys") != _expected_keys():
        raise RuntimeError("retained key roles are missing, reordered, or reclassified")
    for item in record["retained_keys"]:
        read_bound_file(Path(item["path"]), roots=roots, expected_sha256=item["sha256"],
                        allowed_modes={0o600}, expected_uid=uid, label=f"retained key {item['role']}")


def _lock_identity(fd: int, path: Path, roots: tuple[Path, ...], uid: int, role: str) -> None:
    held = os.fstat(fd)
    probe = _open_beneath(path, roots)
    try:
        current = os.fstat(probe)
    finally:
        os.close(probe)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink", "st_mode", "st_uid")
    if any(getattr(held, name) != getattr(current, name) for name in identity):
        raise RuntimeError(f"exclusive lock pathname/descriptor identity drift: {role}")
    if (not stat.S_ISREG(held.st_mode) or held.st_uid != uid or held.st_nlink != 1
            or stat.S_IMODE(held.st_mode) != 0o600 or held.st_size != 0):
        raise RuntimeError(f"unsafe exclusive lock descriptor: {role}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"exclusive lock is held by another owner: {role}") from exc


def _acquire_lock_fds(roots: tuple[Path, ...], uid: int) -> dict[str, int]:
    acquired: dict[str, int] = {}
    try:
        for role, raw_path in EXCLUSIVE_LOCKS:
            fd = _open_beneath(Path(raw_path), roots)
            acquired[role] = fd
            _lock_identity(fd, Path(raw_path), roots, uid, role)
        return acquired
    except BaseException:
        for role, _path in reversed(EXCLUSIVE_LOCKS):
            fd = acquired.pop(role, None)
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        raise


def _verify_exclusive_locks(record: dict[str, Any], roots: tuple[Path, ...], uid: int,
                            held_locks: Mapping[str, int]) -> None:
    if record.get("exclusive_locks") != _expected_locks():
        raise RuntimeError("exclusive lock roles are missing, reordered, or reclassified")
    if list(held_locks) != [role for role, _path in EXCLUSIVE_LOCKS]:
        raise RuntimeError("exclusive lock FD roles are missing, reordered, or extended")
    for item in record["exclusive_locks"]:
        _lock_identity(held_locks[item["role"]], Path(item["path"]), roots, uid, item["role"])


def _descriptor_uid(path: Path, operator_uid: int) -> int:
    try:
        path.relative_to("/usr")
    except ValueError:
        return operator_uid
    return 0


def _verify_entry_files(entry: dict[str, Any], roots: tuple[Path, ...], uid: int) -> None:
    runtime = entry.get("runtime", {})
    descriptors: list[tuple[str, str, str]] = [
        ("contract", str(entry.get("contract", "")), str(entry.get("contract_sha256", ""))),
        ("controller", str(runtime.get("controller", "")), str(runtime.get("controller_sha256", ""))),
    ]
    for name in ("bundle_validator", "completion_validator"):
        value = runtime.get(name)
        if not isinstance(value, dict):
            continue
        if value.get("path") and value.get("sha256"):
            descriptors.append((name, str(value["path"]), str(value["sha256"])))
        if value.get("python") and value.get("python_sha256"):
            descriptors.append((f"{name}_python", str(value["python"]), str(value["python_sha256"])))
        for index, binding in enumerate(value.get("bindings", [])):
            if isinstance(binding, dict):
                descriptors.append((f"{name}_binding_{index}", str(binding.get("path", "")),
                                    str(binding.get("sha256", ""))))
    for index, binding in enumerate(runtime.get("import_bindings", [])):
        if isinstance(binding, dict):
            descriptors.append((f"import_binding_{index}", str(binding.get("path", "")),
                                str(binding.get("sha256", ""))))
    ledger_key = runtime.get("child_ledger_key")
    if isinstance(ledger_key, dict):
        descriptors.append(("child_ledger_key", str(ledger_key.get("path", "")),
                            str(ledger_key.get("sha256", ""))))
    transitive = runtime.get("transitive_runtime")
    if isinstance(transitive, dict):
        descriptors.append(("transitive_runtime_manifest", str(transitive.get("manifest", "")),
                            str(transitive.get("manifest_sha256", ""))))
        verifier = transitive.get("verifier")
        if isinstance(verifier, dict):
            descriptors.append(("transitive_runtime_verifier", str(verifier.get("path", "")),
                                str(verifier.get("sha256", ""))))
    seen: set[str] = set()
    for label, raw_path, expected in descriptors:
        if not raw_path or len(expected) != 64 or raw_path in seen:
            if raw_path in seen:
                continue
            raise RuntimeError(f"recovery-gated {label} descriptor is incomplete")
        seen.add(raw_path)
        descriptor_path = Path(raw_path)
        read_bound_file(descriptor_path, roots=roots, expected_sha256=expected,
                        allowed_modes={0o400, 0o440, 0o444, 0o500, 0o550, 0o555,
                                       0o600, 0o640, 0o644, 0o660, 0o664, 0o700, 0o750, 0o755},
                        expected_uid=_descriptor_uid(descriptor_path, uid), label=f"queue {label}")


def _default_roots() -> tuple[Path, ...]:
    return (
        Path("/home/kojiek/logs/meanaudio_operator_backlog"),
        Path("/home/kojiek/logs/meanaudio_operator_queue_controller"),
        Path("/home/kojiek/logs/rmatched_matrix_nvme_stage_harn"),
        Path("/home/kojiek/logs/rmatched_matrix_repair2_harn"),
        Path("/home/kojiek/logs/rmatched_matrix_continuation_harn"),
        Path("/home/kojiek/MeanAudio"),
        Path("/usr"),
    )


def _verify_gate_held(*, queue_path: Path, queue_key_path: Path, experiment_id: str,
                      approval_record_path: Path | None = None,
                      confinement_roots: Iterable[Path] | None = None,
                      expected_uid: int | None = None,
                      held_locks: Mapping[str, int],
                      proposed_queue: dict[str, Any] | None = None) -> GateResult:
    """Verify complete all-of authority while the declared ordered locks are held."""
    uid = os.geteuid() if expected_uid is None else expected_uid
    roots = tuple(confinement_roots or _default_roots())
    key, key_hash = read_bound_file(queue_key_path, roots=roots, expected_sha256=None,
                                    allowed_modes={0o600}, expected_uid=uid, label="queue signer key")
    if len(key) < 32:
        raise RuntimeError("queue signer key is invalid")
    if proposed_queue is None:
        queue_raw, queue_hash = read_bound_file(
            queue_path, roots=roots, expected_sha256=None,
            allowed_modes={0o600}, expected_uid=uid, label="signed operator queue",
        )
    else:
        queue_raw = (json.dumps(proposed_queue, indent=2, sort_keys=True) + "\n").encode()
        queue_hash = hashlib.sha256(queue_raw).hexdigest()
    queue = _parse_json(queue_raw, "signed operator queue")
    _verify_mac(queue, key, QUEUE_DOMAIN, "signed operator queue")
    if queue.get("document_kind") != "operator_approved_experiment_backlog":
        raise RuntimeError("operator queue document kind is invalid")
    descriptor = queue.get("recovery_authorization")
    if (not isinstance(descriptor, dict) or descriptor.get("mode") != "all_of"
            or descriptor.get("plan_fingerprint") != PLAN_FINGERPRINT):
        raise RuntimeError("signed queue lacks mandatory dual-authority recovery descriptor")
    recovery_path = Path(str(descriptor.get("path", "")))
    recovery_raw, recovery_hash = read_bound_file(
        recovery_path, roots=roots, expected_sha256=str(descriptor.get("sha256", "")),
        allowed_modes={0o400, 0o600}, expected_uid=uid, label="recovery authorization record",
    )
    recovery = _parse_json(recovery_raw, "recovery authorization record")
    _verify_mac(recovery, key, RECOVERY_DOMAIN, "recovery authorization record")
    expected_identity = {
        "document_kind": "dual_authority_recovery_authorization", "schema_version": 3,
        "status": "approved", "plan_fingerprint": PLAN_FINGERPRINT,
        "authorization_mode": "all_of", "blocker_fingerprint": BLOCKER_FINGERPRINT,
        "prior_queue_sha256": PRIOR_QUEUE_SHA256,
    }
    if any(recovery.get(name) != value for name, value in expected_identity.items()):
        raise RuntimeError("recovery authorization identity mismatch")
    execution = recovery.get("execution_authority")
    recovery_role = recovery.get("recovery_authority")
    if (execution != {"role": "scientific_execution", "channel_record_sha256": EXECUTION_AUTHORITY_SHA256,
                      "hmac_domain": "meanaudio-queue-approval-v1"}
            or not isinstance(recovery_role, dict)
            or recovery_role.get("role") != "binding_recovery"
            or recovery_role.get("channel_record_sha256") != RECOVERY_AUTHORITY_SHA256
            or recovery_role.get("hmac_domain") != "meanaudio-dual-authority-recovery-v3"):
        raise RuntimeError("dual authority roles were swapped, weakened, or omitted")
    text_descriptor = recovery_role.get("instruction_evidence")
    if (not isinstance(text_descriptor, dict)
            or text_descriptor.get("sha256") != RECOVERY_AUTHORITY_SHA256
            or text_descriptor.get("byte_length") != RECOVERY_AUTHORITY_BYTES):
        raise RuntimeError("recovery instruction descriptor is invalid")
    text_raw, _ = read_bound_file(Path(str(text_descriptor.get("path", ""))), roots=roots,
                                  expected_sha256=RECOVERY_AUTHORITY_SHA256, allowed_modes={0o400},
                                  expected_uid=uid, label="recovery instruction evidence")
    if len(text_raw) != RECOVERY_AUTHORITY_BYTES or text_raw.endswith(b"\n"):
        raise RuntimeError("recovery instruction bytes are not exact")
    if recovery.get("ordered_closed_revocations") != _expected_revocations():
        raise RuntimeError("closed revocation set is missing, reordered, extended, or rolled back")
    if recovery.get("affected_runs") != [
        {"experiment_id": experiment, "run_id": run} for experiment, run in EXPERIMENTS
    ]:
        raise RuntimeError("recovery affected-run order is invalid")
    _verify_archive(recovery, key, roots, uid)
    _verify_retained_keys(recovery, roots, uid)
    _verify_exclusive_locks(recovery, roots, uid, held_locks)
    if key_hash != dict((role, digest) for role, _path, digest, _class in RETAINED_KEYS)["queue_local_signer"]:
        raise RuntimeError("queue signer does not match retained key classification")

    queue_entries = queue.get("entries")
    if not isinstance(queue_entries, list):
        raise RuntimeError("recovery-gated queue entries are missing")
    affected: dict[str, dict[str, Any]] = {}
    expected_replacements: list[dict[str, Any]] = []
    for expected_experiment, expected_run_id, expected_position, dependencies, ordering in QUEUE_LAYOUT:
        matches = [entry for entry in queue_entries
                   if isinstance(entry, dict) and entry.get("experiment_id") == expected_experiment]
        if len(matches) != 1:
            raise RuntimeError("recovery-gated experiment is not unique in queue")
        current = matches[0]
        if (current.get("run_id") != expected_run_id or current.get("position") != expected_position
                or current.get("dependencies") != list(dependencies)
                or current.get("ordering_dependencies") != list(ordering)
                or current.get("approval_status") != "approved"):
            raise RuntimeError("recovery-gated queue order, dependencies, approval, or run drifted")
        _verify_entry_files(current, roots, uid)
        binding = entry_binding(current)
        revoked_hashes = {digest for _role, digest in ORDERED_CLOSED_REVOCATIONS}
        evidence_hash = current.get("approval_evidence", {}).get("sha256")
        if (current.get("contract_sha256") in revoked_hashes
                or current.get("runtime", {}).get("controller_sha256") in revoked_hashes
                or evidence_hash in revoked_hashes or binding in BASELINE_ONLY_BINDINGS
                or binding in revoked_hashes):
            raise RuntimeError("recovery-gated entry reuses revoked or baseline-only authority")
        affected[expected_experiment] = current
        expected_replacements.append({
            "experiment_id": expected_experiment, "run_id": expected_run_id,
            "contract_sha256": current.get("contract_sha256"),
            "controller_sha256": current.get("runtime", {}).get("controller_sha256"),
            "queue_entry_binding_sha256": binding,
        })
    replacements = recovery.get("replacement_entries")
    if replacements != expected_replacements:
        raise RuntimeError("recovery replacement entry set is missing, reordered, extended, or drifted")
    entry = affected.get(experiment_id)
    if entry is None:
        raise RuntimeError("recovery-gated experiment is not in the closed affected set")
    expected_run = dict(EXPERIMENTS).get(experiment_id)
    if expected_run is None:
        raise RuntimeError("recovery-gated queue entry is wrong-run")
    binding = entry_binding(entry)

    evidence = entry.get("approval_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("recovery-gated entry lacks exact approval evidence")
    selected_path = approval_record_path or Path(str(evidence.get("path", "")))
    if selected_path != Path(str(evidence.get("path", ""))):
        raise RuntimeError("caller approval record differs from signed queue evidence")
    approval_raw, approval_hash = read_bound_file(
        selected_path, roots=roots, expected_sha256=str(evidence.get("sha256", "")),
        allowed_modes={0o400, 0o600}, expected_uid=uid, label="exact execution approval record",
    )
    approval = _parse_json(approval_raw, "exact execution approval record")
    _verify_mac(approval, key, APPROVAL_DOMAIN, "exact execution approval record")
    conjunction = approval.get("authority_conjunction")
    if conjunction != {
        "mode": "all_of", "execution_channel_record_sha256": EXECUTION_AUTHORITY_SHA256,
        "recovery_channel_record_sha256": RECOVERY_AUTHORITY_SHA256,
        "recovery_authorization_sha256": recovery_hash,
        "plan_fingerprint": PLAN_FINGERPRINT,
    }:
        raise RuntimeError("approval record lacks the exact all-of authority conjunction")
    expected_approval = {
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": experiment_id, "run_id": expected_run,
        "contract_sha256": entry.get("contract_sha256"),
        "controller_sha256": entry.get("runtime", {}).get("controller_sha256"),
        "queue_entry_binding_sha256": binding,
        "channel_record_sha256": EXECUTION_AUTHORITY_SHA256,
    }
    if any(approval.get(name) != value for name, value in expected_approval.items()):
        raise RuntimeError("execution approval record identity or role mismatch")
    if entry.get("runtime", {}).get("approval_text_sha256") != EXECUTION_AUTHORITY_SHA256:
        raise RuntimeError("queue execution authority is not the exact c6 record")
    return GateResult(queue_hash, approval_hash, recovery_hash, RECOVERY_AUTHORITY_SHA256,
                      EXECUTION_AUTHORITY_SHA256, binding)


def verify_gate(*, queue_path: Path, queue_key_path: Path, experiment_id: str,
                approval_record_path: Path | None = None,
                confinement_roots: Iterable[Path] | None = None,
                expected_uid: int | None = None) -> GateResult:
    """Read-only verification; ordered locks are acquired and retained until verification ends."""
    uid = os.geteuid() if expected_uid is None else expected_uid
    roots = tuple(confinement_roots or _default_roots())
    lease = GuardLease(_acquire_lock_fds(roots, uid), owned=True, roots=roots, uid=uid,
                       queue_path=queue_path, queue_key_path=queue_key_path)
    try:
        lease.last_result = _verify_gate_held(
            queue_path=queue_path, queue_key_path=queue_key_path, experiment_id=experiment_id,
            approval_record_path=approval_record_path, confinement_roots=roots,
            expected_uid=uid, held_locks=lease._fds,
        )
        return lease.last_result
    finally:
        lease.close()


def production_gate(experiment_id: str, approval_record_path: Path | None = None) -> GateResult:
    return verify_gate(
        queue_path=Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json"),
        queue_key_path=Path("/home/kojiek/logs/meanaudio_operator_backlog/queue_hmac.key"),
        experiment_id=experiment_id, approval_record_path=approval_record_path,
    )


_ACTIVE_LEASE: GuardLease | None = None


def _inherited_lock_fds() -> dict[str, int] | None:
    raw = os.environ.get(LOCK_FD_ENV)
    if raw is None:
        return None
    pairs = raw.split(",") if raw else []
    parsed: dict[str, int] = {}
    for pair in pairs:
        role, separator, fd_text = pair.partition("=")
        if not separator or role in parsed or not fd_text.isdecimal():
            raise RuntimeError("inherited recovery lock FD mapping is malformed")
        parsed[role] = int(fd_text)
    if list(parsed) != [role for role, _path in EXCLUSIVE_LOCKS]:
        raise RuntimeError("inherited recovery lock FD roles are missing, reordered, or extended")
    return parsed


def inherited_lock_env(lease: GuardLease) -> tuple[str, tuple[int, ...]]:
    """Duplicate a lease for pass_fds; the caller owns and closes the duplicates."""
    duplicates, fds = lease.duplicate_lock_fds()
    value = ",".join(f"{role}={duplicates[role]}" for role, _path in EXCLUSIVE_LOCKS)
    return value, fds


@contextmanager
def guarded_action(experiment_id: str, approval_record_path: Path | None = None,
                   *, confinement_roots: Iterable[Path] | None = None,
                   expected_uid: int | None = None,
                   queue_path: Path | None = None,
                   queue_key_path: Path | None = None) -> Iterator[GuardLease]:
    """Hold the exact ordered lock set across verification and caller action.

    Nested actions reverify against the existing lease. A controller-spawned
    child may use only inherited FDs that share the parent's locked open-file
    descriptions; direct invocations acquire the full lock set themselves.
    """
    global _ACTIVE_LEASE
    uid = os.geteuid() if expected_uid is None else expected_uid
    roots = tuple(confinement_roots or _default_roots())
    if _ACTIVE_LEASE is not None:
        _ACTIVE_LEASE.reverify(experiment_id, approval_record_path)
        yield _ACTIVE_LEASE
        return
    inherited = _inherited_lock_fds()
    lease = GuardLease(
        inherited or _acquire_lock_fds(roots, uid), owned=inherited is None,
        roots=roots, uid=uid,
        queue_path=queue_path or Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json"),
        queue_key_path=queue_key_path or Path("/home/kojiek/logs/meanaudio_operator_backlog/queue_hmac.key"),
    )
    _ACTIVE_LEASE = lease
    try:
        lease.reverify(experiment_id, approval_record_path)
        yield lease
    finally:
        _ACTIVE_LEASE = None
        lease.close()
