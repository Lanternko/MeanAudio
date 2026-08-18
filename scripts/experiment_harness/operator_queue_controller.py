#!/usr/bin/env python3
"""Top-level durable controller for automatic MeanAudio queue handoff."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
QUEUE_KEY = QUEUE.parent / "queue_hmac.key"
QUEUE_LOCK = QUEUE.parent / "queue_mutation.lock"
STATE = Path("/home/kojiek/logs/meanaudio_operator_queue_controller")
LOCK = STATE / "controller.lock"
STATUS = STATE / "status.json"
OUTBOX = STATE / "outbox"
LOGS = STATE / "children"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
PYTHON = Path("/usr/bin/python3.12")
POLL_SECONDS = 20
TERMINAL = {"completed", "failed", "interrupted"}
WAITING_PREFIXES = ("queued", "preregistered_waiting", "preregistered_held")
CHILD_LIVE = {"ready", "active", "held"}
DANGEROUS_ENV = {"PYTHONPATH", "PYTHONHOME", "MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE", "MEANAUDIO_NOTIFY_DRY_RUN"}
CONTRACT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RECOVERY_VALIDATOR = ROOT / "scripts/dual_authority_recovery_v3.py"
RECOVERY_VALIDATOR_SHA256 = "ffff8ff4bb8b260db378800296d5202c80784094f620a4deaaf0c2b44f466c19"
RECOVERY_EXPERIMENTS = {
    "rmatched-s1-s2-steps-cfg-matrix-nvme-stage",
    "rmatched-s1-s2-steps-cfg-matrix-repair2",
    "rmatched-s1-s2-steps-cfg-matrix-continuation",
}
LEGACY_RUNTIME_REQUIRED_ROLES = {
    "/home/kojiek/MeanAudio/docs/experiments/rmatched_repair2_runtime_manifest.json": frozenset({
        "system_python", "system_stdlib", "system_native_runtime",
        "stage_bundle_validator", "recovery_validator",
    }),
    "/home/kojiek/MeanAudio/docs/experiments/rmatched_continuation_runtime_manifest.json": frozenset({
        "system_python", "system_stdlib", "dac_pyvenv_cfg", "dac_site_packages",
        "peav_pyvenv_cfg", "peav_site_packages", "workspace_meanaudio",
        "workspace_eval", "workspace_meanaudio_init", "phase4_evaluator",
        "peav_evaluator", "isolated_bootstrap", "system_native_runtime",
        "recovery_validator", "bash", "df", "tail", "tr", "find", "wc",
        "mkdir", "tee", "dirname",
    }),
}


def _load_recovery_validator() -> dict[str, Any]:
    fd = os.open(RECOVERY_VALIDATOR, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("dual-authority recovery validator is not an exact regular file")
        source = os.pread(fd, info.st_size, 0)
        if hashlib.sha256(source).hexdigest() != RECOVERY_VALIDATOR_SHA256:
            raise RuntimeError("dual-authority recovery validator drift")
        namespace: dict[str, Any] = {
            "__name__": "dual_authority_recovery_v3_bound", "__file__": f"/proc/self/fd/{fd}",
        }
        exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)
        return namespace
    finally:
        os.close(fd)


RECOVERY = _load_recovery_validator()
ACTIVE_RECOVERY_LEASE: Any | None = None


@contextmanager
def recovery_guard(experiment_id: str | None = None,
                   approval_record: Path | None = None):
    global ACTIVE_RECOVERY_LEASE
    targets = (experiment_id,) if experiment_id is not None else tuple(sorted(RECOVERY_EXPERIMENTS))
    with RECOVERY["guarded_action"](targets[0], approval_record) as lease:
        prior = ACTIVE_RECOVERY_LEASE
        ACTIVE_RECOVERY_LEASE = lease
        try:
            for target in targets[1:]:
                lease.reverify(target)
            yield lease
        finally:
            ACTIVE_RECOVERY_LEASE = prior


def recovery_gate(experiment_id: str | None = None, approval_record: Path | None = None) -> None:
    with recovery_guard(experiment_id, approval_record):
        pass


def recovery_gate_for_entry(entry: dict[str, Any]) -> None:
    experiment_id = entry.get("experiment_id")
    if experiment_id in RECOVERY_EXPERIMENTS:
        evidence = entry.get("approval_evidence", {})
        recovery_gate(str(experiment_id), Path(str(evidence.get("path", ""))))


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def load_private_key(path: Path | None = None) -> bytes:
    path = path or QUEUE_KEY
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.geteuid() or (stat.st_mode & 0o777) != 0o600:
            raise RuntimeError(f"unsafe queue authentication key: {path}")
        key = os.read(fd, 128)
    finally:
        os.close(fd)
    if len(key) < 32:
        raise RuntimeError("invalid queue authentication key")
    return key


def sign_document(payload: dict[str, Any], domain: bytes, key: bytes | None = None) -> dict[str, Any]:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    signature = hmac.new(key or load_private_key(), domain + canonical(unsigned), hashlib.sha256).hexdigest()
    return {**unsigned, "integrity": signature}


def verify_document(payload: dict[str, Any], domain: bytes, key: bytes | None = None) -> None:
    supplied = payload.get("integrity")
    expected = sign_document(payload, domain, key)["integrity"]
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated queue document signature is invalid")


def entry_approval_binding(entry: dict[str, Any]) -> str:
    runtime = entry.get("runtime", {})
    immutable_runtime = {
        key: runtime.get(key) for key in (
            "controller", "controller_sha256", "status_source", "pending_status_source",
            "bundle_validator", "completion_validator", "approval_cli", "import_bindings", "child_ledger_key",
            "transitive_runtime",
        ) if key in runtime
    }
    descriptor = {
        "experiment_id": entry.get("experiment_id"), "run_id": entry.get("run_id"),
        "contract": entry.get("contract"), "contract_sha256": entry.get("contract_sha256"),
        "dependencies": entry.get("dependencies", []),
        "ordering_dependencies": entry.get("ordering_dependencies", []),
        "controller": immutable_runtime.get("controller"),
        "controller_sha256": immutable_runtime.get("controller_sha256"),
        "status_source": immutable_runtime.get("status_source"),
        "pending_status_source": immutable_runtime.get("pending_status_source"),
    }
    # Validator descriptors are included only when present, matching the
    # Repair2 HARN's binding algorithm while binding newer hardened entries.
    if "bundle_validator" in immutable_runtime:
        descriptor["bundle_validator"] = immutable_runtime["bundle_validator"]
    if "completion_validator" in immutable_runtime:
        descriptor["completion_validator"] = immutable_runtime["completion_validator"]
    for key in ("approval_cli", "import_bindings", "child_ledger_key", "transitive_runtime"):
        if key in immutable_runtime:
            descriptor[key] = immutable_runtime[key]
    return hashlib.sha256(canonical(descriptor)).hexdigest()


def safe_child_env() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE"}
        and key not in DANGEROUS_ENV
    } | {"PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1"}


def child_lock_context() -> tuple[dict[str, str], tuple[int, ...]]:
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("child launch lacks retained recovery lock lease")
    lock_env, lock_fds = RECOVERY["inherited_lock_env"](ACTIVE_RECOVERY_LEASE)
    return safe_child_env() | {RECOVERY["LOCK_FD_ENV"]: lock_env}, lock_fds


def open_verified_fd(path: Path, expected_sha256: str) -> int:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError(f"child executable is not a regular file: {path}")
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(fd, 8 << 20, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"child executable descriptor drift: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def verify_entry_transitive_runtime(entry: dict[str, Any]) -> dict[str, Any]:
    descriptor = entry.get("runtime", {}).get("transitive_runtime")
    if not isinstance(descriptor, dict):
        raise RuntimeError("child has no approval-bound transitive runtime manifest")
    verifier = descriptor.get("verifier")
    if isinstance(verifier, str):
        # Older preregistered contracts represent verifier as a path plus a
        # sibling hash; normalize that immutable form to the descriptor shape
        # used by newer queue entries before performing the same FD/hash gate.
        verifier = {"path": verifier, "sha256": descriptor.get("verifier_sha256")}
    if not isinstance(verifier, dict):
        raise RuntimeError("child transitive runtime verifier descriptor is missing")
    fd = open_verified_fd(Path(str(verifier.get("path", ""))), str(verifier.get("sha256", "")))
    try:
        namespace: dict[str, Any] = {"__name__": "verified_runtime_binding", "__file__": f"/proc/self/fd/{fd}"}
        source = os.pread(fd, os.fstat(fd).st_size, 0)
        exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)
    finally:
        os.close(fd)
    roles = descriptor.get("required_roles")
    if roles is None:
        roles = sorted(LEGACY_RUNTIME_REQUIRED_ROLES.get(str(Path(str(descriptor.get("manifest", "")))), ()))
    if not isinstance(roles, list) or not roles or not all(isinstance(role, str) for role in roles):
        raise RuntimeError("child transitive runtime required roles are invalid")
    return namespace["verify_manifest"](
        Path(str(descriptor.get("manifest", ""))), str(descriptor.get("manifest_sha256", "")), set(roles),
    )


CONTROLLER_BOOTSTRAP = r"""
import importlib.util, json, os, sys
from importlib.machinery import SourceFileLoader
controller_fd = int(sys.argv[1])
imports = json.loads(sys.argv[2])
controller_args = sys.argv[3:]
for name, imported_fd in imports:
    source = f"/proc/self/fd/{int(imported_fd)}"
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verified module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
controller_source = f"/proc/self/fd/{controller_fd}"
sys.argv = [controller_source, *controller_args]
namespace = {"__name__": "__main__", "__file__": controller_source, "__builtins__": __builtins__}
with open(controller_source, "rb") as handle:
    code = compile(handle.read(), controller_source, "exec")
exec(code, namespace, namespace)
"""


def verified_controller_invocation(entry: dict[str, Any], arguments: list[str]) -> tuple[list[str], tuple[int, ...]]:
    runtime = entry.get("runtime", {})
    manifest = verify_entry_transitive_runtime(entry)
    python_entry = next((item for item in manifest["entries"] if item.get("role") == "system_python"), None)
    if not isinstance(python_entry, dict) or python_entry.get("kind") != "file":
        raise RuntimeError("child runtime manifest has no exact system Python")
    python_fd = open_verified_fd(Path(python_entry["path"]), python_entry["sha256"])
    controller = Path(str(runtime.get("controller", "")))
    try:
        controller_fd = open_verified_fd(controller, str(runtime.get("controller_sha256", "")))
    except BaseException:
        os.close(python_fd)
        raise
    fds = [python_fd, controller_fd]
    imports: list[list[Any]] = []
    try:
        descriptors = runtime.get("import_bindings")
        if not isinstance(descriptors, list) or not descriptors:
            raise RuntimeError("child controller has no exact imported-module bindings")
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("module"), str):
                raise RuntimeError("invalid child imported-module descriptor")
            imported_fd = open_verified_fd(Path(str(descriptor.get("path", ""))), str(descriptor.get("sha256", "")))
            fds.append(imported_fd)
            imports.append([descriptor["module"], imported_fd])
        required_imports = {"qwen_s2q_k_mf25_harn", "runtime_binding"}
        if entry.get("experiment_id") in {
            "rmatched-s1-s2-steps-cfg-matrix-nvme-stage",
            "rmatched-s1-s2-steps-cfg-matrix-repair2",
            "rmatched-s1-s2-steps-cfg-matrix-continuation",
        } and not required_imports.issubset({name for name, _fd in imports}):
            raise RuntimeError("hardened child lacks exact shared-HARN/runtime-verifier imports")
        argv = [f"/proc/self/fd/{python_fd}", "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                "-c", CONTROLLER_BOOTSTRAP,
                str(controller_fd), json.dumps(imports), *arguments]
        return argv, tuple(fds)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def acquire_queue_lock() -> int:
    if ACTIVE_RECOVERY_LEASE is not None:
        return os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("queue_mutation"))
    QUEUE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(QUEUE_LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    recovery_gate()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_queue(payload: dict[str, Any]) -> None:
    recovery_gate()
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("queue commit lacks retained recovery lock lease")
    proposed = sign_document(payload, b"meanaudio-operator-queue-v1\0")
    for experiment_id in tuple(sorted(RECOVERY_EXPERIMENTS)):
        matches = [entry for entry in proposed.get("entries", [])
                   if entry.get("experiment_id") == experiment_id]
        if len(matches) != 1:
            raise RuntimeError("proposed queue lacks unique recovery-gated entry")
        evidence = matches[0].get("approval_evidence", {})
        approval = Path(str(evidence.get("path", ""))) if evidence.get("path") else None
        ACTIVE_RECOVERY_LEASE.verify_proposed(proposed, experiment_id, approval)
    atomic_json(QUEUE, proposed, 0o600)


def load_queue(path: Path = QUEUE) -> dict[str, Any]:
    if path == QUEUE:
        recovery_gate()
    payload = json.loads(path.read_text())
    validate_queue(payload, authenticate=True)
    return payload


def validate_approval_record(entry: dict[str, Any]) -> dict[str, Any]:
    recovery_gate_for_entry(entry)
    evidence = entry.get("approval_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"approved entry lacks exact approval evidence: {entry['experiment_id']}")
    path = Path(str(evidence.get("path", "")))
    if not path.is_file() or digest_file(path) != evidence.get("sha256"):
        raise RuntimeError(f"approved entry approval evidence drift: {entry['experiment_id']}")
    record = json.loads(path.read_text())
    verify_document(record, b"meanaudio-queue-approval-v1\0")
    runtime = entry.get("runtime", {})
    expected = {
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": entry.get("experiment_id"), "run_id": entry.get("run_id"),
        "contract_sha256": entry.get("contract_sha256"),
        "controller_sha256": runtime.get("controller_sha256"),
        "queue_entry_binding_sha256": entry_approval_binding(entry),
    }
    mismatches = {key: (value, record.get(key)) for key, value in expected.items() if record.get(key) != value}
    channel_hash = record.get("channel_record_sha256")
    if mismatches or not isinstance(channel_hash, str) or len(channel_hash) != 64:
        raise RuntimeError(f"approved entry exact approval mismatch: {entry['experiment_id']} {mismatches}")
    if runtime.get("approval_text_sha256") not in {None, channel_hash}:
        raise RuntimeError(f"approved entry channel approval mismatch: {entry['experiment_id']}")
    return record


def validate_queue(payload: dict[str, Any], *, authenticate: bool = False) -> None:
    if authenticate:
        verify_document(payload, b"meanaudio-operator-queue-v1\0")
    entries = payload.get("entries")
    if payload.get("document_kind") != "operator_approved_experiment_backlog" or not isinstance(entries, list):
        raise RuntimeError("invalid operator queue document")
    controller_binding = payload.get("controller_binding", {})
    if controller_binding:
        controller_path = Path(controller_binding.get("path", ""))
        if not controller_path.is_file() or digest_file(controller_path) != controller_binding.get("sha256"):
            raise RuntimeError("top-level queue controller binding drift")
    ids = [entry.get("experiment_id") for entry in entries]
    positions = [entry.get("position") for entry in entries]
    if len(ids) != len(set(ids)) or positions != list(range(1, len(entries) + 1)):
        raise RuntimeError("queue IDs must be unique and positions contiguous")
    for entry in entries:
        approval_status = entry.get("approval_status", "pending")
        if approval_status not in {"approved", "pending"}:
            raise RuntimeError(f"invalid approval status: {entry['experiment_id']}")
        contract = Path(entry["contract"])
        if not contract.is_file():
            raise RuntimeError(f"missing queue contract: {contract}")
        expected = entry.get("contract_sha256")
        if expected and digest_file(contract) != expected:
            raise RuntimeError(f"queue contract drift: {entry['experiment_id']}")
        runtime = entry.get("runtime", {})
        controller = runtime.get("controller")
        controller_hash = runtime.get("controller_sha256")
        if controller and controller_hash and digest_file(Path(controller)) != controller_hash:
            raise RuntimeError(f"queue controller drift: {entry['experiment_id']}")
        if authenticate and approval_status == "approved":
            validate_approval_record(entry)
        known = set(ids)
        for field in ("dependencies", "ordering_dependencies"):
            references = entry.get(field, [])
            if not isinstance(references, list) or any(item not in known for item in references):
                raise RuntimeError(f"invalid {field}: {entry['experiment_id']}")


def _verified_validator(descriptor: Any) -> tuple[Path, Path] | None:
    if not isinstance(descriptor, dict):
        return None
    python = Path(str(descriptor.get("python", "")))
    script = Path(str(descriptor.get("path", "")))
    if (not python.is_file() or not script.is_file()
            or digest_file(python) != descriptor.get("python_sha256")
            or digest_file(script) != descriptor.get("sha256")):
        return None
    return python, script


def _verify_child_ledger_hmac(entry: dict[str, Any], ledger: dict[str, Any], terminal_status: str) -> bool:
    descriptor = entry.get("runtime", {}).get("child_ledger_key")
    if not isinstance(descriptor, dict):
        return False
    path = Path(str(descriptor.get("path", "")))
    try:
        if not path.is_file() or path.is_symlink() or digest_file(path) != descriptor.get("sha256"):
            return False
        key = load_private_key(path)
        prior = None
        events = ledger.get("events")
        if not isinstance(events, list) or not events:
            return False
        terminal_kind = {"completed": "experiment_completed", "failed": "experiment_failed",
                         "interrupted": "experiment_interrupted"}.get(terminal_status)
        for sequence, event in enumerate(events, 1):
            if not isinstance(event, dict):
                return False
            supplied = event.get("event_sha256")
            unsigned = {name: value for name, value in event.items() if name != "event_sha256"}
            expected = hmac.new(
                key, b"meanaudio-harn-event-v1\0" + canonical(unsigned), hashlib.sha256,
            ).hexdigest()
            if (event.get("sequence") != sequence or event.get("previous_event_sha256") != prior
                    or not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected)):
                return False
            prior = supplied
        if terminal_kind and not any(event.get("event_kind") == terminal_kind for event in events):
            return False
        return True
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _load_verified_generation(entry: dict[str, Any], source: Path) -> dict[str, Any] | None:
    recovery_gate_for_entry(entry)
    runtime = entry.get("runtime", {})
    validator = _verified_validator(runtime.get("bundle_validator"))
    if validator is None:
        return None
    try:
        generation = Path(source.read_text().strip())
        state_root = source.parent.resolve(strict=True)
        resolved = generation.resolve(strict=True)
        if resolved.parent.parent != state_root or resolved.parent.name != "generations" or generation.is_symlink():
            return None
        paths = {name: resolved / f"{name}.json" for name in ("contract", "preflight", "ledger", "queue")}
        if any(not path.is_file() or path.is_symlink() for path in paths.values()):
            return None
        values = {name: json.loads(path.read_text()) for name, path in paths.items()}
        contract_hash = hashlib.sha256(canonical(values["contract"])).hexdigest()
        preflight_hash = hashlib.sha256(canonical(values["preflight"])).hexdigest()
        ledger_hash = hashlib.sha256(canonical(values["ledger"])).hexdigest()
        child_entries = values["queue"].get("entries", [])
        if len(child_entries) != 1:
            return None
        child = child_entries[0]
        if (values["contract"].get("experiment_id") != entry.get("experiment_id")
                or values["contract"].get("run_id") != entry.get("run_id")
                or values["preflight"].get("experiment_id") != entry.get("experiment_id")
                or values["preflight"].get("run_id") != entry.get("run_id")
                or values["ledger"].get("experiment_id") != entry.get("experiment_id")
                or values["ledger"].get("run_id") != entry.get("run_id")
                or child.get("experiment_id") != entry.get("experiment_id")
                or child.get("run_id") != entry.get("run_id")):
            return None
        contract_bindings = values["contract"].get("bindings", {})
        source_artifacts = values["contract"].get("corpus", {}).get("source_artifacts", [])
        source_map = {item.get("path"): item.get("sha256") for item in source_artifacts if isinstance(item, dict)}
        approval_evidence = entry.get("approval_evidence", {})
        bundle_approval_evidence = approval_evidence
        if child.get("status") in TERMINAL:
            # A terminal bundle may be queue-rebound after completion.  The
            # new queue approval authorizes the preserved successor handoff;
            # the completed bundle must still match the exact approval that
            # was consumed when it ran.
            preserved = runtime.get("transitive_runtime", {}).get("terminal_rebind", {})
            original = preserved.get("original_approval_evidence") if isinstance(preserved, dict) else None
            if isinstance(original, dict):
                bundle_approval_evidence = original
        if (contract_bindings.get("runtime_sha256") != runtime.get("controller_sha256")
                or source_map.get(entry.get("contract")) != entry.get("contract_sha256")
                or source_map.get(bundle_approval_evidence.get("path")) != bundle_approval_evidence.get("sha256")):
            return None
        if (values["preflight"].get("contract_raw_sha256") != contract_hash
                or values["ledger"].get("bindings", {}).get("contract_raw_sha256") != contract_hash
                or values["ledger"].get("bindings", {}).get("preflight_report_raw_sha256") != preflight_hash
                or child.get("bindings", {}).get("contract_raw_sha256") != contract_hash
                or child.get("bindings", {}).get("preflight_report_raw_sha256") != preflight_hash
                or child.get("bindings", {}).get("ledger_raw_sha256") != ledger_hash):
            return None
        if not _verify_child_ledger_hmac(entry, values["ledger"], str(child.get("status", ""))):
            return None
        python, script = validator
        completed = subprocess.run(
            [str(python), str(script), "--contract", str(paths["contract"]),
             "--preflight", str(paths["preflight"]), "--ledger", str(paths["ledger"]),
             "--queue", str(paths["queue"])],
            cwd=ROOT, text=True, capture_output=True, timeout=120, env=safe_child_env(),
        )
        if completed.returncode:
            return None
        return child
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def child_queue_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    source = entry.get("runtime", {}).get("status_source") or (
        QUEUE.parent / "missing" if not entry.get("status_source") else Path(entry["status_source"])
    )
    source = Path(source)
    if source.is_file():
        child = _load_verified_generation(entry, source)
        if child is not None:
            return child
    pending_source = entry.get("runtime", {}).get("pending_status_source")
    pending = Path(pending_source) if pending_source else source.parent / "pending_queue.json"
    if pending.is_file():
        try:
            payload = json.loads(pending.read_text())
            status = payload.get("status")
            contract_path = pending.parent / "pending_contract.json"
            contract = json.loads(contract_path.read_text())
            if (status == "held" and contract.get("experiment_id") == entry.get("experiment_id")
                    and contract.get("run_id") == entry.get("run_id")
                    and contract.get("bindings", {}).get("runtime_sha256") == entry.get("runtime", {}).get("controller_sha256")):
                return {"status": status, "terminal_notification_status": "not_applicable"}
        except (OSError, json.JSONDecodeError):
            pass
    return None


def child_status(entry: dict[str, Any]) -> str | None:
    child = child_queue_entry(entry)
    return child.get("status") if child else None


def terminal_evidence_delivered(entry: dict[str, Any]) -> bool:
    child = child_queue_entry(entry)
    if child is None:
        return False
    return (
        child.get("status") == entry.get("status")
        and child.get("terminal_notification_status") == "delivered"
    )


def dependency_blockers(entry: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_id = {item["experiment_id"]: item for item in entries}
    blockers: list[dict[str, str]] = []
    for dependency in entry.get("ordering_dependencies", []):
        prior = by_id[dependency]
        if prior.get("status") not in TERMINAL or not terminal_evidence_delivered(prior):
            blockers.append({"kind": "ordering", "experiment_id": dependency, "status": prior.get("status", "missing")})
    for dependency in entry.get("dependencies", []):
        prior = by_id[dependency]
        if prior.get("status") != "completed" or not terminal_evidence_delivered(prior):
            blockers.append({"kind": "scientific", "experiment_id": dependency, "status": prior.get("status", "missing")})
        elif not completion_evidence_valid(prior):
            blockers.append({"kind": "scientific_evidence", "experiment_id": dependency, "status": "invalid"})
    return blockers


def completion_evidence_valid(entry: dict[str, Any]) -> bool:
    recovery_gate_for_entry(entry)
    descriptor = entry.get("runtime", {}).get("completion_validator")
    if not isinstance(descriptor, dict):
        return False
    command = descriptor.get("argv")
    bindings = descriptor.get("bindings")
    if (not isinstance(command, list) or not command
            or not all(isinstance(item, str) and item for item in command)
            or not isinstance(bindings, list) or not bindings):
        return False
    bound_paths: set[str] = set()
    for binding in bindings:
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or digest_file(path) != binding.get("sha256"):
            return False
        bound_paths.add(str(path))
    if command[0] not in bound_paths:
        return False
    for argument in command[1:]:
        if argument.startswith("/") and Path(argument).is_file() and argument not in bound_paths:
            return False
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120, env=safe_child_env())
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def dependencies_satisfied(entry: dict[str, Any], entries: list[dict[str, Any]]) -> bool:
    return not dependency_blockers(entry, entries)


def eligibility_blockers(entry: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    blockers = dependency_blockers(entry, entries)
    if entry.get("approval_status", "pending") != "approved":
        blockers.append({
            "kind": "operator_approval", "experiment_id": entry["experiment_id"],
            "status": entry.get("approval_status", "missing"),
        })
    return blockers


def is_waiting(entry: dict[str, Any]) -> bool:
    return entry["status"].startswith(WAITING_PREFIXES)


def process_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return None


def process_alive(pid: Any, expected_start_ticks: Any = None) -> bool:
    try:
        numeric_pid = int(pid)
        os.kill(numeric_pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    return expected_start_ticks is None or process_start_ticks(numeric_pid) == str(expected_start_ticks)


def notify_once(key: str, summary: str, status: str = "test") -> None:
    recovery_gate()
    record = OUTBOX / f"{key}.json"
    if record.is_file() and json.loads(record.read_text()).get("status") == "delivered":
        return
    atomic_json(record, {"status": "attempting", "created_at": now(), "summary_sha256": hashlib.sha256(summary.encode()).hexdigest()})
    completed = subprocess.run(
        [str(PYTHON), str(NOTIFIER), "--status", status,
         "--experiment", "meanaudio-operator-queue", "--summary", summary],
        cwd=ROOT, text=True, capture_output=True, env=safe_child_env(),
    )
    if completed.returncode:
        atomic_json(record, {"status": "failed", "failed_at": now(), "error": completed.stderr[-500:]})
        raise RuntimeError(f"queue notification failed: {key}")
    atomic_json(record, {"status": "delivered", "delivered_at": now()})


def launch(entry: dict[str, Any]) -> None:
    recovery_gate_for_entry(entry)
    runtime = entry.get("runtime", {})
    controller = Path(runtime.get("controller", ""))
    if entry.get("approval_status", "pending") != "approved":
        raise RuntimeError(f"next entry lacks operator approval: {entry['experiment_id']}")
    approval = validate_approval_record(entry)
    evidence = entry["approval_evidence"]
    if (not controller.is_file() or digest_file(controller) != runtime.get("controller_sha256")
            or digest_file(Path(entry["contract"])) != entry.get("contract_sha256")):
        raise RuntimeError(f"next entry is not launch-prepared: {entry['experiment_id']}")
    status_source = Path(runtime["status_source"])
    state_root = status_source.parent
    prior = child_status(entry)
    if prior in TERMINAL:
        raise RuntimeError(f"terminal HARN state requires a new run ID before relaunch: {entry['experiment_id']}")
    if prior not in CHILD_LIVE:
        if (runtime.get("approval_cli") == "exact_record_v1"
                or entry.get("experiment_id") in {
                    "rmatched-s1-s2-steps-cfg-matrix-repair2",
                    "rmatched-s1-s2-steps-cfg-matrix-continuation",
                }):
            init_argv = [
                str(PYTHON), str(controller), "init", "--approval-record", evidence["path"],
                "--queue-entry-sha256", approval["queue_entry_binding_sha256"],
            ]
        else:
            init_argv = [
                str(PYTHON), str(controller), "init", "--approval-text-hash",
                approval["channel_record_sha256"],
            ]
        verified_argv, verified_fds = verified_controller_invocation(entry, init_argv[2:])
        child_env, lock_fds = child_lock_context()
        inherited_fds = (*verified_fds, *lock_fds)
        try:
            initialized = subprocess.run(
                verified_argv, cwd=ROOT, text=True, capture_output=True, env=child_env,
                pass_fds=inherited_fds,
            )
        finally:
            for fd in inherited_fds:
                os.close(fd)
        if initialized.returncode:
            raise RuntimeError(f"HARN init failed for {entry['experiment_id']}: {initialized.stderr[-1000:]}")
        prior = child_status(entry)
    if prior not in {"ready", "held", "active"}:
        raise RuntimeError(f"HARN did not become launchable: {entry['experiment_id']} status={prior}")
    notify_once(
        f"handoff-{entry['experiment_id']}-{entry['run_id']}",
        f"Queue handoff to {entry['experiment_id']}; contract and HARN preflight are ready.",
    )
    LOGS.mkdir(parents=True, exist_ok=True, mode=0o700)
    log = (LOGS / f"{entry['experiment_id']}.log").open("ab", buffering=0)
    verified_argv, verified_fds = verified_controller_invocation(entry, ["run"])
    child_env, lock_fds = child_lock_context()
    inherited_fds = (*verified_fds, *lock_fds)
    try:
        process = subprocess.Popen(
            verified_argv, cwd=ROOT, pass_fds=inherited_fds,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env,
        )
    finally:
        for fd in inherited_fds:
            os.close(fd)
    entry["runtime"]["controller_pid"] = process.pid
    entry["runtime"]["controller_start_ticks"] = process_start_ticks(process.pid)
    entry["runtime"]["launched_at"] = now()
    entry["status"] = "active"
    entry["status_source"] = str(status_source)
    entry["runtime"]["state_root"] = str(state_root)


def transition_once(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    changed = False
    active = next((entry for entry in payload["entries"] if entry["status"] == "active"), None)
    if active:
        observed = child_status(active)
        if observed in TERMINAL:
            active["status"] = observed
            active["terminal_observed_at"] = now()
            payload.setdefault("history", []).append({
                "kind": "terminal", "at": now(), "experiment_id": active["experiment_id"], "status": observed,
            })
            payload["active_experiment"] = None
            changed = True
        elif observed not in CHILD_LIVE:
            return changed, {"state": "queue_hold", "reason": "active_state_unverifiable", "entry": active["experiment_id"]}
        elif not process_alive(
            active.get("runtime", {}).get("controller_pid"),
            active.get("runtime", {}).get("controller_start_ticks"),
        ):
            fingerprint = hashlib.sha256(f"ready-without-controller:{active['experiment_id']}".encode()).hexdigest()[:16]
            active["status"] = "queued_retry_after_controller_exit"
            active["hold_reason"] = "child HARN is ready but its controller process is absent"
            active["hold_fingerprint"] = fingerprint
            payload["active_experiment"] = None
            notify_once(f"hold-{fingerprint}", f"Queue hold for {active['experiment_id']}: child HARN controller is absent.", status="held")
            changed = True
        else:
            return changed, {
                "state": "resource_wait" if observed == "held" else "running",
                "active_experiment": payload.get("active_experiment"),
            }
    next_entry = next(
        (entry for entry in payload["entries"] if is_waiting(entry) and not eligibility_blockers(entry, payload["entries"])),
        None,
    )
    if not next_entry:
        waiting = [entry for entry in payload["entries"] if is_waiting(entry)]
        remaining = [entry["experiment_id"] for entry in waiting]
        state = "drained" if not remaining else "dependency_hold"
        blockers = {entry["experiment_id"]: eligibility_blockers(entry, payload["entries"]) for entry in waiting}
        status = {"state": state, "remaining": remaining, "blockers": blockers}
        if remaining:
            raw = json.dumps(blockers, sort_keys=True)
            fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
            status["fingerprint"] = fingerprint
            notify_once(f"dependency-hold-{fingerprint}", f"Queue dependency hold: {raw}", status="held")
        return changed, status
    try:
        launch(next_entry)
    except Exception as exc:
        fingerprint = hashlib.sha256(f"{next_entry['experiment_id']}:{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]
        next_entry["status"] = "queued_launch_hold"
        next_entry["hold_reason"] = str(exc)
        next_entry["hold_fingerprint"] = fingerprint
        notify_once(f"hold-{fingerprint}", f"Queue hold before {next_entry['experiment_id']}: {exc}", status="held")
        changed = True
        return changed, {
            "state": "queue_hold", "entry": next_entry["experiment_id"],
            "reason": str(exc), "fingerprint": fingerprint,
        }
    else:
        payload["active_experiment"] = {
            "experiment_id": next_entry["experiment_id"],
            "run_id": next_entry["run_id"],
            "status_source": next_entry["runtime"]["status_source"],
        }
        payload.setdefault("history", []).append({
            "kind": "handoff", "at": now(), "experiment_id": next_entry["experiment_id"],
        })
        changed = True
    return changed, {"state": "running", "active_experiment": payload.get("active_experiment")}


def run(once: bool) -> None:
    recovery_gate()
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("top controller run lacks retained recovery lock lease")
    fd = os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("top_controller"))
    try:
        while True:
            queue_fd = acquire_queue_lock()
            try:
                try:
                    payload = load_queue()
                except Exception as exc:
                    fingerprint = hashlib.sha256(f"queue-load:{type(exc).__name__}:{exc}".encode()).hexdigest()[:16]
                    atomic_json(STATUS, {
                        "observed_at": now(), "state": "queue_hold", "reason": str(exc),
                        "fingerprint": fingerprint,
                    })
                    notify_once(f"hold-{fingerprint}", f"Top-level queue hold: {exc}", status="held")
                    if once:
                        raise
                    loop_status = None
                else:
                    prior_status = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
                    if prior_status.get("state") == "queue_hold" and prior_status.get("fingerprint"):
                        fingerprint = prior_status["fingerprint"]
                        notify_once(
                            f"recovery-{fingerprint}",
                            f"Top-level queue recovered from hold {fingerprint}; automatic handoff monitoring resumed.",
                        )
                    try:
                        changed, loop_status = transition_once(payload)
                    except Exception as exc:
                        fingerprint = hashlib.sha256(
                            f"transition:{type(exc).__name__}:{exc}".encode()
                        ).hexdigest()[:16]
                        atomic_json(STATUS, {
                            "observed_at": now(), "state": "queue_hold",
                            "reason": str(exc), "fingerprint": fingerprint,
                        })
                        if once:
                            raise
                        loop_status = None
                    else:
                        if changed:
                            payload["updated_at"] = now()
                            validate_queue(payload, authenticate=False)
                            atomic_queue(payload)
                        atomic_json(STATUS, {
                            "observed_at": now(), "state": "running",
                            "active_experiment": payload.get("active_experiment"),
                            "order": [entry["experiment_id"] for entry in payload["entries"]],
                            **loop_status,
                        })
            finally:
                os.close(queue_fd)
            if once:
                return
            time.sleep(POLL_SECONDS)
    finally:
        os.close(fd)


def insert_entry(payload: dict[str, Any], entry: dict[str, Any], position: int | None,
                 instruction_hash: str | None) -> None:
    if any(item["experiment_id"] == entry.get("experiment_id") for item in payload["entries"]):
        raise ValueError("experiment_id already exists in queue")
    if position is None:
        index = len(payload["entries"])
        mutation = "append"
    else:
        if not instruction_hash or len(instruction_hash) != 64:
            raise ValueError("priority insertion requires operator instruction hash")
        index = max(0, min(position - 1, len(payload["entries"])))
        mutation = "priority_insert"
    previous = [item["experiment_id"] for item in payload["entries"]]
    payload["entries"].insert(index, entry)
    for offset, item in enumerate(payload["entries"], 1):
        item["position"] = offset
    payload.setdefault("history", []).append({
        "kind": mutation, "at": now(), "experiment_id": entry["experiment_id"],
        "operator_instruction_sha256": instruction_hash,
        "previous_order": previous,
        "new_order": [item["experiment_id"] for item in payload["entries"]],
    })


def enqueue(entry_path: Path, position: int | None, instruction_hash: str | None) -> None:
    recovery_gate()
    lock_fd = acquire_queue_lock()
    try:
        payload = load_queue()
        entry = json.loads(entry_path.read_text())
        try:
            insert_entry(payload, entry, position, instruction_hash)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        payload["updated_at"] = now()
        validate_queue(payload, authenticate=False)
        if entry.get("approval_status") == "approved":
            validate_approval_record(entry)
        atomic_queue(payload)
    finally:
        os.close(lock_fd)


def approve(experiment_id: str, approval_record: Path) -> None:
    recovery_gate(experiment_id if experiment_id in RECOVERY_EXPERIMENTS else None)
    lock_fd = acquire_queue_lock()
    try:
        payload = load_queue()
        matches = [entry for entry in payload["entries"] if entry.get("experiment_id") == experiment_id]
        if len(matches) != 1:
            raise RuntimeError("approval target is not a unique queue entry")
        entry = matches[0]
        if entry.get("approval_status", "pending") == "approved":
            raise RuntimeError("queue entry is already approved")
        entry["approval_evidence"] = {"path": str(approval_record), "sha256": digest_file(approval_record)}
        entry["approval_status"] = "approved"
        record = validate_approval_record(entry)
        entry.setdefault("runtime", {})["approval_text_sha256"] = record["channel_record_sha256"]
        payload.setdefault("history", []).append({
            "kind": "approval_transition", "at": now(), "experiment_id": experiment_id,
            "from": "pending", "to": "approved", "approval_record_sha256": digest_file(approval_record),
        })
        payload["updated_at"] = now()
        validate_queue(payload, authenticate=False)
        atomic_queue(payload)
    finally:
        os.close(lock_fd)


def self_test() -> None:
    global child_queue_entry, launch
    base = {
        "schema_version": 1,
        "document_kind": "operator_approved_experiment_backlog",
        "entries": [],
    }
    validate_queue(base)
    assert WAITING_PREFIXES == ("queued", "preregistered_waiting", "preregistered_held")
    original_child_queue_entry = child_queue_entry
    child_queue_entry = lambda entry: {
        "status": entry["status"], "terminal_notification_status": "delivered"
    }
    true_validator = {"argv": ["/bin/true"], "bindings": [{"path": "/bin/true", "sha256": digest_file(Path("/bin/true"))}]}
    assert not dependencies_satisfied({"dependencies": ["a"]}, [{"experiment_id": "a", "status": "completed"}])
    assert dependencies_satisfied(
        {"dependencies": ["a"]},
        [{"experiment_id": "a", "status": "completed", "runtime": {"completion_validator": true_validator}}],
    )
    assert not dependencies_satisfied({"dependencies": ["a"]}, [{"experiment_id": "a", "status": "failed"}])
    assert dependencies_satisfied({"ordering_dependencies": ["a"]}, [{"experiment_id": "a", "status": "failed"}])
    assert eligibility_blockers({"experiment_id": "held", "approval_status": "pending"}, []) == [
        {"kind": "operator_approval", "experiment_id": "held", "status": "pending"}
    ]
    assert completion_evidence_valid({"runtime": {"completion_validator": true_validator}})
    assert not completion_evidence_valid({"runtime": {"completion_validator": {"argv": ["/bin/false"], "bindings": []}}})
    queue = {"entries": [
        {"experiment_id": "active", "position": 1},
        {"experiment_id": "a", "position": 2},
        {"experiment_id": "b", "position": 3},
    ], "history": []}
    insert_entry(queue, {"experiment_id": "tail"}, None, None)
    assert [item["experiment_id"] for item in queue["entries"]] == ["active", "a", "b", "tail"]
    insert_entry(queue, {"experiment_id": "priority"}, 2, "f" * 64)
    assert [item["experiment_id"] for item in queue["entries"]] == ["active", "priority", "a", "b", "tail"]
    try:
        insert_entry(queue, {"experiment_id": "unauthorized"}, 2, None)
    except ValueError:
        pass
    else:
        raise AssertionError("priority insertion without authorization was accepted")
    original_launch = launch
    launched: list[str] = []
    try:
        child_queue_entry = lambda entry: {
            "status": "completed" if entry["experiment_id"] == "current" else entry["status"],
            "terminal_notification_status": "delivered",
        }
        def fake_launch(entry: dict[str, Any]) -> None:
            launched.append(entry["experiment_id"])
            entry["status"] = "active"
            entry.setdefault("runtime", {})["status_source"] = f"/tmp/{entry['experiment_id']}/current"
            entry["runtime"]["controller_pid"] = os.getpid()
            entry["runtime"]["controller_start_ticks"] = process_start_ticks(os.getpid())
        launch = fake_launch
        handoff = {
            "entries": [
                {"experiment_id": "current", "run_id": "r0", "position": 1, "status": "active", "approval_status": "approved", "dependencies": [], "runtime": {"completion_validator": true_validator}},
                {"experiment_id": "inserted", "run_id": "r1", "position": 2, "status": "queued", "approval_status": "approved", "dependencies": ["current"], "runtime": {}},
                {"experiment_id": "original-next", "run_id": "r2", "position": 3, "status": "queued", "approval_status": "approved", "dependencies": ["inserted"], "runtime": {}},
                {"experiment_id": "tail", "run_id": "r3", "position": 4, "status": "queued", "approval_status": "approved", "dependencies": ["original-next"], "runtime": {}},
            ],
            "active_experiment": {"experiment_id": "current"},
        }
        changed, status = transition_once(handoff)
        assert changed and status["state"] == "running"
        assert launched == ["inserted"]
        assert [item["status"] for item in handoff["entries"]] == ["completed", "active", "queued", "queued"]
        assert [item["experiment_id"] for item in handoff["entries"]] == ["current", "inserted", "original-next", "tail"]

        launched.clear()
        failed_order = {
            "entries": [
                {"experiment_id": "failed", "run_id": "r0", "position": 1, "status": "failed", "approval_status": "approved", "dependencies": [], "runtime": {}},
                {"experiment_id": "ordered-next", "run_id": "r1", "position": 2, "status": "queued", "approval_status": "approved", "dependencies": [], "ordering_dependencies": ["failed"], "runtime": {}},
            ],
            "active_experiment": None,
        }
        changed, _ = transition_once(failed_order)
        assert changed and launched == ["ordered-next"]

        launched.clear()
        scan_past_blocked = {
            "entries": [
                {"experiment_id": "repair", "run_id": "r0", "position": 1, "status": "failed", "approval_status": "approved", "dependencies": [], "runtime": {}},
                {"experiment_id": "caption", "run_id": "r1", "position": 2, "status": "queued", "approval_status": "approved", "dependencies": ["repair"], "ordering_dependencies": ["repair"], "runtime": {}},
                {"experiment_id": "independent", "run_id": "r2", "position": 3, "status": "queued", "approval_status": "approved", "dependencies": [], "runtime": {}},
            ],
            "active_experiment": None,
        }
        changed, _ = transition_once(scan_past_blocked)
        assert changed and launched == ["independent"]
        assert scan_past_blocked["entries"][1]["status"] == "queued"

        launched.clear()
        child_queue_entry = lambda entry: {
            "status": "active", "terminal_notification_status": "not_applicable"
        }
        changed, status = transition_once(scan_past_blocked)
        assert not changed and not launched and status["state"] == "running"
    finally:
        launch = original_launch
        child_queue_entry = original_child_queue_entry
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "atomic.json"
        atomic_json(path, base)
        assert json.loads(path.read_text()) == base
    print("[SELFTEST OK] append/priority gate, dependencies, queue invariants, atomic state")


def _guarded_all_locks(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with recovery_guard():
            return function(*args, **kwargs)
    return wrapped


def _guarded_entry_locks(function):
    @wraps(function)
    def wrapped(entry: dict[str, Any], *args, **kwargs):
        evidence = entry.get("approval_evidence", {})
        approval = Path(str(evidence.get("path", ""))) if evidence.get("path") else None
        experiment_id = str(entry.get("experiment_id"))
        selected = experiment_id if experiment_id in RECOVERY_EXPERIMENTS else None
        with recovery_guard(selected, approval if selected else None):
            return function(entry, *args, **kwargs)
    return wrapped


def _guarded_approval_locks(function):
    @wraps(function)
    def wrapped(experiment_id: str, approval_record: Path, *args, **kwargs):
        selected = experiment_id if experiment_id in RECOVERY_EXPERIMENTS else None
        with recovery_guard(selected, approval_record if selected else None):
            return function(experiment_id, approval_record, *args, **kwargs)
    return wrapped


atomic_json = _guarded_all_locks(atomic_json)
atomic_queue = _guarded_all_locks(atomic_queue)
load_queue = _guarded_all_locks(load_queue)
notify_once = _guarded_all_locks(notify_once)
run = _guarded_all_locks(run)
enqueue = _guarded_all_locks(enqueue)
approve = _guarded_approval_locks(approve)
validate_approval_record = _guarded_entry_locks(validate_approval_record)
completion_evidence_valid = _guarded_entry_locks(completion_evidence_valid)
launch = _guarded_entry_locks(launch)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--once", action="store_true")
    enqueue_parser = sub.add_parser("enqueue")
    enqueue_parser.add_argument("--entry-json", type=Path, required=True)
    enqueue_parser.add_argument("--position", type=int)
    enqueue_parser.add_argument("--operator-instruction-hash")
    approve_parser = sub.add_parser("approve")
    approve_parser.add_argument("--experiment-id", required=True)
    approve_parser.add_argument("--approval-record", type=Path, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "run":
        run(args.once)
    elif args.command == "enqueue":
        enqueue(args.entry_json, args.position, args.operator_instruction_hash)
    elif args.command == "approve":
        approve(args.experiment_id, args.approval_record)
    else:
        self_test()


if __name__ == "__main__":
    main()
