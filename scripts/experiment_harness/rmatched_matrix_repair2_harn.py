#!/usr/bin/env python3
"""Durable HARN for the crash-reconciled Matrix repair2 transaction."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn
from runtime_binding import verify_manifest


ROOT = Path("/home/kojiek/MeanAudio")
NVME_EVAL = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/eval_output")
STATE = Path("/home/kojiek/logs/rmatched_matrix_repair2_harn")
PREREG = ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_repair2_contract.json"
REPAIR = ROOT / "scripts/repair_rmatched_matrix_corrupt_flac_v2.py"
EVIDENCE = STATE / "verified_rng_replay_evidence.json"
REPORT = STATE / "repair_report.json"
FAILED_REPAIR1 = Path("/home/kojiek/logs/rmatched_matrix_repair_harn/generations/gen-000003")
PYTHON = Path("/usr/bin/python3.12")
RUNTIME_MANIFEST = ROOT / "docs/experiments/rmatched_repair2_runtime_manifest.json"
RUNTIME_BINDING = ROOT / "scripts/experiment_harness/runtime_binding.py"
TRANSACTION = NVME_EVAL / ".repair2_state/5xIBQGMjiX4_30.repair2-transaction.json"
STAGE_REPORT = Path("/home/kojiek/logs/rmatched_matrix_nvme_stage_harn/stage_report.json")
STAGE_VALIDATOR = ROOT / "scripts/stage_rmatched_matrix_nvme.py"
STAGE_BUNDLE_VALIDATOR = ROOT / "scripts/validate_rmatched_matrix_nvme_stage_bundle.py"
STAGE_CONTRACT = ROOT / "docs/experiments/rmatched_matrix_nvme_stage_contract.json"
REPAIR1_APPROVAL_SHA256 = "6d93c976c3319491d291d8b829ddd958116b5a01782ae4839ee2b6b737e96a77"
CURRENT_APPROVAL_HASH: str | None = None
BASE_NOTIFY = harn.notify
QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
QUEUE_KEY = QUEUE.parent / "queue_hmac.key"
ARCHIVE = STATE / "source_archive"
EXECUTABLE = REPAIR
CAPABILITIES = STATE / "capabilities"
INIT_FAULT_AFTER: str | None = None
DANGEROUS_ENV = {
    "PYTHONPATH", "PYTHONHOME", "MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE",
    "MEANAUDIO_NOTIFY_DRY_RUN", "MEANAUDIO_REPAIR2_CAPABILITY",
    "MEANAUDIO_REPAIR2_APPROVAL_SHA256",
}
CONTRACT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RECOVERY_VALIDATOR = ROOT / "scripts/dual_authority_recovery_v3.py"
RECOVERY_VALIDATOR_SHA256 = "ffff8ff4bb8b260db378800296d5202c80784094f620a4deaaf0c2b44f466c19"


def _load_recovery_validator() -> dict[str, Any]:
    fd = os.open(RECOVERY_VALIDATOR, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        source = os.pread(fd, info.st_size, 0)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or hashlib.sha256(source).hexdigest() != RECOVERY_VALIDATOR_SHA256):
            raise RuntimeError("dual-authority recovery validator drift")
        namespace: dict[str, Any] = {
            "__name__": "dual_authority_recovery_v3_repair2_bound", "__file__": f"/proc/self/fd/{fd}",
        }
        exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)
        return namespace
    finally:
        os.close(fd)


RECOVERY = _load_recovery_validator()
ACTIVE_RECOVERY_LEASE: Any | None = None
BASE_HARN_ATOMIC_JSON = getattr(harn, "_dual_auth_base_atomic_json", harn.atomic_json)
harn._dual_auth_base_atomic_json = BASE_HARN_ATOMIC_JSON
BASE_HARN_ACQUIRE_LOCK = getattr(harn, "_dual_auth_base_acquire_lock", harn.acquire_lock)
harn._dual_auth_base_acquire_lock = BASE_HARN_ACQUIRE_LOCK


@contextmanager
def recovery_guard(approval_record: Path | None = None):
    global ACTIVE_RECOVERY_LEASE
    with RECOVERY["guarded_action"](
            "rmatched-s1-s2-steps-cfg-matrix-repair2", approval_record) as lease:
        prior = ACTIVE_RECOVERY_LEASE
        ACTIVE_RECOVERY_LEASE = lease
        try:
            yield lease
        finally:
            ACTIVE_RECOVERY_LEASE = prior


def recovery_gate(approval_record: Path | None = None) -> None:
    with recovery_guard(approval_record):
        pass


def gated_harn_atomic_json(path: Path, value: Any) -> None:
    with recovery_guard():
        BASE_HARN_ATOMIC_JSON(path, value)


def guarded_harn_acquire_lock() -> int:
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("Repair2 shared-HARN lock requested outside recovery guard")
    return os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("repair2_controller"))


def configure() -> None:
    harn.atomic_json = gated_harn_atomic_json
    harn.acquire_lock = guarded_harn_acquire_lock
    harn.STATE = STATE
    harn.GENERATIONS = STATE / "generations"
    harn.OUTBOX = STATE / "outbox"
    harn.CURRENT = STATE / "current"
    harn.KEY = STATE / "ledger_hmac.key"
    harn.LOCK = STATE / "controller.lock"
    harn.APPROVAL = STATE / "operator_approval.json"
    harn.PROCESS = STATE / "process_identity.json"
    harn.WATCH_STATUS = STATE / "watch_status.json"
    harn.PENDING_CONTRACT = STATE / "pending_contract.json"
    harn.PENDING_PREFLIGHT = STATE / "pending_preflight.json"
    harn.PENDING_LEDGER = STATE / "pending_ledger.json"
    harn.PENDING_QUEUE = STATE / "pending_queue.json"
    harn.RUNNER = REPAIR
    harn.EXPERIMENT = "rmatched-s1-s2-steps-cfg-matrix-repair2"
    harn.RUN_ID = "run-20260814-seed14159265-musiccaps-repair2"
    harn.KS = (0,)
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.storage_model = storage_model
    harn.run = run
    harn.init = init
    harn.notify = safe_notify


def command_registry() -> dict[str, list[str]]:
    if CURRENT_APPROVAL_HASH is None:
        raise RuntimeError("repair2 exact approval is not bound")
    return {
        "audit_repair": [str(PYTHON), str(EXECUTABLE), "audit"],
        "apply_repair": [str(PYTHON), str(EXECUTABLE), "apply"],
        "rollback_repair": [str(PYTHON), str(EXECUTABLE), "rollback"],
        "reconcile_repair": [str(PYTHON), str(EXECUTABLE), "reconcile"],
        "validate_repair": [
            str(PYTHON), str(EXECUTABLE), "validate-report",
            "--expected-approval-sha256", CURRENT_APPROVAL_HASH,
        ],
    }


def storage_model() -> dict[str, int | str]:
    return {
        "path": "/home/kojiek/nvme_experiment_artifacts", "hard_floor_bytes": 50 * harn.GIB,
        "warning_floor_bytes": 60 * harn.GIB,
        "peak_additional_bytes": 1 * 1024 * 1024,
        "transient_bytes": 1 * 1024 * 1024,
        "recovery_reserve_bytes": 0,
    }


def execution_env(**extra: str) -> dict[str, str]:
    if CURRENT_APPROVAL_HASH is None:
        raise RuntimeError("repair2 exact approval is not bound")
    inherited = {
        key: value for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE"}
        and key not in DANGEROUS_ENV
    }
    return {**inherited, "PATH": CONTRACT_PATH, "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1", **extra}


def safe_notify(*args: Any, **kwargs: Any) -> None:
    recovery_gate()
    removed = {key: os.environ.pop(key) for key in DANGEROUS_ENV if key in os.environ}
    try:
        BASE_NOTIFY(*args, **kwargs)
    finally:
        os.environ.update(removed)


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def load_private_key(path: Path) -> bytes:
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


def ensure_harn_key() -> None:
    recovery_gate()
    if harn.KEY.exists():
        load_private_key(harn.KEY)
        return
    fd = os.open(harn.KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, os.urandom(32))
        os.fsync(fd)
    finally:
        os.close(fd)


def signed(payload: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {
        **unsigned,
        "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest(),
    }


def verify_signed(payload: dict[str, Any], key: bytes, domain: bytes) -> None:
    supplied = payload.get("integrity")
    expected = signed(payload, key, domain)["integrity"]
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated approval document signature is invalid")


def queue_entry_binding(entry: dict[str, Any]) -> str:
    runtime = entry.get("runtime", {})
    descriptor = {
        "experiment_id": entry.get("experiment_id"), "run_id": entry.get("run_id"),
        "contract": entry.get("contract"), "contract_sha256": entry.get("contract_sha256"),
        "dependencies": entry.get("dependencies", []),
        "ordering_dependencies": entry.get("ordering_dependencies", []),
        "controller": runtime.get("controller"), "controller_sha256": runtime.get("controller_sha256"),
        "status_source": runtime.get("status_source"),
        "pending_status_source": runtime.get("pending_status_source"),
    }
    for key in ("bundle_validator", "completion_validator", "approval_cli", "import_bindings", "child_ledger_key", "transitive_runtime"):
        if key in runtime:
            descriptor[key] = runtime[key]
    return hashlib.sha256(canonical(descriptor)).hexdigest()


def verify_queue_approval(record_path: Path, expected_entry_binding: str) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_key = load_private_key(QUEUE_KEY)
    queue = json.loads(QUEUE.read_text())
    verify_signed(queue, queue_key, b"meanaudio-operator-queue-v1\0")
    matches = [entry for entry in queue.get("entries", []) if entry.get("experiment_id") == harn.EXPERIMENT]
    if len(matches) != 1:
        raise RuntimeError("Repair2 has no unique approved queue entry")
    entry = matches[0]
    binding = queue_entry_binding(entry)
    if binding != expected_entry_binding:
        raise RuntimeError("Repair2 queue entry does not match the approved exact binding")
    if entry.get("run_id") != harn.RUN_ID or entry.get("approval_status") != "approved":
        raise RuntimeError("Repair2 queue entry is not approved for this exact run")
    runtime_descriptor = entry.get("runtime", {}).get("transitive_runtime", {})
    prereg_runtime = json.loads(PREREG.read_text()).get("transitive_runtime", {})
    if (runtime_descriptor.get("manifest") != prereg_runtime.get("manifest")
            or runtime_descriptor.get("manifest_sha256") != prereg_runtime.get("manifest_sha256")
            or runtime_descriptor.get("verifier", {}).get("path") != prereg_runtime.get("verifier")
            or runtime_descriptor.get("verifier", {}).get("sha256") != prereg_runtime.get("verifier_sha256")):
        raise RuntimeError("Repair2 queue entry lacks the approval-bound transitive runtime")
    evidence = entry.get("approval_evidence", {})
    if Path(str(evidence.get("path", ""))) != record_path or evidence.get("sha256") != harn.digest_file(record_path):
        raise RuntimeError("Repair2 queue entry does not bind the exact approval record")
    record = json.loads(record_path.read_text())
    verify_signed(record, queue_key, b"meanaudio-queue-approval-v1\0")
    expected = {
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "contract_sha256": harn.digest_file(PREREG),
        "controller_sha256": harn.digest_file(Path(__file__)),
        "queue_entry_binding_sha256": binding,
    }
    mismatches = {key: (value, record.get(key)) for key, value in expected.items() if record.get(key) != value}
    approval_hash = record.get("channel_record_sha256")
    if (mismatches or not isinstance(approval_hash, str) or len(approval_hash) != 64
            or approval_hash == REPAIR1_APPROVAL_SHA256):
        raise RuntimeError(f"Repair2 exact approval record is invalid: {mismatches}")
    return entry, record


def archive_executable() -> Path:
    recovery_gate()
    global EXECUTABLE
    digest = harn.digest_file(REPAIR)
    directory = ARCHIVE / digest
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = directory / REPAIR.name
    if destination.exists():
        if harn.digest_file(destination) != digest:
            raise RuntimeError("archived Repair2 executable drift")
    else:
        with REPAIR.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(destination, 0o500)
    EXECUTABLE = destination
    return destination


def open_verified_fd(path: Path, expected_sha256: str) -> int:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"verified executable is not a regular file: {path}")
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(fd, 8 << 20, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"verified executable descriptor drift: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def run_repair_action(action_id: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[Any]:
    recovery_gate()
    approval = read_approval_state()
    fd = open_verified_fd(EXECUTABLE, approval["archived_executable_sha256"])
    try:
        registered = command_registry()[action_id]
        return subprocess.run(
            [registered[0], "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
             f"/proc/self/fd/{fd}", *registered[2:]], cwd=ROOT,
            env=env, pass_fds=(fd,),
        )
    finally:
        os.close(fd)


def read_approval_state() -> dict[str, Any]:
    global EXECUTABLE
    approval = json.loads(harn.APPROVAL.read_text())
    verify_signed(approval, load_private_key(harn.KEY), b"meanaudio-repair2-approval-state-v1\0")
    if approval.get("experiment_id") != harn.EXPERIMENT or approval.get("run_id") != harn.RUN_ID:
        raise RuntimeError("Repair2 local approval state is wrong-run")
    archived = Path(str(approval.get("archived_executable", "")))
    if (not archived.is_file() or archived.is_symlink()
            or harn.digest_file(archived) != approval.get("archived_executable_sha256")):
        raise RuntimeError("Repair2 archived executable state is missing or drifted")
    EXECUTABLE = archived
    return approval


def write_approval_state(approval: dict[str, Any]) -> None:
    recovery_gate(Path(str(approval.get("approval_record_path", ""))) if approval.get("approval_record_path") else None)
    harn.atomic_json(
        harn.APPROVAL,
        signed(approval, load_private_key(harn.KEY), b"meanaudio-repair2-approval-state-v1\0"),
    )


def binding_descriptors(contract: dict[str, Any]) -> list[dict[str, str]]:
    paths: dict[str, str] = {
        item["path"]: item["sha256"] for item in contract["corpus"]["source_artifacts"]
    }
    for path in (
        ROOT / "scripts/experiment_harness/rmatched_matrix_repair2_harn.py",
        EXECUTABLE, ROOT / "AGENTS.md",
        ROOT / "docs/experiments/experiment_notification_policy.md",
        ROOT / "docs/experiments/watcher_policy.md",
    ):
        paths[str(path)] = harn.digest_file(path)
    for path in sorted((ROOT / "docs/experiments/schemas").glob("*.json")):
        paths[str(path)] = harn.digest_file(path)
    return [{"path": path, "sha256": digest} for path, digest in sorted(paths.items())]


def verify_preregistered_descriptors() -> None:
    prereg = json.loads(PREREG.read_text())
    if prereg.get("experiment_id") != harn.EXPERIMENT or prereg.get("run_id") != harn.RUN_ID:
        raise RuntimeError("Repair2 preregistration identity drift")
    implementation = prereg.get("implementation", {})
    descriptors = (
        ("repair_script", "repair_script_sha256"), ("harn", "harn_sha256"),
        ("queue_controller", "queue_controller_sha256"),
        ("shared_harn", "shared_harn_sha256"), ("validator", "validator_sha256"),
        ("notifier", "notifier_sha256"), ("recovery_validator", "recovery_validator_sha256"),
    )
    for path_key, hash_key in descriptors:
        path = Path(str(implementation.get(path_key, "")))
        if not path.is_file() or harn.digest_file(path) != implementation.get(hash_key):
            raise RuntimeError(f"Repair2 preregistered descriptor drift: {path_key}")
    acceptance = implementation.get("acceptance_tests", {})
    for path_key, hash_key in (
        ("filesystem_transaction", "filesystem_transaction_sha256"),
        ("harn_and_queue_recovery", "harn_and_queue_recovery_sha256"),
        ("dual_authority_recovery", "dual_authority_recovery_sha256"),
    ):
        path = Path(str(acceptance.get(path_key, "")))
        if not path.is_file() or harn.digest_file(path) != acceptance.get(hash_key):
            raise RuntimeError(f"Repair2 preregistered test descriptor drift: {path_key}")
    evidence = prereg.get("verified_rng_replay", {})
    evidence_path = Path(str(evidence.get("evidence", "")))
    if not evidence_path.is_file() or harn.digest_file(evidence_path) != evidence.get("evidence_sha256"):
        raise RuntimeError("Repair2 preregistered evidence descriptor drift")
    runtime = prereg.get("transitive_runtime", {})
    if (Path(str(runtime.get("manifest", ""))) != RUNTIME_MANIFEST
            or harn.digest_file(RUNTIME_MANIFEST) != runtime.get("manifest_sha256")
            or Path(str(runtime.get("verifier", ""))) != RUNTIME_BINDING
            or harn.digest_file(RUNTIME_BINDING) != runtime.get("verifier_sha256")):
        raise RuntimeError("Repair2 transitive runtime descriptor drift")
    verify_manifest(
        RUNTIME_MANIFEST, runtime["manifest_sha256"],
        {"system_python", "system_stdlib", "system_native_runtime", "stage_bundle_validator",
         "recovery_validator"},
    )


def verify_runtime_bindings(contract: dict[str, Any]) -> list[dict[str, str]]:
    verify_preregistered_descriptors()
    if contract["bindings"]["runtime_sha256"] != harn.digest_file(Path(__file__)):
        raise RuntimeError("Repair2 HARN executable drift")
    if contract["bindings"]["policy_bundle_sha256"] != harn.policy_hash():
        raise RuntimeError("Repair2 policy binding drift")
    if contract["bindings"]["schema_bundle_sha256"] != harn.schema_hash():
        raise RuntimeError("Repair2 schema binding drift")
    if contract["bindings"]["command_set_sha256"] != harn.digest_bytes(harn.canonical(command_registry())):
        raise RuntimeError("Repair2 command binding drift")
    bindings = binding_descriptors(contract)
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_file() or harn.digest_file(path) != binding["sha256"]:
            raise RuntimeError(f"Repair2 preregistered input drift: {path}")
    return bindings


def verify_approval_authority(approval: dict[str, Any]) -> None:
    record_path = Path(approval["approval_record_path"])
    if harn.digest_file(record_path) != approval["approval_record_sha256"]:
        raise RuntimeError("Repair2 exact approval record drift")
    verify_queue_approval(record_path, approval["queue_entry_binding_sha256"])


def reserve_approval(approval: dict[str, Any]) -> dict[str, Any]:
    if approval.get("state") == "consumed":
        raise RuntimeError("Repair2 approval has already been consumed")
    approval = {**approval, "state": "reserved", "reservation": {
        "run_id": harn.RUN_ID, "controller_pid": os.getpid(),
        "controller_start_ticks": process_start_ticks(os.getpid()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "reserved_at": harn.now(),
    }}
    write_approval_state(approval)
    return read_approval_state()


def process_start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().split()[21]


def issue_capability(action: str, contract: dict[str, Any], approval: dict[str, Any]) -> Path:
    recovery_gate(Path(str(approval.get("approval_record_path", ""))))
    bindings = verify_runtime_bindings(contract)
    verify_approval_authority(approval)
    current = read_approval_state()
    if current.get("state") != "reserved":
        raise RuntimeError("Repair2 approval is not reserved for capability issue")
    CAPABILITIES.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = CAPABILITIES / f"{action}-{time.time_ns()}-{os.getpid()}.json"
    payload = {
        "document_kind": "repair2_write_capability", "schema_version": 1,
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "action": action, "status": "authorized", "nonce": os.urandom(32).hex(),
        "issued_at": harn.now(), "parent_pid": os.getpid(),
        "parent_start_ticks": process_start_ticks(os.getpid()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "harn_lock": str(harn.LOCK), "executable_sha256": harn.digest_file(EXECUTABLE),
        "approval_state_sha256": harn.digest_file(harn.APPROVAL),
        "approval_record_sha256": current["approval_record_sha256"],
        "bindings": bindings,
        "writable_paths": [
            str(NVME_EVAL / "rmatched_s1_s2_steps_cfg_matrix_seed14159265/s2_mf25_cfg0p5/audio"),
            str(NVME_EVAL / "metrics/rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5"),
            str(NVME_EVAL / ".repair2_state"),
        ],
    }
    harn.atomic_json(path, signed(payload, load_private_key(harn.KEY), b"meanaudio-repair2-capability-v1\0"))
    os.chmod(path, 0o600)
    return path


def validate_stage_dependency() -> None:
    recovery_gate()
    prereg = json.loads(PREREG.read_text())
    matches = [item for item in prereg.get("scientific_dependencies", [])
               if item.get("experiment_id") == "rmatched-s1-s2-steps-cfg-matrix-nvme-stage"]
    if len(matches) != 1:
        raise RuntimeError("Repair2 lacks one exact NVMe stage scientific dependency")
    dependency = matches[0]
    if (dependency.get("run_id") != "run-20260815-rmatched-matrix-nvme-stage1"
            or dependency.get("required_state") != "completed"
            or Path(str(dependency.get("contract", ""))) != STAGE_CONTRACT
            or dependency.get("contract_sha256") != harn.digest_file(STAGE_CONTRACT)
            or Path(str(dependency.get("authenticated_bundle_validator", ""))) != STAGE_BUNDLE_VALIDATOR
            or dependency.get("authenticated_bundle_validator_sha256") != harn.digest_file(STAGE_BUNDLE_VALIDATOR)):
        raise RuntimeError("Repair2 NVMe stage dependency descriptor drift")
    completed = subprocess.run(
        [str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
         str(STAGE_BUNDLE_VALIDATOR)],
        cwd="/", env=execution_env(), text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"NVMe stage authenticated completed dependency invalid: {completed.stderr[-1000:]}")


def make_contract() -> dict[str, Any]:
    validate_stage_dependency()
    commands = command_registry()
    zero = "0" * 64
    source_paths = [
        PREREG, REPAIR, EXECUTABLE, EVIDENCE, STAGE_REPORT, STAGE_VALIDATOR,
        STAGE_BUNDLE_VALIDATOR, STAGE_CONTRACT,
        FAILED_REPAIR1 / "contract.json", FAILED_REPAIR1 / "ledger.json", FAILED_REPAIR1 / "queue.json",
        ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_contract.json",
        ROOT / "scripts/experiment_harness/qwen_s2q_k_mf25_harn.py",
        RUNTIME_MANIFEST, RUNTIME_BINDING,
        RECOVERY_VALIDATOR,
        ROOT / "scripts/validate_experiment_harness_documents.py",
        ROOT / "scripts/notify_experiment_webhook.py",
    ]
    if harn.APPROVAL.is_file():
        approval = read_approval_state()
        source_paths.append(Path(approval["approval_record_path"]))
    sources = [{"path": str(path), "sha256": harn.digest_file(path)} for path in source_paths]
    envelope = {
        "envelope_sha256": harn.digest_file(PREREG),
        "writable_paths": [
            str(STATE),
            str(NVME_EVAL / "rmatched_s1_s2_steps_cfg_matrix_seed14159265/s2_mf25_cfg0p5/audio"),
            str(NVME_EVAL / "metrics/rmatched_s1_s2_steps_cfg_matrix_seed14159265_s2_mf25_cfg0p5"),
            str(NVME_EVAL / ".repair2_state"),
        ],
        "test_action_ids": ["audit_repair", "reconcile_repair", "validate_repair"],
        "apply_action_id": "apply_repair", "rollback_action_id": "rollback_repair",
        "resume_action_id": "apply_repair",
        "allowed_process_identities": ["rmatched_matrix_repair2_controller"],
        "reviewer_roles": ["responsible_operator"],
        "budgets": {"max_model_calls": 0, "max_wall_seconds": 600, "max_transient_retries": 0, "max_cost_units": 1},
        "operator_required_conditions": [
            "new_exact_approval", "verified_rng_replay", "same_filesystem_quarantine",
            "validated_nvme_stage_report", "crash_reconciled_transaction", "actual_nvme_storage_gate",
        ],
    }
    return {
        "document_kind": "experiment_contract", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "bindings": {
            "policy_bundle_sha256": harn.policy_hash(), "schema_bundle_sha256": harn.schema_hash(),
            "runtime_sha256": harn.digest_file(Path(__file__)),
            "command_set_sha256": harn.digest_bytes(harn.canonical(commands)),
        },
        "approval_requirement": {"required": True, "responsible_role": "operator", "trusted_channels": ["operator_console"]},
        "corpus": {"kind": "non_generated", "source_artifacts": sources},
        "repair": {"enabled": True, "envelope": envelope},
        "phases": [{
            "phase_id": "repair_audio", "action_id": "apply_repair", "resume_action_id": "apply_repair",
            "input_artifacts": sources, "output_paths": [str(REPORT)],
            "completion_evidence": [{"path": str(REPORT), "sha256": zero}],
        }],
        "filesystems": [storage_model()],
        "commands": [{
            "action_id": action_id, "argv": argv, "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        } for action_id, argv in commands.items()],
        "required_preflight_checks": [
            "approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
            "no_duplicate", "policy_bound", "storage_policy_1.25",
        ],
        "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def init(approval_record: Path, queue_entry_sha256: str) -> None:
    """Initialize only from an authenticated, exact-bound approved queue entry."""
    global CURRENT_APPROVAL_HASH
    recovery_gate(approval_record)
    lock_fd = harn.acquire_lock()
    try:
        for directory in (STATE, harn.GENERATIONS, harn.OUTBOX, CAPABILITIES, ARCHIVE):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        ensure_harn_key()
        _entry, record = verify_queue_approval(approval_record, queue_entry_sha256)
        CURRENT_APPROVAL_HASH = record["channel_record_sha256"]
        if CURRENT_APPROVAL_HASH == REPAIR1_APPROVAL_SHA256:
            raise RuntimeError("repair1 authorization cannot initialize repair2")
        verify_preregistered_descriptors()
        archive_executable()

        if harn.APPROVAL.exists():
            approval = read_approval_state()
            if approval.get("state") == "consumed":
                raise RuntimeError("Repair2 approval is already consumed")
            if (approval.get("approval_record_sha256") != harn.digest_file(approval_record)
                    or approval.get("queue_entry_binding_sha256") != queue_entry_sha256):
                raise RuntimeError("Repair2 initialization conflicts with existing approval state")
        else:
            write_approval_state({
                "document_kind": "repair2_approval_state", "schema_version": 1,
                "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
                "state": "approved", "issued_at": harn.now(),
                "approval_record_path": str(approval_record),
                "approval_record_sha256": harn.digest_file(approval_record),
                "queue_entry_binding_sha256": queue_entry_sha256,
                "channel_record_sha256": CURRENT_APPROVAL_HASH,
                "contract_runtime_sha256": harn.digest_file(Path(__file__)),
                "archived_executable": str(EXECUTABLE),
                "archived_executable_sha256": harn.digest_file(EXECUTABLE),
            })
        contract = make_contract()
        verify_runtime_bindings(contract)
        if INIT_FAULT_AFTER == "approval_written":
            raise ChildProcessError("injected crash after approval write")
        if harn.CURRENT.exists() or harn.PENDING_CONTRACT.exists():
            print("[INIT OK] existing Repair2 state verified and recoverable")
            return

        conflicts = harn.blocking_gpu_processes()
        preflight = harn.make_preflight(contract, CURRENT_APPROVAL_HASH, not conflicts)
        ledger = {
            "document_kind": "event_ledger", "schema_version": "1.0.0",
            "schema_bundle_id": "harn-schema-v1", "experiment_id": harn.EXPERIMENT,
            "run_id": harn.RUN_ID, "bindings": {
                "contract_raw_sha256": "0" * 64, "preflight_report_raw_sha256": "0" * 64,
                "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"],
            }, "events": [],
        }
        harn.append_event(ledger, "contract_registered")
        held = bool(conflicts) or preflight["derived_verdict"] != "pass"
        if held:
            harn.append_event(ledger, "queue_hold", verdict="fail", phase="resource_wait")
            harn.atomic_json(harn.PENDING_CONTRACT, contract)
            harn.atomic_json(harn.PENDING_PREFLIGHT, preflight)
            harn.atomic_json(harn.PENDING_LEDGER, ledger)
            harn.atomic_json(harn.PENDING_QUEUE, {
                "schema_version": 1, "status": "held", "reason": "preflight_hold",
                "updated_at": harn.now(), "next_action": "repeat exact-bound mutable preflight",
            })
        else:
            harn.append_event(ledger, "preflight_passed", verdict="pass")
            recovery_gate(approval_record)
            harn.write_generation(contract, preflight, ledger, "ready")
        if INIT_FAULT_AFTER == "harn_state_written":
            raise ChildProcessError("injected crash after HARN state write")
        location = str(harn.PENDING_QUEUE) if held else harn.CURRENT.read_text().strip()
        print(f"[INIT OK] status={'held' if held else 'ready'} state={location}")
    finally:
        os.close(lock_fd)


def recoverable_transaction(contract: dict[str, Any], approval: dict[str, Any]) -> bool:
    recovery_gate(Path(str(approval.get("approval_record_path", ""))))
    if not TRANSACTION.is_file():
        return False
    capability = issue_capability("reconcile", contract, approval)
    completed = run_repair_action(
        "reconcile_repair",
        env=execution_env(MEANAUDIO_REPAIR2_CAPABILITY=str(capability)),
    )
    return completed.returncode == 0


def wait_for_complete_preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global CURRENT_APPROVAL_HASH
    approval = read_approval_state()
    recovery_gate(Path(str(approval.get("approval_record_path", ""))))
    CURRENT_APPROVAL_HASH = approval["channel_record_sha256"]
    verify_approval_authority(approval)
    if harn.CURRENT.exists():
        values = harn.load_current()
        verify_runtime_bindings(values[0])
        return values
    contract = json.loads(harn.PENDING_CONTRACT.read_text())
    ledger = json.loads(harn.PENDING_LEDGER.read_text())
    while True:
        verify_runtime_bindings(contract)
        verify_approval_authority(approval)
        conflicts = harn.blocking_gpu_processes()
        preflight = harn.make_preflight(contract, approval["channel_record_sha256"], not conflicts)
        if preflight["derived_verdict"] == "pass":
            harn.append_event(ledger, "preflight_passed", verdict="pass")
            recovery_gate(Path(str(approval.get("approval_record_path", ""))))
            harn.write_generation(contract, preflight, ledger, "ready")
            return contract, preflight, ledger
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.atomic_json(harn.WATCH_STATUS, {
            "observed_at": harn.now(), "status": "held", "reason": "preflight_hold",
            "failed_checks": [item["check_id"] for item in preflight["checks"] if item["verdict"] != "pass"],
            "storage": harn.storage_check(), "blocking_gpu_processes": conflicts, "assigned_resource": None,
        })
        time.sleep(60)


def run() -> None:
    global CURRENT_APPROVAL_HASH
    recovery_gate()
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = wait_for_complete_preflight()
        approval = read_approval_state()
        verify_approval_authority(approval)
        CURRENT_APPROVAL_HASH = approval["channel_record_sha256"]
        already_started = any(event["event_kind"] == "experiment_started" for event in ledger["events"])
        approval = reserve_approval(approval)
        if INIT_FAULT_AFTER == "approval_reserved":
            raise ChildProcessError("injected crash after approval reservation")
        if not already_started:
            harn.append_event(ledger, "resources_acquired", phase="repair_audio")
            started = harn.append_event(ledger, "experiment_started", phase="repair_audio", notification="pending")
            harn.notify("start", "Started crash-reconciled installation of the verified Matrix replacement.")
            harn.append_event(ledger, "notification_delivery", relation=started, phase="repair_audio", notification="delivered")
            recovery_gate(Path(str(approval.get("approval_record_path", ""))))
            harn.write_generation(contract, preflight, ledger, "active")
        actions = ("apply_repair",) if TRANSACTION.is_file() else ("audit_repair", "apply_repair")
        for action in actions:
            verify_runtime_bindings(contract)
            verify_approval_authority(approval)
            env = execution_env()
            if action == "apply_repair":
                capability = issue_capability("apply", contract, approval)
                env["MEANAUDIO_REPAIR2_CAPABILITY"] = str(capability)
            completed = run_repair_action(action, env=env)
            if completed.returncode:
                if action == "apply_repair" and recoverable_transaction(contract, approval):
                    harn.append_event(ledger, "queue_hold", verdict="fail", phase="repair_audio")
                    recovery_gate(Path(str(approval.get("approval_record_path", ""))))
                    harn.write_generation(contract, preflight, ledger, "held")
                    raise ChildProcessError(f"recoverable repair2 crash with exit {completed.returncode}")
                raise RuntimeError(f"{action} failed with exit {completed.returncode}")
        verify_runtime_bindings(contract)
        verify_approval_authority(approval)
        validated = run_repair_action("validate_repair", env=execution_env())
        if validated.returncode:
            raise RuntimeError("repair2 report did not validate")
        gate = harn.append_event(ledger, "gate_result", verdict="pass", phase="repair_audio", notification="pending")
        harn.notify("repair_complete", "Matrix repair2 completed with one changed audio artifact.", report=REPORT)
        harn.append_event(ledger, "notification_delivery", relation=gate, phase="repair_audio", notification="delivered")
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.write_generation(contract, preflight, ledger, "active")
        approval = {**read_approval_state(), "state": "consumed", "consumed_at": harn.now()}
        write_approval_state(approval)
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.terminal(contract, preflight, ledger, True, "Matrix repair2 completed; continuation remains separately storage-gated.")
    except BaseException as exc:
        try:
            if isinstance(exc, ChildProcessError):
                raise
            if harn.CURRENT.exists():
                contract, preflight, ledger = harn.load_current()
                terminal = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
                if not any(event["event_kind"] in terminal for event in ledger["events"]):
                    recovery_gate()
                    harn.terminal(contract, preflight, ledger, False, f"Repair2 failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


def _guarded_repair2(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with recovery_guard():
            return function(*args, **kwargs)
    return wrapped


def _guarded_repair2_init(function):
    @wraps(function)
    def wrapped(approval_record: Path, *args, **kwargs):
        with recovery_guard(approval_record):
            return function(approval_record, *args, **kwargs)
    return wrapped


def watch(once: bool) -> None:
    harn.watch(once)


safe_notify = _guarded_repair2(safe_notify)
ensure_harn_key = _guarded_repair2(ensure_harn_key)
archive_executable = _guarded_repair2(archive_executable)
run_repair_action = _guarded_repair2(run_repair_action)
write_approval_state = _guarded_repair2(write_approval_state)
issue_capability = _guarded_repair2(issue_capability)
validate_stage_dependency = _guarded_repair2(validate_stage_dependency)
recoverable_transaction = _guarded_repair2(recoverable_transaction)
wait_for_complete_preflight = _guarded_repair2(wait_for_complete_preflight)
run = _guarded_repair2(run)
watch = _guarded_repair2(watch)
init = _guarded_repair2_init(init)


configure()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--approval-record", type=Path, required=True)
    init_parser.add_argument("--queue-entry-sha256", required=True)
    sub.add_parser("run")
    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    approval_record = args.approval_record if args.command == "init" else None
    with recovery_guard(approval_record):
        if args.command == "init":
            init(args.approval_record, args.queue_entry_sha256)
        elif args.command == "run":
            harn.atomic_json(harn.PROCESS, {
                "controller_pid": os.getpid(), "controller_start_ticks": process_start_ticks(os.getpid()),
                "started_at": harn.now(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            })
            run()
        else:
            watch(args.once)


if __name__ == "__main__":
    main()
