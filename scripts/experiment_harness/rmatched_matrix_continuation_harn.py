#!/usr/bin/env python3
"""Storage-correct durable HARN for continuing the repaired R-Matched matrix."""
from __future__ import annotations

import argparse
import json
import hashlib
import hmac
import os
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
PYTHON = Path("/usr/bin/python3.12")
STATE = Path("/home/kojiek/logs/rmatched_matrix_continuation_harn")
PREREG = ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_continuation_contract.json"
RUNNER = ROOT / "scripts/eval/eval_rmatched_s1_s2_steps_cfg_matrix_continuation.sh"
REPAIR_REPORT = Path("/home/kojiek/logs/rmatched_matrix_repair2_harn/repair_report.json")
REPAIR_APPROVAL = Path("/home/kojiek/logs/rmatched_matrix_repair2_harn/operator_approval.json")
REPAIR_VALIDATOR = ROOT / "scripts/repair_rmatched_matrix_corrupt_flac_v2.py"
REPAIR_HARN_KEY = REPAIR_APPROVAL.parent / "ledger_hmac.key"
QUEUE_KEY = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue_hmac.key")
RUNTIME_MANIFEST = ROOT / "docs/experiments/rmatched_continuation_runtime_manifest.json"
RUNTIME_BINDING = ROOT / "scripts/experiment_harness/runtime_binding.py"
ISOLATED_BOOTSTRAP = ROOT / "scripts/experiment_harness/isolated_python_bootstrap.py"
MEANAUDIO_INIT = ROOT / "meanaudio/__init__.py"
S1 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage1_400000/phase8_qwen_caption10s_multisent_noq_full_stage1_400000_ema_final.pth"
S2 = ROOT / "exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
REPORT = Path("/home/kojiek/logs/rmatched_s1_s2_steps_cfg_matrix_seed14159265_REPORT.json")
QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
EXPECTED_CELLS = {
    "s1_fm1_cfg0p5", "s1_fm1_cfg4p5", "s1_fm25_cfg0p5", "s1_fm25_cfg4p5",
    "s2_mf1_cfg0p5", "s2_mf1_cfg4p5", "s2_mf25_cfg0p5", "s2_mf25_cfg4p5",
}
DANGEROUS_ENV = {"PYTHONPATH", "PYTHONHOME", "MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE", "MEANAUDIO_NOTIFY_DRY_RUN"}
CONTRACT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
BASE_NOTIFY = harn.notify
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
            "__name__": "dual_authority_recovery_v3_continuation_bound", "__file__": f"/proc/self/fd/{fd}",
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
            "rmatched-s1-s2-steps-cfg-matrix-continuation", approval_record) as lease:
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
        raise RuntimeError("continuation shared-HARN lock requested outside recovery guard")
    return os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("continuation_controller"))


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


def verify_signed(payload: dict[str, Any], key: bytes, domain: bytes) -> None:
    supplied = payload.get("integrity")
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated Repair2 dependency signature is invalid")


def signed(payload: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {**unsigned, "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()}


def entry_binding(entry: dict[str, Any]) -> str:
    runtime = entry.get("runtime", {})
    descriptor: dict[str, Any] = {
        "experiment_id": entry.get("experiment_id"), "run_id": entry.get("run_id"),
        "contract": entry.get("contract"), "contract_sha256": entry.get("contract_sha256"),
        "dependencies": entry.get("dependencies", []), "ordering_dependencies": entry.get("ordering_dependencies", []),
        "controller": runtime.get("controller"), "controller_sha256": runtime.get("controller_sha256"),
        "status_source": runtime.get("status_source"), "pending_status_source": runtime.get("pending_status_source"),
    }
    for key in ("bundle_validator", "completion_validator", "approval_cli", "import_bindings", "child_ledger_key", "transitive_runtime"):
        if key in runtime:
            descriptor[key] = runtime[key]
    return hashlib.sha256(canonical(descriptor)).hexdigest()


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


def verify_queue_approval(record_path: Path, expected_binding: str) -> dict[str, Any]:
    key = load_private_key(QUEUE_KEY)
    queue = json.loads(QUEUE.read_text())
    verify_signed(queue, key, b"meanaudio-operator-queue-v1\0")
    matches = [item for item in queue.get("entries", []) if item.get("experiment_id") == harn.EXPERIMENT]
    if len(matches) != 1:
        raise RuntimeError("continuation has no unique queue entry")
    entry = matches[0]
    binding = entry_binding(entry)
    evidence = entry.get("approval_evidence", {})
    if (binding != expected_binding or entry.get("run_id") != harn.RUN_ID
            or entry.get("approval_status") != "approved"
            or Path(str(evidence.get("path", ""))) != record_path
            or evidence.get("sha256") != harn.digest_file(record_path)):
        raise RuntimeError("continuation queue approval binding mismatch")
    runtime_descriptor = entry.get("runtime", {}).get("transitive_runtime", {})
    prereg_runtime = json.loads(PREREG.read_text()).get("transitive_runtime", {})
    if (runtime_descriptor.get("manifest") != prereg_runtime.get("manifest")
            or runtime_descriptor.get("manifest_sha256") != prereg_runtime.get("manifest_sha256")
            or runtime_descriptor.get("verifier", {}).get("path") != prereg_runtime.get("verifier")
            or runtime_descriptor.get("verifier", {}).get("sha256") != prereg_runtime.get("verifier_sha256")):
        raise RuntimeError("continuation queue lacks approval-bound transitive runtime")
    record = json.loads(record_path.read_text())
    verify_signed(record, key, b"meanaudio-queue-approval-v1\0")
    expected = {
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
        "contract_sha256": harn.digest_file(PREREG), "controller_sha256": harn.digest_file(Path(__file__)),
        "queue_entry_binding_sha256": binding,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise RuntimeError("continuation approval record is unsigned, wrong-run, or drifted")
    approval_hash = record.get("channel_record_sha256")
    if not isinstance(approval_hash, str) or len(approval_hash) != 64:
        raise RuntimeError("continuation authenticated channel record is invalid")
    return record


def read_approval_state() -> dict[str, Any]:
    value = json.loads(harn.APPROVAL.read_text())
    verify_signed(value, load_private_key(harn.KEY), b"meanaudio-continuation-approval-state-v1\0")
    if value.get("experiment_id") != harn.EXPERIMENT or value.get("run_id") != harn.RUN_ID:
        raise RuntimeError("continuation approval state is wrong-run")
    return value


def write_approval_state(value: dict[str, Any]) -> None:
    recovery_gate(Path(str(value.get("approval_record_path", ""))) if value.get("approval_record_path") else None)
    harn.atomic_json(harn.APPROVAL, signed(value, load_private_key(harn.KEY), b"meanaudio-continuation-approval-state-v1\0"))


def verify_continuation_authority(value: dict[str, Any]) -> None:
    record = Path(value["approval_record_path"])
    if harn.digest_file(record) != value["approval_record_sha256"]:
        raise RuntimeError("continuation approval record drift")
    verify_queue_approval(record, value["queue_entry_binding_sha256"])


def reserve_approval(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("state") == "consumed":
        raise RuntimeError("continuation approval is consumed")
    value = {**value, "state": "reserved", "reservation": {
        "run_id": harn.RUN_ID, "controller_pid": os.getpid(),
        "controller_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "reserved_at": harn.now(),
    }}
    write_approval_state(value)
    return read_approval_state()


def safe_env() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE"}
        and key not in DANGEROUS_ENV
    } | {"PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1"}


def safe_notify(*args: Any, **kwargs: Any) -> None:
    recovery_gate()
    removed = {key: os.environ.pop(key) for key in DANGEROUS_ENV if key in os.environ}
    try:
        BASE_NOTIFY(*args, **kwargs)
    finally:
        os.environ.update(removed)


def configure() -> None:
    harn.atomic_json = gated_harn_atomic_json
    harn.acquire_lock = guarded_harn_acquire_lock
    for name, value in {
        "STATE": STATE, "GENERATIONS": STATE / "generations", "OUTBOX": STATE / "outbox",
        "CURRENT": STATE / "current", "KEY": STATE / "ledger_hmac.key", "LOCK": STATE / "controller.lock",
        "APPROVAL": STATE / "operator_approval.json", "PROCESS": STATE / "process_identity.json",
        "WATCH_STATUS": STATE / "watch_status.json", "PENDING_CONTRACT": STATE / "pending_contract.json",
        "PENDING_PREFLIGHT": STATE / "pending_preflight.json", "PENDING_LEDGER": STATE / "pending_ledger.json",
        "PENDING_QUEUE": STATE / "pending_queue.json",
    }.items():
        setattr(harn, name, value)
    harn.RUNNER = RUNNER
    harn.EXPERIMENT = "rmatched-s1-s2-steps-cfg-matrix-continuation"
    harn.RUN_ID = "run-20260814-seed14159265-musiccaps-continuation1"
    harn.KS = (0,)
    harn.command_registry = command_registry
    harn.make_contract = make_contract
    harn.storage_model = storage_model
    harn.run = run
    harn.init = init
    harn.notify = safe_notify


def command_registry() -> dict[str, list[str]]:
    return {"resume_matrix": ["/bin/bash", str(RUNNER)]}


def storage_model() -> dict[str, int | str]:
    return {
        "path": "/home/kojiek/nvme_experiment_artifacts", "hard_floor_bytes": 150 * harn.GIB,
        "warning_floor_bytes": 180 * harn.GIB, "peak_additional_bytes": 40 * harn.GIB,
        "transient_bytes": 15 * harn.GIB, "recovery_reserve_bytes": 10 * harn.GIB,
    }


def make_contract() -> dict[str, Any]:
    validate_repair_dependency()
    commands = command_registry()
    zero = "0" * 64
    source_paths = [
        PREREG, RUNNER, REPAIR_REPORT, REPAIR_APPROVAL, REPAIR_VALIDATOR, S1, S2, TSV,
        ROOT / "docs/experiments/rmatched_s1_s2_steps_cfg_matrix_contract.json",
        ROOT / "scripts/experiment_harness/rmatched_matrix_harn.py", RUNTIME_MANIFEST, RUNTIME_BINDING,
        RECOVERY_VALIDATOR,
    ]
    if harn.APPROVAL.is_file():
        source_paths.append(Path(read_approval_state()["approval_record_path"]))
    sources = [{"path": str(path), "sha256": harn.digest_file(path)} for path in source_paths]
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
        "repair": {"enabled": False},
        "phases": [{
            "phase_id": "resume_matrix", "action_id": "resume_matrix", "resume_action_id": "resume_matrix",
            "input_artifacts": sources, "output_paths": [str(REPORT)],
            "completion_evidence": [{"path": str(REPORT), "sha256": zero}],
        }],
        "filesystems": [storage_model()],
        "commands": [{
            "action_id": "resume_matrix", "argv": commands["resume_matrix"], "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        }],
        "required_preflight_checks": [
            "approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
            "no_duplicate", "policy_bound", "storage_policy_1.25",
        ],
        "notification_events": ["start", "gate", "terminal", "gpu_idle", "disk", "stall"],
    }


def init(approval_record: Path, queue_entry_sha256: str) -> None:
    recovery_gate(approval_record)
    lock_fd = harn.acquire_lock()
    try:
        for directory in (STATE, harn.GENERATIONS, harn.OUTBOX):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        ensure_harn_key()
        record = verify_queue_approval(approval_record, queue_entry_sha256)
        verify_transitive_runtime()
        if harn.APPROVAL.exists():
            approval = read_approval_state()
            if approval.get("state") == "consumed":
                raise RuntimeError("continuation approval is consumed")
            if approval.get("approval_record_sha256") != harn.digest_file(approval_record):
                raise RuntimeError("continuation approval conflicts with existing state")
        else:
            write_approval_state({
                "document_kind": "continuation_approval_state", "schema_version": 1,
                "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID, "state": "approved",
                "issued_at": harn.now(), "approval_record_path": str(approval_record),
                "approval_record_sha256": harn.digest_file(approval_record),
                "queue_entry_binding_sha256": queue_entry_sha256,
                "channel_record_sha256": record["channel_record_sha256"],
            })
        if harn.CURRENT.exists() or harn.PENDING_CONTRACT.exists():
            verify_continuation_authority(read_approval_state())
            print("[INIT OK] existing continuation state verified")
            return
        contract = make_contract()
        conflicts = harn.blocking_gpu_processes()
        preflight = harn.make_preflight(contract, record["channel_record_sha256"], not conflicts)
        ledger = {"document_kind": "event_ledger", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
                  "experiment_id": harn.EXPERIMENT, "run_id": harn.RUN_ID,
                  "bindings": {"contract_raw_sha256": "0" * 64, "preflight_report_raw_sha256": "0" * 64,
                               "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}, "events": []}
        harn.append_event(ledger, "contract_registered")
        if conflicts or preflight["derived_verdict"] != "pass":
            harn.append_event(ledger, "queue_hold", verdict="fail", phase="resource_wait")
            harn.atomic_json(harn.PENDING_CONTRACT, contract)
            harn.atomic_json(harn.PENDING_PREFLIGHT, preflight)
            harn.atomic_json(harn.PENDING_LEDGER, ledger)
            harn.atomic_json(harn.PENDING_QUEUE, {"schema_version": 1, "status": "held", "reason": "preflight_hold"})
        else:
            harn.append_event(ledger, "preflight_passed", verdict="pass")
            recovery_gate(approval_record)
            harn.write_generation(contract, preflight, ledger, "ready")
    finally:
        os.close(lock_fd)


def wait_for_complete_preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    approval = read_approval_state()
    recovery_gate(Path(str(approval.get("approval_record_path", ""))))
    verify_continuation_authority(approval)
    if harn.CURRENT.exists():
        return harn.load_current()
    contract = json.loads(harn.PENDING_CONTRACT.read_text())
    ledger = json.loads(harn.PENDING_LEDGER.read_text())
    while True:
        conflicts = harn.blocking_gpu_processes()
        verify_continuation_authority(approval)
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


def validate_matrix() -> None:
    payload = json.loads(REPORT.read_text())
    results = payload.get("results", {})
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if payload.get("status") != "passed" or set(results) != EXPECTED_CELLS:
        raise RuntimeError("continued matrix report is incomplete")
    for cell in EXPECTED_CELLS:
        if set(results[cell].get("metrics", {})) != required or results[cell].get("peav", {}).get("n_pairs") != 5521:
            raise RuntimeError(f"continued matrix evidence is incomplete for {cell}")


def validate_repair_dependency() -> None:
    recovery_gate()
    manifest = verify_transitive_runtime()
    approval = json.loads(REPAIR_APPROVAL.read_text())
    verify_signed(approval, load_private_key(REPAIR_HARN_KEY), b"meanaudio-repair2-approval-state-v1\0")
    approval_hash = approval.get("channel_record_sha256", "")
    if (approval.get("state") != "consumed" or len(approval_hash) != 64
            or approval_hash == "6d93c976c3319491d291d8b829ddd958116b5a01782ae4839ee2b6b737e96a77"
            or approval.get("experiment_id") != "rmatched-s1-s2-steps-cfg-matrix-repair2"
            or approval.get("run_id") != "run-20260814-seed14159265-musiccaps-repair2"):
        raise RuntimeError("repair2 exact approval evidence is invalid")
    approval_record = Path(str(approval.get("approval_record_path", "")))
    if (not approval_record.is_file()
            or harn.digest_file(approval_record) != approval.get("approval_record_sha256")):
        raise RuntimeError("repair2 exact approval record drift")
    record = json.loads(approval_record.read_text())
    verify_signed(record, load_private_key(QUEUE_KEY), b"meanaudio-queue-approval-v1\0")
    if (record.get("status") != "approved" or record.get("experiment_id") != approval.get("experiment_id")
            or record.get("run_id") != approval.get("run_id")
            or record.get("channel_record_sha256") != approval_hash):
        raise RuntimeError("repair2 authenticated approval record is invalid")
    validator = Path(str(approval.get("archived_executable", "")))
    if (not validator.is_file()
            or harn.digest_file(validator) != approval.get("archived_executable_sha256")):
        raise RuntimeError("repair2 immutable validator executable drift")
    python_binding = manifest_entry(manifest, "system_python", "file")
    python_fd = open_verified_fd(Path(python_binding["path"]), python_binding["sha256"])
    validator_fd = open_verified_fd(validator, approval["archived_executable_sha256"])
    try:
        completed = subprocess.run(
            [f"/proc/self/fd/{python_fd}", "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
             f"/proc/self/fd/{validator_fd}", "validate-report",
             "--expected-approval-sha256", approval_hash],
            cwd="/", text=True, capture_output=True, env=safe_env(), pass_fds=(python_fd, validator_fd),
        )
    finally:
        os.close(python_fd)
        os.close(validator_fd)
    if completed.returncode:
        raise RuntimeError(f"repair2 completion evidence is invalid: {completed.stderr[-1000:]}")


def verify_runtime_bindings(contract: dict[str, Any]) -> None:
    verify_transitive_runtime()
    prereg = json.loads(PREREG.read_text())
    for descriptor, expected in (
        (Path(prereg["runner"]["path"]), prereg["runner"]["sha256"]),
        (Path(prereg["harn"]["path"]), prereg["harn"]["sha256"]),
        (Path(prereg["immutable_protocol_source"]["contract"]), prereg["immutable_protocol_source"]["contract_sha256"]),
    ):
        if not descriptor.is_file() or harn.digest_file(descriptor) != expected:
            raise RuntimeError(f"continuation preregistered descriptor drift: {descriptor}")
    if contract["bindings"]["runtime_sha256"] != harn.digest_file(Path(__file__)):
        raise RuntimeError("continuation HARN executable drift")
    if contract["bindings"]["policy_bundle_sha256"] != harn.policy_hash():
        raise RuntimeError("continuation policy binding drift")
    if contract["bindings"]["schema_bundle_sha256"] != harn.schema_hash():
        raise RuntimeError("continuation schema binding drift")
    if contract["bindings"]["command_set_sha256"] != harn.digest_bytes(harn.canonical(command_registry())):
        raise RuntimeError("continuation command binding drift")
    for binding in contract["corpus"]["source_artifacts"]:
        path = Path(binding["path"])
        if not path.is_file() or harn.digest_file(path) != binding["sha256"]:
            raise RuntimeError(f"continuation preregistered input drift: {path}")


def verify_transitive_runtime() -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text())
    runtime = prereg.get("transitive_runtime", {})
    if (Path(str(runtime.get("manifest", ""))) != RUNTIME_MANIFEST
            or harn.digest_file(RUNTIME_MANIFEST) != runtime.get("manifest_sha256")
            or Path(str(runtime.get("verifier", ""))) != RUNTIME_BINDING
            or harn.digest_file(RUNTIME_BINDING) != runtime.get("verifier_sha256")):
        raise RuntimeError("continuation transitive runtime descriptor drift")
    return verify_manifest(RUNTIME_MANIFEST, runtime["manifest_sha256"], {
        "system_python", "system_stdlib", "dac_pyvenv_cfg", "dac_site_packages",
        "peav_pyvenv_cfg", "peav_site_packages", "workspace_meanaudio", "workspace_eval",
        "workspace_meanaudio_init", "phase4_evaluator", "peav_evaluator", "isolated_bootstrap",
        "system_native_runtime", "recovery_validator", "bash",
        "df", "tail", "tr", "find", "wc", "mkdir", "tee", "dirname",
    })


def manifest_entry(manifest: dict[str, Any], role: str, kind: str | None = None) -> dict[str, Any]:
    matches = [entry for entry in manifest.get("entries", []) if entry.get("role") == role]
    if len(matches) != 1 or (kind is not None and matches[0].get("kind") != kind):
        raise RuntimeError(f"continuation runtime role is missing or invalid: {role}")
    return matches[0]


def open_verified_fd(path: Path, expected_sha256: str) -> int:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("continuation runner is not a regular file")
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(fd, 8 << 20, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("continuation runner descriptor drift")
        return fd
    except BaseException:
        os.close(fd)
        raise


def run_continuation(contract: dict[str, Any]) -> subprocess.CompletedProcess[Any]:
    recovery_gate()
    manifest = verify_transitive_runtime()
    expected = next(item["sha256"] for item in contract["corpus"]["source_artifacts"] if item["path"] == str(RUNNER))
    descriptors = {
        "MEANAUDIO_SYSTEM_PYTHON_FD": manifest_entry(manifest, "system_python", "file"),
        "MEANAUDIO_ISOLATED_BOOTSTRAP_FD": manifest_entry(manifest, "isolated_bootstrap", "file"),
        "MEANAUDIO_WORKSPACE_EVAL_FD": manifest_entry(manifest, "workspace_eval", "file"),
        "MEANAUDIO_PHASE4_EVALUATOR_FD": manifest_entry(manifest, "phase4_evaluator", "file"),
        "MEANAUDIO_PEAV_EVALUATOR_FD": manifest_entry(manifest, "peav_evaluator", "file"),
        "MEANAUDIO_PACKAGE_INIT_FD": manifest_entry(manifest, "workspace_meanaudio_init", "file"),
    }
    runner_fd = open_verified_fd(RUNNER, expected)
    opened: list[int] = [runner_fd]
    try:
        environment = safe_env()
        for variable, descriptor in descriptors.items():
            opened_fd = open_verified_fd(Path(descriptor["path"]), descriptor["sha256"])
            opened.append(opened_fd)
            environment[variable] = str(opened_fd)
        return subprocess.run(
            ["/bin/bash", f"/proc/self/fd/{runner_fd}"], cwd=ROOT, env=environment, pass_fds=tuple(opened),
        )
    finally:
        for opened_fd in opened:
            os.close(opened_fd)


def run() -> None:
    recovery_gate()
    lock_fd = harn.acquire_lock()
    try:
        contract, preflight, ledger = wait_for_complete_preflight()
        verify_runtime_bindings(contract)
        validate_repair_dependency()
        approval = reserve_approval(read_approval_state())
        verify_continuation_authority(approval)
        harn.append_event(ledger, "resources_acquired", phase="resume_matrix")
        started = harn.append_event(ledger, "experiment_started", phase="resume_matrix", notification="pending")
        harn.notify("start", "Started storage-correct continuation of the repaired R-Matched matrix.")
        harn.append_event(ledger, "notification_delivery", relation=started, phase="resume_matrix", notification="delivered")
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.write_generation(contract, preflight, ledger, "active")
        verify_runtime_bindings(contract)
        validate_repair_dependency()
        completed = run_continuation(contract)
        if completed.returncode:
            raise RuntimeError(f"matrix continuation failed with exit {completed.returncode}")
        validate_matrix()
        gate = harn.append_event(ledger, "gate_result", verdict="pass", phase="resume_matrix", notification="pending")
        harn.notify("matrix_complete", "Storage-correct R-Matched matrix continuation completed.", report=REPORT)
        harn.append_event(ledger, "notification_delivery", relation=gate, phase="resume_matrix", notification="delivered")
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.write_generation(contract, preflight, ledger, "active")
        write_approval_state({**read_approval_state(), "state": "consumed", "consumed_at": harn.now()})
        recovery_gate(Path(str(approval.get("approval_record_path", ""))))
        harn.terminal(contract, preflight, ledger, True, "R-Matched matrix continuation completed; queue handoff is next.")
    except BaseException as exc:
        try:
            if harn.CURRENT.exists():
                contract, preflight, ledger = harn.load_current()
                terminal = {"experiment_completed", "experiment_failed", "experiment_interrupted"}
                if not any(event["event_kind"] in terminal for event in ledger["events"]):
                    recovery_gate()
                    harn.terminal(contract, preflight, ledger, False, f"Continuation failure: {type(exc).__name__}: {exc}")
        finally:
            raise
    finally:
        os.close(lock_fd)


def _guarded_continuation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with recovery_guard():
            return function(*args, **kwargs)
    return wrapped


def _guarded_continuation_init(function):
    @wraps(function)
    def wrapped(approval_record: Path, *args, **kwargs):
        with recovery_guard(approval_record):
            return function(approval_record, *args, **kwargs)
    return wrapped


def watch(once: bool) -> None:
    harn.watch(once)


ensure_harn_key = _guarded_continuation(ensure_harn_key)
write_approval_state = _guarded_continuation(write_approval_state)
safe_notify = _guarded_continuation(safe_notify)
wait_for_complete_preflight = _guarded_continuation(wait_for_complete_preflight)
validate_repair_dependency = _guarded_continuation(validate_repair_dependency)
run_continuation = _guarded_continuation(run_continuation)
run = _guarded_continuation(run)
watch = _guarded_continuation(watch)
init = _guarded_continuation_init(init)


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
            harn.atomic_json(harn.PROCESS, {"controller_pid": os.getpid(), "started_at": harn.now(),
                "controller_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
            run()
        else:
            watch(args.once)


if __name__ == "__main__":
    main()
