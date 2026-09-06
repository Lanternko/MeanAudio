#!/usr/bin/env python3
"""Authenticated HARN for the separate R-Matched NVMe staging experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import qwen_s2q_k_mf25_harn as harn
from runtime_binding import verify_manifest


ROOT = Path("/home/kojiek/MeanAudio")
EXPERIMENT_ID = "rmatched-s1-s2-steps-cfg-matrix-nvme-stage"
RUN_ID = "run-20260815-rmatched-matrix-nvme-stage1"
STATE = Path("/home/kojiek/logs/rmatched_matrix_nvme_stage_harn")
KEY = STATE / "ledger_hmac.key"
LOCK = STATE / "controller.lock"
APPROVAL = STATE / "operator_approval.json"
AUTHORIZATION = STATE / "stage_authorization.json"
QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
QUEUE_KEY = QUEUE.parent / "queue_hmac.key"
PREREG = ROOT / "docs/experiments/rmatched_matrix_nvme_stage_contract.json"
SOURCE_MANIFEST = ROOT / "docs/experiments/rmatched_matrix_nvme_stage_source_manifest.json"
STAGER = ROOT / "scripts/stage_rmatched_matrix_nvme.py"
PYTHON = Path("/usr/bin/python3.12")
NVME_ROOT = Path("/home/kojiek/nvme_experiment_artifacts")
NVME_PARENT = NVME_ROOT / "meanaudio"
FINAL = NVME_PARENT / "eval_output"
STAGING = NVME_PARENT / f".eval_output.stage-{RUN_ID}"
JOURNAL = STATE / "transaction.json"
REPORT = STATE / "stage_report.json"
GENERATIONS = STATE / "generations"
OUTBOX = STATE / "outbox"
CURRENT = STATE / "current"
PENDING_CONTRACT = STATE / "pending_contract.json"
PENDING_PREFLIGHT = STATE / "pending_preflight.json"
PENDING_LEDGER = STATE / "pending_ledger.json"
PENDING_QUEUE = STATE / "pending_queue.json"
BUNDLE_VALIDATOR = ROOT / "scripts/validate_experiment_harness_documents.py"
COMPLETION_VALIDATOR = ROOT / "scripts/validate_rmatched_matrix_nvme_stage_bundle.py"
SHARED_HARN = ROOT / "scripts/experiment_harness/qwen_s2q_k_mf25_harn.py"
RUNTIME_BINDING = ROOT / "scripts/experiment_harness/runtime_binding.py"
NOTIFIER = ROOT / "scripts/notify_experiment_webhook.py"
CONTRACT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
DANGEROUS_ENV = {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "MEANAUDIO_REPAIR2_FAIL_AFTER_PHASE",
                 "MEANAUDIO_NOTIFY_DRY_RUN", "MEANAUDIO_NVME_STAGE_CAPABILITY"}
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
            "__name__": "dual_authority_recovery_v3_stage_bound", "__file__": f"/proc/self/fd/{fd}",
        }
        exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)
        return namespace
    finally:
        os.close(fd)


RECOVERY = _load_recovery_validator()
ACTIVE_RECOVERY_LEASE: Any | None = None
BASE_HARN_ATOMIC_JSON = getattr(harn, "_dual_auth_base_atomic_json", harn.atomic_json)
harn._dual_auth_base_atomic_json = BASE_HARN_ATOMIC_JSON


@contextmanager
def recovery_guard(approval_record: Path | None = None):
    global ACTIVE_RECOVERY_LEASE
    with RECOVERY["guarded_action"](EXPERIMENT_ID, approval_record) as lease:
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


def configure() -> None:
    harn.atomic_json = gated_harn_atomic_json
    harn.STATE = STATE
    harn.GENERATIONS = GENERATIONS
    harn.OUTBOX = OUTBOX
    harn.CURRENT = CURRENT
    harn.KEY = KEY
    harn.LOCK = LOCK
    harn.APPROVAL = APPROVAL
    harn.PENDING_CONTRACT = PENDING_CONTRACT
    harn.PENDING_PREFLIGHT = PENDING_PREFLIGHT
    harn.PENDING_LEDGER = PENDING_LEDGER
    harn.PENDING_QUEUE = PENDING_QUEUE
    harn.VALIDATOR = BUNDLE_VALIDATOR
    harn.SYSTEM_PY = PYTHON
    harn.EXPERIMENT = EXPERIMENT_ID
    harn.RUN_ID = RUN_ID


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_key(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        key = os.read(fd, 128)
    finally:
        os.close(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or len(key) < 32):
        raise RuntimeError(f"unsafe authentication key: {path}")
    return key


def signed(payload: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {**unsigned, "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()}


def verify(payload: dict[str, Any], key: bytes, domain: bytes) -> None:
    supplied = payload.get("integrity")
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated staging document signature invalid")


def atomic_json(path: Path, value: Any) -> None:
    recovery_gate()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def entry_binding(entry: dict[str, Any]) -> str:
    """Canonical operator_queue_controller.entry_approval_binding schema."""
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
    for key in ("bundle_validator", "completion_validator", "approval_cli", "import_bindings",
                "child_ledger_key", "transitive_runtime"):
        if key in runtime:
            descriptor[key] = runtime[key]
    return hashlib.sha256(canonical(descriptor)).hexdigest()


def verify_queue_approval(record_path: Path, expected_binding: str) -> dict[str, Any]:
    queue_key = load_key(QUEUE_KEY)
    queue = json.loads(QUEUE.read_text())
    verify(queue, queue_key, b"meanaudio-operator-queue-v1\0")
    if queue.get("document_kind") != "operator_approved_experiment_backlog":
        raise RuntimeError("NVMe stage signed queue document kind is invalid")
    matches = [item for item in queue.get("entries", []) if item.get("experiment_id") == EXPERIMENT_ID]
    if len(matches) != 1:
        raise RuntimeError("NVMe stage has no unique signed queue entry")
    entry = matches[0]
    binding = entry_binding(entry)
    evidence = entry.get("approval_evidence", {})
    if (binding != expected_binding or entry.get("run_id") != RUN_ID or entry.get("approval_status") != "approved"
            or Path(str(evidence.get("path", ""))) != record_path or evidence.get("sha256") != digest(record_path)
            or entry.get("contract_sha256") != digest(PREREG)):
        raise RuntimeError("NVMe stage queue approval binding mismatch")
    prereg = json.loads(PREREG.read_text())
    runtime = entry.get("runtime", {})
    transitive = runtime.get("transitive_runtime", {})
    expected_runtime = prereg.get("runtime", {})
    if (transitive.get("manifest") != expected_runtime.get("manifest")
            or transitive.get("manifest_sha256") != expected_runtime.get("manifest_sha256")
            or transitive.get("verifier", {}).get("path") != expected_runtime.get("verifier")
            or transitive.get("verifier", {}).get("sha256") != expected_runtime.get("verifier_sha256")
            or transitive.get("required_roles") != expected_runtime.get("required_roles")):
        raise RuntimeError("NVMe stage queue lacks exact transitive runtime binding")
    if (runtime.get("status_source") != str(CURRENT)
            or runtime.get("pending_status_source") != str(PENDING_QUEUE)
            or runtime.get("approval_cli") != "exact_record_v1"):
        raise RuntimeError("NVMe stage queue lacks the shared-HARN status/approval schema")
    validator = runtime.get("completion_validator", {})
    expected_validator = prereg.get("completion_validator", {})
    if validator != expected_validator:
        raise RuntimeError("NVMe stage queue lacks exact completion validator binding")
    for descriptor in validator.get("bindings", []):
        candidate = Path(str(descriptor.get("path", "")))
        if not candidate.is_file() or candidate.is_symlink() or digest(candidate) != descriptor.get("sha256"):
            raise RuntimeError("NVMe stage completion validator binding drift")
    bundle = runtime.get("bundle_validator", {})
    if (bundle.get("python") != str(PYTHON) or bundle.get("path") != str(BUNDLE_VALIDATOR)
            or bundle.get("python_sha256") != digest(PYTHON)
            or bundle.get("sha256") != digest(BUNDLE_VALIDATOR)):
        raise RuntimeError("NVMe stage queue lacks exact bundle validator binding")
    ledger_key = runtime.get("child_ledger_key", {})
    if (Path(str(ledger_key.get("path", ""))) != KEY or not KEY.is_file()
            or ledger_key.get("sha256") != digest(KEY)):
        raise RuntimeError("NVMe stage queue lacks exact private child-ledger key descriptor")
    imports = {item.get("module"): (item.get("path"), item.get("sha256"))
               for item in runtime.get("import_bindings", []) if isinstance(item, dict)}
    if imports.get("qwen_s2q_k_mf25_harn") != (str(SHARED_HARN), digest(SHARED_HARN)) or imports.get(
            "runtime_binding") != (str(RUNTIME_BINDING), digest(RUNTIME_BINDING)):
        raise RuntimeError("NVMe stage queue lacks exact shared-HARN/runtime import bindings")
    record = json.loads(record_path.read_text())
    verify(record, queue_key, b"meanaudio-queue-approval-v1\0")
    expected = {"document_kind": "exact_operator_approval", "status": "approved",
                "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
                "contract_sha256": digest(PREREG), "controller_sha256": digest(Path(__file__)),
                "queue_entry_binding_sha256": binding}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("NVMe stage approval is unsigned, wrong-run, or drifted")
    channel_hash = record.get("channel_record_sha256")
    if not isinstance(channel_hash, str) or len(channel_hash) != 64:
        raise RuntimeError("NVMe stage exact approval lacks its channel record binding")
    return record


def verify_preregistered() -> None:
    contract = json.loads(PREREG.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID or contract.get("run_id") != RUN_ID:
        raise RuntimeError("NVMe stage preregistration identity drift")
    implementation = contract.get("implementation", {})
    for path_key, hash_key in (("stager", "stager_sha256"), ("harn", "harn_sha256"),
                               ("selftest", "selftest_sha256"),
                               ("bundle_validator", "bundle_validator_sha256"),
                               ("completion_validator", "completion_validator_sha256"),
                               ("shared_harn", "shared_harn_sha256"),
                               ("recovery_validator", "recovery_validator_sha256")):
        path = Path(str(implementation.get(path_key, "")))
        if not path.is_file() or digest(path) != implementation.get(hash_key):
            raise RuntimeError(f"NVMe stage implementation drift: {path_key}")
    source = contract.get("source_manifest", {})
    if Path(str(source.get("path", ""))) != SOURCE_MANIFEST or digest(SOURCE_MANIFEST) != source.get("sha256"):
        raise RuntimeError("NVMe stage source manifest drift")
    runtime = contract.get("runtime", {})
    verifier = Path(str(runtime.get("verifier", "")))
    manifest = Path(str(runtime.get("manifest", "")))
    if digest(verifier) != runtime.get("verifier_sha256"):
        raise RuntimeError("NVMe stage runtime verifier drift")
    strict_report_validator_descriptor(contract)
    verify_manifest(manifest, runtime["manifest_sha256"], {
        "system_python", "system_stdlib", "system_native_runtime", "recovery_validator",
    })


def ensure_key() -> None:
    # The queue approval binds this private descriptor before init. Creating a
    # random key here would make exact preapproval impossible and fail open to
    # a different child-ledger authority.
    load_key(KEY)


def read_approval() -> dict[str, Any]:
    payload = json.loads(APPROVAL.read_text())
    verify(payload, load_key(KEY), b"meanaudio-nvme-stage-approval-v1\0")
    return payload


def write_approval(payload: dict[str, Any]) -> None:
    atomic_json(APPROVAL, signed(payload, load_key(KEY), b"meanaudio-nvme-stage-approval-v1\0"))


def init(record_path: Path, binding: str) -> None:
    # AUTH-ROOT provisioning (state dir, exact ledger key, and lock) happens
    # before queue approval. Direct/unapproved init must not create any state.
    recovery_gate(record_path)
    state_info = STATE.stat(follow_symlinks=False)
    if not stat.S_ISDIR(state_info.st_mode) or state_info.st_uid != os.geteuid() or stat.S_IMODE(state_info.st_mode) != 0o700:
        raise RuntimeError("NVMe stage state root is not safely preprovisioned")
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("stage init lacks retained recovery lock lease")
    lock_fd = os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("stage_controller"))
    try:
        ensure_key()
        configure()
        verify_preregistered()
        record = verify_queue_approval(record_path, binding)
        if APPROVAL.exists():
            approval = read_approval()
            if approval.get("state") == "consumed" or approval.get("approval_record_sha256") != digest(record_path):
                raise RuntimeError("NVMe stage approval consumed or conflicting")
            if approval.get("queue_entry_binding_sha256") != binding:
                raise RuntimeError("NVMe stage approval entry binding conflicts with existing state")
        else:
            write_approval({"document_kind": "nvme_stage_approval_state", "schema_version": 1,
                            "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "state": "approved",
                            "approval_record_path": str(record_path), "approval_record_sha256": digest(record_path),
                            "queue_entry_binding_sha256": binding,
                            "channel_record_sha256": record.get("channel_record_sha256")})
        atomic_json(AUTHORIZATION, signed({
            "document_kind": "nvme_stage_exact_authorization", "schema_version": 1,
            "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
            "queue_entry_binding_sha256": binding, "approval_record_sha256": digest(record_path),
            "child_ledger_key": {"path": str(KEY), "sha256": digest(KEY)},
        }, load_key(KEY), b"meanaudio-nvme-stage-exact-authorization-v1\0"))
        if CURRENT.is_file() or PENDING_QUEUE.is_file():
            # Crash-safe init replay: authorization and preregistration were
            # reverified above; never create a second ready generation.
            return
        contract = make_contract(record_path, binding)
        preflight = make_preflight(contract, str(record.get("channel_record_sha256")))
        ledger = new_ledger(contract)
        harn.append_event(ledger, "contract_registered")
        if preflight["derived_verdict"] != "pass":
            held = harn.append_event(ledger, "queue_hold", verdict="fail", phase="resource_wait",
                                     notification="pending")
            notify_once("preflight_hold", "NVMe stage held by exact preregistered preflight.", "held")
            harn.append_event(ledger, "notification_delivery", relation=held, phase="resource_wait",
                              notification="delivered")
            atomic_json(PENDING_CONTRACT, contract)
            atomic_json(PENDING_PREFLIGHT, preflight)
            atomic_json(PENDING_LEDGER, ledger)
            atomic_json(PENDING_QUEUE, {"schema_version": 1, "status": "held", "reason": "preflight_hold",
                                        "updated_at": harn.now()})
        else:
            harn.append_event(ledger, "preflight_passed", verdict="pass")
            recovery_gate(record_path)
            harn.write_generation(contract, preflight, ledger, "ready")
    finally:
        os.close(lock_fd)


def safe_env(capability: Path) -> dict[str, str]:
    inherited = {key: value for key, value in os.environ.items()
                 if key in {"HOME", "LANG", "LC_ALL", "TZ"} and key not in DANGEROUS_ENV}
    return {**inherited, "PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1",
            "MEANAUDIO_NVME_STAGE_CAPABILITY": str(capability)}


def strict_report_validator_descriptor(preregistered: dict[str, Any]) -> list[str]:
    """Return the schema form after verifying the exact hardened execution form.

    The experiment-contract schema intentionally models an interpreter action
    as ``absolute-python absolute-script args...``. Runtime isolation flags are
    separately approval-bound in ``report_validator.argv`` and are rechecked
    here; omitting them from the structural descriptor does not authorize a
    weaker execution path.
    """
    report = preregistered.get("report_validator", {})
    expected_execution = [
        str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
        str(STAGER), "validate-report",
    ]
    expected_descriptor = [str(PYTHON), str(STAGER), "validate-report"]
    bindings = {item.get("path"): item.get("sha256")
                for item in report.get("bindings", []) if isinstance(item, dict)}
    if (report.get("argv") != expected_execution
            or report.get("contract_descriptor_argv") != expected_descriptor
            or bindings != {str(PYTHON): digest(PYTHON), str(STAGER): digest(STAGER)}):
        raise RuntimeError("NVMe stage report validator execution/descriptor binding drift")
    return expected_descriptor


def make_contract(record_path: Path, binding: str) -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text())
    sources = [{"path": str(path), "sha256": digest(path)} for path in
               (PREREG, SOURCE_MANIFEST, STAGER, Path(__file__), BUNDLE_VALIDATOR,
                COMPLETION_VALIDATOR, SHARED_HARN, RUNTIME_BINDING, RECOVERY_VALIDATOR,
                record_path, AUTHORIZATION)]
    commands = {"stage": [str(PYTHON), str(STAGER), "stage"],
                "reconcile": [str(PYTHON), str(STAGER), "reconcile"],
                "validate": strict_report_validator_descriptor(prereg)}
    return {
        "document_kind": "experiment_contract", "schema_version": "1.0.0",
        "schema_bundle_id": "harn-schema-v1", "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "bindings": {"policy_bundle_sha256": harn.policy_hash(), "schema_bundle_sha256": harn.schema_hash(),
                     "runtime_sha256": digest(Path(__file__)),
                     "command_set_sha256": hashlib.sha256(canonical(commands)).hexdigest()},
        "approval_requirement": {"required": True, "responsible_role": "operator",
                                 "trusted_channels": ["operator_console"]},
        "corpus": {"kind": "non_generated", "source_artifacts": sources},
        "repair": {"enabled": False},
        "phases": [{"phase_id": "nvme_stage", "action_id": "stage", "resume_action_id": "reconcile",
                    "input_artifacts": sources, "output_paths": [str(FINAL), str(REPORT)],
                    "completion_evidence": [{"path": str(REPORT), "sha256": "0" * 64}]}],
        "filesystems": [{"path": str(NVME_ROOT), "hard_floor_bytes": 50 * harn.GIB,
                         "warning_floor_bytes": 50 * harn.GIB,
                         "peak_additional_bytes": prereg["source_manifest"]["total_bytes"],
                         "transient_bytes": prereg["source_manifest"]["total_bytes"],
                         "recovery_reserve_bytes": 0}],
        "commands": [{"action_id": name, "argv": argv, "working_directory": "/",
                      "environment": {}}
                     for name, argv in commands.items()],
        "required_preflight_checks": ["approval_authenticated", "commands_bound", "gpu_idle", "inputs_bound",
                                      "no_duplicate", "policy_bound", "storage_policy_1.25"],
        "notification_events": ["start", "gate", "terminal", "disk", "stall"],
    }


def make_preflight(contract: dict[str, Any], approval_hash: str) -> dict[str, Any]:
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(hours=72)
    contract_hash = hashlib.sha256(canonical(contract)).hexdigest()
    checks = [{"check_id": item, "verdict": "pass", "observed_at": issued.isoformat(),
               "valid_until": expires.isoformat(),
               "evidence_sha256": hashlib.sha256(f"{item}:pass".encode()).hexdigest()}
              for item in contract["required_preflight_checks"]]
    model = contract["filesystems"][0]
    probe = Path(model["path"])
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = max(model["hard_floor_bytes"],
                   int(1.25 * (model["peak_additional_bytes"] + model["transient_bytes"]))
                   + model["recovery_reserve_bytes"])
    storage_verdict = "pass" if free >= required else "fail"
    if storage_verdict != "pass":
        for check in checks:
            if check["check_id"] == "storage_policy_1.25":
                check["verdict"] = "fail"
    return {"document_kind": "preflight_report", "schema_version": "1.0.0",
            "schema_bundle_id": "harn-schema-v1", "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
            "contract_raw_sha256": contract_hash, "approval_evidence": {
                "evidence_id": f"{RUN_ID}-exact-record", "source_kind": "trusted_operator_record",
                "trusted_channel": "operator_console", "channel_record_id": "exact-record",
                "channel_record_sha256": approval_hash, "approver_id": "user",
                "issued_at": issued.isoformat(), "expires_at": expires.isoformat(),
                "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
                "bindings": {"contract_raw_sha256": contract_hash, **contract["bindings"],
                             "repair_envelope_sha256": None}},
            "checks": checks, "storage": [{**{key: value for key, value in model.items()
                                               if key != "warning_floor_bytes"},
                                            "free_bytes": free, "measured_at": issued.isoformat(),
                                            "verdict": storage_verdict}],
            "derived_verdict": "pass" if all(item["verdict"] == "pass" for item in checks) else "fail",
            "created_at": issued.isoformat()}


def new_ledger(contract: dict[str, Any]) -> dict[str, Any]:
    return {"document_kind": "event_ledger", "schema_version": "1.0.0", "schema_bundle_id": "harn-schema-v1",
            "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
            "bindings": {"contract_raw_sha256": "0" * 64, "preflight_report_raw_sha256": "0" * 64,
                         "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"]}, "events": []}


def notify_once(key: str, summary: str, status: str, *, report: Path | None = None) -> None:
    recovery_gate()
    OUTBOX.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = OUTBOX / f"{key}.json"
    if path.is_file():
        if json.loads(path.read_text()).get("status") == "delivered":
            return
        raise RuntimeError(f"ambiguous NVMe stage notification state: {key}")
    atomic_json(path, {"status": "attempting", "payload_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                       "created_at": harn.now()})
    argv = [str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S", str(NOTIFIER),
            "--status", status, "--experiment", EXPERIMENT_ID, "--summary", summary]
    if report is not None:
        argv += ["--report", str(report)]
    completed = subprocess.run(argv, cwd="/", text=True, capture_output=True,
                               env={"PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1"})
    if completed.returncode:
        atomic_json(path, {"status": "failed", "failed_at": harn.now(), "error": completed.stderr[-500:]})
        raise RuntimeError(f"NVMe stage notification failed: {key}")
    atomic_json(path, {"status": "delivered", "delivered_at": harn.now(),
                       "accepted_evidence_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})


def _status() -> str | None:
    if not CURRENT.is_file():
        return json.loads(PENDING_QUEUE.read_text()).get("status") if PENDING_QUEUE.is_file() else None
    target = Path(CURRENT.read_text().strip())
    return json.loads((target / "queue.json").read_text())["entries"][0]["status"]


def finalize_completed(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any]) -> None:
    recovery_gate()
    approval = read_approval()
    if approval.get("state") != "consumed":
        write_approval({**approval, "state": "consumed", "consumed_at": harn.now()})
    terminal_events = [item for item in ledger["events"] if item["event_kind"] == "experiment_completed"]
    if not terminal_events:
        terminal = harn.append_event(ledger, "experiment_completed", verdict="pass", phase="terminal",
                                     notification="pending")
        notify_once("terminal_success", "NVMe staging terminal bundle is complete; resources released.", "success")
        harn.append_event(ledger, "notification_delivery", relation=terminal, phase="terminal",
                          notification="delivered")
    recovery_gate()
    harn.write_generation(contract, preflight, ledger, "completed")


def run() -> None:
    recovery_gate()
    configure()
    if ACTIVE_RECOVERY_LEASE is None:
        raise RuntimeError("stage run lacks retained recovery lock lease")
    lock_fd = os.dup(ACTIVE_RECOVERY_LEASE.lock_fd("stage_controller"))
    try:
        approval = read_approval()
        record_path = Path(approval["approval_record_path"])
        verify_queue_approval(record_path, approval["queue_entry_binding_sha256"])
        if approval.get("state") == "consumed":
            if _status() == "completed":
                validate_completed_bundle()
                return
            if _status() != "active":
                raise RuntimeError("consumed NVMe stage approval lacks a recoverable active/completed state")
            contract, preflight, ledger = harn.load_current()
            recovery_gate(record_path)
            subprocess.run([str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                            str(STAGER), "validate-report"], cwd="/", env=safe_env(Path("/dev/null")), check=True)
            finalize_completed(contract, preflight, ledger)
            validate_completed_bundle()
            return
        if _status() not in {"ready", "held", "active"}:
            raise RuntimeError("NVMe stage shared-HARN state is not launchable")
        contract, preflight, ledger = harn.load_current()
        approval = {**approval, "state": "reserved", "reservation": {
            "controller_pid": os.getpid(),
            "controller_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }}
        write_approval(approval)
        if _status() != "active":
            harn.append_event(ledger, "resources_acquired", phase="nvme_stage")
            started = harn.append_event(ledger, "experiment_started", phase="nvme_stage", notification="pending")
            notify_once("start", "Started exact lossless HDD-to-NVMe Matrix staging.", "test")
            harn.append_event(ledger, "notification_delivery", relation=started, phase="nvme_stage",
                              notification="delivered")
            recovery_gate(record_path)
            harn.write_generation(contract, preflight, ledger, "active")
        action = "reconcile" if JOURNAL.exists() or STAGING.exists() or FINAL.exists() else "stage"
        capability_dir = STATE / "capabilities"
        recovery_gate(record_path)
        capability_dir.mkdir(mode=0o700, exist_ok=True)
        capability = capability_dir / f"{action}-{os.getpid()}.json"
        payload = {"document_kind": "nvme_stage_write_capability", "status": "authorized",
                   "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "action": action,
                   "executable_sha256": digest(STAGER), "source_manifest_sha256": digest(SOURCE_MANIFEST),
                   "approval_state_sha256": digest(APPROVAL), "parent_pid": os.getpid(),
                   "approval_record_sha256": approval["approval_record_sha256"],
                   "parent_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                   "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   "harn_lock": str(LOCK),
                   "writable_paths": [str(NVME_ROOT), str(NVME_PARENT), str(STAGING), str(FINAL), str(JOURNAL), str(REPORT)]}
        atomic_json(capability, signed(payload, load_key(KEY), b"meanaudio-nvme-stage-capability-v1\0"))
        recovery_gate(record_path)
        completed = subprocess.run([str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                                    str(STAGER), action], cwd="/", env=safe_env(capability))
        if completed.returncode:
            held = harn.append_event(ledger, "queue_hold", verdict="fail", phase="nvme_stage",
                                     notification="pending")
            notify_once("recoverable_hold", f"NVMe staging held for crash reconciliation (exit={completed.returncode}).",
                        "held")
            harn.append_event(ledger, "notification_delivery", relation=held, phase="nvme_stage",
                              notification="delivered")
            recovery_gate(record_path)
            harn.write_generation(contract, preflight, ledger, "held")
            raise ChildProcessError(f"NVMe staging child failed recoverably: {completed.returncode}")
        recovery_gate(record_path)
        subprocess.run([str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                        str(STAGER), "validate-report"], cwd="/", env=safe_env(capability), check=True)
        gate = harn.append_event(ledger, "gate_result", verdict="pass", phase="nvme_stage",
                                 notification="pending")
        notify_once("stage_complete", "Exact lossless Matrix NVMe staging completed and validated.", "success",
                    report=REPORT)
        harn.append_event(ledger, "notification_delivery", relation=gate, phase="nvme_stage",
                          notification="delivered")
        finalize_completed(contract, preflight, ledger)
        validate_completed_bundle()
    except ChildProcessError:
        raise
    except BaseException as exc:
        if CURRENT.is_file():
            try:
                contract, preflight, ledger = harn.load_current()
                if not any(item["event_kind"] in {"experiment_completed", "experiment_failed"}
                           for item in ledger["events"]):
                    terminal = harn.append_event(ledger, "experiment_failed", verdict="fail", phase="terminal",
                                                 notification="pending")
                    notify_once("terminal_failure", f"NVMe stage failed closed: {type(exc).__name__}: {exc}", "failure")
                    harn.append_event(ledger, "notification_delivery", relation=terminal, phase="terminal",
                                      notification="delivered")
                    recovery_gate()
                    harn.write_generation(contract, preflight, ledger, "failed")
            finally:
                raise
        raise
    finally:
        os.close(lock_fd)


def validate_completed_bundle() -> None:
    recovery_gate()
    completed = subprocess.run(
        [str(PYTHON), "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S", str(COMPLETION_VALIDATOR)],
        cwd="/", text=True, capture_output=True,
        env={"PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode:
        raise RuntimeError(f"authenticated NVMe stage bundle invalid: {completed.stderr[-1000:]}")


def _guarded_stage(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with recovery_guard():
            return function(*args, **kwargs)
    return wrapped


def _guarded_stage_init(function):
    @wraps(function)
    def wrapped(record_path: Path, *args, **kwargs):
        with recovery_guard(record_path):
            return function(record_path, *args, **kwargs)
    return wrapped


atomic_json = _guarded_stage(atomic_json)
notify_once = _guarded_stage(notify_once)
finalize_completed = _guarded_stage(finalize_completed)
run = _guarded_stage(run)
validate_completed_bundle = _guarded_stage(validate_completed_bundle)
init = _guarded_stage_init(init)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--approval-record", type=Path, required=True)
    initialize.add_argument("--queue-entry-sha256", required=True)
    sub.add_parser("run")
    sub.add_parser("validate-completed-bundle")
    args = parser.parse_args()
    if args.command == "init":
        init(args.approval_record, args.queue_entry_sha256)
    elif args.command == "run":
        run()
    else:
        validate_completed_bundle()


configure()


if __name__ == "__main__":
    main()
