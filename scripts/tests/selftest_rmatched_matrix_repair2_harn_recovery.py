#!/usr/bin/env python3
"""No-artifact HARN and top-controller restart checks for Matrix repair2."""
from __future__ import annotations

import importlib.util
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
HARN_DIR = ROOT / "scripts/experiment_harness"
sys.path.insert(0, str(HARN_DIR))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "recovery_gate"):
        module.recovery_gate = lambda *_args, **_kwargs: None
    if hasattr(module, "recovery_gate_for_entry"):
        module.recovery_gate_for_entry = lambda *_args, **_kwargs: None
    if hasattr(module, "recovery_guard"):
        class FixtureLease:
            fd = os.open("/dev/null", os.O_RDONLY)

            def lock_fd(self, _role: str) -> int:
                return self.fd

            def reverify(self, *_args, **_kwargs):
                return None

            def verify_proposed(self, *_args, **_kwargs):
                return None

        lease = FixtureLease()

        @contextmanager
        def fixture_guard(*_args, **_kwargs):
            prior = getattr(module, "ACTIVE_RECOVERY_LEASE", None)
            module.ACTIVE_RECOVERY_LEASE = lease
            try:
                yield lease
            finally:
                module.ACTIVE_RECOVERY_LEASE = prior

        module.recovery_guard = fixture_guard
    return module


def harn_recovery_check(root: Path) -> None:
    module = load("repair2_harn_test", HARN_DIR / "rmatched_matrix_repair2_harn.py")
    harn = module.harn
    approval_hash = "b" * 64
    approval = root / "approval.json"
    approval.write_text(json.dumps({"channel_record_sha256": approval_hash, "state": "reserved"}))
    transaction = root / "transaction.json"
    transaction.write_text(json.dumps({"phase": "audio_quarantined"}))
    module.TRANSACTION = transaction
    harn.APPROVAL = approval
    harn.CURRENT = root / "current"
    ledger = {"events": [{"event_kind": "experiment_started"}]}
    contract: dict = {}
    preflight: dict = {}
    written: list[str] = []
    terminal: list[bool] = []
    responses = iter((97, 0, 0, 0))
    approval_state = {"channel_record_sha256": approval_hash, "state": "reserved"}

    module.wait_for_complete_preflight = lambda: (contract, preflight, ledger)
    module.read_approval_state = lambda: dict(approval_state)
    module.verify_approval_authority = lambda _approval: None
    module.verify_runtime_bindings = lambda _contract: []
    module.reserve_approval = lambda value: approval_state.update({**value, "state": "reserved"}) or dict(approval_state)
    module.write_approval_state = lambda value: approval_state.update(value)
    capability = root / "capability.json"
    capability.write_text("{}")
    module.issue_capability = lambda *_args: capability
    harn.acquire_lock = lambda: os.open("/dev/null", os.O_RDONLY)
    harn.append_event = lambda target, kind, **kwargs: target["events"].append({"event_kind": kind}) or kind
    harn.atomic_json = lambda path, payload: path.write_text(json.dumps(payload))
    harn.write_generation = lambda _c, _p, _l, status: written.append(status)
    harn.notify = lambda *args, **kwargs: None
    harn.terminal = lambda _c, _p, _l, success, summary: terminal.append(success)
    module.run_repair_action = lambda *args, **kwargs: SimpleNamespace(returncode=next(responses))

    try:
        module.run()
    except ChildProcessError:
        pass
    else:
        raise AssertionError("recoverable child crash did not leave the HARN")
    assert written[-1] == "held" and not terminal

    module.run()
    assert terminal == [True]
    assert approval_state["state"] == "consumed"


def reservation_window_check(root: Path) -> None:
    module = load("repair2_reservation_test", HARN_DIR / "rmatched_matrix_repair2_harn.py")
    harn = module.harn
    module.TRANSACTION = root / "transaction.json"
    module.TRANSACTION.write_text(json.dumps({"phase": "audio_quarantined"}))
    harn.acquire_lock = lambda: os.open("/dev/null", os.O_RDONLY)
    contract: dict = {}
    preflight: dict = {}
    ledger = {"events": []}
    module.wait_for_complete_preflight = lambda: (contract, preflight, ledger)
    state = {"channel_record_sha256": "b" * 64, "state": "approved"}
    module.read_approval_state = lambda: dict(state)
    module.reserve_approval = lambda value: state.update({**value, "state": "reserved"}) or dict(state)
    module.write_approval_state = lambda value: state.update(value)
    module.verify_approval_authority = lambda _approval: None
    module.verify_runtime_bindings = lambda _contract: []
    capability = root / "capability.json"
    capability.write_text("{}")
    module.issue_capability = lambda *_args: capability
    harn.append_event = lambda target, kind, **kwargs: target["events"].append({"event_kind": kind}) or kind
    harn.write_generation = lambda *_args: None
    harn.notify = lambda *_args, **_kwargs: None
    terminal: list[bool] = []
    harn.terminal = lambda _c, _p, _l, success, _summary: terminal.append(success)
    module.run_repair_action = lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    module.INIT_FAULT_AFTER = "approval_reserved"
    try:
        try:
            module.run()
        except ChildProcessError:
            pass
        else:
            raise AssertionError("approval reservation crash window did not fire")
        assert state["state"] == "reserved" and not terminal and not ledger["events"]
        module.INIT_FAULT_AFTER = None
        module.run()
        assert terminal == [True] and state["state"] == "consumed"
    finally:
        pass


def top_controller_restart_check() -> None:
    controller = load("top_controller_test", HARN_DIR / "operator_queue_controller.py")
    launched: list[str] = []
    controller.child_queue_entry = lambda entry: {
        "status": "held", "terminal_notification_status": "not_applicable",
    }
    controller.notify_once = lambda *args, **kwargs: None
    def fake_launch(entry: dict) -> None:
        launched.append(entry["experiment_id"])
        entry["status"] = "active"
        entry["runtime"]["status_source"] = "/tmp/repair2/current"
    controller.launch = fake_launch
    payload = {
        "active_experiment": {"experiment_id": "repair2"},
        "entries": [{
            "position": 1, "experiment_id": "repair2", "run_id": "r1", "status": "active",
            "approval_status": "approved", "dependencies": [], "ordering_dependencies": [],
            "runtime": {"controller_pid": 999999999, "controller_start_ticks": "missing"},
        }],
    }
    changed, status = controller.transition_once(payload)
    assert changed and status["state"] == "running" and launched == ["repair2"]
    assert payload["entries"][0]["status"] == "active"


def canonical(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def queue_security_checks(root: Path) -> None:
    controller = load("queue_security_test", HARN_DIR / "operator_queue_controller.py")
    key = hashlib.sha256(b"queue-security-fixture").digest()
    controller.QUEUE_KEY = root / "queue.key"
    controller.QUEUE_KEY.write_bytes(key)
    os.chmod(controller.QUEUE_KEY, 0o600)
    contract = root / "contract.json"
    executable = root / "controller.py"
    contract.write_text("{}")
    executable.write_text("# fixture\n")
    validator = {"argv": ["/bin/true"], "bindings": [{
        "path": "/bin/true", "sha256": controller.digest_file(Path("/bin/true")),
    }]}
    entry = {
        "position": 1, "experiment_id": "repair2", "run_id": "r1", "status": "queued",
        "approval_status": "approved", "contract": str(contract),
        "contract_sha256": controller.digest_file(contract), "dependencies": [], "ordering_dependencies": [],
        "runtime": {"controller": str(executable), "controller_sha256": controller.digest_file(executable),
                    "status_source": str(root / "state/current"), "completion_validator": validator},
    }
    approval = controller.sign_document({
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": "repair2", "run_id": "r1", "contract_sha256": entry["contract_sha256"],
        "controller_sha256": entry["runtime"]["controller_sha256"],
        "queue_entry_binding_sha256": controller.entry_approval_binding(entry),
        "channel_record_sha256": "a" * 64,
    }, b"meanaudio-queue-approval-v1\0", key)
    approval_path = root / "approval.json"
    approval_path.write_text(json.dumps(approval))
    entry["approval_evidence"] = {"path": str(approval_path), "sha256": controller.digest_file(approval_path)}
    queue = controller.sign_document({
        "document_kind": "operator_approved_experiment_backlog", "schema_version": 1,
        "entries": [entry],
    }, b"meanaudio-operator-queue-v1\0", key)
    controller.validate_queue(queue, authenticate=True)
    forged = json.loads(json.dumps(queue))
    forged["entries"][0]["runtime"]["completion_validator"]["argv"] = ["/bin/false"]
    try:
        controller.validate_queue(forged, authenticate=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsigned validator injection was accepted")
    assert controller.completion_evidence_valid(entry)
    assert not controller.completion_evidence_valid({"runtime": {}})
    injected = json.loads(json.dumps(entry))
    injected["runtime"]["completion_validator"]["argv"] = ["/bin/false"]
    assert not controller.completion_evidence_valid(injected)
    drift_file = root / "validator"
    drift_file.write_text("v1")
    drift_entry = {"runtime": {"completion_validator": {"argv": [str(drift_file)], "bindings": [{
        "path": str(drift_file), "sha256": controller.digest_file(drift_file),
    }]}}}
    drift_file.write_text("v2")
    assert not controller.completion_evidence_valid(drift_entry)
    prior_path = os.environ.get("PATH")
    prior_ld = os.environ.get("LD_LIBRARY_PATH")
    os.environ["PYTHONPATH"] = "/attacker"
    os.environ["MEANAUDIO_NOTIFY_DRY_RUN"] = "1"
    os.environ["PATH"] = "/attacker"
    os.environ["LD_LIBRARY_PATH"] = "/attacker/lib"
    try:
        safe = controller.safe_child_env()
    finally:
        os.environ.pop("PYTHONPATH", None)
        os.environ.pop("MEANAUDIO_NOTIFY_DRY_RUN", None)
        if prior_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = prior_path
        if prior_ld is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = prior_ld
    assert ("PYTHONPATH" not in safe and "MEANAUDIO_NOTIFY_DRY_RUN" not in safe
            and "LD_LIBRARY_PATH" not in safe and safe["PATH"] == controller.CONTRACT_PATH
            and safe["PYTHONDONTWRITEBYTECODE"] == "1")
    swap = root / "swap-controller.py"
    swap.write_text("approved-bytes")
    approved_hash = controller.digest_file(swap)
    fd = controller.open_verified_fd(swap, approved_hash)
    try:
        replacement = root / "replacement-controller.py"
        replacement.write_text("hostile-bytes")
        os.replace(replacement, swap)
        assert os.pread(fd, 100, 0) == b"approved-bytes"
    finally:
        os.close(fd)
    verified_controller = root / "verified-controller.py"
    verified_shared = root / "verified-shared.py"
    verified_controller.write_text("import verified_shared\nprint(verified_shared.VALUE)\n")
    verified_shared.write_text("VALUE = 'approved-import-bytes'\n")
    runtime_helper_path = HARN_DIR / "runtime_binding.py"
    runtime_helper = load("runtime_binding_fixture", runtime_helper_path)
    runtime_tree = root / "runtime-tree"
    runtime_tree.mkdir()
    (runtime_tree / "module.py").write_text("VALUE = 1\n")
    tree_hash, pth_files, file_count = runtime_helper.tree_sha256(runtime_tree)
    runtime_manifest_path = root / "runtime-manifest.json"
    runtime_manifest_path.write_text(json.dumps({
        "document_kind": "meanaudio_transitive_runtime_manifest", "schema_version": 1,
        "tree_binding_rules": runtime_helper.TREE_BINDING_RULES,
        "entries": [
            {"kind": "file", "role": "system_python", "path": "/usr/bin/python3.12",
             "sha256": controller.digest_file(Path("/usr/bin/python3.12"))},
            {"kind": "tree", "role": "fixture_tree", "path": str(runtime_tree),
             "sha256": tree_hash, "file_count": file_count, "pth_files": pth_files},
        ],
    }))
    invocation_entry = {"runtime": {
        "controller": str(verified_controller), "controller_sha256": controller.digest_file(verified_controller),
        "import_bindings": [{"module": "verified_shared", "path": str(verified_shared),
                             "sha256": controller.digest_file(verified_shared)}],
        "transitive_runtime": {"manifest": str(runtime_manifest_path),
            "manifest_sha256": controller.digest_file(runtime_manifest_path),
            "required_roles": ["system_python", "fixture_tree"],
            "verifier": {"path": str(runtime_helper_path),
                         "sha256": controller.digest_file(runtime_helper_path)}},
    }}
    argv, fds = controller.verified_controller_invocation(invocation_entry, [])
    try:
        controller_replacement = root / "controller-replacement.py"
        shared_replacement = root / "shared-replacement.py"
        controller_replacement.write_text("print('hostile-controller')\n")
        shared_replacement.write_text("VALUE = 'hostile-import'\n")
        os.replace(controller_replacement, verified_controller)
        os.replace(shared_replacement, verified_shared)
        executed = subprocess.run(argv, pass_fds=fds, env=controller.safe_child_env(), text=True, capture_output=True)
        assert executed.returncode == 0 and executed.stdout.strip() == "approved-import-bytes"
    finally:
        for opened_fd in fds:
            os.close(opened_fd)

    # A syntactically plausible terminal child queue is rejected unless every
    # contract/preflight/ledger hash edge matches the exact generation bundle.
    state = root / "state"
    generation = state / "generations/gen-000001"
    generation.mkdir(parents=True)
    current = state / "current"
    current.write_text(str(generation))
    bundle_entry = json.loads(json.dumps(entry))
    bundle_entry["runtime"]["status_source"] = str(current)
    bundle_entry["runtime"]["bundle_validator"] = {
        "python": "/bin/true", "python_sha256": controller.digest_file(Path("/bin/true")),
        "path": "/bin/true", "sha256": controller.digest_file(Path("/bin/true")),
    }
    child_key = root / "child-ledger.key"
    child_key.write_bytes(hashlib.sha256(b"child-ledger-key").digest())
    os.chmod(child_key, 0o600)
    bundle_entry["runtime"]["child_ledger_key"] = {
        "path": str(child_key), "sha256": controller.digest_file(child_key),
    }
    child_contract = {"experiment_id": "repair2", "run_id": "r1",
        "bindings": {"runtime_sha256": bundle_entry["runtime"]["controller_sha256"]},
        "corpus": {"source_artifacts": [
            {"path": bundle_entry["contract"], "sha256": bundle_entry["contract_sha256"]},
            {"path": bundle_entry["approval_evidence"]["path"],
             "sha256": bundle_entry["approval_evidence"]["sha256"]},
        ]}}
    contract_hash = hashlib.sha256(canonical(child_contract)).hexdigest()
    child_preflight = {"experiment_id": "repair2", "run_id": "r1", "contract_raw_sha256": contract_hash}
    preflight_hash = hashlib.sha256(canonical(child_preflight)).hexdigest()
    terminal_event = {"sequence": 1, "event_kind": "experiment_completed", "previous_event_sha256": None}
    terminal_event["event_sha256"] = hmac.new(
        child_key.read_bytes(), b"meanaudio-harn-event-v1\0" + canonical(terminal_event), hashlib.sha256,
    ).hexdigest()
    child_ledger = {"experiment_id": "repair2", "run_id": "r1", "bindings": {
        "contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
    }, "events": [terminal_event]}
    ledger_hash = hashlib.sha256(canonical(child_ledger)).hexdigest()
    child = {"experiment_id": "repair2", "run_id": "r1", "status": "completed",
             "terminal_notification_status": "delivered", "bindings": {
                 "contract_raw_sha256": contract_hash, "preflight_report_raw_sha256": preflight_hash,
                 "ledger_raw_sha256": ledger_hash,
             }}
    for name, value in (("contract", child_contract), ("preflight", child_preflight),
                        ("ledger", child_ledger), ("queue", {"entries": [child]})):
        (generation / f"{name}.json").write_text(json.dumps(value))
    assert controller.child_queue_entry(bundle_entry)["status"] == "completed"
    child_ledger["events"][0]["event_sha256"] = "0" * 64
    forged_ledger_hash = hashlib.sha256(canonical(child_ledger)).hexdigest()
    child["bindings"]["ledger_raw_sha256"] = forged_ledger_hash
    (generation / "ledger.json").write_text(json.dumps(child_ledger))
    (generation / "queue.json").write_text(json.dumps({"entries": [child]}))
    assert controller.child_queue_entry(bundle_entry) is None


def init_fault_recovery_check(root: Path) -> None:
    for fault in ("approval_written", "harn_state_written"):
        fixture = root / fault
        fixture.mkdir()
        module = load(f"repair2_init_{fault}", HARN_DIR / "rmatched_matrix_repair2_harn.py")
        harn = module.harn
        module.STATE = fixture / "state"
        module.ARCHIVE = module.STATE / "archive"
        module.CAPABILITIES = module.STATE / "capabilities"
        for name in ("GENERATIONS", "OUTBOX", "CURRENT", "KEY", "LOCK", "APPROVAL", "PENDING_CONTRACT",
                     "PENDING_PREFLIGHT", "PENDING_LEDGER", "PENDING_QUEUE"):
            suffix = name.lower()
            setattr(harn, name, module.STATE / suffix)
        record_path = fixture / "approval.json"
        record_path.write_text("{}")
        record = {"channel_record_sha256": "b" * 64}
        module.verify_queue_approval = lambda *_args: ({}, record)
        module.verify_preregistered_descriptors = lambda: None
        module.archive_executable = lambda: module.REPAIR
        module.verify_runtime_bindings = lambda _contract: []
        module.make_contract = lambda: {"bindings": {"runtime_sha256": "c" * 64,
            "schema_bundle_sha256": "d" * 64}, "required_preflight_checks": []}
        harn.blocking_gpu_processes = lambda: ["held"]
        harn.make_preflight = lambda *_args: {"derived_verdict": "fail"}
        harn.append_event = lambda ledger, kind, **kwargs: ledger["events"].append({"event_kind": kind}) or kind
        module.INIT_FAULT_AFTER = fault
        try:
            module.init(record_path, "e" * 64)
        except ChildProcessError:
            pass
        else:
            raise AssertionError(f"init fault point {fault} did not fire")
        module.INIT_FAULT_AFTER = None
        module.init(record_path, "e" * 64)
        assert harn.APPROVAL.is_file()
        assert harn.PENDING_CONTRACT.is_file() or harn.CURRENT.is_file()
    fixture = root / "repair1-rejected"
    fixture.mkdir()
    module = load("repair2_init_repair1", HARN_DIR / "rmatched_matrix_repair2_harn.py")
    harn = module.harn
    module.STATE = fixture / "state"
    module.ARCHIVE = module.STATE / "archive"
    module.CAPABILITIES = module.STATE / "capabilities"
    for name in ("GENERATIONS", "OUTBOX", "CURRENT", "KEY", "LOCK", "APPROVAL", "PENDING_CONTRACT",
                 "PENDING_PREFLIGHT", "PENDING_LEDGER", "PENDING_QUEUE"):
        setattr(harn, name, module.STATE / name.lower())
    record_path = fixture / "approval.json"
    record_path.write_text("{}")
    module.verify_queue_approval = lambda *_args: ({}, {"channel_record_sha256": module.REPAIR1_APPROVAL_SHA256})
    try:
        module.init(record_path, "e" * 64)
    except RuntimeError as exc:
        assert "repair1" in str(exc).lower()
    else:
        raise AssertionError("Repair1 approval initialized Repair2")


def continuation_direct_init_check(root: Path) -> None:
    module = load("continuation_auth_test", HARN_DIR / "rmatched_matrix_continuation_harn.py")
    old_path, old_ld = os.environ.get("PATH"), os.environ.get("LD_LIBRARY_PATH")
    os.environ["PATH"] = "/hostile"
    os.environ["LD_LIBRARY_PATH"] = "/hostile/lib"
    try:
        environment = module.safe_env()
    finally:
        if old_path is None: os.environ.pop("PATH", None)
        else: os.environ["PATH"] = old_path
        if old_ld is None: os.environ.pop("LD_LIBRARY_PATH", None)
        else: os.environ["LD_LIBRARY_PATH"] = old_ld
    assert environment["PATH"] == module.CONTRACT_PATH and "LD_LIBRARY_PATH" not in environment
    key = hashlib.sha256(b"continuation-queue-key").digest()
    module.QUEUE_KEY = root / "queue.key"
    module.QUEUE = root / "queue.json"
    module.QUEUE_KEY.write_bytes(key)
    os.chmod(module.QUEUE_KEY, 0o600)
    contract = module.PREREG
    controller_path = HARN_DIR / "rmatched_matrix_continuation_harn.py"
    entry = {"position": 1, "experiment_id": module.harn.EXPERIMENT, "run_id": module.harn.RUN_ID,
        "approval_status": "approved", "status": "queued", "contract": str(contract),
        "contract_sha256": module.harn.digest_file(contract), "dependencies": [], "ordering_dependencies": [],
        "runtime": {"controller": str(controller_path), "controller_sha256": module.harn.digest_file(controller_path),
                    "status_source": str(root / "state/current"), "transitive_runtime": {
                        "manifest": str(module.RUNTIME_MANIFEST),
                        "manifest_sha256": module.harn.digest_file(module.RUNTIME_MANIFEST),
                        "required_roles": ["system_python"],
                        "verifier": {"path": str(module.RUNTIME_BINDING),
                                     "sha256": module.harn.digest_file(module.RUNTIME_BINDING)},
                    }}}
    binding = module.entry_binding(entry)
    record_path = root / "approval.json"
    unsigned = {"document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": module.harn.EXPERIMENT, "run_id": module.harn.RUN_ID,
        "contract_sha256": entry["contract_sha256"], "controller_sha256": entry["runtime"]["controller_sha256"],
        "queue_entry_binding_sha256": binding, "channel_record_sha256": "a" * 64}
    record_path.write_text(json.dumps(unsigned))
    entry["approval_evidence"] = {"path": str(record_path), "sha256": module.harn.digest_file(record_path)}
    queue = signed_for_test({"document_kind": "operator_approved_experiment_backlog", "entries": [entry]}, key,
                            b"meanaudio-operator-queue-v1\0")
    module.QUEUE.write_text(json.dumps(queue))
    try:
        module.verify_queue_approval(record_path, binding)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsigned continuation approval was accepted")
    wrong = signed_for_test({**unsigned, "run_id": "wrong-run"}, key, b"meanaudio-queue-approval-v1\0")
    record_path.write_text(json.dumps(wrong))
    entry["approval_evidence"]["sha256"] = module.harn.digest_file(record_path)
    module.QUEUE.write_text(json.dumps(signed_for_test(
        {"document_kind": "operator_approved_experiment_backlog", "entries": [entry]}, key,
        b"meanaudio-operator-queue-v1\0")))
    try:
        module.verify_queue_approval(record_path, binding)
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong-run continuation approval was accepted")
    completed = subprocess.run(
        [sys.executable, str(controller_path), "init", "--approval-text-hash", "a" * 64],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode != 0 and "--approval-record" in completed.stderr
    correct = signed_for_test(unsigned, key, b"meanaudio-queue-approval-v1\0")
    record_path.write_text(json.dumps(correct))
    entry["approval_evidence"]["sha256"] = module.harn.digest_file(record_path)
    module.QUEUE.write_text(json.dumps(signed_for_test(
        {"document_kind": "operator_approved_experiment_backlog", "entries": [entry]}, key,
        b"meanaudio-operator-queue-v1\0")))
    state = root / "continuation-state"
    module.STATE = state
    harn = module.harn
    for name in ("GENERATIONS", "OUTBOX", "CURRENT", "KEY", "LOCK", "APPROVAL", "PENDING_CONTRACT",
                 "PENDING_PREFLIGHT", "PENDING_LEDGER", "PENDING_QUEUE"):
        setattr(harn, name, state / name.lower())
    module.make_contract = lambda: {"bindings": {"schema_bundle_sha256": "d" * 64},
                                    "required_preflight_checks": []}
    harn.blocking_gpu_processes = lambda: ["held"]
    harn.make_preflight = lambda *_args: {"derived_verdict": "fail"}
    harn.append_event = lambda ledger, kind, **kwargs: ledger["events"].append({"event_kind": kind}) or kind
    module.init(record_path, binding)
    approval_state = module.read_approval_state()
    assert approval_state["state"] == "approved" and approval_state["run_id"] == module.harn.RUN_ID


def signed_for_test(payload: dict, key: bytes, domain: bytes) -> dict:
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    return {**unsigned, "integrity": hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()}


def runtime_manifest_mutation_check(root: Path) -> None:
    runtime = load("runtime_binding_mutation", HARN_DIR / "runtime_binding.py")
    tree = root / "site-packages"
    tree.mkdir()
    module = tree / "package.py"
    pth = tree / "injection.pth"
    evaluator = root / "evaluator.py"
    module.write_text("VALUE = 1\n")
    pth.write_text("/approved/path\n")
    evaluator.write_text("print('approved')\n")

    def make_manifest() -> tuple[Path, str]:
        tree_hash, pths, count = runtime.tree_sha256(tree)
        path = root / "manifest.json"
        path.write_text(json.dumps({"document_kind": "meanaudio_transitive_runtime_manifest", "schema_version": 1,
            "tree_binding_rules": runtime.TREE_BINDING_RULES,
            "entries": [
                {"kind": "tree", "role": "site", "path": str(tree), "sha256": tree_hash,
                 "file_count": count, "pth_files": pths},
                {"kind": "file", "role": "evaluator", "path": str(evaluator),
                 "sha256": runtime.sha256_file(evaluator)},
            ]}))
        return path, runtime.sha256_file(path)

    for target, hostile in ((pth, "import attacker\n"), (module, "VALUE = 'attacker'\n"),
                            (evaluator, "print('attacker')\n")):
        path, expected = make_manifest()
        original = target.read_text()
        target.write_text(hostile)
        try:
            runtime.verify_manifest(path, expected, {"site", "evaluator"})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"runtime mutation was accepted before GPU: {target}")
        target.write_text(original)
    path, expected = make_manifest()
    path.write_text(path.read_text() + " ")
    try:
        runtime.verify_manifest(path, expected, {"site", "evaluator"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("runtime manifest mutation was accepted before GPU")

    cache_dir = tree / "__pycache__"
    cache_dir.mkdir()
    cache_file = cache_dir / "package.cpython-312.pyc"
    before = runtime.tree_sha256(tree)
    cache_file.write_bytes(b"derived cache v1")
    after_create = runtime.tree_sha256(tree)
    cache_file.write_bytes(b"derived cache v2")
    after_mutation = runtime.tree_sha256(tree)
    assert before == after_create == after_mutation


def continuation_launcher_and_shadow_isolation_check(root: Path) -> None:
    """A retargeted venv launcher and repo-root module cannot affect execution."""
    bootstrap = HARN_DIR / "isolated_python_bootstrap.py"
    continuation = HARN_DIR / "rmatched_matrix_continuation_harn.py"
    runner = ROOT / "scripts/eval/eval_rmatched_s1_s2_steps_cfg_matrix_continuation.sh"
    assert "/venvs/dac/bin/python" not in continuation.read_text()
    assert "/venvs/dac/bin/python" not in runner.read_text()
    assert "/venvs/peav/bin/python" not in runner.read_text()

    fake_venv = root / "venv/bin"
    fake_venv.mkdir(parents=True)
    launcher = fake_venv / "python"
    launcher.symlink_to("/bin/false")
    site_packages = root / "exact-site-packages"
    package_dir = root / "meanaudio"
    site_packages.mkdir()
    package_dir.mkdir()
    (root / "torch.py").write_text("raise RuntimeError('repo-root shadow imported')\n")
    (site_packages / "torch.py").write_text("ORIGIN = 'exact-site-packages'\n")
    (site_packages / "torch.pyc").write_bytes(b"hostile legacy bytecode")
    package_init = package_dir / "__init__.py"
    package_init.write_text("ORIGIN = 'exact-meanaudio-package'\n")
    script = root / "script.py"
    script.write_text(
        "import os, sys, torch, meanaudio\n"
        "assert torch.ORIGIN == 'exact-site-packages'\n"
        "assert meanaudio.ORIGIN == 'exact-meanaudio-package'\n"
        "assert os.getcwd() not in sys.path\n"
        "assert %r not in sys.path\n"
        "print(torch.ORIGIN + ':' + meanaudio.ORIGIN)\n" % str(root)
    )

    def execute() -> subprocess.CompletedProcess[str]:
        paths = [Path("/usr/bin/python3.12"), bootstrap, package_init, script]
        fds = [os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)) for path in paths]
        try:
            python_fd, bootstrap_fd, package_fd, script_fd = fds
            argv = [
                f"/proc/self/fd/{python_fd}", "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                f"/proc/self/fd/{bootstrap_fd}",
                "--site-packages", str(site_packages), "--package-name", "meanaudio",
                "--package-init-fd", str(package_fd), "--package-dir", str(package_dir),
                "--script-fd", str(script_fd), "--display", str(script), "--",
            ]
            assert str(launcher) not in argv
            return subprocess.run(
                argv, cwd=root, env={"PATH": "/usr/bin:/bin"}, pass_fds=tuple(fds),
                text=True, capture_output=True,
            )
        finally:
            for fd in fds:
                os.close(fd)

    first = execute()
    assert first.returncode == 0, first.stderr
    launcher.unlink()
    launcher.symlink_to("/bin/echo")
    second = execute()
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout == "exact-site-packages:exact-meanaudio-package\n"

    # Import the actual production networks module under the same bootstrap.
    # A cwd/repository-style torch.py shadow must remain unreachable.
    production_script = root / "production-networks-import.py"
    production_script.write_text(
        "import os, sys, torch\n"
        "from meanaudio.model import networks\n"
        "repo = '/home/kojiek/MeanAudio'\n"
        "shadow = os.path.join(os.getcwd(), 'torch.py')\n"
        "assert repo not in sys.path and os.getcwd() not in sys.path\n"
        "assert os.path.realpath(torch.__file__) != os.path.realpath(shadow)\n"
        "assert os.path.realpath(networks.__file__) == repo + '/meanaudio/model/networks.py'\n"
        "assert networks.__cached__ is None\n"
        "assert hasattr(networks, 'MeanAudio') and hasattr(networks, 'FluxAudio')\n"
        "print('production-networks-isolated')\n"
    )
    real_package_dir = ROOT / "meanaudio"
    real_init = real_package_dir / "__init__.py"
    real_site = Path("/home/kojiek/venvs/dac/lib/python3.12/site-packages")
    paths = [Path("/usr/bin/python3.12"), bootstrap, real_init, production_script]
    fds = [os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)) for path in paths]
    try:
        python_fd, bootstrap_fd, package_fd, script_fd = fds
        real = subprocess.run([
            f"/proc/self/fd/{python_fd}", "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
            f"/proc/self/fd/{bootstrap_fd}",
            "--site-packages", str(real_site), "--package-name", "meanaudio",
            "--package-init-fd", str(package_fd), "--package-dir", str(real_package_dir),
            "--script-fd", str(script_fd), "--display", str(production_script), "--",
        ], cwd=root, env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""},
            pass_fds=tuple(fds), text=True, capture_output=True)
    finally:
        for fd in fds:
            os.close(fd)
    assert real.returncode == 0, real.stderr
    assert real.stdout == "production-networks-isolated\n"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="repair2-harn-recovery-") as raw:
        root = Path(raw)
        run_root = root / "run-recovery"
        run_root.mkdir()
        harn_recovery_check(run_root)
        reservation_root = root / "reservation-recovery"
        reservation_root.mkdir()
        reservation_window_check(reservation_root)
        queue_root = root / "queue-security"
        queue_root.mkdir()
        queue_security_checks(queue_root)
        init_root = root / "init-recovery"
        init_root.mkdir()
        init_fault_recovery_check(init_root)
        continuation_root = root / "continuation-auth"
        continuation_root.mkdir()
        continuation_direct_init_check(continuation_root)
        runtime_root = root / "runtime-mutations"
        runtime_root.mkdir()
        runtime_manifest_mutation_check(runtime_root)
        isolation_root = root / "continuation-import-isolation"
        isolation_root.mkdir()
        continuation_launcher_and_shadow_isolation_check(isolation_root)
    top_controller_restart_check()
    print("[PASS] repair2 HARN/approval recovery, signed queue, validators, forged child rejection, env scrub, relaunch, continuation import isolation")


if __name__ == "__main__":
    main()
