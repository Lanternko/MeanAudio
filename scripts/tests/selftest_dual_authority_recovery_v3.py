#!/usr/bin/env python3
"""Temp-only abuse and determinism tests for PLAN-DUAL-AUTH-RECOVERY-V3."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/dual_authority_recovery_v3.py"
HARN_DIR = ROOT / "scripts/experiment_harness"
SURFACES = {
    ROOT / "scripts/experiment_harness/operator_queue_controller.py": (
        "atomic_json", "atomic_queue", "load_queue", "validate_approval_record",
        "completion_evidence_valid", "notify_once", "launch", "run", "enqueue", "approve",
    ),
    ROOT / "scripts/experiment_harness/rmatched_matrix_nvme_stage_harn.py": (
        "atomic_json", "init", "notify_once", "finalize_completed", "run",
        "validate_completed_bundle",
    ),
    ROOT / "scripts/experiment_harness/rmatched_matrix_repair2_harn.py": (
        "safe_notify", "ensure_harn_key", "archive_executable", "run_repair_action",
        "write_approval_state", "issue_capability", "validate_stage_dependency", "init",
        "recoverable_transaction", "wait_for_complete_preflight", "run",
    ),
    ROOT / "scripts/experiment_harness/rmatched_matrix_continuation_harn.py": (
        "ensure_harn_key", "write_approval_state", "safe_notify", "init",
        "wait_for_complete_preflight", "validate_repair_dependency", "run_continuation", "run",
    ),
}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    os.chmod(path, mode)


def assert_common_gate_surface() -> None:
    """Every reviewed write/process/notification entrypoint must fail through one common gate."""
    for path, required in SURFACES.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in required:
            function = functions.get(name)
            assert function is not None, f"missing reviewed entrypoint: {path}:{name}"
            gates = [
                call for call in ast.walk(function)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in {"recovery_gate", "recovery_gate_for_entry"}
            ]
            assert gates, f"common recovery gate missing: {path}:{name}"
            first_gate = min(call.lineno for call in gates)
            prior_calls = [call for call in ast.walk(function)
                           if isinstance(call, ast.Call) and call.lineno < first_gate]
            forbidden = {
                "atomic_json", "atomic_queue", "mkdir", "chmod", "replace", "unlink",
                "Popen", "run", "notify", "notify_once", "write_approval_state",
                "issue_capability", "launch", "archive_executable", "ensure_harn_key",
            }
            def call_name(call: ast.Call) -> str:
                return (call.func.id if isinstance(call.func, ast.Name)
                        else call.func.attr if isinstance(call.func, ast.Attribute) else "")
            assert not [call for call in prior_calls if call_name(call) in forbidden], (
                f"write/process/notification occurs before common recovery gate: {path}:{name}"
            )


class Fixture:
    def __init__(self, root: Path, name: str = "normal") -> None:
        self.root = root / name
        self.root.mkdir(mode=0o700)
        self.module = load(f"dual_authority_{name}")
        self.key = hashlib.sha256(f"fixture-key:{name}".encode()).digest()
        self.queue_key = self.root / "backlog/queue_hmac.key"
        self.queue = self.root / "backlog/queue.json"
        self.approvals = self.root / "backlog/approvals"
        self.cutover = self.root / "controller/cutovers/recovery"
        self.instruction = self.approvals / "recovery.txt"
        self.recovery_record = self.approvals / "recovery-authorization.json"
        self.archive_manifest = self.cutover / "archive_manifest.json"
        self.prior_queue = self.cutover / "revoked/queue.before.json"
        write(self.queue_key, self.key)
        recovery_bytes = f"exact recovery authority {name}".encode()
        write(self.instruction, recovery_bytes, 0o400)
        write(self.prior_queue, b"prior signed queue fixture")

        self.contracts: dict[str, Path] = {}
        self.controllers: dict[str, Path] = {}
        experiments = (("stage", "stage-run"), ("repair2", "repair2-run"),
                       ("continuation", "continuation-run"))
        for experiment, run in experiments:
            contract = self.root / f"workspace/{experiment}-contract.json"
            controller = self.root / f"workspace/{experiment}-controller.py"
            write(contract, json.dumps({"experiment_id": experiment, "run_id": run}).encode(), 0o600)
            write(controller, f"# {experiment}\n".encode(), 0o600)
            self.contracts[experiment] = contract
            self.controllers[experiment] = controller

        lock_roles = (("top_controller", "controller/controller.lock"),
                      ("queue_mutation", "backlog/queue_mutation.lock"),
                      ("stage_controller", "stage/controller.lock"),
                      ("repair2_controller", "repair2/controller.lock"),
                      ("continuation_controller", "continuation/controller.lock"))
        self.locks = tuple((role, str(self.root / relative)) for role, relative in lock_roles)
        for _role, path in self.locks:
            write(Path(path), b"")

        retained = (("queue_local_signer", str(self.queue_key), digest(self.queue_key),
                     "AUTH_ROOT_local_signer"),)
        for role in ("stage_child_ledger", "repair2_child_ledger", "continuation_child_ledger"):
            key_path = self.root / f"{role}/ledger_hmac.key"
            write(key_path, hashlib.sha256(role.encode()).digest())
            classification = "retained_exact_binding_key" if role == "stage_child_ledger" else "retained_child_ledger_key"
            retained += ((role, str(key_path), digest(key_path), classification),)

        module = self.module
        module.EXECUTION_AUTHORITY_SHA256 = "a" * 64
        module.RECOVERY_AUTHORITY_SHA256 = hashlib.sha256(recovery_bytes).hexdigest()
        module.RECOVERY_AUTHORITY_BYTES = len(recovery_bytes)
        module.PRIOR_QUEUE_SHA256 = digest(self.prior_queue)
        module.BLOCKER_FINGERPRINT = "fixture-blocker"
        module.EXPERIMENTS = experiments
        module.QUEUE_LAYOUT = tuple(
            (experiment, run, position,
             () if position == 1 else (experiments[position - 2][0],),
             () if position == 1 else (experiments[position - 2][0],))
            for position, (experiment, run) in enumerate(experiments, 1)
        )
        module.RETAINED_KEYS = retained
        module.EXCLUSIVE_LOCKS = self.locks
        # Preserve the production schema's complete ordered role inventory in
        # the fixture.  Fixture digests are deterministic but intentionally do
        # not grant authority over production bytes.
        production_roles = tuple(role for role, _digest in module.ORDERED_CLOSED_REVOCATIONS)
        module.ORDERED_CLOSED_REVOCATIONS = tuple(
            (role, module.PRIOR_QUEUE_SHA256 if role == "prior_signed_queue"
             else hashlib.sha256(f"fixture-revocation:{role}".encode()).hexdigest())
            for role in production_roles
        )

        archive_entry = {
            "mode": "0o600", "nlink": 1, "path": "revoked/queue.before.json",
            "sha256": digest(self.prior_queue), "size": self.prior_queue.stat().st_size,
            "uid": os.geteuid(),
        }
        archive = module.sign_for_fixture({
            "document_kind": "authenticated_failed_init_recovery_archive",
            "blocker_fingerprint": module.BLOCKER_FINGERPRINT,
            "prior_live_queue_sha256": module.PRIOR_QUEUE_SHA256,
            "recovery_channel_record": {"sha256": module.RECOVERY_AUTHORITY_SHA256},
            "entries": [archive_entry],
            "entries_tree_sha256": hashlib.sha256(module.canonical([archive_entry])).hexdigest(),
        }, self.key, module.ARCHIVE_DOMAIN)
        write(self.archive_manifest, json.dumps(archive, sort_keys=True).encode())
        module.ARCHIVE_MANIFEST_SHA256 = digest(self.archive_manifest)

        self.entries = []
        system_python = Path("/usr/bin/python3.12")
        system_python_sha256 = digest(system_python)
        for position, (experiment, run) in enumerate(experiments, 1):
            entry = {
                "position": position, "experiment_id": experiment, "run_id": run,
                "status": "queued", "approval_status": "approved",
                "contract": str(self.contracts[experiment]),
                "contract_sha256": digest(self.contracts[experiment]),
                "dependencies": [] if position == 1 else [experiments[position - 2][0]],
                "ordering_dependencies": [] if position == 1 else [experiments[position - 2][0]],
                "runtime": {
                    "controller": str(self.controllers[experiment]),
                    "controller_sha256": digest(self.controllers[experiment]),
                    "status_source": str(self.root / f"{experiment}/current"),
                    "pending_status_source": str(self.root / f"{experiment}/pending_queue.json"),
                    "approval_cli": "exact_record_v1",
                    "approval_text_sha256": module.EXECUTION_AUTHORITY_SHA256,
                    "bundle_validator": {
                        "path": str(controller), "sha256": digest(controller),
                        "python": str(system_python), "python_sha256": system_python_sha256,
                    },
                    "completion_validator": {
                        "path": str(controller), "sha256": digest(controller),
                        "bindings": [{"path": str(system_python), "sha256": system_python_sha256}],
                    },
                },
            }
            self.entries.append(entry)

        replacements = [{
            "experiment_id": entry["experiment_id"], "run_id": entry["run_id"],
            "contract_sha256": entry["contract_sha256"],
            "controller_sha256": entry["runtime"]["controller_sha256"],
            "queue_entry_binding_sha256": module.entry_binding(entry),
        } for entry in self.entries]
        self.recovery_payload = module.sign_for_fixture({
            "document_kind": "dual_authority_recovery_authorization", "schema_version": 3,
            "status": "approved", "plan_fingerprint": module.PLAN_FINGERPRINT,
            "authorization_mode": "all_of", "blocker_fingerprint": module.BLOCKER_FINGERPRINT,
            "prior_queue_sha256": module.PRIOR_QUEUE_SHA256,
            "execution_authority": {
                "role": "scientific_execution", "channel_record_sha256": module.EXECUTION_AUTHORITY_SHA256,
                "hmac_domain": "meanaudio-queue-approval-v1",
            },
            "recovery_authority": {
                "role": "binding_recovery", "channel_record_sha256": module.RECOVERY_AUTHORITY_SHA256,
                "hmac_domain": "meanaudio-dual-authority-recovery-v3",
                "instruction_evidence": {"path": str(self.instruction),
                                         "sha256": module.RECOVERY_AUTHORITY_SHA256,
                                         "byte_length": module.RECOVERY_AUTHORITY_BYTES},
            },
            "ordered_closed_revocations": module._expected_revocations(),
            "affected_runs": [{"experiment_id": experiment, "run_id": run} for experiment, run in experiments],
            "archive_manifest": {"path": str(self.archive_manifest),
                                 "sha256": module.ARCHIVE_MANIFEST_SHA256},
            "retained_keys": module._expected_keys(), "exclusive_locks": module._expected_locks(),
            "replacement_entries": replacements,
        }, self.key, module.RECOVERY_DOMAIN)
        write(self.recovery_record, json.dumps(self.recovery_payload, sort_keys=True).encode())
        recovery_hash = digest(self.recovery_record)
        for entry in self.entries:
            approval_path = self.approvals / f"{entry['experiment_id']}.json"
            approval = module.sign_for_fixture({
                "document_kind": "exact_operator_approval", "status": "approved",
                "experiment_id": entry["experiment_id"], "run_id": entry["run_id"],
                "contract_sha256": entry["contract_sha256"],
                "controller_sha256": entry["runtime"]["controller_sha256"],
                "queue_entry_binding_sha256": module.entry_binding(entry),
                "channel_record_sha256": module.EXECUTION_AUTHORITY_SHA256,
                "authority_conjunction": {
                    "mode": "all_of", "execution_channel_record_sha256": module.EXECUTION_AUTHORITY_SHA256,
                    "recovery_channel_record_sha256": module.RECOVERY_AUTHORITY_SHA256,
                    "recovery_authorization_sha256": recovery_hash,
                    "plan_fingerprint": module.PLAN_FINGERPRINT,
                },
            }, self.key, module.APPROVAL_DOMAIN)
            write(approval_path, json.dumps(approval, sort_keys=True).encode())
            entry["approval_evidence"] = {"path": str(approval_path), "sha256": digest(approval_path)}
        self.queue_payload = module.sign_for_fixture({
            "document_kind": "operator_approved_experiment_backlog", "schema_version": 1,
            "recovery_authorization": {"mode": "all_of", "plan_fingerprint": module.PLAN_FINGERPRINT,
                                       "path": str(self.recovery_record), "sha256": recovery_hash},
            "entries": self.entries,
        }, self.key, module.QUEUE_DOMAIN)
        self.write_queue()

    def write_queue(self) -> None:
        write(self.queue, json.dumps(self.queue_payload, sort_keys=True).encode())

    def verify(self, experiment: str = "repair2"):
        return self.module.verify_gate(queue_path=self.queue, queue_key_path=self.queue_key,
                                       experiment_id=experiment, confinement_roots=(self.root, Path("/usr")),
                                       expected_uid=os.geteuid())

    def resign_recovery_and_queue(self) -> None:
        self.recovery_payload = self.module.sign_for_fixture(
            self.recovery_payload, self.key, self.module.RECOVERY_DOMAIN)
        write(self.recovery_record, json.dumps(self.recovery_payload, sort_keys=True).encode())
        recovery_hash = digest(self.recovery_record)
        self.queue_payload["recovery_authorization"]["sha256"] = recovery_hash
        self.queue_payload = self.module.sign_for_fixture(self.queue_payload, self.key, self.module.QUEUE_DOMAIN)
        self.write_queue()


def expect_rejected(fixture: Fixture, label: str) -> None:
    before = {str(path.relative_to(fixture.root)): digest(path) for path in fixture.root.rglob("*") if path.is_file()}
    try:
        fixture.verify()
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"abuse case accepted: {label}")
    after = {str(path.relative_to(fixture.root)): digest(path) for path in fixture.root.rglob("*") if path.is_file()}
    assert before == after, f"rejected gate mutated fixture: {label}"


def operational_entrypoints_fail_closed(fixtures: list[tuple[str, Fixture]], temp_root: Path,
                                        valid_fixture: Fixture) -> None:
    if str(HARN_DIR) not in sys.path:
        sys.path.insert(0, str(HARN_DIR))

    def load_surface(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    top = load_surface("dual_surface_top", HARN_DIR / "operator_queue_controller.py")
    stage = load_surface("dual_surface_stage", HARN_DIR / "rmatched_matrix_nvme_stage_harn.py")
    repair = load_surface("dual_surface_repair", HARN_DIR / "rmatched_matrix_repair2_harn.py")
    continuation = load_surface("dual_surface_continuation", HARN_DIR / "rmatched_matrix_continuation_harn.py")
    entry = {"experiment_id": "repair2", "approval_evidence": {"path": str(temp_root / "approval")}}
    surfaces = {
        top: (
            lambda: top.atomic_json(temp_root / "top.json", {}), lambda: top.run(True),
            lambda: top.enqueue(temp_root / "entry.json", None, None),
            lambda: top.approve("repair2", temp_root / "approval"),
            lambda: top.validate_approval_record(entry), lambda: top.completion_evidence_valid(entry),
            lambda: top.launch(entry), lambda: top.notify_once("fixture", "fixture"),
        ),
        stage: (
            lambda: stage.atomic_json(temp_root / "stage.json", {}),
            lambda: stage.init(temp_root / "approval", "0" * 64), lambda: stage.run(),
            lambda: stage.validate_completed_bundle(),
            lambda: stage.notify_once("fixture", "fixture", "held"),
            lambda: stage.finalize_completed({}, {}, {}),
        ),
        repair: (
            lambda: repair.init(temp_root / "approval", "0" * 64), lambda: repair.run(),
            lambda: repair.watch(True), lambda: repair.recoverable_transaction({}, {}),
            lambda: repair.run_repair_action("reconcile_repair", env={}),
            lambda: repair.issue_capability("apply", {}, {}), lambda: repair.write_approval_state({}),
            lambda: repair.safe_notify("fixture", "fixture"),
        ),
        continuation: (
            lambda: continuation.init(temp_root / "approval", "0" * 64),
            lambda: continuation.run(), lambda: continuation.watch(True),
            lambda: continuation.run_continuation({}), lambda: continuation.write_approval_state({}),
            lambda: continuation.safe_notify("fixture", "fixture"),
        ),
    }
    for label, fixture in fixtures:
        before = {str(path.relative_to(fixture.root)): digest(path)
                  for path in fixture.root.rglob("*") if path.is_file()}
        for module, calls in surfaces.items():
            @contextmanager
            def rejected_guard(*_args, **_kwargs):
                with fixture.module.guarded_action(
                        "repair2", queue_path=fixture.queue, queue_key_path=fixture.queue_key,
                        confinement_roots=(fixture.root, Path("/usr")), expected_uid=os.geteuid()):
                    raise AssertionError(f"invalid operational gate yielded: {label}")
                    yield  # pragma: no cover - keeps this a context manager
            module.recovery_guard = rejected_guard
            for call in calls:
                try:
                    call()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"operational entrypoint accepted invalid {label}")
        after = {str(path.relative_to(fixture.root)): digest(path)
                 for path in fixture.root.rglob("*") if path.is_file()}
        assert before == after, f"operational rejection mutated fixture: {label}"
        assert not any(temp_root.iterdir()), f"operational rejection performed an action: {label}"

    action_order: list[str] = []

    @contextmanager
    def valid_operational_guard(*_args, **_kwargs):
        with valid_fixture.module.guarded_action(
                "repair2", queue_path=valid_fixture.queue, queue_key_path=valid_fixture.queue_key,
                confinement_roots=(valid_fixture.root, Path("/usr")),
                expected_uid=os.geteuid()) as lease:
            action_order.append("gate")
            yield lease

    for module, decorator_name in (
        (top, "_guarded_all_locks"), (stage, "_guarded_stage"),
        (repair, "_guarded_repair2"), (continuation, "_guarded_continuation"),
    ):
        module.recovery_guard = valid_operational_guard
        guarded = getattr(module, decorator_name)(lambda: action_order.append("action"))
        guarded()
        assert action_order[-2:] == ["gate", "action"]

    with valid_operational_guard() as lease:
        top.ACTIVE_RECOVERY_LEASE = lease
        child_env, child_fds = top.child_lock_context()
        try:
            assert valid_fixture.module.LOCK_FD_ENV in child_env
            assert len(child_fds) == len(valid_fixture.locks)
        finally:
            for fd in child_fds:
                os.close(fd)
            top.ACTIVE_RECOVERY_LEASE = None


def proposed_queue_commit_fails_closed(fixture: Fixture, temp_root: Path) -> None:
    """Exercise actual top atomic_queue/enqueue paths against a priority insertion."""
    if str(HARN_DIR) not in sys.path:
        sys.path.insert(0, str(HARN_DIR))
    spec = importlib.util.spec_from_file_location(
        "dual_proposed_queue_top", HARN_DIR / "operator_queue_controller.py")
    top = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(top)

    top.QUEUE = fixture.queue
    top.QUEUE_KEY = fixture.queue_key
    top.QUEUE_LOCK = Path(dict(fixture.locks)["queue_mutation"])
    top.RECOVERY_EXPERIMENTS = {"stage", "repair2", "continuation"}
    before = digest(fixture.queue)

    with fixture.module.guarded_action(
            "repair2", queue_path=fixture.queue, queue_key_path=fixture.queue_key,
            confinement_roots=(fixture.root, Path("/usr")),
            expected_uid=os.geteuid()) as lease:
        top.ACTIVE_RECOVERY_LEASE = lease
        def gate(experiment_id=None, approval_record=None):
            lease.reverify(experiment_id or "repair2", approval_record)
        top.recovery_gate = gate

        proposed = copy.deepcopy(fixture.queue_payload)
        proposed["entries"].insert(0, {
            "experiment_id": "priority", "run_id": "priority-run", "position": 1,
            "status": "queued", "approval_status": "pending", "dependencies": [],
            "ordering_dependencies": [],
        })
        for position, entry in enumerate(proposed["entries"], 1):
            entry["position"] = position
        try:
            top.atomic_queue(proposed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("atomic_queue accepted priority insertion before protected position 4")
        assert digest(fixture.queue) == before

        # The public enqueue path must reach the same proposed-state gate and
        # must not commit even when its legacy shape checks accept the entry.
        entry_path = temp_root / "priority-entry.json"
        write(entry_path, json.dumps({
            "experiment_id": "priority", "run_id": "priority-run",
            "status": "queued", "approval_status": "pending",
        }).encode())
        top.load_queue = lambda: copy.deepcopy(fixture.queue_payload)
        top.validate_queue = lambda *_args, **_kwargs: None
        try:
            top.enqueue(entry_path, 1, "f" * 64)
        except RuntimeError:
            pass
        else:
            raise AssertionError("enqueue accepted priority insertion before protected position 4")
        assert digest(fixture.queue) == before
        top.ACTIVE_RECOVERY_LEASE = None


def double_main_lease_fixture(fixture: Fixture, temp_root: Path, module_name: str,
                              filename: str, experiment_id: str) -> None:
    """A competing direct main loses before PROCESS or any guarded run action."""
    if str(HARN_DIR) not in sys.path:
        sys.path.insert(0, str(HARN_DIR))
    spec = importlib.util.spec_from_file_location(module_name, HARN_DIR / filename)
    surface = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(surface)

    process_path = temp_root / f"{experiment_id}-process.json"
    audit_path = temp_root / f"{experiment_id}-process-writers.txt"
    surface.harn.PROCESS = process_path

    @contextmanager
    def guard(approval_record=None):
        with fixture.module.guarded_action(
                experiment_id, approval_record, queue_path=fixture.queue,
                queue_key_path=fixture.queue_key,
                confinement_roots=(fixture.root, Path("/usr")),
                expected_uid=os.geteuid()) as lease:
            yield lease

    def atomic_process(path, payload, *_args, **_kwargs):
        assert path == process_path
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        write(path, (json.dumps(payload, sort_keys=True) + "\n").encode())

    surface.recovery_guard = guard
    surface.harn.atomic_json = atomic_process
    ready_read, ready_write = os.pipe()
    surface.run = lambda: (os.write(ready_write, b"READY"), time.sleep(0.75))

    winner = os.fork()
    if winner == 0:
        try:
            os.close(ready_read)
            sys.argv = [filename, "run"]
            surface.main()
            os._exit(0)
        except BaseException:
            os._exit(91)
    os.close(ready_write)
    assert os.read(ready_read, 5) == b"READY"
    os.close(ready_read)

    loser = os.fork()
    if loser == 0:
        try:
            sys.argv = [filename, "run"]
            surface.main()
        except RuntimeError:
            os._exit(0)
        except BaseException:
            os._exit(92)
        os._exit(93)
    waited, status = os.waitpid(loser, 0)
    assert waited == loser and os.waitstatus_to_exitcode(status) == 0
    waited, status = os.waitpid(winner, 0)
    assert waited == winner and os.waitstatus_to_exitcode(status) == 0
    assert audit_path.read_text().splitlines() == [str(winner)]
    assert json.loads(process_path.read_text())["controller_pid"] == winner


def main() -> None:
    assert_common_gate_surface()
    with tempfile.TemporaryDirectory(prefix="dual-authority-recovery-v3-") as raw:
        root = Path(raw)
        normal = Fixture(root, "normal")
        first = normal.verify()
        second = normal.verify()
        assert first.entry_binding_sha256 == second.entry_binding_sha256

        missing = Fixture(root, "missing")
        del missing.queue_payload["recovery_authorization"]
        missing.queue_payload = missing.module.sign_for_fixture(missing.queue_payload, missing.key,
                                                                 missing.module.QUEUE_DOMAIN)
        missing.write_queue()
        expect_rejected(missing, "missing recovery role")

        swapped = Fixture(root, "swapped")
        swapped.recovery_payload["execution_authority"], swapped.recovery_payload["recovery_authority"] = (
            swapped.recovery_payload["recovery_authority"], swapped.recovery_payload["execution_authority"])
        swapped.resign_recovery_and_queue()
        expect_rejected(swapped, "authority role swap")

        reordered = Fixture(root, "reordered")
        reordered.recovery_payload["ordered_closed_revocations"].reverse()
        reordered.resign_recovery_and_queue()
        expect_rejected(reordered, "closed revocation reorder")

        replay = Fixture(root, "replay")
        replay.recovery_payload["prior_queue_sha256"] = "0" * 64
        replay.resign_recovery_and_queue()
        expect_rejected(replay, "prior queue rollback/replay")

        missing_continuation = Fixture(root, "missing-continuation")
        missing_continuation.recovery_payload["ordered_closed_revocations"] = [
            item for item in missing_continuation.recovery_payload["ordered_closed_revocations"]
            if item["role"] != "old_continuation_exact_approval"
        ]
        missing_continuation.resign_recovery_and_queue()
        expect_rejected(missing_continuation, "conditional continuation revocation")

        duplicate_replacement = Fixture(root, "duplicate-replacement")
        duplicate_replacement.recovery_payload["replacement_entries"].append(
            copy.deepcopy(duplicate_replacement.recovery_payload["replacement_entries"][1])
        )
        duplicate_replacement.resign_recovery_and_queue()
        expect_rejected(duplicate_replacement, "duplicate recovery replacement")

        duplicate_run = Fixture(root, "duplicate-run")
        duplicate_run.recovery_payload["affected_runs"].append(
            copy.deepcopy(duplicate_run.recovery_payload["affected_runs"][1])
        )
        duplicate_run.resign_recovery_and_queue()
        expect_rejected(duplicate_run, "duplicate affected run")

        bad_lock = Fixture(root, "bad-lock")
        os.chmod(Path(bad_lock.locks[0][1]), 0o644)
        expect_rejected(bad_lock, "unsafe exclusive lock")

        wrong_execution = Fixture(root, "wrong-execution")
        approval_path = Path(wrong_execution.entries[1]["approval_evidence"]["path"])
        approval = json.loads(approval_path.read_text())
        approval["channel_record_sha256"] = wrong_execution.module.RECOVERY_AUTHORITY_SHA256
        approval = wrong_execution.module.sign_for_fixture(approval, wrong_execution.key,
                                                            wrong_execution.module.APPROVAL_DOMAIN)
        write(approval_path, json.dumps(approval, sort_keys=True).encode())
        wrong_execution.entries[1]["approval_evidence"]["sha256"] = digest(approval_path)
        wrong_execution.queue_payload = wrong_execution.module.sign_for_fixture(
            wrong_execution.queue_payload, wrong_execution.key, wrong_execution.module.QUEUE_DOMAIN)
        wrong_execution.write_queue()
        expect_rejected(wrong_execution, "recovery authority substituted for execution")

        recovery_as_execution = Fixture(root, "recovery-as-execution")
        recovery_as_execution.recovery_payload["recovery_authority"]["channel_record_sha256"] = (
            recovery_as_execution.module.EXECUTION_AUTHORITY_SHA256
        )
        recovery_as_execution.resign_recovery_and_queue()
        expect_rejected(recovery_as_execution, "recovery role uses c6 execution authority")

        derived_recovery = Fixture(root, "derived-recovery")
        derived_recovery.recovery_payload["recovery_authority"]["channel_record_sha256"] = "d" * 64
        derived_recovery.resign_recovery_and_queue()
        expect_rejected(derived_recovery, "derived recovery authority replaces exact 983 role")

        added_revocation = Fixture(root, "added-revocation")
        added_revocation.recovery_payload["ordered_closed_revocations"].append(
            {"role": "unreviewed", "sha256": "f" * 64}
        )
        added_revocation.resign_recovery_and_queue()
        expect_rejected(added_revocation, "added closed-set revocation")

        # The full production-shaped inventory is closed, not merely a set of
        # approval roles.  Every superseded artifact is independently required.
        for index, (role, _digest) in enumerate(normal.module.ORDERED_CLOSED_REVOCATIONS):
            omitted = Fixture(root, f"omitted-revocation-{index}")
            omitted.recovery_payload["ordered_closed_revocations"] = [
                item for item in omitted.recovery_payload["ordered_closed_revocations"]
                if item["role"] != role
            ]
            omitted.resign_recovery_and_queue()
            expect_rejected(omitted, f"omitted closed-set role {role}")

        archive_drift = Fixture(root, "archive-drift")
        write(archive_drift.prior_queue, b"drifted archived queue")
        expect_rejected(archive_drift, "authenticated archive entry drift")

        rollback = Fixture(root, "full-rollback")
        rollback.module.BASELINE_ONLY_BINDINGS = {
            rollback.module.entry_binding(rollback.entries[1])
        }
        expect_rejected(rollback, "formerly HMAC-valid full queue/approval rollback")

        operational_actions = root / "operational-actions"
        operational_actions.mkdir(mode=0o700)
        operational_valid = Fixture(root, "operational-valid")
        operational_entrypoints_fail_closed([
            ("recovery-c6", recovery_as_execution),
            ("recovery-derived", derived_recovery),
            ("archive-drift", archive_drift),
            ("added-revocation", added_revocation),
            ("full-rollback", rollback),
        ], operational_actions, operational_valid)

        proposed_root = root / "proposed-actions"
        proposed_root.mkdir(mode=0o700)
        proposed_queue_commit_fails_closed(Fixture(root, "proposed-queue"), proposed_root)

        main_root = root / "double-main-actions"
        main_root.mkdir(mode=0o700)
        double_main_lease_fixture(
            Fixture(root, "double-main-repair2"), main_root,
            "dual_double_main_repair2", "rmatched_matrix_repair2_harn.py", "repair2")
        double_main_lease_fixture(
            Fixture(root, "double-main-continuation"), main_root,
            "dual_double_main_continuation", "rmatched_matrix_continuation_harn.py",
            "continuation")

        # A real independent process holding any one declared lock makes the
        # entire gate fail closed before action. Exercise every lock role.
        lock_holder = (
            "import fcntl,os,sys,time; "
            "fd=os.open(sys.argv[1],os.O_RDWR|os.O_NOFOLLOW); "
            "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); print('READY',flush=True); time.sleep(60)"
        )
        for index, (role, lock_path) in enumerate(normal.locks):
            locked = Fixture(root, f"independent-lock-{index}")
            path = Path(locked.locks[index][1])
            process = subprocess.Popen(
                [sys.executable, "-B", "-I", "-S", "-c", lock_holder, str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            try:
                assert process.stdout is not None and process.stdout.readline().strip() == "READY"
                expect_rejected(locked, f"independent process holds {role}")
            finally:
                process.terminate()
                process.wait(timeout=5)

        valid_guard = Fixture(root, "valid-guard")
        actions: list[str] = []
        with valid_guard.module.guarded_action(
                "repair2", queue_path=valid_guard.queue, queue_key_path=valid_guard.queue_key,
                confinement_roots=(valid_guard.root, Path("/usr")), expected_uid=os.geteuid()) as lease:
            actions.append("first_guarded_action")
            lock_env, duplicated = valid_guard.module.inherited_lock_env(lease)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                try:
                    os.close(read_fd)
                    valid_guard.module._ACTIVE_LEASE = None
                    os.environ[valid_guard.module.LOCK_FD_ENV] = lock_env
                    with valid_guard.module.guarded_action(
                            "repair2", queue_path=valid_guard.queue,
                            queue_key_path=valid_guard.queue_key,
                            confinement_roots=(valid_guard.root, Path("/usr")),
                            expected_uid=os.geteuid()):
                        os.write(write_fd, b"PASS")
                except BaseException as exc:
                    os.write(write_fd, ("FAIL:" + repr(exc)).encode())
                finally:
                    os._exit(0)
            os.close(write_fd)
            for fd in duplicated:
                os.close(fd)
            child_result = os.read(read_fd, 4096)
            os.close(read_fd)
            waited, status = os.waitpid(pid, 0)
            assert waited == pid and os.waitstatus_to_exitcode(status) == 0
            assert child_result == b"PASS", child_result
        assert actions == ["first_guarded_action"]

        # Same-FD authority: replacing the pathname after open cannot change
        # bytes read from the held O_NOFOLLOW descriptor.
        race = root / "fd-race"
        race.mkdir(mode=0o700)
        target = race / "authority.json"
        write(target, b"approved")
        expected = digest(target)
        module = normal.module
        original_open = module._open_beneath
        def swap_after_open(path, roots):
            fd = original_open(path, roots)
            replacement = race / "replacement"
            write(replacement, b"hostile")
            os.replace(replacement, target)
            return fd
        module._open_beneath = swap_after_open
        try:
            try:
                module.read_bound_file(target, roots=(race,), expected_sha256=expected,
                                       allowed_modes={0o600}, expected_uid=os.geteuid(),
                                       label="fd race fixture")
            except RuntimeError:
                pass
            else:
                raise AssertionError("pathname swap race was accepted")
            assert target.read_bytes() == b"hostile"
        finally:
            module._open_beneath = original_open

    print("[PASS] dual authority roles/full closed inventory, proposed queue validation, whole-main lease exclusion, per-owner descriptors, inherited-FD lifecycle, all operational fail-closed surfaces, exact repeatability")


if __name__ == "__main__":
    main()
