#!/usr/bin/env python3
"""No-production-mutation tests for exact NVMe staging and crash recovery."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import importlib.util
import json
import os
import subprocess
import tempfile
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage_rmatched_matrix_nvme.py"
HARN_DIR = ROOT / "scripts/experiment_harness"
sys.path.insert(0, str(HARN_DIR))


def bypass_recovery(module) -> None:
    module.recovery_gate = lambda *_args, **_kwargs: None
    if hasattr(module, "recovery_gate_for_entry"):
        module.recovery_gate_for_entry = lambda *_args, **_kwargs: None
    if not hasattr(module, "recovery_guard"):
        return

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


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, root: Path) -> tuple[Path, Path]:
    hdd = root / "hdd"
    matrix = hdd / "matrix"
    metrics = hdd / "metrics"
    matrix.mkdir(parents=True)
    metrics.mkdir()
    (matrix / "a.flac").write_bytes(b"audio-a")
    (matrix / "b.flac").write_bytes(b"audio-b")
    (metrics / "metrics.txt").write_text("clap_score: 1\n")
    nvme = root / "nvme"
    state = root / "logs/stage"
    state.mkdir(parents=True)
    module.NVME_ROOT = nvme
    module.NVME_PARENT = nvme / "meanaudio"
    module.FINAL = module.NVME_PARENT / "eval_output"
    module.STAGING = module.NVME_PARENT / f".eval_output.stage-{module.RUN_ID}"
    module.STATE = state
    module.JOURNAL = state / "transaction.json"
    module.REPORT = state / "stage_report.json"
    module.HARN_KEY = state / "ledger_hmac.key"
    module.APPROVAL_STATE = state / "operator_approval.json"
    module.HARN_LOCK = state / "controller.lock"
    module.SOURCE_MANIFEST = root / "source-manifest.json"
    module.LEGACY_EVAL_OUTPUT = hdd
    roots = []
    for role, source, destination in (("matrix", matrix, "matrix"), ("metrics", metrics, "metrics/cell")):
        snapshot = module.tree_snapshot(source)
        roots.append({"role": role, "source": str(source), "destination_relative": destination,
                      "file_count": snapshot["file_count"], "total_bytes": snapshot["total_bytes"],
                      "tree_sha256": snapshot["tree_sha256"]})
    manifest = {"document_kind": "rmatched_nvme_stage_source_manifest", "schema_version": 1,
                "roots": roots, "totals": {"file_count": sum(item["file_count"] for item in roots),
                                             "total_bytes": sum(item["total_bytes"] for item in roots)}}
    module.SOURCE_MANIFEST.write_text(json.dumps(manifest))
    key = hashlib.sha256(b"fixture-stage-key").digest()
    module.HARN_KEY.write_bytes(key)
    os.chmod(module.HARN_KEY, 0o600)
    approval_record = state / "exact-approval.json"
    approval_record.write_text("exact approval")
    approval = {"document_kind": "nvme_stage_approval_state", "schema_version": 1,
                "experiment_id": module.EXPERIMENT_ID, "run_id": module.RUN_ID, "state": "reserved",
                "approval_record_sha256": module.sha256_file(approval_record), "reservation": {
                    "controller_pid": os.getpid(), "controller_start_ticks": module.process_start(os.getpid()),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}}
    approval["integrity"] = hmac.new(key, b"meanaudio-nvme-stage-approval-v1\0" + module.canonical(approval),
                                     hashlib.sha256).hexdigest()
    module.APPROVAL_STATE.write_text(json.dumps(approval))
    module.HARN_LOCK.touch(mode=0o600)
    return hdd, state


def issue_capability(module, root: Path, action: str) -> Path:
    payload = {"document_kind": "nvme_stage_write_capability", "status": "authorized",
               "experiment_id": module.EXPERIMENT_ID, "run_id": module.RUN_ID, "action": action,
               "executable_sha256": module.sha256_file(Path(module.__file__)),
               "source_manifest_sha256": module.sha256_file(module.SOURCE_MANIFEST),
               "approval_state_sha256": module.sha256_file(module.APPROVAL_STATE),
               "approval_record_sha256": json.loads(module.APPROVAL_STATE.read_text())["approval_record_sha256"],
               "parent_pid": os.getpid(), "parent_start_ticks": module.process_start(os.getpid()),
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "harn_lock": str(module.HARN_LOCK),
               "writable_paths": [str(module.NVME_ROOT), str(module.NVME_PARENT), str(module.STAGING),
                                  str(module.FINAL), str(module.JOURNAL), str(module.REPORT)]}
    payload["integrity"] = hmac.new(module.HARN_KEY.read_bytes(), module.CAPABILITY_DOMAIN + module.canonical(payload),
                                     hashlib.sha256).hexdigest()
    path = root / f"capability-{action}.json"
    path.write_text(json.dumps(payload))
    return path


def source_hashes(root: Path) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


def run_case(root: Path, fault: str | None) -> None:
    module = load(f"nvme_stage_{fault or 'normal'}")
    hdd, state = configure(module, root)
    before = source_hashes(hdd)
    lock_fd = os.open(module.HARN_LOCK, os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original_getppid = module.os.getppid
    module.os.getppid = lambda: os.getpid()
    module.ALLOW_TEST_FAULTS = fault is not None
    module.FAULT_AFTER = fault
    try:
        capability = issue_capability(module, root, "stage")
        os.environ[module.CAPABILITY_ENV] = str(capability)
        if fault:
            try:
                module.stage("stage")
            except ChildProcessError:
                pass
            else:
                raise AssertionError(f"fault did not trigger: {fault}")
            module.FAULT_AFTER = None
            capability = issue_capability(module, root, "reconcile")
            os.environ[module.CAPABILITY_ENV] = str(capability)
            module.stage("reconcile")
        else:
            module.stage("stage")
        module.validate_report()
    finally:
        module.os.getppid = original_getppid
        os.environ.pop(module.CAPABILITY_ENV, None)
        os.close(lock_fd)
    assert source_hashes(hdd) == before
    assert module.FINAL.is_dir() and not module.STAGING.exists()
    assert json.loads(module.REPORT.read_text())["hdd_sources_unchanged"] is True
    for path in root.rglob("*"):
        if path.is_file() and path not in set(hdd.rglob("*")) and path != module.SOURCE_MANIFEST:
            assert path.is_relative_to(module.NVME_ROOT) or path.is_relative_to(state) or path.name.startswith("capability-")


def rejection_cases(root: Path) -> None:
    module = load("nvme_stage_rejections")
    hdd, _ = configure(module, root)
    os.environ.pop(module.CAPABILITY_ENV, None)
    before_missing_capability = source_hashes(hdd)
    try:
        module.stage("stage")
    except RuntimeError as exc:
        assert "capability is missing" in str(exc)
    else:
        raise AssertionError("direct staging without capability was accepted")
    assert not module.NVME_ROOT.exists() and source_hashes(hdd) == before_missing_capability
    (hdd / "matrix/link").symlink_to(hdd / "matrix/a.flac")
    try:
        module.tree_snapshot(hdd / "matrix")
    except RuntimeError:
        pass
    else:
        raise AssertionError("recursive source symlink accepted")
    (hdd / "matrix/link").unlink()
    os.link(hdd / "matrix/a.flac", hdd / "matrix/hardlink.flac")
    try:
        module.tree_snapshot(hdd / "matrix")
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected source hardlink accepted")

    escape_root = root / "legacy-symlink"
    escape_root.mkdir()
    escape = load("nvme_stage_legacy_symlink")
    legacy, _ = configure(escape, escape_root)
    escape.NVME_ROOT.symlink_to(legacy, target_is_directory=True)
    before = source_hashes(legacy)
    lock_fd = os.open(escape.HARN_LOCK, os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original_getppid = escape.os.getppid
    escape.os.getppid = lambda: os.getpid()
    capability = issue_capability(escape, escape_root, "stage")
    os.environ[escape.CAPABILITY_ENV] = str(capability)
    try:
        try:
            escape.stage("stage")
        except RuntimeError as exc:
            assert "legacy HDD" in str(exc)
        else:
            raise AssertionError("write-capable path through legacy HDD symlink accepted")
    finally:
        escape.os.getppid = original_getppid
        os.environ.pop(escape.CAPABILITY_ENV, None)
        os.close(lock_fd)
    assert source_hashes(legacy) == before


def no_gpu_path_contract() -> None:
    repair_path = ROOT / "scripts/repair_rmatched_matrix_corrupt_flac_v2.py"
    spec = importlib.util.spec_from_file_location("repair_nvme_path_fixture", repair_path)
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    nvme = Path("/home/kojiek/nvme_experiment_artifacts/meanaudio/eval_output")
    for path in (repair.AUDIO, repair.METRICS, repair.QUARANTINED_TARGET,
                 repair.QUARANTINED_METRICS, repair.STAGED_TARGET, repair.TRANSACTION):
        assert path.resolve(strict=False).is_relative_to(nvme)
    runner = (ROOT / "scripts/eval/eval_rmatched_s1_s2_steps_cfg_matrix_continuation.sh").read_text()
    assert 'OUT_ROOT="$NVME_EVAL/$TAG"' in runner
    assert 'METRIC_ROOT="$NVME_EVAL/metrics"' in runner
    assert 'STORAGE_PATH=/home/kojiek/nvme_experiment_artifacts' in runner
    assert '--out_dir "$METRIC_ROOT"' in runner
    assert 'OUT_ROOT="$ROOT/eval_output' not in runner


def signed_queue_harn_check(root: Path) -> None:
    harn_path = HARN_DIR / "rmatched_matrix_nvme_stage_harn.py"
    spec = importlib.util.spec_from_file_location("nvme_stage_harn_fixture", harn_path)
    harn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harn)
    bypass_recovery(harn)
    direct_state = root / "unapproved-direct-init"
    harn.STATE = direct_state
    harn.LOCK = direct_state / "controller.lock"
    harn.KEY = direct_state / "ledger_hmac.key"
    try:
        harn.init(root / "arbitrary-approval.json", "0" * 64)
    except (OSError, RuntimeError):
        pass
    else:
        raise AssertionError("unapproved direct stage init was accepted")
    assert not direct_state.exists(), "unapproved direct stage init mutated state"
    state = root / "state"
    state.mkdir(mode=0o700)
    os.chmod(state, 0o700)
    for name, value in {
        "STATE": state, "KEY": state / "ledger_hmac.key", "LOCK": state / "controller.lock",
        "APPROVAL": state / "operator_approval.json", "GENERATIONS": state / "generations",
        "AUTHORIZATION": state / "stage_authorization.json",
        "OUTBOX": state / "outbox", "CURRENT": state / "current",
        "PENDING_CONTRACT": state / "pending_contract.json", "PENDING_PREFLIGHT": state / "pending_preflight.json",
        "PENDING_LEDGER": state / "pending_ledger.json", "PENDING_QUEUE": state / "pending_queue.json",
        "REPORT": state / "stage_report.json", "JOURNAL": state / "transaction.json",
    }.items():
        setattr(harn, name, value)
    harn.QUEUE = root / "queue.json"
    harn.QUEUE_KEY = root / "queue.key"
    harn.PREREG = root / "stage-prereg.json"
    harn.SOURCE_MANIFEST = root / "source-manifest.json"
    harn.SOURCE_MANIFEST.write_text(json.dumps({"totals": {"file_count": 1, "total_bytes": 1}}))
    harn.NVME_ROOT = root / "nvme"
    harn.NVME_PARENT = harn.NVME_ROOT / "meanaudio"
    harn.FINAL = harn.NVME_PARENT / "eval_output"
    harn.STAGING = harn.NVME_PARENT / f".eval_output.stage-{harn.RUN_ID}"
    fixture_stager = root / "stage-fixture.py"
    fixture_stager.write_text("import sys\nraise SystemExit(0)\n")
    harn.STAGER = fixture_stager
    queue_key = hashlib.sha256(b"fixture-queue-key").digest()
    child_key = hashlib.sha256(b"fixture-child-key").digest()
    harn.QUEUE_KEY.write_bytes(queue_key)
    os.chmod(harn.QUEUE_KEY, 0o600)
    harn.KEY.write_bytes(child_key)
    os.chmod(harn.KEY, 0o600)
    harn.LOCK.touch(mode=0o600)
    wrapper = root / "validate-stage-fixture.py"
    validator_path = ROOT / "scripts/validate_rmatched_matrix_nvme_stage_bundle.py"
    wrapper.write_text(
        "import importlib.util\n"
        f"s=importlib.util.spec_from_file_location('v',{str(validator_path)!r})\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        f"m.validate_bundle(state=__import__('pathlib').Path({str(state)!r}),"
        f"prereg=__import__('pathlib').Path({str(harn.PREREG)!r}),"
        f"operator_queue=__import__('pathlib').Path({str(harn.QUEUE)!r}),"
        f"operator_queue_key=__import__('pathlib').Path({str(harn.QUEUE_KEY)!r}),"
        f"experiment_id={harn.EXPERIMENT_ID!r},run_id={harn.RUN_ID!r})\n")
    completion = {"argv": [str(Path('/usr/bin/python3.12')), str(wrapper)], "bindings": [
        {"path": "/usr/bin/python3.12", "sha256": harn.digest(Path('/usr/bin/python3.12'))},
        {"path": str(wrapper), "sha256": harn.digest(wrapper)},
    ]}
    harn.PREREG.write_text(json.dumps({
        "experiment_id": harn.EXPERIMENT_ID, "run_id": harn.RUN_ID,
        "source_manifest": {"total_bytes": 1},
        "runtime": {"manifest": str(ROOT/'docs/experiments/rmatched_repair2_runtime_manifest.json'),
                    "manifest_sha256": harn.digest(ROOT/'docs/experiments/rmatched_repair2_runtime_manifest.json'),
                    "verifier": str(ROOT/'scripts/experiment_harness/runtime_binding.py'),
                    "verifier_sha256": harn.digest(ROOT/'scripts/experiment_harness/runtime_binding.py'),
                    "required_roles": ["system_python", "system_stdlib", "system_native_runtime",
                                       "stage_bundle_validator"]},
        "report_validator": {
            "argv": ["/usr/bin/python3.12", "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
                     str(fixture_stager), "validate-report"],
            "contract_descriptor_argv": ["/usr/bin/python3.12", str(fixture_stager), "validate-report"],
            "bindings": [
                {"path": "/usr/bin/python3.12", "sha256": harn.digest(Path('/usr/bin/python3.12'))},
                {"path": str(fixture_stager), "sha256": harn.digest(fixture_stager)},
            ],
        },
        "completion_validator": completion,
    }))
    bundle_validator = ROOT / "scripts/validate_experiment_harness_documents.py"
    shared = ROOT / "scripts/experiment_harness/qwen_s2q_k_mf25_harn.py"
    runtime_binding = ROOT / "scripts/experiment_harness/runtime_binding.py"
    entry = {"position": 1, "experiment_id": harn.EXPERIMENT_ID, "run_id": harn.RUN_ID,
             "approval_status": "approved", "status": "queued", "contract": str(harn.PREREG),
             "contract_sha256": harn.digest(harn.PREREG), "dependencies": [], "ordering_dependencies": [],
             "runtime": {"controller": str(harn_path), "controller_sha256": harn.digest(harn_path),
                         "status_source": str(state / "current"),
                         "pending_status_source": str(state / "pending_queue.json"),
                         "approval_cli": "exact_record_v1",
                         "bundle_validator": {"python": "/usr/bin/python3.12", "path": str(bundle_validator),
                            "python_sha256": harn.digest(Path('/usr/bin/python3.12')),
                            "sha256": harn.digest(bundle_validator)},
                         "completion_validator": completion,
                         "child_ledger_key": {"path": str(harn.KEY), "sha256": harn.digest(harn.KEY)},
                         "import_bindings": [
                            {"module": "qwen_s2q_k_mf25_harn", "path": str(shared), "sha256": harn.digest(shared)},
                            {"module": "runtime_binding", "path": str(runtime_binding), "sha256": harn.digest(runtime_binding)}],
                         "transitive_runtime": {"manifest": str(ROOT/'docs/experiments/rmatched_repair2_runtime_manifest.json'),
                            "manifest_sha256": harn.digest(ROOT/'docs/experiments/rmatched_repair2_runtime_manifest.json'),
                            "required_roles": ["system_python", "system_stdlib", "system_native_runtime",
                                               "stage_bundle_validator"],
                            "verifier": {"path": str(ROOT/'scripts/experiment_harness/runtime_binding.py'),
                                         "sha256": harn.digest(ROOT/'scripts/experiment_harness/runtime_binding.py')}}}}
    controller_spec = importlib.util.spec_from_file_location("nvme_top_controller_fixture", HARN_DIR / "operator_queue_controller.py")
    controller = importlib.util.module_from_spec(controller_spec)
    controller_spec.loader.exec_module(controller)
    bypass_recovery(controller)
    assert harn.entry_binding(entry) == controller.entry_approval_binding(entry)
    binding = harn.entry_binding(entry)
    record = harn.signed({"document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": harn.EXPERIMENT_ID, "run_id": harn.RUN_ID,
        "contract_sha256": entry["contract_sha256"], "controller_sha256": entry["runtime"]["controller_sha256"],
        "queue_entry_binding_sha256": binding, "channel_record_sha256": "a" * 64},
        queue_key, b"meanaudio-queue-approval-v1\0")
    record_path = root / "approval.json"
    record_path.write_text(json.dumps(record))
    entry["approval_evidence"] = {"path": str(record_path), "sha256": harn.digest(record_path)}
    queue = harn.signed({"document_kind": "operator_approved_experiment_backlog", "schema_version": 1,
                         "entries": [entry]}, queue_key, b"meanaudio-operator-queue-v1\0")
    harn.QUEUE.write_text(json.dumps(queue))
    assert harn.verify_queue_approval(record_path, binding)["status"] == "approved"
    unsigned = dict(record)
    unsigned["integrity"] = "0" * 64
    record_path.write_text(json.dumps(unsigned))
    entry["approval_evidence"]["sha256"] = harn.digest(record_path)
    harn.QUEUE.write_text(json.dumps(harn.signed(
        {"document_kind": "operator_approved_experiment_backlog", "schema_version": 1, "entries": [entry]},
        queue_key, b"meanaudio-operator-queue-v1\0")))
    try:
        harn.verify_queue_approval(record_path, binding)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsigned staging approval accepted")

    # Restore exact record and exercise exact-record init through the shared
    # HARN ready -> active -> completed lifecycle without a production child.
    record_path.write_text(json.dumps(record))
    entry["approval_evidence"]["sha256"] = harn.digest(record_path)
    harn.QUEUE.write_text(json.dumps(harn.signed(
        {"document_kind": "operator_approved_experiment_backlog", "schema_version": 1, "entries": [entry]},
        queue_key, b"meanaudio-operator-queue-v1\0")))
    harn.verify_preregistered = lambda: None
    original_run = harn.subprocess.run
    def fake_run(argv, *args, **kwargs):
        if str(bundle_validator) in argv:
            return original_run(argv, *args, **kwargs)
        if str(harn.STAGER) in argv and "stage" in argv:
            harn.REPORT.write_text(json.dumps({"status": "passed"}))
        return SimpleNamespace(returncode=0, stdout="accepted", stderr="")
    harn.subprocess.run = fake_run
    try:
        harn.init(record_path, binding)
        assert harn._status() == "ready"
        init_generation = Path(harn.CURRENT.read_text().strip())
        init_contract = json.loads((init_generation / "contract.json").read_text())
        validate_command = next(item for item in init_contract["commands"] if item["action_id"] == "validate")
        assert validate_command["argv"] == ["/usr/bin/python3.12", str(fixture_stager), "validate-report"]
        assert json.loads(harn.PREREG.read_text())["report_validator"]["argv"][1:6] == [
            "-X", "pycache_prefix=/dev/null", "-B", "-I", "-S",
        ]
        harn.run()
    finally:
        harn.subprocess.run = original_run
    assert harn._status() == "completed"
    assert harn.read_approval()["state"] == "consumed"
    assert json.loads((harn.OUTBOX / "terminal_success.json").read_text())["status"] == "delivered"

    # Independently validate the real authenticated bundle, then make the top
    # controller observe it and unblock an exact Repair2 scientific dependency.
    validator_spec = importlib.util.spec_from_file_location("nvme_stage_bundle_fixture", validator_path)
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    validator.validate_bundle(state=state, prereg=harn.PREREG, operator_queue=harn.QUEUE,
                              operator_queue_key=harn.QUEUE_KEY, experiment_id=harn.EXPERIMENT_ID,
                              run_id=harn.RUN_ID)
    controller.ROOT = ROOT
    stage_entry = json.loads(json.dumps(entry))
    stage_entry["status"] = "completed"
    generation = Path(harn.CURRENT.read_text().strip())
    structural = subprocess.run(
        ["/usr/bin/python3.12", str(bundle_validator), "--contract", str(generation / "contract.json"),
         "--preflight", str(generation / "preflight.json"), "--ledger", str(generation / "ledger.json"),
         "--queue", str(generation / "queue.json")], text=True, capture_output=True)
    assert structural.returncode == 0, structural.stderr + structural.stdout
    child = controller.child_queue_entry(stage_entry)
    assert child and child["status"] == "completed" and child["terminal_notification_status"] == "delivered"
    repair_entry = {"experiment_id": "rmatched-s1-s2-steps-cfg-matrix-repair2", "run_id": "repair-fixture",
                    "status": "queued", "approval_status": "approved",
                    "dependencies": [harn.EXPERIMENT_ID], "ordering_dependencies": [], "runtime": {}}
    assert controller.eligibility_blockers(repair_entry, [stage_entry, repair_entry]) == []

    repair_path = HARN_DIR / "rmatched_matrix_repair2_harn.py"
    repair_spec = importlib.util.spec_from_file_location("repair2_stage_dependency_fixture", repair_path)
    repair = importlib.util.module_from_spec(repair_spec)
    repair_spec.loader.exec_module(repair)
    bypass_recovery(repair)
    repair_prereg = root / "repair2-prereg.json"
    repair.STAGE_CONTRACT = harn.PREREG
    repair.STAGE_BUNDLE_VALIDATOR = wrapper
    repair.CURRENT_APPROVAL_HASH = "b" * 64
    repair.PREREG = repair_prereg
    repair_prereg.write_text(json.dumps({"scientific_dependencies": [{
        "experiment_id": harn.EXPERIMENT_ID, "run_id": harn.RUN_ID, "required_state": "completed",
        "contract": str(harn.PREREG), "contract_sha256": harn.digest(harn.PREREG),
        "authenticated_bundle_validator": str(wrapper),
        "authenticated_bundle_validator_sha256": harn.digest(wrapper),
    }]}))
    repair.validate_stage_dependency()

    # A report plus a valid-looking completed queue is insufficient when the
    # exact approval transition was not consumed.
    consumed = harn.read_approval()
    harn.write_approval({**consumed, "state": "reserved"})
    try:
        try:
            repair.validate_stage_dependency()
        except RuntimeError as exc:
            assert "authenticated completed dependency invalid" in str(exc)
        else:
            raise AssertionError("Repair2 accepted stage report without consumed approval")
        assert controller.eligibility_blockers(repair_entry, [stage_entry, repair_entry])
    finally:
        harn.write_approval(consumed)

    # A forged terminal child ledger must fail both the exact validator and top
    # terminal acceptance even though all raw bundle paths still exist.
    ledger_path = generation / "ledger.json"
    ledger_raw = ledger_path.read_bytes()
    forged = json.loads(ledger_raw)
    forged["events"][-1]["notification_status"] = "not_applicable"
    ledger_path.write_text(json.dumps(forged, separators=(",", ":"), sort_keys=True))
    try:
        try:
            validator.validate_bundle(state=state, prereg=harn.PREREG, operator_queue=harn.QUEUE,
                                      operator_queue_key=harn.QUEUE_KEY, experiment_id=harn.EXPERIMENT_ID,
                                      run_id=harn.RUN_ID)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forged stage terminal ledger accepted")
        assert controller.child_queue_entry(stage_entry) is None
    finally:
        ledger_path.write_bytes(ledger_raw)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nvme-stage-selftest-") as raw:
        root = Path(raw)
        for fault in (None, "copy_started", "during_copy", "staged_verified", "final_renamed"):
            case = root / (fault or "normal")
            case.mkdir()
            run_case(case, fault)
        rejected = root / "rejections"
        rejected.mkdir()
        rejection_cases(rejected)
        no_gpu_path_contract()
        queue = root / "signed-queue"
        queue.mkdir()
        signed_queue_harn_check(queue)
    print("[PASS] NVMe stage capability, source immutability, exact destination, crash reconciliation, path isolation")


if __name__ == "__main__":
    main()
