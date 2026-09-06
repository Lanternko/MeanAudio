#!/usr/bin/env python3
"""Fail-closed validator for the authenticated NVMe-stage terminal bundle."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/home/kojiek/MeanAudio")
EXPERIMENT_ID = "rmatched-s1-s2-steps-cfg-matrix-nvme-stage"
RUN_ID = "run-20260815-rmatched-matrix-nvme-stage1"
STATE = Path("/home/kojiek/logs/rmatched_matrix_nvme_stage_harn")
PREREG = ROOT / "docs/experiments/rmatched_matrix_nvme_stage_contract.json"
OPERATOR_QUEUE = Path("/home/kojiek/logs/meanaudio_operator_backlog/queue.json")
OPERATOR_QUEUE_KEY = OPERATOR_QUEUE.parent / "queue_hmac.key"
CONTRACT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


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
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise RuntimeError(f"unsafe authentication key: {path}")
        key = os.read(fd, 128)
    finally:
        os.close(fd)
    if len(key) < 32:
        raise RuntimeError(f"short authentication key: {path}")
    return key


def verify_document(payload: dict[str, Any], key: bytes, domain: bytes) -> None:
    supplied = payload.get("integrity")
    unsigned = {name: value for name, value in payload.items() if name != "integrity"}
    expected = hmac.new(key, domain + canonical(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise RuntimeError("authenticated document signature invalid")


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


def load_generation(state: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    current = state / "current"
    if not current.is_file() or current.is_symlink():
        raise RuntimeError("stage current-generation pointer is missing or unsafe")
    generation = Path(current.read_text().strip())
    state_resolved = state.resolve(strict=True)
    generation_resolved = generation.resolve(strict=True)
    if (generation.is_symlink() or generation_resolved.parent.parent != state_resolved
            or generation_resolved.parent.name != "generations"):
        raise RuntimeError("stage generation escapes its authenticated state root")
    paths = {name: generation_resolved / f"{name}.json" for name in ("contract", "preflight", "ledger", "queue")}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise RuntimeError("stage terminal generation is incomplete or unsafe")
    return generation_resolved, {name: json.loads(path.read_text()) for name, path in paths.items()}


def validate_bundle(*, state: Path, prereg: Path, operator_queue: Path,
                    operator_queue_key: Path, experiment_id: str, run_id: str) -> None:
    prereg_hash = digest(prereg)
    prereg_value = json.loads(prereg.read_text())
    if prereg_value.get("experiment_id") != experiment_id or prereg_value.get("run_id") != run_id:
        raise RuntimeError("stage preregistration identity mismatch")

    queue_key = load_key(operator_queue_key)
    operator = json.loads(operator_queue.read_text())
    verify_document(operator, queue_key, b"meanaudio-operator-queue-v1\0")
    if operator.get("document_kind") != "operator_approved_experiment_backlog":
        raise RuntimeError("stage signed operator queue document kind is invalid")
    matches = [item for item in operator.get("entries", [])
               if item.get("experiment_id") == experiment_id and item.get("run_id") == run_id]
    if len(matches) != 1:
        raise RuntimeError("stage has no unique exact operator queue entry")
    operator_entry = matches[0]
    if (operator_entry.get("approval_status") != "approved"
            or Path(str(operator_entry.get("contract", ""))) != prereg
            or operator_entry.get("contract_sha256") != prereg_hash):
        raise RuntimeError("stage operator queue entry is not exact and approved")
    binding = entry_binding(operator_entry)
    evidence = operator_entry.get("approval_evidence", {})
    approval_record = Path(str(evidence.get("path", "")))
    if (not approval_record.is_file() or approval_record.is_symlink()
            or digest(approval_record) != evidence.get("sha256")):
        raise RuntimeError("stage exact approval evidence drift")
    record = json.loads(approval_record.read_text())
    verify_document(record, queue_key, b"meanaudio-queue-approval-v1\0")
    expected_record = {
        "document_kind": "exact_operator_approval", "status": "approved",
        "experiment_id": experiment_id, "run_id": run_id,
        "contract_sha256": prereg_hash,
        "controller_sha256": operator_entry.get("runtime", {}).get("controller_sha256"),
        "queue_entry_binding_sha256": binding,
    }
    if any(record.get(key) != value for key, value in expected_record.items()):
        raise RuntimeError("stage exact approval record mismatch")

    generation, values = load_generation(state)
    contract, preflight, ledger, child_queue = (values[name] for name in ("contract", "preflight", "ledger", "queue"))
    for value in (contract, preflight, ledger):
        if value.get("experiment_id") != experiment_id or value.get("run_id") != run_id:
            raise RuntimeError("stage terminal bundle identity mismatch")
    source_map = {item.get("path"): item.get("sha256") for item in
                  contract.get("corpus", {}).get("source_artifacts", []) if isinstance(item, dict)}
    authorization_path = state / "stage_authorization.json"
    if (not authorization_path.is_file() or authorization_path.is_symlink()
            or source_map.get(str(authorization_path)) != digest(authorization_path)):
        raise RuntimeError("stage terminal contract lacks its exact authorization artifact")
    authorization = json.loads(authorization_path.read_text())
    if (source_map.get(str(prereg)) != prereg_hash
            or source_map.get(str(approval_record)) != evidence.get("sha256")
            or authorization.get("queue_entry_binding_sha256") != binding
            or authorization.get("approval_record_sha256") != evidence.get("sha256")):
        raise RuntimeError("stage terminal contract lacks exact approval/entry bindings")

    contract_hash = hashlib.sha256(canonical(contract)).hexdigest()
    preflight_hash = hashlib.sha256(canonical(preflight)).hexdigest()
    ledger_hash = hashlib.sha256(canonical(ledger)).hexdigest()
    entries = child_queue.get("entries", [])
    if len(entries) != 1:
        raise RuntimeError("stage terminal queue must contain exactly one entry")
    child = entries[0]
    if (contract.get("document_kind") != "experiment_contract"
            or preflight.get("document_kind") != "preflight_report"
            or ledger.get("document_kind") != "event_ledger"
            or child_queue.get("document_kind") != "queue_state"
            or child.get("experiment_id") != experiment_id or child.get("run_id") != run_id
            or child.get("status") != "completed"
            or child.get("terminal_notification_status") != "delivered"
            or preflight.get("contract_raw_sha256") != contract_hash
            or ledger.get("bindings", {}).get("contract_raw_sha256") != contract_hash
            or ledger.get("bindings", {}).get("preflight_report_raw_sha256") != preflight_hash
            or child.get("bindings", {}).get("contract_raw_sha256") != contract_hash
            or child.get("bindings", {}).get("preflight_report_raw_sha256") != preflight_hash
            or child.get("bindings", {}).get("ledger_raw_sha256") != ledger_hash):
        raise RuntimeError("stage completed bundle raw-hash edges are invalid")

    key_descriptor = operator_entry.get("runtime", {}).get("child_ledger_key", {})
    key_path = Path(str(key_descriptor.get("path", "")))
    if (not key_path.is_file() or key_path.is_symlink() or digest(key_path) != key_descriptor.get("sha256")
            or authorization.get("child_ledger_key") != key_descriptor):
        raise RuntimeError("stage child-ledger key is not exact approval-bound evidence")
    child_key = load_key(key_path)
    verify_document(authorization, child_key, b"meanaudio-nvme-stage-exact-authorization-v1\0")
    if (authorization.get("document_kind") != "nvme_stage_exact_authorization"
            or authorization.get("experiment_id") != experiment_id or authorization.get("run_id") != run_id):
        raise RuntimeError("stage exact authorization identity mismatch")
    prior = None
    completion_id = None
    delivered_relations: set[str] = set()
    for sequence, event in enumerate(ledger.get("events", []), 1):
        supplied = event.get("event_sha256") if isinstance(event, dict) else None
        unsigned = {name: value for name, value in event.items() if name != "event_sha256"}
        expected = hmac.new(child_key, b"meanaudio-harn-event-v1\0" + canonical(unsigned), hashlib.sha256).hexdigest()
        if (event.get("sequence") != sequence or event.get("previous_event_sha256") != prior
                or not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected)):
            raise RuntimeError("stage child ledger HMAC chain invalid")
        prior = supplied
        if event.get("event_kind") == "experiment_completed":
            completion_id = event.get("event_id")
        if event.get("event_kind") == "notification_delivery" and event.get("notification_status") == "delivered":
            delivered_relations.add(str(event.get("relates_to_event_id")))
    if completion_id is None or completion_id not in delivered_relations:
        raise RuntimeError("stage authenticated terminal notification is not delivered")

    approval_state_path = state / "operator_approval.json"
    if not approval_state_path.is_file() or approval_state_path.is_symlink():
        raise RuntimeError("stage approval state is missing or unsafe")
    approval_state = json.loads(approval_state_path.read_text())
    verify_document(approval_state, child_key, b"meanaudio-nvme-stage-approval-v1\0")
    if (approval_state.get("experiment_id") != experiment_id or approval_state.get("run_id") != run_id
            or approval_state.get("state") != "consumed"
            or approval_state.get("approval_record_sha256") != evidence.get("sha256")
            or approval_state.get("queue_entry_binding_sha256") != binding):
        raise RuntimeError("stage approval was not exact-bound and consumed")

    descriptor = prereg_value.get("report_validator", {})
    argv = descriptor.get("argv")
    bindings = descriptor.get("bindings")
    if not isinstance(argv, list) or not argv or not isinstance(bindings, list) or not bindings:
        raise RuntimeError("stage report completion validator is missing")
    bound = {str(item.get("path")): item.get("sha256") for item in bindings if isinstance(item, dict)}
    if len(bound) != len(bindings):
        raise RuntimeError("stage report validator has duplicate/invalid bindings")
    for path, expected_hash in bound.items():
        candidate = Path(path)
        if not candidate.is_file() or candidate.is_symlink() or digest(candidate) != expected_hash:
            raise RuntimeError(f"stage report validator binding drift: {path}")
    if argv[0] not in bound or any(
        item.startswith("/") and Path(item).is_file() and item not in bound for item in argv[1:]
    ):
        raise RuntimeError("stage report validator command contains unbound executable input")
    completed = subprocess.run(
        argv, cwd="/", text=True, capture_output=True, timeout=120,
        env={"PATH": CONTRACT_PATH, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode:
        raise RuntimeError(f"stage exact report validator failed: {completed.stderr[-1000:]}")
    print(f"[VALID] authenticated completed NVMe stage bundle {generation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=STATE)
    parser.add_argument("--preregistered-contract", type=Path, default=PREREG)
    parser.add_argument("--operator-queue", type=Path, default=OPERATOR_QUEUE)
    parser.add_argument("--operator-queue-key", type=Path, default=OPERATOR_QUEUE_KEY)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    validate_bundle(state=args.state_root, prereg=args.preregistered_contract,
                    operator_queue=args.operator_queue, operator_queue_key=args.operator_queue_key,
                    experiment_id=args.experiment_id, run_id=args.run_id)


if __name__ == "__main__":
    main()
