#!/usr/bin/env python3
"""CPU-only positive and abuse-case tests for HARN-SCHEMA-V1."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_experiment_harness_documents as validator  # noqa: E402


NOW = datetime.now(timezone.utc).replace(microsecond=0)
GIB = 1024**3


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def timestamp(delta: timedelta) -> str:
    return (NOW + delta).isoformat().replace("+00:00", "Z")


def command(action_id: str) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "argv": ["/usr/bin/true"],
        "working_directory": "/home/kojiek/MeanAudio",
        "environment": {"PYTHONUNBUFFERED": "1"},
    }


def make_contract(*, generated: bool, repair_enabled: bool) -> dict[str, Any]:
    action_ids = ["run", "resume"]
    if repair_enabled:
        action_ids.extend(["repair_test", "repair_apply", "repair_rollback", "repair_resume"])
    corpus: dict[str, Any]
    checks = ["policy", "provenance", "storage"]
    if generated:
        checks.append("generated_corpus_full_gate")
        corpus = {
            "kind": "generated",
            "corpus_artifact": {"path": "/data/corpus.jsonl", "sha256": digest("corpus")},
            "corpus_schema_sha256": digest("corpus-schema"),
            "classifier_version": "classifier-v1",
            "defect_taxonomy_sha256": digest("taxonomy"),
            "stop_behavior_test_id": "stop-token-v1",
            "full_gate_report": {"path": "/reports/corpus-gate.json", "sha256": digest("gate")},
            "downstream_bindings": {
                "tsv_manifest_sha256": digest("tsv"),
                "feature_cache_report_sha256": digest("feature-cache"),
            },
            "required_gate_points": ["launch", "pre_training", "post_change"],
        }
    else:
        corpus = {
            "kind": "non_generated",
            "source_artifacts": [{"path": "/data/source.tsv", "sha256": digest("source")}],
        }

    repair: dict[str, Any] = {"enabled": False}
    if repair_enabled:
        repair = {
            "enabled": True,
            "envelope": {
                "envelope_sha256": digest("repair-envelope"),
                "writable_paths": ["/home/kojiek/MeanAudio/worktrees/repair"],
                "test_action_ids": ["repair_test"],
                "apply_action_id": "repair_apply",
                "rollback_action_id": "repair_rollback",
                "resume_action_id": "repair_resume",
                "allowed_process_identities": ["experiment-child"],
                "reviewer_roles": ["security-reviewer"],
                "budgets": {
                    "max_model_calls": 1,
                    "max_wall_seconds": 3600,
                    "max_transient_retries": 1,
                    "max_cost_units": 100,
                },
                "operator_required_conditions": ["scientific_contract_change", "shared_host_change"],
            },
        }
    return {
        "document_kind": "experiment_contract",
        "schema_version": "1.0.0",
        "schema_bundle_id": validator.BUNDLE_ID,
        "experiment_id": "schema-selftest",
        "run_id": "run-001",
        "bindings": {
            "policy_bundle_sha256": digest("policy"),
            "schema_bundle_sha256": digest("schemas"),
            "runtime_sha256": digest("runtime"),
            "command_set_sha256": digest("commands"),
        },
        "approval_requirement": {
            "required": True,
            "responsible_role": "responsible-operator",
            "trusted_channels": ["signed_approval_store"],
        },
        "corpus": corpus,
        "repair": repair,
        "phases": [{
            "phase_id": "train",
            "action_id": "run",
            "input_artifacts": [{"path": "/data/input.bin", "sha256": digest("input")}],
            "output_paths": ["/artifacts/run-001"],
            "completion_evidence": [{"path": "/reports/final.json", "sha256": digest("final")}],
            "resume_action_id": "resume",
        }],
        "filesystems": [{
            "path": "/artifacts",
            "hard_floor_bytes": 60 * GIB,
            "warning_floor_bytes": 80 * GIB,
            "peak_additional_bytes": 10 * GIB,
            "transient_bytes": 2 * GIB,
            "recovery_reserve_bytes": 4 * GIB,
        }],
        "commands": [command(action_id) for action_id in action_ids],
        "required_preflight_checks": checks,
        "notification_events": ["start", "gate", "terminal"],
    }


def make_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    contract_hash = digest("raw-contract")
    envelope_hash = (
        contract["repair"]["envelope"]["envelope_sha256"]
        if contract["repair"]["enabled"] else None
    )
    return {
        "document_kind": "preflight_report",
        "schema_version": "1.0.0",
        "schema_bundle_id": validator.BUNDLE_ID,
        "experiment_id": contract["experiment_id"],
        "run_id": contract["run_id"],
        "contract_raw_sha256": contract_hash,
        "approval_evidence": {
            "evidence_id": "approval-001",
            "source_kind": "trusted_operator_record",
            "trusted_channel": "signed_approval_store",
            "channel_record_id": "change-001",
            "channel_record_sha256": digest("approval-channel-record"),
            "approver_id": "operator-001",
            "issued_at": timestamp(timedelta(hours=-2)),
            "expires_at": timestamp(timedelta(hours=2)),
            "experiment_id": contract["experiment_id"],
            "run_id": contract["run_id"],
            "bindings": {
                "contract_raw_sha256": contract_hash,
                "policy_bundle_sha256": contract["bindings"]["policy_bundle_sha256"],
                "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"],
                "runtime_sha256": contract["bindings"]["runtime_sha256"],
                "repair_envelope_sha256": envelope_hash,
                "command_set_sha256": contract["bindings"]["command_set_sha256"],
            },
        },
        "checks": [{
            "check_id": check_id,
            "verdict": "pass",
            "observed_at": timestamp(timedelta(minutes=-10)),
            "valid_until": timestamp(timedelta(hours=1)),
            "evidence_sha256": digest(f"check:{check_id}"),
        } for check_id in contract["required_preflight_checks"]],
        "storage": [{
            "path": filesystem["path"],
            "measured_at": timestamp(timedelta(minutes=-5)),
            "free_bytes": 100 * GIB,
            "hard_floor_bytes": filesystem["hard_floor_bytes"],
            "peak_additional_bytes": filesystem["peak_additional_bytes"],
            "transient_bytes": filesystem["transient_bytes"],
            "recovery_reserve_bytes": filesystem["recovery_reserve_bytes"],
            "verdict": "pass",
        } for filesystem in contract["filesystems"]],
        "derived_verdict": "pass",
        "created_at": timestamp(timedelta(minutes=-1)),
    }


def ledger_event(
    sequence: int, kind: str, *, verdict: str = "none", relation: str | None = None,
    notification: str = "not_applicable"
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "event_id": f"event-{sequence}",
        "idempotency_key": f"schema-selftest:run-001:{sequence}",
        "event_kind": kind,
        "occurred_at": timestamp(timedelta(minutes=-20 + sequence)),
        "phase": "train" if sequence >= 4 else None,
        "verdict": verdict,
        "relates_to_event_id": relation,
        "notification_status": notification,
        "previous_event_sha256": None if sequence == 1 else digest(f"event:{sequence - 1}"),
        "event_sha256": digest(f"event:{sequence}"),
    }


def make_ledger(contract: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    events = [
        ledger_event(1, "contract_registered"),
        ledger_event(2, "preflight_passed", verdict="pass"),
        ledger_event(3, "resources_acquired"),
        ledger_event(4, "experiment_started"),
        ledger_event(5, "gate_result", verdict="pass"),
        ledger_event(6, "notification_delivery", relation="event-5", notification="delivered"),
        ledger_event(7, "promotion_started", relation="event-5"),
        ledger_event(8, "experiment_completed", verdict="pass", notification="pending"),
        ledger_event(9, "notification_delivery", relation="event-8", notification="delivered"),
    ]
    return {
        "document_kind": "event_ledger",
        "schema_version": "1.0.0",
        "schema_bundle_id": validator.BUNDLE_ID,
        "experiment_id": contract["experiment_id"],
        "run_id": contract["run_id"],
        "bindings": {
            "contract_raw_sha256": preflight["contract_raw_sha256"],
            "preflight_report_raw_sha256": digest("raw-preflight"),
            "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"],
        },
        "events": events,
    }


def make_queue(contract: dict[str, Any], preflight: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_kind": "queue_state",
        "schema_version": "1.0.0",
        "schema_bundle_id": validator.BUNDLE_ID,
        "queue_id": "selftest-queue",
        "updated_at": timestamp(timedelta(minutes=-1)),
        "entries": [{
            "entry_id": "queue-entry-1",
            "position": 1,
            "experiment_id": contract["experiment_id"],
            "run_id": contract["run_id"],
            "status": "completed",
            "dependencies": [],
            "assigned_resource": None,
            "bindings": {
                "contract_raw_sha256": preflight["contract_raw_sha256"],
                "preflight_report_raw_sha256": ledger["bindings"]["preflight_report_raw_sha256"],
                "ledger_raw_sha256": digest("raw-ledger"),
                "schema_bundle_sha256": contract["bindings"]["schema_bundle_sha256"],
            },
            "terminal_notification_status": "delivered",
        }],
    }


def make_bundle(*, generated: bool = False, repair_enabled: bool = False) -> list[dict[str, Any]]:
    contract = make_contract(generated=generated, repair_enabled=repair_enabled)
    preflight = make_preflight(contract)
    ledger = make_ledger(contract, preflight)
    queue = make_queue(contract, preflight, ledger)
    return [contract, preflight, ledger, queue]


def assert_valid(name: str, bundle: list[dict[str, Any]], schemas: dict[str, Any]) -> None:
    issues = validator.validate_documents(*bundle, now=NOW, schemas=schemas)
    if issues:
        rendered = "\n".join(issue.render() for issue in issues)
        raise AssertionError(f"{name}: expected valid bundle:\n{rendered}")


def assert_invalid(
    name: str, mutate: Callable[[list[dict[str, Any]]], None], schemas: dict[str, Any],
    expected_code: str | None = None
) -> None:
    bundle = copy.deepcopy(make_bundle())
    mutate(bundle)
    issues = validator.validate_documents(*bundle, now=NOW, schemas=schemas)
    if not issues:
        raise AssertionError(f"{name}: invalid fixture unexpectedly passed")
    if expected_code is not None and not any(issue.code == expected_code for issue in issues):
        codes = ", ".join(issue.code for issue in issues)
        raise AssertionError(f"{name}: expected {expected_code}, observed {codes}")


def append_queue_entry(queue: dict[str, Any], *, entry_id: str, position: int) -> dict[str, Any]:
    entry = copy.deepcopy(queue["entries"][0])
    entry.update({
        "entry_id": entry_id,
        "position": position,
        "experiment_id": f"other-{entry_id}",
        "run_id": f"run-{position}",
        "status": "ready",
        "terminal_notification_status": "not_applicable",
    })
    queue["entries"].append(entry)
    return entry


def assert_loader_rejects(
    temp_dir: Path, name: str, raw: bytes, *, forbidden_output: bytes | None = None
) -> None:
    path = temp_dir / f"{name}.json"
    path.write_bytes(raw)
    try:
        validator.load_json_document(path)
    except validator.DocumentLoadError as exc:
        if forbidden_output is not None and forbidden_output.decode("ascii") in str(exc):
            raise AssertionError(f"{name}: loader error exposed rejected secret material") from None
    else:
        raise AssertionError(f"{name}: unsafe JSON was accepted")


def run_loader_tests(temp_dir: Path) -> int:
    duplicate = temp_dir / "duplicate.json"
    duplicate.write_text('{"document_kind":"x","document_kind":"y"}', encoding="utf-8")
    try:
        validator.load_json_document(duplicate)
    except validator.DuplicateKeyError:
        pass
    else:
        raise AssertionError("duplicate JSON key was accepted")

    secret_cases = {
        "secret-key": (b'{"api_key":"SECRET_VALUE_SENTINEL"}', b"SECRET_VALUE_SENTINEL"),
        "nested-token-key": (b'{"outer":{"token":"SECRET_VALUE_SENTINEL"}}', b"SECRET_VALUE_SENTINEL"),
        "nested-credential-key": (
            b'{"outer":{"credential":"SECRET_VALUE_SENTINEL"}}', b"SECRET_VALUE_SENTINEL"
        ),
        "nested-authorization-key": (
            b'{"outer":{"authorization":"SECRET_VALUE_SENTINEL"}}', b"SECRET_VALUE_SENTINEL"
        ),
        "nested-camel-token-key": (
            b'{"outer":{"accessToken":"SECRET_VALUE_SENTINEL"}}', b"SECRET_VALUE_SENTINEL"
        ),
        "discord-webhook": (
            b'{"value":"https://discord.com/api/webhooks/1/SECRET_VALUE_SENTINEL"}',
            b"SECRET_VALUE_SENTINEL",
        ),
        "discord-token": (
            b'{"value":"mfa.AAAAAAAAAAAAAAAAAAAAAAAA"}', b"mfa.AAAAAAAAAAAAAAAAAAAAAAAA"
        ),
        "slack-webhook": (
            b'{"value":"https://hooks.slack.com/services/T000/B000/SECRET_VALUE_SENTINEL"}',
            b"SECRET_VALUE_SENTINEL",
        ),
        "github-token": (
            b'{"value":"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}',
            b"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ),
        "github-pat": (
            b'{"value":"github_pat_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}',
            b"github_pat_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ),
        "bearer-token": (
            b'{"value":"Bearer SECRET_VALUE_SENTINEL_123456"}', b"SECRET_VALUE_SENTINEL_123456"
        ),
    }
    for name, (raw, forbidden_output) in secret_cases.items():
        assert_loader_rejects(temp_dir, name, raw, forbidden_output=forbidden_output)

    assert_loader_rejects(temp_dir, "overflowing-float", b'{"value":1e1000000}')

    bounded_false_positives = temp_dir / "bounded-false-positives.json"
    bounded_false_positives.write_text(
        '{"tokenizer_version":"v1","token_count":42,"authorizer_version":"v2"}',
        encoding="utf-8",
    )
    validator.load_json_document(bounded_false_positives)
    return 2 + len(secret_cases)


def write_bundle(directory: Path, bundle: list[dict[str, Any]]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    contract, preflight, ledger, queue = bundle
    paths = [directory / f"{name}.json" for name in ("contract", "preflight", "ledger", "queue")]

    def write_document(path: Path, document: dict[str, Any]) -> str:
        raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    contract_hash = write_document(paths[0], contract)
    preflight["contract_raw_sha256"] = contract_hash
    preflight["approval_evidence"]["bindings"]["contract_raw_sha256"] = contract_hash
    preflight_hash = write_document(paths[1], preflight)
    ledger["bindings"]["contract_raw_sha256"] = contract_hash
    ledger["bindings"]["preflight_report_raw_sha256"] = preflight_hash
    ledger_hash = write_document(paths[2], ledger)
    queue_binding = queue["entries"][0]["bindings"]
    queue_binding["contract_raw_sha256"] = contract_hash
    queue_binding["preflight_report_raw_sha256"] = preflight_hash
    queue_binding["ledger_raw_sha256"] = ledger_hash
    write_document(paths[3], queue)
    return paths


def run_cli(paths: list[Path]) -> subprocess.CompletedProcess[str]:
    command_line = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_experiment_harness_documents.py"),
        "--contract", str(paths[0]), "--preflight", str(paths[1]),
        "--ledger", str(paths[2]), "--queue", str(paths[3]),
    ]
    return subprocess.run(command_line, cwd=REPO_ROOT, text=True, capture_output=True,
                          timeout=30, check=False)


def run_cli_test(temp_dir: Path, bundle: list[dict[str, Any]]) -> None:
    paths = write_bundle(temp_dir / "positive", bundle)
    completed = run_cli(paths)
    if completed.returncode != 0:
        raise AssertionError(f"CLI positive fixture failed:\n{completed.stdout}\n{completed.stderr}")
    if validator.NOTICE not in completed.stdout:
        raise AssertionError("CLI omitted the non-authorization notice")

    for index, expected_code in (
        (0, "raw.contract_binding"),
        (1, "raw.preflight_binding"),
        (2, "raw.ledger_binding"),
    ):
        mutation_dir = temp_dir / f"raw-mutation-{index}"
        mutation_paths = write_bundle(mutation_dir, make_bundle())
        mutation_paths[index].write_bytes(mutation_paths[index].read_bytes() + b"\n")
        rejected = run_cli(mutation_paths)
        if rejected.returncode == 0 or expected_code not in rejected.stderr:
            raise AssertionError(
                f"CLI accepted actual raw-byte mutation for document {index}:\n"
                f"{rejected.stdout}\n{rejected.stderr}"
            )


def main() -> int:
    if validator.jsonschema is None:
        print("FAIL: jsonschema >=4.10,<5 is required", file=sys.stderr)
        return 2
    schemas = validator.load_schemas()
    assert_valid("non-generated + repair disabled", make_bundle(), schemas)
    assert_valid("generated + repair enabled", make_bundle(generated=True, repair_enabled=True), schemas)

    tests: list[tuple[str, Callable[[list[dict[str, Any]]], None], str | None]] = [
        ("unknown kind/version", lambda b: b[0].update(document_kind="unknown", schema_version="9"), "schema.const"),
        ("missing generated gate", lambda b: (
            b.__setitem__(slice(None), make_bundle(generated=True)),
            b[0]["required_preflight_checks"].remove("generated_corpus_full_gate"),
            b[1]["checks"].pop(),
        ), "contract.generated_gate_missing"),
        ("repair-disabled with envelope", lambda b: b[0]["repair"].update(envelope={}), "schema.oneOf"),
        ("expired approval", lambda b: b[1]["approval_evidence"].update(
            issued_at=timestamp(timedelta(hours=-3)), expires_at=timestamp(timedelta(hours=-1))),
         "preflight.approval_expired"),
        ("mismatched approval hash", lambda b: b[1]["approval_evidence"]["bindings"].update(
            contract_raw_sha256=digest("wrong")), "preflight.approval_binding_mismatch"),
        ("bash -c", lambda b: b[0]["commands"][0].update(argv=["/bin/bash", "-c", "true"]),
         "contract.inline_code"),
        ("python -c", lambda b: b[0]["commands"][0].update(argv=["/usr/bin/python3", "-c", "pass"]),
         "contract.inline_code"),
        ("php -r", lambda b: b[0]["commands"][0].update(argv=["/usr/bin/php", "-r", "echo 1;"]),
         "contract.inline_code"),
        ("lua -e", lambda b: b[0]["commands"][0].update(argv=["/usr/bin/lua", "-e", "print(1)"]),
         "contract.inline_code"),
        ("Rscript -e", lambda b: b[0]["commands"][0].update(argv=["/usr/bin/Rscript", "-e", "print(1)"]),
         "contract.inline_code"),
        ("awk inline program", lambda b: b[0]["commands"][0].update(
            argv=["/usr/bin/awk", "{print $1}", "/data/input.tsv"]), "contract.inline_code"),
        ("path traversal", lambda b: b[0]["commands"][0].update(working_directory="/tmp/../root"),
         "contract.unsafe_path"),
        ("bare parent traversal", lambda b: b[0]["commands"][0].update(
            argv=["/usr/bin/true", ".."]), "contract.command_path_traversal"),
        ("option parent traversal", lambda b: b[0]["commands"][0].update(
            argv=["/usr/bin/true", "--path=.."]), "contract.command_path_traversal"),
        ("relative command path", lambda b: b[0]["commands"][0].update(
            argv=["/usr/bin/python3", "scripts/train.py"]), "contract.relative_command_path"),
        ("arbitrary environment", lambda b: b[0]["commands"][0]["environment"].update(
            UNREGISTERED_VALUE="1"), "schema.additionalProperties"),
        ("failed preflight check", lambda b: b[1]["checks"][0].update(verdict="fail"),
         "preflight.check_not_passed"),
        ("missing preflight check", lambda b: b[1]["checks"].pop(), "preflight.check_set_mismatch"),
        ("stale preflight check", lambda b: b[1]["checks"][0].update(
            valid_until=timestamp(timedelta(minutes=-1))), "preflight.stale_check"),
        ("insufficient storage", lambda b: (
            b[1]["storage"][0].update(free_bytes=1, verdict="fail"),
            b[1].update(derived_verdict="fail"),
        ), "preflight.insufficient_storage"),
        ("duplicate ledger event id", lambda b: b[2]["events"][1].update(event_id="event-1"),
         "ledger.duplicate_event_id"),
        ("duplicate ledger idempotency id", lambda b: b[2]["events"][1].update(
            idempotency_key=b[2]["events"][0]["idempotency_key"]), "ledger.duplicate_idempotency_key"),
        ("gapped ledger sequence", lambda b: b[2]["events"][3].update(sequence=99),
         "ledger.sequence_gap"),
        ("broken ledger hash chain", lambda b: b[2]["events"][3].update(
            previous_event_sha256=digest("wrong")), "ledger.hash_chain"),
        ("promotion before delivered notification", lambda b: b[2]["events"][5].update(
            notification_status="failed"), "ledger.promotion_notification_order"),
        ("duplicate queue id", lambda b: append_queue_entry(b[3], entry_id="queue-entry-1", position=2),
         "queue.duplicate_entry"),
        ("missing queue dependency", lambda b: b[3]["entries"][0].update(dependencies=["absent"]),
         "queue.unknown_dependency"),
        ("cyclic queue dependencies", lambda b: (
            append_queue_entry(b[3], entry_id="queue-entry-2", position=2).update(dependencies=["queue-entry-1"]),
            b[3]["entries"][0].update(dependencies=["queue-entry-2"]),
        ), "queue.dependency_cycle"),
        ("two active entries sharing GPU", lambda b: (
            b[3]["entries"][0].update(status="active", terminal_notification_status="not_applicable",
                                      assigned_resource={"resource_type": "gpu", "resource_id": "gpu-0"}),
            append_queue_entry(b[3], entry_id="queue-entry-2", position=2).update(
                status="active", assigned_resource={"resource_type": "gpu", "resource_id": "gpu-0"}),
        ), "queue.resource_conflict"),
        ("completed with pending terminal notification", lambda b: b[3]["entries"][0].update(
            terminal_notification_status="pending"), "queue.terminal_notification"),
        ("cross-document hash mismatch", lambda b: b[3]["entries"][0]["bindings"].update(
            contract_raw_sha256=digest("wrong")), "queue.binding_mismatch"),
    ]
    for name, mutate, expected_code in tests:
        assert_invalid(name, mutate, schemas, expected_code)

    raw_required = validator.validate_documents(
        *make_bundle(), now=NOW, schemas=schemas, require_raw_hashes=True
    )
    if not any(issue.code == "raw.hashes_required" for issue in raw_required):
        raise AssertionError("launch-ready in-memory validation did not require exact raw hashes")

    with tempfile.TemporaryDirectory(prefix="meanaudio-harness-schema-") as temporary:
        temp_dir = Path(temporary)
        loader_rejections = run_loader_tests(temp_dir)
        run_cli_test(temp_dir, make_bundle())

    print(
        f"PASS: 2 positive modes, {len(tests)} semantic/schema abuse cases, "
        f"{loader_rejections} strict-loader rejections, 3 actual raw-byte mutation cases, "
        "explicit in-memory raw-hash gating, offline schema meta-validation, and CLI notice"
    )
    print(validator.NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
