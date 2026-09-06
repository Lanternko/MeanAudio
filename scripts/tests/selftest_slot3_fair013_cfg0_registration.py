#!/usr/bin/env python3
"""No-GPU abuse and continuity fixtures for slot3 CFG0 registration."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import install_slot3_cfg0_interceptor as installer  # noqa: E402
import register_slot3_fair013_cfg0 as reg  # noqa: E402
import validate_slot3_fair013_cfg0_registration as registration_validator  # noqa: E402


def expect_hold(call: Callable[[], Any], contains: str | None = None) -> None:
    try:
        call()
    except (OSError, reg.RegistrationError) as exc:
        if contains is not None:
            assert contains in str(exc), (contains, str(exc))
        return
    raise AssertionError("expected fail-closed registration hold")


def process_set() -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    for child in Path("/proc").iterdir():
        if not child.name.isdecimal():
            continue
        try:
            cmdline = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if " eval.py " in f" {cmdline} " or "phase4_eval.py" in cmdline:
            result.add((int(child.name), cmdline))
    return result


def fake_checkpoint(arm: dict[str, Any], discriminator: int = 0) -> dict[str, Any]:
    return {
        "path": arm["checkpoint"],
        "sha256": hashlib.sha256(f"{arm['arm_id']}:{discriminator}".encode()).hexdigest(),
        "device": 1,
        "inode": 1000 + arm["sequence"],
        "owner_uid": os.geteuid(),
        "owner_gid": os.getegid(),
        "mode": "0600",
        "link_count": 1,
        "size": 16,
        "mtime_ns": 1,
    }


def fake_caller(template: dict[str, Any]) -> dict[str, Any]:
    return {
        **template["caller"],
        "script_owner_uid": template["caller"]["uid"],
        "hook_pid": 4000001,
        "hook_uid": template["caller"]["uid"],
        "hook_parent_pid": template["caller"]["pid"],
        "hook_executable": "/usr/bin/bash",
        "hook_script_fd": 255,
        "hook_script_sha256": "a" * 64,
        "hook_script_inode": 777,
    }


def arm_argv(arm: dict[str, Any]) -> list[str]:
    return [arm["legacy_label"], arm["checkpoint"], *arm["legacy_conditioning_argv"]]


def invoke_registration(
    template: dict[str, Any], paths: reg.StorePaths, arm: dict[str, Any],
    notify: Callable[[dict[str, Any]], None], *, fault: Callable[[str], None] | None = None,
    checkpoint_discriminator: int = 0,
) -> dict[str, Any]:
    return reg.register_once(
        template=template,
        argv=arm_argv(arm),
        caller=fake_caller(template),
        paths=paths,
        manifest_sha256="b" * 64,
        approval_sha256="e" * 64,
        notify=notify,
        checkpoint_identity=fake_checkpoint(arm, checkpoint_discriminator),
        fault=fault,
    )


def request_counter(counts: dict[str, int], *, fail_after_request: bool = False) -> Callable[[dict[str, Any]], None]:
    def notify(payload: dict[str, Any]) -> None:
        event_id = payload["event_id"]
        counts[event_id] = counts.get(event_id, 0) + 1
        if fail_after_request:
            raise reg.RegistrationError("injected crash during notifier request")
    return notify


def fault_at(target: str) -> Callable[[str], None]:
    def inject(point: str) -> None:
        if point == target:
            raise reg.RegistrationError(f"injected crash at {target}")
    return inject


def event_unsigned(paths: reg.StorePaths, arm: dict[str, Any]) -> dict[str, Any]:
    key = reg.create_or_load_key(paths.key)
    path = paths.outbox / f"{arm['sequence']:02d}_{arm['arm_id']}.json"
    return reg.verify_signed(reg.load_json(path), key, reg.NOTIFICATION_DOMAIN)


def run_notification_crash_fixtures(template: dict[str, Any]) -> None:
    scenarios = (
        "before_state",
        "after_attempting_before_send",
        "during_notifier_request",
        "after_notifier_return_before_delivered",
        "after_delivered_before_state",
    )
    for arm_index, arm in enumerate(template["arms"]):
        for scenario in scenarios:
            with tempfile.TemporaryDirectory(prefix=f"slot3-cfg0-{arm['arm_id']}-{scenario}-") as directory:
                paths = reg.StorePaths.under(Path(directory) / "runtime")
                counts: dict[str, int] = {}
                for prefix in template["arms"][:arm_index]:
                    result = invoke_registration(template, paths, prefix, request_counter(counts))
                    assert result["status"] == "accepted"
                prefix_request_count = sum(counts.values())

                if scenario == "during_notifier_request":
                    expect_hold(
                        lambda: invoke_registration(
                            template, paths, arm, request_counter(counts, fail_after_request=True)
                        ),
                        "during notifier",
                    )
                else:
                    expect_hold(
                        lambda scenario=scenario: invoke_registration(
                            template, paths, arm, request_counter(counts), fault=fault_at(scenario)
                        ),
                        "injected crash",
                    )

                target_requests = sum(counts.values()) - prefix_request_count
                if scenario == "before_state":
                    if arm_index == 0:
                        assert not paths.state.exists() and not paths.state.is_symlink()
                    else:
                        key = reg.create_or_load_key(paths.key)
                        state = reg.load_state(
                            paths.root, key, reg.sha256_bytes(reg.canonical(template)),
                            "b" * 64, "e" * 64,
                        )
                        assert reg.verify_signed(state, key, reg.STATE_DOMAIN)["sequence"] == arm_index
                    assert not (paths.outbox / f"{arm['sequence']:02d}_{arm['arm_id']}.json").exists()
                    result = invoke_registration(template, paths, arm, request_counter(counts))
                    assert result["status"] == "accepted"
                    assert sum(counts.values()) - prefix_request_count == 1
                    continue

                if scenario == "after_delivered_before_state":
                    assert target_requests == 1
                    assert event_unsigned(paths, arm)["delivery_status"] == "delivered"
                    result = invoke_registration(template, paths, arm, request_counter(counts))
                    assert result["status"] == "accepted"
                    replay = invoke_registration(template, paths, arm, request_counter(counts))
                    assert replay["status"] == "idempotent"
                    assert sum(counts.values()) - prefix_request_count == 1
                    continue

                expected_requests = 0 if scenario == "after_attempting_before_send" else 1
                assert target_requests == expected_requests
                assert event_unsigned(paths, arm)["delivery_status"] == "attempting"
                expect_hold(
                    lambda: invoke_registration(template, paths, arm, request_counter(counts)),
                    "delivery_ambiguous",
                )
                ambiguous = event_unsigned(paths, arm)
                assert ambiguous["delivery_status"] == "delivery_ambiguous"
                assert ambiguous["automatic_resend_allowed"] is False
                expect_hold(
                    lambda: invoke_registration(template, paths, arm, request_counter(counts)),
                    "delivery_ambiguous",
                )
                assert sum(counts.values()) - prefix_request_count == expected_requests


def run_registration_fixtures(template: dict[str, Any]) -> None:
    manifest_hash = "b" * 64
    approval_hash = "e" * 64
    caller = fake_caller(template)
    delivered: list[str] = []
    queue_before = reg.sha256_path(reg.TOP_QUEUE)
    gpu_before = process_set()
    with tempfile.TemporaryDirectory(prefix="slot3-cfg0-selftest-") as directory:
        paths = reg.StorePaths.under(Path(directory) / "runtime")
        expect_hold(lambda: reg.register_once(
            template=template, argv=arm_argv(template["arms"][1]), caller=caller,
            paths=paths, manifest_sha256=manifest_hash, approval_sha256=approval_hash,
            notify=lambda _: None,
            checkpoint_identity=fake_checkpoint(template["arms"][1]),
        ), "arm order")

        parent_trace = ["slot3_caption_complete", "k3_training_complete"]
        for index, arm in enumerate(template["arms"]):
            def successful_notify(payload: dict[str, Any]) -> None:
                delivered.append(payload["event_id"])

            result = reg.register_once(
                template=template, argv=arm_argv(arm), caller=caller,
                paths=paths, manifest_sha256=manifest_hash, approval_sha256=approval_hash,
                notify=successful_notify,
                checkpoint_identity=fake_checkpoint(arm),
            )
            assert result["status"] == "accepted"
            replay = reg.register_once(
                template=template, argv=arm_argv(arm), caller={**caller, "hook_pid": caller["hook_pid"] + 1},
                paths=paths, manifest_sha256=manifest_hash, approval_sha256=approval_hash,
                notify=lambda _: (_ for _ in ()).throw(AssertionError("idempotent replay notified twice")),
                checkpoint_identity=fake_checkpoint(arm),
            )
            assert replay["status"] == "idempotent"
            if index == 0:
                parent_trace.extend(["k3_registered", "best_training_complete"])
            elif index == 1:
                parent_trace.extend(["best_registered", "worst_training_complete"])
            else:
                parent_trace.extend(["worst_registered", "slot0_restored", "FAIR013_CHAIN_DONE"])

        assert parent_trace == [
            "slot3_caption_complete", "k3_training_complete", "k3_registered",
            "best_training_complete", "best_registered", "worst_training_complete",
            "worst_registered", "slot0_restored", "FAIR013_CHAIN_DONE",
        ]
        assert len(delivered) == 3 and len(set(delivered)) == 3
        expect_hold(lambda: reg.arm_for_argv(template, ["fourth", "/tmp/fourth", "--no_q"]), "argv")
        wrong_path_argv = arm_argv(template["arms"][0])
        wrong_path_argv[1] = "/tmp/wrong-checkpoint.pth"
        expect_hold(lambda: reg.arm_for_argv(template, wrong_path_argv), "argv")
        expect_hold(lambda: reg.register_once(
            template=template, argv=arm_argv(template["arms"][2]), caller=caller,
            paths=paths, manifest_sha256=manifest_hash, approval_sha256=approval_hash,
            notify=lambda _: None,
            checkpoint_identity=fake_checkpoint(template["arms"][2], discriminator=1),
        ), "duplicate drift")

        event_path = paths.outbox / "03_fair013_worst_noq.json"
        event_bytes = event_path.read_bytes()
        event = json.loads(event_bytes)
        event["delivery_status"] = "attempting"
        event_path.write_text(json.dumps(event))
        os.chmod(event_path, 0o600)
        expect_hold(lambda: reg.register_once(
            template=template, argv=arm_argv(template["arms"][2]), caller=caller,
            paths=paths, manifest_sha256=manifest_hash, approval_sha256=approval_hash,
            notify=lambda _: None, checkpoint_identity=fake_checkpoint(template["arms"][2]),
        ), "HMAC")
        event_path.write_bytes(event_bytes)
        os.chmod(event_path, 0o600)

        state = json.loads(paths.state.read_text())
        state["status"] = "tampered"
        paths.state.write_text(json.dumps(state))
        os.chmod(paths.state, 0o600)
        key = reg.create_or_load_key(paths.key)
        expect_hold(lambda: reg.load_state(
            paths.root, key, reg.sha256_bytes(reg.canonical(template)), manifest_hash, approval_hash
        ), "HMAC")

    with tempfile.TemporaryDirectory(prefix="slot3-cfg0-modes-") as directory:
        bad = Path(directory) / "bad"
        bad.mkdir(mode=0o700)
        os.chmod(bad, 0o777)
        expect_hold(lambda: reg.prepare_runtime(bad, trust_anchor=Path(directory)), "owner/mode")
        real = Path(directory) / "real"
        real.mkdir(mode=0o700)
        link = Path(directory) / "link"
        link.symlink_to(real, target_is_directory=True)
        expect_hold(lambda: reg.prepare_runtime(link, trust_anchor=Path(directory)), "owner/mode")

        writable_parent = Path(directory) / "group-writable-parent"
        writable_parent.mkdir(mode=0o700)
        os.chmod(writable_parent, 0o770)
        nested_runtime = writable_parent / "runtime"
        expect_hold(
            lambda: reg.prepare_runtime(nested_runtime, trust_anchor=Path(directory)),
            "runtime ancestor",
        )
        assert not nested_runtime.exists() and not nested_runtime.is_symlink()

    with tempfile.TemporaryDirectory(prefix="slot3-cfg0-state-modes-") as directory:
        root = Path(directory) / "runtime"
        reg.prepare_runtime(root, trust_anchor=Path(directory))
        key = reg.create_or_load_key(root / "registration_hmac.key")
        template_hash = reg.sha256_bytes(reg.canonical(template))
        state_path = root / "state.json"
        reg.load_state(root, key, template_hash, manifest_hash, approval_hash)
        os.chmod(state_path, 0o666)
        expect_hold(lambda: reg.load_state(root, key, template_hash, manifest_hash, approval_hash), "owner/mode")
        os.chmod(state_path, 0o600)
        sentinel = Path(directory) / "sentinel.json"
        sentinel.write_text(state_path.read_text())
        state_path.unlink()
        state_path.symlink_to(sentinel)
        expect_hold(lambda: reg.load_state(root, key, template_hash, manifest_hash, approval_hash), "owner/mode")

    assert reg.sha256_path(reg.TOP_QUEUE) == queue_before
    assert process_set() == gpu_before
    run_notification_crash_fixtures(template)


def run_caller_fixtures(template: dict[str, Any]) -> None:
    expected = template["caller"]
    replacement = "a" * 64
    observed = fake_caller(template)
    reg.validate_caller(observed, expected, replacement)
    for field, value in (
        ("pid", expected["pid"] + 1),
        ("uid", expected["uid"] + 1),
        ("start_ticks", expected["start_ticks"] + 1),
        ("boot_id", "wrong-boot"),
        ("script_fd", 254),
        ("script_inode", expected["script_inode"] + 1),
        ("script_sha256", "c" * 64),
        ("argv", ["/bin/bash", "/tmp/wrong.sh"]),
    ):
        changed = copy.deepcopy(observed)
        changed[field] = value
        expect_hold(lambda changed=changed: reg.validate_caller(changed, expected, replacement), field)
    changed = copy.deepcopy(observed)
    changed["hook_script_sha256"] = "d" * 64
    expect_hold(lambda: reg.validate_caller(changed, expected, replacement), "hook script")


def run_installer_fixtures() -> None:
    with tempfile.TemporaryDirectory(prefix="slot3-cfg0-install-") as directory:
        target = Path(directory) / "eval_musiccaps_mf25.sh"
        old = b"#!/bin/sh\nexit 99\n"
        new = b"#!/bin/sh\nexit 0\n"
        target.write_bytes(old)
        target.chmod(0o755)
        expect_hold(lambda: installer._atomic_replace_live(new, "0" * 64, target=target), "changed")
        assert target.read_bytes() == old
        installer._atomic_replace_live(new, reg.sha256_bytes(old), target=target)
        assert target.read_bytes() == new and stat.S_IMODE(target.stat().st_mode) == 0o755
        sentinel = Path(directory) / "sentinel"
        sentinel.write_text("preserve")
        target.unlink()
        target.symlink_to(sentinel)
        expect_hold(lambda: installer._atomic_replace_live(old, reg.sha256_bytes(new), target=target))
        assert sentinel.read_text() == "preserve"


def run_outer_loop_fixture() -> None:
    attempts = [125, 125, 0]
    pauses: list[str] = []
    parent_remainder: list[str] = []

    def attempt() -> int:
        rc = attempts.pop(0)
        if rc == 0:
            parent_remainder.extend(["best", "worst", "restore_slot0"])
        return rc

    assert reg.outer_loop(attempt, pause=lambda: pauses.append("poll"), max_attempts=3) == 0
    assert pauses == ["poll", "poll"]
    assert parent_remainder == ["best", "worst", "restore_slot0"]
    assert reg.outer_loop(lambda: 125, pause=lambda: None, max_attempts=2) == 125


def run_security_receipt_fixtures(template: dict[str, Any]) -> None:
    plan_sha256 = reg.sha256_path(reg.PLAN)
    fixture = {
        "schema_version": 1,
        "document_kind": "pilotfish_security_readiness_review",
        "readiness_unit_id": "slot3-cfg0-priority-interception-registration-v1",
        "reviewed_plan_sha256": plan_sha256,
        "verdict": "READY",
    }
    reg.validate_security_receipt(fixture, plan_sha256)
    substitutions = (
        ("document_kind", "pilotfish_outcome_verification"),
        ("readiness_unit_id", "different-readiness-unit"),
        ("reviewed_plan_sha256", hashlib.sha256(b"different plan").hexdigest()),
        ("verdict", "REVISE"),
    )
    for field, value in substitutions:
        changed = {**fixture, field: value}
        expect_hold(
            lambda changed=changed: reg.validate_security_receipt(changed, plan_sha256),
            field,
        )
    actual = reg.load_json(reg.SECURITY_REVIEW)
    expect_hold(
        lambda: reg.validate_security_receipt(actual, plan_sha256),
        "reviewed_plan_sha256",
    )
    expect_hold(
        lambda: installer.validate_descriptor(require_approved=False),
        "reviewed_plan_sha256",
    )
    expect_hold(
        registration_validator.validate_pending_readiness,
        "reviewed_plan_sha256",
    )
    expect_hold(
        lambda: reg.verify_production_bindings(template),
        "reviewed_plan_sha256",
    )


def main() -> None:
    template = reg.load_json(reg.TEMPLATE)
    reg.validate_template(template)
    assert [arm["arm_id"] for arm in template["arms"]] == [
        "fair013_k3_q9", "fair013_best_noq", "fair013_worst_noq"
    ]
    assert len({arm["canonical_label"] for arm in template["arms"]}) == 3
    for arm in template["arms"]:
        assert "_cfg0_" in arm["canonical_label"]
        assert template["scientific_protocol"]["cfg_strength"] == 0
        assert template["scientific_protocol"]["num_steps"] == 25
    hook = reg.HOOK.read_text(encoding="utf-8")
    assert "while :" in hook and 'coordinator_status" -eq 0' in hook
    assert "eval.py" not in hook and "phase4_eval.py" not in hook and "cfg_strength 4.5" not in hook
    rollback = installer.rollback_stub_bytes().decode("utf-8")
    assert "CFG0_INTERCEPTOR_ROLLED_BACK" in rollback and "exit 125" in rollback
    assert "cfg4p5" not in rollback and "eval.py" not in rollback
    manifest = reg.runtime_manifest_payload()
    order = reg.load_json(reg.ORDER)
    registration_validator.validate_manifest(manifest, template)
    registration_validator.validate_priority_order(order, template)
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["sources"][0]["sha256"] = "0" * 64
    expect_hold(lambda: registration_validator.validate_manifest(changed_manifest, template), "manifest")
    changed_order = copy.deepcopy(order)
    changed_order["durable_evaluation_order"][1], changed_order["durable_evaluation_order"][2] = (
        changed_order["durable_evaluation_order"][2], changed_order["durable_evaluation_order"][1]
    )
    expect_hold(lambda: registration_validator.validate_priority_order(changed_order, template), "order")
    run_security_receipt_fixtures(template)
    run_caller_fixtures(template)
    run_registration_fixtures(template)
    run_installer_fixtures()
    run_outer_loop_fixture()
    print("PASS slot3 fair013 CFG0 registration/interception no-GPU fixtures")


if __name__ == "__main__":
    main()
