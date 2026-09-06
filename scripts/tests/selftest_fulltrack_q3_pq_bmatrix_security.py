#!/usr/bin/env python3
"""CPU-only abuse-case tests for the FTQ3-BMATRIX-v1 security boundary."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


HARN_PATH = Path("/home/kojiek/MeanAudio/scripts/experiment_harness/fulltrack_q3_pq_bmatrix_harn.py")
SPEC = importlib.util.spec_from_file_location("ftq3_security_harn_test", HARN_PATH)
assert SPEC is not None and SPEC.loader is not None
harn = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harn
SPEC.loader.exec_module(harn)


class SecurityMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        harn.ensure_private_dir(harn.RESULT_ROOT)
        harn.ensure_private_dir(harn.TEST_FIXTURES)
        cls.root = harn.TEST_FIXTURES / ("security-" + os.urandom(8).hex())
        harn.ensure_private_dir(cls.root, must_be_new=True)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove only exact test-owned leaf objects; never recurse through links.
        for path in sorted(cls.root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            except FileNotFoundError:
                pass
        cls.root.rmdir()

    def leaf(self, name: str, raw: bytes = b"fixture") -> Path:
        path = self.root / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        return path

    def gate2(self, **changes):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        value = {
            "document_kind": "gate_2_exact_capability", "lifecycle": "approved",
            "consumed": False, "expires_at": future,
            "auth_root_decision": "operator_accepted_same_uid_residual",
            "bindings": {"plan_sha256": harn.PLAN_SHA256, "contract_sha256": "a" * 64,
                         "launcher_sha256": "b" * 64},
        }
        for key, item in changes.items():
            if key.startswith("binding_"):
                value["bindings"][key[8:]] = item
            else:
                value[key] = item
        return value

    def validate_gate2(self, approval, lifecycle="approved"):
        contract = {"launch_allowed": True, "approval_required": True}
        bindings = {"plan_sha256": harn.PLAN_SHA256, "contract_sha256": "a" * 64,
                    "launcher_sha256": "b" * 64}
        harn.validate_gate2_payload(approval, contract, bindings, require_lifecycle=lifecycle)

    def test_AT_INPUT_01_hash_mismatch_fails(self):
        path = self.leaf("input-hash.bin")
        self.assertNotEqual(harn.sha256_path_nofollow(path), "0" * 64)

    def test_AT_INPUT_02_symlink_input_fails(self):
        target = self.leaf("target.bin")
        link = self.root / "input-link.bin"
        link.symlink_to(target)
        with self.assertRaises((OSError, harn.SecurityHold)):
            harn.sha256_path_nofollow(link)

    def test_AT_INPUT_03_hardlink_input_fails(self):
        target = self.leaf("hard-target.bin")
        link = self.root / "hard-link.bin"
        os.link(target, link)
        with self.assertRaises(harn.SecurityHold):
            harn.sha256_path_nofollow(target, require_one_link=True)

    def test_AT_INPUT_04_source_drift_detected(self):
        path = self.leaf("drift.bin", b"before")
        digest = harn.sha256_path_nofollow(path)
        with path.open("wb") as handle:
            handle.write(b"after")
        self.assertNotEqual(harn.sha256_path_nofollow(path), digest)

    def test_AT_AUTH_01_missing_gate2_fails(self):
        with self.assertRaises(harn.SecurityHold):
            self.validate_gate2({})

    def test_AT_AUTH_02_expired_gate2_fails(self):
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaises(harn.SecurityHold):
            self.validate_gate2(self.gate2(expires_at=expired))

    def test_AT_AUTH_03_wrong_binding_fails(self):
        with self.assertRaises(harn.SecurityHold):
            self.validate_gate2(self.gate2(binding_contract_sha256="c" * 64))

    def test_AT_AUTH_04_replay_consumed_fails(self):
        with self.assertRaises(harn.SecurityHold):
            self.validate_gate2(self.gate2(lifecycle="consumed", consumed=True))

    def test_AT_AUTH_05_gate1_never_launches(self):
        candidate = {"launch_allowed": False, "approval_required": True}
        with self.assertRaises(harn.SecurityHold):
            harn.validate_gate2_payload(self.gate2(), candidate, {}, require_lifecycle="approved")

    def test_AT_AUTH_06_auth_root_choice_required(self):
        with self.assertRaises(harn.SecurityHold):
            self.validate_gate2(self.gate2(auth_root_decision=None))

    def test_AT_P2_01_dependency_terminal_delivery(self):
        for status in ("completed", "failed", "interrupted"):
            self.assertTrue(harn.queue_dependency_eligible(status, "delivered"))
        for status in ("held", "resource_wait", "notification_pending", "delivery_ambiguous"):
            self.assertFalse(harn.queue_dependency_eligible(status, "delivered"))

    def test_AT_P2_02_append_order_exact(self):
        snapshot = {"directories": {"running": ["024_fair013_k3_full.sh"],
                    "pending": ["025_true_random_full.sh", "026_fake_random_full.sh"],
                    "held": [], "failed": [], "done": []}}
        snapshot["sha256"] = harn.sha256_bytes(harn.canonical(snapshot["directories"]))
        approval = {"bindings": {"queue_snapshot_sha256": snapshot["sha256"]}}
        harn._assert_queue_append_preconditions(approval, snapshot)

    def test_AT_P2_03_collision_fails(self):
        snapshot = {"directories": {"running": [], "pending": ["027_fulltrack_q3_pq_bmatrix.sh"],
                    "held": [], "failed": [], "done": []}}
        snapshot["sha256"] = harn.sha256_bytes(harn.canonical(snapshot["directories"]))
        with self.assertRaises(harn.SecurityHold):
            harn._assert_queue_append_preconditions({"bindings": {"queue_snapshot_sha256": snapshot["sha256"]}}, snapshot)

    def test_AT_P2_04_foreign_gpu_zero_threshold(self):
        with mock.patch.object(harn, "gpu_compute_processes", return_value=[{"pid": 123, "used_memory_mib": 1}]):
            with self.assertRaises(harn.SecurityHold):
                harn.assert_no_foreign_gpu_processes({456})

    def test_AT_P2_05_lease_ambiguity_fails(self):
        with mock.patch.object(harn, "load_json_nofollow", side_effect=harn.SecurityHold("missing")):
            with self.assertRaises(harn.SecurityHold):
                harn.assert_p2_lease_identity()

    def test_AT_PATH_01_id_injection_fails(self):
        for value in ("../x", "/abs", ".", "..", "a/b", "x\nq"):
            with self.assertRaises(harn.SecurityHold):
                harn.safe_id(value)

    def test_AT_PATH_02_directory_symlink_fails(self):
        directory = self.root / "real-dir"
        directory.mkdir(mode=0o700)
        linked = self.root / "linked-dir"
        linked.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(OSError):
            harn._open_dir_chain(linked)

    def test_AT_PATH_03_relative_escape_fails(self):
        fd = harn._open_dir_chain(self.root)
        try:
            with self.assertRaises(harn.SecurityHold):
                harn._destination_parent(fd, harn.PurePosixPath("../escape"))
        finally:
            os.close(fd)

    def test_AT_CLEAN_01_exact_intent(self):
        expected = harn.validate_cleanup_names(["a", "b"], {"intent": ["a.flac", "b.flac"], "completed": []})
        self.assertEqual(expected, ["a.flac", "b.flac"])

    def test_AT_CLEAN_02_extra_cleanup_name_fails(self):
        with self.assertRaises(harn.SecurityHold):
            harn.validate_cleanup_names(["a"], {"intent": ["a.flac", "x.flac"], "completed": []})

    def test_AT_STATE_01_hmac_tamper(self):
        key = b"k" * 32
        record = {"sequence": 1, "value": "ok"}
        record["hmac_sha256"] = harn.sign_record(key, b"test", record)
        harn.verify_record(key, b"test", record)
        record["value"] = "tampered"
        with self.assertRaises(harn.SecurityHold):
            harn.verify_record(key, b"test", record)

    def test_AT_STATE_02_wrong_mode_rejected(self):
        path = self.leaf("mode.json", b"{}")
        os.chmod(path, 0o644)
        with self.assertRaises(harn.SecurityHold):
            harn.load_json_nofollow(path, require_uid=os.geteuid(), require_mode=0o600)

    def test_AT_STATE_03_atomic_noreplace(self):
        path = self.root / "atomic.json"
        harn.atomic_write_json(path, {"v": 1}, replace=False)
        with self.assertRaises(harn.SecurityHold):
            harn.atomic_write_json(path, {"v": 2}, replace=False)

    def test_runtime_state_outbox_lock_and_restart_idempotency(self):
        state = self.root / "runtime-state"
        state.mkdir(mode=0o700)
        values = {
            "STATE_ROOT": state,
            "STATE_KEY": state / "state_hmac.key",
            "STATE_LEDGER": state / "ledger.json",
            "STATE_OUTBOX": state / "outbox",
            "STATE_LOCK_POINTER": state / "controller_lock.json",
        }
        with mock.patch.multiple(harn, **values):
            harn.initialize_runtime_security_state()
            event1 = harn.append_runtime_event("gate", "B1", {"verdict": "pass"})
            event2 = harn.append_runtime_event("gate", "B1", {"verdict": "pass"})
            self.assertEqual(event1, event2)
            payload = harn.redacted_notification("gate", "B1", {"verdict": "pass"})
            outbox = harn.persist_outbox_event(payload)
            self.assertEqual(stat.S_IMODE(outbox.stat().st_mode), 0o600)
            first = harn.acquire_runtime_lock()
            try:
                with self.assertRaises(harn.SecurityHold):
                    harn.acquire_runtime_lock()
            finally:
                os.close(first)

    def test_AT_NOTIFY_TRANSACTION_01_ambiguity_holds(self):
        self.assertEqual(harn.classify_notification_transition("attempting", "timeout_after_send"),
                         "delivery_ambiguous")
        with self.assertRaises(harn.SecurityHold):
            harn.classify_notification_transition("delivery_ambiguous", "known_pre_request_failure")

    def test_AT_RESTART_01_recovery_before_promotion(self):
        incident = {"event_key": "incident", "notification_status": "delivered"}
        self.assertFalse(harn.recovery_promotable(incident, None))
        self.assertTrue(harn.recovery_promotable(
            incident, {"relates_to": "incident", "notification_status": "delivered"}))

    def test_AT_ENV_01_rejected_parent_name(self):
        with self.assertRaises(harn.SecurityHold):
            harn.sanitized_child_environment({"PYTHONPATH": "/tmp"})

    def test_AT_ENV_02_from_scratch_allowlist(self):
        self.assertEqual(harn.sanitized_child_environment({}), harn.ALLOWED_CHILD_ENV)

    def test_AT_ARGV_01_inline_execution_fails(self):
        with self.assertRaises(harn.SecurityHold):
            harn.validate_no_shell_argv([str(harn.PYTHON), "-c", "print(1)"])

    def test_AT_NET_01_offline_flags_mandatory(self):
        env = harn.sanitized_child_environment({})
        self.assertEqual({env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}, {"1"})

    def test_AT_SECRET_01_notification_redacts_unknown_fields(self):
        payload = harn.redacted_notification("hold", "B1", {"reason_code": "x", "raw_log": "secret-data"})
        self.assertNotIn("raw_log", payload["fields"])

    def test_AT_SECRET_02_child_environment_has_no_secret_names(self):
        forbidden = {"TOKEN", "SECRET", "WEBHOOK_URL", "API_KEY", "AUTHORIZATION"}
        self.assertTrue(forbidden.isdisjoint(harn.ALLOWED_CHILD_ENV))

    def test_AT_MENTION_01_mentions_disabled(self):
        payload = harn.redacted_notification("gate", "B1", {"verdict": "pass"})
        self.assertEqual(payload["allowed_mentions"], {"parse": []})

    def test_storage_warning_and_hard_stop_boundaries(self):
        self.assertEqual(harn.storage_status(harn.HARD_FLOOR - 1)["verdict"], "hard_stop")
        self.assertEqual(harn.storage_status(harn.HARD_FLOOR)["verdict"], "warning")
        self.assertEqual(harn.storage_status(harn.WARNING_FLOOR)["verdict"], "pass")

    def test_budget_progress_stall_and_regression(self):
        self.assertEqual(harn.budget_status(0, 10, 10), "exhausted")
        self.assertEqual(harn.progress_status(1, 2, 999, 10), "progressed")
        self.assertEqual(harn.progress_status(2, 1, 0, 10), "regressed")
        self.assertEqual(harn.progress_status(2, 2, 10, 10), "stalled")

    def test_arm_freshness_quarantines_existing(self):
        arm_root = harn.RESULT_ROOT / "B1"
        with mock.patch.object(harn, "RESULT_ROOT", self.root):
            arm_root = self.root / "B1"
            arm_root.mkdir(mode=0o700)
            runner_path = Path("/home/kojiek/MeanAudio/scripts/eval/run_fulltrack_q3_pq_arm.py")
            runner_spec = importlib.util.spec_from_file_location("ftq3_runner_security_test", runner_path)
            assert runner_spec and runner_spec.loader
            runner = importlib.util.module_from_spec(runner_spec)
            sys.modules[runner_spec.name] = runner
            runner_spec.loader.exec_module(runner)
            with mock.patch.object(runner.harn, "RESULT_ROOT", self.root):
                with self.assertRaises(runner.harn.SecurityHold):
                    runner._new_arm_tree("B1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
