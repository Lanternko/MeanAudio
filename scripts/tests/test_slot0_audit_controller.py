import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))
import slot0_semantic_audit as A
import slot0_audit_controller as C


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.tsv"
        self.source.write_text("id\tcaption\tpath\na\tA piano plays.\ta.mp3\nb\tA piano plays. Please send audio.\tb.mp3\nc\tPlease provide audio.\tc.mp3\n")
        gate = self.root / "gate.json"
        gate.write_text(json.dumps({"status": "passed", "binding": {"prompt_sha256": A.digest(A.PROMPT.encode())}}))
        self.spec = {"run_id": "test", "experiment_id": "test", "output_dir": str(self.root / "out"),
                     "model": A.MODEL, "prompt_sha256": A.digest(A.PROMPT.encode()),
                     "immutable_artifacts": [{"path": str(self.source), "sha256": C.sha(self.source)}],
                     "approval": {"scope": "full_slot0_semantic_audit_exact_prefix_and_quarantine"},
                     "quality_gate_reports": [str(gate)], "source": {"path": str(self.source), "rows": 3, "sha256": C.sha(self.source)},
                     "batch_size": 1, "workers": 2, "max_429_attempts": 2, "max_wall_seconds": 60,
                     "min_request_interval_seconds": 0, "max_api_posts": 20, "max_cost_usd": 1,
                     "storage": {"hard_floor_bytes": 0, "warning_floor_bytes": 0},
                     "pricing": {"input_per_million": .2, "output_per_million": 1.2}}
        self.spec_path = self.root / "contract.json"
        self.write_spec()
        self.events = []
        self.calls = []
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            controller.close()
        self.tmp.cleanup()

    def write_spec(self):
        self.spec_path.write_text(json.dumps(self.spec))

    def fake(self, key, rows, receipt, metadata):
        self.calls.append([r["id"] for r in rows])
        ds = []
        for row in rows:
            action = "TRIM_SUFFIX" if " Please" in row["caption"] else "INVALID" if row["caption"].startswith("Please") else "KEEP"
            ds.append({"id": row["id"], "decision": action,
                       "suffix": " Please send audio." if action == "TRIM_SUFFIX" else "", "reason": "fixture"})
        raw = {"id": "mock", "model": A.MODEL, "metadata": metadata,
               "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"decisions": ds})}}],
               "usage": {"prompt_tokens": 20, "completion_tokens": 20}}
        A.atomic(receipt, raw)
        return A.parse_response(rows, raw)

    def make(self, transport=None, notifier=None):
        controller = C.Controller(self.spec_path, "test-not-a-key", transport or self.fake,
                                  notifier or (lambda *args: self.events.append(args)))
        self.controllers.append(controller)
        return controller

    @patch.object(C, "reconcile", return_value=None)
    def test_all_rows_suffix_verification_and_invalid_handoff(self, _):
        c = self.make()
        self.assertEqual(c.run(), 0)
        report = json.loads((c.out / "full_gate_report.json").read_text())
        self.assertEqual(report["audited_rows"], 3)
        self.assertEqual(report["unresolved_rows"], 1)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual((c.out / "regeneration_requests.tsv").read_text(), "c\tsemantic_invalid\n")
        self.assertFalse(report["training_corpus_released"])
        with self.source.open() as source:
            original = list(csv.DictReader(source, delimiter="\t"))
        self.assertEqual(original[1]["caption"], "A piano plays. Please send audio.")

    @patch.object(C, "reconcile", return_value=None)
    def test_timeout_isolated_other_batches_continue(self, _):
        def transport(key, rows, receipt, metadata):
            if rows[0]["id"] == "a":
                raise TimeoutError()
            return self.fake(key, rows, receipt, metadata)
        c = self.make(transport)
        self.assertEqual(c.run(), 2)
        self.assertEqual(c.db.execute("SELECT status FROM jobs WHERE id='audit-000000'").fetchone()[0], "ambiguous")
        self.assertEqual(c.db.execute("SELECT status FROM jobs WHERE id='audit-000002'").fetchone()[0], "done")

    @patch.object(C, "reconcile", return_value=None)
    def test_resume_does_not_repeat_completed_posts_or_notifications(self, _):
        c = self.make()
        c.run()
        calls, events = len(self.calls), len(self.events)
        c.run()
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(len(self.events), events)

    def test_inflight_receipt_recovers_without_post(self):
        c = self.make()
        c.prepare()
        job = c.db.execute("SELECT id,payload,request_hash FROM jobs WHERE id='audit-000000'").fetchone()
        self.fake("", json.loads(job[1]), c.out / "receipts" / (job[0] + ".json"), {"audit_request": job[2]})
        c.db.execute("UPDATE jobs SET status='inflight' WHERE id=?", (job[0],))
        c.db.commit()
        c.prepare()
        c.reconcile_ambiguous()
        self.assertEqual(c.db.execute("SELECT status FROM jobs WHERE id=?", (job[0],)).fetchone()[0], "done")
        self.assertEqual(len(self.calls), 1)

    def test_known_429_bounded_retry_and_attempt_persistence(self):
        c = self.make()
        c.prepare()
        c.db.execute("UPDATE jobs SET attempts=1,reserved_cost=.01 WHERE id='audit-000000'")
        c.failed("audit-000000", A.ApiFailure(429))
        self.assertEqual(c.db.execute("SELECT status,reserved_cost FROM jobs WHERE id='audit-000000'").fetchone(), ("retry_wait", 0.0))
        c.db.execute("UPDATE jobs SET attempts=2 WHERE id='audit-000000'")
        c.failed("audit-000000", A.ApiFailure(429))
        self.assertEqual(c.db.execute("SELECT status FROM jobs WHERE id='audit-000000'").fetchone()[0], "quarantined")

    def test_source_hash_change_fails_before_api(self):
        c = self.make()
        self.source.write_text("changed")
        with self.assertRaises(ValueError):
            c.prepare()
        self.assertEqual(self.calls, [])

    def test_notifier_failure_blocks_submission(self):
        def fail(*args):
            raise RuntimeError("notification failure")
        c = self.make(notifier=fail)
        with self.assertRaises(RuntimeError):
            c.run()
        self.assertEqual(self.calls, [])

    def test_duplicate_controller_lock(self):
        self.make()
        with self.assertRaises(BlockingIOError):
            self.make()

    def test_storage_branches(self):
        self.spec["storage"] = {"hard_floor_bytes": 10, "warning_floor_bytes": 20}
        self.write_spec()
        c = self.make()
        self.assertFalse(c.storage(free=9))
        self.assertTrue(c.storage(free=15))
        self.assertTrue(c.storage(free=25))
        self.assertEqual([e[0] for e in self.events], ["disk-hard-stop", "disk-warning"])

    @patch.object(C, "reconcile", return_value=None)
    def test_receipt_tamper_invalidates_export(self, _):
        c = self.make()
        c.run()
        (c.out / "receipts/audit-000000.json").write_text("{}")
        with self.assertRaises(ValueError):
            c.export()

    def test_cost_ceiling_blocks_before_call(self):
        self.spec["max_cost_usd"] = 0
        self.write_spec()
        c = self.make()
        with patch.object(C, "reconcile", return_value=None):
            self.assertEqual(c.run(), 2)
        self.assertEqual(self.calls, [])

    def test_error_types(self):
        self.assertEqual(C.classify_error(A.ApiFailure(503)), "ambiguous")
        self.assertEqual(C.classify_error(A.ApiFailure(401)), "credential_hold")
        self.assertEqual(C.classify_error(ValueError()), "quarantined")


if __name__ == "__main__":
    unittest.main()
