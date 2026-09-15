import importlib.util
from pathlib import Path
import unittest
import json
import os
import tempfile

path = Path(__file__).resolve().parents[1] / "preprocess/slot0_semantic_audit.py"
spec = importlib.util.spec_from_file_location("audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PrefixTests(unittest.TestCase):
    def test_exact_unicode_preservation(self):
        original = "A singer’s voice rises.  Please upload audio."
        self.assertEqual(audit.apply_decision(original, {
            "decision": "TRIM_SUFFIX", "suffix": "  Please upload audio."}),
            "A singer’s voice rises.")

    def test_mid_sentence_or_rewritten_suffix_rejected(self):
        for suffix in ["voice rises.", " Please send audio.", ""]:
            with self.assertRaises(ValueError):
                audit.apply_decision("A voice rises.", {
                    "decision": "TRIM_SUFFIX", "suffix": suffix})

    def test_unresolved_has_no_output(self):
        for decision in ["REVIEW", "INVALID"]:
            self.assertIsNone(audit.apply_decision("Please send audio.", {
                "decision": decision, "suffix": ""}))

    def test_keep_never_normalizes(self):
        original = "A piano  plays at 120 BPM."
        self.assertEqual(audit.apply_decision(original, {
            "decision": "KEEP", "suffix": ""}), original)

    def test_inconsistent_decision_rejected(self):
        with self.assertRaises(ValueError):
            audit.apply_decision("A piano plays.", {
                "decision": "KEEP", "suffix": "plays."})

    def test_short_valid_caption_is_not_filtered(self):
        self.assertEqual(audit.structural_defects("A piano plays."), [])

    def test_structural_wrapper_cannot_be_accepted(self):
        self.assertIn("markdown_wrapper", audit.structural_defects("```A piano plays.```"))

    def test_role_and_multiline_defects(self):
        defects = audit.structural_defects("A piano plays.\nAssistant: Please send audio.")
        self.assertIn("multiline", defects)
        self.assertIn("turn_marker", defects)

    def test_truncated_refused_and_wrong_coverage_responses(self):
        rows = [{"id": "a", "caption": "A piano plays."}]
        response = {"id": "test", "model": audit.MODEL, "choices": [{"finish_reason": "length",
                    "message": {"content": "{}"}}]}
        with self.assertRaises(ValueError):
            audit.parse_response(rows, response)
        response["choices"][0] = {"finish_reason": "stop", "message": {"refusal": "refused", "content": "{}"}}
        with self.assertRaises(ValueError):
            audit.parse_response(rows, response)
        response["choices"][0]["message"] = {"content": json.dumps({"decisions": []})}
        with self.assertRaises(ValueError):
            audit.parse_response(rows, response)

    def test_credential_permissions_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_text("test-only\n")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                audit.read_key_file(path)
            path.chmod(0o600)
            self.assertEqual(audit.read_key_file(path), "test-only")
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                audit.read_key_file(link)


if __name__ == "__main__":
    unittest.main()
