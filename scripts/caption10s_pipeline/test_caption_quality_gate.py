#!/usr/bin/env python3
"""Adversarial fixtures for the structural caption classifier."""
from __future__ import annotations

import unittest

from repair_multisent_first_entity_line import classify


class CaptionQualityGateTest(unittest.TestCase):
    def assert_tag(self, text: str, tag: str) -> None:
        self.assertIn(tag, classify(text), text)

    def test_rejects_known_structural_defects(self) -> None:
        self.assert_tag('{"caption": "A valid sentence."}', "json_wrapper")
        self.assert_tag("The caption is as follows:", "degenerate_leadin")
        self.assert_tag(
            "The caption for the music excerpt provided would be:",
            "degenerate_leadin",
        )
        self.assert_tag("The caption should include the following details:", "degenerate_leadin")
        self.assert_tag("A valid sentence followed by ∞∞∞∞∞∞", "character_run")
        self.assert_tag("word " * 105 + "truncated", "no_terminal_punctuation")
        self.assert_tag(
            "The caption for this music clip is: " * 4,
            "repeated_leadin",
        )
        self.assert_tag(
            "The caption for this audio clip is: 'A complete sentence.']",
            "bracket_wrapper",
        )
        self.assert_tag(
            "A valid sentence. (Note: generated from supplied information.)",
            "meta_disclaimer",
        )

    def test_preserves_non_defect_style_variation(self) -> None:
        valid = [
            "A cheerful acoustic guitar tune with a joyful melody.",
            "The caption is: A complete, content-bearing musical description.",
            "A piano melody centers on C♯4 with a calm, reflective mood.",
            "The tempo is approximately ~120 BPM with a clear mix.",
            "A solo piano piece in a classical style",
        ]
        for text in valid:
            self.assertEqual([], classify(text), text)


if __name__ == "__main__":
    unittest.main()
