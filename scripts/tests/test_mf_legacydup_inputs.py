import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('legacydup', Path(__file__).parents[1] / 'preprocess/build_mf_legacydup_inputs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BackfillTests(unittest.TestCase):
    def test_exact_clip_mapping_all_members_and_cap(self):
        base = [{'id': f't_segment_{n}_0', 'caption': 'Base.'} for n in (0, 1, 2, 3)]
        old = [{'id': f't_segment_{n}', 'caption': 'Duplicate.' if n < 2 else 'Unique.'} for n in (0, 1, 2)]
        rows, changes, eligible = m.backfill(base, old, 100000)
        self.assertEqual([r['caption'] for r in rows], ['Duplicate.', 'Duplicate.', 'Base.', 'Base.'])
        self.assertEqual((len(changes), eligible), (2, 2))
        self.assertEqual(m.backfill(base, old, 1)[0][1]['caption'], 'Base.')
        self.assertEqual(rows, m.backfill(base, old, 100000)[0])

    def test_duplicate_ids_fail(self):
        row = {'id': 't_segment_0', 'caption': 'Music.'}
        with self.assertRaises(ValueError):
            m.backfill([], [row, row], 100000)

    def test_structural_defects_and_short_caption(self):
        self.assertEqual(m.classify('Music.'), [])
        cases = [(None, 'null'), ('未完成。', 'cjk'), ('User: hello.', 'turn_marker'),
                 ('{"caption":"Music."}', 'json_wrapper'), ('**Music.**', 'markdown_wrapper'),
                 ('www.example.org.', 'url'), (r'\\begin{a}.', 'latex'),
                 ('Music\ncontinues.', 'multiline'), ('Music!!!!', 'character_run'),
                 ('The music stops', 'missing_terminal_punctuation'), ('import os.', 'code')]
        for text, tag in cases:
            self.assertIn(tag, m.classify(text), (text, tag))


if __name__ == '__main__':
    unittest.main()
