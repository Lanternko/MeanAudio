#!/usr/bin/env python3
"""Build an instrumental-only MusicCaps eval set for ATTM-protocol benchmarking.

ATTM restricts the task to 10s instrumental music. MusicCaps is mixed, so we
carve out the subset whose captions carry no vocal/speech language, using the
same VOCAL_RE the negprompt re-evals already use, and intersect it with the
reference wavs we actually hold (needed for FAD).

Writes:
  musiccaps_instrumental_test.tsv        prompts for generation + CLAP/CCS
  musiccaps_instrumental_ref/            symlinked reference wavs for FAD
"""
import csv, re, sys, os
from pathlib import Path

# identical to scripts/eval/negprompt_reeval_full_arms.py:51 so the split is
# comparable with every novocal number already on record
VOCAL_RE = re.compile(
    r"\b(vocal|vocals|vocalist|vocalists|vocalisation|vocalization|singer|singers|"
    r"singing|sings|sung|sing|voice|voices|choir|chorus|chant|chanting|rap|rapper|"
    r"rapping|lyric|lyrics|acappella|harmonies|humming|hums|falsetto|soprano|tenor|"
    r"baritone|spoken|speaks|speech|narrate|narration|narrator)\b", re.I)

TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
REF = Path('/mnt/HDD/kojiek/musiccaps_reference')
OUT_DIR = Path('/home/kojiek/eval_tsvs_p100')
OUT_TSV = OUT_DIR / 'musiccaps_instrumental_test.tsv'
OUT_REF = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm/musiccaps_instrumental_ref')

rows = list(csv.DictReader(TSV.open(), delimiter='\t'))
instrumental = [r for r in rows if not VOCAL_RE.search(r['caption'])]
have_ref = {p.stem for p in REF.glob('*.wav')}
with_ref = [r for r in instrumental if r['id'] in have_ref]

print(f'MusicCaps test          : {len(rows)}')
print(f'  instrumental (caption): {len(instrumental)}  ({len(instrumental)/len(rows):.1%})')
print(f'  vocal                 : {len(rows)-len(instrumental)}')
print(f'reference wavs on disk  : {len(have_ref)}')
print(f'  instrumental & has ref: {len(with_ref)}   <- FAD reference set size')

OUT_DIR.mkdir(parents=True, exist_ok=True)
with OUT_TSV.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'caption'], delimiter='\t')
    w.writeheader()
    w.writerows({'id': r['id'], 'caption': r['caption']} for r in instrumental)
print(f'wrote {OUT_TSV} ({len(instrumental)} rows)')

OUT_REF.mkdir(parents=True, exist_ok=True)
for old in OUT_REF.glob('*.wav'):
    old.unlink()
for r in with_ref:
    (OUT_REF / f"{r['id']}.wav").symlink_to(REF / f"{r['id']}.wav")
print(f'linked {len(with_ref)} reference wavs -> {OUT_REF}')
