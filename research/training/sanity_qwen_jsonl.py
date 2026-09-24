"""
P9.5 prep: Qwen merged caption JSONL sanity check.

Checks per Codex 2026-05-03 review:
  1. Row count matches phase7_v1_train.tsv (251,599)
  2. Every row has exactly 5 captions
  3. No empty / None / whitespace-only captions
  4. ID order matches TSV order (1-to-1, same sequence)
  5. Slot diversity check on N_DIVERSITY clips: per-clip uniqueness rate
  6. Slot 0 prompt-style spot-check (prints 3 examples per slot)

No GPU, no NPZ touched. Pure CSV+JSONL parsing.

Usage:
  python sanity_qwen_jsonl.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
N_DIVERSITY = 200  # how many clips to sample for diversity stats

EXPECTED_SLOTS = ['Writing', 'Summary', 'Paraphrase', 'Attribute', 'NaturalProse']


def load_tsv_ids():
    with open(TSV) as f:
        return [row['id'] for row in csv.DictReader(f, delimiter='\t')]


def load_jsonl():
    rows = []
    with open(JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    print(f'Loading TSV ids: {TSV}')
    tsv_ids = load_tsv_ids()
    print(f'  {len(tsv_ids):,} rows')

    print(f'\nLoading JSONL: {JSONL}')
    rows = load_jsonl()
    print(f'  {len(rows):,} rows')

    fail = 0

    # ── Check 1: row count ───────────────────────────────────────────
    print('\n[1] Row count parity')
    if len(rows) != len(tsv_ids):
        print(f'  ❌ JSONL {len(rows):,} != TSV {len(tsv_ids):,}')
        fail += 1
    else:
        print(f'  ✅ {len(rows):,} == {len(tsv_ids):,}')

    # ── Check 2: 5 captions per row ──────────────────────────────────
    print('\n[2] Cap count == 5 per row')
    cap_counts = Counter(len(r.get('captions', [])) for r in rows)
    print(f'  cap count distribution: {dict(cap_counts)}')
    if cap_counts.get(5, 0) != len(rows):
        print(f'  ❌ {len(rows) - cap_counts.get(5, 0):,} rows have != 5 captions')
        fail += 1
    else:
        print(f'  ✅ all {len(rows):,} rows have 5 captions')

    # ── Check 3: no empty captions ───────────────────────────────────
    print('\n[3] No empty / null / whitespace-only captions')
    empty_clips = []
    null_total = 0
    empty_total = 0
    for r in rows:
        caps = r.get('captions', [])
        for slot, c in enumerate(caps):
            if c is None:
                null_total += 1
                empty_clips.append((r.get('id', '?'), slot, 'NULL'))
            elif not isinstance(c, str) or not c.strip():
                empty_total += 1
                empty_clips.append((r.get('id', '?'), slot, repr(c)[:40]))
    if null_total or empty_total:
        print(f'  ❌ null={null_total} empty={empty_total}')
        for cid, slot, val in empty_clips[:10]:
            print(f'      id={cid} slot={slot} val={val}')
        if len(empty_clips) > 10:
            print(f'      ... and {len(empty_clips) - 10} more')
        fail += 1
    else:
        print(f'  ✅ all {sum(len(r["captions"]) for r in rows):,} captions non-empty')

    # ── Check 4: ID order matches TSV order ──────────────────────────
    print('\n[4] ID order matches phase7_v1_train.tsv order')
    if len(rows) != len(tsv_ids):
        print('  ⚠️  skipped (length mismatch from check 1)')
    else:
        mismatches = []
        for i, (jid, tid) in enumerate(zip([r['id'] for r in rows], tsv_ids)):
            if jid != tid:
                mismatches.append((i, jid, tid))
                if len(mismatches) >= 5:
                    break
        if mismatches:
            print(f'  ❌ {len(mismatches)}+ mismatches (showing first 5):')
            for i, jid, tid in mismatches:
                print(f'      idx {i}: jsonl={jid} tsv={tid}')
            fail += 1
        else:
            print(f'  ✅ all {len(rows):,} ids match in order')

    # ── Check 5: per-clip slot diversity ─────────────────────────────
    print(f'\n[5] Per-clip slot diversity (sample N={N_DIVERSITY})')
    sample = rows[:N_DIVERSITY]
    unique_rates = []
    full_unique = 0
    one_collapse = 0
    for r in sample:
        caps = r.get('captions', [])
        valid = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
        if not valid:
            continue
        u = len(set(valid))
        rate = u / len(valid)
        unique_rates.append(rate)
        if u == len(valid):
            full_unique += 1
        if u == 1:
            one_collapse += 1
    if unique_rates:
        avg = sum(unique_rates) / len(unique_rates)
        print(f'  avg uniqueness rate: {avg:.3f}  (target > 0.9)')
        print(f'  fully unique (5/5):  {full_unique}/{len(unique_rates)}')
        print(f'  collapsed (1/5):     {one_collapse}/{len(unique_rates)}')
        if avg < 0.9:
            print('  ⚠️  uniqueness below 0.9 — captioner may be collapsing')
        if one_collapse > 0:
            print('  ⚠️  some clips have all 5 caps identical')
    else:
        print('  ⚠️  no valid samples')

    # ── Check 6: per-slot length distribution + sample print ─────────
    print('\n[6] Per-slot caption length stats + 3 random samples')
    for slot in range(5):
        lens = [len(r['captions'][slot].split())
                for r in sample
                if isinstance(r['captions'][slot], str) and r['captions'][slot].strip()]
        if lens:
            print(f'\n  slot {slot} ({EXPECTED_SLOTS[slot] if slot < 5 else "?"}): '
                  f'n={len(lens)} mean_words={sum(lens)/len(lens):.1f} '
                  f'min={min(lens)} max={max(lens)}')
            for r in sample[:3]:
                cap = r['captions'][slot]
                if isinstance(cap, str):
                    print(f'    [{r["id"][:30]}] {cap[:120]}')

    # ── Summary ───────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    if fail == 0:
        print('✅ All hard checks passed. Ready for NPZ generation.')
    else:
        print(f'❌ {fail} hard check(s) failed. Fix before NPZ generation.')
        sys.exit(1)


if __name__ == '__main__':
    main()
