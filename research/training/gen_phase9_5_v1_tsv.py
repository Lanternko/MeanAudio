"""
P9.5 V1 prep: CPU-only TSV generation (no GPU needed for V1 NoQ run).

Why CPU-only:
  - V1 uses use_q_conditioning=false → q_level column is read but value ignored
  - multi_cap=True training reads captions from NPZ, not from TSV → caption col
    is also effectively ignored
  - Only the id column matters for ordering and NPZ lookup
  - Saves GPU contention with shared users (hsiehyian) when only V1 is being staged

Output:
  phase9_5_train.tsv  with columns: id \t caption \t q_level
    - id: copied verbatim from phase7_v1_train.tsv (251,599 rows, same order)
    - caption: slot 0 (Writing) from Qwen merged JSONL — useful for human inspection
      and TSV-level grep, but NOT loaded at training time when multi_cap=True
    - q_level: dummy 5 (median bin) — this column will be REPLACED later by
      gen_phase9_5_q_levels.py when V2 is staged (post V1 result evaluation)

Usage:
  python gen_phase9_5_v1_tsv.py  # ~5 seconds, no GPU
"""

import csv
import json
from pathlib import Path

INPUT_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
JSONL     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
OUT_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_5_train.tsv')

DUMMY_Q = '5'  # median; ignored under use_q_conditioning=false


def main():
    print(f'[1/3] reading TSV ids: {INPUT_TSV}')
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    print(f'      {len(ids):,} rows')

    print(f'[2/3] reading Qwen slot 0 captions: {JSONL}')
    slot0_cap = {}
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            slot0_cap[d['id']] = d['captions'][0]
    print(f'      {len(slot0_cap):,} clips with slot 0')

    print(f'[3/3] writing TSV → {OUT_TSV}')
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    n_match = 0
    n_fallback = 0
    with open(OUT_TSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        w.writeheader()
        for r in rows:
            cid = r['id']
            cap = slot0_cap.get(cid)
            if cap is None:
                cap = r['caption']  # fallback to LP-MC caption from input TSV
                n_fallback += 1
            else:
                n_match += 1
            # normalize whitespace per project convention
            cap = cap.replace('\n', ' ').replace('\r', ' ').strip()
            w.writerow({'id': cid, 'caption': cap, 'q_level': DUMMY_Q})

    print(f'\n  matched (Qwen slot 0): {n_match:,}')
    print(f'  fallback (LP-MC):      {n_fallback:,}')
    print(f'\n✅ done. Path: {OUT_TSV}')
    print('   Next: NPZ generation (gen_multicap_npz.py) once GPU is free.')
    print('   Note: gen_phase9_5_q_levels.py will OVERWRITE this TSV with')
    print('         real q_levels when V2 is staged.')


if __name__ == '__main__':
    main()
