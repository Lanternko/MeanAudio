"""
Gen Qwen-style eval TSVs for Jamendo seed42 2048 test set.

After running gen_qwen_test_captions.py (5 task slots × 2048 clips):
  - reads merged qwen_test_seed42_2048_captions.jsonl
  - encodes 5 caps via CLAP, computes pairwise sim
  - outputs:
      qwen_test_seed42_2048_random.tsv  (random seed=42 per clip)
      qwen_test_seed42_2048_bc.tsv      (best-consensus per clip)

Only 2048 clips × 5 caps = 10,240 captions to encode → ~30 sec on 5090.

Usage:
  source ~/venvs/dac/bin/activate
  python gen_qwen_test_eval_tsvs.py
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

JSONL     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_captions.jsonl')
INPUT_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv')
OUT_RANDOM = Path('/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_random.tsv')
OUT_BC     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_bc.tsv')
OUT_SEL    = Path.home() / 'research/meanaudio_training/qwen_test_seed42_2048_selections.json'

CLAP_CKPT = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
TEXT_BATCH = 256
RANDOM_SEED = 42


def load_clap():
    import laion_clap
    print('[CLAP] loading text encoder...')
    m = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    m.load_ckpt(str(CLAP_CKPT), verbose=False)
    m.eval()
    return m.cuda()


def encode(model, texts):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), TEXT_BATCH):
            batch = texts[i:i + TEXT_BATCH]
            e = model.get_text_embedding(batch, use_tensor=True)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    print(f'Loading Qwen test captions: {JSONL}')
    if not JSONL.exists():
        raise FileNotFoundError(f'{JSONL} not found — run gen_qwen_test_captions.py first')
    qwen_lookup = {}
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            qwen_lookup[d['id']] = d['captions']
    print(f'  {len(qwen_lookup):,} clips')

    print(f'\nReading test TSV: {INPUT_TSV}')
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    print(f'  {len(rows):,} test rows')

    # Build flat texts in TSV order
    flat_texts, flat_clip = [], []
    for cid in ids:
        caps = qwen_lookup.get(cid)
        if caps is None or len(caps) != 5:
            continue
        for c in caps:
            flat_texts.append(c)
            flat_clip.append(cid)
    print(f'  {len(flat_texts):,} captions ({len(flat_texts)//5:,} clips with 5 caps)')

    model = load_clap()
    embs = encode(model, flat_texts)
    embs = embs.reshape(-1, 5, embs.shape[-1])  # (N, 5, D)

    # Pairwise sim per clip + bc_idx
    sim = np.einsum('nid,njd->nij', embs, embs)  # (N, 5, 5)
    K = 5
    row_sum = sim.sum(axis=2)
    diag = np.diagonal(sim, axis1=1, axis2=2)
    row_mean_offdiag = (row_sum - diag) / (K - 1)
    mean_sim_per_clip = row_mean_offdiag.mean(axis=1)
    bc_idx_per_clip = row_mean_offdiag.argmax(axis=1).astype(np.int8)

    # Map back to clip ids
    sel = {}
    seen = set()
    cur = 0
    rng = random.Random(RANDOM_SEED)
    for cid in flat_clip:
        if cid not in seen:
            sel[cid] = {
                'bestconsensus_idx': int(bc_idx_per_clip[cur]),
                'mean_sim': float(mean_sim_per_clip[cur]),
            }
            seen.add(cid)
            cur += 1

    # Random selection (seed=42), only over clips with 5 caps
    for cid in ids:
        if cid in sel:
            sel[cid]['random_seed42_idx'] = rng.randint(0, 4)

    OUT_SEL.write_text(json.dumps(sel, indent=1))
    print(f'\nselections → {OUT_SEL}')

    # Sanity stats
    from collections import Counter
    rand_dist = Counter(sel[c].get('random_seed42_idx') for c in ids if c in sel)
    bc_dist   = Counter(sel[c].get('bestconsensus_idx') for c in ids if c in sel)
    msim = [sel[c]['mean_sim'] for c in ids if c in sel]
    print(f'  random slot dist: {dict(sorted(rand_dist.items()))}')
    print(f'  bc slot dist:     {dict(sorted(bc_dist.items()))}')
    print(f'  mean_sim: min={min(msim):.4f} median={np.median(msim):.4f} max={max(msim):.4f}')

    # Write TSVs
    def write_tsv(path, idx_key, label):
        n = 0
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'caption'], delimiter='\t')
            w.writeheader()
            for r in rows:
                cid = r['id']
                if cid in qwen_lookup and cid in sel:
                    cap = qwen_lookup[cid][sel[cid][idx_key]]
                    cap = cap.replace('\n', ' ').replace('\r', ' ').strip()
                    w.writerow({'id': cid, 'caption': cap})
                    n += 1
                else:
                    # fallback: keep LP-MC test caption
                    w.writerow({'id': cid, 'caption': r['caption']})
        print(f'  → {path}  ({label}, matched={n})')

    print(f'\nWriting eval TSVs:')
    write_tsv(OUT_RANDOM, 'random_seed42_idx', 'random seed=42')
    write_tsv(OUT_BC,     'bestconsensus_idx', 'bestconsensus')
    print(f'\n✅ done.')


if __name__ == '__main__':
    main()
