"""
Qwen single-cap rerun prep: produce caption-index selections per clip for
three rerun configs, plus train TSVs.

For each of 251,599 Jamendo clips with 5 Qwen task-framed captions:
  - random_seed42_idx: deterministic random pick (seed=42)
  - bestconsensus_idx: argmax of row mean of 5×5 pairwise CLAP cos sim
                        (cap most similar to the other 4 on average)
  - mean_sim:          off-diagonal mean of pairwise sim (P9.5 V2 q signal)
  - q_level:           Qwen-local percentile-equal-frequency bin of mean_sim

Outputs:
  selections.json                       — id → cap_idx + mean_sim + q_level
  qwen_singlecap_random_train.tsv       — id, caption(slot=random_seed42), q=dummy 5
  qwen_singlecap_random_q_train.tsv     — id, caption(slot=random_seed42), q=qwen_local
  qwen_singlecap_bc_train.tsv           — id, caption(slot=bestconsensus), q=dummy 5

Reuses cached mean_sim from gen_phase9_5_q_levels.py if available.

Usage:
  source ~/venvs/dac/bin/activate
  export CUDA_VISIBLE_DEVICES=0
  python gen_qwen_singlecap_selections.py
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

JSONL     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
INPUT_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
DATA_DIR  = Path('/mnt/HDD/kojiek/phase4_jamendo_data')
RES_DIR   = Path.home() / 'research/meanaudio_training'

OUT_SEL   = RES_DIR / 'qwen_singlecap_selections.json'
OUT_BINS  = RES_DIR / 'qwen_singlecap_bin_edges.json'
SIM_CACHE = RES_DIR / 'phase9_5_mean_sim.npz'   # reuse if exists

OUT_TSV_RANDOM    = DATA_DIR / 'qwen_singlecap_random_train.tsv'      # P8-Qwen
OUT_TSV_RANDOM_Q  = DATA_DIR / 'qwen_singlecap_random_q_train.tsv'    # P7V1-Qwen
OUT_TSV_BC        = DATA_DIR / 'qwen_singlecap_bc_train.tsv'          # P4V2-Qwen

CLAP_CKPT  = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
TEXT_BATCH = 256
N_BINS     = 10
RANDOM_SEED = 42


def load_qwen_jsonl():
    lookup = {}
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            lookup[d['id']] = d['captions']
    return lookup


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


def compute_pairwise_sim_matrices(model, ids, lookup):
    """For each clip: encode 5 caps and return both:
       - mean_sim (off-diagonal mean of 5×5 sim)
       - bestconsensus_idx (argmax of row mean, equivalent to argmax of off-diag row mean)
    """
    flat_texts = []
    flat_clip = []
    for cid in ids:
        caps = lookup.get(cid)
        if caps is None or len(caps) != 5:
            continue
        for c in caps:
            flat_texts.append(c)
            flat_clip.append(cid)

    print(f'  encoding {len(flat_texts):,} captions ({len(flat_texts)//5:,} clips)...')
    embs = encode(model, flat_texts)        # (5N, D)
    embs = embs.reshape(-1, 5, embs.shape[-1])  # (N, 5, D)

    print('  computing pairwise sim + bestconsensus...')
    sim = np.einsum('nid,njd->nij', embs, embs)  # (N, 5, 5)
    K = 5
    # mean_sim: off-diagonal mean = (row_sum - diag) / (K-1)
    row_sum  = sim.sum(axis=2)               # (N, 5)
    diag     = np.diagonal(sim, axis1=1, axis2=2)  # (N, 5), all ~1.0 since normalized
    row_mean_offdiag = (row_sum - diag) / (K - 1)  # (N, 5)
    mean_sim_per_clip = row_mean_offdiag.mean(axis=1)  # (N,)
    bc_idx_per_clip = row_mean_offdiag.argmax(axis=1).astype(np.int8)  # (N,)

    sim_out = {}
    bc_out  = {}
    seen = set()
    cur = 0
    for cid in flat_clip:
        if cid not in seen:
            sim_out[cid] = float(mean_sim_per_clip[cur])
            bc_out[cid]  = int(bc_idx_per_clip[cur])
            seen.add(cid)
            cur += 1
    return sim_out, bc_out


def build_bins(values):
    arr = np.asarray(sorted(values))
    edges = np.percentile(arr, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    n_bins = len(edges) - 1

    print(f'\n── Qwen mean_sim distribution (n={len(arr):,}) ──')
    print(f'  min={arr.min():.4f} p25={np.percentile(arr, 25):.4f} '
          f'median={np.median(arr):.4f} p75={np.percentile(arr, 75):.4f} '
          f'max={arr.max():.4f}')
    print(f'  mean={arr.mean():.4f} std={arr.std():.4f}')
    print(f'\n── Percentile bin edges ({n_bins} bins) ──')
    counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (arr >= lo) & (arr < hi if i < n_bins - 1 else arr <= hi)
        counts.append(int(mask.sum()))
        print(f'  q={i}: [{lo:.4f}, {hi:.4f}{")" if i < n_bins - 1 else "]"}  n={mask.sum():,}')

    def quantize(v):
        idx = np.searchsorted(edges[1:-1], v, side='right')
        return int(min(idx, n_bins - 1))

    return edges.tolist(), counts, quantize, n_bins


def main():
    print(f'Loading Qwen JSONL: {JSONL}')
    qwen_lookup = load_qwen_jsonl()
    print(f'  {len(qwen_lookup):,} clips')

    print(f'\nReading TSV: {INPUT_TSV}')
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    print(f'  {len(rows):,} rows')

    # Step 1: mean_sim (try cache, otherwise compute) + bestconsensus
    sim_map = None
    if SIM_CACHE.exists():
        print(f'\n[cache] loading mean_sim from {SIM_CACHE}')
        c = np.load(SIM_CACHE, allow_pickle=True)
        sim_map = {k: float(v) for k, v in zip(c['ids'], c['mean_sim'])}
        if all(i in sim_map for i in ids):
            print(f'  full cache hit ({len(sim_map):,} ids); but bestconsensus needs re-compute (not cached)')
        else:
            print(f'  partial cache; recomputing all')
            sim_map = None

    if sim_map is None:
        model = load_clap()
        sim_map, bc_map = compute_pairwise_sim_matrices(model, ids, qwen_lookup)
        # save cache for next time
        np.savez(SIM_CACHE,
                 ids=np.array(list(sim_map.keys())),
                 mean_sim=np.array(list(sim_map.values()), dtype=np.float32))
        print(f'\n[cache] saved mean_sim → {SIM_CACHE}')
    else:
        # cache only had mean_sim, we need bc_idx → recompute (must encode again unfortunately)
        print('  recomputing bc_idx (cache only has mean_sim)...')
        model = load_clap()
        _, bc_map = compute_pairwise_sim_matrices(model, ids, qwen_lookup)

    # Step 2: bins
    values = [sim_map[i] for i in ids if i in sim_map]
    edges, counts, quantize, n_bins = build_bins(values)
    median_q = n_bins // 2
    OUT_BINS.write_text(json.dumps({
        'n_bins': n_bins,
        'edges': edges,
        'counts_per_bin': counts,
        'fallback_q': median_q,
        'source': 'phase9_omni_captions.jsonl (Qwen2.5-Omni 5 task caps)',
        'random_seed': RANDOM_SEED,
    }, indent=2))
    print(f'\n[bins] saved → {OUT_BINS}')

    # Step 3: random selection (seed=42, identical algorithm to gen_phase7_v1_tsv.py)
    rng = random.Random(RANDOM_SEED)
    random_idx_map = {}
    for cid in ids:
        if cid in qwen_lookup and len(qwen_lookup[cid]) == 5:
            random_idx_map[cid] = rng.randint(0, 4)

    # Step 4: write selections JSON (canonical lookup for slicer + TSV gen)
    print(f'\nWriting selections → {OUT_SEL}')
    sel = {}
    for cid in ids:
        if cid not in qwen_lookup:
            continue
        sel[cid] = {
            'random_seed42_idx': random_idx_map.get(cid),
            'bestconsensus_idx': bc_map.get(cid),
            'mean_sim': sim_map.get(cid),
            'q_level': quantize(sim_map[cid]) if cid in sim_map else median_q,
        }
    OUT_SEL.write_text(json.dumps(sel, indent=1))

    # Step 5: write 3 train TSVs
    def write_tsv(path, choose_idx_fn, q_fn, label):
        print(f'  → {path}  ({label})')
        path.parent.mkdir(parents=True, exist_ok=True)
        n_match = n_fallback = 0
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
            w.writeheader()
            for r in rows:
                cid = r['id']
                if cid in qwen_lookup and cid in sel:
                    cap = qwen_lookup[cid][choose_idx_fn(cid)]
                    cap = cap.replace('\n', ' ').replace('\r', ' ').strip()
                    w.writerow({'id': cid, 'caption': cap, 'q_level': q_fn(cid)})
                    n_match += 1
                else:
                    w.writerow({'id': cid, 'caption': r['caption'], 'q_level': str(median_q)})
                    n_fallback += 1
        print(f'    matched={n_match:,} fallback={n_fallback:,}')

    print(f'\nWriting TSVs:')
    write_tsv(OUT_TSV_RANDOM,
              choose_idx_fn=lambda c: sel[c]['random_seed42_idx'],
              q_fn=lambda c: str(median_q),
              label='P8-Qwen (random seed=42, dummy q)')

    write_tsv(OUT_TSV_RANDOM_Q,
              choose_idx_fn=lambda c: sel[c]['random_seed42_idx'],
              q_fn=lambda c: str(sel[c]['q_level']),
              label='P7V1-Qwen (random seed=42, Qwen-local q)')

    write_tsv(OUT_TSV_BC,
              choose_idx_fn=lambda c: sel[c]['bestconsensus_idx'],
              q_fn=lambda c: str(median_q),
              label='P4V2-Qwen (bestconsensus, dummy q)')

    # Step 6: sanity stats
    print(f'\n── Selection sanity ──')
    rand_idx_dist = {i: sum(1 for s in sel.values() if s['random_seed42_idx'] == i) for i in range(5)}
    bc_idx_dist   = {i: sum(1 for s in sel.values() if s['bestconsensus_idx'] == i) for i in range(5)}
    print(f'  random_seed42 slot distribution: {rand_idx_dist}  (target ~equal)')
    print(f'  bestconsensus slot distribution:  {bc_idx_dist}  (skew toward consistent slots is expected)')
    overlap = sum(1 for s in sel.values() if s['random_seed42_idx'] == s['bestconsensus_idx'])
    print(f'  random == bestconsensus: {overlap:,}/{len(sel):,} ({100*overlap/len(sel):.1f}%, expect ~20% if uncorrelated)')

    print(f'\n✅ done. {len(sel):,} clips have selections.')


if __name__ == '__main__':
    main()
