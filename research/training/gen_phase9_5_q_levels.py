"""
P9.5 V2 q_level generation: Qwen 5-cap pairwise CLAP text-text mean_sim
                            → Qwen-local percentile-equal-frequency bin (0..9)
                            → phase9_5_train.tsv with q_level column

Per Codex 2026-05-03 review:
  - Qwen-local bins (do NOT share LP-MC bin edges; distributions will shift)
  - Document raw mean_sim distribution + bin edges in output
  - q=N is captioner-local percentile bucket, NOT absolute cross-captioner level

Outputs:
  - {OUT_TSV}: id \t caption \t q_level   (caption = slot 0 'Writing'; runtime
               multi_cap=True will randomly draw from 5 NPZ caps anyway)
  - {OUT_CACHE}: ids[] mean_sim[] (npz cache for reproducibility / re-bin later)
  - {OUT_BINS}: bin_edges[] + per-bin count (json, for paper figures)

Usage:
  source ~/venvs/dac/bin/activate
  export CUDA_VISIBLE_DEVICES=0
  python gen_phase9_5_q_levels.py [--limit N] [--no_cache]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

JSONL      = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
INPUT_TSV  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUT_TSV    = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_5_v2_train.tsv')  # V2-specific (V1 dummy at phase9_5_train.tsv)
OUT_CACHE  = Path.home() / 'research/meanaudio_training/phase9_5_mean_sim.npz'
OUT_BINS   = Path.home() / 'research/meanaudio_training/phase9_5_bin_edges.json'
CLAP_CKPT  = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'

TEXT_BATCH = 256
N_BINS     = 10  # q_level 0..9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--no_cache', action='store_true')
    return p.parse_args()


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


def compute_mean_sim(model, ids, lookup):
    """For each clip: pairwise cos sim of 5 caps, return mean of off-diagonal."""
    flat_texts = []
    flat_clip  = []
    for cid in ids:
        caps = lookup.get(cid)
        if caps is None or len(caps) != 5:
            continue
        for c in caps:
            flat_texts.append(c)
            flat_clip.append(cid)

    print(f'  encoding {len(flat_texts):,} captions ({len(flat_texts)//5:,} clips)...')
    embs = encode(model, flat_texts)  # (5N, D)
    embs = embs.reshape(-1, 5, embs.shape[-1])  # (N, 5, D)

    print('  computing pairwise mean_sim...')
    # sim[i,j] = embs[:, i] @ embs[:, j]
    sim = np.einsum('nid,njd->nij', embs, embs)  # (N, 5, 5)
    K = 5
    off_diag_sum = sim.sum(axis=(1, 2)) - np.trace(sim, axis1=1, axis2=2)
    n_off_diag = K * (K - 1)
    mean_sim = off_diag_sum / n_off_diag

    out = {}
    seen = set()
    cur_clip = None
    cur_idx = 0
    for cid in flat_clip:
        if cid not in seen:
            out[cid] = float(mean_sim[cur_idx])
            seen.add(cid)
            cur_idx += 1
    return out


def build_bins(values):
    """Percentile equal-frequency 0..N_BINS-1 bins on Qwen-local values."""
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
    if n_bins < N_BINS:
        print(f'  ⚠️  only {n_bins} bins (mean_sim has duplicate percentiles)')

    def quantize(v):
        idx = np.searchsorted(edges[1:-1], v, side='right')
        return int(min(idx, n_bins - 1))

    return edges.tolist(), counts, quantize, n_bins


def main():
    args = parse_args()

    print(f'Loading Qwen JSONL: {JSONL}')
    qwen_lookup = load_qwen_jsonl()
    print(f'  {len(qwen_lookup):,} clips')

    print(f'\nReading TSV: {INPUT_TSV}')
    rows_in = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows_in = list(reader)
    if args.limit:
        rows_in = rows_in[:args.limit]
    ids = [r['id'] for r in rows_in]
    print(f'  {len(rows_in):,} rows')

    # ── compute or load mean_sim cache ───────────────────────────────
    if OUT_CACHE.exists() and not args.no_cache:
        print(f'\n[cache] loading {OUT_CACHE}')
        c = np.load(OUT_CACHE, allow_pickle=True)
        sim_map = {k: float(v) for k, v in zip(c['ids'], c['mean_sim'])}
        missing = [i for i in ids if i not in sim_map]
        if missing:
            print(f'  {len(missing):,} ids uncached; computing...')
            model = load_clap()
            extra = compute_mean_sim(model, missing, qwen_lookup)
            sim_map.update(extra)
            np.savez(OUT_CACHE,
                     ids=np.array(list(sim_map.keys())),
                     mean_sim=np.array(list(sim_map.values()), dtype=np.float32))
    else:
        model = load_clap()
        sim_map = compute_mean_sim(model, ids, qwen_lookup)
        np.savez(OUT_CACHE,
                 ids=np.array(list(sim_map.keys())),
                 mean_sim=np.array(list(sim_map.values()), dtype=np.float32))
        print(f'\n[cache] saved → {OUT_CACHE}')

    # ── build bins on collected mean_sim values ──────────────────────
    values = [sim_map[i] for i in ids if i in sim_map]
    if len(values) < len(ids):
        print(f'⚠️  {len(ids) - len(values):,} ids missing mean_sim; will get q=median fallback')

    edges, counts, quantize, n_bins = build_bins(values)
    median_q = n_bins // 2

    OUT_BINS.write_text(json.dumps({
        'n_bins': n_bins,
        'edges': edges,
        'counts_per_bin': counts,
        'fallback_q': median_q,
        'source': 'phase9_omni_captions.jsonl (Qwen2.5-Omni 5 task caps)',
    }, indent=2))
    print(f'\n[bins] saved → {OUT_BINS}')

    # ── write TSV ────────────────────────────────────────────────────
    print(f'\nWriting TSV → {OUT_TSV}')
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        w.writeheader()
        n_match = 0
        n_fallback = 0
        for row in rows_in:
            cid = row['id']
            caps = qwen_lookup.get(cid)
            if caps is None or cid not in sim_map:
                # fallback: keep original caption + median q
                w.writerow({'id': cid, 'caption': row['caption'], 'q_level': str(median_q)})
                n_fallback += 1
            else:
                w.writerow({
                    'id': cid,
                    'caption': caps[0],          # slot 0 (Writing); runtime overrides via NPZ
                    'q_level': str(quantize(sim_map[cid])),
                })
                n_match += 1
    print(f'  matched={n_match:,}  fallback={n_fallback:,}')
    print(f'\n✅ done. Next: gen_multicap_npz.py --jsonl {JSONL} --tsv {OUT_TSV}')


if __name__ == '__main__':
    main()
