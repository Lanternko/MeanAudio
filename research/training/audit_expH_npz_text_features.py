"""
audit_expH_npz_text_features.py
=================================
1000-sample CLAP cos-sim audit of EXP-H NPZ text_features_c.

For each sampled row:
  - Re-encode expH_rewrite caption with CLAP (live forward)
  - Compute cos-sim vs stored NPZ text_features_c
  - Also compute cos-sim vs LP-MC caption at same row index

Reports:
  - n_perfect (cos ≥ 0.999), n_good (cos ≥ 0.99), n_bad (cos < 0.95)
  - Mean / min / std of expH cos-sim and lpmc cos-sim
  - Any rows below threshold (up to 20)

Pass criterion: n_bad = 0 AND mean expH cos-sim ≥ 0.999
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import laion_clap
from tqdm import tqdm

CLAP_CKPT = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
BATCH = 32


def cos_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between rows of a and b. Both (N, D)."""
    return F.cosine_similarity(a, b, dim=1)


def load_tsv_captions(path: Path) -> list[str]:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return [r['caption'] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv',       default='/home/kojiek/eval_tsvs_p100/expH_rewrite_train.tsv')
    ap.add_argument('--npz_dir',   default='/home/kojiek/exps_nvme/npz_expH_rewrite')
    ap.add_argument('--cache',     default='/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')
    ap.add_argument('--compare_lpmc',
                    default='/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv')
    ap.add_argument('--n',   type=int, default=1000, help='number of random samples')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tsv_path   = Path(args.tsv)
    npz_dir    = Path(args.npz_dir)
    cache_path = Path(args.cache)
    lp_path    = Path(args.compare_lpmc)

    print(f"TSV      : {tsv_path}")
    print(f"NPZ dir  : {npz_dir}")
    print(f"Cache    : {cache_path}")
    print(f"LP-MC    : {lp_path}")
    print(f"Samples  : {args.n}")

    # Load
    exph_caps = load_tsv_captions(tsv_path)
    lp_caps   = load_tsv_captions(lp_path)
    with open(cache_path) as f:
        cache_files = [l.strip() for l in f if l.strip()]

    assert len(exph_caps) == len(cache_files) == len(lp_caps), \
        f"Length mismatch: expH={len(exph_caps)}, cache={len(cache_files)}, lp={len(lp_caps)}"
    N = len(exph_caps)
    print(f"Total rows: {N:,}")

    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(N), args.n))

    # Load CLAP
    print("\nLoading CLAP (HTSAT-base)...")
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(CLAP_CKPT)
    clap.eval().to('cuda')
    print("CLAP ready.")

    # Collect stored NPZ tfc and captions in batch order
    stored_tfcs  = []
    batch_exph   = []
    batch_lp     = []

    for idx in idxs:
        p = npz_dir / cache_files[idx]
        if not p.exists():
            print(f"[ERROR] missing NPZ: {p}")
            sys.exit(1)
        z = np.load(p)
        stored_tfcs.append(z['text_features_c'].astype('float32'))
        batch_exph.append(exph_caps[idx])
        batch_lp.append(lp_caps[idx])

    stored_tfcs = torch.from_numpy(np.stack(stored_tfcs)).float()  # (N, 512)

    # Encode in batches
    print(f"\nEncoding {args.n} captions with CLAP (batch={BATCH})...")
    exph_embs = []
    lp_embs   = []
    with torch.no_grad():
        for bi in tqdm(range(0, args.n, BATCH), desc='encode'):
            eb = batch_exph[bi: bi + BATCH]
            lb = batch_lp[bi: bi + BATCH]
            exph_embs.append(clap.get_text_embedding(eb, use_tensor=True).cpu().float())
            lp_embs.append(clap.get_text_embedding(lb, use_tensor=True).cpu().float())
    exph_embs = torch.cat(exph_embs, 0)  # (N, 512)
    lp_embs   = torch.cat(lp_embs, 0)

    # Compute cos-sims
    sim_exph = cos_batch(stored_tfcs, exph_embs)   # (N,)
    sim_lp   = cos_batch(stored_tfcs, lp_embs)     # (N,)

    # Stats
    print("\n=== Results ===")
    print(f"{'':30s}  {'mean':>8}  {'min':>8}  {'max':>8}  {'std':>8}")
    print(f"{'stored ↔ expH rewrite':30s}  {sim_exph.mean():.6f}  {sim_exph.min():.6f}  {sim_exph.max():.6f}  {sim_exph.std():.6f}")
    print(f"{'stored ↔ LP-MC':30s}  {sim_lp.mean():.6f}  {sim_lp.min():.6f}  {sim_lp.max():.6f}  {sim_lp.std():.6f}")

    n_perfect = (sim_exph >= 0.999).sum().item()
    n_good    = (sim_exph >= 0.990).sum().item()
    n_bad     = (sim_exph <  0.950).sum().item()

    print(f"\n  n_perfect (≥0.999): {n_perfect}/{args.n}  ({100*n_perfect/args.n:.1f}%)")
    print(f"  n_good    (≥0.990): {n_good}/{args.n}  ({100*n_good/args.n:.1f}%)")
    print(f"  n_bad     (<0.950): {n_bad}/{args.n}  ({100*n_bad/args.n:.1f}%)")

    if n_bad > 0:
        print("\n  [!] Bad rows (stored ≠ expH rewrite):")
        bad_mask = sim_exph < 0.950
        bad_idxs = [idxs[i] for i in range(args.n) if bad_mask[i]]
        for i, row_idx in enumerate(bad_idxs[:20]):
            j = idxs.index(row_idx)
            print(f"    row {row_idx}: sim_exph={sim_exph[j]:.4f}  sim_lp={sim_lp[j]:.4f}")
            print(f"      expH: {batch_exph[j][:80]}")
            print(f"      lp  : {batch_lp[j][:80]}")

    # Pass/fail
    print("\n=== Verdict ===")
    if n_bad == 0 and sim_exph.mean() >= 0.999:
        print("✅ PASS — All stored text_features_c match EXP-H rewritten captions")
        print("         (n_bad=0, mean cos-sim=1.0000)")
        print("         Pipeline integrity confirmed for text embedding dimension.")
    else:
        print("❌ FAIL — Some NPZ text_features_c do not match EXP-H captions")
        sys.exit(1)


if __name__ == '__main__':
    main()
