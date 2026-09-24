"""
compare_caption_embedding_distributions.py
===========================================
Compare T5 and CLAP text embedding distributions across three caption corpora:
  - LP-MC   (phase7_v1_train.tsv — healthy, gold standard)
  - Qwen    (qwen_slot0_train.tsv — raw Qwen slot-0, collapsed)
  - EXP-H   (expH_rewrite_train.tsv — LP-MC acoustic-style rewrite, also collapsed)

Metrics (all computed from 2000 random samples per corpus):
  T5 (77×1024 → use mean-pooled 1024-dim vector):
    - mean ‖embedding‖
    - mean pairwise off-diagonal cosine (intra-corpus clustering)
    - PCA: variance explained by top-5 components
    - inter-corpus centroid cosine (LP-MC ↔ Qwen, LP-MC ↔ EXP-H, Qwen ↔ EXP-H)
    - 1-NN overlap: what fraction of each corpus's 1-NN are from same corpus vs other

  CLAP (512-dim):
    - same metrics

Hypothesis being tested:
  EXP-H surface style resembles LP-MC (trigrams, acoustic vocab, length) but
  T5/CLAP embedding distribution may still resemble Qwen → explains collapse.

  If EXP-H embeddings are closer to LP-MC than Qwen in T5/CLAP space,
  then embedding distribution is NOT the explanation and we need a different frame.
  If EXP-H embeddings are still Qwen-like, that supports the deeper-distribution hypothesis.

Output: printed table + saved JSON at --out path.

Usage:
  python compare_caption_embedding_distributions.py [--n 2000] [--seed 42]
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import laion_clap
from transformers import AutoTokenizer, T5EncoderModel

CLAP_CKPT = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
T5_MODEL  = 'google/flan-t5-large'
BATCH     = 32

CORPORA = {
    'LP-MC':  '/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv',
    'Qwen':   '/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv',
    'EXP-H':  '/home/kojiek/eval_tsvs_p100/expH_rewrite_train.tsv',
}


# ── helpers ─────────────────────────────────────────────────────────────────

def load_captions(path: str, n: int, seed: int) -> list[str]:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    return [r['caption'] for r in sample]


def encode_t5(caps: list[str], tok, model, batch=BATCH) -> np.ndarray:
    """Returns mean-pooled T5 embeddings: (N, 1024)"""
    all_vecs = []
    with torch.no_grad():
        for bi in tqdm(range(0, len(caps), batch), desc='T5', leave=False):
            bc = caps[bi: bi + batch]
            enc = tok(bc, return_tensors='pt', padding='max_length',
                      truncation=True, max_length=77).to('cuda')
            out = model(**enc).last_hidden_state  # (B, 77, 1024)
            # mean-pool over sequence (ignore padding with attention_mask)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1)  # (B, 1024)
            all_vecs.append(pooled.cpu().float().numpy())
    return np.concatenate(all_vecs, 0)


def encode_clap(caps: list[str], model, batch=BATCH) -> np.ndarray:
    """Returns CLAP text embeddings: (N, 512)"""
    all_vecs = []
    with torch.no_grad():
        for bi in tqdm(range(0, len(caps), batch), desc='CLAP', leave=False):
            bc = caps[bi: bi + batch]
            emb = model.get_text_embedding(bc, use_tensor=True).cpu().float().numpy()
            all_vecs.append(emb)
    return np.concatenate(all_vecs, 0)


def embedding_stats(X: np.ndarray, name: str) -> dict:
    """Compute stats for (N, D) embedding matrix."""
    N = X.shape[0]
    Xt = torch.from_numpy(X).float()
    norms = Xt.norm(dim=1)
    Xn   = F.normalize(Xt, dim=1)

    # Off-diagonal pairwise cosine (sample 10K pairs to stay fast)
    rng = np.random.default_rng(0)
    n_pairs = min(10000, N * (N - 1) // 2)
    i_idx = rng.integers(0, N, n_pairs)
    j_idx = rng.integers(0, N, n_pairs)
    same  = i_idx == j_idx
    i_idx, j_idx = i_idx[~same], j_idx[~same]
    cos_pairs = (Xn[i_idx] * Xn[j_idx]).sum(1).numpy()

    # PCA: variance explained by top-5 components
    X_centered = X - X.mean(0)
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        var_ratio = (S**2) / (S**2).sum()
        pca_top5  = float(var_ratio[:5].sum())
    except Exception:
        pca_top5 = float('nan')

    return {
        'name':          name,
        'n':             N,
        'mean_norm':     float(norms.mean()),
        'std_norm':      float(norms.std()),
        'offdiag_cos_mean': float(cos_pairs.mean()),
        'offdiag_cos_std':  float(cos_pairs.std()),
        'pca_var_top5':  pca_top5,
        'centroid':      Xn.mean(0).numpy(),  # for inter-corpus distance
    }


def inter_corpus_cos(stats_a: dict, stats_b: dict) -> float:
    ca = torch.from_numpy(stats_a['centroid']).float()
    cb = torch.from_numpy(stats_b['centroid']).float()
    return float(F.cosine_similarity(ca.unsqueeze(0), cb.unsqueeze(0)))


def nearest_neighbor_overlap(X_a: np.ndarray, X_b: np.ndarray,
                              X_c: np.ndarray,
                              name_a='A', name_b='B', name_c='C',
                              n_query=200) -> dict:
    """
    For n_query random items from each corpus, find their nearest neighbor
    in the combined pool and count what fraction fall in each corpus.
    Returns fraction of 1-NN that land in same corpus vs others.
    """
    rng = np.random.default_rng(42)
    corpora = {name_a: X_a, name_b: X_b, name_c: X_c}
    labels = (
        [name_a] * len(X_a) +
        [name_b] * len(X_b) +
        [name_c] * len(X_c)
    )
    all_X = np.concatenate([X_a, X_b, X_c], 0)
    all_X_n = all_X / (np.linalg.norm(all_X, axis=1, keepdims=True) + 1e-9)
    offsets = {name_a: 0, name_b: len(X_a), name_c: len(X_a) + len(X_b)}
    sizes   = {name_a: len(X_a), name_b: len(X_b), name_c: len(X_c)}

    results = {}
    for qname, Xq in corpora.items():
        q_idxs = rng.integers(0, len(Xq), n_query)
        q_vecs  = Xq[q_idxs]
        q_vecs_n = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)

        # cosine to all (exclude self)
        cos_all = q_vecs_n @ all_X_n.T  # (n_query, total)
        # blank out self
        for qi, row_i in enumerate(q_idxs):
            cos_all[qi, offsets[qname] + row_i] = -2.0
        nn_idx = cos_all.argmax(1)  # (n_query,)
        nn_labels = [labels[i] for i in nn_idx]

        counts = {k: nn_labels.count(k) for k in corpora}
        total  = len(nn_labels)
        results[qname] = {k: counts[k] / total for k in corpora}
    return results


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',    type=int,  default=2000)
    ap.add_argument('--seed', type=int,  default=42)
    ap.add_argument('--out',  default='~/research/meanaudio_training/embedding_dist_results.json')
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    print(f"n={args.n}, seed={args.seed}")
    print(f"Output: {out_path}\n")

    # ── Load captions ──────────────────────────────────────────────────────
    print("Loading captions...")
    caps = {}
    for corpus_name, tsv_path in CORPORA.items():
        caps[corpus_name] = load_captions(tsv_path, args.n, args.seed)
        print(f"  {corpus_name}: {len(caps[corpus_name])} captions")
        # Show 2 examples
        print(f"    ex0: {caps[corpus_name][0][:80]}")
        print(f"    ex1: {caps[corpus_name][1][:80]}")
    print()

    # ── Load models ────────────────────────────────────────────────────────
    print("Loading T5 (flan-t5-large)...")
    tok = AutoTokenizer.from_pretrained(T5_MODEL)
    t5  = T5EncoderModel.from_pretrained(T5_MODEL).eval().to('cuda')

    print("Loading CLAP (HTSAT-base)...")
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(CLAP_CKPT)
    clap.eval().to('cuda')
    print()

    # ── Encode ────────────────────────────────────────────────────────────
    t5_embs   = {}
    clap_embs = {}
    for corpus_name in CORPORA:
        print(f"Encoding {corpus_name}...")
        t5_embs[corpus_name]   = encode_t5(caps[corpus_name], tok, t5)
        clap_embs[corpus_name] = encode_clap(caps[corpus_name], clap)
        print(f"  T5   : {t5_embs[corpus_name].shape}")
        print(f"  CLAP : {clap_embs[corpus_name].shape}")
    print()

    # ── Intra-corpus stats ────────────────────────────────────────────────
    print("Computing intra-corpus stats...")
    t5_stats   = {k: embedding_stats(t5_embs[k],   k) for k in CORPORA}
    clap_stats = {k: embedding_stats(clap_embs[k], k) for k in CORPORA}

    # ── Inter-corpus centroid cosines ─────────────────────────────────────
    pairs = [('LP-MC', 'Qwen'), ('LP-MC', 'EXP-H'), ('Qwen', 'EXP-H')]
    t5_inter   = {f"{a}↔{b}": inter_corpus_cos(t5_stats[a],   t5_stats[b])   for a,b in pairs}
    clap_inter = {f"{a}↔{b}": inter_corpus_cos(clap_stats[a], clap_stats[b]) for a,b in pairs}

    # ── 1-NN overlap ──────────────────────────────────────────────────────
    print("Computing 1-NN overlap...")
    t5_nn   = nearest_neighbor_overlap(t5_embs['LP-MC'],   t5_embs['Qwen'],   t5_embs['EXP-H'],
                                       'LP-MC', 'Qwen', 'EXP-H')
    clap_nn = nearest_neighbor_overlap(clap_embs['LP-MC'], clap_embs['Qwen'], clap_embs['EXP-H'],
                                       'LP-MC', 'Qwen', 'EXP-H')

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("T5 EMBEDDING DISTRIBUTION (mean-pooled, 1024-dim)")
    print("=" * 70)
    print(f"{'':20s}  {'mean‖x‖':>10}  {'offdiag_cos':>12}  {'pca_top5%':>10}")
    for k in CORPORA:
        s = t5_stats[k]
        print(f"  {k:18s}  {s['mean_norm']:>10.4f}  {s['offdiag_cos_mean']:>12.4f}  {100*s['pca_var_top5']:>9.1f}%")

    print("\nInter-corpus centroid cosines (T5):")
    for pair, v in t5_inter.items():
        print(f"  {pair:20s}  {v:.4f}")

    print("\n1-NN corpus breakdown (T5, fraction of each query's NN landing in each corpus):")
    print(f"  {'query→':10s}  {'→LP-MC':>10}  {'→Qwen':>10}  {'→EXP-H':>10}")
    for qname in CORPORA:
        row = t5_nn[qname]
        print(f"  {qname:10s}  {row['LP-MC']:>10.3f}  {row['Qwen']:>10.3f}  {row['EXP-H']:>10.3f}")

    print("\n" + "=" * 70)
    print("CLAP EMBEDDING DISTRIBUTION (512-dim)")
    print("=" * 70)
    print(f"{'':20s}  {'mean‖x‖':>10}  {'offdiag_cos':>12}  {'pca_top5%':>10}")
    for k in CORPORA:
        s = clap_stats[k]
        print(f"  {k:18s}  {s['mean_norm']:>10.4f}  {s['offdiag_cos_mean']:>12.4f}  {100*s['pca_var_top5']:>9.1f}%")

    print("\nInter-corpus centroid cosines (CLAP):")
    for pair, v in clap_inter.items():
        print(f"  {pair:20s}  {v:.4f}")

    print("\n1-NN corpus breakdown (CLAP):")
    print(f"  {'query→':10s}  {'→LP-MC':>10}  {'→Qwen':>10}  {'→EXP-H':>10}")
    for qname in CORPORA:
        row = clap_nn[qname]
        print(f"  {qname:10s}  {row['LP-MC']:>10.3f}  {row['Qwen']:>10.3f}  {row['EXP-H']:>10.3f}")

    # ── Interpretation helper ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    lq_clap = clap_inter['LP-MC↔Qwen']
    lh_clap = clap_inter['LP-MC↔EXP-H']
    qh_clap = clap_inter['Qwen↔EXP-H']
    lq_t5   = t5_inter['LP-MC↔Qwen']
    lh_t5   = t5_inter['LP-MC↔EXP-H']
    qh_t5   = t5_inter['Qwen↔EXP-H']

    print(f"\n  CLAP: LP-MC↔Qwen={lq_clap:.3f}  LP-MC↔EXP-H={lh_clap:.3f}  Qwen↔EXP-H={qh_clap:.3f}")
    print(f"  T5:   LP-MC↔Qwen={lq_t5:.3f}  LP-MC↔EXP-H={lh_t5:.3f}  Qwen↔EXP-H={qh_t5:.3f}")

    # EXP-H closer to LP-MC or Qwen?
    for space, lq, lh, qh in [('CLAP', lq_clap, lh_clap, qh_clap),
                                ('T5',   lq_t5,   lh_t5,   qh_t5)]:
        if lh_clap > lq_clap and space == 'CLAP' or lh_t5 > lq_t5 and space == 'T5':
            closer = 'LP-MC'
        else:
            closer = 'Qwen'
        print(f"\n  [{space}] EXP-H centroid is closer to {closer} (LP-MC↔EXP-H={lh:.3f} vs LP-MC↔Qwen={lq:.3f})")
        if closer == 'LP-MC':
            print(f"       → EXP-H rewrite shifted embedding distribution toward LP-MC")
            print(f"       → If collapse persists despite this shift, embedding proximity is NOT sufficient")
        else:
            print(f"       → EXP-H rewrite did NOT close the gap to LP-MC in {space} space")
            print(f"       → This supports: surface style changed but embedding dist still Qwen-like")

    # ── Save ─────────────────────────────────────────────────────────────
    # Convert numpy to python floats for JSON
    def to_py(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.float32, np.float64)): return float(x)
        if isinstance(x, dict): return {k: to_py(v) for k, v in x.items()}
        return x

    result = {
        't5': {
            'intra': {k: {kk: to_py(v) for kk, v in t5_stats[k].items() if kk != 'centroid'}
                      for k in CORPORA},
            'inter': to_py(t5_inter),
            '1nn':   to_py(t5_nn),
        },
        'clap': {
            'intra': {k: {kk: to_py(v) for kk, v in clap_stats[k].items() if kk != 'centroid'}
                      for k in CORPORA},
            'inter': to_py(clap_inter),
            '1nn':   to_py(clap_nn),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == '__main__':
    main()
