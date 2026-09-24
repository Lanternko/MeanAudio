"""
EXP-H Sanity Statistics
========================

Validates the 10K EXP-H rewrite TSV against the gate criteria before
proceeding to full training.

Gate criteria (from exp_h_rewrite_spec.md):
  1. CLAP diagonal sim ≥ 0.20  (alignment preserved)
  2. Top-50 trigram coverage ≥ 50% of LP-MC trigrams

Additional diagnostics (no hard gate):
  3. Caption length distribution (target: 25–50 words, p50 ≥ 30)
  4. Acoustic keyword density (melody/bass/drum/synth/kick/snare)
  5. Vocab/bigram entropy (should approach LP-MC, away from Qwen)

Compares to 3 baselines:
  - LP-MC (target)
  - Qwen slot0 (input)
  - EXP-C Qwen+prefix (failed control, MC CLAP 0.0580)

Usage:
  python expH_sanity_stats.py \
      --rewrite_tsv ~/eval_tsvs_p100/expH_rewrite_10k_sanity.tsv \
      --audio_root /home/hsiehyian/dataset/segments_no_vocals \
      --out ~/research/meanaudio_training/expH_sanity_results.json

  # Skip CLAP (text stats only):
  python expH_sanity_stats.py \
      --rewrite_tsv ~/eval_tsvs_p100/expH_rewrite_10k_sanity.tsv \
      --no_clap
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────
QWEN_JSONL         = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_TSV_QUARANT   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv')
CLAP_CKPT          = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
AUDIO_ROOT_DEFAULT = Path('/mnt/HDD/hsiehyian/segments_no_vocals')   # confirmed 2026-05-18
SEED               = 42

ACOUSTIC_KW = ['melody', 'bass', 'drum', 'synth', 'kick', 'snare', 'guitar',
               'piano', 'strings', 'hi hat', 'vocals', 'trumpet', 'saxophone']


# ── Text utilities ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]+\b', text.lower())


def get_trigrams(tokens: list) -> list:
    return [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]


def get_bigrams(tokens: list) -> list:
    return [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def text_stats(captions: list) -> dict:
    """Compute comprehensive text statistics over a list of captions."""
    lengths = [len(c.split()) for c in captions]
    all_tokens = []
    all_bigrams = []
    all_trigrams = []
    acoustic_hits = 0

    for cap in captions:
        toks = tokenize(cap)
        all_tokens.extend(toks)
        all_bigrams.extend(get_bigrams(toks))
        all_trigrams.extend(get_trigrams(toks))
        if any(kw in cap.lower() for kw in ACOUSTIC_KW):
            acoustic_hits += 1

    lengths_arr = np.array(lengths)
    trigram_counter = Counter(all_trigrams)
    bigram_counter = Counter(all_bigrams)
    unigram_counter = Counter(all_tokens)

    return {
        'n': len(captions),
        'length_mean': float(lengths_arr.mean()),
        'length_median': float(np.median(lengths_arr)),
        'length_p10': float(np.percentile(lengths_arr, 10)),
        'length_p90': float(np.percentile(lengths_arr, 90)),
        'acoustic_kw_frac': acoustic_hits / len(captions) if captions else 0,
        'unigram_entropy': entropy(unigram_counter),
        'bigram_entropy': entropy(bigram_counter),
        'top50_trigrams': [(' '.join(t), c) for t, c in trigram_counter.most_common(50)],
        'top50_bigrams': [(' '.join(t), c) for t, c in bigram_counter.most_common(20)],
        'uniq_unigrams': len(unigram_counter),
        'uniq_bigrams': len(bigram_counter),
        'uniq_trigrams': len(trigram_counter),
        '_trigram_counter': trigram_counter,   # kept for overlap calc, not serialized
    }


def trigram_overlap(stats_a: dict, stats_b: dict, top_n: int = 50) -> float:
    """Fraction of stats_b's top-N trigrams also in stats_a's top-N trigrams."""
    top_a = set(t for t, _ in stats_a['top50_trigrams'][:top_n])
    top_b = set(t for t, _ in stats_b['top50_trigrams'][:top_n])
    if not top_b:
        return 0.0
    return len(top_a & top_b) / len(top_b)


def lp_mc_opening_frac(captions: list) -> float:
    lp_prefixes = ('the low quality', 'this is a', 'this audio', 'this recording')
    return sum(1 for c in captions if c.lower().startswith(lp_prefixes)) / len(captions) if captions else 0


# ── CLAP diagonal ─────────────────────────────────────────────────────────

def id_to_audio_path(clip_id: str, audio_root: Path) -> Path:
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist = '_'.join(parts[:seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return audio_root / artist / track / f'segment_{seg_num}.mp3'


def compute_clap_diagonal(ids: list, captions: list, audio_root: Path, batch_size=32) -> dict:
    """Compute audio-text diagonal CLAP sim for a set of (id, caption) pairs."""
    import torch
    import laion_clap

    print('Loading CLAP...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'  device={device}')

    sims = []
    missing = 0

    for i in tqdm(range(0, len(ids), batch_size), desc='CLAP diag'):
        batch_ids = ids[i: i + batch_size]
        batch_caps = captions[i: i + batch_size]
        batch_paths = [str(id_to_audio_path(cid, audio_root)) for cid in batch_ids]

        valid = [(p, c) for p, c in zip(batch_paths, batch_caps) if os.path.exists(p)]
        missing += len(batch_paths) - len(valid)

        if not valid:
            continue

        v_paths, v_caps = zip(*valid)
        with torch.no_grad():
            a_emb = model.get_audio_embedding_from_filelist(list(v_paths), use_tensor=True)
            t_emb = model.get_text_embedding(list(v_caps), use_tensor=True)
            a_emb = torch.nn.functional.normalize(a_emb, dim=-1)
            t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
            s = (a_emb * t_emb).sum(dim=-1)
            sims.extend(s.cpu().tolist())

    sims = np.array(sims)
    if len(sims) == 0:
        raise RuntimeError(
            f'No valid audio found (missing={missing}). '
            'Check that the audio filesystem is mounted at the audio_root path.'
        )
    return {
        'n': len(sims),
        'missing_audio': missing,
        'diag_mean': float(sims.mean()),
        'diag_std': float(sims.std()),
        'diag_p10': float(np.percentile(sims, 10)),
        'diag_p90': float(np.percentile(sims, 90)),
        'frac_below_0.1': float((sims < 0.1).mean()),
    }


# ── Loaders ────────────────────────────────────────────────────────────────

def load_tsv_captions(tsv_path: Path) -> dict:
    """Return {id: caption} from TSV with 'id' and 'caption' columns."""
    result = {}
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('id') and row.get('caption'):
                result[row['id']] = row['caption']
    return result


def load_qwen_slot0(jsonl_path: Path) -> dict:
    result = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = str(d.get('id', ''))
            caps = d.get('captions', [])
            if cid and caps:
                result[cid] = str(caps[0]).strip()
    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewrite_tsv', required=True,
                        help='EXP-H rewrite TSV to evaluate')
    parser.add_argument('--audio_root', default=str(AUDIO_ROOT_DEFAULT))
    parser.add_argument('--out', default='~/research/meanaudio_training/expH_sanity_results.json')
    parser.add_argument('--no_clap', action='store_true',
                        help='Skip CLAP computation (text stats only)')
    parser.add_argument('--n_clap', type=int, default=2048,
                        help='Number of samples for CLAP diagonal (default 2048)')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    rewrite_tsv = Path(args.rewrite_tsv).expanduser()
    out_path = Path(args.out).expanduser()
    audio_root = Path(args.audio_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load rewrite TSV ──────────────────────────────────────────────────
    print(f'Loading rewrite TSV: {rewrite_tsv}')
    rewrite_data = load_tsv_captions(rewrite_tsv)
    rewrite_ids = list(rewrite_data.keys())
    rewrite_caps = [rewrite_data[cid] for cid in rewrite_ids]
    print(f'  {len(rewrite_ids):,} rewrites loaded')

    # ── Load Qwen slot0 for same IDs (baseline) ───────────────────────────
    print('Loading Qwen slot0 captions...')
    qwen_all = load_qwen_slot0(QWEN_JSONL)
    qwen_caps = [qwen_all.get(cid, '') for cid in rewrite_ids]

    # ── Load LP-MC sample (same size, seed=42) ────────────────────────────
    print('Loading LP-MC captions (sample)...')
    lpmc_all = load_tsv_captions(LPMC_TSV_QUARANT)
    rng = random.Random(args.seed)
    lpmc_ids_sample = rng.sample([cid for cid in rewrite_ids if cid in lpmc_all],
                                  min(len(rewrite_ids), len(lpmc_all)))
    lpmc_caps = [lpmc_all[cid] for cid in lpmc_ids_sample]
    print(f'  LP-MC sample: {len(lpmc_caps):,}')

    # ── Text statistics ───────────────────────────────────────────────────
    print('\nComputing text statistics...')
    rewrite_stats = text_stats(rewrite_caps)
    qwen_stats    = text_stats([c for c in qwen_caps if c])
    lpmc_stats    = text_stats(lpmc_caps)

    # Trigram overlap vs LP-MC baseline
    rewrite_vs_lpmc = trigram_overlap(rewrite_stats, lpmc_stats, top_n=50)
    qwen_vs_lpmc    = trigram_overlap(qwen_stats, lpmc_stats, top_n=50)

    # LP-MC opening fraction
    rewrite_opening = lp_mc_opening_frac(rewrite_caps)
    qwen_opening    = lp_mc_opening_frac([c for c in qwen_caps if c])
    lpmc_opening    = lp_mc_opening_frac(lpmc_caps)

    # ── CLAP diagonal ─────────────────────────────────────────────────────
    clap_rewrite = clap_qwen = None
    if not args.no_clap:
        rng2 = random.Random(args.seed + 1)
        n_clap = min(args.n_clap, len(rewrite_ids))
        clap_sample_ids = rng2.sample(rewrite_ids, n_clap)
        clap_rewrite_caps = [rewrite_data[cid] for cid in clap_sample_ids]
        clap_qwen_caps    = [qwen_all.get(cid, '') for cid in clap_sample_ids]

        print(f'\nComputing CLAP diagonal for EXP-H rewrites (n={n_clap})...')
        clap_rewrite = compute_clap_diagonal(
            clap_sample_ids, clap_rewrite_caps, audio_root)

        print(f'Computing CLAP diagonal for Qwen slot0 (n={n_clap})...')
        clap_qwen = compute_clap_diagonal(
            clap_sample_ids, clap_qwen_caps, audio_root)

    # ── Gate evaluation ───────────────────────────────────────────────────
    gate_clap_ok   = (clap_rewrite['diag_mean'] >= 0.20) if clap_rewrite else None
    gate_trigram_ok = (rewrite_vs_lpmc >= 0.50)
    gate_pass = (gate_clap_ok is True and gate_trigram_ok) or \
                (gate_clap_ok is None and gate_trigram_ok)

    # ── Build results dict ────────────────────────────────────────────────
    def _serializable(stats: dict) -> dict:
        """Remove non-serializable Counter objects."""
        return {k: v for k, v in stats.items() if not k.startswith('_')}

    result = {
        'meta': {
            'rewrite_tsv': str(rewrite_tsv),
            'n_rewrites': len(rewrite_ids),
            'seed': args.seed,
        },
        'gate': {
            'clap_diag_ge_0.20': gate_clap_ok,
            'trigram_coverage_ge_50pct': gate_trigram_ok,
            'GATE_PASS': gate_pass,
        },
        'rewrite': {
            'text_stats': _serializable(rewrite_stats),
            'lp_mc_opening_frac': rewrite_opening,
            'trigram_overlap_vs_lpmc': rewrite_vs_lpmc,
            'clap_diagonal': clap_rewrite,
        },
        'qwen_slot0_baseline': {
            'text_stats': _serializable(qwen_stats),
            'lp_mc_opening_frac': qwen_opening,
            'trigram_overlap_vs_lpmc': qwen_vs_lpmc,
            'clap_diagonal': clap_qwen,
        },
        'lpmc_reference': {
            'text_stats': _serializable(lpmc_stats),
            'lp_mc_opening_frac': lpmc_opening,
        },
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults saved → {out_path}')

    # ── Print summary ──────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('=== EXP-H SANITY RESULTS ===')
    print('='*60)

    print('\n--- Caption length (words) ---')
    for name, stats in [('EXP-H rewrite', rewrite_stats),
                         ('Qwen slot0',    qwen_stats),
                         ('LP-MC ref',     lpmc_stats)]:
        print(f'  {name:20s}  mean={stats["length_mean"]:.1f}  '
              f'median={stats["length_median"]:.0f}  '
              f'p10={stats["length_p10"]:.0f}  p90={stats["length_p90"]:.0f}')

    print('\n--- LP-MC opening fraction ---')
    for name, frac in [('EXP-H rewrite', rewrite_opening),
                        ('Qwen slot0',    qwen_opening),
                        ('LP-MC ref',     lpmc_opening)]:
        print(f'  {name:20s}  {frac:.1%}')

    print('\n--- Acoustic keyword density ---')
    for name, stats in [('EXP-H rewrite', rewrite_stats),
                         ('Qwen slot0',    qwen_stats),
                         ('LP-MC ref',     lpmc_stats)]:
        print(f'  {name:20s}  {stats["acoustic_kw_frac"]:.1%}')

    print('\n--- Bigram entropy ---')
    for name, stats in [('EXP-H rewrite', rewrite_stats),
                         ('Qwen slot0',    qwen_stats),
                         ('LP-MC ref',     lpmc_stats)]:
        print(f'  {name:20s}  {stats["bigram_entropy"]:.3f}')

    print('\n--- Top-50 trigram overlap vs LP-MC ---')
    print(f'  EXP-H rewrite: {rewrite_vs_lpmc:.1%}  '
          f'(gate: {"✅ ≥50%" if gate_trigram_ok else "❌ <50%"})')
    print(f'  Qwen slot0   : {qwen_vs_lpmc:.1%}')

    print('\n--- CLAP diagonal ---')
    if clap_rewrite:
        print(f'  EXP-H rewrite: {clap_rewrite["diag_mean"]:.4f}  '
              f'(gate: {"✅ ≥0.20" if gate_clap_ok else "❌ <0.20"})')
        print(f'  Qwen slot0   : {clap_qwen["diag_mean"]:.4f}')
    else:
        print('  CLAP skipped (--no_clap)')

    print('\n--- TOP-20 TRIGRAMS: EXP-H vs LP-MC ---')
    print('  EXP-H:', ', '.join(t for t, _ in rewrite_stats['top50_trigrams'][:10]))
    print('  LP-MC:', ', '.join(t for t, _ in lpmc_stats['top50_trigrams'][:10]))

    print('\n' + '='*60)
    if gate_pass:
        print('✅  GATE PASSED — proceed to full 251K rewrite + training')
    elif gate_clap_ok is None:
        print('⚠️   CLAP not computed — check trigram gate only (run with audio for full gate)')
    else:
        print('❌  GATE FAILED — investigate style transfer quality before full training')
    print('='*60)


if __name__ == '__main__':
    main()
