"""
Audio-caption CLAP semantic alignment audit.

Computes proper diagonal vs off-diagonal retrieval metrics:
  - diagonal:  CLAP(audio_i, caption_i)  = correct pair
  - off-diag:  CLAP(audio_i, caption_j)  = wrong pair (shuffled)
  - R@1, R@10 retrieval metrics

Requires audio files to be accessible (segments_no_vocals).

Usage:
  # Eval set (2048, seed42 Jamendo subset):
  python audit_audio_caption_clap_alignment.py \
    --audio_root /home/hsiehyian/dataset/segments_no_vocals \
    --lp_tsv /mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv \
    --qwen_tsv /mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_random.tsv \
    --qwen_jsonl /mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_captions.jsonl \
    --num_samples 2048 \
    --out ~/research/meanaudio_training/qwen_audio_caption_alignment_seed42_full.json

  # Training sample (4096):
  python audit_audio_caption_clap_alignment.py \
    --audio_root /home/hsiehyian/dataset/segments_no_vocals \
    --lp_tsv /mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase7_v1_train.tsv \
    --qwen_tsv /mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_random_train.tsv \
    --qwen_jsonl /mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl \
    --num_samples 4096 \
    --out ~/research/meanaudio_training/qwen_audio_caption_alignment_train4096.json

Interpretation:
  diag_mean >> shuffled_mean  →  alignment working
  R@10 >> random (0.49%)      →  captions discriminate clips
  diag ≈ shuffled             →  potential audio-caption mismatch bug
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CLAP_CKPT = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'


def id_to_audio_path(clip_id: str, audio_root: Path) -> Path:
    """94_1317594_segment_0 → audio_root/94/1317594/segment_0.mp3"""
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist = '_'.join(parts[:seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return audio_root / artist / track / f'segment_{seg_num}.mp3'


def load_tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def load_qwen_jsonl(path):
    """Returns id → [cap0..cap4] dict."""
    lookup = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = str(d.get('id', ''))
            if not cid:
                continue
            caps = []
            for k, v in d.items():
                if k == 'id':
                    continue
                if isinstance(v, str):
                    caps.append(v.strip())
                elif isinstance(v, list):
                    caps += [str(x).strip() for x in v if x]
            lookup[cid] = caps
    return lookup


def compute_clap_retrieval(audio_paths, captions_lp, captions_qwen5, model, batch_size=32):
    """
    Returns:
      results_lp:  list of {id, diag_sim, rank_among_n, hit@1, hit@10}  for LP-MC
      results_qw:  same for Qwen (using random 1-of-5 slot)
    """
    N = len(audio_paths)
    device = next(model.parameters()).device

    # Encode all audios
    print(f'  Encoding {N} audio files...')
    audio_embs = []
    missing_idx = set()
    for i in tqdm(range(0, N, batch_size), desc='audio-emb'):
        batch_paths = audio_paths[i:i+batch_size]
        valid = [(j+i, p) for j, p in enumerate(batch_paths) if os.path.exists(p)]
        if not valid:
            for j in range(len(batch_paths)):
                missing_idx.add(i + j)
            audio_embs.extend([None] * len(batch_paths))
            continue
        v_idx, v_paths = zip(*valid)
        with torch.no_grad():
            embs = model.get_audio_embedding_from_filelist(list(v_paths), use_tensor=True)
        emb_dict = {vi: embs[ei] for ei, vi in enumerate(v_idx)}
        for j in range(len(batch_paths)):
            audio_embs.append(emb_dict.get(i + j))

    # Remove missing
    valid_mask = [e is not None for e in audio_embs]
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    print(f'  Missing audio: {N - len(valid_idx)}/{N}')

    if len(valid_idx) < 10:
        raise RuntimeError('Too few valid audio files')

    audio_matrix = torch.stack([audio_embs[i] for i in valid_idx])  # (M, 512)
    audio_matrix = torch.nn.functional.normalize(audio_matrix, dim=-1)

    # Encode LP-MC captions
    print('  Encoding LP-MC captions...')
    lp_caps = [captions_lp[i] for i in valid_idx]
    lp_embs = []
    for i in tqdm(range(0, len(lp_caps), batch_size), desc='lp-text-emb'):
        batch = lp_caps[i:i+batch_size]
        with torch.no_grad():
            embs = model.get_text_embedding(batch, use_tensor=True)
        lp_embs.append(embs)
    lp_matrix = torch.cat(lp_embs, dim=0)  # (M, 512)
    lp_matrix = torch.nn.functional.normalize(lp_matrix, dim=-1)

    # Encode Qwen captions (random 1-of-5)
    rng = random.Random(42)
    print('  Encoding Qwen captions (random 1-of-5)...')
    qw_caps = [rng.choice(captions_qwen5[i]) for i in valid_idx]
    qw_embs = []
    for i in tqdm(range(0, len(qw_caps), batch_size), desc='qw-text-emb'):
        batch = qw_caps[i:i+batch_size]
        with torch.no_grad():
            embs = model.get_text_embedding(batch, use_tensor=True)
        qw_embs.append(embs)
    qw_matrix = torch.cat(qw_embs, dim=0)  # (M, 512)
    qw_matrix = torch.nn.functional.normalize(qw_matrix, dim=-1)

    # Compute MxM audio-text sim matrices
    print('  Computing retrieval matrices...')
    # audio→text: for each audio_i, rank all M text captions
    lp_sim_matrix = (audio_matrix @ lp_matrix.T).cpu().numpy()  # (M, M)
    qw_sim_matrix = (audio_matrix @ qw_matrix.T).cpu().numpy()  # (M, M)
    M = len(valid_idx)

    def retrieval_metrics(sim_matrix):
        diag = np.diag(sim_matrix)                       # diagonal sims
        shuffled_mean = (sim_matrix.sum() - diag.sum()) / (M * (M - 1))
        ranks = []
        for i in range(M):
            row = sim_matrix[i]
            rank = 1 + (row > row[i]).sum()              # rank of correct item
            ranks.append(int(rank))
        ranks = np.array(ranks)
        return {
            'diag_mean': float(diag.mean()),
            'diag_std':  float(diag.std()),
            'shuffled_mean': float(shuffled_mean),
            'diag_minus_shuffled': float(diag.mean() - shuffled_mean),
            'median_rank': float(np.median(ranks)),
            'R@1':  float((ranks <= 1).mean()),
            'R@5':  float((ranks <= 5).mean()),
            'R@10': float((ranks <= 10).mean()),
            'n': M,
            'random_R@1':  float(1/M),
            'random_R@10': float(10/M),
        }

    lp_metrics = retrieval_metrics(lp_sim_matrix)
    qw_metrics = retrieval_metrics(qw_sim_matrix)

    # Worst 20 Qwen clips
    qw_diag = np.diag(qw_sim_matrix)
    worst_order = np.argsort(qw_diag)[:20]
    worst = [{'idx': int(valid_idx[i]), 'diag_sim': float(qw_diag[i])} for i in worst_order]

    return lp_metrics, qw_metrics, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_root', required=True)
    parser.add_argument('--lp_tsv',   required=True, help='LP-MC caption TSV')
    parser.add_argument('--qwen_tsv', required=True, help='Qwen single-cap TSV (for clip id order)')
    parser.add_argument('--qwen_jsonl', required=True, help='Qwen 5-cap JSONL')
    parser.add_argument('--num_samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    lp_rows = load_tsv(args.lp_tsv)
    qw_rows = load_tsv(args.qwen_tsv)
    qw_jsonl = load_qwen_jsonl(args.qwen_jsonl)

    # Align by id
    lp_by_id = {r['id']: r['caption'] for r in lp_rows}
    assert len(lp_rows) == len(qw_rows), f'TSV length mismatch: {len(lp_rows)} vs {len(qw_rows)}'

    # Sample
    rng = random.Random(args.seed)
    all_ids = [r['id'] for r in qw_rows]
    sample_ids = rng.sample(all_ids, min(args.num_samples, len(all_ids)))

    audio_paths  = [str(id_to_audio_path(cid, audio_root)) for cid in sample_ids]
    captions_lp  = [lp_by_id.get(cid, '') for cid in sample_ids]
    captions_qw5 = [qw_jsonl.get(cid, ['']) for cid in sample_ids]

    # Load CLAP
    import laion_clap
    print('Loading CLAP...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'  device={device}')

    lp_metrics, qw_metrics, worst = compute_clap_retrieval(
        audio_paths, captions_lp, captions_qw5, model
    )

    result = {
        'meta': {
            'date': '2026-05-17',
            'audio_root': str(audio_root),
            'lp_tsv': args.lp_tsv,
            'qwen_tsv': args.qwen_tsv,
            'num_samples': args.num_samples,
        },
        'lp_mc_retrieval': lp_metrics,
        'qwen_retrieval': qw_metrics,
        'worst_20_qwen': worst,
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n=== Results ===')
    print(f'LP-MC:  diag={lp_metrics["diag_mean"]:.4f}  shuffled={lp_metrics["shuffled_mean"]:.4f}  '
          f'gap={lp_metrics["diag_minus_shuffled"]:.4f}  R@1={100*lp_metrics["R@1"]:.1f}%  R@10={100*lp_metrics["R@10"]:.1f}%')
    print(f'Qwen:   diag={qw_metrics["diag_mean"]:.4f}  shuffled={qw_metrics["shuffled_mean"]:.4f}  '
          f'gap={qw_metrics["diag_minus_shuffled"]:.4f}  R@1={100*qw_metrics["R@1"]:.1f}%  R@10={100*qw_metrics["R@10"]:.1f}%')
    print(f'Random: R@1={100*qw_metrics["random_R@1"]:.2f}%  R@10={100*qw_metrics["random_R@10"]:.2f}%')
    print(f'\nSaved → {out_path}')

    if qw_metrics['diag_minus_shuffled'] > 0.01:
        print('\n✅ Qwen audio-caption alignment: diagonal > shuffled (normal)')
    elif qw_metrics['diag_minus_shuffled'] < -0.01:
        print('\n❌ WARNING: Qwen diagonal < shuffled — possible audio-caption mismatch bug')
    else:
        print('\n⚠️  Qwen diagonal ≈ shuffled — ambiguous (low discriminability)')


if __name__ == '__main__':
    main()
