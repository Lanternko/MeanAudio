"""
B1: Specificity test
====================
H2 直接證據：Qwen captions raw CLAP sim 高 (0.298) > LP-MC (0.249)，但訓練後
模型 eval CLAP 崩到 0.06。檢查是否「Qwen captions 對自己 audio sim 高、但對隨機
其他 audio 也一樣高」（generic / low-specificity）。

對 seed=42 random 2048 subset：
  - 算 audio embedding (2048 個)
  - 算 caption embedding：
      * Qwen mean-of-5（先平均 5 個 caption embedding）
      * Qwen slot 0 (single)
      * LP-MC（從 phase7_v1_train.tsv random 1-of-4，跟 phase8_v3_clap_sim 同源）
  - 對每個 caption 算 sim 對「自己 audio」vs 對「其他 2047 audio」
  - discrimination = self_sim - mean(other_sims)
  - 同時看 self_sim vs max(other_sims) — 看 caption 對「最像的另一個 audio」差多少

輸出：
  ~/research/meanaudio_training/diag_b1_specificity.json
"""

import json
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import laion_clap
import csv

CLAP_CKPT      = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
QWEN_JSONL     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_TSV       = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
AUDIO_ROOT     = Path('/home/hsiehyian/dataset/segments_no_vocals')
OUT_JSON       = Path('/home/kojiek/research/meanaudio_training/diag_b1_specificity.json')
SEED           = 42
N_SAMPLE       = 2048
BATCH_AUDIO    = 32


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist  = '_'.join(parts[:seg_idx - 1])
    track   = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f'segment_{seg_num}.mp3'


def main():
    # ── 1. Load Qwen captions JSONL ───────────────────────────────
    print('讀 Qwen JSONL...')
    qwen_data = {}
    with open(QWEN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            qwen_data[d['id']] = d['captions']
    print(f'  Qwen ids = {len(qwen_data):,}')

    # ── 2. Load LP-MC TSV ─────────────────────────────────────────
    print('讀 LP-MC TSV (phase7_v1_train, 1-of-4 random)...')
    lpmc_data = {}
    with open(LPMC_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            lpmc_data[row['id']] = row['caption']
    print(f'  LP-MC ids = {len(lpmc_data):,}')

    # ── 3. Sample 2048 ─────────────────────────────────────────────
    common = sorted(set(qwen_data) & set(lpmc_data))
    random.seed(SEED)
    sample_ids = random.sample(common, N_SAMPLE)
    # Filter: keep only ids with audio file
    sample_ids = [i for i in sample_ids if id_to_audio_path(i).exists()]
    print(f'  sample (audio exists) = {len(sample_ids)}')

    audio_paths = [str(id_to_audio_path(i)) for i in sample_ids]

    # ── 4. Load CLAP ────────────────────────────────────────────────
    print('\n載入 CLAP...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(CLAP_CKPT)
    model.eval().to('cuda')

    # ── 5. Embed all audio (N x 512) ────────────────────────────────
    print(f'embed {len(audio_paths)} audio...')
    audio_embs = []
    for i in tqdm(range(0, len(audio_paths), BATCH_AUDIO), desc='audio'):
        batch = audio_paths[i:i + BATCH_AUDIO]
        with torch.no_grad():
            emb = model.get_audio_embedding_from_filelist(batch, use_tensor=True)
        audio_embs.append(emb.cpu())
    A = torch.cat(audio_embs, 0)        # (N, 512)
    A = torch.nn.functional.normalize(A, dim=-1)
    print(f'  audio_emb shape = {A.shape}')

    # ── 6. Embed text — 3 variants ──────────────────────────────────
    BATCH_TEXT = 64

    def embed_text(text_list):
        embs = []
        for i in tqdm(range(0, len(text_list), BATCH_TEXT), desc='text', leave=False):
            with torch.no_grad():
                e = model.get_text_embedding(text_list[i:i+BATCH_TEXT], use_tensor=True)
            embs.append(e.cpu())
        T = torch.cat(embs, 0)
        return torch.nn.functional.normalize(T, dim=-1)

    print('\nembed Qwen mean-of-5...')
    qwen_per_slot_emb = []
    for s in range(5):
        caps_s = [qwen_data[i][s] for i in sample_ids]
        qwen_per_slot_emb.append(embed_text(caps_s))     # (N, 512)
    Q5 = torch.stack(qwen_per_slot_emb, 0).mean(0)        # (N, 512), pre-normalize
    Q5 = torch.nn.functional.normalize(Q5, dim=-1)
    Q0 = qwen_per_slot_emb[0]                              # slot 0 only

    print('embed LP-MC...')
    L = embed_text([lpmc_data[i] for i in sample_ids])

    # ── 7. Pairwise sim matrix (N x N), diag = self ──────────────────
    def stats_for(T, name):
        # T: (N, 512), A: (N, 512)
        S = T @ A.T   # (N, N) cosine matrix
        N = S.size(0)
        eye = torch.eye(N, dtype=torch.bool)
        self_sim = S[eye]                       # (N,)
        # mask diagonal → other sims
        S_masked = S.clone()
        S_masked[eye] = float('-inf')
        max_other  = S_masked.max(dim=1).values  # (N,)
        S_masked[eye] = 0
        mean_other = (S_masked.sum(dim=1)) / (N - 1)
        discrim_mean = self_sim - mean_other
        discrim_max  = self_sim - max_other      # 可為負
        # rank of self among all N audios for each text
        # rank = how many other audios scored ≥ self_sim
        rank_among_all = (S >= self_sim.unsqueeze(1)).sum(dim=1)   # 1 = best (only self)
        recall_at_1  = (rank_among_all == 1).float().mean().item()
        recall_at_10 = (rank_among_all <= 10).float().mean().item()
        return {
            'name': name,
            'n': N,
            'self_sim_mean':   round(self_sim.mean().item(), 4),
            'self_sim_median': round(self_sim.median().item(), 4),
            'mean_other_mean': round(mean_other.mean().item(), 4),
            'max_other_mean':  round(max_other.mean().item(), 4),
            'discrim_mean':    round(discrim_mean.mean().item(), 4),
            'discrim_max':     round(discrim_max.mean().item(), 4),
            'recall@1':        round(recall_at_1, 4),
            'recall@10':       round(recall_at_10, 4),
        }

    results = []
    for T, name in [(Q5, 'Qwen mean-of-5'), (Q0, 'Qwen slot 0'), (L, 'LP-MC random 1-of-4')]:
        r = stats_for(T, name)
        results.append(r)
        print(f'\n{name}:')
        for k, v in r.items():
            if k == 'name': continue
            print(f'  {k:20s} {v}')

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f'\n→ {OUT_JSON}')

    # ── 8. Print compact comparison ────────────────────────────────
    print('\n' + '='*70)
    print('COMPACT — discrimination (caption × OWN audio − OTHER audio)')
    print('='*70)
    print(f"{'variant':<25} {'self':>7} {'mean_other':>10} {'discrim':>8} {'R@1':>6} {'R@10':>6}")
    for r in results:
        print(f"{r['name']:<25} {r['self_sim_mean']:>7.4f} {r['mean_other_mean']:>10.4f} "
              f"{r['discrim_mean']:>8.4f} {r['recall@1']:>6.3f} {r['recall@10']:>6.3f}")
    print('\n判讀：')
    print('  - discrim_mean 高 = caption 對「自己 audio」比「平均其他」明顯更貼 → 有特異性')
    print('  - discrim_mean 低（接近 0）= caption 對誰都差不多貼 → generic / low specificity')
    print('  - R@1 = 用 caption 找回正確 audio 的 top-1 正確率（純文字檢索）')


if __name__ == '__main__':
    main()
