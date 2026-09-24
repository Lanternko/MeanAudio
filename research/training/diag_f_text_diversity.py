"""
F + B2.5: Text-text inter-caption diversity + T5 NPZ verify
============================================================

F1 (CLAP space):
  - 2048 random ids, embed CLAP text for LP-MC vs Qwen (mean-of-5 / slot 0)
  - 算 N×N cosine matrix, mean off-diagonal = inter-caption similarity
  - 高 off-diag = caption 之間互相靠近 → batch contrast 弱

F2 (T5 space):
  - 同樣的 caption 用 google/flan-t5-large encode
  - mean pool over tokens (mask padding)
  - 算 N×N cosine, mean off-diagonal
  - 直接反映訓練時 cross-attention 看到的 text feature 分佈

B2.5 (T5 NPZ verify):
  - 從 ~/phase9_5_bc_singlecap_npz/ 抽 5 個 NPZ
  - 取 text_features (77, 1024) — masked mean pool
  - 對應 caption 用 fresh T5 forward + same pooling
  - cosine 應 ≈ 1.0；< 0.99 → T5 cache 有 bug

輸出：
  ~/research/meanaudio_training/diag_f_text_diversity.json
"""

import json
import csv
import random
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

QWEN_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
QWEN_SINGLECAP_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_bc_train.tsv')
NPZ_DIR_QWEN_SINGLE = Path('/home/kojiek/phase9_5_bc_singlecap_npz')
OUT_JSON   = Path('/home/kojiek/research/meanaudio_training/diag_f_text_diversity.json')
SEED = 42
N_SAMPLE = 2048


def load_qwen():
    d = {}
    with open(QWEN_JSONL) as f:
        for line in f:
            j = json.loads(line)
            d[j['id']] = j['captions']
    return d


def load_lpmc():
    d = {}
    with open(LPMC_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            d[row['id']] = row['caption']
    return d


def load_tsv_rows(p):
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows


def offdiag_mean(emb_normalized):
    """emb: (N, D) normalized. Return mean of off-diagonal cosine."""
    N = emb_normalized.size(0)
    S = emb_normalized @ emb_normalized.T   # (N, N)
    eye = torch.eye(N, dtype=torch.bool)
    S_off = S.clone()
    S_off[eye] = 0
    return float(S_off.sum()) / (N * (N - 1))


def main():
    print('讀 Qwen + LP-MC...')
    qwen = load_qwen()
    lpmc = load_lpmc()
    common = sorted(set(qwen) & set(lpmc))
    random.seed(SEED)
    sample_ids = random.sample(common, N_SAMPLE)

    qwen_caps_slot0    = [qwen[i][0] for i in sample_ids]
    qwen_caps_meanof5  = qwen   # need to embed all 5, average then normalize
    lpmc_caps          = [lpmc[i] for i in sample_ids]

    # ─── F1: CLAP space text-text matrix ─────────────────────────────
    print('\n=== F1: CLAP text-text inter-caption similarity ===')
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt('/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt')
    clap.eval().to('cuda')

    BATCH = 64

    def clap_embed(caps):
        embs = []
        for i in range(0, len(caps), BATCH):
            with torch.no_grad():
                e = clap.get_text_embedding(caps[i:i+BATCH], use_tensor=True)
            embs.append(e.cpu())
        T = torch.cat(embs, 0)
        return torch.nn.functional.normalize(T, dim=-1)

    print('  CLAP embed Qwen slot 0...')
    Cq0 = clap_embed(qwen_caps_slot0)
    print('  CLAP embed Qwen all 5 (avg+norm)...')
    Cq5 = []
    for s in range(5):
        Cq5.append(clap_embed([qwen[i][s] for i in sample_ids]))
    Cq5 = torch.nn.functional.normalize(torch.stack(Cq5).mean(0), dim=-1)
    print('  CLAP embed LP-MC...')
    Cl  = clap_embed(lpmc_caps)

    f1 = {
        'qwen_slot0_offdiag':    round(offdiag_mean(Cq0), 4),
        'qwen_meanof5_offdiag':  round(offdiag_mean(Cq5), 4),
        'lpmc_offdiag':          round(offdiag_mean(Cl), 4),
    }
    print(f'  CLAP off-diag mean (caption-caption similarity within dataset):')
    for k, v in f1.items():
        print(f'    {k:30s} {v:.4f}')
    print('  判讀：高 = 不同 audio 的 caption 在 CLAP 空間互相靠近，contrast 弱')

    # 釋放 CLAP
    del clap
    torch.cuda.empty_cache()

    # ─── F2: T5 space text-text matrix ───────────────────────────────
    print('\n=== F2: T5 (flan-t5-large) text-text inter-caption similarity ===')
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
    t5 = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().to('cuda')

    @torch.no_grad()
    def t5_embed(caps, max_length=77):
        embs = []
        for i in range(0, len(caps), BATCH):
            batch = caps[i:i+BATCH]
            tk = tok(batch, return_tensors='pt', padding='max_length',
                     truncation=True, max_length=max_length).to('cuda')
            out = t5(**tk).last_hidden_state    # (B, T, 1024)
            mask = tk['attention_mask'].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.cpu())
        T = torch.cat(embs, 0)
        return torch.nn.functional.normalize(T, dim=-1)

    print('  T5 embed Qwen slot 0...')
    Tq0 = t5_embed(qwen_caps_slot0)
    print('  T5 embed Qwen all 5 (avg+norm)...')
    Tq5 = []
    for s in range(5):
        Tq5.append(t5_embed([qwen[i][s] for i in sample_ids]))
    Tq5 = torch.nn.functional.normalize(torch.stack(Tq5).mean(0), dim=-1)
    print('  T5 embed LP-MC...')
    Tl  = t5_embed(lpmc_caps)

    f2 = {
        'qwen_slot0_offdiag':    round(offdiag_mean(Tq0), 4),
        'qwen_meanof5_offdiag':  round(offdiag_mean(Tq5), 4),
        'lpmc_offdiag':          round(offdiag_mean(Tl), 4),
    }
    print(f'  T5 off-diag mean:')
    for k, v in f2.items():
        print(f'    {k:30s} {v:.4f}')

    # ─── B2.5: T5 NPZ verify ─────────────────────────────────────────
    print('\n=== B2.5: T5 NPZ cache verify (5 random ids) ===')
    tsv_rows = load_tsv_rows(QWEN_SINGLECAP_TSV)
    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(tsv_rows)), 8)

    @torch.no_grad()
    def t5_full(cap, max_length=77):
        tk = tok([cap], return_tensors='pt', padding='max_length',
                 truncation=True, max_length=max_length).to('cuda')
        out = t5(**tk).last_hidden_state[0].cpu().numpy()  # (77, 1024)
        return out, tk['attention_mask'][0].cpu().numpy()

    b25 = {'samples': []}
    for idx in sample_idx:
        row = tsv_rows[idx]
        cid = row['id']
        cap = row['caption']
        npz_path = NPZ_DIR_QWEN_SINGLE / f'{idx}.npz'
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        cached_t5 = z['text_features']      # (77, 1024)

        fresh_t5, mask = t5_full(cap)

        # masked mean pool
        m = mask[:, None].astype(np.float32)
        cp = (cached_t5 * m).sum(0) / m.sum(0).clip(min=1)
        fp = (fresh_t5  * m).sum(0) / m.sum(0).clip(min=1)
        cos_pool = float(np.dot(cp, fp) / (np.linalg.norm(cp) * np.linalg.norm(fp) + 1e-8))

        # frame-by-frame cosine on valid positions
        valid = mask.astype(bool)
        c_valid = cached_t5[valid]
        f_valid = fresh_t5[valid]
        norms_c = np.linalg.norm(c_valid, axis=1)
        norms_f = np.linalg.norm(f_valid, axis=1)
        frame_cos = (c_valid * f_valid).sum(1) / (norms_c * norms_f + 1e-8)

        b25['samples'].append({
            'idx': idx, 'id': cid, 'cap_head': cap[:60],
            'pool_cos': round(cos_pool, 4),
            'frame_cos_mean': round(float(frame_cos.mean()), 4),
            'frame_cos_min':  round(float(frame_cos.min()),  4),
        })
        print(f'  idx={idx:>6}  pool_cos={cos_pool:.4f}  frame_mean={frame_cos.mean():.4f}  min={frame_cos.min():.4f}')

    pool_coses = [s['pool_cos'] for s in b25['samples']]
    if pool_coses:
        b25['pool_cos_mean'] = round(float(np.mean(pool_coses)), 4)
        b25['pool_cos_min']  = round(float(np.min(pool_coses)), 4)
        b25['n_below_999']   = int(sum(1 for c in pool_coses if c < 0.999))
        if b25['n_below_999'] > 0:
            print(f'  ⚠️  T5 NPZ cache 與 fresh forward 不一致 — n<0.999={b25["n_below_999"]}/{len(pool_coses)}')
        else:
            print(f'  ✅ T5 NPZ cache 正確（全部 cos > 0.999）')

    out = {'f1_clap_offdiag': f1, 'f2_t5_offdiag': f2, 'b25_t5_npz_verify': b25}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f'\n→ {OUT_JSON}')

    # ─── Summary ─────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('COMPACT — text-text inter-caption similarity (off-diag mean of N×N)')
    print('='*70)
    print(f"{'space':<10} {'qwen_slot0':>12} {'qwen_meanof5':>14} {'lpmc':>10}")
    print(f"{'CLAP':<10} {f1['qwen_slot0_offdiag']:>12.4f} {f1['qwen_meanof5_offdiag']:>14.4f} {f1['lpmc_offdiag']:>10.4f}")
    print(f"{'T5':<10} {f2['qwen_slot0_offdiag']:>12.4f} {f2['qwen_meanof5_offdiag']:>14.4f} {f2['lpmc_offdiag']:>10.4f}")
    print('\n判讀：')
    print('  - off-diag 高 = 不同 audio 的 caption embedding 互相像 → batch contrast 弱')
    print('  - 訓練時 model 看到的是 T5 (training), CLAP cond (CFG signal)')


if __name__ == '__main__':
    main()
