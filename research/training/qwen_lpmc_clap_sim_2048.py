"""
2048 subset 比較：Qwen 5-task captions vs LP-MusicCaps caption × audio CLAP sim

- 從 phase9_omni_captions.jsonl 抽 seed=42 random 2048 ids（與其他 2048 sub-set
  習慣一致）
- 對每個 id：load 1 段 audio → 跟 5 條 Qwen captions 算 CLAP cosine
- 同時讀 phase8_v3_clap_sim.jsonl 拿同一 id 的 LP-MC 1-of-4 random caption sim
  作對照（已是 251K 全量算過的子集）

輸出：
  ~/research/meanaudio_training/qwen_vs_lpmc_clap_sim_2048.jsonl
    每行：{"id":"...", "qwen_sims":[s0..s4], "lpmc_sim": float|null}

CLAP 模型與 eval 完全相同（HTSAT-base music_speech_audioset_epoch_15_esc_89.98）。
"""

import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import laion_clap

CLAP_CKPT      = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
QWEN_JSONL     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_SIM_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v3_clap_sim.jsonl')
AUDIO_ROOT     = Path('/home/hsiehyian/dataset/segments_no_vocals')
OUTPUT_JSONL   = Path('/home/kojiek/research/meanaudio_training/qwen_vs_lpmc_clap_sim_2048.jsonl')
SEED           = 42
N_SAMPLE       = 2048
BATCH_SIZE     = 32   # 訓練同時跑，留 buffer


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist  = '_'.join(parts[:seg_idx - 1])
    track   = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f'segment_{seg_num}.mp3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=N_SAMPLE)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # ── 1. 讀 Qwen captions JSONL ────────────────────────────────
    print(f'讀 {QWEN_JSONL.name} ...')
    qwen_data = {}
    with open(QWEN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            qwen_data[d['id']] = d['captions']
    print(f'  Qwen 共 {len(qwen_data):,} ids')

    # ── 2. 讀 LP-MC sim JSONL（251K 已算）─────────────────────────
    print(f'讀 {LPMC_SIM_JSONL.name} ...')
    lpmc_sim = {}
    with open(LPMC_SIM_JSONL) as f:
        for line in f:
            d = json.loads(line)
            lpmc_sim[d['id']] = d.get('clap_sim')
    print(f'  LP-MC sim 共 {len(lpmc_sim):,} ids')

    # ── 3. 取交集後 seed=42 抽樣 ─────────────────────────────────
    common_ids = sorted(set(qwen_data.keys()) & set(lpmc_sim.keys()))
    print(f'  Qwen ∩ LP-MC = {len(common_ids):,} ids')
    random.seed(SEED)
    sample_ids = random.sample(common_ids, args.n)
    print(f'  抽 seed={SEED} {args.n} ids')

    # ── 4. 載入 CLAP ─────────────────────────────────────────────
    print('\n載入 CLAP（HTSAT-base）...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(CLAP_CKPT)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'CLAP 就緒，device={device}')

    # ── 5. 跑 batch ───────────────────────────────────────────────
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = n_error = 0

    with open(OUTPUT_JSONL, 'w') as fout:
        for i in tqdm(range(0, len(sample_ids), args.batch), desc='CLAP sim'):
            batch_ids = sample_ids[i : i + args.batch]
            batch_paths = [str(id_to_audio_path(cid)) for cid in batch_ids]

            valid = [(cid, p) for cid, p in zip(batch_ids, batch_paths) if os.path.exists(p)]
            missing = [cid for cid, p in zip(batch_ids, batch_paths) if not os.path.exists(p)]

            for cid in missing:
                fout.write(json.dumps({'id': cid, 'qwen_sims': None, 'lpmc_sim': lpmc_sim.get(cid)}) + '\n')
                n_missing += 1

            if not valid:
                continue

            v_ids   = [x[0] for x in valid]
            v_paths = [x[1] for x in valid]

            try:
                with torch.no_grad():
                    audio_emb = model.get_audio_embedding_from_filelist(v_paths, use_tensor=True)
                    # 對每個 slot 算一次 text emb
                    slot_sims = []  # shape: [5, batch]
                    for slot in range(5):
                        caps = [qwen_data[cid][slot] for cid in v_ids]
                        text_emb = model.get_text_embedding(caps, use_tensor=True)
                        sims = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=-1)
                        slot_sims.append(sims.cpu().tolist())

                # transpose: [batch, 5]
                per_id_sims = list(zip(*slot_sims))
                for cid, sims5 in zip(v_ids, per_id_sims):
                    fout.write(json.dumps({
                        'id': cid,
                        'qwen_sims': [round(s, 6) for s in sims5],
                        'lpmc_sim': lpmc_sim.get(cid),
                    }) + '\n')
                    n_ok += 1

            except Exception as e:
                for cid in v_ids:
                    fout.write(json.dumps({'id': cid, 'qwen_sims': None, 'lpmc_sim': lpmc_sim.get(cid)}) + '\n')
                    n_error += 1
                tqdm.write(f'[WARN] batch error: {e}')

    print(f'\n完成 ok={n_ok:,}  missing={n_missing:,}  error={n_error:,}')
    print(f'輸出 → {OUTPUT_JSONL}')

    # ── 6. 統計 summary ──────────────────────────────────────────
    print('\n=== Summary ===')
    qwen_per_slot = [[] for _ in range(5)]
    qwen_mean5 = []
    qwen_random_1of5 = []
    lpmc_arr = []

    rng = random.Random(SEED + 1)  # 抽 1-of-5 用獨立 rng
    with open(OUTPUT_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d['qwen_sims'] is not None:
                for s in range(5):
                    qwen_per_slot[s].append(d['qwen_sims'][s])
                qwen_mean5.append(sum(d['qwen_sims']) / 5)
                qwen_random_1of5.append(d['qwen_sims'][rng.randrange(5)])
            if d.get('lpmc_sim') is not None:
                lpmc_arr.append(d['lpmc_sim'])

    def stats(name, arr):
        a = np.array(arr)
        print(f'  {name:30s} n={len(a):5d}  mean={a.mean():.4f}  median={np.median(a):.4f}  std={a.std():.4f}')

    print(f'\n--- Qwen (5 task captions per audio) ---')
    for s in range(5):
        stats(f'slot {s}', qwen_per_slot[s])
    stats('mean of 5 per audio', qwen_mean5)
    stats('random 1-of-5 per audio', qwen_random_1of5)
    print(f'\n--- LP-MusicCaps (random 1-of-4 per audio, same 2048) ---')
    stats('lpmc_sim', lpmc_arr)


if __name__ == '__main__':
    main()
