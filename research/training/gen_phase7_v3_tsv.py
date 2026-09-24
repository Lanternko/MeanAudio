"""
生成 Phase 7 V3 訓練 TSV
對每個 clip 的 5 個候選 caption，計算 text-text pairwise cosine similarity，
取 avg_similarity 最低的 caption（worst-consensus，最偏離群組的描述）。

用法：
  python3 gen_phase7_v3_tsv.py [--limit N] [--cache_path PATH]

  --limit N       只處理前 N 筆（測試用）
  --cache_path    similarity 快取路徑（預設 ~/research/meanaudio_training/text_sim_scores_p7v3.npz）
"""

import json
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

JSONL_PATH    = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
INPUT_TSV     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV    = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v3_train.tsv')
CLAP_CKPT     = Path('/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt')
DEFAULT_CACHE = Path.home() / 'research/meanaudio_training/text_sim_scores_p7v3.npz'

TEXT_BATCH = 256  # 純文字，batch 可以更大


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cache_path', type=str, default=str(DEFAULT_CACHE))
    return parser.parse_args()


def relative_path_to_id(relative_path: str) -> str:
    """00/1002000/segment_0.mp3 → 00_1002000_segment_0"""
    return relative_path.replace('.mp3', '').replace('/', '_')


def load_jsonl(jsonl_path):
    """id → [caption_0, ..., caption_4] 的 lookup dict（永遠讀全部）"""
    lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            clip_id = relative_path_to_id(d['relative_path'])
            captions = [c['caption'].replace('\n', ' ').replace('\r', ' ').strip()
                        for c in d['caption_details']]
            lookup[clip_id] = captions
    return lookup


def load_clap_model():
    import torch
    import laion_clap
    print('[CLAP] 載入模型（text encoder only）...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model


def get_text_embeddings(model, texts):
    """批次取得 text embeddings，回傳 normalized numpy array (N, D)"""
    import torch
    import laion_clap
    device = next(model.parameters()).device
    all_embs = []
    for i in range(0, len(texts), TEXT_BATCH):
        batch = texts[i: i + TEXT_BATCH]
        with torch.no_grad():
            emb = model.get_text_embedding(batch, use_tensor=True)  # (B, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def compute_worst_consensus_indices(model, clip_ids, lookup):
    """
    對每個 clip 的 5 個 caption 計算 text-text pairwise cosine similarity，
    回傳 {clip_id: worst_idx}（avg_similarity 最低的 caption index）
    """
    import torch

    # 攤平所有 (clip_id, caption_idx, caption_text)
    flat_ids   = []   # clip_id for each caption
    flat_cidx  = []   # caption index (0~4)
    flat_texts = []

    for cid in tqdm(clip_ids, desc='展開 captions'):
        caps = lookup.get(cid)
        if caps is None:
            continue
        for ci, cap in enumerate(caps):
            flat_ids.append(cid)
            flat_cidx.append(ci)
            flat_texts.append(cap)

    print(f'共 {len(flat_texts):,} 個 caption，開始 encode...')
    embs = get_text_embeddings(model, flat_texts)   # (N_total, D)

    # 重組成 {clip_id: emb_matrix (5, D)}
    from collections import defaultdict
    clip_embs = defaultdict(list)
    clip_order = defaultdict(list)
    for i, (cid, ci) in enumerate(zip(flat_ids, flat_cidx)):
        clip_embs[cid].append(embs[i])
        clip_order[cid].append(ci)

    # 對每個 clip 計算 pairwise similarity，找 worst
    result = {}
    for cid in clip_ids:
        if cid not in clip_embs:
            result[cid] = 0  # fallback
            continue
        mat = np.stack(clip_embs[cid], axis=0)    # (K, D)
        order = clip_order[cid]
        sim = mat @ mat.T                           # (K, K)
        K = len(order)
        avg_sim = (sim.sum(axis=1) - 1.0) / max(K - 1, 1)   # 排除對角線
        worst_local_idx = int(np.argmin(avg_sim))
        result[cid] = order[worst_local_idx]
    return result


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)

    print(f'讀取 JSONL: {JSONL_PATH}')
    lookup = load_jsonl(JSONL_PATH)
    print(f'  載入 {len(lookup):,} 筆 caption lookup')

    # 讀取 TSV（套用 limit 只影響輸出行數）
    rows_in = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            rows_in.append(row)
    if args.limit:
        rows_in = rows_in[:args.limit]

    clip_ids = [r['id'] for r in rows_in]

    # ── 快取 ────────────────────────────────────────────
    if cache_path.exists():
        print(f'[快取] 載入 {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        worst_idx_map = {k: int(v) for k, v in zip(cached['ids'], cached['worst_idx'])}
        # 檢查是否有未快取的 clip
        missing = [cid for cid in clip_ids if cid not in worst_idx_map]
        if missing:
            print(f'  {len(missing):,} 筆未命中快取，補算...')
            model = load_clap_model()
            extra = compute_worst_consensus_indices(model, missing, lookup)
            worst_idx_map.update(extra)
            # 合併後重存
            all_ids  = list(worst_idx_map.keys())
            all_widx = [worst_idx_map[k] for k in all_ids]
            np.savez(cache_path, ids=all_ids, worst_idx=all_widx)
            print(f'  [快取] 已更新 → {cache_path}')
    else:
        model = load_clap_model()
        worst_idx_map = compute_worst_consensus_indices(model, clip_ids, lookup)
        # 存快取
        all_ids  = list(worst_idx_map.keys())
        all_widx = [worst_idx_map[k] for k in all_ids]
        np.savez(cache_path, ids=all_ids, worst_idx=all_widx)
        print(f'[快取] 儲存 → {cache_path}')

    # ── 生成 TSV ────────────────────────────────────────
    rows_out   = []
    n_replaced = 0
    n_missing  = 0

    for row in rows_in:
        cid  = row['id']
        caps = lookup.get(cid)
        if caps is None:
            n_missing += 1
            rows_out.append(row)
            continue
        widx = worst_idx_map.get(cid, 0)
        new_cap = caps[widx]
        new_row = dict(row)
        new_row['caption'] = new_cap
        rows_out.append(new_row)
        n_replaced += 1

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'\n完成！')
    print(f'  worst-consensus 替換：{n_replaced:,}')
    print(f'  找不到 JSONL 對應：   {n_missing:,}')
    print(f'  輸出 TSV → {OUTPUT_TSV}')


if __name__ == '__main__':
    main()
