"""
生成 Phase 7 V2 訓練 TSV
對每個 clip 的 5 個候選 caption，計算 caption-audio CLAP 相似度，取最高分的 caption。

用法：
  python3 gen_phase7_v2_tsv.py [--limit N] [--cache_path PATH]

  --limit N       只處理前 N 筆（測試用）
  --cache_path    CLAP 分數快取路徑（預設 ~/research/meanaudio_training/clap_scores_p7v2.npz）
"""

import json
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

JSONL_PATH    = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
INPUT_TSV     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV    = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v2_train.tsv')
AUDIO_ROOT    = Path('/home/kojiek/data/segments_no_vocals')
CLAP_CKPT     = Path('/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt')
DEFAULT_CACHE = Path.home() / 'research/meanaudio_training/clap_scores_p7v2.npz'

CLAP_BATCH = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cache_path', type=str, default=str(DEFAULT_CACHE))
    return parser.parse_args()


def relative_path_to_id(relative_path: str) -> str:
    """00/1002000/segment_0.mp3 → 00_1002000_segment_0"""
    return relative_path.replace('.mp3', '').replace('/', '_')


def id_to_audio_path(clip_id: str) -> Path:
    """00_1002000_segment_0 → /home/kojiek/data/segments_no_vocals/00/1002000/segment_0.mp3"""
    parts = clip_id.split('_')
    prefix   = parts[0]
    track_id = parts[1]
    seg_name = '_'.join(parts[2:]) + '.mp3'
    return AUDIO_ROOT / prefix / track_id / seg_name


def load_jsonl(jsonl_path, limit=None):
    """id → [caption, ...] 的 lookup dict"""
    lookup = {}
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            clip_id = relative_path_to_id(d['relative_path'])
            captions = [c['caption'].replace('\n', ' ').replace('\r', ' ').strip()
                        for c in d['caption_details']]
            lookup[clip_id] = captions
    return lookup


def load_clap_model():
    import torch
    import laion_clap
    print('[CLAP] 載入模型...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model


def compute_clap_scores(model, clip_ids, lookup):
    """
    對每個 clip_id 的所有候選 caption 計算 audio-caption CLAP 相似度。
    回傳 {clip_id: [score_0, score_1, ...]}
    """
    import torch
    import torch.nn.functional as F

    scores = {}
    missing_audio = 0

    for clip_id in tqdm(clip_ids, desc='CLAP 相似度計算'):
        audio_path = id_to_audio_path(clip_id)
        if not audio_path.exists():
            missing_audio += 1
            scores[clip_id] = None
            continue

        captions = lookup.get(clip_id)
        if not captions:
            scores[clip_id] = None
            continue

        try:
            audio_embed = model.get_audio_embedding_from_filelist(
                [str(audio_path)], use_tensor=True
            )
            text_embeds = model.get_text_embedding(captions, use_tensor=True)
            # audio_embed: (1, D), text_embeds: (N, D)
            sims = F.cosine_similarity(
                audio_embed.expand(len(captions), -1), text_embeds, dim=-1
            )
            scores[clip_id] = sims.tolist()
        except Exception as e:
            scores[clip_id] = None

    print(f'[CLAP] 完成：{len(scores) - missing_audio} 筆，音訊缺失 {missing_audio} 筆')
    return scores


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)

    # 讀取 JSONL lookup（全量，不受 --limit 影響）
    print(f'讀取 JSONL: {JSONL_PATH}')
    lookup = load_jsonl(JSONL_PATH)
    print(f'  載入 {len(lookup):,} 筆 caption lookup')

    # 讀取訓練 TSV
    print(f'讀取訓練 TSV: {INPUT_TSV}')
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]
    print(f'  載入 {len(rows):,} 筆')

    clip_ids = [r['id'] for r in rows]

    # 計算或載入 CLAP 分數快取
    if cache_path.exists():
        print(f'[CLAP] 載入快取：{cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        clap_scores = cached['scores'].item()
    else:
        model = load_clap_model()
        clap_scores = compute_clap_scores(model, clip_ids, lookup)
        np.savez(cache_path, scores=clap_scores)
        print(f'[CLAP] 快取儲存至：{cache_path}')

    # 生成 TSV：每個 clip 取最高 CLAP 分的 caption
    rows_out = []
    n_best   = 0
    n_random = 0

    for row in rows:
        clip_id  = row['id']
        captions = lookup.get(clip_id)
        scores   = clap_scores.get(clip_id)

        if captions and scores and len(captions) == len(scores):
            best_idx     = int(np.argmax(scores))
            row['caption'] = captions[best_idx]
            n_best += 1
        else:
            # 找不到音訊或 caption 不匹配，保留原本 caption
            n_random += 1

        rows_out.append(row)

    print(f'\n統計：')
    print(f'  總筆數：         {len(rows_out):,}')
    print(f'  CLAP best 替換： {n_best:,}')
    print(f'  fallback 保留：  {n_random:,}')

    # 前 3 筆預覽
    print('\n前 3 筆預覽：')
    for r in rows_out[:3]:
        cid = r['id']
        sc  = clap_scores.get(cid)
        print(f"  id={cid}  q_level={r.get('q_level', '-')}")
        if sc:
            print(f"  scores={[f'{s:.3f}' for s in sc]}")
        print(f"  caption={r['caption'][:100]}...")
        print()

    # 寫出
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'輸出：{OUTPUT_TSV}')
    in_lines  = sum(1 for _ in open(INPUT_TSV))
    out_lines = sum(1 for _ in open(OUTPUT_TSV))
    print(f'行數驗證：input={in_lines:,}  output={out_lines:,}  {"✅ 一致" if in_lines == out_lines else "❌ 不一致"}')


if __name__ == '__main__':
    main()
