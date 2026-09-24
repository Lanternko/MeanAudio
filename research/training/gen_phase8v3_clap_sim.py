"""
Phase 8 V3 前置：計算 audio-text CLAP similarity

輸入：
  - phase7_v1_train.tsv（caption 欄位）
  - /home/hsiehyian/dataset/segments_no_vocals/{artist_id}/{track_id}/segment_N.mp3

輸出：
  - /mnt/HDD/kojiek/phase4_jamendo_data/phase8_v3_clap_sim.jsonl
    每行：{"id": "...", "clap_sim": 0.xx}

CLAP 模型與 eval 時完全相同：
  laion_clap, HTSAT-base, music_speech_audioset_epoch_15_esc_89.98.pt

用法：
  python gen_phase8v3_clap_sim.py [--resume]

  --resume：跳過已寫入 JSONL 的 id，支援斷點續跑
"""

import os
import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import laion_clap

CLAP_CKPT   = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
INPUT_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v3_clap_sim.jsonl')
AUDIO_ROOT  = Path('/home/hsiehyian/dataset/segments_no_vocals')
BATCH_SIZE  = 64   # audio + text 各一批，調小可降 VRAM 壓力


def id_to_audio_path(clip_id: str) -> Path:
    """94_1317594_segment_0 → AUDIO_ROOT/94/1317594/segment_0.mp3"""
    parts = clip_id.split('_')
    # 格式：{artist}_{track}_segment_{n}
    # artist 可能有多位，track 也可能有多位 → 找 'segment' 關鍵字
    seg_idx = parts.index('segment')
    artist  = '_'.join(parts[:seg_idx - 1])
    track   = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f'segment_{seg_num}.mp3'


def load_done(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                done.add(d['id'])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='跳過已完成的 id（斷點續跑）')
    args = parser.parse_args()

    # ── 讀 TSV ───────────────────────────────────────────────
    rows = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append((row['id'], row['caption']))
    print(f'TSV 共 {len(rows):,} 筆')

    # ── 斷點續跑 ──────────────────────────────────────────────
    done = set()
    if args.resume:
        done = load_done(OUTPUT_JSONL)
        print(f'已完成 {len(done):,} 筆，跳過')
    rows = [(cid, cap) for cid, cap in rows if cid not in done]
    print(f'待處理 {len(rows):,} 筆')

    # ── 載入 CLAP ─────────────────────────────────────────────
    print('\n載入 CLAP 模型...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(CLAP_CKPT)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'CLAP 載入完成，device={device}')

    # ── 計算 ─────────────────────────────────────────────────
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = n_error = 0

    with open(OUTPUT_JSONL, 'a') as fout:
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc='CLAP sim'):
            batch = rows[i : i + BATCH_SIZE]
            paths   = [str(id_to_audio_path(cid)) for cid, _ in batch]
            caps    = [cap for _, cap in batch]
            ids     = [cid for cid, _ in batch]

            # 過濾不存在的音訊
            valid = [(cid, p, cap)
                     for cid, p, cap in zip(ids, paths, caps)
                     if os.path.exists(p)]
            missing_ids = [cid for cid, p, _ in zip(ids, paths, caps)
                           if not os.path.exists(p)]

            for cid in missing_ids:
                fout.write(json.dumps({'id': cid, 'clap_sim': None}) + '\n')
                n_missing += 1

            if not valid:
                continue

            v_ids   = [x[0] for x in valid]
            v_paths = [x[1] for x in valid]
            v_caps  = [x[2] for x in valid]

            try:
                with torch.no_grad():
                    audio_emb = model.get_audio_embedding_from_filelist(
                        v_paths, use_tensor=True
                    )
                    text_emb = model.get_text_embedding(v_caps, use_tensor=True)
                    sims = torch.nn.functional.cosine_similarity(
                        audio_emb, text_emb, dim=-1
                    )

                for cid, sim in zip(v_ids, sims.tolist()):
                    fout.write(json.dumps({'id': cid, 'clap_sim': round(sim, 6)}) + '\n')
                    n_ok += 1

            except Exception as e:
                # 整批失敗時逐筆寫 None
                for cid in v_ids:
                    fout.write(json.dumps({'id': cid, 'clap_sim': None}) + '\n')
                    n_error += 1
                tqdm.write(f'[WARN] batch error: {e}')

    print(f'\n完成！ok={n_ok:,}  missing={n_missing:,}  error={n_error:,}')
    print(f'輸出 → {OUTPUT_JSONL}')


if __name__ == '__main__':
    main()
