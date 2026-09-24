"""
Phase 8 V4 前置：用 Qwen2-Audio-7B-Instruct 對 251K clip 生成 caption

設計決策：
  - Model：Qwen/Qwen2-Audio-7B-Instruct（chat format，prompt 可控）
  - Prompt：對齊 LP-MusicCaps 風格（樂器、tempo、情緒、genre、錄音品質）
  - Clip：30 秒 MP3，不超過 Whisper encoder 上限，不需截斷
  - Resume：支援斷點續跑（--resume）

輸出：
  /mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_captions.jsonl
  每行：{"id": "...", "caption": "..."}

用法：
  tmux new-session -d -s qwen_captions \\
    "cd ~/research/meanaudio_training && \\
     source ~/venvs/dac/bin/activate && \\
     export CUDA_VISIBLE_DEVICES=0 && \\
     python gen_qwen2audio_captions.py 2>&1 | tee ~/logs/phase8_v4_captions.log"
"""

import os
import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

INPUT_TSV     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_JSONL  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_captions.jsonl')
AUDIO_ROOT    = Path('/home/hsiehyian/dataset/segments_no_vocals')
MODEL_ID      = 'Qwen/Qwen2-Audio-7B-Instruct'
BATCH_SIZE    = 4    # 30s clips at FP16；5090 VRAM 夠，可調至 8
MAX_NEW_TOKENS = 128

PROMPT = (
    "Describe this music in one sentence. "
    "Include: instruments, tempo, mood, genre, and recording quality."
)


def id_to_audio_path(clip_id: str) -> Path:
    """94_1317594_segment_0 → AUDIO_ROOT/94/1317594/segment_0.mp3"""
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist  = '_'.join(parts[:seg_idx - 1])
    track   = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f'segment_{seg_num}.mp3'


def load_done(jsonl_path: Path) -> set:
    """只把有效 caption（非 null）的 id 視為完成，null caption 下次 resume 會重試。
    容忍不完整尾行（程序被 kill 時最後一行可能是殘缺 JSON）。"""
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get('caption') is not None:
                        done.add(d['id'])
                except json.JSONDecodeError:
                    print(f'[WARN] load_done: 第 {lineno} 行 JSON 損壞，跳過（可能是上次被中斷的尾行）')
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='跳過已完成的 id')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # ── 讀 TSV ───────────────────────────────────────────────
    ids = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ids.append(row['id'])
    print(f'TSV 共 {len(ids):,} 筆')

    # ── 斷點續跑 ──────────────────────────────────────────────
    if args.resume:
        done = load_done(OUTPUT_JSONL)
        ids = [i for i in ids if i not in done]
        print(f'已完成 {len(done):,} 筆，待處理 {len(ids):,} 筆')
    else:
        print(f'待處理 {len(ids):,} 筆')

    # ── 載入模型 ──────────────────────────────────────────────
    print(f'\n載入 {MODEL_ID}...')
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    sr = processor.feature_extractor.sampling_rate  # 16000
    print(f'模型載入完成，target sr={sr}')

    # ── 推理 ─────────────────────────────────────────────────
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = n_error = 0

    with open(OUTPUT_JSONL, 'a') as fout:
        for i in tqdm(range(0, len(ids), args.batch_size), desc='Qwen2-Audio caption'):
            batch_ids = ids[i : i + args.batch_size]

            # 載入音訊，過濾不存在的
            audios, valid_ids = [], []
            for cid in batch_ids:
                p = id_to_audio_path(cid)
                if not p.exists():
                    fout.write(json.dumps({'id': cid, 'caption': None}) + '\n')
                    n_missing += 1
                    continue
                try:
                    wav, _ = librosa.load(str(p), sr=sr, mono=True)
                    audios.append(wav)
                    valid_ids.append(cid)
                except Exception as e:
                    tqdm.write(f'[WARN] load error {cid}: {e}')
                    fout.write(json.dumps({'id': cid, 'caption': None}) + '\n')
                    n_error += 1

            if not valid_ids:
                continue

            # 建 chat conversation（Instruct 格式）
            # audio_url 必須是真實存在的檔案路徑，否則 processor 無法插入 audio token
            conversations = [
                [{"role": "user", "content": [
                    {"type": "audio", "audio_url": str(id_to_audio_path(cid))},
                    {"type": "text",  "text": PROMPT},
                ]}]
                for cid in valid_ids
            ]

            try:
                texts = [
                    processor.apply_chat_template(
                        conv, add_generation_prompt=True, tokenize=False
                    )
                    for conv in conversations
                ]

                with torch.no_grad():
                    inputs = processor(
                        text=texts,
                        audio=audios,   # NOTE: 'audio' not 'audios' (transformers 4.57+)
                        sampling_rate=sr,
                        return_tensors='pt',
                        padding=True,
                    ).to(model.device)

                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                    # 只保留新生成的 token
                    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                    captions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                for cid, cap in zip(valid_ids, captions):
                    fout.write(json.dumps({'id': cid, 'caption': cap.strip()}) + '\n')
                    n_ok += 1

                # Sanity check：每 500 batch 抽查 caption 多樣性，提早發現 collapse
                if (i // args.batch_size) % 500 == 0 and n_ok > 0:
                    unique_rate = len(set(captions)) / max(len(captions), 1)
                    if unique_rate < 0.5:
                        tqdm.write(f'[WARN] Caption collapse 疑似發生！batch 內唯一率={unique_rate:.2f}，範例：{captions[0][:80]}')

            except Exception as e:
                tqdm.write(f'[WARN] batch error: {e}')
                for cid in valid_ids:
                    fout.write(json.dumps({'id': cid, 'caption': None}) + '\n')
                    n_error += 1

    print(f'\n完成！ok={n_ok:,}  missing={n_missing:,}  error={n_error:,}')
    print(f'輸出 → {OUTPUT_JSONL}')
    print('\n下一步：python gen_qwen2audio_tsv.py')


if __name__ == '__main__':
    main()
