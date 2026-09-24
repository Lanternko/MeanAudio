"""
Phase 9.5 前置：用 Qwen2.5-Omni-3B 對 251K clips 生成 5 個 comprehensive caption
（鏡像 LP-MusicCaps 的 4 task + 1 variant 設計）

設計決策：
  - 5 個 prompt 都要求綜合描述（含樂器/情緒/節奏/風格），不是 aspect 切片
  - Diversity 來自 task-framing（Writing / Summary / Paraphrase / Attribute / NaturalProse）
    + LLM sampling（temperature=0.8）
  - 這樣 mean_similarity 保留「跨任務的 captioning confidence」語義，可作為 q 信號
  - 每個 prompt slot 獨立跑一次 pass，支援斷點續跑
  - 最終 merge 成每行 {"id": "...", "captions": [cap0, cap1, cap2, cap3, cap4]}

用法：
  # Slot 0–4 分別跑（或串接跑完全部）
  tmux new-session -d -s qwen_omni \\
    "cd ~/research/meanaudio_training && \\
     source ~/venvs/dac/bin/activate && \\
     export CUDA_VISIBLE_DEVICES=0 && \\
     python gen_qwen25omni_captions.py --slot all 2>&1 | tee ~/logs/phase9_omni_captions.log"

  # 只跑單一 slot（方便個別 resume）
  python gen_qwen25omni_captions.py --slot 0
  python gen_qwen25omni_captions.py --slot 1 --resume

  # 全部跑完後 merge
  python gen_qwen25omni_captions.py --merge

輸出：
  /mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions_slot{N}.jsonl  （每個 slot）
  /mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl          （merge 後）
  每行：{"id": "...", "captions": ["cap0", "cap1", "cap2", "cap3", "cap4"]}
"""

import os
import json
import csv
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import torch
import librosa
import numpy as np

INPUT_TSV    = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
AUDIO_ROOT   = Path('/mnt/HDD/hsiehyian/segments_no_vocals')  # overridable via --audio_root
OUTPUT_DIR   = Path('/mnt/HDD/kojiek/phase4_jamendo_data')
MODEL_ID       = 'Qwen/Qwen2.5-Omni-3B'
BATCH_SIZE     = 20    # 降至 20（~20GB）以讓出空間給共用 GPU 上 hsiehyian 訓練（7.7GB）；原 32→28GB 會 OOM
MAX_NEW_TOKENS = 60    # 實測 7,601 captions: p99=46 tokens, max=64, > 60 僅佔 0.01%；cap 長尾 batch
SR           = 16000

# 5 個 prompt 鏡像 LP-MusicCaps 的 4 task + 1 variant 設計：
# 每個 caption 都是 comprehensive 描述（含樂器 + 情緒 + 節奏 + 風格），
# diversity 來自 task-framing（Writing / Summary / Paraphrase / Attribute / NaturalProse），
# 不是來自 aspect 切片。這樣 mean_similarity 可作為 caption-confidence q 信號。
PROMPTS = [
    # Slot 0 — Writing：詳細自然描述句
    "Write a detailed one-sentence caption describing this music, covering the main instruments, mood, tempo, and genre.",
    # Slot 1 — Summary：壓縮為短句
    "Summarize this music in one concise sentence that captures its main instruments, mood, and style.",
    # Slot 2 — Paraphrase：豐富詞彙改寫
    "Describe this music in one sentence using rich and varied vocabulary, avoiding common generic words.",
    # Slot 3 — Attribute Prediction：以屬性為主的描述
    "In one flowing sentence, list the key musical attributes of this piece: genre, mood, instruments, tempo, and production style.",
    # Slot 4 — Natural Prose variant：中性自然敘述
    "In natural prose, describe in one sentence what you hear in this music, including the instruments and overall feel.",
]


def id_to_audio_path(clip_id: str) -> Path:
    """94_1317594_segment_0 → AUDIO_ROOT/94/1317594/segment_0.mp3"""
    parts = clip_id.split('_')
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
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get('caption') is not None:
                        done.add(d['id'])
                except json.JSONDecodeError:
                    continue
    return done


def first_sentence(s: str) -> str:
    """取第一句（到第一個句號為止），避免 model 幻覺出後續 Q&A。"""
    idx = s.find('.')
    return (s[:idx + 1] if idx != -1 else s).strip()


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
    print(f'載入 {MODEL_ID}...')
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="sdpa",   # Blackwell + PyTorch dev build 下 FA2 不可用，SDPA 是最佳選擇
        device_map={'': 0},
    )
    model.eval()
    print('模型載入完成')
    return model, processor


def run_slot(slot: int, resume: bool, limit: int = None):
    assert 0 <= slot <= 4, f'slot 必須是 0–4，得到 {slot}'
    prompt = PROMPTS[slot]
    suffix = f'_sanity{limit}' if limit else ''
    out_path = OUTPUT_DIR / f'phase9_omni_captions_slot{slot}{suffix}.jsonl'

    print(f'\n=== Slot {slot} ===')
    print(f'Prompt: {prompt}')
    print(f'輸出: {out_path}')

    # 讀 TSV
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    print(f'共 {len(ids):,} 筆')

    # resume
    done = set()
    if resume and out_path.exists():
        done = load_done(out_path)
        print(f'已完成 {len(done):,} 筆，繼續...')

    todo = [cid for cid in ids if cid not in done]
    if limit:
        todo = todo[:limit]
        print(f'⚠️ sanity check 模式：只跑前 {limit} 個 clip')
    if not todo:
        print('全部已完成，跳過')
        return

    model, processor = load_model()

    n_ok = n_error = 0
    _profile_iters = 0   # [TIMING] 只 profile 前 10 個 iter
    with open(out_path, 'a') as fout:
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc=f'slot{slot}'):
            batch_ids = todo[i: i + BATCH_SIZE]
            _t0 = time.perf_counter()   # [TIMING]

            # 讀音訊（ThreadPoolExecutor 平行 decode，librosa/soundfile/ffmpeg 釋放 GIL）
            def _load_one(cid):
                try:
                    audio, _ = librosa.load(str(id_to_audio_path(cid)), sr=SR, mono=True)
                    return cid, audio, None
                except Exception as e:
                    return cid, None, str(e)

            with ThreadPoolExecutor(max_workers=12) as ex:
                results = list(ex.map(_load_one, batch_ids))
            _t1 = time.perf_counter()   # [TIMING]

            audios, valid_ids = [], []
            for cid, audio, err in results:
                if err is not None:
                    tqdm.write(f'[WARN] load error {cid}: {err}')
                    fout.write(json.dumps({'id': cid, 'caption': None}) + '\n')
                    n_error += 1
                else:
                    audios.append(audio)
                    valid_ids.append(cid)

            if not valid_ids:
                fout.flush()
                continue

            # 建 conversation（Qwen2.5-Omni chat format）
            conversations = [
                [{"role": "user", "content": [
                    {"type": "audio", "audio": str(id_to_audio_path(cid))},
                    {"type": "text",  "text": prompt},
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
                        audio=audios,
                        return_tensors='pt',
                        padding=True,
                        sampling_rate=SR,
                    ).to(model.device)
                    torch.cuda.synchronize()   # [TIMING]
                    _t2 = time.perf_counter()   # [TIMING]

                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.8,
                    )
                    torch.cuda.synchronize()   # [TIMING]
                    _t3 = time.perf_counter()   # [TIMING]
                    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
                    captions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    _t4 = time.perf_counter()   # [TIMING]

                for cid, cap in zip(valid_ids, captions):
                    fout.write(json.dumps({'id': cid, 'caption': first_sentence(cap)}) + '\n')
                    n_ok += 1

                # [TIMING] 前 10 iter 報時
                if _profile_iters < 10:
                    tqdm.write(f'[TIMING iter={_profile_iters}] '
                               f'load {_t1-_t0:.2f}s  proc {_t2-_t1:.2f}s  '
                               f'gen {_t3-_t2:.2f}s  dec {_t4-_t3:.2f}s  '
                               f'total {_t4-_t0:.2f}s')
                    _profile_iters += 1

            except Exception as e:
                tqdm.write(f'[WARN] batch error: {e}')
                for cid in valid_ids:
                    fout.write(json.dumps({'id': cid, 'caption': None}) + '\n')
                    n_error += 1

            fout.flush()   # 每個 batch 落盤，kill 時不丟資料

    print(f'Slot {slot} 完成：ok={n_ok:,}  error={n_error:,}')


def merge_slots():
    """合併 5 個 slot JSONL → 每行 {"id": "...", "captions": [...]}"""
    print('合併 5 個 slot...')

    # 每個 slot 讀進 id → caption
    slot_maps = []
    for slot in range(5):
        path = OUTPUT_DIR / f'phase9_omni_captions_slot{slot}.jsonl'
        if not path.exists():
            print(f'[ERROR] {path} 不存在，請先跑完 slot {slot}')
            return
        m = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    m[d['id']] = d.get('caption')
                except json.JSONDecodeError:
                    continue
        slot_maps.append(m)
        print(f'  slot {slot}: {len(m):,} 筆')

    # 以 TSV 順序為準
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))

    out_path = OUTPUT_DIR / 'phase9_omni_captions.jsonl'
    n_complete = n_partial = 0
    with open(out_path, 'w') as fout:
        for row in tqdm(rows, desc='merge'):
            cid = row['id']
            captions = [slot_maps[s].get(cid) for s in range(5)]
            n_none = captions.count(None)
            if n_none == 0:
                n_complete += 1
            else:
                n_partial += 1
            fout.write(json.dumps({'id': cid, 'captions': captions}) + '\n')

    print(f'\n完成！')
    print(f'  全部 5 個 caption：{n_complete:,} 筆')
    print(f'  部分 None：{n_partial:,} 筆')
    print(f'  輸出 → {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--slot', default='all',
                   help='0–4 跑單一 slot，"all" 跑全部，"merge" 只做合併')
    p.add_argument('--resume', action='store_true', help='跳過已完成的 clip')
    p.add_argument('--limit', type=int, default=None,
                   help='只跑前 N 個 clip（sanity check 用，輸出檔名加 _sanityN 後綴）')
    p.add_argument('--audio_root', type=str, default=None,
                   help='audio 根目錄（預設：/mnt/HDD/hsiehyian/segments_no_vocals）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.audio_root is not None:
        AUDIO_ROOT = Path(args.audio_root)
        print(f'[INFO] --audio_root override: {AUDIO_ROOT}')

    if args.slot == 'merge':
        merge_slots()
    elif args.slot == 'all':
        for s in range(5):
            run_slot(s, resume=args.resume, limit=args.limit)
        if args.limit is None:
            merge_slots()
    else:
        run_slot(int(args.slot), resume=args.resume, limit=args.limit)
