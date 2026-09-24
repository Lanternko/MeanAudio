"""
Phase 8 V4 caption sanity check — 50 clips，確認無 collapse
"""
import csv, json, torch, librosa
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

AUDIO_ROOT = Path('/home/hsiehyian/dataset/segments_no_vocals')
MODEL_ID   = 'Qwen/Qwen2-Audio-7B-Instruct'
PROMPT     = 'Describe this music in one sentence. Include: instruments, tempo, mood, genre, and recording quality.'
OUT        = Path('/tmp/phase8_v4_captions_test50.jsonl')
N          = 50
BATCH      = 4


def id_to_audio_path(clip_id):
    parts = clip_id.split('_')
    seg_idx = parts.index('segment')
    artist  = '_'.join(parts[:seg_idx - 1])
    track   = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f'segment_{seg_num}.mp3'


# 抽 50 個不同 track 的 clip
test_ids, seen_tracks = [], set()
with open('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        cid = row['id']
        parts = cid.split('_')
        seg_idx = parts.index('segment')
        track = parts[seg_idx - 1]
        if track not in seen_tracks and id_to_audio_path(cid).exists():
            test_ids.append(cid)
            seen_tracks.add(track)
        if len(test_ids) == N:
            break
print(f'抽到 {len(test_ids)} 個不同 track 的 clip')

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map='auto'
)
model.eval()
sr = processor.feature_extractor.sampling_rate

results = []
for i in tqdm(range(0, len(test_ids), BATCH), desc='caption'):
    batch_ids = test_ids[i:i+BATCH]
    audios, valid_ids = [], []
    for cid in batch_ids:
        try:
            wav, _ = librosa.load(str(id_to_audio_path(cid)), sr=sr, mono=True)
            audios.append(wav)
            valid_ids.append(cid)
        except Exception as e:
            print(f'[WARN] {cid}: {e}')

    if not valid_ids:
        continue

    conversations = [
        [{'role': 'user', 'content': [
            {'type': 'audio', 'audio_url': str(id_to_audio_path(cid))},
            {'type': 'text',  'text': PROMPT},
        ]}]
        for cid in valid_ids
    ]
    texts = [
        processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
        for c in conversations
    ]

    with torch.no_grad():
        inputs = processor(
            text=texts, audio=audios, sampling_rate=sr,
            return_tensors='pt', padding=True
        ).to(model.device)
        gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen_ids = gen_ids[:, inputs.input_ids.size(1):]
        captions = processor.batch_decode(gen_ids, skip_special_tokens=True)

    for cid, cap in zip(valid_ids, captions):
        results.append({'id': cid, 'caption': cap.strip()})

# 寫出結果
with open(OUT, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

# Sanity check
caps = [r['caption'] for r in results]
unique = len(set(caps))
print('\n=== Sanity Check ===')
print(f'總筆數:    {len(caps)}')
print(f'唯一 caption: {unique}')
print(f'重複數:    {len(caps) - unique}')
print(f'Collapse: {"YES ❌ 有問題！" if unique == 1 else "NO ✅ 正常"}')
print('\n前 5 筆 caption:')
for r in results[:5]:
    print(f'  [{r["id"]}] {r["caption"][:100]}')
