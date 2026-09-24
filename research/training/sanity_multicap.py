"""
Multi-cap NPZ sanity check（50 clips）

驗三件事：
  1. NPZ shape 是否為 [5, 77, 1024] / [5, 512]
  2. 5 個 caption 是否有足夠多樣性（不全重複）
  3. ExtractedAudio(multi_cap=True) 多次 __getitem__ 是否真的抽到不同 caption

用法：
  source ~/venvs/dac/bin/activate
  export CUDA_VISIBLE_DEVICES=0
  python sanity_multicap.py
"""

import json, csv, sys, random
import numpy as np
import torch
from pathlib import Path
from transformers import T5EncoderModel, AutoTokenizer
import laion_clap

TSV_PATH   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
JSONL_PATH = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
SRC_NPZ    = Path.home() / 'research/meanaudio_training/npz'
OUT_NPZ    = Path('/tmp/sanity_multicap_npz')
CLAP_CKPT  = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
N_CLIPS    = 50

OUT_NPZ.mkdir(parents=True, exist_ok=True)

# ── Step 0: 讀 TSV + JSONL ────────────────────────────────────────────────
print('Step 0: 讀資料...')
with open(TSV_PATH) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))[:N_CLIPS]
ids = [r['id'] for r in rows]

def rel_to_id(rel):
    return rel.replace('.mp3','').replace('/','_')

lookup = {}
with open(JSONL_PATH) as f:
    for line in f:
        d = json.loads(line)
        cid = rel_to_id(d['relative_path'])
        if cid in ids:
            lookup[cid] = [c['caption'].replace('\n',' ').strip()
                           for c in d['caption_details']]

found = sum(1 for i in ids if i in lookup)
print(f'  TSV clips: {len(ids)}, JSONL 找到: {found}/{len(ids)}')
assert found > 40, f'JSONL 比對率太低：{found}/50'

# ── Step 1: 載入模型，encode 50 clips × 5 captions ────────────────────────
print('\nStep 1: 載入 T5 + CLAP...')
t5_tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
t5_mod = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().cuda()
clap   = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval()
clap.load_ckpt(str(CLAP_CKPT), verbose=False)
print('  模型載入完成')

print('\nStep 1b: encode captions & 寫 NPZ...')
for idx, row in enumerate(rows):
    cid = row['id']
    caps = lookup.get(cid)
    if caps is None:
        caps = [row['caption']] * 5
    while len(caps) < 5:
        caps.append(caps[-1])
    caps = caps[:5]

    toks = t5_tok(caps, max_length=77, padding='max_length',
                  truncation=True, return_tensors='pt')
    with torch.no_grad():
        tf  = t5_mod(input_ids=toks.input_ids.cuda(),
                     attention_mask=toks.attention_mask.cuda())[0]  # [5,77,1024]
        tfc = clap.get_text_embedding(caps, use_tensor=True)         # [5,512]

    src = np.load(f'{SRC_NPZ}/{idx}.npz')
    np.savez(f'{OUT_NPZ}/{idx}.npz',
             mean=src['mean'], std=src['std'],
             text_features=tf.cpu().numpy(),
             text_features_c=tfc.cpu().numpy(),
             text_attention_mask=toks.attention_mask.cpu().bool().numpy())

print(f'  {N_CLIPS} 個 NPZ 寫入 {OUT_NPZ}')

# ── Step 2: 驗 NPZ shape ─────────────────────────────────────────────────
print('\nStep 2: 驗 NPZ shape...')
ok_shape = 0
for i in range(N_CLIPS):
    npz = np.load(f'{OUT_NPZ}/{i}.npz')
    tf_shape  = npz['text_features'].shape    # 預期 (5, 77, 1024)
    tfc_shape = npz['text_features_c'].shape  # 預期 (5, 512)
    assert tf_shape  == (5, 77, 1024), f'idx {i}: text_features shape {tf_shape}'
    assert tfc_shape == (5, 512),      f'idx {i}: text_features_c shape {tfc_shape}'
    ok_shape += 1
print(f'  ✅ 全部 {ok_shape} 個 NPZ shape 正確 (5,77,1024) / (5,512)')

# ── Step 3: 驗 caption 多樣性 ─────────────────────────────────────────────
print('\nStep 3: 驗 caption 多樣性...')
from sklearn.metrics.pairwise import cosine_similarity

all_unique_rates = []
all_pairwise_sims = []
for i in range(min(20, N_CLIPS)):
    cid = ids[i]
    caps = lookup.get(cid)
    if caps is None:
        continue
    caps = caps[:5]
    unique_rate = len(set(caps)) / len(caps)
    all_unique_rates.append(unique_rate)

    # pairwise CLAP sim
    npz = np.load(f'{OUT_NPZ}/{i}.npz')
    tfc = npz['text_features_c']  # (5, 512)
    sims = []
    for a in range(5):
        for b in range(a+1, 5):
            sim = cosine_similarity(tfc[a:a+1], tfc[b:b+1])[0][0]
            sims.append(sim)
    all_pairwise_sims.append(np.mean(sims))

    if i < 3:
        print(f'\n  clip {cid}:')
        for j, cap in enumerate(caps):
            print(f'    [{j}] {cap[:90]}')

avg_unique = np.mean(all_unique_rates)
avg_sim    = np.mean(all_pairwise_sims)
print(f'\n  平均 unique rate: {avg_unique:.3f}  (希望 > 0.9)')
print(f'  平均 pairwise CLAP sim: {avg_sim:.3f}  (希望 < 0.95)')
assert avg_unique > 0.8, f'unique rate 太低：{avg_unique:.3f}'
assert avg_sim    < 0.98, f'caption 幾乎完全相同：{avg_sim:.3f}'
print('  ✅ 多樣性驗證通過')

# ── Step 4: 驗 ExtractedAudio multi_cap 隨機性 ────────────────────────────
print('\nStep 4: 驗 ExtractedAudio(multi_cap=True) 隨機性...')
sys.path.insert(0, str(Path.home() / 'MeanAudio'))
from meanaudio.data.extracted_audio import ExtractedAudio

dataset = ExtractedAudio(
    tsv_path=TSV_PATH,
    concat_text_fc=False,
    npz_dir=OUT_NPZ,
    data_dim={'latent_seq_len': 312, 'text_seq_len': 77,
              'text_dim': 1024, 'text_c_dim': 512},
    repa_npz_dir=None,
    exclude_cls=False,
    repa_version=1,
    multi_cap=True,
)

# 同一 idx 取 20 次，收集 caption embedding，確認有差異
idx_to_test = 0
seen_tfc = []
for _ in range(20):
    item = dataset[idx_to_test]
    seen_tfc.append(item['text_features_c'].numpy())

seen_tfc = np.stack(seen_tfc)
# 若全部相同，std 幾乎為 0
std_across_draws = seen_tfc.std(axis=0).mean()
n_unique_draws   = len(set(tuple(v.tolist()) for v in seen_tfc))

print(f'  idx=0，20 次抽樣：{n_unique_draws} 個不同 caption embedding')
print(f'  std across draws: {std_across_draws:.6f}')
assert n_unique_draws > 1, '20 次都抽到同一個 caption！multi_cap 隨機性壞了'
print('  ✅ __getitem__ 有在隨機抽不同 caption')

# ── 最終報告 ──────────────────────────────────────────────────────────────
print('\n' + '='*50)
print('✅ Sanity check 全部通過！')
print(f'  NPZ shape:       (5, 77, 1024) / (5, 512) ✓')
print(f'  Caption 唯一率:  {avg_unique:.3f} ✓')
print(f'  Pairwise sim:    {avg_sim:.3f} ✓')
print(f'  Random draws:    {n_unique_draws}/20 unique ✓')
print('\n下一步：執行全量 gen_multicap_npz.py（251K clips，~2-3h）')
