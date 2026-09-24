"""
B2 + B3 + B4: Pipeline integrity diagnostics (CPU only)
========================================================

B2: NPZ caption verify
  - 從 ~/phase9_5_bc_singlecap_npz/ 抽 20 個 NPZ
  - 抽 NPZ.text_features_c (CLAP 512)
  - 對應 id 的 TSV caption fresh forward CLAP
  - cosine 應 > 0.999；< 0.99 = NPZ cache 拿到錯 caption

B3: TSV id-caption alignment
  - 從 qwen_singlecap_bc_train.tsv 與 phase9_5_train.tsv 各抽 30 row
  - 每 row 的 caption 必須是 phase9_omni_captions.jsonl 該 id 的 5 條 candidate 之一
  - 不在 → TSV 生成腳本錯位

B4: T5 token truncation
  - MeanAudio T5 (flan-t5-large) max_length=77, truncation=True
  - 讀 LP-MC + Qwen 全部 captions（251K），用 T5 tokenizer 計 token 數
  - 比較 length distribution 與 truncation rate

輸出：
  ~/research/meanaudio_training/diag_b234_pipeline.json
"""

import json
import csv
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

QWEN_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
LPMC_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
QWEN_SINGLECAP_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_bc_train.tsv')
QWEN_MULTICAP_TSV  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_5_train.tsv')

NPZ_DIR_QWEN_SINGLE = Path('/home/kojiek/phase9_5_bc_singlecap_npz')
NPZ_CACHE_TXT = Path('/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')

OUT_JSON   = Path('/home/kojiek/research/meanaudio_training/diag_b234_pipeline.json')
SEED       = 42


def load_qwen():
    d = {}
    with open(QWEN_JSONL) as f:
        for line in f:
            j = json.loads(line)
            d[j['id']] = j['captions']
    return d


def load_tsv(p):
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows


# ─── B3: TSV alignment ──────────────────────────────────────────────
def b3_tsv_align(qwen_data):
    print('\n=== B3: TSV id-caption alignment ===')
    out = {}
    rng = random.Random(SEED)
    for tsv_path in [QWEN_SINGLECAP_TSV, QWEN_MULTICAP_TSV]:
        rows = load_tsv(tsv_path)
        sample = rng.sample(rows, 30)
        n_match = 0
        n_total = 0
        mismatches = []
        for r in sample:
            cid = r['id']
            cap = r['caption']
            if cid not in qwen_data:
                continue
            n_total += 1
            if cap in qwen_data[cid]:
                n_match += 1
            else:
                mismatches.append({'id': cid, 'tsv_caption_head': cap[:80]})
        rate = n_match / n_total if n_total else 0
        out[tsv_path.name] = {
            'n_checked': n_total,
            'n_match_one_of_5': n_match,
            'match_rate': round(rate, 3),
            'mismatch_examples': mismatches[:3],
        }
        print(f'  {tsv_path.name}: {n_match}/{n_total} = {rate:.1%}')
        if mismatches:
            print(f'    mismatch ex: id={mismatches[0]["id"]}')
            print(f'      TSV cap: {mismatches[0]["tsv_caption_head"]}...')
    return out


# ─── B4: T5 token truncation ────────────────────────────────────────
def b4_token_trunc(qwen_data):
    print('\n=== B4: T5 token truncation (max_length=77) ===')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')

    out = {}
    rng = random.Random(SEED)

    # LP-MC sample
    lpmc_rows = load_tsv(LPMC_TSV)
    lpmc_sample = rng.sample(lpmc_rows, 5000)
    lpmc_caps = [r['caption'] for r in lpmc_sample]

    # Qwen sample (5 caps per audio, take all 5 → 5x)
    qwen_ids = list(qwen_data.keys())
    qwen_sample_ids = rng.sample(qwen_ids, 1000)
    qwen_caps_per_slot = [[qwen_data[i][s] for i in qwen_sample_ids] for s in range(5)]
    qwen_caps_all = [c for slot in qwen_caps_per_slot for c in slot]   # 5000

    def stat(name, caps):
        print(f'  tokenize {name} (n={len(caps)})...')
        toks = tok(caps, return_tensors=None, padding=False, truncation=False, add_special_tokens=True)
        lens = [len(t) for t in toks['input_ids']]
        a = np.array(lens)
        d = {
            'n':            int(len(a)),
            'mean':         round(float(a.mean()), 1),
            'median':       round(float(np.median(a)), 1),
            'p90':          round(float(np.percentile(a, 90)), 1),
            'p95':          round(float(np.percentile(a, 95)), 1),
            'p99':          round(float(np.percentile(a, 99)), 1),
            'max':          int(a.max()),
            'frac_gt_77':   round(float((a > 77).mean()), 4),
            'frac_gt_64':   round(float((a > 64).mean()), 4),
        }
        print(f'    mean={d["mean"]}  median={d["median"]}  p90={d["p90"]}  p95={d["p95"]}  '
              f'max={d["max"]}  frac>77={d["frac_gt_77"]:.2%}')
        return d

    out['lpmc']           = stat('LP-MC', lpmc_caps)
    out['qwen_per_slot']  = [stat(f'Qwen slot {s}', qwen_caps_per_slot[s]) for s in range(5)]
    out['qwen_all']       = stat('Qwen all 5', qwen_caps_all)
    return out


# ─── B2: NPZ caption verify ─────────────────────────────────────────
def b2_npz_verify(qwen_data):
    print('\n=== B2: NPZ caption verify (Qwen single-cap NPZ) ===')
    # Need: NPZ idx -> id mapping
    if not NPZ_CACHE_TXT.exists():
        print(f'  ⚠️  npz_cache_train.txt 缺，跳過 B2')
        return {'status': 'skipped — no npz_cache_train.txt'}

    # Load idx → id mapping (each line = filename of NPZ; index by line number? or path?)
    with open(NPZ_CACHE_TXT) as f:
        npz_files = [l.strip() for l in f]
    print(f'  npz_cache_train.txt 共 {len(npz_files):,} entries')
    print(f'  ex: {npz_files[:3]}')

    # singlecap NPZ uses TSV row order — load TSV
    tsv_rows = load_tsv(QWEN_SINGLECAP_TSV)
    print(f'  qwen_singlecap_bc_train.tsv = {len(tsv_rows):,} rows')

    # Pick 10 random rows
    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(tsv_rows)), 10)

    # Load CLAP for fresh forward
    print('  載入 CLAP for fresh embed...')
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt('/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt')
    model.eval().to('cuda')

    out = {'samples': [], 'note': 'cached_vs_fresh CLAP cosine; 1.0 = identical'}
    for idx in sample_idx:
        row = tsv_rows[idx]
        cid = row['id']
        cap_in_tsv = row['caption']

        # Look up NPZ — assume idx-indexed (idx.npz)
        npz_path = NPZ_DIR_QWEN_SINGLE / f'{idx}.npz'
        if not npz_path.exists():
            print(f'  [skip] idx={idx} NPZ missing: {npz_path}')
            continue
        z = np.load(npz_path, allow_pickle=True)
        cached_clap = z['text_features_c']    # (512,)

        # Fresh forward TSV caption
        with torch.no_grad():
            fresh = model.get_text_embedding([cap_in_tsv], use_tensor=False)[0]   # (512,)

        cos = float(np.dot(cached_clap, fresh) / (np.linalg.norm(cached_clap) * np.linalg.norm(fresh) + 1e-8))

        # Also: which Qwen slot does the TSV caption match?
        slot_match = None
        if cid in qwen_data:
            for s, c in enumerate(qwen_data[cid]):
                if c == cap_in_tsv:
                    slot_match = s
                    break

        out['samples'].append({
            'idx': idx,
            'id': cid,
            'tsv_caption_head': cap_in_tsv[:80],
            'qwen_slot': slot_match,
            'cached_vs_fresh_cos': round(cos, 4),
        })
        print(f'  idx={idx:>6}  id={cid:<35}  cos={cos:.4f}  slot={slot_match}')

    coses = [s['cached_vs_fresh_cos'] for s in out['samples']]
    if coses:
        out['cos_mean'] = round(float(np.mean(coses)), 4)
        out['cos_min']  = round(float(np.min(coses)), 4)
        out['n_below_999'] = int(sum(1 for c in coses if c < 0.999))
        print(f'\n  cos mean = {out["cos_mean"]:.4f}  min = {out["cos_min"]:.4f}  '
              f'n<0.999 = {out["n_below_999"]}/{len(coses)}')
        if out['n_below_999'] > 0:
            print('  ⚠️  有 NPZ cache 的 CLAP embedding 跟 TSV caption fresh forward 不一致 → BUG suspect')
        else:
            print('  ✅ NPZ cache 與 TSV caption 對得上')
    return out


def main():
    print('讀 Qwen JSONL...')
    qwen = load_qwen()
    print(f'  {len(qwen):,} ids')

    results = {}
    results['b3_tsv_align']    = b3_tsv_align(qwen)
    results['b4_token_trunc']  = b4_token_trunc(qwen)
    results['b2_npz_verify']   = b2_npz_verify(qwen)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f'\n=> {OUT_JSON}')


if __name__ == '__main__':
    main()
