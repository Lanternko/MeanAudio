"""
計算 Audiobox PQ 分數與 mean_similarity 的 Spearman correlation
（在 Jamendo 訓練集 1000 個 clip 的 random sample 上）

用法：
  python correlation_pq_vs_meansim.py [--n_samples 1000] [--output results.csv]
"""

import json
import argparse
import random
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import soundfile as sf

JSONL_PATH  = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
AUDIO_ROOT  = Path('/home/kojiek/data/segments_no_vocals')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default=str(Path.home() / 'research/meanaudio_eval/correlation_pq_vs_meansim.csv'))
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def relative_path_to_audio(relative_path):
    """00/1002000/segment_0.mp3 → Path"""
    return AUDIO_ROOT / relative_path


def load_aes_predictor():
    """載入 Audiobox AES，套用 soundfile monkey-patch"""
    import audiobox_aesthetics.infer as _aes_infer

    def _read_wav_sf(meta):
        wav, sr = sf.read(meta['path'], dtype='float32', always_2d=True)
        wav = torch.from_numpy(wav.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

    _aes_infer.read_wav = _read_wav_sf
    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=None, batch_size=32)
    return predictor


def compute_pq_scores(predictor, audio_paths, batch_size=32):
    """批次計算 PQ 分數，回傳 {path_str: pq_score}"""
    results = {}
    for i in tqdm(range(0, len(audio_paths), batch_size), desc='Audiobox PQ'):
        batch_paths = audio_paths[i: i + batch_size]
        inputs = [{'path': str(p)} for p in batch_paths]
        try:
            scores = predictor.forward(inputs)
            for p, s in zip(batch_paths, scores):
                results[str(p)] = float(s['PQ'])
        except Exception as e:
            print(f'  [WARN] batch {i}: {e}')
            for p in batch_paths:
                results[str(p)] = float('nan')
    return results


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f'讀取 JSONL: {JSONL_PATH}')
    entries = []
    with open(JSONL_PATH) as f:
        for line in f:
            d = json.loads(line)
            rel_path = d['relative_path']
            audio_path = relative_path_to_audio(rel_path)
            if not audio_path.exists():
                continue
            mean_sim = d.get('credibility_analysis', {}).get('mean_similarity')
            if mean_sim is None:
                continue
            entries.append({
                'relative_path': rel_path,
                'audio_path': audio_path,
                'mean_similarity': float(mean_sim),
            })

    print(f'  有效 clip 數：{len(entries):,}')
    sampled = rng.sample(entries, min(args.n_samples, len(entries)))
    print(f'  抽樣 {len(sampled):,} 筆（seed={args.seed}）')

    print('\n載入 Audiobox AES...')
    predictor = load_aes_predictor()

    audio_paths = [e['audio_path'] for e in sampled]
    pq_map = compute_pq_scores(predictor, audio_paths)

    # 合併結果
    rows = []
    for e in sampled:
        pq = pq_map.get(str(e['audio_path']), float('nan'))
        rows.append({
            'relative_path': e['relative_path'],
            'mean_similarity': e['mean_similarity'],
            'PQ': pq,
        })

    # 存 CSV
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path', 'mean_similarity', 'PQ'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n結果存至 {output}')

    # 計算 Spearman correlation
    valid = [(r['mean_similarity'], r['PQ']) for r in rows if not np.isnan(r['PQ'])]
    if len(valid) < 10:
        print('有效樣本太少，無法計算相關性')
        return

    ms_arr = np.array([v[0] for v in valid])
    pq_arr = np.array([v[1] for v in valid])

    from scipy.stats import spearmanr, pearsonr
    srcc, srcc_p = spearmanr(ms_arr, pq_arr)
    pcc, pcc_p   = pearsonr(ms_arr, pq_arr)

    print(f'\n========================================')
    print(f'  Spearman r (SRCC) = {srcc:.4f}  (p={srcc_p:.4f})')
    print(f'  Pearson  r (PCC)  = {pcc:.4f}  (p={pcc_p:.4f})')
    print(f'  n = {len(valid)}')
    print(f'========================================')
    print()
    if abs(srcc) > 0.7:
        print('  → 高相關：換 Audiobox PQ 邊際效益有限，考慮 caption 過濾方向')
    elif abs(srcc) < 0.5:
        print('  → 低相關：兩個維度正交，Phase 9 用 Audiobox PQ 取代 mean_sim 值得做')
    else:
        print('  → 中等相關：謹慎評估，建議看 PQ 分佈再決定')


if __name__ == '__main__':
    main()
