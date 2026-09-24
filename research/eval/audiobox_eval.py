"""
Audiobox Aesthetics 批次評估腳本
計算指標：CE、CU、PC、PQ（Meta Audiobox Aesthetics）

用法範例：
  python3 audiobox_eval.py --gen_dir ~/MeanAudio/eval_output/phase6_v2_q6_jamendo/audio --exp_name phase6_v2_q6
  python3 audiobox_eval.py --gen_dir ~/MeanAudio/eval_output/phase4_v2_test --exp_name phase4_v2 --num_samples 2048
"""

import os
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEFAULT_OUT_DIR = '/home/kojiek/MeanAudio/eval_output/metrics'


def parse_args():
    parser = argparse.ArgumentParser(description='Audiobox Aesthetics 評估：CE / CU / PC / PQ')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='生成音訊目錄（遞迴搜尋 .flac 檔）')
    parser.add_argument('--exp_name', type=str, default='',
                        help='實驗名稱（用於輸出檔名）')
    parser.add_argument('--num_samples', type=int, default=2048,
                        help='抽樣筆數（預設 2048）')
    parser.add_argument('--seed', type=int, default=42,
                        help='抽樣隨機種子（預設 42）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每批送入 predictor 的筆數（預設 32）')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR,
                        help='評估結果輸出根目錄')
    return parser.parse_args()


def collect_flac_files(gen_dir):
    """遞迴找所有 .flac 檔，回傳路徑 list"""
    files = sorted(Path(gen_dir).rglob('*.flac'))
    return [str(f) for f in files]


def sample_files(files, num_samples, seed):
    if num_samples and num_samples < len(files):
        random.seed(seed)
        return random.sample(files, num_samples)
    return files


def compute_aesthetics(files, batch_size):
    """分批送入 AesPredictor，回傳四個指標的均值"""
    # torchaudio 新版預設用 torchcodec，但此環境 FFmpeg 未裝 → patch 成 soundfile
    import torchaudio
    import soundfile as sf
    import torch as _torch

    def _load_sf(path, **kwargs):
        data, sr = sf.read(str(path), always_2d=True)
        return _torch.from_numpy(data.T).float(), sr

    torchaudio.load = _load_sf

    from audiobox_aesthetics.infer import AesPredictor

    print('\n[Audiobox] 載入模型（首次執行會從 HF 下載權重）...')
    predictor = AesPredictor(checkpoint_pth=None, batch_size=batch_size)

    all_scores = {axis: [] for axis in ['CE', 'CU', 'PC', 'PQ']}

    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    for batch_paths in tqdm(batches, desc='Audiobox Aesthetics'):
        batch = [{'path': p} for p in batch_paths]
        results = predictor.forward(batch)  # list of dict: [{'CE':x,'CU':x,...}, ...]
        for item in results:
            for axis in all_scores:
                all_scores[axis].append(item[axis])

    return {axis: float(np.mean(vals)) for axis, vals in all_scores.items()}


def main():
    args = parse_args()

    exp_name = args.exp_name or Path(args.gen_dir).name
    out_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'實驗：{exp_name}')
    print(f'生成目錄：{args.gen_dir}')

    all_files = collect_flac_files(args.gen_dir)
    print(f'找到 {len(all_files)} 個 .flac 檔')

    sampled = sample_files(all_files, args.num_samples, args.seed)
    print(f'抽樣 {len(sampled)} 筆（seed={args.seed}）')

    scores = compute_aesthetics(sampled, args.batch_size)

    # 儲存結果
    result_path = os.path.join(out_dir, 'audiobox_metrics.txt')
    with open(result_path, 'w') as f:
        f.write(f'Experiment: {exp_name}\n')
        f.write(f'Generated audio: {args.gen_dir}\n')
        f.write(f'Samples: {len(sampled)} (seed={args.seed})\n')
        f.write('─' * 40 + '\n')
        for k, v in scores.items():
            f.write(f'{k}: {v:.4f}\n')

    print(f'\n結果已儲存至 {result_path}')
    print('─' * 40)
    for k, v in scores.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
