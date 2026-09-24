"""
Phase 8 V2 前置工作：對所有訓練 clip 跑 Audiobox PQ inference
輸出 JSONL：每行 {"id": "...", "pq": 7.23}

用法：
  python gen_pq_labels.py [--tsv PATH] [--output PATH] [--batch_size 32] [--resume]

  --tsv        訓練 TSV（預設 phase7_v1_train.tsv）
  --output     輸出 JSONL 路徑
  --batch_size Audiobox batch size（預設 32）
  --resume     若輸出檔案已存在，跳過已完成的 clip（斷點續跑）
"""

import csv
import json
import argparse
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

INPUT_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
AUDIO_ROOT  = Path('/home/kojiek/data/segments_no_vocals')
DEFAULT_OUT = Path.home() / 'research/meanaudio_training/pq_labels.jsonl'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv',        type=str, default=str(INPUT_TSV))
    parser.add_argument('--output',     type=str, default=str(DEFAULT_OUT))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume',     action='store_true',
                        help='跳過輸出 JSONL 中已有的 clip（斷點續跑）')
    return parser.parse_args()


def id_to_audio_path(clip_id: str) -> Path:
    """94_1317594_segment_0 → AUDIO_ROOT/94/1317594/segment_0.mp3"""
    parts = clip_id.split('_')
    prefix   = parts[0]
    track_id = parts[1]
    seg_name = '_'.join(parts[2:]) + '.mp3'
    return AUDIO_ROOT / prefix / track_id / seg_name


def load_predictor(batch_size: int):
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
    predictor = AesPredictor(checkpoint_pth=None, batch_size=batch_size)
    return predictor


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀取 TSV 取得所有 clip_id
    clip_ids = []
    with open(args.tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            clip_ids.append(row['id'])
    print(f'TSV 共 {len(clip_ids):,} 筆')

    # 斷點續跑：載入已完成的 id
    done_ids = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(d['id'])
                except Exception:
                    pass
        print(f'已完成 {len(done_ids):,} 筆，跳過')

    # 過濾待處理清單
    todo = [cid for cid in clip_ids if cid not in done_ids]
    print(f'待處理 {len(todo):,} 筆')

    if not todo:
        print('全部已完成！')
        return

    print('載入 Audiobox AES...')
    predictor = load_predictor(args.batch_size)

    n_missing  = 0
    n_error    = 0

    # 以 append 模式寫入（支援斷點續跑）
    with open(output_path, 'a') as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc='PQ inference'):
            batch_ids   = todo[i: i + args.batch_size]
            batch_paths = [id_to_audio_path(cid) for cid in batch_ids]

            # 過濾不存在的檔案
            valid_ids   = []
            valid_paths = []
            for cid, path in zip(batch_ids, batch_paths):
                if path.exists():
                    valid_ids.append(cid)
                    valid_paths.append(path)
                else:
                    n_missing += 1
                    fout.write(json.dumps({'id': cid, 'pq': None, 'error': 'file_not_found'}) + '\n')

            if not valid_ids:
                continue

            try:
                inputs = [{'path': str(p)} for p in valid_paths]
                scores = predictor.forward(inputs)
                for cid, score in zip(valid_ids, scores):
                    fout.write(json.dumps({'id': cid, 'pq': float(score['PQ'])}) + '\n')
            except Exception as e:
                n_error += len(valid_ids)
                for cid in valid_ids:
                    fout.write(json.dumps({'id': cid, 'pq': None, 'error': str(e)}) + '\n')

    print(f'\n完成！')
    print(f'  找不到音訊檔：{n_missing:,}')
    print(f'  推理錯誤：    {n_error:,}')
    print(f'  輸出 → {output_path}')
    print(f'\n下一步：python analyze_pq_distribution.py 查看 PQ 分佈，再決定量化 bin')


if __name__ == '__main__':
    main()
