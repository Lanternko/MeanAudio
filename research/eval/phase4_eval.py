"""
Phase 4/3 音訊生成評估腳本
計算指標：CLAP Score、Audiobox Aesthetics（CE / CU / PC / PQ）、FAD（選用）

用法範例：
  # CLAP + AES（預設）
  python3 phase4_eval.py --gen_dir ./eval_output/phase4_v2_test

  # CLAP + AES + FAD
  python3 phase4_eval.py --gen_dir ./eval_output/phase4_v2_test --fad --num_samples 2048

  # 快速測試
  python3 phase4_eval.py --gen_dir ./eval_output/phase3_test --test_mode

腳本位置：~/research/meanaudio_eval/phase4_eval.py
"""

import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── 預設路徑 ──────────────────────────────────────────────
DEFAULT_REF_DIR   = '/mnt/HDD/kojiek/music_semantic_fidelity/original_audio'
DEFAULT_TSV       = '/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv'
DEFAULT_CLAP_CKPT = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
DEFAULT_OUT_DIR   = '/home/kojiek/MeanAudio/eval_output/metrics'
# ─────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description='音訊生成評估：CLAP Score + Audiobox Aesthetics + FAD（選用）')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='生成音訊目錄（.flac 檔案）')
    parser.add_argument('--tsv', type=str, default=DEFAULT_TSV,
                        help='Test TSV 路徑（id, caption）')
    parser.add_argument('--ref_dir', type=str, default=DEFAULT_REF_DIR,
                        help='原始參考音訊根目錄（用於 FAD）')
    parser.add_argument('--clap_ckpt', type=str, default=DEFAULT_CLAP_CKPT,
                        help='CLAP checkpoint 路徑')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR,
                        help='評估結果輸出目錄')
    parser.add_argument('--exp_name', type=str, default='',
                        help='實驗名稱（用於輸出檔名）')
    parser.add_argument('--aes_batch_size', type=int, default=32,
                        help='Audiobox Aesthetics batch size')
    parser.add_argument('--skip_clap', action='store_true',
                        help='跳過 CLAP 計算')
    parser.add_argument('--skip_aes', action='store_true',
                        help='跳過 Audiobox Aesthetics 計算')
    parser.add_argument('--fad', action='store_true',
                        help='啟用 FAD 計算（預設關閉）')
    parser.add_argument('--num_samples', type=int, default=2048,
                        help='FAD 抽樣筆數（預設 2048）')
    parser.add_argument('--test_mode', action='store_true',
                        help='只跑前 100 筆，快速驗證流程')
    parser.add_argument('--seed', type=int, default=42,
                        help='抽樣隨機種子')
    return parser.parse_args()


def load_test_tsv(tsv_path, limit=None):
    """讀取 TSV，回傳 [(clip_id, caption), ...]"""
    records = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            records.append((row['id'], row['caption']))
            if limit and len(records) >= limit:
                break
    return records


def get_ref_audio_path(clip_id, ref_dir):
    """
    Resolve reference audio path for a given clip_id.

    Tries (in order):
      1. Jamendo layout: <ref_dir>/<prefix>/<track_id>/<segment>.mp3
         (clip_id like "62_1317562_segment_1")
      2. Flat layout: <ref_dir>/<clip_id>.wav  (MusicCaps style)
      3. Flat layout: <ref_dir>/<clip_id>.mp3
      4. Flat layout: <ref_dir>/<clip_id>.flac

    Returns the first existing path, else None.
    """
    # Try Jamendo nested layout first if clip_id splits as expected
    parts = clip_id.split('_')
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        prefix = parts[0]
        track_id = parts[1]
        seg_name = '_'.join(parts[2:]) + '.mp3'
        jamendo_path = os.path.join(ref_dir, prefix, track_id, seg_name)
        if os.path.exists(jamendo_path):
            return jamendo_path

    # Flat MusicCaps-style layout
    for ext in ('wav', 'mp3', 'flac'):
        flat = os.path.join(ref_dir, f'{clip_id}.{ext}')
        if os.path.exists(flat):
            return flat

    # Fallback: return Jamendo-shaped guess (for back-compat when file missing)
    if len(parts) >= 3:
        prefix = parts[0]
        track_id = parts[1]
        seg_name = '_'.join(parts[2:]) + '.mp3'
        return os.path.join(ref_dir, prefix, track_id, seg_name)
    return None


# ── CLAP Score ────────────────────────────────────────────
def compute_clap_score(records, gen_dir, clap_ckpt):
    """計算所有生成音訊與對應 caption 的 CLAP 餘弦相似度均值"""
    import torch
    import laion_clap

    print('\n[CLAP] 載入模型...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(clap_ckpt)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    scores = []
    missing = 0

    for clip_id, caption in tqdm(records, desc='CLAP Score'):
        gen_path = os.path.join(gen_dir, f'{clip_id}.flac')
        if not os.path.exists(gen_path):
            missing += 1
            continue
        try:
            audio_embed = model.get_audio_embedding_from_filelist(
                [gen_path], use_tensor=True
            )
            text_embed = model.get_text_embedding([caption], use_tensor=True)
            sim = torch.nn.functional.cosine_similarity(
                audio_embed, text_embed, dim=-1
            )
            scores.append(sim.item())
        except Exception as e:
            missing += 1
            tqdm.write(f'[CLAP][WARN] skip {clip_id}: {e}')

    print(f'[CLAP] 完成：{len(scores)} 筆，跳過 {missing} 筆')
    if not scores:
        raise RuntimeError(
            f'[CLAP] 所有 {missing} 筆音訊都被跳過，CLAP 無法計算。'
            f' 請確認 gen_dir 路徑（{gen_dir}）和音訊完整性。'
        )
    return float(np.mean(scores))


# ── Audiobox Aesthetics ───────────────────────────────────
def compute_aes(records, gen_dir, batch_size=32):
    """
    計算 Audiobox Aesthetics 四個子指標的均值。
    回傳 dict: {CE, CU, PC, PQ}
    """
    import torch
    import torchaudio
    import audiobox_aesthetics.infer as _aes_infer

    # torchcodec/FFmpeg not available in this env — patch read_wav to use soundfile
    def _read_wav_sf(meta):
        import soundfile as sf
        import torch
        try:
            wav, sr = sf.read(meta['path'], dtype='float32', always_2d=True)
        except Exception as e:
            raise IOError(f"Cannot read {meta['path']}: {e}")
        wav = torch.from_numpy(wav.T)  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

    _aes_infer.read_wav = _read_wav_sf

    from audiobox_aesthetics.infer import AesPredictor

    print('\n[AES] 載入模型...')
    predictor = AesPredictor(checkpoint_pth=None, batch_size=batch_size)

    paths = []
    for clip_id, _ in records:
        gen_path = os.path.join(gen_dir, f'{clip_id}.flac')
        if os.path.exists(gen_path):
            paths.append(gen_path)

    missing = len(records) - len(paths)
    print(f'[AES] 共 {len(paths)} 筆（跳過 {missing} 筆）')

    all_CE, all_CU, all_PC, all_PQ = [], [], [], []
    skipped_aes = 0
    for i in tqdm(range(0, len(paths), batch_size), desc='AES'):
        batch = [{'path': p} for p in paths[i:i + batch_size]]
        try:
            results = predictor.forward(batch)
        except Exception as e:
            skipped_aes += len(batch)
            continue
        for r in results:
            all_CE.append(r['CE'])
            all_CU.append(r['CU'])
            all_PC.append(r['PC'])
            all_PQ.append(r['PQ'])
    if skipped_aes:
        print(f'[AES] 跳過損壞 batch：{skipped_aes} 筆')

    return {
        'aes_CE': float(np.mean(all_CE)),
        'aes_CU': float(np.mean(all_CU)),
        'aes_PC': float(np.mean(all_PC)),
        'aes_PQ': float(np.mean(all_PQ)),
    }


# ── FAD ───────────────────────────────────────────────────
def compute_fad(records, gen_dir, ref_dir, num_samples=2048, seed=42):
    """
    計算 FAD。
    num_samples: 抽樣筆數（None = 全量，2048 已足夠穩定）。
    """
    import shutil
    import soundfile as sf
    import librosa
    from frechet_audio_distance import FrechetAudioDistance

    if num_samples and num_samples < len(records):
        random.seed(seed)
        sampled = random.sample(records, num_samples)
        print(f'\n[FAD] 抽樣 {num_samples}/{len(records)} 筆（seed={seed}）')
    else:
        sampled = records
        print(f'\n[FAD] 全量 {len(records)} 筆')

    tmp_ref = '/tmp/phase_fad_ref'
    tmp_gen = '/tmp/phase_fad_gen'
    os.makedirs(tmp_ref, exist_ok=True)
    os.makedirs(tmp_gen, exist_ok=True)

    copied = 0
    for clip_id, _ in tqdm(sampled, desc='FAD 音訊準備'):
        ref_path = get_ref_audio_path(clip_id, ref_dir)
        gen_path = os.path.join(gen_dir, f'{clip_id}.flac')

        if not ref_path or not os.path.exists(ref_path):
            continue
        if not os.path.exists(gen_path):
            continue

        ref_out = os.path.join(tmp_ref, f'{clip_id}.wav')
        gen_out = os.path.join(tmp_gen, f'{clip_id}.wav')

        if not os.path.exists(ref_out):
            audio, _ = librosa.load(ref_path, sr=16000, mono=True)
            sf.write(ref_out, audio, 16000)

        if not os.path.exists(gen_out):
            audio, _ = librosa.load(gen_path, sr=16000, mono=True)
            sf.write(gen_out, audio, 16000)

        copied += 1

    print(f'[FAD] 準備完成：{copied} 對')

    fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)
    score = fad.score(tmp_ref, tmp_gen)

    shutil.rmtree(tmp_ref, ignore_errors=True)
    shutil.rmtree(tmp_gen, ignore_errors=True)

    return score


# ── 主程式 ────────────────────────────────────────────────
def main():
    args = parse_args()

    exp_name = args.exp_name or Path(args.gen_dir).name
    out_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    limit = 100 if args.test_mode else None
    print(f'實驗：{exp_name}')
    print(f'模式：{"測試（100筆）" if args.test_mode else "完整"}')
    print(f'生成目錄：{args.gen_dir}')

    records = load_test_tsv(args.tsv, limit=limit)
    print(f'載入 {len(records)} 筆 test records')

    results = {}

    # CLAP Score
    if not args.skip_clap:
        clap = compute_clap_score(records, args.gen_dir, args.clap_ckpt)
        results['clap_score'] = clap
        print(f'\n✅ CLAP Score: {clap:.4f}')

    # Audiobox Aesthetics
    if not args.skip_aes:
        aes = compute_aes(records, args.gen_dir, batch_size=args.aes_batch_size)
        results.update(aes)
        print(f'\n✅ Audiobox Aesthetics:')
        for k, v in aes.items():
            print(f'   {k}: {v:.4f}')

    # FAD（選用）
    if args.fad:
        fad = compute_fad(
            records, args.gen_dir, args.ref_dir,
            num_samples=args.num_samples, seed=args.seed
        )
        results['fad'] = fad
        print(f'✅ FAD (n={args.num_samples}): {fad:.4f}')

    # 儲存結果
    result_path = os.path.join(out_dir, 'metrics.txt')
    with open(result_path, 'w') as f:
        f.write(f'Experiment: {exp_name}\n')
        f.write(f'Test TSV: {args.tsv}\n')
        f.write(f'Generated audio: {args.gen_dir}\n')
        f.write(f'Test clips: {len(records)}\n')
        f.write('─' * 40 + '\n')
        for k, v in results.items():
            f.write(f'{k}: {v:.4f}\n')

    print(f'\n結果已儲存至 {result_path}')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
