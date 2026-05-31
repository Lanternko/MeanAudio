"""
Phase 4 預處理腳本
==================
將 Jamendo 421K 音訊片段從 MP3 轉換為 NPZ 訓練特徵。

流程：
    Step 1：MP3 → WAV（16kHz, mono，平坦目錄，16 核心平行）
    Step 2：生成 clips_tsv（直接生成，跳過 partition_clips.py）
    Step 3：extract_audio_latents.py（GPU 提取 NPZ）

目錄結構：
    原始音訊：/mnt/HDD/kojiek/music_semantic_fidelity/original_audio/<xx>/<id>/segment_N.mp3
    輸出 WAV： /mnt/HDD/kojiek/phase4_jamendo_data/wav_audio/<xx_id_segment_N>.wav
    輸出 NPZ： /mnt/HDD/kojiek/phase4_jamendo_data/npz/

使用方式：
    tmux new-session -s phase4_prep
    source /home/kojiek/venvs/dac/bin/activate
    cd ~/MeanAudio
    python3 phase4_preprocess.py 2>&1 | tee ~/logs/phase4_preprocess.log
"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# 路徑設定
# ============================================================
AUDIO_BASE   = Path("/mnt/HDD/kojiek/music_semantic_fidelity/original_audio")
WAV_DIR      = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
NPZ_DIR      = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz")
CAPTIONS_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/meanaudio_captions.tsv")
CLIPS_TSV    = Path("/mnt/HDD/kojiek/phase4_jamendo_data/clips.tsv")
META_JSON    = Path("/mnt/HDD/kojiek/music_semantic_fidelity/metadata/meta_all.json")
MEANAUDIO    = Path("/home/kojiek/MeanAudio")

SAMPLE_RATE  = 16000
NUM_WORKERS  = 16  # MP3→WAV 平行進程數
BATCH_SIZE   = 8
NUM_LOADERS  = 10

# ============================================================
# Step 1：MP3 → WAV（平行轉換）
# ============================================================

def convert_one(args):
    """將單一 MP3 轉換為 WAV，回傳 (wav_id, success)"""
    mp3_path, wav_path = args
    if wav_path.exists():
        return wav_path.stem, True  # 斷點續傳：已存在則跳過
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path),
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
             str(wav_path), "-loglevel", "error"],
            capture_output=True, timeout=60
        )
        return wav_path.stem, result.returncode == 0
    except Exception:
        return wav_path.stem, False

def step1_convert_mp3_to_wav():
    print("\n" + "="*60)
    print("Step 1：MP3 → WAV（16kHz mono，平行 16 進程）")
    print("="*60)

    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有 MP3 路徑
    print("  掃描 MP3 檔案...")
    tasks = []
    with open(META_JSON) as f:
        data = json.load(f)
    for item in data:
        mp3_path = AUDIO_BASE / item["path"]
        wav_name = item["path"].replace("/", "_").replace(".mp3", ".wav")
        wav_path = WAV_DIR / wav_name
        tasks.append((mp3_path, wav_path))

    print(f"  總任務數：{len(tasks):,}")

    # 統計已存在
    already = sum(1 for _, w in tasks if w.exists())
    print(f"  已完成（斷點續傳跳過）：{already:,}")

    remaining = [(m, w) for m, w in tasks if not w.exists()]
    print(f"  待轉換：{len(remaining):,}")

    if not remaining:
        print("  全部已完成，跳過轉換。")
        return

    success = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(convert_one, t): t for t in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining), desc="  轉換進度"):
            _, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1

    print(f"  成功：{success + already:,}  失敗：{failed:,}")

# ============================================================
# Step 2：生成 clips_tsv
# ============================================================

def step2_generate_clips_tsv():
    print("\n" + "="*60)
    print("Step 2：生成 clips_tsv")
    print("="*60)

    if CLIPS_TSV.exists():
        print(f"  已存在，跳過：{CLIPS_TSV}")
        return

    # 掃描 WAV 目錄中實際存在的檔案
    print("  掃描 WAV 目錄...")
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    print(f"  找到 WAV 檔案：{len(wav_files):,}")

    records = []
    end_sample = SAMPLE_RATE * 10  # 10 秒固定長度

    for wav_path in tqdm(wav_files, desc="  生成 clips"):
        name = wav_path.stem          # e.g. 62_1317562_segment_1
        audio_id = f"{name}_0"        # e.g. 62_1317562_segment_1_0
        records.append({
            "id": audio_id,
            "name": name,
            "start_sample": 0,
            "end_sample": end_sample,
        })

    df = pd.DataFrame(records, columns=["id", "name", "start_sample", "end_sample"])
    df.to_csv(CLIPS_TSV, index=False, sep="\t")
    print(f"  clips_tsv 寫入完成：{len(df):,} 筆 → {CLIPS_TSV}")

# ============================================================
# Step 3：extract_audio_latents.py
# ============================================================

def step3_extract_npz():
    print("\n" + "="*60)
    print("Step 3：extract_audio_latents.py（NPZ 提取）")
    print("  預計需要 2-3 天，請確保在 tmux 內執行")
    print("="*60)

    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        str(MEANAUDIO / "training/extract_audio_latents.py"),
        "--data_dir",      str(WAV_DIR),
        "--captions_tsv",  str(CAPTIONS_TSV),
        "--clips_tsv",     str(CLIPS_TSV),
        "--output_dir",    str(NPZ_DIR),
        "--text_encoder",  "t5_clap",
        "--batch_size",    str(BATCH_SIZE),
        "--num_workers",   str(NUM_LOADERS),
    ]

    print("  執行指令：")
    print("  " + " \\\n    ".join(cmd))
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(MEANAUDIO)

    result = subprocess.run(cmd)
    if result.returncode == 0:
        npz_count = len(list(NPZ_DIR.glob("*.npz")))
        print(f"\n  NPZ 提取完成！共 {npz_count:,} 個檔案")
    else:
        print(f"\n  [ERROR] extract_audio_latents.py 退出碼：{result.returncode}")

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    import sys

    print("Phase 4 預處理開始")
    print(f"  音訊來源：{AUDIO_BASE}")
    print(f"  WAV 輸出：{WAV_DIR}")
    print(f"  NPZ 輸出：{NPZ_DIR}")

    # 可透過命令列參數只跑特定 step，例如：python3 phase4_preprocess.py step2
    run_all = len(sys.argv) == 1
    run_step = sys.argv[1] if len(sys.argv) > 1 else None

    if run_all or run_step == "step1":
        step1_convert_mp3_to_wav()

    if run_all or run_step == "step2":
        step2_generate_clips_tsv()

    if run_all or run_step == "step3":
        step3_extract_npz()

    print("\n全部完成。")
