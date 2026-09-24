#!/usr/bin/env python3
"""
Phase 1: MeanAudio 訓練管線驗證 - 資料準備腳本

目標：
1. 從完整資料集隨機抽取 50 首歌曲（分層抽樣）
2. 生成 TSV 檔案供 MeanAudio 訓練使用
3. 準備音訊檔案（轉檔為 WAV 格式）
4. 產生預處理指令範本

使用方式：
    python phase1_data_preparation.py

輸出：
    - phase1_test.tsv: 訓練用的 TSV 檔案
    - audio/: WAV 格式音訊檔案目錄
    - phase1_commands.sh: 預處理與訓練指令範本

更新紀錄：
    - v1.1: 改為轉檔 WAV（解決 torchaudio MP3 載入失敗問題）
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import sys
import os
import subprocess

class Phase1DataPreparation:
    """Phase 1 資料準備"""
    
    def __init__(self, 
                 jsonl_path="~/music_cleaning_results/results_20260119_043407.jsonl",
                 audio_base_dir="/mnt/HDD/kojiek/music_semantic_fidelity/original_audio",
                 output_dir="~/research_dev/meanaudio_training/phase1"):
        
        self.jsonl_path = Path(jsonl_path).expanduser()
        self.audio_base_dir = Path(audio_base_dir)
        self.output_dir = Path(output_dir).expanduser()
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目錄
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        print(f"資料來源: {self.jsonl_path}")
        print(f"音訊目錄: {self.audio_base_dir}")
        print(f"輸出目錄: {self.output_dir}")
        print()
    
    def load_and_stratify_samples(self, target_count=50):
        """載入資料並進行分層抽樣"""
        print("載入資料並分層...")
        
        # 按 credibility 分層
        high_cred = []  # >= 0.8
        mid_cred = []   # 0.6 - 0.8
        low_cred = []   # < 0.6
        
        total_loaded = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                total_loaded += 1
                
                # 提取資料
                cred_score = record['credibility_analysis']['credibility_score']
                
                # 分層
                if cred_score >= 0.8:
                    high_cred.append(record)
                elif cred_score >= 0.6:
                    mid_cred.append(record)
                else:
                    low_cred.append(record)
        
        print(f"  載入 {total_loaded:,} 筆記錄")
        print(f"  高可信度 (>=0.8):  {len(high_cred):,} 筆")
        print(f"  中可信度 (0.6-0.8): {len(mid_cred):,} 筆")
        print(f"  低可信度 (<0.6):  {len(low_cred):,} 筆")
        print()
        
        # 分層抽樣（保持比例）
        n_high = int(target_count * 0.36)  # 36%
        n_mid = int(target_count * 0.48)   # 48%
        n_low = target_count - n_high - n_mid  # 剩餘
        
        print(f"抽樣目標: {target_count} 首")
        print(f"  高可信度: {n_high} 首")
        print(f"  中可信度: {n_mid} 首")
        print(f"  低可信度: {n_low} 首")
        print()
        
        # 隨機抽取
        random.seed(42)  # 固定 seed 確保可重現
        
        selected_high = random.sample(high_cred, n_high)
        selected_mid = random.sample(mid_cred, n_mid)
        selected_low = random.sample(low_cred, n_low)
        
        selected = selected_high + selected_mid + selected_low
        random.shuffle(selected)  # 打亂順序
        
        return selected
    
    def generate_tsv(self, selected_samples):
        """生成 TSV 檔案"""
        print("生成 TSV 檔案...")
        
        tsv_path = self.output_dir / "phase1_test.tsv"
        
        # 寫入 TSV
        with open(tsv_path, 'w', encoding='utf-8') as f:
            # 寫入標頭
            f.write("id\tcaption\n")
            
            # 寫入資料
            for idx, record in enumerate(selected_samples):
                # 使用 representative caption（最佳 caption）
                caption = record['representative_caption']['caption']
                
                # 清理 caption（移除換行和 tab）
                caption = caption.replace('\n', ' ').replace('\t', ' ').strip()
                
                # 寫入
                f.write(f"{idx}\t{caption}\n")
        
        print(f"  TSV 檔案已生成: {tsv_path}")
        print(f"  包含 {len(selected_samples)} 筆記錄")
        print()
        
        return tsv_path
    
    def prepare_audio_files(self, selected_samples):
        """
        準備音訊檔案 — 轉檔為 WAV 格式
        
        MeanAudio 的 partition_clips.py 使用 torchaudio.load() 載入音訊。
        為確保相容性，統一轉為 16kHz mono WAV。
        """
        print("準備音訊檔案（MP3 -> WAV 轉檔）...")
        
        # 先檢查 ffmpeg 是否可用
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            use_ffmpeg = True
            print("  使用 ffmpeg 進行轉檔")
        except (FileNotFoundError, subprocess.CalledProcessError):
            use_ffmpeg = False
            print("  ffmpeg 不可用，嘗試使用 torchaudio 轉檔")
        
        success_count = 0
        failed_files = []
        
        for idx, record in enumerate(selected_samples):
            # 取得原始音訊路徑
            audio_path = Path(record['audio_path'])
            
            # 目標檔案名稱（WAV 格式）
            target_path = self.audio_dir / f"{idx}.wav"
            
            # 如果已存在且大小 > 0，跳過
            if target_path.exists() and target_path.stat().st_size > 0:
                success_count += 1
                continue
            
            # 檢查來源檔案
            if not audio_path.exists():
                failed_files.append((idx, str(audio_path), "來源檔案不存在"))
                print(f"  [SKIP] #{idx}: 來源檔案不存在 - {audio_path.name}")
                continue
            
            # 轉檔
            try:
                if use_ffmpeg:
                    # ffmpeg 轉為 16kHz mono WAV
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(audio_path),
                        "-ar", "16000",      # 16kHz（MeanAudio 預設取樣率）
                        "-ac", "1",          # mono
                        "-f", "wav",
                        str(target_path)
                    ]
                    result = subprocess.run(
                        cmd, capture_output=True, timeout=30
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
                else:
                    # 使用 torchaudio 作為備選
                    import torchaudio
                    waveform, sr = torchaudio.load(str(audio_path))
                    # 轉為 mono
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    # 重新取樣到 16kHz
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    torchaudio.save(str(target_path), waveform, 16000)
                
                success_count += 1
                
            except Exception as e:
                failed_files.append((idx, str(audio_path), str(e)))
                print(f"  [FAIL] #{idx}: {audio_path.name} - {e}")
                # 清除不完整的輸出檔案
                if target_path.exists():
                    target_path.unlink()
        
        print()
        print(f"  成功轉檔: {success_count} / {len(selected_samples)}")
        
        if failed_files:
            print(f"  失敗: {len(failed_files)} 個")
            fail_log = self.output_dir / "phase1_failed_files.txt"
            with open(fail_log, 'w') as f:
                for idx, path, reason in failed_files:
                    f.write(f"{idx}\t{path}\t{reason}\n")
            print(f"  失敗清單: {fail_log}")
        print()
        
        return success_count, failed_files
    
    def save_metadata(self, selected_samples):
        """儲存選取樣本的完整 metadata"""
        print("儲存 metadata...")
        
        metadata_path = self.output_dir / "phase1_metadata.jsonl"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for idx, record in enumerate(selected_samples):
                # 添加 idx 資訊
                record['phase1_id'] = idx
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"  Metadata 已儲存: {metadata_path}")
        print()
    
    def generate_command_script(self):
        """生成預處理與訓練指令範本"""
        print("生成指令範本...")
        
        script_path = self.output_dir / "phase1_commands.sh"
        
        script_content = f"""#!/bin/bash
#
# Phase 1: MeanAudio 訓練管線驗證 - 執行指令
#
# 用途: 執行預處理與訓練測試
# 音訊格式: WAV (16kHz mono)
#

set -e  # 遇到錯誤立即停止

echo "=================================================="
echo "Phase 1: MeanAudio 訓練管線驗證"
echo "=================================================="
echo ""

# ============================================
# 設定路徑
# ============================================
PHASE1_DIR="{self.output_dir}"
MEANAUDIO_DIR="$HOME/MeanAudio"
CAPTIONS_TSV="$PHASE1_DIR/phase1_test.tsv"
AUDIO_DIR="$PHASE1_DIR/audio"
OUTPUT_NPZ_DIR="$PHASE1_DIR/npz"
OUTPUT_PARTITION="$PHASE1_DIR/partition.tsv"
LATENT_DIR="$PHASE1_DIR/latents"

# ============================================
# 前置檢查
# ============================================
echo "前置檢查..."

WAV_COUNT=$(ls -1 $AUDIO_DIR/*.wav 2>/dev/null | wc -l)
echo "  WAV 檔案數量: $WAV_COUNT"

if [ "$WAV_COUNT" -eq 0 ]; then
    echo "錯誤: audio 目錄中沒有 WAV 檔案"
    echo "請先執行: python phase1_data_preparation.py"
    exit 1
fi

# 清理上次可能殘留的中間檔案
rm -rf $LATENT_DIR $OUTPUT_NPZ_DIR
mkdir -p $LATENT_DIR $OUTPUT_NPZ_DIR

echo ""

# ============================================
# 步驟 1: 切換到 MeanAudio 目錄
# ============================================
echo "切換到 MeanAudio 目錄..."
cd $MEANAUDIO_DIR

# ============================================
# 步驟 2: 啟動虛擬環境
# ============================================
echo "啟動虛擬環境..."
source $HOME/venvs/dac/bin/activate

# ============================================
# 步驟 3: 執行音訊分割 (Partition)
# ============================================
echo ""
echo "=================================================="
echo "步驟 1/3: 音訊分割 (Partition)"
echo "=================================================="

python training/partition_clips.py \\
    --data_dir $AUDIO_DIR \\
    --output_dir $OUTPUT_PARTITION

# 驗證 partition 結果
PARTITION_LINES=$(wc -l < $OUTPUT_PARTITION)
PARTITION_DATA_LINES=$((PARTITION_LINES - 1))
echo ""
echo "  Partition 產生 $PARTITION_DATA_LINES 個片段"

if [ "$PARTITION_DATA_LINES" -eq 0 ]; then
    echo "錯誤: Partition 沒有產生任何片段！"
    echo "可能原因: 音訊檔案太短（< 5 秒）或格式無法載入"
    echo ""
    echo "診斷: 嘗試手動載入第一個檔案..."
    python -c "
import torchaudio
import os
audio_dir = '$AUDIO_DIR'
files = sorted(os.listdir(audio_dir))[:3]
for f in files:
    path = os.path.join(audio_dir, f)
    try:
        w, sr = torchaudio.load(path)
        print(f'  OK: {{f}} -> shape={{w.shape}}, sr={{sr}}, duration={{w.shape[1]/sr:.1f}}s')
    except Exception as e:
        print(f'  FAIL: {{f}} -> {{e}}')
"
    exit 1
fi

echo "Partition 完成"
echo ""

# ============================================
# 步驟 4: 提取音訊特徵 (Extract Audio Latents)
# ============================================
echo "=================================================="
echo "步驟 2/3: 提取音訊特徵"
echo "=================================================="
echo "注意：此步驟可能需要 5-10 分鐘"
echo ""

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \\
    --captions_tsv $CAPTIONS_TSV \\
    --data_dir $AUDIO_DIR \\
    --clips_tsv $OUTPUT_PARTITION \\
    --latent_dir $LATENT_DIR \\
    --output_dir $OUTPUT_NPZ_DIR \\
    --text_encoder='t5_clap'

echo "音訊特徵提取完成"
echo ""

# ============================================
# 步驟 5: 驗證 NPZ 檔案
# ============================================
echo "=================================================="
echo "步驟 3/3: 驗證 NPZ 檔案"
echo "=================================================="

NPZ_COUNT=$(ls -1 $OUTPUT_NPZ_DIR/*.npz 2>/dev/null | wc -l)
echo "生成的 NPZ 檔案數量: $NPZ_COUNT"

if [ "$NPZ_COUNT" -gt 0 ]; then
    echo "NPZ 生成成功"
else
    echo "警告: 沒有生成 NPZ 檔案"
fi

# 檢查第一個 NPZ 檔案的內容
echo ""
echo "檢查第一個 NPZ 檔案的結構..."
python -c "
import numpy as np
import os
npz_dir = '$OUTPUT_NPZ_DIR'
npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
if npz_files:
    npz_file = os.path.join(npz_dir, npz_files[0])
    data = np.load(npz_file)
    print('NPZ 檔案結構:')
    for key in data.files:
        print(f'  {{key}}: {{data[key].shape}}')
else:
    print('沒有找到 NPZ 檔案')
"

echo ""
echo "=================================================="
echo "預處理完成！"
echo "=================================================="
echo ""
echo "下一步：執行訓練測試"
echo ""
echo "訓練指令範本（請根據需要調整）："
echo ""
echo "export CUDA_VISIBLE_DEVICES=0"
echo "cd $MEANAUDIO_DIR"
echo ""
echo "torchrun --standalone --nproc_per_node=1 train.py \\\\"
echo "    --config-name train_config.yaml \\\\"
echo "    exp_id=phase1_test \\\\"
echo "    model=meanaudio_s \\\\"
echo "    batch_size=4 \\\\"
echo "    num_iterations=500 \\\\"
echo "    val_interval=100 \\\\"
echo "    save_checkpoint_interval=100 \\\\"
echo "    mini_train=False \\\\"
echo "    data.AudioCaps_npz.tsv=$CAPTIONS_TSV \\\\"
echo "    data.AudioCaps_npz.npz_dir=$OUTPUT_NPZ_DIR \\"
echo "    data.AudioCaps_val_npz.tsv=$CAPTIONS_TSV \\"
echo "    data.AudioCaps_val_npz.npz_dir=$OUTPUT_NPZ_DIR \\"
echo "    ++use_rope=False \\"
echo "    ++use_wandb=False \\"
echo "    ++ema.enable=False"
echo ""
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 設定執行權限
        os.chmod(script_path, 0o755)
        
        print(f"  指令範本已生成: {script_path}")
        print()
        
        return script_path
    
    def run(self):
        """執行完整的資料準備流程"""
        print("=" * 60)
        print("Phase 1: MeanAudio 訓練管線驗證 - 資料準備")
        print("=" * 60)
        print()
        
        # 1. 載入並分層抽樣
        selected_samples = self.load_and_stratify_samples(target_count=50)
        
        # 2. 生成 TSV
        tsv_path = self.generate_tsv(selected_samples)
        
        # 3. 準備音訊檔案（轉為 WAV）
        success_count, failed_files = self.prepare_audio_files(selected_samples)
        
        # 4. 儲存 metadata
        self.save_metadata(selected_samples)
        
        # 5. 生成指令範本
        script_path = self.generate_command_script()
        
        # 總結
        print("=" * 60)
        print("Phase 1 資料準備完成！")
        print("=" * 60)
        print()
        print("準備結果:")
        print(f"  總樣本數: 50")
        print(f"  成功轉檔: {success_count}")
        print(f"  失敗: {len(failed_files)}")
        print()
        print("輸出檔案:")
        print(f"  TSV 檔案:  {tsv_path}")
        print(f"  音訊目錄:  {self.audio_dir}  (WAV 格式)")
        print(f"  指令範本:  {script_path}")
        print()
        print("下一步:")
        print(f"  1. 執行預處理:")
        print(f"     bash {script_path}")
        print()
        print(f"  2. 預處理完成後，檢查 NPZ 檔案:")
        print(f"     ls -lh {self.output_dir}/npz/")
        print()
        print(f"  3. 執行訓練測試（參考指令範本末尾）")
        print()


def main():
    """主程式"""
    try:
        # 建立資料準備器
        prep = Phase1DataPreparation()
        
        # 執行準備流程
        prep.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n使用者中斷")
        return 1
    
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())