#!/usr/bin/env python3
"""
Phase 2: MeanAudio 大規模資料準備腳本 (1,000 首)

支援四種實驗變體：
1. 'baseline'       : 隨機分層抽樣，使用原始 representative_caption。
2. 'hard_filtering' : 僅抽取 High Credibility (>= 0.8) 的資料。
3. 'credibility_tag': 在 caption 前方加入 [High/Medium/Low Credibility] 標籤。
4. 'best_similarity': (預設等同 baseline，可依需求改為選取特定 similarity 最高之 caption)。

使用方式：
    python phase2_data_preparation.py
"""

import json
import random
import subprocess
import concurrent.futures
from pathlib import Path

# ==========================================
# 參數設定區 (請依需求修改)
# ==========================================
VARIANT = "hard_filtering"           # 選項: baseline, hard_filtering, credibility_tag, best_similarity
TARGET_COUNT = 1000            # Phase 2 預設 1000 首
NUM_WORKERS = 8                # 多執行緒轉檔數量

JSONL_PATH = "~/music_cleaning_results/results_20260119_043407.jsonl"
AUDIO_BASE_DIR = "/mnt/HDD/kojiek/music_semantic_fidelity/original_audio"
OUTPUT_DIR = f"~/research_dev/meanaudio_training/phase2_{VARIANT}"

class Phase2DataPreparation:
    def __init__(self):
        self.jsonl_path = Path(JSONL_PATH).expanduser()
        self.output_dir = Path(OUTPUT_DIR).expanduser()
        self.audio_dir = self.output_dir / "audio"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Phase 2 資料準備開始 | 變體: {VARIANT}")
        print(f"目標數量: {TARGET_COUNT} 首")
        print(f"輸出目錄: {self.output_dir}\n")

    def load_and_sample(self):
        """依據變體邏輯載入並抽樣資料"""
        print("📥 載入資料與抽樣...")
        high_cred, mid_cred, low_cred = [], [], []
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                score = record['credibility_analysis']['credibility_score']
                
                if score >= 0.8:
                    high_cred.append(record)
                elif score >= 0.6:
                    mid_cred.append(record)
                else:
                    low_cred.append(record)
                    
        random.seed(42) # 固定 seed
        selected = []
        
        if VARIANT == "hard_filtering":
            # Hard Filtering: 只拿 >= 0.8 的資料
            print(f"  [Hard Filtering] 僅從 High Credibility ({len(high_cred)} 首) 中抽取")
            selected = random.sample(high_cred, min(TARGET_COUNT, len(high_cred)))
        else:
            # Baseline, Credibility Tag, Best Similarity: 分層抽樣 (36%, 48%, 16%)
            n_high = int(TARGET_COUNT * 0.36)
            n_mid = int(TARGET_COUNT * 0.48)
            n_low = TARGET_COUNT - n_high - n_mid
            
            selected += random.sample(high_cred, n_high)
            selected += random.sample(mid_cred, n_mid)
            selected += random.sample(low_cred, n_low)
            print(f"  [分層抽樣] High: {n_high}, Mid: {n_mid}, Low: {n_low}")
            
        random.shuffle(selected)
        return selected

    def generate_tsv(self, samples):
        """生成訓練用的 TSV 檔案，並處理 Caption 變體"""
        tsv_path = self.output_dir / f"phase2_{VARIANT}.tsv"
        print(f"📝 生成 TSV 檔案: {tsv_path.name}...")
        
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("id\tcaption\n")
            
            for idx, record in enumerate(samples):
                caption = record['representative_caption']['caption']
                score = record['credibility_analysis']['credibility_score']
                
                # 根據變體處理 Caption
                if VARIANT == "credibility_tag":
                    if score >= 0.8:
                        tag = "[High Credibility] "
                    elif score >= 0.6:
                        tag = "[Medium Credibility] "
                    else:
                        tag = "[Low Credibility] "
                    caption = tag + caption
                
                caption = caption.replace('\n', ' ').replace('\t', ' ').strip()
                f.write(f"{idx}\t{caption}\n")
                
        return tsv_path

    def _convert_single_audio(self, args):
        """處理單一音訊轉檔任務"""
        idx, record = args
        audio_path = Path(record['audio_path'])
        target_path = self.audio_dir / f"{idx}.wav"
        
        if target_path.exists() and target_path.stat().st_size > 0:
            return True, None
            
        if not audio_path.exists():
            return False, f"#{idx}: 來源檔案不存在 {audio_path.name}"
            
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-f", "wav", str(target_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return False, f"#{idx}: ffmpeg 錯誤"
        return True, None

    def prepare_audio_parallel(self, samples):
        """多執行緒並行轉檔 WAV"""
        print(f"🎵 準備音訊檔案 (多執行緒: {NUM_WORKERS})...")
        
        success_count = 0
        failed_files = []
        args_list = [(idx, record) for idx, record in enumerate(samples)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(self._convert_single_audio, args_list))
            
        for success, err_msg in results:
            if success:
                success_count += 1
            else:
                failed_files.append(err_msg)
                
        print(f"  成功: {success_count} / {len(samples)} | 失敗: {len(failed_files)}\n")
        return success_count

    def generate_commands(self, tsv_path):
        """生成預處理與訓練腳本"""
        script_path = self.output_dir / f"phase2_commands_{VARIANT}.sh"
        
        content = f"""#!/bin/bash
set -e

# ============================================
# Phase 2: {VARIANT} 變體執行指令
# ============================================
PHASE2_DIR="{self.output_dir}"
MEANAUDIO_DIR="$HOME/MeanAudio"
CAPTIONS_TSV="$PHASE2_DIR/{tsv_path.name}"
AUDIO_DIR="$PHASE2_DIR/audio"
OUTPUT_NPZ_DIR="$PHASE2_DIR/npz"
OUTPUT_PARTITION="$PHASE2_DIR/partition.tsv"
LATENT_DIR="$PHASE2_DIR/latents"

echo "切換到 MeanAudio 目錄並啟動環境..."
cd $MEANAUDIO_DIR
source $HOME/venvs/dac/bin/activate

# 1. 音訊分割 (Partition)
echo "執行 Partition..."
python training/partition_clips.py --data_dir $AUDIO_DIR --output_dir $OUTPUT_PARTITION

# 2. 提取特徵 (Extract Latents)
echo "執行特徵提取 (約需10-15分鐘)..."
export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \\
    --captions_tsv $CAPTIONS_TSV \\
    --data_dir $AUDIO_DIR \\
    --clips_tsv $OUTPUT_PARTITION \\
    --latent_dir $LATENT_DIR \\
    --output_dir $OUTPUT_NPZ_DIR \\
    --text_encoder='t5_clap'

echo "預處理完成！請使用以下指令啟動訓練："
echo "========================================"
echo "export CUDA_VISIBLE_DEVICES=0"
echo "cd $MEANAUDIO_DIR"
echo "torchrun --standalone --nproc_per_node=1 train.py \\\\"
echo "    --config-name train_config.yaml \\\\"
echo "    exp_id=phase2_{VARIANT} \\\\"
echo "    model=meanaudio_s \\\\"
echo "    batch_size=8 \\\\"
echo "    num_iterations=10000 \\\\"
echo "    val_interval=1000 \\\\"
echo "    save_checkpoint_interval=2000 \\\\"
echo "    mini_train=False \\\\"
echo "    data.AudioCaps_npz.tsv=$CAPTIONS_TSV \\\\"
echo "    data.AudioCaps_npz.npz_dir=$OUTPUT_NPZ_DIR \\\\"
echo "    data.AudioCaps_val_npz.tsv=$CAPTIONS_TSV \\\\"
echo "    data.AudioCaps_val_npz.npz_dir=$OUTPUT_NPZ_DIR \\\\"
echo "    ++use_rope=False \\\\"
echo "    ++use_wandb=True \\\\"
echo "    ++ema.enable=True"
"""
        with open(script_path, 'w') as f:
            f.write(content)
        script_path.chmod(0o755)
        print(f"📜 指令腳本已生成: {script_path}")

    def run(self):
        samples = self.load_and_sample()
        # 先生成臨時 TSV（用於預處理）
        tsv_path = self.generate_tsv(samples)
        # 準備音訊並執行預處理（會生成 NPZ）
        self.prepare_audio_parallel(samples)
        # 預處理完成後，根據實際 NPZ 數量重新生成 TSV
        print("\n📝 根據實際 NPZ 數量重新生成 TSV...")
        tsv_path = self.generate_tsv(samples)
        # 生成訓練指令
        self.generate_commands(tsv_path)
        print("\n✅ Phase 2 資料準備完成！")

if __name__ == "__main__":
    Phase2DataPreparation().run()