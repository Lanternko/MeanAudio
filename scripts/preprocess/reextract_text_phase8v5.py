"""
re-extract text embeddings only for phase8_v5.

舊 NPZ 的 mean/std（audio latent）不變，只重新 encode caption 取 text_features / text_features_c。
不需要 WAV 檔、不需要 VAE，只跑 T5 + LAION-CLAP。

用法：
    source ~/venvs/dac/bin/activate
    cd ~/MeanAudio
    python reextract_text_phase8v5.py \
        [--batch_size 64] [--num_workers 4] [--dry_run]
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, T5EncoderModel
import laion_clap

# ── 固定路徑 ──────────────────────────────────────────────────
OLD_NPZ_DIR   = Path("/home/kojiek/research/meanaudio_training/npz")
NEW_NPZ_DIR   = Path("/home/kojiek/research/meanaudio_training/npz_phase8v5")
TRAIN_TSV     = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v5_train.tsv")
GT_CACHE      = Path("/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt")
CLAP_CKPT     = Path("./weights/music_speech_audioset_epoch_15_esc_89.98.pt")
T5_MODEL_NAME = "google/flan-t5-large"
MAX_TEXT_LEN  = 128   # 與訓練時一致

def encode_t5(tokenizer, model, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_TEXT_LEN).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state   # [B, seq, 1024]
    # 補零/截斷到固定長度 77（與現有 NPZ 一致）
    seq_len = 77
    B, L, D = out.shape
    if L < seq_len:
        pad = torch.zeros(B, seq_len - L, D, device=device)
        out = torch.cat([out, pad], dim=1)
    else:
        out = out[:, :seq_len, :]
    return out.cpu().float().numpy()   # [B, 77, 1024]

def encode_clap(model, texts):
    with torch.no_grad():
        emb = model.get_text_embedding(texts, use_tensor=True)  # [B, 512]
    return emb.cpu().float().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true",
                        help="只處理前 100 筆，用於快速驗證")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 讀取 TSV 和 gt_cache ──────────────────────────────────
    df = pd.read_csv(TRAIN_TSV, sep="\t")
    print(f"Train TSV: {len(df):,} rows")

    with open(GT_CACHE) as f:
        npz_files = [line.strip() for line in f if line.strip()]
    assert len(npz_files) == len(df), \
        f"gt_cache 行數 ({len(npz_files)}) != TSV 行數 ({len(df)})"
    print(f"gt_cache: {len(npz_files):,} entries")

    if args.dry_run:
        df = df.iloc[:100]
        npz_files = npz_files[:100]
        print("DRY RUN: 只處理前 100 筆")

    # ── 載入 text encoder ────────────────────────────────────
    print("Loading T5...")
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    t5_model = T5EncoderModel.from_pretrained(T5_MODEL_NAME).eval().to(device)

    print("Loading CLAP...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    clap_model.load_ckpt(str(CLAP_CKPT), verbose=False)
    clap_model = clap_model.to(device)

    # ── 準備輸出目錄 ──────────────────────────────────────────
    NEW_NPZ_DIR.mkdir(parents=True, exist_ok=True)

    # ── 主迴圈 ───────────────────────────────────────────────
    captions = df["caption"].tolist()
    total = len(captions)
    bs = args.batch_size
    skipped = 0

    for start in tqdm(range(0, total, bs), desc="Re-encoding text"):
        end = min(start + bs, total)
        batch_captions = captions[start:end]
        batch_npz_files = npz_files[start:end]

        # text encoding
        text_features   = encode_t5(t5_tokenizer, t5_model, batch_captions, device)
        text_features_c = encode_clap(clap_model, batch_captions)

        # 讀舊 NPZ mean/std，存新 NPZ
        for i, npz_fname in enumerate(batch_npz_files):
            old_path = OLD_NPZ_DIR / npz_fname
            new_path = NEW_NPZ_DIR / npz_fname

            if new_path.exists():
                skipped += 1
                continue

            old_data = np.load(old_path)
            np.savez(
                new_path,
                mean=old_data["mean"],
                std=old_data["std"],
                text_features=text_features[i],
                text_features_c=text_features_c[i],
            )

    print(f"\nDone. Output: {NEW_NPZ_DIR}")
    print(f"  Total: {total:,}  Skipped (already exists): {skipped:,}")

    # ── 抽樣驗證 ─────────────────────────────────────────────
    print("\n[Validation] 抽樣比對 5 筆...")
    for idx in [0, 1, 100, 1000, 10000]:
        if idx >= len(npz_files):
            break
        old = np.load(OLD_NPZ_DIR / npz_files[idx])
        new = np.load(NEW_NPZ_DIR / npz_files[idx])
        mean_match = np.allclose(old["mean"], new["mean"])
        std_match  = np.allclose(old["std"],  new["std"])
        text_same  = np.allclose(old["text_features"], new["text_features"])
        print(f"  [{idx:>6}] mean/std unchanged={mean_match and std_match}  "
              f"text_changed={not text_same}  "
              f"caption: {df['caption'].iloc[idx][:60]}...")

if __name__ == "__main__":
    main()
