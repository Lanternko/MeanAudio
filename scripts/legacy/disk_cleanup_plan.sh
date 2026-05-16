#!/bin/bash
# ============================================================
# disk_cleanup_plan.sh — HDD 緊急清理預案
#
# 產出日期：2026-04-18 09:15，Phase 9 V1 Stage 1 訓練中
# 現況：/mnt/HDD 419 GB free, 95% 用量
# 預估：Lane B V1+V2 訓練 + eval 還會用 ~125 GB，最後剩 ~294 GB
#       ⇒ 安全，但緊急時此腳本可立即釋放 73 GB（tier A）或 147 GB（tier A+B）
#
# 使用：
#   bash disk_cleanup_plan.sh --tier A        # 只刪安全類
#   bash disk_cleanup_plan.sh --tier A,B      # 加上需確認類
#   bash disk_cleanup_plan.sh --dry           # 只列出會刪什麼，不動
# ============================================================

TIER="${1:-A}"
DRY=0
if [[ "$*" == *"--dry"* ]]; then DRY=1; fi
if [[ "$*" == *"--tier"* ]]; then TIER="$2"; fi

echo "================================================"
echo " HDD cleanup  —  tier=$TIER dry=$DRY"
echo " 執行前 free:"; df -h /mnt/HDD | tail -1
echo "================================================"
echo ""

# 共用 rm wrapper（dry 模式只印）
do_rm() {
    local target="$1"
    local reason="$2"
    if [ ! -e "$target" ]; then
        echo "  [SKIP] $target 已不存在"
        return
    fi
    local sz
    sz=$(du -sh "$target" 2>/dev/null | awk '{print $1}')
    if [ "$DRY" = "1" ]; then
        echo "  [DRY] $sz  $target  ($reason)"
    else
        echo "  [RM]  $sz  $target  ($reason)"
        rm -rf "$target"
    fi
}

if [[ "$TIER" == *"A"* ]]; then
    echo "[tier A — 完全安全，無副作用]"
    # phase8_v4 在 CLAUDE.md 明確標示 ❌ 廢棄 (Qwen2-Audio 只有 1 cap/clip，不支援 true random)
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase8_v4_stage1_400000" \
          "phase 8 v4 廢棄（CLAUDE.md 明示）"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase8_v4_stage2_200000" \
          "phase 8 v4 廢棄（CLAUDE.md 明示）"

    # HDD 上那個 stale phase9_multicap_npz（只 1,219 檔，真正訓練用的在 ~/phase9_multicap_npz/）
    do_rm "/mnt/HDD/kojiek/phase4_jamendo_data/phase9_multicap_npz" \
          "stale dup，真正 NPZ dir 在 ~/phase9_multicap_npz/（251K 檔）"

    # phase7_v1 (目前最佳) 的中間 EMA ckpts — S2 已完成，ema_final 保留，中間 redundant
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage1_400000/ema_ckpts" \
          "S1 中間 EMA；S2 已完成，留 ema_final.pth 即可"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage1_400000/phase7_v1_stage1_400000_ckpt_last.pth" \
          "S1 resume ckpt；S2 已完成，不再 resume"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage1_400000/phase7_v1_stage1_400000_ckpt_shadow.pth" \
          "S1 resume shadow；同上"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage2_200000/ema_ckpts" \
          "S2 中間 EMA；留 ema_final.pth 即可（目前最佳模型本體）"
    echo ""
fi

if [[ "$TIER" == *"B"* ]]; then
    echo "[tier B — 需人工確認]"
    # phase8_v5 不在 CLAUDE.md phase table，可能是 abandoned test
    echo "  # ⚠️  phase8_v5 不在 CLAUDE.md；請先確認不需要再動手"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase8_v5_stage1_400000" \
          "phase 8 v5 可能 abandoned（不在 CLAUDE.md 表）"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase8_v5_stage2_200000" \
          "同上"
    # phase7_v1 S2 的 ckpt_last/shadow — 需 resume 才需要，ema_final 足以 eval
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage2_200000/phase7_v1_stage2_200000_ckpt_last.pth" \
          "S2 resume ckpt；不再 resume 即可刪"
    do_rm "/mnt/HDD/kojiek/MeanAudio_exps/phase7_v1_stage2_200000/phase7_v1_stage2_200000_ckpt_shadow.pth" \
          "S2 resume shadow；同上"
    echo ""
fi

if [[ "$TIER" == *"D"* ]]; then
    echo "[tier D — 巨獎，需極度謹慎確認]"
    echo "  # wav_audio/ = 412 GB, 421,401 個 16kHz mono WAV 快取"
    echo "  # 真正訓練用的是 symlink 到 /home/hsiehyian/.../segments_no_vocals/ 的 mp3"
    echo "  # grep 掃過 ~/MeanAudio 和 ~/research 都沒有任何腳本引用 wav_audio/"
    echo "  # 推測：2026-03 某次實驗做的預處理快取，之後改用 on-the-fly decode 不再需要"
    echo "  # ⚠️  執行前請至少 spot-check 一個：和 mp3 版比對，確認音訊內容重複"
    do_rm "/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio" \
          "412 GB 舊 WAV 快取；真正訓練用 mp3 symlink，未被引用"
    echo ""
fi

if [[ "$TIER" == *"C"* ]]; then
    echo "[tier C — 舊 log 歸檔]"
    # 超過 7 天的舊 log 打包到 archives 後刪
    if [ "$DRY" = "1" ]; then
        echo "  [DRY] 會 tar ~/logs 中 >7 天的 log 到 ~/archives/ 然後刪原檔"
        find "$HOME/logs" -type f -mtime +7 2>/dev/null | head -5
    else
        mkdir -p "$HOME/archives/old_logs"
        ARCHIVE="$HOME/archives/old_logs/logs_pre_$(date +%Y%m%d).tar.gz"
        find "$HOME/logs" -type f -mtime +7 -print0 2>/dev/null | \
            tar -czf "$ARCHIVE" --null -T - 2>/dev/null && \
            find "$HOME/logs" -type f -mtime +7 -delete 2>/dev/null
        echo "  [OK] 歸檔至 $ARCHIVE"
    fi
    echo ""
fi

echo "================================================"
echo " 執行後 free:"; df -h /mnt/HDD | tail -1
echo "================================================"
