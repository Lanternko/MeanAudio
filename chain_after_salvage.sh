#!/bin/bash
# ============================================================
# chain_after_salvage.sh — 等 p9v1_salvage 結束後自動接 ablation
#
# 用法（在新 tmux 跑，跟 salvage 並存不互衝）：
#   tmux new -d -s p9v1_chain 'bash ~/MeanAudio/chain_after_salvage.sh'
# ============================================================

set -o pipefail

LOG="$HOME/logs/chain_after_salvage.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "===== chain start ====="
log "等待 tmux p9v1_salvage 結束..."

WAITED=0
until ! tmux has-session -t p9v1_salvage 2>/dev/null; do
    sleep 60
    WAITED=$((WAITED + 60))
    # 每 30 min 報一次還活著
    if [ $((WAITED % 1800)) -eq 0 ]; then
        log "...等待中（已 $((WAITED/60)) min）"
    fi
done

log "p9v1_salvage tmux 已結束（總等待 $((WAITED/60)) min）"

# 等 GPU 真的釋放（process 退出可能延遲）
log "等 GPU process 釋放..."
GPU_WAIT=0
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
    GPU_WAIT=$((GPU_WAIT + 30))
    if [ $GPU_WAIT -ge 600 ]; then
        log "[FAIL] GPU 10 min 後仍未釋放，abort chain"
        exit 1
    fi
done
log "[OK] GPU idle，準備啟動 ablation"

# 確認 salvage 跑成功還是失敗（讀 metrics file）
SALVAGE_METRICS="$HOME/MeanAudio/eval_output/metrics/phase9_v1_salvage_stage2_200000_no_q/metrics.txt"
if [ -f "$SALVAGE_METRICS" ]; then
    SALVAGE_CLAP=$(grep -E '^clap_score:' "$SALVAGE_METRICS" | awk '{print $2}')
    log "Salvage 完成，CLAP=$SALVAGE_CLAP"
    log "  ≥0.15 → S1 multi 仍 OK，S2 multi 是元凶"
    log "  <0.15 → S1 multi 也壞，需 control 組"
else
    log "[WARN] Salvage metrics 不存在 — 可能 salvage 失敗或被中止，仍繼續 ablation"
fi

# 啟動 ablation
log "===== 啟動 ablation: p9v1_ablation tmux ====="
tmux new -d -s p9v1_ablation 'bash $HOME/MeanAudio/train_pipeline_phase9_v1_ablation_s1fixed_s2multi.sh'

log "===== ablation 已 detach 進 tmux p9v1_ablation，chain 結束 ====="
