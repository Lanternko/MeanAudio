#!/bin/bash
# ============================================================
# babysit_fullpipe.sh — 獨立 tmux 中的 pipeline 保姆（不依賴 Claude session）
#
# 每 5 分鐘檢查：
#   - tmux fullpipe 是否活著
#   - GPU util / mem / process
#   - master log 最新 iter / loss / MARKER
#   - 任何 Traceback/CRITICAL/OOM/nan
# 寫入 dashboard 檔，供隨時 `cat` 查閱。若發現異常，在 dashboard 上留明顯 ALERT。
#
# 使用：
#   tmux new -d -s babysitter 'bash ~/MeanAudio/babysit_fullpipe.sh'
# ============================================================

DASHBOARD="$HOME/logs/fullpipe_dashboard.txt"
HIST="$HOME/logs/fullpipe_babysit_history.log"
INTERVAL=300   # 5 min

# 會自動找的 tmux session 名（第一個活的就追，順序 = 優先級）
CANDIDATES="p9v1_ablation p9v1_salvage fullpipe"

# 會自動找的 log file（按時間排序，最新的當 master）
find_latest_log() {
    ls -t "$HOME/logs/"phase9_v1*.log "$HOME/logs/"fullpipe_*.log 2>/dev/null \
        | grep -v "babysit\|dashboard" \
        | head -1
}

while true; do
    TS=$(date -Iseconds)

    # 找活著的 tmux session
    ACTIVE_TMUX=""
    for s in $CANDIDATES; do
        if tmux has-session -t "$s" 2>/dev/null; then
            ACTIVE_TMUX="$s"
            break
        fi
    done

    if [ -n "$ACTIVE_TMUX" ]; then
        TMUX_STATE="ALIVE ($ACTIVE_TMUX)"
    else
        TMUX_STATE="DEAD (none of: $CANDIDATES)"
    fi

    # 找最新 master log（追最活躍的）
    MASTER=$(find_latest_log)
    if [ -z "$MASTER" ]; then
        MASTER="(no log found)"
    fi
    # 同步更新 symlink 讓外部也能讀
    [ -f "$MASTER" ] && ln -sfn "$MASTER" "$HOME/logs/fullpipe_latest.log"

    # GPU
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d '% ')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    GPU_PROC=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 | tr '\n' '|')

    # log 尾 120 行，解最新資訊
    if [ -f "$MASTER" ]; then
        TAIL=$(tail -120 "$MASTER" | tr '\r' '\n')
    else
        TAIL=""
    fi

    # 最新 iter / loss（訓練中）
    LATEST_IT=$(echo "$TAIL" | grep -oE 'it +[0-9]+:.*loss:[0-9.]+' | tail -1)
    # 最新 MARKER — 整個 log 掃（tail 120 行可能全是 tqdm 進度條，markers 藏在前面）
    if [ -f "$MASTER" ]; then
        LATEST_MARKER=$(grep -E 'LANE_[A-Z0-9_]+|MC_EVAL_|fullpipe (START|FINISH)|\[Stage [12]\]|\[遷移\]|\[Pre-flight|\[Eval S[12]\]|訓練完成|移植完成|NPZ dir is valid|\[FAIL\]|\[WARN\]' "$MASTER" \
            | tail -1 \
            | grep -oE '(LANE_[A-Z0-9_]+|MC_EVAL_[A-Z]+ [a-z0-9_]+|fullpipe (START|FINISH)|\[Stage [12]\][^[]*|\[遷移\][^[]*|\[Pre-flight\][^[]*|\[Eval S[12]\][^[]*|訓練完成|移植完成|NPZ dir is valid[^[]*|\[FAIL\][^[]*|\[WARN\][^[]*)' \
            | head -c 100)
    else
        LATEST_MARKER=""
    fi
    # 最新 eval 進度百分比（若在 eval 階段）
    LATEST_EVAL_PROG=$(echo "$TAIL" | grep -oE '[0-9]+%\|[^|]+\| +[0-9]+/90063' | tail -1)
    [ -z "$LATEST_EVAL_PROG" ] && LATEST_EVAL_PROG=$(echo "$TAIL" | grep -oE '[0-9]+%\|[^|]+\| +[0-9]+/5521' | tail -1)
    # 最新 CLAP/AES 數字
    LATEST_CLAP=$(echo "$TAIL" | grep -E '^✅ CLAP Score:' | tail -1)
    LATEST_AES=$(echo "$TAIL" | grep -E '^ +aes_PQ:' | tail -1)

    # 異常偵測
    ALERT=""
    if [ -z "$ACTIVE_TMUX" ]; then
        ALERT="ALERT: no active pipeline tmux (tried: $CANDIDATES)"
    elif echo "$TAIL" | tail -40 | grep -qE 'Traceback|CRITICAL|CUDA out of memory|OOM|Killed|loss:[nN][aA][nN]'; then
        MATCH=$(echo "$TAIL" | tail -40 | grep -oE 'Traceback.*|CRITICAL.*|CUDA out of memory|OOM.*|Killed.*|loss:[nN][aA][nN].*' | head -1)
        ALERT="ALERT: crash signature detected — $MATCH"
    elif [ -n "$GPU_UTIL" ] && [ "$GPU_UTIL" -lt 10 ] && [ "$TMUX_STATE" = "ALIVE" ]; then
        # GPU 低於 10% 但 tmux alive — 可能卡住，但也可能是 phase boundary transient
        # 只有連續 2 次（10 min）低 util 才報警
        LAST_LOW=$(grep -c "GPU_LOW_UTIL" "$HIST" 2>/dev/null | tail -1)
        echo "$TS GPU_LOW_UTIL=$GPU_UTIL" >> "$HIST"
        RECENT_LOW=$(tail -3 "$HIST" 2>/dev/null | grep -c "GPU_LOW_UTIL")
        if [ "$RECENT_LOW" -ge 2 ]; then
            ALERT="ALERT: GPU util <10% for >=2 consecutive 5-min windows (${RECENT_LOW}× recent) — may be stuck"
        fi
    else
        # 清掉 GPU_LOW_UTIL 累積（util 正常）
        grep -v "GPU_LOW_UTIL" "$HIST" 2>/dev/null > "${HIST}.tmp" && mv "${HIST}.tmp" "$HIST" 2>/dev/null || true
    fi

    # 寫 dashboard (覆蓋，永遠顯示最新)
    {
        echo "============================================================"
        echo " Fullpipe dashboard  —  updated: $TS"
        echo "============================================================"
        echo ""
        echo "[STATE]"
        echo "  tmux fullpipe   : $TMUX_STATE"
        echo "  master log      : $MASTER"
        echo "  GPU util / mem  : ${GPU_UTIL:-?}% / ${GPU_MEM:-?}"
        echo "  GPU process     : ${GPU_PROC%|*|}"
        echo ""
        echo "[PROGRESS]"
        echo "  latest MARKER   : ${LATEST_MARKER:-(none yet)}"
        echo "  latest iter/loss: ${LATEST_IT:-(not in training)}"
        [ -n "$LATEST_EVAL_PROG" ] && echo "  latest eval prog: $LATEST_EVAL_PROG"
        [ -n "$LATEST_CLAP" ] && echo "  latest CLAP     : $LATEST_CLAP"
        [ -n "$LATEST_AES"  ] && echo "  latest aes_PQ   : $LATEST_AES"
        echo ""
        if [ -n "$ALERT" ]; then
            echo "================= 🚨 $ALERT ================="
            echo ""
            echo "[last 15 log lines]"
            echo "$TAIL" | tail -15
        else
            echo "[status] healthy — next check in ${INTERVAL}s"
        fi
    } > "$DASHBOARD"

    # History 記錄 (一行 per check, 給 debugging 用)
    echo "$TS state=$TMUX_STATE gpu=${GPU_UTIL:-?}% marker=${LATEST_MARKER:-none} alert=${ALERT:-none}" >> "$HIST"

    sleep "$INTERVAL"
done
