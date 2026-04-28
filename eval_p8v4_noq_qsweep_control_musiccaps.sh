#!/bin/bash
# ============================================================
# P8 V4 NoQ q-sweep control on MusicCaps (q=5..9, dual-ref)
# eval_p8v4_noq_qsweep_control_musiccaps.sh
#
# 用途：Wei-Jaw Fig.2 + Codex Round 3 P1 後續：用 bug-free
#       P8 V4 NoQ ckpt 跑乾淨 control，q=5..9 全部都是 random
#       (P8 V4 訓練時 null token 訓在 q[10]，本腳本 preflight 驗證)。
#
# 對照 P8 NoQ qsweep（Apr 2，bug 在 → q[9] 是 trained null）
# vs 本實驗（Apr 26，bug 已修 → q[5..9] 全 random）。
#
# Codex Round 4 fixes:
#   P1: Audio identity manifest (.gen_manifest.json)
#       - count check 不夠，加 ckpt_sha/gen_tsv/q/cfg 全 manifest
#       - mismatch → abort (不 silent corrupt stale audio)
#   P2: Preflight q_embed diff（驗證 q[5..9] random、q[10] trained）
#   P2: --no_q baseline 在同腳本跑（不靠 hardcoded 0.0665/0.0571）
#   P2: Marker 包含完整 eval identity (q, cfg, ref TSVs, eval mode)
#
# Caveat：P8 V4 訓練帶 [consistency=0.90] prefix，CLAP 數量級
#       在 0.06-0.07，不是 P8 的 0.18。**不能跟 P7 V1 同圖比**，
#       只能作為 P8 V4 NoQ 內部「trained null vs random q」對照
#       (paper supplementary)。
#
# 預估：preflight ~30s + baseline ~3min + 5 q × ~13 min = ~70 min
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase8_v4_stage2_200000"
EXP_S1="phase8_v4_stage1_400000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
S1_EMA="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
GEN_TSV="$DATA_DIR/phase8_v4_musiccaps_test.tsv"      # prefixed gen TSV (s=0.90)
NATURAL_TSV="$DATA_DIR/musiccaps_test.tsv"            # natural ref TSV (no prefix)
NUM_SAMPLES=5521
CFG=0.5
SCRIPT_VERSION="codex_round4_v1"

if [ ! -f "$EMA" ]; then echo "❌ EMA 不存在：$EMA"; exit 1; fi
if [ ! -f "$S1_EMA" ]; then echo "❌ S1 EMA 不存在：$S1_EMA"; exit 1; fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

# Identity hashes
ckpt_mtime=$(stat -c %Y "$EMA")
ckpt_sha=$(sha256sum "$EMA" | awk '{print $1}' | head -c 16)
gen_tsv_sha=$(sha256sum "$GEN_TSV" | awk '{print $1}' | head -c 16)
natural_tsv_sha=$(sha256sum "$NATURAL_TSV" | awk '{print $1}' | head -c 16)

PREFLIGHT_LOG="$LOG_DIR/${EXP}_qsweep_control_preflight.log"

echo "======================================================"
echo "  P8 V4 NoQ q-sweep control on MusicCaps (q=5..9, dual-ref)"
echo "  EMA           : $EMA"
echo "  ckpt mtime/sha: $ckpt_mtime / $ckpt_sha"
echo "  GEN TSV       : $GEN_TSV (sha16 $gen_tsv_sha)"
echo "  NATURAL TSV   : $NATURAL_TSV (sha16 $natural_tsv_sha)"
echo "  cfg=$CFG  num_samples=$NUM_SAMPLES  script_version=$SCRIPT_VERSION"
echo "======================================================"

# ============================================================
# Codex P2: Preflight q_embed row-diff verification
# ============================================================
echo
echo "[Preflight] q_embed S1 vs S2 row-diff verification..." | tee "$PREFLIGHT_LOG"
python - <<PY 2>&1 | tee -a "$PREFLIGHT_LOG"
import torch
from pathlib import Path

s1 = Path("$S1_EMA")
s2 = Path("$EMA")

def load_q(p):
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt.get("weights", ckpt) if isinstance(ckpt, dict) else ckpt
    for k, v in sd.items():
        if "q_embed.weight" in k:
            return v.float()
    raise KeyError("q_embed.weight not found")

q1, q2 = load_q(s1), load_q(s2)
print(f"{'row':>3} | {'delta_l2':>10} | status")

# Verify expected pattern: q[10] TRAINED, q[0..9] untouched
expected = {i: ("untouched" if i != 10 else "TRAINED") for i in range(11)}
all_ok = True
for i in range(11):
    delta = (q2[i] - q1[i]).norm().item()
    actual = "TRAINED" if delta > 0.5 else ("untouched" if delta < 0.001 else "ambiguous")
    ok = (actual == expected[i])
    flag = "✅" if ok else "❌"
    print(f"{i:>3} | {delta:>10.6f} | {actual:>10} (expect {expected[i]}) {flag}")
    if not ok:
        all_ok = False

if not all_ok:
    print("❌ Preflight FAILED: q_embed pattern not as expected (P8 V4 should have q[10] trained, q[0..9] untouched)")
    raise SystemExit(1)
print("✅ Preflight PASSED: q[5..9] random (untouched), q[10] trained (null)")
PY

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Preflight verification failed; aborting"
    exit 1
fi

# ============================================================
# Codex P2: Same-pipeline --no_q baseline (trained q[10])
# ============================================================
BASELINE_OUT="$WORK_DIR/eval_output/${EXP}_no_q_qsweep_baseline_musiccaps"
BASELINE_BASE="${EXP}_no_q_qsweep_baseline_musiccaps"
BASELINE_PREFIXED_METRIC="$WORK_DIR/eval_output/metrics/${BASELINE_BASE}_prefixed_ref/metrics.txt"
BASELINE_NATURAL_METRIC="$WORK_DIR/eval_output/metrics/${BASELINE_BASE}_natural_ref/metrics.txt"
BASELINE_MARKER="$WORK_DIR/eval_output/metrics/${BASELINE_BASE}_prefixed_ref/.run_manifest"
BASELINE_LOG="$LOG_DIR/${EXP}_no_q_qsweep_baseline_musiccaps_eval.log"

baseline_marker_str=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s' \
    "$EMA" "$GEN_TSV" "$NATURAL_TSV" "$NUM_SAMPLES" "$CFG" "no_q" \
    "$ckpt_mtime" "$ckpt_sha" "$gen_tsv_sha" "$natural_tsv_sha")

baseline_audio_manifest="$BASELINE_OUT/audio/.gen_manifest.json"
baseline_audio_id=$(printf '{"ckpt_sha":"%s","gen_tsv_sha":"%s","q":"no_q","cfg":"%s","num_samples":%s,"script_version":"%s"}' \
    "$ckpt_sha" "$gen_tsv_sha" "$CFG" "$NUM_SAMPLES" "$SCRIPT_VERSION")

echo
echo "[#0 baseline --no_q] same-pipeline rerun"

if [ -f "$BASELINE_PREFIXED_METRIC" ] && [ -f "$BASELINE_NATURAL_METRIC" ] && [ -f "$BASELINE_MARKER" ]; then
    if [ "$(cat $BASELINE_MARKER)" = "$baseline_marker_str" ]; then
        echo "  ✅ baseline metrics 已存在且 marker match，skip"
    fi
fi

if [ ! -f "$BASELINE_PREFIXED_METRIC" ] || [ "$(cat $BASELINE_MARKER 2>/dev/null)" != "$baseline_marker_str" ]; then
    # Codex P1: audio identity check
    if [ -d "$BASELINE_OUT/audio" ]; then
        n=$(ls -1 $BASELINE_OUT/audio/*.flac 2>/dev/null | wc -l)
        if [ -f "$baseline_audio_manifest" ] && [ "$n" -eq "$NUM_SAMPLES" ]; then
            if [ "$(cat $baseline_audio_manifest)" = "$baseline_audio_id" ]; then
                echo "  audio 完整且 identity match，skip gen"
            else
                echo "  ❌ audio identity mismatch (manifest != expected); aborting (避免 stale audio)"
                exit 1
            fi
        elif [ -d "$BASELINE_OUT/audio" ]; then
            echo "  ❌ audio dir 存在但無 manifest 或 count 不對; aborting"
            exit 1
        fi
    else
        echo "  gen → $BASELINE_OUT (--no_q)"
        python eval.py --variant "meanaudio_s" --model_path "$EMA" \
            --output "$BASELINE_OUT/audio" --tsv "$GEN_TSV" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength $CFG --no_q --full_precision \
            2>&1 | tee "$BASELINE_LOG"
        echo "$baseline_audio_id" > "$baseline_audio_manifest"
    fi

    # dual-ref metrics
    python "$EVAL_SCRIPT" --gen_dir "$BASELINE_OUT/audio" --tsv "$GEN_TSV" \
        --exp_name "${BASELINE_BASE}_prefixed_ref" --num_samples $NUM_SAMPLES \
        2>&1 | tee -a "$BASELINE_LOG"
    python "$EVAL_SCRIPT" --gen_dir "$BASELINE_OUT/audio" --tsv "$NATURAL_TSV" \
        --exp_name "${BASELINE_BASE}_natural_ref" --num_samples $NUM_SAMPLES \
        2>&1 | tee -a "$BASELINE_LOG"

    mkdir -p "$(dirname $BASELINE_MARKER)"
    echo -n "$baseline_marker_str" > "$BASELINE_MARKER"
fi

baseline_clap_p=$(grep clap_score "$BASELINE_PREFIXED_METRIC" 2>/dev/null | awk '{print $2}')
baseline_clap_n=$(grep clap_score "$BASELINE_NATURAL_METRIC" 2>/dev/null | awk '{print $2}')
echo "  Baseline (--no_q, trained q[10]):  prefixed=$baseline_clap_p  natural=$baseline_clap_n"

# ============================================================
# q-sweep control with full identity manifest
# ============================================================
for Q in 5 6 7 8 9; do
    EVAL_OUT="$WORK_DIR/eval_output/${EXP}_q${Q}_musiccaps_qsweep_control"
    LOG="$LOG_DIR/${EXP}_q${Q}_musiccaps_qsweep_control_eval.log"
    BASE="${EXP}_q${Q}_musiccaps_qsweep_control"

    PREFIXED_METRIC="$WORK_DIR/eval_output/metrics/${BASE}_prefixed_ref/metrics.txt"
    NATURAL_METRIC="$WORK_DIR/eval_output/metrics/${BASE}_natural_ref/metrics.txt"
    MARKER="$WORK_DIR/eval_output/metrics/${BASE}_prefixed_ref/.run_manifest"

    # Codex P2: full eval identity in marker
    expected_marker=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s' \
        "$EMA" "$GEN_TSV" "$NATURAL_TSV" "$NUM_SAMPLES" "$CFG" "q=$Q" \
        "$ckpt_mtime" "$ckpt_sha" "$gen_tsv_sha" "$natural_tsv_sha")

    if [ -f "$PREFIXED_METRIC" ] && [ -f "$NATURAL_METRIC" ] && [ -f "$MARKER" ]; then
        if [ "$(cat $MARKER)" = "$expected_marker" ]; then
            echo "[q=$Q] ✅ dual-ref metrics 已存在且 marker match，skip"
            continue
        else
            echo "[q=$Q] ⚠️  marker mismatch, 重跑"
        fi
    fi

    # Codex P1: audio identity manifest
    audio_manifest="$EVAL_OUT/audio/.gen_manifest.json"
    expected_audio_id=$(printf '{"ckpt_sha":"%s","gen_tsv_sha":"%s","q":%s,"cfg":"%s","num_samples":%s,"script_version":"%s"}' \
        "$ckpt_sha" "$gen_tsv_sha" "$Q" "$CFG" "$NUM_SAMPLES" "$SCRIPT_VERSION")

    if [ -d "$EVAL_OUT/audio" ]; then
        n=$(ls -1 $EVAL_OUT/audio/*.flac 2>/dev/null | wc -l)
        if [ -f "$audio_manifest" ] && [ "$n" -eq "$NUM_SAMPLES" ]; then
            if [ "$(cat $audio_manifest)" = "$expected_audio_id" ]; then
                echo "[q=$Q] audio 完整且 identity match，skip gen"
            else
                echo "[q=$Q] ❌ audio identity mismatch, aborting"
                echo "  expected: $expected_audio_id"
                echo "  actual  : $(cat $audio_manifest)"
                exit 1
            fi
        else
            echo "[q=$Q] ❌ audio dir 存在但無 manifest 或 count=$n != $NUM_SAMPLES, aborting (避免 stale audio)"
            exit 1
        fi
    else
        echo "[q=${Q}] gen → $EVAL_OUT"
        python eval.py \
            --variant "meanaudio_s" --model_path "$EMA" \
            --output "$EVAL_OUT/audio" --tsv "$GEN_TSV" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength $CFG --quality_level $Q --full_precision \
            2>&1 | tee "$LOG"
        echo "$expected_audio_id" > "$audio_manifest"
    fi

    # dual-ref metric pass
    echo "[Metric / prefixed_ref] $BASE" | tee -a "$LOG"
    python "$EVAL_SCRIPT" --gen_dir "$EVAL_OUT/audio" --tsv "$GEN_TSV" \
        --exp_name "${BASE}_prefixed_ref" --num_samples $NUM_SAMPLES \
        2>&1 | tee -a "$LOG"
    echo "[Metric / natural_ref] $BASE" | tee -a "$LOG"
    python "$EVAL_SCRIPT" --gen_dir "$EVAL_OUT/audio" --tsv "$NATURAL_TSV" \
        --exp_name "${BASE}_natural_ref" --num_samples $NUM_SAMPLES \
        2>&1 | tee -a "$LOG"

    mkdir -p "$(dirname $MARKER)"
    echo -n "$expected_marker" > "$MARKER"
done

echo "======================================================"
echo "  P8 V4 NoQ q-sweep control 完成（q=5..9 × MusicCaps n=5521 dual-ref）"
echo "  Baseline (--no_q, trained q[10]):"
echo "    prefixed=$baseline_clap_p  natural=$baseline_clap_n"
echo "  q=5..9 results in eval_output/metrics/${EXP}_q{5..9}_musiccaps_qsweep_control_{prefixed,natural}_ref/"
echo "======================================================"
