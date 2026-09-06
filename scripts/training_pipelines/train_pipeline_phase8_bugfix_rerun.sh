#!/bin/bash
# ============================================================
# MeanAudio Phase 8 — BUG-FIX RERUN (full S1+S2 retrain)
# train_pipeline_phase8_bugfix_rerun.sh
#
# 目的：單 caption 情境下，驗證 bug fix 對 no-Q 模型的影響
#       用於判讀 P9 V1 bug-fix rerun 的結果是否「multi_cap 特有」
#
# 注意：Phase 8 的原始 S1 ckpt_last.pth 已不存在（只剩 _ema_final.pth），
#       因此必須 S1 + S2 都從頭重訓。整個 pipeline ~19-20 hr。
#       如果 P9 V1 bug-fix rerun 結果已足夠說明問題（CLAP 大幅恢復），
#       此腳本可以跳過，節省 12 hr S1 訓練時間。
#
#   Bug #1 (critical): networks.py MeanAudio.forward/ode_wrapper
#                      q=None → 填 9（應為 10 null token）
#   Bug #2 (minor):    runner_meanflow.py text_f_undrop = text_f 別名
#
# 設計：
#   - S1: 從頭訓練 400k iter（原 ckpt 遺失，code 已修）
#         single-cap NPZ (phase7_v1 random seed=42 已寫入 TSV)
#   - S2: 從頭訓練 200k iter，single-cap NPZ，no-Q
#   - Eval: MusicCaps (primary)、Jamendo (optional)
#
# 基線對照（修前）：
#   Jamendo:   CLAP 0.1851
#   MusicCaps: CLAP 0.1851, CE 5.91, CU 6.75, PC 4.98, PQ 6.54
#
# 判讀標準（與修前對比）：
#   MusicCaps CLAP 提升 → bug fix 對 no-Q 有正向影響
#   MusicCaps CLAP 持平 → bug 在 single-cap 情境影響不顯著
#   MusicCaps CLAP 下降 → 修錯了、或 code 其他變動污染
#
# 使用方式：
#   tmux new -s phase8_bugfix
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash train_pipeline_phase8_bugfix_rerun.sh
# ============================================================

set -euo pipefail

# ============================================================
# 實驗參數
# ============================================================

EXP_PREFIX="${EXP_PREFIX:-phase8_bugfix_rerun}"

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
S1_ITERATIONS="${S1_ITERATIONS:-400000}"
S2_ITERATIONS="${S2_ITERATIONS:-200000}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LINEAR_WARMUP_STEPS="${LINEAR_WARMUP_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_WEIGHTS_INTERVAL="${SAVE_WEIGHTS_INTERVAL:-10000}"
SAVE_CHECKPOINT_INTERVAL="${SAVE_CHECKPOINT_INTERVAL:-20000}"
EMA_CHECKPOINT_INTERVAL="${EMA_CHECKPOINT_INTERVAL:-10000}"
EXPECTED_ROWS="${EXPECTED_ROWS:-251599}"
LEGACY_Q_DEFAULT="${USE_Q_CONDITIONING:-false}"
S1_USE_Q_CONDITIONING="${S1_USE_Q_CONDITIONING:-$LEGACY_Q_DEFAULT}"
S2_USE_Q_CONDITIONING="${S2_USE_Q_CONDITIONING:-$LEGACY_Q_DEFAULT}"
USE_TEXT_ATTENTION_MASK="${USE_TEXT_ATTENTION_MASK:-true}"
EXPERIMENT_REGIME="${EXPERIMENT_REGIME:-custom}"
EXPERIMENT_RUN_MODE="${EXPERIMENT_RUN_MODE:-resume}"
EVAL_Q_MODE="${EVAL_Q_MODE:-auto}"

RUN_PRIMARY_EVAL="${RUN_PRIMARY_EVAL:-true}"
RUN_JAMENDO_EVAL="${RUN_JAMENDO_EVAL:-false}"     # true 會多跑 ~3 hr
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-2048}"
EVAL_SKIP_AES="${EVAL_SKIP_AES:-false}"
TRAIN_SEED="${TRAIN_SEED:-}"

# ============================================================
# 固定路徑
# ============================================================

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_CKPT="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

TRAIN_TSV="${TRAIN_TSV:-$DATA_DIR/_QUARANTINED_phase7_v1_train.tsv}"
GT_CACHE="${GT_CACHE:-$DATA_DIR/npz_cache_train.txt}"
SINGLECAP_NPZ="${SINGLECAP_NPZ:-$HOME/research/meanaudio_training/npz_phase7_clean}"
NPZ_MANIFEST="${NPZ_MANIFEST:-$SINGLECAP_NPZ/MANIFEST.tsv}"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

TSV_MUSICCAPS="${TSV_MUSICCAPS:-$DATA_DIR/musiccaps_test.tsv}"
TSV_JAMENDO="${TSV_JAMENDO:-$DATA_DIR/phase4_test.tsv}"

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))

for bool_var in S1_USE_Q_CONDITIONING S2_USE_Q_CONDITIONING \
                USE_TEXT_ATTENTION_MASK RUN_PRIMARY_EVAL RUN_JAMENDO_EVAL \
                EVAL_SKIP_AES; do
    case "${!bool_var}" in
        true|false) ;;
        *) echo "[FAIL] $bool_var must be true or false (got ${!bool_var})" >&2; exit 2 ;;
    esac
done

case "$EXPERIMENT_RUN_MODE" in
    fresh|resume) ;;
    *) echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2; exit 2 ;;
esac

case "$EVAL_Q_MODE" in
    auto)
        if [ "$S2_USE_Q_CONDITIONING" = "true" ]; then
            RESOLVED_EVAL_Q_MODE=q9
        else
            RESOLVED_EVAL_Q_MODE=no_q
        fi
        ;;
    no_q|q9) RESOLVED_EVAL_Q_MODE="$EVAL_Q_MODE" ;;
    *) echo "[FAIL] EVAL_Q_MODE must be auto, no_q, or q9" >&2; exit 2 ;;
esac

# Fail closed for the paper's clean catalog-matched NoQ experiment.  A single
# global Q flag is deliberately insufficient here: both stages and eval must be
# stated and checked independently so a future wrapper cannot silently change
# only one side of the train/eval contract.
if [ "$EXPERIMENT_REGIME" = "clean_noq" ]; then
    if [ "$S1_USE_Q_CONDITIONING" != "false" ] || \
       [ "$S2_USE_Q_CONDITIONING" != "false" ] || \
       [ "$USE_TEXT_ATTENTION_MASK" != "false" ] || \
       [ "$RESOLVED_EVAL_Q_MODE" != "no_q" ]; then
        echo "[FAIL] clean_noq contract violated" >&2
        echo "  required: S1_Q=false S2_Q=false text_mask=false eval_q=no_q" >&2
        echo "  actual  : S1_Q=$S1_USE_Q_CONDITIONING S2_Q=$S2_USE_Q_CONDITIONING text_mask=$USE_TEXT_ATTENTION_MASK eval_q=$RESOLVED_EVAL_Q_MODE" >&2
        exit 2
    fi
fi

COMMON_ARGS=(
    batch_size=$BATCH_SIZE
    +accumulation_steps=$ACCUM_STEPS
    learning_rate=$LEARNING_RATE
    linear_warmup_steps=$LINEAR_WARMUP_STEPS
    num_workers=$NUM_WORKERS
    save_weights_interval=$SAVE_WEIGHTS_INTERVAL
    save_checkpoint_interval=$SAVE_CHECKPOINT_INTERVAL
    "++ema.checkpoint_every=$EMA_CHECKPOINT_INTERVAL"
    +use_rope=False
    +use_wandb=False
    "+use_text_attention_mask=$USE_TEXT_ATTENTION_MASK"
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$TRAIN_TSV"
    "+data.AudioCaps_npz.gt_cache=$GT_CACHE"
    # Validation is disabled; reuse the aligned train dataset so no stale
    # quarantined validation cache is instantiated.
    "data.AudioCaps_val_npz.tsv=$TRAIN_TSV"
    "data.AudioCaps_val_npz.gt_cache=$GT_CACHE"
    "++data.AudioCaps_npz.npz_dir=$SINGLECAP_NPZ"
    "++data.AudioCaps_val_npz.npz_dir=$SINGLECAP_NPZ"
    "++multi_cap=False"
)
if [ -n "$TRAIN_SEED" ]; then
    COMMON_ARGS+=("seed=$TRAIN_SEED")
fi

cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

if [ "$EXPERIMENT_RUN_MODE" = "fresh" ]; then
    FRESH_CONFLICTS=()
    for path in \
        "$WORK_DIR/exps/$EXP_S1" \
        "$WORK_DIR/exps/$EXP_S2" \
        "$WORK_DIR/eval_output/${EXP_S2}_musiccaps" \
        "$WORK_DIR/eval_output/metrics/${EXP_S2}_musiccaps" \
        "$LOG_DIR/${EXP_S1}.log" \
        "$LOG_DIR/${EXP_S2}.log" \
        "$LOG_DIR/${EXP_S2}_musiccaps_eval.log" \
        "$LOG_DIR/${EXP_PREFIX}_contract.json"; do
        if [ -f "$path" ]; then
            FRESH_CONFLICTS+=("$path")
        elif [ -d "$path" ] && [ -n "$(find "$path" -mindepth 1 -print -quit 2>/dev/null)" ]; then
            FRESH_CONFLICTS+=("$path")
        fi
    done
    if [ "${#FRESH_CONFLICTS[@]}" -gt 0 ]; then
        echo "[FAIL] fresh run refused: existing artifacts would contaminate the experiment" >&2
        printf '  %s\n' "${FRESH_CONFLICTS[@]}" >&2
        echo "Use a new EXP_PREFIX, or explicitly set EXPERIMENT_RUN_MODE=resume after verifying the checkpoints." >&2
        exit 2
    fi
fi

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"

echo "======================================================"
echo "  Phase 8 — controlled full S1+S2 retrain"
echo "  regime      : $EXPERIMENT_REGIME ($EXPERIMENT_RUN_MODE)"
echo "  S1 exp_id    : $EXP_S1"
echo "  S2 exp_id    : $EXP_S2"
echo "  NPZ dir      : $SINGLECAP_NPZ"
echo "  Train TSV    : $TRAIN_TSV"
echo "  multi_cap    : False"
echo "  S1 Q cond    : $S1_USE_Q_CONDITIONING"
echo "  S2 Q cond    : $S2_USE_Q_CONDITIONING"
echo "  Eval Q mode  : $RESOLVED_EVAL_Q_MODE"
echo "  text mask    : $USE_TEXT_ATTENTION_MASK"
echo "  Expected rows: $EXPECTED_ROWS"
echo "  Primary eval : $RUN_PRIMARY_EVAL ($EVAL_NUM_SAMPLES samples)"
echo "  Eval MusicCaps: $TSV_MUSICCAPS"
echo "  Eval Jamendo : skip=$([ $RUN_JAMENDO_EVAL = true ] && echo NO || echo YES)"
echo "  Estimated ETA: ~19-20 hr (S1 12.3 + S2 6.7 + eval)"
echo "======================================================"

# ============================================================
# Pre-flight: Bug fix 驗證
# ============================================================
echo "[Pre-flight] 驗證 bug fix 已套用..."

for required in "$TRAIN_TSV" "$GT_CACHE" "$SINGLECAP_NPZ" "$NPZ_MANIFEST" \
                "$TSV_MUSICCAPS" "$EVAL_SCRIPT"; do
    if [ ! -e "$required" ]; then
        echo "[FAIL] 缺少必要檔案：$required"
        exit 1
    fi
done

python - "$TRAIN_TSV" "$GT_CACHE" "$SINGLECAP_NPZ" "$NPZ_MANIFEST" "$EXPECTED_ROWS" <<'PYEOF'
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

tsv, cache, npz_dir, manifest_path = map(Path, sys.argv[1:5])
expected_rows = int(sys.argv[5])
with tsv.open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
with cache.open() as f:
    names = [line.strip() for line in f if line.strip()]
if manifest_path.suffix == ".json":
    provenance = json.loads(manifest_path.read_text())
    if provenance.get("status") != "passed" or provenance.get("completed_rows") != expected_rows:
        raise SystemExit("[FAIL] JSON NPZ provenance manifest is not passed/full")
    manifest = None
else:
    with manifest_path.open() as f:
        manifest = list(csv.DictReader(f, delimiter="\t"))
if not (len(rows) == len(names) == expected_rows and (manifest is None or len(manifest) == expected_rows)):
    raise SystemExit(
        f"[FAIL] row mismatch: tsv={len(rows)}, cache={len(names)}, "
        f"manifest={len(manifest) if manifest is not None else 'json-provenance'}"
    )
probe_indices = sorted(
    {i for i in [0, 1, 100, 1000, 10000, len(rows) - 1] if i < len(rows)}
)
for i in probe_indices:
    if manifest is not None:
        item = manifest[i]
        if (item["clip_id"], item["npz_fname"]) != (rows[i]["id"], names[i]):
            raise SystemExit(f"[FAIL] manifest mismatch at row {i}")
    path = npz_dir / names[i]
    if not path.exists():
        raise SystemExit(f"[FAIL] missing NPZ: {path}")
    with np.load(path) as data:
        if data["text_features"].shape != (77, 1024):
            raise SystemExit(f"[FAIL] bad text feature shape: {path}")
        if data["text_features_c"].shape != (512,):
            raise SystemExit(f"[FAIL] bad CLAP feature shape: {path}")
        if "clip_id" not in data.files:
            raise SystemExit(
                f"[FAIL] NPZ lacks embedded clip provenance: {path}. "
                "TSV/cache/manifest agreement alone does not prove audio-text alignment."
            )
        if str(data["clip_id"].item()) != rows[i]["id"]:
            raise SystemExit(f"[FAIL] embedded clip_id mismatch: {path}")
        if "caption_sha256" not in data.files:
            raise SystemExit(f"[FAIL] NPZ lacks caption provenance: {path}")
        expected_hash = hashlib.sha256(rows[i]["caption"].encode("utf-8")).hexdigest()
        if str(data["caption_sha256"].item()) != expected_hash:
            raise SystemExit(f"[FAIL] embedded caption hash mismatch: {path}")
        if manifest is not None and "q_level" in rows[i] and "historical_q_level" in item:
            if item["historical_q_level"] != rows[i]["q_level"]:
                raise SystemExit(f"[FAIL] historical q provenance mismatch: {path}")
print(f"[OK] provenance-backed NPZ alignment verified ({len(rows):,} rows)")
PYEOF

python - "$LOG_DIR/${EXP_PREFIX}_contract.json" "$EXPERIMENT_REGIME" \
    "$EXPERIMENT_RUN_MODE" "$EXP_PREFIX" "$EXP_S1" "$EXP_S2" \
    "$S1_USE_Q_CONDITIONING" "$S2_USE_Q_CONDITIONING" \
    "$USE_TEXT_ATTENTION_MASK" "$RESOLVED_EVAL_Q_MODE" \
    "$TRAIN_TSV" "$GT_CACHE" "$SINGLECAP_NPZ" "$TSV_MUSICCAPS" \
    "$EXPECTED_ROWS" "$S1_ITERATIONS" "$S2_ITERATIONS" <<'PYEOF'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out, regime, run_mode, prefix, exp_s1, exp_s2, s1_q, s2_q, text_mask,
    eval_q, train_tsv, gt_cache, npz_dir, eval_tsv, expected_rows,
    s1_iterations, s2_iterations,
) = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

critical = [
    root / "scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh",
    root / "meanaudio/runner_flowmatching.py",
    root / "meanaudio/runner_meanflow.py",
    root / "meanaudio/model/networks.py",
]
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "regime": regime,
    "run_mode_at_launch": run_mode,
    "prefix": prefix,
    "stage1_exp": exp_s1,
    "stage2_exp": exp_s2,
    "stage1_use_q_conditioning": s1_q == "true",
    "stage2_use_q_conditioning": s2_q == "true",
    "use_text_attention_mask": text_mask == "true",
    "eval_q_mode": eval_q,
    "multi_cap": False,
    "train_tsv": train_tsv,
    "gt_cache": gt_cache,
    "npz_dir": npz_dir,
    "eval_tsv": eval_tsv,
    "expected_rows": int(expected_rows),
    "stage1_iterations": int(s1_iterations),
    "stage2_additional_iterations": int(s2_iterations),
    "stage2_final_iteration": int(s1_iterations) + int(s2_iterations),
    "critical_file_sha256": {str(p.relative_to(root)): sha256(p) for p in critical},
}
try:
    payload["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
except Exception:
    payload["git_head"] = None

path = Path(out)
if path.exists():
    previous = json.loads(path.read_text())
    immutable = [
        "prefix", "stage1_exp", "stage2_exp",
        "stage1_use_q_conditioning", "stage2_use_q_conditioning",
        "use_text_attention_mask", "eval_q_mode", "multi_cap", "train_tsv",
        "gt_cache", "npz_dir", "eval_tsv", "expected_rows",
        "stage1_iterations", "stage2_additional_iterations",
        "stage2_final_iteration",
    ]
    changed = [key for key in immutable if previous.get(key) != payload.get(key)]
    if previous.get("regime") not in (regime, "custom"):
        changed.append("regime")
    previous_hashes = previous.get("critical_file_sha256") or {}
    changed_hashes = [
        rel for rel, current_hash in payload["critical_file_sha256"].items()
        if previous_hashes.get(rel) != current_hash
        and not (
            previous.get("regime") == "custom"
            and rel == "scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
        )
    ]
    if changed_hashes:
        changed.append(f"critical_file_sha256:{changed_hashes}")
    if changed:
        raise SystemExit(f"[FAIL] resume contract drift in {changed}: {path}")
    print(f"[OK] immutable run contract unchanged: {path}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(f"[OK] wrote immutable run contract: {path}")
PYEOF

python -c "
import re
with open('meanaudio/model/networks.py') as f:
    code = f.read()
mf_start = code.find('class MeanAudio')
mf_code = code[mf_start:]
if re.search(r'q = torch\.full\(\([^,]+,\), 9,', mf_code):
    raise SystemExit('[FAIL] Bug #1 未修')
print('[OK] Bug #1 fix 已套用')
" || exit 1

if [ "$USE_TEXT_ATTENTION_MASK" = "false" ]; then
    python - "$TRAIN_TSV" "$GT_CACHE" "$SINGLECAP_NPZ" <<'PYEOF'
import sys

from meanaudio.data.extracted_audio import ExtractedAudio

tsv, cache, npz_dir = sys.argv[1:]
dataset = ExtractedAudio(
    tsv_path=tsv,
    concat_text_fc=False,
    npz_dir=npz_dir,
    data_dim={"latent_seq_len": 312, "text_seq_len": 77, "text_dim": 1024, "text_c_dim": 512},
    repa_npz_dir=None,
    exclude_cls=False,
    repa_version=1,
    gt_cache=cache,
    multi_cap=False,
    use_text_attention_mask=False,
)
sample = dataset[0]
if "text_attention_mask" in sample:
    raise SystemExit("[FAIL] legacy NoMask preflight still returned text_attention_mask")
print("[OK] legacy NoMask path omits text_attention_mask from training batches")
PYEOF
fi

python -c "
with open('meanaudio/runner_meanflow.py') as f:
    code = f.read()
if 'text_f_undrop = text_f.clone()' not in code or 'text_f_c_undrop = text_f_c.clone()' not in code:
    raise SystemExit('[FAIL] Bug #2 未修')
print('[OK] Bug #2 fix 已套用')
" || exit 1

# ============================================================
# Stage 1
# ============================================================
S1_CKPT_COMPLETE=false
if [ -f "$S1_CKPT" ]; then
    CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))" 2>/dev/null)
    if [ -z "$CKPT_IT" ]; then
        mv "$S1_CKPT" "${S1_CKPT}.corrupted_$(date +%Y%m%d_%H%M%S)"
    elif [ "$CKPT_IT" -ge "$S1_ITERATIONS" ]; then
        S1_CKPT_COMPLETE=true
        echo "[Stage 1] 已完成 (iter $CKPT_IT)"
    fi
fi

if [ "$S1_CKPT_COMPLETE" = "true" ]; then
    echo "[Stage 1] 跳過訓練"
else
    echo "[Stage 1] 開始 / 繼續訓練：$EXP_S1"
    python "$STAGE_SCRIPT" --stage 1

    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio \
        model=fluxaudio_s \
        exp_id="$EXP_S1" \
        num_iterations=$S1_ITERATIONS \
        "lr_schedule_steps=[320000,360000]" \
        "+use_q_conditioning=$S1_USE_Q_CONDITIONING" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${EXP_S1}.log"

    echo "[Stage 1] 訓練完成"
fi

# ============================================================
# Migrate
# ============================================================
if [ -f "$S2_CKPT" ]; then
    echo "[Migrate] S2 checkpoint 已存在，跳過以保留 resume state：$S2_CKPT"
else
    echo "[Migrate] $S1_CKPT → $S2_CKPT"
    python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"
fi

# ============================================================
# Stage 2
# ============================================================
if [ -f "$S2_EMA" ]; then
    echo "[Stage 2] EMA final 已存在，跳過訓練：$S2_EMA"
else
    echo "[Stage 2] 開始 / 繼續訓練：$EXP_S2"
    python "$STAGE_SCRIPT" --stage 2

    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio \
        model=meanaudio_s \
        exp_id="$EXP_S2" \
        num_iterations=$(( S1_ITERATIONS + S2_ITERATIONS )) \
        "lr_schedule_steps=[999999,999999]" \
        "+use_q_conditioning=$S2_USE_Q_CONDITIONING" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

    echo "[Stage 2] 訓練完成"
fi

TEXT_MASK_EVAL_FLAG=()
if [ "$USE_TEXT_ATTENTION_MASK" = "false" ]; then
    TEXT_MASK_EVAL_FLAG+=(--no_text_attention_mask)
fi

# Eval q flag must match how the model was trained (reference_eval_q_flag_rule):
# Q-trained models are evaluated in-support (--quality_level 9); under the fixed
# runners q=10 is only ever the CFG-unconditional marker, so --no_q on a Q-trained
# model generates unconditionally (phase8_legacy_repro 2026-07-18: CLAP 0.0134
# with --no_q vs 0.1684 with --quality_level 9, same checkpoint).
Q_EVAL_FLAG=(--no_q)
if [ "$RESOLVED_EVAL_Q_MODE" = "q9" ]; then
    Q_EVAL_FLAG=(--quality_level 9)
fi

EVAL_METRIC_EXTRA_ARGS=()
if [ "$EVAL_SKIP_AES" = "true" ]; then
    EVAL_METRIC_EXTRA_ARGS+=(--skip_aes)
fi

# ============================================================
# Eval #1: MusicCaps (primary)
# ============================================================
if [ "$RUN_PRIMARY_EVAL" = "true" ]; then
    EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_musiccaps"
    echo "[Eval MusicCaps] → $EVAL_OUT_MC"

    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 "${Q_EVAL_FLAG[@]}" \
        "${TEXT_MASK_EVAL_FLAG[@]}" \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --exp_name "${EXP_S2}_musiccaps" \
        --num_samples "$EVAL_NUM_SAMPLES" \
        "${EVAL_METRIC_EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"
fi

# ============================================================
# Eval #2: Jamendo (optional)
# ============================================================
if [ "$RUN_JAMENDO_EVAL" = "true" ]; then
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo"
    echo "[Eval Jamendo] → $EVAL_OUT_JM"

    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 "${Q_EVAL_FLAG[@]}" \
        "${TEXT_MASK_EVAL_FLAG[@]}" \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO" \
        --exp_name "${EXP_S2}_jamendo" \
        --num_samples "$EVAL_NUM_SAMPLES" \
        "${EVAL_METRIC_EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_jamendo_eval.log"
fi

echo "======================================================"
echo "  Phase 8 controlled retrain complete"
echo "  S1 EMA   : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA   : $S2_EMA"
echo "  MusicCaps: eval_output/metrics/${EXP_S2}_musiccaps/metrics.txt"
echo ""
echo "  基線（修前）:"
echo "    MusicCaps: CLAP 0.1851, CE 5.91, CU 6.75, PC 4.98, PQ 6.54"
echo "    Jamendo:   CLAP 0.1851"
echo "======================================================"
