#!/bin/bash
# 084 NegMF arm, one training seed (prereg docs/experiments/negprompt_distill_meanflow_084_20260927.md).
#
# The guidance branch of the MeanFlow CFG training target (mean_flow.py loss(): the u_t term,
# normally the fixed null features) reads the fidelity8 negative prompt instead:
#   n100  every sample             (++mf_guide_t_min=0.0)
#   nhi   only samples with t > 2/3 (++mf_guide_t_min=0.6667; t=1 is noise)
# S2-only branch: migrate the nmv2pair slot0clean control's own S1 ckpt_last and train 50k S2
# with the control's exact recipe and seed (caption2p0_nmv2pair_action.sh slot0clean), so the
# data order is shared and the target is the only difference.
#
# Steps
#   0  preflight: guide features match their manifest, control inputs / S1 / stock reports exist
#   1  migrate control S1 ckpt_last -> arm S2 ckpt (skipped when it exists)
#   2  Stage 2 with the guide keys; the guide log line must be present; NaN gate;
#      drop S2 shadows, thin S2 EMA snapshots
#   3  eval, arm:     stock cfg0 + cfg3_neg (mc_mf25_eval.sh), 1-NFE cfg0 (mc_nfe1_cfg0_eval.sh)
#            control: 1-NFE cfg0 (stock cells already exist)
#      every cell: FAD (mf25 cells; before its audio is deleted), -30 LUFS rescore, then audio deleted.
#      Control stock cells: FAD only if missing, then their audio is deleted (lvl30 already exists).
#
# Usage: NEGMF_ARM=n100|nhi NEGMF_SEED=<seed> negmf_084_action.sh
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
MC_TSV="$DATA/musiccaps_test.tsv"
FAD_REF="/mnt/HDD/kojiek/musiccaps_reference"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARM="${NEGMF_ARM:?NEGMF_ARM}"
SEED="${NEGMF_SEED:?NEGMF_SEED}"
case "$ARM" in
  n100) T_MIN=0.0 ;;
  nhi)  T_MIN=0.6667 ;;
  *) echo "[FAIL] NEGMF_ARM must be n100 or nhi" >&2; exit 2 ;;
esac
S1_UPDATES=100000; S2_ADD=50000; FINAL_IT=$((S1_UPDATES + S2_ADD)); LR=1e-4; BATCH=8
GUIDE_DIR="$WORK_DIR/weights/negmf_084"
GUIDE_T5="$GUIDE_DIR/fidelity8_t5.pth"; GUIDE_CLAP="$GUIDE_DIR/fidelity8_clap_c.pth"
GUIDE_MANIFEST="$GUIDE_DIR/fidelity8_manifest.json"
INPUTS="$HOME/exps_nvme/slot0clean_nmv2matched/arm_inputs"
TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
CACHE_LIST="$INPUTS/cache_train.txt"; MANIFEST="$INPUTS/manifest.json"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"; OVERLAY="$HOME/text_overlays/slot0clean"
EVAL_ROOT="$HOME/eval_output_nvme"

EXP_PREFIX="phase8_qwen_caption2p0_slot0clean_negmf${ARM}_noq_quarter_s${SEED}"
CTRL_PREFIX="phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s${SEED}"
CTRL_S1_CKPT="$WORK_DIR/exps/${CTRL_PREFIX}_stage1_${S1_UPDATES}/${CTRL_PREFIX}_stage1_${S1_UPDATES}_ckpt_last.pth"
CTRL_EMA="$WORK_DIR/exps/${CTRL_PREFIX}_stage2_${S2_ADD}/${CTRL_PREFIX}_stage2_${S2_ADD}_ema_final.pth"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
free_b(){ df -B1 --output=avail "$1" | tail -1; }
thin_ema(){
  local d="$1/ema_ckpts" keep="$2" p
  [ -d "$d" ] || return 0
  for p in "$d"/*.pt; do
    [ -e "$p" ] || continue
    case " $keep " in *" $(basename "$p") "*) ;; *) rm -f -- "$p" ;; esac
  done
}
drop_shadows(){ rm -f -- "$1"/*_ckpt_shadow.pth "$1"/*_shadow.pth; }

log "[Step 0] 084 NegMF arm=$ARM (t > $T_MIN) seed=$SEED ($EXP_PREFIX)"
if [ ! -f "$S2_EMA" ]; then NEED=13000000000; else NEED=5000000000; fi
if [ "$(free_b "$HOME")" -lt "$NEED" ]; then
  log "[FAIL] NVMe free $(( $(free_b "$HOME") / 1000000000 ))G < $((NEED / 1000000000))G"; exit 3
fi
[ -f "$CTRL_EMA" ] || { log "[FAIL] control checkpoint missing: $CTRL_EMA"; exit 2; }
for C in cfg0 cfg3_neg; do
  d="$EVAL_ROOT/${CTRL_PREFIX}_mc_mf25_${C}"
  [ -f "$d/${CTRL_PREFIX}_mc_mf25_${C}_REPORT.json" ] || { log "[FAIL] control stock report missing: $d"; exit 2; }
  ls "$d"_lvl30/*/per_clip.tsv >/dev/null 2>&1 || ls "$EVAL_ROOT/d2_075_lvl30/${CTRL_PREFIX}_mc_mf25_${C}_lvl30"/*/per_clip.tsv >/dev/null 2>&1 \
    || { log "[FAIL] control lvl30 metrics missing for $C"; exit 2; }
done
GUIDE_T5="$GUIDE_T5" GUIDE_CLAP="$GUIDE_CLAP" GUIDE_MANIFEST="$GUIDE_MANIFEST" TRAIN_TSV="$TRAIN_TSV" \
CACHE_LIST="$CACHE_LIST" MANIFEST="$MANIFEST" "$PY" - <<'PYEOF'
import hashlib, json, os
E = os.environ
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
g = json.load(open(E["GUIDE_MANIFEST"]))
assert g["G0b_training_vs_inference_path"]["pass"], "[FAIL] guide G0b did not pass"
assert g["text"] == "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi", "[FAIL] guide text"
assert sha(E["GUIDE_T5"]) == g["t5_sha256"], "[FAIL] guide t5 drift"
assert sha(E["GUIDE_CLAP"]) == g["clap_sha256"], "[FAIL] guide clap drift"
m = json.load(open(E["MANIFEST"]))
assert sha(E["TRAIN_TSV"]) == m["train_tsv_sha256"], "[FAIL] control train tsv drift"
assert sha(E["CACHE_LIST"]) == m["cache_list_sha256"], "[FAIL] control cache list drift"
print(f"  guide + control inputs verified (rows={m['rows']})")
PYEOF

COMMON=(
  data=meanaudio "lr_schedule_steps=[999999,999999]"
  "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1
  learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4
  save_weights_interval=10000 save_checkpoint_interval=10000
  ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False
  +use_text_attention_mask=false val_interval=999999 eval_interval=999999
  save_eval_interval=999999
  "++multi_cap=false" "++cap_index_fixed=0"
  "data.AudioCaps_npz.tsv=$TRAIN_TSV"
  "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
  "++data.AudioCaps_npz.gt_cache=$CACHE_LIST"
  "++data.AudioCaps_npz.text_npz_dir=$OVERLAY"
  "++data.AudioCaps_npz.require_text_overlay=true"
  "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv"
  "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4"
  "++data.AudioCaps_val_npz.gt_cache=null"
)
GUIDE=( "++mf_guide_t5=$GUIDE_T5" "++mf_guide_clap_c=$GUIDE_CLAP" "++mf_guide_t_min=$T_MIN" )

if [ ! -f "$S2_EMA" ]; then
  if [ ! -f "$S2_CKPT" ]; then
    log "[Step 1] migrate control S1 -> $S2_EXP"
    [ -f "$CTRL_S1_CKPT" ] || { log "[FAIL] control S1 ckpt_last missing: $CTRL_S1_CKPT"; exit 2; }
    mkdir -p "$S2_DIR"
    rm -f -- "$S2_CKPT.tmp"   # a stale tmp would make migrate write a 2.3 GB backup copy
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$CTRL_S1_CKPT" --s2_out "$S2_CKPT.tmp" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
    mv -f -- "$S2_CKPT.tmp" "$S2_CKPT"
  fi
  log "[Step 2] Stage 2 $S2_EXP (guide t > $T_MIN)"
  "$PY" set_training_stage.py --stage 2
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=meanaudio_s exp_id="$S2_EXP" num_iterations="$FINAL_IT" \
    "${COMMON[@]}" "${GUIDE[@]}" "checkpoint=$S2_CKPT" 2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 2] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }
restore_stage_1
# the run must have used the guide branch (otherwise it is a stock rerun and the numbers mean nothing)
grep -aq "MeanFlow CFG target guidance branch uses $GUIDE_T5 for t > $T_MIN" "$S2_DIR"/train-*-rank0.log \
  || { log "[FAIL] guide log line missing in $S2_DIR"; exit 5; }
"$PY" - "$S2_DIR" <<'PYEOF'
import glob, re, sys
txt = "".join(open(p, errors="replace").read() for p in glob.glob(f"{sys.argv[1]}/train-*-rank0.log"))
n_loss_nan = len(re.findall(r"loss:[ ]*nan", txt))
n_grad = len(re.findall(r"grad_norm:", txt)); n_grad_nan = len(re.findall(r"grad_norm:[ ]*nan", txt))
print(f"  NaN gate: loss nan {n_loss_nan}, grad_norm nan {n_grad_nan}/{n_grad}")
if n_loss_nan or (n_grad and n_grad_nan / n_grad > 0.05):
    sys.exit("[FAIL] NaN gate")
PYEOF
drop_shadows "$S2_DIR"
thin_ema "$S2_DIR" "0.110000.pt 0.130000.pt 1.110000.pt 1.130000.pt"

# ---- eval -------------------------------------------------------------------------------
fad_cell(){      # FAD on a cell's audio, before it is deleted
  local label="$1" d="$EVAL_ROOT/$1"
  ls "${d}_fad"/*/metrics.json >/dev/null 2>&1 && return 0
  [ -d "$d/audio" ] || { log "[WARN] $label: no audio left for FAD"; return 0; }
  "$PY" scripts/eval/eval_metrics.py --gen_dir "$d/audio" --tsv "$MC_TSV" --exp_name "${label}_fad" \
    --out_dir "${d}_fad" --skip_clap --skip_aes --skip_level --fad --ref_dir "$FAD_REF" \
    --fad_num_samples 2048 2>&1 | tee -a "$STATE/eval.log"
  ls "${d}_fad"/*/metrics.json >/dev/null 2>&1 || { log "[FAIL] FAD missing for $label"; exit 4; }
}
finish_cell(){   # -30 LUFS rescore (unless already there), then drop audio
  local label="$1" d="$EVAL_ROOT/$1"
  [ -f "$d/${label}_REPORT.json" ] || { log "[FAIL] missing report $d/${label}_REPORT.json"; exit 4; }
  if ! ls "$d"_lvl30/*/per_clip.tsv >/dev/null 2>&1 && \
     ! ls "$EVAL_ROOT/d2_075_lvl30/${label}_lvl30"/*/per_clip.tsv >/dev/null 2>&1; then
    [ -d "$d/audio" ] || { log "[FAIL] $label: no audio to rescore and no lvl30 metrics"; exit 4; }
    "$PY" scripts/eval/level_match_rescore.py --cell_dir "$d" --tsv "$MC_TSV" 2>&1 | tee -a "$STATE/eval.log"
    ls "$d"_lvl30/*/per_clip.tsv >/dev/null 2>&1 || { log "[FAIL] $label lvl30 rescore missing"; exit 4; }
  fi
  rm -rf -- "$d/audio" "${d}_lvl30/audio"
}

log "[Step 3] eval arm $EXP_PREFIX"
bash scripts/eval/mc_mf25_eval.sh "$EXP_PREFIX" "$S2_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
for C in cfg0 cfg3_neg; do
  fad_cell "${EXP_PREFIX}_mc_mf25_${C}"; finish_cell "${EXP_PREFIX}_mc_mf25_${C}"
done
bash scripts/eval/mc_nfe1_cfg0_eval.sh "$EXP_PREFIX" "$S2_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
finish_cell "${EXP_PREFIX}_mc_nfe1_cfg0"

log "[Step 3] eval control $CTRL_PREFIX (new cells only)"
for C in cfg0 cfg3_neg; do
  fad_cell "${CTRL_PREFIX}_mc_mf25_${C}"; finish_cell "${CTRL_PREFIX}_mc_mf25_${C}"
done
bash scripts/eval/mc_nfe1_cfg0_eval.sh "$CTRL_PREFIX" "$CTRL_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
finish_cell "${CTRL_PREFIX}_mc_nfe1_cfg0"

"$PY" scripts/analysis/negmf_084_analysis.py 2>&1 | tee -a "$STATE/eval.log" || log "[WARN] analysis failed (cells are complete)"
log "[DONE] $EXP_PREFIX"
