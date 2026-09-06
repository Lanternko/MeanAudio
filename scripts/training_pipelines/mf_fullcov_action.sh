#!/bin/bash
# Music Flamingo at c2p0 scale: 251,599 clips, short_direct_v2 captions.
#
# What this settles that nothing before it could. 036/037 trained MF on a
# 100k slice whose captions failed the corpus audit (73.17% unique, 79.05%
# truncated at the T5 window), so its deficit was unattributable. paired59k
# then held audio, rows, order, recipe and budget fixed and moved only the
# caption text -- MF lost to Qwen by 0.0073 CLAP with all four AES metrics
# inside the seed floor -- but only over the 59,614-clip intersection, i.e.
# 23.7% of the corpus. Coverage was the one asymmetry that control could not
# absorb. The full-coverage recaption removes it: same audio latents, same
# 251,599 rows in the same order, same recipe, same budget as the c2p0 arms,
# and the only remaining difference is who wrote the captions.
#
# Nothing is re-extracted. The arms read the c2p0 audio NPZ cache through its
# own cache list and take caption text from a text_npz_dir overlay, so the
# audio side is byte-identical to every c2p0 arm.
#
# require_text_overlay is ON here, unlike paired59k. The c2p0 audio NPZs carry
# clip_id, and build_mf_fullcov_arm_inputs.py emits the matching slot-suffixed
# id, so the loader can verify the audio/text/TSV binding on every single row
# instead of trusting an offline sample.
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# $1 = quarter | full. Both share one TSV and one text overlay, so the full arm
# reuses whatever the quarter arm built.
SCALE="${1:-full}"
case "$SCALE" in
  quarter) S1_UPDATES=100000; S2_ADD=50000  ;;
  full)    S1_UPDATES=400000; S2_ADD=200000 ;;
  *) echo "[FAIL] unknown scale: $SCALE"; exit 2 ;;
esac
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="mf_fullcov_noq_${SCALE}"
EXPECTED_N=251599
LR=1e-4
BATCH=8
SEED=14159265

INPUTS="$HOME/exps_nvme/mf_full_coverage/arm_inputs"
TRAIN_TSV="$INPUTS/mf_fullcov_train.tsv"
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"
OVERLAY="$HOME/text_overlays/mf_fullcov"
JSONL="$HOME/eval_output/mf_recaption_full_coverage/caption.jsonl"

S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# ---- Step 0: disk guard -----------------------------------------------------
# The overlay is ~76G (302 KB/row x 251,599, measured on paired59k) and the two
# stages hold ~28G of checkpoints between them.
FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
NEED=110000000000
if [ ! -f "$OVERLAY/DONE.json" ] && [ "$FREE_NVME" -lt "$NEED" ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < $((NEED/1000000000))G needed for overlay + checkpoints"
  exit 3
fi

# ---- Step 0b: early-kill gate on the quarter arm (full only) ---------------
# 037 was seated automatically after 036 returned a number that its own contract
# said should cancel it, and had to be killed by hand 27k iterations in. The
# queue has no dependency mechanism -- ordering is purely lexicographic -- so the
# rule enforces itself here instead of relying on an operator being awake.
#
# The hypothesis this arm tests is that full coverage plus the enforced
# short_direct_v2 captions fix what the 100k slice could not. The quarter arm is
# budget-matched to the 100k quarter that scored 0.1774 CFG0, so if the quarter
# does not clear 0.1900 the coverage story has already failed and the ~19h full
# run is not justified. Override deliberately by touching the file named below.
GATE_MIN=0.1900
QUARTER_REPORT="$HOME/cfg0_eval_runtime/reports/mf_fullcov_noq_quarter_musiccaps_mf25_cfg0_noq_REPORT.json"
OVERRIDE="$HOME/exps_nvme/mf_full_coverage/PROCEED_TO_FULL_ANYWAY"
if [ "$SCALE" = "full" ] && [ ! -f "$OVERRIDE" ]; then
  if [ ! -f "$QUARTER_REPORT" ]; then
    log "[FAIL] quarter arm has no CFG0 report yet; nothing to gate on"; exit 5
  fi
  "$PY" - <<GATEPY
import json
d = json.load(open("$QUARTER_REPORT"))
clap = d["metrics"]["clap_score"]
print(f"  quarter CFG0 clap={clap:.4f} gate=$GATE_MIN")
if clap < $GATE_MIN:
    raise SystemExit(
        f"[FAIL] quarter CFG0 {clap:.4f} < $GATE_MIN; the coverage hypothesis "
        f"failed at quarter budget (the 100k-slice quarter was 0.1774). Touch "
        f"$OVERRIDE to run anyway.")
GATEPY
fi

# ---- Step 1: training TSV bound to the c2p0 row order -----------------------
log "[Step 1] arm inputs"
if [ ! -f "$INPUTS/bindings.json" ] || [ ! -f "$TRAIN_TSV" ]; then
  "$PY" scripts/preprocess/build_mf_fullcov_arm_inputs.py \
    --jsonl "$JSONL" --out-dir "$INPUTS" 2>&1 | tee "$STATE/build_inputs.log"
else
  log "[Step 1] arm inputs already built"
fi
[ -f "$TRAIN_TSV" ] || { log "[FAIL] no train tsv"; exit 2; }

# ---- Step 2: corpus audit gate ---------------------------------------------
# 037 trained for 27k iterations on a corpus that was only 73.17% unique before
# anyone measured it. The gate is CLAUDE.md's pre-experiment checklist item 2,
# enforced here so it cannot be skipped.
log "[Step 2] corpus audit gate"
"$PY" - <<PYEOF
import csv, json
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open("$TRAIN_TSV", newline=""), delimiter="\t"))
assert len(rows) == $EXPECTED_N, f"[FAIL] tsv rows {len(rows)} != $EXPECTED_N"
caps = [r["caption"] for r in rows]
uniq = len(set(caps)) / len(caps)
empty = sum(1 for c in caps if not c.strip())
print(f"  rows={len(rows)} unique_rate={uniq:.4f} empty={empty}")
if uniq < 0.90:
    raise SystemExit(f"[FAIL] unique caption rate {uniq:.4f} < 0.90 (CLAUDE.md checklist item 2)")
if empty:
    raise SystemExit(f"[FAIL] {empty} empty captions")
PYEOF

# ---- Step 3: text overlay (encode once, both scales share it) ---------------
log "[Step 3] text overlay"
if [ ! -f "$OVERLAY/DONE.json" ]; then
  "$PY" scripts/preprocess/build_single_cap_text_overlay.py \
    --train-tsv "$TRAIN_TSV" --cache-list "$CACHE_LIST" \
    --output-dir "$OVERLAY" --batch-size 48 2>&1 | tee -a "$STATE/overlay.log"
else
  log "[Step 3] overlay already complete"
fi
[ -f "$OVERLAY/DONE.json" ] || { log "[FAIL] overlay not finished"; exit 2; }

# ---- Step 4: Stage 1 --------------------------------------------------------
# Arguments are the c2p0 launcher's verbatim, so recipe is not a free variable
# in the comparison; the two additions are text_npz_dir and
# require_text_overlay.
COMMON=(
  data=meanaudio "lr_schedule_steps=[999999,999999]"
  "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1
  learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4
  save_weights_interval=10000 save_checkpoint_interval=10000
  ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False
  +use_text_attention_mask=false val_interval=999999 eval_interval=999999
  save_eval_interval=999999
  "data.AudioCaps_npz.tsv=$TRAIN_TSV"
  "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
  "++data.AudioCaps_npz.gt_cache=$CACHE_LIST"
  "++data.AudioCaps_npz.text_npz_dir=$OVERLAY"
  "++data.AudioCaps_npz.require_text_overlay=true"
  "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv"
  "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4"
  "++data.AudioCaps_val_npz.gt_cache=null"
)

if [ ! -f "$S1_EMA" ]; then
  log "[Step 4] Stage 1 $S1_EXP"
  "$PY" set_training_stage.py --stage 1
  S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=fluxaudio_s exp_id="$S1_EXP" num_iterations="$S1_UPDATES" \
    "${COMMON[@]}" "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
else
  log "[Step 4] S1 already complete"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1"; exit 2; }

# ---- Step 5: migrate + Stage 2 ---------------------------------------------
if [ ! -f "$S2_EMA" ]; then
  log "[Step 5] Stage 2 $S2_EXP"
  "$PY" set_training_stage.py --stage 2
  if [ ! -f "$S2_CKPT" ]; then
    SRC="$S1_CKPT"; [ -f "$SRC" ] || SRC="$S1_EMA"
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=meanaudio_s exp_id="$S2_EXP" num_iterations="$FINAL_IT" \
    "${COMMON[@]}" 2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 5] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }

# ---- Step 6: arm-comparison eval (CFG 3.0 + fidelity negative) -------------
NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
MC_TSV="$DATA/musiccaps_test.tsv"
OUT="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_cfg3_neg"
mkdir -p "$OUT/audio"
HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE" -lt 5400 ]; then
  log "[Step 6] generating (have $HAVE)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$OUT/audio" --tsv "$MC_TSV" --use_meanflow \
    --num_steps 25 --cfg_strength 3.0 --negative_prompt "$NEG" \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_gen.log"
fi
GOT=$(find "$OUT/audio" -name '*.flac' | wc -l)
log "generated $GOT / 5521"
[ "$GOT" -ge 5400 ] || { log "[FAIL] only $GOT clips"; exit 4; }

"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$OUT/audio" --tsv "$MC_TSV" \
  --exp_name "${EXP_PREFIX}_mc_mf25_cfg3_neg" \
  --out_dir "$OUT" 2>&1 | tee "$STATE/eval_metrics.log"

# ---- Step 7: canonical CFG 0 eval + harness completion evidence ------------
# harn_guest.py only records "completed" when completion_evidence.cfg0_report
# validates; without it a finished run is filed as held (memory
# project_cfg0_harness_unsatisfiable_gates_2026_08_29.md). The number is worth
# having anyway -- it puts the arm on the canonical table beside c2p0 slot0.
LABEL="${EXP_PREFIX}_musiccaps_mf25_cfg0_noq"
C0_OUT="$HOME/cfg0_eval_runtime/output/$LABEL"
C0_METRICS="$HOME/cfg0_eval_runtime/metrics/$LABEL"
C0_REPORT="$HOME/cfg0_eval_runtime/reports/${LABEL}_REPORT.json"
mkdir -p "$C0_OUT/audio" "$C0_METRICS" "$(dirname "$C0_REPORT")"
HAVE0=$(find "$C0_OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE0" -lt 5400 ]; then
  log "[Step 7] CFG0 generation (have $HAVE0)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$C0_OUT/audio" --tsv "$MC_TSV" --use_meanflow \
    --num_steps 25 --cfg_strength 0 \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_cfg0_gen.log"
fi
GOT0=$(find "$C0_OUT/audio" -name '*.flac' | wc -l)
[ "$GOT0" -ge 5400 ] || { log "[FAIL] cfg0 only $GOT0 clips"; exit 4; }
"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$C0_OUT/audio" --tsv "$MC_TSV" --exp_name "$LABEL" \
  --out_dir "$HOME/cfg0_eval_runtime/metrics" 2>&1 | tee "$STATE/eval_cfg0_metrics.log"

"$PY" - <<PYEOF
import hashlib, json, datetime, soundfile as sf
from pathlib import Path
m = {}
for line in Path("$C0_METRICS/metrics.txt").read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        try: m[k.strip()] = float(v.strip())
        except ValueError: pass
need = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
missing = need - set(m)
if missing:
    raise SystemExit(f"[FAIL] cfg0 metrics missing {missing}")
files = sorted(Path("$C0_OUT/audio").glob("*.flac"))
info = sf.info(str(files[0]))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
json.dump({
    "status": "passed",
    "label": "$LABEL",
    "conditioning": "no_q",
    "cfg_strength": 0,
    "num_steps": 25,
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 0; NoMask; seed 42; full precision",
    "checkpoint": "$S2_EMA",
    "checkpoint_sha256": sha("$S2_EMA"),
    "metrics": {k: m[k] for k in sorted(need)},
    "metrics_path": "$C0_METRICS/metrics.txt",
    "metrics_sha256": sha("$C0_METRICS/metrics.txt"),
    "audio_validation": {"rows": len(files), "unique_ids": len({f.stem for f in files}),
                         "sample_rate": info.samplerate, "channels": info.channels},
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, open("$C0_REPORT", "w"), indent=1, sort_keys=True)
print("wrote $C0_REPORT")
PYEOF

log "[DONE] $EXP_PREFIX"
