#!/bin/bash
# MF short_direct 100k, retrained under the c2p0 recipe.
#
# Why this exists: A3 (2026-05-31) and the c2p0 arms were NOT recipe-matched.
# A3 inherited lr_schedule=step, so runner_flowmatching.py:214 assigned
# [0.8N, 0.9N] with gamma 0.1 and its LR fell to 1e-6 for the last 20% of each
# stage, while every c2p0 arm held 1e-4 throughout. A3 also ran half the
# iteration budget (S1 200k / S2 100k vs 400k / 200k) and passed no gt_cache.
# This script copies the c2p0 launcher's arguments verbatim so the only
# remaining differences are the caption corpus and its 100k row count.
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# $1 = quarter | full. Both share one NPZ cache, so the full arm reuses
# whatever the quarter arm extracted.
SCALE="${1:-full}"
case "$SCALE" in
  quarter) S1_UPDATES=100000; S2_ADD=50000  ;;
  full)    S1_UPDATES=400000; S2_ADD=200000 ;;
  *) echo "[FAIL] unknown scale: $SCALE"; exit 2 ;;
esac
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="mfshort100k_direct_noq_c2p0recipe_${SCALE}"
SHARED=mfshort100k_direct_noq_c2p0recipe
EXPECTED_N=100000
LR=1e-4
BATCH=8
SEED=14159265

TRAIN_TSV="$DATA/music_flamingo_slice10_100k_short_direct_train.tsv"
CLIPS_TSV="$DATA/music_flamingo_slice10_100k_short_direct_clips.tsv"
WAV_DIR="$DATA/wav_audio"
NPZ_DIR="$HOME/exps_nvme/${SHARED}_npz"
NPZ_TSV="$HOME/exps_nvme/${SHARED}_npz.tsv"
LATENT_DIR="/mnt/HDD/kojiek/${SHARED}_latents_tmp"
CACHE_LIST="$HOME/exps_nvme/${SHARED}_npz_cache_train.txt"

S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# ---- Step 0: disk guard -----------------------------------------------------
FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ "$FREE_NVME" -lt 120000000000 ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < 120G needed for 100k NPZ"; exit 3
fi

# ---- Step 1: row counts -----------------------------------------------------
log "[Step 1] TSV row counts"
"$PY" - <<PYEOF
import csv
for label, path, n in [("train", "$TRAIN_TSV", $EXPECTED_N),
                       ("clips", "$CLIPS_TSV", $EXPECTED_N)]:
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    print(f"  {label}: {len(rows)}")
    if len(rows) != n:
        raise SystemExit(f"[FAIL] {label} expected {n}, got {len(rows)}")
PYEOF

# ---- Step 2: NPZ re-extraction (the May cache was deleted in cleanup) --------
NPZ_COUNT=$(find "$NPZ_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$NPZ_COUNT" -ne "$EXPECTED_N" ] || [ ! -f "$NPZ_TSV" ]; then
  log "[Step 2] extracting NPZ (have $NPZ_COUNT / $EXPECTED_N)"
  rm -rf "$NPZ_DIR" "$LATENT_DIR" "$NPZ_TSV"
  "$TORCHRUN" --standalone --nproc_per_node=1 training/extract_audio_latents.py \
    --data_dir "$WAV_DIR" --captions_tsv "$TRAIN_TSV" --clips_tsv "$CLIPS_TSV" \
    --latent_dir "$LATENT_DIR" --output_dir "$NPZ_DIR" \
    --batch_size "$BATCH" --num_workers 4 --text_encoder t5_clap \
    2>&1 | tee "$STATE/extract_npz.log"
else
  log "[Step 2] NPZ cache present: $NPZ_COUNT"
fi

# ---- Step 3: explicit cache list + real pairing audit -----------------------
# A3 passed no gt_cache, so extracted_audio.py:49 fell back to sequential
# i.npz. That is self-consistent here (training reads the extraction TSV), but
# CLAUDE.md forbids relying on it unaudited, so bind the list explicitly and
# verify pairing by re-encoding sampled captions.
log "[Step 3] cache list + pairing audit"
"$PY" - <<PYEOF
import csv, random, numpy as np, torch
from pathlib import Path
rows = list(csv.DictReader(open("$NPZ_TSV"), delimiter="\t"))
assert len(rows) == $EXPECTED_N, f"npz tsv rows {len(rows)}"
Path("$CACHE_LIST").write_text("".join(f"{i}.npz\n" for i in range(len(rows))))
d = np.load(Path("$NPZ_DIR") / "0.npz")
assert d["mean"].shape == (312, 20), d["mean"].shape
assert d["text_features"].shape == (77, 1024), d["text_features"].shape
assert d["text_features_c"].shape == (512,), d["text_features_c"].shape
assert "text_attention_mask" in d.files, "text_attention_mask missing"

from meanaudio.model.utils.features_utils import FeaturesUtils
fu = FeaturesUtils(enable_conditions=True, encoder_name="t5_clap").eval().cuda()
idx = random.Random(0).sample(range(len(rows)), 20)
bad = 0
for i in idx:
    cap = rows[i]["caption"]
    _, fc = fu.encode_text([cap])
    stored = torch.from_numpy(np.load(Path("$NPZ_DIR") / f"{i}.npz")["text_features_c"]).cuda()
    cos = torch.nn.functional.cosine_similarity(fc[0].float(), stored.float(), dim=0).item()
    if cos < 0.999:
        bad += 1; print(f"  [MISMATCH] row {i} cos={cos:.4f}")
print(f"  pairing audit: {20-bad}/20 ok")
if bad:
    raise SystemExit("[FAIL] NPZ/TSV pairing audit failed")
PYEOF

# ---- Step 4: Stage 1 (c2p0 arguments, verbatim) -----------------------------
if [ ! -f "$S1_EMA" ]; then
  log "[Step 4] Stage 1 $S1_EXP"
  "$PY" set_training_stage.py --stage 1
  S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=fluxaudio_s exp_id="$S1_EXP" \
    num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
    "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NPZ_TSV" \
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "++data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
    "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv" \
    "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4" \
    "++data.AudioCaps_val_npz.gt_cache=null" \
    "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
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
    data=meanaudio model=meanaudio_s exp_id="$S2_EXP" \
    num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
    "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NPZ_TSV" \
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "++data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
    "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv" \
    "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4" \
    "++data.AudioCaps_val_npz.gt_cache=null" \
    2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 5] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }

# ---- Step 6: eval under the arm-comparison protocol ------------------------
# MusicCaps 5521 / MeanFlow 25 / CFG 3.0 / fidelity negative / NoMask /
# seed 42 / full precision / --no_q. Same protocol as the slot0-vs-fulltrack
# table, so this number is comparable to slot0 full 0.2605 and NOT to the
# canonical CFG 0 table.
NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
OUT="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_cfg3_neg"
mkdir -p "$OUT/audio"
HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE" -lt 5400 ]; then
  log "[Step 6] generating (have $HAVE)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$OUT/audio" --tsv "$DATA/musiccaps_test.tsv" --use_meanflow \
    --num_steps 25 --cfg_strength 3.0 --negative_prompt "$NEG" \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_gen.log"
fi
GOT=$(find "$OUT/audio" -name '*.flac' | wc -l)
log "generated $GOT / 5521"
[ "$GOT" -ge 5400 ] || { log "[FAIL] only $GOT clips"; exit 4; }

MC_TSV="$DATA/musiccaps_test.tsv"
"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$OUT/audio" --tsv "$MC_TSV" \
  --exp_name "${EXP_PREFIX}_mc_mf25_cfg3_neg" \
  --out_dir "$OUT" 2>&1 | tee "$STATE/eval_metrics.log"

# ---- Step 7: canonical CFG 0 eval + harness completion evidence ------------
# The queue's harn_guest.py only writes terminal status "completed" when the
# contract's completion_evidence.cfg0_report validates (cfg_strength == 0,
# 5521 rows, the five required metrics, checkpoint resolving to the EMA).
# Without it a finished run is classified "held" -- the trap recorded in
# memory/project_cfg0_harness_unsatisfiable_gates_2026_08_29.md. Producing the
# CFG 0 number anyway is worth it on its own: it puts this arm on the canonical
# table next to slot0 full 0.2149.
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
