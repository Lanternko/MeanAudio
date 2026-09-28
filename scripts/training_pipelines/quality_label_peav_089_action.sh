#!/bin/bash
# 089 PE-AV quality-label prefix arm, one training seed (prereg docs/experiments/quality_label_peav_089_20260929.md).
#
# = 081 (quality_label_081_action.sh) with the bottom / top 20% tiers ranked by PE-AV
# cos(caption, training window) instead of AES PQ. Same corpus, prefixes, recipe, seeds,
# eval cells. The control is the same nmv2pair run as 081; all its cells (stock + lqneg +
# hqpos, raw and lvl30) already exist from 081, so only the 5 arm cells are evaluated.
#
# Steps
#   0  archive earlier 089 seeds' finished run dirs to HDD if it has room (it usually does not)
#   1  PE-AV corpus scores (~/venvs/peav, first seed only, ~8 h, resumable), then the builder
#      `all` (window gate, tiers, overlays: 081's where identical, else new on NVMe; verify)
#   2  Stage 1; migrate; drop S1 shadows, thin S1 EMA snapshots, archive the S1 dir
#   3  Stage 2; drop S2 shadows, thin S2 EMA snapshots
#   4  eval, arm only: stock cfg0 + cfg3_neg, hqpos cfg0, cfg3_lqneg, hqpos cfg3_lqneg;
#      every cell: -30 LUFS rescore, then its audio is deleted.
#
# Usage: QL_SEED=<seed> quality_label_peav_089_action.sh     (QL_VERIFY_ONLY=1 stops after step 1)
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
MC_TSV="$DATA/musiccaps_test.tsv"
PY="$HOME/venvs/dac/bin/python"
PEAV_PY="$HOME/venvs/peav/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEED="${QL_SEED:?QL_SEED}"
SEEDS_ALL=(14159265 16180339 27182818)
S1_UPDATES=100000; S2_ADD=50000; FINAL_IT=$((S1_UPDATES + S2_ADD)); LR=1e-4; BATCH=8
ROOT="$HOME/exps_nvme/quality_label_089"
INPUTS="$ROOT/arm_inputs"
TRAIN_TSV="$INPUTS/train.tsv"; CACHE_LIST="$INPUTS/cache_train.txt"; MANIFEST="$INPUTS/manifest.json"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"; OVERLAY="$ROOT/overlay_farm"
HQ_TSV="$HOME/exps_nvme/quality_label_081/musiccaps_test_hqprefix.tsv"
LQ_NEG="Low quality recording."
EVAL_ROOT="$HOME/eval_output_nvme"

prefix_of(){ echo "phase8_qwen_caption2p0_slot0clean_peavlabel_noq_quarter_s$1"; }
EXP_PREFIX="$(prefix_of "$SEED")"
CTRL_PREFIX="phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s${SEED}"
CTRL_EMA="$WORK_DIR/exps/${CTRL_PREFIX}_stage2_50000/${CTRL_PREFIX}_stage2_50000_ema_final.pth"
S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"; S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
free_b(){ df -B1 --output=avail "$1" | tail -1; }

# Archive one finished run dir (exps_nvme/<name>) to the HDD if there is room; warn otherwise.
archive_dir(){
  local name="$1" src="$HOME/exps_nvme/$1" need
  [ -d "$src" ] && [ ! -L "$src" ] || return 0
  need=$(( $(du -sb "$src" | cut -f1) + 5000000000 ))
  if [ "$(free_b /mnt/HDD)" -lt "$need" ]; then
    log "[WARN] HDD too full to archive $name; left on NVMe"; return 0
  fi
  bash "$WORK_DIR/scripts/archive_exps_nvme_to_hdd.sh" "$name" || true
  [ -L "$src" ] && log "archived $name" || log "[WARN] archive of $name did not complete; left on NVMe"
}

# Keep only the S1 quartile anchors / the S2 snapshots 080-style autoguidance reads.
thin_ema(){
  local d="$1/ema_ckpts" keep="$2" p
  [ -d "$d" ] || return 0
  for p in "$d"/*.pt; do
    [ -e "$p" ] || continue
    case " $keep " in *" $(basename "$p") "*) ;; *) rm -f -- "$p" ;; esac
  done
}

# Delete a run's own shadow copies (only called once that run's ema_final exists).
drop_shadows(){ rm -f -- "$1"/*_ckpt_shadow.pth "$1"/*_shadow.pth; }

log "[Step 0] 089 seed $SEED ($EXP_PREFIX)"
for s in "${SEEDS_ALL[@]}"; do
  [ "$s" = "$SEED" ] && continue
  P="$(prefix_of "$s")"
  R="$EVAL_ROOT/${P}_mc_mf25_cfg0/${P}_mc_mf25_cfg0_REPORT.json"
  # an earlier seed's S2 dir is the evidence of its own (already classified) queue job
  if [ -f "$R" ]; then archive_dir "${P}_stage2_${S2_ADD}"; fi
done

if [ ! -f "$S2_EMA" ]; then
  if [ -f "$S1_EMA" ]; then NEED=13000000000; else NEED=20000000000; fi
else
  NEED=3000000000
fi
if [ "$(free_b "$HOME")" -lt "$NEED" ]; then
  log "[FAIL] NVMe free $(( $(free_b "$HOME") / 1000000000 ))G < $((NEED / 1000000000))G"; exit 3
fi
[ -f "$CTRL_EMA" ] || { log "[FAIL] control checkpoint missing: $CTRL_EMA"; exit 2; }
for L in "${CTRL_PREFIX}_mc_mf25_cfg0" "${CTRL_PREFIX}_mc_mf25_cfg3_neg" "${CTRL_PREFIX}_mc_mf25_cfg3_lqneg" \
         "${CTRL_PREFIX}_hqpos_mc_mf25_cfg0" "${CTRL_PREFIX}_hqpos_mc_mf25_cfg3_lqneg"; do
  [ -f "$EVAL_ROOT/$L/${L}_REPORT.json" ] || { log "[FAIL] control report missing: $L"; exit 2; }
  compgen -G "$EVAL_ROOT/${L}_lvl30/*/per_clip.tsv" >/dev/null \
    || compgen -G "$EVAL_ROOT/d2_075_lvl30/${L}_lvl30/*/per_clip.tsv" >/dev/null \
    || { log "[FAIL] control lvl30 per_clip missing: $L"; exit 2; }
done

log "[Step 1] PE-AV corpus scores (resumable), then arm inputs (build if missing, then verify)"
mkdir -p "$ROOT"
[ -f /mnt/HDD/kojiek/quality_label_081/overlay_new/0000/120.npz ] || { log "[FAIL] 081 HDD overlays (reused) missing"; exit 2; }
"$PEAV_PY" scripts/preprocess/score_peav_corpus_089.py --root "$ROOT" 2>&1 | tee -a "$ROOT.build.log"
set +e
"$PY" scripts/preprocess/build_quality_label_peav_089_arm_inputs.py all 2>&1 | tee -a "$ROOT.build.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" = 0 ] || { log "[FAIL] builder exit $RC (3 = NVMe too full for the overlays)"; exit "$RC"; }
[ -f "$HQ_TSV" ] || { log "[FAIL] 081 HQ eval TSV missing: $HQ_TSV"; exit 2; }
TRAIN_TSV="$TRAIN_TSV" CACHE_LIST="$CACHE_LIST" MANIFEST="$MANIFEST" OVERLAY="$OVERLAY" NPZ_DIR="$NPZ_DIR" \
HQ_TSV="$HQ_TSV" MC_TSV="$MC_TSV" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, random
import numpy as np
csv.field_size_limit(10**9)
E = os.environ
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(E["MANIFEST"]))
assert m["status"] == "arm_inputs_ready", m["status"]
assert sha(E["TRAIN_TSV"]) == m["train_tsv_sha256"], "[FAIL] train tsv drift"
assert sha(E["CACHE_LIST"]) == m["cache_list_sha256"], "[FAIL] cache list drift"
assert m["text_npz_dir"] == E["OVERLAY"] and m["npz_dir"] == E["NPZ_DIR"]
v = json.load(open(os.path.join(os.path.dirname(E["MANIFEST"]), "verify.json")))
assert v["verified_rows"] == m["rows"]
# the cache list and every id must be the control's, in order
src = json.load(open(m["source_manifest"]))
assert m["cache_list_sha256"] == src["cache_list_sha256"] == m["source_cache_list_sha256"]
rows = list(csv.DictReader(open(E["TRAIN_TSV"], newline=""), delimiter="\t"))
names = [l.strip() for l in open(E["CACHE_LIST"]) if l.strip()]
random.seed(20260926)
for i in random.sample(range(len(rows)), 64):
    a = np.load(f"{E['NPZ_DIR']}/{names[i]}"); t = np.load(f"{E['OVERLAY']}/{names[i]}")
    assert str(a["clip_id"].item()) == rows[i]["id"] == str(t["clip_id"].item()), f"[FAIL] clip_id at {i}"
# HQ eval TSV: same ids in order, caption = prefix + plain caption, no q_level column
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline=""), delimiter="\t"))
mc, hq = rd(E["MC_TSV"]), rd(E["HQ_TSV"])
assert list(hq[0]) == ["id", "caption"] and len(hq) == len(mc) == 5521
assert all(h["id"] == c["id"] and h["caption"] == "High quality recording. " + c["caption"] for h, c in zip(hq, mc))
st = m["stats"]
print(f"  rows={m['rows']} tiers={st['tier_counts']} cut={st['tier_cut']} spearman(peav,pq)={st['spearman_peav_vs_pq']:.3f} "
      f"pq_contrast={st['pq_contrast_high_minus_low']} overlays={st['overlays']} verify ok; HQ tsv ok")
PYEOF
log "[Step 1] inputs verified"
[ "${QL_VERIFY_ONLY:-0}" = 1 ] && exit 0

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

if [ ! -f "$S2_EMA" ]; then
  if [ ! -f "$S2_CKPT" ]; then
    if [ ! -f "$S1_EMA" ]; then
      log "[Step 2] Stage 1 $S1_EXP"
      mkdir -p "$S1_DIR"
      "$PY" set_training_stage.py --stage 1
      S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
      "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        model=fluxaudio_s exp_id="$S1_EXP" num_iterations="$S1_UPDATES" \
        "${COMMON[@]}" "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
    else
      log "[Step 2] S1 already complete"
    fi
    [ -f "$S1_CKPT" ] && [ -f "$S1_EMA" ] || { log "[FAIL] S1 incomplete"; exit 2; }
    mkdir -p "$S2_DIR"
    rm -f -- "$S2_CKPT.tmp"   # a stale tmp would make migrate write a 2.3 GB backup copy
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT.tmp" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
    mv -f -- "$S2_CKPT.tmp" "$S2_CKPT"
  fi
  # S1 is no longer read once the migrated S2 checkpoint exists
  if [ -d "$S1_DIR" ] && [ ! -L "$S1_DIR" ] && [ -f "$S1_EMA" ]; then
    drop_shadows "$S1_DIR"
    thin_ema "$S1_DIR" "0.30000.pt 0.50000.pt 0.80000.pt 0.100000.pt 1.30000.pt 1.50000.pt 1.80000.pt 1.100000.pt"
    archive_dir "$S1_EXP"
  fi

  log "[Step 3] Stage 2 $S2_EXP"
  "$PY" set_training_stage.py --stage 2
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=meanaudio_s exp_id="$S2_EXP" num_iterations="$FINAL_IT" \
    "${COMMON[@]}" "checkpoint=$S2_CKPT" 2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 2-3] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }
restore_stage_1
if [ ! -L "$S2_DIR" ]; then
  drop_shadows "$S2_DIR"
  thin_ema "$S2_DIR" "0.110000.pt 0.130000.pt 1.110000.pt 1.130000.pt"
fi

# ---- eval -------------------------------------------------------------------------------
# rescore a finished cell at -30 LUFS, then drop both audio dirs (reports + per_clip stay)
finish_cell(){
  local label="$1" d="$EVAL_ROOT/$1"
  [ -f "$d/${label}_REPORT.json" ] || { log "[FAIL] missing report $d/${label}_REPORT.json"; exit 4; }
  if ! ls "$d"_lvl30/*/per_clip.tsv >/dev/null 2>&1; then
    [ -d "$d/audio" ] || { log "[FAIL] $label: no audio to rescore and no lvl30 metrics"; exit 4; }
    "$PY" scripts/eval/level_match_rescore.py --cell_dir "$d" --tsv "$MC_TSV" 2>&1 | tee -a "$STATE/eval.log"
  fi
  ls "$d"_lvl30/*/per_clip.tsv >/dev/null 2>&1 || { log "[FAIL] $label lvl30 rescore missing"; exit 4; }
  rm -rf -- "$d/audio" "${d}_lvl30/audio"
}

eval_model(){   # <exp prefix> <ema> <stock:0|1>
  local P="$1" C="$2"
  if [ "$3" = 1 ]; then
    bash scripts/eval/mc_mf25_eval.sh "$P" "$C" --no_q 2>&1 | tee -a "$STATE/eval.log"
    finish_cell "${P}_mc_mf25_cfg0"; finish_cell "${P}_mc_mf25_cfg3_neg"
  fi
  bash scripts/eval/mc_mf25_eval.sh "${P}_hqpos" "$C" --no_q --gen_tsv "$HQ_TSV" cfg0 2>&1 | tee -a "$STATE/eval.log"
  finish_cell "${P}_hqpos_mc_mf25_cfg0"
  bash scripts/eval/mc_mf25_negvariant_eval.sh "$P" "$C" --no_q --neg "$LQ_NEG" --tag lqneg 2>&1 | tee -a "$STATE/eval.log"
  finish_cell "${P}_mc_mf25_cfg3_lqneg"
  bash scripts/eval/mc_mf25_negvariant_eval.sh "${P}_hqpos" "$C" --no_q --gen_tsv "$HQ_TSV" \
    --neg "$LQ_NEG" --tag lqneg 2>&1 | tee -a "$STATE/eval.log"
  finish_cell "${P}_hqpos_mc_mf25_cfg3_lqneg"
}

log "[Step 4] eval arm $EXP_PREFIX (control cells exist from 081)"
eval_model "$EXP_PREFIX" "$S2_EMA" 1
log "[DONE] $EXP_PREFIX"
