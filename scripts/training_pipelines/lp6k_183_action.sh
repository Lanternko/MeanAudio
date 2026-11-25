#!/bin/bash
# 183 arm L: Stage 2 retrain on a 6 kHz-lowpass copy of the nmv2pair control corpus.
#
# Design: docs/experiments/muq_train_exploit_183_20261108.md (arm L). Never touches MuQ.
#   build        -> build_lp6k_corpus_183.py full build (resumable) + G_L2 loader smoke
#   train <seed> -> migrate the control's S1 ckpt_last, train S2 50k on the lp6k latents.
#                   Same TSV, cache list, slot0clean overlay, seed and recipe as
#                   caption2p0_nmv2pair_action.sh slot0clean; only npz_dir changes.
# No canonical 5521 eval: scoring is muq_train_exploit_183.py (Part B 1,000 prompts).
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

ROOT="$HOME/exps_nvme/lp6k_corpus_183"
ARM_INPUTS="$ROOT/arm_inputs"
MANIFEST="$ARM_INPUTS/manifest.json"
NPZ_DIR="$ROOT/npz"
OVERLAY="$HOME/text_overlays/slot0clean"
SRC_INPUTS="$HOME/exps_nvme/slot0clean_nmv2matched/arm_inputs"
TRAIN_TSV="$ARM_INPUTS/phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
CACHE_LIST="$ARM_INPUTS/cache_train.txt"

# Mode comes from $1 or env LP6K_MODE (queue contracts use env so the action path is the
# last element of commands.run). train_all = train for all three seeds in order.
MODE="${1:-${LP6K_MODE:?mode build|train|train_all}}"

verify_inputs() {
  MANIFEST="$MANIFEST" SRC_INPUTS="$SRC_INPUTS" OVERLAY="$OVERLAY" NPZ_DIR="$NPZ_DIR" \
  SRC_NPZ="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, random
import numpy as np
csv.field_size_limit(10**9)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(os.environ["MANIFEST"]))
assert m["status"] == "lp6k_ready", "[FAIL] lp6k manifest not ready"
sm = json.load(open(os.path.join(os.environ["SRC_INPUTS"], "manifest.json")))
assert sha(m["train_tsv"]) == m["train_tsv_sha256"] == sm["train_tsv_sha256"], "[FAIL] tsv differs from control"
assert sha(m["cache_list"]) == m["cache_list_sha256"] == sm["cache_list_sha256"], "[FAIL] cache differs from control"
rows = list(csv.DictReader(open(m["train_tsv"], newline=""), delimiter="\t"))
names = [l.strip() for l in open(m["cache_list"]) if l.strip()]
assert len(rows) == len(names) == m["rows"], "[FAIL] row count"
assert sum(m["status_counts"].values()) == len(names), "[FAIL] status counts do not cover corpus"
# G_L2 loader smoke: the arm reads mean/std from lp6k, text from the slot0clean overlay.
random.seed(20261108)
idx = random.sample(range(len(rows)), 64)
for i in idx:
    z = np.load(os.path.join(os.environ["NPZ_DIR"], names[i]))
    s = np.load(os.path.join(os.environ["SRC_NPZ"], names[i]))
    o = np.load(os.path.join(os.environ["OVERLAY"], names[i]), allow_pickle=True)
    cid = rows[i]["id"]
    assert str(z["clip_id"].item()) == cid == str(s["clip_id"].item()) == str(o["clip_id"].item()), f"[FAIL] clip_id {i}"
    for k in ("mean", "std"):
        assert z[k].shape == s[k].shape and z[k].dtype == np.float32, f"[FAIL] {k} shape/dtype {i}"
        assert np.isfinite(z[k]).all(), f"[FAIL] {k} not finite {i}"
    assert (z["std"] > 0).all(), f"[FAIL] std <= 0 {i}"
    if str(z["lp6k_status"].item()) != "silent":
        assert not np.array_equal(z["mean"], s["mean"]), f"[FAIL] latent unchanged for filtered row {i}"
# Real loader path: the training Dataset with the arm's exact npz_dir / overlay / cache.
import sys
sys.path.insert(0, os.getcwd())
import yaml
from meanaudio.data.extracted_audio import ExtractedAudio
cfg = yaml.safe_load(open("config/base_config.yaml"))
from meanaudio.model.sequence_config import CONFIG_16K
cfg["data_dim"]["latent_seq_len"] = CONFIG_16K.latent_seq_len  # as train.py sets it
ds = ExtractedAudio(tsv_path=m["train_tsv"], concat_text_fc=cfg["concat_text_fc"],
                    data_dim=cfg["data_dim"], npz_dir=os.environ["NPZ_DIR"],
                    text_npz_dir=os.environ["OVERLAY"], repa_npz_dir=None, exclude_cls=False,
                    repa_version=1, gt_cache=m["cache_list"], require_text_overlay=True,
                    multi_cap=False, cap_index_fixed=0, use_text_attention_mask=False)
for i in idx[:16]:
    item = ds[i]
    assert item["id"] == rows[i]["id"], f"[FAIL] dataset id {i}"
print(f"  G_L2 ok: rows={len(rows)} status={m['status_counts']} sampled={len(idx)} dataset_items=16")
PYEOF
}

case "$MODE" in
build)
  if [ -f "$MANIFEST" ] && grep -q '"lp6k_ready"' "$MANIFEST"; then
    log "[Step 1] lp6k corpus already built"
  else
    log "[Step 1] build lp6k corpus"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env -u MKL_NUM_THREADS \
      "$PY" scripts/preprocess/build_lp6k_corpus_183.py --root "$ROOT"
  fi
  log "[Step 2] G_L2 loader smoke"
  verify_inputs
  echo "{\"status\": \"g_l2_pass\", \"time\": \"$(date -u +%FT%TZ)\"}" > "$ROOT/G_L2.json"
  log "[DONE] lp6k build"
  ;;
train)
  SEED="${2:?seed}"
  [ -f "$ROOT/G_L2.json" ] || { log "[FAIL] G_L2 not passed"; exit 2; }
  CTRL="phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s${SEED}"
  S1_CKPT="$WORK_DIR/exps/${CTRL}_stage1_100000/${CTRL}_stage1_100000_ckpt_last.pth"
  EXP_PREFIX="phase8_qwen_caption2p0_slot0clean_lp6k_noq_quarter_s${SEED}"
  S2_EXP="${EXP_PREFIX}_stage2_50000"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
  S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
  STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S2_DIR"
  if [ -f "$S2_EMA" ]; then log "[DONE] $S2_EXP already complete"; exit 0; fi
  FREE=$(df -B1 --output=avail "$WORK_DIR/exps/" | tail -1)
  [ "$FREE" -ge 15000000000 ] || { log "[FAIL] exps disk free < 15G"; exit 3; }
  [ -f "$S1_CKPT" ] || { log "[FAIL] no control S1 $S1_CKPT"; exit 2; }
  log "[Step 1] verify inputs (seed $SEED)"
  verify_inputs
  COMMON=(
    data=meanaudio "lr_schedule_steps=[999999,999999]"
    "+use_q_conditioning=false" batch_size=8 +accumulation_steps=1
    learning_rate=1e-4 seed="$SEED" linear_warmup_steps=1000 num_workers=4
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
  log "[Step 2] Stage 2 $S2_EXP"
  "$PY" set_training_stage.py --stage 2
  if [ ! -f "$S2_CKPT" ]; then
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=meanaudio_s exp_id="$S2_EXP" num_iterations=150000 \
    "${COMMON[@]}" "checkpoint=$S2_CKPT" 2>&1 | tee -a "$STATE/train_s2.log"
  [ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }
  restore_stage_1
  log "[DONE] $S2_EXP"
  ;;
preflight_build)
  FREE=$(df -B1 --output=avail "$HOME" | tail -1 | tr -d ' ')
  [ "$FREE" -ge 32212254720 ] || { log "[PREFLIGHT FAIL] NVMe free < 30G"; exit 1; }
  [ -f "$SRC_INPUTS/manifest.json" ] && [ -d "$OVERLAY" ] || { log "[PREFLIGHT FAIL] control inputs missing"; exit 1; }
  ;;
validate_build)
  [ -f "$ROOT/G_L2.json" ] && grep -q g_l2_pass "$ROOT/G_L2.json" || { log "[INVALID] no G_L2 pass"; exit 1; }
  ;;
preflight_train)
  # 183 runs first in FIFO order: a missing G_L2 means it failed -> hold (rc 1), not wait.
  [ -f "$ROOT/G_L2.json" ] || { log "[PREFLIGHT FAIL] G_L2.json missing (183 not passed)"; exit 1; }
  FREE=$(df -B1 --output=avail "$WORK_DIR/exps/" | tail -1 | tr -d ' ')
  [ "$FREE" -ge 15000000000 ] || { log "[PREFLIGHT FAIL] exps disk free < 15G"; exit 1; }
  ;;
validate_train)
  for S in 14159265 16180339 27182818; do
    E="phase8_qwen_caption2p0_slot0clean_lp6k_noq_quarter_s${S}_stage2_50000"
    [ -f "$WORK_DIR/exps/$E/${E}_ema_final.pth" ] || { log "[INVALID] missing $E ema_final"; exit 1; }
  done
  ;;
train_all)
  for S in 14159265 16180339 27182818; do
    LP6K_MODE= /bin/bash "${BASH_SOURCE[0]}" train "$S"
  done
  log "[DONE] lp6k train_all"
  ;;
*) echo "[FAIL] unknown mode $MODE"; exit 2 ;;
esac
