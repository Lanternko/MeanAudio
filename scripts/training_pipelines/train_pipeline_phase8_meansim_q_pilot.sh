#!/usr/bin/env bash
# Quarter-scale S2 pilot for the repaired MeanSimilarity-Q pathway.
#
# Stage 1 is the completed 400k catalog-matched No-Q checkpoint (effective q=10).
# Stage 2 runs only 50k updates by default.  At the transition, q=0..9 are
# initialized exactly from the trained q=10 row, avoiding a random embedding
# discontinuity.  Run aligned and shuffled arms from the same source.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz
GT_CACHE="$DATA/npz_cache_train.txt"
SOURCE_ID=phase8_catalog_matched_noq_stage1_400000
SOURCE="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ckpt_last.pth"
SOURCE_IT=400000
S2_UPDATES="${S2_UPDATES:-50000}"
FINAL_IT=$((SOURCE_IT + S2_UPDATES))
LR="${LR:-1e-4}"
SEED="${SEED:-14159265}"
EXPECTED_ROWS=251599
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py
EVAL_TSV="${EVAL_TSV:-$DATA/musiccaps_test.tsv}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-true}"
ARM="${ARM:?ARM must be aligned or shuffled}"

case "$ARM" in
    aligned)
        TRAIN_TSV="$DATA/phase8_legacy_catalog_train_meansim_aligned.tsv"
        Q_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_aligned.manifest.json"
        ;;
    shuffled)
        TRAIN_TSV="$DATA/phase8_legacy_catalog_train_meansim_aligned_q_shuffled_seed424242.tsv"
        Q_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_aligned_q_shuffled_seed424242.manifest.json"
        ;;
    *)
        echo "[FAIL] ARM=$ARM; expected aligned or shuffled" >&2
        exit 2
        ;;
esac
case "$PREFLIGHT_ONLY" in true|false) ;; *)
    echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2
    exit 2
esac
if [ "$S2_UPDATES" -le 0 ]; then
    echo "[FAIL] S2_UPDATES must be positive" >&2
    exit 2
fi

EXP_PREFIX="phase8_meansim_qpilot_${ARM}"
EXP_ID="${EXP_PREFIX}_s2_${S2_UPDATES}"
EXP_DIR="$ROOT/exps/$EXP_ID"
CKPT="$EXP_DIR/${EXP_ID}_ckpt_last.pth"
EMA="$EXP_DIR/${EXP_ID}_ema_final.pth"
INIT_LOG="$LOG_ROOT/${EXP_ID}_qinit.log"
TRAIN_LOG="$LOG_ROOT/${EXP_ID}.log"
CONTRACT="$LOG_ROOT/${EXP_ID}_contract.json"

for path in "$SOURCE" "$TRAIN_TSV" "$Q_MANIFEST" "$GT_CACHE" "$NPZ_DIR/MANIFEST.tsv" \
    "$NPZ_DIR/FULL_VALIDATION.json" "$NPZ_DIR/FULL_GATE_PASSED.json" \
    "$EVAL_TSV" "$EVAL_SCRIPT"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate

python - "$SOURCE" "$TRAIN_TSV" "$Q_MANIFEST" "$GT_CACHE" "$NPZ_DIR" \
    "$EXPECTED_ROWS" "$ARM" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

source, tsv, q_manifest, cache, npz_dir = map(Path, sys.argv[1:6])
expected = int(sys.argv[6])
arm = sys.argv[7]

state = torch.load(source, map_location="cpu", weights_only=False)
if state.get("it") != 400000:
    raise SystemExit(f"[FAIL] source iteration={state.get('it')}, expected 400000")
if not all(key in state for key in ("weights", "ema", "optimizer", "scheduler")):
    raise SystemExit("[FAIL] source is not a resumable Stage-1 checkpoint")

with tsv.open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with cache.open() as handle:
    names = [line.strip() for line in handle if line.strip()]
with (npz_dir / "MANIFEST.tsv").open() as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))
if not (len(rows) == len(names) == len(manifest) == expected):
    raise SystemExit(
        f"[FAIL] row mismatch tsv={len(rows)} cache={len(names)} "
        f"manifest={len(manifest)} expected={expected}"
    )

q = [int(row["q_level"]) for row in rows]
if min(q) < 0 or max(q) > 9:
    raise SystemExit("[FAIL] q_level outside 0..9")
hist = dict(sorted(Counter(q).items()))
if sorted(hist) != [3, 4, 5, 6, 7, 8, 9]:
    raise SystemExit(f"[FAIL] unexpected repaired MeanSim-Q support: {hist}")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

q_contract = json.loads(q_manifest.read_text())
expected_hash = (
    q_contract.get("output_sha256")
    if arm == "aligned"
    else q_contract.get("output_sha256")
)
if expected_hash != sha(tsv):
    raise SystemExit("[FAIL] training TSV hash disagrees with Q manifest")
if arm == "aligned":
    if q_contract.get("signal") != (
        "credibility_analysis.mean_similarity from the actual catalog clip id"
    ):
        raise SystemExit("[FAIL] aligned arm lacks actual-clip MeanSimilarity provenance")
    if q_contract.get("changed_rows", 0) < 100000:
        raise SystemExit("[FAIL] aligned manifest does not record the row-position-Q repair")
else:
    if q_contract.get("method") != "fixed-seed permutation of q_level only":
        raise SystemExit("[FAIL] shuffled arm lacks fixed-seed permutation provenance")

for i in (0, 1, 100, 1000, 10000, expected - 1):
    if (manifest[i]["clip_id"], manifest[i]["npz_fname"]) != (rows[i]["id"], names[i]):
        raise SystemExit(f"[FAIL] row/cache/manifest mismatch at {i}")
    with np.load(npz_dir / names[i]) as data:
        if str(data["clip_id"].item()) != rows[i]["id"]:
            raise SystemExit(f"[FAIL] embedded clip id mismatch at {i}")

for gate_name in ("FULL_VALIDATION.json", "FULL_GATE_PASSED.json"):
    gate = json.loads((npz_dir / gate_name).read_text())
    if gate.get("status") != "passed":
        raise SystemExit(f"[FAIL] cache gate is not passed: {gate_name}")
print(f"[OK] {arm} preflight rows={len(rows):,} q_histogram={hist}")
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] No checkpoint, training, evaluation, or GPU process was started."
    echo "Run explicitly with: ARM=$ARM PREFLIGHT_ONLY=false bash $0"
    exit 0
fi

for path in "$EXP_DIR" "$CONTRACT" "$INIT_LOG" "$TRAIN_LOG"; do
    if [ -e "$path" ]; then
        echo "[FAIL] fresh-only pilot found existing artifact: $path" >&2
        exit 2
    fi
done
mkdir -p "$EXP_DIR" "$LOG_ROOT"

python - "$CONTRACT" "$ARM" "$TRAIN_TSV" "$Q_MANIFEST" "$SOURCE" \
    "$S2_UPDATES" "$FINAL_IT" "$LR" "$SEED" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out, arm, train_tsv, q_manifest, source = sys.argv[1:6]
s2_updates, final_it, lr, seed = sys.argv[6:]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "arm": arm,
    "source_checkpoint": source,
    "source_checkpoint_sha256": sha(source),
    "source_iteration": 400000,
    "stage1_effective_q": 10,
    "stage1_use_q_conditioning": False,
    "stage2_use_q_conditioning": True,
    "q_transition_initialization": "copy_q10_exactly_to_q0_through_q9",
    "train_tsv": train_tsv,
    "train_tsv_sha256": sha(train_tsv),
    "q_manifest": q_manifest,
    "q_manifest_sha256": sha(q_manifest),
    "s2_updates": int(s2_updates),
    "final_iteration": int(final_it),
    "learning_rate": float(lr),
    "seed": int(seed),
    "use_text_attention_mask": False,
    "multi_cap": False,
    "eval_q": [9, 6, 10, 0],
}
path = Path(out)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
print(f"[OK] wrote pilot contract: {path}")
PY

python set_training_stage.py --stage 2
python migrate_stage1_to_stage2_ckpt.py \
    --s1_ckpt "$SOURCE" --s2_out "$CKPT" --q-init copy-null | tee "$INIT_LOG"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=meanaudio_s exp_id="$EXP_ID" \
    num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
    +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=5000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TRAIN_TSV" \
    "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
    "data.AudioCaps_val_npz.tsv=$TRAIN_TSV" \
    "data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
    2>&1 | tee "$TRAIN_LOG"

for q in 9 6 10 0; do
    out="$ROOT/eval_output/${EXP_ID}_musiccaps_q$q"
    eval_log="$LOG_ROOT/${EXP_ID}_musiccaps_q${q}_eval.log"
    python eval.py --variant meanaudio_s --model_path "$EMA" \
        --output "$out/audio" --tsv "$EVAL_TSV" --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
        --quality_level "$q" --no_text_attention_mask --full_precision \
        2>&1 | tee "$eval_log"
    python "$EVAL_SCRIPT" --gen_dir "$out/audio" --tsv "$EVAL_TSV" \
        --exp_name "${EXP_ID}_musiccaps_q$q" --num_samples 5521 \
        2>&1 | tee -a "$eval_log"
done

echo "[COMPLETE] $EXP_ID"
