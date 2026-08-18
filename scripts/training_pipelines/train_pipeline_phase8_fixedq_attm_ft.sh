#!/usr/bin/env bash
# One arm of the Phase-8 Fixed-Q prior vs matched-NoQ residual fine-tune.
#
# Required env:
#   ARM=noq|fixedq9
#
# Optional env:
#   EXPERIMENT_RUN_MODE=fresh|resume   (default: fresh)
#   EXP_PREFIX / TRAIN_TSV             (defaults derived from ARM)
#
# Fresh by default.  Resume only when EXPERIMENT_RUN_MODE=resume is set
# explicitly.  Does not invent a third arm.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz
GT_CACHE="$DATA/npz_cache_train.txt"
MUSICCAPS="$DATA/musiccaps_test.tsv"
CATALOG_TSV="$DATA/phase8_legacy_catalog_train.tsv"
FIXEDQ9_TSV="$DATA/phase8_legacy_catalog_train_fixedq9.tsv"
FIXEDQ9_MANIFEST="$DATA/phase8_legacy_catalog_train_fixedq9.manifest.json"
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py
SOURCE_ID=phase8_catalog_matched_noq_stage2_200000
SOURCE="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ckpt_last.pth"
SOURCE_EMA="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ema_final.pth"
SOURCE_IT=600000
FT_ITERS=100000
FINAL_IT=700000
LR=3e-5
SEED=14159265
EXPECTED_ROWS=251599
MONITOR_STATE="$LOG_ROOT/phase8_fixedq_attm_monitor"

ARM="${ARM:?ARM is required (noq|fixedq9)}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
case "$ARM" in
    noq|fixedq9) ;;
    *)
        echo "[FAIL] ARM=$ARM (expected noq|fixedq9)" >&2
        exit 2
        ;;
esac
case "$RUN_MODE" in
    fresh|resume) ;;
    *)
        echo "[FAIL] EXPERIMENT_RUN_MODE=$RUN_MODE (expected fresh|resume)" >&2
        exit 2
        ;;
esac

if [ "$ARM" = "noq" ]; then
    DEFAULT_PREFIX=phase8_matched_noq_ft100k
    DEFAULT_TSV="$CATALOG_TSV"
    USE_Q=false
    INIT_MODE=noq
    EVAL_LABEL=noq
    EVAL_Q_FLAG=(--no_q)
else
    DEFAULT_PREFIX=phase8_fixedq9_prior_ft100k
    DEFAULT_TSV="$FIXEDQ9_TSV"
    USE_Q=true
    INIT_MODE=fixedq9
    EVAL_LABEL=q9
    EVAL_Q_FLAG=(--quality_level 9)
fi

EXP_PREFIX="${EXP_PREFIX:-$DEFAULT_PREFIX}"
TRAIN_TSV="${TRAIN_TSV:-$DEFAULT_TSV}"
EXP_ID="${EXP_PREFIX}_stage2_ft${FT_ITERS}"
EXP_DIR="$ROOT/exps/$EXP_ID"
CKPT="$EXP_DIR/${EXP_ID}_ckpt_last.pth"
EMA="$EXP_DIR/${EXP_ID}_ema_final.pth"
LOG="$LOG_ROOT/${EXP_ID}.log"
CONTRACT="$LOG_ROOT/${EXP_PREFIX}_contract.json"
INIT_MANIFEST="$LOG_ROOT/${EXP_PREFIX}_init.json"
FINAL_AUDIT="$MONITOR_STATE/${EXP_PREFIX}_FINAL_AUDIT.json"
EVAL_OUT="$ROOT/eval_output/${EXP_ID}_musiccaps_${EVAL_LABEL}"
EVAL_METRICS="$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_${EVAL_LABEL}"
EVAL_LOG="$LOG_ROOT/${EXP_ID}_musiccaps_${EVAL_LABEL}_eval.log"

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

for path in "$SOURCE" "$SOURCE_EMA" "$TRAIN_TSV" "$GT_CACHE" "$MUSICCAPS" \
    "$NPZ_DIR/MANIFEST.tsv" "$NPZ_DIR/FULL_VALIDATION.json" \
    "$NPZ_DIR/FULL_GATE_PASSED.json" "$EVAL_SCRIPT" \
    "$ROOT/scripts/init_phase8_fixedq_attm_checkpoint.py" \
    "$ROOT/scripts/audit_phase8_fixedq_attm_ft.py"; do
    if [ ! -e "$path" ]; then
        echo "[FAIL] missing $path" >&2
        exit 2
    fi
done

if [ "$ARM" = "fixedq9" ]; then
    if [ ! -f "$FIXEDQ9_MANIFEST" ]; then
        echo "[FAIL] fixedq9 TSV manifest missing: $FIXEDQ9_MANIFEST" >&2
        exit 2
    fi
fi

python - "$SOURCE" "$TRAIN_TSV" "$NPZ_DIR" "$GT_CACHE" "$EXPECTED_ROWS" "$ARM" \
    "$CATALOG_TSV" "$FIXEDQ9_MANIFEST" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

source, tsv, npz, cache = map(Path, sys.argv[1:5])
expected = int(sys.argv[5])
arm = sys.argv[6]
catalog = Path(sys.argv[7])
fixed_manifest = Path(sys.argv[8])

state = torch.load(source, map_location="cpu", weights_only=False)
if state.get("it") != 600000:
    raise SystemExit(f"[FAIL] baseline checkpoint it={state.get('it')}")
if not all(key in state for key in ("weights", "ema")):
    raise SystemExit("[FAIL] baseline is not a resumable MeanAudio checkpoint")
if "q_embed.weight" not in state["weights"]:
    raise SystemExit("[FAIL] baseline lacks q_embed.weight")

rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
names = [line.strip() for line in cache.open() if line.strip()]
if len(rows) != expected or len(names) != expected:
    raise SystemExit(f"[FAIL] row count tsv={len(rows)} cache={len(names)}")

q = [int(row["q_level"]) for row in rows]
support = sorted(set(q))
if arm == "fixedq9":
    if support != [9]:
        raise SystemExit(f"[FAIL] fixedq9 TSV Q support must be [9], got {support}")
    if not fixed_manifest.is_file():
        raise SystemExit(f"[FAIL] missing fixedq9 manifest: {fixed_manifest}")
    man = json.loads(fixed_manifest.read_text())
    if man.get("unique_q_support") != [9] or man.get("rows") != expected:
        raise SystemExit(f"[FAIL] fixedq9 manifest invalid: {man}")
else:
    if min(q) < 0 or max(q) > 9:
        raise SystemExit(f"[FAIL] invalid Q values: {Counter(q)}")
    if len(support) < 5:
        raise SystemExit(f"[FAIL] noq arm expects multi-level catalog Q, got {support}")

for gate in (npz / "FULL_VALIDATION.json", npz / "FULL_GATE_PASSED.json"):
    if json.loads(gate.read_text()).get("status") != "passed":
        raise SystemExit(f"[FAIL] cache gate not passed: {gate}")

if arm == "fixedq9":
    # Row order / id / caption must match catalog exactly.
    cat_rows = list(csv.DictReader(catalog.open(), delimiter="\t"))
    if len(cat_rows) != len(rows):
        raise SystemExit("[FAIL] fixedq9 vs catalog row count mismatch")
    for i, (a, b) in enumerate(zip(cat_rows, rows)):
        if a["id"] != b["id"] or a["caption"] != b["caption"]:
            raise SystemExit(f"[FAIL] fixedq9 id/caption drift at row {i}")

print(
    f"[OK] baseline it=600000; arm={arm}; rows={len(rows)}; "
    f"Q={dict(sorted(Counter(q).items()))}"
)
PY

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$EXP_DIR" "$CONTRACT" "$INIT_MANIFEST" \
        "$EVAL_OUT" "$EVAL_METRICS" "$EVAL_LOG" "$FINAL_AUDIT"; do
        if [ -e "$path" ]; then
            conflicts+=("$path")
        fi
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        printf '[FAIL] fresh artifact exists: %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi

mkdir -p "$EXP_DIR" "$MONITOR_STATE"
SOURCE_SHA=$(sha256sum "$SOURCE" | awk '{print $1}')

if [ ! -f "$CKPT" ]; then
    if [ "$RUN_MODE" = resume ]; then
        echo "[FAIL] resume requested but checkpoint missing: $CKPT" >&2
        exit 2
    fi
    python scripts/init_phase8_fixedq_attm_checkpoint.py \
        --source "$SOURCE" \
        --output "$CKPT" \
        --manifest "$INIT_MANIFEST" \
        --mode "$INIT_MODE" \
        --expected-it "$SOURCE_IT" \
        --source-sha256 "$SOURCE_SHA"
fi

python - "$CONTRACT" "$EXP_PREFIX" "$ARM" "$TRAIN_TSV" "$SOURCE" \
    "$SOURCE_EMA" "$SOURCE_SHA" "$INIT_MANIFEST" "$USE_Q" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out,
    prefix,
    arm,
    train_tsv,
    source,
    source_ema,
    source_sha,
    init_manifest,
    use_q_raw,
) = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")
use_q = use_q_raw.lower() == "true"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


critical = [
    "scripts/init_phase8_fixedq_attm_checkpoint.py",
    "scripts/training_pipelines/train_pipeline_phase8_fixedq_attm_ft.sh",
    "scripts/audit_phase8_fixedq_attm_ft.py",
    "scripts/preprocess/make_phase8_fixedq9_tsv.py",
    "meanaudio/runner_meanflow.py",
    "meanaudio/model/mean_flow.py",
    "meanaudio/model/networks.py",
    "eval.py",
]
init_label = (
    "copy_q10_exactly_to_q0_through_q9"
    if arm == "fixedq9"
    else "preserve_q_embed_matched_optimizer_reset"
)
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_fixedq_attm_chain",
    "prefix": prefix,
    "arm": arm,
    "source_checkpoint": source,
    "source_checkpoint_sha256": source_sha,
    "source_ema": source_ema,
    "source_iteration": 600000,
    "initialization": init_label,
    "init_manifest": init_manifest,
    "train_tsv": train_tsv,
    "train_tsv_sha256": sha(train_tsv),
    "use_q_conditioning": use_q,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "fine_tune_iterations": 100000,
    "final_iteration": 700000,
    "learning_rate": 3e-5,
    "batch_size": 8,
    "accumulation_steps": 1,
    "seed": 14159265,
    "eval_tsv": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
    "eval_mode": "quality_level_9" if arm == "fixedq9" else "no_q",
    "eval_primary_q": 9 if arm == "fixedq9" else None,
    "clap_metric_checkpoint": "music_speech_audioset_epoch_15_esc_89.98.pt",
    "baseline_clap": 0.1888,
    "restoration_target_clap": 0.1900,
    "critical_file_sha256": {rel: sha(root / rel) for rel in critical},
    "contracts": {
        "q_none_maps_to": 10,
        "meanflow_unconditional_q": 10,
        "fixedq9_conditional_rows_use": 9,
        "primary_checkpoint_iteration": 700000,
        "no_checkpoint_cherrypick": True,
        "attm_official_90_14_blocked_until_100_prompt_csv": True,
    },
}
path = Path(out)
if path.exists():
    old = json.loads(path.read_text())
    drift = [k for k, v in payload.items() if k != "created_at" and old.get(k) != v]
    if drift:
        raise SystemExit(f"[FAIL] immutable contract drift: {drift}")
else:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
print(f"[OK] contract {path}")
PY

python set_training_stage.py --stage 2

if [ ! -f "$EMA" ]; then
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=meanaudio_s exp_id="$EXP_ID" \
        num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
        "+use_q_conditioning=${USE_Q}" batch_size=8 +accumulation_steps=1 \
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
        2>&1 | tee -a "$LOG"
fi

if [ ! -f "$EVAL_METRICS/metrics.txt" ]; then
    python eval.py --variant meanaudio_s --model_path "$EMA" \
        --output "$EVAL_OUT/audio" --tsv "$MUSICCAPS" --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
        "${EVAL_Q_FLAG[@]}" --no_text_attention_mask --full_precision \
        2>&1 | tee "$EVAL_LOG"
    python "$EVAL_SCRIPT" --gen_dir "$EVAL_OUT/audio" --tsv "$MUSICCAPS" \
        --exp_name "${EXP_ID}_musiccaps_${EVAL_LABEL}" --num_samples 2048 \
        2>&1 | tee -a "$EVAL_LOG"
fi

python scripts/audit_phase8_fixedq_attm_ft.py --prefix "$EXP_PREFIX" \
    --arm "$ARM" --phase final --json-out "$FINAL_AUDIT"
echo "[COMPLETE] arm=$ARM prefix=$EXP_PREFIX"
