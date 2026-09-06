#!/usr/bin/env bash
# Train one controlled Phase-8 S2-only Q arm from the completed clean-NoQ S1.
# Required environment:
#   Q_MODE=real|shuffled
#   EXP_PREFIX=unique artifact prefix
#   TRAIN_TSV=aligned TSV with the selected q assignment

set -euo pipefail

WORK_DIR=/home/kojiek/MeanAudio
DATA_DIR=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_DIR=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz
GT_CACHE="$DATA_DIR/npz_cache_train.txt"
MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py
SOURCE_S1_ID=phase8_catalog_matched_noq_stage1_400000
SOURCE_S1="$WORK_DIR/exps/$SOURCE_S1_ID/${SOURCE_S1_ID}_ckpt_last.pth"
S1_ITERATIONS=400000
S2_ITERATIONS=200000
FINAL_ITERATION=600000
EXPECTED_ROWS=251599

Q_MODE="${Q_MODE:?Q_MODE must be real or shuffled}"
EXP_PREFIX="${EXP_PREFIX:?EXP_PREFIX is required}"
TRAIN_TSV="${TRAIN_TSV:?TRAIN_TSV is required}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
case "$Q_MODE" in real|shuffled) ;; *) echo "[FAIL] Q_MODE=$Q_MODE" >&2; exit 2 ;; esac
case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] EXPERIMENT_RUN_MODE=$RUN_MODE" >&2; exit 2 ;; esac

EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"
EXP_DIR="$WORK_DIR/exps/$EXP_S2"
S2_CKPT="$EXP_DIR/${EXP_S2}_ckpt_last.pth"
S2_EMA="$EXP_DIR/${EXP_S2}_ema_final.pth"
CONTRACT="$LOG_DIR/${EXP_PREFIX}_contract.json"
FINAL_AUDIT="$LOG_DIR/phase8_s2_q_ablation_monitor/${EXP_PREFIX}_FINAL_AUDIT.json"

cd "$WORK_DIR"
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

for required in "$SOURCE_S1" "$TRAIN_TSV" "$GT_CACHE" "$NPZ_DIR/MANIFEST.tsv" \
    "$NPZ_DIR/FULL_VALIDATION.json" "$NPZ_DIR/FULL_GATE_PASSED.json" \
    "$MUSICCAPS" "$EVAL_SCRIPT"; do
    if [ ! -e "$required" ]; then
        echo "[FAIL] missing required input: $required" >&2
        exit 2
    fi
done

python set_training_stage.py --stage 2

python - "$SOURCE_S1" "$TRAIN_TSV" "$GT_CACHE" "$NPZ_DIR" \
    "$EXPECTED_ROWS" "$Q_MODE" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

s1, tsv, cache, npz_dir = map(Path, sys.argv[1:5])
expected = int(sys.argv[5])
q_mode = sys.argv[6]
state = torch.load(s1, map_location="cpu", weights_only=False)
if state.get("it") != 400000:
    raise SystemExit(f"[FAIL] source S1 iteration={state.get('it')}, expected 400000")
if not all(key in state for key in ("weights", "optimizer", "scheduler", "ema")):
    raise SystemExit("[FAIL] source S1 is not a resumable training checkpoint")

with tsv.open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with cache.open() as handle:
    names = [line.strip() for line in handle if line.strip()]
with (npz_dir / "MANIFEST.tsv").open() as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))
if not (len(rows) == len(names) == len(manifest) == expected):
    raise SystemExit(
        f"[FAIL] row mismatch tsv={len(rows)} cache={len(names)} manifest={len(manifest)}"
    )
q = [int(row["q_level"]) for row in rows]
if any(value < 0 or value > 9 for value in q):
    raise SystemExit("[FAIL] q_level outside 0..9")
if len(set(q)) < 5:
    raise SystemExit(f"[FAIL] degenerate Q support: {sorted(set(q))}")

for i in (0, 1, 100, 1000, 10000, expected - 1):
    item = manifest[i]
    if (item["clip_id"], item["npz_fname"]) != (rows[i]["id"], names[i]):
        raise SystemExit(f"[FAIL] alignment mismatch at row {i}")
    with np.load(npz_dir / names[i]) as data:
        if str(data["clip_id"].item()) != rows[i]["id"]:
            raise SystemExit(f"[FAIL] embedded clip id mismatch at row {i}")

validation = json.loads((npz_dir / "FULL_VALIDATION.json").read_text())
gate = json.loads((npz_dir / "FULL_GATE_PASSED.json").read_text())
if validation.get("status") != "passed" or gate.get("status") != "passed":
    raise SystemExit("[FAIL] structural/semantic cache gate is not passed")
print(f"[OK] {q_mode} Q preflight rows={len(rows):,} histogram={dict(sorted(Counter(q).items()))}")
PY

if [ "$Q_MODE" = shuffled ]; then
    SHUFFLE_MANIFEST="${TRAIN_TSV%.tsv}.manifest.json"
    python - "$SHUFFLE_MANIFEST" "$TRAIN_TSV" "$DATA_DIR/phase8_legacy_catalog_train.tsv" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, shuffled_path, source_path = map(Path, sys.argv[1:])
if not manifest_path.is_file():
    raise SystemExit(f"[FAIL] shuffled-Q manifest missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
if manifest.get("output_sha256") != sha(shuffled_path):
    raise SystemExit("[FAIL] shuffled TSV hash disagrees with manifest")
if manifest.get("input_sha256") != sha(source_path):
    raise SystemExit("[FAIL] source TSV changed after shuffled control creation")
if manifest.get("changed_rows", 0) < manifest.get("rows", 0) // 2:
    raise SystemExit("[FAIL] shuffled control did not break enough Q assignments")
print(f"[OK] immutable shuffled-Q manifest: {manifest_path}")
PY
fi

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$EXP_DIR" \
        "$WORK_DIR/eval_output/${EXP_S2}_musiccaps_q9" \
        "$WORK_DIR/eval_output/${EXP_S2}_musiccaps_q6" \
        "$WORK_DIR/eval_output/metrics/${EXP_S2}_musiccaps_q9" \
        "$WORK_DIR/eval_output/metrics/${EXP_S2}_musiccaps_q6" \
        "$LOG_DIR/${EXP_S2}.log" "$LOG_DIR/${EXP_S2}_musiccaps_q9_eval.log" \
        "$LOG_DIR/${EXP_S2}_musiccaps_q6_eval.log" "$CONTRACT"; do
        if [ -f "$path" ]; then
            conflicts+=("$path")
        elif [ -d "$path" ] && [ -n "$(find "$path" -mindepth 1 -print -quit 2>/dev/null)" ]; then
            conflicts+=("$path")
        fi
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        echo "[FAIL] fresh run found existing artifacts:" >&2
        printf '  %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi

mkdir -p "$LOG_DIR" "$EXP_DIR" "$(dirname "$FINAL_AUDIT")"

python - "$CONTRACT" "$EXP_PREFIX" "$Q_MODE" "$TRAIN_TSV" "$SOURCE_S1" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out, prefix, q_mode, train_tsv, source_s1 = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
critical = [
    "scripts/training_pipelines/train_pipeline_phase8_s2_q_ablation.sh",
    "meanaudio/runner_meanflow.py",
    "meanaudio/model/mean_flow.py",
    "meanaudio/model/networks.py",
    "eval.py",
]
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "prefix": prefix,
    "q_mode": q_mode,
    "source_s1_checkpoint": source_s1,
    "source_s1_iteration": 400000,
    "source_s1_sha256": sha(source_s1),
    "stage1_use_q_conditioning": False,
    "stage2_use_q_conditioning": True,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "train_tsv": train_tsv,
    "train_tsv_sha256": sha(train_tsv),
    "gt_cache": "/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt",
    "npz_dir": "/mnt/HDD/kojiek/phase8_legacy_matched_npz",
    "expected_rows": 251599,
    "stage2_final_iteration": 600000,
    "eval_tsv": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
    "eval_primary_q": 9,
    "eval_secondary_q": 6,
    "baseline_clap": 0.1888,
    "historical_best_clap": 0.1998,
    "critical_file_sha256": {rel: sha(root / rel) for rel in critical},
}
path = Path(out)
if path.exists():
    previous = json.loads(path.read_text())
    immutable = set(payload) - {"created_at"}
    drift = [key for key in immutable if previous.get(key) != payload.get(key)]
    if drift:
        raise SystemExit(f"[FAIL] immutable contract drift: {drift}")
    print(f"[OK] immutable contract unchanged: {path}")
else:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(f"[OK] wrote immutable contract: {path}")
PY

if [ ! -f "$S2_CKPT" ]; then
    python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SOURCE_S1" --s2_out "$S2_CKPT"
fi

if [ ! -f "$S2_EMA" ]; then
    echo "[TRAIN] $Q_MODE S2-only Q: $EXP_S2 (400k → 600k)"
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=meanaudio_s exp_id="$EXP_S2" \
        num_iterations="$FINAL_ITERATION" "lr_schedule_steps=[999999,999999]" \
        +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
        learning_rate=1e-4 linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=10000 save_checkpoint_interval=20000 \
        ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
        save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TRAIN_TSV" \
        "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$TRAIN_TSV" \
        "data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
        "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
        "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
        2>&1 | tee "$LOG_DIR/${EXP_S2}.log"
fi

for q in 9 6; do
    label="q$q"
    out="$WORK_DIR/eval_output/${EXP_S2}_musiccaps_${label}"
    log="$LOG_DIR/${EXP_S2}_musiccaps_${label}_eval.log"
    python eval.py --variant meanaudio_s --model_path "$S2_EMA" \
        --output "$out/audio" --tsv "$MUSICCAPS" --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
        --quality_level "$q" --no_text_attention_mask --full_precision \
        2>&1 | tee "$log"
    python "$EVAL_SCRIPT" --gen_dir "$out/audio" --tsv "$MUSICCAPS" \
        --exp_name "${EXP_S2}_musiccaps_${label}" --num_samples 2048 \
        2>&1 | tee -a "$log"
done

python scripts/audit_phase8_s2_q_ablation.py --prefix "$EXP_PREFIX" \
    --q-mode "$Q_MODE" --phase final --json-out "$FINAL_AUDIT"

echo "[COMPLETE] $EXP_PREFIX passed final contract audit"
