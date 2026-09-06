#!/usr/bin/env bash
# One arm of the Q-safe residual fine-tuning experiment.
# Required: Q_MODE=real|shuffled, EXP_PREFIX, TRAIN_TSV.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz
GT_CACHE="$DATA/npz_cache_train.txt"
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py
SOURCE_ID=phase8_catalog_matched_noq_stage2_200000
SOURCE="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ckpt_last.pth"
SOURCE_EMA="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ema_final.pth"
SOURCE_IT=600000
FT_ITERS=100000
FINAL_IT=700000
LR=3e-5
# Actual Hydra/default and NoQ baseline seed. Contract previously claimed 42
# but never passed it to train.py; Real-Q already ran with 14159265.
SEED=14159265
EXPECTED_ROWS=251599

Q_MODE="${Q_MODE:?Q_MODE is required}"
EXP_PREFIX="${EXP_PREFIX:?EXP_PREFIX is required}"
TRAIN_TSV="${TRAIN_TSV:?TRAIN_TSV is required}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
case "$Q_MODE" in real|shuffled) ;; *) echo "[FAIL] Q_MODE=$Q_MODE" >&2; exit 2 ;; esac
case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] RUN_MODE=$RUN_MODE" >&2; exit 2 ;; esac

EXP_ID="${EXP_PREFIX}_stage2_ft${FT_ITERS}"
EXP_DIR="$ROOT/exps/$EXP_ID"
CKPT="$EXP_DIR/${EXP_ID}_ckpt_last.pth"
EMA="$EXP_DIR/${EXP_ID}_ema_final.pth"
LOG="$LOG_ROOT/${EXP_ID}.log"
CONTRACT="$LOG_ROOT/${EXP_PREFIX}_contract.json"
INIT_MANIFEST="$LOG_ROOT/${EXP_PREFIX}_qsafe_init.json"
FINAL_AUDIT="$LOG_ROOT/phase8_qsafe_ft_monitor/${EXP_PREFIX}_FINAL_AUDIT.json"

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

for path in "$SOURCE" "$SOURCE_EMA" "$TRAIN_TSV" "$GT_CACHE" "$MUSICCAPS" \
    "$NPZ_DIR/MANIFEST.tsv" "$NPZ_DIR/FULL_VALIDATION.json" \
    "$NPZ_DIR/FULL_GATE_PASSED.json" "$EVAL_SCRIPT"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

python - "$SOURCE" "$TRAIN_TSV" "$NPZ_DIR" "$GT_CACHE" "$EXPECTED_ROWS" <<'PY'
import csv, json, sys
from collections import Counter
from pathlib import Path
import torch

source, tsv, npz, cache = map(Path, sys.argv[1:5])
expected = int(sys.argv[5])
state = torch.load(source, map_location="cpu", weights_only=False)
if state.get("it") != 600000:
    raise SystemExit(f"[FAIL] baseline checkpoint it={state.get('it')}")
rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
names = [line.strip() for line in cache.open() if line.strip()]
if len(rows) != expected or len(names) != expected:
    raise SystemExit(f"[FAIL] row count tsv={len(rows)} cache={len(names)}")
q = [int(row["q_level"]) for row in rows]
if min(q) < 0 or max(q) > 9 or len(set(q)) < 5:
    raise SystemExit(f"[FAIL] invalid Q support {Counter(q)}")
for gate in (npz / "FULL_VALIDATION.json", npz / "FULL_GATE_PASSED.json"):
    if json.loads(gate.read_text()).get("status") != "passed":
        raise SystemExit(f"[FAIL] cache gate not passed: {gate}")
print(f"[OK] baseline it=600000; rows={len(rows)}; Q={dict(sorted(Counter(q).items()))}")
PY

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$EXP_DIR" "$CONTRACT" "$INIT_MANIFEST" \
        "$ROOT/eval_output/${EXP_ID}_musiccaps_q9" \
        "$ROOT/eval_output/${EXP_ID}_musiccaps_q6" \
        "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q9" \
        "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q6" \
        "$LOG_ROOT/${EXP_ID}_musiccaps_q9_eval.log" \
        "$LOG_ROOT/${EXP_ID}_musiccaps_q6_eval.log"; do
        if [ -e "$path" ]; then conflicts+=("$path"); fi
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        printf '[FAIL] fresh artifact exists: %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi

mkdir -p "$EXP_DIR" "$(dirname "$FINAL_AUDIT")"
SOURCE_SHA=$(sha256sum "$SOURCE" | awk '{print $1}')
if [ ! -f "$CKPT" ]; then
    python scripts/init_qsafe_s2_checkpoint.py \
        --source "$SOURCE" --output "$CKPT" --manifest "$INIT_MANIFEST" \
        --expected-it "$SOURCE_IT" --source-sha256 "$SOURCE_SHA"
fi

python - "$CONTRACT" "$EXP_PREFIX" "$Q_MODE" "$TRAIN_TSV" "$SOURCE" \
    "$SOURCE_EMA" "$SOURCE_SHA" "$INIT_MANIFEST" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

out, prefix, q_mode, train_tsv, source, source_ema, source_sha, init_manifest = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()
critical = [
    "scripts/init_qsafe_s2_checkpoint.py",
    "scripts/training_pipelines/train_pipeline_phase8_qsafe_ft.sh",
    "scripts/audit_phase8_qsafe_ft.py",
    "meanaudio/runner_meanflow.py", "meanaudio/model/mean_flow.py",
    "meanaudio/model/networks.py", "eval.py",
]
payload = {
    "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
    "prefix": prefix, "q_mode": q_mode,
    "source_checkpoint": source, "source_checkpoint_sha256": source_sha,
    "source_ema": source_ema, "source_iteration": 600000,
    "initialization": "copy_q10_exactly_to_q0_through_q9",
    "init_manifest": init_manifest,
    "train_tsv": train_tsv, "train_tsv_sha256": sha(train_tsv),
    "use_q_conditioning": True, "use_text_attention_mask": False,
    "multi_cap": False, "fine_tune_iterations": 100000,
    "final_iteration": 700000, "learning_rate": 3e-5,
    "batch_size": 8, "accumulation_steps": 1, "seed": 14159265,
    "eval_tsv": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
    "eval_primary_q": 9, "eval_secondary_q": 6,
    "baseline_clap": 0.1888, "restoration_target_clap": 0.1900,
    "critical_file_sha256": {rel: sha(root/rel) for rel in critical},
}
p=Path(out)
if p.exists():
    old=json.loads(p.read_text()); drift=[k for k,v in payload.items() if k!="created_at" and old.get(k)!=v]
    if drift: raise SystemExit(f"[FAIL] immutable contract drift: {drift}")
else:
    tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,p)
print(f"[OK] contract {p}")
PY

python set_training_stage.py --stage 2
if [ ! -f "$EMA" ]; then
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
        2>&1 | tee "$LOG"
fi

for q in 9 6; do
    out="$ROOT/eval_output/${EXP_ID}_musiccaps_q$q"
    eval_log="$LOG_ROOT/${EXP_ID}_musiccaps_q${q}_eval.log"
    python eval.py --variant meanaudio_s --model_path "$EMA" \
        --output "$out/audio" --tsv "$MUSICCAPS" --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
        --quality_level "$q" --no_text_attention_mask --full_precision \
        2>&1 | tee "$eval_log"
    python "$EVAL_SCRIPT" --gen_dir "$out/audio" --tsv "$MUSICCAPS" \
        --exp_name "${EXP_ID}_musiccaps_q$q" --num_samples 2048 \
        2>&1 | tee -a "$eval_log"
done

python scripts/audit_phase8_qsafe_ft.py --prefix "$EXP_PREFIX" \
    --q-mode "$Q_MODE" --phase final --json-out "$FINAL_AUDIT"
echo "[COMPLETE] $EXP_PREFIX"
