#!/usr/bin/env bash
# Train one aligned Qwen Q-bucket arm. Long runs must be invoked through
# scripts/run_with_experiment_report.sh by the sequence driver.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
GT_CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
GRID_MANIFEST="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
QWEN_NPZ_MANIFEST="$DATA/phase8_qwen_official_matched_npz_manifest.json"
QWEN_CACHE_AUDIT="$DATA/phase8_qwen_official_matched_qwen_cache_audit.json"
K="${K:?K must be 2, 3, 5, or 10}"
STRATEGY="${STRATEGY:?STRATEGY must be balanced or fixed}"
SCALE="${SCALE:?SCALE must be pilot or quarter}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-true}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
SEED=14159265
LR=1e-4

case "$K" in 2|3|5|10) ;; *) echo "[FAIL] invalid K=$K" >&2; exit 2;; esac
case "$STRATEGY" in balanced|fixed) ;; *)
    echo "[FAIL] invalid STRATEGY=$STRATEGY" >&2; exit 2;;
esac
case "$SCALE" in
    pilot) S1_UPDATES=25000; S2_UPDATES=12500 ;;
    quarter) S1_UPDATES=100000; S2_UPDATES=50000 ;;
    *) echo "[FAIL] invalid SCALE=$SCALE" >&2; exit 2;;
esac
case "$PREFLIGHT_ONLY" in true|false) ;; *)
    echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2; exit 2;;
esac
case "$RUN_MODE" in fresh|resume) ;; *)
    echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2; exit 2;;
esac

FINAL_IT=$((S1_UPDATES + S2_UPDATES))
TRAIN_TSV="$DATA/phase8_qwen_meansim_k${K}_${STRATEGY}.tsv"
PREFIX="phase8_qwen_bucket_${SCALE}_k${K}_${STRATEGY}"
EXP_S1="${PREFIX}_stage1_${S1_UPDATES}"
EXP_S2="${PREFIX}_stage2_${S2_UPDATES}"
S1_DIR="$ROOT/exps/$EXP_S1"
S2_DIR="$ROOT/exps/$EXP_S2"
S1_CKPT="$S1_DIR/${EXP_S1}_ckpt_last.pth"
S1_EMA="$S1_DIR/${EXP_S1}_ema_final.pth"
S2_CKPT="$S2_DIR/${EXP_S2}_ckpt_last.pth"
S2_EMA="$S2_DIR/${EXP_S2}_ema_final.pth"
S1_LOG="$LOG_ROOT/${EXP_S1}.log"
S2_LOG="$LOG_ROOT/${EXP_S2}.log"
MIGRATE_LOG="$LOG_ROOT/${EXP_S2}_migrate.log"
CONTRACT="$LOG_ROOT/${PREFIX}_contract.json"
FINAL_AUDIT="$LOG_ROOT/${PREFIX}_FINAL_TRAIN_AUDIT.json"

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" "$NPZ_DIR" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" \
    "$ROOT/migrate_stage1_to_stage2_ckpt.py" "$ROOT/set_training_stage.py"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

python - "$GRID_MANIFEST" "$TRAIN_TSV" "$K" "$STRATEGY" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path, tsv_path = map(Path, sys.argv[1:3])
k, strategy = int(sys.argv[3]), sys.argv[4]
payload = json.loads(manifest_path.read_text())
arm = payload.get("outputs", {}).get(f"k{k}_{strategy}", {})
digest = hashlib.sha256(tsv_path.read_bytes()).hexdigest()
if payload.get("status") != "passed" or arm.get("sha256") != digest:
    raise SystemExit("[FAIL] TSV is not bound to passed grid manifest")
with tsv_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
hist = Counter(int(row["q_level"]) for row in rows)
if len(rows) != 251599 or len({row["id"] for row in rows}) != len(rows):
    raise SystemExit("[FAIL] TSV cardinality/uniqueness mismatch")
expected = {int(q): n for q, n in arm["q_histogram"].items()}
if hist != Counter(expected):
    raise SystemExit(f"[FAIL] histogram drift: {hist}")
if arm.get("high_q") != 9:
    raise SystemExit("[FAIL] comparable high endpoint must be q9")
print(f"[OK] k={k} strategy={strategy} rows={len(rows):,} histogram={dict(hist)}")
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] $PREFIX; no checkpoint or GPU process started."
    exit 0
fi

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$S1_DIR" "$S2_DIR" "$S1_LOG" "$S2_LOG" "$MIGRATE_LOG" \
        "$CONTRACT" "$FINAL_AUDIT"; do
        [ -e "$path" ] && conflicts+=("$path")
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        printf '[FAIL] fresh artifact exists: %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi
mkdir -p "$S1_DIR" "$S2_DIR" "$LOG_ROOT"

python - "$CONTRACT" "$PREFIX" "$SCALE" "$K" "$STRATEGY" "$TRAIN_TSV" \
    "$GRID_MANIFEST" "$S1_UPDATES" "$S2_UPDATES" "$FINAL_IT" "$GT_CACHE" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out, prefix, scale, k, strategy, tsv, grid, s1, s2, final_it,
    gt_cache, npz_manifest, cache_audit,
) = sys.argv[1:]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": prefix,
    "scale": scale,
    "k": int(k),
    "strategy": strategy,
    "train_tsv": tsv,
    "train_tsv_sha256": sha(tsv),
    "grid_manifest": grid,
    "grid_manifest_sha256": sha(grid),
    "qwen_cache_list": gt_cache,
    "qwen_cache_list_sha256": sha(gt_cache),
    "qwen_npz_manifest": npz_manifest,
    "qwen_npz_manifest_sha256": sha(npz_manifest),
    "qwen_cache_audit": cache_audit,
    "qwen_cache_audit_sha256": sha(cache_audit),
    "stage1_updates": int(s1),
    "stage2_updates": int(s2),
    "stage2_final_iteration": int(final_it),
    "q_conditioning": True,
    "q_initialization": "preserve",
    "caption_semantics": "official Qwen caption mapped to exact Jamendo track",
    "quality_semantics": "actual-clip MeanSimilarity",
    "seed": 14159265,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "save_only_at_final_iteration": True,
    "critical_file_sha256": {
        relative: sha(Path("/home/kojiek/MeanAudio") / relative)
        for relative in (
            "scripts/preprocess/make_phase8_qwen_bucket_grid.py",
            "scripts/training_pipelines/train_phase8_qwen_bucket_arm.sh",
            "scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh",
            "scripts/training_pipelines/sequence_phase8_qwen_bucket_grid.sh",
            "migrate_stage1_to_stage2_ckpt.py",
            "meanaudio/runner_flowmatching.py",
            "meanaudio/runner_meanflow.py",
            "meanaudio/model/networks.py",
        )
    },
}
try:
    payload["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/home/kojiek/MeanAudio", text=True
    ).strip()
except Exception:
    payload["git_head"] = None
path = Path(out)
if path.exists():
    old = json.loads(path.read_text())
    drift = [
        key for key, value in payload.items()
        if key not in {"created_at", "git_head"} and old.get(key) != value
    ]
    if drift:
        raise SystemExit(f"[FAIL] immutable contract drift: {drift}")
else:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
print(f"[OK] contract: {path}")
PY

checkpoint_it() {
    python - "$1" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
print(-1 if not path.is_file() else int(torch.load(path, map_location="cpu", weights_only=False)["it"]))
PY
}

s1_it=$(checkpoint_it "$S1_CKPT")
if [ "$s1_it" -lt "$S1_UPDATES" ]; then
    python set_training_stage.py --stage 1
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=fluxaudio_s exp_id="$EXP_S1" \
        num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
        +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval="$S1_UPDATES" save_checkpoint_interval="$S1_UPDATES" \
        ++ema.checkpoint_every="$S1_UPDATES" +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
        save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TRAIN_TSV" \
        "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$TRAIN_TSV" \
        "data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
        "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
        "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
        2>&1 | tee "$S1_LOG"
elif [ "$s1_it" -ne "$S1_UPDATES" ]; then
    echo "[FAIL] unexpected S1 checkpoint iteration=$s1_it" >&2
    exit 2
else
    echo "[SKIP] complete S1 checkpoint: $S1_CKPT"
fi
[ -f "$S1_EMA" ] || { echo "[FAIL] missing S1 EMA: $S1_EMA" >&2; exit 2; }

if [ ! -f "$S2_CKPT" ]; then
    python set_training_stage.py --stage 2
    python migrate_stage1_to_stage2_ckpt.py \
        --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT" --q-init preserve \
        2>&1 | tee "$MIGRATE_LOG"
fi
s2_it=$(checkpoint_it "$S2_CKPT")
if [ "$s2_it" -lt "$FINAL_IT" ]; then
    python set_training_stage.py --stage 2
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=meanaudio_s exp_id="$EXP_S2" \
        num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
        +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval="$FINAL_IT" save_checkpoint_interval="$FINAL_IT" \
        ++ema.checkpoint_every="$FINAL_IT" +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
        save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TRAIN_TSV" \
        "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$TRAIN_TSV" \
        "data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
        "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
        "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
        2>&1 | tee "$S2_LOG"
elif [ "$s2_it" -ne "$FINAL_IT" ]; then
    echo "[FAIL] unexpected S2 checkpoint iteration=$s2_it" >&2
    exit 2
else
    echo "[SKIP] complete S2 checkpoint: $S2_CKPT"
fi
[ -f "$S2_EMA" ] || { echo "[FAIL] missing S2 EMA: $S2_EMA" >&2; exit 2; }

python - "$S1_CKPT" "$S2_CKPT" "$CONTRACT" "$FINAL_AUDIT" \
    "$S1_UPDATES" "$FINAL_IT" "$K" "$STRATEGY" "$SCALE" \
    "$S1_DIR" "$S2_DIR" "$TRAIN_TSV" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf

s1_path, s2_path, contract, out = sys.argv[1:5]
s1_expected, s2_expected, k = map(int, sys.argv[5:8])
strategy, scale = sys.argv[8:10]
s1_dir, s2_dir, train_tsv = sys.argv[10:13]
s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
s2 = torch.load(s2_path, map_location="cpu", weights_only=False)
issues = []
if s1.get("it") != s1_expected:
    issues.append(f"S1 iteration={s1.get('it')}")
if s2.get("it") != s2_expected:
    issues.append(f"S2 iteration={s2.get('it')}")
for label, state in (("S1", s1), ("S2", s2)):
    if "q_embed.weight" not in state.get("weights", {}):
        issues.append(f"{label} lacks q_embed.weight")
configs = {}
for label, directory, model, iterations in (
    ("S1", Path(s1_dir), "fluxaudio_s", s1_expected),
    ("S2", Path(s2_dir), "meanaudio_s", s2_expected),
):
    candidates = sorted(
        directory.glob("train-*-hydra/config.yaml"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        issues.append(f"{label} Hydra config missing")
        continue
    config_path = candidates[-1]
    cfg = OmegaConf.load(config_path)
    configs[label] = str(config_path)
    checks = {
        "model": model,
        "num_iterations": iterations,
        "seed": 14159265,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "accumulation_steps": 1,
        "use_q_conditioning": True,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "data.AudioCaps_npz.tsv": train_tsv,
        "data.AudioCaps_npz.npz_dir":
            "/mnt/HDD/kojiek/phase8_qwen_official_matched_npz",
    }
    for key, expected in checks.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            issues.append(
                f"{label} config {key}={actual!r}, expected={expected!r}"
            )
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "failed" if issues else "passed",
    "issues": issues,
    "scale": scale,
    "k": k,
    "strategy": strategy,
    "stage1_iteration": s1.get("it"),
    "stage2_iteration": s2.get("it"),
    "q_conditioning": True,
    "contract": contract,
    "hydra_configs": configs,
}
path = Path(out)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
print(json.dumps(payload, indent=2, sort_keys=True))
if issues:
    raise SystemExit(2)
PY

echo "[COMPLETE] $PREFIX S1=$S1_UPDATES S2=$S2_UPDATES"
