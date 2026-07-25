#!/usr/bin/env bash
# Official-aligned Qwen No-Q quarter adapter.
#
# This is intentionally separate from the legacy/LP No-Q pipeline.  It uses
# the exact K=2 balanced Qwen TSV, row order, NPZ cache, and cache list while
# disabling Q conditioning in both stages.  Invoke long runs through
# scripts/run_with_experiment_report.sh.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
GT_CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
TRAIN_TSV="$DATA/phase8_qwen_meansim_k2_balanced.tsv"
GRID_MANIFEST="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
QWEN_NPZ_MANIFEST="$DATA/phase8_qwen_official_matched_npz_manifest.json"
QWEN_CACHE_AUDIT="$DATA/phase8_qwen_official_matched_qwen_cache_audit.json"
MUSICCAPS="$DATA/musiccaps_test.tsv"
HOLDOUT="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_holdout_n5009.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py

PREFIX=phase8_qwen_bucket_quarter_noq
S1_UPDATES=100000
S2_UPDATES=50000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
EXPECTED_ROWS=251599
SEED=14159265
LR=1e-4
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"

case "$RUN_MODE" in fresh|resume) ;; *)
    echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2; exit 2 ;;
esac
for value_name in PREFLIGHT_ONLY DRY_RUN; do
    case "${!value_name}" in true|false) ;; *)
        echo "[FAIL] $value_name must be true or false" >&2; exit 2 ;;
    esac
done

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
REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] experiment=$PREFIX mode=$RUN_MODE"
    echo "[DRY RUN] data=$TRAIN_TSV rows=$EXPECTED_ROWS q_conditioning=false"
    echo "[DRY RUN] S1=$EXP_S1 updates=$S1_UPDATES"
    echo "[DRY RUN] S2=$EXP_S2 additional_updates=$S2_UPDATES final_it=$FINAL_IT"
    echo "[DRY RUN] report=$REPORT"
    exit 0
fi

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" "$NPZ_DIR" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" "$MUSICCAPS" "$HOLDOUT" \
    "$EVALUATOR" "$ROOT/migrate_stage1_to_stage2_ckpt.py" \
    "$ROOT/set_training_stage.py"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

# Bind No-Q to the exact official-aligned K=2 balanced arm.  q_level remains
# present in the TSV but is ignored by the model in both stages.
python - "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" "$NPZ_DIR" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" "$EXPECTED_ROWS" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

tsv, grid_path, cache_path, npz_dir, npz_manifest_path, audit_path = map(
    Path, sys.argv[1:7]
)
expected = int(sys.argv[7])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

grid = json.loads(grid_path.read_text())
arm = grid.get("outputs", {}).get("k2_balanced", {})
if grid.get("status") != "passed" or arm.get("sha256") != sha(tsv):
    raise SystemExit("[FAIL] No-Q TSV is not the passed K=2 balanced grid output")
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
names = [line.strip() for line in cache_path.open() if line.strip()]
if not (len(rows) == len(names) == expected):
    raise SystemExit(
        f"[FAIL] cardinality rows={len(rows)} cache={len(names)} "
        f"expected={expected}"
    )
if len({row["id"] for row in rows}) != expected or len(set(names)) != expected:
    raise SystemExit("[FAIL] duplicate TSV id or cache filename")
hist = Counter(int(row["q_level"]) for row in rows)
expected_hist = Counter({int(k): int(v) for k, v in arm["q_histogram"].items()})
if hist != expected_hist:
    raise SystemExit(f"[FAIL] K=2 balanced histogram drift: {hist}")

npz_manifest = json.loads(npz_manifest_path.read_text())
audit = json.loads(audit_path.read_text())
if (
    npz_manifest.get("status") != "passed"
    or npz_manifest.get("completed_rows") != expected
    or npz_manifest.get("output_dir") != str(npz_dir)
    or npz_manifest.get("cache_list") != str(cache_path)
    or npz_manifest.get("cache_list_sha256") != sha(cache_path)
    or audit.get("status") != "passed"
    or audit.get("rows") != expected
    or audit.get("cache_names") != expected
    or audit.get("npz_dir") != str(npz_dir)
    or audit.get("cache_list_sha256") != sha(cache_path)
    or audit.get("semantic_gate", {}).get("status") != "passed"
):
    raise SystemExit("[FAIL] official Qwen NPZ manifest/cache audit is not passed/full")

indices = sorted({round(i * (expected - 1) / 31) for i in range(32)})
for index in indices:
    row, name = rows[index], names[index]
    with np.load(npz_dir / name) as data:
        if str(data["clip_id"].item()) != row["id"]:
            raise SystemExit(f"[FAIL] embedded clip id mismatch at {index}")
        caption_hash = hashlib.sha256(row["caption"].encode("utf-8")).hexdigest()
        if str(data["caption_sha256"].item()) != caption_hash:
            raise SystemExit(f"[FAIL] embedded Qwen caption mismatch at {index}")
print(
    f"[OK] official Qwen No-Q adapter rows={expected:,}; "
    "TSV/cache/NPZ exactly match K=2 balanced"
)
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] $PREFIX; no checkpoint or GPU process started."
    exit 0
fi

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$S1_DIR" "$S2_DIR" "$S1_LOG" "$S2_LOG" "$MIGRATE_LOG" \
        "$CONTRACT" "$FINAL_AUDIT" "$REPORT"; do
        [ -e "$path" ] && conflicts+=("$path")
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        printf '[FAIL] fresh artifact exists: %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi
mkdir -p "$S1_DIR" "$S2_DIR" "$LOG_ROOT"

python - "$CONTRACT" "$TRAIN_TSV" "$GRID_MANIFEST" "$GT_CACHE" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out, tsv, grid, cache, npz_manifest, cache_audit = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_qwen_bucket_quarter_noq",
    "scale": "quarter",
    "arm": "noq",
    "matched_bucket_arm": "k2_balanced",
    "train_tsv": tsv,
    "train_tsv_sha256": sha(tsv),
    "grid_manifest": grid,
    "grid_manifest_sha256": sha(grid),
    "qwen_cache_list": cache,
    "qwen_cache_list_sha256": sha(cache),
    "qwen_npz_manifest": npz_manifest,
    "qwen_npz_manifest_sha256": sha(npz_manifest),
    "qwen_cache_audit": cache_audit,
    "qwen_cache_audit_sha256": sha(cache_audit),
    "expected_rows": 251599,
    "stage1_updates": 100000,
    "stage2_updates": 50000,
    "stage2_final_iteration": 150000,
    "stage1_use_q_conditioning": False,
    "stage2_use_q_conditioning": False,
    "q_semantics": "K=2 balanced q_level retained in TSV but ignored",
    "caption_semantics": "official Qwen caption mapped to exact Jamendo track",
    "seed": 14159265,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "accumulation_steps": 1,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "critical_file_sha256": {
        rel: sha(root / rel)
        for rel in (
            "scripts/training_pipelines/execute_phase8_qwen_noq_arm_eval.sh",
            "migrate_stage1_to_stage2_ckpt.py",
            "meanaudio/runner_flowmatching.py",
            "meanaudio/runner_meanflow.py",
            "meanaudio/model/networks.py",
        )
    },
}
try:
    payload["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
except Exception:
    payload["git_head"] = None
path = Path(out)
if path.exists():
    previous = json.loads(path.read_text())
    drift = [
        key for key, value in payload.items()
        if key not in {"created_at", "git_head"} and previous.get(key) != value
    ]
    if drift:
        raise SystemExit(f"[FAIL] immutable No-Q contract drift: {drift}")
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
print(-1 if not path.is_file() else int(torch.load(
    path, map_location="cpu", weights_only=False
)["it"]))
PY
}

s1_it=$(checkpoint_it "$S1_CKPT")
if [ "$s1_it" -lt "$S1_UPDATES" ]; then
    python set_training_stage.py --stage 1
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=fluxaudio_s exp_id="$EXP_S1" \
        num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
        +use_q_conditioning=false batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=25000 save_checkpoint_interval=25000 \
        ++ema.checkpoint_every=25000 +use_rope=False +use_wandb=False \
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
        --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT" \
        2>&1 | tee "$MIGRATE_LOG"
fi
s2_it=$(checkpoint_it "$S2_CKPT")
if [ "$s2_it" -lt "$FINAL_IT" ]; then
    python set_training_stage.py --stage 2
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=meanaudio_s exp_id="$EXP_S2" \
        num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
        +use_q_conditioning=false batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=25000 save_checkpoint_interval=25000 \
        ++ema.checkpoint_every=25000 +use_rope=False +use_wandb=False \
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
    "$S1_DIR" "$S2_DIR" "$TRAIN_TSV" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

s1_path, s2_path, contract, out, s1_dir, s2_dir, train_tsv = sys.argv[1:]
s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
s2 = torch.load(s2_path, map_location="cpu", weights_only=False)
issues = []
if s1.get("it") != 100000:
    issues.append(f"S1 iteration={s1.get('it')}")
if s2.get("it") != 150000:
    issues.append(f"S2 iteration={s2.get('it')}")
configs = {}
for label, directory, model, iterations in (
    ("S1", Path(s1_dir), "fluxaudio_s", 100000),
    ("S2", Path(s2_dir), "meanaudio_s", 150000),
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
        "use_q_conditioning": False,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "data.AudioCaps_npz.tsv": train_tsv,
        "data.AudioCaps_npz.npz_dir":
            "/mnt/HDD/kojiek/phase8_qwen_official_matched_npz",
    }
    for key, expected in checks.items():
        actual = OmegaConf.select(cfg, key)
        if actual != expected:
            issues.append(f"{label} config {key}={actual!r}, expected={expected!r}")
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "failed" if issues else "passed",
    "issues": issues,
    "scale": "quarter",
    "arm": "noq",
    "matched_bucket_arm": "k2_balanced",
    "stage1_iteration": s1.get("it"),
    "stage2_iteration": s2.get("it"),
    "stage1_use_q_conditioning": False,
    "stage2_use_q_conditioning": False,
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

run_eval() {
    local label="$1" variant="$2" model="$3"
    local out="$ROOT/eval_output/$label"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local eval_log="$LOG_ROOT/${label}_eval.log"
    local provenance="$out/provenance.json"
    local protocol=()
    local protocol_name
    if [ "$variant" = fluxaudio_s ]; then
        protocol=(--num_steps 25 --cfg_strength 4.5)
        protocol_name=FM25_CFG4.5_NOQ
    else
        protocol=(--use_meanflow --num_steps 1 --cfg_strength 0.5)
        protocol_name=MF1_CFG0.5_NOQ
    fi
    mkdir -p "$out/audio"
    if [ -f "$metrics" ]; then
        audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
        [ "$audio_n" -eq 5521 ] || {
            echo "[FAIL] stale metrics audio count $audio_n/5521 for $label" >&2
            exit 2
        }
        grep -F "Test TSV: $MUSICCAPS" "$metrics" >/dev/null
        grep -F "Generated audio: $out/audio" "$metrics" >/dev/null
        grep -F "Test clips: 5521" "$metrics" >/dev/null
        [ -f "$provenance" ] || {
            echo "[FAIL] metrics exist without provenance: $provenance" >&2
            exit 2
        }
        python - "$provenance" "$label" "$model" "$MUSICCAPS" \
            "$protocol_name" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
provenance, label, model, prompts, protocol = sys.argv[1:]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected = {
    "schema_version": 1, "label": label, "model": model,
    "model_sha256": sha(model), "prompts": prompts,
    "prompts_sha256": sha(prompts), "rows": 5521,
    "q_mode": "no_q", "protocol": protocol,
}
if json.loads(Path(provenance).read_text()) != expected:
    raise SystemExit("[FAIL] No-Q evaluation provenance drift")
PY
        echo "[SKIP] verified complete metrics: $metrics"
        return
    fi
    python eval.py --variant "$variant" --model_path "$model" \
        --output "$out/audio" --tsv "$MUSICCAPS" \
        --encoder_name t5_clap --text_c_dim 512 --no_text_attention_mask \
        --full_precision "${protocol[@]}" --no_q 2>&1 | tee "$eval_log"
    audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
    [ "$audio_n" -eq 5521 ] || {
        echo "[FAIL] $label generated $audio_n/5521 audio files" >&2; exit 2
    }
    python - "$provenance" "$label" "$model" "$MUSICCAPS" \
        "$protocol_name" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
provenance, label, model, prompts, protocol = sys.argv[1:]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": 1, "label": label, "model": model,
    "model_sha256": sha(model), "prompts": prompts,
    "prompts_sha256": sha(prompts), "rows": 5521,
    "q_mode": "no_q", "protocol": protocol,
}
path = Path(provenance)
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
    python "$EVALUATOR" --gen_dir "$out/audio" --tsv "$MUSICCAPS" \
        --exp_name "$label" --num_samples 5521 2>&1 | tee -a "$eval_log"
    [ -f "$metrics" ] || { echo "[FAIL] no metrics for $label" >&2; exit 2; }
}

S1_LABEL="${EXP_S1}_musiccaps_n5521_fm25_noq"
S2_LABEL="${EXP_S2}_musiccaps_n5521_mf1_noq"
run_eval "$S1_LABEL" fluxaudio_s "$S1_EMA"
run_eval "$S2_LABEL" meanaudio_s "$S2_EMA"

score_holdout() {
    local source_label="$1"
    local label="${source_label}_holdout5009"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local eval_log="$LOG_ROOT/${label}_eval.log"
    if [ ! -f "$metrics" ]; then
        python "$EVALUATOR" \
            --gen_dir "$ROOT/eval_output/$source_label/audio" \
            --tsv "$HOLDOUT" --exp_name "$label" --num_samples 5009 \
            2>&1 | tee "$eval_log"
    fi
    [ -f "$metrics" ] || { echo "[FAIL] no holdout metrics for $label" >&2; exit 2; }
    grep -F "Test TSV: $HOLDOUT" "$metrics" >/dev/null
    grep -F "Test clips: 5009" "$metrics" >/dev/null
    echo "$label"
}
S1_HOLDOUT=$(score_holdout "$S1_LABEL" | tail -1)
S2_HOLDOUT=$(score_holdout "$S2_LABEL" | tail -1)

python - "$REPORT" "$TRAIN_TSV" "$GRID_MANIFEST" "$MUSICCAPS" \
    "$FINAL_AUDIT" "$CONTRACT" "$S1_EMA" "$S2_EMA" \
    "$S1_LABEL" "$S2_LABEL" "$S1_HOLDOUT" "$S2_HOLDOUT" <<'PY'
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out, train_tsv, grid_path, prompts, audit, contract,
    s1_model, s2_model, s1_label, s2_label, s1_holdout, s2_holdout,
) = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read(label):
    path = root / "eval_output" / "metrics" / label / "metrics.txt"
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                values[key] = float(value)
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if set(values) != required or not all(math.isfinite(v) for v in values.values()):
        raise SystemExit(f"[FAIL] incomplete/nonfinite metrics: {path}: {values}")
    return {"label": label, **values}

grid = json.loads(Path(grid_path).read_text())
if grid["outputs"]["k2_balanced"]["sha256"] != sha(train_tsv):
    raise SystemExit("[FAIL] report TSV no longer matches K=2 balanced")
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "experiment": "phase8_qwen_bucket_quarter_noq",
    "scale": "quarter",
    "arm": "noq",
    "matched_bucket_arm": "k2_balanced",
    "q_conditioning": False,
    "train_rows": 251599,
    "train_tsv": {"path": train_tsv, "sha256": sha(train_tsv)},
    "prompts": {"path": prompts, "sha256": sha(prompts), "rows": 5521},
    "training_audit": audit,
    "training_contract": {"path": contract, "sha256": sha(contract)},
    "models": {
        "stage1": {"path": s1_model, "sha256": sha(s1_model)},
        "global": {"path": s2_model, "sha256": sha(s2_model)},
    },
    "stage1": {
        "protocol": "MusicCaps 5521; FluxAudio FM25 CFG4.5; no_q",
        "no_q": read(s1_label),
        "holdout5009_no_q": read(s1_holdout),
    },
    "global": {
        "protocol": "MusicCaps 5521; MeanFlow1 CFG0.5; no_q",
        "no_q": read(s2_label),
        "holdout5009_no_q": read(s2_holdout),
    },
}
path = Path(out)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[COMPLETE] $PREFIX S1=100k S2=50k report=$REPORT"
