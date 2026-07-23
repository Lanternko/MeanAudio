#!/usr/bin/env bash
# One Qwen-caption, Q-conditioned, end-to-end quarter-scale arm.
# S1=100k and S2=50k; Full-Q and Half-Q differ only in q_level.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
GT_CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
QWEN_TSV="$DATA/phase8_qwen_official_matched.tsv"
QWEN_MAPPER_MANIFEST="$DATA/phase8_qwen_official_matched_manifest.json"
QWEN_NPZ_MANIFEST="$DATA/phase8_qwen_official_matched_npz_manifest.json"
QWEN_CACHE_AUDIT="$DATA/phase8_qwen_official_matched_qwen_cache_audit.json"
OFFICIAL_JSON=/home/kojiek/reference-repos/ICME26-ATTM-GC-FluxAudio/data/captions/jamendo_qwen.json
SOURCE_JSONL=/home/kojiek/research/music_cleaning/results_20260119_043407.jsonl
ALIGNED_TSV="$DATA/phase8_legacy_catalog_train_meansim_aligned.tsv"
ALIGNED_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_aligned.manifest.json"
LEGACY_HALFQ_TSV="$DATA/phase8_legacy_catalog_train_meansim_halfq.tsv"
LEGACY_HALFQ_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_halfq.manifest.json"
FULLQ_TSV="$DATA/phase8_qwen_meansim_fullq.tsv"
HALFQ_TSV="$DATA/phase8_qwen_meansim_halfq.tsv"
Q_MANIFEST="$DATA/phase8_qwen_meansim_fullq_halfq.manifest.json"

S1_UPDATES=100000
S2_UPDATES=50000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
EXPECTED_ROWS=251599
SEED=14159265
LR=1e-4
ARM="${ARM:?ARM must be fullq or halfq}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-true}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"

case "$ARM" in
    fullq)
        PREFIX=phase8_qwen_quarter_e2e_fullq
        TRAIN_TSV="$FULLQ_TSV"
        ;;
    halfq)
        PREFIX=phase8_qwen_quarter_e2e_halfq
        TRAIN_TSV="$HALFQ_TSV"
        ;;
    *)
        echo "[FAIL] ARM=$ARM; expected fullq or halfq" >&2
        exit 2
        ;;
esac
case "$PREFLIGHT_ONLY" in true|false) ;; *)
    echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2
    exit 2
esac
case "$RUN_MODE" in fresh|resume) ;; *)
    echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2
    exit 2
esac

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

for path in "$QWEN_TSV" "$QWEN_MAPPER_MANIFEST" "$QWEN_NPZ_MANIFEST" \
    "$QWEN_CACHE_AUDIT" "$GT_CACHE" "$OFFICIAL_JSON" "$SOURCE_JSONL" \
    "$ALIGNED_TSV" "$ALIGNED_MANIFEST" "$LEGACY_HALFQ_TSV" \
    "$LEGACY_HALFQ_MANIFEST" "$ROOT/migrate_stage1_to_stage2_ckpt.py" \
    "$ROOT/set_training_stage.py"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

# Recompute actual-clip Full-Q and independently reverify Half-Q before joining
# either assignment to Qwen captions.
audit_dir=$(mktemp -d /tmp/phase8-qwen-fullq-halfq.XXXXXX)
trap 'rm -rf "$audit_dir"' EXIT
python scripts/preprocess/align_meansim_q_to_catalog.py \
    --input "$ALIGNED_TSV" --source-jsonl "$SOURCE_JSONL" \
    --manifest "$audit_dir/aligned-audit.json" --require-current-match
python scripts/preprocess/make_phase8_halfq_tsv.py \
    --input "$ALIGNED_TSV" --aligned-manifest "$ALIGNED_MANIFEST" \
    --source-jsonl "$SOURCE_JSONL" --output "$LEGACY_HALFQ_TSV" \
    --manifest "$LEGACY_HALFQ_MANIFEST" --expected-rows "$EXPECTED_ROWS"
python scripts/preprocess/make_phase8_qwen_meansim_tsvs.py \
    --qwen-tsv "$QWEN_TSV" \
    --qwen-mapper-manifest "$QWEN_MAPPER_MANIFEST" \
    --qwen-npz-manifest "$QWEN_NPZ_MANIFEST" \
    --qwen-cache-audit "$QWEN_CACHE_AUDIT" \
    --qwen-cache-list "$GT_CACHE" --official-json "$OFFICIAL_JSON" \
    --aligned-tsv "$ALIGNED_TSV" --aligned-manifest "$ALIGNED_MANIFEST" \
    --halfq-tsv "$LEGACY_HALFQ_TSV" \
    --halfq-manifest "$LEGACY_HALFQ_MANIFEST" \
    --fullq-output "$FULLQ_TSV" --halfq-output "$HALFQ_TSV" \
    --manifest "$Q_MANIFEST" --expected-rows "$EXPECTED_ROWS"

python - "$FULLQ_TSV" "$HALFQ_TSV" "$Q_MANIFEST" "$GT_CACHE" \
    "$NPZ_DIR" "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" \
    "$EXPECTED_ROWS" "$ARM" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

full_path, half_path, manifest_path, cache_path, npz_dir, npz_manifest_path, audit_path = map(
    Path, sys.argv[1:8]
)
expected = int(sys.argv[8])
arm = sys.argv[9]

def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

full = rows(full_path)
half = rows(half_path)
names = [line.strip() for line in cache_path.open(encoding="utf-8") if line.strip()]
if not (len(full) == len(half) == len(names) == expected):
    raise SystemExit(
        f"[FAIL] cardinality full={len(full)} half={len(half)} "
        f"cache={len(names)} expected={expected}"
    )
if len({row["id"] for row in full}) != expected or len(set(names)) != expected:
    raise SystemExit("[FAIL] duplicate TSV id or cache name")
for index, (left, right) in enumerate(zip(full, half)):
    if left["id"] != right["id"]:
        raise SystemExit(f"[FAIL] arm row-order mismatch at {index}")
    for key in left:
        if key != "q_level" and left[key] != right[key]:
            raise SystemExit(f"[FAIL] arm drift row={index} field={key}")
full_hist = Counter(int(row["q_level"]) for row in full)
half_hist = Counter(int(row["q_level"]) for row in half)
if full_hist != Counter({3: 1, 4: 306, 5: 11056, 6: 49174, 7: 74184, 8: 76562, 9: 40316}):
    raise SystemExit(f"[FAIL] Full-Q histogram drift: {full_hist}")
if half_hist != Counter({0: 125799, 9: 125800}):
    raise SystemExit(f"[FAIL] Half-Q histogram drift: {half_hist}")

manifest = json.loads(manifest_path.read_text())
npz_manifest = json.loads(npz_manifest_path.read_text())
audit = json.loads(audit_path.read_text())
if manifest.get("status") != "passed":
    raise SystemExit("[FAIL] combined alignment manifest not passed")
if npz_manifest.get("status") != "passed" or npz_manifest.get("completed_rows") != expected:
    raise SystemExit("[FAIL] Qwen NPZ manifest not passed/full")
if audit.get("status") != "passed" or audit.get("rows") != expected:
    raise SystemExit("[FAIL] exhaustive Qwen cache audit not passed/full")
if audit.get("semantic_gate", {}).get("status") != "passed":
    raise SystemExit("[FAIL] Qwen cache semantic gate not passed")

# Deterministic distributed probes bind the selected TSV caption/id to the
# embedded NPZ provenance without another 90-GiB full-cache scan.
indices = sorted({round(i * (expected - 1) / 31) for i in range(32)})
selected = full if arm == "fullq" else half
for index in indices:
    path = npz_dir / names[index]
    if not path.is_file():
        raise SystemExit(f"[FAIL] sampled Qwen NPZ missing: {path}")
    with np.load(path) as data:
        if str(data["clip_id"].item()) != selected[index]["id"]:
            raise SystemExit(f"[FAIL] embedded clip id mismatch at row {index}")
        digest = hashlib.sha256(selected[index]["caption"].encode("utf-8")).hexdigest()
        if str(data["caption_sha256"].item()) != digest:
            raise SystemExit(f"[FAIL] embedded Qwen caption mismatch at row {index}")
print(
    f"[OK] arm={arm} rows={expected:,}; official Qwen caption mapping, "
    f"NPZ provenance, and actual-clip Q alignment passed"
)
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] ARM=$ARM; no checkpoint or GPU process started."
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

python - "$CONTRACT" "$ARM" "$TRAIN_TSV" "$Q_MANIFEST" "$GT_CACHE" \
    "$QWEN_NPZ_MANIFEST" "$QWEN_CACHE_AUDIT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out, arm, tsv, q_manifest, cache, npz_manifest, cache_audit = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

critical = [
    "scripts/preprocess/make_phase8_qwen_meansim_tsvs.py",
    "scripts/training_pipelines/train_pipeline_phase8_qwen_fullq_halfq_quarter.sh",
    "scripts/training_pipelines/sequence_phase8_qwen_fullq_halfq_quarter.sh",
    "migrate_stage1_to_stage2_ckpt.py",
    "meanaudio/runner_flowmatching.py",
    "meanaudio/runner_meanflow.py",
    "meanaudio/model/networks.py",
]
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_qwen_fullq_halfq_quarter_e2e",
    "arm": arm,
    "from_scratch": True,
    "train_tsv": tsv,
    "train_tsv_sha256": sha(tsv),
    "q_alignment_manifest": q_manifest,
    "q_alignment_manifest_sha256": sha(q_manifest),
    "qwen_cache_list_sha256": sha(cache),
    "qwen_npz_manifest_sha256": sha(npz_manifest),
    "qwen_cache_audit_sha256": sha(cache_audit),
    "stage1_updates": 100000,
    "stage2_updates": 50000,
    "stage2_final_iteration": 150000,
    "stage1_use_q_conditioning": True,
    "stage2_use_q_conditioning": True,
    "stage1_to_stage2_q_initialization": "preserve",
    "q_semantics": (
        "actual-clip MeanSimilarity decile q3..q9"
        if arm == "fullq"
        else "actual-clip MeanSimilarity balanced rank split q0/q9"
    ),
    "caption_semantics": "official ATTM Qwen caption mapped by exact Jamendo track",
    "seed": 14159265,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "accumulation_steps": 1,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "matched_controls": {
        "same_qwen_captions": True,
        "same_audio_latents": True,
        "same_row_order": True,
        "same_qwen_npz_cache": True,
        "same_seed_optimizer_and_schedule": True,
        "only_q_granularity_differs": True,
    },
    "critical_file_sha256": {rel: sha(root / rel) for rel in critical},
}
try:
    payload["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
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

python - "$ARM" "$S1_CKPT" "$S2_CKPT" "$CONTRACT" "$FINAL_AUDIT" \
    "$S1_DIR" "$S2_DIR" "$TRAIN_TSV" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf

arm, s1_path, s2_path, contract, out, s1_dir, s2_dir, train_tsv = sys.argv[1:]
s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
s2 = torch.load(s2_path, map_location="cpu", weights_only=False)
issues = []
if s1.get("it") != 100000:
    issues.append(f"S1 iteration={s1.get('it')}")
if s2.get("it") != 150000:
    issues.append(f"S2 iteration={s2.get('it')}")
for label, state in (("S1", s1), ("S2", s2)):
    if "q_embed.weight" not in state.get("weights", {}):
        issues.append(f"{label} lacks q_embed.weight")
configs = {}
for label, directory, model, iterations in (
    ("S1", Path(s1_dir), "fluxaudio_s", 100000),
    ("S2", Path(s2_dir), "meanaudio_s", 150000),
):
    candidates = sorted(directory.glob("train-*-hydra/config.yaml"), key=lambda p: p.stat().st_mtime)
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
        "data.AudioCaps_npz.npz_dir": "/mnt/HDD/kojiek/phase8_qwen_official_matched_npz",
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
    "arm": arm,
    "stage1_iteration": s1.get("it"),
    "stage2_iteration": s2.get("it"),
    "stage1_use_q_conditioning": True,
    "stage2_use_q_conditioning": True,
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

echo "[COMPLETE] ARM=$ARM S1=100k S2=50k"
