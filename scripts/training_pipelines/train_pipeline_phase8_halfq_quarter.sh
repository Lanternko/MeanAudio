#!/usr/bin/env bash
# Matched quarter-scale comparison:
#   baseline: existing No-Q S2 EMA at 400k + 50k = iteration 450k
#   half-Q:   same Stage-1 source, 50k S2 updates, balanced actual-clip
#             MeanSimilarity lower-half=q0 / upper-half=q9
#
# Metrics:
#   Stage 1: full 5,521-prompt MusicCaps, native 25-step Flow Matching
#   Global:  full 5,521-prompt MusicCaps, 1-step MeanFlow for baseline No-Q
#            and half-Q q9/q0.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz
GT_CACHE="$DATA/npz_cache_train.txt"
SOURCE_JSONL=/home/kojiek/research/music_cleaning/results_20260119_043407.jsonl
ALIGNED_TSV="$DATA/phase8_legacy_catalog_train_meansim_aligned.tsv"
ALIGNED_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_aligned.manifest.json"
HALFQ_TSV="$DATA/phase8_legacy_catalog_train_meansim_halfq.tsv"
HALFQ_MANIFEST="$DATA/phase8_legacy_catalog_train_meansim_halfq.manifest.json"
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py

SOURCE_ID=phase8_catalog_matched_noq_stage1_400000
SOURCE="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ckpt_last.pth"
SOURCE_EMA="$ROOT/exps/$SOURCE_ID/${SOURCE_ID}_ema_final.pth"
SOURCE_IT=400000
S2_UPDATES=50000
FINAL_IT=450000
SEED=14159265
LR=1e-4
EXPECTED_ROWS=251599

BASELINE_PARENT=phase8_catalog_matched_noq_stage2_200000
BASELINE_CONFIG="$ROOT/exps/$BASELINE_PARENT/train-2026-07-19_22-07-08-hydra/config.yaml"
BASELINE_EMA="$ROOT/exps/$BASELINE_PARENT/ema_ckpts/0.450000.pt"
BASELINE_LABEL=phase8_quarter_noq_baseline_it450000_musiccaps_noq
BASELINE_OUT="$ROOT/eval_output/$BASELINE_LABEL"
BASELINE_METRICS="$ROOT/eval_output/metrics/$BASELINE_LABEL/metrics.txt"
BASELINE_EVAL_LOG="$LOG_ROOT/${BASELINE_LABEL}_eval.log"

S1_LABEL=phase8_catalog_matched_noq_stage1_400000_musiccaps_fm25_noq_nomask
S1_METRICS="$ROOT/eval_output/metrics/$S1_LABEL/metrics.txt"

EXP_ID=phase8_halfq_qpilot_s2_50000
EXP_DIR="$ROOT/exps/$EXP_ID"
CKPT="$EXP_DIR/${EXP_ID}_ckpt_last.pth"
HALFQ_EMA="$EXP_DIR/ema_ckpts/0.450000.pt"
TRAIN_LOG="$LOG_ROOT/${EXP_ID}.log"
INIT_LOG="$LOG_ROOT/${EXP_ID}_qinit.log"
CONTRACT="$LOG_ROOT/${EXP_ID}_contract.json"
FINAL_REPORT="$LOG_ROOT/${EXP_ID}_FINAL_METRICS.json"

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-true}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
case "$PREFLIGHT_ONLY" in true|false) ;; *)
    echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2
    exit 2
esac
case "$RUN_MODE" in fresh|resume) ;; *)
    echo "[FAIL] EXPERIMENT_RUN_MODE must be fresh or resume" >&2
    exit 2
esac

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for path in "$SOURCE" "$SOURCE_EMA" "$BASELINE_CONFIG" "$BASELINE_EMA" \
    "$ALIGNED_TSV" "$ALIGNED_MANIFEST" "$SOURCE_JSONL" "$GT_CACHE" \
    "$NPZ_DIR/MANIFEST.tsv" "$NPZ_DIR/FULL_VALIDATION.json" \
    "$NPZ_DIR/FULL_GATE_PASSED.json" "$MUSICCAPS" "$EVALUATOR" \
    "$ROOT/scripts/preprocess/make_phase8_halfq_tsv.py" \
    "$ROOT/scripts/preprocess/align_meansim_q_to_catalog.py"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

# Idempotent full-data verification. If the half-Q files do not yet exist this
# creates them; otherwise every row and immutable manifest field is recomputed.
python scripts/preprocess/make_phase8_halfq_tsv.py \
    --input "$ALIGNED_TSV" \
    --aligned-manifest "$ALIGNED_MANIFEST" \
    --source-jsonl "$SOURCE_JSONL" \
    --output "$HALFQ_TSV" \
    --manifest "$HALFQ_MANIFEST" \
    --expected-rows "$EXPECTED_ROWS"

audit_dir=$(mktemp -d /tmp/phase8-halfq-preflight.XXXXXX)
trap 'rm -rf "$audit_dir"' EXIT
python scripts/preprocess/align_meansim_q_to_catalog.py \
    --input "$ALIGNED_TSV" \
    --source-jsonl "$SOURCE_JSONL" \
    --manifest "$audit_dir/aligned-audit.json" \
    --require-current-match

python - "$SOURCE" "$BASELINE_EMA" "$BASELINE_CONFIG" "$HALFQ_TSV" \
    "$HALFQ_MANIFEST" "$GT_CACHE" "$NPZ_DIR" "$S1_METRICS" \
    "$EXPECTED_ROWS" <<'PY'
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

(
    source_path,
    baseline_ema_path,
    baseline_config_path,
    halfq_tsv_path,
    halfq_manifest_path,
    cache_path,
    npz_dir_path,
    s1_metrics_path,
    expected_raw,
) = sys.argv[1:]
source_path = Path(source_path)
baseline_ema_path = Path(baseline_ema_path)
baseline_config_path = Path(baseline_config_path)
halfq_tsv_path = Path(halfq_tsv_path)
halfq_manifest_path = Path(halfq_manifest_path)
cache_path = Path(cache_path)
npz_dir = Path(npz_dir_path)
s1_metrics_path = Path(s1_metrics_path)
expected = int(expected_raw)

source = torch.load(source_path, map_location="cpu", weights_only=False)
if source.get("it") != 400000:
    raise SystemExit(f"[FAIL] Stage-1 source iteration={source.get('it')}")
if not all(key in source for key in ("weights", "ema", "optimizer", "scheduler")):
    raise SystemExit("[FAIL] Stage-1 source is not resumable")
if "q_embed.weight" not in source["weights"]:
    raise SystemExit("[FAIL] Stage-1 source lacks q_embed.weight")

baseline_ema = torch.load(baseline_ema_path, map_location="cpu", weights_only=True)
if baseline_ema.get("_extra_state", {}).get("step") != 450000:
    raise SystemExit("[FAIL] quarter baseline EMA is not iteration 450000")
if "ema_model.q_embed.weight" not in baseline_ema:
    raise SystemExit("[FAIL] quarter baseline EMA lacks q_embed")

cfg = OmegaConf.load(baseline_config_path)
required_cfg = {
    "seed": 14159265,
    "batch_size": 8,
    "accumulation_steps": 1,
    "learning_rate": 1e-4,
    "use_q_conditioning": False,
    "use_text_attention_mask": False,
}
actual_cfg = {key: OmegaConf.select(cfg, key) for key in required_cfg}
if actual_cfg != required_cfg:
    raise SystemExit(
        f"[FAIL] quarter baseline config drift: "
        f"actual={actual_cfg} required={required_cfg}"
    )

with halfq_tsv_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with cache_path.open(encoding="utf-8") as handle:
    names = [line.strip() for line in handle if line.strip()]
with (npz_dir / "MANIFEST.tsv").open(encoding="utf-8", newline="") as handle:
    manifest = list(csv.DictReader(handle, delimiter="\t"))
if not (len(rows) == len(names) == len(manifest) == expected):
    raise SystemExit(
        f"[FAIL] cardinality mismatch rows={len(rows)} cache={len(names)} "
        f"manifest={len(manifest)} expected={expected}"
    )
if len({row["id"] for row in rows}) != expected or len(set(names)) != expected:
    raise SystemExit("[FAIL] duplicate TSV id or cache filename")
q_hist = Counter(int(row["q_level"]) for row in rows)
if q_hist != Counter({0: 125799, 9: 125800}):
    raise SystemExit(f"[FAIL] half-Q histogram mismatch: {q_hist}")

for index, (row, name, item) in enumerate(zip(rows, names, manifest)):
    if item["row_index"] != str(index):
        raise SystemExit(f"[FAIL] manifest row_index mismatch at {index}")
    if (item["clip_id"], item["npz_fname"]) != (row["id"], name):
        raise SystemExit(f"[FAIL] TSV/cache/manifest mismatch at {index}")

for index in (0, 1, 100, 1000, 10000, expected - 1):
    with np.load(npz_dir / names[index]) as data:
        if str(data["clip_id"].item()) != rows[index]["id"]:
            raise SystemExit(f"[FAIL] embedded clip_id mismatch at {index}")
        expected_caption_hash = hashlib.sha256(
            rows[index]["caption"].encode("utf-8")
        ).hexdigest()
        if str(data["caption_sha256"].item()) != expected_caption_hash:
            raise SystemExit(f"[FAIL] embedded caption hash mismatch at {index}")

halfq_manifest = json.loads(halfq_manifest_path.read_text())
if halfq_manifest.get("historical_q_rows_verified") != expected:
    raise SystemExit("[FAIL] half-Q manifest lacks full historical-Q verification")
if halfq_manifest.get("unique_source_rows") != expected:
    raise SystemExit("[FAIL] half-Q manifest source rows are not one-to-one")
if halfq_manifest.get("resolution_histogram") != {
    "stripped_final_partition_suffix": expected
}:
    raise SystemExit("[FAIL] half-Q id normalization provenance changed")
if halfq_manifest.get("q_histogram") != {"0": 125799, "9": 125800}:
    raise SystemExit("[FAIL] half-Q manifest histogram changed")

for gate_name in ("FULL_VALIDATION.json", "FULL_GATE_PASSED.json"):
    gate = json.loads((npz_dir / gate_name).read_text())
    if gate.get("status") != "passed":
        raise SystemExit(f"[FAIL] NPZ gate not passed: {gate_name}")

metric_values = {}
for line in s1_metrics_path.read_text().splitlines():
    if ":" not in line:
        continue
    key, value = line.split(":", 1)
    key = key.strip()
    if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
        metric_values[key] = float(value)
if set(metric_values) != {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
    raise SystemExit(f"[FAIL] incomplete Stage-1 metrics: {metric_values}")
if not all(math.isfinite(value) for value in metric_values.values()):
    raise SystemExit("[FAIL] non-finite Stage-1 metric")

print(
    "[OK] quarter preflight: "
    f"source_it=400000 baseline_it=450000 rows={expected:,} "
    f"halfq={dict(sorted(q_hist.items()))} s1_clap={metric_values['clap_score']}"
)
PY

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] No training, generation, or GPU metric process started."
    exit 0
fi

if [ "$RUN_MODE" = fresh ]; then
    conflicts=()
    for path in "$EXP_DIR" "$TRAIN_LOG" "$INIT_LOG" "$CONTRACT" "$FINAL_REPORT" \
        "$BASELINE_OUT" "$ROOT/eval_output/metrics/$BASELINE_LABEL" \
        "$ROOT/eval_output/${EXP_ID}_musiccaps_q9" \
        "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q9" \
        "$ROOT/eval_output/${EXP_ID}_musiccaps_q0" \
        "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q0"; do
        [ -e "$path" ] && conflicts+=("$path")
    done
    if [ "${#conflicts[@]}" -gt 0 ]; then
        printf '[FAIL] fresh artifact exists: %s\n' "${conflicts[@]}" >&2
        exit 2
    fi
fi

mkdir -p "$EXP_DIR" "$LOG_ROOT"

python - "$CONTRACT" "$SOURCE" "$SOURCE_EMA" "$BASELINE_EMA" \
    "$BASELINE_CONFIG" "$ALIGNED_TSV" "$ALIGNED_MANIFEST" "$HALFQ_TSV" \
    "$HALFQ_MANIFEST" "$S1_METRICS" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out,
    source,
    source_ema,
    baseline_ema,
    baseline_config,
    aligned_tsv,
    aligned_manifest,
    halfq_tsv,
    halfq_manifest,
    s1_metrics,
) = sys.argv[1:]

def sha(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_halfq_quarter",
    "stage1_source": source,
    "stage1_source_sha256": sha(source),
    "stage1_ema": source_ema,
    "stage1_ema_sha256": sha(source_ema),
    "stage1_iteration": 400000,
    "stage1_use_q_conditioning": False,
    "stage1_effective_q": 10,
    "stage1_metrics": s1_metrics,
    "stage1_metrics_sha256": sha(s1_metrics),
    "baseline_reuses_saved_trajectory": True,
    "baseline_ema": baseline_ema,
    "baseline_ema_sha256": sha(baseline_ema),
    "baseline_iteration": 450000,
    "baseline_config": baseline_config,
    "baseline_config_sha256": sha(baseline_config),
    "baseline_use_q_conditioning": False,
    "baseline_effective_q": 10,
    "aligned_tsv": aligned_tsv,
    "aligned_tsv_sha256": sha(aligned_tsv),
    "aligned_manifest": aligned_manifest,
    "aligned_manifest_sha256": sha(aligned_manifest),
    "halfq_tsv": halfq_tsv,
    "halfq_tsv_sha256": sha(halfq_tsv),
    "halfq_manifest": halfq_manifest,
    "halfq_manifest_sha256": sha(halfq_manifest),
    "halfq_mapping": {
        "lower_rank_half": 0,
        "upper_rank_half": 9,
        "rank_key": ["mean_similarity", "source_id"],
        "counts": {"0": 125799, "9": 125800},
    },
    "halfq_initialization": "copy_q10_exactly_to_q0_through_q9",
    "halfq_use_q_conditioning": True,
    "s2_updates": 50000,
    "final_iteration": 450000,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "accumulation_steps": 1,
    "seed": 14159265,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "metrics": {
        "stage1": "MusicCaps 5521; 25-step native Flow Matching; no-Q",
        "global": "MusicCaps 5521; 1-step MeanFlow; baseline no-Q and half-Q q9/q0",
        "note": "Stage-1 FM25 and global MeanFlow1 are reported separately",
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
    previous = json.loads(path.read_text())
    drift = [
        key for key, value in payload.items()
        if key not in {"created_at", "git_head"} and previous.get(key) != value
    ]
    if drift:
        raise SystemExit(f"[FAIL] immutable contract drift: {drift}")
else:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
print(f"[OK] contract: {path}")
PY

run_eval() {
    local label="$1"
    local model="$2"
    shift 2
    local out="$ROOT/eval_output/$label"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local log="$LOG_ROOT/${label}_eval.log"

    if [ -f "$metrics" ]; then
        echo "[SKIP] complete metrics: $metrics"
        return 0
    fi
    mkdir -p "$out/audio"
    python eval.py --variant meanaudio_s --model_path "$model" \
        --output "$out/audio" --tsv "$MUSICCAPS" --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 --cfg_strength 0.5 \
        --no_text_attention_mask --full_precision "$@" 2>&1 | tee "$log"
    audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
    if [ "$audio_n" -ne 5521 ]; then
        echo "[FAIL] $label generated $audio_n/5521 files" | tee -a "$log" >&2
        return 2
    fi
    python "$EVALUATOR" --gen_dir "$out/audio" --tsv "$MUSICCAPS" \
        --exp_name "$label" --num_samples 5521 2>&1 | tee -a "$log"
    [ -f "$metrics" ] || {
        echo "[FAIL] evaluator did not create $metrics" >&2
        return 2
    }
}

# Evaluate the exact saved 50k No-Q trajectory before training the matched arm.
run_eval "$BASELINE_LABEL" "$BASELINE_EMA" --no_q

if [ ! -f "$CKPT" ]; then
    python set_training_stage.py --stage 2
    python migrate_stage1_to_stage2_ckpt.py \
        --s1_ckpt "$SOURCE" --s2_out "$CKPT" --q-init copy-null \
        2>&1 | tee "$INIT_LOG"
fi

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=meanaudio_s exp_id="$EXP_ID" \
    num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
    +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$HALFQ_TSV" \
    "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
    "data.AudioCaps_val_npz.tsv=$HALFQ_TSV" \
    "data.AudioCaps_val_npz.gt_cache=$GT_CACHE" \
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" ++multi_cap=False \
    2>&1 | tee "$TRAIN_LOG"

[ -f "$HALFQ_EMA" ] || {
    echo "[FAIL] missing matched half-Q EMA at iteration 450000: $HALFQ_EMA" >&2
    exit 2
}

run_eval "${EXP_ID}_musiccaps_q9" "$HALFQ_EMA" --quality_level 9
run_eval "${EXP_ID}_musiccaps_q0" "$HALFQ_EMA" --quality_level 0

python - "$S1_METRICS" "$BASELINE_METRICS" \
    "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q9/metrics.txt" \
    "$ROOT/eval_output/metrics/${EXP_ID}_musiccaps_q0/metrics.txt" \
    "$CONTRACT" "$FINAL_REPORT" <<'PY'
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

s1_path, baseline_path, q9_path, q0_path, contract_path, out_path = map(
    Path, sys.argv[1:]
)

def metrics(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(value)
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if set(values) != required or not all(math.isfinite(v) for v in values.values()):
        raise SystemExit(f"[FAIL] incomplete/non-finite metrics at {path}: {values}")
    return values

s1 = metrics(s1_path)
baseline = metrics(baseline_path)
q9 = metrics(q9_path)
q0 = metrics(q0_path)
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_halfq_quarter",
    "contract": str(contract_path),
    "stage1": {
        "protocol": "MusicCaps 5521; FluxAudio; FM25; CFG 4.5; no-Q",
        "metrics": s1,
    },
    "global": {
        "protocol": "MusicCaps 5521; MeanAudio; MeanFlow1; CFG 0.5",
        "quarter_noq_baseline": baseline,
        "halfq_q9": q9,
        "halfq_q0": q0,
        "halfq_q9_minus_baseline_clap": (
            q9["clap_score"] - baseline["clap_score"]
        ),
        "halfq_q0_minus_baseline_clap": (
            q0["clap_score"] - baseline["clap_score"]
        ),
        "halfq_q9_minus_q0_clap": q9["clap_score"] - q0["clap_score"],
    },
    "interpretation": {
        "stage1_and_global_protocols_are_not_directly_comparable": True,
        "primary_halfq_endpoint": "q9",
        "diagnostic_halfq_endpoint": "q0",
        "no_checkpoint_cherrypick": True,
    },
}
tmp = out_path.with_suffix(out_path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[COMPLETE] quarter baseline + half-Q; report=$FINAL_REPORT"
