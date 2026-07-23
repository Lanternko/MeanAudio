#!/usr/bin/env bash
# One matched end-to-end quarter-scale arm.
#
#   S1: 100k / historical 400k
#   S2:  50k / historical 200k
#
# Both arms use the same aligned half-Q TSV, seed, row order, cache, optimizer
# settings, and schedule. The No-Q arm ignores q_level in both stages; the
# half-Q arm consumes q0/q9 in both stages.

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

S1_UPDATES=100000
S2_UPDATES=50000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
EXPECTED_ROWS=251599
SEED=14159265
LR=1e-4

ARM="${ARM:?ARM must be noq or halfq}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-true}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
case "$ARM" in
    noq)
        PREFIX=phase8_quarter_e2e_noq
        USE_Q=false
        ;;
    halfq)
        PREFIX=phase8_quarter_e2e_halfq
        USE_Q=true
        ;;
    *)
        echo "[FAIL] ARM=$ARM; expected noq or halfq" >&2
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

for path in "$ALIGNED_TSV" "$ALIGNED_MANIFEST" "$SOURCE_JSONL" "$GT_CACHE" \
    "$NPZ_DIR/MANIFEST.tsv" "$NPZ_DIR/FULL_VALIDATION.json" \
    "$NPZ_DIR/FULL_GATE_PASSED.json" \
    "$ROOT/scripts/preprocess/make_phase8_halfq_tsv.py" \
    "$ROOT/scripts/preprocess/align_meansim_q_to_catalog.py" \
    "$ROOT/migrate_stage1_to_stage2_ckpt.py" "$ROOT/set_training_stage.py"; do
    [ -e "$path" ] || { echo "[FAIL] missing $path" >&2; exit 2; }
done

# Recompute the actual-clip MeanSimilarity assignment before accepting the
# binary dataset. Existing artifacts are verified, never trusted by name.
python scripts/preprocess/make_phase8_halfq_tsv.py \
    --input "$ALIGNED_TSV" \
    --aligned-manifest "$ALIGNED_MANIFEST" \
    --source-jsonl "$SOURCE_JSONL" \
    --output "$HALFQ_TSV" \
    --manifest "$HALFQ_MANIFEST" \
    --expected-rows "$EXPECTED_ROWS"

audit_dir=$(mktemp -d /tmp/phase8-halfq-e2e.XXXXXX)
trap 'rm -rf "$audit_dir"' EXIT
python scripts/preprocess/align_meansim_q_to_catalog.py \
    --input "$ALIGNED_TSV" \
    --source-jsonl "$SOURCE_JSONL" \
    --manifest "$audit_dir/aligned-audit.json" \
    --require-current-match

python - "$HALFQ_TSV" "$HALFQ_MANIFEST" "$GT_CACHE" "$NPZ_DIR" \
    "$EXPECTED_ROWS" "$ARM" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

tsv, manifest_path, cache, npz_dir = map(Path, sys.argv[1:5])
expected = int(sys.argv[5])
arm = sys.argv[6]
rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
names = [line.strip() for line in cache.open(encoding="utf-8") if line.strip()]
catalog_manifest = list(
    csv.DictReader(
        (npz_dir / "MANIFEST.tsv").open(encoding="utf-8"), delimiter="\t"
    )
)
if not (len(rows) == len(names) == len(catalog_manifest) == expected):
    raise SystemExit(
        f"[FAIL] cardinality rows={len(rows)} cache={len(names)} "
        f"manifest={len(catalog_manifest)} expected={expected}"
    )
if len({row["id"] for row in rows}) != expected or len(set(names)) != expected:
    raise SystemExit("[FAIL] duplicate TSV id or cache filename")
histogram = Counter(int(row["q_level"]) for row in rows)
if histogram != Counter({0: 125799, 9: 125800}):
    raise SystemExit(f"[FAIL] half-Q histogram changed: {histogram}")

for index, (row, name, item) in enumerate(zip(rows, names, catalog_manifest)):
    if item["row_index"] != str(index):
        raise SystemExit(f"[FAIL] manifest position mismatch at {index}")
    if (item["clip_id"], item["npz_fname"]) != (row["id"], name):
        raise SystemExit(f"[FAIL] row/cache/manifest mismatch at {index}")

for index in (0, 1, 100, 1000, 10000, expected - 1):
    with np.load(npz_dir / names[index]) as data:
        if str(data["clip_id"].item()) != rows[index]["id"]:
            raise SystemExit(f"[FAIL] embedded clip id mismatch at {index}")
        caption_hash = hashlib.sha256(
            rows[index]["caption"].encode("utf-8")
        ).hexdigest()
        if str(data["caption_sha256"].item()) != caption_hash:
            raise SystemExit(f"[FAIL] embedded caption hash mismatch at {index}")

contract = json.loads(manifest_path.read_text())
required = {
    "rows": expected,
    "historical_q_rows_verified": expected,
    "unique_source_rows": expected,
    "q_histogram": {"0": 125799, "9": 125800},
    "resolution_histogram": {"stripped_final_partition_suffix": expected},
}
for key, value in required.items():
    if contract.get(key) != value:
        raise SystemExit(f"[FAIL] half-Q manifest drift at {key}")
for gate_name in ("FULL_VALIDATION.json", "FULL_GATE_PASSED.json"):
    gate = json.loads((npz_dir / gate_name).read_text())
    if gate.get("status") != "passed":
        raise SystemExit(f"[FAIL] cache gate not passed: {gate_name}")
print(
    f"[OK] arm={arm} full alignment preflight rows={expected:,} "
    f"halfq={dict(sorted(histogram.items()))}"
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

python - "$CONTRACT" "$ARM" "$USE_Q" "$HALFQ_TSV" "$HALFQ_MANIFEST" \
    "$SEED" "$LR" "$S1_UPDATES" "$S2_UPDATES" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out, arm, use_q_raw, tsv, manifest, seed, lr, s1_updates, s2_updates
) = sys.argv[1:]
root = Path("/home/kojiek/MeanAudio")

def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

critical = [
    "scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh",
    "scripts/training_pipelines/sequence_phase8_halfq_quarter.sh",
    "scripts/preprocess/make_phase8_halfq_tsv.py",
    "migrate_stage1_to_stage2_ckpt.py",
    "meanaudio/runner_flowmatching.py",
    "meanaudio/runner_meanflow.py",
    "meanaudio/model/networks.py",
]
use_q = use_q_raw == "true"
payload = {
    "schema_version": 2,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_halfq_quarter_e2e",
    "arm": arm,
    "from_scratch": True,
    "train_tsv": tsv,
    "train_tsv_sha256": sha(tsv),
    "halfq_manifest": manifest,
    "halfq_manifest_sha256": sha(manifest),
    "stage1_updates": int(s1_updates),
    "stage2_updates": int(s2_updates),
    "stage2_final_iteration": int(s1_updates) + int(s2_updates),
    "stage1_use_q_conditioning": use_q,
    "stage2_use_q_conditioning": use_q,
    "q_semantics": (
        "lower actual-clip MeanSimilarity rank half q0; upper half q9"
        if use_q
        else "q_level present but ignored; effective q10"
    ),
    "stage1_to_stage2_q_initialization": "preserve",
    "seed": int(seed),
    "learning_rate": float(lr),
    "batch_size": 8,
    "accumulation_steps": 1,
    "use_text_attention_mask": False,
    "multi_cap": False,
    "matched_controls": {
        "same_random_seed": True,
        "same_tsv_and_row_order": True,
        "same_cache": True,
        "same_optimizer_and_schedule": True,
        "only_conditioning_route_differs": True,
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

checkpoint_it() {
    local path="$1"
    python - "$path" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
if not path.is_file():
    print(-1)
else:
    print(int(torch.load(path, map_location="cpu", weights_only=False)["it"]))
PY
}

s1_it=$(checkpoint_it "$S1_CKPT")
if [ "$s1_it" -lt "$S1_UPDATES" ]; then
    python set_training_stage.py --stage 1
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio model=fluxaudio_s exp_id="$EXP_S1" \
        num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
        "+use_q_conditioning=$USE_Q" batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=25000 save_checkpoint_interval=25000 \
        ++ema.checkpoint_every=25000 +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
        save_eval_interval=999999 "data.AudioCaps_npz.tsv=$HALFQ_TSV" \
        "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$HALFQ_TSV" \
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
        "+use_q_conditioning=$USE_Q" batch_size=8 +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=25000 save_checkpoint_interval=25000 \
        ++ema.checkpoint_every=25000 +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
        save_eval_interval=999999 "data.AudioCaps_npz.tsv=$HALFQ_TSV" \
        "+data.AudioCaps_npz.gt_cache=$GT_CACHE" \
        "data.AudioCaps_val_npz.tsv=$HALFQ_TSV" \
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

python - "$ARM" "$USE_Q" "$S1_CKPT" "$S2_CKPT" "$CONTRACT" \
    "$FINAL_AUDIT" "$S1_DIR" "$S2_DIR" "$HALFQ_TSV" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

(
    arm,
    use_q_raw,
    s1_path,
    s2_path,
    contract_path,
    out_path,
    s1_dir_raw,
    s2_dir_raw,
    train_tsv,
) = sys.argv[1:]
use_q = use_q_raw == "true"
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
for label, directory, expected_model, expected_iterations in (
    ("S1", Path(s1_dir_raw), "fluxaudio_s", 100000),
    ("S2", Path(s2_dir_raw), "meanaudio_s", 150000),
):
    candidates = sorted(
        directory.glob("train-*-hydra/config.yaml"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        issues.append(f"{label} Hydra config missing")
        continue
    config_path = candidates[-1]
    config = OmegaConf.load(config_path)
    configs[label] = str(config_path)
    checks = {
        "model": expected_model,
        "num_iterations": expected_iterations,
        "seed": 14159265,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "accumulation_steps": 1,
        "use_q_conditioning": use_q,
        "use_text_attention_mask": False,
        "multi_cap": False,
        "data.AudioCaps_npz.tsv": train_tsv,
        "data.AudioCaps_npz.npz_dir": (
            "/mnt/HDD/kojiek/phase8_legacy_matched_npz"
        ),
    }
    for key, expected in checks.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            issues.append(
                f"{label} config {key}={actual!r}, expected={expected!r}"
            )
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "failed" if issues else "passed",
    "issues": issues,
    "arm": arm,
    "stage1_iteration": s1.get("it"),
    "stage2_iteration": s2.get("it"),
    "stage1_use_q_conditioning": use_q,
    "stage2_use_q_conditioning": use_q,
    "contract": contract_path,
    "hydra_configs": configs,
}
path = Path(out_path)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
print(json.dumps(payload, indent=2, sort_keys=True))
if issues:
    raise SystemExit(2)
PY

echo "[COMPLETE] ARM=$ARM S1=100k S2=50k"
