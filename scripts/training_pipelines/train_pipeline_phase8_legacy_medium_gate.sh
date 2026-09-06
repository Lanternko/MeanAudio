#!/usr/bin/env bash
# End-to-end medium gate for the provenance-correct Phase-8 legacy reproduction.
# This deliberately runs only 100 S1 + 100 S2 updates and 64 eval prompts.  It
# proves launcher/config/data/migration/checkpoint/eval integration, not final
# model quality or convergence.

set -euo pipefail

source /home/kojiek/venvs/dac/bin/activate

WORK_DIR=/home/kojiek/MeanAudio
SMOKE_DIR=/home/kojiek/smoke_data/phase8_legacy_gate_n4096
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz_gate_n4096
TRAIN_TSV="$SMOKE_DIR/train_n4096.tsv"
TRAIN_CACHE="$SMOKE_DIR/npz_cache_n4096.txt"
EVAL_TSV=/home/kojiek/smoke_data/phase8_mask_ab_n128/eval_train_n64.tsv
PREFIX=phase8_legacy_medium_gate_n4096

cd "$WORK_DIR"

ORIGINAL_STAGE=$(python - <<'PY'
from pathlib import Path
from set_training_stage import detect_current_stage

print(detect_current_stage(Path("meanaudio/model/mean_flow.py").read_text()))
PY
)
restore_stage() {
    python set_training_stage.py --stage "$ORIGINAL_STAGE" >/dev/null
}
trap restore_stage EXIT

for required in \
    "$TRAIN_TSV" \
    "$TRAIN_CACHE" \
    "$NPZ_DIR/MANIFEST.tsv" \
    "$EVAL_TSV"; do
    if [ ! -f "$required" ]; then
        echo "[STOP] Missing medium-gate input: $required" >&2
        exit 2
    fi
done

export EXP_PREFIX="$PREFIX"
export TRAIN_TSV
export GT_CACHE="$TRAIN_CACHE"
export SINGLECAP_NPZ="$NPZ_DIR"
export EXPECTED_ROWS=4096
export S1_ITERATIONS=100
export S2_ITERATIONS=100
export SAVE_WEIGHTS_INTERVAL=50
export SAVE_CHECKPOINT_INTERVAL=50
export EMA_CHECKPOINT_INTERVAL=50
export USE_Q_CONDITIONING=true
export USE_TEXT_ATTENTION_MASK=false
export RUN_PRIMARY_EVAL=true
export RUN_JAMENDO_EVAL=false
export TSV_MUSICCAPS="$EVAL_TSV"
export EVAL_NUM_SAMPLES=64
export EVAL_SKIP_AES=true

bash scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh

python - "$PREFIX" <<'PY'
import re
import sys
from pathlib import Path

import torch

root = Path("/home/kojiek/MeanAudio")
prefix = sys.argv[1]
s1 = root / "exps" / f"{prefix}_stage1_100" / f"{prefix}_stage1_100_ckpt_last.pth"
s2_dir = root / "exps" / f"{prefix}_stage2_100"
s2 = s2_dir / f"{prefix}_stage2_100_ckpt_last.pth"
ema = s2_dir / f"{prefix}_stage2_100_ema_final.pth"
metrics = (
    root / "eval_output" / "metrics" / f"{prefix}_stage2_100_musiccaps" / "metrics.txt"
)

if torch.load(s1, map_location="cpu", weights_only=False)["it"] != 100:
    raise SystemExit("[FAIL] medium gate S1 checkpoint is not at iteration 100")
if torch.load(s2, map_location="cpu", weights_only=False)["it"] != 200:
    raise SystemExit("[FAIL] medium gate S2 checkpoint is not at total iteration 200")
if not ema.is_file():
    raise SystemExit("[FAIL] medium gate S2 EMA is missing")
text = metrics.read_text()
match = re.search(r"^clap_score:\s*([-+0-9.eE]+)$", text, re.MULTILINE)
if not match:
    raise SystemExit("[FAIL] medium gate eval did not produce a CLAP metric")
print("[OK] medium end-to-end gate completed")
print("[OK] S1=100, migrated S2 total=200, EMA present, eval=64")
print(f"[INFO] short-run CLAP={float(match.group(1)):.4f} (wiring only; not convergence)")
PY
