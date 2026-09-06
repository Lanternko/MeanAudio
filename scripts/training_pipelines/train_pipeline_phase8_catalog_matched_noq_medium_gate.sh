#!/usr/bin/env bash
# Small-scale end-to-end gate for catalog-matched true NoQ.
#
# Proves wiring only (not convergence):
#   100 S1 + migrate + 100 S2 + 64-prompt MusicCaps eval
# Same n4096 smoke slice as legacy medium gate, but USE_Q_CONDITIONING=false
# and bug-fixed --no_q eval (null token q=10).
#
# Do NOT treat short-run CLAP as a quality signal.

set -euo pipefail

source /home/kojiek/venvs/dac/bin/activate

WORK_DIR=/home/kojiek/MeanAudio
SMOKE_DIR=/home/kojiek/smoke_data/phase8_legacy_gate_n4096
NPZ_DIR=/mnt/HDD/kojiek/phase8_legacy_matched_npz_gate_n4096
TRAIN_TSV="$SMOKE_DIR/train_n4096.tsv"
# File on disk is npz_cache_n4096.txt (historical naming from gate builder).
TRAIN_CACHE="$SMOKE_DIR/npz_cache_n4096.txt"
EVAL_TSV=/home/kojiek/smoke_data/phase8_mask_ab_n128/eval_train_n64.tsv
PREFIX=phase8_catalog_matched_noq_medium_gate_n4096
GATE_SENTINEL=/home/kojiek/logs/phase8_legacy_repro_guard/noq_medium_gate_PASSED.json
GATE_LOG=/home/kojiek/logs/phase8_catalog_matched_noq_medium_gate.log

cd "$WORK_DIR"
mkdir -p /home/kojiek/logs "$(dirname "$GATE_SENTINEL")"

if [ ! -f "$TRAIN_CACHE" ]; then
    # Rebuild from MANIFEST if the cache list was removed.
    python - "$NPZ_DIR/MANIFEST.tsv" "$TRAIN_CACHE" <<'PY'
import csv
import sys
from pathlib import Path
man, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(man.open(), delimiter="\t"))
out.write_text("\n".join(r["npz_fname"] for r in rows) + "\n")
print(f"[OK] rebuilt {out} ({len(rows)} names)")
PY
fi

ORIGINAL_STAGE=$(python - <<'PY'
from pathlib import Path
from set_training_stage import detect_current_stage

print(detect_current_stage(Path("meanaudio/model/mean_flow.py").read_text()))
PY
)
restore_stage() {
    python set_training_stage.py --stage "$ORIGINAL_STAGE" >/dev/null || true
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
export LINEAR_WARMUP_STEPS=10
export EXPERIMENT_REGIME=clean_noq
# The gate is intentionally resumable because a prior successful wiring run may
# already exist.  The post-run audit below still rejects any Q/mask drift.
export EXPERIMENT_RUN_MODE=resume
export S1_USE_Q_CONDITIONING=false
export S2_USE_Q_CONDITIONING=false
export EVAL_Q_MODE=no_q
export USE_TEXT_ATTENTION_MASK=false
export RUN_PRIMARY_EVAL=true
export RUN_JAMENDO_EVAL=false
export TSV_MUSICCAPS="$EVAL_TSV"
export EVAL_NUM_SAMPLES=64
export EVAL_SKIP_AES=true

echo "======================================================"
echo "  Catalog-matched NoQ MEDIUM GATE (wiring only)"
echo "  prefix=$PREFIX  S1=100 S2=100 eval=64"
echo "  contract: S1_Q=false S2_Q=false eval=no_q NoMask"
echo "  log=$GATE_LOG"
echo "======================================================"

bash scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh \
    2>&1 | tee -a "$GATE_LOG"

python - "$PREFIX" "$GATE_SENTINEL" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

root = Path("/home/kojiek/MeanAudio")
prefix = sys.argv[1]
sentinel = Path(sys.argv[2])
s1 = root / "exps" / f"{prefix}_stage1_100" / f"{prefix}_stage1_100_ckpt_last.pth"
s2_dir = root / "exps" / f"{prefix}_stage2_100"
s2 = s2_dir / f"{prefix}_stage2_100_ckpt_last.pth"
ema = s2_dir / f"{prefix}_stage2_100_ema_final.pth"
metrics = (
    root / "eval_output" / "metrics" / f"{prefix}_stage2_100_musiccaps" / "metrics.txt"
)

issues = []
if not s1.is_file():
    issues.append("missing S1 ckpt")
else:
    it = torch.load(s1, map_location="cpu", weights_only=False).get("it", -1)
    if it != 100:
        issues.append(f"S1 it={it}, expected 100")

if not s2.is_file():
    issues.append("missing S2 ckpt")
else:
    it = torch.load(s2, map_location="cpu", weights_only=False).get("it", -1)
    if it != 200:
        issues.append(f"S2 total it={it}, expected 200")

if not ema.is_file():
    issues.append("missing S2 ema_final")

# Config must be true NoQ + NoMask.  Stage is NOT a hydra key — it is set by
# set_training_stage.py which switches the runner; the durable signal in
# config.yaml is model name (S1=fluxaudio_s, S2=meanaudio_s).
expected_models = {"S1": "fluxaudio_s", "S2": "meanaudio_s"}
for stage_name, exp_dir in (
    ("S1", root / "exps" / f"{prefix}_stage1_100"),
    ("S2", s2_dir),
):
    configs = sorted(exp_dir.glob("train-*-hydra/config.yaml"))
    if not configs:
        issues.append(f"{stage_name} missing hydra config")
        continue
    cfg = yaml.safe_load(configs[-1].read_text())
    if cfg.get("use_q_conditioning") is not False:
        issues.append(f"{stage_name} use_q_conditioning={cfg.get('use_q_conditioning')} (want false)")
    if cfg.get("use_text_attention_mask") is not False:
        issues.append(
            f"{stage_name} use_text_attention_mask={cfg.get('use_text_attention_mask')} (want false)"
        )
    want_model = expected_models[stage_name]
    got_model = cfg.get("model")
    if got_model != want_model:
        issues.append(f"{stage_name} model={got_model!r} (want {want_model!r})")

clap = None
if not metrics.is_file():
    issues.append("missing metrics.txt")
else:
    text = metrics.read_text()
    match = re.search(r"^clap_score:\s*([-+0-9.eE]+)$", text, re.MULTILINE)
    if not match:
        issues.append("metrics.txt lacks clap_score")
    else:
        clap = float(match.group(1))
        if not (clap == clap):  # NaN
            issues.append("clap_score is NaN")

# NoQ training must be evaluated with --no_q (q=10 null token), not q=9.
eval_log = Path(f"/home/kojiek/logs/{prefix}_stage2_100_musiccaps_eval.log")
if eval_log.is_file():
    el = eval_log.read_text(errors="replace")
    if "'no_q': True" not in el and '"no_q": True' not in el:
        issues.append("eval log missing no_q=True (NoQ models must use --no_q)")
else:
    issues.append(f"missing eval log: {eval_log}")

payload = {
    "status": "passed" if not issues else "failed",
    "audited_at": datetime.now(timezone.utc).isoformat(),
    "prefix": prefix,
    "clap_score": clap,
    "note": "wiring gate only; CLAP is not a convergence metric",
    "issues": issues,
    "checks": {
        "s1_model": "fluxaudio_s",
        "s2_model": "meanaudio_s",
        "use_q_conditioning": False,
        "use_text_attention_mask": False,
        "eval_no_q": True,
    },
}
tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, sentinel)

if issues:
    print(json.dumps(payload, indent=2))
    raise SystemExit(f"[FAIL] NoQ medium gate failed: {issues}")

print("[OK] catalog-matched NoQ medium gate completed")
print("[OK] S1=100, S2 total=200, EMA present, eval=64, use_q=false, NoMask")
print(f"[INFO] short-run CLAP={clap:.4f} (wiring only; not convergence)")
print(f"[OK] sentinel: {sentinel}")
PY
