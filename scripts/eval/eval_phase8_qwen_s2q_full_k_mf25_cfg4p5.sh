#!/usr/bin/env bash
# Evaluate one preregistered full-scale Q-conditioned Stage 2 K arm.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
PYTHON=/home/kojiek/venvs/dac/bin/python
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
TSV="$DATA/musiccaps_test.tsv"
EXPECTED=5521
SEED=42
K="${1:-${K:-}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

[ -n "$K" ] || { echo "[FAIL] pass K as argv[1] or environment K" >&2; exit 2; }
case "$K" in 2|3|5|10) ;; *) echo "[FAIL] unsupported K=$K" >&2; exit 2 ;; esac
case "$PREFLIGHT_ONLY" in true|false) ;; *) echo "[FAIL] PREFLIGHT_ONLY must be true or false" >&2; exit 2 ;; esac

BASE="${BASE_OVERRIDE:-phase8_qwen_s2q_from_noq_full_k${K}_balanced_stage2_200000}"
CHECKPOINT="$ROOT/exps/$BASE/${BASE}_ema_final.pth"
SOURCE_REPORT="${SOURCE_REPORT_OVERRIDE:-$LOG_ROOT/phase8_qwen_s2q_from_noq_full_k${K}_balanced_FINAL_METRICS.json}"
LABEL="${BASE}_musiccaps_n5521_mf25_cfg4p5_q9"
OUT="$ROOT/eval_output/$LABEL"
METRIC_ROOT="$ROOT/eval_output/metrics"
METRICS="$METRIC_ROOT/$LABEL/metrics.txt"
PROVENANCE="$OUT/provenance.json"
REPORT="$LOG_ROOT/${LABEL}_REPORT.json"
EVAL_LOG="$LOG_ROOT/${LABEL}_eval.log"

for path in "$TSV" "$CHECKPOINT" "$SOURCE_REPORT" "$PYTHON" "$EVALUATOR"; do
    [ -f "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

"$PYTHON" - "$TSV" "$EXPECTED" "$CHECKPOINT" "$SOURCE_REPORT" "$K" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

tsv, expected, checkpoint, source, k = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != expected or any(not row.get("id") or not row.get("caption") for row in rows):
    raise SystemExit(f"[FAIL] invalid MusicCaps TSV: {len(rows)}/{expected}")
payload = json.loads(source.read_text())
if payload.get("status") != "passed" or payload.get("k") != k or payload.get("strategy") != "balanced":
    raise SystemExit("[FAIL] source report identity/status mismatch")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if payload.get("model", {}).get("sha256") != digest:
    raise SystemExit("[FAIL] checkpoint hash differs from passed source report")
print(f"[OK] K={k} inputs bound; checkpoint_sha256={digest}")
PY

FREE_BYTES=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
[ "$FREE_BYTES" -ge 161061273600 ] || { echo "[FAIL] root storage below 150 GiB" >&2; exit 2; }
echo "[OK] storage gate: free_bytes=$FREE_BYTES"
[ "$PREFLIGHT_ONLY" = false ] || { echo "[PREFLIGHT ONLY] no GPU work started"; exit 0; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT/audio" "$LOG_ROOT"

"$PYTHON" - "$PROVENANCE" "$LABEL" "$CHECKPOINT" "$TSV" "$K" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

path, label, checkpoint, tsv, k = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
payload = {
    "schema_version": 1, "label": label, "variant": "meanaudio_s",
    "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
    "prompts": str(tsv), "prompts_sha256": sha(tsv), "rows": 5521,
    "stage": 2, "scale": "full", "k": k, "strategy": "balanced",
    "quality_level": 9, "solver": "MeanFlow", "num_steps": 25,
    "cfg_strength": 4.5, "seed": 42, "no_text_attention_mask": True,
    "full_precision": True,
}
if path.exists() and json.loads(path.read_text()) != payload:
    raise SystemExit("[FAIL] evaluation provenance drift")
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY

if [ ! -f "$METRICS" ]; then
    "$PYTHON" eval.py --variant meanaudio_s --model_path "$CHECKPOINT" \
        --output "$OUT/audio" --tsv "$TSV" --use_meanflow \
        --num_steps 25 --cfg_strength 4.5 --quality_level 9 \
        --encoder_name t5_clap --text_c_dim 512 --seed "$SEED" \
        --no_text_attention_mask --full_precision 2>&1 | tee "$EVAL_LOG"
fi

AUDIO_N=$(find "$OUT/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
[ "$AUDIO_N" -eq "$EXPECTED" ] || { echo "[FAIL] generated $AUDIO_N/$EXPECTED clips" >&2; exit 2; }
if [ ! -f "$METRICS" ]; then
    "$PYTHON" "$EVALUATOR" --gen_dir "$OUT/audio" --tsv "$TSV" \
        --out_dir "$METRIC_ROOT" --exp_name "$LABEL" --num_samples "$EXPECTED" 2>&1 | tee -a "$EVAL_LOG"
fi
[ -f "$METRICS" ] || { echo "[FAIL] missing metrics: $METRICS" >&2; exit 2; }

"$PYTHON" - "$REPORT" "$PROVENANCE" "$METRICS" <<'PY'
import json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, provenance, metrics = map(Path, sys.argv[1:])
values = {}
for line in metrics.read_text().splitlines():
    if ":" in line:
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(raw)
if len(values) != 5 or not all(math.isfinite(value) for value in values.values()):
    raise SystemExit("[FAIL] incomplete or non-finite metrics")
payload = {"schema_version": 1, "status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
           "provenance": json.loads(provenance.read_text()), "metrics": values, "metrics_path": str(metrics)}
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
