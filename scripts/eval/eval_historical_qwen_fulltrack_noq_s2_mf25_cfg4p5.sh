#!/usr/bin/env bash
# Historical full-track/single-caption Qwen NoQ Stage-2 fair 25-step evaluation.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
PYTHON=/home/kojiek/venvs/dac/bin/python
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
CHECKPOINT="$ROOT/exps/phase8_qwen_official_noq_full_stage2_200000/phase8_qwen_official_noq_full_stage2_200000_ema_final.pth"
CHECKPOINT_SHA256=2519e83638c431aff006bf7690023ab53f17ff86b8894200f83d23c0ddceeeca
TRAIN_CONTRACT="$LOG_ROOT/phase8_qwen_official_noq_full_official_contract.json"
TRAIN_REPORT="$LOG_ROOT/phase8_qwen_official_noq_full_FINAL_METRICS.json"
TSV="$DATA/musiccaps_test.tsv"
EXPECTED=5521
LABEL=phase8_qwen_official_noq_full_stage2_200000_musiccaps_mf25_cfg4p5_noq
OUT="$ROOT/eval_output/$LABEL"
METRIC_ROOT="$ROOT/eval_output/metrics"
METRICS="$METRIC_ROOT/$LABEL/metrics.txt"
REPORT="$LOG_ROOT/${LABEL}_REPORT.json"
EVAL_LOG="$LOG_ROOT/${LABEL}_eval.log"
PREFLIGHT_ONLY=false

if [ "${1:-}" = "--preflight-only" ]; then
    PREFLIGHT_ONLY=true
elif [ "$#" -ne 0 ]; then
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
fi

for path in "$CHECKPOINT" "$TRAIN_CONTRACT" "$TRAIN_REPORT" "$TSV" "$PYTHON" "$EVALUATOR"; do
    [ -f "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

"$PYTHON" - "$CHECKPOINT" "$CHECKPOINT_SHA256" "$TRAIN_CONTRACT" "$TRAIN_REPORT" "$TSV" "$EXPECTED" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path

checkpoint, expected_sha, contract_path, report_path, tsv, expected = (
    Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
    Path(sys.argv[5]), int(sys.argv[6])
)
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != expected_sha:
    raise SystemExit(f"[FAIL] checkpoint drift: {digest}")
contract = json.loads(contract_path.read_text())
prior = json.loads(report_path.read_text())
if not (
    contract.get("caption_semantics") == "official Qwen caption mapped to exact Jamendo track"
    and contract.get("stage1_use_q_conditioning") is False
    and contract.get("stage2_use_q_conditioning") is False
    and contract.get("stage2_final_iteration") == 600000
    and prior.get("status") == "passed"
    and prior.get("model", {}).get("sha256") == digest
):
    raise SystemExit("[FAIL] historical full-track Qwen identity/provenance mismatch")
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != expected or any(not row.get("id") or not row.get("caption") for row in rows):
    raise SystemExit(f"[FAIL] MusicCaps rows={len(rows)}/{expected}")
print(f"[OK] historical full-track Qwen NoQ S2 checkpoint bound; sha256={digest}")
PY

FREE_BYTES=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
[ "$FREE_BYTES" -ge 161061273600 ] || { echo "[FAIL] root storage below 150 GiB" >&2; exit 2; }
echo "[OK] storage gate: free_bytes=$FREE_BYTES"
[ "$PREFLIGHT_ONLY" = false ] || { echo "[PREFLIGHT ONLY] no GPU work started"; exit 0; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT/audio" "$LOG_ROOT"

count=$(find "$OUT/audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
if [ "$count" -ne "$EXPECTED" ]; then
    "$PYTHON" eval.py --variant meanaudio_s --model_path "$CHECKPOINT" \
        --output "$OUT/audio" --tsv "$TSV" --use_meanflow \
        --num_steps 25 --cfg_strength 4.5 --no_q --no_text_attention_mask \
        --encoder_name t5_clap --text_c_dim 512 --seed 42 --full_precision \
        2>&1 | tee "$EVAL_LOG"
fi
count=$(find "$OUT/audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
[ "$count" -eq "$EXPECTED" ] || { echo "[FAIL] clips=$count/$EXPECTED" >&2; exit 2; }

if [ ! -f "$METRICS" ]; then
    "$PYTHON" "$EVALUATOR" --gen_dir "$OUT/audio" --tsv "$TSV" \
        --out_dir "$METRIC_ROOT" --exp_name "$LABEL" --num_samples "$EXPECTED" \
        2>&1 | tee -a "$EVAL_LOG"
fi
[ -f "$METRICS" ] || { echo "[FAIL] missing metrics: $METRICS" >&2; exit 2; }

"$PYTHON" - "$REPORT" "$CHECKPOINT" "$CHECKPOINT_SHA256" "$TRAIN_CONTRACT" "$TSV" "$METRICS" "$EXPECTED" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, checkpoint, expected_sha, contract, tsv, metrics = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
    Path(sys.argv[5]), Path(sys.argv[6])
)
expected = int(sys.argv[7])
values = {}
for line in metrics.read_text().splitlines():
    if ":" in line:
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(raw)
if len(values) != 5 or not all(math.isfinite(value) for value in values.values()):
    raise SystemExit("[FAIL] incomplete/non-finite metrics")
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
if sha(checkpoint) != expected_sha:
    raise SystemExit("[FAIL] checkpoint changed during evaluation")
payload = {
    "schema_version": 1,
    "status": "passed",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": "historical-qwen-fulltrack-noq-s2-mf25-cfg4p5",
    "caption_semantics": "one official Qwen caption per full Jamendo track, mapped to track segments",
    "protocol": "MusicCaps 5521; MeanFlow25; CFG4.5; NoQ; NoMask; seed42; full precision",
    "checkpoint": {"path": str(checkpoint), "sha256": expected_sha, "stage2_final_iteration": 600000},
    "training_contract": {"path": str(contract), "sha256": sha(contract)},
    "prompts": {"path": str(tsv), "sha256": sha(tsv), "rows": expected},
    "metrics": values,
    "metrics_path": str(metrics),
}
report.parent.mkdir(parents=True, exist_ok=True)
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
