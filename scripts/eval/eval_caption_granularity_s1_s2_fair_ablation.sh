#!/usr/bin/env bash
# Three missing cells for the four-way FM/MF25 CFG4.5 caption ablation.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
PYTHON=/home/kojiek/venvs/dac/bin/python
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
TSV="$DATA/musiccaps_test.tsv"
EXPECTED=5521
SEED=42
TAG=caption_granularity_s1_s2_fair_ablation
OUT_ROOT="$ROOT/eval_output/$TAG"
METRIC_ROOT="$ROOT/eval_output/metrics"
REPORT="$LOG_ROOT/${TAG}_REPORT.json"
BASELINE_EVIDENCE="$LOG_ROOT/phase8_qwen_official_noq_full_STAGE1_METRICS.json"
CAP1_S2="$ROOT/exps/phase8_qwen_caption10s_noq_full_stage2_200000/phase8_qwen_caption10s_noq_full_stage2_200000_ema_final.pth"
CAP2_S1="$ROOT/exps/phase8_qwen_caption10s_multisent_noq_full_stage1_400000/phase8_qwen_caption10s_multisent_noq_full_stage1_400000_ema_final.pth"
CAP2_S2="$ROOT/exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
PREFLIGHT_ONLY=false

if [ "${1:-}" = "--preflight-only" ]; then
    PREFLIGHT_ONLY=true
elif [ "$#" -ne 0 ]; then
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
fi

for path in "$TSV" "$BASELINE_EVIDENCE" "$CAP1_S2" "$CAP2_S1" "$CAP2_S2" "$PYTHON" "$EVALUATOR"; do
    [ -f "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

"$PYTHON" - "$TSV" "$EXPECTED" "$BASELINE_EVIDENCE" "$CAP1_S2" "$CAP2_S1" "$CAP2_S2" <<'PY'
import csv, json, sys
from pathlib import Path

tsv, expected = Path(sys.argv[1]), int(sys.argv[2])
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != expected or any(not row.get("id") or not row.get("caption") for row in rows):
    raise SystemExit(f"[FAIL] invalid MusicCaps TSV: {len(rows)}/{expected}")
existing = json.loads(Path(sys.argv[3]).read_text())
if existing.get("audio_count") != expected or existing.get("protocol") != "MusicCaps 5521; FluxAudio FM25; CFG4.5; NoQ; NoMask":
    raise SystemExit("[FAIL] existing Stage 1 cell has a mismatched protocol")
for raw in sys.argv[4:]:
    checkpoint = Path(raw)
    if checkpoint.is_symlink() or checkpoint.stat().st_size < 100_000_000:
        raise SystemExit(f"[FAIL] invalid checkpoint: {checkpoint}")
print("[OK] four-way ablation inputs and existing cell verified")
PY

FREE_BYTES=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
[ "$FREE_BYTES" -ge 161061273600 ] || { echo "[FAIL] root storage below 150 GiB" >&2; exit 2; }
echo "[OK] storage gate: free_bytes=$FREE_BYTES"
[ "$PREFLIGHT_ONLY" = false ] || { echo "[PREFLIGHT ONLY] no GPU work started"; exit 0; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CELLS=(
    "caption2p0_s2_mf25_cfg4p5|meanaudio_s|$CAP2_S2|true"
    "caption1p0_s2_mf25_cfg4p5|meanaudio_s|$CAP1_S2|true"
    "caption2p0_s1_fm25_cfg4p5|fluxaudio_s|$CAP2_S1|false"
)

for spec in "${CELLS[@]}"; do
    IFS='|' read -r cell variant checkpoint meanflow <<<"$spec"
    audio="$OUT_ROOT/$cell/audio"
    exp_name="${TAG}_${cell}"
    metrics="$METRIC_ROOT/$exp_name/metrics.txt"
    log="$LOG_ROOT/${TAG}_${cell}.log"
    if [ -f "$metrics" ]; then
        echo "[SKIP] existing metrics: $cell"
        continue
    fi
    mkdir -p "$audio"
    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    if [ "$count" -ne "$EXPECTED" ]; then
        args=(eval.py --variant "$variant" --model_path "$checkpoint" --output "$audio" --tsv "$TSV"
              --num_steps 25 --cfg_strength 4.5 --encoder_name t5_clap --text_c_dim 512
              --seed "$SEED" --no_q --no_text_attention_mask --full_precision)
        [ "$meanflow" = false ] || args+=(--use_meanflow)
        "$PYTHON" "${args[@]}" 2>&1 | tee "$log"
    fi
    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    [ "$count" -eq "$EXPECTED" ] || { echo "[FAIL] $cell clips=$count/$EXPECTED" >&2; exit 2; }
    "$PYTHON" "$EVALUATOR" --gen_dir "$audio" --tsv "$TSV" --exp_name "$exp_name" --num_samples "$EXPECTED" 2>&1 | tee -a "$log"
    [ -f "$metrics" ] || { echo "[FAIL] missing metrics: $metrics" >&2; exit 2; }
done

"$PYTHON" - "$REPORT" "$BASELINE_EVIDENCE" "$METRIC_ROOT" "$TAG" "$CAP1_S2" "$CAP2_S1" "$CAP2_S2" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, existing_path, metric_root = map(Path, sys.argv[1:4])
tag, cap1_s2, cap2_s1, cap2_s2 = sys.argv[4:]
existing = json.loads(existing_path.read_text())
results = {"baseline_s1_fm25_cfg4p5": {
    "source": "existing_valid",
    "source_path_note": "Legacy evidence filename retained for provenance; it is not the scientific label.",
    "metrics": existing["metrics"],
}}
for cell in ("caption2p0_s2_mf25_cfg4p5", "caption1p0_s2_mf25_cfg4p5", "caption2p0_s1_fm25_cfg4p5"):
    values = {}
    path = metric_root / f"{tag}_{cell}" / "metrics.txt"
    for line in path.read_text().splitlines():
        if ":" in line:
            key, raw = (part.strip() for part in line.split(":", 1))
            if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                values[key] = float(raw)
    if len(values) != 5 or not all(math.isfinite(value) for value in values.values()):
        raise SystemExit(f"[FAIL] incomplete metrics: {cell}")
    results[cell] = {"source": "new", "metrics": values}

def sha(raw):
    digest = hashlib.sha256()
    with Path(raw).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "status": "passed",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": "caption-granularity-s1-s2-fair-ablation",
    "protocol": "MusicCaps 5521; generation seed 42; 25 steps; CFG4.5; NoQ; NoMask; full precision",
    "checkpoints": {
        "caption1p0_stage2": {"path": cap1_s2, "sha256": sha(cap1_s2)},
        "caption2p0_stage1": {"path": cap2_s1, "sha256": sha(cap2_s1)},
        "caption2p0_stage2": {"path": cap2_s2, "sha256": sha(cap2_s2)},
    },
    "results": results,
}
report.parent.mkdir(parents=True, exist_ok=True)
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report)
print(json.dumps(results, indent=2, sort_keys=True))
print(f"[COMPLETE] report={report}")
PY
