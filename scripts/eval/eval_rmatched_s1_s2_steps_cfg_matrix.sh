#!/usr/bin/env bash
# Full MusicCaps Stage-1/Stage-2 x steps x CFG evaluation matrix.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
PYTHON=/home/kojiek/venvs/dac/bin/python
PEAV_PYTHON=/home/kojiek/venvs/peav/bin/python
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
PEAV=/home/kojiek/research/meanaudio_eval/peav_eval.py
TSV="$DATA/musiccaps_test.tsv"
S1="$ROOT/exps/phase8_qwen_caption10s_multisent_noq_full_stage1_400000/phase8_qwen_caption10s_multisent_noq_full_stage1_400000_ema_final.pth"
S2="$ROOT/exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth"
TAG=rmatched_s1_s2_steps_cfg_matrix_seed14159265
OUT_ROOT="$ROOT/eval_output/$TAG"
METRIC_ROOT="$ROOT/eval_output/metrics"
REPORT="$LOG_ROOT/${TAG}_REPORT.json"
EXPECTED=5521
SEED=42
PREFLIGHT_ONLY=false

if [ "${1:-}" = "--preflight-only" ]; then
    PREFLIGHT_ONLY=true
elif [ "$#" -ne 0 ]; then
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
fi

for path in "$TSV" "$S1" "$S2" "$PYTHON" "$PEAV_PYTHON" "$EVALUATOR" "$PEAV"; do
    [ -f "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

"$PYTHON" - "$TSV" "$EXPECTED" "$S1" "$S2" <<'PY'
import csv, sys
from pathlib import Path

tsv, expected = Path(sys.argv[1]), int(sys.argv[2])
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != expected or any(not row.get("id") or not row.get("caption") for row in rows):
    raise SystemExit(f"[FAIL] invalid benchmark TSV: rows={len(rows)} expected={expected}")
for raw in sys.argv[3:]:
    path = Path(raw)
    if path.is_symlink() or path.stat().st_size < 100_000_000:
        raise SystemExit(f"[FAIL] invalid checkpoint: {path}")
print(f"[OK] inputs verified; rows={len(rows)}")
PY

FREE_BYTES=$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
if [ "$FREE_BYTES" -lt 161061273600 ]; then
    echo "[FAIL] storage gate: free_bytes=$FREE_BYTES required=161061273600" >&2
    exit 2
fi
echo "[OK] storage gate: free_bytes=$FREE_BYTES"
[ "$PREFLIGHT_ONLY" = false ] || { echo "[PREFLIGHT ONLY] no GPU work started"; exit 0; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CELLS=(
    "s2_mf25_cfg0p5|meanaudio_s|$S2|25|0.5|true"
    "s2_mf25_cfg4p5|meanaudio_s|$S2|25|4.5|true"
    "s2_mf1_cfg0p5|meanaudio_s|$S2|1|0.5|true"
    "s2_mf1_cfg4p5|meanaudio_s|$S2|1|4.5|true"
    "s1_fm25_cfg0p5|fluxaudio_s|$S1|25|0.5|false"
    "s1_fm25_cfg4p5|fluxaudio_s|$S1|25|4.5|false"
    "s1_fm1_cfg0p5|fluxaudio_s|$S1|1|0.5|false"
    "s1_fm1_cfg4p5|fluxaudio_s|$S1|1|4.5|false"
)

for spec in "${CELLS[@]}"; do
    IFS='|' read -r cell variant checkpoint steps cfg meanflow <<<"$spec"
    audio="$OUT_ROOT/$cell/audio"
    exp_name="${TAG}_${cell}"
    metrics="$METRIC_ROOT/$exp_name/metrics.txt"
    peav="$OUT_ROOT/$cell/peav/peav_metrics.json"
    log="$LOG_ROOT/${TAG}_${cell}.log"

    if "$PYTHON" - "$metrics" "$peav" "$EXPECTED" <<'PY' 2>/dev/null
import json, math, sys
from pathlib import Path
metrics, peav, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
values = {}
for line in metrics.read_text().splitlines():
    if ":" in line:
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in required:
            values[key] = float(raw)
p = json.loads(peav.read_text())
assert set(values) == required and all(math.isfinite(v) for v in values.values())
assert p["n_pairs"] == expected
assert all(math.isfinite(float(p[k])) for k in ("peav_score_mean", "t2a_R@10", "a2t_R@10"))
PY
    then
        echo "[SKIP] validated cell: $cell"
        continue
    fi

    mkdir -p "$audio" "$(dirname "$peav")"
    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    if [ "$count" -ne "$EXPECTED" ]; then
        args=(eval.py --variant "$variant" --model_path "$checkpoint" --output "$audio" --tsv "$TSV"
              --num_steps "$steps" --cfg_strength "$cfg" --encoder_name t5_clap --text_c_dim 512
              --seed "$SEED" --no_q --no_text_attention_mask --full_precision)
        [ "$meanflow" = false ] || args+=(--use_meanflow)
        "$PYTHON" "${args[@]}" 2>&1 | tee "$log"
    fi

    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    [ "$count" -eq "$EXPECTED" ] || { echo "[FAIL] $cell clips=$count/$EXPECTED" >&2; exit 2; }
    "$PYTHON" "$EVALUATOR" --gen_dir "$audio" --tsv "$TSV" --exp_name "$exp_name" --num_samples "$EXPECTED" 2>&1 | tee -a "$log"
    "$PEAV_PYTHON" "$PEAV" --gen_dir "$audio" --tsv "$TSV" --out "$peav" --batch_size 8 2>&1 | tee -a "$log"
done

"$PYTHON" - "$REPORT" "$OUT_ROOT" "$METRIC_ROOT" "$TAG" "$S1" "$S2" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, out_root, metric_root = map(Path, sys.argv[1:4])
tag, s1, s2 = sys.argv[4:]
cells = [
    ("s1_fm1_cfg0p5", 1, 1, 0.5), ("s1_fm1_cfg4p5", 1, 1, 4.5),
    ("s1_fm25_cfg0p5", 1, 25, 0.5), ("s1_fm25_cfg4p5", 1, 25, 4.5),
    ("s2_mf1_cfg0p5", 2, 1, 0.5), ("s2_mf1_cfg4p5", 2, 1, 4.5),
    ("s2_mf25_cfg0p5", 2, 25, 0.5), ("s2_mf25_cfg4p5", 2, 25, 4.5),
]
results = {}
for cell, stage, steps, cfg in cells:
    metrics = metric_root / f"{tag}_{cell}" / "metrics.txt"
    values = {}
    for line in metrics.read_text().splitlines():
        if ":" in line:
            key, raw = (part.strip() for part in line.split(":", 1))
            if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                values[key] = float(raw)
    peav_path = out_root / cell / "peav" / "peav_metrics.json"
    peav = json.loads(peav_path.read_text())
    if len(values) != 5 or not all(math.isfinite(v) for v in values.values()) or peav.get("n_pairs") != 5521:
        raise SystemExit(f"[FAIL] incomplete result: {cell}")
    results[cell] = {"stage": stage, "steps": steps, "cfg": cfg, "metrics": values, "peav": peav}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "status": "passed",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment_id": "rmatched-s1-s2-steps-cfg-matrix",
    "protocol": "MusicCaps 5521; generation seed 42; NoQ NoMask full precision",
    "checkpoints": {"stage1": {"path": s1, "sha256": sha(s1)}, "stage2": {"path": s2, "sha256": sha(s2)}},
    "results": results,
}
report.parent.mkdir(parents=True, exist_ok=True)
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report)
print(json.dumps(results, indent=2, sort_keys=True))
print(f"[COMPLETE] report={report}")
PY
