#!/usr/bin/env bash
# Evaluate one R-Matched full checkpoint on the registered dual benchmark.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
PEAV=/home/kojiek/research/meanaudio_eval/peav_eval.py
PEAV_PYTHON=/home/kojiek/venvs/peav/bin/python
PYTHON=/home/kojiek/venvs/dac/bin/python

CHECKPOINT=""
LABEL=""
REPORT=""
PREFLIGHT_ONLY=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --label) LABEL="$2"; shift ;;
        --report) REPORT="$2"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=true ;;
        *) echo "[FAIL] unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$CHECKPOINT" ] && [ -n "$LABEL" ] && [ -n "$REPORT" ] || {
    echo "usage: $0 --checkpoint ABS --label ID --report ABS [--preflight-only]" >&2
    exit 2
}
case "$LABEL" in *[!A-Za-z0-9._-]*|'') echo "[FAIL] unsafe label" >&2; exit 2;; esac
case "$REPORT" in /*) ;; *) echo "[FAIL] report must be absolute" >&2; exit 2;; esac

MC_TSV="$DATA/musiccaps_test.tsv"
JM_TSV="$DATA/phase4_test_seed42_2048.tsv"
OUT_ROOT="$ROOT/eval_output/rmatched_validation/$LABEL"
METRIC_ROOT="$ROOT/eval_output/metrics"

for path in "$CHECKPOINT" "$MC_TSV" "$JM_TSV" "$EVALUATOR" "$PEAV" "$PEAV_PYTHON" "$PYTHON"; do
    [ -e "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done

"$PYTHON" - "$MC_TSV" 5521 "$JM_TSV" 2048 "$CHECKPOINT" <<'PY'
import csv, sys
from pathlib import Path
for raw, expected in ((sys.argv[1], int(sys.argv[2])), (sys.argv[3], int(sys.argv[4]))):
    path = Path(raw)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected or any(not row.get("id") or not row.get("caption") for row in rows):
        raise SystemExit(f"[FAIL] invalid benchmark TSV: {path}")
checkpoint = Path(sys.argv[5])
if not checkpoint.is_file() or checkpoint.is_symlink() or checkpoint.stat().st_size < 100_000_000:
    raise SystemExit("[FAIL] invalid checkpoint")
print("[OK] dual-benchmark inputs verified")
PY
[ "$PREFLIGHT_ONLY" = false ] || { echo "[PREFLIGHT ONLY] no GPU work started"; exit 0; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"

run_one() {
    local benchmark="$1"
    local tsv="$2"
    local expected="$3"
    local audio="$OUT_ROOT/$benchmark/audio"
    local exp_name="rmatched_${LABEL}_${benchmark}"
    local metrics="$METRIC_ROOT/$exp_name/metrics.txt"
    mkdir -p "$audio"
    local count
    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    if [ "$count" -ne "$expected" ]; then
        python eval.py \
            --variant meanaudio_s --model_path "$CHECKPOINT" \
            --output "$audio" --tsv "$tsv" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --no_q --no_text_attention_mask \
            --full_precision
    fi
    count=$(find "$audio" -maxdepth 1 -type f -name '*.flac' -links 1 | wc -l)
    [ "$count" -eq "$expected" ] || { echo "[FAIL] $benchmark clips=$count/$expected" >&2; exit 2; }
    find "$audio" -mindepth 1 -maxdepth 1 \( -type l -o ! -type f -o -links +1 \) -print -quit | \
        grep -q . && { echo "[FAIL] unsafe/foreign audio entry in $audio" >&2; exit 2; } || true

    python "$EVALUATOR" --gen_dir "$audio" --tsv "$tsv" \
        --exp_name "$exp_name" --num_samples "$expected"
    [ -f "$metrics" ] || { echo "[FAIL] metrics missing: $metrics" >&2; exit 2; }
    mkdir -p "$OUT_ROOT/$benchmark/peav"
    "$PEAV_PYTHON" "$PEAV" --gen_dir "$audio" --tsv "$tsv" \
        --out "$OUT_ROOT/$benchmark/peav/peav_metrics.json" --batch_size 8
}

run_one musiccaps "$MC_TSV" 5521
run_one jamendo_seed42_2048 "$JM_TSV" 2048

"$PYTHON" - "$LABEL" "$CHECKPOINT" "$REPORT" "$OUT_ROOT" "$METRIC_ROOT" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

label, checkpoint_raw, report_raw, out_raw, metrics_raw = sys.argv[1:]
checkpoint, report, out_root, metric_root = map(Path, (checkpoint_raw, report_raw, out_raw, metrics_raw))
specs = {"musiccaps": 5521, "jamendo_seed42_2048": 2048}
result = {}
for benchmark, expected in specs.items():
    metric_file = metric_root / f"rmatched_{label}_{benchmark}" / "metrics.txt"
    values = {}
    for line in metric_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            try: values[key] = float(raw)
            except ValueError: pass
    peav_file = out_root / benchmark / "peav" / "peav_metrics.json"
    peav = json.loads(peav_file.read_text())
    audio = out_root / benchmark / "audio"
    count = sum(1 for p in audio.iterdir() if p.is_file() and p.suffix == ".flac")
    if count != expected or len(values) != 5 or not all(math.isfinite(v) for v in values.values()):
        raise SystemExit(f"[FAIL] incomplete/nonfinite {benchmark} result")
    if peav.get("n_pairs") != expected or not all(math.isfinite(peav.get(k, float("nan"))) for k in ("peav_score_mean", "t2a_R@10")):
        raise SystemExit(f"[FAIL] incomplete/nonfinite PE-AV {benchmark} result")
    result[benchmark] = {"n": count, "metrics": values, "peav": peav,
                         "metrics_path": str(metric_file), "peav_path": str(peav_file)}
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""): h.update(block)
    return h.hexdigest()
payload = {"schema_version": 1, "status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
           "label": label, "protocol": "MF1 CFG0.5 NoQ NoMask full precision; generation seed 42",
           "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "benchmarks": result}
report.parent.mkdir(parents=True, exist_ok=True)
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

