#!/bin/bash
# Canonical MusicCaps 25-step eval for Caption 2.0 fair compare.
#
# Protocol (operator-authorized 2026-08-20):
#   MusicCaps 5521, MeanFlow, num_steps=25, cfg=0, seed=42,
#   t5_clap / text_c_dim=512, no_text_attention_mask, full_precision.
#   NoQ arms: pass --no_q
#   Q arms: pass --quality_level N (report at least q0 and q9)
#
# Usage:
#   eval_musiccaps_mf25.sh <label> <s2_ema.pth> [--no_q | --quality_level N]
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

LABEL="${1:?label}"
CKPT="${2:?s2 ema checkpoint}"
shift 2

[ -n "${CFG0_CONTRACT:-}" ] && [ -n "${CFG0_ARM:-}" ] || {
  echo "FAIL canonical CFG0 evaluation requires CFG0_CONTRACT and CFG0_ARM" >&2
  exit 2
}

case "$LABEL" in
  *"/"*|*".."*|*"_cfg4p5_"*) echo "FAIL unsafe or historical label: $LABEL" >&2; exit 2 ;;
esac
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*_mf25_cfg0_(noq|q[0-9]+)$ ]] || {
  echo "FAIL canonical label must end in _mf25_cfg0_noq or _mf25_cfg0_qN" >&2
  exit 2
}

COND_ARGS=()
if [ "$#" -eq 1 ] && [ "$1" = "--no_q" ]; then
  COND_MODE=no_q
  LABEL_COND=noq
  COND_ARGS=(--no_q)
elif [ "$#" -eq 2 ] && [ "$1" = "--quality_level" ] && [[ "$2" =~ ^[0-9]$ ]]; then
  COND_MODE="q$2"
  LABEL_COND="$COND_MODE"
  COND_ARGS=(--quality_level "$2")
else
  echo "FAIL conditioning must be exactly --no_q or --quality_level 0..9" >&2
  exit 2
fi
[[ "$LABEL" == *"_${LABEL_COND}" ]] || {
  echo "FAIL label conditioning does not match argv: $LABEL vs $COND_MODE" >&2
  exit 2
}

MEANAUDIO=/home/kojiek/MeanAudio
MUSICCAPS=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
RUNTIME_ROOT=/home/kojiek/cfg0_eval_runtime
OUTPUT_ROOT="$RUNTIME_ROOT/output"
METRICS_ROOT="$RUNTIME_ROOT/metrics"
REPORT_ROOT="$RUNTIME_ROOT/reports"
OUT="$OUTPUT_ROOT/$LABEL"
METRICS="$METRICS_ROOT/$LABEL/metrics.txt"
REPORT="$REPORT_ROOT/${LABEL}_REPORT.json"
AUDIO="$OUT/audio"
STRICT_VALIDATOR="$MEANAUDIO/scripts/eval/validate_caption2p0_cfg0_report.py"
PATH_VALIDATOR="$MEANAUDIO/scripts/eval/validate_cfg0_output_path.py"

[ -f "$CKPT" ] || { echo "FAIL missing ckpt $CKPT" >&2; exit 2; }
[ -f "$MUSICCAPS" ] || { echo "FAIL missing $MUSICCAPS" >&2; exit 2; }
CKPT_REAL=$(readlink -f "$CKPT")
# The registered exps root is a symlink farm: /home/kojiek/MeanAudio/exps ->
# /home/kojiek/exps_nvme, and individual experiments inside it may themselves be
# symlinks onto the HDD archive. Comparing the fully resolved checkpoint path
# against the *logical* root alone therefore rejected every real checkpoint, so
# compare against the resolved form of each registered storage root instead.
# NVMe and HDD backing store are equally canonical.
CKPT_OK=0
for _root in \
  /home/kojiek/MeanAudio/exps \
  /home/kojiek/exps_nvme \
  /mnt/HDD/kojiek/meanaudio_exps \
  /mnt/HDD/kojiek/MeanAudio_exps_hdd
do
  [ -d "$_root" ] || continue
  _root_real=$(readlink -f "$_root")
  case "$CKPT_REAL/" in
    "$_root_real"/*) CKPT_OK=1; break ;;
  esac
done
[ "$CKPT_OK" = 1 ] || {
  echo "FAIL checkpoint outside registered exps roots: $CKPT_REAL" >&2
  exit 2
}

# The canonical CFG0 runtime is intentionally on the current user's protected
# home ancestry. The historical HDD eval root is shared-host unsafe (0777).
install -d -m 700 "$RUNTIME_ROOT" "$OUTPUT_ROOT" "$METRICS_ROOT" "$REPORT_ROOT"
HARD_STOP_FREE_BYTES=$(python - "$CFG0_CONTRACT" <<'PY'
import json, sys
from pathlib import Path
contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int((contract.get("runtime_storage") or {}).get("hard_stop_free_bytes") or 0))
PY
)
if [ "$HARD_STOP_FREE_BYTES" -gt 0 ]; then
  FREE_BYTES=$(df -B1 --output=avail "$RUNTIME_ROOT" | tail -n 1 | tr -d ' ')
  if [ "$FREE_BYTES" -lt "$HARD_STOP_FREE_BYTES" ]; then
    echo "HOLD canonical CFG0 storage hard stop: free_bytes=$FREE_BYTES hard_stop=$HARD_STOP_FREE_BYTES" >&2
    # Returning success leaves the expensive phase unlaunched; the outer HARN
    # then classifies the missing canonical report as held rather than failed.
    exit 0
  fi
fi
python "$PATH_VALIDATOR" \
  --output-root "$OUTPUT_ROOT" --metrics-root "$METRICS_ROOT" --report-root "$REPORT_ROOT" \
  --out "$OUT" --audio "$AUDIO" --metrics-dir "$(dirname "$METRICS")" --report "$REPORT"

if [ -f "$REPORT" ]; then
  python - "$REPORT" "$METRICS" "$CKPT_REAL" "$MUSICCAPS" "$LABEL" "$COND_MODE" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

report, metrics, ckpt, tsv = map(Path, sys.argv[1:5])
label, conditioning = sys.argv[5:7]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p = json.loads(report.read_text())
expected = {
    "status": "passed", "label": label, "cfg_strength": 0,
    "num_steps": 25, "conditioning": conditioning,
    "checkpoint_sha256": sha(ckpt), "tsv_sha256": sha(tsv),
    "metrics_sha256": sha(metrics),
}
bad = {k: (p.get(k), v) for k, v in expected.items() if p.get(k) != v}
vals = p.get("metrics", {})
if bad or set(vals) != {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"} or not all(math.isfinite(x) for x in vals.values()):
    raise SystemExit(f"FAIL stale or drifted completed report: {bad}")
print(f"SKIP_VALIDATED {report}")
PY
  python "$STRICT_VALIDATOR" --contract "$CFG0_CONTRACT" --arm "$CFG0_ARM" --report "$REPORT"
  exit 0
fi
[ ! -f "$METRICS" ] || { echo "FAIL metrics exist without a bound final report: $METRICS" >&2; exit 2; }

cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

echo "EVAL_MF25 $LABEL $(date -u +%FT%TZ)"
if [ ! -f "$METRICS" ]; then
  install -d -m 700 "$OUT" "$AUDIO" "$(dirname "$METRICS")"
  python "$PATH_VALIDATOR" \
    --output-root "$OUTPUT_ROOT" --metrics-root "$METRICS_ROOT" --report-root "$REPORT_ROOT" \
    --out "$OUT" --audio "$AUDIO" --metrics-dir "$(dirname "$METRICS")" --report "$REPORT"
  python eval.py --variant meanaudio_s --model_path "$CKPT_REAL" \
    --output "$AUDIO" --tsv "$MUSICCAPS" --use_meanflow \
    --num_steps 25 --cfg_strength 0 \
    --encoder_name t5_clap --text_c_dim 512 --seed 42 \
    --no_text_attention_mask --full_precision \
    "${COND_ARGS[@]}" \
    2>&1 | tee "/home/kojiek/logs/${LABEL}_eval.log"
  python - "$MUSICCAPS" "$AUDIO" <<'PY'
import csv, sys
from pathlib import Path
import soundfile as sf

tsv, audio = map(Path, sys.argv[1:3])
with tsv.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = [row["id"] for row in rows]
actual = [path.stem for path in audio.glob("*.flac") if path.is_file() and not path.is_symlink()]
if len(expected) != 5521 or len(set(expected)) != 5521 or set(actual) != set(expected) or len(actual) != 5521:
    raise SystemExit(f"FAIL audio identity expected={len(expected)}/5521 actual={len(actual)} unique={len(set(actual))}")
for audio_id in expected:
    info = sf.info(audio / f"{audio_id}.flac")
    if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
        raise SystemExit(f"FAIL invalid audio {audio_id}: {info}")
print("AUDIO_VALIDATION passed rows=5521 unique=5521 mono=5521 sr16000=5521")
PY
  python "$EVALUATOR" --gen_dir "$AUDIO" --tsv "$MUSICCAPS" \
    --out_dir "$METRICS_ROOT" --exp_name "$LABEL" \
    --num_samples 5521 \
    2>&1 | tee -a "/home/kojiek/logs/${LABEL}_eval.log"
  # Same reason as the report: the evaluator writes metrics.txt with the default
  # umask, which the output-path validator would reject on any later run.
  chmod 600 "$METRICS"
else
  echo "SKIP_EVAL $METRICS"
fi
[ -f "$METRICS" ] || { echo "FAIL missing $METRICS" >&2; exit 2; }

python - "$REPORT" "$METRICS" "$CKPT_REAL" "$MUSICCAPS" "$LABEL" "$COND_MODE" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, metrics, ckpt, tsv = map(Path, sys.argv[1:5])
label, conditioning = sys.argv[5:7]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
vals = {}
for line in metrics.read_text().splitlines():
    if ":" not in line:
        continue
    key, raw = (part.strip() for part in line.split(":", 1))
    if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
        vals[key] = float(raw)
if len(vals) != 5 or not all(math.isfinite(x) for x in vals.values()):
    raise SystemExit(f"incomplete metrics {vals}")
payload = {
    "schema_version": 1,
    "status": "passed",
    "label": label,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 0; seed 42; NoMask; full precision",
    "cfg_strength": 0,
    "num_steps": 25,
    "seed": 42,
    "conditioning": conditioning,
    "checkpoint": str(ckpt),
    "checkpoint_sha256": sha(ckpt),
    "tsv": str(tsv),
    "tsv_sha256": sha(tsv),
    "audio_validation": {"rows": 5521, "unique_ids": 5521, "sample_rate": 16000, "channels": 1},
    "metrics": vals,
    "metrics_path": str(metrics),
    "metrics_sha256": sha(metrics),
}
report.parent.mkdir(parents=True, exist_ok=True)
tmp = report.with_name(f".{report.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
# validate_cfg0_output_path.py rejects any output component carrying group or
# other permission bits (shared host). The default umask would leave 0644 here
# and make every later re-validation of this same report fail.
os.chmod(tmp, 0o600)
os.replace(tmp, report)
print(json.dumps(payload, indent=2))
PY

python "$STRICT_VALIDATOR" --contract "$CFG0_CONTRACT" --arm "$CFG0_ARM" --report "$REPORT"

# Registered cleanup: only exact FLACs under this label's contained audio dir,
# and only after the bound final report exists.
[ -f "$REPORT" ] || { echo "FAIL refusing cleanup without report" >&2; exit 2; }
python - "$MUSICCAPS" "$AUDIO" <<'PY'
import csv, os, stat, sys
from pathlib import Path

tsv, audio = map(Path, sys.argv[1:3])
if audio.is_symlink() or not audio.is_dir():
    raise SystemExit(f"FAIL unsafe cleanup directory: {audio}")
with tsv.open(encoding="utf-8", newline="") as handle:
    expected = [row["id"] for row in csv.DictReader(handle, delimiter="\t")]
actual = {path.name for path in audio.iterdir()}
expected_names = {f"{audio_id}.flac" for audio_id in expected}
if actual != expected_names:
    raise SystemExit("FAIL cleanup manifest differs from exact expected-ID set")
for name in expected_names:
    path = audio / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise SystemExit(f"FAIL unsafe cleanup file: {path}")
for name in expected_names:
    (audio / name).unlink()
audio.rmdir()
print("CLEANUP_OK exact_flacs=5521")
PY
