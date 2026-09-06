#!/usr/bin/env bash
# Evaluate the completed official-Qwen full NoQ Stage-1 EMA only.
# Native FluxAudio protocol: FM25 / CFG4.5 / NoQ / NoMask.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
EXP=phase8_qwen_official_noq_full_stage1_400000_musiccaps_fm25_noq_nomask
WEIGHTS="$ROOT/exps/phase8_qwen_official_noq_full_stage1_400000/phase8_qwen_official_noq_full_stage1_400000_ema_final.pth"
TSV="$DATA/musiccaps_test.tsv"
OUT="$ROOT/eval_output/$EXP"
METRICS="$ROOT/eval_output/metrics/$EXP/metrics.txt"
LOG="$LOG_ROOT/${EXP}_eval.log"
REPORT="$LOG_ROOT/phase8_qwen_official_noq_full_STAGE1_METRICS.json"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
source "$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
phase8_nvidia_compat_apply || { echo "[FAIL] NVIDIA preflight: $PHASE8_NVIDIA_COMPAT_ERROR" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0

for path in "$WEIGHTS" "$TSV" "$EVALUATOR"; do
    [ -f "$path" ] || { echo "[FAIL] missing required file: $path" >&2; exit 2; }
done

if [ ! -f "$METRICS" ]; then
    mkdir -p "$OUT/audio" "$(dirname "$METRICS")"
    python eval.py --variant fluxaudio_s --model_path "$WEIGHTS" --output "$OUT/audio" --tsv "$TSV" \
        --num_steps 25 --cfg_strength 4.5 --encoder_name t5_clap --text_c_dim 512 \
        --no_q --no_text_attention_mask --full_precision 2>&1 | tee "$LOG"
    audio_n=$(find "$OUT/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
    [ "$audio_n" -eq 5521 ] || { echo "[FAIL] generated $audio_n/5521 audio files" | tee -a "$LOG" >&2; exit 2; }
    python "$EVALUATOR" --gen_dir "$OUT/audio" --tsv "$TSV" --exp_name "$EXP" --num_samples 5521 2>&1 | tee -a "$LOG"
fi

python - "$REPORT" "$METRICS" "$WEIGHTS" "$TSV" "$OUT/audio" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, metrics, weights, tsv, audio = map(Path, sys.argv[1:])
values = {}
for line in metrics.read_text().splitlines():
    if ': ' in line:
        key, value = line.split(': ', 1)
        if key in {'clap_score', 'aes_CE', 'aes_CU', 'aes_PC', 'aes_PQ'}:
            values[key] = float(value)
required = {'clap_score', 'aes_CE', 'aes_CU', 'aes_PC', 'aes_PQ'}
if set(values) != required or not all(math.isfinite(value) for value in values.values()):
    raise SystemExit('[FAIL] invalid or incomplete Stage-1 metrics')
audio_n = len(list(audio.glob('*.flac')))
if audio_n != 5521:
    raise SystemExit(f'[FAIL] Stage-1 audio count {audio_n}/5521')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
payload = {
    'schema_version': 1, 'completed_at': datetime.now(timezone.utc).isoformat(),
    'status': 'passed', 'experiment': 'phase8_qwen_official_noq_full',
    'stage': 1, 'model_variant': 'fluxaudio_s',
    'protocol': 'MusicCaps 5521; FluxAudio FM25; CFG4.5; NoQ; NoMask',
    'metrics': values, 'audio_count': audio_n,
    'model': {'path': str(weights), 'sha256': sha(weights)},
    'test_tsv': {'path': str(tsv), 'sha256': sha(tsv)},
    'metrics_path': str(metrics),
}
tmp = report.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n'); os.replace(tmp, report)
print(f"[COMPLETE] Stage-1 CLAP={values['clap_score']:.4f}; report={report}")
PY
