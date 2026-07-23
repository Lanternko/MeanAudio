#!/usr/bin/env bash
# Durable matched end-to-end quarter chain:
#   No-Q   S1 100k -> S2 50k
#   Half-Q S1 100k -> S2 50k
# followed by full MusicCaps Stage-1 and global metrics.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_halfq_quarter_e2e"
LOG="$LOG_ROOT/phase8_halfq_quarter_e2e_sequence.log"
LOCK="$STATE/sequence.lock"
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
FINAL_REPORT="$LOG_ROOT/phase8_halfq_quarter_e2e_FINAL_METRICS.json"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"

NOQ_S1=phase8_quarter_e2e_noq_stage1_100000
NOQ_S2=phase8_quarter_e2e_noq_stage2_50000
HALFQ_S1=phase8_quarter_e2e_halfq_stage1_100000
HALFQ_S2=phase8_quarter_e2e_halfq_stage2_50000

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$STATE" "$LOG_ROOT"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[FAIL] end-to-end quarter sequence already running" >&2
    exit 3
fi

log() {
    echo "[HALFQ-E2E] $(date --iso-8601=seconds) $*" | tee -a "$LOG"
}

gpu_blockers() {
    python - <<'PY'
import subprocess
from pathlib import Path

result = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
blockers = []
for line in result.stdout.splitlines():
    if not line.strip():
        continue
    pid_raw, memory_raw = [part.strip() for part in line.split(",", 1)]
    pid = int(pid_raw)
    try:
        command = (
            (Path("/proc") / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        continue
    if "Irodori-TTS" in command:
        continue
    blockers.append((pid, int(memory_raw), command))
for pid, memory, command in blockers:
    print(f"{pid}\t{memory} MiB\t{command}")
raise SystemExit(0 if not blockers else 1)
PY
}

run_arm() {
    local arm="$1"
    local audit="$LOG_ROOT/phase8_quarter_e2e_${arm}_FINAL_TRAIN_AUDIT.json"
    if [ -f "$audit" ]; then
        python - "$audit" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"[FAIL] prior arm audit is not passed: {payload}")
print(f"[SKIP] completed arm audit: {sys.argv[1]}")
PY
        return
    fi
    log "training arm=$arm mode=$RUN_MODE"
    ARM="$arm" PREFLIGHT_ONLY=false EXPERIMENT_RUN_MODE="$RUN_MODE" \
        bash scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh
    [ -f "$audit" ] || { echo "[FAIL] missing arm audit: $audit" >&2; exit 2; }
}

run_eval() {
    local label="$1"
    local variant="$2"
    local model="$3"
    shift 3
    local out="$ROOT/eval_output/$label"
    local metrics="$ROOT/eval_output/metrics/$label/metrics.txt"
    local eval_log="$LOG_ROOT/${label}_eval.log"
    local protocol_args=()

    if [ -f "$metrics" ]; then
        log "skip complete metric label=$label"
        return
    fi
    if [ "$variant" = fluxaudio_s ]; then
        protocol_args=(--num_steps 25 --cfg_strength 4.5)
    else
        protocol_args=(--use_meanflow --num_steps 1 --cfg_strength 0.5)
    fi
    mkdir -p "$out/audio"
    log "evaluation starts label=$label"
    python eval.py --variant "$variant" --model_path "$model" \
        --output "$out/audio" --tsv "$MUSICCAPS" \
        --encoder_name t5_clap --text_c_dim 512 --no_text_attention_mask \
        --full_precision "${protocol_args[@]}" "$@" 2>&1 | tee "$eval_log"
    audio_n=$(find "$out/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
    if [ "$audio_n" -ne 5521 ]; then
        echo "[FAIL] $label generated $audio_n/5521 audio files" >&2
        exit 2
    fi
    python "$EVALUATOR" --gen_dir "$out/audio" --tsv "$MUSICCAPS" \
        --exp_name "$label" --num_samples 5521 2>&1 | tee -a "$eval_log"
    [ -f "$metrics" ] || {
        echo "[FAIL] evaluator did not create $metrics" >&2
        exit 2
    }
    log "evaluation complete label=$label"
}

log "CPU preflight: No-Q arm"
ARM=noq PREFLIGHT_ONLY=true \
    bash scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh
log "CPU preflight: Half-Q arm"
ARM=halfq PREFLIGHT_ONLY=true \
    bash scripts/training_pipelines/train_pipeline_phase8_halfq_quarter.sh

while ! blockers=$(gpu_blockers); do
    log "waiting for GPU; blockers: ${blockers//$'\n'/; }"
    sleep "$POLL_SECONDS"
done

log "GPU clear; matched end-to-end training starts"
run_arm noq
run_arm halfq

NOQ_S1_EMA="$ROOT/exps/$NOQ_S1/${NOQ_S1}_ema_final.pth"
NOQ_S2_EMA="$ROOT/exps/$NOQ_S2/${NOQ_S2}_ema_final.pth"
HALFQ_S1_EMA="$ROOT/exps/$HALFQ_S1/${HALFQ_S1}_ema_final.pth"
HALFQ_S2_EMA="$ROOT/exps/$HALFQ_S2/${HALFQ_S2}_ema_final.pth"
for model in "$NOQ_S1_EMA" "$NOQ_S2_EMA" "$HALFQ_S1_EMA" "$HALFQ_S2_EMA"; do
    [ -f "$model" ] || { echo "[FAIL] missing evaluation model: $model" >&2; exit 2; }
done

# Stage-1 metrics use the native 25-step Flow-Matching protocol.
run_eval "${NOQ_S1}_musiccaps_fm25_noq" fluxaudio_s "$NOQ_S1_EMA" --no_q
run_eval "${HALFQ_S1}_musiccaps_fm25_q9" fluxaudio_s "$HALFQ_S1_EMA" \
    --quality_level 9
run_eval "${HALFQ_S1}_musiccaps_fm25_q0" fluxaudio_s "$HALFQ_S1_EMA" \
    --quality_level 0

# Global metrics use the one-step MeanFlow protocol.
run_eval "${NOQ_S2}_musiccaps_mf1_noq" meanaudio_s "$NOQ_S2_EMA" --no_q
run_eval "${HALFQ_S2}_musiccaps_mf1_q9" meanaudio_s "$HALFQ_S2_EMA" \
    --quality_level 9
run_eval "${HALFQ_S2}_musiccaps_mf1_q0" meanaudio_s "$HALFQ_S2_EMA" \
    --quality_level 0

python - "$ROOT" "$FINAL_REPORT" "$NOQ_S1" "$NOQ_S2" "$HALFQ_S1" \
    "$HALFQ_S2" <<'PY'
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, out, noq_s1, noq_s2, halfq_s1, halfq_s2 = sys.argv[1:]
root = Path(root)
out = Path(out)
metrics_root = root / "eval_output" / "metrics"

def read(label: str) -> dict[str, float]:
    path = metrics_root / label / "metrics.txt"
    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            values[key] = float(value)
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if set(values) != required or not all(math.isfinite(x) for x in values.values()):
        raise SystemExit(f"[FAIL] incomplete metrics at {path}: {values}")
    return values

s1_noq = read(f"{noq_s1}_musiccaps_fm25_noq")
s1_q9 = read(f"{halfq_s1}_musiccaps_fm25_q9")
s1_q0 = read(f"{halfq_s1}_musiccaps_fm25_q0")
s2_noq = read(f"{noq_s2}_musiccaps_mf1_noq")
s2_q9 = read(f"{halfq_s2}_musiccaps_mf1_q9")
s2_q0 = read(f"{halfq_s2}_musiccaps_mf1_q0")
payload = {
    "schema_version": 2,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_halfq_quarter_e2e",
    "scale": {
        "stage1": "100k / 400k = 1/4",
        "stage2": "50k / 200k = 1/4",
        "both_arms_from_scratch": True,
    },
    "stage1": {
        "protocol": "MusicCaps 5521; FluxAudio; FM25; CFG 4.5",
        "quarter_noq_baseline": s1_noq,
        "halfq_q9": s1_q9,
        "halfq_q0": s1_q0,
        "halfq_q9_minus_baseline_clap": (
            s1_q9["clap_score"] - s1_noq["clap_score"]
        ),
        "halfq_q0_minus_baseline_clap": (
            s1_q0["clap_score"] - s1_noq["clap_score"]
        ),
        "halfq_q9_minus_q0_clap": s1_q9["clap_score"] - s1_q0["clap_score"],
    },
    "global": {
        "protocol": "MusicCaps 5521; MeanAudio; MeanFlow1; CFG 0.5",
        "quarter_noq_baseline": s2_noq,
        "halfq_q9": s2_q9,
        "halfq_q0": s2_q0,
        "halfq_q9_minus_baseline_clap": (
            s2_q9["clap_score"] - s2_noq["clap_score"]
        ),
        "halfq_q0_minus_baseline_clap": (
            s2_q0["clap_score"] - s2_noq["clap_score"]
        ),
        "halfq_q9_minus_q0_clap": s2_q9["clap_score"] - s2_q0["clap_score"],
    },
    "interpretation": {
        "primary_halfq_endpoint": "q9",
        "q0_is_binary_axis_diagnostic": True,
        "stage1_and_global_protocols_are_reported_separately": True,
        "no_checkpoint_cherrypick": True,
    },
}
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

log "complete report=$FINAL_REPORT"
