#!/usr/bin/env bash
# Continue Q-safe sequence after Real-Q completed: Shuffled-Q + paired bootstrap.
# Skips Real-Q if its final audit already passed.

set -euo pipefail
ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_qsafe_ft_monitor"
SEQ_LOG="$LOG_ROOT/phase8_qsafe_ft_sequence.log"
REAL_PREFIX=phase8_qsafe_realq_ft100k
SHUF_PREFIX=phase8_qsafe_shuffledq_ft100k
SHUF_TSV="$DATA/phase8_legacy_catalog_train_q_shuffled_seed424242.tsv"
REAL_AUDIT="$STATE/${REAL_PREFIX}_FINAL_AUDIT.json"

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
mkdir -p "$STATE"

{
    echo "[CONTINUE] Q-safe continuation starts $(date --iso-8601=seconds)"

    python - "$REAL_AUDIT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"[FAIL] missing Real-Q final audit: {p}")
audit = json.loads(p.read_text())
if audit.get("status") != "passed":
    raise SystemExit(f"[FAIL] Real-Q final audit not passed: {audit.get('status')} issues={audit.get('issues')}")
m = (audit.get("metrics") or {}).get("q9") or {}
print(f"[OK] Real-Q final audit passed; q9 CLAP={m.get('clap_score')}")
PY

    echo "[SEQUENCE] Q-safe Shuffled-Q starts $(date --iso-8601=seconds)"
    Q_MODE=shuffled EXP_PREFIX="$SHUF_PREFIX" TRAIN_TSV="$SHUF_TSV" \
        EXPERIMENT_RUN_MODE=fresh \
        bash scripts/training_pipelines/train_pipeline_phase8_qsafe_ft.sh

    BASE="$ROOT/eval_output/phase8_catalog_matched_noq_stage2_200000_musiccaps/audio"
    REAL="$ROOT/eval_output/${REAL_PREFIX}_stage2_ft100000_musiccaps_q9/audio"
    SHUF="$ROOT/eval_output/${SHUF_PREFIX}_stage2_ft100000_musiccaps_q9/audio"
    for d in "$BASE" "$REAL" "$SHUF"; do
        [ -d "$d" ] || { echo "[FAIL] missing audio dir $d" >&2; exit 2; }
    done

    python scripts/eval/paired_clap_bootstrap_phase8_qsafe.py \
        --tsv "$DATA/musiccaps_test.tsv" --baseline-dir "$BASE" \
        --real-dir "$REAL" --shuffled-dir "$SHUF" \
        --output "$STATE/PAIRED_CLAP_BOOTSTRAP.json" \
        --scores-csv "$STATE/PAIRED_CLAP_SCORES.csv"

    python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
root=Path('/home/kojiek/MeanAudio/eval_output/metrics')
state=Path('/home/kojiek/logs/phase8_qsafe_ft_monitor')
def read(prefix,q):
    p=root/f'{prefix}_stage2_ft100000_musiccaps_q{q}'/'metrics.txt'; out={}
    for line in p.read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            if k.strip() in {'clap_score','aes_CE','aes_CU','aes_PC','aes_PQ'}: out[k.strip()]=float(v)
    return out
real9=read('phase8_qsafe_realq_ft100k',9); shuf9=read('phase8_qsafe_shuffledq_ft100k',9)
paired=json.loads((state/'PAIRED_CLAP_BOOTSTRAP.json').read_text())
payload={
  'completed_at':datetime.now(timezone.utc).isoformat(), 'baseline_noq_clap':0.1888,
  'restoration_target_clap':0.1900,
  'real_q9':real9, 'real_q6':read('phase8_qsafe_realq_ft100k',6),
  'shuffled_q9':shuf9, 'shuffled_q6':read('phase8_qsafe_shuffledq_ft100k',6),
  'real_minus_noq_q9':real9['clap_score']-0.1888,
  'real_minus_shuffled_q9':real9['clap_score']-shuf9['clap_score'],
  'paired_clap':paired,
  'q_information_supported':paired['q_information_supported'],
  'net_q_gain_supported':paired['net_q_gain_supported'],
  'restored_clap_0p19':real9['clap_score']>=0.1900,
  'seed':14159265,
  'continuation_note':'Real-Q retained after seed-contract correction; Shuffled-Q run with explicit seed=14159265.',
}
payload['primary_objective_met']=payload['net_q_gain_supported'] or payload['restored_clap_0p19']
tmp=state/'FINAL_COMPARISON.json.tmp'; tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); tmp.replace(state/'FINAL_COMPARISON.json')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
    echo "[SEQUENCE] complete $(date --iso-8601=seconds)"
} 2>&1 | tee -a "$SEQ_LOG"
