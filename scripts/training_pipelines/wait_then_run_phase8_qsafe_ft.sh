#!/usr/bin/env bash
# Durable queue: wait for the current random-init S2-Q sequence, then launch.

set -euo pipefail
ROOT=/home/kojiek/MeanAudio
OLD_STATE=/home/kojiek/logs/phase8_s2_q_ablation_monitor
LOG=/home/kojiek/logs/phase8_qsafe_ft_queue.log
cd "$ROOT"

echo "[QUEUE] waiting for current experiment $(date --iso-8601=seconds)" | tee -a "$LOG"
while true; do
    if [ -f "$OLD_STATE/FINAL_COMPARISON.json" ] && \
       python - "$OLD_STATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
files=[p/'phase8_catalog_matched_s2_realq_FINAL_AUDIT.json',p/'phase8_catalog_matched_s2_shuffledq_FINAL_AUDIT.json']
raise SystemExit(0 if all(x.is_file() and json.loads(x.read_text()).get('status')=='passed' for x in files) else 1)
PY
    then
        break
    fi
    sleep 60
done

# Do not overlap any lingering generation/training process from the old chain.
while pgrep -af 'phase8_catalog_matched_s2_(realq|shuffledq)|eval.py.*phase8_catalog_matched_s2' >/dev/null; do
    sleep 30
done

FREE=$(df --output=avail -BG / | tail -n1 | tr -dc '0-9')
if [ "$FREE" -lt 80 ]; then
    echo "[QUEUE][FAIL] root free ${FREE}G < 80G safety gate" | tee -a "$LOG"
    exit 3
fi
echo "[QUEUE] predecessor passed; starting Q-safe sequence $(date --iso-8601=seconds)" | tee -a "$LOG"
exec env EXPERIMENT_RUN_MODE=fresh bash scripts/training_pipelines/run_phase8_qsafe_ft_sequence.sh
