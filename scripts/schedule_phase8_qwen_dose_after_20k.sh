#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTH="${1:?usage: $0 /path/to/dose_authorization.json}"
PARENT_REPORT="/home/kojiek/exps_nvme/phase8_qwen_official_matched/paired_final_report.json"
PARENT_LOG="/home/kojiek/logs/phase8_qwen_official_matched_queue.log"
PARENT_MANIFEST="/home/kojiek/logs/phase8_qwen_official_matched_monitor/execution_manifest.json"
DOSE_LOG="/home/kojiek/logs/phase8_qwen_dose_queue.log"

while true; do
  if /home/kojiek/venvs/dac/bin/python "$ROOT/scripts/phase8_qwen_parent_completion_gate.py" \
      --report "$PARENT_REPORT" --manifest "$PARENT_MANIFEST"
  then
    break
  fi
  if ! pgrep -f 'phase8_qwen_probe_queue.py --execute' >/dev/null; then
    printf '[FAIL] parent queue ended without a passed paired report\n' | tee -a "$DOSE_LOG"
    tail -n 40 "$PARENT_LOG" >> "$DOSE_LOG" || true
    exit 1
  fi
  sleep 60
done

if ! tmux has-session -t p8_qwen_dose_luna 2>/dev/null; then
  tmux new-session -d -s p8_qwen_dose_luna \
    "bash -lc 'cd /home/kojiek/MeanAudio && source /home/kojiek/venvs/dac/bin/activate && bash scripts/phase8_qwen_dose_luna_loop.sh >> /home/kojiek/logs/phase8_qwen_dose_monitor/luna_loop.log 2>&1'"
fi

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
set -o pipefail
python scripts/phase8_qwen_dose_queue.py --execute --run-mode fresh --authorization "$AUTH" \
  2>&1 | tee -a "$DOSE_LOG"
