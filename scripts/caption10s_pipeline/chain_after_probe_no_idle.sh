#!/bin/bash
set -euo pipefail
LOG=/home/kojiek/logs/c2p0_qwen3cap_chain.log
exec >>"$LOG" 2>&1
echo "CHAIN_START $(date -u +%FT%TZ)"

# 1) wait for 512-probe train session
while tmux has-session -t c2p0_k_probe 2>/dev/null; do
  echo "wait_probe $(date -u +%FT%TZ)"
  sleep 20
done
echo "PROBE_DONE $(date -u +%FT%TZ)"

# 2) full two-slot caption generation
/home/kojiek/research/meanaudio_training/caption10s_pipeline/run_c2p0_qwen3cap_full_slots.sh

# 3) meansim + K=3 quarter (S1 100k + S2 50k), then best/worst NoQ quarter
/home/kojiek/research/meanaudio_training/caption10s_pipeline/run_c2p0_qwen3cap_k3_s2_after_gen.sh
echo "CHAIN_DONE $(date -u +%FT%TZ)"
