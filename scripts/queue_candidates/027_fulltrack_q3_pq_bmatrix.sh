#!/bin/bash
set -euo pipefail

# Dormant Gate-1 artifact.  It is not a queue registration and cannot run until
# the exact final contract and consumed one-use Gate-2 capability both exist.
exec /usr/bin/env -i \
  PATH=/home/kojiek/venvs/dac/bin:/usr/bin:/bin \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONUNBUFFERED=1 \
  PYTHONNOUSERSITE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/kojiek/venvs/dac/bin/python \
  /home/kojiek/MeanAudio/scripts/experiment_harness/fulltrack_q3_pq_bmatrix_harn.py run
