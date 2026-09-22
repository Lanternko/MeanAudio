#!/bin/bash
# 075 D2 chain: build inputs -> defectlab -> defectunlab -> D1 probe on lab / unlab / 066 control.
# Holds gpu0.lock for the whole chain so the P2 host does not start a job underneath.
set -eo pipefail
cd "$HOME/MeanAudio"
PY="$HOME/venvs/dac/bin/python"
ROOT="$HOME/exps_nvme/defect_negsample"
LOG="$HOME/logs/075_d2_defect_negsample_chain.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
export CUDA_VISIBLE_DEVICES=0

exec 9< <("$PY" "$HOME/gpu_queue/hold_lock.py" "$HOME/gpu_queue/gpu0.lock" --timeout 60 --watch-pid $$)
read -r LOCKSTATE <&9
[ "$LOCKSTATE" = LOCKED ] || { log "[FAIL] gpu0.lock: $LOCKSTATE"; exit 5; }
log "gpu0.lock held"

if [ ! -f "$ROOT/arm_unlab/manifest.json" ] || [ ! -f "$ROOT/arm_lab/manifest.json" ]; then
  log "[build] arm inputs"
  "$PY" scripts/preprocess/build_defect_negsample_arm_inputs.py --root "$ROOT" > "$HOME/logs/075_build.log" 2>&1
  log "[build] done: $(tail -1 "$HOME/logs/075_build.log")"
fi
for ARM in defectlab defectunlab; do
  log "[arm] $ARM"
  bash scripts/training_pipelines/d2_defect_negsample_action.sh "$ARM" 2>&1 | tee -a "$HOME/logs/075_${ARM}.log"
done

# Manipulation check: the D1 probe (held-out defect strings) on each quarter checkpoint.
CTRL=phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265
for TAG in defectlab defectunlab control066; do
  if [ "$TAG" = control066 ]; then
    CK="$HOME/MeanAudio/exps/${CTRL}_stage2_50000/${CTRL}_stage2_50000_ema_final.pth"
  else
    E=phase8_qwen_caption2p0_slot0clean_${TAG}_noq_quarter_s14159265
    CK="$HOME/MeanAudio/exps/${E}_stage2_50000/${E}_stage2_50000_ema_final.pth"
  fi
  log "[d1 probe] $TAG"
  CKPT="$CK" OUT_ROOT="$HOME/eval_output_nvme/d2_075_d1probe/$TAG" \
    bash scripts/eval/run_d1_noise_probe.sh 2>&1 | tee -a "$LOG"
done
log "[DONE] 075 chain"
