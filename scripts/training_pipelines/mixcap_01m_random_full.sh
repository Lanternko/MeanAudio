#!/bin/bash
# Full-scale entry point for the mixcap_01m line, plus every CFG3+neg cell this
# line needs.
#
# Why the eval work is bolted here rather than inside the shared action: the
# action was already running under seat 046 when the operator asked for both
# eval cells, and bash reads a script incrementally, so editing a live action is
# unsafe (memory reference_bash_script_buffered_reads.md) and would also break
# the digest both wrappers pin. The action therefore stays immutable and the
# wrapper carries the additions.
#
# The two control arms have never been evaluated at CFG 3.0 + negative prompt --
# the only rotation numbers on the board are CFG0 -- so without Step 1 the new
# arm's CFG3+neg cell would have nothing to be compared against.
#
# Order is deliberate: the evals run before the 19h training, so the quarter
# comparison is complete even if the Step 0 gate inside the action stops the
# full arm.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/mixcap_01m_random_action.sh"
EXPECTED="48e08857b596713417ca27e08168e5a39d171234bcf58689c1aed1d6f949151e"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
CFG3NEG="/home/kojiek/MeanAudio/scripts/eval/mc_mf25_cfg3neg_eval.sh"
EXPS="/home/kojiek/MeanAudio/exps"

# ---- Step 1: CFG3+neg for the two all-Qwen controls -------------------------
CTRL_Q=phase8_qwen_caption2p0_k3_true012_random_noq_quarter_stage2_50000
CTRL_F=phase8_qwen_caption2p0_k3_true_random_noq_full_stage2_200000
/bin/bash "$CFG3NEG" "${CTRL_Q%_stage2_50000}"  "$EXPS/$CTRL_Q/${CTRL_Q}_ema_final.pth"
/bin/bash "$CFG3NEG" "${CTRL_F%_stage2_200000}" "$EXPS/$CTRL_F/${CTRL_F}_ema_final.pth"

# ---- Step 2: CFG3+neg for the quarter arm (046 produced only its CFG0 cell) --
ARM_Q=mixcap_01m_random_noq_quarter_stage2_50000
/bin/bash "$CFG3NEG" mixcap_01m_random_noq_quarter "$EXPS/$ARM_Q/${ARM_Q}_ema_final.pth"

# ---- Step 3: gate, full training, canonical CFG0 ----------------------------
/bin/bash "$SHARED" full

# ---- Step 4: CFG3+neg for the full arm --------------------------------------
ARM_F=mixcap_01m_random_noq_full_stage2_200000
exec /bin/bash "$CFG3NEG" mixcap_01m_random_noq_full "$EXPS/$ARM_F/${ARM_F}_ema_final.pth"
