#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
CONTRACT=$ROOT/docs/experiments/caption2p0_quarter_cfg0_rerun_contract.json
EVALUATOR=$ROOT/scripts/caption10s_pipeline/eval_musiccaps_mf25.sh
PYTHON=/home/kojiek/venvs/dac/bin/python

usage() { echo "usage: $0 --preflight | <caption2p0|bestof3|worstof3|qwen3cap_k3_q9>" >&2; }
[ "$#" -eq 1 ] || { usage; exit 2; }

if [ "$1" = "--preflight" ]; then
  "$PYTHON" - "$CONTRACT" "$EVALUATOR" <<'PY'
import hashlib, json, sys
from pathlib import Path

contract, evaluator = map(Path, sys.argv[1:3])
p = json.loads(contract.read_text())
assert p["fixed_protocol"]["num_steps"] == 25
assert p["fixed_protocol"]["cfg_strength"] == 0
assert [c["cell_id"] for c in p["cells"]] == p["execution"]["order"]
assert len({c["label"] for c in p["cells"]}) == 4
assert sum(c["conditioning_argv"] == ["--no_q"] for c in p["cells"]) == 3
assert sum(c["conditioning_argv"] == ["--quality_level", "9"] for c in p["cells"]) == 1
for cell in p["cells"]:
    ckpt = Path(cell["checkpoint"])
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == cell["checkpoint_sha256"]
tsv = Path(p["fixed_protocol"]["tsv"])
assert hashlib.sha256(tsv.read_bytes()).hexdigest() == p["fixed_protocol"]["tsv_sha256"]
assert evaluator.is_file()
print("PREFLIGHT_OK cells=4 noq=3 q9=1 steps=25 cfg=0")
PY
  exit 0
fi

ARM=$1
readarray -t FIELDS < <("$PYTHON" - "$CONTRACT" "$ARM" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
matches = [c for c in p["cells"] if c["cell_id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit("unknown or duplicate arm")
c = matches[0]
print(c["label"])
print(c["checkpoint"])
print("\t".join(c["conditioning_argv"]))
PY
)
[ "${#FIELDS[@]}" -eq 3 ] || { echo "FAIL malformed contract cell" >&2; exit 2; }
LABEL=${FIELDS[0]}
CKPT=${FIELDS[1]}
IFS=$'\t' read -r -a CONDITIONING <<<"${FIELDS[2]}"

CFG0_CONTRACT="$CONTRACT" CFG0_ARM="$ARM" \
  "$EVALUATOR" "$LABEL" "$CKPT" "${CONDITIONING[@]}"
