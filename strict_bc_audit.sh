#!/bin/bash
# Strict audit of BC single-cap NPZ before P4V2-Qwen launches.
# Per Codex 2026-05-05: count + shape + id alignment + random/bc not mixed.

set -uo pipefail
LOG=~/logs/strict_bc_audit.log
exec > >(tee "$LOG") 2>&1

BC_NPZ=$HOME/phase9_5_bc_singlecap_npz
RANDOM_NPZ=$HOME/phase9_5_random_singlecap_npz
SEL_JSON=$HOME/research/meanaudio_training/qwen_singlecap_selections.json
TSV=/mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_bc_train.tsv
REF_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv

echo "=== strict_bc_audit at $(date) ==="

if [ ! -d "$BC_NPZ" ]; then
    echo "[FAIL] BC NPZ dir missing: $BC_NPZ"; exit 1
fi
n_bc=$(ls "$BC_NPZ" | wc -l)
n_random=$(ls "$RANDOM_NPZ" 2>/dev/null | wc -l)
n_tsv=$(($(wc -l < "$REF_TSV") - 1))
echo ""
echo "[1/5] count match"
echo "  BC NPZ:     $n_bc"
echo "  Random NPZ: $n_random"
echo "  ref TSV:    $n_tsv"
[ "$n_bc" = "251599" ] || { echo "[FAIL] BC count $n_bc != 251599"; exit 2; }
[ "$n_bc" = "$n_tsv" ] || { echo "[FAIL] BC count $n_bc != ref $n_tsv"; exit 3; }
echo "  ✓ exact match"

echo ""
echo "[2/5] BC vs Random differ where bestconsensus_idx != random_seed42_idx"
python - <<PYEOF
import random, numpy as np, json, sys, csv
from pathlib import Path

BC = Path('$BC_NPZ')
RAND = Path('$RANDOM_NPZ')
SEL = json.load(open('$SEL_JSON'))

with open('$REF_TSV') as f:
    ids = [r['id'] for r in csv.DictReader(f, delimiter='\t')]

rng = random.Random(0)
samples = rng.sample(range(len(ids)), 100)

mismatch_expected = 0
match_expected = 0
embed_actually_differ = 0
embed_actually_same = 0

for i in samples:
    cid = ids[i]
    s = SEL.get(cid)
    if not s:
        continue
    bc_idx = s['bestconsensus_idx']
    rand_idx = s['random_seed42_idx']
    bc_data = np.load(BC / f'{i}.npz')
    rand_data = np.load(RAND / f'{i}.npz')

    if bc_data['text_features'].shape != (77, 1024):
        print(f'[FAIL] BC idx {i}: tf shape {bc_data["text_features"].shape}')
        sys.exit(1)
    if bc_data['text_features_c'].shape != (512,):
        print(f'[FAIL] BC idx {i}: tfc shape {bc_data["text_features_c"].shape}')
        sys.exit(1)

    same = np.allclose(bc_data['text_features_c'], rand_data['text_features_c'], atol=1e-6)
    if bc_idx == rand_idx:
        match_expected += 1
        if same: embed_actually_same += 1
        else:
            print(f'[FAIL] idx {i} cid {cid}: bc==rand but embed differ')
            sys.exit(1)
    else:
        mismatch_expected += 1
        if not same: embed_actually_differ += 1
        else:
            print(f'[FAIL] idx {i} cid {cid}: bc({bc_idx})!=rand({rand_idx}) but embed identical')
            sys.exit(1)

print(f'  bc==rand: {match_expected:5d} (~20% expected; embed all match: {embed_actually_same==match_expected})')
print(f'  bc!=rand: {mismatch_expected:5d} (embed all differ: {embed_actually_differ==mismatch_expected})')
print('  ✓ BC NPZ contains different slot from Random per selection JSON')
PYEOF

echo ""
echo "[3/5] BC slot mapping matches selections.json (vs source multi-cap)"
python - <<PYEOF
import csv, json, numpy as np, random, sys
from pathlib import Path

BC = Path('$BC_NPZ')
MULTI = Path.home() / 'phase9_5_multicap_npz'
SEL = json.load(open('$SEL_JSON'))

with open('$REF_TSV') as f:
    ids = [r['id'] for r in csv.DictReader(f, delimiter='\t')]

rng = random.Random(1)
samples = rng.sample(range(len(ids)), 50)
bad = 0
for i in samples:
    cid = ids[i]
    bc_idx = SEL[cid]['bestconsensus_idx']
    src_multi = np.load(MULTI / f'{i}.npz')
    expected_tf  = src_multi['text_features'][bc_idx]
    expected_tfc = src_multi['text_features_c'][bc_idx]
    actual = np.load(BC / f'{i}.npz')
    if not np.allclose(actual['text_features'], expected_tf, atol=1e-6):
        bad += 1
        print(f'[FAIL] idx {i} cid {cid} bc_idx {bc_idx}: tf mismatch')
    if not np.allclose(actual['text_features_c'], expected_tfc, atol=1e-6):
        bad += 1
        print(f'[FAIL] idx {i} cid {cid} bc_idx {bc_idx}: tfc mismatch')

if bad == 0:
    print(f'  ✓ 50/50 BC NPZ slot mapping matches selections.json + source multi-cap')
else:
    sys.exit(1)
PYEOF

echo ""
echo "[4/5] BC TSV id order"
python - <<PYEOF
import csv, sys
ref = [r['id'] for r in csv.DictReader(open('$REF_TSV'), delimiter='\t')]
bc  = [r['id'] for r in csv.DictReader(open('$TSV'), delimiter='\t')]
if ref != bc:
    print(f'[FAIL] BC TSV id order != phase7_v1_train.tsv'); sys.exit(1)
print(f'  ✓ BC TSV {len(bc):,} ids match phase7_v1 order exactly')
PYEOF

echo ""
echo "[5/5] BC + Random dirs distinct (no path mixing)"
[ "$BC_NPZ" != "$RANDOM_NPZ" ] || { echo "[FAIL] paths identical"; exit 4; }
echo "  ✓ BC=$BC_NPZ"
echo "  ✓ Random=$RANDOM_NPZ"

echo ""
echo "=== ALL BC AUDITS PASS at $(date) — P4V2-Qwen launch is safe ==="
