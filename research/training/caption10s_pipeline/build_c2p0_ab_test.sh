#!/bin/bash
# Build 4-system MusicCaps AB test (c20, best, worst, k3-q9) and push
# https://lanternko.github.io/audio-ab-test/
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

MEANAUDIO=/home/kojiek/MeanAudio
AB=/home/kojiek/audio-ab-test
PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
POOL=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/c2p0_ab_pool.json
TSV=/tmp/c2p0_ab_musiccaps30.tsv
STAGING=/tmp/c2p0_ab_audio
MUSICCAPS=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv

C20=$MEANAUDIO/exps/phase8_qwen_caption10s_multisent_noq_quarter_stage2_50000/phase8_qwen_caption10s_multisent_noq_quarter_stage2_50000_ema_final.pth
BEST=$MEANAUDIO/exps/phase8_qwen_caption2p0_bestof3_noq_quarter_stage2_50000/phase8_qwen_caption2p0_bestof3_noq_quarter_stage2_50000_ema_final.pth
WORST=$MEANAUDIO/exps/phase8_qwen_caption2p0_worstof3_noq_quarter_stage2_50000/phase8_qwen_caption2p0_worstof3_noq_quarter_stage2_50000_ema_final.pth
K3=$MEANAUDIO/exps/phase8_qwen_caption2p0_qwen3cap_k3_balanced_quarter_stage2_50000/phase8_qwen_caption2p0_qwen3cap_k3_balanced_quarter_stage2_50000_ema_final.pth

echo "AB_BUILD_START $(date -u +%FT%TZ)"
[ -f "$POOL" ] || { echo "FAIL missing $POOL"; exit 2; }
cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

python3 - "$POOL" "$MUSICCAPS" "$TSV" <<'PY'
import csv, json, sys
from pathlib import Path
pool = json.loads(Path(sys.argv[1]).read_text())
wanted = {r["id"]: r for r in pool}
with open(sys.argv[2], encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
out = []
for r in rows:
    if r["id"] in wanted:
        out.append({"id": r["id"], "caption": r["caption"]})
if len(out) != 30:
    raise SystemExit(f"tsv subset {len(out)}/30")
Path(sys.argv[3]).write_text(
    "id\tcaption\n" + "".join(f"{r['id']}\t{r['caption']}\n" for r in out)
)
print("tsv", sys.argv[3], len(out))
PY

gen_one() {
  local tag="$1" ckpt="$2"
  shift 2
  local dest="$STAGING/$tag"
  mkdir -p "$dest"
  local n
  n=$(find "$dest" -name "*.flac" | wc -l)
  if [ "$n" -eq 30 ]; then
    echo "SKIP_GEN $tag"
    return
  fi
  echo "GEN $tag $(date -u +%FT%TZ)"
  python eval.py --variant meanaudio_s --model_path "$ckpt" \
    --output "$dest" --tsv "$TSV" --use_meanflow \
    --num_steps 25 --cfg_strength 4.5 \
    --encoder_name t5_clap --text_c_dim 512 --seed 42 \
    --no_text_attention_mask --full_precision \
    "$@" \
    2>&1 | tee "/home/kojiek/logs/c2p0_ab_gen_${tag}.log"
  n=$(find "$dest" -name "*.flac" | wc -l)
  [ "$n" -eq 30 ] || { echo "FAIL $tag generated $n/30"; exit 2; }
}

gen_one c20 "$C20" --no_q
gen_one best "$BEST" --no_q
gen_one worst "$WORST" --no_q
gen_one k3 "$K3" --quality_level 9

python3 - "$POOL" "$STAGING" "$AB" <<'PY'
import json, shutil
from pathlib import Path

pool = json.loads(Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/c2p0_ab_pool.json").read_text())
staging = Path("/tmp/c2p0_ab_audio")
ab = Path("/home/kojiek/audio-ab-test")
audio = ab / "audio"
audio.mkdir(exist_ok=True)
variants = ["c20", "best", "worst", "k3"]
mapping = {}
for rec in pool:
    mc, cid = rec["mc"], rec["id"]
    files = {}
    for tag in variants:
        src = staging / tag / f"{cid}.flac"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        dst_name = f"{mc}_{tag}.flac"
        shutil.copy2(src, audio / dst_name)
        files[tag] = dst_name
    mapping[mc] = files

boot = ab / "fullscale_boot.js"
lines = ["// C2.0 / best-of-3 / worst-of-3 / K=3-q9 quarter MF25 CFG4.5 AB test.", "const FULLSCALE_AUDIO = {"]
for rec in pool:
    mc = rec["mc"]
    f = mapping[mc]
    lines.append(
        f'  {mc}: {{ c20: "{f["c20"]}", best: "{f["best"]}", worst: "{f["worst"]}", k3: "{f["k3"]}" }},'
    )
lines += [
    "};",
    "",
    'DATA.project = "audio_ab_test_c20_best_worst_k3_quarter_mf25";',
    'DATA.projectLabel = "C2.0 · Best · Worst · K3";',
    'DATA.variants = ["c20", "best", "worst", "k3"];',
    "DATA.roundSize = 12;",
    "DATA.variantDescriptions = {",
    '  c20: { shortName: "C2.0", displayName: "Caption 2.0", en: "C2.0 multisent NoQ quarter; MusicCaps MF25 CFG4.5.", zh: "Caption 2.0 多句 NoQ quarter；MusicCaps MF25 CFG4.5。" },',
    '  best: { shortName: "Best", displayName: "Best-of-3", en: "Per-clip highest CLAP caption among 3 slots.", zh: "三槽裡 CLAP 最高的那句當訓練 caption。" },',
    '  worst: { shortName: "Worst", displayName: "Worst-of-3", en: "Per-clip lowest CLAP caption among 3 slots.", zh: "三槽裡 CLAP 最低的那句當訓練 caption。" },',
    '  k3: { shortName: "K3", displayName: "K=3 q9", en: "Qwen-3cap MeanSim K=3, inferred at q=9.", zh: "Qwen 三句 MeanSim K=3，推論固定 q=9。" },',
    "};",
    "DATA.pool = DATA.pool.map(item => ({ ...item, files: FULLSCALE_AUDIO[item.clipId] }));",
    "",
]
# keep the existing helper functions from current boot
old = boot.read_text()
idx = old.find("function fullscaleShuffle")
if idx < 0:
    raise SystemExit("fullscaleShuffle missing")
boot.write_text("\n".join(lines) + "\n" + old[idx:])
print("wrote", boot)

manifest = {
    "project": "audio_ab_test_c20_best_worst_k3_quarter_mf25",
    "variants": variants,
    "round_size": 12,
    "pool_size": 30,
    "pair_schedule": "six unordered pairs, each repeated twice per round; A/B randomized",
    "audio_format": "FLAC, 16-bit mono 16 kHz",
    "source_protocols": {
        "c20": "Caption 2.0 multisent NoQ quarter S2; MusicCaps MF25 CFG4.5",
        "best": "Best-of-3 CLAP caption NoQ quarter S2; MusicCaps MF25 CFG4.5",
        "worst": "Worst-of-3 CLAP caption NoQ quarter S2; MusicCaps MF25 CFG4.5",
        "k3": "Qwen-3cap MeanSim K=3 balanced quarter S2; infer quality_level=9; MusicCaps MF25 CFG4.5",
    },
    "matching": pool,
}
(ab / "fullscale_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

html = (ab / "index.html").read_text()
html = html.replace("fullscale_boot.js?v=1", "fullscale_boot.js?v=2")
html = html.replace("fullscale_results.jsx?v=1", "fullscale_results.jsx?v=2")
html = html.replace("audio-ab-test-state-fullscale-v1", "audio-ab-test-state-c20-best-worst-k3-v1")
html = html.replace("audio-ab-test-seen-fullscale-v1", "audio-ab-test-seen-c20-best-worst-k3-v1")
(ab / "index.html").write_text(html)
print("site files updated")
PY

cd "$AB"
git add audio/mc*_c20.flac audio/mc*_best.flac audio/mc*_worst.flac audio/mc*_k3.flac \
  fullscale_boot.js fullscale_manifest.json index.html
git status --short | head -40
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -m "$(cat <<EOF
Add C2.0 / best / worst / K3-q9 quarter MF25 AB test

- 30 MusicCaps clips, 4 systems, same six-pair fullscale schedule
- Delivered at https://lanternko.github.io/audio-ab-test/
EOF
)"
  git push origin HEAD
fi
echo "AB_BUILD_DONE $(date -u +%FT%TZ)"
echo "https://lanternko.github.io/audio-ab-test/"
