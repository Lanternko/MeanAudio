"""Deterministically sample N prompts from MusicCaps test TSV for v3 subjective A/B.

Output: eval_output/subjective_ab_v3/sampled_prompts.tsv (clip_name, mc_id, prompt)
"""
import csv
import random
from pathlib import Path

TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
OUT_DIR = Path("eval_output/subjective_ab_v3")
N_SAMPLES = 24
SAMPLE_SEED = 42  # Python random seed for reproducibility (separate from infer seed)

OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with open(TSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append((row["id"], row["caption"]))

print(f"Loaded {len(rows)} rows from {TSV.name}")

rng = random.Random(SAMPLE_SEED)
sampled = rng.sample(rows, N_SAMPLES)

out_tsv = OUT_DIR / "sampled_prompts.tsv"
with open(out_tsv, "w", encoding="utf-8") as f:
    f.write("clip_name\tmc_id\tprompt\n")
    for i, (mc_id, caption) in enumerate(sampled, start=1):
        clip_name = f"mc{i:02d}"
        # Strip any tabs/newlines in caption just in case
        caption_safe = caption.replace("\t", " ").replace("\n", " ").strip()
        f.write(f"{clip_name}\t{mc_id}\t{caption_safe}\n")

print(f"Wrote {out_tsv} ({N_SAMPLES} rows, sample_seed={SAMPLE_SEED})")
print(f"First 3:")
for i, (mc_id, cap) in enumerate(sampled[:3], start=1):
    print(f"  mc{i:02d} ({mc_id}): {cap[:80]}...")
