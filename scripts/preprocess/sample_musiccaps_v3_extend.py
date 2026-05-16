"""Extend v3 from 24 -> 30 prompts: keep existing mc01-mc24, append mc25-mc30.

Strategy: preserve the existing 48 generated audio files untouched. Sample 6
*additional* prompts from MusicCaps test (excluding the 24 already sampled)
with a fresh seed so they're deterministic but non-overlapping.
"""
import csv
import random
from pathlib import Path

TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv")
OUT_DIR = Path("eval_output/subjective_ab_v3")
EXISTING_TSV = OUT_DIR / "sampled_prompts.tsv"
N_EXTEND = 6  # 24 -> 30
EXTEND_SEED = 43  # different from original (42) to avoid overlap patterns

# Load full MusicCaps pool
rows = []
with open(TSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append((row["id"], row["caption"]))
print(f"Loaded {len(rows)} rows from {TSV.name}")

# Load existing 24 to exclude
existing_ids = set()
existing_lines = []
with open(EXISTING_TSV, "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        existing_lines.append(line)
        clip_name, mc_id, prompt = line.rstrip("\n").split("\t", 2)
        existing_ids.add(mc_id)
print(f"Existing: {len(existing_ids)} prompts (mc01-mc{len(existing_ids):02d})")

# Sample 6 more from the remainder
remaining = [(mc_id, cap) for (mc_id, cap) in rows if mc_id not in existing_ids]
print(f"Remaining pool: {len(remaining)}")
rng = random.Random(EXTEND_SEED)
new_samples = rng.sample(remaining, N_EXTEND)

# Append to TSV (mc25-mc30)
with open(EXISTING_TSV, "a", encoding="utf-8") as f:
    start_idx = len(existing_ids) + 1
    for i, (mc_id, caption) in enumerate(new_samples, start=start_idx):
        clip_name = f"mc{i:02d}"
        caption_safe = caption.replace("\t", " ").replace("\n", " ").strip()
        f.write(f"{clip_name}\t{mc_id}\t{caption_safe}\n")
        print(f"  +{clip_name} ({mc_id}): {caption_safe[:70]}...")

print(f"\nExtended {EXISTING_TSV}: 24 -> {24 + N_EXTEND} rows")
