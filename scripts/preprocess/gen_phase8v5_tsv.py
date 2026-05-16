"""
生成 phase8_v5 的 TSV 檔案：
  - phase8_v5_train.tsv：實際 CLAP sim 前綴 + phase7_v1 random caption，無 q_level 欄
  - phase8_v5_test.tsv ：固定前綴 0.58 + phase4_test.tsv 原始 caption，無 q_level 欄

格式：id \t caption（兩欄）
Caption 格式：Text alignment is {clap_sim:.2f}. {原始 caption}
"""

import json
import csv
import sys

DATA_DIR = "/mnt/HDD/kojiek/phase4_jamendo_data"

CLAP_SIM_FILE  = f"{DATA_DIR}/phase8_v3_clap_sim.jsonl"
TRAIN_SRC_TSV  = f"{DATA_DIR}/phase7_v1_train.tsv"
TEST_SRC_TSV   = f"{DATA_DIR}/phase4_test.tsv"
TRAIN_OUT_TSV  = f"{DATA_DIR}/phase8_v5_train.tsv"
TEST_OUT_TSV   = f"{DATA_DIR}/phase8_v5_test.tsv"

INFERENCE_PREFIX_VAL = 0.58   # phase8_v3 CLAP sim 99 percentile

# ── 1. 讀入 CLAP sim（id → float）───────────────────────────
print("Loading CLAP sim...", flush=True)
clap_sim = {}
with open(CLAP_SIM_FILE) as f:
    for line in f:
        obj = json.loads(line)
        clap_sim[obj["id"]] = obj["clap_sim"]
print(f"  Loaded {len(clap_sim):,} entries", flush=True)

# ── 2. 生成 phase8_v5_train.tsv ─────────────────────────────
print("Generating train TSV...", flush=True)
missing = 0
written = 0
with open(TRAIN_SRC_TSV, newline="") as fin, \
     open(TRAIN_OUT_TSV, "w", newline="") as fout:

    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
    writer.writerow(["id", "caption"])   # header（無 q_level）

    for row in reader:
        clip_id = row["id"]
        caption = row["caption"]
        if clip_id not in clap_sim:
            missing += 1
            continue
        sim = clap_sim[clip_id]
        new_caption = f"Text alignment is {sim:.2f}. {caption}"
        writer.writerow([clip_id, new_caption])
        written += 1

print(f"  Written: {written:,}  Missing clap_sim: {missing}", flush=True)

# ── 3. 生成 phase8_v5_test.tsv ──────────────────────────────
print("Generating test TSV...", flush=True)
written_test = 0
with open(TEST_SRC_TSV, newline="") as fin, \
     open(TEST_OUT_TSV, "w", newline="") as fout:

    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
    writer.writerow(["id", "caption"])

    for row in reader:
        clip_id = row["id"]
        caption = row["caption"]
        new_caption = f"Text alignment is {INFERENCE_PREFIX_VAL:.2f}. {caption}"
        writer.writerow([clip_id, new_caption])
        written_test += 1

print(f"  Written: {written_test:,}", flush=True)

print("Done.")
print(f"  Train → {TRAIN_OUT_TSV}")
print(f"  Test  → {TEST_OUT_TSV}")
