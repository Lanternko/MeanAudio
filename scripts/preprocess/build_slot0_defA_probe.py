#!/usr/bin/env python3
"""Build the definition-A recall/false-flag probe for choosing a local screen model.

Test set only; never used to select rows for cleaning. Deterministic (seed fixed).
Strata (label = expected A decision):
  fixture_flag / fixture_keep : calibration_v5 + heldout_v3 relabeled under A
                                (slot4-corpus stress rows and the ambiguous
                                "The music has a tempo of." fragment excluded)
  luna_real_flag              : the 11 of Luna's 28 full_v1 non-KEEP rows that are
                                contamination under A (hand-classified from captions)
  luna_hard_keep              : the other 17 (grammar slips / contradictions)
  luna_random_keep            : 300 random Luna-KEEP rows from full_v1
  pattern_meta_flag           : 40 random slot0 rows starting "The caption ... is/reads/would be:"
  pattern_should_flag         : 15 random slot0 rows starting "The caption should"
"""
from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/experiments/slot0_semantic_audit_20260915"
SRC = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")
LUNA_DB = Path.home() / "exps_nvme/slot0_semantic_audit_20260915/full_v1/state.sqlite"
SEED = 20260915

FIXTURE_EXCLUDE = {"review_02"}  # "The music has a tempo of." -- damaged fragment, not A-contamination, not clean
LUNA_A_FLAG = {  # hand-classified 2026-09-15 from caption text (see README v4)
    "00_1194000_segment_1_0": "instruction_or_prompt_echo",
    "01_1248601_segment_0_0": "instruction_or_prompt_echo",
    "01_191401_segment_4_0": "instruction_or_prompt_echo",
    "00_588700_segment_6_0": "metatext",
    "01_1125001_segment_1_0": "metatext",
    "01_133101_segment_0_0": "metatext",
    "01_1378101_segment_5_0": "metatext",
    "02_1118602_segment_25_0": "metatext",
    "02_1137202_segment_0_0": "metatext",
    "00_383000_segment_5_0": "model_commentary",
    "01_374601_segment_1_0": "model_commentary",
}
META_RE = re.compile(r"^\s*The caption(?: text)?(?: (?:for|of) (?:this|the) (?:music clip|audio clip|audio|music|clip))?"
                     r"(?: is| reads| would be)\s*:", re.I)
SHOULD_RE = re.compile(r"^\s*The caption should\b", re.I)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1,
                    help="1 = development probe (seed 20260915); 2 = held-out probe built after the "
                         "prompt fix: different seed, no overlap with v1, fixtures/Luna-real strata dropped "
                         "(they were seen during development)")
    args = ap.parse_args()
    global SEED
    v1_ids = set()
    if args.version == 2:
        SEED = 20260916
        v1_ids = {x["id"] for x in json.loads((DOCS / "probe_defA_v1.json").read_text())}
    items = []
    for fx in (("calibration_v5.json", "heldout_v3.json") if args.version == 1 else ()):
        for r in json.loads((DOCS / fx).read_text()):
            if r["id"] in FIXTURE_EXCLUDE or r["id"].startswith("slot4_"):
                continue
            label = "KEEP" if r["expected_decisions"] == ["KEEP"] else "FLAG"
            items.append({"id": f"fx:{fx}:{r['id']}", "caption": r["caption"], "label": label,
                          "stratum": f"fixture_{label.lower()}"})

    db = sqlite3.connect(f"file:{LUNA_DB}?mode=ro", uri=True)
    caps, luna = {}, {}
    for payload, result in db.execute("SELECT payload, result FROM jobs WHERE phase='audit' AND status='done'"):
        for row in json.loads(payload):
            caps[row["id"]] = row["caption"]
        for d in json.loads(result)["decisions"]:
            luna[d["id"]] = d["decision"]
    non_keep = sorted(i for i, d in luna.items() if d != "KEEP")
    assert set(LUNA_A_FLAG) <= set(non_keep) and len(non_keep) == 28
    for i in (non_keep if args.version == 1 else []):
        flag = i in LUNA_A_FLAG
        items.append({"id": i, "caption": caps[i], "label": "FLAG" if flag else "KEEP",
                      "stratum": "luna_real_flag" if flag else "luna_hard_keep"})
    rng = random.Random(SEED)
    keep_ids = sorted(i for i, d in luna.items() if d == "KEEP" and i not in v1_ids)
    for i in rng.sample(keep_ids, 300 if args.version == 1 else 500):
        items.append({"id": i, "caption": caps[i], "label": "KEEP", "stratum": "luna_random_keep"})

    csv.field_size_limit(10**9)
    meta, should = [], []
    with SRC.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["id"] in luna or r["id"] in v1_ids:
                continue
            if META_RE.search(r["caption"]):
                meta.append((r["id"], r["caption"]))
            elif SHOULD_RE.search(r["caption"]):
                should.append((r["id"], r["caption"]))
    for stratum, pool, n in (("pattern_meta_flag", meta, 40), ("pattern_should_flag", should, 15)):
        for i, c in rng.sample(pool, min(n, len(pool))):
            items.append({"id": i, "caption": c, "label": "FLAG", "stratum": stratum})

    assert len({x["id"] for x in items}) == len(items)
    out = DOCS / f"probe_defA_v{args.version}.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n")
    counts = {}
    for x in items:
        counts[x["stratum"]] = counts.get(x["stratum"], 0) + 1
    print(json.dumps({"total": len(items), "pattern_pool": {"meta": len(meta), "should": len(should)}, **counts}))


if __name__ == "__main__":
    main()
