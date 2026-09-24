#!/usr/bin/env python3
"""Length + 3-way Qwen MeanSim for the Caption 2.0 512-clip variant pilot."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

OUT = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_pilot_n512")
LP_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_meansim_k3_balanced.tsv")
MODEL = "all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") and rec.get("caption"):
                out[rec["id"]] = rec
    return out


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return {}
    return {
        "n": n,
        "mean": round(float(statistics.mean(xs)), 3),
        "p10": round(xs[max(0, int(n * 0.1) - 1)], 3),
        "p50": round(xs[n // 2], 3),
        "p90": round(xs[min(n - 1, int(n * 0.9))], 3),
        "min": round(xs[0], 3),
        "max": round(xs[-1], 3),
    }


def pairwise_mean(embs: torch.Tensor) -> float:
    sims = []
    for i, j in combinations(range(embs.size(0)), 2):
        sims.append(float(torch.nn.functional.cosine_similarity(embs[i], embs[j], dim=0)))
    return float(np.mean(sims))


def main() -> None:
    slot0 = load_jsonl(OUT / "slot0_c2p0.jsonl")
    slot1 = load_jsonl(OUT / "slot1_temp115.jsonl")
    slot2 = load_jsonl(OUT / "slot2_syntax.jsonl")
    common = sorted(set(slot0) & set(slot1) & set(slot2))
    report: dict = {
        "n_slot0": len(slot0),
        "n_slot1": len(slot1),
        "n_slot2": len(slot2),
        "n_common": len(common),
    }
    if len(common) < 50:
        raise SystemExit(f"too few common ids: {len(common)}")

    length = {}
    for name, store in ("slot0_c2p0", slot0), ("slot1_temp115", slot1), ("slot2_syntax", slot2):
        words = [len(store[i]["caption"].split()) for i in common]
        chars = [len(store[i]["caption"]) for i in common]
        sents = []
        for i in common:
            rec = store[i]
            sents.append(int(rec["n_sents"]) if rec.get("n_sents") is not None else rec["caption"].count(".") )
        length[name] = {"words": summarize(words), "chars": summarize(chars), "sents": summarize([float(s) for s in sents])}
    report["length"] = length

    model = SentenceTransformer(MODEL)
    texts0 = [slot0[i]["caption"] for i in common]
    texts1 = [slot1[i]["caption"] for i in common]
    texts2 = [slot2[i]["caption"] for i in common]
    e0 = model.encode(texts0, convert_to_tensor=True, normalize_embeddings=True)
    e1 = model.encode(texts1, convert_to_tensor=True, normalize_embeddings=True)
    e2 = model.encode(texts2, convert_to_tensor=True, normalize_embeddings=True)
    pair01, pair02, pair12, triple = [], [], [], []
    for a, b, c in zip(e0, e1, e2):
        s01 = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        s02 = float(torch.nn.functional.cosine_similarity(a, c, dim=0))
        s12 = float(torch.nn.functional.cosine_similarity(b, c, dim=0))
        pair01.append(s01)
        pair02.append(s02)
        pair12.append(s12)
        triple.append(float(np.mean([s01, s02, s12])))
    report["meansim"] = {
        "model": MODEL,
        "slot0_slot1": summarize(pair01),
        "slot0_slot2": summarize(pair02),
        "slot1_slot2": summarize(pair12),
        "triple_pairwise_mean": summarize(triple),
    }

    lp = {}
    if LP_TSV.is_file():
        with LP_TSV.open() as f:
            lp = {r["id"]: int(r["q_level"]) for r in csv.DictReader(f, delimiter="\t")}
    scores = triple[:]
    ranked = sorted(range(len(scores)), key=lambda i: (scores[i], common[i]))
    k = 3
    codes = [0, 5, 9]
    qwen_q = [-1] * len(scores)
    base, rem = divmod(len(scores), k)
    cursor = 0
    for bucket in range(k):
        size = base + (bucket >= k - rem)
        for idx in ranked[cursor: cursor + size]:
            qwen_q[idx] = codes[bucket]
        cursor += size
    assign = []
    same_lp = 0
    have_lp = 0
    for i, cid in enumerate(common):
        row = {
            "id": cid,
            "qwen3_meansim": round(scores[i], 6),
            "qwen_k3": qwen_q[i],
            "lp_k3": lp.get(cid),
        }
        if row["lp_k3"] is not None:
            have_lp += 1
            if row["lp_k3"] == row["qwen_k3"]:
                same_lp += 1
        assign.append(row)
    report["k3_assignment"] = {
        "qwen_hist": dict(sorted(Counter(qwen_q).items())),
        "lp_overlap_n": have_lp,
        "agree_with_lp_k3": same_lp,
        "agree_rate": round(same_lp / max(have_lp, 1), 3),
    }
    assign_path = OUT / "qwen3_vs_lp_k3.tsv"
    with assign_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "qwen3_meansim", "qwen_k3", "lp_k3"], delimiter="\t")
        w.writeheader()
        w.writerows(assign)
    summary_path = OUT / "PILOT_SUMMARY.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("wrote", summary_path)
    print("wrote", assign_path)


if __name__ == "__main__":
    main()
