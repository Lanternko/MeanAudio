#!/usr/bin/env python3
"""Build full-corpus Qwen-3cap MeanSim K=3 TSV in official-matched row order."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

FULL = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full")
SLOT0 = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/captions_full_251599_10s_multisent.jsonl")
BASE_TSV = Path("/home/kojiek/MeanAudio/data/phase8_caption2p0_k3_balanced_train.tsv")
MODEL = "all-MiniLM-L6-v2"
CODES = [0, 5, 9]


def load_jsonl(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") and rec.get("caption"):
                out[rec["id"]] = rec["caption"]
    return out


def main() -> None:
    s0 = load_jsonl(SLOT0)
    s1 = load_jsonl(FULL / "slot1_temp115.jsonl")
    s2 = load_jsonl(FULL / "slot2_syntax.jsonl")
    with BASE_TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = [r["id"] for r in rows]
    missing = [i for i in ids if not (i in s0 and i in s1 and i in s2)]
    if missing:
        raise SystemExit(f"incomplete 3-cap coverage: missing {len(missing)} / {len(ids)}")
    print(f"encoding n={len(ids)}", flush=True)
    model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    e0 = model.encode([s0[i] for i in ids], convert_to_tensor=True, normalize_embeddings=True, batch_size=256)
    e1 = model.encode([s1[i] for i in ids], convert_to_tensor=True, normalize_embeddings=True, batch_size=256)
    e2 = model.encode([s2[i] for i in ids], convert_to_tensor=True, normalize_embeddings=True, batch_size=256)
    s01 = (e0 * e1).sum(dim=-1)
    s02 = (e0 * e2).sum(dim=-1)
    s12 = (e1 * e2).sum(dim=-1)
    scores = ((s01 + s02 + s12) / 3.0).detach().cpu().numpy()
    ranked = sorted(range(len(ids)), key=lambda i: (float(scores[i]), ids[i]))
    q = [-1] * len(ids)
    k = 3
    base, rem = divmod(len(ids), k)
    cursor = 0
    for bucket in range(k):
        size = base + (bucket >= k - rem)
        for idx in ranked[cursor: cursor + size]:
            q[idx] = CODES[bucket]
        cursor += size
    out_tsv = FULL / "phase8_caption2p0_qwen3cap_k3_balanced_train.tsv"
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "caption", "q_level"], delimiter="\t")
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({"id": row["id"], "caption": row["caption"], "q_level": q[i]})
    summary = {
        "n": len(ids),
        "meansim_mean": float(np.mean(scores)),
        "q_hist": {str(c): q.count(c) for c in CODES},
        "tsv": str(out_tsv),
    }
    (FULL / "QWEN3CAP_K3_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
