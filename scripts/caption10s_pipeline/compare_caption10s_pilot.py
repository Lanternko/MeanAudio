#!/usr/bin/env python3
"""Compare old official captions vs new 10s Qwen captions with CLAP vs first-10s audio.

Used for n=512 (and any pilot). Emits SUMMARY.json for the review gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

SR = 16000
WINDOW_SAMPLES = SR * 10
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
CLAP_CKPT = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)
OFFICIAL_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv"
)


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_old_map(tsv: Path) -> dict[str, str]:
    with tsv.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r["caption"] for r in csv.DictReader(f, delimiter="\t")}


def load_new_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id"):
                out[rec["id"]] = rec
    return out


def load_crop(cid: str):
    path = id_to_audio_path(cid)
    full, _ = librosa.load(str(path), sr=SR, mono=True)
    full = np.asarray(full, dtype=np.float32)
    crop = full[:WINDOW_SAMPLES]
    if crop.shape[0] < WINDOW_SAMPLES:
        crop = np.pad(crop, (0, WINDOW_SAMPLES - crop.shape[0]))
    return full, crop


def load_clap():
    import laion_clap

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt(str(CLAP_CKPT), verbose=False)
    clap.eval()
    return clap


@torch.inference_mode()
def clap_pairs(clap, audios, texts, bs=16):
    scores = []
    for i in range(0, len(audios), bs):
        a = audios[i : i + bs]
        t = texts[i : i + bs]
        a_emb = np.asarray(clap.get_audio_embedding_from_data(x=a, use_tensor=False), dtype=np.float32)
        t_emb = np.asarray(clap.get_text_embedding(t, use_tensor=False), dtype=np.float32)
        a_emb /= np.linalg.norm(a_emb, axis=1, keepdims=True) + 1e-8
        t_emb /= np.linalg.norm(t_emb, axis=1, keepdims=True) + 1e-8
        scores.extend((a_emb * t_emb).sum(axis=1).tolist())
    return np.asarray(scores, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_jsonl", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--old_tsv", type=Path, default=OFFICIAL_TSV)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_map = load_old_map(args.old_tsv)
    new_map = load_new_jsonl(args.new_jsonl)
    ids = [i for i, r in new_map.items() if r.get("caption") and i in old_map]
    print(f"comparable ids: {len(ids)}", flush=True)
    if len(ids) < 10:
        raise SystemExit("too few comparable captions")

    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {cid: ex.submit(load_crop, cid) for cid in ids}
        for cid, fut in futs.items():
            try:
                full, crop = fut.result()
            except Exception as e:
                print(f"[WARN] audio {cid}: {e}", flush=True)
                continue
            items.append(
                {
                    "id": cid,
                    "old": old_map[cid],
                    "new": new_map[cid]["caption"],
                    "full": full,
                    "crop": crop,
                    "full_dur": float(len(full) / SR),
                }
            )
    print(f"loaded audio for {len(items)}", flush=True)

    clap = load_clap()
    old_10 = clap_pairs(clap, [it["crop"] for it in items], [it["old"] for it in items])
    new_10 = clap_pairs(clap, [it["crop"] for it in items], [it["new"] for it in items])
    old_30 = clap_pairs(clap, [it["full"] for it in items], [it["old"] for it in items])
    new_30 = clap_pairs(clap, [it["full"] for it in items], [it["new"] for it in items])
    delta = new_10 - old_10

    null_rate = 1.0 - (len(ids) / max(1, len(new_map)))
    # recompute null from full new_map
    n_total = len(new_map)
    n_null = sum(1 for r in new_map.values() if not r.get("caption"))
    null_rate = n_null / max(1, n_total)

    summary = {
        "tag": args.tag,
        "n_new_jsonl": n_total,
        "n_null": n_null,
        "null_rate": null_rate,
        "n_compared": len(items),
        "mean_full_dur": float(np.mean([it["full_dur"] for it in items])),
        "clap": {
            "old_vs_10s": {
                "mean": float(old_10.mean()),
                "median": float(np.median(old_10)),
                "std": float(old_10.std()),
            },
            "new10s_vs_10s": {
                "mean": float(new_10.mean()),
                "median": float(np.median(new_10)),
                "std": float(new_10.std()),
            },
            "old_vs_30s": {
                "mean": float(old_30.mean()),
                "median": float(np.median(old_30)),
                "std": float(old_30.std()),
            },
            "new10s_vs_30s": {
                "mean": float(new_30.mean()),
                "median": float(np.median(new_30)),
                "std": float(new_30.std()),
            },
        },
        "delta_new_minus_old_on_10s": {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "frac_positive": float((delta > 0).mean()),
            "p25": float(np.percentile(delta, 25)),
            "p75": float(np.percentile(delta, 75)),
        },
        "bug_confirmed_old_prefers_30s": bool(old_30.mean() > old_10.mean()),
        "old_30_minus_10": float(old_30.mean() - old_10.mean()),
        "mean_new_caption_len": float(np.mean([len(it["new"]) for it in items])),
    }

    # examples
    order = sorted(range(len(items)), key=lambda i: delta[i], reverse=True)
    summary["examples_most_improved"] = [
        {
            "id": items[i]["id"],
            "delta": float(delta[i]),
            "old": items[i]["old"][:220],
            "new": items[i]["new"][:220],
        }
        for i in order[:5]
    ]
    summary["examples_most_regressed"] = [
        {
            "id": items[i]["id"],
            "delta": float(delta[i]),
            "old": items[i]["old"][:220],
            "new": items[i]["new"][:220],
        }
        for i in order[-5:][::-1]
    ]

    out_sum = args.out_dir / f"{args.tag}_SUMMARY.json"
    out_tsv = args.out_dir / f"{args.tag}_compare.tsv"
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "full_dur",
                "old_caption",
                "new_caption",
                "clap_old_10s",
                "clap_new_10s",
                "clap_old_30s",
                "delta",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for i, it in enumerate(items):
            w.writerow(
                {
                    "id": it["id"],
                    "full_dur": f"{it['full_dur']:.2f}",
                    "old_caption": it["old"],
                    "new_caption": it["new"],
                    "clap_old_10s": f"{old_10[i]:.4f}",
                    "clap_new_10s": f"{new_10[i]:.4f}",
                    "clap_old_30s": f"{old_30[i]:.4f}",
                    "delta": f"{delta[i]:.4f}",
                }
            )
    out_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[COMPLETE] {out_sum}")


if __name__ == "__main__":
    main()
