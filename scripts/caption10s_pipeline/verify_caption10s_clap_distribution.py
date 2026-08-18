#!/usr/bin/env python3
"""Distributional CLAP audit: new 10s captions vs old captions on first-10s audio.

Requires broadly better alignment, not just a tiny mean lift.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import torch

SR = 16000
WINDOW = SR * 10
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
CLAP_CKPT = Path("/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt")


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_tsv_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r["caption"] for r in csv.DictReader(f, delimiter="\t")}


def load_crop(cid: str):
    path = id_to_audio_path(cid)
    full, _ = librosa.load(str(path), sr=SR, mono=True)
    full = np.asarray(full, dtype=np.float32)
    crop = full[:WINDOW]
    if crop.shape[0] < WINDOW:
        crop = np.pad(crop, (0, WINDOW - crop.shape[0]))
    return crop, float(len(full) / SR)


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


def bootstrap_mean_ci(x: np.ndarray, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    means = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(x[idx].mean()))
    means = np.asarray(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_tsv", type=Path, required=True)
    ap.add_argument("--new_tsv", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=Path, required=True)
    # distributional gates
    ap.add_argument("--min_mean_delta", type=float, default=0.015)
    ap.add_argument("--min_median_delta", type=float, default=0.01)
    ap.add_argument("--min_frac_positive", type=float, default=0.58)
    ap.add_argument("--min_frac_delta_ge_0p02", type=float, default=0.40)
    ap.add_argument("--require_ci_lo_gt_0", action="store_true", default=True)
    args = ap.parse_args()

    old_map = load_tsv_map(args.old_tsv)
    new_map = load_tsv_map(args.new_tsv)
    common = [i for i in new_map if i in old_map and new_map[i] and old_map[i]]
    rng = random.Random(args.seed)
    rng.shuffle(common)

    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for cid in common:
            if len(futs) >= args.n * 2:  # oversample for load failures
                break
            futs.append((cid, ex.submit(load_crop, cid)))
        for cid, fut in futs:
            if len(items) >= args.n:
                break
            try:
                crop, full_dur = fut.result()
            except Exception as e:
                print(f"[WARN] {cid}: {e}", flush=True)
                continue
            items.append(
                {
                    "id": cid,
                    "old": old_map[cid],
                    "new": new_map[cid],
                    "crop": crop,
                    "full_dur": full_dur,
                }
            )
    if len(items) < int(args.n * 0.9):
        raise SystemExit(f"too few samples loaded: {len(items)}")

    print(f"loaded n={len(items)} mean_full_dur={np.mean([i['full_dur'] for i in items]):.3f}", flush=True)
    clap = load_clap()
    old10 = clap_pairs(clap, [i["crop"] for i in items], [i["old"] for i in items])
    new10 = clap_pairs(clap, [i["crop"] for i in items], [i["new"] for i in items])
    delta = new10 - old10
    ci_lo, ci_hi = bootstrap_mean_ci(delta)

    stats = {
        "n": len(items),
        "seed": args.seed,
        "mean_full_dur": float(np.mean([i["full_dur"] for i in items])),
        "old_vs_10s": {
            "mean": float(old10.mean()),
            "median": float(np.median(old10)),
            "std": float(old10.std()),
        },
        "new_vs_10s": {
            "mean": float(new10.mean()),
            "median": float(np.median(new10)),
            "std": float(new10.std()),
        },
        "delta_new_minus_old": {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "std": float(delta.std()),
            "p10": float(np.percentile(delta, 10)),
            "p25": float(np.percentile(delta, 25)),
            "p75": float(np.percentile(delta, 75)),
            "p90": float(np.percentile(delta, 90)),
            "frac_positive": float((delta > 0).mean()),
            "frac_delta_ge_0p01": float((delta >= 0.01).mean()),
            "frac_delta_ge_0p02": float((delta >= 0.02).mean()),
            "frac_delta_ge_0p05": float((delta >= 0.05).mean()),
            "bootstrap_mean_ci95": [ci_lo, ci_hi],
        },
    }

    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    dmean = stats["delta_new_minus_old"]["mean"]
    dmed = stats["delta_new_minus_old"]["median"]
    fpos = stats["delta_new_minus_old"]["frac_positive"]
    f02 = stats["delta_new_minus_old"]["frac_delta_ge_0p02"]

    add("mean_delta", dmean >= args.min_mean_delta, f"mean={dmean:.4f} min={args.min_mean_delta}")
    add("median_delta", dmed >= args.min_median_delta, f"median={dmed:.4f} min={args.min_median_delta}")
    add(
        "frac_positive",
        fpos >= args.min_frac_positive,
        f"frac_pos={fpos:.3f} min={args.min_frac_positive}",
    )
    add(
        "frac_solid_gain",
        f02 >= args.min_frac_delta_ge_0p02,
        f"frac_delta>=0.02 = {f02:.3f} min={args.min_frac_delta_ge_0p02}",
    )
    add(
        "bootstrap_ci_lo_gt_0",
        ci_lo > 0,
        f"ci95=[{ci_lo:.4f},{ci_hi:.4f}]",
    )
    # broad improvement: not driven by a thin tail — p25 should not be deeply negative
    p25 = stats["delta_new_minus_old"]["p25"]
    add("p25_not_deeply_negative", p25 > -0.03, f"p25={p25:.4f}")

    failed = [c for c in checks if not c["ok"]]
    status = "passed" if not failed else "failed"
    order = np.argsort(-delta)
    examples_best = [
        {
            "id": items[i]["id"],
            "delta": float(delta[i]),
            "old": items[i]["old"][:180],
            "new": items[i]["new"][:180],
        }
        for i in order[:5]
    ]
    examples_worst = [
        {
            "id": items[i]["id"],
            "delta": float(delta[i]),
            "old": items[i]["old"][:180],
            "new": items[i]["new"][:180],
        }
        for i in order[-5:][::-1]
    ]

    payload = {
        "schema_version": 1,
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "CLAP(caption, first-10s-audio) paired delta new-old; distributional gates",
        "stats": stats,
        "checks": checks,
        "failed": [c["name"] for c in failed],
        "examples_most_improved": examples_best,
        "examples_most_regressed": examples_worst,
        "old_tsv": str(args.old_tsv),
        "new_tsv": str(args.new_tsv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status != "passed":
        raise SystemExit(2)
    print("[GATE PASS] distributional CLAP improvement confirmed")


if __name__ == "__main__":
    main()
