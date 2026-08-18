#!/usr/bin/env python3
"""Paired CLAP(caption, first-10s audio) for C2.0 slot0/1/2.

CPU by default. Sample n ids that already have all three captions.
"""
from __future__ import annotations

import argparse
import json
import random
import re
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
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_jsonl_caps(path: Path, want: set[str] | None = None) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r.get("id")
            if want is not None and cid not in want:
                continue
            cap = (r.get("caption") or "").strip()
            if cid and cap:
                out[cid] = cap
    return out


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
def embed_audio(clap, audios, bs: int):
    embs = []
    for i in range(0, len(audios), bs):
        a = audios[i : i + bs]
        e = np.asarray(clap.get_audio_embedding_from_data(x=a, use_tensor=False), dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        embs.append(e)
        print(f"audio_emb {min(i + bs, len(audios))}/{len(audios)}", flush=True)
    return np.concatenate(embs, axis=0)


@torch.inference_mode()
def embed_text(clap, texts, bs: int):
    embs = []
    for i in range(0, len(texts), bs):
        t = texts[i : i + bs]
        e = np.asarray(clap.get_text_embedding(t, use_tensor=False), dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        embs.append(e)
    return np.concatenate(embs, axis=0)


def summarize(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }


def bootstrap_mean_ci(x: np.ndarray, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(x)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument(
        "--slot0",
        type=Path,
        default=Path(
            "/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/captions_full_251599_10s_multisent.jsonl"
        ),
    )
    ap.add_argument(
        "--slot1",
        type=Path,
        default=Path(
            "/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/slot1_temp115.jsonl"
        ),
    )
    ap.add_argument(
        "--slot2",
        type=Path,
        default=Path(
            "/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/slot2_syntax.jsonl"
        ),
    )
    ap.add_argument(
        "--out_json",
        type=Path,
        default=Path(
            "/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/clap_slots_n1024.json"
        ),
    )
    args = ap.parse_args()

    print("device_cuda", torch.cuda.is_available(), flush=True)
    print("load slot2...", flush=True)
    s2 = load_jsonl_caps(args.slot2)
    ids = set(s2)
    print("slot2", len(s2), flush=True)
    print("load slot1 match...", flush=True)
    s1 = load_jsonl_caps(args.slot1, ids)
    print("slot1 match", len(s1), flush=True)
    print("load slot0 match...", flush=True)
    s0 = load_jsonl_caps(args.slot0, ids)
    print("slot0 match", len(s0), flush=True)

    common = [i for i in s2 if i in s1 and i in s0]
    rng = random.Random(args.seed)
    rng.shuffle(common)
    print(f"triple overlap={len(common)} sample_target={args.n}", flush=True)

    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for cid in common:
            if len(futs) >= args.n * 2:
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
                    "s0": s0[cid],
                    "s1": s1[cid],
                    "s2": s2[cid],
                    "crop": crop,
                    "full_dur": full_dur,
                }
            )
    if len(items) < int(args.n * 0.9):
        raise SystemExit(f"too few samples loaded: {len(items)}")
    durs = [it["full_dur"] for it in items]
    print(f"loaded n={len(items)} mean_full_dur={float(np.mean(durs)):.3f}", flush=True)

    clap = load_clap()
    a_emb = embed_audio(clap, [it["crop"] for it in items], args.bs)
    t0 = embed_text(clap, [it["s0"] for it in items], args.bs)
    print("text0 done", flush=True)
    t1 = embed_text(clap, [it["s1"] for it in items], args.bs)
    print("text1 done", flush=True)
    t2 = embed_text(clap, [it["s2"] for it in items], args.bs)
    print("text2 done", flush=True)

    sim0 = (a_emb * t0).sum(axis=1)
    sim1 = (a_emb * t1).sum(axis=1)
    sim2 = (a_emb * t2).sum(axis=1)
    sim0_shuf = (a_emb * np.roll(t0, 1, axis=0)).sum(axis=1)

    is_cjk = np.array(
        [bool(CJK_RE.search(it["s0"] + it["s1"] + it["s2"])) for it in items],
        dtype=bool,
    )
    en = ~is_cjk

    def pack(mask, tag):
        s0v, s1v, s2v = sim0[mask], sim1[mask], sim2[mask]
        sh = sim0_shuf[mask]
        d10 = s1v - s0v
        d20 = s2v - s0v
        d21 = s2v - s1v
        best = np.maximum(np.maximum(s0v, s1v), s2v)
        return {
            "tag": tag,
            "n": int(mask.sum()),
            "slot0": summarize(s0v),
            "slot1": summarize(s1v),
            "slot2": summarize(s2v),
            "best_of_3": summarize(best),
            "mean_of_3": summarize((s0v + s1v + s2v) / 3),
            "shuffled_slot0": summarize(sh),
            "delta_s1_minus_s0": {
                **summarize(d10),
                "frac_positive": float((d10 > 0).mean()),
                "ci95": list(bootstrap_mean_ci(d10)),
            },
            "delta_s2_minus_s0": {
                **summarize(d20),
                "frac_positive": float((d20 > 0).mean()),
                "ci95": list(bootstrap_mean_ci(d20)),
            },
            "delta_s2_minus_s1": {
                **summarize(d21),
                "frac_positive": float((d21 > 0).mean()),
                "ci95": list(bootstrap_mean_ci(d21)),
            },
            "win_rate": {
                "s0_best": float(((s0v >= s1v) & (s0v >= s2v)).mean()),
                "s1_best": float(((s1v > s0v) & (s1v >= s2v)).mean()),
                "s2_best": float(((s2v > s0v) & (s2v > s1v)).mean()),
            },
            "slot0_ci95": list(bootstrap_mean_ci(s0v)),
            "slot1_ci95": list(bootstrap_mean_ci(s1v)),
            "slot2_ci95": list(bootstrap_mean_ci(s2v)),
        }

    def examples(sim, k=4, worst=False):
        order = np.argsort(sim)
        if not worst:
            order = order[::-1]
        out = []
        for i in order[:k]:
            out.append(
                {
                    "id": items[i]["id"],
                    "sim0": float(sim0[i]),
                    "sim1": float(sim1[i]),
                    "sim2": float(sim2[i]),
                    "s0": items[i]["s0"][:180],
                    "s1": items[i]["s1"][:180],
                    "s2": items[i]["s2"][:180],
                }
            )
        return out

    report = {
        "schema_version": 1,
        "protocol": "CLAP HTSAT-base music_speech_audioset_epoch_15; cosine(caption, first-10s-audio@16kHz)",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_requested": args.n,
        "n_loaded": len(items),
        "seed": args.seed,
        "slot2_pool": len(common),
        "cuda": bool(torch.cuda.is_available()),
        "mean_full_dur": float(np.mean(durs)),
        "n_cjk_any_slot": int(is_cjk.sum()),
        "all": pack(np.ones(len(items), dtype=bool), "all"),
        "english_only": pack(en, "english_only") if en.any() else None,
        "examples_highest_slot0": examples(sim0, 4, False),
        "examples_lowest_slot0": examples(sim0, 4, True),
        "examples_s1_biggest_drop": examples(sim1 - sim0, 4, True),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["all"], indent=2))
    if report["english_only"]:
        print("ENGLISH_ONLY")
        print(json.dumps(report["english_only"], indent=2))
    print("WROTE", args.out_json)


if __name__ == "__main__":
    main()
