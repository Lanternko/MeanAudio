#!/usr/bin/env python3
"""
Pilot: Qwen2.5-Omni captions on first-10s crop vs full-30s audio.
Compares against existing official matched captions (often track/30s-level).

Outputs:
  - jsonl with old / new10s / new30s captions
  - comparison TSV
  - CLAP scores: caption × audio_0_10s  (and caption × full30s for ref)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SR = 16000
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
PROMPT = (
    "Write a detailed one-sentence caption describing this music, "
    "covering the main instruments, mood, tempo, and genre."
)
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
OFFICIAL_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv")
CLAP_CKPT = Path(
    "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
)
OUT_DIR = Path("/home/kojiek/research/meanaudio_training/outputs/caption10s_pilot")


def id_to_audio_path(clip_id: str) -> Path:
    """00_1014400_segment_2_0 → AUDIO_ROOT/00/1014400/segment_2.mp3"""
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def first_sentence(s: str) -> str:
    s = (s or "").strip()
    idx = s.find(".")
    return (s[: idx + 1] if idx != -1 else s).strip()


def load_audio_pair(cid: str):
    path = id_to_audio_path(cid)
    if not path.exists():
        raise FileNotFoundError(str(path))
    full, _ = librosa.load(str(path), sr=SR, mono=True)
    full = np.asarray(full, dtype=np.float32)
    crop = full[:WINDOW_SAMPLES]
    if crop.shape[0] < WINDOW_SAMPLES:
        crop = np.pad(crop, (0, WINDOW_SAMPLES - crop.shape[0]))
    return {
        "id": cid,
        "path": str(path),
        "full": full,
        "crop10": crop,
        "full_dur": float(len(full) / SR),
        "crop_dur": float(len(crop) / SR),
    }


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    print(f"Loading {MODEL_ID}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()
    print("Model ready", flush=True)
    return model, processor


@torch.inference_mode()
def caption_batch(model, processor, items, audio_key: str, seed: int):
    """items: list of dicts with audio_key arrays."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    audios = [it[audio_key] for it in items]
    paths = [it["path"] for it in items]
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": p},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        for p in paths
    ]
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in conversations
    ]
    inputs = processor(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=SR,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
    )
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [first_sentence(c) for c in captions]


def load_clap():
    import laion_clap

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt(str(CLAP_CKPT), verbose=False)
    clap.eval()
    return clap


@torch.inference_mode()
def clap_audio_text(clap, audios_16k: list[np.ndarray], texts: list[str]):
    # laion_clap expects list of audio arrays and texts
    a_emb = clap.get_audio_embedding_from_data(
        x=audios_16k, use_tensor=False
    )  # (B, D)
    t_emb = clap.get_text_embedding(texts, use_tensor=False)  # (B, D)
    a = np.asarray(a_emb, dtype=np.float32)
    t = np.asarray(t_emb, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    return (a * t).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--skip_clap", action="store_true")
    ap.add_argument("--skip_full30", action="store_true",
                    help="Only generate 10s captions (faster)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.n}_seed{args.seed}"
    out_jsonl = OUT_DIR / f"pilot_{tag}.jsonl"
    out_tsv = OUT_DIR / f"pilot_{tag}_compare.tsv"
    out_summary = OUT_DIR / f"pilot_{tag}_SUMMARY.json"

    # sample rows from official matched
    with OFFICIAL_TSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rng = random.Random(args.seed)
    # stratified-ish: shuffle then take first n that have audio
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)

    selected = []
    for i in idxs:
        cid = rows[i]["id"]
        path = id_to_audio_path(cid)
        if path.exists():
            selected.append(rows[i])
        if len(selected) >= args.n:
            break
    if len(selected) < args.n:
        raise SystemExit(f"Only found {len(selected)}/{args.n} clips with audio")

    print(f"Selected {len(selected)} clips", flush=True)
    print("Loading audio...", flush=True)
    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(load_audio_pair, r["id"]) for r in selected]
        for fut, row in zip(futs, selected):
            try:
                it = fut.result()
            except Exception as e:
                print(f"[WARN] skip {row['id']}: {e}", flush=True)
                continue
            it["old_caption"] = row["caption"]
            it["q_level"] = row.get("q_level")
            items.append(it)
    print(f"Loaded {len(items)} audios; mean full_dur={np.mean([i['full_dur'] for i in items]):.2f}s", flush=True)

    model, processor = load_model()

    # Generate captions
    for mode, key, skip in [
        ("cap10s", "crop10", False),
        ("cap30s", "full", args.skip_full30),
    ]:
        if skip:
            for it in items:
                it[mode] = None
            continue
        print(f"Generating {mode}...", flush=True)
        caps = []
        t0 = time.time()
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            try:
                batch_caps = caption_batch(
                    model, processor, batch, audio_key=key, seed=args.seed + i
                )
            except Exception as e:
                print(f"[WARN] batch {i} failed: {e}; retry one-by-one", flush=True)
                batch_caps = []
                for j, it in enumerate(batch):
                    try:
                        batch_caps.append(
                            caption_batch(
                                model, processor, [it], audio_key=key, seed=args.seed + i + j
                            )[0]
                        )
                    except Exception as e2:
                        print(f"[WARN] {it['id']} failed: {e2}", flush=True)
                        batch_caps.append(None)
            caps.extend(batch_caps)
            if (i // args.batch_size) % 5 == 0:
                print(f"  {mode} {min(i+args.batch_size,len(items))}/{len(items)}", flush=True)
        for it, c in zip(items, caps):
            it[mode] = c
        print(f"  {mode} done in {time.time()-t0:.1f}s", flush=True)

    # free GPU for CLAP
    del model, processor
    torch.cuda.empty_cache()

    # CLAP comparison
    clap_stats = {}
    if not args.skip_clap:
        print("Loading CLAP...", flush=True)
        clap = load_clap()
        pairs = [
            ("old_vs_10s_audio", "old_caption", "crop10"),
            ("cap10s_vs_10s_audio", "cap10s", "crop10"),
            ("cap30s_vs_10s_audio", "cap30s", "crop10"),
            ("old_vs_30s_audio", "old_caption", "full"),
            ("cap10s_vs_30s_audio", "cap10s", "full"),
            ("cap30s_vs_30s_audio", "cap30s", "full"),
        ]
        for name, cap_key, audio_key in pairs:
            valid = [
                it
                for it in items
                if it.get(cap_key) and it.get(audio_key) is not None
            ]
            if not valid:
                continue
            scores = []
            bs = 16
            for i in range(0, len(valid), bs):
                batch = valid[i : i + bs]
                s = clap_audio_text(
                    clap,
                    [b[audio_key] for b in batch],
                    [b[cap_key] for b in batch],
                )
                scores.extend(s.tolist())
            for it, s in zip(valid, scores):
                it.setdefault("clap", {})[name] = float(s)
            arr = np.asarray(scores, dtype=np.float32)
            clap_stats[name] = {
                "n": int(len(arr)),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
            }
            print(f"CLAP {name}: mean={arr.mean():.4f} median={np.median(arr):.4f}", flush=True)

    # write outputs
    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "id": it["id"],
                "path": it["path"],
                "full_dur": it["full_dur"],
                "old_caption": it["old_caption"],
                "cap10s": it.get("cap10s"),
                "cap30s": it.get("cap30s"),
                "clap": it.get("clap", {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "id",
            "full_dur",
            "old_caption",
            "cap10s",
            "cap30s",
            "clap_old_vs_10s_audio",
            "clap_cap10s_vs_10s_audio",
            "clap_cap30s_vs_10s_audio",
            "clap_delta_10s_minus_old",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for it in items:
            clap = it.get("clap", {})
            old10 = clap.get("old_vs_10s_audio")
            new10 = clap.get("cap10s_vs_10s_audio")
            delta = None
            if old10 is not None and new10 is not None:
                delta = new10 - old10
            w.writerow(
                {
                    "id": it["id"],
                    "full_dur": f"{it['full_dur']:.2f}",
                    "old_caption": it["old_caption"],
                    "cap10s": it.get("cap10s") or "",
                    "cap30s": it.get("cap30s") or "",
                    "clap_old_vs_10s_audio": "" if old10 is None else f"{old10:.4f}",
                    "clap_cap10s_vs_10s_audio": "" if new10 is None else f"{new10:.4f}",
                    "clap_cap30s_vs_10s_audio": ""
                    if clap.get("cap30s_vs_10s_audio") is None
                    else f"{clap['cap30s_vs_10s_audio']:.4f}",
                    "clap_delta_10s_minus_old": "" if delta is None else f"{delta:.4f}",
                }
            )

    # qualitative: top improved / top regressed
    deltas = []
    for it in items:
        clap = it.get("clap", {})
        if "old_vs_10s_audio" in clap and "cap10s_vs_10s_audio" in clap:
            deltas.append(
                (
                    clap["cap10s_vs_10s_audio"] - clap["old_vs_10s_audio"],
                    it,
                )
            )
    deltas.sort(key=lambda x: x[0], reverse=True)

    summary = {
        "n": len(items),
        "seed": args.seed,
        "prompt": PROMPT,
        "model": MODEL_ID,
        "window_sec": WINDOW_SEC,
        "mean_full_dur": float(np.mean([i["full_dur"] for i in items])),
        "clap_stats": clap_stats,
        "delta_cap10s_minus_old_on_10s_audio": None,
        "examples_most_improved": [],
        "examples_most_regressed": [],
        "outputs": {
            "jsonl": str(out_jsonl),
            "tsv": str(out_tsv),
        },
    }
    if deltas:
        darr = np.asarray([d for d, _ in deltas], dtype=np.float32)
        summary["delta_cap10s_minus_old_on_10s_audio"] = {
            "mean": float(darr.mean()),
            "median": float(np.median(darr)),
            "frac_positive": float((darr > 0).mean()),
            "p25": float(np.percentile(darr, 25)),
            "p75": float(np.percentile(darr, 75)),
        }
        for d, it in deltas[:5]:
            summary["examples_most_improved"].append(
                {
                    "id": it["id"],
                    "delta": float(d),
                    "old": it["old_caption"][:200],
                    "cap10s": (it.get("cap10s") or "")[:200],
                }
            )
        for d, it in deltas[-5:][::-1]:
            summary["examples_most_regressed"].append(
                {
                    "id": it["id"],
                    "delta": float(d),
                    "old": it["old_caption"][:200],
                    "cap10s": (it.get("cap10s") or "")[:200],
                }
            )

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[COMPLETE] {out_summary}")


if __name__ == "__main__":
    main()
