#!/usr/bin/env python3
"""Pilot: 10s crop captions WITHOUT one-sentence / first_sentence truncation.

Compare length vs current caption10s (one-sentence + max_new_tokens=80).
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
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SR = 16000
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
# Multi-sentence, no forced single-sentence, encourage detail + production
PROMPT = (
    "Listen carefully to this music clip. Write a rich, detailed caption "
    "in 2-5 sentences describing instruments, arrangement, mood, tempo, genre, "
    "and production quality (e.g. reverb, mix, recording fidelity) if audible. "
    "Do not limit yourself to a single sentence."
)
AUDIO_ROOT = Path("/mnt/HDD/hsiehyian/segments_no_vocals")
DEFAULT_TSV = Path(
    "/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched.tsv"
)


def id_to_audio_path(clip_id: str) -> Path:
    parts = clip_id.split("_")
    seg_idx = parts.index("segment")
    artist = "_".join(parts[: seg_idx - 1])
    track = parts[seg_idx - 1]
    seg_num = parts[seg_idx + 1]
    return AUDIO_ROOT / artist / track / f"segment_{seg_num}.mp3"


def load_crop(cid: str):
    path = id_to_audio_path(cid)
    if not path.exists():
        raise FileNotFoundError(str(path))
    full, _ = librosa.load(str(path), sr=SR, mono=True)
    full = np.asarray(full, dtype=np.float32)
    crop = full[:WINDOW_SAMPLES]
    if crop.shape[0] < WINDOW_SAMPLES:
        crop = np.pad(crop, (0, WINDOW_SAMPLES - crop.shape[0]))
    return path, crop


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

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
def caption_batch(model, processor, paths, crops, seed: int, max_new_tokens: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
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
        audio=crops,
        return_tensors="pt",
        padding=True,
        sampling_rate=SR,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
    )
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # NO first_sentence truncation
    return [(c or "").strip() for c in captions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = [r["id"] for r in rows]
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(ids)
    ids = ids[: args.limit]
    print(
        f"pilot n={len(ids)} max_new_tokens={args.max_new_tokens} "
        f"prompt=multisent NO first_sentence",
        flush=True,
    )

    model, processor = load_model()
    n_ok = n_err = 0
    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ids), args.batch_size), desc="caption10s-multi"):
            batch_ids = ids[i : i + args.batch_size]
            paths, crops, meta = [], [], []

            def _one(cid):
                try:
                    path, crop = load_crop(cid)
                    return cid, path, crop, None
                except Exception as e:
                    return cid, None, None, str(e)

            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(_one, batch_ids))

            valid_paths, valid_crops, valid_meta = [], [], []
            for cid, path, crop, err in results:
                if err is not None:
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": None,
                                "error": err,
                                "window_sec": WINDOW_SEC,
                                "variant": "multisent_max256",
                            }
                        )
                        + "\n"
                    )
                    n_err += 1
                else:
                    valid_paths.append(str(path))
                    valid_crops.append(crop)
                    valid_meta.append(cid)

            if not valid_paths:
                continue
            try:
                caps = caption_batch(
                    model,
                    processor,
                    valid_paths,
                    valid_crops,
                    seed=args.seed + i,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                for cid in valid_meta:
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": None,
                                "error": f"generate: {e}",
                                "window_sec": WINDOW_SEC,
                                "variant": "multisent_max256",
                            }
                        )
                        + "\n"
                    )
                    n_err += 1
                continue

            for cid, cap, path in zip(valid_meta, caps, valid_paths):
                n_sent = max(1, cap.count(".") + cap.count("!") + cap.count("?"))
                # crude sentence count
                import re

                sents = [s for s in re.split(r"[.!?]+", cap) if s.strip()]
                rec = {
                    "id": cid,
                    "caption": cap,
                    "n_chars": len(cap),
                    "n_words": len(cap.split()),
                    "n_sents": len(sents),
                    "audio_path": path,
                    "window_sec": WINDOW_SEC,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt": PROMPT,
                    "variant": "multisent_max256_no_first_sentence",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()

    print(f"DONE ok={n_ok} err={n_err} out={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
