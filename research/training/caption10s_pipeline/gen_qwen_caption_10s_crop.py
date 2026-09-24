#!/usr/bin/env python3
"""Generate Qwen2.5-Omni captions on the FIRST 10s of each clip (training window).

Reads official matched TSV ids, loads segment audio, crops to 10s @16kHz,
writes jsonl {id, caption, audio_path, full_dur, crop_dur, window}.
Supports --limit for pilot and --resume for full runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
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
PROMPT = (
    "Write a detailed one-sentence caption describing this music, "
    "covering the main instruments, mood, tempo, and genre."
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


def first_sentence(s: str) -> str:
    s = (s or "").strip()
    idx = s.find(".")
    return (s[: idx + 1] if idx != -1 else s).strip()


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec.get("caption"):
                done.add(rec["id"])
    return done


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


def load_crop(cid: str):
    path = id_to_audio_path(cid)
    if not path.exists():
        raise FileNotFoundError(str(path))
    full, _ = librosa.load(str(path), sr=SR, mono=True)
    full = np.asarray(full, dtype=np.float32)
    crop = full[:WINDOW_SAMPLES]
    if crop.shape[0] < WINDOW_SAMPLES:
        crop = np.pad(crop, (0, WINDOW_SAMPLES - crop.shape[0]))
    return path, full, crop


def resolve_stop_ids(processor):
    """Qwen2_5OmniThinker ships generation_config.eos_token_id = None, so generate()
    never stops and runs to max_new_tokens, then continues into a new chat turn.
    Resolve the ids explicitly and assert them."""
    tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
    assert tok is not None, "no tokenizer on processor — cannot resolve stop ids"
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id
    assert eos_id is not None and eos_id >= 0, f"<|im_end|> id unresolved: {eos_id!r}"
    assert pad_id is not None and pad_id >= 0, f"pad_token_id unresolved: {pad_id!r}"
    return tok, eos_id, pad_id


@torch.inference_mode()
def caption_batch(model, processor, paths, crops, seed: int):
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
    tok, eos_id, pad_id = resolve_stop_ids(processor)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    # Truncate at the first stop id — skip_special_tokens deletes <|im_end|> instead
    # of cutting there, letting the next chat turn bleed into the caption.
    captions = []
    for row in generated_ids.tolist():
        stop = next((j for j, t in enumerate(row) if t in (eos_id, pad_id)), None)
        real = row if stop is None else row[:stop]
        captions.append(tok.decode(real, skip_special_tokens=True).strip())
    return [first_sentence(c) for c in captions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shuffle_seed", type=int, default=None,
                    help="If set with --limit, sample a seeded random subset")
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = [r["id"] for r in rows]

    # Optional fixed subset (pilot). Deterministic when shuffle_seed set.
    if args.limit is not None:
        pool = ids[:]
        if args.shuffle_seed is not None:
            rng = random.Random(args.shuffle_seed)
            rng.shuffle(pool)
        subset = pool[: args.limit]
        ids = subset
        print(f"subset limit={args.limit} shuffle_seed={args.shuffle_seed}", flush=True)

    done = load_done(args.out_jsonl) if args.resume else set()
    if not args.resume and args.out_jsonl.exists() and args.limit is not None:
        # fresh pilot: start clean
        args.out_jsonl.unlink()
        done = set()
    todo = [i for i in ids if i not in done]
    print(f"target={len(ids)} done={len(done)} todo={len(todo)}", flush=True)

    if not todo:
        print("Nothing to do", flush=True)
        return

    model, processor = load_model()
    n_ok = n_err = 0
    mode = "a" if args.resume and args.out_jsonl.exists() else "w"
    with args.out_jsonl.open(mode, encoding="utf-8") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="caption10s"):
            batch_ids = todo[i : i + args.batch_size]
            paths, crops, meta, valid = [], [], [], []

            def _one(cid):
                try:
                    path, full, crop = load_crop(cid)
                    return cid, path, full, crop, None
                except Exception as e:
                    return cid, None, None, None, str(e)

            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(_one, batch_ids))

            for cid, path, full, crop, err in results:
                if err is not None:
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": None,
                                "error": err,
                                "window_sec": WINDOW_SEC,
                            }
                        )
                        + "\n"
                    )
                    n_err += 1
                else:
                    paths.append(str(path))
                    crops.append(crop)
                    meta.append((cid, path, full))
                    valid.append(cid)

            if not valid:
                fout.flush()
                continue

            try:
                caps = caption_batch(
                    model, processor, paths, crops, seed=args.seed + i
                )
                for (cid, path, full), cap in zip(meta, caps):
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": cap,
                                "audio_path": str(path),
                                "full_dur": float(len(full) / SR),
                                "crop_dur": WINDOW_SEC,
                                "window_sec": WINDOW_SEC,
                                "model": MODEL_ID,
                                "prompt": PROMPT,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_ok += 1
            except Exception as e:
                tqdm.write(f"[WARN] batch fail: {e}; one-by-one")
                for cid, path, full, crop in [
                    (m[0], m[1], m[2], c) for m, c in zip(meta, crops)
                ]:
                    try:
                        cap = caption_batch(
                            model,
                            processor,
                            [str(path)],
                            [crop],
                            seed=args.seed + i,
                        )[0]
                        fout.write(
                            json.dumps(
                                {
                                    "id": cid,
                                    "caption": cap,
                                    "audio_path": str(path),
                                    "full_dur": float(len(full) / SR),
                                    "crop_dur": WINDOW_SEC,
                                    "window_sec": WINDOW_SEC,
                                    "model": MODEL_ID,
                                    "prompt": PROMPT,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_ok += 1
                    except Exception as e2:
                        fout.write(
                            json.dumps(
                                {
                                    "id": cid,
                                    "caption": None,
                                    "error": str(e2),
                                    "window_sec": WINDOW_SEC,
                                }
                            )
                            + "\n"
                        )
                        n_err += 1
            fout.flush()

    print(f"[DONE] ok={n_ok} err={n_err} out={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
