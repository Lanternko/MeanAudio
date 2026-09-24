#!/usr/bin/env python3
"""Qwen2.5-Omni captions on first 10s crop — multi-sentence, leak-cleaned.

Fair-compare twin of gen_qwen_caption_10s_crop.py (one-sentence + max80 + first_sentence):
same audio window, same model, same id set; only prompt / decode / postprocess differ.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
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
VARIANT = "multisent_max160_stop_clean_v1"

# Explicit: multi-sentence + ONLY caption (anti chat-leak)
PROMPT = (
    "Listen carefully to this music clip. Write a rich, detailed caption "
    "in 2-5 sentences describing instruments, arrangement, mood, tempo, genre, "
    "and production quality (e.g. reverb, mix, recording fidelity) if audible. "
    "Output ONLY the caption text. Do not write questions, dialogue, code, "
    "math, or any text after the caption."
)

# Strings that indicate the model left caption mode
STOP_STRINGS = [
    "Human:",
    "\nHuman",
    "Assistant:",
    "User:",
    "System:",
    "\nCompute the",
    "\nWrite a Python",
    "\nFill in the",
]

LEAK_SPLIT = [
    "\nHuman:",
    "\nAssistant:",
    "\nUser:",
    "\nSystem:",
    "Human:",
    "Assistant:",
    "User:",
    "System:",
    "\nCompute the",
    "\nWrite a Python",
    "\nFill in the",
    "\nWhat is the sum",
    "\nHow many sides",
    "\nSolve ",
]

LEAK_LINE_RE = re.compile(
    r"^(Human|Assistant|User|System)\s*:|"
    r"^(Compute|Write a Python|Fill in|What is the sum|How many sides|"
    r"Solve |def |import |```)",
    re.I,
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


def clean_caption(text: str) -> tuple[str, bool]:
    """Return (cleaned, was_leaky). Never applies first_sentence truncation."""
    raw = (text or "").strip()
    if not raw:
        return "", False
    t = raw
    leaked = False
    for m in LEAK_SPLIT:
        if m in t:
            t = t.split(m, 1)[0]
            leaked = True
    kept: list[str] = []
    for ln in t.splitlines():
        if LEAK_LINE_RE.match(ln.strip()):
            leaked = True
            break
        kept.append(ln)
    t = "\n".join(kept).strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    # if clean emptied but raw had content, mark leak
    if not t and raw:
        leaked = True
    return t, leaked


def n_sents(text: str) -> int:
    parts = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(parts)


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
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


def load_crop(cid: str):
    path = id_to_audio_path(cid)
    if not path.exists():
        raise FileNotFoundError(str(path))
    # Only decode the first 10s — these files are ~30s; full decode was stalling the GPU.
    full, _ = librosa.load(str(path), sr=SR, mono=True, duration=WINDOW_SEC)
    crop = np.asarray(full, dtype=np.float32)
    if crop.shape[0] < WINDOW_SAMPLES:
        crop = np.pad(crop, (0, WINDOW_SAMPLES - crop.shape[0]))
    elif crop.shape[0] > WINDOW_SAMPLES:
        crop = crop[:WINDOW_SAMPLES]
    return path, crop


def resolve_stop_ids(processor):
    """Qwen2_5OmniThinker ships generation_config.eos_token_id = None, so generate()
    never stops and runs to max_new_tokens, then continues into a new chat turn.
    Resolve the ids explicitly and assert them — silently missing ids are the bug
    that contaminated ~40% of the 2026-08-09 multisent corpus."""
    tok = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
    assert tok is not None, "no tokenizer on processor — cannot resolve stop ids"
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id
    assert eos_id is not None and eos_id >= 0, f"<|im_end|> id unresolved: {eos_id!r}"
    assert pad_id is not None and pad_id >= 0, f"pad_token_id unresolved: {pad_id!r}"
    return tok, eos_id, pad_id


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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Model ready", flush=True)
    return model, processor


@torch.inference_mode()
def caption_batch(model, processor, paths, crops, seed: int, max_new_tokens: int,
                  temperature: float = 0.8, prompt: str | None = None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    prompt_text = prompt if prompt is not None else PROMPT
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": p},
                    {"type": "text", "text": prompt_text},
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
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tok,
    )

    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    # Truncate at the first stop id. batch_decode(skip_special_tokens=True) DELETES
    # <|im_end|> rather than cutting there, which is what glued caption + next-turn
    # junk together across a newline in the pre-fix corpus.
    captions = []
    for row in generated_ids.tolist():
        stop = next((j for j, t in enumerate(row) if t in (eos_id, pad_id)), None)
        real = row if stop is None else row[:stop]
        captions.append(tok.decode(real, skip_special_tokens=True).strip())
    return captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids_from_jsonl", type=Path, default=None,
                    help="Reuse exact id order from a prior pilot jsonl (fair re-pilot)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_seed", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--variant", type=str, default=VARIANT)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.ids_from_jsonl is not None:
        ids = []
        with args.ids_from_jsonl.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("id"):
                    ids.append(rec["id"])
        print(f"ids_from_jsonl n={len(ids)} src={args.ids_from_jsonl}", flush=True)
    else:
        with args.tsv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        ids = [r["id"] for r in rows]
        if args.limit is not None:
            pool = ids[:]
            if args.shuffle_seed is not None:
                rng = random.Random(args.shuffle_seed)
                rng.shuffle(pool)
            ids = pool[: args.limit]
            print(
                f"subset limit={args.limit} shuffle_seed={args.shuffle_seed}",
                flush=True,
            )

    done = load_done(args.out_jsonl) if args.resume else set()
    if not args.resume and args.out_jsonl.exists() and (args.limit is not None or args.ids_from_jsonl):
        args.out_jsonl.unlink()
        done = set()
    todo = [i for i in ids if i not in done]
    print(
        f"target={len(ids)} done={len(done)} todo={len(todo)} "
        f"max_new_tokens={args.max_new_tokens} variant={args.variant} T={args.temperature}",
        flush=True,
    )
    if not todo:
        print("Nothing to do", flush=True)
        return

    model, processor = load_model()
    def _one(cid):
        try:
            path, crop = load_crop(cid)
            return cid, path, crop, None
        except Exception as e:
            return cid, None, None, str(e)

    n_ok = n_err = n_leak = 0
    mode = "a" if args.resume and args.out_jsonl.exists() else "w"
    batch_starts = list(range(0, len(todo), args.batch_size))
    prefetch_workers = min(16, max(8, args.batch_size))
    pool = ThreadPoolExecutor(max_workers=prefetch_workers)

    def submit_batch(ids):
        return [pool.submit(_one, cid) for cid in ids]

    pending = submit_batch(todo[batch_starts[0] : batch_starts[0] + args.batch_size])
    with args.out_jsonl.open(mode, encoding="utf-8") as fout:
        for bi, i in enumerate(tqdm(batch_starts, desc="caption10s-ms")):
            if bi + 1 < len(batch_starts):
                nxt = todo[batch_starts[bi + 1] : batch_starts[bi + 1] + args.batch_size]
                nxt_pending = submit_batch(nxt)
            else:
                nxt_pending = None
            results = [fut.result() for fut in pending]
            pending = nxt_pending
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
                                "variant": args.variant,
                            }
                        )
                        + "\n"
                    )
                    n_err += 1
                else:
                    valid_paths.append(str(path))
                    valid_crops.append(crop)
                    valid_meta.append((cid, str(path)))

            if not valid_meta:
                continue

            try:
                raws = caption_batch(
                    model,
                    processor,
                    valid_paths,
                    valid_crops,
                    seed=args.seed + i,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    prompt=args.prompt,
                )
            except Exception as e:
                for cid, path in valid_meta:
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": None,
                                "error": f"generate: {e}",
                                "window_sec": WINDOW_SEC,
                                "variant": args.variant,
                            }
                        )
                        + "\n"
                    )
                    n_err += 1
                continue

            for (cid, path), raw in zip(valid_meta, raws):
                cap, leaked = clean_caption(raw)
                if leaked:
                    n_leak += 1
                if not cap:
                    fout.write(
                        json.dumps(
                            {
                                "id": cid,
                                "caption": None,
                                "caption_raw": raw,
                                "error": "empty_after_clean",
                                "leaked": leaked,
                                "window_sec": WINDOW_SEC,
                                "variant": args.variant,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_err += 1
                    continue
                rec = {
                    "id": cid,
                    "caption": cap,
                    "caption_raw": raw if leaked else None,
                    "leaked": leaked,
                    "n_chars": len(cap),
                    "n_words": len(cap.split()),
                    "n_sents": n_sents(cap),
                    "audio_path": path,
                    "window_sec": WINDOW_SEC,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt": (args.prompt or PROMPT),
                    "variant": args.variant,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()
    pool.shutdown(wait=True)

    print(
        f"DONE ok={n_ok} err={n_err} leak_flagged={n_leak} "
        f"leak_rate={n_leak / max(n_ok + n_leak, 1):.1%} out={args.out_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
