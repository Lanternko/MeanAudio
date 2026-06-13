#!/usr/bin/env python3
"""Generate Qwen2.5-Omni captions for exact 10-second slice WAVs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import torch
from tqdm import tqdm


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SR = 16_000
MAX_NEW_TOKENS = 60

PROMPTS = [
    "Write a detailed one-sentence caption describing this 10-second music clip, covering the main instruments, mood, tempo, and genre. Only describe what is audible in this clip.",
    "Summarize this 10-second music clip in one concise sentence that captures its main instruments, mood, and style. Only describe this clip.",
    "Describe this 10-second music clip in one sentence using rich and varied vocabulary, avoiding generic words. Only describe what you hear.",
    "In one flowing sentence, list the key musical attributes audible in this 10-second clip: genre, mood, instruments, tempo, and production style.",
    "In natural prose, describe in one sentence what you hear in this 10-second music clip, including the instruments and overall feel.",
]


def first_sentence(text: str) -> str:
    idx = text.find(".")
    return (text[: idx + 1] if idx != -1 else text).strip()


def read_review_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("caption"):
                done.add(rec["id"])
    return done


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def load_audio(path: str):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio


def run_slot(model, processor, rows: list[dict], slot: int, out_path: Path, batch_size: int, resume: bool) -> None:
    prompt = PROMPTS[slot]
    done = load_done(out_path) if resume else set()
    todo = [row for row in rows if row["id"] not in done]

    print(f"\n=== slot {slot} ===")
    print(f"prompt={prompt}")
    print(f"out={out_path}")
    print(f"done={len(done)} todo={len(todo)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_error = 0

    with out_path.open("a" if resume else "w") as fout:
        for i in tqdm(range(0, len(todo), batch_size), desc=f"qwen-slot{slot}"):
            batch = todo[i : i + batch_size]
            t0 = time.perf_counter()

            with ThreadPoolExecutor(max_workers=8) as ex:
                audios = list(ex.map(lambda r: load_audio(r["review_audio_path"]), batch))

            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": row["review_audio_path"]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                for row in batch
            ]

            try:
                texts = [
                    processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                    for conv in conversations
                ]
                with torch.no_grad():
                    inputs = processor(
                        text=texts,
                        audio=audios,
                        return_tensors="pt",
                        padding=True,
                        sampling_rate=SR,
                    ).to(model.device)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.8,
                    )
                    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
                    captions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                for row, cap in zip(batch, captions):
                    fout.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "caption": first_sentence(cap),
                                "slot": slot,
                                "review_audio_path": row["review_audio_path"],
                                "runtime_batch_sec": round(time.perf_counter() - t0, 3),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_ok += 1
            except Exception as e:
                tqdm.write(f"[WARN] slot {slot} batch error: {e}")
                for row in batch:
                    fout.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "caption": None,
                                "slot": slot,
                                "review_audio_path": row["review_audio_path"],
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_error += 1
            fout.flush()

    print(f"slot {slot} done: ok={n_ok} error={n_error}")


def load_slot_map(path: Path) -> dict[str, str | None]:
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[rec["id"]] = rec.get("caption")
    return rows


def write_outputs(rows: list[dict], out_dir: Path, prefix: str) -> None:
    slot_maps = [load_slot_map(out_dir / f"{prefix}_slot{s}.jsonl") for s in range(5)]
    merged_path = out_dir / f"{prefix}.jsonl"
    slot0_tsv = out_dir / "qwen_slot0.tsv"
    mean5_tsv = out_dir / "qwen_mean5.tsv"
    review_slot0 = out_dir / "review_qwen_slot0.tsv"
    review_mean5 = out_dir / "review_qwen_mean5.tsv"

    with merged_path.open("w") as fout:
        for row in rows:
            captions = [slot_maps[s].get(row["id"]) for s in range(5)]
            fout.write(json.dumps({"id": row["id"], "captions": captions}, ensure_ascii=False) + "\n")

    def write_caption_tsv(path: Path, pick):
        with path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "caption"])
            for row in rows:
                cap = pick(row["id"])
                if cap:
                    writer.writerow([row["id"], cap])

    def write_review_tsv(path: Path, pick):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "track_id",
                    "source_audio_path",
                    "review_audio_path",
                    "slice_start_sec",
                    "slice_duration_sec",
                    "music_flamingo_slice10_caption",
                    "lpmc_caption",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for row in rows:
                cap = pick(row["id"])
                if not cap:
                    continue
                out = dict(row)
                out["music_flamingo_slice10_caption"] = cap
                writer.writerow(out)

    write_caption_tsv(slot0_tsv, lambda sid: slot_maps[0].get(sid))
    write_caption_tsv(mean5_tsv, lambda sid: " ".join(c for c in [slot_maps[s].get(sid) for s in range(5)] if c))
    write_review_tsv(review_slot0, lambda sid: slot_maps[0].get(sid))
    write_review_tsv(review_mean5, lambda sid: " ".join(c for c in [slot_maps[s].get(sid) for s in range(5)] if c))

    print(f"merged={merged_path}")
    print(f"slot0_tsv={slot0_tsv}")
    print(f"mean5_tsv={mean5_tsv}")
    print(f"review_slot0={review_slot0}")
    print(f"review_mean5={review_mean5}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="qwen_slice10_400")
    parser.add_argument("--slot", default="all", help='0-4, "all", or "merge"')
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_review_rows(args.review_tsv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.slot == "merge":
        write_outputs(rows, args.out_dir, args.prefix)
        return

    model, processor = load_model()
    slots = range(5) if args.slot == "all" else [int(args.slot)]
    for slot in slots:
        run_slot(
            model,
            processor,
            rows,
            slot,
            args.out_dir / f"{args.prefix}_slot{slot}.jsonl",
            args.batch_size,
            args.resume,
        )

    if args.slot == "all":
        write_outputs(rows, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
