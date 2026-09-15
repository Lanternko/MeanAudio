#!/usr/bin/env python3
"""Rewrite c2p0 slot0 captions that contain digits until none remain.

Only those rows are sent to the model. Digit-free rows are copied verbatim.
Failed LLM attempts fall back to a lexicon (808→drum machine, 4/4→common time,
80s→eighties, …) then strip leftover digit runs. The slot0 surface form is
kept: no LP-MusicCaps template, no new instruments.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

csv.field_size_limit(10**9)

SRC_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DIGIT = re.compile(r"\d")
MAX_ATTEMPTS = 5
MAX_NEW_TOKENS = 128

SYSTEM = """\
Rewrite the music caption. Keep instruments, mood, genre, vocals, arrangement, \
and qualitative tempo (slow / moderate / fast). Output one caption only.

Hard rules:
- No digits of any kind (0-9). No BPM, time signatures, Hz, dB, years, durations, \
sample rates, octave numbers, or counts written as numerals.
- Replace 808/909 with "drum machine", 8-bit with "chiptune", 80s/90s/70s with \
"eighties"/"nineties"/"seventies", 4/4 with "common time", 3/4 with "waltz time", \
4-piece with "small band", "4 on the floor" with "four-on-the-floor".
- Do not add instruments, genres, or fidelity claims that were not in the input.
- Do not open with a template. Keep the original voice and similar length."""

FEW_SHOT = [
    (
        "The music is a rhythmic electronic piece, featuring synthesizer arpeggios, a strong bass line, and fast drum beats. The mood is energetic and futuristic. The tempo is fast and upbeat, with a consistent 120 BPM. The genre is electronic and techno.",
        "The music is a rhythmic electronic piece, featuring synthesizer arpeggios, a strong bass line, and fast drum beats. The mood is energetic and futuristic. The tempo is fast and upbeat. The genre is electronic and techno.",
    ),
    (
        "The music features a solo piano playing a poignant and reflective melody in a classical style, evoking a soft and melancholic mood. The performance is in the key of D major and has a tempo of 83.0 BPM, with a 4/4 time signature.",
        "The music features a solo piano playing a poignant and reflective melody in a classical style, evoking a soft and melancholic mood. The performance is in the key of D major at a moderate tempo in common time.",
    ),
    (
        "An instrumental Techno/Trance piece at 125 BPM in F minor, featuring steady, driving electronic drums and an 808 bass.",
        "An instrumental Techno/Trance piece at a driving tempo in F minor, featuring steady, driving electronic drums and a drum-machine bass.",
    ),
    (
        "The audio is a clip of a 24-bit/48kHz audio file featuring a romantic pop rock style with piano, drums, electric guitar, bass, and synthesizer. Reminiscent of the 1980s.",
        "The audio features a romantic pop rock style with piano, drums, electric guitar, bass, and synthesizer. Reminiscent of the eighties.",
    ),
]

# Longer patterns first.
LEXICON = [
    (re.compile(r"\b4[\s\-]*on[\s\-]*the[\s\-]*floor\b", re.I), "four-on-the-floor"),
    (re.compile(r"\b24[\s\-]*bit(?:/\s*48\s*khz)?\b", re.I), "high-resolution"),
    (re.compile(r"\b16[\s\-]*bit\b", re.I), "vintage digital"),
    (re.compile(r"\b8[\s\-]*bit\b", re.I), "chiptune"),
    (re.compile(r"\b44(?:\.1)?\s*khz\b", re.I), ""),
    (re.compile(r"\b48\s*khz\b", re.I), ""),
    (re.compile(r"\b(?:bpm\s*[=:]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*bpm)\b", re.I), ""),
    (re.compile(r"\btempo(?:\s+(?:of|at|around|is))?\s*(?:about\s+)?\d+(?:\.\d+)?\b", re.I), "moderate tempo"),
    (re.compile(r"\b12/8\b", re.I), "compound time"),
    (re.compile(r"\b9/8\b", re.I), "compound time"),
    (re.compile(r"\b6/8\b", re.I), "compound time"),
    (re.compile(r"\b5/4\b", re.I), "irregular meter"),
    (re.compile(r"\b7/8\b", re.I), "irregular meter"),
    (re.compile(r"\b4/4\b", re.I), "common time"),
    (re.compile(r"\b3/4\b", re.I), "waltz time"),
    (re.compile(r"\b2/4\b", re.I), "common time"),
    (re.compile(r"\b2/2\b", re.I), "cut time"),
    (re.compile(r"\b808s?\b", re.I), "drum machine"),
    (re.compile(r"\b909s?\b", re.I), "drum machine"),
    (re.compile(r"\b2000s\b", re.I), "two-thousands"),
    (re.compile(r"\b1980s\b", re.I), "eighties"),
    (re.compile(r"\b1990s\b", re.I), "nineties"),
    (re.compile(r"\b1970s\b", re.I), "seventies"),
    (re.compile(r"\b1960s\b", re.I), "sixties"),
    (re.compile(r"\b80s\b", re.I), "eighties"),
    (re.compile(r"\b90s\b", re.I), "nineties"),
    (re.compile(r"\b70s\b", re.I), "seventies"),
    (re.compile(r"\b60s\b", re.I), "sixties"),
    (re.compile(r"\b50s\b", re.I), "fifties"),
    (re.compile(r"\b00s\b", re.I), "two-thousands"),
    (re.compile(r"\b\d+[\s\-]*piece\b", re.I), "small"),
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:hz|khz|db|lufs|seconds?|secs?|minutes?|mins?|ms)\b", re.I), ""),
    (re.compile(r"\b[A-G](?:#|b)?\d\b"), lambda m: m.group(0)[:-1]),
    (re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I), ""),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), ""),
]


def has_digit(text: str) -> bool:
    return bool(DIGIT.search(text))


def clean_output(raw: str) -> str:
    text = raw.strip()
    if text.lower().startswith("output:"):
        text = text[len("output:") :].strip()
    stop = re.search(
        r"(?:\n| )(?:Human|Assistant|User|System)\s*:|"
        r"\nInput:|\nOutput:|\n\nInput:|---",
        text,
    )
    if stop:
        text = text[: stop.start()].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences and not sentences[-1].rstrip().endswith((".", "!", "?")):
        sentences = sentences[:-1]
    text = " ".join(sentences).strip()
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;")
    return text


def lexicon_strip(text: str) -> str:
    out = text
    for pattern, repl in LEXICON:
        out = pattern.sub(repl, out)
    out = DIGIT.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"[,;:]\s*([,.;:])", r"\1", out)
    return out.strip(" ,;")


def build_prompt(caption: str) -> str:
    lines = [SYSTEM, ""]
    for src, dst in FEW_SHOT:
        lines.append(f"Input: {src}")
        lines.append(f"Output: {dst}")
        lines.append("")
    lines.append(f"Input: {caption}")
    lines.append("Output:")
    return "\n".join(lines)


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["id"]] = row
    return done


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()
    return model, processor


def run_batch(model, processor, prompts: list[str], temperature: float) -> list[str]:
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in conversations
    ]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    new_ids = generated_ids[:, inputs.input_ids.size(1) :]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [clean_output(item) for item in raw]


def fallback(original: str, current: str) -> tuple[str, str]:
    stripped = lexicon_strip(current if current else original)
    if stripped and not has_digit(stripped) and len(stripped.split()) >= 8:
        return stripped, "lexicon"
    stripped = lexicon_strip(original)
    if not stripped or has_digit(stripped):
        raise RuntimeError(f"could not strip digits: {original[:180]!r}")
    return stripped, "lexicon_from_original"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-tsv", type=Path, default=SRC_TSV)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.src_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    done = load_done(args.log_jsonl) if args.resume else {}
    pending = [
        row
        for row in rows
        if has_digit(row["caption"]) and row["id"] not in done
    ]
    print(f"rows={len(rows)} digit_rows={sum(has_digit(r['caption']) for r in rows)} pending={len(pending)}")

    model = processor = None
    if pending:
        model, processor = load_model()
        work = [
            {"id": row["id"], "src": row["caption"], "current": row["caption"], "attempts": 0}
            for row in pending
        ]
        with args.log_jsonl.open("a", encoding="utf-8") as handle:
            for attempt in range(MAX_ATTEMPTS):
                if not work:
                    break
                temp = 0.3 + 0.15 * attempt
                still = []
                for offset in tqdm(
                    range(0, len(work), args.batch_size),
                    desc=f"rewrite attempt {attempt + 1}",
                ):
                    batch = work[offset : offset + args.batch_size]
                    outs = run_batch(
                        model,
                        processor,
                        [build_prompt(item["current"]) for item in batch],
                        temp,
                    )
                    for item, out in zip(batch, outs):
                        item["attempts"] = attempt + 1
                        if out and not has_digit(out) and len(out.split()) >= 8:
                            rec = {
                                "id": item["id"],
                                "method": "llm",
                                "attempts": item["attempts"],
                                "src": item["src"],
                                "dst": out,
                            }
                            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            handle.flush()
                            done[item["id"]] = rec
                        else:
                            if out:
                                item["current"] = out
                            still.append(item)
                work = still
            for item in work:
                caption, method = fallback(item["src"], item["current"])
                rec = {
                    "id": item["id"],
                    "method": method,
                    "attempts": item["attempts"],
                    "src": item["src"],
                    "dst": caption,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                done[item["id"]] = rec

    out_rows = []
    n_rewritten = n_copied = 0
    for row in rows:
        if row["id"] in done:
            caption = done[row["id"]]["dst"]
            n_rewritten += 1
        else:
            caption = row["caption"]
            n_copied += 1
        if has_digit(caption):
            raise SystemExit(f"[FAIL] digit remains in {row['id']}")
        out_rows.append({"id": row["id"], "caption": caption})

    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    manifest = {
        "src_tsv": str(args.src_tsv),
        "out_tsv": str(args.out_tsv),
        "rows": len(out_rows),
        "rewritten": n_rewritten,
        "copied_verbatim": n_copied,
        "digit_rows_out": 0,
    }
    args.out_tsv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
