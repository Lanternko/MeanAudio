#!/usr/bin/env python3
"""Rewrite Music Flamingo captions into short MeanAudio-friendly captions.

This is a deterministic semantic compressor rather than a hard truncator. It
keeps concrete acoustic terms early in the caption and removes long production
commentary that tends to consume the 77-token MeanAudio text window.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path


DEFAULT_IN = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k/caption.jsonl")
DEFAULT_OUT = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k_short_rewrite/caption.jsonl")

GENRES = [
    "ambient-techno",
    "thrash-metal",
    "heavy-metal",
    "pop-rock",
    "hard-rock",
    "folk-pop",
    "funk-soul",
    "downtempo",
    "electronic",
    "classical",
    "orchestral",
    "cinematic",
    "ambient",
    "minimalist",
    "techno",
    "metal",
    "rock",
    "pop",
    "folk",
    "funk",
    "soul",
    "jazz",
    "blues",
    "hip-hop",
    "trap",
    "edm",
    "indie",
    "shoegaze",
    "experimental",
]

INSTRUMENTS = [
    "distorted electric guitar",
    "clean electric guitar",
    "electric guitar",
    "acoustic guitar",
    "bass guitar",
    "synth bass",
    "bass line",
    "electronic drum",
    "drum pattern",
    "drum groove",
    "kick drum",
    "snare",
    "hand clap",
    "percussion",
    "shaker",
    "arpeggiated synth",
    "bell-like synth",
    "synth melody",
    "synth pad",
    "synth",
    "piano",
    "organ",
    "strings",
    "string section",
    "violin",
    "cello",
    "brass",
    "woodwind",
    "flute",
    "saxophone",
    "vocal",
    "choir",
    "guitar riff",
    "guitar",
    "bass",
    "drums",
]

RHYTHM_TERMS = [
    "fast-paced",
    "mid-tempo",
    "slow",
    "steady",
    "driving",
    "syncopated",
    "punchy",
    "repetitive",
    "danceable",
    "groove",
    "pulse",
    "beat",
]

MOODS = [
    "calm",
    "reflective",
    "relaxed",
    "introspective",
    "aggressive",
    "fierce",
    "intense",
    "energetic",
    "uplifting",
    "melancholic",
    "dreamy",
    "suspenseful",
    "dramatic",
    "atmospheric",
    "hypnotic",
    "sleek",
    "raw",
    "warm",
    "bright",
    "dark",
    "gentle",
    "smooth",
]

DROP_PHRASES = [
    "short excerpt",
    "short audio excerpt",
    "short audio segment",
    "10-second excerpt",
    "10-second audio slice",
    "10-second segment",
    "high-fidelity",
    "stereo image",
    "stereo field",
    "left-right field",
    "full song structure",
    "verses",
    "choruses",
    "bridges",
    "suitable for",
]


def normalize(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ordered_matches(text: str, terms: Iterable[str], limit: int) -> list[str]:
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for term in terms:
        pattern = r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])"
        match = re.search(pattern, low)
        if match:
            hits.append((match.start(), term))
    seen = set()
    out = []
    for _, term in sorted(hits):
        key = term.lower()
        if key in seen:
            continue
        if any(key in kept.lower() for kept in out):
            continue
        out = [kept for kept in out if kept.lower() not in key]
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def sentence_with(text: str, terms: Iterable[str], max_words: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        low = sent.lower()
        if any(term in low for term in terms):
            sent = re.sub(r"\b(The|This|It)\s+(short\s+)?(excerpt|segment|clip)\s+(is|features)\s+", "", sent, flags=re.I)
            sent = re.sub(r"\s+", " ", sent).strip(" .")
            words = sent.split()
            if len(words) > max_words:
                sent = " ".join(words[:max_words]).rstrip(",;:")
            return sent
    return ""


def join_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def trim_to_token_limit(text: str, tokenizer, max_tokens: int, min_words: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if tokenizer is None:
        words = text.split()
        return " ".join(words[:max(min_words, 60)])

    while len(tokenizer(text, add_special_tokens=True).input_ids) > max_tokens:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            text = " ".join(sentences[:-1]).strip()
        else:
            words = text.split()
            if len(words) <= min_words:
                break
            text = " ".join(words[:-5]).rstrip(",;:")
        text = text.rstrip(" .") + "."
    return text


def phrase_or_default(terms: list[str], default: str) -> str:
    return join_terms(terms) if terms else default


def format_genres(terms: list[str]) -> str:
    terms = ["EDM" if term == "edm" else term for term in terms]
    return phrase_or_default(terms, "instrumental music")


def sentence_case_preserve(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def format_rhythm(terms: list[str]) -> str:
    if not terms:
        return "The rhythm is built around a steady pulse."
    nouns = [term for term in terms if term in {"groove", "pulse", "beat"}]
    adjectives = [term for term in terms if term not in {"groove", "pulse", "beat"}]
    if nouns:
        adj = " ".join(adjectives[:2])
        noun = nouns[0]
        phrase = f"{adj} {noun}".strip()
        return f"The rhythm is built around a {phrase}."
    return f"The rhythm feels {join_terms(adjectives[:3])}."


def compress_caption(text: str, tokenizer=None, max_tokens: int = 77) -> str:
    text = normalize(text)

    genres = ordered_matches(text, GENRES, 3)
    instruments = ordered_matches(text, INSTRUMENTS, 5)
    moods = ordered_matches(text, MOODS, 3)
    rhythm_terms = ordered_matches(text, RHYTHM_TERMS, 3)

    is_no_vocals = any(p in text.lower() for p in ["no vocals", "without any vocal", "without vocals"])
    has_vocals = bool(re.search(r"\bvocal(s|ist)?\b|\bsinging\b|\bvoice\b", text.lower())) and not is_no_vocals
    if is_no_vocals:
        instruments = [t for t in instruments if t != "vocal"]

    genre_phrase = format_genres(genres)
    instrument_phrase = phrase_or_default(instruments, "layered instruments")
    mood_phrase = phrase_or_default(moods, "focused")
    vocal_phrase = "No vocals are present" if is_no_vocals else ("Vocals are present" if has_vocals else "")

    parts = [
        f"{sentence_case_preserve(genre_phrase)} track featuring {instrument_phrase}.",
        format_rhythm(rhythm_terms),
    ]
    if vocal_phrase:
        parts.append(vocal_phrase + ".")
    parts.append(f"The mood is {mood_phrase}.")

    caption = " ".join(parts)
    caption = re.sub(r"\bfeaturing ([^.]+) featuring\b", r"featuring \1 with", caption, flags=re.I)
    caption = re.sub(r"\s+", " ", caption).strip()
    return trim_to_token_limit(caption, tokenizer, max_tokens=max_tokens, min_words=25)


def load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:
        print(f"[warn] tokenizer unavailable ({exc}); falling back to word trimming")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--max-t5-tokens", type=int, default=77)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = ok = over = 0
    with args.input_jsonl.open() as fin, args.output_jsonl.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            raw = ((rec.get("output") or {}).get("text") or rec.get("raw_text") or "").strip()
            if rec.get("ok") and raw:
                short = compress_caption(raw, tokenizer=tokenizer, max_tokens=args.max_t5_tokens)
                ok += 1
            else:
                short = ""
            if tokenizer and short and len(tokenizer(short, add_special_tokens=True).input_ids) > args.max_t5_tokens:
                over += 1

            out = dict(rec)
            out["raw_text_long"] = raw
            out["raw_text"] = short
            out["output"] = {"text": short}
            out["prompt_version"] = "slice10_short_rewrite_v1"
            out["rewrite_source_model"] = rec.get("model")
            out["rewrite_method"] = "deterministic_acoustic_compressor_v1"
            out["ok"] = bool(short)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"total={total}")
    print(f"ok={ok}")
    print(f"over_{args.max_t5_tokens}_t5_tokens={over}")
    print(f"output_jsonl={args.output_jsonl}")


if __name__ == "__main__":
    main()
