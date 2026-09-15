#!/usr/bin/env python3
"""Quality audit of the slot4v2 corpus against slot0.

Checks every rewritten row for: digits, spelled-out numbers / units, typos (tokens that
never occur anywhere in slot0), non-ASCII, repeated words, article errors, dangling
fragments, punctuation damage, chat / instruction residue, very short or shrunken
captions, decade words used for tempo ("tempo in the seventies"), meter swaps,
content words lost, and words the rewrite invented. Prints counts plus examples and
writes a JSON report. --strict exits 1 on any hard failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite_slot4v2_no_digits as R  # noqa: E402

csv.field_size_limit(10**9)
DATA = Path("/mnt/HDD/kojiek/phase4_jamendo_data")

SPELLED = re.compile(
    r"\b(?:hundred|thousand|bpm|beats?\s+per\s+minute|hertz|khz|kilohertz|decibels?|db|lufs|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.I,
)
ARTICLE = re.compile(r"\ba\s+(?:[aeiou]\w*)\b|\ban\s+(?![aeiouh])\w+", re.I)
ARTICLE_OK = re.compile(r"\ba\s+(?:one|uni\w*|use\w*|euro\w*|u[bklrst]\w*)\b|\ban\s+(?:hour|honest|herb|[aeiou]|x|mp|fm|lfo|mc|r&b|sfx|ep\b|edm|mp)", re.I)
REPEAT = re.compile(r"\b(\w+)\s+\1\b", re.I)
PUNCT = re.compile(r"\s[,.;:!]|[,;:]\s*[.,;:]|\.\.|\s{2,}|\(\s*\)|^\s|\s$|^[a-z]")
RESIDUE = re.compile(
    r"\?|\n|\t|\"|\b(?:none|rewrite|sentence|input|output|caption|user|assistant|human|system)\s*:|"
    r"^\s*none\s*\.?\s*$|\bNONE\b|\b(?:answer|translate|instruction|here is|as an ai)\b",
    re.I,
)
DECADE = re.compile(r"\b(?:eighties|nineties|seventies|sixties|fifties|two-thousands)\b", re.I)
TEMPO = re.compile(r"\b(?:tempo|bpm|pace|beats)\b", re.I)


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8", newline=""), delimiter="\t"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tsv", type=Path, default=DATA / "phase8_qwen_caption10s_multisent_train.tsv")
    ap.add_argument("--dst-tsv", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    src = load(args.src_tsv)
    dst = load(args.dst_tsv)
    assert len(src) == len(dst), (len(src), len(dst))

    vocab = Counter()
    for r in src:
        vocab.update(R.words(r["caption"]))

    hits: dict[str, list] = defaultdict(list)
    stats = Counter()
    lost_words = Counter()
    new_words = Counter()

    def flag(name, rid, a, b, detail=""):
        hits[name].append({"id": rid, "detail": detail, "src": a, "dst": b})

    for a_row, b_row in zip(src, dst):
        rid = a_row["id"]
        if rid != b_row["id"]:
            flag("id_mismatch", rid, a_row["id"], b_row["id"])
            continue
        a, b = a_row["caption"], b_row["caption"]
        if not R.has_digit(a):
            if a != b:
                flag("digit_free_row_changed", rid, a, b)
            continue
        stats["rewritten_rows"] += 1
        if a == b:
            flag("digit_row_unchanged", rid, a, b)
        if R.has_digit(b):
            flag("digit", rid, a, b)
        if any(ord(ch) > 127 for ch in b) and not any(ord(ch) > 127 for ch in a):
            flag("new_non_ascii", rid, a, b)
        # spelled numbers / units that were not already in the source
        # Lexicon output of the source ("1960s" -> "sixties", "12-bar" -> "twelve-bar") is expected.
        expected = R.lexicon_tokens(a)
        sa = Counter(m.group(0).lower() for m in SPELLED.finditer(expected))
        for m in SPELLED.finditer(b):
            w = m.group(0).lower()
            if sa[w] <= 0:
                flag("spelled_number_or_unit", rid, a, b, w)
            sa[w] -= 1
        bw, aw = R.words(b), R.words(a)
        exp_words = set(R.words(expected))
        typos = sorted({w for w in bw if vocab[w] == 0 and w not in exp_words})
        if typos:
            flag("token_not_in_slot0_vocab", rid, a, b, ",".join(typos))
        rare = sorted({w for w in bw if vocab[w] <= 2 and w not in aw})
        if rare:
            flag("rare_token_not_in_src", rid, a, b, ",".join(rare))
        for m in REPEAT.finditer(b):
            if not REPEAT.search(a) or m.group(0).lower() not in a.lower():
                flag("repeated_word", rid, a, b, m.group(0))
        for m in ARTICLE.finditer(b):
            if ARTICLE_OK.match(m.group(0)) or m.group(0).lower() in a.lower():
                continue
            flag("article", rid, a, b, m.group(0))
        if len(R.DANGLING.findall(b)) > len(R.DANGLING.findall(a)):
            flag("dangling", rid, a, b, "; ".join(x.group(0) for x in R.DANGLING.finditer(b)))
        if len(PUNCT.findall(b)) > len(PUNCT.findall(a)):
            flag("punctuation", rid, a, b, "; ".join(repr(x.group(0)) for x in PUNCT.finditer(b)))
        if not b.rstrip().endswith((".", "!")) and a.rstrip().endswith((".", "!")):
            flag("no_terminal_punct", rid, a, b)
        if len(RESIDUE.findall(b)) > len(RESIDUE.findall(a)):
            flag("chat_or_instruction_residue", rid, a, b)
        if len(bw) < 8:
            flag("short_caption_lt8_words", rid, a, b, str(len(bw)))
        if len(bw) < 0.6 * len(aw):
            flag("shrunk_below_60pct", rid, a, b, f"{len(bw)}/{len(aw)}")
        if R.TEMPO_DECADE.search(a) and len(DECADE.findall(b)) > len(DECADE.findall(a)):
            flag("decade_word_for_tempo", rid, a, b)
        inv = Counter(R.TEMPO_ADJ.findall(b.lower()))
        inv.subtract(Counter(R.TEMPO_ADJ.findall(a.lower())))
        if any(v > 0 for v in inv.values()):
            flag("invented_tempo_word", rid, a, b, ",".join(k for k, v in inv.items() if v > 0))
        if re.search(r"(?:^|[.!]\s+)[a-z]", b) and not re.search(r"(?:^|[.!]\s+)[a-z]", a):
            flag("lowercase_sentence_start", rid, a, b)
        if re.search(r"\w+-on-the-floor", b, re.I) and not re.search(r"on[\s\-]*the[\s\-]*floor", a, re.I):
            flag("meter_swap", rid, a, b)
        lost = R.content_words(a) - set(bw)
        lost_words.update(lost)
        if lost:
            stats["rows_with_content_loss"] += 1
            if len(lost) >= 3:
                flag("content_loss_ge3_words", rid, a, b, ",".join(sorted(lost)))
        new = set(bw) - set(aw) - R.STOP
        new_words.update(new)
        stats["src_words"] += len(aw)
        stats["dst_words"] += len(bw)

    hard = ["id_mismatch", "digit_free_row_changed", "digit", "new_non_ascii",
            "chat_or_instruction_residue", "no_terminal_punct", "meter_swap", "dangling",
            "repeated_word", "spelled_number_or_unit", "decade_word_for_tempo", "punctuation",
            "invented_tempo_word", "lowercase_sentence_start"]
    # Soft (reviewed by hand 2026-09-14, all false positives): "article" fires on note names
    # ("a mix of A and D#"), "token_not_in_slot0_vocab" on correct inflections ("adheres").
    report = {
        "stats": dict(stats),
        "counts": {k: len(v) for k, v in sorted(hits.items())},
        "top_lost_content_words": lost_words.most_common(40),
        "top_new_words": new_words.most_common(60),
        "examples": {k: v[: args.examples] for k, v in hits.items()},
        "hard_failures": {k: len(hits[k]) for k in hard if hits.get(k)},
    }
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"stats": report["stats"], "counts": report["counts"]}, indent=1))
    print("top_new_words", report["top_new_words"][:40])
    print("top_lost_content_words", report["top_lost_content_words"][:25])
    for k, v in hits.items():
        print(f"\n=== {k} ({len(v)})")
        for e in v[: args.examples]:
            print(f"  [{e['detail']}]\n   SRC: {e['src']}\n   DST: {e['dst']}")
    if args.strict and report["hard_failures"]:
        print("[FAIL] hard failures:", report["hard_failures"])
        sys.exit(1)


if __name__ == "__main__":
    main()
