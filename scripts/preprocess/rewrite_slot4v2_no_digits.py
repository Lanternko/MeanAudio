#!/usr/bin/env python3
"""slot4v2: strip every digit from c2p0 slot0 captions, editing only digit sentences.

Fixes the three defects of rewrite_slot4_no_digits.py (slot4, 050/052):
  1. generate() had no eos_token_id (Qwen2.5-Omni generation_config.eos_token_id is
     None), so outputs ran on into the next chat turn ("Given the current trend of
     incorporating artificial intelligence ...").
  2. The lexicon fallback stripped the last *LLM output* instead of the original, so
     the junk tail survived and bare deletions left "with a tempo of." fragments.
  3. The model rewrote whole captions and dropped 36-44% of digit-free sentences.

v2 works per sentence. Digit-free sentences are copied byte-identical (asserted).
Each digit sentence goes to the LLM alone and the output is accepted only if it
passes hard gates (no digits, one sentence, no chat/instruction markers, no dangling
function word, bounded length, at most two words not in the source). The model may
answer NONE when the sentence carries nothing but the number. Otherwise the
fallback edits the ORIGINAL sentence deterministically: token lexicon first, then
drop only the clauses that still hold a digit, and drop the sentence only when its
main clause is the number.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

csv.field_size_limit(10**9)

SRC_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv")
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DIGIT = re.compile(r"\d")
MAX_ATTEMPTS = 4
RESTRUCTURE_ATTEMPTS = 3
PASS3_ATTEMPTS = 3
MAX_NEW_TOKENS = 96
SENT_SPLIT = re.compile(r"(?<=[.!?])(\s+)")

SYSTEM = """\
Rewrite ONE sentence from a music caption so that it contains no digits.

Rules:
- Remove numeric measurements: BPM, beats per minute, time signatures, Hz, dB, \
years, durations, counts. Keep every other word and the original wording as much \
as possible, and keep the sentence grammatical.
- Replace 808/909 with "drum machine", 8-bit with "chiptune", 80s/90s/70s with \
"eighties"/"nineties"/"seventies", 4/4 with "common time", 3/4 with "waltz time", \
"4 on the floor" with "four-on-the-floor", small counts with words (12-bar -> twelve-bar).
- Do not invent a tempo word (slow / fast / moderate) that is not already in the sentence.
- Do not add instruments, genres, moods, or quality claims.
- If nothing is left once the number is removed, answer exactly NONE.
- Answer with the rewritten sentence only."""

FEW_SHOT = [
    ("The tempo is fast and upbeat, with a consistent 120 BPM.",
     "The tempo is fast and upbeat."),
    ("The tempo is around 135 BPM.", "NONE"),
    ("The performance is in the key of D major and has a tempo of 83.0 BPM, with a 4/4 time signature.",
     "The performance is in the key of D major, in common time."),
    ("An instrumental Techno/Trance piece at 125 BPM in F minor, featuring steady, driving electronic drums and an 808 bass.",
     "An instrumental Techno/Trance piece in F minor, featuring steady, driving electronic drums and a drum machine bass."),
    ("The mood is intense and trance-like, with a fast tempo of 141 beats per minute.",
     "The mood is intense and trance-like, with a fast tempo."),
    ("The sound is reminiscent of the 1980s synth-pop era, built on a 12-bar blues progression.",
     "The sound is reminiscent of the eighties synth-pop era, built on a twelve-bar blues progression."),
]

SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen", "twenty"]
ORDINAL_CENTURY = {"17": "seventeenth", "18": "eighteenth", "19": "nineteenth",
                   "20": "twentieth", "21": "twenty-first"}

_PAREN_DIGIT = re.compile(r"\s*\([^()]*\d[^()]*\)")

# Token-level edits only. Nothing here deletes a numeric clause: tempo/BPM phrases
# are removed by drop_digit_clauses, which cuts whole comma/and/with/at clauses.
LEXICON = [
    (re.compile(r"\b4[\s\-]*on[\s\-]*the[\s\-]*floor\b", re.I), "four-on-the-floor"),
    (re.compile(r"\bmid[\s\-]*(\d{2})(?:st|nd|rd|th)[\s\-]*century\b", re.I),
     lambda m: f"mid-{ORDINAL_CENTURY.get(m.group(1), 'past')}-century"),
    (re.compile(r"\b(\d{2})(?:st|nd|rd|th)[\s\-]*century\b", re.I),
     lambda m: f"{ORDINAL_CENTURY.get(m.group(1), 'past')}-century"),
    (re.compile(r"\b(?:a|an)\s+8[\s\-]*bit(?:\s+chiptune)?\b", re.I), "a chiptune"),
    (re.compile(r"\b8[\s\-]*bit(?:\s+chiptune)?\b", re.I), "chiptune"),
    (re.compile(r"\ban\s+(?:tr-?)?(?:808|909)s?\b", re.I), "a drum machine"),
    (re.compile(r"\b24[\s\-]*bit(?:\s*/\s*48\s*khz)?\b", re.I), "high-resolution"),
    (re.compile(r"\b16[\s\-]*bit\b", re.I), "sixteen-bit"),
    # Keep the article and "signature": "a 4/4 time signature" -> "a common time signature".
    (re.compile(r"\b(?:12/8|9/8|6/8)(?:\s+time)?", re.I), "compound time"),
    (re.compile(r"\b(?:a|an)\s+(?:5/4|7/8)(?:\s+time)?", re.I), "an irregular time"),
    (re.compile(r"\b(?:5/4|7/8)(?:\s+time)?", re.I), "irregular time"),
    (re.compile(r"\b2/4(?:\s+time)?", re.I), "duple time"),
    (re.compile(r"\b(?:4/4|4:4)(?:\s+time)?", re.I), "common time"),
    (re.compile(r"\b3/4(?:\s+time)?", re.I), "waltz time"),
    (re.compile(r"\b2/2(?:\s+time)?", re.I), "cut time"),
    (re.compile(r"\b(?:tr-?)?808s?\b", re.I), "drum machine"),
    (re.compile(r"\b(?:tr-?)?909s?\b", re.I), "drum machine"),
    (re.compile(r"\b(?:19)?80'?s\b", re.I), "eighties"),
    (re.compile(r"\b(?:19)?90'?s\b", re.I), "nineties"),
    (re.compile(r"\b(?:19)?70'?s\b", re.I), "seventies"),
    (re.compile(r"\b(?:19)?60'?s\b", re.I), "sixties"),
    (re.compile(r"\b(?:19)?50'?s\b", re.I), "fifties"),
    (re.compile(r"\b(?:20)?00'?s\b", re.I), "two-thousands"),
    (re.compile(r"\b[A-G](?:#|b)?\d\b"), lambda m: m.group(0)[:-1]),
    (re.compile(r"\b(\d{1,2})\s+(?=(?:string|bar|piece|note|voice|part|track|step)\b)", re.I),
     lambda m: SMALL[int(m.group(1))] + "-" if int(m.group(1)) <= 20 else m.group(0)),
    (re.compile(r"\b(\d{1,2})(?=-[a-z])", re.I),
     lambda m: SMALL[int(m.group(1))] if int(m.group(1)) <= 20 else m.group(0)),
]

DECADE_PATTERNS = {pat.pattern for pat, repl in LEXICON if isinstance(repl, str) and re.search(r"(?:eighties|nineties|seventies|sixties|fifties|two-thousands)", repl)}
INJECTION = re.compile(
    r"\?|\n|\b(?:user|assistant|human|system)\s*:|\b(?:input|output|rewrite|sentence|caption|note)\s*:|"
    r"\b(?:answer the|write (?:a|the|an)|create a|translate|choose one|given the|here is)\b",
    re.I,
)
# Anything that reads as a cut-off phrase, anywhere in the sentence.
DANGLING = re.compile(
    r"(?<![\w-])(?:is|are|was|were|be|set|has|have|having|at|of|with|in|by|to|and|or|a|an|the|than|"
    r"from|between|around|about|approximately|roughly|nearly|possibly|likely|clocking\s+in|played|performed|"
    r"recorded|maintained|maintaining|running|moving|clocked|clocking|falling|ranging|hovering|sitting)\s*[,.;:!]"
    r"|\b(?:a|an|the)\s+(?:\w+\s+)?(?:fast|slow|moderate|quick|steady|consistent|lively|"
    r"brisk|high|low|relaxed|high-energy|medium|rapid)\s*[,.;:!]"
    r"|\b(?:has|have|having|featuring|with|and|,)\s+a\s+tempo\s*(?:[,.;]|and\b)"
    r"|\bis\s+(?:with|at|of|in)\s+a\s+time\b"
    r"|\b(?:is|are|was|were)\s+(?:with|and)\b"
    r"|(?:^|\s)[-/]\w|\(\s*\)|,\s*[,.]|^\W",
    re.I,
)
STOP = {"a", "an", "the", "and", "or", "of", "with", "in", "at", "is", "to", "its", "it",
        "this", "that", "which", "by", "for", "on", "as", "are", "has", "have", "while"}
# Words that only carry the numeric measurement; losing them is expected.
NUMERIC_CONTEXT = STOP | {
    "tempo", "bpm", "beats", "beat", "per", "minute", "minutes", "time", "signature", "around",
    "about", "approximately", "roughly", "nearly", "set", "clocking", "consistent", "steady",
    "constant", "range", "between", "mark", "second", "seconds", "maintaining", "moving",
    "bar", "bit", "hz", "khz", "db", "rate", "pace", "s", "somewhere", "likely", "possibly",
    "estimated", "measured", "precisely", "exactly", "rhythm", "meter", "falls", "falling",
    "ranging", "within", "sits", "hovering", "at", "speed", "counts", "count", "be", "played"}
LEXICON_WORDS = set(SMALL) | {
    "drum", "machine", "chiptune", "eighties", "nineties", "seventies", "sixties",
    "fifties", "two-thousands", "common", "duple", "time", "waltz", "compound", "cut", "irregular",
    "meter", "signature", "four-on-the-floor", "high-resolution", "sixteen-bit", "century",
    "twentieth", "nineteenth", "twenty-first", "eighteenth", "seventeenth"}
TEMPO_ADJ = re.compile(
    r"(?<![\w-])(slow|slower|slowly|fast|faster|moderate|moderately|quick|quickly|brisk|rapid|medium|mid-tempo|"
    r"upbeat|lively|relaxed|leisurely|steady|energetic|high-energy|uptempo|up-tempo|downtempo|driving|"
    r"fast-paced|slow-paced|medium-paced|moderate-paced|mellow|calm|intense)(?![\w-])"
)
DECADE_WORD = re.compile(r"\b(?:eighties|nineties|seventies|sixties|fifties|two-thousands)\b", re.I)
TEMPO_DECADE = re.compile(
    r"(?:tempo|bpm|pace|beats)[^.]{0,40}?\b(?:in|around|within|into)\s+the\s+(?:(?:high|low|mid|upper|lower)[\s\-]+)?\d0'?s\b"
    r"|\b\d0'?s\s+(?:bpm|range|tempo)\b",
    re.I,
)
SPELLED_ROOT = re.compile(
    r"(?:^|-)(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
    r"ninety|hundred|thousand|half|halves|quarters?|eighths?|sixteenths?)(?:$|-)",
    re.I,
)
CLAUSE_SPLIT = re.compile(
    r"(,?\s+(?:played\s+|performed\s+|moving\s+|running\s+|clocking\s+in\s+|set\s+)?at\s+|,\s+(?:and\s+|with\s+|while\s+|which\s+)?|;\s+|\s+and\s+|\s+with\s+)", re.I)
# Function words that cannot end a sentence (participles such as "moving." can).
DANGLING_END = re.compile(
    r"(?<![\w-])(?:is|are|was|were|be|set|has|have|having|at|of|with|in|by|to|and|or|a|an|the|than|"
    r"from|between|around|about|approximately|roughly|nearly|possibly|likely|played)\s*[.!]$"
    r"|\b(?:a|an|the)\s+(?:\w+\s+)?(?:fast|slow|moderate|quick|steady|consistent|lively|brisk|high|low|"
    r"relaxed|high-energy|medium|rapid)\s*[.!]$"
    r"|\b(?:a|an|the)\s+[\w-]+-(?:minute|second|bar|piece|string|note|bit|step|beat)\s*[.!]$",
    re.I,
)
AT_DELIM = re.compile(r"^,?\s+(?:played\s+|performed\s+|moving\s+|running\s+|clocking\s+in\s+|set\s+)?at\s+$", re.I)
INDEPENDENT = re.compile(r"^(?:,?\s+(?:and|while)\s+|;\s+)$", re.I)
SUBJECT = re.compile(r"^(?:the|its|it|this|there)\b", re.I)


def has_digit(text: str) -> bool:
    return bool(DIGIT.search(text))


def split_sentences(caption: str) -> tuple[list[str], list[str]]:
    parts = SENT_SPLIT.split(caption)
    return parts[0::2], parts[1::2]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'\-]*", text.lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in NUMERIC_CONTEXT}


def tidy(text: str) -> str:
    out = re.sub(r"\s{2,}", " ", text)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)
    out = re.sub(r"^\s*[,;:]\s*", "", out).strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and not out.endswith((".", "!")):
        out = out.rstrip(",;: ") + "."
    return out


ARTICLE_TO_AN = re.compile(r"\b([Aa])(?=\s+(?:eight\w*|eleven\w*|eighteen\w*|irregular)\b)")
ARTICLE_TO_A = re.compile(
    r"\b([Aa])n(?=\s+(?:chiptune|drum|common|duple|waltz|compound|cut|nineties|seventies|sixties|fifties|"
    r"twelve|two|twenty|twentieth|nineteenth|seventeenth|four|five|six|seven|nine|ten|high-resolution|vintage)\b)"
)


def fix_articles(text: str) -> str:
    """Lexicon swaps change the next word's sound: "a 1980s" -> "an eighties", "an 808" -> "a drum"."""
    # "with a time signature of 4/4" -> lexicon "with a time signature of common time"; say it plainly.
    text = re.sub(r"\b(with|in)\s+a\s+time\s+signature\s+of\s+((?:common|waltz|compound|cut|irregular)\s+time)\b",
                  r"in \2", text, flags=re.I)
    text = ARTICLE_TO_AN.sub(lambda m: m.group(1) + "n", text)
    return ARTICLE_TO_A.sub(lambda m: m.group(1), text)


def stem(word: str) -> str:
    for suffix in ("ing", "es", "ed", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def sentence_ok(src: str, dst: str, novel_limit: int = 1, check_recall: bool = True,
                stem_match: bool = False) -> tuple[bool, str]:
    if not dst:
        return False, "empty"
    if has_digit(dst):
        return False, "digit"
    if len(INJECTION.findall(dst)) > len(INJECTION.findall(src)):
        return False, "injection"
    if len(SENT_SPLIT.split(dst)) > 1:
        return False, "multi_sentence"
    if not dst.endswith((".", "!")):
        return False, "no_terminal"
    if len(DANGLING.findall(dst)) > len(DANGLING.findall(src)):
        return False, "dangling"
    # The source ends in a number, so any dangling ending in dst is new.
    if DANGLING_END.search(dst):
        return False, "dangling_end"
    if re.search(r"\bbpm\b|\bbeats?\s+per\s+minute\b", dst, re.I):
        return False, "bpm_word"
    if re.search(r"four-on-the-floor", dst, re.I) and not re.search(r"on[\s\-]*the[\s\-]*floor", src, re.I):
        return False, "meter_swap"
    sw, dw = words(src), words(dst)
    for m in re.finditer(r"\b(\w+)[\s\-]+\1\b", dst, re.I):
        if m.group(0).lower() not in src.lower():
            return False, "repeated_word"
    if len(dw) < 3:
        return False, "too_short"
    if len(dw) > len(sw) + 3:
        return False, "too_long"
    # Replacement vocabulary is allowed only where the source had the matching number.
    allowed = set(sw) | set(words(lexicon_tokens(src)))
    invented = [w for w in dw if w in LEXICON_WORDS and w not in allowed]
    if invented:
        return False, "lexicon_hallucination"
    # Spelled-out counts / meters the lexicon did not produce ("two-on-the-floor",
    # "seven-eighths", "one hundred twenty").
    spelled = [w for w in dw if w not in allowed and SPELLED_ROOT.search(w)]
    if spelled:
        return False, "spelled_number"
    if re.search(r"\w+-on-the-floor", dst, re.I) and not re.search(r"on[\s\-]*the[\s\-]*floor", src, re.I):
        return False, "meter_swap"
    # A tempo / energy adjective that the source did not have is an invented claim
    # ("the tempo is 117.9 bpm" -> "the tempo is moderate").
    tempo_gain = Counter(TEMPO_ADJ.findall(dst.lower()))
    tempo_gain.subtract(Counter(TEMPO_ADJ.findall(src.lower())))
    if any(v > 0 for v in tempo_gain.values()):
        return False, "invented_tempo_word"
    if dst[:1].islower():
        return False, "lowercase_start"
    if TEMPO_DECADE.search(src) and DECADE_WORD.search(dst) and not DECADE_WORD.search(src):
        return False, "decade_for_tempo"
    if any("-" in w for w in dw if w not in allowed):
        return False, "novel_compound"
    # Verb-form changes are allowed only from participles ("creating" -> "creates").
    participle_stems = {stem(w) for w in sw if w.endswith(("ing", "ed"))}
    # Information the lexicon carries over (80s -> eighties, 4/4 -> common time, 12-bar ->
    # twelve-bar) must survive: dropping it is content loss, not digit removal.
    carried = (set(words(lexicon_tokens(src))) - set(sw)) & (LEXICON_WORDS | {w for w in words(lexicon_tokens(src)) if SPELLED_ROOT.search(w)})
    if check_recall and carried - set(dw):
        return False, "carried_info_lost"
    novel = [w for w in dw if w not in allowed and w not in STOP and w != "pace"
             and not (stem_match and stem(w) in participle_stems)]
    if stem_match:
        novel_limit = 0
    if len(novel) > novel_limit:
        return False, "novel_words"
    if check_recall:
        missing = content_words(src) - set(dw)
        if stem_match:
            dst_stems = {stem(w) for w in dw}
            missing = {w for w in missing if not (w.endswith(("ing", "ed")) and stem(w) in dst_stems)}
        if missing:
            return False, "content_lost"
    return True, "ok"


def lexicon_tokens(src: str) -> str:
    out = _PAREN_DIGIT.sub("", src)
    tempo_decade = bool(TEMPO_DECADE.search(src))
    for pattern, repl in LEXICON:
        # "The tempo is in the 80s" is a BPM range, not a decade: leave the digits so
        # drop_digit_clauses removes that clause instead of writing "eighties".
        if tempo_decade and pattern.pattern in DECADE_PATTERNS:
            continue
        out = pattern.sub(repl, out)
    return fix_articles(out)


def drop_digit_clauses(text: str) -> str | None:
    body = text.rstrip(".!? ")
    parts = CLAUSE_SPLIT.split(body)
    clauses, delims = parts[0::2], parts[1::2]
    start = 0
    if has_digit(clauses[0]):
        # "The tempo is 93 BPM, and the production quality is clear" -> keep the
        # independent clause that follows.
        for i, delim in enumerate(delims, start=1):
            if INDEPENDENT.match(delim) and SUBJECT.match(clauses[i]) and not has_digit(clauses[i]):
                start = i
                break
        else:
            return None
    keep = [not has_digit(c) for c in clauses]
    # A clause that directly follows a removed one must be able to stand on its own;
    # "in A minor, guitar, bass, and synthesizer" (list tail of a removed clause) cannot.
    for i in range(start + 1, len(clauses)):
        if keep[i] and not keep[i - 1] and delims[i - 1].strip() in (",",) and not re.match(
                r"(?:and|with|while|which|the|its|it|this|there|a|an|[a-z]+ing|[a-z]+ed)\b", clauses[i], re.I):
            return None
    # "with a tempo at 139 BPM": the stub before an "at" clause goes too.
    for i in range(start + 2, len(clauses)):
        if (not keep[i] and AT_DELIM.match(delims[i - 1]) and len(words(clauses[i - 1])) <= 3
                and re.search(r"\b(?:tempo|pace|speed)\b", clauses[i - 1], re.I)):
            keep[i - 1] = False
    out = clauses[start]
    for i in range(start + 1, len(clauses)):
        if keep[i]:
            out += delims[i - 1] + clauses[i]
    return tidy(out)


def lexicon_sentence(src: str) -> tuple[str | None, str]:
    """Deterministic edit of the ORIGINAL sentence. Returns (dst or None=drop, method)."""
    tok = tidy(lexicon_tokens(src))
    if sentence_ok(src, tok)[0]:
        return tok, "lexicon"
    # Smallest cut first: only the number, its unit and its lead-in words.
    span = cut_number_spans(lexicon_tokens(src))
    if span and not re.search(r"\bat\s+a\s+tempo\s*[,.;]", span, re.I) and sentence_ok(src, span, check_recall=False)[0] \
            and not (content_words(src) - set(words(span)) - SPAN_OK_LOSS):
        return span, "lexicon_span"
    clause = drop_digit_clauses(lexicon_tokens(src))
    if clause and sentence_ok(src, clause, check_recall=False)[0]:
        return clause, "lexicon_clause"
    prefix = prefix_before_digit(lexicon_tokens(src))
    if prefix and len(words(prefix)) >= 5 and sentence_ok(src, prefix, check_recall=False)[0]:
        return prefix, "lexicon_prefix"
    return None, "lexicon_drop"


NUMBER_SPAN = re.compile(
    r"(?P<lead>\s*(?:(?:of|around|about|approximately|roughly|nearly|in\s+the\s+range\s+of|ranging\s+from|between)\s+)*"
    r"(?:a\s+|an\s+)?(?:(?:consistent|steady|constant)\s+)?)"
    r"\d+(?:\.\d+)?(?:\s*(?:-|–|to|and)\s*\d+(?:\.\d+)?)?\s*(?:bpm|beats?\s+per\s+minute)\b",
    re.I,
)
# The word right before a cut must not need an object ("maintaining 120 BPM", "is 120 BPM").
SPAN_BAD_BEFORE = re.compile(
    r"(?:\b(?:is|are|was|be|been|at|of|to|with|has|have|had|maintaining|maintains|maintain|keeping|keeps|"
    r"holding|holds|sits|hovers|reaches|reaching|set|clocking\s+in|clocks\s+in|a|an|the|about|around)|,)\s*$",
    re.I,
)
# Only cut right after "<adjective> tempo/pace": "a fast tempo of 128 BPM in D minor".
SPAN_GOOD_BEFORE = re.compile(r"\b(?!(?:a|an|the|its|to|of|at|with|set)\b)[a-z][\w-]*\s+(?:tempo|pace)\s*$", re.I)
# Words a span cut may remove along with the number.
SPAN_OK_LOSS = {"consistent", "steady", "constant"}


def cut_number_spans(text: str) -> str | None:
    out, pos = [], 0
    for m in NUMBER_SPAN.finditer(text):
        before = text[pos:m.start()]
        if SPAN_BAD_BEFORE.search(text[: m.start()]) or not SPAN_GOOD_BEFORE.search(text[: m.start()]):
            return None
        out.append(before)
        pos = m.end()
    out.append(text[pos:])
    return tidy(fix_articles("".join(out)))


TRAILING_FUNCTION = re.compile(
    r"(?:[\s,;:]+(?:in|of|at|with|a|an|the|and|or|to|by|from|between|around|about|approximately|"
    r"roughly|is|are|was|has|have|that|which|keeps?|played|set|featuring))+\s*$|[\s,;:]+$",
    re.I,
)


def prefix_before_digit(text: str, comma_only: bool = True) -> str | None:
    """Cut at the last comma before the first digit ("A jazz piece featuring sax, played
    with a tempo of 141 BPM, ..." -> "A jazz piece featuring sax.")."""
    m = DIGIT.search(text)
    if not m:
        return tidy(text)
    head = text[: m.start()]
    # Only cut at a comma that opens a new phrase ("..., with a tempo of", "..., creating"),
    # never inside an adjective list ("a soothing, melodic piece").
    phrase_start = re.compile(
        r",\s+(?:with|featuring|and|while|which|in|at|played|accompanied|characterized|set|[a-z]+ing)\b", re.I)
    candidates = [head[: m.start()] for m in phrase_start.finditer(head)][::-1]
    if not comma_only:
        candidates.append(head)
    for cand in candidates:
        # Comma cuts end on a whole phrase already; stripping would eat "dancing around".
        cut = cand if comma_only else TRAILING_FUNCTION.sub("", cand)
        cut = tidy(cut)
        # Relative: the source may already contain "The caption is: ..." style prefixes.
        if cut and len(DANGLING.findall(cut)) <= len(DANGLING.findall(text)) and not DANGLING_END.search(cut) \
                and len(words(cut)) >= 3:
            return cut
    return None


def none_is_fine(src: str) -> bool:
    """NONE from the LLM is accepted only when the sentence has no content beyond the number."""
    return not content_words(lexicon_tokens(src))



RESTRUCTURE_SYSTEM = """\
Rewrite ONE sentence from a music caption so that it contains no digits, when the \
number is woven into the grammar and cannot simply be deleted.

Rules:
- Remove the number and its unit (BPM, beats per minute, Hz, time signature digits).
- Restructure the sentence minimally so it stays grammatical. You may change the verb \
form (creating -> creates, contributing -> contributes) or merge clauses.
- Keep EVERY other descriptive word: genre, mood, instruments, key, feel, purpose.
- Replace 4/4 with "common time", 3/4 with "waltz time", 808 with "drum machine".
- Do not invent a tempo word (slow / fast / moderate) that is not already in the sentence.
- Do not add anything new. Answer with the rewritten sentence only."""

RESTRUCTURE_SHOT = [
    ("The tempo is 92 BPM, contributing to the aggressive feel.",
     "The tempo contributes to the aggressive feel."),
    ("The tempo is a relaxed 75 BPM, creating an airy, ethereal mood.",
     "The relaxed tempo creates an airy, ethereal mood."),
    ("It features a tempo of 112 BPM and is in the key of G major.",
     "It is in the key of G major."),
    ("The track has a consistent tempo of 99 BPM and the genre is hip hop, with a lively and upbeat atmosphere.",
     "The track has a consistent tempo and the genre is hip hop, with a lively and upbeat atmosphere."),
    ("The audio is a 116.48 bpm funk track in the key of C major, with a 4/4 time signature featuring electric guitars, drums, bass guitar, and keyboards.",
     "The audio is a funk track in the key of C major, with a common time signature featuring electric guitars, drums, bass guitar, and keyboards."),
    ("The tempo is around 120 BPM and the genre is techno.",
     "The genre is techno."),
    ("It has a tempo of 120.2 bpm, a key of D minor, and a 4/4 time signature.",
     "It has a key of D minor and a common time signature."),
    ("The music has a tempo of approximately 200.3 bpm, suggesting a fast-paced and energetic genre, possibly classical or cinematic.",
     "The music suggests a fast-paced and energetic genre, possibly classical or cinematic."),
]


def build_restructure_prompt(sentence: str) -> str:
    lines = [RESTRUCTURE_SYSTEM, ""]
    for src, dst in RESTRUCTURE_SHOT:
        lines += [f"Sentence: {src}", f"Rewrite: {dst}", ""]
    lines += [f"Sentence: {sentence}", "Rewrite:"]
    return "\n".join(lines)


def load_text_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    model.eval()
    return tok, model


def run_text_batch(tok, model, sentences: list[str], temperature: float) -> list[str]:
    convs = []
    for s in sentences:
        msgs = [{"role": "system", "content": RESTRUCTURE_SYSTEM}]
        for src, dst in RESTRUCTURE_SHOT:
            msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]
        msgs.append({"role": "user", "content": s})
        convs.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    inputs = tok(convs, return_tensors="pt", padding=True).to(model.device)
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                  pad_token_id=tok.pad_token_id)
    kwargs.update(dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0 else dict(do_sample=False))
    with torch.no_grad():
        gen = model.generate(**inputs, **kwargs)
    return [x.strip() for x in tok.batch_decode(gen[:, inputs.input_ids.size(1):], skip_special_tokens=True)]


def build_prompt(sentence: str) -> str:
    lines = [SYSTEM, ""]
    for src, dst in FEW_SHOT:
        lines += [f"Sentence: {src}", f"Rewrite: {dst}", ""]
    lines += [f"Sentence: {sentence}", "Rewrite:"]
    return "\n".join(lines)


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0},
    )
    model.eval()
    tok = processor.tokenizer
    tok.padding_side = "left"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(eos, int) and eos >= 0, eos
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    return model, processor, eos, pad


def run_batch(model, processor, eos, pad, prompts: list[str], temperature: float) -> list[str]:
    texts = [
        processor.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, eos_token_id=eos, pad_token_id=pad)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        generated = model.generate(**inputs, **kwargs)
    new_ids = generated[:, inputs.input_ids.size(1):]
    raw = processor.batch_decode(new_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    out = []
    for item in raw:
        text = item.strip()
        if text.lower().startswith("rewrite:"):
            text = text[len("rewrite:"):].strip()
        out.append(text)
    return out


def audit(rows: list[dict], out_rows: list[dict]) -> dict:
    stats = Counter()
    for src, dst in zip(rows, out_rows):
        assert src["id"] == dst["id"]
        a, b = src["caption"], dst["caption"]
        if has_digit(b):
            raise SystemExit(f"[FAIL] digit remains in {src['id']}")
        if not b.strip():
            raise SystemExit(f"[FAIL] empty caption {src['id']}")
        if not has_digit(a):
            if a != b:
                raise SystemExit(f"[FAIL] digit-free row changed {src['id']}")
            continue
        stats["rewritten_rows"] += 1
        for s in split_sentences(a)[0]:
            if not has_digit(s):
                stats["digit_free_sentences_kept"] += 1
                if s not in b:
                    raise SystemExit(f"[FAIL] digit-free sentence lost in {src['id']}: {s!r}")
        if INJECTION.search(b) and not INJECTION.search(a):
            raise SystemExit(f"[FAIL] injection marker in {src['id']}: {b!r}")
        new_dangling = len(DANGLING.findall(b)) - len(DANGLING.findall(a))
        if new_dangling > 0:
            raise SystemExit(f"[FAIL] new dangling fragment in {src['id']}: {b!r}")
        stats["src_words"] += len(a.split())
        stats["dst_words"] += len(b.split())
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-tsv", type=Path, default=SRC_TSV)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="lexicon path only (offline test)")
    parser.add_argument("--pass3-model", default="",
                        help="text-only instruct model for sentences the Omni thinker failed")
    parser.add_argument("--pass3-redo-fallback", action="store_true",
                        help="re-queue logged lexicon_* sentences into pass 3 only")
    parser.add_argument("--revalidate", action="store_true",
                        help="re-check logged records against the current gates: LLM records that now "
                             "fail are re-queued, lexicon records are recomputed")
    args = parser.parse_args()
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.src_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    done: dict[str, dict] = {}
    if args.resume and args.log_jsonl.exists():
        for line in args.log_jsonl.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec
    redo: dict[str, dict] = {}
    updates: list[dict] = []
    if args.revalidate:
        for k, r in list(done.items()):
            m = r["method"]
            if m == "llm_none":
                if not none_is_fine(r["src"]):
                    del done[k]
                continue
            if m.startswith("lexicon"):
                dst, method = lexicon_sentence(r["src"])
                if (dst, method) != (r["dst"], m):
                    updates.append({**r, "dst": dst, "method": method, "revalidated": True})
                continue
            fixed = fix_articles(r["dst"])
            restructure = m in ("llm_restructure", "llm_pass3")
            ok = sentence_ok(r["src"], fixed, stem_match=restructure)[0]
            if ok:
                if fixed != r["dst"]:
                    updates.append({**r, "dst": fixed, "revalidated": True})
                continue
            del done[k]
            if restructure:
                redo[k] = r
        print(f"revalidate: updated={len(updates)} requeued_pass2={len(redo)}")
    pass3_keys: dict[str, dict] = {}
    if args.pass3_redo_fallback:
        pass3_keys = {k: r for k, r in done.items() if r["method"].startswith("lexicon")}
        for k in pass3_keys:
            del done[k]
    units = []
    for row in rows:
        if not has_digit(row["caption"]):
            continue
        for i, s in enumerate(split_sentences(row["caption"])[0]):
            if has_digit(s) and f"{row['id']}#{i}" not in done:
                key = f"{row['id']}#{i}"
                prev = redo.get(key) or pass3_keys.get(key)
                units.append({"key": key, "src": s, "attempts": prev["attempts"] if prev else 0,
                              "reasons": list(prev["reasons"]) if prev else [],
                              "pass2_only": key in redo, "pass3_only": key in pass3_keys})
    print(f"rows={len(rows)} digit_rows={sum(has_digit(r['caption']) for r in rows)} "
          f"pending_sentences={len(units)} done={len(done)}")

    with args.log_jsonl.open("a", encoding="utf-8") as handle:
        def emit(unit, method, dst):
            rec = {"key": unit["key"], "method": method, "attempts": unit["attempts"],
                   "reasons": unit["reasons"], "src": unit["src"], "dst": dst}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[unit["key"]] = rec

        for rec in updates:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[rec["key"]] = rec

        work = [u for u in units if not u["pass2_only"] and not u["pass3_only"]]
        redo_units = [u for u in units if u["pass2_only"]]
        pass3_units = [u for u in units if u["pass3_only"]]
        model = processor = eos = pad = None
        if work and not args.no_llm:
            model, processor, eos, pad = load_model()
            for attempt in range(MAX_ATTEMPTS):
                if not work:
                    break
                temp = 0.0 if attempt == 0 else 0.3 + 0.2 * (attempt - 1)
                still = []
                for off in tqdm(range(0, len(work), args.batch_size), desc=f"attempt {attempt + 1}"):
                    batch = work[off: off + args.batch_size]
                    outs = run_batch(model, processor, eos, pad,
                                     [build_prompt(u["src"]) for u in batch], temp)
                    for unit, out in zip(batch, outs):
                        unit["attempts"] = attempt + 1
                        if out.strip().rstrip(".").upper() == "NONE":
                            if none_is_fine(unit["src"]):
                                emit(unit, "llm_none", None)
                            else:
                                unit["reasons"].append("none_but_content")
                                still.append(unit)
                            continue
                        out = fix_articles(out)
                        ok, why = sentence_ok(unit["src"], out)
                        if ok:
                            emit(unit, "llm", out)
                        else:
                            unit["reasons"].append(why)
                            still.append(unit)
                    handle.flush()
                work = still
        work = work + redo_units
        if work and not args.no_llm:
            # Pass 2: the number is part of the grammar ("The tempo is 92 BPM, contributing
            # to ..."). Allow verb-form changes (stem match) but still no content loss.
            if model is None:
                model, processor, eos, pad = load_model()
            for attempt in range(RESTRUCTURE_ATTEMPTS):
                if not work:
                    break
                temp = 0.0 if attempt == 0 else 0.4
                still = []
                for off in tqdm(range(0, len(work), args.batch_size), desc=f"restructure {attempt + 1}"):
                    batch = work[off: off + args.batch_size]
                    outs = run_batch(model, processor, eos, pad,
                                     [build_restructure_prompt(u["src"]) for u in batch], temp)
                    for unit, out in zip(batch, outs):
                        unit["attempts"] += 1
                        out = fix_articles(out)
                        ok, why = sentence_ok(unit["src"], out, novel_limit=1, stem_match=True)
                        if ok:
                            emit(unit, "llm_restructure", out)
                        else:
                            unit["reasons"].append(f"r:{why}")
                            still.append(unit)
                    handle.flush()
                work = still
        work = work + pass3_units
        if work and not args.no_llm and args.pass3_model:
            # Pass 3: a stronger text-only instruct model for what the 3B Omni thinker could
            # not do (it tends to echo "The tempo is 130.0 bpm and ..." unchanged). Same gates.
            del model
            torch.cuda.empty_cache()
            tok3, model3 = load_text_model(args.pass3_model)
            for attempt in range(PASS3_ATTEMPTS):
                if not work:
                    break
                temp = 0.0 if attempt == 0 else 0.5
                still = []
                for off in tqdm(range(0, len(work), args.batch_size), desc=f"pass3 {attempt + 1}"):
                    batch = work[off: off + args.batch_size]
                    outs = run_text_batch(tok3, model3, [u["src"] for u in batch], temp)
                    for unit, out in zip(batch, outs):
                        unit["attempts"] += 1
                        out = fix_articles(out)
                        ok, why = sentence_ok(unit["src"], out, novel_limit=1, stem_match=True)
                        if ok:
                            emit(unit, "llm_pass3", out)
                        else:
                            unit["reasons"].append(f"p3:{why}")
                            still.append(unit)
                    handle.flush()
                work = still
        for unit in work:
            dst, method = lexicon_sentence(unit["src"])
            emit(unit, method, dst)
        handle.flush()

    out_rows = []
    row_last_resort: list[dict] = []
    for row in rows:
        caption = row["caption"]
        if has_digit(caption):
            pieces = []
            for i, s in enumerate(split_sentences(caption)[0]):
                if not has_digit(s):
                    pieces.append(s)
                elif done[f"{row['id']}#{i}"]["dst"]:
                    pieces.append(done[f"{row['id']}#{i}"]["dst"])
            caption = " ".join(pieces)
            if not caption.strip():
                # Last resort so no row goes empty: the digit-free head of the first
                # sentence, trimmed back to a whole phrase. Logged in the manifest.
                first = lexicon_tokens(split_sentences(row["caption"])[0][0])
                caption = prefix_before_digit(first, comma_only=False) or ""
                row_last_resort.append({"id": row["id"], "src": row["caption"], "dst": caption})
            if not caption.strip() or has_digit(caption):
                raise SystemExit(f"[FAIL] every sentence dropped in {row['id']}")
        out_rows.append({"id": row["id"], "caption": caption})

    stats = audit(rows, out_rows)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption"], delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    manifest = {"src_tsv": str(args.src_tsv), "out_tsv": str(args.out_tsv),
                "rows": len(out_rows),
                "sentence_methods": dict(Counter(r["method"] for r in done.values())),
                "row_last_resort": row_last_resort,
                "audit": stats}
    args.out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
