#!/usr/bin/env python3
"""slot0nmv2: slot0clean with guessed measurements removed, redone without slot0nm's defects.

slot0nm (057) was built on top of slot4v2, whose rule was "every digit goes", so style
descriptors that happen to contain digits were rewritten: 80s -> eighties, 808 -> drum
machine, 8-bit -> chiptune, 12-bar -> twelve-bar, and BPM ranges ("tempo in the 80s")
were read as decades. Its LLM also dropped whole clauses (333 rows lost mood/genre words)
and a stem-based novelty gate let "primarilys" through.

This version:
  - base = slot0clean for every row (never slot4v2);
  - removes only MEASUREMENTS: tempo numbers / BPM (including "tempo in the high 120s"),
    meter / time signature, key / mode / chord quality, Hz / dB, durations and timestamps,
    note names with octave;
  - PROTECTED tokens (decades, years, 8-bit, 808, 12-bar, 12-string, 20th century,
    16th note, 4 on the floor, ...) must survive every edit exactly;
  - gate is word-exact: the edit may only delete words, plus a small glue whitelist;
    every content word outside the measurement spans must survive;
  - every accepted edit is also judged by the same model (grammatical / lost description /
    added description); deletion-only edits of hard sentences leave fragments, so failures go
    to a "restructure" pass (function words and plain verbs may be added, no new description);
  - whatever is left is listed as unresolved and the corpus is not written (--unresolved).
  Thinking mode was tried and dropped: it obeyed "delete only" by leaving fragments.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**9)

SENT_SPLIT = re.compile(r"(?<=[.!?])(\s+)")
NUM = r"\d+(?:\.\d+)?"
RANGE = rf"{NUM}(?:\s*(?:-|–|to|and|or)\s*{NUM})?"
UNIT = (r"(?:bpms?|beats?[\s-]+per[\s-]+(?:minute|second|measure|bar)|k?hz|db|decibels?|hertz|"
        r"seconds?|secs?|minutes?|mins?|ms|milliseconds?)\b")
NOTE = r"(?-i:[A-G])(?:#|♯|(?-i:b)|♭|\s+sharp|\s+flat)?"
TEMPO_RANGE = (r"(?:(?:mid|high|low|early|late|upper|lower)(?:[\s-](?:to[\s-])?(?:mid|high|low|late))?[\s-])?"
               r"(?:1\d0|2\d0|\d0)'?s")
NOT_STYLE_NUM = r"(?!(?:808|909|303|606|707)\b|(?:1[5-9]|20)\d\d\b)"
BPM_WORD = r"(?:bpms?|beats?[\s-]+per[\s-]+minute|beat[\s-]+per[\s-]+minute|metronome)"

MEASURE = re.compile(
    rf"{RANGE}\s*{UNIT}"                                            # 120 BPM, 88 and 92 BPM, 11 seconds
    rf"|\b{NUM}-(?:second|minute)\b|\b{NUM}\s*/\s*min(?:ute)?\b"    # 60-second clip, 95/minute
    r"|\b(?:1\d|2\d)0(?:'?s|th)\b|\b\d+(?:st|nd|rd|th)\s+bpm\b"   # 100s..290s / "195th BPM": never a decade
    rf"|\btempo\b[^.;,!?]{{0,25}}?(?P<g1>\bin (?:the )?{TEMPO_RANGE})"  # tempo is in the (high) 80s
    rf"|{TEMPO_RANGE}\s*(?:bpm|beats|tempo|beat-per-minute|range)\b"
    rf"|\b(?:tempo|pace|speed|beat|{BPM_WORD})\b(?:\s*\([^)]*\))?(?:\s+(?!\d)\w+){{0,5}}?"
    rf"(?P<g2>(?:\s*[:=]\s*|\s(?:of|at|is|around|about|approximately|roughly|between|near|over|under|above|below|slightly|just|exactly|precisely|some)(?=\s))*\s*{NOT_STYLE_NUM}{RANGE}\b(?![-/:]|'?s\b|\s*(?:st|nd|rd|th)\b|[\s-]?(?:bars?|string|bit|piece|track|step|beat|on the floor|years?)\b))"
    r"|\b\d+\s*/\s*\d+\b(?:\s+(?:time signature|time|meter|metre)\b)?|\b\d\s*:\s*\d\b|\b\d+:\d\d\b"             # 4/4, 4:4, 0:15
    r"|\bcounts? to \d+(?: or \d+)?\b|\b\d+ beats? per (?:measure|bar)\b"
    r"|\bbpms?\b|beats?[\s-]+per[\s-]+minute|\bhertz\b|\bk?hz\b|\bdecibels?\b|\bdb\b"
    rf"|\b(?:(?:song|musical)\s+)?keys? of {NOTE}(?![a-z])(?:[\s-]+(?:major|minor|dorian|mixolydian|phrygian|lydian|aeolian))?|\bkeys? (?:signatures?|changes?|modulations?)\b|\bin the (?:same |original |home )?key\b"
    r"|\b(?:major|minor|same|different|relative|parallel|home|original|new|higher|lower) keys?(?: signatures?)?\b"
    r"|\btonal (?:center|centre)\b"
    r"|\b(?:major|minor)(?: (?:and|or|to) (?:major|minor))? (?:key signatures?|keys|chords?|thirds?|sixths?|sevenths?|triads?|progressions?|intervals?)\b"
    r"|\bin (?:a|the) (?:same|different|relative|parallel|home|original|new|higher|lower|dominant|tonic|"
    r"single|given|particular|certain|specific|flat|sharp|major|minor) key\b"
    rf"|(?<![\w-]){NOTE}[\s-]+(?:major|minor|dorian|mixolydian|phrygian|lydian|aeolian|ionian|locrian)(?:\s*\d+\b|\b)"
    r"|\b(?:major|minor) (?:key signatures?|scale|mode|chord progression|tonality|key)\b|\bin (?:a )?(?:major|minor)(?:\s+(?:key signature|keys?|scales?|modes?|tonality|chords?))?\b"
    r"|\b(?-i:[A-G])(?:#|b|♯|♭)?(?:maj|min|m|dim|aug|sus)\d*\b|\b(?-i:[A-G])(?:#|♯|♭)?\d{1,3}\b"  # Emaj, D7, C#4, A440
    r"|\b\d+(?:st|nd|rd|th)(?:\s+(?:or|and)\s+(?:a\s+)?\d+(?:st|nd|rd|th))?(?:[\s-](?:chords?|intervals?)|\s+chord progressions?)\b"
    r"|\b\d+(?:st|nd|rd|th) (?:fret|string)\b"
    r"|time signature|\b(?:common|waltz|cut|duple|triple|compound|irregular|odd) time\b|\bmet(?:er|re)\b"
    r"|\b(?:four|three|six|two|five|seven|nine|twelve)[\s-](?:four|eight|two)(?: time)?\b",
    re.I,
)
PROTECT = re.compile(
    r"(?:\b(?:mid|early|late)-)?(?<![\w])'?(?:1[5-9]|20)?\d0(?:'?s|th's)\b(?:-(?:style|inspired|era|influenced))?"
    r"|\b(?:1[5-9]|20)\d\d\b"
    r"|\b\d+[\s-]?bit\b|\b(?:808|909|303|606|707)s?\b(?:-style)?"
    r"|\b\d+[\s-]bars?\b|\b\d+[\s-]string\b|\b\d+-piece\b|\b\d+-track\b|\b\d+(?:st|nd|rd|th)[\s-]centur(?:y|ies)\b"
    r"|\b\d+(?:st|nd|rd|th)(?:[\s-]notes?)\b|\b\d+[\s-]on[\s-]the[\s-]floor\b|\b\d+-step\b|\b\d+-beat\b",
    re.I,
)
DECADE_WORDS = re.compile(r"\b(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties|"
                          r"noughties|chiptune|drum machine|twelve-bar|common time|four-on-the-floor)\b", re.I)
INJECTION = re.compile(
    r"\?|\n|\b(?:user|assistant|human|system)\s*:|\b(?:input|output|rewrite|sentence|caption|note|delete|keep)\s*:|"
    r"\b(?:answer the|write (?:a|the|an)|create a|translate|choose one|given the|here is)\b|</?think>",
    re.I,
)
DANGLING = re.compile(
    r"(?<![\w-])(?:is|are|was|were|be|set|has|have|having|at|of|with|in|by|to|and|or|a|an|the|than|"
    r"from|between|around|about|approximately|roughly|nearly|possibly|likely)\s*[,.;:!]"
    r"|\b(?:primarily|mainly|predominantly|mostly|largely|probably|typically|usually|often)\s*[,.;:!]"
    r"|\b(?:a|an|the)\s+(?:[\w-]+\s+)?(?:fast|slow|moderate|quick|steady|consistent|constant|stable|high|low|"
    r"medium|rapid|driving|lively|distinct|regular|standard|simple)\s*[,.;:!]"
    r"|\b(?:is|are|was|were|be|a|an|the|of|in|at|with|by|and|or|from|to)\s+(?:and|or|but|,)(?=\s|$)"
    r"|\b(?:is|are|was|were)\s+with\b"
    r"|\b(?:with|has|have|had|at|maintains?|maintaining|keeps?|keeping)\s+(?:a|an|its|the)\s+tempo\b(?!\s+(?:that|which|of|is|in|at|around|between|approximately|about|near|ranging|\())"
    r"|^A\s+tempo\b|^The\s+tempo\s*,"
    r"|\b(?:is|are)\s+(?:primarily|mainly|predominantly|mostly|largely)\s+\w+ing\b"
    r"|\b(?:follows|features|featuring|includes|including|uses|using|employs|playing|set|consists|"
    r"has|have|with|between|of|in|at|by)\s*[.!]$"
    r"|,\s*[,.]|^\W|\(\s*\)",
    re.I,
)
STOP = {"a", "an", "the", "and", "or", "of", "with", "in", "at", "is", "to", "its", "it", "this", "that",
        "which", "by", "for", "on", "as", "are", "has", "have", "while", "be", "was", "there", "also", "s"}
# words that only exist to carry a measurement; losing them with it is fine
ORPHAN = {"consistent", "steady", "stable", "constant", "pace", "tempo", "speed", "rate", "set", "estimated",
          "range", "clocking", "clocked", "measured", "precise", "precisely", "exact", "exactly", "played",
          "performed", "written", "composed", "around", "about", "approximately", "roughly", "nearly",
          "likely", "possibly", "falls", "sits", "maintained", "maintaining", "beat", "beats", "per", "minute",
          "signature", "key", "keys", "scale", "mode", "chord", "chords", "time", "measure", "measures",
          "count", "counts", "value", "which", "that", "while", "both", "alternating", "between", "within",
          "mark", "second", "seconds", "interval", "intervals", "fret", "metronome", "follows", "following",
          "consists", "consisting", "featuring", "includes", "including", "appears", "seems", "playing",
          "flows", "flowing", "keeping", "moving", "falling", "sitting", "running", "hovering", "stays",
          "remains", "uses", "using", "employs", "adheres", "adhering", "song", "consistently", "steadily", "revolves", "revolving", "mainly", "primarily", "predominantly",
          "mostly", "mix", "shifts", "shift", "shifting", "into", "occasional", "occasionally", "indicated",
          "indicating", "frequency", "vibrating", "centered", "centred", "based"}
# glue an edit may add when the measurement was the grammatical head of a clause
GLUE = {"has", "have", "features", "featuring", "creates", "creating", "gives", "giving", "is", "are", "with",
        "and", "a", "an", "the", "music", "track", "piece", "song", "it", "its", "this"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", text.lower())


def measure_spans(s: str) -> list[tuple[int, int]]:
    """Spans to delete. For "tempo ... <number>" only the connector + number is the
    measurement: "The tempo is moderate at 135 bpm" keeps "The tempo is moderate"."""
    out = []
    for m in MEASURE.finditer(s):
        g = next((n for n in ("g1", "g2") if m.group(n)), None)
        a, b = m.span(g) if g else m.span()
        while a < b and s[a].isspace():
            a += 1
        out.append((a, b))
    return out


def protected(s: str) -> list[str]:
    ms = measure_spans(s)
    out = []
    for m in PROTECT.finditer(s):
        if not any(a < m.end() and m.start() < b for a, b in ms):
            out.append(m.group(0))
    return out


def unknown_digits(s: str) -> list[str]:
    cover = measure_spans(s) + [(m.start(), m.end()) for m in PROTECT.finditer(s)]
    return [s[max(0, m.start() - 40): m.end() + 30] for m in re.finditer(r"\d+", s)
            if not any(a <= m.start() < b for a, b in cover)]


def content_outside_measures(s: str) -> Counter:
    """Content words that must survive. A single word glued to a numeric measurement
    ("a lively 120 BPM", "a 132 BPM rhythm", "a distinct 4/4") may go with it."""
    ms = measure_spans(s)
    kept = list(s)
    for a, b in ms:
        kept[a:b] = " " * (b - a)
        if re.match(r"\s*\d", s[a:b]):
            pre = re.search(r"([A-Za-z-]+)\s+$", s[:a])
            if pre and pre.group(1).lower() not in {"moderate", "fast", "slow", "quick", "rapid", "brisk", "upbeat",
                                                     "fast-paced", "slow-paced", "mid", "mid-tempo", "high", "low"}:
                kept[pre.start(1):pre.end(1)] = " " * (pre.end(1) - pre.start(1))
        if re.search(r"(?:\d|bpm|minute)\s*$", s[a:b], re.I):
            post = re.match(r"\s+([A-Za-z-]+)", s[b:])
            if post and post.group(1).lower() in {"rhythm", "pace", "tempo", "beat", "groove", "pulse", "range", "feel", "count"}:
                kept[b + post.start(1):b + post.end(1)] = " " * (post.end(1) - post.start(1))
    return Counter(w for w in words("".join(kept))
                   if w not in STOP and w not in ORPHAN and w not in MEASURE_NOUNS and not re.search(r"\d", w))


MEASURE_NOUNS = {"key", "keys", "scale", "scales", "mode", "modes", "signature", "signatures", "tonality",
                 "metronome", "bpm", "meter", "metre", "time"}


def measure_noun_excess(src: str, dst: str) -> list[str]:
    ms = measure_spans(src)
    outside = Counter(w for w in words("".join(" " if any(a <= i < b for a, b in ms) else c
                                               for i, c in enumerate(src))) if w in MEASURE_NOUNS)
    have = Counter(w for w in words(dst) if w in MEASURE_NOUNS)
    return sorted((have - outside).keys())


def tidy(text: str) -> str:
    out = re.sub(r"\s{2,}", " ", text.strip())
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)
    out = re.sub(r"^\s*[,;:]\s*", "", out).strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and not out.endswith((".", "!")):
        out = out.rstrip(",;: ") + "."
    return out


def tempo_word_allowed(src: str) -> set[str]:
    ok = set()
    if re.search(r"\b(?:high|fast|rapid|quick|brisk)\s+(?:bpm|beats per minute)", src, re.I):
        ok |= {"fast", "tempo"}
    if re.search(r"\b(?:low|slow)\s+(?:bpm|beats per minute)", src, re.I):
        ok |= {"slow", "tempo"}
    return ok


VOCAB: Counter = Counter()   # corpus word counts, filled in main(); guards inflection swaps


def stems(w: str) -> set[str]:
    """Candidate base forms: evokes/evoke, fits/fitting/fit, indicates/indicating/indicate."""
    out = {w}
    for suf in ("s", "es", "ed", "d", "ing"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[: -len(suf)]
            out |= {b, b + "e"}
            if suf in ("ing", "ed") and len(b) > 3 and b[-1] == b[-2]:
                out.add(b[:-1])
    return out


def stem(w: str) -> str:
    return min(stems(w), key=len)


GLUE_EXT = GLUE | {"shows", "show", "uses", "use", "provides", "provide", "contributes", "contribute", "adds",
                   "add", "sets", "set", "plays", "play", "carries", "carry", "moves", "move", "builds", "build",
                   "creates", "gives", "has", "is", "are", "was", "which", "that", "while", "of", "to", "in", "at",
                   "by", "for", "from", "on", "as", "be", "being", "its", "their", "there"}


def gate(src: str, dst: str, relaxed: bool = False) -> list[str]:
    """Reasons to reject; empty = accept. dst == NONE handled by caller.
    relaxed (restructure mode): function words / simple verbs may be added and up to two
    words may be lost; the judge then decides whether description was lost or added."""
    why = []
    if MEASURE.search(dst):
        why.append("measure_left")
    if Counter(protected(src)) != Counter(protected(dst)):
        why.append(f"protected:{sorted(protected(src))}->{sorted(protected(dst))}")
    if re.search(r"\d", PROTECT.sub("", dst)):
        why.append("new_digit")
    if len(DECADE_WORDS.findall(dst)) > len(DECADE_WORDS.findall(src)):
        why.append("respelled_style_word")
    if INJECTION.search(dst) and not INJECTION.search(src):
        why.append("injection")
    if len(DANGLING.findall(dst)) > len(DANGLING.findall(src)):
        why.append("dangling")
    if measure_noun_excess(src, dst):
        why.append(f"orphan_noun:{','.join(measure_noun_excess(src, dst))}")
    if len(SENT_SPLIT.split(dst.strip())) > 1:
        why.append("multi_sentence")
    if not dst.endswith((".", "!")):
        why.append("no_terminal")
    if len(dst.split()) > len(src.split()):
        why.append("longer")
    src_words = set(words(src))
    novel = [w for w in words(dst) if w not in src_words and w not in (GLUE_EXT if relaxed else GLUE) | tempo_word_allowed(src)]
    lost = content_outside_measures(src) - Counter(words(dst))
    # "giving" -> "gives" is fine; "primarily" -> "primarilys" is not (not a corpus word)
    src_stems = set().union(*(stems(x) for x in src_words)) if src_words else set()
    for w in [w for w in words(dst) if w not in src_words]:
        if not (stems(w) & src_stems) or VOCAB[w] < 20:
            continue
        if w in novel:
            novel.remove(w)
        match = next((l for l in lost if lost[l] > 0 and stems(l) & stems(w)), None)
        if match:
            lost[match] -= 1
    lost = +lost
    if novel:
        why.append(f"novel:{','.join(novel[:4])}")
    if relaxed and sum(lost.values()) <= 2:
        lost = Counter()
    if tempo_word_allowed(src):
        lost = Counter({w: n for w, n in lost.items() if w not in {"high", "low", "fast", "slow", "rapid", "quick", "brisk"}})
    if lost:
        why.append(f"lost:{','.join(sorted(lost)[:4])}")
    return why


def verb_only_loss(src: str, dst: str) -> bool:
    """True when every description word missing from dst is an -ing verb in the source
    ("contributing to", "creating a", "making it") - a verb swap, not lost description."""
    lost = content_outside_measures(src) - Counter(words(dst))
    for w in lost:
        if not (w.endswith("ing") and re.search(rf"\b{re.escape(w)}\s+(?:to|a|an|the|its|it|in|with|into|for)\b", src, re.I)):
            return False
    return True


def none_ok_relaxed(src: str) -> bool:
    """restructure mode may also answer NONE when at most two description words remain"""
    return not protected(src) and sum(content_outside_measures(src).values()) <= 2


def none_ok(src: str) -> bool:
    rest = [w for w in content_outside_measures(src)
            if w not in {"music", "song", "track", "piece", "audio", "performance", "sound"}]
    return not rest and not protected(src)


SYSTEM = """\
You edit ONE sentence of a music caption. The caption was written by a model that could
only guess musical MEASUREMENTS, so they must be deleted:
- tempo numbers and BPM in any form ("120 BPM", "a tempo of 96", "around 120-125 beats per
  minute", "the tempo is in the high 120s", "the tempo is in the 80s" = a BPM range here)
- meter / time signature ("4/4", "a 3/4 time signature", "common time", "counts to 4")
- key, mode, chord quality, note names ("in the key of C minor", "in D major", "in a minor
  key", "minor chords", "Emaj and D7", "C#4"); keep "chord progression" itself
- Hz, dB, durations and timestamps ("60-second clip", "around 11 seconds")

The user message lists the exact spans to delete ("Delete") and tokens that MUST stay
character-for-character ("Keep"). Kept tokens are style descriptors, not measurements:
decades ("80s", "1980s", "'70s", "80's") stay exactly as written - never spell them out as
words, never drop them; likewise "8-bit", "808", "12-bar", "20th century", "16th note".

Rules: delete only the listed spans plus the connective words that become ungrammatical
("with a", "at", "set at", "a consistent ... pace"). Keep every other word in the same order.
Do not add words, genres, moods, instruments, or tempo adjectives (do not turn a number into
"moderate" or "fast"). Exception: "high BPM" may become "fast tempo". If the only thing the
sentence said was a measurement, answer exactly NONE.
Qualitative tempo descriptions are NOT measurements: in "The tempo is moderate at 110 BPM"
only "at 110 BPM" goes; "The tempo is moderate" stays. Never drop a clause just because it
mentions tempo. Answer with the edited sentence only."""

FEW_SHOT = [
    ("The tempo is moderate, with a consistent 120 BPM pace.", None, None, "The tempo is moderate."),
    ("The tempo is slow at 89.5 BPM, and the mix is well-processed.", None, None,
     "The tempo is slow, and the mix is well-processed."),
    ("The tempo is moderately fast at 95 BPM, creating a danceable atmosphere.", None, None,
     "The tempo is moderately fast, creating a danceable atmosphere."),
    ("The tempo is 106 BPM.", None, None, "NONE"),
    ("The performance is in the key of D major, in common time.", None, None, "NONE"),
    ("The music has a retro 80s feel with a tempo of 118 BPM.", None, None,
     "The music has a retro 80s feel."),
    ("The tempo is in the 80s, and the genre is electronic.", None, None, "The genre is electronic."),
    ("The tempo is in the high 120s BPM, and the groove is reminiscent of 1980s synth-pop.",
     None, None, "The groove is reminiscent of 1980s synth-pop."),
    ("The audio is a 116.48 bpm funk track in the key of C major, with a 4/4 time signature featuring electric guitars, drums, and keyboards.", None, None,
     "The audio is a funk track featuring electric guitars, drums, and keyboards."),
    ("A nostalgic 8-bit chiptune melody plays over a steady 4/4 beat in A minor, typical of '80s video games.", None, None,
     "A nostalgic 8-bit chiptune melody plays over a steady beat, typical of '80s video games."),
    ("A gentle acoustic guitar piece is played in a minor key, creating a serene atmosphere.", None, None, "A gentle acoustic guitar piece is played, creating a serene atmosphere."),
    ("The music has a high BPM with a fast tempo, creating an exhilarating atmosphere.", None, None,
     "The music has a fast tempo, creating an exhilarating atmosphere."),
    ("It features an 808 bassline and a minor chord progression at 140 BPM, with a melancholic mood.", None, None,
     "It features an 808 bassline and a chord progression, with a melancholic mood."),
]


RESTRUCTURE_SYSTEM = SYSTEM.split("Rules:")[0] + """\
This sentence cannot be fixed by deleting words alone. Rewrite it WITHOUT the listed
measurements as one complete, natural English sentence. You may reorder, change a verb form,
drop a verb that only carried the measurement, and add small function words or plain verbs
(is, has, features, shows, creates, gives, contributes, provides, uses, plays). Do NOT add
any description: no new adjectives, nouns, instruments, genres, moods or tempo words. Keep
every other descriptive word (moods, textures, instruments, qualitative tempo words). If
nothing descriptive is left once the measurements are gone, answer exactly NONE. Answer
with the rewritten sentence only."""

RESTRUCTURE_SHOTS = [
    ("The tempo is 99 BPM, showcasing a moderate pace.", "The music has a moderate pace."),
    ("The chord progression mainly revolves around F minor, C minor, and F major, giving the piece a somewhat melancholic yet energetic mood.",
     "The chord progression gives the piece a somewhat melancholic yet energetic mood."),
    ("The music appears to be in a minor key, with a moderate tempo and a slightly melancholic mood.",
     "The music has a moderate tempo and a slightly melancholic mood."),
    ("The tempo is fast-paced, maintaining a steady 125 BPM throughout the clip.", "The tempo is fast-paced throughout the clip."),
    ("The piece is in the key of C minor, with a tempo of 110 BPM, giving it a swinging feel.", "The piece has a swinging feel."),
    ("The audio features smooth guitar chords that transition gracefully between major and minor keys, creating a calm mood.",
     "The audio features smooth guitar chords that transition gracefully, creating a calm mood."),
    ("The chord progression mainly comprises G minor, C minor, and C Major.", "NONE"),
    ("The time signature is 4/4, creating a steady and rhythmic feel, and the tempo is 95.9 bpm, giving the music a lively and energetic character.",
     "The music has a steady and rhythmic feel and a lively and energetic character."),
]

JUDGE_SYSTEM = """\
You check an automatic edit of one sentence from a music caption. The edit had to remove the
listed measurements (tempo numbers/BPM, meter, key/mode/chord quality, Hz, dB, durations) and
keep everything else. Judge EDITED against ORIGINAL and answer JSON only:
{"grammatical": true|false, "lost_description": true|false, "added_description": true|false}
grammatical = EDITED is a complete, natural English sentence. false for fragments: a subject
  with no verb ("The tempo, contributing to the mood."), a verb missing its object ("The chord
  progression mainly comprises."), "is with", "is giving it", "between major,", a clause that
  stops ("maintaining throughout the clip", "is delivered.", "is characterized."). Plain or
  slightly awkward style is fine. Captions are often noun phrases without a main verb ("A solo
  piano piece with a gentle melody."): if ORIGINAL is such a noun phrase, a noun-phrase EDITED
  is grammatical.
lost_description = something from ORIGINAL other than the measurements is missing: an
  instrument, genre, mood, texture, era, production detail, or a qualitative tempo word
  (fast, slow, moderate, steady, upbeat). Losing only the measurement and words that just
  carried it ("at", "set at", "a consistent ... pace", "the key of") is fine.
added_description = EDITED states a property ORIGINAL did not."""

JUDGE_SHOTS = [
    ("The tempo is moderate at 110 BPM, and the genre is electronic.", "The tempo is moderate, and the genre is electronic.",
     {"grammatical": True, "lost_description": False, "added_description": False}),
    ("The tempo is 99 BPM, showcasing a moderate pace.", "The tempo, showcasing a moderate pace.",
     {"grammatical": False, "lost_description": False, "added_description": False}),
    ("The tempo is moderate at 135 bpm, and the recording quality is top-notch.", "The recording quality is top-notch.",
     {"grammatical": True, "lost_description": True, "added_description": False}),
    ("The chord progression is in E minor with a 4/4 time signature, giving the song a steady and driving feel.",
     "The chord progression is giving the song a steady and driving feel.",
     {"grammatical": False, "lost_description": False, "added_description": False}),
    ("The tempo is 128 BPM, typical of house music.", "The tempo is fast, typical of house music.",
     {"grammatical": True, "lost_description": False, "added_description": True}),
    ("A soft guitar strums in a minor key, accompanied by a smooth synth pad.", "A soft guitar strums, accompanied by a smooth synth pad.",
     {"grammatical": True, "lost_description": False, "added_description": False}),
    ("A piano solo with a melancholic and reflective mood, played in a minor key.", "A piano solo with a melancholic and reflective mood.",
     {"grammatical": True, "lost_description": False, "added_description": False}),
]


def judge_msg(src: str, dst: str) -> str:
    dels = "; ".join(json.dumps(src[a:b]) for a, b in measure_spans(src))
    return f"ORIGINAL: {src}\nMEASUREMENTS: {dels}\nEDITED: {dst}"


def user_msg(s: str) -> str:
    dels = "; ".join(json.dumps(s[a:b]) for a, b in measure_spans(s))
    keep = "; ".join(json.dumps(p) for p in protected(s)) or "(none)"
    return f"Sentence: {s}\nDelete: {dels}\nKeep: {keep}"


class Engine:
    def __init__(self, model: str, max_len: int, max_seqs: int):
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        # Qwen3.6 is hybrid (Gated DeltaNet): per-sequence mamba state is large, and vLLM's
        # cudagraph profiling allocates it for max_num_seqs up front (256 OOMs on 32 GB)
        self.llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=0.90,
                       max_num_seqs=max_seqs, enable_prefix_caching=True, seed=0,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        shots = []
        for s, _, _, o in FEW_SHOT:
            shots += [{"role": "user", "content": user_msg(s)}, {"role": "assistant", "content": o}]
        self.prefix = {"edit": [{"role": "system", "content": SYSTEM}, *shots]}
        shots = []
        for s, o in RESTRUCTURE_SHOTS:
            shots += [{"role": "user", "content": user_msg(s)}, {"role": "assistant", "content": o}]
        self.prefix["restructure"] = [{"role": "system", "content": RESTRUCTURE_SYSTEM}, *shots]
        shots = []
        for s, d, o in JUDGE_SHOTS:
            shots += [{"role": "user", "content": judge_msg(s, d)}, {"role": "assistant", "content": json.dumps(o)}]
        self.judge_prefix = [{"role": "system", "content": JUDGE_SYSTEM}, *shots]

    def judge(self, pairs: list[tuple[str, str]]) -> list[dict | None]:
        if not pairs:
            return []
        convs = [[*self.judge_prefix, {"role": "user", "content": judge_msg(s, d)}] for s, d in pairs]
        outs = self.llm.chat(convs, self.SP(temperature=0.0, max_tokens=60), use_tqdm=True,
                             chat_template_kwargs={"enable_thinking": False})
        res = []
        for o in outs:
            m = re.search(r"\{.*\}", o.outputs[0].text, re.S)
            try:
                res.append(json.loads(m.group(0)) if m else None)
            except json.JSONDecodeError:
                res.append(None)
        return res

    def run(self, sentences: list[str], temperature: float, seed: int, think: bool,
            mode: str = "edit") -> list[str]:
        convs = [[*self.prefix[mode], {"role": "user", "content": user_msg(s)}] for s in sentences]
        params = self.SP(temperature=temperature, top_p=0.95 if temperature else 1.0,
                         max_tokens=6000 if think else 200, seed=seed)
        outs = self.llm.chat(convs, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": think})
        res = []
        for o in outs:
            t = o.outputs[0].text if o.outputs[0].finish_reason == "stop" else ""
            t = re.sub(r"(?s)^.*</think>", "", t).strip()
            res.append(t)
        return res


def read_tsv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-tsv", type=Path, required=True, help="slot0clean train TSV (251,599 rows)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=str(next(Path.home().glob(
        ".cache/huggingface/hub/models--cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/*"), "")))
    ap.add_argument("--attempts", type=int, default=4, help="restructure passes after the two edit passes")
    ap.add_argument("--think-attempts", type=int, default=2, help="thinking restructure passes for what is left")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N unique measure sentences only (dry run)")
    ap.add_argument("--sample-seed", type=int, default=-1, help="with --limit: random sample instead of first N")
    ap.add_argument("--skip-llm", action="store_true", help="assemble from sentence_log.jsonl only")
    ap.add_argument("--unresolved", choices=["block", "drop_sentence"], default="block",
                    help="block: do not write the corpus; drop_sentence: remove those sentences (listed)")
    ap.add_argument("--inventory", action="store_true", help="print measure/protect/unknown-digit counts and exit")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = read_tsv(args.clean_tsv)
    VOCAB.update(w for r in base for w in words(r["caption"]))
    units: dict[str, None] = {}
    unknown = []
    for r in base:
        for s in SENT_SPLIT.split(r["caption"])[0::2]:
            if MEASURE.search(s):
                units.setdefault(s, None)
            unknown += unknown_digits(s)
    if args.inventory:
        prot = Counter(re.sub(r"\d", "N", p.lower()) for r in base for p in protected(r["caption"]))
        print(json.dumps({"rows": len(base), "measure_sentences_unique": len(units),
                          "unknown_digit_spans": len(unknown), "protected": prot.most_common()}, indent=1))
        for u in unknown:
            print("UNK:", u)
        return 0

    todo = sorted(units)
    if args.limit:
        if args.sample_seed >= 0:
            import random
            todo = sorted(random.Random(args.sample_seed).sample(todo, args.limit))
        else:
            todo = todo[: args.limit]
    print(json.dumps({"rows": len(base), "measure_sentences_unique": len(units), "todo": len(todo),
                      "unknown_digit_spans_kept": len(unknown), "model": args.model}), flush=True)

    log_path = args.out_dir / "sentence_log.jsonl"
    judge_path = args.out_dir / "judge_log.jsonl"
    verdicts: dict[tuple[str, str], dict | None] = {}
    if judge_path.exists():
        for line in judge_path.read_text().splitlines():
            j = json.loads(line)
            verdicts[(j["src"], j["dst"])] = j["verdict"]

    def judged_ok(src: str, dst: str, mode: str | None) -> bool:
        # in edit (delete-only) mode the word-exact gate already guarantees no description
        # word outside the measurements is lost; the judge's lost_description over-fires there
        v = verdicts.get((src, dst))
        return bool(v) and v.get("grammatical") is True and v.get("added_description") is False \
            and (mode != "restructure" or v.get("lost_description") is False or verb_only_loss(src, dst))

    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            rec = json.loads(line)
            relaxed = rec.get("mode") == "restructure"
            ok = (rec["dst"] is None and (none_ok(rec["src"]) or relaxed and none_ok_relaxed(rec["src"]))
                  or rec["dst"] and not gate(rec["src"], rec["dst"], relaxed))
            if rec["method"] != "unresolved" and ok and rec["src"] in units:   # revalidate under the current gates
                done[rec["src"]] = rec
    last_reason: dict[str, object] = {}
    eng = None

    def run_judge(recs: list[dict]) -> None:
        todo_j = [(r["src"], r["dst"]) for r in recs if r["dst"] and (r["src"], r["dst"]) not in verdicts]
        todo_j = list(dict.fromkeys(todo_j))
        if not todo_j:
            return
        out = eng.judge(todo_j)
        with judge_path.open("a") as fh:
            for (s, d), v in zip(todo_j, out):
                verdicts[(s, d)] = v
                fh.write(json.dumps({"src": s, "dst": d, "verdict": v}, ensure_ascii=False) + "\n")

    todo_set = set(todo)
    if not args.skip_llm and any(s in todo_set and r["dst"] and (s, r["dst"]) not in verdicts for s, r in done.items()):
        eng = Engine(args.model, args.max_len, args.max_seqs)
        run_judge([r for s, r in done.items() if s in todo_set])
    for s in [s for s, r in done.items() if r["dst"] and not judged_ok(s, r["dst"], r.get("mode"))]:
        last_reason[s] = ["judge", verdicts.get((s, done[s]["dst"])), done[s]["dst"]]
        del done[s]
    pending = [s for s in todo if s not in done]
    print(json.dumps({"after_log_and_judge": len(todo) - len(pending), "pending": len(pending)}), flush=True)

    if pending and not args.skip_llm:
        eng = eng or Engine(args.model, args.max_len, args.max_seqs)
        ladder = [("edit", 0.0, False), ("edit", 0.4, False)]
        ladder += [("restructure", t, False) for t in (0.0, 0.3, 0.5, 0.7)][: args.attempts]
        ladder += [("restructure", 0.6, True)] * args.think_attempts
        for attempt, (mode, temp, think) in enumerate(ladder):
            if not pending:
                break
            relaxed = mode == "restructure"
            outs = eng.run(pending, temp, seed=100 + attempt, think=think, mode=mode)
            cand, still = {}, []
            for s, o in zip(pending, outs):
                if o.strip() == "NONE":
                    if none_ok(s) or (relaxed and none_ok_relaxed(s)):
                        cand[s] = {"src": s, "dst": None, "method": "llm_none", "attempt": attempt, "mode": mode, "think": think}
                    else:
                        last_reason[s] = ["none_rejected"]
                elif o:
                    o = tidy(o)
                    why = gate(s, o, relaxed)
                    if not why:
                        cand[s] = {"src": s, "dst": o, "method": "llm", "attempt": attempt, "mode": mode, "think": think}
                    else:
                        last_reason[s] = why + [o]
                else:
                    last_reason[s] = ["empty_or_truncated"]
            run_judge(list(cand.values()))
            with log_path.open("a") as fh:
                for s in pending:
                    rec = cand.get(s)
                    if rec and rec["dst"] and not judged_ok(s, rec["dst"], mode):
                        last_reason[s] = ["judge", verdicts.get((s, rec["dst"])), rec["dst"]]
                        rec = None
                    if rec:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done[s] = rec
                    else:
                        still.append(s)
            pending = still
            print(json.dumps({"attempt": attempt, "mode": mode, "temp": temp, "think": think,
                              "accepted": len(todo) - len(pending), "pending": len(pending)}), flush=True)
    unresolved = [{"src": s, "last_reason": last_reason.get(s)} for s in todo if s not in done]
    (args.out_dir / "unresolved.json").write_text(json.dumps(unresolved, ensure_ascii=False, indent=1))
    if args.limit or (unresolved and args.unresolved == "block"):
        print(json.dumps({"status": "not_written", "limit": args.limit, "unresolved": len(unresolved)}))
        return 0 if args.limit else 3
    for u in unresolved:
        done[u["src"]] = {"src": u["src"], "dst": None, "method": "unresolved_dropped"}

    out_rows, excluded, stats = [], [], Counter()
    for r in base:
        kept = []
        for s in SENT_SPLIT.split(r["caption"])[0::2]:
            if not MEASURE.search(s):
                kept.append(s)
                continue
            rec = done[s]
            stats[rec["method"]] += 1
            if rec["dst"]:
                kept.append(rec["dst"])
        if not kept:
            excluded.append({"id": r["id"], "caption": r["caption"]})
            continue
        out_rows.append({**r, "caption": " ".join(kept)})

    # audits: no measurement left; protected tokens identical per row; measure-free
    # sentences byte-identical and in order; no respelled style words; no new injection
    base_by_id = {r["id"]: r["caption"] for r in base}
    for r in out_rows:
        src = base_by_id[r["id"]]
        if MEASURE.search(r["caption"]):
            raise SystemExit(f"[FAIL] measurement left in {r['id']}: {r['caption']!r}")
        if Counter(protected(src)) != Counter(protected(r["caption"])):
            raise SystemExit(f"[FAIL] protected tokens changed in {r['id']}")
        if len(DECADE_WORDS.findall(r["caption"])) > len(DECADE_WORDS.findall(src)):
            raise SystemExit(f"[FAIL] style word respelled in {r['id']}")
        pos = 0
        for s in (x for x in SENT_SPLIT.split(src)[0::2] if not MEASURE.search(x)):
            idx = r["caption"].find(s, pos)
            if idx < 0:
                raise SystemExit(f"[FAIL] measure-free sentence lost/reordered in {r['id']}: {s!r}")
            pos = idx + len(s)
        if INJECTION.search(r["caption"]) and not INJECTION.search(src):
            raise SystemExit(f"[FAIL] injection marker introduced in {r['id']}")

    out_tsv = args.out_dir / "phase8_caption2p0_slot0nmv2_train.tsv"
    tmp = out_tsv.with_name("." + out_tsv.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(base[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp, out_tsv)
    (args.out_dir / "excluded_rows.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=1))
    with (args.out_dir / "final_sentences.jsonl").open("w") as fh:   # what the corpus actually uses
        for s in sorted(units):
            r = done[s]
            fh.write(json.dumps({"src": s, "dst": r["dst"], "method": r["method"], "mode": r.get("mode", "edit"),
                                 "think": bool(r.get("think"))}, ensure_ascii=False) + "\n")
    prot_in = Counter(p for c in base_by_id.values() for p in protected(c))
    prot_out = Counter(p for r in out_rows for p in protected(r["caption"]))
    summary = {"status": "rewrite_complete_not_released", "rows_in": len(base), "rows_out": len(out_rows),
               "excluded_all_sentences_dropped": len(excluded),
               "rows_changed": sum(1 for r in out_rows if r["caption"] != base_by_id[r["id"]]),
               "sentence_methods": stats, "unique_sentences": Counter(v["method"] for v in done.values()),
               "restructure_accepted_unique": sum(1 for v in done.values() if v.get("mode") == "restructure"),
               "judged_pairs": len(verdicts),
               "unresolved_policy": args.unresolved, "unresolved_unique": len(unresolved),
               "protected_tokens_in": sum(prot_in.values()), "protected_tokens_out": sum(prot_out.values()),
               "decade_words_in": sum(len(DECADE_WORDS.findall(c)) for c in base_by_id.values()),
               "decade_words_out": sum(len(DECADE_WORDS.findall(r["caption"])) for r in out_rows),
               "model": args.model}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
