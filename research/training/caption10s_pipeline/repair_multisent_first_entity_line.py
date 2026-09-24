#!/usr/bin/env python
"""Repair multisent captions truncated by the Qwen-Omni no-EOS generation bug.

Root cause: Qwen2_5OmniThinkerForConditionalGeneration.generation_config.eos_token_id
is None, so generate() never stops and runs to max_new_tokens. The model does emit
<|im_end|> after the caption, but generation continues into a new conversation turn.
batch_decode(skip_special_tokens=True) deletes <|im_end|> instead of truncating there,
gluing caption + junk together across a newline.

Repair rule (deliberately structural, NOT leak-string matching):
  normalize newlines -> take the FIRST non-empty line -> collapse inner whitespace.
This is provably equivalent to truncating at <|im_end|>: with a corrected EOS the
model emits single-paragraph captions (0/32 contained a newline), and the naive
decode's first line ended in sentence-final punctuation 64/64.

Writes atomically. Also emits the id list of rows that still fail quality checks
after truncation; those are regenerated separately with a fixed EOS, never edited.
"""
import argparse
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

REPAIR_VERSION = "first_entity_line_v1"
ROOT_CAUSE = "qwen2_5_omni_generation_config_eos_token_id_none"

# CJK / kana / hangul / fullwidth — explicit ranges, so emoji are reported separately
CJK_RE = re.compile(
    r"[⺀-⻿　-〿぀-ヿ㄀-ㄯ㄰-㆏"
    r"ㇰ-ㇿ㐀-䶿一-鿿ꥠ-꥿가-힯"
    r"豈-﫿︰-﹏＀-￯]"
)
# conversation-turn / instruction-leak markers used by the pipeline gate
TURN_MARKERS = [
    "Human:", "Assistant:", "User:", "System:",
    "Compute the", "Write a Python",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
]
# Lead-in-only output is unusable because the actual caption was emitted after
# the newline discarded by the structural repair.  Content-bearing lead-ins are
# intentionally retained: selecting them for regeneration would introduce a new
# style-based filter that was not applied to the comparison arm.
LEAD_IN_CORE = (
    r"(?:here(?:'s| is| are)(?: the)?(?: requested)?(?: caption| caption text)?|"
    r"the caption should include the following details|"
    r"the caption for (?:the )?music excerpt provided would be|"
    r"the caption(?: text)?(?: for| of)?(?: the| this)?(?: described)?(?: music| audio)?"
    r"(?: clip| excerpt| piece)?(?: would be| could be| should be| is)(?: as follows| provided below)?|"
    r"caption(?: text)?|sure|okay|ok)"
)
DEGENERATE_RE = re.compile(
    rf"^\s*(?:\*+\s*)?{LEAD_IN_CORE}\s*[:：]\s*(?:\*+)?\s*$", re.I
)
LEAD_IN_PREFIX_RE = re.compile(
    rf"^\s*(?:\*+\s*)?{LEAD_IN_CORE}\s*[:：]", re.I
)
JSON_WRAPPER_RE = re.compile(r"^\s*\{.*\}\s*$", re.S)
CHAR_RUN_RE = re.compile(r"([^\w\s])\1{3,}")
MARKDOWN_RE = re.compile(r"(?:```|^\s{0,3}#{1,6}\s|^\s*[-*+]\s+|\*\*[^*]+\*\*)", re.M)
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
LATEX_RE = re.compile(r"(?:\\begin\{|\\end\{|\\frac\{|\$\$)")
TERMINAL_RE = re.compile(r"[.!](?:[\"'”’\)\]\}]+)?$")
DISCLAIMER_RE = re.compile(r"\(\s*note\s*:", re.I)
MIN_WORDS = 5


def first_entity_line(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    for line in t.split("\n"):
        if line.strip():
            return re.sub(r"[ \t ]+", " ", line).strip()
    return ""


def n_sents(text: str) -> int:
    return max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))


def classify(cap: str) -> list[str]:
    """Return list of defect tags; empty means the row is clean."""
    tags = []
    if not cap:
        tags.append("empty")
        return tags
    if CJK_RE.search(cap):
        tags.append("cjk")
    if DEGENERATE_RE.match(cap):
        tags.append("degenerate_leadin")
    if JSON_WRAPPER_RE.match(cap):
        tags.append("json_wrapper")
    if CHAR_RUN_RE.search(cap):
        tags.append("character_run")
    if MARKDOWN_RE.search(cap):
        tags.append("markdown_wrapper")
    if URL_RE.search(cap):
        tags.append("url")
    if LATEX_RE.search(cap):
        tags.append("latex")
    if len(cap.split()) < MIN_WORDS:
        tags.append("too_short")
    for m in TURN_MARKERS:
        if m in cap:
            tags.append("turn_marker")
            break
    if "\n" in cap or "\r" in cap:
        tags.append("multiline")
    if cap.lower().count("the caption for this music clip is:") >= 3:
        tags.append("repeated_leadin")
    if LEAD_IN_PREFIX_RE.match(cap) and cap.rstrip().endswith("]"):
        tags.append("bracket_wrapper")
    if DISCLAIMER_RE.search(cap):
        tags.append("meta_disclaimer")
    if cap.rstrip().endswith("?"):
        tags.append("question_terminal")
    elif len(cap.split()) >= 100 and not TERMINAL_RE.search(cap.rstrip()):
        tags.append("no_terminal_punctuation")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--backup_sha256", required=True)
    ap.add_argument("--regen_ids", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    with args.in_jsonl.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    changed = 0
    defects: dict[str, list[str]] = {}
    words, sents = [], []
    out_rows = []
    for r in rows:
        old = (r.get("caption") or "").strip()
        new = first_entity_line(old)
        did_change = new != old
        changed += int(did_change)

        tags = classify(new)
        if tags:
            defects[r["id"]] = tags

        rec = dict(r)
        rec["caption"] = new
        rec["n_chars"] = len(new)
        rec["n_words"] = len(new.split())
        rec["n_sents"] = n_sents(new)
        rec["repair"] = {
            "version": REPAIR_VERSION,
            "applied_at": now,
            "root_cause": ROOT_CAUSE,
            "rule": "normalize newlines; keep first non-empty line; collapse inner whitespace",
            "changed": did_change,
            "orig_n_chars": len(old),
            "orig_n_words": len(old.split()),
            "source_backup": args.backup.name,
            "source_sha256": args.backup_sha256,
        }
        out_rows.append(rec)
        words.append(rec["n_words"])
        sents.append(rec["n_sents"])

    tag_counts: dict[str, int] = {}
    for tags in defects.values():
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    report = {
        "repair_version": REPAIR_VERSION,
        "applied_at": now,
        "in_rows": len(rows),
        "changed_rows": changed,
        "changed_pct": round(changed / max(len(rows), 1), 6),
        "backup": str(args.backup),
        "backup_sha256": args.backup_sha256,
        "stats_after": {
            "mean_words": round(st.mean(words), 2),
            "median_words": st.median(words),
            "mean_sents": round(st.mean(sents), 3),
            "still_1sent_frac": round(sum(1 for s in sents if s == 1) / len(sents), 4),
        },
        "defect_rows": len(defects),
        "defect_tag_counts": tag_counts,
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        print("[DRY RUN] nothing written")
        return 0

    tmp = args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(args.out_jsonl)

    args.regen_ids.write_text(
        "".join(f"{i}\t{','.join(t)}\n" for i, t in sorted(defects.items())),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_jsonl}")
    print(f"wrote {args.regen_ids} ({len(defects)} ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
