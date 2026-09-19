#!/usr/bin/env python3
"""Independent local review of the slot0nmv2 rewrite (reviewer != writer model).

The writer is Qwen3.6-27B; the reviewer is Qwen2.5-32B-Instruct-AWQ. A second LLM is
not ground truth - this estimates rates with CIs, it does not certify the corpus.
Three things are measured separately (feedback: residual / wrongful deletion / audio):

  E  edits:     uniform sample of changed sentences (src -> dst). Reviewer answers
                lost_info (non-measurement description removed), added_info, broken
                grammar, measurement_left.
  U  untouched: uniform sample of rows the regex left alone. Reviewer answers whether a
                measurement (tempo number/BPM, meter, key/mode, Hz/dB, duration) is still
                there -> estimates what the regex misses.
Audio agreement is not measurable from text and is not claimed.
Writes review.json (counts + Clopper-Pearson 95%) and review_flags.jsonl for reading.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewrite_slot0nmv2_measurements as W  # noqa: E402
from slot0_audit_crosscheck import clopper_pearson  # noqa: E402

EDIT_SYSTEM = """\
You audit an automatic edit of one sentence from a music caption. The edit was supposed to
delete ONLY musical measurements (tempo numbers/BPM, time signature/meter, musical key,
mode or chord quality, note names, Hz, dB, durations/timestamps) and keep everything else.
Decades ("80s", "1980s"), "8-bit", "808", "12-bar" are style words and must be kept as written.
Answer with JSON only:
{"lost_info": true|false, "added_info": true|false, "broken_grammar": true|false,
 "measurement_left": true|false, "note": "<short reason if any flag is true>"}
lost_info = a non-measurement description (instrument, genre, mood, texture, style, era,
qualitative tempo word like fast/slow, production) from ORIGINAL is missing in EDITED.
added_info = EDITED states something ORIGINAL did not."""

KEEP_SYSTEM = """\
You check one music caption. Does it still state a musical MEASUREMENT: a tempo number or
BPM value or BPM range, a time signature or meter, a musical key, mode or chord quality
(e.g. "in C major", "minor key", "minor chords"), a note name, Hz, dB, or a duration/timestamp?
Decades ("80s"), "8-bit", "808", "12-bar", and qualitative words (fast, slow, upbeat) are NOT
measurements. Answer with JSON only: {"measurement": true|false, "quote": "<exact words or empty>"}"""


def parse(txt: str) -> dict | None:
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="rewrite out dir (has final_sentences.jsonl)")
    ap.add_argument("--clean-tsv", type=Path, required=True)
    ap.add_argument("--n-edit", type=int, default=600, help="delete-only edits (stratum 1)")
    ap.add_argument("--n-restructure", type=int, default=300, help="restructured edits (stratum 2)")
    ap.add_argument("--n-untouched", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026091901)
    ap.add_argument("--model", default=str(next(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/*"), "")))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(x) for x in (args.out_dir / "final_sentences.jsonl").read_text().splitlines()]
    pop = {m: sorted((r for r in recs if r["dst"] and r["mode"] == m), key=lambda r: r["src"])
           for m in ("edit", "restructure")}
    want = {"edit": args.n_edit, "restructure": args.n_restructure}
    edits = [r for m in pop for r in rng.sample(pop[m], min(want[m], len(pop[m])))]
    base = W.read_tsv(args.clean_tsv)
    untouched = [r for r in base if not W.MEASURE.search(r["caption"])]
    untouched = rng.sample(untouched, min(args.n_untouched, len(untouched)))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    conv_e = [[{"role": "system", "content": EDIT_SYSTEM},
               {"role": "user", "content": f"ORIGINAL: {r['src']}\nEDITED: {r['dst']}"}] for r in edits]
    conv_u = [[{"role": "system", "content": KEEP_SYSTEM},
               {"role": "user", "content": r["caption"]}] for r in untouched]
    out_e = [parse(o.outputs[0].text) for o in llm.chat(conv_e, sp, use_tqdm=True)]
    out_u = [parse(o.outputs[0].text) for o in llm.chat(conv_u, sp, use_tqdm=True)]

    flags = []
    res = {"reviewer": args.model, "seed": args.seed, "edits": {}, "untouched": {}}
    n_e = sum(o is not None for o in out_e)
    res["edits"]["unparsed"] = len(out_e) - n_e
    # stratified: per-mode rates with CIs, overall = population-weighted
    total = sum(len(v) for v in pop.values())
    for key in ("lost_info", "added_info", "broken_grammar", "measurement_left"):
        est = 0.0
        for mode in pop:
            sub = [o for r, o in zip(edits, out_e) if r["mode"] == mode and o is not None]
            x = sum(bool(o.get(key)) for o in sub)
            res["edits"][f"{mode}:{key}"] = {"x": x, "n": len(sub), "pop": len(pop[mode]),
                                             "rate_ci95": clopper_pearson(x, len(sub)) if sub else None}
            est += (x / len(sub) if sub else 0) * len(pop[mode]) / total
        res["edits"][f"weighted:{key}"] = round(est, 4)
    for r, o in zip(edits, out_e):
        if o is None or any(o.get(k) for k in ("lost_info", "added_info", "broken_grammar", "measurement_left")):
            flags.append({"set": "edit", "mode": r["mode"], "think": r["think"], "src": r["src"], "dst": r["dst"], "review": o})
    n_u = sum(o is not None for o in out_u)
    x_u = sum(bool(o and o.get("measurement")) for o in out_u)
    res["untouched"] = {"measurement": {"x": x_u, "n": n_u, "rate_ci95": clopper_pearson(x_u, n_u)},
                        "unparsed": len(out_u) - n_u}
    for r, o in zip(untouched, out_u):
        if o is None or o.get("measurement"):
            flags.append({"set": "untouched", "id": r["id"], "caption": r["caption"], "review": o})
    (args.out_dir / "review.json").write_text(json.dumps(res, indent=1))
    with (args.out_dir / "review_flags.jsonl").open("w") as fh:
        for f in flags:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
