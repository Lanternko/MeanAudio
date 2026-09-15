#!/usr/bin/env python
"""One regeneration round for flagged slot0 rows: one candidate caption per id.

Same captioner (Qwen2.5-Omni-3B, pinned revision), same PROMPT, same 10 s crop
as the original corpus; only the seed changes per attempt (seed + attempt*1000,
matching regen_multisent_defect_ids.py). Candidates cut at max_new_tokens or
failing the structural classifier (multiline included; never
cut to the first line) are recorded as failures and never sent to
the semantic audit. Acceptance is decided by slot0_regen_loop.py, not here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_qwen_caption_10s_multisent import (  # noqa: E402
    MODEL_REVISION,
    PROMPT,
    caption_batch,
    load_crop,
    load_model,
)
from regen_multisent_defect_ids import selftest_stops_at_im_end  # noqa: E402
from repair_multisent_first_entity_line import classify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, required=True, help="one id per line")
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = [l.strip() for l in args.ids.read_text().splitlines() if l.strip()]
    seed = args.seed + args.attempt * 1000
    print(f"attempt {args.attempt}: {len(ids)} ids, seed {seed}", flush=True)
    model, processor = load_model()
    meta = {}
    for cid in ids:
        path, crop = load_crop(cid)
        meta[cid] = (str(path), crop)
    probe = ids[:8]
    selftest_stops_at_im_end(model, processor, [meta[c][1] for c in probe], [meta[c][0] for c in probe],
                             max_new_tokens=args.max_new_tokens)

    tmp = args.out_jsonl.with_name(f".{args.out_jsonl.name}.tmp.{os.getpid()}")
    n_struct = n_trunc = 0
    with tmp.open("w", encoding="utf-8") as f:
        for i in range(0, len(ids), args.batch_size):
            chunk = ids[i:i + args.batch_size]
            caps, truncs = caption_batch(model, processor, [meta[c][0] for c in chunk],
                                         [meta[c][1] for c in chunk], seed + i, args.max_new_tokens,
                                         return_truncation=True)
            for cid, raw, trunc in zip(chunk, caps, truncs):
                # No first-line cutting: a multiline candidate is rejected, not repaired.
                # too_short is not a defect under the current audit policy.
                cap = raw.strip()
                defects = ["hit_max_new_tokens"] if trunc else [t for t in classify(cap) if t != "too_short"]
                n_trunc += int(trunc)
                n_struct += int(bool(defects) and not trunc)
                f.write(json.dumps({"id": cid, "caption": None if defects else cap, "raw": raw,
                                    "structural_defects": defects, "attempt": args.attempt,
                                    "seed": seed + i, "prompt": PROMPT,
                                    "model_revision": MODEL_REVISION}, ensure_ascii=False) + "\n")
            if (i // args.batch_size) % 50 == 0:
                print(f"  {min(i + args.batch_size, len(ids))}/{len(ids)}", flush=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out_jsonl)
    print(json.dumps({"attempt": args.attempt, "ids": len(ids), "structural_fail": n_struct,
                      "truncated": n_trunc}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
