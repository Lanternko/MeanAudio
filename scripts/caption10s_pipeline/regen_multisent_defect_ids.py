#!/usr/bin/env python
"""Regenerate the defect ids from the multisent corpus with a corrected EOS setting.

Defect rows are never hand-edited, translated or stripped — they are regenerated
from audio and atomically merged back. Any row still failing validation after
--max_attempts aborts the run without touching the corpus.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from gen_qwen_caption_10s_multisent import (  # noqa: E402
    PROMPT,
    VARIANT,
    WINDOW_SEC,
    caption_batch,
    load_crop,
    load_model,
    n_sents,
    resolve_stop_ids,
)
from repair_multisent_first_entity_line import (  # noqa: E402
    MIN_WORDS,
    classify,
    first_entity_line,
)


def selftest_stops_at_im_end(model, processor, crops, paths, max_new_tokens=160):
    """Assert generation actually terminates on <|im_end|> instead of running to the cap."""
    tok, eos_id, pad_id = resolve_stop_ids(processor)
    print(f"[selftest] eos_id(<|im_end|>)={eos_id} pad_id={pad_id}", flush=True)
    print(f"[selftest] model.generation_config.eos_token_id={model.generation_config.eos_token_id} "
          f"(None here is exactly the bug being guarded against)", flush=True)
    conversations = [
        [{"role": "user", "content": [{"type": "audio", "audio": p},
                                      {"type": "text", "text": PROMPT}]}]
        for p in paths
    ]
    texts = [processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
             for c in conversations]
    inputs = processor(text=texts, audio=crops, return_tensors="pt", padding=True,
                       sampling_rate=16000).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.8, eos_token_id=eos_id, pad_token_id=pad_id)
    gen = out[:, inputs.input_ids.size(1):]
    stopped, lens = 0, []
    for row in gen.tolist():
        pos = next((j for j, t in enumerate(row) if t in (eos_id, pad_id)), None)
        stopped += int(pos is not None)
        lens.append(pos if pos is not None else len(row))
    n = gen.size(0)
    print(f"[selftest] stopped_before_cap={stopped}/{n}  lens={lens}", flush=True)
    assert stopped == n, (
        f"generation did not stop at <|im_end|> for {n - stopped}/{n} samples — "
        "refusing to regenerate with a broken stop condition"
    )
    assert max(lens) < max_new_tokens, "some sample ran to the token cap"
    print("[selftest] PASS — generation stops at <|im_end|>", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen_ids", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--max_attempts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids, tags = [], {}
    for line in args.regen_ids.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, _, tg = line.partition("\t")
        ids.append(cid)
        tags[cid] = tg
    print(f"regen targets: {len(ids)}", flush=True)

    model, processor = load_model()

    meta = {}
    for cid in ids:
        try:
            path, crop = load_crop(cid)
            meta[cid] = (str(path), crop)
        except FileNotFoundError as e:
            print(f"[FAIL] missing audio for {cid}: {e}", flush=True)
            return 2

    probe = ids[:8]
    selftest_stops_at_im_end(
        model, processor,
        [meta[c][1] for c in probe], [meta[c][0] for c in probe],
        max_new_tokens=args.max_new_tokens,
    )

    accepted: dict[str, str] = {}
    pending = list(ids)
    for attempt in range(1, args.max_attempts + 1):
        if not pending:
            break
        seed = args.seed + attempt * 1000
        print(f"\n[attempt {attempt}] pending={len(pending)} seed={seed}", flush=True)
        still = []
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            caps = caption_batch(
                model, processor,
                [meta[c][0] for c in chunk], [meta[c][1] for c in chunk],
                seed, args.max_new_tokens,
            )
            for cid, raw in zip(chunk, caps):
                cap = first_entity_line(raw)
                if classify(cap):
                    still.append(cid)
                else:
                    accepted[cid] = cap
        print(f"[attempt {attempt}] accepted={len(accepted)}/{len(ids)} "
              f"still_failing={len(still)}", flush=True)
        pending = still

    if pending:
        print(f"\n[FAIL] {len(pending)} ids still failing after {args.max_attempts} attempts:")
        for cid in pending[:30]:
            print(f"  {cid}  tags={tags.get(cid)}")
        print("aborting — corpus NOT modified")
        return 3

    now = datetime.now(timezone.utc).isoformat()
    tmp = args.out_jsonl.with_name(f".{args.out_jsonl.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for cid in ids:
            cap = accepted[cid]
            f.write(json.dumps({
                "id": cid,
                "caption": cap,
                "n_chars": len(cap),
                "n_words": len(cap.split()),
                "n_sents": n_sents(cap),
                "audio_path": meta[cid][0],
                "window_sec": WINDOW_SEC,
                "max_new_tokens": args.max_new_tokens,
                "prompt": PROMPT,
                "variant": VARIANT,
                "regen": {
                    "reason": tags.get(cid),
                    "at": now,
                    "fix": "eos_token_id=<|im_end|>, pad_token_id set, decode truncated at stop id",
                },
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out_jsonl)
    print(f"\nwrote {args.out_jsonl} ({len(accepted)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
