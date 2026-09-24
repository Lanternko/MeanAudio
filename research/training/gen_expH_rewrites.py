"""
EXP-H: Qwen slot0 → LP-MC acoustic-structure style rewrite
===========================================================

Uses Qwen2.5-Omni-3B in text-only chat mode to rewrite Qwen captions
into LP-MusicCaps acoustic-structure style, preserving semantic content
(instruments, genre, mood, tempo) while adopting LP-MC surface form.

Hypothesis: collapse is caused by caption *style*, not caption *content*.
If style transfer works, EXP-H should recover MC CLAP ≥ 0.15.

Usage:
  # 10K sanity (default):
  python gen_expH_rewrites.py \
      --out ~/eval_tsvs_p100/expH_rewrite_10k_sanity.tsv \
      --n 10000

  # Full 251K run:
  python gen_expH_rewrites.py \
      --out ~/eval_tsvs_p100/expH_rewrite_train.tsv \
      --n all

  # Resume a partial run:
  python gen_expH_rewrites.py \
      --out ~/eval_tsvs_p100/expH_rewrite_10k_sanity.tsv \
      --n 10000 --resume

Output TSV columns: id, caption, q_level  (q_level=5 for all rows)

Notes:
  - Source captions: phase9_omni_captions.jsonl slot0
  - 10K sanity sample uses seed=42 (reproducible)
  - Full run processes ALL 251,599 clips in canonical TSV order
  - Text-only inference: no audio loading, ~4-8x faster than captioning
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────
QWEN_JSONL   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl')
CANONICAL_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase9_5_train.tsv')  # id order
MODEL_ID     = 'Qwen/Qwen2.5-Omni-3B'
SEED         = 42
BATCH_SIZE   = 32
MAX_NEW_TOKENS = 80
Q_LEVEL      = 5   # mid-quality level for all rewrites

# ── Few-shot prompt (8 examples, 8 diverse genres) ───────────────────────────
# Each example shows: abstract Qwen style → LP-MC acoustic-structure style.
# Transformation rules illustrated:
#   1. LP-MC opening: "The low quality recording features a..." or "This is a [genre] piece."
#   2. Concrete instrument vocabulary: punchy kick, shimmering hi hats, groovy bass, synth pad,
#      arpeggiated guitar, sustained strings, distorted electric guitar, brushed drums, etc.
#   3. Length expansion: ~15 words → 30-45 words
#   4. Mood appendage: "It sounds [adj]." or "The recording is noisy and in mono."
#   5. NO new instruments beyond what Qwen mentioned

FEW_SHOT_SYSTEM = """\
You are a music caption rewriter. Convert a short music description into LP-MusicCaps \
acoustic-structure style. Follow these rules exactly:
- Open with "The low quality recording features a..." or "This is a [genre] piece."
- Use concrete instrument vocabulary: punchy kick, snare hits, shimmering hi hats, \
groovy bass, synth pad, arpeggiated guitar, sustained strings melody, distorted electric guitar, etc.
- Preserve ALL content from the input (genre, instruments, mood, tempo). Do NOT invent new instruments.
- Target 30–45 words total.
- End with "It sounds [adjective]." or "The recording is noisy and in mono." or both."""

FEW_SHOT_EXAMPLES = [
    # 1. Electronic dance
    (
        "A lively electronic track with fast tempo and synthesized sounds is played.",
        "The low quality recording features an energetic electronic dance track with a driving "
        "synth lead, punchy kick drum hits, shimmering hi hats and a groovy synth bass line. "
        "It sounds lively and exhilarating.",
    ),
    # 2. Jazz fusion
    (
        "An easy listening, jazz fusion song with a laidback beat, featuring a rhythmic "
        "electric piano and soft drums.",
        "This is a jazz fusion piece. There is a smooth electric piano melody with a syncopated "
        "groove. Soft brushed drums and a walking bass line accompany the laid-back feel. "
        "The atmosphere is relaxed and soulful.",
    ),
    # 3. Latin / world
    (
        "This track is a lively Latin piece, primarily featuring guitar and percussion, "
        "with an upbeat and rhythmic ambiance suitable for a festive setting.",
        "The low quality recording features a lively Latin piece with an arpeggiated acoustic "
        "guitar melody, wooden percussion patterns and shimmering shakers. A groovy bass line "
        "drives the festive rhythm. It sounds energetic and celebratory.",
    ),
    # 4. Orchestral / cinematic
    (
        "The track is a symphony of strings, brass, and drums, creating an epic and uplifting "
        "mood with a steady tempo.",
        "The low quality recording features a cinematic orchestral piece with sustained strings "
        "melody, bold brass section hits and driving snare and kick drums. It sounds epic and "
        "uplifting. The recording is noisy and in mono.",
    ),
    # 5. Punk / rock
    (
        "The clip features a fast-paced instrumental piece that has a punk rock mood and "
        "includes a guitar and drums.",
        "This is a punk rock piece. There is a distorted electric guitar playing a fast, driving "
        "riff with punchy kick and snare hits and shimmering hi hats. A bass guitar underpins "
        "the relentless tempo. The atmosphere is raw and intense.",
    ),
    # 6. Ambient / chill synth
    (
        "A low key and relaxing, synthesizer beat music track with a chill vibe, tempo of 80.",
        "This is a chill ambient piece. There is a gentle synth pad melody drifting over a slow "
        "drum machine beat with a subtle bass line. The atmosphere is hazy and serene. "
        "This piece could be used in a relaxation or study playlist.",
    ),
    # 7. Classical solo
    (
        "The music is a lively harp solo with fast tempo and a playful mood, in the genre "
        "of classical.",
        "The low quality recording features a classical solo performance with a fast arpeggiated "
        "harp melody. No other instruments are present. It sounds playful, bright and nimble. "
        "The recording is noisy and in mono.",
    ),
    # 8. Rock / jazz fusion with big band
    (
        "This is a slow rock and jazz fusion piece featuring a big band feel with piano as "
        "the main instrument.",
        "The low quality recording features a slow rock and jazz fusion piece with a prominent "
        "piano melody, sustained brass section chords and a steady rhythm section of kick drum "
        "and bass. It sounds mellow and sophisticated.",
    ),
]


def build_few_shot_prompt(qwen_caption: str) -> str:
    """Build the full few-shot rewrite prompt for one Qwen slot0 caption."""
    lines = [FEW_SHOT_SYSTEM, ""]
    for inp, out in FEW_SHOT_EXAMPLES:
        lines.append(f"Input: {inp}")
        lines.append(f"Output: {out}")
        lines.append("")
    lines.append(f"Input: {qwen_caption}")
    lines.append("Output:")
    return "\n".join(lines)


def load_qwen_slot0(jsonl_path: Path) -> dict:
    """Return {id: slot0_caption} for all clips."""
    data = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = str(d.get('id', ''))
            caps = d.get('captions', [])
            if cid and caps:
                data[cid] = str(caps[0]).strip()
    return data


def load_canonical_order(tsv_path: Path) -> list:
    """Return list of clip IDs in canonical training order."""
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return [r['id'] for r in reader]


def load_done(tsv_path: Path) -> set:
    """Return set of already-processed IDs from output TSV."""
    done = set()
    if not tsv_path.exists():
        return done
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('id'):
                done.add(row['id'])
    return done


def clean_output(raw: str) -> str:
    """
    Extract the rewrite from model output.
    The model may repeat 'Output:' or continue with 'Input:...' — stop there.
    """
    # Strip leading whitespace / "Output:" prefix if model echoes it
    text = raw.strip()
    if text.lower().startswith("output:"):
        text = text[len("output:"):].strip()

    # Stop at next "Input:" or chat-format markers (Human:/Assistant:)
    # The model sometimes continues the conversation after the rewrite.
    # Catch both newline-prefixed and space-prefixed variants.
    chat_stop = re.search(
        r'(?:\n| )(?:Human|Assistant|User|System)\s*:|'
        r'\nInput:|\nOutput:|\n\nInput:|---',
        text
    )
    if chat_stop:
        text = text[:chat_stop.start()].strip()

    # Stop at hallucinated Q&A / instruction suffixes
    # e.g. "It sounds great. What is the main instrument..."
    #      "It sounds great. Write a summary based on the passage."
    halluc_match = re.search(
        r'\.\s+(?:What |How |Who |Write |Describe |Define |Answer |'
        r'Based on |Given the |Using the |In the (?:passage|context|above)|'
        r'the passage|unanswerable|Identify |Explain )',
        text,
        re.IGNORECASE
    )
    if halluc_match:
        text = text[:halluc_match.start() + 1].strip()  # keep up to the period

    # Remove trailing incomplete sentence (no period at end)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences and not sentences[-1].rstrip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    text = ' '.join(sentences).strip()

    # Fallback: if empty, return a minimal LP-MC template
    if len(text) < 10:
        text = "The low quality recording features an instrumental piece."

    return text


def load_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
    print(f'Loading {MODEL_ID} (text-only mode)...')
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={'': 0},
    )
    model.eval()
    print('  model loaded.')
    return model, processor


def run_batch(model, processor, prompts: list[str]) -> list[str]:
    """
    Run text-only chat inference for a batch of prompts.
    Returns list of cleaned rewrite strings.
    """
    # Build text-only conversations (no audio)
    conversations = [
        [{"role": "user", "content": p}]
        for p in prompts
    ]
    texts = [
        processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        for conv in conversations
    ]
    with torch.no_grad():
        inputs = processor(
            text=texts,
            return_tensors='pt',
            padding=True,
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode only the newly generated tokens
    new_ids = generated_ids[:, inputs.input_ids.size(1):]
    raw_outputs = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [clean_output(r) for r in raw_outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Output TSV path')
    parser.add_argument('--n', default='10000',
                        help='Number of clips to process ("all" for full 251K, or integer)')
    parser.add_argument('--resume', action='store_true', help='Skip already-processed IDs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_run = (args.n.lower() == 'all')
    n_target = None if full_run else int(args.n)

    print(f'EXP-H Rewrite Generator')
    print(f'  Source JSONL : {QWEN_JSONL}')
    print(f'  Output TSV   : {out_path}')
    print(f'  N target     : {"all (251K)" if full_run else n_target}')
    print(f'  Batch size   : {args.batch_size}')
    print(f'  Resume       : {args.resume}')

    # ── Load data ────────────────────────────────────────────────────────────
    print('\nLoading Qwen slot0 captions...')
    qwen_slot0 = load_qwen_slot0(QWEN_JSONL)
    print(f'  {len(qwen_slot0):,} captions loaded')

    print('Loading canonical ID order...')
    all_ids = load_canonical_order(CANONICAL_TSV)
    print(f'  {len(all_ids):,} IDs in canonical order')

    # For 10K sanity: random seed=42 sample
    if not full_run and n_target is not None and n_target < len(all_ids):
        rng = random.Random(args.seed)
        todo_ids = rng.sample(all_ids, n_target)
        print(f'  Sampled {n_target:,} IDs (seed={args.seed})')
    else:
        todo_ids = all_ids
        print(f'  Using all {len(todo_ids):,} IDs')

    # Resume: skip already-done
    if args.resume:
        done = load_done(out_path)
        todo_ids = [cid for cid in todo_ids if cid not in done]
        print(f'  Resume: {len(done):,} done, {len(todo_ids):,} remaining')

    # Filter to IDs that have Qwen captions
    missing = [cid for cid in todo_ids if cid not in qwen_slot0]
    if missing:
        print(f'  WARNING: {len(missing):,} IDs have no Qwen slot0 caption, skipping')
    todo_ids = [cid for cid in todo_ids if cid in qwen_slot0]
    print(f'  Will process: {len(todo_ids):,} clips')

    if not todo_ids:
        print('Nothing to do.')
        return

    # ── Load model ───────────────────────────────────────────────────────────
    model, processor = load_model()

    # ── Write header if new file ─────────────────────────────────────────────
    write_header = not out_path.exists() or not args.resume
    fout = open(out_path, 'a', newline='')
    writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
    if write_header:
        writer.writerow(['id', 'caption', 'q_level'])

    # ── Batch loop ────────────────────────────────────────────────────────────
    n_ok = n_error = 0
    bs = args.batch_size

    for i in tqdm(range(0, len(todo_ids), bs), desc='rewriting'):
        batch_ids = todo_ids[i: i + bs]
        batch_captions = [qwen_slot0[cid] for cid in batch_ids]
        batch_prompts = [build_few_shot_prompt(cap) for cap in batch_captions]

        try:
            rewrites = run_batch(model, processor, batch_prompts)
            for cid, rewrite in zip(batch_ids, rewrites):
                writer.writerow([cid, rewrite, Q_LEVEL])
                n_ok += 1
        except Exception as e:
            tqdm.write(f'[WARN] batch {i//bs} error: {e}')
            # Write fallback for failed batch
            for cid, orig_cap in zip(batch_ids, batch_captions):
                fallback = f"The low quality recording features an instrumental piece. It sounds {_generic_mood(orig_cap)}."
                writer.writerow([cid, fallback, Q_LEVEL])
                n_error += 1

        if (i // bs) % 50 == 0 and i > 0:
            fout.flush()

    fout.flush()
    fout.close()

    print(f'\n=== Done ===')
    print(f'  Written: {n_ok:,}  Fallback: {n_error:,}')
    print(f'  Output : {out_path}')

    # Quick sanity on output
    with open(out_path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    lengths = [len(r['caption'].split()) for r in rows if r.get('caption')]
    if lengths:
        import statistics
        print(f'\n--- Output caption stats ---')
        print(f'  N       : {len(lengths):,}')
        print(f'  mean len: {sum(lengths)/len(lengths):.1f} words')
        print(f'  median  : {statistics.median(lengths):.0f} words')
        print(f'  p10     : {sorted(lengths)[len(lengths)//10]} words')
        print(f'  p90     : {sorted(lengths)[int(len(lengths)*0.9)]} words')

        # Check LP-MC opening adoption
        lp_opening = sum(1 for r in rows if r.get('caption', '').startswith(('The low quality', 'This is a', 'This audio')))
        print(f'  LP-MC opening fraction: {lp_opening/len(rows):.1%}')


def _generic_mood(caption: str) -> str:
    """Extract a generic mood word from a Qwen caption for fallback."""
    for word in ('energetic', 'relaxing', 'upbeat', 'lively', 'calm', 'epic', 'melancholic'):
        if word in caption.lower():
            return word
    return 'expressive'


if __name__ == '__main__':
    main()
