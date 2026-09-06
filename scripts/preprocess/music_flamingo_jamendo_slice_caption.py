#!/usr/bin/env python3
"""Run Music Flamingo on Jamendo slices aligned to MeanAudio training.

MeanAudio's local Jamendo latent extraction uses the first 10 seconds of each
30-second `wav_audio/{segment_id}.wav` file. This script mirrors that exact
audio window before captioning, so generated captions align with the training
audio latent rather than the full song or full 30-second segment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


MODEL_ID = "nvidia/music-flamingo-2601-hf"
DEFAULT_TSV = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
DEFAULT_WAV_ROOT = Path("/mnt/HDD/kojiek/phase4_jamendo_data/wav_audio")
PROMPT_VERSION = "slice10_v1"
SAMPLE_RATE = 16_000
SLICE_SECONDS = 10.0
NUM_SAMPLES = int(SAMPLE_RATE * SLICE_SECONDS)

PROMPT_CAPTION = (
    "Describe only this 10-second music audio slice. "
    "Focus on what is audible in this clip: likely genre, mood, instruments, "
    "vocals if present, rhythm/tempo feel, production texture, and energy. "
    "Do not infer full-song structure, lyrics, key, BPM, or events outside this clip."
)
PROMPT_SHORT_DIRECT = (
    "Describe only this 10-second music audio slice as one compact training caption "
    "of 35-50 words. Put concrete acoustic nouns first: genre/style, audible "
    "instruments or sounds, vocals if present, rhythm or energy, production texture, "
    "and mood. Do not mention the caption length, outside-song structure, lyrics, "
    "key, BPM, or anything not audible in this clip."
)

PROMPT_SHORT_DIRECT_V2 = (
    "Describe only this 10-second music audio slice in AT MOST 45 words, as one "
    "compact training caption. Hard limit: 45 words. Put concrete acoustic nouns "
    "first: genre/style, audible instruments or sounds, vocals if present, rhythm "
    "or energy, production texture, and mood. Name what is specific to THIS clip "
    "rather than generic praise. Do not open with \"This music is an instrumental\". "
    "Do not mention the caption length, outside-song structure, lyrics, key, BPM, "
    "or anything not audible in this clip."
)

PROMPT_PRESETS = {
    "slice10_v1": PROMPT_CAPTION,
    "short_direct_v1": PROMPT_SHORT_DIRECT,
    "short_direct_v2": PROMPT_SHORT_DIRECT_V2,
}

# flan-t5-large is the text encoder MeanAudio actually trains against, and
# features_utils.py truncates hard at 77 tokens. The short_direct_v1 corpus was
# written without ever measuring this: 79% of it was truncated and the visible
# (post-truncation) unique rate fell to 64%. Enforcement below closes that.
T5_MODEL = "google/flan-t5-large"
T5_WINDOW = 77
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_cuda_error(e: Exception) -> bool:
    msg = str(e)
    return any(k in msg for k in ("CUDA error", "device-side assert", "cudaErrorAssert", "CUDA out of memory"))


def track_id_from_segment_id(segment_id: str) -> str:
    if "_segment_" not in segment_id:
        return segment_id
    return segment_id.split("_segment_", 1)[0]


def collect_segments(tsv_path: Path, wav_root: Path, n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    with tsv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segment_id = row["id"]
            rows.append(
                {
                    "id": segment_id,
                    "track_id": track_id_from_segment_id(segment_id),
                    "source_audio_path": str(wav_root / f"{segment_id}.wav"),
                    "lpmc_caption": row.get("caption", ""),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_done_captions(out_dir: Path) -> set:
    """Rebuild the global uniqueness set when resuming an enforced run."""
    caps: set[str] = set()
    path = out_dir / "caption.jsonl"
    if not path.exists():
        return caps
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("ok") and rec.get("raw_text"):
                caps.add(rec["raw_text"])
    return caps


def load_done_ids(out_dir: Path) -> set[str]:
    done = set()
    p = out_dir / "caption.jsonl"
    if p.exists():
        with p.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("ok"):
                    done.add(str(rec.get("id")))

    skip_p = out_dir / ".cuda_skip_segment_ids"
    if skip_p.exists():
        with skip_p.open() as f:
            for line in f:
                sid = line.strip()
                if sid:
                    done.add(sid)
    return done


def write_slice_wav(source_audio_path: str, temp_dir: Path) -> Path:
    source = Path(source_audio_path)
    if not source.exists():
        raise FileNotFoundError(source)

    audio, sr = sf.read(source, frames=NUM_SAMPLES, always_2d=True, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}: {source}")

    audio = audio.mean(axis=1)
    if audio.shape[0] < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - audio.shape[0]), mode="constant")

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
    tmp = tempfile.NamedTemporaryFile(prefix=f"{safe_stem}_slice10_", suffix=".wav", dir=temp_dir, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return tmp_path


def run_caption(model, processor, audio_path: str, prompt: str, max_new_tokens: int,
                do_sample: bool = False, temperature: float = 1.0) -> dict:
    t0 = time.time()
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            out = model.generate(**inputs, **gen_kwargs)

        raw_text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

        return {
            "raw_text": raw_text,
            "output": {"text": raw_text},
            "runtime_sec": round(time.time() - t0, 3),
            "ok": True,
            "error": None,
        }
    except Exception as e:
        if is_cuda_error(e):
            raise
        return {
            "raw_text": None,
            "output": {},
            "runtime_sec": round(time.time() - t0, 3),
            "ok": False,
            "error": str(e),
        }


def run_caption_batch(model, processor, audio_paths: list[str], prompt: str,
                      max_new_tokens: int, do_sample: bool = False,
                      temperature: float = 1.0) -> dict:
    """Caption a batch of slices in one generate() call.

    MusicFlamingoProcessor.apply_chat_template accepts a list of conversations
    and its defaults already set text padding on with padding_side="left",
    which is what decoder-only batched generation needs: every row's generated
    tokens then start at the same offset, so the prompt can be sliced off with
    a single input_ids.shape[1].
    """
    t0 = time.time()
    try:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": path},
                    ],
                }
            ]
            for path in audio_paths
        ]

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            out = model.generate(**inputs, **gen_kwargs)

        texts = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        elapsed = time.time() - t0
        return {
            "ok": True,
            "texts": [t.strip() for t in texts],
            "runtime_sec": round(elapsed, 3),
            "per_clip_sec": round(elapsed / max(len(audio_paths), 1), 3),
            "error": None,
        }
    except Exception as e:
        if is_cuda_error(e):
            raise
        return {
            "ok": False,
            "texts": [],
            "runtime_sec": round(time.time() - t0, 3),
            "per_clip_sec": 0.0,
            "error": str(e),
        }


def enforced_caption_batch(model, processor, items: list[dict], prompt: str,
                           max_new_tokens: int, tok, seen: set, window: int,
                           max_attempts: int, sentence_trim: bool) -> list[dict]:
    """Batched counterpart of enforced_caption.

    Each round regenerates only the clips that have not been accepted yet, so
    the batch shrinks as clips pass. Acceptance is sequential within a round:
    two clips in the same batch can sample the same text, and the second one
    must lose so the corpus keeps its global uniqueness guarantee.
    """
    n = len(items)
    state = [
        {"log": [], "spent": 0.0, "best": None, "text": None, "done": False}
        for _ in range(n)
    ]
    active = list(range(n))

    for attempt in range(max_attempts):
        if not active:
            break
        paths = [items[i]["tmp_path"] for i in active]
        batch = run_caption_batch(
            model, processor, paths, prompt, max_new_tokens,
            do_sample=attempt > 0, temperature=0.7 + 0.2 * attempt,
        )
        if not batch["ok"]:
            for i in active:
                state[i]["error"] = batch["error"]
            break

        still_active = []
        for i, text in zip(active, batch["texts"]):
            st = state[i]
            st["spent"] += batch["per_clip_sec"]
            n_tok = len(tok(text, add_special_tokens=True)["input_ids"])
            too_long = n_tok > window
            duplicate = text in seen
            incomplete = not text.rstrip().endswith((".", "!", "?"))
            st["log"].append({"attempt": attempt, "tokens": n_tok,
                              "too_long": too_long, "duplicate": duplicate,
                              "incomplete": incomplete})
            if st["best"] is None:
                st["best"] = text
            if not too_long and not duplicate and not incomplete:
                st["text"] = text
                st["done"] = True
                seen.add(text)
            else:
                still_active.append(i)
        active = still_active

    results = []
    for i, st in enumerate(state):
        if st["done"]:
            text = st["text"]
            reasons = []
        else:
            text = st["best"] if st["best"] is not None else ""
            if sentence_trim and text:
                text = fit_to_window(text, tok, window)
                if not text.rstrip().endswith((".", "!", "?")):
                    parts = SENTENCE_SPLIT.split(text.strip())
                    if len(parts) > 1:
                        text = " ".join(parts[:-1])
            reasons = []
            if not text:
                reasons.append("generation_failed")
            else:
                if len(tok(text, add_special_tokens=True)["input_ids"]) > window:
                    reasons.append("over_window")
                if text in seen:
                    reasons.append("duplicate")
                if not text.rstrip().endswith((".", "!", "?")):
                    reasons.append("incomplete")
            if text and not reasons:
                seen.add(text)
        n_tok = len(tok(text, add_special_tokens=True)["input_ids"]) if text else 0
        results.append({
            "raw_text": text or None,
            "output": {"text": text} if text else {},
            "runtime_sec": round(st["spent"], 3),
            "ok": bool(text),
            "error": st.get("error"),
            "tokens": n_tok,
            "attempts": len(st["log"]),
            "attempt_log": st["log"],
            "enforced_ok": st["done"] or not reasons,
            "reject_reason": ",".join(reasons) or None,
        })
    return results


def fit_to_window(text: str, tok, window: int) -> str:
    """Drop trailing sentences until the caption fits the T5 window.

    Preferred over a raw token cut: a mid-sentence truncation is what produced
    651 identical visible strings in the v1 corpus.
    """
    if len(tok(text, add_special_tokens=True)["input_ids"]) <= window:
        return text
    sentences = SENTENCE_SPLIT.split(text.strip())
    while len(sentences) > 1:
        sentences.pop()
        candidate = " ".join(sentences)
        if len(tok(candidate, add_special_tokens=True)["input_ids"]) <= window:
            return candidate
    return text


def enforced_caption(model, processor, audio_path, prompt, max_new_tokens, tok,
                     seen: set, window: int, max_attempts: int,
                     sentence_trim: bool) -> dict:
    """Generate a caption that fits the T5 window and is not already in `seen`.

    Attempt 0 is greedy so an unconstrained corpus stays reproducible; later
    attempts must sample, because greedy decoding is deterministic per clip and
    would return the identical colliding caption forever.
    """
    attempts = []
    spent = 0.0
    best = None
    for attempt in range(max_attempts):
        do_sample = attempt > 0
        temperature = 0.7 + 0.2 * attempt
        result = run_caption(model, processor, audio_path, prompt, max_new_tokens,
                             do_sample=do_sample, temperature=temperature)
        if not result["ok"]:
            result["attempts"] = attempt + 1
            result["enforced_ok"] = False
            result["reject_reason"] = "generation_failed"
            return result

        text = result["raw_text"]
        n_tok = len(tok(text, add_special_tokens=True)["input_ids"])
        too_long = n_tok > window
        duplicate = text in seen
        # A low --max-new-tokens is the cheapest way to keep captions inside the
        # window, but it stops generation mid-sentence ~15% of the time. An
        # unfinished clause is a worse training target than a shorter one, so
        # reject it and let the fallback sentence-trim handle the last resort.
        incomplete = not text.rstrip().endswith((".", "!", "?"))
        attempts.append({"attempt": attempt, "tokens": n_tok,
                         "too_long": too_long, "duplicate": duplicate,
                         "incomplete": incomplete,
                         "runtime_sec": result["runtime_sec"]})
        spent += result["runtime_sec"]
        if best is None:
            best = (text, n_tok)

        if not too_long and not duplicate and not incomplete:
            result.update(raw_text=text, output={"text": text}, tokens=n_tok,
                          attempts=attempt + 1, attempt_log=attempts,
                          runtime_sec=round(spent, 3),
                          enforced_ok=True, reject_reason=None)
            return result

    # Exhausted: optionally make it fit, then report honestly whether the
    # duplicate constraint still fails so the corpus audit can see it.
    text = best[0]
    if sentence_trim:
        text = fit_to_window(text, tok, window)
        # drop a trailing clause that never terminated
        if not text.rstrip().endswith((".", "!", "?")):
            parts = SENTENCE_SPLIT.split(text.strip())
            if len(parts) > 1:
                text = " ".join(parts[:-1])
    n_tok = len(tok(text, add_special_tokens=True)["input_ids"])
    reasons = []
    if n_tok > window:
        reasons.append("over_window")
    if text in seen:
        reasons.append("duplicate")
    if not text.rstrip().endswith((".", "!", "?")):
        reasons.append("incomplete")
    return {
        "raw_text": text,
        "output": {"text": text},
        "runtime_sec": round(spent, 3),
        "ok": True,
        "error": None,
        "tokens": n_tok,
        "attempts": max_attempts,
        "attempt_log": attempts,
        "enforced_ok": not reasons,
        "reject_reason": ",".join(reasons) or None,
    }


def load_model():
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    gc.collect()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def summarize(out_file: Path) -> None:
    ok_count = 0
    with out_file.open() as f:
        for line in f:
            try:
                ok_count += bool(json.loads(line).get("ok"))
            except Exception:
                pass
    print(f"\nDone. ok={ok_count} out={out_file}")


def run_batched_loop(args, todo, temp_dir, out_file, mode, model, processor,
                     prompt, prompt_version, tok, seen) -> None:
    total = len(todo)
    written = 0
    with out_file.open(mode) as f_out:
        for start in range(0, total, args.batch_size):
            chunk = todo[start : start + args.batch_size]
            items = []
            for row in chunk:
                try:
                    tmp_path = write_slice_wav(row["source_audio_path"], temp_dir)
                except Exception as e:
                    items.append({"row": row, "tmp_path": None, "error": str(e)})
                    continue
                items.append({"row": row, "tmp_path": str(tmp_path), "error": None})

            usable = [it for it in items if it["tmp_path"]]
            results_by_id = {}
            if usable:
                try:
                    batch_results = enforced_caption_batch(
                        model, processor, usable, prompt, args.max_new_tokens,
                        tok, seen, args.window, args.max_attempts,
                        sentence_trim=not args.no_sentence_trim,
                    )
                except Exception as e:
                    if is_cuda_error(e):
                        for it in usable:
                            with (args.out_dir / ".cuda_skip_segment_ids").open("a") as sf_out:
                                sf_out.write(it["row"]["id"] + "\n")
                        print(f"[CUDA ERROR] batch at {start}; restart with --resume")
                        for it in items:
                            if it["tmp_path"]:
                                Path(it["tmp_path"]).unlink(missing_ok=True)
                        sys.exit(3)
                    raise
                for it, res in zip(usable, batch_results):
                    results_by_id[it["row"]["id"]] = res

            for it in items:
                if it["tmp_path"]:
                    Path(it["tmp_path"]).unlink(missing_ok=True)

            for it in items:
                row = it["row"]
                res = results_by_id.get(row["id"]) or {
                    "raw_text": None, "output": {}, "runtime_sec": 0.0,
                    "ok": False, "error": it["error"] or "slice failed",
                    "tokens": 0, "attempts": 0, "attempt_log": [],
                    "enforced_ok": False, "reject_reason": "slice_failed",
                }
                record = {
                    "id": row["id"],
                    "track_id": row["track_id"],
                    "source_audio_path": row["source_audio_path"],
                    "slice_start_sec": 0.0,
                    "slice_duration_sec": SLICE_SECONDS,
                    "slice_num_samples": NUM_SAMPLES,
                    "model": MODEL_ID,
                    "prompt_version": prompt_version,
                    "task": "caption",
                    **res,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            f_out.flush()
            done_ok = sum(1 for it in items if results_by_id.get(it["row"]["id"], {}).get("ok"))
            print(f"  [{written}/{total}] batch ok={done_ok}/{len(items)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("/home/kojiek/eval_output/music_flamingo_slice10_10k"))
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--wav_root", type=Path, default=DEFAULT_WAV_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="slice10_v1")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--enforce", action="store_true",
                        help="enforce the T5 window and global caption uniqueness")
    parser.add_argument("--window", type=int, default=T5_WINDOW)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="clips per generate() call; >1 requires --enforce")
    parser.add_argument("--no-sentence-trim", action="store_true",
                        help="with --enforce, do not sentence-trim a caption that "
                             "never fit within --max-attempts")
    args = parser.parse_args()

    prompt = args.prompt_text or PROMPT_PRESETS[args.prompt_preset]
    prompt_version = args.prompt_version or args.prompt_preset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.out_dir / "tmp_slices"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "caption.jsonl"

    print(f"[config] n={args.n} seed={args.seed} out={args.out_dir}")
    print(f"[config] tsv={args.tsv}")
    print(f"[config] wav_root={args.wav_root}")
    print(f"[config] slice=first {NUM_SAMPLES} samples ({SLICE_SECONDS:.1f}s @ {SAMPLE_RATE} Hz)")
    print(f"[config] prompt_version={prompt_version} max_new_tokens={args.max_new_tokens}")

    segments = collect_segments(args.tsv, args.wav_root, args.n, args.seed)
    done_ids = load_done_ids(args.out_dir) if args.resume else set()
    todo = [row for row in segments if row["id"] not in done_ids]
    print(f"[data] selected={len(segments)} done={len(done_ids)} todo={len(todo)}")

    tok = None
    seen: set[str] = set()
    if args.enforce:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(T5_MODEL)
        if args.resume:
            seen = load_done_captions(args.out_dir)
        print(f"[enforce] window={args.window} max_attempts={args.max_attempts} "
              f"sentence_trim={not args.no_sentence_trim} seen={len(seen)}")
    if args.batch_size > 1 and not args.enforce:
        sys.exit("[fatal] --batch-size > 1 requires --enforce")
    if args.batch_size > 1:
        print(f"[batch] batch_size={args.batch_size}")

    print(f"\nLoading {MODEL_ID} ...")
    model, processor = load_model()
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  -> loaded alloc={alloc_gb:.2f} GB")

    mode = "a" if args.resume else "w"

    if args.batch_size > 1:
        run_batched_loop(args, todo, temp_dir, out_file, mode, model, processor,
                         prompt, prompt_version, tok, seen)
        summarize(out_file)
        return

    with out_file.open(mode) as f_out:
        for i, row in enumerate(todo, 1):
            tmp_path: Path | None = None
            exit_after_record = False
            try:
                tmp_path = write_slice_wav(row["source_audio_path"], temp_dir)
                if args.enforce:
                    result = enforced_caption(
                        model, processor, str(tmp_path), prompt, args.max_new_tokens,
                        tok, seen, args.window, args.max_attempts,
                        sentence_trim=not args.no_sentence_trim,
                    )
                    if result["ok"] and result["raw_text"]:
                        seen.add(result["raw_text"])
                else:
                    result = run_caption(model, processor, str(tmp_path), prompt,
                                         args.max_new_tokens)
            except Exception as e:
                if is_cuda_error(e):
                    result = {
                        "raw_text": None,
                        "output": {},
                        "runtime_sec": 0.0,
                        "ok": False,
                        "error": str(e)[:300],
                    }
                    with (args.out_dir / ".cuda_skip_segment_ids").open("a") as sf_out:
                        sf_out.write(row["id"] + "\n")
                    exit_after_record = True
                else:
                    result = {
                        "raw_text": None,
                        "output": {},
                        "runtime_sec": 0.0,
                        "ok": False,
                        "error": str(e),
                    }
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

            record = {
                "id": row["id"],
                "track_id": row["track_id"],
                "source_audio_path": row["source_audio_path"],
                "slice_start_sec": 0.0,
                "slice_duration_sec": SLICE_SECONDS,
                "slice_num_samples": NUM_SAMPLES,
                "model": MODEL_ID,
                "prompt_version": prompt_version,
                "task": "caption",
                **result,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            status = "ok" if result["ok"] else "fail"
            extra = ""
            if args.enforce and result["ok"]:
                extra = (f" tok={result.get('tokens')} att={result.get('attempts')}"
                         f"{'' if result.get('enforced_ok') else ' REJECT=' + str(result.get('reject_reason'))}")
            print(f"  [{i}/{len(todo)}] {status} {row['id']} {result['runtime_sec']:.1f}s{extra}")

            if exit_after_record:
                print(f"[CUDA ERROR] {row['id']} -> added to .cuda_skip_segment_ids, restart with --resume")
                sys.exit(3)

    summarize(out_file)


if __name__ == "__main__":
    main()
