"""
Music Flamingo — Jamendo eval pipeline (v1)
===========================================

Runs nvidia/music-flamingo-2601-hf on full Jamendo tracks and writes three JSONL:
  - caption.jsonl      : song-level free-text description
  - music_tags.jsonl   : structured genre/mood/instrument/vocal/tempo/energy
  - climax.jsonl       : pseudo-label climax segment with confidence + reason

Usage:
  # 100-track smoke eval (default)
  ~/venvs/music_flamingo/bin/python music_flamingo_jamendo_eval.py \
      --n 100 --out_dir ~/eval_output/music_flamingo_smoke

  # 1K eval
  ~/venvs/music_flamingo/bin/python music_flamingo_jamendo_eval.py \
      --n 1000 --out_dir ~/eval_output/music_flamingo_1k

  # resume a partial run
  ~/venvs/music_flamingo/bin/python music_flamingo_jamendo_eval.py \
      --n 100 --out_dir ~/eval_output/music_flamingo_smoke --resume

Input:
  TSV : /mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv  (11532 unique tracks)
  Audio: /mnt/HDD/MTG-Jamendo_Master_Dataset_Original_Quality/{prefix}/{trackid}.mp3

Output schema per line:
  {
    "track_id": "04_1318704",
    "audio_path": "/mnt/.../04/1318704.mp3",
    "duration_sec": 163.59,
    "model": "nvidia/music-flamingo-2601-hf",
    "prompt_version": "v1",
    "task": "caption" | "music_tags" | "climax",
    "output": { ... },
    "raw_text": "...",
    "runtime_sec": 4.73,
    "ok": true,
    "error": null
  }
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import time
from pathlib import Path

import torch


def _is_cuda_error(e: Exception) -> bool:
    msg = str(e)
    return any(k in msg for k in ("CUDA error", "device-side assert", "cudaErrorAssert", "CUDA out of memory"))

MODEL_ID  = "nvidia/music-flamingo-2601-hf"
AUDIO_ROOT = Path("/mnt/HDD/MTG-Jamendo_Master_Dataset_Original_Quality")
TSV_PATH   = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
PROMPT_VERSION = "v1"

# ── Prompts ──────────────────────────────────────────────────────────────────

PROMPT_CAPTION = (
    "Describe this music track in detail. "
    "Include the likely genre, overall mood and emotional tone, "
    "instruments you can hear, vocal characteristics (or note if instrumental), "
    "production style, and how the energy changes throughout the track."
)

PROMPT_TAGS = (
    "Analyse this music track and return a JSON object with these exact keys:\n"
    '  "primary_genre": string (e.g. "pop", "rock", "jazz", "electronic", "classical")\n'
    '  "secondary_genres": list of strings (may be empty)\n'
    '  "moods": list of strings (e.g. ["energetic", "uplifting"])\n'
    '  "instruments": list of strings (e.g. ["guitar", "drums", "piano"])\n'
    '  "vocal": one of "instrumental", "male", "female", "mixed", "unknown"\n'
    '  "tempo_feel": one of "slow", "medium", "fast"\n'
    '  "energy": one of "low", "medium", "high"\n'
    "Return only the JSON object, no other text."
)

PROMPT_CLIMAX = (
    "Identify the climax or emotional peak segment of this track. "
    "Return a JSON object with these exact keys:\n"
    '  "climax_start_sec": number (start time of the climax in seconds)\n'
    '  "climax_end_sec": number (end time of the climax in seconds)\n'
    '  "confidence": number between 0.0 and 1.0\n'
    '  "reason": string (1-2 sentences explaining why this segment is the climax)\n'
    "Return only the JSON object, no other text."
)

TASKS = {
    "caption":    PROMPT_CAPTION,
    "music_tags": PROMPT_TAGS,
    "climax":     PROMPT_CLIMAX,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_unique_tracks(tsv_path: Path, n: int, seed: int = 42) -> list[dict]:
    """Return n unique tracks from TSV, sampled with fixed seed."""
    seen = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seg_id = row["id"]
            parts = seg_id.split("_")
            prefix, trackid = parts[0], parts[1]
            key = f"{prefix}_{trackid}"
            if key not in seen:
                audio = AUDIO_ROOT / prefix / f"{trackid}.mp3"
                if audio.exists():
                    seen[key] = {"track_id": key, "audio_path": str(audio)}
    tracks = list(seen.values())
    rng = random.Random(seed)
    rng.shuffle(tracks)
    return tracks[:n]


def load_done_ids(out_dir: Path, task: str) -> set[str]:
    """Return set of track_ids to skip: ok=True records + permanently cuda-skipped."""
    done = set()
    p = out_dir / f"{task}.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        done.add(rec["track_id"])
                except Exception:
                    pass
    # also skip tracks that caused unrecoverable CUDA errors
    skip_p = out_dir / ".cuda_skip_ids"
    if skip_p.exists():
        with open(skip_p) as f:
            for line in f:
                tid = line.strip()
                if tid:
                    done.add(tid)
    return done


def try_parse_json(text: str) -> dict | None:
    """Extract first JSON object from model output.
    Handles both standard JSON (double quotes) and Python dict syntax (single quotes).
    """
    import ast
    text = text.strip()
    # try direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        blob = m.group()
        try:
            return json.loads(blob)
        except Exception:
            pass
        # fallback: Python dict syntax (model sometimes outputs single-quoted dicts)
        try:
            result = ast.literal_eval(blob)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    # last resort: ast on full text
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def get_audio_duration(audio_path: str) -> float | None:
    """Get duration via soundfile or librosa."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=False, duration=None)
        return y.shape[-1] / sr
    except Exception:
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def run_task(model, processor, audio_path: str, task: str, prompt: str) -> dict:
    """Run one task on one track. Returns result dict."""
    t0 = time.time()
    try:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": audio_path},
        ]}]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
            )

        raw_text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        dt = time.time() - t0

        # parse structured output for tags / climax
        parsed = None
        if task in ("music_tags", "climax"):
            parsed = try_parse_json(raw_text)

        if task == "caption":
            output = {"text": raw_text}
        elif parsed is not None:
            output = parsed
        else:
            output = {"raw": raw_text}   # fallback if JSON parsing failed

        return {
            "raw_text": raw_text,
            "output": output,
            "runtime_sec": round(dt, 3),
            "ok": True,
            "error": None,
        }

    except Exception as e:
        if _is_cuda_error(e):
            raise  # propagate to main loop for model reload
        return {
            "raw_text": None,
            "output": {},
            "runtime_sec": round(time.time() - t0, 3),
            "ok": False,
            "error": str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,  default=100)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--out_dir",   type=str,  default="~/eval_output/music_flamingo_smoke")
    parser.add_argument("--tsv",       type=str,  default=str(TSV_PATH))
    parser.add_argument("--tasks",     nargs="+", default=["caption", "music_tags", "climax"])
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] n={args.n}  seed={args.seed}  out={out_dir}")
    print(f"[config] tasks={args.tasks}  resume={args.resume}")

    # collect tracks
    print(f"\nCollecting {args.n} unique tracks from {args.tsv} ...")
    tracks = collect_unique_tracks(Path(args.tsv), args.n, args.seed)
    print(f"  → {len(tracks)} tracks ready")

    def load_model():
        from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
        gc.collect()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicFlamingoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        _model.eval()
        return _model, _processor

    # load model once
    print(f"\nLoading {MODEL_ID} ...")
    model, processor = load_model()
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  → loaded  alloc={alloc_gb:.2f} GB")

    # run tasks
    for task in args.tasks:
        prompt = TASKS[task]
        out_file = out_dir / f"{task}.jsonl"

        done_ids = load_done_ids(out_dir, task) if args.resume else set()
        todo = [t for t in tracks if t["track_id"] not in done_ids]

        print(f"\n── task={task}  todo={len(todo)} / {len(tracks)} ──")

        with open(out_file, "a" if args.resume else "w") as f_out:
            for i, track in enumerate(todo):
                duration = get_audio_duration(track["audio_path"])
                try:
                    result = run_task(model, processor, track["audio_path"], task, prompt)
                except Exception as e:
                    if _is_cuda_error(e):
                        # CUDA context permanently corrupted — write error, add to skip list, exit
                        # The outer shell wrapper will restart with --resume, which reads .cuda_skip_ids
                        result = {
                            "raw_text": None, "output": {},
                            "runtime_sec": 0.0, "ok": False, "error": str(e)[:200],
                        }
                        record = {
                            "track_id": track["track_id"], "audio_path": track["audio_path"],
                            "duration_sec": round(duration, 2) if duration else None,
                            "model": MODEL_ID, "prompt_version": PROMPT_VERSION, "task": task,
                            **result,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        skip_p = out_dir / ".cuda_skip_ids"
                        with open(skip_p, "a") as sf:
                            sf.write(track["track_id"] + "\n")
                        print(f"\n  [CUDA ERROR] {track['track_id']} → added to .cuda_skip_ids, restarting process")
                        import sys
                        sys.exit(3)
                    else:
                        raise

                record = {
                    "track_id":      track["track_id"],
                    "audio_path":    track["audio_path"],
                    "duration_sec":  round(duration, 2) if duration else None,
                    "model":         MODEL_ID,
                    "prompt_version": PROMPT_VERSION,
                    "task":          task,
                    **result,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                status = "✓" if result["ok"] else "✗"
                print(f"  [{i+1}/{len(todo)}] {status} {track['track_id']}  "
                      f"{result['runtime_sec']:.1f}s  "
                      f"{'JSON ok' if isinstance(result['output'], dict) and result['output'] and 'raw' not in result['output'] else 'raw'}")

        ok_count = sum(1 for _ in open(out_file) if json.loads(_).get("ok"))
        print(f"  → {out_file.name}: {ok_count} ok")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
