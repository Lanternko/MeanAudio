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

PROMPT_PRESETS = {
    "slice10_v1": PROMPT_CAPTION,
    "short_direct_v1": PROMPT_SHORT_DIRECT,
}


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


def run_caption(model, processor, audio_path: str, prompt: str, max_new_tokens: int) -> dict:
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
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

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

    print(f"\nLoading {MODEL_ID} ...")
    model, processor = load_model()
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  -> loaded alloc={alloc_gb:.2f} GB")

    mode = "a" if args.resume else "w"
    with out_file.open(mode) as f_out:
        for i, row in enumerate(todo, 1):
            tmp_path: Path | None = None
            exit_after_record = False
            try:
                tmp_path = write_slice_wav(row["source_audio_path"], temp_dir)
                result = run_caption(model, processor, str(tmp_path), prompt, args.max_new_tokens)
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
            print(f"  [{i}/{len(todo)}] {status} {row['id']} {result['runtime_sec']:.1f}s")

            if exit_after_record:
                print(f"[CUDA ERROR] {row['id']} -> added to .cuda_skip_segment_ids, restart with --resume")
                sys.exit(3)

    ok_count = 0
    with out_file.open() as f:
        for line in f:
            try:
                ok_count += bool(json.loads(line).get("ok"))
            except Exception:
                pass
    print(f"\nDone. ok={ok_count} out={out_file}")


if __name__ == "__main__":
    main()
