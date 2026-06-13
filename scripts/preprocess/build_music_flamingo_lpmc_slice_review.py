#!/usr/bin/env python3
"""Build a human-review sheet for slice-level Music Flamingo vs LP-MC captions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_FLAMINGO = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k/caption.jsonl")
DEFAULT_LPMC = Path("/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv")
DEFAULT_OUT_DIR = Path("/home/kojiek/eval_output/music_flamingo_slice10_10k/lpmc_review")
SAMPLE_RATE = 16_000
NUM_SAMPLES = SAMPLE_RATE * 10


def track_id_from_segment_id(segment_id: str) -> str:
    if "_segment_" not in segment_id:
        return segment_id
    return segment_id.split("_segment_", 1)[0]


def clean_cell(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").split())


def load_flamingo(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            caption = ((rec.get("output") or {}).get("text") or rec.get("raw_text") or "").strip()
            if not caption:
                continue
            rows[rec["id"]] = {
                "id": rec["id"],
                "track_id": rec.get("track_id") or track_id_from_segment_id(rec["id"]),
                "source_audio_path": rec.get("source_audio_path", ""),
                "slice_start_sec": rec.get("slice_start_sec", 0.0),
                "slice_duration_sec": rec.get("slice_duration_sec", 10.0),
                "caption": caption,
            }
    return rows


def load_lpmc(path: Path, wanted: set[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segment_id = row.get("id", "")
            if segment_id in wanted:
                rows[segment_id] = (row.get("caption") or "").strip()
    return rows


def sample_one_slice_per_track(common_ids: list[str], n: int, seed: int) -> list[str]:
    by_track: dict[str, list[str]] = defaultdict(list)
    for segment_id in common_ids:
        by_track[track_id_from_segment_id(segment_id)].append(segment_id)

    rng = random.Random(seed)
    track_ids = sorted(by_track)
    sampled_tracks = rng.sample(track_ids, min(n, len(track_ids)))

    sampled_segments = []
    for track_id in sampled_tracks:
        choices = sorted(by_track[track_id])
        sampled_segments.append(rng.choice(choices))
    return sampled_segments


def export_audio_slices(audio_dir: Path, sampled: list[str], flamingo: dict[str, dict]) -> dict[str, Path]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for idx, segment_id in enumerate(sampled, 1):
        src = Path(flamingo[segment_id]["source_audio_path"])
        audio, sr = sf.read(src, frames=NUM_SAMPLES, always_2d=True, dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}: {src}")
        audio = audio.mean(axis=1)
        if audio.shape[0] < NUM_SAMPLES:
            audio = np.pad(audio, (0, NUM_SAMPLES - audio.shape[0]), mode="constant")
        out = audio_dir / f"{idx:02d}_{segment_id}_slice10.wav"
        sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
        exported[segment_id] = out
    return exported


def write_tsv(
    path: Path,
    sampled: list[str],
    flamingo: dict[str, dict],
    lpmc: dict[str, str],
    exported_audio: dict[str, Path],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "track_id",
                "source_audio_path",
                "review_audio_path",
                "slice_start_sec",
                "slice_duration_sec",
                "music_flamingo_slice10_caption",
                "lpmc_caption",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for segment_id in sampled:
            row = flamingo[segment_id]
            writer.writerow(
                {
                    "id": segment_id,
                    "track_id": row["track_id"],
                    "source_audio_path": row["source_audio_path"],
                    "review_audio_path": str(exported_audio.get(segment_id, "")),
                    "slice_start_sec": row["slice_start_sec"],
                    "slice_duration_sec": row["slice_duration_sec"],
                    "music_flamingo_slice10_caption": clean_cell(row["caption"]),
                    "lpmc_caption": clean_cell(lpmc[segment_id]),
                }
            )


def write_markdown(
    path: Path,
    sampled: list[str],
    flamingo: dict[str, dict],
    lpmc: dict[str, str],
    seed: int,
    exported_audio: dict[str, Path],
) -> None:
    lines = [
        "# Music Flamingo Slice-10 vs LP-MC Jamendo Review",
        "",
        f"- sample_size: {len(sampled)}",
        f"- seed: {seed}",
        f"- source_flamingo_rows: {len(flamingo)}",
        "- granularity: first 10 seconds of each 30-second Jamendo segment",
        "",
    ]
    for idx, segment_id in enumerate(sampled, 1):
        row = flamingo[segment_id]
        lines.extend(
            [
                f"## {idx}. {segment_id}",
                "",
                f"- track_id: `{row['track_id']}`",
                f"- source_audio_path: `{row['source_audio_path']}`",
                f"- review_audio_path: `{exported_audio.get(segment_id, '')}`",
                f"- slice: {row['slice_start_sec']}s to {float(row['slice_start_sec']) + float(row['slice_duration_sec']):.1f}s",
                "",
                "### Music Flamingo Slice-10 Caption",
                "",
                row["caption"].strip(),
                "",
                "### LP-MC Caption",
                "",
                lpmc[segment_id].strip(),
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flamingo-jsonl", type=Path, default=DEFAULT_FLAMINGO)
    parser.add_argument("--lpmc-tsv", type=Path, default=DEFAULT_LPMC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--export-audio", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flamingo = load_flamingo(args.flamingo_jsonl)
    lpmc = load_lpmc(args.lpmc_tsv, set(flamingo))
    common = sorted(set(flamingo) & set(lpmc))
    if not common:
        raise SystemExit("No overlapping segment ids between Music Flamingo and LP-MC")

    sampled = sample_one_slice_per_track(common, args.n, args.seed)
    stem = f"sample{len(sampled)}_tracks_seed{args.seed}_slice10"
    md_path = args.out_dir / f"{stem}_flamingo_lpmc.md"
    tsv_path = args.out_dir / f"{stem}_flamingo_lpmc.tsv"
    ids_path = args.out_dir / f"{stem}_ids.txt"
    audio_dir = args.out_dir / f"{stem}_audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    exported_audio = export_audio_slices(audio_dir, sampled, flamingo) if args.export_audio else {}

    write_markdown(md_path, sampled, flamingo, lpmc, args.seed, exported_audio)
    write_tsv(tsv_path, sampled, flamingo, lpmc, exported_audio)
    ids_path.write_text("\n".join(sampled) + "\n")

    print(f"flamingo_ok={len(flamingo)}")
    print(f"lpmc_overlap={len(common)}")
    print(f"sampled={len(sampled)}")
    print(f"markdown={md_path}")
    print(f"tsv={tsv_path}")
    print(f"ids={ids_path}")
    if args.export_audio:
        print(f"audio_dir={audio_dir}")


if __name__ == "__main__":
    main()
