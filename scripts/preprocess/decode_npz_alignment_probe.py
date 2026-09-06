#!/usr/bin/env python3
"""Decode cached audio latents and write two caption-reference TSVs.

The two input TSVs must describe the same cache-list positions.  Audio is
decoded once under synthetic IDs, allowing the existing evaluator to compare
the identical reconstructions against two competing caption mappings.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from meanaudio.model.utils.features_utils import FeaturesUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tsv-a", type=Path, required=True)
    parser.add_argument("--output-tsv-b", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Decode a deterministic random subset of the aligned input rows.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260717)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not {"id", "caption"}.issubset(rows[0]):
        raise SystemExit(f"Invalid reference TSV: {path}")
    return rows


def write_reference(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "caption"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    with args.cache.open() as handle:
        names = [line.strip() for line in handle if line.strip()]
    reference_a = read_tsv(args.reference_a)
    reference_b = read_tsv(args.reference_b)
    if not (len(names) == len(reference_a) == len(reference_b)):
        raise SystemExit(
            f"Length mismatch: cache={len(names)}, A={len(reference_a)}, "
            f"B={len(reference_b)}"
        )
    if args.sample_size is not None:
        if not 0 < args.sample_size <= len(names):
            raise SystemExit(
                f"--sample-size must be in [1, {len(names)}], got {args.sample_size}"
            )
        rng = random.Random(args.sample_seed)
        positions = sorted(rng.sample(range(len(names)), args.sample_size))
        names = [names[position] for position in positions]
        reference_a = [reference_a[position] for position in positions]
        reference_b = [reference_b[position] for position in positions]
        print(
            f"[Sampling] selected={len(names):,}, seed={args.sample_seed}, "
            f"input-position-range={positions[0]}..{positions[-1]}"
        )

    common_a = []
    common_b = []
    for position, (row_a, row_b) in enumerate(zip(reference_a, reference_b)):
        probe_id = f"probe_{position:04d}"
        common_a.append({"id": probe_id, "caption": row_a["caption"]})
        common_b.append({"id": probe_id, "caption": row_b["caption"]})
    write_reference(args.output_tsv_a, common_a)
    write_reference(args.output_tsv_b, common_b)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        position
        for position in range(len(names))
        if not (args.output_dir / f"probe_{position:04d}.flac").exists()
    ]
    print(f"[Input] rows={len(names)}, pending decodes={len(pending)}")
    if pending:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = FeaturesUtils(
            tod_vae_ckpt="weights/v1-16.pth",
            bigvgan_vocoder_ckpt="weights/best_netG.pt",
            enable_conditions=False,
            mode="16k",
            need_vae_encoder=False,
        ).to(device, torch.float32).eval()
        for start in tqdm(
            range(0, len(pending), args.batch_size), desc="decode NPZ means"
        ):
            positions = pending[start : start + args.batch_size]
            means = []
            for position in positions:
                with np.load(args.npz_dir / names[position]) as data:
                    means.append(data["mean"].astype(np.float32))
            latents = torch.from_numpy(np.stack(means)).to(device)
            mel = features.decode(latents)
            audio = features.vocode(mel).float().cpu().numpy()
            for batch_index, position in enumerate(positions):
                waveform = np.squeeze(audio[batch_index])
                if not np.isfinite(waveform).all():
                    raise SystemExit(f"Non-finite decoded audio at position {position}")
                sf.write(
                    args.output_dir / f"probe_{position:04d}.flac",
                    waveform,
                    16_000,
                )

    count = len(list(args.output_dir.glob("probe_*.flac")))
    if count != len(names):
        raise SystemExit(f"Decoded file count mismatch: {count} != {len(names)}")
    print(f"[Validation] decoded {count} common audio files")


if __name__ == "__main__":
    main()
