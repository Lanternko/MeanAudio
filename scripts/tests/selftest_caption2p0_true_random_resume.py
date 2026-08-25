#!/usr/bin/env python3
"""No-GPU proof that true-random caption choice survives exact resume."""

from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from meanaudio.data.extracted_audio import ExtractedAudio
from meanaudio.utils.training_resume import resume_coordinates


SEED = 14159265


def expected(epoch: int, audio_id: str) -> int:
    payload = f"k3-true-random-v1\0{SEED}\0{epoch}\0{audio_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3


def build_fixture(root: Path, count: int = 11) -> ExtractedAudio:
    audio, text = root / "audio", root / "text"
    audio.mkdir(); text.mkdir()
    tsv = root / "train.tsv"
    cache = root / "cache.txt"
    with tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption", "q_level"], delimiter="\t")
        writer.writeheader()
        for index in range(count):
            audio_id = f"clip-{index:03d}"
            writer.writerow({"id": audio_id, "caption": "index-only", "q_level": 9})
            np.savez(audio / f"{index}.npz", clip_id=np.asarray(audio_id),
                     mean=np.zeros((250, 32), np.float32), std=np.ones((250, 32), np.float32))
            np.savez(text / f"{index}.npz", clip_id=np.asarray(audio_id),
                     text_features=np.stack([np.full((77, 1024), x, np.float32) for x in range(3)]),
                     text_features_c=np.stack([np.full((512,), x, np.float32) for x in range(3)]),
                     text_attention_mask=np.ones((3, 77), np.int64))
    cache.write_text("".join(f"{index}.npz\n" for index in range(count)))
    return ExtractedAudio(
        tsv, concat_text_fc=False, npz_dir=audio, text_npz_dir=text,
        data_dim={"latent_seq_len": 250, "text_seq_len": 77, "text_dim": 1024, "text_c_dim": 512},
        repa_npz_dir=None, exclude_cls=False, repa_version=1, gt_cache=cache,
        require_text_overlay=True, multi_cap=True, use_text_attention_mask=True,
    )


def choices(dataset: ExtractedAudio, epochs: int) -> list[int]:
    result = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        for index, audio_id in enumerate(dataset.ids):
            got = int(dataset[index]["text_features"][0, 0].item())
            assert got == expected(epoch, audio_id)
            result.append(got)
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = build_fixture(Path(directory))
        continuous = choices(dataset, 5)
        batches_per_epoch = len(dataset)
        pause_at = 27
        epoch, offset = resume_coordinates(pause_at, batches_per_epoch)
        dataset.set_epoch(epoch)
        resumed = continuous[:pause_at]
        resumed.extend(int(dataset[index]["text_features"][0, 0].item()) for index in range(offset, len(dataset)))
        for next_epoch in range(epoch + 1, 5):
            dataset.set_epoch(next_epoch)
            resumed.extend(int(dataset[index]["text_features"][0, 0].item()) for index in range(len(dataset)))
        assert resumed == continuous
        assert len(set(continuous)) == 3
    print("[SELFTEST OK] true-random epoch+ID caption choices resume exactly")


if __name__ == "__main__":
    main()
