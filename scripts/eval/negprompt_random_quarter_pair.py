#!/usr/bin/env python3
"""Run the preregistered CFG-1.5 negative-prompt comparison for 013 random quarters."""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

import negprompt_reeval_full_arms as sweep


OUT = Path(
    "/home/kojiek/nvme_experiment_artifacts/meanaudio/"
    "negprompt_random_quarter_cfg1p5"
)
ARMS = [
    (
        "c2p0_013_true_random_quarter_noq",
        "phase8_qwen_caption2p0_k3_true_random_noq_quarter_stage2_50000",
        ["--no_q"],
    ),
    (
        "c2p0_013_fake_random_quarter_noq",
        "phase8_qwen_caption2p0_k3_fake_random_noq_quarter_stage2_50000",
        ["--no_q"],
    ),
]


def remove_incomplete_audio() -> None:
    """Remove only unreadable transient FLACs left by an interrupted generation."""
    for label, _, _ in ARMS:
        audio_dir = OUT / "_audio" / label
        if not audio_dir.is_dir():
            continue
        for path in audio_dir.glob("*.flac"):
            try:
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise RuntimeError("empty audio")
            except Exception:
                path.unlink()


def main() -> None:
    remove_incomplete_audio()
    sweep.OUT = OUT
    sweep.AUDIO_ROOT = OUT / "_audio"
    sweep.ARMS = ARMS
    sweep.main()


if __name__ == "__main__":
    sys.exit(main())
