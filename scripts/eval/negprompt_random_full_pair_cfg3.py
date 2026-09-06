#!/usr/bin/env python3
"""true-vs-fake random (013, full scale) under the CFG 3.0 negative-prompt protocol.

Secondary (non-canonical) protocol:

    MusicCaps 5521; MeanFlow 25; CFG 3.0; NoMask; seed 42; full precision;
    negative prompt = the fidelity string fixed by the 36-cell ablation.

CFG 3.0 is the optimum that ablation found (docs/experiments/... negprompt
ablation, 2026-08-31), and it found it *with* this negative prompt, so the two
travel together here. That means this table changes two things at once relative
to the canonical CFG 0 pair (0.2221 vs 0.2186): it is comparable to the other
negprompt_reeval arms and to itself, NOT cell-for-cell against CFG 0.

The canonical CFG 0 wrapper refuses any label that is not *_mf25_cfg0_*, so this
deliberately does not go through it and produces no contract report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import negprompt_reeval_full_arms as sweep


OUT = Path(
    "/home/kojiek/nvme_experiment_artifacts/meanaudio/"
    "negprompt_random_full_cfg3"
)
CFG_STRENGTH = "3.0"
ARMS = [
    (
        "c2p0_013_true_random_full_noq",
        "phase8_qwen_caption2p0_k3_true_random_noq_full_stage2_200000",
        ["--no_q"],
    ),
    (
        "c2p0_013_fake_random_full_noq",
        "phase8_qwen_caption2p0_k3_fake_random_noq_full_stage2_200000",
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
    sweep.CFG_STRENGTH = CFG_STRENGTH
    sweep.PROTOCOL = (
        f"MusicCaps 5521; MeanFlow 25; CFG {CFG_STRENGTH}; NoMask; seed 42; "
        f'full precision; negative_prompt="{sweep.NEGATIVE_PROMPT}"'
    )
    sweep.main()


if __name__ == "__main__":
    sys.exit(main())
