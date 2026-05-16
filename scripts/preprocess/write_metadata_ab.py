"""Write metadata.json for the subjective A/B test."""
import json
from pathlib import Path

OUT = Path("eval_output/subjective_ab")

CLIPS = [
    ("piano",     "Piano",         "neutral", "This is a piano cover of a glam metal music piece. The piece is being played gently on a keyboard with a grand piano sound. There is a calming, relaxing atmosphere in this piece."),
    ("metal",     "Heavy Metal",   "neutral", "This is the recording of a heavy metal music piece. There is a male vocalist singing melodically in the lead. The main tune is being played by the distorted electric guitar while the bass guitar is playing in the background. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is aggressive."),
    ("lofi",      "Lo-Fi Folk",    "low",     "The low quality recording features a live performance of a folk song that consists of an arpeggiated electric guitar melody played over groovy bass, punchy snare and shimmering cymbals. It sounds energetic and the recording is noisy and in mono."),
    ("edm",       "EDM",           "neutral", "This is an electronic dance music piece. There is a synth lead playing the main melody. The beat consists of a kick drum, clap, hi-hat and synthesized bass. The atmosphere is energetic and euphoric."),
    ("cinematic", "Cinematic",     "neutral", "This is a cinematic orchestral piece. There are strings playing a sweeping melody with brass accents. The piece builds in intensity with a dramatic crescendo. The atmosphere is epic and emotional."),
    ("acoustic",  "Acoustic",      "neutral", "A solo acoustic guitar piece with fingerpicking. Gentle and melancholic."),
    ("jazz",      "Jazz",          "high",    "A smooth jazz piece with a saxophone lead, upright bass, and brushed drums. Studio quality recording with warm tones."),
    ("ambient",   "Ambient",       "low",     "A dark ambient soundscape with drone pads, distant reverb, and subtle noise. Lo-fi texture with tape saturation."),
]
SEEDS = [42, 1337, 2026]
VARIANTS = ["p7v1", "p8v1"]

meta = {
    "experiment": "subjective_ab_p7v1_vs_p8v1",
    "variants": {
        "p7v1": {
            "label": "P7 V1 (Q=9)",
            "model": "phase7_v1_stage2_200000_ema_final.pth",
            "description": "Jamendo full, single-caption, Q-conditioned (infer with quality_level=9)",
            "color_side": "pink (left)",
        },
        "p8v1": {
            "label": "P8 V1 (NoQ)",
            "model": "phase8_stage2_200000_ema_final.pth",
            "description": "Jamendo full, single-caption, no Q conditioning (null token, quality_level=10 workaround)",
            "color_side": "lavender (right)",
        },
    },
    "inference_config": {
        "num_steps": 25,
        "cfg_strength": 3.5,
        "duration_sec": 9.975,
        "sample_rate": 16000,
        "encoder": "t5_clap",
        "text_c_dim": 512,
        "use_meanflow": True,
        "full_precision": True,
        "peak_normalized_dbfs": -1.0,
        "format": "WAV PCM_16",
    },
    "seeds": SEEDS,
    "rounds": {f"round_{i+1}": f"seed_{s}" for i, s in enumerate(SEEDS)},
    "clips": [],
}

for clip_name, genre, quality_cue, prompt in CLIPS:
    entry = {
        "clip_name": clip_name,
        "genre_display": genre,
        "prompt": prompt,
        "prompt_length_words": len(prompt.split()),
        "quality_cue": quality_cue,
        "files": {},
    }
    for seed in SEEDS:
        entry["files"][f"seed_{seed}"] = {
            v: f"{clip_name}_s{seed}_{v}.wav" for v in VARIANTS
        }
    meta["clips"].append(entry)

meta_path = OUT / "metadata.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"Wrote {meta_path}")
print(f"  {len(CLIPS)} clips x {len(SEEDS)} seeds x {len(VARIANTS)} variants = {len(CLIPS)*len(SEEDS)*len(VARIANTS)} files expected")
