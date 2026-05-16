"""Write metadata.json for v3 (MusicCaps random sample)."""
import csv
import json
from pathlib import Path

OUT = Path("eval_output/subjective_ab_v3")
TSV = OUT / "sampled_prompts.tsv"
INFER_SEED = 42
VARIANTS = ["p7v1", "p8v1"]

clips = []
with open(TSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        clip_name = row["clip_name"]
        mc_id = row["mc_id"]
        prompt = row["prompt"]
        clips.append({
            "clip_name": clip_name,
            "musiccaps_id": mc_id,
            "prompt": prompt,
            "prompt_length_words": len(prompt.split()),
            "files": {v: f"{clip_name}_{v}.wav" for v in VARIANTS},
        })

meta = {
    "experiment": "subjective_ab_v3_musiccaps_p7v1_vs_p8v1",
    "prompt_source": {
        "dataset": "MusicCaps test set",
        "tsv_path": "/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv",
        "total_rows": 5521,
        "sampled_n": len(clips),
        "sample_seed": 42,
        "note": "Prompts deterministically sampled via Python random.Random(42).sample()",
    },
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
        "infer_seed": INFER_SEED,
        "peak_normalized_dbfs": -1.0,
        "format": "WAV PCM_16",
    },
    "design_note": "24 unique prompts x 2 variants x 1 seed = 48 files. MusicCaps captions provide prompt diversity so seed variation is dropped.",
    "clips": clips,
}

meta_path = OUT / "metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"Wrote {meta_path}")
print(f"  {len(clips)} clips x {len(VARIANTS)} variants = {len(clips)*len(VARIANTS)} files expected")
