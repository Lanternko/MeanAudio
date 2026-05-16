"""CLAP score for subjective_ab_v3 (60 wav, n=30 prompts x 2 variants).

For each (prompt, audio) pair compute CLAP cosine similarity. Group by variant
and report mean/std + per-file breakdown.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
import laion_clap
from tqdm import tqdm

OUT_ROOT = Path("eval_output/subjective_ab_v4")
AUDIO = OUT_ROOT / "audio"
TSV = OUT_ROOT / "sampled_prompts.tsv"
CLAP_CKPT = "/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
RESULT = OUT_ROOT / "clap_scores.json"


def main():
    # clip_name -> prompt
    prompts = {}
    with open(TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            prompts[row["clip_name"]] = row["prompt"]
    print(f"[clap] loaded {len(prompts)} prompts")

    wavs = sorted(AUDIO.glob("*.wav"))
    print(f"[clap] {len(wavs)} wav files")

    print("[clap] loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(CLAP_CKPT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    scores_by_file = {}
    for w in tqdm(wavs, desc="CLAP"):
        # parse clip_name from filename: mc17_p7v1.wav -> mc17 + p7v1
        stem = w.stem
        clip_name, variant = stem.rsplit("_", 1)
        prompt = prompts.get(clip_name)
        if prompt is None:
            print(f"[warn] no prompt for {stem}")
            continue
        try:
            audio_e = model.get_audio_embedding_from_filelist([str(w)], use_tensor=True)
            text_e = model.get_text_embedding([prompt], use_tensor=True)
            sim = torch.nn.functional.cosine_similarity(audio_e, text_e, dim=-1)
            scores_by_file[w.name] = {
                "clip_name": clip_name,
                "variant": variant,
                "clap_sim": float(sim.item()),
            }
        except Exception as e:
            print(f"[warn] {w.name}: {e}")

    # aggregate
    groups = {"p7v1": [], "p8v1": []}
    for fname, d in scores_by_file.items():
        groups[d["variant"]].append(d["clap_sim"])

    summary = {
        "per_variant": {},
        "per_file": scores_by_file,
    }
    for v, vals in groups.items():
        summary["per_variant"][v] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # paired comparison (p7v1 vs p8v1 on same clip)
    pair_diffs = []
    paired_by_clip = {}
    for fname, d in scores_by_file.items():
        paired_by_clip.setdefault(d["clip_name"], {})[d["variant"]] = d["clap_sim"]
    for clip, pair in paired_by_clip.items():
        if "p7v1" in pair and "p8v1" in pair:
            pair_diffs.append((clip, pair["p7v1"] - pair["p8v1"]))
    pair_diffs.sort(key=lambda x: x[1])
    summary["paired"] = {
        "n_pairs": len(pair_diffs),
        "mean_p7_minus_p8": float(np.mean([d for _, d in pair_diffs])),
        "p7_wins": sum(1 for _, d in pair_diffs if d > 0),
        "p8_wins": sum(1 for _, d in pair_diffs if d < 0),
        "biggest_p8_wins": [(c, round(d, 4)) for c, d in pair_diffs[:3]],
        "biggest_p7_wins": [(c, round(d, 4)) for c, d in pair_diffs[-3:]],
    }

    RESULT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[clap] wrote {RESULT}")

    print("\n=== Per-variant CLAP (n=30 each) ===")
    print(f"{'variant':8s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    for v in ["p7v1", "p8v1"]:
        s = summary["per_variant"][v]
        print(f"{v:8s} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f}")

    print(f"\n=== Paired (n={summary['paired']['n_pairs']} clips) ===")
    p = summary["paired"]
    print(f"  mean(P7 - P8) = {p['mean_p7_minus_p8']:+.4f}")
    print(f"  P7 wins:  {p['p7_wins']}/{p['n_pairs']}")
    print(f"  P8 wins:  {p['p8_wins']}/{p['n_pairs']}")
    print(f"  biggest P8 wins: {p['biggest_p8_wins']}")
    print(f"  biggest P7 wins: {p['biggest_p7_wins']}")


if __name__ == "__main__":
    main()
