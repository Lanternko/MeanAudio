"""
Audiobox Aesthetics scoring for subjective_ab_v3 (48 wav, cfg=0.5).

Groups files by variant (p7v1 vs p8v1), reports per-file and per-variant
mean CE / CU / PC / PQ. Also flags clips at the ends of the distribution
so we can sanity-check mc11 / mc18 after the cfg=3.5 -> cfg=0.5 fix.
"""

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

OUT_ROOT = Path("eval_output/subjective_ab_v4")
AUDIO = OUT_ROOT / "audio"
RESULT = OUT_ROOT / "aes_scores.json"


def load_predictor(batch_size=16):
    # patch torchaudio.load -> soundfile (no ffmpeg in this env)
    import torchaudio, soundfile as sf, torch
    def _load_sf(path, **kw):
        d, sr = sf.read(str(path), always_2d=True)
        return torch.from_numpy(d.T).float(), sr
    torchaudio.load = _load_sf
    from audiobox_aesthetics.infer import AesPredictor
    return AesPredictor(checkpoint_pth=None, batch_size=batch_size)


def main():
    wavs = sorted(AUDIO.glob("*.wav"))
    assert len(wavs) in (48, 60), f"expected 48 or 60 files, got {len(wavs)}"
    print(f"[aes] {len(wavs)} wav files in {AUDIO}")

    predictor = load_predictor(batch_size=16)

    scores_by_file = {}
    axes = ["CE", "CU", "PC", "PQ"]

    # run in batches of 16 so we share model state
    B = 16
    for i in tqdm(range(0, len(wavs), B), desc="AES"):
        batch = [{"path": str(p)} for p in wavs[i:i + B]]
        results = predictor.forward(batch)
        for p, r in zip(wavs[i:i + B], results):
            scores_by_file[p.name] = {a: float(r[a]) for a in axes}

    # group by variant (filename suffix)
    groups = {"p7v1": {}, "p8v1": {}}
    for fname, s in scores_by_file.items():
        if fname.endswith("_p7v1.wav"):
            groups["p7v1"][fname] = s
        elif fname.endswith("_p8v1.wav"):
            groups["p8v1"][fname] = s

    summary = {"per_variant": {}, "per_file": scores_by_file}
    for v, files in groups.items():
        means = {a: float(np.mean([s[a] for s in files.values()])) for a in axes}
        stds = {a: float(np.std([s[a] for s in files.values()])) for a in axes}
        summary["per_variant"][v] = {
            "n": len(files),
            "mean": means,
            "std": stds,
        }

    # flag lowest / highest per axis per variant (PQ is the most relevant for saturation)
    outliers = {}
    for v, files in groups.items():
        outliers[v] = {}
        for a in axes:
            ranked = sorted(files.items(), key=lambda kv: kv[1][a])
            outliers[v][a] = {
                "lowest_3": [(k, round(v_[a], 3)) for k, v_ in ranked[:3]],
                "highest_3": [(k, round(v_[a], 3)) for k, v_ in ranked[-3:]],
            }
    summary["outliers"] = outliers

    RESULT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[aes] wrote {RESULT}")

    # console summary
    print("\n=== Per-variant mean (n=24 each) ===")
    print(f"{'variant':8s} {'CE':>8s} {'CU':>8s} {'PC':>8s} {'PQ':>8s}")
    for v in ["p7v1", "p8v1"]:
        m = summary["per_variant"][v]["mean"]
        s = summary["per_variant"][v]["std"]
        print(f"{v:8s} "
              f"{m['CE']:.3f}±{s['CE']:.2f}  "
              f"{m['CU']:.3f}±{s['CU']:.2f}  "
              f"{m['PC']:.3f}±{s['PC']:.2f}  "
              f"{m['PQ']:.3f}±{s['PQ']:.2f}")

    print("\n=== Lowest PQ per variant (saturation canary) ===")
    for v in ["p7v1", "p8v1"]:
        lo = outliers[v]["PQ"]["lowest_3"]
        print(f"  {v}: {lo}")

    print("\n=== mc11 / mc18 status (original york135 complaints) ===")
    for clip in ["mc11", "mc18"]:
        for v in ["p7v1", "p8v1"]:
            k = f"{clip}_{v}.wav"
            if k in scores_by_file:
                s = scores_by_file[k]
                print(f"  {k}: CE={s['CE']:.2f} CU={s['CU']:.2f} "
                      f"PC={s['PC']:.2f} PQ={s['PQ']:.2f}")


if __name__ == "__main__":
    main()
