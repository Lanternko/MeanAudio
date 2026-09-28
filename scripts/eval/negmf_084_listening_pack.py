#!/usr/bin/env python
"""084 NegMF five-prompt blind listening pack.

Conditions (training seed 14159265, MusicCaps-eval protocol: MeanFlow 25 steps,
fp32, NoMask, --no_q; generated through eval.py so the negative branch follows
the same code path as the standard CFG3+neg cell):

  ctrl_cfg0     nmv2pair control, CFG 0
  ctrl_cfg3neg  nmv2pair control, CFG 3 + fidelity8 negative prompt
  n100_cfg0     NegMF N100, CFG 0

Each prompt is generated with two inference seeds. Clips are written raw and
loudness-matched to -23 LUFS (the listening copy), as FLAC, under blinded letters
A/B/C that are shuffled per (prompt, seed) with a fixed RNG. ANSWER_KEY.md holds
the mapping plus per-clip LUFS / peak / CLAP / AES from eval_metrics.py.
"""
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = Path.home() / "MeanAudio"
PY = str(Path.home() / "venvs/dac/bin/python")
OUT = ROOT / "deliverables/negmf_084_listening_20260929"
WORK = Path.home() / "eval_output_nvme/negmf_084_listening_20260929"
NEG = "low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi"
EXPS = ROOT / "exps"
CTRL = "phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000"
N100 = "phase8_qwen_caption2p0_slot0clean_negmfn100_noq_quarter_s14159265_stage2_50000"
CONDS = {
    "ctrl_cfg0": (CTRL, 0.0, None),
    "ctrl_cfg3neg": (CTRL, 3.0, NEG),
    "n100_cfg0": (N100, 0.0, None),
}
SEEDS = [42, 43]
TARGET_LUFS = -23.0

# docs/eval/subjective_prompts.md
PROMPTS = [
    ("1_piano", "This is a piano cover of a glam metal music piece. The piece is being played gently on a keyboard with a grand piano sound. There is a calming, relaxing atmosphere in this piece."),
    ("2_metal", "This is the recording of a heavy metal music piece. There is a male vocalist singing melodically in the lead. The main tune is being played by the distorted electric guitar while the bass guitar is playing in the background. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is aggressive."),
    ("3_lofi_folk", "The low quality recording features a live performance of a folk song that consists of an arpeggiated electric guitar melody played over groovy bass, punchy snare and shimmering cymbals. It sounds energetic and the recording is noisy and in mono."),
    ("4_edm", "This is an electronic dance music piece. There is a synth lead playing the main melody. The beat consists of a kick drum, clap, hi-hat and synthesized bass. The atmosphere is energetic and euphoric."),
    ("5_cinematic", "This is a cinematic orchestral piece. There are strings playing a sweeping melody with brass accents. The piece builds in intensity with a dramatic crescendo. The atmosphere is epic and emotional."),
]


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"FAIL ({r.returncode}): {' '.join(cmd[:3])} ... see {log}")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    tsv = WORK / "prompts.tsv"
    with open(tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "caption"])
        w.writerows(PROMPTS)

    scores = {}
    for cond, (exp, cfg, neg) in CONDS.items():
        ckpt = EXPS / exp / f"{exp}_ema_final.pth"
        assert ckpt.is_file(), ckpt
        for seed in SEEDS:
            d = WORK / f"{cond}_s{seed}"
            audio = d / "audio"
            if not (d / "metrics" / "per_clip.tsv").is_file() and not list((d / "metrics").glob("*/per_clip.tsv")):
                shutil.rmtree(d, ignore_errors=True)
                audio.mkdir(parents=True)
                cmd = [PY, "eval.py", "--variant", "meanaudio_s", "--model_path", str(ckpt),
                       "--output", str(audio), "--tsv", str(tsv), "--use_meanflow",
                       "--num_steps", "25", "--cfg_strength", str(cfg),
                       "--no_text_attention_mask", "--encoder_name", "t5_clap",
                       "--text_c_dim", "512", "--seed", str(seed), "--full_precision", "--no_q"]
                if neg:
                    cmd += ["--negative_prompt", neg]
                run(cmd, d / "gen.log")
                run([PY, "scripts/eval/eval_metrics.py", "--gen_dir", str(audio), "--tsv", str(tsv),
                     "--exp_name", f"{cond}_s{seed}", "--out_dir", str(d / "metrics")], d / "metrics.log")
            per = next((d / "metrics").rglob("per_clip.tsv"))
            with open(per) as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    scores[(cond, seed, row["id"])] = row

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = random.Random(20260929)
    key = []
    for pid, caption in PROMPTS:
        for seed in SEEDS:
            conds = list(CONDS)
            rng.shuffle(conds)
            pdir = OUT / f"{pid}_seed{seed}"
            pdir.mkdir()
            for letter, cond in zip("ABC", conds):
                wav = WORK / f"{cond}_s{seed}" / "audio" / f"{pid}.flac"
                x, sr = sf.read(wav, always_2d=True)
                meter = pyln.Meter(sr)
                lufs = meter.integrated_loudness(x)
                y = x * 10 ** ((TARGET_LUFS - lufs) / 20) if np.isfinite(lufs) else x
                peak_raw = float(np.abs(x).max())
                peak_lm = float(np.abs(y).max())
                if peak_lm >= 1.0:
                    y = y / peak_lm * 0.999
                sf.write(pdir / f"{letter}.flac", y, sr)
                sf.write(pdir / f"{letter}_raw.flac", x, sr)
                s = scores.get((cond, seed, pid), {})
                key.append(dict(prompt=pid, seed=seed, letter=letter, cond=cond, lufs=lufs,
                                peak_raw=peak_raw, peak_lm=peak_lm, **{k: s.get(k) for k in s if k not in ("id", "lufs", "peak")}))

    json.dump(key, open(OUT / "answer_key.json", "w"), indent=1, default=float)
    cols = [c for c in ("clap", "PQ", "CE", "CU", "PC", "crest") if c in key[0]]
    lines = ["# Answer key (open after listening)", "",
             "`A/B/C.flac` = loudness-matched to -23 LUFS; `*_raw.flac` = as generated.", "",
             "| clip | letter | condition | raw LUFS | raw peak | " + " | ".join(cols) + " |",
             "|---|---|---|---:|---:|" + "---:|" * len(cols)]
    for k in key:
        vals = " | ".join(f"{float(k[c]):.3f}" if k.get(c) not in (None, "") else "—" for c in cols)
        lines.append(f"| {k['prompt']}_seed{k['seed']} | {k['letter']} | {k['cond']} | {k['lufs']:.1f} | {k['peak_raw']:.3f} | {vals} |")
    (OUT / "ANSWER_KEY.md").write_text("\n".join(lines) + "\n")
    (OUT / "README.md").write_text(README)
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    print("OK", OUT)


README = """# 084 NegMF 五首固定 prompt 盲聽包

每個資料夾是一首固定 prompt（`docs/eval/subjective_prompts.md`）× 一個推論 seed（42／43）。
資料夾內的 A／B／C 是三種條件，字母對應每個資料夾各自打亂；聽完再開 `ANSWER_KEY.md`。

三種條件（都是訓練 seed 14159265、MeanFlow 25 步、fp32、NoMask、NoQ，與 MusicCaps 標準 eval 同一條 eval.py 路徑）：

- control CFG0：nmv2pair control，不開 guidance
- control CFG3+neg：同一個 control，CFG 3＋fidelity8 負向 prompt（目前的標準第二格）
- N100 CFG0：NegMF N100（S2 訓練時把 CFG 目標的 ∅ 分支換成 fidelity8），不開 guidance

要回答的問題：N100 在 CFG0 下的 PQ 增益（+0.97 lvl30）與 FAD 變差（3.81→6.16）在耳朵上站哪一邊；
以及 N100 CFG0 和 control CFG3+neg 聽起來是否同一類聲音。

`A.flac` 等已對齊到 −23 LUFS，避免「大聲＝好聽」偏差；`*_raw.flac` 是原始輸出，響度差本身也是模型行為。
建議每個資料夾記：最好／最差、是否有明顯雜訊／失真／悶／過度壓縮、是否符合 prompt。
"""

if __name__ == "__main__":
    main()
