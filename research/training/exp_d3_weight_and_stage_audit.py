"""
EXP-D3: Weight-norm audit + S1 vs S2 activation comparison.

Two questions from EXP-D2:
  (Q1) M1 vs M2 — is the activation collapse driven by learned weight shrinkage,
       or by LayerNorm/bias cancellation downstream of normal weights?
  (Q2) Stage attribution — does the collapse already exist after Stage 1 (FluxAudio),
       or only appear after Stage 2 (MeanFlow distillation)?

For each of 4 conditions (P8, EXP-A, EXP-B, EXP-C), load BOTH the S1 (FluxAudio)
and S2 (MeanAudio) ema_final.pth and:

  Part A — weight norm audit
    For each Linear / Conv weight tensor under `text_cond_proj` and `text_input_proj`,
    report ‖W‖_2, ‖W‖_F / sqrt(numel), and ‖bias‖. If collapsed and healthy have
    similar weight norms but very different activation magnitudes → M2 (downstream cancellation).
    If collapsed weights are much smaller → M1 (learned shrinkage).

  Part B — activation magnitude on fixed prompt batch
    Same 8 prompts encoded via T5+CLAP. Run net.text_cond_proj(...) and
    net.text_input_proj(...). Report output ‖x‖ mean & std. Compare S1 vs S2 per
    condition.

  Part C — intermediate-layer activation breakdown
    Both projections are nn.Sequential(Linear, MLP). Capture activations BETWEEN
    Linear and MLP for both pathways. If Linear output is already small in collapsed
    → collapse is at the first linear projection. If Linear output is healthy but
    MLP output collapses → collapse is in MLP/LayerNorm/residual stack.
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, '/home/kojiek/MeanAudio')
os.chdir('/home/kojiek/MeanAudio')

import torch
import torch.nn as nn
import numpy as np


from meanaudio.model.networks import get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils


# (S1 = FluxAudio, S2 = MeanAudio); use_meanflow not relevant for projection inspection
CHECKPOINTS = [
    ('P8_healthy',          'fluxaudio_s', '/home/kojiek/MeanAudio/exps/phase8_stage1_400000/phase8_stage1_400000_ema_final.pth'),
    ('P8_healthy',          'meanaudio_s', '/home/kojiek/MeanAudio/exps/phase8_stage2_200000/phase8_stage2_200000_ema_final.pth'),
    ('EXP_A_LPMCstripped',  'fluxaudio_s', '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage1_400000/p_lpmc_destructured_stage1_400000_ema_final.pth'),
    ('EXP_A_LPMCstripped',  'meanaudio_s', '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage2_200000/p_lpmc_destructured_stage2_200000_ema_final.pth'),
    ('EXP_B_Qwen_slot0',    'fluxaudio_s', '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage1_400000/p_qwen_slot0_stage1_400000_ema_final.pth'),
    ('EXP_B_Qwen_slot0',    'meanaudio_s', '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth'),
    ('EXP_C_Qwen_boilerplate', 'fluxaudio_s', '/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage1_400000/p_qwen_slot0_boilerplate_stage1_400000_ema_final.pth'),
    ('EXP_C_Qwen_boilerplate', 'meanaudio_s', '/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage2_200000/p_qwen_slot0_boilerplate_stage2_200000_ema_final.pth'),
]

PROMPTS = [
    "The low quality recording features a folk song, which consists of acoustic guitar strumming a chord progression in a bright key.",
    "This is the kind of music that would be played to create a calm and mysterious ambiance. There is a bright sustained synth pad and a fuzzy low-end bass synth used to play the melody.",
    "An indie pop track featuring a vibrant and uplifting melody with a steady drum beat and warm electric guitar tones, creating a feel-good atmosphere throughout.",
    "A soothing folk-inspired ballad with gentle acoustic guitar fingerpicking, atmospheric string pads, and a melancholic vocal melody that evokes nostalgic feelings.",
    "Acoustic guitar with soft drums and bass.",
    "Electronic dance music with synthesizers and four-on-the-floor drums.",
    "A piano playing a slow melancholic melody.",
    "Heavy metal with distorted electric guitar and double bass drums.",
]

DEVICE = 'cuda'
DTYPE = torch.bfloat16


def weight_norms(module, prefix=''):
    """Return list of (name, ‖W‖_2 (full), ‖W‖_F/sqrt(numel) (avg-mag), ‖bias‖) for all leaf params."""
    rows = []
    for name, p in module.named_parameters(recurse=True):
        fullname = f"{prefix}.{name}" if prefix else name
        w = p.detach().float()
        n = w.numel()
        l2 = w.norm().item()
        avg = (l2 / (n ** 0.5)) if n else 0.0
        rows.append((fullname, n, round(l2, 5), round(avg, 6)))
    return rows


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading FeaturesUtils (T5 + CLAP)...")
    feat = FeaturesUtils(
        tod_vae_ckpt='./weights/v1-16.pth',
        enable_conditions=True,
        encoder_name='t5_clap',
        mode='16k',
        bigvgan_vocoder_ckpt='./weights/best_netG.pt',
        need_vae_encoder=False,
    ).to(DEVICE, DTYPE).eval()

    with torch.no_grad():
        text_features, text_features_c = feat.encode_text(PROMPTS)
    text_features = text_features.to(DTYPE)
    text_features_c = text_features_c.to(DTYPE)
    print(f"  text_features {tuple(text_features.shape)}   text_features_c {tuple(text_features_c.shape)}")
    in_t5_mag  = text_features.float().pow(2).sum(dim=-1).sqrt().mean().item()
    in_clap_mag = text_features_c.float().norm(dim=-1).mean().item()
    print(f"  INPUT magnitudes: CLAP ‖x‖={in_clap_mag:.3f}   T5 per-token ‖x‖={in_t5_mag:.3f}")

    del feat
    torch.cuda.empty_cache()

    results = {
        'input': {'clap_mag': in_clap_mag, 't5_pertok_mag': in_t5_mag},
        'rows': [],
    }

    for cond_name, arch, ckpt in CHECKPOINTS:
        stage = 'S1' if arch == 'fluxaudio_s' else 'S2'
        print(f"\n[{time.strftime('%H:%M:%S')}] === {cond_name} {stage} ({arch}) ===")
        try:
            net = get_mean_audio(arch, use_rope=False, text_c_dim=512).to(DEVICE, DTYPE).eval()
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
            net.load_weights(state)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            continue

        # --- Part A: weight norms ---
        cond_w = weight_norms(net.text_cond_proj, 'text_cond_proj')
        in_w   = weight_norms(net.text_input_proj, 'text_input_proj')

        # Headline: per-pathway aggregate weight L2
        cond_l2_total = sum(row[2] for row in cond_w)
        in_l2_total   = sum(row[2] for row in in_w)
        cond_avg_mag  = float(np.mean([row[3] for row in cond_w]))
        in_avg_mag    = float(np.mean([row[3] for row in in_w]))

        # --- Part B: end-to-end activation on prompts ---
        with torch.no_grad():
            cond_out = net.text_cond_proj(text_features_c)   # (B, hidden_dim)
            text_out = net.text_input_proj(text_features)    # (B, 77, hidden_dim)
        cond_mag_mean = cond_out.float().norm(dim=-1).mean().item()
        cond_mag_std  = cond_out.float().norm(dim=-1).std().item()
        text_mag_mean = text_out.float().pow(2).sum(dim=-1).sqrt().mean().item()

        # --- Part C: intermediate (Linear-only) activations ---
        # text_cond_proj[0] is Linear(text_c_dim, hidden_dim); text_input_proj[0] is Linear(text_dim, hidden_dim)
        # That's the FIRST projection. Compare to full output to localize where collapse happens.
        with torch.no_grad():
            cond_lin = net.text_cond_proj[0](text_features_c)  # after first Linear
            text_lin = net.text_input_proj[0](text_features)   # after first Linear
        cond_lin_mag = cond_lin.float().norm(dim=-1).mean().item()
        text_lin_mag = text_lin.float().pow(2).sum(dim=-1).sqrt().mean().item()

        row = {
            'cond': cond_name, 'stage': stage,
            'text_cond_proj_total_W_L2': round(cond_l2_total, 4),
            'text_cond_proj_avg_mag_W':   round(cond_avg_mag, 6),
            'text_input_proj_total_W_L2': round(in_l2_total, 4),
            'text_input_proj_avg_mag_W':   round(in_avg_mag, 6),
            'cond_lin_out_mag': round(cond_lin_mag, 4),
            'cond_full_out_mag': round(cond_mag_mean, 4),
            'text_lin_out_mag': round(text_lin_mag, 4),
            'text_full_out_mag': round(text_mag_mean, 4),
            'cond_inter_prompt_std': round(cond_mag_std, 4),
        }
        results['rows'].append(row)

        print(f"  WEIGHTS:")
        print(f"    cond_proj total ‖W‖_2 = {cond_l2_total:8.3f}   avg-mag = {cond_avg_mag:.5f}")
        print(f"    text_proj total ‖W‖_2 = {in_l2_total:8.3f}   avg-mag = {in_avg_mag:.5f}")
        print(f"  ACTIVATIONS on 8 prompts:")
        print(f"    CLAP cond:  Linear-only ‖x‖ = {cond_lin_mag:7.3f}   full ‖x‖ = {cond_mag_mean:7.3f}")
        print(f"    T5 input :  Linear-only ‖x‖ = {text_lin_mag:7.3f}   full ‖x‖ = {text_mag_mean:7.3f}")

        del net
        torch.cuda.empty_cache()

    # ========= Summary tables =========
    out_path = '/home/kojiek/research/meanaudio_training/exp_d3_audit_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] Saved {out_path}")

    print("\n=== WEIGHT-NORM AUDIT (Part A) ===")
    print(f"{'Cond':<24} | {'St':<3} | {'cond_W_L2':>10} | {'cond_W_avg':>11} | {'text_W_L2':>10} | {'text_W_avg':>11}")
    print("-" * 92)
    for r in results['rows']:
        print(f"{r['cond']:<24} | {r['stage']:<3} | {r['text_cond_proj_total_W_L2']:>10.3f} | {r['text_cond_proj_avg_mag_W']:>11.5f} | {r['text_input_proj_total_W_L2']:>10.3f} | {r['text_input_proj_avg_mag_W']:>11.5f}")

    print("\n=== ACTIVATION AUDIT (Part B+C) ===")
    print(f"{'Cond':<24} | {'St':<3} | {'CLAP_lin':>9} | {'CLAP_full':>10} | {'T5_lin':>8} | {'T5_full':>9}")
    print("-" * 80)
    for r in results['rows']:
        print(f"{r['cond']:<24} | {r['stage']:<3} | {r['cond_lin_out_mag']:>9.3f} | {r['cond_full_out_mag']:>10.3f} | {r['text_lin_out_mag']:>8.3f} | {r['text_full_out_mag']:>9.3f}")

    print("\n=== Stage delta (S2 / S1 ratio per condition) ===")
    # Build a {cond: {stage: row}} dict for easy lookup
    by_cond = {}
    for r in results['rows']:
        by_cond.setdefault(r['cond'], {})[r['stage']] = r
    print(f"{'Cond':<24} | {'cond_W_L2 S1→S2':>20} | {'cond_full_act S1→S2':>22} | {'text_full_act S1→S2':>22}")
    for cond, sd in by_cond.items():
        if 'S1' in sd and 'S2' in sd:
            s1, s2 = sd['S1'], sd['S2']
            cw = f"{s1['text_cond_proj_total_W_L2']:.2f}→{s2['text_cond_proj_total_W_L2']:.2f}"
            ca = f"{s1['cond_full_out_mag']:.3f}→{s2['cond_full_out_mag']:.3f}"
            ta = f"{s1['text_full_out_mag']:.3f}→{s2['text_full_out_mag']:.3f}"
            print(f"{cond:<24} | {cw:>20} | {ca:>22} | {ta:>22}")

    print("\nInterpretation guide:")
    print("  (M1) Weight shrinkage: cond_W_L2 and text_W_L2 both small in collapsed → confirmed M1")
    print("  (M2) Activation cancellation: weights similar across models BUT activations small → M2 (LayerNorm/bias)")
    print("  Stage origin: if S1 cond_full_act already much smaller in collapsed conditions → S1-origin (caption regime).")
    print("                if S1 activations similar to P8 but S2 collapses → S2-origin (MeanFlow distillation).")


if __name__ == '__main__':
    main()
