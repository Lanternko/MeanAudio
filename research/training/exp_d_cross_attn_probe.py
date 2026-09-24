"""
EXP-D: Cross-attention probe — does collapsed model ignore text channel?

For each of 4 models (P8 healthy + EXP-A/B/C collapsed):
  - Build MeanAudio (S2) architecture
  - Load checkpoint
  - Encode N fixed prompts (T5 + CLAP) once
  - Run a single forward at fixed t,r with fixed noise latent
  - Capture joint attention from every JointBlock (monkey-patched attention())
  - Compute per-block audio→text statistics:
      * attn mass (fraction of each audio token's attention going to text tokens)
      * normalized entropy of audio→text distribution (per audio token)

If collapsed models show systematically lower mass or near-uniform entropy
vs P8, we have direct evidence the text channel is being underused.
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, '/home/kojiek/MeanAudio')
os.chdir('/home/kojiek/MeanAudio')

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ============================================================
# Monkey-patch attention() to capture weights (explicit softmax)
# ============================================================
import meanaudio.model.transformer_layers as tl

CAPTURED = []  # list of attention matrices captured by this run

def attention_capture(q, k, v):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    # Capture (move to CPU fp32 to save GPU memory)
    CAPTURED.append(attn.detach().to(torch.float32).cpu())
    out = attn @ v
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out

tl.attention = attention_capture
# ============================================================

from meanaudio.model.networks import get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils


MODELS = {
    'P8_healthy_LPMC':       '/home/kojiek/MeanAudio/exps/phase8_stage2_200000/phase8_stage2_200000_ema_final.pth',
    'EXP_A_LPMC_stripped':   '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage2_200000/p_lpmc_destructured_stage2_200000_ema_final.pth',
    'EXP_B_Qwen_slot0':      '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth',
    'EXP_C_Qwen_boilerplate':'/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage2_200000/p_qwen_slot0_boilerplate_stage2_200000_ema_final.pth',
}

PROMPTS = [
    # LP-MC writing-task style (P8 / EXP-A training distribution)
    "The low quality recording features a folk song, which consists of acoustic guitar strumming a chord progression in a bright key.",
    "This is the kind of music that would be played to create a calm and mysterious ambiance. There is a bright sustained synth pad and a fuzzy low-end bass synth used to play the melody.",
    # Qwen style (EXP-B / EXP-C training distribution)
    "An indie pop track featuring a vibrant and uplifting melody with a steady drum beat and warm electric guitar tones, creating a feel-good atmosphere throughout.",
    "A soothing folk-inspired ballad with gentle acoustic guitar fingerpicking, atmospheric string pads, and a melancholic vocal melody that evokes nostalgic feelings.",
    # Short neutral
    "Acoustic guitar with soft drums and bass.",
    "Electronic dance music with synthesizers and four-on-the-floor drums.",
    "A piano playing a slow melancholic melody.",
    "Heavy metal with distorted electric guitar and double bass drums.",
]

DEVICE = 'cuda'
DTYPE = torch.bfloat16

LATENT_SEQ_LEN = 312
LATENT_DIM = 20
TEXT_SEQ_LEN = 77


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
    print(f"  text_features: {tuple(text_features.shape)}  text_features_c: {tuple(text_features_c.shape)}")

    B = len(PROMPTS)
    L = LATENT_SEQ_LEN
    T = TEXT_SEQ_LEN

    # Free encoder GPU memory
    del feat
    torch.cuda.empty_cache()

    results = {}
    for name, ckpt_path in MODELS.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] === {name} ===")
        CAPTURED.clear()

        net = get_mean_audio('meanaudio_s', use_rope=False, text_c_dim=512).to(DEVICE, DTYPE).eval()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        net.load_weights(state)

        # Pure noise latent (fixed seed) — mid-trajectory probe
        g = torch.Generator(device=DEVICE).manual_seed(42)
        latent = torch.randn(B, L, LATENT_DIM, generator=g, device=DEVICE, dtype=DTYPE)
        # MeanFlow time params
        t = torch.full((B,), 0.5, device=DEVICE, dtype=DTYPE)
        r = torch.full((B,), 0.5, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            _ = net.forward(
                latent=latent,
                text_f=text_features,
                text_f_c=text_features_c,
                r=r, t=t, q=None,
            )

        print(f"  Forward done. Captured {len(CAPTURED)} attention matrices.")

        # Identify JointBlocks: shape (B, H, L+T, L+T). Fused are (B, H, L, L).
        per_block = []
        for i, attn in enumerate(CAPTURED):
            shape = tuple(attn.shape)
            seq_total = shape[-1]
            is_joint = (seq_total == L + T)
            if not is_joint:
                continue
            # Audio→text portion
            a2t = attn[:, :, :L, L:]  # (B, H, L, T)
            # Mass per audio token going to text (sum over T)
            a2t_mass = a2t.sum(dim=-1)  # (B, H, L)
            avg_mass = a2t_mass.mean().item()
            std_mass = a2t_mass.std().item()
            # Normalized entropy: renormalize a2t along T to sum=1 per audio token, compute entropy / log(T)
            a2t_renorm = a2t / (a2t.sum(dim=-1, keepdim=True) + 1e-9)
            ent = -(a2t_renorm * (a2t_renorm + 1e-9).log()).sum(dim=-1)  # (B, H, L)
            avg_ent_norm = (ent.mean() / float(np.log(T))).item()
            # Top-1 text token concentration (max a2t per audio token, mean)
            top1_mass = a2t.max(dim=-1).values.mean().item()

            per_block.append({
                'block_idx': i,
                'shape': list(shape),
                'avg_a2t_mass': round(avg_mass, 6),
                'std_a2t_mass': round(std_mass, 6),
                'avg_a2t_norm_entropy': round(avg_ent_norm, 6),
                'avg_top1_text_mass': round(top1_mass, 6),
            })

        results[name] = per_block

        # Summary print
        masses = [m['avg_a2t_mass'] for m in per_block]
        ents = [m['avg_a2t_norm_entropy'] for m in per_block]
        tops = [m['avg_top1_text_mass'] for m in per_block]
        print(f"  JointBlocks: {len(per_block)}")
        print(f"    avg a2t mass     = {np.mean(masses):.4f}  (per-block range {min(masses):.4f}..{max(masses):.4f})")
        print(f"    avg norm entropy = {np.mean(ents):.4f}  (1.0=uniform, lower=more concentrated)")
        print(f"    avg top1 text mass = {np.mean(tops):.4f}  (higher=sharper attention to single text token)")

        del net
        torch.cuda.empty_cache()

    out_path = '/home/kojiek/research/meanaudio_training/exp_d_cross_attn_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to {out_path}")

    # Top-level comparison
    print("\n=== Cross-model summary (averaged over JointBlocks) ===")
    print(f"{'Model':<26} {'a2t_mass':>10} {'norm_entropy':>14} {'top1_mass':>10}")
    print("-" * 64)
    for name, blocks in results.items():
        masses = [m['avg_a2t_mass'] for m in blocks]
        ents   = [m['avg_a2t_norm_entropy'] for m in blocks]
        tops   = [m['avg_top1_text_mass'] for m in blocks]
        print(f"{name:<26} {np.mean(masses):>10.4f} {np.mean(ents):>14.4f} {np.mean(tops):>10.4f}")


if __name__ == '__main__':
    main()
