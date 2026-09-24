"""
EXP-D-2: Extended pathway probe — go beyond cross-attn weights.

Round 1 (exp_d_cross_attn_probe.py) showed all 4 models attend to text at similar
magnitudes — cross-attn weights do NOT separate healthy from collapsed. This
round probes the OTHER pathways that could carry the collapse signature:

  P1. CLAP cond pathway: text_features_c (512) → text_cond_proj → extended_c
      drives AdaLN modulation in the latent (audio) branch. If this pathway is
      clustered or low-magnitude in collapsed models, conditioning is dead.

  P2. Per-JointBlock gate_msa: the scalar gate that multiplies cross-attn output
      onto the latent residual. If gate ≈ 0 in collapsed models, attention is
      computed but its contribution to the audio representation is zeroed.

  P3. Text input projection: text_input_proj(T5_features) — the actual keys/values
      that audio attends to. Magnitude / inter-prompt diversity.

  P4. Cross-attn OUTPUT norm: ||x_attn_out||, the actual audio-side product of
      attention before the gate. Distinguishes "attention is there but signal is small"
      from "attention contributes."

  P5. Multi-timestep: run at t=0.2 / 0.5 / 0.8 to check pattern stability.

  P6. Per-block breakdown (not just averaged) — collapsed/healthy may diverge at
      specific layers (early vs late).
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
# Monkey-patch attention() to capture both weights and output norm
# ============================================================
import meanaudio.model.transformer_layers as tl

CAPTURED_ATTN = []          # list of full attention matrices (B, H, N, N)
CAPTURED_ATTN_OUT_NORM = [] # list of per-token output norms (B, N) - last dim collapsed

def attention_capture(q, k, v):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    CAPTURED_ATTN.append(attn.detach().to(torch.float32).cpu())
    out = attn @ v  # (B, H, N, D)
    # Per-token output norm (before head-merge) across heads-and-dims
    out_norm = out.float().pow(2).sum(dim=(1, 3)).sqrt().cpu()  # (B, N)
    CAPTURED_ATTN_OUT_NORM.append(out_norm)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out

tl.attention = attention_capture
# ============================================================

from meanaudio.model.networks import get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils


MODELS = {
    'P8_healthy':           '/home/kojiek/MeanAudio/exps/phase8_stage2_200000/phase8_stage2_200000_ema_final.pth',
    'EXP_A_LPMCstripped':   '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage2_200000/p_lpmc_destructured_stage2_200000_ema_final.pth',
    'EXP_B_Qwen_slot0':     '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth',
    'EXP_C_Qwen_boilerplate':'/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage2_200000/p_qwen_slot0_boilerplate_stage2_200000_ema_final.pth',
}

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

L = 312      # latent_seq_len
LATENT_DIM = 20
T = 77       # text_seq_len
TIMESTEPS = [0.2, 0.5, 0.8]


def pairwise_cos_offdiag_mean(x_BD):
    """Mean off-diagonal cosine similarity. x: (B, D). Returns scalar."""
    x = x_BD.float()
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    cos = x @ x.T  # (B, B)
    B = x.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    return cos[mask].mean().item()


def per_token_pairwise_cos_offdiag_mean(x_BLD):
    """For sequence features (B, L, D), average pairwise off-diag cos per token, then mean over L."""
    x = x_BLD.float()
    x_normed = x / (x.norm(dim=-1, keepdim=True) + 1e-9)  # (B, L, D)
    # for each position l, compute (B,B) cos matrix
    B = x_normed.shape[0]
    L_ = x_normed.shape[1]
    # cos[b1, b2, l] = sum over d of x[b1,l,d] * x[b2,l,d]
    cos = torch.einsum('bld,cld->bcl', x_normed, x_normed)  # (B, B, L)
    mask = ~torch.eye(B, dtype=torch.bool)
    cos_offdiag = cos[mask, :]  # (B*(B-1), L)
    return cos_offdiag.mean().item()


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

    # Inter-prompt diversity in input space (sanity check; should be identical across models)
    in_clap_cos = pairwise_cos_offdiag_mean(text_features_c)
    in_t5_cos   = per_token_pairwise_cos_offdiag_mean(text_features)
    print(f"  INPUT inter-prompt cos: CLAP={in_clap_cos:.4f}  T5(per-tok mean)={in_t5_cos:.4f}")

    del feat
    torch.cuda.empty_cache()

    B = len(PROMPTS)

    results = {
        'input_clap_cos_offdiag': in_clap_cos,
        'input_t5_pertok_cos_offdiag': in_t5_cos,
        'models': {},
    }

    for name, ckpt_path in MODELS.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] === {name} ===")
        net = get_mean_audio('meanaudio_s', use_rope=False, text_c_dim=512).to(DEVICE, DTYPE).eval()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        net.load_weights(state)

        model_res = {}

        # --- P1: CLAP cond pathway ---
        with torch.no_grad():
            cond_proj_out = net.text_cond_proj(text_features_c)  # (B, hidden_dim)
        cond_mag = cond_proj_out.float().norm(dim=-1).cpu()  # (B,)
        cond_cos = pairwise_cos_offdiag_mean(cond_proj_out.float().cpu())
        model_res['P1_clap_cond_path'] = {
            'magnitude_mean': cond_mag.mean().item(),
            'magnitude_std':  cond_mag.std().item(),
            'inter_prompt_cos_offdiag': cond_cos,
        }
        print(f"  P1 CLAP→cond_proj: |x|={cond_mag.mean():.3f}±{cond_mag.std():.3f}  inter-prompt cos={cond_cos:.4f}")

        # --- P3: Text input projection ---
        with torch.no_grad():
            text_proj_out = net.text_input_proj(text_features)  # (B, 77, hidden_dim)
        text_proj_mag = text_proj_out.float().pow(2).sum(dim=-1).sqrt().cpu()  # (B, 77)
        text_proj_pertok_cos = per_token_pairwise_cos_offdiag_mean(text_proj_out.float().cpu())
        model_res['P3_text_input_proj'] = {
            'pertoken_mag_mean': text_proj_mag.mean().item(),
            'pertoken_mag_std':  text_proj_mag.std().item(),
            'pertok_inter_prompt_cos_offdiag': text_proj_pertok_cos,
        }
        print(f"  P3 T5→text_proj: |x|={text_proj_mag.mean():.3f}  per-tok inter-prompt cos={text_proj_pertok_cos:.4f}")

        # --- Per-timestep forward + capture P2 (gate_msa), P4 (attn out norm), per-block attn ---
        # Register hooks on every joint_blocks[i].latent_block.adaLN_modulation
        gate_captures_per_t = {}
        def make_hook(t_key, idx):
            def hook(module, input, output):
                # output: (B, 6*dim). Split into 6 chunks per AdaLN: shift_msa, scale_msa, gate_msa, ...
                chunks = output.chunk(6, dim=-1)
                gate_msa = chunks[2]
                gate_captures_per_t[t_key].append((idx, gate_msa.detach().float().cpu()))
            return hook

        per_t_blocks = {}
        for t_val in TIMESTEPS:
            t_key = f"t{t_val}"
            gate_captures_per_t[t_key] = []
            handles = []
            for i, jb in enumerate(net.joint_blocks):
                # latent_block is MMDitSingleBlock; its adaLN_modulation outputs 6*dim for non-pre-only
                if not jb.latent_block.pre_only:
                    h = jb.latent_block.adaLN_modulation.register_forward_hook(make_hook(t_key, i))
                    handles.append(h)

            # Reset captures
            CAPTURED_ATTN.clear()
            CAPTURED_ATTN_OUT_NORM.clear()

            # Fixed noise latent (same seed across timesteps and models for reproducibility)
            g = torch.Generator(device=DEVICE).manual_seed(42)
            latent = torch.randn(B, L, LATENT_DIM, generator=g, device=DEVICE, dtype=DTYPE)
            t = torch.full((B,), t_val, device=DEVICE, dtype=DTYPE)
            r = torch.full((B,), t_val, device=DEVICE, dtype=DTYPE)

            with torch.no_grad():
                _ = net.forward(latent=latent, text_f=text_features, text_f_c=text_features_c,
                                r=r, t=t, q=None)

            # Per-JointBlock attention stats (joint blocks only)
            joint_block_metrics = []
            cross_blk_idx = 0  # counter for matching gate captures
            for cap_idx, attn in enumerate(CAPTURED_ATTN):
                if attn.shape[-1] != L + T:
                    continue
                a2t = attn[:, :, :L, L:]
                a2t_mass = a2t.sum(dim=-1).mean().item()
                a2t_renorm = a2t / (a2t.sum(dim=-1, keepdim=True) + 1e-9)
                ent = -(a2t_renorm * (a2t_renorm + 1e-9).log()).sum(dim=-1)
                norm_ent = (ent.mean() / float(np.log(T))).item()
                top1 = a2t.max(dim=-1).values.mean().item()
                # P4: cross-attn audio-side output norm
                out_norm = CAPTURED_ATTN_OUT_NORM[cap_idx][:, :L].mean().item()
                joint_block_metrics.append({
                    'cap_idx': cap_idx,
                    'a2t_mass': round(a2t_mass, 5),
                    'a2t_norm_entropy': round(norm_ent, 5),
                    'a2t_top1_mass': round(top1, 5),
                    'audio_attn_out_norm': round(out_norm, 5),
                })

            # Match gate_msa captures (idx → blk_idx in joint_blocks list)
            gate_caps = sorted(gate_captures_per_t[t_key], key=lambda x: x[0])
            for j, (blk_idx, gate_msa) in enumerate(gate_caps):
                if j < len(joint_block_metrics):
                    # gate_msa shape varies: typically (B, dim) if c is (B,D) or (B, N, dim) if extended
                    gm = gate_msa.float()
                    joint_block_metrics[j]['joint_blk_idx'] = blk_idx
                    joint_block_metrics[j]['gate_msa_mean_abs'] = round(gm.abs().mean().item(), 5)
                    joint_block_metrics[j]['gate_msa_l2'] = round(gm.norm().item() / (gm.numel() ** 0.5), 5)

            per_t_blocks[t_key] = joint_block_metrics

            for h in handles:
                h.remove()

        model_res['per_timestep'] = per_t_blocks

        # Compact per-model summary across timesteps × blocks
        all_mass = []
        all_ent = []
        all_outnorm = []
        all_gate = []
        for t_key, blocks in per_t_blocks.items():
            for blk in blocks:
                all_mass.append(blk['a2t_mass'])
                all_ent.append(blk['a2t_norm_entropy'])
                all_outnorm.append(blk['audio_attn_out_norm'])
                if 'gate_msa_mean_abs' in blk:
                    all_gate.append(blk['gate_msa_mean_abs'])

        model_res['summary'] = {
            'a2t_mass_mean':       round(float(np.mean(all_mass)), 4),
            'a2t_norm_entropy_mean': round(float(np.mean(all_ent)), 4),
            'attn_out_norm_mean':  round(float(np.mean(all_outnorm)), 4),
            'gate_msa_abs_mean':   round(float(np.mean(all_gate)), 4) if all_gate else None,
        }

        print(f"  Summary across t×block:  a2t_mass={model_res['summary']['a2t_mass_mean']:.4f} "
              f"norm_ent={model_res['summary']['a2t_norm_entropy_mean']:.4f} "
              f"attn_out_norm={model_res['summary']['attn_out_norm_mean']:.4f} "
              f"|gate_msa|={model_res['summary']['gate_msa_abs_mean']}")

        results['models'][name] = model_res
        del net
        torch.cuda.empty_cache()

    out_path = '/home/kojiek/research/meanaudio_training/exp_d2_pathway_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to {out_path}")

    # =========================
    # Cross-model summary table
    # =========================
    print("\n=== EXP-D2 cross-model summary ===")
    print(f"INPUT inter-prompt cos: CLAP={results['input_clap_cos_offdiag']:.4f}  T5(per-tok)={results['input_t5_pertok_cos_offdiag']:.4f}")
    print()
    print(f"{'Model':<24} | {'CLAP_cond_proj |x|':>20} | {'cond_cos':>9} | {'attn_out_norm':>14} | {'|gate_msa|':>11} | {'a2t_mass':>9}")
    print("-" * 102)
    for name, mres in results['models'].items():
        p1 = mres['P1_clap_cond_path']
        s = mres['summary']
        print(f"{name:<24} | {p1['magnitude_mean']:>20.3f} | {p1['inter_prompt_cos_offdiag']:>9.4f} | {s['attn_out_norm_mean']:>14.4f} | {str(s['gate_msa_abs_mean']):>11} | {s['a2t_mass_mean']:>9.4f}")

    print("\nInterpretation guide:")
    print("  - If P8 has LOW cond_cos and collapsed have HIGH cond_cos → CLAP-clustering bottleneck (H12)")
    print("  - If P8 has HIGH attn_out_norm and collapsed have LOW → cross-attn output dies before residual")
    print("  - If P8 has HIGH |gate_msa| and collapsed have LOW → cross-attn gated to zero in collapsed")


if __name__ == '__main__':
    main()
