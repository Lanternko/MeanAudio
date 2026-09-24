"""
EXP-D4: Projection transplant — does swapping P8's text_cond_proj + text_input_proj
        into EXP-A/B/C S2 ckpts restore text-conditioning behavior?

Steps:
  1. Load P8 S2 ema_final.pth → extract keys starting with 'text_cond_proj.' or
     'text_input_proj.'  (these are state_dict tensors, no architecture change)
  2. For each collapsed S2 ckpt (EXP-A/B/C), patch those keys with P8's values,
     leaving all other weights (joint_blocks, fused_blocks, latent paths, q_embed,
     t_embed, r_embed, etc.) untouched. Save to a new path; never overwrite
     original.
  3. Sanity probe: forward the patched ckpt at fixed t/r with the same 8 prompts
     used in EXP-D2, confirm CLAP cond_proj output ‖x‖ and T5 text_proj output ‖x‖
     are now close to P8 levels (≈6.9 and ≈160). If not, the transplant didn't
     take — likely a key-name or dtype issue.

Output:
  - 3 new ema_final.pth under exps/{cond}_p8proj_transplant/
  - exp_d4_sanity_results.json with pre/post activation magnitudes per model
"""

import sys, os, json, time
from pathlib import Path

sys.path.insert(0, '/home/kojiek/MeanAudio')
os.chdir('/home/kojiek/MeanAudio')

import torch
import numpy as np


from meanaudio.model.networks import get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils


P8_S2 = '/home/kojiek/MeanAudio/exps/phase8_stage2_200000/phase8_stage2_200000_ema_final.pth'

TRANSPLANTS = [
    # (cond_name, source_ckpt, target_outdir_name)
    ('EXP_A_LPMCstripped',     '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage2_200000/p_lpmc_destructured_stage2_200000_ema_final.pth',         'exp_a_p8proj_transplant'),
    ('EXP_B_Qwen_slot0',       '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth',                       'exp_b_p8proj_transplant'),
    ('EXP_C_Qwen_boilerplate', '/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage2_200000/p_qwen_slot0_boilerplate_stage2_200000_ema_final.pth', 'exp_c_p8proj_transplant'),
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


def main():
    # -------- Step 1: pick P8 projection keys --------
    print(f"[{time.strftime('%H:%M:%S')}] Loading P8 S2 state_dict...")
    p8_sd = torch.load(P8_S2, map_location='cpu', weights_only=True)

    proj_keys = [k for k in p8_sd.keys() if k.startswith('text_cond_proj.') or k.startswith('text_input_proj.')]
    print(f"  Projection keys to transplant ({len(proj_keys)}):")
    for k in proj_keys:
        print(f"    {k}  shape={tuple(p8_sd[k].shape)}")

    if not proj_keys:
        print("ERROR: no projection keys found in P8 S2 ckpt — abort.")
        return

    # -------- Step 2: build patched ckpts --------
    patched_paths = []
    for cond_name, src_ckpt, outdir_name in TRANSPLANTS:
        outdir = Path(f'/home/kojiek/MeanAudio/exps/{outdir_name}')
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f'{outdir_name}_ema_final.pth'

        print(f"\n[{time.strftime('%H:%M:%S')}] Patching {cond_name}")
        print(f"  src : {src_ckpt}")
        print(f"  dst : {out_path}")

        sd = torch.load(src_ckpt, map_location='cpu', weights_only=True)

        # Verify keys match P8
        missing = [k for k in proj_keys if k not in sd]
        if missing:
            print(f"  WARNING: target ckpt missing these projection keys (will add): {missing}")

        n_patched = 0
        for k in proj_keys:
            if k in sd and sd[k].shape != p8_sd[k].shape:
                print(f"  SHAPE MISMATCH for {k}: src {sd[k].shape} vs P8 {p8_sd[k].shape} — SKIP")
                continue
            sd[k] = p8_sd[k].clone()
            n_patched += 1
        print(f"  Patched {n_patched}/{len(proj_keys)} keys, total target ckpt keys: {len(sd)}")

        torch.save(sd, str(out_path))
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"  Saved {size_mb:.1f} MB")
        patched_paths.append((cond_name, str(out_path)))

    # -------- Step 3: sanity probe --------
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading FeaturesUtils for sanity probe...")
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
    del feat
    torch.cuda.empty_cache()

    # Probe targets: P8 (reference), 3 originals, 3 patched
    PROBE_TARGETS = [
        ('P8_S2_reference', P8_S2),
        ('EXP_A_S2_orig',   '/home/kojiek/MeanAudio/exps/p_lpmc_destructured_stage2_200000/p_lpmc_destructured_stage2_200000_ema_final.pth'),
        ('EXP_A_S2_p8proj', f'/home/kojiek/MeanAudio/exps/exp_a_p8proj_transplant/exp_a_p8proj_transplant_ema_final.pth'),
        ('EXP_B_S2_orig',   '/home/kojiek/MeanAudio/exps/p_qwen_slot0_stage2_200000/p_qwen_slot0_stage2_200000_ema_final.pth'),
        ('EXP_B_S2_p8proj', f'/home/kojiek/MeanAudio/exps/exp_b_p8proj_transplant/exp_b_p8proj_transplant_ema_final.pth'),
        ('EXP_C_S2_orig',   '/home/kojiek/MeanAudio/exps/p_qwen_slot0_boilerplate_stage2_200000/p_qwen_slot0_boilerplate_stage2_200000_ema_final.pth'),
        ('EXP_C_S2_p8proj', f'/home/kojiek/MeanAudio/exps/exp_c_p8proj_transplant/exp_c_p8proj_transplant_ema_final.pth'),
    ]

    sanity = []
    for name, ckpt in PROBE_TARGETS:
        if not os.path.exists(ckpt):
            print(f"  SKIP {name}: not found at {ckpt}")
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] Probing {name}")
        net = get_mean_audio('meanaudio_s', use_rope=False, text_c_dim=512).to(DEVICE, DTYPE).eval()
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        net.load_weights(state)

        with torch.no_grad():
            cond_out = net.text_cond_proj(text_features_c)
            text_out = net.text_input_proj(text_features)
        cond_mag = cond_out.float().norm(dim=-1).mean().item()
        text_mag = text_out.float().pow(2).sum(dim=-1).sqrt().mean().item()

        print(f"  CLAP cond ‖x‖ = {cond_mag:.4f}")
        print(f"  T5   text ‖x‖ = {text_mag:.4f}")
        sanity.append({'name': name, 'cond_mag': round(cond_mag, 4), 'text_mag': round(text_mag, 4)})

        del net
        torch.cuda.empty_cache()

    out_path = '/home/kojiek/research/meanaudio_training/exp_d4_sanity_results.json'
    with open(out_path, 'w') as f:
        json.dump({'sanity': sanity, 'patched_paths': patched_paths}, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] Saved {out_path}")

    print("\n=== EXP-D4 transplant sanity ===")
    print(f"{'Target':<24} | {'CLAP cond ‖x‖':>14} | {'T5 text ‖x‖':>14}")
    print("-" * 60)
    for s in sanity:
        print(f"{s['name']:<24} | {s['cond_mag']:>14.4f} | {s['text_mag']:>14.4f}")

    print("\nExpected: *_p8proj rows should have ‖x‖ very close to P8_S2_reference. If not, transplant key/shape mismatch.")


if __name__ == '__main__':
    main()
