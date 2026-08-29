import logging
from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import torchaudio
import soundfile as sf
import csv
from meanaudio.eval_utils import (ModelConfig, all_model_cfg, generate_fm, generate_mf, setup_eval_logging)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import MeanAudio, get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm
log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='meanaudio_mf',
                        help='meanaudio_mf, fluxaudio_fm')
    
    parser.add_argument('--audio_path', type=str, help='Input audio', default='')
    parser.add_argument('--duration', type=float, default=9.975)  # for 312 latents, seq_config should has a duration of 9.975s 
    parser.add_argument('--cfg_strength', type=float, default=0.0,
                        help='Canonical evaluation default: pure-conditional CFG 0. Use guided CFG only in a separately named, preregistered secondary protocol.')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--quality_level', type=int, default=9, help='Quality level for inference (0-9)')
    parser.add_argument('--no_q', action='store_true', help='Disable q conditioning (use null token=10); for models trained without q conditioning')
    parser.add_argument(
        '--no_text_attention_mask', action='store_true',
        help='Reproduce the legacy path where all 77 T5 positions participate in joint attention')
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Text pushed away from by classifier-free guidance. Only has an effect at '
             '--cfg_strength >= 1.0, where ode_wrapper mixes cfg*cond + (1-cfg)*negative; '
             'below 1.0 the pure conditional branch is returned and this is ignored. '
             'Empty (the default) falls back to the network\'s stored empty_string_feat, '
             'i.e. the null condition training used for CFG dropout, which is textbook '
             'classifier-free guidance. Do not pass an explicit empty string expecting '
             'that: T5-encoding \'\' at inference gives a different tensor.')
    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--model_path', type=str, help='Ckpt path of trained model')
    parser.add_argument('--encoder_name', choices=['clip', 't5', 't5_clap'], type=str, help='text encoder name')
    parser.add_argument('--use_rope', action='store_true', help='Whether or not use position embedding for model')
    parser.add_argument('--text_c_dim', type=int, default=512, 
                        help='Dim of the text_features_c, 1024 for pooled T5 and 512 for CLAP')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tsv', type=str,
                        default='./sets/test-audiocaps.tsv',
                        help='Path to TSV file with id and caption columns')
    parser.add_argument('--use_meanflow', action='store_true', help='Whether or not use mean flow for inference')
    args = parser.parse_args()
    log.info(f'Eval args: {vars(args)}')

    if args.debug:
        import debugpy
        debugpy.listen(6665) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]  # model is just the model config
    # model.download_if_needed()
    seq_cfg = model.seq_cfg  

    negative_prompt: str = args.negative_prompt
    # An empty --negative_prompt must fall back to the network's stored
    # empty_string_feat (weights/empty_string_t5.pth), which is the null condition
    # training used for CFG dropout. Passing [''] instead T5-encodes the empty
    # string at inference, and that tensor is NOT the same one: whole-tensor
    # cosine -0.158 against the stored feature (the stored one is constant across
    # all 77 positions, live T5 is not). Only matters at cfg >= 1.0, where the
    # negative branch is actually evaluated.
    negative_text_arg = [negative_prompt] if negative_prompt else None
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)
    # load a pretrained model
    net: MeanAudio = get_mean_audio(model.model_name, 
                                    use_rope=args.use_rope, 
                                    text_c_dim=args.text_c_dim).to(device, dtype).eval() 
    net.load_weights(torch.load(args.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {args.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    if args.use_meanflow:
        mf = MeanFlow(steps=num_steps)
    else:
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                    enable_conditions=True,
                                    encoder_name=args.encoder_name, 
                                    mode=model.mode,
                                    bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                    need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len)

    eval_file = args.tsv
    audio_ids=[]  
    text_prompts=[]
    q_levels=[]
    with open(eval_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t') 
            for row in reader:
                audio_ids.append(row['id'])
                text_prompts.append(row['caption'])
                if args.no_q:
                    q_levels.append(10)  # null token, consistent with training without q conditioning
                else:
                    q_levels.append(int(row['q_level']) if 'q_level' in row else args.quality_level)

    for k in tqdm(range(0, len(text_prompts))):
        save_paths = output_dir / f'{audio_ids[k]}.flac'
        if save_paths.exists():
            continue
        prompt = text_prompts[k]
        if args.use_meanflow:
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_mf([prompt],
                                negative_text=negative_text_arg,
                                feature_utils=feature_utils,
                                net=net,
                                mf=mf,
                                rng=rng,
                                cfg_strength=cfg_strength,
                                q_level=q_levels[k],
                                use_text_attention_mask=not args.no_text_attention_mask)
            audio = audios.float().cpu()[0]
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                log.warning(f'NaN/Inf in audio for {audio_ids[k]}, skipping')
                continue
            try:
                sf.write(str(save_paths), audio.squeeze(0).numpy(), seq_cfg.sampling_rate)
            except Exception as e:
                log.warning(f'Failed to write audio for {audio_ids[k]}: {e}, skipping')
                continue
            log.info(f'Audio saved to {save_paths}')
            log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

        else:
            prompt = text_prompts[k]
            log.info(f'Prompt: {prompt}')
            log.info(f'Negative prompt: {negative_prompt}')
            audios = generate_fm([prompt],
                                negative_text=negative_text_arg,
                                feature_utils=feature_utils,
                                net=net,
                                fm=fm,
                                rng=rng,
                                cfg_strength=cfg_strength,
                                q_level=q_levels[k],
                                use_text_attention_mask=not args.no_text_attention_mask)
            audio = audios.float().cpu()[0]
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                log.warning(f'NaN/Inf in audio for {audio_ids[k]}, skipping')
                continue
            try:
                sf.write(str(save_paths), audio.squeeze(0).numpy(), seq_cfg.sampling_rate)
            except Exception as e:
                log.warning(f'Failed to write audio for {audio_ids[k]}: {e}, skipping')
                continue
            log.info(f'Audio saved to {save_paths}')
            log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

    
if __name__ == '__main__':
    main()
