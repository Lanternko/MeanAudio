"""
Manual EMA synthesis for EXP-B Stage 1.
The training crashed at synthesize_ema due to corrupt 1.70000.pt (already moved aside).
This script re-runs synthesize_ema with the cleaned ema_ckpts directory.
"""
import sys, os
sys.path.insert(0, '/home/kojiek/MeanAudio')
os.chdir('/home/kojiek/MeanAudio')

import torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from meanaudio.utils.synthesize_ema import synthesize_ema

EXP = 'p_qwen_slot0_stage1_400000'
HYDRA_DIR = f'/home/kojiek/MeanAudio/exps/{EXP}/train-2026-05-10_21-42-05-hydra'
OUT = f'/home/kojiek/MeanAudio/exps/{EXP}/{EXP}_ema_final.pth'

# Load the actual config that was used in training
from omegaconf import OmegaConf
cfg = OmegaConf.load(f'{HYDRA_DIR}/config.yaml')
# Register dummy hydra resolver to allow ${hydra:...} to resolve to actual exp dir
OmegaConf.register_new_resolver('hydra', lambda key: f'/home/kojiek/MeanAudio/exps/{EXP}', replace=True)
# Override resolved checkpoint_folder explicitly so we don't rely on the dummy
OmegaConf.set_struct(cfg, False)
cfg.ema.checkpoint_folder = f'/home/kojiek/MeanAudio/exps/{EXP}/ema_ckpts'
print(f'Loaded config; exp_id={cfg.exp_id}')
print(f'EMA: sigma_rels={list(cfg.ema.sigma_rels)}, default_output_sigma={cfg.ema.default_output_sigma}')
print(f'EMA folder: {cfg.ema.checkpoint_folder}')

print(f'\nSynthesizing EMA at sigma={cfg.ema.default_output_sigma} ...')
state_dict = synthesize_ema(cfg, cfg.ema.default_output_sigma, step=None)
print(f'Got state_dict with {len(state_dict)} keys')

torch.save(state_dict, OUT)
print(f'\nSaved -> {OUT}')
print(f'Size: {os.path.getsize(OUT)/1e6:.1f} MB')
