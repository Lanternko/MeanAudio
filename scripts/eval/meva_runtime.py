"""Pinned MEva f03 adapter. Uses upstream HookedMusicGen, SAE and CNN classes.

No training, audio generation, loudness normalization or placeholder features.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'runtime/meva_20261003'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_architecture(path):
    # Execute only the exact upstream inference classes; avoid importing training CLI.
    import torch
    tree = ast.parse(Path(path).read_text())
    wanted = {'_masked_mean_pool', 'SAETower', 'SAEOnlyModel'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError('Missing upstream architecture')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

class Evaluator:
    def __init__(self, device='cuda'):
        import torch
        os.environ['HF_HOME'] = str(RUNTIME / 'hf-cache')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path[:0] = [str(ROOT / '.external/musicdiscovery'), str(ROOT / '.external/MEva')]
        from sae_components.musicgen_hooked import HookedMusicGen
        from sae_lens import SAE
        from audiocraft.models import MusicGen
        from audiocraft.modules.transformer import set_efficient_attention_backend
        # xformers binary was built against a different torch: use native torch attention.
        set_efficient_attention_backend('torch')
        class PinnedHook(HookedMusicGen):
            def get_model(self, model_name, device):
                return MusicGen.get_pretrained(str(RUNTIME / 'models/musicgen'), device=device)
        self.device = device
        self.musicgen = PinnedHook('facebook/musicgen-small', device=device).eval()
        self.sae = SAE.load_from_pretrained(str(RUNTIME / 'models/sae'), device=device).eval()
        if (self.sae.cfg.hook_name, self.sae.cfg.d_in, self.sae.cfg.d_sae) != ('hook_layers.11', 1024, 4096):
            raise ValueError('SAE hook or dimensions do not match registered evaluator')
        classes = load_architecture(ROOT / '.external/MEva/training/hybrid/train_cnn.py')
        self.cnn = classes['SAEOnlyModel'](classes['SAETower'](sae_dim=4096)).to(device).eval()
        checkpoint = RUNTIME / 'models/pooled/small/f03_sae_only_cnn_clean_best.pth'
        self.cnn.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        from lib.repro.audio_window import load_audio_mono, uniform_pool_time_first
        self.load_audio = load_audio_mono
        self.pool = uniform_pool_time_first

    def score(self, audio):
        import numpy as np
        import torch
        wave = self.load_audio(str(audio)).squeeze(0)
        if wave.numel() < 3200 or not torch.isfinite(wave).all():
            raise ValueError('Audio is too short or contains nonfinite samples')
        # Official chunked extraction; preserve full rater-aligned clip, up to 360 s.
        if wave.numel() > 360 * 32000:
            raise ValueError('Audio exceeds registered 360 s budget; no silent truncation')
        parts = []
        with torch.inference_mode():
            for start in range(0, wave.numel(), 30 * 32000):
                chunk = wave[start:start + 30 * 32000]
                if chunk.numel() < 3200:
                    raise ValueError('Final chunk shorter than 0.1 s; requires explicit protocol decision')
                _, cache = self.musicgen.run_with_cache(chunk.unsqueeze(0).to(self.device),
                                                        names_filter=[self.sae.cfg.hook_name])
                z = self.sae.encode(cache[self.sae.cfg.hook_name]).squeeze(0).float().cpu().numpy()
                if z.ndim != 2 or z.shape[1] != 4096 or not np.isfinite(z).all():
                    raise ValueError('Invalid SAE feature tensor')
                if wave.numel() <= 30 * 32000:
                    # Upstream short-clip batch path strips delay-pattern tail.
                    z = z[:min(int(wave.numel() / 32000 * 50), 1500)]
                parts.append(z)
            features = np.concatenate(parts, axis=0)
            if len(features) > 1500:
                features = self.pool(features, 1500)
            length = len(features)
            padded = np.zeros((1500, 4096), dtype=np.float32)
            padded[:length] = features
            score = self.cnn(None, torch.from_numpy(padded).unsqueeze(0).to(self.device),
                             sae_len=torch.tensor([length], device=self.device)).item()
        if not np.isfinite(score):
            raise ValueError('Nonfinite model score')
        return {'meva_raw': score, 'sae_frames': length, 'duration_sec': wave.numel() / 32000,
                'status': 'scored', 'protocol': 'f03_small_hook11_short_native_long_chunk30_pool1500_masked_raw'}
