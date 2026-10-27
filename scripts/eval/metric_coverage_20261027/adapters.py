#!/usr/bin/env python3
"""Backend adapters for score_coverage.py.  Models are imported, never re-implemented.

Every adapter exposes
  cols         score columns (or embedding columns for the reference-based rows)
  batch        clips per call
  score_batch(items) -> list[dict | Exception]   (same order as items; 'path' is a file)

Model code reused (§2.3): newmetrics_{muqeval,cmirm,pam}, bench_r2/score_r2.make_singmos,
metric_win_tie_126d.load_songeval/score_songeval_batch, metric_win_tie_133 embed helpers,
mir_metrics.score_mir, eval_metrics.score_aes (not scored by this driver — AES is 6/6 done).
"""
import base64
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path('/home/kojiek/MeanAudio')
# bench_r2 keeps its s3prl copy next to the r2 datasets (bench_datasets.D.ROOT), not in the repo
PYLIB_SINGMOS = Path('/mnt/seagate/benchmarks_r2/pylib_singmos')
sys.path.insert(0, str(REPO / 'scripts/eval'))
sys.path.insert(0, str(REPO / 'scripts/eval/bench_r2'))

EMB_SR = 24000
EMB_BATCH = 8


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _b64(v):
    return base64.b64encode(np.asarray(v, dtype=np.float16).tobytes()).decode()


def _unb64(s, dim):
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)[:dim]


def cosim(a, b, dim=None):
    """cosine of two float16 base64 embeddings (both already L2-normalised)."""
    import numpy as np
    x = np.frombuffer(base64.b64decode(a), dtype=np.float16).astype(np.float32)
    y = np.frombuffer(base64.b64decode(b), dtype=np.float16).astype(np.float32)
    if dim:
        x, y = x[:dim], y[:dim]
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-8))


class PerClip:
    """Wraps f(path) -> dict for models without batching."""
    cols = None
    batch = 8

    def __init__(self):
        self.load()

    def load(self):
        pass

    def one(self, path):
        raise NotImplementedError

    def score_batch(self, items):
        out = []
        for it in items:
            try:
                out.append(dict(self.one(it['path'])))
            except Exception as e:  # per-clip failure, never silently dropped
                out.append(RuntimeError(f'{type(e).__name__}: {str(e)[:200]}'))
        return out


class MuQEval(PerClip):
    def __init__(self, variant):
        self.variant = variant
        self.cols = ['mi', 'ta']
        super().__init__()

    def load(self):
        from newmetrics_common import setup_env
        setup_env()
        import newmetrics_muqeval  # noqa: F401  (fail early on a broken env)
        self.fn = newmetrics_muqeval.score_muqeval

    def one(self, path):
        r = self.fn(path, None, variant=self.variant, device=_device())
        return {'mi': r['MI'], 'ta': r['TA']}


class CmiRM(PerClip):
    cols = ['quality', 'alignment']

    def load(self):
        # upstream calls HfApi.model_info -> needs the hub even with warm weights (§2.3)
        from newmetrics_common import setup_env
        setup_env(offline=False)
        import newmetrics_cmirm
        self.fn = newmetrics_cmirm.score_cmirm

    def one(self, path):
        r = self.fn(path, None, prompt_mode='empty', device=_device())
        return {'quality': r['quality'], 'alignment': r['alignment']}


class Pam(PerClip):
    cols = ['pam']

    def load(self):
        from newmetrics_common import setup_env
        setup_env()
        import newmetrics_pam
        self.fn = newmetrics_pam.score_pam

    def one(self, path):
        return {'pam': self.fn(path, None, device=_device())['pam']}


class SingMos(PerClip):
    cols = ['singmos']

    def load(self):
        import types
        import torch
        import torchaudio
        sys.path.insert(0, str(PYLIB_SINGMOS))
        # v1.1.2 imports torchaudio's removed sox API at import time; the predictor never uses it
        torchaudio.set_audio_backend = lambda *a, **k: None
        se = types.ModuleType('torchaudio.sox_effects')
        se.apply_effects_tensor = se.apply_effects_file = None
        sys.modules['torchaudio.sox_effects'] = se
        torchaudio.sox_effects = se
        repo = Path.home() / '.cache/torch/hub/South-Twilight_SingMOS_v1.1.2'
        self.m = torch.hub.load(str(repo), 'singmos_pro', source='local',
                                trust_repo=True).cuda().eval()

    def one(self, path):
        import soundfile as sf
        import torch
        import torchaudio
        a, sr = sf.read(path, dtype='float32', always_2d=True)
        t = torch.from_numpy(a.mean(axis=1))
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        t = t.view(1, 1, -1).cuda()
        with torch.no_grad():
            v = self.m(t, torch.tensor([t.shape[-1]]).cuda())
        return {'singmos': float(np.asarray(v.float().cpu()).reshape(-1)[0])}


class SongEval:
    cols = ['se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus']
    batch = 8

    def __init__(self):
        import torch
        import metric_win_tie_126d as S
        self.S = S
        self.dev = _device()
        self.muq, self.gen = S.load_songeval(self.dev)

    def fixed30(self, path):
        w = self.S.load_24k(path)
        n = 30 * self.S.SE_SR
        if len(w) >= n:
            s = (len(w) - n) // 2
            return w[s:s + n]
        return np.concatenate([w, np.zeros(n - len(w), dtype=np.float32)])

    def score_batch(self, items):
        out = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            try:
                wavs = [self.fixed30(it['path']) for it in chunk]
                sc = self.S.score_songeval_batch(wavs, self.muq, self.gen, self.dev)
                for s in sc:
                    out.append({c: float(v) for c, v in zip(self.cols, s)})
            except Exception as e:
                out += [RuntimeError(f'{type(e).__name__}: {str(e)[:200]}')] * len(chunk)
        return out


class Embed:
    """MERT-v1-95M or MuQ-large mean-pooled L2-normalised embedding (24 kHz, whole clip)."""
    batch = EMB_BATCH

    def __init__(self, which):
        import torch
        import metric_win_tie_133 as M
        self.which, self.M, self.dev = which, M, _device()
        if which == 'mert':
            self.cols = ['mert_emb']
            self.feat, self.mdl = M.load_mert(self.dev)
        else:
            self.cols = ['muq_emb']
            self.mdl = M.load_muq(self.dev)

    def score_batch(self, items):
        out = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            try:
                wavs = [self.M.load_24k(it['path']) for it in chunk]
                if self.which == 'mert':
                    emb = self.M.embed_mert_batch([w.tolist() for w in wavs], self.feat, self.mdl,
                                                  self.dev)
                    out += [{'mert_emb': _b64(e)} for e in emb]
                else:
                    emb = self.M.embed_muq_batch(wavs, self.mdl, self.dev)
                    out += [{'muq_emb': _b64(e)} for e in emb]
            except Exception as e:
                out += [RuntimeError(f'{type(e).__name__}: {str(e)[:200]}')] * len(chunk)
        return out


class Mir:
    """CPU: mir_metrics.score_mir over a whole shard (worker subprocesses per env)."""
    cols = ['pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'chroma_entropy']
    batch = 2000

    def __init__(self):
        import mir_metrics
        self.fn = mir_metrics.score_mir

    def score_batch(self, items):
        paths = [it['path'] for it in items]
        try:
            per = self.fn(paths)
        except Exception as e:
            return [RuntimeError(f'{type(e).__name__}: {str(e)[:200]}')] * len(items)
        out = []
        for p in paths:
            r = per.get(p) or {}
            out.append({c: float(r[c]) for c in self.cols if c in r} or RuntimeError('mir missing'))
        return out


_LOAD_LOCK = threading.Lock()


def build(model):
    with _LOAD_LOCK:
        if model == 'muqeval_A1':
            return MuQEval('A1')
        if model == 'muqeval_A3a':
            return MuQEval('A3a')
        if model == 'cmirm':
            return CmiRM()
        if model == 'pam':
            return Pam()
        if model == 'singmos':
            return SingMos()
        if model == 'songeval':
            return SongEval()
        if model == 'mert':
            return Embed('mert')
        if model == 'muq':
            return Embed('muq')
        if model == 'mir':
            return Mir()
    raise SystemExit(f'unknown model {model}')
