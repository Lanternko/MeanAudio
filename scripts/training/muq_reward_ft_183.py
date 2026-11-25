#!/usr/bin/env python
"""183 arm R: differentiable MuQ-Eval A1 reward fine-tune of a MeanAudio S2 checkpoint.

loss = L_MF(batch) + lambda * (-A1_MI(1-NFE CFG0 sample of B_R batch captions))
MuQ is only ever called on *training* captions (Jamendo slot0clean corpus) and on a
track-disjoint Jamendo validation slice; MusicCaps test prompts are never touched.

Modes
  --mode gate   G_R0: MI parity, decode parity, gradient finite / non-zero, finite
                difference, peak memory and s/step at B_R.
  --mode train  pilot or full run; validates every --val_every steps, saves the
                network state_dict at each validation point (pilot) or only at the end.

Deviations from the preregistered doc are listed in the 183 doc (implementation notes):
starts from the control's ema_final with EMA off; mean_flow.py is stage-2 patched in
memory (the shared file stays as is); DDP is bypassed (world size 1).

Run from ~/MeanAudio with ~/venvs/dac/bin/python.
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
import types
from pathlib import Path

for k, v in dict(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', MASTER_ADDR='127.0.0.1',
                 MASTER_PORT=str(29500 + os.getpid() % 1000)).items():
    os.environ.setdefault(k, v)

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path[:0] = [str(REPO), str(REPO / 'scripts/eval')]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torchaudio  # noqa: E402

# ── in-memory stage-2 patch of meanaudio.model.mean_flow ────────────────────────
import set_training_stage as STS  # noqa: E402

_src = Path(STS.MEAN_FLOW_PATH)
if not _src.is_absolute():
    _src = REPO / _src
_code = _src.read_text()
if STS.detect_current_stage(_code) != 2:
    _code, _n = STS.apply_stage(_code, 2)
assert STS.detect_current_stage(_code) == 2, 'stage-2 patch failed'
_spec = importlib.util.spec_from_loader('meanaudio.model.mean_flow', loader=None, origin=str(_src))
_mf_mod = importlib.util.module_from_spec(_spec)
_mf_mod.__file__ = str(_src)
import meanaudio.model  # noqa: E402,F401
sys.modules['meanaudio.model.mean_flow'] = _mf_mod
exec(compile(_code, str(_src), 'exec'), _mf_mod.__dict__)
meanaudio.model.mean_flow = _mf_mod

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import open_dict  # noqa: E402

from meanaudio.data.data_setup import setup_training_datasets  # noqa: E402
from meanaudio.model.sequence_config import CONFIG_16K  # noqa: E402
from meanaudio.runner_meanflow import RunnerMeanFlow  # noqa: E402
from meanaudio.utils.logger import TensorboardLogger  # noqa: E402

assert sys.modules['meanaudio.runner_meanflow'].MeanFlow is _mf_mod.MeanFlow

HOME = Path.home()
CONTROL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}'
INPUTS = HOME / 'exps_nvme/slot0clean_nmv2matched/arm_inputs'
TRAIN_TSV = INPUTS / 'phase8_caption2p0_slot0clean_nmv2matched_train.tsv'
VAL_SRC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv')
TARGET_LUFS, PEAK_CAP, SILENT_DBFS = -30.0, 0.999, -45.0


def common_overrides(seed, lr, exp_id):
    return [
        'data=meanaudio', 'lr_schedule_steps=[999999,999999]', '+use_q_conditioning=false',
        'batch_size=8', '+accumulation_steps=1', f'learning_rate={lr}', f'seed={seed}',
        'linear_warmup_steps=1', 'num_workers=4', '+use_rope=False', '+use_wandb=False',
        '+use_text_attention_mask=false', '++multi_cap=false', '++cap_index_fixed=0',
        f'data.AudioCaps_npz.tsv={TRAIN_TSV}',
        '++data.AudioCaps_npz.npz_dir=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz',
        f'++data.AudioCaps_npz.gt_cache={INPUTS}/cache_train.txt',
        f'++data.AudioCaps_npz.text_npz_dir={HOME}/text_overlays/slot0clean',
        '++data.AudioCaps_npz.require_text_overlay=true',
        'data.AudioCaps_val_npz.tsv=/mnt/HDD/kojiek/phase4_jamendo_data/_QUARANTINED_phase4_val.tsv',
        '++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4',
        '++data.AudioCaps_val_npz.gt_cache=null',
        'model=meanaudio_s', 'ema.enable=false', f'exp_id={exp_id}',
    ]


class NoDDP:
    """Stands in for the DDP wrapper: world size 1, two backwards per step."""

    def __init__(self, m):
        self.module = m

    def __call__(self, *a, **k):
        return self.module(*a, **k)

    def parameters(self):
        return self.module.parameters()


# ── MuQ-Eval with gradients ──────────────────────────────────────────────────────
def load_muq(variant):
    import newmetrics_common as NC
    NC.HF_HOME = HOME / '.cache/huggingface'
    from newmetrics_muqeval import _load
    m, _ = _load(variant, 'cuda')
    m.eval().requires_grad_(False)
    n = 0
    for mod in m.modules():
        if type(mod).__name__ == 'MuQModel':
            for name in ('preprocessing', 'normalize'):
                f = getattr(type(mod), name)
                if hasattr(f, '__wrapped__'):
                    setattr(mod, name, types.MethodType(f.__wrapped__, mod))
            n += 1
    assert n >= 1, 'no MuQModel found to un-no_grad'
    return m


def to24k(w16):
    w = torchaudio.functional.resample(w16, 16000, 24000)
    L = 240000
    if w.shape[-1] >= L:
        s = (w.shape[-1] - L) // 2
        return w[..., s:s + L]
    return torch.nn.functional.pad(w, (0, L - w.shape[-1]))


def muq_score(m, w16):
    """w16 [B, T] float32 @16k -> (MI [B], TA [B]) with A3a ordinal heads expected."""
    out = m(to24k(w16))
    res = []
    for k in ('MI', 'TA'):
        v = out[k].float()
        if v.dim() > 1 and v.shape[-1] > 1:
            v = (torch.softmax(v, -1) * torch.linspace(1, 5, v.shape[-1], device=v.device)).sum(-1)
        res.append(v.reshape(-1))
    return res


class Decoder:
    def __init__(self, trainer):
        self.vae = trainer.features.tod.vae
        self.voc = trainer.features.tod.vocoder.vocoder
        self.mean = trainer.network.module.latent_mean
        self.std = trainer.network.module.latent_std

    def latent_to_wav(self, x_norm):
        """normalized latent [B, L, C] -> waveform [B, T] float32 (differentiable)."""
        z = x_norm.float() * self.std + self.mean
        mel = self.vae.decode(z.transpose(1, 2))
        w = voc_forward(self.voc, mel)
        return w.reshape(w.shape[0], -1).float()


def voc_forward(voc, x):
    """BigVGAN.forward, same ops in the same order, with each upsample layer, AMP resblock and
    the post stage checkpointed when grad is on (2026-11-10: one fp32 10 s clip held every
    Snake activation for backward and OOMed at ~25 GiB). Values are identical to voc(x)."""
    if not torch.is_grad_enabled():
        return voc(x)
    ck = lambda f, *a: torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False)
    x = voc.conv_pre(x)
    for i in range(voc.num_upsamples):
        for up in voc.ups[i]:
            x = ck(up, x)
        xs = None
        for j in range(voc.num_kernels):
            y = ck(voc.resblocks[i * voc.num_kernels + j], x)
            xs = y if xs is None else xs + y
        x = xs / voc.num_kernels
    return ck(lambda t: torch.tanh(voc.conv_post(voc.activation_post(t))), x)


def sample_1nfe(net, noise, text_f, text_f_c):
    b = noise.shape[0]
    t = torch.ones(b, device=noise.device, dtype=noise.dtype)
    r = torch.zeros_like(t)
    u = net(latent=noise, text_f=text_f, text_f_c=text_f_c, r=r, t=t)
    return noise - u


def reward_mi(net, dec, muq, noise, text_f, text_f_c, muq_bf16=False, ckpt=True):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        x = sample_1nfe(net, noise, text_f, text_f_c)

    def chain(xi):
        with torch.amp.autocast('cuda', enabled=False):
            w = dec.latent_to_wav(xi)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=muq_bf16):
            return muq_score(muq, w)[0]

    mis = []
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        mis.append(torch.utils.checkpoint.checkpoint(chain, xi, use_reentrant=False) if ckpt else chain(xi))
    return torch.cat(mis), x


# ── validation ──────────────────────────────────────────────────────────────────
def build_val_tsv(out, n=1000):
    import csv
    train_tracks = set()
    with open(TRAIN_TSV, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            train_tracks.add(row['track_id'].split('/')[-1])
    rows = []
    with open(VAL_SRC_TSV, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['id'].split('_')[1] not in train_tracks:
                rows.append(row)
                if len(rows) == n:
                    break
    assert len(rows) == n, len(rows)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        for r in rows:
            w.writerow([r['id'], r['caption']])
    return rows


def lvl30(w):
    import pyloudnorm
    w = w.astype(np.float64)
    lufs = pyloudnorm.Meter(16000).integrated_loudness(w[:, None])
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    g = 0.0
    if math.isfinite(lufs) and peak > 0:
        g = TARGET_LUFS - lufs
        if peak * 10 ** (g / 20) > PEAK_CAP:
            g = 20 * math.log10(PEAK_CAP / peak)
    out = np.clip(w * 10 ** (g / 20), -1.0, 1.0)
    return out, (lufs if math.isfinite(lufs) else float('nan'))


def pcm16(w):
    return np.round(np.clip(w, -1.0, 1.0) * 32767.0) / 32767.0


@torch.no_grad()
def validate(trainer, dec, muqs, captions, seed, bs=16):
    net = trainer.network.module
    net.eval()
    g = torch.Generator(device='cuda').manual_seed(seed)
    acc = {f'{v}_{k}_{c}': [] for v in muqs for k in ('MI', 'TA') for c in ('raw', 'lvl30')}
    lufs_all, silent = [], 0
    for i in range(0, len(captions), bs):
        caps = captions[i:i + bs]
        tf, tfc = trainer.features.encode_text(caps)
        noise = torch.randn(len(caps), net.latent_seq_len, net.latent_dim, device='cuda', generator=g)
        x = sample_1nfe(net, noise, tf.float(), tfc.float())
        wav = dec.latent_to_wav(x).cpu().numpy()
        raw_b, lv_b = [], []
        for w in wav:
            w = pcm16(w)
            rms = 20 * math.log10(math.sqrt(float(np.mean(w ** 2))) + 1e-12)
            silent += rms < SILENT_DBFS
            lw, lu = lvl30(w)
            lufs_all.append(lu)
            raw_b.append(w)
            lv_b.append(pcm16(lw))
        for c, arr in (('raw', raw_b), ('lvl30', lv_b)):
            t = torch.from_numpy(np.stack(arr)).float().cuda()
            for v, m in muqs.items():
                mi, ta = muq_score(m, t)
                acc[f'{v}_MI_{c}'] += mi.tolist()
                acc[f'{v}_TA_{c}'] += ta.tolist()
    net.train()
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    lu = np.array(lufs_all)
    res['lufs_median'] = float(np.nanmedian(lu))
    res['silent_frac'] = silent / len(captions)
    res['n'] = len(captions)
    return res


# ── setup ───────────────────────────────────────────────────────────────────────
def setup(args):
    dist.init_process_group('nccl')
    seed = args.seed
    exp_id = args.exp_id
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(REPO / 'config'), version_base='1.3.2'):
        cfg = compose('train_config.yaml', overrides=common_overrides(seed, args.lr, exp_id))
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = CONFIG_16K.latent_seq_len
    log = TensorboardLogger(exp_id, str(run_dir), logging.getLogger(), is_rank0=True, enable_email=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    _, sampler, loader = setup_training_datasets(cfg)
    lm, ls = torch.load(cfg.data.latent_mean), torch.load(cfg.data.latent_std)
    trainer = RunnerMeanFlow(cfg, log=log, run_path=str(run_dir), for_training=True,
                             latent_mean=lm, latent_std=ls).enter_train()
    init = Path(args.init) if args.init else (
        REPO / 'exps' / f'{CONTROL.format(seed=seed)}_stage2_50000' /
        f'{CONTROL.format(seed=seed)}_stage2_50000_ema_final.pth')
    trainer.load_weights(str(init))
    trainer.network = NoDDP(trainer.network.module)
    trainer.features.requires_grad_(False)
    for g in trainer.optimizer.param_groups:
        g['lr'] = args.lr
    return cfg, trainer, sampler, loader, init


def batches(sampler, loader):
    ep = 0
    while True:
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(ep)
        for d in loader:
            yield d
        ep += 1


def grads_vec(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
                      for p in params])


# ── gate ────────────────────────────────────────────────────────────────────────
def run_gate(args):
    cfg, trainer, sampler, loader, init = setup(args)
    net = trainer.network.module
    dec = Decoder(trainer)
    a1 = load_muq('A1')
    rep = {'init': str(init), 'B_R': args.br}
    import newmetrics_muqeval as NM

    # 1. MI parity (our differentiable path vs score_muqeval on the same waveform) and
    #    decode parity (our decode vs FeaturesUtils.decode/vocode), 64 clips.
    it = batches(sampler, loader)
    dmi, dwav = [], []
    while len(dmi) < 64:
        d = next(it)
        tf, tfc = d['text_features'].cuda().float(), d['text_features_c'].cuda().float()
        noise = torch.randn(tf.shape[0], net.latent_seq_len, net.latent_dim, device='cuda')
        with torch.no_grad():
            x = sample_1nfe(net, noise, tf, tfc)
            w = dec.latent_to_wav(x)
            ref = trainer.features.vocode(trainer.features.decode(net.unnormalize(x.clone())))
            ref = ref.reshape(ref.shape[0], -1).float()
            mi = torch.cat([muq_score(a1, w[j:j + 1])[0] for j in range(w.shape[0])])  # per item, as in the reward path
        for j in range(w.shape[0]):
            dwav.append(float((w[j] - ref[j]).abs().max()))
            r = NM.score_muqeval(w[j].cpu().numpy(), sr=16000, variant='A1')['MI']
            dmi.append(abs(float(mi[j]) - r))
    rep['mi_parity_max_abs'] = max(dmi)
    rep['decode_parity_max_abs'] = max(dwav)
    rep['wav_len'] = int(w.shape[-1])
    print(json.dumps(rep, indent=2), flush=True)
    if args.parity_only:
        rep['gates'] = {'mi_parity': rep['mi_parity_max_abs'] <= 1e-3}
        (Path(args.out_dir) / 'gate_R0_parity.json').write_text(json.dumps(rep, indent=2))
        return

    # 2. gradient finite / non-zero, finite difference (fp32, fixed noise, reward only)
    d = next(it)
    tf, tfc = d['text_features'][:args.br].cuda().float(), d['text_features_c'][:args.br].cuda().float()
    g = torch.Generator(device='cuda').manual_seed(1)
    noise = torch.randn(args.br, net.latent_seq_len, net.latent_dim, device='cuda', generator=g)
    params = [p for p in net.parameters() if p.requires_grad]

    def fp32_mi():
        x = sample_1nfe(net, noise, tf, tfc)
        f = lambda xi: muq_score(a1, dec.latent_to_wav(xi))[0]
        return torch.cat([torch.utils.checkpoint.checkpoint(f, x[i:i + 1], use_reentrant=False)
                          for i in range(args.br)]).mean()

    trainer.optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=False):
        mi0 = fp32_mi()
        mi0.backward()
    gv = grads_vec(params)
    gn = float(gv.norm())
    rep['grad_finite'] = bool(torch.isfinite(gv).all())
    rep['grad_norm'] = gn
    rep['grad_nonzero_frac'] = float((gv != 0).float().mean())
    eps = args.fd_eps
    dirs = [p.grad.detach().float() / gn if p.grad is not None else torch.zeros_like(p) for p in params]
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        for p, u in zip(params, dirs):
            p.add_(u, alpha=eps)
        mp = float(fp32_mi())
        for p, u in zip(params, dirs):
            p.add_(u, alpha=-2 * eps)
        mm = float(fp32_mi())
        for p, u in zip(params, dirs):
            p.add_(u, alpha=eps)
    fd = (mp - mm) / (2 * eps)
    rep['fd_directional'] = fd
    rep['fd_rel_err'] = abs(fd - gn) / max(gn, 1e-12)
    trainer.optimizer.zero_grad(set_to_none=True)

    # 3. memory and s/step at B_R (full training step, lambda 1)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(args.gate_steps):
        torch.cuda.synchronize()
        t0 = time.time()
        train_step(trainer, dec, a1, next(it), args, lam=1.0, log_ratio=False)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    rep['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
    rep['s_per_step_median'] = float(np.median(times[2:] if len(times) > 4 else times))
    rep['gates'] = {
        'mi_parity': rep['mi_parity_max_abs'] <= 1e-3,
        'grad': rep['grad_finite'] and rep['grad_norm'] > 0,
        'fd': rep['fd_rel_err'] <= 0.1,
        'mem': rep['peak_mem_gb'] <= 30,
        'speed': rep['s_per_step_median'] <= 4.0,
    }
    rep['pass'] = all(rep['gates'].values())
    out = Path(args.out_dir) / f'gate_R0_br{args.br}.json'
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# ── training ────────────────────────────────────────────────────────────────────
def train_step(trainer, dec, a1, data, args, lam, log_ratio):
    net = trainer.network.module
    params = [p for p in net.parameters() if p.requires_grad]
    tf = data['text_features'].cuda(non_blocking=True)
    tfc = data['text_features_c'].cuda(non_blocking=True)
    tf_r, tfc_r = tf[:args.br].clone(), tfc[:args.br].clone()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, _, l_mf, _, _ = trainer.train_fn(tf, tfc, data['a_mean'].cuda(non_blocking=True),
                                            data['a_std'].cuda(non_blocking=True))
    trainer.scaler.scale(l_mf).backward()
    g_mf = grads_vec(params).clone() if log_ratio else None
    out = {'l_mf': float(l_mf)}
    if lam > 0:
        noise = torch.randn(args.br, net.latent_seq_len, net.latent_dim, device='cuda')
        mi, _ = reward_mi(net, dec, a1, noise, tf_r, tfc_r, muq_bf16=args.muq_bf16)
        trainer.scaler.scale(-lam * mi.mean()).backward()
        out['mi_train'] = float(mi.mean())
        if log_ratio:
            g_all = grads_vec(params)
            out['grad_ratio_reward_over_mf'] = float((g_all - g_mf).norm() / g_mf.norm().clamp_min(1e-12))
    trainer.scaler.unscale_(trainer.optimizer)
    out['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(params, trainer.clip_grad_norm))
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad(set_to_none=True)
    return out


def run_train(args):
    cfg, trainer, sampler, loader, init = setup(args)
    out_dir = Path(args.out_dir)
    dec = Decoder(trainer)
    a1 = load_muq('A1')
    muqs = {'A1': a1, 'A3a': load_muq('A3a')}
    val_tsv = out_dir / 'jamendo_val_trackdisjoint_1000.tsv'
    rows = build_val_tsv(val_tsv, args.val_n)
    caps = [r['caption'] for r in rows]
    hist_p = out_dir / 'val_history.jsonl'
    done = set()
    if hist_p.is_file():
        done = {json.loads(l)['step'] for l in hist_p.read_text().splitlines() if l.strip()}
    if 0 not in done:
        v = validate(trainer, dec, muqs, caps, args.val_seed)
        v.update(step=0, lam=args.lam)
        with open(hist_p, 'a') as f:
            f.write(json.dumps(v) + '\n')
        print('val', json.dumps(v), flush=True)
    it = batches(sampler, loader)
    t0 = time.time()
    with open(out_dir / 'train_log.jsonl', 'a') as flog:
        for step in range(1, args.steps + 1):
            rec = train_step(trainer, dec, a1, next(it), args, args.lam, log_ratio=(step % args.log_every == 0))
            if step % args.log_every == 0:
                rec.update(step=step, elapsed=time.time() - t0)
                flog.write(json.dumps(rec) + '\n')
                flog.flush()
                print(json.dumps(rec), flush=True)
            if not all(math.isfinite(v) for k, v in rec.items() if isinstance(v, float)):
                raise RuntimeError(f'non-finite at step {step}: {rec}')
            if step % args.val_every == 0 or step == args.steps:
                v = validate(trainer, dec, muqs, caps, args.val_seed)
                v.update(step=step, lam=args.lam)
                with open(hist_p, 'a') as f:
                    f.write(json.dumps(v) + '\n')
                print('val', json.dumps(v), flush=True)
                if args.save_each_val or step == args.steps:
                    torch.save(trainer.network.module.state_dict(), out_dir / f'net_step{step:06d}.pth')
    (out_dir / 'DONE').write_text(json.dumps({'steps': args.steps, 'lam': args.lam, 'init': str(init)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['gate', 'train'], required=True)
    p.add_argument('--seed', type=int, default=14159265)
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--br', type=int, default=4)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--val_every', type=int, default=500)
    p.add_argument('--val_n', type=int, default=1000)
    p.add_argument('--val_seed', type=int, default=183)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--save_each_val', action='store_true')
    p.add_argument('--muq_bf16', action='store_true')
    p.add_argument('--fd_eps', type=float, default=1e-3)
    p.add_argument('--gate_steps', type=int, default=12)
    p.add_argument('--init', default='')
    p.add_argument('--parity_only', action='store_true')
    p.add_argument('--exp_id', default='muq_reward_ft_183')
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f'args_{args.mode}.json').write_text(json.dumps(vars(args), indent=2))
    run_gate(args) if args.mode == 'gate' else run_train(args)


if __name__ == '__main__':
    main()
