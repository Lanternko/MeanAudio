#!/usr/bin/env python
"""179 generation: six arms on checkpoint 114 (MeanFlow 25 steps, seed 42, fp32, NoQ/NoMask).

A0-A3 run eval.py verbatim. A4/A5 (and the lambda=0 consistency cell) monkeypatch
meanaudio.eval_utils.generate_mf with a four-condition CFG combination:

    u = u(neg) + 3*(u(c) - u(neg)) + lam*(u(c_dark) - u(c_bright))

networks.py / eval.py / mc_mf25_eval.sh are never edited: eval.py runs through
runpy inside this process with the module function swapped around it.

  python scripts/eval/muq_gen_steer_179_gen.py --out <root> --tsv <tsv> --arms A0,A1
"""
import argparse
import logging
import os
import runpy
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import muq_gen_steer_179_common as C  # noqa: E402


def make_steer_generate(lam, dark_suffix, bright_suffix):
    """Replacement for meanaudio.eval_utils.generate_mf (value-identical tail)."""
    import torch

    def generate_mf(text, *, negative_text=None, feature_utils, net, mf, rng,
                    cfg_strength, q_level=0, use_text_attention_mask=True):
        cfg = float(cfg_strength)
        device = feature_utils.device
        dtype = feature_utils.dtype
        bs = len(text)

        text_features, text_features_c, text_attention_mask = feature_utils.encode_text(
            text, return_attention_mask=True)
        if not use_text_attention_mask:
            text_attention_mask = None

        dark_text = [f'{t}, {dark_suffix}' for t in text]
        bright_text = [f'{t}, {bright_suffix}' for t in text]
        dark_f, dark_c, dark_m = feature_utils.encode_text(dark_text, return_attention_mask=True)
        bright_f, bright_c, bright_m = feature_utils.encode_text(bright_text, return_attention_mask=True)
        if not use_text_attention_mask:
            dark_m = bright_m = None

        if negative_text is not None:
            assert len(negative_text) == bs
            negative_text_features = feature_utils.encode_text(negative_text, return_attention_mask=True)
            if not use_text_attention_mask:
                negative_text_features = negative_text_features[:2]
        else:
            negative_text_features = net.get_empty_string_sequence(bs)

        x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                         device=device, dtype=dtype, generator=rng)
        cond_c = net.preprocess_conditions(text_features, text_features_c, text_attention_mask)
        cond_d = net.preprocess_conditions(dark_f, dark_c, dark_m)
        cond_b = net.preprocess_conditions(bright_f, bright_c, bright_m)
        cond_neg = net.get_empty_conditions(
            bs, negative_text_features=negative_text_features if negative_text is not None else None)
        q = torch.full((bs,), q_level, dtype=torch.long, device=device)

        def ode(t, r, x):
            tt = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
            rr = r * torch.ones(len(x), device=x.device, dtype=x.dtype)
            # same call order and expression as MeanAudio.ode_wrapper, so lam=0 is
            # bit-identical to eval.py (the reordered a_neg + cfg*(a_c - a_neg) is not)
            a_c = net.predict_flow(x, tt, rr, cond_c, q)
            a_neg = net.predict_flow(x, tt, rr, cond_neg, q)
            out = cfg * a_c + (1 - cfg) * a_neg
            if lam != 0.0:
                a_d = net.predict_flow(x, tt, rr, cond_d, q)
                a_b = net.predict_flow(x, tt, rr, cond_b, q)
                out = out + lam * (a_d - a_b)
            return out

        x1 = mf.to_data(ode, x0)
        x1 = net.unnormalize(x1)
        spec = feature_utils.decode(x1)
        return feature_utils.vocode(spec)

    return generate_mf


@contextmanager
def steer_patch(lam):
    import meanaudio.eval_utils as EU
    original = EU.generate_mf
    EU.generate_mf = make_steer_generate(lam, C.DARK, C.BRIGHT_VEC)
    try:
        yield
    finally:
        EU.generate_mf = original


def expected_clips(tsv):
    return len(C.read_rows(tsv))


def generate_arm(arm, spec, tsv, out_dir, ckpt):
    out_dir = Path(out_dir)
    want = expected_clips(tsv)
    if out_dir.is_dir() and len(list(out_dir.glob('*.flac'))) >= want:
        return {'arm': arm, 'clips': len(list(out_dir.glob('*.flac'))), 'skipped': True}
    if out_dir.exists():
        # eval.py seeds one RNG for the whole run and does not draw noise for
        # clips it skips, so a partial directory cannot be topped up.
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    argv = ['eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
            '--output', str(out_dir), '--tsv', str(tsv), '--use_meanflow',
            '--num_steps', '25', '--cfg_strength', str(C.CFG),
            '--no_text_attention_mask', '--encoder_name', 't5_clap',
            '--text_c_dim', '512', '--seed', str(C.SEED), '--full_precision', '--no_q']
    if spec['neg']:
        argv += ['--negative_prompt', spec['neg']]
    if spec['suffix']:
        argv += ['--prompt_suffix', spec['suffix']]

    saved_argv, saved_cwd = sys.argv, os.getcwd()
    os.chdir(ROOT)
    sys.argv = argv
    # eval.py installs a log handler at import; runpy re-executes it once per arm,
    # so without this every line would be printed once per arm run so far.
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    root_logger.handlers = []
    try:
        with steer_patch(spec['lam']) if spec['kind'] == 'steer' else _nullcontext():
            runpy.run_path(str(ROOT / 'eval.py'), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(f'[FAIL] eval.py exited {exc.code} for arm {arm}')
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        logging.getLogger().handlers = saved_handlers
    n = len(list(out_dir.glob('*.flac')))
    if n != want:
        raise SystemExit(f'[FAIL] arm {arm}: {n}/{want} clips generated')
    return {'arm': arm, 'clips': n, 'skipped': False}


@contextmanager
def _nullcontext():
    yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='arms root (one subdir per arm)')
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--ckpt', default=str(C.CKPT))
    ap.add_argument('--arms', default=','.join(C.FULL_ARMS))
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    for arm in arms:
        if arm not in C.ARMS:
            raise SystemExit(f'[FAIL] unknown arm {arm}')
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    C.write_json(root / 'gen_manifest.json', {
        'arms': {k: C.ARMS[k] for k in arms},
        'tsv': str(Path(a.tsv).resolve()), 'tsv_sha256': C.sha256(a.tsv),
        'ckpt': str(Path(a.ckpt).resolve()), 'ckpt_sha256': C.sha256(a.ckpt),
        'cfg': C.CFG, 'seed': C.SEED, 'steps': 25})
    for arm in arms:
        rec = generate_arm(arm, C.ARMS[arm], Path(a.tsv), root / arm / 'audio', Path(a.ckpt))
        print(f'[gen] {rec}', flush=True)


if __name__ == '__main__':
    main()
