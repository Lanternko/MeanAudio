#!/usr/bin/env python
"""Render one trial: run eval.py verbatim with an optional guidance/time-grid patch.

Canonical argv is identical to scripts/eval/mc_mf25_eval.sh for the unpatched
path (same flags, same order-sensitive defaults: no --negative_prompt at cfg 0).
The patch only replaces MeanAudio.ode_wrapper / MeanFlow.run_t0_to_t1 for the
duration of the run (see patch_meanflow.py).

    render.py --out DIR --tsv TSV --checkpoint CKPT --configs rounds/r1.json --config-id F2_short
    render.py --out DIR --tsv TSV --checkpoint CKPT --num-steps 25 --cfg 3.0 \
              --prompt-id stock --seed 42 [--interval 0.667:1.0] [--schedule ramp_up]
              [--grid cos] [--geometry adg --adg-gamma 0.5]
"""
import argparse
import logging
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT)):                # works under `python -I` from guest.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                                    # noqa: E402
import patch_meanflow                            # noqa: E402

assert common.ROOT == ROOT


def parse_interval(text):
    lo, hi = (float(x) for x in text.split(':'))
    return [lo, hi]


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='raw audio directory (created)')
    p.add_argument('--tsv', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--configs', help='rounds/rN.json (with --config-id)')
    p.add_argument('--config-id')
    p.add_argument('--num-steps', type=int, default=25)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--prompt-id', default='stock')
    p.add_argument('--prompt-text', help='override the prereg prompt text')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--interval', type=parse_interval)
    p.add_argument('--schedule')
    p.add_argument('--grid', default='uniform')
    p.add_argument('--geometry', default='vanilla')
    p.add_argument('--adg-gamma', type=float, default=0.0)
    p.add_argument('--apg-eta', type=float, default=0.0)
    return p


def resolve(args):
    """CLI/config -> (config dict, prompt text)."""
    if args.configs:
        plan = common.load_plan(args.configs)
        if not args.config_id:
            raise SystemExit('[FAIL] --configs needs --config-id')
        cfg = next((c for c in plan['configs'] if c['id'] == args.config_id), None)
        if cfg is None:
            raise SystemExit(f'[FAIL] config {args.config_id} not in {args.configs}')
        cfg = dict(cfg)
        if args.prompt_text:
            cfg['prompt_text'] = args.prompt_text
        return cfg, prompt_text(cfg, args)
    cfg = {
        'id': args.config_id or 'adhoc',
        'cfg': args.cfg,
        'prompt': args.prompt_id,
        'interval': args.interval,
        'schedule': args.schedule,
        'geometry': args.geometry,
        'adg_gamma': args.adg_gamma,
        'apg_eta': args.apg_eta,
        'steps': args.num_steps,
        'grid': args.grid,
    }
    if args.prompt_text:
        cfg['prompt_text'] = args.prompt_text
    return cfg, prompt_text(cfg, args)


def prompt_text(cfg, args):
    if cfg.get('prompt_text'):
        return str(cfg['prompt_text'])
    table = common.prompts()
    name = cfg.get('prompt', 'stock')
    if name not in table:
        raise SystemExit(f'[FAIL] unknown prompt id {name!r}')
    return table[name]


def eval_argv(out_audio, tsv, ckpt, steps, cfg_strength, neg, seed):
    argv = [
        'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
        '--output', str(out_audio), '--tsv', str(tsv), '--use_meanflow',
        '--num_steps', str(steps), '--cfg_strength', str(cfg_strength),
    ]
    if cfg_strength >= 1.0 and neg:
        argv += ['--negative_prompt', neg]
    argv += [
        '--no_text_attention_mask', '--encoder_name', 't5_clap', '--text_c_dim', '512',
        '--seed', str(seed), '--full_precision', '--no_q',
    ]
    return argv


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg, neg = resolve(args)
    out_audio = Path(args.out)
    out_audio.mkdir(parents=True, exist_ok=True)
    spec = patch_meanflow.spec_from_config(cfg, steps=args.num_steps, grid=args.grid)
    steps = spec['steps']

    logs = logging.getLogger()
    saved_handlers = logs.handlers[:]
    saved_argv, saved_cwd = sys.argv[:], os.getcwd()
    argv_eval = eval_argv(out_audio, args.tsv, args.checkpoint, steps,
                          float(cfg.get('cfg', 3.0)), neg, args.seed)
    print(f'[render] config={cfg.get("id")} patch={spec["geometry"]} '
          f'interval={spec["interval"]} schedule={spec["schedule"]} grid={spec["grid"]} '
          f'steps={steps} cfg={cfg.get("cfg")}', flush=True)
    print(f'[render] argv={" ".join(argv_eval)}', flush=True)
    os.chdir(ROOT)
    sys.argv = argv_eval
    logs.handlers = []
    try:
        with patch_meanflow.patch(spec):
            runpy.run_path(str(ROOT / 'eval.py'), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(exc.code)
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        logs.handlers = saved_handlers
    n = len(list(out_audio.glob('*.flac')))
    print(f'[render] done: {n} flac in {out_audio}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
