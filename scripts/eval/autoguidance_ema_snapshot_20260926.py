#!/usr/bin/env python
"""080: autoguidance with earlier EMA snapshots of the same run as the "bad" model.

Autoguidance (Karras et al., arXiv 2406.02507) replaces the unconditional /
negative branch of CFG with a weaker version of the same model under the SAME
conditioning:

    pure : u = A + (w-1) * (A - A_bad)
    combo: u = [3*A + (1-3)*B_fid8] + (w-1) * (A - A_bad)   (stock CFG3+neg plus the AG term)

A     = conditional prediction of the final checkpoint (ema_final)
B     = fidelity8 negative branch of the final checkpoint (stock CFG3+neg cell)
A_bad = conditional prediction of an earlier EMA snapshot of the same run,
        with its own text projection of the same raw text features

The 075 follow-ups ruled out "push size" and "push away from programmatic defects"
as the reason the fidelity8 negative prompt helps. This asks whether a model-based
negative branch, which carries no text at all, reaches the text negative's PQ gain.

Base run: slot0clean_nmv2pair quarter (the 075 control), 3 training seeds.
Bad models: S2 EMA snapshots at 110k / 130k (MeanFlow, same objective; primary)
and S1 EMA snapshots at 30k / 100k (FluxAudio, instantaneous velocity, no r;
exploratory, the difference term then also carries the objective mismatch).

Rows are the first N rows of the standard MusicCaps TSV with the stock eval flags,
so every clip shares prompt and noise with the stock full cfg0 / cfg3_neg cells
of the same checkpoint (eval.py seeds one RNG per run; the first N rows draw the
same noise, verified bit-exact by d2_075_segment_cfg). The replication gate loads
ema_final itself as the "bad" model: A - A_bad is then exactly zero and the pure /
combo cells must reproduce the stock cfg0 / cfg3_neg audio sample-for-sample.

Stages
  gate  : pure(self,w2) vs stock cfg0, combo(self,w2) vs stock cfg3_neg, 256 rows, 0 mismatches
  pilot : s14159265, 1024 rows, 4 bad x w{1.5,2,3} pure + 2 combo (w2)
  B     : selected pure + selected combo on the other 2 seeds, 1024 rows (always)
  C     : selected pure (and combo) at 5521 rows on s14159265, only if it clears the
          registered gate on all 3 seeds

Usage: python autoguidance_ema_snapshot_20260926.py [--preflight | --validate-only]
"""
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402

ROOT = Path('/home/kojiek/MeanAudio')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/autoguidance_ema_snapshot_20260926')
CELLS = OUT / 'cells'
SUMMARY = OUT / 'summary.json'
EV = Path.home() / 'eval_output_nvme'
PY = sys.executable

SEEDS = (14159265, 27182818, 16180339)
PILOT_SEED = 14159265
# name: (stage dir, snapshot file (sigma_rel 0.05 = default_output_sigma), network kind)
BAD = {
    's2_110k': ('stage2_50000', '0.110000.pt', 'meanflow'),
    's2_130k': ('stage2_50000', '0.130000.pt', 'meanflow'),
    's1_100k': ('stage1_100000', '0.100000.pt', 'flux'),
    's1_30k': ('stage1_100000', '0.30000.pt', 'flux'),
}
PRIMARY_BAD = ('s2_110k', 's2_130k')
WEIGHTS = (1.5, 2.0, 3.0)
COMBO_W = 2.0
COMBO_BAD = ('s2_110k', 's1_100k')
CFG_NEG = 3.0
GATE_ROWS, PILOT_ROWS, FULL_ROWS = 256, 1024, 5521
CLAP_FLOOR = -0.005
BOOT_N, BOOT_SEED = 10000, 20260926
HARD_STOP_FREE = 10_000_000_000
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap')
IMMUTABLE = {
    'eval.py': 'ba66c66b2ca3b7db0a698338932f6ee474208c1302c4592724f1955f3ccb2339',
    'meanaudio/model/networks.py': '5970fd615640c3d5a2b38aa025c3f3f26dee3412f5b8de810414c16d732cbe69',
    'meanaudio/model/mean_flow.py': 'ac8c6239612d7bcf26d7de6f2fe4eacdf8a81d91e1c711d93883db3934d1bee2',
    'scripts/eval/eval_metrics.py': '47406ee5bf30c837733a00be306e813d8c27301a2a8ea30de9f8f28b1dfec67d',
    'scripts/eval/level_match_rescore.py': '133ac816e3de21e732effd5a8f75a067029868aa589fa3f4d0c64ede4ca88714',
    'scripts/eval/guidance_geometry_adg_apg_20260922.py': '3a0bb4ff1ec7021855052dc698ef43ac0a416886f558247d40effb65f5888e15',
}
STATE = {'bad': None, 'kind': None, 'cobj': None, 'cbad': None}
GEO = defaultdict(list)


# ── paths ────────────────────────────────────────────────
def exp(seed, stage):
    return f'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}_{stage}'


def good_ckpt(seed):
    e = exp(seed, 'stage2_50000')
    return ROOT / 'exps' / e / f'{e}_ema_final.pth'


def bad_path(seed, bad):
    if bad == 'self':
        return good_ckpt(seed)
    stage, f, _ = BAD[bad]
    return ROOT / 'exps' / exp(seed, stage) / 'ema_ckpts' / f


def bad_kind(bad):
    return 'meanflow' if bad == 'self' else BAD[bad][2]


def stock_name(seed, c):
    return f'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}_mc_mf25_{c}'


def stock_per_clip(seed, c, matched):
    name = stock_name(seed, c)
    if matched:
        d = EV / 'd2_075_lvl30' / f'{name}_lvl30'
        if not d.exists():
            d = EV / f'{name}_lvl30'
    else:
        d = EV / name
    return next(d.glob('*/per_clip.tsv'))


def cell(seed, bad, w, neg, rows):
    return {'name': f's{seed}__{bad}__w{w}__{"combo" if neg else "pure"}__n{rows}',
            'seed': seed, 'bad': bad, 'w': w, 'neg': neg, 'rows': rows}


def cell_dir(c):
    return CELLS / c['name']


def smoke_tsv(rows):
    return OUT / f'_smoke_{rows}.tsv'


# ── model patching ───────────────────────────────────────
def load_bad(good, bad, seed):
    """Build the bad network and load a snapshot with an explicit key-set check.

    MeanAudio.load_weights is strict=False, so a FluxAudio snapshot (no r_embed)
    loaded into a copy of the good net would silently keep the good r_embed.
    """
    import torch
    from meanaudio.model.networks import fluxaudio_s
    sd = torch.load(bad_path(seed, bad), map_location='cpu', weights_only=False)
    sd = {k.removeprefix('ema_model.'): v for k, v in sd.items() if k != '_extra_state'}
    dtype = good.latent_mean.dtype
    if bad_kind(bad) == 'meanflow':
        net = copy.deepcopy(good)
    else:
        net = fluxaudio_s(use_rope=False, text_c_dim=512).to(good.device, dtype).eval()
        net.update_seq_lengths(good.latent_seq_len)
    recomputed = {'t_embed.freqs', 'r_embed.freqs', 'latent_rot', 'text_rot'}
    sd = {k: v for k, v in sd.items() if k not in recomputed}
    expected = set(net.state_dict()) - recomputed
    missing, unexpected = expected - set(sd), set(sd) - expected
    if missing or unexpected:
        raise SystemExit(f'[FAIL] bad model {bad} s{seed}: missing={sorted(missing)[:8]} '
                         f'unexpected={sorted(unexpected)[:8]}')
    net.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()},
                        strict=False)
    for buf in ('latent_rot', 'text_rot'):
        x, y = getattr(net, buf), getattr(good, buf)       # None when use_rope=False
        if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
            raise SystemExit(f'[FAIL] bad model {bad}: {buf} differs from the good net')
    net.requires_grad_(False)
    return net


def patch_preprocess():
    """Keep the raw text features on every PreprocessedConditions so the bad model
    can project them with its own text_input_proj / text_cond_proj."""
    from meanaudio.model.networks import MeanAudio
    orig = MeanAudio.preprocess_conditions

    def preprocess_conditions(self, text_f, text_f_c, text_attention_mask=None):
        out = orig(self, text_f, text_f_c, text_attention_mask)
        out._raw = (text_f, text_f_c, text_attention_mask)
        return out

    MeanAudio.preprocess_conditions = preprocess_conditions
    return MeanAudio, orig


def make_wrapper(spec):
    import torch

    def third(t):
        return 't_hi' if t > 2 / 3 else ('t_mid' if t > 1 / 3 else 't_lo')

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        if STATE['bad'] is None:
            STATE['bad'], STATE['kind'] = load_bad(self, spec['bad'], spec['seed']), bad_kind(spec['bad'])
        bad = STATE['bad']
        if STATE['cobj'] is not conditions:      # new batch; hold a reference so ids cannot recycle
            STATE['cobj'] = conditions
            STATE['cbad'] = bad.preprocess_conditions(*conditions._raw)
        a = self.predict_flow(latent, t, r, conditions, q)
        if STATE['kind'] == 'flux':
            a_bad = bad.predict_flow(latent, t, STATE['cbad'], q)
        else:
            a_bad = bad.predict_flow(latent, t, r, STATE['cbad'], q)
        base = a
        if spec['neg']:
            b = self.predict_flow(latent, t, r, empty_conditions, q)
            base = CFG_NEG * a + (1 - CFG_NEG) * b          # the stock CFG3+neg expression
        d = a - a_bad
        dims = tuple(range(1, a.ndim))
        na, nd = a.norm(dim=dims), d.norm(dim=dims)
        cos = (a * a_bad).sum(dim=dims) / (na * a_bad.norm(dim=dims) + 1e-12)
        key = third(float(t[0]))
        GEO[f'{key}_rel'] += (nd / (na + 1e-12)).tolist()
        GEO[f'{key}_cos'] += cos.tolist()
        return base + (spec['w'] - 1.0) * d

    return ode_wrapper


def generate(c):
    from meanaudio.model.networks import MeanAudio
    d = cell_dir(c)
    G.OUT, G.CKPT, G.AUDIO_ROOT = OUT, good_ckpt(c['seed']), d
    STATE.update(bad=None, kind=None, cobj=None, cbad=None)
    GEO.clear()
    cls, orig_pp = patch_preprocess()
    original = MeanAudio.ode_wrapper
    MeanAudio.ode_wrapper = make_wrapper(c)
    try:
        G.generate({'name': 'audio', 'family': 'N8', 'cfg': CFG_NEG, 'geometry': 'vanilla',
                    'tsv': str(G.MC_TSV), 'stage': 'diag'}, limit=c['rows'])
    finally:
        MeanAudio.ode_wrapper = original
        cls.preprocess_conditions = orig_pp
        STATE.update(bad=None, cobj=None, cbad=None)
    n = len(list((d / 'audio').glob('*.flac')))
    if n != c['rows']:
        raise SystemExit(f'[FAIL] {c["name"]}: {n} clips, expected {c["rows"]}')
    return {k: float(np.mean(v)) for k, v in GEO.items()}


# ── scoring ──────────────────────────────────────────────
def per_clip(path):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def score(c):
    """Unmatched metrics + -30 LUFS matched metrics; matched audio is deleted after."""
    d = cell_dir(c)
    tsv = smoke_tsv(c['rows'])
    un = d / 'metrics' / 'per_clip.tsv'
    if not un.exists():
        subprocess.run([PY, str(ROOT / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(d / 'audio'),
                        '--tsv', str(tsv), '--exp_name', 'metrics', '--out_dir', str(d)], check=True)
    lvl = d.parent / f'{d.name}_lvl30'
    if not list(lvl.glob('*/per_clip.tsv')):
        subprocess.run([PY, str(ROOT / 'scripts/eval/level_match_rescore.py'), '--cell_dir', str(d),
                        '--tsv', str(tsv)], check=True)
    shutil.rmtree(lvl / 'audio', ignore_errors=True)
    return un, next(lvl.glob('*/per_clip.tsv'))


def boot(x, rng):
    x = np.asarray(x, dtype=np.float64)
    b = rng.choice(x, (BOOT_N, len(x))).mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def contrasts(c, un_path, lvl_path):
    rng = np.random.default_rng(BOOT_SEED)
    un, lv = per_clip(un_path), per_clip(lvl_path)
    ids = sorted(un)
    out = {'n': len(ids)}
    ref = {k: (per_clip(stock_per_clip(c['seed'], k, False)), per_clip(stock_per_clip(c['seed'], k, True)))
           for k in ('cfg0', 'cfg3_neg')}
    for k, (s_un, s_lv) in ref.items():
        missing = [i for i in ids if i not in s_un or i not in s_lv or i not in lv]
        if missing:
            raise SystemExit(f'[FAIL] {c["name"]}: {len(missing)} ids missing from stock {k}')
        blk = {}
        for m in METRICS:
            blk[m] = boot([float(un[i][m]) - float(s_un[i][m]) for i in ids], rng)
            blk[f'{m}_lvl30'] = boot([float(lv[i][m]) - float(s_lv[i][m]) for i in ids], rng)
        for m in ('lufs', 'crest'):
            v = [float(un[i][m]) - float(s_un[i][m]) for i in ids]
            blk[m] = float(np.nanmean(v))
        blk['silent_n'] = int(sum(int(un[i]['silent']) for i in ids))
        blk['silent_n_stock'] = int(sum(int(s_un[i]['silent']) for i in ids))
        out[f'vs_{k}'] = blk
    # the text-negative gain on the same clips, for reading the AG deltas against
    s0u, s0l = ref['cfg0']
    s3u, s3l = ref['cfg3_neg']
    out['text_neg_gain'] = {m: boot([float(s3u[i][m]) - float(s0u[i][m]) for i in ids], rng) for m in METRICS}
    out['text_neg_gain'].update({f'{m}_lvl30': boot([float(s3l[i][m]) - float(s0l[i][m]) for i in ids], rng)
                                 for m in METRICS})
    return out


def run_cell(c):
    path = CELLS / f'{c["name"]}.json'
    if path.exists():
        return json.loads(path.read_text())
    free = shutil.disk_usage(OUT).free
    if free < HARD_STOP_FREE:
        raise SystemExit(f'[FAIL] disk hard stop: {free / 1e9:.1f} GB free')
    geo = generate(c)
    un, lvl = score(c)
    rec = {**c, 'checkpoint': str(good_ckpt(c['seed'])), 'bad_checkpoint': str(bad_path(c['seed'], c['bad'])),
           'geometry': geo, **contrasts(c, un, lvl)}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, indent=2) + '\n')
    tmp.replace(path)
    v0, v3 = rec['vs_cfg0'], rec['vs_cfg3_neg']
    print(f'[cell] {c["name"]}: vs cfg0 PQlvl {v0["PQ_lvl30"]["mean"]:+.3f} CLAP {v0["clap"]["mean"]:+.4f} '
          f'| vs cfg3neg PQlvl {v3["PQ_lvl30"]["mean"]:+.3f} CLAP {v3["clap"]["mean"]:+.4f} '
          f'| textneg PQlvl {rec["text_neg_gain"]["PQ_lvl30"]["mean"]:+.3f} | dLUFS {v0["lufs"]:+.2f} '
          f'| |A-Abad|/|A| {np.mean([geo[k] for k in geo if k.endswith("_rel")]):.3f}', flush=True)
    return rec


# ── gate ─────────────────────────────────────────────────
def replication_gate():
    import soundfile as sf
    path = CELLS / 'gate.json'
    if path.exists():
        return json.loads(path.read_text())
    res = {}
    for neg, ref in ((False, 'cfg0'), (True, 'cfg3_neg')):
        c = cell(PILOT_SEED, 'self', 2.0, neg, GATE_ROWS)
        generate(c)
        refdir = EV / stock_name(PILOT_SEED, ref) / 'audio'
        bad = 0
        files = sorted((cell_dir(c) / 'audio').glob('*.flac'))
        for f in files:
            a, _ = sf.read(f, dtype='int16')
            b, _ = sf.read(refdir / f.name, dtype='int16')
            bad += int(a.shape != b.shape or not np.array_equal(a, b))
        res[c['name']] = {'vs': ref, 'n': len(files), 'mismatch': bad}
        print(f'[gate] {c["name"]} vs stock {ref}: {bad}/{len(files)} clips differ', flush=True)
        shutil.rmtree(cell_dir(c))
    if any(v['mismatch'] for v in res.values()) or any(v['n'] != GATE_ROWS for v in res.values()):
        raise SystemExit(f'[FAIL] replication gate: {res}')
    path.write_text(json.dumps(res, indent=2) + '\n')
    return res


# ── selection ────────────────────────────────────────────
def select(recs, ref):
    ok = [r for r in recs if r[f'vs_{ref}']['clap']['mean'] >= CLAP_FLOOR]
    pool = ok or recs
    best = max(pool, key=lambda r: r[f'vs_{ref}']['PQ_lvl30']['mean'])
    return best, bool(ok)


def clears(r, ref):
    v = r[f'vs_{ref}']
    return v['PQ_lvl30']['ci95'][0] > 0 and v['clap']['mean'] >= CLAP_FLOOR


def pilot_cells():
    cs = [cell(PILOT_SEED, b, w, False, PILOT_ROWS) for b in BAD for w in WEIGHTS]
    cs += [cell(PILOT_SEED, b, COMBO_W, True, PILOT_ROWS) for b in COMBO_BAD]
    return cs


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(exist_ok=True)
    gate = replication_gate()
    pilot = [run_cell(c) for c in pilot_cells()]
    pure_sel, pure_ok = select([r for r in pilot if not r['neg']], 'cfg0')
    combo_sel, combo_ok = select([r for r in pilot if r['neg']], 'cfg3_neg')
    print(f'[select] pure {pure_sel["name"]} (clap floor met: {pure_ok}); '
          f'combo {combo_sel["name"]} (clap floor met: {combo_ok})', flush=True)
    stage_b = {}
    for sel in (pure_sel, combo_sel):
        reps = [sel] + [run_cell(cell(s, sel['bad'], sel['w'], sel['neg'], PILOT_ROWS))
                        for s in SEEDS if s != PILOT_SEED]
        stage_b[sel['name']] = [r['name'] for r in reps]
    pure_reps = [json.loads((CELLS / f'{n}.json').read_text()) for n in stage_b[pure_sel['name']]]
    combo_reps = [json.loads((CELLS / f'{n}.json').read_text()) for n in stage_b[combo_sel['name']]]
    go_pure = all(clears(r, 'cfg0') for r in pure_reps)
    go_combo = all(clears(r, 'cfg3_neg') for r in combo_reps)
    print(f'[stageC] pure clears on all seeds: {go_pure}; combo clears on all seeds: {go_combo}', flush=True)
    stage_c = []
    for sel, go in ((pure_sel, go_pure), (combo_sel, go_combo)):
        if go:
            stage_c.append(run_cell(cell(PILOT_SEED, sel['bad'], sel['w'], sel['neg'], FULL_ROWS))['name'])
    summary = {
        'experiment_id': 'autoguidance-ema-snapshot-20260926',
        'gate': gate,
        'pilot': [r['name'] for r in pilot],
        'selected': {'pure': pure_sel['name'], 'pure_clap_floor_met': pure_ok,
                     'combo': combo_sel['name'], 'combo_clap_floor_met': combo_ok},
        'stage_b': stage_b,
        'stage_c_gate': {'pure': go_pure, 'combo': go_combo},
        'stage_c': stage_c,
        'cells': {p.stem: json.loads(p.read_text()) for p in sorted(CELLS.glob('s*.json'))},
    }
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(SUMMARY)
    print(f'wrote {SUMMARY}', flush=True)


# ── preflight / postflight ───────────────────────────────
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    errs = []
    for rel, want in IMMUTABLE.items():
        if sha(ROOT / rel) != want:
            errs.append(f'{rel} changed since the contract was pinned')
    for s in SEEDS:
        if not good_ckpt(s).exists():
            errs.append(f'missing {good_ckpt(s)}')
        for b in BAD:
            if not bad_path(s, b).exists():
                errs.append(f'missing {bad_path(s, b)}')
        for k in ('cfg0', 'cfg3_neg'):
            for matched in (False, True):
                try:
                    n = len(per_clip(stock_per_clip(s, k, matched)))
                    if n < FULL_ROWS:
                        errs.append(f'stock {k} s{s} matched={matched} has {n} rows')
                except StopIteration:
                    errs.append(f'no stock per_clip for {k} s{s} matched={matched}')
        if not (EV / stock_name(s, 'cfg0') / 'audio').is_dir():
            errs.append(f'stock cfg0 audio missing for s{s}')
    for k in ('cfg0', 'cfg3_neg'):
        if not (EV / stock_name(PILOT_SEED, k) / 'audio').is_dir():
            errs.append(f'stock {k} audio missing for gate seed')
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < 2 * HARD_STOP_FREE:
        errs.append(f'less than {2 * HARD_STOP_FREE / 1e9:.0f} GB free at start')
    # key-set check of every bad snapshot on CPU against the network it will load into
    import torch
    from meanaudio.model.networks import fluxaudio_s, meanaudio_s
    nets = {'meanflow': set(meanaudio_s(use_rope=False, text_c_dim=512).state_dict()),
            'flux': set(fluxaudio_s(use_rope=False, text_c_dim=512).state_dict())}
    recomputed = {'t_embed.freqs', 'r_embed.freqs', 'latent_rot', 'text_rot'}
    for b, (_, _, kind) in BAD.items():
        sd = torch.load(bad_path(PILOT_SEED, b), map_location='cpu', weights_only=False)
        keys = {k.removeprefix('ema_model.') for k in sd if k != '_extra_state'} - recomputed
        exp_keys = nets[kind] - recomputed
        if keys != exp_keys:
            errs.append(f'{b}: key mismatch missing={sorted(exp_keys - keys)[:5]} '
                        f'unexpected={sorted(keys - exp_keys)[:5]}')
    if errs:
        print('\n'.join(f'[preflight] {e}' for e in errs))
        return 1
    print('[preflight] ok')
    return 0


def validate_only():
    if not SUMMARY.exists():
        print('[validate] no summary')
        return 1
    s = json.loads(SUMMARY.read_text())
    errs = []
    if any(v['mismatch'] for v in s['gate'].values()) or len(s['gate']) != 2:
        errs.append('gate not passed')
    expected = {c['name'] for c in pilot_cells()}
    if set(s['pilot']) != expected:
        errs.append('pilot cell set differs from the registered grid')
    for names in s['stage_b'].values():
        if len(names) != len(SEEDS):
            errs.append('stage B missing a seed')
    for name in [n for names in s['stage_b'].values() for n in names] + s['pilot'] + s['stage_c']:
        r = s['cells'].get(name)
        if r is None:
            errs.append(f'{name} missing')
            continue
        for k in ('vs_cfg0', 'vs_cfg3_neg'):
            if 'PQ_lvl30' not in r[k] or not np.isfinite(r[k]['PQ_lvl30']['mean']):
                errs.append(f'{name}: no matched read vs {k}')
    if errs:
        print('\n'.join(f'[validate] {e}' for e in errs))
        return 1
    print('[validate] ok')
    return 0


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if '--preflight' in sys.argv:
        raise SystemExit(preflight())
    if '--validate-only' in sys.argv:
        raise SystemExit(validate_only())
    run()
