#!/usr/bin/env python3
"""LALM quality judges on stimuli with a known answer (p2 124; 2026-10-14 meeting).

Two audio LLMs (Music Flamingo 8B, Qwen2.5-Omni-3B thinker) score every clip one at a
time. No text is sampled: the score is the expectation of the digit 1..5 under the
next-token distribution after a fixed rating prompt, so a clip has one deterministic
score per (judge, prompt). Four prompts per judge, plus a paired task on the MIDI
controls (clean and intervened 10 s excerpts joined in one file, both orders).

Sets (all mono 16 kHz at -30 LUFS):
  musiceval  2,748 generated clips with expert MOS (the validity anchor)
  noise      2,500 white-noise ladder clips (4 sources x clean / SNR 40..0, + pure noise)
  midi       real-piece MIDI controls from midi_realpiece_stimuli.py (p2 123 stimuli)
  ttm        6 systems x 5,131 matched MusicCaps prompts (real, MusicGen, MusicLDM x2, ours x2)

  ~/venvs/dac/bin/python            scripts/eval/lalm_quality_judge.py --prepare   # CPU, once
  ~/venvs/music_flamingo/bin/python scripts/eval/lalm_quality_judge.py [--preflight|--validate-only|--smoke N]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path('/home/kojiek/eval_output_nvme/lalm_judge_124')
MIDI = Path('/home/kojiek/eval_output_nvme/midi_realpiece_123')
MUSICEVAL = Path('/mnt/seagate/kojiek_data/musiceval/MusicEval-full')
NOISE = Path('/home/kojiek/eval_output_nvme/ttm_noise_ladder_20261012')
TTM = Path('/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009')
TTM_SYSTEMS = ('musiccaps_real', 'musicgen_medium', 'musicldm', 'musicldm_fid8', 'meanaudio_cfg0', 'meanaudio_cfg3_neg')
SR, TARGET, PEAK_CAP, N_BOOT, SEED = 16000, -30.0, 0.999, 10000, 20261014
MODELS = {'mf': 'nvidia/music-flamingo-2601-hf', 'qwen': 'Qwen/Qwen2.5-Omni-3B'}
TAIL = ' Answer with a single digit from 1 to 5.'
PROMPTS = {
    'overall': 'Rate the overall quality of this music recording on a scale from 1 (very bad) to 5 (excellent).' + TAIL,
    'production': 'Rate the audio production quality of this recording (clarity, fidelity, absence of noise, distortion '
                  'and glitches) on a scale from 1 (very bad) to 5 (excellent).' + TAIL,
    'musical': 'Rate the musical quality of this recording (melody, harmony, rhythm, and how well the performance holds '
               'together) on a scale from 1 (very bad) to 5 (excellent).' + TAIL,
    'professional': 'How likely is it that this is a professionally performed and produced music recording, on a scale '
                    'from 1 (certainly not) to 5 (certainly)?' + TAIL,
}
PAIR_PROMPT = ('This audio contains two music excerpts separated by a short silence. Which excerpt has the better overall '
               'quality, the first or the second? Answer with a single digit: 1 or 2.')
PAIR_SEC, GAP_SEC, LEAD_SEC = 10.0, 0.5, 0.5
SETS = ('musiceval', 'pam', 'noise', 'midi', 'ttm', 'seeds')
HUMAN_SETS = {'musiceval': ('mos_impression', 'mos_alignment'), 'pam': ('ovl', 'rel')}
EXPLAIN_SETS = ('musiceval', 'pam', 'noise', 'midi')
EXPLAIN_PROMPT = ('Listen to this music recording. In at most two short sentences (under 40 words in total), describe its '
                  'most noticeable strengths or flaws in sound quality and musical performance. Then finish with '
                  '"Rating: N", where N is a single digit from 1 (very bad) to 5 (excellent).')
EXPLAIN_TOKENS, EXPLAIN_TTM_IDS = 128, 2500
PAM = Path('/home/kojiek/MeanAudio/research/eval/pam_human_eval/human_eval/music')
SEEDS = Path('/home/kojiek/eval_output_nvme/ttm_metrics_seeds_118_20261011')
SEED_SYSTEMS = ('meanaudio_s16180339_cfg0', 'meanaudio_s16180339_cfg3_neg', 'meanaudio_s27182818_cfg0', 'meanaudio_s27182818_cfg3_neg')
PY = {'mf': '/home/kojiek/venvs/music_flamingo/bin/python', 'qwen': '/home/kojiek/venvs/dac/bin/python'}
LOG_NAME = 'main'


def log(msg):
    d = OUT / 'logs' / LOG_NAME  # two levels below the root: the queue guest watches */*/*.log for stalls
    d.mkdir(parents=True, exist_ok=True)
    line = f'{time.strftime("%F %T")} {msg}'
    print(line, flush=True)
    with open(d / 'run.log', 'a') as f:
        f.write(line + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── items ──
def prepare():
    """Level MusicEval to -30 LUFS and freeze the item list. CPU, dac venv."""
    import pyloudnorm
    import soxr
    rows = []

    def level(src, dst):
        if dst.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        x, sr = sf.read(src, dtype='float64', always_2d=True)
        x = soxr.resample(x.mean(axis=1), sr, SR, quality='VHQ') if sr != SR else x.mean(axis=1)
        lufs, peak = float(pyloudnorm.Meter(SR).integrated_loudness(x)), float(np.max(np.abs(x)))
        if not np.isfinite(lufs) or peak <= 0:
            raise RuntimeError(f'unmeasurable loudness: {src}')
        g = 10 ** ((TARGET - lufs) / 20)
        sf.write(dst, x * (PEAK_CAP / peak if peak * g > PEAK_CAP else g), SR, format='FLAC', subtype='PCM_24')

    for name, mos1, mos2 in csv.reader(open(MUSICEVAL / 'sets/total_mos_list.txt')):
        dst = OUT / 'audio' / 'musiceval' / (Path(name).stem + '.flac')
        level(MUSICEVAL / 'wav' / name, dst)
        system = Path(name).stem.split('-')[-1].split('_')[0]
        rows.append({'key': f'musiceval__{dst.stem}', 'set': 'musiceval', 'path': str(dst), 'group': system,
                     'cluster': dst.stem.split('_')[-1], 'base': '', 'human': mos1, 'human2': mos2})
    for r in csv.DictReader(open(PAM / 'scores.csv')):
        dst = OUT / 'audio' / 'pam' / f'{r["Model"]}__{r["File Name"]}.flac'
        level(PAM / r['Model'] / (r['File Name'] + '.wav'), dst)
        rows.append({'key': f'pam__{dst.stem}', 'set': 'pam', 'path': str(dst), 'group': r['Model'],
                     'cluster': r['File Name'], 'base': '', 'human': r['OVL'], 'human2': r['REL']})
    for r in csv.DictReader(open(NOISE / 'manifest.tsv'), delimiter='\t'):
        rows.append({'key': f'noise__{r["key"]}', 'set': 'noise', 'path': str(NOISE / 'lvl30/audio' / (r['key'] + '.flac')),
                     'group': f'{r["source"]}__{r["cond"]}', 'cluster': r['id'], 'base': '', 'human': '', 'human2': ''})
    for r in json.loads((MIDI / 'items.json').read_text()):
        rows.append({'key': f'midi__{r["key"]}', 'set': 'midi', 'path': r['path'], 'group': f'{r["class"]}__{r["case"]}',
                     'cluster': r['excerpt'], 'base': f'midi__{r["base"]}' if r['base'] else '', 'human': '', 'human2': ''})
    ids = sorted(p.stem for p in (TTM / 'musiccaps_real/lvl30/audio').glob('*.flac'))
    for system in TTM_SYSTEMS:
        for i in ids:
            rows.append({'key': f'ttm__{system}__{i}', 'set': 'ttm', 'path': str(TTM / system / 'lvl30/audio' / f'{i}.flac'),
                         'group': system, 'cluster': i, 'base': '', 'human': '', 'human2': ''})
    for system in SEED_SYSTEMS:
        for i in ids:
            rows.append({'key': f'seeds__{system}__{i}', 'set': 'seeds', 'path': str(SEEDS / system / 'lvl30/audio' / f'{i}.flac'),
                         'group': system, 'cluster': i, 'base': '', 'human': '', 'human2': ''})
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} missing files, first: {missing[0]}')
    with open(OUT / 'items.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r['set'] == s for r in rows) for s in SETS}
    (OUT / 'items.json').write_text(json.dumps({'n': len(rows), 'sets': counts, 'sha256': sha(OUT / 'items.tsv')}, indent=1))
    print(json.dumps(json.loads((OUT / 'items.json').read_text())))


def items():
    meta = json.loads((OUT / 'items.json').read_text())
    if sha(OUT / 'items.tsv') != meta['sha256']:
        raise SystemExit('items.tsv does not match the frozen sha256')
    return list(csv.DictReader(open(OUT / 'items.tsv'), delimiter='\t'))


def tasks(rows, judges=None):
    """Ordered (judge, task, set, work list). Small validity sets first, so a bad judge shows early."""
    by = {s: [r for r in rows if r['set'] == s] for s in SETS}
    path = {r['key']: r['path'] for r in rows}
    sub = set(sorted({r['cluster'] for r in by['ttm']})[:EXPLAIN_TTM_IDS])
    pairs = [{'key': f'{r["key"]}|{order}', 'path': (r['base'], r['key']) if order == 'clean_first' else (r['key'], r['base'])}
             for r in by['midi'] if r['base'] for order in ('clean_first', 'clean_second')]
    pairs = [{'key': p['key'], 'path': (path[p['path'][0]], path[p['path'][1]])} for p in pairs]
    out = []
    for judge in (judges or MODELS):
        for s in ('musiceval', 'pam', 'noise', 'midi'):
            out += [(judge, p, s, by[s]) for p in PROMPTS]
        out.append((judge, 'pair', 'midi', pairs))
        for s in ('ttm', 'seeds'):
            out += [(judge, p, s, by[s]) for p in PROMPTS]
        out += [(judge, 'explain', s, by[s]) for s in EXPLAIN_SETS]
        out.append((judge, 'explain', 'ttm', [r for r in by['ttm'] if r['cluster'] in sub]))
    return out


def shard_path(root, judge, task, s):
    return root / 'scores' / f'{judge}__{task}__{s}.jsonl'


def read_done(path):
    done = {}
    if path.is_file():
        for line in open(path):
            try:
                d = json.loads(line)
                done[d['key']] = d
            except json.JSONDecodeError:
                pass
    return done


# ── judges ──
class Judge:
    def __init__(self, name):
        import torch
        self.torch, self.name = torch, name
        if name == 'mf':
            from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(MODELS[name])
            self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
                MODELS[name], torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        else:
            from transformers import AutoProcessor
            from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
            # local snapshot path: this transformers build queries the hub for a repo id even when offline
            snap = sorted(Path.home().glob(f'.cache/huggingface/hub/models--{MODELS[name].replace("/", "--")}/snapshots/*'))[-1]
            self.proc = AutoProcessor.from_pretrained(snap, trust_remote_code=True)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                snap, dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa').eval()
        tok = self.proc.tokenizer
        ids = [tok.encode(str(d), add_special_tokens=False) for d in range(1, 6)]
        if any(len(i) != 1 for i in ids):
            raise SystemExit(f'{name}: digits are not single tokens: {ids}')
        self.digits = torch.tensor([i[0] for i in ids], device='cuda')
        self.tmp = Path(f'/dev/shm/lalm124_{os.getpid()}.wav')
        # Qwen2.5-Omni ships generation_config.eos_token_id=None and would always run to max_new_tokens
        self.eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids('<|im_end|>')] if name == 'qwen' else [])

    def inputs(self, path, prompt):
        if self.name == 'mf':
            conv = [[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'audio', 'path': path}]}]]
            inp = self.proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True).to('cuda')
            inp['input_features'] = inp['input_features'].to(self.model.dtype)
            return inp
        x, sr = sf.read(path, dtype='float32')
        if x.ndim != 1:
            raise RuntimeError(f'expected mono: {path}')
        if sr != SR:  # the MusicGen ttm renders are 32 kHz; Flamingo's processor resamples them itself (first p2 124 attempt)
            import soxr
            x = soxr.resample(x, sr, SR, quality='VHQ').astype('float32')
        conv = [{'role': 'user', 'content': [{'type': 'audio', 'audio': path}, {'type': 'text', 'text': prompt}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        return self.proc(text=[text], audio=[x], return_tensors='pt', padding=True, sampling_rate=SR).to('cuda')

    def explain(self, path):
        """Greedy free-text critique that ends in 'Rating: N'. ev is None when no rating can be parsed."""
        import re
        inp = self.inputs(path, EXPLAIN_PROMPT)
        with self.torch.inference_mode():
            ids = self.model.generate(**inp, max_new_tokens=EXPLAIN_TOKENS, do_sample=False, eos_token_id=self.eos,
                                      pad_token_id=self.eos[0])
        text = self.proc.tokenizer.decode(ids[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        m = re.findall(r'Rating:\s*\**\s*([1-5])', text)
        return {'ev': int(m[-1]) if m else None, 'text': text}

    def score(self, path, prompt, n_digits=5):
        torch = self.torch
        if isinstance(path, tuple):
            a, b = (sf.read(p, dtype='float32')[0] for p in path)
            cut = lambda x: x[int(LEAD_SEC * SR):int((LEAD_SEC + PAIR_SEC) * SR)]
            sf.write(self.tmp, np.concatenate([cut(a), np.zeros(int(GAP_SEC * SR), np.float32), cut(b)]), SR, subtype='FLOAT')
            path = str(self.tmp)
        inp = self.inputs(path, prompt)
        with torch.inference_mode():
            lg = self.model(**inp).logits[0, -1].float()
        ids = self.digits[:n_digits]
        p = torch.softmax(lg[ids], -1)
        mass = float(torch.softmax(lg, -1)[ids].sum())
        ev = float((p * torch.arange(1, n_digits + 1, device='cuda')).sum())
        if not np.isfinite(ev):
            raise RuntimeError(f'non-finite score: {path}')
        return {'ev': round(ev, 5), 'p': [round(float(v), 5) for v in p], 'mass': round(mass, 5)}


def run(root, limit=None, judges=None):
    rows = items()
    plan = tasks(rows, judges)
    judge, t_start, n_done = None, time.time(), 0
    for name, task, s, work in plan:
        work = work[:limit] if limit else work
        dst = shard_path(root, name, task, s)
        dst.parent.mkdir(parents=True, exist_ok=True)
        done = read_done(dst)
        todo = [w for w in work if w['key'] not in done]
        if not todo:
            continue
        if judge is None or judge.name != name:
            del judge
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            judge = Judge(name)
            log(f'loaded {name}')
        prompt, nd = (PAIR_PROMPT, 2) if task == 'pair' else (PROMPTS.get(task), 5)
        t0 = time.time()
        with open(dst, 'a') as f:
            for i, w in enumerate(todo):
                res = judge.explain(w['path']) if task == 'explain' else judge.score(w['path'], prompt, nd)
                f.write(json.dumps({'key': w['key'], **res}) + '\n')
                if (i + 1) % 200 == 0 or i + 1 == len(todo):
                    f.flush()
                    log(f'{name} {task} {s}: {i + 1}/{len(todo)}  {(time.time() - t0) / (i + 1):.3f} s/clip')
        n_done += len(todo)
        (dst.parent / f'{dst.stem}.metrics.json').write_text(json.dumps({'n': len(work), 'sec': time.time() - t0}))
    log(f'scoring done: {n_done} new scores in {(time.time() - t_start) / 3600:.2f} h')


# ── analysis ──
def boot(cluster_means, rng):
    v = np.asarray(cluster_means, float)
    draws = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
    return [float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def by_cluster(pairs):
    d = {}
    for c, v in pairs:
        d.setdefault(c, []).append(v)
    return [float(np.mean(v)) for v in d.values()]


def analyze(root, limit=None):
    from scipy.stats import pearsonr, spearmanr
    rows = items()
    meta = {r['key']: r for r in rows}
    rng = np.random.default_rng(SEED)
    out = {'n_items': len(rows), 'prompts': PROMPTS, 'pair_prompt': PAIR_PROMPT, 'explain_prompt': EXPLAIN_PROMPT, 'judges': {}}
    for name in MODELS:
        J = {}
        for task in list(PROMPTS) + ['explain']:
            raw = {}
            for s in SETS:
                raw.update(read_done(shard_path(root, name, task, s)))
            sc = {k: d['ev'] for k, d in raw.items() if d['ev'] is not None}
            if not sc:
                continue
            T = {'n_scored': len(raw), 'parse_rate': len(sc) / len(raw)}
            if task != 'explain':
                T['digit_mass_mean'] = float(np.mean([d['mass'] for d in raw.values()]))
            # human-rated sets: agreement with the listening panel
            for hs, (h1n, h2n) in HUMAN_SETS.items():
                me = [(meta[k], v) for k, v in sc.items() if meta[k]['set'] == hs]
                if len(me) > 10:
                    x, h1, h2 = (np.array(a, float) for a in zip(*[(v, r['human'], r['human2']) for r, v in me]))
                    sysm = {}
                    for r, v in me:
                        sysm.setdefault(r['group'], []).append((v, float(r['human'])))
                    sm = np.array([np.mean(v, axis=0) for v in sysm.values()])
                    T[hs] = {'n': len(me), f'clip_spearman_{h1n}': float(spearmanr(x, h1)[0]),
                             f'clip_pearson_{h1n}': float(pearsonr(x, h1)[0]),
                             f'clip_spearman_{h2n}': float(spearmanr(x, h2)[0]),
                             'n_systems': len(sm), f'system_spearman_{h1n}': float(spearmanr(sm[:, 0], sm[:, 1])[0]),
                             'system_means': {g: [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))] for g, v in sorted(sysm.items())},
                             'score_mean': float(x.mean()), 'score_sd': float(x.std())}
            # noise ladder: every condition against the same clip clean
            nz = {}
            for k, v in sc.items():
                if meta[k]['set'] == 'noise':
                    nz.setdefault(meta[k]['group'], {})[meta[k]['cluster']] = v
            if nz:
                T['noise'] = {}
                for g, d in sorted(nz.items()):
                    src, cond = g.split('__')
                    base = nz.get(f'{src}__clean', {})
                    e = {'n': len(d), 'mean': float(np.mean(list(d.values())))}
                    if cond != 'clean' and base:
                        diff = [d[c] - base[c] for c in d if c in base]
                        if diff:
                            e['delta_vs_clean'] = boot(diff, rng)
                            e['frac_below_clean'] = float(np.mean(np.array(diff) < 0))
                    T['noise'][g] = e
            # MIDI controls: every case against the clean render of the same excerpt and bank
            md = {}
            for k, v in sc.items():
                r = meta[k]
                if r['set'] == 'midi' and r['base'] and r['base'] in sc:
                    md.setdefault(r['group'], []).append((r['cluster'], v - sc[r['base']]))
            if md:
                T['midi'] = {'clean_mean': float(np.mean([v for k, v in sc.items() if meta[k]['group'] == 'base__clean']))}
                for g, d in sorted(md.items()):
                    cm = by_cluster(d)
                    T['midi'][g] = {'n': len(d), 'delta_vs_clean': boot(cm, rng), 'frac_below_clean': float(np.mean([v < 0 for _, v in d]))}
                for cls in ('degrade', 'control', 'ambiguous', 'floor'):
                    d = [x for g, v in md.items() if g.startswith(cls + '__') for x in v]
                    if d:
                        T['midi'][f'class__{cls}'] = {'n': len(d), 'delta_vs_clean': boot(by_cluster(d), rng),
                                                      'mean_abs_case_delta': float(np.mean([abs(np.mean([v for _, v in x])) for g, x in md.items() if g.startswith(cls + '__')]))}
            # TTM systems against the real recording of the same prompt
            tt = {}
            for k, v in sc.items():
                if meta[k]['set'] in ('ttm', 'seeds'):
                    tt.setdefault(meta[k]['group'], {})[meta[k]['cluster']] = v
            if tt:
                T['ttm'] = {}
                real = tt.get('musiccaps_real', {})
                for g, d in tt.items():
                    e = {'n': len(d), 'mean': float(np.mean(list(d.values())))}
                    diff = [d[c] - real[c] for c in d if c in real]
                    if g != 'musiccaps_real' and diff:
                        e['delta_vs_real'] = boot(diff, rng)
                        e['frac_above_real'] = float(np.mean(np.array(diff) > 0))
                    T['ttm'][g] = e
            J[task] = T
        pr = read_done(shard_path(root, name, 'pair', 'midi'))
        if pr:
            pc, first = {}, []
            for k, d in pr.items():
                item, order = k.split('|')
                p_clean = d['p'][0] if order == 'clean_first' else d['p'][1]
                first.append(d['p'][0])
                pc.setdefault(meta[item]['group'], {}).setdefault((meta[item]['cluster'], item), []).append(p_clean)
            P = {'p_first_mean': float(np.mean(first)), 'digit_mass_mean': float(np.mean([d['mass'] for d in pr.values()]))}
            for g, d in sorted(pc.items()):
                both = [(c, float(np.mean(v))) for (c, _), v in d.items() if len(v) == 2]
                if both:
                    P[g] = {'n': len(both), 'p_clean_better': boot(by_cluster(both), rng),
                            'frac_clean_preferred': float(np.mean([v > 0.5 for _, v in both]))}
            J['pair'] = P
        if J:
            out['judges'][name] = J
    (root / 'summary.json').write_text(json.dumps(out, indent=1))
    log(f'wrote {root / "summary.json"}')
    return out


def validate(root, limit=None):
    rows = items()
    bad = []
    for name, task, s, work in tasks(rows):
        work = work[:limit] if limit else work
        done = read_done(shard_path(root, name, task, s))
        miss = sum(w['key'] not in done for w in work)
        if miss:
            bad.append(f'{name} {task} {s}: {miss} missing')
    if not (root / 'summary.json').is_file():
        bad.append('summary.json missing')
    if bad:
        raise SystemExit('INVALID: ' + '; '.join(bad[:8]))
    print('valid')


def preflight():
    import shutil
    rows = items()
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} item files missing, first: {missing[0]}')
    if shutil.disk_usage(OUT).free < 5 * 2 ** 30:
        raise SystemExit('less than 5 GiB free under the output root')
    for mid in MODELS.values():
        if not list(Path.home().glob(f'.cache/huggingface/hub/models--{mid.replace("/", "--")}/snapshots/*')):
            raise SystemExit(f'model not in the HF cache: {mid}')
    print(f'preflight ok: {len(rows)} items, {sum(len(w) for *_, w in tasks(rows))} scores planned')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--smoke', type=int)
    ap.add_argument('--judge', choices=sorted(MODELS), help='score with this judge only, then exit (each judge has its own venv)')
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    if a.preflight:
        return preflight()
    root = OUT / 'smoke' if a.smoke else OUT
    if a.validate_only:
        return validate(root, a.smoke)
    if a.judge:
        global LOG_NAME
        import logging
        LOG_NAME = a.judge
        logging.disable(logging.WARNING)  # Qwen logs a system-prompt warning on every call
        return run(root, a.smoke, [a.judge])
    if not a.analyze_only:
        import subprocess
        for j in MODELS:  # one process per judge: the two models need different transformers builds
            subprocess.run([PY[j], str(Path(__file__).resolve()), '--judge', j] + (['--smoke', str(a.smoke)] if a.smoke else []), check=True)
    analyze(root, a.smoke)
    validate(root, a.smoke)


if __name__ == '__main__':
    sys.exit(main())
