#!/usr/bin/env python3
"""133: MERT-v1-95M + MuQ-embed cosine distance on 126/126b/126c stimulus sets.

Design: docs/experiments/metric_win_tie_133_mert_20261020.md
Judging rule: same as 126d.
  score(x, sys) = cosim(L2_norm(mean_pool(embed(x))), μ_sys)
  where μ_sys = L2-normalised mean embedding of 5,131 clean clips for each of the 5 TTM systems.
  d = score(case) - score(clean_base_same_source); tie if |d| ≤ m, m = 0.10 S;
  S = range(mean_score(clean, sys) across 5 systems); pass = 80 %.

Two embedding functions:
  mert   – m-a-p/MERT-v1-95M  rev 12af15fef9d0ac838c3f475bfbbf26d2060dd4f5  768-dim
  muq    – OpenMuQ/MuQ-large-msd-iter  rev 0562a57814f6f8bbd9fdea0a25921a2fce1a841a  1024-dim

Scope:
  126c   36,366 FLAC  → MERT + MuQ
  126b   87,653 FLAC  → MERT + MuQ
  126a   1,334,060 clips on-the-fly → MERT only  (MuQ on 126a would hit the 7h cap)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A
import metric_win_tie_126b as B

EV = Path('/home/kojiek/eval_output_nvme')
OUT = EV / 'metric_win_tie_133_20261020'
OUT_126C = EV / 'metric_win_tie_126c_20261019'
OUT_126B = EV / 'metric_win_tie_126b_20261016'
R115 = EV / 'ttm_metrics_115_20261009'

MERT_REPO = 'm-a-p/MERT-v1-95M'
MERT_REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
MUQ_REPO = 'OpenMuQ/MuQ-large-msd-iter'
MUQ_REVISION = '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'

REPORT = REPO / 'docs/experiments/results/metric_win_tie_133_summary.json'
EMBED_SR = 24000
BATCH = 16
MIN_FREE = 5 * 2 ** 30
M_FRAC, PASS = 0.10, 0.80
SYSTEMS = A.SYSTEMS
log = A.log


# ── Embedding models ──

def load_mert(device='cuda'):
    from transformers import AutoModel, AutoFeatureExtractor
    feat = AutoFeatureExtractor.from_pretrained(MERT_REPO, revision=MERT_REVISION,
                                                trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MERT_REPO, revision=MERT_REVISION,
                                    trust_remote_code=True).to(device).eval()
    return feat, mdl


def load_muq(device='cuda'):
    from muq import MuQ
    mdl = MuQ.from_pretrained(MUQ_REPO, revision=MUQ_REVISION).to(device).eval()
    return mdl


def mean_pool_norm(hidden):
    """Mean-pool time dim and L2-normalise. hidden: (B, T, D) tensor → (B, D) numpy."""
    import torch
    e = hidden.mean(dim=1)  # (B, D)
    norms = e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (e / norms).cpu().numpy().astype(np.float32)


def embed_mert_batch(wavs_24k, feat, mdl, device):
    """Returns (N, 768) L2-normalised embeddings."""
    import torch
    inputs = feat(wavs_24k, sampling_rate=EMBED_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        h = mdl(**inputs).last_hidden_state
    return mean_pool_norm(h)


def embed_muq_batch(wavs_24k, mdl, device):
    """Returns (N, 1024) L2-normalised embeddings."""
    import torch
    max_len = max(len(w) for w in wavs_24k)
    padded = np.zeros((len(wavs_24k), max_len), dtype=np.float32)
    for i, w in enumerate(wavs_24k):
        padded[i, :len(w)] = w
    x = torch.from_numpy(padded).to(device)
    with torch.no_grad():
        h = mdl(x, output_hidden_states=True).last_hidden_state
    return mean_pool_norm(h)


def load_24k(path):
    x, sr = sf.read(str(path), dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != EMBED_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=EMBED_SR)
    return x


def embed_files(paths, feat_mert, mdl_mert, mdl_muq, device, use_muq=True, progress=True):
    """Score list of paths. Returns {path: {'mert': (768,), 'muq': (1024,) or None}}."""
    results = {}
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i:i + BATCH]
        wavs = [load_24k(p) for p in batch_paths]
        wavs_list = [w.tolist() for w in wavs]
        emb_mert = embed_mert_batch(wavs_list, feat_mert, mdl_mert, device)
        emb_muq = embed_muq_batch(wavs, mdl_muq, device) if use_muq else None
        for j, p in enumerate(batch_paths):
            results[str(p)] = {
                'mert': emb_mert[j],
                'muq': emb_muq[j] if use_muq else None,
            }
        if progress and (i // BATCH) % 50 == 0:
            log(f'  embedded {min(i + BATCH, len(paths))}/{len(paths)}')
    return results


def cosim_rows(emb_dict, key_a, key_b, dim='mert'):
    """Cosine similarity between two clip embeddings."""
    ea = emb_dict[key_a][dim]
    eb = emb_dict[key_b][dim]
    return float(np.dot(ea, eb))


def write_cosim_tsv(path, rows, dims=('mert_cosim', 'muq_cosim')):
    """rows: {key: {dim: float}}"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['key', *dims])
        for key, vals in rows.items():
            w.writerow([key, *[f'{vals.get(d, float("nan")):.8f}' for d in dims]])


def read_cosim_tsv(path, dims=('mert_cosim', 'muq_cosim')):
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            result[row['key']] = {d: float(row[d]) for d in dims if d in row}
    return result


# ── score 126c ──

def score_126c(feat_mert, mdl_mert, mdl_muq, device):
    out_mert = OUT / '126c_mert_cosim.tsv'
    out_muq = OUT / '126c_muq_cosim.tsv'
    if out_mert.is_file() and out_muq.is_file():
        log('126c already done, skipping')
        return

    log('=== embedding 126c ===')
    items = json.loads((OUT_126C / 'stimuli/items.json').read_text())
    # Build path → key and base-key mappings
    paths = sorted(set(r['path'] for r in items if Path(r['path']).is_file()))
    log(f'  {len(paths)} clips')

    embs = embed_files(paths, feat_mert, mdl_mert, mdl_muq, device, use_muq=True)

    # Group by window, compute per-window base (the 'mix' condition)
    key_to_path = {r['key']: r['path'] for r in items}
    key_to_base = {r['key']: r['base'] for r in items}

    mert_rows, muq_rows = {}, {}
    for r in items:
        key = r['key']
        base_key = r.get('base', key)
        p = r['path']
        bp = key_to_path.get(base_key, p)
        if p not in embs or bp not in embs:
            continue
        mert_c = cosim_rows({p: embs[p], bp: embs[bp]}, p, bp, 'mert')
        muq_c = cosim_rows({p: embs[p], bp: embs[bp]}, p, bp, 'muq')
        mert_rows[key] = {'mert_cosim': mert_c}
        muq_rows[key] = {'muq_cosim': muq_c}

    write_cosim_tsv(out_mert, {k: {'mert_cosim': v['mert_cosim']} for k, v in mert_rows.items()},
                    dims=['mert_cosim'])
    write_cosim_tsv(out_muq, {k: {'muq_cosim': v['muq_cosim']} for k, v in muq_rows.items()},
                    dims=['muq_cosim'])
    log(f'  wrote {out_mert.name}, {out_muq.name}')


# ── score 126b ──

def score_126b(feat_mert, mdl_mert, mdl_muq, device):
    out_mert = OUT / '126b_mert_cosim.tsv'
    out_muq = OUT / '126b_muq_cosim.tsv'
    if out_mert.is_file() and out_muq.is_file():
        log('126b already done, skipping')
        return

    log('=== embedding 126b ===')
    items = json.loads((OUT_126B / 'stimuli/items.json').read_text())
    paths = sorted(set(r['path'] for r in items if Path(r['path']).is_file()))
    log(f'  {len(paths)} clips')

    embs = embed_files(paths, feat_mert, mdl_mert, mdl_muq, device, use_muq=True)

    key_to_path = {r['key']: r['path'] for r in items}

    mert_rows, muq_rows = {}, {}
    for r in items:
        key = r['key']
        base_key = r.get('base', key)
        p = r['path']
        bp = key_to_path.get(base_key, p)
        if p not in embs or bp not in embs:
            continue
        mert_rows[key] = {'mert_cosim': cosim_rows({p: embs[p], bp: embs[bp]}, p, bp, 'mert')}
        muq_rows[key] = {'muq_cosim': cosim_rows({p: embs[p], bp: embs[bp]}, p, bp, 'muq')}

    write_cosim_tsv(out_mert, {k: {'mert_cosim': v['mert_cosim']} for k, v in mert_rows.items()},
                    dims=['mert_cosim'])
    write_cosim_tsv(out_muq, {k: {'muq_cosim': v['muq_cosim']} for k, v in muq_rows.items()},
                    dims=['muq_cosim'])
    log(f'  wrote {out_mert.name}, {out_muq.name}')


# ── score 126a (MERT only) ──

def score_126a(feat_mert, mdl_mert, device):
    out_dir = OUT / '126a'
    done_marker = OUT / '126a_mert_done.json'
    if done_marker.is_file():
        log('126a MERT already done, skipping')
        return

    log('=== embedding 126a (MERT only) ===')
    # Precompute clean embeddings for each system (5 × 5131 clips)
    clean_mu = {}  # system -> (1024,) mean embed
    log('  precomputing clean system means...')
    for sname, sfolder in SYSTEMS.items():
        clean_unit_dir = OUT / f'126a_clean_embeds/{sname}'
        emb_file = clean_unit_dir / 'mean_embed.npy'
        if emb_file.is_file():
            clean_mu[sname] = np.load(str(emb_file))
            continue
        clean_unit_dir.mkdir(parents=True, exist_ok=True)
        clips = sorted((R115 / sfolder / 'lvl30/audio').glob('*.flac'))
        if not clips:
            log(f'  WARNING: no clean clips found for {sname}')
            continue
        clip_embs = []
        for i in range(0, len(clips), BATCH):
            batch = clips[i:i + BATCH]
            wavs = [load_24k(p).tolist() for p in batch]
            emb = embed_mert_batch(wavs, feat_mert, mdl_mert, device)
            clip_embs.append(emb)
        clip_embs = np.vstack(clip_embs)  # (N, 768)
        mu = clip_embs.mean(axis=0)
        mu /= np.linalg.norm(mu) + 1e-8
        np.save(str(emb_file), mu.astype(np.float32))
        clean_mu[sname] = mu
        log(f'  clean mean {sname}: {len(clips)} clips')

    # For S computation: mean cosim(clean_i, mu_s) per system
    clean_mean_cosim = {}  # sname -> float
    for sname, sfolder in SYSTEMS.items():
        if sname not in clean_mu:
            continue
        mu = clean_mu[sname]
        clips = sorted((R115 / sfolder / 'lvl30/audio').glob('*.flac'))[:5131]
        cosims = []
        for i in range(0, len(clips), BATCH):
            batch = clips[i:i + BATCH]
            wavs = [load_24k(p).tolist() for p in batch]
            emb = embed_mert_batch(wavs, feat_mert, mdl_mert, device)
            cosims.extend([float(np.dot(e, mu)) for e in emb])
        clean_mean_cosim[sname] = np.mean(cosims)
        log(f'  clean cosim {sname}: mean={clean_mean_cosim[sname]:.6f}')

    S_mert = max(clean_mean_cosim.values()) - min(clean_mean_cosim.values()) if clean_mean_cosim else 0.05
    m_mert = M_FRAC * S_mert
    log(f'  S_mert={S_mert:.6f}  m_mert={m_mert:.6f}')
    (OUT / 's_values_mert.json').write_text(json.dumps(
        {'S_mert': S_mert, 'm_mert': m_mert, 'clean_mean_cosim': clean_mean_cosim}, indent=2))

    # Score all 126a variants
    variants = A.variants()  # list of (vname, kind) tuples
    n_total = len(variants) * len(SYSTEMS)
    done = 0
    for vname, vkind in variants:
        for sname, sfolder in SYSTEMS.items():
            out_tsv = out_dir / vname / sname / 'mert_cosim.tsv'
            if out_tsv.is_file():
                done += 1
                continue
            out_tsv.parent.mkdir(parents=True, exist_ok=True)

            mu = clean_mu.get(sname)
            if mu is None:
                continue

            # Generate clips using 126a transform
            ids = A.items(vname, sname)
            rows = {}
            batch_keys, batch_wavs = [], []
            for clip_id in ids:
                src = R115 / sfolder / 'lvl30/audio' / f'{clip_id}.flac'
                if not src.is_file():
                    continue
                x, sr = sf.read(str(src), dtype='float32')
                if x.ndim > 1: x = x.mean(1)
                rng = A.rng_for(vname, sname, clip_id)
                x_var = A.transform(vname, x, sr, rng)
                if x_var is None:
                    continue
                if sr != EMBED_SR:
                    x_var = librosa.resample(x_var, orig_sr=sr, target_sr=EMBED_SR)
                batch_keys.append(f'{vname}/{sname}/{clip_id}')
                batch_wavs.append(x_var.tolist())
                if len(batch_keys) >= BATCH:
                    emb = embed_mert_batch(batch_wavs, feat_mert, mdl_mert, device)
                    for k, e in zip(batch_keys, emb):
                        rows[k] = float(np.dot(e, mu))
                    batch_keys, batch_wavs = [], []
            if batch_keys:
                emb = embed_mert_batch(batch_wavs, feat_mert, mdl_mert, device)
                for k, e in zip(batch_keys, emb):
                    rows[k] = float(np.dot(e, mu))

            write_cosim_tsv(out_tsv, {k: {'mert_cosim': v} for k, v in rows.items()},
                            dims=['mert_cosim'])
            done += 1
            if done % 10 == 0:
                log(f'  126a MERT unit {done}/{n_total}')

    done_marker.write_text(json.dumps({'done': True, 'S_mert': S_mert, 'm_mert': m_mert}))
    log('  126a MERT done')


# ── analysis ──

def analyze():
    log('=== analyze ===')
    s_file = OUT / 's_values_mert.json'
    if not s_file.is_file():
        log('ERROR: s_values_mert.json not found')
        sys.exit(1)
    sv = json.loads(s_file.read_text())
    S_mert = sv['S_mert']
    m_mert = sv['m_mert']

    # Load 126b and 126c clean cosim distributions
    from metric_win_tie_126b import VARIANTS as B_VARIANTS
    b_summary_path = OUT_126B / 'summary.json'
    b_items_path = OUT_126B / 'stimuli/items.json'
    b_items = json.loads(b_items_path.read_text()) if b_items_path.is_file() else []

    mert_126b = read_cosim_tsv(OUT / '126b_mert_cosim.tsv', dims=['mert_cosim']) \
        if (OUT / '126b_mert_cosim.tsv').is_file() else {}
    muq_126b = read_cosim_tsv(OUT / '126b_muq_cosim.tsv', dims=['muq_cosim']) \
        if (OUT / '126b_muq_cosim.tsv').is_file() else {}
    mert_126c = read_cosim_tsv(OUT / '126c_mert_cosim.tsv', dims=['mert_cosim']) \
        if (OUT / '126c_mert_cosim.tsv').is_file() else {}
    muq_126c = read_cosim_tsv(OUT / '126c_muq_cosim.tsv', dims=['muq_cosim']) \
        if (OUT / '126c_muq_cosim.tsv').is_file() else {}

    def judge_cell(cosim_dict, items, vname, kind, m, dim):
        """Judge win/tie/lose for one (vname, kind) cell."""
        # d = cosim(case, base) - cosim(clean, base)
        # For individual-reference: cosim(clean, clean) = 1.0 always → d = cosim - 1
        # Lose: d < -m → cosim < 1 - m
        # Tie: |d| ≤ m → cosim ≥ 1 - m  (since d ≤ 0 always)
        # Win: not applicable for distance-based (d never > 0)
        thresh = 1.0 - m
        n_correct = n_total = 0
        for r in items:
            if r.get('vname') != vname:
                continue
            key = r['key']
            if key not in cosim_dict:
                continue
            c = cosim_dict[key].get(dim, float('nan'))
            if not np.isfinite(c):
                continue
            n_total += 1
            if kind == 'lose':
                # Correct if cosim < thresh (i.e., clearly worse)
                n_correct += int(c < thresh)
            elif kind == 'tie':
                # Correct if cosim ≥ thresh (i.e., similar)
                n_correct += int(c >= thresh)
        if n_total == 0:
            return None
        return {'n': n_total, 'pass_rate': n_correct / n_total, 'pass': n_correct / n_total >= PASS}

    # Build results
    results = {
        'S_mert': S_mert, 'm_mert': m_mert,
        '126b': {}, '126c': {}
    }

    # Read 126b items for per-case analysis
    if b_items:
        b_by_vname = {}
        for r in b_items:
            vname = r.get('vname', r.get('case', ''))
            b_by_vname.setdefault(vname, []).append(r)

        for vname, vrecs in b_by_vname.items():
            kind = vrecs[0].get('kind', 'lose')
            mert_j = judge_cell(mert_126b, vrecs, vname, kind, m_mert, 'mert_cosim')
            muq_j = judge_cell(muq_126b, vrecs, vname, kind, m_mert, 'muq_cosim')
            results['126b'][vname] = {'kind': kind, 'mert': mert_j, 'muq': muq_j}

    # Read 126c items
    c_items_path = OUT_126C / 'stimuli/items.json'
    if c_items_path.is_file():
        c_items = json.loads(c_items_path.read_text())
        c_by_vname = {}
        for r in c_items:
            vname = r.get('vname', r.get('case', ''))
            c_by_vname.setdefault(vname, []).append(r)
        for vname, vrecs in c_by_vname.items():
            kind = vrecs[0].get('kind', 'lose')
            mert_j = judge_cell(mert_126c, vrecs, vname, kind, m_mert, 'mert_cosim')
            muq_j = judge_cell(muq_126c, vrecs, vname, kind, m_mert, 'muq_cosim')
            results['126c'][vname] = {'kind': kind, 'mert': mert_j, 'muq': muq_j}

    # 126a summary (if available)
    done_126a = OUT / '126a_mert_done.json'
    if done_126a.is_file():
        # Aggregate 126a pass rates from per-unit TSV files
        a_results = {}
        for vname, vkind in A.variants():
            cell_rates = []
            for sname in SYSTEMS:
                tsv = OUT / '126a' / vname / sname / 'mert_cosim.tsv'
                if not tsv.is_file():
                    continue
                cosims = [v['mert_cosim'] for v in read_cosim_tsv(tsv, dims=['mert_cosim']).values()]
                if not cosims:
                    continue
                thresh = 1.0 - m_mert
                if vkind == 'lose':
                    rate = sum(c < thresh for c in cosims) / len(cosims)
                else:
                    rate = sum(c >= thresh for c in cosims) / len(cosims)
                cell_rates.append(rate)
            if cell_rates:
                a_results[vname] = {
                    'kind': vkind,
                    'mean_pass_rate': np.mean(cell_rates),
                    'pass': np.mean(cell_rates) >= PASS,
                    'n_systems': len(cell_rates),
                }
        results['126a'] = a_results

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))
    log(f'  wrote {REPORT}')


# ── preflight / validate ──

def check_disk():
    import shutil
    free = shutil.disk_usage(str(OUT.parent)).free
    if free < MIN_FREE:
        log(f'ERROR: only {free/2**30:.1f} GB free (need 5 GB)')
        return False
    return True


def preflight():
    import shutil
    OUT.mkdir(parents=True, exist_ok=True)
    if not check_disk():
        sys.exit(75)

    # Check MERT cached
    from huggingface_hub import try_to_load_from_cache
    r = try_to_load_from_cache(MERT_REPO, 'config.json', revision=MERT_REVISION)
    if not r:
        log(f'ERROR: MERT {MERT_REVISION} not in HF cache')
        sys.exit(75)

    # Check 126a source data
    for sname, sfolder in SYSTEMS.items():
        clips = list((R115 / sfolder / 'lvl30/audio').glob('*.flac'))[:5
        ]
        if not clips:
            log(f'ERROR: no R115 clips for {sname}')
            sys.exit(75)

    # Check 126b/126c stimuli
    for p in [OUT_126C / 'stimuli/items.json', OUT_126B / 'stimuli/items.json']:
        if not p.is_file():
            log(f'ERROR: {p} not found')
            sys.exit(75)

    print('PREFLIGHT OK')
    sys.exit(0)


def validate():
    if not REPORT.is_file():
        log(f'ERROR: {REPORT} not found')
        sys.exit(1)
    d = json.loads(REPORT.read_text())
    if 'S_mert' not in d:
        log('ERROR: malformed report')
        sys.exit(1)
    print('VALIDATE OK')
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    if args.preflight:
        preflight()
    if args.validate_only:
        validate()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device}')

    OUT.mkdir(parents=True, exist_ok=True)
    if not check_disk():
        sys.exit(1)

    feat_mert, mdl_mert = load_mert(device)
    mdl_muq = load_muq(device)

    score_126c(feat_mert, mdl_mert, mdl_muq, device)
    score_126b(feat_mert, mdl_mert, mdl_muq, device)
    score_126a(feat_mert, mdl_mert, device)   # MuQ skipped for 126a (time budget)
    analyze()


if __name__ == '__main__':
    main()
