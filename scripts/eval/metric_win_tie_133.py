#!/usr/bin/env python3
"""133: MERT-v1-95M + MuQ-embed cosine distance on 126/126b/126c stimulus sets.

Design: docs/experiments/metric_win_tie_133_mert_20261020.md

r1 failure (2026-10-20) and design corrections applied in r2:
  1. score_126a: switched to pairwise cosim(embed(variant), embed(clean, same clip+sys)).
     Per-clip clean MERT embeds saved as per_clip_embeds.npy per system.
     S_mert keeps the preregistered centroid definition; reuses s_values_mert.json from r1.
  2. S_muq: computed from MuQ centroids on 5 TTM systems' clean lvl30 audio
     (same centroid definition as S_mert). Saved to s_values_muq.json.
     MuQ judging now uses m_muq, not m_mert.
  3. analyze(): uses items['case'] field (not absent 'vname'); excludes base=='' items
     from judging; provides descriptive stats (no pass/fail) for soft/floor/descriptive.

Judging rule (126b, 126c):
  d = cosim(embed(case), embed(base_clean)) - 1.0   [cosim(clean,clean)=1]
  tie if |d| <= m; lose if d < -m; m = 0.10 * S; pass = 80%.

Judging rule (126a):
  d = cosim(embed(variant_x), embed(clean_x))   [pairwise, same clip same sys] - 1.0
  Same m and pass threshold.

S_mert = range(mean cosim(clean_i, mu_sys) across 5 TTM systems)  [centroid, preregistered]
S_muq  = same definition with MuQ embeddings.

Scope:
  126c  36,366 FLAC -> MERT + MuQ  (pairwise cosim vs base; complete in r1)
  126b  87,653 FLAC -> MERT + MuQ  (pairwise cosim vs base; complete in r1)
  126a  ~1.33M on-the-fly clips   -> MERT only (pairwise vs same clean clip)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'  # pool workers fork; avoid BLAS oversubscription
os.environ.pop('MKL_NUM_THREADS', None)

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A
import metric_win_tie_126b as B  # noqa: F401 (imported for B_VARIANTS reference)

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


# ── Embedding models ──────────────────────────────────────────────────────────

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
    """Mean-pool time dim and L2-normalise. hidden: (B, T, D) tensor -> (B, D) numpy."""
    import torch
    e = hidden.mean(dim=1)
    norms = e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (e / norms).cpu().numpy().astype(np.float32)


def embed_mert_batch(wavs_24k, feat, mdl, device):
    """Returns (N, 768) L2-normalised MERT embeddings."""
    import torch
    inputs = feat(wavs_24k, sampling_rate=EMBED_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        h = mdl(**inputs).last_hidden_state
    return mean_pool_norm(h)


def embed_muq_batch(wavs_24k, mdl, device):
    """Returns (N, 1024) L2-normalised MuQ embeddings."""
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
    """Embed list of paths. Returns {str(path): {'mert': (768,), 'muq': (1024,)|None}}."""
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


# ── score 126c (pairwise, complete in r1) ────────────────────────────────────

def score_126c(feat_mert, mdl_mert, mdl_muq, device):
    out_mert = OUT / '126c_mert_cosim.tsv'
    out_muq = OUT / '126c_muq_cosim.tsv'
    if out_mert.is_file() and out_muq.is_file():
        log('126c already done, skipping')
        return

    log('=== embedding 126c ===')
    items = json.loads((OUT_126C / 'stimuli/items.json').read_text())
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
        ea_m, eb_m = embs[p]['mert'], embs[bp]['mert']
        ea_q, eb_q = embs[p]['muq'], embs[bp]['muq']
        mert_rows[key] = {'mert_cosim': float(np.dot(ea_m, eb_m))}
        muq_rows[key] = {'muq_cosim': float(np.dot(ea_q, eb_q))}

    write_cosim_tsv(out_mert, {k: {'mert_cosim': v['mert_cosim']} for k, v in mert_rows.items()},
                    dims=['mert_cosim'])
    write_cosim_tsv(out_muq, {k: {'muq_cosim': v['muq_cosim']} for k, v in muq_rows.items()},
                    dims=['muq_cosim'])
    log(f'  wrote {out_mert.name}, {out_muq.name}')


# ── score 126b (pairwise, complete in r1) ────────────────────────────────────

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
        ea_m, eb_m = embs[p]['mert'], embs[bp]['mert']
        ea_q, eb_q = embs[p]['muq'], embs[bp]['muq']
        mert_rows[key] = {'mert_cosim': float(np.dot(ea_m, eb_m))}
        muq_rows[key] = {'muq_cosim': float(np.dot(ea_q, eb_q))}

    write_cosim_tsv(out_mert, {k: {'mert_cosim': v['mert_cosim']} for k, v in mert_rows.items()},
                    dims=['mert_cosim'])
    write_cosim_tsv(out_muq, {k: {'muq_cosim': v['muq_cosim']} for k, v in muq_rows.items()},
                    dims=['muq_cosim'])
    log(f'  wrote {out_mert.name}, {out_muq.name}')


# ── compute S_muq ─────────────────────────────────────────────────────────────

def compute_s_muq(mdl_muq, device):
    """MuQ centroid calibration on 5 TTM systems' clean lvl30 clips. Saves s_values_muq.json."""
    s_file = OUT / 's_values_muq.json'
    if s_file.is_file():
        log('S_muq already computed, skipping')
        return

    log('=== computing S_muq (MuQ centroids on TTM clean lvl30) ===')
    clean_mean_cosim_muq = {}

    for sname, sfolder in SYSTEMS.items():
        clips = sorted((R115 / sfolder / 'lvl30/audio').glob('*.flac'))[:5131]
        if not clips:
            log(f'  WARNING: no clips for {sname}')
            continue
        all_embs = []
        for i in range(0, len(clips), BATCH):
            batch = clips[i:i + BATCH]
            wavs = [load_24k(p).tolist() for p in batch]
            emb = embed_muq_batch(wavs, mdl_muq, device)
            all_embs.append(emb)
            if (i // BATCH) % 40 == 0:
                log(f'  MuQ {sname}: {min(i + BATCH, len(clips))}/{len(clips)}')
        arr = np.vstack(all_embs).astype(np.float32)  # (N, 1024)
        mu = arr.mean(0)
        mu /= np.linalg.norm(mu) + 1e-8
        mean_cosim = float((arr @ mu).mean())
        clean_mean_cosim_muq[sname] = mean_cosim
        log(f'  MuQ {sname}: mean_cosim={mean_cosim:.6f}')

    S_muq = max(clean_mean_cosim_muq.values()) - min(clean_mean_cosim_muq.values()) \
        if clean_mean_cosim_muq else 0.05
    m_muq = M_FRAC * S_muq
    log(f'  S_muq={S_muq:.6f}  m_muq={m_muq:.6f}')
    s_file.write_text(json.dumps({
        'S_muq': S_muq, 'm_muq': m_muq,
        'clean_mean_cosim_muq': clean_mean_cosim_muq,
    }, indent=2))


# ── score 126a (MERT only, pairwise cosim vs same clean clip) ─────────────────

LIMIT_126A = None     # smoke: first N clips
VARS_126A = None      # smoke: subset of variant names
POOL_WORKERS = 12


def _prep_126a(job):
    """Read lvl30 clip, apply the 126a variant exactly as A.make_one does (incl. the FLAC
    subtype quantisation, done in memory), resample to 24 kHz. Returns (clip_id, float32)."""
    src, sname, vname = job
    var = A.VARIANTS[vname]
    x, sr = sf.read(src, dtype='float64')
    if x.ndim > 1:
        x = x.mean(1)
    y, _ = A.transform(x, var, A.rng_for(sname, vname, Path(src).stem), int(sr))
    q = 2 ** 15 if var['subtype'] == 'PCM_16' else 2 ** 23
    y = (np.round(y * q) / q).astype(np.float32)
    if int(sr) != EMBED_SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=EMBED_SR)
    return Path(src).stem, y

def score_126a(feat_mert, mdl_mert, device):
    out_dir = OUT / '126a'
    done_marker = OUT / '126a_mert_done.json'
    if done_marker.is_file():
        log('126a MERT already done, skipping')
        return

    log('=== embedding 126a (MERT only, pairwise) ===')

    # Load preregistered S_mert from r1 (centroid definition)
    s_file = OUT / 's_values_mert.json'
    s_r1 = json.loads(s_file.read_text()) if s_file.is_file() else None

    # Step 1: per-clip clean MERT embeddings per system
    clean_per_clip = {}    # sname -> {clip_id: (768,) float32 L2-normed}
    clean_mu = {}          # sname -> (768,) centroid
    clean_mean_cosim = {}  # sname -> float (for S verification)

    log('  computing per-clip clean embeddings...')
    for sname, sfolder in SYSTEMS.items():
        clip_dir = OUT / f'126a_clean_embeds/{sname}'
        clip_dir.mkdir(parents=True, exist_ok=True)
        per_clip_file = clip_dir / 'per_clip_embeds.npy'
        ids_file = clip_dir / 'clip_ids.json'
        mean_file = clip_dir / 'mean_embed.npy'

        clips = sorted((R115 / sfolder / 'lvl30/audio').glob('*.flac'))
        if LIMIT_126A is not None:
            keep = {cid for cid, _ in A.items(limit=LIMIT_126A)}
            clips = [p for p in clips if p.stem in keep]
        clip_ids_sys = [p.stem for p in clips]

        if per_clip_file.is_file() and ids_file.is_file():
            arr = np.load(str(per_clip_file))               # (N, 768)
            saved_ids = json.loads(ids_file.read_text())
            if mean_file.is_file():
                mu = np.load(str(mean_file))
            else:
                mu = arr.mean(0); mu /= np.linalg.norm(mu) + 1e-8
                mu = mu.astype(np.float32); np.save(str(mean_file), mu)
            mean_cosim = float((arr @ mu).mean())
            clean_per_clip[sname] = dict(zip(saved_ids, arr))
            clean_mu[sname] = mu
            clean_mean_cosim[sname] = mean_cosim
            log(f'  {sname}: loaded {len(saved_ids)} clips, mean_cosim={mean_cosim:.6f}')
            continue

        if not clips:
            log(f'  WARNING: no clean clips for {sname}')
            continue

        all_embs = []
        for i in range(0, len(clips), BATCH):
            batch = clips[i:i + BATCH]
            wavs = [load_24k(p).tolist() for p in batch]
            emb = embed_mert_batch(wavs, feat_mert, mdl_mert, device)
            all_embs.append(emb)
            if (i // BATCH) % 64 == 0:
                log(f'  {sname}: {min(i + BATCH, len(clips))}/{len(clips)}')

        arr = np.vstack(all_embs).astype(np.float32)  # (N, 768)
        if mean_file.is_file():
            mu = np.load(str(mean_file))
        else:
            mu = arr.mean(0); mu /= np.linalg.norm(mu) + 1e-8
            mu = mu.astype(np.float32); np.save(str(mean_file), mu)

        np.save(str(per_clip_file), arr)
        ids_file.write_text(json.dumps(clip_ids_sys))

        mean_cosim = float((arr @ mu).mean())
        clean_per_clip[sname] = dict(zip(clip_ids_sys, arr))
        clean_mu[sname] = mu
        clean_mean_cosim[sname] = mean_cosim
        log(f'  {sname}: {len(clips)} clips, mean_cosim={mean_cosim:.6f}')

    # Step 2: verify S_mert vs r1 (must agree within 1e-4)
    if not clean_mean_cosim:
        log('ERROR: no clean embeddings computed'); sys.exit(1)

    S_computed = max(clean_mean_cosim.values()) - min(clean_mean_cosim.values())
    if s_r1 is not None:
        diff = abs(S_computed - s_r1['S_mert'])
        log(f'  S_mert check: computed={S_computed:.6f} r1={s_r1["S_mert"]:.6f} diff={diff:.2e}')
        if diff > 1e-4:
            log(f'  WARNING: S_mert diff {diff:.2e} exceeds 1e-4 tolerance')
        S_mert = s_r1['S_mert']
        m_mert = s_r1['m_mert']
    else:
        S_mert = S_computed
        m_mert = M_FRAC * S_mert
        s_file.write_text(json.dumps(
            {'S_mert': S_mert, 'm_mert': m_mert, 'clean_mean_cosim': clean_mean_cosim}, indent=2))
        log(f'  S_mert={S_mert:.6f}  m_mert={m_mert:.6f}  (no r1 file, computed fresh)')

    # Step 3: score all 126a variants (pairwise cosim). Transforms (incl. ffmpeg mp3
    # round-trips) run in a CPU pool; the main process only batches MERT on GPU.
    all_clip_ids = [cid for cid, _ in A.items(limit=LIMIT_126A)]
    variants_list = [v for v in A.variants() if VARS_126A is None or v['name'] in VARS_126A]
    n_total = len(variants_list) * len(SYSTEMS)
    done = 0
    from multiprocessing import get_context
    pool = get_context('fork').Pool(POOL_WORKERS)
    try:
        for var in variants_list:
            vname = var['name']
            for sname, sfolder in SYSTEMS.items():
                out_tsv = out_dir / vname / sname / 'mert_cosim.tsv'
                done += 1
                if out_tsv.is_file():
                    continue
                per_clip = clean_per_clip.get(sname, {})
                if not per_clip:
                    log(f'ERROR: no clean embeddings for {sname}'); sys.exit(1)
                t0 = time.time()
                jobs = [(str(R115 / sfolder / 'lvl30/audio' / f'{cid}.flac'), sname, vname)
                        for cid in all_clip_ids if cid in per_clip]
                rows, bk, bw = {}, [], []

                def flush():
                    embs = embed_mert_batch(bw, feat_mert, mdl_mert, device)
                    for cid, e in zip(bk, embs):
                        rows[f'{vname}/{sname}/{cid}'] = float(np.dot(e, per_clip[cid]))
                    bk.clear(); bw.clear()

                for cid, x24 in pool.imap(_prep_126a, jobs, chunksize=8):
                    bk.append(cid); bw.append(x24)
                    if len(bk) >= BATCH:
                        flush()
                if bk:
                    flush()
                if len(rows) != len(jobs):
                    log(f'ERROR: {vname}/{sname} scored {len(rows)}/{len(jobs)}'); sys.exit(1)
                out_tsv.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_tsv.with_suffix('.tmp')
                write_cosim_tsv(tmp, {k: {'mert_cosim': v} for k, v in rows.items()}, dims=['mert_cosim'])
                tmp.rename(out_tsv)
                c = np.array(list(rows.values()))
                log(f'  126a unit {done}/{n_total} {vname}/{sname}: n={len(c)} '
                    f'mean={c.mean():.6f} min={c.min():.6f} ({time.time() - t0:.0f}s)')
    finally:
        pool.close(); pool.join()

    if LIMIT_126A is not None or VARS_126A is not None:
        log('  126a subset run (smoke): not writing done marker')
        return
    done_marker.write_text(json.dumps({'done': True, 'S_mert': S_mert, 'm_mert': m_mert}))
    log('  126a MERT done')


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze():
    log('=== analyze ===')
    s_mert_file = OUT / 's_values_mert.json'
    s_muq_file = OUT / 's_values_muq.json'

    if not s_mert_file.is_file():
        log('ERROR: s_values_mert.json not found'); sys.exit(1)
    sv_mert = json.loads(s_mert_file.read_text())
    S_mert, m_mert = sv_mert['S_mert'], sv_mert['m_mert']

    if s_muq_file.is_file():
        sv_muq = json.loads(s_muq_file.read_text())
        S_muq, m_muq = sv_muq['S_muq'], sv_muq['m_muq']
    else:
        log('WARNING: s_values_muq.json not found; using m_mert for MuQ (suboptimal)')
        S_muq, m_muq = S_mert, m_mert

    mert_126b = read_cosim_tsv(OUT / '126b_mert_cosim.tsv', dims=['mert_cosim']) \
        if (OUT / '126b_mert_cosim.tsv').is_file() else {}
    muq_126b = read_cosim_tsv(OUT / '126b_muq_cosim.tsv', dims=['muq_cosim']) \
        if (OUT / '126b_muq_cosim.tsv').is_file() else {}
    mert_126c = read_cosim_tsv(OUT / '126c_mert_cosim.tsv', dims=['mert_cosim']) \
        if (OUT / '126c_mert_cosim.tsv').is_file() else {}
    muq_126c = read_cosim_tsv(OUT / '126c_muq_cosim.tsv', dims=['muq_cosim']) \
        if (OUT / '126c_muq_cosim.tsv').is_file() else {}

    JUDGE_KINDS = {'lose', 'tie'}

    def judge_cell(cosim_dict, recs, kind, m, dim):
        thresh = 1.0 - m
        n_correct = n_total = 0
        for r in recs:
            if r.get('base', '') == '':
                continue   # no reference; cosim undefined
            key = r['key']
            if key not in cosim_dict:
                continue
            c = cosim_dict[key].get(dim, float('nan'))
            if not np.isfinite(c):
                continue
            n_total += 1
            n_correct += int(c < thresh) if kind == 'lose' else int(c >= thresh)
        if n_total == 0:
            return None
        return {'n': n_total, 'pass_rate': n_correct / n_total, 'pass': n_correct / n_total >= PASS}

    def describe_cell(cosim_dict, recs, dim):
        """Descriptive stats (no pass/fail) for soft/floor/descriptive/ref kinds."""
        vals = []
        for r in recs:
            if r.get('base', '') == '':
                continue
            key = r['key']
            if key not in cosim_dict:
                continue
            c = cosim_dict[key].get(dim, float('nan'))
            if np.isfinite(c):
                vals.append(c)
        if not vals:
            return None
        v = np.array(vals)
        return {'n': len(v), 'mean': float(v.mean()), 'std': float(v.std()),
                'p10': float(np.percentile(v, 10)), 'median': float(np.median(v)),
                'p90': float(np.percentile(v, 90))}

    def analyze_set(items, mert_dict, muq_dict, m_m, m_q):
        """Group by case, judge or describe each case cell."""
        by_case = {}
        for r in items:
            by_case.setdefault(r['case'], []).append(r)
        out = {}
        for case, recs in by_case.items():
            kind = recs[0].get('kind', 'lose')
            if kind in JUDGE_KINDS:
                out[case] = {
                    'kind': kind,
                    'mert': judge_cell(mert_dict, recs, kind, m_m, 'mert_cosim'),
                    'muq':  judge_cell(muq_dict,  recs, kind, m_q, 'muq_cosim'),
                }
            else:
                out[case] = {
                    'kind': kind,
                    'mert': describe_cell(mert_dict, recs, 'mert_cosim'),
                    'muq':  describe_cell(muq_dict,  recs, 'muq_cosim'),
                }
        return out

    results = {
        'S_mert': S_mert, 'm_mert': m_mert,
        'S_muq': S_muq,   'm_muq': m_muq,
        '126b': {}, '126c': {},
    }

    b_items_path = OUT_126B / 'stimuli/items.json'
    if b_items_path.is_file():
        b_items = json.loads(b_items_path.read_text())
        results['126b'] = analyze_set(b_items, mert_126b, muq_126b, m_mert, m_muq)
        log(f'  126b: {len(results["126b"])} cases')

    c_items_path = OUT_126C / 'stimuli/items.json'
    if c_items_path.is_file():
        c_items = json.loads(c_items_path.read_text())
        results['126c'] = analyze_set(c_items, mert_126c, muq_126c, m_mert, m_muq)
        log(f'  126c: {len(results["126c"])} cases')

    done_126a = OUT / '126a_mert_done.json'
    if done_126a.is_file():
        # same cell layout as 126a analyze(): one cell per (variant, system); judge() rule
        a_results = {}
        for var in A.variants():
            vname, vkind = var['name'], var['kind']
            for sname in SYSTEMS:
                tsv = OUT / '126a' / vname / sname / 'mert_cosim.tsv'
                if not tsv.is_file():
                    continue
                c = np.array([v['mert_cosim'] for v in read_cosim_tsv(tsv, dims=['mert_cosim']).values()])
                d = c - 1.0
                cell = {'kind': vkind, 'n': int(len(c)), 'mean_cosim': float(c.mean()),
                        'p10_cosim': float(np.percentile(c, 10)),
                        'tie': float(np.mean(np.abs(d) <= m_mert)), 'worse': float(np.mean(d < -m_mert))}
                if vkind == 'tie':
                    cell['pass'] = cell['tie'] >= PASS
                elif vkind == 'lose':
                    cell['pass'] = cell['worse'] >= PASS
                a_results.setdefault(vname, {})[sname] = cell
        results['126a'] = a_results
        log(f'  126a: {len(a_results)} variants x {len(SYSTEMS)} systems')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))
    log(f'  wrote {REPORT}')


# ── preflight / validate ───────────────────────────────────────────────────────

def check_disk():
    import shutil
    free = shutil.disk_usage(str(OUT.parent)).free
    if free < MIN_FREE:
        log(f'ERROR: only {free/2**30:.1f} GB free (need 5 GB)')
        return False
    return True


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    if not check_disk():
        sys.exit(75)

    from huggingface_hub import try_to_load_from_cache
    r = try_to_load_from_cache(MERT_REPO, 'config.json', revision=MERT_REVISION)
    if not r:
        log(f'ERROR: MERT {MERT_REVISION} not in HF cache')
        sys.exit(75)

    for sname, sfolder in SYSTEMS.items():
        clips = list((R115 / sfolder / 'lvl30/audio').glob('*.flac'))[:5]
        if not clips:
            log(f'ERROR: no R115 clips for {sname}')
            sys.exit(75)

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
    for key in ('S_mert', 'S_muq', '126b', '126c'):
        if key not in d:
            log(f'ERROR: malformed report (missing {key})')
            sys.exit(1)
    # Verify no None cells in judged (lose/tie) rows
    for setname in ('126b', '126c'):
        for case, cell in d.get(setname, {}).items():
            if cell.get('kind') in ('lose', 'tie'):
                if cell.get('mert') is None or cell.get('muq') is None:
                    log(f'ERROR: {setname}/{case} has None cell (kind={cell["kind"]})')
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
    compute_s_muq(mdl_muq, device)

    del mdl_muq
    import torch as _torch
    _torch.cuda.empty_cache()

    score_126a(feat_mert, mdl_mert, device)
    analyze()


if __name__ == '__main__':
    main()
