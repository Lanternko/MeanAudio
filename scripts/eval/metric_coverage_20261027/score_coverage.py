#!/usr/bin/env python3
"""Shared scoring driver for the metric x intervention coverage jobs 161-177.

Design: docs/experiments/metric_coverage_completion_20261027.md §3, §4.4, §6.

  # queue jobs read their spec + bindings from the contract
  python -I score_coverage.py --job 164 [--preflight|--validate-only|--dry-run]
  # smoke / manual use (§3.2 primitive; <= 8 clips per model for the GPU smoke)
  python -I score_coverage.py --protocol 126a --model pam --clips 8 --out DIR
  python -I score_coverage.py --protocol 126b --model pam --clips 8 --out DIR

Per (model, protocol): one TSV per shard, append-only under flock, completed keys are
skipped on restart (§3.4).  126a is sampled (§4.4) and every unit whose sampled pass
rate lands in 0.80 +/- 0.04 is rerun on the full 5,131 clips inside the same job.
Reference-based models (mert, muq) write embeddings first and cosine against the base
clip afterwards, so a base may live in another shard.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402
import adapters  # noqa: E402

DRIVER = Path(__file__).resolve()


# ── contract ──────────────────────────────────────────────────────────────────
def contract_path(job):
    if os.environ.get('GPU_QUEUE_CONTRACT'):
        return Path(os.environ['GPU_QUEUE_CONTRACT'])
    return C.DOCS / f'metric_coverage_{job}_contract.json'


def load_contract(job):
    p = contract_path(job)
    c = C.read_json(p)
    if not c:
        raise SystemExit(f'contract missing or unreadable: {p}')
    return c


def job_specs(job):
    spec = C.JOBS[job]
    if spec['kind'] == 'multi':
        return [(s['model'], s['protocols'], spec.get('sample_n'), spec.get('systems'))
                for s in spec['specs']]
    return [(spec.get('model'), spec.get('protocols', []), spec.get('sample_n'),
             spec.get('systems'))]


def out_dir(model, protocol):
    return C.OUT_ROOT / model / protocol


# ── prerequisites (§6: fail immediately, never wait) ─────────────────────────
def require(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print('PREREQUISITE MISSING:', file=sys.stderr)
        for m in missing:
            print('  ' + m, file=sys.stderr)
        raise SystemExit(4)


def s_file(model):
    return C.S_FILE / f's_values_{model}.json'


def s_usable(model):
    """162's S must exist and carry a positive, finite value for every column."""
    d = C.read_json(s_file(model))
    if not d:
        return False
    S = d.get('S') or {}
    return all(isinstance(S.get(c), (int, float)) and math.isfinite(S[c]) and S[c] > 0
               for c in C.MODELS[model]['cols'])


def prerequisites(job):
    spec = C.JOBS[job]
    kind = spec['kind']
    if kind == 'prep':
        return [C.R115 / d / 'lvl30/audio' for d in C.SYSTEM_DIRS.values()] + \
               list(C.ITEMS.values()) + [Path(C.MUSICCAPS_TSV), C.PROTOCOL_126A / 'units']
    if kind == 'clean_s':
        return [C.OUT_ROOT / 'prep_161.json', C.R115 / 'selection.json']
    if kind == 'mir':
        return [C.OUT_ROOT / 'prep_161.json', C.OUT_ROOT / 'audio_cache_manifest.json']
    if kind == 'judge':
        return [C.OUT_ROOT / 'jobs' / f'job_done_{j}.json' for j in range(163, 177)] + \
               [C.OUT_ROOT / 'prep_161.json', C.S_FILE / 's_values_mir.json']
    need = [C.OUT_ROOT / 'prep_161.json', C.OUT_ROOT / 'audio_cache_manifest.json']
    for model, protocols, _n, _s in job_specs(job):
        if model in ('mert', 'muq', 'songeval'):
            continue  # S already in the 126d/133 summaries
        need.append(s_file(model))
    if '126a' in [p for _, ps, _, _ in job_specs(job) for p in ps]:
        need.append(C.sample_manifest_path())
    if job == 171:
        # §6.1: only one GPU -> 170 then 171, strictly serial
        need.append(C.OUT_ROOT / 'jobs' / 'job_done_170.json')
    return need


# ── preflight (§3.5) ─────────────────────────────────────────────────────────
def check_pins(models=None):
    pins = C.read_json(C.DOCS / 'metric_coverage_model_pins_20261027.json')
    if not pins:
        return ['model pin file missing: docs/experiments/metric_coverage_model_pins_20261027.json']
    errs = []
    full_hash = os.environ.get('METRIC_COVERAGE_PIN_FULL_HASH') == '1'
    for m, p in (pins.get('models') or {}).items():
        if models and m not in models:
            continue
        for d in p.get('hf_snapshot_dirs', []):
            if not Path(d).exists():
                errs.append(f'{m}: HF snapshot missing {d}')
        sizes = p.get('file_size') or {}
        for f, sha in (p.get('file_sha256') or {}).items():
            if not Path(f).exists():
                errs.append(f'{m}: weight missing {f}')
                continue
            if Path(f).stat().st_size != sizes.get(f, Path(f).stat().st_size):
                errs.append(f'{m}: weight size mismatch {f}')
            # full sha is the slow check (13 GB); do it once in job 161 and for
            # small files, otherwise trust the recorded size
            if full_hash or Path(f).stat().st_size < 256 * 2 ** 20:
                if C.sha256_file(f) != sha:
                    errs.append(f'{m}: weight sha mismatch {f}')
        vt = p.get('vendored_tree')
        if vt:
            tree_sha = p.get('vendored_tree_sha256')
            if not Path(vt).exists():
                errs.append(f'{m}: vendored tree missing {vt}')
            elif tree_sha and C.tree_sha256(vt) != tree_sha:
                errs.append(f'{m}: vendored tree hash mismatch')
        clone = p.get('clone')
        if clone and p.get('clone_commit'):
            if not Path(clone).is_dir():
                errs.append(f'{m}: clone missing {clone}')
            else:
                import subprocess
                got = subprocess.run(['git', '-C', clone, 'rev-parse', 'HEAD'],
                                     capture_output=True, text=True).stdout.strip()
                if got != p['clone_commit']:
                    errs.append(f'{m}: clone commit {got} != {p["clone_commit"]}')
    return errs


def check_contract_shas(c):
    errs = []
    for k, v in (c.get('bindings') or {}).items():
        if k.endswith('_sha256') and isinstance(v, str) and v and not v.startswith('pending:'):
            f = c['bindings'].get(k[:-7])
            if f and Path(f).is_file() and C.sha256_file(f) != v:
                errs.append(f'{k}: sha mismatch for {f}')
    for f, v in (c.get('code_bindings') or {}).items():
        p = Path(f) if Path(f).is_absolute() else C.REPO / f
        if p.is_file() and C.sha256_file(p) != v:
            errs.append(f'code binding sha mismatch: {f}')
    return errs


def check_disk(protocols):
    free = shutil.disk_usage('/home/kojiek').free
    need = 60 * 2 ** 30 if '126a' in protocols else 10 * 2 ** 30
    if free < need:
        return [f'disk: {free / 2**30:.1f} GB free < {need / 2**30:.0f} GB']
    return []


def check_manifest():
    errs = []
    p = C.sample_manifest_path()
    if not p.is_file():
        return [f'sample manifest missing: {p}']
    man = C.read_json(p)
    if C.sha256_json(man) != C.sha256_json(C.build_sample_manifest()):
        errs.append('sample manifest does not regenerate to the committed file (seed 20261027)')
    if len(man.get('clip_ids', [])) != C.SAMPLE_N:
        errs.append('sample manifest clip_ids != 512')
    if len(man.get('clip_ids_1024', [])) != 1024 or \
            man['clip_ids_1024'][:C.SAMPLE_N] != man.get('clip_ids'):
        errs.append('sample manifest clip_ids_1024 is not a 512-prefixed 1024 list')
    return errs


def check_stimuli(protocols, model=None):
    errs = []
    for proto in protocols:
        if proto == '126a':
            continue
        items = C.load_items_file(proto)
        if not items:
            errs.append(f'{proto}: items.json empty')
            continue
        bad = [it['path'] for it in items[:3] if not Path(it['path']).exists()]
        if bad:
            errs.append(f'{proto}: first item path missing {bad[0]}')
    if '126a' in protocols:
        for s, d in C.SYSTEM_DIRS.items():
            p = C.R115 / d / 'lvl30/audio'
            if not p.is_dir() or not list(p.glob('*.flac')):
                errs.append(f'126a: R115 audio missing {p}')
                continue
            files = sorted(p.glob('*.flac'))
            import soundfile as sf
            for f in (files[0], files[-1]):
                try:
                    if sf.info(str(f)).frames <= 0:
                        errs.append(f'126a: undecodable {f}')
                except Exception as e:
                    errs.append(f'126a: {f}: {e}')
        nvar = len(C.VARIANTS)
        if nvar != 53:
            errs.append(f'variant list is {nvar}, expected 53')
    return errs


def preflight(job):
    spec = C.JOBS[job]
    errs = []
    if job not in (161, 177):
        models = {m for m, _ps, _n, _s in job_specs(job) if m}
        errs += check_pins(models or None)
    try:
        c = load_contract(job)
        errs += check_contract_shas(c)
    except SystemExit:
        errs.append(f'contract missing: {contract_path(job)}')
    protocols = [p for _, ps, _, _ in job_specs(job) for p in ps]
    if spec['kind'] in ('prep', 'clean_s', 'mir'):
        protocols = ['126a']
    errs += check_disk(protocols)
    if spec['kind'] == 'prep' and job == 161:
        import prep161
        errs += prep161.check_inputs()
    else:
        errs += check_manifest()
        errs += check_stimuli(protocols)
    try:
        require(prerequisites(job))
    except SystemExit:
        errs.append('prerequisite outputs missing (see PREREQUISITE MISSING above)')
    for model, _ps, _n, _s in job_specs(job):
        if model and job not in C.NO_S and not C.MODELS[model].get('ref') \
                and model not in ('songeval',):
            if not s_file(model).is_file():
                errs.append(f'{model}: S file missing ({s_file(model)})')
            elif not s_usable(model):
                print(f'warning: {model} S is unusable; job scores but judge will say '
                      f'inconclusive (§8.1)', file=sys.stderr)
    if errs:
        print('PREFLIGHT FAILED', file=sys.stderr)
        for e in errs:
            print('  - ' + e, file=sys.stderr)
        return 2
    print(f'preflight ok: job {job} protocols={protocols}')
    return 0


# ── 126a materialisation (on-the-fly transform, identical to 126a.make_one) ───
def render_126a(src, dst, system, variant):
    import soundfile as sf
    var = C.A.VARIANTS[variant]
    x, sr = sf.read(src, dtype='float64')
    if getattr(x, 'ndim', 1) > 1:
        x = x.mean(axis=1)
    y, _ = C.A.transform(x, var, C.A.rng_for(system, variant, Path(src).stem), int(sr))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, int(sr), format='FLAC', subtype=var['subtype'])


def resolve_paths(items, tmp_root):
    """Point item['path'] at a playable file; returns the number rendered on the fly."""
    made = 0
    for it in items:
        cache = it.get('cache_path')
        if cache and Path(cache).exists():
            it['path'] = cache
            continue
        if it.get('src_path'):
            dst = Path(tmp_root) / f"{it['variant']}__{it['system']}" / f"{it['clip_id']}.flac"
            if not dst.exists():
                render_126a(it['src_path'], dst, it['system'], it['variant'])
                made += 1
            it['path'] = str(dst)
    return made


# ── scoring primitives ───────────────────────────────────────────────────────
def _rows_for(items, res, cols):
    rows, errs = [], []
    for it, r in zip(items, res):
        if isinstance(r, Exception):
            errs.append({'key': it['key'], 'error': repr(r)[:300]})
            continue
        vals = {c: r.get(c) for c in cols}
        if not all(isinstance(v, float) and math.isfinite(v) for v in vals.values()):
            errs.append({'key': it['key'], 'error': 'missing or nonfinite'})
            continue
        meta = {k: C.fmt(it.get(k)) for k in C.TSV_META}
        rows.append({**meta, **{c: C.fmt(vals[c]) for c in cols}})
    return rows, errs


def score_group(model, protocol, items, out_path, scorer, tmp_root, done_keys=None):
    """Score one shard into `out_path` (append-only).  Returns (ok, err, rendered)."""
    cols = list(scorer.cols)
    done = dict(done_keys or {})
    done.update(C.read_tsv(out_path))
    todo = [it for it in items if it['key'] not in done and C.scoreable(protocol, model, it)]
    if not todo:
        C.compact_shard(out_path)
        return 0, 0, 0
    rendered = resolve_paths(todo, tmp_root)
    header = C.TSV_META + cols
    t0, n_ok, n_bad = time.time(), 0, 0
    for i in range(0, len(todo), scorer.batch):
        batch = todo[i:i + scorer.batch]
        rows, errs = _rows_for(batch, scorer.score_batch(batch), cols)
        if rows:
            C.append_rows(out_path, header, rows)
            n_ok += len(rows)
        if errs:
            C.append_rows(out_path.with_name(out_path.name + '.err.tsv'), ['key', 'error'], errs)
            n_bad += len(errs)
        if (i // scorer.batch) % 10 == 0:
            C.log(f'  {model}/{protocol} {out_path.name} {min(i + scorer.batch, len(todo))}/'
                  f'{len(todo)} ok={n_ok} err={n_bad} {time.time() - t0:.0f}s')
    C.compact_shard(out_path)
    return n_ok, n_bad, rendered


def emb_stage(model, protocol, items, out_path, scorer, tmp_root):
    """Stage A for reference-based rows: store the raw embedding under its key."""
    col = C.emb_col(model)
    done = C.read_tsv(out_path)
    todo = [it for it in items if it['key'] not in done and C.scoreable(protocol, model, it)]
    if not todo:
        C.compact_shard(out_path)
        return 0, 0, 0
    rendered = resolve_paths(todo, tmp_root)
    t0, n_ok, n_bad = time.time(), 0, 0
    for i in range(0, len(todo), scorer.batch):
        batch = todo[i:i + scorer.batch]
        rows, errs = [], []
        for it, r in zip(batch, scorer.score_batch(batch)):
            if isinstance(r, Exception) or col not in r:
                errs.append({'key': it['key'], 'error': repr(r)[:300]})
            else:
                rows.append({'key': it['key'], col: r[col]})
        if rows:
            C.append_rows(out_path, ['key', col], rows)
            n_ok += len(rows)
        if errs:
            C.append_rows(out_path.with_name(out_path.name + '.err.tsv'), ['key', 'error'], errs)
            n_bad += len(errs)
        if (i // scorer.batch) % 10 == 0:
            C.log(f'  {model}/{protocol} {out_path.name} {min(i + scorer.batch, len(todo))}/'
                  f'{len(todo)} ok={n_ok} err={n_bad} {time.time() - t0:.0f}s')
    C.compact_shard(out_path)
    return n_ok, n_bad, rendered


def load_embeddings(emb_dir, protocol, model):
    col = C.emb_col(model)
    out = {}
    for f in sorted(Path(emb_dir).glob(f'emb_{protocol}_{model}*.tsv')):
        for k, r in C.read_tsv(f).items():
            if col in r:
                out[k] = r[col]
    return out


def cosim_stage(model, protocol, items, out_path, embs, dim):
    """Stage B: raw cosine against the row's base (133 convention), written as <col>_cosim.

    A row without a base (the clean / reference row) is its own base, so its cosine is 1.
    Judging subtracts base from case exactly as it does for the 133 TSVs, which yields
    d = cosim(case, base) - 1 (§3.3).
    """
    col = C.MODELS[model]['cols'][0]
    done = C.read_tsv(out_path)
    rows, n_missing = [], 0
    for it in items:
        if it['key'] in done or not C.scoreable(protocol, model, it):
            continue
        base = it.get('base') or it['key']
        if it['key'] not in embs or base not in embs:
            n_missing += 1
            continue
        d = adapters.cosim(embs[it['key']], embs[base], dim)
        meta = {k: C.fmt(it.get(k)) for k in C.TSV_META}
        rows.append({**meta, col: C.fmt(d)})
    if rows:
        C.append_rows(out_path, C.TSV_META + [col], rows)
    C.compact_shard(out_path)
    return len(rows), n_missing


def embedding_dim(model):
    return 768 if model == 'mert' else 1024


def run_protocol(model, protocol, items, out_dir, scorer, shard='0/1', limit=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'emb').mkdir(exist_ok=True)
    shards = C.pick_shards(C.plan_shards(protocol, items), shard)
    if limit:
        shards = [(i, g[:limit], nm) for i, g, nm in shards[:1]]
    ref = bool(C.MODELS[model].get('ref'))
    tmp_root = Path(tempfile.mkdtemp(prefix='cov_score_', dir='/home/kojiek/tmp'))
    n_ok = n_bad = n_rendered = 0
    t0 = time.time()
    try:
        if ref:
            for idx, group, name in shards:
                a, b, r = emb_stage(model, protocol, group,
                                    C.emb_file(out_dir, protocol, model, idx, name),
                                    scorer, tmp_root)
                n_ok += a
                n_bad += b
                n_rendered += r
            embs = load_embeddings(out_dir / 'emb', protocol, model)
            n_ok = n_bad = 0
            for idx, group, name in shards:
                a, b = cosim_stage(model, protocol, group,
                                   C.shard_file(out_dir, protocol, model, idx, name),
                                   embs, embedding_dim(model))
                n_ok += a
                n_bad += b
        else:
            for idx, group, name in shards:
                a, b, r = score_group(model, protocol, group,
                                      C.shard_file(out_dir, protocol, model, idx, name),
                                      scorer, tmp_root)
                n_ok += a
                n_bad += b
                n_rendered += r
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    dt = time.time() - t0
    return {'n_ok': n_ok, 'n_err': n_bad, 'n_shards': len(shards), 'n_rendered': n_rendered,
            'elapsed_s': round(dt, 1), 'clips_per_s': round(n_ok / dt, 2) if dt else None}


# ── §4.4 stage 2: ambiguous units get the full 5,131 clips ────────────────────
def sampled_scores(out_dir, protocol, model):
    cols = C.MODELS[model]['cols']
    scores = {}
    for f in C.final_tsv_files(out_dir, protocol, model, full=False):
        scores.update(C.parse_score_tsv(f, cols))
    return scores


def unit_verdicts(model, protocol, items, out_dir):
    """Sampled pass rate per judged unit -> [{'variant','system','rate','ambiguous'}]."""
    cols = C.MODELS[model]['cols']
    signs = C.MODELS[model]['signs']
    directed = C.MODELS[model]['directed']
    scores = sampled_scores(out_dir, protocol, model)
    systems = sorted({it['system'] for it in items})
    out = []
    for v in C.VARIANTS:
        if v['kind'] not in ('tie', 'lose'):
            continue
        for s in systems:
            ds = {c: [] for c in cols}
            for it in items:
                if it['variant'] != v['name'] or it['system'] != s:
                    continue
                a, b = scores.get(it['key']), scores.get(it.get('base') or '')
                if not a or not b:
                    continue
                for c, sg in zip(cols, signs):
                    if c in a and c in b:
                        ds[c].append(sg * (a[c] - b[c]))
            rates, amb = {}, False
            for c, dr in zip(cols, directed):
                ms, _S = C.m_for(model, '126a', c)
                if ms is None or not ds[c]:
                    rates[c] = None
                    continue
                r = C.judge_deltas(ds[c], ms, v['kind'], dr)
                rate = r.get('tie') if v['kind'] == 'tie' else r.get('worse')
                rates[c] = rate
                if rate is not None and abs(rate - C.PASS) <= C.AMBIG_BAND:
                    amb = True
            out.append({'variant': v['name'], 'system': s, 'kind': v['kind'],
                        'rates': rates, 'ambiguous': amb})
    return out


def full_units(verdicts, require_full=False):
    if require_full:
        return [(v['variant'], v['system']) for v in verdicts]
    return [(v['variant'], v['system']) for v in verdicts if v['ambiguous']]


def full_items_for(units):
    """Unit items on the full 5,131-clip pool + the clean base of every used system."""
    all_ids = [cid for cid, _ in C.A.items()]
    need = list(units) + [('clean', s) for s in sorted({u[1] for u in units})]
    seen, out = set(), []
    for vname, system in need:
        if (vname, system) in seen:
            continue
        seen.add((vname, system))
        var = C.A.VARIANTS[vname]
        for cid in all_ids:
            out.append({
                'key': f'{vname}__{system}__{cid}',
                'base': '' if vname == 'clean' else f'clean__{system}__{cid}',
                'variant': vname, 'system': system, 'clip_id': cid, 'kind': var['kind'],
                'source': system, 'case': vname, 'axis': '', 'impl': '', 'value': None,
                'over_fs': 0,
                'path': str(C.OUT_ROOT / 'audio_cache' / system / vname / f'{cid}.flac'),
                'cache_path': str(C.OUT_ROOT / 'audio_cache' / system / vname / f'{cid}.flac'),
                'src_path': str(C.R115 / C.SYSTEM_DIRS[system] / 'lvl30' / 'audio' /
                                f'{cid}.flac'),
            })
    return out


def full_rerun(model, protocol, out_dir, scorer, sample_n, job=None, require_full=False):
    """Score full 5,131 clips for every ambiguous unit plus the clean base of its system."""
    out_dir = Path(out_dir)
    tag = f'_{job}' if job else ''
    items = C.protocol_items('126a', model, sample_n)
    verdicts = unit_verdicts(model, protocol, items, out_dir)
    amb = [v for v in verdicts if v['ambiguous']]
    units = full_units(verdicts, require_full)
    C.write_json_atomic(out_dir / f'ambiguous_units{tag}.json',
                        {'document_kind': 'metric_coverage_ambiguous_v1',
                         'written_at': C.written_at(), 'job': job, 'sample_n': sample_n,
                         'band': C.AMBIG_BAND, 'units': verdicts,
                         'n_ambiguous': len(amb),
                         'full_rerun_units': [list(u) for u in units],
                         'require_full_126a': bool(require_full),
                         'overflow_gt_48': len(amb) > 48})
    if not units:
        C.write_json_atomic(out_dir / f'full_rerun{tag}.json',
                            {'units': [], 'written_at': C.written_at(), 'job': job})
        return {'n_full_units': 0, 'n_ok': 0, 'n_err': 0}
    ref = bool(C.MODELS[model].get('ref'))
    tmp_root = Path(tempfile.mkdtemp(prefix='cov_full_', dir='/home/kojiek/tmp'))
    n_ok = n_bad = 0
    try:
        for vname, system in units:
            name = f'{vname}__{system}'
            all_items = full_items_for([(vname, system)])
            group = [it for it in all_items if it['variant'] == vname]
            bases = [it for it in all_items if it['variant'] == 'clean']
            if ref:
                for items, path in ((group, C.emb_file(out_dir, protocol, model, 0, name, True)),
                                    (bases, C.emb_file(out_dir, protocol, model, 0,
                                                       f'clean__{system}', True))):
                    _a, b, _r = emb_stage(model, protocol, items, path, scorer, tmp_root)
                    n_bad += b
                embs = load_embeddings(out_dir / 'emb', protocol, model)
                a, b = cosim_stage(model, protocol, group,
                                   C.full_file(out_dir, protocol, model, name),
                                   embs, embedding_dim(model))
            else:
                a, b, _r = score_group(model, protocol, group,
                                       C.full_file(out_dir, protocol, model, name),
                                       scorer, tmp_root)
            n_ok += a
            n_bad += b
            C.log(f'  full rerun {name}: {len(group)} clips ({a} ok, {b} err)')
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    C.write_json_atomic(out_dir / f'full_rerun{tag}.json',
                        {'units': [list(u) for u in units], 'n_ok': n_ok, 'n_err': n_bad,
                         'written_at': C.written_at(), 'job': job})
    return {'n_full_units': len(units), 'n_ok': n_ok, 'n_err': n_bad}


# ── job bodies ───────────────────────────────────────────────────────────────
def run_scoring(job, dry_run=False):
    spec = C.JOBS[job]
    summary = {'job': job, 'started_at': C.written_at(), 'specs': []}
    for model, protocols, sample_n, systems in job_specs(job):
        scorer = None
        for protocol in protocols:
            t0 = time.time()
            eff_n, escalated = (C.effective_sample_n(sample_n)
                                if protocol == '126a' and sample_n else (sample_n, False))
            items = C.protocol_items(protocol, model, eff_n, systems)
            od = out_dir(model, protocol)
            od.mkdir(parents=True, exist_ok=True)
            info = {'model': model, 'protocol': protocol, 'n_items': len(items),
                    'sample_n': sample_n, 'sample_n_effective': eff_n,
                    'sample_n_escalated': escalated,
                    'systems': systems or list(C.SYSTEMS)}
            C.log(f'job {job} {model}/{protocol}: {len(items)} items -> {od} '
                  f'(N={eff_n}{" escalated" if escalated else ""})')
            if dry_run:
                info['dry_run'] = True
                summary['specs'].append(info)
                continue
            scorer = scorer or adapters.build(model)
            info.update(run_protocol(model, protocol, items, od, scorer))
            if protocol == '126a':
                info.update(full_rerun(model, protocol, od, scorer, eff_n, job=job,
                                       require_full=C.require_full_126a()))
            run = {'document_kind': 'metric_coverage_run_v1', 'job': job, 'model': model,
                   'protocol': protocol, 'n_items': len(items), 'sample_n': sample_n,
                   'sample_n_effective': eff_n, 'systems': systems or list(C.SYSTEMS),
                   'manifest_sha256': C.sha256_json(items),
                   'sample_manifest_sha256': (C.sha256_file(C.sample_manifest_path())
                                              if protocol == '126a' else None),
                   'items_file': str(C.ITEMS[protocol]) if protocol in C.ITEMS else None,
                   'items_sha256': (C.sha256_file(C.ITEMS[protocol])
                                    if protocol in C.ITEMS else None),
                   's_values_sha256': (C.sha256_file(s_file(model))
                                       if s_file(model).is_file() else None),
                   'driver_sha256': C.sha256_file(DRIVER),
                   'adapter_cols': list(scorer.cols),
                   **info, 'elapsed_s': round(time.time() - t0, 1),
                   'written_at': C.written_at()}
            C.write_json_atomic(od / f'run_{job}.json', run)
            summary['specs'].append(info)
        del scorer
    summary['finished_at'] = C.written_at()
    C.write_json_atomic(C.OUT_ROOT / 'jobs' / f'job_done_{job}.json', summary)
    return 0


def expected_keys(model, protocol, items):
    ref = bool(C.MODELS[model].get('ref'))
    return [it['key'] for it in items if C.scoreable(protocol, model, it)
            and (not ref or it.get('base'))]


def validate(job):
    """Every expected key has a finite score row or an error row; bases covered."""
    spec = C.JOBS[job]
    errs = []
    kind = spec['kind']
    if kind == 'prep':
        need = [C.OUT_ROOT / 'prep_161.json', C.OUT_ROOT / 'sample_agreement_20261027.json',
                C.OUT_ROOT / 'audio_cache_manifest.json',
                C.sample_manifest_path()]
        need += [C.OUT_ROOT / 'backfill' / f'rec_floor_{m}.json'
                 for m in ('songeval', 'mert', 'muq')]
        for p in need:
            if not Path(p).is_file():
                errs.append(f'prep output missing: {p}')
    elif kind == 'clean_s':
        import prep162
        errs += prep162.validate()
    elif kind == 'mir':
        import mir_prep
        errs += mir_prep.validate()
    elif kind == 'judge':
        for p in (C.OUT_ROOT / 'coverage_summary.json',
                  C.OUT_ROOT / 'coverage_matrix.md',
                  C.OUT_ROOT / 'coverage_matrix.html',
                  C.OUT_ROOT / 'deck_data.json'):
            if not Path(p).is_file():
                errs.append(f'judge output missing: {p}')
    else:
        for model, protocols, sample_n, systems in job_specs(job):
            eff_n, _ = (C.effective_sample_n(sample_n)
                        if '126a' in protocols and sample_n else (sample_n, False))
            for protocol in protocols:
                items = C.protocol_items(protocol, model, eff_n, systems)
                od = out_dir(model, protocol)
                seen = {}
                for f in C.final_tsv_files(od, protocol, model, full=False):
                    seen.update(C.parse_score_tsv(f, C.MODELS[model]['cols']))
                missing = [k for k in expected_keys(model, protocol, items) if k not in seen]
                if missing:
                    errs.append(f'{model}/{protocol}: {len(missing)} keys without a score '
                                f'(first: {missing[:3]})')
                fr = C.read_json(Path(od) / f'full_rerun_{job}.json')
                if fr:
                    for vname, system in fr.get('units', []):
                        f = C.full_file(od, protocol, model, f'{vname}__{system}')
                        got = len(C.parse_score_tsv(f, C.MODELS[model]['cols']))
                        if got < 5131:
                            errs.append(f'{model}/{protocol}: full rerun {vname}/{system} '
                                        f'has {got}/5131')
                if C.MODELS[model].get('ref'):
                    embs = load_embeddings(Path(od) / 'emb', protocol, model)
                    mb = [it['key'] for it in items
                          if it.get('base') and it['key'] not in embs]
                    if mb:
                        errs.append(f'{model}/{protocol}: {len(mb)} case keys without '
                                    f'embeddings')
    if errs:
        print('VALIDATE FAILED', file=sys.stderr)
        for e in errs:
            print('  - ' + e, file=sys.stderr)
        return 2
    print('validate ok')
    return 0


# ── smoke items (<= 8 clips per model; §4.5 pilot) ───────────────────────────
def smoke_items(protocol, model, n, system=None):
    if protocol == '126a':
        system = system or 'real'
        clips = C.clips_for_n(C.SAMPLE_N)[:n]
        tie = next(v['name'] for v in C.VARIANTS if v['kind'] == 'tie')
        lose = next(v['name'] for v in C.VARIANTS if v['kind'] == 'lose')
        out = []
        for vname in ('clean', tie, lose):
            var = C.A.VARIANTS[vname]
            for cid in clips:
                out.append({
                    'key': f'{vname}__{system}__{cid}',
                    'base': '' if vname == 'clean' else f'clean__{system}__{cid}',
                    'variant': vname, 'system': system, 'clip_id': cid, 'kind': var['kind'],
                    'source': system, 'case': vname, 'axis': '', 'impl': '', 'value': None,
                    'over_fs': 0,
                    'path': str(C.OUT_ROOT / 'audio_cache' / system / vname / f'{cid}.flac'),
                    'cache_path': str(C.OUT_ROOT / 'audio_cache' / system / vname / f'{cid}.flac'),
                    'src_path': str(C.R115 / C.SYSTEM_DIRS[system] / 'lvl30/audio' / f'{cid}.flac'),
                })
        return out
    items = C.protocol_items(protocol, model)
    by_key = {it['key']: it for it in items}
    cases, seen = [], set()
    for it in items:
        if len(cases) >= n:
            break
        if it.get('kind') in ('tie', 'lose') and it.get('base') in by_key:
            cases.append(it)
            seen.update((it['key'], it['base']))
    if not cases:
        raise SystemExit(f'{protocol}: no tie/lose case with a base in items.json')
    return [it for it in items if it['key'] in seen]


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=int)
    ap.add_argument('--protocol', choices=C.PROTOCOLS)
    ap.add_argument('--model', choices=sorted(C.MODELS))
    ap.add_argument('--sample-n', type=int)
    ap.add_argument('--out')
    ap.add_argument('--clips', type=int, help='smoke: N clips per model (<=8 for the pilot)')
    ap.add_argument('--system', help='smoke 126a: one TTM system')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (C.OUT_ROOT / 'jobs').mkdir(parents=True, exist_ok=True)

    if a.job:
        spec = C.JOBS[a.job]
        if a.preflight:
            return preflight(a.job)
        if a.validate_only:
            return validate(a.job)
        if spec['kind'] == 'prep':
            import prep161
            return prep161.run(dry_run=a.dry_run)
        if spec['kind'] == 'clean_s':
            import prep162
            return prep162.run(dry_run=a.dry_run)
        if spec['kind'] == 'mir':
            import mir_prep
            return mir_prep.run(dry_run=a.dry_run)
        if spec['kind'] == 'judge':
            import judge_coverage
            return judge_coverage.run_cli(dry_run=a.dry_run)
        return run_scoring(a.job, a.dry_run)

    if not (a.protocol and a.model):
        ap.error('pass --job N, or --protocol and --model')
    if a.clips:
        items = smoke_items(a.protocol, a.model, a.clips, a.system)
        od = Path(a.out) if a.out else C.OUT_ROOT / 'smoke' / a.model / a.protocol
    else:
        items = C.protocol_items(a.protocol, a.model, a.sample_n, [a.system] if a.system else None)
        od = Path(a.out) if a.out else C.OUT_ROOT / a.model / a.protocol
    if a.preflight:
        errs = check_manifest() + check_disk([a.protocol]) + check_stimuli([a.protocol], a.model)
        for e in errs:
            print(e, file=sys.stderr)
        return 2 if errs else 0
    od.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        print(f'{len(items)} items -> {od}')
        return 0
    scorer = adapters.build(a.model)
    info = run_protocol(a.model, a.protocol, items, od, scorer, a.shard, a.limit)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
