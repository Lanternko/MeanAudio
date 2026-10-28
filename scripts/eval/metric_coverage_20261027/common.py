#!/usr/bin/env python3
"""Shared definitions for the metric x intervention coverage jobs 161-177.

Design: docs/experiments/metric_coverage_completion_20261027.md (preregistered 2026-10-27).

Everything that more than one of {score_coverage.py, judge_coverage.py, the 161/162 prep,
the unit tests, make_contracts.py} needs lives here: paths, the model registry, the
protocol item model (126a is synthesised from the sample manifest, 126b/126c/134 come
from the existing stimuli/items.json), S lookup, the win/tie/lose judge primitives and
the TSV IO used by every shard.

Nothing here re-implements a model: the adapters import scripts/eval/bench_r2/score_r2.py,
newmetrics_*.py, metric_win_tie_126d.py, metric_win_tie_133.py and mir_metrics.py.
"""
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path('/home/kojiek/MeanAudio')
EV = Path('/home/kojiek/eval_output_nvme')
R115 = EV / 'ttm_metrics_115_20261009'
OUT_ROOT = EV / 'metric_coverage_20261027'
DOCS = REPO / 'docs/experiments'

sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A  # noqa: E402  (variants(), SYSTEMS, transform(), rng_for())

PROTOCOLS = ('126a', '126b', '126c', '134')
PROTOCOL_126B = EV / 'metric_win_tie_126b_20261016'
PROTOCOL_126C = EV / 'metric_win_tie_126c_20261019'
PROTOCOL_134 = EV / 'speed_pitch_extension_134_20261020'
PROTOCOL_126A = EV / 'metric_win_tie_126a_20261016'
PROTOCOL_126D = EV / 'metric_win_tie_126d_20261020'
PROTOCOL_133 = EV / 'metric_win_tie_133_20261020'
ITEMS = {
    '126b': PROTOCOL_126B / 'stimuli/items.json',
    '126c': PROTOCOL_126C / 'stimuli/items.json',
    '134': PROTOCOL_134 / 'stimuli/items.json',
}
MUSICCAPS_TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
SAMPLE_SEED = 20261027
SAMPLE_N = 512
PAM_SAMPLE_N = 256
M_FRAC = 0.10
PASS = 0.80
AMBIG_BAND = 0.04
BOOTSTRAP_B = 200

SYSTEMS = tuple(A.SYSTEMS)                 # real, ours_cfg3neg, ours_cfg0, musicgen, musicldm
SYSTEM_DIRS = dict(A.SYSTEMS)
VARIANTS = A.variants()
VARIANT_NAMES = [v['name'] for v in VARIANTS]
VARIANT_KIND = {v['name']: v['kind'] for v in VARIANTS}

# ── model registry ────────────────────────────────────────────────────────────
# cols:    score columns in the per-clip TSV, in output order.
# signs:   +1 higher-is-better, -1 lower-is-better.  Judging multiplies by the sign
#          (the original 126b/134 TSVs already store MIR sign-flipped; here the raw
#          reading is stored and flipped at judge time, same result).
# directed: whether the original protocol defines a lose direction for the column
#          (126a/126b DIRECTED omits AES PC; reproduce that, do not invent one).
# ref:     reference-based metric (embedding cosine against the base clip).
# emb_col: stage-A column for ref rows (the raw embedding, not the cosine).
MODELS = {
    'muqeval_A1': {'cols': ['mi', 'ta'], 'signs': [1, 1], 'directed': [True, True],
                   'adapter': 'muqeval_A1', 'label': 'MuQ-Eval A1'},
    'muqeval_A3a': {'cols': ['mi', 'ta'], 'signs': [1, 1], 'directed': [True, True],
                    'adapter': 'muqeval_A3a', 'label': 'MuQ-Eval A3a'},
    'cmirm': {'cols': ['quality', 'alignment'], 'signs': [1, 1], 'directed': [True, True],
              'adapter': 'cmirm', 'label': 'CMI-RM'},
    'pam': {'cols': ['pam'], 'signs': [1], 'directed': [True], 'adapter': 'pam', 'label': 'PAM'},
    'singmos': {'cols': ['singmos'], 'signs': [1], 'directed': [True], 'adapter': 'singmos',
                'label': 'SingMOS'},
    'songeval': {'cols': ['se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus'],
                 'signs': [1] * 5, 'directed': [True] * 5, 'adapter': 'songeval',
                 'label': 'SongEval'},
    'mert': {'cols': ['mert_cosim'], 'signs': [1], 'directed': [True], 'adapter': 'mert',
             'ref': True, 'emb_col': 'mert_emb', 'label': 'MERT-ref'},
    'muq': {'cols': ['muq_cosim'], 'signs': [1], 'directed': [True], 'adapter': 'muq',
            'ref': True, 'emb_col': 'muq_emb', 'label': 'MuQ-embed-ref'},
    'mir': {'cols': ['pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'chroma_entropy'],
            'signs': [1, -1, 1, -1], 'directed': [True] * 4, 'adapter': 'mir',
            'label': 'MIR'},
}


def emb_col(model):
    """Stage-A column of a reference-based row (defaults to the score column)."""
    m = MODELS[model]
    return m.get('emb_col') or m['cols'][0]

# Rows of the coverage matrix (22 rows, §1.2).  (row id, model key, dim, label)
MATRIX_ROWS = [
    ('aes_PQ', None, 'PQ', 'AES PQ'), ('aes_CE', None, 'CE', 'AES CE'),
    ('aes_CU', None, 'CU', 'AES CU'), ('aes_PC', None, 'PC', 'AES PC'),
    ('pam', 'pam', 'pam', 'PAM'), ('meva', None, 'meva', 'MEva'),
    ('muqeval_A1', 'muqeval_A1', None, 'MuQ-Eval A1 (MI/TA)'),
    ('muqeval_A3a', 'muqeval_A3a', None, 'MuQ-Eval A3a (MI/TA)'),
    ('cmirm', 'cmirm', None, 'CMI-RM (quality/alignment)'),
    ('se_coh', None, 'se_coh', 'SongEval se_coh'), ('se_mem', None, 'se_mem', 'SongEval se_mem'),
    ('se_nat', None, 'se_nat', 'SongEval se_nat'), ('se_cla', None, 'se_cla', 'SongEval se_cla'),
    ('se_mus', None, 'se_mus', 'SongEval se_mus'),
    ('singmos', 'singmos', 'singmos', 'SingMOS predictor'),
    ('mert', 'mert', 'mert_cosim', 'MERT-ref (cosine)'),
    ('muq', 'muq', 'muq_cosim', 'MuQ-embed-ref (cosine)'),
    ('clap', None, 'clap', 'CLAP'),
    ('mir_pulse', 'mir', 'pulse_clarity', 'MIR pulse_clarity'),
    ('mir_ibi', 'mir', 'ibi_cv', 'MIR ibi_cv'),
    ('mir_key', 'mir', 'key_cnn_conf', 'MIR key_cnn_conf'),
    ('mir_chroma', 'mir', 'chroma_entropy', 'MIR chroma_entropy'),
]
EXISTING = {  # matrix row -> source key in EXISTING_SOURCES (None = new model)
    'aes_PQ': 'aes', 'aes_CE': 'aes', 'aes_CU': 'aes', 'aes_PC': 'aes', 'meva': 'meva',
    'se_coh': 'songeval', 'se_mem': 'songeval', 'se_nat': 'songeval', 'se_cla': 'songeval',
    'se_mus': 'songeval', 'mert': 'mert', 'muq': 'muq', 'clap': 'clap', 'mir_pulse': 'mir',
    'mir_ibi': 'mir', 'mir_key': 'mir', 'mir_chroma': 'mir',
}

# ── job registry (§6) ─────────────────────────────────────────────────────────
ALL_SYSTEMS = list(SYSTEMS)
JOBS = {
    161: {'kind': 'prep', 'title': 'CPU prep: sample manifest, agreement, 126a audio cache, rec/floor backfill'},
    162: {'kind': 'clean_s', 'models': ['muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'singmos', 'muq'],
          'title': 'clean S on 5 TTM systems'},
    163: {'kind': 'score', 'model': 'muqeval_A1', 'protocols': ['126b', '126c', '134']},
    164: {'kind': 'score', 'model': 'muqeval_A1', 'protocols': ['126a'], 'sample_n': 512},
    165: {'kind': 'score', 'model': 'muqeval_A3a', 'protocols': ['126b', '126c', '134']},
    166: {'kind': 'score', 'model': 'muqeval_A3a', 'protocols': ['126a'], 'sample_n': 512},
    167: {'kind': 'score', 'model': 'cmirm', 'protocols': ['126b', '126c', '134']},
    168: {'kind': 'score', 'model': 'cmirm', 'protocols': ['126a'], 'sample_n': 512},
    169: {'kind': 'score', 'model': 'pam', 'protocols': ['126b', '126c', '134']},
    170: {'kind': 'score', 'model': 'pam', 'protocols': ['126a'], 'sample_n': 256,
          'systems': ['real', 'ours_cfg3neg', 'ours_cfg0']},
    171: {'kind': 'score', 'model': 'pam', 'protocols': ['126a'], 'sample_n': 256,
          'systems': ['musicgen', 'musicldm']},
    172: {'kind': 'score', 'model': 'singmos', 'protocols': ['126b', '126c', '134']},
    173: {'kind': 'score', 'model': 'singmos', 'protocols': ['126a'], 'sample_n': 512},
    174: {'kind': 'score', 'model': 'muq', 'protocols': ['126a', '134'], 'sample_n': 512},
    175: {'kind': 'multi', 'specs': [{'model': 'songeval', 'protocols': ['134']},
                                     {'model': 'mert', 'protocols': ['134']}]},
    176: {'kind': 'mir'},
    177: {'kind': 'judge'},
}
# jobs that only need the committed sample manifest / items and no 162 output
NO_S = {161, 162, 175, 176, 177}
MODEL_OF_JOB = {j: s.get('model') for j, s in JOBS.items()}


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tree_sha256(root, skip_parts=('__pycache__', '.git')):
    """Content hash of a directory tree (relative path + file sha256, sorted)."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_dir() or any(part in p.parts for part in skip_parts):
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(b'\0')
        h.update(sha256_file(p).encode())
        h.update(b'\0')
    return h.hexdigest()


def written_at():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


# ── protocols ─────────────────────────────────────────────────────────────────
def load_items_file(protocol):
    d = json.loads(ITEMS[protocol].read_text())
    return d if isinstance(d, list) else d.get('items', d)


def item_domain(model, protocol, it):
    """§5.3 domain marker; '' = in-domain."""
    if protocol == '126b':
        src = it.get('source', '')
        if src == 'floor':
            return 'OOD-synthetic'
        if src.startswith('midi_'):
            if model == 'songeval':
                return 'OOD-10s-instrumental'
            if model == 'singmos':
                return 'OOD-instrumental'
            return 'OOD-MIDI' if model != 'mir' else ''
        if model == 'singmos' and src != 'poor_sing':
            return 'OOD-instrumental'
        return ''
    if protocol == '126c':
        if model == 'pam':
            return 'OOD-stems'
        if model == 'singmos':
            return 'OOD-instrumental'
        return ''
    if protocol == '134':
        if model == 'singmos':
            return 'OOD-instrumental'
        if model == 'songeval':
            return 'OOD-10s-instrumental'
        if it.get('kind') == 'lose' and _is_extreme_134(it):
            return 'OOD-extreme'
        return ''
    return ''


def _is_extreme_134(it):
    axis, v = it.get('axis'), it.get('value')
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    if axis == 'tempo':
        return v <= 0.80 or v >= 1.25
    if axis == 'pitch':
        return abs(v) >= 5
    return False


def sample_manifest_path():
    return DOCS / 'metric_coverage_sample_manifest_20261027.json'


def build_sample_manifest():
    """Deterministic (§4.4): one permutation, shared by every variant and system.
    N=512 is the first half of N=1024 (prefix rule), PAM's 256 the first half of 512.
    Pool = A.items() (the 5,131-clip intersection)."""
    pool = [cid for cid, _ in A.items()]
    rng = A.np.random.default_rng(SAMPLE_SEED)
    perm = rng.permutation(len(pool))
    clips_1024 = [pool[int(i)] for i in perm[:1024]]
    clips = clips_1024[:SAMPLE_N]
    return {
        'document_kind': 'metric_coverage_sample_manifest_v1',
        'seed': SAMPLE_SEED, 'n_pool': len(pool),
        'pool_sha256': hashlib.sha256('\n'.join(pool).encode()).hexdigest(),
        'clip_ids': clips,
        'clip_ids_1024': clips_1024,
        'pam_clip_ids': clips[:PAM_SAMPLE_N],
        'systems': list(SYSTEMS), 'variants': list(VARIANT_NAMES),
        'note': 'preregistered artifact; job 161 regenerates it and fails on mismatch; '
                '§4.4 escalation uses 1024 = permutation[:1024] (512 is its prefix)',
    }


def clips_for_n(sample_n):
    man = read_json(sample_manifest_path())
    if not man:
        raise SystemExit(f'sample manifest missing: {sample_manifest_path()}')
    key = {PAM_SAMPLE_N: 'pam_clip_ids', SAMPLE_N: 'clip_ids', 1024: 'clip_ids_1024'}.get(sample_n)
    if key is None:
        raise SystemExit(f'unsupported sample_n {sample_n}')
    clips = man.get(key) or []
    if len(clips) != sample_n:
        raise SystemExit(f'sample manifest {key} has {len(clips)} clips, job asks {sample_n}')
    return clips


def sample_agreement():
    return read_json(OUT_ROOT / 'sample_agreement_20261027.json') or {}


def effective_sample_n(sample_n):
    """§4.4 pre-fixed escalation: 512 -> 1024 when 161's agreement is below the doc floor."""
    esc = (sample_agreement().get('escalation') or {})
    if esc.get('to_1024'):
        return {PAM_SAMPLE_N: SAMPLE_N, SAMPLE_N: 1024}.get(sample_n, sample_n), True
    return sample_n, False


def require_full_126a():
    """§4.4: when N=1024 agreement is still < 0.9716 every cell needs a full rerun."""
    return bool((sample_agreement().get('escalation') or {}).get('require_full_126a'))


def items_126a(model, sample_n, systems=None):
    """Unit-major item list: 53 variants x requested systems x sample_n clips."""
    clips = clips_for_n(sample_n)
    systems = systems or list(SYSTEMS)
    out = []
    for v in VARIANTS:
        for s in systems:
            for cid in clips:
                out.append({
                    'key': f"{v['name']}__{s}__{cid}",
                    'base': '' if v['name'] == 'clean' else f'clean__{s}__{cid}',
                    'variant': v['name'], 'system': s, 'clip_id': cid, 'kind': v['kind'],
                    'source': s, 'case': v['name'], 'axis': '', 'impl': '', 'value': None,
                    'over_fs': 0,
                    'path': str(OUT_ROOT / 'audio_cache' / s / v['name'] / f'{cid}.flac'),
                    'cache_path': str(OUT_ROOT / 'audio_cache' / s / v['name'] / f'{cid}.flac'),
                    'src_path': str(R115 / SYSTEM_DIRS[s] / 'lvl30/audio' / f'{cid}.flac'),
                })
    return out


def protocol_items(protocol, model, sample_n=None, systems=None):
    if protocol == '126a':
        if not sample_n:
            raise SystemExit('126a needs --sample-n / job sample_n')
        return items_126a(model, sample_n, systems)
    out = []
    for it in load_items_file(protocol):
        row = {k: it.get(k) for k in
               ('key', 'base', 'source', 'kind', 'case', 'axis', 'impl', 'value', 'path',
                'over_fs', 'system', 'caption')}
        row['variant'] = ''
        row['clip_id'] = it.get('key', '')
        row['kind'] = it.get('kind') or it.get('ladder') or ''
        row['domain'] = item_domain(model, protocol, it)
        out.append(row)
    return out


def scoreable(protocol, model, it):
    """MERT/MuQ floor cells are N/A (§1.4): they are skipped, never self-paired."""
    if protocol in ('126b', 'floor') and model in ('mert', 'muq') and it.get('kind') == 'floor':
        return False
    return True


# ── shards ────────────────────────────────────────────────────────────────────
def plan_shards(protocol, items):
    """[(shard_idx, [items], name)].  126a: one shard per (variant, system) unit;
    other protocols: contiguous chunks of ~4000 items.  The caller picks shard i of n."""
    if protocol == '126a':
        units = []
        order = {}
        for it in items:
            u = (it['variant'], it['system'])
            if u not in order:
                order[u] = []
                units.append(u)
            order[u].append(it)
        groups = [order[u] for u in units]
        names = [f'{u[0]}__{u[1]}' for u in units]
    else:
        step = 4000
        groups = [items[i:i + step] for i in range(0, len(items), step)]
        names = [f'{i * step:06d}' for i in range(len(groups))]
    return [(i, groups[i], names[i]) for i in range(len(groups))]


def pick_shards(shards, shard='0/1'):
    i, n = (int(x) for x in str(shard).split('/'))
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f'bad --shard {shard}')
    return [s for s in shards if s[0] % n == i]


def shard_file(out_dir, protocol, model, idx, name=''):
    """126a shards are named by unit (`variant__system`) so that job 170 and job 171,
    which share model=pam/protocol=126a, can never collide on a file name."""
    tag = f'__{name}' if protocol == '126a' else f'_shard{idx:03d}'
    return Path(out_dir) / f'{protocol}_{model}{tag}.tsv'


def err_file(out_dir, protocol, model, idx, name=''):
    tag = f'__{name}' if protocol == '126a' else f'_shard{idx:03d}'
    return Path(out_dir) / f'{protocol}_{model}{tag}.err.tsv'


def full_file(out_dir, protocol, model, name):
    return Path(out_dir) / f'{protocol}_{model}__full__{name}.tsv'


def emb_file(out_dir, protocol, model, idx, name='', full=False):
    tag = (f'__full__{name}' if full else (f'__{name}' if protocol == '126a'
                                          else f'_shard{idx:03d}'))
    return Path(out_dir) / 'emb' / f'emb_{protocol}_{model}{tag}.tsv'


def final_tsv_files(out_dir, protocol, model, full=False):
    """Finished score TSVs (never .err.tsv).  126a is unit-named, the rest shard-named."""
    out = []
    for pat in (f'{protocol}_{model}__*.tsv', f'{protocol}_{model}_shard*.tsv'):
        for f in sorted(Path(out_dir).glob(pat)):
            if f.name.endswith('.err.tsv'):
                continue
            if full != ('__full__' in f.name):
                continue
            if f not in out:
                out.append(f)
    return out


TSV_META = ['key', 'base', 'source', 'kind', 'case', 'axis', 'impl', 'value', 'domain',
            'over_fs']


# ── TSV io (append under flock, atomic rewrite on shard completion) ───────────
def read_tsv(path):
    rows = {}
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return rows
    with open(p, newline='') as f:
        rd = csv.DictReader(f, delimiter='\t')
        for r in rd:
            if r.get('key'):
                rows[r['key']] = r          # later duplicate wins (idempotent append)
    return rows


def _atomic_rewrite(path, header, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, '') for c in header])
    os.replace(tmp, path)


def append_rows(path, header, rows):
    """Append under an exclusive lock; fsync so a killed job loses at most one batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    with open(lock, 'a') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        new = not path.exists() or path.stat().st_size == 0
        with open(path, 'a', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            if new:
                w.writerow(header)
            for r in rows:
                w.writerow([r.get(c, '') for c in header])
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lf, fcntl.LOCK_UN)


def compact_shard(path):
    """Rewrite a finished shard with one row per key (§3.4 idempotent write)."""
    rows = read_tsv(path)
    if not rows:
        return
    with open(path, newline='') as f:
        header = next(csv.reader(f, delimiter='\t'))
    _atomic_rewrite(path, header, list(rows.values()))


def fmt(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return '' if not math.isfinite(v) else repr(v)
    return str(v)


def parse_score_tsv(path, cols):
    """-> {key: {col: float}}; blank = NaN (dropped)."""
    out = {}
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return out
    with open(p, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            vals = {}
            for c in cols:
                try:
                    v = float(r[c])
                    if math.isfinite(v):
                        vals[c] = v
                except (KeyError, TypeError, ValueError):
                    pass
            if vals:
                out[r['key']] = vals
    return out


# ── judging primitives (§5) ───────────────────────────────────────────────────
def pair_deltas(scores_case, scores_base, key, base, cols, signs):
    """d = sign * (case - base) for the columns both rows carry."""
    a, b = scores_case.get(key), scores_base.get(base)
    if not a or not b:
        return {}
    return {c: s * (a[c] - b[c]) for c, s in zip(cols, signs) if c in a and c in b}


def judge_deltas(d, m, kind, directed=True):
    """One unit/cell: tie pass = share |d|<=m; lose pass = share d < -m.
    kind: 'tie' or 'lose'; §5.1 says 'soft' (and 126c 'descriptive', 134 'ctl'/'transition',
    the 126a 'clean' ref) is descriptive only -> pass stays None.
    directed=False => no pass verdict (AES PC loses have no direction).
    Returns dict with n, tie, worse, better, pass (None when not applicable)."""
    import numpy as np
    d = np.asarray([x for x in d if math.isfinite(x)], dtype=float)
    if not len(d):
        return {'n': 0}
    tie = float(np.mean(np.abs(d) <= m))
    worse = float(np.mean(d < -m))
    better = float(np.mean(d > m))
    out = {'n': int(len(d)), 'tie': tie, 'worse': worse, 'better': better,
           'mean_d': float(d.mean()), 'sd_d': float(d.std())}
    k = 'tie' if kind == 'tie' else ('lose' if kind == 'lose' else None)
    if k == 'tie':
        out['pass'] = tie >= PASS
    elif k == 'lose':
        out['pass'] = (worse >= PASS) if directed else None
    else:
        out['pass'] = None
    out['kind'] = kind
    if out['sd_d'] == 0.0:
        out['degenerate'] = True
    return out


def aggregate_units(unit_results, kind):
    """Combine per-unit verdicts of one metric x protocol into the matrix cell.
    judged units only; returns pass counts and a label per §5.4."""
    judged = [u for u in unit_results if u.get('pass') is not None]
    n_judged = len(judged)
    n_pass = sum(1 for u in judged if u['pass'])
    degenerate = any(u.get('degenerate') for u in judged)
    if not n_judged:
        return {'label': 'n/a', 'n_units': 0, 'n_pass': 0, 'kind': kind}
    rate = n_pass / n_judged
    if degenerate:
        label = 'degenerate'
    elif rate >= PASS:
        label = 'pass'
    elif n_pass:
        label = 'partial'
    else:
        label = 'fail'
    return {'label': label, 'n_units': n_judged, 'n_pass': n_pass, 'pass_rate': rate,
            'kind': kind}


# ── S values ──────────────────────────────────────────────────────────────────
S_FILE = OUT_ROOT / 's_values'
S_126A_SUMMARY = DOCS / 'results/metric_win_tie_126a_summary.json'
S_126B_SUMMARY = PROTOCOL_126B / 'summary.json'
S_126C_SUMMARY = PROTOCOL_126C / 'summary.json'
S_134_SUMMARY = PROTOCOL_134 / 'summary.json'
S_126D_SUMMARY = DOCS / 'results/metric_win_tie_126d_summary.json'
S_133_SUMMARY = DOCS / 'results/metric_win_tie_133_summary.json'


def s_values(model, protocol):
    """S for (model, protocol) per §5.1.  New models: job 162's s_values_<model>.json
    (126a measured; 126b/126c/134 share the 126a S unless the protocol has its own).
    Existing metrics: the original protocol summary.  Returns {} when unmeasured."""
    if model in ('mert', 'muq', 'songeval') and protocol in ('126b', '126c'):
        # 133/126d measured 126b/126c with the 126a S (reference/embedding rows)
        protocol = '126a'
    if model in ('mir',):
        return _mir_s(protocol)
    if model in ('aes', 'meva', 'clap'):
        if protocol == '134':
            return read_json(S_134_SUMMARY, {}).get('protocol', {}).get('S', {})
        return read_json(S_126A_SUMMARY, {}).get('protocol', {}).get('S', {})
    if model == 'songeval':
        return read_json(S_126D_SUMMARY, {}).get('S_songeval', {})
    if model in ('mert', 'muq'):
        d = read_json(S_133_SUMMARY, {})
        return {'mert_cosim': d.get('S_mert')} if model == 'mert' else {'muq_cosim': d.get('S_muq')}
    f = S_FILE / f's_values_{model}.json'
    d = read_json(f)
    if not d:
        return {}
    if protocol in ('126b', '126c', '134') and d.get('S_by_protocol', {}).get(protocol):
        return d['S_by_protocol'][protocol]
    return d.get('S', {})


def _mir_s(protocol):
    if protocol == '126a':
        return read_json(S_FILE / 's_values_mir.json', {}).get('S_126a', {})
    if protocol == '134':
        d = read_json(S_134_SUMMARY, {}).get('protocol', {}).get('S', {})
        extra = read_json(S_FILE / 's_values_mir.json', {}).get('S_134_extra', {})
        return {**d, **extra}
    if protocol == '126b':
        return read_json(S_126B_SUMMARY, {}).get('protocol', {}).get('S', {})
    if protocol == '126c':
        return read_json(S_126C_SUMMARY, {}).get('protocol', {}).get('S', {})
    return {}


def m_for(model, protocol, dim):
    s = s_values(model, protocol).get(dim)
    if isinstance(s, (int, float)) and math.isfinite(s) and s > 0:
        return M_FRAC * s, s
    return None, s
