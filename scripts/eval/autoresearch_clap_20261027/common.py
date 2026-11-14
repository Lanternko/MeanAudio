"""Shared paths, split handling and IO for the autoresearch inference-CLAP line.

Design: docs/experiments/autoresearch_inference_clap_20261027.md
Prereg: docs/experiments/autoresearch_clap_20261027_prereg.json (sha-bound by contracts)

Everything here is CPU-only; the GPU work lives in render.py / trial.py /
fad_cache.py and is always launched as a subprocess so each step reclaims VRAM.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]              # /home/kojiek/MeanAudio
HERE = Path(__file__).resolve().parent
SCRATCH = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/autoresearch_clap_20261027')
PY = '/home/kojiek/venvs/dac/bin/python'
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
REF_DIR = Path('/mnt/HDD/kojiek/musiccaps_reference')
PREREG = ROOT / 'docs/experiments/autoresearch_clap_20261027_prereg.json'
EVAL_ROOT = Path.home() / 'eval_output_nvme'

SPLIT_RECORDS = {
    'dev256': (0, 256),
    'dev1024': (0, 1024),
    'heldout1024': (1024, 2048),
    'rest3473': (2048, None),
    'full5521': (0, None),
}
# splits whose trial rows are a contiguous prefix of the canonical full run (paired x0)
PAIRED_SPLITS = ('dev256', 'dev1024', 'full5521')

SEEDS = ('s14159265', 's16180339', 's27182818')
EXP = {s: f'phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_{s}' for s in SEEDS}
CHECKPOINTS = {
    s: Path.home() / 'exps_nvme' / f'{EXP[s]}_stage2_200000' / f'{EXP[s]}_stage2_200000_ema_final.pth'
    for s in SEEDS
}
CHECKPOINT_SHA256 = {
    's14159265': '7cc14d093480ebf1728f1d72c4ec545a5a0885c7fd3f71751c9db32285b813b1',
    's16180339': '997077fbc3863cbf89db04ddcda738b7524ba8171d644240ba5e3e423088a483',
    's27182818': '4f98eefc7de465549a1c49896a10511007060ea3864a1a5262f4a33f0a52e053',
}
CFG_KEYS = ('cfg0', 'cfg3_neg')
CANONICAL_LVL30_EXPECTED = {          # prereg objective.baseline_numbers_lvl30_full5521
    's14159265': {'cfg0': 0.1927, 'cfg3_neg': 0.2325},
    's16180339': {'cfg0': 0.1949, 'cfg3_neg': 0.2365},
    's27182818': {'cfg0': 0.1922, 'cfg3_neg': 0.2375},
}
CANONICAL_FAD_EXPECTED = {            # prereg objective.baseline_fad_full5521_1924pairs
    's14159265': {'cfg0': 3.2640, 'cfg3_neg': 5.0836},
    's16180339': {'cfg0': 3.1927, 'cfg3_neg': 5.4385},
    's27182818': {'cfg0': 3.3426, 'cfg3_neg': 4.9986},
}

THRESHOLDS = {                        # prereg thresholds
    'dev': 0.005, 'heldout': 0.004, 'combo_over_single': 0.002,
    'fad': 0.30, 'pq': -0.10,
}
SIGMA_SEED_BY_N = {256: 0.0014, 1024: 0.0007, 5521: 0.0003}


# ── hashes / json ────────────────────────────────────────
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def atomic_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def free_bytes(path=Path.home()) -> int:
    import shutil
    return shutil.disk_usage(str(path)).free


# ── prereg ───────────────────────────────────────────────
def load_prereg() -> dict:
    return json.loads(PREREG.read_text())


def prompts() -> dict:
    return dict(load_prereg()['prompts'])


# ── splits ───────────────────────────────────────────────
def rows_for_tsv(tsv):
    """[(id, caption), ...] in file order, csv-record aware (same parsing as eval.py)."""
    import csv
    out = []
    with open(tsv, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            out.append((row['id'], row['caption']))
    ids = [i for i, _ in out]
    if len(set(ids)) != len(ids):
        raise SystemExit(f'[FAIL] duplicate ids in {tsv}')
    return out


def split_tsv(scratch, split) -> Path:
    """Subset TSV under <scratch>/tsv; the full split is the canonical source TSV."""
    split = str(split)
    if split == 'full5521':
        return MC_TSV
    path = Path(scratch) / 'tsv' / f'{split}.tsv'
    if not path.is_file():
        raise SystemExit(f'[FAIL] missing split TSV {path} (run split_subsets.py)')
    return path


def split_rows(scratch, split):
    return rows_for_tsv(split_tsv(scratch, split))


def split_ids(scratch, split):
    return [i for i, _ in split_rows(scratch, split)]


# ── plans / trials ───────────────────────────────────────
def load_plan(path) -> dict:
    plan = read_json(path)
    if not plan:
        raise SystemExit(f'[FAIL] unreadable plan {path}')
    for key in ('round', 'split', 'checkpoints', 'seeds', 'configs'):
        if key not in plan:
            raise SystemExit(f'[FAIL] plan {path} missing {key}')
    if plan['split'] not in SPLIT_RECORDS:
        raise SystemExit(f'[FAIL] unknown split {plan["split"]}')
    return plan


def effective_seed(plan, cfg) -> int:
    """A single-seed plan lets a control config carry its own seed (C1/C2)."""
    seeds = [int(s) for s in plan['seeds']]
    if len(seeds) == 1:
        return int(cfg.get('seed', seeds[0]))
    return seeds[0]


def trial_dir(scratch, plan, cfg, ckpt, seed) -> Path:
    base = Path(scratch) / 'trials' / plan['round']
    cross = len(plan['checkpoints']) > 1 or len(plan['seeds']) > 1
    name = cfg['id'] if not cross else f"{cfg['id']}__{ckpt}__s{seed}"
    return base / name


def config_sha(cfg, ckpt, seed, split) -> str:
    payload = {'config': cfg, 'checkpoint': ckpt, 'seed': int(seed), 'split': split,
               'objective': 'lvl30 clap (launch_clap HTSAT-base, batch 1)'}
    return sha256_text(canonical_json(payload))


def baseline_cfgkey(cfg) -> str:
    """Gate baseline: the stock cell of the same checkpoint on the same side of CFG 1."""
    return 'cfg0' if float(cfg.get('cfg', 0.0)) < 1.0 else 'cfg3_neg'


def baseline_path(scratch, split, ckpt, cfgkey) -> Path:
    return Path(scratch) / 'baselines' / split / ckpt / cfgkey / 'baseline.json'


def load_baseline(scratch, split, ckpt, cfgkey):
    path = baseline_path(scratch, split, ckpt, cfgkey)
    rec = read_json(path)
    if not rec:
        raise SystemExit(f'[FAIL] missing baseline {path} (run baselines.py --split {split})')
    return rec


# ── canonical cells ──────────────────────────────────────
def canonical_cell(ckpt, cfgkey, kind) -> Path:
    """kind: 'raw' | 'lvl30' | 'fad'."""
    if cfgkey not in CFG_KEYS:
        raise SystemExit(f'[FAIL] unknown cfg key {cfgkey}')
    tag = {'raw': '', 'lvl30': '_lvl30', 'fad': '_fad'}[kind]
    name = f'{EXP[ckpt]}_mc_mf25_{cfgkey}{tag}'
    return EVAL_ROOT / name


def cell_metrics(cell) -> dict:
    """<cell>/<cell>/metrics.json -> the 'metrics' dict."""
    rec = read_json(Path(cell) / Path(cell).name / 'metrics.json')
    if not rec:
        raise SystemExit(f'[FAIL] missing metrics.json under {cell}')
    return rec['metrics']


def read_per_clip(path):
    """per_clip.tsv -> list of dicts (floats where parseable)."""
    import csv
    out = []
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rec = {'id': row['id']}
            for k, v in row.items():
                if k == 'id':
                    continue
                try:
                    rec[k] = float(v) if v != '' else None
                except ValueError:
                    rec[k] = v
            out.append(rec)
    return out


def read_gain_status(path):
    import csv
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def sigma_floor(n) -> float:
    """0.0003 * sqrt(5521/n), nearest registered n (prereg noise_floor.sigma_seed_by_n)."""
    if n in SIGMA_SEED_BY_N:
        return SIGMA_SEED_BY_N[n]
    return 0.0003 * math.sqrt(5521.0 / n)
