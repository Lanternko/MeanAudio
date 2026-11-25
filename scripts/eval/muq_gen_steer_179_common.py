"""179 shared constants and helpers: arms, prompt subset, paths, small IO."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.home() / 'nvme_experiment_artifacts/meanaudio/muq_gen_steer_179'
RESULTS_DIR = ROOT / 'docs/experiments/results'
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
CKPT = Path('/home/kojiek/exps_nvme/phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s14159265_stage2_200000/'
            'phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s14159265_stage2_200000_ema_final.pth')

N_PROMPTS = 1000
SEED = 42
CFG = 3.0

FID = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
BRIGHT_NEG = 'bright, crisp, sparkling treble, airy high frequencies'
BRIGHT_VEC = 'bright, crisp, sparkling treble'
DARK = 'warm, dark, muffled sound'

# kind=vanilla goes through the unpatched eval.py; kind=steer monkeypatches
# meanaudio.eval_utils.generate_mf with the four-branch combination.
ARMS = {
    'A0': dict(kind='vanilla', neg=FID, suffix='', lam=None),
    'A1': dict(kind='vanilla', neg=BRIGHT_NEG, suffix='', lam=None),
    'A2': dict(kind='vanilla', neg=f'{FID}, {BRIGHT_NEG}', suffix='', lam=None),
    'A3': dict(kind='vanilla', neg=FID, suffix=DARK, lam=None),
    'A4': dict(kind='steer', neg=FID, suffix='', lam=1.0),
    'A5': dict(kind='steer', neg=FID, suffix='', lam=2.0),
    # harness consistency check only (lambda=0 must reproduce A0 up to float order)
    'A4_L0': dict(kind='steer', neg=FID, suffix='', lam=0.0),
}
MAIN_ARMS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
FAMILY = ('A1', 'A2', 'A3', 'A4', 'A5')   # Holm family (A0 is the reference)
FULL_ARMS = MAIN_ARMS
SMOKE_ARMS = MAIN_ARMS + ('A4_L0',)

METRICS = ['muq_A1_MI', 'muq_A1_TA', 'muq_A3a_MI', 'muq_A3a_TA',
           'aes_PQ', 'aes_CE', 'aes_CU', 'aes_PC', 'clap',
           'lufs', 'band_ratio_db', 'centroid_hz']


def write_subset_tsv(dst, n=N_PROMPTS, src=MC_TSV):
    """First n csv RECORDS (not lines: five MusicCaps captions span two lines)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f, delimiter='\t')
        fields = rd.fieldnames
        rows = [r for _, r in zip(range(n), rd)]
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return dst, len(rows)


def read_rows(tsv):
    with open(tsv, newline='', encoding='utf-8') as f:
        return [(r['id'], r['caption']) for r in csv.DictReader(f, delimiter='\t')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + '\n',
                 encoding='utf-8')
