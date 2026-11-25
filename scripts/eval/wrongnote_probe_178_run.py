#!/usr/bin/env python
"""178 orchestrator: stimuli -> features -> rule baselines -> analysis, resumable per stage.

  python scripts/eval/wrongnote_probe_178_run.py --root <dir> [--workers 20]
  python scripts/eval/wrongnote_probe_178_run.py --root <dir> --preflight
  python scripts/eval/wrongnote_probe_178_run.py --root <dir> --validate-only

Every stage skips what is already on disk (FLAC present, feature chunk present, rule row
present), so a restart after a pause only redoes the missing work. Writes <root>/run.json.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrongnote_probe_178_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = '/home/kojiek/venvs/dac/bin/python'
MIR_PY = '/home/kojiek/venvs/mir/bin/python'
MADMOM_PY = '/home/kojiek/venvs/madmom/bin/python'
MIN_FREE = 16 * 2 ** 30


def sh(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'\n=== {" ".join(map(str, cmd))} @ {time.strftime("%F %T")} ===\n')
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    return {'cmd': [str(c) for c in cmd], 'rc': rc, 'seconds': round(time.time() - t, 1)}


def meta_n(path):
    meta = C.read_json(path)
    return int(meta['n']) if meta and meta.get('n') else -1


def expected_clips(root):
    st = C.read_json(Path(root) / 'stimuli.json')
    return int(st['clips']) if st and st.get('clips') else None


def stages(root, workers):
    root = Path(root)
    st = [
        ('stimuli', [PY, HERE / 'wrongnote_probe_178_stimuli.py', '--root', root,
                     '--workers', workers]),
        ('features_mert', [PY, HERE / 'wrongnote_probe_178_features.py', '--root', root, '--what', 'mert']),
        ('features_aes', [PY, HERE / 'wrongnote_probe_178_features.py', '--root', root, '--what', 'aes']),
        ('rules_chroma', [MIR_PY, HERE / 'wrongnote_probe_178_rules.py', '--mode', 'chroma',
                          '--root', root, '--workers', workers]),
        ('rules_madmom', [MADMOM_PY, HERE / 'wrongnote_probe_178_rules.py', '--mode', 'madmom',
                          '--root', root, '--workers', max(4, workers // 2)]),
        ('analyze', [PY, HERE / 'wrongnote_probe_178_analyze.py', '--root', root]),
    ]
    return st


def stage_done(name, root):
    root = Path(root)
    n = expected_clips(root)
    if name == 'stimuli':
        man = root / 'manifest.tsv'
        return bool(n) and man.is_file() and len(C.read_tsv(man)) == n
    if name == 'features_mert':
        return bool(n) and meta_n(root / 'features/mert/meta.json') == n and \
            meta_n(root / 'features/mert_random/meta.json') == n
    if name == 'features_aes':
        return bool(n) and meta_n(root / 'features/aes/meta.json') == n
    if name in ('rules_chroma', 'rules_madmom'):
        info = C.read_json(root / 'features' / f'rule_{name.split("_")[1]}.json')
        return bool(n) and bool(info) and info.get('missing') == 0 and info.get('scored') == n
    return False


def preflight(root):
    problems = []
    for p in ('/mnt/seagate/kojiek_data/maestro_v3_midi/maestro-v3.0.0/maestro-v3.0.0.csv',
              str(C.ROOT.parent), PY, MIR_PY, MADMOM_PY):
        if not Path(p).exists():
            problems.append(f'missing {p}')
    sys.path.insert(0, str(HERE))
    import midi_realpiece_stimuli as stim
    for bank, p in stim.BANKS.items():
        if not Path(p).is_file():
            problems.append(f'missing soundfont {bank}: {p}')
    free = os.statvfs('/home/kojiek').f_bavail * os.statvfs('/home/kojiek').f_frsize
    if free < MIN_FREE:
        problems.append(f'NVMe free {free} < {MIN_FREE}')
    if problems:
        print('[preflight] FAIL: ' + '; '.join(problems), flush=True)
        return 2
    print(f'[preflight] ok: root={root} free={free / 2 ** 30:.0f} GiB', flush=True)
    return 0


def validate(root):
    root = Path(root)
    problems = []
    n = expected_clips(root)
    if not n:
        problems.append('stimuli.json missing/empty')
    man = root / 'manifest.tsv'
    if not man.is_file():
        problems.append('manifest.tsv missing')
    else:
        rows = C.read_tsv(man)
        if n and len(rows) != n:
            problems.append(f'manifest {len(rows)} rows != {n}')
        miss = [r['path'] for r in rows if not Path(r['path']).is_file()]
        if miss:
            problems.append(f'{len(miss)} audio files missing')
    for name in ('mert', 'mert_random', 'aes'):
        if meta_n(root / 'features' / name / 'meta.json') != n:
            problems.append(f'features/{name} incomplete')
    s = C.read_json(root / 'summary.json')
    if not s:
        problems.append('summary.json missing')
    else:
        auc = (s.get('criteria') or {}).get('auc')
        if auc is None or not (0.0 <= float(auc) <= 1.0):
            problems.append(f'primary AUC invalid: {auc}')
    if not (root / 'results.md').is_file():
        problems.append('results.md missing')
    if problems:
        print('[validate] FAIL: ' + '; '.join(problems), flush=True)
        return 2
    print('[validate] ok', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(C.ROOT))
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--force', action='store_true', help='run every stage even if its marker exists')
    a = ap.parse_args()
    if a.preflight:
        return preflight(a.root)
    if a.validate_only:
        return validate(a.root)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    log = root / 'logs' / 'run.log'
    report = {'root': str(root), 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'stages': {}}
    t0 = time.time()
    for name, cmd in stages(root, a.workers):
        if not a.force and stage_done(name, root):
            print(f'[run] {name}: already complete, skipped', flush=True)
            report['stages'][name] = {'skipped': True}
            continue
        print(f'[run] {name}: starting', flush=True)
        r = sh(cmd, log)
        report['stages'][name] = r
        print(f'[run] {name}: rc={r["rc"]} {r["seconds"]} s', flush=True)
        if r['rc']:
            report['status'] = 'failed'
            report['failed_stage'] = name
            C.write_json(root / 'run.json', report)
            return r['rc']
    rc = validate(root)
    report['status'] = 'complete' if rc == 0 else 'invalid'
    report['seconds'] = round(time.time() - t0, 1)
    man = root / 'manifest.tsv'
    report['manifest_sha256'] = hashlib.sha256(man.read_bytes()).hexdigest() if man.is_file() else None
    report['clips'] = expected_clips(root)
    for stage, rel in (('stimuli', 'stimuli.json'), ('rules_chroma', 'features/rule_chroma.json'),
                       ('rules_madmom', 'features/rule_madmom.json')):
        info = C.read_json(root / rel)
        if info:
            report.setdefault('timings', {})[stage] = info
    C.write_json(root / 'run.json', report)
    print(f'[run] {"complete" if rc == 0 else "INVALID"} in {time.time() - t0:.0f} s', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
