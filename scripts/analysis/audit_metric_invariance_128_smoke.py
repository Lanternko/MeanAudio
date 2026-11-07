#!/usr/bin/env python3
"""Reproduce 128's failed smoke diagnosis. No models or GPU inference are run.

The default audit uses committed TSV snapshots and Python's standard library.
--recheck-audio also needs the original host inputs and the pinned evaluator
source; it checks decoded float32 samples and CPU resampling at 1/2 threads.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'docs/experiments/harn/metric_invariance_128/smoke_evidence'
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap', 'meva')
TOL = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def audit(directory):
    provenance = json.loads((directory/'provenance.json').read_text())
    for name, expected in provenance['exports'].items():
        if sha(directory/name) != expected:
            raise ValueError(f'export hash mismatch: {name}')
    if sha(directory/'smoke_done.json') != provenance['sources']['smoke_done']['sha256']:
        raise ValueError('smoke metadata hash mismatch')
    smoke = rows(directory/'smoke_50.tsv')
    historical = rows(directory/'historical_clean_vs_shuffled_5131.tsv')
    assert len(smoke) == 50 and len(historical) == 5131
    assert [int(r['index']) for r in smoke] == list(range(50))
    assert len({r['id'] for r in smoke}) == 50
    result = {'n_smoke': 50, 'n_historical': 5131, 'identity_tolerance': TOL,
              'metrics': {}, 'all_exceedances_in_last_18': True}
    for metric in METRICS:
        delta = [abs(float(r[f'smoke_{metric}'])-float(r[f'reference_126a_{metric}'])) for r in smoke]
        assert all(math.isfinite(v) for v in delta)
        indices = [i for i, v in enumerate(delta) if v > TOL]
        record = {'max_abs_diff': max(delta), 'over_tolerance_count': len(indices),
                  'over_tolerance_indices': indices,
                  'first_32_max': max(delta[:32]), 'last_18_max': max(delta[32:]),
                  'passed': not indices}
        if any(i < 32 for i in indices):
            result['all_exceedances_in_last_18'] = False
        if smoke[0][f'reference_115_{metric}']:
            record['historical_126a_vs_115_max_on_50'] = max(
                abs(float(r[f'reference_126a_{metric}'])-float(r[f'reference_115_{metric}'])) for r in smoke)
        if metric in METRICS[:4]:
            old = [abs(float(r[f'clean_{metric}'])-float(r[f'shuffled_{metric}'])) for r in historical]
            record['historical_clean_vs_shuffled_max_on_5131'] = max(old)
            record['historical_clean_vs_shuffled_over_tolerance_count'] = sum(v > TOL for v in old)
        result['metrics'][metric] = record
    result['identity_gate_passed'] = all(r['passed'] for r in result['metrics'].values())
    return result


def recheck_audio(directory, scratch):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    contract = json.loads((REPO/'docs/experiments/metric_invariance_128a_20261017_contract.json').read_text())
    source = REPO/'scripts/eval/metric_invariance_128.py'
    if sha(source) != contract['bindings']['action_sha256']:
        raise ValueError('registered evaluator source changed; check out the failed-run revision')
    transform_source = REPO/'scripts/eval/metric_win_tie_126a.py'
    if sha(transform_source) != contract['immutable_files'][str(transform_source)]:
        raise ValueError('126a transform binding changed')
    sys.path.insert(0, str(REPO/'scripts/eval'))
    import metric_invariance_128 as M

    records = json.loads((directory/'audio_inputs_50.json').read_text())
    result = {'clips': len(records), 'sample_mismatch_clips': 0,
              'sample_max_abs_diff': 0., 'resample_thread_max_abs_diff': 0., 'per_clip': []}
    original_threads = torch.get_num_threads()
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as td:
            for r in records:
                if sha(r['path']) != r['sha256']:
                    raise ValueError(f'input changed: {r["id"]}')
                legacy = Path(td)/'historical.flac'
                M.A.make_one((r['path'], str(legacy), 'musicgen', 'clean'))
                current = M.prepare_one((r, 'musicgen', 'clean', 'C0', td))['paths'][0]
                a, sr = sf.read(legacy, dtype='float32')
                b, sr2 = sf.read(current, dtype='float32')
                assert sr == sr2 and a.shape == b.shape
                delta = float(np.max(np.abs(a-b)))
                wave = torch.from_numpy(a).unsqueeze(0)
                torch.set_num_threads(1)
                one = torchaudio.functional.resample(wave, sr, 16000)
                torch.set_num_threads(2)
                two = torchaudio.functional.resample(wave, sr, 16000)
                resample_delta = float(torch.max(torch.abs(one-two)))
                result['sample_mismatch_clips'] += int(delta != 0)
                result['sample_max_abs_diff'] = max(result['sample_max_abs_diff'], delta)
                result['resample_thread_max_abs_diff'] = max(result['resample_thread_max_abs_diff'], resample_delta)
                result['per_clip'].append({'id': r['id'], 'sample_rate': sr, 'samples': len(a),
                    'historical_decoded_float32_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                    'new_decoded_float32_sha256': hashlib.sha256(b.tobytes()).hexdigest(),
                    'sample_max_abs_diff': delta, 'resample_thread_max_abs_diff': resample_delta})
                legacy.unlink(); Path(current).unlink()
    finally:
        torch.set_num_threads(original_threads)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=DEFAULT)
    parser.add_argument('--recheck-audio', action='store_true')
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.evidence_dir)
    if args.recheck_audio:
        result['cpu_audio_identity'] = recheck_audio(args.evidence_dir, args.scratch)
    raw = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    print(raw, end='')
    # Reproducing the known failed gate is a successful audit, not a passing smoke.


if __name__ == '__main__':
    main()
