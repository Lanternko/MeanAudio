#!/usr/bin/env python3
"""Generate the two preregistered artifacts the jobs bind by sha:

  docs/experiments/metric_coverage_sample_manifest_20261027.json  (seed 20261027, §4.4)
  docs/experiments/metric_coverage_model_pins_20261027.json       (§2.2 weights/code pins)

The manifest is deterministic; re-running this file must reproduce the committed bytes
(`common.build_sample_manifest` is the single implementation, job 161 regenerates and
compares it).  Pins record the weights that exist on this machine; missing weights are
reported, never downloaded (§8.3).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402

NEWMODELS = Path('/mnt/seagate/newmodels_20261020')
REPOS = NEWMODELS / 'repos'
HF_CACHE = Path.home() / '.cache/huggingface/hub'
SONGEVAL_TREE = C.REPO / 'scripts/eval/songeval_src'
SINGMOS_CLONE = Path.home() / '.cache/torch/hub/South-Twilight_SingMOS_v1.1.2'

MODEL_PINS = {
    'muqeval_A1': {'hf_repo': 'zhudi2825/MuQ-Eval-A1',
                   'revision': 'b082e238827df51d99ac66bdd1b6f0d2406bc90e',
                   'weights': 'hf/hub/models--zhudi2825--MuQ-Eval-A1/snapshots/'
                              'b082e238827df51d99ac66bdd1b6f0d2406bc90e/model_state_dict.pt',
                   'code': str(REPOS / 'MuQ-Eval')},
    'muqeval_A3a': {'hf_repo': 'zhudi2825/MuQ-Eval',
                    'revision': 'cc0cb22ea4026f7d3fb1bae384927b870a24319f',
                    'weights': 'hf/hub/models--zhudi2825--MuQ-Eval/snapshots/'
                               'cc0cb22ea4026f7d3fb1bae384927b870a24319f/model_state_dict.pt',
                    'code': str(REPOS / 'MuQ-Eval')},
    'cmirm': {'hf_repo': 'HaiwenXia/CMI-RM',
              'revision': '704f2dc3a11a1d867cc9a683a49c17d9cc7d50e4',
              'weights': 'hf/hub/models--HaiwenXia--CMI-RM/snapshots/'
                         '704f2dc3a11a1d867cc9a683a49c17d9cc7d50e4/model.safetensors',
              'code': str(REPOS / 'CMI-RewardBench')},
    'pam': {'hf_repo': 'soham97/PAM',
            'revision': 'ead6e7f9d907268e30c73b29526f50523a34b4aa',
            'weights': 'hf/hub/models--microsoft--msclap/snapshots/'
                       'c47d441165daa21986ead0850660917636a81775/CLAP_weights_2023.pth',
            'text_backbone': 'openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e',
            'extra_weights': ['hf/hub/models--openai-community--gpt2/snapshots/'
                              '607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors'],
            'code': str(REPOS / 'PAM')},
    'songeval': {'repo': '848fb2ff3a2a9d64dcb20d46f07238c28abd7add',
                 'ckpt': '0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd',
                 'backbone': 'OpenMuQ/MuQ-large-msd-iter @ 0562a57814f6f8bbd9fdea0a25921a2fce1a841a',
                 'extra_weights': [str(HF_CACHE / 'models--OpenMuQ--MuQ-large-msd-iter/snapshots/'
                                       '0562a57814f6f8bbd9fdea0a25921a2fce1a841a/model.safetensors')],
                 'vendored_tree': str(SONGEVAL_TREE)},
    'mert': {'hf_repo': 'm-a-p/MERT-v1-95M',
             'revision': '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5',
             'weights': str(HF_CACHE / 'models--m-a-p--MERT-v1-95M/snapshots/'
                            '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5/pytorch_model.bin')},
    'muq': {'hf_repo': 'OpenMuQ/MuQ-large-msd-iter',
            'revision': '0562a57814f6f8bbd9fdea0a25921a2fce1a841a',
            'weights': str(HF_CACHE / 'models--OpenMuQ--MuQ-large-msd-iter/snapshots/'
                           '0562a57814f6f8bbd9fdea0a25921a2fce1a841a/model.safetensors'),
            'extra_weights': ['hf/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots/'
                              '0562a57814f6f8bbd9fdea0a25921a2fce1a841a/model.safetensors']},
    'singmos': {'repo': 'South-Twilight/SingMOS', 'tag': 'v1.1.2',
                'commit': '959691e5ca9465f774b2533f44a609536955d3ed',
                'clone': str(SINGMOS_CLONE)},
    'mir': {'madmom': str(Path.home() / 'venvs/madmom'),
            'librosa': str(Path.home() / 'venvs/mir')},
}


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pins():
    weights_file = NEWMODELS / 'weights_sha256.txt'
    recorded = {}
    if weights_file.is_file():
        for line in weights_file.read_text().splitlines():
            if not line.strip():
                continue
            h, rel = line.split(None, 1)
            recorded[rel.strip()] = h
    models = {}
    for model, meta in MODEL_PINS.items():
        entry = dict(meta)
        cand = []
        for rel in [meta.get('weights')] + list(meta.get('extra_weights') or []):
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else NEWMODELS / rel
            cand.append(str(p))
        if entry.get('weights') and not Path(entry['weights']).is_absolute():
            entry['weights'] = str(NEWMODELS / entry['weights'])
            entry['weights_sha256_recorded'] = recorded.get(meta['weights'])
        if entry.get('weights'):
            entry['weights_present'] = Path(entry['weights']).is_file()
            entry['weights_sha256'] = sha(entry['weights']) if entry['weights_present'] else None
            entry['hf_snapshot_dirs'] = [str(Path(entry['weights']).parent)]
            entry['hf_snapshot_dirs'] += [str(Path(p).parent) for p in cand[1:]]
        if entry.get('code'):
            entry['code_present'] = Path(entry['code']).is_dir()
            entry['code_commit'] = git_commit(entry['code'])
        if model == 'songeval':
            entry['vendored_tree_present'] = SONGEVAL_TREE.is_dir()
            entry['vendored_tree_sha256'] = (C.tree_sha256(SONGEVAL_TREE)
                                             if SONGEVAL_TREE.is_dir() else None)
            ckpt = SONGEVAL_TREE / 'ckpt/model.safetensors'
            if ckpt.is_file():
                entry['weights'] = str(ckpt)
                entry['weights_present'] = True
                entry['weights_sha256'] = sha(ckpt)
                cand.append(str(ckpt))
        # schema used by score_coverage.check_pins: per-file sha + size
        entry['file_sha256'] = {p: sha(p) for p in dict.fromkeys(cand) if Path(p).is_file()}
        entry['file_size'] = {p: Path(p).stat().st_size
                              for p in entry['file_sha256']}
        entry['extra_weights'] = [p for p in dict.fromkeys(cand)
                                  if p != entry.get('weights')]
        if model == 'singmos':
            entry['clone_present'] = SINGMOS_CLONE.is_dir()
            entry['clone_commit'] = git_commit(SINGMOS_CLONE)
        models[model] = entry
    missing = [m for m, e in models.items()
               if (e.get('weights') and not e.get('weights_present'))
               or (e.get('vendored_tree') and not e.get('vendored_tree_present'))
               or (e.get('clone') and not e.get('clone_present'))]
    return {'document_kind': 'metric_coverage_model_pins_v1',
            'written_at': C.written_at(),
            'source': str(weights_file) if weights_file.is_file() else None,
            'models': models, 'missing': missing,
            'note': 'pins are read-only inputs; a missing weight is reported, never '
                    'downloaded in this run'}


def write_manifest(dry_run=False):
    doc = C.build_sample_manifest()
    p = C.sample_manifest_path()
    if dry_run:
        print(f'manifest: {len(doc["clip_ids"])} clips -> {p}')
        return 0
    C.write_json_atomic(p, doc)
    print(f'manifest: {len(doc["clip_ids"])} clips sha256={C.sha256_file(p)}')
    return 0


def write_pins(dry_run=False):
    doc = pins()
    p = C.DOCS / 'metric_coverage_model_pins_20261027.json'
    if dry_run:
        print(json.dumps(doc['missing'], ensure_ascii=False))
        return 0
    C.write_json_atomic(p, doc)
    print(f'pins -> {p} missing={doc["missing"]}')
    for m, e in doc['models'].items():
        print(f"  {m}: weights={'present' if e.get('weights_present', e.get('vendored_tree_present', e.get('clone_present'))) else 'n/a'} "
              f"sha={e.get('weights_sha256') or e.get('vendored_tree_sha256')}")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--pins-only', action='store_true')
    a = ap.parse_args()
    if not a.pins_only:
        write_manifest(a.dry_run)
    if not a.manifest_only:
        write_pins(a.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
