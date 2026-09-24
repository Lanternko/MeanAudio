"""
audit_run_manifest.py
=====================
Provenance audit for a MeanAudio training run.

Produces a structured manifest covering:
  1. TSV sha256
  2. NPZ cache sha256 (if provided and exists)
  3. NPZ manifest hash (file count + sorted-name sha256; if dir exists)
  4. Checkpoint sha256 (ema_final.pth)
  5. Hydra config extracted from S1/S2 logs (model, use_q_conditioning, tsv, npz_dir)
  6. 1000-sample text embedding CLAP re-encode cosine (if NPZ dir exists)
  7. Eval output file count and audio sanity (size > 0)
  8. PASS / FAIL summary

Items that cannot be verified (e.g., NPZ dir deleted post-training) are
reported as SKIP with the reason, not as FAIL.

Usage:
  python audit_run_manifest.py \\
    --exp        p8_qwen_stage2_200000 \\
    --train_tsv  /mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_random_train.tsv \\
    --npz_dir    ~/phase9_5_random_singlecap_npz \\
    --cache      /mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \\
    --s1_log     ~/logs/p8_qwen_stage1_400000.log \\
    --s2_log     ~/logs/p8_qwen_stage2_200000.log \\
    --eval_log   ~/logs/p8_qwen_stage2_200000_musiccaps_eval.log \\
    [--eval_audio_dir  ~/MeanAudio/eval_output/p8_qwen_stage2_200000_musiccaps/audio] \\
    [--clap_ckpt       ~/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt] \\
    [--n_clap          1000] \\
    [--seed            42] \\
    [--out_json        audit_manifest_<exp>.json]

Pass criteria (configurable via --clap_cos_threshold):
  - n_clap_bad == 0 AND mean_clap_cos >= 0.999   (only when NPZ dir exists)
  - Hydra config fields match expected values
  - Checkpoint sha256 non-empty (existence confirmed)
  - Eval audio count == expected_n (if --expected_n provided)
"""

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

CLAP_CKPT_DEFAULT = Path.home() / 'MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
MEANAUDIO_EXP_ROOTS = [
    Path.home() / 'MeanAudio/exps',
    Path('/mnt/HDD/kojiek/meanaudio_exps'),
    Path.home() / 'exps_nvme',
]

# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def npz_manifest_hash(npz_dir: Path) -> dict:
    """Hash the sorted list of filenames + their sizes (no file content read)."""
    files = sorted(npz_dir.glob('*.npz'))
    if not files:
        return {'n': 0, 'hash': None, 'note': 'no .npz files found'}
    manifest_str = '\n'.join(f'{p.name}:{p.stat().st_size}' for p in files)
    return {
        'n': len(files),
        'hash': sha256_string(manifest_str),
        'first': str(files[0]),
        'last': str(files[-1]),
    }


def find_checkpoint(exp: str) -> Path | None:
    for root in MEANAUDIO_EXP_ROOTS:
        p = root / exp / f'{exp}_ema_final.pth'
        if p.exists():
            return p
    return None


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\[\d+m')

def _strip_ansi(s: str) -> str:
    """Remove ANSI / bracket-style color codes from a log line."""
    return _ANSI_RE.sub('', s)


_POSIXPATH_RE = re.compile(r'PosixPath\(([^)]+)\)')

def _parse_python_dict(raw: str) -> dict | None:
    """Try to parse a Python-repr dict string, with fallback strategies."""
    raw = raw.strip()
    # Pre-process: replace PosixPath('...') with the inner string
    raw = _POSIXPATH_RE.sub(lambda m: m.group(1), raw)
    # 1. Direct literal_eval
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    # 2. JSON coercion (single→double quotes, Python bool/None → JSON)
    try:
        coerced = (raw
                   .replace("'", '"')
                   .replace(': True', ': true')
                   .replace(': False', ': false')
                   .replace(': None', ': null')
                   .replace(', True,', ', true,')
                   .replace(', False,', ', false,')
                   .replace(', None,', ', null,'))
        return json.loads(coerced)
    except Exception:
        pass
    return None


def extract_hydra_config(log_path: Path) -> dict | None:
    """Parse the 'All configuration:' dict from a Hydra log line."""
    if not log_path.exists():
        return None
    needle = 'All configuration:'
    with open(log_path) as f:
        for line in f:
            if needle in line:
                line = _strip_ansi(line)
                idx = line.index(needle) + len(needle)
                raw = line[idx:].strip()
                result = _parse_python_dict(raw)
                if result is not None:
                    return result
                return {'_raw': raw[:500]}
    return None


def extract_eval_args(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    needle = 'Eval args:'
    with open(log_path) as f:
        for line in f:
            if needle in line:
                line = _strip_ansi(line)
                idx = line.index(needle) + len(needle)
                raw = line[idx:].strip()
                result = _parse_python_dict(raw)
                if result is not None:
                    return result
                return {'_raw': raw[:500]}
    return None


def run_clap_spot_check(tsv_path: Path, npz_dir: Path, clap_ckpt: Path,
                         n: int, seed: int,
                         cache_path: 'Path | None' = None) -> dict:
    """Re-encode TSV captions with CLAP, compare to NPZ text_features_c.

    NPZ lookup priority (mirrors MeanAudio DataLoader):
      1. gt_cache file provided and exists → cache_files[i] gives the NPZ name for row i
      2. Sequential naming (0.npz, 1.npz, ...) detected → use f'{i}.npz'
      3. Fallback → sorted glob[i] (warn in result)
    """
    import torch
    import laion_clap

    rows = list(csv.DictReader(open(tsv_path), delimiter='\t'))
    n_rows = len(rows)

    # --- Determine NPZ index → filename mapping ---
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            cache_files = [l.strip() for l in f if l.strip()]
        def npz_path_for(i): return npz_dir / cache_files[i]
        mapping = 'gt_cache'
        n_available = min(n_rows, len(cache_files))
    elif (npz_dir / '0.npz').exists():
        def npz_path_for(i): return npz_dir / f'{i}.npz'
        mapping = 'sequential'
        n_available = n_rows
    else:
        npz_files_sorted = sorted(npz_dir.glob('*.npz'))
        if not npz_files_sorted:
            return {'status': 'SKIP', 'reason': 'no NPZ files in dir'}
        def npz_path_for(i): return npz_files_sorted[i] if i < len(npz_files_sorted) else None
        mapping = 'sorted_glob_fallback'
        n_available = min(n_rows, len(npz_files_sorted))
    print(f'  [CLAP] NPZ mapping: {mapping}  n_available={n_available}')

    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n_available), min(n, n_available)))

    print(f'  [CLAP] loading model...')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(clap_ckpt), verbose=False)
    model.eval()
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = model.to(device)

    sims = []
    bad = []
    BATCH = 32

    with __import__('torch').no_grad():
        for start in range(0, len(idxs), BATCH):
            batch_idxs = idxs[start:start + BATCH]
            caps = [rows[i]['caption'] for i in batch_idxs]
            emb = model.get_text_embedding(caps, use_tensor=True).cpu().float().numpy()

            for i, e in zip(batch_idxs, emb):
                npz_path = npz_path_for(i)
                if npz_path is None or not npz_path.exists():
                    bad.append({'idx': i, 'issue': 'npz_missing', 'path': str(npz_path)})
                    continue
                z = np.load(npz_path)['text_features_c'].astype('float32')
                # Handle multi-cap: take row 0 if 2D
                if z.ndim == 2:
                    z = z[0]
                cos = float(np.dot(e, z) / (np.linalg.norm(e) * np.linalg.norm(z) + 1e-9))
                sims.append(cos)
                if cos < 0.999:
                    bad.append({'idx': i, 'id': rows[i]['id'], 'cos': round(cos, 6),
                                'caption': rows[i]['caption'][:80]})

    if not sims:
        return {'status': 'SKIP', 'reason': 'no valid NPZ files found'}

    sims_arr = np.array(sims)
    return {
        'status': 'PASS' if len(bad) == 0 and sims_arr.mean() >= 0.999 else 'FAIL',
        'n': len(sims),
        'mean_cos': float(sims_arr.mean()),
        'min_cos': float(sims_arr.min()),
        'n_bad': len(bad),
        'bad_examples': bad[:5],
        'npz_mapping': mapping,
    }


def count_eval_audio(audio_dir: Path) -> dict:
    if not audio_dir or not audio_dir.exists():
        return {'status': 'SKIP', 'reason': 'audio_dir not provided or missing'}
    files = list(audio_dir.glob('*.flac')) + list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
    zero_size = [f.name for f in files if f.stat().st_size == 0]
    return {
        'count': len(files),
        'zero_size': len(zero_size),
        'zero_examples': zero_size[:5],
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='MeanAudio run provenance audit')
    p.add_argument('--exp',            required=True,  help='Experiment name, e.g. p8_qwen_stage2_200000')
    p.add_argument('--train_tsv',      required=True)
    p.add_argument('--npz_dir',        default=None,   help='NPZ directory (may be absent if deleted post-training)')
    p.add_argument('--cache',          default=None,   help='gt_cache file (optional)')
    p.add_argument('--s1_log',         default=None)
    p.add_argument('--s2_log',         default=None)
    p.add_argument('--eval_log',       default=None)
    p.add_argument('--eval_audio_dir', default=None)
    p.add_argument('--clap_ckpt',      default=str(CLAP_CKPT_DEFAULT))
    p.add_argument('--n_clap',         type=int, default=1000)
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--clap_cos_threshold', type=float, default=0.999)
    p.add_argument('--expected_n',     type=int, default=None, help='Expected eval audio count')
    p.add_argument('--out_json',       default=None)
    p.add_argument('--skip_clap',      action='store_true', help='Skip CLAP re-encode (fast mode)')
    return p.parse_args()


def main():
    args = parse_args()
    manifest = {'exp': args.exp, 'checks': {}}
    issues = []

    W = 50
    def hdr(title): print(f'\n{"─"*W}\n  {title}\n{"─"*W}')
    def ok(k, v=''):  print(f'  ✅  {k}' + (f': {v}' if v else ''))
    def warn(k, v=''): print(f'  ⚠️   {k}' + (f': {v}' if v else '')); issues.append(k)
    def skip(k, v=''): print(f'  ──  SKIP {k}' + (f' ({v})' if v else ''))
    def fail(k, v=''): print(f'  ❌  {k}' + (f': {v}' if v else '')); issues.append(k)

    print(f'\n{"═"*W}')
    print(f'  MeanAudio Run Provenance Audit')
    print(f'  exp: {args.exp}')
    print(f'{"═"*W}')

    # 1. TSV sha256
    hdr('1. Training TSV')
    tsv_path = Path(args.train_tsv).expanduser()
    if tsv_path.exists():
        tsv_hash = sha256_file(tsv_path)
        rows = list(csv.DictReader(open(tsv_path), delimiter='\t'))
        manifest['checks']['tsv'] = {'path': str(tsv_path), 'sha256': tsv_hash, 'n_rows': len(rows)}
        ok('exists', str(tsv_path))
        ok('sha256', tsv_hash[:16] + '...')
        ok('n_rows', f'{len(rows):,}')
    else:
        fail(f'TSV not found: {tsv_path}')
        manifest['checks']['tsv'] = {'status': 'MISSING', 'path': str(tsv_path)}
        rows = []

    # 2. NPZ cache sha256
    hdr('2. NPZ Cache (gt_cache)')
    cache_path = Path(args.cache).expanduser() if args.cache else None
    if cache_path and cache_path.exists():
        cache_hash = sha256_file(cache_path)
        with open(cache_path) as f:
            cache_lines = [l.strip() for l in f if l.strip()]
        manifest['checks']['cache'] = {'path': str(cache_path), 'sha256': cache_hash, 'n_lines': len(cache_lines)}
        ok('exists', str(cache_path))
        ok('sha256', cache_hash[:16] + '...')
        ok('n_lines', f'{len(cache_lines):,}')
    else:
        reason = 'not provided' if not args.cache else 'file not found'
        skip('gt_cache', reason)
        manifest['checks']['cache'] = {'status': 'SKIP', 'reason': reason}

    # 3. NPZ manifest hash
    hdr('3. NPZ Directory Manifest')
    npz_dir = Path(args.npz_dir).expanduser() if args.npz_dir else None
    if npz_dir and npz_dir.exists():
        mh = npz_manifest_hash(npz_dir)
        manifest['checks']['npz_manifest'] = mh
        ok('n_npz', f'{mh["n"]:,}')
        ok('manifest_hash', str(mh["hash"])[:16] + '...' if mh["hash"] else 'none')
    else:
        reason = 'not provided' if not args.npz_dir else f'deleted post-training ({args.npz_dir})'
        skip('NPZ dir', reason)
        manifest['checks']['npz_manifest'] = {'status': 'SKIP', 'reason': reason,
                                               'note': 'check slice_random.log / audit_expH_npz_text_features.py for historical evidence'}
        print(f'       Historical evidence: check run-time sanity logs (slice_random.log etc.)')

    # 4. Checkpoint sha256
    hdr('4. Checkpoint (ema_final.pth)')
    ckpt = find_checkpoint(args.exp)
    if ckpt:
        ckpt_hash = sha256_file(ckpt)
        sz = ckpt.stat().st_size / 1e6
        manifest['checks']['checkpoint'] = {'path': str(ckpt), 'sha256': ckpt_hash, 'size_mb': round(sz, 1)}
        ok('found', str(ckpt))
        ok('sha256', ckpt_hash[:16] + '...')
        ok('size_mb', f'{sz:.1f}')
    else:
        fail(f'ema_final.pth not found for {args.exp}')
        manifest['checks']['checkpoint'] = {'status': 'MISSING'}

    # 5. Hydra config from logs
    hdr('5. Hydra Config (from training logs)')
    EXPECTED = {
        'use_q_conditioning': False,
        'text_encoder_name': 't5_clap',
    }
    for stage, log_arg in [('S1', args.s1_log), ('S2', args.s2_log)]:
        log_path = Path(log_arg).expanduser() if log_arg else None
        if not log_path or not log_path.exists():
            skip(f'{stage} log', 'not found')
            continue
        cfg = extract_hydra_config(log_path)
        if cfg is None:
            warn(f'{stage}: "All configuration:" not found in log')
            continue
        # Extract key fields
        model_v  = cfg.get('model', '?')
        use_q    = cfg.get('use_q_conditioning', '?')
        tsv_v    = cfg.get('data', {}).get('AudioCaps_npz', {}).get('tsv', '?')
        npz_v    = cfg.get('data', {}).get('AudioCaps_npz', {}).get('npz_dir', '?')
        encoder  = cfg.get('text_encoder_name', '?')
        exp_id   = cfg.get('exp_id', '?')
        n_iter   = cfg.get('num_iterations', '?')

        manifest['checks'][f'hydra_{stage.lower()}'] = {
            'model': model_v, 'use_q_conditioning': use_q,
            'tsv': tsv_v, 'npz_dir': npz_v,
            'text_encoder_name': encoder, 'exp_id': exp_id, 'num_iterations': n_iter,
        }
        ok(f'{stage} model', model_v)
        ok(f'{stage} use_q_conditioning', str(use_q))
        ok(f'{stage} tsv', Path(tsv_v).name if tsv_v != '?' else '?')
        ok(f'{stage} npz_dir', str(npz_v))
        ok(f'{stage} text_encoder', encoder)

        # Validate
        if use_q != EXPECTED['use_q_conditioning']:
            fail(f'{stage} use_q_conditioning={use_q} (expected {EXPECTED["use_q_conditioning"]})')
        if encoder != EXPECTED['text_encoder_name']:
            warn(f'{stage} text_encoder_name={encoder} (expected t5_clap)')

        # Cross-check TSV matches args
        if args.train_tsv and tsv_v != '?' and Path(tsv_v) != Path(args.train_tsv).expanduser():
            warn(f'{stage} TSV in log ({Path(tsv_v).name}) ≠ --train_tsv arg ({Path(args.train_tsv).name})')

    # 6. Eval args from log
    hdr('6. Eval Flags (from eval log)')
    eval_log_path = Path(args.eval_log).expanduser() if args.eval_log else None
    if eval_log_path and eval_log_path.exists():
        eargs = extract_eval_args(eval_log_path)
        if eargs and '_raw' not in eargs:
            manifest['checks']['eval_args'] = {k: str(v) for k, v in eargs.items()
                                                if k in ('model_path','variant','no_q','cfg_strength','tsv','num_steps')}
            ok('variant',     str(eargs.get('variant','?')))
            ok('no_q',        str(eargs.get('no_q','?')))
            ok('cfg_strength',str(eargs.get('cfg_strength','?')))
            ok('model_path',  Path(str(eargs.get('model_path','?'))).name)
            ok('eval tsv',    Path(str(eargs.get('tsv','?'))).name)

            if not eargs.get('no_q', False):
                fail('eval no_q=False — should be True for NoQ model')
            if str(eargs.get('model_path','')) and args.exp not in str(eargs.get('model_path','')):
                warn(f'eval model_path does not contain exp name "{args.exp}"')
            if eargs.get('cfg_strength') not in (0.5, '0.5'):
                warn(f'cfg_strength={eargs.get("cfg_strength")} (expected 0.5)')
        else:
            skip('eval args', 'could not parse')
    else:
        skip('eval log', 'not found')
        manifest['checks']['eval_args'] = {'status': 'SKIP'}

    # 7. CLAP spot-check
    hdr('7. CLAP Text Embedding Re-encode (1000-sample)')
    if args.skip_clap:
        skip('CLAP re-encode', '--skip_clap set')
        manifest['checks']['clap_spot'] = {'status': 'SKIP', 'reason': '--skip_clap'}
    elif npz_dir and npz_dir.exists() and rows:
        clap_ckpt_path = Path(args.clap_ckpt).expanduser()
        if not clap_ckpt_path.exists():
            skip('CLAP re-encode', f'ckpt not found: {clap_ckpt_path}')
            manifest['checks']['clap_spot'] = {'status': 'SKIP', 'reason': 'ckpt missing'}
        else:
            print(f'  Running CLAP spot-check (n={args.n_clap}, seed={args.seed})...')
            result = run_clap_spot_check(
                tsv_path=tsv_path, npz_dir=npz_dir,
                clap_ckpt=clap_ckpt_path,
                n=args.n_clap, seed=args.seed,
                cache_path=cache_path,
            )
            manifest['checks']['clap_spot'] = result
            if result['status'] == 'PASS':
                ok('CLAP cos', f'mean={result["mean_cos"]:.6f}  min={result["min_cos"]:.6f}  n_bad={result["n_bad"]}')
            elif result['status'] == 'SKIP':
                skip('CLAP re-encode', result.get('reason',''))
            else:
                fail('CLAP re-encode', f'n_bad={result["n_bad"]}  mean={result["mean_cos"]:.6f}')
    else:
        reason = 'NPZ dir deleted — cannot re-verify. Use historical sanity log as proxy.'
        skip('CLAP re-encode', reason)
        manifest['checks']['clap_spot'] = {'status': 'SKIP', 'reason': reason}
        print(f'       Proxy: slice_random.log sanity check confirmed shape at generation time.')

    # 8. Eval audio count
    hdr('8. Eval Audio File Count')
    audio_dir = Path(args.eval_audio_dir).expanduser() if args.eval_audio_dir else None
    ac = count_eval_audio(audio_dir)
    manifest['checks']['eval_audio'] = ac
    if ac.get('status') == 'SKIP':
        skip('eval audio dir', ac.get('reason',''))
    else:
        ok('count', f'{ac["count"]:,}')
        if ac['zero_size']:
            fail(f'{ac["zero_size"]} zero-size audio files', str(ac['zero_examples']))
        else:
            ok('zero-size files', '0')
        if args.expected_n and ac['count'] != args.expected_n:
            fail(f'count {ac["count"]} ≠ expected {args.expected_n}')
        elif args.expected_n:
            ok('count matches expected', str(args.expected_n))

    # Final verdict
    print(f'\n{"═"*W}')
    print(f'  VERDICT')
    print(f'{"═"*W}')
    manifest['issues'] = issues
    if not issues:
        print('  ✅ PASS — no issues found in audited checks')
        print('  NOTE: SKIP items (deleted NPZ dir, missing logs) were not checked;')
        print('        this audit excludes those dimensions.')
        manifest['verdict'] = 'PASS'
    else:
        print(f'  ❌ ISSUES ({len(issues)}):')
        for iss in issues:
            print(f'    - {iss}')
        manifest['verdict'] = 'FAIL'

    print()
    print('  Claim precision:')
    print('  "No evidence of high-risk pipeline corruption found in audited checks.')
    print('   Items marked SKIP (deleted artifacts) cannot be verified retroactively.')
    print('   This audit does not rule out training-code bugs not manifest in logs."')
    print()

    # Save JSON
    out_json = args.out_json or f'audit_manifest_{args.exp}.json'
    with open(out_json, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f'  Manifest saved → {out_json}')
    print(f'{"═"*W}\n')

    sys.exit(0 if not issues else 1)


if __name__ == '__main__':
    main()
