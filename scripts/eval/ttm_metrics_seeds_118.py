#!/usr/bin/env python3
"""p2 118 action: the p2 115 loudness/metrics protocol for the 116/117 training seeds.

Contract: docs/experiments/ttm_metrics_seeds_118_20261011_contract.json (GPU_QUEUE_CONTRACT).
Design: docs/experiments/ttm_quality_comparison_plan_20261007.md §9.4 (pre-registered
three-seed reading), §7.7 (loudness rules).

The 115 selection is frozen, so this job scores only the four new MeanAudio cells
(s27182818 / s16180339 x CFG0 / CFG3+neg). Every processing and scoring step is the 115
function itself, imported from scripts/eval/ttm_metrics_115.py (sha-bound in the contract):
gain rule, silent rule, peak cap, FLAC PCM_24, eval_metrics.py (CLAP batch 1, AES, level),
MEva, FAD on raw only with the explicit --ref_dir.

Differences from 115, all outside the scored path:
  - cells and their REPORT sha come from the contract (no selection.json to freeze);
  - MIR is not recomputed: the standard eval (mc_mf25_eval_mir.sh) already wrote
    mir_metrics.json for these cells on the same raw audio with the same mir_metrics.py;
    the files are sha-bound in the contract and copied into the summary;
  - the silent union for the exclusion sensitivity is the 115 generated-system union
    plus the silent clips of the four new cells;
  - summary.json adds the three-seed lvl30 table (114 from the 115 summary) and the
    pre-registered same-sign reading against MusicGen.

  (no flag)        run every unfinished step (each step is skipped once its output validates)
  --preflight      inputs, bindings, free disk, MEva runtime check
  --validate-only  every step present and complete; writes the registered summary.json
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttm_metrics_115 as p115  # noqa: E402

from ttm_metrics_115 import (AES, BLOCKS, REPO, SILENT_DBFS, SILENT_LUFS, PEAK_CAP, UNEQUAL_RATE,  # noqa: E402
                             contract, fmean, gain_done, gain_summary_path, log, make_block,
                             metrics_done, metrics_json, meva_done, meva_json, num, per_clip,
                             root, run_metrics, run_meva, sha256)

TABLE = ('clap', 'PQ', 'CE', 'CU', 'PC', 'meva')


def systems(c):
    """The four cells in the 115 system-record shape, so the 115 step functions apply."""
    return [{'name': x['name'], 'kind': 'meanaudio', 'audio': x['audio'], 'sampling_rate': 16000,
             'fad': True, 'seed': x['seed'], 'cell': x['cell'], 'mir_json': x['mir_json']}
            for x in c['meanaudio_cells']]


def mir_ok(c, x):
    p = Path(x['mir_json'])
    if not p.is_file() or sha256(p) != x['mir_json_sha256']:
        return False
    m = json.loads(p.read_text())
    return (m['n_present'] == c['fixed_protocol']['n_prompts'] and m['gen_dir'] == x['audio']
            and m['tsv_sha256'] == c['fixed_protocol']['tsv_sha256'])


def preflight(c):
    p, b = c['fixed_protocol'], c['bindings']
    if sha256(p['tsv']) != p['tsv_sha256']:
        sys.exit('TSV sha mismatch')
    for k in ('protocol_module', 'eval_metrics', 'meva_scorer', 'mir_metrics'):
        if sha256(b[k]) != b[f'{k}_sha256']:
            sys.exit(f'{k} sha mismatch')
    if Path(p115.__file__).resolve() != Path(b['protocol_module']):
        sys.exit('protocol module imported from an unexpected path')
    ref = c['summary_115']
    if sha256(ref['path']) != ref['sha256']:
        sys.exit('115 summary changed')
    for x in c['meanaudio_cells']:
        if sha256(x['report']) != x['report_sha256']:
            sys.exit(f'{x["name"]}: REPORT changed')
        if len(list(Path(x['audio']).glob('*.flac'))) != p['n_prompts']:
            sys.exit(f'{x["name"]}: audio count != {p["n_prompts"]}')
        if not mir_ok(c, x):
            sys.exit(f'{x["name"]}: MIR json missing, changed or not on this audio')
    if not Path(p['fad_ref_dir']).is_dir():
        sys.exit('FAD reference dir missing')
    for py in (p['python'], p['meva_python']):
        if not os.access(py, os.X_OK):
            sys.exit(f'missing interpreter {py}')
    root(c).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root(c)).free < c['storage']['min_free_bytes']:
        sys.exit('not enough free disk')
    if not (root(c) / 'meva_runtime_check.json').is_file():
        r = subprocess.run([p['meva_python'], str(REPO / 'scripts/eval/meva_score_dir.py'),
                            '--runtime_check', str(p['meva_runtime_check_n'])],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'MEva runtime check failed: {r.stdout[-500:]} {r.stderr[-500:]}')
        (root(c) / 'meva_runtime_check.json').write_text(r.stdout.strip().splitlines()[-1])
    print(f'preflight ok; cells: {[x["name"] for x in c["meanaudio_cells"]]}')


def run(c):
    for block in BLOCKS:
        for s in systems(c):
            if BLOCKS[block] is not None and not gain_done(c, s, block):
                make_block(c, s, block)
            if not metrics_done(c, s, block):
                run_metrics(c, s, block)
            if not meva_done(c, s, block):
                run_meva(c, s, block)
    log('all steps done')


def silent_union_115(c):
    """Silent ids of the generated systems of 115 (its exclusion union)."""
    base = Path(c['summary_115']['path']).parent
    sel = json.loads((base / 'selection.json').read_text())
    union = set()
    for s in sel['systems']:
        if s['kind'] != 'reference':
            union |= {i for i, r in per_clip(base / s['name'] / 'lvl30' / 'gain_status.tsv').items()
                      if r['status'] == 'silent_raw'}
    return union


def three_seed(c, out):
    """Pre-registered reading (plan §9.4): a lvl30 lead over MusicGen holds only if all three
    training seeds have the same sign; otherwise it is written as a tie."""
    ref = json.loads(Path(c['summary_115']['path']).read_text())['systems']
    base = c['three_seed']
    res = {'block': 'lvl30', 'reference_system': base['reference_system'], 'rule': base['rule'], 'cells': {}}
    for cfg in base['cfgs']:
        rows = {str(base['seed_114']): ref[f'meanaudio_{cfg}']['blocks']['lvl30']}
        for x in c['meanaudio_cells']:
            if x['cfg'] == cfg:
                rows[str(x['seed'])] = out['systems'][x['name']]['blocks']['lvl30']
        cell = {'per_seed': {s: {k: r[k] for k in TABLE} for s, r in rows.items()}}
        cell['mean'] = {k: statistics.fmean(r[k] for r in rows.values()) for k in TABLE}
        cell['range'] = {k: max(r[k] for r in rows.values()) - min(r[k] for r in rows.values()) for k in TABLE}
        cell['vs_reference'] = {}
        for k in base['verdict_metrics']:
            mg = ref[base['reference_system']]['blocks']['lvl30'][k]
            d = {s: r[k] - mg for s, r in rows.items()}
            same = all(v > 0 for v in d.values()) or all(v < 0 for v in d.values())
            cell['vs_reference'][k] = {
                'reference': mg, 'delta_per_seed': d, 'delta_mean': statistics.fmean(d.values()),
                'same_sign': same,
                'reading': ('ours_higher' if min(d.values()) > 0 else 'ours_lower') if same else 'tie'}
        res['cells'][cfg] = cell
    return res


def validate(c):
    sysm = systems(c)
    bad = []
    for s in sysm:
        for block in BLOCKS:
            if BLOCKS[block] is not None and not gain_done(c, s, block):
                bad.append(f'{s["name"]} {block}: gain')
            if not metrics_done(c, s, block):
                bad.append(f'{s["name"]} {block}: metrics')
            if not meva_done(c, s, block):
                bad.append(f'{s["name"]} {block}: meva')
    for x in c['meanaudio_cells']:
        if not mir_ok(c, x):
            bad.append(f'{x["name"]}: mir')
    if bad:
        sys.exit('; '.join(bad))

    n = c['fixed_protocol']['n_prompts']
    silent = {s['name']: {i for i, r in per_clip(root(c) / s['name'] / 'lvl30' / 'gain_status.tsv').items()
                          if r['status'] == 'silent_raw'} for s in sysm}
    u115 = silent_union_115(c)
    union = u115.union(*silent.values())
    out = {'document_kind': 'ttm_metrics_seeds_118_summary_v1', 'written_at': time.strftime('%FT%T%z'),
           'tsv': c['fixed_protocol']['tsv'], 'protocol': 'p2 115 (scripts/eval/ttm_metrics_115.py functions)',
           'silent_rule': f'raw RMS < {SILENT_DBFS} dBFS or LUFS < {SILENT_LUFS} / not finite: not gained, kept, counted',
           'peak_rule': f'gain reduced so peak = {PEAK_CAP}; processing_unequal if lvl30 rate > {UNEQUAL_RATE}',
           'silent_union_115_generated_n': len(u115), 'silent_union_n': len(union),
           'markers': c['markers'], 'systems': {}}
    nonfinite = []
    for s in sysm:
        rec = {'kind': s['kind'], 'seed': s['seed'], 'cfg_cell': s['cell'], 'sampling_rate': 16000,
               'audio': s['audio'], 'n': n, 'silent_n': len(silent[s['name']]),
               'silent_rate': len(silent[s['name']]) / n, 'blocks': {}}
        for block in BLOCKS:
            m = json.loads(metrics_json(c, s, block).read_text())['metrics']
            pc = per_clip(metrics_json(c, s, block).parent / 'per_clip.tsv')
            mv = {k: float(v['meva_raw']) for k, v in per_clip(meva_json(c, s, block).parent / 'meva_per_clip.tsv').items()}
            b = {'clap': m['clap_score'], **{k: m[f'aes_{k}'] for k in AES},
                 'meva': json.loads(meva_json(c, s, block).read_text())['meva_mean'],
                 'lufs_mean': m['level_lufs_mean'], 'crest_mean': m['level_crest_mean'],
                 'clipped_n': m['level_clipped_n'], 'silent_rms_n': m['level_silent_n']}
            if 'fad' in m:
                b['fad'], b['fad_pairs'] = m['fad'], m['fad_pairs']
            if BLOCKS[block] is not None:
                g = json.loads(gain_summary_path(c, s, block).read_text())
                b['gain_status'], b['peak_capped_rate'] = g['status'], g['peak_capped_rate']
                b['processing_unequal'] = g['processing_unequal']
            nonfinite += [f'{s["name"]} {block} {k}' for k in (*TABLE, 'lufs_mean', 'crest_mean')
                          if not isinstance(b[k], (int, float)) or b[k] != b[k] or abs(b[k]) == float('inf')]
            keep = [i for i in pc if i not in union]
            b['excl_silent_union'] = {
                'n': len(keep), 'clap': fmean(num(pc[i]['clap']) for i in keep),
                **{k: fmean(num(pc[i][k]) for i in keep) for k in AES},
                'meva': fmean(mv[i] for i in keep if i in mv)}
            rec['blocks'][block] = b
        mir = json.loads(Path(s['mir_json']).read_text())
        rec['mir_raw'], rec['mir_source'] = mir['metrics'], s['mir_json']
        out['systems'][s['name']] = rec
    if nonfinite:
        sys.exit('non-finite means: ' + '; '.join(nonfinite))
    out['three_seed_lvl30'] = three_seed(c, out)
    report = Path(c['reports'][0]['path'])
    report.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    c = contract()
    if '--preflight' in sys.argv:
        preflight(c)
    elif '--validate-only' in sys.argv:
        validate(c)
    else:
        run(c)
