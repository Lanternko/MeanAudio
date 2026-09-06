"""Positive-side counterpart to the negprompt ablation matrix.

The 43 cells run so far all vary --negative_prompt and --cfg_strength; the
positive prompt is the TSV caption verbatim in every one of them. So the
negprompt verdict -- that the gain comes from fidelity-domain VOCABULARY and
not from defect polarity -- has only ever been tested in the slot where text
acts as a CFG reference point. If vocabulary is really the active ingredient,
the same words should do something from the positive slot too. If instead the
gain needs the negative slot specifically, that is evidence the mechanism is
about guidance geometry, not about the words.

There is a concrete prior AGAINST the positive slot working: T5 features are
mean-pooled before conditioning, so appended tokens dilute the caption's own
semantics. P8V4 put synthetic `[consistency=X.XX]` markers in the positive
stream at TRAINING time and CLAP fell to 0.0571. That was a different
experiment (training-side, non-natural-language markers), but it is the reason
to expect dilution here and to watch CLAP, not just PQ.

Design: `hifi` is character-identical to NEGATIVES['reversed'], so the same
string is tested in both slots and the comparison is exact.

  cfg 3.0 (PQ-optimal, full content ladder already exists there)
    none__POShifi     vs existing __cfg3.0__none      -- does positive vocab do anything
                      vs existing __cfg3.0__fidelity  -- positive slot vs negative slot
    none__POSneutral  -- any-text control: isolates "appended tokens" from "fidelity words"
    fidelity__POShifi -- does positive stack on top of the best known config

  cfg 1.5 (where `reversed` was measured in the negative slot)
    none__POShifi     vs existing __cfg1.5__reversed  -- SAME STRING, opposite slot

Writes into the negprompt_ablation output dir with distinct labels so the
subset, protocol, CLAP batch size and the cfg0 pairing reference are shared
exactly. Existing cells are never touched (different filenames + skip-if-exists).

CLAP is scored against the ORIGINAL caption -- the suffix goes only to the
generator via eval.py --prompt_suffix and the TSV is not rewritten. Scoring the
rewritten text would inflate CLAP by construction.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import negprompt_ablation_matrix as NM

OUT = NM.OUT
AUDIO_ROOT = NM.AUDIO_ROOT

POSITIVES = {
    # identical string to NEGATIVES['reversed'], so the slot is the only variable
    'hifi': 'high quality recording, clean, professional, pristine, hi-fi',
    # identical string to NEGATIVES['neutral']: any-text-in-the-positive-slot control
    'neutral': 'music',
}

# (label, arm, cfg, negative_key, positive_key)
CELLS = [
    ('c2p0_slot0__cfg3.0__none__POShifi',        'c2p0_slot0', '3.0', 'none',     'hifi'),
    ('c2p0_slot0__cfg3.0__none__POSneutral',     'c2p0_slot0', '3.0', 'none',     'neutral'),
    ('c2p0_slot0__cfg1.5__none__POShifi',        'c2p0_slot0', '1.5', 'none',     'hifi'),
    ('c2p0_slot0__cfg3.0__fidelity__POShifi',    'c2p0_slot0', '3.0', 'fidelity', 'hifi'),
]


def generate(exp_id, qflags, cfg, neg_key, pos_key, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    if len(list(audio_dir.glob('*.flac'))) >= NM.SUBSET_N:
        print('  [skip gen] already present')
        return
    ckpt = NM.EXPS / exp_id / f'{exp_id}_ema_final.pth'
    if not ckpt.is_file():
        raise SystemExit(f'[FAIL] missing checkpoint {ckpt}')
    cmd = [NM.PYTHON, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt),
           '--output', str(audio_dir), '--tsv', str(NM.SUBSET_TSV), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', cfg,
           '--no_text_attention_mask', '--encoder_name', 't5_clap',
           '--text_c_dim', '512', '--seed', '42', '--full_precision'] + qflags
    neg = NM.NEGATIVES[neg_key]
    if neg is not None:
        cmd += ['--negative_prompt', neg]
    cmd += ['--prompt_suffix', POSITIVES[pos_key]]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=NM.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'[FAIL] generation failed: {exp_id} cfg={cfg} '
                         f'neg={neg_key} pos={pos_key}')
    got = len(list(audio_dir.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time() - t0) / 60:.1f} min')
    if got < NM.SUBSET_N * 0.99:
        raise SystemExit(f'[FAIL] only {got}/{NM.SUBSET_N} clips')


def main():
    if not NM.CLAP_CKPT.is_file():
        raise SystemExit(f'[FAIL] missing CLAP checkpoint {NM.CLAP_CKPT}')
    for _, arm, _, _, _ in CELLS:
        exp_id = NM.ARMS[arm][0]
        ckpt = NM.EXPS / exp_id / f'{exp_id}_ema_final.pth'
        if not ckpt.is_file():
            raise SystemExit(f'[FAIL] missing checkpoint for {arm}: {ckpt}')
    if not NM.SUBSET_TSV.exists():
        raise SystemExit(f'[FAIL] subset TSV missing: {NM.SUBSET_TSV}. '
                         'Refusing to regenerate it -- the existing cells are keyed to it.')
    # eval.py must actually have the flag, or the suffix is silently dropped and
    # every cell here becomes a duplicate of an existing negprompt cell.
    helptext = subprocess.run([NM.PYTHON, 'eval.py', '--help'], cwd=NM.ROOT,
                              capture_output=True, text=True).stdout
    if '--prompt_suffix' not in helptext:
        raise SystemExit('[FAIL] eval.py has no --prompt_suffix; suffix would be dropped')
    print('[preflight] checkpoints, CLAP, shared subset TSV and --prompt_suffix all present')

    rows = NM.load_rows()
    print(f'{len(CELLS)} cells, {len(rows)} rows each')
    for label, arm, cfg, neg_key, pos_key in CELLS:
        result_path = OUT / f'{label}.json'
        if result_path.exists():
            print(f'[skip] {label}')
            continue
        exp_id, qflags = NM.ARMS[arm]
        print(f'[cell] {label}')
        audio_dir = AUDIO_ROOT / label
        generate(exp_id, qflags, cfg, neg_key, pos_key, audio_dir)
        per_sig = NM.per_clip_signal(audio_dir)
        sat = NM.signal_stats(per_sig)
        per = NM.score(rows, audio_dir)          # CLAP vs ORIGINAL caption
        aggs, lofi = NM.aggregate(per, rows)
        payload = {'label': label, 'arm': arm, 'exp_id': exp_id,
                   'cfg_strength': float(cfg),
                   'negative_key': neg_key, 'negative_prompt': NM.NEGATIVES[neg_key],
                   'positive_key': pos_key, 'prompt_suffix': POSITIVES[pos_key],
                   'scoring_note': 'CLAP text side is the unmodified TSV caption',
                   'subset': {'tsv': str(NM.SUBSET_TSV), 'n': NM.SUBSET_N,
                              'seed': NM.SUBSET_SEED},
                   'signal_stats': sat,
                   'per_clip_signal': per_sig,
                   'aggregates': aggs,
                   'lofi_ids': lofi,
                   'paired_delta_vs_cfg0': NM.paired_delta(arm, per),
                   'per_clip': per}
        result_path.write_text(json.dumps(payload, indent=1))
        shutil.rmtree(audio_dir, ignore_errors=True)
        a = payload['aggregates']['full']
        print(f"  PQ {a['PQ']:.4f}  CLAP {a['clap']:.4f}  crest_min {sat['crest_min']:.2f}")
    print('\nALL CELLS DONE')


if __name__ == '__main__':
    sys.exit(main())
