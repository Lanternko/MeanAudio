#!/usr/bin/env python3
"""Rewrite sha256 bindings of docs/experiments/attack_change_amount_199_contract.json (run after any edit to bound files)."""
import hashlib, json
from pathlib import Path
R = Path('/home/kojiek/MeanAudio'); E = Path('/home/kojiek/eval_output_nvme/attack_change_amount_199')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
c_path = R / 'docs/experiments/attack_change_amount_199_contract.json'
c = json.loads(c_path.read_text()) if c_path.is_file() else {}
PY = '/home/kojiek/venvs/dac/bin/python'; ACT = str(R / 'scripts/eval/attack_change_amount_199.py')
c.update({
 'schema_version': 1, 'document_kind': 'eval_only_metrics_contract', 'experiment_id': 'attack-change-amount-199-20261125',
 'status': 'authorized_p2_pending', 'launch_allowed': True, 'priority': 'P2', 'queue_name': '199_attack_change_amount.sh',
 'queue_role': 'p2', 'dependencies': [], 'ordering_dependencies': [], 'approval_required': True,
 'launch_authorization': {'gpu_launch_allowed': False, 'queue_position': 'p2', 'valid': True,
   'scope': 'Eval-only. Change amount (rho/mel/d_clap/d_mert/fd_clap) for all 197 table rows; 6 listenable bases x 1000 ids (T1) + MAESTRO 150 x 2 SoundFont (T2A); mix and white-noise ladders as yardsticks. No A1/PQ/CE/CMI re-scoring.'},
 'question': 'How large is each 197 attack objectively: small change, medium, or replaced music?',
 'design_document': str(R / 'docs/experiments/attack_change_amount_199_20261125.md'),
 'design_document_sha256': sh(R / 'docs/experiments/attack_change_amount_199_20261125.md'),
 'fixed_protocol': {'prereg': 'docs/experiments/attack_change_amount_199_20261125.md', 'seed': 20261117,
   'bootstrap': {'n': 5000, 'seed': 20261117}, 'clap_batch': 8, 'fifo': 'GPU free < 8 GiB or MemAvailable < 14 GiB -> rc 75', 'hard_cap_hours': 5},
 'deviations': {'see': 'design doc deviation log (end of file)'},
 'storage': {'path': str(E), 'min_free_bytes': 21474836480},
 'resume': {'kind': 'from_scratch_with_autoresume', 'checkpoint': None, 'iteration': 0,
   'pause_progress': str(E / 'pause_progress_199_attack_change_amount.json'), 'autoresume': str(E / 'pause_progress_199_attack_change_amount.json'),
   'note': 'per-cell per_clip/*.tsv caches; incomplete cell is rescored'},
 'watcher': {'stall_seconds': 5400, 'progress_globs': [str(E / 'per_clip/*'), str(E / '*.json'), str(E / 'emb/*')],
   'done_globs': [str(E / 'per_clip/*.tsv')]},
 'commands': {'preflight': [PY, '-I', ACT, '--preflight'], 'run': [PY, '-I', ACT], 'postflight': [PY, '-I', ACT, '--validate-only']},
 'reports': [{'path': str(R / 'docs/experiments/results/attack_change_amount_199_summary.json')}, {'path': str(E / 'timing.json')}],
 'completion_evidence': {'method': 'guest writes terminal (eval_only) with sha256 of reports after postflight passes'},
 'gpu_launch_allowed': True})
q = '/home/kojiek/gpu_queue/p2/pending/199_attack_change_amount.sh'
c['bindings'] = {'launcher': q, 'launcher_sha256': sh(q if Path(q).is_file() else q.replace('p2/pending', 'queue_candidates')),
  'harn': str(R / 'scripts/eval/attack_change_amount_199_guest.py'), 'harn_sha256': sh(R / 'scripts/eval/attack_change_amount_199_guest.py'),
  'action': ACT, 'action_sha256': sh(ACT)}
used = ['attack_metric_matrix_197.py', 'muq_overlay_matrix_189.py', 'muq_attractor_probe_188.py', 'muq_postproc_attack_190.py',
        'muq_filter_ladder_193.py', 'metric_win_tie_133.py', 'ce_weakness_195.py', 'ce_push_196.py', 'midi_realpiece_stimuli.py',
        'newmetrics_common.py', 'metric_win_tie_126d.py', 'level_match_rescore.py', 'eval_metrics.py']
cb = [{'path': str(R / 'scripts/eval' / f), 'sha256': sh(R / 'scripts/eval' / f)} for f in used if (R / 'scripts/eval' / f).is_file()]
cb.append({'path': str(R / 'scripts/analysis/attack_change_amount_tables.py'), 'sha256': sh(R / 'scripts/analysis/attack_change_amount_tables.py')})
cb.append({'path': str(R / 'scripts/analysis/attack_metric_matrix_tables.py'), 'sha256': sh(R / 'scripts/analysis/attack_metric_matrix_tables.py')})
c['code_bindings'] = cb
c_path.write_text(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
print('bound', len(cb), 'files ->', c_path)
