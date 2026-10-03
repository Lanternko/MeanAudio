# AES queue repair, 2026-10-03

Operator requested “修復？” after observing failed entries 095/096 and idle P2.

096 failed before evaluation with `ModuleNotFoundError: demucs_infer`. Its registered isolated packages existed, but the native launcher omitted PYTHONPATH. New entry 098 restores that exact process-local package path. A new preflight loads the pinned local htdemucs_6s model on CPU, verifies its six sources, forbids CUDA initialization, then executes the original immutable preflight. Contract binds all dependency files. Models, audio, prompts, selections, normalizer, thresholds, score caching and analysis are unchanged. Original failed terminal evidence is retained; 098 is appended at the tail. No shared environment or services are changed.

095 failed its gain-control gate on `QUB_vpjogmo_170`, N100 seed 27182818 historical CFG0. PQ difference 0.02218056 exceeds the unchanged 0.01 bound. The old normalizer gave -23 LUFS for clean and -6 dB derivatives, but their actual RMS differed by 0.5366642 dB: BS.1770 absolute gating selected different branches. This is a real normalization defect, not a queue failure. The old gate remains failed.

Draft `normalize_candidate.py` seeds both signals at a fixed -23 dB RMS before applying the original -23 LUFS solver. Original waveform is otherwise preserved; target and tolerance unchanged. This changes metric inputs and needs explicit operator authorization for a new scientific contract. It is not active in 098. Changed waveform hashes must invalidate cached scores; completed old results cannot be silently relabeled. All original files and scientific artifacts remain intact.

Recovery criteria for 098: dependency admission passes without GPU initialization; native P2 seat owns exact PID/start-time; child loads Demucs and AES/CLAP; a fresh valid scored record and progress must appear; durable required notifications are delivered. Terminal success remains contingent on all six 5,009-prompt cells and original coverage/control/postflight gates. Controller failure preserves artifacts and hands off through the native queue. No claim of experiment completion at launch.

Frozen source snapshots preserve the externally located, previously registered implementation in Git; `frozen_source/manifest.json` binds them to their original paths and bytes. P2 continues using those immutable original files.
