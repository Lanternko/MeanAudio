# MEva in MeanAudio

MEva is deployed as a **shadow batch evaluator**, not a replacement for AES or CLAP.
Experiment design: `docs/experiments/meva_095_20261003.md`.
Runtime contract: `docs/experiments/meva_095_contract.json`.
Schema-v1 bundle: `docs/experiments/harn/meva_095/`.
Queue entry: P2 `097_meva_promptcc_shadow.sh`, after existing 095/096 AES work.

## Installation performed on this host

1. Clone YoEv/MEva at `93fdc17fd324b5a7ba8ad56292aa77538f7717c2` under `.external/MEva`; clone PapayaResearch/musicdiscovery at `a378cbb264f409c5b79413e78e7f2f10f19a45dc` under `.external/musicdiscovery`.
2. Create `runtime/meva_20261003/venv` with Python 3.12. Its `dac-base.pth` reads the existing dac packages; locally installed overlays do not alter dac. The validated torch is `2.11.0.dev20260127+cu128`, required for this RTX5090 port, rather than the upstream torch2.1 pins. Save the resolved versions in `runtime/meva_20261003/requirements.resolved.txt`.
3. Install local overlays for audiocraft1.3.0, sae-lens5.3.2, transformer-lens2.11.0, transformers4.41.2, tokenizers0.19.1 and their import dependencies. The old packages declare incompatible torch version pins; this port uses actual numerical parity tests, not a claim of clean upstream dependency resolution. xformers0.0.34 is present only for imports; attention explicitly uses torch, never its mismatched CUDA extension.
4. In the local audiocraft `models/loaders.py`, make its three trusted, hash-pinned checkpoint `torch.load` calls explicit `weights_only=False` for modern torch. Never apply that patch to other environments. Evaluator CNN remains `weights_only=True` with strict state loading. Model lock includes the patched loader hash.
5. Pre-download MusicGen-small revision `4c8334b02c6ec4e8664a91979669a501ec497792` (audiocraft state and compression state), pooled-small-f03 evaluator revision `c536169ac98b16449f3aa3be6355bf5620c1466b`, exact SAE `sae-4_k_32_11` without fallback, and T5-base revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. Store in the registered runtime models/cache directories. The CLAP imports transitively require only bert/roberta/bart tokenizer cache files; these are also hash-pinned. Inference is offline.
6. `prepare_meva_095.py` freezes audio/report/per-clip TSV coverage and audio SHA256. Re-running against the same manifest is rejected; changed inputs need a new contract.
7. Run no-GPU fixtures and the notified, GPU-locked real-audio `run_meva_095_smoke.sh` before registration. Smoke verifies the adapter against official batch1 extraction and repeatability. It is not proof of human validity or cross-hardware torch2.1 parity.
8. `register_meva_095.py` builds a staged launcher and schema bundle. Validate all four documents, run runtime `--preflight`, check `accept_guest`, publish source changes, deliver registration event, then atomically append the launcher to P2 pending. Do not run the long scorer directly.

## Operations

```bash
cd /home/kojiek/MeanAudio
runtime/meva_20261003/venv/bin/python scripts/eval/meva_promptcc.py --preflight
runtime/meva_20261003/venv/bin/python scripts/validate_experiment_harness_documents.py \
  --contract docs/experiments/harn/meva_095/contract.json \
  --preflight docs/experiments/harn/meva_095/preflight.json \
  --ledger docs/experiments/harn/meva_095/ledger.json \
  --queue docs/experiments/harn/meva_095/queue.json
```

Follow queue state at `~/gpu_queue/p2/{pending,running,done,held,failed}` and `~/logs/p2_host.log`. MEva runtime progress is `runtime/meva_20261003/progress.json`; pending entries have no scoring progress yet.

First useful result is `runtime/meva_20261003/pam_validation.json`; complete result is `summary.json` with exactly 99,878 verified scores and separately retained coverage. A PAM-only interim summary is not full completion: the queue guest requires full postflight and report hashes.

Scalar records are under `runtime/meva_20261003/results/<source-cell>/<clip-id>.json`; comparison TSVs under `tables/`. Every record binds audio SHA256 and model-lock SHA256. Historical audio and AES files are read-only. Missing old audio is listed, not silently regenerated. Resume skips only fully bound valid scalar records.

The existing P2 host owns GPU0 and notifies transitions. The guest checks exact PID/start-time/seat ownership, monitors the scoring child, holds on failed notification, storage below50GiB, stall>1800s, or total active compute>24h. Pause kills only its own child process group and binds the progress checkpoint before returning75. Repair is disabled; failed gates or stale input hashes require a new reviewed contract rather than mutation of a launched run.

Rollback leaves AES intact. Remove only this unseated entry if canceling before launch. For active work, use the existing queue's scoped pause/stop workflow; never kill unrelated GPU services or change shared drivers.
