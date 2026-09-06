# Phase-8 official-Qwen matched probe

> **Naming clarification:** `official-Qwen` is a legacy experiment/file name.
> It means the Qwen caption JSON distributed by the upstream
> `ICME26-ATTM-GC-FluxAudio` reference repository, not human-authored official
> MTG-Jamendo labels. That JSON has one row per upstream track path and this
> experiment maps it by track ID onto every retained local `segment_N`; it is
> therefore track-matched but not segment/time-window-matched. See
> [caption provenance、音訊粒度與 AES 控制實驗](../../caption_provenance_granularity_and_aes_controls.md).

The immutable contract is [phase8_qwen_official_matched_contract.json](../../phase8_qwen_official_matched_contract.json). The queue performs metadata mapping, a separate 512-row NPZ probe and audit, the measured full rebuild and audit, the neutral q9 sanity, then independently reset control/Qwen 20k continuations and identical MusicCaps evaluation. It fails closed and never selects a best checkpoint.

Launch only after Codex review:

```bash
cd /home/kojiek/MeanAudio
source /home/kojiek/venvs/dac/bin/activate
python scripts/phase8_qwen_probe_queue.py --execute --run-mode fresh \
  2>&1 | tee /home/kojiek/logs/phase8_qwen_official_matched_queue.log
```

For an audited prefix only, use `--run-mode resume`; it refuses to convert a
fresh run into a resume or vice versa. A failed step stops the queue. Metric
targets are descriptive and never trigger retraining.

Status:

```bash
python scripts/phase8_qwen_monitor.py --once --expect-active
```

Three-hour Luna monitor:

```bash
bash scripts/phase8_qwen_luna_3h.sh
```

Luna writes only diagnostics/proposals in this worktree. Any stop, change, or
relaunch requires Codex review and machine-readable Sol high approval.
