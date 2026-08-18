# Phase-8 Fixed-Q / ATTM durable experiment chain — proposal & handoff

Date: 2026-07-22
Branch: `grok/phase8-fixedq-attm-chain`
Worktree: `/home/kojiek/grok-worktrees/phase8-fixedq-attm-chain`
Live checkout (READ-ONLY for this proposal): `/home/kojiek/MeanAudio`

## Scientific objective

Test whether the historical Phase-8 MusicCaps benefit is better explained by a
**single learned default/high-Q prior** than by **ordinal per-row Q
information**.

Design: a fail-closed, paired residual fine-tune of the completed clean-NoQ
MeanAudio S2 checkpoint under two matched arms that share data order, cache,
seed, LR, batch, accumulation, NoMask, single-cap, save intervals, warmup, and
optimizer/scheduler reset.

| Arm | Prefix | Controlled difference |
|---|---|---|
| Fixed-Q9 prior | `phase8_fixedq9_prior_ft100k` | TSV forces every `q_level=9`; `use_q_conditioning=true`; q0..q9 initialized exactly from source q10; eval `--quality_level 9` |
| Matched NoQ | `phase8_matched_noq_ft100k` | Catalog multi-level Q TSV ignored by runner (`use_q_conditioning=false` → `q=None` → q10); q_embed preserved; eval `--no_q` |

If Fixed-Q9 recovers CLAP and the paired bootstrap CI95 lower bound for
FixedQ−NoQ is > 0, that supports a **high-Q prior** explanation.  If both arms
are similar, the historical gap is not explained by a constant-Q prior alone.

## Fixed baseline / contracts

| Item | Value |
|---|---|
| Source ckpt | `/home/kojiek/MeanAudio/exps/phase8_catalog_matched_noq_stage2_200000/phase8_catalog_matched_noq_stage2_200000_ckpt_last.pth` |
| Source iteration | **600000** (required) |
| FT updates | exactly **100000** → primary checkpoint **700000** |
| LR / batch / accum / seed | `3e-5` / `8` / `1` / `14159265` |
| Text mask | `use_text_attention_mask=false` (NoMask) |
| Multi-cap | `false` |
| Cache / NPZ | `npz_cache_train.txt` + `phase8_legacy_matched_npz` (gates must be `passed`) |
| Rows | **251599**; Fixed-Q TSV unique Q support exactly **`[9]`** |
| q semantics | `q=None` → **q10**; MeanFlow unconditional CFG pass → **q10**; Fixed-Q conditional rows → **q9** |
| Eval metric | MusicCaps, 1-step MeanFlow, CFG 0.5, internal CLAP **89.98** (continuity) |
| ATTM official | **Blocked** until exact official 100-prompt CSV; future evaluator uses **90.14** |

### Pre-registered interpretation (no cherry-picking)

1. Primary checkpoint is iteration **700000** only.
2. Fixed-Q9 restoration target: MusicCaps CLAP **≥ 0.1900**.
3. Evidence of Fixed-Q benefit requires paired bootstrap **CI95 lower bound of FixedQ−NoQ > 0**.
4. Do **not** stop merely because loss plateaus near **0.98**.
5. Report a negative result unchanged; never move the Q target or select a mid-run checkpoint on MusicCaps.

`primary_objective_met` means the strict Q-benefit claim (paired CI lower bound
above zero).  `fallback_restoration_met` separately records CLAP ≥0.1900;
`program_goal_met` is their OR and must not be presented as proof that Q helps.

### Hard-stop candidates

Contract/hash drift; NaN/Inf loss or LR; persistent nonfinite gradient
(continuous ≥2, recent 20 ≥3, or recent 100 ≥10); repeated loss >5; repeated
grad norm >100; OOM / NCCL / segfault / traceback; missing process with stale
log; disk <50 GB.  Stale/GPU/process incidents require **two observations**.  A
single AMP grad NaN that immediately recovers is **review-only**.

## Deliverables (this branch)

| Path | Role |
|---|---|
| `scripts/preprocess/make_phase8_fixedq9_tsv.py` | Deterministic Fixed-Q=9 TSV builder + SHA256 manifest |
| `scripts/init_phase8_fixedq_attm_checkpoint.py` | Continuation init: `noq` / `fixedq9`, matched opt/sched reset |
| `scripts/training_pipelines/train_pipeline_phase8_fixedq_attm_ft.sh` | One-arm train+eval launcher (`ARM=noq\|fixedq9`) |
| `scripts/training_pipelines/sequence_phase8_fixedq_attm.sh` | Durable queue: fixedq9 → noq → paired bootstrap |
| `scripts/audit_phase8_fixedq_attm_ft.py` | Independent contract/runtime/final audit |
| `scripts/monitor_phase8_fixedq_attm_ft.py` | Read-only JSON monitor (transient vs persistent nonfinite) |
| `scripts/eval/paired_clap_bootstrap_phase8_fixedq_attm.py` | Paired per-prompt CLAP bootstrap FixedQ vs NoQ |
| `scripts/preprocess/build_official_caption_inventory.py` | CPU-only official caption coverage/hash inventory |
| `scripts/tests/selftest_phase8_fixedq_attm_chain.py` | CPU synthetic self-tests |
| `docs/experiments/phase8_fixedq_attm_chain_proposal.json` | Machine-readable proposal |

**No historical/core files modified.**

## Queue / runtime plan

1. Build Fixed-Q=9 TSV once (if missing), assert 251599 rows and Q support `[9]`.
2. **Arm 1 — fixedq9**: init q10→q0..q9, train 600k→700k with Q on, eval q9.
3. **Arm 2 — matched noq**: preserve q_embed, train with Q off, eval `--no_q`.
4. Paired CLAP bootstrap on the two final 5521 MusicCaps audio dirs.
5. Write `FINAL_COMPARISON.json`.

Guarantees:

- Single GPU (`CUDA_VISIBLE_DEVICES=0`).
- Flock + process probe prevent duplicate launch.
- No fixed sleeps between arms; next arm starts only after previous audit passes.
- Does **not** invent or launch a third training arm.
- Fresh by default; resume only with explicit `EXPERIMENT_RUN_MODE=resume`.

### Artifacts

| Kind | Path |
|---|---|
| Sequence log | `/home/kojiek/logs/phase8_fixedq_attm_sequence.log` |
| Monitor state | `/home/kojiek/logs/phase8_fixedq_attm_monitor/` |
| Contracts | `/home/kojiek/logs/phase8_*_contract.json` |
| Init manifests | `/home/kojiek/logs/phase8_*_init.json` |
| Fixed-Q TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train_fixedq9.tsv` |
| Final comparison | `/home/kojiek/logs/phase8_fixedq_attm_monitor/FINAL_COMPARISON.json` |

### Tmux / monitor

- Training queue tmux: `p8_fixedq_attm`
- Monitor (read-only):
  `python /home/kojiek/MeanAudio/scripts/monitor_phase8_fixedq_attm_ft.py --once`

## Proposed live launch command (DO NOT RUN until Codex approve)

After this branch is reviewed and scripts are present on the live tree (or the
approved command uses the merged commit), the exact launch is:

```bash
tmux new-session -d -s p8_fixedq_attm \
  "cd /home/kojiek/MeanAudio && source /home/kojiek/venvs/dac/bin/activate && \
   EXPERIMENT_RUN_MODE=fresh bash scripts/training_pipelines/sequence_phase8_fixedq_attm.sh \
   2>&1 | tee -a /home/kojiek/logs/phase8_fixedq_attm_sequence.log"
```

**This proposal does not execute that command.**

## Resource / ETA (order-of-magnitude)

| Stage | Estimate |
|---|---|
| Fixedq9 FT 100k | ~3.3 h (S2-scale ~6.7 h / 200k) |
| Fixedq9 MusicCaps eval | ~11–15 min gen + metrics |
| Matched NoQ FT 100k | ~3.3 h |
| Matched NoQ eval | ~11–15 min |
| Paired CLAP bootstrap 5521×2 | ~1–2 h GPU |
| **Total** | **~8–10 h** wall on one GPU |

## Rollback / stop

- Fresh mode refuses existing exp dirs / contracts / eval outputs.
- Immutable per-arm contracts pin source hash, TSV hash, critical file hashes.
- Audit fails closed on Hydra drift, wrong eval flags, wrong iteration, wrong audio count.
- Monitor never auto-stops; human/Codex SOL stop authority remains outside this proposal.
- Loss plateau near 0.98 is **not** a stop reason.

## Official caption inventory (ATTM prep, non-blocking)

```bash
python scripts/preprocess/build_official_caption_inventory.py \
  --local-tsv /mnt/HDD/kojiek/phase4_jamendo_data/phase8_legacy_catalog_train.tsv \
  --official-qwen-json /path/to/official_qwen.json \
  --official-musicflamingo-json /path/to/official_musicflamingo.json \
  --output /home/kojiek/logs/phase8_fixedq_attm_monitor/official_caption_inventory.json
```

CPU-only; no caption encoding; no GPU.  ATTM 90.14 official scoring stays
blocked until the exact 100-prompt CSV is supplied.

## Self-tests run in worktree

```bash
cd /home/kojiek/grok-worktrees/phase8-fixedq-attm-chain
source /home/kojiek/venvs/dac/bin/activate
python scripts/tests/selftest_phase8_fixedq_attm_chain.py
```

Does **not** load the live 2.4 GB checkpoint.

## Known risks

1. Live launch requires scripts to exist under `/home/kojiek/MeanAudio` (merge or
   copy after Codex approval); worktree-only files are not visible to live train.
2. Fixed-Q TSV write on first sequence start mutates HDD data dir; path is new and
   deterministic, but disk must allow write.
3. Paired bootstrap depends on `laion_clap` + 89.98 weights; GPU preferred.
4. Source EMA final is checked for existence but training resumes from
   `ckpt_last` init path (same pattern as Q-safe FT).
5. Historical Q-safe / S2-Q ablation arms are **not** re-run and must not be
   confused with these prefixes.
