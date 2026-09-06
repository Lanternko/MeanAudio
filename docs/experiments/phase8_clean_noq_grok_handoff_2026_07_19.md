# Phase 8 clean-NoQ retrain：Grok `/loop` 交接指令

日期：2026-07-19（Asia/Taipei）

## 已註冊的實驗問題

驗證 `phase8_legacy_repro` 的 CLAP 0.1684 是否主要由「S1 開啟 per-row Q」造成，而不是 cache、caption pairing、text mask 或 eval Q flag 造成。

這是一個 catalog-matched clean-NoQ control。唯一要移除的變因是 Q pathway；禁止在途中改資料、seed、batch、LR、mask、CFG 或 eval protocol。

| 項目 | 固定契約 |
|---|---|
| S1 | FluxAudio，400,000 iter，`use_q_conditioning=false`，因此 runner 傳 `q=None`、network 使用 q=10 |
| S2 | MeanAudio，再訓練 200,000 iter，checkpoint total it=600,000，`use_q_conditioning=false`，同樣使用 q=10 |
| Text mask | S1/S2 都是 `use_text_attention_mask=false`（legacy NoMask） |
| Eval | MusicCaps、1-step MeanFlow、CFG 0.5、seed 42、`--no_q --no_text_attention_mask` |
| Data | 251,599 rows；`phase8_legacy_catalog_train.tsv` + `npz_cache_train.txt` + `phase8_legacy_matched_npz` |
| 其他 | single-caption、`multi_cap=false`、batch 8、accumulation 1、LR 1e-4、warmup 1,000 |

Q 結論很明確：本實驗 **S1 與 S2 都不要 Q**。`S1 NoQ + S2 Q` 不是 clean NoQ；`S1 Q + S2 NoQ` 也不是。舊 April bug-era 的 S1 q10 / S2-uncond q9 是另一個 forensic emulation，不得混進主實驗。

## 現在的實際狀態

`p8_catalog_noq` 已在所有 contract/monitor 修改完成後，於 2026-07-19
00:52 以 `EXPERIMENT_REGIME=clean_noq`、`EXPERIMENT_RUN_MODE=fresh`
從零啟動，請勿再啟動第二份。00:38 的 pre-hardening run 已在 it=6,050
優雅停止並完整保留於
`/home/kojiek/logs/archive/phase8_catalog_matched_noq_pre_hardening_20260719_0052/`
及同名 `_pre_hardening_20260719_0052` experiment 目錄，不是本次結果的一部分。

首次 runtime audit 已確認：

- S1 從零開始，Hydra `model=fluxaudio_s`。
- `use_q_conditioning=false`、`use_text_attention_mask=false`、`multi_cap=false`。
- train/val 的 TSV、cache、NPZ 路徑完全相同且正確。
- structural/semantic cache gates 都 passed。
- immutable launch contract 是 `clean_noq/fresh`，沒有 warning。
- strict S1 runtime audit passed；監控結果 healthy、無 ALERT，loss 約 0.99 且為有限值。
- Grok 4.5 durable loop job `019f798b2408` 每 5 分鐘監控一次；tmux 是
  `p8_noq_grok_loop`。scheduler 已確認只有這一個 job。

2026-07-19 08:12 的 it=228200 曾出現一次 `grad_norm:nan`；loss 有限，
後續約 1,700 iter / 34 筆 log 全部恢復。舊 monitor 與舊 loop prompt
錯把這個 AMP GradScaler 已跳過的單次 optimizer step 當成永久 hard
failure，Grok 於 08:15 誤送 Ctrl-C。it=229935 的 ckpt_last 已全量掃描：
weights、optimizer、EMA non-finite 都是 0。這是 monitor policy incident，
不是 Q/config/training corruption。2026-07-19 16:43 已用
`EXPERIMENT_RUN_MODE=resume` 載入 it=229935 的 weights/optimizer/scheduler/EMA；
16:44 正常通過 it=230000，loss/grad/lr 有限，runtime audit passed、monitor
healthy、ALERT 已清除。不得重開 fresh。

runtime Hydra、immutable launch contract 與 stage-transition audit 必須三者一致；任一漂移都是 hard failure。

## 直接交給 Grok 的 `/loop` 指令

以下整段可直接交給 Grok：

```text
/loop 5m

你負責監控既有的 MeanAudio Phase-8 catalog-matched clean-NoQ 實驗。工作目錄固定為 /home/kojiek/MeanAudio；tmux 固定為 p8_catalog_noq。不要啟動第二份訓練，不要編輯 live `/home/kojiek/MeanAudio` 的 code/config，不要改現行 Q、mask、資料、seed、batch、LR、checkpoint 或 eval 參數，也不要刪除任何 artifact。獨立 `grok/*` proposal worktree 的權限依下方分工契約。

實驗契約：S1 use_q_conditioning=false；S2 use_q_conditioning=false；兩 stage use_text_attention_mask=false、multi_cap=false；eval 必須 no_q=True、no_text_attention_mask=True。S1 target=400000；S2 是追加 200000，checkpoint total target=600000。主實驗不是 Q-trained，也不是 April bug-era mixed-Q emulation。

分工契約：Grok 是低成本 monitor/scout/implementation drafter；Codex SOL (`gpt-5.6-sol`) 是 stop、code change 與新實驗的 senior reviewer。Grok 可以分析問題、提出新實驗，並在**獨立 worktree** 建立 `grok/<slug>` git branch、修改/測試/commit；絕對禁止在 live `/home/kojiek/MeanAudio` 切 branch、改檔或 merge，因為目前 pipeline 的 S2/eval 仍會讀 live worktree。worktree 固定放 `/home/kojiek/grok-worktrees/<slug>`。未經 Codex 結構化 verdict 明確 `decision=approve` 且 `execution_authorized=true`，不得執行新訓練/eval、merge、套 patch 或改現行實驗。

Grok proposal 必須寫在其 worktree 內並 commit，至少包含：問題與證據、可證偽假說、單一 controlled variable、固定 baseline/contract、完整 train/eval commands、觀察指標與成功/失敗門檻、資源與 ETA、唯一 artifact prefix、stop/rollback 設計、測試結果。完成後呼叫：

bash /home/kojiek/MeanAudio/scripts/review_grok_proposal_with_codex.sh \
  /home/kojiek/grok-worktrees/<slug> \
  /home/kojiek/grok-worktrees/<slug>/<proposal-file>

Codex verdict 若為 revise/reject、指令失敗、commit 對不上、verdict 過期或 branch 又有新 commit，禁止執行。approve 時只能執行 verdict 的 exact `approved_command`；不得自行加參數。任何會與目前 Phase8 爭 GPU、改 live prefix/artifact 或觸碰 live training 的 proposal，即使其他部分合理，也必須延後到目前 run 完成或重新交 Codex 審核。

每輪先執行：
cd /home/kojiek/MeanAudio
source /home/kojiek/venvs/dac/bin/activate
python scripts/monitor_phase8_clean_noq.py --once

再讀：
/home/kojiek/logs/phase8_catalog_matched_noq_monitor/status.json
若存在，再讀 /home/kojiek/logs/phase8_catalog_matched_noq_monitor/ALERT.json

每輪觀察並記錄：phase、iteration/target、progress_pct、loss、grad_norm、lr、log_age_sec、GPU util/memory/temp、root free disk、contract_audit.status、issues。loss 約 0.98–1.00 的 plateau 在這套 MeanFlow 訓練不是失敗證據；不要用「loss 沒持續下降」自行改實驗。已知的 `Error in extra logging: Could not load libtorchcodec` 只影響額外視覺化，若主 loss 持續更新，不視為 hard failure。

若 status=healthy：不要做任何修改，只在以下事件回報使用者：每 25k iter、S1 400k 完成、S2 啟動、每個 hard/review issue、eval 啟動、final metrics 完成。其餘輪次安靜監控。`transient_amp_grad_overflow` 是 review 訊息；只要後續梯度與 loss 有限，就保持訓練。

S2 的第一份 Hydra config 出現時，額外執行：
python scripts/audit_phase8_clean_noq_contract.py --phase s2 --json-out /home/kojiek/logs/phase8_catalog_matched_noq_monitor/s2_transition_audit.json
必須確認 S1/S2 都是 use_q_conditioning=false，S2 model=meanaudio_s、num_iterations=600000。若不符，立即視為 contract drift。

eval log 第一行 `Eval args` 出現時，額外執行：
python scripts/audit_phase8_clean_noq_contract.py --phase eval --json-out /home/kojiek/logs/phase8_catalog_matched_noq_monitor/eval_transition_audit.json
必須確認 no_q=True、no_text_attention_mask=True、cfg_strength=0.5、num_steps=1、use_meanflow=True、正確 EMA 與 MusicCaps TSV。不要因字典同時顯示 quality_level=9 而誤判；no_q=True 時 q 會是 None/q10，quality_level 欄位不生效。

Incident candidate 定義：contract audit failed；NaN/Inf loss/lr；persistent/dense grad NaN/Inf（連續 >=2、最近 20 筆 >=3、或最近 100 筆 >=10）；連續三個 loss >5；連續三個 grad_norm >100；CUDA OOM；NCCL/ChildFailed/segfault/traceback；訓練 process 消失且 log 超過 1200 秒不更新；訓練中 log 超過 1200 秒不更新；root free disk <50 GB。單次 AMP `grad_norm:nan/inf` 且下一筆恢復只能 review，不是 stop candidate。stale/process/GPU 類問題必須在下一個 5 分鐘輪次再次重現。

**Grok 永遠不能憑 monitor exit code 或自己的判斷直接停訓。** 禁止自行執行 `tmux send-keys ... C-c`、`kill`、`pkill` 或 `tmux kill-session`。出現任何 hard incident candidate 時，先保留 status/ALERT/contract audit/最近 100 行 log，再呼叫 read-only Codex SOL second opinion：

cd /home/kojiek/MeanAudio
bash scripts/adjudicate_phase8_stop_with_codex.sh
cat /home/kojiek/logs/phase8_catalog_matched_noq_monitor/codex_sol_verdict.json

若 Codex 指令失敗、verdict 無法解析、`decision` 是 `continue/escalate`、`stop_authorized` 不是 true、verdict 檔案已超過 10 分鐘、或 tmux/process 已不存在，全部都**禁止送停止訊號**，只能回報使用者並繼續觀察。只有 verdict 同時滿足 `decision=stop`、`stop_authorized=true`、檔案 mtime 在 10 分鐘內，而且重新跑 monitor 後同一 incident 仍存在，Grok 才能對 `p8_catalog_noq` 發送一次 Ctrl-C。送出後不得自行改參數或重啟，並須回報 Codex verdict、exact command、最後 iteration、最近 100 行 log、ALERT.json 與 contract audit JSON。

Final metrics 產生後執行：
python scripts/audit_phase8_clean_noq_contract.py --phase final --json-out /home/kojiek/logs/phase8_catalog_matched_noq_monitor/FINAL_AUDIT.json

品質判讀是實驗結果，不得拿來事後改設定：CLAP >=0.18 = target met，支持 clean-NoQ recovery；0.17–0.18 = partial recovery；0.15–0.17 = 沒支持 recovery hypothesis，但 contract 正確時仍是有效負結果，不可盲目重跑；<0.15 = collapse，需 forensic review。主要對照是同資料 full-Q q9 的 0.1684；historical 0.1851/0.1907 因舊 code 的 q 語義不同，只能作參考，不可冒充完全同條件 baseline。最後同時回報 CLAP、CE、CU、PC、PQ、音檔數、S1/S2 checkpoint iteration，以及所有 contract warnings。
```

## 人工操作備忘

只看一次狀態：

```bash
cd /home/kojiek/MeanAudio
source /home/kojiek/venvs/dac/bin/activate
python scripts/monitor_phase8_clean_noq.py --once
```

看完整 runtime contract：

```bash
python scripts/audit_phase8_clean_noq_contract.py --phase auto
```

目前 run 若是正常進行，不要再執行 full launcher。未來要開全新 prefix 時，才用：

```bash
EXP_PREFIX=phase8_catalog_matched_noq_v2 \
EXPERIMENT_RUN_MODE=fresh \
bash scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq.sh
```

`fresh` 會拒絕任何已有 checkpoint、Hydra、log、audio 或 metrics 的 prefix；只有驗證過的同一 run crash-resume 才改用 `EXPERIMENT_RUN_MODE=resume`。
