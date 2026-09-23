# MeanAudio — Claude Code 上手指南

## 專案概覽

**MeanAudio** 是文字驅動音訊生成系統，兩階段訓練：
- **Stage 1**：FluxAudio（Flow Matching，單向 ODE）
- **Stage 2**：MeanAudio（Mean Flow，更快推理）

**目前重心**（2026-07-20）：
1. **P8 legacy repro 完成** — MusicCaps CLAP **0.1684**（`--quality_level 9` + NoMask；對照歷史 q=9 條件 0.1907，delta −0.022 在 audit ±0.03 內，量級符合 S1-effective-q-training penalty ~0.02）。首次 pipeline eval 誤用 `--no_q` 得 0.0134（Q-trained 模型 + q=10=uncond 記號 → unconditional 模式），已修：pipeline 從 `USE_Q_CONDITIONING` 推導 eval 旗標、audit 目標改 0.1907、無效產物存 `*_noq_invalid`。詳見 forensics addendum（`docs/experiments/history/phase8/phase8_baseline_forensics_2026_07_17.md`）+ memory `project_legacy_repro_noq_eval_trap_2026_07_18.md`
2. **✅ `phase8_catalog_matched_noq` clean full 完成** — S1/S2 NoQ + NoMask + `--no_q`，MusicCaps **CLAP 0.1888**（CE 5.7252 / CU 6.4241 / PC 4.8893 / PQ 6.4174）；5,521/5,521 audio、final contract audit PASSED。訓練中唯一一次 AMP grad NaN 已恢復，ckpt/EMA 無 corruption；舊 Grok job `019f798b2408` 已取消。
3. **🟢 Phase8 clean S2-only Q 因果對照 RUNNING**（tmux `p8_s2_q_ablation`，2026-07-20 12:19 起）
   - 共用上述 clean-NoQ S1 400k；依序跑 `phase8_catalog_matched_s2_realq` → `phase8_catalog_matched_s2_shuffledq`，各自 S2 追加 200k。
   - Real-Q 用原始 per-row `q_level`；Shuffled-Q 只用 seed 424242 打亂 Q（資料順序、id/caption、NPZ/cache、Q histogram 不變）。MusicCaps q9 primary、q6 secondary。
   - 目的：Real-Q 必須同時超過 NoQ 0.1888 與 Shuffled-Q 才支持 Q information 有貢獻；q9 ≥0.1998 才達歷史最佳門檻。
   - Grok durable 5m watcher job `019f7dc10ba5`；任何停止須取得 Codex SOL `stop_authorized=true`，Grok 不可自行停止或修改 live run。Handoff：`docs/experiments/archive/ops/phase8_s2_q_ablation_grok_handoff_2026_07_20.md`。
4. **Phase 9 multi-cap clean rebuild**（P0，disk-blocked）— 需 ~413G，HDD 僅 ~314G free；tooling 已修好，等空間後再 encode
5. **Music Flamingo caption ablations** A1–A6 已完成（見 `docs/experiments/results/phase8/music_flamingo_ablation_todo.md`）
6. **Qwen collapse diagnostic** EXP-A~H 完成；**ISMIR 2026 paper 487** reviews 已歸檔

**目前最佳（paper primary）**：`LP-Rnd-Q`（Phase 7 V1，`JamendoFull-Random-MeanSim-Q`）— Jamendo + MusicCaps 跨 benchmark 最穩。歷史 10-exp 定稿見 `docs/experiments/results/benchmarks/ten_exp_full_benchmark.md`。

**4-token paper-facing 命名**（2026-05-08 統一）：`{Caption}-{Sel}-{Q}` 預設 MC eval；非預設加括號 `(JMQ)` 等。Phase ID → 4-token 速查表見 `docs/experiments/phase_status.md` 頂端。Music Flamingo 實驗沿用 pipeline/exp 名（如 `mfshort100k_direct`），尚未併入 4-token 表。

> Phase 編號作內部追蹤；對外報告/論文一律用 4-token 命名（範例：`LP-Rnd-Q`、`Qwen-Rnd-NoQ (JMQ)`）。

---

## 文件導覽

| 何時查閱 | 文件 |
|---------|------|
| Phase 狀態、實驗進度、P9 cache 失效、P8 clean retrain | `docs/experiments/phase_status.md` |
| 完整實驗數字（Jamendo + MusicCaps 10-exp + P8V4） | `docs/experiments/best_results.md` |
| Music Flamingo caption ablations（A1–A6 結果） | `docs/experiments/results/phase8/music_flamingo_ablation_todo.md` |
| **MF 全覆蓋線**（038/039 queue job、判讀規則、early-kill gate） | `docs/experiments/mf_full_coverage_line.md` |
| **跨 captioner rotation 線**（046/047、slot0/slot1/MF vs 全 Qwen random） | `docs/experiments/mixcap_01m_line.md` |
| **c2p0 slot4 / slot4v2 剝數字線**（050/055，**已收線**：剝數字在 MusicCaps 沒有可測幫助；被 057 slot0nm 取代。2026-09-17 兩個 overlay 已刪、run 搬 HDD，語料 TSV 仍在 `~/exps_nvme/slot4{,v2}/arm_inputs/`，要重跑須先重編 overlay） | `docs/experiments/caption2p0_slot4_no_digits_line.md` |
| **crest 介入線**（061、固定 LUFS 移動 crest，檢定 051 的 +0.44 crest↔PQ 關聯是否因果） | `docs/experiments/crest_intervention_cfg3_20260916.md` |
| **crest 介入結果**（061 收線：主斜率反號但為處理劣化污染；crest 不可當訓練目標） | `docs/experiments/results/crest_intervention_cfg3_20260916_results.md` |
| **絕對響度階梯線**（063、純量增益 −18→0 dB 階梯＋+6 dB 餘裕子集，檢定 051 的響度效應是單調還是倒 U；只有 p6 真的比原檔大聲） | `docs/experiments/gain_ladder_aes_20260918.md` |
| **絕對響度階梯結果**（063 收線：PQ/CU 越小聲越高、CE/PC 頂點 −6～−3 dB、**CLAP 越大聲越高** → arm 間 PQ/CLAP 比較都要先鎖響度） | `docs/experiments/results/gain_ladder_aes_20260918_results.md` |
| **底線階梯線**（065、純量衰減到 −120 dB＋x42-dpl 向下 limiting，F/Q 雙模式拆開評分器／PCM_16 格式／LUFS 門檻三種底線；pilot：PQ 頂點約 −21 dB，063 的單調只在 −18～0 成立） | `docs/experiments/floor_ladder_aes_20260918.md` |
| **底線階梯結果**（065 收線：PQ/CU 頂點 −21 dB、CE/PC/CLAP 原音量最高；評分器底線 AES ≈−96～−108、CLAP ≈−120；PCM_16 格式損傷 PQ 從 −42 dB、CLAP 從 −66 dB；limiter 向下壓同響度 PQ −0.57～−0.07，躲底線不划算；canonical CLAP 的 −0.006 偏移只在接近滿刻度存在） | `docs/experiments/results/floor_ladder_aes_20260918_results.md` |
| **淺層向下 limiting 線**（072、−18～0 dB 補 064／065 之間的空白，四個 AES 軸一起） | `docs/experiments/shallow_limit_ladder_aes_20260922.md` |
| **淺層向下 limiting 結果**（072 收線：最輕一檔就要 8.4 dB GR、付 −0.24 PQ，輕度 limiting 無免費區間；代價在 GR 25 dB 已達飽和 94%；CU/CE/PC 首次讀出；GR < 8.4 dB 量不到） | `docs/experiments/results/shallow_limit_ladder_aes_20260922_results.md` |
| **微幅向下 limiting 線**（074、把 limiter 曲線從 −3 dB 接到原音量；階距訂在 L\* 軸上，GR 被動取樣到 3.2～8.7 dB，補 072 造不出來的區間） | `docs/experiments/micro_limit_ladder_aes_20260922.md` |
| **微幅向下 limiting 結果**（074 收線：GR 3.15 dB 就付 −0.027 PQ，四個 AES 軸 CI 都不跨零；邊際代價隨 GR 遞增 → 代價曲線是 S 形、從零連續長出但無免費區間；CLAP 要 GR 4.6 dB 才可測；曲線已從 −84 dB 接到 +6.8 LU；GR < 3.15 dB 造不出來） | `docs/experiments/results/micro_limit_ladder_aes_20260922_results.md` |
| **limiter 響度提升線**（064、x42-dpl 把 LUFS 推高＋響度對齊雙胞胎拆 level/processing；064b 換 own/alimiter/hyrax/loudnorm 驗穩健性） | `docs/experiments/limiter_loudness_aes_20260918.md` |
| **limiter 響度提升結果**（064：T14 PQ −0.220；PQ/CU/CE/CLAP 全降、只有 PC 升 → limiter 提升 LUFS 不是免費提升。064b：方向對 5 種 limiter 穩健，響度部分 ≈ −0.10 PQ 穩健，處理部分依 limiter 差 12 倍） | `docs/experiments/results/limiter_loudness_aes_20260918_results.md` |
| **D2 雜訊負樣本訓練線**（075、25k 程式化劣化列（noise/clip/lowpass/bitcrush/crackle，LUFS 對齊）加進 066 語料；lab（caption 點名缺陷）vs unlab（不點名）vs control066；操弄檢查用 D1 probe 的波形簽名不用 CLAP；主端點 negprompt 增益 ΔPQ 差 ≥ 0.19） | `docs/experiments/d2_defect_negsample_075_20260923.md` |
| **guidance 幾何：先 normalize 再減（073 延伸，已收線）**（純 CFG 上是 no-op；fidelity8 上**拆掉了 negative 分支的範數煞車** → 大聲 1.69 LU、crest 崩 1.02、PQ/PC 降，響度對齊後仍在；early-kill 未過，不進全量） | `docs/experiments/results/guidance_geometry_prenorm_20260923_results.md` |
| **guidance 幾何線**（073、ADG 範數保持／APG 正交投影取代樸素 CFG 外插；推論期不重訓，primary=純 CFG cfg4.5 的浪費能否換成 PQ，響度閘門必跑） | `docs/experiments/guidance_geometry_adg_apg_20260922.md` |
| **caption 內容編輯線收線**（2026-09-22：剝數字／去量測／rotation 四類干預在 MusicCaps 全測不出；只有換整個 captioner 動得了 CLAP。含「收線不等於證明」與重啟條件） | `docs/experiments/caption_content_editing_line_retired.md` |
| **slot0nmv2 3-seed 成對 quarter 結果**（066–071 收線：去量測在 MusicCaps CLAP/AES 兩格都無可測效果；CFG0 CLAP CI 下界 −0.0097 未過預登錄非劣性界 −0.0084 → 報 inconclusive；nmv2 一致 crest 較高、略小聲、靜音略多） | `docs/experiments/results/phase8/nmv2pair_three_seed_results.md` |
| **Score-aware Beta timestep 線**（053、arXiv 2606.07387 復現；2026-09-15 收線：base 不 overfit、λ 越大越差、shuffled-S 不一致） | `docs/experiments/tscore_beta_schedule_line.md` |
| **Q 解析度 × caption rotation 線**（048/049、013 true-random × K=3/K=10 balanced） | `docs/experiments/c2p0_truerandom_q_granularity_line.md` |
| Qwen collapse 10-model audit | `docs/experiments/history/phase8/qwen_collapse_audit_10model.md` |
| Qwen collapse root-cause EXP-A~H | `docs/experiments/history/phase8/qwen_collapse_root_cause_2026_05_08.md` |
| Qwen single-cap rerun 總結 | `docs/experiments/history/phase8/qwen_rerun_summary.md` |
| Phase 9/9.5 實驗設計、multi_cap 機制 | `docs/experiments/history/phase9/phase9_design.md` |
| Phase 9.5 完整總結（歷史；multi-cap 結果已失效） | `docs/experiments/history/phase9/phase9_5_summary.md` |
| 訓練 / Eval 時間估算 | `docs/experiments/training_time_estimates.md` |
| ISMIR 2026 reviews + correctness plan | `docs/reviews/ismir2026-487-promptcc/` |
| **新方向盤點**（CFG normalize 順序／caption 合寫／enc-dec／雜訊負樣本四條線與優先序） | `docs/meetings/2026-09-22_new_directions_cfg_encdec_noise.md` |
| 教授討論紀錄（Lane A/B/C、data leakage） | `docs/meetings/` |
| Meta Audiobox Aesthetics 指標細節 | `docs/metrics/audiobox_aesthetics.md` |
| 五首固定主觀 prompt + 下載指令 | `docs/eval/subjective_prompts.md` |
| Phase 4→8 完整對比表（歷史） | `docs/experiments/history/phase4-phase8/Phase4_to_Phase8_Complete_Summary.md` |
| 文獻啟示（Audiobox、Resonate、PE-AV） | `docs/literature/Literature_Insights.md` |
| **Negative prompting / prompt engineering 文獻定位**（QA-MDT、NAG/VSF、APG、Open Prompt Challenge） | `docs/literature/negative_prompting_and_prompt_engineering_2026_09_04.md` |
| **D1 缺陷方向 probe 結果**（缺陷 prompt 推不進參考臂定義的缺陷區；波形簽名對不上、沒音樂時模型以靜音逃逸 → 負向 prompt 的上界被訓練分布釘死。**CLAP 的缺陷 caption 分數被純靜音刷過真削波／真低通，不可當缺陷驗收指標**） | `docs/experiments/results/d1_defect_direction_probe_20260922_results.md` |
| **負樣本進訓練的文獻定位**（四家族：推論期負向 prompt／訓練期品質條件 QA-MDT／偏好對 DPO Tango2·MusicRL／負向分支換成模型 NPO·autoguidance；與 CFG 的關係；三個可行動選項） | `docs/literature/negative_samples_in_training_and_cfg_2026_09_22.md` |
| **品質／美學指標效度文獻定位**（SongEval + RF-Limits；PC 相關性最低 0.408、乾淨音訊上指標塌到 chance、crest-as-reward 崩潰、可抄的 protocol） | `docs/literature/quality_metric_validity_2026_09_18.md` |
| 早期累積實驗數字（→ 改查 `best_results.md`） | `EXPERIMENT_LOG.md` |

---

## 主動更新規則（Claude 必讀）

> CLAUDE.md 與 `~/.claude/projects/-home-kojiek-MeanAudio/memory/` 必須在下列事件發生時主動更新，不等使用者提醒。

**必須更新 CLAUDE.md / docs：**
1. Phase 狀態轉換 → `docs/experiments/phase_status.md`
2. 新實驗設計確認 → 對應 `docs/experiments/*.md` 或新 doc
3. 環境限制發現 → 本檔「環境」段
4. 最佳數字更新 → `docs/experiments/best_results.md`
5. 教授新方向 → 新增 `docs/meetings/YYYY-MM-DD_*.md`
6. 關鍵檔案/指令改名 → 本檔對應段
7. Ablation / pipeline 完成 → 對應 todo doc + `scripts/README.md` 必要時更新

**必須寫入 memory：**
1. 使用者糾正做法 → `feedback_*.md`
2. 非直覺選擇被確認 → `feedback_*.md`
3. 長期研究原則（diversity hypothesis、data leakage 等）→ `feedback_*.md`
4. 硬體/環境特殊性 → `project_*.md` 或 `reference_*.md`
5. 使用者身份偏好（首次知道）→ `user_*.md`

**流程**：做決策 → 立即更新 → 更新 `MEMORY.md` index → 繼續任務。

---

## 環境

```bash
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
```

Music Flamingo captioning 另用 `~/venvs/music_flamingo/`（見 memory `reference_music_flamingo_setup.md`）。

---

## 目錄結構

完整 repo 地圖見 `STRUCTURE.md`；scripts/ 子目錄索引見 `scripts/README.md`。

```
MeanAudio/
├── meanaudio/                       # 套件：model/、data/、runner_*.py、eval_utils.py
├── train.py / eval.py / infer.py    # 入口
├── train_pipeline.sh                # CANONICAL S1 → migrate → S2 → eval
├── set_training_stage.py            # 切 Stage 1/2（patch runner）
├── migrate_stage1_to_stage2_ckpt.py # S1→S2 ckpt 轉換（吃 ckpt_last.pth，不是 ema_final.pth）
├── scripts/                         # 所有 helper（training_pipelines/, eval/, preprocess/, analysis/, legacy/, runs/）
├── config/ sets/ data/ training/    # configs、latent stats、symlinks、訓練工具
├── av-benchmark -> .external/av-benchmark
├── .archive/                        # 隱藏歷史/次要資料
├── .external/                       # 隱藏外部 checkout：av-benchmark/
├── .side_projects/                  # 非 MeanAudio 主線的 side projects
└── docs/                            # experiments/ meetings/ eval/ metrics/ literature/ reviews/
```

資料路徑：`/mnt/HDD/kojiek/phase4_jamendo_data/`（NPZ + TSV）；舊 eval audio：`/mnt/HDD/kojiek/MeanAudio_eval_output_OLD/`（2026-05-16 reorg 搬出 repo root）

**實驗變體 pipeline** 一律在 `scripts/training_pipelines/`，啟動：`bash scripts/training_pipelines/<name>.sh`（每個都自帶 `cd "$WORK_DIR"`）。

**近期架構變更**：`text_attention_mask` 已接入 joint attention + mean pooling（commit `a148aaf`），T5 padding 不再污染 text conditioning。

---

## 實驗前 Checklist

> **違反任何一項都可能燒掉數小時 GPU 時間。**

1. **腳本推 GitHub**：`~/research` 下所有會用到的腳本先 `git add && commit && push`
2. **Caption 多樣性 sanity check**（唯一率 < 90% 停止訓練）：
   ```bash
   cd ~/research/meanaudio_training && python sanity_check_50.py
   ```
3. **Multi-cap / NPZ pairing audit**（若涉及 multi-caption cache）：
   - 必須走 `npz_cache_train.txt` mapping，**禁止** row-index → `i.npz`
   - 用新版 `gen_multicap_npz.py --gt-cache ...` + `validate_multicap_npz.py`（v2 manifest / mean-std exact equality）
   - 舊 Phase 9 cache **不可**重訓或引用為 method result
4. **Eval TSV 確認**（必須明確傳 `--tsv`，不依賴 hardcode default）：
   - **預設：MusicCaps**（ISMIR 黃金標準，2026-04-19 定為主要 benchmark）：`/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv`（5,527 筆，~11 min eval）
     - 理由：ISMIR benchmark 發表用；無 data leakage（訓練 Jamendo、eval MusicCaps）；16x 比 Jamendo 快
   - **次要：Jamendo** 歷史比較：`phase4_test.tsv`（90,063 筆，~3.1 hr eval）— 只在需要跟 Phase 4-8 舊數字對照時才跑
   - **快速 sanity**：`eval.py` **無** `--num_samples` 參數；要做 2048 筆小 subset sanity 須先 `head -n 2049 <TSV> > <TSV>_2048.tsv` 切檔再傳 `--tsv`。`eval_metrics.py`（與舊 `phase4_eval.py`）的 `--num_samples` 只控制 FAD 抽樣數，不影響生成數量；要小量 smoke test 用 `--limit N`
5. **Caption-source 換訓練時**：必須有 same-distribution eval；跨 style 結果只能寫 generalization，不是 conditioning test

---

## 實驗中/後 Monitoring（launch ≠ done）

> **啟動實驗不等於完成 — 不 monitor 就等於沒跑（可能悄悄掛掉、stall、或產生 NaN/全 0 音檔而你不知道）。**

**每次啟動 tmux 訓練或 eval job 後，必須主動排定 monitoring：**

1. **第一次 check（啟動後 1~2 分鐘內）**：確認 job 真的在跑
   - `tmux ls` 看 session 還在
   - `nvidia-smi` GPU 有吃到（memory > 1 GB、util > 0%）
   - 讀 log 開頭：無 import error / OOM / checkpoint load error
   - 有 progress bar（`it/s` 合理，不是卡住）

2. **定期 check（用 `ScheduleWakeup` 或 `/loop`，間隔依 job 長度）**：
   - **短 job（<30 min）**：每 5~10 min check 一次
   - **中 job（30 min ~ 2 hr）**：每 20~30 min check（用 `ScheduleWakeup delaySeconds=1200~1800`）
   - **長 job（>2 hr）**：每 30~60 min，關鍵轉折點（Stage 切換、eval 開始）加 check

3. **每次 check 要看的東西**：
   - tmux session 還活著（`tmux ls`）
   - GPU 還在跑（util > 0%，memory 沒降到閒置）
   - log tail 有新行（沒 stall）
   - 無 exception / NaN / OOM traceback
   - 預期的階段轉換有發生（e.g. gen 完換 metrics、Q 切換到下一個）

4. **結果 sanity check（每個階段完成時）**：
   - Gen 完：`ls <output>/audio | wc -l` 接近預期數、抽一個 `soxi` 看長度/取樣率正常、檔案 size > 0
   - Metrics 完：`cat metrics.txt` 無 NaN、數字在合理範圍（CLAP 0.05~0.25、CE 5~8、PQ 5~8）
   - 發現異常 → 先懷疑 bug（見 `memory/feedback_suspect_bug_before_explaining.md`）

5. **禁止「啟動後就當完成」** — 沒排 monitoring 等於沒做這份工作。

---

## GPU idle backlog policy

> **GPU 不該 idle；但只有已定義、可恢復、可插隊的實驗，才能在 idle 時自動接手跑。**

4 個 guardrail（詳見 `memory/feedback_gpu_idle_backlog_policy_2026_04_21.md`）：

1. 只能自動開**已排隊、已定義目的**的實驗 — 不能因 GPU 空就臨時發明題目
2. 必須**可恢復** — checkpoint 有、resume 驗證過、save interval 合理
3. 優先級分類：
   - **P0**：短 probe / sanity / bug verification — 隨時可插隊
   - **P1**：關鍵 control run（例 P8 bug-free retrain、clean multi-cap rebuild）
   - **P2**：探索型長實驗 — 沒更明確 backlog 時才跑
4. 啟動時**留紀錄**（跑什麼、為什麼現在跑、checkpoint 點、被插隊時的停機點），不默默開

**流程**：自動啟動前先逐項檢查這 4 個 guardrail，檢查通過才動。

### 走 queue 還是直接跑（2026-09-23 定）

Discord 通知只由 `~/gpu_queue` 的 host 發（seat／done／failed／idle）。直接用 tmux＋自己握 `gpu0.lock` 跑的 job **完全沒有通知**，也沒有 contract 檢查與 done/held 分類（2026-09-22 的 D1 probe、prenorm pilot、075 chain 都因此靜默）。

- **預期 > 30 分鐘的 job（訓練、全量 eval）→ 一律走 queue**（`p2/pending/NNN_*.sh` ＋ contract）。
- **短 probe／pilot 可直接跑，但腳本開頭必須掛通知 trap**：
  ```bash
  set -eo pipefail
  source "$HOME/MeanAudio/scripts/notify_lib.sh"
  notify_on_exit "<exp_name>" "$LOG"   # 立刻發 start，結束時依 exit code 發 success/failure/interrupted
  ```
- 已在跑、沒掛 trap 的直接 job：用 `scripts/notify_when_pid_exits.sh <pid> <exp> <log> <done_marker> [start_epoch]` 補 watcher（bash 啟動時已緩衝整支腳本，改原腳本無效）。
- 直接跑的理由要寫進實驗 doc（為什麼不走 queue）。

---

## 訓練流程

> 長時間任務一律用 **tmux**（超過 5 分鐘的 job 都用 `tmux new-session -d -s <name>`）
> 連續任務一律用 **`&&`** 串接，不要分段等使用者回來
> 修改主 repo 檔案時路徑必須是 **`~/MeanAudio/`**，不是 worktree
> shell 外層也要 `set -eo pipefail`（見 `feedback_pipefail_silent_crash_2026_04_22.md`）

```bash
tmux new -s phaseX
cd ~/MeanAudio && source ~/venvs/dac/bin/activate && bash train_pipeline.sh
```

`train_pipeline.sh` 只需改參數區塊：`EXP_PREFIX` / `S1_ITERATIONS` / `S2_ITERATIONS` / `LEARNING_RATE`。

**S1 必須** `model=fluxaudio_s`（不是 `meanaudio_s`）— 見 `feedback_pipeline_s1_uses_fluxaudio_s.md`。

---

## 關鍵架構：Quality Conditioning

**q_embed**：`nn.Embedding(11, hidden_dim)` — idx 0~9 = 品質等級，idx 10 = null token。

- `FluxAudio.predict_flow(q=None)` → `q=None` 填 null token（**必須是 10**；歷史 bug 曾填 9）
- `FluxLoss.loss(q=...)` → conditional pass 用 q、unconditional 用 null
- `set_training_stage.py --stage {1,2}` → patch runner 切 FluxAudio / MeanAudio

**Checkpoint S1→S2 遷移**（`migrate_stage1_to_stage2_ckpt.py`）：
1. `t_embed → r_embed`（S2 新增）
2. **保留 S1 已訓練的 q_embed**（Phase 6 V2+）
3. 清除 optimizer / scheduler state
4. **輸入必須是 `ckpt_last.pth`**，不是 `ema_final.pth`（後者缺 `it`/optimizer keys）

**Backward compat**：pre-Phase 6 checkpoint 無 `q_embed.weight` → `load_weights()` 自動歸零 + WARNING。

**已修結構性 bug（必須知道）**：
1. `networks.py` MeanAudio `q=None` 曾填 9 → 已改 10
2. `runner_meanflow.py` `text_f_undrop` 別名 → 已 `.clone()`
3. `runner_flowmatching.py` 曾完全不傳 q → 已修 6 處（歷史 Phase 6–8 +Q = half-Q）

---

## Eval

主要指標：**CLAP ↑、CE ↑、PQ ↑**（FAD 僅歷史參考；PE-AV 作 fine-grained retrieval）。

### ⚠️ q 旗標選擇規則（混用會啟用未訓練 embedding 污染結果）

| 訓練時 q conditioning | Phase 例 | Eval 旗標 |
|---|---|---|
| **true**（q 0~9 訓練） | 6 V2、7 V1/V2、9 V2、9.5 V2 | `--quality_level N` |
| **false**（永遠 null token） | 8、9 V1、9.5 V1、MF NoQ 系列 | `--no_q` |
| **pre-Phase 6**（無 q_embed 層） | 4 V2、5 V1/V2 | 兩者等價 |

詳見 `memory/reference_eval_q_flag_rule.md`。踩坑：2026-04-17 Phase 8 誤用 `--quality_level 9` → CLAP 0.1907（污染），`--no_q` 正確 0.1851。

### 指令

**標準 eval（2026-09-18 定）**：MusicCaps 5521 / MeanFlow 25 步 / seed 42 / fp32 / NoMask，**每個 checkpoint 跑兩格：CFG0 與 CFG3+neg**（cfg 3.0 + 固定 fidelity8 負向 prompt）。一支 wrapper 跑完兩格：

```bash
bash scripts/eval/mc_mf25_eval.sh <EXP> exps/<EXP>/<EXP>_ema_final.pth {--no_q | --quality_level N}
#   [--mask]            模型是開 text mask 訓練的才加（label 加 _mask）
#   [--gen_tsv PATH]    prefix 訓練（P8 V4）用 prefixed TSV 生成；CLAP 一律對原始 caption 算
#   [cfg0] [cfg3neg]    只跑其中一格
# Q 模型至少報 q9 與 q0（各跑一次 wrapper）
```

輸出：`~/eval_output_nvme/<EXP>_mc_mf25_{cfg0,cfg3_neg}[_qN]/`（`audio/`、`<label>/{metrics.txt,metrics.json,per_clip.tsv}`、`<label>_REPORT.json`）。有 REPORT 就跳過；沒有 REPORT 的殘缺目錄會**整格重跑**，因為 `eval.py` 整個 run 只 seed 一次 RNG，且跳過已存在檔案時不抽 noise，補齊的 clip 會跟一次跑完的不同。生成旗標與舊的 `caption10s_pipeline/eval_musiccaps_mf25.sh`（CFG0）/ `mc_mf25_cfg3neg_eval{,_q}.sh`（CFG3+neg）相同，已驗證音檔逐樣本一致；舊 wrapper 被 contract 綁 sha，凍結不改。

只算 metrics（音檔已存在）：

```bash
python scripts/eval/eval_metrics.py --gen_dir <DIR>/audio \
    --tsv /mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv --exp_name <LABEL> --out_dir <DIR>
```

`eval_metrics.py` = CLAP batch 1 + AES + level（LUFS / RMS / crest / 靜音 < −45 dBFS）＋選用 `--fad`；`--tsv` 必填、缺檔或評分失敗直接失敗（`--allow_missing` 才放行）。跨 arm 比較 AES/CLAP 前先看 `level_lufs_mean` 與 `level_silent_n`。完整數字見 `docs/experiments/best_results.md`。

**CLAP 一律逐檔（batch 1）**（2026-09-18 定）：laion_clap 在 batch > 8 時 padding 不同，b32 比逐檔高 +0.004～+0.025 且會翻排名（062）。`eval_metrics.py` 沒有 batch 參數；新 sweep 要算 CLAP 就 `from eval_metrics import score_clap`，不要自己寫 batch 迴圈。舊的 `~/research/meanaudio_eval/phase4_eval.py` 凍結不改（歷史 contract 綁 sha；CLAP 本來就是逐檔，兩者逐位一致）；`negprompt_reeval_full_arms.py`、`novocal_reeval_full_arms.py`、`negprompt_ablation_matrix.py`、`attm_protocol_eval.py` 是 b32 的歷史 driver，只用來重現舊表。

主觀評估五首 prompt 見 `docs/eval/subjective_prompts.md`（25 steps + **cfg 0.5**）。主觀試聽／`infer.py` **沒有負向 prompt 時不要用 cfg ≥ 2.0** — 在非 null Q + 高能量 prompt 會觸發波形飽和（crest < 2.0，2026-04-21 於 subjective_ab v3 踩坑，mc18_abl_A–J 證實，york135 指出）。標準 eval 的 CFG3+neg 格是 negprompt 消融定的 cfg 3.0，飽和用 metrics 的 `level_clipped_n` / `level_crest_mean` 監看。

`infer.py` **沒有 `--no_q`** — NoQ 模型用 `--quality_level 10`（null token workaround）。

---

## NEVER

- **不要動 `meanaudio/model/networks.py` 裡的 MeanAudio 類別**（Stage 2 架構），只改 FluxAudio
- **不要在 worktree（`~/.claude/worktrees/...`）改主 repo 檔案**，必須是 `~/MeanAudio/`
- **不要混用 q 旗標**（見上方規則）
- **不要 commit 臨時腳本**：`run_*.sh` 已在 `.gitignore`，臨時腳本照此命名
- **不要用 CLAP 過濾訓練資料同時用 CLAP eval**（data leakage — 教授 2026-03-27 原則）
- **不要把 Phase 9/9.5 multi-cap 歷史數字當 method result**（2026-07-16 pairing 全量錯配）
- **不要對 Qwen-trained 模型只用 LP/MusicCaps prompt 就下「沒學到」結論**（需 same-distribution eval）

---

## Git

- Remote: `https://github.com/Lanternko/MeanAudio.git`
- Branch: 直接在 `main`，每個 Phase 結束後 tag
- Identity: `lanternko <jerry86012@gmail.com>`（token 已存在 credential store）
- Commit 格式：`phaseN_vX: 簡短描述` + bullet 改動
