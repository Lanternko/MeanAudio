# 084 NegMF：把 fidelity8 負向 prompt 蒸餾進 MeanFlow S2 的 CFG 訓練目標（預註冊）

2026-09-27 設計。狀態：**設計完成、閘門 G0／G1 已過，queue 檔放在 staging，等 operator 放行**（見 §9）。

- 文獻定位：`docs/literature/guidance_distillation_and_negative_in_training_2026_09_27.md`
- 對照組：066／070／071 的 nmv2pair slot0clean control（`phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{14159265,16180339,27182818}`）
- 程式：`meanaudio/model/mean_flow.py`（`guide_*` 參數）、`meanaudio/runner_meanflow.py`（`mf_guide_*` 設定）、
  `scripts/preprocess/build_negmf_guide_feats_084.py`、`scripts/training_pipelines/negmf_084_action.sh`、
  `scripts/eval/mc_nfe1_cfg0_eval.sh`、`scripts/analysis/negmf_084_analysis.py`

## 1. 為什麼是這個實驗

本週四條線都在問「fidelity8 負向 prompt 的 +0.8～1.0 PQ 能不能用別的東西拿到」，答案都是不行：

| 本週結果 | 結論 |
|---|---|
| 073 ADG／APG | 純 CFG 只收回 0.022 PQ（fidelity8 的 2%）；幾何不能取代負向文字 |
| 075 D2 缺陷負樣本 | 訓練資料點名缺陷反而讓負向增益縮小；增益 93–100% 來自前 9 步（t>2/3） |
| 080 autoguidance | 負向分支換成較差的 EMA 快照只有 +0.06 PQ，且與文字負向不可疊加 |
| 081–083（跑中） | QA-MDT 式品質前綴，測的是「訓練期品質條件」這一支 |

共同訊息：**負向槽要的是文字**。但目前所有負向增益都要推論期兩次前向（CFG3+neg）。
MeanFlow 的 S2 訓練目標本來就把一個 ∅ 分支的 CFG 蒸餾進權重（MeanAudio：ω=0.3、κ=0.9，
有效 scale 3）。把那個 ∅ 分支的**文字**換成 fidelity8，就是免老師、免新輸入、
不動 MeanAudio 類別的負向蒸餾。文獻上沒有人在 MeanFlow 目標上做過（見文獻文件 §4 最後一列）。

## 2. 機制與固定點

現行 target（`mean_flow.py` `loss()`）：`v_hat = w*v + k*u_t_c + (1-w-k)*u_t`。
其中 `u_t_c` 讀未 drop 的 caption，`u_t` 讀固定的 null 特徵，兩者都是 stop-gradient 的模型輸出（r=t）。

| | stock（control） | N100（084） |
|---|---|---|
| guidance 分支讀的文字 | null（`empty_string_t5.pth`＝常數向量×77；CLAP('')） | fidelity8（推論路徑編碼） |
| caption 樣本的固定點 | u(c) = 3v(c) − 2u(∅) | u(c) = 3v(c) − 2u(n) |
| drop 樣本（10%）的固定點 | u(∅) = v̄ | u(∅) = 3v̄ − 2u(n) |

- n 從來不是訓練輸入（沒有 caption 等於 fidelity8），所以 u(n) **只由泛化決定**。
  如果網路把 n 當成一般 caption，u(n) 本身滿足 u(n) = 3v(n) − 2u(n)，得 u(n) = v(n)，
  於是 u(c) = 3v(c) − 2v(n)：**把內建 guidance 的參考點從無條件平均 v̄ 換成「fidelity8 類音訊的條件平均」**。
  這正是推論期負向 prompt 的語意，但只在 scale 3、而且只在泛化假設成立時。
- 推論期 stock CFG3+neg 的有效場是 3u(c) − 2u(n)＝約 9v(c) − 6v̄ − 2u(n)，比 N100 的 CFG0 強得多。
  所以 **N100 CFG0 不預期完全追上 stock CFG3+neg**；主端點量的是 CFG0 對 CFG0 的增益，
  並報告「收回比例」R = Δ / G_neg（G_neg = control 的 CFG3+neg − CFG0）。
- 風險：u(n) 是移動目標。若泛化假設不成立，target 可能漂移（loss 發散、靜音逃逸）。
  §6 的停機規則處理這件事。
- **Nhi** 只在 t > 2/3（t=1 是雜訊）的樣本換成 fidelity8，其他樣本維持 null。
  依 `sample_t_r`（lognorm 兩點取 max）約 25.6% 的樣本落在這區，對應推論 25 步的前約 9 步。
  它同時測兩件事：075 的分段結果能不能在訓練期複現，以及限制區間能不能減少 FAD 代價
  （guidance interval 2404.07724：高雜訊端 guidance 主要傷 FID／多樣性）。

**失敗假說（要能被證偽）**：NASA（2412.02687）報告負向只放在老師端再蒸餾，學生稍變差。
如果 084 的 CFG0 增益 ≈ 0，就是這個結論在 MeanFlow 目標上複現，表示負向的效果需要推論期的兩次前向差分。

## 3. Arms 與訓練

| arm | `mf_guide_t_min` | 說明 |
|---|---|---|
| **N100** | 0.0 | 所有樣本的 guidance 分支讀 fidelity8 |
| **Nhi** | 0.6667 | 只有 t > 2/3 的樣本讀 fidelity8 |
| control（已存在） | — | nmv2pair slot0clean，stock target |

- **S2-only 分支**：從各 seed control 的 **S1 `ckpt_last.pth`** migrate，跑 50k S2。
  recipe 與 control 的 S2 **逐項相同**（`caption2p0_nmv2pair_action.sh` slot0clean：NoQ、NoMask、LR 1e-4、BS 8、
  seed 同 control、`cap_index_fixed=0`、`require_text_overlay`、同一份 TSV／cache list／overlay）。
  同 seed＋同起點＝同一條資料順序，唯一差別是 target 的 guidance 分支文字。
- 新增的 Hydra 鍵：`++mf_guide_t5=weights/negmf_084/fidelity8_t5.pth ++mf_guide_clap_c=weights/negmf_084/fidelity8_clap_c.pth ++mf_guide_t_min=…`。
  runner 啟動時會 log `MeanFlow CFG target guidance branch uses … for t > …`，action 會 grep 這一行，沒有就失敗。
- 命名：`phase8_qwen_caption2p0_slot0clean_negmf{n100,nhi}_noq_quarter_s<SEED>_stage2_50000`。
- 額外計算量 ≈ 0（guidance 分支本來就要跑一次前向，只換文字）。

## 4. 閘門（已過）

| 閘門 | 內容 | 結果 |
|---|---|---|
| G0a（只報告） | null 特徵與編碼器的關係 | `empty_string_t5.pth` 是**常數向量（範數 1.32）×77**，不是 T5('')（三種 padding cos −0.158／0.496／0.050）；`empty_string_clap_c.pth` = CLAP('')（cos 1.0000） |
| G0b（擋） | fidelity8 的推論路徑編碼 vs 訓練 overlay 編碼器 | 27/27 有效 token，min cos 0.9999998；CLAP cos 1.0000001 ✅ |
| G1a | `guide_text_f=None` 時與 HEAD 逐位元相同（同 torch＋numpy seed） | ✅ 相同；`set_training_stage.py --check` 仍是 Stage 1、PATCH 片段未動 |
| G1b | 單元測試：N100 全部樣本換、Nhi 只換 t>2/3、Mask 模式直接報錯 | ✅ |
| G1c | 執行期冒煙：Stage 1 模式 40 iter，guide 開、Nhi | ✅ guide log 行在、loss 0.995 有限（見 §4.1） |
| G1d | 1-NFE eval wrapper 冒煙（control s14159265，64 clip） | ✅ 64/64、0 靜音 0 削波；CLAP 0.197（= MF25 同 64 clip）、PQ 6.03（MF25 6.25，−0.22） |
| G1e | FAD 能算（參考目錄要明確傳 `/mnt/HDD/kojiek/musiccaps_reference`） | ✅ control s14159265：CFG0 FAD 3.814、CFG3+neg FAD 5.233（各 1924 pairs，約 6 分鐘，CPU）。推論期負向讓 FAD 變差 1.42，與 `project_negprompt_hurts_fad` 同向，所以 FAD 是 084 必報的反向讀數 |

### 4.1 G1c 結果

2026-09-27，Stage 1 模式（`fluxaudio_s`，from scratch，seed 1，Nhi 設定，slot0clean control 的 TSV／overlay），
不動 stage patch（當時 082 在跑 eval）：

- rank0 log 有 `MeanFlow CFG target guidance branch uses weights/negmf_084/fidelity8_t5.pth for t > 0.6667` ✅
- 40 iter 全部跑完，loss 0.995、grad_norm 4.3～7.4，無 NaN ✅
- 結束時 `synthesize_ema` 因為冒煙把 `ema.checkpoint_every` 設成 999999（沒有快照可合成）而報錯，
  這是冒煙設定造成的，正式 run 用 10000。冒煙目錄已刪。
- 限制：S2（`meanaudio_s`＋Stage 2 patch）的前向沒有在執行期冒煙，只由 G1a／G1b 的單元測試覆蓋；
  guide 分支的程式碼在 `loss()` 裡，與 stage 無關。

## 5. 端點（預註冊）

所有 Δ 都是**同 seed、同 clip 配對**（arm − control），bootstrap 95% CI（B=10,000），跨 seed 用 clip 平均的配對差。
PQ 端點一律用 **−30 LUFS 對齊後**（lvl30）的分數；CLAP 用原始音量（063：CLAP 越大聲越高，同時報 lvl30 版本）。

| 端點 | 定義 | 判準 |
|---|---|---|
| **E1（主）** | ΔPQ_lvl30：N@CFG0 − C@CFG0 | Stage A：≥ **+0.31**（2× 訓練 seed 底線 0.155）且 CI 下界 > 0。Stage B：三 seed 各自 > 0 且合併 ≥ +0.31。同時報 R = Δ / G_neg(C) |
| **E2（獨立讀數）** | ΔCLAP：N@CFG0 − C@CFG0 | 非劣性：CI 下界 > **−0.008**（2× CLAP 底線 0.004）。fidelity8 是依 PQ 選的，所以 CLAP 是唯一獨立讀數；若 ΔCLAP ≥ +0.008 寫成獨立支持 |
| E3（成本對等） | N@CFG0 − C@CFG3+neg（PQ lvl30、CLAP） | 描述性。一次前向能拿到兩次前向的多少 |
| E4（疊加） | N@CFG3+neg − C@CFG3+neg | 描述性；監看 crest_mean、clipped_n、LUFS（有效 scale 變大，可能飽和） |
| E5（1-NFE） | N − C，兩者都用 1 步 CFG0 | 描述性；MeanFlow 蒸餾唯一不靠推論期 guidance 的設定 |
| FAD | N、C 的 CFG0 與 CFG3+neg，2048 抽樣、ref `/mnt/HDD/kojiek/musiccaps_reference` | 描述性（單值、無 CI）；negprompt 線唯一的負帳是 FAD +0.046，guidance interval 預測 N100 比 Nhi 付更多 |
| 響度閘門 | ΔLUFS、Δcrest、silent_n、clipped_n | 任何格 silent_n > 2× control → 該 arm 判 **fail（靜音逃逸）**，不論 PQ |

參考數字（control，MF25，n=5521）：

| seed | CFG0 PQ / CLAP | CFG3+neg PQ / CLAP | lvl30 PQ CFG0 → CFG3+neg | G_neg（lvl30 PQ） | silent_n CFG0 / CFG3+neg |
|---|---|---|---|---|---|
| 14159265 | 6.430 / 0.1977 | 7.258 / 0.2243 | 6.648 → 7.476 | 0.828 | 52 / 67 |
| 16180339 | 6.488 / 0.1989 | 7.383 / 0.2322 | 6.729 → 7.583 | 0.854 | 39 / 49 |
| 27182818 | 6.508 / 0.1997 | 7.431 / 0.2302 | 6.724 → 7.553 | 0.828 | 39 / 115 |

所以 E1 的門檻 +0.31 ≈ G_neg 的 37%。

## 6. 階段與停機規則

| 階段 | 內容 | 進下一階段的條件 |
|---|---|---|
| **A**（pilot） | seed 14159265 × {N100, Nhi} | 至少一個 arm 過 E1＋E2 且沒有靜音逃逸 |
| **B**（複製） | 過 A 的 arm × seed {16180339, 27182818} | — |
| C（條件式） | 過 B 後才排：reversed 文字（"high quality recording, clean, professional, …"）放 guidance 分支 | 回答「任何非空文字都行」還是「負向文字才行」（08-31 消融：reversed 在推論期複製 51% 增益） |

停機（action 內自動）：

- S2 log 出現 `loss:[ ]*nan`，或 `grad_norm:nan` 比例 > 5%（正常約每 2000 iter 一次）→ 停、判 failed。
- 找不到 guide log 行 → 停（代表跑成 stock，數字無意義）。
- NVMe 剩餘 < 13 GB → 停（exit 3）。

判讀（Stage A 後）：

| 結果 | 寫法 |
|---|---|
| E1 過、E2 過 | 進 Stage B；只能寫「單 seed 正向訊號」 |
| E1 為正但未過 +0.31 | inconclusive；不複製，除非 Nhi 過而 N100 沒過（那是區間效應，值得複製） |
| E1 ≈ 0 或負 | 收線；寫成「NASA 的結論在 MeanFlow 目標上複現：負向效果需要推論期差分」（單 seed 限制要寫） |
| 靜音逃逸或發散 | 收線；寫成「u(n) 無直接訓練時 target 不穩定」 |

最壞浪費：每個 arm 約 3.5 小時（S2 2h16m ＋ eval）。

## 7. Eval 格（每個 arm × seed）

1. `mc_mf25_eval.sh <arm> <ema> --no_q` → CFG0、CFG3+neg
2. 兩格各跑 FAD（`eval_metrics.py --skip_clap --skip_aes --skip_level --fad --ref_dir /mnt/HDD/kojiek/musiccaps_reference`，輸出 `<cell>_fad/`）
3. `mc_nfe1_cfg0_eval.sh <arm> <ema> --no_q` → 1-NFE CFG0
4. 每格 `level_match_rescore.py`（−30 LUFS），之後刪音檔（metrics 與 per_clip 留下）

control 要補的格（action 會跳過已存在的）：兩格 FAD（s14159265 的已由 G1e 算好）、1-NFE CFG0 與其 lvl30。
**control 的 stock 音檔只在算完 FAD 後才刪**（lvl30 已存在，音檔可重生）。

## 8. 資源

- 每個 arm 的 S2 目錄：migrate 後 ckpt 2.4 GB ＋ ema_final／last 各 0.48 GB ＋ EMA 快照（thin 後留 4 個）≈ 5.5 GB；峰值約 9 GB。
- 每個 eval 格的音檔約 1.2 GB，評完即刪。action 要求 NVMe 剩 ≥ 13 GB 才開跑。
- HDD 目前 3.7 GB free，**無法歸檔**；run 目錄留在 NVMe。2026-09-27 NVMe 剩 86 GB（081–083 還在消耗）。
- **不可刪**：三個 control 的 `*_stage1_100000_ckpt_last.pth`（084 的起點，NVMe 上各 2.4 GB），直到 084 全部收線。

## 9. Queue 計畫

- 優先級 **P2**（探索、可恢復：S2 每 10k 存 ckpt，action 可從 `ckpt_last` 續跑）。
- 編號：084 = N100 s14159265、085 = Nhi s14159265（Stage A）。Stage B 的編號在 A 收線後才分配。
- queue 檔與 contract 在 `docs/experiments/harn/negmf_084/`，contract `status: awaiting_operator`、`launch_allowed: false`。
  **放行方式**（operator 決定後）：把 contract 的 `status` 改成 `authorized_p2_pending`、`launch_allowed` 與
  `launch_authorization.gpu_launch_allowed／valid` 改成 true，再把 `queue/084_*.sh`、`queue/085_*.sh` 複製進
  `~/gpu_queue/p2/pending/`。launcher 內容不含 status，所以 contract 裡綁的 launcher sha 不必重算
  （binding 路徑已經寫成 pending 位置）。誤排也安全：`harn_guest.py` 見到 `launch_allowed` 非 true 會寫 held。
- contract 綁了 `mean_flow.py`、`runner_meanflow.py`、action、wrapper、兩支 eval wrapper、eval_metrics、
  level_match_rescore、分析腳本與 guide 特徵的 sha。放行前若改過其中任何一支，要重算對應 sha。
- guide 特徵（`weights/negmf_084/`）被 `.gitignore` 的 `weights/*` 擋在 repo 外；遺失時用
  `build_negmf_guide_feats_084.py` 重生，sha 要與 contract 相同（T5／CLAP 前向是決定性的）。
- 分析：`scripts/analysis/negmf_084_analysis.py`（兩個 arm 都讀，缺格列出後跳過），輸出
  `docs/experiments/results/negmf_084_summary.json`。
- 為什麼不直接排：這份工作的目標是「設計」；081／083 已經在 p2 pending，GPU 沒有 idle；
  而且 084 跟 081–083 共用 NVMe 空間，排在它們後面最安全。

### 9.1 放行紀錄（2026-09-28）

- operator：「放行 084，排進 queue」→ 084（N100）與 085（Nhi）兩個 Stage A arm 一起放行。
- 偏離：staged contract 漏了 `resume.checkpoint_sha256`（`accept_guest` 對非空 resume checkpoint 必須驗 hash），
  放行時補上 control S1 `ckpt_last` 的 sha（`db3a6081…`，2,403,806,639 bytes），記在 contract `deviations`。
- 084 於 16:23 上位；dataloader 快轉約 14 分鐘（與 control 相同）；it 100000／100050 loss 0.99430／0.99358，
  control 同位置 0.99430／0.99357；guide log 行 `for t > 0.0` 在。

## 10. 可寫層級（預先劃好）

| 若結果成立可寫 | 高可信推論 | 不能這樣寫 |
|---|---|---|
| 在 slot0clean quarter 上，guidance 分支換成 fidelity8 讓 CFG0 的 lvl30 PQ 上升 X（R = …） | 負向 prompt 的效果可以部分收進 MeanFlow 權重 | 「不需要推論期負向 prompt 了」（除非 E3 ≈ 0 且 FAD 不變差） |
| 1-NFE CFG0 的 Δ | MeanFlow 的一步取樣可以帶負向 | 「負向蒸餾在所有 guidance 蒸餾框架都成立」（只測了 MeanFlow 目標） |
| `empty_string_t5.pth` 是常數向量 | null 分支從來不是任何文字的語意 | 「換掉 null 就是換掉空字串的語意」 |
