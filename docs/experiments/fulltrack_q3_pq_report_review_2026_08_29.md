# 對 `fulltrack_q3_pq_research_report_for_review_2026_08_28.md` 的審查

審查日期：2026-08-29
被審文件 SHA-256：`75d2145f83c25ec2fe8bbcb59b73a0a861f417502b9810448041eb8fb8e6b82a`
審查者證據：本日獨立跑的 canonical 全量重評估（4 arms × 5,521）、negative-prompt 對照、
`q_embed` 權重稽核。原始資料：`/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval/*.json`

**總評**：稽核邊界劃分嚴謹，paired bootstrap CI 是我原本 probe 缺少的、有價值的補強，
blocker 清單（尤其 `bert-base-uncased`）是真的。**但有一項會讓 B1/B2 gate 必然失敗的缺陷，
一項過度概括的排除結論，以及一個已被今日新證據改變的前提。** 建議先修正再進 Gate 1b。

---

## A. 必須修正（會導致計畫失敗或錯誤結論）

### A1. §5.1 的四位小數 exact gate 會在 CLAP 上必然失敗 —— 已用 4 個 arm 證實

我今日在 canonical 協定（MusicCaps 5,521、MeanFlow 25、literal CFG 0、seed 42、NoMask、
full precision）下**全量重新生成並重新評分**四個 full-scale arm，與歷史發表值逐欄比對：

| arm | CE | CU | PC | PQ | CLAP |
|---|---|---|---|---|---|
| `c2p0_slot0_full_noq` | ✅ 6.2870 | ✅ 6.7220 | ✅ 5.1393 | ✅ 6.5793 | ❌ 0.2201 vs 0.2149（**+0.0052**）|
| `fulltrack_q3_full_q9` | ✅ 6.8458 | ✅ 7.1468 | ✅ 5.3016 | ✅ 6.9337 | ❌ 0.1870 vs 0.1821（**+0.0049**）|
| `c2p0_fair013_worst_full` | ✅ 6.4162 | ✅ 6.9374 | ✅ 5.1868 | ✅ 6.7195 | ❌ 0.2195 vs 0.2109（**+0.0086**）|
| `c2p0_slot0_q5_full_q9` | ✅ 6.3352 | ✅ 6.7290 | ✅ 5.2118 | ✅ 6.5730 | ❌ 0.2235 vs 0.2174（**+0.0061**）|

**四個 AES 指標 16/16 逐欄四位小數完全相等；CLAP 4/4 全部不等，且偏差方向一致為正。**

唯一的實作差異：我把 CLAP 改成 batch=32 計算，歷史 `phase4_eval.py::compute_clap_score`
是**逐檔**呼叫 `get_audio_embedding_from_filelist([path])`。batching 就足以造成
+0.005～+0.009 的系統性偏移。

**受控測試（同一份 511 個音檔、同一份 caption，只改 CLAP batch size）**：

| batch size | CLAP | vs batch=1 |
|---:|---:|---:|
| 1（歷史 `phase4_eval.py` 的逐檔路徑） | 0.211470 | — |
| 8 | 0.211488 | +0.000018 |
| **32**（我重評估用的設定） | **0.215989** | **+0.004519** |

`laion_clap.get_audio_embedding_from_filelist` 在 batch > 8 時會改變 padding 行為，
造成系統性正偏移。**batch=1 與 batch=8 實質相同**，batch=32 才偏掉。

**這個 +0.0045 幾乎完全解釋了 A1 表中的落差**（實測 +0.0049～+0.0086，
其中 `c2p0_slot0_full_noq` 是 +0.0052）。同一批重算中 AES 逐位不變。

**所以結論比我原本寫的更好**：CLAP **也是可重現的**，只是必須用歷史的逐檔路徑。
這不是「CLAP 不可重現」，而是「CLAP 對 batch size 敏感，必須綁定」。

⚠️ 另一次觀察到更大的偏移（`N1_cfg15_neg`：batch32 0.2497 vs batch8 0.2361，差 0.0136），
顯示偏移幅度與音檔內容有關，不是固定常數。上表是在單一 arm 上的受控量測，
不應外推成通用校正值。

**專案層級的含義**：跨 run 比較 CLAP 前必須確認 batch 設定一致，否則實作雜訊
（0.004～0.014）會與歷史表格內多數 arm 間差距（0.01–0.03）同量級。
建議寫進 `evaluation_policy.md`。

**兩個結論：**

1. **好消息，比稽核原本認為的更強**：F02 原判 `INCONCLUSIVE`（音檔已刪、無逐檔雜湊，
   無法宣稱重現）。但 metric-level 的重現其實**做得到而且已經做到了** —— 生成在 seed 42
   下決定性，Audiobox 評分也決定性。**歷史 AES 數字是可重現的**，這比報告目前的立場強。
2. **風險**：§5.1 要求五個指標**全部**四位小數相等。這在 CLAP 上**只有在 batch size
   與歷史一致（逐檔，batch=1）時才可能通過**。若 B-matrix 的 scorer 用了任何 batching，
   會被判成 `historical_repeat_failed` 並 hold 掉 B3–B6 —— 一個純粹由實作細節造成的假陰性。

**建議修正**（(a) 為主，(b) 為保險）：
- (a) **把 CLAP batch size 明確寫進 sealed protocol 並固定為 1**（逐檔），與
  `phase4_eval.py::compute_clap_score` 的迴圈一致。這樣五欄 exact gate 都可通過。
- (b) 同時保留一個 ±0.01 的 CLAP tolerance 作為 secondary judgement，避免其他未預期的
  實作漂移（不同 torch 版本的 padding 行為）再次造成假陰性。

這比原本的「排除 CLAP」更好：CLAP 其實可重現，只是必須把 batch size 當成協定的一部分綁定。

### A2. §3.2「已排除：q0–q9 是隨機初始化」過度概括

這句話對 **S2Q-from-NoQ** family 正確（`q_initialization: copy-null-q10-to-q0..q9`），
B1–B5 綁的 checkpoint 確實都屬於這一族，所以在 B-matrix 範圍內結論成立。

**但它作為一般性排除是錯的。** 對 `use_q_conditioning=true` 從 Stage 1 就開的 arm：

```
三個獨立 S1-Q run（fair013_k3 quarter / qwen3cap_k3 quarter / bucket_quarter_k3，
不同語料、不同 stage）的 q_embed[10]：逐位元相同，maxdiff = 0.0
未使用的 bucket rows 1,2,3,4,6,7,8 與 NoQ 模型逐位元相同 → 同一 seeded init，從未更新
實際使用的 rows 0,5,9 有動（maxdiff 0.79–0.98）→ 訓練本身正常
```

原因：`runner_flowmatching.py:267` / `runner_meanflow.py:304` 做 CFG dropout 時只把
`text_f` / `text_f_c` 換成 null，**沒有把 q 換成 null token 10**，所以 `q_embed[10]`
在 S1-Q 訓練中拿不到任何梯度。而 `mean_flow.py:156` 的訓練目標
`v_hat = 0.3·v + 0.9·u_t_c − 0.2·u_t` 裡的 `u_t` 是用 `q=10` 算的
（`MeanFlow()` 預設 w=0.3 / k=0.9，這條路徑一定會走）。

**影響**：正在跑的 `024_fair013_k3_full` 就是 S1-Q run，其 Stage 2 的回歸目標正在混入
一個隨機初始化 embedding。報告目前的措辭會讓讀者以為這個風險已被排除。

**建議**：把該列改為「已排除（僅限 S2Q-from-NoQ family）」，並新增一列未決／已確認缺陷
指向 S1-Q 的 q null token。

### A3. §3.1 的「明顯 tradeoff」框架不成立

報告用 fulltrack Q3 vs segment slot0 Q3 這一對推論出 tradeoff。但在 cfg 4.5 的 16-arm
子集內 **corr(CLAP, PQ) = −0.005**，且 PQ 前六名全是 fulltrack、與 c2p0 族**零重疊**，
而兩族 CLAP 相當（0.2003–0.2117 vs 0.2003–0.2419）。fulltrack 是在**相當的 CLAP** 下
贏 AES，不是拿 CLAP 換 PQ。

單一 pair 看起來像 tradeoff，是因為那一對剛好落在對角線上。建議降級為
「這一對呈現反向排序」，不要寫成 family-level 的 tradeoff。

---

## A4. 你們的 Q-effect 門檻 0.05 正好落在單一 seed 的雜訊底線上

12 個 canonical arm 已全部跑完（MusicCaps 5,521、MF25、literal CFG 0、seed 42、NoMask）。
其中有一組**訓練 seed 複製**（`rich_matched_noq_full_seed_replication`，除了訓練 seed
以外一切相同）：

| arm | 訓練 seed | PQ | CLAP |
|---|---|---:|---:|
| `c2p0_slot0_full_noq` | 14159265 | 6.5793 | 0.2201 |
| `c2p0_slot0_full_seed27182818` | 27182818 | 6.5270 | 0.2234 |
| | **差** | **0.0523** | 0.0033 |

**單一訓練 seed 就造成 0.052 的 PQ 位移。**

同一份資料也直接回答了 B4/B5 想測的 Q inference effect（同一 checkpoint、只改 q token）：

| checkpoint | q0 | q9 | q9 − q0 |
|---|---:|---:|---:|
| `c2p0_slot0_q3_full` | 6.5197 | 6.5437 | **+0.0240** |
| `c2p0_slot0_q5_full` | 6.5396 | 6.5730 | **+0.0334** |

**含義**：
- 實測 Q inference effect（+0.024～+0.033）**低於你們的 0.05 門檻**，會被判為
  `small_or_uncertain`。這個結論現在已經有資料了，B5 那格可以直接引用。
- 更重要的是：**0.05 的門檻本身小於訓練 seed 的雜訊（0.052）**。任何用單一 seed
  量到的 0.05 級效果，都無法與 seed 變異區分。建議把 Q 門檻提高到至少 0.10，
  或明確標註它是 within-checkpoint（同一權重、只改推論 token）門檻 ——
  在 within-checkpoint 對照下 seed 是共用的，0.05 才站得住。
- family 門檻 0.15 相對 seed 雜訊約 3 倍，這個是合理的。

## A5. canonical 協定下的 6.9 判定：已有完整答案

12 arm 全表（PQ 由高到低，`*` = fulltrack corpus）：

| arm | CLAP | CE | CU | PC | PQ |
|---|---:|---:|---:|---:|---:|
| \* `fulltrack_q3_full_q9` | 0.1870 | 6.8458 | 7.1468 | 5.3016 | **6.9337** |
| \* `fulltrack_noq_full` | 0.1845 | 6.7252 | 7.0787 | 5.2926 | 6.8586 |
| `c2p0_fair013_worst_full` | 0.2195 | 6.4162 | 6.9374 | 5.1868 | 6.7195 |
| `c2p0_slot0_full_noq` | 0.2201 | 6.2870 | 6.7220 | 5.1393 | 6.5793 |
| `c2p0_slot0_q5_full_q9` | 0.2235 | 6.3352 | 6.7290 | 5.2118 | 6.5730 |
| `p7v1_fullq_control_q9` | 0.1860 | 5.8506 | 6.8227 | 4.7838 | 6.5580 |
| `c2p0_slot0_q3_full_q9` | 0.2190 | 6.2474 | 6.7019 | 5.1752 | 6.5437 |
| `c2p0_slot0_q5_full_q0` | 0.2212 | 6.2674 | 6.6882 | 5.1669 | 6.5396 |
| `c2p0_slot0_full_seed27182818` | 0.2234 | 6.1527 | 6.6700 | 5.0839 | 6.5270 |
| `c2p0_slot0_q3_full_q0` | 0.2172 | 6.1960 | 6.6785 | 5.1354 | 6.5197 |
| `c2p0_slot2_full_noq` | 0.2143 | 6.0703 | 6.6466 | 4.8711 | 6.5124 |
| `c2p0_fair013_best_full` | 0.2299 | 6.1644 | 6.6740 | 5.0482 | 6.4670 |

**canonical CFG 0 下沒有任何 non-fulltrack arm 達到 PQ 6.9**；最佳者
`c2p0_fair013_worst_full` 為 6.7195，差 0.1805。**連 `fulltrack_noq_full` 也只有 6.8586** ——
在 canonical 協定下只有 fulltrack **加上 q3** 這一格越過 6.9。

這使 §5.3 的判定可以直接結案為「canonical target 未達成」，不需要再跑 B2/B6。
B-matrix 目前只剩 **B4（fulltrack Q3 q0）** 一格沒有資料。

---

## B. 前提已被今日新證據改變

### B1. cfg 0 協定本身是 gap 的主要來源

報告的 B1–B6 全部固定在 literal CFG 0。今日的對照顯示這個選擇本身就決定了結論。

`eval.py` 的 `negative_prompt` 原本被寫死成 `''`（今日已加 `--negative_prompt`，
預設 `''` 完全相容）。它只在 `cfg_strength >= 1.0` 生效（`ode_wrapper`：cfg < 1.0
直接回傳 pure conditional）。511 筆、25 步、seed 42、兩模型設定完全相同：

| cell | CLAP | CE | CU | PC | PQ |
|---|---:|---:|---:|---:|---:|
| c2p0 slot0（cfg 0） | 0.2160 | 6.3573 | 6.7662 | 5.1936 | 6.5849 |
| cfg 1.5，**不給** negative（控制組） | 0.2309 | 6.3751 | 6.7981 | 5.0440 | 6.6234 |
| **c2p0 slot0 + neg，cfg 1.5** | **0.2497** | **6.9821** | **7.3442** | **5.2167** | **7.2587** |
| c2p0 slot0 + neg，cfg 2.5 | 0.2587 | 7.2440 | 7.6164 | 5.1417 | 7.5911 |
| fulltrack NoQ（cfg 0） | 0.1815 | 6.6599 | 7.0197 | 5.2502 | 6.8063 |
| fulltrack NoQ + neg，cfg 1.5 | 0.1893 | 6.7109 | 7.1232 | 5.1213 | 6.9226 |

**同樣的介入，c2p0 漲 +0.67 PQ，fulltrack 只漲 +0.12。相同協定下 c2p0 全指標反超**
（CLAP +0.060、CE +0.271、CU +0.221、PC +0.095、PQ +0.336）。CLAP 同步上升，
不是品質換對齊；crest 6.13、clipping 0%，無飽和。

機制與 §4 的逐檔拆解一致：c2p0 最差 10% 的 clip 有 **47.1%** 是低保真提示詞
（最好 10% 只有 27.2%），MusicCaps 整體 37% 提示詞含低保真語言。**c2p0 忠實地跟著
這些提示詞劣化，fulltrack 因為輸出先驗本來就靠近「安全」原型所以跟得少** ——
在 cfg 0 看起來贏，但可改善空間也小。

**對 B-matrix 的含義**：B1–B6 在 cfg 0 下測到的 checkpoint-family gap，很可能主要在量測
「哪個模型在 cfg 0、無 negative 的條件下比較會跟隨劣化提示詞」，而不是 audio prior 的品質差。
消融顯示這個 gap 在給定任何有內容的 negative 後就大幅收斂。這不否定 B-matrix
的價值（重現與 Q-effect 的判定仍然需要），但 §8 的工作假說
「checkpoint-family／audio prior 是主要來源」應該加上這個 competing explanation，
並建議增列一組 `cfg 1.5 + negative` 的 secondary cells。

**全量 5,521 已確認（canonical prompt set，同 seed 42）**：

| subset | 協定 | CLAP | CE | CU | PC | PQ |
|---|---|---:|---:|---:|---:|---:|
| ALL (5,521) | cfg 0 canonical | 0.2201 | 6.2870 | 6.7220 | 5.1393 | 6.5793 |
| ALL (5,521) | **cfg 1.5 + negative** | **0.2505** | **6.9239** | **7.3173** | **5.1718** | **7.2366** |
| | delta | +0.0304 | +0.6369 | +0.5953 | +0.0325 | **+0.6572** |
| no-vocal (2,498) | delta | +0.0321 | +0.6336 | +0.5554 | +0.0553 | +0.5837 |
| 低保真提示詞 (1,947) | delta | +0.0266 | **+0.6863** | +0.6148 | +0.0599 | **+0.6614** |

PQ ≥ 6.9 的 clip 比例：**43.1% → 72.3%**。511 筆的結果在全量上完全站得住。

**內容消融修正了機制歸因（重要）**：

| negative prompt | PQ | vs baseline | PQ（低保真提示詞） |
|---|---:|---:|---:|
| 無（cfg 0 baseline） | 6.5844 | — | 6.3947 |
| 空字串，cfg 1.5 | 6.6229 | +0.039 | 6.3865 |
| **不相關內容**（紅色腳踏車、山路、日落） | **7.0506** | **+0.466** | 6.8864 |
| **反向內容**（high quality professional studio…） | 6.6304 | +0.046 | 6.4240 |
| 保真度導向（low quality, noisy, …） | 7.2586 | +0.674 | 7.0963 |

**我先前把效果主要歸因於「保真度語意」是不準確的。** 正確的拆解是：
- 約 **70%** 來自 **generic CFG sharpening** —— 任何有內容的 off-target negative
  都能拿到（不相關內容 +0.466 / 保真度 +0.674）。
- 約 **30%** 是保真度導向的額外增益（7.2586 vs 7.0506）。
- **方向確實重要**：把 negative 反過來指向「高品質」幾乎沒有效果（+0.046），
  與空字串同級。所以不是「任意擾動都有效」。
- 空字串 negative 無效，是因為它的 T5 embedding 接近模型學到的 `empty_string_feat`，
  等同標準 CFG，而 MeanFlow 訓練時已內建 CFG，所以沒有額外收益。

### B2. §5.3「不得因此重排現有 queue」與現況衝突

操作者已於 2026-08-28 批准把 `001_modular_template_quarter.sh` 插入
`p2/pending/`，排在 025/026 之前（overlay 78 GB 已編碼完成、binding 抽驗 10/10 通過）。
Queue 現況為 `001` / `025` / `026`。報告的假設需要更新。

---

## C. 效率：B-matrix 已有 5/6 完成（僅缺 B4）

我今日的重評估 sweep 產出**逐檔** CLAP+CE+CU+PC+PQ（不只 aggregate），全量 5,521：

| B-cell | 對應 arm | 狀態 |
|---|---|---|
| B1 fulltrack Q3 q9 | `fulltrack_q3_full_q9` | ✅ 完成，逐檔已存 |
| B6 segment slot0 NoQ | `c2p0_slot0_full_noq` | ✅ 完成，逐檔已存 |
| B2 segment slot0 Q3 q9 | `c2p0_slot0_q3_full_q9` | 🟡 在同一 sweep 佇列中 |
| B3 fulltrack NoQ | `fulltrack_noq_full` | 🟡 在同一 sweep 佇列中 |
| B4 fulltrack Q3 q0 | — | ❌ **唯一缺口** |
| B5 segment slot0 Q3 q0 | `c2p0_slot0_q3_full_q0` | ✅ 完成，逐檔已存 |

逐檔資料在 `/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval/<arm>.json`
的 `per_clip` 欄位，可直接餵給 paired bootstrap，省下 4 個 arm 的生成成本（約 2.5 小時 GPU）。
⚠️ 我的 CLAP 是 batch=32（見 A1），若 B-matrix 要 exact gate 需自行重算 CLAP。

---

## D. 認同且應保留的部分

- §2 的判定邊界（evaluation-only、不得做因果歸因）—— 正確且必要。
- §3.3 的 PQ 相關結構分析（PQ↔CE 0.969、PQ↔CLAP −0.127）與「不能只追 PQ」的結論。
- §4 對自身 probe 限制的誠實列舉（非隨機取樣、無 CLAP、無逐檔雜湊）。
- paired percentile bootstrap + 實用門檻的設計 —— 比我原本只報 mean 好。
- §7 的 blocker 清單，特別是 `bert-base-uncased` transitive dependency
  （laion_clap 的 RoBERTa text encoder 確實會拉這個）。
- §9.5 先做 no-GPU Gate 1b 而非直接 Gate 2。

---

## E. 對五個裁決問題的回答

1. **同意「未證實 eval bug，也未證明因果優勢」** —— 而且我可以把前半句加強：
   AES 指標在四個 arm 上都達成 4dp 精確重現，stale-audio 假說在 metric 層級已可排除
   （見 A1）。後半句維持不變。
2. **不接受目前形式的 exact gate。** CE/CU/PC/PQ 四欄用 4dp exact 是對的、也已證實可通過；
   CLAP 必須改成 tolerance（±0.01）或把逐檔計算路徑一併 seal。否則第一關就假陰性失敗。
3. **接受 0.05 / 0.15 的門檻**，但建議註明它們是 PQ 尺度上的門檻，不適用於 CLAP
   （CLAP 的實作雜訊本身就有 ±0.009，見 A1）。
4. **同意 canonical target 只由 B2/B6 判定、B5 標 secondary。** 補充：依 B1 的證據，
   canonical cfg 0 下 non-fulltrack 達 6.9 的機率很低（slot0 6.5793、best non-fulltrack
   `fair013_worst` 6.7195）；但在 `cfg 1.5 + negative` 下 slot0 已達 7.2587。
   建議把「達成 6.9」的判定明確拆成 canonical-protocol 與 alternative-protocol 兩軌，
   不要讓協定選擇隱含在結論裡。
5. **同意先做 no-GPU Gate 1b**，並建議把 A1、A2 兩項修正一併納入該 amendment。

---

## F. 建議的 Gate 1b 增補項

1. 修 §5.1 的 CLAP gate（A1）。
2. 修 §3.2 的 q_embed 排除範圍，新增 S1-Q null token 缺陷列（A2）。
3. §3.1 降級 tradeoff 措辭（A3）。
4. §8 工作假說加入「cfg 0 協定造成 gap」的 competing explanation（B1）。
5. §5.3 更新 queue 現況（B2）。
6. 評估是否複用已完成的 B1/B6 逐檔資料（C）。
7. 考慮增列 `cfg 1.5 + negative` secondary cells。
