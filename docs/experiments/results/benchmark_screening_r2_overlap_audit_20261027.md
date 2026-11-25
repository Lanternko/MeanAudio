# r2 訓練重疊旗標查核（2026-10-27）

**對象**：`docs/experiments/benchmark_screening_r2_20261025.md` §5 的「未核實」格，以及 §1 未查核的兩個候選資料集（PAM400、MusicPrefs）。
**本檔只新增，不改任何既有檔案、不改預登錄旗標、不重算任何分數。** 下面所有「若變獨立會怎樣」都用 r2 結果檔裡已經印出的數字，沒有重跑。

## 0. 方法與判定規則

1. **一手證據**才計入：論文正文／表格（附 arXiv ID 與節號）、HF model card、repo 設定檔（附 commit 與行號）、實際下載的 manifest／metadata（附筆數）。
2. 判定只有三類：**獨立（有證據）**、**訓練過／重疊（有證據）**、**仍無法核實（寫查了什麼、缺什麼）**。
3. 沿用 round-1 定義：「獨立」＝**該模型的訓練沒有用這份資料的標籤**，不保證音訊或系統分布沒見過。時間先後、資料集「性質不同」都不算證據。
4. 標籤獨立與音訊獨立分開寫：一張表可能「標籤獨立、音訊未核實」。
5. r2 的 §5 與 §7.3 是**預登錄**：本檔只提出「有證據的旗標更正」，判定結果照 §7.3 重讀，**不改門檻、不改種子、不改判定規則**。

證據檔全部在 `~/tmp/r2_overlap_audit/`（下載物，未進 repo）；MEva repo 的 commit 是 `a356765b14323cdcf732a3af36dcdcc64a977e57`。

---

## 1. 更新後的重疊表（原 §5；`[改]` ＝本次有證據的更正）

| 指標 | B1 ODAQ | B2 AIME | B3 SongEval | B4 SingMOS-Pro | B5 MC522 | B6 Music Arena |
|---|---|---|---|---|---|---|
| AES | 獨立（標籤；音訊重疊未核實） | **獨立（生成軌，音訊亦獨立）；100 檔 MTG-Jamendo 實錄仍未核實** `[改]` | **獨立（標籤＋音訊；訓練不用合成資料）** `[改]` | **獨立（生成軌）；578 檔 GT 實唱仍未核實** `[改]` | 獨立（round-1：重疊未知→描述） | **獨立（標籤＋音訊；全為生成）** `[改]` |
| MEva | 獨立 | **重疊**（AIME 是訓練 benchmark） | **重疊**（SongEval 是訓練 benchmark） | 獨立 | 獨立（round-1） | **重疊**（pooled checkpoint 訓練含 Music Arena train split） `[改：未核實→重疊]` |
| MuQ-Eval A1 | 獨立 | **獨立（只訓練於 MusicEval）** `[改]` | **獨立（論文 0 次訓練提及；僅列為未測的 cross-dataset）** `[改]` | 獨立 | 獨立（round-1） | **獨立（只訓練於 MusicEval）** `[改]` |
| MuQ-Eval A3a | 獨立 | **獨立（同上）** `[改]` | **獨立（A3a＝LoRA 適應，不是 SongEval 維度）** `[改：重疊→獨立]` | 獨立 | 獨立 | **獨立（同上）** `[改]` |
| CMI-RM | 獨立 | **獨立（訓練＝CMI-Pref-Pseudo＋CMI-Pref train＋MusicEval train）** `[改]` | **獨立（明文未用，列為 future work）** `[改：重疊→獨立]` | 獨立 | 獨立（round-1） | **獨立（Music Arena 只當 held-out 評測）** `[改]` |
| PAM | 獨立（零樣本 CLAP） | 獨立 | 獨立 | 獨立 | 獨立 | 獨立 |
| SongEval | 獨立（域外） | 獨立（域外） | **重疊（訓練集＝SongEval 2,199 首；逐檔 split 未公布→交集數仍無法核實）** `[改：重疊／未核實→重疊]` | 獨立（域外） | 獨立（域外） | 獨立（域外） |
| SingMOS | 獨立（域外） | 獨立（域外） | 獨立（域外） | 重疊／test 獨立（predictor 訓練於 SingMOS-Pro） | 獨立（域外） | 獨立（域外） |
| MERT-ref／MuQ-ref | 獨立（無監督品質標籤） | — | — | — | — | — |
| MERT-cent／MuQ-cent | — | 獨立（只描述） | — | 獨立（只描述） | — | — |
| CLAP | — | 獨立（標籤） | — | — | 獨立（round-1） | 獨立（標籤） |
| SongBench／HEAR | 未核實 | 未核實 | 未核實／可能重疊 | 未核實 | 未核實 | 未核實 |

**逐格變更紀錄**

| # | 格 | 原旗標 | 新旗標 | 一句話依據 |
|---|---|---|---|---|
| 1 | MEva × B6 | 未核實 | **重疊** | r2 跑的 checkpoint 是 `pooled/small/f03_sae_only_cnn_clean_best.pth`，HF card 寫「pooled = trained on the union of the five benchmarks」，含 Music Arena |
| 2 | MuQ A1 × B2 | 未核實 | **獨立** | 論文 §IV-A：只訓練／評估於 MusicEval（5-fold CV）；AIME 全文 0 次提及 |
| 3 | MuQ A1 × B3 | 重疊（「A1 未核實」） | **獨立** | 同上；SongEval 只出現在 limitation「cross-dataset 未測」與 future work |
| 4 | MuQ A1 × B6 | 未核實 | **獨立** | 同上；Music Arena 全文 0 次提及 |
| 5 | MuQ A3a × B2 | 未核實 | **獨立** | A3a 與 A1 共用同一訓練集；A3a 是 LoRA 變體 |
| 6 | MuQ A3a × B3 | 重疊 | **獨立** | 預登錄的「A3a 含 SongEval 維度」是誤記：論文的 A3a＝LoRA encoder adaptation |
| 7 | MuQ A3a × B6 | 未核實 | **獨立** | 同 #5 |
| 8 | CMI-RM × B2 | 未核實 | **獨立** | §3.3.2 列出訓練集三個來源，不含 AIME；§3.1 明寫 AIME 等「future work 才可能當訓練集」 |
| 9 | CMI-RM × B3 | 重疊 | **獨立** | 同一句明文排除 SongEval；§4.2 的訓練集變體實驗也沒有 SongEval |
| 10 | CMI-RM × B6 | 未核實 | **獨立** | Music Arena 只出現在 CMI-RewardBench 的 held-out 評測（1,340 乾淨偏好對） |
| 11 | SongEval × B3 | 重疊／未核實 | **重疊**（不再「未核實」） | 論文 §4.1／§5.1：2,199 train／200 held-out，工具包模型訓練於 SongEval train；但沒公布逐檔 split |
| 12 | AES × B2/B3/B4/B6 | 音訊未核實 | **生成軌獨立；實錄仍未核實** | §4.2.2 明文「we did not use any synthetic data to train」；97k 訓練語料的資料集清單未公布 |

---

## 2. 逐格證據清單

### 2.1 MEva × B6 Music Arena（未核實 → 重疊）

| 證據 | 內容 |
|---|---|
| `scripts/eval/meva_runtime.py:14` | `RUNTIME = ROOT / 'runtime/meva_20261003'` |
| `scripts/eval/meva_runtime.py:58` | `checkpoint = RUNTIME / 'models/pooled/small/f03_sae_only_cnn_clean_best.pth'` ← **r2 實際載入的權重** |
| HF `Dev4IC` MEva model card（`hf_meva_readme.md`） | 「`pooled/{small,large}/` # trained on the union of the five benchmarks」；per-benchmark 目錄另有 `…/music_arena/` |
| MEva repo `config/data/data_splits.yaml`（commit `a356765`） | `split_seed: 42`、`split_strategy: "source_stratified"`、`dataset_scopes.large_scale_merged: ["all_5_datasets"]` |
| MEva repo `config/data/full_datasets.yaml` | `large_scale_merged: name: "all_5_datasets"` → `data/full_splits/all_5_datasets/{train,val,test}.csv` |
| MEva repo `scripts/data/gen_full_splits.py` | `make_splits(aime…)`、`make_splits(mp, "data/full_splits/musicpref")`、`make_splits(ma, "data/full_splits/music_arena")`、`make_splits(se, …)`；`train_frac=0.8, val_frac=0.1, seed=42` |
| 論文 arXiv **2609.37710** §4.1 | 五個 benchmark 含 Music Arena（6,080 clips、≈61 h）；「Splits are train/val/test with fixed seeds, stratified by generator system」 |

**結論**：r2 用 pooled checkpoint，訓練集是五個 benchmark 的 train split 聯集；Music Arena 的約 80% clips 在訓練裡。r2 評的是 2,000 場 battle（3,998 檔）全量、沒有排除 train split → **重疊**。
**交集數**：MEva 的 split CSV **沒有釋出**（repo 只有產生腳本），無法逐檔比對；MEva 用 6,080 clips、r2 用 3,998 檔、都來自同一份 HF Music Arena → 若 r2 抽樣與 split 獨立，期望重疊 ≈ 80%（約 3,200 檔）。**這是估計，不是實測**。
**與原本假設的差別**：r2 原本担心的是「MusicPref 是否含 Music Arena」；實際上不需要那條路徑就已是重疊。

### 2.2 MuQ-Eval A1／A3a × B2 AIME、B3 SongEval、B6 Arena（未核實／重疊 → 獨立）

| 證據 | 內容 |
|---|---|
| 論文 arXiv **2603.22677** §IV-A | 「We train and evaluate on MusicEval, consisting of 2,748 clips from 31 text-to-music systems… Under 5-fold CV, each fold uses ~1,540 clips for training and ~385 for testing.」 |
| 同上 §IV-B | 「5-fold cross-validation stratified by TTM model」 |
| 同上 §III（A2／A3a 定義） | 「A3a: LoRA encoder adaptation… LoRA adapters to all attention projections, rank r=16, α=32」← **A3a 不是 SongEval 維度** |
| 同上 §IV-E | 訓練細節（AdamW、bf16、RTX 4080、24 kHz、10 s）沒有其他資料集 |
| HF `zhudi2825/MuQ-Eval`、`zhudi2825/MuQ-Eval-A1` card | 「built by training lightweight prediction heads on frozen MuQ-310M representations **using the MusicEval dataset**」 |
| 全文關鍵詞計數（本檔 §7 指令） | AIME 0 次、Music Arena 0 次、MusicPref 0 次、SongEval 4 次（§II-E 資料集回顧 1 次、limitation「cross-dataset generalization to SongEval is untested」、future work「cross-dataset validation on SongEval」、參考文獻 1 次） |

**結論**：A1 與 A3a 的**標籤**只來自 MusicEval → AIME／SongEval／Arena 三格皆**獨立**（標籤）。
**保留限制**：MusicEval 有 31 個 TTM 系統，AIME 13 個、Arena 15 個；系統族明顯重疊（MusicGen、AudioLDM2、Riffusion、Stable Audio 等），但**MusicEval 的 31 系統清單本次沒有取得 → 系統重疊數未核實**。依 round-1 定義這不影響「獨立」旗標，但「未見過系統分布」不能寫。

### 2.3 CMI-RM × B2／B3／B6（未核實／重疊 → 獨立）

| 證據 | 內容 |
|---|---|
| 論文 arXiv **2603.00610v3** §3.3.2（Training Strategy） | 「The datasets involved include CMI-Pref-Pseudo for preference pre-training, together with the **training split of CMI-Pref and the training split of MusicEval** for expert fine-tuning.」 |
| 同上 §3.1（Table 1 前的段落） | 「We compare additional resources in Table 1, including **AIME, MusicPref, and SongEval**, but do not include them as main benchmark test sets… **These datasets lack official splits aligned with our protocol, and future work could explore them as additional training sets.**」 |
| 同上 §3.2（CMI-RewardBench） | 「It uses **only held-out data**: PAM provides 500 scalar ratings…」；Music Arena 1,340 clean preference pairs 亦為評測 |
| 全文關鍵詞計數 | AIME 4 次、MusicPref 4 次、SongEval 20 次、Music Arena 41 次——全部在比較表／評測／相關工作，訓練段落只列 CMI-Pref-Pseudo＋CMI-Pref train＋MusicEval train |

**結論**：AIME／SongEval／Music Arena 三格**獨立**（標籤）；MusicEval 與 CMI-Pref 仍是重疊（CMI-Pref 不是本輪 benchmark）。
**保留限制**：與 MuQ-Eval 相同，訓練用 MusicEval 的系統族與 AIME／Arena 重疊，系統分布未核實。

### 2.4 AES × B2／B3／B4／B6（「音訊未核實」→ 生成軌獨立）

| 證據 | 內容 |
|---|---|
| 論文 arXiv **2502.05139** §4.2.2 | 「The slightly worse performance of the proposed AES predictors compared to PAM especially in music might be caused by the unseen data issue that **we did not use any synthetic data to train the Audiobox-Aesthetics models**.」 |
| 同上 §2.2 | 訓練資料＝「a super-set of open-sourced and licensed datasets of 3 different audio modalities」，97k 樣本／≈500 h，**沒有列資料集名稱** |
| 同上 §4.2.2 | PAM-music 500 檔是**評測集**：另外雇用 10 位有音訊背景的標註者標 AES 四軸（GT-PQ/CE/CU/PC），不是訓練資料 |

**結論**：
- B6 Arena（15 系統全為生成）、B3 SongEval（2,399 首生成歌曲）、B4 SingMOS 生成軌、B2 AIME 生成軌 → **音訊獨立**（模型沒见过任何合成音訊）＋標籤獨立 → **獨立**。
- **仍未核實**：B2 的 100 檔 MTG-Jamendo 實錄（7.7%）、B4 的 578 檔 GT 實唱（7.2%）、B5 MC522、B1 ODAQ——97k 訓練語料的資料集清單未公布，無法排除這些真實錄音在內。
- 量化：AIME 92.3% 生成、SingMOS 92.8% 生成、Arena／SongEval 100% 生成。

### 2.5 SongEval predictor × B3（重疊／未核實 → 重疊）

| 證據 | 內容 |
|---|---|
| 論文 arXiv **2505.10793** §4.1 | 「We **randomly select 2,199 songs for training** and reserve 200 songs as the evaluation set. Additionally, we include 50 non-copyrighted real songs to evaluation set」 |
| 同上 §5.1 | 「we train the assessment models **on SongEval**」 |
| repo `scripts/eval/songeval_src/README.md` | 「This repository provides a **trained aesthetic evaluation toolkit based on SongEval**」；權重 `songeval_src/ckpt/model.safetensors`（100,792,276 B）即 r2 載入的模型（`metric_win_tie_126d.py:73 load_songeval`） |
| HF `ASLP-lab/SongEval` tree（API） | 只有 `metadata.jsonl`（1,267,359 B）＋`mp3/`；**metadata 沒有 split 欄**（r2 §13 #4 已記） |

**結論**：模型訓練於 SongEval train → **重疊**（旗標本來就是保守的重疊，這次把「未核實」拿掉）。
**交集數**：官方 2,199／2,399 訓練，r2 的 S800 分層抽樣從全部 2,399 抽 → 期望 ≈733 首在訓練集（91.7%）；**逐檔交集無法算**（split 未公布）。§7.3 另有預登錄「B3 不進任何閘門」，本檔不動那條規則。

### 2.6 SongBench／HEAR（維持未核實）

- r2 §14.2：J6 依 §9.3「可跳過」未執行；§13 #10／#12 明載未讀 LICENSE 與輸入維度。
- 本檔**沒有**再查這兩個 repo：它們不是本輪 benchmark，且判定的六個指標已全部由其他格觸發。
- 結論：**仍無法核實**，缺的是 SongBench／HEAR 官方訓練資料說明與權重授權；要補就是讀兩個 repo 的 model card ＋訓練腳本，不在本次範圍。

### 2.7 沒動的格（避免過度更正）

| 格 | 為什麼不動 |
|---|---|
| MEva × B1／B4／B5 | 五個訓練 benchmark 不含 ODAQ／SingMOS／MC522（論文 §4.1 列舉完整），維持獨立 |
| MuQ × B1／B4／B5 | 只訓練於 MusicEval，維持獨立 |
| CMI-RM × B1／B4／B5 | 訓練集三個來源，維持獨立 |
| PAM × 全部 | PAM 是零樣本 CLAP 指標，不是學出來的評分器；維持獨立（PAM400 那格例外，見 §3） |
| SingMOS × B4 | predictor 訓練於 SingMOS-Pro，test split 才有官方分隔，維持「test 獨立／train 重疊」 |
| MERT-ref／MuQ-ref | 無監督，無人類標籤可洩漏 |

---

## 3. PAM400 資料卡與訓練使用

### 3.1 資料卡

| 欄位 | 內容 | 證據 |
|---|---|---|
| 出處 | Deshmukh et al., *PAM: Prompting Audio-Language Models for Audio Quality Assessment*, arXiv **2402.00282**；資料 `human_eval.zip` 在 Zenodo record **10737388** | 論文 §D.2；`zenodo_pam.json` |
| 規模 | **500 檔＝4 個 TTM 系統 ×100 prompt（400 生成）＋100 真實 MusicCaps 曲目**；r2 用的 PAM400＝前 400 檔 | 論文 §D.2；`scores.csv` 實測 500 列（`audioldm2`／`musicgen_large`／`musicgen_melody`／`musicldm`／`real` 各 100） |
| 系統 | AudioLDM2-m、MusicLDM、MusicGen-l、MusicGen-mel（＋real） | 論文 §D.1／§D.2 |
| Prompt | 100 條 MusicCaps caption（隨機抽） | 論文 §D.2；`scores.csv` 實測 100 個唯一 Text |
| 評分者 | **每檔 10 位** Amazon Mechanical Turk、全部位於美國；共 8,000 個分數（OVL＋REL） | 論文 §D.2 |
| 資格／品管 | 排除「超過 5 檔給同分（std=0）」或「單次 <10 秒」的參與者；音檔順序完全隨機 | 論文 §D.2 |
| 題目 | OVL（Overall Quality）與 REL（Relevance to text）各 1–5 Likert | 論文 §D.2、Table 11／12 |
| 每檔分數 | 釋出的 `scores.csv` 是**每檔平均**（連續值，如 2.1／2.5；OVL 1.0–4.5、REL 1.2–4.9），不是逐位評分者 | `scores.csv` 實測 |
| 音訊 | 生成 5.0 s、16／32 kHz；real 9.985 s、22.05 kHz | 本地檔實測（r2 之前的 smoke） |
| 授權 | Zenodo **CC-BY-4.0** | `zenodo_pam.json`（`human_eval.zip` 1,201,120,327 B） |
| 與其他 benchmark 的交集 | PAM 的 100 個 real 檔 **100/100 都是 MusicCaps 曲目**（ID 對得上 `musiccaps_test.tsv` 去掉 `_30` 後綴）；prompt 與 AIME／Arena／MusicPrefs **各 0 交集**（精確與正規化後都是 0） | 本檔 §7 指令 |

### 3.2 哪些指標訓練或調參用過 PAM

| 指標 | 用過？ | 證據 | r3 可否算獨立 |
|---|---|---|---|
| AES | **沒有訓練用**；PAM-music 是它的評測集（另外找人標 AES 四軸） | 2502.05139 §4.2.2；且明文不用合成資料訓練 | 可（標籤＋音訊） |
| MEva | 沒有（全文 0 次提及 PAM） | 2609.37710 關鍵詞計數 0 | 可 |
| MuQ-Eval A1／A3a | 沒有訓練用（只在 related work 引用 2 次） | 2603.22677 關鍵詞計數；§IV-A 訓練集只有 MusicEval | 可 |
| CMI-RM | 沒有訓練用；PAM500 是 CMI-RewardBench 的 **held-out** 子集 | 2603.00610v3 §3.2 | 可 |
| PAM（指標本身） | **設計用資料**：PAM 論文的 prompt 選擇與報告都在這份資料上 | 2402.00282 §4.2／§5 | **不可**（描述性） |
| SongEval predictor | 沒有（0 次提及） | 2505.10793 關鍵詞計數 0 | 可 |
| SingMOS predictor | 沒有（0 次提及） | 2510.01812 關鍵詞計數 0 | 可 |
| MusicPrefs 指標 | 不適用（MusicPrefs 是資料集不是指標） | — | — |

**一句話可信度**：PAM400 符合 r2 §1 對 AIME／Arena 的要求（逐檔人類分數、評分者數與協定寫明、授權清楚、檔案可下載），但只有 **4 個系統 ×100 prompt、5 秒片段**，系統多樣性遠低於 AIME 的 13 系統與 Arena 的 15 系統——**當 G1 可以，當 G2（系統內）與系統排名的證據很弱**（4 個系統、每系統 100 檔）。

---

## 4. MusicPrefs 資料卡與重疊表

### 4.1 資料卡

| 欄位 | 內容 | 證據 |
|---|---|---|
| 出處 | *Aligning Text-to-Music Evaluation with Human Preferences*, arXiv **2503.16669**（ISMIR 2025）；HF `i-need-sleep/musicprefs` | 論文；HF card 的 `arxiv:2503.16669` tag |
| 規模 | 論文：**2,520 對**，21 個系統對 × 120 對；釋出 `human_preference.csv` 實測 **2,515 列**；HF `train` split **5,040 個 audio**（5,030 個檔名被引用，10 個多出） | 論文 §3；`musicprefs_human_preference.csv`；HF API `cardData` |
| Prompt | **120 條**唯一 prompt（每對共用一條；受試者看不到 prompt） | csv 實測；論文 §3「we choose to not display that prompt」 |
| 系統 | **7 個**：audioldm2、musicldm、stable_audio_open、musicgen_small／medium／large、riffusion_v1 | csv 實測 |
| 評分者 | **9 位**，來自 Turker Nation Slack；用 Thickstun et al. 的資格測驗，9/9 全對；每人貢獻 3–16% | 論文 Appendix D |
| 報酬 | 每對 US$0.40，中位 68 秒／對（≈US$21/h） | 論文 Appendix D |
| 題目 | **成對偏好**，兩個獨立軸：fidelity、musicality；可選「tie」 | 論文 §3、Appendix D；csv 欄位 `musicality`／`fidelity` |
| Ties | 有。論文：fidelity 25%、musicality 19%；csv 實測 **626/2,515 = 24.9%**、**466/2,515 = 18.5%**（一致） | 論文 §3；csv 實測 |
| 音訊 | 5,040 檔、**10.242 s**、44.1 kHz（HF 範例）／實測 16／32／44.1 kHz 依系統而異、單／雙聲道 | `mp_sample_0..2.wav` 實測 |
| 大小 | download 5,662,399,950 B（5.66 GB）、dataset 5,906,446,942 B | HF API |
| 授權 | **未宣告**：HF API `cardData` 無 license、repo 15 個檔案裡沒有 LICENSE | HF API；README |
| 下載方式 | `datasets.load_dataset('i-need-sleep/musicprefs')`（parquet，`data/train-*`） | HF README |
| 與 AIME／Arena 的交集 | 逐檔音訊：**沒有共同檔名空間**（AIME 是 parquet shard、Arena 是 `audio_files/<月>/prebaked-…mp3`），未做 bytes 比對；**prompt 精確交集 0／正規化交集 0**（vs AIME 100 條、vs Arena 1,375 條）；系統族重疊：與 AIME 共有 AudioLDM2／MusicGen L,M,S／Riffusion（3 族），與 Arena 共有 MusicGen M,S／Riffusion（2 族） | 本檔 §7 指令 |

### 4.2 重疊表（r3 若拿 MusicPrefs 當 benchmark）

| 指標 | 訓練／調參用過 MusicPrefs？ | 證據 | r3 判定 |
|---|---|---|---|
| MEva | **用過（訓練）**：MusicPref 是五個訓練 benchmark 之一，pooled 與 per-benchmark 都有 checkpoint | 2609.37710 §4.1；HF card pooled 說明；`f03_sae_only_cnn_clean_best.pth` | **重疊** |
| MuQ-Eval A1／A3a | 沒有（全文 0 次提及 MusicPref） | 2603.22677 關鍵詞計數 0 | 獨立 |
| CMI-RM | 沒有（明文列為 future work 的「可能訓練集」） | 2603.00610v3 §3.1 | 獨立 |
| AES | 沒有（論文早於 MusicPrefs、0 次提及；且不用合成資料訓練） | 2502.05139 §4.2.2、關鍵詞計數 0 | 獨立 |
| PAM | 沒有（零樣本 CLAP；PAM 論文 0 次提及） | 2402.00282 關鍵詞計數 0 | 獨立 |
| SongEval predictor | 沒有（0 次提及） | 2505.10793 關鍵詞計數 0 | 獨立 |
| SingMOS predictor | 沒有（0 次提及） | 2510.01812 關鍵詞計數 0 | 獨立 |
| CLAP | 零樣本 | — | 獨立（只描述） |

**一句話可信度**：MusicPrefs 的協定與評分者資訊寫得比 Arena 細（資格測驗、報酬、逐位貢獻比例），但**沒有授權宣告**、釋出量比論文少 5 對、且只有 7 個系統——比 Arena（15 系統、8,117 場）窄；可當**成對偏好**的補充證據，不適合當唯一的生成音樂證據。

---

## 5. 對 r2 判定的影響（不重算；只用結果檔已印出的數字）

### 5.1 若 AIME／Arena 的格子依本檔改成獨立

| 指標 | 格子 | 既有數字 | 會不會過 |
|---|---|---|---|
| MuQ A1 MI | AIME | G1 **0.540 [0.500, 0.580]**；G2 0.234；G3 0.641 | **G1 過**（點 ≥0.50、CI 下界 0.500 > 0.30）；**G2 不過**（0.234 < 0.50）；G3 CI 未印 → 不能宣稱過 |
| MuQ A3a MI | AIME | G1 **0.637 [0.604, 0.671]**；G2 0.270；G3 0.664 | **G1 過**；**G2 不過**；G3 CI 未印 |
| CMI-RM quality | AIME | G1 **0.586 [0.549, 0.622]**；G2 0.296；G3 0.667 | **G1 過**；**G2 不過**（0.296）；G3 CI 未印 |
| MuQ A1 MI | Arena | G1 **0.221 [0.186, 0.258]**；G2 −0.011；G3 0.605 | **G1 不過，且 CI 上界 <0.50 → 觸發排除** |
| MuQ A3a MI | Arena | G1 **0.220 [0.183, 0.256]**；G2 −0.016；G3 0.598 | 同上 |
| CMI-RM quality | Arena | G1 **0.216 [0.180, 0.253]**；G2 0.000；G3 **0.633 [0.615, 0.653]** | **G1 不過 → 觸發排除**；G3 過 |
| MEva | Arena | G1 0.252 [0.216, 0.287]；G2 −0.010；G3 0.636 [0.618, 0.654] | 旗標變**重疊** → 不進閘門（本來也未核實） |
| MuQ A1／A3a、CMI-RM | SongEval VFULL | G1 0.753／0.794／0.692（CI 下界都 >0.30） | 旗標變獨立，**但 §7.3 預登錄「B3 不進任何閘門」仍然成立** → 判定不變；若未來解除那條規則，三者 G1 都會過 |

### 5.2 判定結果

**六個排除判定全部不變。** 理由鏈更清楚：

1. **AIME 的高分是真的過 G1**（0.54–0.64），不是「未核實所以不算」；但三個指標在 AIME 的 **G2（系統內）全部不過（0.234–0.296）**，依 §7.3「適用的 G1／G2／G3 都成立」不可能成為候選。
2. **Arena 從「未核實」變成「獨立且失敗」**：MuQ／CMI-RM 的 G1 CI 上界 0.253–0.258 < 0.50，直接觸發 §7.3 的排除條件；CMI-RM 的 G3 反而過（0.633 [0.615, 0.653]）。
3. **MEva 的排除與 Arena 無關**：由 SingMOS G1 0.138 [0.103, 0.173] 觸發；Arena 變重疊不改變結論（只是少了一格描述）。
4. **AES 的排除不變**：AIME G1 0.403 [0.356, 0.450]、Arena G1 0.149 [0.112, 0.186] 都上界 <0.50；這兩格現在是「音訊也獨立」，排除更站得住。
5. **沒有任何指標升級成候選**；r2 的「本輪沒有候選」與「六個排除 + 一個未定」維持。
6. **唯一需要改寫的是敘事**：結果檔 §3 那句「MuQ／CMI-RM 很高的數字全部落在重疊／未核實格」在 AIME 上不再成立——AIME 的 0.54–0.64 是**合法的 G1 通過**；真正擋住它們的是 G2 與 Arena／SingMOS 的 G1。這反而讓結論更強：不是「證據不足」，是「證據顯示它們只在 AIME 過得了其中一個閘門」。

---

## 6. 仍未核實清單（本檔沒有解決的）

| # | 項目 | 缺什麼 | 影響 |
|---|---|---|---|
| 1 | MEva × Arena 的逐檔交集數 | MEva 的 split CSV 未釋出；要自己用 `fit_pairwise_manifests.py` + `gen_full_splits.py` 從 25 GB Music Arena 重建 | 只影響「重疊幾 %」的精確度；旗標已是重疊，不進閘門 |
| 2 | MusicEval 的 31 個系統清單 | 本機沒有 MusicEval metadata；需要下載 prompt_info | 「系統族重疊數」無法量化；不影響標籤獨立的判定 |
| 3 | AES 97k 訓練語料的資料集清單 | 論文未公布（只說 open-source + licensed） | AIME 100 檔 MTG-Jamendo、SingMOS 578 檔 GT、MC522、ODAQ 的真實音訊重疊**永遠無法核實**，只能維持「未核實」 |
| 4 | SongEval 官方 2,199／200 逐檔 split | metadata 無 split 欄、repo 無 split 檔 | 只能報期望重疊 ≈91.7%，不能報實際交集 |
| 5 | SongBench／HEAR | J6 未執行；未讀 model card／訓練資料 | 維持未核實（不在判定路徑上） |
| 6 | MuQ-Eval 釋出的 A1 checkpoint 是哪一 fold | model card 只說 5-fold CV，沒說釋出的是 fold 0 或全量重訓 | 只影響「MusicEval 重疊比例」，不影響 AIME／Arena／SongEval 的獨立性 |
| 7 | 各論文對 PAM／MusicPrefs 的自動化計數 | 用 HTML 全文關鍵詞計數，**沒有逐句人工複核**（表格與圖說可能漏） | 若某篇在表格裡用了 PAM 當訓練資料而未在正文提及，本檔會漏；目前沒有這種跡象 |

---

## 7. 可重跑的查核指令（全部在本機、無 GPU）

```bash
# 論文全文抽關鍵詞（HTML 已下載在 ~/tmp/r2_overlap_audit/）
cd ~/tmp/r2_overlap_audit
python -I ~/tmp/r2_overlap_audit_scripts/extract.py html_2603.22677.html "MusicEval|A3a|SongEval"
python -I ~/tmp/r2_overlap_audit_scripts/extract.py html_2603.00610v3.html "training split|future work"
python -I ~/tmp/r2_overlap_audit_scripts/extract.py html_2502.05139.html "synthetic data|97k"
python -I ~/tmp/r2_overlap_audit_scripts/extract.py html_2505.10793.html "2,199|trained on the SongEval"
python -I ~/tmp/r2_overlap_audit_scripts/extract.py html_2402.00282.html "10 different participants|500 audio samples"

# 關鍵詞計數（§2 的 0 次／4 次）
python -I -c "...各論文 PAM/MusicPref/AIME/SongEval 出現次數..."

# MEva 的 split 與 checkpoint
curl -sS https://raw.githubusercontent.com/YoEv/MEva/a356765b14323cdcf732a3af36dcdcc64a977e57/config/data/data_splits.yaml
curl -sS https://raw.githubusercontent.com/YoEv/MEva/a356765b14323cdcf732a3af36dcdcc64a977e57/config/data/full_datasets.yaml
sed -n '14p;58p' ~/MeanAudio/scripts/eval/meva_runtime.py

# MusicPrefs 實測
python -I -c "import csv;rows=list(csv.DictReader(open('musicprefs_human_preference.csv')));print(len(rows))"
curl -sS https://huggingface.co/api/datasets/i-need-sleep/musicprefs

# prompt 交集（AIME／Arena／PAM／MusicPrefs）
python -I -c "...正規化後比對，四個資料集兩兩交集皆 0..."
```

---

## 附錄：本檔與 r2 預登錄的關係

- 本檔**只更正重疊旗標**（§5 是資料查核欄位，不是門檻）；§7 的門檻、§7.3 的判定規則、§6 的統計量、§3 的輸入協定**一行未動**。
- r2 結果檔的數字**全部沿用**，沒有重算、沒有補跑；本檔 §5 的「會不會過」是把既有數字套回 §7.1／§7.3 的字面門檻。
- 「B3 SongEval 不進任何閘門」是預登錄規則，不是重疊旗標的推論；即使 MuQ／CMI-RM × SongEval 改成獨立，那條規則仍然有效，直到下一次預登錄重新寫定。
