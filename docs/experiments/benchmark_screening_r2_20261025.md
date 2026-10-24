# 指標 × 人類評分 benchmark 海選 round 2（2026-10-25 預登錄）

**狀態：預登錄草案。本檔在任何 round-2 評分之前 commit；實作是另一個任務。**
本檔只寫設計、資料查核、協定、統計、門檻、GPU 預算與下載清單；**沒有**跑任何 GPU、沒有對任何新 benchmark 評分。已下載的只有小型 metadata（見 §10.4）。

**與 round 1 的關係**：round 1（`newmodels_screening_20261020.md`／`results/newmodels_screening_20261020_results.md`）在 MC522／PAM400／ME 三個 benchmark 上沒有任何指標通過。本輪**不重跑那三個 benchmark、不改寫 round-1 判定**，只補齊三件 round 1 缺的東西：

1. **有人類評分的訊號劣化資料**（ODAQ MUSHRA）——我們 126a／126c 的劣化全是自己造的、**沒有人類 ground truth**（`docs/meetings/2026-10-22_metric_perturbation_review.md` §0、§5）。
2. **歌唱品質**（SingMOS-Pro）與**歌曲層美學**（SongEval）——教授 2026-10-20 會議第 5 項明列歌唱評估。
3. **大規模成對人類偏好**（AIME、Music Arena）——round 1 只有 PAM400（4 系統）與 ME，且沒有 TIE 標籤；Music Arena 是唯一有 TIE／BOTH_BAD 的公開資料。

本輪就算有新指標通過，也只能標成「round-2 候選」，不能回頭說 round 1 的「未定／排除」判錯；要改寫必須另立預登錄。

---

## 0. 一段話總結

用 5 個已查核可下載、有人類評分、且涵蓋不同域的 benchmark（ODAQ 訊號劣化 MUSHRA、SongEval 歌曲 5 維、SingMOS-Pro 歌唱 MOS、AIME 生成音樂成對偏好、Music Arena 真實使用者成對偏好），加上既有的 MC522 做段長控制，把 round-1 的指標清單（AES 4 軸、MEva、MuQ-Eval A1/A3a、CMI-RM、PAM）與新指標（SongEval 5 維、SingMOS predictor、MERT/MuQ reference-based 距離、CLAP、SongBench/HEAR 探索）一次測完。每個資料集都跑**段長變體**（5 s／10 s／30 s／full chunked mean）與**響度變體**（raw／−23 LUFS），直接回答教授「5 s vs 10 s」與我們「AES 響度敏感」兩個問題。統計沿用 round-1 的 G1/G2/G3，另加 ODAQ 專屬的 within-trial 閘門 G6 與段長／響度讀數 G4/G5。GPU 名目 19.3 h、加 15% I/O 與啟動約 22.2 h（上限 24 h；跳過探索 J6 約 21.8 h），切成 8 個 p2 job（最大約 4.9 h，J3 若超過 4 h 由實作再拆兩半），全部可逐 clip resume。

**五個要回答的問題**

| # | 問題 | 主要資料 | 對應會議 |
|---|---|---|---|
| Q1 | 在「有人類評分的訊號劣化」上，哪些指標的排序與人類 MUSHRA 一致？ | ODAQ | 2026-10-22 §5 第 1–3 項 |
| Q2 | 新候選（CMI-RM、MuQ-Eval、SongEval、SingMOS、MERT/MuQ）在獨立人類評分上過得了 round-1 門檻嗎？ | 全部 | 2026-10-20 第 5 項 |
| Q3 | 輸入從 10 s 換成 5 s／30 s／full，與人類的相關會掉多少？ | SongEval S800、ODAQ、MC522 | 2026-10-20 第 8 項與後續探索 |
| Q4 | 把輸入正規化到 −23 LUFS，與人類的相關會變好還是變差？ | 全部（成本高者用子集） | 051／063／065、128 |
| Q5 | 歌唱與歌曲域的指標（SingMOS、SongEval）在短純器樂／生成歌曲上還成立嗎？ | SingMOS-Pro、SongEval | 2026-10-20 第 5–6 項 |

---

## 1. 候選資料集查核（2026-10-25）

查核方式：HF／Zenodo／GitHub API 與網頁、實際抓取小型 metadata 檢查欄位與筆數（§10.4）。「已核實」＝本次有抓到檔案或 API 回傳；「未核實」＝只讀到文件描述，沒有實際檔案證據。

| 代號 | 資料集 | 內容與 n | 人類評分型態 | 音訊格式／大小 | 授權 | 下載 | 查核狀態 | 採用 |
|---|---|---|---|---|---|---|---|---|
| **ODAQ** | ODAQ 2024-icassp（Fraunhofer IIS／Netflix，ICASSP 2024） | **30 trials**；每 trial＝1 source＋5 processed＋hidden ref＋2 anchors＝8 檔，合計 **240 檔**；來源 25 段（14 music、11 film-style）；平均 12.5 s | **MUSHRA 0–100**，26 位專家／檔，6,240 分 | stereo 44.1／48 kHz；`ODAQ.zip` = **1,048,071,792 B**（Zenodo API 實查，md5 `9b4c1eba…`） | GitHub README 稱全部為 Creative Commons；逐檔條款在 zip 內 `_license_disclaimer.txt`／`_detailed_license.csv`（**細項未核實**） | [Zenodo 10.5281/zenodo.10405774](https://zenodo.org/records/10405774)（HTTP 200 已驗） | 已核實下載點、大小、md5；論文方法已核實 | **B1** |
| **SongEval** | SongEval（ASLP-lab，arXiv 2505.10793） | 2,399 首完整歌曲（metadata 實測 2,399 列），5 維美學 | 5 點 Likert（含 0.5）× 5 維；**metadata 實測每首 4 位標註者**（論文稱 16 位評分者，輪換方式未核實） | `mp3/N.mp3`；HF 回報 **11.1 GB**；取樣率未標（實測檔需 ffprobe，未核實） | 資料集 **CC BY-NC-SA 4.0**；code Apache-2.0 | [HF ASLP-lab/SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval) | 已核實 metadata 2,399 列與 5 維欄位 | **B3（S800 子集）** |
| **SingMOS-Pro** | SingMOS-Pro（ICASSP 2026，arXiv 2510.01812） | 7,981 clips／11.15 h／141 systems（SVS 3,425、SVC 1,307、SVR 2,671、GT 578） | **per-clip MOS**；5／8／10／20 位評審不等（實測 5 位者 7,353 檔）；另有 lyrics／melody 子分 | WAV；**1.48 GB**；16 kHz（6,937）／24 kHz（631）／44.1 kHz（413） | **CC BY 4.0**（HF API 實查） | [HF TangRain/SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro) | 已核實 metadata 7,981 列、split train 4,880／test 3,101 | **B4** |
| **AIME** | AIME（NeurIPS 2025；12 TTM 系統＋Jamendo real） | **1,300 個唯一 10 s snippet**（實測）；15,600 成對票（Music Quality 7,800、Text-Audio Alignment 7,800）；13 個 systems | **成對偏好**（answer 1／2；無 TIE）；同一 snippet 被不同對手重複評 | 音訊在 210 個 parquet shard；**62.3 GB**；survey metadata 295,100 B；所有 snippet 實測 10.0 s | 生成軌 **CC-BY-4.0**；MTG-Jamendo 逐軌授權；prompt tags CC BY-NC-SA | [HF disco-eth/AIME-survey](https://huggingface.co/datasets/disco-eth/AIME-survey)、[disco-eth/AIME](https://huggingface.co/datasets/disco-eth/AIME) | 已核實 survey 1,300 snippet／13 systems；音訊 shard 未下載 | **B2** |
| **Music Arena** | Music Arena（NeurIPS 2025 Creative AI，arXiv 2507.20900） | **8,117 battles**（12 個月度 config 的 battle 數加總＝JSON 檔數）、**13,090 個 mp3**；main tree 實測 **25.18 GB**（HF `usedStorage` 35.69 GB 含版本歷史） | **成對偏好 A／B／TIE／BOTH_BAD**（實測欄位）；使用者自選 prompt；無 per-clip MOS | MP3；32／44.1／48 kHz；長度變動大（實測 30.0 s 與 199.9 s 各一例）；README 明載 `sao`／`sao-small` 的 `audio_a`／`audio_b` 為 null | **CC BY 4.0**（HF API 實查） | [HF music-arena/music-arena-dataset](https://huggingface.co/datasets/music-arena/music-arena-dataset) | 已核實欄位、檔案數、storage、battle 數；偏好分布未核實 | **B6（2,000 battles 子集）** |
| CMI-Pref | CMI-Pref（HaiwenXia） | 4,027 票（文件稱 3,527 train／500 test；頁面未顯示列數） | 成對偏好（musicality／alignment） | JSON＋音訊；大小未核實 | CC BY-NC-SA 4.0 | [HF HaiwenXia/cmi-pref](https://huggingface.co/datasets/HaiwenXia/cmi-pref) | 只讀到卡片；**列數與音訊未核實** | 不採用：CMI-RM 訓練已含 CMI-Pref 人類標註 → 對主候選重疊 |
| MusicPrefs | MusicPrefs（ISMIR 2025） | 未核實 | 成對偏好 | 未核實 | 未核實 | [HF i-need-sleep/musicprefs](https://huggingface.co/datasets/i-need-sleep/musicprefs) | **未核實**（只確認 repo 存在） | 不採用：AIME＋Music Arena 已覆蓋成對偏好，下載與授權成本未核實 |
| SVCC／VoiceMOS singing | Singing Voice Conversion Challenge 歷屆 | 未核實 | MOS | 未核實 | 未核實 | — | **未核實** | 不採用：SingMOS-Pro 已涵蓋歌唱 MOS 且有 per-clip 分數 |

**SongBench／HEAR 不是 benchmark，是候選指標**（見 §4）。SongBench repo 的 `ckpt/songbench.safetensors` = 100,808,636 B、repo license 為 `NOASSERTION`（自訂／未宣告，未核實可否再散布）；HEAR 的 `earlab/EAR_HEAR` 有 `track_1.pth`／`track_2.pth`、license Apache-2.0（HF API 實查）。兩者只放進探索性 job J6，不進閘門。

**沒採用的理由（一句話）**：CMI-Pref 與 CMI-RM 訓練重疊；MusicPrefs 授權／大小未核實；SVCC 未核實且與 SingMOS-Pro 重複；SongBench dataset 不是獨立人類評分 benchmark。

---

## 2. 選定 benchmark 與角色

| 代號 | 資料集 | 角色 | 為什麼是它 | 主要人類目標（primary） | 次要（descriptive） |
|---|---|---|---|---|---|
| **B1** | ODAQ（全部 240 檔，分析時分 music／DE） | **訊號劣化的人類錨** | 唯一有逐條件 MUSHRA 的公開音樂／影視品質資料；涵蓋 codec artifacts（LP／TM／UN／SH／PE）與 source separation（DE），正好對上 126a／126c | **MUSHRA BAQ**（within-trial） | hidden ref 排名、anchor 校準 |
| **B2** | AIME（1,300 snippets） | 生成音樂成對偏好 | 13 systems、1,300 個 10 s snippet、7,800 票 Music Quality；比 PAM400 的 4 系統廣 | **Music Quality 成對偏好** | Text-Audio Alignment（CLAP／CMI-RM caption） |
| **B3** | SongEval（2,399 首；S800 子集做變體） | 歌曲層美學 | 2,399 首生成歌曲、5 維、4 位標註／首；唯一有 Memorability／Structure 維度 | **Overall Musicality** | Coherence、Clarity；Memorability／Naturalness 只描述 |
| **B4** | SingMOS-Pro（7,981 clips；test 3,101） | 歌唱品質 | 141 systems、per-clip MOS、官方 train/test；教授明列歌唱 | **per-clip MOS**（G1 用 test split） | 全部 clips 描述；center-5 s 截斷 |
| **B5** | MC522（既有，522 檔 10 s） | **段長控制（重用）** | 已經有人類 PQ、10 s 檔；不用下載 | 人類 PQ | — |
| **B6** | Music Arena（2,000 battles 子集） | 真實使用者偏好＋TIE | 唯一有 TIE／BOTH_BAD 的公開資料；多系統、真實 prompt | 成對偏好（排除 TIE／BOTH_BAD） | TIE 校準、BOTH_BAD、系統排名 |

**B5 不下載、不重跑 round-1 的 MC522 全量指標**：只用來回答段長（5 s vs 10 s）與響度（raw vs −23 LUFS）；這兩格是新協定，不是重跑 round 1。

**子集與抽樣（寫死）**

- **SongEval S800**：以 `gender`（male 1,489／female 902／mix 8）分層，按比例抽 800 首；`random_state=20261025`；排序鍵 `file_name`。S800 是**所有變體共用**的母體（full／30 s／10 s／5 s／norm），确保 paired bootstrap 的 n 一致。
- **SongEval 全量 center-10 s**：2,399 首全跑，只為 G1 的統計功率；不參與段長比較。
- **Music Arena 2,000 battles**：先下載全部 `battle_data/*/*.json`（8,117 個、25 MB），**先排除 `audio_a`／`audio_b` 為 null 的 battle**（README 明載 `sao`／`sao-small` 無音訊），再依 `preference` 分層（A／B／TIE／BOTH_BAD）與月份分層抽 2,000 場；`random_state=20261025`。若可用 battle < 2,000 則全用並報實際 n；若 TIE 總數 < 150，則全取 TIE 再補足。音訊以去重後的 `audio_files/...` 為評分單位（估計 ~3,200 檔）。
- **AIME**：1,300 snippets 全用，不抽樣。
- **SingMOS-Pro**：7,981 全用；G1 與 SingMOS predictor 只在官方 **test 3,101** 上算（predictor 訓練資料是 SingMOS-Pro，見 §5）。
- **ODAQ**：240 全用；music 子集（LP／TM／UN／SH／PE）與 DE 子集分開報。

---

## 3. 固定輸入協定（寫死，實作不得改）

### 3.1 讀檔與前處理

沿用 round-1 §1 的共用規則，只加本輪明定的部分：

1. 讀檔不改取樣率；多聲道取**通道平均**成單聲道；**默认不做任何響度／峰值正規化**。
2. 各模型內部重採樣到自己的原生 SR（AES 16 kHz、MuQ／CMI-RM 24 kHz、PAM 44.1 kHz、CLAP 48 kHz、MERT 24 kHz、SingMOS 16 kHz；MEva／SongBench／HEAR 依 pinned runtime，**未核實**）。
3. 不做 VAD、不做靜音裁切、不補零（除模型明確要求固定長度；見 §4 的 SongEval／SingMOS 例外）。
4. 同一 (dataset, item_id, variant) 的音訊 bytes 只解碼一次，之後所有模型共用；解碼後寫 16 kHz／模型 SR 的中間快取（FLAC 或 float WAV），快取 key 含原檔 sha256 前 16 碼、variant、SR。

### 3.2 段長變體（V）

| 代號 | 定義 | 適用 | 說明 |
|---|---|---|---|
| `native` | 原始檔全長 | ODAQ、AIME、SingMOS、Music Arena | 人類評分就是在這個長度上聽的 |
| `V5` | **中央** 5 s（`[max(0,(L−5)/2), +5]`） | ODAQ、SongEval S800、MC522、SingMOS test | 若 L < 5 s，用全長不補零 |
| `V10` | 中央 10 s | ODAQ、SongEval S800、MC522 | 若 L < 10 s，用全長不補零 |
| `V30` | 中央 30 s | SongEval S800 | |
| `VFULL` | 全長，切成**不重疊** 30 s 窗，最後一窗 < 5 s 丟棄；分數取算術平均 | SongEval S800（210 s 平均 → 約 7 窗） | 同時報 median 與 first-window（描述） |

段長問題的乾淨比較只用**人類評分長度 > 變體長度**的資料：SongEval S800（人類聽整首）、ODAQ（12.5 s）、MC522（10 s）。AIME（10 s）與 SingMOS（平均 5 s）的 V5 是**截斷 counterfactual**，只描述，不進 G4。

### 3.3 響度變體（L）

- `raw`：原檔。
- `norm`：以 **pyloudnorm、integrated LUFS、目標 −23.0 LUFS** 做純量增益；**不做 limiting**；峰值超過 0 dBFS 的樣本照實保留並記錄 `over_fs` 比例（沿用 128 §C1 的處理，只把目標從 −30 改 −23）。
- 近靜音（pyloudnorm 回 −∞）→ 該格 `invalid`，不默默替換；paired Δ 時整對排除並報數量。
- 覆蓋：B1 全、B2 全、B3 VFULL 與 V10（S800）、B4 全、B5 V10、B6 raw 全＋norm 子集（800 檔，隨機抽、seed 20261025）。

### 3.4 文字協定

- **CMI-RM caption**：B2 用 `description`（AIME 的 3 個 tag）、B6 用 `prompt`、B5 用 `musiccaps_test.tsv` 的 caption；其餘資料集沒有文字 → 只跑 `empty`。
- **CLAP**：B2（對 `description`）、B5（對 caption）、B6（對 `prompt`）；B1／B3／B4 沒有可比文字 → 不跑。
- `alignment` 在 `empty` 下不是文字對齊量測（round-1 §1 的既有注意事項）。

### 3.5 模型專屬固定處理

| 模型 | 本輪固定處理 |
|---|---|
| AES | 原生 SR、16 kHz 內部；一次 forward 出 PQ／CE／CU／PC |
| MEva | 沿用 round-1 的 pinned runtime；輸入窗長度依 runtime（**未核實**；若只吃 10 s，VFULL 逐窗呼叫再平均） |
| MuQ-Eval A1／A3a | 24 kHz、**中央裁 10 s**、不足補零（round-1 寫死） |
| CMI-RM | 官方 `RewardModelInference(mode='final')`、24 kHz、30 s 分塊、fp32 |
| PAM | 44.1 kHz、<7 s 重複補到 7 s、≥7 s 切 7 s 塊＋尾塊、塊平均 |
| SongEval | 沿用 126d 的呼叫方式；輸入固定 30 s（不足補零、超過裁中心 30 s）；VFULL 逐窗呼叫再平均 |
| SingMOS predictor | `torch.hub` `South-Twilight/SingMOS:v1.1.2`、`singmos_pro`、16 kHz、全 clip |
| MERT-ref／MuQ-ref | 24 kHz、全長（或 30 s 窗平均）；分數＝`−cosim(x, hidden_ref)`（愈高愈好） |
| MERT-cent／MuQ-cent | 24 kHz；分數＝`cosim(x, μ_sys^{−i})`（leave-one-out system centroid），只描述 |
| SongBench／HEAR | 依官方 repo；輸入 30 s 窗平均；**探索性** |

---

## 4. 指標清單、輸出與授權

| 指標 | 類別 | 輸出 | 方向 | 授權／限制 | round-1 狀態 |
|---|---|---|---|---|---|
| **AES**（Meta Audiobox Aesthetics） | no-reference 美學 | PQ／CE／CU／PC | PQ／CE／CU ↑；PC 只描述 | 權重 **CC-BY-4.0**（HF 實查） | 未定（PQ）、排除（PC） |
| **MEva** | no-reference 品質 | 1 純量 | ↑ | 沿用 pinned runtime；模型授權依原 repo（未重查） | 排除 |
| **MuQ-Eval A1** | no-reference 品質 | MI／TA | ↑ | code MIT；MuQ 編碼器 **CC-BY-NC-4.0** | 未定 |
| **MuQ-Eval A3a** | no-reference 品質（ordinal） | MI／TA | ↑ | 同上 | 未定 |
| **CMI-RM** | 文字／參考條件品質 | quality／alignment | ↑ | 權重 **CC-BY-NC-4.0**（非商用） | 未定（最接近） |
| **PAM** | CLAP 對比 prompt | P(clear/clean) | ↑ | code MS-PL；MS-CLAP 2023 | 未定 |
| **SongEval** | 歌曲美學 5 維 | se_coh／se_mem／se_nat／se_cla／se_mus | ↑ | code Apache-2.0；模型權重依 repo（未重查） | 126d 已測（未過） |
| **SingMOS predictor** | 歌唱 MOS | 1 純量 | ↑ | repo MIT；**權重 payload 授權未核實** | 未測 |
| **MERT-ref／MuQ-ref** | **有參考**嵌入距離 | −cosim | ↑ | MERT MIT／MuQ CC-BY-NC-4.0 | 133 用過（無參考版） |
| **MERT-cent／MuQ-cent** | 無參考典型性 | cosim | ↑（只描述） | 同上 | 133 用過 |
| **CLAP** | 音文對齊 | cosine | ↑ | laion_clap（依 repo） | round 1 未含 |
| **SongBench** | 歌曲 7 維 | 7 維 | ↑ | repo license `NOASSERTION`（未核實） | 未測（round-1 §6 不接） |
| **HEAR Track1／2** | 歌曲 Musicality／SongEval 5 維 | 1／5 維 | ↑ | Apache-2.0（HF 實查） | 未測 |

**只進閘門的候選**：AES PQ、MEva、MuQ-Eval A1 MI、A3a MI、CMI-RM empty quality、PAM、SongEval se_mus（僅 B3）、SingMOS（僅 B4）、MERT-ref／MuQ-ref（僅 B1）。其餘（PC、TA、alignment、centroid、CLAP、SongBench、HEAR、SongEval 其餘 4 維）只描述。

**SongEval 與 SingMOS 的域外警告**：SongEval 頭在 10 s 純器樂上已由 126d 判過；本輪把它放回它自己的歌曲域，測「指標本身在訓練域內是否至少與人類一致」，不能外推成短片段有效。SingMOS 只在歌唱 clips 上聲稱有效，不套到純器樂。

---

## 5. 訓練重疊旗標（metric × dataset，**預先寫定**）

規則沿用 round 1：「獨立」只指標籤未用於該模型訓練，**不保證音訊或系統分布未見過**。重疊格只能描述，不進閘門。`未核實` 一律不進閘門。

| 指標 | B1 ODAQ | B2 AIME | B3 SongEval | B4 SingMOS-Pro | B5 MC522 | B6 Music Arena |
|---|---|---|---|---|---|---|
| AES | 獨立（標籤；音訊重疊未核實） | 獨立（標籤；音訊未核實） | 獨立（標籤；音訊未核實） | 獨立（標籤；音訊未核實） | 獨立（round-1：重疊未知→描述） | 獨立（標籤；音訊未核實） |
| MEva | 獨立 | **重疊**（AIME 是其訓練 benchmark；round-1 §2） | **重疊**（SongEval 是訓練 benchmark） | 獨立 | 獨立（round-1） | **未核實**（MusicPref 是否含 Music Arena） |
| MuQ-Eval A1 | 獨立 | 未核實 | **重疊**（A3a 含 SongEval 維度；A1 未核實） | 獨立 | 獨立（round-1） | 未核實 |
| MuQ-Eval A3a | 獨立 | 未核實 | **重疊** | 獨立 | 獨立 | 未核實 |
| CMI-RM | 獨立 | 未核實（CMI-Pref 是否含 AIME） | **重疊**（訓練含 SongEval） | 獨立 | 獨立（round-1） | 未核實 |
| PAM | 獨立（零樣本 CLAP） | 獨立 | 獨立 | 獨立 | 獨立 | 獨立 |
| SongEval | 獨立（域外） | 獨立（域外） | **重疊／未核實**（訓練資料；官方 held-out split 未核實） | 獨立（域外） | 獨立（域外） | 獨立（域外） |
| SingMOS | 獨立（域外） | 獨立（域外） | 獨立（域外） | **test 獨立／train 重疊**（predictor 訓練於 SingMOS-Pro） | 獨立（域外） | 獨立（域外） |
| MERT-ref／MuQ-ref | 獨立（無監督品質標籤） | — | — | — | — | — |
| MERT-cent／MuQ-cent | — | 獨立（只描述） | — | 獨立（只描述） | — | — |
| CLAP | — | 獨立（標籤） | — | — | 獨立（round-1） | 獨立（標籤） |
| SongBench／HEAR | 未核實 | 未核實 | **未核實／可能重疊**（SongEval 維度） | 未核實 | 未核實 | 未核實 |

**因此每個指標真正能進閘門的 dataset 數**：AES 6、MEva 4（B1/B4/B5＋B6 未核實→不算）、MuQ A1/A3a 3–4、CMI-RM 3–4、PAM 6、SongEval 0（B3 重疊；其他域外 → 全部只描述）、SingMOS 1（B4 test）、MERT-ref 1（B1）。

> 注意：這張表會讓「通過 round 2」變得很難——這是刻意的。round 1 的教訓是不要用重疊格當泛化證據。

---

## 6. 統計量與 CI

### 6.1 母體與 cluster

| 資料集 | 分析單位 | cluster（bootstrap） | 人類 target |
|---|---|---|---|
| B1 ODAQ | trial（同來源 excerpt）內 5 個 processed 條件 | trial | MUSHRA BAQ |
| B2 AIME | snippet（1,300）× 對手；票為重複 | snippet（同一 snippet 的多票相關） | 成對偏好（Music Quality） |
| B3 SongEval | song（2,399；S800 子集做變體） | song | Musicality（primary）／其餘 4 維 |
| B4 SingMOS | clip（7,981；G1 用 test 3,101） | system（141）＋clip 巢狀；主報 clip-level cluster | per-clip MOS |
| B5 MC522 | clip（522） | clip | 人類 PQ |
| B6 Music Arena | battle（2,000） | battle（或 prompt；取較保守的 battle） | 成對偏好（排除 TIE／BOTH_BAD） |

### 6.2 每個 cell 報什麼

1. Pearson、Spearman、Kendall τ-b（metric vs human）。
2. **系統內** Spearman：分數與人類都先扣掉 system 平均（B4 141、B2 13、B6 依系統數；B3 無 model 標籤 → 不報；B1 用 within-trial，見下）。
3. **成對一致率**：metric 選人類偏好者的比例；|Δmetric| < m 算平手、計 0.5；m = **0.10 × S**，S = 該資料集 unique audio 上 metric 的 range（126 慣例）。另報 m=0 的 forced-choice 作為敏感度。
4. **人類天花板**：split-half reliability（聽眾／標註者對半）與 Spearman，只當參考不進門檻。B2／B6 每票只有一個答案 → 用 vote-level bootstrap，不報 split-half。
5. **系統排名**：B2（13 systems）與 B6（多 systems）用 Bradley-Terry 配適人類偏好，與 metric 的系統平均排名算 Kendall τ（描述）。
6. **ODAQ within-trial**：每個 trial 內 5 個 processed 條件的 metric 與 MUSHRA 的 Spearman；報 trial 平均與 trial bootstrap CI。另報 Q1（最差）vs Q5（最好）方向正確的 trial 比例（126 的 80% 規則）。
7. **ODAQ 成對**：trial 內 C(5,2)=10 對；只取「人類 26 位聽眾的 ΔMUSHRA bootstrap 95% CI 不含 0」的對；一致率與 CI（描述，因對數 < 500）。

### 6.3 Bootstrap

- 2,000 次、seed **20261025**、percentile 95%。
- 同一組索引重抽所有指標 → 差值 CI 是成對的。
- 段長 Δρ（V5−V10、V30−V10、VFULL−V10）與響度 Δρ（norm−raw）用同一組 item 索引重抽，CI 是 paired。
- 相關係數的 CI 用 Fisher z 或 percentile，兩者都報；以 percentile 為主。

---

## 7. 預登錄門檻

### 7.1 沿用 round-1 的三個閘門

只用 §5 標「獨立」且樣本數足夠的格子；重疊／未核實格照報但不計入。

- **G1（相關）**：Spearman 點估計 ≥ **0.50** 且 95% CI 下界 > **0.30**。
- **G2（系統內）**：系統內 Spearman ≥ **0.50** 且 CI 下界 > **0.30**；只套用在 ≥5 個系統、每系統 ≥20 clips 的資料集（B2、B4、B6）。
- **G3（成對）**：一致率 CI 下界 > **0.55**；只套用在 ≥500 個人類可分辨配對的資料集（B2 約 7,800 票、B6 2,000 battles）。

### 7.2 新增（本輪專屬，先寫死）

- **G4（段長穩定，描述性閘門，不判指標好壞）**：對 B1、B3、B5 的人類評分長度 > 變體長度者，Δρ = ρ(V_short) − ρ(V10)。
  - `stable`：|Δρ| < **0.05** 且 paired 95% CI 含 0。
  - `sensitive`：|Δρ| ≥ **0.05** 且 CI 不含 0。
  - 其餘 `inconclusive`。
  - 方向（5 s 比 10 s 好或差）照實報，不預設。
- **G5（響度穩定，描述性閘門）**：Δρ = ρ(norm) − ρ(raw)。
  - `loudness-robust`：|Δρ| < **0.05** 且 CI 含 0。
  - `loudness-dependent`：|Δρ| ≥ **0.10** 且 CI 不含 0。
  - 0.05–0.10 或 CI 跨零 → `inconclusive`。
- **G6（ODAQ within-trial）**：
  - **G6a**：trial 平均 within-trial Spearman ≥ **0.50** 且 trial bootstrap CI 下界 > **0.30**。
  - **G6b**：Q1 vs Q5 方向正確的 trial 比例 ≥ **0.80**（126 的 80% 規則）；CI 下界 > 0.60。
  - 兩者都要成立才算過。
- **G6 的頻寬排除（預先寫死）**：依各指標的實際輸入 Nyquist 判定 LP 條件是否可見——**16 kHz 路徑**（AES、SingMOS；MEva 若為 16 kHz）Nyquist 8 kHz → 只有 **LP Q1（5 kHz）** 可見，Q2（9 kHz）以上 `bandwidth-blind`；**24 kHz 路徑**（MuQ-Eval、CMI-RM、MERT）Nyquist 12 kHz → LP Q1–Q3（≤10.5 kHz）可見、Q4（12 kHz）邊界、Q5（15 kHz）blind；**44.1／48 kHz 路徑**（PAM、CLAP）全部可見。SH 的頻帶分布未核實，只用「部分可見」處理（不整組排除）。`bandwidth-blind` 的條件不進 G6a／G6b，但照跑照報。**G6a** 的 within-trial Spearman 只用該指標可見的條件計算。**G6b** 改用「該 trial 內可見條件中，人類 MUSHRA 最高 vs 最低」判方向；LP 在 16 kHz 指標下整組不進 G6（只剩 Q1 可見）；LP 在 24 kHz 指標下用 Q1 vs Q3。

### 7.3 判定規則

- **round-2 候選**：在 ≥2 個「獨立」dataset 上，適用的 G1／G2／G3 都成立，其中 ≥1 個是生成／合成音樂（B2、B3、B4、B6），且若 B1 可算則 G6 也成立。
- **排除**：在 ≥1 個獨立的生成音樂 dataset 上，G1 或 G3 的 CI **上界** < 門檻（0.50／0.55）——即證據顯示未達門檻，不只是不確定。
- **未定**：其餘（含獨立 dataset < 2、CI 跨門檻）。
- **不得改寫 round 1**：本輪判定只記在 round-2 結果檔；round-1 的「未定／排除」保持原樣。
- **多重比較**：不校正（約 9 個候選 × 6 個 dataset）；通過者要註記這個自由度。描述性格子不進判定。
- **B3 SongEval 不進任何閘門**（SongEval 頭重疊、其他指標域外或音訊重疊未核實），只描述；它的價值在段長與 5 維的相對排序。
- **B6 的 TIE 分析只描述**：n_tie 可能 < 500，且沒有 per-clip 分數可校準 m。

---

## 8. 與 126a／126c 介入的對應（人類排序 vs 指標）

這節是本輪相對 round 1 最大的新增：**人類覺得哪種劣化更差**，直接對上我們自己造的介入。ODAQ 的 6 個方法類別與 126 的對應如下（方向預先寫死；`↑`＝MUSHRA 隨 Q 編號上升，metric 也應上升）：

| ODAQ 方法 | 條件（Q1→Q5） | 126a／126c 對應 | 預期人類方向 | 16 kHz 可見性 | 預測 |
|---|---|---|---|---|---|
| **LP**（low-pass） | 5, 9, 10.5, 12, 15 kHz | 126a lowpass 1／2／4 kHz | Q1 最差、Q5 最好 | 16 kHz：只有 Q1 可見；24 kHz：Q1–Q3 可見、Q4 邊界、Q5 blind；44.1／48 kHz：全可見 | PQ／CE 應在可見範圍內單調；blind 格應平手 |
| **UN**（unmasked noise） | 3, 5, 7, 9, 10.5 kHz | 126a 白噪 SNR 40→0 dB | Q1 最差 | 大部分可見 | PQ 應單調；MEva 可能全盲（126a 白噪 0/5） |
| **TM**（tonality mismatch） | 3, 5, 7, 9, 10.5 kHz | 126a 白噪／低通混合 | Q1 最差 | 大部分可見 | 比 UN 更難；預測多數指標弱反應 |
| **SH**（spectral holes） | 70／50／30／20／10% | 126a dropout／bitcrush | Q1（70%）最差 | 高頻洞 blind；低頻洞可見 | CE／CU 應比 PQ 敏感（126a bitcrush 模式） |
| **PE**（pre-echo） | 5 檔 | 126a mp3 160k／64k | Q1 最差 | 可見 | 預測多數指標**測不到**（126a mp3 只有 1/5 平手） |
| **DE**（source separation） | 5 個系統 | **126c htdemucs 去軌／stemsum** | 系統間有明顯差異 | 可見 | 126c 曾見「分離殘差讓 PQ 變高」；本輪用人類 MUSHRA 判這個方向是否合理 |

**預先登錄的判讀規則**：

1. hidden reference 必須是該 trial 內 metric 分數最高（或前 2）——若做不到，該指標在該 trial 的「絕對錨」失效，記 `ref-anchor fail`。
2. 對每個方法類別，報「Q1 vs Q5 方向正確的 trial 比例」與「within-trial Spearman」；G6 只看 overall，不分方法。
3. **LP 的 `bandwidth-blind` 格**（16 kHz 的 Q2–Q5、24 kHz 的 Q5）與 SH 的部分頻帶若出現「人類差很大、metric 判平手」，**不算指標失效**；反之若 metric 判出大差，才是異常（可能是重採樣 alias）。
4. DE 的人類排序用來檢定 126c 的「分離殘差讓 AES PQ 上升」是否與人類相反；若人類也覺得 htdemucs 去軌比較好，則 126c 的負面解讀要修正。

---

## 9. GPU 時間表與 queue job 切分

### 9.1 吞吐假設（全部來自實測或既有 log）

| 模型 | 假設 | 來源 |
|---|---|---|
| AES | 0.0008 s／音訊秒（0.008 s／10 s） | karaoke MEva log：96 batch / 25 s；batch 32 → 0.008 s/clip |
| MEva | 0.0040 s／音訊秒（0.040 s／10 s） | karaoke log：3,051 clips / 123 s |
| MuQ-Eval A1 | 0.12 s／clip（固定中央 10 s） | round-1：2,748 / ~300 s；MC522 含啟動 0.28 |
| MuQ-Eval A3a | 0.08 s／clip | round-1：2,748 / ~200 s |
| CMI-RM | 0.35 s／30 s chunk（≈0.012 s／音訊秒；81 s 檔 3 chunks ≈ 1.05 s） | round-1：2,748 / 950 s（30 s rounds）；MC522 100 / 50 s |
| PAM | 0.028 s／音訊秒（0.29 s／10 s） | round-1：MC522 300 / 86 s |
| SongEval | 0.0027 s／音訊秒（0.08 s／30 s 窗） | **估計**（126d 未記逐檔時間）；誤差可能 ±50% |
| MERT／MuQ | 0.0012 s／音訊秒（約 98 clips/s） | 133 預登錄：1.33M / 98 |
| CLAP | 0.008 s／音訊秒（0.08 s／10 s） | 標準 eval 的 batch-1 經驗 |
| SingMOS | 0.05 s／clip | **估計**（wav2vec2-large、16 kHz） |
| SongBench／HEAR | 0.0027 s／音訊秒 | **估計**（未實作） |

### 9.2 每個資料集 × 模型（名目小時）

| job／資料集 | 輸入數 | 平均長度 | 最大三項 | 小計 |
|---|---|---|---|---|
| J1 B1 ODAQ（240×4 變體） | 960 | 10.0 s | CMI-RM 0.09、PAM 0.07、A1 0.03 | **0.24 h** |
| J1 B5 MC522（522×3） | 1,566 | 8.3 s | CMI-RM empty 0.15、caption 0.15、PAM 0.10 | **0.54 h** |
| J2 B2 AIME（1,300×2） | 2,600 | 10.0 s | CMI-RM empty 0.25、caption 0.25、PAM 0.20 | **0.96 h** |
| J3 B4 SingMOS（7,981×2） | 15,962 | 5.0 s | CMI-RM 1.55、PAM 0.62、A1 0.53 | **3.45 h** |
| J3 B4b SingMOS center-5（test） | 3,101 | 5.0 s | CMI-RM 0.30、PAM 0.12、A1 0.10 | **0.66 h** |
| J4a B3 SongEval full raw（S800） | 5,600 窗 | 30.0 s | PAM 1.31、CMI-RM 0.54、MEva 0.19 | **2.76 h** |
| J4b B3 SongEval full norm（S800） | 5,600 窗 | 30.0 s | PAM 1.31、CMI-RM 0.54、MEva 0.19 | **2.51 h** |
| J4c B3 SongEval 段長窗（V30／V10 全量／V5） | 4,799 | 11.7 s | CMI-RM 0.47、PAM 0.44、A1 0.16 | **1.29 h** |
| J5a B6 Music Arena raw（PAM／AES／MEva） | 3,780 檔 | 81.3 s | PAM 2.39、MEva 0.34、AES 0.07 | **2.80 h** |
| J5b B6 Music Arena raw（CMI／MuQ／CLAP） | 3,780 檔 | 81.3 s | CMI empty 1.10、caption 1.10、CLAP 0.68 | **3.10 h** |
| J5c B6 Music Arena norm 子集 | 500 檔 | 81.3 s | PAM 0.32、CMI empty 0.15、caption 0.15 | **0.78 h** |
| J6 SongBench＋HEAR（探索） | 5,600 窗 | 30.0 s | SongBench 0.13、HEAR 0.13 | **0.25 h** |

**合計：名目 19.34 h；加 15% I/O／啟動 22.24 h。** 上限設 **24 h**（名目已包含 Music Arena 實測平均長度 81.3 s；若 SongEval／SongBench 估計慢 50%，J4a／J4b／J6 約 +2.5 h，仍在 24 h 內）。

### 9.3 Job 切分（p2 queue，全部 < 4 h，可 resume）

| Job | 內容 | 名目 | 含 15% | 完成判準（contract） | resume key |
|---|---|---|---|---|---|
| **J1** | ODAQ＋MC522，全指標 | 0.69 h | **0.8 h** | 每 (dataset, variant, model) TSV 列數 = manifest 列數、無 NaN | `(dataset, item_id, variant, model, protocol)` |
| **J2** | AIME，全指標＋caption／CLAP | 1.05 h | **1.2 h** | 同上；另 caption TSV 只出現在 AIME | 同上 |
| **J3** | SingMOS-Pro，全指標＋SingMOS predictor | 4.22 h | **4.9 h** | 7,981×2 列、test 3,101 列 | 同上 |
| **J4a** | SongEval S800 full raw | 1.79 h | **2.1 h** | 800 首×窗數、每首至少 5 窗 | `(song, window_idx, model)` |
| **J4b** | SongEval S800 full norm | 1.79 h | **2.1 h** | 同上 | 同上 |
| **J4c** | SongEval 段長窗 V30／V10／V5 | 1.39 h | **1.6 h** | 800×2＋2,399 列 | `(song, variant, model)` |
| **J5** | Music Arena 2,000 battles | 3.46 h | **4.0 h** | 每 battle 兩側都有分數；TIE／BOTH_BAD 保留 | `(battle_uuid, side, model)` |
| **J6** | SongBench＋HEAR（探索，可跳過） | 0.24 h | **0.4 h** | 兩模型都有輸出；失敗不影響主表 | 同上 |

**Queue 注意事項**（依 memory `reference_gpu_queue_job_authoring_gotchas.md`）：contract 先於 wrapper；`pending/` 不可 staging dotfile；>30 min 一律走 p2；job 開頭掛 `notify_on_exit`；p1 host 已停用、不搶佔。J6 可跳過，總計仍 16.4 h（在 12–24 h 內）。

### 9.4 為什麼不是更多／更少

- 不把 Music Arena 全 8,117 battles 放進來：PAM 對長音訊的成本會讓單一資料集 > 8 h；2,000 場已能覆蓋 4 種偏好類別與 12 個月。
- 不把 SongEval 2,399 首全跑 VFULL×2：PAM 成本翻 3 倍（約 +5 h），而 S800 的 paired CI 已足夠回答段長。
- 不重跑 PAM400／ME／MC522 全量：round 1 已定，重跑只是燒 GPU。

---

## 10. 磁碟、下載與前置作業

### 10.1 路徑

全部放 **`/mnt/seagate/benchmarks_r2/`**（ext4、2.9 TB free）；NVMe 目前 78 GB free（98% 滿），**不放大檔**。`/mnt/HDD` 是 exFAT 且 87% 滿，避開。

```
/mnt/seagate/benchmarks_r2/
├── downloads/     # 原始 zip／parquet／mp3（~85 GB；AIME 62.3 GB 占大部分，Music Arena 只抓抽中子集）
├── derived/       # 解碼後 16 kHz／模型 SR 快取（~15 GB）
├── scores/        # 逐 clip TSV／JSONL（~2 GB）
├── analysis/      # bootstrap、gate 表、圖（~1 GB）
├── weights/       # 新權重（SongBench、HEAR、SingMOS；~3 GB）
├── meta/          # 已下載的小 metadata（13 MB，本次查核用）
└── logs/
```

### 10.2 下載清單（精確指令）

```bash
mkdir -p /mnt/seagate/benchmarks_r2/{downloads,derived,scores,analysis,weights,meta,logs}
cd /mnt/seagate/benchmarks_r2/downloads

# B1 ODAQ（1.05 GB）
curl -L --retry 3 -o ODAQ.zip "https://zenodo.org/records/10405774/files/ODAQ.zip?download=1"
md5sum ODAQ.zip          # 期望 9b4c1eba479ea698995798523eb80999
unzip -q ODAQ.zip -d ODAQ
head -5 ODAQ/_detailed_license.csv     # 逐檔授權，先讀再決定能否使用

# B3 SongEval（11.1 GB，2,399 mp3 + metadata.jsonl）
~/venvs/dac/bin/huggingface-cli download ASLP-lab/SongEval --repo-type dataset \
  --local-dir /mnt/seagate/benchmarks_r2/downloads/SongEval

# B4 SingMOS-Pro（1.48 GB）
~/venvs/dac/bin/huggingface-cli download TangRain/SingMOS-Pro --repo-type dataset \
  --local-dir /mnt/seagate/benchmarks_r2/downloads/SingMOS-Pro

# B2 AIME（62.3 GB，210 parquet shards；survey metadata 另抓）
~/venvs/dac/bin/huggingface-cli download disco-eth/AIME --repo-type dataset \
  --local-dir /mnt/seagate/benchmarks_r2/downloads/AIME
curl -L -o /mnt/seagate/benchmarks_r2/meta/aime_survey.parquet \
  "https://huggingface.co/datasets/disco-eth/AIME-survey/resolve/main/data/train-00000-of-00001.parquet"

# B6 Music Arena：先抓全部 battle JSON（8,117 個、25 MB），排除 audio 為 null（sao／sao-small）後抽 2,000 場，再抓對應 mp3
~/venvs/dac/bin/huggingface-cli download music-arena/music-arena-dataset --repo-type dataset \
  --local-dir /mnt/seagate/benchmarks_r2/downloads/MusicArena --include "battle_data/*/*.json"
# 之後：python scripts/eval/bench_r2/pick_music_arena.py --battles ... --n 2000 --seed 20261025
# 產生 audio list，逐檔 hf_hub_download（13,090 檔中約 3,200 檔）

# 權重
mkdir -p /mnt/seagate/benchmarks_r2/weights
curl -L -o /mnt/seagate/benchmarks_r2/weights/songbench.safetensors \
  "https://github.com/Tencent/SongBench/raw/main/ckpt/songbench.safetensors"   # 100,808,636 B
~/venvs/dac/bin/huggingface-cli download earlab/EAR_HEAR \
  --local-dir /mnt/seagate/benchmarks_r2/weights/EAR_HEAR                    # track_1/2.pth
# SingMOS predictor：torch.hub 首次呼叫會下載到 ~/.cache/torch/hub
```

### 10.3 下載後、評分前的檢查（全部 CPU，不佔 GPU）

1. **檔數與 manifest**：SongEval 2,399 個 mp3、SingMOS 7,981 個 wav、ODAQ 240 檔、AIME 1,300 snippet（從 parquet 解出後比對）。
2. **格式**：每資料集抽 20 檔 `ffprobe`，記錄 sample rate／channels／bit_rate／duration；SongEval 的 mp3 bitrate 若 < 64 kbps，記為限制（見 §11）。
3. **授權**：ODAQ `_detailed_license.csv` 逐檔確認；AIME 的 MTG-Jamendo 500 軌逐軌授權；SongEval NC-SA 不再散布。
4. **重複與洩漏**：AIME snippet 是否與 MusicCaps／PAM 重疊（用 track id／檔名比對，不能比音訊）；SingMOS 的 dataset 欄位與 round-1 無關。
5. **近靜音／削波**：每資料集報 LUFS、peak、靜音比例（< −45 dBFS）；ODAQ hidden ref 的 peak 應 < 1.0。
6. **smoke**：每模型 × 每資料集抽 8 檔跑通、輸出有限值；**這一步仍是 GPU，必須在預登錄 commit 之後、正式 job 之前**，且結果只記 smoke，不進分析。

### 10.4 本次已下載的小 metadata（13 MB）

| 檔案 | 大小 | 用途 |
|---|---|---|
| `meta/aime_survey/train.parquet` | 295,100 B | 驗證 1,300 snippet、10.0 s、13 systems、7,800／7,800 票 |
| `meta/songeval/metadata.jsonl` | 1,267,359 B | 驗證 2,399 列、4 位標註／首、5 維欄位 |
| `meta/singmos_pro/{metadata,score,split,sys_info}.json` | 11.6 MB | 驗證 7,981 列、test 3,101、141 systems、SR 分布 |
| `meta/music_arena/battle_sample.json` | 1.4 KB | 驗證 TIE／BOTH_BAD 欄位、duration 與 SR |

沒有下載任何多 GB 音訊；沒有跑 GPU。

---

## 11. 風險與限制

1. **ODAQ 授權需逐檔確認**：README 稱全部為 Creative Commons，但未列條款組合；逐檔表在 zip 內（`_license_disclaimer.txt`／`_detailed_license.csv`，細項未核實）。若某些檔是 NC，我們只做非商用研究，但仍不能把分數檔或音訊再散布；分數 TSV 只留內部。
2. **ODAQ 的 16 kHz 頻寬盲**：LP Q4／Q5（12／15 kHz）與 SH 高頻洞在 16 kHz 輸入下不存在；已用 G6 的 `bandwidth-blind` 排除，但這會讓「低通」結論只覆蓋 5–9 kHz。
3. **SongEval 的 mp3 可能低 bitrate**：130 KB／檔的樣本暗示低位元率；若真是 32 kbps，編碼失真會混進美學評分，且 SongEval 不適合當 codec 敏感度證據。下載後用 ffprobe 確認。
4. **SongEval 沒有官方 held-out split**：SongEval 指標在 B3 一律描述；不得用 B3 說 SongEval 指標有效。
5. **SongEval 每首只有 4 位標註者**：人類天花板低於 MC522（10 位）與 ODAQ（26 位）；G1 的 ρ 上限會受壓縮。
6. **AIME 62.3 GB 下載**：網路與 seagate 空間都要先確認；MTG-Jamendo 500 軌的逐軌授權可能限制再散布。
7. **Music Arena 沒有品質控制**：使用者自選 prompt、無黃金標準；TIE 可能很少（<150），TIE 分析只描述；音訊長度變動大（實測 30–200 s），成本假設平均 60 s。由 1.92 MB/檔推算，若平均其實 ~120 s（128 kbps），J5 增加約 2.4 h → 總計約 17.0 h 名目／19.5 h 含 15%，仍在 24 h 內。README 明載 `sao`／`sao-small` 的 battle 無音訊，抽樣前排除，實際可用 battle < 8,117。
8. **模型授權**：CMI-RM、MuQ 編碼器是 CC-BY-NC-4.0；SongBench 是 NOASSERTION；SingMOS 權重 payload 授權未核實。全部只做非商用研究，不再散布權重。
9. **重疊旗標的保守性**：MEva 在 AIME／SongEval 重疊、CMI-RM 在 SongEval 重疊、MuQ-Eval A3a 在 SongEval 重疊；這些格只描述，會讓「通過」變難。
10. **吞吐估計**：SongEval／SingMOS／SongBench／HEAR 的 s/clip 是估計，±50%；J4a/J4b 各 2.1 h、J3 4.9 h，上限 22 h 吸收。
11. **段長解釋**：AIME／SingMOS 的 V5 是截斷 counterfactual，不是「人類聽 5 s」；只有 B1／B3／B5 能回答真正的段長問題。
12. **沒有我們 arm 的人類評分**：本輪仍是 benchmark 層；不能直接說「我們的 TTM 輸出被哪個指標正確評價」。
13. **沒有多數人類 TIE 的 per-clip 資料**：只有 Music Arena 的 battle TIE；不能把 TIE 校準外推到其他資料集。
14. **MEva 輸入長度未核實**：若它固定吃 10 s，VFULL 的語義是「逐窗平均」，與 SongEval 指標的 30 s 窗不同；已在 §3.5 寫死。

---

## 12. 實作範圍（另一個任務）

**新增檔案（只加不改）**

```
scripts/eval/bench_r2/
├── datasets.py        # manifest：B1–B6、變體、抽樣（seed 20261025）
├── decode_cache.py    # 解碼／重採樣／快取；key 含 sha256
├── pick_music_arena.py
├── score_r2.py        # --dataset --variant --model --protocol；逐列寫、skip 既有
├── analyze_r2.py      # G1–G6、bootstrap、split-half、BT 排名
└── README.md
```

**不得修改**：`scripts/eval/eval_metrics.py`、`scripts/eval/mc_mf25_eval*.sh`、`scripts/eval/metric_win_tie_126*.py`、`scripts/eval/metric_win_tie_133.py`、`scripts/eval/newmetrics_*.py`（全部被 contract 綁 sha）。新 driver 只 import，不複製貼上。

**輸出**

```
/mnt/seagate/benchmarks_r2/scores/<dataset>/<model>__<protocol>__<variant>.tsv
/mnt/seagate/benchmarks_r2/analysis/r2_cells.json
/mnt/seagate/benchmarks_r2/analysis/r2_gate_tables.md
docs/experiments/results/benchmark_screening_r2_20261025_results.md   # 結果（另一個任務）
```

**Queue**：J1–J6 對應 p2 job（編號由 queue 指派）；contract 先於 wrapper；每個 job 掛 `notify_on_exit`；完成後把 `r2_cells.json` 與 gate 表寫回 repo（小檔）。

**順序**：commit 本預登錄 → 下載／檢查（CPU）→ 寫 driver＋smoke（GPU < 10 min，不進分析）→ commit 程式 → p2 J1…J6 → 分析 → 結果檔。

---

## 13. 未核實清單（2026-10-25 實作時核實；設計未因此更動）

| # | 項目 | 結果 | 狀態 |
|---|---|---|---|
| 1 | ODAQ 逐檔授權 | `_detailed_license.csv` 66 列（含 15 個空行與 `[1]` 引文列）；原始素材 CC BY 4.0／3.0／2.5、CC0（glockenspiel／claves／SitaSings 等）；Fraunhofer 與 Netflix 自有音樂（`female_speech_music_*_BG／LD*`、`Lonely Path`）為 **CC BY-NC 4.0**。逐檔計數 CC BY 31／BY-NC 9／CC0 5（README 表）。結果檔只發表指標分數、不散布音檔 → NC 不構成問題 | 已核實 |
| 2 | ODAQ 30 trials 對應 | 30 trial＝10 個 DE 資料夾（`DE_*_LD*`，語音＋背景混音）＋5 個方法（LP、…）× 4 個素材（`LP_11_guitar`、`LP_23_jazz`、`LP_AmateurOnPurpose`、`LP_CreatureFromTheBlackjackTable` 等）；240 檔＝30 trial × 8 process | 已核實 |
| 3 | `ODAQ_results.csv` 欄位 | 6,241 行（6,240 筆評分）；欄位 `score,method,condition,process,item,subject`；26 位聽者（`Subject n: <site-id>`，另有每位聽者的 XML）；MUSHRA 0–100，BAQ 以 trial×process 對聽者取平均 | 已核實 |
| 4 | SongEval 官方 split 與 SongEval 指標訓練範圍 | `metadata.jsonl` 只有 `file_name／gender／annotation` 三欄，**沒有 split 欄**；官方 train／test 切分與 SongEval 模型的訓練集沒有逐檔核實。§5 的重疊旗標維持預登錄的「SongEval 指標 × SongEval 資料＝重疊」，結論只寫 observation | **未完全核實**（保守旗標） |
| 5 | SongEval mp3 bitrate／SR | ffprobe 20 檔（seed 20261025）：mp3、44.1／48 kHz、單／雙聲道、64–320 kbps、127–365 s；無錯誤 | 已核實 |
| 6 | AIME track id 對應 | `description` 欄＋`id`＋`model` 可對上 survey；parquet 210 shard、6,500 track；1,300 snippet 全部找到 track；與 MC522 key 重疊 0 | 已核實（來源 hash 見 §14.3） |
| 7 | Music Arena 偏好分布／TIE | 抽樣後 2,000 battle＝3,998 個 mp3（見 `meta/music_arena_pick.json`）；其餘統計由 analyze_r2 在評分後輸出，**此處未另外列表** | 部分核實 |
| 8 | Music Arena 排除 `sao` 後 mp3 數與 404 | 2,000 battle × 2＝4,000 條，2 條 404（`meta/music_arena_missing.json`）→ 3,998；受影響 battle 兩邊不齊，分析時排除（§14.6） | 已核實 |
| 9 | SingMOS predictor 權重授權 | predictor 經 torch.hub 取得並可執行（見 §14.4）；權重 payload 的 LICENSE 沒有逐檔讀 | **未核實** |
| 10 | SongBench `NOASSERTION` | J6 跳過、未連接 SongBench，故未讀 LICENSE（§14.2） | **未核實（不影響主表）** |
| 11 | MEva 輸入長度與 SR | 32 kHz 輸入、30 s 切塊最長 360 s、pool 到 1,500 frame；因此 analyze_r2 的 bandwidth-blind LP 規則把 MEva 路徑訂為 Nyquist 16 kHz（> Q5 15 kHz → 全可見；原先誤設 16 kHz 已改，見 §14.8） | 已核實 |
| 12 | HEAR 輸入長度／維度對應 | J6 跳過，未讀 | **未核實（J6 跳過）** |
| 13 | CMI-RM 訓練是否含 AIME | 本輪沒有讀 CMI-Pref 論文／資料卡；§5 沿用預登錄旗標 | **未核實** |
| 14 | MEva 訓練是否含 Music Arena | 未讀附錄；§5 沿用預登錄旗標 | **未核實** |
| 15 | SongEval 16 位評分者 vs 每首 4 位 | 每首 `annotation` 恆為 4 筆（metadata 驗證），16 位為輪換池，與論文描述一致；未逐篇比對論文 §3 | 部分核實 |

### 13.1 §10.3 檢查結果（`analysis/cache_check.json`，只含已解碼部分）

| 資料集 | manifest／index | invalid | ffprobe 20 檔 | 峰值／LUFS／靜音 |
|---|---|---|---|---|
| ODAQ | 240／960 | 0 | 48 kHz 雙聲道，pcm 16／24／f32，10.0–16.3 s | peak max 0.971（無 ≥ 1.0，hidden ref 條件通過）、LUFS p50 −26.6、靜音 0 |
| MC522 | 522／1,566 | 0 | 16 kHz 單聲道 pcm_s16，10 s | 43 檔 peak = 1.0（滿刻度）、LUFS p50 −16.7、靜音 proxy 1 檔（LUFS < −45） |
| SongEval | 2,399／（解碼中，僅 smoke 4 列） | 0 | 見 #5 | 只有 smoke 列 |
| Music Arena | 3,998／（解碼中，僅 smoke 列） | 0 | 32／44.1／48 kHz mp3，83–202 kbps，20–253 s | smoke 64 列中 33 列 peak ≥ 1.0（mp3 解碼過衝，最大 2.72；norm 變體 `over_fs` 欄記錄） |

SingMOS、AIME 與上列兩個解碼中的資料集待解碼完成後重跑 `check_cache.py`；靜音比例是 LUFS_in < −45 的 proxy，不是 §10.3 原文的樣本級 −45 dBFS。授權（§10.3 第 3 點）：ODAQ 已核實；AIME 逐軌授權、SongEval NC-SA 未逐軌核對。重疊（第 4 點）：AIME vs MC522 以 key 比對 0 重疊（與 PAM 未比）。

---

## 14. 實作偏離與限制（只記實作層面；預登錄門檻、種子、變體、重疊旗標皆未改）

1. **J1 沒掛 `notify_on_exit`**：J1 的 wrapper 沒有 `notify_on_exit` trap（140 失敗後 141→142 重排時沿用，未補）；之後 143–150 由 `make_queue_job.py` 產生，已走 queue 的通知。
2. **J6（SongBench＋HEAR 探索）跳過**：依 §9.3「可跳過」，不影響主表；#10、#12 因此未核實。
3. **AIME manifest 以 metadata hash 建立**：59 GB 音訊在共用碟 I/O 競爭下讀取過慢（已終止），`aime_track_index` 只讀 `id／model／description` 三欄，來源 provenance hash＝sha256(`id|model|description`)[:16]，**不是音訊 bytes hash**；音訊在 decode 階段才逐 shard 讀取（每 shard 一次）。
4. **SingMOS predictor**：torchaudio 缺 API，以 stub（`pylib_singmos/`）執行；smoke 通過（AIME smoke 8 檔 singmos 2.84），但與官方環境的數值一致性沒有另外驗證；權重授權未核實（#9）。
5. **J3／J5 切半，且 AIME 獨立成 J2**：J3→J3a／J3b（2.4／3.2 h）、J5→J5a／J5b（3.2／3.7 h），單 job 皆 < 4 h；queue 編號 142 J1、143 J3a、144 J3b、145 J5a、146 J5b、147 J4a、148 J4b、149 J4c、150 J2；合計約 21.5 h（含 15% 開銷，J6 不計），落在 12–24 h 內。實測 J1 因共用碟競爭較名目慢，後續工時可能超出，stall 設 28,800 s。
6. **Music Arena 2 個 mp3 404**：受影響 battle 兩側不齊，分析排除，不補抽樣（預登錄抽樣 seed 20261025 不重抽）。
7. **126d／133 的 sha 未 pin**：contract 的 bindings 沒有納入 `metric_win_tie_126d.py`／`metric_win_tie_133.py`／`newmetrics_*.py`（這些檔案目前有未 commit 的修改，不屬於本任務）；bench_r2 只 import 其評分函式，若上游變動，結果不可逐位重現。
8. **analyze_r2 的 MEva LP 規則**：原假設 16 kHz，核實為 32 kHz 輸入，已改為 44（全可見）；此為實作 bug fix，規則本身（依輸入 Nyquist）不變。
9. **analyze_r2 只在部分 ODAQ 分數上測過**；Kendall τ 的 CI 沒有實作（只報點估計與 bootstrap 的 Spearman／配對勝率）；正式結果前需在完整分數上再跑。
10. **J1 contract sha 已不吻合**：`decode_cache.py`／`bench_datasets.py`／launcher 在 J1 啟動後為 AIME 修改（只動 AIME 分支）。contract sha 僅在 `run_job --preflight` 檢查，J1 已通過；`--validate-only` 只讀 summary 報告、不重算 sha，因此不影響 J1 收尾。143–150 已用新 sha 重新產生 contract。J1 不重排。
11. **跨檔案載入**：`datasets` 名稱遮蔽 → `bench_datasets.py`；CLAP 一律逐檔（`eval_metrics.score_clap`，batch 1）。

## 附錄 A：與 round-1 預登錄的差異（逐條）

| 項目 | round 1 | round 2 | 理由 |
|---|---|---|---|
| benchmark | MC522、PAM400、ME | ODAQ、AIME、SongEval（S800）、SingMOS-Pro、Music Arena（2,000）、MC522（段長） | round 1 沒有訊號劣化人類錨、歌唱、歌曲、TIE |
| 指標 | AES、MEva、MuQ-Eval A1/A3a、CMI-RM、PAM | 上列＋SongEval 5 維、SingMOS、MERT/MuQ-ref、CLAP、SongBench/HEAR（探索） | 教授第 5 項、round-1 §6 的未接項 |
| 閘門 | G1/G2/G3 | 同上＋G4 段長、G5 響度、G6 ODAQ | 回答新問題 |
| 變體 | 無 | 5 s／10 s／30 s／full；raw／−23 LUFS | 教授第 8 項、051／063／065、128 |
| 判定 | 通過／排除／未定 | 同上，但**不得改寫 round 1** | 避免事後改規則 |

## 附錄 B：一句話給教授

> 我們用五個公開、有人類評分的資料集（含 ODAQ 的 MUSHRA 訊號劣化、SingMOS 歌唱、AIME／Music Arena 成對偏好）把現有與新指標一次測完；所有門檻與變體在評分前寫死，會同時回答「5 秒 vs 10 秒」、「響度正規化」與「哪種劣化人類覺得更差」三個問題。這一輪不改寫上一輪的判定，只補證據。
