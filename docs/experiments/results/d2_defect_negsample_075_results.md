# 075 收線 — D2 雜訊負樣本：教會模型缺陷的名字，反而**縮小**了 negprompt 增益

2026-09-23。預註冊：`../d2_defect_negsample_075_20260923.md`。三顆 quarter checkpoint，同訓練 seed 14159265：

- **defectlab**：066 語料 ＋ 25k 劣化列，caption = 缺陷句前綴 ＋ 原文
- **defectunlab**：同一批劣化音訊，caption = 原文
- **control066**：`phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265`

評估：MusicCaps 5521 / MF25 / seed 42 / fp32 / NoMask，CFG0 與 CFG3+fidelity8，CLAP batch 1。
響度閘門觸發（CFG3+neg 的 LUFS：lab −18.75、unlab −20.57、control −16.32），所有 E2/E3 數字同時報
**−30 LUFS 響度對齊**重評（`scripts/eval/level_match_rescore.py`；每格 peak_capped ≤ 16、無限響度 ≤ 3 clip）。

腳本：`scripts/eval/d2_075_e2.py`（E2/E3，逐 prompt 配對）、`scripts/eval/d2_075_signature.py`（E1）。
（E2/E3 初稿由平行 session 5fd55e97 於 E1 出來前寫成，本檔合併其響度行為與隔離觀察。）
原始表：`~/eval_output_nvme/d2_075_e2_{raw,lvl30}.md`、`~/eval_output_nvme/d2_075_d1probe/signature.md`。

> CI 是逐 prompt bootstrap（5521 對），**只反映 prompt 抽樣，不含訓練 seed 變異**。判定一律用預註冊的
> 2× 3-seed 全距門檻（ΔG 0.19、CFG0 CLAP 0.004、PQ 0.155）。

## 三句話

1. **E2 反向不過**：點名缺陷讓 fidelity8 的 PQ 增益**縮小 0.37**（響度對齊後；反方向、約門檻 0.19 的 1.9 倍、3-seed 全距的 3.8 倍），
   CLAP 增益也縮 0.014。不點名的劣化音訊則讓增益略增（+0.16，對齊後低於門檻）。
2. **E1 只在雜訊族部分通過**：lab 臂對「static／hiss」產生真的高平坦度音訊（+0.25，真雜訊約 0.56），
   削波邊緣、低通沒有；「white noise」的改善來自看過雜訊音訊，不是標籤。
3. **E3：lab 通過、unlab 邊界**：lab 的乾淨生成沒被污染（CFG0 各軸落在 control 3-seed 範圍內，打平而非改善）；unlab 的 CFG0 CLAP
   原始 −0.0037（判定欄，界內貼線）、響度對齊後 −0.0052（超界 0.004）。標籤把缺陷音訊隔離在缺陷詞上。

## E2 — negprompt 增益（主端點）

`G = metric(CFG3+neg) − metric(CFG0)`，同 checkpoint 逐 clip。

| | G_lab | G_unlab | G_control | lab−unlab | lab−control | unlab−control |
|---|---:|---:|---:|---|---|---|
| PQ（原始） | 0.634 | 1.085 | 0.828 | **−0.451** [−0.472, −0.430] | −0.194 [−0.216, −0.171] | +0.257 [+0.235, +0.278] |
| PQ（−30 LUFS） | 0.624 | 0.989 | 0.828 | **−0.365** [−0.385, −0.345] | −0.204 [−0.226, −0.182] | +0.161 [+0.140, +0.183] |
| CE（−30 LUFS） | 0.629 | 1.010 | 0.914 | −0.381 | −0.285 | +0.097 |
| CU（−30 LUFS） | 0.559 | 0.906 | 0.801 | −0.347 | −0.242 | +0.105 |
| CLAP（−30 LUFS） | 0.0287 | 0.0428 | 0.0402 | −0.0141 | −0.0115 | +0.0026 |

- **響度不是解釋**：對齊後 lab−unlab 從 −0.451 收到 −0.365，仍是門檻的 ~1.9 倍（反方向）。
- 對照：control 的 3-seed 增益全距只有 0.0955；lab−unlab 的差是它的 3.8 倍。
- **兩個處理臂都改變了 negprompt 的響度行為**：CFG0→CFG3+neg 的 LUFS 變化 control **+2.67 LU**（變大聲）、
  lab **+0.00**、unlab **−1.92**（變小聲，crest 6.28→7.34）。「只看過壞音訊」本身就動到了 negprompt 的作用方式，不只標籤。
- unlab−control 原始 +0.257 過門檻、對齊後 +0.161 不過 → 「看過壞音訊讓增益變大」**不可宣稱**，
  一部分是 unlab 在 CFG3+neg 較小聲（control 自身 3 個 seed 的 CFG3+neg LUFS 就跨 −16.3～−20.1，是 seed 不穩定的量）。

### CFG3+neg 絕對分數（−30 LUFS 對齊）

| | CLAP | PQ | CE | CU | PC |
|---|---:|---:|---:|---:|---:|
| lab | 0.2055 | 7.308 | 6.569 | 7.363 | 4.615 |
| unlab | 0.2153 | 7.566 | 6.787 | 7.616 | 4.676 |
| control | 0.2179 | 7.476 | 6.764 | 7.545 | 4.726 |

lab 在 negprompt 格是三者最差（PQ −0.17、CLAP −0.012 vs control）。

## E1 — 操弄檢查：有沒有文字可達的缺陷方向（D1 probe，held-out 字串，波形簽名）

效果 = 該 prompt 組平均 − 同 checkpoint 的 `rock` 乾淨 stem 平均（n=128/組）。cfg3（null-neg）：

| prompt | 簽名（預期） | lab | unlab | control | 判讀 |
|---|---|---:|---:|---:|---|
| static noise and tape hiss, no music | flatness ↑ | **+0.252** | +0.029 | −0.018 | ✅ 標籤特有，lab−unlab [+0.212, +0.235] |
| full 缺陷串 ＋ rock | flatness ↑ | **+0.022** | +0.004 | +0.011 | ✅ ≥2× 兩者 |
| harsh digital distortion and clipping, no music | crest ↓ | −0.68 | −0.20 | +0.73 | ⚠ 方向對、vs unlab CI 跨零 [−1.01, +0.04] |
| white noise | flatness ↑ | +0.025 | +0.030 | −0.017 | ✗ lab≈unlab → 來自音訊不是標籤 |
| rock ＋ noisy/hiss | flatness ↑ | +0.020 | +0.011 | +0.034 | ✗ 不如 control |
| rock ＋ distortion/clipping | crest ↓ | −0.09 | −0.37 | −0.56 | ✗ 反而最弱 |
| rock ＋ muffled | centroid ↓ | −103 Hz | −153 | +200 | ✗ 不如 unlab |
| muffled, no music | centroid ↓ | +84 Hz | −125 | −303 | ✗ 反向 |

cfg0 同向（static +0.186 vs +0.020 / −0.010）。

- 按預註冊判準（≥2 軸、方向正確、≥2× unlab 與 control）**形式上過**（兩軸），但兩軸**都是雜訊族**、
  且都用 flatness —— 實質上只建立了「雜訊」一個方向。lab 的 static 平坦度 0.25 約是真雜訊（0.56）的 45%，
  是 D1 以來第一次有 prompt 讓模型產出**真的像雜訊**的音訊。
- 預註冊列的操弄檢查字串**與訓練缺陷句有詞彙重疊**（static、hiss）；「held-out」只到字串層，不到詞彙層。
- **靜音逃逸**：cfg0 純缺陷 prompt 的靜音比例兩個處理臂都下降（white 16.4% → 7.8/9.4%、clip 6.2% → 0/0.8%），
  lab≈unlab → 是看過劣化音訊的效果。cfg3 white noise 三臂都 16–20%，沒變。

## E3 — 非劣性（CFG0）

| | lab − control | unlab − control |
|---|---|---|
| CLAP（原始） | +0.0009 | −0.0037（界 −0.004 內） |
| CLAP（−30 LUFS） | −0.0008 | **−0.0052**（略超界） |
| PQ（原始／對齊） | +0.024 / +0.036 | −0.077 / −0.072（界內） |
| silent_n（control 52） | 29 | 31 |

lab 通過。unlab 有兩種讀法，並列報：CFG0 三臂 LUFS 差 < 0.5 LU（−18.66／−18.74／−19.00），**響度閘門未觸發 →
預註冊的判定欄是原始值，−0.0037 在界內（貼線）**；但響度對齊後 −0.0052 超界。寫成「邊界」，不寫成乾淨通過。

**標籤把缺陷音訊隔離到缺陷詞上**：CFG0 下 lab − unlab PQ +0.108、CE +0.164、CLAP +0.0044（對齊後），
方向與「沒標籤的壞音訊滲進正常 caption 的條件分布」一致；lab 的正常 caption 保持乾淨（甚至略好於 control）。

## 判讀

對到預註冊判定表的「**E1 過、E2 不過** → 模型學到了缺陷方向，但 fidelity8 的增益不靠它 → 031『領域詞彙』解讀更強」，
而且比該列更強：**不是「不靠它」，是「有了它反而變差」**。

- 031 的解釋（增益來自領域詞彙、不是缺陷極性）**得到支持**（supports，非 proves）：一旦負向槽裡的
  詞彙有了對應的訓練方向，fidelity8 的效果**下降** 0.37 PQ。
- 標籤確實有作用（lab ≠ unlab，CFG0 與 G 兩處都分得開），只是對 fidelity8 negprompt 是負作用。
- 最直接的機制假說（**未驗證**）：fidelity8 的詞原本指向一個**籠統的「非典型／差」方向**，這才是它能大幅推高 PQ 的原因；
  lab 讓這些詞綁到**具體**的程式化缺陷，負向分支變窄，推開的東西變少。幾何上的預測：control 裡 fidelity8 沒有被訓練過的意義，negative 分支 B 輸出的是一個
  離條件分支 A 很遠的「領域平均」；lab 裡 B 變成「同一段音樂的劣化版」，離 A 更近 → `A−B` 變小／變窄 →
  外插推力變小。可用 `scripts/eval/run_prenorm_branch_norm_diag.py` 量兩臂的 `cos(A,B)` 與 `‖A−B‖` 檢定。
- **對 D 線的實務結論**：用「壞音訊＋點名」做負樣本訓練，在我們的 negprompt 協定下是**淨負**。
  若要保留缺陷方向（E1 的雜訊族），代價是 negprompt 格 PQ −0.17 / CLAP −0.012（vs control）。

## 限制

1. 單一訓練 seed；lab−unlab 的 E2 差是 3-seed 全距的 3.8×，E1／E3 的小效果不可讀成穩定。
2. 25k 列、quarter budget；劑量–反應沒量（E1 只有雜訊族成形，可能是劑量不足而非不可學）。
3. 只測 fidelity8 一條負向 prompt；「缺陷方向讓**特定缺陷**的 negprompt 更準」沒測（例如 lab 臂用
   `static, hiss` 當負向 prompt 是否壓得掉雜訊）。
4. E4（reversed 極性）未跑：E1 只在雜訊族成立，而 E2 已反向，極性測試失去原本的問題意識。

## 後續（未排程）

- **P0 機制 probe**：兩臂 × fidelity8 cfg3 的分支幾何（`cos(A,B)`、`‖A−B‖/‖A‖`），8 clip × 25 步，幾分鐘。
- 若要延續 D 線：**特定缺陷負向 prompt** 在 lab 臂是否有效（`static, hiss` → 雜訊子集的 flatness／PQ），
  這是 E1 已建立的唯一方向，也是這個臂唯一可能贏的用法。
