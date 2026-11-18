# 198 CE 疊加內插階梯：疊進去的東西，CE 是不是都落在中間？（預登錄，2026-11-19）

狀態：設計登錄；p2 198，eval-only，只用 AES 評分，約 2–3 h，可續跑。上游：`docs/experiments/results/ce_push_196_results.md`、`ce_weakness_195_results.md`。無人類評分，結果只描述評分器方向。

## 0. 要回答的問題

使用者問：「測試不同的東西疊加是不是都落在中間？是正中間嗎？還是會有其他因素。高分的音量單調上升，會讓 CE 也單調上升嗎？」

196 的觀察：疊加後的 CE 落在「底的 CE」與「疊入聲單獨的 CE」之間；同響度時位置 p（從底走向疊入聲的比例）約 0.33–0.47；疊入聲大 18 LU 時 p 只有 0.33–0.76；逐段單調的比例只有 42%。但 196 只用一種疊入聲（e00 合成、鋼琴琶音）。198 換 8 種疊入聲、把音量範圍拉到 ±24 LU，看 p 是由什麼決定的。

## 1. 底與刺激

- 底：196 的 9 個 MusicCaps 底（`C.MC_BASES`：真實、MeanAudio CFG0 × 3、CFG3＋neg × 3、MusicGen、MusicLDM），id 用 196 的 dev 200（`~/eval_output_nvme/ce_push_196/split.json` 的 `dev_mc`，seed 20261113）。16 kHz、10 s、底先鎖 −30 LUFS。
- 疊入聲 8 種（疊入聲先各自鎖 −30 LUFS 再按 r_rel 調增益）：

| 名稱 | 內容 | 有音高 | 隨底變 |
|---|---|---|---|
| `e00` | 195 黑箱最佳合成聲，主音與大小調換成底的調性（196 的 `syn`） | 是 | 是 |
| `pnoarp` | 底調性主和弦鋼琴琶音（generaluser，196 的 `pno`） | 是 | 是 |
| `pnoblock` | 同樣四個音（根音、三度、五度、八度）同時按下的柱狀和弦，每秒一次，同音色 | 是 | 是 |
| `sine1k` | 1 kHz 正弦 | 是 | 否 |
| `white` | 白噪音（固定 seed） | 否 | 否 |
| `pink` | 粉紅噪音（固定 seed） | 否 | 否 |
| `music` | 另一段真實 MusicCaps 錄音（固定 seed 的循環錯位，每個 id 對應另一個 id） | 是 | 是（逐 id） |
| `drums` | 每 0.5 秒大鼓、0.5＋n 秒小鼓、每 0.25 秒閉合镲，無音高 | 否 | 否 |

- r_rel ∈ {−24, −18, −12, −6, 0, +6, +12, +18, +24} LU（正＝疊入聲比底大聲），另有 `base`（原底）與 `only`（疊入聲單獨，−30 LUFS、10 s、16 kHz）。
- 混合：`mix = base + overlay × 10^((−30 + r_rel − lufs_overlay)/20)`（float64，不裁切疊入聲峰值），再 `C.lock`（−30 LUFS）；AES 走 196 同一條呼叫（16 kHz）。
- 共 9 底 ×（1 base＋8 疊入聲 ×（9 r_rel＋1 only））× 200 ≈ 145,800 次 AES 評分。

## 2. 重現閘門

e00 在 r_rel ∈ {−12, −6, 0, +6, +12} 必須重現 196 的 `dev_<底>_{syn12, syn6, syn0, synm6, synm12}.tsv`：逐段 |ΔCE| 中位數 ≤ 2e-3、最大 ≤ 0.02，否則 job 失敗（AES 對 batch 組成有約 1e-3 的漂移，故容差 2e-3）。每個底在 e00 格評完後立即檢查。`base`／`only` 與 +18 只當資訊列：196 在 +18 時把疊入聲峰值裁到 0.999（等於實際疊入聲比名義小聲），198 不裁切，所以 +18 不可能逐段一致，不進閘門（smoke 兩個底 +18 的 |ΔCE| 中位數 0.02–0.08，與預期相符）。

## 3. 指標與判讀規則（先於結果登錄）

對每個底、疊入聲、r_rel：Δm = CE(mix) − CE(base)、Δo = CE(only) − CE(base)；內插位置 p = mean(Δm)/mean(Δo)（逐段 bootstrap 2000 次，seed 198；9 底平均為 pooled p，CI 為 9 底 bootstrap 平均的 2.5–97.5 百分位）。

- **「正中間」**：pooled p 的 bootstrap 95% CI 含 0.5。
- **「按能量加權」**：|p − w| < 0.1，w = 10^(r/10)/(1 + 10^(r/10))（疊入聲占混合能量的比例）。
- **跑出區間**：混合 CE 高於 max(底, only) 或低於 min 的逐段比例（`frac_above_max`、`frac_below_min`）。
- **單調**：CE 隨 r_rel 的曲線相鄰差 ≥ −0.119（m_CE ＝ 0.10 × S_CE_lvl30）；方向依 only 比底高（期望上升）或低（期望下降）而定；另報逐段單調比例（每段 9 點）。
- PQ 同樣計算（容差 0.0914）。
- 若 |Δo| < 1e-3（如真實底 × music 疊入聲，循環錯位後平均與底相同），p 無定義，記為 null。
- 解釋因素：r_rel＝0 的 72 個（底 × 疊入聲）格，p 對下列變數的 Spearman：Δo（only 與底的差）、底 CE、疊入聲與底的 log 頻譜重心比、是否有音高；另報只取 |Δo| ≥ m 的子集（避免分母近零）。

不做的事：不排名疊入聲、不宣稱 AES 好壞，沒有人類評分；結論只寫「CE 在這些疊加下怎麼走」。

## 4. 實作與執行

- 驅動：`scripts/eval/ce_overlay_interp_198.py`（匯入 195／196／188／189／190 的 helper，皆不修改）；輕量 AES-only scorer（`Q.Y` 的 `aes4`），不載入 MuQ／CLAP／PAM。
- 輸出：`~/eval_output_nvme/ce_overlay_interp_198/`：`per_clip/<底>_<疊入聲>_<格>.tsv`、`<底>_base.tsv`、`gate.json`、`summary.json`、`summary_table.md`、`projection.json`、`gpu_peak.json`、`keys.json`、`features.json`。逐格 TSV 續跑時跳過。
- 摘要：`scripts/analysis/ce_overlay_interp_198_summary.py`（只讀 summary.json 出表）。
- CUDA OOM 時重試（host 只在 seat 時查顯存）。
- queue：`198_ce_overlay_interp.sh`，contract `docs/experiments/ce_overlay_interp_198_contract.json`；估 2–3 h，hard cap 6 h。

## 5. 偏離與限制

- D1：疊入聲增益不裁峰值（196 裁 0.999）；+18 與 +24 的 drums 峰值很大，但混合後仍鎖 −30 LUFS，只在 `lock` 內被 0.999 上限處理，若觸發由 `C.lock` 的規則為準。
- D2：`music` 疊入聲取自 `real` 底的同一批 id，故 `real × music` 的 Δo ≡ 0（循環錯位），該格 p 記 null。
- D3：`base` 重評而非讀 196（避免 batch 組成漂移混入）。
