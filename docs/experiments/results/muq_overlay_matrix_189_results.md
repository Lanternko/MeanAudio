# 189 MuQ A1 疊加攻擊全矩陣：結果

2026-11-12 收線（p2 189）。預登錄：`docs/experiments/muq_overlay_matrix_189_20261111.md`；contract `muq_overlay_matrix_189_contract.json`；產出 `~/eval_output_nvme/muq_overlay_matrix_189/summary.json`。eval-only，無人類評分。

## 主判定（預登錄）

主端點：188 凍結配方 `osynth`（P3_sawlp_const）以 r18（比底小 18 LU）疊進九個 MusicCaps 底，`success_specific` 個數（判準 v3：A1 漲過 m 0.055 且 CE 降，並且 A1 與 CE 都要比 `sine1k`、`music` 兩個對照聲做得更多）。

**2/9（只有 ma_cfg0 s14、s16）→ `only_188_base`；MUSDB ΔA1 +0.004 [−0.010, +0.018]，不同號。**

失敗原因不在 A1：九個底 A1 全漲（+0.149 真實～+0.243 cfg3neg），A1 比 sine1k 多 +0.12 以上、比 music 多 +0.24，九個底 `a1_ok` 全過。失敗在 CE：CE 確實降（−0.07～−0.19），但多數底降得**不比 1 kHz 正弦多**（真實 CE_diff vs sine1k +0.019，CI 跨零），7/9 標為 `A1_specific_CE_nonspecific`。

## osynth 依音量

| r (LU) | success_specific／9 | 其餘 | MUSDB |
|---|---|---|---|
| 24 | 0 | A1 專一 CE 不專一 8、CE 沒降 1 | no_effect |
| 18 | **2** | A1 專一 CE 不專一 7 | no_effect |
| 12 | 4 | 5 | no_effect |
| 6 | 6 | 3 | no_effect |
| 0 | 5 | 4 | no_effect（ΔA1 +0.065、CE −1.96） |

越大聲 CE 越像「被這個聲音特別傷到」，但 r6 以上已經不是小聲疊加，MUSDB 也始終 no_effect。

## 其他配方（dev 200 筆在 r6 選出）

- `osynth2`（P2_saw_am0.24）：r18 8/9 專一成立、r6／r0 9/9；但它本身破壞大：PQ −0.24～−0.42、CE −0.39～−0.63、CLAP −0.002～−0.025；A3a 也漲（+0.18～+0.25）。真實底是 no_effect（A1 只 +0.075）。MUSDB ΔA1 **−0.071**（反向）。
- `osynth3`（P3_sine_am0.32）：A1 +0.09～+0.14、CE 幾乎不動 → 全部不專一。
- `recipe_replication_r18`：osynth2 8、osynth3 7（A1 部分），`single_recipe` = false：不同循環音配方都能把 A1 推高，但 CE 那半依配方而定。

## 音高專一性（r18）

osynth 對移調 +2 半音版 `osynth_tr2` 多 +0.14～+0.23 A1、對同頻譜噪音 `specnoise` 也勝，十個底（含 MUSDB +0.105）全過 → 188 的「得分綁在特定音高」在 1,000 筆規模複製。

## 副作用與獨立讀數（osynth r18）

- A3a 跟漲 +0.10～+0.20（約 A1 的 65–90%）。
- PAM −0.02～−0.04、CLAP −0.002～−0.015、CMI-RM(q) −0.05～−0.09：獨立評分器都略降，方向一致是「變差」。
- PQ：真實／MeanAudio −0.004～−0.08，MusicGen／MusicLDM 反升 +0.03／+0.05。
- FAD（osynth）：r18 真實 +0.96、MeanAudio cfg0 +0.54～+0.61、cfg3neg +0.21～+0.26；r0 +3～+5.7。
- MEva r18：真實 +0.96、cfg0 +0.54～+0.61（MEva 也被推高）。

## 結論

小聲疊進 188 的合成循環音，在九個 MusicCaps 底上**普遍、而且依音高專一地**推高 MuQ A1（+0.15～+0.24，約 0.3–0.4 S），PAM／CLAP／CMI-RM／FAD 同時變差 → A1 被騙這件事在跨系統的 MusicCaps 底上成立。但預登錄的「專一成立」需要 CE 降得比 1 kHz 正弦多，這一半只在 2/9 底成立，MUSDB 完全沒效果 → 依預登錄讀成 `only_188_base`。CE 這一半失敗與 CE 痛點挖掘的發現一致：疊進任何聲音 CE 都降，CE 很難區分「加了循環音」與「加了正弦」。

## 限制

- 無人類評分；「變差」只由 CE／PAM／CLAP／CMI-RM／FAD 判。
- MUSDB 是 44.1 kHz 分軌混音重採樣、只有 551 窗，與 MusicCaps 底不同分布；效果消失可能是底的差異，也可能 188 配方只對 10 秒片段有效，本實驗分不開。
- osynth2／osynth3 是在 dev 子集選出，屬事後選擇，只作複製參考，不進主判定。
