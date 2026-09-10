# 跨 captioner rotation 線（mixcap_01m）

> 2026-09-09 建立。queue job `046_mixcap_01m_random_quarter.sh` / `047_mixcap_01m_random_full.sh`，
> contract 在 `docs/experiments/mixcap_01m_random_{quarter,full}_cfg0_contract.json`，
> caption pool 在 `docs/experiments/mixcap_01m_caption_pool.json`。

## 這條線要回答什麼

到目前為止每一條 rotation 線都只在**同一個 captioner 內部**輪替：013（slot0/1/3，full，
CFG0 CLAP 0.2221）和 012（slot0/1/2，quarter，0.2053）都是全 Qwen。034 又證明**輪替哪幾個
Qwen slot 五個指標全落在 seed 雜訊內** —— 也就是說，同一 clip 的三條 Qwen caption 近乎可互換。

所以真正沒被問過的是：**當三條 caption 裡有一條來自另一個 captioner（失效模式不同）時，
rotation 有沒有多買到東西？**

單獨比較時 MF 是輸的：MF 單獨 full 0.2078 < Qwen slot0 full 0.2149；paired59k 的
captioner-only delta 是 CLAP +0.0073 給 Qwen，四項 AES 全在雜訊內。所以樸素預測是**混入 MF
會讓 pool 變差或持平**。如果反而更好，那 rotation 買到的是 captioner 多樣性，不是 caption 數量。

## 設計

pool = **c2p0 slot0 + c2p0 slot1 + mf_dedup**（K=3）。

兩個規模都是**對全 Qwen control 的單一替換**：

| 規模 | arm | control | control CFG0 CLAP |
|---|---|---|---|
| quarter | slot0 / slot1 / **MF** | 012 = slot0 / slot1 / slot2 | 0.2053 |
| full | slot0 / slot1 / **MF** | 013 = slot0 / slot1 / slot3 | 0.2221 |

**為什麼是 K=3 不是 K=2**：control 是 K=3。K=2（slot0 + MF）會把「換 captioner」和
「pool 從 3 變 2」綁在一起，而且沒有任何 K=2 全 Qwen control，要能讀就得再跑一條，成本翻倍。

**為什麼第三格選 slot1 不是 slot2**：選 slot1 之後，arm 對 quarter 的 012 和 full 的 013
**兩個規模都只差一格**。選 slot2 的話對 full control 會同時動兩格。

**為什麼是 mf_dedup 不是 mf_fullcov**：mf_dedup 是贏了 040/042 決策、拿到 full 預算的語料
（full 0.2078），mf_fullcov 的 27,264 條重複 caption 已在其中重生。

**零磁碟**：position 0/1 是既有 013 stack 的 index 0/1，position 2 是 mf_dedup 單槽 overlay。
三者 encoder fingerprint 都是 `27e88fac…`、都覆蓋同一份 251,599 列 cache list 且順序相同，
所以 pool 在 load 時組裝（`text_npz_sources`），新增 0 bytes。專用 3-stack 要 225 GB，NVMe 只剩 63 GB。

recipe 照抄 c2p0：seed 14159265、batch 8、lr 1e-4、NoQ、no text attention mask、
`require_text_overlay=true`。quarter = S1 100k / S2 50k；full = S1 400k / S2 200k。

## 判讀規則（launch 前登記）

CFG0 / MusicCaps 5521 / MF25 / NoMask / seed 42 / full precision / `--no_q`。
CFG0 training-seed floor CLAP = 0.0042，**2× floor = 0.0084**；差距不到 2× 一律寫成平手。

**quarter（046）**，對 012 的 0.2053：

| quarter CFG0 CLAP | 判讀 |
|---|---|
| ≥ 0.2137 | 混合 pool 勝過全 Qwen pool → rotation 買到的是 captioner 多樣性 |
| 0.1969 – 0.2137 | 平手 → 儘管 MF 單獨較差，一條 MF caption 在 rotation 裡可無損取代一條 Qwen caption |
| < 0.1969 | 混入 MF 確實有害 → rotation 洗不白較弱的 captioner；047 自我中止 |

**full（047）**，對 013 的 0.2221：分界線同樣是 ±0.0084，即 ≥ 0.2305 / 0.2137–0.2305 / < 0.2137。

四項 AES 用同一條 2× floor 規則一起報；CLAP 之所以是 primary，只因為對照表是用它建的。

### early-kill 寫在 action 裡

queue 沒有 dependency 機制（`lib_scheduler.py` 純字典序）。所以 `SCALE=full` 時
`mixcap_01m_random_action.sh` 的 Step 0 自己去讀 046 的 CFG0 report，`clap_score < 0.1969`
就 exit 5，不開訓練。強制跑：`touch ~/exps_nvme/mixcap_01m/PROCEED_TO_FULL_ANYWAY`。
（037 就是敗在這裡：036 的數字依其 contract 應該取消 037，037 仍被 seat，最後手動殺在 it 27,062。）

## launch 前已驗證

`validate_composed_text_overlay.py`，2000 列 × 4 epoch：

- 三個 source 的 `clip_id` 在全部 2000 列都對上 TSV
- pool position 1 對照 `phase8_caption2p0_slot1_train.tsv`、position 2 對照 `mf_dedup_train.tsv`，
  embedding 逐位元相符
- rotation share 0.3392 / 0.3307 / 0.3300，failures **0**
- 另外確認 `k3_true_random_train.tsv`、`mf_dedup_train.tsv`、`phase8_caption2p0_slot1_train.tsv`
  三份 TSV 的 id 序列在全部 251,599 列完全相同

報告：`docs/experiments/mixcap_01m_composed_overlay_validation_20260909.json`。
訓練期還有 `require_text_overlay=true` 逐 row 再驗一次。

## caveat（不能寫掉的）

1. 兩個 control 的第三格不同（quarter 是 slot2、full 是 slot3）。034 讓「slot 身分是雜訊」
   成為合理假設，但那是**引用**不是這條 arm 內部量到的。
2. mf_dedup 的 caption 唯一率 0.9845，slot0 是 1.0000 —— position 2 帶的相異文字略少。
3. quarter 的 S1 100k = 3.18 epoch，rotation 只覆蓋到 2.19/3，是在 undertrained regime 量
   regulariser。所有 quarter rotation arm 共有此問題，內部仍是 budget-matched。

## Eval：兩個 cell

每個 arm 都取兩個數字（操作者 2026-09-09 追加）：

| cell | 協定 | 產生者 |
|---|---|---|
| **CFG0**（primary，gate 用） | MusicCaps 5521 / MF25 / cfg 0 / seed 42 / NoMask / `--no_q` | 既有 canonical harness `eval_musiccaps_mf25.sh` |
| **CFG3+neg**（secondary） | 同上但 cfg 3.0 + fidelity negative prompt | 新的 `scripts/eval/mc_mf25_cfg3neg_eval.sh` |

canonical harness 把 `cfg=0` 寫死且拒絕其他強度，所以每條 arm 的 CFG3+neg cell 一向是各自
action 裡的同一段 code。新腳本就是那一段抽出來，協定與產出 comparator 數字的
`mf_dedup_action.sh` Step 6 逐字相同（negative prompt 字串寫死在腳本裡，改了就作廢所有跨 arm 比較）。

**兩個全 Qwen control 都還沒有 CFG3+neg 數字** —— 板上的 rotation 數字全是 CFG0。所以
047 的 Step 1 會先補跑 012 quarter 和 013 full 兩個 control 的 CFG3+neg，否則新 arm 的
CFG3+neg cell 沒有對照。

047 的執行順序（全部在同一個 seat，不與訓練搶 GPU）：

1. control CFG3+neg ×2（012 quarter、013 full）
2. quarter arm CFG3+neg（046 只產了 CFG0 cell）
3. gate → full 訓練 → canonical CFG0
4. full arm CFG3+neg

eval 排在訓練**之前**是刻意的：就算 Step 3 的 gate 擋下 full，quarter 的兩個 cell 也已經完整。

CFG3+neg 的 seed floor 是 CLAP 0.0003 / CE 0.2960 / CU 0.1053 / PC 0.1884 / PQ 0.1416，
分界同樣取 control ± 2× floor。**gate 仍然只看 CFG0** —— 現有 rotation comparator 全部
是在那個協定下量的。

> CFG3+neg 的 code 放在 full wrapper 而不是 shared action：追加需求進來時 046 已經在跑，
> bash 是逐段讀 script 的（memory `reference_bash_script_buffered_reads.md`），改動執行中的
> action 不安全，而且兩個 wrapper 都 pin 了它的 digest。action 視同 immutable。

## 檔案

| 角色 | 路徑 |
|---|---|
| queue 進入點 | `gpu_queue/p2/pending/04{6,7}_mixcap_01m_random_{quarter,full}.sh` |
| contract | `docs/experiments/mixcap_01m_random_{quarter,full}_cfg0_contract.json` |
| caption pool | `docs/experiments/mixcap_01m_caption_pool.json` |
| per-scale wrapper | `scripts/training_pipelines/mixcap_01m_random_{quarter,full}.sh` |
| 共用 action | `scripts/training_pipelines/mixcap_01m_random_action.sh` |
| CFG3+neg eval | `scripts/eval/mc_mf25_cfg3neg_eval.sh` |
| pool 驗證報告 | `docs/experiments/mixcap_01m_composed_overlay_validation_20260909.json` |

---

## 結果

### 046 quarter — 完成，rc=0（2026-09-09）

CFG0 / MusicCaps 5521 / MF25 / NoMask / seed 42 / `--no_q`。訓練 150,000 it 全程 0 個 NaN。

| metric | mixcap_01m quarter | 012 control（全 Qwen） | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP | 0.1982 | 0.2053 | −0.0071 | 1.69× | tie |
| CE | 6.1791 | 6.3422 | −0.1631 | 1.21× | tie |
| CU | 6.7617 | 6.8654 | −0.1037 | 1.99× | tie |
| PC | 4.9864 | 5.0760 | −0.0896 | 1.62× | tie |
| PQ | 6.5921 | 6.6997 | −0.1076 | 2.06× | **LOSS** |

依 launch 前登記的規則：CLAP 0.1982 落在 tie 帶（0.1969–0.2137）的**下緣**，gate 通過，
但只多出 0.0013（0.31× seed floor）。047 因此被 seat。

**規則沒抓到的訊號**：五個指標**方向全部一致向下**，PQ 已跨過 2× floor。逐指標看是「四平一負」，
但五個獨立指標同號的機率本身就低，這比任一單項的 tie 判定更值得注意。

定位（quarter CFG0 CLAP）：

```
MF 單獨 0.1865  <  mixcap_01m 0.1982  <  slot2 0.2017 < slot0 0.2029 < slot1 0.2047 < 012 rotation 0.2053
```

混合 pool 明顯高於 MF 單獨，但**低於任何一條單獨的 Qwen slot**，也低於全 Qwen rotation。
這與「rotation 大致複製其成分的平均」一致 —— 換句話說，在 quarter 尺度上**沒有**看到
captioner 多樣性帶來額外增益，較弱的 captioner 也沒有被 rotation 洗白。

### CFG3+neg cell（047 Step 1/2 補跑，2026-09-10）

兩個全 Qwen rotation control 之前從來沒有 CFG3+neg 數字，這次一併補上。

quarter，mixcap_01m vs 012 control：

| metric | mixcap_01m | 012 control | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP | 0.2323 | 0.2337 | −0.0014 | 4.67× | **LOSS** |
| CE | 6.8971 | 7.0639 | −0.1668 | 0.56× | tie |
| CU | 7.5937 | 7.5816 | +0.0121 | 0.11× | tie |
| PC | 4.7422 | 4.9469 | −0.2047 | 1.09× | tie |
| PQ | 7.4753 | 7.4207 | +0.0546 | 0.39× | tie |

**與 CFG0 的圖像不同，不能當成同一件事的再確認**：

1. CLAP 差距從 −0.0071 縮到 **−0.0014**。兩者都判給 control，但在 negprompt 協定下混合 pool
   幾乎追平。
2. CFG0 的「五指標同號向下」**沒有重現** —— CU 與 PQ 在這裡轉正。
3. 混合 pool 在此協定下**勝過它的兩個單 captioner 成分**：0.2323 > slot0 0.2248、> mf_dedup 0.2233，
   只輸給全 Qwen rotation 的 0.2337。

CFG3+neg 的 CLAP seed floor 只有 0.0003（CFG0 是 0.0042），所以 4.67× 的絕對差距其實只有 0.0014。
正確說法是「這個差在 seed 之間可重現」，不是「這個差很大」。

### control 自身的協定相依性（副產品）

補跑出來的 control 數字暴露一件與本 arm 無關但值得記的事：**rotation 對單槽的優劣在兩個協定下反號**。

| | CFG0 | CFG3+neg |
|---|---|---|
| quarter：012 rotation vs slot0 | 0.2053 vs 0.2029（+0.0024） | 0.2337 vs 0.2248（**+0.0089**） |
| full：013 rotation vs slot0 | 0.2221 vs 0.2149（+0.0072，1.71× floor＝平手） | 0.2515 vs 0.2605（**−0.0090**，30× floor＝LOSS） |

full 尺度加上 negative prompt 之後，單槽 slot0 反而勝過 rotation。「per-epoch rotation 有沒有幫助」
因此是協定相依的，不能只憑 CFG0 表下結論。

| CFG3+neg 參照（quarter） | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| 012 rotation | 0.2337 | 7.0639 | 7.5816 | 4.9469 | 7.4207 |
| mixcap_01m | 0.2323 | 6.8971 | 7.5937 | 4.7422 | 7.4753 |
| c2p0 slot0 | 0.2248 | 6.6952 | 7.3871 | 4.6661 | 7.3101 |
| mf_dedup | 0.2233 | 6.8383 | 7.4619 | 4.8793 | 7.3294 |

| CFG3+neg 參照（full） | CLAP | CE | CU | PC | PQ |
|---|---|---|---|---|---|
| c2p0 slot0 | 0.2605 | 7.2114 | 7.6251 | 5.1059 | 7.5992 |
| 013 rotation | 0.2515 | 7.1205 | 7.6737 | 4.8476 | 7.6111 |
| mf_dedup | 0.2420 | 6.7534 | 7.3490 | 4.8045 | 7.2140 |

### 047 full — 完成，rc=0（2026-09-11）

訓練 600,000 it（S1 400k + S2 200k）全程 0 個 NaN。

**CFG0，vs 013 全 Qwen rotation（primary）：五項全平**

| metric | mixcap_01m full | 013 control | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP | 0.2187 | 0.2221 | −0.0034 | 0.81× | tie |
| CE | 6.3510 | 6.3893 | −0.0383 | 0.29× | tie |
| CU | 6.8687 | 6.8719 | −0.0032 | 0.06× | tie |
| PC | 5.1195 | 5.1883 | −0.0688 | 1.24× | tie |
| PQ | 6.6391 | 6.6513 | −0.0122 | 0.23× | tie |

對 MF 單獨（mf_dedup full）：CLAP +0.0109（2.60× **WIN**）、PC +0.1546（2.79× **WIN**），其餘平手。
對 slot0 單槽（0.2149）：+0.0038（0.90×，平手）。

**CFG3+neg，vs 013 control：同樣五項全平**

| metric | mixcap_01m full | 013 control | delta | ×floor | 判定 |
|---|---|---|---|---|---|
| CLAP | 0.2517 | 0.2515 | +0.0002 | 0.67× | tie |
| CE | 7.0024 | 7.1205 | −0.1181 | 0.40× | tie |
| CU | 7.5761 | 7.6737 | −0.0976 | 0.93× | tie |
| PC | 4.8943 | 4.8476 | +0.0467 | 0.25× | tie |
| PQ | 7.4641 | 7.6111 | −0.1470 | 1.04× | tie |

對 mf_dedup：CLAP +0.0097（32× **WIN**）、CU +0.2271（2.16× **WIN**）。
對 slot0：CLAP −0.0088（29× **LOSS**）—— 與 013 control 對 slot0 的 −0.0090 幾乎相同，
即混合 pool **完整繼承了 rotation 在 negprompt 協定下輸給單槽的那個行為**，
見 [rotation vs 單槽的協定反號](#control-自身的協定相依性副產品)。

### 結論

登記的 tie 帶對應的判讀成立：**儘管 MF 單獨比 Qwen 單獨差，一條 MF caption 在 rotation 裡
可以無損取代一條 Qwen caption。** 兩個協定、十個指標，沒有任何一項判給 control。
同時 arm 在兩個協定下都明確勝過 MF 單獨。

原始假說（「贏才代表 rotation 買到 captioner 多樣性」）**沒有成立** —— 沒有增益，是等價。

**規模改變了結論，quarter 會誤導：**

| | quarter | full |
|---|---|---|
| CLAP 對 control | −0.0071（1.69×） | −0.0034（0.81×） |
| 最大單項偏離 | PQ 2.06×（LOSS） | PC 1.24×（全平） |
| 對 slot0 單槽 | −0.0047（低於每一條 Qwen slot） | +0.0038（高於 slot0） |

quarter 的落後有相當部分來自 undertrained：rotation 只覆蓋 2.19/3，full 才跑滿 12.72 epoch。
若依 quarter 的「五指標同號向下」下結論，會得到偏負的錯誤答案。**這是 rotation 類 arm 不該只用
quarter 判定的直接證據。**
