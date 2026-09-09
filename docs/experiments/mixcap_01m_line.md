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

## 檔案

| 角色 | 路徑 |
|---|---|
| queue 進入點 | `gpu_queue/p2/pending/04{6,7}_mixcap_01m_random_{quarter,full}.sh` |
| contract | `docs/experiments/mixcap_01m_random_{quarter,full}_cfg0_contract.json` |
| caption pool | `docs/experiments/mixcap_01m_caption_pool.json` |
| per-scale wrapper | `scripts/training_pipelines/mixcap_01m_random_{quarter,full}.sh` |
| 共用 action | `scripts/training_pipelines/mixcap_01m_random_action.sh` |
| pool 驗證報告 | `docs/experiments/mixcap_01m_composed_overlay_validation_20260909.json` |

---

## 結果

（046 於 2026-09-09 06:35 UTC seat，待填）
