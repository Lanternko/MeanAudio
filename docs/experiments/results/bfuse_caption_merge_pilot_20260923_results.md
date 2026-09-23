# B 線：Qwen＋MF caption 合寫 pilot（2026-09-23，未過閘，不進訓練）

**來源**：`docs/meetings/2026-09-22_new_directions_cfg_encdec_noise.md` B 線（「合寫 / 讓 7B 重寫」）。進 GPU 訓練前的閘門：**合寫 caption 要比兩個來源各自都更貼音訊**，否則不花訓練預算。

**設定**：paired59k 同一批 clip（Qwen c2p0 slot0 vs paired59k MF arm 的 `mf_recaption`），seed 0 抽 2,000 列。
Qwen2.5-7B-Instruct（vLLM, greedy）按規則合寫：一致的事實寫一次、只出現在一方且不矛盾的保留、矛盾的兩邊都丟、刪使用情境與 hedge、≤45 字。0 列因長度截斷。對照加一欄 naive 串接（Qwen + MF）。
對齊度：laion_clap HTSAT-base，音訊＝**前 10 秒**（captioner 看到的視窗）、48 kHz、int16 round-trip、逐檔；文字走 CLAP 自身 77-token 截斷（＝訓練時 CLAP 路徑看到的）。R@k 為 2,000 池內 text→audio。**這是語料診斷，不是用 CLAP 篩訓練資料。**
腳本：`scripts/preprocess/{fuse_qwen_mf_caption_pilot.py,score_caption_audio_alignment.py}`；輸出 `~/eval_output_nvme/bfuse_pilot/`。

| 欄 | CLAP cos | Δ vs Qwen [95% CI] | R@1 | R@10 | T5 p50 | T5 >77 |
|---|---|---|---|---|---|---|
| Qwen | 0.2974 | — | 2.05% | 12.5% | 75 | 45.5% |
| MF (recaption) | 0.2718 | −0.0255 [−0.0303, −0.0209] | 1.75% | 9.5% | 65 | 0.0% |
| 串接 Qwen+MF | 0.3005 | +0.0031 [+0.0012, +0.0049] | 2.45% | 13.25% | 137 | 99.6% |
| **7B 合寫** | 0.2885 | **−0.0089 [−0.0126, −0.0050]** | 2.10% | 11.85% | 66 | 20.9% |

## 讀法

1. **合寫比 Qwen 單獨更不貼音訊**（CI 不跨零），閘門不過 → 不開訓練 arm。合寫確實把長度壓進窗口（>77 從 45.5% 降到 20.9%），但換來的是對齊度下降：MF 本身比 Qwen 低 0.026，混進 MF 的內容把平均拉低。
2. **串接的 +0.003 不能用**：CLAP 路徑截在 77 token ≈ Qwen 全文＋MF 開頭幾個字；T5 路徑 99.6% 被截斷，訓練時看到的幾乎就是 Qwen。增益小、而且機制是「Qwen 後面多幾個字」，不是融合。
3. 與既有證據一致：paired59k 訓練對照中 captioner 差異只在 CLAP（Qwen +0.0073）；這裡在語料層面 Qwen 也比 MF 更貼音訊，所以「讓 MF 補 Qwen」沒有方向上的理由。

## 限制（收線不等於證明）

- 只試了一個 merge prompt、一個 7B 模型；更大的模型或「以 Qwen 為主、只補 MF 獨有且可驗證細節」的 prompt 沒試。
- 對齊度只量 CLAP 文字路徑；T5 路徑（模型主要條件）沒有音訊對齊的直接量尺。
- CLAP 對齊是語料診斷的代理指標，不等於訓練後的 MusicCaps 表現。
- 重啟條件：出現一個在此量尺上 Δ vs Qwen 為正（CI 不跨零）且 T5 >77 比例不高於 Qwen 的合寫版本。
