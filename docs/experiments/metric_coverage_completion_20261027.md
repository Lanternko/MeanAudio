# 指標 × 介入覆蓋補完計畫（2026-10-27 預登錄）

> 目的：把 `wide_fail_list_full_20261022` 與 `open-slide/slides/wide-standard-scorecard` 上還空著的格子補滿。**本檔只設計，不排 queue、不跑 GPU、不改既有 contract／script／doc。**
>
> 這份文件是預登錄（prereg）：所有判準、抽樣量、S 的定義、out-of-domain 標記方式在跑之前固定；跑完只看結果，不改門檻。新模型的 S（每個指標的尺度）由 job 162 量測後寫入 `s_values_<model>.json`，本檔先標「待補」。

---

## 0. 先分清楚兩個「MuQ」與兩個「PAM／CLAP」

這是這份文件最容易讀錯的地方，先把名字釘死：

| 名稱 | 本檔寫法 | 這是什麼 | 目前介入覆蓋 |
|---|---|---|---|
| **MuQ-Eval A1** | `MuQ-Eval A1` | `zhudi2825/MuQ-Eval-A1` 的**訓練過品質頭**，輸出 MI／TA。全新模型，六個協定都沒跑過。 | 0／6 |
| **MuQ-Eval A3a** | `MuQ-Eval A3a` | `zhudi2825/MuQ-Eval` 的 A3a 訓練頭，輸出 MI／TA。全新模型，六個協定都沒跑過。 | 0／6 |
| **MuQ-embed cosine** | `MuQ-embed-ref` | 133 的 **reference-based** 讀數：MuQ-large embedding 的 cosine，`d = cosim(case, clean base) − 1`。**不是** MuQ-Eval A1／A3a。deck 上那列寫「MuQ」的就是這個。 | 126b／126c 已有；126a／134／rec／floor 缺 |
| **PAM** | `PAM` | `soham97/PAM` 的 zero-shot「clear and clean」機率，骨幹是 MS-CLAP + GPT-2。**不是** CLAP alignment。 | 0／6 |
| **CLAP** | `CLAP` | laion_clap 的 text-audio cosine；既有指標。 | 6／6 |

> 規則：`MuQ-Eval A1`／`A3a` 與 `MuQ-embed-ref` 是**不同列**，不可合併、不可互相代替。deck 的「MuQ」欄在補完後仍只代表 `MuQ-embed-ref`；A1／A3a 另外新增兩列。

---

## 1. 覆蓋矩陣（改動前 → 改動後）

### 1.1 符號

| 符號 | 意義 |
|---|---|
| `●` | 已完成：該協定的刺激與分數都在，判定也已做（或可直接由現有 TSV 算） |
| `◐` | 部分：**分數已存在**，只缺 rec／floor 的分析（不需重跑模型） |
| `○` | 缺：沒有分數 |
| `—` | 不適用（N/A），理由寫在該列註腳 |

「改動後」的目標是：除了 2 個 N/A，其餘 130 格全部 `●`。新跑的原則是**預設全跑**，即使 out-of-domain（例如 SingMOS 在器樂、PAM 在分軌）也跑，結果標 OOD，不因 OOD 跳過。

### 1.2 矩陣

欄位：`126a`＝訊號層 53 變體 × 5 TTM 系統；`126b`＝MIDI 音符層＋真人錄音＋底線；`rec`＝126b 的真人錄音配對；`floor`＝126b 的 0／1 波形底線；`126c`＝MUSDB18-HQ 分軌編輯；`134`＝速度／音高延伸。

| # | 指標（維度） | 126a | 126b | rec | floor | 126c | 134 |
|---|---|---|---|---|---|---|---|
| 1 | AES PQ | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 2 | AES CE | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 3 | AES CU | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 4 | AES PC | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 5 | **PAM** | ○→● | ○→● | ○→● | ○→● | ○→● | ○→● |
| 6 | MEva | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 7 | **MuQ-Eval A1（MI／TA）** | ○→● | ○→● | ○→● | ○→● | ○→● | ○→● |
| 8 | **MuQ-Eval A3a（MI／TA）** | ○→● | ○→● | ○→● | ○→● | ○→● | ○→● |
| 9 | **CMI-RM（quality／alignment）** | ○→● | ○→● | ○→● | ○→● | ○→● | ○→● |
| 10 | SongEval `se_coh` | ●→● | ●→● | ◐→● | ◐→● | ●→● | ○→● |
| 11 | SongEval `se_mem` | ●→● | ●→● | ◐→● | ◐→● | ●→● | ○→● |
| 12 | SongEval `se_nat` | ●→● | ●→● | ◐→● | ◐→● | ●→● | ○→● |
| 13 | SongEval `se_cla` | ●→● | ●→● | ◐→● | ◐→● | ●→● | ○→● |
| 14 | SongEval `se_mus` | ●→● | ●→● | ◐→● | ◐→● | ●→● | ○→● |
| 15 | **SingMOS predictor** | ○→● | ○→● | ○→● | ○→● | ○→● | ○→● |
| 16 | **MERT-ref**（cosine） | ●→● | ●→● | ◐→● | —→— | ●→● | ○→● |
| 17 | **MuQ-embed-ref**（cosine） | ○→● | ●→● | ◐→● | —→— | ●→● | ○→● |
| 18 | CLAP | ●→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 19 | MIR `pulse_clarity` | ○→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 20 | MIR `ibi_cv` | ○→● | ●→● | ●→● | ●→● | ●→● | ●→● |
| 21 | MIR `key_cnn_conf` | ○→● | ●→● | ●→● | ●→● | ●→● | ○→● |
| 22 | MIR `chroma_entropy` | ○→● | ●→● | ●→● | ●→● | ●→● | ○→● |

> 註：A1／A3a／CMI 在矩陣中各佔一列（各 6 格），但判定時 MI／TA、quality／alignment 分開報，不平均。SongEval 五維與 MIR 四項各佔一列，所以是 22 列 × 6 欄＝132 格。
>
> MIR 的 134：deck 上 `pulse_clarity` 與 `ibi_cv` 已有數值（65／38），`key_cnn_conf`／`chroma_entropy` 是 null。本計畫只補 key／chroma 的 134；pulse／ibi 不重跑。若為了同一版 `mir_metrics.py` 的一致性而要重算四項，job 176 已包含 134 的 49,587 clip，成本 +1.5 h（推估）。

### 1.3 覆蓋計數（以 22 列 × 6 欄計）

| 狀態 | 改動前 | 改動後 |
|---|---:|---:|
| `●` 已完成 | **74** | **130** |
| `◐` 只缺分析 | **12** | **0** |
| `○` 缺 | **44** | **0** |
| `—` N/A | **2** | **2** |
| 合計 | 132 | 132 |

改動前的 74／12／44／2 對應：AES 4 列＝24●、MEva 6●、CLAP 6●、SongEval 15●＋10◐＋5○、MERT 3●＋1◐＋1○＋1—、MuQ-embed 2●＋1◐＋2○＋1—、MIR pulse 5●＋1○、ibi 5●＋1○、key 4●＋2○、chroma 4●＋2○、PAM／A1／A3a／CMI／SingMOS 共 30○。

### 1.4 N/A 與 OOD 的理由

| 格 | 標記 | 理由 |
|---|---|---|
| MERT-ref × floor | `—` | reference-based 的 `d = cosim(case, base) − 1` 需要 base；floor 是「0／1 波形／正弦／掃頻」底線，沒有可配對的 base。若硬把 floor 跟自己比，cosim 恆為 1、d 恆為 0，會製造假平手。**不做這個比較**，也不補任何替代分數。 |
| MuQ-embed-ref × floor | `—` | 同上。 |
| 其餘 floor 格 | `●`＋`OOD-synthetic` | floor 不是音樂；所有模型都跑，結果只做描述性百分位，不進模型好壞判定。 |
| SongEval × 126a／126b MIDI／134 | `●`＋`OOD-10s-instrumental` | SongEval 在整首歌上訓練；126a／126b MIDI／134 是 10 秒、多為器樂或鋼琴。照跑，標 OOD。 |
| SingMOS × 126a／126b MIDI／126c／134／floor | `●`＋`OOD-instrumental` | SingMOS 是歌唱 MOS；只有 126b 的 `poor_sing` rec 是 in-domain。其餘照跑，標 OOD，**不用 OOD 的 fail 來否決模型**。 |
| PAM × 126c | `●`＋`OOD-stems` | PAM 訓練在混音；126c 是分軌編輯。照跑，標 OOD。 |
| PAM × 126b MIDI、A1／A3a／CMI × 126b MIDI | `●`＋`OOD-MIDI` | 合成鋼琴／SoundFont，不是自然音樂分布；照跑，標 OOD。 |
| 134 的速度 ≤0.80／≥1.25、音高 ≥±5 | `●`＋`OOD-extreme` | 極端延伸；照跑，標 OOD。tie 帶（0.95–1.05、±1 半音）不標 OOD。 |

---

## 2. 模型環境、checkpoint、輸入契約、現有腳本

### 2.1 環境

| 模型／指標 | Python 環境 | 備註 |
|---|---|---|
| AES PQ／CE／CU／PC | `~/venvs/dac` | 既有 Audiobox Aesthetics；不新增 pin |
| MEva | `runtime/meva_20261003/venv/bin/python` | pinned f03 adapter；`scripts/eval/meva_runtime.py` |
| MuQ-Eval A1／A3a、CMI-RM、PAM、SongEval、MERT、MuQ-embed、CLAP、SingMOS | `~/venvs/dac` | 新模型都走同一個 venv；HF cache 在 `/mnt/seagate/newmodels_20261020/hf` |
| MIR 四項 | `~/venvs/madmom`（py3.9）＋`~/venvs/mir`（librosa 1.0） | CPU；`scripts/eval/mir_metrics.py` |

### 2.2 Pin、輸入契約、輸出

| 模型 | Pin（weights／code） | 輸入取樣率與長度 | 輸出維度 | 現有入口 |
|---|---|---|---|---|
| **MuQ-Eval A1** | `zhudi2825/MuQ-Eval-A1` @ `b082e238827df51d99ac66bdd1b6f0d2406bc90e`；code `MuQ-Eval` @ `60a88f8ac0909ca1fd1a78af3660f1fc376977a1` | 24 kHz；`AudioProcessor(24000, 240000)`，中心取 10 s、不足零補 | MI、TA | `scripts/eval/newmetrics_muqeval.py` |
| **MuQ-Eval A3a** | `zhudi2825/MuQ-Eval` @ `cc0cb22ea4026f7d3fb1bae384927b870a24319f`；code 同上 | 同上 | MI、TA | 同上 |
| **CMI-RM** | `HaiwenXia/CMI-RM` @ `704f2dc3a11a1d867cc9a683a49c17d9cc7d50e4`；code `CMI-RewardBench` @ `9235ef5c34106b63669958adfaa2eab61bd0e2cd` | 24 kHz；≤30 s，30 s chunk；fp32；prompt mode `empty`／`caption` | quality、alignment | `scripts/eval/newmetrics_cmirm.py` |
| **PAM** | `soham97/PAM` @ `ead6e7f9d907268e30c73b29526f50523a34b4aa`；MS-CLAP `c47d441165daa21986ead0850660917636a81775`；GPT-2 `607a30d` | 44.1 kHz；<7 s tile-repeat；≥7 s 每 7 s 一段＋最後 7 s | P(clear and clean) | `scripts/eval/newmetrics_pam.py` |
| **SingMOS predictor** | `South-Twilight/SingMOS@v1.1.2`，local clone commit `959691e5ca9465f774b2533f44a609536955d3ed` | 16 kHz；整段 clip | MOS | `scripts/eval/bench_r2/score_r2.py:make_singmos` |
| **SongEval** | repo `848fb2ff3a2a9d64dcb20d46f07238c28abd7add`；ckpt `0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd`；MuQ `0562a57814f6f8bbd9fdea0a25921a2fce1a841a` | 24 kHz；30 s 中心裁切／零補；batch 16 | `se_coh`／`se_mem`／`se_nat`／`se_cla`／`se_mus` | `scripts/eval/metric_win_tie_126d.py`；backend `score_r2.py:SongEval` |
| **MERT-ref** | `m-a-p/MERT-v1-95M` @ `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` | 24 kHz；30 s chunk，尾端 <5 s 丟棄；batch 16 | 1024-d embedding → cosine | `scripts/eval/metric_win_tie_133.py`；backend `score_r2.py:Embed` |
| **MuQ-embed-ref** | MuQ-large @ `0562a57814f6f8bbd9fdea0a25921a2fce1a841a` | 同上 | 1024-d embedding → cosine | 同上 |
| **MEva** | pinned f03 `models/pooled/small/f03_sae_only_cnn_clean_best.pth` | 原生 16 kHz | MEva | `scripts/eval/meva_runtime.py` |
| **CLAP** | 既有 laion_clap ckpt（不新增 pin） | 48 kHz；10 s 中心裁切 | cosine | `scripts/eval/eval_metrics.py:score_clap`；backend `score_r2.py:Clap` |
| **AES** | 既有 Audiobox Aesthetics（不新增 pin） | 16 kHz FLAC；模型內部合約照既有 scorer | PQ／CE／CU／PC | `scripts/eval/bench_r2/score_r2.py:Aes`；`metric_win_tie_126a.py` |
| **MIR** | `madmom`／`librosa` 既有環境 | 16 kHz；10 s | `pulse_clarity`↑、`ibi_cv`↓、`key_cnn_conf`↑、`chroma_entropy`↓ | `scripts/eval/mir_metrics.py` |

> SongEval 的 repo pin `848fb2f…` 寫在 `metric_win_tie_126d.py` docstring；`scripts/eval/songeval_src` 是 vendored tree、沒有 nested `.git`，**無法用 `git rev-parse` 驗證**。實作時要對 vendored tree 做 `sha256` 樹雜湊並綁進 contract（見 §5.3）。

### 2.3 既有 runtime 與可重用後端

| 檔案 | 可重用的東西 | 不要重寫 |
|---|---|---|
| `scripts/eval/bench_r2/score_r2.py` | `Aes`、`Meva`、`SongEval`、`Embed`（MERT＋MuQ）、`Clap`、`make_singmos`、`make_muqeval`、`make_cmirm`、`make_pam` | 模型前向與前處理 |
| `scripts/eval/bench_r2/run_job.py` | contract 前檢、decode cache、per-dataset 迴圈、MEva 專用 venv 切換 | queue action 的 preflight 邏輯 |
| `scripts/eval/newmetrics_muqeval.py`／`newmetrics_cmirm.py`／`newmetrics_pam.py` | 模型載入、輸入契約、分數抽取 | pin 與預處理 |
| `scripts/eval/meva_runtime.py` | MEva pinned adapter | — |
| `scripts/eval/metric_win_tie_126a.py` | `variants()` 的 53 個變體、`transform()`、`judge()`、`analyze()`、R115 路徑 | 變體定義與 RNG seed |
| `scripts/eval/metric_win_tie_126b.py`／`126c.py`／`speed_pitch_extension_134.py` | 刺激 manifest、`judge()`／`analyze()`、S 的原始定義 | 標籤與方向 |
| `scripts/eval/metric_win_tie_126d.py`／`metric_win_tie_133.py` | 126a／126b／126c 的逐 clip TSV 產生器、S 的算法 | 既有 TSV 的欄位格式 |
| `scripts/eval/mir_metrics.py` | 四項 MIR 特徵 | — |

---

## 3. 共用評分 driver 設計（實作階段才建檔）

### 3.1 目標

一個 driver 吃「manifest ＋ model 名稱」，吐「per-clip TSV」，欄位與 126d／133 的 TSV 相容；126a、126b、126c、134 四個協定共用同一支，避免每個模型各寫一份迴圈。

### 3.2 介面

```bash
python -I scripts/eval/metric_coverage_20261027/score_coverage.py \
  --protocol 126a|126b|126c|134 \
  --model pam|muqeval_A1|muqeval_A3a|cmirm|singmos|songeval|mert|muq \
  --manifest <items.json|sample_manifest.json> \
  --out <cell_dir> \
  --shard <i>/<n> \
  [--audio-cache <dir>] \
  [--full-cells <ambiguous_cells.json>] \
  [--limit N] \
  [--preflight|--validate-only] \
  [--dry-run]
```

- `--protocol`：決定 manifest schema 與 base 的配對方式。
- `--model`：dispatch 到 §2.3 的 backend；driver 不複製模型程式。
- `--manifest`：
  - 126a：`sample_manifest.json` 或 `units_manifest.json`；每筆是 `{key, variant, system, clip_id, base, source, kind, domain, over_fs}`。
  - 126b／126c／134：直接吃既有的 `stimuli/items.json`；每筆已有 `key`、`base`、`source`、`kind`、`case`、`path`。
- `--out`：`~/eval_output_nvme/metric_coverage_20261027/<model>/<protocol>/`。
- `--shard i/n`：126a 以 unit（variant×system）為單位切；其他協定以 item 區間切。
- `--audio-cache`：126a 的取樣音檔快取；沒有就 on-the-fly 用 `metric_win_tie_126a.transform` 生成。
- `--full-cells`：只對 `ambiguous_cells.json` 列出的 unit 跑完整 5,131 clip（第二階段）。
- `--preflight`：只驗 pin、manifest sha、S sha、磁碟、音檔數量／長度，不載模型。
- `--validate-only`：只驗輸出 TSV 的列數、NaN、base 覆蓋，不評分。

### 3.3 輸出格式

`<out>/<protocol>_<model>_shard<i>.tsv`，欄位：

```text
key	base	source	kind	case	axis	impl	value	domain	over_fs	<score columns>
```

- `key`／`base`／`source`／`kind`／`case` 與 126d／133 的 TSV 同名同義；`axis`／`impl`／`value` 只在 134 用。
- score columns 依模型：`PQ,CE,CU,PC`／`meva`／`mi,ta`／`quality,alignment`／`pam`／`singmos`／`se_coh,...,se_mus`／`mert_cosim`／`muq_cosim`／`clap`。
- `mert_cosim`／`muq_cosim` 存**原始 cosine**（與 133 相同）；沒有 base 的列（clean／ref）以自己為 base、cosine = 1。判分時 d = case − base，才會得到 §1 定義的 `cosim(case, clean base) − 1`。
- 126a 的 `key = <variant>__<system>__<clip_id>`，`base = clean__<system>__<clip_id>`；同一 system 內所有 variant 用同一組 clip 子集，配對才成立。
- 126b／126c／134 的 `key`／`base` 直接沿用 items.json，不重新編號。

### 3.4 Resume 與冪等

- 每個 shard 一個 append-only TSV；啟動時讀既有 `key` 集合，已完成且分數非 NaN 的 key 跳過。
- 寫入用 `tmp + os.replace`；同一 shard 加 `flock`，避免兩個 worker 寫同一檔。
- `run.json` 記：manifest sha、model pin、S sha、driver sha、開始／結束時間、clip 數、每小時 clip 數。
- 中途被搶佔：下一個 job 從缺的 key 續跑；不重跑已完成的 unit。

### 3.5 前檢（preflight）必過項

1. model pin 與 §2.2 一致（HF snapshot 的 commit、torch.hub clone 的 HEAD、vendored tree 的樹雜湊）。
2. manifest sha 與 contract 一致。
3. S json 存在且 sha 一致（162 產物）。
4. 磁碟：126a 快取 ≥60 GB 可用；非 126a ≥10 GB。
5. 刺激：R115 的 37,867 個 FLAC 存在（2026-10-27 實測，只含五個協定 system 的 `lvl30`；舊記 70,603 是 offload 前含所有層級的數字）；126b／126c／134 的 items.json 筆數與路徑存在；MusicCaps TSV 存在。
6. 抽第一筆與最後一筆音檔，檢查取樣率、長度、可解碼。
7. 126a 的 variant 清單與 `metric_win_tie_126a.variants()` 逐項相同（53 個）。

### 3.6 判定器

`scripts/eval/metric_coverage_20261027/judge_coverage.py`：

- 讀 per-clip TSV＋items＋S json，算 `d`、`m=0.10S`、每個 unit／pair 的勝／平／負。
- 126a：每個 unit（variant×system）獨立判 80%；矩陣格填「22 tie units 通過數／30 lose units 通過數」，不把 units 平均成單一分數。
- 抽樣單元：標 `sampled=true`；ambiguous 的 unit 用第二階段 full 結果覆蓋，最終標 `full_rerun=true`。
- 輸出 `coverage_summary.json`、`coverage_matrix.md`、`coverage_matrix.html`。
- 所有判定先寫進 summary，再產出 deck 的格子；不手動改格子。

---

## 4. GPU 時間估算

### 4.1 實測速率與出處

| 模型 | 速率 | 出處／說明 |
|---|---|---|
| AES 4 軸 | 0.0008 s／audio-s | `docs/experiments/benchmark_screening_r2_20261025.md:274–288`（實測） |
| MEva | 0.0040 s／audio-s | 同上（實測） |
| CLAP | 0.008 s／audio-s | 同上（實測） |
| MuQ-Eval A1 | 0.12 s／clip（10 s） | 同上（實測） |
| MuQ-Eval A3a | 0.08 s／clip | 同上（實測） |
| CMI-RM | 0.35 s／30 s chunk | 同上（實測）；10 s clip 以 1 chunk 計 → 0.117 s／clip（**推估**，線性換算） |
| PAM | 0.028 s／audio-s → 0.29 s／10 s clip | 同上（實測 0.028 s／audio-s；0.29 s／clip 為換算） |
| SongEval | 55.4 clips／s（126a，含 transform） | `~/eval_output_nvme/metric_win_tie_126d_20261020/126a/*/*/songeval.tsv` 的 265 檔 mtime 03:23:31→10:12:47，409.27 min／1,359,715 clips |
| MERT-ref | 137.7 clips／s（126a，含 transform） | `~/eval_output_nvme/metric_win_tie_133_20261020/126a/*/*/mert_cosim.tsv` 的 265 檔 mtime 13:27:44→16:12:12，164.47 min／1,359,715 clips |
| MuQ-embed-ref | 51.8 clips／s（**推估**） | 133 的 126c→126b TSV mtime 11:45:11→12:23:59，38.8 min 內完成 87,653×2 embeddings；與 MERT 137.7／s 反推 MuQ ≈51.8／s |
| SingMOS | 0.05 s／clip（**推估**） | r2 doc 標「推估」；pilot 會實測 |
| MIR | 9.2 clips／s（CPU） | `CLAUDE.md:341`「5,521 clips 約 10 分鐘／格」→ 9.2／s |
| 126a 端到端 | 19.5 clips／s（含 AES＋MEva＋CLAP＋VGGish＋transform） | `~/gpu_queue/p2/done/126_metric_win_tie_signal.terminal.json`：2026-10-15T19:50:46Z→2026-10-16T15:17:18Z＝19.44 h／1,359,715 clips |

> 上表的 GPU 小時是**模型前向**，不含 transform 與 I/O；job 時長另加 15%（見 §5）。126a 的 SongEval／MERT 速率已含 transform，因為那兩輪是端到端跑完 265 個 unit。

### 4.2 全量 vs 抽樣：clip 數

| 協定 | clip 數／模型 | 說明 |
|---|---:|---|
| 126a 全量 | 1,359,715 | 5,131 clip × 5 systems × 53 variants（265 units） |
| 126a 抽樣 N=512 | 135,680 | 265 units × 512 clip；同一 system 內所有 variant 共用同一組 512 clip |
| 126a 抽樣 N=256（PAM） | 67,840 | 265 units × 256 clip |
| 126b＋126c＋134 | 173,606 | 87,653＋36,366＋49,587 |
| clean S（新模型） | 25,655 | 5,131 clip × 5 TTM clean systems |

### 4.3 每模型 × 每協定 GPU 小時

只列**新跑**；已完成的 126a（AES／MEva／CLAP／SongEval／MERT）不重跑。`full` 與 `sampled` 皆為模型前向小時；`non-126a`＝126b＋126c＋134；`clean S` 由 job 162 另計。

| 模型 | 126a full | 126a sampled | non-126a | clean S | full 合計 | sampled 合計 |
|---|---:|---:|---:|---:|---:|---:|
| MuQ-Eval A1 | 45.32 | 4.52 | 5.79 | 0.86 | **51.11** | **10.31** |
| MuQ-Eval A3a | 30.22 | 3.02 | 3.86 | 0.57 | **34.07** | **6.87** |
| CMI-RM | 44.06 | 4.40 | 5.63 | 0.83 | **49.69** | **10.02** |
| PAM | 109.53 | 5.46（N=256） | 13.98 | 2.07 | **123.52** | **19.45** |
| SingMOS | 18.88 | 1.88 | 2.41 | 0.36 | **21.30** | **4.30** |
| MuQ-embed-ref | 7.29 | 0.73 | 0.93 | 0.14 | **8.22** | **1.66** |
| SongEval（只補 134） | 0（已做） | 0 | 0.25 | 0（S 已有） | **0.25** | **0.25** |
| MERT-ref（只補 134） | 0（已做） | 0 | 0.10 | 0（S 已有） | **0.10** | **0.10** |
| **小計** | **255.30** | **20.01** | **32.95** | **4.82** | **293.08** | **57.78** |
| **加 15% I/O／啟動** | — | — | — | — | **337.0 h** | **66.4 h** |

> 小計與合計用未四捨五入的原始值計算；逐列相加可能差 ±0.01。
>
> 全量 293.1 h 的組成：A1 51.11＋A3a 34.07＋CMI 49.69＋PAM 123.52＋SingMOS 21.30＋MuQ 8.22＋SongEval 0.25＋MERT 0.10＋clean S 4.82＝293.08。抽樣 57.78 h 同式。
>
> PAM 是唯一在全量下超過 100 h 的模型；抽樣把 PAM 從 109.5 h 壓到 5.5 h（126a 部分），是整個計畫可行的關鍵。

#### 4.3.1 與 §6 job 表的對帳（66.4 h → 102.8 h）

§4.3 的 **66.4 h** 只是「模型前向」——sampled 126a＋non-126a＋clean S，`57.78 × 1.15` 之後的數字。§6 的 **102.8 h** 是 17 個 job 的總和，多了 126a 的 ambiguous full rerun 與三個 CPU job。對帳如下（同一個 15% I/O／啟動係數，避免重複計入）：

| 項目 | 小時 | 說明 |
|---|---:|---|
| §4.3 模型前向（未含 I/O） | 57.78 | sampled 126a 20.01＋non-126a 32.95＋clean S 4.82 |
| 126a ambiguous full rerun（24 units／模型，未含 I/O） | 24.06 | 27.67 ÷ 1.15；逐 job 差額見下 |
| ＋15% I/O／啟動 | 12.28 | 81.84 × 0.15 |
| 161 CPU 前置 | 2.0 | sample manifest／agreement／快取／rec·floor 回填 |
| 176 MIR（CPU） | 6.4 | 126a 抽樣四項＋134 key／chroma＋clean S |
| 177 判定與矩陣 | 1.2 | 只讀 TSV |
| 逐列四捨五入差 | −0.92 | §4.3 用原始值、§6 每列四捨五入到 0.1 h |
| **合計（§6 表列）** | **102.8** | 161–177 |

full rerun 的 27.67 h（已含 15%）是 §6 中 126a job 減掉同模型 §4.3 sampled 前向的差額：164 4.70＋166 3.13＋168 4.74＋170／171 11.42＋173 1.94＋174 1.74。若 ambiguous union 到 48 units，這一塊會再增加約 27 h（§4.4 已預留）。

### 4.4 126a 抽樣量怎麼選：用既有 126a 結果做 sample-vs-full

**方法**：對每個既有指標（PQ／CE／CU／PC／MEva／CLAP），在 260 個 cell（52 case × 5 systems）上，從 5,131 clip 隨機抽 N 個（B=200 bootstrap，seed 20261027，所有 variant／system 共用同一組 clip），用抽樣資料判「tie 是否 ≥80% 通過／lose 是否 ≥80% 通過」，再與全量判定比。`ambiguous` 的定義是抽樣 pass rate 落在 0.80±0.04；`out-of-band error` 是全量 pass rate 已在 0.80±0.04 之外時，抽樣與全量的最大差距。

> **單位**：`out-of-band error` 是 **pass-rate 的差**（0.0026＝0.26 個百分點），不是 per-pair `d` 的差。因此下表只能跟同單位的東西比：跟 `ambiguous` 帶寬 0.04、以及 80% 門檻到 0／100% 的 0.20 餘裕比；**不可以拿來跟 `m = 0.10×S`（分數單位）比大小**。per-pair `d` 的抽樣誤差本輪沒有模擬，列為 §8.2 的風險，由 pilot 實測。

| N | 總一致率 | ambiguous cell 數（每指標） | out-of-band error 上限（pass-rate 單位） | 與 0.04 ambiguous 帶的比 |
|---:|---|---:|---:|---:|
| 256 | 91.80–100% | 0–24／260 | ≤0.0026 | 帶寬的 1/15 |
| **512** | **96.34–99.61%** | **0–24／260** | **≤0.0004** | **帶寬的 1/100** |
| 1024 | 97.16–100% | 0–23／260 | ≤0.0000 | 帶寬的 0 |

N=512 的逐指標結果（格式：`一致率／ambiguous 數／out-of-band error`）：

| 指標 | tie | lose |
|---|---|---|
| PQ | 0.9871／7／0.0000 | 0.9865／20／0.0000 |
| CE | 0.9909／8／0.0000 | 0.9842／20／0.0000 |
| CU | 0.9898／8／0.0000 | 0.9634／24／0.0004 |
| PC | 0.9920／4／0.0000 | 0.9817／14／0.0003 |
| MEva | 1.0000／0／0.0000 | 0.9911／7／0.0000 |
| CLAP | 0.9961／6／0.0000 | 0.9781／18／0.0001 |

**決策**：

- A1／A3a／CMI／SingMOS／MuQ-embed 用 **N=512**；PAM 用 **N=256**（PAM 最貴，N=256 的 out-of-band error ≤0.0026 pass-rate＝0.26 個百分點，是 0.04 ambiguous 帶的 1/15，且落在帶外的 cell 由 full-rerun 規則接住；這個比較是 pass-rate 對 pass-rate，不是對 `m`）。
- 所有 ambiguous unit 進第二階段：**完整 5,131 clip 重跑**，最終判定用 full 結果；summary 同時報 `sampled` 與 `full_rerun` 兩個數字。
- 若模型的分數分布與既有六指標差異很大（pilot 看到 score range < 0.1×S 或 > 10×S），把 N 加倍到 1024 再跑；不得事後只挑「看起來不 ambiguous」的 unit。
- 這組模擬是**本輪 scratch 分析，尚未落檔**；job 161 會用同一 seed 重跑並寫 `sample_agreement_20261027.json`，**正式數字以那個檔為準**（不是本表）。161 的 JSON 同時寫 `min_agreement_at_512`（六指標取最小）、每指標 ambiguous 數與 out-of-band error。
- **161 低於本表時的預先規定**（不改 m、不改 80%、不事後挑 unit）：
  - 若 `min_agreement_at_512` < 0.9634（本表 N=512 的下限）：所有 126a job 改用 **N=1024**（PAM 用 **N=512**）。已完成的 512 抽樣不丟，只補新增的 512（PAM 補 256）clip；抽樣 key 以 manifest 前綴判定，重跑時自動跳過。
  - 若 1024 的 `min_agreement` 仍 < 0.9716（本表 N=1024 的下限）：N 維持 1024，**所有 cell 都必須有 full rerun 才算判定**；沒有 full 結果的 cell 標 `inconclusive`，不得用 sampled 判 pass。
  - ambiguous unit 一律 full rerun，無論 N 為何；union 超過 48 units 時照跑（成本增加，見 §4.3.1），不得刪 unit。
- 兩維度模型的 ambiguous union：A1／A3a（MI／TA）與 CMI（quality／alignment）的 ambiguous unit 可能各自到 24 個；若 union 超過 24，full-rerun 成本按實際 unit 數線性增加。job 164／166／168 的 9.9／6.6／9.8 h 是以 24 units 估；union 到 48 時約 14.6／9.8／14.2 h。

### 4.5 Pilot 程序（每個新模型開跑前）

1. 抽 3 個 unit（clean＋一個 tie＋一個 lose）× 64 clip，跑完整 driver。
2. 檢查：TSV 欄位、base 覆蓋、NaN、分數變異（不是常數）、實測 clips／s。
3. 用實測 clips／s 更新該 job 的 ETA；若比 §4.1 慢 >2×，先查前處理（取樣率、chunk、batch）再繼續。
4. Pilot 資料不進最終 summary；只當 smoke test。

---

## 5. 預登錄判準

### 5.1 共通規則（套用到每個模型 × 每個協定）

- `d = score(case) − score(base)`，以 `key` 逐 clip 配對；base 必須是同一個 clip／同一個 system／同一個 window。
- `m = 0.10 × S`；S 的定義**沿用原協定**：
  - 126a：5 個 TTM clean system 平均值的 range（每個指標各自算；既有值 PQ 0.9189、CE 1.2005、CU 0.8452、PC 0.3447、MEva 0.1724、CLAP 0.1038）。
  - 126b／126c／134：神經指標沿用 126a 的 S；MIR 的 126b S＝所有 clean 的 SD、126c S＝mix segments 的 SD；SongEval 五維的 S_i＝5 個 TTM clean system 平均值的 range（既有 `se_coh` 0.1814／`se_mem` 0.1620／`se_nat` 0.1387／`se_cla` 0.1512／`se_mus` 0.1498）；MERT `S=0.0393`、MuQ-embed `S=0.1087`；134 的 S 見 `speed_pitch_extension_20261020.md`（PQ 0.919、CE 1.200、CU 0.845、PC 0.345、MEva 0.172、CLAP 0.104、pulse 0.272、ibi 0.026）。
  - 新模型的 S 由 job 162 用同一協定量；**不得**沿用別人的 S，除非該指標的 S 定義本來就是跨系統 range 且 162 判定不可用（此時標 `S_fallback` 並寫理由）。
- tie cell 通過＝≥80% 的 pair `|d| ≤ m`。
- lose cell 通過＝≥80% 的 pair `d < −m`（方向依原協定；若指標是「越低越好」，先依原 doc 轉向，不可自行改方向）。
- `soft` 只做描述，不判通過與否。
- rec：真人錄音配對，方向與 126b 原 doc 相同；報 overall 與 per-source（pro／poor／poor_sing／csd_real），不把不同 source 混成一個數字。
- floor：只報百分位（底線分數高過多少比例的真實／MIDI clean），**不判 pass／fail**；floor 不是模型好壞的證據。
- 134：tie 帶 tempo 0.95–1.05、pitch ±1 半音；lose 帶 tempo ≤0.80／≥1.25、pitch ±2／±5／±12；每個軸分開判，不平均。
- 多維度模型（A1／A3a 的 MI／TA、CMI 的 quality／alignment、SongEval 五維）**每個維度分開判**；不允許先平均再判，也不允許只報最好的那一維。
- 常數分數（score 的 SD＝0 或所有 d＝0）：標 `degenerate`，不得算 pass；這是發現，不是成功。
- S 無法量測（NaN／0／樣本不足）：標 `inconclusive`，不補 0、不縮小 m。

### 5.2 每個協定的具體判準

| 協定 | tie 標籤 | lose 標籤 | 配對方式 | 判定單位 | 主要輸出 |
|---|---|---|---|---|---|
| 126a | 22 個 tie variant | 30 個 lose variant | 同 system、同 clip、variant vs clean | 每個 unit（variant×system） | 22 tie units 通過數／30 lose units 通過數 |
| 126b MIDI | `tie`／`soft` | `lose` | 同 piece、同 excerpt、同 SoundFont | 每個 item pair | 依 `source`（generaluser／fluidr3）分開 |
| 126b rec | `tie` | `lose` | 同曲 pro vs poor vs poor_sing | 每個 pair | per-source；`poor_sing` 是 SingMOS 唯一 in-domain |
| floor | — | — | 無配對 | 每個底線檔 | 百分位，描述性 |
| 126c | `tie`／`soft` | `lose` | 同 window 的 mix vs 編輯版 | 每個 window×case | 依 `case` 分開 |
| 134 | tempo 0.95–1.05、pitch ±1 | tempo ≤0.80／≥1.25、pitch ±2／±5／±12 | 同 excerpt 的 clean vs 延伸 | 每個 axis×impl×value | 每個軸分開；OOD-extreme 另標 |

### 5.3 模型 × 協定的判定與 domain 標記

| 模型 | 126a | 126b MIDI | 126b rec | floor | 126c | 134 |
|---|---|---|---|---|---|---|
| MuQ-Eval A1（MI／TA） | 判；in-domain | 判；`OOD-MIDI` | 判；in-domain | 描述；`OOD-synthetic` | 判；in-domain | 判；極端標 `OOD-extreme` |
| MuQ-Eval A3a（MI／TA） | 同上 | 同上 | 同上 | 同上 | 同上 | 同上 |
| CMI-RM（quality／alignment） | 判；in-domain | 判；`OOD-MIDI` | 判；in-domain | 描述；`OOD-synthetic` | 判；in-domain | 判；極端標 `OOD-extreme` |
| PAM | 判；in-domain | 判；`OOD-MIDI` | 判；in-domain | 描述；`OOD-synthetic` | 判；`OOD-stems` | 判；極端標 `OOD-extreme` |
| SingMOS | 判；`OOD-instrumental` | 判；`OOD-instrumental` | 判；只有 `poor_sing` in-domain，其餘 `OOD-instrumental` | 描述；`OOD-synthetic` | 判；`OOD-instrumental` | 判；`OOD-instrumental` |
| SongEval 五維 | 判；`OOD-10s-instrumental` | 判；`OOD-10s-instrumental` | 判；in-domain | 描述；`OOD-synthetic` | 判；in-domain | 判；`OOD-10s-instrumental` |
| MERT-ref | 判；in-domain | 判；in-domain | 判；in-domain | `—` | 判；in-domain | 判；極端標 `OOD-extreme` |
| MuQ-embed-ref | 判；in-domain | 判；in-domain | 判；in-domain | `—` | 判；in-domain | 判；極端標 `OOD-extreme` |
| MIR 四項 | 判；in-domain | 判；in-domain | 判；in-domain | 描述；`OOD-synthetic` | 判；in-domain | pulse／ibi 已做；key／chroma 補跑；極端標 `OOD-extreme` |

> `OOD-*` 的結果照報，但**不進模型否決**：例如 SingMOS 在器樂上 fail 不代表 SingMOS 不好，只代表它不適用。矩陣上仍填 `●`（有跑、有判），旁邊加 OOD 標記。

### 5.4 判定標籤

每個模型 × 協定 × 維度輸出一個標籤：

| 標籤 | 條件 |
|---|---|
| `pass` | 該協定的 tie 與 lose 都 ≥80%，且無 degenerate／inconclusive |
| `partial` | tie 與 lose 其中一個 ≥80%，另一個 <80%（或只跑了一部分 unit） |
| `fail` | tie 或 lose 的通過率 <80%，且不是 OOD 造成 |
| `inconclusive` | S 無法量測、ambiguous unit 未完成 full rerun、或分數常數 |
| `ood_only` | 只有 OOD 格有結果，in-domain 格不足 |
| `n/a` | 定義上不可判（MERT／MuQ floor） |

判定寫進 `coverage_summary.json`；矩陣與 deck 只讀這個檔，不手動改。

---

## 6. Job 切分（編號自 161 起）

> 截至 2026-10-27，`~/gpu_queue/p2` 最新編號是 **160**（`pending/160_bench_r2_J7_musicprefs.sh`），所以 161 起可用。實際排程前仍要 `ls` 一次 pending／done，避免与其他 session 撞號。
>
> 所有 job 的 action 都走 `score_coverage.py`；contract 先於 wrapper 寫好（見 `reference_gpu_queue_job_authoring_gotchas` 的教訓）。每個 job 的時長已含 15% I/O／啟動。

| Job | 內容 | 模型／協定 | 估計時長 | 依賴 | 可搶佔 | Contract 應綁 |
|---:|---|---|---:|---|---|---|
| **161** | CPU 前置：sample manifest（seed 20261027）、sample agreement JSON、126a 取樣音檔快取、SongEval／MERT／MuQ 的 rec／floor 分析回填 | 全部；無 GPU | 2.0 h（**推估**） | 無 | 是（檔案邊界） | R115 `selection.json`、MusicCaps TSV、126b／126c／134 `items.json`、`metric_win_tie_126a.py`、sample seed／manifest |
| **162** | Clean S：5 個 TTM clean system × 5,131 clip，算新模型的 S | A1／A3a／CMI／PAM／SingMOS／MuQ-embed | 5.5 h | 161 | 是（per model） | model pins、clean manifest、driver、S 產生器 |
| **163** | A1 非 126a | A1：126b＋126c＋134 | 6.7 h | 162 | 是（per shard） | 162 的 `s_values_muqeval_A1.json`、items shas、A1 pin |
| **164** | A1 126a（N=512＋ambiguous full） | A1 | 9.9 h | 162 | 是（per unit） | sample manifest、A1 pin、S json、driver、ambiguous 清單 |
| **165** | A3a 非 126a | A3a | 4.4 h | 162 | 是（per shard） | 同 163，換 A3a pin |
| **166** | A3a 126a | A3a | 6.6 h | 162 | 是（per unit） | 同 164，換 A3a pin |
| **167** | CMI 非 126a | CMI | 6.5 h | 162 | 是（per shard） | 162 的 `s_values_cmirm.json`、CMI pin |
| **168** | CMI 126a | CMI | 9.8 h | 162 | 是（per unit） | 同 164，換 CMI pin |
| **169** | PAM 非 126a | PAM | 16.1 h | 162 | 是（per shard） | 162 的 `s_values_pam.json`、PAM pin |
| **170** | PAM 126a part 1：real＋ours_cfg3neg＋ours_cfg0 | PAM | 10.6 h | 162 | 是（per unit） | sample manifest、PAM pin、S json、part 1 的 system 清單 |
| **171** | PAM 126a part 2：musicgen＋musicldm | PAM | 7.1 h | **170**（只有一張 GPU，必須序列） | 是（per unit） | 同 170，換 system 清單 |
| **172** | SingMOS 非 126a | SingMOS | 2.8 h | 162 | 是（per shard） | 162 的 `s_values_singmos.json`、SingMOS clone commit |
| **173** | SingMOS 126a | SingMOS | 4.1 h | 162 | 是（per unit） | sample manifest、SingMOS pin、S json |
| **174** | MuQ-embed 126a＋134 | MuQ-embed | 2.7 h | 162 | 是（per unit） | MuQ pin、sample manifest、S json |
| **175** | SongEval 134＋MERT 134 | SongEval／MERT | 0.4 h | 161（134 items）；S 已有 | 是（per shard） | SongEval ckpt／repo 樹雜湊、MERT pin、134 items、既有 S json |
| **176** | MIR：126a 取樣四項＋134 key／chroma（＋clean S 25,655） | MIR | 6.4 h（CPU） | 161 | 是（per shard） | sample manifest、audio cache manifest、`mir_metrics.py`、134 items |
| **177** | 判定與矩陣：讀所有 TSV，產 `coverage_summary.json`／matrix／deck 資料 | 全部 | 1.2 h | 163–176 全部 | 是（重跑成本低） | 所有 summary TSV 的 sha、S jsons、`judge_coverage.py` |

**總計**：161–177 約 **102.8 h**（含 15%；逐列四捨五入，±0.1 h）。最長單一 job 是 169（16.1 h）；PAM 的 169–171 合計 33.8 h，建議最先排。

### 6.1 依賴圖

```text
161 ─┬─ 162 ─┬─ 163 ─ 164
     │        ├─ 165 ─ 166
     │        ├─ 167 ─ 168
     │        ├─ 169 ─ 170 ─ 171
     │        ├─ 172 ─ 173
     │        └─ 174
     ├─ 175
     └─ 176
163–176 ────────────────── 177
```

- 162 必須在 163–174 之前，因為 S 是判定的一部分；但 175 只需要既有的 SongEval／MERT S，不必等 162。
- **170 → 171 是序列**：本機只有一張 GPU（154 跑完前由 queue 的 `gpu0.lock` 排他），171 不得與 170 同時排；171 的 preflight 要看到 170 的 `job_done_170.json`（或 170 的終止狀態為 completed／paused 且 TSV 完整），否則立刻失敗。兩者分開編號只是為了「per unit 可搶佔」與失敗時只重跑其中一半，不是為了平行。
- 176 是 CPU-only，可與 163–175 平行跑（不同資源）；若 queue 只允許一個 job，則排在 GPU job 之間，不佔 GPU 計算。
- 177 最後跑；它的輸入是 163–176 的 summary TSV，不是原始音檔。

### 6.2 Contract 綁定清單

每個 job 的 contract 至少綁：

- `launcher_sha256`、`experiment_harness` guest 的 `sha256`、action `sha256`、`contract_json_sha256`。
- `score_coverage.py` 的 `sha256`（driver）。
- `model_pins_20261027.json` 的 `sha256`（內含 §2.2 的 repo commit／weights sha／vendored tree 樹雜湊）。
- 輸入 manifest 的 `sha256`：`sample_manifest_20261027.json`、`items.json`（126b／126c／134）、R115 `selection.json`、MusicCaps TSV。
- 162 產出的 `s_values_<model>.json` 的 `sha256`（163–174 綁）。
- `expected_counts`：每個 shard 預期 clip 數；少於此數視為失敗，不可靜默通過。
- 磁碟下限：126a 60 GB、非 126a 10 GB。
- **不要綁** 126a 的 `units/` 目錄或 1.36M 音檔的逐檔 sha（太慢）；綁 unit 數（265）＋manifest sha＋抽樣清單 sha 即可。

### 6.3 失敗與續跑

- 本計畫的 queue job 以 **TSV 成長**為進度：每個 shard／unit 的 append-only TSV 記已完成 `key`，被搶佔（pause）或失敗後重排時，driver 從缺的 key 續跑，已完成且分數非 NaN 的不重跑。126a 以 unit 為續跑粒度，126b／126c／134 以 4,000-item 區間為粒度。
- pause：guest 收到 `control/pause.request.json` 後寫 `pause_progress_<job>.json`（含已完成的 shard／unit 清單）並以 pause 結束碼退出；launcher 重排後讀同一份進度，不需要人手動改檔。
- 被 queue 歸到 `held/` 或 `failed/`：先看 launcher 的 stderr 與 `run.json`／`.err.tsv`，修好後**重新排進 `pending/`**（不是原地重試）。contract 的 sha 或 manifest 不符時，先改 contract 並重綁 sha，再排。
- 模型載入失敗、pin 不符、manifest sha 不符、NaN：**不要重試同一 job**，先查前處理與 pin；修好後從頭跑該 shard。
- exit code 75 是 guest 的「可重試」訊號（例如 GPU 暫時被佔用），queue 會把 job 放回 pending；其餘非零碼一律當失敗處理，不自動重排。
- 126a 的 ambiguous full rerun 若只完成一部分：summary 標 `partial`，不得把未完成的 unit 當 pass；重排後只補未完成的 unit。

---

## 7. 磁碟與重生成

### 7.1 目前磁碟（2026-10-27 `df -h` 實測）

| 掛載點 | 大小 | 已用 | 可用 | 使用率 |
|---|---:|---:|---:|---:|
| `/`（NVMe） | 3.6 T | 2.9 T | **610 G** | 83% |
| `/mnt/HDD` | 7.3 T | 6.3 T | **1.1 T** | 87% |
| `/mnt/seagate` | 4.6 T | 2.6 T | **2.0 T** | 57% |

> 數字變很快；每個 job 的 preflight 都要重跑 `df -h`，不要信這張表的快照。

### 7.2 刺激是否在磁碟上

| 刺激 | 位置 | 大小／筆數 | 狀態 |
|---|---|---|---|
| R115（126a 的來源音檔） | `/mnt/seagate/nvme_offload_20261027/ttm_metrics_115_20261009`（`~/eval_output_nvme` 已 symlink 到 seagate） | 30 G／37,867 FLAC（`lvl30`，2026-10-27 實測） | **在**；126a 的 audio 不落地，從這裡即時 transform |
| 126a 分數 | `~/eval_output_nvme/metric_win_tie_126a_20261016/units/` | 628 MB／265 TSV | **在**；只有分數，沒有音檔 |
| 126b 刺激 | `~/eval_output_nvme/metric_win_tie_126b_20261016/stimuli` → `/mnt/seagate/kojiek_data/metric_invariance_128_inputs/126b_stimuli` | 25 G／87,653 FLAC | **在**（`du -L` 實測） |
| 126c 刺激 | `~/eval_output_nvme/metric_win_tie_126c_20261019/stimuli` | 16 G／36,366 FLAC | **在** |
| 134 刺激 | `~/eval_output_nvme/speed_pitch_extension_134_20261020/stimuli` | 14 G／49,587 FLAC | **在** |
| MusicCaps TSV | `/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv` | 5,521 筆 | **在** |

### 7.3 126a 需要重生成嗎？

- **不需要重生成既有分數**：§1.2 的 126a `●` 格已經有分數。
- **新模型需要音檔**：126a 的音檔当初沒有存；用 `metric_win_tie_126a.transform` 從 R115 即時生成，**確定性**（同 seed／同 variant／同 clip）與 126 当时一致。
- **不要**把 1,359,715 個音檔全部落地：以平均 425 KB／FLAC 估，約 **550–580 GiB**（**推估**）；seagate 雖有 2.0 T，但沒有必要，且會拖慢 I/O。
- **取樣快取**（建議）：135,680 個 FLAC × 425 KB ≈ **58 GB**（**推估**），放 NVMe `~/eval_output_nvme/metric_coverage_20261027/audio_cache/`；PAM 只用其中 67,840 個（約 29 GB）。job 161 建快取，全部 job 共用；177 完成後可刪。
- 若 NVMe 可用 <100 GB：跳过快取，on-the-fly transform；代價是每個模型多 5–10%（**推估**；126a 的 MERT／SongEval 端到端速率已含 transform，未見 transform 成為瓶頸）。

### 7.4 輸出大小

| 輸出 | 估算 |
|---|---|
| 新模型 per-clip TSV（6 模型 × 約 309k 列 × 200 B） | < 0.5 GB |
| summary JSON／matrix／HTML | < 50 MB |
| 126a 取樣快取 | ≈58 GB（可刪） |
| 暫存 decode cache | < 5 GB／job |

> 峰值需求：約 **65 GB NVMe**（快取＋輸出＋暫存）。目前 610 G 可用，足夠。

---

## 8. 風險、待補、以及不可以做的事

### 8.1 待補（實作時由哪個 job 補）

| 項目 | 狀態 | 補法 |
|---|---|---|
| 新模型的 S（A1／A3a／CMI／PAM／SingMOS／MuQ-embed） | **待補** | job 162 用 5×5,131 clean 算，寫 `s_values_<model>.json`；**正式 S 只認這個檔**（§4.4 的 scratch 數字不算）。**162 的 S 不可用時**（NaN／≤0／樣本不足／SD＝0）：該模型 × 該協定標 `inconclusive`，矩陣格留 `◐`（有分數、無判定），不縮小 m、不補 0。唯一例外是 **MuQ-embed**：它的 S 定義（5 個 clean system 平均的 range）與 133 完全相同，可用 133 的 `S_muq=0.108679` 當 `S_fallback`，但 summary 必須同時寫 `S_source="133"`、`S_fallback=true` 與理由，且矩陣格仍標 `inconclusive`、不得算 pass。其餘五個模型沒有同定義的既有 S，**不得借用** 126a 六指標或別人的 S。 |
| sample-vs-full agreement 的正式 JSON | **待補**（本輪只有 scratch） | job 161 用 seed 20261027 重跑，寫 `sample_agreement_20261027.json`；**正式一致率與 ambiguous 數以這個檔為準**。若 `min_agreement_at_512` < 0.9634 或任一指標 ambiguous >24，N 依 §4.4 的預先規定加倍（126a 全體 1024、PAM 512）；若 1024 仍 < 0.9716，所有 cell 都必須 full rerun，否則 `inconclusive`。不改 m、不改 80%、不事後挑 unit。 |
| SingMOS 的實測 clips／s | **推估**（0.05 s／clip） | job 172 的 pilot＋`run.json` |
| CMI 的 10 s clip 實測速率 | **推估**（由 0.35 s／30 s chunk 線性換算） | job 167 的 pilot |
| 161 的 2.0 h | **推估** | 建快取與 rec／floor 分析後記 `run.json` |
| 取樣快取 58 GB、全量 550–580 GiB | **推估**（平均 425 KB／FLAC） | 161 建完後 `du -sh` 實測 |
| SongEval vendored tree 的 pin | 只有 docstring 的 `848fb2f…`，`git rev-parse` 驗不到 | 實作時對 `scripts/eval/songeval_src` 做樹雜湊，綁進 contract |
| PAM／CMI 的授權 | 非商業研究用；本檔不處理授權 | 只做內部研究，不對外散布權重 |

### 8.2 風險

- **抽樣誤差**：N=512 的 agreement 是用既有六指標模擬的；新模型可能更不穩定。緩解：pilot、ambiguous full rerun、必要時 N=1024。
- **PAM 的 N=256**：若某些 cell 的 pass rate 正好在 0.80 附近，full rerun 會把 PAM 的 126a 成本拉高；最壞情況 170＋171 從 17.7 h 增加到約 30 h。已在 job 拆分中預留。
- **OOD 結果被誤讀**：SingMOS 在器樂、PAM 在分軌、SongEval 在 10 秒器樂的 fail 不是模型缺點。矩陣與 summary 都會帶 `OOD-*` 標記；寫論文時只能放在「適用範圍」段，不可放進「模型排名」。
- **MIR 的 key／chroma 不在 134 per_clip**：134 既有 TSV 只有 pulse／ibi；job 176 要重跑 key／chroma，不能從既有檔補。
- **MIR 的 126a S**：要用完整 5×5,131 clean 算，不能用取樣 clean 算；否則 S 與其他指標不可比。job 176 已含 25,655 clean clip。
- **Queue 撞號**：161 起是依 2026-10-27 的 160 為基準；排程前要再確認。
- **MERT／MuQ floor 的假平手**：floor 跟自己比會 cosim=1、d=0；本檔明確標 N/A，實作時不得為了「填滿矩陣」而偷跑。
- **AES／CLAP 的既有 pin**：本檔不新增；若既有 contract 的 pin 與現在環境不同，先查 contract，不要用「最新版」覆蓋。
- **沒有人類評分**：這整個矩陣是指標行為，不是人類 ground truth；結論只能寫「指標能不能分辨介入」，不能寫「哪個模型更好聽」。

### 8.3 不可以做的事

- 不排 queue、不跑 GPU、不改既有 contract／script／doc（本檔是一次性設計）。
- 不把 `MuQ-Eval A1／A3a` 與 `MuQ-embed-ref` 合併。
- 不把 OOD 的 fail 當模型 fail。
- 不用 floor 的 self-cosine 填 MERT／MuQ 的 floor 格。
- 不事後改 m、S、80% 門檻或 OOD 標記。
- 不把 sampled 的結果當 full 結果；ambiguous 一定 full rerun。
- 不把 126a 的 1.36M 音檔全部落地。

---

## 9. 附錄：這份文件的數字出處

| 數字 | 出處 |
|---|---|
| 126a 5,131×5×53＝1,359,715、265 units | `docs/experiments/metric_win_tie_126_20261016.md:135–169`；`scripts/eval/metric_win_tie_126a.py` |
| 126a S（PQ 0.9189、CE 1.2005、CU 0.8452、PC 0.3447、MEva 0.1724、CLAP 0.1038） | `docs/experiments/results/metric_win_tie_126a_*` summary（既有結果） |
| 126b 87,653（MIDI 54,672／rec 32,807／floor 174） | `docs/experiments/metric_win_tie_126_20261016.md`；`~/eval_output_nvme/metric_win_tie_126b_20261016/stimuli/items.json` |
| 126c 36,366、65 cases／window | `docs/experiments/metric_win_tie_126c_musdb_20261019.md`；`items.json` |
| 134 49,587；tie／lose 帶 | `docs/experiments/speed_pitch_extension_20261020.md:25–31`；`items.json` |
| SongEval S_i | `docs/experiments/results/metric_win_tie_126d_summary.json` |
| MERT／MuQ S | `docs/experiments/results/metric_win_tie_133_summary.json`（S_mert 0.039296、S_muq 0.108679） |
| 速率 0.0008／0.0040／0.008／0.12／0.08／0.35／0.028 s | `docs/experiments/benchmark_screening_r2_20261025.md:274–288` |
| SongEval 55.4／s、MERT 137.7／s | 265 個 126a TSV 的 mtime 連續區間（見 §4.1） |
| 126a 端到端 19.44 h | `~/gpu_queue/p2/done/126_metric_win_tie_signal.terminal.json` |
| MIR 9.2／s | `CLAUDE.md:341`（5,521 clips 約 10 分鐘／格） |
| 磁碟 610 G／1.1 T／2.0 T | 2026-10-27 `df -h / /mnt/HDD /mnt/seagate` |
| 126b 25 G、126c 16 G、134 14 G、R115 30 G | `du -L -sh`（2026-10-27） |
| SingMOS clone commit | `git -C ~/.cache/torch/hub/South-Twilight_SingMOS_v1.1.2 rev-parse HEAD` → `959691e…` |
| queue 最新 160 | `ls ~/gpu_queue/p2/pending ~/gpu_queue/p2/done` 的編號 |

---

*本檔為預登錄；實作階段的任何偏離都要先改這份文件並記理由，不能只在程式裡改。*
