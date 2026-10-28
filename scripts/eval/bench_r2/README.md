# bench_r2 — round-2 人類評分 benchmark 篩選（實作）

設計與預登錄：`docs/experiments/benchmark_screening_r2_20261025.md`（§12 實作範圍、§13 核實、§14 偏離）。
本目錄只實作，不改設計。資料與分數一律在 `/mnt/seagate/benchmarks_r2/`（不寫 NVMe）。

## 檔案

| 檔案 | 用途 |
|---|---|
| `bench_datasets.py` | 各資料集 builder／manifest／PLAN（每個 job 要哪些 variant×loudness×subset）。檔名刻意不叫 `datasets.py`（會遮住 HuggingFace `datasets`） |
| `pick_music_arena.py` | 預登錄的 2,000 battle 抽樣（seed 20261025）與 mp3 下載 |
| `decode_cache.py` | 解碼到 float32 mono WAV 快取 `derived/<ds>/`＋`index.tsv`（段長 V5/V10/V30/VFULL、響度 raw／norm −23 LUFS 純增益）；`--limit N` 只解前 N 個 work item；冪等、可 resume；AIME 以 parquet shard 為單位（每 shard 只讀一次音訊欄） |
| `score_r2.py` | 各模型評分，輸出 `scores/<ds>/<model>__<variant>_<loud>.tsv`＋`.err.tsv`，以 unit_id 為單位 row-level resume；`--smoke N` 寫 `scores_smoke/` |
| `run_job.py` | queue action：`--preflight`（有同資料集的 decode_cache 在跑時 rc 75＝稍後重試）／run／`--validate-only` |
| `make_queue_job.py` | 產生 `scripts/queue_candidates/<no>_bench_r2_<label>.sh` 與 `docs/experiments/bench_r2_<label>_contract.json`（contract 先、wrapper 後；綁 sha） |
| `../../experiment_harness/bench_r2_guest.py` | queue guest（stall 28,800 s，progress 看 scores 與 derived tsv 大小） |
| `check_cache.py` | §10.3 CPU 檢查（檔數、ffprobe 20 檔、peak／LUFS／靜音 proxy）→ `analysis/cache_check.json` |
| `analyze_r2.py` | §6–§8 統計與 G1–G6 判定 → `analysis/summary.json`＋`REPORT.md` |

## 資料與規模

| 代號 | 資料集 | manifest 項數 | 原生 SR／備註 |
|---|---|---|---|
| B1 | ODAQ | 240（30 trial × 8 process；`derived` 960 unit：V10×raw/norm 之外見 PLAN） | 48 kHz stereo，10–16 s；授權 CC BY 31／BY-NC 9／CC0 5 |
| B2 | AIME | 1,300 snippet（13 系統 × 100，各一條不同 track；parquet 6,500 track） | 16 kHz 級（AIME 音訊）；10 s snippet |
| B3 | SongEval | 2,399 首（S800 子集 800） | mp3 44.1／48 kHz，64–320 kbps，127–365 s |
| B4 | SingMOS-Pro | 7,981（test 3,101） | 16 kHz wav，約 4 s |
| B5 | MC522 | 522 | 16 kHz，10 s |
| B6 | Music Arena | 2,000 battle＝3,998 mp3（2 首下載 404，見 `meta/music_arena_missing.json`）；norm800 子集 800 | mp3 32／44.1／48 kHz，20–253 s |

快取 `index.tsv` 欄位：`unit_id key variant loud sr n_samples dur n_win files full invalid gain_db over_fs peak lufs_in lufs_out`。
`invalid` 非空＝解碼失敗，該 unit 不評分、在 `validate-only` 時要列出。

## 使用

```bash
# 解碼（dac venv；-I 因資料為不受信任輸入）
~/venvs/dac/bin/python -I decode_cache.py <dataset> --job <J> --workers 6 [--limit N]
# smoke（先 decode --limit，再每個模型各開一個行程；cmirm 不可與 songeval 同行程；clap 要 HF_HUB_OFFLINE=1；meva 用 MEva venv）
~/venvs/dac/bin/python -I score_r2.py --dataset aime --job J2 --models aes --smoke 8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/venvs/dac/bin/python -I score_r2.py --dataset aime --job J2 --models clap --smoke 8
~/MeanAudio/runtime/meva_20261003/venv/bin/python -I score_r2.py --dataset aime --job J2 --models meva --smoke 8
# 生成 queue job（contract 綁的檔案改動後必須重新產生）
~/venvs/dac/bin/python -I make_queue_job.py 150 J2 J2 aime=aes,meva,... --hours 1.2 --question "..."
# 全部評完後
~/venvs/dac/bin/python -I check_cache.py
~/venvs/dac/bin/python -I analyze_r2.py --nboot 2000
```

## Queue job（p2）

| job | 內容 | 預估（含 15% 開銷） |
|---|---|---|
| 142 J1 | ODAQ＋MC522 全模型 | 0.8 h（實測因共用碟 I/O 競爭明顯較慢） |
| 150 J2 | AIME 8 指標 | 1.2 h |
| 143 J3a／144 J3b | SingMOS-Pro 兩半 | 2.4／3.2 h |
| 147 J4a／148 J4b／149 J4c | SongEval 三份 | 2.1／2.1／1.6 h |
| 145 J5a／146 J5b | Music Arena | 3.2／3.7 h |
| 合計 | | ≈ 21.5 h（J6 跳過；範圍 12–24 h） |

## 已知陷阱

- HF `datasets` 遮蔽 → `bench_datasets.py`。
- `HF_HUB_OFFLINE`：clap 要設 1；cmirm／singmos 要拿掉（wrapper 預設 1，`run_job.py` 逐模型處理）。
- cmirm 與 songeval 的 `muq` 模組衝突 → 一模型一行程（`run_job.py` 已是如此）。
- MEva 需 MEva venv（`runtime/meva_20261003/venv/bin/python`）；MEva 輸入為 32 kHz。
- Decode worker 要 pin MKL／OMP（已做）。
- 共用碟 I/O 競爭：J1 與多個 decode 同時跑時 GPU util 常 < 5%，是磁碟瓶頸，不是 CPU fallback（`AesPredictor` 在 `torch.cuda.is_available()` 時用 cuda，且 score_r2 行程佔 VRAM 7 GB）。
- AIME src 字串 `aime:<shard>:<row>:<H:M:S>:<H:M:S>` 含冒號，解析要用 `parts[3:6]`／`parts[6:9]`。
