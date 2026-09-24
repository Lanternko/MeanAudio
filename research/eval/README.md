# research/eval

Moved from `~/research/meanaudio_eval/` on 2026-09-24（舊路徑是指向這裡的 symlink）。

| 檔案 | 用途 |
|---|---|
| `phase4_eval.py` | 舊版 CLAP/AES metric script。**凍結**：歷史 contract 以 `/home/kojiek/research/meanaudio_eval/phase4_eval.py` 綁 sha，不可修改。新 eval 用 `scripts/eval/eval_metrics.py` |
| `peav_eval.py` | PE-AV 評估（`~/venvs/peav`，見 memory `reference_peav_eval_setup.md`） |
| `audiobox_eval.py` | Audiobox Aesthetics 評分 |
| `music_flamingo_*` | Music Flamingo caption 對照 |
| `meanaudio_rtf_benchmark*.py`、`test_*rtx5090*.py` | RTF／硬體相容性測試（2026 初期） |
| `core/`、`bagpipe_analysis/`、`output/` | 2026 初期可行性評估（LP-MusicCaps bagpipes 偏差）；結論已過時，只作歷史參考 |
