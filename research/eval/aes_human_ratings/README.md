# Audiobox Aesthetics 公開人類評分

來源：https://github.com/facebookresearch/audiobox-aesthetics （`evaluation_data/`、`audiomos2025_track2/`），CC-BY 4.0，2026-09-29 下載。
只有分數與 `data_path`，音檔要自己從原資料集取得。每筆 4 軸 × 10 位標註者原始分數。

- `AES_natural_music.jsonl`：MusicCaps 549（全部在 `musiccaps_test.tsv`，522 在 `/mnt/HDD/kojiek/musiccaps_reference`）＋MUSDB18 test 451（stems）
- `AES_natural_sound.jsonl`：AudioSet 1000；`AES_natural_speech.jsonl`：CV13/LibriTTS/EARS 950
- `AES_PAM.jsonl`：PAM human_eval（zenodo 10737388），含 4 個 TTM 系統＋real 各 100
- `audiomos2025_track2/`：AES-natural 重切 train 2700／dev 250；eval 3060 筆為 36 個匿名生成系統

分析：`research/eval/aes_human_corr.py`；說明見 `docs/metrics/audiobox_aesthetics.md`。
