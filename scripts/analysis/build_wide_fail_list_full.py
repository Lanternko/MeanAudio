#!/usr/bin/env python3
"""Build the full wide-criterion failure list page.

Reads only existing artifacts:
  - docs/experiments/results/wide_fail_list_126ab_20261018.html  (D / ABS / NOTES)
  - docs/experiments/results/metric_win_tie_126c_summary.json
  - docs/experiments/results/metric_win_tie_126d_summary.json
  - docs/experiments/results/metric_win_tie_133_summary.json
  - docs/experiments/results/speed_pitch_extension_134_summary.json
  - eval_output_nvme/metric_win_tie_133_20261020/**/mert_cosim.tsv, muq_cosim.tsv
  - eval_output_nvme/metric_win_tie_126{b,c}_*/stimuli/items.json

Writes:
  - docs/experiments/results/wide_fail_list_full_20261022.html
  - docs/experiments/results/wide_fail_list_full_20261022.md

The page is a single self-contained HTML file (no external resources).
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "experiments" / "results"
BASE_HTML = RESULTS / "wide_fail_list_126ab_20261018.html"
S126C = RESULTS / "metric_win_tie_126c_summary.json"
S126D = RESULTS / "metric_win_tie_126d_summary.json"
S133 = RESULTS / "metric_win_tie_133_summary.json"
S134 = RESULTS / "speed_pitch_extension_134_summary.json"
OUT_HTML = RESULTS / "wide_fail_list_full_20261022.html"
OUT_MD = RESULTS / "wide_fail_list_full_20261022.md"

EV = Path.home() / "eval_output_nvme"
EV133 = EV / "metric_win_tie_133_20261020"
ITEMS_126B = EV / "metric_win_tie_126b_20261016" / "stimuli" / "items.json"
ITEMS_126C = EV / "metric_win_tie_126c_20261019" / "stimuli" / "items.json"

M = 0.10
PROTO_ORDER = ["126a", "126b", "126c", "126d", "133", "134"]
SYSTEMS = ["real", "ours_cfg3neg", "ours_cfg0", "musicgen", "musicldm"]

ALLM = [
    "PQ", "CE", "CU", "meva", "clap",
    "pulse_clarity", "key_cnn_conf", "ibi_cv", "chroma_entropy",
    "se_coh", "se_mem", "se_nat", "se_cla", "se_mus",
    "mert", "muq",
]
MN = {
    "PQ": "PQ", "CE": "CE", "CU": "CU", "PC": "PC", "meva": "MEva", "clap": "CLAP",
    "pulse_clarity": "pulse", "key_cnn_conf": "key", "ibi_cv": "ibi_cv",
    "chroma_entropy": "chroma",
    "se_coh": "SE-Coh", "se_mem": "SE-Mem", "se_nat": "SE-Nat",
    "se_cla": "SE-Cla", "se_mus": "SE-Mus",
    "mert": "MERT", "muq": "MuQ",
}
SONGEVAL = ["se_coh", "se_mem", "se_nat", "se_cla", "se_mus"]

GN = OrderedDict([
    ("126a-tie", "126a｜該平手：訊號層"),
    ("126b-lose", "126b｜該變差：訊號層"),
    ("126b-note", "126b｜該變差：音符層（鋼琴 MIDI）"),
    ("126b-soft", "126b｜不重要差異：該平手"),
    ("126b-rec", "126b｜真人演奏"),
    ("126b-floor", "126b｜底線訊號"),
    ("126c-tie", "126c｜MUSDB18-HQ：該平手／soft"),
    ("126c-lose", "126c｜MUSDB18-HQ：該變差"),
    ("126d-126a", "126d｜SongEval × 126a 刺激（訊號層）"),
    ("126d-126b", "126d｜SongEval × 126b 刺激（名曲 MIDI／真人）"),
    ("126d-126c", "126d｜SongEval × 126c 刺激（MUSDB 分軌）"),
    ("133-126a", "133｜MERT × 126a 刺激（5 個系統）"),
    ("133-126b", "133｜MERT／MuQ × 126b 刺激"),
    ("133-126c", "133｜MERT／MuQ × 126c 刺激"),
    ("134", "134｜速度／音高延伸（MAESTRO／126b 鋼琴／TTM）"),
])

SRCNAME = {
    "musdb": "MUSDB18-HQ 全體（551 窗）",
    "songeval": "SongEval 全體",
    "all": "全體 clips（單一聚合）",
    "maestro_midi": "MAESTRO·MIDI",
    "maestro_rb": "MAESTRO·rubberband",
    "piano126b_midi": "126b 鋼琴·MIDI",
    "piano126b_rb": "126b 鋼琴·rubberband",
    "ttm_rb": "TTM·rubberband",
}

FAM_TITLE = {
    "gain": "音量增減", "lead": "開頭平移", "mp3": "MP3 壓縮",
    "misc": "極性／重取樣／16 bit",
    "noise": "白噪音", "clip": "削波", "bitcrush": "位元壓低",
    "clicks": "爆音點", "dropout": "20 ms 斷音", "lowpass": "低通",
    "blast": "80 ms 爆衝", "shuffle": "切段打亂／倒放",
    "wrong": "錯音", "randpitch": "隨機音高／音高打亂",
    "jitter": "時間抖動", "unsteady": "速度不穩",
    "posshuffle": "音符位置／段落打亂", "dropextra": "掉音／多音",
    "blastnote": "單音爆大聲／力度亂",
    "soft": "不重要差異（移調／速度／力度）", "swap": "換 SoundFont",
    "cross": "真人演奏：poor 對 pro", "floors": "底線訊號 對 最差的音樂系統",
}

FAM_WHAT = {
    "gain": "整段乘上同一個倍數，波形形狀完全不變，只有音量變。應該平手。",
    "lead": "開頭插入一段靜音、尾巴切掉同樣長度，總長不變。應該平手。",
    "mp3": "整段壓成 MP3 再解回來。應該平手（低碼率是否該平手有爭議）。",
    "misc": "極性反轉、重取樣 44.1 kHz、PCM 16、原檔重排。應該平手。",
    "noise": "整段疊白噪音後把整體音量調回原檔。應該變差。",
    "clip": "把峰值以上削平，削完把整體音量調回原檔。應該變差。",
    "bitcrush": "把振幅量化成少量階，加完把音量調回原檔。應該變差。",
    "clicks": "在隨機位置插入單一樣本爆音點。應該變差。",
    "dropout": "隨機挖掉 20 ms 音訊。應該變差。",
    "lowpass": "低通濾波。應該變差。",
    "blast": "插入 80 ms 爆衝。應該變差。",
    "shuffle": "切段打亂或整段倒放。應該變差。",
    "wrong": "把音符改成錯的音高。應該變差。",
    "randpitch": "音高隨機抽取或在時間上打亂。應該變差。",
    "jitter": "音符時間抖動。應該變差。",
    "unsteady": "速度不穩。應該變差。",
    "posshuffle": "音符位置或段落打亂／逆行。應該變差。",
    "dropextra": "掉音或多出音。應該變差。",
    "blastnote": "單音爆大聲或力度隨機。應該變差。",
    "soft": "移調 ±1 半音、速度 ±5%、力度 ×0.85。應該平手。",
    "swap": "同一 MIDI 換 SoundFont。應該平手。",
    "cross": "真人演奏 poor 對 pro，Δ = poor − pro。應該變差。",
    "floors": "底線訊號對五個 TTM 系統中最低的乾淨平均。應該變差。",
}

C126_TITLE = {
    "bass_semi": "貝斯移調 +1 半音",
    "clicks": "爆音點（5/s）",
    "clip": "削波 6 dB",
    "drums_lag": "鼓延遲",
    "gain_down": "整體音量衰減",
    "identity": "分軌相加（stemsum）",
    "lead": "開頭平移 10 ms",
    "lowpass": "低通 4 kHz",
    "mp3": "MP3 64 kbps",
    "noise": "白噪音 SNR",
    "pcm16": "PCM 16",
    "polarity": "極性反轉",
    "separation": "htdemucs 重新相加（dmx_resum）",
    "swap": "換軌（鼓／人聲）",
    "vocal_pv0": "人聲 phase-vocoder 處理",
    "vocals_detune": "人聲走音",
    "vocals_lag": "人聲延遲",
}
for stem in ("vocals", "drums", "bass", "other"):
    C126_TITLE[f"rebal_{stem}_up"] = f"{stem} rebalance 提升（+dB）"
    C126_TITLE[f"rebal_{stem}_down"] = f"{stem} rebalance 衰減（−dB）"

C126_WHAT = {
    "bass_semi": "MUSDB 貝斯軌移調 +1 半音。應該變差。",
    "clicks": "在混音中插入每秒 5 個爆音點。應該變差。",
    "clip": "整段削波 6 dB。應該變差。",
    "drums_lag": "鼓軌相對其他軌延遲。應該變差。",
    "gain_down": "整段衰減 1／3 dB。應該平手。",
    "identity": "把分軌加回成混音。應該平手。",
    "lead": "開頭平移 10 ms。應該平手。",
    "lowpass": "低通 4 kHz。應該變差。",
    "mp3": "壓成 MP3 64 kbps。應該平手。",
    "noise": "疊白噪音。應該變差。",
    "pcm16": "存成 PCM 16。應該平手。",
    "polarity": "波形上下翻轉。應該平手。",
    "separation": "htdemucs 去軌後重新相加。應該平手。",
    "swap": "換掉鼓或人聲軌。應該變差。",
    "vocal_pv0": "人聲 phase-vocoder 處理。soft，該平手。",
    "vocals_detune": "人聲軌音高偏移 25／50／100 cents。應該變差。",
    "vocals_lag": "人聲軌延遲。應該變差。",
}
for stem in ("vocals", "drums", "bass", "other"):
    C126_WHAT[f"rebal_{stem}_up"] = f"{stem} 軌增益 +1／+3／+6 dB。±1 dB 該平手，±3/6 dB soft。"
    C126_WHAT[f"rebal_{stem}_down"] = f"{stem} 軌增益 −1／−3／−6 dB。±1 dB 該平手，±3/6 dB soft。"

# Canonical step order (names are cell / variant names).
CANON = {
    "gain": ["gain-30", "gain-20", "gain-15", "gain-10", "gain-6", "gain-3",
             "gain-2", "gain-1", "gain-0.5", "gain+1", "gain+3", "gain+6"],
    "lead": ["lead10ms", "lead50ms", "lead200ms"],
    "mp3": ["mp3_160k", "mp3_64k", "mp3_32k"],
    "misc": ["polarity", "resample44k", "pcm16", "clean_shuf"],
    "noise": ["noise_snr50", "noise_snr40", "noise_snr30", "noise_snr25",
              "noise_snr20", "noise_snr15", "noise_snr10", "noise_snr5", "noise_snr0"],
    "clip": ["clip3db", "clip6db", "clip12db", "clip20db"],
    "bitcrush": ["bitcrush8", "bitcrush6", "bitcrush4"],
    "clicks": ["clicks1ps", "clicks5ps", "clicks20ps"],
    "dropout": ["dropout20ms_x3", "dropout20ms_x10", "dropout20ms_x30"],
    "lowpass": ["lowpass4k", "lowpass2k", "lowpass1k"],
    "blast": ["blast80ms_x1", "blast80ms_x3"],
    "shuffle": ["shuffle1.0s", "shuffle0.25s", "reverse"],
    "wrong": ["wrong1_semi", "wrong1_tritone", "wrong5pct", "wrong10pct",
              "wrong30pct", "wrong100pct"],
    "randpitch": ["randpitch10pct", "randpitch30pct", "randpitch100pct",
                  "pitchshuffle2s", "pitchshuffle_all"],
    "jitter": ["jitter10ms", "jitter30ms", "jitter80ms", "jitter200ms"],
    "unsteady": ["unsteady5pct", "unsteady15pct", "unsteady30pct"],
    "posshuffle": ["posshuffle10pct", "posshuffle30pct", "posshuffle100pct",
                   "blockshuffle3.0s", "blockshuffle1.5s", "blockshuffle0.5s",
                   "retrograde"],
    "dropextra": ["drop10pct", "drop30pct", "drop60pct", "extra10pct", "extra30pct"],
    "blastnote": ["blastnote_x1", "blastnote_x3", "blastnote_x10", "vel_random"],
    "soft": ["transpose-1", "transpose+1", "tempo_x0.95", "tempo_x0.98",
             "tempo_x1.02", "tempo_x1.05", "vel_x0.85"],
    "swap": ["fluidr3_vs_generaluser"],
    "cross": ["pro_vs_poor"],
    "floors": ["binary_01_hold_lvl30", "binary_01_lvl30", "binary_pm1_lvl30",
               "dc_lvl30", "pulse10ps_lvl30", "silence_lvl30", "sine1k_lvl30",
               "square100_lvl30", "square1k_lvl30", "sweep_lvl30", "white_lvl30"],
}

CANON_126C = {
    "bass_semi": ["bass_semi+1"],
    "clicks": ["clicks5ps"],
    "clip": ["clip6db"],
    "drums_lag": ["drums_lag20ms", "drums_lag50ms", "drums_lag100ms", "drums_lag250ms"],
    "gain_down": ["gain-1", "gain-3"],
    "identity": ["stemsum"],
    "lead": ["lead10ms"],
    "lowpass": ["lowpass4k"],
    "mp3": ["mp3_64k"],
    "noise": ["noise_snr40", "noise_snr30", "noise_snr20", "noise_snr10"],
    "pcm16": ["pcm16"],
    "polarity": ["polarity"],
    "separation": ["dmx_resum"],
    "swap": ["swap_drums", "swap_vocals"],
    "vocal_pv0": ["vocal_pv0"],
    "vocals_detune": ["vocals_detune25c", "vocals_detune50c", "vocals_detune100c"],
    "vocals_lag": ["vocals_lag50ms", "vocals_lag100ms", "vocals_lag250ms"],
}
for stem in ("vocals", "drums", "bass", "other"):
    CANON_126C[f"rebal_{stem}_up"] = [f"rebal_{stem}_+1dB", f"rebal_{stem}_+3dB",
                                      f"rebal_{stem}_+6dB"]
    CANON_126C[f"rebal_{stem}_down"] = [f"rebal_{stem}_-1dB", f"rebal_{stem}_-3dB",
                                        f"rebal_{stem}_-6dB"]

ORDER_126A = ["gain", "lead", "mp3", "misc", "noise", "clip", "bitcrush",
              "clicks", "dropout", "lowpass", "blast", "shuffle"]
ORDER_NOTE = ["wrong", "randpitch", "jitter", "unsteady", "posshuffle",
              "dropextra", "blastnote", "soft", "swap"]
ORDER_126B = ["noise", "clip", "bitcrush", "clicks", "dropout", "lowpass",
              "blast", "shuffle", "wrong", "randpitch", "jitter", "unsteady",
              "posshuffle", "dropextra", "blastnote", "soft", "swap"]
ORDER_126B_ALL = ORDER_126A + ORDER_NOTE
ORDER_126C = [
    "gain_down", "lead", "mp3", "pcm16", "polarity", "identity",
    "noise", "clip", "lowpass", "clicks", "separation", "swap",
    "vocal_pv0", "drums_lag", "vocals_lag", "vocals_detune",
    "rebal_vocals_up", "rebal_vocals_down", "rebal_drums_up", "rebal_drums_down",
    "rebal_bass_up", "rebal_bass_down", "rebal_other_up", "rebal_other_down",
    "bass_semi",
]

B_LADDER_MAP = {
    "gain_down": "gain", "gain_up": "gain", "lead": "lead", "mp3": "mp3",
    "polarity": "misc", "pcm16": "misc", "resample44k": "misc", "clean_shuf": "misc",
    "noise": "noise", "clip": "clip", "bitcrush": "bitcrush", "clicks": "clicks",
    "dropout": "dropout", "lowpass": "lowpass", "blast": "blast",
    "shuffle": "shuffle", "reverse": "shuffle",
    "wrong": "wrong", "randpitch": "randpitch", "pitchshuffle": "randpitch",
    "jitter": "jitter", "unsteady": "unsteady",
    "posshuffle": "posshuffle", "blockshuffle": "posshuffle", "structure": "posshuffle",
    "drop": "dropextra", "extra": "dropextra",
    "blastnote": "blastnote", "dynamics": "blastnote",
    "tempo": "soft", "transpose": "soft", "dynamics_soft": "soft",
    "swap": "swap",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def r4(v):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return round(v, 4)


def f3(v):
    return None if v is None else round(float(v), 3)


def kind_class(k):
    return {"tie": "tie", "soft": "tie", "lose": "lose", "note": "lose"}.get(k)


def is_fail(kc, v):
    if v is None:
        return False
    return abs(v) > M if kc == "tie" else v >= -M


def count_fails(kc, vals):
    n = sum(1 for v in vals if v is not None)
    k = sum(1 for v in vals if v is not None and is_fail(kc, v))
    return k, n


def cell_fail(kc, md, metrics, srcs):
    n = k = 0
    for m in metrics:
        for s in srcs:
            for v in md[m].get(s, []):
                if v is None:
                    continue
                n += 1
                k += int(is_fail(kc, v))
    return k, n


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return statistics.fmean(vals) if vals else None


def read_col(path, col):
    vals = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                vals.append(float(row[col]))
            except (KeyError, ValueError):
                pass
    return vals


def btw_from_deltas(deltas, m):
    if not deltas:
        return None
    w = t = b = 0
    for d in deltas:
        if d < -m:
            w += 1
        elif d > m:
            b += 1
        else:
            t += 1
    n = len(deltas)
    return [round(w / n, 3), round(t / n, 3), round(b / n, 3)]


def load_base():
    s = BASE_HTML.read_text(encoding="utf-8")
    out = {}
    for key in ("D", "ABS", "NOTES"):
        m = re.search(rf"^const {key} = (.*)$", s, re.M)
        if not m:
            raise SystemExit(f"cannot find const {key} in {BASE_HTML}")
        out[key] = json.loads(m.group(1).rstrip(";"))
    return out, s


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── labels / ordering ────────────────────────────────────────────────────────

def label_cell(c, fam):
    if fam == "gain":
        return c[4:].replace("-", "−") + " dB"
    if fam == "lead":
        return c[4:].replace("ms", "") + " ms"
    if fam == "mp3":
        return c.split("_")[1]
    if fam == "misc":
        return {"polarity": "極性反轉", "resample44k": "重取樣 44.1k",
                "pcm16": "PCM 16", "clean_shuf": "原檔重排"}.get(c, c)
    if fam == "noise":
        return c.split("snr")[-1]
    if fam == "clip":
        return c[4:-2] + " dB"
    if fam == "bitcrush":
        return c[8:] + " bit"
    if fam == "clicks":
        return c[6:-2] + "/s"
    if fam == "dropout":
        return "×" + c.split("_x")[-1]
    if fam == "lowpass":
        return c[7:-1] + "k"
    if fam == "blast":
        return "×" + c.split("_x")[-1]
    if fam == "shuffle":
        if c.startswith("shuffle"):
            return c[7:-1] + " s 段"
        return {"reverse": "倒放"}.get(c, c)
    if fam == "wrong":
        return {"wrong1_semi": "1 個半音", "wrong1_tritone": "1 個三全音"}.get(
            c, c.replace("wrong", "").replace("pct", "%"))
    if fam == "randpitch":
        if c.startswith("randpitch"):
            return "隨機 " + c[9:].replace("pct", "%")
        return {"pitchshuffle2s": "2 s 內打亂", "pitchshuffle_all": "全曲打亂"}.get(c, c)
    if fam in ("jitter",):
        return c[6:] + " ms"
    if fam == "unsteady":
        return c[8:].replace("pct", "%")
    if fam == "posshuffle":
        if c.startswith("posshuffle"):
            return "位置 " + c[10:].replace("pct", "%")
        if c.startswith("blockshuffle"):
            return c[12:-1] + " s 區塊"
        return {"retrograde": "逆行"}.get(c, c)
    if fam == "dropextra":
        if c.startswith("drop"):
            return "掉 " + c[4:].replace("pct", "%")
        return "多 " + c[5:].replace("pct", "%")
    if fam == "blastnote":
        if c == "vel_random":
            return "力度隨機"
        return "爆 ×" + c.split("_x")[-1]
    if fam == "soft":
        if c.startswith("transpose"):
            return "移調 " + c[9:].replace("-", "−")
        if c.startswith("tempo_x"):
            return "速度 " + c[7:]
        if c.startswith("vel_x"):
            return "力度 ×" + c[5:]
        return c
    if fam == "swap":
        return "FluidR3 → GeneralUser"
    if fam == "cross":
        return "poor − pro"
    if fam == "floors":
        return c.replace("_lvl30", "")
    # 126c
    if fam == "gain_down":
        return c[4:].replace("-", "−") + " dB"
    if fam in ("bass_semi",):
        return "+1 半音"
    if fam in ("clicks", "clip", "lowpass", "mp3", "pcm16", "polarity", "identity",
               "separation", "vocal_pv0"):
        return {"clicks": "5/s", "clip": "6 dB", "lowpass": "4k", "mp3": "64k",
                "pcm16": "PCM 16", "polarity": "極性反轉",
                "identity": "stemsum", "separation": "dmx_resum",
                "vocal_pv0": "pv0"}.get(fam, c)
    if fam == "noise":
        return "SNR " + c.split("snr")[-1]
    if fam in ("drums_lag", "vocals_lag"):
        return c.split("_lag")[-1].replace("ms", "") + " ms"
    if fam == "vocals_detune":
        return c.split("detune")[-1].replace("c", "") + " cents"
    if fam.startswith("rebal_"):
        return c.split("rebal_")[-1].replace("dB", " dB").replace("-", "−").replace("_", " ")
    if fam == "swap":
        return {"swap_drums": "換鼓", "swap_vocals": "換人聲"}.get(c, c)
    return c


def order_cells(cells, fam, canon_map=None):
    canon = (canon_map or CANON).get(fam)
    if canon:
        order = {c: i for i, c in enumerate(canon)}
        known = [c for c in cells if c in order]
        unknown = sorted(c for c in cells if c not in order)
        return sorted(known, key=lambda c: order[c]) + unknown
    # fallback: numeric prefix
    def key(c):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", c)
        return float(m.group(0)) if m else 0.0
    return sorted(cells, key=key)


# ── family construction ─────────────────────────────────────────────────────

def make_family(fid, proto, group, title, kc, srcs, metrics, steps,
                md, btw=None, absv=None, variants=None, unit="", what=""):
    nfail, ntot = cell_fail(kc, md, metrics, srcs)
    fam = {
        "id": fid, "proto": proto, "group": group, "title": title,
        "kind": kc, "unit": unit, "steps": steps, "variants": variants or steps,
        "srcs": srcs, "metrics": metrics, "md": md, "what": what,
        "nfail": nfail, "ntot": ntot,
    }
    if btw:
        fam["btw"] = btw
    if absv:
        fam["ABS"] = absv
    return fam


def auto_note(fam):
    kc = fam["kind"]
    out = [f"沒過 {fam['nfail']}/{fam['ntot']} 點（Δ/S 用 {fam['proto']} 自己的 S）。"]
    per = []
    for m in fam["metrics"]:
        vals = [v for s in fam["srcs"] for v in fam["md"][m].get(s, []) if v is not None]
        if not vals:
            continue
        k = sum(1 for v in vals if is_fail(kc, v))
        per.append((k, len(vals), m, vals))
    if per:
        per.sort(key=lambda x: (-x[0], x[2]))
        out.append("各指標沒過：" + "、".join(
            f"{MN[m]} {k}/{n}" for k, n, m, _ in per[:8]) + "。")
        worst = None
        for k, n, m, vals in per:
            v = min(vals) if kc == "lose" else max(vals, key=abs)
            if worst is None or (abs(v) > abs(worst[1])):
                worst = (m, v)
        if worst:
            mv, vv = worst
            out.append(f"最大 |Δ/S|：{MN[mv]} {vv:+.3f}。")
    return {"sev": "proto", "tag": "新協定",
            "what": fam.get("what", ""), "lines": out}


# ── loaders ─────────────────────────────────────────────────────────────────

def build_126c(items):
    d = load_json(S126C)
    S = d["protocol"]["S"]
    case_meta = {}
    for r in items:
        case_meta.setdefault(r["case"], r)
    by_ladder = defaultdict(list)
    for r in items:
        if r["case"] != "mix":
            by_ladder[r["ladder"]].append(r["case"])
    fams = []
    for ladder in ORDER_126C:
        cases = []
        for c in by_ladder.get(ladder, []):
            if c not in cases:
                cases.append(c)
        if not cases:
            continue
        groups = defaultdict(list)
        for c in cases:
            kc = kind_class(case_meta[c]["kind"])
            if kc:
                groups[kc].append(c)
        for kc in ("tie", "lose"):
            cs = groups.get(kc)
            if not cs:
                continue
            cs = order_cells(cs, ladder, CANON_126C)
            metrics = [m for m in ALLM
                       if m != "PC" and any(m in d["cells"][c]["metrics"] for c in cs)]
            md = {m: {"musdb": []} for m in metrics}
            btw = {m: {"musdb": []} for m in metrics}
            for c in cs:
                cell = d["cells"][c]
                for m in metrics:
                    rec = cell["metrics"].get(m)
                    if rec is None:
                        md[m]["musdb"].append(None)
                        btw[m]["musdb"].append(None)
                        continue
                    md[m]["musdb"].append(r4(rec["mean_delta"] / S[m]))
                    at = rec.get("at_m") or {}
                    btw[m]["musdb"].append(
                        [f3(at.get("worse")), f3(at.get("tie")), f3(at.get("better"))]
                        if at else None)
            title = C126_TITLE.get(ladder, ladder)
            if len(groups) > 1:
                title += "（該平手）" if kc == "tie" else "（該變差）"
            fid = f"126c-{ladder}-{kc}"
            fam = make_family(fid, "126c", f"126c-{kc}", title, kc, ["musdb"],
                              metrics, [label_cell(c, ladder) for c in cs],
                              md, btw=btw, variants=cs,
                              what=C126_WHAT.get(ladder, ""))
            fams.append(fam)
    return fams


def build_126d(items_by_sec, meta126c):
    d = load_json(S126D)
    S = d["S_songeval"]
    meta_b = {}
    for r in items_by_sec["126b"]:
        meta_b.setdefault(r["case"], r)
    fams = []
    for sec in ("126a", "126b", "126c"):
        cells = d[sec]
        groups = defaultdict(list)
        famkeys = {}
        for c, cell in cells.items():
            kc = kind_class(cell.get("kind"))
            if not kc:
                continue
            if sec == "126a":
                fk = fam126a(c)
            elif sec == "126b":
                fk = B_LADDER_MAP.get(meta_b.get(c, {}).get("ladder"))
            else:
                fk = meta126c.get(c, {}).get("ladder")
            if not fk:
                continue
            famkeys[c] = fk
            groups[(fk, kc)].append(c)
        order = ORDER_126A if sec == "126a" else (
            ORDER_126B_ALL if sec == "126b" else ORDER_126C)
        for fk in order:
            for kc in ("tie", "lose"):
                cs = groups.get((fk, kc))
                if not cs:
                    continue
                canon = CANON_126C if sec == "126c" else CANON
                cs = order_cells(cs, fk, canon)
                metrics = [m for m in SONGEVAL if m in cells[cs[0]]]
                md = {m: {"songeval": []} for m in metrics}
                for c in cs:
                    for m in metrics:
                        md[m]["songeval"].append(r4(cells[c][m]["mean_d"] / S[m]))
                if sec == "126c":
                    base_title = C126_TITLE.get(fk, fk)
                else:
                    base_title = FAM_TITLE.get(fk, fk)
                title = base_title + ("（該平手）" if kc == "tie" else "（該變差）") \
                    if sum(1 for (k, kk) in groups if k == fk) > 1 else base_title
                fid = f"126d-{sec}-{fk}-{kc}"
                fam = make_family(fid, "126d", f"126d-{sec}", title, kc,
                                  ["songeval"], metrics,
                                  [label_cell(c, fk) for c in cs], md, variants=cs,
                                  what=(C126_WHAT if sec == "126c" else FAM_WHAT).get(fk, ""))
                fams.append(fam)
    return fams


def fam126a(c):
    if c.startswith("gain"):
        return "gain"
    if c.startswith("lead"):
        return "lead"
    if c.startswith("mp3"):
        return "mp3"
    if c in ("polarity", "pcm16", "resample44k", "clean_shuf"):
        return "misc"
    if c.startswith("noise_snr"):
        return "noise"
    if c.startswith("clip"):
        return "clip"
    if c.startswith("bitcrush"):
        return "bitcrush"
    if c.startswith("clicks"):
        return "clicks"
    if c.startswith("dropout"):
        return "dropout"
    if c.startswith("lowpass"):
        return "lowpass"
    if c.startswith("blast80ms"):
        return "blast"
    if c.startswith("shuffle") or c == "reverse":
        return "shuffle"
    return None


def build_133(items_by_sec, meta126c):
    d = load_json(S133)
    S = {"mert": d["S_mert"], "muq": d["S_muq"]}
    # 126a per-clip files
    a_stats = {}
    for vdir in sorted(p for p in (EV133 / "126a").iterdir()
                       if p.is_dir() and p.name != "clean"):
        sys_stats = {}
        for sysname in SYSTEMS:
            p = vdir / sysname / "mert_cosim.tsv"
            if not p.is_file():
                continue
            vals = read_col(p, "mert_cosim")
            if not vals:
                continue
            delta = mean(vals) - 1.0
            sys_stats[sysname] = (delta, len(vals), vals)
            # cross-check against summary mean_cosim
            ref = d["126a"].get(vdir.name, {}).get(sysname, {}).get("mean_cosim")
            if ref is not None:
                assert abs((delta + 1.0) - ref) < 2e-6, (vdir.name, sysname, delta + 1, ref)
        if sys_stats:
            a_stats[vdir.name] = sys_stats

    fams = []
    groups = defaultdict(list)
    for v, st in a_stats.items():
        fk = fam126a(v)
        if not fk:
            continue
        kc = "tie" if fk in ("gain", "lead", "mp3", "misc") else "lose"
        groups[(fk, kc)].append(v)
    for fk in ORDER_126A:
        for kc in ("tie", "lose"):
            vs = groups.get((fk, kc))
            if not vs:
                continue
            vs = order_cells(vs, fk, CANON)
            md = {"mert": {s: [] for s in SYSTEMS}}
            btw = {"mert": {s: [] for s in SYSTEMS}}
            absv = {"mert": {s: [] for s in SYSTEMS}}
            for v in vs:
                st = a_stats[v]
                for s in SYSTEMS:
                    if s not in st:
                        continue
                    delta, n, _ = st[s]
                    md["mert"][s].append(r4(delta / S["mert"]))
                    absv["mert"][s].append([1.0, round(1.0 + delta, 6)])
                for s in SYSTEMS:
                    if s not in st:
                        continue
                    _, _, vals = st[s]
                    btw["mert"][s].append(btw_from_deltas([x - 1.0 for x in vals],
                                                          M * S["mert"]))
            title = FAM_TITLE.get(fk, fk) + ("（該平手）" if kc == "tie" else "（該變差）") \
                if len({k for (k, _) in groups if k == fk}) > 1 else FAM_TITLE.get(fk, fk)
            fid = f"133-126a-{fk}-{kc}"
            fam = make_family(fid, "133", "133-126a", title, kc, SYSTEMS,
                              ["mert"], [label_cell(v, fk) for v in vs], md,
                              btw=btw, absv=absv, variants=vs,
                              what=FAM_WHAT.get(fk, ""))
            fams.append(fam)

    for sec in ("126b", "126c"):
        tsv_m = EV133 / f"{sec}_mert_cosim.tsv"
        tsv_q = EV133 / f"{sec}_muq_cosim.tsv"
        cm = read_col(tsv_m, "mert_cosim")
        cq = read_col(tsv_q, "muq_cosim")
        # map key -> value
        def read_keyed(path, col):
            out = {}
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    try:
                        out[row["key"]] = float(row[col])
                    except (KeyError, ValueError):
                        pass
            return out
        km = read_keyed(tsv_m, "mert_cosim")
        kq = read_keyed(tsv_q, "muq_cosim")
        items = items_by_sec[sec]
        if isinstance(items, dict):
            items = list(items.values())
        case_vals = defaultdict(lambda: {"mert": [], "muq": []})
        case_meta = {}
        for r in items:
            case_meta.setdefault(r["case"], r)
            if not r.get("base"):
                continue
            if r["key"] in km:
                case_vals[r["case"]]["mert"].append(km[r["key"]])
            if r["key"] in kq:
                case_vals[r["case"]]["muq"].append(kq[r["key"]])
        groups = defaultdict(list)
        for case, cell in case_meta.items():
            if case == "mix" or cell.get("kind") in ("ref", "floor"):
                continue
            kc = kind_class(cell.get("kind"))
            if not kc:
                continue
            fk = B_LADDER_MAP.get(cell.get("ladder")) if sec == "126b" else cell.get("ladder")
            if not fk:
                continue
            groups[(fk, kc)].append(case)
        order = ORDER_126B_ALL if sec == "126b" else ORDER_126C
        canon = CANON if sec == "126b" else CANON_126C
        for fk in order:
            for kc in ("tie", "lose"):
                cs = groups.get((fk, kc))
                if not cs:
                    continue
                cs = order_cells(cs, fk, canon)
                metrics = ["mert", "muq"]
                md = {m: {"all": []} for m in metrics}
                btw = {m: {"all": []} for m in metrics}
                absv = {m: {"all": []} for m in metrics}
                for c in cs:
                    for m, dim in (("mert", "mert"), ("muq", "muq")):
                        vals = case_vals[c][dim]
                        if not vals:
                            md[m]["all"].append(None)
                            btw[m]["all"].append(None)
                            absv[m]["all"].append(None)
                            continue
                        delta = mean(vals) - 1.0
                        md[m]["all"].append(r4(delta / S[m]))
                        absv[m]["all"].append([1.0, round(1.0 + delta, 6)])
                        btw[m]["all"].append(btw_from_deltas(
                            [x - 1.0 for x in vals], M * S[m]))
                base_title = C126_TITLE.get(fk, fk) if sec == "126c" else FAM_TITLE.get(fk, fk)
                title = base_title + ("（該平手）" if kc == "tie" else "（該變差）") \
                    if len({k for (k, _) in groups if k == fk}) > 1 else base_title
                fid = f"133-{sec}-{fk}-{kc}"
                fam = make_family(fid, "133", f"133-{sec}", title, kc, ["all"],
                                  metrics, [label_cell(c, fk) for c in cs], md,
                                  btw=btw, absv=absv, variants=cs,
                                  what=(C126_WHAT if sec == "126c" else FAM_WHAT).get(fk, ""))
                fams.append(fam)
        if sec == "126b":
            # floor cells have base='' -> MERT/MuQ null; keep as grey rows
            flo = [c for c, r in case_meta.items() if r.get("kind") == "floor"
                   and r.get("level") == "lvl30"]
            flo = order_cells(flo, "floors", CANON)
            md = {"mert": {"all": [None] * len(flo)},
                  "muq": {"all": [None] * len(flo)}}
            fam = make_family("133-126b-floor-null", "133", "133-126b",
                              "底線訊號（null：沒有 base，133 無定義）", "lose",
                              ["all"], ["mert", "muq"],
                              [label_cell(c, "floors") for c in flo], md,
                              variants=flo,
                              what="底線訊號沒有自己的 base，133 的 cosim 距離沒有定義；"
                                   "這一列保留成灰色，不計入失敗。")
            fam["null_family"] = True
            fams.append(fam)
    return fams


def build_134():
    d = load_json(S134)
    S = d["protocol"]["S"]
    fams = []
    groups = defaultdict(list)
    meta = {}
    for key, cell in d["cells"].items():
        m = re.match(r"^([^/]+)/([^/]+)/(pitch|tempo)_(?:midi|rb)_(.+)$", key)
        if not m:
            continue
        source, method, param, val = m.groups()
        kc = kind_class(cell.get("kind"))
        if not kc:
            continue
        fk = (source, method, param, kc)
        groups[fk].append((float(val), key))
        meta[key] = (source, method, param, kc, float(val))
    src_label = {
        ("maestro", "midi"): "maestro_midi", ("maestro", "rb"): "maestro_rb",
        ("piano126b", "midi"): "piano126b_midi", ("piano126b", "rb"): "piano126b_rb",
        ("ttm", "rb"): "ttm_rb",
    }
    order_src = [("maestro", "midi"), ("maestro", "rb"),
                 ("piano126b", "midi"), ("piano126b", "rb"), ("ttm", "rb")]
    metrics = ["PQ", "CE", "CU", "meva", "clap", "pulse_clarity", "ibi_cv"]
    for source, method in order_src:
        for param in ("pitch", "tempo"):
            for kc in ("tie", "lose"):
                items = sorted(groups.get((source, method, param, kc), []))
                if not items:
                    continue
                keys = [k for _, k in items]
                md = {m: {src_label[(source, method)]: []} for m in metrics
                      if m in d["cells"][keys[0]]["metrics"]}
                btw = {m: {src_label[(source, method)]: []} for m in md}
                for k in keys:
                    cell = d["cells"][k]
                    for m in md:
                        rec = cell["metrics"][m]
                        md[m][src_label[(source, method)]].append(
                            r4(rec["mean_delta"] / S[m]))
                        at = rec.get("at_m") or {}
                        btw[m][src_label[(source, method)]].append(
                            [f3(at.get("worse")), f3(at.get("tie")), f3(at.get("better"))]
                            if at else None)
                param_zh = {"pitch": "移調（半音）", "tempo": "速度（×）"}[param]
                title = f"{SRCNAME[src_label[(source, method)]]}｜{param_zh}"
                title += "（該平手帶）" if kc == "tie" else "（該變差帶）"
                fid = f"134-{source}-{method}-{param}-{kc}"
                what = ("把整段音高乘上 2^(n/12)。±1 半音該平手，±2 以上該變差。"
                        if param == "pitch" else
                        "把整段播放速度乘上倍率。0.95–1.05 該平手，超出該變差。")
                fam = make_family(fid, "134", "134", title, kc,
                                  [src_label[(source, method)]], list(md.keys()),
                                  [f"{v:.2f}" if param == "tempo" else f"{v:+g}"
                                   for v, _ in items], md, btw=btw, variants=keys,
                                  what=what)
                fams.append(fam)
    return fams


# ── page assembly ───────────────────────────────────────────────────────────

def row_key(f):
    fid = f["id"]
    if f["proto"] in ("126a", "126b"):
        return f"ab:{fid}"
    if fid.startswith("126c-"):
        return "c:" + fid[len("126c-"):].rsplit("-", 1)[0]
    if f["proto"] in ("126d", "133"):
        sec, rest = fid.split("-", 2)[1:]
        fk = rest.rsplit("-", 1)[0]
        if sec == "126c":
            return f"c:{fk}"
        return "ab:" + ("floors" if fk == "floor" else fk)
    return fid


def build_data():
    base, raw_html = load_base()
    items_126b = load_json(ITEMS_126B)
    items_126c = load_json(ITEMS_126C)
    meta126c = {}
    for r in items_126c:
        meta126c.setdefault(r["case"], r)
    items_by_sec = {
        "126b": items_126b,
        "126c": items_126c,
    }

    fams = []
    # existing 126a / 126b families, verbatim
    for f in base["D"]["fams"]:
        f = dict(f)
        f["proto"] = "126a" if f["id"] in ("gain", "lead", "mp3", "misc") else "126b"
        f["group"] = f"{f['proto']}-{f['group']}"
        f["ABS"] = base["ABS"].get(f["id"], {})
        k, n = cell_fail(f["kind"], f["md"], f["metrics"], f["srcs"])
        assert (k, n) == (f["nfail"], f["ntot"]), (f["id"], k, n, f["nfail"], f["ntot"])
        fams.append(f)

    fams += build_126c(items_126c)
    fams += build_126d(items_by_sec, meta126c)
    fams += build_133(items_by_sec, meta126c)
    fams += build_134()

    # overview row: the same intervention across protocols shares one matrix row
    for f in fams:
        f["row"] = row_key(f)

    # notes: existing verbatim, new auto
    notes = dict(base["NOTES"])
    for f in fams:
        if f["id"] not in notes:
            notes[f["id"]] = auto_note(f)
        f.pop("what", None)
        if "ABS" not in f:
            f.pop("ABS", None)

    # ABS global object for hover
    abs_all = dict(base["ABS"])
    for f in fams:
        if f.get("ABS"):
            abs_all[f["id"]] = f["ABS"]

    # metric list: original order then new
    allm = [m for m in ALLM if any(m in f["metrics"] for f in fams)]

    srcname = dict(base["D"]["srcname"])
    srcname.update(SRCNAME)
    # original source names must win where they exist
    for k, v in base["D"]["srcname"].items():
        srcname[k] = v

    # protocol S table
    d126c = load_json(S126C)
    d126d = load_json(S126D)
    d133 = load_json(S133)
    d134 = load_json(S134)
    sproto = {
        "126a": base["D"]["S"],
        "126b": base["D"]["S"],
        "126c": d126c["protocol"]["S"],
        "126d": d126d["S_songeval"],
        "133": {"mert": d133["S_mert"], "muq": d133["S_muq"]},
        "134": d134["protocol"]["S"],
    }

    return {
        "D": {"S": base["D"]["S"], "Sproto": sproto, "srcname": srcname, "fams": fams},
        "ABS": abs_all,
        "NOTES": notes,
        "MN": MN,
        "ALLM": allm,
        "GN": dict(GN),
    }, raw_html, sproto


def protocol_summary(fams):
    agg = OrderedDict()
    for p in PROTO_ORDER:
        fs = [f for f in fams if f["proto"] == p]
        if not fs:
            continue
        nf = sum(f["nfail"] for f in fs)
        nt = sum(f["ntot"] for f in fs)
        agg[p] = {"families": len(fs), "nfail": nf, "ntot": nt}
    return agg


def s_table_html(sproto):
    rows = []
    for p in PROTO_ORDER:
        s = sproto.get(p, {})
        cells = "、".join(f"{MN.get(k, k)} {v:.4g}" for k, v in s.items() if k != "PC")
        rows.append(f"<tr><td class='f'>{p}</td><td class='f'>{cells}</td></tr>")
    return ("<table class='b'><tr><th class='f'>協定</th><th class='f'>S（每個指標）</th></tr>"
            + "".join(rows) + "</table>")


def summary_table(agg):
    rows = []
    for p, a in agg.items():
        pct = 100 * a["nfail"] / a["ntot"] if a["ntot"] else 0
        rows.append(f"<tr><td class='f'>{p}</td><td>{a['families']}</td>"
                    f"<td>{a['nfail']}</td><td>{a['ntot']}</td><td>{pct:.1f}%</td></tr>")
    total_f = sum(a["nfail"] for a in agg.values())
    total_n = sum(a["ntot"] for a in agg.values())
    rows.append(f"<tr><td class='f'><b>合計</b></td>"
                f"<td><b>{sum(a['families'] for a in agg.values())}</b></td>"
                f"<td><b>{total_f}</b></td><td><b>{total_n}</b></td>"
                f"<td><b>{100*total_f/total_n if total_n else 0:.1f}%</b></td></tr>")
    return ("<table class='b'><tr><th class='f'>協定</th><th>項目數</th><th>沒過</th>"
            "<th>總點數</th><th>失敗率</th></tr>" + "".join(rows) + "</table>")


def generic_js(raw_html):
    s = raw_html
    i = s.find("const isTie")
    j = s.find("</script>", i)
    js = s[i:j]
    # S per protocol (Sm was unused in the original; keep it consistent)
    js = js.replace("const hx=svg.querySelector('.hx'), Sm=D.S[m];",
                    "const hx=svg.querySelector('.hx'), Sm=(D.Sproto[f.proto]||{})[m];")
    # source styles for the new aggregate sources
    old_src = re.search(r"const SRCSTY=\{.*?\};", js, re.S).group(0)
    new_src = ("const SRCSTY={'real':['#3b6fe0',0],'ours_cfg3neg':['#2fb34a',0],"
               "'ours_cfg0':['#00b3a4',0],'musicgen':['#f5a623',0],'musicldm':['#a46cf0',0],"
               "'midi/midi_generaluser':['#9a9aa3',1],'midi/midi_fluidr3':['#5a5a66',1],"
               "'rec/csd_real':['#e86fd0',1],'rec/pro':['#b8864b',1],"
               "'rec/poor':['#e0cf2a',1],'rec/poor_sing':['#7fd8ff',1],"
               "'musdb':['#c8372d',0],'songeval':['#8a5cf6',0],'all':['#4b8b8b',0],"
               "'maestro_midi':['#3b6fe0',1],'maestro_rb':['#3b6fe0',0],"
               "'piano126b_midi':['#b8864b',1],'piano126b_rb':['#b8864b',0],"
               "'ttm_rb':['#2fb34a',0]};")
    js = js.replace(old_src, new_src)
    js = js.replace("const PAL=['#3b6fe0','#2fb34a','#f5a623','#a46cf0','#00b3a4','#e86fd0','#b8864b','#9a9aa3'];",
                    "const PAL=['#3b6fe0','#2fb34a','#f5a623','#a46cf0','#00b3a4','#e86fd0','#b8864b','#9a9aa3','#c8372d','#8a5cf6','#4b8b8b'];")
    # bars: tolerates protocols without per-pair win/tie/lose
    js = js.replace(
        "function bars(f,m){\n  const lab =",
        "function bars(f,m){\n  if(!f.btw || !f.btw[m]){\n"
        "    document.getElementById('btitle').textContent = `逐段勝／平／負：${MN[m]}`;\n"
        "    document.getElementById('blegend').innerHTML = '';\n"
        "    document.getElementById('bars').innerHTML = '<p class=\"cap\">這個協定只提供群體平均 Δ，沒有逐段勝／平／負的拆分（126d SongEval summary 沒有 at_m）。</p>';\n"
        "    return;\n  }\n  const lab =")
    js = js.replace(
        "      const w = f.btw[m][src][k], v = f.md[m][src][k];\n      if(!w){ h+='<td></td>'; return; }",
        "      const bw = f.btw[m] ? f.btw[m][src] : null;\n"
        "      const w = bw ? bw[k] : null, v = f.md[m][src][k];\n"
        "      if(!w){ h+='<td></td>'; return; }")
    # show(): protocol tag + per-protocol S in the how-to line
    js = js.replace(
        "document.getElementById('fsub').textContent = `${GN[f.group]}｜${f.nfail}/${f.ntot} 點沒過｜來源 ${f.srcs.length} 個` + (N.axis? `｜${N.axis}`:'');",
        "document.getElementById('fsub').textContent = `${GN[f.group]}｜${f.nfail}/${f.ntot} 點沒過｜來源 ${f.srcs.length} 個｜協定 ${f.proto}` + (N.axis? `｜${N.axis}`:'');")
    js = js.replace(
        "document.getElementById('howto').innerHTML = `<b>圖：</b>縱軸＝${yl}（÷S）。${goal}。`+",
        "const Sv=(D.Sproto[f.proto]||{})[curM];\n"
        "  document.getElementById('howto').innerHTML = `<b>圖：</b>縱軸＝${yl}（÷S；${f.proto} 的 S${Sv?`=${Sv.toFixed(4)}`:''}）。${goal}。`+")
    # matrix has 16 columns
    js = js.replace("table.m{border-collapse:collapse;font-size:13px;min-width:720px;width:100%}",
                    "table.m{border-collapse:collapse;font-size:13px;min-width:1180px;width:100%}")
    # empty chart (e.g. 133 floor rows, no cosim definition)
    js = js.replace(
        "  let vals = (single||SHOWSRC ? series.flat() : med).filter(v=>v!=null);\n",
        "  let vals = (single||SHOWSRC ? series.flat() : med).filter(v=>v!=null);\n"
        "  if(!vals.length) return `<svg viewBox=\"0 0 280 170\" role=\"img\" aria-label=\"${MN[m]}\"><text x=\"140\" y=\"88\" text-anchor=\"middle\" fill=\"var(--muted)\" font-size=\"12\">沒有讀數</text></svg>`;\n")
    # do not connect across missing steps
    js = js.replace(
        "    s+=`<polyline points=\"${med.map((v,j)=>`${X(j)},${Y(v)}`).join(' ')}\" fill=\"none\" stroke=\"var(--med)\" stroke-width=\"2\"/>`;\n",
        "    { const segs=[]; let sg=[];\n"
        "      med.forEach((v,j)=>{ if(v==null){ if(sg.length) segs.push(sg); sg=[]; } else sg.push(`${X(j)},${Y(v)}`); });\n"
        "      if(sg.length) segs.push(sg);\n"
        "      segs.forEach(p=>{ if(p.length>1) s+=`<polyline points=\"${p.join(' ')}\" fill=\"none\" stroke=\"var(--med)\" stroke-width=\"2\"/>`; }); }\n")
    # 0/0 rows (floor-null) should read as a dash
    js = js.replace(
        "h+=`<tr data-i=\"${i}\"><td class=\"f\">${f.title}</td><td>${f.nfail}/${f.ntot}</td>`;",
        "h+=`<tr data-i=\"${i}\"><td class=\"f\">${f.title}</td><td>${f.ntot?f.nfail+'/'+f.ntot:'—'}</td>`;")
    js = js.replace(
        "document.getElementById('fsub').textContent = `${GN[f.group]}｜${f.nfail}/${f.ntot} 點沒過｜來源 ${f.srcs.length} 個｜協定 ${f.proto}`",
        "document.getElementById('fsub').textContent = `${GN[f.group]}｜${f.ntot?f.nfail+'/'+f.ntot+' 點沒過':'沒有可計分的點'}｜來源 ${f.srcs.length} 個｜協定 ${f.proto}`")
    js = js.replace(
        "d.innerHTML=`<h4>${MN[mm]}<span title=\"所有來源×步階的點中沒過的比例\">${Math.round(fr*100)}% 的點沒過</span></h4>`+chart(f,mm);",
        "d.innerHTML=`<h4>${MN[mm]}<span title=\"所有來源×步階的點中沒過的比例\">${fr==null?'—':Math.round(fr*100)+'% 的點沒過'}</span></h4>`+chart(f,mm);")
    # overview: merge rows that share f.row (same intervention, different protocols)
    i0 = js.find("function buildMatrix(){")
    i1 = js.find("\nfunction median", i0)
    assert i0 >= 0 and i1 > i0
    js = js[:i0] + MATRIX_JS + js[i1 + 1:]
    return js


MATRIX_JS = r'''function cellCnt(f,m){
  if(!f.metrics.includes(m)) return null;
  let n=0,k=0;
  for(const s of f.srcs) for(const v of f.md[m][s]){ if(v==null) continue; n++; if(fails(f,v)) k++; }
  return n? [k,n] : null;
}
function buildMatrix(){
  let tot=0,nf=0; fams.forEach(f=>{tot+=f.ntot;nf+=f.nfail;});
  document.getElementById('nf').textContent=nf; document.getElementById('nt').textContent=tot;
  // one row per intervention: families of different protocols with the same row key share it
  const rows=[], byKey={};
  fams.forEach((f,i)=>{ const key=f.row||f.id; if(!(key in byKey)){ byKey[key]=rows.length; rows.push({f0:f,idx:[]}); } rows[byKey[key]].idx.push(i); });
  let h='<tr><th class="f">項目</th><th>協定</th><th>沒過</th>'+ALLM.map(m=>`<th>${MN[m]}</th>`).join('')+'</tr>';
  let g=null;
  rows.forEach(r=>{
    const f0=r.f0;
    if(f0.group!==g){ g=f0.group; h+=`<tr class="g"><td colspan="${ALLM.length+3}">${GN[g]}</td></tr>`; }
    let rk=0,rn=0; const ps=[];
    r.idx.forEach(i=>{ rk+=fams[i].nfail; rn+=fams[i].ntot; if(!ps.includes(fams[i].proto)) ps.push(fams[i].proto); });
    h+=`<tr data-i="${r.idx[0]}"><td class="f">${f0.title}</td><td style="color:var(--muted);font-size:11px">${ps.join(' ')}</td><td>${rn?rk+'/'+rn:'—'}</td>`;
    for(const m of ALLM){
      let k=0,n=0,first=null; const tip=[];
      r.idx.forEach(i=>{ const c=cellCnt(fams[i],m); if(!c) return; if(first==null) first=i; k+=c[0]; n+=c[1]; const fi=fams[i], sec=(fi.proto==='126d'||fi.proto==='133')?fi.id.split('-')[1]:''; tip.push(`${fi.proto}${sec?`（${sec} 素材）`:''}：${c[0]}/${c[1]} 沒過`); });
      if(first==null){ h+='<td style="color:var(--line)">·</td>'; continue; }
      const x=k/n;
      h+=`<td class="c" data-i="${first}" data-m="${m}" title="${tip.join('\n')}" style="background:${heat(x)}">${Math.round(x*100)}%</td>`;
    }
    h+='</tr>';
  });
  const t=document.getElementById('mx'); t.innerHTML=h;
  t.querySelectorAll('td.c').forEach(td=>td.onclick=()=>{ show(+td.dataset.i, td.dataset.m); document.getElementById('dt').scrollIntoView({behavior:'smooth'}); });
}
'''


def build_html(data, raw_html, sproto):
    agg = protocol_summary(data["D"]["fams"])
    js = generic_js(raw_html)
    dumps = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    data_js = (
        f"const D = {dumps(data['D'])};\n"
        f"const ABS = {dumps(data['ABS'])};\n"
        f"const NOTES = {dumps(data['NOTES'])};\n"
        f"const MN = {dumps(data['MN'])};\n"
        f"const ALLM = {dumps(data['ALLM'])};\n"
        f"const GN = {dumps(data['GN'])};\n"
        f"const fams = D.fams;\n"
    )
    # keep the original CSS exactly; only the matrix min-width changes
    css = raw_html.split("<style>", 1)[1].split("</style>", 1)[0]
    css = css.replace("min-width:720px", "min-width:1180px")
    nf = sum(a["nfail"] for a in agg.values())
    nt = sum(a["ntot"] for a in agg.values())
    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<title>寬標準失敗清單（完整版）</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <h1>寬標準失敗清單（完整版：126a／126b／126c／126d／133／134）</h1>
  <p class="muted">2026-10-22。每項介入看群體平均差 Δ 除以該協定的尺度 S，不看逐段勝負。沒有人類評分，「該平手／該變差」是協定預設，不是聽感結果。</p>
  <div class="panel">
    <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:baseline">
      <div><span class="big" id="nf"></span> <span class="muted">/ <span id="nt"></span> 個（介入 × 來源 × 指標）點沒過寬標準</span></div>
    </div>
    <div class="rule">
      <div><b>該平手</b>（音量、平移、極性、移調…）<br>|平均 Δ| ≤ 0.1·S 算過；soft 用同一條</div>
      <div><b>該變差</b>（噪音、削波、錯音…）<br>平均 Δ &lt; −0.1·S 算過；note 用同一條</div>
      <div><b>真人 poor 對 pro</b>：poor − pro &lt; −0.1·S 算過<br><b>底線訊號</b>：比五個 TTM 系統中最低的乾淨平均還低 0.1·S 才算過</div>
    </div>
    <p class="cap"><b>S 依協定而異，Δ/S 不可跨協定比較。</b>126a／126b 用 126a 的 S；126c 用 126c protocol.S；126d 用 S_songeval；133 用 S_mert／S_muq；134 用 134 protocol.S。MIR 指標（pulse_clarity、key_cnn_conf、ibi_cv、chroma_entropy）已轉成「越高越好」（ibi_cv、chroma_entropy 取負）。PC（製作複雜度）不是品質軸，這頁不列。126d／133 沒有逐段勝平負拆分的協定，色條會留白。descriptive／ref／transition／floor-null 不列入失敗計數。</p>
    <h2 style="font-size:15px;margin:14px 0 6px">各協定失敗數／總點數</h2>
    <div style="overflow-x:auto">{summary_table(agg)}</div>
    <h2 style="font-size:15px;margin:14px 0 6px">各協定的 S</h2>
    <div style="overflow-x:auto">{s_table_html(sproto)}</div>
  </div>

  <h2>總覽：每格是該項在這個指標上沒過的比例</h2>
  <p class="muted">同一個介入只佔一列：AES／MEva／CLAP／MIR（126a／126b／126c）、SongEval（126d）、MERT／MuQ（133）的讀數並排在同一列，「協定」欄列出這一列用到哪些協定。各協定的 S 不同，所以只能在同一格裡比較，不要跨欄比 Δ/S。126a 與 126b 的素材在 126d／133 中各自算一次，同一格合併計數；滑鼠停在格上可以看各協定各自沒過幾點，點格子會打開第一個協定的那一項（其餘協定的項目可以在「逐項檢視」的選單裡找到）。灰色「·」代表這一列沒有這個指標的讀數。</p>
  <div class="panel mx"><table class="m" id="mx"></table></div>

  <h2 id="dt">逐項檢視</h2>
  <div class="nav">
    <button id="prev">← 上一項</button>
    <select id="pick"></select>
    <button id="next">下一項 →</button>
    <span class="muted" id="pos"></span>
  </div>
  <div class="panel">
    <div style="font-size:19px;font-weight:700" id="ftitle"></div>
    <div class="muted" id="fsub" style="font-size:13.5px"></div>
    <div class="what" id="fwhat"></div>
    <div class="ktitle">結果</div>
    <ul class="note" id="fnote"></ul>
    <div class="howto" id="howto"></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--band)"></span>過關區</span>
      <span><span class="sw" style="background:var(--fail);border-radius:50%"></span>沒過</span>
      <span><span class="sw" style="background:var(--med);border-radius:50%"></span>過了</span>
      <label style="margin-left:auto;cursor:pointer"><input type="checkbox" id="showsrc"> 拆開看每個來源</label>
    </div>
    <div class="legend" id="srclegend" style="display:none"></div>
    <div class="grid" id="charts"></div>
    <h2 style="font-size:16px" id="btitle"></h2>
    <div class="legend" id="blegend"></div>
    <div class="bars" id="bars"></div>
    <p class="cap">逐段判定用 m = 0.10·S：差在 ±m 內算平手。紅框是寬標準沒過的格子。滑過色條可看比例。</p>
  </div>
</div>
<script>
{data_js}{js}
</script>
</body>
</html>
"""
    return html, agg


# ── md report ───────────────────────────────────────────────────────────────

def build_md(data, agg, sproto):
    fams = data["D"]["fams"]
    s126d = load_json(S126D)

    # One row per (family, metric, source, step), with the per-pair 80% verdict
    # where the protocol provides one (126d: summary passed; others: btw).
    rows = []
    for f in fams:
        for m in f["metrics"]:
            for s in f["srcs"]:
                for i, v in enumerate(f["md"][m].get(s, [])):
                    if v is None:
                        continue
                    wide_fail = is_fail(f["kind"], v)
                    if f["proto"] == "126d":
                        rec = s126d.get(f["id"].split("-")[1], {}) \
                                    .get(f["variants"][i], {}).get(m)
                        if rec is None:
                            continue
                        frac = float(rec["pass_pct"])
                        pair_pass = bool(rec["passed"])
                    else:
                        bw = f.get("btw", {}).get(m, {}).get(s)
                        if not bw or i >= len(bw) or not bw[i]:
                            continue
                        w, t, b = bw[i]
                        frac = t if f["kind"] == "tie" else w
                        pair_pass = frac >= 0.8
                    rows.append({
                        "proto": f["proto"], "fid": f["id"], "m": m,
                        "step": f["steps"][i], "src": s, "v": v,
                        "wide_pass": not wide_fail, "pair_pass": pair_pass,
                        "frac": frac,
                    })

    cmp_by = OrderedDict((p, [0, 0, 0, 0]) for p in PROTO_ORDER)
    for r in rows:
        a = cmp_by[r["proto"]]
        a[0] += 1
        if r["wide_pass"] == r["pair_pass"]:
            a[1] += 1
        elif r["wide_pass"] and not r["pair_pass"]:
            a[2] += 1
        else:
            a[3] += 1
    n_cmp = sum(a[0] for a in cmp_by.values())
    n_wide_pass_pair_fail = sum(a[2] for a in cmp_by.values())
    n_wide_fail_pair_pass = sum(a[3] for a in cmp_by.values())

    # metric-level failure totals (for the "no metric is clean" statement)
    metric_tot = OrderedDict()
    for m in data["ALLM"]:
        nf = nt = 0
        for f in fams:
            if m not in f["metrics"]:
                continue
            for s in f["srcs"]:
                for v in f["md"][m].get(s, []):
                    if v is None:
                        continue
                    nt += 1
                    nf += int(is_fail(f["kind"], v))
        metric_tot[m] = (nf, nt)
    clean_metrics = [m for m, (nf, nt) in metric_tot.items() if nt and nf == 0]

    def row(fid, m, step, src=None):
        for r in rows:
            if r["fid"] == fid and r["m"] == m and r["step"] == step \
                    and (src is None or r["src"] == src):
                return r
        return None

    def ex(fid, m, step, src=None, name=""):
        r = row(fid, m, step, src)
        if not r:
            return ""
        return (f"- {name or fid}｜{MN[m]}｜{r['step']}：Δ/S = {r['v']:+.3f}，"
                f"逐對通過率 {r['frac']*100:.1f}% → 本頁判"
                f"{'過' if r['wide_pass'] else '沒過'}、逐對判"
                f"{'過' if r['pair_pass'] else '沒過'}。\n")

    lines = []
    lines.append("# 寬標準失敗清單（完整版）— 2026-10-22\n")
    lines.append("由 `scripts/analysis/build_wide_fail_list_full.py` 產生；只讀既有 summary／TSV／"
                 "原有 126ab 頁的內嵌 D／ABS／NOTES，沒有重跑評分、沒有用 GPU、沒有人類評分。\n")
    lines.append("## 判準\n")
    lines.append("每個（介入 × 來源 × 指標）點看群體平均差 Δ/S。kind＝tie／soft 時 "
                 "|Δ/S| ≤ 0.1 算過；kind＝lose／note 時 Δ/S < −0.1 算過。"
                 "rec 用 poor−pro；floor 用底線對最差 TTM 乾淨平均。"
                 "descriptive／ref／transition／floor-null 不列入失敗計數。PC 不列。\n")
    lines.append("**S 依協定而異，Δ/S 不可跨協定比較。** 126a／126b 用 126a 的 S；"
                 "126c 用 126c protocol.S；126d 用 S_songeval；133 用 S_mert／S_muq；"
                 "134 用 134 protocol.S。\n")
    lines.append("## 各協定失敗數／總點數\n")
    lines.append("| 協定 | 項目數 | 沒過 | 總點數 | 失敗率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for p, a in agg.items():
        pct = 100 * a["nfail"] / a["ntot"] if a["ntot"] else 0
        lines.append(f"| {p} | {a['families']} | {a['nfail']} | {a['ntot']} | {pct:.1f}% |")
    tf = sum(a["nfail"] for a in agg.values())
    tn = sum(a["ntot"] for a in agg.values())
    lines.append(f"| **合計** | **{sum(a['families'] for a in agg.values())}** | "
                 f"**{tf}** | **{tn}** | **{100*tf/tn if tn else 0:.1f}%** |\n")
    lines.append("126a 與 126b 的部分直接沿用原頁（不重算）：126a 317/1100、"
                 "126b 821/2374，合計 1138/3474，與原頁完全一致。\n")

    lines.append("## 與逐對判準的關係（對照 2026-10-22 review）\n")
    lines.append("review 的主判準是**逐對**：m = 0.1·S，每個介入要有 ≥ 80% 的 clip 過關。"
                 "本頁是**群體平均** Δ/S。兩者不是同一個量。\n")
    lines.append("| 協定 | 可比點數 | 兩邊一致 | 本頁過／逐對沒過 | 本頁沒過／逐對過 |")
    lines.append("|---|---:|---:|---:|---:|")
    for p, a in cmp_by.items():
        if not a[0]:
            continue
        lines.append(f"| {p} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |")
    lines.append(f"| **合計** | **{n_cmp}** | **{n_cmp-n_wide_pass_pair_fail-n_wide_fail_pair_pass}** "
                 f"| **{n_wide_pass_pair_fail}** | **{n_wide_fail_pair_pass}** |\n")
    lines.append(f"群體平均是比逐對 80% 更寬的判準：{n_cmp} 個可比重疊點中，"
                 f"**{n_wide_pass_pair_fail} 點（{100*n_wide_pass_pair_fail/n_cmp:.1f}%）"
                 f"本頁判過、逐對判沒過；反方向 {n_wide_fail_pair_pass} 點**。"
                 "所以本頁的失敗數一定偏低，不能拿來覆蓋 review 的逐對結論。\n")

    lines.append("## 與 review 主結論一致的地方\n")
    lines.append(f"- 「沒有任何評分模型同時做到該判差就判差、該平手就平手」：本頁 16 個指標欄"
                 f"沒有任何一欄零失敗（最少的是 MERT {metric_tot['mert'][0]}/{metric_tot['mert'][1]}，"
                 f"最多的是 chroma_entropy {metric_tot['chroma_entropy'][0]}/{metric_tot['chroma_entropy'][1]}）。\n")
    def v_of(fid, m, step):
        r = row(fid, m, step, "all")
        return r["v"] if r else float("nan")
    lines.append("- MERT 從 30% 錯音起判差：133 的 126b `wrong30pct` 在 MERT／MuQ 的 Δ/S = "
                 f"{v_of('133-126b-wrong-lose', 'mert', '30%'):+.3f}／"
                 f"{v_of('133-126b-wrong-lose', 'muq', '30%'):+.3f}，"
                 f"`wrong100pct` = {v_of('133-126b-wrong-lose', 'mert', '100%'):+.3f}／"
                 f"{v_of('133-126b-wrong-lose', 'muq', '100%'):+.3f}；兩者本頁都判過。\n")
    lines.append("- review 說 126d SongEval 的音符層幾乎全盲：126d 的 126b 音符層"
                 "（wrong／randpitch／jitter／unsteady／posshuffle／dropextra／blastnote）"
                 "在本頁仍有大量點沒過，方向一致；但個別格會因群體平均而翻成「過」，見下。\n")
    lines.append("- review 說 134 沒有指標同時過平手帶與變差帶：134 在本頁仍有 "
                 f"{agg['134']['nfail']}/{agg['134']['ntot']} 點沒過，且沒有任何指標欄零失敗。\n")

    lines.append("## 具體不一致（群體平均 vs 逐對 80%）\n")
    lines.append(ex("126c-drums_lag-lose", "PQ", "100 ms",
                    name="126c 鼓延遲 100 ms"))
    lines.append(ex("126c-clicks-lose", "clap", "5/s",
                    name="126c 爆音點 5/s"))
    lines.append(ex("126c-gain_down-tie", "PQ", "−3 dB",
                    name="126c 整體衰減 −3 dB（該平手）"))
    lines.append(ex("126d-126b-wrong-lose", "se_coh", "100%",
                    name="126d SongEval × 126b 全錯音"))
    lines.append(ex("126d-126a-lead-tie", "se_coh", "10 ms",
                    name="126d SongEval × 126a 開頭平移 10 ms（該平手）"))
    lines.append(ex("133-126a-lead-tie", "mert", "50 ms", src="real",
                    name="133 MERT × 126a 開頭平移 50 ms（該平手，real）"))
    lines.append(ex("134-ttm-rb-tempo-lose", "PQ", "0.40",
                    name="134 TTM·rubberband 速度 0.40×"))
    pq = row("126c-drums_lag-lose", "PQ", "100 ms", "musdb")
    pc = row("126c-drums_lag-lose", "pulse_clarity", "100 ms", "musdb")
    if pq and pc:
        lines.append("其中 126c 鼓延遲 100 ms 的 PQ 是本頁最像「與 review 相反」的一格："
                     "review 說 126c 只有 pulse_clarity 抓到鼓延遲，但 PQ 的群體平均 "
                     f"Δ/S = {pq['v']:.3f}，本頁會判過；逐對只有 {pq['frac']*100:.1f}% 的 clip 變差，"
                     "所以 review 判沒過。"
                     f"同一格 pulse_clarity Δ/S = {pc['v']:.3f}、逐對 {pc['frac']*100:.1f}%，"
                     "兩邊都判過。\n")

    lines.append("## 限制\n")
    lines.append("- 每個協定用自己的 S；跨協定的 Δ/S 不能比大小，只能看各協定內部的失敗比例。\n")
    lines.append("- 126d 與 133 的 summary 是聚合值（126d 每介入一個 mean_d；133 每 case 一個 "
                 "cosim 平均），沒有逐來源拆分；133 的 126a MERT 才有 5 個系統。\n")
    lines.append("- 126d 沒有逐段勝／平／負的 `at_m`；頁面在 126d 的勝平負色條留白。"
                 "133 的勝平負由 per-clip cosim 重新聚合。\n")
    lines.append("- 133 的 tie 判定在本頁用 |Δ/S| ≤ 0.1（照本頁判準）；133 原程式對 tie 用 "
                 "cosim ≥ 1−m。因為 cosim ≤ 1，|Δ/S| ≤ 0.1 與 cosim ≥ 1−m 在單邊上等價，"
                 "但本頁的 tie 會多排除 cosim > 1+m 的情形（實務上不存在）。\n")
    lines.append("- 沒有任何針對這些刺激的人類評分；所有「該不該平手」都是協定設定。\n")
    return "\n".join(lines) + "\n"


def main():
    data, raw_html, sproto = build_data()
    html, agg = build_html(data, raw_html, sproto)
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_MD.write_text(build_md(data, agg, sproto), encoding="utf-8")

    # ── verification printout ──
    print("wrote", OUT_HTML, len(html), "bytes")
    print("wrote", OUT_MD)
    print("protocols:")
    for p, a in agg.items():
        print(f"  {p}: families={a['families']} failed={a['nfail']} total={a['ntot']}")
    print("126a", sum(f['nfail'] for f in data['D']['fams'] if f['proto'] == '126a'),
          sum(f['ntot'] for f in data['D']['fams'] if f['proto'] == '126a'))
    print("126b", sum(f['nfail'] for f in data['D']['fams'] if f['proto'] == '126b'),
          sum(f['ntot'] for f in data['D']['fams'] if f['proto'] == '126b'))
    print("total families", len(data['D']['fams']))

    # hand-check values
    def show(fid, m, step=None):
        f = next((x for x in data['D']['fams'] if x['id'] == fid), None)
        if not f:
            print("MISSING", fid)
            return
        vals = f['md'][m]
        for s, arr in vals.items():
            for i, v in enumerate(arr):
                if step is None or i == step:
                    print(f"  {fid} {m} {s} step={f['steps'][i]} md={v}")
    print("hand checks:")
    show("126c-drums_lag-lose", "PQ", 2)
    show("126c-rebal_vocals_up-tie", "CE", 2)
    show("126c-noise-lose", "meva", 2)
    show("133-126c-bass_semi-lose", "mert", 0)
    show("133-126c-clicks-lose", "muq", 0)
    show("133-126c-clip-lose", "mert", 0)
    show("133-126a-gain-tie", "mert", 0)
    show("133-126b-blastnote-lose", "mert", 0)


if __name__ == "__main__":
    main()
