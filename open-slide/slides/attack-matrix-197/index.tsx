import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';
import { Step, Steps, useSlidePageNumber } from '@open-slide/core';
import type { CSSProperties, ReactNode, RefObject } from 'react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';

// 197 攻擊 × 指標兩張大表（2026-11-21 會議用，8 頁）
// 數字來源：docs/experiments/results/attack_metric_matrix_197_{table1,table2}.md
// 要改數字：直接改下面的 T1_LEFT / T1_RIGHT / T2A / T2B 陣列。
// 每格寫法：[數字字串, 判讀]；判讀 = ok | hit | blind | wrong | over | na
// 版面：素色（紙色底、黑字、只有「方向錯」用紅色），排版參考 Worlds 2022 的黑白編輯風格。

export const meta: SlideMeta = {
  title: '197 攻擊 × 指標兩張大表',
  createdAt: '2026-11-18T09:00:00.000Z',
};

export const design: DesignSystem = {
  palette: { bg: '#f1f0ed', text: '#111111', accent: '#d0261d' },
  fonts: {
    display: '"Noto Sans CJK TC", "Noto Sans TC", system-ui, sans-serif',
    body: '"Noto Sans CJK TC", "Noto Sans TC", system-ui, sans-serif',
  },
  typeScale: { hero: 112, body: 30 },
  radius: 0,
};

const C = {
  ink: '#111111',
  paper: '#f1f0ed',
  muted: '#77756f',
  faint: '#a9a7a0',
  rule: '#c8c6c0',
  fill: '#d9d7d1',
  red: '#d0261d',
};
const mono = '"Noto Sans Mono", "DejaVu Sans Mono", ui-monospace, monospace';
const narrow = '"Noto Sans", "Noto Sans CJK TC", sans-serif';

// ───────────────────────── 判讀樣式 ─────────────────────────

// 資料只寫 blind；畫的時候依星號與正負拆成 flat／small／smallup（見 resolve）
type Verdict = 'ok' | 'hit' | 'drift' | 'blind' | 'flat' | 'small' | 'smallup' | 'wrong' | 'over' | 'na';
type Cell = [string, Verdict];
type Row = { label: string; group?: string; cells: [Cell, Cell, Cell, Cell] };

// 三級：做對＝無色、做錯＝淺紅、做錯嚴重＝深紅（超出平手帶 m 的錯）
const GOOD = { bg: 'transparent', fg: C.ink, weight: 500 };
const BAD = { bg: '#f4cfcb', fg: '#9e1b14', weight: 700 };
const WORSE = { bg: C.red, fg: '#ffffff', weight: 800 };
const V: Record<Verdict, { bg: string; fg: string; weight: number; name: string; short: string }> = {
  ok: { ...GOOD, name: '符合預期', short: '做對' },
  hit: { ...GOOD, name: '抓得到（該變差、下降超出 m）', short: '做對' },
  drift: { ...BAD, name: '該不變卻偏移，CI 整段超出 m/2 但未超出 m', short: '做錯' },
  blind: { ...BAD, name: '該變差卻在平手帶內', short: '做錯' },
  flat: { ...BAD, name: '該變差卻完全沒動（CI 跨零）', short: '做錯' },
  small: { ...BAD, name: '該變差，有降但不到 m', short: '做錯' },
  smallup: { ...BAD, name: '該變差，反而微升、不到 m', short: '做錯' },
  over: { ...WORSE, name: '該不變卻超出 m', short: '做錯嚴重' },
  wrong: { ...WORSE, name: '該不變或變差，卻加分超出 m', short: '做錯嚴重' },
  na: { bg: 'transparent', fg: C.faint, weight: 400, name: '無法判斷，只報讀數', short: '無法判斷（灰字）' },
};

// 星號不上投影片：顏色就是判定結果（資料裡的 * / ** 只留給 resolve 判斷帶內有沒有動）
const shown = (txt: string) => txt.replace(/\*+$/, '');

// blind 拆三種：沒星號＝CI 跨零（完全沒動）；有星號且為正＝該變差卻微升；其餘＝有降但不到 m
const resolve = ([txt, v]: Cell): Verdict => {
  if (v !== 'blind') return v;
  if (!txt.endsWith('*')) return 'flat';
  return txt.startsWith('+') ? 'smallup' : 'small';
};

// 依欄計數：嚴重＝wrong/over；1 類（誤報）＝drift；2 類（漏報）＝該變差列的帶內格
const countCol = (rows: Row[], pick: Verdict[]) =>
  [0, 1, 2, 3].map((j) => rows.filter((r) => pick.includes(resolve(r.cells[j]))).length);

// ───────────────────────── 表 1 資料 ─────────────────────────

const T1_LEFT: Row[] = [
  { group: '該不變', label: '音量 +6 dB', cells: [['+0.029*', 'ok'], ['−0.079*', 'drift'], ['+0.100*', 'drift'], ['+0.023*', 'ok']] },
  { label: '音量 −6 dB', cells: [['−0.024*', 'ok'], ['+0.064*', 'drift'], ['−0.159**', 'over'], ['−0.046*', 'drift']] },
  { label: '前面加 0.5 秒靜音', cells: [['+0.012*', 'ok'], ['+0.003*', 'ok'], ['−0.040*', 'ok'], ['−0.032*', 'ok']] },
  { label: 'rubberband 原速原調', cells: [['−0.016*', 'ok'], ['−0.006*', 'ok'], ['−0.048*', 'ok'], ['−0.028*', 'ok']] },
  { label: 'htdemucs 分離再相加', cells: [['+0.018*', 'ok'], ['−0.030*', 'ok'], ['−0.011*', 'ok'], ['+0.005*', 'ok']] },
  { group: '不變或變差', label: '移調 +1 半音', cells: [['−0.071**', 'ok'], ['−0.207**', 'ok'], ['−0.084*', 'ok'], ['−0.108**', 'ok']] },
  { label: '移調 −1 半音', cells: [['+0.073**', 'wrong'], ['−0.428**', 'ok'], ['−0.253**', 'ok'], ['−0.262**', 'ok']] },
  { label: '移調 +2 半音', cells: [['−0.074**', 'ok'], ['−0.237**', 'ok'], ['−0.083*', 'ok'], ['−0.091**', 'ok']] },
  { label: '移調 −2 半音', cells: [['+0.222**', 'wrong'], ['−0.558**', 'ok'], ['−0.337**', 'ok'], ['−0.230**', 'ok']] },
  { label: '變速 0.9x（放慢）', cells: [['+0.065**', 'wrong'], ['−0.142**', 'ok'], ['+0.025*', 'ok'], ['−0.006', 'ok']] },
  { label: '變速 1.1x（加快）', cells: [['−0.079**', 'ok'], ['−0.128**', 'ok'], ['−0.077*', 'ok'], ['−0.059*', 'ok']] },
  { label: '低通 6 kHz', cells: [['+0.153**', 'wrong'], ['−0.338**', 'ok'], ['−0.195**', 'ok'], ['−0.133**', 'ok']] },
  { label: '低通 4 kHz', cells: [['+0.007', 'ok'], ['−0.550**', 'ok'], ['−0.230**', 'ok'], ['−0.191**', 'ok']] },
  { label: '低通 2 kHz', cells: [['−0.174**', 'ok'], ['−1.125**', 'ok'], ['−0.602**', 'ok'], ['−0.564**', 'ok']] },
  { label: 'high pass 200 Hz', cells: [['−0.145**', 'ok'], ['−0.502**', 'ok'], ['−0.368**', 'ok'], ['−0.113**', 'ok']] },
  { label: '250 Hz ＋10 dB', cells: [['+0.065**', 'wrong'], ['−0.079*', 'ok'], ['−0.077*', 'ok'], ['−0.023*', 'ok']] },
];

const T1_RIGHT: Row[] = [
  { group: '該變差', label: '規律大鼓（每 0.5 秒）', cells: [['−0.006', 'blind'], ['+0.017*', 'blind'], ['−0.006', 'blind'], ['−0.100**', 'hit']] },
  { label: '隨機時間大鼓', cells: [['−0.141**', 'hit'], ['+0.030*', 'blind'], ['−0.015*', 'blind'], ['−0.070*', 'blind']] },
  { label: '規律 1 ms 脈衝', cells: [['−0.060*', 'blind'], ['−0.367**', 'hit'], ['−0.388**', 'hit'], ['−0.214**', 'hit']] },
  { label: '220 Hz 方波（小 18 LU）', cells: [['−0.180**', 'hit'], ['−0.639**', 'hit'], ['−0.654**', 'hit'], ['−0.112**', 'hit']] },
  { label: '50 ms 白噪爆音', cells: [['−0.226**', 'hit'], ['+0.053*', 'blind'], ['−0.249**', 'hit'], ['−0.226**', 'hit']] },
  { label: '中段白噪 SNR 10', cells: [['−0.693**', 'hit'], ['−0.107*', 'blind'], ['−0.051*', 'blind'], ['+0.036*', 'blind']] },
  { label: '頭尾白噪 SNR 10', cells: [['+0.000', 'blind'], ['−0.134**', 'hit'], ['−0.072*', 'blind'], ['+0.057*', 'blind']] },
  { label: '疊 1 kHz 正弦（小 18 LU）', cells: [['+0.013*', 'blind'], ['−0.042*', 'blind'], ['−0.161**', 'hit'], ['−0.048*', 'blind']] },
  { label: '疊合成循環音（小 18 LU）', cells: [['+0.200**', 'wrong'], ['−0.037*', 'blind'], ['−0.125*', 'blind'], ['−0.069*', 'blind']] },
  { label: '同響度調內鋼琴琶音', cells: [['+0.229**', 'wrong'], ['+0.373**', 'wrong'], ['+0.420**', 'wrong'], ['+0.267**', 'wrong']] },
];

// ───────────────────────── 表 2 資料 ─────────────────────────

const T2A: Row[] = [
  { group: '該變差', label: '錯音 10%', cells: [['−0.061*', 'blind'], ['−0.013*', 'blind'], ['−0.017*', 'blind'], ['−0.150**', 'hit']] },
  { label: '錯音 30%', cells: [['−0.171**', 'hit'], ['−0.036*', 'blind'], ['−0.057*', 'blind'], ['−0.489**', 'hit']] },
  { label: '全部錯音', cells: [['−0.240**', 'hit'], ['−0.039*', 'blind'], ['−0.073*', 'blind'], ['−0.739**', 'hit']] },
  { label: '多 30% 隨機音高', cells: [['−0.259**', 'hit'], ['−0.150**', 'hit'], ['−0.022', 'blind'], ['−0.515**', 'hit']] },
  { label: 'onset 抖動 50 ms', cells: [['−0.233**', 'hit'], ['−0.028*', 'blind'], ['−0.055*', 'blind'], ['−0.353**', 'hit']] },
  { label: '拿掉 30% 音符', cells: [['−0.236**', 'hit'], ['+0.110**', 'wrong'], ['+0.006', 'blind'], ['−0.159**', 'hit']] },
  { group: '該不變', label: 'MIDI 移調 +1', cells: [['−0.029', 'ok'], ['+0.021*', 'ok'], ['+0.004', 'ok'], ['−0.041', 'ok']] },
  { label: 'MIDI 移調 −1', cells: [['−0.050*', 'ok'], ['−0.015*', 'ok'], ['−0.021', 'ok'], ['−0.058', 'ok']] },
  { label: 'MIDI 變速 0.9x', cells: [['−0.011', 'ok'], ['+0.010', 'ok'], ['+0.007', 'ok'], ['+0.014', 'ok']] },
  { label: 'MIDI 變速 1.1x', cells: [['+0.010', 'ok'], ['−0.016*', 'ok'], ['−0.009', 'ok'], ['+0.022', 'ok']] },
  { group: '無法判斷', label: '多 30% 和聲內音符', cells: [['−0.192**', 'na'], ['−0.101*', 'na'], ['+0.071*', 'na'], ['−0.260**', 'na']] },
  { label: '高八度', cells: [['−0.190**', 'na'], ['+0.155**', 'na'], ['−0.061*', 'na'], ['−0.048', 'na']] },
  { label: '換 SoundFont', cells: [['+0.014', 'na'], ['+0.212**', 'na'], ['−0.892**', 'na'], ['−0.085*', 'na']] },
];

// B 區：[對比, n, A1, PQ, CE, CMI]；第 7 欄是要強調的欄位索引（0=A1 … 3=CMI）
const T2B: { label: string; vals: [string, string, string, string]; hot?: number[] }[] = [
  { label: '上行琶音 − 柱狀和弦', vals: ['+0.031', '+0.336**', '+0.769**', '+0.180**'], hot: [2] },
  { label: '下行琶音 − 柱狀和弦', vals: ['+0.027', '+0.315**', '+0.708**', '+0.029'], hot: [2] },
  { label: '重複音 − 柱狀和弦', vals: ['+0.280**', '+0.231**', '+0.104', '−0.235**'] },
  { label: '上行琶音 − 重複音', vals: ['−0.249**', '+0.105*', '+0.665**', '+0.415**'], hot: [2] },
  { label: '高八度（o5 − o4）', vals: ['−0.191**', '+0.142*', '−0.001', '−0.165**'], hot: [2] },
  { label: '小調 − 大調', vals: ['+0.110', '+0.009', '+0.026', '−0.157*'] },
];

// ───────────────────────── 共用元件 ─────────────────────────

// 邊緣的小字標籤：底部一排、左側直排（Worlds 2022 的版面骨架）
const Shell = ({ section, children }: { section: string; children: ReactNode }) => {
  const { current, total } = useSlidePageNumber();
  const label: CSSProperties = { fontFamily: mono, fontSize: 15, letterSpacing: '0.12em', color: C.muted };
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
        padding: '64px 100px 92px 120px',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--osd-bg)',
        color: 'var(--osd-text)',
        fontFamily: 'var(--osd-font-body)',
        position: 'relative',
      }}
    >
      {children}
      <div
        style={{
          ...label,
          position: 'absolute',
          left: 44,
          top: 64,
          bottom: 92,
          writingMode: 'vertical-rl',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>MEANAUDIO 197</span>
        <span>ATTACK × METRIC</span>
      </div>
      <div
        style={{
          ...label,
          position: 'absolute',
          left: 120,
          right: 100,
          bottom: 40,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr auto',
        }}
      >
        <span>{section}</span>
        <span>D ＝ 處理後 − 同段原版</span>
        <span>無人類評分</span>
        <span>
          {String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}
        </span>
      </div>
    </div>
  );
};

const Title = ({ children, sub }: { children: ReactNode; sub?: ReactNode }) => (
  <div style={{ marginBottom: 30 }}>
    <div style={{ fontSize: 64, fontWeight: 900, lineHeight: 1.1, letterSpacing: '-0.01em' }}>{children}</div>
    {sub ? <div style={{ fontSize: 28, marginTop: 14, lineHeight: 1.4 }}>{sub}</div> : null}
    <div style={{ height: 3, background: C.ink, marginTop: 22 }} />
  </div>
);

const Kicker = ({ children }: { children: ReactNode }) => (
  <div style={{ fontSize: 26, fontWeight: 900, paddingBottom: 10, borderBottom: `2px solid ${C.ink}` }}>{children}</div>
);

const HEAD = ['A1', 'PQ', 'CE', 'CMI'];

type Tally = { label: string; vals: number[]; red?: boolean };

const MatrixTable = ({
  rows,
  rowH = 38,
  font = 21,
  labelW = 330,
  groupW = 118,
  tallies = [],
}: {
  rows: Row[];
  rowH?: number;
  font?: number;
  labelW?: number;
  groupW?: number;
  tallies?: Tally[];
}) => {
  const cols = `${groupW}px ${labelW}px repeat(4, 1fr)`;
  return (
    <div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: cols,
          height: rowH,
          alignItems: 'end',
          paddingBottom: 6,
          borderBottom: `2px solid ${C.ink}`,
          fontFamily: mono,
          fontSize: 16,
          letterSpacing: '0.12em',
          color: C.muted,
        }}
      >
        <div>預期</div>
        <div>處理</div>
        {HEAD.map((h) => (
          <div key={h} style={{ textAlign: 'center', color: C.ink, fontSize: 19, fontWeight: 700 }}>
            {h}
          </div>
        ))}
      </div>
      {rows.map((r, i) => (
        <div
          key={r.label}
          style={{
            display: 'grid',
            gridTemplateColumns: cols,
            height: rowH,
            alignItems: 'stretch',
            borderTop: r.group && i > 0 ? `2px solid ${C.ink}` : i > 0 ? `1px solid ${C.rule}` : 'none',
            fontSize: font,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', fontSize: font - 3, fontWeight: 900 }}>{r.group ?? ''}</div>
          <div style={{ display: 'flex', alignItems: 'center', whiteSpace: 'nowrap' }}>{r.label}</div>
          {r.cells.map((cell, j) => {
            const v = resolve(cell);
            return (
            <div
              key={j}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: V[v].bg,
                color: V[v].fg,
                fontWeight: V[v].weight,
                fontFamily: narrow,
                fontSize: font + 4,
                borderLeft: `3px solid ${C.paper}`,
              }}
            >
              {shown(cell[0])}
            </div>
            );
          })}
        </div>
      ))}
      {tallies.map((t, k) => (
        <div
          key={t.label}
          style={{
            display: 'grid',
            gridTemplateColumns: `${groupW + labelW}px repeat(4, 1fr)`,
            alignItems: 'baseline',
            borderTop: k === 0 ? `2px solid ${C.ink}` : 'none',
            paddingTop: k === 0 ? 8 : 0,
          }}
        >
          <div style={{ fontSize: font, fontWeight: 900 }}>{t.label}</div>
          {t.vals.map((v, j) => (
            <div
              key={j}
              style={{
                textAlign: 'center',
                fontFamily: narrow,
                fontWeight: 700,
                fontSize: 40,
                lineHeight: 1.15,
                color: v === 0 ? C.faint : t.red ? C.red : C.ink,
              }}
            >
              {v}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

const Swatch = ({ v, children }: { v: Verdict; children?: ReactNode }) => (
  <span
    style={{
      display: 'inline-block',
      minWidth: children ? 0 : 22,
      height: children ? undefined : 22,
      background: V[v].bg,
      color: V[v].fg,
      fontWeight: V[v].weight,
      fontFamily: mono,
      padding: children ? '2px 12px' : 0,
      boxShadow: V[v].bg === 'transparent' ? `inset 0 0 0 1px ${C.rule}` : 'none',
      verticalAlign: 'middle',
    }}
  >
    {children}
  </span>
);

const Legend = ({ items }: { items: Verdict[] }) => (
  <div style={{ display: 'flex', gap: 24, fontSize: 19, flexWrap: 'wrap' }}>
    {items.map((v) => (
      <span key={v} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Swatch v={v} />
        {V[v].short}
      </span>
    ))}
  </div>
);

// 無框、細線分隔的條列
const Lines = ({ items, size = 23 }: { items: ReactNode[]; size?: number }) => (
  <div style={{ fontSize: size, lineHeight: 1.45 }}>
    {items.map((x, i) => (
      <div key={i} style={{ padding: '10px 0', borderTop: i > 0 ? `1px solid ${C.rule}` : 'none' }}>
        {x}
      </div>
    ))}
  </div>
);

const B = ({ children }: { children: ReactNode }) => <b style={{ fontWeight: 900 }}>{children}</b>;
const Red = ({ children }: { children: ReactNode }) => <b style={{ color: C.red, fontWeight: 900 }}>{children}</b>;

// ───────────────────────── 第 1 頁：資料與處理 ─────────────────────────

const FACTS: [string, string, ReactNode][] = [
  ['BASES', '6 × 1,000', <>真實 MusicCaps、MeanAudio CFG3＋neg × 3 seed、MusicGen、MusicLDM；3 個 CFG0 底在附表</>],
  ['METRICS', '4', <>MuQ A1、AES PQ、AES CE、CMI-RM quality；CLAP、PAM 當旁證</>],
  ['TABLE 1', '26', <>任何音檔都能做的處理：音量、移調、變速、濾波、疊聲音、加噪音</>],
  ['TABLE 2', '13 + 384', <>MIDI：MAESTRO 150 段 × 2 SoundFont 的音符層介入；384 段合成和弦進行</>],
];

const Page1: Page = () => (
  <Shell section="METHOD 1/2">
    <Title sub="同一包音檔、同一組評分器；每一格都是處理後減同一段原版">資料與處理</Title>

    <div>
      {FACTS.map(([k, n, d], i) => (
        <div
          key={k}
          style={{
            display: 'grid',
            gridTemplateColumns: '170px 300px 1fr',
            alignItems: 'baseline',
            padding: '6px 0 10px',
            borderTop: i > 0 ? `1px solid ${C.rule}` : 'none',
          }}
        >
          <span style={{ fontFamily: mono, fontSize: 16, letterSpacing: '0.12em', color: C.muted }}>{k}</span>
          <span style={{ fontFamily: narrow, fontWeight: 700, fontSize: 60, lineHeight: 1.05 }}>{n}</span>
          <span style={{ fontSize: 25 }}>{d}</span>
        </div>
      ))}
    </div>

    <div style={{ display: 'flex', gap: 64, marginTop: 34 }}>
      <div style={{ flex: 1.15 }}>
        <Kicker>表 1 的處理（一句話配方，不用梯度）</Kicker>
        <div style={{ display: 'grid', gridTemplateColumns: '160px 1fr', rowGap: 12, marginTop: 14, fontSize: 23, lineHeight: 1.4 }}>
          <B>該不變</B>
          <span>音量 ±6 dB、前面加 0.5 秒靜音、rubberband 原速原調、分離再相加</span>
          <B>不變或變差</B>
          <span>移調 ±1／±2 半音、變速 0.9x／1.1x、低通 6／4／2 kHz、high pass 200 Hz、250 Hz ＋10 dB</span>
          <B>該變差</B>
          <span>規律／隨機大鼓、1 ms 脈衝、220 Hz 方波、白噪爆音、中段／頭尾白噪、疊正弦、疊合成循環音、疊鋼琴琶音</span>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <Kicker>表 2 的處理（鋼琴 MIDI）</Kicker>
        <div style={{ display: 'grid', gridTemplateColumns: '90px 1fr', rowGap: 12, marginTop: 14, fontSize: 23, lineHeight: 1.4 }}>
          <B>A 區</B>
          <span>錯音 10／30／100%、多 30% 音符、onset 抖動 50 ms、拿掉 30% 音符；MIDI 層移調、變速；高八度、換 SoundFont</span>
          <B>B 區</B>
          <span>和弦進行 × 拆法（柱狀／上行／下行／重複音）× 八度（C4／C5）× 大小調 × 12 個主音，看 CE 跟的是調還是音高</span>
        </div>
      </div>
    </div>

    <div style={{ marginTop: 'auto', fontSize: 21, color: C.muted }}>
      所有音檔鎖 −30 LUFS；評分設定與 189／190／193 相同，重整列與新評列同底、同 id、同評分器。
    </div>
  </Shell>
);

// ───────────────────────── 第 2 頁：判定規則 ─────────────────────────

const EXAMPLE: Row[] = [
  { label: '低通 6 kHz', group: '不變或變差', cells: [['+0.153**', 'wrong'], ['−0.338**', 'ok'], ['−0.195**', 'ok'], ['−0.133**', 'ok']] },
];

const Page2: Page = () => (
  <Shell section="METHOD 2/2">
    <Title sub="方向在跑之前登錄；移調、變速、低通只能不變或變差，變好就是錯">判定規則</Title>

    <div style={{ display: 'flex', gap: 64 }}>
      <div style={{ flex: 1 }}>
        <Kicker>數字與平手帶</Kicker>
        <Lines
          size={26}
          items={[
            <><B>d</B>＝處理後 − 同段原版，逐段配對；主表是 6 個底的等權平均</>,
            <>95% CI 整段超出平手帶 m＝明顯變了；整段超出 m/2＝有偵測到、但不明顯</>,
            <>格子不標星號，顏色就是判定</>,
          ]}
        />
        <div style={{ marginTop: 18, fontSize: 21, color: C.muted }}>平手帶 m＝5 個 TTM 系統（MusicGen、MusicLDM×2、MeanAudio×2）平均分數全距的 10%，即系統間差距的一成</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', marginTop: 6, borderTop: `2px solid ${C.ink}` }}>
          {[
            ['A1', '0.055'],
            ['PQ', '0.091'],
            ['CE', '0.119'],
            ['CMI', '0.066'],
          ].map(([k, v], i) => (
            <div key={k} style={{ paddingTop: 8, paddingLeft: i > 0 ? 18 : 0, borderLeft: i > 0 ? `1px solid ${C.rule}` : 'none' }}>
              <div style={{ fontFamily: mono, fontSize: 16, letterSpacing: '0.12em', color: C.muted }}>{k}</div>
              <div style={{ fontFamily: narrow, fontWeight: 700, fontSize: 52, lineHeight: 1.1 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ flex: 1.2 }}>
        <Kicker>預期方向 × 實際 → 判讀</Kicker>
        <div style={{ display: 'grid', gridTemplateColumns: '170px 1fr', fontSize: 25, alignItems: 'center' }}>
          {(
            [
              ['該不變', <>不到 m/2 <Swatch v="ok">做對</Swatch>　m/2～m <Swatch v="drift">做錯</Swatch>　超出 m <Swatch v="over">做錯嚴重</Swatch></>],
              ['不變或變差', <>不變或下降 <Swatch v="ok">做對</Swatch>　上升 m/2～m <Swatch v="drift">做錯</Swatch>　超出 m <Swatch v="wrong">做錯嚴重</Swatch></>],
              ['該變差', <>下降超出 m <Swatch v="hit">做對</Swatch>　帶內 <Swatch v="small">做錯</Swatch>　上升超出 m <Swatch v="wrong">做錯嚴重</Swatch></>],
              ['無法判斷', <><Swatch v="na">只報讀數</Swatch>　表 2 末三列（沒有正確方向）</>],
            ] as [string, ReactNode][]
          ).map(([k, d], i) => (
            <div key={k} style={{ display: 'contents' }}>
              <div style={{ fontWeight: 900, padding: '20px 0', borderTop: i > 0 ? `1px solid ${C.rule}` : 'none' }}>{k}</div>
              <div style={{ padding: '20px 0', borderTop: i > 0 ? `1px solid ${C.rule}` : 'none' }}>{d}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: `2px solid ${C.ink}`, fontSize: 23, lineHeight: 1.5 }}>
          <B>嚴重錯誤</B>＝深紅格數　<B>1 類錯誤</B>＝誤報（該不變卻偏移）　<B>2 類錯誤</B>＝漏報（該變差沒降過 m）
          <div style={{ color: C.muted, fontSize: 21 }}>k/6＝6 個底裡有幾個自己也過；主表平均會蓋掉逐底差異，引用時看附表</div>
        </div>
      </div>
    </div>

    <div style={{ marginTop: 'auto' }}>
      <div style={{ fontSize: 21, color: C.muted, marginBottom: 4 }}>例：低通 6 kHz 只有 A1 方向錯（A1 在 6 個底裡 5 個自己也過）</div>
      <MatrixTable rows={EXAMPLE} rowH={44} font={23} groupW={170} labelW={330} />
    </div>
  </Shell>
);

// ───────────────────────── 第 3 頁：表 1 ─────────────────────────

const Page3: Page = () => (
  <Shell section="TABLE 1 · 6 BASES × 1,000">
    <Title sub={<>A1 方向錯 5 格，全是頻譜往低移；同響度鋼琴琶音四個指標<Red>全錯</Red></>}>表 1　任何音檔都能做的處理</Title>

    <div style={{ display: 'flex', gap: 56, flex: 1 }}>
      <div style={{ flex: 1 }}>
        <MatrixTable rows={T1_LEFT} rowH={38} font={21} labelW={280} groupW={130} />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <MatrixTable
          rows={T1_RIGHT}
          rowH={38}
          font={21}
          labelW={320}
          groupW={90}
          tallies={[
            { label: '嚴重錯誤', vals: countCol([...T1_LEFT, ...T1_RIGHT], ['wrong', 'over']), red: true },
            { label: '1 類錯誤', vals: countCol([...T1_LEFT, ...T1_RIGHT], ['drift']) },
            { label: '2 類錯誤', vals: countCol([...T1_LEFT, ...T1_RIGHT], ['small', 'smallup', 'flat']) },
          ]}
        />
        <div style={{ fontSize: 17, color: C.muted, marginTop: 4 }}>
          兩欄合計。1 類＝誤報：該不變卻偏移；2 類＝漏報：該變差卻沒降超過 m
        </div>
        <div style={{ marginTop: 14, borderTop: `2px solid ${C.ink}` }}>
          <Lines
            size={21}
            items={[
              <>A1：降調、放慢、低通 6 kHz、補 250 Hz 都<Red>加分</Red>；扣掉 rubberband 處理底仍在（−2 半音 +0.238）</>,
              <>鋼琴琶音只有 CLAP 抓到（<B>−0.054</B>）；中段白噪 SNR 10 只有 A1 抓到</>,
            ]}
          />
        </div>
      </div>
    </div>

    <Legend items={['ok', 'small', 'wrong']} />
  </Shell>
);

// ───────────────────────── 第 4 頁：表 2 ─────────────────────────

const Page4: Page = () => {
  const cols = '330px repeat(4, 1fr)';
  return (
    <Shell section="TABLE 2 · MIDI">
      <Title sub="PQ、CE 對錯音只微降、不到平手帶；CE 跟的是絕對音高，不是調名">表 2　MIDI</Title>

      <div style={{ display: 'flex', gap: 56, flex: 1 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 22, fontWeight: 900, marginBottom: 2 }}>A 區　音符層介入（MAESTRO 150 段 × 2 SoundFont）</div>
          <MatrixTable
            rows={T2A}
            rowH={38}
            font={21}
            labelW={260}
            groupW={120}
            tallies={[
              { label: '嚴重錯誤', vals: countCol(T2A, ['wrong', 'over']), red: true },
              { label: '1 類錯誤', vals: countCol(T2A, ['drift']) },
              { label: '2 類錯誤', vals: countCol(T2A, ['small', 'smallup', 'flat']) },
            ]}
          />
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 22, fontWeight: 900, marginBottom: 2 }}>B 區　和弦進行的對比（384 段，段與段配對）</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: cols,
              height: 38,
              alignItems: 'end',
              paddingBottom: 6,
              borderBottom: `2px solid ${C.ink}`,
              fontFamily: mono,
              fontSize: 16,
              letterSpacing: '0.12em',
              color: C.muted,
            }}
          >
            <div>對比</div>
            {HEAD.map((h) => (
              <div key={h} style={{ textAlign: 'center', color: C.ink, fontSize: 19, fontWeight: 700 }}>
                {h}
              </div>
            ))}
          </div>
          {T2B.map((r, i) => (
            <div
              key={r.label}
              style={{
                display: 'grid',
                gridTemplateColumns: cols,
                height: 38,
                alignItems: 'stretch',
                borderTop: i > 0 ? `1px solid ${C.rule}` : 'none',
                fontSize: 21,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center' }}>{r.label}</div>
              {r.vals.map((v, j) => {
                const hot = r.hot?.includes(j);
                const sig = v.endsWith('**');
                return (
                  <div
                    key={j}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontFamily: narrow,
                      fontSize: 25,
                      color: sig || hot ? C.ink : C.faint,
                      fontWeight: hot ? 900 : sig ? 600 : 400,
                      boxShadow: hot ? `inset 0 0 0 3px ${C.ink}` : 'none',
                    }}
                  >
                    {shown(v)}
                  </div>
                );
              })}
            </div>
          ))}
          <div style={{ fontSize: 17, color: C.muted, marginTop: 6 }}>B 區沒有預期方向，只報讀數；粗體＝超出平手帶、灰字＝帶內；粗框＝CE 的關鍵格</div>

          <div style={{ marginTop: 22 }}>
            <Kicker>CE 跟的是調，還是音高？</Kicker>
            <Lines
              size={22}
              items={[
                <>拆成琶音 CE <B>+0.7～+0.8</B>；整體高八度 <B>−0.001</B>、大小調不顯著</>,
                <>逐主音曲線全距 0.40–0.51；C4 與 C5 兩條曲線<B>負相關</B>（大調 −0.70、小調 −0.37）</>,
                <>約 A4–F5 一帶最高 → 跟<B>絕對音高</B>，不是調名（PQ 同樣，−0.92／−0.94）</>,
              ]}
            />
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 20 }}>
        <Legend items={['ok', 'small', 'wrong', 'na']} />
        <span style={{ fontSize: 20, whiteSpace: 'nowrap' }}>
          拿掉 30% 音符 PQ <Red>+0.110</Red>；換 SoundFont CE −0.892；CMI-RM 盲點 0
        </span>
      </div>
    </Shell>
  );
};


// ───────────────────────── 第 5 頁：疊加是內插 ─────────────────────────
// 數字：196 dev 200 段 × 9 底的平均 CE（~/eval_output_nvme/ce_push_196/per_clip/dev_*）
// x：原底、合成聲相對底 −12…+18 LU、只有合成聲（全部鎖 −30 LUFS）

const MIX_X = ['原本的底', '−12', '−6', '0', '+6', '+12', '+18', '只有合成聲'];
const MIX: { name: string; v: number[]; hot?: boolean }[] = [
  { name: 'MusicLDM', v: [5.59, 6.0, 6.25, 6.53, 6.76, 6.99, 7.2, 7.71], hot: true },
  { name: 'CFG0 s1', v: [5.9, 6.09, 6.26, 6.48, 6.7, 6.88, 7.05, 7.62] },
  { name: 'CFG0 s2', v: [6.04, 6.24, 6.38, 6.59, 6.77, 6.92, 7.08, 7.64] },
  { name: 'CFG0 s3', v: [5.74, 6.02, 6.19, 6.43, 6.65, 6.83, 7.01, 7.6] },
  { name: 'MusicGen', v: [6.4, 6.66, 6.8, 6.98, 7.11, 7.2, 7.27, 7.62] },
  { name: '真實', v: [6.15, 6.35, 6.42, 6.44, 6.53, 6.76, 7.03, 7.68] },
  { name: 'CFG3+neg s1', v: [6.96, 7.12, 7.17, 7.2, 7.18, 7.17, 7.23, 7.66] },
  { name: 'CFG3+neg s2', v: [7.03, 7.22, 7.27, 7.29, 7.26, 7.21, 7.24, 7.66] },
  { name: 'CFG3+neg s3', v: [6.88, 7.13, 7.19, 7.26, 7.25, 7.21, 7.25, 7.68] },
];

const MixChart = () => {
  const W = 1000, H = 620, L = 70, R = 150, T = 20, Bm = 70;
  const y0 = 5.4, y1 = 7.9;
  const x = (i: number) => L + (i * (W - L - R)) / (MIX_X.length - 1);
  const y = (v: number) => T + ((y1 - v) * (H - T - Bm)) / (y1 - y0);
  const ticks = [5.5, 6.0, 6.5, 7.0, 7.5];
  const order = [...MIX].sort((a, b) => Number(!!a.hot) - Number(!!b.hot));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke={C.rule} strokeWidth={1} />
          <text x={L - 12} y={y(t) + 7} textAnchor="end" fontSize={20} fontFamily={narrow} fill={C.muted}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={y(6.41)} y2={y(6.41)} stroke={C.ink} strokeWidth={1.5} strokeDasharray="6 6" />
      <text x={W - R + 10} y={y(6.41) + 7} fontSize={18} fill={C.ink}>真實中位數 6.41</text>
      {MIX_X.map((t, i) => (
        <text key={t} x={x(i)} y={H - Bm + 34} textAnchor="middle" fontSize={i === 0 || i === MIX_X.length - 1 ? 19 : 22}
          fontFamily={i === 0 || i === MIX_X.length - 1 ? undefined : narrow} fill={C.ink} fontWeight={i === 0 || i === MIX_X.length - 1 ? 700 : 400}>
          {t}
        </text>
      ))}
      <text x={(x(1) + x(6)) / 2} y={H - 4} textAnchor="middle" fontSize={18} fill={C.muted}>合成聲比底大幾 LU（疊完整段再鎖 −30 LUFS）</text>
      {order.map((s) => {
        const pts = s.v.map((v, i) => `${x(i)},${y(v)}`).join(' ');
        return (
          <g key={s.name}>
            <polyline points={pts} fill="none" stroke={s.hot ? C.ink : C.faint} strokeWidth={s.hot ? 5 : 2} />
            {s.v.map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r={s.hot ? 7 : 3.5} fill={s.hot ? C.ink : C.faint} />
            ))}
            {s.hot
              ? s.v.map((v, i) => (
                  <text key={'t' + i} x={x(i)} y={y(v) - 16} textAnchor="middle" fontSize={21} fontWeight={900} fontFamily={narrow} fill={C.ink}>
                    {v.toFixed(2)}
                  </text>
                ))
              : null}
          </g>
        );
      })}
      <text x={W - R + 10} y={y(7.2) + 40} fontSize={20} fontWeight={900} fill={C.ink}>MusicLDM</text>
      <text x={W - R + 10} y={y(7.2) + 64} fontSize={17} fill={C.muted}>灰線：其他 8 個底</text>
    </svg>
  );
};

const Page5: Page = () => (
  <Shell section="196 DEV · 9 BASES × 200">
    <Title sub={<>把 e00 合成聲越疊越大聲，CE 從底的分數<B>往合成聲單獨的分數移</B>；不是取兩者較高</>}>疊加 CE：落在兩者之間</Title>
    <div style={{ display: 'flex', gap: 56, flex: 1, minHeight: 0 }}>
      <div style={{ flex: 1.7, minHeight: 0 }}>
        <MixChart />
      </div>
      <div style={{ flex: 1 }}>
        <Kicker>位置＝（疊後 − 底）÷（只有合成聲 − 底）</Kicker>
        <Lines
          size={22}
          items={[
            <>同響度時位置 <B>0.33～0.47</B>：略低於正中間</>,
            <>合成聲大 18 LU（能量占 98%）位置也只有 <B>0.33～0.76</B> → 不是按能量加權</>,
            <>CE 已高的底（CFG3+neg）在同響度後<B>不再上升</B>；逐段看只有 42% 的段單調（容許 m＝0.119）</>,
            <>疊後高過兩者的段只占 3～9%</>,
          ]}
        />
      </div>
    </div>
  </Shell>
);


// ───────────────────────── 第 6 頁：拆法樂譜 ─────────────────────────
// 數字來源：research/eval/ce_pitch_octaves_20261118/scores.json（鋼琴，12 主音平均）

type Voicing = { name: string; desc: string; ce: string; d?: string; bars: number[][][] };
// 每小節是一串「同時發聲的音」；音高用從 E4（最下面那條線）起算的音級：C4=-2, E4=0, G4=2, B4=4, D5=6
const VOICINGS: Voicing[] = [
  { name: '柱狀 block', desc: '三音同時，每秒 2 下', ce: '5.90', bars: [[[-2, 0, 2], [-2, 0, 2]], [[2, 4, 6], [2, 4, 6]]] },
  { name: '上行 up', desc: '一次一音，每秒 4 下', ce: '6.90', d: '+1.00', bars: [[[-2], [0], [2], [-2]], [[2], [4], [6], [2]]] },
  { name: '下行 down', desc: '同上，順序反過來', ce: '6.89', d: '+0.99', bars: [[[2], [0], [-2], [2]], [[6], [4], [2], [6]]] },
  { name: '重複根音 rep', desc: '每秒 4 下，只彈根音', ce: '6.34', d: '+0.44', bars: [[[-2], [-2], [-2], [-2]], [[2], [2], [2], [2]]] },
];

const Staff = ({ bars, chords }: { bars: number[][][]; chords: boolean }) => {
  const sp = 13; // 線距
  const top = 60;
  const bottom = top + 4 * sp;
  const yOf = (step: number) => bottom - (step * sp) / 2;
  const x0 = 96;
  const barW = 230;
  const W = x0 + barW * 2 + 6;
  const els: ReactNode[] = [];
  bars.forEach((bar, bi) => {
    const bx = x0 + bi * barW;
    const n = bar.length;
    const xs = n === 2 ? [bx + barW * 0.28, bx + barW * 0.72] : bar.map((_, i) => bx + 34 + (i * (barW - 60)) / Math.max(1, n - 1));
    els.push(
      <text key={'c' + bi} x={bx + 20} y={18} fontSize={20} fontWeight={700} fill={C.ink}>{bi === 0 ? 'C' : 'G'}</text>,
    );
    const avg = bar.flat().reduce((a, b) => a + b, 0) / bar.flat().length;
    const up = avg < 4;
    const stemTips: number[] = [];
    bar.forEach((notes, i) => {
      const x = xs[i];
      notes.forEach((st, k) => {
        els.push(<ellipse key={`h${bi}-${i}-${k}`} cx={x} cy={yOf(st)} rx={8} ry={5.6} transform={`rotate(-20 ${x} ${yOf(st)})`} fill={C.ink} />);
        if (st <= -2) els.push(<line key={`l${bi}-${i}-${k}`} x1={x - 13} x2={x + 13} y1={yOf(-2)} y2={yOf(-2)} stroke={C.ink} strokeWidth={1.6} />);
      });
      const hi = Math.max(...notes);
      const lo = Math.min(...notes);
      const sx = up ? x + 7.2 : x - 7.2;
      const tip = up ? yOf(hi) - 3.4 * sp : yOf(lo) + 3.4 * sp;
      stemTips.push(tip);
      els.push(<line key={`s${bi}-${i}`} x1={sx} x2={sx} y1={up ? yOf(lo) : yOf(hi)} y2={tip} stroke={C.ink} strokeWidth={1.8} />);
    });
    if (!chords) {
      const by = up ? Math.min(...bar.map((ns) => yOf(Math.max(...ns)))) - 3.4 * sp : Math.max(...bar.map((ns) => yOf(Math.min(...ns)))) + 3.4 * sp;
      // 把每根符桿拉到同一條符槓
      bar.forEach((notes, i) => {
        const x = xs[i];
        const sx = up ? x + 7.2 : x - 7.2;
        els.push(<line key={`e${bi}-${i}`} x1={sx} x2={sx} y1={up ? yOf(Math.max(...notes)) : yOf(Math.min(...notes))} y2={by} stroke={C.ink} strokeWidth={1.8} />);
      });
      const bx1 = (up ? xs[0] + 7.2 : xs[0] - 7.2) - 0.9;
      const bx2 = (up ? xs[n - 1] + 7.2 : xs[n - 1] - 7.2) + 0.9;
      els.push(<rect key={'b' + bi} x={bx1} y={up ? by : by - 6} width={bx2 - bx1} height={6} fill={C.ink} />);
    }
    els.push(<line key={'bar' + bi} x1={bx + barW} x2={bx + barW} y1={top} y2={bottom} stroke={C.ink} strokeWidth={1.4} />);
  });
  return (
    <svg viewBox={`0 0 ${W} ${bottom + 30}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {[0, 1, 2, 3, 4].map((i) => (
        <line key={i} x1={8} x2={W - 6} y1={top + i * sp} y2={top + i * sp} stroke={C.ink} strokeWidth={1.1} />
      ))}
      <line x1={8} x2={8} y1={top} y2={bottom} stroke={C.ink} strokeWidth={1.4} />
      <text x={12} y={bottom + 10} fontSize={66} fontFamily='"Noto Music", "Apple Symbols", "Segoe UI Symbol", serif' fill={C.ink}>𝄞</text>
      <text x={62} y={top + 2 * sp - 2} fontSize={26} fontWeight={900} textAnchor="middle" fontFamily={narrow} fill={C.ink}>2</text>
      <text x={62} y={bottom - 2} fontSize={26} fontWeight={900} textAnchor="middle" fontFamily={narrow} fill={C.ink}>4</text>
      {els}
    </svg>
  );
};

// up − block、rep − block，依八度：2 SoundFont × 大小調 × 12 主音平均
const OCT_AVG: { label: string; v: [string, string, string] }[] = [
  { label: '上行 − 柱狀', v: ['+1.0', '+1.3', '+0.2'] },
  { label: '重複根音 − 柱狀', v: ['+0.4', '+0.6', '−0.4'] },
];

const Page6: Page = () => (
  <Shell section="CE · CHORD VOICING">
    <Title sub={<>同一組和弦拆成琶音，CE 多 <Red>+1 分</Red>；整段移到第 5 八度，加分幾乎消失</>}>同樣的和弦，換彈法</Title>
    <div style={{ display: 'flex', gap: 56, flex: 1, minHeight: 0 }}>
      <div style={{ flex: 1.25 }}>
        <Kicker>C → G，120 BPM，鋼琴（generaluser 大調第 4 八度）</Kicker>
        {VOICINGS.map((v, i) => (
          <div key={v.name} style={{ display: 'grid', gridTemplateColumns: '250px 1fr', alignItems: 'center', borderTop: i > 0 ? `1px solid ${C.rule}` : 'none', padding: '6px 0' }}>
            <div>
              <div style={{ fontSize: 26, fontWeight: 900 }}>{v.name}</div>
              <div style={{ fontSize: 20, color: C.muted }}>{v.desc}</div>
              <div style={{ fontSize: 22, marginTop: 2 }}>
                CE <span style={{ fontFamily: narrow, fontWeight: 700, fontSize: 30 }}>{v.ce}</span>
                {v.d ? <span style={{ color: C.muted }}>（{v.d}）</span> : null}
              </div>
            </div>
            <Staff bars={v.bars} chords={i === 0} />
          </div>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <Kicker>比柱狀多幾分</Kicker>
        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr repeat(3, 1fr)', alignItems: 'stretch' }}>
          {['', '第 3 八度', '第 4 八度', '第 5 八度'].map((h, j) => (
            <div key={'h' + j} style={{ fontSize: 20, fontWeight: j === 3 ? 900 : 500, color: j === 3 ? C.red : C.muted, padding: '8px 0', textAlign: j ? 'right' : 'left' }}>{h}</div>
          ))}
          {OCT_AVG.map((r) => (
            <div key={r.label} style={{ display: 'contents' }}>
              <div style={{ fontSize: 26, fontWeight: 900, padding: '10px 0', borderTop: `2px solid ${C.ink}`, display: 'flex', alignItems: 'center' }}>{r.label}</div>
              {r.v.map((v, j) => (
                <div key={j} style={{ textAlign: 'right', fontFamily: narrow, fontSize: 64, fontWeight: 700, lineHeight: 1, padding: '10px 0', borderTop: `2px solid ${C.ink}`, color: j === 2 ? C.red : C.ink }}>{v}</div>
              ))}
            </div>
          ))}
        </div>
        <div style={{ fontSize: 17, color: C.muted, marginTop: 10 }}>2 個 SoundFont × 大小調 × 12 主音平均；4 種組合方向都相同</div>
        <div style={{ marginTop: 34 }}>
          <Lines
            size={24}
            items={[
              <>拆開彈就加分，到第 5 八度<Red>幾乎不加</Red></>,
              <>重複根音到第 5 八度<Red>反而扣分</Red></>,
              <>CE 吃的是<Red>音域</Red>，不是琶音本身</>,
            ]}
          />
        </div>
      </div>
    </div>
  </Shell>
);


// ───────────────────────── 第 7 頁：瑪麗有隻小綿羊 ─────────────────────────
// 數字來源：research/eval/ce_factor_e00_20261119/scores{4,8}.json（AES CE，−30 LUFS，16 kHz，10 秒）
// 8 種音色 × 12 主音；只算所有音都在該樂器音域內的段；柱高＝原版平均 5.78 ＋ 同段配對差，鬚＝95% bootstrap CI
// 間隔拉大＝以原曲平均音高為中心放大 ×k（平均音高不變，run8）

const MARY_BASE = 5.78;
const MARY: { group: string; bars: { label: string; d: number; lo: number; hi: number; n: string }[]; hot?: boolean }[] = [
  { group: '1 原版', bars: [{ label: '', d: 0, lo: 0, hi: 0, n: '96' }] },
  { group: '2 打亂音序', bars: [{ label: '', d: 0.01, lo: -0.013, hi: 0.033, n: '288' }] },
  {
    group: '3 間隔拉大',
    bars: [
      { label: '×2', d: 0.258, lo: 0.19, hi: 0.325, n: '93' },
      { label: '×3', d: 0.208, lo: 0.114, hi: 0.296, n: '87' },
      { label: '×4', d: 0.135, lo: 0.017, hi: 0.252, n: '77' },
      { label: '×6', d: 0.012, lo: -0.099, hi: 0.126, n: '50' },
    ],
  },
  {
    group: '4 相同節奏＋隨機音高',
    hot: true,
    bars: [
      { label: '1 個八度內', d: 0.477, lo: 0.429, hi: 0.524, n: '282' },
      { label: '2 個八度內', d: 0.645, lo: 0.585, hi: 0.708, n: '251' },
    ],
  },
];

const MaryChart = () => {
  const W = 1040, H = 600, L = 70, R = 10, T = 30, Bm = 110;
  const y0 = 5.0, y1 = 6.6;
  const y = (v: number) => T + ((y1 - v) * (H - T - Bm)) / (y1 - y0);
  const bw = 74, gapIn = 14, gapOut = 64;
  const groups: { g: (typeof MARY)[number]; x0: number; w: number }[] = [];
  let cx = L + 30;
  MARY.forEach((g) => {
    const w = g.bars.length * bw + (g.bars.length - 1) * gapIn;
    groups.push({ g, x0: cx, w });
    cx += w + gapOut;
  });
  const ticks = [5.0, 5.5, 6.0, 6.5];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke={C.rule} strokeWidth={1} />
          <text x={L - 12} y={y(t) + 7} textAnchor="end" fontSize={20} fontFamily={narrow} fill={C.muted}>{t.toFixed(1)}</text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={y(MARY_BASE)} y2={y(MARY_BASE)} stroke={C.ink} strokeWidth={1.5} strokeDasharray="6 6" />
      {groups.map(({ g, x0, w }) => (
        <g key={g.group}>
          {g.bars.map((b, i) => {
            const x = x0 + i * (bw + gapIn);
            const v = MARY_BASE + b.d;
            const fill = g.hot ? C.ink : C.fill;
            return (
              <g key={b.label + i}>
                <rect x={x} y={y(v)} width={bw} height={y(y0) - y(v)} fill={fill} stroke={C.ink} strokeWidth={1.5} />
                {b.hi > b.lo ? (
                  <g stroke={C.red} strokeWidth={2.5}>
                    <line x1={x + bw / 2} x2={x + bw / 2} y1={y(MARY_BASE + b.lo)} y2={y(MARY_BASE + b.hi)} />
                    <line x1={x + bw / 2 - 9} x2={x + bw / 2 + 9} y1={y(MARY_BASE + b.hi)} y2={y(MARY_BASE + b.hi)} />
                    <line x1={x + bw / 2 - 9} x2={x + bw / 2 + 9} y1={y(MARY_BASE + b.lo)} y2={y(MARY_BASE + b.lo)} />
                  </g>
                ) : null}
                <text x={x + bw / 2} y={y(MARY_BASE + Math.max(b.hi, b.d)) - 12} textAnchor="middle" fontSize={22} fontWeight={900} fontFamily={narrow} fill={C.ink}>
                  {b.d === 0 ? v.toFixed(2) : (b.d > 0 ? '+' : '−') + Math.abs(b.d).toFixed(2)}
                </text>
                {b.label ? (
                  <text x={x + bw / 2} y={H - Bm + 28} textAnchor="middle" fontSize={g.hot ? 17 : 21} fontFamily={g.hot ? undefined : narrow} fill={C.ink}>{b.label}</text>
                ) : null}
                <text x={x + bw / 2} y={H - Bm + 50} textAnchor="middle" fontSize={15} fontFamily={narrow} fill={C.muted}>n {b.n}</text>
              </g>
            );
          })}
          <text x={x0 + w / 2} y={H - Bm + 84} textAnchor="middle" fontSize={22} fontWeight={900} fill={g.hot ? C.red : C.ink}>{g.group}</text>
        </g>
      ))}
    </svg>
  );
};

const Page7: Page = () => (
  <Shell section="CE · MARY HAD A LITTLE LAMB">
    <Title sub={<>同一首「瑪麗有隻小綿羊」：順序打亂、間隔拉大都動不太了 CE；<Red>節奏不變、換成更多不同的音高</Red>才明顯加分</>}>CE 要的是更多不同的音</Title>
    <div style={{ display: 'flex', gap: 56, flex: 1, minHeight: 0 }}>
      <div style={{ flex: 1.8, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <MaryChart />
        </div>
        <div style={{ fontSize: 16, color: C.muted, marginTop: 4 }}>
          AES CE；8 種音色 × 12 主音，−30 LUFS。柱高＝原版平均 5.78＋同段配對差，紅鬚＝95% CI；只算所有音都在樂器音域內的段（n）
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <Kicker>原曲只用 4 個音（C D E G）</Kicker>
        <Lines
          size={22}
          items={[
            <>打亂順序：<B>+0.01</B>，旋律順序 CE 看不到</>,
            <>間隔拉大（平均音高不變）：×2 <B>+0.26</B>，之後遞減，×6 歸零；超出音域就扣分</>,
            <>相同節奏＋隨機音高：1 個八度內（11 個音）<Red>+0.48</Red>、2 個八度內（17 個音）<Red>+0.65</Red>，九成的段勝原版</>,
            <>→ 加分主要來自<B>用到的不同音高數</B>，不是音程大小，也不是旋律順序</>,
          ]}
        />
      </div>
    </div>
  </Shell>
);

// ───────────────────────── 第 8 頁：不用 e00，一張一張疊上去 ─────────────────────────
// 數字來源：research/eval/ce_factor_e00_20261119/scores10f_ladder.json（run10f.py）
// 從瑪麗原曲（鋼琴、C4、4 個音）一項一項累加到第 10 輪最高分組合；AES CE，−30 LUFS，16 kHz，10 秒，seed 2000–2011 平均
// 每張卡的 Δ＝加這一項後減前一步（同 seed 配對）。demo 順序：先加伴奏、瑪麗的音高留到第 9 張才改；
// 換主音色與疊小提琴合成一張。順序會改變每張卡的 Δ（10c／10d／10e 三種順序對照見結果文件）

const STACK_BASE = 6.671;
const STACK: { name: string; how: string; v: number; d: number; note?: string }[] = [
  { name: '低音線', how: 'Synth Bass 2', v: 7.417, d: 0.746 },
  { name: '短音', how: '還是瑪麗的音，每秒 3.7 顆、斷奏', v: 7.444, d: 0.028, note: 'CI 跨零' },
  { name: '換主音色＋疊小提琴', how: '合成音色 FX 2＋同旋律小提琴 −6 dB', v: 7.586, d: 0.142 },
  { name: '小聲的鼓', how: 'rock 鼓，小 19 dB', v: 7.61, d: 0.024 },
  { name: '鼓型換 trap', how: '同音量，擊點更密', v: 7.814, d: 0.204 },
  { name: '力度隨機', how: '力度 120 ± 25', v: 8.038, d: 0.224 },
  { name: '小聲和弦墊底', how: '合唱 pad，小 18 dB', v: 8.111, d: 0.073 },
  { name: '短殘響', how: 'RT60 0.3 秒', v: 8.156, d: 0.045 },
  { name: '音高種類多', how: '同節奏，改成隨機音、不重複前一音', v: 8.228, d: 0.072 },
  { name: '大跳、音域寬', how: '跳動範圍 36 半音，中心 C#4', v: 8.361, d: 0.133 },
  { name: '小聲副旋律', how: '另一條旋律，小 20 dB', v: 8.396, d: 0.035 },
];
const STACK_MARKS: { v: number; label: string; red?: boolean }[] = [
  { v: 6.03, label: 'MusicCaps 真實錄音平均' },
  { v: 8.09, label: 'e00 合成聲' },
  { v: 8.5, label: '目標', red: true },
];

const STACK_CSS = `
.stk [data-osd-step] > .stk-card { transition: transform 520ms cubic-bezier(.2,.9,.25,1); }
.stk [data-osd-step="pending"] > .stk-card { transform: translateX(-140px); }
@keyframes stkPop { 0% { transform: translateY(18px); opacity: 0 } 100% { transform: none; opacity: 1 } }
.stk-pop { animation: stkPop 420ms cubic-bezier(.2,.9,.25,1) both; }
`;

const fmtD = (d: number) => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(2);

// 讀左邊已出現的卡數（Step 本身不回報，改看 data-osd-step）
const useRevealedCount = (ref: RefObject<HTMLDivElement | null>) => {
  const [n, setN] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setN(el.querySelectorAll('[data-osd-step="revealed"]').length);
    read();
    const mo = new MutationObserver(read);
    mo.observe(el, { subtree: true, attributes: true, attributeFilter: ['data-osd-step'] });
    return () => mo.disconnect();
  }, [ref]);
  return n;
};

// 分數從上一個值滾到新值
const useTween = (target: number, ms = 700) => {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const a = from.current, t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      const x = a + (target - a) * e;
      from.current = x;
      setV(x);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
};

const StackMeter = ({ v }: { v: number }) => {
  const W = 620, H = 150, L = 20, R = 20, lo = 5.8, hi = 8.7;
  const x = (s: number) => L + ((s - lo) * (W - L - R)) / (hi - lo);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <rect x={L} y={56} width={W - L - R} height={22} fill={C.fill} />
      <rect x={L} y={56} width={Math.max(0, x(v) - L)} height={22} fill={C.ink} />
      {STACK_MARKS.map((m, i) => (
        <g key={m.label}>
          <line x1={x(m.v)} x2={x(m.v)} y1={44} y2={90} stroke={m.red ? C.red : C.muted} strokeWidth={m.red ? 3 : 2} strokeDasharray={m.red ? '5 4' : undefined} />
          <text x={x(m.v)} y={i % 2 ? 122 : 32} textAnchor={m.v > 8.3 ? 'end' : m.v < 6.5 ? 'start' : 'middle'} fontSize={19} fontWeight={m.red ? 900 : 500} fill={m.red ? C.red : C.muted}>
            {m.label} {m.v.toFixed(2)}
          </text>
        </g>
      ))}
      {[6, 7, 8].map((t) => (
        <text key={t} x={x(t)} y={146} textAnchor="middle" fontSize={17} fontFamily={narrow} fill={C.faint}>{t.toFixed(1)}</text>
      ))}
    </svg>
  );
};

const Page8: Page = () => {
  const ref = useRef<HTMLDivElement>(null);
  const n = useRevealedCount(ref);
  const cur = n === 0 ? STACK_BASE : STACK[n - 1].v;
  const shownV = useTween(cur);
  const last = n === 0 ? null : STACK[n - 1];
  return (
    <Shell section="CE · STACKING WITHOUT E00">
      <style>{STACK_CSS}</style>
      <Title sub={<>從瑪麗原曲出發，只用 SoundFont、不用 e00，一次加一項；每一項都讓 CE 往上，最後 <B>8.40</B>，<Red>沒到 8.5</Red></>}>一張一張疊上去</Title>
      <div className="stk" style={{ display: 'flex', gap: 64, flex: 1, minHeight: 0 }}>
        <div ref={ref} style={{ flex: 1.15, display: 'flex', flexDirection: 'column', gap: 7 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '44px 1fr auto', alignItems: 'baseline', padding: '6px 16px', fontSize: 22, color: C.muted, borderBottom: `1px solid ${C.rule}` }}>
            <span style={{ fontFamily: mono }}>00</span>
            <span>瑪麗有隻小綿羊：鋼琴、C4 附近、只用 4 個音</span>
            <span style={{ fontFamily: narrow, fontWeight: 700 }}>{STACK_BASE.toFixed(2)}</span>
          </div>
          <Steps>
            {STACK.map((s, i) => (
              <Step key={s.name} duration={260}>
                <div
                  className="stk-card"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '44px 1fr auto',
                    alignItems: 'center',
                    columnGap: 8,
                    padding: '5px 16px',
                    background: i === n - 1 ? C.ink : 'transparent',
                    color: i === n - 1 ? C.paper : C.ink,
                    border: `2px solid ${C.ink}`,
                    transitionProperty: 'transform, background, color',
                  }}
                >
                  <span style={{ fontFamily: mono, fontSize: 18, opacity: 0.7 }}>{String(i + 1).padStart(2, '0')}</span>
                  <span style={{ lineHeight: 1.2 }}>
                    <span style={{ fontSize: 25, fontWeight: 900 }}>{s.name}</span>
                    <span style={{ fontSize: 18, marginLeft: 14, opacity: 0.75 }}>{s.how}</span>
                    {s.note ? <span style={{ fontSize: 16, marginLeft: 10, color: i === n - 1 ? '#f4cfcb' : C.red, fontWeight: 700 }}>（{s.note}）</span> : null}
                  </span>
                  <span style={{ fontFamily: narrow, fontSize: 25, fontWeight: 900 }}>{fmtD(s.d)}</span>
                </div>
              </Step>
            ))}
          </Steps>
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ fontFamily: mono, fontSize: 22, letterSpacing: '0.14em', color: C.muted }}>AES CE · 12 SEED 平均</div>
          <div style={{ fontFamily: narrow, fontSize: 260, fontWeight: 900, lineHeight: 0.95, letterSpacing: '-0.03em', fontVariantNumeric: 'tabular-nums' }}>
            {shownV.toFixed(2)}
          </div>
          <div style={{ height: 70, marginTop: 8 }}>
            {last ? (
              <div key={n} className="stk-pop" style={{ fontSize: 34, fontWeight: 900 }}>
                {fmtD(last.d)}<span style={{ fontWeight: 500, marginLeft: 16, fontSize: 28 }}>{last.name}</span>
              </div>
            ) : (
              <div style={{ fontSize: 28, color: C.muted }}>按一下加入第一項</div>
            )}
          </div>
          <div style={{ marginTop: 26 }}>
            <StackMeter v={shownV} />
          </div>
          <div style={{ fontSize: 22, marginTop: 14, lineHeight: 1.5, minHeight: 66, visibility: n === STACK.length ? 'visible' : 'hidden' }}>
            單檔最高 <B>8.48</B>；全換原聲樂器 8.31；鎖 −23／−16 LUFS 不會更高
          </div>
        </div>
      </div>
      <div style={{ fontSize: 16, color: C.muted, marginTop: 10 }}>
        AES CE，−30 LUFS、16 kHz、10 秒；12 個沒參與搜尋的 seed 平均，每張卡的 Δ 與前一步同 seed 配對。每拿掉任一項都扣分（CI 不跨零）。無人類評分
      </div>
    </Shell>
  );
};

export default [Page1, Page2, Page3, Page4, Page6, Page7, Page8, Page5] satisfies Page[];
