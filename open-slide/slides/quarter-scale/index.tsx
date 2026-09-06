import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

export const design: DesignSystem = {
  palette: {
    bg: '#0a0b0c',
    text: '#f8fafc',
    accent: '#2dd4bf',
  },
  fonts: {
    display: '"Inter", "Noto Sans TC", "PingFang TC", system-ui, sans-serif',
    body: '"Inter", "Noto Sans TC", "PingFang TC", system-ui, sans-serif',
  },
  typeScale: { hero: 70, body: 28 },
  radius: 18,
};

const C = {
  bg: '#0a0b0c',
  panel: '#121a2d',
  panel2: '#172238',
  text: '#f8fafc',
  soft: '#d6deea',
  muted: '#8fa1b8',
  line: 'rgba(255,255,255,.12)',
  teal: '#2dd4bf',
  amber: '#f59e0b',
  red: '#fb7185',
  green: '#4ade80',
  blue: '#60a5fa',
  gray: '#94a3b8',
  purple: '#c084fc',
  pink: '#f472b6',
};

const font = design.fonts.body;
const mono = '"JetBrains Mono", "SF Mono", Menlo, monospace';

/** All primary MusicCaps endpoints at quarter scale (S1=100k, S2=50k) unless noted. */
type Row = {
  family: 'caption10s' | 'noq' | 'bucket_q' | 's2q' | 'ref';
  name: string;
  k?: number | null;
  strategy?: string | null;
  clap: number;
  ce: number;
  cu: number;
  pc: number;
  pq: number;
  endpoint: string;
  note?: string;
  color: string;
  date: string;
};

const rows: Row[] = [
  // Today caption bugfix
  {
    family: 'caption10s',
    name: 'Caption10s NoQ (today)',
    clap: 0.1734,
    ce: 5.7702,
    cu: 6.4237,
    pc: 5.0037,
    pq: 6.3559,
    endpoint: 'NoQ · MF1 CFG0.5',
    note: 'Qwen first-10s captions; NPZ text in-place aligned',
    color: C.green,
    date: '2026-08-06',
  },
  // Full-scale reference
  {
    family: 'ref',
    name: 'Official NoQ FULL (ref)',
    clap: 0.1735,
    ce: 6.6156,
    cu: 6.9649,
    pc: 5.2447,
    pq: 6.8596,
    endpoint: 'NoQ · MF1 CFG0.5',
    note: 'Full 400k+200k; old 30s captions',
    color: C.purple,
    date: '2026-08-01',
  },
  // Last week NoQ quarter (old captions)
  {
    family: 'noq',
    name: 'Bucket NoQ quarter',
    clap: 0.1659,
    ce: 6.2572,
    cu: 6.6886,
    pc: 5.2848,
    pq: 6.5391,
    endpoint: 'NoQ · MF1 CFG0.5',
    note: 'Old captions; same quarter budget',
    color: C.gray,
    date: '2026-07-26',
  },
  // S2Q from NoQ K ablation (last week)
  {
    family: 's2q',
    name: 'S2Q K3 balanced',
    k: 3,
    strategy: 'balanced',
    clap: 0.168,
    ce: 6.3001,
    cu: 6.6572,
    pc: 5.2884,
    pq: 6.5084,
    endpoint: 'q=9 · MF1 CFG0.5',
    note: 'NoQ S1 → MeanSim-Q S2 only',
    color: C.teal,
    date: '2026-07-30',
  },
  {
    family: 's2q',
    name: 'S2Q K3 fixed',
    k: 3,
    strategy: 'fixed',
    clap: 0.1669,
    ce: 6.2991,
    cu: 6.6994,
    pc: 5.2781,
    pq: 6.5724,
    endpoint: 'q=9 · MF1 CFG0.5',
    note: 'NoQ S1 → MeanSim-Q S2 only',
    color: C.teal,
    date: '2026-07-31',
  },
  {
    family: 's2q',
    name: 'S2Q K2 balanced',
    k: 2,
    strategy: 'balanced',
    clap: 0.1667,
    ce: 6.2479,
    cu: 6.6225,
    pc: 5.2727,
    pq: 6.467,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.teal,
    date: '2026-07-30',
  },
  {
    family: 's2q',
    name: 'S2Q K2 fixed',
    k: 2,
    strategy: 'fixed',
    clap: 0.1664,
    ce: 6.1889,
    cu: 6.6308,
    pc: 5.2724,
    pq: 6.4894,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.teal,
    date: '2026-07-30',
  },
  {
    family: 's2q',
    name: 'S2Q K5 fixed',
    k: 5,
    strategy: 'fixed',
    clap: 0.164,
    ce: 6.1855,
    cu: 6.5915,
    pc: 5.2564,
    pq: 6.469,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.teal,
    date: '2026-07-31',
  },
  {
    family: 's2q',
    name: 'S2Q K5 balanced',
    k: 5,
    strategy: 'balanced',
    clap: 0.1639,
    ce: 6.2002,
    cu: 6.5743,
    pc: 5.2516,
    pq: 6.4364,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.teal,
    date: '2026-07-31',
  },
  // Full-Q bucket K ablation (both stages Q)
  {
    family: 'bucket_q',
    name: 'Bucket K3 bal (full-Q)',
    k: 3,
    strategy: 'balanced',
    clap: 0.1573,
    ce: 6.0844,
    cu: 6.3622,
    pc: 5.4182,
    pq: 6.1782,
    endpoint: 'q=9 · MF1 CFG0.5',
    note: 'S1+S2 both MeanSim-Q',
    color: C.amber,
    date: '2026-07-29',
  },
  {
    family: 'bucket_q',
    name: 'Bucket K5 bal (full-Q)',
    k: 5,
    strategy: 'balanced',
    clap: 0.1547,
    ce: 6.5946,
    cu: 7.0416,
    pc: 5.5125,
    pq: 6.9318,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.amber,
    date: '2026-07-29',
  },
  {
    family: 'bucket_q',
    name: 'Bucket K5 fixed (full-Q)',
    k: 5,
    strategy: 'fixed',
    clap: 0.151,
    ce: 5.7078,
    cu: 6.2116,
    pc: 5.3437,
    pq: 5.9346,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.amber,
    date: '2026-07-30',
  },
  {
    family: 'bucket_q',
    name: 'Bucket K10 bal (full-Q)',
    k: 10,
    strategy: 'balanced',
    clap: 0.1488,
    ce: 5.4735,
    cu: 5.9818,
    pc: 5.4138,
    pq: 5.8815,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.amber,
    date: '2026-07-29',
  },
  {
    family: 'bucket_q',
    name: 'Bucket K2 bal (full-Q)',
    k: 2,
    strategy: 'balanced',
    clap: 0.1473,
    ce: 6.3347,
    cu: 6.6302,
    pc: 5.1817,
    pq: 6.4947,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.amber,
    date: '2026-07-26',
  },
  {
    family: 'bucket_q',
    name: 'Bucket K10 fixed (full-Q)',
    k: 10,
    strategy: 'fixed',
    clap: 0.1318,
    ce: 5.2386,
    cu: 5.8087,
    pc: 5.2434,
    pq: 5.6926,
    endpoint: 'q=9 · MF1 CFG0.5',
    color: C.red,
    date: '2026-07-30',
  },
];

const sortedByClap = [...rows].sort((a, b) => b.clap - a.clap);

const globalCss = `
  * { box-sizing: border-box; }
  .mean-slide h1, .mean-slide h2, .mean-slide p { margin: 0; }
`;

const Shell = ({ children, page }: { children: React.ReactNode; page: string }) => (
  <div
    className="mean-slide"
    style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden',
      background: C.bg,
      color: C.text,
      fontFamily: font,
    }}
  >
    <style>{globalCss}</style>
    {children}
    <div
      style={{
        position: 'absolute',
        left: 56,
        right: 56,
        bottom: 22,
        display: 'flex',
        justifyContent: 'space-between',
        color: C.muted,
        fontSize: 17,
        fontFamily: mono,
      }}
    >
      <span>MeanAudio · Quarter scale · MusicCaps n=5521 · CLAP / AES ↑</span>
      <span>{page}</span>
    </div>
  </div>
);

const Header = ({ title, subtitle }: { title: string; subtitle: string }) => (
  <div style={{ marginBottom: 18 }}>
    <h1 style={{ fontSize: 48, lineHeight: 1.08, fontWeight: 900 }}>{title}</h1>
    <p style={{ marginTop: 8, color: C.soft, fontSize: 22 }}>{subtitle}</p>
  </div>
);

const Chip = ({ label, color }: { label: string; color: string }) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 12px',
      borderRadius: 999,
      border: `1px solid ${color}55`,
      background: `${color}18`,
      color: C.soft,
      fontSize: 16,
    }}
  >
    <span style={{ width: 10, height: 10, borderRadius: 3, background: color }} />
    {label}
  </span>
);

// ── Page 1: Cover ──────────────────────────────────────────
const Cover: Page = () => (
  <Shell page="01 / 06">
    <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(900px 500px at 20% 10%, #134e4a66, transparent), radial-gradient(700px 400px at 90% 80%, #1e3a5f55, transparent)' }} />
    <div style={{ position: 'absolute', left: 80, right: 80, top: 120, zIndex: 1 }}>
      <div style={{ color: C.teal, fontFamily: mono, fontSize: 22, fontWeight: 800, letterSpacing: 2 }}>
        PHASE 8 · QUARTER SCALE
      </div>
      <h1 style={{ marginTop: 18, fontSize: 78, fontWeight: 950, lineHeight: 1.05, maxWidth: 1400 }}>
        Quarter-scale 全部結果
      </h1>
      <p style={{ marginTop: 24, fontSize: 30, color: C.soft, maxWidth: 1300, lineHeight: 1.35 }}>
        上週 K-ablation（bucket full-Q / S2Q-from-NoQ）＋ 今日 caption–audio 視窗修正（10s Qwen recaption NoQ）
      </p>
      <div style={{ marginTop: 40, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <Chip label="S1 100k + S2 50k" color={C.blue} />
        <Chip label="MusicCaps 5521" color={C.blue} />
        <Chip label="MeanFlow1 CFG0.5" color={C.blue} />
        <Chip label="Caption10s NoQ 0.1734" color={C.green} />
        <Chip label="Full NoQ ref 0.1735" color={C.purple} />
      </div>
      <div
        style={{
          marginTop: 56,
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 18,
        }}
      >
        {[
          ['今日修正', 'Caption10s NoQ', '0.1734', C.green, '≈ full NoQ baseline'],
          ['上週最佳 S2Q', 'K3 balanced', '0.1680', C.teal, 'NoQ S1 → Q S2'],
          ['上週 NoQ Qtr', 'Bucket NoQ', '0.1659', C.gray, 'old 30s captions'],
          ['上週 best full-Q', 'K3 balanced', '0.1573', C.amber, 'both stages Q'],
        ].map(([tag, name, val, color, sub]) => (
          <div
            key={String(name)}
            style={{
              padding: '22px 24px',
              borderRadius: 16,
              border: `1px solid ${color}55`,
              background: 'rgba(18,26,45,.9)',
            }}
          >
            <div style={{ color: C.muted, fontSize: 16, fontFamily: mono }}>{tag}</div>
            <div style={{ marginTop: 8, fontSize: 22, fontWeight: 800 }}>{name}</div>
            <div style={{ marginTop: 10, fontSize: 42, fontWeight: 950, fontFamily: mono, color: color as string }}>
              {val}
            </div>
            <div style={{ marginTop: 6, color: C.soft, fontSize: 17 }}>{sub}</div>
          </div>
        ))}
      </div>
    </div>
  </Shell>
);

// ── Page 2: Ranking bar ────────────────────────────────────
const Ranking: Page = () => {
  const min = 0.125;
  const max = 0.18;
  return (
    <Shell page="02 / 06">
      <div style={{ position: 'absolute', left: 56, right: 56, top: 40, bottom: 56 }}>
        <Header
          title="CLAP ranking · all quarter primary endpoints"
          subtitle="橫條視覺放大 0.125–0.180 · 綠=今日 caption 修正 · 紫=full 參考 · 青=S2Q · 琥珀=full-Q bucket · 灰=舊 NoQ"
        />
        <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          <Chip label="Caption10s (today)" color={C.green} />
          <Chip label="Full NoQ ref" color={C.purple} />
          <Chip label="S2Q-from-NoQ" color={C.teal} />
          <Chip label="Bucket full-Q" color={C.amber} />
          <Chip label="Bucket NoQ" color={C.gray} />
        </div>
        <div
          style={{
            background: 'rgba(18,26,45,.94)',
            border: `1px solid ${C.line}`,
            borderRadius: 14,
            overflow: 'hidden',
            maxHeight: 820,
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '340px 1fr 110px 90px',
              gap: 12,
              padding: '10px 14px',
              color: C.muted,
              fontFamily: mono,
              fontSize: 15,
              borderBottom: `1px solid ${C.line}`,
            }}
          >
            <span>實驗</span>
            <span>CLAP</span>
            <span>CLAP</span>
            <span>CE</span>
          </div>
          {sortedByClap.map((r) => {
            const w = Math.max(0, ((r.clap - min) / (max - min)) * 100);
            return (
              <div
                key={r.name}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '340px 1fr 110px 90px',
                  gap: 12,
                  alignItems: 'center',
                  padding: '6px 14px',
                  borderBottom: `1px solid ${C.line}`,
                  minHeight: 44,
                }}
              >
                <div style={{ fontSize: 18, fontWeight: 760, lineHeight: 1.15 }}>{r.name}</div>
                <div style={{ height: 22, borderRadius: 5, background: 'rgba(255,255,255,.05)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${w}%`,
                      minWidth: 8,
                      borderRadius: 5,
                      background: `linear-gradient(90deg, ${r.color}77, ${r.color})`,
                    }}
                  />
                </div>
                <div style={{ fontFamily: mono, fontSize: 20, fontWeight: 900, color: r.color }}>
                  {r.clap.toFixed(4)}
                </div>
                <div style={{ fontFamily: mono, fontSize: 18, color: C.soft }}>{r.ce.toFixed(2)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </Shell>
  );
};

// ── Page 3: Caption bug fix ────────────────────────────────
const CaptionFix: Page = () => (
  <Shell page="03 / 06">
    <div style={{ position: 'absolute', left: 56, right: 56, top: 40, bottom: 56 }}>
      <Header
        title="今日：Caption–audio 視窗修正"
        subtitle="Bug：caption 用 30s 全文，訓練 audio 只取前 10s · Fix：Qwen recaption on first-10s crop + NPZ text in-place"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: 22 }}>
        <div style={{ display: 'grid', gap: 16 }}>
          {[
            {
              title: 'Before · Bucket NoQ quarter',
              clap: '0.1659',
              detail: 'Old catalog captions (often 30s-aware) · S1 100k + S2 50k · NoQ',
              color: C.gray,
            },
            {
              title: 'After · Caption10s NoQ quarter',
              clap: '0.1734',
              detail: 'Qwen first-10s recaption · text_* rewritten in NPZ · S1 100k + S2 50k · NoQ',
              color: C.green,
            },
            {
              title: 'Reference · Official NoQ FULL',
              clap: '0.1735',
              detail: 'Old captions · S1 400k + S2 200k · NoQ · same MusicCaps protocol',
              color: C.purple,
            },
          ].map((c) => (
            <div
              key={c.title}
              style={{
                padding: '26px 28px',
                borderRadius: 16,
                border: `1px solid ${c.color}55`,
                borderLeft: `7px solid ${c.color}`,
                background: 'rgba(18,26,45,.94)',
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 800 }}>{c.title}</div>
              <div style={{ marginTop: 10, fontFamily: mono, fontSize: 52, fontWeight: 950, color: c.color }}>
                {c.clap}
              </div>
              <div style={{ marginTop: 10, color: C.soft, fontSize: 19, lineHeight: 1.35 }}>{c.detail}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'grid', gap: 16, alignContent: 'start' }}>
          <div style={{ padding: 26, borderRadius: 16, background: 'rgba(18,26,45,.94)', border: `1px solid ${C.line}` }}>
            <div style={{ color: C.teal, fontWeight: 900, fontSize: 22 }}>Δ CLAP</div>
            <div style={{ marginTop: 14, display: 'grid', gap: 12 }}>
              {[
                ['Caption10s − old NoQ quarter', '+0.0075', C.green],
                ['Caption10s − full NoQ ref', '−0.0001', C.soft],
                ['1/4 steps ≈ full NoQ CLAP', 'matched', C.green],
              ].map(([l, v, c]) => (
                <div key={String(l)} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <span style={{ color: C.soft, fontSize: 20 }}>{l}</span>
                  <span style={{ fontFamily: mono, fontSize: 28, fontWeight: 900, color: c as string }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ padding: 26, borderRadius: 16, background: 'rgba(18,26,45,.94)', border: `1px solid ${C.line}` }}>
            <div style={{ color: C.amber, fontWeight: 900, fontSize: 22 }}>AES (Caption10s quarter)</div>
            <div
              style={{
                marginTop: 16,
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 12,
                fontFamily: mono,
                fontSize: 22,
              }}
            >
              {[
                ['CE', '5.770'],
                ['CU', '6.424'],
                ['PC', '5.004'],
                ['PQ', '6.356'],
              ].map(([k, v]) => (
                <div key={k} style={{ padding: '12px 14px', borderRadius: 10, background: 'rgba(255,255,255,.04)' }}>
                  <div style={{ color: C.muted, fontSize: 15 }}>{k}</div>
                  <div style={{ fontWeight: 900, marginTop: 4 }}>{v}</div>
                </div>
              ))}
            </div>
            <p style={{ marginTop: 14, color: C.muted, fontSize: 17, lineHeight: 1.4 }}>
              AES CE/CU/PQ 低於舊 NoQ quarter / full — CLAP 對齊但美感維度尚未同步拉高。
            </p>
          </div>
          <div style={{ padding: 22, borderRadius: 16, background: 'rgba(45,212,191,.08)', border: `1px solid ${C.teal}44` }}>
            <div style={{ fontWeight: 900, fontSize: 20, color: C.teal }}>CLAP gate (n=1024 paired)</div>
            <div style={{ marginTop: 10, color: C.soft, fontSize: 18, lineHeight: 1.4, fontFamily: mono }}>
              meanΔ +0.0345 · medianΔ +0.0332 · frac+ 61.6% · frac≥0.02 55.5%
              <br />
              CI95 [0.0282, 0.0409] · only p25 failed (−0.0343) → manual pass
            </div>
          </div>
        </div>
      </div>
    </div>
  </Shell>
);

// ── Page 4: S2Q K ablation ─────────────────────────────────
const S2QTable: Page = () => {
  const s2q = rows.filter((r) => r.family === 's2q').sort((a, b) => b.clap - a.clap);
  const noq = rows.find((r) => r.family === 'noq')!;
  return (
    <Shell page="04 / 06">
      <div style={{ position: 'absolute', left: 56, right: 56, top: 40, bottom: 56 }}>
        <Header
          title="上週 · S2Q-from-NoQ K ablation"
          subtitle="Design: NoQ S1 (100k) → MeanSim-Q S2 (50k only) · Eval q=9 · balanced vs fixed · baseline NoQ quarter CLAP=0.1659"
        />
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 0.75fr', gap: 20 }}>
          <div style={{ border: `1px solid ${C.line}`, borderRadius: 14, overflow: 'hidden', background: 'rgba(18,26,45,.94)' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '90px 120px 1fr 100px 90px 90px 90px 90px',
                padding: '12px 14px',
                background: C.panel2,
                color: C.muted,
                fontFamily: mono,
                fontSize: 15,
                fontWeight: 800,
              }}
            >
              {['K', 'Strategy', 'Endpoint', 'CLAP', 'CE', 'CU', 'PC', 'PQ'].map((h) => (
                <div key={h}>{h}</div>
              ))}
            </div>
            {/* baseline row */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '90px 120px 1fr 100px 90px 90px 90px 90px',
                padding: '12px 14px',
                borderBottom: `1px solid ${C.line}`,
                background: 'rgba(148,163,184,.08)',
                fontSize: 18,
                alignItems: 'center',
              }}
            >
              <div style={{ color: C.muted }}>—</div>
              <div style={{ color: C.muted }}>NoQ</div>
              <div>Bucket NoQ baseline</div>
              <div style={{ fontFamily: mono, fontWeight: 900, color: C.gray }}>{noq.clap.toFixed(4)}</div>
              <div style={{ fontFamily: mono, color: C.soft }}>{noq.ce.toFixed(2)}</div>
              <div style={{ fontFamily: mono, color: C.soft }}>{noq.cu.toFixed(2)}</div>
              <div style={{ fontFamily: mono, color: C.soft }}>{noq.pc.toFixed(2)}</div>
              <div style={{ fontFamily: mono, color: C.soft }}>{noq.pq.toFixed(2)}</div>
            </div>
            {s2q.map((r) => {
              const d = r.clap - noq.clap;
              return (
                <div
                  key={r.name}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '90px 120px 1fr 100px 90px 90px 90px 90px',
                    padding: '11px 14px',
                    borderBottom: `1px solid ${C.line}`,
                    fontSize: 18,
                    alignItems: 'center',
                  }}
                >
                  <div style={{ fontFamily: mono, fontWeight: 900, color: C.teal }}>K{r.k}</div>
                  <div>{r.strategy}</div>
                  <div>
                    {r.name.replace('S2Q ', '')}
                    <span style={{ marginLeft: 10, fontFamily: mono, fontSize: 15, color: d >= 0 ? C.green : C.red }}>
                      {d >= 0 ? '+' : ''}
                      {d.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ fontFamily: mono, fontWeight: 900, color: C.teal }}>{r.clap.toFixed(4)}</div>
                  <div style={{ fontFamily: mono, color: C.soft }}>{r.ce.toFixed(2)}</div>
                  <div style={{ fontFamily: mono, color: C.soft }}>{r.cu.toFixed(2)}</div>
                  <div style={{ fontFamily: mono, color: C.soft }}>{r.pc.toFixed(2)}</div>
                  <div style={{ fontFamily: mono, color: C.soft }}>{r.pq.toFixed(2)}</div>
                </div>
              );
            })}
          </div>
          <div style={{ display: 'grid', gap: 14, alignContent: 'start' }}>
            {[
              ['Best S2Q', 'K3 balanced', '0.1680', '+0.0021 vs NoQ'],
              ['K2 ≈ K3 fixed', '0.1664–0.1669', 'tight band', 'small Δ'],
              ['K5 dips', '0.1639–0.1640', 'below NoQ', '−0.002'],
              ['Takeaway', 'mild gain', 'best at K=3 bal', 'NoQ S1 is key'],
            ].map(([a, b, c, d]) => (
              <div
                key={String(a)}
                style={{
                  padding: '18px 20px',
                  borderRadius: 14,
                  border: `1px solid ${C.line}`,
                  background: 'rgba(18,26,45,.94)',
                }}
              >
                <div style={{ color: C.muted, fontFamily: mono, fontSize: 14 }}>{a}</div>
                <div style={{ marginTop: 6, fontSize: 24, fontWeight: 900 }}>{b}</div>
                <div style={{ marginTop: 4, fontFamily: mono, fontSize: 28, color: C.teal, fontWeight: 900 }}>{c}</div>
                <div style={{ marginTop: 4, color: C.soft, fontSize: 17 }}>{d}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Shell>
  );
};

// ── Page 5: Bucket full-Q K ablation ───────────────────────
const BucketQTable: Page = () => {
  const bq = rows.filter((r) => r.family === 'bucket_q').sort((a, b) => b.clap - a.clap);
  const noq = rows.find((r) => r.family === 'noq')!;
  return (
    <Shell page="05 / 06">
      <div style={{ position: 'absolute', left: 56, right: 56, top: 40, bottom: 56 }}>
        <Header
          title="上週 · Bucket full-Q K ablation"
          subtitle="S1+S2 都用 MeanSim-Q · Eval high_q9 · 全部低於 NoQ quarter（0.1659）· fixed 通常更差"
        />
        <div style={{ border: `1px solid ${C.line}`, borderRadius: 14, overflow: 'hidden', background: 'rgba(18,26,45,.94)' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '80px 120px 220px 110px 100px 100px 100px 100px 1fr',
              padding: '12px 14px',
              background: C.panel2,
              color: C.muted,
              fontFamily: mono,
              fontSize: 15,
              fontWeight: 800,
            }}
          >
            {['K', 'Strategy', 'Name', 'CLAP', 'CE', 'CU', 'PC', 'PQ', 'Δ vs NoQ'].map((h) => (
              <div key={h}>{h}</div>
            ))}
          </div>
          {bq.map((r) => {
            const d = r.clap - noq.clap;
            return (
              <div
                key={r.name}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 120px 220px 110px 100px 100px 100px 100px 1fr',
                  padding: '12px 14px',
                  borderBottom: `1px solid ${C.line}`,
                  fontSize: 19,
                  alignItems: 'center',
                }}
              >
                <div style={{ fontFamily: mono, fontWeight: 900, color: C.amber }}>K{r.k}</div>
                <div>{r.strategy}</div>
                <div style={{ fontWeight: 700 }}>{r.name.replace('Bucket ', '')}</div>
                <div style={{ fontFamily: mono, fontWeight: 900, color: r.color }}>{r.clap.toFixed(4)}</div>
                <div style={{ fontFamily: mono, color: C.soft }}>{r.ce.toFixed(2)}</div>
                <div style={{ fontFamily: mono, color: C.soft }}>{r.cu.toFixed(2)}</div>
                <div style={{ fontFamily: mono, color: C.soft }}>{r.pc.toFixed(2)}</div>
                <div style={{ fontFamily: mono, color: C.soft }}>{r.pq.toFixed(2)}</div>
                <div style={{ fontFamily: mono, fontWeight: 800, color: C.red }}>
                  {d.toFixed(4)} ({((d / noq.clap) * 100).toFixed(1)}%)
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {[
            ['Best full-Q', 'K3 balanced 0.1573', 'still −0.0086 vs NoQ'],
            ['Worst', 'K10 fixed 0.1318', 'deep collapse'],
            ['Pattern', 'balanced > fixed', 'higher K often hurts'],
          ].map(([t, a, b]) => (
            <div
              key={t}
              style={{
                padding: '20px 22px',
                borderRadius: 14,
                border: `1px solid ${C.line}`,
                background: 'rgba(18,26,45,.94)',
              }}
            >
              <div style={{ color: C.amber, fontWeight: 900, fontSize: 18 }}>{t}</div>
              <div style={{ marginTop: 8, fontSize: 24, fontWeight: 900 }}>{a}</div>
              <div style={{ marginTop: 6, color: C.soft, fontSize: 18 }}>{b}</div>
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
};

// ── Page 6: Full detail + takeaways ────────────────────────
const DetailAll: Page = () => (
  <Shell page="06 / 06">
    <div style={{ position: 'absolute', left: 40, right: 40, top: 36, bottom: 52 }}>
      <Header
        title="全部 quarter 主端點 · 明細"
        subtitle="來源：logs/*FINAL_METRICS.json · Protocol 註記於 endpoint 欄 · Full NoQ 僅作參考（非 quarter budget）"
      />
      <div style={{ border: `1px solid ${C.line}`, borderRadius: 12, overflow: 'hidden', background: 'rgba(18,26,45,.96)' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '250px 90px 100px 200px 95px 78px 78px 78px 78px 1fr',
            padding: '10px 12px',
            background: C.panel2,
            color: C.muted,
            fontFamily: mono,
            fontSize: 13,
            fontWeight: 800,
          }}
        >
          {['Name', 'Date', 'Family', 'Endpoint', 'CLAP', 'CE', 'CU', 'PC', 'PQ', 'Note'].map((h) => (
            <div key={h}>{h}</div>
          ))}
        </div>
        {sortedByClap.map((r) => (
          <div
            key={r.name}
            style={{
              display: 'grid',
              gridTemplateColumns: '250px 90px 100px 200px 95px 78px 78px 78px 78px 1fr',
              padding: '8px 12px',
              borderBottom: `1px solid ${C.line}`,
              fontSize: 15,
              alignItems: 'center',
              minHeight: 40,
            }}
          >
            <div style={{ fontWeight: 780, color: r.color }}>{r.name}</div>
            <div style={{ fontFamily: mono, color: C.muted, fontSize: 13 }}>{r.date.slice(5)}</div>
            <div style={{ color: C.soft, fontSize: 14 }}>{r.family}</div>
            <div style={{ color: C.soft, fontSize: 14 }}>{r.endpoint}</div>
            <div style={{ fontFamily: mono, fontWeight: 900, color: r.color }}>{r.clap.toFixed(4)}</div>
            <div style={{ fontFamily: mono, color: C.soft }}>{r.ce.toFixed(2)}</div>
            <div style={{ fontFamily: mono, color: C.soft }}>{r.cu.toFixed(2)}</div>
            <div style={{ fontFamily: mono, color: C.soft }}>{r.pc.toFixed(2)}</div>
            <div style={{ fontFamily: mono, color: C.soft }}>{r.pq.toFixed(2)}</div>
            <div style={{ color: C.muted, fontSize: 13, lineHeight: 1.2 }}>{r.note || (r.k ? `K=${r.k} ${r.strategy}` : '')}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        {[
          ['1', 'Caption 對齊 audio 視窗有效', '1/4 NoQ 已追上 full NoQ CLAP（0.1734 ≈ 0.1735）'],
          ['2', 'S2Q 溫和優於 NoQ', '最佳 K3 bal +0.0021；K5 反而掉'],
          ['3', 'Full-Q bucket 全面輸 NoQ', 'Q 兩階段訓練在 quarter 上傷害 CLAP'],
        ].map(([n, t, d]) => (
          <div
            key={n}
            style={{
              padding: '14px 16px',
              borderRadius: 12,
              background: 'rgba(18,26,45,.94)',
              border: `1px solid ${C.line}`,
            }}
          >
            <div style={{ color: C.teal, fontFamily: mono, fontWeight: 900 }}>TAKEAWAY {n}</div>
            <div style={{ marginTop: 6, fontWeight: 900, fontSize: 18 }}>{t}</div>
            <div style={{ marginTop: 4, color: C.soft, fontSize: 16, lineHeight: 1.3 }}>{d}</div>
          </div>
        ))}
      </div>
    </div>
  </Shell>
);

export const meta: SlideMeta = {
  title: 'Quarter-scale: K-ablation + Caption10s fix',
};

export default [Cover, Ranking, CaptionFix, S2QTable, BucketQTable, DetailAll] satisfies Page[];
