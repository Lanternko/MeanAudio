import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

export const design: DesignSystem = {
  palette: { bg: '#0a0b0c', text: '#f8fafc', accent: '#2dd4bf' },
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
  text: '#f8fafc',
  soft: '#d6deea',
  muted: '#8fa1b8',
  line: 'rgba(255,255,255,.12)',
  accent: '#2dd4bf',
  baseline: '#94a3b8',
  caption1: '#60a5fa',
  caption2s1: '#f59e0b',
  caption2s2: '#c084fc',
};

const mono = '"JetBrains Mono", "SF Mono", Menlo, monospace';

const rows = [
  {
    name: 'Stage 1 NoQ baseline',
    short: 'BASELINE',
    color: C.baseline,
    clap: 0.2003,
    ce: 6.5038,
    cu: 7.0474,
    pc: 4.6308,
    pq: 6.8943,
  },
  {
    name: 'Caption 1.0 Stage 2',
    short: 'CAPTION 1.0',
    color: C.caption1,
    clap: 0.2123,
    ce: 5.3443,
    cu: 6.4768,
    pc: 4.0105,
    pq: 6.3913,
  },
  {
    name: 'Caption 2.0 Stage 1',
    short: 'CAPTION 2.0 · S1',
    color: C.caption2s1,
    clap: 0.2287,
    ce: 6.1257,
    cu: 6.8474,
    pc: 4.3176,
    pq: 6.7082,
  },
  {
    name: 'Caption 2.0 Stage 2',
    short: 'CAPTION 2.0 · S2',
    color: C.caption2s2,
    clap: 0.2419,
    ce: 6.2105,
    cu: 6.6855,
    pc: 4.6891,
    pq: 6.5823,
  },
] as const;

type Metric = 'clap' | 'ce' | 'cu' | 'pc' | 'pq';

const maxima: Record<Metric, number> = {
  clap: Math.max(...rows.map((row) => row.clap)),
  ce: Math.max(...rows.map((row) => row.ce)),
  cu: Math.max(...rows.map((row) => row.cu)),
  pc: Math.max(...rows.map((row) => row.pc)),
  pq: Math.max(...rows.map((row) => row.pq)),
};

const MetricCell = ({ metric, value, color }: { metric: Metric; value: number; color: string }) => {
  const isBest = value === maxima[metric];
  return (
    <div
      style={{
        height: 54,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 10,
        color: isBest ? C.accent : color,
        background: isBest ? 'rgba(45,212,191,.12)' : 'transparent',
        border: `1px solid ${isBest ? 'rgba(45,212,191,.48)' : 'transparent'}`,
        fontFamily: mono,
        fontSize: 22,
        fontWeight: 900,
      }}
    >
      {value.toFixed(4)}
      {isBest && <span style={{ marginLeft: 7, fontSize: 13, letterSpacing: 0.5 }}>MAX</span>}
    </div>
  );
};

const CaptionAesTable: Page = () => (
  <div
    style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden',
      background: C.bg,
      color: C.text,
      fontFamily: design.fonts.body,
    }}
  >
    <style>{`* { box-sizing: border-box; } h1, p { margin: 0; }`}</style>

    <div style={{ position: 'absolute', left: 54, right: 54, top: 38, bottom: 48 }}>
      <div style={{ color: C.accent, fontFamily: mono, fontSize: 15, fontWeight: 850, letterSpacing: 1.2 }}>
        CAPTION GRANULARITY × TRAINING STAGE
      </div>
      <h1 style={{ marginTop: 7, fontSize: 41, lineHeight: 1.08, fontWeight: 950 }}>
        CLAP 與 Audiobox Aesthetics 比較
      </h1>
      <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <p style={{ color: C.soft, fontSize: 18 }}>
          MusicCaps 5,521 · 25 steps · CFG 4.5 · NoMask · full precision · 全指標 ↑
        </p>
        <div style={{ color: C.muted, fontFamily: mono, fontSize: 14 }}>
          AES = CE · CU · PC · PQ
        </div>
      </div>

      <div
        style={{
          marginTop: 18,
          background: C.panel,
          border: `1px solid ${C.line}`,
          borderRadius: 16,
          padding: '12px 14px 14px',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '2.15fr repeat(5, 0.82fr)',
            gap: 9,
            alignItems: 'center',
            minHeight: 42,
            padding: '0 8px 8px',
            color: C.muted,
            fontFamily: mono,
            fontSize: 14,
            fontWeight: 800,
          }}
        >
          <span>EXPERIMENT</span>
          <span style={{ textAlign: 'center' }}>CLAP</span>
          <span style={{ textAlign: 'center' }}>CE</span>
          <span style={{ textAlign: 'center' }}>CU</span>
          <span style={{ textAlign: 'center' }}>PC</span>
          <span style={{ textAlign: 'center' }}>PQ</span>
        </div>

        {rows.map((row) => (
          <div
            key={row.name}
            style={{
              display: 'grid',
              gridTemplateColumns: '2.15fr repeat(5, 0.82fr)',
              gap: 9,
              alignItems: 'center',
              minHeight: 66,
              padding: '5px 8px',
              borderTop: `1px solid ${C.line}`,
              boxShadow: `inset 4px 0 0 ${row.color}`,
            }}
          >
            <div style={{ paddingLeft: 16 }}>
              <div style={{ color: row.color, fontFamily: mono, fontSize: 12, fontWeight: 900, letterSpacing: 0.7 }}>
                {row.short}
              </div>
              <div style={{ marginTop: 4, fontSize: 19, fontWeight: 850 }}>{row.name}</div>
            </div>
            <MetricCell metric="clap" value={row.clap} color={row.color} />
            <MetricCell metric="ce" value={row.ce} color={row.color} />
            <MetricCell metric="cu" value={row.cu} color={row.color} />
            <MetricCell metric="pc" value={row.pc} color={row.color} />
            <MetricCell metric="pq" value={row.pq} color={row.color} />
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 15 }}>
        {[
          {
            tag: 'BASELINE',
            title: 'Track-level caption',
            text: 'upstream Qwen · 描述整首音軌，並非只看前 10 秒',
            color: C.baseline,
          },
          {
            tag: 'CAPTION 1.0',
            title: '前 10 秒 · 一句',
            text: 'local Qwen · 單句、精簡的局部音訊描述',
            color: C.caption1,
          },
          {
            tag: 'CAPTION 2.0',
            title: '前 10 秒 · 多句',
            text: 'local Qwen · 多句、較完整的局部音訊描述；S1 / S2 共用此 caption 定義',
            color: C.caption2s2,
          },
        ].map((item) => (
          <div
            key={item.tag}
            style={{
              minHeight: 112,
              background: 'rgba(18,26,45,.72)',
              border: `1px solid ${item.color}55`,
              borderRadius: 14,
              padding: '13px 15px',
            }}
          >
            <div style={{ color: item.color, fontFamily: mono, fontSize: 12, fontWeight: 900, letterSpacing: 0.7 }}>{item.tag}</div>
            <div style={{ marginTop: 5, fontSize: 19, fontWeight: 900 }}>{item.title}</div>
            <div style={{ marginTop: 6, color: C.soft, fontSize: 15, lineHeight: 1.35 }}>{item.text}</div>
          </div>
        ))}
      </div>
    </div>

    <div
      style={{
        position: 'absolute',
        left: 54,
        right: 54,
        bottom: 17,
        display: 'flex',
        justifyContent: 'space-between',
        color: C.muted,
        fontFamily: mono,
        fontSize: 13,
      }}
    >
      <span>CE Content Enjoyment · CU Content Usefulness · PC Production Complexity · PQ Production Quality</span>
      <span style={{ color: C.accent }}>accent = column maximum</span>
    </div>
  </div>
);

export const meta: SlideMeta = {
  title: 'Caption granularity · CLAP and AES',
  createdAt: '2026-08-13T00:00:00.000Z',
};

export default [CaptionAesTable] satisfies Page[];
