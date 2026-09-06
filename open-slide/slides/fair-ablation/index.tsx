import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';

export const design: DesignSystem = {
  palette: { bg: '#090b0f', text: '#f8fafc', accent: '#ffcf4a' },
  fonts: {
    display: '"Inter", "Noto Sans TC", "PingFang TC", system-ui, sans-serif',
    body: '"Inter", "Noto Sans TC", "PingFang TC", system-ui, sans-serif',
  },
  typeScale: { hero: 64, body: 26 },
  radius: 8,
};

const C = {
  bg: '#090b0f',
  panel: '#111722',
  panel2: '#171f2d',
  text: '#f8fafc',
  soft: '#d7deea',
  muted: '#91a0b5',
  line: 'rgba(255,255,255,.12)',
  baseline: '#60a5fa',
  caption1: '#f59e0b',
  caption2: '#2dd4bf',
  accent: '#ffcf4a',
};

const font = design.fonts.body;
const mono = '"JetBrains Mono", "SF Mono", Menlo, monospace';

type Metric = 'clap' | 'ce' | 'cu' | 'pc' | 'pq';
type Row = {
  experiment: string;
  stage: string;
  protocol: string;
  color: string;
  clap: number;
  ce: number;
  cu: number;
  pc: number;
  pq: number;
};

const rows: Row[] = [
  { experiment: 'Stage 1 NoQ baseline', stage: 'S1 400k', protocol: 'FM25 · CFG4.5', color: C.baseline, clap: 0.2003, ce: 6.5038, cu: 7.0474, pc: 4.6308, pq: 6.8943 },
  { experiment: 'Caption 1.0', stage: 'S2 +200k', protocol: 'MF1 · CFG0.5', color: C.caption1, clap: 0.1927, ce: 5.7088, cu: 6.4092, pc: 4.9280, pq: 6.3793 },
  { experiment: 'Caption 1.0', stage: 'S2 +200k', protocol: 'MF25 · CFG4.5', color: C.caption1, clap: 0.2123, ce: 5.3443, cu: 6.4768, pc: 4.0105, pq: 6.3913 },
  { experiment: 'Caption 2.0', stage: 'S1 400k', protocol: 'FM25 · CFG4.5', color: C.caption2, clap: 0.2287, ce: 6.1257, cu: 6.8474, pc: 4.3176, pq: 6.7082 },
  { experiment: 'Caption 2.0', stage: 'S2 +200k', protocol: 'MF1 · CFG0.5', color: C.caption2, clap: 0.2100, ce: 6.1519, cu: 6.5419, pc: 5.2592, pq: 6.5297 },
  { experiment: 'Caption 2.0', stage: 'S2 +200k', protocol: 'MF25 · CFG4.5', color: C.caption2, clap: 0.2419, ce: 6.2105, cu: 6.6855, pc: 4.6891, pq: 6.5823 },
];

const metrics: Metric[] = ['clap', 'ce', 'cu', 'pc', 'pq'];
const maxima = Object.fromEntries(metrics.map((metric) => [metric, Math.max(...rows.map((row) => row[metric]))])) as Record<Metric, number>;

const MetricCell = ({ row, metric }: { row: Row; metric: Metric }) => {
  const isMax = row[metric] === maxima[metric];
  return (
    <div
      style={{
        height: '100%',
        minWidth: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 7,
        padding: '0 8px',
        background: isMax ? `${C.accent}1d` : 'transparent',
        boxShadow: isMax ? `inset 0 0 0 1px ${C.accent}a8` : undefined,
        color: isMax ? C.accent : C.soft,
        fontFamily: mono,
        fontSize: 20,
        fontWeight: isMax ? 900 : 700,
      }}
    >
      {row[metric].toFixed(4)}
      {isMax && (
        <span style={{ color: C.bg, background: C.accent, borderRadius: 3, padding: '2px 4px', fontSize: 9, fontWeight: 950 }}>
          MAX
        </span>
      )}
    </div>
  );
};

const FairAblationTable: Page = () => (
  <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', background: C.bg, color: C.text, fontFamily: font }}>
    <style>{`* { box-sizing: border-box; } h1, p { margin: 0; }`}</style>
    <div style={{ position: 'absolute', left: 48, right: 48, top: 34, bottom: 30 }}>
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 24 }}>
        <div>
          <div style={{ color: C.accent, fontFamily: mono, fontSize: 14, fontWeight: 900 }}>FULL-SCALE · FAIR ABLATION</div>
          <h1 style={{ marginTop: 5, fontSize: 38, lineHeight: 1.08, fontWeight: 930 }}>Caption × Stage × Inference Protocol</h1>
          <p style={{ marginTop: 7, color: C.soft, fontSize: 17 }}>MusicCaps n=5,521 · seed 42 · NoQ · NoMask · full precision · metrics ↑</p>
        </div>
        <div style={{ display: 'flex', gap: 18, paddingBottom: 4, color: C.soft, fontSize: 14 }}>
          {[
            ['Baseline', C.baseline],
            ['Caption 1.0', C.caption1],
            ['Caption 2.0', C.caption2],
            ['Metric max', C.accent],
          ].map(([label, color]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ width: 11, height: 11, borderRadius: 2, background: color }} />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 18, border: `1px solid ${C.line}`, borderRadius: 7, overflow: 'hidden', background: C.panel }}>
        <div style={{ display: 'grid', gridTemplateColumns: '300px 150px 210px repeat(5, 1fr)', height: 42, background: C.panel2, color: C.muted, fontFamily: mono, fontSize: 13, fontWeight: 850 }}>
          {['Experiment', 'Stage', 'Protocol', 'CLAP', 'CE', 'CU', 'PC', 'PQ'].map((label, index) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: index < 3 ? 'flex-start' : 'center', padding: '0 12px', borderRight: index < 7 ? `1px solid ${C.line}` : undefined }}>
              {label}
            </div>
          ))}
        </div>
        {rows.map((row, index) => (
          <div key={`${row.experiment}-${row.stage}-${row.protocol}`} style={{ display: 'grid', gridTemplateColumns: '300px 150px 210px repeat(5, 1fr)', height: 58, borderTop: `1px solid ${C.line}`, background: index % 2 ? 'rgba(255,255,255,.018)' : 'transparent' }}>
            <div style={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 11, padding: '0 12px', borderRight: `1px solid ${C.line}`, fontSize: 18, fontWeight: 860 }}>
              <span style={{ width: 5, alignSelf: 'stretch', background: row.color }} />
              <span style={{ color: row.color, whiteSpace: 'nowrap' }}>{row.experiment}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', padding: '0 12px', borderRight: `1px solid ${C.line}`, color: C.soft, fontSize: 16, fontWeight: 750 }}>{row.stage}</div>
            <div style={{ display: 'flex', alignItems: 'center', padding: '0 12px', borderRight: `1px solid ${C.line}`, color: C.soft, fontFamily: mono, fontSize: 15 }}>{row.protocol}</div>
            {metrics.map((metric) => <MetricCell key={metric} row={row} metric={metric} />)}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 18, paddingTop: 14, borderTop: `1px solid ${C.line}` }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 28 }}>
          {[
            {
              title: 'Baseline', color: C.baseline,
              body: 'Upstream track-level Qwen caption',
              detail: '同一首歌的 retained 10s segments 共用 track-level 描述；時間粒度較粗。',
            },
            {
              title: 'Caption 1.0', color: C.caption1,
              body: 'Local first-10s Qwen · one sentence',
              detail: '直接描述訓練使用的前 10 秒；單句、資訊較精簡。',
            },
            {
              title: 'Caption 2.0', color: C.caption2,
              body: 'Local first-10s Qwen · multi-sentence',
              detail: '同樣對齊前 10 秒，但保留多句、更細的聲音與結構描述。',
            },
          ].map((item) => (
            <div key={item.title} style={{ minWidth: 0, borderLeft: `4px solid ${item.color}`, paddingLeft: 14 }}>
              <div style={{ color: item.color, fontSize: 17, fontWeight: 900 }}>{item.title}</div>
              <div style={{ marginTop: 4, color: C.text, fontSize: 15, fontWeight: 760 }}>{item.body}</div>
              <div style={{ marginTop: 4, color: C.muted, fontSize: 13, lineHeight: 1.35 }}>{item.detail}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 13, display: 'flex', justifyContent: 'space-between', color: C.muted, fontSize: 12.5 }}>
          <span>注意：MF1/CFG0.5 → MF25/CFG4.5 同時改變 steps 與 CFG，不是純步數效果。</span>
          <span style={{ fontFamily: mono }}>2026-08-13 · MeanAudio Phase 8</span>
        </div>
      </div>
    </div>
  </div>
);

export const meta: SlideMeta = {
  title: 'Full-scale caption-stage fair ablation',
  createdAt: '2026-08-13T00:00:00.000Z',
};

export default [FairAblationTable] satisfies Page[];
