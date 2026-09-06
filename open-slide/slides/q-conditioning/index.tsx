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
  teal2: '#0f766e',
  amber: '#f59e0b',
  red: '#fb7185',
  green: '#4ade80',
  blue: '#60a5fa',
  gray: '#94a3b8',
};

const font = design.fonts.body;
const mono = '"JetBrains Mono", "SF Mono", Menlo, monospace';

const experiments = [
  {
    group: '歷史實驗',
    name: '舊 Stage2 - Q',
    clap: 0.1975,
    ce: 6.02,
    s1: 'nullQ，q=10（舊版 runner bug）',
    s2: '逐筆對齊 MeanSim-Q，q=0–9',
    steps: '400K + 200K',
    eval: 'MusicCaps，q=9',
    color: C.teal,
  },
  {
    group: '歷史實驗',
    name: 'Full - Q (q=9)',
    clap: 0.1748,
    ce: 5.5436,
    s1: '逐筆對齊 Q，q=0–9',
    s2: '逐筆對齊 Q，q=0–9',
    steps: '400K + 200K',
    eval: 'MusicCaps，q=9',
    color: C.red,
  },
  {
    group: '歷史實驗',
    name: '舊 No-Q baseline',
    clap: 0.1851,
    ce: 5.91,
    s1: 'nullQ，q=10',
    s2: '設定 No-Q；舊 runner 實際讀 q=0–9（bug）',
    steps: '400K + 200K',
    eval: 'MusicCaps，No-Q',
    color: C.gray,
  },
  {
    group: '歷史實驗',
    name: '舊 No-Q baseline eval = 9',
    clap: 0.1907,
    ce: 5.79,
    s1: '同舊 No-Q baseline',
    s2: '同一 checkpoint',
    steps: '同一 checkpoint',
    eval: 'MusicCaps，q=9',
    color: C.blue,
  },
  {
    group: '本週 clean experiments',
    name: 'Full-Q',
    clap: 0.1684,
    ce: 5.36,
    s1: '逐筆對齊 Q，q=0–9',
    s2: '逐筆對齊 Q，q=0–9',
    steps: '400K + 200K',
    eval: 'MusicCaps，q=9',
    color: C.red,
  },
  {
    group: '本週 clean experiments',
    name: 'No-Q（no bug）',
    clap: 0.1888,
    ce: 5.73,
    s1: 'nullQ，q=10',
    s2: 'nullQ，q=10',
    steps: '400K + 200K',
    eval: 'MusicCaps，No-Q / q=10',
    color: C.green,
  },
  {
    group: '本週 clean experiments',
    name: 'S2 - Q (no bug)',
    clap: 0.1426,
    ce: 4.96,
    s1: 'nullQ，q=10',
    s2: '逐筆對齊 Q，q=0–9',
    steps: 'S1 400K + S2 200K',
    eval: 'MusicCaps，q=9',
    color: C.teal,
  },
  {
    group: '本週 clean experiments',
    name: 'S2 Shuffled-Q',
    clap: 0.1591,
    ce: 5.34,
    s1: 'nullQ，q=10',
    s2: '打亂後 Q，q=0–9；不與 audio 對齊',
    steps: 'S1 400K + S2 200K',
    eval: 'MusicCaps，q=9',
    color: C.amber,
  },
  {
    group: '本週 clean experiments',
    name: 'No-Q600K + fine-tune Q',
    clap: 0.1823,
    ce: 5.35,
    s1: 'No-Q 600K，q=10',
    s2: '逐筆對齊 Q fine-tune',
    steps: '100K；600K→700K',
    eval: 'MusicCaps，q=9',
    color: C.teal,
  },
  {
    group: '本週 clean experiments',
    name: 'No-Q600K  + fine-tune Shuffled-Q',
    clap: 0.1848,
    ce: 5.41,
    s1: 'No-Q 600K，q=10',
    s2: 'Shuffled-Q fine-tune',
    steps: '100K；600K→700K',
    eval: 'MusicCaps，q=9',
    color: C.amber,
  },
] as const;

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
        left: 66,
        right: 66,
        bottom: 24,
        display: 'flex',
        justifyContent: 'space-between',
        color: C.muted,
        fontSize: 18,
        fontFamily: mono,
      }}
    >
      <span>MeanAudio · MusicCaps · CLAP / CE ↑</span>
      <span>{page}</span>
    </div>
  </div>
);

const Header = ({ title, subtitle }: { title: string; subtitle: string }) => (
  <div style={{ position: 'relative', zIndex: 1, marginBottom: 22 }}>
    <h1 style={{ fontSize: 54, lineHeight: 1.08, fontWeight: 900 }}>{title}</h1>
    <p style={{ marginTop: 9, color: C.soft, fontSize: 24 }}>{subtitle}</p>
  </div>
);

const Legend = () => (
  <div style={{ display: 'flex', gap: 18, alignItems: 'center', fontSize: 19, color: C.soft }}>
    {[
      ['No-Q / q=10', C.gray],
      ['Aligned-Q', C.teal],
      ['Shuffled-Q', C.amber],
      ['Full-Q', C.red],
    ].map(([label, color]) => (
      <div key={label} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <span style={{ width: 14, height: 14, borderRadius: 4, background: color }} />
        <span>{label}</span>
      </div>
    ))}
    <span style={{ marginLeft: 'auto', color: C.muted }}>q=10 是 null token，不是品質等級 10</span>
  </div>
);

const ScoreRow = ({ exp }: { exp: (typeof experiments)[number] }) => {
  const min = 0.13;
  const max = 0.205;
  const width = Math.max(0, ((exp.clap - min) / (max - min)) * 100);
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '375px 1fr 100px 96px',
        gap: 16,
        alignItems: 'center',
        minHeight: 57,
        padding: '7px 14px',
        borderBottom: `1px solid ${C.line}`,
      }}
    >
      <div style={{ fontSize: 22, lineHeight: 1.12, fontWeight: 760 }}>{exp.name}</div>
      <div style={{ height: 28, borderRadius: 6, background: 'rgba(255,255,255,.055)', overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${width}%`,
            minWidth: 12,
            borderRadius: 6,
            background: `linear-gradient(90deg, ${exp.color}88, ${exp.color})`,
          }}
        />
      </div>
      <div style={{ fontFamily: mono, fontSize: 24, fontWeight: 900, color: exp.color }}>
        {exp.clap.toFixed(4)}
      </div>
      <div style={{ fontFamily: mono, fontSize: 21, color: C.soft }}>{exp.ce}</div>
    </div>
  );
};

const ScoreComparison: Page = () => {
  const historical = experiments.filter((e) => e.group === '歷史實驗');
  const clean = experiments.filter((e) => e.group === '本週 clean experiments');
  return (
    <Shell page="01 / 02">
      <div style={{ position: 'absolute', left: 66, right: 66, top: 48, bottom: 62, zIndex: 1 }}>
        <Header title="Q-conditioning 實驗總覽" subtitle="使用指定實驗名稱；橫條視覺放大 0.13–0.205 的 CLAP 差異" />
        <Legend />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 390px', gap: 24, marginTop: 20 }}>
          <div
            style={{
              background: 'rgba(18,26,45,.92)',
              border: `1px solid ${C.line}`,
              borderRadius: 16,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '375px 1fr 100px 96px',
                gap: 16,
                padding: '12px 14px',
                color: C.muted,
                fontFamily: mono,
                fontSize: 17,
                borderBottom: `1px solid ${C.line}`,
              }}
            >
              <span>實驗</span><span>CLAP BAR</span><span>CLAP</span><span>CE</span>
            </div>
            <div style={{ padding: '8px 14px', color: C.blue, fontWeight: 900, fontSize: 20 }}>歷史實驗</div>
            {historical.map((exp) => <ScoreRow key={exp.name} exp={exp} />)}
            <div style={{ padding: '10px 14px 8px', color: C.green, fontWeight: 900, fontSize: 20 }}>本週 clean experiments</div>
            {clean.map((exp) => <ScoreRow key={exp.name} exp={exp} />)}
          </div>
          <div style={{ display: 'grid', gap: 16, alignContent: 'start' }}>
            {[
              ['歷史 half-Q 最高', '0.1975', C.teal],
              ['P7 Full-Q', '−0.0227', C.red],
              ['P8 clean Full-Q', '−0.0204', C.red],
              ['S2：Shuffled > Aligned', '+0.0165', C.amber],
              ['100K FT：Shuffled > Aligned', '+0.0025', C.amber],
            ].map(([label, value, color]) => (
              <div
                key={label}
                style={{
                  minHeight: 116,
                  padding: '20px 22px',
                  borderRadius: 14,
                  border: `1px solid ${color}66`,
                  borderLeft: `6px solid ${color}`,
                  background: 'rgba(23,34,56,.9)',
                }}
              >
                <div style={{ color: C.soft, fontSize: 21 }}>{label}</div>
                <div style={{ marginTop: 8, color, fontFamily: mono, fontSize: 36, fontWeight: 900 }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Shell>
  );
};

const metricScale = (value: number, min: number, max: number) => {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const hue = 5 + t * 125;
  return {
    text: `hsl(${hue} 92% 67%)`,
  };
};

const clapValues = experiments.map((e) => e.clap);
const ceValues = experiments.map((e) => e.ce);
const clapRange = [Math.min(...clapValues), Math.max(...clapValues)] as const;
const ceRange = [Math.min(...ceValues), Math.max(...ceValues)] as const;

const DetailRow = ({ exp }: { exp: (typeof experiments)[number] }) => {
  const clapColor = metricScale(exp.clap, ...clapRange);
  const ceColor = metricScale(exp.ce, ...ceRange);
  const cells = [
    { content: <span style={{ fontWeight: 820, color: C.text }}>{exp.name}</span> },
    { content: exp.s1 },
    { content: exp.s2 },
    { content: exp.steps },
    { content: exp.eval },
    {
      content: <span style={{ color: clapColor.text, fontFamily: mono, fontWeight: 900 }}>{exp.clap.toFixed(4)}</span>,
    },
    {
      content: <span style={{ color: ceColor.text, fontFamily: mono, fontWeight: 900 }}>{exp.ce}</span>,
    },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '375px 275px 430px 240px 230px 125px 105px',
        minHeight: 61,
        borderBottom: `1px solid ${C.line}`,
        alignItems: 'stretch',
      }}
    >
      {cells.map((cell, index) => (
        <div
          key={index}
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '6px 10px',
            borderRight: index < 6 ? `1px solid ${C.line}` : undefined,
            color: index === 0 ? C.text : C.soft,
            fontSize: index === 0 ? 18.5 : 17,
            lineHeight: 1.18,
          }}
        >
          {cell.content}
        </div>
      ))}
    </div>
  );
};

const DetailTable: Page = () => {
  const historical = experiments.filter((e) => e.group === '歷史實驗');
  const clean = experiments.filter((e) => e.group === '本週 clean experiments');
  return (
    <Shell page="02 / 02">
      <div style={{ position: 'absolute', left: 48, right: 48, top: 40, bottom: 58, zIndex: 1 }}>
        <Header title="S1 / S2 訓練路徑與評估設定" subtitle="Aligned-Q 與來源 row 對齊；Shuffled-Q 只打亂 Q label · 數值色階：低紅 → 黃 → 高綠" />
        <div
          style={{
            background: 'rgba(18,26,45,.94)',
            border: `1px solid ${C.line}`,
            borderRadius: 14,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '375px 275px 430px 240px 230px 125px 105px',
              minHeight: 42,
              background: C.panel2,
              color: C.muted,
              fontFamily: mono,
              fontSize: 16,
              fontWeight: 800,
            }}
          >
            {['實驗', 'S1', 'S2 / fine-tune', '訓練長度', 'Eval', 'CLAP', 'CE'].map((label) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', padding: '6px 10px', borderRight: `1px solid ${C.line}` }}>
                {label}
              </div>
            ))}
          </div>
          <div style={{ padding: '5px 10px', color: C.blue, fontWeight: 900, fontSize: 17, background: 'rgba(96,165,250,.07)' }}>
            歷史實驗
          </div>
          {historical.map((exp) => <DetailRow key={exp.name} exp={exp} />)}
          <div style={{ padding: '5px 10px', color: C.green, fontWeight: 900, fontSize: 17, background: 'rgba(74,222,128,.07)' }}>
            本週 clean experiments
          </div>
          {clean.map((exp) => <DetailRow key={exp.name} exp={exp} />)}
        </div>
        <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', color: C.muted, fontSize: 17 }}>
          <span>Fine-tune 兩組皆為 100K：600K → 700K</span>
          <span>初始化：q embedding rows 0–9 複製自 null row q=10</span>
        </div>
      </div>
    </Shell>
  );
};

export const meta: SlideMeta = { title: 'PromptCC Q-conditioning experiments' };
export default [ScoreComparison, DetailTable] satisfies Page[];
