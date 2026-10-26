import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';
import { useSlidePageNumber } from '@open-slide/core';
import type { CSSProperties, ReactNode } from 'react';

export const meta: SlideMeta = {
  title: '評分模型健檢：寬標準成績單',
  createdAt: '2026-10-25T05:11:37.862Z',
};

export const design: DesignSystem = {
  palette: { bg: '#fbfaf7', text: '#1c1f24', accent: '#2563eb' },
  fonts: {
    display: '"Noto Sans CJK TC", "Noto Sans TC", system-ui, sans-serif',
    body: '"Noto Sans CJK TC", "Noto Sans TC", system-ui, sans-serif',
  },
  typeScale: { hero: 112, body: 32 },
  radius: 20,
};

const C = {
  panel: '#ffffff',
  line: '#e5e7eb',
  muted: '#6b7280',
  soft: '#374151',
  green: '#16a34a',
  greenBg: '#dcfce7',
  amber: '#d97706',
  amberBg: '#fef3c7',
  red: '#dc2626',
  redBg: '#fee2e2',
  blue: '#2563eb',
  blueBg: '#dbeafe',
};
const mono = '"JetBrains Mono", "DejaVu Sans Mono", ui-monospace, monospace';

const tone = (v: number) => (v >= 70 ? C.green : v >= 40 ? C.amber : C.red);
const toneBg = (v: number) => (v >= 70 ? C.greenBg : v >= 40 ? C.amberBg : C.redBg);

// ── shared chrome ──────────────────────────────────────────

const Footer = () => {
  const { current, total } = useSlidePageNumber();
  return (
    <div
      style={{
        position: 'absolute',
        left: 120,
        right: 120,
        bottom: 40,
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 22,
        color: C.muted,
      }}
    >
      <span>評分模型健檢 · 寬標準 · 2026-10-25</span>
      <span style={{ fontFamily: mono }}>
        {current} / {total}
      </span>
    </div>
  );
};

const Shell = ({ children }: { children: ReactNode }) => (
  <div
    style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      background: 'var(--osd-bg)',
      color: 'var(--osd-text)',
      fontFamily: 'var(--osd-font-body)',
      padding: '84px 120px 110px',
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
    }}
  >
    {children}
    <Footer />
  </div>
);

const Title = ({ step, children, sub }: { step?: string; children: ReactNode; sub?: ReactNode }) => (
  <div style={{ marginBottom: 40 }}>
    {step && (
      <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: 2, marginBottom: 12 }}>
        {step}
      </div>
    )}
    <div style={{ fontSize: 64, fontWeight: 900, lineHeight: 1.15 }}>{children}</div>
    {sub && <div style={{ fontSize: 30, color: C.muted, marginTop: 14 }}>{sub}</div>}
  </div>
);

const Pill = ({ color, bg, children }: { color: string; bg: string; children: ReactNode }) => (
  <span
    style={{
      display: 'inline-block',
      padding: '6px 18px',
      borderRadius: 999,
      background: bg,
      color,
      fontSize: 26,
      fontWeight: 800,
    }}
  >
    {children}
  </span>
);

// ── 1. Cover ───────────────────────────────────────────────

const Cover: Page = () => (
  <Shell>
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
      <div style={{ fontSize: 30, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: 3 }}>評分模型健檢</div>
      <div style={{ fontSize: 112, fontWeight: 900, lineHeight: 1.1, marginTop: 24 }}>
        該差的判差
        <br />
        該平的判平
      </div>
      <div style={{ fontSize: 34, color: C.muted, marginTop: 44 }}>12 個指標 × 183 組介入 × 7,043 個檢查點</div>
    </div>
  </Shell>
);

// ── 2. Goal ────────────────────────────────────────────────

const GoalCard = ({ icon, color, bg, head, line1, line2, ex }: { icon: string; color: string; bg: string; head: string; line1: string; line2: string; ex: string }) => (
  <div style={{ flex: 1, background: C.panel, borderRadius: 24, border: `2px solid ${C.line}`, padding: '48px 52px' }}>
    <div style={{ width: 96, height: 96, borderRadius: 48, background: bg, color, fontSize: 56, fontWeight: 900, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {icon}
    </div>
    <div style={{ fontSize: 52, fontWeight: 900, marginTop: 32, color }}>{head}</div>
    <div style={{ fontSize: 34, marginTop: 24, lineHeight: 1.5 }}>
      {line1}
      <br />
      {line2}
    </div>
    <div style={{ fontSize: 28, color: C.muted, marginTop: 28 }}>{ex}</div>
  </div>
);

const Goal: Page = () => (
  <Shell>
    <Title step="大目標">好的評分模型要做到兩件事</Title>
    <div style={{ display: 'flex', gap: 48, flex: 1, alignItems: 'stretch' }}>
      <GoalCard icon="↓" color={C.red} bg={C.redBg} head="分得出好壞" line1="音樂被弄壞" line2="→ 分數要下降" ex="例：加噪音、錯音、鼓對不上拍" />
      <GoalCard icon="=" color={C.blue} bg={C.blueBg} head="說得出平手" line1="不重要的差異" line2="→ 分數要不動" ex="例：小聲 1 dB、反相、mp3、移調" />
    </div>
  </Shell>
);

// ── 3. How we judge ────────────────────────────────────────

const Ruler: Page = () => (
  <Shell>
    <Title step="怎麼判" sub="S＝各指標自己的尺，約等於 5 個生成系統之間的分數差距">把分數變化換成「幾個 S」</Title>
    <svg width="1680" height="380" viewBox="0 0 1680 380" style={{ marginTop: 20 }}>
      <rect x="140" y="120" width="560" height="90" rx="14" fill={C.redBg} />
      <rect x="700" y="120" width="280" height="90" fill={C.blueBg} />
      <rect x="980" y="120" width="560" height="90" rx="14" fill="#f3f4f6" />
      <line x1="140" y1="165" x2="1540" y2="165" stroke={C.soft} strokeWidth="3" />
      <line x1="700" y1="100" x2="700" y2="230" stroke={C.soft} strokeWidth="3" strokeDasharray="8 6" />
      <line x1="980" y1="100" x2="980" y2="230" stroke={C.soft} strokeWidth="3" strokeDasharray="8 6" />
      <line x1="840" y1="110" x2="840" y2="220" stroke={C.soft} strokeWidth="2" />
      <text x="700" y="270" textAnchor="middle" fontSize="30" fontFamily={mono} fill={C.soft}>−0.1 S</text>
      <text x="840" y="270" textAnchor="middle" fontSize="30" fontFamily={mono} fill={C.soft}>0</text>
      <text x="980" y="270" textAnchor="middle" fontSize="30" fontFamily={mono} fill={C.soft}>+0.1 S</text>
      <text x="420" y="178" textAnchor="middle" fontSize="36" fontWeight="900" fill={C.red}>判差</text>
      <text x="840" y="70" textAnchor="middle" fontSize="36" fontWeight="900" fill={C.blue}>平手</text>
      <text x="1260" y="178" textAnchor="middle" fontSize="36" fontWeight="900" fill={C.muted}>反而加分</text>
      <text x="140" y="335" fontSize="30" fill={C.soft}>該變差 → 要落在紅區</text>
      <text x="1540" y="335" textAnchor="end" fontSize="30" fill={C.soft}>該平手 → 要落在藍區</text>
    </svg>
    <div style={{ fontSize: 28, color: C.muted, marginTop: 'auto' }}>寬標準＝看整組平均，比逐對 80% 寬鬆；這裡沒過的，嚴格標準下幾乎也不會過</div>
  </Shell>
);

// ── 4. Overall ─────────────────────────────────────────────

const Donut = () => {
  const r = 190;
  const circ = 2 * Math.PI * r;
  const fail = 0.343;
  return (
    <svg width="520" height="520" viewBox="0 0 520 520">
      <circle cx="260" cy="260" r={r} fill="none" stroke={C.greenBg} strokeWidth="70" />
      <circle
        cx="260"
        cy="260"
        r={r}
        fill="none"
        stroke={C.red}
        strokeWidth="70"
        strokeDasharray={`${circ * fail} ${circ}`}
        transform="rotate(-90 260 260)"
      />
      <text x="260" y="270" textAnchor="middle" fontSize="110" fontWeight="900" fill={C.red}>34%</text>
      <text x="260" y="330" textAnchor="middle" fontSize="30" fill={C.soft}>沒過</text>
    </svg>
  );
};

const Stat = ({ big, color, text }: { big: string; color: string; text: string }) => (
  <div style={{ display: 'flex', alignItems: 'baseline', gap: 28 }}>
    <div style={{ fontSize: 88, fontWeight: 900, color, fontFamily: mono, minWidth: 260, textAlign: 'right' }}>{big}</div>
    <div style={{ fontSize: 34 }}>{text}</div>
  </div>
);

const Overall: Page = () => (
  <Shell>
    <Title step="總成績">三分之一的檢查點沒過</Title>
    <div style={{ display: 'flex', alignItems: 'center', gap: 90, flex: 1 }}>
      <Donut />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 44 }}>
        <Stat big="2,417" color={C.red} text="個檢查點沒過（共 7,043）" />
        <Stat big="0" color={C.red} text="個指標全部過關" />
        <Stat big="16%" color={C.green} text="最好的一組（MERT／MuQ）失敗率" />
      </div>
    </div>
  </Shell>
);

// ── 5. Scatter ─────────────────────────────────────────────

const px = (v: number) => 140 + (v / 100) * 900;
const py = (v: number) => 640 - ((v - 30) / 70) * 600;

const Dot = ({ name, x, y, dx = 18, dy = 10, anchor = 'start', color }: { name: string; x: number; y: number; dx?: number; dy?: number; anchor?: 'start' | 'end' | 'middle'; color: string }) => (
  <g>
    <circle cx={px(x)} cy={py(y)} r="14" fill={color} stroke="#fff" strokeWidth="3" />
    <text x={px(x) + dx} y={py(y) + dy} textAnchor={anchor} fontSize="28" fontWeight="800" fill={C.text}>
      {name}
    </text>
  </g>
);

const Scatter: Page = () => (
  <Shell>
    <Title step="一張圖看全部">沒有指標落在右上角</Title>
    <div style={{ display: 'flex', gap: 40, flex: 1 }}>
      <svg width="1080" height="720" viewBox="0 0 1080 720">
        <rect x={px(90)} y={py(100)} width={px(100) - px(90)} height={py(90) - py(100)} fill={C.greenBg} />
        <text x={px(95)} y={py(100) + 36} textAnchor="middle" fontSize="26" fontWeight="800" fill={C.green}>理想</text>
        <line x1="140" y1="640" x2="1040" y2="640" stroke={C.soft} strokeWidth="2" />
        <line x1="140" y1="40" x2="140" y2="640" stroke={C.soft} strokeWidth="2" />
        <text x="140" y="680" textAnchor="middle" fontSize="24" fill={C.muted} fontFamily={mono}>0</text>
        <text x={px(50)} y="680" textAnchor="middle" fontSize="24" fill={C.muted} fontFamily={mono}>50</text>
        <text x={px(100)} y="680" textAnchor="middle" fontSize="24" fill={C.muted} fontFamily={mono}>100%</text>
        <text x="124" y={py(100) + 8} textAnchor="end" fontSize="24" fill={C.muted} fontFamily={mono}>100</text>
        <text x="124" y={py(65) + 8} textAnchor="end" fontSize="24" fill={C.muted} fontFamily={mono}>65</text>
        <text x="124" y={py(30) + 8} textAnchor="end" fontSize="24" fill={C.muted} fontFamily={mono}>30</text>
        <text x={px(50)} y="716" textAnchor="middle" fontSize="28" fontWeight="700" fill={C.red}>該變差 → 判差的比例</text>
        <text x="30" y="340" textAnchor="middle" fontSize="28" fontWeight="700" fill={C.blue} transform="rotate(-90 30 340)">該平手 → 判平的比例</text>

        <Dot name="MERT" x={85} y={86} color={C.green} />
        <Dot name="MuQ" x={86} y={74} color={C.green} />
        <Dot name="CE" x={71} y={81} dx={-20} anchor="end" color={C.amber} />
        <Dot name="CU" x={67} y={89} dx={-20} anchor="end" color={C.amber} />
        <Dot name="PQ" x={70} y={72} color={C.amber} />
        <Dot name="CLAP" x={57} y={72} dx={-20} anchor="end" color={C.amber} />
        <Dot name="pulse" x={63} y={97} color={C.blue} />
        <Dot name="ibi_cv" x={43} y={94} dy={-18} color={C.blue} />
        <Dot name="key" x={37} y={96} dx={-20} anchor="end" color={C.blue} />
        <Dot name="SongEval" x={36} y={81} dx={-20} anchor="end" color={C.red} />
        <Dot name="chroma" x={15} y={77} dy={46} anchor="middle" dx={0} color={C.red} />
        <Dot name="MEva" x={64} y={40} color={C.red} />
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 36, fontSize: 32, lineHeight: 1.45 }}>
        <div><b style={{ color: C.green }}>MERT／MuQ</b><br />最接近理想</div>
        <div><b style={{ color: C.blue }}>MIR 特徵</b><br />很會判平手<br />但抓不到壞</div>
        <div><b style={{ color: C.red }}>MEva</b><br />該平手時亂動</div>
      </div>
    </div>
  </Shell>
);

// ── 6. Heatmap ─────────────────────────────────────────────

const COLW = 172;
const HCell = ({ v }: { v: number | null }) =>
  v === null ? (
    <div style={{ width: COLW, height: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#c4c8cf', fontSize: 28 }}>·</div>
  ) : (
    <div
      style={{
        width: COLW - 8,
        height: 46,
        margin: '2px 4px',
        borderRadius: 8,
        background: toneBg(v),
        color: tone(v),
        fontFamily: mono,
        fontWeight: 800,
        fontSize: 28,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {v}
    </div>
  );

const HRow = ({ name, v }: { name: string; v: (number | null)[] }) => (
  <div style={{ display: 'flex', alignItems: 'center' }}>
    <div style={{ width: 210, fontSize: 28, fontWeight: 800 }}>{name}</div>
    {v.slice(0, 5).map((x, i) => (
      <HCell key={i} v={x} />
    ))}
    <div style={{ width: 24 }} />
    {v.slice(5).map((x, i) => (
      <HCell key={i} v={x} />
    ))}
  </div>
);

const HHead = ({ t }: { t: string }) => (
  <div style={{ width: COLW, textAlign: 'center', fontSize: 26, fontWeight: 700, color: C.soft }}>{t}</div>
);

const Heatmap: Page = () => (
  <Shell>
    <Title step="拆開看">每個指標在哪類失手（通過率 %）</Title>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', marginBottom: 6 }}>
        <div style={{ width: 210 }} />
        <div style={{ width: COLW * 5, textAlign: 'center', fontSize: 28, fontWeight: 900, color: C.red, borderBottom: `3px solid ${C.red}` }}>該變差</div>
        <div style={{ width: 24 }} />
        <div style={{ width: COLW * 2, textAlign: 'center', fontSize: 28, fontWeight: 900, color: C.blue, borderBottom: `3px solid ${C.blue}` }}>該平手</div>
      </div>
      <div style={{ display: 'flex', marginBottom: 8 }}>
        <div style={{ width: 210 }} />
        <HHead t="訊號劣化" />
        <HHead t="音符錯誤" />
        <HHead t="分軌編輯" />
        <HHead t="大幅變速" />
        <HHead t="業餘／底線" />
        <div style={{ width: 24 }} />
        <HHead t="訊號小差" />
        <HHead t="音樂小差" />
      </div>
      <HRow name="PQ" v={[90, 1, 69, 62, 63, 75, 65]} />
      <HRow name="CE" v={[83, 15, 62, 69, 95, 78, 88]} />
      <HRow name="CU" v={[89, 4, 69, 54, 74, 86, 96]} />
      <HRow name="CLAP" v={[68, 10, 8, 56, 89, 71, 73]} />
      <HRow name="MEva" v={[54, 81, 46, 72, 84, 46, 26]} />
      <HRow name="SongEval" v={[49, 14, 28, null, null, 78, 87]} />
      <HRow name="MERT" v={[87, 82, 62, null, null, 95, 48]} />
      <HRow name="MuQ" v={[89, 91, 62, null, null, 76, 73]} />
      <HRow name="pulse" v={[null, 62, 54, 65, 38, null, 97]} />
      <HRow name="ibi_cv" v={[null, 57, 31, 38, 38, null, 94]} />
      <HRow name="key" v={[null, 32, 31, null, 88, null, 96]} />
      <HRow name="chroma" v={[null, 13, 15, null, 25, null, 76]} />
    </div>
  </Shell>
);

// ── 7. Should drop but didn't ──────────────────────────────

const Bar = ({ name, v, max = 100 }: { name: string; v: number; max?: number }) => (
  <div style={{ display: 'flex', alignItems: 'center', height: 52 }}>
    <div style={{ width: 200, fontSize: 30, fontWeight: 800 }}>{name}</div>
    <div style={{ width: 900, height: 34, background: '#f1f2f4', borderRadius: 8, position: 'relative' }}>
      <div style={{ width: `${Math.max((v / max) * 100, 0.8)}%`, height: '100%', background: tone(v), borderRadius: 8 }} />
    </div>
    <div style={{ width: 120, textAlign: 'right', fontSize: 32, fontWeight: 900, fontFamily: mono, color: tone(v) }}>{v}%</div>
  </div>
);

const NoteBlind: Page = () => (
  <Shell>
    <Title step="該變差沒變差" sub="錯音、亂音高、節奏抖動、打亂順序：判差的比例">AES 和 CLAP 聽不到錯音</Title>
    <div style={{ display: 'flex', gap: 60, flex: 1 }}>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <Bar name="PQ" v={1} />
        <Bar name="CU" v={4} />
        <Bar name="CLAP" v={10} />
        <Bar name="chroma" v={13} />
        <Bar name="SongEval" v={14} />
        <Bar name="CE" v={15} />
        <Bar name="key" v={32} />
        <Bar name="ibi_cv" v={57} />
        <Bar name="pulse" v={62} />
        <Bar name="MEva" v={81} />
        <Bar name="MERT" v={82} />
        <Bar name="MuQ" v={91} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 30, fontSize: 32, lineHeight: 1.45 }}>
        <div>錯音 100%<br /><b style={{ color: C.red }}>PQ 幾乎不變</b></div>
        <div>分軌拿掉一軌<br /><b style={{ color: C.red }}>CLAP 只抓到 8%</b></div>
      </div>
    </div>
  </Shell>
);

// ── 8. Should tie but didn't ───────────────────────────────

const TieBar = ({ who, what, v }: { who: string; what: string; v: number }) => (
  <div style={{ display: 'flex', alignItems: 'center', height: 70, borderBottom: `1px solid ${C.line}` }}>
    <div style={{ width: 300, fontSize: 32, fontWeight: 900 }}>{who}</div>
    <div style={{ width: 460, fontSize: 32 }}>{what}</div>
    <div style={{ width: 640, height: 34, background: '#f1f2f4', borderRadius: 8 }}>
      <div style={{ width: `${Math.max(v, 1)}%`, height: '100%', background: tone(v), borderRadius: 8 }} />
    </div>
    <div style={{ width: 140, textAlign: 'right', fontSize: 34, fontWeight: 900, fontFamily: mono, color: tone(v) }}>{v}%</div>
  </div>
);

const TieFail: Page = () => (
  <Shell>
    <Title step="該平手沒平手" sub="聽起來一樣或不重要的差異：判平手的比例（應該接近 100%）">對無關差異過敏</Title>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <TieBar who="SongEval" what="mp3 壓縮" v={0} />
      <TieBar who="MuQ" what="mp3 壓縮" v={0} />
      <TieBar who="MEva、MERT" what="波形反相" v={0} />
      <TieBar who="PQ、CE、CLAP" what="換一套鋼琴音色" v={0} />
      <TieBar who="PQ" what="分離後再相加" v={0} />
      <TieBar who="MERT" what="移調 ±1、微變速" v={14} />
      <TieBar who="MEva" what="小幅音量 ±dB" v={33} />
    </div>
    <div style={{ fontSize: 28, color: C.muted, marginTop: 'auto' }}>反相：人耳完全聽不出差別，仍被判成大差</div>
  </Shell>
);

// ── 9. Praise ──────────────────────────────────────────────

const Praise = ({ who, big, what, why }: { who: string; big: string; what: string; why: string }) => (
  <div style={{ background: C.panel, border: `2px solid ${C.greenBg}`, borderRadius: 24, padding: '34px 40px', display: 'flex', flexDirection: 'column' }}>
    <div style={{ fontSize: 34, fontWeight: 900 }}>{who}</div>
    <div style={{ fontSize: 84, fontWeight: 900, color: C.green, fontFamily: mono, lineHeight: 1.15, marginTop: 8 }}>{big}</div>
    <div style={{ fontSize: 30, marginTop: 6 }}>{what}</div>
    <div style={{ fontSize: 26, color: C.muted, marginTop: 14 }}>{why}</div>
  </div>
);

const Kudos: Page = () => (
  <Shell>
    <Title step="意外的好表現">沒預期能做到，卻做到了</Title>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, flex: 1 }}>
      <Praise who="MERT／MuQ" big="82–91%" what="抓到錯音、亂音高、打亂" why="通用音樂嵌入，沒人教它音準" />
      <Praise who="pulse_clarity" big="100%" what="抓到鼓延遲 100 ms、節奏抖動" why="節奏抖動 AES 全部 0%" />
      <Praise who="MEva" big="81%" what="音符錯誤判差" why="同為品質分數，PQ 只有 1%" />
      <Praise who="MERT" big="100%" what="音量 ±dB 判平手" why="AES 對音量一直很敏感" />
    </div>
  </Shell>
);

// ── 10. Report card ────────────────────────────────────────

const Grade = ({ name, good, bad }: { name: string; good: string; bad: string }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 1fr', alignItems: 'center', height: 76, borderBottom: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 32, fontWeight: 900 }}>{name}</div>
    <div style={{ fontSize: 30, color: C.green }}>✓ {good}</div>
    <div style={{ fontSize: 30, color: C.red }}>✗ {bad}</div>
  </div>
);

const ReportCard: Page = () => (
  <Shell>
    <Title step="各指標成績單">用在哪裡可以信</Title>
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 1fr', fontSize: 26, color: C.muted, paddingBottom: 10 }}>
      <span>指標</span>
      <span>可以信</span>
      <span>別拿來看</span>
    </div>
    <Grade name="PQ／CE／CU" good="噪音、削波、低通" bad="錯音、節奏、換音色" />
    <Grade name="CLAP" good="業餘演奏、底線" bad="分軌、音符錯誤" />
    <Grade name="MEva" good="音符與節奏錯誤" bad="音量、反相（會亂扣）" />
    <Grade name="SongEval" good="判平手大多穩" bad="音符錯誤、mp3" />
    <Grade name="MERT／MuQ" good="兩邊都最穩" bad="反相、移調、mp3" />
    <Grade name="pulse／ibi_cv" good="鼓和節奏錯誤" bad="音高錯誤" />
    <Grade name="key／chroma" good="判平手很穩" bad="幾乎什麼都抓不到" />
  </Shell>
);

// ── 11–13. Screening r2 (human ratings) ────────────────────

const Domain = ({ head, sets, what }: { head: string; sets: string; what: string }) => (
  <div style={{ flex: 1, background: C.panel, borderRadius: 24, border: `2px solid ${C.line}`, padding: '40px 44px' }}>
    <div style={{ fontSize: 44, fontWeight: 900, color: 'var(--osd-accent)' }}>{head}</div>
    <div style={{ fontSize: 34, fontWeight: 800, marginTop: 20 }}>{sets}</div>
    <div style={{ fontSize: 28, color: C.muted, marginTop: 14, lineHeight: 1.45 }}>{what}</div>
  </div>
);

const ScreenIntro: Page = () => (
  <Shell>
    <Title step="海選 round 2" sub="前面的「該差／該平」是我們定的；這裡改問：分數排序跟人類評分一不一致">換人類評分來考</Title>
    <div style={{ display: 'flex', gap: 36 }}>
      <Domain head="訊號劣化" sets="ODAQ" what="專家 MUSHRA 評壓縮、分離瑕疵" />
      <Domain head="真實錄音" sets="MusicCaps 522 首" what="真實音樂的品質評分" />
      <Domain head="生成音樂" sets="AIME、Music Arena" what="聽者在生成系統之間選喜歡的" />
    </div>
    <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 18, fontSize: 32 }}>
      <div><b>過一格</b>：和人類評分的相關 ρ ≥ 0.50（信賴區間下界 &gt; 0.30）</div>
      <div><b>過關</b>：至少兩個<b style={{ color: C.red }}>沒被拿去訓練過</b>的資料集過，其中一個是生成音樂</div>
    </div>
  </Shell>
);

type RCell = [number, 'i' | 't' | 'u'] | null;
const RHO: [string, RCell[]][] = [
  ['AES PQ', [[0.55, 'i'], [0.83, 't'], [0.4, 'i'], [0.15, 'i']]],
  ['AES CE', [[0.72, 'i'], [0.63, 't'], [0.38, 'i'], [0.11, 'i']]],
  ['PAM', [[0.6, 'i'], [0.61, 'i'], [0.47, 'i'], [-0.06, 'i']]],
  ['MEva', [[-0.05, 'i'], [0.36, 'i'], [0.83, 't'], [0.25, 't']]],
  ['MuQ A1', [[0.18, 'i'], [0.47, 'i'], [0.54, 'i'], [0.22, 'i']]],
  ['MuQ A3a', [[0.26, 'i'], [0.45, 'i'], [0.64, 'i'], [0.22, 'i']]],
  ['CMI-RM', [[0.1, 'i'], [0.37, 'i'], [0.59, 'i'], [0.22, 'i']]],
  ['SingMOS', [[0.27, 'i'], [0.34, 'i'], [0.02, 'i'], [-0.06, 'i']]],
];
const BLUES = ['#eef4fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95'];
const blue = (v: number) => BLUES[Math.max(0, Math.min(BLUES.length - 1, Math.floor((Math.max(v, 0) / 0.9) * BLUES.length)))];
const RW = 300;

const RhoCell = ({ c }: { c: RCell }) => {
  if (!c) return <div style={{ width: RW - 12, height: 64, margin: '4px 6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#c4c8cf', fontSize: 30 }}>沒評</div>;
  const [v, s] = c;
  const pass = s === 'i' && v >= 0.5;
  const trained = s !== 'i';
  const bg = trained ? 'transparent' : pass ? C.green : blue(v);
  const dark = pass || (!trained && BLUES.indexOf(bg) >= 4);
  return (
    <div
      style={{
        width: RW - 12,
        height: 64,
        margin: '4px 6px',
        borderRadius: 10,
        background: bg,
        outline: trained ? '3px dashed #c4c8cf' : 'none',
        outlineOffset: -4,
        color: trained ? '#9aa0a8' : dark ? '#ffffff' : 'var(--osd-text)',
        fontFamily: mono,
        fontWeight: 800,
        fontSize: 32,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {v.toFixed(2)}
    </div>
  );
};

const Key = ({ sw, text }: { sw: CSSProperties; text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
    <span style={{ width: 34, height: 26, borderRadius: 6, display: 'inline-block', ...sw }} />
    <span>{text}</span>
  </div>
);

const ScreenHeat: Page = () => (
  <Shell>
    <Title step="海選結果">和人類評分的相關 ρ</Title>
    <div style={{ display: 'flex' }}>
      <div style={{ width: 220 }} />
      <div style={{ width: RW, textAlign: 'center', fontSize: 28, fontWeight: 900 }}>訊號劣化</div>
      <div style={{ width: RW, textAlign: 'center', fontSize: 28, fontWeight: 900 }}>真實錄音</div>
      <div style={{ width: RW * 2, textAlign: 'center', fontSize: 28, fontWeight: 900, borderBottom: `3px solid ${C.soft}` }}>生成音樂</div>
    </div>
    <div style={{ display: 'flex', marginBottom: 6, marginTop: 6 }}>
      <div style={{ width: 220 }} />
      {['ODAQ', 'MC522', 'AIME', 'Arena'].map((t) => (
        <div key={t} style={{ width: RW, textAlign: 'center', fontSize: 26, color: C.soft }}>{t}</div>
      ))}
    </div>
    {RHO.map(([name, cells]) => (
      <div key={name} style={{ display: 'flex', alignItems: 'center' }}>
        <div style={{ width: 220, fontSize: 30, fontWeight: 800 }}>{name}</div>
        {cells.map((c, i) => (
          <RhoCell key={i} c={c} />
        ))}
      </div>
    ))}
    <div style={{ display: 'flex', gap: 44, fontSize: 26, color: C.soft, marginTop: 'auto' }}>
      <Key sw={{ background: '#6da7ec' }} text="沒過（越深越高）" />
      <Key sw={{ background: C.green }} text="過 0.50" />
      <Key sw={{ outline: '3px dashed #c4c8cf', outlineOffset: -3 }} text="訓練過，不算" />
    </div>
  </Shell>
);

const Finding = ({ tag, color, bg, head, text }: { tag: string; color: string; bg: string; head: string; text: string }) => (
  <div style={{ display: 'flex', gap: 36, alignItems: 'flex-start' }}>
    <div style={{ width: 200, flexShrink: 0, paddingTop: 6 }}>
      <Pill color={color} bg={bg}>{tag}</Pill>
    </div>
    <div>
      <div style={{ fontSize: 40, fontWeight: 900 }}>{head}</div>
      <div style={{ fontSize: 30, color: C.soft, marginTop: 8 }}>{text}</div>
    </div>
  </div>
);

const ScreenVerdict: Page = () => (
  <Shell>
    <Title step="海選結論">沒有指標過關，卡在生成音樂</Title>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 50, marginTop: 10 }}>
      <Finding tag="AES／PAM" color={C.amber} bg={C.amberBg} head="訊號劣化過，到生成音樂就掉下來" text="AES PQ：ODAQ 0.55 → AIME 0.40 → Arena 0.15；PAM 在 Arena −0.06" />
      <Finding tag="查核後" color={C.muted} bg="#f3f4f6" head="AIME 過的，到 Arena 都掉到 0.22" text="MuQ 0.54–0.64、CMI-RM 0.59 沒訓練過 AIME，算數；但 Arena 只剩 0.22。MEva 兩格都訓練過" />
      <Finding tag="看勝負" color={C.green} bg={C.greenBg} head="改看「兩首誰比較好」，AES PQ 過了" text="勝負一致率 AIME 0.61、Arena 0.58（門檻 0.55）；Arena 標籤只有勝／負，ρ 本來就高不了" />
    </div>
    <div style={{ fontSize: 28, color: C.muted, marginTop: 'auto' }}>預登錄的規則照原樣判；改看勝負屬於事後探索，要正式採用得另寫 round 3 預登錄</div>
  </Shell>
);

// ── 14. Caveats ────────────────────────────────────────────

const Caveat = ({ n, head, text }: { n: string; head: string; text: string }) => (
  <div style={{ display: 'flex', gap: 36, alignItems: 'flex-start' }}>
    <div style={{ width: 76, height: 76, borderRadius: 38, background: C.amberBg, color: C.amber, fontSize: 40, fontWeight: 900, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
      {n}
    </div>
    <div>
      <div style={{ fontSize: 40, fontWeight: 900 }}>{head}</div>
      <div style={{ fontSize: 30, color: C.soft, marginTop: 8 }}>{text}</div>
    </div>
  </div>
);

const Caveats: Page = () => (
  <Shell>
    <Title step="讀數字前">三個限制</Title>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 56, marginTop: 20 }}>
      <Caveat n="1" head="寬標準偏寬鬆" text="看整組平均；逐對標準下失敗只會更多" />
      <Caveat n="2" head="S 是各指標自己的尺" text="通過率可以比，Δ/S 數值不要跨指標比" />
      <Caveat n="3" head="沒有人類評分" text="介入的「該平」「該差」是我們定的；人類評分的海選還沒有指標過關" />
    </div>
  </Shell>
);

// ── 15. Takeaway ───────────────────────────────────────────

const Take = ({ color, bg, tag, text }: { color: string; bg: string; tag: string; text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 36 }}>
    <div style={{ width: 230, flexShrink: 0 }}>
      <Pill color={color} bg={bg}>{tag}</Pill>
    </div>
    <div style={{ fontSize: 40, fontWeight: 700 }}>{text}</div>
  </div>
);

const Takeaway: Page = () => (
  <Shell>
    <Title step="結論">一句話帶走</Title>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 52, marginTop: 20 }}>
      <Take color={C.red} bg={C.redBg} tag="沒有全能" text="12 個指標，沒有一個兩條要求全過" />
      <Take color={C.amber} bg={C.amberBg} tag="AES 的盲點" text="PQ 只看聲音乾不乾淨，聽不到彈錯" />
      <Take color={C.green} bg={C.greenBg} tag="最接近" text="MERT／MuQ，但要先修反相與移調" />
      <Take color={C.blue} bg={C.blueBg} tag="建議" text="AES＋MERT＋pulse 組合，再用人耳驗證" />
    </div>
  </Shell>
);

export default [Cover, Goal, Ruler, Overall, Scatter, Heatmap, NoteBlind, TieFail, Kudos, ReportCard, ScreenIntro, ScreenHeat, ScreenVerdict, Caveats, Takeaway] satisfies Page[];
