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
};

const font = design.fonts.body;
const mono = '"JetBrains Mono", "SF Mono", Menlo, monospace';

const Shell = ({ children, page, line }: { children: React.ReactNode; page: string; line?: string }) => (
  <div className="mean-slide" style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', background: C.bg, color: C.text, fontFamily: font }}>
    <style>{`* { box-sizing: border-box; } .mean-slide h1,.mean-slide h2,.mean-slide p { margin: 0; }`}</style>
    {children}
    <div style={{ position: 'absolute', left: 52, right: 52, bottom: 20, display: 'flex', justifyContent: 'space-between', color: C.muted, fontSize: 16, fontFamily: mono }}>
      <span>{line || 'MeanAudio · Phase 8'} · MusicCaps n=5521 · CLAP ↑</span>
      <span>{page}</span>
    </div>
  </div>
);

const Header = ({ kicker, title, subtitle }: { kicker: string; title: string; subtitle: string }) => (
  <div style={{ marginBottom: 18 }}>
    <div style={{ color: C.teal, fontFamily: mono, fontSize: 16, fontWeight: 800, letterSpacing: 1.2 }}>{kicker}</div>
    <h1 style={{ marginTop: 8, fontSize: 42, lineHeight: 1.1, fontWeight: 920 }}>{title}</h1>
    <p style={{ marginTop: 8, color: C.soft, fontSize: 20, lineHeight: 1.35 }}>{subtitle}</p>
  </div>
);

const Card = ({ children, color = C.line, style = {} as React.CSSProperties }: { children: React.ReactNode; color?: string; style?: React.CSSProperties }) => (
  <div style={{ background: 'rgba(18,26,45,.94)', border: `1px solid ${color}66`, borderRadius: 14, padding: 18, ...style }}>{children}</div>
);

// ── L1-A Full design ───────────────────────────────────────
const L1Design: Page = () => (
  <Shell page="01 / 07" line="主線 1 · Full scale">
    <div style={{ position: 'absolute', left: 52, right: 52, top: 40, bottom: 52 }}>
      <Header
        kicker="MAIN LINE 1 · FULL SCALE"
        title="NoQ / Stage1 / S2×K — 實驗設計"
        subtitle="固定 full 預算（S1 400k + S2 200k）· 比較「要不要 Q」以及「Q 只在 S2」時桶數 K"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        {[
          { t: 'NoQ full', c: C.gray, d: 'S1+S2 皆 null-Q（q=10）', e: 'Eval: NoQ · MF1 CFG0.5', r: 'CLAP 0.1735' },
          { t: 'Stage1 臂', c: C.blue, d: 'S1 路徑 / 對照系統（人耳 AB 中的 stage1）', e: '與 NoQ / K 並排比較', r: '見人耳頁' },
          { t: 'S2Q × K', c: C.teal, d: '同一 NoQ S1 ckpt → migrate S2', e: 'S2 only MeanSim-Q · K∈{2,3,5,10} bal', r: 'design: NoQ_S1_to_Q_S2_only' },
        ].map((x) => (
          <Card key={x.t} color={x.c} style={{ minHeight: 280 }}>
            <div style={{ color: x.c, fontWeight: 900, fontSize: 22 }}>{x.t}</div>
            <div style={{ marginTop: 14, color: C.soft, fontSize: 19, lineHeight: 1.4 }}>{x.d}</div>
            <div style={{ marginTop: 12, color: C.muted, fontSize: 17, lineHeight: 1.35 }}>{x.e}</div>
            <div style={{ marginTop: 20, fontFamily: mono, fontSize: 22, fontWeight: 900, color: x.c }}>{x.r}</div>
          </Card>
        ))}
      </div>
      <Card style={{ marginTop: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 1fr 1fr', gap: 10, fontFamily: mono, fontSize: 17 }}>
          <div style={{ color: C.muted }}>Pipeline</div>
          <div style={{ color: C.green }}>① Train NoQ full S1+S2</div>
          <div style={{ color: C.teal }}>② Fork S1 → S2Q for each K</div>
          <div style={{ color: C.amber }}>③ Eval q=9 (high_q9) + optional AB</div>
        </div>
        <p style={{ marginTop: 12, color: C.soft, fontSize: 18 }}>
          重點：K-ablation <b>不重訓 S1</b>，只改 S2 的 Q 分桶，分離「NoQ 音質底」與「後期加 Q」的貢獻。
        </p>
      </Card>
    </div>
  </Shell>
);

// ── L1-B Full results ──────────────────────────────────────
const fullS2q = [
  { k: 'NoQ', clap: 0.1735, ce: 6.62, note: 'baseline · both stages NoQ', color: C.gray },
  { k: 'K2 bal', clap: 0.1741, ce: 6.71, note: 'S2Q · high_q9', color: C.teal },
  { k: 'K3 bal', clap: 0.1775, ce: 6.74, note: 'S2Q · high_q9', color: C.teal },
  { k: 'K5 bal', clap: 0.1779, ce: 6.72, note: 'S2Q · best CLAP', color: C.green },
  { k: 'K10 bal', clap: 0.1754, ce: 6.72, note: 'S2Q · high_q9', color: C.teal },
];

const L1Results: Page = () => {
  const min = 0.172;
  const max = 0.179;
  return (
    <Shell page="02 / 07" line="主線 1 · Full scale">
      <div style={{ position: 'absolute', left: 52, right: 52, top: 40, bottom: 52 }}>
        <Header
          kicker="MAIN LINE 1 · RESULTS"
          title="Full scale · CLAP（S2Q K balanced）"
          subtitle="Aug 1–3 · 共享 Official NoQ S1 · S2 200k with MeanSim-Q · MusicCaps high_q9（NoQ 列為 NoQ eval）"
        />
        <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 0.75fr', gap: 18 }}>
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 110px 90px 1fr', gap: 10, color: C.muted, fontFamily: mono, fontSize: 15, marginBottom: 8 }}>
              <span>System</span><span>CLAP bar</span><span>CLAP</span><span>CE</span><span>Δ vs NoQ</span>
            </div>
            {fullS2q.map((r) => {
              const w = Math.max(4, ((r.clap - min) / (max - min)) * 100);
              const d = r.clap - 0.1735;
              return (
                <div key={r.k} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 110px 90px 1fr', gap: 10, alignItems: 'center', minHeight: 56, borderTop: `1px solid ${C.line}` }}>
                  <div style={{ fontWeight: 800, fontSize: 20 }}>{r.k}</div>
                  <div style={{ height: 24, borderRadius: 5, background: 'rgba(255,255,255,.05)' }}>
                    <div style={{ height: '100%', width: `${w}%`, borderRadius: 5, background: `linear-gradient(90deg, ${r.color}77, ${r.color})` }} />
                  </div>
                  <div style={{ fontFamily: mono, fontWeight: 900, fontSize: 22, color: r.color }}>{r.clap.toFixed(4)}</div>
                  <div style={{ fontFamily: mono, color: C.soft }}>{r.ce.toFixed(2)}</div>
                  <div style={{ fontFamily: mono, color: d >= 0 ? C.green : C.red, fontWeight: 800 }}>
                    {r.k === 'NoQ' ? '—' : `${d >= 0 ? '+' : ''}${d.toFixed(4)}`}
                    <span style={{ marginLeft: 8, color: C.muted, fontWeight: 500, fontSize: 15 }}>{r.note}</span>
                  </div>
                </div>
              );
            })}
          </Card>
          <div style={{ display: 'grid', gap: 12 }}>
            {[
              ['Best S2Q', 'K5 balanced', '0.1779', '+0.0044 vs NoQ'],
              ['Runner-up', 'K3 balanced', '0.1775', '+0.0040'],
              ['Takeaway', 'S2 加 Q 有增益', 'K=3–5 最佳', 'K=10 略回落'],
            ].map(([a, b, c, d]) => (
              <Card key={String(a)} color={C.teal}>
                <div style={{ color: C.muted, fontFamily: mono, fontSize: 14 }}>{a}</div>
                <div style={{ marginTop: 6, fontSize: 22, fontWeight: 900 }}>{b}</div>
                <div style={{ marginTop: 6, fontFamily: mono, fontSize: 28, fontWeight: 900, color: C.teal }}>{c}</div>
                <div style={{ marginTop: 4, color: C.soft, fontSize: 17 }}>{d}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </Shell>
  );
};

// ── L2-A Quarter K design ──────────────────────────────────
const L2Design: Page = () => (
  <Shell page="03 / 07" line="主線 2 · Quarter × K">
    <div style={{ position: 'absolute', left: 52, right: 52, top: 40, bottom: 52 }}>
      <Header
        kicker="MAIN LINE 2 · QUARTER SCALE × K"
        title="同一 1/4 預算下的 Q-schedule × K"
        subtitle="S1 100k + S2 50k · 隔離「何時加 Q」與「桶數 K / balanced vs fixed」"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <Card color={C.gray} style={{ minHeight: 320 }}>
          <div style={{ color: C.gray, fontWeight: 900, fontSize: 22 }}>A · NoQ quarter</div>
          <div style={{ marginTop: 16, fontSize: 19, color: C.soft, lineHeight: 1.4 }}>S1+S2 皆 NoQ</div>
          <div style={{ marginTop: 20, fontFamily: mono, fontSize: 36, fontWeight: 900 }}>0.1659</div>
          <div style={{ color: C.muted, marginTop: 8 }}>baseline for quarter K grid</div>
        </Card>
        <Card color={C.amber} style={{ minHeight: 320 }}>
          <div style={{ color: C.amber, fontWeight: 900, fontSize: 22 }}>B · Bucket full-Q</div>
          <div style={{ marginTop: 16, fontSize: 19, color: C.soft, lineHeight: 1.4 }}>
            S1+S2 <b>都</b> MeanSim-Q<br />K∈{2,3,5,10} × bal/fixed
          </div>
          <div style={{ marginTop: 20, fontFamily: mono, fontSize: 22, fontWeight: 900, color: C.amber }}>Eval high_q9</div>
          <div style={{ color: C.muted, marginTop: 8 }}>問：全程 Q 是否傷害生成？</div>
        </Card>
        <Card color={C.teal} style={{ minHeight: 320 }}>
          <div style={{ color: C.teal, fontWeight: 900, fontSize: 22 }}>C · S2Q-from-NoQ</div>
          <div style={{ marginTop: 16, fontSize: 19, color: C.soft, lineHeight: 1.4 }}>
            NoQ S1 → Q S2 only<br />K∈{2,3,5} × bal/fixed
          </div>
          <div style={{ marginTop: 20, fontFamily: mono, fontSize: 22, fontWeight: 900, color: C.teal }}>Eval high_q9</div>
          <div style={{ color: C.muted, marginTop: 8 }}>問：只在 S2 加 Q 能否贏 NoQ？</div>
        </Card>
      </div>
      <Card style={{ marginTop: 16 }}>
        <div style={{ fontSize: 20, fontWeight: 800, color: C.soft }}>對照邏輯</div>
        <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 18, color: C.soft, lineHeight: 1.4 }}>
          <div>B vs A → full-Q 兩階段是否劣於 NoQ</div>
          <div>C vs A → 後期加 Q 是否溫和增益（對齊 full 主線）</div>
          <div>bal vs fixed → 取樣策略</div>
          <div>K 掃描 → 桶過細是否 degenerate</div>
        </div>
      </Card>
    </div>
  </Shell>
);

// ── L2-B Quarter K results ─────────────────────────────────
const L2Results: Page = () => (
  <Shell page="04 / 07" line="主線 2 · Quarter × K">
    <div style={{ position: 'absolute', left: 48, right: 48, top: 36, bottom: 50 }}>
      <Header
        kicker="MAIN LINE 2 · RESULTS"
        title="Quarter · K 結果總表"
        subtitle="左：S2Q-from-NoQ（多數 ≥ NoQ）· 右：Bucket full-Q（全面 < NoQ）"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card color={C.teal}>
          <div style={{ color: C.teal, fontWeight: 900, fontSize: 20, marginBottom: 10 }}>S2Q-from-NoQ quarter · high_q9</div>
          <div style={{ display: 'grid', gridTemplateColumns: '70px 100px 90px 80px', gap: 6, color: C.muted, fontFamily: mono, fontSize: 14 }}>
            <span>K</span><span>Strategy</span><span>CLAP</span><span>ΔNoQ</span>
          </div>
          {[
            [3, 'balanced', 0.168, 0.0021],
            [3, 'fixed', 0.1669, 0.001],
            [2, 'balanced', 0.1667, 0.0008],
            [2, 'fixed', 0.1664, 0.0005],
            [5, 'fixed', 0.164, -0.0019],
            [5, 'balanced', 0.1639, -0.002],
          ].map((r) => (
            <div key={String(r[0]) + r[1]} style={{ display: 'grid', gridTemplateColumns: '70px 100px 90px 80px', gap: 6, padding: '8px 0', borderTop: `1px solid ${C.line}`, fontSize: 17, fontFamily: mono }}>
              <span style={{ fontWeight: 900 }}>K{r[0]}</span>
              <span>{r[1]}</span>
              <span style={{ fontWeight: 900, color: C.teal }}>{(r[2] as number).toFixed(4)}</span>
              <span style={{ color: (r[3] as number) >= 0 ? C.green : C.red }}>{(r[3] as number) >= 0 ? '+' : ''}{(r[3] as number).toFixed(4)}</span>
            </div>
          ))}
          <div style={{ marginTop: 10, color: C.soft, fontSize: 16 }}>Best: <b>K3 bal 0.1680</b> · K5 掉到 NoQ 下</div>
        </Card>
        <Card color={C.amber}>
          <div style={{ color: C.amber, fontWeight: 900, fontSize: 20, marginBottom: 10 }}>Bucket full-Q quarter · high_q9</div>
          <div style={{ display: 'grid', gridTemplateColumns: '70px 100px 90px 80px', gap: 6, color: C.muted, fontFamily: mono, fontSize: 14 }}>
            <span>K</span><span>Strategy</span><span>CLAP</span><span>ΔNoQ</span>
          </div>
          {[
            [3, 'balanced', 0.1573, -0.0086],
            [5, 'balanced', 0.1547, -0.0112],
            [5, 'fixed', 0.151, -0.0149],
            [10, 'balanced', 0.1488, -0.0171],
            [2, 'balanced', 0.1473, -0.0186],
            [10, 'fixed', 0.1318, -0.0341],
          ].map((r) => (
            <div key={String(r[0]) + r[1]} style={{ display: 'grid', gridTemplateColumns: '70px 100px 90px 80px', gap: 6, padding: '8px 0', borderTop: `1px solid ${C.line}`, fontSize: 17, fontFamily: mono }}>
              <span style={{ fontWeight: 900 }}>K{r[0]}</span>
              <span>{r[1]}</span>
              <span style={{ fontWeight: 900, color: C.amber }}>{(r[2] as number).toFixed(4)}</span>
              <span style={{ color: C.red }}>{(r[3] as number).toFixed(4)}</span>
            </div>
          ))}
          <div style={{ marginTop: 10, color: C.soft, fontSize: 16 }}>Best full-Q still <b>−0.0086</b> vs NoQ · fixed 更差</div>
        </Card>
      </div>
      <Card style={{ marginTop: 14 }}>
        <div style={{ fontSize: 19, fontWeight: 800 }}>Quarter takeaway</div>
        <div style={{ marginTop: 8, color: C.soft, fontSize: 18, lineHeight: 1.4 }}>
          全程 Q（B）在 quarter 上全面輸 NoQ；只在 S2 加 Q（C）可小幅超過 NoQ，且 <b>K≈3</b> 最佳——與 full S2Q 主線一致。
        </div>
      </Card>
    </div>
  </Shell>
);

// ── L3-A Caption bug problem ───────────────────────────────
const L3Problem: Page = () => (
  <Shell page="05 / 07" line="主線 3 · Caption alignment">
    <div style={{ position: 'absolute', left: 52, right: 52, top: 40, bottom: 52 }}>
      <Header
        kicker="MAIN LINE 3 · CAPTION–AUDIO WINDOW"
        title="Bug：caption 視窗 ≠ 訓練 audio"
        subtitle="訓練/VAE 用 clip 前 10s · 舊 caption 常描述整段 30s → text–audio 錯位"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <Card color={C.red} style={{ minHeight: 340 }}>
          <div style={{ color: C.red, fontWeight: 900, fontSize: 24 }}>Before</div>
          <div style={{ marginTop: 18, fontSize: 20, color: C.soft, lineHeight: 1.45 }}>
            · Audio / latents：<b>first 10s</b><br />
            · Caption：catalog / Qwen on longer context（≈30s）<br />
            · NPZ <code style={{ color: C.amber }}>text_*</code> 跟舊 caption 綁定<br />
            · 模型被要求對「聽不到的後半」對齊文字
          </div>
          <div style={{ marginTop: 24, fontFamily: mono, fontSize: 20, color: C.red }}>→ CLAP / 聽感雙輸風險</div>
        </Card>
        <Card color={C.green} style={{ minHeight: 340 }}>
          <div style={{ color: C.green, fontWeight: 900, fontSize: 24 }}>Fix pipeline</div>
          <div style={{ marginTop: 18, fontSize: 20, color: C.soft, lineHeight: 1.5 }}>
            1. Qwen recaption on <b>first-10s crop</b><br />
            2. 建 <code style={{ color: C.teal }}>caption10s_train.tsv</code><br />
            3. NPZ <b>in-place</b> rewrite text_* only<br />
            4. Keep mean/std VAE latents<br />
            5. Distributional CLAP gate n=1024
          </div>
          <div style={{ marginTop: 20, fontFamily: mono, fontSize: 17, color: C.green }}>
            meanΔ +0.0345 · frac+ 61.6% · manual pass (p25 only fail)
          </div>
        </Card>
      </div>
      <Card style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ fontSize: 18, color: C.soft }}>Train recipe after fix：</div>
          <div style={{ fontFamily: mono, fontSize: 18, color: C.green }}>NoQ · quarter 100k+50k → full 400k+200k</div>
          <div style={{ fontFamily: mono, fontSize: 16, color: C.muted }}>exp: phase8_qwen_caption10s_noq_*</div>
        </div>
      </Card>
    </div>
  </Shell>
);

// ── L3-B Caption results ───────────────────────────────────
const L3Results: Page = () => (
  <Shell page="06 / 07" line="主線 3 · Caption alignment">
    <div style={{ position: 'absolute', left: 52, right: 52, top: 40, bottom: 52 }}>
      <Header
        kicker="MAIN LINE 3 · RESULTS"
        title="Caption10s NoQ · quarter 已驗證"
        subtitle="同一 NoQ 協議下，對齊視窗後 1/4 步數逼近舊 full NoQ"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        {[
          { t: 'Old NoQ quarter', v: '0.1659', s: '30s-aware captions', c: C.gray },
          { t: 'Caption10s NoQ quarter', v: '0.1734', s: 'first-10s Qwen + NPZ text', c: C.green },
          { t: 'Official NoQ FULL (ref)', v: '0.1735', s: '400k+200k · old captions', c: C.purple },
        ].map((x) => (
          <Card key={x.t} color={x.c} style={{ minHeight: 220 }}>
            <div style={{ color: C.muted, fontSize: 16 }}>{x.t}</div>
            <div style={{ marginTop: 12, fontFamily: mono, fontSize: 48, fontWeight: 950, color: x.c }}>{x.v}</div>
            <div style={{ marginTop: 10, color: C.soft, fontSize: 18 }}>{x.s}</div>
          </Card>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.9fr', gap: 16, marginTop: 16 }}>
        <Card>
          <div style={{ fontWeight: 900, fontSize: 20 }}>Δ / AES note</div>
          <div style={{ marginTop: 12, fontFamily: mono, fontSize: 20, lineHeight: 1.6, color: C.soft }}>
            Caption10s − old quarter NoQ = <span style={{ color: C.green }}>+0.0075</span><br />
            Caption10s − full NoQ ref = <span style={{ color: C.soft }}>−0.0001</span><br />
            AES CE 5.77（低於舊 NoQ 的 CE；CLAP 優先對齊）
          </div>
        </Card>
        <Card color={C.amber}>
          <div style={{ color: C.amber, fontWeight: 900, fontSize: 20 }}>Full scale status</div>
          <div style={{ marginTop: 12, fontSize: 19, color: C.soft, lineHeight: 1.45 }}>
            <b>Caption10s NoQ full</b> S1 400k 訓練中<br />
            → 接著 S2 200k + MusicCaps eval<br />
            目標：驗證 full 是否延續 quarter 增益
          </div>
        </Card>
      </div>
    </div>
  </Shell>
);

// ── Human AB ───────────────────────────────────────────────
const HumanAB: Page = () => (
  <Shell page="07 / 07" line="人耳 · Full scale K">
    <div style={{ position: 'absolute', left: 48, right: 48, top: 36, bottom: 50 }}>
      <Header
        kicker="LISTENING TEST · FULL SCALE"
        title="人耳 CMOS：Stage1 · NoQ · K2 · K3"
        subtitle="participant hello · round1 · 12 pairs · scale −3…+3（正=偏好 A）· metrics: audioQuality + promptFollowing"
      />
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: 16 }}>
        <Card>
          <div style={{ fontWeight: 900, fontSize: 20, marginBottom: 10 }}>Win counts（summary field）</div>
          <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr 1fr 1fr', gap: 8, fontFamily: mono, fontSize: 16, color: C.muted }}>
            <span></span><span>stage1</span><span>noq</span><span>k2</span><span>k3</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr 1fr 1fr', gap: 8, marginTop: 10, fontFamily: mono, fontSize: 22, fontWeight: 900 }}>
            <span style={{ color: C.muted, fontSize: 16, fontWeight: 600 }}>AQ wins</span>
            <span style={{ color: C.green }}>5</span><span style={{ color: C.soft }}>3</span><span style={{ color: C.soft }}>1</span><span style={{ color: C.red }}>0</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr 1fr 1fr', gap: 8, marginTop: 8, fontFamily: mono, fontSize: 22, fontWeight: 900 }}>
            <span style={{ color: C.muted, fontSize: 16, fontWeight: 600 }}>PF wins</span>
            <span style={{ color: C.green }}>3</span><span style={{ color: C.soft }}>1</span><span style={{ color: C.soft }}>0</span><span style={{ color: C.soft }}>0</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr 1fr 1fr', gap: 8, marginTop: 8, fontFamily: mono, fontSize: 18 }}>
            <span style={{ color: C.muted }}>AQ score</span>
            <span>10</span><span>5</span><span>1</span><span>0</span>
          </div>
          <div style={{ marginTop: 14, color: C.muted, fontSize: 16 }}>Ties: AQ 3 · PF 8（prompt 多數難分）</div>
        </Card>
        <Card color={C.teal}>
          <div style={{ fontWeight: 900, fontSize: 20, marginBottom: 10 }}>Mean preference（越高越好）</div>
          {[
            ['stage1', 1.667, 0.5, C.green],
            ['noq', 0.667, 0.167, C.soft],
            ['k2', -0.833, -0.167, C.amber],
            ['k3', -1.5, -0.5, C.red],
          ].map(([n, aq, pf, c]) => (
            <div key={String(n)} style={{ display: 'grid', gridTemplateColumns: '100px 1fr 1fr', gap: 10, padding: '10px 0', borderTop: `1px solid ${C.line}`, alignItems: 'center' }}>
              <div style={{ fontWeight: 800, fontSize: 20, color: c as string }}>{n}</div>
              <div style={{ fontFamily: mono, fontSize: 20 }}>AQ <b style={{ color: c as string }}>{(aq as number) >= 0 ? '+' : ''}{(aq as number).toFixed(2)}</b></div>
              <div style={{ fontFamily: mono, fontSize: 20 }}>PF <b>{(pf as number) >= 0 ? '+' : ''}{(pf as number).toFixed(2)}</b></div>
            </div>
          ))}
        </Card>
      </div>
      <Card style={{ marginTop: 14 }} color={C.amber}>
        <div style={{ fontSize: 20, fontWeight: 900 }}>Reading（fullscale K 人耳）</div>
        <div style={{ marginTop: 8, fontSize: 19, color: C.soft, lineHeight: 1.4 }}>
          <b>stage1 ≫ noq ≫ k2 ≳ k3</b> · K-bucket fullscale 聽感明顯輸 NoQ / stage1。
          與機器 CLAP「S2Q K3/K5 略贏 NoQ」不同維度——此 AB 的 k2/k3 是 fullscale 對決臂，需與 ckpt 對表後再寫 paper claim。
        </div>
      </Card>
    </div>
  </Shell>
);

export const meta: SlideMeta = {
  title: 'Three main lines + human AB',
  createdAt: '2026-08-06T00:00:00.000Z',
};

export default [L1Design, L1Results, L2Design, L2Results, L3Problem, L3Results, HumanAB] satisfies Page[];
