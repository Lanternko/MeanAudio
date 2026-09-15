"""Blind A/B listening page: mf_dedup full CFG3+neg against three comparators.

All four arms are generated fresh on the same 20-clip MusicCaps subset with
identical flags (MeanFlow 25 / seed 42 / NoMask / --no_q), so the noise draw is
matched across arms. Subset generation does not reproduce the canonical
n=5521 noise draw, so these clips are for listening only -- do not score them.
"""
import base64, json, random, html
from pathlib import Path

SP = Path("/tmp/claude-1005/-home-kojiek-MeanAudio/e5c98664-5ac5-4456-86f6-c36bbad1c45b/scratchpad")
SEL = json.load(open(SP / "ab_sel.json"))
AUD = SP / "ab_audio_mp3"   # 64 kbps mono MP3; FLAC inline was too heavy to open

ARMS = {
    "A": ("A_mfdedup_full_cfg3",    "MF mf_dedup full · CFG 3+neg"),
    "B": ("B_slot0_full_cfg3",      "Qwen c2p0 slot0 full · CFG 3+neg"),
    "C": ("C_mfdedup_quarter_cfg3", "MF mf_dedup quarter · CFG 3+neg"),
    "D": ("D_mfdedup_full_cfg0",    "MF mf_dedup full · CFG 0"),
}
# 20 clips -> 10 captioner trials, 5 budget trials, 5 negprompt trials
PLAN = [("captioner", "B")] * 10 + [("budget", "C")] * 5 + [("negprompt", "D")] * 5

rng = random.Random(20260909)
items = []
for i, (r, (contrast, opp)) in enumerate(zip(SEL, PLAN), 1):
    pair = ["A", opp]
    rng.shuffle(pair)
    enc = {}
    for slot, arm in zip(("x", "y"), pair):
        b = (AUD / ARMS[arm][0] / f"{r['id']}.mp3").read_bytes()
        enc[slot] = "data:audio/mpeg;base64," + base64.b64encode(b).decode()
    items.append({"n": i, "id": r["id"], "caption": r["caption"], "contrast": contrast,
                  "x": enc["x"], "y": enc["y"], "armX": pair[0], "armY": pair[1]})
DATA = json.dumps(items, ensure_ascii=False)
NAMES = json.dumps({k: v[1] for k, v in ARMS.items()}, ensure_ascii=False)

CSS = """
:root{
  --paper:#f3f5f7; --surface:#ffffff; --ink:#15181d; --ink-soft:#5b6470;
  --line:#d5dbe3; --line-soft:#e6eaef; --accent:#0d6a70; --accent-soft:#e0efef;
  --hot:#b4611c; --on-accent:#ffffff; --shadow:0 1px 2px rgba(20,28,38,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101317; --surface:#171b21; --ink:#e8ecf1; --ink-soft:#98a3b1;
  --line:#2a323c; --line-soft:#222932; --accent:#4fb3b8; --accent-soft:#122b2d;
  --hot:#d98b45; --on-accent:#0b1113; --shadow:none;
}}
:root[data-theme="dark"]{
  --paper:#101317; --surface:#171b21; --ink:#e8ecf1; --ink-soft:#98a3b1;
  --line:#2a323c; --line-soft:#222932; --accent:#4fb3b8; --accent-soft:#122b2d;
  --hot:#d98b45; --on-accent:#0b1113; --shadow:none;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Noto Sans TC",system-ui,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:780px;margin:0 auto;padding:32px 20px 72px}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;
  border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:26px}
h1{font-size:1.5rem;font-weight:600;letter-spacing:-.01em;margin:0;text-wrap:balance}
.proto{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)}
.lede{color:var(--ink-soft);font-size:.95rem;max-width:62ch;margin:0 0 22px}
.rail{height:3px;background:var(--line-soft);border-radius:2px;overflow:hidden;margin-bottom:6px}
.rail i{display:block;height:100%;background:var(--accent);width:0;transition:width .25s}
.meta{display:flex;justify-content:space-between;font-size:.75rem;color:var(--ink-soft);margin-bottom:24px}
.prompt{border-left:2px solid var(--accent);padding:2px 0 2px 16px;margin-bottom:24px}
.prompt .lab{font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-soft)}
.prompt p{margin:6px 0 0;font-size:.95rem}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:560px){.pair{grid-template-columns:1fr}}
.side{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:16px;display:flex;flex-direction:column;gap:12px;box-shadow:var(--shadow)}
.side.on{border-color:var(--hot)}
.tag{font-size:1.05rem;font-weight:600}
.tag small{display:block;font-weight:400;font-size:.72rem;color:var(--ink-soft);
  letter-spacing:.05em;text-transform:uppercase}
audio{width:100%}
.heard{font-size:.72rem;color:var(--ink-soft)}
.heard.ok{color:var(--accent)}
.q{margin-top:22px}
.q .lab{font-size:.8rem;font-weight:600;display:block;margin-bottom:8px}
.q .sub{font-weight:400;color:var(--ink-soft);font-size:.78rem}
.votes{display:flex;gap:10px;flex-wrap:wrap}
button{font:inherit;cursor:pointer;border-radius:7px;border:1px solid var(--line);
  background:var(--surface);color:var(--ink);padding:10px 16px}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.42;cursor:not-allowed}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.sel{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.hint{font-size:.78rem;color:var(--ink-soft);margin-top:14px}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin:8px 0 20px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line-soft)}
th:first-child,td:first-child{text-align:left}
th{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-soft);font-weight:500}
td.num{font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
h2{font-size:1.05rem;font-weight:600;margin:30px 0 10px}
.note{font-size:.85rem;color:var(--ink-soft);max-width:64ch}
.log{font-size:.78rem}
.log td{padding:6px 10px}
.win{color:var(--accent);font-weight:600}
textarea{width:100%;height:150px;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.75rem;background:var(--surface);color:var(--ink);border:1px solid var(--line);
  border-radius:7px;padding:10px}
"""

JS = """
const DATA = __DATA__;
const NAMES = __NAMES__;
const CONTRAST = {captioner:"captioner（vs Qwen slot0 full）",
                  budget:"budget（vs 自己的 quarter）",
                  negprompt:"negative prompt（vs 自己的 CFG 0）"};
const order = DATA.map((d,i)=>i);
for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
let step=0, votes=[], heard={x:false,y:false}, pick={match:null, qual:null};
const $=s=>document.querySelector(s);

function render(){
  const d = DATA[order[step]];
  $("#rail").style.width = (step/DATA.length*100)+"%";
  $("#count").textContent = `第 ${step+1} 題 / 共 ${DATA.length}`;
  $("#clip").textContent = d.id;
  $("#cap").textContent = d.caption;
  heard={x:false,y:false}; pick={match:null, qual:null};
  document.querySelectorAll(".vb").forEach(b=>b.classList.remove("sel"));
  for(const s of ["x","y"]){
    const a=$("#a"+s); a.src=d[s]; a.load();
    $("#h"+s).textContent="尚未聽滿 3 秒"; $("#h"+s).className="heard";
    $("#side"+s).classList.remove("on");
  }
  lock();
}
function lock(){
  const ok = heard.x && heard.y;
  document.querySelectorAll(".vb").forEach(b=>b.disabled=!ok);
  const done = pick.match && pick.qual;
  $("#next").disabled = !(ok && done);
  $("#hint").textContent = !ok ? "兩段都聽滿 3 秒後才會開放投票。"
             : (done ? "兩題都答了，可以進下一題。" : "兩個問題都要選。");
}
function mark(side){
  const a=$("#a"+side);
  document.querySelectorAll("audio").forEach(o=>{if(o!==a)o.pause();});
  $("#sidex").classList.toggle("on", side==="x" && !a.paused);
  $("#sidey").classList.toggle("on", side==="y" && !a.paused);
  if(a.currentTime>=3 && !heard[side]){
    heard[side]=true; $("#h"+side).textContent="已聽過"; $("#h"+side).className="heard ok"; lock();
  }
}
function choose(q, c, btn){
  pick[q]=c;
  document.querySelectorAll(`.vb[data-q="${q}"]`).forEach(b=>b.classList.remove("sel"));
  btn.classList.add("sel"); lock();
}
function winner(d, c){ return c==="tie" ? null : (c==="x"?d.armX:d.armY); }
function next(){
  const d = DATA[order[step]];
  votes.push({n:d.n, id:d.id, contrast:d.contrast, armX:d.armX, armY:d.armY,
              match:pick.match, quality:pick.qual,
              matchWinner:winner(d,pick.match), qualityWinner:winner(d,pick.qual)});
  document.querySelectorAll("audio").forEach(a=>a.pause());
  step++;
  if(step>=DATA.length) results(); else render();
}
function tallyRows(field){
  return Object.keys(CONTRAST).map(k=>{
    const vs = votes.filter(v=>v.contrast===k);
    const a = vs.filter(v=>v[field]==="A").length;
    const o = vs.filter(v=>v[field] && v[field]!=="A").length;
    const t = vs.filter(v=>!v[field]).length;
    return `<tr><td>${CONTRAST[k]}</td><td class="num">${a}</td>`+
           `<td class="num">${o}</td><td class="num">${t}</td>`+
           `<td class="num">${a+o?Math.round(a/(a+o)*100)+"%":"—"}</td></tr>`;
  }).join("");
}
function results(){
  $("#test").hidden=true; $("#res").hidden=false;
  $("#rail").style.width="100%";
  $("#tallyMatch").innerHTML = tallyRows("matchWinner");
  $("#tallyQual").innerHTML  = tallyRows("qualityWinner");
  $("#log").innerHTML = votes.map((v,i)=>{
    const m = v.matchWinner ? NAMES[v.matchWinner] : "無法分辨";
    const q = v.qualityWinner ? NAMES[v.qualityWinner] : "無法分辨";
    return `<tr><td class="mono">${i+1}</td><td class="mono">${v.id}</td>`+
           `<td style="text-align:left">${CONTRAST[v.contrast].split("（")[0]}</td>`+
           `<td style="text-align:left" class="${v.matchWinner==="A"?'win':''}">${m}</td>`+
           `<td style="text-align:left" class="${v.qualityWinner==="A"?'win':''}">${q}</td></tr>`;
  }).join("");
  $("#raw").value = JSON.stringify(votes);
}
window.addEventListener("DOMContentLoaded",()=>{
  for(const s of ["x","y"]){
    const a=document.querySelector("#a"+s);
    a.addEventListener("timeupdate",()=>mark(s));
    a.addEventListener("play",()=>mark(s));
    a.addEventListener("pause",()=>{document.querySelector("#side"+s).classList.remove("on");});
  }
  document.querySelectorAll(".vb").forEach(b=>
    b.addEventListener("click",()=>choose(b.dataset.q, b.dataset.c, b)));
  document.querySelector("#next").addEventListener("click",next);
  document.querySelector("#start").addEventListener("click",()=>{
    document.querySelector("#intro").hidden=true;
    document.querySelector("#test").hidden=false; render();
  });
  document.querySelector("#copy").addEventListener("click",()=>{
    const t=document.querySelector("#raw"); t.select();
    navigator.clipboard.writeText(t.value).then(()=>{
      document.querySelector("#copy").textContent="已複製";},()=>{});
  });
});
"""

BODY = f"""<title>mf_dedup 盲聽對照</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>mf_dedup CFG 3 盲聽對照</h1>
  <span class="proto mono">MusicCaps 20 clips · MeanFlow 25 · seed 42 · NoQ</span>
</header>

<section id="intro">
  <p class="lede">主角固定是 <b>MF mf_dedup full · CFG 3 + fidelity negative</b>，
  每一題拿它跟三個對照之一比。20 題裡有 10 題比 captioner、5 題比訓練預算、
  5 題比 negative prompt 開關；題目順序打散，你看不出當下這題屬於哪一組。</p>

  <div class="scroll"><table>
    <thead><tr><th>對照</th><th>對手</th><th>客觀差距（主角 − 對手）</th></tr></thead>
    <tbody>
      <tr><td>captioner（10 題）</td><td>Qwen c2p0 slot0 full · CFG 3+neg</td>
          <td style="text-align:left">CLAP −0.0185、CU −0.276、PQ −0.385（皆超過 2× seed 底線）</td></tr>
      <tr><td>budget（5 題）</td><td>MF mf_dedup <b>quarter</b> · CFG 3+neg</td>
          <td style="text-align:left">CLAP +0.0187、AES 全在底線內</td></tr>
      <tr><td>negative prompt（5 題）</td><td>MF mf_dedup full · <b>CFG 0</b></td>
          <td style="text-align:left">CLAP +0.0342、CE +0.503、CU +0.557、PQ +0.639</td></tr>
    </tbody>
  </table></div>

  <p class="note">每題兩個問題：<b>哪個比較符合文字描述</b>、<b>哪個音質比較好</b>。
  客觀指標在這兩件事上會給出不同答案（CLAP vs CU/PQ），所以分開問。
  左右順序與甲乙對應每題重抽，兩段都聽滿 3 秒才開放投票，全部作答後才揭曉。</p>

  <p class="note">四個 arm 都是在同一份 20 題子集上、用同一組旗標重新生成的，
  所以彼此的雜訊抽樣是對齊的；但這不重現 n=5521 的 canonical 抽樣，
  <b>這些音檔只能拿來聽，不能拿來算分數</b>。
  頁面裡的音檔是 64 kbps 單聲道 MP3（原始輸出是 16 kHz mono FLAC），
  四個 arm 用完全相同的轉檔設定，所以不影響誰對誰的比較。</p>

  <p class="note">20 題是從 5,521 題隨機抽的（seed 20260909），沒有挑過。
  其中有一兩題的 prompt 本身描述的就是很安靜的內容，兩段可能都近乎無聲——
  那種情況選「差不多」就對了，不要為了分出高下而硬選。</p>

  <div class="votes" style="margin-top:20px"><button id="start" class="primary">開始 20 題</button></div>
</section>

<section id="test" hidden>
  <div class="rail"><i id="rail"></i></div>
  <div class="meta"><span id="count"></span><span class="mono" id="clip"></span></div>

  <div class="prompt">
    <span class="lab">Prompt</span>
    <p id="cap"></p>
  </div>

  <div class="pair">
    <div class="side" id="sidex">
      <div class="tag">甲<small>Sample A</small></div>
      <audio id="ax" controls preload="none"></audio>
      <span class="heard" id="hx">尚未聽滿 3 秒</span>
    </div>
    <div class="side" id="sidey">
      <div class="tag">乙<small>Sample B</small></div>
      <audio id="ay" controls preload="none"></audio>
      <span class="heard" id="hy">尚未聽滿 3 秒</span>
    </div>
  </div>

  <div class="q">
    <span class="lab">1 · 哪一段比較符合上面的文字描述？<span class="sub">樂器、風格、情緒是否對得上</span></span>
    <div class="votes">
      <button class="vb" data-q="match" data-c="x" disabled>甲</button>
      <button class="vb" data-q="match" data-c="tie" disabled>差不多</button>
      <button class="vb" data-q="match" data-c="y" disabled>乙</button>
    </div>
  </div>

  <div class="q">
    <span class="lab">2 · 哪一段音質比較好？<span class="sub">清晰度、雜訊、悅耳程度，先不管文字</span></span>
    <div class="votes">
      <button class="vb" data-q="qual" data-c="x" disabled>甲</button>
      <button class="vb" data-q="qual" data-c="tie" disabled>差不多</button>
      <button class="vb" data-q="qual" data-c="y" disabled>乙</button>
    </div>
  </div>

  <div class="votes" style="margin-top:24px">
    <button id="next" class="primary" disabled>下一題</button>
  </div>
  <p class="hint" id="hint"></p>
</section>

<section id="res" hidden>
  <h2>符合文字描述</h2>
  <div class="scroll"><table>
    <thead><tr><th>對照</th><th>主角勝</th><th>對手勝</th><th>平手</th><th>主角勝率</th></tr></thead>
    <tbody id="tallyMatch"></tbody>
  </table></div>

  <h2>音質</h2>
  <div class="scroll"><table>
    <thead><tr><th>對照</th><th>主角勝</th><th>對手勝</th><th>平手</th><th>主角勝率</th></tr></thead>
    <tbody id="tallyQual"></tbody>
  </table></div>

  <p class="note">主角 = MF mf_dedup full · CFG 3+neg。每組只有 5–10 題、單一聽者，
  只能看方向，不足以判定顯著；要當數字用需要多位聽者或每組 40 題以上。</p>

  <h2>逐題</h2>
  <div class="scroll"><table class="log">
    <thead><tr><th>#</th><th>clip</th><th>對照</th><th>文字</th><th>音質</th></tr></thead>
    <tbody id="log"></tbody>
  </table></div>

  <h2>原始作答</h2>
  <p class="note">貼回對話就能做統計。</p>
  <textarea id="raw" readonly></textarea>
  <div class="votes" style="margin-top:10px"><button id="copy">複製 JSON</button></div>
</section>
</div>
<script>{JS.replace("__DATA__", DATA).replace("__NAMES__", NAMES)}</script>
"""

out = SP / "mf_dedup_ab.html"
out.write_text(BODY, encoding="utf-8")
print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
