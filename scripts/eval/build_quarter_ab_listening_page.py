import base64, json, random, html
from pathlib import Path

SEL = json.load(open("/tmp/claude-1005/-home-kojiek-MeanAudio/2ad5a224-8c1a-476c-9a36-7877a8604376/scratchpad/ab_sel.json"))
ARMS = {
    "mf": "/home/kojiek/eval_output_nvme/mf_fullcov_noq_quarter_mc_mf25_cfg3_neg/audio",
    "slot0": "/home/kojiek/eval_output_nvme/c2p0_slot0_quarter_mc_mf25_cfg3_neg/audio",
}
rng = random.Random(20260907)
items = []
for i, r in enumerate(SEL, 1):
    pair = ["mf", "slot0"]
    rng.shuffle(pair)                      # which arm is x vs y, per clip
    enc = {}
    for slot, arm in zip(("x", "y"), pair):
        b = Path(f"{ARMS[arm]}/{r['id']}.flac").read_bytes()
        enc[slot] = "data:audio/flac;base64," + base64.b64encode(b).decode()
    items.append({"n": i, "id": r["id"], "caption": r["caption"],
                  "x": enc["x"], "y": enc["y"],
                  "armX": pair[0], "armY": pair[1]})
DATA = json.dumps(items, ensure_ascii=False)

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
.votes{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
button{font:inherit;cursor:pointer;border-radius:7px;border:1px solid var(--line);
  background:var(--surface);color:var(--ink);padding:11px 18px}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.42;cursor:not-allowed}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.hint{font-size:.78rem;color:var(--ink-soft);margin-top:12px}
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
"""

JS = """
const DATA = __DATA__;
const NAMES = {mf:"MF full-coverage quarter (038)", slot0:"c2p0 slot0 quarter"};
const order = DATA.map((d,i)=>i);
for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
let step=0, votes=[], heard={x:false,y:false};
const $=s=>document.querySelector(s);

function render(){
  const d = DATA[order[step]];
  $("#rail").style.width = (step/DATA.length*100)+"%";
  $("#count").textContent = `第 ${step+1} 題 / 共 ${DATA.length}`;
  $("#clip").textContent = d.id;
  $("#cap").textContent = d.caption;
  heard={x:false,y:false};
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
  $("#hint").textContent = ok ? "選出你覺得比較好的一段，或選無法分辨。"
                              : "兩段都聽滿 3 秒後才會開放投票。";
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
function vote(choice){
  const d = DATA[order[step]];
  votes.push({id:d.id, choice, armX:d.armX, armY:d.armY,
              winner: choice==="tie" ? null : (choice==="x"?d.armX:d.armY)});
  document.querySelectorAll("audio").forEach(a=>a.pause());
  step++;
  if(step>=DATA.length) results(); else render();
}
function results(){
  $("#test").hidden=true; $("#res").hidden=false;
  $("#rail").style.width="100%";
  const tally={mf:0,slot0:0}, ties=votes.filter(v=>!v.winner).length;
  votes.forEach(v=>{if(v.winner)tally[v.winner]++;});
  const decided=votes.length-ties;
  $("#tally").innerHTML = Object.keys(NAMES).map(k=>
    `<tr><td>${NAMES[k]}</td><td class="num">${tally[k]}</td>`+
    `<td class="num">${decided?Math.round(tally[k]/decided*100):0}%</td></tr>`).join("")
    + `<tr><td>無法分辨</td><td class="num">${ties}</td><td class="num">—</td></tr>`;
  $("#log").innerHTML = votes.map((v,i)=>{
    const w = v.winner ? NAMES[v.winner] : "無法分辨";
    return `<tr><td class="mono">${i+1}</td><td class="mono">${v.id}</td>`+
           `<td style="text-align:left" class="${v.winner?'win':''}">${w}</td></tr>`;
  }).join("");
}
window.addEventListener("DOMContentLoaded",()=>{
  for(const s of ["x","y"]){
    const a=document.querySelector("#a"+s);
    a.addEventListener("timeupdate",()=>mark(s));
    a.addEventListener("play",()=>mark(s));
    a.addEventListener("pause",()=>{document.querySelector("#side"+s).classList.remove("on");});
  }
  document.querySelectorAll(".vb").forEach(b=>b.addEventListener("click",()=>vote(b.dataset.c)));
  document.querySelector("#start").addEventListener("click",()=>{
    document.querySelector("#intro").hidden=true;
    document.querySelector("#test").hidden=false; render();
  });
});
"""

BODY = f"""<title>Quarter Arm Blind Listen</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Quarter arm 盲聽</h1>
  <span class="proto mono">MusicCaps · MeanFlow 25 · CFG 3.0 + fidelity neg · seed 42</span>
</header>

<section id="intro">
  <p class="lede">兩個 quarter arm，同一批 MusicCaps prompt、同一組生成設定，
  唯一的差別是訓練語料的 caption 來源。客觀分數上它們差得很近——CLAP 差 0.0076、
  四項 AES 反向——所以這一題只有耳朵能回答。</p>

  <div class="scroll"><table>
    <thead><tr><th>arm</th><th>CLAP</th><th>CE</th><th>CU</th><th>PC</th><th>PQ</th></tr></thead>
    <tbody>
      <tr><td>c2p0 slot0 quarter</td><td class="num">0.2248</td><td class="num">6.6952</td>
          <td class="num">7.3871</td><td class="num">4.6661</td><td class="num">7.3101</td></tr>
      <tr><td>MF full-coverage quarter</td><td class="num">0.2172</td><td class="num">6.9567</td>
          <td class="num">7.5810</td><td class="num">4.7247</td><td class="num">7.4771</td></tr>
    </tbody>
  </table></div>

  <p class="note">12 題，每題兩段匿名音檔。左右順序與甲乙對應都是隨機的，
  而且每一題重抽，所以「甲」不是固定的某一個模型。兩段都聽滿 3 秒才能投票。
  全部作答後才揭曉。</p>
  <div class="votes"><button id="start" class="primary">開始 12 題</button></div>
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

  <div class="votes">
    <button class="vb" data-c="x" disabled>甲比較好</button>
    <button class="vb" data-c="tie" disabled>無法分辨</button>
    <button class="vb" data-c="y" disabled>乙比較好</button>
  </div>
  <p class="hint" id="hint"></p>
</section>

<section id="res" hidden>
  <h2>你的結果</h2>
  <div class="scroll"><table>
    <thead><tr><th>arm</th><th>勝場</th><th>勝率（不計平手）</th></tr></thead>
    <tbody id="tally"></tbody>
  </table></div>
  <p class="note">12 題的樣本量只能看方向，不足以判定顯著。真正要當數字用的話，
  同一份題目需要多位聽者、或擴大到 40 題以上。</p>
  <h2>逐題</h2>
  <div class="scroll"><table class="log">
    <thead><tr><th>#</th><th>clip</th><th>你選的</th></tr></thead>
    <tbody id="log"></tbody>
  </table></div>
</section>
</div>
<script>{JS.replace("__DATA__", DATA)}</script>
"""

out = Path("/tmp/claude-1005/-home-kojiek-MeanAudio/2ad5a224-8c1a-476c-9a36-7877a8604376/scratchpad/quarter_ab.html")
out.write_text(BODY, encoding="utf-8")
print("wrote", out, round(out.stat().st_size/1e6, 2), "MB")
