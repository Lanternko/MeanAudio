"use client";

import { useMemo, useState } from "react";

type ModelId = "m0" | "m1" | "m2" | "m3";
type PromptId = "p1" | "p2" | "p3";

type Question = {
  id: string;
  prompt: PromptId;
  left: ModelId;
  right: ModelId;
};

type Vote = {
  questionId: string;
  prompt: PromptId;
  winner: ModelId;
  loser: ModelId;
};

const models: Record<ModelId, { name: string; detail: string }> = {
  m0: { name: "Stage 1 NoQ baseline", detail: "FM25 · CFG 4.5" },
  m1: { name: "Caption 1.0 Stage 2", detail: "MF25 · CFG 4.5" },
  m2: { name: "Caption 2.0 Stage 1", detail: "FM25 · CFG 4.5" },
  m3: { name: "Caption 2.0 Stage 2", detail: "MF25 · CFG 4.5" },
};

const prompts: Record<PromptId, { eyebrow: string; title: string; text: string }> = {
  p1: {
    eyebrow: "女聲抒情",
    title: "弦樂、鋼琴與柔和女聲",
    text: "低傳真抒情曲，持續弦樂與柔和鋼琴托住輕柔女聲；情緒悲傷、有靈魂，帶有週日禮拜音樂的氛圍。",
  },
  p2: {
    eyebrow: "吉他器樂",
    title: "放鬆的電吉他小品",
    text: "電吉他主奏搭配簡單鼓點、低音與鋼琴和弦，沒有歌聲；整體放鬆，適合咖啡店播放。",
  },
  p3: {
    eyebrow: "人聲節奏",
    title: "男聲旋律與規律指響",
    text: "男聲唱出速度變化的旋律，同時規律打指響；錄音帶有空房間的自然回音與練習感。",
  },
};

const modelIds = Object.keys(models) as ModelId[];
const promptIds = Object.keys(prompts) as PromptId[];
const pairs: [ModelId, ModelId][] = [];

for (let i = 0; i < modelIds.length; i += 1) {
  for (let j = i + 1; j < modelIds.length; j += 1) {
    pairs.push([modelIds[i], modelIds[j]]);
  }
}

function shuffle<T>(items: T[]) {
  const next = [...items];
  for (let i = next.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [next[i], next[j]] = [next[j], next[i]];
  }
  return next;
}

function buildQuestions(total: 6 | 12 | 18): Question[] {
  const repetitions = total / pairs.length;
  const questions: Question[] = [];

  for (let repeat = 0; repeat < repetitions; repeat += 1) {
    pairs.forEach(([first, second], pairIndex) => {
      const prompt = promptIds[(pairIndex + repeat) % promptIds.length];
      const swap = Math.random() > 0.5;
      questions.push({
        id: `${repeat}-${pairIndex}-${Math.random().toString(36).slice(2, 8)}`,
        prompt,
        left: swap ? second : first,
        right: swap ? first : second,
      });
    });
  }

  return shuffle(questions);
}

function audioPath(prompt: PromptId, model: ModelId) {
  return `/audio/${prompt}/${model}.flac`;
}

export default function Home() {
  const [phase, setPhase] = useState<"intro" | "test" | "results">("intro");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [index, setIndex] = useState(0);
  const [votes, setVotes] = useState<Vote[]>([]);
  const [listened, setListened] = useState({ left: false, right: false });
  const [copied, setCopied] = useState(false);

  const current = questions[index];
  const progress = questions.length ? ((index + 1) / questions.length) * 100 : 0;

  const ranking = useMemo(() => {
    return modelIds
      .map((id) => {
        const appearances = votes.filter((vote) => vote.winner === id || vote.loser === id).length;
        const wins = votes.filter((vote) => vote.winner === id).length;
        return { id, wins, appearances, rate: appearances ? wins / appearances : 0 };
      })
      .sort((a, b) => b.rate - a.rate || b.wins - a.wins);
  }, [votes]);

  function start(total: 6 | 12 | 18) {
    setQuestions(buildQuestions(total));
    setVotes([]);
    setIndex(0);
    setListened({ left: false, right: false });
    setCopied(false);
    setPhase("test");
  }

  function markListened(side: "left" | "right", audio: HTMLAudioElement) {
    document.querySelectorAll("audio").forEach((item) => {
      if (item !== audio) item.pause();
    });
    if (audio.currentTime >= 2) {
      setListened((value) => ({ ...value, [side]: true }));
    }
  }

  function vote(side: "left" | "right") {
    if (!current || !listened.left || !listened.right) return;
    const winner = current[side];
    const loser = side === "left" ? current.right : current.left;
    setVotes((value) => [
      ...value,
      { questionId: current.id, prompt: current.prompt, winner, loser },
    ]);

    if (index === questions.length - 1) {
      setPhase("results");
    } else {
      setIndex((value) => value + 1);
      setListened({ left: false, right: false });
    }
  }

  async function copyResults() {
    const lines = ranking.map(
      (item, position) =>
        `${position + 1}. ${models[item.id].name}: ${item.wins}/${item.appearances} (${Math.round(item.rate * 100)}%)`,
    );
    await navigator.clipboard.writeText(
      [`MeanAudio 25-step 盲聽結果（${votes.length} 題）`, ...lines].join("\n"),
    );
    setCopied(true);
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="MeanAudio Blind Lab 首頁">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
          <span>MeanAudio <strong>Blind Lab</strong></span>
        </a>
        <div className="protocol"><span /> Seed 42 · 25 steps · CFG 4.5</div>
      </header>

      {phase === "intro" && (
        <section className="intro" id="top">
          <div className="kicker">FOUR MODELS · ONE FAIR LISTEN</div>
          <h1>別讓分數替<br />你的耳朵做決定。</h1>
          <p className="lede">
            四個模型、相同 prompt、相同生成設定。每題只比較兩段匿名音檔，
            完成後才揭曉模型與你的個人勝率。
          </p>

          <div className="round-picker" aria-label="選擇測試長度">
            <button onClick={() => start(6)}>
              <span className="round-count">6</span>
              <span><strong>快速試聽</strong><small>每種模型配對一次</small></span>
              <b>約 3 分鐘 →</b>
            </button>
            <button className="featured" onClick={() => start(12)}>
              <em>推薦</em>
              <span className="round-count">12</span>
              <span><strong>平衡測試</strong><small>每種模型配對兩次</small></span>
              <b>約 6 分鐘 →</b>
            </button>
            <button onClick={() => start(18)}>
              <span className="round-count">18</span>
              <span><strong>完整測試</strong><small>所有配對 × 三種內容</small></span>
              <b>約 9 分鐘 →</b>
            </button>
          </div>

          <div className="rules">
            <div><span>01</span><p><strong>匿名隨機</strong>模型名稱、左右位置與題序都不公開。</p></div>
            <div><span>02</span><p><strong>兩邊都聽</strong>各播放至少 2 秒後才能選擇。</p></div>
            <div><span>03</span><p><strong>選你喜歡的</strong>不必猜哪段 CLAP 比較高。</p></div>
          </div>
        </section>
      )}

      {phase === "test" && current && (
        <section className="test-shell" id="top">
          <div className="progress-row">
            <span>ROUND {String(index + 1).padStart(2, "0")} / {String(questions.length).padStart(2, "0")}</span>
            <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
            <button onClick={() => setPhase("intro")}>結束測試</button>
          </div>

          <div className="prompt-card">
            <span>{prompts[current.prompt].eyebrow}</span>
            <h2>{prompts[current.prompt].title}</h2>
            <p>{prompts[current.prompt].text}</p>
          </div>

          <div className="comparison" key={current.id}>
            {(["left", "right"] as const).map((side, sideIndex) => (
              <article className={`audio-card ${listened[side] ? "heard" : ""}`} key={side}>
                <div className="sample-label">
                  <span>SAMPLE</span>
                  <strong>{sideIndex === 0 ? "A" : "B"}</strong>
                </div>
                <div className="fake-wave" aria-hidden="true">
                  {Array.from({ length: 31 }, (_, bar) => <i key={bar} />)}
                </div>
                <audio
                  controls
                  preload="metadata"
                  src={audioPath(current.prompt, current[side])}
                  onTimeUpdate={(event) => markListened(side, event.currentTarget)}
                />
                <button
                  className="vote-button"
                  disabled={!listened.left || !listened.right}
                  onClick={() => vote(side)}
                >
                  選擇 {sideIndex === 0 ? "A" : "B"} <span>→</span>
                </button>
                <small>{listened[side] ? "已完成試聽" : "播放至少 2 秒"}</small>
              </article>
            ))}
            <div className="versus">VS</div>
          </div>

          <p className="hint">
            {listened.left && listened.right
              ? "兩段都已試聽，選擇你整體更喜歡的一段。"
              : "請先播放兩段音檔；切換播放時，另一段會自動暫停。"}
          </p>
        </section>
      )}

      {phase === "results" && (
        <section className="results" id="top">
          <div className="kicker">YOUR BLIND LISTENING RESULT</div>
          <h1>耳朵已經投票。</h1>
          <p className="lede">共完成 {votes.length} 題。勝率是獲選次數除以該模型實際出場次數。</p>

          <div className="ranking">
            {ranking.map((item, position) => (
              <article key={item.id} className={position === 0 ? "winner" : ""}>
                <span className="rank">{String(position + 1).padStart(2, "0")}</span>
                <div className="model-name">
                  <strong>{models[item.id].name}</strong>
                  <small>{models[item.id].detail}</small>
                </div>
                <div className="score-bar"><i style={{ width: `${item.rate * 100}%` }} /></div>
                <div className="score">
                  <strong>{Math.round(item.rate * 100)}%</strong>
                  <small>{item.wins} / {item.appearances} 勝</small>
                </div>
              </article>
            ))}
          </div>

          <div className="result-actions">
            <button className="primary" onClick={copyResults}>{copied ? "已複製" : "複製結果"}</button>
            <button onClick={() => start(questions.length as 6 | 12 | 18)}>同長度再測一次</button>
            <button onClick={() => setPhase("intro")}>更換測試長度</button>
          </div>
          <p className="caveat">這是個人偏好測試，不等同於統計顯著性或模型整體品質排名。</p>
        </section>
      )}

      <footer>
        <span>MeanAudio · Phase 8 listening study</span>
        <span>MusicCaps · NoQ · NoMask · Full precision</span>
      </footer>
    </main>
  );
}
