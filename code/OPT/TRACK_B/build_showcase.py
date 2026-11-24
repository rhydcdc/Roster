"""Build standalone ROSTER_SHOWCASE.html (+ VIZ/roster_showcase.html) from showcase.json + showcase_js.py."""
import json
import sys
from pathlib import Path

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
DATA_JSON = PROJ / 'VIZ' / 'data_roster' / 'showcase.json'
OUT_ROOT = PROJ / 'ROSTER_SHOWCASE.html'
OUT_VIZ = PROJ / 'VIZ' / 'roster_showcase.html'

if str(HERE_B) not in sys.path:
    sys.path.insert(0, str(HERE_B))

from showcase_js import SHOWCASE_JS  # noqa: E402

CSS = r"""
:root {
  --bg: #060913;
  --bg2: #0b1224;
  --glass: rgba(12, 18, 34, 0.78);
  --border: rgba(148, 163, 184, 0.16);
  --text: #f1f5f9;
  --muted: #94a3b8;
  --cyan: #22d3ee;
  --gold: #fbbf24;
  --green: #34d399;
  --coral: #fb7185;
  --violet: #a78bfa;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: 'Inter', 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: radial-gradient(ellipse 80% 50% at 50% -10%, #101e3a 0%, var(--bg) 65%);
  background-color: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

/* Top Sticky Glass Bar */
.topbar {
  position: sticky; top: 0; z-index: 50;
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 10px 18px;
  background: rgba(6, 9, 19, 0.84);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 12px var(--cyan);
}
.brand-name {
  font-family: var(--mono); font-weight: 800; font-size: 15px; letter-spacing: 0.08em;
  background: linear-gradient(90deg, #67e8f9, #34d399);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-sub { font-size: 12px; color: var(--muted); font-weight: 500; }

.nav-chs { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.nav-ch {
  display: inline-flex; align-items: center; gap: 5px;
  background: transparent; color: var(--muted);
  border: 1px solid transparent; border-radius: 999px;
  padding: 5px 11px; font-size: 12px; font-weight: 600; cursor: pointer;
  transition: all 0.18s ease;
}
.nav-ch:hover { color: var(--text); background: rgba(148,163,184,0.08); }
.nav-ch.active {
  color: #e0f2fe; background: rgba(34, 211, 238, 0.14);
  border-color: rgba(34, 211, 238, 0.42);
}
.ch-num { font-family: var(--mono); font-size: 10.5px; color: var(--cyan); }

.top-actions { display: flex; align-items: center; gap: 6px; }
.seg-grp {
  display: inline-flex; background: rgba(15,23,42,0.85);
  border: 1px solid var(--border); border-radius: 999px; padding: 2px;
}
.pill-btn {
  background: transparent; color: var(--muted); border: none; border-radius: 999px;
  padding: 5px 11px; font-size: 11.5px; font-weight: 600; cursor: pointer;
  transition: all 0.16s ease; text-decoration: none; white-space: nowrap;
}
.pill-btn:hover { color: var(--text); }
.pill-btn.active { background: rgba(34, 211, 238, 0.18); color: #67e8f9; }
.pill-outline {
  border: 1px solid var(--border); background: rgba(15, 23, 42, 0.72);
  color: var(--text); border-radius: 999px; padding: 5px 11px;
  font-size: 11.5px; font-weight: 700; cursor: pointer; text-decoration: none;
  font-family: var(--mono);
}
.pill-outline:hover { border-color: var(--cyan); color: var(--cyan); }

/* Stage Container */
.stage-wrap {
  max-width: 1320px; margin: 0 auto; padding: 12px 18px 92px;
}
body.mode-cinema .stage-wrap {
  padding: 10px 18px 78px;
}

/* Scene Sections */
.scene-sec {
  display: flex; flex-direction: column; gap: 8px;
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 13px 16px;
  box-shadow: 0 20px 50px rgba(0,0,0,0.45);
  margin-bottom: 24px;
}
body.mode-cinema .scene-sec { display: none; margin-bottom: 0; }
body.mode-cinema .scene-sec.is-active { display: flex; }

.sec-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 10px; flex-wrap: wrap;
}
.sec-tag {
  font-family: var(--mono); font-size: 10.5px; font-weight: 700;
  color: var(--cyan); text-transform: uppercase; letter-spacing: 0.08em;
}
.sec-title {
  font-size: clamp(15px, 2.0vw, 20px); font-weight: 800; letter-spacing: -0.02em;
  color: #f8fafc; margin-top: 1px;
}

/* 4-Chip Live Numeric HUD */
.hud-row {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px;
}
.hud-card {
  background: rgba(8, 13, 26, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 10px; padding: 7px 11px;
  display: flex; flex-direction: column; gap: 1px;
}
.hud-lbl { font-size: 10px; font-weight: 600; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hud-val {
  font-family: var(--mono); font-size: clamp(13.5px, 1.65vw, 18px); font-weight: 800;
  font-variant-numeric: tabular-nums; color: #f8fafc;
}
.hud-val.c-cyan { color: var(--cyan); }
.hud-val.c-green { color: var(--green); }
.hud-val.c-gold { color: var(--gold); }
.hud-val.c-coral { color: var(--coral); }

/* Canvas Viewport */
.cv-box {
  position: relative; width: 100%;
  height: clamp(330px, 50vh, 490px);
  border-radius: 14px; overflow: hidden;
  background: #050812;
  border: 1px solid rgba(148, 163, 184, 0.14);
}
body.mode-cinema .cv-box {
  height: clamp(300px, calc(100vh - 340px), 520px);
}
.cv-box canvas { width: 100%; height: 100%; display: block; cursor: pointer; }

/* Scene Control Dock */
.ctrl-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; flex-wrap: wrap;
}
.ctrl-grp { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.c-btn {
  background: rgba(15, 23, 42, 0.85); color: #cbd5e1;
  border: 1px solid var(--border); border-radius: 9px;
  padding: 6px 11px; font-size: 11.5px; font-weight: 600; cursor: pointer;
  transition: all 0.15s ease; white-space: nowrap;
}
.c-btn:hover { border-color: rgba(34, 211, 238, 0.5); color: #f8fafc; }
.c-btn.active {
  background: rgba(34, 211, 238, 0.16); color: #67e8f9;
  border-color: rgba(34, 211, 238, 0.55);
}
.c-btn.t-gold.active { background: rgba(251,191,36,0.18); color: #fde68a; border-color: #fbbf24; }
.c-btn.t-green.active { background: rgba(52,211,153,0.18); color: #6ee7b7; border-color: #34d399; }
.c-btn.t-violet.active { background: rgba(167,139,250,0.18); color: #ddd6fe; border-color: #a78bfa; }
.c-btn.t-coral.active { background: rgba(251,113,133,0.18); color: #fecdd3; border-color: #fb7185; }

/* Scene 1 Image Strip */
.ex-strip { display: flex; align-items: center; gap: 6px; overflow-x: auto; padding: 2px 0; }
.ex-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(15, 23, 42, 0.85); color: #cbd5e1;
  border: 1px solid var(--border); border-radius: 10px;
  padding: 4px 9px 4px 5px; font-size: 11.5px; font-weight: 600; cursor: pointer;
  flex-shrink: 0; transition: all 0.15s ease;
}
.ex-chip img { border-radius: 5px; image-rendering: pixelated; background: #000; }
.ex-chip:hover { border-color: rgba(34, 211, 238, 0.5); }
.ex-chip.active { background: rgba(34, 211, 238, 0.16); border-color: var(--cyan); color: #e0f2fe; }

/* Scene 3 Morph Scrubber */
.morph-slider {
  flex: 1; min-width: 140px; accent-color: var(--green); cursor: pointer;
}

/* Bottom Floating Cinema Playbar */
.playbar {
  position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
  z-index: 60; width: min(94vw, 960px);
  background: rgba(8, 13, 26, 0.92);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 14px; padding: 7px 12px;
  display: flex; align-items: center; gap: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65);
}
/* In Scroll mode, make playbar an unobtrusive compact bottom-right floating pill */
body.mode-scroll .playbar {
  left: auto; right: 14px; transform: none; width: auto;
  padding: 6px 10px; border-radius: 999px;
}
body.mode-scroll .scrub-bar,
body.mode-scroll .spd-grp,
body.mode-scroll #btn-prev,
body.mode-scroll #btn-next,
body.mode-scroll #btn-fs { display: none; }
body.mode-cinema .footer { display: none; }
.play-btn {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, #06b6d4, #10b981);
  color: #041019; border: none; border-radius: 10px;
  padding: 7px 13px; font-size: 12px; font-weight: 800; cursor: pointer;
  flex-shrink: 0;
}
.play-btn:hover { filter: brightness(1.08); }

.scrub-bar { flex: 1; display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.scrub-seg { cursor: pointer; display: flex; flex-direction: column; gap: 4px; }
.scrub-track {
  height: 5px; border-radius: 99px; background: rgba(51, 65, 85, 0.7); overflow: hidden;
}
.scrub-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--cyan), var(--green)); border-radius: 99px; }
.scrub-seg.active .scrub-lbl { color: var(--cyan); }
.scrub-lbl {
  font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--muted);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

.spd-grp { display: inline-flex; gap: 3px; }
.spd-btn {
  background: rgba(15,23,42,0.8); color: var(--muted);
  border: 1px solid var(--border); border-radius: 7px;
  padding: 4px 7px; font-family: var(--mono); font-size: 10.5px; font-weight: 700; cursor: pointer;
}
.spd-btn.active { color: #fde68a; border-color: rgba(251,191,36,0.5); background: rgba(251,191,36,0.12); }

.footer {
  text-align: center; font-size: 11.5px; color: var(--muted);
  padding: 12px 16px 24px;
}

/* Responsive Mobile Tuning */
@media (max-width: 760px) {
  .topbar { padding: 8px 12px; }
  .brand-sub { display: none; }
  .nav-chs { order: 3; width: 100%; justify-content: space-between; }
  .nav-ch { padding: 4px 8px; font-size: 11px; }
  .ch-txt { display: none; }
  .stage-wrap { padding: 10px 10px 84px; }
  .scene-sec { padding: 12px; border-radius: 14px; }
  .hud-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cv-box { height: 360px; }
  .playbar { padding: 7px 10px; gap: 8px; }
  .scrub-lbl { display: none; }
  .spd-grp, #btn-fs { display: none; }
}
"""

HTML_TMPL = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ROSTER · Sparse FFN Visual Showcase</title>
<style>__CSS__</style>
</head>
<body class="mode-cinema">

<header class="topbar">
  <div class="brand">
    <span class="brand-dot"></span>
    <span class="brand-name">ROSTER</span>
    <span id="brand-sub" class="brand-sub">희소 FFN 시각 쇼케이스</span>
  </div>

  <nav class="nav-chs" aria-label="Chapters">
    <button class="nav-ch active" data-ch="0"></button>
    <button class="nav-ch" data-ch="1"></button>
    <button class="nav-ch" data-ch="2"></button>
    <button class="nav-ch" data-ch="3"></button>
    <button class="nav-ch" data-ch="4"></button>
  </nav>

  <div class="top-actions">
    <div class="seg-grp">
      <button id="btn-mode-cinema" class="pill-btn active">시네마</button>
      <button id="btn-mode-scroll" class="pill-btn">스크롤</button>
    </div>
    <button id="btn-lang" class="pill-outline">EN</button>
    <a id="btn-detail" class="pill-outline" href="ROSTER_VIZ.html">상세 분석</a>
  </div>
</header>

<main class="stage-wrap">

  <!-- ================= SCENE 01: SINGLE-TOKEN IGNITION ================= -->
  <section id="sec-s1" class="scene-sec is-active">
    <div class="sec-head">
      <div>
        <div class="sec-tag">01 · 단일 점화</div>
        <h1 class="sec-title">뉴런 86.5%는 잠든 채, 6.27배 빠르게</h1>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-card"><span id="s1-hl1" class="hud-lbl">평균 활성 뉴런</span><span id="s1-hv1" class="hud-val c-cyan">13.5%</span></div>
      <div class="hud-card"><span id="s1-hl2" class="hud-lbl">단일 요청 속도</span><span id="s1-hv2" class="hud-val c-gold">6.27×</span></div>
      <div class="hud-card"><span id="s1-hl3" class="hud-lbl">시험 정확도</span><span id="s1-hv3" class="hud-val c-green">90.81%</span></div>
      <div class="hud-card"><span id="s1-hl4" class="hud-lbl">예측 결과</span><span id="s1-hv4" class="hud-val">운동화 99%</span></div>
    </div>
    <div class="cv-box"><canvas id="cv-s1"></canvas></div>
    <div class="ctrl-bar">
      <div id="s1-ex-strip" class="ex-strip"></div>
      <div class="ctrl-grp">
        <button class="c-btn s1-dim-btn active" data-dim="4096">d=4,096</button>
        <button class="c-btn s1-dim-btn" data-dim="1024">d=1,024</button>
        <button id="s1-replay" class="c-btn">다시 쏘기</button>
      </div>
    </div>
  </section>

  <!-- ================= SCENE 02: THE BATCHING WALL ================= -->
  <section id="sec-s2" class="scene-sec">
    <div class="sec-head">
      <div>
        <div class="sec-tag">02 · 배치의 벽</div>
        <h2 class="sec-title">각각은 13.5%인데, 뭉치면 99.8%가 켜진다</h2>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-card"><span id="s2-hl1" class="hud-lbl">토큰당 활성</span><span id="s2-hv1" class="hud-val c-cyan">22.5%</span></div>
      <div class="hud-card"><span id="s2-hl2" class="hud-lbl">열 합집합 포화</span><span id="s2-hv2" class="hud-val c-coral">99.8%</span></div>
      <div class="hud-card"><span id="s2-hl3" class="hud-lbl">16×16 고정격자 ×0</span><span id="s2-hv3" class="hud-val c-gold">74.2%</span></div>
      <div class="hud-card"><span id="s2-hl4" class="hud-lbl">순진한 배치 가속</span><span id="s2-hv4" class="hud-val c-coral">1.00×</span></div>
    </div>
    <div class="cv-box"><canvas id="cv-s2"></canvas></div>
    <div class="ctrl-bar">
      <div class="ctrl-grp">
        <button id="s2-btn-rain" class="c-btn active">토큰 유입 (합집합 포화)</button>
        <button id="s2-btn-grid" class="c-btn t-coral">고정 16×16 격자 (×0 낭비)</button>
        <button id="s2-btn-hover" class="c-btn">토큰 1줄 탐색</button>
      </div>
    </div>
  </section>

  <!-- ================= SCENE 03: CH2 MAGNETIC MORPH ================= -->
  <section id="sec-s3" class="scene-sec">
    <div class="sec-head">
      <div>
        <div class="sec-tag">03 · CH2 정렬</div>
        <h2 class="sec-title">흩어진 별이 32열 텐서 코어 벽으로 뭉친다</h2>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-card"><span id="s3-hl1" class="hud-lbl">×0 낭비 비율</span><span id="s3-hv1" class="hud-val c-green">12.6%</span></div>
      <div class="hud-card"><span id="s3-hl2" class="hud-lbl">P2 합집합</span><span id="s3-hv2" class="hud-val c-cyan">16.1%</span></div>
      <div class="hud-card"><span id="s3-hl3" class="hud-lbl">DRAM 읽기</span><span id="s3-hv3" class="hud-val c-gold">64.5%</span></div>
      <div class="hud-card"><span id="s3-hl4" class="hud-lbl">1* 누락</span><span id="s3-hv4" class="hud-val c-green">0%</span></div>
    </div>
    <div class="cv-box"><canvas id="cv-s3"></canvas></div>
    <div class="ctrl-bar">
      <div class="ctrl-grp">
        <button class="c-btn s3-st-btn active" data-st="0">0 무작위</button>
        <button class="c-btn s3-st-btn" data-st="1">1 M=32</button>
        <button class="c-btn s3-st-btn" data-st="2">2 B1 가족</button>
        <button class="c-btn s3-st-btn t-green" data-st="3">3 B2 32열</button>
      </div>
      <input id="s3-slider" class="morph-slider" type="range" min="0" max="3" step="0.01" value="0" aria-label="Morph Stage">
    </div>
  </section>

  <!-- ================= SCENE 04: NON-GRID 5-TIER TILING + A3V4 ================= -->
  <section id="sec-s4" class="scene-sec">
    <div class="sec-head">
      <div>
        <div class="sec-tag">04 · 비격자 타일</div>
        <h2 class="sec-title">격자를 깬 5단계 틀 + Attention 뒤 0.00ms 배치</h2>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-card"><span id="s4-hl1" class="hud-lbl">전체 타일 (B=2,048)</span><span id="s4-hv1" class="hud-val c-gold">4,103</span></div>
      <div class="hud-card"><span id="s4-hl2" class="hud-lbl">64×32 대형틀 비중</span><span id="s4-hv2" class="hud-val c-cyan">76.7%</span></div>
      <div class="hud-card"><span id="s4-hl3" class="hud-lbl">타일 채움률</span><span id="s4-hv3" class="hud-val c-green">87.4%</span></div>
      <div class="hud-card"><span id="s4-hl4" class="hud-lbl">노출 배치 오버헤드</span><span id="s4-hv4" class="hud-val c-green">0.00 ms</span></div>
    </div>
    <div class="cv-box"><canvas id="cv-s4"></canvas></div>
    <div class="ctrl-bar">
      <div class="ctrl-grp">
        <button id="s4-f-all" class="c-btn s4-f-btn active" data-f="all">전체</button>
        <button id="s4-f-ng" class="c-btn s4-f-btn t-gold" data-f="ng">비격자만</button>
        <button class="c-btn s4-f-btn t-gold" data-f="4">64×32</button>
        <button class="c-btn s4-f-btn" data-f="0">32×32</button>
        <button class="c-btn s4-f-btn t-violet" data-f="1">32×16</button>
        <button class="c-btn s4-f-btn t-green" data-f="2">16×16</button>
        <button class="c-btn s4-f-btn t-coral" data-f="3">8×16</button>
      </div>
      <button id="s4-replay" class="c-btn">타일 덮기 재생</button>
    </div>
  </section>

  <!-- ================= SCENE 05: GPU SILICON RACE ================= -->
  <section id="sec-s5" class="scene-sec">
    <div class="sec-head">
      <div>
        <div class="sec-tag">05 · 실리콘 레이스</div>
        <h2 class="sec-title">RTX 4060 실측 속도 — 단일 6.27배 · 대형 배치 4.57배</h2>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-card"><span id="s5-hl1" class="hud-lbl">단일 요청 (B=1)</span><span id="s5-hv1" class="hud-val c-cyan">6.27×</span></div>
      <div class="hud-card"><span id="s5-hl2" class="hud-lbl">배치 겹침 E2E</span><span id="s5-hv2" class="hud-val c-green">4.57×</span></div>
      <div class="hud-card"><span id="s5-hl3" class="hud-lbl">순수 직렬 E2E</span><span id="s5-hv3" class="hud-val c-gold">3.10×</span></div>
      <div class="hud-card"><span id="s5-hl4" class="hud-lbl">vs BSMR (10회)</span><span id="s5-hv4" class="hud-val c-green">타일 -80%</span></div>
    </div>
    <div class="cv-box"><canvas id="cv-s5"></canvas></div>
    <div class="ctrl-bar">
      <div class="ctrl-grp">
        <button id="s5-ovlp-btn" class="c-btn t-green active">Attention 겹침</button>
        <button id="s5-ser-btn" class="c-btn t-gold">순수 직렬</button>
        <button class="c-btn s5-dim-btn active" data-dim="4096">d=4,096</button>
        <button class="c-btn s5-dim-btn" data-dim="1024">d=1,024</button>
      </div>
      <button id="s5-replay" class="c-btn">레이스 재시작</button>
    </div>
  </section>

  <footer id="footer-note" class="footer"></footer>
</main>

<!-- Floating Cinema Playbar -->
<div class="playbar" role="region" aria-label="Playback Controls">
  <button id="btn-play" class="play-btn"><span id="btn-play-ico">⏸</span><span id="btn-play-lbl">일시정지</span></button>
  <button id="btn-prev" class="spd-btn" title="Previous Scene">◀</button>
  <div class="scrub-bar">
    <div class="scrub-seg active" data-ch="0"><div class="scrub-track"><div class="scrub-fill"></div></div><span class="scrub-lbl">01</span></div>
    <div class="scrub-seg" data-ch="1"><div class="scrub-track"><div class="scrub-fill"></div></div><span class="scrub-lbl">02</span></div>
    <div class="scrub-seg" data-ch="2"><div class="scrub-track"><div class="scrub-fill"></div></div><span class="scrub-lbl">03</span></div>
    <div class="scrub-seg" data-ch="3"><div class="scrub-track"><div class="scrub-fill"></div></div><span class="scrub-lbl">04</span></div>
    <div class="scrub-seg" data-ch="4"><div class="scrub-track"><div class="scrub-fill"></div></div><span class="scrub-lbl">05</span></div>
  </div>
  <button id="btn-next" class="spd-btn" title="Next Scene">▶</button>
  <div class="spd-grp">
    <button class="spd-btn" data-spd="0.5">0.5×</button>
    <button class="spd-btn active" data-spd="1">1×</button>
    <button class="spd-btn" data-spd="2">2×</button>
  </div>
  <button id="btn-fs" class="pill-outline" title="Fullscreen">⛶</button>
</div>

<script>window.__SHOWCASE_DATA__ = __DATA__;</script>
<script>__JS__</script>
</body>
</html>
"""


def main():
    if not DATA_JSON.exists():
        raise FileNotFoundError(f'Run extract_showcase_data.py first: missing {DATA_JSON}')
    raw_json = DATA_JSON.read_text(encoding='utf-8')
    # Sanity-validate JSON
    parsed = json.loads(raw_json)
    assert len(parsed['examples']) == 8
    assert len(parsed['batch_scene']['tiles']) > 0

    html = (
        HTML_TMPL
        .replace('__CSS__', CSS.strip())
        .replace('__DATA__', raw_json)
        .replace('__JS__', SHOWCASE_JS.strip())
    )
    OUT_ROOT.write_text(html, encoding='utf-8')
    # VIZ/roster_showcase.html points back to ../ROSTER_VIZ.html for its detail button
    OUT_VIZ.write_text(html.replace('href="ROSTER_VIZ.html"', 'href="roster_viz.html"'), encoding='utf-8')
    print(f'Built {OUT_ROOT} ({OUT_ROOT.stat().st_size / 1024:.1f} KB)')
    print(f'Built {OUT_VIZ} ({OUT_VIZ.stat().st_size / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
