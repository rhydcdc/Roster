"""레시피 HTML(VIZ/neuron_recipe.html)에 '묶어서 계산' 보기를 덧붙이고, RECIPE_VIZ.html을 다시 만든다. 기존 내용은 건드리지 않는다.
끼워 넣는 조각 넷은 모두 <!-- 배칭 시작 --> ... <!-- 배칭 끝 --> 안에 있다. 그 밖에 바꾸는 곳은 아키텍처 이름(NAME) 두 곳뿐이다.
조각을 빼고 이름을 되돌리면 원래 파일과 바이트까지 같은지 확인한다.
  1. 맨 위 막대의 전환에 버튼 '묶어서 계산'
  2. 본 스크립트 앞: 처음 주소의 #batch를 기억하는 한 줄 (기존 스크립트가 주소를 #train으로 바꾸기 전에)
  3. 푸터 앞: 새 보기 (꾸밈은 bt- 이름만)
  4. 파일 끝: 새 스크립트 (기존 스크립트는 고치지 않는다. 전환 버튼에 동작을 덧붙인다)
데이터: runs/viz_batch.json (viz_batch_data.py). 고른 방법(뉴런 쪽 블록 + 블록 합치기)만 넣는다.
python viz_batch_inject.py
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
PAGE = os.path.join(PROJ, 'VIZ', 'neuron_recipe.html')
BUILD = os.path.join(PROJ, 'VIZ', 'CODE', 'build_local.py')
LOCAL = os.path.join(PROJ, 'RECIPE_VIZ.html')
MS, ME = '<!-- 배칭 시작 -->', '<!-- 배칭 끝 -->'

TAB = MS + '<button id="tab-batch" type="button" aria-pressed="false">묶어서 계산</button>' + ME

PRE = MS + '<script>window.__HASH0__ = location.hash;</script>' + ME

VIEW = MS + r'''
  <!-- ============ 묶어서 계산 (배칭) ============ -->
  <div class="view" id="view-batch" hidden>
  <style>
  #view-batch .step, #view-batch .split { align-items: start; }
  #view-batch .bt-wide { grid-column: 1 / -1; }
  #view-batch .bt-wide .bt-tbl th, #view-batch .bt-wide .bt-tbl td { white-space: nowrap; }
  .bt-intro { padding-block: 36px 8px; border-top: 1px solid var(--line); }
  .bt-intro .no { font-family: var(--mono); font-size: 12px; color: var(--pick); }
  .bt-intro h2 { font-size: 22px; margin-top: 8px; }
  .bt-intro p { color: var(--fg-2); margin-top: 10px; max-width: 46em; }
  .bt-intro .stats { margin-top: 18px; }
  .bt-mat svg, .bt-tile svg, .bt-res svg { display: block; width: 100%; height: auto; }
  .bt-mat text, .bt-tile text, .bt-res text { font-family: var(--mono); font-size: 11px; fill: var(--fg-3); }
  .bt-mat .lab, .bt-tile .lab, .bt-res .lab { fill: var(--fg-2); font-size: 11.5px; }
  .bt-steps { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .bt-steps .where { font-size: 13px; color: var(--fg-2); flex: 1; min-width: 12em; }
  .bt-steps .where b { color: var(--pick); font-weight: 500; }
  .bt-side { display: grid; gap: 10px; font-size: 13px; color: var(--fg-2); }
  .bt-side b { color: var(--fg); font-weight: 600; }
  .bt-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  .bt-tbl th, .bt-tbl td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); }
  .bt-tbl th { color: var(--fg-3); font-weight: 500; }
  .bt-tbl td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .bt-tbl tr.sum td { color: var(--fg); border-bottom: 0; }
  .bt-tbl .ok { color: var(--on); } .bt-tbl .no-ok { color: var(--fg-3); }
  .bt-real { position: relative; }
  .bt-real canvas { width: 100%; border-radius: 8px; background: var(--bg); cursor: crosshair; }
  .bt-bar { display: flex; height: 10px; border-radius: 3px; overflow: hidden; background: var(--bg); gap: 2px; }
  .bt-bar i { display: block; height: 100%; }
  </style>

  <section class="bt-intro">
    <div class="no">묶어서 계산 · 여러 요청을 한 번에</div>
    <h2>같은 뉴런을 쓰는 토큰끼리 모아, 그 뉴런의 가중치를 한 번만 읽는다</h2>
    <p>서버는 여러 요청의 토큰을 한 번에 계산한다. 토큰마다 켜는 뉴런이 다르니, <b style="color:var(--fg);font-weight:600">뉴런마다 그 뉴런을 쓰는 토큰을 모아 블록으로 묶는다.</b> 블록마다 그 뉴런의 가중치를 한 번 읽어, 그 뉴런을 쓰는 토큰에만 계산한다. 블록이 너무 잘아서 GPU의 계산 조각이 비면 블록 둘을 합친다. 이것이 고른 방법, <b style="color:var(--fg);font-weight:600">뉴런 쪽 블록 + 블록 합치기</b>다. 읽기가 계산보다 훨씬 비싸서(뉴런 하나의 가중치를 읽는 시간 ≈ 토큰 하나 × 뉴런 하나를 계산하는 시간의 약 100배) 이 방법을 골랐다.</p>
    <div class="stats">
      <div class="stat"><div class="v num on">5.1%</div><div class="k">읽는 가중치 · 기본 풀 모델의 몇 % (토큰 256개)</div></div>
      <div class="stat"><div class="v num">51.6%</div><div class="k">하는 연산 · 기본 풀 모델의 몇 % (토큰 256개)</div></div>
      <div class="stat"><div class="v num">90.68%</div><div class="k">시험 정확도 · 묶어도 그대로</div></div>
    </div>
    <p style="font-size:13px;color:var(--fg-3);margin-top:12px">기본 풀 모델 = 뉴런을 다 켠 베이스 모델(91.02%)이 토큰을 들어온 순서대로 16개씩 계산하는 것. 연산량은 GPU가 실제로 계산하는 칸(16 단위로 올림), 읽는 양은 꺼내는 뉴런 수(뉴런 하나 = gate 행 + up 행 + down 열).</p>
  </section>

  <section class="step" id="b1">
    <div class="txt">
      <div class="no">묶기 1 · GPU의 계산 단위</div>
      <h2>GPU의 텐서 코어는 행렬곱을 고정 크기 조각으로 잘라 계산한다</h2>
      <p class="key">조각을 다 못 채워도 조각 하나를 통째로 계산한다. 빈자리도 시간이 든다.</p>
      <p class="why">그래서 토큰이나 뉴런이 적은 블록은 손해다. 블록 둘을 합쳐 빈자리를 채우면 칸이 준다.</p>
      <p class="note">실제 조각은 16 × 16이다. 아래 예시는 숫자를 줄여 4 × 4로 센다. 결과의 연산량은 모두 16 단위로 올려 센 값이다.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between"><h3 style="font-size:14px">토큰 5개 × 뉴런 2개를 4 × 4 조각으로</h3><span class="lbl">필요한 계산 10칸 → GPU가 하는 계산 32칸</span></div>
      <div class="bt-tile" id="bt-tile1"></div>
      <div class="legend"><span><i style="background:var(--on)"></i>필요한 계산</span><span><i style="background:var(--off);box-shadow:inset 0 0 0 1px var(--line-2)"></i>빈칸 (계산은 하지만 쓰지 않음)</span></div>
      <p style="font-size:12.5px;color:var(--fg-2)">토큰 5개는 4의 배수인 8로, 뉴런 2개는 4로 올라간다. 조각 두 개 = 8 × 4 = 32칸. 실제 GPU(16 × 16)에서는 토큰 5개도 16개의 시간이 들고, 토큰 17개면 32개의 시간이 든다.</p>
    </div>
  </section>

  <section class="step" id="b2">
    <div class="txt">
      <div class="no">묶기 2 · 예시 한 걸음</div>
      <h2>뉴런마다 그 뉴런을 쓰는 토큰을 모아 블록을 만들고, 칸이 줄면 합친다</h2>
      <p class="key">블록마다 그 뉴런의 가중치를 한 번만 읽고, 그 뉴런을 쓰는 토큰에만 계산한다.</p>
      <ul>
        <li>서로 다른 요청의 토큰 8개(T1 ~ T8), 층의 뉴런 12개.</li>
        <li>토큰마다 층 입력과 가장 가까운 중심점이 정해지고, 그 중심점의 뉴런 목록이 그 토큰이 켤 뉴런이다.</li>
        <li>GPU 칸은 4 × 4 조각으로 센다 (실제는 16 × 16).</li>
      </ul>
      <p class="why">합친 블록에서 토큰은 자기 목록에 없는 뉴런에 0을 곱한다. 그래서 출력은 토큰마다 자기 뉴런만 계산한 것과 똑같고, 정확도는 그대로다.</p>
    </div>
    <div class="vis">
      <div class="bt-steps"><button class="btn" id="bt-prev" type="button">이전</button><button class="btn primary" id="bt-next" type="button">다음</button><span class="where" id="bt-where"></span></div>
      <div class="bt-mat" id="bt-mat"></div>
      <div class="legend"><span><i style="background:var(--on)"></i>쓰는 칸 (토큰이 그 뉴런을 쓴다)</span><span><i style="background:var(--pick)"></i>0을 곱하는 칸 (블록 안이지만 토큰이 안 쓴다)</span><span><i style="background:var(--off)"></i>계산하지 않음</span></div>
      <div class="bt-side" id="bt-side"></div>
    </div>
  </section>

  <section class="step" id="b3">
    <div class="txt">
      <div class="no">묶기 3 · 실제 대기열</div>
      <h2>실제: 시험 사진 256장을 한 번에, 층마다의 블록</h2>
      <p class="key">층 1 ~ 3은 이번 걸음에 쓰이는 뉴런이 4,096개 가운데 4,078 ~ 4,093개로 거의 전부다. 그래도 뉴런마다 한 번씩만 읽는다.</p>
      <ul>
        <li>줄 하나 = 토큰 하나(시험 사진 256장). 간 중심점이 비슷한 토큰끼리 이웃하게 놓았다.</li>
        <li>칸 하나 = 읽는 뉴런 하나. 블록 순서로 놓았고, 왼쪽일수록 토큰이 많은 블록이다.</li>
        <li>블록 띠마다 청록은 그 토큰이 쓰는 칸, 호박색은 0을 곱하는 칸, 어두운 곳은 그 블록이 계산하지 않는 토큰이다.</li>
      </ul>
      <p class="why">층마다 연산량이 크게 다르다. 토큰마다 뉴런의 21 ~ 22%를 켜는 층 1, 2는 블록 안 토큰들의 목록이 서로 달라 0을 곱하는 칸이 많다. 1.1%를 켜는 층 4는 연산이 아주 적다.</p>
      <p class="note">그림에 마우스를 올리면 그 블록의 뉴런 수, 토큰 수, GPU 칸이 보인다.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between;gap:12px"><div class="chips" id="bt-layers"></div><span class="lbl" id="bt-rcap"></span></div>
      <div class="bt-real"><canvas id="bt-cv"></canvas></div>
      <div class="legend"><span><i style="background:var(--on)"></i>쓰는 칸</span><span><i style="background:var(--pick)"></i>0을 곱하는 칸</span><span><i style="background:var(--off)"></i>이 블록이 계산하지 않는 토큰</span></div>
      <div class="split">
        <div class="panel"><h3>읽는 양</h3><dl class="kv" id="bt-kv-r"></dl></div>
        <div class="panel"><h3>연산량 (16 단위 칸)</h3><dl class="kv" id="bt-kv-c"></dl><div class="bt-bar" id="bt-bar"></div><div class="legend" id="bt-bar-lg"></div></div>
      </div>
    </div>
  </section>

  <section class="step" id="b4">
    <div class="txt">
      <div class="no">묶기 4 · 결과</div>
      <h2>결과: 토큰 256개를 한 번에 계산할 때, 기본 풀 모델이 읽던 가중치의 5.1%만 읽고, 하던 연산의 51.6%만 한다</h2>
      <p class="key">읽는 양은 가능한 최소다. 이번 걸음에 쓰이는 뉴런을 한 번씩만 읽으니, 이보다 적게 읽을 수는 없다.</p>
      <ul>
        <li>연산량 51.6% 가운데 실제로 필요한 계산(쓰는 칸)은 13.8%다. 나머지는 0을 곱하는 칸 36.1%와 빈칸 1.7%다.</li>
        <li>0을 곱하는 칸이 큰 까닭: 중심점마다 뉴런 목록이 절반 가까이 달라서, 한 블록에 모인 토큰들이 그 블록의 뉴런을 다 쓰지는 않는다.</li>
        <li>토큰이 많을수록 읽는 양의 몫이 준다 (100개 11.2% → 1,000개 1.3%).</li>
      </ul>

    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between"><h3 style="font-size:14px">기본 풀 모델의 몇 %인가 (층 4개 합)</h3><span class="lbl">토큰 수 = 한 번에 계산하는 요청 수</span></div>
      <div class="bt-res" id="bt-res"></div>
      <div class="legend"><span><i style="background:var(--on)"></i>연산: 쓰는 칸</span><span><i style="background:var(--pick)"></i>연산: 0을 곱하는 칸</span><span><i style="background:var(--line-2)"></i>연산: 빈칸</span><span><i style="background:var(--fg-2)"></i>읽는 양</span></div>
      <p style="font-size:12.5px;color:var(--fg-2)">평상시 서버가 한 걸음에 함께 계산하는 요청은 수십 ~ 200개 남짓이라, 100개와 256개가 실제에 가깝다.</p>
    </div>
    <details class="fold bt-wide"><summary>어떻게 셌나, 줄인 모델 대비</summary><div class="body">
      <p><b>기본 풀 모델</b> = 다 켠 베이스 모델을 들어온 순서대로 16개씩 계산. <b>줄인 모델</b> = 이 레시피의 최종 모델(13.73%)을 들어온 순서대로 16개씩, 묶음마다 목록의 합집합을 계산. 줄인 모델 대비는 묶는 법만의 몫이고, 기본 풀 모델 대비는 사후 학습과 묶는 법을 합친 몫이다.</p>
      <table class="bt-tbl" id="bt-tbl"></table>
      <p><b>연산량</b> = 블록마다 (토큰 수를 16의 배수로 올림) × (뉴런 수를 16의 배수로 올림)의 합. <b>읽는 양</b> = 블록마다 꺼내는 뉴런 수의 합 (뉴런마다 한 블록에만 있으니 이번 걸음에 쓰이는 뉴런 수). 층 4개의 합.</p>
      <p>대기열은 시험 사진에서 아무렇게나 뽑았다(씨앗 0). 100개는 20번, 256개는 10번, 1,000개는 5번 뽑아 평균했다.</p>
      <p><b>블록 짜는 법</b>: 쓰는 토큰이 똑같은 뉴런끼리 처음 블록을 만들고, 쓰는 토큰이 겹치는 정도로 나무를 만들어 칸이 가장 적게 자른 뒤, 블록 둘을 합쳐 칸이 줄면 합친다. 블록 짜기는 실제 LLM에서 그 층의 어텐션과 나란히 한다고 본다.</p>
      <p>다 켠 모델도 토큰 256개를 행렬곱 한 번으로 계산하면 가중치를 한 번만 읽는다. 그와 견주면 고른 방법이 읽는 뉴런은 층 뉴런의 약 82%다 (토큰 256개의 목록을 합치면 뉴런 대부분이 쓰인다).</p>
    </div></details>
  </section>
  </div>
  ''' + ME

POST = MS + r'''<script>
/* 묶어서 계산 (배칭) 보기: 기존 스크립트를 고치지 않고 따로 둔다 */
window.__BATCH__ = __DATA__;
(() => {
'use strict';
const B = window.__BATCH__;
const $ = (s, e = document) => e.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = 'http://www.w3.org/2000/svg';
const C = {};
const st = { step: 0, layer: 0, hover: -1 };
const nf = v => Number(v).toLocaleString('ko-KR');
const pc = (a, b, d = 1) => (100 * a / b).toFixed(d) + '%';
function el(t, a = {}, p) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
function tx(p, x, y, s, a = {}) { const t = el('text', Object.assign({ x, y }, a), p); t.textContent = s; return t; }
function rgb(h) { const n = parseInt(h.replace('#', ''), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
const u8 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const bit = (b, i) => (b[i >> 3] >> (7 - (i & 7))) & 1;

// ---- 묶기 1: 조각 그림 ----
function drawTile1() {
  const host = $('#bt-tile1'); host.innerHTML = '';
  const W = Math.max(260, host.clientWidth || 360), cs = Math.min(34, (W - 70) / 5), H = 8 * cs + 30;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '토큰 5개 x 뉴런 2개를 4 x 4 조각 두 개로 계산' }, host);
  const x0 = 64, y0 = 22, names = ['2', '3', '', ''];
  tx(svg, x0 - 8, y0 - 7, '뉴런', { 'text-anchor': 'end' });
  names.forEach((n, j) => { if (n) tx(svg, x0 + j * cs + cs / 2, y0 - 7, n, { 'text-anchor': 'middle', class: 'lab' }); });
  for (let r = 0; r < 8; r++) {
    tx(svg, x0 - 8, y0 + r * cs + cs / 2 + 4, r < 5 ? 'T' + (r + 1) : '(빈칸)', { 'text-anchor': 'end', class: r < 5 ? 'lab' : '' });
    for (let c = 0; c < 4; c++) {
      const use = r < 5 && c < 2;
      el('rect', { x: x0 + c * cs + 2, y: y0 + r * cs + 2, width: cs - 4, height: cs - 4, rx: 3, fill: use ? C.on : C.off, stroke: use ? 'none' : C.line2, 'stroke-width': 1 }, svg);
    }
  }
  [0, 4].forEach((r, i) => { el('rect', { x: x0, y: y0 + r * cs, width: 4 * cs, height: 4 * cs, fill: 'none', stroke: C.pick, 'stroke-width': 1.5, rx: 4 }, svg); tx(svg, x0 + 4 * cs + 8, y0 + r * cs + 2 * cs + 4, i ? '둘째 조각' : '첫 조각 (4 × 4)', { class: 'lab' }); });
}

// ---- 묶기 2: 예시 한 걸음 ----
const TOK = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'], CEN = ['가', '가', '가', '나', '나', '다', '다', '다'];
const LIST = { '가': [1, 2, 3, 4, 5], '나': [1, 2, 3, 6, 7], '다': [1, 8, 9] };
const uses = (r, n) => LIST[CEN[r]].includes(n);
const RG = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const STEPS = [
  { title: '토큰 8개가 들어온다', order: RG(1, 12), blocks: [] },
  { title: '① 뉴런마다 그 뉴런을 쓰는 토큰을 모아 블록을 만든다', order: RG(1, 12), blocks: [
    { name: '㉠', cols: [1], rows: [0, 7] }, { name: '㉡', cols: [2, 3], rows: [0, 4] }, { name: '㉢', cols: [4, 5], rows: [0, 2] },
    { name: '㉣', cols: [6, 7], rows: [3, 4] }, { name: '㉤', cols: [8, 9], rows: [5, 7] }] },
  { title: '② 합쳐서 칸이 줄면 합친다', order: [1, 6, 7, 2, 3, 4, 5, 8, 9, 10, 11, 12], blocks: [
    { name: 'A', cols: [1, 6, 7], rows: [0, 7] }, { name: 'B', cols: [2, 3, 4, 5], rows: [0, 4] }, { name: 'C', cols: [8, 9], rows: [5, 7] }] },
  { title: '③ GPU가 블록마다 계산하고, 토큰마다 출력을 더한다', order: [1, 6, 7, 2, 3, 4, 5, 8, 9, 10, 11, 12], focus: 'A', blocks: [
    { name: 'A', cols: [1, 6, 7], rows: [0, 7] }, { name: 'B', cols: [2, 3, 4, 5], rows: [0, 4] }, { name: 'C', cols: [8, 9], rows: [5, 7] }] },
];
function drawMat() {
  const S = STEPS[st.step], host = $('#bt-mat'); host.innerHTML = '';
  const W = Math.max(300, host.clientWidth || 520), lx = 62, gap = S.blocks.length ? 6 : 0;
  const groups = S.order.map(n => S.blocks.findIndex(b => b.cols.includes(n)));
  const ngap = groups.reduce((a, g, i) => a + (i && g !== groups[i - 1] ? 1 : 0), 0);
  const cs = Math.min(34, (W - lx - 6 - ngap * gap) / 12), top = 42, H = top + 8 * cs + 8;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '토큰 8개 x 뉴런 12개: ' + S.title }, host);
  const xs = []; let x = lx;
  S.order.forEach((n, i) => { if (i && groups[i] !== groups[i - 1]) x += gap; xs.push(x); x += cs; });
  S.order.forEach((n, i) => tx(svg, xs[i] + cs / 2, top - 6, n, { 'text-anchor': 'middle' }));
  for (let r = 0; r < 8; r++) tx(svg, lx - 8, top + r * cs + cs / 2 + 4, `${TOK[r]} ${CEN[r]}`, { 'text-anchor': 'end', class: 'lab' });
  for (let r = 0; r < 8; r++) S.order.forEach((n, i) => {
    const b = S.blocks.find(q => q.cols.includes(n) && r >= q.rows[0] && r <= q.rows[1]);
    const u = uses(r, n), dim = S.focus && b && b.name !== S.focus;
    let f = C.off, o = 1;
    if (!S.blocks.length) f = u ? C.on : C.off;
    else if (b) f = u ? C.on : C.pick;
    else if (groups[i] < 0) o = 0.45;
    if (dim) o = 0.3;
    el('rect', { x: xs[i] + 2, y: top + r * cs + 2, width: cs - 4, height: cs - 4, rx: 3, fill: f, opacity: o }, svg);
    if (b && !u) tx(svg, xs[i] + cs / 2, top + r * cs + cs / 2 + 4, '×0', { 'text-anchor': 'middle', fill: C.bg, opacity: dim ? 0.4 : 1, 'font-size': Math.max(9, cs * 0.32) });
  });
  S.blocks.forEach(b => {
    const i0 = S.order.indexOf(b.cols[0]), i1 = S.order.indexOf(b.cols[b.cols.length - 1]);
    const foc = !S.focus || b.name === S.focus;
    el('rect', { x: xs[i0], y: top + b.rows[0] * cs, width: xs[i1] + cs - xs[i0], height: (b.rows[1] - b.rows[0] + 1) * cs, fill: 'none', stroke: foc ? C.fg : C.fg3, 'stroke-width': foc ? 1.8 : 1, rx: 5, opacity: foc ? 1 : 0.5 }, svg);
    tx(svg, (xs[i0] + xs[i1] + cs) / 2, 18, b.name, { 'text-anchor': 'middle', fill: foc ? C.pick : C.fg3, style: 'font-family: var(--sans); font-size: 19px; font-weight: 600' });
  });
  if (S.blocks.length) { const i0 = S.order.indexOf(10); tx(svg, xs[i0] + 1.5 * cs, 13, '안 읽음', { 'text-anchor': 'middle' }); }
  $('#bt-where').innerHTML = `<b>${S.title}</b> <span class="mono" style="color:var(--fg-3)">(${st.step + 1} / ${STEPS.length})</span>`;
  drawSide();
}
function drawSide() {
  const s = st.step, side = $('#bt-side');
  if (s === 0) side.innerHTML = `<p>토큰마다 켤 뉴런 (중심점의 목록):</p><table class="bt-tbl"><tr><th>토큰</th><th>가장 가까운 중심점</th><th>켤 뉴런</th></tr><tr><td>T1, T2, T3</td><td>가</td><td class="num">1, 2, 3, 4, 5</td></tr><tr><td>T4, T5</td><td>나</td><td class="num">1, 2, 3, 6, 7</td></tr><tr><td>T6, T7, T8</td><td>다</td><td class="num">1, 8, 9</td></tr></table><p>뉴런 10 ~ 12는 이번 걸음에 아무 토큰도 쓰지 않는다. <b>그래서 읽지 않는다.</b></p>`;
  else if (s === 1) side.innerHTML = `<p><b>쓰는 토큰이 똑같은 뉴런끼리 한 블록</b>이다. 블록마다 GPU 칸 = (토큰 수를 4의 배수로) × (뉴런 수를 4의 배수로).</p><table class="bt-tbl"><tr><th>블록</th><th>뉴런</th><th>쓰는 토큰</th><th>GPU 칸</th></tr><tr><td>㉠</td><td class="num">1</td><td>T1 ~ T8 (8개)</td><td class="num">8 × <b>4</b> = 32</td></tr><tr><td>㉡</td><td class="num">2, 3</td><td>T1 ~ T5 (5개)</td><td class="num"><b>8</b> × <b>4</b> = 32</td></tr><tr><td>㉢</td><td class="num">4, 5</td><td>T1 ~ T3 (3개)</td><td class="num"><b>4</b> × <b>4</b> = 16</td></tr><tr><td>㉣</td><td class="num">6, 7</td><td>T4, T5 (2개)</td><td class="num"><b>4</b> × <b>4</b> = 16</td></tr><tr><td>㉤</td><td class="num">8, 9</td><td>T6 ~ T8 (3개)</td><td class="num"><b>4</b> × <b>4</b> = 16</td></tr><tr class="sum"><td>합</td><td colspan="2">읽는 뉴런 9개, 각각 한 번</td><td class="num">112칸</td></tr></table><p>블록이 잘아서 빈칸이 많다. 실제로 필요한 계산은 34칸뿐이다.</p>`;
  else if (s === 2) side.innerHTML = `<p>블록 둘을 합친 칸이 따로 둔 칸의 합보다 적으면 합친다. 블록끼리 뉴런이 겹치지 않아 <b>읽는 양은 그대로</b>다.</p><table class="bt-tbl"><tr><th>합치는 블록</th><th>따로</th><th>합치면</th><th></th></tr><tr><td>㉠ + ㉣</td><td class="num">32 + 16 = 48</td><td class="num">8 × 4 = 32</td><td class="ok">합친다</td></tr><tr><td>㉡ + ㉢</td><td class="num">32 + 16 = 48</td><td class="num">8 × 4 = 32</td><td class="ok">합친다</td></tr><tr><td>(㉠㉣) + ㉤</td><td class="num">32 + 16 = 48</td><td class="num">8 × 8 = 64</td><td class="no-ok">안 합친다</td></tr></table><p>㉠ + ㉣: 뉴런 1개와 2개는 각각 4칸을 썼는데, 합쳐 3개여도 4칸이다. ㉣의 토큰(T4, T5)은 ㉠의 토큰 안에 이미 있어 토큰 줄도 늘지 않는다. 대신 T1 ~ T3, T6 ~ T8은 뉴런 6, 7에 0을 곱한다.</p><table class="bt-tbl"><tr><th>최종 블록</th><th>뉴런</th><th>토큰</th><th>GPU 칸</th></tr><tr><td>A</td><td class="num">1, 6, 7</td><td>T1 ~ T8</td><td class="num">32</td></tr><tr><td>B</td><td class="num">2, 3, 4, 5</td><td>T1 ~ T5</td><td class="num">32</td></tr><tr><td>C</td><td class="num">8, 9</td><td>T6 ~ T8</td><td class="num">16</td></tr><tr class="sum"><td>합</td><td colspan="2">읽는 뉴런 9개, 각각 한 번</td><td class="num">80칸 (합치기 전 112)</td></tr></table>`;
  else {
    side.innerHTML = `<p><b>블록 A를 GPU가 계산하는 모습</b> (4 × 4 조각 두 개):</p><div class="bt-tile" id="bt-tileA"></div><p>① 뉴런 1, 6, 7의 가중치(gate 행, up 행, down 열)를 <b>한 번</b> 꺼낸다. ② T1 ~ T8의 층 입력을 모아 한꺼번에 곱한다. ③ 자기 목록에 없는 뉴런에는 0을 곱한다(×0). ④ 결과를 토큰마다 출력에 더한다.</p><table class="bt-tbl"><tr><th>토큰</th><th>들어간 블록</th><th>계산된 뉴런 (×0 뺌)</th><th>자기 목록과</th></tr><tr><td>T1 (가)</td><td>A + B</td><td class="num">1 + 2, 3, 4, 5</td><td class="ok">같다</td></tr><tr><td>T4 (나)</td><td>A + B</td><td class="num">1, 6, 7 + 2, 3</td><td class="ok">같다</td></tr><tr><td>T6 (다)</td><td>A + C</td><td class="num">1 + 8, 9</td><td class="ok">같다</td></tr></table><p>그래서 출력은 토큰마다 자기 뉴런만 계산한 것과 똑같다.</p>`;
    drawTileA();
  }
}
function drawTileA() {
  const host = $('#bt-tileA'); if (!host) return; host.innerHTML = '';
  const W = Math.max(240, host.clientWidth || 320), cs = Math.min(26, (W - 90) / 4), H = 8 * cs + 26;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '블록 A: 토큰 8개 x 뉴런 3개 + 빈칸 1열' }, host);
  const x0 = 62, y0 = 20, cols = [1, 6, 7, 0];
  tx(svg, x0 - 8, y0 - 6, '뉴런', { 'text-anchor': 'end' });
  cols.forEach((n, j) => tx(svg, x0 + j * cs + cs / 2, y0 - 6, n ? String(n) : '빈', { 'text-anchor': 'middle', class: n ? 'lab' : '' }));
  for (let r = 0; r < 8; r++) {
    tx(svg, x0 - 8, y0 + r * cs + cs / 2 + 4, `${TOK[r]} ${CEN[r]}`, { 'text-anchor': 'end', class: 'lab' });
    cols.forEach((n, j) => {
      const u = n && uses(r, n);
      el('rect', { x: x0 + j * cs + 2, y: y0 + r * cs + 2, width: cs - 4, height: cs - 4, rx: 3, fill: !n ? C.off : u ? C.on : C.pick, stroke: !n ? C.line2 : 'none' }, svg);
      if (n && !u) tx(svg, x0 + j * cs + cs / 2, y0 + r * cs + cs / 2 + 4, '×0', { 'text-anchor': 'middle', fill: C.bg, 'font-size': 9 });
    });
  }
  [0, 4].forEach(r => el('rect', { x: x0, y: y0 + r * cs, width: 4 * cs, height: 4 * cs, fill: 'none', stroke: C.pick, 'stroke-width': 1.3, rx: 4 }, svg));
  tx(svg, x0 + 4 * cs + 8, y0 + 4 * cs + 4, '쓰는 12 · ×0 12 · 빈칸 8 = 32칸', { class: 'lab' });
}

// ---- 묶기 3: 실제 대기열 ----
const IMG = {};
function layerImage(l) {
  if (IMG[l]) return IMG[l];
  const L = B.real[l], N = B.N, W = L.ncols, cv = document.createElement('canvas');
  cv.width = W; cv.height = N;
  const x = cv.getContext('2d'), im = x.createImageData(W, N), on = rgb(C.on), pk = rgb(C.pick), off = rgb(C.off);
  const offs = []; let c0 = 0;
  L.blocks.forEach(b => {
    const rm = u8(b.r), ub = u8(b.u); let k = 0;
    for (let r = 0; r < N; r++) {
      const inb = bit(rm, r);
      for (let j = 0; j < b.n; j++) {
        let c = off;
        if (inb) c = bit(ub, k * b.n + j) ? on : pk;
        const p = (r * W + c0 + j) * 4; im.data[p] = c[0]; im.data[p + 1] = c[1]; im.data[p + 2] = c[2]; im.data[p + 3] = 255;
      }
      if (inb) k++;
    }
    offs.push(c0); c0 += b.n;
  });
  x.putImageData(im, 0, 0);
  IMG[l] = { cv, offs };
  return IMG[l];
}
function drawReal() {
  const L = B.real[st.layer], cv = $('#bt-cv'), img = layerImage(st.layer);
  const d = window.devicePixelRatio || 1, w = Math.max(280, cv.clientWidth), h = Math.round(Math.min(340, Math.max(200, w * 0.42)));
  cv.style.height = h + 'px'; cv.width = Math.round(w * d); cv.height = Math.round(h * d);
  const x = cv.getContext('2d');
  x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high';
  x.drawImage(img.cv, 0, 0, cv.width, cv.height);
  if (st.hover >= 0) {
    const b = L.blocks[st.hover], x0 = img.offs[st.hover] / L.ncols * cv.width, x1 = (img.offs[st.hover] + b.n) / L.ncols * cv.width;
    x.strokeStyle = C.fg; x.lineWidth = 1.5 * d; x.strokeRect(x0, 1, Math.max(2 * d, x1 - x0), cv.height - 2);
  }
  $('#bt-rcap').textContent = `층 ${st.layer + 1} · 토큰 ${B.N}개 × 읽는 뉴런 ${nf(L.ncols)}개 · 블록 ${L.nblocks}개`;
}
function drawRealStats() {
  const L = B.real[st.layer];
  $('#bt-kv-r').innerHTML = `<dt>읽는 뉴런</dt><dd>${nf(L.reads)}개 / 4,096</dd><dt>기본 풀 모델</dt><dd>${nf(L.full_reads)} (4,096 × 16번)</dd><dt>기본 풀 모델의</dt><dd style="color:var(--on)">${pc(L.reads, L.full_reads)}</dd>`;
  const pad = L.compute - L.dense, zero = L.dense - L.ones;
  $('#bt-kv-c').innerHTML = `<dt>고른 방법</dt><dd>${nf(L.compute)}칸</dd><dt>기본 풀 모델</dt><dd>${nf(L.full_compute)}칸</dd><dt>기본 풀 모델의</dt><dd style="color:var(--on)">${pc(L.compute, L.full_compute)}</dd>`;
  const parts = [[L.ones, C.on, '쓰는 칸'], [zero, C.pick, '0을 곱하는 칸'], [pad, C.line2, '빈칸']];
  $('#bt-bar').innerHTML = parts.map(p => `<i style="width:${100 * p[0] / L.compute}%;background:${p[1]}"></i>`).join('');
  $('#bt-bar-lg').innerHTML = parts.map(p => `<span><i style="background:${p[1]}"></i>${p[2]} ${pc(p[0], L.full_compute)}</span>`).join('');
}
function realChips() {
  const host = $('#bt-layers'); host.innerHTML = '';
  for (let l = 0; l < 4; l++) { const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain'; b.textContent = '층 ' + (l + 1); b.setAttribute('aria-pressed', st.layer === l); b.onclick = () => { st.layer = l; st.hover = -1; realChips(); drawReal(); drawRealStats(); }; host.appendChild(b); }
}
function realHover(ev) {
  const L = B.real[st.layer], cv = $('#bt-cv'), r = cv.getBoundingClientRect(), img = layerImage(st.layer), tip = $('#tip');
  const col = (ev.clientX - r.left) / r.width * L.ncols;
  let k = 0; while (k + 1 < img.offs.length && img.offs[k + 1] <= col) k++;
  if (k !== st.hover) { st.hover = k; drawReal(); }
  const b = L.blocks[k];
  tip.hidden = false;
  tip.textContent = `블록 ${k + 1}: 뉴런 ${b.n}개 × 토큰 ${b.m}개 · GPU 칸 ${nf(b.cells)} · 쓰는 칸 ${pc(b.ones, b.cells, 0)}`;
  tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, ev.clientX + 12) + 'px'; tip.style.top = (ev.clientY - 34) + 'px';
}
function realLeave() { $('#tip').hidden = true; st.hover = -1; drawReal(); }

// ---- 묶기 4: 결과 ----
function drawRes() {
  const host = $('#bt-res'); host.innerHTML = '';
  const W = Math.max(300, host.clientWidth || 520), lx = 96, rx = 24, bh = 14, gh = 54, top = 30, ns = ['100', '256', '1000'], H = top + ns.length * gh + 20;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '대기열 크기마다 연산량과 읽는 양 (기본 풀 모델 = 100%)' }, host);
  const sx = v => lx + v / 100 * (W - lx - rx);
  for (let v = 0; v <= 100; v += 10) { el('line', { x1: sx(v), x2: sx(v), y1: top - 8, y2: H - 20, stroke: C.line, 'stroke-width': 1 }, svg); if (v % 20 === 0) tx(svg, sx(v), H - 6, v + '%', { 'text-anchor': 'middle' }); }
  el('line', { x1: sx(100), x2: sx(100), y1: top - 14, y2: H - 20, stroke: C.fg2, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
  tx(svg, sx(100), top - 18, '기본 풀 모델 = 100%', { 'text-anchor': 'end', class: 'lab' });
  ns.forEach((n, i) => {
    const R = B.results[n], y = top + i * gh;
    tx(svg, lx - 10, y + bh + 4, `토큰 ${nf(+n)}개`, { 'text-anchor': 'end', class: 'lab' });
    let a = 0;
    [[R.ones, C.on], [R.zero, C.pick], [R.pad, C.line2]].forEach(p => { el('rect', { x: sx(a), y, width: Math.max(0, sx(a + p[0]) - sx(a) - 1.5), height: bh, rx: 2, fill: p[1] }, svg); a += p[0]; });
    tx(svg, sx(R.compute) + 6, y + bh - 3, '연산 ' + R.compute.toFixed(1) + '%', { class: 'lab' });
    el('rect', { x: sx(0), y: y + bh + 5, width: Math.max(2, sx(R.reads) - sx(0)), height: bh, rx: 2, fill: C.fg2 }, svg);
    tx(svg, sx(R.reads) + 6, y + 2 * bh + 2, '읽기 ' + R.reads.toFixed(1) + '%', { class: 'lab' });
  });
  $('#bt-tbl').innerHTML = '<tr><th>토큰 수</th><th>기본 풀 모델 대비 연산 / 읽기</th><th>줄인 모델 대비 연산 / 읽기</th><th>연산 = 쓰는 + 0곱 + 빈칸</th></tr>' +
    ns.map(n => { const R = B.results[n]; return `<tr><td>${nf(+n)}개</td><td class="num">${R.compute.toFixed(1)}% / ${R.reads.toFixed(1)}%</td><td class="num">${R.compute_seq.toFixed(1)}% / ${R.reads_seq.toFixed(1)}%</td><td class="num">${R.ones.toFixed(1)} + ${R.zero.toFixed(1)} + ${R.pad.toFixed(1)}</td></tr>`; }).join('');
}

// ---- 보기 전환 (기존 버튼에 동작을 덧붙인다) ----
let ready = false;
function drawAll() { drawTile1(); drawMat(); realChips(); drawReal(); drawRealStats(); drawRes(); }
function show() {
  ['tab-train', 'tab-infer'].forEach(id => $('#' + id).setAttribute('aria-pressed', 'false'));
  $('#tab-batch').setAttribute('aria-pressed', 'true');
  $('#view-train').hidden = true; $('#view-infer').hidden = true; $('#view-batch').hidden = false;
  drawAll();
  try { history.replaceState(null, '', '#batch'); } catch (e) { location.hash = 'batch'; }
}
function hide() { $('#tab-batch').setAttribute('aria-pressed', 'false'); $('#view-batch').hidden = true; }
function init() {
  ['--bg', '--bg-2', '--off', '--on', '--pick', '--fg', '--fg-2', '--fg-3', '--line', '--line-2'].forEach(n => C[n.slice(2).replace('-', '')] = css(n));
  $('#tab-batch').addEventListener('click', show);
  $('#tab-train').addEventListener('click', hide); $('#tab-infer').addEventListener('click', hide);
  $('#bt-prev').onclick = () => { st.step = Math.max(0, st.step - 1); drawMat(); };
  $('#bt-next').onclick = () => { st.step = Math.min(STEPS.length - 1, st.step + 1); drawMat(); };
  const cv = $('#bt-cv'); cv.addEventListener('pointermove', realHover); cv.addEventListener('pointerleave', realLeave);
  let rt; new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (!$('#view-batch').hidden) drawAll(); }, 160); }).observe(document.body);
  ready = true;
  if (window.__HASH0__ === '#batch') show();
}
(function wait(t0) {                                            // 기존 스크립트가 데이터를 다 불러온 뒤에 시작한다
  const ld = $('#loading');
  if (ld && ld.hidden) init(); else if (performance.now() - t0 < 15000) requestAnimationFrame(() => wait(t0)); else init();
})(performance.now());
})();
</script>''' + ME


def strip(s):
    return re.sub(re.escape(MS) + '.*?' + re.escape(ME), '', s, flags=re.S)


# 아키텍처 이름 (사용자, 2026-11-22): 기존 내용 가운데 바꾸는 곳은 이 둘뿐이다. 되돌리면 원래와 같다.
NAME = [('<title>뉴런 고르기 레시피</title>', '<title>Roster · 뉴런 고르기 레시피</title>'),
        ('<div class="brand"><b>뉴런 고르기 레시피</b><span>FFN · Fashion-MNIST · 층 4개 × 뉴런 4,096개</span></div>',
         '<div class="brand"><b>Roster</b><span>뉴런 고르기 레시피 · FFN · Fashion-MNIST · 층 4개 × 뉴런 4,096개</span></div>')]


def unname(s):
    for o, n in NAME:
        s = s.replace(n, o)
    return s


def main():
    raw = open(PAGE, 'rb').read()
    crlf = b'\r\n' in raw
    page = raw.decode('utf-8').replace('\r\n', '\n')
    base = unname(strip(page))                                     # 이미 넣은 적이 있으면 빼고(이름도 되돌리고) 다시 넣는다
    data = json.load(open(os.path.join(HERE, 'runs', 'viz_batch.json'), encoding='utf-8'))
    post = POST.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    a1 = '<button id="tab-infer" type="button" aria-pressed="false">추론 과정</button>'
    a2 = '<script>\n(() => {'
    a3 = '  <footer>'
    for a in (a1, a2, a3):
        assert base.count(a) == 1, a
    assert base.rstrip('\n').endswith('</script>')
    out = base.replace(a1, a1 + TAB).replace(a2, PRE + a2).replace(a3, VIEW + a3)
    tail = len(base) - len(base.rstrip('\n'))
    out = out[:len(out) - tail] + post + out[len(out) - tail:] if tail else out + post
    for o, n in NAME:
        assert out.count(o) == 1, o
        out = out.replace(o, n)
    assert unname(strip(out)) == base, '빼고 이름을 되돌리면 원래와 달라진다'
    enc = out.replace('\n', '\r\n') if crlf else out
    open(PAGE, 'wb').write(enc.encode('utf-8'))
    print('neuron_recipe.html: %.1f KB -> %.1f KB' % (len(raw) / 1024, len(enc.encode('utf-8')) / 1024))
    subprocess.run([sys.executable, BUILD], check=True)


if __name__ == '__main__':
    main()
