"""Interactive 60fps Canvas Engine + KO/EN i18n + 5-Scene Storyboard for ROSTER_SHOWCASE.html."""

SHOWCASE_JS = r"""
(() => {
'use strict';

const D = window.__SHOWCASE_DATA__;
if (!D) { console.error('Missing __SHOWCASE_DATA__'); return; }

// ======================== I18N DICTIONARY (KO / EN) ========================
const I18N = {
  ko: {
    brandSub: '희소 FFN 시각 쇼케이스',
    modeCinema: '시네마',
    modeScroll: '스크롤',
    detailBtn: '상세 분석',
    play: '재생',
    pause: '일시정지',
    replay: '다시 쏘기',
    fullscreen: '전체화면',
    chapters: [
      { id: 's1', num: '01', short: '단일 점화', title: '전체 평균 86.5%는 잠든 채, 6.27배 빠르게' },
      { id: 's2', num: '02', short: '배치의 벽', title: '각각은 희소한데, 뭉치면 열이 100% 포화된다' },
      { id: 's3', num: '03', short: 'CH2 사후학습', title: '공유 사후학습(CH2)으로 흩어진 활성을 32열 벽으로 재편' },
      { id: 's4', num: '04', short: '비격자 타일', title: '격자를 깬 5단계 틀 + Attention 뒤 0.00ms 배치' },
      { id: 's5', num: '05', short: '실리콘 레이스', title: 'RTX 4060 실측 속도 — 단일 6.27배 · 대형 배치 4.57배' },
    ],
    // Scene 1
    s1_hud1: '이 입력 활성 (전체 13.5%)',
    s1_hud2: '단일 요청 실측',
    s1_hud3: '시험 정확도',
    s1_hud4: '예측 결과',
    s1_dense: '일반 FFN (Dense)',
    s1_denseSub: '뉴런 4,096개 전부 점화 · 1.00×',
    s1_roster: 'Roster FFN (CH2)',
    s1_rosterSub: '중심점 룩업으로 앞 K개 인덱스만 선택 점화',
    s1_arrived: '도달!',
    s1_pick: '입력 선택',
    s1_prism: '중심점',
    s1_in: '입력',
    s1_out: '출력',
    dim4096: 'd=4,096 (LLM 폭)',
    dim1024: 'd=1,024 (CH2 실가중치)',
    // Scene 2
    s2_hud1: '토큰당 활성 (L1 Strip #2)',
    s2_hud2: '열 합집합 (Strip #2)',
    s2_hud3: '미정렬 16×16 격자 ×0',
    s2_hud4: '순진한 배치 가속',
    s2_m_rain: '토큰 유입 (합집합 포화)',
    s2_m_grid: '미정렬 16×16 격자 (×0 낭비)',
    s2_m_hover: '토큰 1줄 탐색',
    s2_unionBar: '누적 켜진 열(뉴런) 합집합 (L1 Strip #2)',
    s2_tokAxis: '512 토큰 (무작위 도착)',
    s2_neuAxis: '512 뉴런 (L1 Strip #2)',
    s2_hoverHint: '행 위에 마우스를 올려 개별 토큰의 희소 패턴 확인',
    s2_tokRow: '토큰',
    s2_collapsed: '1.00× (희소성 붕괴)',
    // Scene 3
    s3_hud1: '×0 낭비 비율 (B=2,048)',
    s3_hud2: 'P2 합집합 (B=2,048)',
    s3_hud3: 'DRAM 읽기 (4층 평균)',
    s3_hud4: '1* 누락',
    s3_stages: ['0 이전 · M=256 개별학습', '1 CH2 · M=32 통합', '2 CH2 · B1 가족정렬', '3 CH2 · B2 32열 블록'],
    s3_before: '이전 · M=256 개별학습',
    s3_after: '이후 · CH2 공유 사후학습',
    s3_tiles: '타일',
    s3_exact: '0% (배치 무관)',
    // Scene 4
    s4_hud1: '전체 타일 (B=2,048 · 4층)',
    s4_hud2: '64×32 대형틀 비중',
    s4_hud3: '타일 채움률 (B=2,048)',
    s4_hud4: '노출 배치 오버헤드',
    s4_all: '전체',
    s4_ngOnly: '비격자만',
    s4_replay: '타일 덮기 재생',
    s4_inspector: '선택 타일',
    s4_tier: '틀 크기',
    s4_fill: '채움률',
    s4_zeros: '×0 칸',
    s4_bridge: '비격자 자유 짝짓기',
    s4_yes: '예 (분리된 행/열 연결)',
    s4_no: '연속 블록',
    s4_hoverTile: '타일 클릭 또는 호버로 행·열 연결 확인',
    s4_stripCap: 'L1 Strip #2 (B=512 실측)',
    s4_pipeTitle: '병렬 파이프라인 (d=4,096 · B=2,048 · 1층 · Attention 겹침 기준, 비겹침은 05장면)',
    s4_gpuLane: 'GPU 스트림',
    s4_cpuLane: 'CPU AVX2',
    s4_attnLbl: 'Attention (13.30 ms)',
    s4_ffnLbl: 'Roster v4 FFN (2.09 ms)',
    s4_a3Lbl: 'A3V4 (1.81ms) + H2D (0.29ms)',
    s4_hidden: 'Attention 뒤 완전 은닉 → 0.00 ms 노출',
    // Scene 5
    s5_hud1: '단일 요청 (B=1)',
    s5_hud2: '배치 겹침 E2E',
    s5_hud3: '순수 직렬 E2E',
    s5_hud4: 'vs BSMR (10회)',
    s5_ovlp: 'Attention 겹침',
    s5_serial: '순수 직렬 (비겹침)',
    s5_t_base: 'Base cuBLAS (100% 밀집)',
    s5_t_mask0: '0-Mask 대조군 (0 곱하기만)',
    s5_t_trka: 'Track A (M=256 개별 + v4)',
    s5_t_ch2: 'Roster CH2 (M=32+B1/B2 + v4 + A3V4)',
    s5_finish: '완료',
    s5_tilesDown: '타일 -80% · ×0 -54.7%',
    s5_replay: '레이스 재시작',
    footerNote: '모든 점화·행렬·타일·속도는 RTX 4060 실측 CH2_M32_B1B2_r9 + A3V4 + v4 실제 데이터입니다.',
  },
  en: {
    brandSub: 'Sparse FFN Visual Architecture',
    modeCinema: 'Cinema',
    modeScroll: 'Scroll',
    detailBtn: 'Full Report',
    play: 'Play',
    pause: 'Pause',
    replay: 'Pulse Again',
    fullscreen: 'Fullscreen',
    chapters: [
      { id: 's1', num: '01', short: 'Ignition', title: '86.5% Avg Neurons Stay Dark — 6.27× Faster' },
      { id: 's2', num: '02', short: 'Batch Wall', title: 'Sparse Per Token, Yet Columns Saturate to 100%' },
      { id: 's3', num: '03', short: 'CH2 Co-Design', title: 'CH2 Shared Post-Training Reshapes Activations into 32-Col Walls' },
      { id: 's4', num: '04', short: '5-Tier Tile', title: 'Non-Grid 5-Tier Tiling + 0.00 ms Hidden Dispatch' },
      { id: 's5', num: '05', short: 'GPU Race', title: 'RTX 4060 Silicon Race — 6.27× Single · 4.57× Batched' },
    ],
    // Scene 1
    s1_hud1: 'This Input Active (Avg 13.5%)',
    s1_hud2: 'B=1 Measured Speedup',
    s1_hud3: 'Test Accuracy',
    s1_hud4: 'Prediction',
    s1_dense: 'Standard Dense FFN',
    s1_denseSub: 'All 4,096 neurons fire · 1.00×',
    s1_roster: 'Roster FFN (CH2)',
    s1_rosterSub: 'Centroid lookup selects top-K neuron indices',
    s1_arrived: 'DONE!',
    s1_pick: 'Select Input',
    s1_prism: 'Centroid',
    s1_in: 'Input',
    s1_out: 'Output',
    dim4096: 'd=4,096 (LLM-Width)',
    dim1024: 'd=1,024 (Real CH2)',
    // Scene 2
    s2_hud1: 'Per-Token Active (L1 Strip #2)',
    s2_hud2: 'Col Union (Strip #2)',
    s2_hud3: 'Unsorted 16×16 Grid ×0',
    s2_hud4: 'Naive Batch Speedup',
    s2_m_rain: 'Token Rain (Union Collapse)',
    s2_m_grid: 'Unsorted 16×16 Grid (×0 Waste)',
    s2_m_hover: 'Inspect 1 Token',
    s2_unionBar: 'Cumulative Active Column Union (L1 Strip #2)',
    s2_tokAxis: '512 Tokens (Random Arrival)',
    s2_neuAxis: '512 Neurons (L1 Strip #2)',
    s2_hoverHint: 'Hover any row to isolate one sparse token vs full union',
    s2_tokRow: 'Token',
    s2_collapsed: '1.00× (Sparsity Collapsed)',
    // Scene 3
    s3_hud1: '×0 Waste (B=2,048)',
    s3_hud2: 'P2 Union (B=2,048)',
    s3_hud3: 'DRAM Read (4-Layer Avg)',
    s3_hud4: '1* Drop',
    s3_stages: ['0 Before · M=256 Indiv', '1 CH2 · M=32 Merge', '2 CH2 · B1 Family', '3 CH2 · B2 32-Col Walls'],
    s3_before: 'Before · M=256 Individual',
    s3_after: 'After · CH2 Co-Design',
    s3_tiles: 'tiles',
    s3_exact: '0% (Exact)',
    // Scene 4
    s4_hud1: 'Total Tiles (B=2,048 · 4L)',
    s4_hud2: '64×32 Flagship Share',
    s4_hud3: 'Tile Fill Rate (B=2,048)',
    s4_hud4: 'Exposed Dispatch',
    s4_all: 'All',
    s4_ngOnly: 'Non-Grid Only',
    s4_replay: 'Replay Tiling',
    s4_inspector: 'Selected Tile',
    s4_tier: 'Tile Tier',
    s4_fill: 'Fill Rate',
    s4_zeros: '×0 Cells',
    s4_bridge: 'Non-Grid Re-Pairing',
    s4_yes: 'YES (Bridges Separate Bands)',
    s4_no: 'Contiguous Block',
    s4_hoverTile: 'Click or hover any tile to inspect row/col bridges',
    s4_stripCap: 'L1 Strip #2 (B=512)',
    s4_pipeTitle: 'Parallel Pipeline (d=4,096 · B=2,048 · 1L · Attention-Overlapped; Serial in Scene 05)',
    s4_gpuLane: 'GPU Stream',
    s4_cpuLane: 'CPU AVX2',
    s4_attnLbl: 'Attention (13.30 ms)',
    s4_ffnLbl: 'Roster v4 FFN (2.09 ms)',
    s4_a3Lbl: 'A3V4 (1.81ms) + H2D (0.29ms)',
    s4_hidden: 'Hidden Behind Attention → 0.00 ms Exposed',
    // Scene 5
    s5_hud1: 'Single Request (B=1)',
    s5_hud2: 'Overlap E2E',
    s5_hud3: 'Zero-Overlap Serial',
    s5_hud4: 'vs BSMR (10-trial)',
    s5_ovlp: 'Attention Overlap',
    s5_serial: 'Zero-Overlap Serial',
    s5_t_base: 'Base cuBLAS (100% Dense)',
    s5_t_mask0: '0-Mask Control (Zero-out only)',
    s5_t_trka: 'Track A (M=256 Indiv + v4)',
    s5_t_ch2: 'Roster CH2 (M=32+B1/B2 + v4 + A3V4)',
    s5_finish: 'DONE',
    s5_tilesDown: '-80% tiles · -54.7% ×0',
    s5_replay: 'Restart Race',
    footerNote: 'Every activation, matrix, tile coordinate, and ms timing is measured live from CH2_M32_B1B2_r9 + A3V4 + v4 on RTX 4060.',
  },
};

// ======================== UTILITIES ========================
const TIER_COLORS = {
  4: '#fbbf24', // 64x32 gold
  0: '#22d3ee', // 32x32 cyan
  1: '#a78bfa', // 32x16 violet
  2: '#34d399', // 16x16 emerald
  3: '#fb7185', // 8x16 coral
};
const TIER_ORDER = [4, 0, 1, 2, 3];
const TIER_NAMES = { 4: '64×32', 0: '32×32', 1: '32×16', 2: '16×16', 3: '8×16' };

function centroidColor(cid, alpha = 1) {
  const hue = (cid * 137.508 + 185) % 360;
  return `hsla(${hue.toFixed(0)}, 82%, 62%, ${alpha})`;
}

function easeInOutCubic(t) {
  t = Math.max(0, Math.min(1, t));
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a, b, t) { return a + (b - a) * t; }

function unpackBits512(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const out = new Uint8Array(512 * 512);
  let k = 0;
  for (let i = 0; i < bytes.length && k < out.length; i++) {
    const b = bytes[i];
    out[k++] = (b >> 7) & 1;
    out[k++] = (b >> 6) & 1;
    out[k++] = (b >> 5) & 1;
    out[k++] = (b >> 4) & 1;
    out[k++] = (b >> 3) & 1;
    out[k++] = (b >> 2) & 1;
    out[k++] = (b >> 1) & 1;
    out[k++] = b & 1;
  }
  return out;
}

function makeOffscreenMatrix(bits, rgbOn, bgAlpha = 0) {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 512;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(512, 512);
  const d = img.data;
  const [r, g, b] = rgbOn;
  for (let i = 0; i < 512 * 512; i++) {
    const p = i * 4;
    if (bits[i]) {
      d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 235;
    } else {
      d[p] = 8; d[p + 1] = 13; d[p + 2] = 26; d[p + 3] = bgAlpha;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Pre-decode real 512x512 binary matrices once (<15ms)
const MATS = {
  arrival: unpackBits512(D.batch_scene.mats.arrival),
  track_a: unpackBits512(D.batch_scene.mats.track_a),
  ch2: unpackBits512(D.batch_scene.mats.ch2),
};
const OFFSCREEN = {
  arrival: makeOffscreenMatrix(MATS.arrival, [34, 211, 238], 0),
  track_a: makeOffscreenMatrix(MATS.track_a, [251, 113, 133], 0),
  ch2: makeOffscreenMatrix(MATS.ch2, [52, 211, 153], 0),
  ch2Dim: makeOffscreenMatrix(MATS.ch2, [45, 212, 191], 0),
};

// Precompute cumulative column union array for Scene 2 (first row where each col lights up)
const COL_FIRST_ROW = new Int16Array(512).fill(999);
for (let r = 0; r < 512; r++) {
  const base = r * 512;
  for (let c = 0; c < 512; c++) {
    if (MATS.arrival[base + c] && COL_FIRST_ROW[c] === 999) COL_FIRST_ROW[c] = r;
  }
}

// Preload 8 Fashion-MNIST example images
const EX_IMGS = D.examples.map(ex => {
  const im = new Image();
  im.src = ex.img;
  return im;
});

// ======================== GLOBAL STATE ========================
const SCENE_DUR = 9.5; // seconds per chapter in Auto-Cinema mode
const STATE = {
  lang: 'ko',
  mode: 'cinema', // 'cinema' | 'scroll'
  scene: 0,       // 0..4
  playing: true,
  speed: 1.0,
  cinemaTime: 0,  // 0 .. 5 * SCENE_DUR
  // Scene 1
  exIdx: 0,
  s1Scale: '4096',
  s1Pulse: 0,     // 0..1 progress
  // Scene 2
  s2Mode: 'rain', // 'rain' | 'grid' | 'hover'
  s2Scan: 0,      // 0..512
  s2HoverRow: 120,
  // Scene 3
  s3Morph: 0,     // 0..3 continuous
  s3Target: null, // null when auto-flowing, 0..3 when user scrubbed
  s3UserLocked: false,
  // Scene 4
  s4Filter: 'all', // 'all' | 'ng' | 4 | 0 | 1 | 2 | 3
  s4Reveal: 0,     // 0..1
  s4Selected: -1,  // index into D.batch_scene.tiles
  s4Hover: -1,
  s4PipeT: 0,      // 0..1 repeating parallel timeline pulse
  // Scene 5
  s5Batch: 2048,   // 1 | 512 | 1024 | 2048
  s5Ovlp: true,
  s5Scale: '4096',
  s5RaceT: 0,      // 0..1.25 repeating race
};

// Pick a dramatic default non-grid 64x32 tile in Scene 4 to showcase right away!
const DEFAULT_NG_TILE = (() => {
  const ts = D.batch_scene.tiles;
  let best = ts.findIndex(t => t.t === 4 && t.ng_r && t.fill >= 78);
  if (best < 0) best = ts.findIndex(t => (t.ng_r || t.ng_c) && t.t === 4);
  if (best < 0) best = ts.findIndex(t => t.ng_r || t.ng_c);
  return Math.max(0, best);
})();
STATE.s4Selected = DEFAULT_NG_TILE;

// ======================== DOM HELPERS ========================
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const T = () => I18N[STATE.lang];

function setCanvasSize(cv) {
  if (!cv) return null;
  const rect = cv.getBoundingClientRect();
  const w = Math.max(200, Math.round(rect.width));
  const h = Math.max(180, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ======================== UI SYNC & LOCALIZATION ========================
function syncTexts() {
  const t = T();
  $('#brand-sub').textContent = t.brandSub;
  $('#btn-mode-cinema').textContent = t.modeCinema;
  $('#btn-mode-scroll').textContent = t.modeScroll;
  $('#btn-detail').textContent = t.detailBtn;
  $('#btn-lang').textContent = STATE.lang === 'ko' ? 'EN' : '한국어';
  $('#btn-play-lbl').textContent = STATE.playing ? t.pause : t.play;
  $('#footer-note').textContent = t.footerNote;

  // Chapter nav + cinema scrub labels
  t.chapters.forEach((ch, idx) => {
    const navBtn = $(`.nav-ch[data-ch="${idx}"]`);
    if (navBtn) navBtn.innerHTML = `<span class="ch-num">${ch.num}</span><span class="ch-txt">${ch.short}</span>`;
    const scrubLbl = $(`.scrub-seg[data-ch="${idx}"] .scrub-lbl`);
    if (scrubLbl) scrubLbl.textContent = `${ch.num} ${ch.short}`;
    const secTitle = $(`#sec-${ch.id} .sec-title`);
    if (secTitle) secTitle.textContent = ch.title;
    const secTag = $(`#sec-${ch.id} .sec-tag`);
    if (secTag) secTag.textContent = `${ch.num} · ${ch.short}`;
  });

  // Scene 1 labels & image chips
  $('#s1-hl1').textContent = t.s1_hud1;
  $('#s1-hl2').textContent = t.s1_hud2;
  $('#s1-hl3').textContent = t.s1_hud3;
  $('#s1-hl4').textContent = t.s1_hud4;
  $('#s1-replay').textContent = t.replay;
  $$('.s1-dim-btn[data-dim="4096"], .s5-dim-btn[data-dim="4096"]').forEach(b => b.textContent = t.dim4096);
  $$('.s1-dim-btn[data-dim="1024"], .s5-dim-btn[data-dim="1024"]').forEach(b => b.textContent = t.dim1024);
  $$('.ex-chip').forEach((el, idx) => {
    const ex = D.examples[idx];
    el.querySelector('.ex-nm').textContent = STATE.lang === 'ko' ? ex.ko : ex.en;
  });

  // Scene 2 labels
  $('#s2-hl1').textContent = t.s2_hud1;
  $('#s2-hl2').textContent = t.s2_hud2;
  $('#s2-hl3').textContent = t.s2_hud3;
  $('#s2-hl4').textContent = t.s2_hud4;
  $('#s2-btn-rain').textContent = t.s2_m_rain;
  $('#s2-btn-grid').textContent = t.s2_m_grid;
  $('#s2-btn-hover').textContent = t.s2_m_hover;

  // Scene 3 labels
  $('#s3-hl1').textContent = t.s3_hud1;
  $('#s3-hl2').textContent = t.s3_hud2;
  $('#s3-hl3').textContent = t.s3_hud3;
  $('#s3-hl4').textContent = t.s3_hud4;
  t.s3_stages.forEach((lbl, idx) => {
    const b = $(`.s3-st-btn[data-st="${idx}"]`);
    if (b) b.textContent = lbl;
  });

  // Scene 4 labels
  $('#s4-hl1').textContent = t.s4_hud1;
  $('#s4-hl2').textContent = t.s4_hud2;
  $('#s4-hl3').textContent = t.s4_hud3;
  $('#s4-hl4').textContent = t.s4_hud4;
  $('#s4-f-all').textContent = t.s4_all;
  $('#s4-f-ng').textContent = t.s4_ngOnly;
  $('#s4-replay').textContent = t.s4_replay;

  // Scene 5 labels
  $('#s5-hl1').textContent = t.s5_hud1;
  $('#s5-hl2').textContent = t.s5_hud2;
  $('#s5-hl3').textContent = t.s5_hud3;
  $('#s5-hl4').textContent = t.s5_hud4;
  $('#s5-ovlp-btn').textContent = t.s5_ovlp;
  $('#s5-ser-btn').textContent = t.s5_serial;
  $('#s5-replay').textContent = t.s5_replay;
  $('#s5-hv4').textContent = t.s5_tilesDown;

  updateAllHUDs();
}

function updateAllHUDs() {
  const t = T();
  // Scene 1 HUD
  const ex = D.examples[STATE.exIdx];
  const avgK = ex.layers.reduce((a, b) => a + b.pct, 0) / 4;
  const b1 = STATE.s1Scale === '4096' ? D.bench.hw4096.b1 : D.bench.hw1024.b1;
  $('#s1-hv1').textContent = `${avgK.toFixed(1)}%`;
  $('#s1-hv2').textContent = `${b1.speedup.toFixed(2)}×`;
  $('#s1-hv3').textContent = `${D.bench.accuracy.ch2.toFixed(2)}%`;
  const predNm = STATE.lang === 'ko' ? D.classes_ko[ex.pred] : D.classes_en[ex.pred];
  $('#s1-hv4').textContent = `${predNm} ${(ex.conf * 100).toFixed(0)}%`;

  // Scene 2 HUD
  $('#s2-hv1').textContent = `${D.batch_scene.strip_tok_dens.toFixed(1)}%`;
  $('#s2-hv2').textContent = `${D.batch_scene.strip_union_pct.toFixed(1)}%`;
  $('#s2-hv3').textContent = `${D.batch_scene.grid16_arrival.x0.toFixed(1)}%`;
  $('#s2-hv4').textContent = t.s2_collapsed;

  // Scene 3 HUD (live interpolated along s3Morph 0..3!)
  const mFrac = Math.max(0, Math.min(1, STATE.s3Morph / 3));
  const ta = D.bench.codesign.track_a;
  const tb = D.bench.codesign.track_b;
  $('#s3-hv1').textContent = `${lerp(ta.x0_2048, tb.x0_2048, mFrac).toFixed(1)}%`;
  $('#s3-hv2').textContent = `${lerp(ta.p2_2048, tb.p2_2048, mFrac).toFixed(1)}%`;
  $('#s3-hv3').textContent = `${lerp(ta.dram_2048, tb.dram_2048, mFrac).toFixed(1)}%`;
  $('#s3-hv4').textContent = t.s3_exact;

  // Scene 4 HUD
  const r2048 = D.bench.hw4096.rows.find(r => r.B === 2048);
  $('#s4-hv1').textContent = `${r2048.tiles.toLocaleString()}`;
  $('#s4-hv2').textContent = `76.7%`;
  $('#s4-hv3').textContent = `${r2048.fill_rate.toFixed(1)}%`;
  $('#s4-hv4').textContent = `0.00 ms`;

  // Scene 5 HUD
  const hw = STATE.s5Scale === '4096' ? D.bench.hw4096 : D.bench.hw1024;
  const row2k = hw.rows.find(r => r.B === 2048);
  $('#s5-hv1').textContent = `${hw.b1.speedup.toFixed(2)}×`;
  $('#s5-hv2').textContent = `${row2k.gpu_x.toFixed(2)}×`;
  $('#s5-hv3').textContent = `${row2k.e2e_serial_x.toFixed(2)}×`;
}

function setMode(mode) {
  STATE.mode = mode;
  document.body.classList.toggle('mode-cinema', mode === 'cinema');
  document.body.classList.toggle('mode-scroll', mode === 'scroll');
  $('#btn-mode-cinema').classList.toggle('active', mode === 'cinema');
  $('#btn-mode-scroll').classList.toggle('active', mode === 'scroll');
  syncActiveSceneDOM();
}

function setScene(idx, resetAnim = true) {
  idx = Math.max(0, Math.min(4, idx));
  STATE.scene = idx;
  STATE.cinemaTime = idx * SCENE_DUR;
  if (resetAnim) {
    if (idx === 0) STATE.s1Pulse = 0;
    if (idx === 1) STATE.s2Scan = 0;
    if (idx === 2) { STATE.s3Morph = 0; STATE.s3UserLocked = false; }
    if (idx === 3) STATE.s4Reveal = 0;
    if (idx === 4) STATE.s5RaceT = 0;
  }
  syncActiveSceneDOM();
  if (STATE.mode === 'scroll') {
    const el = $(`#sec-${T().chapters[idx].id}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function syncActiveSceneDOM() {
  $$('.scene-sec').forEach((el, i) => {
    el.classList.toggle('is-active', i === STATE.scene);
  });
  $$('.nav-ch').forEach((el, i) => {
    el.classList.toggle('active', i === STATE.scene);
  });
  $$('.scrub-seg').forEach((el, i) => {
    el.classList.toggle('active', i === STATE.scene);
    const fill = el.querySelector('.scrub-fill');
    if (fill) fill.style.width = i === STATE.scene ? '100%' : '0%';
  });
}

// ======================== SCENE 1 RENDERER: NEURAL IGNITION ========================
function renderScene1(now) {
  const cv = $('#cv-s1');
  const fit = setCanvasSize(cv);
  if (!fit) return;
  const { ctx, w, h } = fit;
  const t = T();
  const ex = D.examples[STATE.exIdx];
  const b1 = STATE.s1Scale === '4096' ? D.bench.hw4096.b1 : D.bench.hw1024.b1;
  const speedup = b1.speedup; // 6.27 or 6.20
  const isCompact = w < 640;

  ctx.clearRect(0, 0, w, h);

  // Subtle ambient grid
  ctx.strokeStyle = 'rgba(100,140,220,0.045)';
  ctx.lineWidth = 1;
  for (let gx = 0; gx < w; gx += 48) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke(); }
  for (let gy = 0; gy < h; gy += 48) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke(); }

  const padX = isCompact ? 12 : 24;
  const gapY = isCompact ? 10 : 14;
  const panelH = (h - gapY * 3) / 2;

  // Pulse progress: Roster finishes at s1Pulse = 1 / speedup (~0.16), Dense finishes at s1Pulse = 0.88
  const pDense = Math.min(1, STATE.s1Pulse / 0.86);
  const pRoster = Math.min(1, (STATE.s1Pulse * speedup) / 0.86);

  function drawNetworkPanel(y0, ph, isRoster, prog) {
    const px = padX, pw = w - padX * 2;
    // Glass card backdrop
    roundRect(ctx, px, y0, pw, ph, 14);
    ctx.fillStyle = isRoster ? 'rgba(9, 20, 38, 0.78)' : 'rgba(22, 11, 20, 0.65)';
    ctx.fill();
    ctx.strokeStyle = isRoster ? 'rgba(34, 211, 238, 0.34)' : 'rgba(251, 113, 133, 0.24)';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Header strip inside panel
    const panelTitle = isRoster ? t.s1_roster : t.s1_dense;
    ctx.font = `700 ${isCompact ? 11 : 13}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = isRoster ? '#22d3ee' : '#fb7185';
    ctx.textAlign = 'left';
    ctx.fillText(panelTitle, px + 14, y0 + 21);
    const titleW = ctx.measureText(panelTitle).width;

    ctx.font = `500 ${isCompact ? 10 : 11}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = '#94a3b8';
    const subTxt = isRoster
      ? `${t.s1_rosterSub} · ${speedup.toFixed(2)}×`
      : t.s1_denseSub;
    if (!isCompact) ctx.fillText(subTxt, px + 14 + titleW + 14, y0 + 21);

    // Live ms / arrival pill on top right
    const msDone = isRoster ? b1.ours_ms : b1.base_ms;
    const msCur = (msDone * prog).toFixed(3);
    const pillTxt = prog >= 1 ? `${t.s1_arrived} ${msDone.toFixed(3)} ms` : `${msCur} ms`;
    ctx.font = `700 ${isCompact ? 10 : 12}px "JetBrains Mono", monospace`;
    const pillW = ctx.measureText(pillTxt).width + 18;
    roundRect(ctx, px + pw - pillW - 12, y0 + 7, pillW, 20, 10);
    ctx.fillStyle = prog >= 1
      ? (isRoster ? 'rgba(52, 211, 153, 0.22)' : 'rgba(251, 113, 133, 0.20)')
      : 'rgba(148, 163, 184, 0.12)';
    ctx.fill();
    ctx.fillStyle = prog >= 1 ? (isRoster ? '#34d399' : '#fda4af') : '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.fillText(pillTxt, px + pw - pillW / 2 - 12, y0 + 21);

    // Layout x coordinates: Input -> L1 -> L2 -> L3 -> L4 -> Output
    const innerTop = y0 + 34;
    const innerH = ph - 46;
    const midY = innerTop + innerH / 2;
    const imgSize = Math.min(isCompact ? 38 : 54, innerH * 0.68);
    const xIn = px + (isCompact ? 28 : 48);
    const xOut = px + pw - (isCompact ? 42 : 68);
    const span = xOut - xIn;
    const layerXs = [0.20, 0.42, 0.64, 0.84].map(f => xIn + span * f);
    const prismOffset = isCompact ? 18 : 34;

    // Input image box
    const im = EX_IMGS[STATE.exIdx];
    ctx.save();
    roundRect(ctx, xIn - imgSize / 2, midY - imgSize / 2, imgSize, imgSize, 8);
    ctx.fillStyle = '#040710';
    ctx.fill();
    ctx.strokeStyle = isRoster ? 'rgba(34,211,238,0.65)' : 'rgba(251,113,133,0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.clip();
    if (im && im.complete) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(im, xIn - imgSize / 2 + 2, midY - imgSize / 2 + 2, imgSize - 4, imgSize - 4);
    }
    ctx.restore();

    // Draw 40 visual nodes per layer where Roster lights up EXACTLY the true active proportion (L.pct%)!
    const N_VIS = 40;
    const nodeStep = Math.min(innerH / (N_VIS + 1), isCompact ? 3.2 : 4.6);
    const stackH = nodeStep * (N_VIS - 1);
    const stackTop = midY - stackH / 2;

    const getLayerAct = (lIdx) => {
      const L = ex.layers[lIdx];
      const arr = new Float32Array(N_VIS);
      const onA = new Float32Array(N_VIS);
      const scores = [];
      for (let i = 0; i < N_VIS; i++) {
        const gIdx = Math.min(63, Math.floor((i * 64) / N_VIS));
        const aVal = isRoster ? L.s64[gIdx] : L.d64[gIdx];
        const oVal = L.on64[gIdx];
        arr[i] = aVal;
        scores.push([oVal * 0.7 + aVal * 0.3, i]);
      }
      if (isRoster) {
        // Light up EXACTLY the true active neuron fraction so 86.5% visibly stay dark!
        const kVis = Math.max(1, Math.round((N_VIS * L.pct) / 100));
        scores.sort((a, b) => b[0] - a[0]);
        for (let k = 0; k < kVis; k++) {
          const idx = scores[k][1];
          onA[idx] = 1.0;
          arr[idx] = Math.max(0.45, arr[idx]);
        }
      } else {
        onA.fill(1.0);
      }
      return { arr, onA, L };
    };

    const layerData = [0, 1, 2, 3].map(getLayerAct);

    // Signal wave X position
    const waveX = lerp(xIn, xOut, prog);

    // Draw connecting synapses first (under nodes)
    for (let l = 0; l < 4; l++) {
      const lx = layerXs[l];
      const prevX = l === 0 ? xIn + imgSize / 2 : layerXs[l - 1];
      const srcX = isRoster ? lx - prismOffset : prevX;
      const reached = Math.max(0, Math.min(1, (waveX - prevX) / Math.max(1, lx - prevX)));
      const { arr, onA } = layerData[l];

      // In Roster mode: horizontal cyan residual stream carries h, while golden lookup tap queries centroid prism c#j!
      if (isRoster) {
        const prismY = midY + (isCompact ? 22 : 30);
        const pReached = Math.max(0, Math.min(1, (waveX - prevX) / Math.max(1, srcX - prevX)));
        // Cyan residual highway (h passes directly into the layer, NOT bottlenecked through centroid!)
        ctx.beginPath();
        ctx.moveTo(prevX, midY);
        ctx.lineTo(lerp(prevX, lx, reached), midY);
        ctx.strokeStyle = `rgba(34, 211, 238, ${0.22 + 0.45 * reached})`;
        ctx.lineWidth = 2.0;
        ctx.stroke();

        // Golden lookup tap branching from h down to centroid prism c#j
        const tapX = lerp(prevX, srcX, 0.55);
        ctx.beginPath();
        ctx.moveTo(tapX, midY);
        ctx.lineTo(lerp(tapX, srcX, pReached), lerp(midY, prismY, pReached));
        ctx.strokeStyle = `rgba(251, 191, 36, ${0.35 + 0.6 * pReached})`;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      // Edges into layer l's nodes
      const prismY = midY + (isCompact ? 22 : 30);
      for (let i = 0; i < N_VIS; i++) {
        const ny = stackTop + i * nodeStep;
        const active = isRoster ? onA[i] > 0.06 : true;
        if (!active) continue;
        const inten = arr[i];
        if (!isRoster && i % 2 !== 0 && isCompact) continue;

        ctx.beginPath();
        if (isRoster) {
          // Cyan activation ray from residual h -> active neuron
          ctx.moveTo(srcX, midY);
          const cpX = (srcX + lx) * 0.5;
          ctx.quadraticCurveTo(cpX, lerp(midY, ny, 0.45), lx, ny);
          const alpha = reached > 0 ? (0.18 + 0.58 * inten * reached) : 0.05;
          ctx.strokeStyle = `rgba(34, 211, 238, ${alpha.toFixed(3)})`;
          ctx.lineWidth = 0.9 + inten * 1.1;
          ctx.stroke();

          // Subtle golden dashed index-selection ray from centroid prism c#j -> active neuron
          ctx.save();
          ctx.setLineDash([2.5, 3]);
          ctx.beginPath();
          ctx.moveTo(srcX, prismY);
          ctx.lineTo(lx, ny);
          ctx.strokeStyle = `rgba(251, 191, 36, ${(reached > 0 ? 0.22 * reached : 0.03).toFixed(3)})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
          ctx.restore();
        } else {
          // Dense web: heavy saturated connections
          const sy = l === 0 ? midY : stackTop + ((i * 7 + l * 5) % N_VIS) * nodeStep;
          ctx.moveTo(prevX, sy);
          ctx.lineTo(lx, ny);
          const alpha = reached > 0 ? (0.09 + 0.16 * reached) : 0.03;
          ctx.strokeStyle = `rgba(251, 113, 133, ${alpha.toFixed(3)})`;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }

        // Moving photon dot along active Roster edge!
        if (isRoster && reached > 0.05 && reached < 0.98 && inten > 0.25) {
          const pxDot = lerp(srcX, lx, reached);
          const pyDot = lerp(midY, ny, reached);
          ctx.beginPath();
          ctx.arc(pxDot, pyDot, 2.0, 0, Math.PI * 2);
          ctx.fillStyle = '#fde68a';
          ctx.fill();
        }
      }
    }

    // Output rays from L4 -> Output badge
    const outReached = Math.max(0, Math.min(1, (waveX - layerXs[3]) / Math.max(1, xOut - layerXs[3])));
    for (let i = 0; i < N_VIS; i++) {
      const { onA, arr } = layerData[3];
      if (isRoster && onA[i] <= 0.04) continue;
      if (!isRoster && i % 3 !== 0) continue;
      const ny = stackTop + i * nodeStep;
      ctx.beginPath();
      ctx.moveTo(layerXs[3], ny);
      ctx.lineTo(lerp(layerXs[3], xOut - 18, outReached), lerp(ny, midY, outReached));
      ctx.strokeStyle = isRoster
        ? `rgba(52, 211, 153, ${0.25 + 0.65 * arr[i] * outReached})`
        : `rgba(251, 113, 133, ${0.14 * outReached})`;
      ctx.lineWidth = isRoster ? 1.5 : 0.7;
      ctx.stroke();
    }

    // Draw Centroid Prisms (Roster only) + Layer Nodes + Layer Captions
    for (let l = 0; l < 4; l++) {
      const lx = layerXs[l];
      const { arr, onA, L } = layerData[l];
      const lit = waveX >= lx - 6;

      if (isRoster) {
        const px0 = lx - prismOffset;
        const py0 = midY + (isCompact ? 22 : 30);
        const pLit = waveX >= px0 - 6;
        const pr = isCompact ? 6.5 : 8.5;
        ctx.save();
        ctx.translate(px0, py0);
        ctx.rotate(Math.PI / 4);
        if (pLit) {
          ctx.shadowColor = '#fbbf24';
          ctx.shadowBlur = 12;
        }
        ctx.fillStyle = pLit ? '#fbbf24' : 'rgba(251,191,36,0.22)';
        ctx.fillRect(-pr / 2, -pr / 2, pr, pr);
        ctx.restore();

        ctx.font = `700 ${isCompact ? 8 : 9.5}px "JetBrains Mono", monospace`;
        ctx.fillStyle = pLit ? '#fde68a' : '#64748b';
        ctx.textAlign = 'center';
        ctx.fillText(`c#${L.j}`, px0, py0 + (isCompact ? 15 : 18));
        if (!isCompact) {
          ctx.font = `500 8.5px "JetBrains Mono", monospace`;
          ctx.fillStyle = '#64748b';
          ctx.fillText(`M=${L.M}`, px0, py0 + 29);
        }
      }

      // Draw 32 vertical neuron nodes
      for (let i = 0; i < N_VIS; i++) {
        const ny = stackTop + i * nodeStep;
        const active = isRoster ? onA[i] > 0.05 : true;
        const inten = arr[i];
        const rDot = isCompact ? 1.6 : 2.3;

        ctx.beginPath();
        if (active && lit) {
          ctx.arc(lx, ny, rDot + inten * (isRoster ? 1.5 : 0.6), 0, Math.PI * 2);
          ctx.fillStyle = isRoster
            ? (inten > 0.55 ? '#67e8f9' : '#22d3ee')
            : `rgba(251, 113, 133, ${(0.45 + 0.5 * inten).toFixed(2)})`;
          if (isRoster && inten > 0.4) {
            ctx.shadowColor = '#22d3ee';
            ctx.shadowBlur = 8;
          }
          ctx.fill();
          ctx.shadowBlur = 0;
        } else {
          // Dormant neuron (86.5% stay dark!)
          ctx.arc(lx, ny, rDot * 0.72, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(71, 85, 105, 0.35)';
          ctx.fill();
        }
      }

      // Layer bottom caption
      ctx.font = `600 ${isCompact ? 8.5 : 10}px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center';
      ctx.fillStyle = isRoster ? (lit ? '#67e8f9' : '#64748b') : (lit ? '#fda4af' : '#64748b');
      const lbl = isRoster ? `L${l + 1} · ${L.pct.toFixed(1)}%` : `L${l + 1} · 100%`;
      ctx.fillText(lbl, lx, y0 + ph - 7);
    }

    // Output prediction badge
    const outDone = prog >= 0.99;
    const obR = isCompact ? 17 : 24;
    ctx.beginPath();
    ctx.arc(xOut, midY, obR, 0, Math.PI * 2);
    ctx.fillStyle = outDone
      ? (isRoster ? 'rgba(52, 211, 153, 0.22)' : 'rgba(251, 113, 133, 0.2)')
      : 'rgba(15, 23, 42, 0.8)';
    if (outDone) {
      ctx.shadowColor = isRoster ? '#34d399' : '#fb7185';
      ctx.shadowBlur = 16;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = outDone ? (isRoster ? '#34d399' : '#fb7185') : 'rgba(100,116,139,0.4)';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    ctx.font = `700 ${isCompact ? 9 : 11}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = outDone ? '#f8fafc' : '#64748b';
    ctx.textAlign = 'center';
    const predTxt = STATE.lang === 'ko' ? D.classes_ko[ex.pred] : D.classes_en[ex.pred];
    ctx.fillText(outDone ? predTxt : '...', xOut, midY + 4);
  }

  drawNetworkPanel(gapY, panelH, false, pDense);
  drawNetworkPanel(gapY * 2 + panelH, panelH, true, pRoster);
}

// ======================== SCENE 2 RENDERER: THE BATCHING WALL ========================
let S2_BOX = { x: 0, y: 0, size: 300 };

function renderScene2() {
  const cv = $('#cv-s2');
  const fit = setCanvasSize(cv);
  if (!fit) return;
  const { ctx, w, h } = fit;
  const t = T();
  const isCompact = w < 660;

  ctx.clearRect(0, 0, w, h);

  const pad = isCompact ? 14 : 24;
  const unionH = isCompact ? 46 : 56;
  const sideW = isCompact ? 0 : Math.min(260, w * 0.30);
  const availW = w - pad * 2 - sideW - (sideW ? 20 : 0);
  const availH = h - pad * 2 - unionH - 28;
  const matW = Math.max(180, Math.min(availW, availH * 1.45));
  const matH = Math.max(150, Math.min(availH, matW * 0.85));
  const mx = pad + (availW - matW) / 2;
  const my = pad + 18;
  S2_BOX = { x: mx, y: my, w: matW, h: matH };

  // Frame
  roundRect(ctx, mx - 2, my - 2, matW + 4, matH + 4, 8);
  ctx.fillStyle = '#050914';
  ctx.fill();
  ctx.strokeStyle = 'rgba(148,163,184,0.22)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Axis captions
  ctx.font = `600 10.5px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  ctx.fillText(`${t.s2_tokAxis}  ×  ${t.s2_neuAxis}`, mx, my - 7);

  const scanRow = STATE.s2Mode === 'rain' ? Math.floor(STATE.s2Scan) : 512;

  // Draw real unsorted arrival matrix up to scanRow
  ctx.save();
  ctx.beginPath();
  ctx.rect(mx, my, matW, matH);
  ctx.clip();
  ctx.imageSmoothingEnabled = false;

  if (STATE.s2Mode === 'hover') {
    ctx.globalAlpha = 0.25;
    ctx.drawImage(OFFSCREEN.arrival, mx, my, matW, matH);
    ctx.globalAlpha = 1;
    // Highlight single hovered row brightly!
    const hr = Math.max(0, Math.min(511, STATE.s2HoverRow));
    const hy = my + (hr / 512) * matH;
    ctx.fillStyle = 'rgba(34, 211, 238, 0.14)';
    ctx.fillRect(mx, hy - 3, matW, 7);
    const base = hr * 512;
    let rowCnt = 0;
    ctx.fillStyle = '#67e8f9';
    for (let c = 0; c < 512; c++) {
      if (MATS.arrival[base + c]) {
        rowCnt++;
        const cx = mx + (c / 512) * matW;
        ctx.fillRect(cx, hy - 2, Math.max(1.5, matW / 512), 5);
      }
    }
    ctx.fillStyle = '#fde68a';
    ctx.font = `700 11px "JetBrains Mono", monospace`;
    ctx.textAlign = 'left';
    ctx.fillText(`${t.s2_tokRow} #${hr}: ${((rowCnt / 512) * 100).toFixed(1)}% active`, mx + 10, Math.max(my + 16, hy - 8));
  } else {
    const frac = Math.min(1, scanRow / 512);
    if (frac > 0) {
      ctx.drawImage(OFFSCREEN.arrival, 0, 0, 512, Math.max(1, Math.round(512 * frac)), mx, my, matW, matH * frac);
    }
    // Glowing laser scanline in Rain mode
    if (STATE.s2Mode === 'rain' && scanRow < 512) {
      const sy = my + (scanRow / 512) * matH;
      const g = ctx.createLinearGradient(0, sy - 16, 0, sy + 2);
      g.addColorStop(0, 'rgba(34,211,238,0)');
      g.addColorStop(1, 'rgba(34,211,238,0.65)');
      ctx.fillStyle = g;
      ctx.fillRect(mx, sy - 16, matW, 18);
      ctx.fillStyle = '#e0f2fe';
      ctx.fillRect(mx, sy, matW, 1.5);
    }
    // Fixed 16x16 BSMR Grid overlay showing massive coral-red x0 waste boxes!
    if (STATE.s2Mode === 'grid') {
      const boxes = D.batch_scene.grid16_arrival.boxes;
      const sx = matW / 512, sy = matH / 512;
      for (let i = 0; i < boxes.length; i++) {
        const [r0, c0, ones] = boxes[i];
        const waste = 1 - ones / 256;
        ctx.fillStyle = `rgba(251, 113, 133, ${(0.08 + waste * 0.28).toFixed(2)})`;
        ctx.fillRect(mx + c0 * sx, my + r0 * sy, 16 * sx, 16 * sy);
        ctx.strokeStyle = 'rgba(251, 113, 133, 0.36)';
        ctx.lineWidth = 0.6;
        ctx.strokeRect(mx + c0 * sx, my + r0 * sy, 16 * sx, 16 * sy);
      }
    }
  }
  ctx.restore();

  // Bottom Cumulative Column-Union Saturation Bar!
  const uy = my + matH + 14;
  ctx.font = `600 10.5px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#cbd5e1';
  ctx.textAlign = 'left';
  let litCols = 0;
  for (let c = 0; c < 512; c++) {
    if (COL_FIRST_ROW[c] < scanRow) litCols++;
  }
  const unionPct = (litCols / 512) * 100;
  ctx.fillText(`${t.s2_unionBar}: ${unionPct.toFixed(1)}%  (${litCols} / 512)`, mx, uy + 10);

  const barY = uy + 18, barH = 18;
  roundRect(ctx, mx, barY, matW, barH, 5);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(251, 113, 133, 0.35)';
  ctx.stroke();

  const cw = matW / 512;
  for (let c = 0; c < 512; c++) {
    if (COL_FIRST_ROW[c] < scanRow) {
      ctx.fillStyle = unionPct > 85 ? '#fb7185' : '#fbbf24';
      ctx.fillRect(mx + c * cw, barY + 2, Math.max(1, cw + 0.3), barH - 4);
    }
  }

  // Right visual diagnostic gauge card (Desktop)
  if (!isCompact && sideW > 0) {
    const rx = mx + matW + 20;
    const ry = my;
    roundRect(ctx, rx, ry, sideW, matH + unionH + 14, 12);
    ctx.fillStyle = 'rgba(12, 18, 34, 0.72)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(148,163,184,0.16)';
    ctx.stroke();

    // Big circular saturation ring
    const cx = rx + sideW / 2;
    const cy = ry + 96;
    const rad = Math.min(54, sideW * 0.3);
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    ctx.lineWidth = 10;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, rad, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2) * (unionPct / 100));
    ctx.strokeStyle = unionPct > 85 ? '#fb7185' : '#fbbf24';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();

    ctx.font = `800 22px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.fillText(`${unionPct.toFixed(0)}%`, cx, cy + 7);
    ctx.font = `600 10px Inter, system-ui, sans-serif`;
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('UNION', cx, cy + 24);

    // 3 crisp visual mini bars underneath
    const rows = [
      { lbl: STATE.lang === 'ko' ? '1토큰 활성' : '1 Token Active', val: D.batch_scene.strip_tok_dens, col: '#22d3ee' },
      { lbl: STATE.lang === 'ko' ? '512토큰 합집합' : '512-Tok Union', val: unionPct, col: '#fb7185' },
      { lbl: STATE.lang === 'ko' ? '16×16 격자 ×0' : '16×16 Grid ×0', val: D.batch_scene.grid16_arrival.x0, col: '#fbbf24' },
    ];
    rows.forEach((r, idx) => {
      const by = cy + rad + 36 + idx * 46;
      if (by + 28 > ry + matH + unionH) return;
      ctx.font = `600 10.5px Inter, system-ui, sans-serif`;
      ctx.fillStyle = '#cbd5e1';
      ctx.textAlign = 'left';
      ctx.fillText(r.lbl, rx + 16, by);
      ctx.textAlign = 'right';
      ctx.font = `700 11px "JetBrains Mono", monospace`;
      ctx.fillStyle = r.col;
      ctx.fillText(`${r.val.toFixed(1)}%`, rx + sideW - 16, by);

      roundRect(ctx, rx + 16, by + 7, sideW - 32, 9, 4);
      ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
      ctx.fill();
      roundRect(ctx, rx + 16, by + 7, Math.max(4, (sideW - 32) * (r.val / 100)), 9, 4);
      ctx.fillStyle = r.col;
      ctx.fill();
    });
  }
}

// ======================== SCENE 3 RENDERER: CH2 MAGNETIC MORPH ========================
function renderScene3() {
  const cv = $('#cv-s3');
  const fit = setCanvasSize(cv);
  if (!fit) return;
  const { ctx, w, h } = fit;
  const t = T();
  const isCompact = w < 680;
  const m = Math.max(0, Math.min(3, STATE.s3Morph));

  ctx.clearRect(0, 0, w, h);

  const pad = isCompact ? 14 : 24;
  const sideW = isCompact ? 0 : Math.min(250, w * 0.28);
  const availW = w - pad * 2 - sideW - (sideW ? 20 : 0);
  const availH = h - pad * 2 - 22;
  const matW = Math.max(180, Math.min(availW - 16, availH * 1.45));
  const matH = Math.max(150, Math.min(availH, matW * 0.85));
  const mx = pad + 12 + (availW - 12 - matW) / 2;
  const my = pad + 18;

  // Frame
  roundRect(ctx, mx - 2, my - 2, matW + 4, matH + 4, 8);
  ctx.fillStyle = '#050914';
  ctx.fill();
  ctx.strokeStyle = m > 2.4 ? 'rgba(52, 211, 153, 0.45)' : 'rgba(148, 163, 184, 0.22)';
  ctx.lineWidth = 1.2;
  ctx.stroke();

  // Stage label above matrix
  const stIdx = Math.min(3, Math.round(m));
  ctx.font = `700 11.5px "JetBrains Mono", monospace`;
  ctx.fillStyle = m > 2.4 ? '#34d399' : '#38bdf8';
  ctx.textAlign = 'left';
  ctx.fillText(t.s3_stages[stIdx], mx, my - 7);

  // Subtle 32-column Tensor Core block guide rails fading in during Stage 2 -> 3
  const railAlpha = Math.max(0, Math.min(1, m - 2));
  if (railAlpha > 0.02) {
    for (let b = 0; b < 16; b++) {
      const bx = mx + (b / 16) * matW;
      if (b % 2 === 0) {
        ctx.fillStyle = `rgba(52, 211, 153, ${(0.045 * railAlpha).toFixed(3)})`;
        ctx.fillRect(bx, my, matW / 16, matH);
      }
      ctx.beginPath();
      ctx.moveTo(bx, my); ctx.lineTo(bx, my + matH);
      ctx.strokeStyle = `rgba(52, 211, 153, ${(0.18 * railAlpha).toFixed(3)})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  // Left centroid color ribbon fading in from Stage 0 -> 1+
  const bandAlpha = Math.max(0, Math.min(1, m));
  if (bandAlpha > 0.02) {
    D.batch_scene.bands.forEach(([cid, r0, rlen]) => {
      const by = my + (r0 / 512) * matH;
      const bh = (rlen / 512) * matH;
      ctx.fillStyle = centroidColor(cid, 0.85 * bandAlpha);
      ctx.fillRect(mx - 10, by, 6, Math.max(1.5, bh - 0.5));
    });
  }

  // Fade in real Track A (M=256) density backdrop near Stage 0, and real CH2 density backdrop near Stage 3!
  if (m < 0.55) {
    const aBefore = Math.max(0, 1 - m / 0.55) * 0.36;
    ctx.save();
    ctx.globalAlpha = aBefore;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(OFFSCREEN.track_a, mx, my, matW, matH);
    ctx.restore();
  }
  if (m > 2.35) {
    const aSheet = Math.min(1, (m - 2.35) / 0.65) * 0.42;
    ctx.save();
    ctx.globalAlpha = aSheet;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(OFFSCREEN.ch2, mx, my, matW, matH);
    ctx.restore();
  }

  // Draw 2,200 REAL active particles morphing from Track A (M=256 indiv) -> CH2 (M=32 -> B1 -> B2 32-col walls)!
  // particle = [r_ta, c_ta, r_m32, c_raw, r_olo, c_b2, cid]
  const pts = D.batch_scene.particles;
  const sx = matW / 512, sy = matH / 512;
  const dotSize = isCompact ? 1.9 : 2.4;

  const t01 = easeInOutCubic(Math.min(1, m));
  const t12 = easeInOutCubic(Math.max(0, Math.min(1, m - 1)));
  const t23 = easeInOutCubic(Math.max(0, Math.min(1, m - 2)));

  for (let i = 0; i < pts.length; i++) {
    const [rTa, cTa, rM32, cRaw, rOlo, cB2, cid] = pts[i];
    let rPos, cPos;
    if (m <= 1) {
      rPos = lerp(rTa, rM32, t01);
      cPos = lerp(cTa, cRaw, t01);
    } else if (m <= 2) {
      rPos = lerp(rM32, rOlo, t12);
      cPos = cRaw;
    } else {
      rPos = rOlo;
      cPos = lerp(cRaw, cB2, t23);
    }
    const px = mx + cPos * sx;
    const py = my + rPos * sy;
    if (m < 0.35) ctx.fillStyle = '#fb7185';
    else if (m > 2.6) ctx.fillStyle = '#34d399';
    else ctx.fillStyle = centroidColor(cid, 0.9);
    ctx.fillRect(px, py, dotSize, dotSize);
  }

  // Right Before (Track A M=256) vs After (CH2) real comparison cards (Desktop)
  if (!isCompact && sideW > 0) {
    const rx = mx + matW + 20;
    const cardH = (matH - 12) / 2;
    const ta = D.bench.codesign.track_a;
    const tb = D.bench.codesign.track_b;

    function drawMiniCard(cy, title, offImg, accent, x0, tiles, gpuX) {
      roundRect(ctx, rx, cy, sideW, cardH, 12);
      ctx.fillStyle = 'rgba(12, 18, 34, 0.78)';
      ctx.fill();
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.1;
      ctx.stroke();

      ctx.font = `700 11px Inter, system-ui, sans-serif`;
      ctx.fillStyle = accent;
      ctx.textAlign = 'left';
      ctx.fillText(title, rx + 12, cy + 20);

      const thumbS = Math.min(cardH - 42, 86);
      ctx.save();
      roundRect(ctx, rx + 12, cy + 29, thumbS, thumbS, 6);
      ctx.fillStyle = '#040710';
      ctx.fill();
      ctx.clip();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offImg, rx + 12, cy + 29, thumbS, thumbS);
      ctx.restore();

      const tx = rx + 22 + thumbS;
      ctx.font = `700 13px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(`×0 ${x0.toFixed(1)}%`, tx, cy + 49);
      ctx.font = `600 11px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`${tiles.toLocaleString()} ${t.s3_tiles}`, tx, cy + 70);
      ctx.fillStyle = accent;
      ctx.fillText(`GPU ${gpuX.toFixed(2)}×`, tx, cy + 91);
    }

    drawMiniCard(my, t.s3_before, OFFSCREEN.track_a, '#fb7185', ta.x0_2048, ta.tiles_2048, ta.gpu_2048_x);
    drawMiniCard(my + cardH + 12, t.s3_after, OFFSCREEN.ch2, '#34d399', tb.x0_2048, tb.tiles_2048, tb.gpu_2048_x);
  }
}

// ======================== SCENE 4 RENDERER: 5-TIER NON-GRID TILING + A3V4 ========================
let S4_BOX = { x: 0, y: 0, w: 300, h: 300 };

function tileContains(tile, r512, c512) {
  let rHit = false, cHit = false;
  for (const [rs, rl] of tile.rr) { if (r512 >= rs && r512 < rs + rl) { rHit = true; break; } }
  if (!rHit) return false;
  for (const [cs, cl] of tile.cr) { if (c512 >= cs && c512 < cs + cl) { cHit = true; break; } }
  return cHit;
}

function getFilteredTiles() {
  const all = D.batch_scene.tiles;
  if (STATE.s4Filter === 'all') return all;
  if (STATE.s4Filter === 'ng') return all.filter(x => x.ng_r || x.ng_c);
  const tid = Number(STATE.s4Filter);
  return all.filter(x => x.t === tid);
}

function renderScene4() {
  const cv = $('#cv-s4');
  const fit = setCanvasSize(cv);
  if (!fit) return;
  const { ctx, w, h } = fit;
  const t = T();
  const isCompact = w < 680;

  ctx.clearRect(0, 0, w, h);

  const pad = isCompact ? 12 : 20;
  const pipeH = isCompact ? 92 : 106;
  const sideW = isCompact ? 0 : Math.min(256, w * 0.30);
  const topAvailH = h - pipeH - pad * 2 - 14;
  const topAvailW = w - pad * 2 - sideW - (sideW ? 18 : 0);
  const matW = Math.max(180, Math.min(topAvailW, topAvailH * 1.65));
  const matH = Math.max(140, Math.min(topAvailH - 16, matW * 0.85));
  const mx = pad + (topAvailW - matW) / 2;
  const my = pad + 14;
  S4_BOX = { x: mx, y: my, w: matW, h: matH };

  // Matrix backdrop
  roundRect(ctx, mx - 2, my - 2, matW + 4, matH + 4, 8);
  ctx.fillStyle = '#050914';
  ctx.fill();
  ctx.strokeStyle = 'rgba(148,163,184,0.22)';
  ctx.stroke();

  // Dim real CH2 sheet underneath
  ctx.save();
  ctx.globalAlpha = 0.28;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(OFFSCREEN.ch2Dim, mx, my, matW, matH);
  ctx.restore();

  const tiles = getFilteredTiles();
  const nShow = Math.max(1, Math.ceil(tiles.length * Math.min(1, STATE.s4Reveal)));
  const sx = matW / 512, sy = matH / 512;

  const activeIdx = STATE.s4Hover >= 0 ? STATE.s4Hover : STATE.s4Selected;
  const activeTile = D.batch_scene.tiles[activeIdx] || tiles[0];

  // Draw real A3V4 tiles
  for (let i = 0; i < nShow; i++) {
    const tl = tiles[i];
    const col = TIER_COLORS[tl.t] || '#22d3ee';
    const isSel = activeTile && tl.id === activeTile.id;
    const alphaFill = isSel ? 0.42 : 0.14;
    const alphaStroke = isSel ? 0.98 : 0.48;

    ctx.fillStyle = col;
    ctx.strokeStyle = col;
    for (const [rs, rl] of tl.rr) {
      for (const [cs, cl] of tl.cr) {
        const rx = mx + cs * sx, ry = my + rs * sy;
        const rw = Math.max(1.5, cl * sx), rh = Math.max(1.5, rl * sy);
        ctx.globalAlpha = alphaFill;
        ctx.fillRect(rx, ry, rw, rh);
        ctx.globalAlpha = alphaStroke;
        ctx.lineWidth = isSel ? 1.8 : 0.7;
        ctx.strokeRect(rx, ry, rw, rh);
      }
    }
  }
  ctx.globalAlpha = 1;

  // SIGNATURE NON-GRID VISUAL: Glowing curved laser bridges connecting non-contiguous bands of activeTile!
  if (activeTile) {
    const col = TIER_COLORS[activeTile.t] || '#fbbf24';
    ctx.save();
    ctx.strokeStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur = 12;
    ctx.lineWidth = 2.0;

    // Row bridges (when a single Tensor Core tile freely pairs separated centroid bands!)
    if (activeTile.rr.length > 1) {
      const cMid = mx + (activeTile.cr[0][0] + activeTile.cr[0][1] * 0.5) * sx;
      for (let k = 0; k < activeTile.rr.length - 1; k++) {
        const y1 = my + (activeTile.rr[k][0] + activeTile.rr[k][1]) * sy;
        const y2 = my + activeTile.rr[k + 1][0] * sy;
        const bend = Math.min(28, Math.abs(y2 - y1) * 0.45 + 10);
        ctx.beginPath();
        ctx.moveTo(cMid, y1);
        ctx.quadraticCurveTo(cMid + bend, (y1 + y2) / 2, cMid, y2);
        ctx.stroke();
      }
    }
    // Column bridges (when a tile freely pairs separated neuron columns!)
    if (activeTile.cr.length > 1) {
      const rMid = my + (activeTile.rr[0][0] + activeTile.rr[0][1] * 0.5) * sy;
      for (let k = 0; k < activeTile.cr.length - 1; k++) {
        const x1 = mx + (activeTile.cr[k][0] + activeTile.cr[k][1]) * sx;
        const x2 = mx + activeTile.cr[k + 1][0] * sx;
        const bend = Math.min(24, Math.abs(x2 - x1) * 0.4 + 8);
        ctx.beginPath();
        ctx.moveTo(x1, rMid);
        ctx.quadraticCurveTo((x1 + x2) / 2, rMid - bend, x2, rMid);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // Strip caption above matrix
  ctx.font = `600 10.5px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  const ss = D.batch_scene.strip_stats;
  ctx.fillText(`${t.s4_stripCap} · ${nShow}/${tiles.length} tiles · ${ss.non_grid_tiles} non-grid`, mx, my - 5);

  // Right Tile Inspector Panel (Desktop)
  if (!isCompact && sideW > 0 && activeTile) {
    const rx = mx + matW + 18;
    const ry = my;
    const col = TIER_COLORS[activeTile.t] || '#fbbf24';

    roundRect(ctx, rx, ry, sideW, matH, 12);
    ctx.fillStyle = 'rgba(12, 18, 34, 0.78)';
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.font = `700 11.5px Inter, system-ui, sans-serif`;
    ctx.fillStyle = col;
    ctx.textAlign = 'left';
    ctx.fillText(`${t.s4_inspector} #${activeTile.id}`, rx + 14, ry + 22);

    // Big Tier pill
    ctx.font = `800 22px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(TIER_NAMES[activeTile.t], rx + 14, ry + 50);

    const isNG = activeTile.ng_r || activeTile.ng_c;
    roundRect(ctx, rx + 14, ry + 60, sideW - 28, 22, 6);
    ctx.fillStyle = isNG ? 'rgba(251, 191, 36, 0.16)' : 'rgba(52, 211, 153, 0.14)';
    ctx.fill();
    ctx.font = `700 10px Inter, system-ui, sans-serif`;
    ctx.fillStyle = isNG ? '#fde68a' : '#6ee7b7';
    ctx.fillText(isNG ? t.s4_yes : t.s4_no, rx + 22, ry + 75);

    // Real zoomed pixel grid of the selected tile's actual 1s (cyan) and ×0s (red)!
    const zoomTop = ry + 94;
    const zoomW = sideW - 28;
    const zoomH = Math.max(44, Math.min(matH - 156, 92));
    roundRect(ctx, rx + 14, zoomTop, zoomW, zoomH, 6);
    ctx.fillStyle = '#040710';
    ctx.fill();

    // Expand exact row/col lists for zoomed pixel readout
    const rList = [], cList = [];
    for (const [rs, rl] of activeTile.rr) for (let k = 0; k < rl; k++) rList.push(rs + k);
    for (const [cs, cl] of activeTile.cr) for (let k = 0; k < cl; k++) cList.push(cs + k);
    const pw = zoomW / Math.max(1, cList.length);
    const ph = zoomH / Math.max(1, rList.length);
    for (let ri = 0; ri < rList.length; ri++) {
      const rOff = rList[ri] * 512;
      for (let ci = 0; ci < cList.length; ci++) {
        const bit = MATS.ch2[rOff + cList[ci]];
        ctx.fillStyle = bit ? col : 'rgba(251, 113, 133, 0.38)';
        ctx.fillRect(rx + 14 + ci * pw, zoomTop + ri * ph, Math.max(1, pw - 0.3), Math.max(1, ph - 0.3));
      }
    }

    // Compact numeric stats at bottom of inspector
    const sy0 = zoomTop + zoomH + 20;
    ctx.font = `600 11px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`${t.s4_fill}: ${activeTile.fill.toFixed(1)}%`, rx + 14, sy0);
    ctx.fillStyle = '#fda4af';
    ctx.fillText(`${t.s4_zeros}: ${activeTile.zeros} / ${activeTile.area}`, rx + 14, sy0 + 18);
    ctx.font = `500 9.5px Inter, system-ui, sans-serif`;
    ctx.fillStyle = '#64748b';
    ctx.fillText(`${activeTile.rr.length}R × ${activeTile.cr.length}C`, rx + sideW - 54, sy0);
  }

  // ================= LOWER STAGE: PARALLEL GPU ATTENTION vs CPU A3V4 TIMELINE =================
  const py = h - pipeH - pad / 2;
  const px = pad, pw = w - pad * 2;
  roundRect(ctx, px, py, pw, pipeH, 12);
  ctx.fillStyle = 'rgba(10, 16, 30, 0.82)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.28)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.font = `700 ${isCompact ? 9.5 : 11}px Inter, system-ui, sans-serif`;
  ctx.fillStyle = '#cbd5e1';
  ctx.textAlign = 'left';
  ctx.fillText(t.s4_pipeTitle, px + 14, py + 18);

  if (!isCompact) {
    ctx.font = `700 11px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#34d399';
    ctx.textAlign = 'right';
    ctx.fillText(t.s4_hidden, px + pw - 14, py + 18);
  }

  const lblW = isCompact ? 62 : 94;
  const trackX = px + lblW + 10;
  const trackW = pw - lblW - 24;
  const totMs = 13.30 + 2.09; // 15.39 ms (1 layer Attention + 1 layer Roster v4 FFN)
  const attnW = (13.30 / totMs) * trackW;
  const ffnW = (2.09 / totMs) * trackW;
  const a3W = (1.81 / totMs) * trackW;
  const h2dW = (0.29 / totMs) * trackW;

  // GPU Lane
  const yG = py + 30, laneH = 24;
  ctx.font = `700 ${isCompact ? 9 : 10.5}px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  ctx.fillText(t.s4_gpuLane, px + 14, yG + 16);

  roundRect(ctx, trackX, yG, attnW - 3, laneH, 6);
  ctx.fillStyle = 'rgba(99, 102, 241, 0.28)';
  ctx.fill();
  ctx.strokeStyle = '#818cf8';
  ctx.stroke();
  ctx.fillStyle = '#e0e7ff';
  ctx.textAlign = 'center';
  ctx.fillText(t.s4_attnLbl, trackX + attnW / 2, yG + 15.5);

  roundRect(ctx, trackX + attnW, yG, ffnW, laneH, 6);
  ctx.fillStyle = 'rgba(34, 211, 238, 0.30)';
  ctx.fill();
  ctx.strokeStyle = '#22d3ee';
  ctx.stroke();
  ctx.fillStyle = '#cffafe';
  ctx.fillText(isCompact ? 'FFN' : 'v4 FFN', trackX + attnW + ffnW / 2, yG + 15.5);

  // CPU AVX2 Lane (runs in parallel with Attention and finishes 6× earlier!)
  const yC = py + 62;
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  ctx.fillText(t.s4_cpuLane, px + 14, yC + 16);

  roundRect(ctx, trackX, yC, Math.max(24, a3W), laneH, 6);
  ctx.fillStyle = 'rgba(52, 211, 153, 0.34)';
  ctx.fill();
  ctx.strokeStyle = '#34d399';
  ctx.stroke();

  roundRect(ctx, trackX + Math.max(24, a3W) + 2, yC, Math.max(8, h2dW), laneH, 4);
  ctx.fillStyle = 'rgba(251, 191, 36, 0.38)';
  ctx.fill();
  ctx.strokeStyle = '#fbbf24';
  ctx.stroke();

  ctx.font = `700 ${isCompact ? 9 : 10.5}px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#6ee7b7';
  ctx.textAlign = 'left';
  ctx.fillText(t.s4_a3Lbl, trackX + Math.max(24, a3W) + Math.max(8, h2dW) + 12, yC + 15.5);

  // Animated playhead sweeping across the parallel timeline!
  const headX = trackX + STATE.s4PipeT * trackW;
  ctx.beginPath();
  ctx.moveTo(headX, yG - 4);
  ctx.lineTo(headX, yC + laneH + 4);
  ctx.strokeStyle = 'rgba(253, 224, 71, 0.85)';
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

// ======================== SCENE 5 RENDERER: END-TO-END GPU SILICON RACE ========================
let S5_TOWERS = [];

function renderScene5() {
  const cv = $('#cv-s5');
  const fit = setCanvasSize(cv);
  if (!fit) return;
  const { ctx, w, h } = fit;
  const t = T();
  const isCompact = w < 640;

  ctx.clearRect(0, 0, w, h);

  const hw = STATE.s5Scale === '4096' ? D.bench.hw4096 : D.bench.hw1024;
  const B = STATE.s5Batch;

  // Build 4 racing lanes for current B & overlap mode
  const shortNm = (full, mob) => (isCompact ? mob : full);
  let lanes;
  if (B === 1) {
    const b1 = hw.b1;
    lanes = [
      { name: shortNm(t.s5_t_base, 'Base cuBLAS'), ms: b1.base_ms, x: 1.00, col: '#fb7185' },
      { name: shortNm(t.s5_t_mask0, '0-Mask'), ms: b1.mask0_ms, x: b1.speedup_mask0, col: '#94a3b8' },
      { name: shortNm(t.s5_t_ch2, 'Roster CH2'), ms: b1.ours_ms, x: b1.speedup, col: '#22d3ee', hero: true },
    ];
  } else {
    const r = hw.rows.find(x => x.B === B);
    const ch2Ms = STATE.s5Ovlp ? r.gpu_ms : r.e2e_serial_ms;
    const ch2X = STATE.s5Ovlp ? r.gpu_x : r.e2e_serial_x;
    const trkAMs = STATE.s5Ovlp ? r.track_a_ms : (r.base_ms / r.a_e2e_serial_x);
    const trkAX = STATE.s5Ovlp ? r.track_a_x : r.a_e2e_serial_x;
    lanes = [
      { name: shortNm(t.s5_t_base, 'Base cuBLAS'), ms: r.base_ms, x: 1.00, col: '#fb7185' },
      { name: shortNm(t.s5_t_mask0, '0-Mask'), ms: r.mask0_ms, x: r.mask0_x, col: '#94a3b8' },
      { name: shortNm(t.s5_t_trka, 'Track A (M=256)'), ms: trkAMs, x: trkAX, col: '#a78bfa' },
      { name: shortNm(t.s5_t_ch2, 'Roster CH2'), ms: ch2Ms, x: ch2X, col: '#34d399', hero: true },
    ];
  }

  const maxMs = Math.max(...lanes.map(l => l.ms));
  const simMs = STATE.s5RaceT * maxMs; // current race clock in ms

  const pad = isCompact ? 14 : 24;
  const towerH = isCompact ? 102 : 120;
  const raceH = h - towerH - pad * 2 - 14;
  const laneGap = 10;
  const laneH = Math.min(64, (raceH - laneGap * (lanes.length - 1)) / lanes.length);
  const startY = pad + (raceH - (laneH * lanes.length + laneGap * (lanes.length - 1))) / 2;
  const trackW = w - pad * 2;

  lanes.forEach((ln, idx) => {
    const ly = startY + idx * (laneH + laneGap);
    const prog = Math.min(1, simMs / ln.ms);
    const done = prog >= 1;

    roundRect(ctx, pad, ly, trackW, laneH, 10);
    ctx.fillStyle = ln.hero ? 'rgba(9, 28, 38, 0.82)' : 'rgba(13, 19, 34, 0.72)';
    ctx.fill();
    ctx.strokeStyle = ln.hero ? 'rgba(52, 211, 153, 0.48)' : 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = ln.hero ? 1.5 : 1;
    ctx.stroke();

    // Label top-left
    ctx.font = `${ln.hero ? 700 : 600} ${isCompact ? 10.5 : 12}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = ln.hero ? '#e2e8f0' : '#cbd5e1';
    ctx.textAlign = 'left';
    ctx.fillText(ln.name, pad + 14, ly + 19);

    // Speedup + ms top-right
    ctx.font = `800 ${isCompact ? 11.5 : 13.5}px "JetBrains Mono", monospace`;
    ctx.fillStyle = ln.col;
    ctx.textAlign = 'right';
    const statusTxt = `${ln.x.toFixed(2)}×  ·  ${ln.ms.toFixed(2)} ms${done ? '  ✓' : ''}`;
    ctx.fillText(statusTxt, pad + trackW - 14, ly + 20);

    // Racing bar channel
    const bx = pad + 14, bw = trackW - 28, by = ly + laneH - 18, bh = 9;
    roundRect(ctx, bx, by, bw, bh, 5);
    ctx.fillStyle = 'rgba(30, 41, 59, 0.75)';
    ctx.fill();

    // Glowing light beam
    const fillW = Math.max(4, bw * prog);
    const grd = ctx.createLinearGradient(bx, 0, bx + fillW, 0);
    grd.addColorStop(0, ln.col + '33');
    grd.addColorStop(0.7, ln.col + 'cc');
    grd.addColorStop(1, ln.col);
    roundRect(ctx, bx, by, fillW, bh, 5);
    ctx.fillStyle = grd;
    if (ln.hero) {
      ctx.shadowColor = ln.col;
      ctx.shadowBlur = 14;
    }
    ctx.fill();
    ctx.shadowBlur = 0;

    // Bright racing head spark
    if (!done && prog > 0.01) {
      ctx.beginPath();
      ctx.arc(bx + fillW, by + bh / 2, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
  });

  // ================= BOTTOM 4 CLICKABLE BATCH SPEEDUP TOWERS =================
  const ty = h - towerH - pad / 2;
  const batches = [1, 512, 1024, 2048];
  const cardGap = isCompact ? 8 : 14;
  const cardW = (trackW - cardGap * 3) / 4;
  S5_TOWERS = [];

  batches.forEach((bVal, idx) => {
    const cx = pad + idx * (cardW + cardGap);
    S5_TOWERS.push({ B: bVal, x: cx, y: ty, w: cardW, h: towerH });
    const isSel = bVal === B;
    let sp, sub;
    if (bVal === 1) {
      sp = hw.b1.speedup;
      sub = 'GEMV';
    } else {
      const r = hw.rows.find(x => x.B === bVal);
      sp = STATE.s5Ovlp ? r.gpu_x : r.e2e_serial_x;
      sub = STATE.s5Ovlp ? `P1+P2` : `Serial`;
    }

    roundRect(ctx, cx, ty, cardW, towerH, 12);
    ctx.fillStyle = isSel ? 'rgba(14, 34, 54, 0.90)' : 'rgba(11, 17, 31, 0.72)';
    ctx.fill();
    ctx.strokeStyle = isSel ? '#22d3ee' : 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = isSel ? 1.8 : 1;
    ctx.stroke();

    // Speedup bar inside card (vertical on desktop, clean bottom horizontal bar on mobile!)
    const maxSp = 7.0;
    if (isCompact) {
      const hbw = cardW - 18;
      const fillH = Math.max(6, (Math.min(sp, maxSp) / maxSp) * hbw);
      roundRect(ctx, cx + 9, ty + towerH - 15, hbw, 6, 3);
      ctx.fillStyle = 'rgba(30, 41, 59, 0.7)';
      ctx.fill();
      roundRect(ctx, cx + 9, ty + towerH - 15, fillH, 6, 3);
      ctx.fillStyle = bVal === 1 ? '#22d3ee' : '#34d399';
      ctx.fill();

      ctx.textAlign = 'left';
      ctx.font = `700 9.5px "JetBrains Mono", monospace`;
      ctx.fillStyle = isSel ? '#67e8f9' : '#94a3b8';
      ctx.fillText(`B=${bVal.toLocaleString()}`, cx + 9, ty + 20);

      ctx.font = `800 15.5px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(`${sp.toFixed(2)}×`, cx + 9, ty + 50);

      ctx.font = `600 9px Inter, system-ui, sans-serif`;
      ctx.fillStyle = '#64748b';
      ctx.fillText(sub, cx + 9, ty + towerH - 22);
    } else {
      const barHMax = towerH - 44;
      const bh = Math.max(8, (Math.min(sp, maxSp) / maxSp) * barHMax);
      const barX = cx + cardW - 28;
      const barW = 12;
      roundRect(ctx, barX, ty + towerH - 12 - barHMax, barW, barHMax, 4);
      ctx.fillStyle = 'rgba(30, 41, 59, 0.7)';
      ctx.fill();
      roundRect(ctx, barX, ty + towerH - 12 - bh, barW, bh, 4);
      ctx.fillStyle = bVal === 1 ? '#22d3ee' : '#34d399';
      ctx.fill();

      ctx.textAlign = 'left';
      ctx.font = `700 11.5px "JetBrains Mono", monospace`;
      ctx.fillStyle = isSel ? '#67e8f9' : '#94a3b8';
      ctx.fillText(`B=${bVal.toLocaleString()}`, cx + 12, ty + 24);

      ctx.font = `800 24px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(`${sp.toFixed(2)}×`, cx + 12, ty + 64);

      ctx.font = `600 10.5px Inter, system-ui, sans-serif`;
      ctx.fillStyle = '#64748b';
      ctx.fillText(sub, cx + 12, ty + towerH - 14);
    }
  });
}

// ======================== ANIMATION LOOP ========================
let lastTs = performance.now();

function tick(now) {
  const dt = Math.min(0.05, (now - lastTs) / 1000) * STATE.speed;
  lastTs = now;

  // Animate active scene loops smoothly without ever auto-jumping to another scene
  if (STATE.playing) {
    // Scene 1: repeating pulse every 3.6s
    STATE.s1Pulse = (STATE.s1Pulse + dt / 3.6) % 1.08;

    // Scene 2: sweeping token rain
    if (STATE.s2Mode === 'rain') {
      STATE.s2Scan = (STATE.s2Scan + dt * 110) % 580;
    }

    // Scene 3: smooth 0 -> 3 magnetic morph (hold briefly at each stage)
    if (!STATE.s3UserLocked) {
      const cycle = ((now * 0.001 * STATE.speed) % 9.0) / 9.0; // 0..1 over 9s
      let targetM;
      if (cycle < 0.14) targetM = 0;
      else if (cycle < 0.34) targetM = (cycle - 0.14) / 0.20;
      else if (cycle < 0.44) targetM = 1;
      else if (cycle < 0.62) targetM = 1 + (cycle - 0.44) / 0.18;
      else if (cycle < 0.70) targetM = 2;
      else if (cycle < 0.88) targetM = 2 + (cycle - 0.70) / 0.18;
      else targetM = 3;
      STATE.s3Morph = targetM;
      const sl = $('#s3-slider');
      if (sl) sl.value = STATE.s3Morph.toFixed(2);
      const stIdx = Math.min(3, Math.round(STATE.s3Morph));
      $$('.s3-st-btn').forEach((b, i) => b.classList.toggle('active', i === stIdx));
      updateScene3HUDOnly();
    }

    // Scene 4: tile burst reveal + parallel timeline playhead
    STATE.s4Reveal = Math.min(1, STATE.s4Reveal + dt * 0.42);
    STATE.s4PipeT = (STATE.s4PipeT + dt * 0.22) % 1.0;

    // Scene 5: GPU race loop
    STATE.s5RaceT = (STATE.s5RaceT + dt * 0.30) % 1.28;
  }

  // Render visible scenes (in Cinema mode only active scene; in Scroll mode all 5!)
  if (STATE.mode === 'cinema') {
    if (STATE.scene === 0) renderScene1(now);
    else if (STATE.scene === 1) renderScene2();
    else if (STATE.scene === 2) renderScene3();
    else if (STATE.scene === 3) renderScene4();
    else if (STATE.scene === 4) renderScene5();
  } else {
    renderScene1(now);
    renderScene2();
    renderScene3();
    renderScene4();
    renderScene5();
  }

  requestAnimationFrame(tick);
}

function updateScene3HUDOnly() {
  const mFrac = Math.max(0, Math.min(1, STATE.s3Morph / 3));
  const ta = D.bench.codesign.track_a;
  const tb = D.bench.codesign.track_b;
  $('#s3-hv1').textContent = `${lerp(ta.x0_2048, tb.x0_2048, mFrac).toFixed(1)}%`;
  $('#s3-hv2').textContent = `${lerp(ta.p2_2048, tb.p2_2048, mFrac).toFixed(1)}%`;
  $('#s3-hv3').textContent = `${lerp(ta.dram_2048, tb.dram_2048, mFrac).toFixed(1)}%`;
}

// ======================== EVENT WIRING ========================
function init() {
  // Build Scene 1 example image chips
  const strip = $('#s1-ex-strip');
  D.examples.forEach((ex, idx) => {
    const btn = document.createElement('button');
    btn.className = 'ex-chip' + (idx === 0 ? ' active' : '');
    btn.innerHTML = `<img src="${ex.img}" alt="" width="24" height="24"><span class="ex-nm">${ex.ko}</span>`;
    btn.addEventListener('click', () => {
      STATE.exIdx = idx;
      STATE.s1Pulse = 0;
      $$('.ex-chip').forEach((c, i) => c.classList.toggle('active', i === idx));
      updateAllHUDs();
    });
    strip.appendChild(btn);
  });

  // Top nav & mode toggles
  $$('.nav-ch').forEach(btn => btn.addEventListener('click', () => setScene(Number(btn.dataset.ch), true)));
  $$('.scrub-seg').forEach(seg => seg.addEventListener('click', () => setScene(Number(seg.dataset.ch), true)));
  $('#btn-mode-cinema').addEventListener('click', () => setMode('cinema'));
  $('#btn-mode-scroll').addEventListener('click', () => setMode('scroll'));
  $('#btn-lang').addEventListener('click', () => {
    STATE.lang = STATE.lang === 'ko' ? 'en' : 'ko';
    syncTexts();
  });

  // Cinema playbar controls (Play/Pause current scene animation + Prev/Next manual chapter step)
  $('#btn-play').addEventListener('click', () => {
    STATE.playing = !STATE.playing;
    $('#btn-play-ico').textContent = STATE.playing ? '⏸' : '▶';
    $('#btn-play-lbl').textContent = STATE.playing ? T().pause : T().play;
  });
  $('#btn-prev')?.addEventListener('click', () => setScene((STATE.scene + 4) % 5, true));
  $('#btn-next')?.addEventListener('click', () => setScene((STATE.scene + 1) % 5, true));
  $$('.spd-btn').forEach(btn => btn.addEventListener('click', () => {
    STATE.speed = Number(btn.dataset.spd);
    $$('.spd-btn').forEach(b => b.classList.toggle('active', b === btn));
  }));
  $('#btn-fs').addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  });

  // Scene 1 controls
  $('#s1-replay').addEventListener('click', () => { STATE.s1Pulse = 0; });
  $$('.s1-dim-btn').forEach(btn => btn.addEventListener('click', () => {
    STATE.s1Scale = btn.dataset.dim;
    STATE.s1Pulse = 0;
    $$('.s1-dim-btn').forEach(b => b.classList.toggle('active', b === btn));
    updateAllHUDs();
  }));

  // Scene 2 controls + hover
  const s2Btns = { rain: $('#s2-btn-rain'), grid: $('#s2-btn-grid'), hover: $('#s2-btn-hover') };
  Object.entries(s2Btns).forEach(([k, btn]) => btn.addEventListener('click', () => {
    STATE.s2Mode = k;
    if (k === 'rain') STATE.s2Scan = 0;
    Object.entries(s2Btns).forEach(([kk, b]) => b.classList.toggle('active', kk === k));
  }));
  $('#cv-s2').addEventListener('mousemove', e => {
    const r = e.currentTarget.getBoundingClientRect();
    const my = e.clientY - r.top;
    if (my >= S2_BOX.y && my <= S2_BOX.y + S2_BOX.h) {
      STATE.s2HoverRow = Math.floor(((my - S2_BOX.y) / S2_BOX.h) * 512);
    }
  });

  // Scene 3 controls
  $('#s3-slider').addEventListener('input', e => {
    STATE.s3UserLocked = true;
    STATE.s3Morph = parseFloat(e.target.value);
    const stIdx = Math.min(3, Math.round(STATE.s3Morph));
    $$('.s3-st-btn').forEach((b, i) => b.classList.toggle('active', i === stIdx));
    updateScene3HUDOnly();
  });
  $$('.s3-st-btn').forEach(btn => btn.addEventListener('click', () => {
    STATE.s3UserLocked = true;
    STATE.s3Morph = Number(btn.dataset.st);
    $('#s3-slider').value = STATE.s3Morph.toFixed(2);
    $$('.s3-st-btn').forEach(b => b.classList.toggle('active', b === btn));
    updateScene3HUDOnly();
  }));

  // Scene 4 controls + tile hover/click
  $$('.s4-f-btn').forEach(btn => btn.addEventListener('click', () => {
    STATE.s4Filter = btn.dataset.f;
    STATE.s4Reveal = 1;
    $$('.s4-f-btn').forEach(b => b.classList.toggle('active', b === btn));
  }));
  $('#s4-replay').addEventListener('click', () => { STATE.s4Reveal = 0; });
  function pickTileAt(e) {
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    if (mx < S4_BOX.x || mx > S4_BOX.x + S4_BOX.w || my < S4_BOX.y || my > S4_BOX.y + S4_BOX.h) return -1;
    const c512 = Math.floor(((mx - S4_BOX.x) / S4_BOX.w) * 512);
    const r512 = Math.floor(((my - S4_BOX.y) / S4_BOX.h) * 512);
    const vis = getFilteredTiles();
    for (let i = vis.length - 1; i >= 0; i--) {
      if (tileContains(vis[i], r512, c512)) {
        return D.batch_scene.tiles.findIndex(x => x.id === vis[i].id);
      }
    }
    return -1;
  }
  $('#cv-s4').addEventListener('mousemove', e => { STATE.s4Hover = pickTileAt(e); });
  $('#cv-s4').addEventListener('mouseleave', () => { STATE.s4Hover = -1; });
  $('#cv-s4').addEventListener('click', e => {
    const idx = pickTileAt(e);
    if (idx >= 0) STATE.s4Selected = idx;
  });

  // Scene 5 controls + clickable speedup towers
  $('#s5-ovlp-btn').addEventListener('click', () => {
    STATE.s5Ovlp = true; STATE.s5RaceT = 0;
    $('#s5-ovlp-btn').classList.add('active'); $('#s5-ser-btn').classList.remove('active');
    updateAllHUDs();
  });
  $('#s5-ser-btn').addEventListener('click', () => {
    STATE.s5Ovlp = false; STATE.s5RaceT = 0;
    $('#s5-ser-btn').classList.add('active'); $('#s5-ovlp-btn').classList.remove('active');
    updateAllHUDs();
  });
  $$('.s5-dim-btn').forEach(btn => btn.addEventListener('click', () => {
    STATE.s5Scale = btn.dataset.dim; STATE.s5RaceT = 0;
    $$('.s5-dim-btn').forEach(b => b.classList.toggle('active', b === btn));
    updateAllHUDs();
  }));
  $('#s5-replay').addEventListener('click', () => { STATE.s5RaceT = 0; });
  $('#cv-s5').addEventListener('click', e => {
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    for (const tw of S5_TOWERS) {
      if (mx >= tw.x && mx <= tw.x + tw.w && my >= tw.y && my <= tw.y + tw.h) {
        STATE.s5Batch = tw.B;
        STATE.s5RaceT = 0;
        break;
      }
    }
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); $('#btn-play').click(); }
    else if (e.code === 'ArrowRight') setScene((STATE.scene + 1) % 5, true);
    else if (e.code === 'ArrowLeft') setScene((STATE.scene + 4) % 5, true);
    else if (e.key === 'l' || e.key === 'L') $('#btn-lang').click();
    else if (e.key === 'c' || e.key === 'C') setMode(STATE.mode === 'cinema' ? 'scroll' : 'cinema');
  });

  // Track active chapter cleanly while scrolling in Scroll mode
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      if (STATE.mode !== 'scroll') return;
      for (const ent of entries) {
        if (ent.isIntersecting) {
          const idx = ['sec-s1', 'sec-s2', 'sec-s3', 'sec-s4', 'sec-s5'].indexOf(ent.target.id);
          if (idx >= 0) {
            STATE.scene = idx;
            syncActiveSceneDOM();
          }
        }
      }
    }, { threshold: 0.45 });
    $$('.scene-sec').forEach(sec => io.observe(sec));
  }

  // Auto-switch to Scroll mode on narrow mobile screens for natural thumb scrolling!
  if (window.innerWidth < 760) setMode('scroll');
  else setMode('cinema');

  syncTexts();
  requestAnimationFrame(tick);
}

// Expose test hooks for automated QA inspection
window.__SHOWCASE__ = {
  STATE,
  setMode,
  setScene,
  setLang: (l) => { STATE.lang = l; syncTexts(); },
  setS2Mode: (m) => { STATE.s2Mode = m; STATE.s2Scan = 400; },
  setS3Morph: (v) => { STATE.s3UserLocked = true; STATE.s3Morph = v; $('#s3-slider').value = v; updateScene3HUDOnly(); },
  setS4Reveal: (v) => { STATE.s4Reveal = v; },
  setS5Batch: (b, ovlp = true) => { STATE.s5Batch = b; STATE.s5Ovlp = ovlp; STATE.s5RaceT = 0.72; updateAllHUDs(); },
  pause: () => {
    STATE.playing = false;
    $('#btn-play-ico').textContent = '▶';
    $('#btn-play-lbl').textContent = T().play;
  },
};

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
})();
"""
