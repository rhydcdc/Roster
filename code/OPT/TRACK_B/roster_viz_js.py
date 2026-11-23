"""Unified client-side JS for ROSTER_VIZ.html (all 4 tabs: Train, Infer, Batch, Contrib)."""

ROSTER_JS = r"""(() => {
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = 'http://www.w3.org/2000/svg';
const NB = 4, NN = 4096;
const C = {};
const SPR = {};
let D = null, B = null;

const S = {
  view: 'train',
  layer: 0, feat: 0, ex: 0, inStep: 0,
  t2layer: 0, t2sel: null,
  t5demo: 2,
  r6b: 1, r7: 9, t7mode: 0,
  p1: 1, step: 0, batch: '2048', cache: 2, strip: 0,
  p2mode: 0, p2tm: 1, p2b: '2048',
  c1mode: 1, c1b: 2048,
  c2b: '2048',
  c3mode: 3, c3c: 0, c3t: -1, c3sb: '2048',
};

// ---- Base64 / formatting helpers ----
const u8 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const i8 = s => new Int8Array(u8(s).buffer);
const unbits = (s, n = NN) => {
  const b = u8(s), o = new Uint8Array(n);
  for (let i = 0; i < n; i++) o[i] = (b[i >> 3] >> (7 - (i & 7))) & 1;
  return o;
};
const deq = q => Float32Array.from(i8(q.q), v => v / 127 * q.s);
const fmt = (v, d = 2) => Number(v).toFixed(d);
const pct = (v, d = 2) => fmt(v, d) + '%';
const nf = v => Number(v).toLocaleString('ko-KR');

function rgba(hex, a) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function hexRGB(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function fit(cv) {
  const r = cv.getBoundingClientRect();
  const d = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * d));
  const h = Math.max(1, Math.round(r.height * d));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const x = cv.getContext('2d');
  x.setTransform(1, 0, 0, 1, 0, 0);
  return { x, w, h, d };
}
function el(t, a = {}, p) {
  const e = document.createElementNS(NS, t);
  for (const k in a) e.setAttribute(k, a[k]);
  if (p) p.appendChild(e);
  return e;
}
function tx(p, x, y, s, a = {}) {
  const t = el('text', Object.assign({ x, y }, a), p);
  if (a['font-size'] != null) t.style.fontSize = a['font-size'] + 'px';
  if (a.fill != null) t.style.fill = a.fill;
  if (a['font-weight'] != null) t.style.fontWeight = a['font-weight'];
  t.textContent = s;
  return t;
}
function showTip(ev, text) {
  const tip = $('#tip');
  tip.hidden = false;
  tip.textContent = text;
  tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, ev.clientX + 12) + 'px';
  tip.style.top = Math.max(8, ev.clientY - 34) + 'px';
}
function hideTip() { $('#tip').hidden = true; }

// ---- Canvas primitives (64x64 neuron grid, 32x32 heatmap, 28x28 photo, sprite thumb) ----
function grid(cv, o) {
  const { x, w } = fit(cv);
  const n = 64, cs = w / n, pad = Math.max(0.5, cs * 0.12), rr = Math.max(0.6, cs * 0.22);
  x.fillStyle = C.bg; x.fillRect(0, 0, w, w);
  const br = o.bright ? u8(o.bright) : null, on = o.on || null, gone = o.gone || null;
  for (let i = 0; i < NN; i++) {
    const cx = (i % n) * cs, cy = Math.floor(i / n) * cs;
    let col;
    if (on && on[i]) { const b = br ? br[i] / 255 : 1; col = rgba(C.on, 0.5 + 0.5 * b); }
    else if (gone && gone[i]) col = rgba(C.pick, 0.78);
    else if (!on && br) { const b = br[i] / 255; col = b > 0.02 ? rgba(C.on, 0.08 + 0.92 * b) : C.off; }
    else col = C.off;
    x.fillStyle = col;
    x.beginPath();
    if (x.roundRect) x.roundRect(cx + pad, cy + pad, cs - 2 * pad, cs - 2 * pad, rr);
    else x.rect(cx + pad, cy + pad, cs - 2 * pad, cs - 2 * pad);
    x.fill();
  }
}
function heat(cv, vals) {
  cv.width = 32; cv.height = 32;
  const x = cv.getContext('2d'), im = x.createImageData(32, 32);
  let m = 0; for (const v of vals) m = Math.max(m, Math.abs(v));
  const on = hexRGB(C.on), ng = hexRGB(C.neg), bg = hexRGB(C.bg);
  for (let i = 0; i < 1024; i++) {
    const t = Math.min(1, Math.abs(vals[i]) / (m || 1)), c = vals[i] >= 0 ? on : ng, k = Math.pow(t, 0.7);
    im.data[i * 4] = bg[0] + (c[0] - bg[0]) * k;
    im.data[i * 4 + 1] = bg[1] + (c[1] - bg[1]) * k;
    im.data[i * 4 + 2] = bg[2] + (c[2] - bg[2]) * k;
    im.data[i * 4 + 3] = 255;
  }
  x.putImageData(im, 0, 0);
}
function photo(cv, b64img) {
  cv.width = 28; cv.height = 28;
  const x = cv.getContext('2d'), im = x.createImageData(28, 28), p = u8(b64img);
  for (let i = 0; i < 784; i++) {
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = p[i];
    im.data[i * 4 + 3] = 255;
  }
  x.putImageData(im, 0, 0);
}
function thumb(spr, idx, cls = '') {
  const s = SPR[spr], d = document.createElement('div');
  d.className = 'thumb ' + cls;
  const sz = cls === 's' ? 28 : cls === 'l' ? 64 : 42, k = sz / 28;
  d.style.backgroundImage = `url(${s.url})`;
  d.style.backgroundSize = `${s.cols * 28 * k}px auto`;
  d.style.backgroundPosition = `${-(idx % s.cols) * 28 * k}px ${-Math.floor(idx / s.cols) * 28 * k}px`;
  return d;
}

// ---- Reusable SVG line chart ----
function ticks(a, b, n, log) {
  if (log) {
    const o = [];
    for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) {
      const v = Math.pow(10, e);
      if (v >= a * 0.999 && v <= b * 1.001) o.push(v);
    }
    return o;
  }
  const st = Math.pow(10, Math.floor(Math.log10((b - a) / n)));
  const m = [1, 2, 2.5, 5, 10].map(k => k * st).find(s => (b - a) / s <= n) || st * 10;
  const o = [];
  for (let v = Math.ceil(a / m) * m; v <= b + 1e-9; v += m) o.push(+v.toFixed(10));
  return o;
}
function chart(host, o) {
  host.innerHTML = '';
  const W = Math.max(260, host.clientWidth || 400), H = o.h || 210, m = { l: 46, r: o.mr || 54, t: 14, b: 34 };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.aria || '' }, host);
  const [x0, x1] = o.x, [y0, y1] = o.y;
  const sx = o.xlog
    ? v => m.l + (Math.log(Math.max(v, x0)) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) * (W - m.l - m.r)
    : v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const sy = o.ylog
    ? v => H - m.b - (Math.log(Math.max(v, y0)) - Math.log(y0)) / (Math.log(y1) - Math.log(y0)) * (H - m.t - m.b)
    : v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const v of ticks(y0, y1, 4, o.ylog)) {
    el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: C.line, 'stroke-width': 1 }, svg);
    tx(svg, m.l - 8, sy(v) + 4, o.yf ? o.yf(v) : String(v), { 'text-anchor': 'end', fill: C.fg3, 'font-size': 10.5 });
  }
  for (const v of ticks(x0, x1, o.xn || 6, o.xlog)) {
    tx(svg, sx(v), H - m.b + 16, o.xf ? o.xf(v) : String(v), { 'text-anchor': 'middle', fill: C.fg3, 'font-size': 10.5 });
  }
  el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: C.line2, 'stroke-width': 1 }, svg);
  if (o.xlabel) tx(svg, W - m.r, H - 4, o.xlabel, { 'text-anchor': 'end', fill: C.fg3, 'font-size': 10 });
  (o.vlines || []).forEach(v => {
    el('line', { x1: sx(v.x), x2: sx(v.x), y1: m.t, y2: H - m.b, stroke: v.c || C.line2, 'stroke-width': v.w || 1, 'stroke-dasharray': v.dash || '' }, svg);
    if (v.t) {
      const rightSide = sx(v.x) < W - m.r - 60;
      tx(svg, rightSide ? sx(v.x) + 5 : sx(v.x) - 5, m.t + 10, v.t, {
        'text-anchor': rightSide ? 'start' : 'end', class: 'lab', fill: v.tc || C.fg2, 'font-size': 10.5,
      });
    }
  });
  const labs = [];
  (o.hlines || []).forEach(v => {
    el('line', { x1: m.l, x2: W - m.r, y1: sy(v.y), y2: sy(v.y), stroke: v.c || C.fg3, 'stroke-width': v.w || 1, 'stroke-dasharray': v.dash || '4 4' }, svg);
    if (v.t) {
      const t = tx(svg, W - m.r + 4, sy(v.y) + 4, v.t, { class: 'lab', fill: v.tc || C.fg2, 'font-size': 10.5 });
      labs.push(t);
    }
  });
  (o.bands || []).forEach(b => b.pts.forEach(p => {
    el('line', { x1: sx(p[0]), x2: sx(p[0]), y1: sy(p[1]), y2: sy(p[2]), stroke: b.c, 'stroke-width': 2, 'stroke-opacity': 0.7 }, svg);
    el('line', { x1: sx(p[0]) - 4, x2: sx(p[0]) + 4, y1: sy(p[1]), y2: sy(p[1]), stroke: b.c, 'stroke-width': 1.5 }, svg);
    el('line', { x1: sx(p[0]) - 4, x2: sx(p[0]) + 4, y1: sy(p[2]), y2: sy(p[2]), stroke: b.c, 'stroke-width': 1.5 }, svg);
  }));
  (o.series || []).forEach(s => {
    const d = s.x.map((v, i) => (i ? 'L' : 'M') + sx(v).toFixed(1) + ' ' + sy(s.y[i]).toFixed(1)).join('');
    el('path', { d, fill: 'none', stroke: s.c, 'stroke-width': s.w || 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash || '' }, svg);
    if (s.dots) s.x.forEach((v, i) => el('circle', { cx: sx(v), cy: sy(s.y[i]), r: 3.3, fill: s.c, stroke: C.bg2, 'stroke-width': 1.4 }, svg));
    if (s.label) {
      const i = s.x.length - 1;
      const t = tx(svg, sx(s.x[i]) + 7, sy(s.y[i]) + 4, s.label, { class: 'lab', fill: s.c, 'font-size': 10.5 });
      labs.push(t);
    }
  });
  labs.sort((a, b) => +a.getAttribute('y') - +b.getAttribute('y'));
  for (let i = 1; i < labs.length; i++) {
    const p = +labs[i - 1].getAttribute('y'), q = +labs[i].getAttribute('y');
    if (q - p < 14) labs[i].setAttribute('y', p + 14);
  }
  (o.marks || []).forEach(p => {
    el('circle', { cx: sx(p.x), cy: sy(p.y), r: p.r || 5, fill: p.c, stroke: C.bg2, 'stroke-width': 2 }, svg);
    if (p.t) tx(svg, sx(p.x) + (p.dx ?? 8), sy(p.y) + (p.dy ?? -8), p.t, { class: 'lab', fill: p.tc || p.c, 'text-anchor': p.anchor || 'start', 'font-size': 10.5, 'font-weight': 600 });
  });
  if (o.tip) {
    const ov = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    const ln = el('line', { y1: m.t, y2: H - m.b, stroke: C.fg3, 'stroke-width': 1, opacity: 0 }, svg);
    ov.addEventListener('pointermove', ev => {
      const rc = svg.getBoundingClientRect();
      const px = (ev.clientX - rc.left) * W / rc.width;
      const res = o.tip(px, sx);
      if (!res) return;
      ln.setAttribute('x1', res.px); ln.setAttribute('x2', res.px); ln.setAttribute('opacity', 0.6);
      showTip(ev, res.text);
    });
    ov.addEventListener('pointerleave', () => { hideTip(); ln.setAttribute('opacity', 0); });
  }
  return { sx, sy };
}
function nearestTip(xs, lab) {
  return (px, sx) => {
    let bi = 0, bd = 1e9;
    xs.forEach((v, i) => { const d = Math.abs(sx(v) - px); if (d < bd) { bd = d; bi = i; } });
    return { px: sx(xs[bi]), text: lab(bi) };
  };
}

// ---- Shared chips ----
function layerChips(host, get, set) {
  host.innerHTML = '';
  for (let l = 0; l < NB; l++) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'chip plain';
    b.textContent = `층 ${l + 1} (M=${D.Ms[l]})`;
    b.setAttribute('aria-pressed', get() === l);
    b.onclick = () => set(l);
    host.appendChild(b);
  }
}
function featChips(host) {
  host.innerHTML = '';
  D.layers[S.layer].feat.forEach((f, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'chip';
    b.setAttribute('aria-pressed', S.feat === i);
    b.appendChild(thumb('feat', (S.layer * 4 + i) * 8, 's'));
    const t = document.createElement('span');
    t.textContent = '#' + f.j + ' ' + D.classes[[7, 1, 8, 0][i]];
    b.appendChild(t);
    b.onclick = () => { S.feat = i; drawTrain(); };
    host.appendChild(b);
  });
}
function imgPicker(host, onPick) {
  host.innerHTML = '';
  D.examples.forEach((e, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('aria-pressed', S.ex === i);
    b.setAttribute('aria-label', D.classes[e.label]);
    const cv = document.createElement('canvas'); photo(cv, e.img);
    b.appendChild(cv);
    b.onclick = () => { S.ex = i; onPick(); };
    host.appendChild(b);
  });
}
function heroGrids(host, per) {
  host.innerHTML = '';
  per.forEach((s, l) => {
    const c = document.createElement('div');
    c.innerHTML = `<canvas class="cv-grid"></canvas><div class="gcap"><span>층 ${l + 1}</span><b>${(s.K / 40.96).toFixed(1)}%</b></div>`;
    host.appendChild(c);
    grid(c.querySelector('canvas'), { on: unbits(s.on) });
  });
}

// =========================================================================
// TAB 1: 학습 과정
// =========================================================================
function drawPretrain() {
  const p = D.pretrain, xs = p.epoch, drops = [];
  for (let i = 1; i < p.lr.length; i++) if (p.lr[i] < p.lr[i - 1]) drops.push({ x: xs[i] - 0.5, c: C.line2, dash: '3 3' });
  const lo = Math.floor(Math.min(...p.val)), hi = Math.ceil(Math.max(...p.val) + 0.3);
  const bestV = Math.max(...p.val);
  chart($('#ch-pretrain'), {
    x: [1, xs[xs.length - 1]], y: [lo, hi],
    series: [{ x: xs, y: p.val, c: C.con, label: '' }],
    vlines: drops, xlabel: '에폭', yf: v => v + '%',
    aria: '사전 학습의 검증 정확도',
    tip: nearestTip(xs, i => `${xs[i]}에폭  검증 ${fmt(p.val[i])}%  학습률 ${p.lr[i]}`),
    marks: [{ x: xs[p.val.indexOf(bestV)], y: bestV, c: C.con, t: '최고 검증 ' + fmt(bestV) + '%', anchor: 'end', dx: -8 }],
  });
  $('#pt-sum').textContent = `${xs.length}에폭 · 베이스 시험 ${fmt(p.test)}%`;
}

// ---- t2: k-means principle + real t-SNE (M=[32,32,32,256]) ----
const KM = { pts: [], cen: [], asg: [], phase: 0, it: 0, timer: null, anim: null };
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function d2(a, b) { const dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; }
function kmInit() {
  const r = rng(7), g = () => { let u = 0, v = 0; while (!u) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const blobs = [[0.25, 0.3], [0.7, 0.22], [0.5, 0.55], [0.2, 0.75], [0.78, 0.72], [0.55, 0.88]];
  KM.pts = [];
  for (let i = 0; i < 360; i++) {
    const b = blobs[i % 6];
    KM.pts.push([Math.min(0.97, Math.max(0.03, b[0] + g() * 0.075)), Math.min(0.97, Math.max(0.03, b[1] + g() * 0.075))]);
  }
  KM.cen = []; KM.asg = new Array(360).fill(-1); KM.phase = 0; KM.it = 0; KM.r = rng(11);
  kmSay(); kmDraw();
}
function kmStep() {
  if (KM.cen.length < 6) {
    if (!KM.cen.length) KM.cen.push([...KM.pts[Math.floor(KM.r() * 360)]]);
    else {
      const w = KM.pts.map(p => Math.min(...KM.cen.map(c => d2(p, c)))), s = w.reduce((a, b) => a + b, 0);
      let t = KM.r() * s, k = 0; while (t > w[k]) { t -= w[k]; k++; }
      KM.cen.push([...KM.pts[Math.min(k, 359)]]);
    }
    KM.phase = 1;
  } else if (KM.phase !== 2) {
    KM.asg = KM.pts.map(p => { let bi = 0, bd = 9; KM.cen.forEach((c, i) => { const d = d2(p, c); if (d < bd) { bd = d; bi = i; } }); return bi; });
    KM.phase = 2; KM.it++;
  } else {
    const nc = KM.cen.map((c, k) => {
      const m = KM.pts.filter((_, i) => KM.asg[i] === k);
      return m.length ? [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length] : c;
    });
    const from = KM.cen.map(c => [...c]);
    const moved = nc.some((c, i) => d2(c, from[i]) > 1e-8);
    kmTween(from, nc);
    KM.phase = moved ? 3 : 4;
  }
  kmSay(); if (KM.phase !== 3) kmDraw();
}
function kmTween(from, to) {
  const t0 = performance.now(), dur = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 420;
  cancelAnimationFrame(KM.anim);
  const f = now => {
    const t = Math.min(1, (now - t0) / dur), e = t * (2 - t);
    KM.cen = from.map((c, i) => [c[0] + (to[i][0] - c[0]) * e, c[1] + (to[i][1] - c[1]) * e]);
    kmDraw();
    if (t < 1) KM.anim = requestAnimationFrame(f);
  };
  KM.anim = requestAnimationFrame(f);
}
function kmSay() {
  const n = KM.cen.length, say = $('#km-say');
  if (!n) say.textContent = '점 360개를 중심점 6개로 나눈다. ‘한 걸음’을 누르거나 재생한다.';
  else if (n < 6 || KM.phase === 1) say.textContent = n === 1 ? '① 첫 중심점은 아무 점이나 고른다.' : `② 먼 점일수록 잘 뽑히게 다음 중심점을 찍는다 (k-평균++). ${n} / 6`;
  else if (KM.phase === 2) say.textContent = '③ 점마다 가장 가까운 중심점에 붙인다.';
  else if (KM.phase === 3) say.textContent = '④ 중심점을 붙은 점들의 평균 자리로 옮긴다. ③ → ④ 반복.';
  else say.textContent = '중심점이 멈췄다. 실제 모델: 1~3층은 32개, 4층은 256개로 나눈다.';
  $('#km-it').textContent = KM.it + '번';
}
function kmDraw() {
  const cv = $('#cv-km'), { x, w } = fit(cv);
  x.fillStyle = C.bg; x.fillRect(0, 0, w, w);
  if (KM.phase >= 2) {
    x.strokeStyle = rgba(C.fg3, 0.28); x.lineWidth = 1; x.beginPath();
    KM.pts.forEach((p, i) => { const c = KM.cen[KM.asg[i]]; if (!c) return; x.moveTo(p[0] * w, p[1] * w); x.lineTo(c[0] * w, c[1] * w); });
    x.stroke();
  }
  x.fillStyle = rgba(C.fg2, 0.75);
  KM.pts.forEach(p => { x.beginPath(); x.arc(p[0] * w, p[1] * w, w * 0.006, 0, 7); x.fill(); });
  KM.cen.forEach(c => {
    x.beginPath(); x.arc(c[0] * w, c[1] * w, w * 0.022, 0, 7);
    x.fillStyle = C.pick; x.fill(); x.lineWidth = w * 0.006; x.strokeStyle = C.bg; x.stroke();
  });
}
function kmPlay() {
  if (KM.timer) { clearInterval(KM.timer); KM.timer = null; $('#km-play').textContent = '재생'; return; }
  if (KM.phase === 4) kmInit();
  $('#km-play').textContent = '멈춤';
  KM.timer = setInterval(() => {
    kmStep();
    if (KM.phase === 4) { clearInterval(KM.timer); KM.timer = null; $('#km-play').textContent = '재생'; }
  }, 650);
}
function drawTsne() {
  const L = D.layers[S.t2layer], cv = $('#cv-tsne'), { x, w } = fit(cv), P = L.pts, sel = S.t2sel;
  $('#t2-real-title').textContent = `실제: 층 ${S.t2layer + 1} 입력 3,000개와 중심점 ${L.M}개`;
  x.fillStyle = C.bg; x.fillRect(0, 0, w, w);
  const pad = w * 0.04, X = v => pad + v * (w - 2 * pad);
  for (let i = 0; i < P.x.length; i++) {
    const hit = sel !== null && P.key[i] === sel;
    x.fillStyle = hit ? C.on : rgba(C.fg3, 0.45);
    x.beginPath(); x.arc(X(P.x[i]), X(P.y[i]), hit ? w * 0.0055 : w * 0.0032, 0, 7); x.fill();
  }
  const radBase = L.M <= 32 ? 0.013 : 0.0065;
  L.cen.x.forEach((cx, j) => {
    if (cx === null) return;
    const s = j === sel;
    x.beginPath(); x.arc(X(cx), X(L.cen.y[j]), s ? w * 0.019 : w * radBase, 0, 7);
    x.lineWidth = s ? w * 0.005 : w * 0.003; x.strokeStyle = s ? C.pick : rgba(C.pick, 0.78);
    if (s) { x.fillStyle = rgba(C.pick, 0.28); x.fill(); }
    x.stroke();
  });
  const info = $('#t2-info'); info.innerHTML = '';
  if (sel === null) {
    const p = document.createElement('p'); p.style.cssText = 'font-size:12.5px;color:var(--fg-3)';
    p.textContent = '호박색 고리가 중심점이다. 하나를 누르면 그 자리로 모이는 실제 사진이 보인다.';
    info.appendChild(p); return;
  }
  const fi = L.feat.findIndex(f => f.j === sel), box = document.createElement('div'); box.className = 'thumbs';
  if (fi >= 0) for (let k = 0; k < 8; k++) box.appendChild(thumb('feat', (S.t2layer * 4 + fi) * 8 + k));
  else box.appendChild(thumb('rep' + S.t2layer, sel, 'l'));
  const t = document.createElement('div'); t.style.cssText = 'font-size:12.5px;color:var(--fg-2);min-width:0;flex:1';
  t.innerHTML = `<div class="mono" style="color:var(--pick)">층 ${S.t2layer + 1} (M=${L.M}) · 중심점 #${sel}</div><div>학습 사진 55,000장 가운데 <span class="mono">${nf(L.count[sel])}</span>장이 가장 가깝다 (평균 ${nf(Math.round(55000 / L.M))}장/중심점).</div><div style="color:var(--fg-3)">${fi >= 0 ? '가장 가까운 실제 사진 8장' : '대표 사진 1장'}</div>`;
  info.appendChild(t); info.appendChild(box);
}
function tsneClick(ev) {
  const L = D.layers[S.t2layer], cv = $('#cv-tsne'), r = cv.getBoundingClientRect(), w = r.width, pad = w * 0.04;
  const px = (ev.clientX - r.left - pad) / (w - 2 * pad), py = (ev.clientY - r.top - pad) / (w - 2 * pad);
  let bj = null, bd = 1e9;
  L.cen.x.forEach((cx, j) => { if (cx === null) return; const d = (cx - px) ** 2 + (L.cen.y[j] - py) ** 2; if (d < bd) { bd = d; bj = j; } });
  S.t2sel = bj; drawTsne();
}
function t2pick(l) {
  S.t2layer = l; S.t2sel = D.layers[l].feat[0].j;
  layerChips($('#t2-layers'), () => S.t2layer, t2pick);
  drawTsne();
}

// ---- t3: neuron list ----
function drawT3() {
  const L = D.layers[S.layer], f = L.feat[S.feat];
  layerChips($('#t3-layers'), () => S.layer, l => { S.layer = l; drawTrain(); });
  featChips($('#t3-feats'));
  $('#t3-title').textContent = `층 ${S.layer + 1} (M=${L.M}) · 중심점 #${f.j}의 뉴런 4,096개`;
  grid($('#cv-t3'), { bright: f.grid, on: unbits(f.on) });
  const sx = f.sorted.x, sy = f.sorted.y;
  chart($('#ch-t3'), {
    x: [1, 4096], xlog: true, y: [0, 1],
    series: [{ x: sx, y: sy, c: C.con }],
    vlines: [{ x: f.K0, c: C.cpick, dash: '4 3', t: `초기 K=${nf(f.K0)}`, tc: C.cpick }],
    xlabel: '목록 순위 (로그)', yf: v => Math.round(v * 100) + '%',
    aria: '목록 순서대로 놓은 더하는 크기',
    tip: nearestTip(sx, i => `${sx[i]}번째 뉴런  최대 뉴런의 ${Math.round(sy[i] * 100)}%`),
  });
  const y256 = sy[sx.findIndex(v => v >= 256)], yK0 = sy[Math.max(0, sx.findIndex(v => v >= f.K0))];
  $('#t3-say').textContent = `최대 뉴런을 100%로 두면 256번째는 ${Math.round(y256 * 100)}%, 초기 K(${nf(f.K0)}번째)는 ${Math.round(yK0 * 100)}%만 더한다. 학습 사진 ${nf(f.n)}장이 이 중심점에 모인다.`;
  const th = $('#t3-thumbs'); th.innerHTML = '';
  for (let k = 0; k < 8; k++) th.appendChild(thumb('feat', (S.layer * 4 + S.feat) * 8 + k, 's'));
}

// ---- t4: selection op ----
function drawT4() {
  imgPicker($('#t4-imgs'), drawTrain);
  const e = D.examples[S.ex], host = $('#t4-flow'); host.innerHTML = '';
  e.init.forEach((s, l) => {
    const c = document.createElement('div'); c.className = 'cell';
    c.innerHTML = `<div class="lbl">층 ${l + 1} (M=${D.Ms[l]})</div><div class="row" style="gap:8px;flex-wrap:nowrap"></div><canvas class="cv-grid"></canvas><div class="gcap"><span>#${s.j}</span><b>${nf(s.K)}개 (${(s.K / 40.96).toFixed(0)}%)</b></div>`;
    const r = c.querySelector('.row'); r.appendChild(thumb('rep' + l, s.j, 's'));
    const t = document.createElement('span'); t.style.cssText = 'font-size:12px;color:var(--fg-2)'; t.textContent = '가장 가까운 중심점';
    r.appendChild(t); host.appendChild(c);
    grid(c.querySelector('canvas'), { on: unbits(s.on) });
  });
}

// ---- t5: 1-round post-training + NEW Interactive Example #1 (Individual vs B1 Family vs B2 32-col Block) ----
function featOf(r) { return D.rounds[r].feat[S.layer][S.feat]; }
const at = (c, k) => { let i = c.x.findIndex(v => v >= k); if (i < 0) i = c.x.length - 1; return c.y[i]; };
const p1 = v => Math.round(v * 1000) / 10 + '%';
function moveWord(b, a) { return a - b > 0.002 ? '늘었다' : a - b < -0.002 ? '줄었다' : '거의 그대로다'; }

const B1B2_CASES = [
  {
    lbl: '① 개인 순위만 몰기',
    cap: '중심점마다 제각각 4개씩 켬 → 4블록 전부 열려 DRAM 75% 읽기 · 타일 ×0 50%',
    mat: [
      [1, 0, 1, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0],
      [0, 1, 0, 0,  1, 0, 1, 0,  0, 0, 0, 0,  0, 1, 0, 0],
      [1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 1,  0, 0, 0, 0],
      [0, 0, 1, 0,  1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0],
    ],
    kpi: ['열린 뉴런 열: 12 / 16 (4블록 전부 사용)', '4×4 틀 채움 50% (×0 50%) · 빈 블록 0 / 4'],
    col: 'pick',
  },
  {
    lbl: '② + 가족 몰기 (B1)',
    cap: '같은 가족(4 중심점)이 공통 열로 모임 → 열린 열 12→7개, 하지만 블록 3에 1열 자투리 잔존',
    mat: [
      [1, 1, 1, 0,  1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 1, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 1, 1, 0,  0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 0, 1, 0,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0],
    ],
    kpi: ['열린 뉴런 열: 6 / 16 (블록 3에 자투리 1열)', '4×4 틀 채움 76% (×0 24%) · 빈 블록 1 / 4'],
    col: 'neg',
  },
  {
    lbl: '③ + 32열 블록 몰기 (B2 · 최종)',
    cap: '32열 블록 단위로 통째로 비워 블록 3·4 완전 OFF → ×0 0%, DRAM 50% 직감!',
    mat: [
      [1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 1, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
      [1, 1, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
    ],
    kpi: ['열린 뉴런 열: 6 / 16 (블록 3·4 완전 빈 블록!)', '실제 CH2 효과: ×0 12.6% · P2 16.1% · DRAM 64.5%'],
    col: 'on',
  },
];

function drawT5Demo() {
  const btns = $('#t5-b1b2-btns'); btns.innerHTML = '';
  B1B2_CASES.forEach((c, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = c.lbl; b.setAttribute('aria-pressed', S.t5demo === idx);
    b.onclick = () => { S.t5demo = idx; drawT5Demo(); };
    btns.appendChild(b);
  });
  const cur = B1B2_CASES[S.t5demo];
  $('#t5-b1b2-cap').textContent = cur.cap;

  const host = $('#t5-b1b2-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 560);
  const lx = 58, rx = 10, top = 36, bgap = 8;
  const cs = Math.min(28, Math.floor((W - lx - rx - 3 * bgap) / 16));
  const bx = col => lx + col * cs + Math.floor(col / 4) * bgap;
  const H = top + 4 * cs + 36;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '개인 몰기 vs B1 가족 몰기 vs B2 32열 블록 몰기 비교' }, host);

  // Block headers & empty-block highlights
  for (let bk = 0; bk < 4; bk++) {
    let anyOn = false;
    for (let r = 0; r < 4; r++) for (let dc = 0; dc < 4; dc++) if (cur.mat[r][bk * 4 + dc]) anyOn = true;
    const x0 = bx(bk * 4) - 2, bw = 4 * cs + 4;
    el('rect', {
      x: x0, y: top - 2, width: bw, height: 4 * cs + 4, rx: 5,
      fill: anyOn ? 'transparent' : 'rgba(87,213,231,0.04)',
      stroke: anyOn ? (S.t5demo === 2 ? C.on : C.line2) : C.line,
      'stroke-width': anyOn ? 1.2 : 1,
      'stroke-dasharray': anyOn ? '' : '3 3',
    }, svg);
    tx(svg, x0 + bw / 2, top - 11, anyOn ? `블록 ${bk + 1}` : `블록 ${bk + 1} (빈 블록!)`, {
      'text-anchor': 'middle', fill: anyOn ? C.fg2 : C.on, 'font-size': W < 470 ? 8.5 : 9.8, 'font-weight': anyOn ? 500 : 700,
    });
  }

  const rNames = ['중심 C1', '중심 C2', '중심 C3', '중심 C4'];
  for (let r = 0; r < 4; r++) {
    tx(svg, lx - 8, top + r * cs + cs / 2 + 4, rNames[r], { 'text-anchor': 'end', fill: C.fg2, 'font-size': 10.5 });
    for (let c = 0; c < 16; c++) {
      const on = cur.mat[r][c] === 1;
      // Check if column is active in any row (needed in P2 union / tile pad)
      let colAny = false;
      for (let rr = 0; rr < 4; rr++) if (cur.mat[rr][c]) colAny = true;
      const x = bx(c) + 1.5, y = top + r * cs + 1.5, s = cs - 3;
      let fill = C.off, txt = '', tfill = C.bg;
      if (on) { fill = C.on; }
      else if (colAny && S.t5demo < 2) { fill = C.pick; txt = '×0'; }
      else if (colAny && S.t5demo === 2) { fill = C.bg3; }
      el('rect', { x, y, width: s, height: s, rx: 3, fill }, svg);
      if (txt && cs >= 18) tx(svg, bx(c) + cs / 2, top + r * cs + cs / 2 + 3.5, txt, { 'text-anchor': 'middle', fill: tfill, 'font-size': 8.2, 'font-weight': 700 });
    }
  }
  const foot = S.t5demo === 0
    ? '중심점마다 따로 앞쪽을 밀면 같은 가족이어도 12열이 흩어져 ×0과 DRAM 읽기가 큼'
    : S.t5demo === 1
      ? '가족 순위(α=0.35)로 공통 열에 뭉치지만 블록 3에 남은 1열이 새 블록을 열어 버림'
      : '32열 블록 순위(β=0.25) + Group-L2로 블록 3·4를 통째로 비워 DRAM·P2·×0 동시 절감!';
  tx(svg, lx + (bx(15) + cs - lx) / 2, top + 4 * cs + 22, W < 470 ? cur.kpi[0] : foot, {
    'text-anchor': 'middle', fill: S.t5demo === 2 ? C.on : (S.t5demo === 1 ? C.neg : C.pick), 'font-size': W < 470 ? 9 : 10.2, 'font-weight': 600,
  });

  $('#t5-b1b2-kpi').innerHTML =
    `<div class="panel"><div class="k">예시 (가족 4중심점 × 4블록)</div><div class="v ${cur.col}" style="font-size:14px">${cur.kpi[0]}</div></div>
     <div class="panel"><div class="k">타일 채움 · DRAM 영향</div><div class="v" style="font-size:14px">${cur.kpi[1]}</div></div>`;
}

function drawT5() {
  const r = 1, f = featOf(r);
  layerChips($('#t5-layers'), () => S.layer, l => { S.layer = l; drawTrain(); });
  featChips($('#t5-feats'));
  if (f) {
    const bx = f.before.x, after = f.after, k = Math.round(f.Kold / 2);
    chart($('#ch-t5'), {
      x: [1, f.Kold], y: [0, 1],
      series: [{ x: bx, y: f.before.y, c: C.cneg, w: 1.8, dash: '5 4' }, { x: after.x, y: after.y, c: C.con }],
      mr: 24, vlines: [{ x: k, c: C.fg3, dash: '4 3', t: '앞 절반' }],
      xlabel: '목록의 앞 k개', yf: v => Math.round(v * 100) + '%',
      aria: '사후 학습 전후 누적 몫',
      tip: nearestTip(bx, i => `앞 ${bx[i]}개  전 ${Math.round(f.before.y[i] * 100)}%  뒤 ${Math.round(after.y[i] * 100)}%`),
    });
    const b0 = at(f.before, k), a0 = at(after, k), mb = f.match_before, ma = f.match_after;
    $('#t5-kv').innerHTML = `<dt>층 · 중심점</dt><dd>층 ${S.layer + 1} · #${f.j} (학습 사진 ${nf(f.n)}장)</dd><dt>K (고정)</dt><dd>${nf(f.Kold)}개 → 사후 학습 뒤 ${nf(f.Knew)}개</dd><dt>앞 절반(${nf(k)}개)의 몫</dt><dd>${p1(b0)} → <span class="${a0 >= b0 ? 'ok' : 'no-ok'}">${p1(a0)}</span></dd><dt>출력 맞추기 오차</dt><dd>${fmt(mb, 4)} → <span class="${ma <= mb ? 'ok' : 'no-ok'}">${fmt(ma, 4)}</span></dd>`;
    const w = moveWord(b0, a0);
    $('#t5-say').textContent = w === '늘었다'
      ? `1바퀴 공유 사후 학습(α=0.35, β=0.25) 뒤 앞 절반 몫이 ${p1(b0)}에서 ${p1(a0)}로 늘었다. 가족과 32열 블록 단위로 함께 앞쪽에 몰린 덕분이다.`
      : `이 중심점에서는 앞 절반 몫이 ${w}(${p1(b0)} → ${p1(a0)}). 다른 중심점과 층도 눌러 비교해 보라.`;
  }
  drawT5Demo();
}

// ---- t6: K pruning ----
function drawT6() {
  const r = 1, f = featOf(r), R = D.rounds[r];
  $('#rd-6v').textContent = `${r}바퀴 · 허용 오차 ε = ${R.eps}`;
  layerChips($('#t6-layers'), () => S.layer, l => { S.layer = l; drawTrain(); });
  featChips($('#t6-feats'));
  if (!f || !f.ek) return;
  const ks = f.ek.k, es = f.ek.e.map(v => Math.max(v, 1e-5));
  chart($('#ch-t6'), {
    x: [ks[0], Math.max(ks[ks.length - 1], ks[0] + 1)], y: [1e-5, 1], ylog: true,
    series: [{ x: ks, y: es, c: C.con }],
    hlines: [{ y: R.eps, c: C.cpick, dash: '4 3', t: 'ε ' + R.eps, tc: C.cpick }],
    vlines: [{ x: f.Knew, c: C.cpick, w: 1.5, t: '새 K ' + nf(f.Knew), tc: C.cpick }],
    xlabel: '앞 k개만 켬', yf: v => v >= 1 ? '1' : v.toExponential(0), mr: 64,
    aria: '출력 오차 곡선',
    tip: nearestTip(ks, i => `앞 ${ks[i]}개  e(k) = ${f.ek.e[i]}`),
  });
  const on0 = unbits(f.on_old), on1 = unbits(f.on_new), gone = on0.map((v, i) => v && !on1[i] ? 1 : 0);
  grid($('#cv-t6'), { bright: f.grid, on: on1, gone });
  $('#t6-title').textContent = `층 ${S.layer + 1} · 중심점 #${f.j}`;
  $('#t6-k').textContent = `K ${nf(f.Kold)} → ${nf(f.Knew)}`;
  const ok = R.val >= D.final.val_thr;
  $('#t6-say').innerHTML = `허용 오차 ε = ${R.eps}로 모든 중심점·층의 K를 줄이면 검증 정확도는 <span class="${ok ? 'ok' : 'no-ok'}">${fmt(R.val)}%</span>(문턱 ${fmt(D.final.val_thr)}% ${ok ? '이상' : '미만'}), 평균 활성 뉴런은 <span class="mono">${pct(R.avg)}</span>로 줄어든다.`;
}

// ---- t6b: 0..9 rounds staircase ----
function drawT6b() {
  const r = S.r6b, R = D.rounds[r], f = featOf(r);
  $('#rd-6b').value = r; $('#rd-6bv').textContent = `${r}바퀴`;
  layerChips($('#t6b-layers'), () => S.layer, l => { S.layer = l; drawTrain(); });
  featChips($('#t6b-feats'));
  const xs = [-1], ys = [D.K_init];
  D.rounds.forEach((q, i) => { const g = q.feat[S.layer][S.feat]; xs.push(i); ys.push(g ? g.Knew : ys[ys.length - 1]); });
  const cur = xs.indexOf(r);
  chart($('#ch-t6b'), {
    x: [-1, 9], xn: 11, y: [0, Math.max(...ys) * 1.08],
    series: [{ x: xs, y: ys, c: C.con, dots: true }],
    marks: [{ x: r, y: ys[cur], c: C.cpick, r: 6, t: nf(ys[cur]) + '개', dx: r >= 7 ? -10 : 10, dy: r >= 7 ? -12 : -10, anchor: r >= 7 ? 'end' : 'start' }],
    xf: v => v === -1 ? '베이스' : String(v), xlabel: '바퀴',
    yf: v => v >= 1000 ? (v / 1000) + 'k' : String(v),
    aria: '0~9바퀴 중심점 K 추이',
    tip: nearestTip(xs, i => `${xs[i] === -1 ? '순수 베이스' : xs[i] + '바퀴 뒤'}  K = ${nf(ys[i])}개`),
  });
  if (!f) return;
  const on0 = unbits(f.on_old), on1 = unbits(f.on_new), gone = on0.map((v, i) => v && !on1[i] ? 1 : 0);
  grid($('#cv-t6b'), { bright: f.grid, on: on1, gone });
  $('#t6b-title').textContent = `층 ${S.layer + 1} · 중심점 #${f.j}`;
  $('#t6b-k').textContent = `K ${nf(f.Kold)} → ${nf(f.Knew)}`;
  $('#t6b-say').textContent = r === 0
    ? `0바퀴: 순수 베이스(4,096)에서 사후 학습 없이 ε=${R.eps} 잘라내기로 K ${nf(f.Kold)} → ${nf(f.Knew)} (검증 ${fmt(R.val)}%, 평균 ${pct(R.avg)}).`
    : `${r}바퀴: lr=${R.lr}, λ_blk=${R.lam_blk}, ε=${R.eps} → K ${nf(f.Kold)} → ${nf(f.Knew)} (검증 ${fmt(R.val)}%, 평균 ${pct(R.avg)}, B=2,048 DRAM ${fmt(R.dram, 1)}%).`;
}

// ---- t7: 0..9 rounds summary ----
function drawT7() {
  const rs = D.rounds, xs = rs.map(r => r.round);
  const ser = [0, 1, 2, 3].map(l => ({ x: xs, y: rs.map(r => r.layers[l]), c: C['l' + (l + 1)], label: '층 ' + (l + 1), dots: true }));
  ser.push({ x: xs, y: rs.map(r => r.avg), c: C.fg, w: 2.5, label: '평균', dots: true });
  chart($('#ch-t7a'), {
    x: [0, 9], xn: 10, y: [0, 82],
    series: ser, xlabel: '바퀴', yf: v => v + '%', mr: 56,
    aria: '0~9바퀴 층별 활성 뉴런 비율',
    tip: nearestTip(xs, i => `${xs[i]}바퀴  평균 ${pct(rs[i].avg)}  층별 ${rs[i].layers.map(v => fmt(v, 1)).join(' / ')}%`),
    vlines: [{ x: S.r7, c: C.cpick, dash: '3 3' }],
  });

  // Right chart toggle: accuracy vs DRAM/x0
  const tbtn = $('#t7-mode-btns'); tbtn.innerHTML = '';
  ['검증 · 최종 시험 정확도', 'DRAM 읽기 & ×0 추이'].forEach((lbl, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.t7mode === idx);
    b.onclick = () => { S.t7mode = idx; drawT7(); };
    tbtn.appendChild(b);
  });

  if (S.t7mode === 0) {
    chart($('#ch-t7b'), {
      x: [0, 9], xn: 10, y: [90.1, 91.4],
      series: [{ x: xs, y: rs.map(r => r.val), c: C.con, dots: true }],
      bands: [{ c: C.cpick, pts: [[9, D.final.ci[0], D.final.ci[1]]] }],
      marks: [{ x: 9, y: D.final.test, c: C.cpick, r: 5.5, t: `시험 ${fmt(D.final.test)}%`, dx: -10, dy: 16, anchor: 'end', tc: C.cpick }],
      hlines: [
        { y: 91.02, t: '베이스 91.02', c: C.fg3 },
        { y: 90.96, t: '문턱 90.96', c: C.con, tc: C.con, dash: '3 3' },
        { y: 90.52, t: '통과 90.52', c: C.cpick, tc: C.cpick },
      ],
      xlabel: '바퀴', yf: v => fmt(v, 1), mr: 82,
      aria: '0~9바퀴 검증 정확도와 9바퀴 최종 시험 정확도',
      tip: nearestTip(xs, i => `${xs[i]}바퀴  검증 ${pct(rs[i].val)}${rs[i].test_acc ? ` · 최종 10k 시험 ${pct(rs[i].test_acc)}` : ''}`),
      vlines: [{ x: S.r7, c: C.cpick, dash: '3 3' }],
    });
  } else {
    chart($('#ch-t7b'), {
      x: [0, 9], xn: 10, y: [0, 100],
      series: [
        { x: xs, y: rs.map(r => r.dram), c: C.fg2, dots: true, label: 'DRAM' },
        { x: xs, y: rs.map(r => r.ffn), c: C.con, dots: true, label: 'FFN' },
        { x: xs, y: rs.map(r => r.x0), c: C.cpick, dots: true, label: '×0' },
      ],
      xlabel: '바퀴 (B=2,048 검증)', yf: v => v + '%', mr: 58,
      aria: '0~9바퀴 B=2048 DRAM 읽기, FFN 연산량, x0 비율 추이',
      tip: nearestTip(xs, i => `${xs[i]}바퀴  DRAM ${fmt(rs[i].dram, 1)}% · FFN ${fmt(rs[i].ffn, 1)}% · ×0 ${fmt(rs[i].x0, 1)}%`),
      vlines: [{ x: S.r7, c: C.cpick, dash: '3 3' }],
    });
  }

  $('#rd-7').value = S.r7;
  $('#rd-7v').textContent = `${S.r7}바퀴 · K ${pct(rs[S.r7].avg)}`;
  const e = D.examples[S.ex]; photo($('#t7-img'), e.img);
  heroGrids($('#t7-grids'), e.rounds[S.r7]);
}

function drawTrain() { drawT3(); drawT4(); drawT5(); drawT6(); drawT6b(); drawT7(); }
function drawAllTrain() {
  drawPretrain(); kmDraw();
  layerChips($('#t2-layers'), () => S.t2layer, t2pick);
  drawTsne(); drawTrain();
}

// =========================================================================
// TAB 2: 추론 과정 (CH2_M32_B1B2_r9, M=[32,32,32,256], 1*=0%)
// =========================================================================
const STEPS = ['층 입력', '가장 가까운 중심점', '목록의 앞 K개', '켠 K개만 계산 (돋보기)', '출력을 더한다'];
function card(l, k, inner) {
  const d = document.createElement('div'); d.className = 'pcard'; d.dataset.k = l * 5 + k;
  d.innerHTML = `<div class="t"><span>${'①②③④⑤'[k]}</span>${STEPS[k]}</div>` + inner;
  return d;
}
function spark(cv, v) {
  const { x, w, h } = fit(cv); x.fillStyle = C.bg; x.fillRect(0, 0, w, h);
  let m = 0; for (const a of v) m = Math.max(m, Math.abs(a));
  const bw = w / v.length, mid = h / 2;
  for (let i = 0; i < v.length; i++) {
    const a = v[i] / (m || 1); x.fillStyle = a >= 0 ? C.on : C.neg;
    const hh = Math.max(0.5, Math.abs(a) * (mid - 1));
    x.fillRect(i * bw, a >= 0 ? mid - hh : mid, Math.max(1, bw), hh);
  }
}
function probChart(e) {
  const host = $('#ch-prob'); host.innerHTML = '';
  const W = Math.max(280, host.clientWidth || 420), rh = 22, H = rh * 10 + 24, lx = 64, rx = 62;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '부류별 확률' }, host);
  const sx = v => lx + v * (W - lx - rx);
  D.classes.forEach((c, i) => {
    const y = 10 + i * rh;
    tx(svg, lx - 8, y + 10, c, { 'text-anchor': 'end', class: 'lab', fill: C.fg2, 'font-size': 11 });
    el('rect', { x: lx, y: y + 1, width: Math.max(0, sx(e.base_probs[i]) - lx), height: 6, rx: 2, fill: C.fg3 }, svg);
    el('rect', { x: lx, y: y + 8, width: Math.max(0, sx(e.probs[i]) - lx), height: 8, rx: 2, fill: C.con }, svg);
    if (e.probs[i] > 0.01) tx(svg, sx(e.probs[i]) + 6, y + 15, fmt(e.probs[i] * 100, 1) + '%', { fill: C.fg2, 'font-size': 10.5 });
  });
  tx(svg, lx, H - 2, '굵은 청록 = CH2 고른 뉴런만 계산 · 가는 회색 = 베이스 풀 모델', { fill: C.fg3, 'font-size': 10 });
}
function markStep() {
  const n = 21; S.inStep = Math.max(0, Math.min(n - 1, S.inStep));
  document.querySelectorAll('.pcard').forEach(c => {
    const k = +c.dataset.k;
    c.classList.toggle('cur', k === S.inStep);
    c.classList.toggle('later', k > S.inStep);
  });
  const l = Math.floor(S.inStep / 5), k = S.inStep % 5, e = D.examples[S.ex];
  const stL = e.steps[l];
  const say = S.inStep === 20
    ? '<b>⑥ 예측</b> · 마지막 층 출력을 출력층에 넣어 부류 10개의 확률을 낸다 (1*=0%, 배치 크기 무관 동일 출력).'
    : `<b>층 ${l + 1} · ${'①②③④⑤'[k]} ${STEPS[k]}</b> · ` + [
      l === 0 ? '사진 784칸을 입력층에 넣어 1,024칸 층 입력을 만든다.' : '앞 층까지의 결과가 이 층의 입력이다.',
      `정규화한 층 입력과 중심점 ${stL.M}개의 거리를 재서 가장 가까운 #${stL.j}를 고른다. 학습된 라우터가 아니다.`,
      `#${stL.j}의 목록에서 앞 ${nf(stL.K)}개만 켠다. 나머지 ${nf(4096 - stL.K)}개는 계산하지 않는다.`,
      `켠 뉴런 ${nf(stL.K)}개만 gate·up에서 꺼내 계산한다(a = silu(g) × u). 돋보기는 그중 가장 큰 뉴런 하나.`,
      'down의 켠 열만 꺼내 a와 곱해 출력 y를 만들고 h에 더해 다음 층으로 보낸다.',
    ][k];
  $('#in-where').innerHTML = say + ` <span class="mono" style="color:var(--fg-3)">(${S.inStep + 1} / ${n})</span>`;
}
function goStep(d) {
  S.inStep += d; markStep();
  const c = document.querySelector(`.pcard[data-k="${S.inStep}"]`);
  if (c) c.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
}
function drawInfer() {
  imgPicker($('#in-imgs'), () => drawInfer());
  const e = D.examples[S.ex], host = $('#in-layers'); host.innerHTML = '';
  e.steps.forEach((s, l) => {
    const sec = document.createElement('section'); sec.className = 'layer';
    sec.innerHTML = `<header><h3>층 ${l + 1} <span class="lbl" style="margin-left:6px">M=${s.M}</span></h3><span class="lbl">중심점 #${s.j} · K = ${nf(s.K)}개 (${(s.K / 40.96).toFixed(1)}%) · 가중치도 이만큼만 꺼낸다</span></header><div class="pipe"></div>`;
    const P = sec.querySelector('.pipe');
    const c1 = card(l, 0, `<canvas class="hm"></canvas><div class="s">1,024칸 · 크기 ‖h‖ = <span class="mono">${fmt(s.hn, 1)}</span></div><div class="s">${l === 0 ? '사진 784칸 → 1,024칸 입력층' : '앞 층까지의 결과'}</div>`);
    heat(c1.querySelector('canvas'), deq(s.h)); P.appendChild(c1);

    const dmax = s.dist_top[s.dist_top.length - 1].d, dmin = s.dist_top[0].d;
    const rows = s.dist_top.map((t, i) => `<div class="r ${i === 0 ? 'win' : ''}"><span>#${t.j}</span><i style="width:${Math.max(8, 100 * (t.d - dmin * 0.85) / (dmax - dmin * 0.85 || 1))}%"></i><span>${fmt(t.d, 1)}</span></div>`).join('');
    const c2 = card(l, 1, `<canvas class="hm"></canvas><div class="s">정규화 입력(위)과 중심점 ${s.M}개 사이 거리² (가까운 8개)</div><div class="dl">${rows}</div><div class="thumbs"></div>`);
    const cv2 = c2.querySelector('canvas'); heat(cv2, deq(s.key)); cv2.style.width = '46%';
    const th = c2.querySelector('.thumbs'); for (let k = 0; k < 6; k++) th.appendChild(thumb('inf', s.imgs + k, 's'));
    const cap = document.createElement('div'); cap.className = 's'; cap.textContent = `#${s.j}에 가장 가까운 학습 사진 (${nf(s.n_train)}장 중)`;
    c2.appendChild(cap); P.appendChild(c2);

    const c3 = card(l, 2, `<canvas class="cv-grid"></canvas><div class="s">#${s.j} 목록 앞 <span class="mono" style="color:var(--on)">${nf(s.K)}</span>개만 켬. 밝기 = 더하는 크기</div><div class="s mono" style="font-size:11px">목록: ${s.first.slice(0, 8).join(', ')}, …</div>`);
    P.appendChild(c3); setTimeout(() => grid(c3.querySelector('canvas'), { bright: s.grid, on: unbits(s.on) }), 0);

    const z = s.zoom, gz = deq(z.gz), uz = deq(z.uz);
    const c4 = card(l, 3, `<div class="s"><b style="color:var(--fg);font-weight:600">켠 ${nf(s.K)}개를 모두 이렇게 계산.</b> 최대 기여 <span class="mono" style="color:var(--on)">#${z.i}</span> (목록 ${z.rank + 1}위):</div><div class="lbl">gate 행 × z (1,024곱)</div><canvas class="spk" style="width:100%;height:34px"></canvas><div class="lbl">up 행 × z (1,024곱)</div><canvas class="spk" style="width:100%;height:34px"></canvas><div class="eq">g = <b>${fmt(z.g, 3)}</b>, u = <b>${fmt(z.u, 3)}</b><br>a = silu(g)×u = <b>${fmt(z.a, 3)}</b><br>|a|×‖down‖ = <b>${fmt(Math.abs(z.a) * z.col, 3)}</b></div>`);
    P.appendChild(c4); const sp = c4.querySelectorAll('canvas.spk'); setTimeout(() => { spark(sp[0], gz); spark(sp[1], uz); }, 0);

    const c5 = card(l, 4, `<canvas class="hm"></canvas><div class="s">y = down 고른 열 × a · ‖y‖ = <span class="mono">${fmt(s.yn, 1)}</span></div><div class="eq">h ← h + y</div>`);
    heat(c5.querySelector('canvas'), deq(s.y)); P.appendChild(c5);
    host.appendChild(sec);
  });
  const fin = $('#in-final'), pred = e.probs.indexOf(Math.max(...e.probs));
  fin.innerHTML = `<header style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:12px;margin-bottom:12px"><h3 style="font-size:16px">출력층 → 부류 10개의 확률</h3><span class="lbl">예측 <span style="color:var(--on)">${D.classes[pred]}</span> · 정답 ${D.classes[e.label]}</span></header><div class="final"><div class="pcard" data-k="20"><div class="t"><span>⑥</span>예측</div><div class="row"><canvas id="fin-img" width="28" height="28" style="width:84px;height:84px;image-rendering:pixelated;border-radius:8px;background:#000"></canvas><div><div class="big on">${D.classes[pred]}</div><div class="s">확률 <span class="mono">${fmt(e.probs[pred] * 100, 1)}%</span> (베이스 풀 모델 <span class="mono">${fmt(e.base_probs[pred] * 100, 1)}%</span>)</div></div></div><div class="s">층별 켠 뉴런: ${e.steps.map(s => (s.K / 40.96).toFixed(1) + '%').join(' / ')} · 평균 ${(e.steps.reduce((a, s) => a + s.K, 0) / 4 / 40.96).toFixed(1)}%</div></div><div class="chart" id="ch-prob"></div></div>`;
  photo($('#fin-img'), e.img);
  probChart(e);
  markStep();
}

// =========================================================================
// TAB 3: 묶어서 계산 (CH2 + v4 + A3V4)
// =========================================================================
const P1_ROWS = ['T1 가', 'T2 나', 'T3 다', 'T4 라', 'T5 마', 'T6 바'];
const P1_G = [
  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2],
  [0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0],
  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0],
  [0, 1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 1],
];
const P1_BSMR_PANELS = [
  { tag: '패널 1', rows: [0, 4], cols: [3, 6, 0, 10, 1, 9] },
  { tag: '패널 2', rows: [1, 5], cols: [8, 11, 1, 5, 2, 4] },
  { tag: '패널 3', rows: [2, 3], cols: [7, 2, 4, 9, 0, 5, 10, -1, -1] },
];

function drawBsmrGrid(host, isContrib) {
  host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), lx = 58, rx = 10;
  const cs = Math.min(36, (W - lx - rx) / 12);
  const top0 = 14, hdrH = 18, panelGap = 14, panelH = hdrH + 2 * cs;
  const gridEnd = top0 + 3 * panelH + 2 * panelGap, H = gridEnd + 36;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'BSMR 고정 행 패널 예시' }, host);
  let tid = 0;
  P1_BSMR_PANELS.forEach((P, pi) => {
    const pTop = top0 + pi * (panelH + panelGap), yGrid = pTop + hdrH, nTiles = P.cols.length / 3;
    tx(svg, lx - 8, yGrid - 5, P.tag, { 'text-anchor': 'end', fill: C.pick, 'font-size': 9.5, 'font-weight': 600 });
    for (let c = 0; c < P.cols.length; c++) {
      const colIdx = P.cols[c];
      tx(svg, lx + c * cs + cs / 2, yGrid - 5, colIdx >= 0 ? 'N' + (colIdx + 1) : '—', { 'text-anchor': 'middle', fill: colIdx >= 0 ? C.fg2 : C.fg3, 'font-size': Math.min(10.5, cs * 0.34) });
    }
    for (let dr = 0; dr < 2; dr++) tx(svg, lx - 8, yGrid + dr * cs + cs / 2 + 4, P1_ROWS[P.rows[dr]], { 'text-anchor': 'end', class: 'lab', 'font-size': 11 });
    for (let dr = 0; dr < 2; dr++) {
      for (let c = 0; c < 12; c++) {
        const x = lx + c * cs + 2, y = yGrid + dr * cs + 2, s = cs - 4;
        const skipped = c >= P.cols.length, colIdx = skipped ? -1 : P.cols[c];
        const v = colIdx >= 0 ? P1_G[P.rows[dr]][colIdx] : 0;
        let fill = skipped ? C.bg : C.off, stroke = 'none', dash = '', op = skipped ? 0.45 : 1;
        if (!skipped) {
          if (v === 1) fill = C.on;
          else if (v === 2) { fill = C.onsoft; stroke = C.on; dash = '3 2'; }
          else fill = C.pick;
        }
        const a = { x, y, width: s, height: s, rx: 3, fill, opacity: op };
        if (stroke !== 'none') { a.stroke = stroke; a['stroke-width'] = 1.4; if (dash) a['stroke-dasharray'] = dash; }
        el('rect', a, svg);
        if (!skipped) {
          if (v === 2) tx(svg, lx + c * cs + cs / 2, yGrid + dr * cs + cs / 2 + 4, '1*', { 'text-anchor': 'middle', fill: C.on, 'font-size': Math.max(9.5, cs * 0.34), 'font-weight': 600 });
          else if (v === 0) tx(svg, lx + c * cs + cs / 2, yGrid + dr * cs + cs / 2 + 3.5, '×0', { 'text-anchor': 'middle', fill: C.bg, 'font-size': Math.max(8.5, cs * 0.30), 'font-weight': isContrib ? 600 : 400 });
        }
      }
    }
    for (let bc = 0; bc < 4; bc++) {
      const x = lx + bc * 3 * cs, y = yGrid, w = 3 * cs, h = 2 * cs;
      if (bc >= nTiles) {
        el('rect', { x: x + 1, y: y + 1, width: w - 2, height: h - 2, rx: 4, fill: 'transparent', stroke: C.line2, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
        tx(svg, x + w / 2, y + h / 2 + 3.5, '건너뜀', { 'text-anchor': 'middle', fill: C.fg3, 'font-size': Math.max(9.5, cs * 0.30) });
        continue;
      }
      let aCnt = 0;
      for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 3; dc++) { const ci = P.cols[bc * 3 + dc]; if (ci >= 0 && P1_G[P.rows[dr]][ci] > 0) aCnt++; }
      tid++;
      const zCnt = 6 - aCnt, isTail = aCnt <= 1;
      const box = el('rect', {
        x: x + 0.5, y: y + 0.5, width: w - 1, height: h - 1, rx: 4, fill: 'transparent',
        stroke: isTail ? C.pick : C.fg, 'stroke-width': isTail ? 1.8 : 1.3, 'stroke-opacity': isTail ? 0.95 : 0.72,
        ...(isTail ? { 'stroke-dasharray': '4 2' } : {}), style: 'cursor:pointer',
      }, svg);
      const idC = tid, aC = aCnt, zC = zCnt, pTag = P.tag, tailC = isTail;
      box.addEventListener('pointerenter', ev => {
        box.setAttribute('stroke', isContrib ? C.pick : C.on); box.setAttribute('stroke-width', 2.2); box.setAttribute('stroke-opacity', 1);
        showTip(ev, tailC
          ? `BSMR ${pTag} 틀 #${idC} (2×3): 쓰는 1*=${aC}칸 · 빈칸=${zC}칸 — 저밀도 CUDA 우회`
          : `BSMR ${pTag} 틀 #${idC} (2×3): 쓰는 1=${aC}칸 · ×0=${zC}칸 (채움 ${Math.round(100 * aC / 6)}%)`);
      });
      box.addEventListener('pointermove', ev => showTip(ev, $('#tip').textContent));
      box.addEventListener('pointerleave', () => {
        box.setAttribute('stroke', tailC ? C.pick : C.fg); box.setAttribute('stroke-width', tailC ? 1.8 : 1.3); box.setAttribute('stroke-opacity', tailC ? 0.95 : 0.72);
        hideTip();
      });
    }
    if (pi < 2) {
      const yDiv = yGrid + 2 * cs + panelGap / 2;
      el('line', { x1: 4, x2: lx + 12 * cs + 4, y1: yDiv, y2: yDiv, stroke: C.pick, 'stroke-width': 1.6, 'stroke-dasharray': '5 3', opacity: 0.92 }, svg);
    }
  });
  const cx = lx + 6 * cs;
  tx(svg, cx, gridEnd + 14, W < 470 ? '주황 점선 = 고정 행 패널 경계 (행 짝 고정)' : '주황 점선 = 고정 행 패널 경계 (패널마다 열을 정렬해도 행 짝이 고정됨)', { 'text-anchor': 'middle', fill: C.pick, 'font-size': W < 470 ? 8.8 : 9.8, 'font-weight': 600 });
  tx(svg, cx, gridEnd + 29, W < 470 ? '점선 틀 #7 (1/6): 저밀도 꼬리 → CUDA 우회' : '점선 틀 #7 (1/6): 저밀도 꼬리 → BSMR은 CUDA 코어로 우회해야 함', { 'text-anchor': 'middle', class: 'lab', 'font-size': W < 470 ? 8.6 : 9.4 });
}

function drawProb() {
  const btns = $('#bt-p1-btns'); btns.innerHTML = '';
  ['① 토큰 × 뉴런 표', '② 순서대로 틀(2×3) 끼우기', '③ 기존 SDDMM (BSMR · 행 패널)'].forEach((label, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', S.p1 === idx);
    b.onclick = () => { S.p1 = idx; drawProb(); };
    btns.appendChild(b);
  });
  const caps = [
    '토큰 6개 × 뉴런 12개 · 1이 산발적으로 흩어짐',
    '2×3 틀 12개 · 모든 틀에 불필요한 0이 4칸씩',
    '고정 2행 패널 + 패널별 열 정렬 · 빈 5틀 건너뛰어도 7틀 중 42.9%가 ×0',
  ];
  $('#bt-p1-cap').textContent = caps[S.p1];
  const host = $('#bt-p1-svg');
  if (S.p1 === 2) {
    drawBsmrGrid(host, false);
    $('#bt-p1-kpi').innerHTML =
      `<div class="panel"><div class="k">실제 필요한 계산 (1 + 선택적 1*)</div><div class="v on">24칸 <span style="font-size:12px;color:var(--fg-3)">/ 72칸 (33.3%)</span></div></div>
       <div class="panel"><div class="k">최적 BSMR (행 패널 고정 · 7틀)</div><div class="v pick">×0 18칸 (42.9%) <span style="font-size:11.5px;color:var(--fg-3)">· 틀 #7 CUDA 우회</span></div></div>`;
    return;
  }
  host.innerHTML = '';
  const blkAct = Array.from({ length: 3 }, (_, br) => Array.from({ length: 4 }, (_, bc) => {
    let cnt = 0; for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 3; dc++) if (P1_G[br * 2 + dr][bc * 3 + dc] > 0) cnt++;
    return cnt;
  }));
  const W = Math.max(320, host.clientWidth || 540), lx = 58, rx = 10, top = 28;
  const cs = Math.min(36, (W - lx - rx) / 12), H = top + 6 * cs + 14;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '토큰 6개 x 뉴런 12개 표' }, host);
  tx(svg, lx - 8, top - 9, '뉴런', { 'text-anchor': 'end' });
  for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 9, 'N' + (c + 1), { 'text-anchor': 'middle' });
  for (let r = 0; r < 6; r++) tx(svg, lx - 8, top + r * cs + cs / 2 + 4, P1_ROWS[r], { 'text-anchor': 'end', class: 'lab' });
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 12; c++) {
      const v = P1_G[r][c], x = lx + c * cs + 2, y = top + r * cs + 2, s = cs - 4, inTileNow = S.p1 === 1;
      let fill = C.off, stroke = 'none', dash = '';
      if (v === 1) fill = C.on;
      else if (v === 2) { fill = C.onsoft; stroke = C.on; dash = '3 2'; }
      else if (inTileNow) fill = C.pick;
      const a = { x, y, width: s, height: s, rx: 3, fill };
      if (stroke !== 'none') { a.stroke = stroke; a['stroke-width'] = 1.4; if (dash) a['stroke-dasharray'] = dash; }
      el('rect', a, svg);
      if (v === 2) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 4, '1*', { 'text-anchor': 'middle', fill: C.on, 'font-size': Math.max(9.5, cs * 0.34), 'font-weight': 600 });
      else if (v === 0 && inTileNow) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3.5, '×0', { 'text-anchor': 'middle', fill: C.bg, 'font-size': Math.max(8.5, cs * 0.30) });
    }
  }
  if (S.p1 === 1) {
    let tid = 0;
    for (let br = 0; br < 3; br++) {
      for (let bc = 0; bc < 4; bc++) {
        const aCnt = blkAct[br][bc], zCnt = 6 - aCnt;
        const x = lx + bc * 3 * cs, y = top + br * 2 * cs, w = 3 * cs, h = 2 * cs; tid++;
        const box = el('rect', { x: x + 0.5, y: y + 0.5, width: w - 1, height: h - 1, rx: 4, fill: 'transparent', stroke: C.fg, 'stroke-width': 1.3, 'stroke-opacity': 0.72, style: 'cursor:pointer' }, svg);
        const idC = tid, aC = aCnt, zC = zCnt;
        box.addEventListener('pointerenter', ev => { box.setAttribute('stroke', C.on); box.setAttribute('stroke-width', 2.2); showTip(ev, `틀 #${idC} (2×3): 쓰는 1=${aC}칸 · ×0=${zC}칸 (채움 ${Math.round(100 * aC / 6)}%)`); });
        box.addEventListener('pointermove', ev => showTip(ev, $('#tip').textContent));
        box.addEventListener('pointerleave', () => { box.setAttribute('stroke', C.fg); box.setAttribute('stroke-width', 1.3); hideTip(); });
      }
    }
  }
  $('#bt-p1-kpi').innerHTML = S.p1 === 1
    ? `<div class="panel"><div class="k">실제 필요한 계산 (1 + 1*)</div><div class="v on">24칸 <span style="font-size:12px;color:var(--fg-3)">/ 72칸 (33.3%)</span></div></div>
       <div class="panel"><div class="k">순서대로 틀 끼운 결과 (2×3 × 12틀)</div><div class="v pick">불필요한 0이 48칸 (66.7%)</div></div>`
    : `<div class="panel"><div class="k">표 크기 (세로 토큰 × 가로 뉴런)</div><div class="v">6행 × 12열 = 72칸</div></div>
       <div class="panel"><div class="k">최종 CH2 모델 1* 설정</div><div class="v on">1* = 0% (완전 무손실 · 90.81%)</div></div>`;
}

// ---- b2: 5-tier 8x12 interactive walkthrough ----
const J_G = [
  [1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 0],
  [1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1, 0],
  [1, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 1],
  [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0],
  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
  [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
  [1, 1, 1, 0, 2, 1, 0, 0, 2, 0, 0, 1],
  [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2],
];
const J_TILES = [
  { name: '① 8×4 (64×32)', rows: [0, 1, 2, 3, 4, 5, 6, 7], cols: [0, 2, 5, 8],  stat: '31/32 (0 1칸)', accent: 'on' },
  { name: '② 4×4 (32×32)', rows: [0, 1, 3, 5],             cols: [1, 4, 7, 10], stat: '15/16 (0 1칸)', accent: 'on' },
  { name: '③ 4×2 (32×16)', rows: [2, 4, 6, 7],             cols: [1, 4],        stat: '8/8 만점',      accent: 'pick' },
  { name: '④ 2×2 (16×16)', rows: [2, 6],                   cols: [6, 11],       stat: '3/4 (0 1칸)',   accent: 'pick' },
  { name: '⑤ 1×2 (8×16)',  rows: [4],                      cols: [3, 9],        stat: '2/2 만점',      accent: 'neg' },
];
const J_STEPS = [
  { title: '원래 표 (8행 × 12열)', sub: '1순위 8×4(64×32) → 4×4 → 4×2 → 2×2 → 1×2(8×16) 실측 비용 사다리 순으로 빽빽한 덩어리부터 떠낸다.' },
  { title: '① 1순위 8×4 (64×32 비유) 추출', sub: '가장 빽빽한 {R1~R8}×{C1,C3,C6,C9}을 1순위 8×4(31/32, 97%)로 떠내 cuBLAS 90% 효율의 고속 구간을 선점한다.' },
  { title: '② 2순위 4×4 (32×32 비유) 추출', sub: '1번 틀을 지운 뒤 남은 {R1,R2,R4,R6}×{C2,C5,C8,C11}을 2순위 4×4(15/16, 94%)로 떠낸다.' },
  { title: '③ 3순위 4×2 (32×16 비유) 추출', sub: '남은 {R3,R5,R7,R8}×{C2,C5}을 3순위 4×2(8/8 만점)로 깔끔히 떠낸다.' },
  { title: '④ 4순위 2×2 (16×16 비유) 추출', sub: '중간 밀도 자투리 {R3,R7}×{C7,C12}(3/4)을 4순위 2×2로 주워 담는다.' },
  { title: '⑤ 5순위 1×2 (8×16 비유)로 마감', sub: '남은 필수 1 2칸({R5}×{C4,C10})을 5순위 1×2로 0 낭비 없이 마감한다.' },
  { title: '⑥ 종료 — 5틀 겹침 0칸, 77.4%를 1·2순위 고속 틀로 처리', sub: '필수 1이 0개 남아 종료! 선택적 1* 자유도를 켜면 외딴 1* 3개도 자동 OFF할 수 있다(최종 CH2는 1*=0%로도 86.6%를 1·2순위 틀에 채움).' },
];
const inTile = (t, r, c) => t.rows.includes(r) && t.cols.includes(c);

function drawJoint() {
  const ST = J_STEPS[S.step], host = $('#bt-mat'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 560), stack = W < 520, lx = 28, top = 32;
  const cs = stack ? Math.min(28, Math.floor((W - lx - 10) / 12)) : Math.min(24, Math.max(19, Math.floor((W - 252) / 12)));
  const matW = 12 * cs, matH = 8 * cs;
  const rx0 = stack ? lx : lx + matW + Math.min(24, Math.max(14, Math.floor((W - lx - matW - 216) / 2)));
  const rw = Math.max(206, W - rx0 - 4);
  const mcs = stack ? Math.min(16, Math.max(12, Math.floor((rw - 30) / 12))) : Math.min(13, Math.max(11, Math.floor((rw - 24) / 9)));
  const ry0 = stack ? top + matH + 34 : top, col2 = rx0 + Math.floor(rw * 0.51);
  const y_r2_c1 = ry0 + 12 + 8 * mcs + 32, y_r2_c2 = ry0 + 12 + 4 * mcs + 32, y_r3_c2 = y_r2_c2 + 4 * mcs + 32;
  const panelH = Math.max(matH, (y_r3_c2 - ry0) + 1 * mcs + 20);
  const H = stack ? ry0 + panelH + 10 : Math.max(top + matH + 16, top + panelH + 10);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '5단계 Roster-Tile 8x12 예시: ' + ST.title }, host);

  tx(svg, lx, 13, '남은 표 (8행 × 12열)', { class: 'hi' });
  for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 6, 'C' + (c + 1), { 'text-anchor': 'middle', 'font-size': 9 });
  for (let r = 0; r < 8; r++) tx(svg, lx - 5, top + r * cs + cs / 2 + 3.5, 'R' + (r + 1), { 'text-anchor': 'end', class: 'lab', 'font-size': 10 });

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 12; c++) {
      const v = J_G[r][c], x = lx + c * cs + 1.5, y = top + r * cs + 1.5, s = cs - 3;
      let done = false, hi = false;
      for (let ti = 0; ti < J_TILES.length; ti++) {
        if (inTile(J_TILES[ti], r, c)) { if (S.step > ti + 1) done = true; else if (S.step === ti + 1) hi = true; }
      }
      const offOpt = S.step === 6 && v === 2 && !done;
      let fill = C.off, stroke = 'none', dash = '', op = 1, txt = '', tfill = C.bg;
      if (done) { fill = C.bg; stroke = C.line; op = 0.7; txt = '·'; tfill = C.fg3; }
      else if (offOpt) { fill = C.bg; stroke = C.line2; dash = '2 2'; txt = 'OFF'; tfill = C.fg3; }
      else if (hi && v === 0) { fill = C.pick; stroke = C.fg; txt = '×0'; tfill = C.bg; }
      else if (hi && v === 2) { fill = C.neg; stroke = C.fg; txt = '1*'; tfill = C.bg; }
      else if (v === 1) { fill = C.on; if (hi) stroke = C.fg; }
      else if (v === 2) { fill = C.onsoft; stroke = C.on; dash = '2.5 2'; txt = '1*'; tfill = C.on; }
      const a = { x, y, width: s, height: s, rx: 3, fill, opacity: op };
      if (stroke !== 'none') { a.stroke = stroke; a['stroke-width'] = hi ? 1.8 : 1.1; if (dash) a['stroke-dasharray'] = dash; }
      el('rect', a, svg);
      if (txt) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3, txt, { 'text-anchor': 'middle', fill: tfill, 'font-size': txt === 'OFF' ? 6.5 : Math.max(7.5, cs * 0.34), 'font-weight': 600 });
    }
  }
  tx(svg, rx0, ry0 - 15, '5단계 순위대로 확정된 틀', { class: 'hi' });
  const drawMini = (tile, ox, oy) => {
    const col = tile.accent === 'on' ? C.on : tile.accent === 'pick' ? C.pick : C.neg;
    const tw = tile.cols.length * mcs, th = tile.rows.length * mcs;
    tx(svg, ox, oy - 5, tile.name, { fill: col, 'font-size': 9.5, 'font-weight': 600 });
    tile.rows.forEach((r, i) => tile.cols.forEach((c, j) => {
      const v = J_G[r][c];
      el('rect', { x: ox + j * mcs + 1, y: oy + i * mcs + 1, width: mcs - 2, height: mcs - 2, rx: 2, fill: v === 1 ? C.on : v === 2 ? C.neg : C.pick }, svg);
    }));
    el('rect', { x: ox, y: oy, width: tw, height: th, rx: 3, fill: 'none', stroke: col, 'stroke-width': 1.3 }, svg);
    tx(svg, ox, oy + th + 12, tile.stat, { class: 'lab', 'font-size': 9 });
  };
  if (S.step === 0) {
    const ph = stack ? 116 : matH;
    el('rect', { x: rx0, y: ry0, width: rw, height: ph, rx: 8, fill: C.bg, stroke: C.line, 'stroke-dasharray': '4 3' }, svg);
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 - 16, '1순위 8×4 → 2순위 4×4', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 + 1,  '3순위 4×2 → 4순위 2×2 → 1×2', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 + 21, '다음을 눌러 5단계 추출 시작', { 'text-anchor': 'middle', 'font-size': 9.5 });
  } else {
    if (S.step >= 1) drawMini(J_TILES[0], rx0, ry0 + 12);
    if (S.step >= 2) drawMini(J_TILES[1], col2, ry0 + 12);
    if (S.step >= 3) drawMini(J_TILES[2], col2, y_r2_c2);
    if (S.step >= 4) drawMini(J_TILES[3], rx0, y_r2_c1);
    if (S.step >= 5) drawMini(J_TILES[4], col2, y_r3_c2);
  }
  $('#bt-where').innerHTML = `<b>${ST.title}</b> <span class="mono" style="color:var(--fg-3)">(${S.step + 1} / ${J_STEPS.length})</span>`;
  const side = $('#bt-side');
  if (S.step < 6) side.innerHTML = `<p>${ST.sub}</p>`;
  else side.innerHTML = `<p>${ST.sub}</p>
    <div class="bt-tbl-wrap"><table class="bt-tbl">
      <tr><th>비교</th><th>전체 칸</th><th>1·2순위 고속 틀 비중</th><th>불필요한 0</th><th>필수 1 (55개)</th></tr>
      <tr><td>순서대로 1×2 자르기</td><td class="num">84칸 <span class="mut">(전체 96)</span></td><td class="num warn">0% (전부 저속)</td><td class="num warn">22 ~ 29칸</td><td class="num">100%</td></tr>
      <tr class="sum"><td>5단계 Roster-Tile</td><td class="num ok">62칸 (-35.4%)</td><td class="num ok">77.4% (8×4 · 4×4)</td><td class="num ok">3칸 (-91%)</td><td class="num ok">100% · 겹침 0</td></tr>
    </table></div>`;
}

// ---- b3: CH2 + A3V4 real 4-layer server stats ----
function drawLayerStats() {
  const chips = $('#bt-b-chips'); chips.innerHTML = '';
  [['512', '배치 512 (소형 청크)'], ['1024', '배치 1,024 (Chunked Prefill)'], ['2048', '배치 2,048 (vLLM 기본 청크)']].forEach(([k, label]) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', S.batch === k);
    b.onclick = () => { S.batch = k; drawLayerStats(); };
    chips.appendChild(b);
  });
  const T = B.layer_tables[S.batch], tot = T.total;
  $('#bt-b-note').textContent = T.note;
  const rowHtml = (name, r, isSum) => {
    const c = r.counts;
    return `<tr${isSum ? ' class="sum"' : ''}>
      <td>${name} <span class="mut">(${r.density_pct.toFixed(1)}%)</span></td>
      <td class="num"><b style="color:var(--fg)">${r.remain_pct.toFixed(1)}%</b> <span class="mut">(${nf(r.comp_cells)})</span></td>
      <td class="num">${r.p2_union_pct.toFixed(1)}%${r.p2_tm ? ` <span class="mut">(TM=${r.p2_tm})</span>` : ''}</td>
      <td class="num">${nf(c['64x32'])} / ${nf(c['32x32'])} / ${nf(c['32x16'])} / ${nf(c['16x16'])} / ${nf(c['8x16'])}</td>
      <td class="num ok">${r.share_top2.toFixed(1)}% <span class="mut">(채움 ${r.fill_pct.toFixed(1)}%)</span></td>
      <td class="num">${r.dram_pct.toFixed(1)}%</td>
    </tr>`;
  };
  $('#bt-ltbl').innerHTML =
    '<tr><th>층 (활성 1)</th><th>P1 연산 칸</th><th>P2 합집합</th><th>틀 수 (64×32 / 32×32 / 32×16 / 16×16 / 8×16)</th><th>1·2순위 비중 · 채움</th><th>DRAM 읽기</th></tr>' +
    T.layers.map(r => rowHtml('층 ' + r.layer, r, false)).join('') +
    rowHtml('4개 층 합계', tot, true);

  $('#bt-b-mix').innerHTML =
    `<div class="panel"><div class="k">5단계 연산 칸 분포 (1→5순위)</div><div class="v on" style="font-size:13px">64×32 ${tot.share_64x32.toFixed(1)}% · 32×32 ${tot.share_32x32.toFixed(1)}% · 32×16 ${tot.share_32x16.toFixed(1)}% · 16×16 ${tot.share_16x16.toFixed(1)}% · 8×16 ${tot.share_8x16.toFixed(1)}%</div></div>
     <div class="panel"><div class="k">FFN 총 연산 · 정확도 · 디스패치</div><div class="v" style="font-size:14.5px">FFN ${tot.ffn_flops_pct.toFixed(1)}% · ${B.accuracy.test_b2048.toFixed(2)}% (1*=0%) <span style="font-size:11.5px;color:var(--fg-3)">· A3V4 ${tot.dispatch_4l_ms.toFixed(2)}ms</span></div></div>`;
}

// ---- b4: L2 column-strip caching ----
const C_ORIG = [1, 2, 3, 4, 5, 6, 7, 8];
const C_SORT = [1, 5, 3, 7, 2, 6, 4, 8];
const isFamA = n => n % 2 === 1;

function drawCache() {
  const btns = $('#bt-c-btns'); btns.innerHTML = '';
  ['① 그대로 돌리면 (중복 읽기)', '② 옆자리로 이사 (비용 0초)', '③ 구역별 실행 (1회 읽기)'].forEach((label, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', S.cache === idx);
    b.onclick = () => { S.cache = idx; drawCache(); };
    btns.appendChild(b);
  });
  const caps = [
    '작업대(L2) 4칸이 넘쳐 버렸던 N1을 DRAM에서 또 퍼 옴',
    '같이 쓰이는 짝(청록 · 보라)끼리 열 순서를 미리 한 번만 정렬',
    '구역 1개(4칸)가 작업대에 통째로 상주 → 쫓겨남 0개, 1회 읽기',
  ];
  $('#bt-c-cap').textContent = caps[S.cache];
  $('#bt-c-strips-row').style.display = S.cache === 2 ? 'flex' : 'none';
  const sch = $('#bt-c-strips'); sch.innerHTML = '';
  ['구역 1 (N1·N5·N3·N7)', '구역 2 (N2·N6·N4·N8)'].forEach((lbl, s) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.strip === s);
    b.onclick = () => { S.strip = s; drawCache(); };
    sch.appendChild(b);
  });
  const host = $('#bt-c-svg'); host.innerHTML = '';
  const W = Math.max(360, host.clientWidth || 560), H = 224, pad = 16, barW = W - 2 * pad;
  const gap = 6, midGap = 12, bw = (barW - 7 * gap - midGap) / 8;
  const bx = i => pad + i * (bw + gap) + (i >= 4 ? midGap : 0);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '뉴런 8개와 L2 작업대 4칸 캐싱 예시' }, host);
  const drawNeuron = (x, y, h, n, dim) => {
    const a = isFamA(n), col = a ? C.on : C.neg;
    el('rect', { x, y, width: bw, height: h, rx: 5, fill: dim ? C.bg : (a ? C.onsoft : 'rgba(157,140,255,0.14)'), stroke: dim ? C.line : col, 'stroke-width': dim ? 1 : 1.5, opacity: dim ? 0.55 : 1 }, svg);
    tx(svg, x + bw / 2, y + h / 2 + 4, 'N' + n, { 'text-anchor': 'middle', fill: dim ? C.fg3 : col, 'font-size': 11.5, 'font-weight': 600 });
  };
  if (S.cache === 0) {
    tx(svg, pad, 16, W < 470 ? '원래 순서 (청록 N1·3·5·7 / 보라 N2·4·6·8 뒤섞임)' : '원래 뉴런 순서 (같이 쓰이는 짝: 청록 N1·N3·N5·N7 / 보라 N2·N4·N6·N8이 뒤섞임)', { class: 'hi', 'font-size': W < 470 ? 9.8 : 11 });
    C_ORIG.forEach((n, i) => drawNeuron(bx(i), 24, 28, n, false));
    const l2W = Math.min(barW, Math.max(250, barW * 0.62)), l2X = pad + (barW - l2W) / 2;
    el('rect', { x: l2X, y: 82, width: l2W, height: 54, rx: 8, fill: C.picksoft, stroke: C.pick, 'stroke-width': 1.6 }, svg);
    tx(svg, l2X + l2W / 2, 102, 'L2 작업대 (4칸 한계) · N1, N5가 밀려남!', { 'text-anchor': 'middle', fill: C.pick, 'font-size': 11.5, 'font-weight': 600 });
    tx(svg, l2X + l2W / 2, 122, '현재 상주: [N3, N7, N4, N8] ← N1을 쓰려면 DRAM 재읽기', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.2 });
    [0, 1, 2, 4, 5].forEach(i => el('line', { x1: bx(i) + bw / 2, y1: 52, x2: l2X + l2W / 2, y2: 82, stroke: C.pick, 'stroke-width': 1.3, 'stroke-dasharray': '3 3', opacity: 0.85 }, svg));
    tx(svg, pad, 158, '전역 타일 실행 순서 (4칸 작업대 초과로 재읽기 발생):', { class: 'lab', 'font-size': 11 });
    const steps = [['1) {N1,N5}', '올림 (2/4)', C.fg3], ['2) {N2,N6}', '꽉 참 (4/4)', C.fg3], ['3) {N3,N7}', 'N1·N5 버림!', C.pick], ['4) {N1,N3}', 'N1 또 퍼 옴!', C.pick]];
    const cw = (barW - 18) / 4;
    steps.forEach(([t1, t2, col], k) => {
      const tx0 = pad + k * (cw + 6), warn = col === C.pick;
      el('rect', { x: tx0, y: 168, width: cw, height: 42, rx: 6, fill: warn ? C.picksoft : C.bg3, stroke: warn ? C.pick : C.line2, 'stroke-width': warn ? 1.4 : 1 }, svg);
      tx(svg, tx0 + cw / 2, 185, t1, { 'text-anchor': 'middle', fill: warn ? C.fg : C.fg2, 'font-size': 10.5, 'font-weight': 600 });
      tx(svg, tx0 + cw / 2, 201, t2, { 'text-anchor': 'middle', fill: col, 'font-size': 10 });
    });
  } else if (S.cache === 1) {
    tx(svg, pad, 14, '원래 순서 → 바꾼 순서 (오프라인 1회 정렬 · 출력 100% 동일 · 추론 비용 0초)', { class: 'hi', 'font-size': 10.8 });
    C_ORIG.forEach((n, i) => drawNeuron(bx(i), 22, 26, n, false));
    C_ORIG.forEach((n, i) => {
      const j = C_SORT.indexOf(n), x1 = bx(i) + bw / 2, x2 = bx(j) + bw / 2;
      el('path', { d: `M ${x1} 48 C ${x1} 70, ${x2} 70, ${x2} 92`, fill: 'none', stroke: isFamA(n) ? C.on : C.neg, 'stroke-width': 1.5, opacity: 0.8 }, svg);
    });
    C_SORT.forEach((n, i) => drawNeuron(bx(i), 92, 26, n, false));
    const g1x = bx(0), g1w = bx(3) + bw - bx(0), g2x = bx(4), g2w = bx(7) + bw - bx(4);
    el('line', { x1: g1x + g1w / 2, y1: 118, x2: g1x + g1w / 2, y2: 138, stroke: C.on, 'stroke-width': 1.8 }, svg);
    el('line', { x1: g2x + g2w / 2, y1: 118, x2: g2x + g2w / 2, y2: 138, stroke: C.neg, 'stroke-width': 1.8 }, svg);
    el('rect', { x: g1x, y: 138, width: g1w, height: 60, rx: 8, fill: C.onsoft, stroke: C.on, 'stroke-width': 1.4 }, svg);
    tx(svg, g1x + g1w / 2, 162, '구역 1 (작업대 4칸 딱 맞음)', { 'text-anchor': 'middle', fill: C.on, 'font-size': 11, 'font-weight': 600 });
    tx(svg, g1x + g1w / 2, 182, 'N1 · N5 · N3 · N7 (같이 쓰이는 짝)', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.2 });
    el('rect', { x: g2x, y: 138, width: g2w, height: 60, rx: 8, fill: 'rgba(157,140,255,0.12)', stroke: C.neg, 'stroke-width': 1.4 }, svg);
    tx(svg, g2x + g2w / 2, 162, '구역 2 (작업대 4칸 딱 맞음)', { 'text-anchor': 'middle', fill: C.neg, 'font-size': 11, 'font-weight': 600 });
    tx(svg, g2x + g2w / 2, 182, 'N2 · N6 · N4 · N8 (같이 쓰이는 짝)', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.2 });
  } else {
    const s = S.strip, acc = s === 0 ? C.on : C.neg, accSoft = s === 0 ? C.onsoft : 'rgba(157,140,255,0.14)';
    const zStart = s * 4, zNums = C_SORT.slice(zStart, zStart + 4);
    tx(svg, pad, 16, `정렬된 DRAM 가중치 · 현재 구역 ${s + 1}만 DRAM에서 딱 1번 올림`, { class: 'hi', 'font-size': 11 });
    C_SORT.forEach((n, i) => drawNeuron(bx(i), 24, 28, n, Math.floor(i / 4) !== s));
    const gx = bx(zStart) - 3, gw = bx(zStart + 3) + bw - bx(zStart) + 6;
    el('rect', { x: gx, y: 21, width: gw, height: 34, rx: 7, fill: 'none', stroke: acc, 'stroke-width': 1.8 }, svg);
    const l2W = Math.max(gw, 240), l2X = s === 0 ? pad : pad + barW - l2W, cx = gx + gw / 2;
    el('line', { x1: cx, y1: 55, x2: cx, y2: 83, stroke: acc, 'stroke-width': 2 }, svg);
    el('rect', { x: l2X, y: 83, width: l2W, height: 54, rx: 8, fill: accSoft, stroke: acc, 'stroke-width': 1.6 }, svg);
    tx(svg, l2X + l2W / 2, 103, `L2 작업대 (4칸) · 구역 ${s + 1} 통째로 상주`, { 'text-anchor': 'middle', fill: acc, 'font-size': 11.5, 'font-weight': 600 });
    tx(svg, l2X + l2W / 2, 123, `[${zNums.map(n => 'N' + n).join(', ')}] — 끝날 때까지 쫓겨남 0개`, { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
    tx(svg, pad, 159, `구역 ${s + 1} 안의 Roster-Tile 전부 계산 후 ${s === 0 ? '구역 2로 이동' : '층 계산 완료'}:`, { class: 'lab', 'font-size': 11 });
    const tiles = s === 0
      ? [['틀 #1 {N1,N5}', 'L2 100% 히트'], ['틀 #2 {N3,N7}', 'L2 100% 히트'], ['틀 #3 {N1,N3}', 'L2 100% 히트']]
      : [['틀 #1 {N2,N6}', 'L2 100% 히트'], ['틀 #2 {N4,N8}', 'L2 100% 히트'], ['틀 #3 {N2,N4}', 'L2 100% 히트']];
    const cw = (barW - 12) / 3;
    tiles.forEach(([t1, t2], k) => {
      const tx0 = pad + k * (cw + 6);
      el('rect', { x: tx0, y: 169, width: cw, height: 42, rx: 6, fill: C.bg3, stroke: acc, 'stroke-width': 1.2 }, svg);
      tx(svg, tx0 + cw / 2, 186, t1, { 'text-anchor': 'middle', fill: C.fg, 'font-size': 10.5, 'font-weight': 600 });
      tx(svg, tx0 + cw / 2, 202, t2, { 'text-anchor': 'middle', fill: acc, 'font-size': 10 });
    });
  }
  $('#bt-c-kpi').innerHTML =
    `<div class="panel"><div class="k">CH2 + W=512 열-스트립 DRAM 읽기</div><div class="v on">64.2 ~ 64.5% <span style="font-size:12px;color:var(--fg-3)">(중복 읽기 0% · B2 빈 블록 효과)</span></div></div>
     <div class="panel"><div class="k">8구역 병렬 A3V4 CPU 디스패치</div><div class="v">${B.strip_meta.disp_512.toFixed(2)} ~ ${B.strip_meta.disp_2048.toFixed(2)} ms <span style="font-size:12px;color:var(--fg-3)">(초기화 ${B.init_ms.toFixed(1)} ms 1회)</span></div></div>`;
}

// ---- b4b: NEW Interactive Example #2 (P1 vs P2 union & autoTM) + Example #3 (A3V4 Attention-Overlap vs Serial E2E) ----
const P2_MAT = [
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], // T1 (C_A)
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], // T2 (C_A)
  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], // T3 (C_A')
  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], // T4 (C_A')
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], // T5 (C_B)
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], // T6 (C_B)
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], // T7 (C_B')
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], // T8 (C_B')
];

function drawP2AndTimeline() {
  const mBtns = $('#bt-p2t-modes'); mBtns.innerHTML = '';
  ['① P1 vs P2 비대칭 & autoTM', '② A3V4 어텐션 중첩 vs 직렬 타임라인'].forEach((lbl, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.p2mode === idx);
    b.onclick = () => { S.p2mode = idx; drawP2AndTimeline(); };
    mBtns.appendChild(b);
  });

  const subRow = $('#bt-p2t-sub'), subChips = $('#bt-p2t-subchips'), subLbl = $('#bt-p2t-sublbl');
  subRow.style.display = 'flex'; subChips.innerHTML = '';
  const host = $('#bt-p2t-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 560);

  if (S.p2mode === 0) {
    subLbl.textContent = 'P2 토큰 블록 크기 (TM):';
    [['큰 블록 TM=8 (고정)', 0], ['작은 블록 TM=4 (autoTM)', 1]].forEach(([lbl, val]) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
      b.textContent = lbl; b.setAttribute('aria-pressed', S.p2tm === val);
      b.onclick = () => { S.p2tm = val; drawP2AndTimeline(); };
      subChips.appendChild(b);
    });
    const tm = S.p2tm === 0 ? 8 : 4;
    const lx = 58, rx = 12, top = 34;
    const cs = Math.min(30, Math.floor((W - lx - rx) / 12));
    const H = top + 8 * cs + 52;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'P2 W_down 토큰 블록 합집합과 autoTM 비교' }, host);

    const p2Hdr = W < 470
      ? (tm === 8 ? '고정 TM=8: A·B가족 합침 → 합집합 10/12열 (83%)' : 'autoTM (TM=4): 가족 분리 → 합집합 5/12열 (42%)!')
      : (tm === 8 ? '고정 TM=8: A가족(T1~4)과 B가족(T5~8)이 한 블록으로 합쳐짐 → 합집합 10/12열 (83%)' : 'autoTM (TM=4): 같은 가족끼리만 묶여 블록별 합집합이 5/12열 (42%)로 절반 축소!');
    tx(svg, W < 470 ? 6 : lx, 14, p2Hdr, {
      fill: tm === 8 ? C.pick : C.on, 'font-size': W < 470 ? 9.2 : 10.5, 'font-weight': 600,
    });
    for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 6, 'N' + (c + 1), { 'text-anchor': 'middle', 'font-size': 9.2 });

    const nBlk = 8 / tm;
    let unionCells = 0, actCells = 0;
    for (let blk = 0; blk < nBlk; blk++) {
      const r0 = blk * tm, r1 = r0 + tm;
      const colUnion = Array(12).fill(false);
      for (let r = r0; r < r1; r++) for (let c = 0; c < 12; c++) if (P2_MAT[r][c]) { colUnion[c] = true; actCells++; }
      const uCnt = colUnion.filter(Boolean).length;
      unionCells += uCnt * tm;
      for (let r = r0; r < r1; r++) {
        tx(svg, lx - 8, top + r * cs + cs / 2 + 4, `T${r + 1}(${r < 4 ? 'A' : 'B'})`, { 'text-anchor': 'end', fill: r < 4 ? C.on : C.neg, 'font-size': 9.8, 'font-weight': 600 });
        for (let c = 0; c < 12; c++) {
          const on = P2_MAT[r][c] === 1, inU = colUnion[c];
          const x = lx + c * cs + 1.5, y = top + r * cs + 1.5, s = cs - 3;
          let fill = C.off, txt = '';
          if (on) fill = r < 4 ? C.on : C.neg;
          else if (inU) { fill = C.pick; txt = '×0'; }
          el('rect', { x, y, width: s, height: s, rx: 3, fill }, svg);
          if (txt && cs >= 18) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3.5, txt, { 'text-anchor': 'middle', fill: C.bg, 'font-size': 8, 'font-weight': 700 });
        }
      }
      el('rect', { x: lx - 2, y: top + r0 * cs - 1, width: 12 * cs + 4, height: tm * cs + 2, rx: 4, fill: 'none', stroke: tm === 8 ? C.pick : C.fg, 'stroke-width': 1.5, 'stroke-dasharray': tm === 8 ? '5 3' : '' }, svg);
    }
    const uPct = (100 * unionCells / 96).toFixed(1), zPad = unionCells - actCells;
    tx(svg, W / 2, top + 8 * cs + 18, `P2 합집합: ${unionCells}/96칸 (${uPct}%) · 합집합 내 ×0 = ${zPad}칸`, { 'text-anchor': 'middle', fill: tm === 8 ? C.pick : C.on, 'font-size': W < 470 ? 9.2 : 10.5, 'font-weight': 600 });
    tx(svg, W / 2, top + 8 * cs + 36, W < 470 ? '실제 CH2(B=2,048): P1 15.3% · P2 16.1% (-10%p)' : '실제 CH2(B=2,048): P1 15.3% · P2 합집합 16.1% (이전 M=256의 26.1% 대비 -10.0%p 단축!)', { 'text-anchor': 'middle', class: 'lab', 'font-size': W < 470 ? 8.8 : 9.8 });

    $('#bt-p2t-kpi').innerHTML =
      `<div class="panel"><div class="k">예시 P2 합집합 (TM=8 → TM=4)</div><div class="v ${tm === 8 ? 'pick' : 'on'}" style="font-size:14.5px">${unionCells}칸 (${uPct}%) · ×0 ${zPad}칸</div></div>
       <div class="panel"><div class="k">실제 CH2 P1 · P2 균형 (B=2,048)</div><div class="v on" style="font-size:14.5px">P1 15.3% · P2 16.1% <span style="font-size:11.5px;color:var(--fg-3)">(층 1~3 TM=32 · 층 4 TM=16)</span></div></div>`;
  } else {
    subLbl.textContent = '배치 크기 (d=4,096 RTX 4060 실측):';
    [['512', 'B = 512'], ['1024', 'B = 1,024'], ['2048', 'B = 2,048']].forEach(([val, lbl]) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
      b.textContent = lbl; b.setAttribute('aria-pressed', S.p2b === val);
      b.onclick = () => { S.p2b = val; drawP2AndTimeline(); };
      subChips.appendChild(b);
    });
    const R = B.hw_bench['4096'].rows.find(r => String(r.B) === S.p2b);
    const H = 266, pad = 12, lx = 86, rx = 18;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A3V4 어텐션 중첩 vs 직렬 E2E 타임라인' }, host);
    const maxMs = R.base_ms;
    const barW = W - lx - rx;
    const sx = ms => lx + (ms / maxMs) * barW;

    // Section 1: Overlap mode (per-layer margin + 4L total)
    const marginX = (R.attn_1l_ms / (Math.max(...R.dispatch_1l_ms) + R.h2d_ms / 4)).toFixed(1);
    const tlHdr = W < 470
      ? `① 어텐션 중첩 — 1층 ${R.attn_1l_ms.toFixed(1)}ms 뒤 완벽 은폐 (${marginX}× 여유)`
      : `① 어텐션 중첩 서빙 (표준 트랜스포머) — 1층 어텐션 ${R.attn_1l_ms.toFixed(2)}ms 동안 CPU+H2D 완벽 은폐 (여유 ${marginX}×)`;
    tx(svg, pad, 16, tlHdr, { fill: C.on, 'font-size': W < 470 ? 9.2 : 10.4, 'font-weight': 700 });

    // 1-layer overlap mini-Gantt
    const yG = 28, gH = 20;
    const attn1 = R.attn_1l_ms, disp1 = Math.max(...R.dispatch_1l_ms), h2d1 = R.h2d_ms / 4;
    const max1 = Math.max(attn1, disp1 + h2d1) * 1.15;
    const sx1 = ms => lx + (ms / max1) * barW;
    tx(svg, lx - 6, yG + 14, 'GPU 1층', { 'text-anchor': 'end', fill: C.fg2, 'font-size': 10 });
    el('rect', { x: lx, y: yG, width: sx1(attn1) - lx, height: gH, rx: 4, fill: 'rgba(157,140,255,0.22)', stroke: C.neg, 'stroke-width': 1.2 }, svg);
    tx(svg, lx + (sx1(attn1) - lx) / 2, yG + 13.5, `1층 Attention (${attn1.toFixed(2)} ms)`, { 'text-anchor': 'middle', fill: C.neg, 'font-size': 9.8, 'font-weight': 600 });

    const yC = yG + gH + 6;
    tx(svg, lx - 6, yC + 14, 'CPU+H2D', { 'text-anchor': 'end', fill: C.fg2, 'font-size': 10 });
    const wD1 = Math.max(24, sx1(disp1) - lx), wH1 = Math.max(10, sx1(disp1 + h2d1) - sx1(disp1));
    el('rect', { x: lx, y: yC, width: wD1, height: gH, rx: 4, fill: C.onsoft, stroke: C.on, 'stroke-width': 1.2 }, svg);
    el('rect', { x: lx + wD1 + 1, y: yC, width: wH1, height: gH, rx: 3, fill: C.pick }, svg);
    tx(svg, lx + wD1 + wH1 + 6, yC + 13.5, `A3V4 ${disp1.toFixed(2)} + H2D ${h2d1.toFixed(2)}ms → 노출 0.00ms!`, { fill: C.on, 'font-size': W < 470 ? 8.6 : 9.6, 'font-weight': 600 });

    el('line', { x1: pad, x2: W - pad, y1: 88, y2: 88, stroke: C.line, 'stroke-width': 1 }, svg);

    // Section 2: 4-Layer FFN E2E comparison (Base vs Overlap E2E vs Zero-Overlap Serial E2E)
    tx(svg, pad, 108, `② 4개 층 전체 FFN 벽시계 비교 (B=${nf(R.B)}, d=4,096 · Base = ${R.base_ms.toFixed(2)}ms)`, { class: 'hi', 'font-size': W < 470 ? 9 : 10.6 });
    const rowsT = [
      { lbl: '풀 베이스', sub: 'cuBLAS', segs: [{ ms: R.base_ms, col: C.fg3, op: 0.55, txt: `${R.base_ms.toFixed(2)}ms (1.00×)` }], endTxt: '' },
      { lbl: '어텐션 중첩', sub: '노출 0.00ms', segs: [{ ms: R.gpu_ms, col: C.on, op: 0.95, txt: '' }], endTxt: `${R.e2e_ovlp_ms.toFixed(2)}ms (${R.e2e_ovlp_x.toFixed(2)}×)` , endCol: C.on },
      { lbl: '직렬 E2E', sub: '중첩 없음', segs: [
        { ms: R.dispatch_4l_ms, col: C.neg, op: 0.9, txt: '' },
        { ms: R.h2d_ms, col: C.pick, op: 0.9, txt: '' },
        { ms: R.gpu_ms, col: C.on, op: 0.95, txt: '' },
      ], endTxt: `${R.e2e_serial_ms.toFixed(2)}ms (${R.e2e_serial_x.toFixed(2)}×)`, endCol: C.pick },
    ];
    rowsT.forEach((rt, idx) => {
      const y = 120 + idx * 42;
      tx(svg, lx - 6, y + 12, rt.lbl, { 'text-anchor': 'end', fill: idx === 1 ? C.on : (idx === 2 ? C.pick : C.fg2), 'font-size': 10.5, 'font-weight': idx > 0 ? 700 : 500 });
      tx(svg, lx - 6, y + 25, rt.sub, { 'text-anchor': 'end', fill: C.fg3, 'font-size': 9 });
      let curX = lx;
      rt.segs.forEach(sg => {
        const sw = Math.max(3, (sg.ms / maxMs) * barW);
        el('rect', { x: curX, y: y + 4, width: sw - 1, height: 20, rx: 3, fill: sg.col, opacity: sg.op }, svg);
        if (sg.txt) tx(svg, curX + sw - 6, y + 17.5, sg.txt, { 'text-anchor': 'end', fill: C.bg, 'font-size': 9.8, 'font-weight': 700 });
        curX += sw;
      });
      if (rt.endTxt) tx(svg, curX + 6, y + 17.5, rt.endTxt, { fill: rt.endCol, 'font-size': 10.5, 'font-weight': 700 });
    });
    tx(svg, W < 470 ? pad : lx, 254, W < 470 ? '직렬 구성: 보라 A3V4 4층 · 주황 H2D · 청록 GPU v4 FFN' : '직렬 E2E 막대 구성: 보라 = CPU A3V4 4층 디스패치 · 주황 = H2D 전송 · 청록 = GPU v4 FFN 커널', { class: 'lab', 'font-size': W < 470 ? 8.8 : 9.4 });

    $('#bt-p2t-kpi').innerHTML =
      `<div class="panel"><div class="k">어텐션 중첩 E2E (표준 트랜스포머 · B=${nf(R.B)})</div><div class="v on">${R.e2e_ovlp_x.toFixed(2)}× <span style="font-size:12px;color:var(--fg-3)">(${R.base_ms.toFixed(2)} → ${R.e2e_ovlp_ms.toFixed(2)} ms · 노출 0.00ms)</span></div></div>
       <div class="panel"><div class="k">중첩 없는 직렬 E2E (CPU+H2D+GPU 전부 합산)</div><div class="v pick">${R.e2e_serial_x.toFixed(2)}× <span style="font-size:12px;color:var(--fg-3)">(A3V4 ${R.dispatch_4l_ms.toFixed(2)} + H2D ${R.h2d_ms.toFixed(2)} + GPU ${R.gpu_ms.toFixed(2)} ms)</span></div></div>`;
  }
}

// ---- b5: Roofline theoretical ceiling ----
function drawFinal() {
  const host = $('#bt-res'); host.innerHTML = '';
  const items = B.summary.filter(r => r.id !== 'full');
  const W = Math.max(320, host.clientWidth || 540), lx = 132, rx = 68, bh = 13, gh = 52, top = 28;
  const H = top + items.length * gh + 22;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'CH2 이론 상한 비교' }, host);
  const sx = v => lx + v / 100 * (W - lx - rx);
  for (let v = 0; v <= 100; v += 20) {
    el('line', { x1: sx(v), x2: sx(v), y1: top - 6, y2: H - 20, stroke: C.line, 'stroke-width': 1 }, svg);
    tx(svg, sx(v), H - 6, v + '%', { 'text-anchor': 'middle' });
  }
  el('line', { x1: sx(100), x2: sx(100), y1: top - 12, y2: H - 20, stroke: C.fg2, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
  tx(svg, sx(100), top - 16, '실제 풀 모델 = 100%', { 'text-anchor': 'end', class: 'lab' });
  const shortLab = {
    single1: ['단일 요청 (B=1)', '로컬 · 타일링 없음'],
    srv512:  ['실제 서버 (512)', 'A3V4 5단계 · W=512'],
    srv1024: ['실제 서버 (1,024)', 'A3V4 5단계 · W=512'],
    srv2048: ['실제 서버 (2,048)', 'A3V4 5단계 · W=512'],
  };
  items.forEach((R, i) => {
    const y = top + i * gh, isB1 = R.id === 'single1', [l1, l2] = shortLab[R.id];
    tx(svg, lx - 10, y + 10, l1, { 'text-anchor': 'end', class: 'hi', 'font-size': 11, fill: isB1 ? C.neg : C.fg });
    tx(svg, lx - 10, y + 24, l2, { 'text-anchor': 'end', 'font-size': 9.5 });
    const w1 = Math.max(0, sx(R.ones_pct) - sx(0) - (R.zero_pct > 0 ? 1 : 0));
    const w0 = Math.max(0, sx(R.compute_pct) - sx(R.ones_pct));
    el('rect', { x: sx(0), y, width: w1, height: bh, rx: 2, fill: isB1 ? C.neg : C.on }, svg);
    if (w0 > 0) el('rect', { x: sx(R.ones_pct), y, width: w0, height: bh, rx: 2, fill: C.pick }, svg);
    tx(svg, sx(R.compute_pct) + 6, y + bh - 2.5, 'FFN ' + R.compute_pct.toFixed(1) + '%', { fill: isB1 ? C.neg : C.on, 'font-size': 10.5 });
    const dramW = Math.max(2, sx(R.dram_pct) - sx(0));
    el('rect', { x: sx(0), y: y + bh + 4, width: dramW, height: bh, rx: 2, fill: C.fg2, opacity: 0.82 }, svg);
    if (R.dram_pct > 30) tx(svg, sx(R.dram_pct) - 6, y + 2 * bh + 1, 'DRAM ' + R.dram_pct.toFixed(1) + '%', { 'text-anchor': 'end', fill: C.bg, 'font-size': 9.5, 'font-weight': 600 });
    else tx(svg, sx(R.dram_pct) + 6, y + 2 * bh + 1, 'DRAM ' + R.dram_pct.toFixed(1) + '%', { fill: C.fg2, 'font-size': 9.5, 'font-weight': 600 });
    const bx = W - rx + 12;
    el('rect', { x: bx, y: y + 3, width: 52, height: 24, rx: 12, fill: isB1 ? 'rgba(157,140,255,0.16)' : C.onsoft, stroke: isB1 ? C.neg : C.on, 'stroke-width': 1 }, svg);
    tx(svg, bx + 26, y + 19, R.speedup.toFixed(1) + '배', { 'text-anchor': 'middle', fill: isB1 ? C.neg : C.on, 'font-size': 11, 'font-weight': 600 });
  });
  $('#bt-ftbl').innerHTML =
    '<tr><th>환경 구분</th><th>배치 (토큰)</th><th>적용 방식</th><th>FFN 연산량</th><th>DRAM 읽기</th><th>이론 상한</th></tr>' +
    B.summary.map(r => {
      const cls = r.id === 'single1' ? ' class="b1-hi"' : (r.id === 'srv2048' ? ' class="sum"' : '');
      const pillCls = r.id === 'single1' ? 'bt-pill neg' : 'bt-pill';
      return `<tr${cls}>
        <td><b>${r.name}</b> <span class="mut">${r.sub}</span></td>
        <td class="num">${r.batch}</td><td>${r.method}</td>
        <td class="num${r.id !== 'full' ? ' ok' : ''}">${r.compute_pct.toFixed(1)}%</td>
        <td class="num">${r.dram_pct.toFixed(1)}%</td>
        <td class="num">${r.id === 'full' ? '1.0배' : `<span class="${pillCls}">${r.speedup.toFixed(1)}배</span> <span class="mut">${r.speedup_note}</span>`}</td>
      </tr>`;
    }).join('');
}

// ---- b6: RTX 4060 Final E2E Silicon Benchmark (Explicitly separating Attention-Overlap E2E & Zero-Overlap Serial E2E!) ----
function drawHwBench() {
  const H1 = B.hw_bench['1024'], H4 = B.hw_bench['4096'];
  const fmtLayers = arr => arr.map(v => (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + '×').join(' / ');
  const maxB1 = Math.max(H1.b1.speedup, H4.b1.speedup, 6.5);
  const b1Line = (tag, sub, b1) => {
    const wBase = (100 * 1.0 / maxB1).toFixed(1), wOurs = (100 * b1.speedup / maxB1).toFixed(1);
    return `
      <div style="display:grid;gap:3px">
        <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:4px;font-size:12px">
          <span><b style="color:var(--fg)">${tag}</b> <span style="color:var(--fg-3);font-size:11px">${sub}</span></span>
          <span class="bt-pill neg">${b1.speedup.toFixed(2)}배 (${b1.base_ms.toFixed(3)}ms → ${b1.ours_ms.toFixed(3)}ms)</span>
        </div>
        <div style="display:grid;grid-template-columns:82px 1fr 142px;align-items:center;gap:8px;font-family:var(--mono);font-size:11px">
          <span style="color:var(--fg-3);white-space:nowrap">풀 베이스</span>
          <div style="height:8px;background:var(--bg-3);border-radius:3px;overflow:hidden"><div style="width:${wBase}%;height:100%;background:var(--fg-3);opacity:0.6"></div></div>
          <span style="color:var(--fg-2);text-align:right;white-space:nowrap">1.00× (0마스크 ${b1.speedup_mask0.toFixed(2)}×)</span>
        </div>
        <div style="display:grid;grid-template-columns:82px 1fr 142px;align-items:center;gap:8px;font-family:var(--mono);font-size:11px">
          <span style="color:var(--neg);font-weight:600;white-space:nowrap">CH2 단일</span>
          <div style="height:8px;background:var(--bg-3);border-radius:3px;overflow:hidden"><div style="width:${wOurs}%;height:100%;background:var(--neg)"></div></div>
          <span style="color:var(--neg);font-weight:600;text-align:right;white-space:nowrap">${b1.speedup.toFixed(2)}× (${b1.ours_gbs.toFixed(0)} GB/s)</span>
        </div>
      </div>`;
  };
  $('#bt-hw-b1').innerHTML = `
    <div class="hdr">
      <span><b>① 단일 요청 (B=1 · 로컬/개인 LLM)</b> — CH2 활성 뉴런(표본 ${H1.b1.avg_active_pct.toFixed(1)}%, 전체 13.52%) 직접 읽기</span>
      <span class="lbl">10k 시험 ${B.accuracy.test_orig.toFixed(2)}% (99.77%)</span>
    </div>
    ${b1Line('d = 1,024', '실제 CH2 가중치 (4층 101MB > L2)', H1.b1)}
    <div style="height:1px;background:var(--line)"></div>
    ${b1Line('d = 4,096', '큰 모델 LLM 층 (4층 403MB > L2)', H4.b1)}
  `;

  // Server bar chart: 0..5.0x comparing Base, 0-Mask, Serial E2E, GPU/Overlap E2E
  const host = $('#bt-hw-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), lx = 84, rx = 62, bh = 9.5, bgap = 2.5, gh = 58;
  const grpHdr = 22, grpGap = 16, top = 28, blockH = grpHdr + 3 * gh;
  const svgH = top + blockH + grpGap + blockH + 22;
  const svg = el('svg', { viewBox: `0 0 ${W} ${svgH}`, role: 'img', 'aria-label': 'RTX 4060 서버 E2E 실측 배수 (중첩 vs 직렬 분리)' }, host);
  const maxS = 5.0, sx = v => lx + Math.min(v, maxS) / maxS * (W - lx - rx);

  tx(svg, 4, 13, '② 실제 서버 환경 (CH2 + v4 + A3V4 · 4개 층 전체 합계)', { class: 'hi', 'font-size': 11.5 });
  for (const v of [0, 1.0, 2.0, 3.0, 4.0, 5.0]) {
    el('line', { x1: sx(v), x2: sx(v), y1: top - 4, y2: svgH - 18, stroke: v === 1.0 ? C.fg2 : C.line, 'stroke-width': v === 1.0 ? 1.3 : 1, 'stroke-dasharray': v === 1.0 ? '3 3' : '' }, svg);
    tx(svg, sx(v), svgH - 4, v.toFixed(1) + '배', { 'text-anchor': 'middle', 'font-size': 10 });
  }
  const drawGrp = (H, yStart, title, isLarge) => {
    tx(svg, 4, yStart + 13, title, { fill: isLarge ? C.on : C.fg, 'font-size': W < 470 ? 9.8 : 11, 'font-weight': 600 });
    H.rows.forEach((R, i) => {
      const y0 = yStart + grpHdr + i * gh;
      tx(svg, lx - 8, y0 + 18, '배치 ' + nf(R.B), { 'text-anchor': 'end', class: 'hi', 'font-size': 11 });
      tx(svg, lx - 8, y0 + 31, R.base_ms.toFixed(1) + 'ms', { 'text-anchor': 'end', 'font-size': 9.5 });
      const bars = [
        { v: 1.0,            col: C.fg3,  op: 0.55, lbl: '1.00×' },
        { v: R.mask0_x,      col: C.pick, op: 0.85, lbl: R.mask0_x.toFixed(2) + '×' },
        { v: R.e2e_serial_x, col: C.neg,  op: 0.90, lbl: R.e2e_serial_x.toFixed(2) + '× (직렬)' },
        { v: R.gpu_x,        col: C.on,   op: 0.96, lbl: R.gpu_x.toFixed(2) + (isLarge ? '× (중첩)' : '× (GPU)') },
      ];
      bars.forEach((br, k) => {
        const y = y0 + k * (bh + bgap), bw = Math.max(2, sx(br.v) - sx(0));
        el('rect', { x: sx(0), y, width: bw, height: bh, rx: 2, fill: br.col, opacity: br.op }, svg);
        tx(svg, sx(br.v) + 4, y + bh - 1.2, br.lbl, { fill: k >= 2 ? br.col : C.fg2, 'font-size': 9.2, 'font-weight': k >= 2 ? 600 : 400 });
      });
    });
  };
  drawGrp(H1, top, '작은 모델 · d = 1,024 (실제 CH2 가중치 · 층 24MB < L2)', false);
  el('line', { x1: 4, x2: W - 4, y1: top + blockH + grpGap / 2 - 2, y2: top + blockH + grpGap / 2 - 2, stroke: C.line2, 'stroke-width': 1, 'stroke-dasharray': '4 3' }, svg);
  drawGrp(H4, top + blockH + grpGap, '큰 모델 · d = 4,096 (대형 LLM 층 · 층 96MB > L2 · 어텐션 중첩 노출 0.00ms)', true);

  const buildRows = (H, grpTitle, isLarge) => {
    const b1 = H.b1;
    const hdr = `<tr class="grp"><td colspan="8">${grpTitle}</td></tr>`;
    const b1Row = `<tr class="b1-hi">
      <td><b>단일 (B=1)</b></td>
      <td class="num">${b1.base_ms.toFixed(3)} ms</td>
      <td class="num warn">${b1.speedup_mask0.toFixed(2)}×</td>
      <td class="num"><b style="color:var(--neg)">${b1.ours_ms.toFixed(3)} ms</b> <span class="bt-pill neg">${b1.speedup.toFixed(2)}×</span></td>
      <td class="num mut">불필요 (0 ms)</td>
      <td class="num"><span class="bt-pill neg">${b1.speedup.toFixed(2)}배</span></td>
      <td class="num"><span class="bt-pill neg">${b1.speedup.toFixed(2)}배</span></td>
      <td class="num">${fmtLayers(b1.layer_speedups)}</td>
    </tr>`;
    const srv = H.rows.map(r => {
      const hi = r.B === 2048;
      const dispH2d = `${r.dispatch_4l_ms.toFixed(2)} + ${r.h2d_ms.toFixed(2)} ms`;
      const ovlpCell = isLarge
        ? `<span class="bt-pill">${r.e2e_ovlp_x.toFixed(2)}배</span> <span class="mut">(노출 0ms)</span>`
        : `${r.e2e_ovlp_x.toFixed(2)}× <span class="mut">(노출 ${r.attn_exposed_ms.toFixed(1)}ms)</span>`;
      const serCell = `<b style="color:${r.e2e_serial_x >= 1.5 ? 'var(--neg)' : 'var(--fg-2)'}">${r.e2e_serial_x.toFixed(2)}배</b> <span class="mut">(${r.e2e_serial_ms.toFixed(2)}ms)</span>`;
      return `<tr${hi ? ' class="sum"' : ''}>
        <td class="num"><b>서버 (${nf(r.B)})</b></td>
        <td class="num">${r.base_ms.toFixed(2)} ms</td>
        <td class="num warn">${r.mask0_x.toFixed(2)}×</td>
        <td class="num ok"><b>${r.gpu_ms.toFixed(2)} ms</b> <span class="mut">(${r.gpu_x.toFixed(2)}×)</span></td>
        <td class="num mut">${dispH2d}</td>
        <td class="num${isLarge ? ' ok' : ''}">${ovlpCell}</td>
        <td class="num">${serCell}</td>
        <td class="num">${fmtLayers(r.layer_speedups)}</td>
      </tr>`;
    }).join('');
    return hdr + b1Row + srv;
  };
  $('#bt-hwtbl').innerHTML =
    '<tr><th>환경 · 배치</th><th>풀 베이스 (cuBLAS)</th><th>0 마스킹</th><th>GPU FFN (v4 커널)</th><th>A3V4 + H2D</th><th>어텐션 중첩 E2E</th><th>중첩 없는 직렬 E2E</th><th>층별 GPU 배수 (1/2/3/4층)</th></tr>' +
    buildRows(H1, '작은 모델 · d = 1,024 (실제 CH2 가중치 · 1층 어텐션이 0.23~1.30ms로 매우 짧아 직렬/중첩 차이 발생)', false) +
    buildRows(H4, '큰 모델 · d = 4,096 (대형 LLM 층 · 1층 어텐션 3.24~13.30ms 뒤로 A3V4+H2D 100% 은폐 → 노출 0.00ms)', true);

  const maxErr = Math.max(H1.b1.max_rel_err, H4.b1.max_rel_err, ...H1.rows.map(r => r.max_rel_err), ...H4.rows.map(r => r.max_rel_err));
  $('#bt-hw-kpi').innerHTML =
    `<div class="panel"><div class="k">밀집 64×32 커널 기준선 검증 (v4 vs cuBLAS)</div><div class="v on" style="font-size:14.5px">d=1,024 ${H1.dense_eff_pct.toFixed(1)}% · d=4,096 ${H4.dense_eff_pct.toFixed(1)}% <span style="font-size:11.5px;color:var(--fg-3)">(${H4.dense_v4_tf} / ${H4.dense_cublas_tf} TF)</span></div></div>
     <div class="panel"><div class="k">출력 정합성 (PyTorch fp32 기준)</div><div class="v" style="font-size:14.5px">최대 오차 ${(maxErr * 100).toFixed(2)}% <span style="font-size:11.5px;color:var(--fg-3)">(단일·서버 전 설정 · 필수 1 누락 0건)</span></div></div>`;
}

// =========================================================================
// TAB 4: 핵심 기여 (4 Core Axes: #c1 Structure, #c2 Codesign, #c3 Tiling+A3V4, #c4 E2E Proof)
// =========================================================================
function drawC1Struct() {
  const btns = $('#ct-c1-btns'); btns.innerHTML = '';
  ['① 구조 비교 (예측기·라우터 vs 차등 중심점)', '② 배치별 뉴런 겹침 시뮬레이터 (M=32)'].forEach((lbl, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.c1mode === idx);
    b.onclick = () => { S.c1mode = idx; drawC1Struct(); };
    btns.appendChild(b);
  });
  $('#ct-c1-cap').textContent = S.c1mode === 0
    ? '학습 라우터 0개 · 1~3층 M=32 & 4층 M=256 차등 중심점 명단'
    : '1~3층 M=32로 배치가 커질수록 같은 중심점에 16~256토큰이 자연 결집!';
  $('#ct-c1-brow').style.display = S.c1mode === 1 ? 'flex' : 'none';
  if (S.c1mode === 1) {
    const bc = $('#ct-c1-bchips'); bc.innerHTML = '';
    [[1, 'B = 1'], [512, 'B = 512'], [2048, 'B = 2,048'], [8192, 'B = 8,192 (프리필)']].forEach(([val, lbl]) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
      b.textContent = lbl; b.setAttribute('aria-pressed', S.c1b === val);
      b.onclick = () => { S.c1b = val; drawC1Struct(); };
      bc.appendChild(b);
    });
  }
  const host = $('#ct-c1-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), pad = 10;
  if (S.c1mode === 0) {
    const H = 250;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '예측기·라우터 vs 차등 중심점 구조 비교' }, host);
    const rows = [
      { tag: 'Deja Vu · ShadowLLM', step1: '별도 MLP 예측기', step2: '토큰별 산발 패턴', out: '토큰끼리 안 겹침 (붕괴)', col: C.pick, bg: C.bg3 },
      { tag: 'MoEfication · CMoE',  step1: '학습 라우터',     step2: '고정 전문가 분할', out: '공용 뉴런 공유 제약', col: C.fg2, bg: C.bg3 },
      { tag: 'Roster (차등 중심점)', step1: 'M=[32,32,32,256] 내적', step2: '겹침 허용 · 가변 K_j', out: '3행렬 스킵 + 64×32 대결집', col: C.on, bg: C.onsoft },
    ];
    const gap = 14, innerW = W - 2 * pad - 20, bw = (innerW - 2 * gap) / 3;
    rows.forEach((R, i) => {
      const y = 8 + i * 79;
      el('rect', { x: pad, y, width: W - 2 * pad, height: 70, rx: 8, fill: R.bg, stroke: i === 2 ? C.on : C.line2, 'stroke-width': i === 2 ? 1.5 : 1 }, svg);
      tx(svg, pad + 10, y + 19, R.tag, { fill: R.col, 'font-size': 11, 'font-weight': 700 });
      [R.step1, R.step2, R.out].forEach((txt, k) => {
        const bx = pad + 10 + k * (bw + gap), by = y + 29;
        el('rect', { x: bx, y: by, width: bw, height: 31, rx: 5, fill: C.bg, stroke: k === 2 ? R.col : C.line, 'stroke-width': k === 2 ? 1.2 : 0.9 }, svg);
        tx(svg, bx + bw / 2, by + 19.5, txt, { 'text-anchor': 'middle', fill: k === 2 ? R.col : C.fg2, 'font-size': Math.min(10, Math.max(8, bw / 14.5)), 'font-weight': k === 2 ? 600 : 400 });
        if (k < 2) tx(svg, bx + bw + gap / 2, by + 19.5, '→', { 'text-anchor': 'middle', fill: C.fg3, 'font-size': 10 });
      });
    });
    $('#ct-c1-kpi').innerHTML =
      `<div class="panel"><div class="k">학습된 예측기 / 라우터</div><div class="v on" style="font-size:15px">0개 <span style="font-size:11.5px;color:var(--fg-3)">(단순 중심점 내적 1회 · 3행렬 선제 스킵)</span></div></div>
       <div class="panel"><div class="k">층별 차등 중심점 설계</div><div class="v" style="font-size:14.5px">1~3층 M=32 · 4층 M=256 <span style="font-size:11.5px;color:var(--fg-3)">(K=22/22/9.6/0.8%)</span></div></div>`;
  } else {
    const H = 290, bv = S.c1b;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'M=32 배치별 뉴런 겹침 시뮬레이터' }, host);
    const half = (W - 2 * pad - 12) / 2, rx0 = pad + half + 12;
    const meta = {
      1:    { union: 14, coal: '1개 / 중심점', comp: '13.5%', note: '타일링 불필요 · GEMV 6.27×' },
      512:  { union: 96, coal: '평균 16개 / 중심점 (L1~3)', comp: '20.4%', note: '채움 72.6% · 중첩 E2E 3.08×' },
      2048: { union: 100, coal: '평균 64개 / 중심점 (L1~3)', comp: '15.6%', note: '채움 87.4% · 중첩 E2E 4.57×' },
      8192: { union: 100, coal: '평균 256개 / 중심점 (L1~3)', comp: '~13.9%', note: '64×32 풀 포화 · 예상 5~6×' },
    }[bv];
    const gRows = 6, gCols = 8, gcs = Math.min(24, Math.floor((half - 44) / gCols));
    const gxL = pad + 32 + (half - 40 - gCols * gcs) / 2, gy0 = 52;
    const seedPat = [[1,0,0,1,0,0,1,0],[0,1,0,0,1,0,0,1],[0,0,1,0,0,1,1,0],[1,0,0,0,1,0,0,1],[0,1,0,1,0,0,1,0],[0,0,1,0,1,1,0,0]];
    el('rect', { x: pad, y: 6, width: half, height: H - 12, rx: 8, fill: C.bg3, stroke: C.line2 }, svg);
    tx(svg, pad + 10, 24, '기존 토큰별 예측 (Deja Vu 등)', { fill: C.pick, 'font-size': W < 470 ? 9.8 : 11, 'font-weight': 600 });
    tx(svg, pad + 10, 40, `배치 B = ${nf(bv)} · 토큰마다 패턴 제각각`, { class: 'lab', 'font-size': W < 470 ? 8.6 : 9.8 });
    for (let r = 0; r < gRows; r++) {
      const showRow = bv === 1 ? r === 0 : true;
      tx(svg, gxL - 6, gy0 + r * gcs + gcs / 2 + 3.5, 'T' + (r + 1), { 'text-anchor': 'end', fill: showRow ? C.fg2 : C.fg3, 'font-size': 9.5, opacity: showRow ? 1 : 0.3 });
      for (let c = 0; c < gCols; c++) {
        const on = showRow && seedPat[r][c] === 1, padZ = showRow && bv > 1 && !on && (bv >= 2048 || (r + c) % 2 === 0);
        el('rect', { x: gxL + c * gcs + 1.5, y: gy0 + r * gcs + 1.5, width: gcs - 3, height: gcs - 3, rx: 3, fill: on ? C.on : (padZ ? C.pick : C.bg), opacity: showRow ? (padZ ? 0.78 : 1) : 0.2 }, svg);
        if (padZ && gcs >= 18) tx(svg, gxL + c * gcs + gcs / 2, gy0 + r * gcs + gcs / 2 + 3, '×0', { 'text-anchor': 'middle', fill: C.bg, 'font-size': 8, 'font-weight': 700 });
      }
    }
    const barY = gy0 + gRows * gcs + 22, barW = half - 20;
    tx(svg, pad + 10, barY, `배치 합집합 밀도: ${meta.union}%`, { fill: meta.union > 80 ? C.pick : C.fg2, 'font-size': W < 470 ? 9 : 10, 'font-weight': 600 });
    el('rect', { x: pad + 10, y: barY + 6, width: barW, height: 11, rx: 3, fill: C.bg }, svg);
    el('rect', { x: pad + 10, y: barY + 6, width: barW * meta.union / 100, height: 11, rx: 3, fill: meta.union > 80 ? C.pick : C.on }, svg);
    tx(svg, pad + 10, barY + 33, meta.union >= 95 ? '→ 사실상 밀집 연산 (가속 붕괴)' : '→ B=1에서만 희소', { class: 'lab', 'font-size': W < 470 ? 8.6 : 9.8 });

    el('rect', { x: rx0, y: 6, width: half, height: H - 12, rx: 8, fill: C.bg3, stroke: C.on, 'stroke-width': 1.3 }, svg);
    tx(svg, rx0 + 10, 24, '차등 중심점 Roster (M=32)', { fill: C.on, 'font-size': W < 470 ? 9.8 : 11, 'font-weight': 700 });
    tx(svg, rx0 + 10, 40, W < 470 ? meta.coal : `같은 중심점: ${meta.coal}`, { class: 'lab', 'font-size': W < 470 ? 8.4 : 9.5 });
    const gxR = rx0 + 36 + (half - 44 - gCols * gcs) / 2, bandCols = [C.on, C.neg, '#4cd69c'], bandNames = ['A', 'B', 'C'];
    for (let r = 0; r < gRows; r++) {
      const showRow = bv === 1 ? r === 0 : true;
      const grp = bv >= 2048 ? Math.floor(r / 3) : Math.floor(r / 2) % 3;
      tx(svg, gxR - 6, gy0 + r * gcs + gcs / 2 + 3.5, `T${r + 1}(${bandNames[grp]})`, { 'text-anchor': 'end', fill: showRow ? bandCols[grp] : C.fg3, 'font-size': 9, 'font-weight': 600, opacity: showRow ? 1 : 0.3 });
      for (let c = 0; c < gCols; c++) {
        const blk = (grp === 0 && c < 3) || (grp === 1 && c >= 3 && c < 6) || (grp === 2 && c >= 5 && c < 8);
        el('rect', { x: gxR + c * gcs + 1.5, y: gy0 + r * gcs + 1.5, width: gcs - 3, height: gcs - 3, rx: 3, fill: (showRow && blk) ? bandCols[grp] : C.bg, opacity: showRow ? 0.92 : 0.2 }, svg);
      }
    }
    if (bv >= 512) {
      const bRows = bv >= 2048 ? 3 : 2;
      el('rect', { x: gxR, y: gy0, width: 3 * gcs, height: bRows * gcs, rx: 4, fill: 'none', stroke: C.fg, 'stroke-width': 1.5 }, svg);
      el('rect', { x: gxR + 3 * gcs, y: gy0 + bRows * gcs, width: 3 * gcs, height: bRows * gcs, rx: 4, fill: 'none', stroke: C.fg, 'stroke-width': 1.5 }, svg);
    }
    tx(svg, rx0 + 10, barY, `FFN 연산량: ${meta.comp}`, { fill: C.on, 'font-size': W < 470 ? 9 : 10, 'font-weight': 600 });
    const compNum = parseFloat(meta.comp.replace('~', ''));
    el('rect', { x: rx0 + 10, y: barY + 6, width: barW, height: 11, rx: 3, fill: C.bg }, svg);
    el('rect', { x: rx0 + 10, y: barY + 6, width: barW * compNum / 100, height: 11, rx: 3, fill: C.on }, svg);
    tx(svg, rx0 + 10, barY + 33, '→ ' + meta.note, { fill: C.on, 'font-size': W < 470 ? 8.4 : 9.5, 'font-weight': 500 });

    $('#ct-c1-kpi').innerHTML =
      `<div class="panel"><div class="k">배치 B = ${nf(bv)} 같은 중심점 결집</div><div class="v on" style="font-size:14.5px">${meta.coal}</div></div>
       <div class="panel"><div class="k">기존 합집합 vs 우리 FFN 연산</div><div class="v" style="font-size:14.5px">기존 ${meta.union}% → <b style="color:var(--on)">우리 ${meta.comp}</b></div></div>`;
  }
}

// ---- #c2 (NEW Contribution 2): Tile/Block-Friendly Shared Post-Training Codesign ----
function drawC2Codesign() {
  const chips = $('#ct-c2-bchips'); chips.innerHTML = '';
  [['512', 'B = 512'], ['1024', 'B = 1,024'], ['2048', 'B = 2,048']].forEach(([val, lbl]) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.c2b === val);
    b.onclick = () => { S.c2b = val; drawC2Codesign(); };
    chips.appendChild(b);
  });
  const R = B.hw_bench['4096'].rows.find(r => String(r.B) === S.c2b);
  const host = $('#ct-c2-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), lx = W < 470 ? 86 : 104, rx = W < 470 ? 74 : 78, top = 28, gh = 42;
  const metrics = [
    { lbl: 'P1 연산량', a: R.a_p1_comp_pct, b: R.p1_comp_pct, max: 45, unit: '%', lower: true },
    { lbl: 'P2 합집합', a: R.a_p2_union_pct, b: R.p2_union_pct, max: 45, unit: '%', lower: true },
    { lbl: '×0 낭비 비율', a: R.a_x0_pct, b: R.x0_pct, max: 60, unit: '%', lower: true },
    { lbl: 'DRAM 읽기', a: R.a_dram_read_pct, b: R.dram_read_pct, max: 100, unit: '%', lower: true },
    { lbl: 'A3V4 4층 시간', a: R.a_dispatch_4l_ms, b: R.dispatch_4l_ms, max: 8.0, unit: 'ms', lower: true },
    { lbl: '중첩 E2E 배수', a: R.track_a_x, b: R.e2e_ovlp_x, max: 5.0, unit: '×', lower: false },
  ];
  const H = top + metrics.length * gh + 10;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '개별 몰기 대비 B1+B2 공유 사후 학습 코디자인 효과' }, host);
  const barW = W - lx - rx;
  tx(svg, W < 470 ? 6 : lx, 15, W < 470 ? `동일 v4+A3V4 커널 위 코디자인 효과 (d=4,096, B=${nf(R.B)})` : `같은 RTX 4060 v4+A3V4 커널 위 순수 사후 학습 코디자인 효과 (d=4,096, B=${nf(R.B)})`, { class: 'hi', 'font-size': W < 470 ? 9.4 : 10.5 });
  metrics.forEach((m, i) => {
    const y = top + i * gh;
    tx(svg, lx - 8, y + 19, m.lbl, { 'text-anchor': 'end', fill: C.fg2, 'font-size': 10.5, 'font-weight': 500 });
    const wA = Math.max(4, (m.a / m.max) * barW), wB = Math.max(4, (m.b / m.max) * barW);
    el('rect', { x: lx, y: y + 3, width: wA, height: 11, rx: 2.5, fill: C.fg3, opacity: 0.65 }, svg);
    tx(svg, lx + wA + 5, y + 11.5, `${m.a.toFixed(m.unit === 'ms' || m.unit === '×' ? 2 : 1)}${m.unit}`, { fill: C.fg3, 'font-size': 9.5 });
    el('rect', { x: lx, y: y + 17, width: wB, height: 12, rx: 2.5, fill: C.on }, svg);
    const diff = m.lower ? `-${(100 * (1 - m.b / m.a)).toFixed(0)}%` : `+${(100 * (m.b / m.a - 1)).toFixed(0)}%`;
    tx(svg, lx + wB + 5, y + 26.5, `${m.b.toFixed(m.unit === 'ms' || m.unit === '×' ? 2 : 1)}${m.unit} (${diff})`, { fill: C.on, 'font-size': 9.8, 'font-weight': 700 });
  });
  $('#ct-c2-kpi').innerHTML =
    `<div class="panel"><div class="k">동일 커널 순수 GPU 가속 이득 (B=${nf(R.B)})</div><div class="v on">+${R.b_vs_a_gpu_gain_pct.toFixed(1)}% <span style="font-size:12px;color:var(--fg-3)">(${R.track_a_x.toFixed(2)}× → ${R.gpu_x.toFixed(2)}× · 직렬 ${R.a_e2e_serial_x.toFixed(2)}× → ${R.e2e_serial_x.toFixed(2)}×)</span></div></div>
     <div class="panel"><div class="k">정확도 · 배치 일관성</div><div class="v" style="font-size:14.5px">10,000장 90.81% (99.77%) <span style="font-size:11.5px;color:var(--fg-3)">· 1*=0% · 배치 무관 동일 출력</span></div></div>`;
}

// ---- #c3 (Contribution 3): Non-Grid 5-Tier Tiling + A3V4 Dispatcher + Real CH2 10-Trial BSMR Fold ----
const C3_CENTROIDS = [
  { name: '중심점 A (부츠)', raw: [0.62, 0.25, 0.36, 0.71, 0.14, 0.22, 0.68, 0.18, 0.15, 0.58, 0.19, 0.12], top: [0, 3, 6, 9], opt: 2 },
  { name: '중심점 B (셔츠)', raw: [0.16, 0.69, 0.21, 0.14, 0.64, 0.34, 0.18, 0.72, 0.15, 0.12, 0.59, 0.20], top: [1, 4, 7, 10], opt: 5 },
  { name: '중심점 C (가방)', raw: [0.35, 0.18, 0.74, 0.15, 0.21, 0.67, 0.13, 0.19, 0.65, 0.14, 0.16, 0.61], top: [2, 5, 8, 11], opt: 0 },
];
const C3_NG_TILES = [
  { id: 1, tag: '①', rows: [0, 4], cols: [0, 3, 6],  col: '#57d5e7', lbl: '{T1,T5}×{N1,N4,N7}',  stat: '5/6 · ×0 1칸' },
  { id: 2, tag: '②', rows: [1, 5], cols: [1, 8, 11], col: '#9d8cff', lbl: '{T2,T6}×{N2,N9,N12}', stat: '5/6 · ×0 1칸' },
  { id: 3, tag: '③', rows: [2, 5], cols: [2, 4, 7],  col: '#4cd69c', lbl: '{T3,T6}×{N3,N5,N8}',  stat: '5/6 · ×0 1칸' },
  { id: 4, tag: '④', rows: [1, 3], cols: [0, 5, 7],  col: '#5aa9ff', lbl: '{T2,T4}×{N1,N6,N8}',  stat: '4/6 · ×0 2칸' },
  { id: 5, tag: '⑤', rows: [2, 4], cols: [1, 5, 9],  col: '#e38ad8', lbl: '{T3,T5}×{N2,N6,N10}', stat: '3/6 · ×0 3칸' },
];

function drawC3Tiling() {
  const btns = $('#ct-c3-btns'); btns.innerHTML = '';
  ['① MatFormer (정적)', '② 기존 SDDMM (BSMR)', '③ 선택적 1* 원리', '④ 우리: 비격자 5단계 + A3V4'].forEach((label, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', S.c3mode === idx);
    b.onclick = () => { S.c3mode = idx; drawC3Tiling(); };
    btns.appendChild(b);
  });
  const caps = [
    '같은 6×12 표 · 앞 4열 전역 고정 → 문맥별 필수 뉴런 대거 누락',
    '같은 6×12 표 · 고정 행 패널 + 패널별 열 정렬 · 7틀 중 42.9%가 ×0',
    '선택적 자유도(1*): 필요 시 꼬리 뉴런 OFF 가능 (최종 CH2는 1*=0%로 충분)',
    '같은 6×12 표 · 타일마다 행·열 자유 조합 → 7틀→5틀, ×0 18→8칸 · 실제 CH2 64×32 비중 76.7%',
  ];
  $('#ct-c3-cap').textContent = caps[S.c3mode];
  const subRow = $('#ct-c3-subrow'), subLbl = $('#ct-c3-sublbl'), cc = $('#ct-c3-cchips');
  if (S.c3mode === 2) {
    subRow.style.display = 'flex'; subLbl.textContent = '중심점 선택 (각자 다른 뉴런 순위):'; cc.innerHTML = '';
    C3_CENTROIDS.forEach((c, idx) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
      b.textContent = c.name; b.setAttribute('aria-pressed', S.c3c === idx);
      b.onclick = () => { S.c3c = idx; drawC3Tiling(); }; cc.appendChild(b);
    });
  } else if (S.c3mode === 3) {
    subRow.style.display = 'flex'; subLbl.textContent = '비격자 2×3 틀 강조:'; cc.innerHTML = '';
    [{ idx: -1, lbl: '전체 5틀' }, ...C3_NG_TILES.map((t, i) => ({ idx: i, lbl: `${t.tag} ${t.lbl}` }))].forEach(it => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
      b.textContent = it.lbl; b.setAttribute('aria-pressed', S.c3t === it.idx);
      b.onclick = () => { S.c3t = it.idx; drawC3Tiling(); }; cc.appendChild(b);
    });
  } else subRow.style.display = 'none';

  const host = $('#ct-c3-svg'); host.innerHTML = '';
  const leg = $('#ct-c3-leg');
  const W = Math.max(340, host.clientWidth || 540), pad = 12;

  if (S.c3mode === 0) {
    const lx = 58, rx = 10, top = 34, cs = Math.min(36, (W - lx - rx) / 12), H = top + 6 * cs + 36;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'MatFormer 정적 앞 4열 예시' }, host);
    tx(svg, lx - 8, top - 11, '뉴런', { 'text-anchor': 'end', 'font-size': 10.5 });
    for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 11, 'N' + (c + 1), { 'text-anchor': 'middle', fill: c < 4 ? C.on : C.fg3, 'font-size': 10.5, 'font-weight': c < 4 ? 600 : 400 });
    for (let r = 0; r < 6; r++) tx(svg, lx - 8, top + r * cs + cs / 2 + 4, P1_ROWS[r], { 'text-anchor': 'end', class: 'lab', 'font-size': 11 });
    el('rect', { x: lx - 1, y: top - 1, width: 4 * cs + 2, height: 6 * cs + 2, rx: 5, fill: 'rgba(87,213,231,0.06)', stroke: C.on, 'stroke-width': 1.6 }, svg);
    for (let r = 0; r < 6; r++) for (let c = 0; c < 12; c++) {
      const v = P1_G[r][c], x = lx + c * cs + 2, y = top + r * cs + 2, s = cs - 4, inFix = c < 4;
      let fill = C.off, stroke = 'none', dash = '', txt = '', tfill = C.bg;
      if (inFix) { if (v === 1) fill = C.on; else if (v === 2) { fill = C.onsoft; stroke = C.on; dash = '3 2'; txt = '1*'; tfill = C.on; } else { fill = C.picksoft; stroke = C.pick; txt = '×0'; tfill = C.pick; } }
      else { if (v === 1) { fill = C.pick; txt = '누락'; } else if (v === 2) { fill = C.bg; stroke = C.line2; dash = '3 2'; txt = '1*'; tfill = C.fg3; } }
      const a = { x, y, width: s, height: s, rx: 3, fill };
      if (stroke !== 'none') { a.stroke = stroke; a['stroke-width'] = 1.3; if (dash) a['stroke-dasharray'] = dash; }
      el('rect', a, svg);
      if (txt) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3.5, txt, { 'text-anchor': 'middle', fill: tfill, 'font-size': Math.max(8.2, cs * 0.28), 'font-weight': 700 });
    }
    const yBot = top + 6 * cs + 10;
    tx(svg, lx + 2 * cs, yBot + 17, W < 470 ? '↑ 앞 4열 고정' : '↑ 전역 앞 4열 고정 (×0 17칸)', { 'text-anchor': 'middle', fill: C.on, 'font-size': W < 470 ? 8.6 : 10, 'font-weight': 600 });
    tx(svg, lx + 8 * cs, yBot + 17, W < 470 ? '← 필수 12개 누락' : '← N5~N12 문맥별 필수 뉴런 12개 누락', { 'text-anchor': 'middle', fill: C.pick, 'font-size': W < 470 ? 8.6 : 10, 'font-weight': 600 });
    leg.innerHTML = '<span><i style="background:var(--on)"></i>고정 구간 적중 (6)</span><span><i style="background:var(--pick)"></i>누락된 필수 1 (12)</span><span><i style="background:var(--pick-soft);box-shadow:inset 0 0 0 1.2px var(--pick)"></i>고정 구간 ×0 (17)</span>';
    $('#ct-c3-kpi').innerHTML = `<div class="panel"><div class="k">전역 정적 고정 (MatFormer)</div><div class="v pick" style="font-size:15px">필수 뉴런 12 / 18 누락 (66.7%)</div></div><div class="panel"><div class="k">실제 CH2 스케일 검증</div><div class="v" style="font-size:14.5px">동일 예산 시 필수 1의 절반 이상 누락</div></div>`;
  } else if (S.c3mode === 1) {
    drawBsmrGrid(host, true);
    leg.innerHTML = '<span><i style="background:var(--on)"></i>필수 1</span><span><i style="background:var(--on-soft);box-shadow:inset 0 0 0 1.5px var(--on)"></i>선택 1*</span><span><i style="background:var(--pick)"></i>×0 (18칸)</span><span><i style="background:var(--off)"></i>빈 틀 (5개)</span>';
    $('#ct-c3-kpi').innerHTML = `<div class="panel"><div class="k">기존 SDDMM (BSMR · 행 패널 고정)</div><div class="v pick" style="font-size:15px">7틀 (42칸) · ×0 18칸 (42.9%)</div></div><div class="panel"><div class="k">구조적 한계</div><div class="v" style="font-size:14.5px">행 패널 경계에 묶여 자유 재조합 불가</div></div>`;
  } else if (S.c3mode === 2) {
    const H = 248, cur = C3_CENTROIDS[S.c3c], bw = (W - 2 * pad - 11 * 4) / 12, barMax = 50;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '중심점별 앞쪽 몰기와 선택적 1* 구조' }, host);
    tx(svg, pad, 16, `${cur.name} · 학습 전 (중요도 산발 분포)`, { class: 'hi', 'font-size': 11 });
    cur.raw.forEach((v, i) => {
      const x = pad + i * (bw + 4), bh = Math.max(4, v * barMax), y = 76 - bh;
      const isTop = cur.top.includes(i), isOpt = i === cur.opt;
      el('rect', { x, y, width: bw, height: bh, rx: 3, fill: isTop ? C.on : (isOpt ? C.neg : C.off), opacity: isTop ? 0.9 : 0.7 }, svg);
      tx(svg, x + bw / 2, 90, 'N' + (i + 1), { 'text-anchor': 'middle', 'font-size': 9.5, fill: isTop ? C.on : (isOpt ? C.neg : C.fg3) });
    });
    const pillW = Math.min(W - 2 * pad, 310), pillX = (W - pillW) / 2;
    el('rect', { x: pillX, y: 100, width: pillW, height: 22, rx: 11, fill: C.onsoft, stroke: C.on, 'stroke-width': 1 }, svg);
    tx(svg, W / 2, 115, '↓ 고유 순위 + 가족·블록 몰기 (필요 시 꼬리 1* 개방)', { 'text-anchor': 'middle', fill: C.on, 'font-size': 10, 'font-weight': 600 });
    tx(svg, pad, 144, `${cur.name} · 학습 후 (앞쪽 집중 · 최종 CH2는 1*=0%로 통과)`, { class: 'hi', 'font-size': 10.8 });
    const rest = Array.from({ length: 12 }, (_, i) => i).filter(i => !cur.top.includes(i) && i !== cur.opt);
    [...cur.top, cur.opt, ...rest].forEach((nid, pos) => {
      const x = pad + pos * (bw + 4), isMand = pos < 4, isOpt = pos === 4;
      const h = isMand ? [48, 44, 40, 36][pos] : (isOpt ? 18 : 3), y = 204 - h;
      el('rect', { x, y, width: bw, height: h, rx: 3, fill: isMand ? C.on : (isOpt ? C.neg : C.bg3), stroke: isOpt ? C.fg : 'none', 'stroke-width': isOpt ? 1.2 : 0, 'stroke-dasharray': isOpt ? '3 2' : '' }, svg);
      tx(svg, x + bw / 2, 218, 'N' + (nid + 1), { 'text-anchor': 'middle', 'font-size': 9.5, fill: isMand ? C.on : (isOpt ? C.neg : C.fg3), 'font-weight': isMand || isOpt ? 600 : 400 });
    });
    tx(svg, W / 2, 238, '※ 1*는 추가 압축이 필요할 때 켤 수 있는 선택적 자유도이며, 최종 CH2는 1*=0% (90.81%)', { 'text-anchor': 'middle', fill: C.fg2, 'font-size': W < 470 ? 8.5 : 9.6 });
    leg.innerHTML = '<span><i style="background:var(--on)"></i>필수 뉴런 (1)</span><span><i style="background:var(--neg)"></i>선택적 버퍼 (1*)</span><span><i style="background:var(--off)"></i>완전 OFF (0)</span>';
    $('#ct-c3-kpi').innerHTML = `<div class="panel"><div class="k">${cur.name} 필수 명단</div><div class="v on" style="font-size:15px">${cur.top.map(i => 'N' + (i + 1)).join(' · ')}</div></div><div class="panel"><div class="k">최종 CH2 모델 검증 선택</div><div class="v" style="font-size:14.5px">1* = 0% (배치 무관 100% 동일 출력)</div></div>`;
  } else {
    const lx = 58, rx = 10, top = 34, cs = Math.min(36, (W - lx - rx) / 12), matH = 6 * cs, by = top + matH + 52, H = by + 76;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '우리 비격자 5단계 + A3V4 비중' }, host);
    const owner = Array.from({ length: 6 }, () => Array(12).fill(-1));
    C3_NG_TILES.forEach((t, ti) => t.rows.forEach(r => t.cols.forEach(c => { owner[r][c] = ti; })));
    const selT = S.c3t, activeTile = selT >= 0 ? C3_NG_TILES[selT] : null;
    tx(svg, lx - 8, top - 11, '뉴런', { 'text-anchor': 'end', 'font-size': 10.5 });
    for (let c = 0; c < 12; c++) { const hi = activeTile && activeTile.cols.includes(c); tx(svg, lx + c * cs + cs / 2, top - 11, 'N' + (c + 1), { 'text-anchor': 'middle', fill: hi ? activeTile.col : C.fg3, 'font-size': 10.5, 'font-weight': hi ? 700 : 400 }); }
    for (let r = 0; r < 6; r++) { const hi = activeTile && activeTile.rows.includes(r); tx(svg, lx - 8, top + r * cs + cs / 2 + 4, P1_ROWS[r], { 'text-anchor': 'end', fill: hi ? activeTile.col : C.fg2, 'font-size': 11, 'font-weight': hi ? 700 : 400 }); }
    for (let r = 0; r < 6; r++) for (let c = 0; c < 12; c++) {
      const v = P1_G[r][c], ti = owner[r][c], x = lx + c * cs + 2, y = top + r * cs + 2, s = cs - 4;
      const isAutoOff = v === 2 && ti === -1, isFocus = selT === -1 || ti === selT, dim = selT >= 0 && ti !== selT;
      let fill = C.off, stroke = 'none', dash = '', op = dim ? 0.22 : 1;
      if (ti >= 0) { stroke = C3_NG_TILES[ti].col; fill = v === 1 ? C.on : (v === 2 ? C.neg : C.pick); }
      else if (isAutoOff) { fill = C.bg; stroke = C.line2; dash = '3 2'; }
      const a = { x, y, width: s, height: s, rx: 3, fill, opacity: op };
      if (stroke !== 'none') { a.stroke = stroke; a['stroke-width'] = (ti >= 0 && isFocus) ? 2 : 1.1; if (dash) a['stroke-dasharray'] = dash; }
      const rc = el('rect', a, svg);
      if (ti >= 0) {
        rc.style.cursor = 'pointer'; const T = C3_NG_TILES[ti];
        const cx = lx + c * cs + cs / 2, cy = top + r * cs + cs / 2;
        const txt = v === 2 ? '1*' : (v === 0 ? '×0' : T.tag);
        tx(svg, cx, cy + 3.5, txt, { 'text-anchor': 'middle', fill: C.bg, 'font-size': Math.max(8.5, cs * 0.31), 'font-weight': 700, opacity: op, style: 'pointer-events:none' });
        rc.addEventListener('pointerenter', ev => showTip(ev, `비격자 틀 ${T.tag} (2×3): ${T.lbl} · ${T.stat}`));
        rc.addEventListener('pointermove', ev => showTip(ev, $('#tip').textContent));
        rc.addEventListener('pointerleave', hideTip);
        rc.addEventListener('click', () => { S.c3t = (S.c3t === ti ? -1 : ti); drawC3Tiling(); });
      } else if (isAutoOff) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3.5, 'OFF', { 'text-anchor': 'middle', fill: C.fg3, 'font-size': Math.max(8, cs * 0.27), 'font-weight': 600, opacity: op });
    }
    const capTxt = activeTile ? `틀 ${activeTile.tag}: ${activeTile.lbl} (${activeTile.stat})` : (W < 470 ? '행·열 자유 짝짓기 → 7틀→5틀, ×0 18→8칸' : '같은 6×12 표에서 T6이 틀 ②는 T2, 틀 ③은 T3과 자유 짝짓기! (7틀→5틀, ×0 18→8칸)');
    tx(svg, lx + 6 * cs, top + matH + 17, capTxt, { 'text-anchor': 'middle', fill: activeTile ? activeTile.col : C.on, 'font-size': W < 470 ? 8.8 : 10, 'font-weight': 600 });
    el('line', { x1: pad, x2: W - pad, y1: top + matH + 30, y2: top + matH + 30, stroke: C.line, 'stroke-width': 1 }, svg);

    const tot2048 = B.layer_tables['2048'].total;
    const barW = W - 2 * pad;
    tx(svg, pad, by - 4, `실제 CH2 + A3V4 (배치 2,048) 5단계 비중 — 64×32 단독 ${tot2048.share_64x32.toFixed(1)}% · 1·2순위 ${tot2048.share_top2.toFixed(1)}%`, { class: 'hi', 'font-size': W < 470 ? 9.2 : 10.4 });
    const segs = [
      { lbl: '64×32', pct: tot2048.share_64x32, col: C.on },
      { lbl: '32×32', pct: tot2048.share_32x32, col: '#3eb8cc' },
      { lbl: '32×16', pct: tot2048.share_32x16, col: C.neg },
      { lbl: '16×16', pct: tot2048.share_16x16, col: C.pick },
      { lbl: '8×16',  pct: tot2048.share_8x16,  col: C.fg3 },
    ];
    let cx = pad;
    segs.forEach(sg => {
      const sw = Math.max(4, barW * sg.pct / 100);
      el('rect', { x: cx, y: by + 6, width: Math.max(2, sw - 1.5), height: 22, rx: 3, fill: sg.col }, svg);
      if (sw > 52) tx(svg, cx + sw / 2, by + 20.5, `${sg.lbl} (${sg.pct.toFixed(1)}%)`, { 'text-anchor': 'middle', fill: C.bg, 'font-size': 9.5, 'font-weight': 700 });
      cx += sw;
    });
    tx(svg, pad, by + 46, `64×32 ${tot2048.share_64x32.toFixed(1)}% | 32×32 ${tot2048.share_32x32.toFixed(1)}% | 32×16 ${tot2048.share_32x16.toFixed(1)}% | 16×16 ${tot2048.share_16x16.toFixed(1)}% | 8×16 ${tot2048.share_8x16.toFixed(1)}%`, { class: 'lab', 'font-size': W < 470 ? 8.6 : 9.6 });
    tx(svg, pad, by + 64, `→ A3V4 4층 디스패치 ${tot2048.dispatch_4l_ms.toFixed(2)}ms (46초 → 16,000배 단축) · 채움 ${tot2048.fill_pct.toFixed(1)}% · 1*=0%`, { fill: C.on, 'font-size': W < 470 ? 9 : 10, 'font-weight': 600 });

    leg.innerHTML = '<span><i style="background:var(--on)"></i>필수 1 (①~⑤ 비격자 5틀)</span><span><i style="background:var(--neg)"></i>1* 공짜 ON</span><span><i style="background:var(--pick)"></i>×0 (8칸)</span><span><i style="background:var(--off);box-shadow:inset 0 0 0 1px var(--line-2)"></i>외딴 1* OFF</span>';
    $('#ct-c3-kpi').innerHTML =
      `<div class="panel"><div class="k">A3V4 오프라인-인덱스 AVX2 디스패치</div><div class="v on" style="font-size:14.5px">1.49 ~ 2.83 ms <span style="font-size:11.5px;color:var(--fg-3)">(초기화 ${B.init_ms.toFixed(1)}ms · 어텐션 은폐 0ms)</span></div></div>
       <div class="panel"><div class="k">실제 CH2 1·2순위 고속 틀 비중 (B=2,048)</div><div class="v" style="font-size:14.5px">${tot2048.share_top2.toFixed(1)}% <span style="font-size:11.5px;color:var(--fg-3)">(64×32 ${tot2048.share_64x32.toFixed(1)}% · 채움 ${tot2048.fill_pct.toFixed(1)}%)</span></div></div>`;
  }

  const c3Rows = [
    { m: -1, name: '순서대로 끼우기 (기준)',         tiles: '12틀',    ops: '72칸', opsSub: '(100%)',  z: '48칸', zSub: '(66.7%)', zCls: 'warn', keep: '18 / 18', keepCls: '' },
    { m: 0,  name: '① 기존: MatFormer (앞 4열)',     tiles: '고정 4열', ops: '24칸', opsSub: '(33.3%)', z: '17칸', zSub: '(70.8%)', zCls: 'warn', keep: '6 / 18 (12 누락)', keepCls: 'warn' },
    { m: 1,  name: '② 기존: BSMR (행 패널 고정)',    tiles: '7틀',     ops: '42칸', opsSub: '(58.3%)', z: '18칸', zSub: '(42.9%)', zCls: 'warn', keep: '18 / 18 (행 고정)', keepCls: 'mut' },
    { m: 3,  name: '④ 우리: 비격자 자유 조합 커버',   tiles: '5틀',     ops: '30칸', opsSub: '(41.7%)', z: '8칸',  zSub: '(26.7%)', zCls: 'ok',   keep: '18 / 18 (겹침 0)', keepCls: 'ok' },
  ];
  $('#ct-c3-tbl').innerHTML =
    '<tr><th>방식 (같은 6×12 표 · 2×3 틀)</th><th>네모 틀 수</th><th>연산할 칸</th><th>쓸데없는 0 (×0)</th><th>필수 1 보존</th></tr>' +
    c3Rows.map(r => {
      const cls = r.m === 3 ? ' class="sum"' : (r.m === S.c3mode ? ' style="background:rgba(240,180,92,0.08)"' : '');
      return `<tr${cls}><td>${r.m === 3 ? '<b>' + r.name + '</b>' : r.name}</td><td class="num${r.m === 3 ? ' ok' : ''}">${r.tiles}</td><td class="num${r.m === 3 ? ' ok' : ''}"><b>${r.ops}</b> <span class="mut">${r.opsSub}</span></td><td class="num ${r.zCls}"><b>${r.z}</b> <span class="mut">${r.zSub}</span></td><td class="num ${r.keepCls}">${r.keep}</td></tr>`;
    }).join('');
  drawC3Scale();
}

const C3_SCALE_META = {
  dense_64x32:             { name: '기준: 풀 밀집 (Dense 64×32)',                   cls: '' },
  matformer_matched:       { name: '정적 ① MatFormer (동일 활성률 예산)',            cls: '' },
  static1d_32x32:          { name: '정적 ② 고전 SpMM (전역 1D · 32×32)',             cls: '' },
  bsmr_16x16_bsa_strip512: { name: 'SDDMM: BSMR 원 논문 (16×16 · BSA)',              cls: '' },
  bsmr_16x16_strip512:     { name: 'SDDMM: BSMR 최강 OLO (16×16 · Strip512)',        cls: '' },
  bsmr_32x32_strip512:     { name: 'SDDMM: BSMR 대형 틀 (32×32 · Strip512)',         cls: '' },
  bsmr_multiscale_cost:    { name: 'SDDMM: BSMR 5단계 행 패널 상한 (Strip512)',      cls: '' },
  our_a3v4_strip512:       { name: '우리 최종 A3V4 5단계 (1.49~2.83ms · Strip512)',  cls: 'sum' },
  our_ms_strip512:         { name: '참고: 오프라인 정밀 ms_solve 5단계 (46초)',       cls: 'sub-hi' },
};

function drawC3Scale() {
  const sc = B.scale_cmp; if (!sc) return;
  const chips = $('#ct-c3-sb-chips'); chips.innerHTML = '';
  [['512', 'B = 512'], ['1024', 'B = 1,024'], ['2048', 'B = 2,048 (vLLM 기본)']].forEach(([val, lbl]) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', S.c3sb === val);
    b.onclick = () => { S.c3sb = val; drawC3Scale(); };
    chips.appendChild(b);
  });
  const d = sc.batches[S.c3sb];
  $('#ct-c3-sb-cap').textContent = `실제 CH2 4개 층 × ${nf(d.B)} × 4,096 = ${nf(d.full_cells)}칸 (활성 1 ${d.active_pct}%) · 독립 배치 ${sc.n_trials}회 평균 (1*=0% 동일 조건)`;
  const byId = Object.fromEntries(d.rows.map(r => [r.id, r]));
  const ours = byId.our_a3v4_strip512, b5 = byId.bsmr_multiscale_cost, b16 = byId.bsmr_16x16_strip512;
  const zRed5 = (100 * (1 - ours.zeros / b5.zeros)).toFixed(1), sp5 = (b5.cost_pct / ours.cost_pct).toFixed(2);
  const tRed16 = (100 * (1 - ours.tiles / b16.tiles)).toFixed(1), sp16 = (b16.cost_pct / ours.cost_pct).toFixed(2);
  $('#ct-c3-sb-kpi').innerHTML =
    `<div class="panel"><div class="k">BSMR 5단계 행 패널 상한 대비 (같은 CH2 표)</div><div class="v on" style="font-size:14.5px">×0 -${zRed5}% · 채움 ${b5.fill_pct.toFixed(1)}%→${ours.fill_pct.toFixed(1)}% · 비용 ${sp5}× 우위</div></div>
     <div class="panel"><div class="k">BSMR 16×16 (최강 OLO) 대비</div><div class="v on" style="font-size:14.5px">틀 수 -${tRed16}% · 실측 비용 ${sp16}× 우위</div></div>`;
  $('#ct-c3-sb-tbl').innerHTML =
    `<tr><th>방식 (실제 CH2 4층 × ${nf(d.B)} × 4,096 · ${sc.n_trials}회 평균)</th><th>틀 개수</th><th>연산 칸 (풀 대비)</th><th>불필요한 0 (×0)</th><th>필수 1 보존</th><th>실측 비용%</th></tr>` +
    d.rows.map(r => {
      const m = C3_SCALE_META[r.id], isOurs = r.id === 'our_a3v4_strip512' || r.id === 'our_ms_strip512', isMat = r.id === 'matformer_matched';
      return `<tr${m.cls ? ` class="${m.cls}"` : ''}>
        <td>${isOurs ? '<b>' + m.name + '</b>' : m.name}</td>
        <td class="num${isOurs ? ' ok' : ''}">${nf(Math.round(r.tiles))} <span class="mut">±${nf(Math.round(r.tiles_std))}</span></td>
        <td class="num${isOurs ? ' ok' : ''}"><b>${r.cells_pct.toFixed(2)}%</b> <span class="mut">(${nf(r.cells)})</span></td>
        <td class="num${isOurs ? ' ok' : (r.zero_pct >= 30 ? ' warn' : '')}"><b>${r.zero_pct.toFixed(1)}%</b> <span class="mut">(${nf(r.zeros)})</span></td>
        <td class="num">${isMat ? `<span class="warn">${r.mand_pct.toFixed(1)}%</span>` : '<span class="ok">100%</span>'}</td>
        <td class="num${isOurs ? ' ok' : (r.cost_pct >= 35 ? ' warn' : '')}"><b>${r.cost_pct.toFixed(1)}%</b></td>
      </tr>`;
    }).join('');
}

// ---- #c4 (Contribution 4): Single + Server E2E Silicon Proof + Prior-Work Matrix ----
function drawC4Proof() {
  const H1 = B.hw_bench['1024'], H4 = B.hw_bench['4096'], r2048 = H4.rows[2], r1_2048 = H1.rows[2];
  const host = $('#ct-c4-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), H = 268, pad = 10;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '단일 요청과 대형 서버 E2E 실측 파이프라인' }, host);
  const gap = 12, innerW = W - 2 * pad - 20, bw = (innerW - 2 * gap) / 3, hdrFs = W < 470 ? 9.6 : 11;
  const drawFlow = (y, steps, col) => {
    steps.forEach(([t1, t2], k) => {
      const bx = pad + 10 + k * (bw + gap);
      el('rect', { x: bx, y, width: bw, height: 44, rx: 6, fill: C.bg, stroke: k === 2 ? col : C.line2, 'stroke-width': k === 2 ? 1.5 : 1 }, svg);
      tx(svg, bx + bw / 2, y + 18, t1, { 'text-anchor': 'middle', fill: k === 2 ? col : C.fg, 'font-size': Math.min(10, bw / 14), 'font-weight': 600 });
      tx(svg, bx + bw / 2, y + 34, t2, { 'text-anchor': 'middle', class: 'lab', 'font-size': Math.min(9, bw / 16) });
      if (k < 2) tx(svg, bx + bw + gap / 2, y + 25, '→', { 'text-anchor': 'middle', fill: C.fg3, 'font-size': 10 });
    });
  };
  el('rect', { x: pad, y: 8, width: W - 2 * pad, height: 118, rx: 8, fill: 'rgba(157,140,255,0.08)', stroke: C.neg, 'stroke-width': 1.2 }, svg);
  tx(svg, pad + 12, 27, '① 단일 요청 (B = 1 · 로컬/개인 LLM) — 타일링 불필요 · 0 패딩 0%', { fill: C.neg, 'font-size': hdrFs, 'font-weight': 700 });
  drawFlow(40, [
    ['토큰 1개 → 중심점 내적', '활성 13.52% 선제 확정'],
    ['벡터화 bf16 GEMV', '활성 열만 DRAM 로드'],
    [`실측 ${H1.b1.speedup.toFixed(2)}× / ${H4.b1.speedup.toFixed(2)}×`, 'd=1,024 / 4,096 · 90.81%'],
  ], C.neg);
  tx(svg, pad + 12, 110, W < 470 ? `0마스크 0.99× · 대역폭 ${H4.b1.ours_gbs.toFixed(0)} GB/s (표본 한계 94%)` : `0 마스킹 0.99×  |  CH2 전용 GEMV: 대역폭 ${H4.b1.ours_gbs.toFixed(0)} GB/s (표본 이론 한계 6.67×의 94%)`, { class: 'lab', 'font-size': W < 470 ? 8.8 : 9.6 });

  el('rect', { x: pad, y: 136, width: W - 2 * pad, height: 124, rx: 8, fill: C.onsoft, stroke: C.on, 'stroke-width': 1.2 }, svg);
  tx(svg, pad + 12, 155, '② 대형 서버 (B = 512 ~ 2,048) — 어텐션 중첩 4.57× · 직렬 E2E 3.10×', { fill: C.on, 'font-size': hdrFs, 'font-weight': 700 });
  drawFlow(168, [
    ['M=32 + B1/B2 · A3V4', `1.49~2.83ms · 1*=0%`],
    ['5단계 v4 + autoTM', '채움 87.4% · DRAM 64.5%'],
    [`중첩 ${r2048.e2e_ovlp_x.toFixed(2)}× / 직렬 ${r2048.e2e_serial_x.toFixed(2)}×`, `d=4,096 B=2,048 · 90.81%`],
  ], C.on);
  tx(svg, pad + 12, 242, W < 470 ? `d=1,024 GPU ${r1_2048.gpu_x.toFixed(2)}× (직렬 ${r1_2048.e2e_serial_x.toFixed(2)}×) · 오차 ≤0.22%` : `d=1,024 실제 가중치: GPU ${r1_2048.gpu_x.toFixed(2)}× (직렬 ${r1_2048.e2e_serial_x.toFixed(2)}×)  |  0 마스킹 1.02~1.03×  |  fp32 오차 ≤ 0.22%`, { class: 'lab', 'font-size': W < 470 ? 8.8 : 9.4 });

  $('#ct-c4-kpi').innerHTML =
    `<div class="panel"><div class="k">단일 요청 (B=1 · RTX 4060 실측)</div><div class="v neg" style="font-size:15px">${H1.b1.speedup.toFixed(2)}× · ${H4.b1.speedup.toFixed(2)}× <span style="font-size:11.5px;color:var(--fg-3)">(90.81%, 99.77%)</span></div></div>
     <div class="panel"><div class="k">대형 서버 (B=2,048, d=4,096 실측)</div><div class="v on" style="font-size:15px">중첩 ${r2048.e2e_ovlp_x.toFixed(2)}× · 직렬 ${r2048.e2e_serial_x.toFixed(2)}× <span style="font-size:11.5px;color:var(--fg-3)">(1*=0%)</span></div></div>`;

  const cmpRows = [
    ['Deja Vu · ShadowLLM',            'warn', '× (MLP 예측)',   'ok',   '○',             'warn', '× (토큰별)',     'warn', '× (합집합 붕괴)',      'B=1 · 소배치 중심'],
    ['PowerInfer · TurboSparse',       'mut',  '△ (dReLU/예측)', 'mut',  '△',             'ok',   '○ (B=1)',        'warn', '× (로컬 전용)',        '단일 요청 CPU·GPU 오프로드'],
    ['CATS · TEAL',                    'ok',   '○ (임계값)',     'warn', '× (gate 계산)', 'ok',   '○',              'mut',  '△ (소·중배치)',        '학습 불필요 · up/down 스킵'],
    ['CMoE · AWC (2025)',              'mut',  '△ (고정 분할)',  'mut',  '○ / —',         'ok',   '○',              'mut',  '△ (고정 전문가)',      '공용 뉴런 자유 공유 불가'],
    ['Polar Sparsity · Park 등(2026)', 'mut',  '△ (경량 라우터)','warn', '× (SwiGLU 포기)','mut',  '△',              'warn', '× (어텐션 우회)',      '대형 배치 FFN 붕괴 한계 보고'],
    ['BSMR · Fused3S · TC-GNN',        'mut',  '— (커널 계층)',  'mut',  '—',             'ok',   '○ (고정 1)',     'mut',  '△ (행 패널 고정)',     '패널별 열 정렬 · 저밀도 우회'],
    ['PIT · TileSpMM',                 'mut',  '— (커널 계층)',  'mut',  '—',             'ok',   '○ (고정 1)',     'mut',  '△ (정적/마이크로)',    '비격자 Gather · 가변 크기 SpMM'],
    ['Roster (최종 CH2 + A3V4 + v4)',  'ok',   '○ (차등 중심점)','ok',   '○ (3행렬 전부)', 'ok',   '○ (1*=0% 동일)', 'ok',   '○ (중첩 4.57×·직렬 3.10×)', '단일 6.27× · 1.5ms 디스패치 포함 E2E'],
  ];
  $('#ct-cmp-tbl').innerHTML =
    '<tr><th>연구 / 기법</th><th>학습 라우터 없음</th><th>3행렬 선제 스킵</th><th>배치 무관 동일 출력</th><th>디스패치 포함 대형 배치 E2E</th><th>핵심 특징</th></tr>' +
    cmpRows.map((r, i) => `<tr${i === cmpRows.length - 1 ? ' class="sum"' : ''}>
      <td><b>${r[0]}</b></td><td class="${r[1]}">${r[2]}</td><td class="${r[3]}">${r[4]}</td>
      <td class="${r[5]}">${r[6]}</td><td class="${r[7]}">${r[8]}</td>
      <td class="${i === cmpRows.length - 1 ? 'ok' : 'mut'}">${r[9]}</td>
    </tr>`).join('');
}

function drawBatch() { drawProb(); drawJoint(); drawLayerStats(); drawCache(); drawP2AndTimeline(); drawFinal(); drawHwBench(); }
function drawContrib() { drawC1Struct(); drawC2Codesign(); drawC3Tiling(); drawC4Proof(); }

// ---- Unified 4-tab view router ----
function setView(v) {
  if (!['train', 'infer', 'batch', 'contrib'].includes(v)) v = 'train';
  S.view = v;
  ['train', 'infer', 'batch', 'contrib'].forEach(k => {
    $('#tab-' + k).setAttribute('aria-pressed', String(v === k));
    $('#view-' + k).hidden = (v !== k);
  });
  if (v === 'train') drawAllTrain();
  else if (v === 'infer') drawInfer();
  else if (v === 'batch') drawBatch();
  else drawContrib();
  try { history.replaceState(null, '', '#' + v); } catch (e) { location.hash = v; }
}

async function boot() {
  ['--bg', '--bg-2', '--bg-3', '--line', '--line-2', '--fg', '--fg-2', '--fg-3', '--on', '--on-soft', '--pick', '--pick-soft', '--neg', '--off', '--l1', '--l2', '--l3', '--l4']
    .forEach(n => C[n.slice(2).replace(/-/g, '')] = css(n));
  C.con = css('--c-on'); C.cpick = css('--c-pick'); C.cneg = css('--c-neg');
  try {
    D = window.__VIZ__ || await (await fetch('data_roster/viz.json')).json();
    B = window.__BATCH__ || await (await fetch('data_roster/batch.json')).json();
    const sp = window.__SPR__ || {}, files = { feat: D.sprites.feat, inf: D.sprites.inf };
    D.sprites.rep.forEach((s, l) => { files['rep' + l] = s; });
    for (const k in files) SPR[k] = { url: sp[files[k].file] || ('data_roster/' + files[k].file), cols: files[k].cols };
  } catch (err) {
    $('#loading').textContent = '데이터를 불러오지 못했다: ' + err.message;
    return;
  }
  $('#loading').hidden = true;

  // Populate hero live from real CH2 data
  $('#st-acc').textContent = pct(D.final.test);
  $('#st-acc2').textContent = `(베이스의 ${pct(D.final.pct_of_base)})`;
  $('#st-avg').textContent = pct(D.final.avg);
  const r2048 = B.hw_bench['4096'].rows[2];
  $('#st-srv').textContent = `${r2048.e2e_ovlp_x.toFixed(2)}배`;
  $('#st-srv2').textContent = `(직렬 ${r2048.e2e_serial_x.toFixed(2)}×)`;
  $('#st-b1').textContent = `${B.hw_bench['4096'].b1.speedup.toFixed(2)}배`;
  $('#st-b12').textContent = `(실제 ${B.hw_bench['1024'].b1.speedup.toFixed(2)}×)`;

  S.t2sel = D.layers[0].feat[0].j;
  heroGrids($('#hero-grids'), D.examples[0].rounds[9]);
  $('#hero-cap').textContent = `시험 사진 한 장(${D.classes[D.examples[0].label]})이 최종 CH2 모델(9바퀴)의 층 1 ~ 4에서 켜는 뉴런. 칸 하나 = 뉴런 하나(4,096개).`;

  kmInit();
  $('#km-reset').onclick = () => { if (KM.timer) kmPlay(); kmInit(); };
  $('#km-step').onclick = () => { if (KM.phase === 4) kmInit(); else kmStep(); };
  $('#km-play').onclick = kmPlay;
  $('#cv-tsne').addEventListener('click', tsneClick);
  $('#rd-6b').oninput = ev => { S.r6b = +ev.target.value; drawT6b(); };
  $('#rd-7').oninput = ev => { S.r7 = +ev.target.value; drawT7(); };
  $('#in-next').onclick = () => goStep(1);
  $('#in-prev').onclick = () => goStep(-1);
  $('#bt-prev').onclick = () => { S.step = Math.max(0, S.step - 1); drawJoint(); };
  $('#bt-next').onclick = () => { S.step = Math.min(J_STEPS.length - 1, S.step + 1); drawJoint(); };
  ['train', 'infer', 'batch', 'contrib'].forEach(k => { $('#tab-' + k).onclick = () => setView(k); });

  const h0 = (location.hash || '').replace('#', '');
  setView(['train', 'infer', 'batch', 'contrib'].includes(h0) ? h0 : 'train');

  let rt;
  new ResizeObserver(() => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      heroGrids($('#hero-grids'), D.examples[0].rounds[9]);
      if (S.view === 'train') drawAllTrain();
      else if (S.view === 'infer') drawInfer();
      else if (S.view === 'batch') drawBatch();
      else drawContrib();
    }, 150);
  }).observe(document.body);
}
boot();
})();
"""
