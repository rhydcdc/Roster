'use strict';
/* Roster v3 · 공통 도구: DOM 만들기, 스위치, 걸음 버튼, 막대·꺾은선, 도움말 풍선 */
const $ = (q, r = document) => r.querySelector(q);
const $$ = (q, r = document) => Array.from(r.querySelectorAll(q));
const NS = 'http://www.w3.org/2000/svg';
const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let LANG = 'ko';
const L = (ko, en) => (LANG === 'en' && en != null ? en : ko);

function mk(ns, tag, attrs, kids) {
  const e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === 'html') e.innerHTML = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'class') e.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(9)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const h = (tag, attrs, ...kids) => mk(null, tag, attrs, kids);
const s = (tag, attrs, ...kids) => mk(NS, tag, attrs, kids);
const CIRC = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '⑪', '⑫'];
const fmt = (v, d = 1) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/* 자리 바꿈을 부드럽게 (FLIP) */
function flip(els, mutate, ms = 480) {
  if (REDUCED) { mutate(); return; }
  const a = els.map(e => e.getBoundingClientRect());
  mutate();
  els.forEach((e, i) => {
    const b = e.getBoundingClientRect();
    const dx = a[i].left - b.left, dy = a[i].top - b.top;
    if ((dx || dy) && e.animate) e.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: 'none' }], { duration: ms, easing: 'cubic-bezier(.2,.7,.2,1)' });
  });
}

/* 스위치: opts = [{k, t}] */
function seg(opts, onChange, label) {
  let cur = -1;
  const btns = opts.map((o, i) => h('button', { type: 'button', class: 'seg-b', role: 'tab', html: o.t, onclick: () => el.set(i) }));
  const el = h('div', { class: 'seg', role: 'tablist', 'aria-label': label || '' }, btns);
  el.set = (i) => {
    if (i === cur) return;
    cur = i;
    btns.forEach((b, j) => { b.classList.toggle('on', j === i); b.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
    onChange(opts[i].k, i);
  };
  el.cur = () => cur;
  el.n = opts.length;
  return el;
}

/* 걸음 버튼: steps = [{t: 제목, d: 한 문장}] */
function stepper(steps, onStep) {
  let cur = -1;
  const cap = h('div', { class: 'cap', 'aria-live': 'polite' });
  const dots = steps.map((st, i) => h('button', { type: 'button', class: 'dot', 'aria-label': L('걸음 ', 'Step ') + (i + 1), text: String(i + 1), onclick: () => el.go(i) }));
  const prev = h('button', { type: 'button', class: 'nav-b', text: L('이전', 'Back'), onclick: () => el.go(cur - 1) });
  const next = h('button', { type: 'button', class: 'nav-b pri', text: L('다음', 'Next'), onclick: () => el.go(cur + 1) });
  const el = h('div', { class: 'stepper' }, h('div', { class: 'st-row' }, h('div', { class: 'dots' }, dots), h('div', { class: 'st-nav' }, prev, next)), cap);
  const capHtml = (k) => `<b>${k + 1} / ${steps.length} · ${steps[k].t}</b>${steps[k].d || ''}`;
  let measuredAt = -1;
  const measure = () => {   // 가장 긴 캡션의 높이를 재서 고정한다 (폭이 바뀌면 다시 잰다)
    if (!el.isConnected || !el.clientWidth || el.clientWidth === measuredAt) return;
    measuredAt = el.clientWidth;
    cap.style.minHeight = '';
    let mx = 0;
    for (let k = 0; k < steps.length; k++) { cap.innerHTML = capHtml(k); mx = Math.max(mx, cap.offsetHeight); }
    cap.style.minHeight = mx + 'px';
  };
  window.addEventListener('resize', () => { if (cur >= 0) { measure(); cap.innerHTML = capHtml(cur); } });
  el.go = (i) => {
    i = Math.max(0, Math.min(steps.length - 1, i));
    cur = i;
    measure();
    dots.forEach((d, j) => { d.classList.toggle('on', j === i); d.classList.toggle('done', j < i); });
    prev.disabled = i === 0;
    next.disabled = i === steps.length - 1;
    cap.innerHTML = capHtml(i);
    onStep(i, steps[i]);
  };
  el.cur = () => cur;
  el.n = steps.length;
  return el;
}

/* 도움말 풍선 */
let TIP;
function tip(el, html) {
  el.addEventListener('mouseenter', (e) => showTip(html, e.clientX, e.clientY));
  el.addEventListener('mousemove', (e) => showTip(html, e.clientX, e.clientY));
  el.addEventListener('mouseleave', hideTip);
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); showTip(html, r.left + r.width / 2, r.top); });
  el.addEventListener('blur', hideTip);
}
function showTip(html, x, y) {
  if (!TIP) { TIP = h('div', { id: 'tip', role: 'tooltip' }); document.body.append(TIP); }
  TIP.innerHTML = typeof html === 'function' ? html() : html;
  TIP.classList.add('on');
  const w = TIP.offsetWidth, hh = TIP.offsetHeight;
  TIP.style.left = Math.max(6, Math.min(window.innerWidth - w - 6, x + 12)) + 'px';
  TIP.style.top = Math.max(6, y - hh - 10) + 'px';
}
function hideTip() { if (TIP) TIP.classList.remove('on'); }

/* 가로 막대 목록. rows = [{l, v, k, t, s}] (k: ours|prior|on|waste|base, t: 끝에 적는 값, s: 덧붙임) */
function barList(rows, opt = {}) {
  const max = opt.max || Math.max(...rows.map(r => r.v)) * 1.02;
  const el = h('div', { class: 'bl', role: 'img', 'aria-label': opt.label || '' });
  for (const r of rows) {
    const bar = h('div', { class: 'br ' + (r.k || ''), style: { width: '0%' } });
    const row = h('div', { class: 'tr', tabindex: '0' }, bar, h('span', { class: 'vl', html: r.t + (r.s ? `<small>(${r.s})</small>` : '') }));   // 덧붙임은 괄호로 감싸 값과 구분한다
    tip(row, `<b>${r.l}</b><br>${r.t}${r.s ? ' (' + r.s + ')' : ''}`);
    el.append(h('div', { class: 'lb' + (r.k === 'ours' ? ' me' : ''), html: r.l }), row);
    const wpc = Math.max(0.4, (r.v / max) * (opt.room || 78));
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = wpc + '%'; }));
  }
  return el;
}

/* 꺾은선. cfg = {x: [..], xl: [..], series: [{n, k, v: [..]}], unit, min, max} — 세로축 하나 */
function lineChart(cfg) {
  const W = 560, H = 250, pl = 44, pr = 104, pt = 14, pb = 34;
  const n = cfg.x.length, min = cfg.min, max = cfg.max;
  const X = (i) => pl + (i / (n - 1)) * (W - pl - pr);
  const Y = (v) => pt + (1 - (v - min) / (max - min)) * (H - pt - pb);
  const svg = s('svg', { class: 'lc', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.label || '', style: { width: '100%', maxWidth: W + 'px', display: 'block' } });
  for (let t = min; t <= max + 1e-9; t += cfg.tick) {
    svg.append(s('line', { class: 'gl', x1: pl, x2: W - pr, y1: Y(t), y2: Y(t) }), s('text', { x: pl - 8, y: Y(t) + 4, 'text-anchor': 'end', text: t + cfg.unit }));
  }
  svg.append(s('line', { class: 'ax', x1: pl, x2: W - pr, y1: Y(min), y2: Y(min) }));
  cfg.x.forEach((xv, i) => svg.append(s('text', { x: X(i), y: H - 12, 'text-anchor': 'middle', text: cfg.xl[i] })));
  const col = { ours: 'var(--f-ours)', on: 'var(--f-on)', prior: 'var(--f-prior)', waste: 'var(--f-waste)', base: 'var(--f-base)' };
  for (const se of cfg.series) {
    const c = col[se.k];
    svg.append(s('path', { class: 'ln', stroke: c, d: se.v.map((v, i) => (i ? 'L' : 'M') + X(i) + ' ' + Y(v)).join(' ') }));
    se.v.forEach((v, i) => {
      const hit = s('circle', { cx: X(i), cy: Y(v), r: 13, fill: 'transparent', tabindex: '0' });
      tip(hit, `<b>${se.n}</b><br>${cfg.xl[i]} · ${v}${cfg.unit}`);
      svg.append(s('circle', { class: 'pt', cx: X(i), cy: Y(v), r: 5, fill: c }), hit);
    });
    const last = se.v[n - 1];
    svg.append(s('text', { class: 'el', x: X(n - 1) + 10, y: Y(last) + 4, text: last + cfg.unit + ' ' + se.n }));
  }
  const lg = h('div', { class: 'legend' }, cfg.series.map(se => h('span', null, h('i', { class: 'i-' + se.k }), se.n)));
  return h('div', null, svg, lg);
}

/* 위젯 등록과 붙이기 */
const W = {};
function mountAll(root) {
  for (const el of $$('[data-w]', root)) {
    if (el.dataset.done) continue;
    el.dataset.done = '1';
    const fn = W[el.dataset.w];
    if (!fn) { el.textContent = 'widget? ' + el.dataset.w; continue; }
    try { fn(el, el.dataset); } catch (err) { console.error('widget ' + el.dataset.w, err); el.textContent = 'widget error: ' + el.dataset.w; }
  }
}
function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  document.dispatchEvent(new Event('roster-theme'));
  try { localStorage.setItem('roster-theme', t); } catch (e) { /* 저장 못 해도 동작 */ }
  $$('[data-theme-btn]').forEach(b => { b.textContent = t === 'light' ? L('어둡게', 'Dark') : L('밝게', 'Light'); });
}
function initTheme() {
  let t = 'dark';
  try { t = localStorage.getItem('roster-theme') || 'dark'; } catch (e) { /* 기본값 */ }
  const q = new URLSearchParams(location.search).get('theme');
  if (q) t = q;
  setTheme(t);
  $$('[data-theme-btn]').forEach(b => b.addEventListener('click', () => setTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light')));
}
