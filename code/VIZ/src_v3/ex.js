'use strict';
/* Roster v3 · 가이드의 소규모 예시 (GUIDE/01 ~ 05의 숫자 그대로)와 그것을 그리는 부품 네 개 */

/* ── 6토큰 × 8뉴런 표 (GUIDE/03 1단계, 1이 23칸) ── */
const G68 = {
  rows: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'], sub: ['A', 'A', 'A', 'B', 'C', 'D'],
  cols: ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'],
  M: [[1, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1]],
};
const TS = {
  union: [[0, 1], [2, 3], [4, 5]].flatMap(r => [[0, 1], [2, 3], [4, 5], [6, 7]].map(c => ({ r, c }))),
  bsmr: [{ r: [0, 1], c: [0, 1] }, { r: [0, 1], c: [2, 4] }, { r: [2, 3], c: [2, 4] }, { r: [2, 3], c: [0, 1] }, { r: [2, 3], c: [3, 6] }, { r: [4, 5], c: [7, 0] }, { r: [4, 5], c: [1, 3] }, { r: [4, 5], c: [5, 6] }],
  ours: [{ r: [0, 1], c: [0, 1] }, { r: [2, 5], c: [0, 1] }, { r: [0, 1], c: [2, 4] }, { r: [2, 3], c: [2, 4] }, { r: [3, 4], c: [3, 6] }, { r: [4, 5], c: [5, 7] }],
  big: [{ r: [0, 1, 2, 5], c: [0, 1] }, { r: [0, 1, 2, 3], c: [2, 4] }, { r: [2, 3, 4, 5], c: [3, 6] }, { r: [2, 3, 4, 5], c: [5, 7] }],
  multi: [{ r: [0, 1, 2, 5], c: [0, 1] }, { r: [0, 1, 2, 3], c: [2, 4] }, { r: [3, 4], c: [3, 6] }, { r: [4, 5], c: [5, 7] }],
  zone1: [{ r: [0, 1, 2, 5], c: [0, 1] }, { r: [0, 1, 2, 3], c: [2, 3] }, { r: [4, null], c: [2, 3] }],
  zone2: [{ r: [0, 1, 2, 3], c: [4, 6] }, { r: [4, 5], c: [5, 7] }, { r: [4, null], c: [4, 6] }],
};

/* 표 하나를 그리고, 틀 묶음을 그 위에 칠한다 */
function GridW(cfg) {
  const R = cfg.rows.length, C = cfg.cols.length, M = cfg.M;
  const tpl = ['max-content'], colIdx = [];
  let ci = 2, wallCol = 0;
  for (let j = 0; j < C; j++) {
    if (cfg.wall === j) { tpl.push('16px'); wallCol = ci++; }
    else if (cfg.blocks && j > 0 && j % 2 === 0) { tpl.push('6px'); ci++; }
    colIdx[j] = ci++;
    tpl.push('var(--cw)');
  }
  const g = h('div', { class: 'g', style: { gridTemplateColumns: tpl.join(' ') } });
  let row = 1;
  if (cfg.zones) {
    g.append(h('div', { class: 'zl', style: { gridColumn: `${colIdx[0]} / ${colIdx[cfg.wall - 1] + 1}`, gridRow: row }, text: cfg.zones[0] }));
    g.append(h('div', { class: 'zl', style: { gridColumn: `${colIdx[cfg.wall]} / ${colIdx[C - 1] + 1}`, gridRow: row }, text: cfg.zones[1] }));
    row++;
  } else if (cfg.blocks) {
    for (let b = 0; b < C / 2; b++) g.append(h('div', { class: 'bh', style: { gridColumn: `${colIdx[2 * b]} / ${colIdx[2 * b + 1] + 1}`, gridRow: row }, text: L('블록 ', 'Block ') + (b + 1) }));
    row++;
  }
  cfg.cols.forEach((c, j) => g.append(h('div', { class: 'hd', style: { gridColumn: colIdx[j], gridRow: row }, text: c })));
  row++;
  const first = row, cell = [];
  for (let i = 0; i < R; i++, row++) {
    g.append(h('div', { class: 'rl', style: { gridColumn: 1, gridRow: row }, html: cfg.rows[i] + (cfg.sub ? `<small>${cfg.sub[i]}</small>` : '') }));
    cell[i] = cfg.cols.map((c, j) => { const e = h('div', { class: 'c', style: { gridColumn: colIdx[j], gridRow: row } }); g.append(e); return e; });
  }
  if (wallCol) g.append(h('div', { class: 'wl', style: { gridColumn: wallCol, gridRow: `${first} / ${row}` } }));
  let total = 0;
  M.forEach(r => r.forEach(v => { total += v; }));

  // 틀의 테두리: 틀 = 행 묶음 × 열 묶음이므로, 이어진 행 구간 × 이어진 열 구간마다 네모 하나를 친다 (벽은 넘지 않는다)
  const frames = [];
  const runs = (a, wall) => {
    const v = a.filter(x => x != null).slice().sort((x, y) => x - y), out = [];
    v.forEach(x => { const l = out[out.length - 1]; if (l && x === l[1] + 1 && x !== wall) l[1] = x; else out.push([x, x]); });
    return out;
  };
  const frame = (rs, cs, cls) => {
    for (const [r0, r1] of runs(rs)) for (const [c0, c1] of runs(cs, cfg.wall)) {
      const e = h('div', { class: 'tf ' + cls, style: { gridColumn: `${colIdx[c0]} / ${colIdx[c1] + 1}`, gridRow: `${first + r0} / ${first + r1 + 1}` } });
      g.append(e);
      frames.push(e);
    }
  };
  function show(set, upto, focus) {
    const tiles = set && set.tiles ? set.tiles.slice(0, upto == null ? set.tiles.length : upto) : [];
    frames.splice(0).forEach(e => e.remove());
    tiles.forEach((t, k) => frame(t.r, t.c, focus == null ? '' : (k === focus ? 'cur' : 'dim')));
    if (set && set.fixedCols) frame(M.map((r, i) => i), set.fixedCols, '');
    const own = M.map(r => r.map(() => -1));
    tiles.forEach((t, k) => t.r.forEach(i => { if (i == null) return; t.c.forEach(j => { if (own[i][j] < 0) own[i][j] = k; }); }));
    const fixed = set && set.fixedCols;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const one = M[i][j], k = own[i][j];
      let cls = 'c', txt = one ? '1' : '0';
      const mk2 = set && set.mark && set.mark[i + ',' + j];
      if (mk2) { cls += ' ' + mk2[0]; txt = mk2[1]; if (k >= 0 && focus === k) cls += ' cur'; }
      else if (fixed) {
        if (fixed.includes(j)) { cls += one ? ' cov' : ' z'; if (!one) txt = '×0'; }
        else if (one) { cls += ' miss'; txt = L('누락', 'lost'); }
      } else if (k >= 0) {
        cls += one ? ' cov' : ' z';
        txt = one ? (set.plain ? '1' : CIRC[set.num ? set.num[k] : k]) : '×0';
        if (focus != null) cls += k === focus ? ' cur' : ' dim';
      } else if (one) cls += ' one';
      cell[i][j].className = cls;
      cell[i][j].textContent = txt;
    }
  }
  function stats(set, upto) {
    if (set.fixedCols) {
      let kept = 0;
      M.forEach(r => set.fixedCols.forEach(j => { kept += r[j]; }));
      const cells = R * set.fixedCols.length;
      return { tiles: null, cells, zeros: cells - kept, kept, total };
    }
    const tiles = set.tiles.slice(0, upto == null ? set.tiles.length : upto), seen = new Set();
    let cells = 0;
    tiles.forEach(t => { cells += t.r.length * t.c.length; t.r.forEach(i => { if (i == null) return; t.c.forEach(j => { if (M[i][j]) seen.add(i * 100 + j); }); }); });
    return { tiles: tiles.length, cells, zeros: cells - seen.size, kept: seen.size, total };
  }
  function tileZeros(t) {
    let z = 0;
    t.r.forEach(i => t.c.forEach(j => { if (i == null || !M[i][j]) z++; }));
    return z;
  }
  const names = (t) => ({
    r: '{' + t.r.filter(i => i != null).map(i => cfg.rows[i]).join(', ') + '}',
    c: '(' + t.c.map(j => cfg.cols[j]).join(', ') + ')',
  });
  function mini(t) {
    const m = h('div', { class: 'mt', style: { gridTemplateColumns: `repeat(${t.c.length}, 15px)` } });
    t.r.forEach(i => t.c.forEach(j => m.append(h('i', { class: i == null ? 'p' : (M[i][j] ? '' : 'z') }))));
    return m;
  }
  show(null);
  return { el: h('div', { class: 'gx' }, g), show, stats, tileZeros, names, mini, total, cell, size: R * C };
}

/* 표 + 틀 목록 + 성적 + (스위치 / 걸음) 을 한 번에 조립한다.
   sets = [{k, t, tiles, notes, intro, steps, fixedCols, score:false, done}] */
function tileUI(mount, gcfg, sets, opt = {}) {
  const G = GridW(gcfg);
  const list = h('div', { class: 'tl' }), sb = h('div', { class: 'sb' }), stepBox = h('div'), capBox = h('div', { class: 'cap', style: { display: 'none' } });
  let cur = sets[0], upto = null;
  const sv = (label, val, cls, sub) => h('div', { class: 'sv ' + (cls || '') }, h('span', { text: label }), h('b', { text: val }), sub ? h('small', { text: sub }) : null);
  function paint(focus) {
    G.show(cur, upto, focus);
    list.replaceChildren();
    const tiles = cur.tiles ? cur.tiles.slice(0, upto == null ? cur.tiles.length : upto) : [];
    tiles.forEach((t, k) => {
      const nm = G.names(t), z = G.tileZeros(t), lab = cur.labels ? cur.labels[k] : L('틀 ', 'Tile ') + CIRC[k + (cur.offset || 0)];
      const tail = t.r.includes(null);
      const it = h('div', { class: 'ti' + (focus === k ? ' cur' : ''), tabindex: '0' }, G.mini(t),
        h('div', null, h('b', { text: lab }), ' ' + nm.r + ' × ' + nm.c,
          h('span', { class: 'tz' + (z && !tail ? ' bad' : ''), text: tail ? L('자투리 마감', 'leftover') : '×0 ' + z + L('칸', '') }),
          cur.notes && cur.notes[k] ? h('span', { class: 'tn', html: cur.notes[k] }) : null));
      const on = () => G.show(cur, upto, k), off = () => G.show(cur, upto, opt.keepFocus ? focus : null);
      it.addEventListener('mouseenter', on); it.addEventListener('mouseleave', off);
      it.addEventListener('focus', on); it.addEventListener('blur', off);
      list.append(it);
    });
    if (cur.fixedCols) list.append(h('div', { class: 'ti' }, h('div'), h('div', { html: cur.fixedNote || '' })));
    sb.replaceChildren();
    if (cur.score !== false) {
      const st = G.stats(cur, upto);
      sb.append(
        sv(L('네모 틀', 'Tiles'), st.tiles == null ? L('고정 4열', 'fixed 4 cols') : st.tiles + L('틀', '')),
        sv(L('계산하는 칸', 'Cells computed'), st.cells + L('칸', ''), '', '/ ' + G.size),
        sv(L('헛계산 ×0', 'Wasted ×0'), st.zeros + L('칸', ''), st.zeros > 1 ? 'bad' : 'good'),
        sv(L('채움률', 'Fill rate'), st.cells ? fmt(100 * st.kept / st.cells) + '%' : '—'),
        sv(L('필수 1 보존', 'Needed 1s kept'), st.kept + ' / ' + st.total, st.kept < st.total && (upto == null || cur.fixedCols) ? 'bad' : ''));
    } else if (cur.done && (upto == null || upto >= cur.tiles.length)) sb.append(h('div', { class: 'sv', style: { gridColumn: '1 / -1' }, html: cur.done }));
  }
  function pick(k) {
    cur = sets.find(x => x.k === k);
    stepBox.replaceChildren();
    capBox.style.display = 'none';
    if (cur.steps) {
      const steps = cur.plan || [{ t: cur.introT || L('시작', 'Start'), d: cur.intro || '' }].concat(cur.tiles.map((t, i) => ({ t: (cur.labels ? cur.labels[i] : L('틀 ', 'Tile ') + CIRC[i + (cur.offset || 0)]), d: cur.notes ? cur.notes[i] : '' })));
      if (cur.outro && !cur.plan) steps.push({ t: cur.outroT || L('결과', 'Result'), d: cur.outro, end: true });
      const sp = stepper(steps, (i, st) => {
        upto = st.upto != null ? st.upto : (st.end ? cur.tiles.length : i);
        paint(cur.plan || i === 0 || st.end ? null : i - 1);
        if (opt.onStep) opt.onStep(i, st);
      });
      stepBox.append(sp);
      sp.go(opt.startFull ? steps.length - 1 : 0);
      api.sp = sp;
    } else {
      upto = null; api.sp = null;
      if (cur.intro) { capBox.style.display = ''; capBox.innerHTML = cur.intro; }
      paint(null);
    }
  }
  const api = { G, pick, sets, sp: null };
  const head = h('div', { class: 'w-h' });
  if (sets.length > 1) { const sg = seg(sets.map(x => ({ k: x.k, t: x.t })), pick, opt.label); head.append(sg); api.seg = sg; }
  const legend = h('div', { class: 'legend' },
    h('span', null, h('i', { class: 'i-one' }), L('필요한 1 (아직 안 덮음)', 'needed 1 (not yet covered)')),
    h('span', null, h('i', { class: 'i-cov' }), L('틀에 담아 계산한 1', '1 computed in a tile')),
    h('span', null, h('i', { class: 'i-z' }), L('틀에 끼어 헛계산한 0 (×0)', '0 forced into a tile (×0)')),
    h('span', null, h('i', { class: 'i-off' }), L('계산 안 하는 0', '0 not computed')),
    h('span', null, h('i', { class: 'i-frame' }), L('테두리 하나가 한 틀 (떨어져 있으면 같은 번호끼리 한 틀)', 'one outline = one tile (separate pieces with the same number are one tile)')));
  capBox.style.margin = '0 0 14px';
  mount.append(head, stepBox, capBox, h('div', { class: 'gw two' }, G.el, list), sb, legend);
  if (api.seg) api.seg.set(opt.init || 0); else pick(sets[0].k);
  return api;
}

/* ── 출석부 표: 4 중심점 × 8 뉴런 (GUIDE/01 2 ~ 4단계) ── */
const R48 = {
  cen: ['A', 'B', 'C', 'D'], tok: ['T1, T2', 'T3, T4', 'T5, T6', 'T7, T8'], fam: [1, 1, 2, 2],
  v0: [[40, 10, 30, 2, 20, 1, 15, 4], [35, 32, 4, 1, 28, 2, 12, 0], [2, 1, 6, 40, 3, 36, 2, 25], [1, 2, 3, 38, 2, 30, 1, 28]],
  on0: [[1, 3, 5, 7, 2, 8], [1, 2, 5, 7, 3], [4, 6, 8, 3], [4, 6, 8]],
  v1: [[44, 30, 3, null, 26, null, 16, 0.4], [37, 34, 0.5, null, 29, null, 13, null], [null, null, 0.6, 42, null, 38, null, 27], [null, null, null, 38, null, 30, null, 28]],
  on1: [[1, 2, 5, 7], [1, 2, 5, 7], [4, 6, 8], [4, 6, 8]],
  onI: [[1, 5, 7, 3, 8], [1, 2, 5, 7, 3], [3, 8, 4, 6], [8, 4, 6]],
  ord0: [1, 2, 3, 4, 5, 6, 7, 8], ord1: [1, 2, 5, 7, 3, 8, 4, 6],
  mark: { '0,2': 'up', '0,3': 'down', '0,8': 'down', '1,3': 'down', '2,3': 'down' },
};
function RosterW(onRow, opt = {}) {   // opt.compact: 줄 이름을 글자 하나로 줄인다 (쇼케이스의 좁은 무대용)
  const D = R48, tpl = ['max-content', 'max-content'], slotCol = [], rowOf = [3, 4, 6, 7];   // 1열 = 가족, 2열 = 중심점 이름, 5행 = 가족 사이 간격
  let ci = 3;
  for (let sl = 0; sl < 8; sl++) { if (sl && sl % 2 === 0) { tpl.push('var(--bgap, 0px)'); ci++; } slotCol[sl] = ci++; tpl.push('max-content'); }
  tpl.push('max-content');
  const kCol = ci, g = h('div', { class: 'rt', style: { gridTemplateColumns: tpl.join(' ') } });
  const bh = [0, 1, 2, 3].map(b => h('div', { class: 'bh', style: { gridColumn: `${slotCol[2 * b]} / ${slotCol[2 * b + 1] + 1}`, gridRow: 1 } }));
  const nh = {}, dc = D.cen.map(() => ({})), rl = [], kc = [];
  for (let n = 1; n <= 8; n++) { nh[n] = h('div', { class: 'hd', style: { gridRow: 2 }, text: 'N' + n }); g.append(nh[n]); }
  g.append(...bh);
  const fam = [0, 1].map(f => h('div', { class: 'fam', style: { gridColumn: 1, gridRow: `${rowOf[2 * f]} / ${rowOf[2 * f + 1] + 1}` }, text: L('가족 ', 'Family ') + (f + 1) }));
  const fgap = h('div', { style: { gridColumn: '1 / -1', gridRow: 5, height: '0px' } });
  g.append(...fam, fgap);
  D.cen.forEach((c, r) => {
    rl[r] = h('div', { class: 'rl rt-row', style: { gridColumn: 2, gridRow: rowOf[r] }, html: opt.compact ? `<b>${c}</b>` : L('중심점 ', '') + c + `<small>${D.tok[r]}</small>` });
    kc[r] = h('div', { class: 'kc', style: { gridColumn: kCol, gridRow: rowOf[r] } });
    g.append(rl[r], kc[r]);
    for (let n = 1; n <= 8; n++) { dc[r][n] = h('div', { class: 'rc rt-row', style: { gridRow: rowOf[r] } }); g.append(dc[r][n]); }
    if (onRow) [rl[r], kc[r]].concat(Object.values(dc[r])).forEach(e => e.addEventListener('click', () => onRow(r)));
  });
  const movers = Object.values(nh).concat(dc.flatMap(o => Object.values(o)));
  let sel = -1;
  function set(st) {
    const ord = st.ord === 1 ? D.ord1 : D.ord0;
    g.style.setProperty('--bgap', st.blocks ? '8px' : '0px');
    flip(movers, () => {
      fam.forEach(e => { e.style.display = st.fam ? '' : 'none'; });
      fgap.style.height = st.fam ? '7px' : '0px';
      for (let n = 1; n <= 8; n++) {
        const col = slotCol[ord.indexOf(n)];
        nh[n].style.gridColumn = col;
        D.cen.forEach((c, r) => { dc[r][n].style.gridColumn = col; });
      }
    });
    bh.forEach((e, b) => {
      e.style.visibility = st.blocks ? 'visible' : 'hidden';
      e.textContent = L('블록 ', 'Block ') + (b + 1);
      e.classList.toggle('hot', st.hot === b);
    });
    const v = st.view;
    D.cen.forEach((c, r) => {
      const onSet = v === 'onoff1' ? D.on1[r] : v === 'indiv' ? D.onI[r] : D.on0[r];
      for (let n = 1; n <= 8; n++) {
        const e = dc[r][n], on = onSet.includes(n);
        let cls = 'rc rt-row', txt;
        if (v === 'num0') { txt = D.v0[r][n - 1]; if (on) cls += ' on'; }
        else if (v === 'num0x') { txt = on ? D.v0[r][n - 1] : '—'; if (on) cls += ' on'; }
        else if (v === 'num1') { const x = D.v1[r][n - 1]; txt = x == null ? '—' : x; if (x != null) cls += ' on'; const mk2 = D.mark[r + ',' + n]; if (mk2) cls += ' ' + mk2; }
        else { txt = on ? '1' : '0'; if (on) cls += ' solid'; }
        if (st.hi && st.hi.includes(n) && (st.hiRows ? st.hiRows.includes(r) : true)) cls += ' hi';
        e.className = cls; e.textContent = txt;
      }
      const k0 = D.on0[r].length, k1 = D.on1[r].length;
      kc[r].innerHTML = v === 'onoff1' ? `K = ${k1}` + (k1 !== k0 && !opt.compact ? `<small>${k0}→${k1}</small>` : '') : v === 'indiv' ? `K = ${D.onI[r].length}` : `K = ${k0}`;
      rl[r].classList.toggle('sel', r === sel); kc[r].classList.toggle('sel', r === sel);
    });
  }
  const api = { el: h('div', { class: 'gx' }, g), set, select: (r) => { sel = r; rl.forEach((e, i) => { e.classList.toggle('sel', i === r); kc[i].classList.toggle('sel', i === r); }); } };
  return api;
}

/* ── 좌표 평면: 토큰 8개와 중심점 4개 (GUIDE/01 1단계, GUIDE/02 1단계) ── */
const PL = { tok: KM_TOK, cen: KM_CEN, name: 'ABCD', cloud: kmCloud(KM_SEED.cloud, KM_SEED.per, KM_SEED.rad) };
function voronoi(P, box) {
  return P.map((p, i) => {
    let poly = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]];
    P.forEach((q, j) => {
      if (i === j) return;
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2, nx = q[0] - p[0], ny = q[1] - p[1];
      const f = (v) => (v[0] - mx) * nx + (v[1] - my) * ny, out = [];
      for (let a = 0; a < poly.length; a++) {
        const A = poly[a], B = poly[(a + 1) % poly.length], fa = f(A), fb = f(B);
        if (fa <= 0) out.push(A);
        if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) { const t = fa / (fa - fb); out.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); }
      }
      poly = out;
    });
    return poly;
  });
}
function PlaneW(mode, onChange) {
  const VW = 600, VH = 380, X = (x) => 40 + (x + 3) / 6 * 540, Y = (y) => 350 - (y + 2) / 4.2 * 320;
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const near = (p, C) => { let b = 0; C.forEach((c, i) => { if (d2(p, c) < d2(p, C[b])) b = i; }); return b; };
  const svg = s('svg', { class: 'pl', viewBox: `0 0 ${VW} ${VH}`, role: 'img', 'aria-label': L('토큰과 중심점의 2차원 좌표 평면', '2-D plane of tokens and centroids') });
  const gR = s('g'), gL = s('g'), gT = s('g'), gC = s('g'), gN = s('g');
  svg.append(gR, s('line', { class: 'ax', x1: X(-3), x2: X(3), y1: Y(0), y2: Y(0) }), s('line', { class: 'ax', x1: X(0), x2: X(0), y1: Y(-2), y2: Y(2.2) }));
  [-2, -1, 1, 2].forEach(v => svg.append(s('text', { class: 'tk', x: X(v), y: Y(0) + 14, 'text-anchor': 'middle', text: v })));
  [-1, 1, 2].forEach(v => svg.append(s('text', { class: 'tk', x: X(0) - 7, y: Y(v) + 4, 'text-anchor': 'end', text: v })));
  svg.append(gL, gT, gC, gN);
  const regs = voronoi(PL.cen, [-3, -2, 3, 2.2]).map((poly, i) => {
    const e = s('polygon', { class: 'rg', points: poly.map(p => X(p[0]) + ',' + Y(p[1])).join(' ') });
    const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length, cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
    const wm = s('text', { class: 'wm', x: X(cx), y: Y(cy) + 12, text: PL.name[i] });
    gR.append(e, wm);
    return { e, wm };
  });
  PL.cloud.forEach(p => gT.append(s('circle', { class: 'cld', cx: X(p[0]), cy: Y(p[1]), r: 2.4 })));   // 지어낸 토큰 구름
  PL.tok.forEach((p, i) => {
    gT.append(s('circle', { class: 'tok', cx: X(p[0]), cy: Y(p[1]), r: 4.5 }), s('text', { class: 'tl2', x: X(p[0]) + 14, y: Y(p[1]) + (i % 2 ? 17 : -9), text: 'T' + (i + 1) }));
  });
  const cens = PL.cen.map((c, i) => {
    const e = s('g', { class: 'cen' }, s('circle', { r: 11 }), s('text', { y: 4.5, text: PL.name[i] }));
    gC.append(e);
    return e;
  });
  const place = (i, p) => { cens[i].style.transform = `translate(${X(p[0])}px, ${Y(p[1])}px)`; };
  const vis = (els, on) => els.forEach(e => { e.style.display = on ? '' : 'none'; });
  const lg = (cls, t) => h('span', null, h('i', { class: 'rd ' + cls }), t);
  const msg = h('span'), again = h('button', { type: 'button', class: 'nav-b', text: L('되풀이 다시 보기', 'Replay the rounds'), style: { display: 'none' } });
  const api = {
    el: svg, status: h('div', { class: 'km-st', 'aria-live': 'polite' }, msg, again),
    legend: h('div', { class: 'legend' },
      lg('rd-tok', mode === 'kmeans' ? L('토큰 (작은 점. 이름표가 붙은 T1 ~ T8은 그 가운데 여덟 개)', 'tokens (small dots; the labelled T1–T8 are eight of them)') : L('학습 때의 토큰 (작은 점)', 'training tokens (small dots)')),
      lg('rd-cen', mode === 'kmeans' ? L('대표점 A ~ D (크고 밝은 점)', 'representatives A–D (large, bright dots)') : L('고정된 중심점 A ~ D (크고 밝은 점)', 'fixed centroids A–D (large, bright dots)')),
      mode === 'kmeans' ? null : lg('rd-win', L('새 토큰과 당첨된 중심점', 'the new token and its winning centroid'))),
  };

  if (mode === 'kmeans') {
    // 구름 + 이름표 토큰 위에서 k-means를 실제로 한 판 돌려 둔다. 걸음 1 ~ 4는 첫 바퀴, 걸음 5는 나머지 바퀴를 저절로 넘긴다
    const P = PL.cloud.concat(PL.tok), run = kmRun(P, KM_SEED.run), R = run.rounds, n = R.length;
    const put = (C) => C.forEach((p, j) => place(run.letter[j], p));
    const lines = (asg, C) => {
      gL.replaceChildren();
      if (!asg) return;
      const f = document.createDocumentFragment();
      P.forEach((p, i) => { const c = C[asg[i]]; f.append(s('line', { class: 'lk thin', x1: X(p[0]), y1: Y(p[1]), x2: X(c[0]), y2: Y(c[1]) })); });
      gL.append(f);
    };
    const regsOn = (on) => regs.forEach(r => vis([r.e, r.wm], on));
    let timers = [];
    const stop = () => { timers.forEach(clearTimeout); timers = []; };
    const fin = () => {
      put(R[n - 1].to); lines(null); regsOn(true);
      msg.innerHTML = L(`<b>${n}바퀴째</b> 배정이 더 바뀌지 않아 대표점이 움직이지 않는다. 여기서 고정한다.`, `<b>Round ${n}</b> No assignment changes, so the representatives stop moving. They are fixed here.`);
      again.style.display = REDUCED ? 'none' : '';
    };
    const auto = () => {
      again.style.display = 'none';
      if (REDUCED) { fin(); return; }
      let t = 0;
      const T = 950;
      for (let k = 1; k < n; k++) {
        timers.push(setTimeout(() => { put(R[k].from); lines(R[k].asg, R[k].from); regsOn(false); msg.innerHTML = L(`<b>${k + 1}바퀴째</b> 가장 가까운 대표점에 다시 배정한다.`, `<b>Round ${k + 1}</b> Re-assign every token to its nearest representative.`); }, t));
        t += T;
        if (k < n - 1) { timers.push(setTimeout(() => { lines(null); put(R[k].to); msg.innerHTML = L(`<b>${k + 1}바퀴째</b> 대표점을 평균 위치로 다시 옮긴다.`, `<b>Round ${k + 1}</b> Move each representative to the mean again.`); }, t)); t += T; }
      }
      timers.push(setTimeout(fin, t));
    };
    api.step = (k) => {
      stop(); regsOn(false); vis(cens, k >= 1); again.style.display = 'none';
      msg.innerHTML = k === 2 ? L('<b>1바퀴째</b> 배정', '<b>Round 1</b> assign') : k === 3 ? L('<b>1바퀴째</b> 이동', '<b>Round 1</b> move') : '';
      put(k >= 3 ? R[0].to : R[0].from);
      lines(k === 2 ? R[0].asg : null, R[0].from);
      if (k === 4) auto();
    };
    again.addEventListener('click', () => { stop(); regsOn(false); put(R[0].to); lines(null); auto(); });
    api.rounds = n;
    api.step(0);
  } else {
    PL.cen.forEach((p, i) => place(i, p));
    let P = [1.3, 0.9];
    const pt = s('circle', { class: 'nw', r: 7.5, tabindex: '0', 'aria-label': L('새 토큰 (끌어서 옮기기)', 'New token (drag to move)') });
    const lb = s('text', { class: 'dl win', text: 'T_new' });
    const lines = PL.cen.map(() => { const l = s('line', { class: 'lk' }), t = s('text', { class: 'dl', 'text-anchor': 'middle' }); gL.append(l, t); return { l, t }; });
    gN.append(pt, lb);
    const draw = () => {
      const ds = PL.cen.map(c => d2(P, c)), win = near(P, PL.cen);
      pt.setAttribute('cx', X(P[0])); pt.setAttribute('cy', Y(P[1]));
      lb.setAttribute('x', X(P[0]) + 14); lb.setAttribute('y', Y(P[1]) + 4);
      lines.forEach((o, i) => {
        const c = PL.cen[i];
        o.l.setAttribute('x1', X(P[0])); o.l.setAttribute('y1', Y(P[1])); o.l.setAttribute('x2', X(c[0])); o.l.setAttribute('y2', Y(c[1]));
        o.l.setAttribute('class', 'lk' + (i === win ? ' win' : ''));
        o.t.setAttribute('x', (X(P[0]) + X(c[0])) / 2); o.t.setAttribute('y', (Y(P[1]) + Y(c[1])) / 2 - 5);
        o.t.setAttribute('class', 'dl' + (i === win ? ' win' : ''));
        o.t.textContent = Math.hypot(X(P[0]) - X(c[0]), Y(P[1]) - Y(c[1])) < 64 ? '' : ds[i].toFixed(2);  // 너무 가까우면 점에 가려지므로 옆 목록에만 적는다
        regs[i].e.setAttribute('class', 'rg' + (i === win ? ' win' : ''));
        cens[i].setAttribute('class', 'cen' + (i === win ? ' win' : ''));
      });
      if (onChange) onChange({ p: P, d: ds, win });
    };
    const move = (x, y) => { P = [Math.max(-2.9, Math.min(2.9, Math.round(x * 10) / 10)), Math.max(-1.9, Math.min(2.1, Math.round(y * 10) / 10))]; draw(); };
    const fromEvt = (e) => { const r = svg.getBoundingClientRect(), u = (e.clientX - r.left) / r.width * VW, v = (e.clientY - r.top) / r.height * VH; move((u - 40) / 540 * 6 - 3, (350 - v) / 320 * 4.2 - 2); };
    let drag = false;
    pt.addEventListener('pointerdown', (e) => { drag = true; try { pt.setPointerCapture(e.pointerId); } catch (err) { /* 일부 환경 */ } e.preventDefault(); });
    pt.addEventListener('pointermove', (e) => { if (drag) fromEvt(e); });
    const stop = () => { drag = false; };
    pt.addEventListener('pointerup', stop); pt.addEventListener('pointercancel', stop);
    svg.addEventListener('click', (e) => { if (e.target !== pt) fromEvt(e); });
    pt.addEventListener('keydown', (e) => {
      const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
      if (d) { e.preventDefault(); move(P[0] + d[0], P[1] + d[1]); }
    });
    api.move = move;
    draw();
  }
  return api;
}

/* ── 작은 숫자 행렬곱 (GUIDE/04 2 ~ 3단계) ── */
const MM = {
  X: [[2, 1], [1, 2], [1, 0], [1, 1]], Wt: [[1, 2, 1, 2], [1, 0, 2, 1]], Wd: [[1, 2], [2, 1], [1, 0], [2, 1]],
  tiles: [{ r: [0, 1], c: [0, 1] }, { r: [1, 2], c: [2, 3] }],
  T: ['T1', 'T2', 'T3', 'T4'], N: ['N1', 'N2', 'N3', 'N4'],
};
Object.defineProperty(MM, 'D', { get() { return [L('1차원', 'dim 1'), L('2차원', 'dim 2')]; } });
MM.team = (i, j) => MM.tiles.findIndex(t => t.r.includes(i) && t.c.includes(j));
MM.h = MM.T.map((t, i) => MM.N.map((n, j) => (MM.team(i, j) >= 0 ? MM.X[i][0] * MM.Wt[0][j] + MM.X[i][1] * MM.Wt[1][j] : 0)));
function matGrid(title, rl, cl, R, C) {
  const g = h('div', { class: 'mxg', style: { gridTemplateColumns: `max-content repeat(${C}, max-content)` } }, h('div'));
  cl.forEach(c => g.append(h('div', { class: 'hd', text: c })));
  const cells = [];
  for (let i = 0; i < R; i++) {
    g.append(h('div', { class: 'rl', text: rl[i] }));
    cells[i] = [];
    for (let j = 0; j < C; j++) { const e = h('div', { class: 'mc' }); g.append(e); cells[i][j] = e; }
  }
  return { el: h('div', { class: 'mxb' }, h('div', { class: 'mxt', html: title }), g), cells };
}
function MatW(stage, onPick) {
  const el = h('div', { class: 'mx' }), tm = (k) => L('팀 ', 'team ') + (k + 1);
  if (stage === 1) {
    const A = matGrid(L('입력 <b>X</b> (4토큰 × 2차원, 빈칸 없음)', 'input <b>X</b> (4 tokens × 2 dims, dense)'), MM.T, MM.D, 4, 2);
    const B = matGrid(L('가중치 <b>W</b> (2차원 × 4뉴런)', 'weights <b>W</b> (2 dims × 4 neurons)'), MM.D, MM.N, 2, 4);
    const Cc = matGrid(L('결과 <b>h</b> (4토큰 × 4뉴런, 필요한 칸만)', 'result <b>h</b> (4 tokens × 4 neurons, needed cells only)'), MM.T, MM.N, 4, 4);
    el.append(A.el, h('div', { class: 'mxop', text: '×' }), B.el, h('div', { class: 'mxop', text: '=' }), Cc.el);
    const pick = (i, j) => {
      A.cells.forEach((r, a) => r.forEach((e, b) => { e.className = 'mc' + (a === i ? ' src' : ''); e.textContent = MM.X[a][b]; }));
      B.cells.forEach((r, a) => r.forEach((e, b) => { e.className = 'mc' + (b === j ? ' src' : ''); e.textContent = MM.Wt[a][b]; }));
      Cc.cells.forEach((r, a) => r.forEach((e, b) => {
        const k = MM.team(a, b);
        e.className = 'mc ' + (k < 0 ? 'no' : 't' + (k + 1) + ' pick') + (a === i && b === j ? ' sel' : '');
        e.innerHTML = k < 0 ? '0' : MM.h[a][b] + `<small>${tm(k)}</small>`;
      }));
      const k = MM.team(i, j), x = MM.X[i], w = [MM.Wt[0][j], MM.Wt[1][j]];
      if (onPick) onPick({ i, j, k, expr: `h[${MM.T[i]}, ${MM.N[j]}] = ${x[0]}×${w[0]} + ${x[1]}×${w[1]} = <b>${MM.h[i][j]}</b>` });
    };
    Cc.cells.forEach((r, a) => r.forEach((e, b) => { if (MM.team(a, b) >= 0) { e.tabIndex = 0; e.addEventListener('click', () => pick(a, b)); e.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(a, b); } }); } }));
    pick(1, 2);
    return { el, pick };
  }
  const A = matGrid(L('입력 <b>h</b> (방금 1단계 결과)', 'input <b>h</b> (the stage-1 result)'), MM.T, MM.N, 4, 4);
  const B = matGrid(L('가중치 <b>W_down</b> (4뉴런 × 2차원)', 'weights <b>W_down</b> (4 neurons × 2 dims)'), MM.N, MM.D, 4, 2);
  const Cc = matGrid(L('결과 <b>y</b> (4토큰 × 2차원)', 'result <b>y</b> (4 tokens × 2 dims)'), MM.T, MM.D, 4, 2);
  el.append(A.el, h('div', { class: 'mxop', text: '×' }), B.el, h('div', { class: 'mxop', text: '=' }), Cc.el);
  let mode = 'split', si = 1, sj = 0;
  const part = (i, d, k) => MM.tiles[k].c.reduce((a, j) => a + (MM.tiles[k].r.includes(i) ? MM.h[i][j] * MM.Wd[j][d] : 0), 0);
  const terms = (i, d, cols) => cols.filter(j => MM.h[i][j]).map(j => `${MM.h[i][j]}×${MM.Wd[j][d]}`).join(' + ');
  const draw = () => {
    A.cells.forEach((r, a) => r.forEach((e, b) => {
      const k = MM.team(a, b);
      e.className = 'mc ' + (k < 0 ? 'no' : 't' + (k + 1)) + (a === si && k >= 0 ? ' src' : '');
      e.innerHTML = k < 0 ? '0' : MM.h[a][b] + `<small>${L('틀', 'tile')}${CIRC[k]}</small>`;
    }));
    B.cells.forEach((r, a) => r.forEach((e, b) => { e.className = 'mc' + (b === sj ? ' src' : ''); e.textContent = MM.Wd[a][b]; }));
    Cc.cells.forEach((r, a) => r.forEach((e, b) => {
      const p0 = part(a, b, 0), p1 = part(a, b, 1), both = p0 && p1, sel = a === si && b === sj ? ' sel' : '';
      if (!p0 && !p1) { e.className = 'mc no'; e.textContent = '0'; return; }
      if (mode === 'split') {
        e.className = 'mc pick ' + (both ? 'hit' : p0 ? 't1' : 't2') + sel;
        e.innerHTML = both ? `${p0} + ${p1}<small>${L('충돌', 'clash')}</small>` : (p0 || p1) + `<small>${tm(p0 ? 0 : 1)}</small>`;
      } else {
        const k = a < 2 ? 0 : 1;
        e.className = 'mc pick t' + (k + 1) + sel;
        e.innerHTML = (p0 + p1) + `<small>${tm(k)}</small>`;
      }
    }));
    const p0 = part(si, sj, 0), p1 = part(si, sj, 1), nm = `y[${MM.T[si]}, ${MM.D[sj]}]`;
    let expr;
    if (mode === 'split' && p0 && p1) expr = L(`팀 1: ${terms(si, sj, [0, 1])} = <b>${p0}</b>, 팀 2: ${terms(si, sj, [2, 3])} = <b>${p1}</b>. 두 팀이 같은 방 ${nm}에 동시에 더하려다 <b class="c-waste">충돌</b>한다.`,
      `team 1: ${terms(si, sj, [0, 1])} = <b>${p0}</b>, team 2: ${terms(si, sj, [2, 3])} = <b>${p1}</b>. Both teams add into the same slot ${nm} at once: a <b class="c-waste">clash</b>.`);
    else expr = `${nm} = ${terms(si, sj, [0, 1, 2, 3])} = <b>${p0 + p1}</b>` + (mode === 'own' ? L(' — 한 팀이 자기 안에서 다 더한 뒤 1번만 적는다.', ' — one team sums it all internally and writes once.') : L(' — 한 팀만 쓰는 방이라 충돌이 없다.', ' — only one team writes here, so no clash.'));
    if (onPick) onPick({ i: si, j: sj, mode, clash: mode === 'split' && p0 && p1, expr });
  };
  Cc.cells.forEach((r, a) => r.forEach((e, b) => { e.tabIndex = 0; e.addEventListener('click', () => { if (part(a, b, 0) || part(a, b, 1)) { si = a; sj = b; draw(); } }); }));
  draw();
  return { el, setMode: (m) => { mode = m; draw(); }, pick: (a, b) => { si = a; sj = b; draw(); } };
}

/* ── 패러다임 대비 도식 (GUIDE/05 2순위 ①) ──
   기존: 모델이 켜는 뉴런을 예측기가 뒤따라 맞힌다.  우리: 쓸 뉴런을 먼저 정하고, 모델이 거기에 맞춘다.
   opt.compact: 쇼케이스의 한 화면에 들어가게 문장을 줄이고 맺음말을 뺀다 */
function ParadigmW(opt = {}) {
  const T = (ko, en, sko, sen) => (opt.compact ? L(sko, sen) : L(ko, en));
  const strip = (name, on) => h('div', { class: 'pstrip' }, h('span', { text: name }), h('div', { class: 'cells' }, [1, 2, 3, 4, 5, 6, 7, 8].map(n => h('i', { class: on.includes(n) ? 'o' : '' }))));
  const box = (no, title, body, extra) => h('div', { class: 'pbx' }, h('span', { class: 'pn', text: no }), h('b', { html: title }), h('span', { html: body }), extra || null);
  const arrow = () => h('div', { class: 'par', 'aria-hidden': 'true', text: '→' });
  const old = h('div', { class: 'pln old' },
    h('div', { class: 'pln-h' }, h('span', { text: L('기존 패러다임', 'Earlier paradigm') }), L('모델이 켜는 뉴런을 예측기가 뒤따라 맞힌다', 'A predictor chases the neurons the model fires')),
    h('div', { class: 'pfl' },
      box('①', L('모델이 켜는 뉴런을 그대로 둔다', 'The model fires whatever it fires'),
        T('토큰마다 켜지는 뉴런이 제각각이다. 8뉴런이면 최대 2<sup>8</sup> = 256가지, 실제 4,096뉴런이면 2<sup>4096</sup>가지.', 'Every token lights a different set. With 8 neurons that is up to 2<sup>8</sup> = 256 patterns; with 4,096 neurons, 2<sup>4096</sup>.',
          '토큰마다 켜지는 뉴런이 제각각이다. 4,096뉴런이면 2<sup>4096</sup>가지.', 'Every token lights a different set: 2<sup>4096</sup> patterns for 4,096 neurons.')),
      arrow(),
      box('②', L('작은 예측기가 뒤따라 맞힌다', 'A small predictor guesses after the fact'),
        T('별도로 학습한 MLP 예측기가 "이번에는 어느 뉴런이 켜질까"를 맞힌다. 예측 오차가 생긴다.', 'A separately trained MLP predicts which neurons will fire this time. Prediction errors follow.',
          'MLP 예측기가 뒤따라 맞힌다. 예측 오차가 생긴다.', 'An MLP guesses which neurons fire. Errors follow.'),
        // 토큰마다의 무늬는 미리 정해진 것이 아니라 예측기가 그때그때 맞히는 것이므로 ②에 둔다 (①에 두면 미리 정한 것처럼 보인다)
        h('div', { class: 'pstrips' }, strip('T1', [1, 3, 4, 7]), strip('T2', [2, 3, 5, 8]), strip('T3', [1, 2, 6, 8]))),
      arrow(),
      box('③', L('배치로 묶으면 무너진다', 'It breaks down in a batch'),
        T('토큰끼리 무늬가 달라 합집합이 100%로 꽉 찬다. 무늬를 미리 알 수 없어 준비도 할 수 없다.', 'Patterns differ across tokens, so their union fills up to 100%. Nothing can be prepared ahead, because the patterns are unknown.',
          '합집합이 100%로 꽉 찬다. 미리 준비할 수 없다.', 'The union fills up to 100%. Nothing can be prepared ahead.'))));
  const neu = h('div', { class: 'pln new' },
    h('div', { class: 'pln-h' }, h('span', { text: L('우리 패러다임', 'Our paradigm') }), L('쓸 뉴런을 먼저 정하고, 모델이 거기에 맞춘다', 'Fix the neurons first; the model is trained to fit')),
    h('div', { class: 'pfl' },
      box('①', L('쓸 뉴런을 먼저 정한다', 'Fix the neurons first'),
        T('중심점마다 출석부를 정해 둔다. 8뉴런 예시는 4가지, 실제는 앞 3층 32가지 · 4층 256가지로 닫힌다.', 'Each centroid gets a fixed roster. The toy has 4 of them; the real model has 32 for layers 1–3 and 256 for layer 4.',
          '중심점마다 출석부를 정해 둔다. 앞 3층 32가지, 4층 256가지.', 'A fixed roster per centroid: 32 for layers 1–3, 256 for layer 4.'),
        h('div', { class: 'pstrips' }, strip('A', [1, 2, 3, 5, 7, 8]), strip('B', [1, 2, 3, 5, 7]), strip('C', [3, 4, 6, 8]), strip('D', [4, 6, 8]))),
      arrow(),
      box('②', L('모델이 그 규칙에 맞춰 다시 배운다', 'The model is fine-tuned to fit the rule'),
        L('사후학습으로 정확도를 지킨다. 베이스 대비 99.77% (90.81%).', 'Fine-tuning keeps accuracy at 99.77% of the base (90.81%).')),
      arrow(),
      box('③', L('배치로 묶어도 유지된다', 'It holds up in a batch'),
        T('고를 때는 내적 1회다. 같은 중심점 토큰은 뉴런이 100% 겹치고, 표가 고정이라 13.1 ms에 미리 준비한다.', 'Picking takes one dot product. Tokens of the same centroid share 100% of their neurons, and the fixed table is prepared ahead in 13.1 ms.',
          '내적 1회로 고른다. 같은 중심점끼리 100% 겹친다.', 'One dot product picks. Same-centroid tokens overlap 100%.'))));
  const foot = h('div', { class: 'pdg-foot', html: L('<b>뒤집은 것은 순서다.</b> 기존은 모델이 정한 뉴런을 예측기가 따라가고, 우리는 먼저 정한 뉴런에 모델이 맞춘다.', '<b>What is reversed is the order.</b> Before, a predictor followed the model; here, the model follows neurons fixed in advance.') });
  const el = h('div', { class: 'pdg' + (opt.compact ? ' compact' : '') }, old, neu, opt.compact ? null : foot);
  return { el, focus: (k) => { old.classList.toggle('dim', k === 'new'); neu.classList.toggle('dim', k === 'old'); } };
}
