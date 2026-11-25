'use strict';
/* Roster v3 · 쇼케이스: 본편의 작은 예시 8장면을 자동으로 넘겨 보여 준다 (한국어 / 영어) */
const ROOT_EL = document.getElementById('scenes'), VIZ = ROOT_EL.dataset.viz;
const players = [];
let curScene = 0;
const sv = (label, val, cls) => h('div', { class: 'sv ' + (cls || '') }, h('span', { text: label }), h('b', { html: val }));
const N8s = [1, 2, 3, 4, 5, 6, 7, 8];

/* 한 장면 안의 걸음(상태). 저절로 넘어가지 않는다. 아래의 이전/다음 버튼이나 점을 눌러 넘긴다. ctl = {n, go(k), cap(k)} */
function playerFor(stage, ctl, onChange) {
  const cap = h('div', { class: 'scap', 'aria-live': 'polite' });
  let i = 0;
  const show = (k) => { i = k; ctl.go(k); cap.innerHTML = ctl.cap(k); dots.forEach((d, j) => d.classList.toggle('on', j === k)); if (onChange) onChange(); };
  const dots = Array.from({ length: ctl.n }, (_, k) => h('button', { type: 'button', 'aria-label': L('걸음 ', 'Step ') + (k + 1), onclick: () => show(k) }));
  stage.append(cap, h('div', { class: 'sdots' }, ctl.n > 1 ? dots : null));
  return { show, n: ctl.n, cur: () => i };
}

/* 첫 장면: 신경망 그림. 실제 시험 사진 한 장이 층 4개를 지날 때, 베이스는 뉴런 4,096개를 전부 켜고
   우리는 층에 들어가기 전에 중심점을 먼저 찾아 그 출석부의 뉴런만 켠다. 점 하나가 뉴런 하나다 (실제 데이터, net_data.js). */
function netScene(stage) {
  const FONT = '"IBM Plex Sans KR", "Malgun Gothic", sans-serif', MONO = '"IBM Plex Mono", Consolas, monospace';
  let sel = 0, step = 0;
  const cv = h('canvas', { class: 'netcv', role: 'img', 'aria-label': L('베이스는 4개 층의 뉴런을 전부 켜고, 우리는 중심점의 출석부에 있는 뉴런만 켜는 그림', 'The base fires every neuron in 4 layers; ours fires only the roster of the chosen centroid') });
  const bars = h('div', { class: 'netbars' });
  const chips = NET.ex.map((e, i) => h('button', { type: 'button', class: 'nchip', title: L(e.ko, e.en), 'aria-label': L(e.ko, e.en), onclick: () => { sel = i; paint(); } }, h('img', { src: e.img, alt: '', width: 28, height: 28 })));
  const hint = h('div', { class: 'n8t' });
  stage.append(h('div', { class: 'nrow' }, h('div', { class: 'nchips' }, chips), hint), cv, bars);
  // 비트 묶음 풀기: 뉴런 번호 순, 바이트의 높은 비트가 먼저
  const bits = NET.ex.map(e => e.layers.map(l => { const bin = atob(l.on), a = new Uint8Array(4096); for (let n = 0; n < 4096; n++) a[n] = (bin.charCodeAt(n >> 3) >> (7 - (n & 7))) & 1; return a; }));
  const imgs = NET.ex.map(e => { const im = new Image(); im.onload = () => draw(); im.src = e.img; return im; });

  function draw() {
    if (!cv.isConnected || !cv.clientWidth) return;
    const VW = 700, VH = 306, sc = cv.clientWidth / VW, dpr = window.devicePixelRatio || 1;
    cv.style.height = VH * sc + 'px';
    cv.width = Math.round(cv.clientWidth * dpr); cv.height = Math.round(VH * sc * dpr);
    const x = cv.getContext('2d'), css = getComputedStyle(document.documentElement), C = (n) => css.getPropertyValue(n).trim();
    x.setTransform(sc * dpr, 0, 0, sc * dpr, 0, 0);
    x.clearRect(0, 0, VW, VH);
    const e = NET.ex[sel], S = 96, X0 = 118, GAP = 50, P = S / 64, im = imgs[sel];
    const row = (y, ours) => {
      x.textAlign = 'left'; x.textBaseline = 'alphabetic';
      x.font = '600 13px ' + FONT; x.fillStyle = C('--fg');
      x.fillText(ours ? L('우리 (Roster): 미리 고른 뉴런만 켠다', 'Ours (Roster): fire only the neurons picked ahead') : L('베이스: 뉴런을 전부 켠다', 'Base: fire every neuron'), 0, y - S / 2 - 12);
      x.strokeStyle = C('--fg-3'); x.lineWidth = 1.5;
      x.beginPath(); x.moveTo(48, y); x.lineTo(X0 + 4 * S + 3 * GAP, y); x.stroke();   // 층을 차례로 지나는 선
      x.fillStyle = '#000'; x.fillRect(0, y - 24, 48, 48);
      if (im.complete && im.naturalWidth) { x.imageSmoothingEnabled = false; x.drawImage(im, 0, y - 24, 48, 48); }
      x.strokeStyle = C('--line-2'); x.lineWidth = 1; x.strokeRect(0.5, y - 23.5, 47, 47);
      for (let l = 0; l < 4; l++) {
        const bx = X0 + l * (S + GAP), by = y - S / 2, ly = e.layers[l], on = bits[sel][l], lit = ours && step >= 2;
        x.fillStyle = C('--bg'); x.fillRect(bx - 3, by - 3, S + 6, S + 6);
        x.strokeStyle = C('--line-2'); x.lineWidth = 1; x.strokeRect(bx - 2.5, by - 2.5, S + 5, S + 5);
        if (!ours) { x.fillStyle = C('--f-on'); for (let n = 0; n < 4096; n++) x.fillRect(bx + (n & 63) * P, by + (n >> 6) * P, P * 0.72, P * 0.72); }
        else {
          x.fillStyle = C('--off'); for (let n = 0; n < 4096; n++) x.fillRect(bx + (n & 63) * P, by + (n >> 6) * P, P * 0.6, P * 0.6);
          if (lit) { x.fillStyle = C('--on'); for (let n = 0; n < 4096; n++) if (on[n]) x.fillRect(bx + (n & 63) * P - 0.2, by + (n >> 6) * P - 0.2, P + 0.4, P + 0.4); }
        }
        if (ours && step >= 1) {   // 층에 들어가기 전에 중심점을 먼저 찾는다: 선에서 갈라지는 가지와 마름모
          const dx = bx - GAP / 2 - (l ? 0 : 6), dy = y + 22;
          x.strokeStyle = C('--ours'); x.lineWidth = 1.5; x.beginPath(); x.moveTo(dx - 12, y); x.lineTo(dx, dy); x.lineTo(bx - 3, dy); x.stroke();
          x.fillStyle = C('--ours'); x.beginPath(); x.moveTo(dx, dy - 6); x.lineTo(dx + 6, dy); x.lineTo(dx, dy + 6); x.lineTo(dx - 6, dy); x.closePath(); x.fill();
        }
        x.font = '600 12px ' + FONT; x.fillStyle = C('--fg-2');
        const pct = (100 * ly.K / 4096).toFixed(1) + '%';
        x.fillText(L(`${l + 1}층 `, `L${l + 1} `) + (!ours ? '100%' : lit ? pct : ''), bx, y + S / 2 + 17);
        if (ours && step >= 1) { x.font = '500 11px ' + FONT; x.fillStyle = C('--ours'); x.fillText(L(`중심점 ${ly.j}번 (${ly.M}개 중)`, `centroid ${ly.j} of ${ly.M}`), bx, y + S / 2 + 32); }
      }
    };
    row(74, false);
    row(218, true);
  }
  function paint() {
    chips.forEach((c, i) => c.classList.toggle('on', i === sel));
    hint.textContent = L(`실제 시험 사진: ${NET.ex[sel].ko} (눌러서 바꾼다)`, `Real test image: ${NET.ex[sel].en} (click to change)`);
    draw();
    bars.replaceChildren(step >= 3 ? h('div', null, h('div', { class: 'bl-t', text: L('토큰 1개가 층 4개를 지나는 시간 (RTX 4060 실측, d = 4,096, 토큰 32개 표본 평균)', 'Time for one token through 4 layers (measured on an RTX 4060, d = 4,096, mean of a 32-token sample)') }),
      barList([{ l: L('베이스 (다 켬)', 'Base (all on)'), v: 1.7705, t: '1.7705 ms', k: 'base' }, { l: L('우리', 'Ours'), v: 0.2824, t: '0.2824 ms', s: '6.27×', k: 'ours' }], { max: 1.7705, room: 70 })) : '');
  }
  document.addEventListener('roster-theme', draw);
  window.addEventListener('resize', draw);
  return {
    n: 4, go: (k) => { step = k; paint(); },
    cap: (k) => [
      L('<b>베이스</b> 4개 층의 뉴런을 전부 켠다. 점 하나가 뉴런 하나이고, 층마다 4,096개다.', '<b>Base</b> Every neuron in all 4 layers fires. One dot is one neuron; each layer has 4,096.'),
      L('<b>미리 고르기</b> 우리는 층에 들어가기 전에, 입력과 가장 가까운 중심점을 먼저 찾는다(내적 1회). 앞 3층은 32개, 4층은 256개 가운데 하나다.', '<b>Pick ahead</b> Before each layer we find the centroid nearest to the input with one dot product: one of 32 in layers 1–3, one of 256 in layer 4.'),
      L('<b>일부만 켜기</b> 그 중심점의 출석부에 적힌 뉴런만 켠다. 꺼진 뉴런은 메모리에서 읽지도 않는다. 사진을 바꾸면 중심점과 켜지는 뉴런이 달라진다.', '<b>Fire only a few</b> Only the neurons on that centroid\'s roster fire; the rest are never read from memory. Other images pick other centroids.'),
      L('<b>그래서 빠르다</b> 읽는 양이 줄어든 만큼 토큰 1개 계산이 6.27배 빨라진다.', '<b>So it is fast</b> Reading that much less makes one token 6.27× faster.'),
    ][k],
  };
}

const SCENES = [
  {
    eb: () => L('일부만 켠다', 'Fire only a few'), link: '#s02-0', linkT: () => L('본편에서 자세히 보기', 'See it in the full guide'),
    title: () => L('FFN은 토큰마다 뉴런을 전부 켠다. 우리는 층에 들어가기 전에 쓸 뉴런을 미리 골라, 그것만 켠다.', 'An FFN fires every neuron for every token. We pick the neurons before entering each layer, and fire only those.'),
    big: () => '13.52%', sub: () => L('평균 켜는 뉴런. 시험 정확도는 베이스 대비 99.77% (90.81%, 베이스 91.02%)', 'of neurons on, on average. Test accuracy is 99.77% of the base (90.81%; base 91.02%)'),
    build(stage) { return netScene(stage); },
  },
  {
    eb: () => L('학습', 'Training'), link: '#s01-4', linkT: () => L('학습 흐름 자세히 보기', 'See the training flow'),
    title: () => L('입력이 모이는 구역(중심점)마다 켤 뉴런 목록(출석부)을 정하고, 모델이 그 목록에 맞게 다시 배운다.', 'For each region of inputs (a centroid) we fix a roster of neurons to fire, then the model is fine-tuned to fit those rosters.'),
    big: () => L('9바퀴', '9 rounds'), sub: () => L('일을 앞쪽 블록으로 몰고 꼬리를 자르기를 되풀이한다 (예시: 뉴런 8개, 중심점 4개)', 'shift work into the front blocks, cut the tail, repeat (toy: 8 neurons, 4 centroids)'),
    build(stage) {
      const R = RosterW(null, { compact: true });
      stage.append(R.el);
      const S = [
        [{ ord: 0, blocks: false, view: 'num0' }, '<b>기여도 재기</b> 중심점 4개마다 뉴런 8개가 출력에 보탠 힘을 잰다.', '<b>Measure</b> For each of the 4 centroids, measure how much each of the 8 neurons contributes.'],
        [{ ord: 0, blocks: true, view: 'onoff0' }, '<b>출석부</b> 기여도가 큰 앞 K개만 켠다. 원래 순서로는 1과 0이 블록마다 섞인다.', '<b>Roster</b> Keep only the top K. In the original order, 1s and 0s mix in every block.'],
        [{ ord: 1, blocks: true, fam: true, view: 'onoff0', hot: 2 }, '<b>열 재배치</b> 무늬가 비슷한 뉴런을 옆자리로 모은다. 블록 3이 어중간하게 남는다.', '<b>Reorder</b> Put similar neurons side by side. Block 3 is left half-used.'],
        [{ ord: 1, blocks: true, fam: true, view: 'num1', hot: 2 }, '<b>사후학습</b> 벌점이 블록 3의 일을 블록 1·2로 옮긴다 (N3 30 → 3, N2 10 → 30).', '<b>Fine-tune</b> Penalties move block 3\'s work into blocks 1 and 2 (N3 30 → 3, N2 10 → 30).'],
        [{ ord: 1, blocks: true, fam: true, view: 'onoff1' }, '<b>K 조이기</b> 일이 빠진 꼬리를 자른다. 가족 A · B가 같은 블록을 100% 함께 쓴다.', '<b>Tighten K</b> Cut the emptied tail. Family A · B now shares the same blocks 100%.'],
      ];
      return { n: S.length, ms: 3000, go: (k) => R.set(S[k][0]), cap: (k) => L(S[k][1], S[k][2]) };
    },
  },
  {
    eb: () => L('단일 요청', 'Single request'), link: '#s02-1', linkT: () => L('단일 요청 자세히 보기', 'See the single-request flow'),
    title: () => L('새 토큰은 가장 가까운 중심점의 출석부만 읽는다. 꺼진 뉴런은 메모리에서 읽지도 않는다.', 'A new token reads only the roster of its nearest centroid. Neurons that are off are never even read from memory.'),
    big: () => '6.27×', sub: () => L('토큰 1개 (B = 1), RTX 4060 실측. 다 계산하고 0만 곱하는 방식은 0.99×', 'one token (B = 1), measured on an RTX 4060. Computing everything and multiplying by zero gives 0.99×'),
    build(stage) {
      const chips = N8s.map(n => h('div', { text: 'N' + n })), lab = h('div', { class: 'n8t' }), rost = [[1, 2, 5, 7], [1, 2, 5, 7], [4, 6, 8], [4, 6, 8]];
      const P = PlaneW('assign', (o) => {
        chips.forEach((c, i) => c.classList.toggle('on', rost[o.win].includes(i + 1)));
        lab.textContent = L(`중심점 ${PL.name[o.win]}의 출석부: 켜진 ${rost[o.win].length}줄만 메모리에서 읽는다`, `Roster of centroid ${PL.name[o.win]}: only the ${rost[o.win].length} lit rows are read from memory`);
      });
      stage.append(P.el, P.legend, lab, h('div', { class: 'n8' }, chips));
      const pos = [[1.3, 0.9], [2.3, 1.5], [-1.0, -1.3], [-2.2, -0.5]];
      return {
        n: pos.length, ms: 2600, go: (k) => P.move(pos[k][0], pos[k][1]),
        cap: (k) => L(`<b>새 토큰 (${pos[k][0].toFixed(1)}, ${pos[k][1].toFixed(1)})</b> 고정된 중심점 4개와 거리만 한 번 재서 가장 가까운 방을 고른다. 주황색 점을 직접 끌어 볼 수 있다.`,
          `<b>New token (${pos[k][0].toFixed(1)}, ${pos[k][1].toFixed(1)})</b> One distance check against the 4 fixed centroids picks the nearest region. You can drag the orange point yourself.`),
      };
    },
  },
  {
    eb: () => L('배치의 벽', 'The batch wall'), link: '#s03-2', linkT: () => L('기존 방식이 막히는 까닭 보기', 'See why earlier methods stall'),
    title: () => L('서버에는 토큰이 수백 개씩 들어온다. 서로 다른 출석부를 그냥 합치면 다시 전부 계산하게 된다.', 'Servers get hundreds of tokens at once. Simply merging their different rosters means computing everything again.'),
    big: () => '1.0×', sub: () => L('배치 합집합: 48칸을 전부 계산하고 그중 25칸이 헛계산(×0)', 'batch union: all 48 cells computed, 25 of them wasted (×0)'),
    build(stage) {
      const G = GridW(Object.assign({}, G68, { blocks: true })), sb = h('div', { class: 'sb' });
      stage.append(G.el, sb);
      return {
        n: 2, ms: 3400,
        go: (k) => { G.show(k ? { tiles: TS.union } : null); sb.replaceChildren(k ? sv(L('계산하는 칸', 'Cells computed'), '48 / 48') : sv(L('필요한 1', 'Needed 1s'), '23 / 48'), k ? sv(L('헛계산 ×0', 'Wasted ×0'), '25', 'bad') : sv(L('토큰 × 뉴런', 'Tokens × neurons'), '6 × 8')); },
        cap: (k) => (k ? L('<b>배치 합집합</b> 1명이라도 쓰는 뉴런을 모두에게 곱한다. 12틀로 표 전체를 계산한다.', '<b>Batch union</b> Every neuron that anyone uses is multiplied for everyone: 12 tiles cover the whole table.')
          : L('<b>토큰 6명의 출석부</b> 가로줄이 토큰, 세로줄이 뉴런이다. 필요한 칸(1)은 23칸뿐이다.', '<b>Rosters of 6 tokens</b> Rows are tokens, columns are neurons. Only 23 cells (the 1s) are needed.')),
      };
    },
  },
  {
    eb: () => L('비격자 타일링', 'Free-form tiling'), link: '#s03-3', linkT: () => L('틀을 하나씩 넘겨 보기', 'Step through the tiles'),
    title: () => L('GPU는 네모 틀 단위로 계산한다. 틀마다 행과 열의 짝을 자유롭게 바꿔 1만 골라 담는다.', 'A GPU computes in rectangular tiles. We re-pair rows and columns freely for every tile, so tiles hold only the 1s.'),
    big: () => '25 → 1', sub: () => L('같은 표의 헛계산(×0) 칸 수: 합집합 25칸, 고정 행 패널 9칸, 우리 1칸', 'wasted (×0) cells on the same table: union 25, fixed row panels 9, ours 1'),
    build(stage) {
      const G = GridW(Object.assign({}, G68, { blocks: true })), sb = h('div', { class: 'sb' });
      stage.append(G.el, sb);
      const S = [
        [TS.union, '<b>배치 합집합</b> 12틀, ×0이 25칸.', '<b>Batch union</b> 12 tiles, 25 wasted cells.'],
        [TS.bsmr, '<b>고정 2행 패널 (BSMR 최선)</b> 토큰의 행 짝을 못 바꿔 8틀, ×0이 9칸.', '<b>Fixed 2-row panels (best BSMR)</b> Row pairs cannot change: 8 tiles, 9 wasted cells.'],
        [TS.ours, '<b>우리 비격자</b> 틀마다 행과 열을 다시 짝지어 6틀, ×0이 1칸.', '<b>Ours, free-form</b> Rows and columns re-paired per tile: 6 tiles, 1 wasted cell.'],
        [TS.multi, '<b>우리 비격자 + 다중 크기</b> 큰 덩어리는 큰 틀로 덮어 4틀, ×0이 1칸.', '<b>Ours, free-form + mixed sizes</b> Big groups take big tiles: 4 tiles, 1 wasted cell.'],
      ];
      return {
        n: S.length, ms: 3000,
        go: (k) => { const set = { tiles: S[k][0] }, st = G.stats(set); G.show(set); sb.replaceChildren(sv(L('네모 틀', 'Tiles'), String(st.tiles)), sv(L('계산하는 칸', 'Cells computed'), st.cells + ' / 48'), sv(L('헛계산 ×0', 'Wasted ×0'), String(st.zeros), st.zeros > 1 ? 'bad' : 'good')); },
        cap: (k) => L(S[k][1], S[k][2]),
      };
    },
  },
  {
    eb: () => L('실행', 'Execution'), link: '#s04-1', linkT: () => L('실행 순서 자세히 보기', 'See the execution order'),
    title: () => L('타일 계획은 어텐션이 도는 동안 CPU가 짜서 숨긴다. 두 팀이 같은 칸에 쓰는 일이 없어 충돌도 없다.', 'The CPU plans the tiles while attention runs, so planning hides behind it. No two teams ever write the same cell, so nothing collides.'),
    big: () => '0.000 ms', sub: () => L('어텐션 뒤에 드러나는 계획 시간 (d = 4,096에서 계획 + 전송은 1.41 ~ 2.09 ms)', 'planning time exposed after attention (at d = 4,096, plan + transfer takes 1.41–2.09 ms)'),
    build(stage) {
      const D = [['B = 512', 1.41, 3.241, '2.3×'], ['B = 1,024', 1.67, 6.184, '3.7×'], ['B = 2,048', 2.09, 13.299, '6.4×']];
      const att = h('div', { class: 'blk gpu', style: { left: '0', width: '68%' } }), ffn = h('div', { class: 'blk ffn', style: { left: 'calc(68% + 2px)', right: '0' }, text: 'FFN' });
      const cpu = h('div', { class: 'blk cpu', style: { left: '0', width: '0%', padding: '0' } }), out = h('div', { class: 'out' });
      const Mw = MatW(1), picks = [[1, 0], [1, 2], [2, 3]];
      stage.append(h('div', { class: 'flow' }, h('div', { class: 'fl' }, h('div', { class: 'ft', text: 'GPU' }), h('div', { class: 'trk' }, att, ffn)), h('div', { class: 'fl' }, h('div', { class: 'ft', text: 'CPU → GPU' }), h('div', { class: 'trk' }, cpu, out))),
        h('div', { class: 'n8t', text: L('FFN 1단계: 두 팀이 서로 다른 칸에만 적는다 (T2는 두 틀에 걸친다)', 'FFN stage 1: the two teams write to different cells only (T2 sits in both tiles)') }), Mw.el);
      return {
        n: 3, ms: 3000,
        go: (k) => { const d = D[k], w = 68 * d[1] / d[2]; att.textContent = L(`어텐션 ${d[2].toFixed(3)} ms`, `attention ${d[2].toFixed(3)} ms`); cpu.style.width = w + '%'; out.style.left = `calc(${w}% + 6px)`; out.textContent = L(`계획 + 전송 ${d[1].toFixed(2)} ms`, `plan + transfer ${d[1].toFixed(2)} ms`); Mw.pick(picks[k][0], picks[k][1]); },
        cap: (k) => L(`<b>${D[k][0]}</b> 계획 + 전송 ${D[k][1].toFixed(2)} ms가 어텐션 ${D[k][2].toFixed(3)} ms 안에 끝난다 (여유 ${D[k][3]}).`, `<b>${D[k][0]}</b> Plan + transfer (${D[k][1].toFixed(2)} ms) finishes inside attention (${D[k][2].toFixed(3)} ms), with ${D[k][3]} headroom.`),
      };
    },
  },
  {
    eb: () => L('실측', 'Measured'), link: '#s04-4', linkT: () => L('실측 표 전체 보기', 'See all measurements'),
    title: () => L('배치가 커질수록 같은 중심점에 토큰이 더 모여, 오히려 더 빨라진다.', 'The larger the batch, the more tokens share a centroid, so it gets faster, not slower.'),
    big: () => '4.57×', sub: () => L('B = 2,048, d = 4,096, 4층 합 (RTX 4060, bf16, 어텐션 중첩 E2E)', 'B = 2,048, d = 4,096, 4 layers (RTX 4060, bf16, end to end with attention overlap)'),
    build(stage) {
      const box = h('div'), A = [['B = 512', 3.08], ['B = 1,024', 4.09], ['B = 2,048', 4.57]];
      stage.append(box, h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-base' }), L('베이스 (다 켬, cuBLAS)', 'Base (all on, cuBLAS)')), h('span', null, h('i', { class: 'i-ours' }), L('우리', 'Ours'))));
      return {
        n: 1,
        go: () => box.replaceChildren(h('div', { class: 'bl-t', text: L('가속 배수 (베이스 = 1.00×)', 'Speedup (base = 1.00×)') }), barList([{ l: L('베이스', 'Base'), v: 1, t: '1.00×', k: 'base' }].concat(A.map(x => ({ l: x[0], v: x[1], t: x[1].toFixed(2) + '×', k: 'ours' }))), { max: 5, room: 74 })),
        cap: () => L('<b>어텐션 중첩 E2E</b> 배치 512 / 1,024 / 2,048에서 3.08× / 4.09× / 4.57×.', '<b>End to end, overlapped with attention</b> 3.08× / 4.09× / 4.57× at batch 512 / 1,024 / 2,048.'),
      };
    },
  },
  {
    eb: () => L('패러다임 역전', 'The reversal'), link: '#s05-2', linkT: () => L('본편에서 자세히 보기', 'See it in the full guide'),
    title: () => L('기존은 모델이 켜는 뉴런을 뒤따라 맞혔다. 우리는 쓸 뉴런을 먼저 정하고, 모델이 거기에 맞춘다.', 'Earlier work chased the neurons a model happens to fire. We fix the neurons first, and the model is trained to fit.'),
    big: () => L('32가지', '32 patterns'), sub: () => L('앞 3층에서 나올 수 있는 뉴런 무늬의 수 (4층은 256가지). 기존은 2의 4,096제곱 가지', 'possible neuron patterns in layers 1–3 (256 in layer 4). Earlier: 2 to the power 4,096'),
    build(stage) {
      const Pw = ParadigmW({ compact: true });
      stage.append(Pw.el);
      return {
        n: 2, go: (k) => Pw.focus(k ? 'new' : 'old'),
        cap: (k) => (k ? L('<b>우리</b> 출석부가 유한해서 내적 1회로 고르고, 배치로 묶어도 같은 중심점끼리 100% 겹친다.', '<b>Ours</b> Rosters are finite, so one dot product picks them, and tokens of the same centroid overlap 100% even in a batch.')
          : L('<b>기존</b> 토큰마다 무늬가 제각각이라 예측기가 뒤따라 맞히고, 배치로 묶으면 합집합이 100%로 꽉 찬다.', '<b>Earlier</b> Every token has its own pattern, so a predictor guesses after the fact, and in a batch the union fills up to 100%.')),
      };
    },
  },
  {
    eb: () => L('새로운 점', 'What is new'), link: '#05', linkT: () => L('본편에서 차근차근 보기', 'Read the full guide'), cta: true,
    title: () => L('새로운 점은 세 축이고, 서로가 서로의 전제 조건이다.', 'There are three contributions, and each is a precondition for the next.'),
    big: () => L('3축', '3 axes'), sub: () => L('하나라도 빼면 나머지가 무너진다', 'remove any one and the rest collapses'),
    build(stage) {
      const bx = (no, t, p) => h('div', { class: 'bx' }, h('h4', null, h('span', { class: 'no', text: no }), t), h('p', { text: p }));
      const b2 = bx(L('2순위', 'No. 2'), L('규칙을 먼저 정한다', 'Fix the rules first'), L('유한한 출석부(32 / 256가지)를 먼저 정하고, 모델이 사후학습으로 거기에 맞춘다.', 'Fix a finite set of rosters (32 / 256) first; the model is fine-tuned to fit them.'));
      const b1 = bx(L('1순위', 'No. 1'), L('실시간 비격자 5단계 타일링', 'Real-time free-form 5-level tiling'), L('틀마다 행·열을 다시 짝지어 1만 덮는다. 계획은 1.49 ~ 2.83 ms.', 'Re-pair rows and columns per tile to cover only the 1s. Planning takes 1.49–2.83 ms.'));
      const b3 = bx(L('3순위', 'No. 3'), L('단일 요청과 서버를 함께 실측', 'Measured on both single and server'), L('토큰별 뉴런 집합을 유지한 채 6.27× (B = 1), 4.57× (B = 2,048).', 'Per-token neuron sets kept: 6.27× (B = 1), 4.57× (B = 2,048).'));
      stage.append(h('div', { class: 'diag' }, b2, h('div', { class: 'ar', text: '▼' }), b1, h('div', { class: 'ar', text: '▼' }), b3));
      const S = [
        [[], [], '<b>세 축이 모두 있을 때</b> 위에서 아래로 성질이 이어진다.', '<b>All three in place</b> Each property feeds the next, top to bottom.'],
        [[b2], [b1, b3], '<b>2순위를 빼면</b> 출석부가 유한하지 않아 타일 계획을 미리 준비할 수 없다.', '<b>Without No. 2</b> Rosters are not finite, so tile plans cannot be prepared ahead.'],
        [[b1], [b3], '<b>1순위를 빼면</b> 단일 요청의 이득이 대형 배치로 이어지지 않는다.', '<b>Without No. 1</b> The single-request gain does not carry over to large batches.'],
        [[], [b2, b1, b3], '<b>공유 사후학습만 빼면</b> 헛계산(×0)과 메모리 읽기가 다시 커진다.', '<b>Without sharing-aware fine-tuning</b> Wasted cells (×0) and memory reads grow back.'],
      ];
      return { n: S.length, ms: 3200, go: (k) => [b2, b1, b3].forEach(b => { b.classList.toggle('off', S[k][0].includes(b)); b.classList.toggle('broken', S[k][1].includes(b)); }), cap: (k) => L(S[k][2], S[k][3]) };
    },
  },
];

function paintNav() {
  const p = players[curScene];
  $('#sc-pg').textContent = L(`장면 ${curScene + 1} / ${players.length} · 걸음 ${p.cur() + 1} / ${p.n}`, `Scene ${curScene + 1} / ${players.length} · step ${p.cur() + 1} / ${p.n}`);
  $('#sc-prev').disabled = curScene === 0 && p.cur() === 0;
  $('#sc-next').disabled = curScene === players.length - 1 && p.cur() === p.n - 1;
}
function goScene(i, k) {
  curScene = Math.max(0, Math.min(players.length - 1, i));
  players.forEach((p, j) => p.el.classList.toggle('on', j === curScene));
  $$('#sc-tabs .tab-b').forEach((b, j) => { b.classList.toggle('on', j === curScene); b.setAttribute('aria-selected', j === curScene ? 'true' : 'false'); });
  players[curScene].show(k == null ? 0 : k);   // 보이게 한 뒤에 그려야 자리 바꿈이 부드럽다
  window.scrollTo(0, 0);
  try { history.replaceState(null, '', '#sc' + (curScene + 1)); } catch (e) { /* file:// 등에서 막혀도 동작 */ }
  paintNav();
}
function next() { const p = players[curScene]; if (p.cur() < p.n - 1) p.show(p.cur() + 1); else if (curScene < players.length - 1) goScene(curScene + 1, 0); }
function prev() { const p = players[curScene]; if (p.cur() > 0) p.show(p.cur() - 1); else if (curScene > 0) goScene(curScene - 1, players[curScene - 1].n - 1); }

function render() {
  players.length = 0;
  ROOT_EL.replaceChildren();
  const tabs = $('#sc-tabs');
  tabs.replaceChildren();
  SCENES.forEach((S, idx) => {
    const stage = h('div', { class: 'stage' });
    const sec = h('section', { class: 'sc', id: 'sc' + (idx + 1) },
      h('div', null, h('div', { class: 'eb', text: `${idx + 1} / ${SCENES.length} · ${S.eb()}` }), h('h2', { text: S.title().replace(/\)(?=[가-힣])/g, ')\u2060') }), h('div', { class: 'bign', text: S.big() }), h('div', { class: 'bigs', text: S.sub() }),
        h('a', { class: 'more' + (S.cta ? ' cta' : ''), href: VIZ + S.link, text: S.linkT() })), stage);
    ROOT_EL.append(sec);
    const p = playerFor(stage, S.build(stage), () => { if (players[curScene] === p) paintNav(); });
    p.el = sec;
    players.push(p);
    tabs.append(h('button', { type: 'button', class: 'tab-b', role: 'tab', onclick: () => goScene(idx, 0) }, h('span', { class: 'n', text: String(idx + 1) }), S.eb()));
  });
  $('#sc-sub').textContent = L(`${SCENES.length}장면 쇼케이스`, `${SCENES.length}-scene showcase`);
  $('#sc-full').textContent = L('본편 (전체 가이드)', 'Full guide');
  $('#sc-lang').textContent = LANG === 'ko' ? 'EN' : '한국어';
  $('#sc-prev').textContent = L('이전', 'Back');
  $('#sc-next').textContent = L('다음', 'Next');
  document.documentElement.lang = LANG;
  setTheme(document.documentElement.getAttribute('data-theme') || 'dark');
  goScene(curScene, 0);
}

document.addEventListener('DOMContentLoaded', () => {
  try { LANG = localStorage.getItem('roster-lang') || 'ko'; } catch (e) { /* 기본값 */ }
  const q = new URLSearchParams(location.search).get('lang');
  if (q) LANG = q;
  const m = /^#sc(\d)$/.exec(location.hash);
  if (m) curScene = Number(m[1]) - 1;
  initTheme();
  render();
  $('#sc-lang').addEventListener('click', () => { LANG = LANG === 'ko' ? 'en' : 'ko'; try { localStorage.setItem('roster-lang', LANG); } catch (e) { /* 저장 못 해도 동작 */ } render(); });
  $('#sc-prev').addEventListener('click', prev);
  $('#sc-next').addEventListener('click', next);
  window.addEventListener('hashchange', () => { const mm = /^#sc(\d)$/.exec(location.hash); if (mm && Number(mm[1]) - 1 !== curScene) goScene(Number(mm[1]) - 1, 0); });
  document.addEventListener('keydown', (e) => {
    const t = e.target && e.target.closest ? e.target : null;
    if (t && t.closest('.pl')) return;                         // 좌표 평면의 점은 방향키로 옮긴다
    if (e.key === ' ' && t && t.closest('button, a')) return;   // 버튼 위의 스페이스는 그 버튼을 누른다
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); next(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); prev(); }
  });
});

/* 자동 검사용: 두 언어에서 모든 장면의 모든 걸음을 한 번씩 보이고, 넘침과 무대 안 스크롤을 센다 */
window.__selftest = () => {
  const wide = [];
  for (const lang of ['en', 'ko']) {
    LANG = lang;
    render();
    players.forEach((p, i) => {
      goScene(i, 0);
      for (let k = 0; k < p.n; k++) {
        p.show(k);
        for (const e of $$('.gx', p.el)) if (e.scrollWidth > e.clientWidth + 1) wide.push(`scroll:${lang}:scene${i + 1}:state${k + 1}`);   // 무대 안에 가로 스크롤이 생기면 안 된다
        if (window.innerWidth > 1040 && document.documentElement.scrollHeight > window.innerHeight + 1) wide.push(`tall:${lang}:scene${i + 1}:state${k + 1}:+${document.documentElement.scrollHeight - window.innerHeight}px`);   // 넓은 화면에서는 한 장면이 한 화면에 다 들어와야 한다
      }
      if (document.documentElement.scrollWidth > document.documentElement.clientWidth + 1) wide.push(`page:${lang}:scene${i + 1}`);
      const r = p.el.querySelector('.stage').getBoundingClientRect();
      if (r.right > document.documentElement.clientWidth + 1) wide.push(`stage:${lang}:scene${i + 1}`);
    });
    for (let n = 0; n < 60; n++) prev();   // 이전/다음으로 처음부터 끝까지 한 번 훑는다
    for (let n = 0; n < 60; n++) next();
    if (curScene !== players.length - 1) wide.push('nav:' + lang);
  }
  goScene(0, 0);
  return { nums: [], wide };
};
