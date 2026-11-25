'use strict';
/* Roster v3 · 본편: 탭마다의 위젯 연결과 탭 이동 */
const card = (tag, title, ...kids) => h('div', { class: 'card' }, tag ? h('span', { class: 'tag', text: tag }) : null, title ? h('h4', { html: title }) : null, kids);
const capEl = () => h('div', { class: 'cap', 'aria-live': 'polite' });
const stat = (label, val, cls) => h('div', { class: 'sv ' + (cls || '') }, h('span', { text: label }), h('b', { html: val }));
const N8 = [1, 2, 3, 4, 5, 6, 7, 8];

/* ───────── 01 학습 ───────── */
W.kmeans = (el) => {
  const P = PlaneW('kmeans');
  const sp = stepper([
    { t: '토큰을 좌표에 뿌린다', d: '원본 모델에 학습 토큰을 넣어 이 층에 들어오는 좌표를 찍는다. 작은 점 하나가 토큰 하나이고, 이름표가 붙은 T1 ~ T8은 그 가운데 여덟 개다. 정답 라벨은 일절 보지 않고 좌표 위치만 본다.' },
    { t: '대표점 4개를 찍는다 (k-means++)', d: '토큰들 가운데 서로 멀찍이 떨어진 넷을 골라 대표점 A, B, C, D(크고 밝은 점)의 첫 자리로 삼는다.' },
    { t: '가장 가까운 대표점에 배정한다', d: '모든 토큰이 4개 대표점 가운데 거리가 제일 가까운 곳으로 모인다.' },
    { t: '대표점을 평균 위치로 옮긴다', d: '대표점마다 자기에게 모인 토큰들의 좌표 평균으로 자리를 옮긴다.' },
    { t: '되풀이하다가, 더 움직이지 않으면 끝까지 고정한다', d: '대표점이 옮겨 가면 가장 가까운 대표점이 바뀌는 토큰이 생기고, 그러면 평균도 다시 달라진다. 그래서 배정과 이동을 되풀이한다(실제 40회). 대표점이 더 움직이지 않으면 멈춘다. 이 4개가 중심점이고, 선으로 나뉜 영역이 구역 A ~ D다. T1, T2는 A, T3, T4는 B, T5, T6은 C, T7, T8은 D에 모인다.' },
  ], (i) => P.step(i));
  el.append(sp, P.el, P.status, P.legend);
  sp.go(0);
};

W.roster = (el) => {
  const cap = capEl(), ranks = ['N1 &gt; N3 &gt; N5 &gt; N7 &gt; N2 &gt; N8', 'N1 &gt; N2 &gt; N5 &gt; N7 &gt; N3', 'N4 &gt; N6 &gt; N8 &gt; N3', 'N4 &gt; N6 &gt; N8'];
  let view = 'num0', sel = 0;
  const paint = () => {
    R.select(sel);
    R.set({ ord: 0, blocks: false, view });
    cap.innerHTML = `<b>중심점 ${R48.cen[sel]} (${R48.tok[sel]}이 모임)</b>출석부 순위 ${ranks[sel]}. 앞에서 <b>K = ${R48.on0[sel].length}</b>개만 켠다. 표의 다른 줄을 누르면 그 중심점을 본다.`;
  };
  const R = RosterW((r) => { sel = r; paint(); });
  const sg = seg([{ k: 'num0', t: '기여도 숫자' }, { k: 'onoff0', t: '켬(1) / 끔(0)' }], (k) => { view = k; paint(); }, '출석부 표 보기');
  el.append(h('div', { class: 'w-h' }, h('span', { class: 'w-t', text: '초기 출석부 표 (4 중심점 × 8 뉴런)' }), sg), R.el, cap,
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-one' }), '켜는 뉴런 (출석부 앞 K개)'), h('span', null, h('i', { class: 'i-off' }), '끄는 뉴런')));
  sg.set(0);
};

W.family = (el) => {
  const D = R48, sb = h('div', { class: 'sb' }), cap = capEl();
  const rows = D.cen.map((c, r) => h('div', { class: 'pf-row' }, h('div', { class: 'pf-l', text: '중심점 ' + c }),
    h('div', { class: 'pf' }, D.v0[r].map((v) => h('div', { class: 'pf-c' }, h('i', { style: { height: Math.max(2, v / 40 * 40) + 'px' } }), h('span', { text: v }))))));
  const head = h('div', { class: 'pf-row' }, h('div', { class: 'pf-l' }), h('div', { class: 'pf' }, N8.map(n => h('div', { class: 'pf-c hd' }, h('span', { text: 'N' + n })))));
  const verdict = { '0,1': '매우 가까움 → 가족 1', '2,3': '거의 일치 → 가족 2', '0,2': '매우 멂 (다른 가족)', '1,3': '매우 멂 (다른 가족)' };
  const sg = seg([{ k: '0,1', t: 'A ↔ B' }, { k: '2,3', t: 'C ↔ D' }, { k: '0,2', t: 'A ↔ C' }, { k: '1,3', t: 'B ↔ D' }], (k) => {
    const [a, b] = k.split(',').map(Number), A = D.v0[a], B = D.v0[b];
    const dot = A.reduce((s2, v, i) => s2 + v * B[i], 0), na = Math.hypot(...A), nb = Math.hypot(...B), cos = dot / (na * nb);
    rows.forEach((e, r) => e.classList.toggle('dim', r !== a && r !== b));
    sb.replaceChildren(stat('내적', dot.toLocaleString('en-US')), stat('코사인 유사도 (1이 최대)', cos.toFixed(2), cos > 0.5 ? 'good' : 'bad'), stat('코사인 거리 (1 − 유사도)', (1 - cos).toFixed(2)), stat('판정', `<span style="font-size:13.5px;font-family:var(--sans)">${verdict[k]}</span>`));
    cap.innerHTML = cos > 0.5 ? `<b>중심점 ${D.cen[a]}와 ${D.cen[b]}</b>두 가로줄의 힘이 같은 뉴런들에 몰려 있어 방향이 아주 닮았다. 같은 가족이다.` : `<b>중심점 ${D.cen[a]}와 ${D.cen[b]}</b>힘이 몰린 뉴런이 정반대다(한쪽은 N1, N2, N5, N7, 다른 쪽은 N4, N6, N8). 다른 가족이다.`;
  }, '견줄 중심점 쌍');
  el.append(h('div', { class: 'w-h' }, h('span', { class: 'w-t', text: '견줄 쌍' }), sg), h('div', { class: 'gx' }, h('div', { class: 'pfs' }, head, rows)), sb, cap);
  sg.set(0);
};

W.blocks = (el) => {
  const cap = capEl(), R = RosterW();
  const sg = seg([{ k: 0, t: '원래 번호 순서 (N1 ~ N8)' }, { k: 1, t: '열 재배치 후' }], (k) => {
    R.set({ ord: k, blocks: true, fam: true, view: 'onoff0', hot: k ? 2 : null });
    cap.innerHTML = k
      ? '<b>열 재배치 후 (사후학습 전 상태)</b>세로줄 무늬가 비슷한 뉴런끼리 옆자리로 오도록 열 순서를 정했다. N1, N2, N5, N7(A, B만 쓰는 무늬 [1, 1, 0, 0])은 앞쪽에, N4, N6(C, D만 쓰는 무늬 [0, 0, 1, 1])은 뒤쪽에, 양쪽에 어중간하게 걸친 N3, N8은 가운데에 둔다. A와 B는 블록 1·2를 꽉 채워 공유하지만 <b class="c-waste">블록 3 (N3, N8)</b> 때문에 블록을 하나 더 열어야 하고 빈칸(0)도 생긴다. 사후학습은 이 어중간한 블록 3을 정리하러 들어간다.'
      : '<b>원래 번호 순서 그대로 2열씩 블록 경계를 그으면</b>1과 0이 모든 블록에 섞여 있다. 중심점 B를 계산할 때도 블록 4개를 다 열어야 하고 빈칸(0) 계산이 가득 낀다.';
  }, '열 순서');
  el.append(h('div', { class: 'w-h' }, sg), R.el, cap);
  sg.set(0);
};

W.posttrain = (el) => {
  const R = RosterW(), extra = h('div');
  const pen = () => h('div', { class: 'cards' },
    card('벌점 ①', '내 출석부 뒤쪽일수록 (중심점 A의 순위)', barList([{ l: '1등 N1', v: 40, t: '40', k: 'on' }, { l: '2등 N3', v: 30, t: '30', k: 'on' }, { l: '3등 N5', v: 20, t: '20', k: 'on' }, { l: '4등 N7', v: 15, t: '15', k: 'on' }, { l: '5등 N2', v: 10, t: '10', k: 'base' }, { l: '6등 N8', v: 4, t: '4', k: 'base' }], { max: 60 }),
      h('p', { html: '<b>N1(40) &gt; N3(30) &gt; N5(20) &gt; N7(15) &gt; N2(10) &gt; N8(4)</b>. 뒤로 갈수록 벌점이 크다. 전체 뉴런 수를 줄이기 위한 벌점이다.' })),
    card('벌점 ②', '같은 가족(A + B 합산)이 덜 쓰는 뉴런일수록', barList([{ l: 'N2 = 10 + 32', v: 42, t: '42', k: 'on' }, { l: 'N3 = 30 + 4', v: 34, t: '34', k: 'base' }], { max: 60 }), h('p', { html: '<b>N2(42) &gt; N3(34)</b>. 가족 합산으로 보면 N2가 N3보다 더 중요하다.' })),
    card('벌점 ③', '가족이 덜 쓰는 약한 블록일수록 블록 통째로', barList([{ l: '블록 1', v: 117, t: '117', k: 'on' }, { l: '블록 2', v: 75, t: '75', k: 'on' }, { l: '블록 3', v: 38, t: '38', k: 'waste' }], { max: 170 }),
      h('p', { html: '<b>블록 1(117) &gt; 블록 2(75) &gt; 블록 3(38)</b>. 가족 1(A + B)의 블록 힘 합계다. 여러 블록에 흩어 쓰지 말고 한 블록 안에 뭉쳐 쓰라는 압력도 함께 건다. N2는 1등 블록(블록 1)에 있어 보호받고, N3과 N8은 꼴찌 블록(블록 3)에 묶여 강한 벌점을 받는다.' })));
  const sp = stepper([
    { t: '같은 가족 A와 B의 출석부를 나란히 본다', v: { view: 'num0x', hi: [2, 3], hiRows: [0, 1] }, d: 'A: 1등 N1(40) &gt; 2등 N3(30) &gt; 3등 N5(20) &gt; 4등 N7(15) &gt; 5등 N2(10) &gt; 6등 N8(4). B: 1등 N1(35) &gt; 2등 N2(32) &gt; 3등 N5(28) &gt; 4등 N7(12) &gt; 5등 N3(4). B에서는 <b>N2(32)가 2등 핵심 뉴런</b>인데 A 혼자 순위에서는 N2(10)가 5등으로 밀려 있다. A 혼자 순위만 보고 뒤쪽에 벌점을 주면, A는 N3(30)만 살리고 N2(10)를 죽여 같은 가족 B와 어긋난다.' },
    { t: '그래서 벌점 세 가지를 함께 합친다', v: { view: 'num0x', hot: 2 }, pen: true, d: '내 순위, 가족 합산, 블록 합계를 함께 본다. 표에서 붉게 표시된 블록 3이 가장 큰 벌점을 받는다.' },
    { t: '사후학습 1바퀴 직후 (K는 아직 자르기 전)', v: { view: 'num1', hot: 2 }, d: '교사 모델 출력은 유지한 채, 벌점이 큰 블록 3의 일이 블록 1·2로 옮겨 갔다. A에서 N3(30 → 3)과 N8(4 → 0.4)이 하던 일이 <b>N1(40 → 44), N2(10 → 30), N5(20 → 26)</b>으로 옮겨 갔고, B(4 → 0.5)와 C(6 → 0.6)에서도 N3의 기여도가 사실상 0 가까이 빠졌다.' },
    { t: 'K 조이기 (학습 없음, 허용 오차 ε 기준)', v: { view: 'onoff1' }, d: '방금 얻은 새 기여도 표로 출석부를 다시 줄 세우고 실제로 꼬리를 자른다. N3, N8의 기여도가 거의 0이 되었으므로 <b>A는 6 → 4, B는 5 → 4, C는 4 → 3</b>으로 줄어든다. N3은 모든 중심점에서 꺼졌다.' },
    { t: '반복 (실제 9바퀴)', v: { view: 'onoff1' }, d: '다음 바퀴에서는 줄어든 새 표에 맞춰 열 순서를 다시 정렬하고, 사후학습(일 몰아주기) → K 조이기(ε 기준)를 더 줄어들지 않을 때까지 되풀이한다.' },
  ], (i, st) => {
    R.set(Object.assign({ ord: 1, blocks: true, fam: true }, st.v));
    extra.replaceChildren(st.pen ? pen() : '');
  });
  el.append(R.el, h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-one' }), '켠 뉴런의 기여도'), h('span', null, h('i', { class: 'i-ours' }), '일이 늘어난 뉴런'), h('span', null, h('i', { class: 'i-z' }), '일이 빠진 뉴런 · 벌점이 큰 블록'), h('span', null, h('i', { class: 'i-off' }), '꺼 둔 뉴런 (—)')), extra);
  el.prepend(sp);
  sp.go(0);
};

/* ───────── 02 단일 요청 ───────── */
W.assign = (el) => {
  const list = h('div', { class: 'tl' }), cap = capEl();
  const f = (a, b) => `(${a.toFixed(1)}${b < 0 ? '+' : '−'}${Math.abs(b).toFixed(1)})²`;
  const P = PlaneW('assign', (o) => {
    list.replaceChildren(...PL.cen.map((c, i) => h('div', { class: 'ti' + (i === o.win ? ' cur' : '') }, h('b', { text: PL.name[i] }),
      h('div', { html: `중심점 (${c[0].toFixed(1)}, ${c[1].toFixed(1)}) · ${f(o.p[0], c[0])} + ${f(o.p[1], c[1])} = <b>${o.d[i].toFixed(2)}</b>` + (i === o.win ? '<span class="tz">가장 가까움 → 당첨</span>' : '') }))));
    cap.innerHTML = `<b>새 토큰 T_new의 좌표 (${o.p[0].toFixed(1)}, ${o.p[1].toFixed(1)})</b>거리 제곱이 가장 작은 <b>중심점 ${PL.name[o.win]}</b>의 방으로 배정된다. 주황색 점을 끌거나 평면의 다른 자리를 누르면(또는 방향키) 배정이 바뀐다.`;
  });
  el.append(h('div', { class: 'gw half' }, h('div', null, P.el, P.legend), h('div', null, h('div', { class: 'w-t', style: { marginBottom: '8px' }, text: '고정된 중심점 4개와의 거리 제곱' }), list)), cap);
};

W.skip3 = (el) => {
  const on = [1, 2, 5, 7], sb = h('div', { class: 'sb' }), cap = capEl();
  const rows = N8.map(n => { const a = h('div', { class: 'st' }), b = h('div', { class: 'st' }); return { n, a, b, el: h('div', { class: 'r8' }, h('div', { class: 'nm', text: 'N' + n }), a, b) }; });
  const M = {
    base: ['8 / 8 (100%)', '8개', '1.7705 ms (1.00×)', '<b>베이스 모델 (다 켬)</b>N1 ~ N8 8개 전부를 DRAM에서 읽고 8개 전부 곱한다.'],
    mask: ['8 / 8 (100%)', '8개 계산 후 4개 버림', '1.7800 ms (0.99×)', '<b>단순 0-Mask 대조군</b>8개 전부 읽고 전부 계산한 뒤 결과에만 × 0을 한다. 읽는 양이 그대로라 효과가 0이다.'],
    ours: ['4 / 8', '4개', '0.2824 ms (6.27×)', '<b>우리 (Roster)</b>N1, N2, N5, N7 4개만 골라 읽고 4개만 계산한다. 꺼진 4개는 W_gate · W_up · W_down을 전부 건너뛴다.'],
  };
  const sg = seg([{ k: 'base', t: '베이스 (다 켬)' }, { k: 'mask', t: '단순 0-Mask' }, { k: 'ours', t: '우리 (Roster)' }], (k) => {
    rows.forEach(r => {
      const isOn = on.includes(r.n), read = k !== 'ours' || isOn;
      r.a.className = 'st' + (read ? ' read' : ''); r.a.textContent = read ? '읽음' : '읽지 않음';
      if (k === 'mask' && !isOn) { r.b.className = 'st wst'; r.b.textContent = '계산 후 × 0 (버림)'; }
      else { r.b.className = 'st' + (read ? ' use' : ''); r.b.textContent = read ? '계산' : '계산 안 함'; }
    });
    sb.replaceChildren(stat('DRAM에서 읽는 뉴런 줄', M[k][0]), stat('실제 곱셈하는 뉴런', M[k][1]), stat('RTX 4060 실측 (d = 4,096)', M[k][2], k === 'ours' ? 'good' : ''));
    cap.innerHTML = M[k][3];
  }, '방식');
  el.append(h('div', { class: 'w-h' }, sg), h('div', { class: 'gw two' },
    h('div', { class: 'rows8 tight' }, h('div', { class: 'r8 r8h' }, h('div'), h('div', { text: 'DRAM에서 읽기' }), h('div', { text: '곱셈' })), rows.map(r => r.el)),
    h('div', { class: 'side' }, sb, cap)));
  sg.set(0);
};

W.ffn4 = (el) => {
  const on = [1, 2, 5, 7];
  const rows = N8.map(n => { const a = h('div', { class: 'st' }), b = h('div', { class: 'st' }); return { n, a, b, el: h('div', { class: 'r8' }, h('div', { class: 'nm', text: 'N' + n }), a, b) }; });
  const sp = stepper([
    { t: '중심점 A의 출석부를 편다', d: '켤 뉴런은 N1, N2, N5, N7이고, 나머지 4줄은 읽지도 계산하지도 않는다.' },
    { t: '1단계 (gate · up)', d: '켠 뉴런 4개의 W_gate, W_up 줄만 읽어 활성값 <b>a_compact = [a1, a2, a5, a7]</b> 4개 숫자만 빽빽하게 만들어 둔다.' },
    { t: '2단계 (down)', d: '방금 만든 숫자 4개에 대응하는 W_down 4줄만 읽어 <b>a1·w1 + a2·w2 + a5·w5 + a7·w7</b>을 구하고, 원래 잔차 h에 더해 다음 층으로 넘긴다.' },
  ], (i) => rows.forEach(r => {
    const isOn = on.includes(r.n);
    r.a.className = 'st' + (isOn && i >= 1 ? ' read' : ''); r.b.className = 'st' + (isOn && i >= 2 ? ' use' : '');
    r.a.textContent = isOn ? (i >= 1 ? `W_gate[${r.n}], W_up[${r.n}] 읽어 a${r.n} 계산` : 'ON · 대기') : '읽지 않음 · 계산 안 함';
    r.b.textContent = isOn ? (i >= 2 ? `W_down[:, ${r.n}] 읽어 a${r.n}·w${r.n} 더함` : (i >= 1 ? '대기' : 'ON · 대기')) : '읽지 않음 · 계산 안 함';
  }));
  el.append(h('div', { class: 'rows8 mid' }, h('div', { class: 'r8 r8h' }, h('div'), h('div', { text: '1단계 (W_gate, W_up 읽기 → 활성값 a)' }), h('div', { text: '2단계 (W_down 열 읽기 → 출력 합산)' })), rows.map(r => r.el)));
  el.prepend(sp);
  sp.go(0);
};

W.membar = (el) => {
  el.append(h('div', { class: 'w-t', text: '가중치 크기와 L2 캐시 (d = 4,096, 4층 합)' }),
    barList([{ l: '4층 가중치 (DRAM에 있음)', v: 384, t: '384 MB', k: 'base' }, { l: 'GPU의 L2 캐시', v: 32, t: '32 MB', k: 'on', s: '가중치가 12배 크다' }], { max: 400 }),
    h('div', { class: 'sb' }, stat('읽는 가중치', '15%'), stat('이론 가속', '1 / 0.15 = 6.67×'), stat('RTX 4060 실측', '6.27× (이론의 94%)', 'good')));
};

W.layers1 = (el) => {
  const D = {
    1024: { base: 0.1164, ours: [0.0244, 0.0274, 0.0199, 0.0033], x: ['4.77', '4.24', '5.85', '34.88'], tot: ['0.4656', '0.0751', '6.20'] },
    4096: { base: 0.4426, ours: [0.1060, 0.1131, 0.0577, 0.0056], x: ['4.18', '3.91', '7.67', '78.72'], tot: ['1.7705', '0.2824', '6.27'] },
  }, K = ['22.51%', '24.12%', '12.60%', '0.78%'], box = h('div'), note = h('span', { class: 'w-t' });
  const sg = seg([{ k: 4096, t: 'd = 4,096 (대형 LLM 층 폭)' }, { k: 1024, t: 'd = 1,024 (실제 학습 가중치)' }], (k) => {
    // 막대는 전부 실제 시간(ms) 한 눈금이다. 우리 각 층 막대를 이으면 맨 아래 우리 합계 막대가 된다
    const d = D[k], tot = Number(d.tot[0]), rows = [{ l: '베이스 · 4개 층 합계', v: tot, t: d.tot[0] + ' ms', s: '1.00×', k: 'base' }];
    d.ours.forEach((o, i) => rows.push({ l: `우리 · ${i + 1}층 (활성 ${K[i]})`, v: o, t: o.toFixed(4) + ' ms', s: `베이스 단일 층 대비 ${d.x[i]}×`, k: 'ours' }));
    rows.push({ l: '우리 · 4개 층 합계 (활성 15.0%)', v: Number(d.tot[1]), t: d.tot[1] + ' ms', s: `베이스 4개 층 합계 대비 ${d.tot[2]}×`, k: 'ours' });
    note.textContent = `막대 길이는 실제 실행 시간이다 (짧을수록 빠름). 층별 배수는 베이스 단일 층(${d.base.toFixed(4)} ms) 대비다.`;
    box.replaceChildren(barList(rows, { max: tot, room: 56 }));
  }, '층 폭');
  el.append(h('div', { class: 'w-h' }, sg, note), box,
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-base' }), '베이스 (다 켬)'), h('span', null, h('i', { class: 'i-ours' }), '우리')));
  sg.set(0);
};

/* ───────── 03 배치 타일링 ───────── */
W.tiles2 = (el) => {
  const one = (title, rows, M, sub) => {
    const G = GridW({ rows, cols: ['N1', 'N2'], M });
    G.show({ tiles: [{ r: [0, 1], c: [0, 1] }], plain: true });
    return card(null, title, G.el, h('p', { html: sub }));
  };
  el.append(h('div', { class: 'cards' },
    one('예시 ① 꽉 찬 2×2 틀 · {T1, T2} × {N1, N2}', ['T1', 'T2'], [[1, 1], [1, 1]], '유효 4칸 / ×0이 <b>0칸</b>, 효율 <b>100%</b>'),
    one('예시 ② 빈칸 낀 2×2 틀 · {T3, T4} × {N1, N2}', ['T3', 'T4'], [[1, 1], [0, 0]], '유효 2칸 / ×0이 <b>2칸</b>, 효율 <b>50%</b> (강제 헛계산)')),
  h('div', { class: 'cap', html: '<b>두 틀의 실행 시간은 같다</b>두 틀 모두 똑같은 2×2 틀 1개라 GPU 실행 시간은 완전히 같다. 그런데 예시 ②처럼 0(필요 없는 뉴런)이 끼어 있으면 강제로 계산하고 버리는 헛계산(×0)이 된다.' }));
};

W.grid68 = (el) => {
  const G = GridW(Object.assign({}, G68, { blocks: true }));
  const info = [['A', 'K = 4', '{N1, N2, N3, N5}', '3명 도착 (T1, T2, T3)'], ['B', 'K = 4', '{N3, N4, N5, N7}', '1명 도착 (T4)'], ['C', 'K = 4', '{N4, N6, N7, N8}', '1명 도착 (T5)'], ['D', 'K = 3', '{N1, N2, N8}', '1명 도착 (T6)']];
  const dim = (c) => G.cell.forEach((r, i) => r.forEach(e => e.classList.toggle('dim', c != null && G68.sub[i] !== c)));
  const list = h('div', { class: 'tl' }, info.map(x => {
    const it = h('div', { class: 'ti', tabindex: '0' }, h('b', { text: x[0] }), h('div', { html: `중심점 ${x[0]} (${x[1]}): <b>${x[2]}</b> 켬<span class="tn">${x[3]}</span>` }));
    it.addEventListener('mouseenter', () => dim(x[0])); it.addEventListener('mouseleave', () => dim(null));
    it.addEventListener('focus', () => dim(x[0])); it.addEventListener('blur', () => dim(null));
    return it;
  }));
  el.append(h('div', { class: 'gw two' }, G.el, list),
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-one' }), '1 = 그 토큰이 켜야 하는 뉴런'), h('span', null, h('i', { class: 'i-off' }), '0 = 필요 없는 뉴런'), h('span', null, '오른쪽 목록에 마우스를 올리면 그 중심점의 토큰 줄만 남는다')));
};

W.prior = (el) => tileUI(el, Object.assign({}, G68, { blocks: true }), [
  { k: 'union', t: '① 배치 합집합', tiles: TS.union, intro: '<b>기존 방식 1 · 배치 합집합 (Polar Sparsity 등)</b>"배치 안에서 단 1명이라도 쓰는 뉴런을 전부 합쳐서, 6명 모두에게 통째로 행렬곱하자"는 방식이다. T1 ~ T6이 쓰는 뉴런을 합집합하면 N1 ~ N8 8개 전부(100%)가 된다. 6명 × 8뉴런 = 48칸을 전부 계산해 12틀, ×0이 25칸(52.1%)이다. 다 켠 원본 모델과 똑같아져 <b>가속이 1.0×로 무너진다.</b>' },
  {
    k: 'bsmr', t: '② 고정 2행 패널 (BSMR · Fused3S · TC-GNN)', tiles: TS.bsmr, steps: true,
    introT: '토큰을 고정 행 패널로 먼저 묶는다', intro: '전역 1D 정렬보다 진화한 최신 텐서 코어 SDDMM 방식이다. 토큰을 2명씩 고정된 행 패널로 먼저 묶은 뒤, 패널 안에서 비어 있지 않은 열만 골라 밀도순으로 압축해 2×2 틀을 씌운다(패널 안에서 1 0 열끼리 합치는 열 압축은 할 수 있다). 15가지 패널 조합 가운데 가장 좋은 {T1, T2}, {T3, T4}, {T5, T6}으로 덮어 본다.',
    labels: ['패널 1 · 틀 1', '패널 1 · 틀 2', '패널 2 · 틀 3', '패널 2 · 틀 4', '패널 2 · 틀 5', '패널 3 · 틀 6', '패널 3 · 틀 7', '패널 3 · 틀 8'],
    notes: ['T1, T2가 모두 켜는 (N1, N2). 꽉 찬다.', '패널 안에서 1 0 열끼리 압축한 (N3, N5). 꽉 찬다. 패널 1은 2틀, ×0이 0칸.', 'T3, T4가 모두 켜는 (N3, N5). 꽉 찬다.', '(N1, N2)는 T3만 쓰고 T4는 안 쓴다. ×0이 2칸.', '(N4, N7)은 T4만 쓰고 T3은 안 쓴다. ×0이 2칸. 패널 2는 3틀, ×0이 4칸.', '(N8, N1)에서 T5는 N1을 안 쓴다. ×0이 1칸.', '(N2, N4)에서 T5는 N4만, T6은 N2만 쓴다. ×0이 2칸.', '(N6, N7)은 T5만 쓴다. ×0이 2칸. 패널 3은 3틀, ×0이 5칸.'],
    outroT: '8틀이 수학적 한계', outro: '다른 2행 패널 조합을 전부 따져 봐도 8틀(32칸, ×0이 9칸)이 최선이다. <b>치명적 제약: 토큰 하나는 끝까지 단 하나의 고정 행 패널에만 속할 수 있고, 열이 바뀌어도 행 짝을 못 바꾼다.</b>',
  },
], { label: '기존 방식' });

W.ours6 = (el) => tileUI(el, Object.assign({}, G68, { blocks: true }), [{
  k: 'ours', tiles: TS.ours, steps: true, introT: '같은 6 × 8 표에서 시작한다', intro: '1 0 · 0 1 열끼리 다시 합치면서, 틀마다 행 짝도 자유롭게 바꿔 2×2 틀을 만든다.',
  notes: ['같은 중심점 A끼리 곧바로 꽉 채운다.', '<b>T3이 여기서 T6과 행 짝을 맺는다.</b>', '블록 2의 N3과 블록 3의 N5, <b>1 0 조각끼리 열을 재조합</b>한다.', '<b>T3이 이번에는 T4로 행 짝을 바꾼다.</b> (N3, N5) 열 재조합.', '<b>T4가 T5로 행 짝을 바꾼다.</b> (N4, N7) 열 재조합.', '마지막 자투리 마감. 현실적으로 남는 소수의 ×0이 1칸.'],
  outroT: '6틀, ×0은 1칸', outro: '1이 23칸인 표를 6틀(24칸)로 덮었고 ×0은 1칸뿐이다. 오른쪽 목록의 틀에 마우스를 올리면 표에서 그 틀의 칸만 남는다.',
}]);

W.multi = (el) => tileUI(el, Object.assign({}, G68, { blocks: true }), [
  { k: 'big', t: '큰 틀 (4×2) 1종만', tiles: TS.big, intro: '<b>큰 틀(4×2) 1종만 강제</b>4명짜리 덩어리 2개는 꽉 차지만, 2명뿐인 (N4, N7)과 (N6, N8)에도 4×2를 강제로 씌운다. 4틀, 32칸, ×0이 9칸(28.1%). 자투리 덩어리에 빈칸이 폭증한다.' },
  { k: 'small', t: '작은 틀 (2×2) 1종만', tiles: TS.ours, intro: '<b>작은 틀(2×2) 1종만 사용</b>3단계 그대로 2×2 틀 6개다. 6틀, 24칸, ×0이 1칸. 4명 덩어리까지 쪼개져 틀 수가 늘고 효율이 떨어진다.' },
  { k: 'mix', t: '우리: 섞어 쓰기 (4×2 + 2×2)', tiles: TS.multi, intro: '<b>우리 다중 크기 (4×2 + 2×2)</b>4명 덩어리 (N1, N2)와 (N3, N5)는 4×2 2개로, 2명인 (N4, N7)과 자투리 (N6, N8)만 2×2 2개로 마감한다. 4틀, 24칸, ×0이 1칸(4.2%). 고속 큰 틀과 최소 빈칸을 함께 얻는다.' },
], { label: '틀 크기 선택', init: 2 });

W.ladder = (el) => {
  const D = [['64 × 32', 64, 32, '1.00×', 76.7, '1,922틀'], ['32 × 32', 32, 32, '1.22×', 9.9, '496틀'], ['32 × 16', 32, 16, '1.52×', 11.7, '1,173틀'], ['16 × 16', 16, 16, '2.31×', 0.8, '167틀'], ['8 × 16', 8, 16, '3.90×', 0.9, '345틀']];
  el.append(h('div', { class: 'w-t', text: '틀 5종의 실제 비율 (토큰 × 뉴런)과 칸당 상대 GPU 비용' }),
    h('div', { class: 'lad' }, D.map((d, i) => h('div', { class: 'lz' }, h('i', { style: { height: d[1] * 1.5 + 'px', width: d[2] * 1.5 + 'px' } }), h('b', { text: d[0] }), h('span', { text: `${i + 1}순위 · 칸당 ${d[3]}` })))),
    h('div', { class: 'bl-t', text: '실제 모델(B = 2,048)의 연산 칸 비중' }),
    barList(D.map((d, i) => ({ l: `${i + 1}순위 ${d[0]}`, v: d[4], t: d[4] + '%', s: d[5], k: i < 2 ? 'ours' : 'base' })), { max: 100 }),
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-ours' }), '최고속 1·2순위 틀 (합계 86.6%)'), h('span', null, h('i', { class: 'i-base' }), '아래 단계 틀')));
};

W.strip = (el) => {
  const desk = h('div', { class: 'sb' });
  tileUI(el, Object.assign({}, G68, { wall: 4, zones: ['1구역 (N1 ~ N4)', '2구역 (N5 ~ N8)'] }), [{
    k: 's', tiles: TS.zone1.concat(TS.zone2), num: [0, 1, 2, 0, 1, 2], labels: ['틀 1-① (4×2)', '틀 1-② (4×2)', '틀 1-③ (2×2)', '틀 2-① (4×2)', '틀 2-② (2×2)', '틀 2-③ (2×2)'], score: false, steps: true,
    done: '<span>덮은 1</span><b>1구역 14칸 + 2구역 9칸 = 23칸</b><small>누락 0 · 중복 0</small>',
    plan: [
      { t: '① 먼저 벽을 세워 구역을 나눈다 (타일링 전)', upto: 0, d: '전체 8열을 한 판으로 두지 않고, L2에 통째로 들어가는 크기(4열씩)로 세로 벽을 쳐서 2개 구역으로 먼저 자른다.' },
      { t: '② 1구역 안에서만 타일링한다', upto: 3, d: '1구역(N1 ~ N4)을 독립된 표로 보고 틀 3개를 짠다. 벽 너머 열은 절대 섞지 않는다. N3이 벽 너머 N5와 합쳐지지 못해 ×0이 조금 늘지만(틀 1-②에 3칸), 오직 N1 ~ N4만 쓴다. 1을 14칸 전부 덮는다.' },
      { t: '② 2구역 안에서만 타일링한다', upto: 6, d: '2구역(N5 ~ N8)도 따로 짠다. 9칸을 전부 덮어 14 + 9 = 23칸, 누락과 중복이 0이다. 두 구역은 독립이라 CPU 멀티스레드로 동시에 짤 수 있다.' },
      { t: '③ 1구역을 실행한다', upto: 6, desk: 1, d: 'N1 ~ N4 가중치를 느린 DRAM에서 빠른 L2 캐시 책상 위로 딱 1번 올려 두고, 틀 1-① → 1-② → 1-③을 전부 끝낸다. L2 자리 4칸에 딱 맞으므로 중간에 밀려나지 않는다.' },
      { t: '③ 2구역을 실행한다', upto: 6, desk: 2, d: '1구역이 완전히 끝나면 책상을 비우고 N5 ~ N8 가중치를 DRAM에서 딱 1번 올려 틀 2-① → 2-② → 2-③을 끝낸다. 반대로 벽 없이 먼저 타일링해 (N3, N5)처럼 양쪽 구역이 섞이면, 책상에서 N1, N2가 밀려나 DRAM에서 같은 가중치를 몇 번이고 다시 읽어야 한다.' },
    ],
  }], {
    onStep: (i, st) => desk.replaceChildren(
      stat('L2 캐시 책상 위 (자리 4칸)', st.desk === 1 ? 'N1 · N2 · N3 · N4' : st.desk === 2 ? 'N5 · N6 · N7 · N8' : '(아직 비어 있음)', st.desk ? 'good' : ''),
      stat('DRAM에서 가중치를 올린 횟수', st.desk === 1 ? '1구역 1번' : st.desk === 2 ? '1구역 1번 + 2구역 1번' : '0번')),
  });
  el.append(desk);
};

W.scale = (el) => el.append(h('div', { class: 'w-t', text: '배치가 커질수록 채움률과 큰 틀 비중이 함께 오른다 (4층 합)' }),
  lineChart({ x: [512, 1024, 2048], xl: ['B = 512', 'B = 1,024', 'B = 2,048'], unit: '%', min: 40, max: 100, tick: 20, label: '배치 크기별 타일 채움률과 64×32 틀 비중',
    series: [{ n: '채움률', k: 'ours', v: [72.6, 82.3, 87.4] }, { n: '64×32 비중', k: 'on', v: [53.2, 62.2, 76.7] }] }),
  h('div', { class: 'sb' }, stat('앞 3층 중심점당 평균 토큰', '16 → 32 → 64명'), stat('1단계 연산 칸', '19.2 → 16.2 → 15.3%'), stat('헛계산 ×0 비중', '27.4 → 17.7 → 12.6%', 'good')));

W.bsmr10 = (el) => el.append(
  h('div', { class: 'bl-t', text: '실측 단가 비용 (풀 밀집 = 100%, 낮을수록 좋음)' }),
  barList([{ l: '풀 밀집 (Dense 64×32)', v: 100, t: '100.0%', k: 'base' }, { l: '고전 1D SpMM (64×32)', v: 43.2, t: '43.2%', k: 'prior' }, { l: 'BSMR 원 논문 (16×16 BSA)', v: 37.3, t: '37.3%', k: 'prior' }, { l: 'BSMR 최강 OLO (16×16)', v: 36.8, t: '36.8%', k: 'prior' }, { l: 'BSMR 5단계 행 패널 상한', v: 20.2, t: '20.2%', k: 'prior' }, { l: '우리 Roster-Tile', v: 17.4, t: '17.4%', k: 'ours' }], { max: 100 }),
  h('div', { class: 'bl-t', text: '쓸데없는 ×0 칸 (전체 칸 대비, 낮을수록 좋음 · 풀 밀집은 86.44%)' }),
  barList([{ l: '고전 1D SpMM (64×32)', v: 29.59, t: '29.59%', k: 'prior' }, { l: 'MatFormer (같은 예산, 고정 열)', v: 6.27, t: '6.27%', s: '필수 1의 46.5% 누락', k: 'waste' }, { l: 'BSMR 5단계 행 패널 상한', v: 5.14, t: '5.14%', k: 'prior' }, { l: 'BSMR 원 논문 (16×16 BSA)', v: 2.57, t: '2.57%', k: 'prior' }, { l: 'BSMR 최강 OLO (16×16)', v: 2.33, t: '2.33%', k: 'prior' }, { l: '우리 Roster-Tile', v: 2.01, t: '2.01%', s: '상한 대비 -60.9%', k: 'ours' }], { max: 32 }),
  h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-prior' }), '기존 방식'), h('span', null, h('i', { class: 'i-ours' }), '우리'), h('span', null, h('i', { class: 'i-z' }), '필수 1을 잃는 방식')));

/* ───────── 04 실행 ───────── */
W.arrive = (el) => {
  const box = h('div');
  const ln = (no, t, c, extra) => h('div', { class: 'ln1' + (extra || '') }, no + ' ', h('b', { text: t }), ' (' + c + ')');
  const arr = [['0번', 'T5', 'C'], ['1번', 'T1', 'A'], ['2번', 'T6', 'D'], ['3번', 'T2', 'A'], ['4번', 'T4', 'B'], ['5번', 'T3', 'A']];
  const srt = [['1번', 'T1', 'A'], ['3번', 'T2', 'A'], ['5번', 'T3', 'A'], ['4번', 'T4', 'B', ' sp'], ['0번', 'T5', 'C', ' sp'], ['2번', 'T6', 'D', ' sp']];
  const lane = (title, items) => h('div', { class: 'lane' }, h('div', { class: 'lt', text: title }), items.map(x => ln(x[0], x[1], x[2], x[3])));
  const G = GridW(Object.assign({}, G68, { wall: 4 }));
  const zname = (ts, lab) => h('div', { class: 'tl' }, ts.map((t, i) => { const nm = G.names(t); return h('div', { class: 'ti' }, G.mini(t), h('div', null, h('b', { text: lab + CIRC[i] }), ' ' + nm.r + ' × ' + nm.c)); }));
  const views = [
    () => lane('도착 순서 (원래 입력 줄 번호)', arr),
    () => h('div', { class: 'lanes' }, lane('도착 순서 (원래 입력 줄 번호)', arr), h('div', { class: 'mid', text: '→' }), lane('번호표만 닮은 순서 A → B → C → D로 정리', srt)),
    () => h('div', { class: 'cards' }, card('CPU 일꾼 · 1구역 (N1 ~ N4)', null, zname(TS.zone1, '틀 1-')), card('CPU 일꾼 · 2구역 (N5 ~ N8)', null, zname(TS.zone2, '틀 2-'))),
    () => h('div', { class: 'cards' },
      card('2단계 묶음 1', 'T1 (A) + T2 (A)', h('p', { html: '계산할 뉴런 <b>{N1, N2, N3, N5}</b><br>두 토큰이 똑같이 쓴다. 남는 칸이 없다.' })),
      card('2단계 묶음 2', 'T3 (A) + T4 (B)', h('p', { html: '계산할 뉴런 <b>{N1, N2, N3, N4, N5, N7}</b><br>둘 다 쓰는 것 N3, N5. 한쪽만 쓰는 것 N1, N2 (T3), N4, N7 (T4).' })),
      card('2단계 묶음 3', 'T5 (C) + T6 (D)', h('p', { html: '계산할 뉴런 <b>{N1, N2, N4, N6, N7, N8}</b><br>둘 다 쓰는 것 N8. 한쪽만 쓰는 것 N4, N6, N7 (T5), N1, N2 (T6).' }))),
  ];
  const sp = stepper([
    { t: '토큰이 뒤죽박죽 섞여서 도착한다', d: '실제 서버에서는 토큰이 중심점 순서대로 얌전히 들어오지 않는다. 0번 T5(C) · 1번 T1(A) · 2번 T6(D) · 3번 T2(A) · 4번 T4(B) · 5번 T3(A).' },
    { t: '① 번호표 줄 세우기', d: '무거운 벡터 데이터는 손대지 않고 <b>번호표만</b> 닮은 순서로 한 번 훑어 정리한다: [1번(T1), 3번(T2), 5번(T3)] (A) → [4번(T4)] (B) → [0번(T5)] (C) → [2번(T6)] (D). 실제로는 토큰 수만큼 딱 1번 훑는 초고속 계수 정렬이다.' },
    { t: '② 구역별 동시 타일링', d: '03의 5단계처럼 1구역(N1 ~ N4)과 2구역(N5 ~ N8)은 벽으로 나뉘어 100% 독립이다. CPU 일꾼들이 구역을 나눠 맡아 동시에 비격자 타일 목록을 뽑는다. 실제로는 <b>16스레드가 4층 × 8구역 = 32구역을 동시 처리</b>하고, CPU 벡터 연산(AVX2)으로 512열 점수를 한꺼번에 합산한다.' },
    { t: '③ 2단계 묶음이 계산할 뉴런 목록을 함께 만든다', d: '1단계 타일을 뽑는 김에, 곧바로 이어질 2단계에서 묶음마다 계산할 뉴런 목록(묶음 안에서 한 명이라도 켠 뉴런)도 그 자리에서 함께 기록해 GPU로 전송한다. 같은 중심점끼리 묶이면 두 토큰의 목록이 똑같아 남는 칸이 없고, 실제 모델에서는 대부분의 묶음이 그렇다. 1단계 계획과 2단계 계획이 한 번에 완성된다 (4층 합 1.50 ~ 2.83 ms, 최악 1층 1.12 ~ 1.81 ms).' },
  ], (i) => box.replaceChildren(views[i]()));
  el.append(sp, box);
  sp.go(0);
};

W.overlap = (el) => {
  const D = { 512: [1.50, 1.12, '1.17 ms (1.1 MB)', 1.41, 3.241, '2.3×'], 1024: [1.49, 1.36, '1.24 ms (1.8 MB)', 1.67, 6.184, '3.7×'], 2048: [2.83, 1.81, '1.14 ms (3.3 MB)', 2.09, 13.299, '6.4×'] };
  const att = h('div', { class: 'blk gpu', style: { left: '0', width: '68%' } }), ffn = h('div', { class: 'blk ffn', style: { left: 'calc(68% + 2px)', right: '0' }, text: '1층 FFN (즉시 시작)' });
  const cpu = h('div', { class: 'blk cpu', style: { left: '0', padding: '0' } }), out = h('div', { class: 'out' }), sb = h('div', { class: 'sb' });
  const sg = seg([{ k: 512, t: 'B = 512' }, { k: 1024, t: 'B = 1,024' }, { k: 2048, t: 'B = 2,048' }], (k) => {
    const d = D[k], w = 68 * d[3] / d[4];
    att.textContent = `1층 어텐션 ${d[4].toFixed(3)} ms`;
    cpu.style.width = w + '%';
    out.style.left = `calc(${w}% + 6px)`; out.textContent = `계획 + 전송 ${d[3].toFixed(2)} ms · 이미 준비 완료`;
    sb.replaceChildren(stat('최악 1층 계획 + 전송', d[3].toFixed(2) + ' ms'), stat('실제 1층 어텐션 (d = 4,096)', d[4].toFixed(3) + ' ms'), stat('어텐션 노출 지연', '0.000 ms', 'good'), stat('여유 배수', d[5] + ' 여유', 'good'));
  }, '배치 크기');
  el.append(h('div', { class: 'w-h' }, sg, h('span', { class: 'w-t', text: '어텐션 직전 입력에서 두 줄이 동시에 출발한다' })),
    h('div', { class: 'flow' }, h('div', { class: 'fl' }, h('div', { class: 'ft', text: 'GPU' }), h('div', { class: 'trk' }, att, ffn)), h('div', { class: 'fl' }, h('div', { class: 'ft', text: 'CPU → GPU' }), h('div', { class: 'trk' }, cpu, out))), sb);
  sg.set(0);
};

W.mm1 = (el) => {
  const cap = capEl();
  const Mw = MatW(1, (o) => {
    cap.innerHTML = `<b>팀 ${o.k + 1}이 혼자 계산해 혼자 적는 칸</b>${o.expr}. ` + (o.i === 1 ? 'T2는 두 틀에 걸쳐 있지만, 팀 1은 (T2, N1) · (T2, N2) 방에만, 팀 2는 (T2, N3) · (T2, N4) 방에만 적는다.' : '이 칸에 쓰는 팀은 하나뿐이다.');
  });
  el.append(Mw.el, cap, h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-cov' }), '팀 1이 맡은 틀 ①'), h('span', null, h('i', { style: { background: 'color-mix(in srgb, var(--f-on) 48%, var(--bg))', boxShadow: 'inset 0 0 0 1px var(--f-on)' } }), '팀 2가 맡은 틀 ②'), h('span', null, h('i', { class: 'i-off' }), '계산 안 하는 칸 (0)'), h('span', null, '주황 테두리 = 지금 곱하는 행과 열')));
  Mw.pick(1, 2);
};

W.mm2 = (el) => {
  const cap = capEl(), sb = h('div', { class: 'sb' });
  const Mw = MatW(2, (o) => {
    cap.innerHTML = `<b>${o.mode === 'split' ? '1단계처럼 뉴런 열을 두 팀으로 쪼갰을 때' : '한 팀이 토큰 묶음을 끝까지 맡을 때'}</b>${o.expr}`;
    sb.replaceChildren(o.mode === 'split' ? stat('같은 방에 두 팀이 쓰는 칸', '2칸 (T2의 1차원 · 2차원)', 'bad') : stat('같은 방에 두 팀이 쓰는 칸', '0칸 (atomicAdd = 0)', 'good'),
      stat('팀 나누는 기준', o.mode === 'split' ? '뉴런 열 (틀 ① / 틀 ②)' : '토큰 묶음 (T1·T2 / T3·T4)'));
  });
  const sg = seg([{ k: 'split', t: '1단계처럼 열을 쪼개면' }, { k: 'own', t: '한 팀이 묶음을 끝까지' }], (k) => Mw.setMode(k), '팀 나누는 방식');
  el.append(h('div', { class: 'w-h' }, sg), Mw.el, sb, cap);
  sg.set(0);
};

W.pack = (el) => tileUI(el, Object.assign({}, G68, { blocks: true }), [{
  k: 'p', tiles: [{ r: [0, 1], c: [0, 1, 2, 4] }, { r: [2, 3], c: [0, 1, 2, 3, 4, 6] }, { r: [4, 5], c: [0, 1, 3, 5, 6, 7] }], labels: ['묶음 1', '묶음 2', '묶음 3'], score: false, steps: true,
  introT: '정렬된 결과 표 h', intro: '1단계가 닮은 중심점 순서대로 저장해 둔 표다. 위에서부터 2명씩 3묶음으로 자른다.',
  notes: ['T1 (A) + T2 (A). 활성 열 {N1, N2, N3, N5}만 뽑아 <b>8열 → 4열</b>로 압축한다. T1: [a1, a2, a3, a5], T2: [a1, a2, a3, a5]. 실제 계산 열 4 / 8 (50%), 빈칸 0.', 'T3 (A) + T4 (B, 같은 가족). 활성 열 {N1, N2, N3, N4, N5, N7}로 <b>8열 → 6열</b>. T3: [a1, a2, a3, 0, a5, 0], T4: [0, 0, a3, a4, a5, a7]. 실제 계산 열 6 / 8 (75%), 안 쓰는 N6, N8은 제외한다.', 'T5 (C) + T6 (D, 이웃). 활성 열 {N1, N2, N4, N6, N7, N8}로 <b>8열 → 6열</b>. T5: [0, 0, a4, a6, a7, a8], T6: [a1, a2, 0, 0, 0, a8]. 실제 계산 열 6 / 8 (75%), 안 쓰는 N3, N5는 제외한다.'],
  outroT: '묶음마다 한 팀이 전담한다', outro: '묶음 3개를 GPU 작업 팀 3개가 하나씩 맡아, 자기 묶음의 활성 열을 끝까지 다 더한 뒤 각 토큰의 출력에 1번만 적는다.',
}]);

W.final = (el) => {
  const F = {
    4096: [['B = 512', '9.580', '1.02', '3.113', 3.08], ['B = 1,024', '19.071', '1.02', '4.658', 4.09], ['B = 2,048', '38.243', '1.03', '8.373', 4.57]],
    1024: [['B = 512', '2.476', '1.02', '0.825', 3.00], ['B = 1,024', '4.820', '1.02', '1.260', 3.83], ['B = 2,048', '9.873', '1.02', '2.285', 4.32]],
  }, box = h('div');
  const sg = seg([{ k: 4096, t: 'd = 4,096 (대형 LLM 층 폭)' }, { k: 1024, t: 'd = 1,024 (실제 학습 가중치)' }], (k) => {
    box.replaceChildren(...F[k].flatMap(r => [h('div', { class: 'bl-t', text: r[0] + ' · 4층 합' }), barList([
      { l: '베이스 (cuBLAS, 다 켬)', v: 1, t: '1.00×', s: r[1] + ' ms', k: 'base' }, { l: '단순 0-Mask 대조군', v: Number(r[2]), t: r[2] + '×', k: 'base' },
      { l: k === 4096 ? '우리 · 어텐션 중첩 E2E' : '우리 · GPU FFN', v: r[4], t: r[4].toFixed(2) + '×', s: r[3] + ' ms', k: 'ours' }], { max: 5, room: 70 })]));
  }, '층 폭');
  el.append(h('div', { class: 'w-h' }, sg, h('span', { class: 'w-t', text: '가속 배수 (베이스 = 1.00×, 길수록 빠름)' })), box,
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-base' }), '베이스 · 대조군'), h('span', null, h('i', { class: 'i-ours' }), '우리 최종')));
  sg.set(0);
};

/* ───────── 05 새로운 점 ───────── */
W.duel = (el) => tileUI(el, Object.assign({}, G68, { blocks: true }), [
  { k: 'seq', t: '순서대로 끼우기', tiles: TS.union, intro: '<b>기준 · 순서대로 끼우기 (2×2)</b>도착한 순서와 번호 순서 그대로 2×2 틀을 깐다. 12틀, 48칸(100%), ×0이 25칸(52.1%). 빈칸이 절반 이상이다.' },
  { k: 'mat', t: 'MatFormer (앞 4열 고정)', fixedCols: [0, 1, 2, 3], fixedNote: '모든 토큰이 똑같은 앞 4열(N1 ~ N4)만 쓴다.', intro: '<b>기존 ① 정적 앞열 고정 (MatFormer)</b>모든 토큰에게 똑같은 앞 4열만 잘라 쓴다. 24칸만 계산해 가벼워 보이지만, N5 ~ N8에 있는 필수 1 <b class="c-waste">9칸(39.1%)이 통째로 잘려 나간다</b>(특히 T5는 4개 중 3개가 누락).' },
  { k: 'bsmr', t: 'BSMR 최선 (고정 2행 패널)', tiles: TS.bsmr, intro: '<b>기존 ② 최신 텐서 코어 SDDMM (BSMR · Fused3S · TC-GNN)</b>고전 1D 정렬(1-SA, BCSR)보다 진화해 행을 2행짜리 고정 행 패널(실제 16행 창)로 나누고 패널마다 비어 있지 않은 열을 독립 압축한다. BSMR(2026)은 저밀도 꼬리 틀을 CUDA 코어로 우회하기도 한다. <b>남는 벽:</b> 토큰 하나는 끝까지 단 하나의 고정 패널에만 묶인다. 앞쪽 열에서 T3을 T4와 묶었으면, 뒤쪽 열에서 T3이 T6과 완벽히 겹쳐도 행 짝을 바꿀 수 없다. 15가지 패널 조합을 다 따져도 8틀(32칸, ×0이 9칸)이 수학적 한계다.' },
  { k: 'ours', t: '우리 비격자 (2×2)', tiles: TS.ours, intro: '<b>우리 ③ 비격자 자유 조합 (Roster-Tile)</b>고정 행 패널 경계를 깨고 틀마다 행과 열을 둘 다 자유롭게 새로 짝짓는다. 같은 T3이라도 (N1, N2)에서는 {T3, T6}으로, (N3, N5)에서는 {T3, T4}로 짝을 바꾼다. 기존 전역 1D와 고정 행 패널 방식의 모든 해를 완전한 상위 집합으로 포함하며, 같은 표를 6틀(24칸, ×0이 1칸)로 줄인다.' },
  { k: 'multi', t: '우리 비격자 + 다중 크기', tiles: TS.multi, intro: '<b>우리 ④ 비격자 + 다중 크기 (4×2 + 2×2)</b>큰 덩어리는 고속 큰 틀로, 자투리만 작은 틀로 덮는다. 틀 수가 4틀로 줄고 ×0은 그대로 1칸이다.' },
], { label: '방식', init: 3 });

W.starbuf = (el) => {
  const G = GridW(Object.assign({}, G68, { blocks: true })), cap = capEl();
  const sg = seg([{ k: 'free', t: '이미 열린 틀 안의 1* → 공짜로 켠다' }, { k: 'lone', t: '혼자 새 틀을 열어야 하는 1* → 자동으로 끈다' }], (k) => {
    if (k === 'free') {
      G.show({ tiles: TS.ours, mark: { '5,5': ['star', '1*'] } }, null, 5);
      cap.innerHTML = '<b>이미 열린 틀 안에 든 1* → 공짜 ON</b>마지막 틀 {T5, T6} × (N6, N8)에서 원래 비어 있던 (T6, N6) 자리가 T6의 1*이라면, 틀은 이미 열려 있으므로 추가 비용 0으로 그냥 계산해 정확도를 챙긴다.';
    } else {
      G.show({ tiles: TS.ours, mark: { '1,7': ['star', '1*'] } }, null);
      cap.innerHTML = '<b>혼자 새 틀을 열어야 하는 외딴 1* → 자동 OFF</b>(T2, N8)처럼 필수 1 없이 1* 단 1칸 때문에 새 2×2 틀(빈칸 3개 동반)을 열어야 하면, 솔버가 그 1*만 자동으로 꺼서 새 틀 자체를 없앤다.';
    }
  }, '1*의 두 경우');
  el.append(h('div', { class: 'w-h' }, sg), G.el, cap, h('div', { class: 'legend' }, h('span', null, h('i', { style: { background: 'var(--s-ours)', boxShadow: 'inset 0 0 0 1px var(--ours)' } }), '1* = 꺼도 정확도가 버티는 끝쪽 뉴런'), h('span', null, h('i', { class: 'i-cov' }), '틀에 담아 계산한 1'), h('span', null, h('i', { class: 'i-z' }), '×0')));
  sg.set(0);
};

W.paradigm = (el) => { el.append(ParadigmW().el); };

W.split3 = (el) => {
  const cols = N8.map(n => 'N' + n), need = [[1, 2, 5], [1, 2, 6], [1, 7, 8]], row = (ns) => N8.map(n => (ns.includes(n) ? 1 : 0));
  const top = GridW({ rows: ['구역 A', '구역 B', '구역 C'], cols, M: need.map(row) });
  const out = h('div'), cap = capEl();
  const one = (label, set) => { const G = GridW({ rows: [label], cols, M: [row(need[0])] }); G.show(set); return G.el; };
  const V = {
    fixed: () => [h('div', { class: 'w-t', text: '고정 전문가 분할: 방 1 = {N1, N2, N3, N4}, 방 2 = {N5, N6, N7, N8}' }), one('방 1만 켜면', { fixedCols: [0, 1, 2, 3] }), one('방 2까지 켜면', { tiles: [{ r: [0], c: [0, 1, 2, 3, 4, 5, 6, 7] }], plain: true })],
    cmoe: () => [h('div', { class: 'w-t', text: '공유 전문가 + 고정 분할: 공용 방 = {N1}, 나머지 N2 ~ N8은 겹침 없이 분할' }), one('구역 A', { tiles: [{ r: [0], c: [0] }], plain: true, mark: { '0,1': ['star', '?'], '0,4': ['star', '?'] } })],
    ours: () => [h('div', { class: 'w-t', text: '우리: 가중치 N1 ~ N8은 1세트, 구역마다 출석부만 따로' }), one('구역 A', { tiles: [{ r: [0], c: [0, 1, 4] }], plain: true })],
  };
  const T = {
    fixed: '<b>고정 전문가 분할 (MoEfication · LTE · D2DMoE)</b>뉴런을 겹치지 않는 방으로 쪼갠다. 구역 A({N1, N2, N5})가 들어올 때 방 1만 켜면 <b class="c-waste">N5가 잘려 정확도 손실</b>, 방 2까지 켜면 안 쓰는 N3, N4, N6, N7, N8까지 <b class="c-waste">강제 끼워팔기 낭비</b>가 난다.',
    cmoe: '<b>공유 전문가 + 고정 분할 (CMoE)</b>전체 공용 N1만 따로 빼고, 나머지 N2 ~ N8은 여전히 겹침 없이 분할한다. 전체 공용 N1은 해결돼도, A · B 둘만 같이 쓰는 <b>부분 공용 N2</b>와 경계 뉴런 N5(? 표시)에서는 여전히 잘림 또는 끼워팔기가 생긴다.',
    ours: '<b>우리 방식 (Roster)</b>가중치 N1 ~ N8은 딱 1세트만 두고, 중심점마다 중복 허용 가변 K 출석부만 따로 적는다. 전체 공용(N1)과 부분 공용(N2) 모두 방에 가두지 않고 필요한 구역이 자유롭게 중복 공유한다. <b>경계 잘림 0 · 끼워팔기 0.</b>',
  };
  const sg = seg([{ k: 'fixed', t: '고정 전문가 분할' }, { k: 'cmoe', t: '공유 전문가 + 고정 분할 (CMoE)' }, { k: 'ours', t: '우리 (Roster)' }], (k) => { out.replaceChildren(...V[k]()); cap.innerHTML = T[k]; }, '뉴런 관리 방식');
  el.append(h('div', { class: 'w-t', text: '구역마다 필요한 뉴런 (N1은 전체 공용, N2는 A · B만 쓰는 부분 공용)' }), top.el,
    h('div', { class: 'w-h', style: { marginTop: '14px' } }, h('span', { class: 'w-t', text: '구역 A가 들어올 때' }), sg), out, cap,
    h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-cov' }), '필요해서 계산'), h('span', null, h('i', { class: 'i-z' }), '안 쓰는데 강제 계산 (끼워팔기)'), h('span', null, h('i', { class: 'i-miss' }), '필요한데 잘림')));
  sg.set(0);
};

W.share4 = (el) => {
  const R = RosterW(), cap = capEl();
  const sg = seg([{ k: 'indiv', t: '개별 순위만 앞쪽 몰기' }, { k: 'onoff1', t: '우리: 가족 + 블록 공유 사후학습' }], (k) => {
    R.set({ ord: 1, blocks: true, fam: true, view: k, hot: k === 'indiv' ? 2 : null });
    cap.innerHTML = k === 'indiv' ? '<b>개별 순위만 앞쪽 몰기 (이웃·블록 경계 무시)</b>같은 가족 A · B도 어긋나(A는 N2를 끄고 N3, N8을 남긴다) <b class="c-waste">4개 블록이 전부 열린다.</b> ×0과 DRAM 읽기가 늘어난다.'
      : '<b>우리: 가족 + 블록 공유 사후학습 (개인 40% + 가족 35% + 블록 25% + Group-L2)</b>A · B가 블록 1·2를 <b>100% 공유</b>하고, N3은 <b>전면 미사용(DRAM 읽기 0)</b>이 된다.';
  }, '학습 방식');
  el.append(h('div', { class: 'w-h' }, sg), R.el, cap);
  sg.set(0);
};

W.union48 = (el) => tileUI(el, { rows: ['T1', 'T2', 'T3', 'T4'], cols: N8.map(n => 'N' + n), M: [0, 1, 2, 3].map(i => N8.map(n => (n === 2 * i + 1 || n === 2 * i + 2 ? 1 : 0))) }, [
  { k: 'u', t: '배치 합집합 방식', tiles: [{ r: [0, 1, 2, 3], c: [0, 1, 2, 3, 4, 5, 6, 7] }], plain: true, labels: ['합집합 한 덩어리'], intro: '<b>배치 합집합 방식 (Polar Sparsity의 OPT ReLU · Lee 2026)</b>4명 중 단 1명이라도 쓰는 뉴런을 전부 모은 합집합 {N1 ~ N8} 8개를 4명 모두에게 통째로 곱한다. 32칸(100%)을 계산해 가속 1.0×다. T1은 N3 ~ N8이 전혀 필요 없는데 강제 계산한다. 배치가 커지면 즉시 합집합이 100%로 꽉 차 가속이 불가능하다.' },
  { k: 'o', t: '우리: 토큰별 집합 유지', tiles: [0, 1, 2, 3].map(i => ({ r: [i], c: [2 * i, 2 * i + 1] })), plain: true, labels: ['T1의 집합', 'T2의 집합', 'T3의 집합', 'T4의 집합'], intro: '<b>우리 방식 (Roster + Roster-Tile)</b>합집합으로 뭉개지 않고 토큰별 뉴런 집합을 그대로 유지한다. 같은 중심점 결집과 비격자 5단계 틀로 1만 덮는다. 합집합이 커져도 표의 실제 밀도는 그대로 남으므로(실제 13.5%), 1·2단계 모두 희소성을 끝까지 살린다. 실제 B = 2,048에서 전체 FFN의 15.6%만 계산한다.' },
], { label: '방식' });

W.density = (el) => el.append(h('div', { class: 'w-t', text: '층별 가속 배수 (B = 2,048, d = 4,096 · 베이스 = 1.00×)' }),
  barList([{ l: '베이스 (다 켬)', v: 1, t: '1.00×', k: 'base' }, { l: '1층 · 활성 22.07% (고·중밀도)', v: 2.91, t: '2.91×', s: '8.99 → 3.09 ms', k: 'ours' }, { l: '2층 · 활성 21.67% (고·중밀도)', v: 3.14, t: '3.14×', s: '9.32 → 2.97 ms', k: 'ours' }, { l: '3층 · 활성 9.55% (중·저밀도)', v: 5.81, t: '5.81×', s: '9.37 → 1.61 ms', k: 'ours' }, { l: '4층 · 활성 0.80% (초저밀도)', v: 15.22, t: '15.22×', s: '9.24 → 0.61 ms', k: 'ours' }, { l: '4층 합계 · 활성 13.52%', v: 4.57, t: '4.57×', k: 'ours' }], { max: 16, room: 66 }));

W.synergy = (el) => {
  const bx = (no, title, items) => h('div', { class: 'bx' }, h('h4', null, h('span', { class: 'no', text: no }), title), h('ul', null, items.map(t => h('li', { html: t }))));
  const b2 = bx('2순위', '패러다임 역전 (선제 차등 명단 + 가족·32열 블록 공유 사후학습)', ['2<sup>4096</sup>가지 → 고정 32 / 256가지 출석부로 닫힘 ▸ <b>(1) 같은 중심점 토큰 = 100% 같은 행</b> (64×32 뼈대)', '중복 허용 가변 K + 가족·32열 블록 사후학습 ▸ <b>(2) ×0 비중 12.6% · 2단계 합집합 16.1% · DRAM 64.5% · 1* = 0%</b>', '고정 표(추론 전 확정) + 고유 순위 ▸ <b>(3) 13.1 ms 사전 준비 표</b> · 필요하면 끝쪽 1* 버퍼 공급']);
  const b1 = bx('1순위', '실시간 비격자 5단계 타일링', ['고정 행 패널 경계 해제 + 5단계 실측 비용 사다리 + 1.49 ~ 2.83 ms 실시간 조립 (어텐션 0.000 ms 은폐)', 'FFN 1단계(출력 희소, atomicAdd = 0) + FFN 2단계(묶음 전담 합산, atomicAdd = 0)']);
  const b3 = bx('3순위', '단일(6.27×)과 대형 서버(4.57×) SwiGLU FFN E2E 동시 실측', ['합집합이 100%로 꽉 차지 않고 토큰별 뉴런 집합 유지 · 순수 bf16 · 전 밀도 구간(0.8 ~ 22%) 가속 · 정확도 베이스 대비 99.77% (90.81%)']);
  const cap = capEl();
  const S = {
    all: [[], [], '<b>세 축이 모두 있을 때</b>위에서 아래로 성질이 이어진다. 2순위가 만든 유한한 출석부를 1순위가 받아 실시간으로 타일을 짜고, 그 결과가 3순위의 실측으로 나온다. 아래 스위치로 축을 하나씩 빼 보자.'],
    no2: [[b2], [b1, b3], '<b>2순위(선제 유한 명단) 없이 1순위만 쓰면</b>토큰마다 2<sup>4096</sup>가지 산발 패턴이 나와 <b class="c-waste">사전 준비 표 제작이 원리적으로 불가능</b>하고, 같은 행 덩어리가 없어 큰 틀(64×32)과 2단계 묶음 합집합이 모두 무너진다. 교훈: 유한 중심점 명단 없이는 실시간 비격자 타일링 자체가 성립하지 않는다.'],
    no1: [[b1], [b3], '<b>1순위(실시간 비격자 5단계 타일링) 없이 2순위만 쓰면</b>단일 요청(B = 1)은 빠르지만, 대형 서버 배치에서는 <b class="c-waste">배치 합집합(100%로 꽉 참)</b> 또는 <b class="c-waste">고정 행 패널 벽</b>(BSMR 상한, ×0이 2.5배 이상)에 막혀 가속이 크게 꺾인다. 교훈: 비격자 5단계 타일링 없이는 단일 요청의 이득이 대형 배치로 이어지지 않는다.'],
    noS: [[], [b2, b1, b3], '<b>공유 사후학습(가족·32열 블록)만 빼고 유한 명단 + 타일링만 쓰면</b>이웃 중심점끼리 쓰는 뉴런이 흩어져 타일 안 ×0, 2단계 합집합, DRAM 읽기가 모두 커지고, 자투리를 끄려다 <b class="c-waste">1* = 0% 배치 무관 동일 출력을 지키기 어렵다.</b> 교훈: 하드웨어 친화 사후학습이 1단계·2단계·DRAM·일관성 4개 벽을 동시에 뚫는다.'],
  };
  const sg = seg([{ k: 'all', t: '세 축 모두' }, { k: 'no2', t: '2순위를 뺀다' }, { k: 'no1', t: '1순위를 뺀다' }, { k: 'noS', t: '공유 사후학습만 뺀다' }], (k) => {
    [b2, b1, b3].forEach(b => { b.classList.toggle('off', S[k][0].includes(b)); b.classList.toggle('broken', S[k][1].includes(b)); });
    cap.innerHTML = S[k][2];
  }, '역검증: 축 하나 빼 보기');
  el.append(h('div', { class: 'w-h' }, h('span', { class: 'w-t', text: '역검증' }), sg), h('div', { class: 'diag' }, b2, h('div', { class: 'ar', text: '▼' }), b1, h('div', { class: 'ar', text: '▼' }), b3), cap,
    h('div', { class: 'legend' }, h('span', null, '점선 = 뺀 축'), h('span', null, h('i', { class: 'i-z' }), '그 때문에 무너지는 축')));
  sg.set(0);
};

W.bsmr10b = (el) => el.append(
  h('div', { class: 'bl-t', text: '실측 단가 비용 (풀 밀집 = 100%, 낮을수록 좋음 · B = 2,048 10회 평균)' }),
  barList([{ l: '기준: 풀 밀집 (Dense 64×32)', v: 100, t: '100.0%', k: 'base' }, { l: '고전 1D SpMM (32×32)', v: 48.7, t: '48.7%', k: 'prior' }, { l: '고전 1D SpMM (64×32)', v: 43.2, t: '43.2%', k: 'prior' }, { l: 'BSMR 원 논문 (16×16 BSA)', v: 37.3, t: '37.3%', k: 'prior' }, { l: 'BSMR 최강 OLO (16×16)', v: 36.8, t: '36.8%', k: 'prior' }, { l: 'BSMR 대형 틀 (32×32)', v: 22.5, t: '22.5%', k: 'prior' }, { l: 'BSMR 5단계 행 패널 상한', v: 20.2, t: '20.2%', k: 'prior' },
    { l: '우리 Roster-Tile 실시간 타일링', v: 17.4, t: '17.4%', s: '채움률 87.1% · 필수 1 누락 0', k: 'ours' }, { l: '(참고) 오프라인 정밀 솔버 (46초)', v: 16.9, t: '16.9%', k: 'base' }, { l: 'MatFormer (같은 예산)', v: 13.5, t: '13.5%', s: '필수 1의 46.5% 누락', k: 'waste' }], { max: 100, room: 60 }),
  h('div', { class: 'legend' }, h('span', null, h('i', { class: 'i-prior' }), '기존 방식 (필수 1 전부 보존)'), h('span', null, h('i', { class: 'i-ours' }), '우리'), h('span', null, h('i', { class: 'i-z' }), '필수 1을 잃는 방식'), h('span', null, h('i', { class: 'i-base' }), '기준 · 참고')));

/* ───────── 탭 이동 ───────── */
function go(hash, instant) {
  const m = /^#?s?(0[1-5])/.exec(hash || ''), t = m ? m[1] : '01';
  $$('.tab').forEach(e => e.classList.toggle('on', e.id === 'tab-' + t));
  $$('.tab-b').forEach(b => { const on = b.dataset.tab === t; b.classList.toggle('on', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); });
  mountAll($('#tab-' + t));
  const sec = /^#s0/.test(hash || '') ? document.getElementById(hash.slice(1)) : null;
  if (sec) sec.scrollIntoView({ block: 'start', behavior: instant || REDUCED ? 'auto' : 'smooth' });
  else window.scrollTo(0, 0);
}
window.addEventListener('hashchange', () => go(location.hash));
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  $$('.tab-b').forEach(b => b.addEventListener('click', () => { const hs = '#' + b.dataset.tab; if (location.hash === hs) go(hs); else location.hash = hs; }));
  go(location.hash, true);
});

/* 자동 검사용: 모든 탭의 모든 스위치·걸음을 눌러 보고, 화면에 나온 수치를 모아 돌려준다 */
window.__selftest = () => {
  const nums = new Set(), wide = [];
  const grab = () => (document.body.innerText.match(/\d[\d,]*(?:\.\d+)?/g) || []).forEach(x => nums.add(x.replace(/,$/, '')));
  for (const t of ['01', '02', '03', '04', '05']) {
    go('#' + t, true);
    const tab = $('#tab-' + t);
    $$('details', tab).forEach(d => { d.open = true; });
    grab();
    for (const w of $$('.w', tab)) {
      const dots = () => { for (let i = 0; i < 40; i++) { const d = $$('.dot', w)[i]; if (!d) break; d.click(); grab(); } };
      const segs = $$('.seg-b', w);
      if (!segs.length) dots();
      for (const b of segs) { b.click(); grab(); dots(); }
      $$('.rt-row, .mc.pick', w).forEach(x => { x.click(); grab(); });
      $$('.ti', w).forEach(x => { x.dispatchEvent(new MouseEvent('mouseenter')); x.dispatchEvent(new MouseEvent('mouseleave')); });
    }
    if (document.documentElement.scrollWidth > document.documentElement.clientWidth + 1) wide.push(t);
    for (const e of $$('.w, .tw, .cards, .sec', tab)) { const r = e.getBoundingClientRect(); if (r.right > document.documentElement.clientWidth + 1 && !e.closest('.tw, .gx')) wide.push(t + ':' + (e.dataset.w || e.className)); }
  }
  return { nums: Array.from(nums), wide };
};
