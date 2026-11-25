'use strict';
/* Roster v3 · k-means를 눈으로 보이기 위한 토큰 구름과, 그 위에서 실제로 돌린 k-means 한 판 (화면과 무관한 순수 계산).
   가이드 01의 예시는 토큰이 8개뿐이라 한 번에 끝난다. 되풀이하다 멈추는 모습을 보이려고 가이드의 중심점 네 개 둘레에
   지어낸 토큰을 뿌린다. 무리마다 평균이 정확히 그 중심점이 되게 맞춰서, 다 돌고 나면 가이드의 중심점 자리에 멈춘다. */
const KM_CEN = [[1.1, 1.0], [2.1, 1.4], [-1.1, -1.1], [-2.1, -0.7]];
const KM_TOK = [[1.0, 1.2], [1.2, 0.8], [2.2, 1.6], [2.0, 1.2], [-1.2, -1.0], [-1.0, -1.2], [-2.2, -0.6], [-2.0, -0.8]];
const KM_SEED = { cloud: 30, run: 20, per: 34, rad: 0.5 };   // 씨앗은 km_search.mjs로 고른다

function kmRng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

/* 중심점마다 반지름 rad 안에 per개를 뿌리고, 무리의 평균이 중심점과 같아지게 통째로 민다 */
function kmCloud(seed, per, rad) {
  const r = kmRng(seed), pts = [];
  for (const c of KM_CEN) {
    const offs = [];
    while (offs.length < per) { const u = r() * 2 - 1, v = r() * 2 - 1; if (u * u + v * v <= 1) offs.push([u * rad, v * rad]); }
    const mx = offs.reduce((a, o) => a + o[0], 0) / per, my = offs.reduce((a, o) => a + o[1], 0) / per;
    for (const o of offs) pts.push([c[0] + o[0] - mx, c[1] + o[1] - my]);
  }
  return pts;
}

/* k-means++로 첫 대표점 4개를 고르고(먼 점일수록 잘 뽑힌다), 배정 → 평균으로 이동을 더 움직이지 않을 때까지 되풀이한다.
   돌려주는 것: pick(뽑힌 점 번호), rounds([{from, asg, to, move}] 마지막 바퀴는 move = 0), letter(대표점 j가 끝나는 가이드 중심점 번호) */
function kmRun(P, seed) {
  const r = kmRng(seed), d2 = (a, b) => (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
  const near = (p, C) => { let b = 0; for (let i = 1; i < C.length; i++) if (d2(p, C[i]) < d2(p, C[b])) b = i; return b; };
  const pick = [Math.floor(r() * P.length)];
  while (pick.length < KM_CEN.length) {
    const w = P.map(p => Math.min(...pick.map(k => d2(p, P[k]))));
    let t = r() * w.reduce((a, b) => a + b, 0), k = 0;
    while (k < P.length - 1 && t > w[k]) { t -= w[k]; k++; }
    pick.push(k);
  }
  let cen = pick.map(k => P[k].slice());
  const rounds = [];
  for (let it = 0; it < 40; it++) {
    const asg = P.map(p => near(p, cen));
    const to = cen.map((c, j) => {
      let sx = 0, sy = 0, n = 0;
      P.forEach((p, i) => { if (asg[i] === j) { sx += p[0]; sy += p[1]; n++; } });
      return n ? [sx / n, sy / n] : c.slice();
    });
    const move = Math.sqrt(Math.max(...to.map((c, j) => d2(c, cen[j]))));
    rounds.push({ from: cen, asg, to, move });
    cen = to;
    if (move < 1e-9) break;
  }
  return { pick, rounds, letter: cen.map(c => near(c, KM_CEN)) };
}
