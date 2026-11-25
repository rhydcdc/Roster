// km.js의 씨앗 고르기: 되풀이가 눈에 보이면서(여러 바퀴 움직임) 가이드의 중심점 자리에 정확히 멈추는 판을 찾는다.
//   node VIZ/src_v3/km_search.mjs [per] [rad]
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const src = fs.readFileSync(path.join(import.meta.dirname, 'km.js'), 'utf8');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src + '\nthis.api = { KM_CEN, KM_TOK, kmCloud, kmRun };', ctx);
const { KM_CEN, KM_TOK, kmCloud, kmRun } = ctx.api;
const per = Number(process.argv[2] || 34), rad = Number(process.argv[3] || 0.5);
const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const found = [];
for (let cs = 1; cs <= 60; cs++) {
  const cloud = kmCloud(cs, per, rad), P = cloud.concat(KM_TOK);
  for (let rs = 1; rs <= 400; rs++) {
    const R = kmRun(P, rs), n = R.rounds.length, last = R.rounds[n - 1];
    if (R.pick.some(k => k >= cloud.length)) continue;                               // 이름표 붙은 토큰은 첫 대표점으로 쓰지 않는다
    if (new Set(R.letter).size !== 4) continue;                                      // 대표점 넷이 중심점 넷에 하나씩
    if (last.to.some((c, j) => d(c, KM_CEN[R.letter[j]]) > 1e-9)) continue;          // 가이드의 중심점 자리에 정확히 멈춘다
    if (n < 5 || n > 7) continue;                                                    // 움직이는 바퀴 4 ~ 6번
    const mv = R.rounds.map(x => x.move);
    if (mv[0] < 0.25 || mv[1] < 0.10 || mv[2] < 0.05 || mv[3] < 0.02) continue;      // 앞의 네 바퀴는 눈에 보이게 움직인다
    let far = 9;
    for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) far = Math.min(far, d(P[R.pick[a]], P[R.pick[b]]));
    if (far < 0.8) continue;                                                         // 첫 대표점끼리 멀찍이
    found.push({ cs, rs, n, far: far.toFixed(2), mv: mv.map(x => x.toFixed(3)).join(' ') });
  }
}
found.sort((a, b) => b.n - a.n || b.far - a.far);
console.log(`per=${per} rad=${rad} 후보 ${found.length}개`);
for (const f of found.slice(0, 12)) console.log(JSON.stringify(f));
