// Roster v3 HTML 실행 검사 (화면 없는 Chrome을 DevTools 프로토콜로 직접 몬다. 추가 설치 없음, Node 22 이상)
//   node VIZ/src_v3/check.mjs <html 파일> <결과 폴더> [shots.json]
// 하는 일: 데스크톱·모바일 × 어두운·밝은 테마에서 페이지의 __selftest()를 돌려 (모든 스위치와 걸음 누르기)
//          스크립트 오류, 가로 넘침, 가이드 수치 누락을 세고, shots.json에 적힌 화면을 찍는다.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const [file, outDir, shotsFile] = process.argv.slice(2);
const here = import.meta.dirname;
fs.mkdirSync(outDir, { recursive: true });
const port = 9400 + Math.floor(Math.random() * 400);
const prof = fs.mkdtempSync(path.join(os.tmpdir(), 'roster-chk-'));
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, '--no-first-run', '--disable-gpu', '--hide-scrollbars', ...(process.env.MOTION ? [] : ['--force-prefers-reduced-motion']), 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

let tgt;
for (let i = 0; i < 60 && !tgt; i++) {
  try { tgt = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json(); } catch (e) { await sleep(150); }
}
if (!tgt) { console.log('Chrome에 붙지 못함'); chrome.kill(); process.exit(2); }
const ws = new WebSocket(tgt.webSocketDebuggerUrl);
await new Promise(r => { ws.onopen = r; });
let id = 0;
const pend = new Map(), errs = [];
let loaded = null;
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id) { const f = pend.get(m.id); pend.delete(m.id); if (f) f(m); return; }
  if (m.method === 'Page.loadEventFired' && loaded) loaded();
  if (m.method === 'Runtime.exceptionThrown') errs.push('예외: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300));
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errs.push('console.error: ' + m.params.args.map(a => a.value || a.description || '').join(' ').slice(0, 300));
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error' && !/fonts\.g|favicon/.test(m.params.entry.url || '')) errs.push('log: ' + m.params.entry.text.slice(0, 200));
};
const send = (method, params = {}) => new Promise(res => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) { errs.push('평가 오류: ' + (r.result.exceptionDetails.exception?.description || '').slice(0, 300)); return null; }
  return r.result?.result?.value;
};
await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
let navs = 0;
const open = async (w, hh, theme, hash = '', scale = 1) => {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: hh, deviceScaleFactor: scale || 1, mobile: w < 600 });
  const p = new Promise(r => { loaded = r; });
  await send('Page.navigate', { url: pathToFileURL(path.resolve(file)).href + '?theme=' + theme + '&_=' + (++navs) + hash });   // 주소를 매번 다르게 해서 꼭 새로 읽게 한다 (같은 주소면 이전 상태가 남는다)
  await Promise.race([p, sleep(8000)]);
  await sleep(350);
};

let bad = 0;
const numsFile = path.join(here, 'out', 'guide_numbers.json');
const guide = fs.existsSync(numsFile) ? JSON.parse(fs.readFileSync(numsFile, 'utf8')) : null;
for (const [w, hh, theme] of [[1280, 900, 'dark'], [1280, 720, 'dark'], [1060, 800, 'dark'], [420, 900, 'dark'], [1280, 900, 'light'], [420, 900, 'light']]) {
  const before = errs.length;
  await open(w, hh, theme);
  const r = await ev('window.__selftest ? window.__selftest() : null');
  const e2 = errs.length - before, wide = r ? r.wide : ['selftest 없음'];
  bad += e2 + wide.length;
  console.log(`${w}px ${theme}: 스크립트 오류 ${e2}건, 가로 넘침 ${wide.length}건` + (wide.length ? ' → ' + wide.join(', ') : ''));
  if (guide && r && r.nums.length && w === 1280 && theme === 'dark') {
    const have = new Set(r.nums);
    for (const d of Object.keys(guide)) {
      const miss = guide[d].filter(x => !have.has(x));
      console.log(`  가이드 ${d}: 수치 ${guide[d].length}개 중 화면에 없는 것 ${miss.length}개` + (miss.length ? ' → ' + miss.join(' ') : ''));
    }
  }
}
if (shotsFile) {
  for (const sh of JSON.parse(fs.readFileSync(shotsFile, 'utf8'))) {
    await open(sh.w || 1280, sh.h || 900, sh.theme || 'dark', sh.hash || '', sh.scale || 1);
    if (sh.pre) { await ev(sh.pre); await sleep(sh.wait || 500); }
    const cap = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: !!sh.full });
    fs.writeFileSync(path.join(outDir, sh.name + '.png'), Buffer.from(cap.result.data, 'base64'));
  }
}
for (const e of errs.slice(0, 20)) console.log('  ' + e);
console.log(bad ? `문제 ${bad}건` : '실행 검사 통과');
ws.close(); chrome.kill();
try { fs.rmSync(prof, { recursive: true, force: true }); } catch (e) { /* 잠긴 파일은 둔다 */ }
process.exit(bad ? 1 : 0);
