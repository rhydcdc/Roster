"""레시피 HTML(VIZ/neuron_recipe.html)에 '묶어서 계산'(단일 요청 B=1 + 5단계 다중 크기 Roster-Tile + RTX 4060 실측) 보기를 덧붙이고,
RECIPE_VIZ.html을 다시 만든다. 기존 내용('학습 과정', '추론 과정')은 단 1바이트도 건드리지 않는다.
끼워 넣는 조각 넷은 모두 <!-- 배칭 시작 --> ... <!-- 배칭 끝 --> 안에 있다.
조각을 빼고 아키텍처 이름(NAME) 두 곳을 되돌리면 원래 파일과 바이트까지 같은지 스스로 확인한다.
데이터: runs/viz_batch.json (viz_batch_data.py).
python -X utf8 -W ignore BATCHING/CODE/viz_batch_inject.py
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
  <!-- ============ 묶어서 계산 (Roster-Tile) ============ -->
  <div class="view" id="view-batch" hidden>
  <style>
  #view-batch .step, #view-batch .split { align-items: start; }
  #view-batch .bt-wide { grid-column: 1 / -1; }
  #view-batch .bt-tbl-wrap { width: 100%; overflow-x: auto; }
  .bt-intro { padding-block: 36px 8px; border-top: 1px solid var(--line); }
  .bt-intro .no { font-family: var(--mono); font-size: 12px; color: var(--pick); letter-spacing: 0.06em; }
  .bt-intro h2 { font-size: 22px; margin-top: 8px; }
  .bt-intro p { color: var(--fg-2); margin-top: 10px; max-width: 52em; }
  .bt-intro .stats { margin-top: 18px; }
  .bt-svg svg { display: block; width: 100%; height: auto; }
  .bt-svg text { font-family: var(--mono); font-size: 11px; fill: var(--fg-3); }
  .bt-svg .lab { fill: var(--fg-2); font-size: 11.5px; }
  .bt-svg .hi { fill: var(--fg); font-size: 12px; font-weight: 500; }
  .bt-steps { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .bt-steps .where { font-size: 13px; color: var(--fg-2); flex: 1; min-width: 12em; }
  .bt-steps .where b { color: var(--pick); font-weight: 500; }
  .bt-side { display: grid; gap: 10px; font-size: 13px; color: var(--fg-2); }
  .bt-side b { color: var(--fg); font-weight: 600; }
  .bt-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  .bt-tbl th, .bt-tbl td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  .bt-tbl th { color: var(--fg-3); font-weight: 500; }
  .bt-tbl td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  .bt-tbl tr.sum td { color: var(--fg); font-weight: 600; border-top: 1px solid var(--line-2); border-bottom: 0; background: rgba(87, 213, 231, 0.05); }
  .bt-tbl tr.b1-hi td { color: var(--fg); font-weight: 600; background: rgba(157, 140, 255, 0.08); }
  .bt-tbl tr.grp td { color: var(--fg); font-weight: 600; font-size: 12px; background: var(--bg-3); border-top: 1px solid var(--line-2); border-bottom: 1px solid var(--line); padding: 7px 10px; letter-spacing: 0.01em; }
  .bt-tbl .ok { color: var(--on); font-weight: 500; }
  .bt-tbl .warn { color: var(--pick); font-weight: 500; }
  .bt-tbl .mut { color: var(--fg-3); font-size: 11.5px; }
  .bt-pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-family: var(--mono); font-size: 11.5px; font-weight: 500; background: var(--on-soft); color: var(--on); }
  .bt-pill.pick { background: var(--pick-soft); color: var(--pick); }
  .bt-pill.neg { background: rgba(157, 140, 255, 0.16); color: var(--neg); }
  .bt-kpi { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .bt-kpi .panel { gap: 4px; padding: 12px 14px; }
  .bt-kpi .k { font-size: 12px; color: var(--fg-3); }
  .bt-kpi .v { font-family: var(--mono); font-size: 18px; font-weight: 500; }
  .bt-kpi .v.on { color: var(--on); }
  .bt-kpi .v.pick { color: var(--pick); }
  .bt-kpi .v.neg { color: var(--neg); }
  .bt-b1-card { background: rgba(157, 140, 255, 0.07); border: 1px solid rgba(157, 140, 255, 0.32); border-radius: 10px; padding: 12px 14px; display: grid; gap: 8px; }
  .bt-b1-card .hdr { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 6px; font-size: 12.5px; color: var(--fg-2); }
  .bt-b1-card .hdr b { color: var(--fg); font-weight: 600; }
  .bt-strip-chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .bt-strip-chips .chip.plain { padding: 3px 9px; font-size: 12px; font-family: var(--mono); }
  .bt-prob ol { margin: 6px 0 0; padding-left: 18px; display: grid; gap: 6px; color: var(--fg-2); font-size: 13px; }
  .bt-prob ul { margin: 4px 0 0; padding-left: 16px; display: grid; gap: 4px; color: var(--fg-3); font-size: 12.5px; }
  .bt-recipe { margin: 6px 0 0; padding-left: 18px; display: grid; gap: 7px; color: var(--fg-2); font-size: 13px; }
  .bt-recipe b { color: var(--fg); font-weight: 600; }
  @media (max-width: 720px) { .bt-kpi { grid-template-columns: 1fr; } }
  </style>

  <section class="bt-intro">
    <div class="no">묶어서 계산 · 단일 요청 + 5단계 Multi-Scale Roster-Tile</div>
    <h2>단일 요청은 6.55배 직접 가속, 실제 서버는 5단계 틀 순위(64×32 → 8×16) + L2 캐싱으로 최대 2.72배 가속한다</h2>
    <p>로컬·개인 LLM처럼 <b style="color:var(--fg);font-weight:600">단일 요청(배치 1)</b>을 처리할 때는 다른 토큰과 묶을 필요가 없어 필요한 뉴런(13.7%)만 꺼내 읽으면 4개 층 전체가 바로 <b style="color:var(--fg);font-weight:600">5.33 ~ 6.55배</b>(정확도 99.63%) 빨라진다. 반면 <b style="color:var(--fg);font-weight:600">실제 서버 환경(배치 512 ~ 2,048 토큰)</b>에서는 요청마다 켜는 뉴런이 달라 GPU 텐서 코어 틀 안에 <b style="color:var(--fg);font-weight:600">불필요한 0</b>이 가득 찬다. 이 병목을 두 수학 문제로 단순화해 푼 <b style="color:var(--fg);font-weight:600">5단계 다중 크기 Roster-Tile</b>과 L2 캐시용 <b style="color:var(--fg);font-weight:600">열-스트립</b>을 결합해, 양자화 없는 순수 <span class="mono">bf16</span> 그대로 RTX 4060 실측 커널 4개 층 전체 <b style="color:var(--fg);font-weight:600">최대 2.56배(d=1,024) · 2.72배(d=4,096)</b> 가속을 확인했다.</p>
    <div class="stats">
      <div class="stat"><div class="v num" style="color:var(--neg)">5.33 ~ 6.55배</div><div class="k">단일 요청(B=1) 4개 층 전체 실측 · 정확도 99.63% (90.68%)</div></div>
      <div class="stat"><div class="v num on">최대 2.72배</div><div class="k">실제 서버(B=512 ~ 2,048) 4개 층 전체 실측 · 5단계 + L2 캐싱</div></div>
      <div class="stat"><div class="v num">99.59%</div><div class="k">서버 10,000장 시험 정확도 · 원본(91.02%) 대비 90.65% (통과)</div></div>
    </div>
  </section>

  <section class="step" id="b1">
    <div class="txt">
      <div class="no">묶기 1 · 문제 상황</div>
      <h2>토큰마다 필요한 뉴런이 산발적이라, 순서대로 틀을 끼우면 0이 가득 찬다</h2>
      <p class="key">GPU 텐서 코어는 8×16 고정 크기 틀로만 계산한다. 틀 안에 빈칸 0이 들어와도 똑같이 시간이 든다.</p>
      <ul>
        <li>세로축은 들어온 토큰(요청), 가로축은 층의 뉴런이다. 청록 칸(<span class="mono">1</span>)이 그 토큰이 켜는 뉴런이다.</li>
        <li>토큰마다 간 중심점이 달라 <span class="mono">1</span>의 위치가 제각각 흩어져 있다. 들어온 순서대로 틀을 씌우면 틀 안의 대부분이 불필요한 <span class="mono">0</span>(<span class="mono">×0</span>)으로 찬다.</li>
        <li>최종 모델은 중심점 목록의 <b>끝쪽 10%('꺼도 되는 1', <span class="mono">1*</span>)</b>는 꺼도 시험 정확도가 통과한다(일괄 10% 끄기: 90.62%, 통과선 90.52%).</li>
      </ul>
      <p class="why">오른쪽에서 '순서대로 틀 끼우기'를 눌러 보라. 아래 예시는 실제 8×16 틀을 2×3으로 축소한 것이다. 12개 틀을 하나도 건너뛸 수 없어 전체 72칸 중 48칸(66.7%)이 불필요한 0 계산으로 낭비된다.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between;gap:8px">
        <div class="chips" id="bt-p1-btns"></div>
        <span class="lbl" id="bt-p1-cap"></span>
      </div>
      <div class="bt-svg" id="bt-p1-svg"></div>
      <div class="legend">
        <span><i style="background:var(--on)"></i>필수 뉴런 (1)</span>
        <span><i style="background:var(--on-soft);box-shadow:inset 0 0 0 1.5px var(--on)"></i>끝쪽 10% · 꺼도 되는 1 (1*)</span>
        <span><i style="background:var(--pick)"></i>틀 안의 불필요한 0 (×0)</span>
        <span><i style="background:var(--off)"></i>안 쓰는 뉴런 (0)</span>
      </div>
      <div class="bt-kpi" id="bt-p1-kpi"></div>
    </div>
  </section>

  <section class="step" id="b2">
    <div class="txt">
      <div class="no">묶기 2 · 문제 단순화와 5단계 다중 크기 Roster-Tile</div>
      <h2>문제 1·2에 5단계 틀 순위를 매겨, 빽빽한 덩어리는 64×32 · 32×32로 떠내고 자투리만 8×16으로 마감한다</h2>
      <p class="key">텐서 코어 1회 명령은 8×16으로 이미 꽉 차지만, 8×16만 단독으로 돌리면 L2에서 데이터를 나르느라 코어가 26%에서 막힌다. 1순위 64×32(cuBLAS의 최대 98.7%)와 2순위 32×32를 먼저 뽑아야 공유 메모리 재사용으로 풀스피드가 나온다.</p>
      <ul>
        <li><b>0단계 (같은 중심점 자동 합승):</b> 같은 중심점을 고른 요청은 켜는 뉴런이 100% 같다. 이를 먼저 묶어 두면 0을 하나도 넣지 않고 큰 틀 덩어리가 저절로 커진다(출력 100% 동일).</li>
        <li><b>문제 2 확장 (5단계 틀 크기 우선순위):</b> RTX 4060 실측 효율에 맞춰 <b>1순위 <span class="mono">64×32</span> → 2순위 <span class="mono">32×32</span> → 3순위 <span class="mono">32×16</span> → 4순위 <span class="mono">16×16</span> → 5순위 <span class="mono">8×16</span></b> 순위와 실측 타일 비용(<span class="mono">1680 · 1024 · 640 · 486 · 410</span>)을 지정한다. 단위 GPU 시간당 필수 <span class="mono">1</span>을 가장 많이 덮는 큰 틀부터 차례로 뽑고, 느린 <span class="mono">8×16</span>은 1% 미만 자투리만 맡긴다. (A100·H100처럼 더 큰 칩은 최상위 틀만 <span class="mono">128×64</span> 등으로 키우면 같은 기법이 그대로 적용된다.)</li>
        <li><b>연동 풀기 + <span class="mono">1*</span> 자동 정리:</b> 모든 틀은 필수 <span class="mono">1</span>을 우선으로 덮으며, <b>이미 열린 틀에 들어온 <span class="mono">1*</span>은 공짜 ON, 새 틀을 열게 하는 외딴 <span class="mono">1*</span>은 자동 OFF</b>한다.</li>
      </ul>
      <p class="why">오른쪽 예시는 실제 <span class="mono">64×32 → 32×32 → 32×16 → 16×16 → 8×16</span>을 정확히 1/8로 축소한 <span class="mono">8×4 → 4×4 → 4×2 → 2×2 → 1×2</span> 5단계 순위 추출이다. 5개 틀(62칸, 불필요한 0 단 3칸)로 필수 1 55개를 겹침 0으로 덮으며, 전체 연산의 77.4%를 1·2순위 고속 틀로 처리한다.</p>
    </div>
    <div class="vis">
      <div class="bt-steps">
        <button class="btn" id="bt-prev" type="button">이전</button>
        <button class="btn primary" id="bt-next" type="button">다음</button>
        <span class="where" id="bt-where"></span>
      </div>
      <div class="bt-svg" id="bt-mat"></div>
      <div class="legend">
        <span><i style="background:var(--on)"></i>필수 1</span>
        <span><i style="background:var(--neg)"></i>1* 공짜 ON (틀 안)</span>
        <span><i style="background:var(--pick)"></i>불필요한 0 (×0)</span>
        <span><i style="background:var(--off);box-shadow:inset 0 0 0 1px var(--line-2)"></i>이미 지운 칸 · 끈 1*</span>
      </div>
      <div class="bt-side" id="bt-side"></div>
    </div>
    <details class="fold bt-wide bt-prob"><summary>단순화한 문제 1 · 문제 2 (5단계 틀 순위 조건 포함) 원문 보기</summary><div class="body">
      <div class="split">
        <div class="panel">
          <h3>문제 1</h3>
          <p>0과 1로 된 표가 있습니다. 행 8개, 열 32개입니다. 행 하나에 1은 8개쯤입니다. 행마다 1 가운데 무작위로 고른 하나에 '꺼도 되는 1' 표시가 있습니다.</p>
          <ol>
            <li>1을 묶는 직사각형들을 만들 수 있도록 행과 열의 순서를 마음대로 바꾸십시오. 모든 행과 열을 빠짐없이, 딱 한 번씩만 쓰고, 각 행과 열은 처음 번호를 그대로 달고 옮겨집니다.</li>
            <li>되도록 최대한 많은 1을 직사각형들로 묶으십시오. 직사각형끼리 겹치면 안 됩니다. 직사각형끼리는 붙어 있어도 되고 떨어져 있어도 됩니다.</li>
            <li>직사각형들은 최대한 1로만 이루어지게, 최대한 크게 만드십시오. 직사각형 안에는 최대한 적은 0을 포함시키십시오.
              <ul><li>참고: 한 줄짜리(1 × x, x × 1)도 직사각형입니다. 되도록 여러 줄짜리로 만드십시오.</li></ul>
            </li>
            <li>'꺼도 되는 1'은 직사각형을 만드는 데 도움이 되면 끄고, 도움이 안 되면 그대로 두십시오.</li>
          </ol>
        </div>
        <div class="panel">
          <h3>문제 2 (5단계 다중 크기 틀 순위 확장)</h3>
          <p>문제 1에서 만든 직사각형들이 있습니다. 1 × 1도 직사각형입니다. GPU 데이터 재사용률과 실측 실행 비용에 맞춰 우선순위가 매겨진 5단계 틀 목록(RTX 4060 기준 <b>1순위 64×32 → 2순위 32×32 → 3순위 32×16 → 4순위 16×16 → 5순위 8×16</b>)이 있습니다.</p>
          <ol>
            <li><b>(최우선 조건)</b> 틀의 가로줄 하나에는 행 번호 하나만, 세로줄 하나에는 열 번호 하나만 들어갑니다.
              <ul>
                <li>예: 줄인 틀(가로줄 2개, 세로줄 4개)로 보면 이렇습니다.
                  <ul>
                    <li>조각 (행 1, 열 1·2)와 조각 (행 1, 열 7·8): 행 번호가 같아서 가로줄 하나에 나란히 들어갑니다. 가로줄 하나가 남습니다.</li>
                    <li>조각 (행 1, 열 1·2)와 조각 (행 5, 열 7·8): 행 번호가 달라서 가로줄을 하나씩 따로 씁니다. 틀이 다 찼는데 8칸 가운데 4칸은 0으로 채워집니다.</li>
                  </ul>
                </li>
              </ul>
            </li>
            <li><b>(5단계 틀 순위 조건)</b> 공유 메모리 재사용률이 높은 큰 틀을 앞 순위로 두고 각 틀의 실측 비용을 부여합니다. 빽빽한 덩어리는 1순위(<span class="mono">64×32</span>) · 2순위(<span class="mono">32×32</span>)로 먼저 찍어 내고, 중간 밀도는 3·4순위(<span class="mono">32×16 · 16×16</span>)로, 마지막 성긴 자투리만 5순위(<span class="mono">8×16</span>)로 마감합니다.</li>
            <li>표 위에서 모든 필수 1을 빠짐없이, 딱 한 번씩만 찍어 내십시오. 틀을 다 못 채우면 그곳은 0으로 채우며, 전체 실행 비용과 불필요한 0이 최소가 되게 하십시오.</li>
          </ol>
        </div>
      </div>
    </div></details>
  </section>

  <section class="step" id="b3">
    <div class="txt">
      <div class="no">묶기 3 · 실제 서버 환경 4개 층 통계 (5단계 다중 크기)</div>
      <h2>실제 서버 배치(512 ~ 2,048 토큰)에서 연산의 72 ~ 77%를 64×32 · 32×32 고속 틀로 묶고, 저속 8×16은 1% 미만으로 줄였다</h2>
      <p class="key">실제 LLM 서빙 서버의 토큰 청크(배치 <span class="mono">512 / 1,024 / 2,048</span>)에 같은 중심점 자동 합승 + 5단계 순위 + 512열 8구역 스트립을 적용하고, 시험 10,000장 연쇄 정확도까지 새로 실측했다.</p>
      <ul>
        <li><b>배치 512 (소형 청크 · <span class="mono">W=512</span>):</b> 전체 8,388,608칸 → <b>2,338,432칸(27.9%)</b>, 불필요한 0 <b>83.0% 제거</b>. 1·2순위(<span class="mono">64×32 · 32×32</span>) 비중 <b>72.0%</b>, 저속 <span class="mono">8×16</span>은 <b>0.9%</b>에 그친다.</li>
        <li><b>배치 1,024 · 2,048 (Chunked Prefill · vLLM 기본 청크):</b> 배치가 커질수록 같은 중심점 합승이 늘어나 1순위 <span class="mono">64×32</span> 비중이 <b>29.8% → 38.7%</b>로 커지고, 1·2순위 고속 틀 합계가 <b>72.4% → 77.0%</b>에 달한다. 저속 <span class="mono">8×16</span>은 <b>0.3% → 0.2%</b>로 사실상 사라지며 연산량은 <b>23.0% → 19.1%</b>(불필요한 0 <b>92.5% 제거</b>)까지 내려간다.</li>
        <li><b>새로 실측한 시험 10,000장 연쇄 정확도 90.65% (베이스 91.02%의 99.59%, 통과):</b> 묶기 전 최종 모델(90.68%)과 <b>0.03%p 차이</b>로 통계적 동등 수준이며, 전 배치·전 층 필수 <span class="mono">1</span> 누락 0건을 전수 검증했다.</li>
      </ul>
      <p class="note">오른쪽 위 칩을 눌러 실제 서버 배치 512 · 1,024 · 2,048의 4개 층 5단계 통계를 비교해 보라.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between;gap:8px">
        <div class="chips" id="bt-b-chips"></div>
        <span class="lbl" id="bt-b-note"></span>
      </div>
      <div class="bt-tbl-wrap"><table class="bt-tbl" id="bt-ltbl"></table></div>
      <div class="bt-kpi" id="bt-b-mix"></div>
    </div>
  </section>

  <section class="step" id="b4">
    <div class="txt">
      <div class="no">묶기 4 · 캐싱 (DRAM 읽는 양 줄이기)</div>
      <h2>같이 쓰이는 뉴런끼리 미리 옆자리로 이사시켜, L2 크기 구역마다 한 번에 끝낸다</h2>
      <p class="key">대형 LLM은 한 층 가중치가 L2 캐시(작업대)보다 크다. 4,096열 전체에서 타일을 뽑으면 작업대가 좁아 방금 올린 뉴런을 버렸다가 DRAM에서 또 퍼 오게 된다.</p>
      <ul>
        <li><b>① 추론 전 딱 1번 — 옆자리로 이사 (비용 0초):</b> FFN은 뉴런 결과를 모두 더하므로 <span class="mono">gate · up · down</span>의 뉴런(열) 순서만 똑같이 바꿔 저장해도 출력이 100% 같다. 중심점 표를 보고 같이 자주 쓰이는 뉴런끼리 이웃하게 한 번 정렬해 둔다.</li>
        <li><b>② 작업대(L2) 크기 구역으로 나누기:</b> 오른쪽 예시처럼 뉴런 8개·작업대 4칸이면 <span class="mono">[N1, N5, N3, N7]</span>(구역 1)과 <span class="mono">[N2, N6, N4, N8]</span>(구역 2)로 나눈다. 실제 4,096열에서는 대형 LLM L2에 통째로 들어가는 <b>512열씩 8개 구역</b>으로 나눈다.</li>
        <li><b>③ 구역 1을 완전히 끝낸 뒤 구역 2로 이동:</b> 구역 1을 DRAM에서 <b>딱 1번</b> L2에 올리면 그 안의 타일이 끝날 때까지 단 하나의 뉴런도 쫓겨나지 않는다. 그래서 <b>DRAM 중복 읽기가 1.0배(중복 0%)</b>로 고정된다.</li>
      </ul>
      <p class="why">안 정렬하고 그냥 자르면 늘 같이 쓰이는 N1과 N5가 서로 다른 구역으로 찢어져 연산량이 는다. 미리 옆자리로 모아 두면 구역을 잘라도 <span class="mono">64×32 · 32×32</span> 고속 타일이 77%까지 꽉 차고(배치 2,048 연산 19.1%), 8구역 병렬로 타일 계획 시간도 8배 이상 빨라진다.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between;gap:8px">
        <div class="chips" id="bt-c-btns"></div>
        <span class="lbl" id="bt-c-cap"></span>
      </div>
      <div class="bt-svg" id="bt-c-svg"></div>
      <div class="row" id="bt-c-strips-row" style="justify-content:space-between;gap:8px">
        <span class="lbl">실행할 구역 선택 (작업대 4칸):</span>
        <div class="bt-strip-chips" id="bt-c-strips"></div>
      </div>
      <div class="bt-kpi" id="bt-c-kpi"></div>
    </div>
  </section>

  <section class="step" id="b5">
    <div class="txt">
      <div class="no">묶기 5 · 이론 상한 예상 (단일 요청 vs 실제 서버)</div>
      <h2>단일 요청은 7.3배, 실제 서버는 배치 512에서 3.6배 · 대형 배치 2,048에서 5.2배가 이론 상한이다</h2>
      <p class="key">양자화 없는 순수 <span class="mono">bf16</span> 그대로, 두 가지 실제 배포 환경(로컬 단일 요청 vs 서버 대형 배치) 각각의 연산량·DRAM 읽기 기준 하드웨어 이론 상한(Roofline)이다.</p>
      <ul>
        <li><b>단일 요청 환경 (<span class="mono">B = 1</span> · 로컬/개인 LLM):</b> 다른 토큰과 묶을 필요가 없어 타일링 0 패딩이 <b>0%</b>다. 평균 활성 뉴런 <b>13.7%</b>만 DRAM에서 읽고 계산하므로 이론 상한 <b>7.3배</b>(<span class="mono">1 / 0.1373</span>)가 열리며, <span class="mono">1*</span> 탈락도 0개라 시험 정확도 <b>90.68%</b>(베이스의 <b>99.63%</b>)를 그대로 지킨다.</li>
        <li><b>실제 서버 환경 (<span class="mono">B = 512 / 1,024 / 2,048</span> · 5단계 + 열-스트립 <span class="mono">W=512</span>):</b> 10,000장 연쇄 시험 정확도 <b>90.65%</b>(베이스의 <b>99.59%</b>, 통과)를 유지하며, 연산량을 <b>27.9% → 23.0% → 19.1%</b>로 줄여 이론 상한 <b>3.6배 → 4.4배 → 5.2배</b>(완전 연산 바운드)를 달성한다.</li>
        <li><b>읽는 양의 물리적 차이:</b> 단일 요청은 토큰 1개의 뉴런(13.7%)만 읽지만, 서버 배치는 수백~수천 토큰의 합집합(층 1~3 ~99%, 층 4 ~30% → 평균 <b>82.2 ~ 83.0%</b>)을 읽는다. 이 이론치가 실제 GPU 커널에서 얼마나 실현되는지는 바로 아래 묶기 6 실측에서 확인한다.</li>
      </ul>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between"><h3 style="font-size:14px">실제 풀 모델 대비 연산량 · DRAM 읽는 양 · 이론 상한</h3><span class="lbl">bf16 · 풀 모델 = 100%, 1.0배</span></div>
      <div class="bt-svg" id="bt-res"></div>
      <div class="legend">
        <span><i style="background:var(--on)"></i>연산: 쓰는 1</span>
        <span><i style="background:var(--pick)"></i>연산: 남은 0</span>
        <span><i style="background:var(--fg-2)"></i>DRAM 읽는 양 (중복 0%)</span>
      </div>
      <div class="bt-tbl-wrap"><table class="bt-tbl" id="bt-ftbl"></table></div>
    </div>
    <details class="fold bt-wide"><summary>어떻게 셌나 · 정확도 검증 · 이론 상한 기준</summary><div class="body">
      <p><b>실제 서빙 풀 모델 (기준선 = 100%, 1.0배)</b>: 뉴런을 다 켠 베이스 모델(91.02%)을 cuBLAS로 계산한다. 연산량은 <span class="mono">4 × B × 4,096</span>칸, DRAM 읽는 양은 뉴런당 1번(<span class="mono">4 × 4,096 = 16,384</span>열)이다.</p>
      <p><b>시험 정확도 검증</b>: 단일 요청(<span class="mono">B=1</span>)은 <span class="mono">1*</span>을 끄지 않으므로 최종 모델 그대로 <b>90.68%</b>(베이스의 99.63%, 평균 활성 뉴런 13.73%)다. 실제 서버(5단계, <span class="mono">W=512</span>)는 시험 사진 10,000장을 512-배치 19개 + 272-배치 1개로 나눠 층 1 → 2 → 3 → 4 연쇄로 실제 마스크를 적용해 <b>90.65%</b>(베이스 대비 하락 0.37%p [-0.02, 0.76], 통과선 90.52% 통과; 묶기 전 90.68% 대비 하락 0.03%p [-0.11, 0.17], 기존만 맞힘 28장 vs 5단계만 맞힘 25장; 실제 켜는 뉴런 평균 13.05% [21.26 / 19.82 / 10.04 / 1.07%])를 확인했다.</p>
      <p><b>속도 (Roofline 이론 상한)</b>: 연산과 DRAM 읽기가 완전히 겹치는 이상적 한계 <span class="mono">max(B, ridge) / max(연산비율 × B, 읽기비율 × ridge)</span>로 계산했다(실측 RTX 4060 ridge 104, A100 ridge 156, H100 ridge 295).</p>
    </div></details>
  </section>

  <section class="step" id="b6">
    <div class="txt">
      <div class="no">묶기 6 · 실제 GPU 커널 실측 (RTX 4060 Laptop)</div>
      <h2>실제 모델(d=1,024)과 큰 모델(d=4,096) 모두, 4개 층 전체 실측으로 단일 최대 6.55배 · 서버 최대 2.72배 가속을 확인했다</h2>
      <p class="key">실제 RTX 4060에서 단일 요청용 벡터화 <span class="mono">bf16</span> GEMV 커널과 서버용 5단계 <span class="mono">mma.m16n8k16</span> 통합 커널을 직접 짜고, 모든 버전을 같은 CUDA Graph 세션에서 교대 측정해 <b>1·2·3·4층 전체 합계</b> 순수 FFN 벽시계 시간을 검증했다.</p>
      <ul>
        <li><b>① 단일 요청 환경 (<span class="mono">B = 1</span> · 로컬/개인 LLM):</b> 0 마스킹은 <b>1.00배</b> 제자리지만, 활성 뉴런만 꺼내 읽는 전용 GEMV 커널은 실제 모델(<span class="mono">d=1,024</span>)에서 4개 층 전체 <b>5.33배</b>(0.445ms → 0.083ms), 큰 모델(<span class="mono">d=4,096</span>)에서 <b>6.55배</b>(1.810ms → 0.276ms, 표본 이론 한계 6.85배의 <b>95.6%</b>)를 기록했다.</li>
        <li><b>② 실제 모델 · 서버 환경 (<span class="mono">d=1,024</span>, 층 24MB &lt; L2 32MB):</b> 타일 묶기 없는 0 마스킹은 <b>1.03 ~ 1.05배</b>로 가속이 사실상 없다. 5단계 다중 크기 Roster-Tile은 4개 층 전체 <b>1.75배(<span class="mono">B=512</span>) → 2.16배(<span class="mono">B=1,024</span>) → 2.56배(<span class="mono">B=2,048</span>)</b>를 기록했다. 한 층이 L2 안에 통째로 들어가므로 전역과 스트립의 속도가 비슷하다.</li>
        <li><b>③ 큰 모델 LLM 층 · 서버 환경 (<span class="mono">d=4,096</span>, 층 96MB &gt; L2 32MB):</b> 0 마스킹은 <b>1.00 ~ 1.02배</b>, 5단계 전역(캐싱 X)은 <b>1.11 / 1.39 / 1.68배</b>에 머물지만, <span class="mono">W=512</span> 열-스트립 캐싱을 켜면 4개 층 전체 <b>1.68배(<span class="mono">B=512</span>) → 2.15배(<span class="mono">B=1,024</span>) → 2.72배(<span class="mono">B=2,048</span>)</b>로 올라간다.</li>
        <li><b>④ 1 → 4층 구간별 배수 추이:</b> 아래 통합 표 오른쪽 끝 열처럼, 활성 뉴런이 약 21%인 1·2층에서 10%인 3층, 1%인 4층으로 갈수록 희소성에 비례해 층별 배수가 커지며, 대표 배수는 이 4개 층을 모두 합산한 최종 수치다.</li>
      </ul>
      <p class="note">오른쪽 차트와 아래 통합 표에 작은 모델(<span class="mono">d=1,024</span> · 위쪽)과 큰 모델(<span class="mono">d=4,096</span> · 아래쪽) 실측을 한꺼번에 배치했다. 모든 설정에서 PyTorch <span class="mono">fp32</span> 기준값 대비 최대 상대 오차 <span class="mono">0.16 ~ 0.48%</span>(<span class="mono">bf16</span> 반올림 범위)를 통과했다.</p>
    </div>
    <div class="vis">
      <div class="row" style="justify-content:space-between;gap:8px">
        <h3 style="font-size:14px">RTX 4060 실측 속도 배수 (d=1,024 · d=4,096 전체 비교)</h3>
        <span class="lbl">4개 층 합계 · bf16 · CUDA Graph</span>
      </div>
      <div class="bt-b1-card" id="bt-hw-b1"></div>
      <div class="bt-svg" id="bt-hw-svg"></div>
      <div class="legend">
        <span><i style="background:var(--fg-3)"></i>풀 베이스 (cuBLAS)</span>
        <span><i style="background:var(--pick)"></i>타일 없음 (0 마스킹)</span>
        <span><i style="background:var(--neg)"></i>5단계 (캐싱 X · 전역)</span>
        <span><i style="background:var(--on)"></i>5단계 + 스트립 캐싱 (W=512)</span>
      </div>
      <div class="bt-kpi" id="bt-hw-kpi"></div>
    </div>
    <div class="bt-wide">
      <div class="bt-tbl-wrap"><table class="bt-tbl" id="bt-hwtbl"></table></div>
    </div>
    <details class="fold bt-wide"><summary>커널 최적화 레시피 한눈에 보기 (출력 무손실 · 순수 bf16)</summary><div class="body">
      <ol class="bt-recipe">
        <li><b>단일 요청(<span class="mono">B=1</span>) 전용 벡터화 <span class="mono">bf16</span> GEMV 커널:</b> 타일링 없이 활성 뉴런 인덱스 배열만 순회해 128비트(<span class="mono">uint4</span> · 8 <span class="mono">bf16</span>) 로드 + 워프 리덕션으로 <span class="mono">gate·up·SiLU</span>를 레지스터에서 융합하고, 2단계 <span class="mono">down</span>은 <span class="mono">d</span> 차원 블록 분할 병렬 축소로 DRAM 대역폭(213 GB/s, 기준의 96%)을 포화시켰다.</li>
        <li><b>같은 중심점 요청 자동 합승:</b> 같은 중심점을 고른 서버 토큰은 활성 뉴런이 100% 같으므로 타일링 전 자동 합승해 0 추가 없이 <span class="mono">64×32 · 32×32</span> 대형 덩어리를 만든다.</li>
        <li><b>5단계 실측 비용 가중 우선순위 + 단일 통합 커널(<span class="mono">P1Unified</span>):</b> 5개 크기(<span class="mono">64×32</span> 1개/CTA · <span class="mono">32×32</span> 1개/CTA · <span class="mono">32×16</span> 2개/CTA · <span class="mono">16×16</span> 2개/CTA · <span class="mono">8×16</span> 4개/CTA)를 스트립 순서대로 커널 1번에 띄워 1단계 <span class="mono">64×32</span>가 <span class="mono">cuBLAS</span>의 최대 98.7% 효율을 내고 구역 간 웨이브 테일을 없앴다.</li>
        <li><b>전치 <span class="mono">mma.m16n8k16</span> + 무셔플 <span class="mono">SiLU×Up</span> 레지스터 융합:</b> <span class="mono">M = 2×뉴런 수</span>, <span class="mono">N = 토큰 수</span>로 매핑하고 가중치를 <span class="mono">[gate 8줄 | up 8줄]</span> 16줄 블록으로 배치해, 모든 크기에서 코어 100% 채움 + 워프 셔플 없이 같은 스레드에서 <span class="mono">silu(g)·u·mask</span>를 끝낸다.</li>
        <li><b>오프라인 가중치 물리 재배치 (<span class="mono">W_gu</span> 인접 저장):</b> 추론 전 1회, 열 정렬 순서대로 <span class="mono">gate · up · down</span> 행을 물리 재배열해 각 512열 스트립이 DRAM에 연속 구역으로 놓이게 한다(추론 비용 0초).</li>
        <li><b><span class="mono">atomicAdd</span> 없는 2단계 <span class="mono">down + residual</span>(<span class="mono">P2Fast</span>) + LPT + CUDA Graph:</b> L2 상주 중간 버퍼를 32토큰 묶음별 활성 열만 압축(<span class="mono">pack_h</span>)한 뒤, 활성 열 내림차순(LPT) + <span class="mono">grid=(토큰블록, d/128)</span> 스위즐링 + 128비트 <span class="mono">cp.async</span> 2단계 파이프라인으로 <span class="mono">W_down</span> 1.0배 읽기·무충돌 잔차 합산을 수행하고, 합집합 &gt; 62% 밀집 층은 <span class="mono">cuBLAS</span>로 자동 전환하며 전 과정을 CUDA Graph로 묶었다.</li>
      </ol>
    </div></details>
  </section>
  </div>
  ''' + ME

POST = MS + r'''<script>
/* 묶어서 계산 (단일 요청 B=1 + 5단계 다중 크기 Roster-Tile + RTX 4060 실측) 보기: 기존 스크립트를 고치지 않고 따로 둔다 */
window.__BATCH__ = __DATA__;
(() => {
'use strict';
const B = window.__BATCH__;
const $ = (s, e = document) => e.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = 'http://www.w3.org/2000/svg';
const C = {};
const st = { p1: 1, step: 0, batch: '2048', cache: 2, strip: 0 };
const nf = v => Number(v).toLocaleString('ko-KR');
function el(t, a = {}, p) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
function tx(p, x, y, s, a = {}) { const t = el('text', Object.assign({ x, y }, a), p); t.textContent = s; return t; }

// ---- 묶기 1: 문제 상황 (6행 x 12열, 순서대로 2x3 틀 씌우기 — 유지) ----
const P1_ROWS = ['T1 가', 'T2 나', 'T3 다', 'T4 라', 'T5 마', 'T6 바'];
const P1_G = [
  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2],
  [0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0],
  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0],
  [0, 1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 1],
];
function drawProb() {
  const btns = $('#bt-p1-btns'); btns.innerHTML = '';
  ['① 토큰 × 뉴런 표', '② 순서대로 틀(2×3) 끼우기'].forEach((label, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', st.p1 === idx);
    b.onclick = () => { st.p1 = idx; drawProb(); };
    btns.appendChild(b);
  });
  $('#bt-p1-cap').textContent = st.p1 ? '2×3 틀 12개 · 모든 틀에 불필요한 0이 4칸씩' : '토큰 6개 × 뉴런 12개 · 1이 산발적으로 흩어짐';

  const host = $('#bt-p1-svg'); host.innerHTML = '';
  const W = Math.max(320, host.clientWidth || 540), lx = 58, rx = 10, top = 28;
  const cs = Math.min(36, (W - lx - rx) / 12), H = top + 6 * cs + 14;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '토큰 6개 x 뉴런 12개 표와 순서대로 2x3 틀 씌우기' }, host);
  const tip = $('#tip');

  tx(svg, lx - 8, top - 9, '뉴런', { 'text-anchor': 'end' });
  for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 9, 'N' + (c + 1), { 'text-anchor': 'middle' });
  for (let r = 0; r < 6; r++) tx(svg, lx - 8, top + r * cs + cs / 2 + 4, P1_ROWS[r], { 'text-anchor': 'end', class: 'lab' });

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 12; c++) {
      const v = P1_G[r][c], x = lx + c * cs + 2, y = top + r * cs + 2, s = cs - 4;
      let fill = C.off, stroke = 'none', dash = '';
      if (v === 1) fill = C.on;
      else if (v === 2) { fill = C.onsoft; stroke = C.on; dash = '3 2'; }
      else if (st.p1 === 1) fill = C.pick;
      const attrs = { x, y, width: s, height: s, rx: 3, fill };
      if (stroke !== 'none') { attrs.stroke = stroke; attrs['stroke-width'] = 1.4; if (dash) attrs['stroke-dasharray'] = dash; }
      el('rect', attrs, svg);
      if (v === 2) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 4, '1*', { 'text-anchor': 'middle', fill: C.on, 'font-size': Math.max(9.5, cs * 0.34), 'font-weight': 600 });
      else if (v === 0 && st.p1 === 1) tx(svg, lx + c * cs + cs / 2, top + r * cs + cs / 2 + 3.5, '×0', { 'text-anchor': 'middle', fill: C.bg, 'font-size': Math.max(8.5, cs * 0.30) });
    }
  }
  if (st.p1 === 1) {
    let tid = 0;
    for (let br = 0; br < 3; br++) {
      for (let bc = 0; bc < 4; bc++) {
        tid++;
        const x = lx + bc * 3 * cs, y = top + br * 2 * cs, w = 3 * cs, h = 2 * cs;
        const box = el('rect', { x: x + 0.5, y: y + 0.5, width: w - 1, height: h - 1, rx: 4, fill: 'transparent', stroke: C.fg, 'stroke-width': 1.3, 'stroke-opacity': 0.7, style: 'cursor:pointer' }, svg);
        const idCopy = tid;
        box.addEventListener('pointerenter', ev => {
          box.setAttribute('stroke', C.on); box.setAttribute('stroke-width', 2.2); box.setAttribute('stroke-opacity', 1);
          tip.hidden = false;
          tip.textContent = `틀 #${idCopy} (2×3): 쓰는 1 = 2칸 · 불필요한 0 = 4칸 (채움 33%)`;
          tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, ev.clientX + 12) + 'px';
          tip.style.top = (ev.clientY - 34) + 'px';
        });
        box.addEventListener('pointermove', ev => {
          tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, ev.clientX + 12) + 'px';
          tip.style.top = (ev.clientY - 34) + 'px';
        });
        box.addEventListener('pointerleave', () => {
          box.setAttribute('stroke', C.fg); box.setAttribute('stroke-width', 1.3); box.setAttribute('stroke-opacity', 0.7);
          tip.hidden = true;
        });
      }
    }
  }
  $('#bt-p1-kpi').innerHTML = st.p1
    ? `<div class="panel"><div class="k">실제 필요한 계산 (1 + 1*)</div><div class="v on">24칸 <span style="font-size:12px;color:var(--fg-3)">/ 72칸 (33.3%)</span></div></div>
       <div class="panel"><div class="k">순서대로 틀 끼운 결과 (2×3 × 12틀)</div><div class="v pick">불필요한 0이 48칸 (66.7%)</div></div>`
    : `<div class="panel"><div class="k">표 크기 (세로 토큰 × 가로 뉴런)</div><div class="v">6행 × 12열 = 72칸</div></div>
       <div class="panel"><div class="k">필요한 뉴런</div><div class="v on">필수 18칸 + 끝쪽 1* 6칸</div></div>`;
}

// ---- 묶기 2: 5단계 다중 크기 순위 Roster-Tile 7단계 예시 (8행 x 12열, 1/8 축소판 8x4 -> 4x4 -> 4x2 -> 2x2 -> 1x2) ----
// 필수 1 = 55개, 꺼도 되는 1* = 7개 (틀 안 공짜 ON 4개, 외딴 1* 자동 OFF 3개). 5개 틀 겹침 0칸, 불필요한 0 단 3칸!
const J_G = [
  [1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 1, 0], // R1: T1{C1,C3,C6,C9}=4, T2{C2,C5,C8,C11}=4, (R1,C4)=1* OFF
  [1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 1, 0], // R2: T1=4, T2=4, (R2,C10)=1* OFF
  [1, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 1], // R3: T1{C6=1*ON}=4, T3{C2,C5}=2, T4{C7,C12}=2
  [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0], // R4: T1=4, T2{C11=1*ON}=4
  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0], // R5: T1=4, T3{C2,C5}=2, T5{C4,C10}=2
  [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], // R6: T1=4, T2{C8=0}=3/4
  [1, 1, 1, 0, 2, 1, 0, 0, 2, 0, 0, 1], // R7: T1{C9=1*ON}=4, T3{C5=1*ON}=2, T4{C7=0,C12=1}=1/2
  [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2], // R8: T1{C3=0}=3/4, T3{C2,C5}=2, (R8,C12)=1* OFF
];
const J_TILES = [
  { name: '① 8×4 (64×32)', rows: [0, 1, 2, 3, 4, 5, 6, 7], cols: [0, 2, 5, 8],  stat: '31/32 (0 1칸)', accent: 'on' },
  { name: '② 4×4 (32×32)', rows: [0, 1, 3, 5],             cols: [1, 4, 7, 10], stat: '15/16 (0 1칸)', accent: 'on' },
  { name: '③ 4×2 (32×16)', rows: [2, 4, 6, 7],             cols: [1, 4],        stat: '8/8 만점',      accent: 'pick' },
  { name: '④ 2×2 (16×16)', rows: [2, 6],                   cols: [6, 11],       stat: '3/4 (0 1칸)',   accent: 'pick' },
  { name: '⑤ 1×2 (8×16)',  rows: [4],                      cols: [3, 9],        stat: '2/2 만점',      accent: 'neg' },
];
const J_STEPS = [
  { title: '원래 표 (8행 × 12열)', sub: '필수 1이 55개, 꺼도 되는 1*이 7개. 1순위 8×4(64×32) → 4×4 → 4×2 → 2×2 → 1×2(8×16) 순으로 빽빽한 덩어리부터 떠낸다.' },
  { title: '① 1순위 8×4 (64×32 비유) 먼저 추출', sub: '가장 빽빽한 {R1~R8} × {C1,C3,C6,C9}을 1순위 8×4로 떠낸다. (R3,C6)·(R7,C9)의 1*이 공짜로 들어와 31/32(97%)을 최고 효율(cuBLAS 98.7%)로 처리!' },
  { title: '② 2순위 4×4 (32×32 비유) 추출', sub: '1번 틀을 지운 뒤 남은 {R1,R2,R4,R6} × {C2,C5,C8,C11}을 2순위 4×4로 떠낸다. (R4,C11) 1*이 공짜 ON되어 15/16(94%)!' },
  { title: '③ 3순위 4×2 (32×16 비유) 추출', sub: '남은 {R3,R5,R7,R8} × {C2,C5}을 3순위 4×2로 떠낸다. (R7,C5) 1*이 공짜 ON되어 8/8 만점!' },
  { title: '④ 4순위 2×2 (16×16 비유) 추출', sub: '큰 틀 덩어리가 끝나면 4순위 2×2로 {R3,R7} × {C7,C12}(3/4)을 주워 담는다.' },
  { title: '⑤ 5순위 1×2 (8×16 비유)로 마감', sub: '마지막 자투리 필수 1 2개({R5} × {C4,C10})를 5순위 1×2로 0 낭비 없이 딱 맞게 마감한다.' },
  { title: '⑥ 종료 — 외딴 1* 3개 자동 OFF, 77.4%를 1·2순위 고속 틀로 처리', sub: '필수 1이 0개 남아 즉시 종료! 외딴 (R1,C4)·(R2,C10)·(R8,C12) 1* 3개는 새 틀을 열게 하므로 자동 OFF한다.' },
];
const inTile = (t, r, c) => t.rows.includes(r) && t.cols.includes(c);

function drawJoint() {
  const S = J_STEPS[st.step], host = $('#bt-mat'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 560);
  const stack = W < 520;
  const lx = 28, top = 32;
  const cs = stack ? Math.min(28, Math.floor((W - lx - 10) / 12)) : Math.min(24, Math.max(19, Math.floor((W - 252) / 12)));
  const matW = 12 * cs, matH = 8 * cs;
  const rx0 = stack ? lx : lx + matW + Math.min(24, Math.max(14, Math.floor((W - lx - matW - 216) / 2)));
  const rw = Math.max(206, W - rx0 - 4);
  const mcs = stack ? Math.min(16, Math.max(12, Math.floor((rw - 30) / 12))) : Math.min(13, Math.max(11, Math.floor((rw - 24) / 9)));
  const ry0 = stack ? top + matH + 34 : top;
  const col2 = rx0 + Math.floor(rw * 0.51);
  const y_r2_c1 = ry0 + 12 + 8 * mcs + 32;
  const y_r2_c2 = ry0 + 12 + 4 * mcs + 32;
  const y_r3_c2 = y_r2_c2 + 4 * mcs + 32;
  const panelH = Math.max(matH, (y_r3_c2 - ry0) + 1 * mcs + 20);
  const H = stack ? ry0 + panelH + 10 : Math.max(top + matH + 16, top + panelH + 10);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '5단계 다중 크기 Roster-Tile 8x12 예시: ' + S.title }, host);

  tx(svg, lx, 13, '남은 표 (8행 × 12열)', { class: 'hi' });
  for (let c = 0; c < 12; c++) tx(svg, lx + c * cs + cs / 2, top - 6, 'C' + (c + 1), { 'text-anchor': 'middle', 'font-size': 9 });
  for (let r = 0; r < 8; r++) tx(svg, lx - 5, top + r * cs + cs / 2 + 3.5, 'R' + (r + 1), { 'text-anchor': 'end', class: 'lab', 'font-size': 10 });

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 12; c++) {
      const v = J_G[r][c], x = lx + c * cs + 1.5, y = top + r * cs + 1.5, s = cs - 3;
      let done = false, hi = false;
      for (let ti = 0; ti < J_TILES.length; ti++) {
        if (inTile(J_TILES[ti], r, c)) {
          if (st.step > ti + 1) done = true;
          else if (st.step === ti + 1) hi = true;
        }
      }
      const offOpt = st.step === 6 && v === 2 && !done;

      let fill = C.off, stroke = 'none', dash = '', op = 1, txt = '', tfill = C.bg;
      if (done) {
        fill = C.bg; stroke = C.line; op = 0.7; txt = '·'; tfill = C.fg3;
      } else if (offOpt) {
        fill = C.bg; stroke = C.line2; dash = '2 2'; txt = 'OFF'; tfill = C.fg3;
      } else if (hi && v === 0) {
        fill = C.pick; stroke = C.fg; txt = '×0'; tfill = C.bg;
      } else if (hi && v === 2) {
        fill = C.neg; stroke = C.fg; txt = '1*'; tfill = C.bg;
      } else if (v === 1) {
        fill = C.on; if (hi) stroke = C.fg;
      } else if (v === 2) {
        fill = C.onsoft; stroke = C.on; dash = '2.5 2'; txt = '1*'; tfill = C.on;
      }
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
    tile.rows.forEach((r, i) => {
      tile.cols.forEach((c, j) => {
        const v = J_G[r][c];
        const f = v === 1 ? C.on : v === 2 ? C.neg : C.pick;
        el('rect', { x: ox + j * mcs + 1, y: oy + i * mcs + 1, width: mcs - 2, height: mcs - 2, rx: 2, fill: f }, svg);
      });
    });
    el('rect', { x: ox, y: oy, width: tw, height: th, rx: 3, fill: 'none', stroke: col, 'stroke-width': 1.3 }, svg);
    tx(svg, ox, oy + th + 12, tile.stat, { class: 'lab', 'font-size': 9 });
  };

  if (st.step === 0) {
    const ph = stack ? 116 : matH;
    el('rect', { x: rx0, y: ry0, width: rw, height: ph, rx: 8, fill: C.bg, stroke: C.line, 'stroke-dasharray': '4 3' }, svg);
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 - 16, '1순위 8×4 → 2순위 4×4', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 + 1,  '3순위 4×2 → 4순위 2×2 → 1×2', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
    tx(svg, rx0 + rw / 2, ry0 + ph / 2 + 21, '다음을 눌러 5단계 추출 시작', { 'text-anchor': 'middle', 'font-size': 9.5 });
  } else {
    if (st.step >= 1) drawMini(J_TILES[0], rx0,  ry0 + 12);
    if (st.step >= 2) drawMini(J_TILES[1], col2, ry0 + 12);
    if (st.step >= 3) drawMini(J_TILES[2], col2, y_r2_c2);
    if (st.step >= 4) drawMini(J_TILES[3], rx0,  y_r2_c1);
    if (st.step >= 5) drawMini(J_TILES[4], col2, y_r3_c2);
  }

  $('#bt-where').innerHTML = `<b>${S.title}</b> <span class="mono" style="color:var(--fg-3)">(${st.step + 1} / ${J_STEPS.length})</span>`;
  const side = $('#bt-side');
  if (st.step < 6) {
    side.innerHTML = `<p>${S.sub}</p>`;
  } else {
    side.innerHTML = `<p>${S.sub}</p>
      <div class="bt-tbl-wrap"><table class="bt-tbl">
        <tr><th>비교</th><th>전체 칸</th><th>1·2순위 고속 틀 비중</th><th>불필요한 0</th><th>필수 1 (55개)</th></tr>
        <tr><td>순서대로 1×2 자르기</td><td class="num">84칸 <span class="mut">(전체 96)</span></td><td class="num warn">0% (전부 저속)</td><td class="num warn">22 ~ 29칸</td><td class="num">100%</td></tr>
        <tr class="sum"><td>5단계 Roster-Tile</td><td class="num ok">62칸 (-35.4%)</td><td class="num ok">77.4% (8×4 · 4×4)</td><td class="num ok">3칸 (-91%)</td><td class="num ok">100% · 겹침 0</td></tr>
      </table></div>`;
  }
}

// ---- 묶기 3: 실제 서버 환경 4개 층 통계 (5단계 다중 크기 · B=512/1024/2048) ----
function drawLayerStats() {
  const chips = $('#bt-b-chips'); chips.innerHTML = '';
  [['512', '배치 512 (소형 청크)'], ['1024', '배치 1,024 (Chunked Prefill)'], ['2048', '배치 2,048 (vLLM 기본 청크)']].forEach(([k, label]) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', st.batch === k);
    b.onclick = () => { st.batch = k; drawLayerStats(); };
    chips.appendChild(b);
  });
  const T = B.layer_tables[st.batch], tot = T.total;
  $('#bt-b-note').textContent = T.note;

  const rowHtml = (name, r, isSum) => {
    const c = r.counts;
    return `<tr${isSum ? ' class="sum"' : ''}>
      <td>${name} <span class="mut">(${r.density_pct.toFixed(1)}%)</span></td>
      <td class="num">${nf(r.full_cells)}</td>
      <td class="num"><b style="color:var(--fg)">${nf(r.comp_cells)}</b> <span class="mut">(${r.remain_pct.toFixed(1)}% · -${r.reduce_pct.toFixed(1)}%)</span></td>
      <td class="num">${nf(c['64x32'])} / ${nf(c['32x32'])} / ${nf(c['32x16'])} / ${nf(c['16x16'])} / ${nf(c['8x16'])}</td>
      <td class="num ok">${r.share_top2.toFixed(1)}% <span class="mut">(64×32 ${r.share_64x32.toFixed(1)}% · 채움 ${r.fill_pct.toFixed(1)}%)</span></td>
    </tr>`;
  };

  $('#bt-ltbl').innerHTML =
    '<tr><th>층 (활성 1)</th><th>풀 모델 칸</th><th>5단계 연산 칸</th><th>틀 수 (64×32 / 32×32 / 32×16 / 16×16 / 8×16)</th><th>1·2순위 고속 틀 비중</th></tr>' +
    T.layers.map(r => rowHtml('층 ' + r.layer, r, false)).join('') +
    rowHtml('4개 층 합계', tot, true);

  $('#bt-b-mix').innerHTML =
    `<div class="panel"><div class="k">5단계 연산 칸 분포 (1→5순위)</div><div class="v on" style="font-size:13.5px">64×32 ${tot.share_64x32.toFixed(1)}% · 32×32 ${tot.share_32x32.toFixed(1)}% · 32×16 ${tot.share_32x16.toFixed(1)}% · 16×16 ${tot.share_16x16.toFixed(1)}% · 8×16 ${tot.share_8x16.toFixed(1)}%</div></div>
     <div class="panel"><div class="k">불필요한 0 제거율 · 시험 정확도</div><div class="v">${tot.zero_elim_pct.toFixed(1)}% 제거 <span style="font-size:12px;color:var(--fg-3)">· 10,000장 ${B.accuracy.roster_tile.toFixed(2)}% (통과)</span></div></div>`;
}

// ---- 묶기 4: 캐싱 (뉴런 8개 · L2 작업대 4칸 직관 예시 — 유지) ----
const C_ORIG = [1, 2, 3, 4, 5, 6, 7, 8];
const C_SORT = [1, 5, 3, 7, 2, 6, 4, 8];
const isFamA = n => n % 2 === 1;

function drawCache() {
  const btns = $('#bt-c-btns'); btns.innerHTML = '';
  ['① 그대로 돌리면 (중복 읽기)', '② 옆자리로 이사 (비용 0초)', '③ 구역별 실행 (1회 읽기)'].forEach((label, idx) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = label; b.setAttribute('aria-pressed', st.cache === idx);
    b.onclick = () => { st.cache = idx; drawCache(); };
    btns.appendChild(b);
  });
  const caps = [
    '작업대(L2) 4칸이 넘쳐 버렸던 N1을 DRAM에서 또 퍼 옴',
    '같이 쓰이는 짝(청록 · 보라)끼리 열 순서를 미리 한 번만 정렬',
    '구역 1개(4칸)가 작업대에 통째로 상주 → 쫓겨남 0개, 1회 읽기',
  ];
  $('#bt-c-cap').textContent = caps[st.cache];
  $('#bt-c-strips-row').style.display = st.cache === 2 ? 'flex' : 'none';

  const sch = $('#bt-c-strips'); sch.innerHTML = '';
  ['구역 1 (N1·N5·N3·N7)', '구역 2 (N2·N6·N4·N8)'].forEach((lbl, s) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip plain';
    b.textContent = lbl; b.setAttribute('aria-pressed', st.strip === s);
    b.onclick = () => { st.strip = s; drawCache(); };
    sch.appendChild(b);
  });

  const host = $('#bt-c-svg'); host.innerHTML = '';
  const W = Math.max(360, host.clientWidth || 560), H = 224, pad = 16, barW = W - 2 * pad;
  const gap = 6, midGap = 12, bw = (barW - 7 * gap - midGap) / 8;
  const bx = i => pad + i * (bw + gap) + (i >= 4 ? midGap : 0);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '뉴런 8개와 L2 작업대 4칸 캐싱 예시' }, host);

  const drawNeuron = (x, y, h, n, dim) => {
    const a = isFamA(n), col = a ? C.on : C.neg;
    el('rect', {
      x, y, width: bw, height: h, rx: 5,
      fill: dim ? C.bg : (a ? C.onsoft : 'rgba(157,140,255,0.14)'),
      stroke: dim ? C.line : col, 'stroke-width': dim ? 1 : 1.5, opacity: dim ? 0.55 : 1,
    }, svg);
    tx(svg, x + bw / 2, y + h / 2 + 4, 'N' + n, { 'text-anchor': 'middle', fill: dim ? C.fg3 : col, 'font-size': 11.5, 'font-weight': 600 });
  };

  if (st.cache === 0) {
    tx(svg, pad, 16, '원래 뉴런 순서 (같이 쓰이는 짝: 청록 N1·N3·N5·N7 / 보라 N2·N4·N6·N8이 뒤섞임)', { class: 'hi', 'font-size': 11 });
    C_ORIG.forEach((n, i) => drawNeuron(bx(i), 24, 28, n, false));

    const l2W = Math.min(barW, Math.max(250, barW * 0.62)), l2X = pad + (barW - l2W) / 2;
    el('rect', { x: l2X, y: 82, width: l2W, height: 54, rx: 8, fill: C.picksoft, stroke: C.pick, 'stroke-width': 1.6 }, svg);
    tx(svg, l2X + l2W / 2, 102, 'L2 작업대 (4칸 한계) · N1, N5가 밀려남!', { 'text-anchor': 'middle', fill: C.pick, 'font-size': 11.5, 'font-weight': 600 });
    tx(svg, l2X + l2W / 2, 122, '현재 상주: [N3, N7, N4, N8] ← N1을 쓰려면 DRAM에서 또 읽음', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });

    [0, 1, 2, 4, 5].forEach(i => {
      el('line', { x1: bx(i) + bw / 2, y1: 52, x2: l2X + l2W / 2, y2: 82, stroke: C.pick, 'stroke-width': 1.3, 'stroke-dasharray': '3 3', opacity: 0.85 }, svg);
    });

    tx(svg, pad, 158, '전역 타일 실행 순서 (4칸 작업대 초과로 재읽기 발생):', { class: 'lab', 'font-size': 11 });
    const steps = [
      ['1) {N1,N5}', '올림 (2/4)', C.fg3],
      ['2) {N2,N6}', '꽉 참 (4/4)', C.fg3],
      ['3) {N3,N7}', 'N1·N5 버림!', C.pick],
      ['4) {N1,N3}', 'N1 또 퍼 옴!', C.pick],
    ];
    const cw = (barW - 18) / 4;
    steps.forEach(([t1, t2, col], k) => {
      const tx0 = pad + k * (cw + 6), warn = col === C.pick;
      el('rect', { x: tx0, y: 168, width: cw, height: 42, rx: 6, fill: warn ? C.picksoft : C.bg3, stroke: warn ? C.pick : C.line2, 'stroke-width': warn ? 1.4 : 1 }, svg);
      tx(svg, tx0 + cw / 2, 185, t1, { 'text-anchor': 'middle', fill: warn ? C.fg : C.fg2, 'font-size': 10.5, 'font-weight': 600 });
      tx(svg, tx0 + cw / 2, 201, t2, { 'text-anchor': 'middle', fill: col, 'font-size': 10 });
    });
  } else if (st.cache === 1) {
    tx(svg, pad, 14, '원래 순서 → 바꾼 순서 (오프라인 1회 정렬 · 출력 100% 동일 · 추론 시간 0초)', { class: 'hi', 'font-size': 11 });
    C_ORIG.forEach((n, i) => drawNeuron(bx(i), 22, 26, n, false));

    C_ORIG.forEach((n, i) => {
      const j = C_SORT.indexOf(n), x1 = bx(i) + bw / 2, x2 = bx(j) + bw / 2;
      const col = isFamA(n) ? C.on : C.neg;
      el('path', { d: `M ${x1} 48 C ${x1} 70, ${x2} 70, ${x2} 92`, fill: 'none', stroke: col, 'stroke-width': 1.5, opacity: 0.8 }, svg);
    });

    C_SORT.forEach((n, i) => drawNeuron(bx(i), 92, 26, n, false));

    const g1x = bx(0), g1w = bx(3) + bw - bx(0);
    const g2x = bx(4), g2w = bx(7) + bw - bx(4);
    el('line', { x1: g1x + g1w / 2, y1: 118, x2: g1x + g1w / 2, y2: 138, stroke: C.on, 'stroke-width': 1.8 }, svg);
    el('line', { x1: g2x + g2w / 2, y1: 118, x2: g2x + g2w / 2, y2: 138, stroke: C.neg, 'stroke-width': 1.8 }, svg);

    el('rect', { x: g1x, y: 138, width: g1w, height: 60, rx: 8, fill: C.onsoft, stroke: C.on, 'stroke-width': 1.4 }, svg);
    tx(svg, g1x + g1w / 2, 162, '구역 1 (작업대 4칸 딱 맞음)', { 'text-anchor': 'middle', fill: C.on, 'font-size': 11.5, 'font-weight': 600 });
    tx(svg, g1x + g1w / 2, 182, 'N1 · N5 · N3 · N7 (같이 쓰이는 짝)', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });

    el('rect', { x: g2x, y: 138, width: g2w, height: 60, rx: 8, fill: 'rgba(157,140,255,0.12)', stroke: C.neg, 'stroke-width': 1.4 }, svg);
    tx(svg, g2x + g2w / 2, 162, '구역 2 (작업대 4칸 딱 맞음)', { 'text-anchor': 'middle', fill: C.neg, 'font-size': 11.5, 'font-weight': 600 });
    tx(svg, g2x + g2w / 2, 182, 'N2 · N6 · N4 · N8 (같이 쓰이는 짝)', { 'text-anchor': 'middle', class: 'lab', 'font-size': 10.5 });
  } else {
    const s = st.strip, acc = s === 0 ? C.on : C.neg, accSoft = s === 0 ? C.onsoft : 'rgba(157,140,255,0.14)';
    const zStart = s * 4, zNums = C_SORT.slice(zStart, zStart + 4);
    tx(svg, pad, 16, `정렬된 DRAM 가중치 · 현재 구역 ${s + 1}만 DRAM에서 딱 1번 올림`, { class: 'hi', 'font-size': 11 });
    C_SORT.forEach((n, i) => drawNeuron(bx(i), 24, 28, n, Math.floor(i / 4) !== s));

    const gx = bx(zStart) - 3, gw = bx(zStart + 3) + bw - bx(zStart) + 6;
    el('rect', { x: gx, y: 21, width: gw, height: 34, rx: 7, fill: 'none', stroke: acc, 'stroke-width': 1.8 }, svg);

    const l2W = Math.max(gw, 240), l2X = s === 0 ? pad : pad + barW - l2W;
    const cx = gx + gw / 2;
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

  $('#bt-c-kpi').innerHTML = st.cache === 0
    ? `<div class="panel"><div class="k">큰 모델(d=4,096) 캐싱 없을 때</div><div class="v pick">실측 1.11 ~ 1.68배 <span style="font-size:12px;color:var(--fg-3)">(L2 넘침으로 재읽기 발생)</span></div></div>
       <div class="panel"><div class="k">작은 모델(d=1,024 · RTX 4060)</div><div class="v">1.0배 <span style="font-size:12px;color:var(--fg-3)">(층 24MB &lt; L2 32MB라 통째로 들어감)</span></div></div>`
    : `<div class="panel"><div class="k">실제 4,096열 적용 (512열 × 8구역)</div><div class="v on">1.68 ~ 2.72배 <span style="font-size:12px;color:var(--fg-3)">(캐싱 X 1.11 ~ 1.68배 대비 · 중복 0%)</span></div></div>
       <div class="panel"><div class="k">8구역 병렬 + 합승 5단계 계획</div><div class="v">${B.strip_meta.runtime_1024.toFixed(2)}초 <span style="font-size:12px;color:var(--fg-3)">(전역 23.5초 대비 10배 단축)</span></div></div>`;
}

// ---- 묶기 5: 이론 상한 (단일 요청 vs 실제 서버) 차트 + 표 ----
function drawFinal() {
  const host = $('#bt-res'); host.innerHTML = '';
  const items = B.summary.filter(r => r.id !== 'full');
  const W = Math.max(320, host.clientWidth || 540), lx = 132, rx = 68, bh = 13, gh = 52, top = 28;
  const H = top + items.length * gh + 22;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '실제 풀 모델 대비 단일 요청 및 실제 서버 연산량, DRAM 읽는 양, 이론 상한' }, host);
  const sx = v => lx + v / 100 * (W - lx - rx);

  for (let v = 0; v <= 100; v += 20) {
    el('line', { x1: sx(v), x2: sx(v), y1: top - 6, y2: H - 20, stroke: C.line, 'stroke-width': 1 }, svg);
    tx(svg, sx(v), H - 6, v + '%', { 'text-anchor': 'middle' });
  }
  el('line', { x1: sx(100), x2: sx(100), y1: top - 12, y2: H - 20, stroke: C.fg2, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
  tx(svg, sx(100), top - 16, '실제 풀 모델 = 100%', { 'text-anchor': 'end', class: 'lab' });

  const shortLab = {
    single1: ['단일 요청 (B=1)', '로컬 · 타일링 없음'],
    srv512:  ['실제 서버 (512)', '5단계 · 열-스트립'],
    srv1024: ['실제 서버 (1,024)', '5단계 · 열-스트립'],
    srv2048: ['실제 서버 (2,048)', '5단계 · 열-스트립'],
  };

  items.forEach((R, i) => {
    const y = top + i * gh, isB1 = R.id === 'single1';
    const [l1, l2] = shortLab[R.id];
    tx(svg, lx - 10, y + 10, l1, { 'text-anchor': 'end', class: 'hi', 'font-size': 11, fill: isB1 ? C.neg : C.fg });
    tx(svg, lx - 10, y + 24, l2, { 'text-anchor': 'end', 'font-size': 9.5 });

    const w1 = Math.max(0, sx(R.ones_pct) - sx(0) - (R.zero_pct > 0 ? 1 : 0));
    const w0 = Math.max(0, sx(R.compute_pct) - sx(R.ones_pct));
    el('rect', { x: sx(0), y, width: w1, height: bh, rx: 2, fill: isB1 ? C.neg : C.on }, svg);
    if (w0 > 0) el('rect', { x: sx(R.ones_pct), y, width: w0, height: bh, rx: 2, fill: C.pick }, svg);
    tx(svg, sx(R.compute_pct) + 6, y + bh - 2.5, '연산 ' + R.compute_pct.toFixed(1) + '%', { fill: isB1 ? C.neg : C.on, 'font-size': 10.5 });

    const dramW = Math.max(2, sx(R.dram_pct) - sx(0));
    el('rect', { x: sx(0), y: y + bh + 4, width: dramW, height: bh, rx: 2, fill: C.fg2, opacity: 0.82 }, svg);
    if (R.dram_pct > 30) {
      tx(svg, sx(R.dram_pct) - 6, y + 2 * bh + 1, '읽기 ' + R.dram_pct.toFixed(1) + '%', { 'text-anchor': 'end', fill: C.bg, 'font-size': 9.5, 'font-weight': 600 });
    } else {
      tx(svg, sx(R.dram_pct) + 6, y + 2 * bh + 1, '읽기 ' + R.dram_pct.toFixed(1) + '%', { fill: C.fg2, 'font-size': 9.5, 'font-weight': 600 });
    }

    const bx = W - rx + 12;
    el('rect', { x: bx, y: y + 3, width: 52, height: 24, rx: 12, fill: isB1 ? 'rgba(157,140,255,0.16)' : C.onsoft, stroke: isB1 ? C.neg : C.on, 'stroke-width': 1 }, svg);
    tx(svg, bx + 26, y + 19, R.speedup.toFixed(1) + '배', { 'text-anchor': 'middle', fill: isB1 ? C.neg : C.on, 'font-size': 11, 'font-weight': 600 });
  });

  $('#bt-ftbl').innerHTML =
    '<tr><th>환경 구분</th><th>배치 (토큰)</th><th>적용 방식</th><th>연산량</th><th>DRAM 읽는 양</th><th>속도 (이론 상한)</th></tr>' +
    B.summary.map(r => {
      const cls = r.id === 'single1' ? ' class="b1-hi"' : (r.id === 'srv2048' ? ' class="sum"' : '');
      const pillCls = r.id === 'single1' ? 'bt-pill neg' : 'bt-pill';
      return `<tr${cls}>
        <td><b>${r.name}</b> <span class="mut">${r.sub}</span></td>
        <td class="num">${r.batch}</td>
        <td>${r.method}</td>
        <td class="num${r.id !== 'full' ? ' ok' : ''}">${r.compute_pct.toFixed(1)}%</td>
        <td class="num">${r.dram_pct.toFixed(1)}%</td>
        <td class="num">${r.id === 'full' ? '1.0배' : `<span class="${pillCls}">${r.speedup.toFixed(1)}배</span> <span class="mut">${r.speedup_note}</span>`}</td>
      </tr>`;
    }).join('');
}

// ---- 묶기 6: 실제 RTX 4060 커널 실측 — d=1,024(위) + d=4,096(아래) 한꺼번에 표시 ----
function drawHwBench() {
  const H1 = B.hw_bench['1024'], H4 = B.hw_bench['4096'];
  const fmtLayers = arr => arr.map(v => (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + '×').join(' / ');

  // 1) 상단: 단일 요청 (B=1) 실측 카드 — 위 d=1,024, 아래 d=4,096 한꺼번에
  const maxB1 = Math.max(H1.b1.speedup, H4.b1.speedup, 6.6);
  const b1Line = (tag, sub, b1) => {
    const wBase = (100.0 * 1.0 / maxB1).toFixed(1);
    const wOurs = (100.0 * b1.speedup / maxB1).toFixed(1);
    return `
      <div style="display:grid;gap:3px">
        <div style="display:flex;justify-content:space-between;align-items:baseline;font-size:12px">
          <span><b style="color:var(--fg)">${tag}</b> <span style="color:var(--fg-3);font-size:11px">${sub}</span></span>
          <span class="bt-pill neg">${b1.speedup.toFixed(2)}배 (${b1.base_ms.toFixed(3)}ms → ${b1.ours_ms.toFixed(3)}ms)</span>
        </div>
        <div style="display:grid;grid-template-columns:86px 1fr 104px;align-items:center;gap:8px;font-family:var(--mono);font-size:11px">
          <span style="color:var(--fg-3)">풀 베이스</span>
          <div style="height:8px;background:var(--bg-3);border-radius:3px;overflow:hidden"><div style="width:${wBase}%;height:100%;background:var(--fg-3);opacity:0.6"></div></div>
          <span style="color:var(--fg-2);text-align:right">1.00× (0마스크 ${b1.speedup_mask0.toFixed(2)}×)</span>
        </div>
        <div style="display:grid;grid-template-columns:86px 1fr 104px;align-items:center;gap:8px;font-family:var(--mono);font-size:11px">
          <span style="color:var(--neg);font-weight:600">단일 Roster</span>
          <div style="height:8px;background:var(--bg-3);border-radius:3px;overflow:hidden"><div style="width:${wOurs}%;height:100%;background:var(--neg)"></div></div>
          <span style="color:var(--neg);font-weight:600;text-align:right">${b1.speedup.toFixed(2)}× (${b1.ours_gbs.toFixed(0)} GB/s)</span>
        </div>
      </div>`;
  };
  $('#bt-hw-b1').innerHTML = `
    <div class="hdr">
      <span><b>① 단일 요청 환경 (배치 1 · 로컬/개인 LLM)</b> — 활성 뉴런(${H1.b1.avg_active_pct.toFixed(1)}%)만 직접 읽기</span>
      <span class="lbl">정확도 90.68% (99.63%)</span>
    </div>
    ${b1Line('d = 1,024', '실제 학습 모델 (4층 101MB > L2)', H1.b1)}
    <div style="height:1px;background:var(--line)"></div>
    ${b1Line('d = 4,096', '큰 모델 LLM 층 (4층 403MB > L2)', H4.b1)}
  `;

  // 2) 중단: 실제 서버 (B=512 / 1,024 / 2,048) 4버전 통합 바 차트 — 위 d=1,024, 아래 d=4,096
  const host = $('#bt-hw-svg'); host.innerHTML = '';
  const W = Math.max(340, host.clientWidth || 540), lx = 84, rx = 56, bh = 9.5, bgap = 2.5, gh = 56;
  const grpHdr = 22, grpGap = 16, top = 28;
  const blockH = grpHdr + 3 * gh;
  const svgH = top + blockH + grpGap + blockH + 22;
  const svg = el('svg', { viewBox: `0 0 ${W} ${svgH}`, role: 'img', 'aria-label': '실제 서버 d=1,024 및 d=4,096 RTX 4060 실측 속도 배수 비교' }, host);
  const maxS = 3.0;
  const sx = v => lx + Math.min(v, maxS) / maxS * (W - lx - rx);

  tx(svg, 4, 13, '② 실제 서버 환경 (5단계 Roster-Tile · 4개 층 전체 합계)', { class: 'hi', 'font-size': 11.5 });
  for (const v of [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) {
    el('line', { x1: sx(v), x2: sx(v), y1: top - 4, y2: svgH - 18, stroke: v === 1.0 ? C.fg2 : C.line, 'stroke-width': v === 1.0 ? 1.3 : 1, 'stroke-dasharray': v === 1.0 ? '3 3' : '' }, svg);
    tx(svg, sx(v), svgH - 4, v.toFixed(1) + '배', { 'text-anchor': 'middle', 'font-size': 10 });
  }

  const drawServerGroup = (H, yStart, title, isLarge) => {
    tx(svg, 4, yStart + 13, title, { fill: isLarge ? C.on : C.fg, 'font-size': 11, 'font-weight': 600 });
    H.rows.forEach((R, i) => {
      const y0 = yStart + grpHdr + i * gh;
      tx(svg, lx - 8, y0 + 18, '배치 ' + nf(R.B), { 'text-anchor': 'end', class: 'hi', 'font-size': 11 });
      tx(svg, lx - 8, y0 + 31, R.base_ms.toFixed(1) + 'ms', { 'text-anchor': 'end', 'font-size': 9.5 });

      const bars = [
        { v: 1.0,              col: C.fg3,  op: 0.55, lbl: '1.00×' },
        { v: R.speedup_mask0,  col: C.pick, op: 0.85, lbl: R.speedup_mask0.toFixed(2) + '×' },
        { v: R.speedup_global, col: C.neg,  op: 0.88, lbl: R.speedup_global.toFixed(2) + '×' },
        { v: R.speedup_strip,  col: C.on,   op: 0.95, lbl: R.speedup_strip.toFixed(2) + '×' },
      ];
      bars.forEach((br, k) => {
        const y = y0 + k * (bh + bgap);
        const bw = Math.max(2, sx(br.v) - sx(0));
        el('rect', { x: sx(0), y, width: bw, height: bh, rx: 2, fill: br.col, opacity: br.op }, svg);
        const isWin = (isLarge && k === 3) || (!isLarge && ((R.strip_ms <= R.global_ms && k === 3) || (R.global_ms < R.strip_ms && k === 2)));
        tx(svg, sx(br.v) + 4, y + bh - 1.2, br.lbl, { fill: isWin ? br.col : C.fg2, 'font-size': 9.5, 'font-weight': isWin ? 600 : 400 });
      });
    });
  };

  drawServerGroup(H1, top, '작은 모델 · d = 1,024 (실제 학습 가중치 · 층 24MB < L2)', false);
  const divY = top + blockH + grpGap / 2 - 2;
  el('line', { x1: 4, x2: W - 4, y1: divY, y2: divY, stroke: C.line2, 'stroke-width': 1, 'stroke-dasharray': '4 3' }, svg);
  drawServerGroup(H4, top + blockH + grpGap, '큰 모델 · d = 4,096 (대형 LLM 층 · 층 96MB > L2)', true);

  // 3) 전체 폭 통합 실측 표 (위 d=1,024 / 아래 d=4,096, 마지막 열 = 층별 배수 1/2/3/4층)
  const buildGroupRows = (H, grpTitle, isLarge) => {
    const b1 = H.b1;
    const hdr = `<tr class="grp"><td colspan="6">${grpTitle}</td></tr>`;
    const b1Row = `<tr class="b1-hi">
      <td><b>단일 (B=1)</b> <span class="mut">로컬</span></td>
      <td class="num">${b1.base_ms.toFixed(3)} ms <span class="mut">(1.00×)</span></td>
      <td class="num warn">${b1.mask0_ms.toFixed(3)} ms <span class="mut">(${b1.speedup_mask0.toFixed(2)}×)</span></td>
      <td class="num mut" style="text-align:center">— (타일링 불필요) —</td>
      <td class="num"><b style="color:var(--neg)">${b1.ours_ms.toFixed(3)} ms</b> <span class="bt-pill neg">${b1.speedup.toFixed(2)}배</span></td>
      <td class="num">${fmtLayers(b1.layer_speedups)}</td>
    </tr>`;
    const srvRows = H.rows.map(r => {
      const hi = r.B === 2048;
      const bestStrip = isLarge || r.strip_ms <= r.global_ms;
      return `<tr${hi ? ' class="sum"' : ''}>
        <td class="num"><b>서버 (${nf(r.B)})</b></td>
        <td class="num">${r.base_ms.toFixed(2)} ms <span class="mut">(1.00×)</span></td>
        <td class="num warn">${r.mask0_ms.toFixed(2)} ms <span class="mut">(${r.speedup_mask0.toFixed(2)}×)</span></td>
        <td class="num${!bestStrip ? ' ok' : ''}">${r.global_ms.toFixed(2)} ms ${!bestStrip ? `<span class="bt-pill">${r.speedup_global.toFixed(2)}배</span>` : `<span class="mut">(${r.speedup_global.toFixed(2)}×)</span>`}</td>
        <td class="num${bestStrip ? ' ok' : ''}">${r.strip_ms.toFixed(2)} ms ${bestStrip ? `<span class="bt-pill">${r.speedup_strip.toFixed(2)}배</span>` : `<span class="mut">(${r.speedup_strip.toFixed(2)}×)</span>`}</td>
        <td class="num">${fmtLayers(r.layer_speedups)}</td>
      </tr>`;
    }).join('');
    return hdr + b1Row + srvRows;
  };

  $('#bt-hwtbl').innerHTML =
    '<tr><th>환경 · 배치</th><th>풀 베이스 (cuBLAS)</th><th>타일 없음 (0 마스킹)</th><th>5단계 (캐싱 X · 전역)</th><th>최종 (단일 GEMV / 5단계+캐싱)</th><th>층별 배수 (1 / 2 / 3 / 4층)</th></tr>' +
    buildGroupRows(H1, '작은 모델 · d = 1,024 (실제 학습 가중치 · 층 24MB < L2 32MB · 대표 배수 = 4개 층 전체 합계)', false) +
    buildGroupRows(H4, '큰 모델 · d = 4,096 (대형 LLM 층 · 층 96MB > L2 32MB · 대표 배수 = 4개 층 전체 합계)', true);

  const maxErr = Math.max(
    H1.b1.max_rel_err, H4.b1.max_rel_err,
    ...H1.rows.map(r => r.max_rel_err), ...H4.rows.map(r => r.max_rel_err)
  );
  $('#bt-hw-kpi').innerHTML =
    `<div class="panel"><div class="k">밀집 64×32 커널 검증 (1단계 vs cuBLAS)</div><div class="v on" style="font-size:15px">d=1,024 ${H1.p1_eff_pct.toFixed(1)}% · d=4,096 ${H4.p1_eff_pct.toFixed(1)}% <span style="font-size:11.5px;color:var(--fg-3)">(전체 ${H1.dense_eff_pct.toFixed(1)}% / ${H4.dense_eff_pct.toFixed(1)}%)</span></div></div>
     <div class="panel"><div class="k">출력 정합성 (PyTorch fp32 기준)</div><div class="v" style="font-size:15px">최대 오차 ${(maxErr * 100).toFixed(2)}% <span style="font-size:11.5px;color:var(--fg-3)">(단일·서버 전 설정 bf16 통과)</span></div></div>`;
}

// ---- 보기 전환 (기존 버튼에 동작을 덧붙인다) ----
function drawAll() { drawProb(); drawJoint(); drawLayerStats(); drawCache(); drawFinal(); drawHwBench(); }
function show() {
  ['tab-train', 'tab-infer'].forEach(id => $('#' + id).setAttribute('aria-pressed', 'false'));
  $('#tab-batch').setAttribute('aria-pressed', 'true');
  $('#view-train').hidden = true; $('#view-infer').hidden = true; $('#view-batch').hidden = false;
  drawAll();
  try { history.replaceState(null, '', '#batch'); } catch (e) { location.hash = 'batch'; }
}
function hide() { $('#tab-batch').setAttribute('aria-pressed', 'false'); $('#view-batch').hidden = true; }
function init() {
  ['--bg', '--bg-2', '--bg-3', '--off', '--on', '--on-soft', '--pick', '--pick-soft', '--neg', '--fg', '--fg-2', '--fg-3', '--line', '--line-2']
    .forEach(n => C[n.slice(2).replace(/-/g, '')] = css(n));
  $('#tab-batch').addEventListener('click', show);
  $('#tab-train').addEventListener('click', hide); $('#tab-infer').addEventListener('click', hide);
  $('#bt-prev').onclick = () => { st.step = Math.max(0, st.step - 1); drawJoint(); };
  $('#bt-next').onclick = () => { st.step = Math.min(J_STEPS.length - 1, st.step + 1); drawJoint(); };
  let rt; new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (!$('#view-batch').hidden) drawAll(); }, 160); }).observe(document.body);
  if (window.__HASH0__ === '#batch') show();
}
(function wait(t0) {
  const ld = $('#loading');
  if (ld && ld.hidden) init(); else if (performance.now() - t0 < 15000) requestAnimationFrame(() => wait(t0)); else init();
})(performance.now());
})();
</script>''' + ME


def strip(s):
    return re.sub(re.escape(MS) + '.*?' + re.escape(ME), '', s, flags=re.S)


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
    base = unname(strip(page))
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
