"""레시피 HTML의 '묶어서 계산' 보기에 넣을 데이터: 고른 방법(뉴런 쪽 블록 + 블록 합치기, 16 단위)만.
  real: 실제 대기열 하나(토큰 256개, 씨앗 0의 첫 대기열)에서 층마다 실제 블록. 토큰(줄)은 간 중심점의 겹침 순서, 블록은 토큰이 많은 것부터.
        블록마다: 토큰 비트(256), 뉴런 수, (토큰 x 뉴런) 쓰는지 비트. 층마다: 블록 수, 연산량(16 단위), 읽는 양, 쓰는 칸 / 0을 곱하는 칸 / 빈칸.
  results: 대기열 100 / 256 / 1,000개(같은 대기열들 평균)의 연산량과 읽는 양 (runs/batch_s7.json의 chosen_ru16 = batch_s의 무리 전부와 같은 수).
python viz_batch_data.py  -> runs/viz_batch.json
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batch_s import load_keys, queues                                  # noqa: E402
from batch_s7 import plan_chosen, evaluate_ru, bases_ru               # noqa: E402


def b64bits(a):
    return base64.b64encode(np.packbits(np.asarray(a, bool).ravel()).tobytes()).decode()


def main():
    keys, BM, pos = load_keys()
    N, t, order = [q for q in queues(keys.shape[1]) if q[0] == 256][0]
    real = []
    for l in range(4):
        kq = keys[l][order]
        Mq = BM[l][kq]
        rank = np.empty(N, np.int64)
        rank[np.argsort(pos[l][kq], kind='stable')] = np.arange(N)     # 토큰 -> 보이는 줄
        plan, _ = plan_chosen(kq, BM[l], 16)
        e = evaluate_ru(plan, Mq, 16)
        b = bases_ru(kq, BM[l], 16)
        blocks = []
        for rows, cols in plan:
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            rr = np.sort(rank[rows])
            inv = np.argsort(rank)                                     # 보이는 줄 -> 토큰
            toks = inv[rr]
            U = Mq[np.ix_(toks, cols)]
            co = np.argsort(-U.sum(0), kind='stable')                  # 블록 안 뉴런: 쓰는 토큰이 많은 것부터
            U = U[:, co]
            rmask = np.zeros(N, bool)
            rmask[rr] = True
            blocks.append(dict(r=b64bits(rmask), m=int(len(rr)), n=int(len(cols)), u=b64bits(U), ones=int(U.sum()),
                               cells=int(-(-len(rr) // 16) * 16 * -(-len(cols) // 16) * 16)))
        blocks.sort(key=lambda x: (-x['m'], -x['n']))
        real.append(dict(blocks=blocks, nblocks=len(blocks), compute=int(e['compute']), reads=int(e['reads']), ones=int(e['ones']),
                         dense=int(e['dense']), full_compute=int(b['full'][0]), full_reads=int(b['full'][1]),
                         visits=round(e['visits'], 1), ncols=int(sum(x['n'] for x in blocks))))
        print('층 %d: 블록 %d개, 연산량 %d (기본 풀 모델의 %.1f%%), 읽는 양 %d (%.1f%%), 쓰는 칸 %.1f%%' % (
            l + 1, len(blocks), e['compute'], 100 * e['compute'] / b['full'][0], e['reads'], 100 * e['reads'] / b['full'][1],
            100 * e['ones'] / b['full'][0]))
    s7 = json.load(open(os.path.join(HERE, 'runs', 'batch_s7.json'), encoding='utf-8'))
    res = {}
    for n in ('100', '256', '1000'):
        r = s7[n]['chosen_ru16']
        res[n] = dict(compute=round(r['pct_full'][0], 1), reads=round(r['pct_full'][1], 1),
                      compute_seq=round(r['pct_seq'][0], 1), reads_seq=round(r['pct_seq'][1], 1),
                      ones=round(r['parts_full'][0], 1), zero=round(r['parts_full'][1], 1), pad=round(r['parts_full'][2], 1))
        print('대기열 %s개: 연산량 %.1f%% / 읽는 양 %.1f%% (기본 풀 모델), %.1f%% / %.1f%% (줄인 모델), 쓰는 %.1f / 0곱 %.1f / 빈칸 %.1f' % (
            n, res[n]['compute'], res[n]['reads'], res[n]['compute_seq'], res[n]['reads_seq'], res[n]['ones'], res[n]['zero'], res[n]['pad']))
    out = dict(N=N, real=real, results=res)
    path = os.path.join(HERE, 'runs', 'viz_batch.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print('%s %.0f KB' % (path, os.path.getsize(path) / 1024))


if __name__ == '__main__':
    main()
