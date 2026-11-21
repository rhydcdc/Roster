"""BATCHING - S7, S8: 고른 방법(뉴런 쪽 블록 + 블록 합치기)의 0을 곱하는 칸 줄이기. 읽는 양은 바닥값(뉴런마다 한 블록) 그대로.
S7: 토큰 쪽 조각을 ru개로 (텐서 코어 bf16 명령 16 x 8 x 16, 행렬곱을 '뉴런 x 토큰'으로 뒤집으면 토큰 쪽이 8).
    연산량 = c_ru(토큰 수) x c16(뉴런 수). 블록 짜기(나무 자르기, 합치기)도 이 비용으로. 베이스도 같은 규칙으로 다시 센다.
S8: S7의 블록에서 뉴런을 하나씩 다른 블록(또는 새 블록)으로 옮겨 칸이 줄면 옮긴다. 더 줄지 않을 때까지.
확인: ru = 16이면 batch_s.py s6의 '무리 전부'와 같은 수. 모든 계획에서 쓰는 (요청, 뉴런)이 정확히 한 번 덮이는지 센다.
python batch_s7.py s7 | s8
"""
import json
import os
import sys
import time

import numpy as np

HERE7 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE7)
from batch_s import load_keys, queues, QN, NN, RUNS_S                  # noqa: E402
from batch_n import neuron_tree, leaves                                 # noqa: E402


def cu(x, u):
    return -(-int(round(x)) // u) * u


def cuv(x, u):
    return np.ceil(np.asarray(x, dtype=np.float64) / u) * u


def cost_v(r, c, ru):
    return np.where(c > 0, cuv(r, ru) * cuv(c, 16), 0.0)


def cut_ru(masks, size, r, Z, n, ru):
    """batch_n.cut(상한 없음)과 같은 나무 위 최적 자르기, 비용만 c_ru(토큰) x c16(뉴런)."""
    B0 = len(size)
    if Z is None:
        return [0]
    nm, ns, best = [None] * (2 * B0 - 1), np.zeros(2 * B0 - 1), [None] * (2 * B0 - 1)
    for i in range(B0):
        nm[i], ns[i] = masks[i], size[i]
        best[i] = (cu(r[i], ru) * cu(size[i], 16), None)
    for t in range(B0 - 1):
        a, b, k = int(Z[t, 0]), int(Z[t, 1]), B0 + t
        nm[k] = nm[a] | nm[b]
        ns[k] = ns[a] + ns[b]
        merged = cu(float(nm[k] @ n), ru) * cu(ns[k], 16)
        sp = best[a][0] + best[b][0]
        best[k] = (merged, None) if merged <= sp else (sp, (a, b))
        nm[a] = nm[b] = None
    nodes, stack = [], [2 * B0 - 2]
    while stack:
        v = stack.pop()
        if v < B0 or best[v][1] is None:
            nodes.append(v)
        else:
            stack.extend(best[v][1])
    return nodes


def merge_ru(km, nc, n, ru):
    """batch_s.merge_blocks와 같은 쌍 합치기, 비용만 c_ru(토큰) x c16(뉴런)."""
    km, nc = km.copy(), nc.astype(np.float64).copy()
    alive, groups = np.ones(len(nc), bool), [[i] for i in range(len(nc))]
    while alive.sum() > 1:
        idx = np.nonzero(alive)[0]
        M = km[idx].astype(np.float64)
        r = M @ n
        ru_ = r[:, None] + r[None, :] - (M * n) @ M.T
        cs = cuv(r, ru) * cuv(nc[idx], 16)
        gain = cs[:, None] + cs[None, :] - cuv(ru_, ru) * cuv(nc[idx][:, None] + nc[idx][None, :], 16)
        np.fill_diagonal(gain, -1.0)
        a, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[a, b] <= 0:
            break
        i, j = idx[a], idx[b]
        km[i] |= km[j]
        nc[i] += nc[j]
        alive[j] = False
        groups[i] += groups[j]
    return [groups[i] for i in np.nonzero(alive)[0]]


def polish(blk, U, w, ru, max_sweeps=30, seed=0):
    """S8: 뉴런을 하나씩, 칸이 가장 많이 줄어드는 다른 블록(또는 새 블록)으로 옮긴다. 블록의 토큰 = 그 블록 뉴런을 쓰는 토큰 전부.
    blk: [(중심점 bool (P,), 뉴런 번호들)], U: (P, 4096) 중심점마다 쓰는 뉴런, w: 중심점마다 토큰 수. 반환: 블록들, 진단."""
    neurons = np.concatenate([cc for _, cc in blk])
    assign = np.concatenate([np.full(len(cc), j) for j, (_, cc) in enumerate(blk)])
    Pn = U[:, neurons].T.astype(np.float64)                            # (V, P) 뉴런마다 쓰는 중심점
    V, P = Pn.shape
    Bmax = len(blk) + V
    C, nc = np.zeros((Bmax, P)), np.zeros(Bmax)
    np.add.at(C, assign, Pn)
    np.add.at(nc, assign, 1.0)
    KM = (C > 0).astype(np.float64)
    r = KM @ w
    cb = cost_v(r, nc, ru)
    nb = len(blk)
    before = float(cb.sum())
    rng = np.random.default_rng(seed)
    sweeps = moves = 0
    for sweeps in range(1, max_sweeps + 1):
        moved = 0
        for i in rng.permutation(V):
            a, p = assign[i], Pn[i]
            Ca = C[a] - p
            ka = (Ca > 0).astype(np.float64)
            ra, nca = float(ka @ w), nc[a] - 1
            d_rem = float(cost_v(ra, nca, ru)) - cb[a]
            cand = cost_v(r[:nb] + (1.0 - KM[:nb]) @ (p * w), nc[:nb] + 1, ru) - cb[:nb]
            cand[a] = np.inf
            cand[nc[:nb] == 0] = np.inf
            j = int(np.argmin(cand))
            best = cand[j]
            newc = cu(float(p @ w), ru) * 16.0
            if newc < best:
                best, j = newc, -1
            if d_rem + best < -1e-9:
                if j == -1:
                    j, nb = nb, nb + 1
                C[a], KM[a], r[a], nc[a] = Ca, ka, ra, nca
                cb[a] = float(cost_v(ra, nca, ru))
                C[j] += p
                KM[j] = (C[j] > 0)
                r[j] = float(KM[j] @ w)
                nc[j] += 1
                cb[j] = float(cost_v(r[j], nc[j], ru))
                assign[i] = j
                moved += 1
        moves += moved
        if moved == 0:
            break
    out = [(KM[j] > 0, neurons[assign == j]) for j in np.unique(assign)]
    return out, dict(sweeps=sweeps, moves=moves, before=before, after=float(cb[:nb].sum()))


def plan_chosen(kq, BM_l, ru, do_polish=False):
    """고른 방법: 대기열 전부를 한 무리로, 뉴런 쪽 블록(나무 자르기) + 블록 합치기 [+ S8 옮기기]. 비용은 c_ru(토큰) x c16(뉴런)."""
    P, inv_r, cnt = np.unique(kq, return_inverse=True, return_counts=True)
    n = cnt.astype(np.float64)
    U = BM_l[P]
    masks, size, r, Z, inv, cols = neuron_tree(U, n)
    nodes = cut_ru(masks, size, r, Z, n, ru)
    B0 = len(size)
    blk = []
    for v in nodes:
        lv = leaves(Z, B0, v) if Z is not None else [0]
        blk.append((masks[lv].any(0), cols[np.isin(inv, lv)]))
    if len(blk) > 1:
        km = np.stack([b[0] for b in blk])
        gs = merge_ru(km, np.array([len(b[1]) for b in blk]), n, ru)
        blk = [(km[g].any(0), np.concatenate([blk[i][1] for i in g])) for g in gs]
    diag = {}
    if do_polish:
        blk, diag = polish(blk, U, n, ru)
    rows = np.arange(len(kq))
    return [(rows[km[inv_r]], cc) for km, cc in blk], diag


def evaluate_ru(plan, Mq, ru):
    comp = reads = dense = visits = nrect = 0
    cnt = np.zeros(Mq.shape, np.int16)
    for rows, cols in plan:
        if len(rows) == 0 or len(cols) == 0:
            continue
        comp += cu(len(rows), ru) * cu(len(cols), 16)
        reads += len(cols)
        dense += len(rows) * len(cols)
        visits += len(rows)
        nrect += 1
        cnt[np.ix_(rows, cols)] += 1
    ones = int(Mq.sum())
    return dict(compute=comp, reads=reads, rects=nrect, ones=ones, dense=dense, visits=visits / Mq.shape[0],
                ok=bool((cnt[Mq] == 1).all()))


def bases_ru(kq, BM_l, ru):
    fc = fr = sc = sr = 0
    for s in range(0, len(kq), 16):
        ch = kq[s:s + 16]
        u = int(BM_l[np.unique(ch)].any(0).sum())
        fc += cu(len(ch), ru) * NN
        fr += NN
        sc += cu(len(ch), ru) * cu(u, 16)
        sr += u
    return dict(full=(fc, fr), seq=(sc, sr))


def run(tag, methods):
    """methods: [(이름, ru, polish)]. 방법마다 같은 규칙(ru)으로 센 베이스와 견준다."""
    keys, BM, pos = load_keys()
    out, T0, allok = {}, time.perf_counter(), True
    rus = sorted({m[1] for m in methods})
    acc = {}
    for N, t, order in queues(keys.shape[1]):
        if t == 0:
            acc = {m[0]: dict(compute=0, reads=0, rects=0, ones=0, dense=0, visits=0.0, ms=0.0, sweeps=0, moves=0) for m in methods}
            acc.update({('full', u): [0, 0] for u in rus})
            acc.update({('seq', u): [0, 0] for u in rus})
        for l in range(keys.shape[0]):
            kq = keys[l][order]
            Mq = BM[l][kq]
            for u in rus:
                b = bases_ru(kq, BM[l], u)
                for k in ('full', 'seq'):
                    acc[(k, u)][0] += b[k][0]
                    acc[(k, u)][1] += b[k][1]
            for name, ru, pol in methods:
                t0 = time.perf_counter()
                plan, dg = plan_chosen(kq, BM[l], ru, pol)
                ms = 1000 * (time.perf_counter() - t0)
                e = evaluate_ru(plan, Mq, ru)
                allok &= e['ok']
                a = acc[name]
                for f in ('compute', 'reads', 'rects', 'ones', 'dense', 'visits'):
                    a[f] += e[f]
                a['ms'] += ms
                a['sweeps'] += dg.get('sweeps', 0)
                a['moves'] += dg.get('moves', 0)
        print('[%s] 대기열 %d개 %d/%d 끝, %.0f s' % (tag, N, t + 1, QN[N], time.perf_counter() - T0), flush=True)
        if t == QN[N] - 1:
            T, L = QN[N], keys.shape[0]
            res = {}
            for u in rus:
                res['base_ru%d' % u] = dict(full_compute=acc[('full', u)][0] / T, full_reads=acc[('full', u)][1] / T,
                                            seq_compute=acc[('seq', u)][0] / T, seq_reads=acc[('seq', u)][1] / T)
            print('\n=== %s 대기열 %d개 (%d번, 층 4개 합) | 덮기 정확히 한 번 %s ===' % (tag, N, T, 'OK' if allok else '어긋남!'), flush=True)
            for u in rus:
                b = res['base_ru%d' % u]
                print('토큰 쪽 %2d 단위로 센 베이스: 완전 베이스 연산량 %.0f, 읽는 양 %.0f | 줄인 베이스 연산량 %.0f, 읽는 양 %.0f' % (
                    u, b['full_compute'], b['full_reads'], b['seq_compute'], b['seq_reads']), flush=True)
            print('%-26s %10s %8s | %-15s | %-15s | %-22s | %6s %6s %6s %6s %8s' % (
                '방법', '연산량', '읽는 양', '완전 베이스의 %', '줄인 베이스의 %', '쓰는/0곱/빈칸 (완전 %)', '블록/층', '방문', '훑기', '옮김/층', 'ms/층'))
            for name, ru, pol in methods:
                a = acc[name]
                Fc, Fr = acc[('full', ru)]
                Sc, Sr = acc[('seq', ru)]
                ones_p = 100 * a['ones'] / Fc
                zero_p = 100 * (a['dense'] - a['ones']) / Fc
                pad_p = 100 * (a['compute'] - a['dense']) / Fc
                r = dict(ru=ru, polish=pol, compute=a['compute'] / T, reads=a['reads'] / T,
                         pct_full=(100 * a['compute'] / Fc, 100 * a['reads'] / Fr),
                         pct_seq=(100 * a['compute'] / Sc, 100 * a['reads'] / Sr),
                         parts_full=(ones_p, zero_p, pad_p), rects_per_layer=a['rects'] / T / L,
                         visits_per_req_layer=a['visits'] / T / L, ms_per_layer=a['ms'] / T / L,
                         sweeps_per_layer=a['sweeps'] / T / L, moves_per_layer=a['moves'] / T / L)
                res[name] = r
                print('%-26s %10.0f %8.0f | %6.1f%% / %5.1f%% | %6.1f%% / %5.1f%% | %5.1f / %5.1f / %5.1f | %6.1f %6.1f %6.1f %6.0f %8.0f' % (
                    name, r['compute'], r['reads'], *r['pct_full'], *r['pct_seq'], *r['parts_full'], r['rects_per_layer'],
                    r['visits_per_req_layer'], r['sweeps_per_layer'], r['moves_per_layer'], r['ms_per_layer']), flush=True)
            out[N] = res
    with open(os.path.join(RUNS_S, 'batch_%s.json' % tag), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 's7':
        run('s7', [('chosen_ru16', 16, False), ('chosen_ru8', 8, False)])
    elif sys.argv[1] == 's8':
        run('s8', [('chosen_ru8', 8, False), ('chosen_ru8_polish', 8, True)])
