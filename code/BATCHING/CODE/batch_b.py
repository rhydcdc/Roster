"""BATCHING - B: 합칠지 말지. 기준은 '합치면 비용이 줄어드는가' (비용 = 계산량 + c x 꺼내는 가중치, c = 113.8).
묶음 하나의 비용 = (묶음의 요청 수 x 묶음 안 중심점들의 목록 합집합) + c x 합집합.
합친 묶음에서 각 요청은 자기 목록에 없는 뉴런의 출력에 0을 곱한다 -> 출력은 따로 계산한 것과 같다.

B1 나무 자르기: 층마다 중심점 256개를 목록의 겹침(자카드 거리, 평균 연결)으로 미리 나무로 묶고,
   묶을 때마다 나무를 아래에서 위로 훑으며 '합친 비용'과 '나눠 둔 비용' 가운데 작은 쪽을 고른다 (나무 위의 최적 자르기).
B2 욕심쟁이 합치기: 지금 묶음들 가운데 합쳤을 때 비용이 가장 많이 주는 쌍부터 합치고, 더 줄지 않으면 멈춘다 (기준선).
A의 넷(다 켬, 요청마다 따로, 중심점별 묶음, 전부 한 묶음)도 같은 요청 묶음으로 다시 셈한다.
python batch_b.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from batch_a import *                                                  # noqa: E402,F401 (ex2lib, keys_all, MODEL, TABLE)
HERE_B = _HERE
import numpy as np                                                     # noqa: E402
from scipy.cluster.hierarchy import linkage                            # noqa: E402
from scipy.spatial.distance import pdist                               # noqa: E402

NS_B = [1, 2, 4, 8, 16, 32, 64, 100, 256, 1000]
TRIALS_B = {n: (200 if n <= 100 else 50) for n in NS_B}
CR = C_READ


def build_tree(bm):
    """중심점들의 목록(켜는 뉴런 bool (256, 4096))을 겹침으로 묶은 나무 (scipy linkage)."""
    return linkage(pdist(bm, metric='jaccard'), method='average')


def tree_cut(Z, bm, K, cnt):
    """나무 위의 최적 자르기. 반환 (비용, 계산량, 꺼내는 양, 묶음 수)."""
    M = bm.shape[0]
    ub = [None] * (2 * M - 1)
    best = [(0.0, 0.0, 0.0, 0)] * (2 * M - 1)
    nreq = np.zeros(2 * M - 1)
    for j in np.nonzero(cnt)[0]:
        s = float(K[j])
        ub[j], nreq[j] = bm[j], cnt[j]
        best[j] = (cnt[j] * s + CR * s, cnt[j] * s, s, 1)
    for i in range(M - 1):
        a, b, k = int(Z[i, 0]), int(Z[i, 1]), M + i
        n = nreq[a] + nreq[b]
        if n == 0:
            continue
        nreq[k] = n
        u = ub[b] if ub[a] is None else (ub[a] if ub[b] is None else ub[a] | ub[b])
        ub[k] = u
        s = float(u.sum())
        merged = (n * s + CR * s, n * s, s, 1)
        split = tuple(x + y for x, y in zip(best[a], best[b]))
        best[k] = merged if merged[0] <= split[0] else split
    return best[2 * M - 2]


@torch.no_grad()
def greedy(bm_t, cnt_t):
    """욕심쟁이 합치기 (GPU). bm_t: 지금 요청이 있는 중심점들의 목록 (G, 4096) float, cnt_t: 요청 수 (G,)."""
    Gb, n = bm_t.clone(), cnt_t.clone()
    s = Gb.sum(1)
    while Gb.shape[0] > 1:
        inter = Gb @ Gb.T
        uni = s[:, None] + s[None, :] - inter
        cost = n * s + CR * s
        delta = (n[:, None] + n[None, :]) * uni + CR * uni - cost[:, None] - cost[None, :]
        delta.fill_diagonal_(float('inf'))
        v, idx = delta.view(-1).min(0)
        if v.item() >= 0:
            break
        i, j = divmod(int(idx), Gb.shape[0])
        Gb[i] = torch.maximum(Gb[i], Gb[j])
        n[i] += n[j]
        s[i] = Gb[i].sum()
        keep = torch.ones(Gb.shape[0], dtype=torch.bool, device=Gb.device)
        keep[j] = False
        Gb, n, s = Gb[keep], n[keep], s[keep]
    comp, reads = float((n * s).sum()), float(s.sum())
    return comp + CR * reads, comp, reads, int(Gb.shape[0])


def main():
    D = load_data()
    model = load_net(MODEL)
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    keys = keys_all(model, D['xte'], C, NR, KJ)
    BMt = [(NR[l] < KJ[l][:, None]) for l in range(N_BLOCK)]
    BM = [b.cpu().numpy() for b in BMt]
    Kn = [k.cpu().numpy().astype(np.float64) for k in KJ]
    trees = [build_tree(BM[l]) for l in range(N_BLOCK)]
    keys_np = [k.cpu().numpy() for k in keys]
    Nte = D['xte'].shape[0]
    rng = np.random.default_rng(0)
    rows = []
    for n in NS_B:
        T = TRIALS_B[n]
        tot = {m: np.zeros(3) for m in ('dense', 'each', 'group', 'union', 'tree', 'greedy')}   # 계산량, 꺼내는 양, 묶음 수 (층 합)
        t_tree = 0.0
        for _ in range(T):
            idx = rng.choice(Nte, n, replace=False)
            for l in range(N_BLOCK):
                j = keys_np[l][idx]
                cnt = np.bincount(j, minlength=256).astype(np.float64)
                u = np.nonzero(cnt)[0]
                Ku = Kn[l][u]
                uni = float(BM[l][u].any(0).sum())
                tot['dense'] += (n * N_NEURON, N_NEURON, 1)
                tot['each'] += (Kn[l][j].sum(), Kn[l][j].sum(), n)
                tot['group'] += ((cnt[u] * Ku).sum(), Ku.sum(), len(u))
                tot['union'] += (n * uni, uni, 1)
                t0 = time.perf_counter()
                _, cp, rd, g = tree_cut(trees[l], BM[l], Kn[l], cnt)
                t_tree += time.perf_counter() - t0
                tot['tree'] += (cp, rd, g)
                _, cp, rd, g = greedy(BMt[l][torch.as_tensor(u, device=DEV)].float(), torch.as_tensor(cnt[u], device=DEV, dtype=torch.float32))
                tot['greedy'] += (cp, rd, g)
        d = tot['dense']
        cost = {m: v[0] + CR * v[1] for m, v in tot.items()}
        row = dict(n=n, trials=T,
                   cost={m: round(cost['dense'] / cost[m], 3) for m in tot},
                   compute={m: round(d[0] / v[0], 3) for m, v in tot.items()},
                   reads={m: round(d[1] / v[1], 3) for m, v in tot.items()},
                   groups={m: round(v[2] / T / N_BLOCK, 2) for m, v in tot.items()},
                   tree_ms=round(1000 * t_tree / T / N_BLOCK, 3))
        rows.append(row)
        print('N=%4d  비용 배수: 따로 %.2f | 중심점별 %.2f | 한 묶음 %.2f | 나무 자르기 %.2f (층마다 묶음 %.1f) | 욕심쟁이 %.2f (묶음 %.1f) | 나무 자르기 고르는 시간 %.2f ms/층' % (
            n, row['cost']['each'], row['cost']['group'], row['cost']['union'], row['cost']['tree'], row['groups']['tree'],
            row['cost']['greedy'], row['groups']['greedy'], row['tree_ms']), flush=True)
    os.makedirs(os.path.join(HERE_B, 'runs'), exist_ok=True)
    jdump(dict(c=CR, rows=rows), os.path.join(HERE_B, 'runs', 'batch_b.json'))


if __name__ == '__main__':
    main()
