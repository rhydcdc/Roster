"""BATCHING - N: 뉴런 쪽으로 묶기. 비용 모형 = 연산량(16행 단위로 올림)과 읽는 양 (사용자, 2026-11-22).
뉴런마다 그 뉴런을 쓰는 요청들이 있다. 쓰는 요청들이 비슷한 뉴런끼리 블록으로 묶고, 블록마다 그 요청들만 모아 계산한다.
  블록 하나: 연산량 = c16(블록의 요청 수) x c16(블록의 뉴런 수), 읽는 양 = 블록의 뉴런 수. 모든 뉴런은 한 블록에만 -> 읽는 양 = 합집합.
  처음 블록 = 쓰는 중심점들의 모임이 똑같은 뉴런끼리. 블록끼리 거리 = 쓰는 요청의 겹침(요청 수 무게 자카드), 평균 연결로 나무,
  나무 위에서 연산량이 가장 작은 자르기 (블록 수 상한 없음 / 32 / 8).
견줌(같은 대기열, 16 단위):
  완전 베이스(안 줄인 모델)와 줄인 베이스(Lc6)를 들어온 순서대로 g = 16, 64, 대기열 전부씩.
  요청 쪽: 전부 한 묶음(합집합), 겹침 무리(D2, 묶음 수 = N/16).
python batch_n.py
"""
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from batch_b import *                                                  # noqa: E402,F401
from scipy.cluster.hierarchy import fcluster                           # noqa: E402
from scipy.spatial.distance import squareform                          # noqa: E402
HERE_N = _HERE

T16 = 16
QN = {100: 20, 256: 10, 1000: 5}
LIMS = [None, 32, 8]
BASE_G = [16, 64, 'N']


def c16(x):
    return math.ceil(x / T16) * T16


def neuron_tree(U, n):
    """U: 요청이 간 중심점들의 목록 (P, 4096) bool, n: 중심점마다 요청 수 (P,).
    반환: 처음 블록들(뉴런 수, 요청 수, 쓰는 중심점 bool), 나무, 뉴런마다 처음 블록 번호, 합집합 뉴런 번호."""
    cols = np.nonzero(U.any(0))[0]
    Mv = U[:, cols].T                                                  # (V, P) 뉴런마다 쓰는 중심점
    uniq, inv, size = np.unique(np.packbits(Mv, axis=1), axis=0, return_inverse=True, return_counts=True)
    masks = np.unpackbits(uniq, axis=1)[:, :U.shape[0]].astype(bool)  # (B0, P)
    r = masks.astype(np.float64) @ n
    B0 = len(size)
    if B0 == 1:
        return masks, size.astype(np.float64), r, None, inv.reshape(-1), cols
    Wm = (masks * n[None, :]).astype(np.float32)
    inter = Wm @ masks.T.astype(np.float32)
    uni = r[:, None] + r[None, :] - inter
    Dm = 1.0 - inter / np.maximum(uni, 1e-9)
    np.fill_diagonal(Dm, 0.0)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    Z = linkage(squareform(Dm, checks=False), method='average')
    return masks, size.astype(np.float64), r, Z, inv.reshape(-1), cols


def cut(masks, size, r, Z, n, lim):
    """나무 위에서 연산량(16 단위)이 가장 작은 자르기. lim: 블록 수 상한(None = 없음). 반환 (연산량, 블록 수, 고른 마디들)."""
    B0 = len(size)
    if Z is None:
        return c16(r[0]) * c16(size[0]), 1, [0]
    L = lim or 0
    nm, ns, nr = [None] * (2 * B0 - 1), np.zeros(2 * B0 - 1), np.zeros(2 * B0 - 1)
    best = [None] * (2 * B0 - 1)
    for i in range(B0):
        nm[i], ns[i], nr[i] = masks[i], size[i], r[i]
        cst = c16(r[i]) * c16(size[i])
        if lim is None:
            best[i] = (cst, 1, None)
        else:
            dp = np.full(L + 1, np.inf)
            dp[1] = cst
            best[i] = dp
    for t in range(B0 - 1):
        a, b, k = int(Z[t, 0]), int(Z[t, 1]), B0 + t
        nm[k] = nm[a] | nm[b]
        ns[k] = ns[a] + ns[b]
        nr[k] = float(nm[k] @ n)
        merged = c16(nr[k]) * c16(ns[k])
        if lim is None:
            sp = best[a][0] + best[b][0]
            best[k] = (merged, 1, None) if merged <= sp else (sp, best[a][1] + best[b][1], (a, b))
        else:
            da, db = best[a], best[b]
            dp = np.full(L + 1, np.inf)
            dp[1] = merged
            for x in range(1, L):
                if np.isfinite(da[x]):
                    dp[x + 1:] = np.minimum(dp[x + 1:], da[x] + db[1:L + 1 - x])
            best[k] = dp
        nm[a] = nm[b] = None                                           # 메모리
    root = 2 * B0 - 2
    if lim is None:
        nodes, stack = [], [root]
        while stack:                                                   # 고른 마디들 (뒤로 따라가기)
            v = stack.pop()
            if v < B0 or best[v][2] is None:
                nodes.append(v)
            else:
                stack.extend(best[v][2])
        return best[root][0], best[root][1], nodes
    k = int(np.argmin(best[root][1:])) + 1
    return float(best[root][k]), k, None


def leaves(Z, B0, v):
    out, st = [], [v]
    while st:
        x = st.pop()
        if x < B0:
            out.append(x)
        else:
            st += [int(Z[x - B0, 0]), int(Z[x - B0, 1])]
    return out


@torch.no_grad()
def verify(model, D, C, NR, KJ, BMt, order, l):
    """한 대기열, 층 l에서 뉴런 블록으로 실제 계산해 가린 계산과 같은지 본다 (bf16)."""
    m = load_net(MODEL).to(torch.bfloat16)
    F_ = torch.nn.functional
    xb = D['xte'][torch.as_tensor(order, device=DEV)]
    h = m.inp(xb.reshape(len(order), -1).to(torch.bfloat16)).float()
    for ll, blk in enumerate(m.blocks):
        with torch.autocast('cuda', enabled=False):
            j = nearest(lkey(h), C[ll], 1)[:, 0]
        own = (NR[ll][j] < KJ[ll][j][:, None]).to(torch.bfloat16)
        z = blk.norm(h.to(torch.bfloat16))
        ref = F_.linear(F_.silu(blk.gate(z)) * blk.up(z) * own, blk.down.weight)
        if ll == l:
            jn = j.cpu().numpy()
            P, cnt = np.unique(jn, return_counts=True)
            U = BMt[ll][torch.as_tensor(P, device=DEV)].cpu().numpy()
            masks, size, r, Z, inv, cols = neuron_tree(U, cnt.astype(np.float64))
            _, nb, nodes = cut(masks, size, r, Z, cnt.astype(np.float64), None)
            y = torch.zeros_like(ref, dtype=torch.float32)
            B0 = len(size)
            for v in nodes:
                lv = leaves(Z, B0, v) if Z is not None else [0]
                neu = torch.as_tensor(cols[np.isin(inv, lv)], device=DEV)
                users = np.isin(jn, P[masks[lv].any(0)])
                rows = torch.as_tensor(np.nonzero(users)[0], device=DEV)
                zg = z.index_select(0, rows)
                a = F_.silu(zg @ blk.gate.weight.index_select(0, neu).T) * (zg @ blk.up.weight.index_select(0, neu).T)
                a = a * own.index_select(0, rows).index_select(1, neu)
                y.index_add_(0, rows, (a @ blk.down.weight.index_select(1, neu).T).float())
            return nb, float((y - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6))
        h = h + ref.float()


def main():
    D = load_data()
    model = load_net(MODEL)
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    keys = [k.cpu().numpy() for k in keys_all(model, D['xte'], C, NR, KJ)]
    BMt = [(NR[l] < KJ[l][:, None]) for l in range(N_BLOCK)]
    BM = [b.cpu().numpy() for b in BMt]
    rng = np.random.default_rng(0)
    out = {}
    for N, T in QN.items():
        names = ([f'full_g{g}' for g in BASE_G] + [f'seq_g{g}' for g in BASE_G] + ['union', 'reqclust16'] +
                 [f'neuron_{lim or "all"}' for lim in LIMS])
        acc = {k: np.zeros(3) for k in names}                          # 연산량(16 단위), 읽는 양, 블록/묶음 수
        ideal, t_build = 0.0, 0.0
        for _ in range(T):
            order = rng.choice(D['xte'].shape[0], N, replace=False)
            for l in range(N_BLOCK):
                kq = keys[l][order]
                for g in BASE_G:
                    gg = N if g == 'N' else g
                    for s in range(0, N, gg):
                        ch = kq[s:s + gg]
                        uni = float(BM[l][np.unique(ch)].any(0).sum())
                        acc[f'seq_g{g}'] += (c16(len(ch)) * c16(uni), uni, 1)
                        acc[f'full_g{g}'] += (c16(len(ch)) * N_NEURON, N_NEURON, 1)
                P, cnt = np.unique(kq, return_counts=True)
                n = cnt.astype(np.float64)
                U = BM[l][P]
                uni = float(U.any(0).sum())
                acc['union'] += (c16(N) * c16(uni), uni, 1)
                m = int(math.ceil(N / 16))                             # 요청 쪽 겹침 무리 (D2), 묶음 수 = N / 16
                lab = np.arange(len(P)) if len(P) <= m else fcluster(linkage(pdist(U, metric='jaccard'), method='average'), m, criterion='maxclust')
                for c in np.unique(lab):
                    sel = lab == c
                    u2 = float(U[sel].any(0).sum())
                    acc['reqclust16'] += (c16(n[sel].sum()) * c16(u2), u2, 1)
                t0 = time.perf_counter()
                masks, size, r, Z, inv, cols = neuron_tree(U, n)
                t_build += time.perf_counter() - t0
                ideal += float((r * size).sum())
                for lim in LIMS:
                    cp, nb, _ = cut(masks, size, r, Z, n, lim)
                    acc[f'neuron_{lim or "all"}'] += (cp, uni, nb)
        res = {k: dict(compute=v[0] / T, reads=v[1] / T, blocks=round(v[2] / T / N_BLOCK, 1)) for k, v in acc.items()}
        res['ideal_compute'] = ideal / T
        out[N] = dict(trials=T, res=res, build_ms=round(1000 * t_build / T / N_BLOCK, 1))
        print('\n=== 대기열 %d개 (%d번 평균, 층 4개 합, 16 단위) ===' % (N, T), flush=True)
        for k in names:
            v = res[k]
            print('%-14s 연산량 %10.0f  읽는 양 %8.0f  묶음/블록 %6.1f | 완전 베이스(g=16)/이것: 연산량 %.2f배, 읽는 양 %.2f배 | 줄인 베이스(g=16)/이것: 연산량 %.2f배, 읽는 양 %.2f배' % (
                k, v['compute'], v['reads'], v['blocks'], res['full_g16']['compute'] / v['compute'], res['full_g16']['reads'] / v['reads'],
                res['seq_g16']['compute'] / v['compute'], res['seq_g16']['reads'] / v['reads']), flush=True)
        print('이상적 연산량(16 단위 올림 없이, 요청마다 자기 뉴런만) %.0f | 뉴런 나무 만드는 시간 %.1f ms/층' % (ideal / T, out[N]['build_ms']), flush=True)
    order = rng.choice(D['xte'].shape[0], 100, replace=False)
    nb, err = verify(model, D, C, NR, KJ, BMt, order, 0)
    out['verify'] = dict(queue=100, layer=1, blocks=nb, max_rel_err=err)
    print('\n확인: 대기열 100개, 층 1을 뉴런 블록 %d개로 실제 계산 -> 가린 계산과의 차이(최대값 대비) %.1e' % (nb, err), flush=True)
    jdump(out, os.path.join(HERE_N, 'runs', 'batch_n.json'))


if __name__ == '__main__':
    main()
