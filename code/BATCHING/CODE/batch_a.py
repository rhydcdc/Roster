"""BATCHING - A: 확정 구조(Lc6) 그대로, 요청 N개를 층마다 다시 묶을 때의 계산량, 꺼내는 가중치, 비용(c = 113.8).
python batch_a.py
"""
import os
import sys

_HERE3 = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE3))                       # BATCHING/CODE -> 프로젝트 맨 위
sys.path.insert(0, os.path.join(_PROJ, 'AUTO', 'EXP2', 'CODE'))
from ex2lib import *                                                   # noqa: E402
PROJ, HERE3 = _PROJ, _HERE3                                            # 가져온 뒤에 다시 정한다 (HANDOFF 함정 5)

E2RUNS = os.path.join(PROJ, 'AUTO', 'EXP2', 'CODE', 'runs')
MODEL = os.path.join(E2RUNS, 'Lc6_lr9.49367e-07', 'best.pt')
TABLE = os.path.join(E2RUNS, 'rules', 'lv_Lc6.pt')
NS = [1, 2, 4, 8, 16, 32, 64, 100, 256, 1000]
TRIALS = {n: (400 if n <= 100 else 100) for n in NS}
C_R = C_READ                                                           # 113.8


@torch.no_grad()
def keys_all(model, X, C, NR, KJ):
    """시험 사진마다, 층마다 간 중심점 (고른 뉴런만 계산하며 앞으로 간다)."""
    out = [[] for _ in range(N_BLOCK)]
    for s in range(0, X.shape[0], 2500):
        xb = X[s:s + 2500]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                j = nearest(lkey(h), C[l], 1)[:, 0]
            out[l].append(j)
            on = NR[l][j] < KJ[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                h = h + blk.down(a * on.to(a.dtype)).float()
    return [torch.cat(o) for o in out]


def main():
    torch.manual_seed(0)
    D = load_data()
    model = load_net(MODEL)
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    keys = keys_all(model, D['xte'], C, NR, KJ)
    BM = [(NR[l] < KJ[l][:, None]) for l in range(N_BLOCK)]          # 중심점마다 켜는 뉴런 (256, 4096)
    Nte = D['xte'].shape[0]
    g = torch.Generator(device=DEV).manual_seed(0)
    res = []
    for n in NS:
        acc = {m: [0.0, 0.0] for m in ('dense', 'each', 'group', 'union')}   # [계산량, 꺼내는 양] 층 합
        groups, per_group = [0.0] * N_BLOCK, [0.0] * N_BLOCK
        T = TRIALS[n]
        for _ in range(T):
            idx = torch.randperm(Nte, device=DEV, generator=g)[:n]
            for l in range(N_BLOCK):
                j = keys[l][idx]
                k = KJ[l][j].float()
                u, cnt = torch.unique(j, return_counts=True)
                Ku = KJ[l][u].float()
                uni = BM[l][u].any(0).sum().float()
                acc['dense'][0] += n * N_NEURON
                acc['dense'][1] += N_NEURON
                acc['each'][0] += k.sum().item()
                acc['each'][1] += k.sum().item()
                acc['group'][0] += (cnt.float() * Ku).sum().item()
                acc['group'][1] += Ku.sum().item()
                acc['union'][0] += n * uni.item()
                acc['union'][1] += uni.item()
                groups[l] += len(u)
                per_group[l] += n / len(u)
        d = acc['dense']
        cost = {m: v[0] + C_R * v[1] for m, v in acc.items()}
        row = dict(n=n, trials=T,
                   compute={m: round(d[0] / v[0], 3) for m, v in acc.items()},
                   reads={m: round(d[1] / v[1], 3) for m, v in acc.items()},
                   cost={m: round(cost['dense'] / cost[m], 3) for m in acc},
                   groups=[round(x / T, 2) for x in groups], per_group=[round(x / T, 3) for x in per_group],
                   union_pct=None)
        res.append(row)
        print('N=%4d  비용이 다 켤 때보다 줄어드는 배수  요청마다 따로 %.2f  중심점별 묶음 %.2f  전부 한 묶음 %.2f | '
              '계산량 배수 (따로 %.2f, 묶음 %.2f, 한 묶음 %.2f) | 꺼내는 양 배수 (묶음 %.3f, 한 묶음 %.3f) | 층마다 묶음 수 %s' % (
                  n, row['cost']['each'], row['cost']['group'], row['cost']['union'], row['compute']['each'],
                  row['compute']['group'], row['compute']['union'], row['reads']['group'], row['reads']['union'], row['groups']),
              flush=True)
    os.makedirs(os.path.join(HERE3, 'runs'), exist_ok=True)
    jdump(dict(c=C_R, model=MODEL, table=TABLE, rows=res), os.path.join(HERE3, 'runs', 'batch_a.json'))


if __name__ == '__main__':
    main()
