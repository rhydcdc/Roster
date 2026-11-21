"""BATCHING - S: 자율 탐색. 연산량과 읽는 양을 함께 줄이는 묶는 법 (학습 없음).
공용 틀
  계획 = 층마다 직사각형 (요청 번호들, 뉴런 번호들)의 목록. 직사각형 하나는 그 뉴런의 가중치를 한 번 꺼내 그 요청들에 쓴다.
  연산량 = Σ c16(요청 수) x c16(뉴런 수),  읽는 양 = Σ 뉴런 수.
  확인: 요청이 쓰는 (요청, 뉴런)은 모두 정확히 한 직사각형에 든다 (모든 계획에서 센다). 나머지 칸은 0을 곱한다.
  같은 대기열: 씨앗 0, QN의 순서대로 뽑는다 (batch_n, t, h, h2와 같다).
S1 칸 나무: b_sim16의 16행 칸을 이웃끼리 둘씩 합친 나무. 뉴런마다 '연산량 + lam x 읽는 양'이 가장 작은 마디 모음을 DP로 고른다.
S2 무리 안 뉴런 블록: 겹침 순서로 G개씩 무리, 무리마다 뉴런 쪽 블록(연산량 최소), 블록마다 그 뉴런을 쓰는 행만.
S3 층층이 벗기기, S4 블록 합치기(S2 + 합치기), S5 = S3 + 합치기.
python batch_s.py s1 | s2
"""
import json
import os
import sys
import time

import numpy as np

HERE_S = os.path.dirname(os.path.abspath(__file__))
RUNS_S = os.path.join(HERE_S, 'runs')
KEYS = os.path.join(RUNS_S, 'keys.npz')
QN = {100: 20, 256: 10, 1000: 5}
NN = 4096


def c16(x):
    return -(-int(x) // 16) * 16


def load_keys():
    """시험 사진마다 층마다 간 중심점 keys (4, 10000), 중심점마다 켜는 뉴런 BM (4, 256, 4096), 겹침 순서 pos (4, 256)."""
    if not os.path.exists(KEYS):
        sys.path.insert(0, HERE_S)
        from batch_t import load_data, load_net, MODEL, TABLE, DEV, keys_all, N_BLOCK, build_tree, leaves_list
        import torch
        D = load_data()
        model = load_net(MODEL)
        t = torch.load(TABLE, map_location=DEV)
        C, NR, KJ = t['C'], t['NR'], t['KJ']
        keys = np.stack([k.cpu().numpy() for k in keys_all(model, D['xte'], C, NR, KJ)])
        BM = np.stack([(NR[l] < KJ[l][:, None]).cpu().numpy() for l in range(N_BLOCK)])
        pos = np.zeros((N_BLOCK, BM.shape[1]), np.int64)
        for l in range(N_BLOCK):
            o = leaves_list(build_tree(BM[l]))
            pos[l, o] = np.arange(BM.shape[1])
        np.savez_compressed(KEYS, keys=keys, BM=BM, pos=pos)
    z = np.load(KEYS)
    return z['keys'], z['BM'], z['pos']


def queues(ntest):
    rng = np.random.default_rng(0)
    for N, T in QN.items():
        for t in range(T):
            yield N, t, rng.choice(ntest, N, replace=False)


def sim_tiles(kq, pos_l):
    """b_sim16의 칸: 요청을 간 중심점의 겹침 순서로 늘어놓고 16행씩 자른다. 반환: 요청 번호 배열들."""
    idx = np.argsort(pos_l[kq], kind='stable')
    return [idx[s:s + 16] for s in range(0, len(idx), 16)]


def evaluate(plan, Mq, check=True):
    """plan: [(rows, cols)], Mq: (N, 4096) 요청마다 쓰는 뉴런. 반환: 연산량, 읽는 양, 진단."""
    comp = reads = dense = visits = nrect = 0
    cnt = np.zeros(Mq.shape, np.int16) if check else None
    for rows, cols in plan:
        if len(rows) == 0 or len(cols) == 0:
            continue
        comp += c16(len(rows)) * c16(len(cols))
        reads += len(cols)
        dense += len(rows) * len(cols)
        visits += len(rows)
        nrect += 1
        if check:
            cnt[np.ix_(rows, cols)] += 1
    ones = int(Mq.sum())
    ok = bool((cnt[Mq] == 1).all()) if check else None
    return dict(compute=comp, reads=reads, rects=nrect, ones=ones, dense=dense, visits=visits / Mq.shape[0], ok=ok)


def bases(kq, BM_l):
    """완전 베이스와 줄인 베이스 (들어온 순서대로 16개씩)."""
    fc = fr = sc = sr = 0
    for s in range(0, len(kq), 16):
        ch = kq[s:s + 16]
        u = int(BM_l[np.unique(ch)].any(0).sum())
        fc += c16(len(ch)) * NN
        fr += NN
        sc += c16(len(ch)) * c16(u)
        sr += u
    return dict(full=(fc, fr), seq=(sc, sr))


def plan_bsim16(tiles, Mq):
    return [(t, np.nonzero(Mq[t].any(0))[0]) for t in tiles]


def tile_tree(tiles, Mq):
    """칸을 이웃끼리 둘씩 합친 나무. 마디 = dict(rows, use, ch). 반환: 마디 목록 (자식이 부모보다 앞), 뿌리."""
    level = [dict(rows=t, use=Mq[t].any(0), ch=None) for t in tiles]
    nodes = list(level)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                a, b = level[i], level[i + 1]
                nd = dict(rows=np.concatenate([a['rows'], b['rows']]), use=a['use'] | b['use'], ch=(a, b))
                nodes.append(nd)
                nxt.append(nd)
            else:
                nxt.append(level[i])
        level = nxt
    return nodes, level[0]


def plan_tree(nodes, root, lam):
    """뉴런마다 '연산량 + lam x 읽는 양'이 가장 작은 마디 모음 (같으면 위 마디 = 적게 읽기)."""
    for nd in nodes:
        take = np.where(nd['use'], c16(len(nd['rows'])) + lam, 0.0)
        if nd['ch'] is None:
            nd['f'], nd['take'] = take, nd['use'].copy()
        else:
            split = nd['ch'][0]['f'] + nd['ch'][1]['f']
            tk = nd['use'] & (take <= split + 1e-9)
            nd['f'], nd['take'] = np.where(tk, take, split), tk
    plan, stack = [], [(root, np.zeros(NN, bool))]
    while stack:
        nd, cov = stack.pop()
        a = nd['take'] & ~cov
        if a.any():
            plan.append((nd['rows'], np.nonzero(a)[0]))
        if nd['ch'] is not None:
            for ch in nd['ch']:
                stack.append((ch, cov | nd['take']))
    return plan


def run_exp(tag, names, make, pos_file=None):
    """make(ctx) -> {이름: (계획, 고르는 데 쓴 초)}. ctx: kq, Mq, BM_l, pos_l, tiles, order(겹침 순서의 요청 번호)."""
    keys, BM, pos = load_keys()
    if pos_file:
        pos = np.load(os.path.join(RUNS_S, pos_file))
    ref_n = json.load(open(os.path.join(RUNS_S, 'batch_n.json')))
    out, acc, allok = {}, {}, True
    T0 = time.perf_counter()
    for N, t, order in queues(keys.shape[1]):
        if t == 0:
            acc = {k: dict(compute=0, reads=0, rects=0, ones=0, dense=0, visits=0.0, ms=0.0) for k in names + ['full', 'seq']}
        for l in range(keys.shape[0]):
            kq = keys[l][order]
            Mq = BM[l][kq]
            b = bases(kq, BM[l])
            for k in ('full', 'seq'):
                acc[k]['compute'] += b[k][0]
                acc[k]['reads'] += b[k][1]
            sord = np.argsort(pos[l][kq], kind='stable')
            ctx = dict(kq=kq, Mq=Mq, BM_l=BM[l], pos_l=pos[l], order=sord, tiles=[sord[s:s + 16] for s in range(0, N, 16)])
            for k, (p, sec) in make(ctx).items():
                e = evaluate(p, Mq, check=True)
                allok &= e['ok']
                acc[k]['ms'] += 1000 * sec
                for f in ('compute', 'reads', 'rects', 'ones', 'dense', 'visits'):
                    acc[k][f] += e[f]
        print('[%s] 대기열 %d개 %d/%d 끝, %.0f s' % (tag, N, t + 1, QN[N], time.perf_counter() - T0), flush=True)
        if t == QN[N] - 1:
            T, L = QN[N], keys.shape[0]
            F, S = acc['full'], acc['seq']
            res = {}
            for k, v in acc.items():
                r = dict(compute=v['compute'] / T, reads=v['reads'] / T,
                         pct_full=(100 * v['compute'] / F['compute'], 100 * v['reads'] / F['reads']),
                         pct_seq=(100 * v['compute'] / S['compute'], 100 * v['reads'] / S['reads']))
                if k not in ('full', 'seq'):
                    r.update(rects_per_layer=v['rects'] / T / L, zero_frac=1 - v['ones'] / max(v['dense'], 1),
                             pad_frac=1 - v['dense'] / max(v['compute'], 1), visits_per_req_layer=v['visits'] / T / L,
                             ms_per_layer=v['ms'] / T / L, ideal=v['ones'] / T)
                res[k] = r
            nf = ref_n[str(N)]['res']['neuron_all']
            Fa, Sa = F['compute'] / T, S['compute'] / T
            Fr, Sr = F['reads'] / T, S['reads'] / T
            res['neuron_block(ref)'] = dict(compute=nf['compute'], reads=nf['reads'],
                                            pct_full=(100 * nf['compute'] / Fa, 100 * nf['reads'] / Fr),
                                            pct_seq=(100 * nf['compute'] / Sa, 100 * nf['reads'] / Sr))
            out[N] = res
            ideal = res[names[0]]['ideal']
            print('\n=== %s 대기열 %d개 (%d번, 층 4개 합) | 덮기 정확히 한 번 %s | 바닥값: 연산량 %.0f (완전 베이스의 %.1f%%) ===' % (
                tag, N, T, 'OK' if allok else '어긋남!', ideal, 100 * ideal / Fa), flush=True)
            print('%-18s %10s %9s | %-15s | %-15s | %6s %6s %6s %6s %7s' % ('방법', '연산량', '읽는 양', '완전 베이스의 %', '줄인 베이스의 %',
                                                                         '직사각형', '0곱', '빈칸', '방문', 'ms/층'))
            for k in names + ['neuron_block(ref)']:
                r = res[k]
                print('%-18s %10.0f %9.0f | %6.1f%% / %5.1f%% | %6.1f%% / %5.1f%% | %6.1f %5.1f%% %5.1f%% %6.1f %7.2f' % (
                    k, r['compute'], r['reads'], r['pct_full'][0], r['pct_full'][1], r['pct_seq'][0], r['pct_seq'][1],
                    r.get('rects_per_layer', float('nan')), 100 * r.get('zero_frac', float('nan')),
                    100 * r.get('pad_frac', float('nan')), r.get('visits_per_req_layer', float('nan')),
                    r.get('ms_per_layer', float('nan'))), flush=True)
    with open(os.path.join(RUNS_S, 'batch_%s.json' % tag), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def timed(fn, *a):
    t0 = time.perf_counter()
    p = fn(*a)
    return p, time.perf_counter() - t0


LAMS1 = [0.0, 16.0, 64.0, 256.0, 1024.0, 1e12]


def run_s1():
    def make(ctx):
        out = {'b_sim16': timed(plan_bsim16, ctx['tiles'], ctx['Mq'])}
        (nodes, root), s0 = timed(tile_tree, ctx['tiles'], ctx['Mq'])
        for lam in LAMS1:
            p, s = timed(plan_tree, nodes, root, lam)
            out['tree_lam%g' % lam] = (p, s + s0)
        return out
    run_exp('s1', ['b_sim16'] + ['tree_lam%g' % l for l in LAMS1], make)


def c16v(x):
    return np.ceil(x / 16.0) * 16.0


def merge_blocks(km, nc, n):
    """S4: 같은 무리 안 블록들(쓰는 중심점 bool (B, P), 뉴런 수 (B,))을, 합쳐서 연산량이 줄면 합친다 (가장 많이 주는 쌍부터).
    블록끼리 뉴런이 겹치지 않으므로 읽는 양은 그대로다. 반환: 합친 블록마다 원래 블록 번호들."""
    km, nc = km.copy(), nc.astype(np.float64).copy()
    alive, groups = np.ones(len(nc), bool), [[i] for i in range(len(nc))]
    while alive.sum() > 1:
        idx = np.nonzero(alive)[0]
        M = km[idx].astype(np.float64)
        r = M @ n
        ru = r[:, None] + r[None, :] - (M * n) @ M.T
        cs = c16v(r) * c16v(nc[idx])
        gain = cs[:, None] + cs[None, :] - c16v(ru) * c16v(nc[idx][:, None] + nc[idx][None, :])
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


def plan_groups_nb(order, kq, BM_l, G, merge=False):
    """S2: 겹침 순서로 G개씩 무리, 무리마다 뉴런 쪽 블록 (batch_n의 나무 자르기, 연산량 최소). 블록마다 쓰는 행만.
    merge=True (S4): 자른 뒤 블록 합치기."""
    from batch_n import neuron_tree, cut, leaves
    plan = []
    for s in range(0, len(order), G):
        rows = order[s:s + G]
        P, inv_r, cnt = np.unique(kq[rows], return_inverse=True, return_counts=True)
        n = cnt.astype(np.float64)
        masks, size, r, Z, inv, cols = neuron_tree(BM_l[P], n)
        _, nb, nodes = cut(masks, size, r, Z, n, None)
        B0 = len(size)
        blk = []
        for v in nodes:
            lv = leaves(Z, B0, v) if Z is not None else [0]
            blk.append((masks[lv].any(0), cols[np.isin(inv, lv)]))
        if merge and len(blk) > 1:
            km = np.stack([b[0] for b in blk])
            gs = merge_blocks(km, np.array([len(b[1]) for b in blk]), n)
            blk = [(km[g].any(0), np.concatenate([blk[i][1] for i in g])) for g in gs]
        for kmask, cc in blk:
            plan.append((rows[kmask[inv_r]], cc))
    return plan


GS4 = [32, 64, 128, 256, 512, 'N']


def run_s4():
    sys.path.insert(0, HERE_S)

    def make(ctx):
        out = {}
        for G in GS4:
            g = len(ctx['order']) if G == 'N' else G
            out['grp_nb_merge_G%s' % G] = timed(plan_groups_nb, ctx['order'], ctx['kq'], ctx['BM_l'], g, True)
        return out
    run_exp('s4', ['grp_nb_merge_G%s' % G for G in GS4], make)


GS2 = [16, 32, 64, 128, 256, 512, 'N']


def run_s2():
    sys.path.insert(0, HERE_S)

    def make(ctx):
        out = {}
        for G in GS2:
            g = len(ctx['order']) if G == 'N' else G
            out['grp_nb_G%s' % G] = timed(plan_groups_nb, ctx['order'], ctx['kq'], ctx['BM_l'], g)
        return out
    run_exp('s2', ['grp_nb_G%s' % G for G in GS2], make)


def plan_peel(tiles, Mq, kq, lam, merge=False):
    """S3 층층이 벗기기: 칸 나무를 위에서부터. 마디마다 남은 (요청, 뉴런)으로 뉴런 쪽 블록을 만들고(연산량 최소 자르기),
    블록의 '연산량 + lam x 읽는 양'이 칸에서 계산할 때(뉴런마다 그 뉴런을 쓰는 칸 수 x (16 + lam))보다 작거나 같으면 받는다.
    받지 않은 것은 자식 마디로. 맨 아래 칸은 남은 것 전부를 직사각형 하나로 (b_sim16과 같음).
    merge=True (S5): 마디마다 받은 블록들을 S4처럼 합친다 (덮는 칸과 읽는 양은 그대로)."""
    from batch_n import neuron_tree, cut, leaves
    nodes, root = tile_tree(tiles, Mq)
    R = Mq.copy()
    plan, pos_t = [], 0
    for nd in nodes:                                                   # 마디마다 맡은 칸의 범위 [a, b)
        if nd['ch'] is None:
            nd['span'] = (pos_t, pos_t + 1)
            pos_t += 1
        else:
            nd['span'] = (nd['ch'][0]['span'][0], nd['ch'][1]['span'][1])
    stack = [root]
    while stack:
        v = stack.pop()
        rows = v['rows']
        if v['ch'] is None:
            cols = np.nonzero(R[rows].any(0))[0]
            if len(cols):
                plan.append((rows, cols))
                R[np.ix_(rows, cols)] = False
            continue
        P, first, inv_r, cnt = np.unique(kq[rows], return_index=True, return_inverse=True, return_counts=True)
        U = R[rows[first]]                                             # 마디 안에서 같은 중심점의 요청은 남은 것이 같다
        if U.any():
            a, b = v['span']
            Tn = np.stack([R[tiles[i]].any(0) for i in range(a, b)]).sum(0)
            n = cnt.astype(np.float64)
            masks, size, r, Z, inv, cols = neuron_tree(U, n)
            _, nb, bnodes = cut(masks, size, r, Z, n, None)
            B0 = len(size)
            acc = []
            for bn in bnodes:
                lv = leaves(Z, B0, bn) if Z is not None else [0]
                cc = cols[np.isin(inv, lv)]
                km = masks[lv].any(0)
                rr = rows[km[inv_r]]
                here = c16(len(rr)) * c16(len(cc)) + lam * len(cc)
                low = (16 + lam) * float(Tn[cc].sum())
                if here <= low:
                    acc.append((km, cc))
                    R[np.ix_(rr, cc)] = False
            if merge and len(acc) > 1:
                kms = np.stack([x[0] for x in acc])
                gs = merge_blocks(kms, np.array([len(x[1]) for x in acc]), n)
                acc = [(kms[g].any(0), np.concatenate([acc[i][1] for i in g])) for g in gs]
            for km, cc in acc:
                plan.append((rows[km[inv_r]], cc))
        stack.extend(v['ch'])
    return plan


LAMS3 = [0.0, 16.0, 64.0, 256.0, 1e12]


def run_s3():
    sys.path.insert(0, HERE_S)

    def make(ctx):
        return {'peel_lam%g' % lam: timed(plan_peel, ctx['tiles'], ctx['Mq'], ctx['kq'], lam) for lam in LAMS3}
    run_exp('s3', ['peel_lam%g' % lam for lam in LAMS3], make)


def run_s6():
    sys.path.insert(0, HERE_S)

    def make(ctx):
        out = {}
        for G in GS4:
            g = len(ctx['order']) if G == 'N' else G
            out['olo_merge_G%s' % G] = timed(plan_groups_nb, ctx['order'], ctx['kq'], ctx['BM_l'], g, True)
        return out
    run_exp('s6', ['olo_merge_G%s' % G for G in GS4], make, pos_file='pos_olo.npy')


LAMS5 = [0.0, 2.0, 4.0, 8.0, 16.0, 64.0]


def run_s5():
    sys.path.insert(0, HERE_S)

    def make(ctx):
        return {'peel_merge_lam%g' % lam: timed(plan_peel, ctx['tiles'], ctx['Mq'], ctx['kq'], lam, True) for lam in LAMS5}
    run_exp('s5', ['peel_merge_lam%g' % lam for lam in LAMS5], make)


if __name__ == '__main__':
    {'s1': run_s1, 's2': run_s2, 's3': run_s3, 's4': run_s4, 's5': run_s5, 's6': run_s6}[sys.argv[1]]()
