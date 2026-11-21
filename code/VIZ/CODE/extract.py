"""레시피 시각화(VIZ) 페이지에 넣을 데이터를 실제 모델과 표에서 뽑는다. 지어낸 숫자는 없다.
python extract.py   -> ../data/viz.json, ../data/*.png

뽑는 것
  pretrain : 베이스 모델의 사전 학습 곡선 (검증 정확도)
  layers[l]: 중심점 만들기(t-SNE 점구름, 중심점 위치, 중심점마다 대표 사진), 고른 중심점 4개(목록, 더하는 크기, 사진 8장)
  rounds   : 바퀴 0 ~ 6. 바퀴마다 중심점마다 K(층마다 256개), 고른 중심점의 사후 학습 전후 '목록 순위별 누적 몫' 곡선,
             출력 맞추기 오차, K 줄이기의 오차 곡선 e(k)와 eps, 새 K, 켜진 뉴런 격자
  examples : 시험 사진 12장. 바퀴마다 층마다 간 중심점과 켜진 뉴런, 최종 모델의 추론 과정 전부(층 입력, 정규화, 거리, 고른 뉴런,
             돋보기 뉴런 하나의 곱, 출력), 최종 확률과 베이스 모델의 확률
"""
import base64
import io
import json
import math
import os
import sys

_VIZ_CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_VIZ_CODE)), 'AUTO', 'EXP2', 'CODE'))
from lchain import *                                   # noqa: E402  (ex2lib, auto, common을 함께 불러온다)
PROJ = os.path.dirname(os.path.dirname(_VIZ_CODE))     # 가져온 뒤에 정한다 (common의 ROOT, auto의 HERE가 덮어쓴다: HANDOFF 함정 5)
import numpy as np                                     # noqa: E402
from PIL import Image                                  # noqa: E402

OUTD = os.path.join(PROJ, 'VIZ', 'data')
os.makedirs(OUTD, exist_ok=True)
CLASSES = ['티셔츠', '바지', '풀오버', '원피스', '코트', '샌들', '셔츠', '운동화', '가방', '앵클부츠']
FEAT_CLASSES = [7, 1, 8, 0]                            # 고른 중심점: 그 부류의 사진을 가장 많이 받는 중심점 (운동화, 바지, 가방, 티셔츠)
M = 256
EXP2 = os.path.join(PROJ, 'AUTO', 'EXP2', 'CODE', 'runs')
MODELS = [F_PATH] + [os.path.join(EXP2, n, 'best.pt') for n in
                     ['Lb1_lr3e-05', 'Lb2_lr3e-05', 'Lb3_lr3e-05', 'Lb4_lr3e-05', 'L5s_lr9.49367e-07', 'Lc6_lr9.49367e-07']]
TABLES = ['Lb0', 'Lb1', 'Lb2', 'Lb3', 'Lb4', 'Lc5', 'Lc6']
LRS = [None, 3e-5, 3e-5, 3e-5, 3e-5, 9.49367e-07, 9.49367e-07]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def q8(v):
    """부호 있는 값을 int8로 (배율과 함께)."""
    v = np.asarray(v, dtype=np.float32)
    s = float(np.abs(v).max()) or 1.0
    return dict(s=round(s, 5), q=b64(np.clip(np.round(v / s * 127), -127, 127).astype(np.int8)))


def u8(v):
    """0 이상 값을 uint8로 (제곱근으로 펴서, 최댓값 255)."""
    v = np.sqrt(np.clip(np.asarray(v, dtype=np.float32), 0, None))
    m = float(v.max()) or 1.0
    return b64(np.round(v / m * 255).astype(np.uint8))


def bits(on):
    return b64(np.packbits(np.asarray(on, dtype=np.uint8)))


def r(x, n=4):
    return round(float(x), n)


def sprite(imgs, cols, path):
    """28x28 사진들을 한 장의 PNG로 (흰 물건이 검은 바탕에)."""
    n = len(imgs)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * 28, cols * 28), dtype=np.uint8)
    for i, im in enumerate(imgs):
        rr, cc = divmod(i, cols)
        canvas[rr * 28:(rr + 1) * 28, cc * 28:(cc + 1) * 28] = im
    Image.fromarray(canvas, 'L').save(path, optimize=True)
    return dict(file=os.path.basename(path), cols=cols, n=n)


def pix(x):
    """정규화된 사진 -> 0..255."""
    return np.clip(np.round((x.cpu().numpy() * 0.3530 + 0.2860) * 255), 0, 255).astype(np.uint8)


@torch.no_grad()
def walk(model, X, C, NR, KJ, keep=None, chunk=2500):
    """고른 뉴런만 계산하며 앞으로 간다. 층마다 열쇠(가장 가까운 중심점)와 거리를 돌려준다.
    keep: {(l, j): True} 이면 그 중심점으로 간 입력의 a, on, h, full, 그리고 더하는 크기의 합 S를 모은다."""
    model.eval()
    N = X.shape[0]
    keys = [torch.empty(N, dtype=torch.long, device=DEV) for _ in range(N_BLOCK)]
    dist = [torch.empty(N, device=DEV) for _ in range(N_BLOCK)]
    got = {k: dict(S=torch.zeros(N_NEURON, device=DEV), A=[], ON=[], H=[], FULL=[]) for k in (keep or {})}
    for s in range(0, N, chunk):
        xb = X[s:s + chunk]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                kf = lkey(h)
                d = (C[l] * C[l]).sum(1)[None, :] - 2 * kf @ C[l].T + (kf * kf).sum(1, keepdim=True)
                dm, j = d.min(1)
            keys[l][s:s + chunk] = j
            dist[l][s:s + chunk] = dm
            on = NR[l][j] < KJ[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                full = blk.down(a * on.to(a.dtype)).float()
            if keep:
                c = a.float().abs() * blk.colnorm()
                for (ll, jj), g in got.items():
                    if ll != l:
                        continue
                    sel = j == jj
                    if sel.any():
                        g['S'] += (c[sel] * on[sel]).sum(0)
                        g['A'].append(a[sel].float())
                        g['ON'].append(on[sel])
                        g['H'].append(h[sel].float())
                        g['FULL'].append(full[sel])
            h = h + full
    for g in got.values():
        for k in ('A', 'ON', 'H', 'FULL'):
            g[k] = torch.cat(g[k]) if g[k] else None
    return keys, dist, got


@torch.no_grad()
def teacher_err(base, l, H, FULL):
    tb = base.blocks[l]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        t = tb.down(tb.neurons(H)).float()
    return float((((FULL - t) ** 2).sum(1) / ((t ** 2).sum(1) + 1e-6)).mean())


def cum_curve(S, NR_row, Kold, npts=96):
    """목록 순서(지금 켜진 뉴런만)로 놓은 '앞 x개가 일의 몇 %를 하나' 곡선."""
    order = torch.argsort(NR_row)[:Kold]
    v = S[order].clamp_min(0)
    cs = (v.cumsum(0) / (v.sum() + 1e-12)).cpu().numpy()
    xs = np.unique(np.linspace(0, Kold - 1, npts).round().astype(int))
    return dict(x=[int(i + 1) for i in xs], y=[r(cs[i], 4) for i in xs])


@torch.no_grad()
def err_curve(blk, g, NR_row, Kold):
    """K 줄이기: 지금 모델이 더하는 크기로 다시 줄 세우고, 앞 k개만 켰을 때의 출력 오차 e(k) (그 중심점의 학습 사진 평균)."""
    Sl = g['S'].clone()
    Sl[NR_row >= Kold] = -1
    order = Sl.argsort(descending=True)
    nr = torch.empty_like(order)
    nr[order] = torch.arange(N_NEURON, device=DEV)
    A, FULL = g['A'], g['FULL']
    ks, es = [], []
    for k in KGRID:
        kk = min(k, Kold)
        on = (nr < kk)[None, :]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = blk.down(A.to(torch.bfloat16) * on.to(torch.bfloat16)).float()
        e = (((o - FULL) ** 2).sum(1) / ((FULL ** 2).sum(1) + 1e-6)).mean()
        ks.append(int(kk))
        es.append(float(e))
        if k >= Kold:
            break
    return nr, ks, es


def main():
    D = load_data()
    base = load_net(PICK['path'])
    C, RK = get_rule(D, dict(op='ltable', M=M, K=K0))
    KJ_init = [torch.full((M,), K0, device=DEV, dtype=torch.long) for _ in range(N_BLOCK)]
    KJ_all = [torch.full((M,), N_NEURON, device=DEV, dtype=torch.long) for _ in range(N_BLOCK)]
    tabs = []
    for t in TABLES:
        tt = torch.load(os.path.join(RULES, 'lv_%s.pt' % t), map_location=DEV)
        tabs.append((tt['NR'], tt['KJ']))
    Xtr, Ytr = D['xtr'], D['ytr'].cpu().numpy()
    out = dict(classes=CLASSES)

    # ---- 사전 학습 곡선 ----
    rows = [json.loads(x) for x in open(os.path.join(os.path.dirname(PICK['path']), 'log.jsonl'), encoding='utf-8') if '"val_acc"' in x]
    out['pretrain'] = dict(epoch=[x['epoch'] for x in rows], val=[r(x['val_acc'], 2) for x in rows], lr=[x['lr'] for x in rows],
                           test=r(PICK['test_acc'], 2), val_best=r(max(x['val_acc'] for x in rows), 2))

    # ---- 베이스 모델로: 중심점마다 모인 사진, 고른 중심점, 첫 목록의 더하는 크기 ----
    print('베이스 모델 한 바퀴', flush=True)
    kb, db, _ = walk(base, Xtr, C, RK, KJ_all)
    feats = []
    for l in range(N_BLOCK):
        k = kb[l].cpu().numpy()
        fl = []
        for cls in FEAT_CLASSES:
            cnt = np.bincount(k[Ytr == cls], minlength=M)
            for j in np.argsort(-cnt):
                if int(j) not in fl:
                    fl.append(int(j))
                    break
        feats.append(fl)
    keep = {(l, j): True for l in range(N_BLOCK) for j in feats[l]}
    _, _, gb = walk(base, Xtr, C, RK, KJ_all, keep=keep)

    # 대표 사진: 중심점마다 가장 가까운 학습 사진 1장, 고른 중심점은 8장
    rep_imgs, feat_imgs = [], []
    layers = []
    sub = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0))[:20000]
    from sklearn.manifold import TSNE
    for l in range(N_BLOCK):
        k = kb[l].cpu().numpy()
        d = db[l].cpu().numpy()
        cnt = np.bincount(k, minlength=M)
        rep = []
        for j in range(M):
            idx = np.where(k == j)[0]
            rep.append(int(idx[np.argmin(d[idx])]) if len(idx) else -1)
        rep_imgs.append([pix(Xtr[i]) if i >= 0 else np.zeros((28, 28), np.uint8) for i in rep])
        # t-SNE: 중심점을 만든 사진 2만 장 가운데 3,000장의 층 입력(정규화)
        pts = sub[:3000].to(DEV)
        with torch.no_grad():
            h = base.inp(Xtr[pts].reshape(len(pts), -1))
            for ll in range(l):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = h + base.blocks[ll].down(base.blocks[ll].neurons(h)).float()
            f = lkey(h).cpu().numpy()
        print('t-SNE 층 %d' % (l + 1), flush=True)
        Y2 = TSNE(n_components=2, init='pca', perplexity=40, random_state=0).fit_transform(f)
        Y2 = (Y2 - Y2.min(0)) / (Y2.max(0) - Y2.min(0))
        pk = k[pts.cpu().numpy()]
        cx = np.full((M, 2), np.nan)
        for j in range(M):
            m_ = pk == j
            if m_.any():
                cx[j] = Y2[m_].mean(0)
        fl = []
        for j in feats[l]:
            g = gb[(l, j)]
            idx = np.where(k == j)[0]
            near = idx[np.argsort(d[idx])[:8]]
            feat_imgs.extend([pix(Xtr[i]) for i in near] + [np.zeros((28, 28), np.uint8)] * (8 - len(near)))
            S = g['S'] / max(1, len(idx))
            ordS = torch.sort(S, descending=True).values.cpu().numpy()
            xs = np.unique(np.geomspace(1, N_NEURON, 120).round().astype(int)) - 1
            fl.append(dict(j=j, n=int(cnt[j]), cls=[int(x) for x in np.bincount(Ytr[idx], minlength=10)],
                           grid=u8(S.cpu().numpy()), on=bits((RK[l][j] < K0).cpu().numpy()),
                           sorted=dict(x=[int(i + 1) for i in xs], y=[r(ordS[i] / ordS[0], 4) for i in xs])))
        layers.append(dict(count=[int(x) for x in cnt],
                           pts=dict(x=[r(v, 3) for v in Y2[:, 0]], y=[r(v, 3) for v in Y2[:, 1]], key=[int(v) for v in pk],
                                    cls=[int(v) for v in Ytr[pts.cpu().numpy()]]),
                           cen=dict(x=[None if np.isnan(v) else r(v, 3) for v in cx[:, 0]],
                                    y=[None if np.isnan(v) else r(v, 3) for v in cx[:, 1]]),
                           feat=fl))
    out['sprites'] = dict(rep=[sprite(rep_imgs[l], 16, os.path.join(OUTD, 'rep_l%d.png' % (l + 1))) for l in range(N_BLOCK)],
                          feat=sprite(feat_imgs, 8, os.path.join(OUTD, 'feat.png')))
    out['layers'] = layers

    # ---- 바퀴 0 ~ 6 ----
    lb = jload(os.path.join(OUT, 'Lb.json'))['rounds']
    lc = jload(os.path.join(OUT, 'Lc.json'))['rounds']
    info = {x['round']: x for x in lb if 'test' in x}
    info.update({x['round']: x for x in lc if 'test' in x})
    rounds = []
    prevT = (RK, KJ_init)
    prevM = None
    mism = 0
    for rr in range(7):
        print('바퀴 %d' % rr, flush=True)
        Mr = load_net(MODELS[rr])
        NRp, KJp = prevT
        NRn, KJn = tabs[rr]
        _, _, ga = walk(Mr, Xtr, C, NRp, KJp, keep=keep)                      # 학습 뒤 모델, 지난 표
        gbf = walk(load_net(MODELS[rr - 1]), Xtr, C, NRp, KJp, keep=keep)[2] if rr >= 1 else ga   # 학습 전 모델, 지난 표
        eps = info[rr]['eps']
        fr = []
        for l in range(N_BLOCK):
            fl = []
            for j in feats[l]:
                g, g0 = ga[(l, j)], gbf[(l, j)]
                Kold = int(KJp[l][j])
                if g['A'] is None:
                    fl.append(None)
                    continue
                nr, ks, es = err_curve(Mr.blocks[l], g, NRp[l][j], Kold)
                knew = next((k for k, e in zip(ks, es) if e <= eps), Kold)
                knew = min(knew, Kold)
                if knew != int(KJn[l][j]):
                    mism += 1
                fl.append(dict(j=j, Kold=Kold, Knew=int(KJn[l][j]), Kcalc=int(knew), n=int(g['A'].shape[0]),
                               before=cum_curve(g0['S'], NRp[l][j], Kold), after=cum_curve(g['S'], NRp[l][j], Kold),
                               match_before=r(teacher_err(base, l, g0['H'], g0['FULL']), 5),
                               match_after=r(teacher_err(base, l, g['H'], g['FULL']), 5),
                               ek=dict(k=ks, e=[r(e, 5) for e in es]),
                               grid=u8(g['S'].cpu().numpy()), on_old=bits((NRp[l][j] < Kold).cpu().numpy()),
                               on_new=bits((NRn[l][j] < KJn[l][j]).cpu().numpy())))
            fr.append(fl)
        x = info[rr]
        rounds.append(dict(round=rr, lr=LRS[rr], eps=eps, val=r(x['val'], 2), avg=r(x['test']['avg'], 2),
                           layers=[r(v, 2) for v in x['test']['layers']], acc=r(x['test']['acc'], 2),
                           lo=r(91.02 - x['test']['hi'], 2), hi=r(91.02 - x['test']['lo'], 2), verdict=x['test']['verdict'],
                           K=[[int(v) for v in KJn[l].cpu().numpy()] for l in range(N_BLOCK)], feat=fr))
        prevT = (NRn, KJn)
        del Mr
        torch.cuda.empty_cache()
    out['rounds'] = rounds
    out['K_init'] = K0
    print('K 다시 셈과 표의 K가 다른 고른 중심점 수: %d / %d' % (mism, 7 * 16), flush=True)

    # ---- 예시 시험 사진 12장 ----
    Xte, Yte = D['xte'], D['yte'].cpu().numpy()
    fin = load_net(MODELS[6])
    NRf, KJf = tabs[6]
    pf, _, _ = run_masked(fin, Xte, *var_mask(*keys_layer(C), NRf, KJf))
    pb = evaluate(base, Xte, D['yte'])['probs'].argmax(1)
    pf, pb = pf.cpu().numpy(), pb.cpu().numpy()
    ex = []
    for cls in list(range(10)) + [7, 8]:
        for i in np.where((Yte == cls) & (pf == Yte) & (pb == Yte))[0]:
            if int(i) not in ex:
                ex.append(int(i))
                break
    exs = []
    inf_imgs = []
    with torch.no_grad():
        xe = Xte[ex]
        # 바퀴마다 층마다 간 중심점과 켜진 뉴런
        per_round = []
        for rr in range(7):
            Mr = load_net(MODELS[rr])
            NRr, KJr = tabs[rr]
            kr, _, _ = walk(Mr, xe, C, NRr, KJr)
            per_round.append([[dict(j=int(kr[l][e]), K=int(KJr[l][kr[l][e]]),
                                    on=bits((NRr[l][kr[l][e]] < KJr[l][kr[l][e]]).cpu().numpy())) for l in range(N_BLOCK)]
                              for e in range(len(ex))])
        k0, _, _ = walk(load_net(MODELS[0]), xe, C, RK, KJ_init)                # 2-3: F, 처음 표(50%)
        # 최종 모델의 추론 과정
        kf_train, df_train, _ = walk(fin, Xtr, C, NRf, KJf)
        base_probs = torch.softmax(base(xe)[0].float(), -1).cpu().numpy()
        h = fin.inp(xe.reshape(len(ex), -1)).float()
        steps = [[] for _ in ex]
        for l, blk in enumerate(fin.blocks):
            with torch.autocast('cuda', enabled=False):
                kf = lkey(h)
                d = (C[l] * C[l]).sum(1)[None, :] - 2 * kf @ C[l].T + (kf * kf).sum(1, keepdim=True)
            j = d.argmin(1)
            on = NRf[l][j] < KJf[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                y = blk.down(a * on.to(a.dtype)).float()
            c = a.float().abs() * blk.colnorm() * on
            z32 = blk.norm(h.float())
            kt = kf_train[l].cpu().numpy()
            dt = df_train[l].cpu().numpy()
            for e in range(len(ex)):
                je = int(j[e])
                idx = np.where(kt == je)[0]
                near = idx[np.argsort(dt[idx])[:6]]
                base_i = len(inf_imgs)
                inf_imgs.extend([pix(Xtr[i]) for i in near] + [np.zeros((28, 28), np.uint8)] * (6 - len(near)))
                zi = int(c[e].argmax())                                         # 돋보기: 가장 많이 더하는 뉴런
                gw = blk.gate.weight[zi].float()
                uw = blk.up.weight[zi].float()
                gz = (gw * z32[e])
                uz = (uw * z32[e])
                gs, us = float(gz.sum()), float(uz.sum())
                sl = gs / (1 + math.exp(-gs))
                dd = d[e].cpu().numpy()
                top = np.argsort(dd)[:8]
                sel = torch.where(on[e])[0]
                rank = NRf[l][je][sel]
                sel = sel[torch.argsort(rank)]
                steps[e].append(dict(
                    h=q8(h[e].cpu().numpy()), key=q8(kf[e].cpu().numpy()), z=q8(z32[e].detach().cpu().numpy()),
                    dist_top=[dict(j=int(t), d=r(dd[t], 3)) for t in top], dist_all=dict(min=r(dd.min(), 3), max=r(dd.max(), 3),
                                                                                         hist=np.histogram(dd, bins=24)[0].tolist(),
                                                                                         edges=[r(v, 3) for v in np.histogram(dd, bins=24)[1]]),
                    j=je, K=int(KJf[l][je]), n_train=int(len(idx)), imgs=base_i,
                    grid=u8(c[e].float().cpu().numpy()), on=bits(on[e].cpu().numpy()),
                    first=[int(v) for v in sel[:12].cpu().numpy()],
                    zoom=dict(i=zi, rank=int(NRf[l][je][zi]), g=r(gs, 4), u=r(us, 4), silu=r(sl, 4), a=r(sl * us, 4),
                              gz=q8(gz.detach().cpu().numpy()), uz=q8(uz.detach().cpu().numpy()),
                              col=r(float(blk.colnorm()[zi]), 4)),
                    y=q8(y[e].cpu().numpy()), yn=r(float(y[e].norm()), 3), hn=r(float(h[e].norm()), 3)))
            h = h + y
        probs = torch.softmax(fin.head(fin.norm(h)).float(), -1).cpu().numpy()
        for e, i in enumerate(ex):
            exs.append(dict(idx=i, label=int(Yte[i]), img=b64(pix(Xte[i]).reshape(-1)),
                            probs=[r(v, 4) for v in probs[e]], base_probs=[r(v, 4) for v in base_probs[e]],
                            steps=steps[e], init=[dict(j=int(k0[l][e]), on=bits((RK[l][k0[l][e]] < K0).cpu().numpy())) for l in range(N_BLOCK)],
                            rounds=[per_round[rr][e] for rr in range(7)]))
    out['sprites']['inf'] = sprite(inf_imgs, 12, os.path.join(OUTD, 'inf.png'))
    out['examples'] = exs
    out['final'] = dict(test=r((pf == Yte).mean() * 100, 2), base=r((pb == Yte).mean() * 100, 2))
    with open(os.path.join(OUTD, 'viz.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print('viz.json %.1f KB, 최종 시험 %.2f%%, 베이스 %.2f%%' % (os.path.getsize(os.path.join(OUTD, 'viz.json')) / 1024,
                                                           out['final']['test'], out['final']['base']), flush=True)


if __name__ == '__main__':
    main()
