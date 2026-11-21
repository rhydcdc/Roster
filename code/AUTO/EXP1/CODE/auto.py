"""자율 실험: 입력에 정해진 연산만 하면 필요한 뉴런(K)이 나오는 FFN 분류기 (Fashion-MNIST).

목표: 시험 정확도 90.52% 이상(베이스 모델 91.02%)을 지키며 필요한 뉴런(K)을 최대한 낮추고,
      연산이 모델의 필요한 뉴런을 95% 가까이 골라내게 한다.

뉴런 4,096개를 64개씩 무리 64개로 나눈다 (뉴런 i는 무리 i // 64).
  공통 무리 s개: 모든 입력이 늘 켠다.
  연산 무리 64 - s개: 중심점 하나씩을 맡는다. 입력과 가까운 중심점 r개의 무리를 켠다.
필요한 뉴런(K) = (s + r) x 64개.

세 압력 (동시에)
  성능:        연산이 고른 뉴런만 켜고 정답을 맞힌다
  연산에 맞추기: 연산이 고르지 않은 뉴런이 더하는 몫을 줄인다 (세기 lam)
  줄이기:       연산이 고른 무리 가운데 먼 중심점의 무리가 더하는 몫을 줄인다 (세기 lam_s, 무게 = 순위 / r)

python auto.py sweep --set A    학습률 x 세기 훑기
python auto.py queue --set A    정해 둔 실행들을 차례로
"""
import argparse
import math
import os
import sys

def _project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while not (os.path.exists(os.path.join(d, 'GOAL.md')) or (os.path.isdir(os.path.join(d, 'EXP1', 'CODE')) and os.path.isdir(os.path.join(d, 'AUTO')))):   # 프로젝트 맨 위 (공개본: EXP1/CODE와 AUTO가 함께 있는 code/ 폴더)
        d = os.path.dirname(d)
    return d


E1 = os.path.join(_project_root(), 'EXP1', 'CODE')
sys.path.insert(0, E1)
import torch
import common
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS3 = os.path.join(HERE, 'runs')
common.RUNS = RUNS3
PICK = jload(os.path.join(E1, 'runs', 's1_pick.json'))
if not os.path.isabs(PICK['path']):                                     # 공개본: 상대 경로를 EXP1/CODE 기준으로
    PICK['path'] = os.path.join(E1, PICK['path'])
BASE_ACC = PICK['test_acc']
PASS = BASE_ACC - PASS_DROP
NG = 64                                              # 무리 수
G = N_NEURON // NG                                   # 무리 하나의 뉴런 수


# ---- 연산: 가까운 중심점 ----
def nearest(x, C, r):
    X = x.reshape(x.shape[0], -1).float()
    d = (C * C).sum(1)[None, :] - 2 * X @ C.T
    return d.topk(r, dim=1, largest=False).indices


def kmeans(X, M, gen, min_cluster=100):
    n = X.shape[0]
    C = X[torch.randint(0, n, (1,), device=DEV, generator=gen)].clone()
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(M - 1):
        j = torch.multinomial(d2 / d2.sum(), 1, generator=gen)
        C = torch.cat([C, X[j]])
        d2 = torch.minimum(d2, ((X - X[j]) ** 2).sum(1))
    for it in range(40):
        a = nearest(X, C, 1)[:, 0]
        cnt = torch.bincount(a, minlength=M)
        newC = torch.zeros_like(C).index_add_(0, a, X)
        C = torch.where(cnt[:, None] > 0, newC / cnt.clamp_min(1)[:, None].float(), C)
        if it >= 30:
            continue
        for j in (cnt < min_cluster).nonzero().flatten().tolist():
            big = cnt.argmax().item()
            members = (a == big).nonzero().flatten()
            if len(members) < 2:                         # 쪼갤 무리가 없으면 멈춘다
                break
            pick_ = members[torch.randint(0, len(members), (1,), device=DEV, generator=gen)]
            C[j] = X[pick_][0]
            a[pick_] = j                                 # 장부를 실제와 맞춘다
            cnt[big] -= 1
            cnt[j] = 1
    return C


_RULES = {}


def ngs(cfg):
    ng = cfg.get('ng', 64)
    return ng, N_NEURON // ng


def get_table(D, cfg):
    """표 연산: 학습 사진을 k-평균으로 나눈 중심점 M개마다, 베이스 모델에서 그 중심점에 가장 가까운 사진들이
    뉴런마다 더한 것의 합을 구하고, 큰 순서로 뉴런의 순위를 매긴다 (층마다). 입력은 가장 가까운 중심점의 표에서
    순위가 K보다 앞선 뉴런을 켠다. 표는 순위로 정하므로 K를 줄이면 작은 표가 큰 표 안에 들어간다."""
    M = cfg.get('M', 64)
    key = ('table', M)
    if key in _RULES:
        return _RULES[key]
    p = os.path.join(rdir('rules'), f'table_M{M}.pt')
    if not os.path.exists(p):
        pc = os.path.join(rdir('rules'), f'centroid_M{M}.pt')
        if not os.path.exists(pc):
            torch.save(kmeans(D['xtr'].reshape(-1, 784), M, torch.Generator(device=DEV).manual_seed(0)), pc)
        C = torch.load(pc, map_location=DEV)
        X = D['xtr'].reshape(-1, 784)
        a = nearest(X, C, 1)[:, 0]
        net = load_net(PICK['path'])
        S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
        with torch.no_grad():
            for st in range(0, X.shape[0], 2500):
                h = net.inp(X[st:st + 2500])
                for l, blk in enumerate(net.blocks):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        act = blk.neurons(h)
                        h = h + blk.down(act).float()
                    S[l].index_add_(0, a[st:st + 2500], act.float().abs() * blk.colnorm())
        RK = []
        for l in range(N_BLOCK):
            order = S[l].argsort(1, descending=True)
            rk = torch.empty_like(order)
            rk.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(M, N_NEURON))
            RK.append(rk)
        torch.save(dict(C=C, RK=RK), p)
    t = torch.load(p, map_location=DEV)
    _RULES[key] = (2 * t['C'], -(t['C'] * t['C']).sum(1), t['RK'])
    return _RULES[key]


def tfeat(x, feat):
    """표 연산이 거리를 잴 때 쓰는 입력의 꼴. pix: 사진 그대로(784칸), pool2: 2x2 평균으로 줄인 사진(196칸)."""
    if feat == 'pool2':
        return F.avg_pool2d(x.reshape(-1, 1, 28, 28).float(), 2).reshape(x.shape[0], -1)
    return x.reshape(x.shape[0], -1).float()


def get_table2(D, cfg):
    """표 연산의 변형. M: 중심점 수, feat: 거리를 잴 꼴, r: 가까운 중심점 r개의 표를 (평균 크기로) 합쳐 앞의 K개를 켠다.
    중심점마다 베이스 모델에서 근처 사진들이 뉴런마다 더한 것의 평균을 층마다 저장한다."""
    M, feat = cfg.get('M', 64), cfg.get('feat', 'pix')
    key = ('table2', M, feat)
    if key in _RULES:
        return _RULES[key]
    p = os.path.join(rdir('rules'), f'table2_M{M}_{feat}.pt')
    if not os.path.exists(p):
        X = tfeat(D['xtr'], feat)
        C = kmeans(X, M, torch.Generator(device=DEV).manual_seed(0), min_cluster=max(20, 6400 // M))
        a = nearest(X, C, 1)[:, 0]
        cnt = torch.bincount(a, minlength=M).float()
        net = load_net(PICK['path'])
        S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
        with torch.no_grad():
            for st in range(0, X.shape[0], 2500):
                h = net.inp(D['xtr'][st:st + 2500].reshape(-1, 784))
                for l, blk in enumerate(net.blocks):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        act = blk.neurons(h)
                        h = h + blk.down(act).float()
                    S[l].index_add_(0, a[st:st + 2500], act.float().abs() * blk.colnorm())
        S = [s_ / cnt.clamp_min(1)[:, None] for s_ in S]
        torch.save(dict(C=C, S=S, cnt=cnt), p)
    t = torch.load(p, map_location=DEV)
    _RULES[key] = (t['C'], t['S'], feat)
    return _RULES[key]


def table2_ranks(x, cfg, R):
    C, S, feat = R
    r, K = cfg.get('r', 1), cfg['K']
    with torch.autocast('cuda', enabled=False):
        f = tfeat(x, feat)
        idx = nearest(f, C, r)                                         # (B, r)
        out = []
        for l in range(N_BLOCK):
            sc = S[l][idx].sum(1)                                      # (B, 4096) 가까운 r개 표의 평균 크기를 더한 점수
            order = sc.argsort(1, descending=True)
            rk = torch.empty_like(order)
            rk.scatter_(1, order, torch.arange(N_NEURON, device=x.device).expand(x.shape[0], N_NEURON))
            out.append(rk)
    return out


def lkey(h):
    """층 입력의 열쇠: 정규화(학습되는 배율 없이)한 층 입력. 신경망이 아니다."""
    return F.layer_norm(h.float(), (h.shape[-1],))


def get_ltable(D, cfg):
    """층 입력 열쇠의 표 연산: 층마다 베이스 모델(다 켬)의 층 입력을 k-평균으로 나눈 중심점 M개를 두고,
    중심점마다 근처 입력들이 뉴런마다 더한 것의 평균으로 순위를 매긴다. 그 층의 입력과 가장 가까운 중심점의 표를 켠다."""
    M = cfg.get('M', 64)
    key = ('ltable', M)
    if key in _RULES:
        return _RULES[key]
    p = os.path.join(rdir('rules'), f'ltable_M{M}.pt')
    if not os.path.exists(p):
        net = load_net(PICK['path'])
        X = D['xtr'].reshape(-1, 784)
        sub = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))[:max(20000, min(X.shape[0], 60 * M))].to(DEV)
        Cs, Ss = [], []
        with torch.no_grad():
            H = net.inp(X[sub])                                        # 층 1의 입력 (k-평균용 2만 장)
            for l, blk in enumerate(net.blocks):
                C = kmeans(lkey(H), M, torch.Generator(device=DEV).manual_seed(l), min_cluster=max(5, min(20, 6400 // M)))
                Cs.append(C)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    H = H + blk.down(blk.neurons(H)).float()
            S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
            cnt = [torch.zeros(M, device=DEV) for _ in range(N_BLOCK)]
            for st in range(0, X.shape[0], 2500):                     # 표: 학습 사진 전부
                h = net.inp(X[st:st + 2500])
                for l, blk in enumerate(net.blocks):
                    a_ = nearest(lkey(h), Cs[l], 1)[:, 0]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        act = blk.neurons(h)
                        h = h + blk.down(act).float()
                    S[l].index_add_(0, a_, act.float().abs() * blk.colnorm())
                    cnt[l] += torch.bincount(a_, minlength=M).float()
        RK = []
        for l in range(N_BLOCK):
            order = (S[l] / cnt[l].clamp_min(1)[:, None]).argsort(1, descending=True)
            rk = torch.empty_like(order)
            rk.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(M, N_NEURON))
            RK.append(rk)
        torch.save(dict(C=Cs, RK=RK, cnt=cnt), p)
    t = torch.load(p, map_location=DEV)
    _RULES[key] = (t['C'], t['RK'])
    return _RULES[key]


def get_dtable(D, cfg):
    """부호 열쇠와 방향 표: 층마다 고정된 무작위 방향 r개(P)에 정규화한 층 입력을 투영해 부호 r비트를 열쇠로 쓴다(2^r가지).
    뉴런마다 r차원 방향 q = 베이스 모델의 gate 가중치를 P에 투영해 정규화한 것. 열쇠 b마다 q . b가 큰 순서로 뉴런의 순위를 매긴다.
    뉴런이 켜지는 열쇠는 그 뉴런의 gate와 같은 쪽을 가리키는 입력이라, 뉴런의 문이 따라가기 쉽다."""
    rb = cfg.get('rbits', 8)
    key = ('dtable', rb)
    if key in _RULES:
        return _RULES[key]
    net = load_net(PICK['path'])
    Ps, RK = [], []
    codes = ((torch.arange(2 ** rb, device=DEV)[:, None] >> torch.arange(rb, device=DEV)) & 1).float() * 2 - 1   # (2^r, r) 열쇠마다 ±1
    for l, blk in enumerate(net.blocks):
        P = torch.randn(rb, D_MODEL, device=DEV, generator=torch.Generator(device=DEV).manual_seed(100 + l))
        q = blk.gate.weight.detach().float() @ P.T                       # (4096, r)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-6)
        sc = codes @ q.T                                                 # (2^r, 4096)
        order = sc.argsort(1, descending=True)
        rk = torch.empty_like(order)
        rk.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(2 ** rb, N_NEURON))
        Ps.append(P)
        RK.append(rk)
    _RULES[key] = ('dtable', Ps, RK, 2 ** torch.arange(rb, device=DEV))
    return _RULES[key]


def lrank(h, l, cfg, R):
    if R[0] == 'dtable':
        _, Ps, RK, pw = R
        with torch.autocast('cuda', enabled=False):
            j = ((lkey(h) @ Ps[l].T > 0).long() * pw).sum(1)
        return RK[l][j]
    C, RK = R
    with torch.autocast('cuda', enabled=False):
        j = nearest(lkey(h), C[l], 1)[:, 0]
    return RK[l][j]


def lrank_t(h, l, cfg, R):
    """층 입력 열쇠 연산의 (뉴런 순위, 켜는 문턱, 열쇠). lvtable은 문턱이 열쇠마다·층마다 다르다 (B, 1)."""
    if R[0] == 'lvtable':
        _, Cs, NR, KJ = R
        with torch.autocast('cuda', enabled=False):
            j = nearest(lkey(h), Cs[l], 1)[:, 0]
        return NR[l][j], KJ[l][j][:, None], j
    if R[0] == 'dtable':
        _, Ps, RK, pw = R
        with torch.autocast('cuda', enabled=False):
            j = ((lkey(h) @ Ps[l].T > 0).long() * pw).sum(1)
        return RK[l][j], cfg['K'], j
    C, RK = R
    with torch.autocast('cuda', enabled=False):
        j = nearest(lkey(h), C[l], 1)[:, 0]
    return RK[l][j], cfg['K'], j


LT_OPS = ('ltable', 'dtable', 'lvtable')


def get_rule(D, cfg):
    if cfg.get('op') == 'lvtable':                       # 층 입력 열쇠 표 + 열쇠·층마다 가변 K (파일: C 층마다 중심점, NR 순위, KJ 켤 수)
        if cfg['vrule'] not in _RULES:
            t = torch.load(cfg['vrule'], map_location=DEV)
            _RULES[cfg['vrule']] = ('lvtable', t['C'], t['NR'], t['KJ'])
        return _RULES[cfg['vrule']]
    if cfg.get('op') == 'vtable':
        if cfg['vrule'] not in _RULES:
            t = torch.load(cfg['vrule'], map_location=DEV)
            _RULES[cfg['vrule']] = ('vtable', t['W'], t['b'], t['NR'], t['KJ'])
        return _RULES[cfg['vrule']]
    if cfg.get('op') == 'dtable':
        return get_dtable(D, cfg)
    if cfg.get('op') == 'ltable':
        return get_ltable(D, cfg)
    if cfg.get('op') == 'table2':
        return get_table2(D, cfg)
    if cfg.get('op') == 'table':
        return get_table(D, cfg)
    """연산 = 입력에 고정된 무게를 곱한 점수에서 큰 r개 무리를 고른다 (신경망이 아니다, 학습하지 않는다).
    centroid: 중심점과의 거리가 가까운 순서 (무게 2C, 치우침 -|C|^2). random: 고정된 무작위 방향과의 내적."""
    ng, _ = ngs(cfg)
    M, op = ng - cfg['s'], cfg.get('op', 'centroid')
    key = (op, M)
    if key in _RULES:
        return _RULES[key]
    if op == 'centroid':
        p = os.path.join(rdir('rules'), f'centroid_M{M}.pt')
        if not os.path.exists(p):
            torch.save(kmeans(D['xtr'].reshape(-1, 784), M, torch.Generator(device=DEV).manual_seed(0)), p)
        C = torch.load(p, map_location=DEV)
        W, b = 2 * C, -(C * C).sum(1)
    else:
        g = torch.Generator(device=DEV).manual_seed(1)
        W = torch.randn(M, 784, device=DEV, generator=g) / 28.0
        b = torch.zeros(M, device=DEV)
    _RULES[key] = (W, b)
    return _RULES[key]


def pick(x, rule, r):
    W, b = rule
    return (x.reshape(x.shape[0], -1).float() @ W.T + b).topk(r, dim=1).indices


def group_ranks(x, cfg, C):
    """무리마다 순위 (B, 무리 수): 공통 무리는 0, 연산 무리는 점수가 큰 순서 0..r-1, 고르지 않은 무리는 r."""
    s, r = cfg['s'], cfg['r']
    ng, _ = ngs(cfg)
    with torch.autocast('cuda', enabled=False):
        B = x.shape[0]
        rk = torch.full((B, ng), r, dtype=torch.long, device=x.device)
        rk[:, :s] = 0
        idx = pick(x, C, r) + s
        rk.scatter_(1, idx, torch.arange(r, device=x.device).expand(B, r))
    return rk


def neuron_ranks(x, cfg, C):
    return group_ranks(x, cfg, C).repeat_interleave(ngs(cfg)[1], dim=1)   # 뉴런 i의 무리는 i // 무리 크기


def layer_ranks(x, cfg, R):
    """층마다 뉴런의 순위 (B, 4096)와 문턱 t: 순위 < t인 뉴런을 켜고, 줄이기 무게는 순위 / t.
    group 연산: 모든 층이 같은 순위 (무리의 순위), t = r.
    table 연산: 가장 가까운 중심점의 표에서 뉴런의 순위 (층마다 다른 표), t = K."""
    if cfg.get('op') == 'vtable':
        _, W, b, NR, KJ = R
        with torch.autocast('cuda', enabled=False):
            j = (x.reshape(x.shape[0], -1).float() @ W.T + b).argmax(1)
        return [nr[j] for nr in NR], [kj[j][:, None] for kj in KJ]       # 문턱이 입력마다, 층마다 다르다
    if cfg.get('op') == 'table2':
        return table2_ranks(x, cfg, R), cfg['K']
    if cfg.get('op') == 'table':
        W, b, RK = R
        with torch.autocast('cuda', enabled=False):
            j = (x.reshape(x.shape[0], -1).float() @ W.T + b).argmax(1)
        return [rk[j] for rk in RK], cfg['K']
    rk = neuron_ranks(x, cfg, R)
    return [rk] * N_BLOCK, cfg['r']


def kneurons(cfg):
    if cfg.get('op') in ('vtable', 'lvtable'):
        return None
    return cfg['K'] if cfg.get('op') in ('table', 'table2', 'ltable', 'dtable') else (cfg['s'] + cfg['r']) * ngs(cfg)[1]


def fwd(model, x, cfg, C, teacher=None):
    """teacher가 있으면 층마다 '출력 맞추기' 몫도 낸다: 같은 층 입력에서, 연산이 고른 뉴런만으로 낸 FFN 출력과
    다 켠 베이스 모델의 FFN 출력의 차이 (제곱 크기 / 베이스 출력의 제곱 크기)."""
    lt = cfg.get('op') in LT_OPS
    rks, t = (None, None) if lt else layer_ranks(x, cfg, C)
    h = model.inp(x.reshape(x.shape[0], -1))
    align, sparse, match = 0.0, 0.0, 0.0
    for l, b in enumerate(model.blocks):
        if lt:
            rk_l, tl, _ = lrank_t(h, l, cfg, C)
        else:
            rk_l, tl = rks[l], (t[l] if isinstance(t, list) else t)
        on = rk_l < tl
        w = (rk_l.float() / tl) * on                                   # 줄이기 무게 (공통 무리, 가장 가까운 무리, 표의 맨 앞은 0)
        a = b.neurons(h)
        c = a.float().abs() * b.colnorm()
        tot = c.sum(1) + 1e-6
        align = align + ((c * (~on)).sum(1) / tot).mean()
        sparse = sparse + ((c * w).sum(1) / tot).mean()
        out = b.down(a * on.to(a.dtype))
        if teacher is not None:
            with torch.no_grad():
                tb = teacher.blocks[l]
                tout = tb.down(tb.neurons(h.detach())).float()
            match = match + (((out.float() - tout) ** 2).sum(1) / ((tout ** 2).sum(1) + 1e-6)).mean()
        h = h + out
    L = len(model.blocks)
    lg = model.head(model.norm(h))
    if teacher is not None:
        return lg, align / L, sparse / L, match / L
    return lg, align / L, sparse / L


_TEACHER = {}


def get_teacher():
    if 't' not in _TEACHER:
        t = load_net(PICK['path'])
        t.requires_grad_(False)
        _TEACHER['t'] = t
    return _TEACHER['t']


def make_loss(model, cfg, C):
    ce = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    if cfg.get('match'):
        teacher = get_teacher()

        def fm(xb, yb):
            lg, al, sp, mt = fwd(model, xb, cfg, C, teacher)
            return (ce(lg.float(), yb) + cfg['lam'] * al + cfg['lam_s'] * sp + cfg['match'] * mt,
                    torch.stack([al, sp, mt]))
        return fm

    def f(xb, yb):
        lg, al, sp = fwd(model, xb, cfg, C)
        return ce(lg.float(), yb) + cfg['lam'] * al + cfg['lam_s'] * sp, torch.stack([al, sp])
    return f


def make_loss_ramp(model, cfg, C, lam_t):
    """연산에 맞추기의 세기가 텐서 lam_t인 손실 (출력 맞추기 포함). lam_t는 바퀴마다 바깥에서 바꾼다."""
    ce = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    teacher = get_teacher()

    def fm(xb, yb):
        lg, al, sp, mt = fwd(model, xb, cfg, C, teacher)
        return (ce(lg.float(), yb) + lam_t * al + cfg['lam_s'] * sp + cfg['match'] * mt, torch.stack([al, sp, mt]))
    return fm


@torch.no_grad()
def measure(model, x, y, cfg, C, chunk=2500):
    """시험 정확도, 맞힌 비율(개수: 모델 스스로의 필요한 뉴런 K개 가운데 연산이 고른 것에 든 비율),
    맡은 몫(크기: 뉴런 전체가 더하는 것 가운데 연산이 고른 뉴런이 더하는 몫). 층마다도."""
    model.eval()
    L = len(model.blocks)
    correct, n = 0, 0
    hit, mass, used = [0.0] * L, [0.0] * L, [0.0] * L
    lt = cfg.get('op') in LT_OPS
    kc = [None] * L                                                    # 열쇠마다 들어온 입력 수 (층 입력 열쇠일 때)
    for s in range(0, x.shape[0], chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        rks, t = (None, None) if lt else layer_ranks(xb, cfg, C)
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, b in enumerate(model.blocks):
            if lt:
                rk_l, tl, j = lrank_t(h, l, cfg, C)
                bc = torch.bincount(j, minlength=4096).float()            # 열쇠 수는 4,096개 이하
                kc[l] = bc if kc[l] is None else kc[l] + bc
            else:
                rk_l, tl = rks[l], (t[l] if isinstance(t, list) else t)
            on = rk_l < tl
            kk = on.sum(1)                                             # 이 입력이 켠 수
            used[l] += kk.float().sum().item()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = b.neurons(h)
            c = a.float().abs() * b.colnorm()
            rank = torch.empty_like(c, dtype=torch.long).scatter_(
                1, c.argsort(1, descending=True), torch.arange(c.shape[1], device=c.device).expand(c.shape[0], -1))
            own = rank < kk[:, None]                                   # 모델 스스로 더하는 것이 큰 kk개
            hit[l] += ((own & on).sum(1).float() / kk.clamp_min(1)).sum().item()
            mass[l] += ((c * on).sum(1) / (c.sum(1) + 1e-6)).sum().item()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = h + b.down(a * on.to(a.dtype)).float()
        correct += (model.head(model.norm(h)).argmax(1) == yb).sum().item()
        n += xb.shape[0]
    hit = [v / n for v in hit]
    mass = [v / n for v in mass]
    used = [100 * v / n / N_NEURON for v in used]
    out = dict(acc=100 * correct / n, hit=sum(hit) / L, mass=sum(mass) / L, avg_pct=sum(used) / L,
               hit_layers=[round(v, 4) for v in hit], mass_layers=[round(v, 4) for v in mass],
               used_layers=[round(v, 2) for v in used])
    if lt:                                                             # 열쇠가 몇 개로 쏠리면 표 구조가 망가진다: 가장 많이 받은 열쇠의 몫(%), 쓰인 열쇠 수
        out['key_top'] = [round(100 * k.max().item() / n, 2) for k in kc]
        out['keys_used'] = [int((k > 0).sum().item()) for k in kc]
    return out


def kpct(cfg):
    return 100 * kneurons(cfg) / N_NEURON


def new_model(cfg):
    if cfg.get('init'):
        return load_net(cfg['init'])
    torch.manual_seed(cfg.get('seed', 0))
    return Net().to(DEV)


def train(D, cfg):
    C = get_rule(D, cfg)
    d = rdir(cfg['name'])
    m = new_model(cfg)
    ramp = cfg.get('ramp')                               # 연산에 맞추기를 ramp바퀴에 걸쳐 0에서 lam까지 서서히 올린다
    lam_t = torch.tensor(0.0 if ramp else cfg['lam'], device=DEV)
    ep = [0]
    loss_fn = make_loss_ramp(m, cfg, C, lam_t) if ramp else make_loss(m, cfg, C)
    tr = Trainer(m, D['xtr'], D['ytr'], loss_fn, cfg['lr'], ndiag=3 if cfg.get('match') else 2)
    torch.manual_seed(cfg.get('seed', 0))
    if ramp:
        lam_t.fill_(cfg['lam'] / ramp)

    def evalf():
        if ramp:                                         # 이 바퀴가 끝났으니 다음 바퀴의 세기로
            ep[0] += 1
            lam_t.fill_(cfg['lam'] * min(1.0, (ep[0] + 1) / ramp))
        ev = measure(m, D['xva'], D['yva'], cfg, C)
        sc = ev['acc']
        if cfg.get('sel') and ev['mass'] < cfg['sel']:   # 잘 고르기 먼저: 문턱 전에는 맡은 몫이 오르는 것을 나아짐으로 센다
            sc = -200 + 100 * ev['mass']                 # (문턱 뒤의 정확도 점수보다 늘 낮다)
        out = dict(val_acc=ev['acc'], val_hit=ev['hit'], val_mass=ev['mass'], val_avg=ev['avg_pct'],
                   val_used=ev['used_layers'], score=sc)
        if 'key_top' in ev:                              # 층 입력 열쇠는 모델이 바뀌면 열쇠가 옮겨 간다: 켠 수와 열쇠 쏠림을 함께 본다
            out.update(key_top=ev['key_top'], keys_used=ev['keys_used'])
        return out
    res = train_plateau(tr, evalf, d, cfg['lr'], patience=cfg.get('patience', 3), factor=0.3, max_drops=3,
                        max_epochs=cfg.get('max_epochs', 100), seg_opt=False, seg_last=True,
                        final_opt=cfg.get('final_opt', False))
    te = measure(load_net(os.path.join(d, 'best.pt')), D['xte'], D['yte'], cfg, C)
    res.update(cfg=cfg, k_pct=te['avg_pct'], k=kneurons(cfg) if cfg.get('op') != 'vtable' else None,
               used_layers=te['used_layers'], test_acc=te['acc'], hit=te['hit'],
               mass=te['mass'], hit_layers=te['hit_layers'], mass_layers=te['mass_layers'],
               passed=(te['acc'] >= PASS) and not res['nan'],
               sel_ok=(te['mass'] >= cfg['sel']) if cfg.get('sel') else None)
    jdump(res, os.path.join(d, 'final.json'))
    print(f"[{cfg['name']}] 필요한 뉴런(K) 평균 {te['avg_pct']:.3g}% (층별 {te['used_layers']})  시험 정확도 {te['acc']:.2f}% "
          f"({'통과' if res['passed'] else '탈락'})  맞힌 비율 {te['hit']:.3f}  맡은 몫 {te['mass']:.3f}  바퀴 {res['epochs']}", flush=True)
    return res


def sweep_knob(D, base, knob, vals, lrs, steps, name):
    """다른 손잡이(knob) x 학습률을 훑는다. 버티는 가장 큰 값과 그 값에서 가장 좋은 학습률을 고른다. 끝에 걸리면 옮겨 다시."""
    C = get_rule(D, base)
    rows = []

    def one(v, lr):
        cfg = dict(base, lr=lr)
        cfg[knob] = v
        m = new_model(cfg)
        tr = Trainer(m, D['xtr'], D['ytr'], make_loss(m, cfg, C), lr, ndiag=3 if cfg.get('match') else 2)
        torch.manual_seed(0)
        tm = tr.run(steps)
        ev = measure(m, D['xva'], D['yva'], cfg, C)
        rows.append(dict(v=v, lr=lr, val_acc=ev['acc'], val_hit=ev['hit'], val_mass=ev['mass'], grad_norm=tm['grad_norm'],
                         nan_steps=tm['nan_steps'], diag=tm['train_diag']))
        print(dict(v=v, lr=lr, val_acc=round(ev['acc'], 2), val_hit=round(ev['hit'], 3), val_mass=round(ev['mass'], 3),
                   grad_norm=round(tm['grad_norm'], 3), nan=tm['nan_steps'], diag=[round(x, 4) for x in tm['train_diag']]), flush=True)
        del tr, m
        torch.cuda.empty_cache()
    for v in vals:
        for lr in lrs:
            one(v, lr)
    for it in range(3):
        ok = [x for x in rows if x['nan_steps'] == 0]
        top = max(x['val_acc'] for x in ok)
        v = next(u for u in vals if any(x['v'] == u for x in ok) and max(x['val_acc'] for x in ok if x['v'] == u) >= top - PASS_DROP)
        cand = sorted({x['lr'] for x in ok if x['v'] == v})
        best = max((x for x in ok if x['v'] == v), key=lambda x: x['val_acc'])
        if it == 2:                                      # 두 번 옮긴 뒤에는 다시 고르기만 한다
            break
        if best['lr'] == cand[0]:
            more = [cand[0] / 10, cand[0] / 3.16]
        elif best['lr'] == cand[-1]:
            more = [cand[-1] * 3.16, cand[-1] * 10]
        else:
            break
        print('학습률이 끝(%g)에 걸려 %s %g에서 %s를 더 훑는다' % (best['lr'], knob, v, more), flush=True)
        for lr in more:
            one(v, lr)
    out = dict(base=base, knob=knob, steps=steps, rows=rows, v=v, lr=best['lr'])
    jdump(out, os.path.join(rdir('sweeps'), name + '.json'))
    print('[%s] 고른 것: %s %g, 학습률 %g' % (name, knob, v, best['lr']), flush=True)
    return out


def sweep(D, base, lams, lrs, steps, name):
    """학습률 x 세기를 steps걸음씩 훑고, 버티는 가장 큰 세기와 그 세기에서 가장 좋은 학습률을 고른다. 끝에 걸리면 옮겨 다시."""
    C = get_rule(D, base)
    rows = []

    def one(lam, lr):
        cfg = dict(base, lam=lam, lam_s=lam if base.get('tie', True) else base['lam_s'], lr=lr)
        m = new_model(cfg)
        tr = Trainer(m, D['xtr'], D['ytr'], make_loss(m, cfg, C), lr, ndiag=3 if cfg.get('match') else 2)
        torch.manual_seed(0)
        tm = tr.run(steps)
        ev = measure(m, D['xva'], D['yva'], cfg, C)
        rows.append(dict(lam=lam, lr=lr, val_acc=ev['acc'], val_hit=ev['hit'], val_mass=ev['mass'],
                         grad_norm=tm['grad_norm'], nan_steps=tm['nan_steps'], align=tm['train_diag'][0], sparse=tm['train_diag'][1]))
        print({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        del tr, m
        torch.cuda.empty_cache()
    for lam in lams:
        for lr in lrs:
            one(lam, lr)
    for it in range(3):
        ok = [x for x in rows if x['nan_steps'] == 0]
        top = max(x['val_acc'] for x in ok)
        lam = next(l for l in lams if any(x['lam'] == l for x in ok) and max(x['val_acc'] for x in ok if x['lam'] == l) >= top - PASS_DROP)
        cand = sorted({x['lr'] for x in ok if x['lam'] == lam})
        best = max((x for x in ok if x['lam'] == lam), key=lambda x: x['val_acc'])
        if it == 2:
            break
        if best['lr'] == cand[0]:
            more = [cand[0] / 10, cand[0] / 3.16]
        elif best['lr'] == cand[-1]:
            more = [cand[-1] * 3.16, cand[-1] * 10]
        else:
            break
        print(f"학습률이 끝({best['lr']})에 걸려 세기 {lam}에서 {more}를 더 훑는다", flush=True)
        for lr in more:
            one(lam, lr)
    out = dict(base=base, steps=steps, rows=rows, lam=lam, lr=best['lr'], nan_runs=sum(x['nan_steps'] > 0 for x in rows))
    jdump(out, os.path.join(rdir('sweeps'), f'{name}.json'))
    print(f"[{name}] 고른 것: 세기 {lam}, 학습률 {best['lr']}  (NaN 실행 {out['nan_runs']}개)", flush=True)
    return out


# ---- 실험 묶음 ----
def set_A(D):
    """A: 처음부터 학습, 중심점 연산(공통 무리 없음), 세 압력. r마다 따로 처음부터."""
    sw = sweep(D, dict(s=0, r=4), [3.0, 1.0, 0.3], [1e-4, 3e-4, 1e-3], 400, 'A')
    for r in [32, 16, 8, 4, 2, 1]:
        train(D, dict(name=f'A_r{r}', s=0, r=r, lam=sw['lam'], lam_s=sw['lam'], lr=sw['lr']))


def set_B(D):
    """B: A에서 한 가지만 바꿔 정확도를 막는 원인을 가른다 (학습률은 A의 훑기 결과 1e-4).
    B1/B3: 줄이기 압력을 뺀다. B2/B4: 압력을 모두 빼고 성능만 남긴다."""
    for r in [16, 4]:
        train(D, dict(name=f'B_r{r}_nosparse', s=0, r=r, lam=0.3, lam_s=0.0, lr=1e-4))
        train(D, dict(name=f'B_r{r}_ceonly', s=0, r=r, lam=0.0, lam_s=0.0, lr=1e-4))


def set_R(D):
    """R: 기준. 같은 수의 뉴런을 늘 다 켜는 작은 모델(층마다 뉴런 h개)을 같은 방식으로 처음부터 학습한다.
    연산으로 고르는 모델이 이보다 낫지 않으면 작은 모델과 다를 것이 없다."""
    ce = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    for h in [1024, 256]:
        def make(seed=0):
            torch.manual_seed(seed)
            return Net(h=h).to(DEV)

        def lossf(m):
            def f(xb, yb):
                lg, _ = m(xb)
                return ce(lg.float(), yb), None
            return f
        rows = []
        for lr in [1e-4, 3e-4, 1e-3]:                 # 폭이 바뀌었으니 학습률을 다시 훑는다 (200걸음)
            m = make()
            tr = Trainer(m, D['xtr'], D['ytr'], lossf(m), lr)
            torch.manual_seed(0)
            tm = tr.run(200)
            rows.append((evaluate(m, D['xva'], D['yva'])['acc'], lr, tm['nan_steps']))
            del tr, m
        best = max(r for r in rows if r[2] == 0)
        lr = best[1]
        print(f'[R_h{h}] 훑기 {rows} -> 학습률 {lr}', flush=True)
        d = rdir(f'R_h{h}')
        m = make()
        tr = Trainer(m, D['xtr'], D['ytr'], lossf(m), lr)
        torch.manual_seed(0)

        def evalf():
            ev = evaluate(m, D['xva'], D['yva'])
            return dict(val_acc=ev['acc'], score=ev['acc'])
        res = train_plateau(tr, evalf, d, lr, patience=3, factor=0.3, max_drops=3, max_epochs=100, seg_opt=False)
        mb = Net(h=h).to(DEV)
        mb.load_state_dict(torch.load(os.path.join(d, 'best.pt'), map_location=DEV))
        te = evaluate(mb, D['xte'], D['yte'])
        res.update(h=h, k_pct=100 * h / N_NEURON, test_acc=te['acc'], sweep=rows)
        jdump(res, os.path.join(d, 'final.json'))
        print(f"[R_h{h}] 늘 다 켜는 작은 모델 (뉴런 {h}개 = {100 * h / N_NEURON:.3g}%)  시험 정확도 {te['acc']:.2f}%  바퀴 {res['epochs']}", flush=True)


def set_C(D):
    """C: 압력 없이 성능만 남긴 B를 기준으로, 연산 구조를 한 가지씩 바꾼다. 실행마다 학습률을 먼저 훑는다.
    C1/C3: 공통 무리(늘 켬)를 둔다. C2/C4: 무리를 뉴런 16개씩 256개로 잘게 나눈다."""
    runs = [dict(name='C1_r16_shared', s=8, r=8),
            dict(name='C2_r16_fine', s=0, r=64, ng=256),
            dict(name='C3_r4_shared', s=2, r=2),
            dict(name='C4_r4_fine', s=0, r=16, ng=256)]
    for cfg in runs:
        sw = sweep(D, dict(cfg, lam=0.0, lam_s=0.0), [0.0], [3e-5, 1e-4, 3e-4], 400, cfg['name'])
        train(D, dict(cfg, lam=0.0, lam_s=0.0, lr=sw['lr']))


def set_D(D):
    """D: 표 연산(중심점마다 필요한 뉴런 표, 표끼리 겹쳐도 된다)으로 베이스 모델에서 시작해 세 압력으로 사후 학습.
    필요한 뉴런(K)을 50%부터 반씩 줄이며 앞 단계의 모델에서 이어 배운다. 떨어지면 사이를 한 번 더."""
    base = dict(op='table', M=64, init=PICK['path'], patience=4)
    sw = sweep(D, dict(base, K=512), [3.0, 1.0, 0.3], [3e-6, 1e-5, 3e-5], 2000, 'D')
    init, last, fail = PICK['path'], None, None
    for f in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
        K = round(N_NEURON * f)
        res = train(D, dict(base, name=f'D_K{K}', K=K, lam=sw['lam'], lam_s=sw['lam'], lr=sw['lr'], init=init))
        if not res['passed']:
            fail = K
            break
        last = (K, os.path.join(RUNS3, f'D_K{K}', 'best.pt'))
        init = last[1]
    if last and fail:
        K = round(math.sqrt(last[0] * fail))
        train(D, dict(base, name=f'D_K{K}', K=K, lam=sw['lam'], lam_s=sw['lam'], lr=sw['lr'], init=last[1]))


def set_E1(D):
    """E1: D_K1024(세기 1, 탈락 90.50%)에서 세기 하나만 3으로 키워 평원까지. 같은 출발점(D_K2048)."""
    train(D, dict(op='table', M=64, K=1024, lam=3.0, lam_s=3.0, lr=1e-5, patience=4,
                  init=os.path.join(RUNS3, 'D_K2048', 'best.pt'), name='E1_K1024_lam3'))


def set_E2(D):
    """E2: D_K1024(세기 1)에서 '출력 맞추기' 압력 하나만 더한다. 세기는 큰 값부터 훑고, 통과하면 K를 반씩 줄인다."""
    base = dict(op='table', M=64, K=1024, lam=1.0, lam_s=1.0, patience=4, init=os.path.join(RUNS3, 'D_K2048', 'best.pt'))
    p = os.path.join(RUNS3, 'sweeps', 'E2.json')
    sw = jload(p) if os.path.exists(p) else sweep_knob(D, base, 'match', [10.0, 3.0, 1.0], [1e-5, 3e-5], 2000, 'E2')
    init = base['init']
    for K in [1024, 512, 256, 128]:
        res = train(D, dict(base, K=K, match=sw['v'], lr=sw['lr'], init=init, name='E2_K%d' % K))
        if not res['passed']:
            break
        init = os.path.join(RUNS3, 'E2_K%d' % K, 'best.pt')


def chain_K(D, base, lr, name, fracs=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)):
    """필요한 뉴런(K)을 큰 것부터 반씩 줄이며 앞 단계의 모델에서 이어 배운다. 떨어지면 사이를 한 번 더."""
    init, last, fail = base['init'], None, None
    for f in fracs:
        K = round(N_NEURON * f)
        res = train(D, dict(base, K=K, lr=lr, init=init, name='%s_K%d' % (name, K)))
        if not res['passed']:
            fail = K
            break
        last = (K, os.path.join(RUNS3, '%s_K%d' % (name, K), 'best.pt'))
        init = last[1]
    if last and fail:
        K = round(math.sqrt(last[0] * fail))
        train(D, dict(base, K=K, lr=lr, init=last[1], name='%s_K%d' % (name, K)))


def set_E2r(D):
    """E2r: E2에서 빠졌던 단계. 통과한 25%(E2_K1024)와 떨어진 12.5% 사이(K = 724, 17.7%)를 한 번 더."""
    sw = jload(os.path.join(RUNS3, 'sweeps', 'E2.json'))
    train(D, dict(op='table', M=64, K=724, lam=1.0, lam_s=1.0, match=sw['v'], lr=sw['lr'], patience=4,
                  init=os.path.join(RUNS3, 'E2_K1024', 'best.pt'), name='E2_K724'))


def set_E3(D):
    """E3: E2에서 연산 하나만 '층 입력 열쇠 표(중심점 256개)'로 바꾼다. 출력 맞추기 세기 10 그대로, 학습률은 다시 훑는다.
    베이스 모델에서 시작해 필요한 뉴런(K)을 50%부터 반씩 줄인다."""
    base = dict(op='ltable', M=256, lam=1.0, lam_s=1.0, match=10.0, patience=4, init=PICK['path'])
    p = os.path.join(RUNS3, 'sweeps', 'E3.json')
    sw = jload(p) if os.path.exists(p) else sweep_knob(D, dict(base, K=1024), 'match', [10.0], [3e-6, 1e-5, 3e-5], 2000, 'E3')
    chain_K(D, base, sw['lr'], 'E3')


def set_F(D):
    """F: 잘 고르기 먼저 (사용자가 정한 순서: 잘 고르기 > 정확도 > 적게 고르기).
    필요한 뉴런(K) 50%, 층 입력 열쇠 표(중심점 256개), 출력 맞추기 10, 줄이기 1.
    연산에 맞추기의 세기를 큰 값부터(30, 10, 3) 바꿔 평원까지 학습해, 맡은 몫 95% 이상과 정확도 통과를 함께 만드는 세기를 찾는다.
    학습률은 연산이 바뀌었으니 세기 10에서 먼저 훑는다."""
    base = dict(op='ltable', M=256, K=2048, lam_s=1.0, match=10.0, patience=4, init=PICK['path'])
    p = os.path.join(RUNS3, 'sweeps', 'F.json')
    sw = jload(p) if os.path.exists(p) else sweep_knob(D, dict(base, lam=10.0), 'lam', [10.0], [3e-6, 1e-5, 3e-5], 2000, 'F')
    for lam in [30.0, 10.0, 3.0]:
        train(D, dict(base, lam=lam, lr=sw['lr'], name='F_K2048_lam%g' % lam))


def set_F2(D):
    """F2: F에서 고르는 기준만 '잘 고르기 먼저'로 바꾼다 (맡은 몫 95% 이상인 바퀴 가운데 검증 정확도가 가장 높은 것).
    F의 훑기는 정확도로만 골라 학습률이 0 쪽으로 갔다(3e-7). 훑기에서 맡은 몫 95% 언저리가 나온 학습률로 평원까지 학습한다."""
    base = dict(op='ltable', M=256, K=2048, lam_s=1.0, match=10.0, patience=4, init=PICK['path'], sel=0.95)
    for lam, lr in [(10.0, 3e-5), (30.0, 1e-5), (30.0, 3e-5)]:
        train(D, dict(base, lam=lam, lr=lr, name='F2_K2048_lam%g_lr%g' % (lam, lr)))


def set_F3(D):
    """F3: F2(세기 10, 학습률 3e-5)에서 하나씩만 바꿔, 잘 고르기 95%를 지키며 정확도를 되찾는다.
    F3a: 출력 맞추기 10 -> 30.  F3b: 연산에 맞추기를 10바퀴에 걸쳐 0에서 10까지 서서히 올린다."""
    base = dict(op='ltable', M=256, K=2048, lam=10.0, lam_s=1.0, match=10.0, lr=3e-5, patience=4, init=PICK['path'], sel=0.95)
    # F3a(출력 맞추기 30)는 일정 실수(가장 좋은 점수를 -1에서 시작)로 학습률이 일찍 낮아졌지만, 같은 학습률이던 처음 바퀴에서도
    # F2보다 정확도가 낮아 다시 돌리지 않는다. 실수를 고친 뒤 F2(세기 10)를 다시 돌려 기준으로 삼는다.
    # 점수를 고친 뒤(문턱 전에는 맡은 몫, 뒤에는 정확도) 다시 돌린다. 이전 실행(F2fix_lam10, F3b_ramp10)은 문턱 전 점수가
    # 정확도 - 100이라 학습률이 일찍 낮아졌다.
    train(D, dict(base, name='F2s_lam10'))
    train(D, dict(base, ramp=10, name='F3s_ramp10'))


def set_G(D):
    """G: F3s(층 입력 열쇠 표 256, 50%, 세기 10을 10바퀴에 서서히, 출력 맞추기 10, 잘 고르기 먼저)에서 연산 하나만
    '부호 열쇠와 방향 표'(8비트)로 바꾼다. 먼저 학습 없이 출발점을 잰다."""
    base = dict(op='dtable', rbits=8, K=2048, lam=10.0, lam_s=1.0, match=10.0, lr=3e-5, patience=4, init=PICK['path'],
                sel=0.95, ramp=10)
    C = get_rule(D, base)
    net = load_net(PICK['path'])
    for f in [0.5, 0.25, 0.125]:
        ev = measure(net, D['xte'], D['yte'], dict(base, K=round(N_NEURON * f)), C)
        print('[G 학습 없이] 필요한 뉴런(K) %g%%  시험 정확도 %.2f%%  맞힌 비율 %.3f  맡은 몫 %.3f' % (f * 100, ev['acc'], ev['hit'], ev['mass']), flush=True)
    train(D, dict(base, name='G_dtable8_ramp10'))


def set_H(D):
    """H: F3s(층 입력 열쇠 표 256, 50%, 세기 10 서서히, 출력 맞추기 10, 잘 고르기 먼저)에서 하나씩만 바꾼다.
    H1: 줄이기 압력을 뺀다 (50%에서는 줄일 필요가 없고, 고른 뉴런 가운데 순위가 낮은 것까지 눌러 표현력을 깎을 수 있다).
    H2: 중심점 256 -> 1,024개 (학습 없이 잴 때 모델의 원래 필요에 더 가까웠다)."""
    base = dict(op='ltable', M=256, K=2048, lam=10.0, lam_s=1.0, match=10.0, lr=3e-5, patience=4, init=PICK['path'],
                sel=0.95, ramp=10)
    if not os.path.exists(os.path.join(RUNS3, 'H1_nosparse', 'final.json')):
        train(D, dict(base, lam_s=0.0, name='H1_nosparse'))
    train(D, dict(base, M=1024, name='H2_M1024'))


def set_I(D):
    """I1: F3s 설정(줄이기 압력은 H1처럼 뺀다)에서 필요한 뉴런(K)을 75%로 넉넉히 준다. 잘 고르기 95%와 정확도 통과를 함께 만족하는
    자리를 먼저 찾는다.  I2: H1을 더 길게. H1의 가장 좋은 모델에서 학습률 1e-5, 참을성 8바퀴로 새로 시작한다."""
    base = dict(op='ltable', M=256, lam=10.0, lam_s=0.0, match=10.0, patience=4, init=PICK['path'], sel=0.95)
    train(D, dict(base, K=3072, lr=3e-5, ramp=10, name='I1_K3072'))
    train(D, dict(base, K=2048, lr=1e-5, patience=8, init=os.path.join(RUNS3, 'H1_nosparse', 'best.pt'), name='I2_H1long'))


@torch.no_grad()
def build_vtable(D, model, cfg, R, tau, path):
    """지금 모델과 지금 연산(R)으로, 학습 사진에서 열쇠마다·층마다 켜진 뉴런들이 실제로 더하는 크기의 합을 구한다.
    그 크기로 다시 줄 세우고(지금 켜지지 않는 뉴런은 맨 뒤), 몫 tau를 채우는 개수를 그 열쇠·층의 K로 한다 (지금의 K보다 크지 않게)."""
    if R[0] == 'vtable':
        _, W, b, NR0, KJ0 = R
    else:
        W, b, RK = R
        NR0 = RK
        KJ0 = [torch.full((RK[0].shape[0],), cfg['K'], device=DEV, dtype=torch.long) for _ in range(N_BLOCK)]
    M = NR0[0].shape[0]
    S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
    X = D['xtr']
    model.eval()
    for st in range(0, X.shape[0], 2500):
        xb = X[st:st + 2500]
        with torch.autocast('cuda', enabled=False):
            j = (xb.reshape(xb.shape[0], -1).float() @ W.T + b).argmax(1)
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            on = NR0[l][j] < KJ0[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
            S[l].index_add_(0, j, a.float().abs() * blk.colnorm() * on)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = h + blk.down(a * on.to(a.dtype)).float()
    NR, KJ = [], []
    for l in range(N_BLOCK):
        Sl = S[l].clone()
        Sl[NR0[l] >= KJ0[l][:, None]] = -1
        order = Sl.argsort(1, descending=True)
        nr = torch.empty_like(order)
        nr.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(M, N_NEURON))
        srt = Sl.clamp_min(0).gather(1, order)
        cum = srt.cumsum(1) / (srt.sum(1, keepdim=True) + 1e-12)
        kj = torch.minimum((cum < tau).sum(1) + 1, KJ0[l])
        NR.append(nr)
        KJ.append(kj)
    torch.save(dict(W=W, b=b, NR=NR, KJ=KJ, tau=tau), path)
    return path


def set_J(D):
    """J: 가변 K 사후 학습을 되풀이한다. E2_K1024(25% 고정, 통과)에서 시작해, 지금 모델로 열쇠·층마다 K를 몫 tau로 정하고
    (E2와 같은 방식으로) 사후 학습한다. 통과하면 학습된 모델로 tau를 낮춰 다시. 떨어지면 멈춘다. E2에서 바뀐 것은 가변 K 하나."""
    sw = jload(os.path.join(RUNS3, 'sweeps', 'E2.json'))
    prev_cfg = dict(op='table', M=64, K=1024)
    prev_model = os.path.join(RUNS3, 'E2_K1024', 'best.pt')
    R = get_rule(D, prev_cfg)
    for i, tau in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]):
        name = 'J%d_tau%g' % (i + 1, tau)
        path = build_vtable(D, load_net(prev_model), prev_cfg, R, tau, os.path.join(rdir('rules'), 'vtable_%s.pt' % name))
        cfg = dict(op='vtable', vrule=path, lam=1.0, lam_s=1.0, match=sw['v'], lr=sw['lr'], patience=4, init=prev_model, name=name)
        C = get_rule(D, cfg)
        ev = measure(load_net(prev_model), D['xte'], D['yte'], cfg, C)
        print('[%s 학습 전] 필요한 뉴런(K) 평균 %.2f%% (층별 %s)  시험 정확도 %.2f%%' % (name, ev['avg_pct'], ev['used_layers'], ev['acc']), flush=True)
        res = train(D, cfg)
        if not res['passed']:
            break
        prev_cfg, prev_model, R = cfg, os.path.join(RUNS3, name, 'best.pt'), C


SETS = {'A': set_A, 'B': set_B, 'R': set_R, 'C': set_C, 'D': set_D, 'E1': set_E1, 'E2': set_E2, 'E2r': set_E2r,
        'E3': set_F,   # 2026-11-22 사용자가 잘 고르기를 먼저로 정해, 걸어 둔 E3 자리에서 F를 돌린다
        'E3_old': set_E3, 'F': set_F, 'F2': set_F2, 'F3': set_F3, 'G': set_G, 'H': set_H, 'I': set_I, 'J': set_J}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['queue'])
    ap.add_argument('--set', required=True)
    a = ap.parse_args()
    D = load_data()
    SETS[a.set](D)
