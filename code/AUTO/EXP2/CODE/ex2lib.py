"""자율 실험 2의 공용 부품: 짝 비교 판정, 연산으로 가린 앞으로 계산, 열쇠·층마다 가변 K 만들기(몫 기준, 출력 기준).

판정 (사용자와 정한 것, 2026-11-22)
  - 같은 사진에서 베이스 모델(다 켠 모델, 시드 2)과 짝지어 견준다: 사진마다 d = (베이스가 맞힘) - (이 모델이 맞힘).
    하락 = d의 평균, 95% 신뢰 구간 = 평균 +- 1.96 x 표준오차.
  - 하락이 0.5%p 이하면 통과. 구간 전체가 0.5%p 안이면 '확실한 통과', 구간이 0.5%p에 걸치면 '경계', 하락이 0.5%p를 넘으면 '탈락'.
  - 설정은 검증 사진(5,000장)으로만 고르고, 시험 사진(10,000장)으로는 고른 설정 하나만 판정한다.
"""
import math
import os
import sys

HERE2 = os.path.dirname(os.path.abspath(__file__))
AUTO1 = os.path.join(os.path.dirname(os.path.dirname(HERE2)), 'EXP1', 'CODE')
sys.path.insert(0, AUTO1)
import auto
import common
from auto import *

RUNS_A1 = os.path.join(AUTO1, 'runs')
RUNS_E2 = os.path.join(HERE2, 'runs')
auto.RUNS3 = common.RUNS = RUNS_E2


# ---- 판정 ----
def base_preds(D):
    m = load_net(PICK['path'])
    return {s: evaluate(m, D['x' + s], D['y' + s])['probs'].argmax(1) for s in ('va', 'te')}


def paired(pb, pc, y):
    d = (pb == y).float() - (pc == y).float()
    n = d.numel()
    m = d.mean().item() * 100
    se = d.std().item() / math.sqrt(n) * 100
    lo, hi = m - 1.96 * se, m + 1.96 * se
    verdict = '확실한 통과' if hi <= PASS_DROP else ('경계' if m <= PASS_DROP else '탈락')
    return dict(acc=(pc == y).float().mean().item() * 100, drop=m, lo=lo, hi=hi, verdict=verdict,
                base_only=int(((pb == y) & (pc != y)).sum()), cand_only=int(((pb != y) & (pc == y)).sum()))


def fmt(j):
    return '정확도 %.2f%%, 하락 %.2f%%p [%.2f, %.2f], %s (베이스만 맞힘 %d, 이 모델만 맞힘 %d)' % (
        j['acc'], j['drop'], j['lo'], j['hi'], j['verdict'], j['base_only'], j['cand_only'])


# ---- 연산으로 가린 앞으로 계산 ----
@torch.no_grad()
def run_masked(model, x, prep, mask, chunk=2500):
    """prep(xb) -> 문맥(예: 열쇠), mask(l, h, ctx) -> 층 l에서 켤 뉴런 (B, 4096) bool. 반환: 예측, 층마다 켠 뉴런 평균(%)."""
    model.eval()
    preds, used = [], [0.0] * N_BLOCK
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        ctx = prep(xb)
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            on = mask(l, h, ctx)
            used[l] += on.sum().item()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                h = h + blk.down(a * on.to(a.dtype)).float()
        preds.append(model.head(model.norm(h)).argmax(1))
    n = x.shape[0]
    used = [100 * u / n / N_NEURON for u in used]
    return torch.cat(preds), sum(used) / N_BLOCK, [round(u, 2) for u in used]


def image_keys(W, b):
    def prep(xb):
        with torch.autocast('cuda', enabled=False):
            return (xb.reshape(xb.shape[0], -1).float() @ W.T + b).argmax(1)
    return prep


def fixed_table(W, b, RK, K):
    return image_keys(W, b), (lambda l, h, j: RK[l][j] < K)


def var_table(W, b, NR, KJ):
    return image_keys(W, b), (lambda l, h, j: NR[l][j] < KJ[l][j][:, None])


# ---- 열쇠·층마다 가변 K ----
@torch.no_grad()
def table_stats(D, model, W, b, NR0, KJ0, kgrid=None):
    """학습 사진으로, 지금 켜진 뉴런(NR0 < KJ0)이 열쇠마다 더하는 크기의 합 S, 그 크기로 다시 줄 세운 순위 NR.
    kgrid를 주면 출력 기준도 잰다: 열쇠마다, 층마다, 새 순위의 앞 k개만 켜서 낸 FFN 출력이 지금 켜진 뉴런 전부로 낸 출력과
    얼마나 다른지 (차이의 제곱 / 출력의 제곱, 사진 평균)."""
    M = NR0[0].shape[0]
    prep = image_keys(W, b)
    X = D['xtr']
    model.eval()
    S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
    for s in range(0, X.shape[0], 2500):
        xb = X[s:s + 2500]
        j = prep(xb)
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            on = NR0[l][j] < KJ0[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
            S[l].index_add_(0, j, a.float().abs() * blk.colnorm() * on)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = h + blk.down(a * on.to(a.dtype)).float()
    NR, CUM = [], []
    for l in range(N_BLOCK):
        Sl = S[l].clone()
        Sl[NR0[l] >= KJ0[l][:, None]] = -1
        order = Sl.argsort(1, descending=True)
        nr = torch.empty_like(order)
        nr.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(M, N_NEURON))
        NR.append(nr)
        srt = Sl.clamp_min(0).gather(1, order)
        CUM.append(srt.cumsum(1) / (srt.sum(1, keepdim=True) + 1e-12))
    out = dict(NR=NR, CUM=CUM, M=M)
    if kgrid is not None:
        E = [torch.zeros(M, len(kgrid), device=DEV) for _ in range(N_BLOCK)]
        cnt = torch.zeros(M, device=DEV)
        for s in range(0, X.shape[0], 2500):
            xb = X[s:s + 2500]
            j = prep(xb)
            cnt += torch.bincount(j, minlength=M).float()
            h = model.inp(xb.reshape(xb.shape[0], -1))
            for l, blk in enumerate(model.blocks):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a = blk.neurons(h)
                    full_on = NR[l][j] < KJ0[l][j][:, None]
                    full = blk.down(a * full_on.to(a.dtype)).float()
                    for ki, k in enumerate(kgrid):
                        on = NR[l][j] < torch.minimum(torch.full_like(KJ0[l][j], k), KJ0[l][j])[:, None]
                        o = blk.down(a * on.to(a.dtype)).float()
                        err = ((o - full) ** 2).sum(1) / ((full ** 2).sum(1) + 1e-6)
                        E[l][:, ki].index_add_(0, j, err)
                    h = h + full
        out['E'] = [e / cnt.clamp_min(1)[:, None] for e in E]
        out['kgrid'] = kgrid
    return out


def kj_tau(st, KJ0, taus):
    """몫 기준: 층 l, 열쇠 j의 K = 새 순위로 쌓은 크기가 몫 taus[l]을 채우는 개수 (지금의 K보다 크지 않게)."""
    out = []
    for l, t in enumerate(taus):
        if t >= 1.0:
            out.append(KJ0[l].clone())
        else:
            out.append(torch.minimum((st['CUM'][l] < t).sum(1) + 1, KJ0[l]))
    return out


def kj_eps(st, KJ0, epss):
    """출력 기준: 층 l, 열쇠 j의 K = 출력 차이가 epss[l] 이하가 되는 가장 작은 k (격자 위에서)."""
    kg = torch.tensor(st['kgrid'], device=DEV)
    out = []
    for l, e in enumerate(epss):
        ok = st['E'][l] <= e                                           # (M, 격자)
        first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((st['M'],), len(st['kgrid']) - 1, device=DEV, dtype=torch.long))
        out.append(torch.minimum(kg[first], KJ0[l]))
    return out


# ---- 열쇠가 층마다 다른 경우(층 입력 열쇠 표)까지 다루는 일반판 ----
def keys_image(W, b):
    """사진 열쇠: 처음에 한 번 구하고 모든 층이 같은 열쇠를 쓴다."""
    prep = image_keys(W, b)
    return prep, (lambda l, h, ctx: ctx)


def keys_layer(C):
    """층 입력 열쇠: 층마다 그 층의 입력(정규화)과 가장 가까운 중심점. 실제 LLM에서는 어텐션과 동시에 구한다."""
    def keyfn(l, h, ctx):
        with torch.autocast('cuda', enabled=False):
            return nearest(lkey(h), C[l], 1)[:, 0]
    return (lambda xb: None), keyfn


def var_mask(prep, keyfn, NR, KJ):
    def mask(l, h, ctx):
        j = keyfn(l, h, ctx)
        return NR[l][j] < KJ[l][j][:, None]
    return prep, mask


@torch.no_grad()
def table_stats2(D, model, prep, keyfn, M, NR0, KJ0, kgrid=None):
    """table_stats와 같지만 열쇠를 층마다 keyfn(l, h, ctx)로 구한다."""
    X = D['xtr']
    model.eval()
    S = [torch.zeros(M, N_NEURON, device=DEV) for _ in range(N_BLOCK)]
    for s in range(0, X.shape[0], 2500):
        xb = X[s:s + 2500]
        ctx = prep(xb)
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            j = keyfn(l, h, ctx)
            on = NR0[l][j] < KJ0[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
            S[l].index_add_(0, j, a.float().abs() * blk.colnorm() * on)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = h + blk.down(a * on.to(a.dtype)).float()
    NR, CUM = [], []
    for l in range(N_BLOCK):
        Sl = S[l].clone()
        Sl[NR0[l] >= KJ0[l][:, None]] = -1
        order = Sl.argsort(1, descending=True)
        nr = torch.empty_like(order)
        nr.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(M, N_NEURON))
        NR.append(nr)
        srt = Sl.clamp_min(0).gather(1, order)
        CUM.append(srt.cumsum(1) / (srt.sum(1, keepdim=True) + 1e-12))
    out = dict(NR=NR, CUM=CUM, M=M)
    if kgrid is not None:
        E = [torch.zeros(M, len(kgrid), device=DEV) for _ in range(N_BLOCK)]
        cnt = [torch.zeros(M, device=DEV) for _ in range(N_BLOCK)]
        for s in range(0, X.shape[0], 2500):
            xb = X[s:s + 2500]
            ctx = prep(xb)
            h = model.inp(xb.reshape(xb.shape[0], -1))
            for l, blk in enumerate(model.blocks):
                j = keyfn(l, h, ctx)
                cnt[l] += torch.bincount(j, minlength=M).float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a = blk.neurons(h)
                    full_on = NR[l][j] < KJ0[l][j][:, None]
                    full = blk.down(a * full_on.to(a.dtype)).float()
                    for ki, k in enumerate(kgrid):
                        on = NR[l][j] < torch.minimum(torch.full_like(KJ0[l][j], k), KJ0[l][j])[:, None]
                        o = blk.down(a * on.to(a.dtype)).float()
                        E[l][:, ki].index_add_(0, j, ((o - full) ** 2).sum(1) / ((full ** 2).sum(1) + 1e-6))
                    h = h + full
        out['E'] = [E[l] / cnt[l].clamp_min(1)[:, None] for l in range(N_BLOCK)]
        out['kgrid'] = kgrid
    return out
