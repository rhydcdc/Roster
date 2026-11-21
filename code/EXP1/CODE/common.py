"""실험 1 공통 부품: 데이터, 모델, 빠른 학습 걸음(CUDA 그래프), 오르지 않으면 넘어가는 일정, 평가, 묶기 비용."""
import json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda'
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RUNS = os.path.join(ROOT, 'runs')

# ---- 판 ----
D_MODEL, N_NEURON, N_BLOCK, N_CLASS = 1024, 4096, 4, 10
BATCH = 256
MEAN, STD = 0.2860, 0.3530
PASS_DROP = 0.5            # 통과: 정확도 하락 0.5%p 이내
WD, SMOOTH, CLIP, WARMUP = 0.05, 0.1, 1.0, 100

# ---- GPU 사양 (RTX 4060 Laptop) ----
# bf16 텐서 연산(FP32 누적, 희소 없음) = AI TOPS 233 / 8, 대역폭 = 128비트 x 16Gbps
GPU_FLOPS = 233e12 / 8
GPU_BW = 256e9
# c = 뉴런 하나의 가중치를 읽는 시간 / (요청 1 x 뉴런 1)을 계산하는 시간.
# bf16에서 둘 다 6d(바이트, FLOP)라 c = FLOPS / BW
C_READ = GPU_FLOPS / GPU_BW


def rdir(name):
    p = os.path.join(RUNS, name)
    os.makedirs(p, exist_ok=True)
    return p


def jdump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def jload(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class Log:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')

    def __call__(self, **kw):
        self.f.write(json.dumps(kw, ensure_ascii=False) + '\n')
        self.f.flush()
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in kw.items()), flush=True)


# ---- 데이터: 처음에 GPU에 통째로 올린다 ----
def load_data():
    from torchvision.datasets import FashionMNIST
    tr = FashionMNIST(DATA_DIR, train=True, download=True)
    te = FashionMNIST(DATA_DIR, train=False, download=True)

    def prep(ds):
        return ds.data.to(DEV).float().div_(255).sub_(MEAN).div_(STD), ds.targets.to(DEV)
    xtr, ytr = prep(tr)
    xte, yte = prep(te)
    perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0)).to(DEV)
    va, tn = perm[:5000], perm[5000:]
    return dict(xtr=xtr[tn].contiguous(), ytr=ytr[tn].contiguous(),
                xva=xtr[va].contiguous(), yva=ytr[va].contiguous(), xte=xte, yte=yte)


_AR = {}


def augment(x):
    """무작위 자르기(사방 2칸 덧대기)와 좌우 뒤집기. GPU에서 묶음 단위로 한 번에, 그래프 안에서도 돈다."""
    B = x.shape[0]
    if x.device not in _AR:
        _AR[x.device] = torch.arange(28, device=x.device)
    ar = _AR[x.device]
    xp = F.pad(x, (2, 2, 2, 2), value=-MEAN / STD).reshape(B, 32 * 32)
    dy = torch.randint(0, 5, (B, 1, 1), device=x.device)
    dx = torch.randint(0, 5, (B, 1, 1), device=x.device)
    flat = (dy + ar.view(1, 28, 1)) * 32 + (dx + ar.view(1, 1, 28))
    out = xp.gather(1, flat.reshape(B, 784)).reshape(B, 28, 28)
    flip = torch.rand(B, 1, 1, device=x.device) < 0.5
    return torch.where(flip, out.flip(-1), out)


# ---- 모델: FFN 블록만 ----
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def neurons(self, x):                      # 뉴런 출력값 (B, h)
        z = self.norm(x)
        return F.silu(self.gate(z)) * self.up(z)

    def colnorm(self):                         # 나가는 가중치의 크기 (h,)
        return self.down.weight.float().norm(dim=0)


class Net(nn.Module):
    def __init__(self, d=D_MODEL, h=N_NEURON, L=N_BLOCK):
        super().__init__()
        self.inp = nn.Linear(784, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASS)

    def forward(self, x, k=None, masks=None, pen=False, info=False, keep=False):
        """k: 층마다 '더하는 것'이 큰 위 k개만 켠다. masks: 층마다 켤 뉴런(bool, (B,h))을 바깥에서 준다.
        pen: 밑으로 내리는 힘(꺼진 뉴런이 더하던 것의 몫)을 돌려준다. info: 켠 뉴런과 진단 수치를 돌려준다.
        keep: 켠 뉴런만 돌려준다."""
        h = self.inp(x.reshape(x.shape[0], -1))
        aux = dict(pen=0.0, masks=[], offr=[])
        for l, b in enumerate(self.blocks):
            a = b.neurons(h)
            if k is None and masks is None:
                h = h + b.down(a)
                continue
            cn = b.colnorm()
            ac = a.float().abs() * cn                       # 더하는 것의 크기
            if masks is not None:
                m = masks[l].to(a.dtype)
            else:
                idx = ac.detach().topk(k, dim=1).indices
                m = torch.zeros_like(a).scatter_(1, idx, 1.0)
            out = b.down(a * m)
            if pen:
                off = (ac * (1 - m.float())).sum(1) / (ac.sum(1) + 1e-6)
                aux['pen'] = aux['pen'] + off.mean() / len(self.blocks)
            if info or keep:
                aux['masks'].append(m.bool())
            if info:
                full = b.down(a)
                aux['offr'].append(((full - out).float().norm(dim=1) / (full.float().norm(dim=1) + 1e-6)).mean())
            h = h + out
        return self.head(self.norm(h)), aux


class Predictor(nn.Module):
    """입력 사진만 보고 모든 층의 뉴런 점수를 한 번에 낸다."""

    def __init__(self, hid, L=N_BLOCK, h=N_NEURON):
        super().__init__()
        self.L, self.h, self.hid = L, h, hid
        self.body = nn.Sequential(nn.Linear(784, hid), nn.LayerNorm(hid), nn.GELU())
        self.heads = nn.Linear(hid, L * h)

    def forward(self, x):
        return self.heads(self.body(x.reshape(x.shape[0], -1))).view(x.shape[0], self.L, self.h)

    def macs(self):                             # 요청 하나의 곱셈 수 = 가중치 수
        return 784 * self.hid + self.hid * self.L * self.h


def param_groups(model):
    dec = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    nod = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    return [dict(params=dec, weight_decay=WD), dict(params=nod, weight_decay=0.0)]


# ---- 빠른 학습 걸음 ----
class Trainer:
    """학습 한 걸음(앞으로, 뒤로, 자르기, AdamW)을 CUDA 그래프 하나로 잡는다. graph=False면 같은 계산을 그냥 돈다."""

    def __init__(self, model, X, Y, loss_fn, lr, graph=True, aug=True, batch=BATCH, ndiag=1):
        self.model, self.X, self.Y, self.loss_fn = model, X, Y, loss_fn
        self.graph, self.aug, self.batch = graph, aug, batch
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(param_groups(model), lr=torch.tensor(lr, device=DEV),
                                     betas=(0.9, 0.999), fused=True, capturable=True)
        self.idx = torch.zeros(batch, dtype=torch.long, device=DEV)
        self.ndiag = ndiag
        self.acc = torch.zeros(3 + ndiag, device=DEV)  # 손실 합, 기울기 크기 합, NaN 걸음 수, 진단 합(ndiag개)
        self.base_lr, self.step_no = lr, 0
        self._prepare()

    def _body(self):
        xb, yb = self.X[self.idx], self.Y[self.idx]
        if self.aug:
            xb = augment(xb)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            loss, diag = self.loss_fn(xb, yb)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.params, CLIP, foreach=True)
        self.opt.step()
        ld = loss.detach().float()
        self.acc[0] += ld
        self.acc[1] += gn.float()
        self.acc[2] += (~torch.isfinite(ld)).float()
        if torch.is_tensor(diag):
            self.acc[3:] += diag.detach().float().reshape(-1)

    def _prepare(self):
        snap = [p.detach().clone() for p in self.params]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3 if self.graph else 1):    # 옵티마이저 상태를 만들고 그래프를 덥힌다
                self.opt.zero_grad(set_to_none=True)
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        if self.graph:
            self.opt.zero_grad(set_to_none=True)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self._body()
        with torch.no_grad():                          # 덥히느라 바뀐 것을 처음으로 되돌린다
            for p, q in zip(self.params, snap):
                p.copy_(q)
            for st in self.opt.state.values():
                for v in st.values():
                    v.zero_()
        self.acc.zero_()

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'].fill_(lr)

    def step(self):
        if self.step_no < WARMUP:
            self.set_lr(self.base_lr * (self.step_no + 1) / WARMUP)
        elif self.step_no == WARMUP:
            self.set_lr(self.base_lr)
        if self.graph:
            self.g.replay()
        else:
            self.opt.zero_grad(set_to_none=True)
            self._body()
        self.step_no += 1

    def run(self, nsteps=None):
        """한 바퀴(nsteps=None) 또는 nsteps 걸음을 돌고 평균 진단을 돌려준다."""
        n = self.X.shape[0]
        per = n // self.batch
        total = per if nsteps is None else nsteps
        self.acc.zero_()
        done = 0
        while done < total:
            perm = torch.randperm(n, device=DEV)
            for i in range(min(per, total - done)):
                self.idx.copy_(perm[i * self.batch:(i + 1) * self.batch])
                self.step()
                done += 1
        a = self.acc.tolist()
        dg = [v / total for v in a[3:]]
        return dict(train_loss=a[0] / total, grad_norm=a[1] / total, nan_steps=int(a[2]),
                    train_diag=dg[0] if self.ndiag == 1 else dg)

    # 체크포인트: 옵티마이저 상태는 그래프가 잡은 텐서에 제자리로 복사한다
    def opt_tensors(self):
        return [self.opt.state[p] for g in self.opt.param_groups for p in g['params']]

    def state(self):
        return dict(model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    opt=[{k: v.detach().cpu() for k, v in st.items()} for st in self.opt_tensors()],
                    step_no=self.step_no)

    def load(self, ck):
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                v.copy_(ck['model'][k])
            for st, sv in zip(self.opt_tensors(), ck.get('opt', [])):   # 모델만 담긴 체크포인트면 옵티마이저는 새로
                for k, v in st.items():
                    v.copy_(sv[k])
        self.step_no = ck['step_no']


def train_plateau(tr, evalf, run_dir, lr0, patience, factor, max_drops, max_epochs, resume=None, extra=None, seg_opt=True,
                  seg_last=False, final_opt=True):
    """오르지 않으면 넘어가는 일정. 검증 점수가 patience 바퀴 동안 오르지 않으면 구간 체크포인트를 남기고
    학습률을 factor배로 낮춘다. max_drops번 낮춘 뒤에도 오르지 않으면 멈춘다. 가장 좋은 모델은 best.pt."""
    log = Log(os.path.join(run_dir, 'log.jsonl'))
    sch = dict(epoch=0, lr=lr0, best=float('-inf'), since=0, drops=0, seg=0, nan=False)   # 점수가 음수여도 첫 바퀴가 가장 좋은 것이 되게
    if resume:
        # 구간 체크포인트에서 다음 구간을 시작한다. lr0를 주면 그 구간의 학습률만 바꿔 갈라 돌린다
        ck = torch.load(resume, map_location='cpu')
        tr.load(ck)
        sch.update(ck['sched'])
        sch['lr'] = lr0 if lr0 is not None else sch['lr'] * factor
        sch['since'], sch['seg'], sch['drops'] = 0, sch['seg'] + 1, sch['drops'] + 1
        src = os.path.join(os.path.dirname(os.path.abspath(resume)), 'best.pt')
        if os.path.exists(src) and os.path.dirname(src) != os.path.abspath(run_dir):
            import shutil
            shutil.copy(src, os.path.join(run_dir, 'best.pt'))
    tr.base_lr = sch['lr']
    if tr.step_no >= WARMUP:
        tr.set_lr(sch['lr'])
    best_path = os.path.join(run_dir, 'best.pt')

    def save_seg(full=True):
        # seg_opt=False면 구간 체크포인트에는 모델만 남기고(205MB), 옵티마이저 상태는 마지막 것에만 남긴다
        ck = tr.state() if full else dict(model=tr.state()['model'], step_no=tr.step_no)
        ck['sched'] = dict(sch)
        # seg_last=True면 구간 체크포인트를 seg_last.pt 하나에 덮어써서 마지막 구간만 남긴다 (디스크를 아낀다)
        torch.save(ck, os.path.join(run_dir, 'seg_last.pt' if seg_last else f"seg{sch['seg']}.pt"))

    while sch['epoch'] < max_epochs:
        t0 = time.time()
        tm = tr.run()
        tr.model.eval()
        ev = evalf()
        tr.model.train()
        sch['epoch'] += 1
        log(epoch=sch['epoch'], seg=sch['seg'], lr=sch['lr'], **tm, **ev, sec=round(time.time() - t0, 2))
        if tm['nan_steps'] > 0 or not math.isfinite(tm['train_loss']):
            sch['nan'] = True
            log(event='NaN', epoch=sch['epoch'])
            break
        if ev['score'] > sch['best']:
            sch['best'], sch['since'] = ev['score'], 0
            torch.save({k: v.detach().cpu() for k, v in tr.model.state_dict().items()}, best_path)
        else:
            sch['since'] += 1
        if sch['since'] >= patience:
            save_seg(seg_opt)
            if sch['drops'] >= max_drops:
                break
            sch['lr'] *= factor
            sch['drops'] += 1
            sch['seg'] += 1
            sch['since'] = 0
            tr.base_lr = sch['lr']
            tr.set_lr(sch['lr'])
    save_seg(final_opt)            # final_opt=False면 마지막 체크포인트도 모델만 (614MB -> 205MB, 갈라 돌릴 때 옵티마이저는 새로)
    out = dict(best_val=sch['best'], epochs=sch['epoch'], nan=sch['nan'])
    if extra:
        out.update(extra)
    return out


# ---- 평가 ----
@torch.no_grad()
def evaluate(model, x, y, k=None, masks=None, info=False, chunk=2500):
    """정확도, 손실, 확률. info면 켠 뉴런(층마다 bool (N,h)), 꺼진 뉴런이 더하던 것의 몫, 한 번도 안 켜진 뉴런 비율."""
    model.eval()
    probs, offr, ms = [], [], [[] for _ in range(N_BLOCK)]
    for s in range(0, x.shape[0], chunk):
        mk = None if masks is None else [m[s:s + chunk] for m in masks]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, aux = model(x[s:s + chunk], k=k, masks=mk, info=info)
        probs.append(lg.float().softmax(-1))
        if info and aux['masks']:
            offr.append(torch.stack(aux['offr']))
            for l, m in enumerate(aux['masks']):
                ms[l].append(m)
    P = torch.cat(probs)
    out = dict(acc=(P.argmax(1) == y).float().mean().item() * 100,
               loss=F.nll_loss(P.clamp_min(1e-12).log(), y).item(), probs=P)
    if info and offr:
        M = [torch.cat(m) for m in ms]
        out['masks'] = M
        out['offr'] = torch.stack(offr).mean().item()
        out['dead'] = sum(1 - m.any(0).float().mean().item() for m in M) / len(M) * 100
    return out


def confidence(P0, P1, y):
    """둘 다 맞힌 사진에서 정답 확률의 평균 변화, 원래와 답이 달라진 사진의 비율(%)."""
    a0, a1 = P0.argmax(1), P1.argmax(1)
    both = (a0 == y) & (a1 == y)
    py0, py1 = P0.gather(1, y[:, None])[:, 0], P1.gather(1, y[:, None])[:, 0]
    return dict(conf_change=(py1 - py0)[both].mean().item(), flip=(a0 != a1).float().mean().item() * 100)


def load_net(path):
    m = Net().to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV))
    return m


# ---- 묶기와 비용 (단위: 요청 1 x 뉴런 1의 계산, 뉴런 1의 가중치 읽기) ----
def greedy_group(M, c):
    """M: (n, N) 0/1. 요청마다 혼자인 묶음에서 시작해 비용이 가장 많이 주는 두 묶음을 합치고, 줄지 않으면 멈춘다.
    묶음 비용 = 켠 뉴런 수 x (요청 수 + c)."""
    n = M.shape[0]
    G = M.float().clone()
    size = torch.ones(n, device=DEV)
    alive = torch.ones(n, dtype=torch.bool, device=DEV)
    owner = torch.arange(n, device=DEV)
    u = G.sum(1)
    inter = G @ G.T
    eye = torch.eye(n, dtype=torch.bool, device=DEV)
    ninf = torch.tensor(float('-inf'), device=DEV)
    while True:
        U = u[:, None] + u[None, :] - inter
        sep = u * (size + c)
        gain = sep[:, None] + sep[None, :] - U * (size[:, None] + size[None, :] + c)
        gain = torch.where(alive[:, None] & alive[None, :] & ~eye, gain, ninf)
        best = gain.argmax()
        bi, gv = best.item(), gain.view(-1)[best].item()
        if not gv > 0:
            break
        i, j = divmod(bi, n)
        G[i] = torch.maximum(G[i], G[j])
        G[j] = 0
        size[i] += size[j]
        size[j] = 0
        alive[j] = False
        owner[owner == j] = i
        u[i], u[j] = G[i].sum(), 0
        col = G @ G[i]
        inter[i], inter[:, i] = col, col
    return [(owner == r).nonzero().flatten() for r in alive.nonzero().flatten()]


def capped_group(M, cap):
    """정해진 크기로 묶기: 켜는 뉴런이 가장 많이 겹치는 두 묶음부터 합치되, 합친 크기가 cap을 넘지 않게 한다."""
    n = M.shape[0]
    G = M.float().clone()
    size = torch.ones(n, device=DEV)
    alive = torch.ones(n, dtype=torch.bool, device=DEV)
    owner = torch.arange(n, device=DEV)
    inter = G @ G.T
    eye = torch.eye(n, dtype=torch.bool, device=DEV)
    ninf = torch.tensor(float('-inf'), device=DEV)
    while True:
        ok = alive[:, None] & alive[None, :] & ~eye & (size[:, None] + size[None, :] <= cap)
        sc = torch.where(ok, inter, ninf)
        b = sc.argmax()
        if sc.view(-1)[b].item() == float('-inf'):
            break
        i, j = divmod(b.item(), n)
        G[i] = torch.maximum(G[i], G[j])
        G[j] = 0
        size[i] += size[j]
        size[j] = 0
        alive[j] = False
        owner[owner == j] = i
        col = G @ G[i]
        inter[i], inter[:, i] = col, col
    return [(owner == r).nonzero().flatten() for r in alive.nonzero().flatten()]


def batch_costs(M, c):
    """한 층, 한 묶음(요청 n개)의 네 방식. 반환: 방식별 (계산량, 읽는 양)과 우리 묶음의 진단."""
    n, N = M.shape
    Mf = M.float()
    own = Mf.sum().item()
    groups = greedy_group(M, c)
    sizes = [len(g) for g in groups]
    us = [Mf[g].amax(0).sum().item() for g in groups]
    ours = (sum(s * u for s, u in zip(sizes, us)), sum(us))
    arr, s0 = [0.0, 0.0], 0                      # 들어온 순서로 묶기: 우리 묶음 크기 그대로, 순서대로 채운다
    for s in sizes:
        u = Mf[s0:s0 + s].amax(0).sum().item()
        arr[0] += s * u
        arr[1] += u
        s0 += s
    union_all = Mf.amax(0).sum().item()
    return dict(dense=(n * N, N), own=(own, own), arrival=tuple(arr), ours=ours,
                n_groups=len(groups), mean_size=n / len(groups), max_size=max(sizes),
                overlap=sum(us) - union_all)


def grouping_report(masks, order, n_req, c, extra_cost=None):
    """masks: 층마다 bool (N_test, h). order: 시험 사진을 섞은 순서(= 들어온 순서). n_req개씩 묶음을 나눠 다 잰다.
    extra_cost: (요청 하나의 계산량, 묶음 하나의 읽는 양) — 예측기 비용을 뉴런 단위로 더한다(다 켬 제외)."""
    tot = {k: [0.0, 0.0] for k in ('dense', 'own', 'arrival', 'ours')}
    diag = dict(n_groups=0.0, mean_size=0.0, max_size=0, overlap=0.0)
    nb = len(order) // n_req
    for b in range(nb):
        ids = order[b * n_req:(b + 1) * n_req]
        for l in range(len(masks)):
            r = batch_costs(masks[l][ids], c)
            for k in tot:
                tot[k][0] += r[k][0]
                tot[k][1] += r[k][1]
            diag['n_groups'] += r['n_groups'] / (nb * len(masks))
            diag['mean_size'] += r['mean_size'] / (nb * len(masks))
            diag['max_size'] = max(diag['max_size'], r['max_size'])
            diag['overlap'] += r['overlap'] / (nb * len(masks))
        if extra_cost:
            for k in ('own', 'arrival', 'ours'):
                tot[k][0] += n_req * extra_cost[0]
                tot[k][1] += extra_cost[1]
    rep = {}
    d = tot['dense']
    for k, (comp, read) in tot.items():
        cost = comp + c * read
        rep[k] = dict(compute=comp, read=read, cost=cost,
                      x_compute=d[0] / comp, x_read=d[1] / read, x_cost=(d[0] + c * d[1]) / cost)
    rep['diag'] = diag
    return rep


def print_report(title, rep):
    print(f'\n[{title}]  (다 켬 대비 몇 배 주는가)')
    print(f"{'방식':<10}{'계산량':>10}{'읽는 양':>10}{'비용':>10}")
    for k, name in (('dense', '다 켬'), ('own', '제 것만'), ('arrival', '들어온 순서'), ('ours', '우리')):
        r = rep[k]
        print(f"{name:<10}{r['x_compute']:>10.2f}{r['x_read']:>10.2f}{r['x_cost']:>10.2f}")
    print('묶음 진단:', {k: round(v, 2) for k, v in rep['diag'].items()})
