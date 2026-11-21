"""1단계: 원래 모델(다 켬)을 최대한 잘 학습한다.

python s1_train.py check              빠르게 만든 걸음이 같은 계산을 하는지 20걸음으로 견준다
python s1_train.py sweep              학습률 후보를 200걸음씩 훑는다
python s1_train.py run --seed 0       본 실행 (학습률은 sweep 결과, --lr로 바꿀 수 있다)
python s1_train.py run --seed 0 --resume runs/s1_seed0/seg1.pt [--lr X]   구간 체크포인트에서 이어/갈라 돌린다
python s1_train.py pick               시드 3개 가운데 시험 정확도가 가운데인 모델을 고르고 흔들림 폭을 적는다
python s1_train.py all                sweep -> 시드 0,1,2 -> pick
"""
import argparse
import os
import torch
from common import *

LR_CANDS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]   # 처음 1e-4 ~ 1e-2로 훑으니 1e-4(끝)가 가장 좋아서 아래로 옮겼다
SEEDS = [0, 1, 2]
SWEEP_STEPS = 200


def make(seed):
    torch.manual_seed(seed)
    return Net().to(DEV)


def loss_dense(model):
    ce = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)

    def f(xb, yb):
        lg, _ = model(xb)
        return ce(lg.float(), yb), None
    return f


def check(D):
    losses = {}
    for graph in (False, True):
        m = make(0)
        tr = Trainer(m, D['xtr'], D['ytr'], loss_dense(m), 1e-3, graph=graph, aug=False)
        torch.manual_seed(1)
        losses[graph] = [tr.run(1)['train_loss'] for _ in range(20)]
        del tr, m
        torch.cuda.empty_cache()
    diff = max(abs(a - b) for a, b in zip(losses[False], losses[True]))
    print('그냥 돈 손실 :', [round(v, 4) for v in losses[False]])
    print('그래프 손실  :', [round(v, 4) for v in losses[True]])
    print(f'가장 큰 차이 {diff:.2e}  ->', '같다' if diff < 1e-2 else '다르다: 확인 필요')


def sweep(D):
    rows = []
    for lr in LR_CANDS:
        m = make(0)
        tr = Trainer(m, D['xtr'], D['ytr'], loss_dense(m), lr)
        torch.manual_seed(0)
        tm = tr.run(SWEEP_STEPS)
        ev = evaluate(m, D['xva'], D['yva'])
        rows.append(dict(lr=lr, val_acc=ev['acc'], val_loss=ev['loss'], **tm))
        print(rows[-1], flush=True)
        del tr, m
        torch.cuda.empty_cache()
    ok = [r for r in rows if r['nan_steps'] == 0]
    best = max(ok, key=lambda r: r['val_acc'])
    i = LR_CANDS.index(best['lr'])
    edge = i in (0, len(LR_CANDS) - 1)
    if edge:
        print('경고: 가장 좋은 학습률이 후보의 끝에 있다. 범위를 옮겨 다시 훑어야 한다')
    print(f"NaN이 된 실행 {len(rows) - len(ok)}개, 가장 좋은 학습률 {best['lr']}")
    jdump(dict(rows=rows, best_lr=best['lr'], edge=edge), os.path.join(rdir('s1_sweep'), 'sweep.json'))
    return best['lr']


def run(D, seed, lr, resume=None):
    d = rdir(f's1_seed{seed}')
    m = make(seed)
    tr = Trainer(m, D['xtr'], D['ytr'], loss_dense(m), lr or 1e-3)
    torch.manual_seed(seed)

    def evalf():
        ev = evaluate(m, D['xva'], D['yva'])
        return dict(val_acc=ev['acc'], val_loss=ev['loss'], score=ev['acc'])
    res = train_plateau(tr, evalf, d, lr, patience=3, factor=0.3, max_drops=3, max_epochs=100, resume=resume)
    te = evaluate(load_net(os.path.join(d, 'best.pt')), D['xte'], D['yte'])
    res.update(seed=seed, lr=lr, test_acc=te['acc'], test_loss=te['loss'])
    jdump(res, os.path.join(d, 'final.json'))
    print(res)


def pick():
    rs = sorted((jload(os.path.join(RUNS, f's1_seed{s}', 'final.json')) for s in SEEDS), key=lambda r: r['test_acc'])
    med = rs[len(rs) // 2]
    spread = rs[-1]['test_acc'] - rs[0]['test_acc']
    out = dict(seed=med['seed'], test_acc=med['test_acc'], spread=spread,
               all={r['seed']: r['test_acc'] for r in rs},
               path=os.path.join(RUNS, f"s1_seed{med['seed']}", 'best.pt'))
    jdump(out, os.path.join(RUNS, 's1_pick.json'))
    print(out)
    if spread >= PASS_DROP:
        print(f'경고: 시드에 따른 흔들림 {spread:.2f}%p가 통과 기준 {PASS_DROP}%p에 닿는다. 기준을 다시 본다')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'sweep', 'run', 'pick', 'all'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--resume', default=None)
    a = ap.parse_args()
    if a.cmd == 'pick':
        pick()
        raise SystemExit
    D = load_data()
    if a.cmd == 'check':
        check(D)
    elif a.cmd == 'sweep':
        sweep(D)
    elif a.cmd == 'run':
        lr = a.lr if (a.lr or a.resume) else jload(os.path.join(RUNS, 's1_sweep', 'sweep.json'))['best_lr']
        run(D, a.seed, lr, a.resume)
    else:
        lr = sweep(D)
        for s in SEEDS:
            run(D, s, lr)
        pick()
