"""다음 1 (실행 L): 층마다 가변 K로 사후 학습.
F_K2048_lam3(층 입력 열쇠 표, 중심점 256개, 50%, 시험 91.00%)에서 시작해
  줄이기(출력 기준, 검증 문턱 = F의 검증 정확도) -> 사후 학습 -> 다시 줄이기 를 되풀이한다.

F에서 바뀐 것은 하나: 고정 K(50%) -> 열쇠·층마다 가변 K(출력 기준). 세 압력의 세기(연산에 맞추기 3, 줄이기 1, 출력 맞추기 10)는 F 그대로.
학습률: F의 학습률(3e-7)은 함정 1(기준 가까이에서 짧게 훑고 정확도로만 고름)로 고른 것이라 모델이 거의 움직이지 않았다.
  그래서 첫 바퀴에서 후보 3e-6, 1e-5, 3e-5를 평원까지 돌리고, '학습 뒤 다시 줄였을 때 켜는 뉴런 평균이 가장 적은 것'으로 고른다(검증만 씀).
  고른 것이 끝에 걸리면 한 칸 넓힌다. 다음 바퀴부터는 고른 학습률.
검증 문턱: 모든 바퀴에서 'F를 줄이지 않았을 때의 검증 정확도 이상' (고정된 기준점).
멈춤: 한 바퀴에 평균이 1%p 넘게 줄지 않거나, 6바퀴.
최종: 마지막 바퀴의 (모델, 표)를 미리 최종으로 정해 둔다. 시험 짝 비교는 바퀴마다 적지만 고르는 데 쓰지 않는다.

python lchain.py check    연산 확인(학습 없이 잰 값과 같은가, 그래프 걸음과 그냥 돈 걸음의 손실이 같은가)
python lchain.py run      되풀이
"""
from ex2lib import *

F_NAME = 'F_K2048_lam3'
F_PATH = os.path.join(RUNS_A1, F_NAME, 'best.pt')
K0 = 2048
EPS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
KGRID = list(range(32, K0 + 1, 32))
# 처음(L)에는 F와 같은 세기(연산에 맞추기 3)로 돌렸으나 모든 학습률에서 검증이 문턱 아래로 내려가 0바퀴에서 멈췄다.
# lpress.py로 가르니 깎는 것은 연산에 맞추기 압력이었다(빼면 4바퀴 만에 문턱을 넘으며 오른다). 그래서 Lb는 그것 하나만 0으로.
# (가린 앞으로 계산이 이미 연산이 고른 뉴런만 쓰므로, 고르지 않은 뉴런의 '쓰고 싶음'을 누르는 이 압력은 계산 결과에 직접 닿지 않는다.)
TAG = 'Lb'
PRESS = dict(lam=0.0, lam_s=1.0, match=10.0, patience=4)
LRS = [3e-6, 1e-5, 3e-5]
MAX_ROUNDS, MIN_GAIN = 6, 1.0
OUT = rdir('lchain')
RULES = rdir('rules')


def start(D):
    C, RK = get_rule(D, dict(op='ltable', M=256, K=K0))
    M = RK[0].shape[0]
    KJ0 = [torch.full((M,), K0, device=DEV, dtype=torch.long) for _ in range(N_BLOCK)]
    return C, M, RK, KJ0


def val_of(D, model, C, NR, KJ, s='va'):
    prep, keyfn = keys_layer(C)
    p, avg, pl = run_masked(model, D['x' + s], *var_mask(prep, keyfn, NR, KJ))
    return p, (p == D['y' + s]).float().mean().item() * 100, avg, pl


def reduce(D, model, C, M, NR0, KJ0, thr):
    """출력 기준으로 열쇠·층마다 K를 다시 정한다. eps 가운데 검증 정확도가 thr 이상이면서 켜는 뉴런 평균이 가장 적은 것."""
    prep, keyfn = keys_layer(C)
    st = table_stats2(D, model, prep, keyfn, M, NR0, KJ0, kgrid=KGRID)
    rows, best = [], None
    for e in EPS:
        KJ = kj_eps(st, KJ0, [e] * N_BLOCK)
        _, a, avg, pl = val_of(D, model, C, st['NR'], KJ)
        rows.append(dict(eps=e, val_acc=round(a, 2), avg=round(avg, 2), layers=pl))
        if a >= thr and (best is None or avg < best['avg']):
            best = dict(eps=e, val_acc=a, avg=avg, layers=pl, NR=st['NR'], KJ=KJ)
    return best, rows


def save_table(C, NR, KJ, name):
    path = os.path.join(RULES, 'lv_%s.pt' % name)
    torch.save(dict(C=C, NR=NR, KJ=KJ), path)
    return path


def judge(D, PB, model, C, NR, KJ):
    p, _, avg, pl = val_of(D, model, C, NR, KJ, 'te')
    return dict(paired(PB['te'], p, D['yte']), avg=avg, layers=pl)


def check(D):
    """(1) 표를 lvtable 연산으로 불러 auto.measure로 잰 값이 run_masked로 잰 값과 같은가. (2) 그래프 걸음 = 그냥 돈 걸음."""
    C, M, RK, KJ0 = start(D)
    F_ = load_net(F_PATH)
    _, vF, aF, _ = val_of(D, F_, C, RK, KJ0)
    prep, keyfn = keys_layer(C)
    st = table_stats2(D, F_, prep, keyfn, M, RK, KJ0, kgrid=KGRID)
    KJ = kj_eps(st, KJ0, [0.001] * N_BLOCK)
    _, v1, a1, p1 = val_of(D, F_, C, st['NR'], KJ)
    path = save_table(C, st['NR'], KJ, 'check')
    cfg = dict(PRESS, op='lvtable', vrule=path, lr=1e-5)
    ev = measure(F_, D['xva'], D['yva'], cfg, get_rule(D, cfg))
    print('F 줄이지 않음: 검증 %.2f%%, 평균 %.2f%%' % (vF, aF))
    print('eps 0.001  run_masked: 검증 %.2f%%, 평균 %.3f%% %s' % (v1, a1, p1))
    print('eps 0.001  measure   : 검증 %.2f%%, 평균 %.3f%% %s  열쇠 쏠림 %s, 쓰인 열쇠 %s' % (
        ev['acc'], ev['avg_pct'], ev['used_layers'], ev['key_top'], ev['keys_used']))
    losses = {}
    for g in (False, True):
        m = load_net(F_PATH)
        tr = Trainer(m, D['xtr'], D['ytr'], make_loss(m, cfg, get_rule(D, cfg)), 1e-5, graph=g, aug=False, ndiag=3)
        ls = []
        for i in range(10):
            tr.idx.copy_(torch.arange(i * BATCH, (i + 1) * BATCH, device=DEV))
            tr.acc.zero_()
            tr.step()
            ls.append(tr.acc[0].item())
        losses[g] = ls
        del tr, m
    print('그냥 돈 걸음 손실:', [round(x, 6) for x in losses[False]])
    print('그래프 걸음 손실 :', [round(x, 6) for x in losses[True]])
    print('가장 큰 차이:', max(abs(a - b) for a, b in zip(losses[False], losses[True])))
    os.remove(path)


def train_round(D, path, init, lr, name):
    cfg = dict(PRESS, op='lvtable', vrule=path, lr=lr, init=init, name=name)
    res = train(D, cfg)
    return os.path.join(RUNS_E2, name, 'best.pt'), res


def run(D):
    PB = base_preds(D)
    C, M, NR, KJ = start(D)
    model_path = F_PATH
    F_ = load_net(F_PATH)
    _, thr, avg0, _ = val_of(D, F_, C, NR, KJ)
    log = dict(thr=thr, rounds=[])
    print('[L] 검증 문턱(F를 줄이지 않은 검증 정확도) %.2f%%, 시작 평균 %.2f%%' % (thr, avg0), flush=True)

    # 0바퀴: 학습 없이 줄이기
    best, rows = reduce(D, F_, C, M, NR, KJ, thr)
    NR, KJ = best['NR'], best['KJ']
    path = save_table(C, NR, KJ, TAG + '0')
    te = judge(D, PB, F_, C, NR, KJ)
    log['rounds'].append(dict(round=0, model=model_path, eps=best['eps'], val=best['val_acc'], avg=best['avg'],
                              layers=best['layers'], reduce_rows=rows, test=te))
    print('[L0] 학습 없이 줄임: eps %g, 검증 %.2f%%, 평균 %.2f%% %s | [시험] %s' % (
        best['eps'], best['val_acc'], best['avg'], best['layers'], fmt(te)), flush=True)
    jdump(log, os.path.join(OUT, TAG + '.json'))
    cur_avg, lr = best['avg'], None

    for r in range(1, MAX_ROUNDS + 1):
        cands = []
        lrs = LRS if lr is None else [lr]
        tried = set()
        while True:
            for x in lrs:
                if x in tried:
                    continue
                tried.add(x)
                name = TAG + '%d_lr%g' % (r, x)
                bp, res = train_round(D, path, model_path, x, name)
                m = load_net(bp)
                _, v_after, a_after, _ = val_of(D, m, C, NR, KJ)
                b2, rows2 = reduce(D, m, C, M, NR, KJ, thr)
                c = dict(lr=x, name=name, path=bp, epochs=res['epochs'], val_trained=v_after, avg_trained=a_after,
                         reduce_rows=rows2, best=b2)
                cands.append(c)
                print('[L%d 학습률 %g] 학습 뒤 검증 %.2f%% (평균 %.2f%%) -> 다시 줄이면 %s' % (
                    r, x, v_after, a_after,
                    ('eps %g, 검증 %.2f%%, 평균 %.2f%% %s' % (b2['eps'], b2['val_acc'], b2['avg'], b2['layers'])) if b2 else '문턱을 지키는 eps 없음'),
                    flush=True)
                del m
                torch.cuda.empty_cache()
            ok = [c for c in cands if c['best'] is not None]
            if not ok or lr is not None:
                break
            pickc = min(ok, key=lambda c: (c['best']['avg'], -c['best']['val_acc']))
            xs = sorted(c['lr'] for c in cands)
            if pickc['lr'] == xs[0] and xs[0] / 3.16 not in tried and len(tried) < 5:
                lrs = [xs[0] / 3.16]
            elif pickc['lr'] == xs[-1] and xs[-1] * 3.16 not in tried and len(tried) < 5:
                lrs = [xs[-1] * 3.16]
            else:
                break
            print('[L%d] 고른 학습률 %g이 후보의 끝이라 %g를 더 돌린다' % (r, pickc['lr'], lrs[0]), flush=True)
        ok = [c for c in cands if c['best'] is not None]
        rec = dict(round=r, cands=[{k: v for k, v in c.items() if k != 'best'} | (
            dict(best={k: v for k, v in c['best'].items() if k not in ('NR', 'KJ')}) if c['best'] else {}) for c in cands])
        if not ok:
            rec['stop'] = '문턱을 지키며 줄일 수 있는 후보가 없다'
            log['rounds'].append(rec)
            jdump(log, os.path.join(OUT, TAG + '.json'))
            print('[L%d] 멈춤: %s' % (r, rec['stop']), flush=True)
            break
        pickc = min(ok, key=lambda c: (c['best']['avg'], -c['best']['val_acc']))
        if lr is None:
            lr = pickc['lr']
            log['lr'] = lr
        b = pickc['best']
        gain = cur_avg - b['avg']
        model_path, NR, KJ = pickc['path'], b['NR'], b['KJ']
        path = save_table(C, NR, KJ, TAG + '%d' % r)
        te = judge(D, PB, load_net(model_path), C, NR, KJ)
        rec.update(pick=pickc['lr'], model=model_path, eps=b['eps'], val=b['val_acc'], avg=b['avg'], layers=b['layers'],
                   gain=gain, test=te)
        log['rounds'].append(rec)
        jdump(log, os.path.join(OUT, TAG + '.json'))
        print('[L%d] 고른 학습률 %g: eps %g, 검증 %.2f%%, 평균 %.2f%% %s (이번 바퀴 %.2f%%p 줄임) | [시험] %s' % (
            r, pickc['lr'], b['eps'], b['val_acc'], b['avg'], b['layers'], gain, fmt(te)), flush=True)
        cur_avg = b['avg']
        if gain < MIN_GAIN:
            log['stop'] = '한 바퀴에 %.2f%%p만 줄었다 (%g%%p 미만)' % (gain, MIN_GAIN)
            print('[L] 멈춤: %s' % log['stop'], flush=True)
            break
    log['final'] = log['rounds'][-1] if 'test' in log['rounds'][-1] else log['rounds'][-2]
    jdump(log, os.path.join(OUT, TAG + '.json'))
    f = log['final']
    print('[L 최종: 마지막 바퀴] 평균 %.2f%% %s | [시험] %s' % (f['avg'], f['layers'], fmt(f['test'])), flush=True)
    open(os.path.join(OUT, TAG + '.done'), 'w').close()


if __name__ == '__main__':
    D = load_data()
    {'check': check, 'run': run}[sys.argv[1]](D)
