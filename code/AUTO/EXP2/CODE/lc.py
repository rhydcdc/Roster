"""Lb를 더 조이기 (사용자와 정함, 2026-11-21): Lb4(평균 15.89%, 시험 90.63%)에서 5바퀴째부터 이어 간다.
Lb에서 바꾼 것: 학습률. Lb의 5바퀴째는 3e-5에서 학습 뒤 검증이 문턱 아래(90.84%)였고 기울기가 튀었다(바퀴 평균 최대 34).
  5바퀴째에 3e-6, 1e-5를 평원까지 돌리고(3e-5는 Lb5 결과를 그대로 쓴다), 다시 줄였을 때 평균이 가장 적은 것을 고른다(끝이면 한 칸 넓힌다).
  6바퀴째부터는 고른 학습률.
검증 문턱은 두 갈래로 나란히: Lc = Lb와 같은 문턱(F를 줄이지 않은 검증 90.94%), Ld = 0.1%p 엄하게(91.04%).
  Lb에서 바퀴마다 검증은 문턱을 지켰지만 시험은 90.92 -> 90.63%로 내려갔다(검증으로 거듭 고르며 검증이 부풂). 엄한 문턱이 이것을 막는지 본다.
  5바퀴째 학습은 두 갈래가 같아서 함께 쓴다(L5s_*). 6바퀴째부터 갈라진다.
압력과 표, 줄이기는 Lb와 같다. 멈춤: 한 바퀴에 1%p 넘게 줄지 않거나 10바퀴째. 최종은 갈래마다 마지막 바퀴(미리 정함). 시험은 바퀴마다 적되 고르는 데 쓰지 않는다.
python lc.py
"""
from lchain import *

START_MODEL = os.path.join(RUNS_E2, 'Lb4_lr3e-05', 'best.pt')
START_TABLE = os.path.join(RULES, 'lv_Lb4.pt')
LB5 = 'Lb5_lr3e-05'                       # Lb의 5바퀴째(3e-5) 학습: 다시 돌리지 않고 쓴다
LRS5 = [3e-6, 1e-5, 3e-5]
LAST_ROUND = 10


def name_of(arm, r, lr):
    if r == 5:
        return LB5 if lr == 3e-5 else 'L5s_lr%g' % lr
    return '%s%d_lr%g' % (arm, r, lr)


def train_or_load(D, path, init, lr, name):
    fj = os.path.join(RUNS_E2, name, 'final.json')
    if os.path.exists(fj):
        return os.path.join(RUNS_E2, name, 'best.pt'), jload(fj)
    return train_round(D, path, init, lr, name)


def chain(D, PB, arm, thr):
    t = torch.load(START_TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    M = NR[0].shape[0]
    model_path, path = START_MODEL, START_TABLE
    _, v0, a0, pl0 = val_of(D, load_net(model_path), C, NR, KJ)
    log = dict(arm=arm, thr=thr, start=dict(model=model_path, val=v0, avg=a0, layers=pl0), rounds=[])
    print('[%s] 검증 문턱 %.2f%%, 출발(Lb4) 검증 %.2f%%, 평균 %.2f%%' % (arm, thr, v0, a0), flush=True)
    cur_avg, lr = a0, None
    for r in range(5, LAST_ROUND + 1):
        cands, tried = [], set()
        lrs = LRS5 if lr is None else [lr]
        while True:
            for x in lrs:
                if x in tried:
                    continue
                tried.add(x)
                name = name_of(arm, r, x)
                bp, res = train_or_load(D, path, model_path, x, name)
                m = load_net(bp)
                _, v_after, a_after, _ = val_of(D, m, C, NR, KJ)
                b2, rows2 = reduce(D, m, C, M, NR, KJ, thr)
                cands.append(dict(lr=x, name=name, path=bp, epochs=res['epochs'], val_trained=v_after, avg_trained=a_after,
                                  reduce_rows=rows2, best=b2))
                print('[%s%d 학습률 %g] 학습 뒤 검증 %.2f%% (평균 %.2f%%) -> 다시 줄이면 %s' % (
                    arm, r, x, v_after, a_after,
                    ('eps %g, 검증 %.2f%%, 평균 %.2f%% %s' % (b2['eps'], b2['val_acc'], b2['avg'], b2['layers'])) if b2 else '문턱을 지키는 eps 없음'),
                    flush=True)
                del m
                torch.cuda.empty_cache()
            ok = [c for c in cands if c['best'] is not None]
            if not ok or lr is not None:
                break
            pickc = min(ok, key=lambda c: (c['best']['avg'], -c['best']['val_acc']))
            xs = sorted(c['lr'] for c in cands)
            if pickc['lr'] == xs[0] and len(tried) < 5:
                lrs = [xs[0] / 3.16]
            elif pickc['lr'] == xs[-1] and len(tried) < 5:
                lrs = [xs[-1] * 3.16]
            else:
                break
            print('[%s%d] 고른 학습률 %g이 후보의 끝이라 %g를 더 돌린다' % (arm, r, pickc['lr'], lrs[0]), flush=True)
        ok = [c for c in cands if c['best'] is not None]
        rec = dict(round=r, cands=[{k: v for k, v in c.items() if k != 'best'} | (
            dict(best={k: v for k, v in c['best'].items() if k not in ('NR', 'KJ')}) if c['best'] else {}) for c in cands])
        if not ok:
            rec['stop'] = '문턱을 지키며 줄일 수 있는 후보가 없다'
            log['rounds'].append(rec)
            print('[%s%d] 멈춤: %s' % (arm, r, rec['stop']), flush=True)
            break
        pickc = min(ok, key=lambda c: (c['best']['avg'], -c['best']['val_acc']))
        if lr is None:
            lr = pickc['lr']
            log['lr'] = lr
        b = pickc['best']
        gain = cur_avg - b['avg']
        model_path, NR, KJ = pickc['path'], b['NR'], b['KJ']
        path = save_table(C, NR, KJ, '%s%d' % (arm, r))
        te = judge(D, PB, load_net(model_path), C, NR, KJ)
        rec.update(pick=pickc['lr'], model=model_path, eps=b['eps'], val=b['val_acc'], avg=b['avg'], layers=b['layers'],
                   gain=gain, test=te)
        log['rounds'].append(rec)
        jdump(log, os.path.join(OUT, '%s.json' % arm))
        print('[%s%d] 고른 학습률 %g: eps %g, 검증 %.2f%%, 평균 %.2f%% %s (이번 바퀴 %.2f%%p 줄임) | [시험] %s' % (
            arm, r, pickc['lr'], b['eps'], b['val_acc'], b['avg'], b['layers'], gain, fmt(te)), flush=True)
        cur_avg = b['avg']
        if gain < MIN_GAIN:
            log['stop'] = '한 바퀴에 %.2f%%p만 줄었다' % gain
            print('[%s] 멈춤: %s' % (arm, log['stop']), flush=True)
            break
    done = [x for x in log['rounds'] if 'test' in x]
    log['final'] = done[-1] if done else None
    jdump(log, os.path.join(OUT, '%s.json' % arm))
    if done:
        f = done[-1]
        print('[%s 최종: 마지막 바퀴 %d] 평균 %.2f%% %s | [시험] %s' % (arm, f['round'], f['avg'], f['layers'], fmt(f['test'])), flush=True)
    else:
        print('[%s 최종] 한 바퀴도 더 줄이지 못했다 (Lb4 그대로)' % arm, flush=True)


def main(D):
    PB = base_preds(D)
    F_ = load_net(F_PATH)
    C, M, RK, KJF = start(D)
    _, thrF, _, _ = val_of(D, F_, C, RK, KJF)
    chain(D, PB, 'Lc', thrF)
    chain(D, PB, 'Ld', thrF + 0.1)
    open(os.path.join(OUT, 'Lcd.done'), 'w').close()


if __name__ == '__main__':
    main(load_data())
