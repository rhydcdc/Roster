"""BATCHING - Q: 목록 끝쪽을 빼면 정확도가 얼마나 바뀌나 (학습 없이, 짧게 확인).
Lc6의 중심점마다의 목록에서 뒤쪽 f를 뺀다: K' = round(K x (1 - f)), 모든 층 같은 f.
잴 것: 검증 정확도(5,000장), 빼지 않은 Lc6과 같은 검증 사진에서 짝 비교, 켜는 뉴런 평균(층마다).
python tail_probe.py
"""
import json
import os
import sys
import time

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
from batch_t import *                                                   # noqa: E402,F401 (ex2lib: run_masked, paired, keys_layer)
HERE_Q = _H                                                            # 가져온 뒤에 다시 정한다 (HANDOFF 함정 5)
RUNS_Q = os.path.join(HERE_Q, 'runs')
FR = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]


def main():
    T0 = time.perf_counter()
    D = load_data()
    model = load_net(MODEL)
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    prep, keyfn = keys_layer(C)
    yva = D['yva'].cpu()
    yte = D['yte'].cpu()
    pb = base_preds(D)                                                 # 베이스 모델의 예측 (시험 짝 비교)
    p0, out, lines = None, [], []
    with torch.no_grad():
        for f in FR:
            K2 = [torch.round(KJ[l].float() * (1 - f)).to(KJ[l].dtype) for l in range(N_BLOCK)]

            def mask(l, h, ctx, K2=K2):
                j = keyfn(l, h, ctx)
                return NR[l][j] < K2[l][j][:, None]
            preds, used, per = run_masked(model, D['xva'], prep, mask)
            preds = preds.cpu()
            if p0 is None:
                p0 = preds
            j = paired(p0, preds, yva)
            pt, used_t, per_t = run_masked(model, D['xte'], prep, mask)
            jt = paired(pb['te'].cpu(), pt.cpu(), yte)                 # 시험: 베이스 모델과 짝 비교 (통과선 90.52%)
            r = dict(cut_pct=100 * f, val_acc=j['acc'], drop_vs_lc6=j['drop'], lo=j['lo'], hi=j['hi'],
                     lc6_only=j['base_only'], cut_only=j['cand_only'], used_pct=used, used_layers=per,
                     test_acc=jt['acc'], test_drop_vs_base=jt['drop'], test_lo=jt['lo'], test_hi=jt['hi'],
                     test_verdict=jt['verdict'], test_used_pct=used_t, test_used_layers=per_t)
            out.append(r)
            s = '[Q] 끝쪽 %4.0f%% 뺌: 검증 정확도 %.2f%% (Lc6 대비 하락 %.2f%%p [%.2f, %.2f], Lc6만 맞힘 %d / 뺀 쪽만 맞힘 %d) | 켜는 뉴런 평균 %.2f%% %s | 시험 정확도 %.2f%% (베이스 대비 하락 %.2f%%p [%.2f, %.2f] %s), 시험에서 켜는 뉴런 평균 %.2f%% %s | %.0f s' % (
                100 * f, j['acc'], j['drop'], j['lo'], j['hi'], j['base_only'], j['cand_only'], used,
                [round(float(p), 2) for p in per], jt['acc'], jt['drop'], jt['lo'], jt['hi'], jt['verdict'], used_t,
                [round(float(p), 2) for p in per_t], time.perf_counter() - T0)
            print(s, flush=True)
            lines.append(s)
    with open(os.path.join(RUNS_Q, 'tail_probe.json'), 'w', encoding='utf-8') as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE_Q, 'tail_probe.out'), 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
