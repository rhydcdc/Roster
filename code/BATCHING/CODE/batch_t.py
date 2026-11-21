"""BATCHING - T: 요청 쪽(겹침 순서로 16행씩)과 뉴런 쪽(가중치는 한 번만 꺼냄)을 합친다. 같은 대기열(batch_n과 같은 씨앗, 같은 뽑기 순서).
  요청을 층마다, 간 중심점의 나무 순서(목록 겹침으로 만든 B1 나무의 잎 순서)로 늘어놓고 16행씩 자른다 (칸을 꽉 채움).
  b_sim16 : 16행 칸마다 자기 뉴런(합집합)을 따로 꺼내 계산.  연산량 = 16 x c16(칸의 합집합), 읽는 양 = 칸마다 합집합의 합
  tile2d  : 뉴런을 '그 뉴런이 필요한 칸들의 평균 위치' 순서로 늘어놓고 16개씩 블록으로 자른다. 블록마다 가중치를 한 번만 꺼내
            그 블록이 필요한 16행 칸들에 차례로 쓴다 (가중치를 꺼내 둔 채 여러 칸에 쓰는 커널).
            연산량 = 빈칸이 아닌 16 x 16 칸 수 x 256, 읽는 양 = 합집합 (다시 쓰지 않으면 = 칸마다 닿는 블록 수 x 16, 함께 적음)
주 수치: 연산량과 읽는 양 (층 4개 합).
python batch_t.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from batch_n import *                                                  # noqa: E402,F401
from scipy.cluster.hierarchy import leaves_list                        # noqa: E402


def main():
    D = load_data()
    model = load_net(MODEL)
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    keys = [k.cpu().numpy() for k in keys_all(model, D['xte'], C, NR, KJ)]
    BM = [(NR[l] < KJ[l][:, None]).cpu().numpy() for l in range(N_BLOCK)]
    pos = []
    for l in range(N_BLOCK):                                           # 중심점의 나무 순서 (목록 겹침, 미리 한 번)
        o = leaves_list(build_tree(BM[l]))
        p = np.empty(256, dtype=np.int64)
        p[o] = np.arange(256)
        pos.append(p)
    prev = jload(os.path.join(HERE_N, 'runs', 'batch_n.json'))
    rng = np.random.default_rng(0)                                     # batch_n과 같은 대기열
    out = {}
    for N, T in QN.items():
        acc = {k: np.zeros(3) for k in ('b_sim16', 'tile2d')}           # 연산량, 읽는 양, 읽는 양(다시 안 씀)
        for _ in range(T):
            order = rng.choice(D['xte'].shape[0], N, replace=False)
            for l in range(N_BLOCK):
                kq = keys[l][order]
                ks = kq[np.argsort(pos[l][kq], kind='stable')]         # 겹침 순서로 늘어놓기
                tiles = [ks[s:s + 16] for s in range(0, N, 16)]
                need = np.stack([BM[l][np.unique(tt)].any(0) for tt in tiles])     # (칸 수, 4096)
                cnt_rows = np.array([len(tt) for tt in tiles])
                su = need.sum(1).astype(np.float64)
                acc['b_sim16'] += ((np.ceil(cnt_rows / 16) * 16 * np.ceil(su / 16) * 16).sum(), su.sum(), su.sum())
                cols = np.nonzero(need.any(0))[0]
                nd = need[:, cols]
                bary = (nd * np.arange(len(tiles))[:, None]).sum(0) / nd.sum(0)
                nd = nd[:, np.argsort(bary, kind='stable')]
                V = nd.shape[1]
                pad = (-V) % 16
                nd = np.concatenate([nd, np.zeros((nd.shape[0], pad), bool)], 1).reshape(nd.shape[0], -1, 16).any(2)
                acc['tile2d'] += (nd.sum() * 256.0, float(V), nd.sum() * 16.0)
        r = prev[str(N)]['res']
        res = {k: dict(compute=v[0] / T, reads=v[1] / T, reads_noreuse=v[2] / T) for k, v in acc.items()}
        out[N] = res
        print('\n=== 대기열 %d개 ===' % N, flush=True)
        for k in ('seq_g16', 'union', 'reqclust16', 'neuron_all'):
            v = r[k]
            print('%-12s 연산량 %10.0f  읽는 양 %8.0f | 줄인 베이스(16개씩) 대비 연산량 %.2f배, 읽는 양 %.2f배 | 완전 베이스(16개씩) 대비 %.2f배, %.2f배' % (
                k, v['compute'], v['reads'], r['seq_g16']['compute'] / v['compute'], r['seq_g16']['reads'] / v['reads'],
                r['full_g16']['compute'] / v['compute'], r['full_g16']['reads'] / v['reads']), flush=True)
        for k, v in res.items():
            print('%-12s 연산량 %10.0f  읽는 양 %8.0f (다시 안 쓰면 %8.0f) | 줄인 베이스(16개씩) 대비 연산량 %.2f배, 읽는 양 %.2f배 | 완전 베이스(16개씩) 대비 %.2f배, %.2f배' % (
                k, v['compute'], v['reads'], v['reads_noreuse'], r['seq_g16']['compute'] / v['compute'], r['seq_g16']['reads'] / v['reads'],
                r['full_g16']['compute'] / v['compute'], r['full_g16']['reads'] / v['reads']), flush=True)
        print('이상적 연산량 %.0f' % r['ideal_compute'], flush=True)
    jdump(out, os.path.join(HERE_N, 'runs', 'batch_t.json'))


if __name__ == '__main__':
    main()
