"""BATCHING - NEON: 실제 Lc6 표(요청 256개, 층 1~4)에 문제 1+2 연동 풀기(NEON-Joint) 적용.
기존 batch_s.py와 같은 대기열(씨앗 0), 같은 비용 모형(연산량 = 틀 수 x 128, 읽는 양 = 틀마다 쓰는 열 수의 합)으로
기본 풀 모델(16개씩) 및 고른 방법(뉴런 쪽 블록 + 합치기)과 견준다.
python batch_neon.py [--queues 1]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
NEON_DIR = os.path.join(PROJ, 'ROSTER_TILE')
for p in (
    HERE,
    os.path.join(NEON_DIR, 'problem2_256x4096'),
    os.path.join(NEON_DIR, 'scale_256x4096'),
    os.path.join(NEON_DIR, 'src'),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from batch_s import load_keys, queues, bases, evaluate, RUNS_S, NN  # noqa: E402
from batch_s7 import plan_chosen  # noqa: E402
from objective import PROFILES  # noqa: E402
from large_problem import LargeMatrixInstance  # noqa: E402
from large_solver import solve_large_neon  # noqa: E402
from p2_solver import solve_p2_neon_joint, validate_and_measure_p2  # noqa: E402

TABLE = os.path.join(PROJ, 'AUTO', 'EXP2', 'CODE', 'runs', 'rules', 'lv_Lc6.pt')


def build_encoded_tables(BM):
    """중심점별 (256, 4096) int8 인코딩 표: 0=끔, 1=필수(앞 90%), 2=꺼도 되는 1(끝쪽 10%)."""
    t = torch.load(TABLE, map_location='cpu')
    NR, KJ = t['NR'], t['KJ']
    enc_all = []
    for l in range(len(NR)):
        on = (NR[l] < KJ[l][:, None]).numpy()
        assert np.array_equal(on, BM[l]), f'Layer {l} BM mismatch with lv_Lc6.pt!'
        k_mand = torch.round(KJ[l].float() * 0.9).to(KJ[l].dtype)
        mand = (NR[l] < k_mand[:, None]).numpy()
        opt = on & (~mand)
        enc = np.zeros(on.shape, dtype=np.int8)
        enc[mand] = 1
        enc[opt] = 2
        enc_all.append(enc)
    return enc_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queues', type=int, default=1, help='평가할 대기열 수 (256개 요청, 최대 10)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    T0 = time.perf_counter()
    keys, BM, _ = load_keys()
    enc_all = build_encoded_tables(BM)

    q256 = [(t, order) for N, t, order in queues(keys.shape[1]) if N == 256][: args.queues]
    assert len(q256) == args.queues

    per_queue = []
    tot_full_c = tot_full_r = 0
    tot_seq_c = tot_seq_r = 0
    tot_ch_c = tot_ch_r = 0
    tot_nj_c = tot_nj_r = 0
    layer_agg = [dict(full_c=0, full_r=0, ch_c=0, ch_r=0, nj_c=0, nj_r=0,
                      tiles=0, tiles_A=0, tiles_B=0, ones=0, mand=0, opt_on=0, opt_off=0,
                      zeros=0, unique_cols=0, sec=0.0) for _ in range(4)]

    for qi, (t_idx, order) in enumerate(q256):
        q_layers = []
        for l in range(4):
            kq = keys[l][order]
            Mq = BM[l][kq]
            grid_enc = enc_all[l][kq]

            b = bases(kq, BM[l])
            fc, fr = b['full']
            sc, sr = b['seq']

            # 같은 대기열에서 기존 고른 방법(뉴런 쪽 블록 + 합치기, 16 단위)
            plan_ch, _ = plan_chosen(kq, BM[l], 16, False)
            ech = evaluate(plan_ch, Mq, check=True)
            assert ech['ok']

            # NEON-Joint (문제 1 웜스타트 + 문제 2 연동 풀기)
            inst = LargeMatrixInstance(grid_enc=grid_enc, name=f'q{t_idx}_L{l+1}')
            t1 = time.perf_counter()
            p1_sol = solve_large_neon(inst, PROFILES['balanced'], seed=args.seed)
            p2_sol = solve_p2_neon_joint(inst, p1_sol=p1_sol, seed=args.seed)
            sec = time.perf_counter() - t1

            met = validate_and_measure_p2(inst, p2_sol)
            assert met['valid'] and met['missing_cells'] == 0

            nj_c = int(met['total_tile_cells'])  # 128 * num_tiles
            nj_r = int(sum(len(tl.cols) for tl in p2_sol.tiles))
            u_cols = len({c for tl in p2_sol.tiles for c in tl.cols})

            # 타일 간 셀 겹침 진단 (이미 찍은 칸을 다른 타일이 0-마스킹으로 지나가는 경우)
            cell_hits = np.zeros(grid_enc.shape, dtype=np.int16)
            for tl in p2_sol.tiles:
                cell_hits[np.ix_(tl.rows, tl.cols)] += 1
            mand_mask = (grid_enc == 1)
            assert (cell_hits[mand_mask] >= 1).all()
            mand_multi_hit = int((cell_hits[mand_mask] > 1).sum())

            lr = dict(
                layer=l + 1,
                unique_clusters=int(len(np.unique(kq))),
                avg_ones_per_row=float(Mq.sum() / len(kq)),
                full_compute=fc, full_reads=fr,
                seq_compute=sc, seq_reads=sr,
                chosen_compute=ech['compute'], chosen_reads=ech['reads'],
                neon_compute=nj_c, neon_reads=nj_r,
                neon_pct_full=(100.0 * nj_c / fc, 100.0 * nj_r / fr),
                chosen_pct_full=(100.0 * ech['compute'] / fc, 100.0 * ech['reads'] / fr),
                num_tiles=int(met['num_tiles']),
                num_tiles_A=int(met['num_tiles_A']),
                num_tiles_B=int(met['num_tiles_B']),
                ones_fill_rate=float(met['ones_fill_rate']),
                total_zeros=int(met['total_zeros_in_tiles']),
                total_norm_ones=int(met['total_norm_ones']),
                opt_kept_on=int(met['opt_kept_on']),
                opt_turned_off=int(met['opt_turned_off']),
                opt_off_rate=float(met['opt_off_rate']),
                unique_cols_read=u_cols,
                avg_reread_per_active_col=float(nj_r / u_cols) if u_cols > 0 else 0.0,
                mand_multi_hit=mand_multi_hit,
                runtime_sec=float(sec),
            )
            q_layers.append(lr)

            tot_full_c += fc; tot_full_r += fr
            tot_seq_c += sc; tot_seq_r += sr
            tot_ch_c += ech['compute']; tot_ch_r += ech['reads']
            tot_nj_c += nj_c; tot_nj_r += nj_r

            ag = layer_agg[l]
            ag['full_c'] += fc; ag['full_r'] += fr
            ag['ch_c'] += ech['compute']; ag['ch_r'] += ech['reads']
            ag['nj_c'] += nj_c; ag['nj_r'] += nj_r
            ag['tiles'] += lr['num_tiles']
            ag['tiles_A'] += lr['num_tiles_A']
            ag['tiles_B'] += lr['num_tiles_B']
            ag['ones'] += int(met['stamped_active_ones'])
            ag['mand'] += lr['total_norm_ones']
            ag['opt_on'] += lr['opt_kept_on']
            ag['opt_off'] += lr['opt_turned_off']
            ag['zeros'] += lr['total_zeros']
            ag['unique_cols'] += u_cols
            ag['sec'] += sec

            print(
                f'[q{t_idx} 층 {l+1}] 중심점 {lr["unique_clusters"]:3d}종, 행당1 {lr["avg_ones_per_row"]:5.1f} | '
                f'NEON-Joint 틀 {lr["num_tiles"]:4d} (A {lr["num_tiles_A"]:4d}, B {lr["num_tiles_B"]:3d}), '
                f'채움 {100*lr["ones_fill_rate"]:5.1f}%, 1*끔 {100*lr["opt_off_rate"]:4.1f}% | '
                f'기본 풀 모델 대비 연산 {lr["neon_pct_full"][0]:5.1f}% / 읽기 {lr["neon_pct_full"][1]:5.1f}% '
                f'(고른 방법 {lr["chosen_pct_full"][0]:5.1f}% / {lr["chosen_pct_full"][1]:4.1f}%) | '
                f'열당 재읽기 {lr["avg_reread_per_active_col"]:4.1f}x | {sec:.1f}s',
                flush=True,
            )
        per_queue.append(dict(queue_idx=t_idx, layers=q_layers))

    Q = len(q256)
    print(f'\n=== 요청 256개 ({Q}개 대기열 평균, 층 4개 합, 총 {time.perf_counter() - T0:.1f}s) ===', flush=True)
    print(f'기본 풀 모델 (16개씩) : 연산 {tot_full_c/Q:10,.0f}칸 (100.0%), 읽기 {tot_full_r/Q:8,.0f}열 (100.0%)', flush=True)
    print(f'줄인 모델   (16개씩) : 연산 {tot_seq_c/Q:10,.0f}칸 ({100*tot_seq_c/tot_full_c:5.1f}%), 읽기 {tot_seq_r/Q:8,.0f}열 ({100*tot_seq_r/tot_full_r:5.1f}%)', flush=True)
    print(f'고른 방법(뉴런블록+합) : 연산 {tot_ch_c/Q:10,.0f}칸 ({100*tot_ch_c/tot_full_c:5.1f}%), 읽기 {tot_ch_r/Q:8,.0f}열 ({100*tot_ch_r/tot_full_r:5.1f}%)', flush=True)
    print(f'NEON-Joint (연동풀기) : 연산 {tot_nj_c/Q:10,.0f}칸 ({100*tot_nj_c/tot_full_c:5.1f}%), 읽기 {tot_nj_r/Q:8,.0f}열 ({100*tot_nj_r/tot_full_r:5.1f}%)', flush=True)

    summary = dict(
        n_queues=Q,
        full=dict(compute=tot_full_c / Q, reads=tot_full_r / Q),
        seq=dict(compute=tot_seq_c / Q, reads=tot_seq_r / Q,
                 pct_full=(100.0 * tot_seq_c / tot_full_c, 100.0 * tot_seq_r / tot_full_r)),
        chosen=dict(compute=tot_ch_c / Q, reads=tot_ch_r / Q,
                    pct_full=(100.0 * tot_ch_c / tot_full_c, 100.0 * tot_ch_r / tot_full_r)),
        neon_joint=dict(compute=tot_nj_c / Q, reads=tot_nj_r / Q,
                        pct_full=(100.0 * tot_nj_c / tot_full_c, 100.0 * tot_nj_r / tot_full_r)),
        by_layer=[dict(
            layer=l + 1,
            pct_full=(100.0 * ag['nj_c'] / ag['full_c'], 100.0 * ag['nj_r'] / ag['full_r']),
            chosen_pct_full=(100.0 * ag['ch_c'] / ag['full_c'], 100.0 * ag['ch_r'] / ag['full_r']),
            tiles=ag['tiles'] / Q, tiles_A=ag['tiles_A'] / Q, tiles_B=ag['tiles_B'] / Q,
            ones_fill_rate=ag['ones'] / ag['nj_c'],
            opt_off_rate=ag['opt_off'] / (ag['opt_on'] + ag['opt_off']),
            avg_reread=ag['nj_r'] / ag['unique_cols'],
            sec=ag['sec'] / Q,
        ) for l, ag in enumerate(layer_agg)],
        queues=per_queue,
    )
    out_path = os.path.join(RUNS_S, f'batch_neon_q{Q}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'저장: {out_path}', flush=True)


if __name__ == '__main__':
    main()
