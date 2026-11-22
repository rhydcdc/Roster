"""BATCHING - 열-스트립 Roster-Tile (Strip-Mining, W=512):
4,096개 뉴런(열)을 중심점 표(256 x 4,096) 기준으로 오프라인에서 한 번만 정렬(가중치 행렬 물리 재배치와 동치)한 뒤,
L2 캐시에 통째로 들어가는 폭 W(기본 512열, 8개 서로소 스트립)로 나눠 각 스트립 안에서 Roster-Tile(Joint129)을 돌린다.
각 스트립은 L2 안에서 끝나므로 DRAM 중복 읽기가 물리적으로 1.0배(활성 열당 1번)로 고정된다.
배치 B = 256, 1024, 2048에 대해 4개 층 전체의 정확한 연산량, DRAM 읽는 양, Roofline 이론 속도를 확정한다.
python -X utf8 -W ignore BATCHING/CODE/batch_strip.py
"""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

import numpy as np

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

from batch_s import load_keys, queues, RUNS_S, NN  # noqa: E402
from batch_neon import build_encoded_tables  # noqa: E402
from large_solver import get_large_engine, _row_order_hierarchical_olo  # noqa: E402


def compute_offline_col_orders(enc_all):
    """중심점 256개 표(256 x 4096)만으로 4,096열을 오프라인 1회 정렬 (추론 시 비용 0)."""
    eng = get_large_engine()
    col_orders = []
    for l in range(4):
        g = enc_all[l]  # (256, 4096) int8
        W_row = eng.row_sim_matrix(g)
        rp = _row_order_hierarchical_olo(W_row)
        cp = eng.col_bitwise_tsp(g, rp, seed=42 + l)
        # 1라운드 교대 정제
        W_h = eng.row_horiz_sim_matrix(g, cp)
        rp2 = _row_order_hierarchical_olo(W_h)
        cp2 = eng.col_bitwise_tsp(g, rp2, seed=142 + l)
        col_orders.append(np.array(cp2, dtype=np.int32))
    return col_orders


def _worker_strip(args):
    """단일 (층, 스트립)에 대해 Roster-Tile(Joint129) 실행."""
    layer_idx, strip_idx, sub_enc, seed = args
    for p in (
        os.path.join(NEON_DIR, 'problem2_256x4096'),
        os.path.join(NEON_DIR, 'scale_256x4096'),
        os.path.join(NEON_DIR, 'src'),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    from p2_solver import get_p2_engine

    t0 = time.perf_counter()
    mand = (sub_enc == 1)
    opt = (sub_enc == 2)
    if not mand.any():
        return (layer_idx, strip_idx, 0, 0, 0, 0, 0, 0, 0, int(opt.sum()), time.perf_counter() - t0)

    eng = get_p2_engine()
    tiles = eng.solve_joint_coopt(sub_enc, dense_phase_thresh=129, seed=seed)

    covered = np.zeros(sub_enc.shape, dtype=bool)
    used_cols = set()
    tile_reads = 0
    num_A = num_B = 0
    for tl in tiles:
        covered[np.ix_(tl.rows, tl.cols)] = True
        used_cols.update(tl.cols)
        tile_reads += len(tl.cols)
        if tl.frame_type == 'A':
            num_A += 1
        else:
            num_B += 1

    assert covered[mand].all(), f'L{layer_idx+1} strip {strip_idx}: missing mandatory 1!'
    opt_on = int((opt & covered).sum())
    opt_off = int(opt.sum()) - opt_on
    return (
        layer_idx, strip_idx,
        len(tiles), num_A, num_B,
        len(used_cols), tile_reads,
        int(mand.sum()), opt_on, opt_off,
        time.perf_counter() - t0,
    )


def run_batch_size(B, W, keys, enc_all, col_orders, order_idx, ex):
    t0 = time.perf_counter()
    jobs = []
    n_strips = (NN + W - 1) // W
    for l in range(4):
        kq = keys[l][order_idx]
        # 오프라인 정렬된 열 순서로 표 구성
        grid_sorted = enc_all[l][kq][:, col_orders[l]]
        for s in range(n_strips):
            c0 = s * W
            c1 = min(c0 + W, NN)
            sub_enc = np.ascontiguousarray(grid_sorted[:, c0:c1], dtype=np.int8)
            jobs.append((l, s, sub_enc, 42 + 100 * l + s))

    res = list(ex.map(_worker_strip, jobs))
    dt = time.perf_counter() - t0

    layers = []
    tot_tiles = tot_A = tot_B = tot_dram_reads = tot_tile_reads = 0
    tot_mand = tot_opt_on = tot_opt_off = 0
    for l in range(4):
        lres = [r for r in res if r[0] == l]
        lt = sum(r[2] for r in lres)
        la = sum(r[3] for r in lres)
        lb = sum(r[4] for r in lres)
        ldram = sum(r[5] for r in lres)
        ltreads = sum(r[6] for r in lres)
        lmand = sum(r[7] for r in lres)
        lon = sum(r[8] for r in lres)
        loff = sum(r[9] for r in lres)

        full_c = B * NN
        comp = lt * 128
        layers.append(dict(
            layer=l + 1,
            tiles=lt, tiles_A=la, tiles_B=lb,
            compute=comp, compute_pct=100.0 * comp / full_c,
            dram_reads=ldram, dram_reads_pct_serving=100.0 * ldram / NN,
            tile_reads=ltreads,
            fill_rate=100.0 * (lmand + lon) / comp if comp > 0 else 0.0,
            opt_off_rate=100.0 * loff / max(1, lon + loff),
        ))
        tot_tiles += lt; tot_A += la; tot_B += lb
        tot_dram_reads += ldram; tot_tile_reads += ltreads
        tot_mand += lmand; tot_opt_on += lon; tot_opt_off += loff

    full_c_all = 4 * B * NN
    full_r_serving = 4 * NN  # 실제 서빙 풀 모델 (뉴런당 1번 읽기 = 16,384열)
    comp_all = tot_tiles * 128
    comp_ratio = comp_all / full_c_all
    read_ratio = tot_dram_reads / full_r_serving

    def roofline(ridge):
        t_full = max(float(B), float(ridge))
        t_ours = max(comp_ratio * B, read_ratio * ridge)
        return t_full / t_ours

    out = dict(
        B=B, W=W, n_strips=n_strips,
        total_tiles=tot_tiles, total_A=tot_A, total_B=tot_B,
        compute=comp_all, compute_pct=100.0 * comp_ratio,
        dram_reads=tot_dram_reads, dram_reads_pct_serving=100.0 * read_ratio,
        dram_reads_pct_full16=100.0 * tot_dram_reads / (4 * ((B + 15) // 16) * NN),
        fill_rate=100.0 * (tot_mand + tot_opt_on) / comp_all,
        opt_off_rate=100.0 * tot_opt_off / max(1, tot_opt_on + tot_opt_off),
        speedup_rtx4060=roofline(104.0),
        speedup_a100=roofline(156.0),
        speedup_h100=roofline(295.0),
        runtime_sec=dt,
        layers=layers,
    )
    print(
        f'[B={B:4d}, W={W}] 틀 {tot_tiles:5d} (A {tot_A:5d}, B {tot_B:4d}), 채움 {out["fill_rate"]:5.1f}%, 1*끔 {out["opt_off_rate"]:4.1f}% | '
        f'연산 {out["compute_pct"]:5.2f}% / DRAM읽기(실제풀대비) {out["dram_reads_pct_serving"]:5.2f}% | '
        f'속도(이론상한) RTX4060 {out["speedup_rtx4060"]:.2f}x, A100 {out["speedup_a100"]:.2f}x, H100 {out["speedup_h100"]:.2f}x | {dt:.1f}s',
        flush=True,
    )
    for lr in layers:
        print(
            f'   층 {lr["layer"]}: 틀 {lr["tiles"]:5d}, 연산 {lr["compute_pct"]:5.2f}%, DRAM읽기 {lr["dram_reads"]:4d}/4096 ({lr["dram_reads_pct_serving"]:5.1f}%), 채움 {lr["fill_rate"]:5.1f}%',
            flush=True,
        )
    return out


def main():
    T0 = time.perf_counter()
    keys, BM, _ = load_keys()
    enc_all = build_encoded_tables(BM)

    t_ord = time.perf_counter()
    col_orders = compute_offline_col_orders(enc_all)
    print(f'[오프라인 열 정렬 완료] 4개 층 {time.perf_counter() - t_ord:.1f}s', flush=True)

    # B=256은 기존 q0와 동일한 첫 대기열, B=1024/2048은 씨앗 0으로 무작위 비복원 추출
    q0_256 = next(o for N, t, o in queues(keys.shape[1]) if N == 256 and t == 0)
    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)

    n_workers = min(12, os.cpu_count() or 4)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for B, order_idx in [(256, q0_256), (1024, q0_1024), (2048, q0_2048)]:
            results.append(run_batch_size(B, 512, keys, enc_all, col_orders, order_idx, ex))

    out_path = os.path.join(RUNS_S, 'batch_strip_W512.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dict(W=512, total_sec=time.perf_counter() - T0, results=results), f, ensure_ascii=False, indent=2)
    print(f'\n저장: {out_path} (총 {time.perf_counter() - T0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
