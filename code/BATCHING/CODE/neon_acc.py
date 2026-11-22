"""BATCHING - NEON-Joint 실제 선택적 끄기(1*) 마스크를 Lc6 모델에 그대로 적용한
시험 사진 10,000장 전체 정확도(베이스 모델과 짝 비교) + 39개 256-배치 평균 연산량/읽는 양 측정.
층마다 (1) 현재 h로 중심점 결정 -> (2) 256개씩 묶어 NEON-Joint(Joint129)로 타일 추출 ->
(3) 타일 안에 든 필수 1 + 무임승차 1*만 켠 실제 마스크로 FFN 계산 -> 다음 층으로 연쇄 전달.
python -X utf8 -W ignore BATCHING/CODE/neon_acc.py
"""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
NEON_DIR = os.path.join(PROJ, 'ROSTER_TILE')


def _worker_solve(args):
    """워커 프로세스: 단일 배치(B x 4096)에 NEON-Joint(thr=129) 실행 후 실제 활성 마스크와 통계 반환."""
    b_idx, grid_enc, seed = args
    for p in (
        os.path.join(NEON_DIR, 'problem2_256x4096'),
        os.path.join(NEON_DIR, 'scale_256x4096'),
        os.path.join(NEON_DIR, 'src'),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    from p2_solver import get_p2_engine

    eng = get_p2_engine()
    tiles = eng.solve_joint_coopt(grid_enc, dense_phase_thresh=129, seed=seed)

    covered = np.zeros(grid_enc.shape, dtype=bool)
    reads = 0
    num_A = num_B = 0
    for tl in tiles:
        covered[np.ix_(tl.rows, tl.cols)] = True
        reads += len(tl.cols)
        if tl.frame_type == 'A':
            num_A += 1
        else:
            num_B += 1

    mand = (grid_enc == 1)
    opt = (grid_enc == 2)
    assert covered[mand].all(), f'Batch {b_idx}: missing mandatory 1!'
    opt_on_mask = opt & covered
    active = mand | opt_on_mask

    return (
        b_idx,
        active,
        len(tiles),
        num_A,
        num_B,
        reads,
        int(mand.sum()),
        int(opt_on_mask.sum()),
        int(opt.sum() - opt_on_mask.sum()),
    )


def main():
    sys.path.insert(0, HERE)
    from batch_t import (  # noqa: E402
        load_data, load_net, MODEL, TABLE, DEV, N_BLOCK, N_NEURON,
        base_preds, paired, keys_layer, run_masked,
    )
    from batch_s import RUNS_S  # noqa: E402

    T0 = time.perf_counter()
    D = load_data()
    model = load_net(MODEL)
    model.eval()
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    prep, keyfn = keys_layer(C)
    yte = D['yte'].cpu()
    pb = base_preds(D)['te'].cpu()

    # 0. Sanity check: Lc6 원본(0% 끔) 재현 확인 (90.68%가 나와야 함)
    p_orig, u_orig, per_orig = run_masked(
        model, D['xte'], prep, lambda l, h, ctx: NR[l][keyfn(l, h, ctx)] < KJ[l][keyfn(l, h, ctx)][:, None]
    )
    j_orig = paired(pb, p_orig.cpu(), yte)
    print(f'[확인] Lc6 원본(0% 끔): 시험 {j_orig["acc"]:.2f}%, 켜는 뉴런 {u_orig:.2f}% {per_orig}', flush=True)
    assert abs(j_orig['acc'] - 90.68) < 0.02, f'Lc6 sanity mismatch: {j_orig["acc"]}'

    # 중심점별 int8 인코딩 표 (CPU)
    enc_all = []
    for l in range(N_BLOCK):
        on = (NR[l] < KJ[l][:, None]).cpu().numpy()
        k_mand = torch.round(KJ[l].float() * 0.9).to(KJ[l].dtype)
        mand = (NR[l] < k_mand[:, None]).cpu().numpy()
        opt = on & (~mand)
        enc = np.zeros(on.shape, dtype=np.int8)
        enc[mand] = 1
        enc[opt] = 2
        enc_all.append(enc)

    # 시험 사진 10,000장을 현실적인 256-배치 39개 + 16-배치 1개로 무작위 분할 (씨앗 0)
    rng = np.random.default_rng(0)
    N_TEST = D['xte'].shape[0]
    perm = rng.permutation(N_TEST)
    x_perm = D['xte'][torch.from_numpy(perm).to(DEV)]
    slices = [(s, min(s + 256, N_TEST)) for s in range(0, N_TEST, 256)]
    n_workers = min(12, os.cpu_count() or 4)
    print(f'[시작] 시험 {N_TEST}장 = 256-배치 39개 + 16-배치 1개, 워커 {n_workers}개 병렬 실행', flush=True)

    with torch.no_grad():
        # 입력 임베딩
        h_chunks = []
        for s in range(0, N_TEST, 2500):
            xb = x_perm[s:s + 2500]
            h_chunks.append(model.inp(xb.reshape(xb.shape[0], -1)))
        h = torch.cat(h_chunks, dim=0)

        used_per_layer = []
        layer_stats = []
        full256_tiles_sum = 0
        full256_reads_sum = 0

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for l, blk in enumerate(model.blocks):
                tl0 = time.perf_counter()
                # 1) 현재 h로 10,000장 전체의 층 l 중심점 결정
                j_chunks = []
                for s in range(0, N_TEST, 2500):
                    j_chunks.append(keyfn(l, h[s:s + 2500], None).cpu().numpy())
                j_all = np.concatenate(j_chunks, axis=0)

                # 2) 40개 배치를 병렬로 NEON-Joint(thr=129) 풀이
                jobs = []
                for b_idx, (s, e) in enumerate(slices):
                    grid_enc = enc_all[l][j_all[s:e]]
                    jobs.append((b_idx, grid_enc, 42 + 1000 * l + b_idx))

                results = list(ex.map(_worker_solve, jobs))
                results.sort(key=lambda r: r[0])

                active_all = np.concatenate([r[1] for r in results], axis=0)
                used_pct = 100.0 * active_all.sum() / (N_TEST * N_NEURON)
                used_per_layer.append(round(float(used_pct), 2))

                # 39개 완전 256-배치 통계 집계
                full_res = [r for r, (s, e) in zip(results, slices) if e - s == 256]
                avg_tiles = float(np.mean([r[2] for r in full_res]))
                avg_A = float(np.mean([r[3] for r in full_res]))
                avg_B = float(np.mean([r[4] for r in full_res]))
                avg_reads = float(np.mean([r[5] for r in full_res]))
                tot_mand = sum(r[6] for r in results)
                tot_opt_on = sum(r[7] for r in results)
                tot_opt_off = sum(r[8] for r in results)
                opt_off_rate = tot_opt_off / max(1, tot_opt_on + tot_opt_off)

                comp_pct = 100.0 * (avg_tiles * 128.0) / (256 * N_NEURON)
                read_pct = 100.0 * avg_reads / (16 * N_NEURON)
                full256_tiles_sum += avg_tiles
                full256_reads_sum += avg_reads

                # 3) 실제 선택된 마스크 그대로 GPU에서 FFN 계산 후 다음 층으로 전달
                h_next_chunks = []
                for s in range(0, N_TEST, 2500):
                    hc = h[s:s + 2500]
                    mc = torch.from_numpy(active_all[s:s + 2500]).to(DEV)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        a = blk.neurons(hc)
                        h_next_chunks.append(hc + blk.down(a * mc.to(a.dtype)).float())
                h = torch.cat(h_next_chunks, dim=0)

                dt = time.perf_counter() - tl0
                ls = dict(
                    layer=l + 1,
                    used_pct=round(float(used_pct), 2),
                    avg_tiles_256=avg_tiles,
                    avg_A_256=avg_A,
                    avg_B_256=avg_B,
                    avg_reads_256=avg_reads,
                    compute_pct_full=comp_pct,
                    reads_pct_full=read_pct,
                    opt_off_rate=opt_off_rate,
                    total_mand=tot_mand,
                    total_opt_on=tot_opt_on,
                    total_opt_off=tot_opt_off,
                    sec=dt,
                )
                layer_stats.append(ls)
                print(
                    f'[층 {l+1}] 켠 뉴런 {used_pct:5.2f}%, 1*끔 {100*opt_off_rate:4.1f}% | '
                    f'256-배치(39개 평균) 틀 {avg_tiles:6.1f} (연산 {comp_pct:5.1f}% / 읽기 {read_pct:5.1f}%) | {dt:.1f}s',
                    flush=True,
                )

        # 최종 예측 -> 원래 시험 인덱스 순서로 복원해 베이스 모델과 짝 비교
        pred_chunks = []
        for s in range(0, N_TEST, 2500):
            pred_chunks.append(model.head(model.norm(h[s:s + 2500])).argmax(1).cpu())
        preds_perm = torch.cat(pred_chunks, dim=0)
        preds_orig = torch.empty_like(preds_perm)
        preds_orig[torch.from_numpy(perm)] = preds_perm

    jt = paired(pb, preds_orig, yte)
    j_vs_lc6 = paired(p_orig.cpu(), preds_orig, yte)
    used_avg = sum(used_per_layer) / N_BLOCK
    tot_comp = full256_tiles_sum * 128.0
    tot_comp_pct = 100.0 * tot_comp / (4 * 256 * N_NEURON)
    tot_read_pct = 100.0 * full256_reads_sum / (4 * 16 * N_NEURON)

    print(f'\n=== NEON-Joint 실제 선택적 끄기 시험 결과 (시험 {N_TEST}장, 총 {time.perf_counter() - T0:.1f}s) ===', flush=True)
    print(f'시험 정확도: {jt["acc"]:.2f}% (베이스 91.02% 대비 하락 {jt["drop"]:.2f}%p [{jt["lo"]:.2f}, {jt["hi"]:.2f}] -> {jt["verdict"]})', flush=True)
    print(f'Lc6 원본 대비: 하락 {j_vs_lc6["drop"]:.2f}%p [{j_vs_lc6["lo"]:.2f}, {j_vs_lc6["hi"]:.2f}] (Lc6만 맞힘 {j_vs_lc6["base_only"]}, NEON만 맞힘 {j_vs_lc6["cand_only"]})', flush=True)
    print(f'켜는 뉴런 평균: {used_avg:.2f}% {used_per_layer}', flush=True)
    print(f'39개 256-배치 평균 (4개 층 합): 연산 {tot_comp:,.0f}칸 ({tot_comp_pct:.1f}%), 읽기 {full256_reads_sum:,.0f}열 ({tot_read_pct:.1f}%)', flush=True)

    out = dict(
        method='NEON-Joint(thr=129)',
        n_test=N_TEST,
        n_full_256_batches=39,
        test_acc=jt['acc'],
        test_drop_vs_base=jt['drop'],
        test_lo=jt['lo'],
        test_hi=jt['hi'],
        test_verdict=jt['verdict'],
        base_only=jt['base_only'],
        cand_only=jt['cand_only'],
        vs_lc6=j_vs_lc6,
        used_avg_pct=used_avg,
        used_per_layer=used_per_layer,
        avg256_compute=tot_comp,
        avg256_compute_pct_full=tot_comp_pct,
        avg256_reads=full256_reads_sum,
        avg256_reads_pct_full=tot_read_pct,
        layers=layer_stats,
        total_sec=time.perf_counter() - T0,
    )
    out_path = os.path.join(RUNS_S, 'neon_acc.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'저장: {out_path}', flush=True)


if __name__ == '__main__':
    main()
