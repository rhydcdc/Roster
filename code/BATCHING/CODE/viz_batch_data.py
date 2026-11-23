"""레시피 HTML '묶어서 계산'(단일 요청 B=1 + 5단계 다중 크기 Roster-Tile + RTX 4060 실측)에 넣을 데이터를 모은다.
원자료:
  - runs/speed_b1_final.json      : 단일 요청(B=1) 벡터화 bf16 GEMV 커널 RTX 4060 실측 (d=1024/4096)
  - runs/ms_roster5_stats.json    : 5단계 다중 크기 Roster-Tile (64x32->32x32->32x16->16x16->8x16) 통계 + 10,000장 정확도
  - runs/speed_roster5_final.json : 실제 서버(B=512/1024/2048) RTX 4060 실측 (d=1024/4096, 4버전 교대 CUDA Graph)
  - runs/keys.npz                 : 배치별 원본 표의 1/0 개수 (불필요한 0 제거율 산출)
python -X utf8 -W ignore BATCHING/CODE/viz_batch_data.py -> runs/viz_batch.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batch_s import load_keys, RUNS_S, NN  # noqa: E402

F5_NAMES = ('64x32', '32x32', '32x16', '16x16', '8x16')


def full_ones_per_layer(keys, BM, order_idx):
    out = []
    for l in range(4):
        kq = keys[l][order_idx]
        out.append(int(BM[l][kq].sum()))
    return out


def build_layer_table5(B, rec, fones_arr, label, note):
    rows = []
    tf = tc = tt = tfz = trz = tso = 0
    c_tot = {k: 0 for k in F5_NAMES}
    a_tot = {k: 0 for k in F5_NAMES}
    for l, sl in enumerate(rec['layers']):
        full_c = B * NN
        comp = int(sl['compute'])
        fones = fones_arr[l]
        fzeros = full_c - fones
        sones = int(round(comp * sl['fill_rate'] / 100.0))
        rzeros = comp - sones
        cnts = sl['counts']
        cbt = sl['cells_by_type']
        sh_top2 = 100.0 * (cbt['64x32'] + cbt['32x32']) / max(1, comp)
        rows.append(dict(
            layer=l + 1,
            density_pct=round(100.0 * fones / full_c, 1),
            full_cells=full_c,
            comp_cells=comp,
            tiles=int(sl['tiles']),
            counts=cnts,
            share_64x32=round(100.0 * cbt['64x32'] / max(1, comp), 1),
            share_32x32=round(100.0 * cbt['32x32'] / max(1, comp), 1),
            share_top2=round(sh_top2, 1),
            share_8x16=round(100.0 * cbt['8x16'] / max(1, comp), 1),
            reduce_pct=round(100.0 * (1.0 - comp / full_c), 1),
            remain_pct=round(100.0 * comp / full_c, 1),
            full_zeros=fzeros,
            rem_zeros=rzeros,
            zero_elim_pct=round(100.0 * (1.0 - rzeros / fzeros), 1),
            fill_pct=round(sl['fill_rate'], 1),
            dram_reads=int(sl['dram_reads']),
            dram_pct=round(sl['dram_reads_pct_serving'], 1),
        ))
        tf += full_c; tc += comp; tt += int(sl['tiles'])
        tfz += fzeros; trz += rzeros; tso += sones
        for k in F5_NAMES:
            c_tot[k] += int(cnts[k])
            a_tot[k] += int(cbt[k])

    return dict(
        B=B, label=label, note=note, layers=rows,
        total=dict(
            density_pct=round(100.0 * sum(fones_arr) / tf, 1),
            full_cells=tf, comp_cells=tc, tiles=tt,
            counts=c_tot,
            share_64x32=round(100.0 * a_tot['64x32'] / max(1, tc), 1),
            share_32x32=round(100.0 * a_tot['32x32'] / max(1, tc), 1),
            share_top2=round(100.0 * (a_tot['64x32'] + a_tot['32x32']) / max(1, tc), 1),
            share_32x16=round(100.0 * a_tot['32x16'] / max(1, tc), 1),
            share_16x16=round(100.0 * a_tot['16x16'] / max(1, tc), 1),
            share_8x16=round(100.0 * a_tot['8x16'] / max(1, tc), 1),
            reduce_pct=round(100.0 * (1.0 - tc / tf), 1),
            remain_pct=round(100.0 * tc / tf, 1),
            full_zeros=tfz, rem_zeros=trz,
            zero_elim_pct=round(100.0 * (1.0 - trz / tfz), 1),
            fill_pct=round(100.0 * tso / max(1, tc), 1),
            dram_reads=int(rec['dram_reads']),
            dram_pct=round(rec['dram_reads_pct_serving'], 1),
        ),
    )


def main():
    keys, BM, _ = load_keys()
    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)
    q0_512 = np.random.default_rng(0).choice(keys.shape[1], 512, replace=False)

    fones = {
        512: full_ones_per_layer(keys, BM, q0_512),
        1024: full_ones_per_layer(keys, BM, q0_1024),
        2048: full_ones_per_layer(keys, BM, q0_2048),
    }

    ms = json.load(open(os.path.join(RUNS_S, 'ms_roster5_stats.json'), encoding='utf-8'))
    sp = json.load(open(os.path.join(RUNS_S, 'speed_roster5_final.json'), encoding='utf-8'))
    b1 = json.load(open(os.path.join(RUNS_S, 'speed_b1_final.json'), encoding='utf-8'))
    cmp_raw = json.load(open(os.path.join(RUNS_S, 'compare_tiling_scale.json'), encoding='utf-8'))

    # 4) 핵심 기여 1 접이식: 실제 Lc6 4층×4,096 스케일 10회 반복 타일 비교
    cmp_keys = [
        'dense_64x32',
        'matformer_matched',
        'static_1d_32x32',
        'bsmr_16x16_bsa_paper',
        'bsmr_16x16_strip512',
        'bsmr_32x32_strip512',
        'bsmr_5tier_panel_strip512_cost',
        'roster5_strip512_fixed1',
        'roster5_strip512',
        'roster5_global',
    ]
    scale_cmp = dict(
        n_trials=cmp_raw['n_trials'],
        n_global_trials=cmp_raw['n_global_trials'],
        batches={}
    )
    for b_int in (512, 1024, 2048):
        bk = f'B{b_int}'
        br = cmp_raw['results'][bk]
        full_c = br['dense_64x32']['full_cells']
        act_m = int(round(br['dense_64x32']['active']['mean']))
        rows_c = []
        for nm in cmp_keys:
            r = br[nm]
            rows_c.append(dict(
                id=nm,
                trials=r['n_trials'],
                tiles=round(r['tiles']['mean'], 1),
                tiles_std=round(r['tiles']['std'], 1),
                cells=int(round(r['cells']['mean'])),
                cells_pct=round(r['cells_pct']['mean'], 2),
                cells_std=round(r['cells_pct']['std'], 2),
                zeros=int(round(r['zeros']['mean'])),
                zero_pct=round(r['zero_share_pct']['mean'], 1),
                mand_pct=round(r['mand_keep_pct']['mean'], 2),
                opt_off_pct=round(r['opt_off_pct']['mean'], 1),
                cost_pct=round(r['silicon_cost_pct']['mean'], 1),
            ))
        scale_cmp['batches'][str(b_int)] = dict(
            B=b_int, full_cells=full_c, active_mean=act_m,
            active_pct=round(100.0 * act_m / full_c, 2),
            rows=rows_c,
        )

    strip_by_b = {r['B']: r for r in ms['strip_W512']}
    acc = ms['accuracy_10k']

    # 1) 묶기 3: 실제 서버 배치 512 / 1,024 / 2,048 5단계 다중 크기 4개 층 통계 (W=512 열-스트립)
    layer_tables = {
        '512': build_layer_table5(
            512, strip_by_b[512], fones[512],
            '배치 512 · 소형 청크 (열-스트립 W=512)',
            '실제 서버 소형 청크 512토큰 · 오프라인 열 정렬 + 512열 8구역 · 5단계(64×32 → 8×16)',
        ),
        '1024': build_layer_table5(
            1024, strip_by_b[1024], fones[1024],
            '배치 1,024 · Chunked Prefill (W=512)',
            '실제 서버 Chunked Prefill 1,024토큰 · 512열 8구역 · 5단계(64×32 → 8×16)',
        ),
        '2048': build_layer_table5(
            2048, strip_by_b[2048], fones[2048],
            '배치 2,048 · vLLM 기본 청크 (W=512)',
            '실제 서버 vLLM 기본 청크 2,048토큰 · 512열 8구역 · 5단계(64×32 → 8×16)',
        ),
    }

    # 2) 묶기 5: 이론 상한 예상 비교 표 — 단일 요청(B=1) vs 실제 서버(B=512/1024/2048)
    b512 = strip_by_b[512]
    b1024 = strip_by_b[1024]
    b2048 = strip_by_b[2048]

    summary_rows = [
        dict(id='full', name='실제 풀 모델', sub='cuBLAS · 뉴런 다 켬', batch='1 ~ 2,048',
             method='cuBLAS (기준선)', compute_pct=100.0, ones_pct=13.7, zero_pct=86.3,
             dram_pct=100.0, speedup=1.0, speedup_note='기준선'),
        dict(id='single1', name='단일 요청 (로컬 · 개인 LLM)', sub='타일링 불필요 · 활성 뉴런만 읽기', batch='1',
             method='직접 희소 GEMV', compute_pct=13.7, ones_pct=13.7, zero_pct=0.0,
             dram_pct=13.7, speedup=round(100.0 / 13.73, 1), speedup_note='메모리 바운드 · 모든 GPU'),
        dict(id='srv512', name='실제 서버 · 소형 청크', sub='RTX 4060 · A100', batch='512',
             method='5단계 + 열-스트립', compute_pct=round(b512['compute_pct'], 1),
             ones_pct=round(b512['compute_pct'] * b512['fill_rate'] / 100.0, 1),
             zero_pct=round(b512['compute_pct'] * (1.0 - b512['fill_rate'] / 100.0), 1),
             dram_pct=round(b512['dram_reads_pct_serving'], 1),
             speedup=round(b512['speedup_rtx4060'], 1),
             speedup_note=f'4060·A100 (H100 {b512["speedup_h100"]:.1f}배)'),
        dict(id='srv1024', name='실제 서버 · Chunked Prefill', sub='RTX 4060 · A100 · H100', batch='1,024',
             method='5단계 + 열-스트립', compute_pct=round(b1024['compute_pct'], 1),
             ones_pct=round(b1024['compute_pct'] * b1024['fill_rate'] / 100.0, 1),
             zero_pct=round(b1024['compute_pct'] * (1.0 - b1024['fill_rate'] / 100.0), 1),
             dram_pct=round(b1024['dram_reads_pct_serving'], 1),
             speedup=round(b1024['speedup_rtx4060'], 1),
             speedup_note=f'4060·A100 (H100 {b1024["speedup_h100"]:.1f}배)'),
        dict(id='srv2048', name='실제 서버 · vLLM 기본 청크', sub='RTX 4060 · A100 · H100', batch='2,048',
             method='5단계 + 열-스트립', compute_pct=round(b2048['compute_pct'], 1),
             ones_pct=round(b2048['compute_pct'] * b2048['fill_rate'] / 100.0, 1),
             zero_pct=round(b2048['compute_pct'] * (1.0 - b2048['fill_rate'] / 100.0), 1),
             dram_pct=round(b2048['dram_reads_pct_serving'], 1),
             speedup=round(b2048['speedup_h100'], 1),
             speedup_note='4060·A100·H100 동일'),
    ]

    # 3) 묶기 6: 실제 RTX 4060 실측 — 단일 요청(B=1) + 실제 서버(B=512/1024/2048)
    hw_bench = {}
    for d in (1024, 4096):
        ds = sp['dense_sanity'][f'd{d}']
        rb1 = b1['results'][f'd{d}']
        items = []
        for B in (512, 1024, 2048):
            r = sp['results'][f'd{d}_B{B}']
            best_ms = r['global_ms'] if (d == 1024 and r['global_ms'] < r['strip_ms']) else r['strip_ms']
            best_spd = r['speedup_global'] if (d == 1024 and r['global_ms'] < r['strip_ms']) else r['speedup_strip']
            items.append(dict(
                B=B,
                base_ms=r['base_ms'],
                mask0_ms=r['mask0_ms'],
                global_ms=r['global_ms'],
                strip_ms=r['strip_ms'],
                p1_ms=r['p1_ms'],
                p2_ms=r['p2_ms'],
                speedup_mask0=r['speedup_mask0'],
                speedup_global=r['speedup_global'],
                speedup_strip=r['speedup_strip'],
                best_ms=best_ms,
                best_speedup=best_spd,
                caching_speedup=r['caching_speedup'],
                caching_gain_pct=r['caching_gain_pct'],
                max_rel_err=round(r['max_rel_err'], 4),
                layer_speedups=[round(pl['speedup'], 2) for pl in r['per_layer']],
                per_layer=r['per_layer'],
            ))
        hw_bench[str(d)] = dict(
            d=d,
            layer_mb=sp['results'][f'd{d}_B512']['layer_weight_mb'],
            label='작은 모델 d=1,024 (실제 학습 가중치 · 층 24MB < L2)' if d == 1024 else '큰 모델 d=4,096 (대형 LLM 층 · 층 96MB > L2)',
            dense_cublas_tf=ds['cublas_tflops'],
            dense_ours_tf=ds['roster64_tflops'],
            dense_eff_pct=ds['efficiency_pct'],
            p1_cublas_tf=ds['p1_cublas_tf'],
            p1_ours_tf=ds['p1_roster64_tf'],
            p1_eff_pct=ds['p1_efficiency_pct'],
            b1=dict(
                base_ms=rb1['base_ms'],
                mask0_ms=rb1['mask0_ms'],
                ours_ms=rb1['ours_ms'],
                speedup=rb1['speedup_vs_best_base'],
                speedup_mask0=rb1['speedup_mask0'],
                base_gbs=rb1['base_dram_gbs'],
                ours_gbs=rb1['ours_dram_gbs'],
                max_rel_err=round(rb1['max_rel_err'], 4),
                layer_speedups=[round(pl['speedup'], 2) for pl in rb1['per_layer']],
                per_layer=rb1['per_layer'],
                avg_active_pct=b1['avg_active_pct'],
                theoretical_speedup=b1['theoretical_speedup'],
                n_tokens=b1['n_tokens_sampled'],
            ),
            rows=items,
        )

    out = dict(
        version='roster-tile-v3-5tier-twomode',
        gpu=sp['gpu_name'],
        accuracy=dict(
            base=91.02,
            pass_line=90.52,
            orig_model=90.68,
            orig_pct_of_base=round(100.0 * 90.68 / 91.02, 2),
            roster_tile=round(acc['test_acc'], 2),
            pct_of_base=round(acc['pct_of_base'], 2),
            drop_vs_base=round(acc['test_drop_vs_base'], 2),
            ci=[round(acc['test_lo'], 2), round(acc['test_hi'], 2)],
            verdict=acc['test_verdict'],
            drop_vs_orig=round(acc['vs_lc6']['drop'], 2),
            orig_only=int(acc['vs_lc6']['base_only']),
            ms_only=int(acc['vs_lc6']['cand_only']),
            used_avg_pct=round(acc['used_avg_pct'], 2),
            used_per_layer=acc['used_per_layer'],
            avg512_strip_comp_pct=round(acc['avg_batch_compute_pct_full'], 1),
        ),
        layer_tables=layer_tables,
        summary=summary_rows,
        hw_bench=hw_bench,
        scale_cmp=scale_cmp,
        strip_meta=dict(
            W=512, n_strips=8,
            runtime_512=round(b512['runtime_sec'], 2),
            runtime_1024=round(b1024['runtime_sec'], 2),
            runtime_2048=round(b2048['runtime_sec'], 2),
        ),
    )

    path = os.path.join(RUNS_S, 'viz_batch.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print(f'저장: {path} ({os.path.getsize(path) / 1024:.1f} KB)')
    print('=== 묶기 5 이론 상한 (단일 vs 서버) ===')
    for s in summary_rows:
        print(f'  {s["name"]:24s} B={s["batch"]:9s} | 연산 {s["compute_pct"]:5.1f}% | 읽기 {s["dram_pct"]:5.1f}% | 속도 {s["speedup"]:.1f}x ({s["speedup_note"]})')
    print('=== 묶기 6 RTX 4060 실측 ===')
    for dk in ('1024', '4096'):
        h = hw_bench[dk]
        print(f'  [{h["label"]}] 밀집 64x32 P1={h["p1_eff_pct"]:.1f}% (전체 {h["dense_eff_pct"]:.1f}%)')
        print(f'    단일 B=1  : Base {h["b1"]["base_ms"]:7.4f} | 0-Mask {h["b1"]["speedup_mask0"]:.2f}x | Roster {h["b1"]["ours_ms"]:7.4f} ms ({h["b1"]["speedup"]:.2f}x, 층별 {h["b1"]["layer_speedups"]})')
        for r in h['rows']:
            print(f'    서버 B={r["B"]:4d}: Base {r["base_ms"]:7.3f} | 0-Mask {r["speedup_mask0"]:.2f}x | Global {r["speedup_global"]:.2f}x | Strip {r["speedup_strip"]:.2f}x (층별 {r["layer_speedups"]})')


if __name__ == '__main__':
    main()
