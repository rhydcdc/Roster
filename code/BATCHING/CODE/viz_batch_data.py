"""레시피 HTML '묶어서 계산' 보기(Roster-Tile 최신 버전)에 넣을 데이터를 모은다.
원자료:
  - runs/batch_neon_q1.json   : 실제 Lc6 배치 256(q0) 4개 층 상세 통계
  - runs/neon_acc.json        : 실제 Lc6 시험 10,000장 정확도 + 39개 256-배치 평균
  - runs/batch_strip_W512.json: 실제 Lc6 배치 256/1024/2048 열-스트립(W=512) 4개 층 통계
  - runs/keys.npz             : 배치별 원본 표의 1/0 개수 (불필요한 0 제거율 산출)
python -X utf8 -W ignore BATCHING/CODE/viz_batch_data.py -> runs/viz_batch.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batch_s import load_keys, queues, RUNS_S, NN  # noqa: E402


def full_ones_per_layer(keys, BM, order_idx):
    out = []
    for l in range(4):
        kq = keys[l][order_idx]
        out.append(int(BM[l][kq].sum()))
    return out


def main():
    keys, BM, _ = load_keys()
    q0_256 = next(o for N, t, o in queues(keys.shape[1]) if N == 256 and t == 0)
    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)

    fones_256 = full_ones_per_layer(keys, BM, q0_256)
    fones_1024 = full_ones_per_layer(keys, BM, q0_1024)
    fones_2048 = full_ones_per_layer(keys, BM, q0_2048)

    q1 = json.load(open(os.path.join(RUNS_S, 'batch_neon_q1.json'), encoding='utf-8'))
    acc = json.load(open(os.path.join(RUNS_S, 'neon_acc.json'), encoding='utf-8'))
    st = json.load(open(os.path.join(RUNS_S, 'batch_strip_W512.json'), encoding='utf-8'))
    st_by_b = {r['B']: r for r in st['results']}

    # 1) 묶기 3: 배치별 실제 신경망 4개 층 통계 (256=Roster-Tile, 1024/2048=열-스트립 W=512)
    layer_tables = {}

    # B = 256 (q0 Roster-Tile 실측)
    rows_256 = []
    tot_full_256 = tot_comp_256 = tot_t_256 = tot_A_256 = tot_B_256 = 0
    tot_fzero_256 = tot_rzero_256 = tot_ones_256 = 0
    for l, ql in enumerate(q1['queues'][0]['layers']):
        full_c = 256 * NN
        comp = int(ql['neon_compute'])
        fones = fones_256[l]
        fzeros = full_c - fones
        rzeros = int(ql['total_zeros'])
        sones = comp - rzeros
        rows_256.append(dict(
            layer=l + 1,
            density_pct=round(100.0 * fones / full_c, 1),
            full_cells=full_c,
            comp_cells=comp,
            tiles=int(ql['num_tiles']),
            tiles_A=int(ql['num_tiles_A']),
            tiles_B=int(ql['num_tiles_B']),
            reduce_pct=round(100.0 * (1.0 - comp / full_c), 1),
            remain_pct=round(100.0 * comp / full_c, 1),
            full_zeros=fzeros,
            rem_zeros=rzeros,
            zero_elim_pct=round(100.0 * (1.0 - rzeros / fzeros), 1),
            fill_pct=round(100.0 * sones / comp, 1),
            dram_reads=int(ql['unique_cols_read']),
            dram_pct=round(100.0 * ql['unique_cols_read'] / NN, 1),
        ))
        tot_full_256 += full_c; tot_comp_256 += comp
        tot_t_256 += int(ql['num_tiles']); tot_A_256 += int(ql['num_tiles_A']); tot_B_256 += int(ql['num_tiles_B'])
        tot_fzero_256 += fzeros; tot_rzero_256 += rzeros; tot_ones_256 += sones

    layer_tables['256'] = dict(
        B=256,
        label='배치 256 · 작은 모델 (Roster-Tile)',
        note='시험 사진 256장(q0), 4개 층 각 256 × 4,096표 (39개 배치 평균 연산 18.1%)',
        layers=rows_256,
        total=dict(
            density_pct=round(100.0 * sum(fones_256) / tot_full_256, 1),
            full_cells=tot_full_256,
            comp_cells=tot_comp_256,
            tiles=tot_t_256, tiles_A=tot_A_256, tiles_B=tot_B_256,
            reduce_pct=round(100.0 * (1.0 - tot_comp_256 / tot_full_256), 1),
            remain_pct=round(100.0 * tot_comp_256 / tot_full_256, 1),
            full_zeros=tot_fzero_256,
            rem_zeros=tot_rzero_256,
            zero_elim_pct=round(100.0 * (1.0 - tot_rzero_256 / tot_fzero_256), 1),
            fill_pct=round(100.0 * tot_ones_256 / tot_comp_256, 1),
            dram_reads=sum(r['dram_reads'] for r in rows_256),
            dram_pct=round(100.0 * sum(r['dram_reads'] for r in rows_256) / (4 * NN), 1),
        ),
    )

    for B, fones_arr, lbl, nt in [
        (1024, fones_1024, '배치 1,024 · 큰 모델 (열-스트립 W=512)', '시험 사진 1,024장, 오프라인 열 정렬 + 512열 8구역 스트립, 4개 층 각 1,024 × 4,096표'),
        (2048, fones_2048, '배치 2,048 · 큰 모델 (열-스트립 W=512)', '시험 사진 2,048장, 오프라인 열 정렬 + 512열 8구역 스트립, 4개 층 각 2,048 × 4,096표'),
    ]:
        sb = st_by_b[B]
        rows = []
        tf = tc = tt = tA = tB = tfz = trz = tso = 0
        for l, sl in enumerate(sb['layers']):
            full_c = B * NN
            comp = int(sl['compute'])
            fones = fones_arr[l]
            fzeros = full_c - fones
            sones = int(round(comp * sl['fill_rate'] / 100.0))
            rzeros = comp - sones
            rows.append(dict(
                layer=l + 1,
                density_pct=round(100.0 * fones / full_c, 1),
                full_cells=full_c,
                comp_cells=comp,
                tiles=int(sl['tiles']),
                tiles_A=int(sl['tiles_A']),
                tiles_B=int(sl['tiles_B']),
                reduce_pct=round(100.0 * (1.0 - comp / full_c), 1),
                remain_pct=round(100.0 * comp / full_c, 1),
                full_zeros=fzeros,
                rem_zeros=rzeros,
                zero_elim_pct=round(100.0 * (1.0 - rzeros / fzeros), 1),
                fill_pct=round(sl['fill_rate'], 1),
                dram_reads=int(sl['dram_reads']),
                dram_pct=round(sl['dram_reads_pct_serving'], 1),
            ))
            tf += full_c; tc += comp
            tt += int(sl['tiles']); tA += int(sl['tiles_A']); tB += int(sl['tiles_B'])
            tfz += fzeros; trz += rzeros; tso += sones
        layer_tables[str(B)] = dict(
            B=B, label=lbl, note=nt, layers=rows,
            total=dict(
                density_pct=round(100.0 * sum(fones_arr) / tf, 1),
                full_cells=tf, comp_cells=tc,
                tiles=tt, tiles_A=tA, tiles_B=tB,
                reduce_pct=round(100.0 * (1.0 - tc / tf), 1),
                remain_pct=round(100.0 * tc / tf, 1),
                full_zeros=tfz, rem_zeros=trz,
                zero_elim_pct=round(100.0 * (1.0 - trz / tfz), 1),
                fill_pct=round(100.0 * tso / tc, 1),
                dram_reads=int(sb['dram_reads']),
                dram_pct=round(sb['dram_reads_pct_serving'], 1),
            ),
        )

    # 2) 묶기 5: 최종 가속 비교 표 (실제 서빙 풀 모델 = 100%, 1.0배 기준)
    b256_comp = round(acc['avg256_compute_pct_full'], 1)  # 18.1
    b256_dram = round(layer_tables['256']['total']['dram_pct'], 1)  # 81.5
    b256_spd = round(256.0 / max((acc['avg256_compute_pct_full'] / 100.0) * 256.0, (b256_dram / 100.0) * 104.0), 1)

    b1024 = st_by_b[1024]
    b2048 = st_by_b[2048]

    summary_rows = [
        dict(id='full', name='실제 풀 모델', sub='cuBLAS · 뉴런 다 켬', batch='256 / 1,024 / 2,048',
             method='cuBLAS (기준선)', compute_pct=100.0, ones_pct=13.7, zero_pct=86.3,
             dram_pct=100.0, speedup=1.0, speedup_note='기준선'),
        dict(id='small256', name='지금 작은 모델', sub='RTX 4060 · L2 32MB', batch='256',
             method='Roster-Tile', compute_pct=b256_comp,
             ones_pct=round(b256_comp * layer_tables['256']['total']['fill_pct'] / 100.0, 1),
             zero_pct=round(b256_comp * (1.0 - layer_tables['256']['total']['fill_pct'] / 100.0), 1),
             dram_pct=b256_dram, speedup=b256_spd, speedup_note='RTX 4060'),
        dict(id='large1024', name='큰 모델 (대형 LLM)', sub='H100 · Chunked Prefill', batch='1,024',
             method='열-스트립 Roster-Tile', compute_pct=round(b1024['compute_pct'], 1),
             ones_pct=round(b1024['compute_pct'] * b1024['fill_rate'] / 100.0, 1),
             zero_pct=round(b1024['compute_pct'] * (1.0 - b1024['fill_rate'] / 100.0), 1),
             dram_pct=round(b1024['dram_reads_pct_serving'], 1),
             speedup=round(b1024['speedup_h100'], 1),
             speedup_note=f'H100 (A100 {b1024["speedup_a100"]:.1f}배)'),
        dict(id='large2048', name='큰 모델 (대형 LLM)', sub='H100 · vLLM 기본 청크', batch='2,048',
             method='열-스트립 Roster-Tile', compute_pct=round(b2048['compute_pct'], 1),
             ones_pct=round(b2048['compute_pct'] * b2048['fill_rate'] / 100.0, 1),
             zero_pct=round(b2048['compute_pct'] * (1.0 - b2048['fill_rate'] / 100.0), 1),
             dram_pct=round(b2048['dram_reads_pct_serving'], 1),
             speedup=round(b2048['speedup_h100'], 1),
             speedup_note='H100 · A100 동일'),
    ]

    out = dict(
        version='roster-tile-v1',
        accuracy=dict(
            base=91.02,
            pass_line=90.52,
            lc6=90.68,
            roster_tile=round(acc['test_acc'], 2),
            pct_of_base=round(100.0 * acc['test_acc'] / 91.02, 2),
            drop_vs_base=round(acc['test_drop_vs_base'], 2),
            ci=[round(acc['test_lo'], 2), round(acc['test_hi'], 2)],
            verdict=acc['test_verdict'],
            drop_vs_lc6=round(acc['vs_lc6']['drop'], 2),
            lc6_only=int(acc['vs_lc6']['base_only']),
            neon_only=int(acc['vs_lc6']['cand_only']),
            used_avg_pct=round(acc['used_avg_pct'], 2),
            used_per_layer=acc['used_per_layer'],
        ),
        layer_tables=layer_tables,
        summary=summary_rows,
        strip_meta=dict(W=512, n_strips=8, runtime_256=round(st_by_b[256]['runtime_sec'], 1),
                        runtime_1024=round(b1024['runtime_sec'], 1), runtime_2048=round(b2048['runtime_sec'], 1)),
    )

    path = os.path.join(RUNS_S, 'viz_batch.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print(f'저장: {path} ({os.path.getsize(path) / 1024:.1f} KB)')
    for s in summary_rows:
        print(f'  {s["name"]:18s} B={s["batch"]:19s} | 연산 {s["compute_pct"]:5.1f}% | 읽기 {s["dram_pct"]:5.1f}% | 속도 {s["speedup"]:.1f}x ({s["speedup_note"]})')


if __name__ == '__main__':
    main()
