"""Extract ALL real CH2_M32_B1B2_r9 data + sprites for ROSTER_VIZ.html into VIZ/data_roster/.

Zero fabricated numbers:
  - pretrain : Base model s1_seed2 log.jsonl (91.02% test)
  - layers   : Real M_l = [32, 32, 32, 256] centroids from init_M_32_L123_eps002.pt,
               3,000-sample t-SNE per layer, 4 featured centroids per layer (classes 7,1,8,0),
               rep_l1..4.png & feat.png sprites
  - rounds   : Real rounds 0..9 (init + S_M32_B2blk_r1 + CH2_M32_B1B2_r2..r9)
               Round 0 & Round 1 & Round 9 have checkpoints on disk -> exact before/after,
               match_before/match_after, and exact e(k) output-error curves!
               Rounds 2..8 use the exact saved per-round S, NR, KJ tables and CH2_M32_B1B2.json.
  - examples : 12 real Fashion-MNIST test images (classes 0..9, 7, 8) traced through all 4 layers
               of CH2_M32_B1B2_r9 (M_l = 32/32/32/256, 1* = 0%, 90.81% test) + inf.png sprite
  - batch.json : Unified Tabs 3 & 4 payload assembled strictly from:
                 b_10k_accuracy.json, track_b_final.json, speed_b1_ch2.json,
                 compare_bsmr_ch2.json, b_structural_screen.json, CH2_M32_B1B2.json.
"""
import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import torch

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
RUNS_B = HERE_B / 'runs'
OUTD = PROJ / 'VIZ' / 'data_roster'
OUTD.mkdir(parents=True, exist_ok=True)

for p in (str(HERE_B), str(PROJ / 'OPT' / 'TRACK_A'), str(PROJ / 'BATCHING' / 'CODE')):
    if p not in sys.path:
        sys.path.insert(0, p)

from track_b_lib import (  # noqa: E402
    BASE_MODEL_PATH, DEV, KGRID_4096, N_BLOCK, N_NEURON,
    build_encoded_tables_flex, compute_pos_olo_layer, evaluate,
    extract_keys_split, jload, keys_layer, lkey, load_data, load_net,
    nearest, run_masked, var_mask,
)
from a3_dispatch_v4 import A3V4Dispatcher  # noqa: E402
from batch_strip import compute_offline_col_orders  # noqa: E402

CLASSES = ['티셔츠', '바지', '풀오버', '원피스', '코트', '샌들', '셔츠', '운동화', '가방', '앵클부츠']
FEAT_CLASSES = [7, 1, 8, 0]  # 운동화, 바지, 가방, 티셔츠
NN = 4096


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def q8(v):
    v = np.asarray(v, dtype=np.float32)
    s = float(np.abs(v).max()) or 1.0
    return dict(s=round(s, 5), q=b64(np.clip(np.round(v / s * 127), -127, 127).astype(np.int8)))


def u8(v):
    v = np.sqrt(np.clip(np.asarray(v, dtype=np.float32), 0, None))
    m = float(v.max()) or 1.0
    return b64(np.round(v / m * 255).astype(np.uint8))


def bits(on):
    return b64(np.packbits(np.asarray(on, dtype=np.uint8)))


def r(x, n=4):
    return round(float(x), n)


def sprite(imgs, cols, path):
    n = len(imgs)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * 28, cols * 28), dtype=np.uint8)
    for i, im in enumerate(imgs):
        rr, cc = divmod(i, cols)
        canvas[rr * 28:(rr + 1) * 28, cc * 28:(cc + 1) * 28] = im
    Image.fromarray(canvas, 'L').save(str(path), optimize=True)
    return dict(file=Path(path).name, cols=cols, n=n)


def pix(x):
    return np.clip(np.round((x.cpu().numpy() * 0.3530 + 0.2860) * 255), 0, 255).astype(np.uint8)


@torch.no_grad()
def walk(model, X, C, NR, KJ, keep=None, chunk=2500):
    model.eval()
    N = X.shape[0]
    keys = [torch.empty(N, dtype=torch.long, device=DEV) for _ in range(N_BLOCK)]
    dist = [torch.empty(N, device=DEV) for _ in range(N_BLOCK)]
    got = {k: dict(S=torch.zeros(N_NEURON, device=DEV), A=[], ON=[], H=[], FULL=[]) for k in (keep or {})}
    for s in range(0, N, chunk):
        xb = X[s:s + chunk]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                kf = lkey(h)
                d = (C[l] * C[l]).sum(1)[None, :] - 2 * kf @ C[l].T + (kf * kf).sum(1, keepdim=True)
                dm, j = d.min(1)
            keys[l][s:s + chunk] = j
            dist[l][s:s + chunk] = dm
            on = NR[l][j] < KJ[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                full = blk.down(a * on.to(a.dtype)).float()
            if keep:
                c = a.float().abs() * blk.colnorm()
                for (ll, jj), g in got.items():
                    if ll != l:
                        continue
                    sel = j == jj
                    if sel.any():
                        g['S'] += (c[sel] * on[sel]).sum(0)
                        g['A'].append(a[sel].float())
                        g['ON'].append(on[sel])
                        g['H'].append(h[sel].float())
                        g['FULL'].append(full[sel])
            h = h + full
    for g in got.values():
        for k in ('A', 'ON', 'H', 'FULL'):
            g[k] = torch.cat(g[k]) if g[k] else None
    return keys, dist, got


@torch.no_grad()
def teacher_err(base, l, H, FULL):
    tb = base.blocks[l]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        t = tb.down(tb.neurons(H)).float()
    return float((((FULL - t) ** 2).sum(1) / ((t ** 2).sum(1) + 1e-6)).mean())


def cum_curve(S_row, NR_row, Kold, npts=96):
    Kold = max(1, int(Kold))
    order = torch.argsort(NR_row)[:Kold]
    v = S_row[order].float().clamp_min(0)
    cs = (v.cumsum(0) / (v.sum() + 1e-12)).cpu().numpy()
    xs = np.unique(np.linspace(0, Kold - 1, min(npts, Kold)).round().astype(int))
    return dict(x=[int(i + 1) for i in xs], y=[r(cs[i], 4) for i in xs])


@torch.no_grad()
def err_curve(blk, g, NR_row, Kold):
    Sl = g['S'].clone()
    Sl[NR_row >= Kold] = -1.0
    order = Sl.argsort(descending=True)
    nr = torch.empty_like(order)
    nr[order] = torch.arange(N_NEURON, device=DEV)
    A, FULL = g['A'], g['FULL']
    ks, es = [], []
    for k in KGRID_4096:
        kk = min(k, Kold)
        on = (nr < kk)[None, :]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = blk.down(A.to(torch.bfloat16) * on.to(torch.bfloat16)).float()
        e = (((o - FULL) ** 2).sum(1) / ((FULL ** 2).sum(1) + 1e-6)).mean()
        ks.append(int(kk))
        es.append(float(e))
        if k >= Kold:
            break
    return nr, ks, es


def build_batch_json():
    """Assemble VIZ/data_roster/batch.json strictly from official Track B JSON outputs."""
    acc = jload(str(RUNS_B / 'b_10k_accuracy.json'))
    tbf = jload(str(RUNS_B / 'track_b_final.json'))
    sb1 = jload(str(RUNS_B / 'speed_b1_ch2.json'))
    bsmr = jload(str(RUNS_B / 'compare_bsmr_ch2.json'))
    scr = jload(str(RUNS_B / 'b_structural_screen.json'))
    ch2 = jload(str(RUNS_B / 'CH2_M32_B1B2.json'))

    w = acc['primary']
    # Build layer_tables for B = 512, 1024, 2048 using exact_tile_mix + track_b_final
    layer_tables = {}
    b_labels = {
        '512': ('배치 512 · 소형 청크 (열-스트립 W=512)', '실제 서버 소형 청크 512토큰 · CH2 M=[32,32,32,256] · 1*=0% · A3V4 5단계'),
        '1024': ('배치 1,024 · Chunked Prefill (W=512)', '실제 서버 Chunked Prefill 1,024토큰 · CH2 M=[32,32,32,256] · 1*=0% · A3V4 5단계'),
        '2048': ('배치 2,048 · vLLM 기본 청크 (W=512)', '실제 서버 vLLM 기본 청크 2,048토큰 · CH2 M=[32,32,32,256] · 1*=0% · A3V4 5단계'),
    }
    areas = {'64x32': 2048, '32x32': 1024, '32x16': 512, '16x16': 256, '8x16': 128}

    for bs in ('512', '1024', '2048'):
        B = int(bs)
        em = bsmr['exact_tile_mix'][bs]
        tb_row = tbf['results'][f'd4096_B{bs}']
        layers_out = []
        tot_full = 4 * B * NN
        tot_comp = 0
        tot_ones = 0
        tot_dram = 0
        for pl_m, pl_tb in zip(em['per_layer'], tb_row['per_layer']):
            full_c = B * NN
            cnts = pl_m['counts']
            comp_c = sum(cnts[k] * areas[k] for k in areas)
            ones_c = int(round(comp_c * pl_m['fill_pct'] / 100.0))
            rem_z = comp_c - ones_c
            full_z = full_c - ones_c
            sh64 = round(100.0 * cnts['64x32'] * 2048 / max(comp_c, 1), 1)
            sh32 = round(100.0 * cnts['32x32'] * 1024 / max(comp_c, 1), 1)
            sh8 = round(100.0 * cnts['8x16'] * 128 / max(comp_c, 1), 1)
            dram_cols = int(round(NN * pl_m['dram_read_pct'] / 100.0))
            tot_comp += comp_c
            tot_ones += ones_c
            tot_dram += dram_cols
            layers_out.append(dict(
                layer=pl_m['layer'],
                density_pct=round(pl_m['used_pct'], 1),
                full_cells=full_c,
                comp_cells=comp_c,
                tiles=pl_m['tiles'],
                counts=cnts,
                share_64x32=sh64,
                share_32x32=sh32,
                share_top2=round(sh64 + sh32, 1),
                share_8x16=sh8,
                reduce_pct=round(100.0 - pl_m['comp_pct'], 1),
                remain_pct=round(pl_m['comp_pct'], 1),
                p2_union_pct=round(pl_tb['p2_union_pct'], 1),
                p2_tm=int(pl_tb['p2_tm']),
                ffn_flops_pct=round(pl_tb['ffn_flops_pct'], 1),
                full_zeros=full_z,
                rem_zeros=rem_z,
                zero_elim_pct=round(100.0 * (1.0 - rem_z / max(full_z, 1)), 1),
                fill_pct=round(pl_m['fill_pct'], 1),
                x0_pct=round(pl_m['x0_pct'], 1),
                dram_reads=dram_cols,
                dram_pct=round(pl_m['dram_read_pct'], 1),
                dispatch_1l_ms=pl_tb['dispatch_1l_b_ms'],
            ))
        tcnt = em['counts_total']
        tsh64 = round(100.0 * tcnt['64x32'] * 2048 / max(tot_comp, 1), 1)
        tsh32 = round(100.0 * tcnt['32x32'] * 1024 / max(tot_comp, 1), 1)
        tsh3216 = round(100.0 * tcnt['32x16'] * 512 / max(tot_comp, 1), 1)
        tsh16 = round(100.0 * tcnt['16x16'] * 256 / max(tot_comp, 1), 1)
        tsh8 = round(100.0 * tcnt['8x16'] * 128 / max(tot_comp, 1), 1)
        tot_rem_z = tot_comp - tot_ones
        tot_full_z = tot_full - tot_ones
        lbl, note = b_labels[bs]
        layer_tables[bs] = dict(
            B=B, label=lbl, note=note,
            layers=layers_out,
            total=dict(
                density_pct=round(100.0 * tot_ones / tot_full, 1),
                full_cells=tot_full,
                comp_cells=tot_comp,
                tiles=em['tiles_total'],
                counts=tcnt,
                share_64x32=tsh64,
                share_32x32=tsh32,
                share_top2=round(tsh64 + tsh32, 1),
                share_32x16=tsh3216,
                share_16x16=tsh16,
                share_8x16=tsh8,
                reduce_pct=round(100.0 - tb_row['b_p1_comp_pct'], 1),
                remain_pct=round(tb_row['b_p1_comp_pct'], 1),
                p2_union_pct=round(tb_row['b_p2_union_pct'], 1),
                ffn_flops_pct=round(tb_row['b_ffn_flops_pct'], 1),
                full_zeros=tot_full_z,
                rem_zeros=tot_rem_z,
                zero_elim_pct=round(100.0 * (1.0 - tot_rem_z / max(tot_full_z, 1)), 1),
                fill_pct=round(tb_row['b_fill_rate'], 1),
                x0_pct=round(tb_row['b_x0_pct'], 1),
                dram_reads=tot_dram,
                dram_pct=round(tb_row['b_dram_read_pct'], 1),
                dispatch_4l_ms=tb_row['b_dispatch_4l_ms'],
            ),
        )

    # Roofline theoretical summary (using FFN FLOPs% = (2*P1 + P2)/3 and DRAM read%)
    r512 = tbf['results']['d4096_B512']
    r1024 = tbf['results']['d4096_B1024']
    r2048 = tbf['results']['d4096_B2048']
    summary = [
        dict(id='full', name='실제 풀 모델', sub='cuBLAS · 뉴런 다 켬', batch='1 ~ 2,048',
             method='cuBLAS (기준선)', compute_pct=100.0, ones_pct=13.5, zero_pct=86.5,
             dram_pct=100.0, speedup=1.0, speedup_note='기준선'),
        dict(id='single1', name='단일 요청 (로컬 · 개인 LLM)', sub='타일링 불필요 · 활성 뉴런만 읽기', batch='1',
             method='직접 희소 GEMV', compute_pct=13.5, ones_pct=13.5, zero_pct=0.0,
             dram_pct=13.5, speedup=7.4, speedup_note='메모리 바운드 · 모든 GPU (1 / 0.1352)'),
        dict(id='srv512', name='실제 서버 · 소형 청크', sub='RTX 4060 · A100 · H100', batch='512',
             method='A3V4 5단계 + W=512 (1*=0%)', compute_pct=round(r512['b_ffn_flops_pct'], 1),
             ones_pct=round(r512['b_ffn_flops_pct'] * r512['b_fill_rate'] / 100.0, 1),
             zero_pct=round(r512['b_ffn_flops_pct'] * r512['b_x0_pct'] / 100.0, 1),
             dram_pct=round(r512['b_dram_read_pct'], 1),
             speedup=round(100.0 / r512['b_ffn_flops_pct'], 1),
             speedup_note='4060·A100 연산 상한 (H100 2.7배)'),
        dict(id='srv1024', name='실제 서버 · Chunked Prefill', sub='RTX 4060 · A100 · H100', batch='1,024',
             method='A3V4 5단계 + W=512 (1*=0%)', compute_pct=round(r1024['b_ffn_flops_pct'], 1),
             ones_pct=round(r1024['b_ffn_flops_pct'] * r1024['b_fill_rate'] / 100.0, 1),
             zero_pct=round(r1024['b_ffn_flops_pct'] * r1024['b_x0_pct'] / 100.0, 1),
             dram_pct=round(r1024['b_dram_read_pct'], 1),
             speedup=round(100.0 / r1024['b_ffn_flops_pct'], 1),
             speedup_note='4060·A100·H100 연산 바운드'),
        dict(id='srv2048', name='실제 서버 · vLLM 기본 청크', sub='RTX 4060 · A100 · H100', batch='2,048',
             method='A3V4 5단계 + W=512 (1*=0%)', compute_pct=round(r2048['b_ffn_flops_pct'], 1),
             ones_pct=round(r2048['b_ffn_flops_pct'] * r2048['b_fill_rate'] / 100.0, 1),
             zero_pct=round(r2048['b_ffn_flops_pct'] * r2048['b_x0_pct'] / 100.0, 1),
             dram_pct=round(r2048['b_dram_read_pct'], 1),
             speedup=round(100.0 / r2048['b_ffn_flops_pct'], 1),
             speedup_note='4060·A100·H100 동일 연산 상한'),
    ]

    # HW Benchmark payload (both d=1024 and d=4096, with Attention-Overlapped E2E AND Serial E2E separated!)
    hw_bench = {}
    for d in (1024, 4096):
        ds = tbf['dense_sanity'][f'd{d}']
        b1_raw = sb1['results'][f'd{d}']
        rows = []
        for B in (512, 1024, 2048):
            row = tbf['results'][f'd{d}_B{B}']
            rows.append(dict(
                B=B,
                base_ms=row['base_ms'],
                mask0_ms=row['mask0_ms'],
                mask0_x=row['mask0_x'],
                track_a_ms=row['track_a_ms'],
                track_a_x=row['track_a_x'],
                gpu_ms=row['track_b_ms'],
                gpu_x=row['track_b_x'],
                p1_ms=row['track_b_p1_ms'],
                p2_ms=row['track_b_p2_ms'],
                p1_comp_pct=row['b_p1_comp_pct'],
                p2_union_pct=row['b_p2_union_pct'],
                ffn_flops_pct=row['b_ffn_flops_pct'],
                fill_rate=row['b_fill_rate'],
                x0_pct=row['b_x0_pct'],
                dram_read_pct=row['b_dram_read_pct'],
                tiles=row['b_tiles'],
                dispatch_4l_ms=row['b_dispatch_4l_ms'],
                dispatch_1l_ms=row['b_dispatch_1l_ms'],
                h2d_ms=row['h2d_b_ms'],
                h2d_kb=row['h2d_b_kb'],
                attn_1l_ms=row['attn_1l_ms'],
                attn_exposed_ms=row['attn_exposed_b_ms'],
                e2e_ovlp_ms=row['e2e_ovlp_b_ms'],
                e2e_ovlp_x=row['e2e_ovlp_b_x'],
                e2e_serial_ms=row['e2e_serial_b_ms'],
                e2e_serial_x=row['e2e_serial_b_x'],
                max_rel_err=row['max_rel_err_b'],
                layer_speedups=[pl['track_b_x'] for pl in row['per_layer']],
                per_layer=row['per_layer'],
                # Track A comparison stats for #b4b / #c2
                a_p1_comp_pct=row['a_p1_comp_pct'],
                a_p2_union_pct=row['a_p2_union_pct'],
                a_ffn_flops_pct=row['a_ffn_flops_pct'],
                a_fill_rate=row['a_fill_rate'],
                a_x0_pct=row['a_x0_pct'],
                a_dram_read_pct=row['a_dram_read_pct'],
                a_tiles=row['a_tiles'],
                a_dispatch_4l_ms=row['a_dispatch_4l_ms'],
                a_e2e_serial_x=row['e2e_serial_a_x'],
                b_vs_a_gpu_gain_pct=row['b_vs_a_gpu_gain_pct'],
            ))
        hw_bench[str(d)] = dict(
            d=d,
            layer_mb=24.0 if d == 1024 else 96.0,
            label=('작은 모델 d=1,024 (실제 CH2 가중치 · 층 24MB < L2)' if d == 1024
                   else '큰 모델 d=4,096 (대형 LLM 층 · 층 96MB > L2)'),
            dense_cublas_tf=ds['cublas_tf'],
            dense_v4_tf=ds['v4_tf'],
            dense_eff_pct=ds['efficiency_pct'],
            b1=dict(
                base_ms=b1_raw['base_ms'],
                mask0_ms=b1_raw['mask0_ms'],
                ours_ms=b1_raw['ours_ms'],
                speedup=b1_raw['speedup_vs_best_base'],
                speedup_mask0=b1_raw['speedup_mask0'],
                base_gbs=b1_raw['base_dram_gbs'],
                ours_gbs=b1_raw['ours_dram_gbs'],
                max_rel_err=b1_raw['max_rel_err'],
                layer_speedups=[pl['speedup'] for pl in b1_raw['per_layer']],
                per_layer=b1_raw['per_layer'],
                avg_active_pct=sb1['avg_active_pct'],
                avg_active_10k_pct=sb1['avg_active_10k_pct'],
                theoretical_speedup=sb1['theoretical_speedup'],
                theoretical_10k_speedup=sb1['theoretical_10k_speedup'],
                n_tokens=sb1['n_tokens_sampled'],
            ),
            rows=rows,
        )

    # Scale comparison (10 trials on CH2) formatted for #c3 fold
    scale_batches = {}
    row_order = [
        'dense_64x32',
        'matformer_matched',
        'static1d_32x32',
        'bsmr_16x16_bsa_strip512',
        'bsmr_16x16_strip512',
        'bsmr_32x32_strip512',
        'bsmr_multiscale_cost',
        'our_a3v4_strip512',
        'our_ms_strip512',
    ]
    for bs in ('512', '1024', '2048'):
        B = int(bs)
        full_c = 4 * B * NN
        r_b = bsmr['results'][bs]
        act_pct = r_b['dense_64x32']['active_pct']['mean']
        rows_s = []
        for mid in row_order:
            m = r_b[mid]
            cells = int(round(full_c * m['cells_pct']['mean'] / 100.0))
            zeros = int(round(full_c * m['zeros_pct']['mean'] / 100.0))
            rows_s.append(dict(
                id=mid,
                trials=m['n_trials'],
                tiles=m['tiles']['mean'],
                tiles_std=m['tiles']['std'],
                cells=cells,
                cells_pct=m['cells_pct']['mean'],
                cells_std=m['cells_pct']['std'],
                zeros=zeros,
                zero_pct=m['zero_share_pct']['mean'],
                fill_pct=m['fill_pct']['mean'],
                mand_pct=m['mand_keep_pct']['mean'],
                cost_pct=m['cost_pct']['mean'],
            ))
        scale_batches[bs] = dict(
            B=B, full_cells=full_c,
            active_mean=int(round(full_c * act_pct / 100.0)),
            active_pct=act_pct,
            rows=rows_s,
        )

    # Structural codesign progression for Tab 4 #c2 (Track A vs Track B measured in track_b_final.json)
    codesign = dict(
        track_a=dict(
            label='개별 몰기 (M=256)',
            Ms=[256, 256, 256, 256],
            p1_2048=r2048['a_p1_comp_pct'],
            p2_2048=r2048['a_p2_union_pct'],
            ffn_2048=r2048['a_ffn_flops_pct'],
            x0_2048=r2048['a_x0_pct'],
            fill_2048=r2048['a_fill_rate'],
            dram_2048=r2048['a_dram_read_pct'],
            tiles_2048=r2048['a_tiles'],
            disp_2048_ms=r2048['a_dispatch_4l_ms'],
            gpu_2048_x=r2048['track_a_x'],
            serial_2048_x=r2048['e2e_serial_a_x'],
        ),
        track_b=dict(
            label='공유 사후 학습 (M=[32,32,32,256] + B1/B2 · 1*=0%)',
            Ms=[32, 32, 32, 256],
            test_acc=90.81,
            k_pct=13.52,
            p1_2048=r2048['b_p1_comp_pct'],
            p2_2048=r2048['b_p2_union_pct'],
            ffn_2048=r2048['b_ffn_flops_pct'],
            x0_2048=r2048['b_x0_pct'],
            fill_2048=r2048['b_fill_rate'],
            dram_2048=r2048['b_dram_read_pct'],
            tiles_2048=r2048['b_tiles'],
            disp_2048_ms=r2048['b_dispatch_4l_ms'],
            gpu_2048_x=r2048['track_b_x'],
            serial_2048_x=r2048['e2e_serial_b_x'],
        ),
    )

    batch_out = dict(
        version='roster-ch2-v4-a3v4',
        winner=w['name'],
        Ms=w['Ms'],
        gpu=tbf['gpu_name'],
        init_ms=tbf['init_ms_b'],
        accuracy=dict(
            base=91.02,
            pass_line=90.52,
            val_thr=90.96,
            val_acc=w['val_orig'],
            test_orig=round(w['test_orig']['acc'], 2),
            test_b512=round(w['test_a3']['B512']['acc'], 2),
            test_b2048=round(w['test_a3']['B2048']['acc'], 2),
            pct_of_base=w['test_a3']['B512']['pct_of_base'],
            drop_vs_base=round(w['test_orig']['drop'], 2),
            ci=[round(w['test_orig']['lo'], 2), round(w['test_orig']['hi'], 2)],
            verdict=w['test_orig']['verdict'],
            drop_vs_orig=0.0,
            missing_mand_1=0,
            opt_fracs=w['opt_fracs'],
            used_avg_pct=w['test_orig']['used_avg_pct'],
            used_per_layer=w['test_orig']['used_per_layer'],
            val_opt_log=w['val_opt_log'],
        ),
        layer_tables=layer_tables,
        summary=summary,
        hw_bench=hw_bench,
        scale_cmp=dict(n_trials=bsmr['n_trials'], ms_trials=bsmr['ms_trials'], batches=scale_batches),
        codesign=codesign,
        strip_meta=dict(
            W=512, n_strips=8,
            init_ms=tbf['init_ms_b'],
            disp_512=r512['b_dispatch_4l_ms'],
            disp_1024=r1024['b_dispatch_4l_ms'],
            disp_2048=r2048['b_dispatch_4l_ms'],
        ),
    )
    out_p = OUTD / 'batch.json'
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(batch_out, f, ensure_ascii=False, separators=(',', ':'))
    print(f'batch.json saved ({out_p.stat().st_size / 1024:.1f} KB)', flush=True)
    return batch_out


def main():
    t0 = time.perf_counter()
    D = load_data()
    base = load_net(BASE_MODEL_PATH)
    m_r1 = load_net(str(RUNS_B / 'S_M32_B2blk_r1' / 'best.pt'))
    m_r9 = load_net(str(RUNS_B / 'CH2_M32_B1B2_r9_lr1e-06_lb0' / 'best.pt'))

    t0_init = torch.load(str(RUNS_B / 'init_M_32_L123_eps002.pt'), map_location=DEV, weights_only=False)
    C = t0_init['C']
    Ms = [int(c.shape[0]) for c in C]  # [32, 32, 32, 256]
    Gs = [8, 8, 8, 16]

    ch2_json = jload(str(RUNS_B / 'CH2_M32_B1B2.json'))
    acc_json = jload(str(RUNS_B / 'b_10k_accuracy.json'))
    scr_json = jload(str(RUNS_B / 'b_structural_screen.json'))

    # Load all 10 tables (round 0..9)
    tab_paths = [
        RUNS_B / 'init_M_32_L123_eps002.pt',
        RUNS_B / 'lv_S_M32_B2blk_r1.pt',
    ] + [RUNS_B / f'lv_CH2_M32_B1B2_r{rr}.pt' for rr in range(2, 10)]
    tabs = [torch.load(str(p), map_location=DEV, weights_only=False) for p in tab_paths]

    Xtr, Ytr = D['xtr'], D['ytr'].cpu().numpy()
    Xva, Yva = D['xva'], D['yva'].cpu().numpy()
    Xte, Yte = D['xte'], D['yte'].cpu().numpy()
    out = dict(classes=CLASSES, Ms=Ms, Gs=Gs, winner='CH2_M32_B1B2_r9')

    # ---- 1. Pretrain curve (Base s1_seed2) ----
    log_p = Path(BASE_MODEL_PATH).parent / 'log.jsonl'
    pt_rows = [json.loads(x) for x in open(log_p, encoding='utf-8') if '"val_acc"' in x]
    pb_te = evaluate(base, Xte, D['yte'])['probs'].argmax(1).cpu().numpy()
    base_test_acc = round(float((pb_te == Yte).mean() * 100.0), 2)
    out['pretrain'] = dict(
        epoch=[x['epoch'] for x in pt_rows],
        val=[r(x['val_acc'], 2) for x in pt_rows],
        lr=[x['lr'] for x in pt_rows],
        test=base_test_acc,
        val_best=r(max(x['val_acc'] for x in pt_rows), 2),
    )

    # ---- 2. Base model walk -> featured centroids, t-SNE, sprites ----
    print('Base walk for layers & centroids (Ms=%s)...' % Ms, flush=True)
    KJ_all = [torch.full((Ms[l],), NN, device=DEV, dtype=torch.long) for l in range(N_BLOCK)]
    NR_init, KJ_init = tabs[0]['NR'], tabs[0]['KJ']
    kb, db, _ = walk(base, Xtr, C, NR_init, KJ_all)

    feats = []
    for l in range(N_BLOCK):
        k = kb[l].cpu().numpy()
        fl = []
        for cls in FEAT_CLASSES:
            cnt = np.bincount(k[Ytr == cls], minlength=Ms[l])
            for j in np.argsort(-cnt):
                if int(j) not in fl:
                    fl.append(int(j))
                    break
        feats.append(fl)
    keep = {(l, j): True for l in range(N_BLOCK) for j in feats[l]}
    _, _, gb = walk(base, Xtr, C, NR_init, KJ_all, keep=keep)

    rep_imgs, feat_imgs = [], []
    layers = []
    sub = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0))[:20000]
    for l in range(N_BLOCK):
        M_l = Ms[l]
        k = kb[l].cpu().numpy()
        d = db[l].cpu().numpy()
        cnt = np.bincount(k, minlength=M_l)
        rep = []
        for j in range(M_l):
            idx = np.where(k == j)[0]
            rep.append(int(idx[np.argmin(d[idx])]) if len(idx) else -1)
        rep_imgs.append([pix(Xtr[i]) if i >= 0 else np.zeros((28, 28), np.uint8) for i in rep])

        pts = sub[:3000].to(DEV)
        with torch.no_grad():
            h = base.inp(Xtr[pts].reshape(len(pts), -1))
            for ll in range(l):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = h + base.blocks[ll].down(base.blocks[ll].neurons(h)).float()
            f = lkey(h).cpu().numpy()
        print(f'  t-SNE Layer {l + 1} (M={M_l})...', flush=True)
        Y2 = TSNE(n_components=2, init='pca', perplexity=40, random_state=0).fit_transform(f)
        Y2 = (Y2 - Y2.min(0)) / (Y2.max(0) - Y2.min(0))
        pk = k[pts.cpu().numpy()]
        cx = np.full((M_l, 2), np.nan)
        for j in range(M_l):
            m_ = pk == j
            if m_.any():
                cx[j] = Y2[m_].mean(0)

        fl = []
        for j in feats[l]:
            g = gb[(l, j)]
            idx = np.where(k == j)[0]
            near = idx[np.argsort(d[idx])[:8]]
            feat_imgs.extend([pix(Xtr[i]) for i in near] + [np.zeros((28, 28), np.uint8)] * (8 - len(near)))
            S_row = g['S'] / max(1, len(idx))
            ordS = torch.sort(S_row, descending=True).values.cpu().numpy()
            xs = np.unique(np.geomspace(1, NN, 120).round().astype(int)) - 1
            k0_j = int(KJ_init[l][j])
            fl.append(dict(
                j=j, n=int(cnt[j]), K0=k0_j,
                cls=[int(x) for x in np.bincount(Ytr[idx], minlength=10)],
                grid=u8(S_row.cpu().numpy()),
                on=bits((NR_init[l][j] < k0_j).cpu().numpy()),
                sorted=dict(x=[int(i + 1) for i in xs], y=[r(ordS[i] / (ordS[0] + 1e-12), 4) for i in xs]),
            ))
        layers.append(dict(
            M=M_l, G=Gs[l],
            count=[int(x) for x in cnt],
            pts=dict(
                x=[r(v, 3) for v in Y2[:, 0]],
                y=[r(v, 3) for v in Y2[:, 1]],
                key=[int(v) for v in pk],
                cls=[int(v) for v in Ytr[pts.cpu().numpy()]],
            ),
            cen=dict(
                x=[None if np.isnan(v) else r(v, 3) for v in cx[:, 0]],
                y=[None if np.isnan(v) else r(v, 3) for v in cx[:, 1]],
            ),
            feat=fl,
        ))

    out['sprites'] = dict(
        rep=[sprite(rep_imgs[l], 8 if Ms[l] <= 64 else 16, OUTD / f'rep_l{l+1}.png') for l in range(N_BLOCK)],
        feat=sprite(feat_imgs, 8, OUTD / 'feat.png'),
    )
    out['layers'] = layers

    # ---- 3. Rounds 0..9 (10 rounds: 0=Base+init, 1=S_M32_B2blk_r1, 2..9=CH2_M32_B1B2_r2..r9) ----
    print('Extracting rounds 0..9...', flush=True)
    # Round 0 stats on Base + init_M_32_L123_eps002
    p0_va, avg0_va, pl0_va = run_masked(base, Xva, *var_mask(*keys_layer(C), NR_init, KJ_init))
    val0 = round(float((p0_va == D['yva']).float().mean().item() * 100.0), 2)
    p0_te, avg0_te, pl0_te = run_masked(base, Xte, *var_mask(*keys_layer(C), NR_init, KJ_init))
    acc0_te = round(float((p0_te == D['yte']).float().mean().item() * 100.0), 2)
    # Round 0 B=2048 DRAM/x0 quick eval via A3V4Dispatcher
    BM0_list, enc0_list = build_encoded_tables_flex(NR_init, KJ_init, opt_fracs=0.0)
    co0 = t0_init['co']
    pos0 = np.stack([compute_pos_olo_layer(BM0_list[l])[0] for l in range(4)], axis=0)
    enc0_s = np.stack([enc0_list[l][:, co0[l]] for l in range(4)], axis=0)
    disp0 = A3V4Dispatcher(enc0_s, pos0, n_threads=16, force_compile=False)
    keys0_va = extract_keys_split(D, base, C, NR_init, KJ_init, split='va')
    idx0_2048 = np.random.default_rng(0).choice(keys0_va.shape[1], 2048, replace=False)
    kq0 = np.stack([keys0_va[l][idx0_2048] for l in range(4)], axis=0).astype(np.int32)
    disp0.dispatch_4layers(kq0, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1)
    r0_dram = float(np.mean([100.0 * int(disp0.get_layer_active_mask(l, 2048).any(axis=0).sum()) / NN for l in range(4)]))
    r0_ones = sum(int(disp0.get_layer_active_mask(l, 2048).sum()) for l in range(4))
    r0_cells = sum(int(disp0.get_layer_meta(l)['compute_cells']) for l in range(4))
    r0_x0 = round(100.0 * (1.0 - r0_ones / max(r0_cells, 1)), 2)
    disp0.close()

    # Exact walks for Round 0 (Base on all 4096) and Round 1 (Base vs m_r1 on init table)
    _, _, g_base_init = walk(base, Xtr, C, NR_init, KJ_init, keep=keep)
    _, _, g_r1_init = walk(m_r1, Xtr, C, NR_init, KJ_init, keep=keep)

    # Extract R1 eps from structural screen JSON or table cfg
    r1_eps = 0.03
    if isinstance(tabs[1].get('cfg'), dict) and 'eps' in tabs[1]['cfg']:
        r1_eps = float(tabs[1]['cfg']['eps'])
    else:
        for cand in scr_json.get('results', []):
            if cand.get('probe', {}).get('name') == 'S_M32_B2blk' and 'eps' in cand:
                r1_eps = float(cand['eps'])
                break

    rounds = []
    ch2_by_r = {x['round']: x for x in ch2_json['rounds']}

    for rr in range(10):
        NRn, KJn = tabs[rr]['NR'], tabs[rr]['KJ']
        Sn = tabs[rr]['S']
        if rr == 0:
            NRp, KJp, Sp = NR_init, KJ_all, tabs[0]['S']
        else:
            NRp, KJp, Sp = tabs[rr - 1]['NR'], tabs[rr - 1]['KJ'], tabs[rr - 1]['S']

        if rr == 0:
            eps = 0.02
            lr = None
            lam_blk = None
            val_acc = val0
            avg_k = round(avg0_va, 2)
            lyrs = [round(v, 2) for v in pl0_va]
            dram_2048 = round(r0_dram, 1)
            x0_2048 = r0_x0
            ffn_2048 = round(100.0 * r0_cells / (4 * 2048 * NN), 1)
            proxy_x = round(100.0 / max(ffn_2048, 1.0), 2)
            test_acc = acc0_te
        else:
            rec = ch2_by_r[rr]
            ev = rec['eval_val']
            eps = float(rec.get('eps', r1_eps))
            lr = rec.get('lr', 3e-5)
            lam_blk = rec.get('lam_blk', 0.05)
            val_acc = round(ev['val_acc'], 2)
            avg_k = round(ev['avg_k_pct'], 2)
            lyrs = [round(v, 2) for v in ev['used_layers']]
            b2 = ev['batches']['B2048']
            dram_2048 = round(b2['dram_read_pct'], 1)
            x0_2048 = round(b2['x0_pct'], 1)
            ffn_2048 = round(b2['ffn_flops_pct'], 1)
            proxy_x = round(b2['proxy_d4096_x'], 2)
            test_acc = 90.81 if rr == 9 else None

        fr = []
        for l in range(N_BLOCK):
            fl = []
            for j in feats[l]:
                Kold = int(KJp[l][j])
                Knew = int(KJn[l][j])
                if rr == 0:
                    g0 = gb[(l, j)]
                    _, ks, es = err_curve(base.blocks[l], g0, NRp[l][j], Kold)
                    fl.append(dict(
                        j=j, Kold=Kold, Knew=Knew, n=int(g0['A'].shape[0]),
                        before=cum_curve(g0['S'], NRp[l][j], Kold),
                        after=cum_curve(g0['S'], NRp[l][j], Kold),
                        match_before=0.0, match_after=0.0,
                        ek=dict(k=ks, e=[r(e, 5) for e in es]),
                        grid=u8(g0['S'].cpu().numpy()),
                        on_old=bits(np.ones(NN, dtype=np.uint8)),
                        on_new=bits((NRn[l][j] < Knew).cpu().numpy()),
                    ))
                elif rr == 1:
                    g0 = g_base_init[(l, j)]
                    ga = g_r1_init[(l, j)]
                    _, ks, es = err_curve(m_r1.blocks[l], ga, NRp[l][j], Kold)
                    fl.append(dict(
                        j=j, Kold=Kold, Knew=Knew, n=int(ga['A'].shape[0]),
                        before=cum_curve(g0['S'], NRp[l][j], Kold),
                        after=cum_curve(ga['S'], NRp[l][j], Kold),
                        match_before=r(teacher_err(base, l, g0['H'], g0['FULL']), 5),
                        match_after=r(teacher_err(base, l, ga['H'], ga['FULL']), 5),
                        ek=dict(k=ks, e=[r(e, 5) for e in es]),
                        grid=u8(ga['S'].cpu().numpy()),
                        on_old=bits((NRp[l][j] < Kold).cpu().numpy()),
                        on_new=bits((NRn[l][j] < Knew).cpu().numpy()),
                    ))
                else:
                    fl.append(dict(
                        j=j, Kold=Kold, Knew=Knew, n=int(layers[l]['count'][j]),
                        before=cum_curve(Sp[l][j], NRp[l][j], Kold),
                        after=cum_curve(Sn[l][j], NRp[l][j], Kold),
                        grid=u8(Sn[l][j].cpu().numpy()),
                        on_old=bits((NRp[l][j] < Kold).cpu().numpy()),
                        on_new=bits((NRn[l][j] < Knew).cpu().numpy()),
                    ))
            fr.append(fl)

        rounds.append(dict(
            round=rr, lr=lr, lam_blk=lam_blk, eps=eps,
            val=val_acc, avg=avg_k, layers=lyrs,
            dram=dram_2048, x0=x0_2048, ffn=ffn_2048, proxy_x=proxy_x,
            test_acc=test_acc,
            K=[[int(v) for v in KJn[l].cpu().numpy()] for l in range(N_BLOCK)],
            feat=fr,
        ))

    out['rounds'] = rounds
    out['K_init'] = NN

    # ---- 4. 12 real test examples through CH2_M32_B1B2_r9 ----
    print('Tracing 12 real test photos through CH2_M32_B1B2_r9...', flush=True)
    NRf, KJf = tabs[9]['NR'], tabs[9]['KJ']
    pf_t, avg_te_f, pl_te_f = run_masked(m_r9, Xte, *var_mask(*keys_layer(C), NRf, KJf))
    pf = pf_t.cpu().numpy()
    ch2_test_acc = round(float((pf == Yte).mean() * 100.0), 2)
    assert abs(ch2_test_acc - 90.81) < 0.02, f'Expected 90.81%, got {ch2_test_acc}%'

    ex = []
    for cls in list(range(10)) + [7, 8]:
        for i in np.where((Yte == cls) & (pf == Yte) & (pb_te == Yte))[0]:
            if int(i) not in ex:
                ex.append(int(i))
                break

    exs = []
    inf_imgs = []
    with torch.no_grad():
        xe = Xte[ex]
        # Per-round active masks for the 12 examples (r=0: base, r=1: m_r1, r=2..9: m_r9 + exact round table)
        per_round = []
        for rr in range(10):
            Mr = base if rr == 0 else (m_r1 if rr == 1 else m_r9)
            NRr, KJr = tabs[rr]['NR'], tabs[rr]['KJ']
            kr, _, _ = walk(Mr, xe, C, NRr, KJr)
            per_round.append([
                [dict(j=int(kr[l][e]), K=int(KJr[l][kr[l][e]]),
                      on=bits((NRr[l][kr[l][e]] < KJr[l][kr[l][e]]).cpu().numpy())) for l in range(N_BLOCK)]
                for e in range(len(ex))
            ])
        k0 = per_round[0]  # Round 0 init (pure Base + eps=0.02)

        kf_train, df_train, _ = walk(m_r9, Xtr, C, NRf, KJf)
        base_probs = torch.softmax(base(xe)[0].float(), -1).cpu().numpy()
        h = m_r9.inp(xe.reshape(len(ex), -1)).float()
        steps = [[] for _ in ex]
        for l, blk in enumerate(m_r9.blocks):
            with torch.autocast('cuda', enabled=False):
                kf = lkey(h)
                d = (C[l] * C[l]).sum(1)[None, :] - 2 * kf @ C[l].T + (kf * kf).sum(1, keepdim=True)
            j = d.argmin(1)
            on = NRf[l][j] < KJf[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                y = blk.down(a * on.to(a.dtype)).float()
            c = a.float().abs() * blk.colnorm() * on
            z32 = blk.norm(h.float())
            kt = kf_train[l].cpu().numpy()
            dt = df_train[l].cpu().numpy()
            for e in range(len(ex)):
                je = int(j[e])
                idx = np.where(kt == je)[0]
                near = idx[np.argsort(dt[idx])[:6]]
                base_i = len(inf_imgs)
                inf_imgs.extend([pix(Xtr[i]) for i in near] + [np.zeros((28, 28), np.uint8)] * (6 - len(near)))
                zi = int(c[e].argmax())
                gw = blk.gate.weight[zi].float()
                uw = blk.up.weight[zi].float()
                gz = gw * z32[e]
                uz = uw * z32[e]
                gs, us = float(gz.sum()), float(uz.sum())
                sl = gs / (1.0 + math.exp(-gs))
                dd = d[e].cpu().numpy()
                top = np.argsort(dd)[:8]
                sel = torch.where(on[e])[0]
                rank = NRf[l][je][sel]
                sel = sel[torch.argsort(rank)]
                steps[e].append(dict(
                    M=Ms[l],
                    h=q8(h[e].cpu().numpy()), key=q8(kf[e].cpu().numpy()), z=q8(z32[e].detach().cpu().numpy()),
                    dist_top=[dict(j=int(t), d=r(dd[t], 3)) for t in top],
                    j=je, K=int(KJf[l][je]), n_train=int(len(idx)), imgs=base_i,
                    grid=u8(c[e].float().cpu().numpy()), on=bits(on[e].cpu().numpy()),
                    first=[int(v) for v in sel[:12].cpu().numpy()],
                    zoom=dict(i=zi, rank=int(NRf[l][je][zi]), g=r(gs, 4), u=r(us, 4),
                              silu=r(sl, 4), a=r(sl * us, 4),
                              gz=q8(gz.detach().cpu().numpy()), uz=q8(uz.detach().cpu().numpy()),
                              col=r(float(blk.colnorm()[zi]), 4)),
                    y=q8(y[e].cpu().numpy()), yn=r(float(y[e].norm()), 3), hn=r(float(h[e].norm()), 3),
                ))
            h = h + y
        probs = torch.softmax(m_r9.head(m_r9.norm(h)).float(), -1).cpu().numpy()
        for e, i in enumerate(ex):
            exs.append(dict(
                idx=i, label=int(Yte[i]), img=b64(pix(Xte[i]).reshape(-1)),
                probs=[r(v, 4) for v in probs[e]], base_probs=[r(v, 4) for v in base_probs[e]],
                steps=steps[e],
                init=k0[e],
                rounds=[per_round[rr][e] for rr in range(10)],
            ))

    out['sprites']['inf'] = sprite(inf_imgs, 12, OUTD / 'inf.png')
    out['examples'] = exs
    out['final'] = dict(
        test=ch2_test_acc,
        base=base_test_acc,
        avg=round(avg_te_f, 2),
        layers=[round(v, 2) for v in pl_te_f],
        pct_of_base=round(100.0 * ch2_test_acc / base_test_acc, 2),
        ci=[round(91.02 - acc_json['primary']['test_orig']['hi'], 2),
            round(91.02 - acc_json['primary']['test_orig']['lo'], 2)],
        val_thr=90.96,
    )

    viz_p = OUTD / 'viz.json'
    with open(viz_p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print(f'viz.json saved ({viz_p.stat().st_size / 1024:.1f} KB) | CH2 test={ch2_test_acc:.2f}% (K={avg_te_f:.2f}% {pl_te_f}) | Base={base_test_acc:.2f}%', flush=True)

    build_batch_json()
    print(f'All data_roster assets extracted in {time.perf_counter() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
