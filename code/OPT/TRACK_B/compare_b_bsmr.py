"""CH2_M32_B1B2_r9 Actual-Scale 10-Trial Tiling Comparison & Exact 5-Tier Tile Mix.

Computes on the unified Track B Champion (CH2_M32_B1B2_r9, M=[32,32,32,256], 1*=0%, 90.81% test):
  1. Exact 5-tier tile mix & per-layer stats for B = 512, 1,024, 2,048 using the EXACT seed=0
     random batch indices from track_b_final.json (feeds ROSTER_VIZ.html Tab 3 #b3!).
  2. 10-trial random-batch comparison (B = 512, 1,024, 2,048) against:
     - Dense Full Model (64x32, 100%)
     - MatFormer-style Static Front-K (matched budget & 100%-keep)
     - Classic Static 1D SpMM / BCSR (16x16, 32x32, 64x32)
     - State-of-the-Art SDDMM: BSMR (16x16 Global & Strip-512, 32x32, 64x32, 8x16, BSA row-order check)
     - Strongest Fixed Row-Panel Upper Bound: Multi-Scale BSMR (DP over 64/32/16/8 & 32/16)
     - Ours: CH2 + A3V4 5-Tier Roster-Tile (Strip-512, 1*=0% exact, 1.49~2.83 ms dispatch!)
     - Ours (Reference): CH2 + Offline Exact ms_solve 5-Tier (Strip-512, 1*=0%)

Saves to OPT/TRACK_B/runs/compare_bsmr_ch2.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
TRACK_A = PROJ / 'OPT' / 'TRACK_A'
RUNS_B = HERE_B / 'runs'
for p in (str(HERE_B), str(TRACK_A), str(BATCH_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from a3_dispatch_v4 import A3V4Dispatcher  # noqa: E402
from compare_tiling_scale import (  # noqa: E402
    COST_BY_HW,
    aggregate_layer_dicts,
    bsmr_bsa_row_order,
    eval_bsmr_fixed_panel,
    eval_bsmr_multiscale_panel,
    eval_dense,
    eval_matformer,
    eval_static_1d_spmm,
    _worker_roster5_job,
)
from ms_solver import DEFAULT_FRAME_COST_5, FRAME5_AREA, compile_ms_dll  # noqa: E402
from track_b_lib import (  # noqa: E402
    DEV, build_encoded_tables_flex, extract_keys_split, load_data, load_net,
)

NN = 4096
TILE_NAMES = ['64x32', '32x32', '32x16', '16x16', '8x16']
# A3V4 meta['counts'] keys: 0->32x32, 1->32x16, 2->16x16, 3->8x16, 4->64x32
A3_ID_TO_NAME = {4: '64x32', 0: '32x32', 1: '32x16', 2: '16x16', 3: '8x16'}


def summarize_trials(trial_list, B):
    """Average + std across trials (exact same summary schema as compare_tiling_scale.py)."""
    tot_grid = float(4 * B * NN)
    dense_cost = float(4 * (B // 64) * (NN // 32) * COST_BY_HW[(64, 32)])

    def stats(vals):
        a = np.asarray(vals, dtype=np.float64)
        return dict(mean=round(float(a.mean()), 2), std=round(float(a.std(ddof=0)), 2))

    tiles = [t['tiles'] for t in trial_list]
    cells_pct = [100.0 * t['cells'] / tot_grid for t in trial_list]
    act_pct = [100.0 * t['active'] / tot_grid for t in trial_list]
    zero_pct = [100.0 * t['zeros'] / tot_grid for t in trial_list]
    fill_pct = [100.0 * t['active'] / max(t['cells'], 1) for t in trial_list]
    zero_share = [100.0 * t['zeros'] / max(t['cells'], 1) for t in trial_list]
    mand_keep = [100.0 * t['mand_kept'] / max(t['mand_tot'], 1) for t in trial_list]
    cost_pct = [100.0 * t['cost'] / dense_cost for t in trial_list]
    lt25_t = [100.0 * t['tail_lt25_tiles'] / max(t['tiles'], 1) for t in trial_list]
    lt50_t = [100.0 * t['tail_lt50_tiles'] / max(t['tiles'], 1) for t in trial_list]
    lt25_n = [100.0 * t['tail_lt25_nnz'] / max(t['active'], 1) for t in trial_list]
    lt50_n = [100.0 * t['tail_lt50_nnz'] / max(t['active'], 1) for t in trial_list]

    cnt_mean = {}
    for k in TILE_NAMES:
        vs = [t['counts'].get(k, 0) for t in trial_list]
        if any(vs):
            cnt_mean[k] = round(float(np.mean(vs)), 1)

    return dict(
        n_trials=len(trial_list),
        tiles=stats(tiles),
        cells_pct=stats(cells_pct),
        active_pct=stats(act_pct),
        zeros_pct=stats(zero_pct),
        fill_pct=stats(fill_pct),
        zero_share_pct=stats(zero_share),
        mand_keep_pct=stats(mand_keep),
        cost_pct=stats(cost_pct),
        tail_lt25_tile_pct=stats(lt25_t),
        tail_lt50_tile_pct=stats(lt50_t),
        tail_lt25_nnz_pct=stats(lt25_n),
        tail_lt50_nnz_pct=stats(lt50_n),
        counts_mean=cnt_mean,
    )


def eval_a3v4_batch(disp, enc_sorted_b, kq_4l, B):
    """Run A3V4Dispatcher on a 4-layer batch and return aggregate_layer_dicts + per-layer detail."""
    disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1)
    ldicts = []
    per_layer = []
    for l in range(4):
        m = disp.get_layer_meta(l)
        act = disp.get_layer_active_mask(l, B)
        mand = (enc_sorted_b[l][kq_4l[l]] == 1)
        assert act[mand].all(), f'A3V4 missing mandatory 1 on L{l+1} B={B}'
        tot_cells = int(m['compute_cells'])
        ones = int(act.sum())
        mand_tot = int(mand.sum())
        zeros = tot_cells - ones
        cnts = {A3_ID_TO_NAME[k]: int(v) for k, v in m['counts'].items()}
        assert sum(cnts.values()) == int(m['ntiles']), f"count mismatch L{l+1}: {cnts} vs {m['ntiles']}"
        ld = dict(
            tiles=int(m['ntiles']),
            cells=tot_cells,
            active=ones,
            zeros=zeros,
            mand_tot=mand_tot,
            mand_kept=mand_tot,
            opt_tot=0,
            opt_on=0,
            opt_off=0,
            cost=float(m['silicon_cost']),
            counts=cnts,
            tail_lt25_tiles=0,
            tail_lt50_tiles=0,
            tail_lt25_nnz=0,
            tail_lt50_nnz=0,
        )
        ldicts.append(ld)
        per_layer.append(dict(
            layer=l + 1,
            tiles=int(m['ntiles']),
            ncta=int(m['ncta']),
            used_pct=round(100.0 * ones / (B * NN), 2),
            comp_pct=round(100.0 * tot_cells / (B * NN), 2),
            p2_union_pct=round(100.0 * float(m['p2_union_ratio']), 2),
            p2_tm=int(m['p2_uniform_tm']),
            fill_pct=round(100.0 * ones / max(tot_cells, 1), 2),
            x0_pct=round(100.0 * zeros / max(tot_cells, 1), 2),
            dram_read_pct=round(100.0 * int(act.any(axis=0).sum()) / NN, 2),
            counts=cnts,
        ))
    return aggregate_layer_dicts(ldicts), per_layer


def main():
    t0 = time.perf_counter()
    with open(RUNS_B / 'b_10k_accuracy.json', 'r', encoding='utf-8') as f:
        acc_info = json.load(f)
    w_rec = acc_info['primary']
    model_b_path = w_rec['model_path']
    table_b_path = w_rec['table_path']
    opt_fracs_b = w_rec['opt_fracs']  # [0, 0, 0, 0]

    print(f'=== CH2 BSMR 10-Trial Comparison + Exact 5-Tier Tile Mix ({w_rec["name"]}) ===', flush=True)
    D = load_data()
    tb = torch.load(table_b_path, map_location=DEV, weights_only=False)
    C_b, NR_b, KJ_b = tb['C'], tb['NR'], tb['KJ']
    model_b = load_net(model_b_path)
    keys_b = extract_keys_split(D, model_b, C_b, NR_b, KJ_b, split='te')
    del model_b, D
    torch.cuda.empty_cache()

    BM_b_list, enc_b_list = build_encoded_tables_flex(NR_b, KJ_b, opt_fracs=opt_fracs_b)
    BM_b = np.stack(BM_b_list, axis=0)
    ord_b_npz = np.load(str(RUNS_B / 'winner_orders.npz'))
    co_b = [ord_b_npz[f'co_L{l}'] for l in range(4)]
    pos_olo_b = ord_b_npz['pos_olo']
    enc_sorted_b = np.stack([enc_b_list[l][:, co_b[l]] for l in range(4)], axis=0)

    # Empirical column popularity across 10k test set for fair MatFormer static top-K
    col_freq = []
    for l in range(4):
        bm_sorted = BM_b[l][:, co_b[l]]
        cnt_c = np.bincount(keys_b[l], minlength=256).astype(np.float64)
        col_freq.append((cnt_c[:, None] * bm_sorted).sum(axis=0))

    disp = A3V4Dispatcher(enc_sorted_b, pos_olo_b, n_threads=16, force_compile=False)
    compile_ms_dll()

    # 1. Exact seed=0 5-tier tile mix matching track_b_final.json
    rng0 = np.random.default_rng(0)
    q_1024 = rng0.choice(10000, 1024, replace=False)
    q_2048 = rng0.choice(10000, 2048, replace=False)
    q_512 = np.random.default_rng(0).choice(10000, 512, replace=False)
    seed0_map = {512: q_512, 1024: q_1024, 2048: q_2048}

    exact_tile_mix = {}
    for B, idx in seed0_map.items():
        kq_4l = np.stack([keys_b[l][idx] for l in range(4)], axis=0).astype(np.int32)
        agg, per_layer = eval_a3v4_batch(disp, enc_sorted_b, kq_4l, B)
        tot_cnts = {k: int(agg['counts'].get(k, 0)) for k in TILE_NAMES}
        exact_tile_mix[str(B)] = dict(
            B=B,
            tiles_total=int(agg['tiles']),
            counts_total=tot_cnts,
            p1_comp_pct=round(100.0 * agg['cells'] / (4 * B * NN), 2),
            fill_pct=round(100.0 * agg['active'] / max(agg['cells'], 1), 2),
            x0_pct=round(100.0 * agg['zeros'] / max(agg['cells'], 1), 2),
            per_layer=per_layer,
        )
        print(f'  [Seed0 Exact] B={B:4d}: tiles={agg["tiles"]:4d} counts={tot_cnts} '
              f'P1={100*agg["cells"]/(4*B*NN):.2f}% fill={100*agg["active"]/agg["cells"]:.2f}%', flush=True)

    # 2. 10-trial random-batch comparison across B=512, 1024, 2048
    N_TRIALS = 10
    MS_TRIALS = 5  # exact ms_solve offline reference on 5 trials
    out_trials = {}

    for B in (512, 1024, 2048):
        print(f'\n--- Running {N_TRIALS} trials for B={B} ---', flush=True)
        methods = [
            'dense_64x32',
            'matformer_matched',
            'matformer_keep100',
            'static1d_16x16',
            'static1d_32x32',
            'static1d_64x32',
            'bsmr_16x16_global',
            'bsmr_16x16_strip512',
            'bsmr_16x16_bsa_strip512',
            'bsmr_32x32_strip512',
            'bsmr_64x32_strip512',
            'bsmr_8x16_strip512',
            'bsmr_multiscale_cost',
            'our_a3v4_strip512',
            'our_ms_strip512',
        ]
        res = {m: [] for m in methods}
        ms_jobs = []
        ms_meta = {}

        for tr in range(N_TRIALS):
            idx = np.random.default_rng(100 + tr).choice(10000, B, replace=False)
            kq_4l = np.stack([keys_b[l][idx] for l in range(4)], axis=0).astype(np.int32)

            # A3V4
            agg_a3, _ = eval_a3v4_batch(disp, enc_sorted_b, kq_4l, B)
            res['our_a3v4_strip512'].append(agg_a3)

            ld = {m: [] for m in methods if m not in ('our_a3v4_strip512', 'our_ms_strip512')}
            for l in range(4):
                kq = kq_4l[l]
                g = enc_sorted_b[l][kq]  # (B, 4096) in {0, 1}
                row_olo = np.argsort(pos_olo_b[l][kq], kind='stable')
                row_bsa = bsmr_bsa_row_order(BM_b[l][:, co_b[l]], kq)

                ld['dense_64x32'].append(eval_dense(g, 64, 32))
                ld['matformer_matched'].append(eval_matformer(g, col_freq[l], mode='matched'))
                ld['matformer_keep100'].append(eval_matformer(g, col_freq[l], mode='keep100'))
                ld['static1d_16x16'].append(eval_static_1d_spmm(g, row_olo, 16, 16))
                ld['static1d_32x32'].append(eval_static_1d_spmm(g, row_olo, 32, 32))
                ld['static1d_64x32'].append(eval_static_1d_spmm(g, row_olo, 64, 32))
                ld['bsmr_16x16_global'].append(eval_bsmr_fixed_panel(g, row_olo, 16, 16, strip_W=4096))
                ld['bsmr_16x16_strip512'].append(eval_bsmr_fixed_panel(g, row_olo, 16, 16, strip_W=512))
                ld['bsmr_16x16_bsa_strip512'].append(eval_bsmr_fixed_panel(g, row_bsa, 16, 16, strip_W=512))
                ld['bsmr_32x32_strip512'].append(eval_bsmr_fixed_panel(g, row_olo, 32, 32, strip_W=512))
                ld['bsmr_64x32_strip512'].append(eval_bsmr_fixed_panel(g, row_olo, 64, 32, strip_W=512))
                ld['bsmr_8x16_strip512'].append(eval_bsmr_fixed_panel(g, row_olo, 8, 16, strip_W=512))
                ld['bsmr_multiscale_cost'].append(eval_bsmr_multiscale_panel(g, row_olo, strip_W=512, objective='cost'))

                if tr < MS_TRIALS:
                    for s in range(8):
                        sub = g[:, s * 512:(s + 1) * 512]
                        ms_jobs.append(((tr, l, s), kq.astype(np.int32), sub.astype(np.int8), DEFAULT_FRAME_COST_5, 0))

            for m, llist in ld.items():
                res[m].append(aggregate_layer_dicts(llist))

        # Run exact ms_solve reference in parallel
        ms_acc = {(tr, l): dict(counts=[0, 0, 0, 0, 0], mand=0, lt25_t=0, lt50_t=0, lt25_n=0, lt50_n=0)
                  for tr in range(MS_TRIALS) for l in range(4)}
        with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 8)) as ex:
            for jkey, cnts, mand, oon, ooff, l25t, l50t, l25n, l50n in ex.map(_worker_roster5_job, ms_jobs):
                tr, l, _ = jkey
                a = ms_acc[(tr, l)]
                for i in range(5):
                    a['counts'][i] += cnts[i]
                a['mand'] += mand
                a['lt25_t'] += l25t
                a['lt50_t'] += l50t
                a['lt25_n'] += l25n
                a['lt50_n'] += l50n

        areas = [2048, 1024, 512, 256, 128]
        costs = [1680.0, 1024.0, 640.0, 486.0, 410.0]
        for tr in range(MS_TRIALS):
            llist = []
            for l in range(4):
                a = ms_acc[(tr, l)]
                cn = a['counts']
                cells = sum(cn[i] * areas[i] for i in range(5))
                cost = sum(cn[i] * costs[i] for i in range(5))
                llist.append(dict(
                    tiles=sum(cn), cells=cells, active=a['mand'], zeros=cells - a['mand'],
                    mand_tot=a['mand'], mand_kept=a['mand'], opt_tot=0, opt_on=0, opt_off=0,
                    cost=cost, counts={TILE_NAMES[i]: cn[i] for i in range(5)},
                    tail_lt25_tiles=a['lt25_t'], tail_lt50_tiles=a['lt50_t'],
                    tail_lt25_nnz=a['lt25_n'], tail_lt50_nnz=a['lt50_n'],
                ))
            res['our_ms_strip512'].append(aggregate_layer_dicts(llist))

        out_trials[str(B)] = {m: summarize_trials(res[m], B) for m in methods}
        for key_show in ('bsmr_16x16_strip512', 'bsmr_32x32_strip512', 'bsmr_multiscale_cost', 'our_a3v4_strip512', 'our_ms_strip512'):
            s = out_trials[str(B)][key_show]
            print(f'  B={B:4d} {key_show:22s}: tiles={s["tiles"]["mean"]:6.1f} '
                  f'cells={s["cells_pct"]["mean"]:5.2f}% fill={s["fill_pct"]["mean"]:5.1f}% '
                  f'cost={s["cost_pct"]["mean"]:5.2f}%', flush=True)

    disp.close()
    out = dict(
        winner=w_rec['name'],
        Ms=w_rec['Ms'],
        opt_fracs=opt_fracs_b,
        n_trials=N_TRIALS,
        ms_trials=MS_TRIALS,
        exact_tile_mix=exact_tile_mix,
        results=out_trials,
        total_sec=round(time.perf_counter() - t0, 1),
    )
    out_path = RUNS_B / 'compare_bsmr_ch2.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved -> {out_path} ({out["total_sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
