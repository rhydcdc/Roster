"""Actual-Scale Multi-Trial Tiling Comparison (4 Layers x B x 4,096):
Compares tile count, total cells inside tiles, wasted x0 cells, fill rate,
mandatory-1 preservation, and silicon-cost equivalent across:
  0. Dense Full Model (sequential 64x32 baseline, 100%)
  1. Static MatFormer-style (global top-K columns fixed per layer, matched budget & 100%-keep)
  2. Classic Static 1D SpMM / BCSR (global OLO row order + offline col_orders, skip empty tiles)
       - 16x16, 32x32, 64x32
  3. State-of-the-Art SDDMM: BSMR (fixed row panels + per-panel column density sort, fixed-1)
       - Row ordering: OLO centroid-grouped (exact identical centroids contiguous + OLO adjacent)
         plus verification against BSMR paper's BSA weighted-Jaccard clustering!
       - Native 16x16 (Global W=4096 & Strip W=512), 32x32, 64x32, 8x16
       - Multi-Scale Row-Panel BSMR (DP over 64/32/16/8 fixed row panels & 32/16 col widths
         with identical silicon frame_cost — strongest possible fixed-row-panel upper bound!)
       - Also logs BSMR hybrid low-density tail share (tiles < 25% and < 50% fill -> CUDA fallback)
  4. Latest 5-Tier Roster-Tile (64x32 -> 32x32 -> 32x16 -> 16x16 -> 8x16, silicon frame_cost):
       - Official Serving Version: Strip W=512 + 1* (tail 10%)
       - Ablation: Strip W=512 + Fixed-1 (no 1* drop, isolates pure non-grid benefit!)
       - Global Version: W=4096 + 1* (tail 10%)

Runs multiple independent random batches (default 10 trials) for B = 512, 1,024, 2,048
on the actual Lc6 4-layer model (lv_Lc6.pt, keys.npz, col_orders.npz, pos_olo.npy).

Usage:
  python -X utf8 -W ignore BATCHING/CODE/compare_tiling_scale.py [--trials 10] [--global-trials 5]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from batch_s import load_keys, RUNS_S, NN  # noqa: E402
from batch_neon import build_encoded_tables  # noqa: E402
from ms_solver import (  # noqa: E402
    DEFAULT_FRAME_COST_5,
    FRAME5_AREA,
    FRAME5_NAMES,
    FRAME5_ORDER,
    compile_ms_dll,
    get_ms_engine,
)

# Silicon cost map by (H, W) from RTX 4060 measured DEFAULT_FRAME_COST_5:
# [32x32=1024, 32x16=640, 16x16=486, 8x16=410, 64x32=1680]
COST_BY_HW = {
    (64, 32): 1680.0,
    (32, 32): 1024.0,
    (32, 16): 640.0,
    (16, 16): 486.0,
    (8, 16): 410.0,
}


# ---------------------------------------------------------------------------
# Row Ordering Helpers (OLO vs BSMR Paper BSA Weighted-Jaccard Clustering)
# ---------------------------------------------------------------------------
def bsmr_bsa_row_order(bm_layer, kq, tau=16, alpha=0.5):
    """Exact BSMR (Zou et al. 2026, Sec 4.2) BSA block-level weighted-Jaccard clustering
    applied to the unique centroid patterns present in kq (identical centroids have J=1.0).
    """
    uniq_c, inv = np.unique(kq, return_inverse=True)
    # Block-level column encoding of width tau: shape (U, N // tau)
    M_u = bm_layer[uniq_c].reshape(len(uniq_c), NN // tau, tau).sum(axis=2).astype(np.float32)
    norms = M_u.sum(axis=1, keepdims=True)
    E = np.divide(M_u, np.maximum(norms, 1e-8))  # L1-normalized distribution

    # Greedy clustering by weighted Jaccard >= alpha, seed = highest nnz remaining
    nnz_u = norms[:, 0]
    order_seed = np.argsort(-nnz_u)
    unassigned = np.ones(len(uniq_c), dtype=bool)
    c_order = []
    for s in order_seed:
        if not unassigned[s]:
            continue
        rem = np.nonzero(unassigned)[0]
        # Weighted Jaccard: sum(min) / sum(max)
        es = E[s][None, :]
        er = E[rem]
        num = np.minimum(es, er).sum(axis=1)
        den = np.maximum(es, er).sum(axis=1)
        sim = np.divide(num, np.maximum(den, 1e-12))
        sel = rem[sim >= alpha]
        # Order cluster members by descending similarity to seed
        sel = sel[np.argsort(-sim[sim >= alpha])]
        unassigned[sel] = False
        c_order.extend(sel.tolist())

    rank = np.empty(len(uniq_c), dtype=np.int32)
    rank[c_order] = np.arange(len(uniq_c), dtype=np.int32)
    return np.argsort(rank[inv], kind='stable')


# ---------------------------------------------------------------------------
# Fast Vectorized Evaluators for Dense, MatFormer, Static 1D SpMM, BSMR
# ---------------------------------------------------------------------------
def eval_dense(g_enc, H=64, W=32):
    B, N = g_enc.shape
    nt = (B // H) * (N // W)
    tot_cells = nt * H * W
    mand = int((g_enc == 1).sum())
    opt = int((g_enc == 2).sum())
    act = mand + opt
    zeros = tot_cells - act
    cost = nt * COST_BY_HW[(H, W)]
    return dict(
        tiles=nt, cells=tot_cells, active=act, zeros=zeros,
        mand_tot=mand, mand_kept=mand, opt_tot=opt, opt_on=opt, opt_off=0,
        cost=cost, counts={f'{H}x{W}': nt}, tail_lt25_tiles=0, tail_lt50_tiles=0,
        tail_lt25_nnz=0, tail_lt50_nnz=0,
    )


def eval_matformer(g_enc, global_col_freq, mode='matched', H=64, W=32):
    """MatFormer-style global static front-K columns fixed for all tokens in the layer.
    global_col_freq: (N,) popularity across all 256 centroids (weighted by centroid pop or uniform).
    mode='matched': pick K = nearest multiple of W to average active neuron count per row.
    mode='keep100': pick smallest multiple of W covering 100% of mandatory 1s in the batch.
    """
    B, N = g_enc.shape
    col_rank = np.argsort(-global_col_freq)
    mand_mask = (g_enc == 1)
    act_mask = (g_enc > 0)
    mand_tot = int(mand_mask.sum())
    opt_tot = int((g_enc == 2).sum())

    if mode == 'matched':
        avg_k = act_mask.sum() / float(B)
        k_static = int(max(W, round(avg_k / W) * W))
    elif mode == 'keep100':
        # Find furthest mandatory-1 column in global_col_freq order
        any_mand_ranked = mand_mask[:, col_rank].any(axis=0)
        last_idx = int(np.nonzero(any_mand_ranked)[0][-1]) + 1 if any_mand_ranked.any() else 0
        k_static = int(((last_idx + W - 1) // W) * W)
    else:
        raise ValueError(mode)

    sel_cols = col_rank[:k_static]
    sub_mand = mand_mask[:, sel_cols]
    sub_opt = (g_enc[:, sel_cols] == 2)
    mand_kept = int(sub_mand.sum())
    opt_on = int(sub_opt.sum())
    act = mand_kept + opt_on
    nt = (B // H) * (k_static // W)
    tot_cells = nt * H * W
    zeros = tot_cells - act
    cost = nt * COST_BY_HW[(H, W)]
    return dict(
        tiles=nt, cells=tot_cells, active=act, zeros=zeros,
        mand_tot=mand_tot, mand_kept=mand_kept, opt_tot=opt_tot, opt_on=opt_on, opt_off=opt_tot - opt_on,
        cost=cost, counts={f'{H}x{W}': nt}, k_static=k_static,
        tail_lt25_tiles=0, tail_lt50_tiles=0, tail_lt25_nnz=0, tail_lt50_nnz=0,
    )


def eval_static_1d_spmm(g_sorted_enc, row_ord, H, W):
    """Classic Global 1D SpMM / BCSR:
    Fix 1 global row order (row_ord) and 1 global column order (g_sorted_enc already uses col_orders),
    overlay static H x W grid, skip completely empty tiles (fixed-1: keeps all 1 and 1*).
    """
    B, N = g_sorted_enc.shape
    M = (g_sorted_enc[row_ord] > 0)
    mand_tot = int((g_sorted_enc == 1).sum())
    opt_tot = int((g_sorted_enc == 2).sum())
    act_tot = mand_tot + opt_tot

    # Reshape into (B//H, H, N//W, W) and sum active cells per grid tile
    tile_nnz = M.reshape(B // H, H, N // W, W).sum(axis=(1, 3))
    nonempty = tile_nnz > 0
    nt = int(nonempty.sum())
    tot_cells = nt * H * W
    zeros = tot_cells - act_tot
    area = H * W
    nnz_ne = tile_nnz[nonempty]
    lt25 = nnz_ne < (0.25 * area)
    lt50 = nnz_ne < (0.50 * area)
    return dict(
        tiles=nt, cells=tot_cells, active=act_tot, zeros=zeros,
        mand_tot=mand_tot, mand_kept=mand_tot, opt_tot=opt_tot, opt_on=opt_tot, opt_off=0,
        cost=nt * COST_BY_HW[(H, W)], counts={f'{H}x{W}': nt},
        tail_lt25_tiles=int(lt25.sum()), tail_lt50_tiles=int(lt50.sum()),
        tail_lt25_nnz=int(nnz_ne[lt25].sum()), tail_lt50_nnz=int(nnz_ne[lt50].sum()),
    )


def eval_bsmr_fixed_panel(g_sorted_enc, row_ord, R, C, strip_W=4096):
    """BSMR / Fused3S / TC-GNN style Fixed Row-Panel SDDMM (Zou et al. 2026):
    Partition rows (ordered by row_ord) into fixed R-row panels (each row belongs to ONLY 1 panel).
    Inside each panel (and within each strip_W column strip if strip_W < N),
    filter out all-zero columns, sort remaining columns by descending nonzero count,
    and slice into R x C tiles (fixed-1 constraint: all 1 and 1* must be covered).
    """
    B, N = g_sorted_enc.shape
    M = (g_sorted_enc[row_ord] > 0)
    mand_tot = int((g_sorted_enc == 1).sum())
    opt_tot = int((g_sorted_enc == 2).sum())
    act_tot = mand_tot + opt_tot

    n_panels = B // R
    n_strips = N // strip_W
    area = R * C

    nt = 0
    lt25_t = lt50_t = 0
    lt25_nnz = lt50_nnz = 0
    # Vectorized per-strip column counts for all panels: shape (n_panels, N)
    col_cnts = M.reshape(n_panels, R, N).sum(axis=1)  # int counts in [0, R]

    for s in range(n_strips):
        sub = col_cnts[:, s * strip_W:(s + 1) * strip_W]
        for p in range(n_panels):
            nz = sub[p]
            nz = nz[nz > 0]
            u = len(nz)
            if u == 0:
                continue
            nz.sort()
            nz = nz[::-1]  # descending density per BSMR Sec 4.3
            k = (u + C - 1) // C
            nt += k
            # Sum nonzero counts per R x C tile
            if k * C == u:
                t_nnz = nz.reshape(k, C).sum(axis=1)
            else:
                pad = np.zeros(k * C, dtype=nz.dtype)
                pad[:u] = nz
                t_nnz = pad.reshape(k, C).sum(axis=1)
            m25 = t_nnz < (0.25 * area)
            m50 = t_nnz < (0.50 * area)
            lt25_t += int(m25.sum())
            lt50_t += int(m50.sum())
            lt25_nnz += int(t_nnz[m25].sum())
            lt50_nnz += int(t_nnz[m50].sum())

    tot_cells = nt * area
    zeros = tot_cells - act_tot
    return dict(
        tiles=nt, cells=tot_cells, active=act_tot, zeros=zeros,
        mand_tot=mand_tot, mand_kept=mand_tot, opt_tot=opt_tot, opt_on=opt_tot, opt_off=0,
        cost=nt * COST_BY_HW[(R, C)], counts={f'{R}x{C}': nt},
        tail_lt25_tiles=lt25_t, tail_lt50_tiles=lt50_t,
        tail_lt25_nnz=lt25_nnz, tail_lt50_nnz=lt50_nnz,
    )


def eval_bsmr_multiscale_panel(g_sorted_enc, row_ord, strip_W=512, objective='cost'):
    """Strongest-Possible Fixed Row-Panel Upper Bound (Multi-Scale BSMR):
    Rows are still strictly partitioned into disjoint fixed row panels (no cross-panel re-pairing!),
    and 1* cannot be dropped (fixed-1), BUT the solver is allowed to choose the optimal
    panel height R in {64, 32, 16, 8} per 64-row block (hierarchical exact DP) AND optimal
    tile width C in {32, 16} (for R=32) using the EXACT same 5-tier ladder & frame_cost as Roster-Tile!
    Isolates the pure structural advantage of non-grid cross-tile row re-pairing + 1*.
    """
    B, N = g_sorted_enc.shape
    M = (g_sorted_enc[row_ord] > 0)
    mand_tot = int((g_sorted_enc == 1).sum())
    opt_tot = int((g_sorted_enc == 2).sum())
    act_tot = mand_tot + opt_tot
    n_strips = N // strip_W

    def panel_stats(sub_M, R):
        # For a fixed panel of height R across all strips, compute optimal tile count & cost/cells
        cnts = np.zeros(5, dtype=np.int64)  # order: 64x32, 32x32, 32x16, 16x16, 8x16
        tot_cells = 0
        tot_cost = 0.0
        col_sum = sub_M.sum(axis=0)
        for s in range(n_strips):
            u = int((col_sum[s * strip_W:(s + 1) * strip_W] > 0).sum())
            if u == 0:
                continue
            if R == 64:
                k = (u + 31) // 32
                cnts[0] += k
                tot_cells += k * 2048
                tot_cost += k * COST_BY_HW[(64, 32)]
            elif R == 32:
                # Optimal mix of 32x32 (cost 1024, area 1024) and 32x16 (cost 640, area 512)
                # Since 1 x (32x32) costs 1024 < 2 x (32x16) = 1280, use 32x32 for full 32-col chunks,
                # and if remainder in [1..16], use one 32x16 (cost 640 < 1024, area 512 < 1024)!
                k32 = u // 32
                rem = u % 32
                k16 = 0
                if rem > 16:
                    k32 += 1
                elif rem > 0:
                    k16 = 1
                cnts[1] += k32
                cnts[2] += k16
                tot_cells += k32 * 1024 + k16 * 512
                tot_cost += k32 * COST_BY_HW[(32, 32)] + k16 * COST_BY_HW[(32, 16)]
            elif R == 16:
                k = (u + 15) // 16
                cnts[3] += k
                tot_cells += k * 256
                tot_cost += k * COST_BY_HW[(16, 16)]
            elif R == 8:
                k = (u + 15) // 16
                cnts[4] += k
                tot_cells += k * 128
                tot_cost += k * COST_BY_HW[(8, 16)]
        score = tot_cost if objective == 'cost' else float(tot_cells)
        return score, tot_cells, tot_cost, cnts

    tot_cells = 0
    tot_cost = 0.0
    tot_cnts = np.zeros(5, dtype=np.int64)

    # Hierarchical exact DP per 64-row superblock: choose best split tree among {64, 32, 16, 8}
    def best_split(r0, h):
        sc, cl, co, cn = panel_stats(M[r0:r0 + h], h)
        if h == 8:
            return sc, cl, co, cn
        half = h // 2
        sc1, cl1, co1, cn1 = best_split(r0, half)
        sc2, cl2, co2, cn2 = best_split(r0 + half, half)
        if sc1 + sc2 < sc:
            return sc1 + sc2, cl1 + cl2, co1 + co2, cn1 + cn2
        return sc, cl, co, cn

    for r0 in range(0, B, 64):
        _, cl, co, cn = best_split(r0, 64)
        tot_cells += cl
        tot_cost += co
        tot_cnts += cn

    names = ['64x32', '32x32', '32x16', '16x16', '8x16']
    return dict(
        tiles=int(tot_cnts.sum()), cells=int(tot_cells), active=act_tot, zeros=int(tot_cells - act_tot),
        mand_tot=mand_tot, mand_kept=mand_tot, opt_tot=opt_tot, opt_on=opt_tot, opt_off=0,
        cost=float(tot_cost), counts={names[i]: int(tot_cnts[i]) for i in range(5)},
        tail_lt25_tiles=0, tail_lt50_tiles=0, tail_lt25_nnz=0, tail_lt50_nnz=0,
    )


# ---------------------------------------------------------------------------
# Parallel Worker for Latest 5-Tier Roster-Tile (C++ ms_solve_cost_weighted)
# ---------------------------------------------------------------------------
def _worker_roster5_job(args):
    """Run one (trial, layer, strip) of latest 5-tier Roster-Tile."""
    job_key, row_keys, sub_enc, frame_cost, seed = args
    eng = get_ms_engine()
    mand = (sub_enc == 1)
    opt = (sub_enc == 2)
    if not mand.any():
        return (job_key, [0, 0, 0, 0, 0], 0, 0, int(opt.sum()), 0, 0, 0, 0)

    ts = eng.solve_cost(row_keys, sub_enc, frame_cost=frame_cost, chain_same_rows=1, seed=seed)
    covered = np.zeros(sub_enc.shape, dtype=bool)
    counts = [0, 0, 0, 0, 0]  # order: 64x32, 32x32, 32x16, 16x16, 8x16
    t2i = {4: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    lt25_t = lt50_t = lt25_nnz = lt50_nnz = 0

    for ttype, rr, cc in ts:
        sub_cov = covered[np.ix_(rr, cc)]
        sub_act = (sub_enc[np.ix_(rr, cc)] > 0) & (~sub_cov)
        tnnz = int(sub_act.sum())
        area = FRAME5_AREA[ttype]
        if tnnz < 0.25 * area:
            lt25_t += 1
            lt25_nnz += tnnz
        if tnnz < 0.50 * area:
            lt50_t += 1
            lt50_nnz += tnnz
        covered[np.ix_(rr, cc)] = True
        counts[t2i[ttype]] += 1

    assert covered[mand].all(), f'{job_key}: CRITICAL missing mandatory 1!'
    opt_on = int((opt & covered).sum())
    opt_off = int(opt.sum()) - opt_on
    return (job_key, counts, int(mand.sum()), opt_on, opt_off, lt25_t, lt50_t, lt25_nnz, lt50_nnz)


def aggregate_layer_dicts(layer_dicts):
    """Sum 4 layers into one 4-layer batch summary."""
    tot_tiles = sum(d['tiles'] for d in layer_dicts)
    tot_cells = sum(d['cells'] for d in layer_dicts)
    tot_act = sum(d['active'] for d in layer_dicts)
    tot_zeros = sum(d['zeros'] for d in layer_dicts)
    tot_mand = sum(d['mand_tot'] for d in layer_dicts)
    tot_mkept = sum(d['mand_kept'] for d in layer_dicts)
    tot_opt = sum(d['opt_tot'] for d in layer_dicts)
    tot_oon = sum(d['opt_on'] for d in layer_dicts)
    tot_ooff = sum(d['opt_off'] for d in layer_dicts)
    tot_cost = sum(d['cost'] for d in layer_dicts)
    lt25_t = sum(d['tail_lt25_tiles'] for d in layer_dicts)
    lt50_t = sum(d['tail_lt50_tiles'] for d in layer_dicts)
    lt25_n = sum(d['tail_lt25_nnz'] for d in layer_dicts)
    lt50_n = sum(d['tail_lt50_nnz'] for d in layer_dicts)
    cnts = {}
    for d in layer_dicts:
        for k, v in d['counts'].items():
            cnts[k] = cnts.get(k, 0) + v
    return dict(
        tiles=tot_tiles, cells=tot_cells, active=tot_act, zeros=tot_zeros,
        mand_tot=tot_mand, mand_kept=tot_mkept, opt_tot=tot_opt, opt_on=tot_oon, opt_off=tot_ooff,
        cost=tot_cost, counts=cnts,
        tail_lt25_tiles=lt25_t, tail_lt50_tiles=lt50_t,
        tail_lt25_nnz=lt25_n, tail_lt50_nnz=lt50_n,
        per_layer=layer_dicts,
    )


def summarize_trials(trial_list, B):
    """Compute mean & std over trials for a method at batch size B."""
    full_cells = 4 * B * NN
    dense_cost = (4 * (B // 64) * (NN // 32)) * COST_BY_HW[(64, 32)]

    def ms(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

    tiles = [t['tiles'] for t in trial_list]
    cells = [t['cells'] for t in trial_list]
    cells_pct = [100.0 * t['cells'] / full_cells for t in trial_list]
    act = [t['active'] for t in trial_list]
    zeros = [t['zeros'] for t in trial_list]
    zero_pct_of_tiles = [100.0 * t['zeros'] / max(1, t['cells']) for t in trial_list]
    fill_pct = [100.0 * t['active'] / max(1, t['cells']) for t in trial_list]
    mand_keep_pct = [100.0 * t['mand_kept'] / max(1, t['mand_tot']) for t in trial_list]
    opt_off_pct = [100.0 * t['opt_off'] / max(1, t['opt_tot']) for t in trial_list]
    cost_pct = [100.0 * t['cost'] / dense_cost for t in trial_list]
    lt25_tile_pct = [100.0 * t['tail_lt25_tiles'] / max(1, t['tiles']) for t in trial_list]
    lt50_tile_pct = [100.0 * t['tail_lt50_tiles'] / max(1, t['tiles']) for t in trial_list]
    lt25_nnz_pct = [100.0 * t['tail_lt25_nnz'] / max(1, t['active']) for t in trial_list]

    all_keys = sorted({k for t in trial_list for k in t['counts'].keys()})
    cnt_mean = {k: float(np.mean([t['counts'].get(k, 0) for t in trial_list])) for k in all_keys}

    # Per-layer means
    per_layer = []
    for l in range(4):
        fl_c = B * NN
        l_tiles = [t['per_layer'][l]['tiles'] for t in trial_list]
        l_cells = [t['per_layer'][l]['cells'] for t in trial_list]
        l_zeros = [t['per_layer'][l]['zeros'] for t in trial_list]
        l_act = [t['per_layer'][l]['active'] for t in trial_list]
        l_mk = [100.0 * t['per_layer'][l]['mand_kept'] / max(1, t['per_layer'][l]['mand_tot']) for t in trial_list]
        per_layer.append(dict(
            layer=l + 1,
            tiles_mean=float(np.mean(l_tiles)),
            cells_mean=float(np.mean(l_cells)),
            cells_pct=float(100.0 * np.mean(l_cells) / fl_c),
            active_mean=float(np.mean(l_act)),
            zeros_mean=float(np.mean(l_zeros)),
            zero_share_pct=float(100.0 * np.mean([z / max(1, c) for z, c in zip(l_zeros, l_cells)])),
            fill_pct=float(100.0 * np.mean([a / max(1, c) for a, c in zip(l_act, l_cells)])),
            mand_keep_pct=float(np.mean(l_mk)),
        ))

    return dict(
        n_trials=len(trial_list),
        full_cells=full_cells,
        tiles=ms(tiles),
        cells=ms(cells),
        cells_pct=ms(cells_pct),
        active=ms(act),
        zeros=ms(zeros),
        zero_share_pct=ms(zero_pct_of_tiles),
        fill_pct=ms(fill_pct),
        mand_keep_pct=ms(mand_keep_pct),
        opt_off_pct=ms(opt_off_pct),
        silicon_cost_pct=ms(cost_pct),
        tail_lt25_tile_pct=ms(lt25_tile_pct),
        tail_lt50_tile_pct=ms(lt50_tile_pct),
        tail_lt25_nnz_pct=ms(lt25_nnz_pct),
        counts_mean=cnt_mean,
        per_layer=per_layer,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10, help='Number of independent random batches per B')
    ap.add_argument('--global-trials', type=int, default=5, help='Trials for Global W=4096 & Fixed-1 Roster-Tile')
    ap.add_argument('--batches', type=str, default='512,1024,2048', help='Comma-separated batch sizes')
    args = ap.parse_args()

    T0 = time.perf_counter()
    compile_ms_dll()
    keys, BM, _ = load_keys()
    enc_all = build_encoded_tables(BM)
    pos_olo = np.load(os.path.join(RUNS_S, 'pos_olo.npy'))
    co_npz = np.load(os.path.join(RUNS_S, 'col_orders.npz'))
    col_orders = [co_npz[f'L{l}'] for l in range(4)]

    # Build Fixed-1 encoded tables (1* upgraded to mandatory 1) for ablation
    enc_fixed1 = [np.where(enc_all[l] > 0, np.int8(1), np.int8(0)) for l in range(4)]

    # Global neuron popularity across 10,000 test tokens (oracle-best static ranking for MatFormer!)
    global_col_freq = [BM[l][keys[l]].mean(axis=0) for l in range(4)]

    batches = [int(x) for x in args.batches.split(',')]
    n_trials = args.trials
    n_gtrials = min(args.global_trials, n_trials)
    fc = DEFAULT_FRAME_COST_5
    n_workers = min(12, os.cpu_count() or 4)
    f5_names = ['64x32', '32x32', '32x16', '16x16', '8x16']
    f5_areas = [2048, 1024, 512, 256, 128]
    f5_costs = [COST_BY_HW[(64, 32)], COST_BY_HW[(32, 32)], COST_BY_HW[(32, 16)], COST_BY_HW[(16, 16)], COST_BY_HW[(8, 16)]]

    # Pre-sample independent queues per (B, trial)
    queues_by_B = {}
    for B in batches:
        queues_by_B[B] = [np.random.default_rng(t).choice(keys.shape[1], B, replace=False) for t in range(n_trials)]

    # Quick sanity check: verify BSMR OLO row order vs BSMR paper BSA weighted-Jaccard clustering (trial 0)
    idx_chk = queues_by_B[batches[len(batches) // 2]][0]
    B_chk = len(idx_chk)
    print(f'=== [사전 검증] BSMR 행 정렬 비교: OLO 중심점 정렬 vs BSMR 논문 BSA 가중 자카드 군집 (B={B_chk:,}, trial 0) ===', flush=True)
    bsmr_row_check = {}
    for tag, alpha in [('OLO_exact', None), ('BSA_a0.3', 0.3), ('BSA_a0.5', 0.5), ('BSA_a0.7', 0.7)]:
        lds = []
        for l in range(4):
            kq = keys[l][idx_chk]
            g_s = np.ascontiguousarray(enc_all[l][kq][:, col_orders[l]], dtype=np.int8)
            rord = np.argsort(pos_olo[l][kq], kind='stable') if alpha is None else bsmr_bsa_row_order(BM[l], kq, tau=16, alpha=alpha)
            lds.append(eval_bsmr_fixed_panel(g_s, rord, 16, 16, strip_W=4096))
        ag = aggregate_layer_dicts(lds)
        bsmr_row_check[tag] = dict(
            tiles=ag['tiles'], cells=ag['cells'],
            cells_pct=round(100.0 * ag['cells'] / (4 * B_chk * NN), 2),
            zeros=ag['zeros'], fill_pct=round(100.0 * ag['active'] / ag['cells'], 2),
        )
        print(f'  {tag:10s}: 틀 {ag["tiles"]:6d} | 칸 {ag["cells"]:9d} ({100*ag["cells"]/(4*B_chk*NN):5.2f}%) | x0 {ag["zeros"]:9d} (채움 {100*ag["active"]/ag["cells"]:5.1f}%)', flush=True)

    # All results dict: results[f'B{B}'][method_name] = summary
    out = dict(
        model='Lc6 (4 layers x 4,096 neurons, M=256 centroids, lv_Lc6.pt)',
        batches=batches,
        n_trials=n_trials,
        n_global_trials=n_gtrials,
        frame_cost=list(fc),
        bsmr_row_order_check=bsmr_row_check,
        results={},
    )

    for B in batches:
        t_b0 = time.perf_counter()
        print(f'\n==================== B = {B} ({n_trials}회 독립 배치 평균) ====================', flush=True)
        q_list = queues_by_B[B]

        # 1. Fast baselines across all n_trials
        baseline_names = [
            'dense_64x32',
            'matformer_matched',
            'matformer_keep100',
            'static_1d_16x16',
            'static_1d_32x32',
            'static_1d_64x32',
            'bsmr_16x16_bsa_paper',
            'bsmr_16x16_global',
            'bsmr_16x16_strip512',
            'bsmr_32x32_global',
            'bsmr_32x32_strip512',
            'bsmr_64x32_global',
            'bsmr_64x32_strip512',
            'bsmr_8x16_global',
            'bsmr_5tier_panel_strip512_cost',
            'bsmr_5tier_panel_strip512_cells',
            'bsmr_5tier_panel_global_cost',
        ]
        per_method_trials = {nm: [] for nm in baseline_names}

        for t_idx, idx in enumerate(q_list):
            per_m_layers = {nm: [] for nm in baseline_names}
            for l in range(4):
                kq = keys[l][idx]
                g_orig = enc_all[l][kq]
                g_sorted = np.ascontiguousarray(g_orig[:, col_orders[l]], dtype=np.int8)
                rord = np.argsort(pos_olo[l][kq], kind='stable')

                per_m_layers['dense_64x32'].append(eval_dense(g_sorted, 64, 32))
                per_m_layers['matformer_matched'].append(eval_matformer(g_orig, global_col_freq[l], 'matched', 64, 32))
                per_m_layers['matformer_keep100'].append(eval_matformer(g_orig, global_col_freq[l], 'keep100', 64, 32))

                per_m_layers['static_1d_16x16'].append(eval_static_1d_spmm(g_sorted, rord, 16, 16))
                per_m_layers['static_1d_32x32'].append(eval_static_1d_spmm(g_sorted, rord, 32, 32))
                per_m_layers['static_1d_64x32'].append(eval_static_1d_spmm(g_sorted, rord, 64, 32))

                rord_bsa = bsmr_bsa_row_order(BM[l], kq, tau=16, alpha=0.7)
                per_m_layers['bsmr_16x16_bsa_paper'].append(eval_bsmr_fixed_panel(g_sorted, rord_bsa, 16, 16, strip_W=4096))
                per_m_layers['bsmr_16x16_global'].append(eval_bsmr_fixed_panel(g_sorted, rord, 16, 16, strip_W=4096))
                per_m_layers['bsmr_16x16_strip512'].append(eval_bsmr_fixed_panel(g_sorted, rord, 16, 16, strip_W=512))
                per_m_layers['bsmr_32x32_global'].append(eval_bsmr_fixed_panel(g_sorted, rord, 32, 32, strip_W=4096))
                per_m_layers['bsmr_32x32_strip512'].append(eval_bsmr_fixed_panel(g_sorted, rord, 32, 32, strip_W=512))
                per_m_layers['bsmr_64x32_global'].append(eval_bsmr_fixed_panel(g_sorted, rord, 64, 32, strip_W=4096))
                per_m_layers['bsmr_64x32_strip512'].append(eval_bsmr_fixed_panel(g_sorted, rord, 64, 32, strip_W=512))
                per_m_layers['bsmr_8x16_global'].append(eval_bsmr_fixed_panel(g_sorted, rord, 8, 16, strip_W=4096))

                per_m_layers['bsmr_5tier_panel_strip512_cost'].append(eval_bsmr_multiscale_panel(g_sorted, rord, strip_W=512, objective='cost'))
                per_m_layers['bsmr_5tier_panel_strip512_cells'].append(eval_bsmr_multiscale_panel(g_sorted, rord, strip_W=512, objective='cells'))
                per_m_layers['bsmr_5tier_panel_global_cost'].append(eval_bsmr_multiscale_panel(g_sorted, rord, strip_W=4096, objective='cost'))

            for nm in baseline_names:
                per_method_trials[nm].append(aggregate_layer_dicts(per_m_layers[nm]))

        print(f'  [베이스라인 {len(baseline_names)}종 x {n_trials}회 완료: {time.perf_counter() - t_b0:.1f}s]', flush=True)

        # 2. Latest 5-Tier Roster-Tile (parallelized across all (variant, trial, layer, strip) jobs, LPT order!)
        t_r0 = time.perf_counter()
        jobs = []
        # Submit heaviest Global W=4096 jobs FIRST for zero CPU worker wave-tail
        for t_idx in range(n_gtrials):
            idx = q_list[t_idx]
            for l in range(4):
                kq = keys[l][idx]
                g_sorted = np.ascontiguousarray(enc_all[l][kq][:, col_orders[l]], dtype=np.int8)
                jobs.append((('roster5_global', t_idx, l, 0), kq, g_sorted, fc, 42 + 1000 * t_idx + 100 * l))

        # Official Strip W=512 + 1* (n_trials)
        for t_idx in range(n_trials):
            idx = q_list[t_idx]
            for l in range(4):
                kq = keys[l][idx]
                g_sorted = enc_all[l][kq][:, col_orders[l]]
                for s in range(8):
                    sub = np.ascontiguousarray(g_sorted[:, s * 512:(s + 1) * 512], dtype=np.int8)
                    jobs.append((('roster5_strip512', t_idx, l, s), kq, sub, fc, 42 + 1000 * t_idx + 100 * l + s))

        # Ablation: Strip W=512 + Fixed-1 (no 1* drop, n_gtrials)
        for t_idx in range(n_gtrials):
            idx = q_list[t_idx]
            for l in range(4):
                kq = keys[l][idx]
                g_sorted = enc_fixed1[l][kq][:, col_orders[l]]
                for s in range(8):
                    sub = np.ascontiguousarray(g_sorted[:, s * 512:(s + 1) * 512], dtype=np.int8)
                    jobs.append((('roster5_strip512_fixed1', t_idx, l, s), kq, sub, fc, 42 + 1000 * t_idx + 100 * l + s))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            res_list = list(ex.map(_worker_roster5_job, jobs))

        # Group results back into (variant, trial, layer)
        grouped = {}
        for job_key, counts, mand, oon, ooff, lt25_t, lt50_t, lt25_n, lt50_n in res_list:
            var, t_idx, l, s = job_key
            grouped.setdefault((var, t_idx, l), []).append((counts, mand, oon, ooff, lt25_t, lt50_t, lt25_n, lt50_n))

        roster_variants = [
            ('roster5_strip512', n_trials),
            ('roster5_strip512_fixed1', n_gtrials),
            ('roster5_global', n_gtrials),
        ]
        for var, nt_v in roster_variants:
            per_method_trials[var] = []
            for t_idx in range(nt_v):
                lds = []
                for l in range(4):
                    parts = grouped[(var, t_idx, l)]
                    lc = [sum(p[0][i] for p in parts) for i in range(5)]
                    tot_cells = sum(lc[i] * f5_areas[i] for i in range(5))
                    tot_cost = sum(lc[i] * f5_costs[i] for i in range(5))
                    mand = sum(p[1] for p in parts)
                    oon = sum(p[2] for p in parts)
                    ooff = sum(p[3] for p in parts)
                    act = mand + oon
                    lds.append(dict(
                        tiles=sum(lc), cells=tot_cells, active=act, zeros=tot_cells - act,
                        mand_tot=mand, mand_kept=mand, opt_tot=oon + ooff, opt_on=oon, opt_off=ooff,
                        cost=tot_cost, counts={f5_names[i]: lc[i] for i in range(5)},
                        tail_lt25_tiles=sum(p[4] for p in parts),
                        tail_lt50_tiles=sum(p[5] for p in parts),
                        tail_lt25_nnz=sum(p[6] for p in parts),
                        tail_lt50_nnz=sum(p[7] for p in parts),
                    ))
                per_method_trials[var].append(aggregate_layer_dicts(lds))

        print(f'  [최신 5단계 Roster-Tile 3종 완료: {time.perf_counter() - t_r0:.1f}s]', flush=True)

        # Summarize and print concise table
        b_res = {}
        all_names = baseline_names + [v for v, _ in roster_variants]
        for nm in all_names:
            b_res[nm] = summarize_trials(per_method_trials[nm], B)
        out['results'][f'B{B}'] = b_res

        print(f'\n  {"방식":32s} | {"틀 개수":>11s} | {"틀 안 칸 수 (풀대비%)":>22s} | {"필요없는 0 (×0 비중%)":>23s} | {"필수1보존":>7s} | {"비용%":>6s}', flush=True)
        print('  ' + '-' * 116, flush=True)
        for nm in all_names:
            r = b_res[nm]
            print(
                f'  {nm:32s} | {r["tiles"]["mean"]:8.1f}±{r["tiles"]["std"]:3.0f} | '
                f'{r["cells"]["mean"]:10.0f} ({r["cells_pct"]["mean"]:5.2f}±{r["cells_pct"]["std"]:4.2f}%) | '
                f'{r["zeros"]["mean"]:10.0f} ({r["zero_share_pct"]["mean"]:5.1f}%) | '
                f'{r["mand_keep_pct"]["mean"]:6.2f}% | {r["silicon_cost_pct"]["mean"]:5.1f}%',
                flush=True,
            )

    out['total_sec'] = round(time.perf_counter() - T0, 1)
    out_path = os.path.join(RUNS_S, 'compare_tiling_scale.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n저장 완료: {out_path} (총 {out["total_sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
