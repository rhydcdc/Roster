"""Multi-Scale Priority Roster-Tile Python Bridge + Strip-Mining Benchmark + 10,000-Image Accuracy.
Frames (RTX 4060 L2-ridge priority ladder):
  Type 0: 32 x 32 (area 1,024, 100% Tensor Core peak)
  Type 1: 32 x 16 (area   512,  89% peak)
  Type 2: 16 x 16 (area   256,  59% peak)
  Type 3:  8 x 16 (area   128,  36% peak)

Usage:
  python -X utf8 -W ignore BATCHING/CODE/ms_solver.py sweep   # Compare threshold presets on B=256/1024/2048
  python -X utf8 -W ignore BATCHING/CODE/ms_solver.py full    # Run strip+global stats + 10,000-image test accuracy
"""
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
NEON_DIR = PROJ / 'ROSTER_TILE'
for p in (str(HERE), str(NEON_DIR / 'problem2_256x4096'), str(NEON_DIR / 'scale_256x4096'), str(NEON_DIR / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

CPP_PATH = HERE / 'ms_engine.cpp'
DLL_PATH = HERE / 'ms_engine.dll'

FRAME_NAMES = ('32x32', '32x16', '16x16', '8x16')
FRAME_H = (32, 32, 16, 8)
FRAME_W = (32, 16, 16, 16)
FRAME_AREA = (1024, 512, 256, 128)
# Relative RTX 4060 gate+up hardware efficiency (AI / 18 capped at 1.0)
FRAME_EFF_4060 = (1.00, 0.89, 0.59, 0.36)

# 5-tier priority order: Type 4 (64x32) -> Type 0 (32x32) -> Type 1 (32x16) -> Type 2 (16x16) -> Type 3 (8x16)
FRAME5_ORDER = (4, 0, 1, 2, 3)
FRAME5_NAMES = {4: '64x32', 0: '32x32', 1: '32x16', 2: '16x16', 3: '8x16'}
FRAME5_AREA = {4: 2048, 0: 1024, 1: 512, 2: 256, 3: 128}
# Real RTX 4060 silicon cost per tile: [32x32, 32x16, 16x16, 8x16, 64x32]
# Proportional to measured tile latency (area / TFLOPS: 19.6 / 16.1 / 13.5 / 8.8 / 5.3 TF)
DEFAULT_FRAME_COST_5 = (1024.0, 640.0, 486.0, 410.0, 1680.0)

# Default fill thresholds (4-tier legacy): (32x32 >= 50%, 32x16 >= 45%, 16x16 >= 45%, remainder 8x16)
DEFAULT_MIN_FILL = (512, 230, 115)


def compile_ms_dll(force=False):
    if DLL_PATH.exists() and not force:
        if DLL_PATH.stat().st_mtime >= CPP_PATH.stat().st_mtime:
            return
    cmd = [
        'g++', '-O3', '-march=native', '-shared', '-std=c++17',
        '-static', '-static-libgcc', '-static-libstdc++',
        CPP_PATH.name, '-o', DLL_PATH.name,
    ]
    subprocess.run(cmd, cwd=str(HERE), check=True)


class MSEngine:
    def __init__(self):
        compile_ms_dll()
        self.lib = ctypes.CDLL(str(DLL_PATH))
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_dbl_p = ctypes.POINTER(ctypes.c_double)
        self.lib.ms_solve_priority.argtypes = [
            ctypes.c_int, ctypes.c_int, c_int_p, c_i8_p,
            c_int_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            c_int_p, c_int_p, c_int_p, c_int_p, c_int_p,
        ]
        self.lib.ms_solve_priority.restype = ctypes.c_int
        self.lib.ms_solve_priority5.argtypes = [
            ctypes.c_int, ctypes.c_int, c_int_p, c_i8_p,
            c_int_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            c_int_p, c_int_p, c_int_p, c_int_p, c_int_p,
        ]
        self.lib.ms_solve_priority5.restype = ctypes.c_int
        self.lib.ms_solve_cost_weighted.argtypes = [
            ctypes.c_int, ctypes.c_int, c_int_p, c_i8_p,
            c_dbl_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
            c_int_p, c_int_p, c_int_p, c_int_p, c_int_p,
        ]
        self.lib.ms_solve_cost_weighted.restype = ctypes.c_int
        self.lib.ms_solve_row_anchored_strips.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, c_int_p, c_i8_p,
            ctypes.c_int, c_int_p, ctypes.c_double, ctypes.c_uint32, ctypes.c_int,
            c_int_p, c_int_p, c_int_p, c_int_p, c_int_p,
        ]
        self.lib.ms_solve_row_anchored_strips.restype = ctypes.c_int

    def solve_row_anchored(self, row_keys, grid_sorted_enc, strip_w=512, use_64x32=0,
                           min_fill=(1228, 560, 280, 140), chain_rel=0.65, seed=42, max_tiles=65536):
        g = np.ascontiguousarray(grid_sorted_enc, dtype=np.int8)
        rk = np.ascontiguousarray(row_keys, dtype=np.int32)
        mf = np.ascontiguousarray(min_fill, dtype=np.int32)
        m, n = g.shape
        out_type = np.zeros(max_tiles, dtype=np.int32)
        out_nr = np.zeros(max_tiles, dtype=np.int32)
        out_nc = np.zeros(max_tiles, dtype=np.int32)
        out_rows = np.full(max_tiles * 64, -1, dtype=np.int32)
        out_cols = np.full(max_tiles * 32, -1, dtype=np.int32)
        k = self.lib.ms_solve_row_anchored_strips(
            m, n, int(strip_w),
            rk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            int(use_64x32),
            mf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            float(chain_rel), int(seed), int(max_tiles),
            out_type.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        assert k < max_tiles
        r2d = out_rows[:k * 64].reshape(k, 64)
        c2d = out_cols[:k * 32].reshape(k, 32)
        tiles = []
        for i in range(k):
            t = int(out_type[i])
            nr, nc = int(out_nr[i]), int(out_nc[i])
            rr, cc = r2d[i, :nr].copy(), c2d[i, :nc].copy()
            s = int(cc[0]) // strip_w
            tiles.append((s, t, rr, cc))
        return tiles

    def solve_cost(self, row_keys, grid_enc, frame_cost=(1024.0, 640.0, 486.0, 410.0, 0.0), chain_same_rows=1, seed=42, max_tiles=65536):
        g = np.ascontiguousarray(grid_enc, dtype=np.int8)
        rk = np.ascontiguousarray(row_keys, dtype=np.int32)
        fc5 = list(frame_cost) + ([0.0] if len(frame_cost) == 4 else [])
        fc = np.ascontiguousarray(fc5[:5], dtype=np.float64)
        m, n = g.shape
        out_type = np.zeros(max_tiles, dtype=np.int32)
        out_nr = np.zeros(max_tiles, dtype=np.int32)
        out_nc = np.zeros(max_tiles, dtype=np.int32)
        out_rows = np.full(max_tiles * 64, -1, dtype=np.int32)
        out_cols = np.full(max_tiles * 32, -1, dtype=np.int32)
        k = self.lib.ms_solve_cost_weighted(
            m, n,
            rk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            fc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(chain_same_rows), int(seed), int(max_tiles),
            out_type.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        assert k < max_tiles, f'Tile buffer overflow: {k} >= {max_tiles}'
        r2d = out_rows[:k * 64].reshape(k, 64)
        c2d = out_cols[:k * 32].reshape(k, 32)
        return [(int(out_type[i]), r2d[i, :int(out_nr[i])].copy(), c2d[i, :int(out_nc[i])].copy()) for i in range(k)]

    def solve5(self, row_keys, grid_enc, min_fill=(1024, 512, 230, 115), chain_same_rows=1, seed=42, max_tiles=65536):
        g = np.ascontiguousarray(grid_enc, dtype=np.int8)
        rk = np.ascontiguousarray(row_keys, dtype=np.int32)
        mf = np.ascontiguousarray(min_fill, dtype=np.int32)
        m, n = g.shape
        out_type = np.zeros(max_tiles, dtype=np.int32)
        out_nr = np.zeros(max_tiles, dtype=np.int32)
        out_nc = np.zeros(max_tiles, dtype=np.int32)
        out_rows = np.full(max_tiles * 64, -1, dtype=np.int32)
        out_cols = np.full(max_tiles * 32, -1, dtype=np.int32)
        k = self.lib.ms_solve_priority5(
            m, n,
            rk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            mf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(chain_same_rows), int(seed), int(max_tiles),
            out_type.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        assert k < max_tiles, f'Tile buffer overflow: {k} >= {max_tiles}'
        r2d = out_rows[:k * 64].reshape(k, 64)
        c2d = out_cols[:k * 32].reshape(k, 32)
        return [(int(out_type[i]), r2d[i, :int(out_nr[i])].copy(), c2d[i, :int(out_nc[i])].copy()) for i in range(k)]

    def solve(self, row_keys, grid_enc, min_fill=DEFAULT_MIN_FILL, chain_same_rows=1, seed=42, max_tiles=65536):
        g = np.ascontiguousarray(grid_enc, dtype=np.int8)
        rk = np.ascontiguousarray(row_keys, dtype=np.int32)
        mf = np.ascontiguousarray(min_fill, dtype=np.int32)
        m, n = g.shape
        out_type = np.zeros(max_tiles, dtype=np.int32)
        out_nr = np.zeros(max_tiles, dtype=np.int32)
        out_nc = np.zeros(max_tiles, dtype=np.int32)
        out_rows = np.full(max_tiles * 64, -1, dtype=np.int32)
        out_cols = np.full(max_tiles * 32, -1, dtype=np.int32)

        k = self.lib.ms_solve_priority(
            m, n,
            rk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            mf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(chain_same_rows),
            int(seed),
            int(max_tiles),
            out_type.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_nc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        assert k < max_tiles, f'Tile buffer overflow: {k} >= {max_tiles}'
        r2d = out_rows[:k * 64].reshape(k, 64)
        c2d = out_cols[:k * 32].reshape(k, 32)
        tiles = []
        for i in range(k):
            t = int(out_type[i])
            nr = int(out_nr[i])
            nc = int(out_nc[i])
            tiles.append((t, r2d[i, :nr].copy(), c2d[i, :nc].copy()))
        return tiles


_ENG = None


def get_ms_engine():
    global _ENG
    if _ENG is None:
        _ENG = MSEngine()
    return _ENG


def solve_layer_strips(row_keys, grid_sorted_enc, W=512, min_fill=DEFAULT_MIN_FILL, chain_same_rows=1, seed=42, frame_cost=None):
    """Solve one layer split into disjoint W-wide strips (W=4096 -> global, W=512 -> 8 strips).
    If frame_cost is provided (4- or 5-tuple of doubles), uses silicon cost-weighted multi-scale solver.
    Supports both 4-tier (Types 0..3) and 5-tier (Types 4, 0, 1, 2, 3).
    """
    eng = get_ms_engine()
    B, N = grid_sorted_enc.shape
    n_strips = (N + W - 1) // W
    tiles_all = []
    covered = np.zeros((B, N), dtype=bool)
    counts = {t: 0 for t in FRAME5_ORDER}
    cells_by_type = {t: 0 for t in FRAME5_ORDER}
    used_cols = set()

    for s in range(n_strips):
        c0 = s * W
        c1 = min(c0 + W, N)
        sub = np.ascontiguousarray(grid_sorted_enc[:, c0:c1], dtype=np.int8)
        if not (sub == 1).any():
            continue
        if frame_cost is not None:
            ts = eng.solve_cost(row_keys, sub, frame_cost=frame_cost, chain_same_rows=chain_same_rows, seed=seed + s)
        elif len(min_fill) == 4:
            ts = eng.solve5(row_keys, sub, min_fill=min_fill, chain_same_rows=chain_same_rows, seed=seed + s)
        else:
            ts = eng.solve(row_keys, sub, min_fill=min_fill, chain_same_rows=chain_same_rows, seed=seed + s)
        for ttype, rr, cc in ts:
            gcols = cc + c0
            tiles_all.append((s, ttype, rr, gcols))
            covered[np.ix_(rr, gcols)] = True
            counts[ttype] += 1
            cells_by_type[ttype] += FRAME5_AREA[ttype]
            used_cols.update(gcols.tolist())

    mand = (grid_sorted_enc == 1)
    opt = (grid_sorted_enc == 2)
    assert covered[mand].all(), 'CRITICAL: missing mandatory 1!'
    opt_on = opt & covered
    active = mand | opt_on

    comp = sum(cells_by_type.values())
    row_groups_p2 = set()
    for s, ttype, rr, _ in tiles_all:
        row_groups_p2.add((s, tuple(rr.tolist())))

    stats = dict(
        tiles=sum(counts.values()),
        counts={FRAME5_NAMES[t]: counts[t] for t in FRAME5_ORDER},
        cells_by_type={FRAME5_NAMES[t]: cells_by_type[t] for t in FRAME5_ORDER},
        compute=comp,
        compute_pct=100.0 * comp / (B * N),
        mand=int(mand.sum()),
        opt_on=int(opt_on.sum()),
        opt_off=int(opt.sum()) - int(opt_on.sum()),
        fill_rate=100.0 * int(active.sum()) / max(1, comp),
        dram_reads=len(used_cols),
        dram_pct=100.0 * len(used_cols) / N,
        p2_unique_row_groups=len(row_groups_p2),
    )
    return tiles_all, active, stats


def _worker_strip_batch5(args):
    l, s, row_keys, sub_enc, frame_cost, seed = args
    eng = get_ms_engine()
    if not (sub_enc == 1).any():
        return (l, s, [0, 0, 0, 0, 0], 0, 0, int((sub_enc == 2).sum()), 0, 0)
    ts = eng.solve_cost(row_keys, sub_enc, frame_cost=frame_cost, chain_same_rows=1, seed=seed)
    covered = np.zeros(sub_enc.shape, dtype=bool)
    counts = [0, 0, 0, 0, 0]  # order FRAME5_ORDER = (4, 0, 1, 2, 3)
    t2i = {4: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    used_cols = set()
    rsets = set()
    for ttype, rr, cc in ts:
        covered[np.ix_(rr, cc)] = True
        counts[t2i[ttype]] += 1
        used_cols.update(cc.tolist())
        rsets.add(tuple(rr.tolist()))
    mand = (sub_enc == 1)
    opt = (sub_enc == 2)
    assert covered[mand].all(), f'L{l+1} S{s}: missing mandatory 1!'
    opt_on = int((opt & covered).sum())
    return (l, s, counts, int(mand.sum()), opt_on, int(opt.sum()) - opt_on, len(used_cols), len(rsets))


def run_benchmark_config5(B, W, keys, enc_all, col_orders, order_idx, frame_cost, ex):
    from batch_s import NN
    t0 = time.perf_counter()
    n_strips = (NN + W - 1) // W
    jobs = []
    for l in range(4):
        kq = keys[l][order_idx]
        grid_sorted = enc_all[l][kq][:, col_orders[l]]
        for s in range(n_strips):
            c0 = s * W
            c1 = min(c0 + W, NN)
            sub = np.ascontiguousarray(grid_sorted[:, c0:c1], dtype=np.int8)
            jobs.append((l, s, kq, sub, frame_cost, 42 + 100 * l + s))

    res = list(ex.map(_worker_strip_batch5, jobs))
    dt = time.perf_counter() - t0

    f5_names = [FRAME5_NAMES[t] for t in FRAME5_ORDER]
    f5_areas = [FRAME5_AREA[t] for t in FRAME5_ORDER]
    layers = []
    tot_counts = [0, 0, 0, 0, 0]
    tot_comp = tot_mand = tot_on = tot_off = tot_dram = tot_rsets = 0
    for l in range(4):
        lres = [r for r in res if r[0] == l]
        lc = [sum(r[2][i] for r in lres) for i in range(5)]
        lcomp = sum(lc[i] * f5_areas[i] for i in range(5))
        lm = sum(r[3] for r in lres)
        lon = sum(r[4] for r in lres)
        loff = sum(r[5] for r in lres)
        ldram = sum(r[6] for r in lres)
        lrsets = sum(r[7] for r in lres)
        full_c = B * NN
        layers.append(dict(
            layer=l + 1,
            tiles=sum(lc),
            counts={f5_names[i]: lc[i] for i in range(5)},
            cells_by_type={f5_names[i]: lc[i] * f5_areas[i] for i in range(5)},
            compute=lcomp,
            compute_pct=100.0 * lcomp / full_c,
            dram_reads=ldram,
            dram_reads_pct_serving=100.0 * ldram / NN,
            fill_rate=100.0 * (lm + lon) / max(1, lcomp),
            opt_off_rate=100.0 * loff / max(1, lon + loff),
            p2_row_groups=lrsets,
        ))
        for i in range(5):
            tot_counts[i] += lc[i]
        tot_comp += lcomp
        tot_mand += lm; tot_on += lon; tot_off += loff
        tot_dram += ldram; tot_rsets += lrsets

    full_all = 4 * B * NN
    comp_ratio = tot_comp / full_all
    read_ratio = tot_dram / (4 * NN)

    def roofline(ridge):
        return max(float(B), float(ridge)) / max(comp_ratio * B, read_ratio * ridge)

    cells_by_t = {f5_names[i]: tot_counts[i] * f5_areas[i] for i in range(5)}
    pct_by_t = {f5_names[i]: 100.0 * cells_by_t[f5_names[i]] / max(1, tot_comp) for i in range(5)}
    return dict(
        B=B, W=W, n_strips=n_strips,
        frame_cost=list(frame_cost),
        total_tiles=sum(tot_counts),
        counts={f5_names[i]: tot_counts[i] for i in range(5)},
        cells_by_type=cells_by_t,
        cells_share_pct=pct_by_t,
        compute=tot_comp,
        compute_pct=100.0 * comp_ratio,
        dram_reads=tot_dram,
        dram_reads_pct_serving=100.0 * read_ratio,
        fill_rate=100.0 * (tot_mand + tot_on) / max(1, tot_comp),
        opt_off_rate=100.0 * tot_off / max(1, tot_on + tot_off),
        speedup_rtx4060=roofline(104.0),
        speedup_a100=roofline(156.0),
        speedup_h100=roofline(295.0),
        runtime_sec=dt,
        layers=layers,
    )


def _worker_acc_batch5(args):
    b_idx, row_keys, grid_sorted_enc, W, frame_cost, seed = args
    _, active_sorted, stats = solve_layer_strips(row_keys, grid_sorted_enc, W=W, frame_cost=frame_cost, chain_same_rows=1, seed=seed)
    return (b_idx, active_sorted, stats)


def run_accuracy_10k_5tier(col_orders, W=512, batch_size=512, frame_cost=DEFAULT_FRAME_COST_5):
    import torch
    from batch_t import load_data, load_net, MODEL, TABLE, DEV, N_BLOCK, N_NEURON, base_preds, paired, keys_layer, run_masked

    T0 = time.perf_counter()
    D = load_data()
    model = load_net(MODEL)
    model.eval()
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    prep, keyfn = keys_layer(C)
    yte = D['yte'].cpu()
    pb = base_preds(D)['te'].cpu()

    p_orig, u_orig, per_orig = run_masked(
        model, D['xte'], prep, lambda l, h, ctx: NR[l][keyfn(l, h, ctx)] < KJ[l][keyfn(l, h, ctx)][:, None]
    )
    j_orig = paired(pb, p_orig.cpu(), yte)
    print(f'[확인] Lc6 원본(0% 끔): 시험 {j_orig["acc"]:.2f}%, 켜는 뉴런 {u_orig:.2f}% {per_orig}', flush=True)
    assert abs(j_orig['acc'] - 90.68) < 0.02, f'Lc6 sanity mismatch: {j_orig["acc"]}'

    enc_all = []
    inv_orders = []
    for l in range(N_BLOCK):
        on = (NR[l] < KJ[l][:, None]).cpu().numpy()
        k_mand = torch.round(KJ[l].float() * 0.9).to(KJ[l].dtype)
        mand = (NR[l] < k_mand[:, None]).cpu().numpy()
        opt = on & (~mand)
        enc = np.zeros(on.shape, dtype=np.int8)
        enc[mand] = 1
        enc[opt] = 2
        enc_all.append(enc)
        inv = np.empty_like(col_orders[l])
        inv[col_orders[l]] = np.arange(N_NEURON, dtype=np.int32)
        inv_orders.append(inv)

    rng = np.random.default_rng(0)
    N_TEST = D['xte'].shape[0]
    perm = rng.permutation(N_TEST)
    x_perm = D['xte'][torch.from_numpy(perm).to(DEV)]
    slices = [(s, min(s + batch_size, N_TEST)) for s in range(0, N_TEST, batch_size)]
    n_full = sum(1 for s, e in slices if e - s == batch_size)
    n_workers = min(12, os.cpu_count() or 4)
    f5_names = [FRAME5_NAMES[t] for t in FRAME5_ORDER]

    with torch.no_grad():
        h_chunks = [model.inp(x_perm[s:s + 2500].reshape(-1, 784)) for s in range(0, N_TEST, 2500)]
        h = torch.cat(h_chunks, dim=0)

        used_per_layer = []
        layer_stats = []
        avg_comp_sum = 0.0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for l, blk in enumerate(model.blocks):
                tl0 = time.perf_counter()
                j_all = np.concatenate([keyfn(l, h[s:s + 2500], None).cpu().numpy() for s in range(0, N_TEST, 2500)], axis=0)
                jobs = []
                for b_idx, (s, e) in enumerate(slices):
                    rk = j_all[s:e]
                    g_sorted = np.ascontiguousarray(enc_all[l][rk][:, col_orders[l]], dtype=np.int8)
                    jobs.append((b_idx, rk, g_sorted, W, frame_cost, 42 + 1000 * l + b_idx))
                results = list(ex.map(_worker_acc_batch5, jobs))
                results.sort(key=lambda r: r[0])

                active_sorted_all = np.concatenate([r[1] for r in results], axis=0)
                active_orig_all = active_sorted_all[:, inv_orders[l]]
                used_pct = 100.0 * active_orig_all.sum() / (N_TEST * N_NEURON)
                used_per_layer.append(round(float(used_pct), 2))

                full_st = [r[2] for r, (s, e) in zip(results, slices) if e - s == batch_size]
                avg_comp = float(np.mean([st['compute'] for st in full_st]))
                avg_comp_pct = 100.0 * avg_comp / (batch_size * N_NEURON)
                avg_counts = {nm: float(np.mean([st['counts'][nm] for st in full_st])) for nm in f5_names}
                avg_dram = float(np.mean([st['dram_reads'] for st in full_st]))
                avg_comp_sum += avg_comp

                h_next = []
                for s in range(0, N_TEST, 2500):
                    hc = h[s:s + 2500]
                    mc = torch.from_numpy(active_orig_all[s:s + 2500]).to(DEV)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        a = blk.neurons(hc)
                        h_next.append(hc + blk.down(a * mc.to(a.dtype)).float())
                h = torch.cat(h_next, dim=0)

                dt = time.perf_counter() - tl0
                layer_stats.append(dict(
                    layer=l + 1, used_pct=round(float(used_pct), 2),
                    avg_batch_compute_pct=round(avg_comp_pct, 2),
                    avg_batch_counts=avg_counts,
                    avg_batch_dram_reads=round(avg_dram, 1),
                    sec=round(dt, 1),
                ))
                print(
                    f'[10k 정확도 5단계 · 층 {l+1}] 활성 {used_pct:5.2f}% | {batch_size}-배치({n_full}개 평균) 연산 {avg_comp_pct:5.2f}% '
                    f'(64x32:{avg_counts["64x32"]:.0f} 32x32:{avg_counts["32x32"]:.0f} 32x16:{avg_counts["32x16"]:.0f} 16x16:{avg_counts["16x16"]:.0f} 8x16:{avg_counts["8x16"]:.0f}) | {dt:.1f}s',
                    flush=True,
                )

        preds_perm = torch.cat([model.head(model.norm(h[s:s + 2500])).argmax(1).cpu() for s in range(0, N_TEST, 2500)], dim=0)
        preds_orig = torch.empty_like(preds_perm)
        preds_orig[torch.from_numpy(perm)] = preds_perm

    jt = paired(pb, preds_orig, yte)
    j_vs_lc6 = paired(p_orig.cpu(), preds_orig, yte)
    tot_comp_pct = 100.0 * avg_comp_sum / (4 * batch_size * N_NEURON)
    print(f'\n=== 5단계 다중 크기 Roster-Tile (W={W}, B={batch_size}) 시험 10,000장 결과 ({time.perf_counter() - T0:.1f}s) ===', flush=True)
    print(f'시험 정확도: {jt["acc"]:.2f}% (베이스 91.02% 대비 하락 {jt["drop"]:.2f}%p [{jt["lo"]:.2f}, {jt["hi"]:.2f}] -> {jt["verdict"]})', flush=True)
    print(f'원본(묶기전) 대비: 하락 {j_vs_lc6["drop"]:.2f}%p [{j_vs_lc6["lo"]:.2f}, {j_vs_lc6["hi"]:.2f}] (원본만 {j_vs_lc6["base_only"]}, 5단계만 {j_vs_lc6["cand_only"]})', flush=True)
    print(f'켜는 뉴런 평균: {sum(used_per_layer)/N_BLOCK:.2f}% {used_per_layer} | {n_full}개 {batch_size}-배치 평균 연산: {tot_comp_pct:.2f}%', flush=True)

    return dict(
        W=W, batch_size=batch_size, frame_cost=list(frame_cost),
        n_test=N_TEST, n_full_batches=n_full,
        test_acc=jt['acc'],
        pct_of_base=round(100.0 * jt['acc'] / 91.02, 2),
        test_drop_vs_base=jt['drop'],
        test_lo=jt['lo'], test_hi=jt['hi'], test_verdict=jt['verdict'],
        vs_lc6=j_vs_lc6,
        used_avg_pct=round(sum(used_per_layer) / N_BLOCK, 2),
        used_per_layer=used_per_layer,
        avg_batch_compute=avg_comp_sum,
        avg_batch_compute_pct_full=round(tot_comp_pct, 2),
        layers=layer_stats,
        total_sec=round(time.perf_counter() - T0, 1),
    )


def _worker_strip_batch(args):
    l, s, row_keys, sub_enc, min_fill, seed = args
    eng = get_ms_engine()
    if not (sub_enc == 1).any():
        return (l, s, [0, 0, 0, 0], 0, 0, int((sub_enc == 2).sum()), 0, 0)
    ts = eng.solve(row_keys, sub_enc, min_fill=min_fill, chain_same_rows=1, seed=seed)
    covered = np.zeros(sub_enc.shape, dtype=bool)
    counts = [0, 0, 0, 0]
    used_cols = set()
    rsets = set()
    for ttype, rr, cc in ts:
        covered[np.ix_(rr, cc)] = True
        counts[ttype] += 1
        used_cols.update(cc.tolist())
        rsets.add(tuple(rr.tolist()))
    mand = (sub_enc == 1)
    opt = (sub_enc == 2)
    assert covered[mand].all(), f'L{l+1} S{s}: missing mandatory 1!'
    opt_on = int((opt & covered).sum())
    return (l, s, counts, int(mand.sum()), opt_on, int(opt.sum()) - opt_on, len(used_cols), len(rsets))


def run_benchmark_config(B, W, keys, enc_all, col_orders, order_idx, min_fill, ex):
    from batch_s import NN
    t0 = time.perf_counter()
    n_strips = (NN + W - 1) // W
    jobs = []
    for l in range(4):
        kq = keys[l][order_idx]
        grid_sorted = enc_all[l][kq][:, col_orders[l]]
        for s in range(n_strips):
            c0 = s * W
            c1 = min(c0 + W, NN)
            sub = np.ascontiguousarray(grid_sorted[:, c0:c1], dtype=np.int8)
            jobs.append((l, s, kq, sub, min_fill, 42 + 100 * l + s))

    res = list(ex.map(_worker_strip_batch, jobs))
    dt = time.perf_counter() - t0

    layers = []
    tot_counts = [0, 0, 0, 0]
    tot_comp = tot_eff = tot_mand = tot_on = tot_off = tot_dram = tot_rsets = 0
    for l in range(4):
        lres = [r for r in res if r[0] == l]
        lc = [sum(r[2][i] for r in lres) for i in range(4)]
        lcomp = sum(lc[i] * FRAME_AREA[i] for i in range(4))
        leff = sum(lc[i] * FRAME_AREA[i] / FRAME_EFF_4060[i] for i in range(4))
        lm = sum(r[3] for r in lres)
        lon = sum(r[4] for r in lres)
        loff = sum(r[5] for r in lres)
        ldram = sum(r[6] for r in lres)
        lrsets = sum(r[7] for r in lres)
        full_c = B * NN
        layers.append(dict(
            layer=l + 1,
            tiles=sum(lc),
            counts={FRAME_NAMES[i]: lc[i] for i in range(4)},
            cells_by_type={FRAME_NAMES[i]: lc[i] * FRAME_AREA[i] for i in range(4)},
            compute=lcomp,
            compute_pct=100.0 * lcomp / full_c,
            eff_work_4060_pct=100.0 * leff / full_c,
            dram_reads=ldram,
            dram_reads_pct_serving=100.0 * ldram / NN,
            fill_rate=100.0 * (lm + lon) / max(1, lcomp),
            opt_off_rate=100.0 * loff / max(1, lon + loff),
            p2_row_groups=lrsets,
        ))
        for i in range(4):
            tot_counts[i] += lc[i]
        tot_comp += lcomp; tot_eff += leff
        tot_mand += lm; tot_on += lon; tot_off += loff
        tot_dram += ldram; tot_rsets += lrsets

    full_all = 4 * B * NN
    comp_ratio = tot_comp / full_all
    read_ratio = tot_dram / (4 * NN)

    def roofline(ridge):
        return max(float(B), float(ridge)) / max(comp_ratio * B, read_ratio * ridge)

    cells_by_t = {FRAME_NAMES[i]: tot_counts[i] * FRAME_AREA[i] for i in range(4)}
    pct_by_t = {FRAME_NAMES[i]: 100.0 * cells_by_t[FRAME_NAMES[i]] / max(1, tot_comp) for i in range(4)}
    out = dict(
        B=B, W=W, n_strips=n_strips,
        min_fill=list(min_fill),
        total_tiles=sum(tot_counts),
        counts={FRAME_NAMES[i]: tot_counts[i] for i in range(4)},
        cells_by_type=cells_by_t,
        cells_share_pct=pct_by_t,
        compute=tot_comp,
        compute_pct=100.0 * comp_ratio,
        eff_work_4060_pct=100.0 * tot_eff / full_all,
        dram_reads=tot_dram,
        dram_reads_pct_serving=100.0 * read_ratio,
        fill_rate=100.0 * (tot_mand + tot_on) / max(1, tot_comp),
        opt_off_rate=100.0 * tot_off / max(1, tot_on + tot_off),
        speedup_rtx4060=roofline(104.0),
        speedup_a100=roofline(156.0),
        speedup_h100=roofline(295.0),
        runtime_sec=dt,
        layers=layers,
    )
    return out


def _worker_acc_batch(args):
    b_idx, row_keys, grid_sorted_enc, W, min_fill, seed = args
    _, active_sorted, stats = solve_layer_strips(row_keys, grid_sorted_enc, W=W, min_fill=min_fill, chain_same_rows=1, seed=seed)
    return (b_idx, active_sorted, stats)


def run_accuracy_10k(col_orders, W=512, min_fill=DEFAULT_MIN_FILL):
    import torch
    from batch_t import load_data, load_net, MODEL, TABLE, DEV, N_BLOCK, N_NEURON, base_preds, paired, keys_layer, run_masked

    T0 = time.perf_counter()
    D = load_data()
    model = load_net(MODEL)
    model.eval()
    t = torch.load(TABLE, map_location=DEV)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    prep, keyfn = keys_layer(C)
    yte = D['yte'].cpu()
    pb = base_preds(D)['te'].cpu()

    # Lc6 sanity check (must be 90.68%)
    p_orig, u_orig, per_orig = run_masked(
        model, D['xte'], prep, lambda l, h, ctx: NR[l][keyfn(l, h, ctx)] < KJ[l][keyfn(l, h, ctx)][:, None]
    )
    j_orig = paired(pb, p_orig.cpu(), yte)
    print(f'[확인] Lc6 원본(0% 끔): 시험 {j_orig["acc"]:.2f}%, 켜는 뉴런 {u_orig:.2f}% {per_orig}', flush=True)
    assert abs(j_orig['acc'] - 90.68) < 0.02, f'Lc6 sanity mismatch: {j_orig["acc"]}'

    enc_all = []
    inv_orders = []
    for l in range(N_BLOCK):
        on = (NR[l] < KJ[l][:, None]).cpu().numpy()
        k_mand = torch.round(KJ[l].float() * 0.9).to(KJ[l].dtype)
        mand = (NR[l] < k_mand[:, None]).cpu().numpy()
        opt = on & (~mand)
        enc = np.zeros(on.shape, dtype=np.int8)
        enc[mand] = 1
        enc[opt] = 2
        enc_all.append(enc)
        inv = np.empty_like(col_orders[l])
        inv[col_orders[l]] = np.arange(N_NEURON, dtype=np.int32)
        inv_orders.append(inv)

    rng = np.random.default_rng(0)
    N_TEST = D['xte'].shape[0]
    perm = rng.permutation(N_TEST)
    x_perm = D['xte'][torch.from_numpy(perm).to(DEV)]
    slices = [(s, min(s + 256, N_TEST)) for s in range(0, N_TEST, 256)]
    n_workers = min(12, os.cpu_count() or 4)

    with torch.no_grad():
        h_chunks = [model.inp(x_perm[s:s + 2500].reshape(-1, 784)) for s in range(0, N_TEST, 2500)]
        h = torch.cat(h_chunks, dim=0)

        used_per_layer = []
        layer_stats = []
        avg256_comp_sum = 0.0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for l, blk in enumerate(model.blocks):
                tl0 = time.perf_counter()
                j_all = np.concatenate([keyfn(l, h[s:s + 2500], None).cpu().numpy() for s in range(0, N_TEST, 2500)], axis=0)
                jobs = []
                for b_idx, (s, e) in enumerate(slices):
                    rk = j_all[s:e]
                    g_sorted = np.ascontiguousarray(enc_all[l][rk][:, col_orders[l]], dtype=np.int8)
                    jobs.append((b_idx, rk, g_sorted, W, min_fill, 42 + 1000 * l + b_idx))
                results = list(ex.map(_worker_acc_batch, jobs))
                results.sort(key=lambda r: r[0])

                # Map active mask back from sorted column order to original model neuron order!
                active_sorted_all = np.concatenate([r[1] for r in results], axis=0)
                active_orig_all = active_sorted_all[:, inv_orders[l]]
                used_pct = 100.0 * active_orig_all.sum() / (N_TEST * N_NEURON)
                used_per_layer.append(round(float(used_pct), 2))

                full_st = [r[2] for r, (s, e) in zip(results, slices) if e - s == 256]
                avg_comp = float(np.mean([st['compute'] for st in full_st]))
                avg_comp_pct = 100.0 * avg_comp / (256 * N_NEURON)
                avg_counts = {nm: float(np.mean([st['counts'][nm] for st in full_st])) for nm in FRAME_NAMES}
                avg_dram = float(np.mean([st['dram_reads'] for st in full_st]))
                avg256_comp_sum += avg_comp

                h_next = []
                for s in range(0, N_TEST, 2500):
                    hc = h[s:s + 2500]
                    mc = torch.from_numpy(active_orig_all[s:s + 2500]).to(DEV)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        a = blk.neurons(hc)
                        h_next.append(hc + blk.down(a * mc.to(a.dtype)).float())
                h = torch.cat(h_next, dim=0)

                dt = time.perf_counter() - tl0
                layer_stats.append(dict(
                    layer=l + 1, used_pct=round(float(used_pct), 2),
                    avg256_compute_pct=round(avg_comp_pct, 2),
                    avg256_counts=avg_counts,
                    avg256_dram_reads=round(avg_dram, 1),
                    sec=round(dt, 1),
                ))
                print(
                    f'[10k 정확도 · 층 {l+1}] 활성 {used_pct:5.2f}% | 256-배치(39개 평균) 연산 {avg_comp_pct:5.2f}% '
                    f'(32x32:{avg_counts["32x32"]:.0f} 32x16:{avg_counts["32x16"]:.0f} 16x16:{avg_counts["16x16"]:.0f} 8x16:{avg_counts["8x16"]:.0f}) | {dt:.1f}s',
                    flush=True,
                )

        preds_perm = torch.cat([model.head(model.norm(h[s:s + 2500])).argmax(1).cpu() for s in range(0, N_TEST, 2500)], dim=0)
        preds_orig = torch.empty_like(preds_perm)
        preds_orig[torch.from_numpy(perm)] = preds_perm

    jt = paired(pb, preds_orig, yte)
    j_vs_lc6 = paired(p_orig.cpu(), preds_orig, yte)
    tot_comp_pct = 100.0 * avg256_comp_sum / (4 * 256 * N_NEURON)
    print(f'\n=== 다중 크기 Roster-Tile (W={W}) 시험 10,000장 결과 ({time.perf_counter() - T0:.1f}s) ===', flush=True)
    print(f'시험 정확도: {jt["acc"]:.2f}% (베이스 91.02% 대비 하락 {jt["drop"]:.2f}%p [{jt["lo"]:.2f}, {jt["hi"]:.2f}] -> {jt["verdict"]})', flush=True)
    print(f'Lc6 원본 대비: 하락 {j_vs_lc6["drop"]:.2f}%p [{j_vs_lc6["lo"]:.2f}, {j_vs_lc6["hi"]:.2f}] (Lc6만 맞힘 {j_vs_lc6["base_only"]}, 다중크기만 맞힘 {j_vs_lc6["cand_only"]})', flush=True)
    print(f'켜는 뉴런 평균: {sum(used_per_layer)/N_BLOCK:.2f}% {used_per_layer} | 39개 256-배치 평균 연산: {tot_comp_pct:.2f}%', flush=True)

    return dict(
        W=W, min_fill=list(min_fill),
        n_test=N_TEST, n_full_256_batches=39,
        test_acc=jt['acc'],
        pct_of_base=round(100.0 * jt['acc'] / 91.02, 2),
        test_drop_vs_base=jt['drop'],
        test_lo=jt['lo'], test_hi=jt['hi'], test_verdict=jt['verdict'],
        vs_lc6=j_vs_lc6,
        used_avg_pct=round(sum(used_per_layer) / N_BLOCK, 2),
        used_per_layer=used_per_layer,
        avg256_compute=avg256_comp_sum,
        avg256_compute_pct_full=round(tot_comp_pct, 2),
        layers=layer_stats,
        total_sec=round(time.perf_counter() - T0, 1),
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sweep'
    compile_ms_dll()
    from batch_s import load_keys, queues, RUNS_S
    from batch_neon import build_encoded_tables
    from batch_strip import compute_offline_col_orders

    keys, BM, _ = load_keys()
    enc_all = build_encoded_tables(BM)
    col_orders = compute_offline_col_orders(enc_all)

    # Save offline col_orders so CUDA kernel uses the exact same physical weight permutation!
    np.savez(os.path.join(RUNS_S, 'col_orders.npz'), **{f'L{l}': col_orders[l] for l in range(4)})

    q0_256 = next(o for N, t, o in queues(keys.shape[1]) if N == 256 and t == 0)
    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)

    n_workers = min(12, os.cpu_count() or 4)
    if mode == 'sweep':
        presets = [
            ('P_hi_70', (717, 358, 179)),   # 70 / 70 / 70%
            ('P_bal_60', (614, 307, 154)),  # 60 / 60 / 60%
            ('P_gpu_50', (512, 282, 141)),  # 50 / 55 / 55%
            ('P_agg_42', (430, 230, 128)),  # 42 / 45 / 50%
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for name, mf in presets:
                print(f'\n--- Preset {name} min_fill={mf} (W=512) ---', flush=True)
                for B, idx in [(256, q0_256), (1024, q0_1024), (2048, q0_2048)]:
                    r = run_benchmark_config(B, 512, keys, enc_all, col_orders, idx, mf, ex)
                    sh = r['cells_share_pct']
                    print(
                        f'  B={B:4d}: 연산 {r["compute_pct"]:5.2f}% | 4060환산일 {r["eff_work_4060_pct"]:5.2f}% (예상 {100/r["eff_work_4060_pct"]:.2f}x) | '
                        f'채움 {r["fill_rate"]:5.1f}% | 비중 32x32:{sh["32x32"]:4.1f}% 32x16:{sh["32x16"]:4.1f}% 16x16:{sh["16x16"]:4.1f}% 8x16:{sh["8x16"]:4.1f}% | '
                        f'틀 {r["total_tiles"]:5d} | {r["runtime_sec"]:.2f}s',
                        flush=True,
                    )
    elif mode == 'full':
        mf = DEFAULT_MIN_FILL
        if len(sys.argv) >= 5:
            mf = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        print(f'=== Full Multi-Scale Roster-Tile Evaluation (min_fill={mf}) ===', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            res_strip = []
            res_global = []
            for B, idx in [(256, q0_256), (1024, q0_1024), (2048, q0_2048)]:
                rs = run_benchmark_config(B, 512, keys, enc_all, col_orders, idx, mf, ex)
                res_strip.append(rs)
                sh = rs['cells_share_pct']
                print(
                    f'[Strip W=512 B={B:4d}] 연산 {rs["compute_pct"]:5.2f}% | 채움 {rs["fill_rate"]:5.1f}% | '
                    f'비중 32x32:{sh["32x32"]:4.1f}% 32x16:{sh["32x16"]:4.1f}% 16x16:{sh["16x16"]:4.1f}% 8x16:{sh["8x16"]:4.1f}% | '
                    f'DRAM {rs["dram_reads_pct_serving"]:5.2f}% | Roof 4060 {rs["speedup_rtx4060"]:.2f}x A100 {rs["speedup_a100"]:.2f}x H100 {rs["speedup_h100"]:.2f}x | {rs["runtime_sec"]:.2f}s',
                    flush=True,
                )
            for B, idx in [(256, q0_256), (1024, q0_1024), (2048, q0_2048)]:
                rg = run_benchmark_config(B, 4096, keys, enc_all, col_orders, idx, mf, ex)
                res_global.append(rg)
                sh = rg['cells_share_pct']
                print(
                    f'[Global      B={B:4d}] 연산 {rg["compute_pct"]:5.2f}% | 채움 {rg["fill_rate"]:5.1f}% | '
                    f'비중 32x32:{sh["32x32"]:4.1f}% 32x16:{sh["32x16"]:4.1f}% 16x16:{sh["16x16"]:4.1f}% 8x16:{sh["8x16"]:4.1f}% | '
                    f'DRAM {rg["dram_reads_pct_serving"]:5.2f}% | {rg["runtime_sec"]:.2f}s',
                    flush=True,
                )

        acc_strip = run_accuracy_10k(col_orders, W=512, min_fill=mf)
        out_path = os.path.join(RUNS_S, 'ms_roster_stats.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dict(min_fill=list(mf), strip_W512=res_strip, global_W4096=res_global, accuracy_10k=acc_strip),
                      f, ensure_ascii=False, indent=2)
        print(f'\n저장 완료: {out_path}', flush=True)
    elif mode == 'full5':
        fc = DEFAULT_FRAME_COST_5
        q0_512 = np.random.default_rng(0).choice(keys.shape[1], 512, replace=False)
        print(f'=== Full 5-Tier Multi-Scale Roster-Tile Evaluation (frame_cost={fc}, B=512/1024/2048) ===', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            res_strip = []
            res_global = []
            for B, idx in [(512, q0_512), (1024, q0_1024), (2048, q0_2048)]:
                rs = run_benchmark_config5(B, 512, keys, enc_all, col_orders, idx, fc, ex)
                res_strip.append(rs)
                sh = rs['cells_share_pct']
                print(
                    f'[Strip W=512 B={B:4d}] 연산 {rs["compute_pct"]:5.2f}% | 채움 {rs["fill_rate"]:5.1f}% | '
                    f'비중 64x32:{sh["64x32"]:4.1f}% 32x32:{sh["32x32"]:4.1f}% 32x16:{sh["32x16"]:4.1f}% 16x16:{sh["16x16"]:4.1f}% 8x16:{sh["8x16"]:4.1f}% | '
                    f'DRAM {rs["dram_reads_pct_serving"]:5.2f}% | Roof 4060 {rs["speedup_rtx4060"]:.2f}x A100 {rs["speedup_a100"]:.2f}x H100 {rs["speedup_h100"]:.2f}x | {rs["runtime_sec"]:.2f}s',
                    flush=True,
                )
            for B, idx in [(512, q0_512), (1024, q0_1024), (2048, q0_2048)]:
                rg = run_benchmark_config5(B, 4096, keys, enc_all, col_orders, idx, fc, ex)
                res_global.append(rg)
                sh = rg['cells_share_pct']
                print(
                    f'[Global      B={B:4d}] 연산 {rg["compute_pct"]:5.2f}% | 채움 {rg["fill_rate"]:5.1f}% | '
                    f'비중 64x32:{sh["64x32"]:4.1f}% 32x32:{sh["32x32"]:4.1f}% 32x16:{sh["32x16"]:4.1f}% 16x16:{sh["16x16"]:4.1f}% 8x16:{sh["8x16"]:4.1f}% | '
                    f'DRAM {rg["dram_reads_pct_serving"]:5.2f}% | {rg["runtime_sec"]:.2f}s',
                    flush=True,
                )

        acc_strip = run_accuracy_10k_5tier(col_orders, W=512, batch_size=512, frame_cost=fc)
        out_path = os.path.join(RUNS_S, 'ms_roster5_stats.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dict(frame_cost=list(fc), frame_ladder=['64x32', '32x32', '32x16', '16x16', '8x16'],
                           strip_W512=res_strip, global_W4096=res_global, accuracy_10k=acc_strip),
                      f, ensure_ascii=False, indent=2)
        print(f'\n저장 완료: {out_path}', flush=True)


if __name__ == '__main__':
    main()
