"""Python Bridge + Production Benchmark for A3 V4 AVX2 Zero-Branch Dispatcher."""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
if str(BATCH_CODE) not in sys.path:
    sys.path.insert(0, str(BATCH_CODE))

CPP_PATH = HERE / 'a3_dispatch_v4.cpp'
DLL_PATH = HERE / 'a3_dispatch_v4.dll'


def compile_a3v4_dll(force=False):
    if DLL_PATH.exists() and not force:
        if DLL_PATH.stat().st_mtime >= CPP_PATH.stat().st_mtime:
            return
    cmd = [
        'g++', '-O3', '-march=native', '-mavx2', '-mbmi2', '-mpopcnt',
        '-shared', '-std=c++17', '-pthread',
        '-static', '-static-libgcc', '-static-libstdc++',
        CPP_PATH.name, '-o', DLL_PATH.name,
    ]
    subprocess.run(cmd, cwd=str(HERE), check=True)


class A3V4Dispatcher:
    def __init__(self, enc_sorted_4l, pos_olo_4l, n_threads=None, force_compile=False):
        compile_a3v4_dll(force=force_compile)
        if n_threads is None:
            n_threads = min(16, os.cpu_count() or 8)
        self.n_threads = int(n_threads)
        self.lib = ctypes.CDLL(str(DLL_PATH))
        c_i8_p = ctypes.POINTER(ctypes.c_int8)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_u32_p = ctypes.POINTER(ctypes.c_uint32)
        c_dbl_p = ctypes.POINTER(ctypes.c_double)

        self.lib.a3v4_create.argtypes = [c_i8_p, c_i32_p, ctypes.c_int]
        self.lib.a3v4_create.restype = ctypes.c_void_p
        self.lib.a3v4_destroy.argtypes = [ctypes.c_void_p]
        self.lib.a3v4_destroy.restype = None
        self.lib.a3v4_dispatch_4layers.argtypes = [
            ctypes.c_void_p, ctypes.c_int, c_i32_p, c_dbl_p,
            ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, c_dbl_p,
        ]
        self.lib.a3v4_dispatch_4layers.restype = ctypes.c_double
        self.lib.a3v4_dispatch_1layer.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, c_i32_p, c_dbl_p,
            ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, c_dbl_p,
        ]
        self.lib.a3v4_dispatch_1layer.restype = ctypes.c_double
        self.lib.a3v4_get_layer_meta.argtypes = [ctypes.c_void_p, ctypes.c_int, c_i32_p, c_dbl_p]
        self.lib.a3v4_get_layer_meta.restype = None
        self.lib.a3v4_copy_layer_bufs.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            c_i32_p, c_i32_p, c_i32_p, c_u32_p, c_i32_p, c_i32_p,
            c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_i32_p, c_i32_p,
        ]
        self.lib.a3v4_copy_layer_bufs.restype = None
        self.lib.a3v4_copy_active_mask.argtypes = [ctypes.c_void_p, ctypes.c_int, c_u8_p]
        self.lib.a3v4_copy_active_mask.restype = None

        enc_c = np.ascontiguousarray(enc_sorted_4l, dtype=np.int8)
        olo_c = np.ascontiguousarray(pos_olo_4l, dtype=np.int32)
        t0 = time.perf_counter()
        self.handle = self.lib.a3v4_create(
            enc_c.ctypes.data_as(c_i8_p), olo_c.ctypes.data_as(c_i32_p), self.n_threads,
        )
        self.init_ms = (time.perf_counter() - t0) * 1000.0

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.a3v4_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def dispatch_4layers(self, kq_4l, frame_cost=(1024.0, 640.0, 486.0, 410.0, 1680.0),
                         chain_rel=0.72, max_nbr_use=32, fast_1pass=0, tail_pack=1, export_mask=0,
                         us32=0.4330, us16=0.2940):
        kq_c = np.ascontiguousarray(kq_4l, dtype=np.int32)
        B = int(kq_c.shape[1])
        fc_c = np.ascontiguousarray(frame_cost, dtype=np.float64)
        ph = np.zeros(3, dtype=np.float64)
        ms = self.lib.a3v4_dispatch_4layers(
            self.handle, B,
            kq_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            fc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(chain_rel), int(max_nbr_use), int(fast_1pass), int(tail_pack), int(export_mask),
            float(us32), float(us16),
            ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return float(ms), ph.tolist()

    def dispatch_1layer(self, l, kq_l, frame_cost=(1024.0, 640.0, 486.0, 410.0, 1680.0),
                        chain_rel=0.72, max_nbr_use=32, fast_1pass=0, tail_pack=1, export_mask=0,
                        us32=0.4330, us16=0.2940):
        kq_c = np.ascontiguousarray(kq_l, dtype=np.int32)
        B = int(kq_c.shape[0])
        fc_c = np.ascontiguousarray(frame_cost, dtype=np.float64)
        ph = np.zeros(3, dtype=np.float64)
        ms = self.lib.a3v4_dispatch_1layer(
            self.handle, int(l), B,
            kq_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            fc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(chain_rel), int(max_nbr_use), int(fast_1pass), int(tail_pack), int(export_mask),
            float(us32), float(us16),
            ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return float(ms), ph.tolist()

    def get_layer_meta(self, l):
        oi = np.zeros(14, dtype=np.int32)
        od = np.zeros(3, dtype=np.float64)
        self.lib.a3v4_get_layer_meta(
            self.handle, int(l),
            oi.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            od.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return dict(
            ncta=int(oi[0]), ntiles=int(oi[1]), n_tok=int(oi[2]), n_neu=int(oi[3]),
            p2_nblk=int(oi[4]), p2_h_elems=int(oi[5]), p2_max_npad=int(oi[6]),
            p2_uniform_tm=int(oi[7]), p2_kcols_len=int(oi[8]),
            counts={0: int(oi[9]), 1: int(oi[10]), 2: int(oi[11]), 3: int(oi[12]), 4: int(oi[13])},
            compute_cells=int(od[0]), silicon_cost=float(od[1]), p2_union_ratio=float(od[2]),
        )

    def get_layer_active_mask(self, l, B):
        m = np.zeros((B, 4096), dtype=np.uint8)
        self.lib.a3v4_copy_active_mask(self.handle, int(l), m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return m.astype(bool)

    def build_gpu_program(self, l, B, dev='cuda'):
        meta = self.get_layer_meta(l)
        tok_z = np.empty(max(1, meta['n_tok']), dtype=np.int32)
        tok_h = np.empty(max(1, meta['n_tok']), dtype=np.int32)
        neu = np.empty(max(1, meta['n_neu']), dtype=np.int32)
        wmask = np.empty(max(1, meta['n_tok']), dtype=np.uint32)
        tiles = np.empty((max(1, meta['ntiles']), 4), dtype=np.int32)
        cta_desc = np.empty((max(1, meta['ncta']), 4), dtype=np.int32)
        ord_tok = np.empty(B, dtype=np.int32)
        p2_kcols = np.empty(max(32, meta['p2_kcols_len']), dtype=np.int32)
        p2_blk_desc = np.empty((max(1, meta['p2_nblk']), 4), dtype=np.int32)
        p2_h_off = np.empty(max(1, meta['p2_nblk']), dtype=np.int32)
        p2_ncol_pad = np.empty(max(1, meta['p2_nblk']), dtype=np.int32)
        p2_blk_tm = np.empty(max(1, meta['p2_nblk']), dtype=np.int32)

        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_u32_p = ctypes.POINTER(ctypes.c_uint32)
        self.lib.a3v4_copy_layer_bufs(
            self.handle, int(l),
            tok_z.ctypes.data_as(c_i32_p), tok_h.ctypes.data_as(c_i32_p),
            neu.ctypes.data_as(c_i32_p), wmask.ctypes.data_as(c_u32_p),
            tiles.ctypes.data_as(c_i32_p), cta_desc.ctypes.data_as(c_i32_p),
            ord_tok.ctypes.data_as(c_i32_p), p2_kcols.ctypes.data_as(c_i32_p),
            p2_blk_desc.ctypes.data_as(c_i32_p), p2_h_off.ctypes.data_as(c_i32_p),
            p2_ncol_pad.ctypes.data_as(c_i32_p), p2_blk_tm.ctypes.data_as(c_i32_p),
        )

        p2 = dict(
            nblk=meta['p2_nblk'], h_elems=meta['p2_h_elems'], max_npad=meta['p2_max_npad'],
            union_ratio=meta['p2_union_ratio'], uniform_tm=meta['p2_uniform_tm'],
            ord_tok=torch.as_tensor(ord_tok, device=dev),
            kcols=torch.as_tensor(p2_kcols, device=dev),
            blk_desc=torch.as_tensor(p2_blk_desc[:meta['p2_nblk']], device=dev),
            h_off=torch.as_tensor(p2_h_off[:meta['p2_nblk']], device=dev),
            ncol_pad=torch.as_tensor(p2_ncol_pad[:meta['p2_nblk']], device=dev),
            blk_tm=torch.as_tensor(p2_blk_tm[:meta['p2_nblk']], device=dev),
        )
        return dict(
            n64=0, p64=None, ncta=meta['ncta'],
            tok_z=torch.as_tensor(tok_z, device=dev), tok_h=torch.as_tensor(tok_h, device=dev),
            neu=torch.as_tensor(neu, device=dev), wmask=torch.as_tensor(wmask.view(np.int32).copy(), device=dev),
            tiles=torch.as_tensor(tiles[:meta['ntiles']], device=dev),
            cta_desc=torch.as_tensor(cta_desc[:meta['ncta']], device=dev),
            p2=p2, meta=meta,
        )


def main():
    from batch_s import load_keys, RUNS_S, NN
    from batch_neon import build_encoded_tables

    keys, BM, _ = load_keys()
    pos_olo = np.load(os.path.join(RUNS_S, 'pos_olo.npy'))
    enc_all = build_encoded_tables(BM)
    co_npz = np.load(os.path.join(RUNS_S, 'col_orders.npz'))
    col_orders = [co_npz[f'L{l}'] for l in range(4)]
    enc_sorted = np.stack([enc_all[l][:, col_orders[l]] for l in range(4)], axis=0)

    rng0 = np.random.default_rng(0)
    q_1024 = rng0.choice(keys.shape[1], 1024, replace=False)
    q_2048 = rng0.choice(keys.shape[1], 2048, replace=False)
    q_512 = np.random.default_rng(0).choice(keys.shape[1], 512, replace=False)
    online = {512: (27.88, 2516178.0), 1024: (22.97, 4012654.0), 2048: (19.07, 6438606.0)}

    disp = A3V4Dispatcher(enc_sorted, pos_olo, n_threads=16, force_compile=True)
    print(f'=== A3 V4 AVX2 Zero-Branch Dispatcher (16 thr, offline init={disp.init_ms:.1f} ms) ===', flush=True)
    for fast_1p in (0, 1):
        for max_nb in (24, 32, 48):
            print(f'\n  [fast_1pass={fast_1p}, max_nbr={max_nb}, chain_rel=0.72]', flush=True)
            for B, idx in [(512, q_512), (1024, q_1024), (2048, q_2048)]:
                kq_4l = np.stack([keys[l][idx] for l in range(4)], axis=0).astype(np.int32)
                for _ in range(5):
                    disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=max_nb, fast_1pass=fast_1p, export_mask=0)
                runs = [disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=max_nb, fast_1pass=fast_1p, export_mask=0) for _ in range(30)]
                tot = sorted(r[0] for r in runs)
                med = tot[len(tot) // 2]
                p0 = float(np.median([r[1][0] for r in runs]))
                p1 = float(np.median([r[1][1] for r in runs]))
                p2 = float(np.median([r[1][2] for r in runs]))

                disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=max_nb, fast_1pass=fast_1p, export_mask=1)
                tc = tcost = ttiles = tact = 0.0
                for l in range(4):
                    m = disp.get_layer_meta(l)
                    tc += m['compute_cells']; tcost += m['silicon_cost']; ttiles += m['ntiles']
                    act = disp.get_layer_active_mask(l, B)
                    assert act[enc_sorted[l][kq_4l[l]] == 1].all(), f'MISSING MAND 1 L{l+1}'
                    tact += int(act.sum())
                cp = 100.0 * tc / (4 * B * NN)
                oc, ocost = online[B]
                print(
                    f'    B={B:4d}: {med:5.2f}ms (min {tot[0]:4.2f}) [bkt {p0:.2f} + slv {p1:.2f} + pk {p2:.2f}] | '
                    f'comp={cp:5.2f}% ({cp-oc:+5.2f}%p) | cost={tcost:9.0f} ({100*(tcost/ocost-1):+5.1f}%) | '
                    f'fill={100*tact/tc:4.1f}% | tiles={int(ttiles):4d}',
                    flush=True,
                )
    disp.close()


if __name__ == '__main__':
    main()
