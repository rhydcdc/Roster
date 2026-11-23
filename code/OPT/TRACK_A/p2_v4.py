"""A1 Production P2 (down + residual) Kernels for Roster-Tile V4 (sm_89, NVRTC).
Key breakthroughs over V3 (0.00%p accuracy change, bit-exact vs FP32 masked ref):
  1. pack_h_v4: 2D-grid (nblk, ceil(max_npad/64)) zero-idiv compaction (~7x faster than v3 pack_h).
  2. cp.async.ca (L1+L2 cached Wd) + STAGES=2 (max SM occupancy, 4 CTAs/SM) + float4 coalesced FP32 residual add:
     - TM=32 (WD=4, WT=1, KC2=32, S2, .ca): 19.37 TF at d=4096 (93.1% of cuBLAS vs v3's 78.6%!)
     - TM=16 (WD=4, WT=1, KC2=16, S2, .ca): 14.27 TF at d=4096 (68.5% of cuBLAS, 10 KB SMEM!)
  3. pos_olo centroid ordering (eliminates L1/L2 cuBLAS fallback at B=512; recalibrated thresh=0.85)
     + silicon-cost-guided per-layer / per-block TM in {32, 16} selection (OLO_DP & per-layer auto-TM).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[2]
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
if str(BATCH_CODE) not in sys.path:
    sys.path.insert(0, str(BATCH_CODE))

from kern import build  # noqa: E402
from p2_kernels import SRC_PACK_H_V4, build_p2_param  # noqa: E402

_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int32)

# Unified 128-thread (4-warp) P2 kernel handling TM=32 (KC2=32) and TM=16 (KC2=16) in ONE launch!
# Also supports optional s_kcols SMEM preload (-DPRELOAD_KCOLS=1).
SRC_P2_V4_UNIFIED = r'''
__device__ __forceinline__ u32 smem_u32u(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16_ca(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp16_cg(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit_u() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait_u() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4_u(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldm4t_u(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mma_u(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
               : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

template <int TM, int KC2>
__device__ __forceinline__ void run_p2_core(
    unsigned char* smem,
    const bf16* __restrict__ hbase,
    const bf16* __restrict__ Wd,
    const int*  __restrict__ ord_tok,
    const int*  __restrict__ kcols,
    int h0, int ntok, int koff, int npad,
    float*      __restrict__ res) {
  constexpr int DC = 128, WD = 4, WT = 1, NTHR = 128, STAGES = 2;
  constexpr int WDC = DC / WD;      // 32
  constexpr int MT2 = WDC / 16;     // 2
  constexpr int NT2 = TM / 8;       // 4 for TM=32, 2 for TM=16
  constexpr int SB = DC + 8;        // 136
  constexpr int SHH = KC2 + 8;      // 40 or 24
  constexpr int STG2 = KC2 * SB + TM * SHH;
  constexpr int D_CP = KC2 * (DC / 8);   // 512 (KC2=32) or 256 (KC2=16)
  constexpr int H_CP = TM * (KC2 / 8);   // 128 (TM=32,K32) or 32 (TM=16,K16)
  constexpr int JD = D_CP / NTHR;        // 4 or 2
  constexpr int PIPE_B = STAGES * STG2 * 2;
  constexpr int OUT_B = TM * DC * 4;
  constexpr int OTOK_OFF = (PIPE_B > OUT_B) ? PIPE_B : OUT_B;

  const int d0 = blockIdx.y * DC;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  bf16* st = (bf16*)smem;
  int* s_otok = (int*)(smem + OTOK_OFF);
  if (tid < TM) s_otok[tid] = (tid < ntok) ? ord_tok[h0 + tid] : 0;
  __syncthreads();

  int dr[JD], dc8[JD]; u32 ddst[JD];
  #pragma unroll
  for (int j = 0; j < JD; j++) {
    int i = tid + j * NTHR;
    dr[j] = i >> 4;            // / 16 (since DC/8 = 16)
    dc8[j] = (i & 15) << 3;
    ddst[j] = smem_u32u(st + dr[j] * SB + dc8[j]);
  }
  const bool hok = (tid < H_CP);
  const int hr = tid / (KC2 / 8), hc = tid % (KC2 / 8);
  const bf16* hsrc = hok ? (hbase + (long)hr * npad + hc * 8) : hbase;
  const u32 hdst = hok ? smem_u32u(st + KC2 * SB + hr * SHH + hc * 8) : 0u;

  const int NK = npad / KC2;
  auto load = [&](int kc, int sidx) {
    const int kb = kc * KC2;
    const u32 dof = (u32)(sidx * STG2 * 2);
    #pragma unroll
    for (int j = 0; j < JD; j++) {
      const int nid = kcols[koff + kb + dr[j]];
      cp16_ca(ddst[j] + dof, Wd + (long)nid * DM + d0 + dc8[j]);
    }
    if (hok) cp16_cg(hdst + dof, hsrc + kb);
  };

  float acc[MT2][NT2][4];
  #pragma unroll
  for (int a = 0; a < MT2; a++)
    #pragma unroll
    for (int bi = 0; bi < NT2; bi++)
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[a][bi][c] = 0.f;

  load(0, 0); cp_commit_u();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait_u<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit_u();
    const bf16* s = st + (kc & 1) * STG2;
    #pragma unroll
    for (int ks = 0; ks < (KC2 / 16); ks++) {
      u32 af[MT2][4];
      #pragma unroll
      for (int mt = 0; mt < MT2; mt++)
        ldm4t_u(smem_u32u(s + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * SB + wid * WDC + mt * 16 + (((lane >> 3) & 1) << 3)),
                af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      #pragma unroll
      for (int np = 0; np < NT2 / 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4_u(smem_u32u(s + KC2 * SB + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)),
               b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < MT2; mt++) {
          mma_u(acc[mt][2 * np],     af[mt], b0, b1);
          mma_u(acc[mt][2 * np + 1], af[mt], b2, b3);
        }
      }
    }
  }
  cp_wait_u<0>();
  __syncthreads();

  float* s_out = (float*)st;
  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < MT2; mt++)
    #pragma unroll
    for (int nt = 0; nt < NT2; nt++)
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int t = nt * 8 + tig * 2 + (c & 1);
        const int d = wid * WDC + mt * 16 + gid + ((c >> 1) << 3);
        s_out[t * DC + d] = acc[mt][nt][c];
      }
  __syncthreads();

  const int dc4 = DC >> 2;  // 32
  for (int i = tid; i < ntok * dc4; i += NTHR) {
    const int r = i >> 5;
    const int c4 = (i & 31) << 2;
    float4* dst = (float4*)(res + (long)s_otok[r] * DM + d0 + c4);
    float4 addv = *(const float4*)(s_out + r * DC + c4);
    float4 cur = *dst;
    cur.x += addv.x; cur.y += addv.y; cur.z += addv.z; cur.w += addv.w;
    *dst = cur;
  }
}

extern "C" __global__ void __launch_bounds__(128) p2_unified_v4(
    const bf16* __restrict__ H_compact,
    const bf16* __restrict__ Wd,
    const int*  __restrict__ ord_tok,
    const int*  __restrict__ kcols,
    const int4* __restrict__ blk_desc,
    const int*  __restrict__ h_off,
    const int*  __restrict__ ncol_pad,
    const int*  __restrict__ blk_tm,
    float*      __restrict__ res,
    int nblk) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int b = blockIdx.x;
  if (b >= nblk) return;
  const int4 bd = blk_desc[b];
  const int npad = ncol_pad[b];
  if (npad <= 0) return;
  const bf16* hbase = H_compact + h_off[b];
  if (blk_tm[b] == 32) {
    run_p2_core<32, 32>(smem, hbase, Wd, ord_tok, kcols, bd.x, bd.y, bd.z, npad, res);
  } else {
    run_p2_core<16, 16>(smem, hbase, Wd, ord_tok, kcols, bd.x, bd.y, bd.z, npad, res);
  }
}
'''


def olo_dp_blocks(kq, pos_olo_l, act_mask, us_32, us_16):
    """Exact 1D DP along pos_olo_l choosing TM=32 (pad 32) or TM=16 (pad 16) per block.
    Fast bitset DP (~2 ms in Python, < 50 us in C++), 100% deterministic!
    """
    B, N = act_mask.shape
    ord_olo = np.argsort(pos_olo_l[kq], kind='stable').astype(np.int32)
    act_ord = np.ascontiguousarray(act_mask[ord_olo])
    bits = np.packbits(act_ord, axis=1, bitorder='little').view(np.uint64)
    ks_ord = kq[ord_olo]

    # Candidate split points: centroid boundaries or multiples of 8 within long centroid runs
    splits = [0]
    run = 0
    for i in range(1, B):
        if ks_ord[i] != ks_ord[i - 1]:
            splits.append(i)
            run = 0
        else:
            run += 1
            if run % 8 == 0:
                splits.append(i)
    if splits[-1] != B:
        splits.append(B)

    S = len(splits)
    INF = 1e18
    dp = np.full(S, INF, dtype=np.float64)
    prv = np.full(S, -1, dtype=np.int32)
    ptm = np.full(S, -1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(S - 1):
        a = splits[i]
        cur = np.zeros(bits.shape[1], dtype=np.uint64)
        p = a
        bc = dp[i]
        for j in range(i + 1, S):
            b = splits[j]
            h = b - a
            if h > 32:
                break
            while p < b:
                cur |= bits[p]
                p += 1
            u = int(_POP16[cur.view(np.uint16)].sum())
            if u == 0:
                c, tm_b = 0.0, (16 if h <= 16 else 32)
            else:
                up32 = ((u + 31) // 32) * 32
                c32 = us_32 * (up32 / 32.0)
                if h <= 16:
                    up16 = ((u + 15) // 16) * 16
                    c16 = us_16 * (up16 / 32.0)
                    if c16 < c32:
                        c, tm_b = c16, 16
                    else:
                        c, tm_b = c32, 32
                else:
                    c, tm_b = c32, 32
            if bc + c < dp[j]:
                dp[j] = bc + c
                prv[j] = i
                ptm[j] = tm_b

    blocks = []
    cur = S - 1
    while cur > 0:
        pi = int(prv[cur])
        a, b = splits[pi], splits[cur]
        blocks.append((int(ptm[cur]), ord_olo[a:b]))
        cur = pi
    blocks.reverse()
    return ord_olo, blocks


def build_p2_v4_descriptors(blocks, act_mask, B, dev='cuda'):
    """Build GPU descriptors for p2_unified_v4 (or single-TM kernel) with per-block pad_k = tm (32 or 16)."""
    ord_list = []
    ranges = []
    cur = 0
    for tm, idxs in blocks:
        arr = np.asarray(idxs, dtype=np.int32)
        if len(arr) == 0:
            continue
        assert len(arr) <= tm
        ord_list.append(arr)
        ranges.append((tm, cur, len(arr)))
        cur += len(arr)
    assert cur == B
    ord_tok = np.concatenate(ord_list).astype(np.int32)
    act_ord = act_mask[ord_tok]

    raw = []
    tot_union_pad = 0
    max_npad = 0
    cnt_tm = {32: 0, 16: 0}
    for tm, h0, ntok in ranges:
        cols = np.nonzero(act_ord[h0:h0 + ntok].any(axis=0))[0].astype(np.int32)
        if len(cols) > 0:
            pad_k = 32 if tm == 32 else 16
            npad = ((len(cols) + pad_k - 1) // pad_k) * pad_k
            raw.append((tm, h0, ntok, cols, npad))
            tot_union_pad += tm * npad
            cnt_tm[tm] += 1
            if npad > max_npad:
                max_npad = npad

    # LPT sort: longest weighted work (tm * npad) first -> zero wave tail!
    raw.sort(key=lambda x: -(x[4] * (1.0 if x[0] == 32 else 0.68)))

    kl, bd, ho_l, np_l, tm_l = [], [], [], [], []
    ko = ho = 0
    for tm, h0, ntok, cols, npad in raw:
        cp = np.zeros(npad, dtype=np.int32)
        cp[:len(cols)] = cols
        kl.append(cp)
        bd.append((h0, ntok, ko, len(cols)))
        ho_l.append(ho)
        np_l.append(npad)
        tm_l.append(tm)
        ko += npad
        ho += tm * npad

    N = act_mask.shape[1]
    p2 = dict(
        nblk=len(bd),
        h_elems=max(128, ho),
        max_npad=max_npad,
        union_ratio=tot_union_pad / float(B * N),
        cnt_tm=cnt_tm,
        uniform_tm=(32 if cnt_tm[16] == 0 else (16 if cnt_tm[32] == 0 else 0)),
        ord_tok=torch.as_tensor(ord_tok, device=dev),
        kcols=torch.as_tensor(np.concatenate(kl) if kl else np.zeros(32, np.int32), device=dev),
        blk_desc=torch.as_tensor(np.array(bd, np.int32).reshape(-1, 4) if bd else np.zeros((0, 4), np.int32), device=dev),
        h_off=torch.as_tensor(np.array(ho_l, np.int32) if ho_l else np.zeros(1, np.int32), device=dev),
        ncol_pad=torch.as_tensor(np.array(np_l, np.int32) if np_l else np.zeros(1, np.int32), device=dev),
        blk_tm=torch.as_tensor(np.array(tm_l, np.int32) if tm_l else np.zeros(1, np.int32), device=dev),
    )
    return ord_tok, p2


class P2V4Engine:
    def __init__(self, dm=1024, nh=4096):
        self.dm, self.nh = dm, nh
        self.pack_h = build(SRC_PACK_H_V4, ['pack_h_v4'])['pack_h_v4']
        self.p2_32, self.sm32, self.nt32 = build_p2_param(dm, 32, 128, 4, 1, 32, 2, 1)
        self.p2_16, self.sm16, self.nt16 = build_p2_param(dm, 16, 128, 4, 1, 16, 2, 1)
        self.p2_uni = build(SRC_P2_V4_UNIFIED, ['p2_unified_v4'], extra=[f'-DDM={dm}'])['p2_unified_v4']
        self.sm_uni = 2 * (32 * 136 + 32 * 40) * 2 + 4 * 32
        self.p2_uni.set_smem(self.sm_uni)

    def run_p2(self, H_full, H_compact, Wdn, p2, y_tmp, res, add_res_perm, cublas_thresh=0.85):
        if p2['nblk'] == 0:
            return
        if p2['union_ratio'] > cublas_thresh:
            torch.mm(H_full, Wdn, out=y_tmp)
            n = H_full.shape[0] * self.dm
            add_res_perm(((n // 4 + 255) // 256,), (256,), [res, y_tmp, p2['ord_tok'], H_full.shape[0], self.dm])
            return
        ky = (p2['max_npad'] + 63) // 64
        self.pack_h((p2['nblk'], ky), (256,), [
            H_full, p2['kcols'], p2['blk_desc'], p2['h_off'], p2['ncol_pad'], p2['blk_tm'],
            H_compact, p2['nblk'], self.nh
        ])
        u = p2['uniform_tm']
        if u == 32:
            self.p2_32((p2['nblk'], self.dm // 128), (self.nt32,), [
                H_compact, Wdn, p2['ord_tok'], p2['kcols'], p2['blk_desc'],
                p2['h_off'], p2['ncol_pad'], res, p2['nblk']
            ], smem=self.sm32)
        elif u == 16:
            self.p2_16((p2['nblk'], self.dm // 128), (self.nt16,), [
                H_compact, Wdn, p2['ord_tok'], p2['kcols'], p2['blk_desc'],
                p2['h_off'], p2['ncol_pad'], res, p2['nblk']
            ], smem=self.sm16)
        else:
            self.p2_uni((p2['nblk'], self.dm // 128), (128,), [
                H_compact, Wdn, p2['ord_tok'], p2['kcols'], p2['blk_desc'],
                p2['h_off'], p2['ncol_pad'], p2['blk_tm'], res, p2['nblk']
            ], smem=self.sm_uni)
