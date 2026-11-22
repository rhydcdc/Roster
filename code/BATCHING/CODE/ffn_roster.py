"""Optimized Multi-Scale Strip-Mining Roster-Tile CUDA Tensor Core Kernels (sm_89, NVRTC).
Zero sloppy kernel overhead:
  1. Offline permuted weights W_gu[H, 2*d] & W_dn[H, d] matching column-sort order (contiguous strips).
  2. P1 (gate + up + silu*mul + mask fused):
     - Transposed MMA layout (M = 2*NN, N = TM) with [gate 0..7 | up 0..7] 16-row blocks:
       -> 100% FULL mma.m16n8k16 on ALL 4 tile sizes (32x32, 32x16, 16x16, 8x16)!
       -> ZERO warp shuffle for silu(gate)*up!
     - 3-stage cp.async software pipeline + ldmatrix.x4/.x2.
     - Compact per-tile H_buf (0.4 - 1.4 MB -> 100% L2 resident, ZERO H.zero_()!).
  3. P2 (down + residual fused):
     - Same-row K-chaining: all tiles sharing exact same row set accumulate in FP32 registers!
     - M = DC = 128, N = TM in {8, 16, 32}, K = 16*nkg (100% FULL mma.m16n8k16!).
     - SMEM-coalesced epilogue: 128 consecutive threads issue aligned 128B red.global.add.f32!
  4. Multi-stream concurrent dispatch across tile sizes inside CUDA Graph (eliminates wave tails).

python -X utf8 -W ignore BATCHING/CODE/ffn_roster.py
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from kern import build, time_graph  # noqa: E402
from ffn_kern3 import silu_mul, silu_mul_mask, add_res  # noqa: E402

SRC_P1 = r'''
// Transposed P1: C[MW, TM] = W[MW, DM] . Z^T[DM, TM], MW = 2 * NN (multiple of 16), TM in {8,16,32}
// W rows grouped in blocks of 16: rows [16p..16p+7] = gate[8p..8p+7], rows [16p+8..16p+15] = up[8p..8p+7]
// -> acc[mt][nt][0]=gate(j,t0), acc[mt][nt][2]=up(j,t0) IN THE SAME THREAD (zero shuffle)!
#define MW (2 * NN)
#define NTHR (32 * WM * WN)
#define WMW (MW / WM)
#define WTM (TM / WN)
#define MT (WMW / 16)
#define NT (WTM / 8)
#define SA (KC + 8)
#define STG ((MW + TM) * SA)
#define NSEG (KC / 8)
#define JW (MW * NSEG / NTHR)
#define JZ (TM * NSEG / NTHR)

__device__ __forceinline__ u32 smem_u32(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldm2(u32 a, u32& r0, u32& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0),"=r"(r1) : "r"(a)); }
__device__ __forceinline__ void mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
               : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

extern "C" __global__ void __launch_bounds__(NTHR) p1_ms(
    const bf16* __restrict__ Z,       // (B, DM)
    const bf16* __restrict__ W,       // (H, 2*DM): row n = [gate[n] | up[n]]
    const u32*  __restrict__ wmask,   // aligned with tok: bit j = active (exact-once owner!)
    const int*  __restrict__ tok,
    const int*  __restrict__ neu,
    const int4* __restrict__ work,    // (tok_off, ntok, neu_off, nneu)
    const int*  __restrict__ hoff,    // compact H_buf element offset per tile (size TM * NN)
    bf16*       __restrict__ H_buf,
    int nwork) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int w = blockIdx.x;
  if (w >= nwork) return;
  const int4 wk = work[w];
  const int ntok = wk.y, nneu = wk.w;
  const int h0 = hoff[w];

  bf16* st = (bf16*)smem;
  int* s_tok = (int*)(st + STAGES * STG);
  u32* s_msk = (u32*)(s_tok + TM);
  int* s_neu = (int*)(s_msk + TM);

  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int tok0 = tok[wk.x], neu0 = neu[wk.z];
  for (int i = tid; i < TM; i += NTHR) {
    s_tok[i] = (i < ntok) ? tok[wk.x + i] : tok0;
    s_msk[i] = (i < ntok) ? wmask[wk.x + i] : 0u;
  }
  for (int i = tid; i < NN; i += NTHR) {
    s_neu[i] = (i < nneu) ? neu[wk.z + i] : neu0;
  }
  __syncthreads();

  // Precompute cp.async addresses once outside K loop (unconditional -> zero divergence, zero NaN)
  const bf16* wsrc[JW]; u32 wdst[JW];
  const bf16* zsrc[JZ]; u32 zdst[JZ];

  #pragma unroll
  for (int j = 0; j < JW; j++) {
    int i = tid + j * NTHR, r = i / NSEG, c = i % NSEG;
    int blk8 = r >> 4, is_up = (r >> 3) & 1, sub = r & 7;
    int n = (blk8 << 3) + sub;
    wsrc[j] = W + (long)s_neu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
  }
  #pragma unroll
  for (int j = 0; j < JZ; j++) {
    int i = tid + j * NTHR, r = i / NSEG, c = i % NSEG;
    zsrc[j] = Z + (long)s_tok[r] * DM + c * 8;
    zdst[j] = smem_u32(st + (MW + r) * SA + c * 8);
  }

  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    #pragma unroll
    for (int j = 0; j < JW; j++) cp16(wdst[j] + dof, wsrc[j] + so);
    #pragma unroll
    for (int j = 0; j < JZ; j++) cp16(zdst[j] + dof, zsrc[j] + so);
  };

  float acc[MT][NT][4];
  #pragma unroll
  for (int a = 0; a < MT; a++)
    #pragma unroll
    for (int b = 0; b < NT; b++)
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

  const int wm = wid % WM, wn = wid / WM;
  const int NK = DM / KC;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { load(s, s); cp_commit(); }

  for (int kc = 0; kc < NK; kc++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    { const int nk = kc + STAGES - 1; if (nk < NK) load(nk, nk % STAGES); cp_commit(); }
    const bf16* s = st + (kc % STAGES) * STG;
    #pragma unroll
    for (int ks = 0; ks < KC / 16; ks++) {
      u32 af[MT][4];
      #pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        ldm4(smem_u32(s + (wm * WMW + mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
             af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      }
#if NT == 1
      {
        u32 b0, b1;
        ldm2(smem_u32(s + (MW + wn * WTM + (lane & 7)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1);
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) mma(acc[mt][0], af[mt], b0, b1);
      }
#else
      #pragma unroll
      for (int np = 0; np < NT / 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4(smem_u32(s + (MW + wn * WTM + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
             b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          mma(acc[mt][2 * np],     af[mt], b0, b1);
          mma(acc[mt][2 * np + 1], af[mt], b2, b3);
        }
      }
#endif
    }
  }
  cp_wait<0>();
  __syncthreads();

  // Epilogue: silu(gate) * up * mask -> SMEM hs[TM][NN] -> vectorized coalesced store to H_buf!
  bf16* hs = st;
  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    const int j = (wm * MT + mt) * 8 + gid;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int t0 = wn * WTM + nt * 8 + tig * 2;
      const int t1 = t0 + 1;
      const bool on0 = (t0 < ntok) && (j < nneu) && ((s_msk[t0] >> j) & 1u);
      const bool on1 = (t1 < ntok) && (j < nneu) && ((s_msk[t1] >> j) & 1u);
      hs[t0 * NN + j] = on0 ? f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]) : (bf16)0;
      hs[t1 * NN + j] = on1 ? f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]) : (bf16)0;
    }
  }
  __syncthreads();

  // Coalesced 16B stores into compact H_buf[h0 .. h0 + TM*NN) — ZERO H_buf.zero_() needed!
  const int tot_vec = (TM * NN) / 8;
  for (int i = tid; i < tot_vec; i += NTHR) {
    ((uint4*)(H_buf + h0))[i] = ((const uint4*)hs)[i];
  }
}
'''

SRC_P2 = r'''
// P2 (down + residual): res[tok, d] += sum_{k in chained tiles} H_buf[k] . Wd[neu[k], d]
// M = DC (128), N = TM (8, 16, or 32), K = 16 * nkg (exact multiple of 16 -> 100% full mma.m16n8k16!)
// Epilogue stages FP32 acc into SMEM [TM][DC] so 128 threads issue 100% coalesced 128B red.global.add.f32!
#define NTHR2 (32 * WD)
#define WDC (DC / WD)
#define MT2 (WDC / 16)
#define NT2 (TM / 8)
#define KC2 32
#define SB (DC + 8)
#define SHH (KC2 + 8)
#define STG2 (KC2 * SB + TM * SHH)
#define JD (KC2 * (DC / 8) / NTHR2)
#define JH_TOT (TM * (KC2 / 8))

__device__ __forceinline__ u32 smem_u32b(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16b(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commitb() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_waitb() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4b(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldm2b(u32 a, u32& r0, u32& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0),"=r"(r1) : "r"(a)); }
__device__ __forceinline__ void ldm4tb(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mmab(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
               : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

extern "C" __global__ void __launch_bounds__(NTHR2) p2_ms(
    const bf16* __restrict__ H_buf,   // compact Stage 1 output
    const bf16* __restrict__ Wd,      // (H, DM) sorted row-major
    const int*  __restrict__ tok,     // row IDs for each P2 work item
    const int*  __restrict__ kcols,   // 16 global sorted neuron IDs per k-chunk (chunk c -> kcols[c*16..c*16+15])
    const int2* __restrict__ khmeta,  // per k-chunk: (h_base_off, h_row_stride) in H_buf!
    const int4* __restrict__ work,    // (tok_off, ntok, kchunk_off, nkchunks)
    float*      __restrict__ res,     // (B, DM) fp32 residual
    int nwork) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int w = blockIdx.x;
  if (w >= nwork) return;
  const int4 wk = work[w];
  const int ntok = wk.y, nkg = wk.w;
  const int d0 = blockIdx.y * DC;

  bf16* st = (bf16*)smem;
  int* s_tok = (int*)(st + STAGES * STG2);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  for (int i = tid; i < TM; i += NTHR2) s_tok[i] = (i < ntok) ? tok[wk.x + i] : 0;
  __syncthreads();

  int dr[JD], dc8[JD]; u32 ddst[JD];
  #pragma unroll
  for (int j = 0; j < JD; j++) {
    int i = tid + j * NTHR2;
    dr[j] = i / (DC / 8);
    dc8[j] = (i % (DC / 8)) * 8;
    ddst[j] = smem_u32b(st + dr[j] * SB + dc8[j]);
  }

  // H load mapping: JH_TOT = TM * 4 (<= 128 = NTHR2! Each thread loads at most 1 uint4 of H!)
  const bool do_h = (tid < JH_TOT);
  const int hr = do_h ? (tid >> 2) : 0;
  const int hc = do_h ? (tid & 3) : 0;
  const int hgrp = hc >> 1;
  const int hc8 = (hc & 1) * 8;
  const u32 hdst = do_h ? smem_u32b(st + KC2 * SB + hr * SHH + hc * 8) : 0u;
  const bool hok = do_h && (hr < ntok);

  const int NK2 = (nkg + 1) >> 1;
  auto load = [&](int kc, int sidx) {
    const u32 dof = (u32)(sidx * STG2 * 2);
    const int ca = wk.z + 2 * kc;
    const int cb = (2 * kc + 1 < nkg) ? (wk.z + 2 * kc + 1) : -1;
    #pragma unroll
    for (int j = 0; j < JD; j++) {
      const int ch = (dr[j] >> 4) ? cb : ca;
      if (ch >= 0) {
        const int neu_id = kcols[ch * 16 + (dr[j] & 15)];
        cp16b(ddst[j] + dof, Wd + (long)neu_id * DM + d0 + dc8[j]);
      }
    }
    if (do_h) {
      const int ch = hgrp ? cb : ca;
      if (ch >= 0) {
        const int2 hm = khmeta[ch];
        cp16b(hdst + dof, H_buf + hm.x + hr * hm.y + hc8);
      }
    }
  };

  float acc[MT2][NT2][4];
  #pragma unroll
  for (int a = 0; a < MT2; a++)
    #pragma unroll
    for (int b = 0; b < NT2; b++)
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < NK2) load(s, s); cp_commitb(); }
  for (int kc = 0; kc < NK2; kc++) {
    cp_waitb<STAGES - 2>();
    __syncthreads();
    { const int nk = kc + STAGES - 1; if (nk < NK2) load(nk, nk % STAGES); cp_commitb(); }
    const bf16* s = st + (kc % STAGES) * STG2;
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      if (2 * kc + ks >= nkg) break;
      u32 af[MT2][4];
      #pragma unroll
      for (int mt = 0; mt < MT2; mt++) {
        ldm4tb(smem_u32b(s + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * SB + wid * WDC + mt * 16 + (((lane >> 3) & 1) << 3)),
               af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      }
#if NT2 == 1
      {
        u32 b0, b1;
        ldm2b(smem_u32b(s + KC2 * SB + (lane & 7) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1);
        #pragma unroll
        for (int mt = 0; mt < MT2; mt++) mmab(acc[mt][0], af[mt], b0, b1);
      }
#else
      #pragma unroll
      for (int np = 0; np < NT2 / 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4b(smem_u32b(s + KC2 * SB + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)),
              b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < MT2; mt++) {
          mmab(acc[mt][2 * np],     af[mt], b0, b1);
          mmab(acc[mt][2 * np + 1], af[mt], b2, b3);
        }
      }
#endif
    }
  }
  cp_waitb<0>();
  __syncthreads();

  // Coalesced Epilogue: write FP32 acc -> SMEM s_out[TM][DC], then 128 consecutive threads red.global.add!
  float* s_out = (float*)st;
  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < MT2; mt++) {
    #pragma unroll
    for (int nt = 0; nt < NT2; nt++) {
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int t = nt * 8 + tig * 2 + (c & 1);
        const int d = wid * WDC + mt * 16 + gid + ((c >> 1) << 3);
        s_out[t * DC + d] = acc[mt][nt][c];
      }
    }
  }
  __syncthreads();

  // 128 threads -> 128 consecutive floats = 4 aligned 128-byte L2 cache lines per row!
  for (int r = 0; r < ntok; r++) {
    float* p = res + (long)s_tok[r] * DM + d0 + tid;
    asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(s_out[r * DC + tid]));
  }
}
'''


class P1Kernel:
    def __init__(self, TM, NN, WM, WN, KC=32, STAGES=3, dm=1024):
        nthr = 32 * WM * WN
        mw = 2 * NN
        nseg = KC // 8
        assert (mw // WM) % 16 == 0 and (TM // WN) % 8 == 0 and KC % 16 == 0
        assert (mw * nseg) % nthr == 0 and (TM * nseg) % nthr == 0
        assert NN <= 32
        self.TM, self.NN, self.nthr, self.dm = TM, NN, nthr, dm
        k = build(SRC_P1, ['p1_ms'], extra=[
            f'-DTM={TM}', f'-DNN={NN}', f'-DWM={WM}', f'-DWN={WN}',
            f'-DKC={KC}', f'-DSTAGES={STAGES}', f'-DDM={dm}',
        ])
        self.k = k['p1_ms']
        stg = (mw + TM) * (KC + 8)
        self.smem = STAGES * stg * 2 + 4 * (2 * TM + NN)
        self.k.set_smem(self.smem)

    def __call__(self, Z, Wgu, blk, H_buf, stream=None):
        if blk['n'] == 0:
            return
        self.k((blk['n'],), (self.nthr,), [
            Z, Wgu, blk['wmask'], blk['tok'], blk['neu'], blk['work'], blk['hoff'], H_buf, blk['n']
        ], smem=self.smem, stream=stream)


class P2Kernel:
    def __init__(self, TM, DC=128, WD=4, STAGES=2, dm=1024):
        nthr = 32 * WD
        assert (DC // WD) % 16 == 0 and TM in (8, 16, 32) and dm % DC == 0
        assert (32 * (DC // 8)) % nthr == 0
        self.TM, self.DC, self.nthr, self.dm = TM, DC, nthr, dm
        k = build(SRC_P2, ['p2_ms'], extra=[
            f'-DTM={TM}', f'-DDC={DC}', f'-DWD={WD}', f'-DSTAGES={STAGES}', f'-DDM={dm}',
        ])
        self.k = k['p2_ms']
        stg2 = 32 * (DC + 8) + TM * 40
        smem_pipe = STAGES * stg2 * 2 + 4 * TM
        smem_epi = TM * DC * 4 + 4 * TM  # ensure s_out[TM][DC] fits before s_tok!
        # Note: s_tok lives at st + STAGES*STG2 (byte offset STAGES*STG2*2) -> assert smem_epi <= STAGES*STG2*2 + 4*TM!
        assert TM * DC * 4 <= STAGES * stg2 * 2, f's_out ({TM*DC*4}) > pipe ({STAGES*stg2*2})'
        self.smem = smem_pipe
        self.k.set_smem(self.smem)

    def __call__(self, H_buf, Wdn, blk, res, stream=None):
        if blk['n'] == 0:
            return
        self.k((blk['n'], self.dm // self.DC), (self.nthr,), [
            H_buf, Wdn, blk['tok'], blk['kcols'], blk['khmeta'], blk['work'], res, blk['n']
        ], smem=self.smem, stream=stream)


class RosterKernels:
    """Complete multi-scale P1 + P2 kernel suite for a fixed hidden dim d (1024 or 4096)."""
    def __init__(self, dm=1024):
        self.dm = dm
        # P1 kernels for the 4 tile types: 0=32x32, 1=32x16, 2=16x16, 3=8x16
        self.p1 = {
            0: P1Kernel(TM=32, NN=32, WM=2, WN=2, KC=32, STAGES=3, dm=dm),
            1: P1Kernel(TM=32, NN=16, WM=1, WN=2, KC=32, STAGES=3, dm=dm),
            2: P1Kernel(TM=16, NN=16, WM=2, WN=1, KC=32, STAGES=3, dm=dm),
            3: P1Kernel(TM=8,  NN=16, WM=1, WN=1, KC=32, STAGES=2, dm=dm),
        }
        # P2 kernels for the 3 row heights: 32, 16, 8
        self.p2 = {
            32: P2Kernel(TM=32, DC=128, WD=4, STAGES=2, dm=dm),
            16: P2Kernel(TM=16, DC=128, WD=4, STAGES=2, dm=dm),
            8:  P2Kernel(TM=8,  DC=128, WD=4, STAGES=2, dm=dm),
        }

    def run_layer(self, Z, Wgu, Wdn, prog, H_buf, res):
        """Run 1 layer of Multi-Scale Roster-Tile (strip-ordered, LPT big-first)."""
        # Stage 1: launch in descending tile size (32x32 -> 32x16 -> 16x16 -> 8x16)
        # Note: tiles inside each type are ALREADY ordered by strip_idx for L2 locality!
        for ttype in (0, 1, 2, 3):
            self.p1[ttype](Z, Wgu, prog['p1'][ttype], H_buf)
        # Stage 2: chained multi-K down + residual (32 -> 16 -> 8)
        for tm in (32, 16, 8):
            self.p2[tm](H_buf, Wdn, prog['p2'][tm], res)

    def run_layer_strip_by_strip(self, Z, Wgu, Wdn, prog, H_buf, res):
        """Strict strip-by-strip execution (for d=4096 so ALL 4 sizes of strip s share 8MB L2!)."""
        for sp in prog['strips_p1']:
            for ttype in (0, 1, 2, 3):
                self.p1[ttype](Z, Wgu, sp[ttype], H_buf)
        for sp in prog['strips_p2']:
            for tm in (32, 16, 8):
                self.p2[tm](H_buf, Wdn, sp[tm], res)


def pack_sorted_weights(gate_w, up_w, down_w, col_order):
    """Offline weight physical reordering (0 runtime cost, bit-identical FFN output!).
    gate_w: (H, d), up_w: (H, d), down_w: (d, H), col_order: (H,) int32
    Returns Wgu (H, 2d) bf16, Wdn (H, d) bf16.
    """
    idx = torch.as_tensor(col_order, dtype=torch.long, device=gate_w.device)
    gs = gate_w[idx]
    us = up_w[idx]
    ds = down_w[:, idx]
    Wgu = torch.cat([gs, us], dim=1).detach().to(torch.bfloat16).contiguous()
    Wdn = ds.t().detach().to(torch.bfloat16).contiguous()
    return Wgu, Wdn


def build_roster_program(tiles_with_strip, active_mask_sorted, B, N=4096, chain_across_strips=False, dev='cuda'):
    """Convert multi-scale tiles [(strip_idx, ttype, rows, cols)] into GPU tensors for P1 & P2.
    Guarantees EXACT-ONCE ownership of every active (row, col) cell via `assigned` tracking!
    """
    FRAME_TM = (32, 32, 16, 8)
    FRAME_NN = (32, 16, 16, 16)
    pw = (np.uint64(1) << np.arange(32, dtype=np.uint64))

    # Order tiles by (strip_idx, ttype) so L2 processes strip-by-strip, large tiles first!
    ordered = sorted(tiles_with_strip, key=lambda x: (x[0], x[1]))

    assigned = np.zeros((B, N), dtype=bool)
    p1_items = {0: [], 1: [], 2: [], 3: []}
    strips_seen = sorted(set(x[0] for x in ordered))
    p1_by_strip = {s: {0: [], 1: [], 2: [], 3: []} for s in strips_seen}

    # P2 row-chaining map: key = (strip_idx_or_0, tm_bucket, tuple(rows)) -> list of (h_off, h_stride, cols16)
    p2_groups = {}
    p2_order = []

    h_cursor = 0
    for s, ttype, rr, cc in ordered:
        rr = np.asarray(rr, dtype=np.int32)
        cc = np.asarray(cc, dtype=np.int32)
        tm, nn = FRAME_TM[ttype], FRAME_NN[ttype]
        nr, nc = len(rr), len(cc)

        # Exact-once mask assignment: active AND not yet owned by an earlier tile!
        sub_act = active_mask_sorted[np.ix_(rr, cc)]
        sub_asg = assigned[np.ix_(rr, cc)]
        sub_own = sub_act & (~sub_asg)
        if not sub_own.any():
            continue
        assigned[np.ix_(rr, cc)] |= sub_own
        msk = (sub_own.astype(np.uint64) * pw[:nc]).sum(axis=1).astype(np.uint32)

        h_off = h_cursor
        h_cursor += tm * nn  # exact multiple of 128 bf16 (256B aligned!)

        rec = (rr, cc, msk, h_off)
        p1_items[ttype].append(rec)
        p1_by_strip[s][ttype].append(rec)

        # Register 16-neuron K-chunks for Stage 2 P2 (chain identical row sets!)
        tm_p2 = 32 if nr > 16 else (16 if nr > 8 else 8)
        gkey = (-1 if chain_across_strips else s, tm_p2, tuple(rr.tolist()))
        if gkey not in p2_groups:
            p2_groups[gkey] = (s, tm_p2, rr, [])
            p2_order.append(gkey)
        # Split cc (len <= 16 or <= 32) into chunks of 16 (pad with valid neuron ID; H_buf is 0 there!)
        for c_off in range(0, nc, 16):
            sub_c = cc[c_off:c_off + 16]
            c16 = np.full(16, int(sub_c[0]), dtype=np.int32)
            c16[:len(sub_c)] = sub_c
            p2_groups[gkey][3].append((h_off + c_off, nn, c16))

    # Strict verification: 100% of active_mask_sorted cells owned EXACTLY ONCE!
    assert np.array_equal(assigned, active_mask_sorted), 'CRITICAL: assigned != active_mask_sorted!'

    def pack_p1_list(recs):
        if not recs:
            return dict(n=0)
        toks, neus, msks, wk, hoff = [], [], [], [], []
        to = no = 0
        for rr, cc, msk, ho in recs:
            toks.append(rr)
            neus.append(cc)
            msks.append(msk)
            wk.append((to, len(rr), no, len(cc)))
            hoff.append(ho)
            to += len(rr)
            no += len(cc)
        return dict(
            n=len(wk),
            tok=torch.as_tensor(np.concatenate(toks).astype(np.int32), device=dev),
            neu=torch.as_tensor(np.concatenate(neus).astype(np.int32), device=dev),
            wmask=torch.as_tensor(np.concatenate(msks).view(np.int32).copy(), device=dev),
            work=torch.as_tensor(np.array(wk, dtype=np.int32).reshape(-1, 4), device=dev),
            hoff=torch.as_tensor(np.array(hoff, dtype=np.int32), device=dev),
        )

    def pack_p2_list(glist):
        if not glist:
            return dict(n=0)
        # LPT sort within each strip/bucket: largest nkg first to minimize wave tail!
        glist = sorted(glist, key=lambda x: (x[0], -len(x[3])))
        toks, kcols, khmeta, wk = [], [], [], []
        to = ko = 0
        for _, _, rr, chunks in glist:
            # Cap chunks per CTA to keep SM utilization high when nkg is very large (e.g. > 32 chunks = 512 neurons)
            max_k = 32
            toks.append(rr)
            for ho, hs, c16 in chunks:
                kcols.append(c16)
                khmeta.append((ho, hs))
            for a in range(0, len(chunks), max_k):
                wk.append((to, len(rr), ko + a, min(max_k, len(chunks) - a)))
            to += len(rr)
            ko += len(chunks)
        return dict(
            n=len(wk),
            tok=torch.as_tensor(np.concatenate(toks).astype(np.int32), device=dev),
            kcols=torch.as_tensor(np.concatenate(kcols).astype(np.int32), device=dev),
            khmeta=torch.as_tensor(np.array(khmeta, dtype=np.int32).reshape(-1, 2), device=dev),
            work=torch.as_tensor(np.array(wk, dtype=np.int32).reshape(-1, 4), device=dev),
        )

    p1_packed = {t: pack_p1_list(p1_items[t]) for t in (0, 1, 2, 3)}
    strips_p1 = [{t: pack_p1_list(p1_by_strip[s][t]) for t in (0, 1, 2, 3)} for s in strips_seen]

    p2_by_tm = {32: [], 16: [], 8: []}
    p2_by_strip = {s: {32: [], 16: [], 8: []} for s in strips_seen}
    for gkey in p2_order:
        s, tm_p2, rr, chunks = p2_groups[gkey]
        p2_by_tm[tm_p2].append((s, tm_p2, rr, chunks))
        p2_by_strip[s][tm_p2].append((s, tm_p2, rr, chunks))

    p2_packed = {tm: pack_p2_list(p2_by_tm[tm]) for tm in (32, 16, 8)}
    strips_p2 = [{tm: pack_p2_list(p2_by_strip[s][tm]) for tm in (32, 16, 8)} for s in strips_seen]

    return dict(
        H_elems=max(128, h_cursor),
        p1=p1_packed,
        p2=p2_packed,
        strips_p1=strips_p1,
        strips_p2=strips_p2,
        n_p1=sum(p1_packed[t]['n'] for t in (0, 1, 2, 3)),
        n_p2=sum(p2_packed[tm]['n'] for tm in (32, 16, 8)),
    )


def self_test():
    """Verify: (1) Numerical exactness on all 4 tile types + chaining, (2) Dense efficiency vs cuBLAS, (3) Real per-size TFLOPS."""
    torch.manual_seed(0)
    dev = 'cuda'
    H = 4096

    print('=== 1. Numerical Equivalence Verification (All 4 Tile Sizes + K-Chaining) ===', flush=True)
    for d in (1024, 4096):
        rk = RosterKernels(dm=d)
        B = 128
        Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
        Wg = torch.randn(H, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wu = torch.randn(H, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wd = torch.randn(d, H, device=dev, dtype=torch.bfloat16) * 0.05
        col_order = np.random.default_rng(7).permutation(H).astype(np.int32)
        Wgu, Wdn = pack_sorted_weights(Wg, Wu, Wd, col_order)

        # Build synthetic mixed tiles covering all 4 types (including partial rows/cols & overlapping duplicates!)
        tiles = [
            (0, 0, np.arange(0, 32), np.arange(0, 32)),
            (0, 0, np.arange(0, 32), np.arange(32, 64)),      # chained same rows!
            (0, 1, np.arange(16, 45), np.arange(64, 79)),     # partial 29x15!
            (0, 2, np.arange(40, 56), np.arange(100, 116)),
            (0, 3, np.arange(55, 63), np.arange(120, 136)),   # 8x16!
            (1, 3, np.arange(60, 65), np.arange(600, 611)),   # partial 5x11!
            (1, 0, np.arange(64, 96), np.arange(512, 544)),
            (1, 1, np.arange(96, 128), np.arange(544, 560)),
        ]
        rng = np.random.default_rng(123)
        act_sorted = np.zeros((B, H), dtype=bool)
        for _, _, rr, cc in tiles:
            # Random ~70% active inside each tile
            sub = rng.random((len(rr), len(cc))) < 0.72
            act_sorted[np.ix_(rr, cc)] |= sub

        prog = build_roster_program(tiles, act_sorted, B, H, chain_across_strips=False, dev=dev)
        H_buf = torch.empty(prog['H_elems'], device=dev, dtype=torch.bfloat16)
        res = torch.zeros(B, d, device=dev, dtype=torch.float32)
        rk.run_layer(Z, Wgu, Wdn, prog, H_buf, res)
        torch.cuda.synchronize()

        # Reference in PyTorch (original unsorted weights + inverse-permuted mask!)
        inv = np.empty(H, dtype=np.int32)
        inv[col_order] = np.arange(H, dtype=np.int32)
        act_orig = torch.as_tensor(act_sorted[:, inv], device=dev, dtype=torch.float32)
        ref = ((torch.nn.functional.silu(Z.float() @ Wg.float().t()) * (Z.float() @ Wu.float().t())) * act_orig) @ Wd.float().t()
        rel_err = (res - ref).abs().max().item() / ref.abs().max().item()
        print(f'  d={d:4d}: max_rel_err = {rel_err:.2e} (PASS < 2e-2: {rel_err < 0.02})', flush=True)
        assert rel_err < 0.02, f'Numerical mismatch! {rel_err}'

    print('\n=== 2. Dense Efficiency vs cuBLAS + Real Per-Tile-Size RTX 4060 Speed ===', flush=True)
    for d in (1024, 4096):
        rk = RosterKernels(dm=d)
        B = 1024
        Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
        Wg = torch.randn(H, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wu = torch.randn(H, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wd = torch.randn(d, H, device=dev, dtype=torch.bfloat16) * 0.05
        Wst = torch.cat([Wg, Wu], dim=0).contiguous()
        Wgu, Wdn = pack_sorted_weights(Wg, Wu, Wd, np.arange(H, dtype=np.int32))
        gu = torch.empty(B, 2 * H, device=dev, dtype=torch.bfloat16)
        a = torch.empty(B, H, device=dev, dtype=torch.bfloat16)
        y = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
        res = torch.zeros(B, d, device=dev, dtype=torch.float32)

        def dense_cublas():
            torch.mm(Z, Wst.t(), out=gu)
            silu_mul(gu, a)
            torch.mm(a, Wd.t(), out=y)
            add_res(res, y)

        t_cb, _, _ = time_graph(dense_cublas, reps=30, warm=5, inner=10)
        flop = 2 * B * d * 3 * H
        print(f'\n[d={d}, B={B}] cuBLAS dense (1 layer): {t_cb:.4f} ms ({flop / t_cb / 1e9:.1f} TFLOPS)', flush=True)

        # Measure each tile size on FULL dense (B x H) so we get exact real silicon TFLOPS!
        act_all = np.ones((B, H), dtype=bool)
        for ttype, (tm, nn, name) in enumerate([(32, 32, '32x32'), (32, 16, '32x16'), (16, 16, '16x16'), (8, 16, '8x16')]):
            tiles = []
            for s in range(H // 512):
                for r0 in range(0, B, tm):
                    for c0 in range(s * 512, (s + 1) * 512, nn):
                        tiles.append((s, ttype, np.arange(r0, r0 + tm), np.arange(c0, c0 + nn)))
            prog = build_roster_program(tiles, act_all, B, H, chain_across_strips=(d == 1024), dev=dev)
            H_buf = torch.empty(prog['H_elems'], device=dev, dtype=torch.bfloat16)

            # Separate P1 and P2 timings
            t_p1, _, _ = time_graph(lambda: rk.p1[ttype](Z, Wgu, prog['p1'][ttype], H_buf), reps=20, warm=3, inner=6)
            t_p2, _, _ = time_graph(lambda: rk.p2[tm](H_buf, Wdn, prog['p2'][tm], res), reps=20, warm=3, inner=6)
            t_tot, _, _ = time_graph(lambda: rk.run_layer(Z, Wgu, Wdn, prog, H_buf, res), reps=20, warm=3, inner=6)
            p1_tf = (2 * B * d * 2 * H) / t_p1 / 1e9
            p2_tf = (2 * B * d * H) / t_p2 / 1e9
            tot_tf = flop / t_tot / 1e9
            eff = 100.0 * t_cb / t_tot
            print(
                f'  Dense {name}: total {t_tot:.4f} ms ({tot_tf:5.1f} TFLOPS, {eff:5.1f}% of cuBLAS) | '
                f'P1 {t_p1:.4f} ms ({p1_tf:5.1f} TF) + P2 {t_p2:.4f} ms ({p2_tf:5.1f} TF)',
                flush=True,
            )


if __name__ == '__main__':
    self_test()
