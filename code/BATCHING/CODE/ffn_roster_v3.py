"""Roster-Tile V3 Optimal CUDA Tensor Core Kernels (sm_89, NVRTC):
Complete output-preserving hardware acceleration suite:
  1. Offline physically permuted weights W_gu[H, 2*d] & W_dn[H, d] matching column-sort order (0 runtime cost).
  2. P1Unified: SINGLE kernel launch per layer in strict Strip order (W=512 or W=256 -> 1.0x DRAM read of W_gu,
     ZERO wave tails across tile sizes or strips!).
     Supports 5 tile types (128 threads = 4 warps, max SMEM 26.2 KB -> 3-4 CTAs/SM):
       * Type 4: 1 x (64x32)  [KC=32, STAGES=2, 4 warps -> 95% of cuBLAS!]
       * Type 0: 1 x (32x32)  [KC=32, STAGES=2, 4 warps -> 15.7 KB SMEM]
       * Type 1: 2 x (32x16)  [KC=32, STAGES=2, 2 warps/tile]
       * Type 2: 2 x (16x16)  [KC=32, STAGES=2, 2 warps/tile]
       * Type 3: 4 x ( 8x16)  [KC=32, STAGES=2, 1 warp/tile, zero __syncthreads]
     Writes ONLY exact-once active cells into L2-resident H_full[B, 4096] (in centroid-sorted token order).
  3. Stage 2 (down + residual, ZERO atomicAdd):
     - Sparse blocks/layers: pack_h + P2Fast (TM=32, DC=128, WD=4, WT=1, STAGES=2, LPT -ncol sorted,
       100% 128-bit cp.async for BOTH W_dn and H_compact, grid=(nblk, d/128) -> 1.0x DRAM read of W_dn!).
     - Dense-union layers (when a layer's TG=32 token-block union > crossover threshold, e.g. L1/L2 at small B=256):
       direct cuBLAS on exact 0-masked H_full @ W_dn + add_res (bit-equivalent, avoids >70% union overhead!).

python -X utf8 -W ignore BATCHING/CODE/ffn_roster_v3.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from kern import build, time_graph  # noqa: E402
from ffn_kern3 import silu_mul, silu_mul_mask, add_res  # noqa: E402
from ffn_roster import pack_sorted_weights  # noqa: E402
from ffn_roster_v2 import SRC_PACK_H, SRC_P2_FAST  # noqa: E402

SRC_P1_V3 = r'''
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

// ---- Type 4: 1 x (64x32), 4 warps (WM=2, WN=2), KC=32, STAGES=2 (SMEM = 21.4 KB -> 95% cuBLAS!) ----
__device__ __forceinline__ void run_t4_64x32(
    unsigned char* smem, int tile_idx,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 64, NN = 32, MW = 64, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int4 wk = tiles[tile_idx];
  const int ntok = wk.y, nneu = wk.w;

  bf16* st = (bf16*)smem;
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk = (u32*)(s_th + TM);
  int* s_nu = (int*)(s_mk + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (tid < TM) {
    s_tz[tid] = (tid < ntok) ? tok_z[wk.x + tid] : tz0;
    s_th[tid] = (tid < ntok) ? tok_h[wk.x + tid] : 0;
    s_mk[tid] = (tid < ntok) ? wmask[wk.x + tid] : 0u;
  }
  if (tid < NN) s_nu[tid] = (tid < nneu) ? neu[wk.z + tid] : nu0;
  __syncthreads();

  const bf16* wsrc[2]; u32 wdst[2];
  const bf16* zsrc[2]; u32 zdst[2];
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    int i = tid + j * 128, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
    zsrc[j] = Z + (long)s_tz[r] * DM + c * 8;
    zdst[j] = smem_u32(st + (MW + r) * SA + c * 8);
  }
  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    #pragma unroll
    for (int j = 0; j < 2; j++) { cp16(wdst[j] + dof, wsrc[j] + so); cp16(zdst[j] + dof, zsrc[j] + so); }
  };

  float acc[2][4][4] = {{{0}}};
  const int wm = wid & 1, wn = wid >> 1;
  const int NK = DM / KC;
  load(0, 0); cp_commit();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit();
    const bf16* s = st + (kc & 1) * STG;
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      u32 af[2][4];
      #pragma unroll
      for (int mt = 0; mt < 2; mt++)
        ldm4(smem_u32(s + (wm * 32 + mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
             af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      #pragma unroll
      for (int np = 0; np < 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4(smem_u32(s + (MW + wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
             b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          mma(acc[mt][2 * np],     af[mt], b0, b1);
          mma(acc[mt][2 * np + 1], af[mt], b2, b3);
        }
      }
    }
  }
  cp_wait<0>();

  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    const int j = (wm * 2 + mt) * 8 + gid;
    if (j < nneu) {
      const int col = s_nu[j];
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const int t0 = wn * 32 + nt * 8 + tig * 2;
        const int t1 = t0 + 1;
        if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}

// ---- Type 0: 1 x (32x32), 4 warps, KC=32, STAGES=2 (SMEM = 15.9 KB -> high occupancy!) ----
__device__ __forceinline__ void run_t0_32x32(
    unsigned char* smem, int tile_idx,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 32, NN = 32, MW = 64, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int4 wk = tiles[tile_idx];
  const int ntok = wk.y, nneu = wk.w;

  bf16* st = (bf16*)smem;
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk = (u32*)(s_th + TM);
  int* s_nu = (int*)(s_mk + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (tid < TM) {
    s_tz[tid] = (tid < ntok) ? tok_z[wk.x + tid] : tz0;
    s_th[tid] = (tid < ntok) ? tok_h[wk.x + tid] : 0;
    s_mk[tid] = (tid < ntok) ? wmask[wk.x + tid] : 0u;
  }
  if (tid < NN) s_nu[tid] = (tid < nneu) ? neu[wk.z + tid] : nu0;
  __syncthreads();

  const bf16* wsrc[2]; u32 wdst[2];
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    int i = tid + j * 128, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
  }
  const int zr = tid >> 2, zc = tid & 3;
  const bf16* zsrc = Z + (long)s_tz[zr] * DM + zc * 8;
  const u32 zdst = smem_u32(st + (MW + zr) * SA + zc * 8);

  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    cp16(wdst[0] + dof, wsrc[0] + so);
    cp16(wdst[1] + dof, wsrc[1] + so);
    cp16(zdst + dof, zsrc + so);
  };

  float acc[2][2][4] = {{{0}}};
  const int wm = wid & 1, wn = wid >> 1;
  const int NK = DM / KC;
  load(0, 0); cp_commit();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit();
    const bf16* s = st + (kc & 1) * STG;
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      u32 af[2][4];
      #pragma unroll
      for (int mt = 0; mt < 2; mt++)
        ldm4(smem_u32(s + (wm * 32 + mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
             af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      u32 b0, b1, b2, b3;
      ldm4(smem_u32(s + (MW + wn * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
           b0, b1, b2, b3);
      #pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        mma(acc[mt][0], af[mt], b0, b1);
        mma(acc[mt][1], af[mt], b2, b3);
      }
    }
  }
  cp_wait<0>();

  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    const int j = (wm * 2 + mt) * 8 + gid;
    if (j < nneu) {
      const int col = s_nu[j];
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const int t0 = wn * 16 + nt * 8 + tig * 2;
        const int t1 = t0 + 1;
        if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}

// ---- Type 1: 2 x (32x16), 2 warps/tile ----
__device__ __forceinline__ void run_t1_2x32x16(
    unsigned char* smem, int idx0, int idx1,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 32, NN = 16, MW = 32, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  constexpr int HALF_BYTES = 2 * STG * 2 + 4 * (3 * TM + NN);
  const int tid = threadIdx.x, hid = tid >> 6, ltid = tid & 63, lane = tid & 31, lwid = ltid >> 5;
  const int tile_idx = (hid == 0) ? idx0 : idx1;
  const bool valid = (tile_idx >= 0);
  const int4 wk = valid ? tiles[tile_idx] : tiles[idx0];
  const int ntok = valid ? wk.y : 0, nneu = valid ? wk.w : 0;

  bf16* st = (bf16*)(smem + hid * HALF_BYTES);
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk = (u32*)(s_th + TM);
  int* s_nu = (int*)(s_mk + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (ltid < TM) {
    s_tz[ltid] = (ltid < ntok) ? tok_z[wk.x + ltid] : tz0;
    s_th[ltid] = (ltid < ntok) ? tok_h[wk.x + ltid] : 0;
    s_mk[ltid] = (ltid < ntok) ? wmask[wk.x + ltid] : 0u;
  }
  if (ltid < NN) s_nu[ltid] = (ltid < nneu) ? neu[wk.z + ltid] : nu0;
  __syncthreads();

  const bf16* wsrc[2]; u32 wdst[2];
  const bf16* zsrc[2]; u32 zdst[2];
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    int i = ltid + j * 64, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
    zsrc[j] = Z + (long)s_tz[r] * DM + c * 8;
    zdst[j] = smem_u32(st + (MW + r) * SA + c * 8);
  }
  auto load = [&](int kc, int sidx) {
    if (!valid) return;
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    #pragma unroll
    for (int j = 0; j < 2; j++) { cp16(wdst[j] + dof, wsrc[j] + so); cp16(zdst[j] + dof, zsrc[j] + so); }
  };

  float acc[2][2][4] = {{{0}}};
  const int NK = DM / KC;
  load(0, 0); cp_commit();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit();
    if (valid) {
      const bf16* s = st + (kc & 1) * STG;
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        u32 af[2][4];
        #pragma unroll
        for (int mt = 0; mt < 2; mt++)
          ldm4(smem_u32(s + (mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
               af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
        u32 b0, b1, b2, b3;
        ldm4(smem_u32(s + (MW + lwid * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
             b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) { mma(acc[mt][0], af[mt], b0, b1); mma(acc[mt][1], af[mt], b2, b3); }
      }
    }
  }
  cp_wait<0>();
  if (!valid) return;

  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    const int j = mt * 8 + gid;
    if (j < nneu) {
      const int col = s_nu[j];
      #pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const int t0 = lwid * 16 + nt * 8 + tig * 2, t1 = t0 + 1;
        if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}

// ---- Type 2: 2 x (16x16), 2 warps/tile ----
__device__ __forceinline__ void run_t2_2x16x16(
    unsigned char* smem, int idx0, int idx1,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 16, NN = 16, MW = 32, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  constexpr int HALF_BYTES = 2 * STG * 2 + 4 * (3 * TM + NN);
  const int tid = threadIdx.x, hid = tid >> 6, ltid = tid & 63, lane = tid & 31, lwid = ltid >> 5;
  const int tile_idx = (hid == 0) ? idx0 : idx1;
  const bool valid = (tile_idx >= 0);
  const int4 wk = valid ? tiles[tile_idx] : tiles[idx0];
  const int ntok = valid ? wk.y : 0, nneu = valid ? wk.w : 0;

  bf16* st = (bf16*)(smem + hid * HALF_BYTES);
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk = (u32*)(s_th + TM);
  int* s_nu = (int*)(s_mk + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (ltid < TM) {
    s_tz[ltid] = (ltid < ntok) ? tok_z[wk.x + ltid] : tz0;
    s_th[ltid] = (ltid < ntok) ? tok_h[wk.x + ltid] : 0;
    s_mk[ltid] = (ltid < ntok) ? wmask[wk.x + ltid] : 0u;
  }
  if (ltid < NN) s_nu[ltid] = (ltid < nneu) ? neu[wk.z + ltid] : nu0;
  __syncthreads();

  const bf16* wsrc[2]; u32 wdst[2];
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    int i = ltid + j * 64, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
  }
  const int zr = ltid >> 2, zc = ltid & 3;
  const bf16* zsrc = Z + (long)s_tz[zr] * DM + zc * 8;
  const u32 zdst = smem_u32(st + (MW + zr) * SA + zc * 8);

  auto load = [&](int kc, int sidx) {
    if (!valid) return;
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    cp16(wdst[0] + dof, wsrc[0] + so); cp16(wdst[1] + dof, wsrc[1] + so); cp16(zdst + dof, zsrc + so);
  };

  float acc[2][4] = {{0}};
  const int NK = DM / KC;
  load(0, 0); cp_commit();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit();
    if (valid) {
      const bf16* s = st + (kc & 1) * STG;
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        u32 af[4];
        ldm4(smem_u32(s + (lwid * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)), af[0], af[1], af[2], af[3]);
        u32 b0, b1, b2, b3;
        ldm4(smem_u32(s + (MW + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1, b2, b3);
        mma(acc[0], af, b0, b1); mma(acc[1], af, b2, b3);
      }
    }
  }
  cp_wait<0>();
  if (!valid) return;

  const int gid = lane >> 2, tig = lane & 3, j = lwid * 8 + gid;
  if (j < nneu) {
    const int col = s_nu[j];
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const int t0 = nt * 8 + tig * 2, t1 = t0 + 1;
      if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
        H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[nt][0]) * acc[nt][2]);
      if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
        H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[nt][1]) * acc[nt][3]);
    }
  }
}

// ---- Type 3: 4 x (8x16), 1 independent warp/tile ----
__device__ __forceinline__ void run_t3_4x8x16(
    unsigned char* smem, int base_idx, int end_idx,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 8, NN = 16, MW = 32, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  constexpr int WARP_BYTES = 2 * STG * 2 + 4 * (3 * TM + NN);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int tile_idx = base_idx + wid;
  if (tile_idx >= end_idx) return;

  const int4 wk = tiles[tile_idx];
  const int ntok = wk.y, nneu = wk.w;

  bf16* st = (bf16*)(smem + wid * WARP_BYTES);
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk = (u32*)(s_th + TM);
  int* s_nu = (int*)(s_mk + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (lane < TM) {
    s_tz[lane] = (lane < ntok) ? tok_z[wk.x + lane] : tz0;
    s_th[lane] = (lane < ntok) ? tok_h[wk.x + lane] : 0;
    s_mk[lane] = (lane < ntok) ? wmask[wk.x + lane] : 0u;
  }
  if (lane < NN) s_nu[lane] = (lane < nneu) ? neu[wk.z + lane] : nu0;
  __syncwarp();

  const bf16* wsrc[4]; u32 wdst[4];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    int i = lane + j * 32, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32(st + r * SA + c * 8);
  }
  const int zr = lane >> 2, zc = lane & 3;
  const bf16* zsrc = Z + (long)s_tz[zr] * DM + zc * 8;
  const u32 zdst = smem_u32(st + (MW + zr) * SA + zc * 8);

  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    #pragma unroll
    for (int j = 0; j < 4; j++) cp16(wdst[j] + dof, wsrc[j] + so);
    cp16(zdst + dof, zsrc + so);
  };

  float acc[2][4] = {{0}};
  const int NK = DM / KC;
  load(0, 0); cp_commit();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<0>();
    __syncwarp();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit();
    const bf16* s = st + (kc & 1) * STG;
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      u32 af[2][4];
      #pragma unroll
      for (int mt = 0; mt < 2; mt++)
        ldm4(smem_u32(s + (mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
             af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      u32 b0, b1;
      ldm2(smem_u32(s + (MW + (lane & 7)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1);
      mma(acc[0], af[0], b0, b1); mma(acc[1], af[1], b0, b1);
    }
  }
  cp_wait<0>();

  const int gid = lane >> 2, tig = lane & 3, t0 = tig * 2, t1 = t0 + 1;
  #pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    const int j = mt * 8 + gid;
    if (j < nneu) {
      const int col = s_nu[j];
      if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
        H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][0]) * acc[mt][2]);
      if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
        H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][1]) * acc[mt][3]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) p1_v3(
    const bf16* __restrict__ Z,
    const bf16* __restrict__ W,
    const u32*  __restrict__ wmask,
    const int*  __restrict__ tok_z,
    const int*  __restrict__ tok_h,
    const int*  __restrict__ neu,
    const int4* __restrict__ tiles,
    const int4* __restrict__ cta_desc,
    bf16*       __restrict__ H_full,
    int ncta) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int cid = blockIdx.x;
  if (cid >= ncta) return;
  const int4 d = cta_desc[cid];
  if (d.x == 4)      run_t4_64x32(smem, d.y, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
  else if (d.x == 0) run_t0_32x32(smem, d.y, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
  else if (d.x == 1) run_t1_2x32x16(smem, d.y, d.z, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
  else if (d.x == 2) run_t2_2x16x16(smem, d.y, d.z, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
  else               run_t3_4x8x16(smem, d.y, d.z, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
}
'''

SRC_ADD_RES_PERM = r'''
// Fused residual add when y is in centroid-sorted row order: res[ord_tok[r], c] += y[r, c]
extern "C" __global__ void add_res_perm(float* __restrict__ res, const bf16* __restrict__ y,
                                        const int* __restrict__ ord_tok, int B, int D) {
  long idx = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  long tot = (long)B * D;
  if (idx >= tot) return;
  int r = (int)(idx / D);
  int c = (int)(idx % D);
  long dst_off = (long)ord_tok[r] * D + c;
  float4 v = *(float4*)(res + dst_off);
  uint2 p = *(const uint2*)(y + idx);
  v.x += __uint_as_float(p.x << 16); v.y += __uint_as_float(p.x & 0xffff0000u);
  v.z += __uint_as_float(p.y << 16); v.w += __uint_as_float(p.y & 0xffff0000u);
  *(float4*)(res + dst_off) = v;
}
'''


class RosterV3Kernels:
    def __init__(self, dm=1024, nh=4096):
        self.dm, self.nh = dm, nh
        self.p1 = build(SRC_P1_V3, ['p1_v3'], extra=[f'-DDM={dm}', f'-DNH_TOT={nh}'])['p1_v3']
        # Max SMEM = max(Type 4: 21.4KB, Type 3: 26.2KB) = 26,240 B -> 3 CTAs/SM!
        self.smem_p1 = max(2 * 128 * 40 * 2 + 4 * 224, 4 * (2 * 40 * 40 * 2 + 4 * 40))
        self.p1.set_smem(self.smem_p1)
        self.pack_h = build(SRC_PACK_H, ['pack_h'])['pack_h']
        self.add_res_perm = build(SRC_ADD_RES_PERM, ['add_res_perm'])['add_res_perm']

        # Fast STAGES=2 Zero-Atomic P2 (TM=32, DC=128, WD=4, WT=1)
        self.p2 = build(SRC_P2_FAST, ['p2_fast'], extra=[
            '-DTM=32', '-DDC=128', '-DWD=4', '-DWT=1', '-DSTAGES=2', f'-DDM={dm}'
        ])['p2_fast']
        self.smem_p2 = 2 * (32 * 136 + 32 * 40) * 2 + 4 * 32
        self.p2.set_smem(self.smem_p2)

    def run_layer(self, Z, Wgu, Wdn, prog, H_full, H_compact, y_tmp, res, p2_cublas_thresh=0.65):
        H_full.zero_()
        if prog['ncta'] > 0:
            self.p1((prog['ncta'],), (128,), [
                Z, Wgu, prog['wmask'], prog['tok_z'], prog['tok_h'], prog['neu'],
                prog['tiles'], prog['cta_desc'], H_full, prog['ncta']
            ], smem=self.smem_p1)
        p2b = prog['p2']
        if p2b['nblk'] == 0:
            return
        if p2b['union_ratio'] > p2_cublas_thresh:
            # Dense-union layer: direct cuBLAS on exact 0-masked H_full @ Wdn + perm residual add (100% bit-equivalent!)
            torch.mm(H_full, Wdn, out=y_tmp)
            n = H_full.shape[0] * self.dm
            self.add_res_perm(((n // 4 + 255) // 256,), (256,), [res, y_tmp, p2b['ord_tok'], H_full.shape[0], self.dm])
        else:
            self.pack_h((p2b['nblk'],), (256,), [
                H_full, p2b['kcols'], p2b['blk_desc'], p2b['h_off'], p2b['ncol_pad'],
                H_compact, p2b['nblk'], 32, self.nh
            ])
            self.p2((p2b['nblk'], self.dm // 128), (128,), [
                H_compact, Wdn, p2b['ord_tok'], p2b['kcols'], p2b['blk_desc'],
                p2b['h_off'], p2b['ncol_pad'], res, p2b['nblk']
            ], smem=self.smem_p2)


def build_v3_program(tiles_with_strip, active_mask_sorted, tok_sort_order, B, N=4096, dev='cuda'):
    """Build GPU tensors for P1_V3 (strip-ordered, LPT priority 64x32->32x32->32x16->16x16->8x16) + LPT-sorted P2."""
    pw = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    ord_tok = np.asarray(tok_sort_order, dtype=np.int32)
    inv_tok = np.empty(B, dtype=np.int32)
    inv_tok[ord_tok] = np.arange(B, dtype=np.int32)

    # Sort priority within each strip: 4(64x32) -> 0(32x32) -> 1(32x16) -> 2(16x16) -> 3(8x16)
    PRIO = {4: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    ordered = sorted(tiles_with_strip, key=lambda x: (x[0], PRIO[x[1]]))
    assigned = np.zeros((B, N), dtype=bool)

    tok_z_l, tok_h_l, neu_l, msk_l, tile_desc = [], [], [], [], []
    by_strip_type = {}
    to = no = 0

    for s, ttype, rr, cc in ordered:
        rr = np.asarray(rr, dtype=np.int32)
        cc = np.asarray(cc, dtype=np.int32)
        sub_act = active_mask_sorted[np.ix_(rr, cc)]
        sub_asg = assigned[np.ix_(rr, cc)]
        sub_own = sub_act & (~sub_asg)
        if not sub_own.any():
            continue
        assigned[np.ix_(rr, cc)] |= sub_own
        msk = (sub_own.astype(np.uint64) * pw[:len(cc)]).sum(axis=1).astype(np.uint32)

        tid = len(tile_desc)
        tok_z_l.append(rr)
        tok_h_l.append(inv_tok[rr])
        neu_l.append(cc)
        msk_l.append(msk)
        tile_desc.append((to, len(rr), no, len(cc)))
        to += len(rr)
        no += len(cc)
        by_strip_type.setdefault(s, {4: [], 0: [], 1: [], 2: [], 3: []})[ttype].append(tid)

    assert np.array_equal(assigned, active_mask_sorted), 'CRITICAL: assigned != active_mask_sorted!'

    cta_desc = []
    for s in sorted(by_strip_type.keys()):
        st = by_strip_type[s]
        for tid in st[4]:
            cta_desc.append((4, tid, -1, 0))
        for tid in st[0]:
            cta_desc.append((0, tid, -1, 0))
        t1 = st[1]
        for i in range(0, len(t1), 2):
            cta_desc.append((1, t1[i], t1[i + 1] if i + 1 < len(t1) else -1, 0))
        t2 = st[2]
        for i in range(0, len(t2), 2):
            cta_desc.append((2, t2[i], t2[i + 1] if i + 1 < len(t2) else -1, 0))
        t3 = st[3]
        if t3:
            assert t3[-1] - t3[0] + 1 == len(t3)
            for i in range(0, len(t3), 4):
                cta_desc.append((3, t3[i], min(t3[i] + 4, t3[-1] + 1), 0))

    # Build LPT-sorted (-ncol first!) P2 descriptors for TG=32
    act_ord = active_mask_sorted[ord_tok]
    raw_blks = []
    tot_union_cells = 0
    for h0 in range(0, B, 32):
        ntok = min(32, B - h0)
        cols = np.nonzero(act_ord[h0:h0 + ntok].any(axis=0))[0].astype(np.int32)
        if len(cols) > 0:
            npad = ((len(cols) + 31) // 32) * 32
            raw_blks.append((h0, ntok, cols, npad))
            tot_union_cells += ntok * npad
    # LPT sort: longest K blocks launch first -> zero wave tail!
    raw_blks.sort(key=lambda x: -x[3])

    kcols_l, blk_desc, h_offs, npads = [], [], [], []
    ko = ho = 0
    for h0, ntok, cols, npad in raw_blks:
        cpad = np.zeros(npad, dtype=np.int32)
        cpad[:len(cols)] = cols
        kcols_l.append(cpad)
        blk_desc.append((h0, ntok, ko, len(cols)))
        h_offs.append(ho)
        npads.append(npad)
        ko += npad
        ho += 32 * npad

    p2_info = dict(
        nblk=len(blk_desc),
        h_elems=max(128, ho),
        union_ratio=tot_union_cells / float(B * N),
        ord_tok=torch.as_tensor(ord_tok, device=dev),
        kcols=torch.as_tensor(np.concatenate(kcols_l) if kcols_l else np.zeros(32, np.int32), device=dev),
        blk_desc=torch.as_tensor(np.array(blk_desc, np.int32).reshape(-1, 4) if blk_desc else np.zeros((0, 4), np.int32), device=dev),
        h_off=torch.as_tensor(np.array(h_offs, np.int32) if h_offs else np.zeros(1, np.int32), device=dev),
        ncol_pad=torch.as_tensor(np.array(npads, np.int32) if npads else np.zeros(1, np.int32), device=dev),
    )

    return dict(
        ncta=len(cta_desc),
        tok_z=torch.as_tensor(np.concatenate(tok_z_l) if tok_z_l else np.zeros(1, np.int32), device=dev),
        tok_h=torch.as_tensor(np.concatenate(tok_h_l) if tok_h_l else np.zeros(1, np.int32), device=dev),
        neu=torch.as_tensor(np.concatenate(neu_l) if neu_l else np.zeros(1, np.int32), device=dev),
        wmask=torch.as_tensor(np.concatenate(msk_l).view(np.int32).copy() if msk_l else np.zeros(1, np.int32), device=dev),
        tiles=torch.as_tensor(np.array(tile_desc, np.int32).reshape(-1, 4) if tile_desc else np.zeros((0, 4), np.int32), device=dev),
        cta_desc=torch.as_tensor(np.array(cta_desc, np.int32).reshape(-1, 4) if cta_desc else np.zeros((0, 4), np.int32), device=dev),
        p2=p2_info,
    )


def main():
    from batch_s import load_keys, queues, RUNS_S, NN
    from batch_neon import build_encoded_tables
    from batch_t import load_net, MODEL
    from ffn_kernel import pack_bits
    from ms_solver import solve_layer_strips, get_ms_engine

    dev = 'cuda'
    keys, BM, pos = load_keys()
    enc_all = build_encoded_tables(BM)
    co_npz = np.load(os.path.join(RUNS_S, 'col_orders.npz'))
    col_orders = [co_npz[f'L{l}'] for l in range(4)]

    q0_256 = next(o for N, t, o in queues(keys.shape[1]) if N == 256 and t == 0)
    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)

    # 1. Numerical verification (both sparse P2 AND adaptive cuBLAS P2 path!)
    print('=== 1. V3 Numerical Verification (including 64x32 + adaptive P2) ===', flush=True)
    eng = get_ms_engine()
    for d in (1024, 4096):
        torch.manual_seed(0)
        rk = RosterV3Kernels(dm=d)
        print(f'  d={d}: p1_v3 regs={rk.p1.regs()}, smem={rk.smem_p1} | p2 regs={rk.p2.regs()}, smem={rk.smem_p2}', flush=True)
        B = 256
        Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
        Wg = torch.randn(NN, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wu = torch.randn(NN, d, device=dev, dtype=torch.bfloat16) * 0.05
        Wd = torch.randn(d, NN, device=dev, dtype=torch.bfloat16) * 0.05
        Wgu, Wdn = pack_sorted_weights(Wg, Wu, Wd, col_orders[0])
        kq = keys[0][q0_256]
        g_sorted = np.ascontiguousarray(enc_all[0][kq][:, col_orders[0]], dtype=np.int8)
        tiles = eng.solve_row_anchored(kq, g_sorted, strip_w=512, use_64x32=1, min_fill=(1024, 512, 256, 128), chain_rel=0.5, seed=42)
        cov = np.zeros_like(g_sorted, dtype=bool)
        for s, t, rr, cc in tiles: cov[np.ix_(rr, cc)] = True
        act = (g_sorted == 1) | ((g_sorted == 2) & cov)
        tok_ord = np.argsort(pos[0][kq], kind='stable')
        prog = build_v3_program(tiles, act, tok_ord, B, NN, dev=dev)
        H_full = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
        Hc = torch.empty(prog['p2']['h_elems'], device=dev, dtype=torch.bfloat16)
        yt = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
        inv = np.empty(NN, dtype=np.int32); inv[col_orders[0]] = np.arange(NN, dtype=np.int32)
        ref = ((torch.nn.functional.silu(Z.float() @ Wg.float().t()) * (Z.float() @ Wu.float().t())) *
               torch.as_tensor(act[:, inv], device=dev, dtype=torch.float32)) @ Wd.float().t()
        for thr, tag in [(1.01, 'sparse-P2'), (0.0, 'cublas-P2')]:
            res = torch.zeros(B, d, device=dev, dtype=torch.float32)
            rk.run_layer(Z, Wgu, Wdn, prog, H_full, Hc, yt, res, p2_cublas_thresh=thr)
            torch.cuda.synchronize()
            err = (res - ref).abs().max().item() / ref.abs().max().item()
            print(f'    d={d} {tag:10s}: max_rel_err={err:.2e} (PASS={err < 0.02})', flush=True)
            assert err < 0.02
        del Z, Wg, Wu, Wd, Wgu, Wdn, H_full, Hc, yt, prog
        torch.cuda.empty_cache()

    # 2. Full 4-layer benchmark: 4-frame CW_silicon vs WF_60, W=512 vs W=256 vs Global(W=4096), adaptive P2!
    def tg_fn(fn, hint_ms=2.0):
        inner = int(max(3, min(20, 30.0 / max(hint_ms, 0.1))))
        return time_graph(fn, reps=25, warm=4, inner=inner)[0]

    print('\n=== 2. V3 Real RTX 4060 4-Layer FFN Benchmark (B=256, 1024, 2048 x d=1024, 4096) ===', flush=True)
    for d in (1024, 4096):
        torch.manual_seed(0)
        rk = RosterV3Kernels(dm=d)
        model = load_net(MODEL) if d == 1024 else None
        L = []
        for l in range(4):
            if d == 1024:
                blk = model.blocks[l]
                g, u, dn = blk.gate.weight.detach(), blk.up.weight.detach(), blk.down.weight.detach()
            else:
                g = torch.randn(NN, d, device=dev) * 0.02
                u = torch.randn(NN, d, device=dev) * 0.02
                dn = torch.randn(d, NN, device=dev) * 0.02
            Wgu, Wdn = pack_sorted_weights(g, u, dn, col_orders[l])
            L.append(dict(
                Wst=torch.cat([g, u]).to(torch.bfloat16).contiguous(),
                Wd=dn.to(torch.bfloat16).contiguous(),
                Wgu=Wgu, Wdn=Wdn, bits=pack_bits(BM[l]),
            ))
            del g, u, dn
        del model
        torch.cuda.empty_cache()

        for B, idx in [(256, q0_256), (1024, q0_1024), (2048, q0_2048)]:
            Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
            res = torch.zeros(B, d, device=dev, dtype=torch.float32)
            gu = torch.empty(B, 2 * NN, device=dev, dtype=torch.bfloat16)
            a = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            y = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
            H_full = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            key_t = [torch.as_tensor(keys[l][idx].astype(np.int32), device=dev) for l in range(4)]

            def run_base():
                for l in range(4):
                    torch.mm(Z, L[l]['Wst'].t(), out=gu)
                    silu_mul(gu, a)
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

            def run_mask0():
                for l in range(4):
                    torch.mm(Z, L[l]['Wst'].t(), out=gu)
                    silu_mul_mask(gu, a, L[l]['bits'], key_t[l])
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

            tb = tg_fn(run_base, 2.0)
            tm = tg_fn(run_mask0, tb)
            print(f'\n[d={d}, B={B}] Base cuBLAS: {tb:.4f} ms (1.00x) | Unbundled 0-Mask: {tm:.4f} ms ({tb/tm:.2f}x)', flush=True)

            configs = [
                ('CW4_W512',  512,  'cw', (1024.0, 640.0, 486.0, 410.0)),
                ('CW4_W256',  256,  'cw', (1024.0, 640.0, 486.0, 410.0)),
                ('WF4_W512',  512,  'wf', (614, 307, 154)),
                ('CW4_GLOB',  4096, 'cw', (1024.0, 640.0, 486.0, 410.0)),
                ('RA5_W512',  512,  'ra5', (922, 460, 230, 128)),
            ]
            for name, W, kind, param in configs:
                progs = []
                tot_c = tot_a = 0
                max_hc = 128
                for l in range(4):
                    kq = keys[l][idx]
                    g_sorted = np.ascontiguousarray(enc_all[l][kq][:, col_orders[l]], dtype=np.int8)
                    if kind == 'cw':
                        tiles, act, st = solve_layer_strips(kq, g_sorted, W=W, frame_cost=param, seed=42 + l)
                        tot_c += st['compute']
                    elif kind == 'wf':
                        tiles, act, st = solve_layer_strips(kq, g_sorted, W=W, min_fill=param, frame_cost=None, seed=42 + l)
                        tot_c += st['compute']
                    else:
                        tiles = eng.solve_row_anchored(kq, g_sorted, strip_w=W, use_64x32=1, min_fill=param, chain_rel=0.45, seed=42 + l)
                        cov = np.zeros_like(g_sorted, dtype=bool)
                        for s, t, rr, cc in tiles: cov[np.ix_(rr, cc)] = True
                        act = (g_sorted == 1) | ((g_sorted == 2) & cov)
                        tot_c += sum({0:1024, 1:512, 2:256, 3:128, 4:2048}[t[1]] for t in tiles)
                    tok_ord = np.argsort(pos[l][kq], kind='stable')
                    pr = build_v3_program(tiles, act, tok_ord, B, NN, dev=dev)
                    progs.append(pr)
                    max_hc = max(max_hc, pr['p2']['h_elems'])
                    tot_a += int(act.sum())
                Hc = torch.empty(max_hc, device=dev, dtype=torch.bfloat16)

                t_sp = tg_fn(lambda: [rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], progs[l], H_full, Hc, y, res, p2_cublas_thresh=1.01) for l in range(4)], tb / 2)
                t_ad = tg_fn(lambda: [rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], progs[l], H_full, Hc, y, res, p2_cublas_thresh=0.62) for l in range(4)], tb / 2)
                tp1 = tg_fn(lambda: [(H_full.zero_(), rk.p1((progs[l]['ncta'],), (128,), [Z, L[l]['Wgu'], progs[l]['wmask'], progs[l]['tok_z'], progs[l]['tok_h'], progs[l]['neu'], progs[l]['tiles'], progs[l]['cta_desc'], H_full, progs[l]['ncta']], smem=rk.smem_p1)) for l in range(4)], t_sp / 2)
                comp_pct = 100.0 * tot_c / (4 * B * NN)
                fill_pct = 100.0 * tot_a / max(1, tot_c)
                print(
                    f'  {name:9s}: sparseP2 {t_sp:.4f} ms ({tb/t_sp:.2f}x) | adaptP2 {t_ad:.4f} ms ({tb/t_ad:.2f}x) '
                    f'[P1 {tp1:.4f} | P2_ad {t_ad-tp1:.4f}] | comp {comp_pct:5.2f}%, fill {fill_pct:4.1f}%',
                    flush=True,
                )
                del progs, Hc
            del Z, res, gu, a, y, H_full, key_t
            torch.cuda.empty_cache()
        del L
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
