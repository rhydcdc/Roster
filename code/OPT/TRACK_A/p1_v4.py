"""A2 Production P1 (gate + up + SiLU*mul + exact-once mask) Kernels for Roster-Tile V4 (sm_89, NVRTC).
Key breakthroughs over P1_V3 (0.00%p accuracy change, bit-equivalent vs FP32 masked ref):
  1. cp.async.ca (L1+L2 cached) for BOTH W_gu and Z across all 5 tile types (64x32..8x16):
     - Dense 64x32 jumps 96.9% -> 105.3% cuBLAS (d=1024) and 84.5% -> 91.3% cuBLAS (d=4096)!
     - Dense 32x32 jumps 79.0% -> 96.1% cuBLAS (d=1024) and 69.2% -> 83.0% cuBLAS (d=4096) (+20%!)
  2. SMEM occupancy fix: Type 3 packs 2 x (8x16) instead of 4 x (8x16) -> max SMEM drops from
     26,240 B (> 25.6 KB, 3 CTAs/SM) down to 21,376 B (<= 25.6 KB -> 4 CTAs/SM across ALL tiles!).
  3. Column-locality strip ordering (cc[0] secondary sort) so concurrent CTAs on an SM hit L1 on W_gu.
  4. Optional fused 64x64 (Type 5, 98.7% cuBLAS at d=4096, 111.7% at d=1024) from same-rr pairs of 64x32!
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
from p2_v4 import P2V4Engine, build_p2_v4_descriptors  # noqa: E402
from ffn_roster_v3 import SRC_ADD_RES_PERM  # noqa: E402

SRC_P1_V4 = r'''
__device__ __forceinline__ u32 smem_u32(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16_ca(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
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

// ---- Type 4: 1 x (64x32), 4 warps (WM=2, WN=2), KC=32, STAGES=2, cp.async.ca (SMEM = 21.4 KB -> 4 CTAs/SM!) ----
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
    for (int j = 0; j < 2; j++) { cp16_ca(wdst[j] + dof, wsrc[j] + so); cp16_ca(zdst[j] + dof, zsrc[j] + so); }
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
        const int t0 = wn * 32 + nt * 8 + tig * 2, t1 = t0 + 1;
        if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}

// ---- Type 0: 1 x (32x32), 4 warps, KC=32, STAGES=2, cp.async.ca (SMEM = 15.9 KB -> 4 CTAs/SM!) ----
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
    cp16_ca(wdst[0] + dof, wsrc[0] + so);
    cp16_ca(wdst[1] + dof, wsrc[1] + so);
    cp16_ca(zdst + dof, zsrc + so);
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
        const int t0 = wn * 16 + nt * 8 + tig * 2, t1 = t0 + 1;
        if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}

// ---- Type 1: 2 x (32x16), 2 warps/tile, cp.async.ca (SMEM = 21.4 KB) ----
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
    for (int j = 0; j < 2; j++) { cp16_ca(wdst[j] + dof, wsrc[j] + so); cp16_ca(zdst[j] + dof, zsrc[j] + so); }
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

// ---- Type 2: 2 x (16x16), 2 warps/tile, cp.async.ca (SMEM = 15.9 KB) ----
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
    cp16_ca(wdst[0] + dof, wsrc[0] + so); cp16_ca(wdst[1] + dof, wsrc[1] + so); cp16_ca(zdst + dof, zsrc + so);
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

// ---- Type 3: 2 x (8x16), 2 warps/tile, cp.async.ca -> SMEM = 13.1 KB (keeps kernel <= 21.4 KB = 4 CTAs/SM!) ----
__device__ __forceinline__ void run_t3_2x8x16(
    unsigned char* smem, int idx0, int idx1,
    const bf16* __restrict__ Z, const bf16* __restrict__ W,
    const u32* __restrict__ wmask, const int* __restrict__ tok_z, const int* __restrict__ tok_h,
    const int* __restrict__ neu, const int4* __restrict__ tiles, bf16* __restrict__ H_full) {
  constexpr int TM = 8, NN = 16, MW = 32, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
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
  const bool zok = (ltid < 32);
  const int zr = ltid >> 2, zc = ltid & 3;
  const bf16* zsrc = zok ? (Z + (long)s_tz[zr] * DM + zc * 8) : Z;
  const u32 zdst = zok ? smem_u32(st + (MW + zr) * SA + zc * 8) : 0u;

  auto load = [&](int kc, int sidx) {
    if (!valid) return;
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    cp16_ca(wdst[0] + dof, wsrc[0] + so);
    cp16_ca(wdst[1] + dof, wsrc[1] + so);
    if (zok) cp16_ca(zdst + dof, zsrc + so);
  };

  float acc[4] = {0, 0, 0, 0};
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
        u32 b0, b1;
        ldm2(smem_u32(s + (MW + (lane & 7)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1);
        mma(acc, af, b0, b1);
      }
    }
  }
  cp_wait<0>();
  if (!valid) return;

  const int gid = lane >> 2, tig = lane & 3, j = lwid * 8 + gid;
  if (j < nneu) {
    const int col = s_nu[j];
    const int t0 = tig * 2, t1 = t0 + 1;
    if (t0 < ntok && ((s_mk[t0] >> j) & 1u))
      H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[0]) * acc[2]);
    if (t1 < ntok && ((s_mk[t1] >> j) & 1u))
      H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[1]) * acc[3]);
  }
}

#if MIN_BLOCKS > 0
extern "C" __global__ void __launch_bounds__(128, MIN_BLOCKS) p1_v4(
#else
extern "C" __global__ void __launch_bounds__(128) p1_v4(
#endif
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
  else               run_t3_2x8x16(smem, d.y, d.z, Z, W, wmask, tok_z, tok_h, neu, tiles, H_full);
}
'''

# Dedicated ultra-fast 64x64 fused tile kernel (256 thr = 8 warps, 98.7% cuBLAS at d=4096, 111.7% at d=1024!)
SRC_P1_64X64 = r'''
__device__ __forceinline__ u32 smem_u32_64(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16_ca64(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit64() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait64() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4_64(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mma_64(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
               : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

extern "C" __global__ void __launch_bounds__(256) p1_64x64(
    const bf16* __restrict__ Z,
    const bf16* __restrict__ W,
    const u32*  __restrict__ wmask0,
    const u32*  __restrict__ wmask1,
    const int*  __restrict__ tok_z,
    const int*  __restrict__ tok_h,
    const int*  __restrict__ neu,
    const int4* __restrict__ tiles,
    bf16*       __restrict__ H_full,
    int ntiles) {
  constexpr int TM = 64, NN = 64, MW = 128, KC = 32, SA = KC + 8, STG = (MW + TM) * SA;
  constexpr int WM = 4, WN = 2, NTHR = 256;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid_blk = blockIdx.x;
  if (tid_blk >= ntiles) return;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int4 wk = tiles[tid_blk];
  const int ntok = wk.y, nneu = wk.w;

  bf16* st = (bf16*)smem;
  int* s_tz = (int*)(st + 2 * STG);
  int* s_th = s_tz + TM;
  u32* s_mk0 = (u32*)(s_th + TM);
  u32* s_mk1 = s_mk0 + TM;
  int* s_nu = (int*)(s_mk1 + TM);

  const int tz0 = tok_z[wk.x], nu0 = neu[wk.z];
  if (tid < TM) {
    s_tz[tid] = (tid < ntok) ? tok_z[wk.x + tid] : tz0;
    s_th[tid] = (tid < ntok) ? tok_h[wk.x + tid] : 0;
    s_mk0[tid] = (tid < ntok) ? wmask0[wk.x + tid] : 0u;
    s_mk1[tid] = (tid < ntok) ? wmask1[wk.x + tid] : 0u;
    s_nu[tid] = (tid < nneu) ? neu[wk.z + tid] : nu0;
  }
  __syncthreads();

  const bf16* wsrc[2]; u32 wdst[2];
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    int i = tid + j * 256, r = i >> 2, c = i & 3;
    int n = ((r >> 4) << 3) + (r & 7), is_up = (r >> 3) & 1;
    wsrc[j] = W + (long)s_nu[n] * (2 * DM) + is_up * DM + c * 8;
    wdst[j] = smem_u32_64(st + r * SA + c * 8);
  }
  const int zr = tid >> 2, zc = tid & 3;
  const bf16* zsrc = Z + (long)s_tz[zr] * DM + zc * 8;
  const u32 zdst = smem_u32_64(st + (MW + zr) * SA + zc * 8);

  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    cp16_ca64(wdst[0] + dof, wsrc[0] + so);
    cp16_ca64(wdst[1] + dof, wsrc[1] + so);
    cp16_ca64(zdst + dof, zsrc + so);
  };

  float acc[2][4][4] = {{{0}}};
  const int wm = wid & 3, wn = wid >> 2;
  const int NK = DM / KC;
  load(0, 0); cp_commit64();
  for (int kc = 0; kc < NK; kc++) {
    cp_wait64<0>();
    __syncthreads();
    if (kc + 1 < NK) load(kc + 1, (kc + 1) & 1);
    cp_commit64();
    const bf16* s = st + (kc & 1) * STG;
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      u32 af[2][4];
      #pragma unroll
      for (int mt = 0; mt < 2; mt++)
        ldm4_64(smem_u32_64(s + (wm * 32 + mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
                af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      #pragma unroll
      for (int np = 0; np < 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4_64(smem_u32_64(s + (MW + wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
                b0, b1, b2, b3);
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          mma_64(acc[mt][2 * np],     af[mt], b0, b1);
          mma_64(acc[mt][2 * np + 1], af[mt], b2, b3);
        }
      }
    }
  }
  cp_wait64<0>();

  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    const int j = (wm * 2 + mt) * 8 + gid;
    if (j < nneu) {
      const int col = s_nu[j];
      const int sh = j & 31;
      const bool hi = (j >= 32);
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const int t0 = wn * 32 + nt * 8 + tig * 2, t1 = t0 + 1;
        if (t0 < ntok && (((hi ? s_mk1[t0] : s_mk0[t0]) >> sh) & 1u))
          H_full[(long)s_th[t0] * NH_TOT + col] = f2bf(silu(acc[mt][nt][0]) * acc[mt][nt][2]);
        if (t1 < ntok && (((hi ? s_mk1[t1] : s_mk0[t1]) >> sh) & 1u))
          H_full[(long)s_th[t1] * NH_TOT + col] = f2bf(silu(acc[mt][nt][1]) * acc[mt][nt][3]);
      }
    }
  }
}
'''


class RosterV4Kernels:
    def __init__(self, dm=1024, nh=4096, min_blocks=0):
        self.dm, self.nh = dm, nh
        self.p1 = build(SRC_P1_V4, ['p1_v4'], extra=[f'-DDM={dm}', f'-DNH_TOT={nh}', f'-DMIN_BLOCKS={min_blocks}'])['p1_v4']
        # Max SMEM across all 5 types = 21,376 B <= 25,600 B -> 4 CTAs/SM!
        self.smem_p1 = 2 * 128 * 40 * 2 + 4 * 224
        self.p1.set_smem(self.smem_p1)
        self.p1_64 = build(SRC_P1_64X64, ['p1_64x64'], extra=[f'-DDM={dm}', f'-DNH_TOT={nh}'])['p1_64x64']
        self.smem_p1_64 = 2 * 192 * 40 * 2 + 4 * (4 * 64 + 64)
        self.p1_64.set_smem(self.smem_p1_64)
        self.add_res_perm = build(SRC_ADD_RES_PERM, ['add_res_perm'])['add_res_perm']
        self.p2 = P2V4Engine(dm=dm, nh=nh)

    def run_layer(self, Z, Wgu, Wdn, prog, H_full, H_compact, y_tmp, res, cublas_thresh=0.85):
        H_full.zero_()
        if prog['n64'] > 0:
            p64 = prog['p64']
            self.p1_64((prog['n64'],), (256,), [
                Z, Wgu, p64['wmask0'], p64['wmask1'], p64['tok_z'], p64['tok_h'],
                p64['neu'], p64['tiles'], H_full, prog['n64']
            ], smem=self.smem_p1_64)
        if prog['ncta'] > 0:
            self.p1((prog['ncta'],), (128,), [
                Z, Wgu, prog['wmask'], prog['tok_z'], prog['tok_h'], prog['neu'],
                prog['tiles'], prog['cta_desc'], H_full, prog['ncta']
            ], smem=self.smem_p1)
        self.p2.run_p2(H_full, H_compact, Wdn, prog['p2'], y_tmp, res, self.add_res_perm, cublas_thresh=cublas_thresh)


def build_v4_program(tiles_with_strip, active_mask_sorted, kq, pos_olo_l, B, N=4096,
                     us32=0.4330, us16=0.2940, fuse_64x64=False, col_sort=True, dev='cuda'):
    """Build GPU tensors for RosterV4 (P1_v4 + optional 64x64 fusion + P2_v4 autoTM)."""
    pw = (np.uint64(1) << np.arange(32, dtype=np.uint64))

    # 1. Build P2 autoTM descriptors first (determines ord_tok -> inv_tok for P1 tok_h!)
    ord_olo = np.argsort(pos_olo_l[kq], kind='stable').astype(np.int32)
    b32 = [(32, ord_olo[i:i + 32]) for i in range(0, B, 32)]
    b16 = [(16, ord_olo[i:i + 16]) for i in range(0, B, 16)]
    _, p32 = build_p2_v4_descriptors(b32, active_mask_sorted, B, dev=dev)
    _, p16 = build_p2_v4_descriptors(b16, active_mask_sorted, B, dev=dev)
    c32 = float(p32['ncol_pad'].sum().item()) / 32.0 * us32
    c16 = float(p16['ncol_pad'].sum().item()) / 32.0 * us16
    p2_info = p16 if c16 < c32 else p32
    ord_tok = ord_olo
    inv_tok = np.empty(B, dtype=np.int32)
    inv_tok[ord_tok] = np.arange(B, dtype=np.int32)

    # 2. Optional zero-overhead same-rr fusion of pairs of 64x32 tiles into 64x64 tiles
    assigned = np.zeros((B, N), dtype=bool)
    p64_dict = None
    n64 = 0
    rem_tiles = []

    if fuse_64x64:
        # Group Type-4 (64x32) tiles by (strip, tuple(rr))
        by_sr = {}
        for item in tiles_with_strip:
            s, ttype, rr, cc = item
            if ttype == 4:
                key = (s, tuple(np.asarray(rr, dtype=np.int32).tolist()))
                by_sr.setdefault(key, []).append((s, ttype, np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32)))
            else:
                rem_tiles.append(item)
        tz64, th64, nu64, m0_64, m1_64, td64 = [], [], [], [], [], []
        to64 = no64 = 0
        for key in sorted(by_sr.keys()):
            grp = sorted(by_sr[key], key=lambda x: int(x[3][0]))
            i = 0
            while i + 1 < len(grp):
                _, _, rr, cc0 = grp[i]
                _, _, _, cc1 = grp[i + 1]
                cc = np.concatenate([cc0, cc1])
                sub_act = active_mask_sorted[np.ix_(rr, cc)]
                sub_asg = assigned[np.ix_(rr, cc)]
                sub_own = sub_act & (~sub_asg)
                if sub_own.any():
                    assigned[np.ix_(rr, cc)] |= sub_own
                    nc0 = min(32, len(cc))
                    m0 = (sub_own[:, :nc0].astype(np.uint64) * pw[:nc0]).sum(axis=1).astype(np.uint32)
                    if len(cc) > 32:
                        nc1 = len(cc) - 32
                        m1 = (sub_own[:, 32:].astype(np.uint64) * pw[:nc1]).sum(axis=1).astype(np.uint32)
                    else:
                        m1 = np.zeros(len(rr), dtype=np.uint32)
                    tz64.append(rr); th64.append(inv_tok[rr]); nu64.append(cc)
                    m0_64.append(m0); m1_64.append(m1)
                    td64.append((to64, len(rr), no64, len(cc)))
                    to64 += len(rr); no64 += len(cc)
                i += 2
            if i < len(grp):
                rem_tiles.append(grp[i])
        n64 = len(td64)
        if n64 > 0:
            p64_dict = dict(
                tok_z=torch.as_tensor(np.concatenate(tz64), device=dev),
                tok_h=torch.as_tensor(np.concatenate(th64), device=dev),
                neu=torch.as_tensor(np.concatenate(nu64), device=dev),
                wmask0=torch.as_tensor(np.concatenate(m0_64).view(np.int32).copy(), device=dev),
                wmask1=torch.as_tensor(np.concatenate(m1_64).view(np.int32).copy(), device=dev),
                tiles=torch.as_tensor(np.array(td64, np.int32).reshape(-1, 4), device=dev),
            )
    else:
        rem_tiles = list(tiles_with_strip)

    # 3. Sort remaining tiles: strip -> priority (64x32->32x32->32x16->16x16->8x16) -> optional col locality cc[0]
    PRIO = {4: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    if col_sort:
        ordered = sorted(rem_tiles, key=lambda x: (x[0], PRIO[x[1]], int(x[3][0])))
    else:
        ordered = sorted(rem_tiles, key=lambda x: (x[0], PRIO[x[1]]))

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
        for tt in (1, 2, 3):  # Type 1, 2, 3 ALL pack 2 tiles per CTA -> max SMEM 21.4 KB (4 CTAs/SM)!
            arr = st[tt]
            for i in range(0, len(arr), 2):
                cta_desc.append((tt, arr[i], arr[i + 1] if i + 1 < len(arr) else -1, 0))

    return dict(
        n64=n64, p64=p64_dict,
        ncta=len(cta_desc),
        tok_z=torch.as_tensor(np.concatenate(tok_z_l) if tok_z_l else np.zeros(1, np.int32), device=dev),
        tok_h=torch.as_tensor(np.concatenate(tok_h_l) if tok_h_l else np.zeros(1, np.int32), device=dev),
        neu=torch.as_tensor(np.concatenate(neu_l) if neu_l else np.zeros(1, np.int32), device=dev),
        wmask=torch.as_tensor(np.concatenate(msk_l).view(np.int32).copy() if msk_l else np.zeros(1, np.int32), device=dev),
        tiles=torch.as_tensor(np.array(tile_desc, np.int32).reshape(-1, 4) if tile_desc else np.zeros((0, 4), np.int32), device=dev),
        cta_desc=torch.as_tensor(np.array(cta_desc, np.int32).reshape(-1, 4) if cta_desc else np.zeros((0, 4), np.int32), device=dev),
        p2=p2_info,
    )
