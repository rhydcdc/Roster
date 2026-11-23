"""Parameterized High-Throughput Zero-Atomic P2 (down + residual) & Fast pack_h CUDA Kernels (sm_89, NVRTC).
Supports:
  - TM in {8, 16, 32, 64} (100% full mma.m16n8k16 on ALL heights, ldm2 for TM=8 / NT2=1, ldm4 for NT2>=2)
  - DC in {128, 256}, KC2 in {16, 32}, STAGES in {2, 3}
  - Optional cp.async cache policy: .cg (L2-only) vs .ca (L1+L2)
  - Fast 2D-grid pack_h_v4 (zero integer div/mod, vectorized 16B stores, supports variable TM per block!)
  - Unified multi-height P2 kernel (p2_unified_v4) launching TM in {64, 32, 16, 8} in ONE kernel launch!
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

# Fast pack_h supporting variable TM per block (blk_tm[b] in {8, 16, 32, 64}) and 2D grid (nblk, max_k_tiles)
# Eliminates integer division and under-occupied 1D grid when nblk is small (e.g. B=512).
SRC_PACK_H_V4 = r'''
extern "C" __global__ void pack_h_v4(
    const bf16* __restrict__ H_full,
    const int*  __restrict__ kcols,
    const int4* __restrict__ blk_desc,  // (h0, ntok, koff, ncol)
    const int*  __restrict__ h_off,     // element offset in H_compact
    const int*  __restrict__ ncol_pad,  // padded width (multiple of 16 or 32)
    const int*  __restrict__ blk_tm,    // TM for block b (8, 16, 32, or 64)
    bf16*       __restrict__ H_compact,
    int nblk, int NH_TOT) {
  const int b = blockIdx.x;
  if (b >= nblk) return;
  const int4 bd = blk_desc[b];
  const int h0 = bd.x, ntok = bd.y, koff = bd.z, ncol = bd.w;
  const int npad = ncol_pad[b];
  const int k_base = blockIdx.y * 64;
  if (k_base >= npad) return;
  const int tm = blk_tm[b];
  bf16* dst = H_compact + h_off[b];

  // Block has 256 threads: 32 along k (2 passes for 64 cols) x 8 along row (strided over tm)
  // Better: 64 threads along k (lane_k = tid & 63), 4 warps/sub-rows (r0 = tid >> 6)
  const int tid = threadIdx.x;
  const int k_rel = tid & 63;
  const int r_sub = tid >> 6;  // 0..3
  const int k = k_base + k_rel;
  if (k >= npad) return;
  const int col = (k < ncol) ? kcols[koff + k] : -1;
  for (int r = r_sub; r < tm; r += 4) {
    bf16 v = (bf16)0;
    if (r < ntok && col >= 0) {
      v = H_full[(long)(h0 + r) * NH_TOT + col];
    }
    dst[(long)r * npad + k] = v;
  }
}
'''

# Parameterized standalone P2 kernel for micro-benchmarking any (TM, DC, WD, WT, KC2, STAGES, USE_CA)
SRC_P2_PARAM = r'''
#define NTHR2 (32 * WD * WT)
#define WDC (DC / WD)
#define WTT (TM / WT)
#define MT2 (WDC / 16)
#define NT2 (WTT / 8)
#define SB (DC + 8)
#define SHH (KC2 + 8)
#define STG2 (KC2 * SB + TM * SHH)
#define D_CP (KC2 * (DC / 8))
#define H_CP (TM * (KC2 / 8))
#define JD ((D_CP + NTHR2 - 1) / NTHR2)
#define JH ((H_CP + NTHR2 - 1) / NTHR2)
#define PIPE_BYTES (STAGES * STG2 * 2)
#define OUT_BYTES (TM * DC * 4)
#define OTOK_OFF ((PIPE_BYTES > OUT_BYTES) ? PIPE_BYTES : OUT_BYTES)

__device__ __forceinline__ u32 smem_u32b(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
#if USE_CA
__device__ __forceinline__ void cp16w(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
#else
__device__ __forceinline__ void cp16w(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
#endif
__device__ __forceinline__ void cp16h(u32 dst, const void* src) {
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

extern "C" __global__ void __launch_bounds__(NTHR2) p2_param(
    const bf16* __restrict__ H_compact,
    const bf16* __restrict__ Wd,
    const int*  __restrict__ ord_tok,
    const int*  __restrict__ kcols,
    const int4* __restrict__ blk_desc,  // (h0, ntok, koff, ncol)
    const int*  __restrict__ h_off,
    const int*  __restrict__ ncol_pad,
    float*      __restrict__ res,
    int nblk) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int b = blockIdx.x;
  if (b >= nblk) return;
  const int4 bd = blk_desc[b];
  const int h0 = bd.x, ntok = bd.y, koff = bd.z;
  const int npad = ncol_pad[b];
  if (npad <= 0) return;
  const bf16* hbase = H_compact + h_off[b];
  const int d0 = blockIdx.y * DC;

  bf16* st = (bf16*)smem;
  int* s_otok = (int*)(smem + OTOK_OFF);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  for (int i = tid; i < TM; i += NTHR2) s_otok[i] = (i < ntok) ? ord_tok[h0 + i] : 0;
  __syncthreads();

  int dr[JD], dc8[JD]; u32 ddst[JD];
  #pragma unroll
  for (int j = 0; j < JD; j++) {
    int i = tid + j * NTHR2;
    if (i < D_CP) {
      dr[j] = i / (DC / 8);
      dc8[j] = (i % (DC / 8)) * 8;
      ddst[j] = smem_u32b(st + dr[j] * SB + dc8[j]);
    } else {
      dr[j] = -1; dc8[j] = 0; ddst[j] = 0;
    }
  }
  const bf16* hsrc[JH]; u32 hdst[JH]; bool hok[JH];
  #pragma unroll
  for (int j = 0; j < JH; j++) {
    int i = tid + j * NTHR2;
    if (i < H_CP) {
      int r = i / (KC2 / 8), c = i % (KC2 / 8);
      hsrc[j] = hbase + (long)r * npad + c * 8;
      hdst[j] = smem_u32b(st + KC2 * SB + r * SHH + c * 8);
      hok[j] = true;
    } else {
      hsrc[j] = hbase; hdst[j] = 0; hok[j] = false;
    }
  }

  const int NK = npad / KC2;
  auto load = [&](int kc, int sidx) {
    const int kb = kc * KC2;
    const u32 dof = (u32)(sidx * STG2 * 2);
    #pragma unroll
    for (int j = 0; j < JD; j++) {
      if (dr[j] >= 0) {
        const int nid = kcols[koff + kb + dr[j]];
        cp16w(ddst[j] + dof, Wd + (long)nid * DM + d0 + dc8[j]);
      }
    }
    #pragma unroll
    for (int j = 0; j < JH; j++) {
      if (hok[j]) cp16h(hdst[j] + dof, hsrc[j] + kb);
    }
  };

  float acc[MT2][NT2][4];
  #pragma unroll
  for (int a = 0; a < MT2; a++)
    #pragma unroll
    for (int bi = 0; bi < NT2; bi++)
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[a][bi][c] = 0.f;

  const int wd = wid % WD, wt = wid / WD;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { if (s < NK) load(s, s); cp_commitb(); }
  for (int kc = 0; kc < NK; kc++) {
    cp_waitb<STAGES - 2>();
    __syncthreads();
    { const int nk = kc + STAGES - 1; if (nk < NK) load(nk, nk % STAGES); cp_commitb(); }
    const bf16* s = st + (kc % STAGES) * STG2;
    #pragma unroll
    for (int ks = 0; ks < (KC2 / 16); ks++) {
      u32 af[MT2][4];
      #pragma unroll
      for (int mt = 0; mt < MT2; mt++)
        ldm4tb(smem_u32b(s + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * SB + wd * WDC + mt * 16 + (((lane >> 3) & 1) << 3)),
               af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
#if (NT2 == 1)
      u32 b0, b1;
      ldm2b(smem_u32b(s + KC2 * SB + (wt * WTT + (lane & 7)) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)), b0, b1);
      #pragma unroll
      for (int mt = 0; mt < MT2; mt++) mmab(acc[mt][0], af[mt], b0, b1);
#else
      #pragma unroll
      for (int np = 0; np < NT2 / 2; np++) {
        u32 b0, b1, b2, b3;
        ldm4b(smem_u32b(s + KC2 * SB + (wt * WTT + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)),
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

  float* s_out = (float*)st;
  const int gid = lane >> 2, tig = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < MT2; mt++)
    #pragma unroll
    for (int nt = 0; nt < NT2; nt++)
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        const int t = wt * WTT + nt * 8 + tig * 2 + (c & 1);
        const int d = wd * WDC + mt * 16 + gid + ((c >> 1) << 3);
        s_out[t * DC + d] = acc[mt][nt][c];
      }
  __syncthreads();

  // Vectorized float4 coalesced residual add (DC is multiple of 128 -> DC/4 >= 32 float4s per row)
  const int dc4 = DC >> 2;
  for (int i = tid; i < ntok * dc4; i += NTHR2) {
    const int r = i / dc4;
    const int c4 = (i % dc4) << 2;
    float4* dst = (float4*)(res + (long)s_otok[r] * DM + d0 + c4);
    float4 addv = *(const float4*)(s_out + r * DC + c4);
    float4 cur = *dst;
    cur.x += addv.x; cur.y += addv.y; cur.z += addv.z; cur.w += addv.w;
    *dst = cur;
  }
}
'''


def build_p2_param(dm, tm, dc=128, wd=2, wt=2, kc2=32, stages=3, use_ca=0):
    k = build(SRC_P2_PARAM, ['p2_param'], extra=[
        f'-DDM={dm}', f'-DTM={tm}', f'-DDC={dc}', f'-DWD={wd}', f'-DWT={wt}',
        f'-DKC2={kc2}', f'-DSTAGES={stages}', f'-DUSE_CA={use_ca}',
    ])['p2_param']
    stg2 = kc2 * (dc + 8) + tm * (kc2 + 8)
    pipe_b = stages * stg2 * 2
    out_b = tm * dc * 4
    smem = max(pipe_b, out_b) + 4 * tm
    k.set_smem(smem)
    nthr = 32 * wd * wt
    return k, smem, nthr


def build_pack_h_v4():
    return build(SRC_PACK_H_V4, ['pack_h_v4'])['pack_h_v4']


def make_p2_buffers(blocks, active_mask_sorted, B, pad_k=32, dev='cuda'):
    """Convert list of (tm, orig_tok_indices) into GPU descriptors for pack_h_v4 + p2 kernels.
    Note: blocks define the token order ord_tok (concatenation of all block token lists).
    Returns:
      - ord_tok, inv_tok (needed to build P1 tok_h matching P2's row order!)
      - p2 dict with LPT-sorted blocks (-npad first), per-block blk_tm, and stats.
    """
    # Concatenate block tokens -> defines exact row order in H_full!
    ord_list = []
    block_ranges = []
    cur = 0
    for tm, idxs in blocks:
        arr = np.asarray(idxs, dtype=np.int32)
        if len(arr) == 0:
            continue
        assert len(arr) <= tm
        ord_list.append(arr)
        block_ranges.append((tm, cur, len(arr)))
        cur += len(arr)
    assert cur == B, f'Expected {B} tokens, got {cur}'
    ord_tok = np.concatenate(ord_list).astype(np.int32)
    inv_tok = np.empty(B, dtype=np.int32)
    inv_tok[ord_tok] = np.arange(B, dtype=np.int32)

    act_ord = active_mask_sorted[ord_tok]
    raw_blks = []
    tot_union_pad = 0
    tot_union_raw = 0
    max_npad = 0
    for tm, h0, ntok in block_ranges:
        cols = np.nonzero(act_ord[h0:h0 + ntok].any(axis=0))[0].astype(np.int32)
        if len(cols) > 0:
            npad = ((len(cols) + pad_k - 1) // pad_k) * pad_k
            raw_blks.append((tm, h0, ntok, cols, npad))
            tot_union_pad += tm * npad
            tot_union_raw += ntok * len(cols)
            if npad > max_npad:
                max_npad = npad

    # LPT sort: longest npad first!
    raw_blks.sort(key=lambda x: (-x[4], -x[0]))
    kcols_l, blk_desc, h_offs, npads, tms = [], [], [], [], []
    ko = ho = 0
    for tm, h0, ntok, cols, npad in raw_blks:
        cpad = np.zeros(npad, dtype=np.int32)
        cpad[:len(cols)] = cols
        kcols_l.append(cpad)
        blk_desc.append((h0, ntok, ko, len(cols)))
        h_offs.append(ho)
        npads.append(npad)
        tms.append(tm)
        ko += npad
        ho += tm * npad

    N = active_mask_sorted.shape[1]
    p2 = dict(
        nblk=len(blk_desc),
        h_elems=max(128, ho),
        max_npad=max_npad,
        union_ratio=tot_union_pad / float(B * N),
        union_raw_ratio=tot_union_raw / float(B * N),
        ord_tok=torch.as_tensor(ord_tok, device=dev),
        kcols=torch.as_tensor(np.concatenate(kcols_l) if kcols_l else np.zeros(pad_k, np.int32), device=dev),
        blk_desc=torch.as_tensor(np.array(blk_desc, np.int32).reshape(-1, 4) if blk_desc else np.zeros((0, 4), np.int32), device=dev),
        h_off=torch.as_tensor(np.array(h_offs, np.int32) if h_offs else np.zeros(1, np.int32), device=dev),
        ncol_pad=torch.as_tensor(np.array(npads, np.int32) if npads else np.zeros(1, np.int32), device=dev),
        blk_tm=torch.as_tensor(np.array(tms, np.int32) if tms else np.zeros(1, np.int32), device=dev),
    )
    return ord_tok, inv_tok, p2
