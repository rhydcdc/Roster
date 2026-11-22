"""BATCHING - V3 커널: FFN을 두 단계로 쪼갠다 (NVRTC, 텐서 코어 mma.m16n8k16 bf16, 누적 fp32).
P1 (1단계 블록 커널): 직사각형(토큰 번호들 x 뉴런 번호들)마다
     C (토큰 x 열) = Z (토큰 x DM) . W^T,  열 = 뉴런마다 (gate, up) 번갈아 -> 같은 뉴런의 gate와 up이 한 스레드에 모인다
     H[토큰][뉴런] = silu(g) x u  (그 토큰의 목록에 그 뉴런이 있을 때만 적는다. H는 미리 0으로)
   출력은 토큰 하나에 뉴런 수 x 2바이트뿐 (한 커널판은 토큰 하나에 DM칸을 더했다).
2단계: H (N x 4096) . Wd^T 는 표준 행렬곱(cuBLAS) 하나.
그 밖: silu x up에 0 곱하기를 합친 커널(M0), 잔차에 더하기.
가중치: Wgu (4096, 2 DM): 뉴런마다 [gate 행 | up 행] (값은 그대로, 자리만).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kern import build                                                 # noqa: E402

SRC_P1 = r'''
#ifndef PROBE
#define PROBE 0
#endif
#define NTHR (32 * WM * WN)
#define WTM (TM / WM)          // 워프 조각의 토큰 수
#define WNB (NB / WN)          // 워프 조각의 열 수
#define MT (WTM / 16)
#define NT (WNB / 8)
#define SA (KC + 8)            // 공유 메모리 줄 폭 (bf16 칸), 뱅크 겹침을 피하려고 8칸 덧댐
#define STG ((TM + NB) * SA)   // 한 단: Z 조각 TM줄 + W 조각 NB줄
#define NSEG (KC / 8)
#define JZ (TM * NSEG / NTHR)  // 스레드 하나가 맡는 Z 조각(16바이트) 수
#define JW (NB * NSEG / NTHR)

__device__ __forceinline__ u32 smem_u32(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

extern "C" __global__ void __launch_bounds__(NTHR) p1(
    const bf16* __restrict__ Z,       // (토큰, DM)
    const bf16* __restrict__ W,       // (4096, 2 DM): gate | up
    const u32* __restrict__ wmask,    // tok과 나란히: 그 토큰이 이 일의 뉴런(32개까지) 가운데 켤 것의 비트
    const int* __restrict__ tok, const int* __restrict__ neu, const int4* __restrict__ work,
    const int* __restrict__ cbase,    // 일마다, 이 일의 첫 뉴런이 H에서 놓이는 자리
    bf16* __restrict__ H,             // (토큰, HW), 미리 0. 뉴런을 블록 순서로 붙여 둔다
    int HW, int nwork) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int w = blockIdx.x;
  if (w >= nwork) return;
  const int4 wk = work[w];
  const int ntok = wk.y, nneu = wk.w;
  const int cb = cbase[w];
  bf16* st = (bf16*)smem;
  int* s_tok = (int*)(st + STAGES * STG);
  u32* s_msk = (u32*)(s_tok + TM);
  int* s_neu = (int*)(s_msk + TM);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  for (int i = tid; i < TM; i += NTHR) { s_tok[i] = i < ntok ? tok[wk.x + i] : 0; s_msk[i] = i < ntok ? wmask[wk.x + i] : 0u; }
  for (int i = tid; i < NB / 2; i += NTHR) s_neu[i] = i < nneu ? neu[wk.z + i] : 0;
  __syncthreads();
  // 스레드마다 맡은 조각의 원천 주소와 자리 (레지스터에 둔다)
  const bf16* zsrc[JZ]; u32 zdst[JZ]; bool zok[JZ];
  const bf16* wsrc[JW]; u32 wdst[JW]; bool wok[JW];
  #pragma unroll
  for (int j = 0; j < JZ; j++) {
    int i = tid + j * NTHR, r = i / NSEG, c = i % NSEG;
    zok[j] = r < ntok;
    zsrc[j] = Z + (long)s_tok[r] * DM + c * 8;
    zdst[j] = smem_u32(st + r * SA + c * 8);
  }
  #pragma unroll
  for (int j = 0; j < JW; j++) {
    int i = tid + j * NTHR, r = i / NSEG, c = i % NSEG, n = r >> 1;
    wok[j] = n < nneu;
    wsrc[j] = W + (long)s_neu[n] * (2 * DM) + (r & 1) * DM + c * 8;
    wdst[j] = smem_u32(st + (TM + r) * SA + c * 8);
  }
  auto load = [&](int kc, int sidx) {
    const int so = kc * KC;
    const u32 dof = (u32)(sidx * STG * 2);
    #pragma unroll
    for (int j = 0; j < JZ; j++) if (zok[j]) cp16(zdst[j] + dof, zsrc[j] + so);
    #pragma unroll
    for (int j = 0; j < JW; j++) if (wok[j]) cp16(wdst[j] + dof, wsrc[j] + so);
  };
  const int wm = wid % WM, wn = wid / WM;
  int mt_lim = ((ntok + 15) >> 4) - wm * MT;            // 이 워프가 쓰는 토큰 m16 조각 수
  mt_lim = mt_lim < 0 ? 0 : (mt_lim > MT ? MT : mt_lim);
  int np_lim = (2 * nneu - wn * WNB + 15) >> 4;         // 이 워프가 쓰는 열 n8 쌍 수
  np_lim = np_lim < 0 ? 0 : (np_lim > NT / 2 ? NT / 2 : np_lim);
  const bool active = mt_lim > 0 && np_lim > 0;
  float acc[MT][NT][4];
  #pragma unroll
  for (int a = 0; a < MT; a++)
    #pragma unroll
    for (int b = 0; b < NT; b++)
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
  const int NK = DM / KC;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; s++) { load(s, s); cp_commit(); }
  for (int kc = 0; kc < NK; kc++) {
    cp_wait<STAGES - 2>();
    __syncthreads();
#if PROBE != 2
    { const int nk = kc + STAGES - 1; if (nk < NK) load(nk, nk % STAGES); cp_commit(); }
#else
    cp_commit();                                        // 진단: 읽기 없이 계산만 (결과 틀림)
#endif
#if PROBE != 1                                          // 진단 1: 계산 없이 읽기만 (결과 틀림)
    if (active) {
      const bf16* s = st + (kc % STAGES) * STG;
      #pragma unroll
      for (int ks = 0; ks < KC / 16; ks++) {
        u32 af[MT][4];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          if (mt >= mt_lim) break;
          ldm4(smem_u32(s + (wm * WTM + mt * 16 + (lane & 15)) * SA + ks * 16 + ((lane >> 4) << 3)),
               af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
        }
        #pragma unroll
        for (int np = 0; np < NT / 2; np++) {
          if (np >= np_lim) break;
          u32 b0, b1, b2, b3;
          ldm4(smem_u32(s + (TM + wn * WNB + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SA + ks * 16 + (((lane >> 3) & 1) << 3)),
               b0, b1, b2, b3);
          #pragma unroll
          for (int mt = 0; mt < MT; mt++) {
            if (mt >= mt_lim) break;
            mma(acc[mt][2 * np], af[mt], b0, b1);
            mma(acc[mt][2 * np + 1], af[mt], b2, b3);
          }
        }
      }
    }
#endif
  }
  // H = silu(g) x u x (목록에 있으면 1): 공유 메모리에 [토큰][뉴런]으로 모았다가, 토큰 줄마다 16바이트(뉴런 8개)씩 이어서 적는다
  cp_wait<0>();
  __syncthreads();
  bf16* hs = st;                                         // [TM][NB / 2], 계산이 끝난 조각 자리를 다시 쓴다
  {
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
      #pragma unroll
      for (int nt = 0; nt < NT; nt++)
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {
          const int t = wm * WTM + mt * 16 + gid + hf * 8;
          const int j = (wn * WNB + nt * 8) / 2 + tig;
          const bool on = t < ntok && j < nneu && ((s_msk[t] >> j) & 1u);
          hs[t * (NB / 2) + j] = on ? f2bf(silu(acc[mt][nt][hf * 2]) * acc[mt][nt][hf * 2 + 1]) : (bf16)0;
        }
  }
  __syncthreads();
  {
    const int nseg = (nneu + 7) >> 3;
    for (int i = tid; i < TM * (NB / 16); i += NTHR) {
      const int r = i / (NB / 16), c = i % (NB / 16);
      if (r < ntok && c < nseg) *(uint4*)(H + (long)s_tok[r] * HW + cb + c * 8) = *(const uint4*)(hs + r * (NB / 2) + c * 8);
    }
  }
}
'''

SRC_MISC = r'''
extern "C" __global__ void silu_mul(const bf16* __restrict__ gu, bf16* __restrict__ a, int N, int H) {
  // gu: N x 2H (앞 H = gate, 뒤 H = up), a: N x H. 스레드 하나가 8개씩.
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long per = H / 8;
  if (idx >= (long)N * per) return;
  long row = idx / per, c = idx % per;
  uint4 g = ((const uint4*)(gu + row * 2 * H))[c];
  uint4 u = ((const uint4*)(gu + row * 2 * H + H))[c];
  u32 gs[4] = {g.x, g.y, g.z, g.w}, us[4] = {u.x, u.y, u.z, u.w}, o[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float g0 = __uint_as_float(gs[i] << 16), g1 = __uint_as_float(gs[i] & 0xffff0000u);
    float u0 = __uint_as_float(us[i] << 16), u1 = __uint_as_float(us[i] & 0xffff0000u);
    o[i] = pack2(silu(g0) * u0, silu(g1) * u1);
  }
  ((uint4*)(a + row * H))[c] = make_uint4(o[0], o[1], o[2], o[3]);
}
extern "C" __global__ void silu_mul_mask(const bf16* __restrict__ gu, bf16* __restrict__ a, const u32* __restrict__ bits,
                                         const int* __restrict__ key, int N, int H) {
  // silu_mul과 같고, 토큰의 목록에 없는 뉴런은 0 (0 곱하기를 여기에 합친다)
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long per = H / 8;
  if (idx >= (long)N * per) return;
  long row = idx / per, c = idx % per;
  u32 m = (bits[key[row] * 128 + (c >> 2)] >> ((c & 3) * 8)) & 0xffu;
  uint4 g = ((const uint4*)(gu + row * 2 * H))[c];
  uint4 u = ((const uint4*)(gu + row * 2 * H + H))[c];
  u32 gs[4] = {g.x, g.y, g.z, g.w}, us[4] = {u.x, u.y, u.z, u.w}, o[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float g0 = __uint_as_float(gs[i] << 16), g1 = __uint_as_float(gs[i] & 0xffff0000u);
    float u0 = __uint_as_float(us[i] << 16), u1 = __uint_as_float(us[i] & 0xffff0000u);
    float h0 = ((m >> (2 * i)) & 1u) ? silu(g0) * u0 : 0.f;
    float h1 = ((m >> (2 * i + 1)) & 1u) ? silu(g1) * u1 : 0.f;
    o[i] = pack2(h0, h1);
  }
  ((uint4*)(a + row * H))[c] = make_uint4(o[0], o[1], o[2], o[3]);
}
extern "C" __global__ void add_res(float* __restrict__ h, const bf16* __restrict__ y, long n) {
  // 잔차(fp32)에 FFN 출력(bf16)을 더한다. 스레드 하나가 4칸.
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  float4 v = *(float4*)(h + i);
  uint2 p = *(const uint2*)(y + i);
  v.x += __uint_as_float(p.x << 16); v.y += __uint_as_float(p.x & 0xffff0000u);
  v.z += __uint_as_float(p.y << 16); v.w += __uint_as_float(p.y & 0xffff0000u);
  *(float4*)(h + i) = v;
}
'''

_misc = None


def misc():
    global _misc
    if _misc is None:
        _misc = build(SRC_MISC, ['silu_mul', 'silu_mul_mask', 'add_res'])
    return _misc


def silu_mul(gu, a):
    N, H2 = gu.shape
    misc()['silu_mul'](((N * (H2 // 2) // 8 + 255) // 256,), (256,), [gu, a, N, H2 // 2])


def silu_mul_mask(gu, a, bits, key):
    N, H2 = gu.shape
    misc()['silu_mul_mask'](((N * (H2 // 2) // 8 + 255) // 256,), (256,), [gu, a, bits, key, N, H2 // 2])


def add_res(h, y):
    n = h.numel()
    misc()['add_res'](((n // 4 + 255) // 256,), (256,), [h, y, n])


class P1:
    """1단계 블록 커널. TM: 일 하나의 토큰 수, NB: 열 수(= 뉴런 2배), WM x WN: 워프 배치."""

    def __init__(self, TM=128, NB=64, WM=4, WN=1, KC=32, STAGES=3, dm=1024, probe=0):
        nthr = 32 * WM * WN
        nseg = KC // 8
        assert (TM // WM) % 16 == 0 and (NB // WN) % 16 == 0 and KC % 16 == 0
        assert (TM * nseg) % nthr == 0 and (NB * nseg) % nthr == 0, '스레드마다 맡는 조각 수가 고르지 않다'
        assert NB <= 64, '목록 비트를 토큰마다 32비트 하나로 받는다 (일 하나의 뉴런은 32개까지)'
        self.TM, self.NB, self.NB2, self.nthr, self.dm = TM, NB, NB // 2, nthr, dm
        self.cfg = (TM, NB, WM, WN, KC, STAGES)
        k = build(SRC_P1, ['p1'], extra=['-DTM=%d' % TM, '-DNB=%d' % NB, '-DWM=%d' % WM, '-DWN=%d' % WN, '-DKC=%d' % KC,
                                         '-DSTAGES=%d' % STAGES, '-DDM=%d' % dm, '-DPROBE=%d' % probe])
        self.k = k['p1']
        self.smem = STAGES * (TM + NB) * (KC + 8) * 2 + 4 * (2 * TM + NB // 2)
        self.k.set_smem(self.smem)

    def __call__(self, Z, Wgu, prog, H):
        """prog: build_program의 표. H (토큰, HW)는 부르는 쪽이 미리 0으로 둔다."""
        self.k((prog['n'],), (self.nthr,),
               [Z, Wgu, prog['wmask'], prog['tok'], prog['neu'], prog['work'], prog['cbase'], H, prog['HW'], prog['n']], smem=self.smem)


SRC_P2 = r'''
// P2 (2단계 블록 커널): 잔차[토큰][d] += 합_k H[토큰][k] x Wd[뉴런(k)][d]
//   텐서 코어 방향: C^T (d x 토큰) = Wd^T (d x k) . H^T (k x 토큰). d = M(16 단위), 토큰 = N(8 단위).
//   일 = (토큰 조각, 쓸 뉴런 묶음(16개씩)의 목록 가운데 ks2개까지), CTA = 일 x d 조각 (blockIdx.y). 토큰 하나의 더하기는 (쓰는 뉴런 묶음 / ks2)번뿐이다.
//   k는 '블록 순서로 붙인 자리'를 16개 묶음 단위로 골라 온다: H는 그 자리에서 이어서, Wd는 cmap으로 뉴런 줄을 집어 온다.
#define NTHR2 (32 * WD * WT)
#define WDC (DC / WD)          // 워프의 d 수
#define WTT (TM / WT)          // 워프의 토큰 수
#define MT2 (WDC / 16)
#define NT2 (WTT / 8)
#define KC2 32                 // 한 걸음에 k 32개 (묶음 둘)
#define SB (DC + 8)            // Wd 조각 줄 폭 (줄 = k)
#define SHH (KC2 + 8)          // H 조각 줄 폭 (줄 = 토큰)
#define STG2 (KC2 * SB + TM * SHH)
#define JD (KC2 * (DC / 8) / NTHR2)
#define JH (TM * (KC2 / 8) / NTHR2)

__device__ __forceinline__ u32 smem_u32b(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16b(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commitb() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_waitb() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4b(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldm4tb(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mmab(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1)); }

extern "C" __global__ void __launch_bounds__(NTHR2) p2(
    const bf16* __restrict__ H,       // (토큰, HW): 뉴런값, 블록 순서로 붙인 자리
    const bf16* __restrict__ Wd,      // (4096, DM): 뉴런마다 down 열
    const int* __restrict__ cmap,     // 붙인 자리 -> 뉴런 번호 (HW개)
    const int* __restrict__ tok, const int* __restrict__ kgl, const int4* __restrict__ work,   // (tok_off, ntok, kg_off, nkg)
    float* __restrict__ res,          // (토큰, DM) 잔차, fp32
    int HW, int nwork) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int w = blockIdx.x;
  if (w >= nwork) return;
  const int4 wk = work[w];
  const int ntok = wk.y, nkg = wk.w;
  const int d0 = blockIdx.y * DC;
  bf16* st = (bf16*)smem;
  int* s_tok = (int*)(st + STAGES * STG2);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  for (int i = tid; i < TM; i += NTHR2) s_tok[i] = i < ntok ? tok[wk.x + i] : 0;
  __syncthreads();
  // 스레드마다 맡은 조각: Wd는 (k 줄, d 조각), H는 (토큰 줄, k 조각)
  int dr[JD], dc8[JD]; u32 ddst[JD];
  const bf16* hsrc[JH]; int hgrp[JH], hc8[JH]; u32 hdst[JH]; bool hok[JH];
  #pragma unroll
  for (int j = 0; j < JD; j++) {
    int i = tid + j * NTHR2;
    dr[j] = i / (DC / 8); dc8[j] = (i % (DC / 8)) * 8;
    ddst[j] = smem_u32b(st + dr[j] * SB + dc8[j]);
  }
  #pragma unroll
  for (int j = 0; j < JH; j++) {
    int i = tid + j * NTHR2, r = i / (KC2 / 8), c = i % (KC2 / 8);
    hok[j] = r < ntok; hgrp[j] = c >> 1; hc8[j] = (c & 1) * 8;
    hsrc[j] = H + (long)s_tok[r] * HW;
    hdst[j] = smem_u32b(st + KC2 * SB + r * SHH + c * 8);
  }
  const int NK2 = (nkg + 1) >> 1;
  auto load = [&](int kc, int sidx) {
    const u32 dof = (u32)(sidx * STG2 * 2);
    const int ga = kgl[wk.z + 2 * kc];
    const int gb = (2 * kc + 1 < nkg) ? kgl[wk.z + 2 * kc + 1] : -1;
    #pragma unroll
    for (int j = 0; j < JD; j++) {
      const int g = (dr[j] >> 4) ? gb : ga;
      if (g >= 0) cp16b(ddst[j] + dof, Wd + (long)cmap[g * 16 + (dr[j] & 15)] * DM + d0 + dc8[j]);
    }
    #pragma unroll
    for (int j = 0; j < JH; j++) {
      const int g = hgrp[j] ? gb : ga;
      if (hok[j] && g >= 0) cp16b(hdst[j] + dof, hsrc[j] + g * 16 + hc8[j]);
    }
  };
  const int wd = wid % WD, wt = wid / WD;
  int np_lim = (ntok - wt * WTT + 15) >> 4;             // 이 워프가 쓰는 토큰 n8 쌍 수
  np_lim = np_lim < 0 ? 0 : (np_lim > NT2 / 2 ? NT2 / 2 : np_lim);
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
    if (np_lim > 0) {
      const bf16* s = st + (kc % STAGES) * STG2;
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        if (2 * kc + ks >= nkg) break;
        u32 af[MT2][4];
        #pragma unroll
        for (int mt = 0; mt < MT2; mt++)
          ldm4tb(smem_u32b(s + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * SB + wd * WDC + mt * 16 + (((lane >> 3) & 1) << 3)),
                 af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
        #pragma unroll
        for (int np = 0; np < NT2 / 2; np++) {
          if (np >= np_lim) break;
          u32 b0, b1, b2, b3;
          ldm4b(smem_u32b(s + KC2 * SB + (wt * WTT + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * SHH + ks * 16 + (((lane >> 3) & 1) << 3)),
                b0, b1, b2, b3);
          #pragma unroll
          for (int mt = 0; mt < MT2; mt++) { mmab(acc[mt][2 * np], af[mt], b0, b1); mmab(acc[mt][2 * np + 1], af[mt], b2, b3); }
        }
      }
    }
  }
  // 잔차에 더한다 (같은 토큰 조각 x d 조각을 뉴런 쪽으로 쪼갠 일 몇 개가 함께 건드린다 -> 충돌 없는 더하기)
  if (np_lim > 0) {
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT2; mt++)
      #pragma unroll
      for (int nt = 0; nt < NT2; nt++) {
        if (nt >= 2 * np_lim) break;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          const int t = wt * WTT + nt * 8 + tig * 2 + (c & 1);
          if (t < ntok) {
            float* p = res + (long)s_tok[t] * DM + d0 + wd * WDC + mt * 16 + gid + ((c >> 1) << 3);
            asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(acc[mt][nt][c]));
          }
        }
      }
  }
}
'''


class P2:
    '''2단계 블록 커널. DC: d 조각, TM: 토큰 조각, WD x WT: 워프 배치 (d 쪽 x 토큰 쪽).'''

    def __init__(self, DC=128, TM=128, WD=2, WT=4, STAGES=2, dm=1024):
        nthr = 32 * WD * WT
        assert (DC // WD) % 16 == 0 and (TM // WT) % 16 == 0 and dm % DC == 0
        assert (32 * (DC // 8)) % nthr == 0 and (TM * 4) % nthr == 0, '스레드마다 맡는 조각 수가 고르지 않다'
        self.DC, self.TM, self.nthr, self.dm = DC, TM, nthr, dm
        self.cfg = (DC, TM, WD, WT, STAGES)
        k = build(SRC_P2, ['p2'], extra=['-DDC=%d' % DC, '-DTM=%d' % TM, '-DWD=%d' % WD, '-DWT=%d' % WT, '-DSTAGES=%d' % STAGES, '-DDM=%d' % dm])
        self.k = k['p2']
        self.smem = STAGES * (32 * (DC + 8) + TM * 40) * 2 + 4 * TM
        self.k.set_smem(self.smem)

    def __call__(self, H, Wd, prog, res):
        self.k((prog['n2'], self.dm // self.DC), (self.nthr,),
               [H, Wd, prog['cmap'], prog['tok2'], prog['kgl'], prog['work2'], res, prog['HW'], prog['n2']], smem=self.smem)


def build_program(plan, p1, order, tm2, ks2=32, Mq=None):
    '''계획 [(토큰 번호들, 뉴런 번호들)] -> 두 커널이 쓸 표.
    붙인 자리: 블록을 차례로, 블록마다 16의 배수로 덧대어 놓는다 (덧댄 자리의 H는 0으로 남는다).
    P1 일: 블록의 뉴런을 NB2개씩, 토큰을 TM 안쪽으로 고르게. 일마다 토큰마다 켤 뉴런의 비트(Mq로 만든다. 없으면 다 켬).
    P2 일: 토큰을 order(겹침 순서)대로 tm2개씩, 그 토큰들이 든 블록의 자리 묶음들을 ks2개씩 쪼갠 것.'''
    N = len(order)
    base, cmap, blocks = 0, [], []
    member = np.zeros((len(plan), N), bool)
    for b, (rows, cols) in enumerate(plan):
        cols = np.asarray(cols, np.int32)
        pad = (-len(cols)) % 16
        blocks.append((base, len(cols) + pad))
        cmap.append(cols)
        cmap.append(np.full(pad, cols[0], np.int32))
        member[b, np.asarray(rows)] = True
        base += len(cols) + pad
    HW = base
    items = []
    for (rows, cols), (b0, _) in zip(plan, blocks):
        rows = np.asarray(rows, np.int32)
        cols = np.asarray(cols, np.int32)
        nt = -(-len(rows) // p1.TM)
        per = -(-(-(-len(rows) // nt)) // 16) * 16
        for a in range(0, len(cols), p1.NB2):
            for b in range(0, len(rows), per):
                items.append((rows[b:b + per], cols[a:a + p1.NB2], b0 + a))
    items.sort(key=lambda it: -(-(-len(it[0]) // 16) * -(-len(it[1]) // 8)))
    toks, neus, wk, cb, msk = [], [], [], [], []
    to = no = 0
    pw = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    for rr, cc, c0 in items:
        toks.append(rr)
        neus.append(cc)
        if Mq is None:
            msk.append(np.full(len(rr), (1 << len(cc)) - 1, np.uint64))
        else:
            msk.append((Mq[np.ix_(rr, cc)].astype(np.uint64) * pw[:len(cc)]).sum(1))
        wk.append((to, len(rr), no, len(cc)))
        cb.append(c0)
        to += len(rr)
        no += len(cc)
    tok2, kgl, wk2 = [], [], []
    to = ko = 0
    cells2 = 0
    for s in range(0, N, tm2):
        tt = np.asarray(order[s:s + tm2], np.int32)
        used = np.nonzero(member[:, tt].any(1))[0]
        if len(used) == 0:
            continue
        kg = np.concatenate([np.arange(blocks[b][0] // 16, (blocks[b][0] + blocks[b][1]) // 16, dtype=np.int32) for b in used])
        nsp = -(-len(kg) // ks2)                                       # 뉴런 묶음을 고르게 쪼갠다 (일 수를 늘려 SM에 고르게)
        per = -(-len(kg) // nsp)
        tok2.append(tt)
        kgl.append(kg)
        for a in range(0, len(kg), per):
            wk2.append((to, len(tt), ko + a, min(per, len(kg) - a)))
        to += len(tt)
        ko += len(kg)
        cells2 += (-(-len(tt) // 8) * 8) * len(kg) * 16
    wk2.sort(key=lambda w: -(-(-w[1] // 8) * w[3]))
    dev = 'cuda'

    def g(a):
        return torch.as_tensor(np.ascontiguousarray(np.concatenate(a).astype(np.int32)), device=dev)
    wmask = torch.as_tensor(np.concatenate(msk).astype(np.uint32).view(np.int32).copy(), device=dev)
    return dict(HW=HW, cmap=g(cmap), tok=g(toks), neu=g(neus), wmask=wmask, work=torch.as_tensor(np.array(wk, np.int32).reshape(-1, 4), device=dev),
                cbase=torch.as_tensor(np.array(cb, np.int32), device=dev), n=len(wk),
                tok2=g(tok2), kgl=g(kgl), work2=torch.as_tensor(np.array(wk2, np.int32).reshape(-1, 4), device=dev), n2=len(wk2),
                visits=float(sum(w[1] for w in wk)),
                cells1=float(sum((-(-w[1] // 16) * 16) * (-(-w[3] // 8) * 8) for w in wk)), cells2=float(cells2))


def pack_gu(gate_w, up_w):
    """(4096, DM) 둘 -> (4096, 2 DM) bf16: 뉴런마다 gate 행 | up 행."""
    return torch.cat([gate_w, up_w], dim=1).detach().to(torch.bfloat16).contiguous()
