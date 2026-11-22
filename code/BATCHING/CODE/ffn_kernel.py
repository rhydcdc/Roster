"""BATCHING - V2: 직사각형 계획(요청 번호들 x 뉴런 번호들의 목록)을 그대로 받아 FFN을 계산하는 전용 커널 (NVRTC, 텐서 코어 mma.m16n8k16 bf16).
계획이 무엇이든(뉴런 쪽 블록, b_sim16, 무리 G) 같은 커널로 돌려 시간을 견준다.

일 하나(CTA 하나) = 직사각형의 토큰 최대 TT개 x 뉴런 최대 NC개:
  1단계: G, U (뉴런 x 토큰) = Wg, Wu (뉴런 x 1,024) . Z^T (1,024 x 토큰)  [텐서 코어, 뉴런 = M(16 단위), 토큰 = N(8 단위)]
         H = silu(G) x U x (토큰의 목록에 그 뉴런이 있으면 1, 없으면 0)  -> bf16, 공유 메모리에
  2단계: Y^T (1,024 x 토큰) += Wd^T (1,024 x 뉴런) . H (뉴런 x 토큰)  -> fp32 출력에 더하기(red.global.add)
  빈 토큰 조각(8개)과 빈 뉴런 조각(16개)은 계산하지 않는다.
가중치는 뉴런마다 [gate 행 1,024 | up 행 1,024 | down 열 1,024]를 붙여 둔다 (값은 그대로, 자리만).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kern import build                                                 # noqa: E402

D = 1024

SRC = r'''
#ifndef DM
#define DM 1024
#endif
#ifndef OUTMODE
#define OUTMODE 0
#endif
#define NWARP 4
#define NTHR 128
#define KC 64
#define SK (KC + 8)          // 1단계 공유 메모리 줄 폭 (bf16 칸), 뱅크 겹침을 피하려고 8칸 덧댐
#define DC 128
#define SD (DC + 8)          // 2단계 Wd 줄 폭
#define SH (NC + 8)          // H 줄 폭

__device__ __forceinline__ u32 smem_u32(const void* p) {
  u32 r; asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cp16(u32 dst, const void* src, bool ok) {
  int n = ok ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldm4t(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a)); }
__device__ __forceinline__ void mma(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }

// 1단계 워프 배치: 워프 WM x WN, 워프마다 뉴런 m16 조각 MT개 x 토큰 n8 조각 NT개
#define NC (16 * MT * WM)
#define TT (8 * NT * WN)

extern "C" __global__ void __launch_bounds__(NTHR) ffn_rect(
    const bf16* __restrict__ Z,       // (토큰, DM)
    const bf16* __restrict__ Wp,      // (4096, 3*DM): gate | up | down열
    const u32* __restrict__ bits,     // (256, 128): 중심점마다 켤 뉴런 비트
    const int* __restrict__ key,      // (토큰,) 간 중심점
    const int* __restrict__ tok,      // 일마다 토큰 번호들 (이어 붙임)
    const int* __restrict__ neu,      // 일마다 뉴런 번호들
    const int4* __restrict__ work,    // (tok_off, ntok, neu_off, nneu)
    float* __restrict__ Y,            // (토큰, DM) fp32, 더한다
    int nwork) {
  extern __shared__ __align__(16) unsigned char smem[];
  int w = blockIdx.x;
  if (w >= nwork) return;
  int4 wk = work[w];
  const int ntok = wk.y, nneu = wk.w;
  bf16* buf = (bf16*)smem;                                    // 1단계: [2][(2NC + TT) x SK], 2단계: [2][NC x SD]
  const int S1 = (2 * NC + TT) * SK;
  const int S2 = NC * SD;
  const int BUFSZ = (2 * S1 > 2 * S2) ? 2 * S1 : 2 * S2;
  bf16* Hs = buf + BUFSZ;                                     // [TT][SH]
  int* s_tok = (int*)(Hs + TT * SH);                          // [TT]
  int* s_key = s_tok + TT;                                    // [TT]
  int* s_neu = s_key + TT;                                    // [NC]
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  for (int i = tid; i < TT; i += NTHR) { int t = i < ntok ? tok[wk.x + i] : 0; s_tok[i] = t; s_key[i] = i < ntok ? key[t] : 0; }
  for (int i = tid; i < NC; i += NTHR) s_neu[i] = i < nneu ? neu[wk.z + i] : 0;
  __syncthreads();
  const int ntile_t = (ntok + 7) >> 3;                        // 쓰는 토큰 n8 조각 수
  const int mtile_n = (nneu + 15) >> 4;                       // 쓰는 뉴런 m16 조각 수
  const int wm = wid % WM, wn = wid / WM;

  // ---- 1단계 ----
  auto load1 = [&](int kc, int b) {
    bf16* s = buf + b * S1;
    const int k0 = kc * KC;
    // gate 행, up 행: 2NC줄 x 8조각(16바이트)
    for (int i = tid; i < 2 * NC * 8; i += NTHR) {
      int r = i >> 3, c = i & 7;
      int n = r < NC ? r : r - NC;
      bool ok = n < nneu;
      const bf16* src = Wp + (long)s_neu[n] * (3 * DM) + (r < NC ? 0 : DM) + k0 + c * 8;
      cp16(smem_u32(s + r * SK + c * 8), src, ok);
    }
    for (int i = tid; i < TT * 8; i += NTHR) {
      int r = i >> 3, c = i & 7;
      bool ok = r < ntok;
      cp16(smem_u32(s + (2 * NC + r) * SK + c * 8), Z + (long)s_tok[r] * DM + k0 + c * 8, ok);
    }
  };
  float ag[MT][NT][4], au[MT][NT][4];
  #pragma unroll
  for (int a = 0; a < MT; a++)
    #pragma unroll
    for (int b = 0; b < NT; b++)
      #pragma unroll
      for (int c = 0; c < 4; c++) { ag[a][b][c] = 0.f; au[a][b][c] = 0.f; }
  const int NKC = DM / KC;
  load1(0, 0); cp_commit();
  for (int kc = 0; kc < NKC; kc++) {
    if (kc + 1 < NKC) { load1(kc + 1, (kc + 1) & 1); }
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    bf16* s = buf + (kc & 1) * S1;
    #pragma unroll
    for (int ks = 0; ks < KC / 16; ks++) {
      u32 bfr[NT][2];
      #pragma unroll
      for (int np = 0; np < NT; np += 2) {
        int t0 = (wn * NT + np) * 8;
        int row = t0 + (lane & 7) + ((lane >> 4) << 3);
        int col = ks * 16 + (((lane >> 3) & 1) << 3);
        ldm4(smem_u32(s + (2 * NC + row) * SK + col), bfr[np][0], bfr[np][1], bfr[np + 1][0], bfr[np + 1][1]);
      }
      #pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        int m0 = (wm * MT + mt) * 16;
        if ((m0 >> 4) >= mtile_n) continue;
        int row = m0 + (lane & 15);
        int col = ks * 16 + ((lane >> 4) << 3);
        u32 g0, g1, g2, g3, u0, u1, u2, u3;
        ldm4(smem_u32(s + row * SK + col), g0, g1, g2, g3);
        ldm4(smem_u32(s + (NC + row) * SK + col), u0, u1, u2, u3);
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          if (wn * NT + nt >= ntile_t) continue;
          mma(ag[mt][nt], g0, g1, g2, g3, bfr[nt][0], bfr[nt][1]);
          mma(au[mt][nt], u0, u1, u2, u3, bfr[nt][0], bfr[nt][1]);
        }
      }
    }
    __syncthreads();
  }
  // H = silu(G) x U x 0/1 -> Hs[토큰][뉴런]
  {
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT; mt++)
      #pragma unroll
      for (int nt = 0; nt < NT; nt++)
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          int m = (wm * MT + mt) * 16 + gid + ((c >> 1) << 3);
          int t = (wn * NT + nt) * 8 + tig * 2 + (c & 1);
          float h = 0.f;
          if (m < nneu && t < ntok) {
            int n = s_neu[m];
            u32 on = (bits[s_key[t] * 128 + (n >> 5)] >> (n & 31)) & 1u;
            h = on ? silu(ag[mt][nt][c]) * au[mt][nt][c] : 0.f;
          }
          Hs[t * SH + m] = f2bf(h);
        }
  }
  __syncthreads();

  // ---- 2단계: DM을 DC씩 ----
  auto load2 = [&](int dc, int b) {
    bf16* s = buf + b * S2;
    const int d0 = dc * DC;
    for (int i = tid; i < NC * (DC / 8); i += NTHR) {
      int r = i / (DC / 8), c = i % (DC / 8);
      bool ok = r < nneu;
      cp16(smem_u32(s + r * SD + c * 8), Wp + (long)s_neu[r] * (3 * DM) + 2 * DM + d0 + c * 8, ok);
    }
  };
  const int NDC = DM / DC;
  const int NT2 = TT / 8;                     // 2단계: 워프마다 d 32칸(m16 두 조각) x 토큰 전부
  load2(0, 0); cp_commit();
  for (int dc = 0; dc < NDC; dc++) {
    if (dc + 1 < NDC) { load2(dc + 1, (dc + 1) & 1); }
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    bf16* s = buf + (dc & 1) * S2;
    float acc[2][NT2][4];
    #pragma unroll
    for (int a = 0; a < 2; a++)
      #pragma unroll
      for (int b = 0; b < NT2; b++)
        #pragma unroll
        for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    #pragma unroll
    for (int ks = 0; ks < NC / 16; ks++) {
      if (ks >= mtile_n) break;
      u32 bfr[NT2][2];
      #pragma unroll
      for (int np = 0; np < NT2; np += 2) {
        if (np >= ntile_t) break;
        int row = np * 8 + (lane & 7) + ((lane >> 4) << 3);
        int col = ks * 16 + (((lane >> 3) & 1) << 3);
        ldm4(smem_u32(Hs + row * SH + col), bfr[np][0], bfr[np][1], bfr[np + 1][0], bfr[np + 1][1]);
      }
      #pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        int dm0 = wid * 32 + mt * 16;
        int krow = ks * 16 + (lane & 7) + ((lane >> 4) << 3);
        int dcol = dm0 + (((lane >> 3) & 1) << 3);
        u32 a0, a1, a2, a3;
        ldm4t(smem_u32(s + krow * SD + dcol), a0, a1, a2, a3);
        #pragma unroll
        for (int nt = 0; nt < NT2; nt++) {
          if (nt >= ntile_t) break;
          mma(acc[mt][nt], a0, a1, a2, a3, bfr[nt][0], bfr[nt][1]);
        }
      }
    }
    // 출력에 더하기
    {
      const int gid = lane >> 2, tig = lane & 3;
      #pragma unroll
      for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < NT2; nt++) {
          if (nt >= ntile_t) break;
          #pragma unroll
          for (int c = 0; c < 4; c++) {
            int d = dc * DC + wid * 32 + mt * 16 + gid + ((c >> 1) << 3);
            int t = nt * 8 + tig * 2 + (c & 1);
            if (t < ntok) {
              float* p = Y + (long)s_tok[t] * DM + d;
#if OUTMODE == 0
              asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(acc[mt][nt][c]));
#elif OUTMODE == 1
              *p = acc[mt][nt][c];                 // 진단용: 그냥 쓰기 (결과 틀림)
#else
              if (acc[mt][nt][c] == 12345.678f) *p = 0.f;   // 진단용: 쓰지 않기 (결과 틀림)
#endif
            }
          }
        }
    }
    __syncthreads();
  }
}

extern "C" __global__ void to_bf16(const float* __restrict__ Y, bf16* __restrict__ out, long n) {
  long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  float4 v = *(const float4*)(Y + i);
  uint2 o = make_uint2(pack2(v.x, v.y), pack2(v.z, v.w));
  *(uint2*)(out + i) = o;
}
'''


class FFNKernel:
    """설정 하나(MT, NT, WM, WN, DM)의 커널. NC = 16 MT WM 뉴런, TT = 8 NT WN 토큰."""

    def __init__(self, MT=1, NT=4, WM=2, WN=2, dm=D, outmode=0):
        assert WM * WN == 4 and NT % 2 == 0 and (8 * NT * WN) % 16 == 0
        self.MT, self.NT, self.WM, self.WN, self.dm = MT, NT, WM, WN, dm
        self.NC, self.TT = 16 * MT * WM, 8 * NT * WN
        k = build(SRC, ['ffn_rect', 'to_bf16'], extra=['-DMT=%d' % MT, '-DNT=%d' % NT, '-DWM=%d' % WM, '-DWN=%d' % WN, '-DDM=%d' % dm, '-DOUTMODE=%d' % outmode])
        self.k, self.conv = k['ffn_rect'], k['to_bf16']
        SK, SD, SH = 64 + 8, 128 + 8, self.NC + 8
        s1 = (2 * self.NC + self.TT) * SK
        s2 = self.NC * SD
        self.smem = 2 * (2 * max(s1, s2)) + 2 * self.TT * SH + 4 * (2 * self.TT + self.NC)
        if self.smem > 48 * 1024:
            self.k.set_smem(self.smem)

    def work(self, plan):
        """plan: [(토큰 번호들, 뉴런 번호들)] -> GPU 배열 (토큰을 TT개씩, 뉴런을 NC개씩 잘라 일로)."""
        toks, neus, wk = [], [], []
        to = no = 0
        for rows, cols in plan:
            rows = np.asarray(rows, np.int32)
            cols = np.asarray(cols, np.int32)
            for a in range(0, len(cols), self.NC):
                cc = cols[a:a + self.NC]
                for b in range(0, len(rows), self.TT):
                    rr = rows[b:b + self.TT]
                    toks.append(rr)
                    neus.append(cc)
                    wk.append((to, len(rr), no, len(cc)))
                    to += len(rr)
                    no += len(cc)
        dev = 'cuda'
        return dict(tok=torch.as_tensor(np.concatenate(toks), device=dev), neu=torch.as_tensor(np.concatenate(neus), device=dev),
                    work=torch.as_tensor(np.array(wk, np.int32).reshape(-1, 4), device=dev), n=len(wk),
                    visits=float(sum(w[1] for w in wk)))

    def __call__(self, Z, Wp, bits, key, W, Y, out):
        Y.zero_()
        self.k((W['n'],), (128,), [Z, Wp, bits, key, W['tok'], W['neu'], W['work'], Y, W['n']], smem=self.smem)
        n = Y.numel()
        self.conv(((n // 4 + 255) // 256,), (256,), [Y, out, n])


def pack_weights(blk):
    """nn 블록 -> (4096, 3 x 1,024) bf16: 뉴런마다 gate 행 | up 행 | down 열."""
    return torch.cat([blk.gate.weight, blk.up.weight, blk.down.weight.t()], dim=1).to(torch.bfloat16).contiguous()


def pack_bits(BM):
    """BM (256, 4096) bool (numpy 또는 torch) -> (256, 128) int32 비트."""
    b = torch.as_tensor(np.asarray(BM.cpu() if isinstance(BM, torch.Tensor) else BM), dtype=torch.bool)
    b = b.reshape(256, 128, 32).to(torch.int64)
    v = (b << torch.arange(32, dtype=torch.int64)).sum(-1)
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v)
    return v.to(torch.int32).cuda().contiguous()
