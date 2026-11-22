"""NVRTC(PyTorch에 든 nvrtc64_120_0.dll)로 CUDA 커널을 컴파일하고, 드라이버 API(nvcuda.dll)로 PyTorch의 지금 스트림에 띄운다.
이 기계에는 nvcc, MSVC, Triton, CuPy가 없다. 설치 없이 텐서 코어(mma.sync) 커널을 쓰려는 것.
헤더를 쓰지 않는다: bf16은 unsigned short로 다루고, 변환과 mma는 인라인 PTX로 한다.
CUDA 그래프 안에서도 띄울 수 있다 (torch.cuda.graph의 스트림에 cuLaunchKernel).
"""
import ctypes
import os

import torch

_LIB = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.add_dll_directory(_LIB)
_nvrtc = ctypes.WinDLL(os.path.join(_LIB, 'nvrtc64_120_0.dll'))
_cu = ctypes.WinDLL('nvcuda.dll')
_cu.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                               ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
_cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_cu.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]

# 헤더 없이 쓰는 공용 도구 (bf16 <-> f32, silu)
PRELUDE = r'''
typedef unsigned short bf16;
typedef unsigned int u32;
__device__ __forceinline__ float bf2f(bf16 x) { return __uint_as_float(((u32)x) << 16); }
__device__ __forceinline__ bf16 f2bf(float x) { bf16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return r; }
__device__ __forceinline__ u32 pack2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
'''


def _chk(r, what):
    if r != 0:
        raise RuntimeError('%s 실패 (코드 %d)' % (what, r))


class Kernel:
    def __init__(self, func, name):
        self.f, self.name = func, name

    def set_smem(self, nbytes):
        _chk(_cu.cuFuncSetAttribute(self.f, 8, int(nbytes)), 'cuFuncSetAttribute')   # MAX_DYNAMIC_SHARED_SIZE_BYTES

    def regs(self):
        v = ctypes.c_int()
        _chk(_cu.cuFuncGetAttribute(ctypes.byref(v), 4, self.f), 'cuFuncGetAttribute')  # NUM_REGS
        return v.value

    def __call__(self, grid, block, args, smem=0, stream=None):
        """args: torch 텐서(주소를 넘김), int(32비트), float. grid, block: (x, y, z)."""
        keep, ptrs = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = ctypes.c_void_p(a.data_ptr())
            elif isinstance(a, float):
                v = ctypes.c_float(a)
            else:
                v = ctypes.c_int(int(a))
            keep.append(v)
            ptrs.append(ctypes.cast(ctypes.byref(v), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        s = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        _chk(_cu.cuLaunchKernel(self.f, g[0], g[1], g[2], b[0], b[1], b[2], int(smem), ctypes.c_void_p(s), arr, None),
             'cuLaunchKernel(%s)' % self.name)


def build(src, names, arch='sm_89', extra=()):
    """src를 컴파일해 이름들의 커널을 돌려준다. 빠른 수학(--use_fast_math)은 쓰지 않는다 (결과를 같게)."""
    torch.cuda.init()
    torch.zeros(1, device='cuda')                                      # 기본 문맥을 이 스레드에 올린다
    prog = ctypes.c_void_p()
    _chk(_nvrtc.nvrtcCreateProgram(ctypes.byref(prog), (PRELUDE + src).encode(), b'k.cu', 0, None, None), 'nvrtcCreateProgram')
    opts = [('--gpu-architecture=%s' % arch).encode(), b'-std=c++17', b'-lineinfo'] + [e.encode() for e in extra]
    r = _nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    n = ctypes.c_size_t()
    _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    _nvrtc.nvrtcGetProgramLog(prog, log)
    if r != 0:
        raise RuntimeError('컴파일 실패:\n' + log.value.decode(errors='replace'))
    _chk(_nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)), 'nvrtcGetCUBINSize')
    cubin = ctypes.create_string_buffer(n.value)
    _chk(_nvrtc.nvrtcGetCUBIN(prog, cubin), 'nvrtcGetCUBIN')
    mod = ctypes.c_void_p()
    _chk(_cu.cuModuleLoadData(ctypes.byref(mod), cubin), 'cuModuleLoadData')
    out = {}
    for nm in names:
        f = ctypes.c_void_p()
        _chk(_cu.cuModuleGetFunction(ctypes.byref(f), mod, nm.encode()), 'cuModuleGetFunction(%s)' % nm)
        out[nm] = Kernel(f, nm)
    out['_module'] = mod
    out['_cubin'] = cubin
    return out


def time_graph(fn, reps=200, warm=20, inner=20):
    """fn을 inner번 이어 붙여 CUDA 그래프 하나로 잡고, reps번 띄워 fn 한 번의 시간(ms)을 돌려준다 (가운데값, 사분위).
    Windows(WDDM)에서는 그래프를 한 번 띄울 때마다 약 45 ~ 50 us가 재는 값에 붙는다. inner번 이어 붙여 그 몫을 1/inner로 줄인다."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    ts.sort()
    return ts[len(ts) // 2], ts[len(ts) // 4], ts[3 * len(ts) // 4]
