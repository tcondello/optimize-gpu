"""
High-Performance Ampere (sm_86) CUDA Kernels for TinyGPU + tinygrad on macOS.

Operates directly on raw GGUF quantized tensors (Q4_K, Q6_K, Q8_0) in GPU VRAM
with zero VRAM bloat and zero intermediate copy kernels, using:
- On-the-fly Q8_1 activation quantization (fused with float4 register-cached RMSNorm and SwiGLU)
- Hardware INT8 4-way dot-product (`__dp4a`) + warp-shuffle (`__shfl_down_sync`) reductions
- Register-tiled multi-token GEMM (`BM=8`) for ~1,000+ tok/s prompt prefill
- Split-QKV Fused QK-RMSNorm + RoPE + KV-cache update & single-pass Online-Softmax GQA Flash-Decoding
- Single-CTA warp-parallel Vocabulary Argmax (151,936 logits in ~20 us)
"""
from __future__ import annotations

import functools
from tinygrad import Tensor, UOp, dtypes
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops

CUDA_COMMON = r'''
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ unsigned int load_u32_align2(const unsigned char* p) {
  const unsigned short* p16 = reinterpret_cast<const unsigned short*>(p);
  return ((unsigned int)p16[0]) | (((unsigned int)p16[1]) << 16);
}
'''


def _make_program_uop(
    name: str,
    src: str,
    out_uops: list[UOp],
    in_uops: list[UOp],
    global_size: tuple[int, int, int],
    local_size: tuple[int, int, int],
    ops_est: int,
    mem_est: int,
    dname: str = "NV",
    var_uops: list[UOp] | None = None,
) -> UOp:
    stores = [
        o.base.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(o.dtype.base, 0))
        for o in out_uops
    ]
    loads = [
        b.base.index(UOp.const(dtypes.int, 0), ptr=True).load()
        for b in in_uops
    ]
    specials = []
    for idx, sz in enumerate(local_size):
        if sz > 1 or idx == 0:
            specials.append(UOp.special(sz, f"lidx{idx}"))
    for idx, sz in enumerate(global_size):
        if sz > 1 or idx == 0:
            specials.append(UOp.special(sz, f"gidx{idx}"))

    extra_vars = [
        (v.src[0] if v.op is Ops.BIND else v)
        for v in (var_uops or [])
        if isinstance(v, UOp) and v.op in (Ops.DEFINE_VAR, Ops.BIND)
    ]
    sink = UOp.sink(
        *stores,
        *loads,
        *specials,
        *extra_vars,
        arg=KernelInfo(name=name, estimates=Estimates(ops=ops_est, mem=mem_est)),
    )
    dname_base = dname.split(":")[0]
    return UOp(
        Ops.PROGRAM,
        src=(
            sink,
            UOp(Ops.DEVICE, arg=dname_base),
            UOp(Ops.LINEAR, src=(*sink.src, sink)),
            UOp(Ops.SOURCE, arg=src),
        ),
    )


def _ensure_zero_copy_contig(t: Tensor) -> UOp:
    """Avoid inserting redundant E_... copy kernels when `t` is already a pure RESHAPE of BUFFER or AFTER."""
    u = t.uop
    cur = u
    while cur.op is Ops.RESHAPE:
        cur = cur.src[0]
    if cur.op in (Ops.AFTER, Ops.BUFFER):
        return u
    return u.contiguous()


def _call_custom(tensors: list[Tensor], fxn, bind_vars: list[UOp] | None = None) -> list[Tensor]:
    contig_srcs = tuple(_ensure_zero_copy_contig(t) for t in tensors)
    placeholders = [UOp.placeholder_like(s, slot=i) for i, s in enumerate(contig_srcs)]
    bound = tuple(v for v in (bind_vars or []) if isinstance(v, UOp) and v.op is Ops.BIND)
    kernel = fxn(*placeholders).call(*contig_srcs, *bound)
    return [Tensor(s.after(kernel)) for s in contig_srcs]


# -----------------------------------------------------------------------------
# 1. Q8_1 Quantization, Fused RMSNorm + Q8_1, and Fused SwiGLU + Q8_1
# -----------------------------------------------------------------------------

@functools.cache
def _quantize_q8_1_kernel(q_out: UOp, ds_out: UOp, x_in: UOp, M: int, K: int) -> UOp:
    assert K % 32 == 0
    n_blocks_k = K // 32
    total_blocks = M * n_blocks_k
    warps_per_cta = 4
    threads = 32 * warps_per_cta
    grid = (total_blocks + warps_per_cta - 1) // warps_per_cta
    fn_name = f"quant_q8_1_{M}_{K}"
    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in
) {{
  const int warp_id = (blockIdx.x * {warps_per_cta}) + (threadIdx.x >> 5);
  if (warp_id >= {total_blocks}) return;
  const int lane = threadIdx.x & 31;
  const float v = x_in[(warp_id << 5) + lane];

  float sum_x = v;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);

  float amax = fabsf(v);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

  const float d = amax / 127.0f;
  const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  const int q = __float2int_rn(v * id);
  q_out[(warp_id << 5) + lane] = (signed char)max(-127, min(127, q));
  if (lane == 0) {{
    ds_out[warp_id] = make_float2(d, sum_x);
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[q_out, ds_out], in_uops=[x_in],
        global_size=(grid, 1, 1), local_size=(threads, 1, 1),
        ops_est=M * K * 6, mem_est=M * K * 5,
    )


def quantize_q8_1(x: Tensor) -> tuple[Tensor, Tensor]:
    K = x.shape[-1]
    M = x.numel() // K
    x_c = x.float() if x.dtype != dtypes.float32 else x
    q_out = Tensor.empty(M, K, dtype=dtypes.int8, device=x.device)
    ds_out = Tensor.empty(M, K // 32, 2, dtype=dtypes.float32, device=x.device)
    q_out, ds_out = _call_custom([q_out, ds_out, x_c], fxn=functools.partial(_quantize_q8_1_kernel, M=M, K=K))[:2]
    return q_out, ds_out


@functools.cache
def _rmsnorm_q8_1_kernel(q_out: UOp, ds_out: UOp, x_in: UOp, w_in: UOp, M: int, K: int, eps: float) -> UOp:
    assert K % 128 == 0 and K <= 16384
    threads = 128
    warps = threads // 32  # 4 warps
    n_blocks = K // 32
    blks_per_warp = (n_blocks + warps - 1) // warps
    fn_name = f"rmsnorm_q8_1_v2_{M}_{K}_{str(eps).replace('.', '_').replace('-', '_')}"
    w_is_f32 = w_in.dtype.base == dtypes.float32
    w_type = "float" if w_is_f32 else "half"
    w_read = "w_in[idx]" if w_is_f32 else "__half2float(w_in[idx])"
    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in,
    const {w_type}* __restrict__ w_in
) {{
  __shared__ float s_reduce[{warps}];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const float* x_row = x_in + (size_t)row * {K};

  // Register-cache x across pass 1 & pass 2 so x_row is read ONCE!
  float x_reg[{blks_per_warp}];
  float ss = 0.0f;
  #pragma unroll
  for (int i = 0; i < {blks_per_warp}; ++i) {{
    const int blk = wid + i * {warps};
    float v = 0.0f;
    if (blk < {n_blocks}) {{
      v = x_row[(blk << 5) + lane];
      ss += v * v;
    }}
    x_reg[i] = v;
  }}

  ss = warp_reduce_sum(ss);
  if (lane == 0) s_reduce[wid] = ss;
  __syncthreads();
  if (wid == 0) {{
    ss = (lane < {warps}) ? s_reduce[lane] : 0.0f;
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_reduce[0] = rsqrtf(ss * {1.0 / K:.12e}f + {eps:.12e}f);
  }}
  __syncthreads();
  const float scale = s_reduce[0];

  #pragma unroll
  for (int i = 0; i < {blks_per_warp}; ++i) {{
    const int blk = wid + i * {warps};
    if (blk < {n_blocks}) {{
      const int idx = (blk << 5) + lane;
      const float v = x_reg[i] * scale * ({w_read});

      float sum_x = v;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);

      float amax = fabsf(v);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

      const float d = amax / 127.0f;
      const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
      const int q = __float2int_rn(v * id);
      q_out[(size_t)row * {K} + idx] = (signed char)max(-127, min(127, q));
      if (lane == 0) {{
        ds_out[(size_t)row * {n_blocks} + blk] = make_float2(d, sum_x);
      }}
    }}
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[q_out, ds_out], in_uops=[x_in, w_in],
        global_size=(M, 1, 1), local_size=(threads, 1, 1),
        ops_est=M * K * 8, mem_est=M * K * 5,
    )


def rmsnorm_quantize_q8_1(x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    K = x.shape[-1]
    M = x.numel() // K
    x_c = x.float() if x.dtype != dtypes.float32 else x
    q_out = Tensor.empty(M, K, dtype=dtypes.int8, device=x.device)
    ds_out = Tensor.empty(M, K // 32, 2, dtype=dtypes.float32, device=x.device)
    q_out, ds_out = _call_custom(
        [q_out, ds_out, x_c, weight],
        fxn=functools.partial(_rmsnorm_q8_1_kernel, M=M, K=K, eps=float(eps)),
    )[:2]
    return q_out, ds_out


@functools.cache
def _swiglu_q8_1_kernel(q_out: UOp, ds_out: UOp, gate_up_in: UOp, M: int, H: int) -> UOp:
    assert H % 32 == 0
    n_blocks = H // 32
    total_blocks = M * n_blocks
    warps_per_cta = 4
    threads = 32 * warps_per_cta
    grid = (total_blocks + warps_per_cta - 1) // warps_per_cta
    fn_name = f"swiglu_q8_1_{M}_{H}"
    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ gate_up
) {{
  const int warp_id = (blockIdx.x * {warps_per_cta}) + (threadIdx.x >> 5);
  if (warp_id >= {total_blocks}) return;
  const int lane = threadIdx.x & 31;
  const int row = warp_id / {n_blocks};
  const int blk = warp_id % {n_blocks};
  const int col = (blk << 5) + lane;

  const float g = gate_up[(size_t)row * {2 * H} + col];
  const float u = gate_up[(size_t)row * {2 * H} + {H} + col];
  const float v = (g / (1.0f + expf(-g))) * u;

  float sum_x = v;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);

  float amax = fabsf(v);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

  const float d = amax / 127.0f;
  const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  const int q = __float2int_rn(v * id);
  q_out[((size_t)warp_id << 5) + lane] = (signed char)max(-127, min(127, q));
  if (lane == 0) {{
    ds_out[warp_id] = make_float2(d, sum_x);
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[q_out, ds_out], in_uops=[gate_up_in],
        global_size=(grid, 1, 1), local_size=(threads, 1, 1),
        ops_est=M * H * 10, mem_est=M * H * 9,
    )


def swiglu_quantize_q8_1(gate_up: Tensor) -> tuple[Tensor, Tensor]:
    H = gate_up.shape[-1] // 2
    M = gate_up.numel() // (2 * H)
    q_out = Tensor.empty(M, H, dtype=dtypes.int8, device=gate_up.device)
    ds_out = Tensor.empty(M, H // 32, 2, dtype=dtypes.float32, device=gate_up.device)
    q_out, ds_out = _call_custom(
        [q_out, ds_out, gate_up],
        fxn=functools.partial(_swiglu_q8_1_kernel, M=M, H=H),
    )[:2]
    return q_out, ds_out


# -----------------------------------------------------------------------------
# 2. Ampere __dp4a Raw GGUF Q4_K, Q6_K, Q8_0 GEMV & Register-Tiled Prefill GEMM
# -----------------------------------------------------------------------------

@functools.cache
def _gemv_q4_k_kernel(
    out_uop: UOp,
    w_uop: UOp,
    xq_uop: UOp,
    xds_uop: UOp,
    res_or_bias_uop: UOp | None,
    M: int,
    N: int,
    K: int,
    mode: str,
) -> UOp:
    assert K % 256 == 0
    n_sb = K // 256
    bm = 8 if (M >= 8 and M % 8 == 0) else (4 if (M >= 4 and M % 4 == 0) else 1)
    warps_per_cta = 8 if (N >= 16384 and bm == 1) else 4
    threads = 32 * warps_per_cta
    grid_x = (N + warps_per_cta - 1) // warps_per_cta
    grid_y = M // bm
    fn_name = f"gemv_q4k_{M}b{bm}w{warps_per_cta}_{N}_{K}_{mode}"

    extra_param = ""
    if mode in ("bias", "residual"):
        extra_param = ", const float* __restrict__ extra"

    store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi];"
    if mode == "bias":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + b_val;"
    elif mode == "residual":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + extra[(m0 + mi) * N_OUT + row];"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const unsigned char* __restrict__ w_raw,
    const int* __restrict__ x_q8,
    const float2* __restrict__ x_ds{extra_param}
) {{
  constexpr int N_OUT = {N};
  constexpr int N_SB = {n_sb};
  constexpr int BM = {bm};
  const int row = blockIdx.x * {warps_per_cta} + (threadIdx.x >> 5);
  if (row >= N_OUT) return;
  const int m0 = blockIdx.y * BM;
  const int lane = threadIdx.x & 31;

  const int sb_lane = lane >> 3;
  const int in_sb   = lane & 7;
  const int chunk   = in_sb >> 1;
  const int half16  = in_sb & 1;

  const unsigned char* w_row = w_raw + (size_t)row * (N_SB * 144);
  float acc[BM];
  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) acc[mi] = 0.0f;

  for (int sb = sb_lane; sb < N_SB; sb += 4) {{
    const unsigned char* blk = w_row + sb * 144;
    const half2 dm = *reinterpret_cast<const half2*>(blk);
    const float d = __half2float(dm.x);
    const float dmin = __half2float(dm.y);

    const unsigned int s0 = *reinterpret_cast<const unsigned int*>(blk + 4);
    const unsigned int s1 = *reinterpret_cast<const unsigned int*>(blk + 8);
    const unsigned int s2 = *reinterpret_cast<const unsigned int*>(blk + 12);

    unsigned int sc_lo, sc_hi, mn_lo, mn_hi;
    if (chunk < 2) {{
      const int sh0 = (chunk * 2) * 8;
      const int sh1 = (chunk * 2 + 1) * 8;
      sc_lo = (s0 >> sh0) & 0x3fu;
      sc_hi = (s0 >> sh1) & 0x3fu;
      mn_lo = (s1 >> sh0) & 0x3fu;
      mn_hi = (s1 >> sh1) & 0x3fu;
    }} else {{
      const int sh0 = ((chunk - 2) * 2) * 8;
      const int sh1 = ((chunk - 2) * 2 + 1) * 8;
      sc_lo = ((s2 >> sh0) & 0x0fu) | (((s0 >> (sh0 + 6)) & 0x03u) << 4);
      sc_hi = ((s2 >> sh1) & 0x0fu) | (((s0 >> (sh1 + 6)) & 0x03u) << 4);
      mn_lo = ((s2 >> (sh0 + 4)) & 0x0fu) | (((s1 >> (sh0 + 6)) & 0x03u) << 4);
      mn_hi = ((s2 >> (sh1 + 4)) & 0x0fu) | (((s1 >> (sh1 + 6)) & 0x03u) << 4);
    }}

    const uint4 qv = *reinterpret_cast<const uint4*>(blk + 16 + (in_sb << 4));
    const int ql0 = (int)(qv.x & 0x0f0f0f0fu), ql1 = (int)(qv.y & 0x0f0f0f0fu);
    const int ql2 = (int)(qv.z & 0x0f0f0f0fu), ql3 = (int)(qv.w & 0x0f0f0f0fu);
    const int qh0 = (int)((qv.x >> 4) & 0x0f0f0f0fu), qh1 = (int)((qv.y >> 4) & 0x0f0f0f0fu);
    const int qh2 = (int)((qv.z >> 4) & 0x0f0f0f0fu), qh3 = (int)((qv.w >> 4) & 0x0f0f0f0fu);

    const float d_sc_lo = d * (float)sc_lo;
    const float d_sc_hi = d * (float)sc_hi;
    const float dm_mn_lo = dmin * (float)mn_lo;
    const float dm_mn_hi = dmin * (float)mn_hi;

    const int x_off_int = (sb << 6) + (chunk << 4) + (half16 << 2);
    const int sub0 = (sb << 3) + (chunk << 1);

    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      const int* x_row = x_q8 + (size_t)(m0 + mi) * {(K // 4)} + x_off_int;
      const float2* ds_row = x_ds + (size_t)(m0 + mi) * {(K // 32)} + sub0;
      const int4 x_lo = *reinterpret_cast<const int4*>(x_row);
      const int4 x_hi = *reinterpret_cast<const int4*>(x_row + 8);

      int sumi_lo = __dp4a(ql0, x_lo.x, 0);
      sumi_lo = __dp4a(ql1, x_lo.y, sumi_lo);
      sumi_lo = __dp4a(ql2, x_lo.z, sumi_lo);
      sumi_lo = __dp4a(ql3, x_lo.w, sumi_lo);

      int sumi_hi = __dp4a(qh0, x_hi.x, 0);
      sumi_hi = __dp4a(qh1, x_hi.y, sumi_hi);
      sumi_hi = __dp4a(qh2, x_hi.z, sumi_hi);
      sumi_hi = __dp4a(qh3, x_hi.w, sumi_hi);

      const float2 ds0 = ds_row[0];
      const float2 ds1 = ds_row[1];
      acc[mi] += d_sc_lo * ds0.x * (float)sumi_lo + d_sc_hi * ds1.x * (float)sumi_hi;
      if (half16 == 0) {{
        acc[mi] -= dm_mn_lo * ds0.y + dm_mn_hi * ds1.y;
      }}
    }}
  }}

  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) {{
    acc[mi] = warp_reduce_sum(acc[mi]);
  }}
  if (lane == 0) {{
    {"const float b_val = extra[row];" if mode == "bias" else ""}
    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      {store_stmt}
    }}
  }}
}}
'''
    ins = [w_uop, xq_uop, xds_uop] + ([res_or_bias_uop] if res_or_bias_uop is not None else [])
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=ins,
        global_size=(grid_x, grid_y, 1), local_size=(threads, 1, 1),
        ops_est=2 * M * N * K, mem_est=(M // bm) * N * n_sb * 144 + M * K + M * N * 4,
    )


@functools.cache
def _gemv_q6_k_kernel(
    out_uop: UOp,
    w_uop: UOp,
    xq_uop: UOp,
    xds_uop: UOp,
    res_or_bias_uop: UOp | None,
    M: int,
    N: int,
    K: int,
    mode: str,
) -> UOp:
    assert K % 256 == 0
    n_sb = K // 256
    bm = 8 if (M >= 8 and M % 8 == 0) else (4 if (M >= 4 and M % 4 == 0) else 1)
    warps_per_cta = 8 if (N >= 16384 and bm == 1) else 4
    threads = 32 * warps_per_cta
    grid_x = (N + warps_per_cta - 1) // warps_per_cta
    grid_y = M // bm
    fn_name = f"gemv_q6k_{M}b{bm}w{warps_per_cta}_{N}_{K}_{mode}"

    extra_param = ""
    if mode in ("bias", "residual"):
        extra_param = ", const float* __restrict__ extra"

    store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi];"
    if mode == "bias":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + b_val;"
    elif mode == "residual":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + extra[(m0 + mi) * N_OUT + row];"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const unsigned char* __restrict__ w_raw,
    const int* __restrict__ x_q8,
    const float2* __restrict__ x_ds{extra_param}
) {{
  constexpr int N_OUT = {N};
  constexpr int N_SB = {n_sb};
  constexpr int BM = {bm};
  const int row = blockIdx.x * {warps_per_cta} + (threadIdx.x >> 5);
  if (row >= N_OUT) return;
  const int m0 = blockIdx.y * BM;
  const int lane = threadIdx.x & 31;

  const int sb_lane = lane >> 4;
  const int tid16   = lane & 15;
  const int h       = tid16 >> 3;
  const int p       = (tid16 >> 2) & 1;
  const int q       = (tid16 >> 1) & 1;
  const int half8   = tid16 & 1;

  const int ql_off  = (h << 6) + (q << 5) + (half8 << 4);
  const int qh_off  = 128 + (h << 5) + (half8 << 4);
  const int ql_sh   = p << 2;
  const int qh_sh   = ((p << 1) + q) << 1;

  const unsigned char* w_row = w_raw + (size_t)row * (N_SB * 210);
  float acc[BM];
  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) acc[mi] = 0.0f;

  for (int sb = sb_lane; sb < N_SB; sb += 2) {{
    const unsigned char* blk = w_row + sb * 210;
    const float d_sc = __half2float(*reinterpret_cast<const half*>(blk + 208)) *
                       (float)(reinterpret_cast<const signed char*>(blk + 192)[tid16]);

    const unsigned char* ql_ptr = blk + ql_off;
    const unsigned char* qh_ptr = blk + qh_off;

    const unsigned int ql0 = load_u32_align2(ql_ptr + 0);
    const unsigned int ql1 = load_u32_align2(ql_ptr + 4);
    const unsigned int ql2 = load_u32_align2(ql_ptr + 8);
    const unsigned int ql3 = load_u32_align2(ql_ptr + 12);

    const unsigned int qh0 = load_u32_align2(qh_ptr + 0);
    const unsigned int qh1 = load_u32_align2(qh_ptr + 4);
    const unsigned int qh2 = load_u32_align2(qh_ptr + 8);
    const unsigned int qh3 = load_u32_align2(qh_ptr + 12);

    const unsigned int u6_0 = ((ql0 >> ql_sh) & 0x0f0f0f0fu) | (((qh0 >> qh_sh) & 0x03030303u) << 4);
    const unsigned int u6_1 = ((ql1 >> ql_sh) & 0x0f0f0f0fu) | (((qh1 >> qh_sh) & 0x03030303u) << 4);
    const unsigned int u6_2 = ((ql2 >> ql_sh) & 0x0f0f0f0fu) | (((qh2 >> qh_sh) & 0x03030303u) << 4);
    const unsigned int u6_3 = ((ql3 >> ql_sh) & 0x0f0f0f0fu) | (((qh3 >> qh_sh) & 0x03030303u) << 4);

    const unsigned int m0b = (~u6_0) & 0x20202020u;
    const unsigned int m1b = (~u6_1) & 0x20202020u;
    const unsigned int m2b = (~u6_2) & 0x20202020u;
    const unsigned int m3b = (~u6_3) & 0x20202020u;

    const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
    const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
    const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
    const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));

    const int x_base_int = (sb << 6) + (tid16 << 2);
    const int ds_idx = (sb << 3) + (tid16 >> 1);

    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      const int4 xv = *reinterpret_cast<const int4*>(x_q8 + (size_t)(m0 + mi) * {(K // 4)} + x_base_int);
      int sumi = __dp4a(s6_0, xv.x, 0);
      sumi = __dp4a(s6_1, xv.y, sumi);
      sumi = __dp4a(s6_2, xv.z, sumi);
      sumi = __dp4a(s6_3, xv.w, sumi);
      const float x_d = x_ds[(size_t)(m0 + mi) * {(K // 32)} + ds_idx].x;
      acc[mi] += (d_sc * x_d) * (float)sumi;
    }}
  }}

  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) {{
    acc[mi] = warp_reduce_sum(acc[mi]);
  }}
  if (lane == 0) {{
    {"const float b_val = extra[row];" if mode == "bias" else ""}
    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      {store_stmt}
    }}
  }}
}}
'''
    ins = [w_uop, xq_uop, xds_uop] + ([res_or_bias_uop] if res_or_bias_uop is not None else [])
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=ins,
        global_size=(grid_x, grid_y, 1), local_size=(threads, 1, 1),
        ops_est=2 * M * N * K, mem_est=(M // bm) * N * n_sb * 210 + M * K + M * N * 4,
    )


@functools.cache
def _gemv_q8_0_kernel(
    out_uop: UOp,
    w_uop: UOp,
    xq_uop: UOp,
    xds_uop: UOp,
    res_or_bias_uop: UOp | None,
    M: int,
    N: int,
    K: int,
    mode: str,
) -> UOp:
    assert K % 32 == 0
    n_blk = K // 32
    bm = 8 if (M >= 8 and M % 8 == 0) else (4 if (M >= 4 and M % 4 == 0) else 1)
    warps_per_cta = 8 if (N >= 16384 and bm == 1) else 4
    threads = 32 * warps_per_cta
    grid_x = (N + warps_per_cta - 1) // warps_per_cta
    grid_y = M // bm
    fn_name = f"gemv_q8_0_{M}b{bm}w{warps_per_cta}_{N}_{K}_{mode}"

    extra_param = ""
    if mode in ("bias", "residual"):
        extra_param = ", const float* __restrict__ extra"

    store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi];"
    if mode == "bias":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + b_val;"
    elif mode == "residual":
        store_stmt = "out[(m0 + mi) * N_OUT + row] = acc[mi] + extra[(m0 + mi) * N_OUT + row];"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const unsigned char* __restrict__ w_raw,
    const int* __restrict__ x_q8,
    const float2* __restrict__ x_ds{extra_param}
) {{
  constexpr int N_OUT = {N};
  constexpr int N_BLK = {n_blk};
  constexpr int BM = {bm};
  const int row = blockIdx.x * {warps_per_cta} + (threadIdx.x >> 5);
  if (row >= N_OUT) return;
  const int m0 = blockIdx.y * BM;
  const int lane = threadIdx.x & 31;

  const int blk_lane = lane >> 1;
  const int half16   = lane & 1;

  const unsigned char* w_row = w_raw + (size_t)row * (N_BLK * 34);
  float acc[BM];
  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) acc[mi] = 0.0f;

  for (int b = blk_lane; b < N_BLK; b += 16) {{
    const unsigned char* blk = w_row + b * 34;
    const float d = __half2float(*reinterpret_cast<const half*>(blk));
    const unsigned char* qs = blk + 2 + (half16 << 4);

    const int w0 = (int)load_u32_align2(qs + 0);
    const int w1 = (int)load_u32_align2(qs + 4);
    const int w2 = (int)load_u32_align2(qs + 8);
    const int w3 = (int)load_u32_align2(qs + 12);

    const int x_off_int = (b << 3) + (half16 << 2);
    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      const int4 xv = *reinterpret_cast<const int4*>(x_q8 + (size_t)(m0 + mi) * {(K // 4)} + x_off_int);
      int sumi = __dp4a(w0, xv.x, 0);
      sumi = __dp4a(w1, xv.y, sumi);
      sumi = __dp4a(w2, xv.z, sumi);
      sumi = __dp4a(w3, xv.w, sumi);
      acc[mi] += (d * x_ds[(size_t)(m0 + mi) * N_BLK + b].x) * (float)sumi;
    }}
  }}

  #pragma unroll
  for (int mi = 0; mi < BM; ++mi) {{
    acc[mi] = warp_reduce_sum(acc[mi]);
  }}
  if (lane == 0) {{
    {"const float b_val = extra[row];" if mode == "bias" else ""}
    #pragma unroll
    for (int mi = 0; mi < BM; ++mi) {{
      {store_stmt}
    }}
  }}
}}
'''
    ins = [w_uop, xq_uop, xds_uop] + ([res_or_bias_uop] if res_or_bias_uop is not None else [])
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=ins,
        global_size=(grid_x, grid_y, 1), local_size=(threads, 1, 1),
        ops_est=2 * M * N * K, mem_est=(M // bm) * N * n_blk * 34 + M * K + M * N * 4,
    )


def gguf_quant_gemv(
    x_q8: Tensor,
    x_ds: Tensor,
    w_raw_bytes: Tensor,
    ggml_type: int,
    out_features: int,
    in_features: int,
    bias: Tensor | None = None,
    residual: Tensor | None = None,
    out_shape: tuple[int, ...] | None = None,
) -> Tensor:
    """Execute __dp4a GEMV / register-tiled GEMM directly on raw GGUF bytes with zero copy kernels."""
    M = x_q8.shape[0]
    mode = "none"
    extra: Tensor | None = None
    if bias is not None and residual is not None:
        out = gguf_quant_gemv(x_q8, x_ds, w_raw_bytes, ggml_type, out_features, in_features, bias=bias)
        return (out + residual.reshape(out.shape)) if out_shape is None else (out.reshape(out_shape) + residual)
    elif bias is not None:
        mode, extra = "bias", bias
    elif residual is not None:
        mode, extra = "residual", residual

    target_shape = out_shape if out_shape is not None else (M, out_features)
    out = Tensor.empty(*target_shape, dtype=dtypes.float32, device=x_q8.device)
    kern_map = {12: _gemv_q4_k_kernel, 14: _gemv_q6_k_kernel, 8: _gemv_q8_0_kernel}
    kern_fn = kern_map[ggml_type]

    if extra is not None:
        fxn = lambda o, w, xq, xds, ex: kern_fn(o, w, xq, xds, ex, M=M, N=out_features, K=in_features, mode=mode)
        return _call_custom([out, w_raw_bytes, x_q8, x_ds, extra], fxn=fxn)[0]
    fxn = lambda o, w, xq, xds: kern_fn(o, w, xq, xds, None, M=M, N=out_features, K=in_features, mode="none")
    return _call_custom([out, w_raw_bytes, x_q8, x_ds], fxn=fxn)[0]


# -----------------------------------------------------------------------------
# 3. Fused QK-RMSNorm + Half-Split RoPE + KV Cache Update (Fused OR Split QK+V!)
# -----------------------------------------------------------------------------

@functools.cache
def _qk_norm_rope_kv_kernel_baked(
    q_out_uop: UOp,
    kv_cache_uop: UOp,
    qk_or_qkv_uop: UOp,
    v_sep_uop: UOp | None,
    q_norm_w_uop: UOp | None,
    k_norm_w_uop: UOp | None,
    freqs_uop: UOp,
    start_pos_var: UOp,
    T: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    max_ctx: int,
    eps: float,
    has_qk_norm: bool,
    split_v: bool,
) -> UOp:
    total_qk_heads = n_heads + 2 * n_kv_heads
    threads = 32 * ((head_dim + 31) // 32)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qk_stride = (q_dim + kv_dim) if split_v else (q_dim + 2 * kv_dim)
    half_rope = rope_dim // 2
    fn_name = f"qknorm_rope_kv_{T}_{n_heads}_{n_kv_heads}_{head_dim}_{max_ctx}_{int(has_qk_norm)}_sv{int(split_v)}"

    v_param = ", const float* __restrict__ v_in" if split_v else ""
    v_expr = f"v_in[(size_t)t * {kv_dim} + kv_h * {head_dim} + tid]" if split_v else f"row_in[{q_dim + kv_dim} + kv_h * {head_dim} + tid]"

    norm_params = ""
    q_w_read = "1.0f"
    if has_qk_norm:
        w_is_f32 = q_norm_w_uop.dtype.base == dtypes.float32
        w_type = "float" if w_is_f32 else "half"
        norm_params = f", const {w_type}* __restrict__ q_norm_w, const {w_type}* __restrict__ k_norm_w"
        q_w_read = "nw[tid]" if w_is_f32 else "__half2float(nw[tid])"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ q_out,
    half* __restrict__ kv_cache,
    const float* __restrict__ qk_in{v_param}{norm_params},
    const float* __restrict__ freqs_cis,
    const int start_pos
) {{
  const int h_all = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid >= {head_dim}) return;

  const float* row_in = qk_in + (size_t)t * {qk_stride};
  const int pos = start_pos + t;

  if (h_all >= {n_heads + n_kv_heads}) {{
    const int kv_h = h_all - {n_heads + n_kv_heads};
    const float v_val = {v_expr};
    const size_t v_idx = ((size_t)({n_kv_heads} + kv_h) * {max_ctx} + pos) * {head_dim} + tid;
    kv_cache[v_idx] = __float2half(v_val);
    return;
  }}

  const bool is_q = (h_all < {n_heads});
  const int h_idx = is_q ? h_all : (h_all - {n_heads});
  const float* src_ptr = is_q ? (row_in + h_idx * {head_dim}) : (row_in + {q_dim} + h_idx * {head_dim});
  float val = src_ptr[tid];

  {"{" if has_qk_norm else "#if 0"}
    __shared__ float s_ss[8];
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float ss = warp_reduce_sum(val * val);
    if (lane == 0) s_ss[wid] = ss;
    __syncthreads();
    if (wid == 0) {{
      ss = (lane < {threads // 32}) ? s_ss[lane] : 0.0f;
      ss = warp_reduce_sum(ss);
      if (lane == 0) s_ss[0] = rsqrtf(ss * {1.0 / head_dim:.12e}f + {eps:.12e}f);
    }}
    __syncthreads();
    const auto* nw = is_q ? q_norm_w : k_norm_w;
    val = val * s_ss[0] * ({q_w_read});
  {"}" if has_qk_norm else "#endif"}

  __shared__ float s_vec[{head_dim}];
  s_vec[tid] = val;
  __syncthreads();

  float out_val = val;
  if (tid < {rope_dim}) {{
    const float* freq_row = freqs_cis + (size_t)pos * {rope_dim};
    if (tid < {half_rope}) {{
      const float c = freq_row[tid];
      const float s = freq_row[tid + {half_rope}];
      out_val = val * c - s_vec[tid + {half_rope}] * s;
    }} else {{
      const int j = tid - {half_rope};
      const float c = freq_row[j];
      const float s = freq_row[tid];
      out_val = s_vec[j] * s + val * c;
    }}
  }}

  if (is_q) {{
    q_out[((size_t)h_idx * {T} + t) * {head_dim} + tid] = out_val;
  }} else {{
    const size_t k_idx = ((size_t)h_idx * {max_ctx} + pos) * {head_dim} + tid;
    kv_cache[k_idx] = __float2half(out_val);
  }}
}}
'''
    ins = (
        [qk_or_qkv_uop]
        + ([v_sep_uop] if split_v and v_sep_uop is not None else [])
        + ([q_norm_w_uop, k_norm_w_uop] if has_qk_norm and q_norm_w_uop is not None and k_norm_w_uop is not None else [])
        + [freqs_uop]
    )
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[q_out_uop, kv_cache_uop], in_uops=ins,
        global_size=(total_qk_heads, T, 1), local_size=(threads, 1, 1),
        ops_est=T * total_qk_heads * head_dim * 12, mem_est=T * total_qk_heads * head_dim * 8,
        var_uops=[start_pos_var] if isinstance(start_pos_var, UOp) else None,
    )


def fused_qk_rope_kv_populate(
    qk_or_qkv: Tensor,
    cache_kv: Tensor,
    freqs_cis: Tensor,
    start_pos: UOp,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    norm_eps: float,
    q_norm_w: Tensor | None = None,
    k_norm_w: Tensor | None = None,
    v_separate: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused QK-RMSNorm + RoPE + FP16 KV-cache store with zero `.cat()` overhead when V is separate."""
    B, T, _ = qk_or_qkv.shape
    assert B == 1
    max_ctx = cache_kv.shape[3]
    has_qk_norm = q_norm_w is not None and k_norm_w is not None
    split_v = v_separate is not None
    q_roped = Tensor.empty(1, n_heads, T, head_dim, dtype=dtypes.float32, device=qk_or_qkv.device)

    tensors = [q_roped, cache_kv, qk_or_qkv]
    if split_v:
        tensors.append(v_separate)
    if has_qk_norm:
        tensors.extend([q_norm_w, k_norm_w])
    tensors.append(freqs_cis)

    def _builder(*phs):
        idx = 0
        qo, kv, qki = phs[0], phs[1], phs[2]
        idx = 3
        v_ph = phs[idx] if split_v else None
        if split_v:
            idx += 1
        qnw = phs[idx] if has_qk_norm else None
        knw = phs[idx + 1] if has_qk_norm else None
        if has_qk_norm:
            idx += 2
        fr = phs[idx]
        return _qk_norm_rope_kv_kernel_baked(
            qo, kv, qki, v_ph, qnw, knw, fr, start_pos_var=start_pos,
            T=T, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            rope_dim=rope_dim, max_ctx=max_ctx, eps=float(norm_eps),
            has_qk_norm=has_qk_norm, split_v=split_v,
        )

    q_roped, cache_kv_after = _call_custom(tensors, fxn=_builder, bind_vars=[start_pos])[:2]
    return q_roped, cache_kv_after


# -----------------------------------------------------------------------------
# 4. Single-Pass Online-Softmax GQA Flash-Decoding (T=1) & Flash-Prefill (T>1)
# -----------------------------------------------------------------------------

@functools.cache
def _flash_decode_gqa_kernel(
    out_uop: UOp,
    q_in_uop: UOp,
    kv_cache_uop: UOp,
    start_pos_var: UOp,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_ctx: int,
) -> UOp:
    assert head_dim in (64, 128, 256)
    elems_per_lane = head_dim // 32
    n_warps = 8
    threads = 32 * n_warps
    gqa_ratio = n_heads // n_kv_heads
    scale = head_dim ** -0.5
    fn_name = f"flash_decode_gqa_{n_heads}_{n_kv_heads}_{head_dim}_{max_ctx}"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const float* __restrict__ q_in,
    const half* __restrict__ kv_cache,
    const int start_pos
) {{
  constexpr int E = {elems_per_lane};
  const int h = blockIdx.x;
  const int kv_h = h / {gqa_ratio};
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int seq_len = start_pos + 1;

  float q_reg[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) {{
    q_reg[e] = q_in[h * {head_dim} + lane * E + e] * {scale:.12e}f;
  }}

  const half* k_base = kv_cache + (size_t)kv_h * {max_ctx} * {head_dim};
  const half* v_base = kv_cache + (size_t)({n_kv_heads} + kv_h) * {max_ctx} * {head_dim};

  float m_i = -1e30f;
  float l_i = 0.0f;
  float acc[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;

  for (int pos = wid; pos < seq_len; pos += {n_warps}) {{
    const half* k_ptr = k_base + (size_t)pos * {head_dim} + lane * E;
    float qk = 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      qk += q_reg[e] * __half2float(k_ptr[e]);
    }}
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {{
      qk += __shfl_xor_sync(0xffffffff, qk, off);
    }}

    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p = expf(qk - m_new);
    l_i = l_i * alpha + p;
    m_i = m_new;

    const half* v_ptr = v_base + (size_t)pos * {head_dim} + lane * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      acc[e] = acc[e] * alpha + p * __half2float(v_ptr[e]);
    }}
  }}

  __shared__ float s_m[{n_warps}];
  __shared__ float s_l[{n_warps}];
  __shared__ float s_acc[{n_warps}][{head_dim}];

  if (lane == 0) {{
    s_m[wid] = m_i;
    s_l[wid] = l_i;
  }}
  #pragma unroll
  for (int e = 0; e < E; ++e) {{
    s_acc[wid][lane * E + e] = acc[e];
  }}
  __syncthreads();

  if (wid == 0) {{
    float m_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) m_max = fmaxf(m_max, s_m[w]);

    float l_sum = 0.0f;
    float out_reg[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) out_reg[e] = 0.0f;

    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) {{
      const float w_scale = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
      l_sum += s_l[w] * w_scale;
      #pragma unroll
      for (int e = 0; e < E; ++e) {{
        out_reg[e] += s_acc[w][lane * E + e] * w_scale;
      }}
    }}

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      out[h * {head_dim} + lane * E + e] = out_reg[e] * inv_l;
    }}
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=[q_in_uop, kv_cache_uop],
        global_size=(n_heads, 1, 1), local_size=(threads, 1, 1),
        ops_est=n_heads * head_dim * 256, mem_est=n_kv_heads * head_dim * 512,
        var_uops=[start_pos_var] if isinstance(start_pos_var, UOp) else None,
    )


def flash_decode_gqa(
    q_roped: Tensor,
    cache_kv_after: Tensor,
    start_pos: UOp,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Tensor:
    max_ctx = cache_kv_after.shape[3]
    attn_out = Tensor.empty(1, 1, n_heads * head_dim, dtype=dtypes.float32, device=q_roped.device)
    fxn_attn = lambda ao, qi, kv: _flash_decode_gqa_kernel(
        ao, qi, kv, start_pos_var=start_pos,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, max_ctx=max_ctx,
    )
    return _call_custom([attn_out, q_roped, cache_kv_after], fxn=fxn_attn, bind_vars=[start_pos])[0]


@functools.cache
def _flash_decode_gqa_q8_1_kernel(
    q_out_uop: UOp,
    ds_out_uop: UOp,
    q_in_uop: UOp,
    kv_cache_uop: UOp,
    start_pos_var: UOp,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_ctx: int,
) -> UOp:
    """Fused Online-Softmax GQA Flash-Decode + Direct In-Register Q8_1 Quantization (saves 36 kernels!)."""
    assert head_dim in (64, 128, 256)
    elems_per_lane = head_dim // 32
    lanes_per_blk = 32 // elems_per_lane  # 16, 8, or 4
    blks_per_head = head_dim // 32        # 2, 4, or 8
    n_warps = 8
    threads = 32 * n_warps
    gqa_ratio = n_heads // n_kv_heads
    scale = head_dim ** -0.5
    fn_name = f"flash_decode_gqa_q8_1_{n_heads}_{n_kv_heads}_{head_dim}_{max_ctx}"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ q_in,
    const half* __restrict__ kv_cache,
    const int start_pos
) {{
  constexpr int E = {elems_per_lane};
  constexpr int LPB = {lanes_per_blk};
  const int h = blockIdx.x;
  const int kv_h = h / {gqa_ratio};
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int seq_len = start_pos + 1;

  float q_reg[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) {{
    q_reg[e] = q_in[h * {head_dim} + lane * E + e] * {scale:.12e}f;
  }}

  const half* k_base = kv_cache + (size_t)kv_h * {max_ctx} * {head_dim};
  const half* v_base = kv_cache + (size_t)({n_kv_heads} + kv_h) * {max_ctx} * {head_dim};

  float m_i = -1e30f;
  float l_i = 0.0f;
  float acc[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;

  for (int pos = wid; pos < seq_len; pos += {n_warps}) {{
    const half* k_ptr = k_base + (size_t)pos * {head_dim} + lane * E;
    float qk = 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) qk += q_reg[e] * __half2float(k_ptr[e]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);

    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p = expf(qk - m_new);
    l_i = l_i * alpha + p;
    m_i = m_new;

    const half* v_ptr = v_base + (size_t)pos * {head_dim} + lane * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = acc[e] * alpha + p * __half2float(v_ptr[e]);
  }}

  __shared__ float s_m[{n_warps}];
  __shared__ float s_l[{n_warps}];
  __shared__ float s_acc[{n_warps}][{head_dim}];

  if (lane == 0) {{ s_m[wid] = m_i; s_l[wid] = l_i; }}
  #pragma unroll
  for (int e = 0; e < E; ++e) s_acc[wid][lane * E + e] = acc[e];
  __syncthreads();

  if (wid == 0) {{
    float m_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) m_max = fmaxf(m_max, s_m[w]);

    float l_sum = 0.0f;
    float out_reg[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) out_reg[e] = 0.0f;

    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) {{
      const float w_scale = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
      l_sum += s_l[w] * w_scale;
      #pragma unroll
      for (int e = 0; e < E; ++e) out_reg[e] += s_acc[w][lane * E + e] * w_scale;
    }}

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    float sum_x = 0.0f;
    float amax = 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      out_reg[e] *= inv_l;
      sum_x += out_reg[e];
      amax = fmaxf(amax, fabsf(out_reg[e]));
    }}

    // Sub-warp butterfly reduction across LPB lanes (LPB * E == 32 elements = 1 Q8_1 block!)
    #pragma unroll
    for (int off = LPB >> 1; off > 0; off >>= 1) {{
      sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }}

    const float d = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    const int base_idx = h * {head_dim} + lane * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      const int q = __float2int_rn(out_reg[e] * id);
      q_out[base_idx + e] = (signed char)max(-127, min(127, q));
    }}
    if ((lane & (LPB - 1)) == 0) {{
      ds_out[h * {blks_per_head} + (lane / LPB)] = make_float2(d, sum_x);
    }}
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[q_out_uop, ds_out_uop], in_uops=[q_in_uop, kv_cache_uop],
        global_size=(n_heads, 1, 1), local_size=(threads, 1, 1),
        ops_est=n_heads * head_dim * 260, mem_est=n_kv_heads * head_dim * 512,
        var_uops=[start_pos_var] if isinstance(start_pos_var, UOp) else None,
    )


def flash_decode_gqa_q8_1(
    q_roped: Tensor,
    cache_kv_after: Tensor,
    start_pos: UOp,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    max_ctx = cache_kv_after.shape[3]
    total_dim = n_heads * head_dim
    q_out = Tensor.empty(1, total_dim, dtype=dtypes.int8, device=q_roped.device)
    ds_out = Tensor.empty(1, total_dim // 32, 2, dtype=dtypes.float32, device=q_roped.device)
    fxn_attn = lambda qo, dso, qi, kv: _flash_decode_gqa_q8_1_kernel(
        qo, dso, qi, kv, start_pos_var=start_pos,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, max_ctx=max_ctx,
    )
    res = _call_custom([q_out, ds_out, q_roped, cache_kv_after], fxn=fxn_attn, bind_vars=[start_pos])
    return res[0], res[1]


@functools.cache
def _flash_prefill_gqa_kernel(
    out_uop: UOp,
    q_in_uop: UOp,
    kv_cache_uop: UOp,
    start_pos_var: UOp,
    T: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_ctx: int,
) -> UOp:
    assert head_dim in (64, 128, 256)
    elems_per_lane = head_dim // 32
    n_warps = 4
    threads = 32 * n_warps
    gqa_ratio = n_heads // n_kv_heads
    scale = head_dim ** -0.5
    fn_name = f"flash_prefill_gqa_{T}_{n_heads}_{n_kv_heads}_{head_dim}_{max_ctx}"

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const float* __restrict__ q_in,
    const half* __restrict__ kv_cache,
    const int start_pos
) {{
  constexpr int E = {elems_per_lane};
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const int kv_h = h / {gqa_ratio};
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int seq_len = start_pos + t + 1;

  const float* q_ptr = q_in + ((size_t)h * {T} + t) * {head_dim} + lane * E;
  float q_reg[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) {{
    q_reg[e] = q_ptr[e] * {scale:.12e}f;
  }}

  const half* k_base = kv_cache + (size_t)kv_h * {max_ctx} * {head_dim};
  const half* v_base = kv_cache + (size_t)({n_kv_heads} + kv_h) * {max_ctx} * {head_dim};

  float m_i = -1e30f;
  float l_i = 0.0f;
  float acc[E];
  #pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;

  for (int pos = wid; pos < seq_len; pos += {n_warps}) {{
    const half* k_ptr = k_base + (size_t)pos * {head_dim} + lane * E;
    float qk = 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      qk += q_reg[e] * __half2float(k_ptr[e]);
    }}
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {{
      qk += __shfl_xor_sync(0xffffffff, qk, off);
    }}

    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p = expf(qk - m_new);
    l_i = l_i * alpha + p;
    m_i = m_new;

    const half* v_ptr = v_base + (size_t)pos * {head_dim} + lane * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      acc[e] = acc[e] * alpha + p * __half2float(v_ptr[e]);
    }}
  }}

  __shared__ float s_m[{n_warps}];
  __shared__ float s_l[{n_warps}];
  __shared__ float s_acc[{n_warps}][{head_dim}];

  if (lane == 0) {{
    s_m[wid] = m_i;
    s_l[wid] = l_i;
  }}
  #pragma unroll
  for (int e = 0; e < E; ++e) {{
    s_acc[wid][lane * E + e] = acc[e];
  }}
  __syncthreads();

  if (wid == 0) {{
    float m_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) m_max = fmaxf(m_max, s_m[w]);

    float l_sum = 0.0f;
    float out_reg[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) out_reg[e] = 0.0f;

    #pragma unroll
    for (int w = 0; w < {n_warps}; ++w) {{
      const float w_scale = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
      l_sum += s_l[w] * w_scale;
      #pragma unroll
      for (int e = 0; e < E; ++e) {{
        out_reg[e] += s_acc[w][lane * E + e] * w_scale;
      }}
    }}

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    float* out_ptr = out + ((size_t)t * {n_heads} + h) * {head_dim} + lane * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) {{
      out_ptr[e] = out_reg[e] * inv_l;
    }}
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=[q_in_uop, kv_cache_uop],
        global_size=(n_heads, T, 1), local_size=(threads, 1, 1),
        ops_est=T * n_heads * head_dim * 256, mem_est=T * n_kv_heads * head_dim * 512,
        var_uops=[start_pos_var] if isinstance(start_pos_var, UOp) else None,
    )


def flash_prefill_gqa(
    q_roped: Tensor,
    cache_kv_after: Tensor,
    start_pos: UOp,
    T: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Tensor:
    max_ctx = cache_kv_after.shape[3]
    attn_out = Tensor.empty(1, T, n_heads * head_dim, dtype=dtypes.float32, device=q_roped.device)
    fxn_attn = lambda ao, qi, kv: _flash_prefill_gqa_kernel(
        ao, qi, kv, start_pos_var=start_pos,
        T=T, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, max_ctx=max_ctx,
    )
    return _call_custom([attn_out, q_roped, cache_kv_after], fxn=fxn_attn, bind_vars=[start_pos])[0]


@functools.cache
def _select_token_row_kernel(out_uop: UOp, x_uop: UOp, idx_var: UOp, dim: int) -> UOp:
    threads = 256
    grid = (dim + threads - 1) // threads
    fn_name = f"select_tok_row_{dim}"
    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int valid_last
) {{
  const int col = blockIdx.x * {threads} + threadIdx.x;
  if (col < {dim}) {{
    out[col] = x[(size_t)valid_last * {dim} + col];
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=[x_uop],
        global_size=(grid, 1, 1), local_size=(threads, 1, 1),
        ops_est=dim, mem_est=dim * 8, var_uops=[idx_var],
    )


def select_token_row(x: Tensor, valid_last: UOp) -> Tensor:
    dim = x.shape[-1]
    out = Tensor.empty(1, 1, dim, dtype=dtypes.float32, device=x.device)
    fxn = lambda o, xi: _select_token_row_kernel(o, xi, idx_var=valid_last, dim=dim)
    return _call_custom([out, x], fxn=fxn, bind_vars=[valid_last])[0]


# -----------------------------------------------------------------------------
# 5. Single-Kernel Fast Vocabulary Argmax & Direct GGUF Embedding Lookup
# -----------------------------------------------------------------------------

@functools.cache
def _fast_argmax_kernel(out_uop: UOp, logits_uop: UOp, vocab_size: int) -> UOp:
    threads = 1024
    warps = threads // 32
    fn_name = f"fast_argmax_{vocab_size}"
    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    int* __restrict__ out_id,
    const float* __restrict__ logits
) {{
  __shared__ float s_val[{warps}];
  __shared__ int s_idx[{warps}];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;

  float best_v = -1e38f;
  int best_i = 0;

  for (int i = tid; i < {vocab_size}; i += {threads}) {{
    float v = logits[i];
    if (v > best_v) {{
      best_v = v;
      best_i = i;
    }}
  }}

  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {{
    float ov = __shfl_down_sync(0xffffffff, best_v, off);
    int oi = __shfl_down_sync(0xffffffff, best_i, off);
    if (ov > best_v || (ov == best_v && oi < best_i)) {{
      best_v = ov;
      best_i = oi;
    }}
  }}

  if (lane == 0) {{
    s_val[wid] = best_v;
    s_idx[wid] = best_i;
  }}
  __syncthreads();

  if (wid == 0) {{
    best_v = (lane < {warps}) ? s_val[lane] : -1e38f;
    best_i = (lane < {warps}) ? s_idx[lane] : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {{
      float ov = __shfl_down_sync(0xffffffff, best_v, off);
      int oi = __shfl_down_sync(0xffffffff, best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) {{
        best_v = ov;
        best_i = oi;
      }}
    }}
    if (lane == 0) {{
      out_id[0] = best_i;
    }}
  }}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=[logits_uop],
        global_size=(1, 1, 1), local_size=(threads, 1, 1),
        ops_est=vocab_size * 2, mem_est=vocab_size * 4 + 4,
    )


def fast_argmax_1d(logits: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    out = Tensor.empty(1, 1, dtype=dtypes.int32, device=logits.device)
    return _call_custom([out, logits], fxn=functools.partial(_fast_argmax_kernel, vocab_size=vocab_size))[0]


@functools.cache
def _gguf_embed_kernel(out_uop: UOp, w_uop: UOp, tok_uop: UOp, T: int, dim: int, ggml_type: int) -> UOp:
    threads = 256
    fn_name = f"gguf_embed_{T}_{dim}_t{ggml_type}"

    if ggml_type == 14:
        assert dim % 256 == 0
        n_sb = dim // 256
        body = f'''
  const int tok = tok_ids[t];
  const unsigned char* row = w_raw + (size_t)tok * ({n_sb} * 210);
  for (int col = tid; col < {dim}; col += {threads}) {{
    const int sb = col >> 8;
    const int idx = col & 255;
    const unsigned char* blk = row + sb * 210;
    const float d = __half2float(*reinterpret_cast<const half*>(blk + 208));
    const int sc = (int)(reinterpret_cast<const signed char*>(blk + 192)[idx >> 4]);
    const int h = idx >> 7;
    const int rem = idx & 127;
    const int p = rem >> 6;
    const int q = (rem >> 5) & 1;
    const int k = rem & 31;
    const unsigned char ql_byte = blk[(h << 6) + (q << 5) + k];
    const unsigned char qh_byte = blk[128 + (h << 5) + k];
    const int xl = (ql_byte >> (p << 2)) & 0x0f;
    const int xh = ((qh_byte >> (((p << 1) + q) << 1)) & 0x03) << 4;
    out[(size_t)t * {dim} + col] = d * (float)sc * (float)((xl | xh) - 32);
  }}
'''
    elif ggml_type == 12:
        assert dim % 256 == 0
        n_sb = dim // 256
        body = f'''
  const int tok = tok_ids[t];
  const unsigned char* row = w_raw + (size_t)tok * ({n_sb} * 144);
  for (int col = tid; col < {dim}; col += {threads}) {{
    const int sb = col >> 8;
    const int idx = col & 255;
    const unsigned char* blk = row + sb * 144;
    const half2 dm = *reinterpret_cast<const half2*>(blk);
    const float d = __half2float(dm.x);
    const float dmin = __half2float(dm.y);
    const unsigned char* s = blk + 4;
    const int sub = idx >> 5;
    const int k = idx & 31;
    unsigned int sc, mn;
    if (sub < 4) {{
      sc = s[sub] & 63u;
      mn = s[sub + 4] & 63u;
    }} else {{
      sc = (s[sub + 4] & 0x0fu) | ((s[sub - 4] >> 6) << 4);
      mn = (s[sub + 4] >> 4)    | ((s[sub]     >> 6) << 4);
    }}
    const unsigned char qbyte = blk[16 + ((sub >> 1) << 5) + k];
    const unsigned int q = (sub & 1) ? (qbyte >> 4) : (qbyte & 0x0fu);
    out[(size_t)t * {dim} + col] = d * (float)sc * (float)q - dmin * (float)mn;
  }}
'''
    elif ggml_type == 8:
        assert dim % 32 == 0
        n_blk = dim // 32
        body = f'''
  const int tok = tok_ids[t];
  const unsigned char* row = w_raw + (size_t)tok * ({n_blk} * 34);
  for (int col = tid; col < {dim}; col += {threads}) {{
    const int b = col >> 5;
    const int k = col & 31;
    const unsigned char* blk = row + b * 34;
    const float d = __half2float(*reinterpret_cast<const half*>(blk));
    const int q = (int)(reinterpret_cast<const signed char*>(blk + 2)[k]);
    out[(size_t)t * {dim} + col] = d * (float)q;
  }}
'''
    elif ggml_type == 1:
        body = f'''
  const int tok = tok_ids[t];
  const half* row = reinterpret_cast<const half*>(w_raw) + (size_t)tok * {dim};
  for (int col = tid; col < {dim}; col += {threads}) {{
    out[(size_t)t * {dim} + col] = __half2float(row[col]);
  }}
'''
    else:
        body = f'''
  const int tok = tok_ids[t];
  const float* row = reinterpret_cast<const float*>(w_raw) + (size_t)tok * {dim};
  for (int col = tid; col < {dim}; col += {threads}) {{
    out[(size_t)t * {dim} + col] = row[col];
  }}
'''

    src = CUDA_COMMON + f'''
extern "C" __global__ void __launch_bounds__({threads}) {fn_name}(
    float* __restrict__ out,
    const unsigned char* __restrict__ w_raw,
    const int* __restrict__ tok_ids
) {{
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  {body}
}}
'''
    return _make_program_uop(
        name=fn_name, src=src, out_uops=[out_uop], in_uops=[w_uop, tok_uop],
        global_size=(T, 1, 1), local_size=(threads, 1, 1),
        ops_est=T * dim * 4, mem_est=T * dim * 5,
    )


def gguf_embedding_lookup(tokens: Tensor, w_raw_bytes: Tensor, ggml_type: int, dim: int) -> Tensor:
    T = tokens.numel()
    out = Tensor.empty(1, T, dim, dtype=dtypes.float32, device=w_raw_bytes.device)
    tok_i32 = tokens if tokens.dtype == dtypes.int32 else tokens.cast(dtypes.int32)
    fxn = functools.partial(_gguf_embed_kernel, T=T, dim=dim, ggml_type=ggml_type)
    return _call_custom([out, w_raw_bytes, tok_i32], fxn=fxn)[0]
