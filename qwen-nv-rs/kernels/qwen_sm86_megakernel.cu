// =============================================================================
// Custom GA102 (RTX 3080, sm_86, 68 SMs) Persistent Megakernel & Prefill Suite
// for Qwen2.5 / Qwen3 Raw GGUF (Q4_K, Q6_K, Q8_0) Inference over Thunderbolt.
//
// Tuned specifically for MacBook Pro + Razer Core X Chroma TB3 eGPU:
// - 256 threads/CTA (8 warps = 16 Q4_K half-warps/SM, 1088 concurrent rows across
//   68 SMs) -> < 31 KB SMEM (fits 32 KB SMEM + 96 KB L1 config!) & 0B stack spills!
// - Standard PCIe Posted MWr (no `.sys` strong-ordering bus locks over TB3 tunnel).
// - `nanosleep.u32` inside monotonic L2 barrier to prevent PSU di/dt spikes.
// =============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

static constexpr int MK_THREADS  = 512;
static constexpr int MK_WARPS    = 16; // MK_THREADS / 32 (16 warps/SM -> 2x GDDR6X MLP!)
static constexpr int MK_HW       = 32; // MK_WARPS * 2 (32 Q4_K half-warps/SM = 2176 concurrent rows!)

static constexpr int BM4_THREADS = 512;
static constexpr int BM4_WARPS   = 16;
static constexpr int BM4_HW      = 32;
static constexpr int BM4_SMEM_K  = 4096;

// -----------------------------------------------------------------------------
// Shared Host/Device ABI Structures (64-bit aligned, #[repr(C)] in Rust)
// -----------------------------------------------------------------------------

struct __align__(16) QwenLayerDesc {
    uint64_t attn_norm_w;   // const float* [dim]
    uint64_t ffn_norm_w;    // const float* [dim]
    uint64_t q_norm_w;      // const float* [head_dim] (0 if !has_qk_norm)
    uint64_t k_norm_w;      // const float* [head_dim] (0 if !has_qk_norm)

    uint64_t w_qkv_or_qk;   // const uint8_t*
    uint64_t b_qkv_or_qk;   // const float* (0 if none)
    uint64_t w_v;           // const uint8_t* (if split_v == 1)
    uint64_t b_v;           // const float* (0 if none)

    uint64_t w_o;           // const uint8_t* [dim, q_dim]
    uint64_t b_o;           // const float* (0 if none)

    uint64_t w_gate_up;     // const uint8_t* [2*hidden_dim, dim] (or [hidden_dim, dim] if split_gate_up)
    uint64_t w_up_sep;      // const uint8_t* [hidden_dim, dim] (if split_gate_up == 1)
    uint64_t w_down;        // const uint8_t* [dim, hidden_dim]
    uint64_t b_down;        // const float* (0 if none)

    uint64_t kv_cache;      // half* [2, n_kv_heads, max_ctx, head_dim]

    int32_t  qkv_type;      // 12=Q4_K, 14=Q6_K, 8=Q8_0
    int32_t  v_type;
    int32_t  o_type;
    int32_t  gate_up_type;
    int32_t  up_sep_type;
    int32_t  down_type;
    int32_t  split_v;
    int32_t  split_gate_up;
};

struct __align__(16) QwenModelDesc {
    uint64_t layers_ptr;      // const QwenLayerDesc* [num_layers]
    uint64_t emb_w;           // const uint8_t* [vocab_size, dim]
    uint64_t output_norm_w;   // const float* [dim]
    uint64_t lm_head_w;       // const uint8_t* [vocab_size, dim]
    uint64_t freqs_cis;       // const float* [max_ctx, rope_dim]

    // VRAM L2-resident scratch buffers
    uint64_t ws_x_res;        // float* [4 * max_dim]
    uint64_t ws_x_res2;       // float* [4 * max_dim]
    uint64_t ws_xq;           // signed char* [4 * max_hidden]
    uint64_t ws_xds;          // float2* [4 * (max_hidden / 32)]
    uint64_t ws_qkv;          // float* [4 * max_qkv]
    uint64_t ws_q_roped;      // float* [max_heads * 4 * max_head_dim]
    uint64_t ws_gate_up;      // float* [4 * 2 * max_hidden]
    uint64_t ws_cta_val;      // float* [4 * 128]
    uint64_t ws_cta_idx;      // int32_t* [4 * 128]
    uint64_t ws_bar_state;    // uint64_t* [4]: [0]=monotonic_cnt, [1]=step_base

    int32_t  num_layers;
    int32_t  dim;
    int32_t  hidden_dim;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  rope_dim;
    int32_t  vocab_size;
    int32_t  max_ctx;
    int32_t  has_qk_norm;
    int32_t  emb_type;
    int32_t  lm_head_type;
    float    norm_eps;
    float    inv_sqrt_head_dim;
    int32_t  stop_id0;
    int32_t  stop_id1;
    int32_t  stop_id2;
    int32_t  stop_id3;
};

// Host-mapped control & zero-doorbell streaming ring in MAP_SYSMEM_FD (uncached snooped PTE)
struct __align__(64) QwenDecodeCtrl {
    int32_t  start_token;
    int32_t  start_pos;
    int32_t  max_steps;
    float    temperature;
    uint64_t rng_seed;
    int32_t  cancel_flag;       // host writes 1 to abort rollout early
    int32_t  error_code;        // GPU writes >0 if hardware watchdog ever trips
    int32_t  steps_completed;   // GPU publishes after each token via posted PCIe MWr
    int32_t  stop_reason;       // 0=max_steps, 1=eos, 2=cancelled, 3=ctx_limit
    int32_t  spec_tokens_in[4]; // candidate tokens [t0, t1, t2, t3] for BM=4 spec verify
    int32_t  spec_tokens_out[4];// greedy argmax output for each of the 4 spec positions
    int32_t  tokens_out[4096];  // streamed token IDs for zero-doorbell multi-token rollout
};

// -----------------------------------------------------------------------------
// Ampere sm_86 PTX Cache-Controlled Memory & Warp Primitives
// -----------------------------------------------------------------------------

__device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void st_cg_f32(float* p, float v) {
    asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}

__device__ __forceinline__ int4 ld_cg_s32x4(const int* p) {
    int4 v;
    asm volatile("ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float2 ld_cg_f32x2(const float2* p) {
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
                 : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ void st_cg_s8(signed char* p, int v) {
    asm volatile("st.global.cg.s8 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_cg_f32x2(float2* p, float2 v) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(v.x), "f"(v.y) : "memory");
}

__device__ __forceinline__ void st_cg_f16(half* p, half v) {
    unsigned short bits = __half_as_ushort(v);
    asm volatile("st.global.cg.u16 [%0], %1;" :: "l"(p), "h"(bits) : "memory");
}

// Posted PCIe write/read for uncached snooped MAP_SYSMEM_FD (zero `.sys` bus lock TLPs!)
__device__ __forceinline__ void st_host_s32(volatile int32_t* p, int32_t v) {
    asm volatile("st.global.wt.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int32_t ld_host_s32(const volatile int32_t* p) {
    int32_t v;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Read-only 128B-coalesced L1/Texture weight load (`LDG.E.128.CONSTANT`)
__device__ __forceinline__ uint4 ld_cs_u32x4(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ unsigned int load_u32_align2(const unsigned char* p) {
    const unsigned short* p16 = reinterpret_cast<const unsigned short*>(p);
    return ((unsigned int)__ldg(&p16[0])) | (((unsigned int)__ldg(&p16[1])) << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float half_warp_reduce_sum(float val) {
    const unsigned int mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float fast_uniform_01(uint64_t seed, uint32_t step, uint32_t idx) {
    uint64_t x = seed + ((uint64_t)step * 0x9E3779B97F4A7C15ULL) + ((uint64_t)idx * 0xBF58476D1CE4E5B9ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    uint32_t u24 = (uint32_t)(x >> 40);
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, (float)u24 * (1.0f / 16777216.0f)));
}

// Monotonic L2 Hardware Grid Barrier with 200ms Watchdog & nanosleep power-smoothing
__device__ __forceinline__ bool megakernel_barrier(
    uint64_t* bar_cnt,
    uint64_t& step,
    int num_blocks,
    int& s_abort,
    volatile QwenDecodeCtrl* ctrl
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        step += 1ULL;
        const uint64_t target = step * (uint64_t)num_blocks;
        uint64_t old_val;
        asm volatile("atom.add.acq_rel.gpu.u64 %0, [%1], 1;" : "=l"(old_val) : "l"(bar_cnt) : "memory");
        uint64_t cur = old_val + 1ULL;
        if (cur < target) {
            uint32_t spin = 0;
            do {
                asm volatile("nanosleep.u32 16;");
                asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(cur) : "l"(bar_cnt) : "memory");
                if (++spin > 2000000U) {
                    s_abort = 1;
                    st_host_s32(&ctrl->error_code, (int32_t)step);
                    break;
                }
            } while (cur < target);
        }
    }
    __syncthreads();
    return (s_abort == 0);
}

// -----------------------------------------------------------------------------
// Device Building Blocks Inside the 68-SM Persistent Megakernel (256 threads/CTA)
// -----------------------------------------------------------------------------

__device__ inline void dev_embed_row_to_l2(
    float* x_res,
    const unsigned char* emb_w,
    int emb_type,
    int tok_id,
    int dim
) {
    const int tid = threadIdx.x;
    if (emb_type == 14) {
        const int n_sb = dim >> 8;
        const unsigned char* row = emb_w + (size_t)tok_id * (n_sb * 210);
        const int nthreads = (int)blockDim.x;
        for (int col = tid; col < dim; col += nthreads) {
            const int sb = col >> 8, idx = col & 255;
            const unsigned char* blk = row + sb * 210;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208)));
            const int sc = (int)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[idx >> 4]);
            const int h = idx >> 7, rem = idx & 127, p = rem >> 6, q = (rem >> 5) & 1, k = rem & 31;
            const unsigned char ql_byte = __ldg(&blk[(h << 6) + (q << 5) + k]);
            const unsigned char qh_byte = __ldg(&blk[128 + (h << 5) + k]);
            const int xl = (ql_byte >> (p << 2)) & 0x0f;
            const int xh = ((qh_byte >> (((p << 1) + q) << 1)) & 0x03) << 4;
            st_cg_f32(&x_res[col], d * (float)sc * (float)((xl | xh) - 32));
        }
    } else if (emb_type == 12) {
        const int n_sb = dim >> 8;
        const unsigned char* row = emb_w + (size_t)tok_id * (n_sb * 144);
        const int nthreads = (int)blockDim.x;
        for (int col = tid; col < dim; col += nthreads) {
            const int sb = col >> 8, idx = col & 255;
            const unsigned char* blk = row + sb * 144;
            const half2 dm = __ldg(reinterpret_cast<const half2*>(blk));
            const float d = __half2float(dm.x), dmin = __half2float(dm.y);
            const unsigned char* s = blk + 4;
            const int sub = idx >> 5, k = idx & 31;
            unsigned int sc, mn;
            if (sub < 4) { sc = __ldg(&s[sub]) & 63u; mn = __ldg(&s[sub + 4]) & 63u; }
            else {
                sc = (__ldg(&s[sub + 4]) & 0x0fu) | ((__ldg(&s[sub - 4]) >> 6) << 4);
                mn = (__ldg(&s[sub + 4]) >> 4)    | ((__ldg(&s[sub])     >> 6) << 4);
            }
            const unsigned char qbyte = __ldg(&blk[16 + ((sub >> 1) << 5) + k]);
            const unsigned int q = (sub & 1) ? (qbyte >> 4) : (qbyte & 0x0fu);
            st_cg_f32(&x_res[col], d * (float)sc * (float)q - dmin * (float)mn);
        }
    } else if (emb_type == 8) {
        const int n_blk = dim >> 5;
        const unsigned char* row = emb_w + (size_t)tok_id * (n_blk * 34);
        const int nthreads = (int)blockDim.x;
        for (int col = tid; col < dim; col += nthreads) {
            const int b = col >> 5, k = col & 31;
            const unsigned char* blk = row + b * 34;
            st_cg_f32(&x_res[col], __half2float(__ldg(reinterpret_cast<const half*>(blk))) *
                                   (float)__ldg(&reinterpret_cast<const signed char*>(blk + 2)[k]));
        }
    } else if (emb_type == 1) {
        const int nthreads = (int)blockDim.x;
        const half* row = reinterpret_cast<const half*>(emb_w) + (size_t)tok_id * dim;
        for (int col = tid; col < dim; col += nthreads) st_cg_f32(&x_res[col], __half2float(__ldg(&row[col])));
    } else {
        const int nthreads = (int)blockDim.x;
        const float* row = reinterpret_cast<const float*>(emb_w) + (size_t)tok_id * dim;
        for (int col = tid; col < dim; col += nthreads) st_cg_f32(&x_res[col], __ldg(&row[col]));
    }
}

__device__ inline void dev_rmsnorm_q8_1_to_smem(
    int* s_xq_i32,
    float2* s_xds,
    float* s_red,
    const float* x_res,
    const float* norm_w,
    int dim,
    float eps
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5; // 0..7
    const int n_blks = dim >> 5;

    float ss = 0.0f;
    for (int blk = wid; blk < n_blks; blk += MK_WARPS) {
        const float v = ld_cg_f32(&x_res[(blk << 5) + lane]);
        ss += v * v;
    }

    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();

    if (wid == 0) {
        float w_ss = (lane < MK_WARPS) ? s_red[lane] : 0.0f;
        w_ss = warp_reduce_sum(w_ss);
        if (lane == 0) s_red[0] = rsqrtf(w_ss / (float)dim + eps);
    }
    __syncthreads();
    const float scale = s_red[0];

    signed char* s_xq_s8 = reinterpret_cast<signed char*>(s_xq_i32);
    for (int blk = wid; blk < n_blks; blk += MK_WARPS) {
        const int idx = (blk << 5) + lane;
        const float v = ld_cg_f32(&x_res[idx]) * scale * __ldg(&norm_w[idx]);
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        s_xq_s8[idx] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
        if (lane == 0) s_xds[blk] = make_float2(d, sum_x);
    }
    __syncthreads();
}

__device__ inline void dev_load_q8_1_to_smem(
    int* s_xq_i32,
    float2* s_xds,
    const signed char* g_xq,
    const float2* g_xds,
    int K
) {
    const int tid = threadIdx.x;
    const int n_int4 = K >> 4;
    const int* g_i32 = reinterpret_cast<const int*>(g_xq);
    int4* s_i4 = reinterpret_cast<int4*>(s_xq_i32);
    for (int i = tid; i < n_int4; i += MK_THREADS) {
        s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
    }
    const int n_blks = K >> 5;
    for (int b = tid; b < n_blks; b += MK_THREADS) {
        s_xds[b] = ld_cg_f32x2(&g_xds[b]);
    }
    __syncthreads();
}

// Unified GEMV + Optional In-Register LM_Head Argmax/Gumbel Selector (`mode_select == true`)
// Keeping ONE unified GEMV body cuts instruction cache footprint in half and gives 0B stack spills!
__device__ __forceinline__ void dev_gemv_or_select_smem(
    float* out,
    float* cta_best_val,
    int* cta_best_idx,
    float* s_val,
    int* s_idx,
    const unsigned char* w_raw,
    int w_type,
    const int* s_xq,
    const float2* s_xds,
    const float* bias,
    const float* residual,
    int N,
    int K,
    int sm_idx,
    int sm_count,
    bool mode_select,
    float temperature,
    uint64_t rng_seed,
    uint32_t step_idx
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;   // 0..7
    const int lane = tid & 31;

    float local_best_v = -1e38f;
    int local_best_i = 0x7fffffff;
    const bool use_temp = mode_select && (temperature > 1e-6f);
    const float inv_temp = use_temp ? (1.0f / temperature) : 1.0f;

    if (w_type == 12) {
        // Q4_K: 16-lane half-warp per row (2 rows/warp = 16 rows/CTA!)
        const int n_sb = K >> 8;
        const int hw_in_cta = (wid << 1) | (lane >> 4); // 0..15
        const int total_hw = sm_count * MK_HW;
        const int lane16 = lane & 15;
        const int sb_lane = lane16 >> 3;
        const int in_sb = lane16 & 7;
        const int chunk = in_sb >> 1;
        const int half16 = in_sb & 1;

        for (int row = sm_idx * MK_HW + hw_in_cta; row < N; row += total_hw) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 144);
            float acc = 0.0f;

            #pragma unroll 2
            for (int sb = sb_lane; sb < n_sb; sb += 2) {
                const unsigned char* blk = w_row + sb * 144;
                const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
                const half2 dm = *reinterpret_cast<const half2*>(&hdr.x);
                const float d = __half2float(dm.x), dmin = __half2float(dm.y);
                const unsigned int s0 = hdr.y, s1 = hdr.z, s2 = hdr.w;

                unsigned int sc_lo, sc_hi, mn_lo, mn_hi;
                if (chunk < 2) {
                    const int sh0 = (chunk << 1) << 3, sh1 = ((chunk << 1) + 1) << 3;
                    sc_lo = (s0 >> sh0) & 0x3fu; sc_hi = (s0 >> sh1) & 0x3fu;
                    mn_lo = (s1 >> sh0) & 0x3fu; mn_hi = (s1 >> sh1) & 0x3fu;
                } else {
                    const int sh0 = ((chunk - 2) << 1) << 3, sh1 = (((chunk - 2) << 1) + 1) << 3;
                    sc_lo = ((s2 >> sh0) & 0x0fu) | (((s0 >> (sh0 + 6)) & 0x03u) << 4);
                    sc_hi = ((s2 >> sh1) & 0x0fu) | (((s0 >> (sh1 + 6)) & 0x03u) << 4);
                    mn_lo = ((s2 >> (sh0 + 4)) & 0x0fu) | (((s1 >> (sh0 + 6)) & 0x03u) << 4);
                    mn_hi = ((s2 >> (sh1 + 4)) & 0x0fu) | (((s1 >> (sh1 + 6)) & 0x03u) << 4);
                }

                const uint4 qv = ld_cs_u32x4(blk + 16 + (in_sb << 4));
                const int ql0 = (int)(qv.x & 0x0f0f0f0fu), ql1 = (int)(qv.y & 0x0f0f0f0fu);
                const int ql2 = (int)(qv.z & 0x0f0f0f0fu), ql3 = (int)(qv.w & 0x0f0f0f0fu);
                const int qh0 = (int)((qv.x >> 4) & 0x0f0f0f0fu), qh1 = (int)((qv.y >> 4) & 0x0f0f0f0fu);
                const int qh2 = (int)((qv.z >> 4) & 0x0f0f0f0fu), qh3 = (int)((qv.w >> 4) & 0x0f0f0f0fu);

                const int x_off_int = (sb << 6) + (chunk << 4) + (half16 << 2);
                const int sub0 = (sb << 3) + (chunk << 1);
                const int4 x_lo = *reinterpret_cast<const int4*>(s_xq + x_off_int);
                const int4 x_hi = *reinterpret_cast<const int4*>(s_xq + x_off_int + 8);
                const float2 ds0 = s_xds[sub0], ds1 = s_xds[sub0 + 1];

                int sumi_lo = __dp4a(ql0, x_lo.x, 0); sumi_lo = __dp4a(ql1, x_lo.y, sumi_lo);
                sumi_lo = __dp4a(ql2, x_lo.z, sumi_lo); sumi_lo = __dp4a(ql3, x_lo.w, sumi_lo);
                int sumi_hi = __dp4a(qh0, x_hi.x, 0); sumi_hi = __dp4a(qh1, x_hi.y, sumi_hi);
                sumi_hi = __dp4a(qh2, x_hi.z, sumi_hi); sumi_hi = __dp4a(qh3, x_hi.w, sumi_hi);

                acc += (d * (float)sc_lo) * ds0.x * (float)sumi_lo
                     + (d * (float)sc_hi) * ds1.x * (float)sumi_hi;
                if (half16 == 0) acc -= (dmin * (float)mn_lo) * ds0.y + (dmin * (float)mn_hi) * ds1.y;
            }

            acc = half_warp_reduce_sum(acc);
            if (lane16 == 0) {
                if (mode_select) {
                    float score = use_temp ? (acc * inv_temp - logf(-logf(fast_uniform_01(rng_seed, step_idx, (uint32_t)row)))) : acc;
                    if (score > local_best_v || (score == local_best_v && row < local_best_i)) {
                        local_best_v = score; local_best_i = row;
                    }
                } else {
                    if (bias != nullptr) acc += __ldg(&bias[row]);
                    if (residual != nullptr) acc += ld_cg_f32(&residual[row]);
                    st_cg_f32(&out[row], acc);
                }
            }
        }
    } else if (w_type == 14) {
        const int n_sb = K >> 8;
        const int total_w = sm_count * MK_WARPS;
        const int sb_lane = lane >> 4, tid16 = lane & 15;
        const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
        const int ql_off = (h << 6) + (q << 5) + (half8 << 4);
        const int qh_off = 128 + (h << 5) + (half8 << 4);
        const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;

        for (int row = sm_idx * MK_WARPS + wid; row < N; row += total_w) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 210);
            float acc = 0.0f;

            #pragma unroll 2
            for (int sb = sb_lane; sb < n_sb; sb += 2) {
                const unsigned char* blk = w_row + sb * 210;
                const float d_sc = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208))) *
                                   (float)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[tid16]);
                const unsigned char* ql_ptr = blk + ql_off;
                const unsigned char* qh_ptr = blk + qh_off;
                const unsigned int ql0 = load_u32_align2(ql_ptr + 0), ql1 = load_u32_align2(ql_ptr + 4);
                const unsigned int ql2 = load_u32_align2(ql_ptr + 8), ql3 = load_u32_align2(ql_ptr + 12);
                const unsigned int qh0 = load_u32_align2(qh_ptr + 0), qh1 = load_u32_align2(qh_ptr + 4);
                const unsigned int qh2 = load_u32_align2(qh_ptr + 8), qh3 = load_u32_align2(qh_ptr + 12);

                const unsigned int u6_0 = ((ql0 >> ql_sh) & 0x0f0f0f0fu) | (((qh0 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_1 = ((ql1 >> ql_sh) & 0x0f0f0f0fu) | (((qh1 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_2 = ((ql2 >> ql_sh) & 0x0f0f0f0fu) | (((qh2 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_3 = ((ql3 >> ql_sh) & 0x0f0f0f0fu) | (((qh3 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
                const unsigned int m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
                const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
                const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
                const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
                const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));

                const int x_base_int = (sb << 6) + (tid16 << 2);
                const int ds_idx = (sb << 3) + (tid16 >> 1);
                const int4 xv = *reinterpret_cast<const int4*>(s_xq + x_base_int);
                int sumi = __dp4a(s6_0, xv.x, 0); sumi = __dp4a(s6_1, xv.y, sumi);
                sumi = __dp4a(s6_2, xv.z, sumi); sumi = __dp4a(s6_3, xv.w, sumi);
                acc += (d_sc * s_xds[ds_idx].x) * (float)sumi;
            }

            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                if (mode_select) {
                    float score = use_temp ? (acc * inv_temp - logf(-logf(fast_uniform_01(rng_seed, step_idx, (uint32_t)row)))) : acc;
                    if (score > local_best_v || (score == local_best_v && row < local_best_i)) {
                        local_best_v = score; local_best_i = row;
                    }
                } else {
                    if (bias != nullptr) acc += __ldg(&bias[row]);
                    if (residual != nullptr) acc += ld_cg_f32(&residual[row]);
                    st_cg_f32(&out[row], acc);
                }
            }
        }
    } else {
        const int n_blk = K >> 5;
        const int total_w = sm_count * MK_WARPS;
        const int blk_lane = lane >> 1, half16 = lane & 1;

        for (int row = sm_idx * MK_WARPS + wid; row < N; row += total_w) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_blk * 34);
            float acc = 0.0f;
            for (int b = blk_lane; b < n_blk; b += 16) {
                const unsigned char* blk = w_row + b * 34;
                const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                const unsigned char* qs = blk + 2 + (half16 << 4);
                const int w0 = (int)load_u32_align2(qs + 0), w1 = (int)load_u32_align2(qs + 4);
                const int w2 = (int)load_u32_align2(qs + 8), w3 = (int)load_u32_align2(qs + 12);
                const int x_off_int = (b << 3) + (half16 << 2);
                const int4 xv = *reinterpret_cast<const int4*>(s_xq + x_off_int);
                int sumi = __dp4a(w0, xv.x, 0); sumi = __dp4a(w1, xv.y, sumi);
                sumi = __dp4a(w2, xv.z, sumi); sumi = __dp4a(w3, xv.w, sumi);
                acc += (d * s_xds[b].x) * (float)sumi;
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                if (mode_select) {
                    float score = use_temp ? (acc * inv_temp - logf(-logf(fast_uniform_01(rng_seed, step_idx, (uint32_t)row)))) : acc;
                    if (score > local_best_v || (score == local_best_v && row < local_best_i)) {
                        local_best_v = score; local_best_i = row;
                    }
                } else {
                    if (bias != nullptr) acc += __ldg(&bias[row]);
                    if (residual != nullptr) acc += ld_cg_f32(&residual[row]);
                    st_cg_f32(&out[row], acc);
                }
            }
        }
    }

    if (mode_select) {
        __syncthreads();
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, local_best_v, off);
            int oi = __shfl_down_sync(0xffffffff, local_best_i, off);
            if (ov > local_best_v || (ov == local_best_v && oi < local_best_i)) {
                local_best_v = ov; local_best_i = oi;
            }
        }
        if (lane == 0) {
            s_val[wid] = local_best_v;
            s_idx[wid] = local_best_i;
        }
        __syncthreads();
        if (wid == 0) {
            float bv = (lane < MK_WARPS) ? s_val[lane] : -1e38f;
            int bi = (lane < MK_WARPS) ? s_idx[lane] : 0x7fffffff;
            #pragma unroll
            for (int off = (MK_WARPS >> 1); off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                st_cg_f32(&cta_best_val[blockIdx.x], bv);
                asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&cta_best_idx[blockIdx.x]), "r"(bi) : "memory");
            }
        }
    }
}

// =============================================================================
// 1. PERSISTENT 68-SM MULTI-TOKEN AUTOREGRESSIVE MEGAKERNEL (`T=1`, 256 threads)
// =============================================================================

// `__launch_bounds__(576, 1)` caps `ptxas` at `~104` regs/thread -> `0B` stack spills,
// while launching `MK_THREADS = 512` uses `512 * (104 + 16 GT200_DEBUGGER) = 61,440 <= 65,536` regs/SM!
extern "C" __global__ __launch_bounds__(576, 1)
void qwen_persistent_decode_megakernel(
    const QwenModelDesc* __restrict__ m,
    volatile QwenDecodeCtrl* __restrict__ ctrl
) {
    // 29.8 KB Shared Memory per SM -> Fits inside 32KB SMEM config, leaving 96KB L1 Data Cache!
    // MUST be 16-byte aligned (`__align__(16)`) for `STS.128` / `LDS.128` (`int4`) accesses!
    __shared__ __align__(16) int    s_xq[20480 / 4];              // 20,480 B
    __shared__ __align__(16) float2 s_xds[20480 / 32];            //  5,120 B
    __shared__ __align__(16) float  s_scratch[MK_WARPS * 128 + 64]; // 8,448 B
    __shared__ __align__(16) int    s_idx_red[32];
    __shared__ int    s_abort;
    __shared__ int    s_cur_tok;
    __shared__ int    s_cur_pos;
    __shared__ int    s_max_steps;
    __shared__ float  s_temp;
    __shared__ uint64_t s_seed;
    __shared__ int    s_done;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(m->ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    if (tid == 0) {
        s_abort = 0;
        s_done = 0;
        s_cur_tok = ctrl->start_token;
        s_cur_pos = ctrl->start_pos;
        s_max_steps = ctrl->max_steps;
        s_temp = ctrl->temperature;
        s_seed = ctrl->rng_seed;
    }
    __syncthreads();

    const QwenLayerDesc* layers = reinterpret_cast<const QwenLayerDesc*>(m->layers_ptr);
    float* x_res = reinterpret_cast<float*>(m->ws_x_res);
    float* x_res2 = reinterpret_cast<float*>(m->ws_x_res2);
    signed char* ws_xq = reinterpret_cast<signed char*>(m->ws_xq);
    float2* ws_xds = reinterpret_cast<float2*>(m->ws_xds);
    float* ws_qkv = reinterpret_cast<float*>(m->ws_qkv);
    float* ws_q_roped = reinterpret_cast<float*>(m->ws_q_roped);
    float* ws_gate_up = reinterpret_cast<float*>(m->ws_gate_up);
    float* ws_cta_val = reinterpret_cast<float*>(m->ws_cta_val);
    int* ws_cta_idx = reinterpret_cast<int*>(m->ws_cta_idx);
    const float* freqs_cis = reinterpret_cast<const float*>(m->freqs_cis);

    const int dim = m->dim;
    const int hidden_dim = m->hidden_dim;
    const int n_heads = m->n_heads;
    const int n_kv_heads = m->n_kv_heads;
    const int head_dim = m->head_dim;
    const int rope_dim = m->rope_dim;
    const int half_rope = rope_dim >> 1;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int gqa_ratio = n_heads / n_kv_heads;
    const int max_ctx = m->max_ctx;
    const float eps = m->norm_eps;
    const float attn_scale = m->inv_sqrt_head_dim;

    for (int step_idx = 0; step_idx < s_max_steps; ++step_idx) {
        const int cur_tok = s_cur_tok;
        const int pos = s_cur_pos;

        if (bid == 0) {
            dev_embed_row_to_l2(
                x_res,
                reinterpret_cast<const unsigned char*>(m->emb_w),
                m->emb_type, cur_tok, dim
            );
        }
        if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // Unified loop: l = 0..num_layers-1 runs stages 0..3; l == num_layers runs stage 4 (LM_Head)!
        for (int l = 0; l <= m->num_layers; ++l) {
            const int stage_beg = (l < m->num_layers) ? 0 : 4;
            const int stage_end = (l < m->num_layers) ? 4 : 5;
            const QwenLayerDesc& ld = layers[(l < m->num_layers) ? l : 0];

            #pragma unroll 1
            for (int stage = stage_beg; stage < stage_end; ++stage) {
                float* g_out = nullptr;
                const unsigned char* g_w = nullptr;
                int g_wtype = 12;
                const float* g_bias = nullptr;
                const float* g_res = nullptr;
                int g_N = 0, g_K = dim, g_sm_idx = bid, g_sm_cnt = num_sms;
                bool is_select = false;

                if (stage == 0) {
                    dev_rmsnorm_q8_1_to_smem(
                        s_xq, s_xds, s_scratch, x_res,
                        reinterpret_cast<const float*>(ld.attn_norm_w), dim, eps
                    );
                    if (ld.split_v == 0) {
                        g_out = ws_qkv;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_qkv_or_qk);
                        g_wtype = ld.qkv_type;
                        g_bias = reinterpret_cast<const float*>(ld.b_qkv_or_qk);
                        g_N = q_dim + 2 * kv_dim;
                    } else {
                        const int v_sms = max(1, (num_sms * 15) / 68);
                        const int qk_sms = num_sms - v_sms;
                        if (bid < qk_sms) {
                            g_out = ws_qkv;
                            g_w = reinterpret_cast<const unsigned char*>(ld.w_qkv_or_qk);
                            g_wtype = ld.qkv_type;
                            g_bias = reinterpret_cast<const float*>(ld.b_qkv_or_qk);
                            g_N = q_dim + kv_dim;
                            g_sm_idx = bid; g_sm_cnt = qk_sms;
                        } else {
                            g_out = ws_qkv + q_dim + kv_dim;
                            g_w = reinterpret_cast<const unsigned char*>(ld.w_v);
                            g_wtype = ld.v_type;
                            g_bias = reinterpret_cast<const float*>(ld.b_v);
                            g_N = kv_dim;
                            g_sm_idx = bid - qk_sms; g_sm_cnt = v_sms;
                        }
                    }
                } else if (stage == 1) {
                    dev_load_q8_1_to_smem(s_xq, s_xds, ws_xq, ws_xds, q_dim);
                    g_out = x_res2;
                    g_w = reinterpret_cast<const unsigned char*>(ld.w_o);
                    g_wtype = ld.o_type;
                    g_bias = reinterpret_cast<const float*>(ld.b_o);
                    g_res = x_res;
                    g_N = dim; g_K = q_dim;
                } else if (stage == 2) {
                    dev_rmsnorm_q8_1_to_smem(
                        s_xq, s_xds, s_scratch, x_res2,
                        reinterpret_cast<const float*>(ld.ffn_norm_w), dim, eps
                    );
                    if (ld.split_gate_up == 0) {
                        g_out = ws_gate_up;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_gate_up);
                        g_wtype = ld.gate_up_type;
                        g_N = 2 * hidden_dim;
                    } else {
                        const int half_sms = num_sms >> 1;
                        if (bid < half_sms) {
                            g_out = ws_gate_up;
                            g_w = reinterpret_cast<const unsigned char*>(ld.w_gate_up);
                            g_wtype = ld.gate_up_type;
                            g_N = hidden_dim;
                            g_sm_idx = bid; g_sm_cnt = half_sms;
                        } else {
                            g_out = ws_gate_up + hidden_dim;
                            g_w = reinterpret_cast<const unsigned char*>(ld.w_up_sep);
                            g_wtype = ld.up_sep_type;
                            g_N = hidden_dim;
                            g_sm_idx = bid - half_sms; g_sm_cnt = num_sms - half_sms;
                        }
                    }
                } else if (stage == 3) {
                    dev_load_q8_1_to_smem(s_xq, s_xds, ws_xq, ws_xds, hidden_dim);
                    g_out = x_res;
                    g_w = reinterpret_cast<const unsigned char*>(ld.w_down);
                    g_wtype = ld.down_type;
                    g_bias = reinterpret_cast<const float*>(ld.b_down);
                    g_res = x_res2;
                    g_N = dim; g_K = hidden_dim;
                } else {
                    // Stage 4: Final RMSNorm + Fused LM_Head Argmax/Gumbel Select!
                    dev_rmsnorm_q8_1_to_smem(
                        s_xq, s_xds, s_scratch, x_res,
                        reinterpret_cast<const float*>(m->output_norm_w), dim, eps
                    );
                    g_w = reinterpret_cast<const unsigned char*>(m->lm_head_w);
                    g_wtype = m->lm_head_type;
                    g_N = m->vocab_size;
                    g_K = dim;
                    is_select = true;
                }

                dev_gemv_or_select_smem(
                    g_out, ws_cta_val, ws_cta_idx, s_scratch, s_idx_red,
                    g_w, g_wtype, s_xq, s_xds, g_bias, g_res,
                    g_N, g_K, g_sm_idx, g_sm_cnt,
                    is_select, s_temp, s_seed, (uint32_t)step_idx
                );
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (stage == 0) {
                    half* kv_cache = reinterpret_cast<half*>(ld.kv_cache);
                    const int total_qkv_heads = n_heads + 2 * n_kv_heads;
                    for (int h_all = bid; h_all < total_qkv_heads; h_all += num_sms) {
                        if (h_all >= n_heads + n_kv_heads) {
                            if (tid < head_dim) {
                                const int kv_h = h_all - (n_heads + n_kv_heads);
                                const float v_val = ld_cg_f32(&ws_qkv[q_dim + kv_dim + kv_h * head_dim + tid]);
                                st_cg_f16(&kv_cache[((size_t)(n_kv_heads + kv_h) * max_ctx + pos) * head_dim + tid], __float2half(v_val));
                            }
                        } else {
                            const bool is_q = (h_all < n_heads);
                            const int h_idx = is_q ? h_all : (h_all - n_heads);
                            float val = 0.0f;
                            if (tid < head_dim) {
                                const int src_off = is_q ? (h_idx * head_dim + tid) : (q_dim + h_idx * head_dim + tid);
                                val = ld_cg_f32(&ws_qkv[src_off]);
                            }
                            if (m->has_qk_norm) {
                                float ss = (tid < head_dim) ? (val * val) : 0.0f;
                                ss = warp_reduce_sum(ss);
                                if (lane == 0) s_scratch[wid] = ss;
                                __syncthreads();
                                if (wid == 0) {
                                    float wss = (lane < (head_dim >> 5)) ? s_scratch[lane] : 0.0f;
                                    wss = warp_reduce_sum(wss);
                                    if (lane == 0) s_scratch[0] = rsqrtf(wss / (float)head_dim + eps);
                                }
                                __syncthreads();
                                if (tid < head_dim) {
                                    const float* nw = reinterpret_cast<const float*>(is_q ? ld.q_norm_w : ld.k_norm_w);
                                    val = val * s_scratch[0] * __ldg(&nw[tid]);
                                }
                            }
                            if (tid < head_dim) s_scratch[16 + tid] = val;
                            __syncthreads();
                            if (tid < head_dim) {
                                float out_val = val;
                                if (tid < rope_dim) {
                                    const float* freq_row = freqs_cis + (size_t)pos * rope_dim;
                                    if (tid < half_rope) {
                                        out_val = val * __ldg(&freq_row[tid]) - s_scratch[16 + tid + half_rope] * __ldg(&freq_row[tid + half_rope]);
                                    } else {
                                        const int j = tid - half_rope;
                                        out_val = s_scratch[16 + j] * __ldg(&freq_row[tid]) + val * __ldg(&freq_row[j]);
                                    }
                                }
                                if (is_q) st_cg_f32(&ws_q_roped[h_idx * head_dim + tid], out_val);
                                else st_cg_f16(&kv_cache[((size_t)h_idx * max_ctx + pos) * head_dim + tid], __float2half(out_val));
                            }
                            __syncthreads();
                        }
                    }
                    if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    const int seq_len = pos + 1;
                    if (head_dim == 128) {
                        for (int h = bid; h < n_heads; h += num_sms) {
                            const int kv_h = h / gqa_ratio;
                            const int base_q = h * 128 + (lane << 2);
                            float q0 = ld_cg_f32(&ws_q_roped[base_q + 0]) * attn_scale;
                            float q1 = ld_cg_f32(&ws_q_roped[base_q + 1]) * attn_scale;
                            float q2 = ld_cg_f32(&ws_q_roped[base_q + 2]) * attn_scale;
                            float q3 = ld_cg_f32(&ws_q_roped[base_q + 3]) * attn_scale;

                            const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 128;
                            const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 128;
                            float m_i = -1e30f, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

                            for (int p_idx = wid; p_idx < seq_len; p_idx += MK_WARPS) {
                                const uint2 kv_k = *reinterpret_cast<const uint2*>(k_base + (size_t)p_idx * 128 + (lane << 2));
                                const half2 kh0 = *reinterpret_cast<const half2*>(&kv_k.x);
                                const half2 kh1 = *reinterpret_cast<const half2*>(&kv_k.y);
                                float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y)
                                         + q2 * __half2float(kh1.x) + q3 * __half2float(kh1.y);
                                #pragma unroll
                                for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);

                                const float m_new = fmaxf(m_i, qk);
                                const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
                                l_i = l_i * alpha + p_w; m_i = m_new;

                                const uint2 kv_v = *reinterpret_cast<const uint2*>(v_base + (size_t)p_idx * 128 + (lane << 2));
                                const half2 vh0 = *reinterpret_cast<const half2*>(&kv_v.x);
                                const half2 vh1 = *reinterpret_cast<const half2*>(&kv_v.y);
                                a0 = a0 * alpha + p_w * __half2float(vh0.x);
                                a1 = a1 * alpha + p_w * __half2float(vh0.y);
                                a2 = a2 * alpha + p_w * __half2float(vh1.x);
                                a3 = a3 * alpha + p_w * __half2float(vh1.y);
                            }

                            if (lane == 0) { s_scratch[wid] = m_i; s_scratch[MK_WARPS + wid] = l_i; }
                            const int acc_off = 2 * MK_WARPS + wid * 128 + (lane << 2);
                            s_scratch[acc_off + 0] = a0; s_scratch[acc_off + 1] = a1;
                            s_scratch[acc_off + 2] = a2; s_scratch[acc_off + 3] = a3;
                            __syncthreads();

                            if (wid == 0) {
                                float m_max = -1e30f;
                                #pragma unroll 1
                                for (int w = 0; w < MK_WARPS; ++w) m_max = fmaxf(m_max, s_scratch[w]);
                                float l_sum = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                                #pragma unroll 1
                                for (int w = 0; w < MK_WARPS; ++w) {
                                    const float wl = s_scratch[MK_WARPS + w];
                                    const float ws = (wl > 0.0f) ? expf(s_scratch[w] - m_max) : 0.0f;
                                    l_sum += wl * ws;
                                    const int roff = 2 * MK_WARPS + w * 128 + (lane << 2);
                                    o0 += s_scratch[roff + 0] * ws; o1 += s_scratch[roff + 1] * ws;
                                    o2 += s_scratch[roff + 2] * ws; o3 += s_scratch[roff + 3] * ws;
                                }
                                const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                                o0 *= inv_l; o1 *= inv_l; o2 *= inv_l; o3 *= inv_l;

                                float sum_x = o0 + o1 + o2 + o3;
                                float amax = fmaxf(fmaxf(fabsf(o0), fabsf(o1)), fmaxf(fabsf(o2), fabsf(o3)));
                                #pragma unroll
                                for (int off = 4; off > 0; off >>= 1) {
                                    sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                                }
                                const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                                const int packed = (max(-127, min(127, __float2int_rn(o0 * id))) & 0xff)
                                                 | ((max(-127, min(127, __float2int_rn(o1 * id))) & 0xff) << 8)
                                                 | ((max(-127, min(127, __float2int_rn(o2 * id))) & 0xff) << 16)
                                                 | ((max(-127, min(127, __float2int_rn(o3 * id))) & 0xff) << 24);
                                int* ws_xq_i32 = reinterpret_cast<int*>(ws_xq);
                                asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_xq_i32[(h << 5) + lane]), "r"(packed) : "memory");
                                if ((lane & 7) == 0) st_cg_f32x2(&ws_xds[(h << 2) + (lane >> 3)], make_float2(d, sum_x));
                            }
                            __syncthreads();
                        }
                    } else {
                        for (int h = bid; h < n_heads; h += num_sms) {
                            const int kv_h = h / gqa_ratio;
                            const int base_q = h * 64 + (lane << 1);
                            float q0 = ld_cg_f32(&ws_q_roped[base_q + 0]) * attn_scale;
                            float q1 = ld_cg_f32(&ws_q_roped[base_q + 1]) * attn_scale;
                            const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 64;
                            const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 64;
                            float m_i = -1e30f, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f;
                            for (int p_idx = wid; p_idx < seq_len; p_idx += MK_WARPS) {
                                const half2 kh0 = *reinterpret_cast<const half2*>(k_base + (size_t)p_idx * 64 + (lane << 1));
                                float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y);
                                #pragma unroll
                                for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
                                const float m_new = fmaxf(m_i, qk);
                                const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
                                l_i = l_i * alpha + p_w; m_i = m_new;
                                const half2 vh0 = *reinterpret_cast<const half2*>(v_base + (size_t)p_idx * 64 + (lane << 1));
                                a0 = a0 * alpha + p_w * __half2float(vh0.x);
                                a1 = a1 * alpha + p_w * __half2float(vh0.y);
                            }
                            if (lane == 0) { s_scratch[wid] = m_i; s_scratch[MK_WARPS + wid] = l_i; }
                            s_scratch[2 * MK_WARPS + wid * 64 + (lane << 1) + 0] = a0;
                            s_scratch[2 * MK_WARPS + wid * 64 + (lane << 1) + 1] = a1;
                            __syncthreads();
                            if (wid == 0) {
                                float m_max = -1e30f;
                                #pragma unroll 1
                                for (int w = 0; w < MK_WARPS; ++w) m_max = fmaxf(m_max, s_scratch[w]);
                                float l_sum = 0.0f, o0 = 0.0f, o1 = 0.0f;
                                #pragma unroll 1
                                for (int w = 0; w < MK_WARPS; ++w) {
                                    const float wl = s_scratch[MK_WARPS + w];
                                    const float ws = (wl > 0.0f) ? expf(s_scratch[w] - m_max) : 0.0f;
                                    l_sum += wl * ws;
                                    o0 += s_scratch[2 * MK_WARPS + w * 64 + (lane << 1) + 0] * ws;
                                    o1 += s_scratch[2 * MK_WARPS + w * 64 + (lane << 1) + 1] * ws;
                                }
                                const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                                o0 *= inv_l; o1 *= inv_l;
                                float sum_x = o0 + o1, amax = fmaxf(fabsf(o0), fabsf(o1));
                                #pragma unroll
                                for (int off = 8; off > 0; off >>= 1) {
                                    sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                                }
                                const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                                st_cg_s8(&ws_xq[h * 64 + (lane << 1) + 0], max(-127, min(127, __float2int_rn(o0 * id))));
                                st_cg_s8(&ws_xq[h * 64 + (lane << 1) + 1], max(-127, min(127, __float2int_rn(o1 * id))));
                                if ((lane & 15) == 0) st_cg_f32x2(&ws_xds[(h << 1) + (lane >> 4)], make_float2(d, sum_x));
                            }
                            __syncthreads();
                        }
                    }
                    if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                } else if (stage == 2) {
                    const int n_ffn_blks = hidden_dim >> 5;
                    const int total_warps = num_sms * MK_WARPS;
                    for (int blk = bid * MK_WARPS + wid; blk < n_ffn_blks; blk += total_warps) {
                        const int col = (blk << 5) + lane;
                        const float g = ld_cg_f32(&ws_gate_up[col]);
                        const float u = ld_cg_f32(&ws_gate_up[hidden_dim + col]);
                        const float v = (g / (1.0f + expf(-g))) * u;
                        float sum_x = v, amax = fabsf(v);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                        }
                        const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        st_cg_s8(&ws_xq[col], max(-127, min(127, __float2int_rn(v * id))));
                        if (lane == 0) st_cg_f32x2(&ws_xds[blk], make_float2(d, sum_x));
                    }
                    if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                }
            }
        }

        // All-SM Final Winner Reduction across `num_sms <= 128` CTAs
        if (wid < 4) {
            const int idx_cta = tid;
            float bv = (idx_cta < num_sms) ? ld_cg_f32(&ws_cta_val[idx_cta]) : -1e38f;
            int bi = 0x7fffffff;
            if (idx_cta < num_sms) {
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(bi) : "l"(&ws_cta_idx[idx_cta]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                s_scratch[wid] = bv;
                s_idx_red[wid] = bi;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best_v = s_scratch[0];
            int next_tok = s_idx_red[0];
            for (int w = 1; w < 4; ++w) {
                if (s_scratch[w] > best_v || (s_scratch[w] == best_v && s_idx_red[w] < next_tok)) {
                    best_v = s_scratch[w];
                    next_tok = s_idx_red[w];
                }
            }
            s_cur_tok = next_tok;
            s_cur_pos = pos + 1;

            const bool is_eos = (next_tok == m->stop_id0) || (next_tok == m->stop_id1)
                             || (next_tok == m->stop_id2) || (next_tok == m->stop_id3);
            const bool ctx_full = (pos + 1 >= max_ctx);

            if (bid == 0) {
                st_host_s32(&ctrl->tokens_out[step_idx], next_tok);
                const int cancel_req = ld_host_s32(&ctrl->cancel_flag);
                if (is_eos) st_host_s32(&ctrl->stop_reason, 1);
                else if (cancel_req != 0) st_host_s32(&ctrl->stop_reason, 2);
                else if (ctx_full) st_host_s32(&ctrl->stop_reason, 3);
                else if (step_idx + 1 >= s_max_steps) st_host_s32(&ctrl->stop_reason, 0);

                asm volatile("membar.gl;" ::: "memory");
                st_host_s32(&ctrl->steps_completed, step_idx + 1);
                const int stop_sig = (is_eos || cancel_req != 0 || ctx_full) ? 1 : 0;
                asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_cta_idx[127]), "r"(stop_sig) : "memory");
            }
        }

        if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
        if (tid == 0) {
            int stop_sig = 0;
            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(stop_sig) : "l"(&ws_cta_idx[127]));
            s_done = stop_sig;
        }
        __syncthreads();
        if (s_done != 0) break;
    }

    if (bid == 0 && tid == 0) {
        asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
    }
}

// =============================================================================
// 2. PERSISTENT 68-SM BM=4 SPECULATIVE VERIFICATION MEGAKERNEL (256 threads/CTA)
// =============================================================================

__device__ __forceinline__ void dev_gemm_or_select_bm4(
    float* out,
    float* cta_best_val,
    int* cta_best_idx,
    float* s_scratch,
    int* s_idx_red,
    const unsigned char* w_raw,
    int w_type,
    const int* xq_i32,
    const float2* xds,
    bool x_in_smem,
    const float* bias,
    const float* residual,
    int N,
    int K,
    int sm_idx,
    int sm_count,
    bool mode_select
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int k_i32 = K >> 2;
    const int k_ds = K >> 5;

    float best_v[4] = {-1e38f, -1e38f, -1e38f, -1e38f};
    int best_i[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};

    if (w_type == 12) {
        const int n_sb = K >> 8;
        const int hw_in_cta = (wid << 1) | (lane >> 4);
        const int total_hw = sm_count * BM4_HW;
        const int lane16 = lane & 15;
        const int sb_lane = lane16 >> 3, in_sb = lane16 & 7, chunk = in_sb >> 1, half16 = in_sb & 1;

        for (int row = sm_idx * BM4_HW + hw_in_cta; row < N; row += total_hw) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 144);
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll 2
            for (int sb = sb_lane; sb < n_sb; sb += 2) {
                const unsigned char* blk = w_row + sb * 144;
                const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
                const half2 dm = *reinterpret_cast<const half2*>(&hdr.x);
                const float d = __half2float(dm.x), dmin = __half2float(dm.y);
                const unsigned int s0 = hdr.y, s1 = hdr.z, s2 = hdr.w;
                unsigned int sc_lo, sc_hi, mn_lo, mn_hi;
                if (chunk < 2) {
                    const int sh0 = (chunk << 1) << 3, sh1 = ((chunk << 1) + 1) << 3;
                    sc_lo = (s0 >> sh0) & 0x3fu; sc_hi = (s0 >> sh1) & 0x3fu;
                    mn_lo = (s1 >> sh0) & 0x3fu; mn_hi = (s1 >> sh1) & 0x3fu;
                } else {
                    const int sh0 = ((chunk - 2) << 1) << 3, sh1 = (((chunk - 2) << 1) + 1) << 3;
                    sc_lo = ((s2 >> sh0) & 0x0fu) | (((s0 >> (sh0 + 6)) & 0x03u) << 4);
                    sc_hi = ((s2 >> sh1) & 0x0fu) | (((s0 >> (sh1 + 6)) & 0x03u) << 4);
                    mn_lo = ((s2 >> (sh0 + 4)) & 0x0fu) | (((s1 >> (sh0 + 6)) & 0x03u) << 4);
                    mn_hi = ((s2 >> (sh1 + 4)) & 0x0fu) | (((s1 >> (sh1 + 6)) & 0x03u) << 4);
                }
                const uint4 qv = ld_cs_u32x4(blk + 16 + (in_sb << 4));
                const int ql0 = (int)(qv.x & 0x0f0f0f0fu), ql1 = (int)(qv.y & 0x0f0f0f0fu);
                const int ql2 = (int)(qv.z & 0x0f0f0f0fu), ql3 = (int)(qv.w & 0x0f0f0f0fu);
                const int qh0 = (int)((qv.x >> 4) & 0x0f0f0f0fu), qh1 = (int)((qv.y >> 4) & 0x0f0f0f0fu);
                const int qh2 = (int)((qv.z >> 4) & 0x0f0f0f0fu), qh3 = (int)((qv.w >> 4) & 0x0f0f0f0fu);
                const float d_sc_lo = d * (float)sc_lo, d_sc_hi = d * (float)sc_hi;
                const float dm_mn_lo = dmin * (float)mn_lo, dm_mn_hi = dmin * (float)mn_hi;
                const int x_off_int = (sb << 6) + (chunk << 4) + (half16 << 2);
                const int sub0 = (sb << 3) + (chunk << 1);

                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int* xptr = xq_i32 + mi * k_i32 + x_off_int;
                    const float2* dsptr = xds + mi * k_ds + sub0;
                    const int4 x_lo = x_in_smem ? *reinterpret_cast<const int4*>(xptr) : ld_cg_s32x4(xptr);
                    const int4 x_hi = x_in_smem ? *reinterpret_cast<const int4*>(xptr + 8) : ld_cg_s32x4(xptr + 8);
                    const float2 ds0 = x_in_smem ? dsptr[0] : ld_cg_f32x2(&dsptr[0]);
                    const float2 ds1 = x_in_smem ? dsptr[1] : ld_cg_f32x2(&dsptr[1]);
                    int sumi_lo = __dp4a(ql0, x_lo.x, 0); sumi_lo = __dp4a(ql1, x_lo.y, sumi_lo);
                    sumi_lo = __dp4a(ql2, x_lo.z, sumi_lo); sumi_lo = __dp4a(ql3, x_lo.w, sumi_lo);
                    int sumi_hi = __dp4a(qh0, x_hi.x, 0); sumi_hi = __dp4a(qh1, x_hi.y, sumi_hi);
                    sumi_hi = __dp4a(qh2, x_hi.z, sumi_hi); sumi_hi = __dp4a(qh3, x_hi.w, sumi_hi);
                    acc[mi] += d_sc_lo * ds0.x * (float)sumi_lo + d_sc_hi * ds1.x * (float)sumi_hi;
                    if (half16 == 0) acc[mi] -= dm_mn_lo * ds0.y + dm_mn_hi * ds1.y;
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) acc[mi] = half_warp_reduce_sum(acc[mi]);
            if (lane16 == 0) {
                if (mode_select) {
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        if (acc[mi] > best_v[mi] || (acc[mi] == best_v[mi] && row < best_i[mi])) {
                            best_v[mi] = acc[mi]; best_i[mi] = row;
                        }
                    }
                } else {
                    const float bv = (bias != nullptr) ? __ldg(&bias[row]) : 0.0f;
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        float val = acc[mi] + bv;
                        if (residual != nullptr) val += ld_cg_f32(&residual[(size_t)mi * N + row]);
                        st_cg_f32(&out[(size_t)mi * N + row], val);
                    }
                }
            }
        }
    } else if (w_type == 14) {
        const int n_sb = K >> 8;
        const int total_w = sm_count * BM4_WARPS;
        const int sb_lane = lane >> 4, tid16 = lane & 15;
        const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
        const int ql_off = (h << 6) + (q << 5) + (half8 << 4);
        const int qh_off = 128 + (h << 5) + (half8 << 4);
        const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;

        for (int row = sm_idx * BM4_WARPS + wid; row < N; row += total_w) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 210);
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll 2
            for (int sb = sb_lane; sb < n_sb; sb += 2) {
                const unsigned char* blk = w_row + sb * 210;
                const float d_sc = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208))) *
                                   (float)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[tid16]);
                const unsigned char* ql_ptr = blk + ql_off;
                const unsigned char* qh_ptr = blk + qh_off;
                const unsigned int ql0 = load_u32_align2(ql_ptr + 0), ql1 = load_u32_align2(ql_ptr + 4);
                const unsigned int ql2 = load_u32_align2(ql_ptr + 8), ql3 = load_u32_align2(ql_ptr + 12);
                const unsigned int qh0 = load_u32_align2(qh_ptr + 0), qh1 = load_u32_align2(qh_ptr + 4);
                const unsigned int qh2 = load_u32_align2(qh_ptr + 8), qh3 = load_u32_align2(qh_ptr + 12);
                const unsigned int u6_0 = ((ql0 >> ql_sh) & 0x0f0f0f0fu) | (((qh0 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_1 = ((ql1 >> ql_sh) & 0x0f0f0f0fu) | (((qh1 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_2 = ((ql2 >> ql_sh) & 0x0f0f0f0fu) | (((qh2 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int u6_3 = ((ql3 >> ql_sh) & 0x0f0f0f0fu) | (((qh3 >> qh_sh) & 0x03030303u) << 4);
                const unsigned int m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
                const unsigned int m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
                const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
                const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
                const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
                const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));
                const int x_base_int = (sb << 6) + (tid16 << 2);
                const int ds_idx = (sb << 3) + (tid16 >> 1);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int* xptr = xq_i32 + mi * k_i32 + x_base_int;
                    const int4 xv = x_in_smem ? *reinterpret_cast<const int4*>(xptr) : ld_cg_s32x4(xptr);
                    const float xd = x_in_smem ? xds[mi * k_ds + ds_idx].x : ld_cg_f32x2(&xds[mi * k_ds + ds_idx]).x;
                    int sumi = __dp4a(s6_0, xv.x, 0); sumi = __dp4a(s6_1, xv.y, sumi);
                    sumi = __dp4a(s6_2, xv.z, sumi); sumi = __dp4a(s6_3, xv.w, sumi);
                    acc[mi] += (d_sc * xd) * (float)sumi;
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) acc[mi] = warp_reduce_sum(acc[mi]);
            if (lane == 0) {
                if (mode_select) {
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        if (acc[mi] > best_v[mi] || (acc[mi] == best_v[mi] && row < best_i[mi])) {
                            best_v[mi] = acc[mi]; best_i[mi] = row;
                        }
                    }
                } else {
                    const float bv = (bias != nullptr) ? __ldg(&bias[row]) : 0.0f;
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        float val = acc[mi] + bv;
                        if (residual != nullptr) val += ld_cg_f32(&residual[(size_t)mi * N + row]);
                        st_cg_f32(&out[(size_t)mi * N + row], val);
                    }
                }
            }
        }
    } else {
        const int n_blk = K >> 5;
        const int total_w = sm_count * BM4_WARPS;
        const int blk_lane = lane >> 1, half16 = lane & 1;

        for (int row = sm_idx * BM4_WARPS + wid; row < N; row += total_w) {
            const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_blk * 34);
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int b = blk_lane; b < n_blk; b += 16) {
                const unsigned char* blk = w_row + b * 34;
                const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                const unsigned char* qs = blk + 2 + (half16 << 4);
                const int w0 = (int)load_u32_align2(qs + 0), w1 = (int)load_u32_align2(qs + 4);
                const int w2 = (int)load_u32_align2(qs + 8), w3 = (int)load_u32_align2(qs + 12);
                const int x_off_int = (b << 3) + (half16 << 2);
                #pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int* xptr = xq_i32 + mi * k_i32 + x_off_int;
                    const int4 xv = x_in_smem ? *reinterpret_cast<const int4*>(xptr) : ld_cg_s32x4(xptr);
                    const float xd = x_in_smem ? xds[mi * k_ds + b].x : ld_cg_f32x2(&xds[mi * k_ds + b]).x;
                    int sumi = __dp4a(w0, xv.x, 0); sumi = __dp4a(w1, xv.y, sumi);
                    sumi = __dp4a(w2, xv.z, sumi); sumi = __dp4a(w3, xv.w, sumi);
                    acc[mi] += (d * xd) * (float)sumi;
                }
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) acc[mi] = warp_reduce_sum(acc[mi]);
            if (lane == 0) {
                if (mode_select) {
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        if (acc[mi] > best_v[mi] || (acc[mi] == best_v[mi] && row < best_i[mi])) {
                            best_v[mi] = acc[mi]; best_i[mi] = row;
                        }
                    }
                } else {
                    const float bv = (bias != nullptr) ? __ldg(&bias[row]) : 0.0f;
                    #pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        float val = acc[mi] + bv;
                        if (residual != nullptr) val += ld_cg_f32(&residual[(size_t)mi * N + row]);
                        st_cg_f32(&out[(size_t)mi * N + row], val);
                    }
                }
            }
        }
    }

    if (mode_select) {
        __syncthreads();
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            float bv = best_v[mi];
            int bi = best_i[mi];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                s_scratch[mi * BM4_WARPS + wid] = bv;
                s_idx_red[mi * BM4_WARPS + wid] = bi;
            }
        }
        __syncthreads();
        if (wid < 4) {
            const int mi = wid;
            float bv = (lane < BM4_WARPS) ? s_scratch[mi * BM4_WARPS + lane] : -1e38f;
            int bi = (lane < BM4_WARPS) ? s_idx_red[mi * BM4_WARPS + lane] : 0x7fffffff;
            #pragma unroll
            for (int off = (BM4_WARPS >> 1); off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                st_cg_f32(&cta_best_val[mi * 128 + blockIdx.x], bv);
                asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&cta_best_idx[mi * 128 + blockIdx.x]), "r"(bi) : "memory");
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(576, 1)
void qwen_spec_verify_bm4_megakernel(
    const QwenModelDesc* __restrict__ m,
    volatile QwenDecodeCtrl* __restrict__ ctrl
) {
    __shared__ __align__(16) int    s_xq_bm4[4 * (BM4_SMEM_K / 4)];
    __shared__ __align__(16) float2 s_xds_bm4[4 * (BM4_SMEM_K / 32)];
    __shared__ __align__(16) float  s_scratch[BM4_WARPS * 128 + 64];
    __shared__ __align__(16) int    s_idx_red[64];
    __shared__ int   s_abort;
    __shared__ int   s_toks[4];
    __shared__ int   s_start_pos;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(m->ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    if (tid == 0) {
        s_abort = 0;
        s_start_pos = ctrl->start_pos;
        #pragma unroll
        for (int i = 0; i < 4; ++i) s_toks[i] = ctrl->spec_tokens_in[i];
    }
    __syncthreads();

    const QwenLayerDesc* layers = reinterpret_cast<const QwenLayerDesc*>(m->layers_ptr);
    float* x_res = reinterpret_cast<float*>(m->ws_x_res);
    float* x_res2 = reinterpret_cast<float*>(m->ws_x_res2);
    signed char* ws_xq = reinterpret_cast<signed char*>(m->ws_xq);
    int* ws_xq_i32 = reinterpret_cast<int*>(m->ws_xq);
    float2* ws_xds = reinterpret_cast<float2*>(m->ws_xds);
    float* ws_qkv = reinterpret_cast<float*>(m->ws_qkv);
    float* ws_q_roped = reinterpret_cast<float*>(m->ws_q_roped);
    float* ws_gate_up = reinterpret_cast<float*>(m->ws_gate_up);
    float* ws_cta_val = reinterpret_cast<float*>(m->ws_cta_val);
    int* ws_cta_idx = reinterpret_cast<int*>(m->ws_cta_idx);
    const float* freqs_cis = reinterpret_cast<const float*>(m->freqs_cis);

    const int dim = m->dim;
    const int hidden_dim = m->hidden_dim;
    const int n_heads = m->n_heads;
    const int n_kv_heads = m->n_kv_heads;
    const int head_dim = m->head_dim;
    const int rope_dim = m->rope_dim;
    const int half_rope = rope_dim >> 1;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int gqa_ratio = n_heads / n_kv_heads;
    const int max_ctx = m->max_ctx;
    const float eps = m->norm_eps;
    const float attn_scale = m->inv_sqrt_head_dim;
    const int start_pos = s_start_pos;

    if (bid < 4) {
        dev_embed_row_to_l2(
            x_res + (size_t)bid * dim,
            reinterpret_cast<const unsigned char*>(m->emb_w),
            m->emb_type, s_toks[bid], dim
        );
    }
    if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    for (int l = 0; l <= m->num_layers; ++l) {
        const int stage_beg = (l < m->num_layers) ? 0 : 4;
        const int stage_end = (l < m->num_layers) ? 4 : 5;
        const QwenLayerDesc& ld = layers[(l < m->num_layers) ? l : 0];

        #pragma unroll 1
        for (int stage = stage_beg; stage < stage_end; ++stage) {
            if (stage == 0 || stage == 2 || stage == 4) {
                if (bid < 4) {
                    const float* xr = ((stage == 2) ? x_res2 : x_res) + (size_t)bid * dim;
                    const float* nw = reinterpret_cast<const float*>(
                        (stage == 0) ? ld.attn_norm_w : ((stage == 2) ? ld.ffn_norm_w : m->output_norm_w)
                    );
                    const int n_blks = dim >> 5;
                    float ss = 0.0f;
                    for (int blk = wid; blk < n_blks; blk += BM4_WARPS) {
                        const float v = ld_cg_f32(&xr[(blk << 5) + lane]);
                        ss += v * v;
                    }
                    ss = warp_reduce_sum(ss);
                    if (lane == 0) s_scratch[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float wss = (lane < BM4_WARPS) ? s_scratch[lane] : 0.0f;
                        wss = warp_reduce_sum(wss);
                        if (lane == 0) s_scratch[0] = rsqrtf(wss / (float)dim + eps);
                    }
                    __syncthreads();
                    const float scale = s_scratch[0];
                    for (int blk = wid; blk < n_blks; blk += BM4_WARPS) {
                        const int idx = (blk << 5) + lane;
                        const float v = ld_cg_f32(&xr[idx]) * scale * __ldg(&nw[idx]);
                        float sum_x = v, amax = fabsf(v);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                        }
                        const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        st_cg_s8(&ws_xq[(size_t)bid * dim + idx], max(-127, min(127, __float2int_rn(v * id))));
                        if (lane == 0) st_cg_f32x2(&ws_xds[(size_t)bid * n_blks + blk], make_float2(d, sum_x));
                    }
                    __syncthreads();
                }
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            }

            const int qkv_stride = q_dim + 2 * kv_dim;
            float* g_out = nullptr;
            const unsigned char* g_w = nullptr;
            int g_wtype = 12;
            const float* g_bias = nullptr;
            const float* g_res = nullptr;
            int g_N = 0, g_K = dim, g_sm_idx = bid, g_sm_cnt = num_sms;
            bool is_select = false;

            if (stage == 0) {
                if (ld.split_v == 0) {
                    g_out = ws_qkv;
                    g_w = reinterpret_cast<const unsigned char*>(ld.w_qkv_or_qk);
                    g_wtype = ld.qkv_type;
                    g_bias = reinterpret_cast<const float*>(ld.b_qkv_or_qk);
                    g_N = qkv_stride;
                } else {
                    const int v_sms = max(1, (num_sms * 15) / 68);
                    const int qk_sms = num_sms - v_sms;
                    if (bid < qk_sms) {
                        g_out = ws_qkv;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_qkv_or_qk);
                        g_wtype = ld.qkv_type;
                        g_bias = reinterpret_cast<const float*>(ld.b_qkv_or_qk);
                        g_N = q_dim + kv_dim;
                        g_sm_idx = bid; g_sm_cnt = qk_sms;
                    } else {
                        g_out = ws_gate_up;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_v);
                        g_wtype = ld.v_type;
                        g_bias = reinterpret_cast<const float*>(ld.b_v);
                        g_N = kv_dim;
                        g_sm_idx = bid - qk_sms; g_sm_cnt = v_sms;
                    }
                }
            } else if (stage == 1) {
                g_out = x_res2;
                g_w = reinterpret_cast<const unsigned char*>(ld.w_o);
                g_wtype = ld.o_type;
                g_bias = reinterpret_cast<const float*>(ld.b_o);
                g_res = x_res;
                g_N = dim; g_K = q_dim;
            } else if (stage == 2) {
                if (ld.split_gate_up == 0) {
                    g_out = ws_gate_up;
                    g_w = reinterpret_cast<const unsigned char*>(ld.w_gate_up);
                    g_wtype = ld.gate_up_type;
                    g_N = 2 * hidden_dim;
                } else {
                    const int half_sms = num_sms >> 1;
                    if (bid < half_sms) {
                        g_out = ws_gate_up;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_gate_up);
                        g_wtype = ld.gate_up_type;
                        g_N = hidden_dim;
                        g_sm_idx = bid; g_sm_cnt = half_sms;
                    } else {
                        g_out = ws_gate_up + (size_t)4 * hidden_dim;
                        g_w = reinterpret_cast<const unsigned char*>(ld.w_up_sep);
                        g_wtype = ld.up_sep_type;
                        g_N = hidden_dim;
                        g_sm_idx = bid - half_sms; g_sm_cnt = num_sms - half_sms;
                    }
                }
            } else if (stage == 3) {
                g_out = x_res;
                g_w = reinterpret_cast<const unsigned char*>(ld.w_down);
                g_wtype = ld.down_type;
                g_bias = reinterpret_cast<const float*>(ld.b_down);
                g_res = x_res2;
                g_N = dim; g_K = hidden_dim;
            } else {
                g_w = reinterpret_cast<const unsigned char*>(m->lm_head_w);
                g_wtype = m->lm_head_type;
                g_N = m->vocab_size;
                g_K = dim;
                is_select = true;
            }

            const bool use_smem_x = (g_K <= BM4_SMEM_K);
            if (use_smem_x) {
                const int n_i4 = (4 * g_K) >> 4;
                int4* s_i4 = reinterpret_cast<int4*>(s_xq_bm4);
                for (int i = tid; i < n_i4; i += BM4_THREADS) {
                    s_i4[i] = ld_cg_s32x4(ws_xq_i32 + (i << 2));
                }
                const int n_ds = (4 * g_K) >> 5;
                for (int b = tid; b < n_ds; b += BM4_THREADS) {
                    s_xds_bm4[b] = ld_cg_f32x2(&ws_xds[b]);
                }
                __syncthreads();
            }

            dev_gemm_or_select_bm4(
                g_out, ws_cta_val, ws_cta_idx, s_scratch, s_idx_red,
                g_w, g_wtype,
                use_smem_x ? s_xq_bm4 : ws_xq_i32,
                use_smem_x ? s_xds_bm4 : ws_xds,
                use_smem_x,
                g_bias, g_res,
                g_N, g_K, g_sm_idx, g_sm_cnt, is_select
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            if (stage == 0) {
                half* kv_cache = reinterpret_cast<half*>(ld.kv_cache);
                const int total_qkv_heads = n_heads + 2 * n_kv_heads;
                const int total_items = 4 * total_qkv_heads;
                const int row_qk_stride = (ld.split_v == 0) ? qkv_stride : (q_dim + kv_dim);

                for (int item = bid; item < total_items; item += num_sms) {
                    const int t = item / total_qkv_heads;
                    const int h_all = item % total_qkv_heads;
                    const int pos = start_pos + t;
                    if (h_all >= n_heads + n_kv_heads) {
                        if (tid < head_dim) {
                            const int kv_h = h_all - (n_heads + n_kv_heads);
                            const float v_val = (ld.split_v == 0)
                                ? ld_cg_f32(&ws_qkv[(size_t)t * qkv_stride + q_dim + kv_dim + kv_h * head_dim + tid])
                                : ld_cg_f32(&ws_gate_up[(size_t)t * kv_dim + kv_h * head_dim + tid]);
                            st_cg_f16(&kv_cache[((size_t)(n_kv_heads + kv_h) * max_ctx + pos) * head_dim + tid], __float2half(v_val));
                        }
                    } else {
                        const bool is_q = (h_all < n_heads);
                        const int h_idx = is_q ? h_all : (h_all - n_heads);
                        float val = 0.0f;
                        if (tid < head_dim) {
                            const size_t base = (size_t)t * row_qk_stride + (is_q ? (h_idx * head_dim) : (q_dim + h_idx * head_dim));
                            val = ld_cg_f32(&ws_qkv[base + tid]);
                        }
                        if (m->has_qk_norm) {
                            float ss = (tid < head_dim) ? (val * val) : 0.0f;
                            ss = warp_reduce_sum(ss);
                            if (lane == 0) s_scratch[wid] = ss;
                            __syncthreads();
                            if (wid == 0) {
                                float wss = (lane < (head_dim >> 5)) ? s_scratch[lane] : 0.0f;
                                wss = warp_reduce_sum(wss);
                                if (lane == 0) s_scratch[0] = rsqrtf(wss / (float)head_dim + eps);
                            }
                            __syncthreads();
                            if (tid < head_dim) {
                                const float* nw = reinterpret_cast<const float*>(is_q ? ld.q_norm_w : ld.k_norm_w);
                                val = val * s_scratch[0] * __ldg(&nw[tid]);
                            }
                        }
                        if (tid < head_dim) s_scratch[16 + tid] = val;
                        __syncthreads();
                        if (tid < head_dim) {
                            float out_val = val;
                            if (tid < rope_dim) {
                                const float* freq_row = freqs_cis + (size_t)pos * rope_dim;
                                if (tid < half_rope) {
                                    out_val = val * __ldg(&freq_row[tid]) - s_scratch[16 + tid + half_rope] * __ldg(&freq_row[tid + half_rope]);
                                } else {
                                    const int j = tid - half_rope;
                                    out_val = s_scratch[16 + j] * __ldg(&freq_row[tid]) + val * __ldg(&freq_row[j]);
                                }
                            }
                            if (is_q) st_cg_f32(&ws_q_roped[((size_t)t * n_heads + h_idx) * head_dim + tid], out_val);
                            else st_cg_f16(&kv_cache[((size_t)h_idx * max_ctx + pos) * head_dim + tid], __float2half(out_val));
                        }
                        __syncthreads();
                    }
                }
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                const int total_qh = 4 * n_heads;
                if (head_dim == 128) {
                    for (int qh = bid; qh < total_qh; qh += num_sms) {
                        const int t = qh / n_heads;
                        const int h = qh % n_heads;
                        const int kv_h = h / gqa_ratio;
                        const int seq_len = start_pos + t + 1;
                        const size_t base_q = ((size_t)t * n_heads + h) * 128 + (lane << 2);
                        float q0 = ld_cg_f32(&ws_q_roped[base_q + 0]) * attn_scale;
                        float q1 = ld_cg_f32(&ws_q_roped[base_q + 1]) * attn_scale;
                        float q2 = ld_cg_f32(&ws_q_roped[base_q + 2]) * attn_scale;
                        float q3 = ld_cg_f32(&ws_q_roped[base_q + 3]) * attn_scale;

                        const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 128;
                        const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 128;
                        float m_i = -1e30f, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

                        for (int p_idx = wid; p_idx < seq_len; p_idx += BM4_WARPS) {
                            const uint2 kv_k = *reinterpret_cast<const uint2*>(k_base + (size_t)p_idx * 128 + (lane << 2));
                            const half2 kh0 = *reinterpret_cast<const half2*>(&kv_k.x);
                            const half2 kh1 = *reinterpret_cast<const half2*>(&kv_k.y);
                            float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y)
                                     + q2 * __half2float(kh1.x) + q3 * __half2float(kh1.y);
                            #pragma unroll
                            for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
                            const float m_new = fmaxf(m_i, qk);
                            const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
                            l_i = l_i * alpha + p_w; m_i = m_new;
                            const uint2 kv_v = *reinterpret_cast<const uint2*>(v_base + (size_t)p_idx * 128 + (lane << 2));
                            const half2 vh0 = *reinterpret_cast<const half2*>(&kv_v.x);
                            const half2 vh1 = *reinterpret_cast<const half2*>(&kv_v.y);
                            a0 = a0 * alpha + p_w * __half2float(vh0.x);
                            a1 = a1 * alpha + p_w * __half2float(vh0.y);
                            a2 = a2 * alpha + p_w * __half2float(vh1.x);
                            a3 = a3 * alpha + p_w * __half2float(vh1.y);
                        }
                        if (lane == 0) { s_scratch[wid] = m_i; s_scratch[BM4_WARPS + wid] = l_i; }
                        const int acc_off = 2 * BM4_WARPS + wid * 128 + (lane << 2);
                        s_scratch[acc_off + 0] = a0; s_scratch[acc_off + 1] = a1;
                        s_scratch[acc_off + 2] = a2; s_scratch[acc_off + 3] = a3;
                        __syncthreads();
                        if (wid == 0) {
                            float m_max = -1e30f;
                            #pragma unroll 1
                            for (int w = 0; w < BM4_WARPS; ++w) m_max = fmaxf(m_max, s_scratch[w]);
                            float l_sum = 0.0f, o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                            #pragma unroll 1
                            for (int w = 0; w < BM4_WARPS; ++w) {
                                const float wl = s_scratch[BM4_WARPS + w];
                                const float ws = (wl > 0.0f) ? expf(s_scratch[w] - m_max) : 0.0f;
                                l_sum += wl * ws;
                                const int roff = 2 * BM4_WARPS + w * 128 + (lane << 2);
                                o0 += s_scratch[roff + 0] * ws; o1 += s_scratch[roff + 1] * ws;
                                o2 += s_scratch[roff + 2] * ws; o3 += s_scratch[roff + 3] * ws;
                            }
                            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                            o0 *= inv_l; o1 *= inv_l; o2 *= inv_l; o3 *= inv_l;
                            float sum_x = o0 + o1 + o2 + o3;
                            float amax = fmaxf(fmaxf(fabsf(o0), fabsf(o1)), fmaxf(fabsf(o2), fabsf(o3)));
                            #pragma unroll
                            for (int off = 4; off > 0; off >>= 1) {
                                sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                            }
                            const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                            const int packed = (max(-127, min(127, __float2int_rn(o0 * id))) & 0xff)
                                             | ((max(-127, min(127, __float2int_rn(o1 * id))) & 0xff) << 8)
                                             | ((max(-127, min(127, __float2int_rn(o2 * id))) & 0xff) << 16)
                                             | ((max(-127, min(127, __float2int_rn(o3 * id))) & 0xff) << 24);
                            asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_xq_i32[((size_t)t * n_heads + h) * 32 + lane]), "r"(packed) : "memory");
                            if ((lane & 7) == 0) st_cg_f32x2(&ws_xds[((size_t)t * n_heads + h) * 4 + (lane >> 3)], make_float2(d, sum_x));
                        }
                        __syncthreads();
                    }
                } else {
                    for (int qh = bid; qh < total_qh; qh += num_sms) {
                        const int t = qh / n_heads;
                        const int h = qh % n_heads;
                        const int kv_h = h / gqa_ratio;
                        const int seq_len = start_pos + t + 1;
                        const size_t base_q = ((size_t)t * n_heads + h) * 64 + (lane << 1);
                        float q0 = ld_cg_f32(&ws_q_roped[base_q + 0]) * attn_scale;
                        float q1 = ld_cg_f32(&ws_q_roped[base_q + 1]) * attn_scale;
                        const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 64;
                        const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 64;
                        float m_i = -1e30f, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f;
                        for (int p_idx = wid; p_idx < seq_len; p_idx += BM4_WARPS) {
                            const half2 kh0 = *reinterpret_cast<const half2*>(k_base + (size_t)p_idx * 64 + (lane << 1));
                            float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y);
                            #pragma unroll
                            for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
                            const float m_new = fmaxf(m_i, qk);
                            const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
                            l_i = l_i * alpha + p_w; m_i = m_new;
                            const half2 vh0 = *reinterpret_cast<const half2*>(v_base + (size_t)p_idx * 64 + (lane << 1));
                            a0 = a0 * alpha + p_w * __half2float(vh0.x);
                            a1 = a1 * alpha + p_w * __half2float(vh0.y);
                        }
                        if (lane == 0) { s_scratch[wid] = m_i; s_scratch[BM4_WARPS + wid] = l_i; }
                        s_scratch[2 * BM4_WARPS + wid * 64 + (lane << 1) + 0] = a0;
                        s_scratch[2 * BM4_WARPS + wid * 64 + (lane << 1) + 1] = a1;
                        __syncthreads();
                        if (wid == 0) {
                            float m_max = -1e30f;
                            #pragma unroll 1
                            for (int w = 0; w < BM4_WARPS; ++w) m_max = fmaxf(m_max, s_scratch[w]);
                            float l_sum = 0.0f, o0 = 0.0f, o1 = 0.0f;
                            #pragma unroll 1
                            for (int w = 0; w < BM4_WARPS; ++w) {
                                const float wl = s_scratch[BM4_WARPS + w];
                                const float ws = (wl > 0.0f) ? expf(s_scratch[w] - m_max) : 0.0f;
                                l_sum += wl * ws;
                                o0 += s_scratch[2 * BM4_WARPS + w * 64 + (lane << 1) + 0] * ws;
                                o1 += s_scratch[2 * BM4_WARPS + w * 64 + (lane << 1) + 1] * ws;
                            }
                            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
                            o0 *= inv_l; o1 *= inv_l;
                            float sum_x = o0 + o1, amax = fmaxf(fabsf(o0), fabsf(o1));
                            #pragma unroll
                            for (int off = 8; off > 0; off >>= 1) {
                                sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                            }
                            const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                            st_cg_s8(&ws_xq[((size_t)t * n_heads + h) * 64 + (lane << 1) + 0], max(-127, min(127, __float2int_rn(o0 * id))));
                            st_cg_s8(&ws_xq[((size_t)t * n_heads + h) * 64 + (lane << 1) + 1], max(-127, min(127, __float2int_rn(o1 * id))));
                            if ((lane & 15) == 0) st_cg_f32x2(&ws_xds[((size_t)t * n_heads + h) * 2 + (lane >> 4)], make_float2(d, sum_x));
                        }
                        __syncthreads();
                    }
                }
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            } else if (stage == 2) {
                const int n_ffn_blks = hidden_dim >> 5;
                const int total_ffn_blks = 4 * n_ffn_blks;
                const int total_warps = num_sms * BM4_WARPS;
                for (int tb = bid * BM4_WARPS + wid; tb < total_ffn_blks; tb += total_warps) {
                    const int mi = tb / n_ffn_blks;
                    const int blk = tb % n_ffn_blks;
                    const int col = (blk << 5) + lane;
                    const float g = (ld.split_gate_up == 0)
                        ? ld_cg_f32(&ws_gate_up[(size_t)mi * (2 * hidden_dim) + col])
                        : ld_cg_f32(&ws_gate_up[(size_t)mi * hidden_dim + col]);
                    const float u = (ld.split_gate_up == 0)
                        ? ld_cg_f32(&ws_gate_up[(size_t)mi * (2 * hidden_dim) + hidden_dim + col])
                        : ld_cg_f32(&ws_gate_up[(size_t)(4 + mi) * hidden_dim + col]);
                    const float v = (g / (1.0f + expf(-g))) * u;
                    float sum_x = v, amax = fabsf(v);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                    }
                    const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    st_cg_s8(&ws_xq[(size_t)mi * hidden_dim + col], max(-127, min(127, __float2int_rn(v * id))));
                    if (lane == 0) st_cg_f32x2(&ws_xds[(size_t)mi * n_ffn_blks + blk], make_float2(d, sum_x));
                }
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            }
        }
    }

    if (bid == 0) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            float bv = (tid < num_sms) ? ld_cg_f32(&ws_cta_val[mi * 128 + tid]) : -1e38f;
            int bi = 0x7fffffff;
            if (tid < num_sms) {
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(bi) : "l"(&ws_cta_idx[mi * 128 + tid]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0 && wid < 4) {
                s_scratch[mi * 4 + wid] = bv;
                s_idx_red[mi * 4 + wid] = bi;
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                float fv = s_scratch[mi * 4];
                int fi = s_idx_red[mi * 4];
                for (int w = 1; w < 4; ++w) {
                    if (s_scratch[mi * 4 + w] > fv || (s_scratch[mi * 4 + w] == fv && s_idx_red[mi * 4 + w] < fi)) {
                        fv = s_scratch[mi * 4 + w];
                        fi = s_idx_red[mi * 4 + w];
                    }
                }
                st_host_s32(&ctrl->spec_tokens_out[mi], fi);
            }
            asm volatile("membar.gl;" ::: "memory");
            st_host_s32(&ctrl->steps_completed, 4);
            asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        }
    }
}

// =============================================================================
// 3. PARAMETERIZED CHUNKED PREFILL KERNELS (Supports Any T, M, N, K Dynamically!)
// =============================================================================

extern "C" __global__ __launch_bounds__(256)
void qwen_prefill_embed(
    float* __restrict__ out,
    const unsigned char* __restrict__ emb_w,
    const int* __restrict__ tok_ids,
    int dim,
    int emb_type
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int tok = tok_ids[t];
    float* dst = out + (size_t)t * dim;

    if (emb_type == 14) {
        const int n_sb = dim >> 8;
        const unsigned char* row = emb_w + (size_t)tok * (n_sb * 210);
        for (int col = tid; col < dim; col += 256) {
            const int sb = col >> 8, idx = col & 255;
            const unsigned char* blk = row + sb * 210;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208)));
            const int sc = (int)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[idx >> 4]);
            const int h = idx >> 7, rem = idx & 127, p = rem >> 6, q = (rem >> 5) & 1, k = rem & 31;
            const unsigned char ql_byte = __ldg(&blk[(h << 6) + (q << 5) + k]);
            const unsigned char qh_byte = __ldg(&blk[128 + (h << 5) + k]);
            const int xl = (ql_byte >> (p << 2)) & 0x0f;
            const int xh = ((qh_byte >> (((p << 1) + q) << 1)) & 0x03) << 4;
            dst[col] = d * (float)sc * (float)((xl | xh) - 32);
        }
    } else if (emb_type == 12) {
        const int n_sb = dim >> 8;
        const unsigned char* row = emb_w + (size_t)tok * (n_sb * 144);
        for (int col = tid; col < dim; col += 256) {
            const int sb = col >> 8, idx = col & 255;
            const unsigned char* blk = row + sb * 144;
            const half2 dm = __ldg(reinterpret_cast<const half2*>(blk));
            const float d = __half2float(dm.x), dmin = __half2float(dm.y);
            const unsigned char* s = blk + 4;
            const int sub = idx >> 5, k = idx & 31;
            unsigned int sc, mn;
            if (sub < 4) { sc = __ldg(&s[sub]) & 63u; mn = __ldg(&s[sub + 4]) & 63u; }
            else {
                sc = (__ldg(&s[sub + 4]) & 0x0fu) | ((__ldg(&s[sub - 4]) >> 6) << 4);
                mn = (__ldg(&s[sub + 4]) >> 4)    | ((__ldg(&s[sub])     >> 6) << 4);
            }
            const unsigned char qbyte = __ldg(&blk[16 + ((sub >> 1) << 5) + k]);
            const unsigned int q = (sub & 1) ? (qbyte >> 4) : (qbyte & 0x0fu);
            dst[col] = d * (float)sc * (float)q - dmin * (float)mn;
        }
    } else if (emb_type == 8) {
        const int n_blk = dim >> 5;
        const unsigned char* row = emb_w + (size_t)tok * (n_blk * 34);
        for (int col = tid; col < dim; col += 256) {
            const int b = col >> 5, k = col & 31;
            const unsigned char* blk = row + b * 34;
            dst[col] = __half2float(__ldg(reinterpret_cast<const half*>(blk))) *
                       (float)__ldg(&reinterpret_cast<const signed char*>(blk + 2)[k]);
        }
    } else if (emb_type == 1) {
        const half* row = reinterpret_cast<const half*>(emb_w) + (size_t)tok * dim;
        for (int col = tid; col < dim; col += 256) dst[col] = __half2float(__ldg(&row[col]));
    } else {
        const float* row = reinterpret_cast<const float*>(emb_w) + (size_t)tok * dim;
        for (int col = tid; col < dim; col += 256) dst[col] = __ldg(&row[col]);
    }
}

extern "C" __global__ __launch_bounds__(256)
void qwen_prefill_rmsnorm_q8_1(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in,
    const float* __restrict__ norm_w,
    int K,
    float eps
) {
    __shared__ float s_red[8];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int n_blks = K >> 5;
    const float* xr = x_in + (size_t)row * K;

    float ss = 0.0f;
    for (int blk = wid; blk < n_blks; blk += 8) {
        float v = xr[(blk << 5) + lane];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float wss = (lane < 8) ? s_red[lane] : 0.0f;
        wss = warp_reduce_sum(wss);
        if (lane == 0) s_red[0] = rsqrtf(wss / (float)K + eps);
    }
    __syncthreads();
    const float scale = s_red[0];

    for (int blk = wid; blk < n_blks; blk += 8) {
        const int idx = (blk << 5) + lane;
        const float v = xr[idx] * scale * __ldg(&norm_w[idx]);
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        q_out[(size_t)row * K + idx] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
        if (lane == 0) ds_out[(size_t)row * n_blks + blk] = make_float2(d, sum_x);
    }
}

extern "C" __global__ __launch_bounds__(128)
void qwen_prefill_quant_q8_1(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in,
    int total_blks
) {
    const int warp_id = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (warp_id >= total_blks) return;
    const int lane = threadIdx.x & 31;
    const float v = x_in[(warp_id << 5) + lane];
    float sum_x = v, amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    const float d = amax * (1.0f / 127.0f);
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q_out[(warp_id << 5) + lane] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
    if (lane == 0) ds_out[warp_id] = make_float2(d, sum_x);
}

extern "C" __global__ __launch_bounds__(128)
void qwen_prefill_swiglu_q8_1(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ gate_up,
    int M,
    int H
) {
    const int n_blks = H >> 5;
    const int total_blks = M * n_blks;
    const int warp_id = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (warp_id >= total_blks) return;
    const int lane = threadIdx.x & 31;
    const int row = warp_id / n_blks;
    const int blk = warp_id % n_blks;
    const int col = (blk << 5) + lane;

    const float g = gate_up[(size_t)row * (2 * H) + col];
    const float u = gate_up[(size_t)row * (2 * H) + H + col];
    const float v = (g / (1.0f + expf(-g))) * u;

    float sum_x = v, amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    const float d = amax * (1.0f / 127.0f);
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q_out[((size_t)warp_id << 5) + lane] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
    if (lane == 0) ds_out[warp_id] = make_float2(d, sum_x);
}

extern "C" __global__ __launch_bounds__(128)
void qwen_prefill_gemm(
    float* __restrict__ out,
    const unsigned char* __restrict__ w_raw,
    const int* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    int M,
    int N,
    int K,
    int w_type
) {
    const int row = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int m0 = blockIdx.y << 3;
    const int m_cnt = min(8, M - m0);
    const int lane = threadIdx.x & 31;
    const int k_i32 = K >> 2;
    const int k_ds = K >> 5;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (w_type == 12) {
        const int n_sb = K >> 8;
        const int sb_lane = lane >> 3, in_sb = lane & 7, chunk = in_sb >> 1, half16 = in_sb & 1;
        const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 144);

        for (int sb = sb_lane; sb < n_sb; sb += 4) {
            const unsigned char* blk = w_row + sb * 144;
            const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
            const half2 dm = *reinterpret_cast<const half2*>(&hdr.x);
            const float d = __half2float(dm.x), dmin = __half2float(dm.y);
            const unsigned int s0 = hdr.y, s1 = hdr.z, s2 = hdr.w;
            unsigned int sc_lo, sc_hi, mn_lo, mn_hi;
            if (chunk < 2) {
                const int sh0 = (chunk << 1) << 3, sh1 = ((chunk << 1) + 1) << 3;
                sc_lo = (s0 >> sh0) & 0x3fu; sc_hi = (s0 >> sh1) & 0x3fu;
                mn_lo = (s1 >> sh0) & 0x3fu; mn_hi = (s1 >> sh1) & 0x3fu;
            } else {
                const int sh0 = ((chunk - 2) << 1) << 3, sh1 = (((chunk - 2) << 1) + 1) << 3;
                sc_lo = ((s2 >> sh0) & 0x0fu) | (((s0 >> (sh0 + 6)) & 0x03u) << 4);
                sc_hi = ((s2 >> sh1) & 0x0fu) | (((s0 >> (sh1 + 6)) & 0x03u) << 4);
                mn_lo = ((s2 >> (sh0 + 4)) & 0x0fu) | (((s1 >> (sh0 + 6)) & 0x03u) << 4);
                mn_hi = ((s2 >> (sh1 + 4)) & 0x0fu) | (((s1 >> (sh1 + 6)) & 0x03u) << 4);
            }
            const uint4 qv = ld_cs_u32x4(blk + 16 + (in_sb << 4));
            const int ql0 = (int)(qv.x & 0x0f0f0f0fu), ql1 = (int)(qv.y & 0x0f0f0f0fu);
            const int ql2 = (int)(qv.z & 0x0f0f0f0fu), ql3 = (int)(qv.w & 0x0f0f0f0fu);
            const int qh0 = (int)((qv.x >> 4) & 0x0f0f0f0fu), qh1 = (int)((qv.y >> 4) & 0x0f0f0f0fu);
            const int qh2 = (int)((qv.z >> 4) & 0x0f0f0f0fu), qh3 = (int)((qv.w >> 4) & 0x0f0f0f0fu);
            const float d_sc_lo = d * (float)sc_lo, d_sc_hi = d * (float)sc_hi;
            const float dm_mn_lo = dmin * (float)mn_lo, dm_mn_hi = dmin * (float)mn_hi;
            const int x_off_int = (sb << 6) + (chunk << 4) + (half16 << 2);
            const int sub0 = (sb << 3) + (chunk << 1);

            #pragma unroll
            for (int mi = 0; mi < 8; ++mi) {
                if (mi < m_cnt) {
                    const int* xr = x_q8 + (size_t)(m0 + mi) * k_i32 + x_off_int;
                    const float2* dsr = x_ds + (size_t)(m0 + mi) * k_ds + sub0;
                    const int4 x_lo = *reinterpret_cast<const int4*>(xr);
                    const int4 x_hi = *reinterpret_cast<const int4*>(xr + 8);
                    int sumi_lo = __dp4a(ql0, x_lo.x, 0); sumi_lo = __dp4a(ql1, x_lo.y, sumi_lo);
                    sumi_lo = __dp4a(ql2, x_lo.z, sumi_lo); sumi_lo = __dp4a(ql3, x_lo.w, sumi_lo);
                    int sumi_hi = __dp4a(qh0, x_hi.x, 0); sumi_hi = __dp4a(qh1, x_hi.y, sumi_hi);
                    sumi_hi = __dp4a(qh2, x_hi.z, sumi_hi); sumi_hi = __dp4a(qh3, x_hi.w, sumi_hi);
                    const float2 ds0 = dsr[0], ds1 = dsr[1];
                    acc[mi] += d_sc_lo * ds0.x * (float)sumi_lo + d_sc_hi * ds1.x * (float)sumi_hi;
                    if (half16 == 0) acc[mi] -= dm_mn_lo * ds0.y + dm_mn_hi * ds1.y;
                }
            }
        }
    } else if (w_type == 14) {
        const int n_sb = K >> 8;
        const int sb_lane = lane >> 4, tid16 = lane & 15;
        const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
        const int ql_off = (h << 6) + (q << 5) + (half8 << 4), qh_off = 128 + (h << 5) + (half8 << 4);
        const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;
        const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_sb * 210);

        for (int sb = sb_lane; sb < n_sb; sb += 2) {
            const unsigned char* blk = w_row + sb * 210;
            const float d_sc = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208))) *
                               (float)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[tid16]);
            const unsigned char* ql_ptr = blk + ql_off;
            const unsigned char* qh_ptr = blk + qh_off;
            const unsigned int ql0 = load_u32_align2(ql_ptr + 0), ql1 = load_u32_align2(ql_ptr + 4);
            const unsigned int ql2 = load_u32_align2(ql_ptr + 8), ql3 = load_u32_align2(ql_ptr + 12);
            const unsigned int qh0 = load_u32_align2(qh_ptr + 0), qh1 = load_u32_align2(qh_ptr + 4);
            const unsigned int qh2 = load_u32_align2(qh_ptr + 8), qh3 = load_u32_align2(qh_ptr + 12);
            const unsigned int u6_0 = ((ql0 >> ql_sh) & 0x0f0f0f0fu) | (((qh0 >> qh_sh) & 0x03030303u) << 4);
            const unsigned int u6_1 = ((ql1 >> ql_sh) & 0x0f0f0f0fu) | (((qh1 >> qh_sh) & 0x03030303u) << 4);
            const unsigned int u6_2 = ((ql2 >> ql_sh) & 0x0f0f0f0fu) | (((qh2 >> qh_sh) & 0x03030303u) << 4);
            const unsigned int u6_3 = ((ql3 >> ql_sh) & 0x0f0f0f0fu) | (((qh3 >> qh_sh) & 0x03030303u) << 4);
            const unsigned int m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
            const unsigned int m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
            const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
            const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
            const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
            const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));
            const int x_base_int = (sb << 6) + (tid16 << 2);
            const int ds_idx = (sb << 3) + (tid16 >> 1);

            #pragma unroll
            for (int mi = 0; mi < 8; ++mi) {
                if (mi < m_cnt) {
                    const int4 xv = *reinterpret_cast<const int4*>(x_q8 + (size_t)(m0 + mi) * k_i32 + x_base_int);
                    int sumi = __dp4a(s6_0, xv.x, 0); sumi = __dp4a(s6_1, xv.y, sumi);
                    sumi = __dp4a(s6_2, xv.z, sumi); sumi = __dp4a(s6_3, xv.w, sumi);
                    acc[mi] += (d_sc * x_ds[(size_t)(m0 + mi) * k_ds + ds_idx].x) * (float)sumi;
                }
            }
        }
    } else {
        const int n_blk = K >> 5;
        const int blk_lane = lane >> 1, half16 = lane & 1;
        const unsigned char* w_row = w_raw + (size_t)row * (size_t)(n_blk * 34);
        for (int b = blk_lane; b < n_blk; b += 16) {
            const unsigned char* blk = w_row + b * 34;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
            const unsigned char* qs = blk + 2 + (half16 << 4);
            const int w0 = (int)load_u32_align2(qs + 0), w1 = (int)load_u32_align2(qs + 4);
            const int w2 = (int)load_u32_align2(qs + 8), w3 = (int)load_u32_align2(qs + 12);
            const int x_off_int = (b << 3) + (half16 << 2);
            #pragma unroll
            for (int mi = 0; mi < 8; ++mi) {
                if (mi < m_cnt) {
                    const int4 xv = *reinterpret_cast<const int4*>(x_q8 + (size_t)(m0 + mi) * k_i32 + x_off_int);
                    int sumi = __dp4a(w0, xv.x, 0); sumi = __dp4a(w1, xv.y, sumi);
                    sumi = __dp4a(w2, xv.z, sumi); sumi = __dp4a(w3, xv.w, sumi);
                    acc[mi] += (d * x_ds[(size_t)(m0 + mi) * k_ds + b].x) * (float)sumi;
                }
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < 8; ++mi) acc[mi] = warp_reduce_sum(acc[mi]);
    if (lane == 0) {
        const float bv = (bias != nullptr) ? __ldg(&bias[row]) : 0.0f;
        #pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            if (mi < m_cnt) {
                float v = acc[mi] + bv;
                if (residual != nullptr) v += residual[(size_t)(m0 + mi) * N + row];
                out[(size_t)(m0 + mi) * N + row] = v;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void qwen_prefill_qknorm_rope_kv(
    float* __restrict__ q_out,
    half* __restrict__ kv_cache,
    const float* __restrict__ qk_in,
    const float* __restrict__ v_sep,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    const float* __restrict__ freqs_cis,
    int T,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int max_ctx,
    float eps,
    int has_qk_norm,
    int split_v
) {
    __shared__ float s_ss[8];
    __shared__ float s_vec[256];

    const int h_all = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int qk_stride = split_v ? (q_dim + kv_dim) : (q_dim + 2 * kv_dim);
    const int pos = start_pos + t;

    if (h_all >= n_heads + n_kv_heads) {
        const int kv_h = h_all - (n_heads + n_kv_heads);
        const float v_val = split_v
            ? v_sep[(size_t)t * kv_dim + kv_h * head_dim + tid]
            : qk_in[(size_t)t * qk_stride + q_dim + kv_dim + kv_h * head_dim + tid];
        kv_cache[((size_t)(n_kv_heads + kv_h) * max_ctx + pos) * head_dim + tid] = __float2half(v_val);
        return;
    }

    const bool is_q = (h_all < n_heads);
    const int h_idx = is_q ? h_all : (h_all - n_heads);
    const float* row_in = qk_in + (size_t)t * qk_stride;
    float val = is_q ? row_in[h_idx * head_dim + tid] : row_in[q_dim + h_idx * head_dim + tid];

    if (has_qk_norm) {
        const int lane = tid & 31, wid = tid >> 5;
        float ss = warp_reduce_sum(val * val);
        if (lane == 0) s_ss[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            ss = (lane < (head_dim >> 5)) ? s_ss[lane] : 0.0f;
            ss = warp_reduce_sum(ss);
            if (lane == 0) s_ss[0] = rsqrtf(ss / (float)head_dim + eps);
        }
        __syncthreads();
        const float* nw = is_q ? q_norm_w : k_norm_w;
        val = val * s_ss[0] * __ldg(&nw[tid]);
    }

    s_vec[tid] = val;
    __syncthreads();

    float out_val = val;
    const int half_rope = rope_dim >> 1;
    if (tid < rope_dim) {
        const float* freq_row = freqs_cis + (size_t)pos * rope_dim;
        if (tid < half_rope) {
            out_val = val * __ldg(&freq_row[tid]) - s_vec[tid + half_rope] * __ldg(&freq_row[tid + half_rope]);
        } else {
            const int j = tid - half_rope;
            out_val = s_vec[j] * __ldg(&freq_row[tid]) + val * __ldg(&freq_row[j]);
        }
    }

    if (is_q) {
        q_out[((size_t)h_idx * T + t) * head_dim + tid] = out_val;
    } else {
        kv_cache[((size_t)h_idx * max_ctx + pos) * head_dim + tid] = __float2half(out_val);
    }
}

extern "C" __global__ __launch_bounds__(128)
void qwen_prefill_flash_gqa(
    float* __restrict__ out,
    const float* __restrict__ q_in,
    const half* __restrict__ kv_cache,
    int T,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_ctx,
    float scale
) {
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][256];

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int seq_len = start_pos + t + 1;

    if (head_dim == 128) {
        const float* q_ptr = q_in + ((size_t)h * T + t) * 128 + (lane << 2);
        float q0 = q_ptr[0] * scale, q1 = q_ptr[1] * scale, q2 = q_ptr[2] * scale, q3 = q_ptr[3] * scale;
        const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 128;
        const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 128;
        float m_i = -1e30f, l_i = 0.0f, a0 = 0, a1 = 0, a2 = 0, a3 = 0;

        for (int p_idx = wid; p_idx < seq_len; p_idx += 4) {
            const uint2 kv_k = *reinterpret_cast<const uint2*>(k_base + (size_t)p_idx * 128 + (lane << 2));
            const half2 kh0 = *reinterpret_cast<const half2*>(&kv_k.x), kh1 = *reinterpret_cast<const half2*>(&kv_k.y);
            float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y) + q2 * __half2float(kh1.x) + q3 * __half2float(kh1.y);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
            const float m_new = fmaxf(m_i, qk);
            const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
            l_i = l_i * alpha + p_w; m_i = m_new;
            const uint2 kv_v = *reinterpret_cast<const uint2*>(v_base + (size_t)p_idx * 128 + (lane << 2));
            const half2 vh0 = *reinterpret_cast<const half2*>(&kv_v.x), vh1 = *reinterpret_cast<const half2*>(&kv_v.y);
            a0 = a0 * alpha + p_w * __half2float(vh0.x); a1 = a1 * alpha + p_w * __half2float(vh0.y);
            a2 = a2 * alpha + p_w * __half2float(vh1.x); a3 = a3 * alpha + p_w * __half2float(vh1.y);
        }
        if (lane == 0) { s_m[wid] = m_i; s_l[wid] = l_i; }
        s_acc[wid][(lane << 2) + 0] = a0; s_acc[wid][(lane << 2) + 1] = a1;
        s_acc[wid][(lane << 2) + 2] = a2; s_acc[wid][(lane << 2) + 3] = a3;
        __syncthreads();
        if (wid == 0) {
            float m_max = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
            float l_sum = 0.0f, o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const float ws = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
                l_sum += s_l[w] * ws;
                o0 += s_acc[w][(lane << 2) + 0] * ws; o1 += s_acc[w][(lane << 2) + 1] * ws;
                o2 += s_acc[w][(lane << 2) + 2] * ws; o3 += s_acc[w][(lane << 2) + 3] * ws;
            }
            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            float* op = out + ((size_t)t * n_heads + h) * 128 + (lane << 2);
            op[0] = o0 * inv_l; op[1] = o1 * inv_l; op[2] = o2 * inv_l; op[3] = o3 * inv_l;
        }
    } else if (head_dim == 64) {
        const float* q_ptr = q_in + ((size_t)h * T + t) * 64 + (lane << 1);
        float q0 = q_ptr[0] * scale, q1 = q_ptr[1] * scale;
        const half* k_base = kv_cache + (size_t)kv_h * max_ctx * 64;
        const half* v_base = kv_cache + (size_t)(n_kv_heads + kv_h) * max_ctx * 64;
        float m_i = -1e30f, l_i = 0.0f, a0 = 0, a1 = 0;
        for (int p_idx = wid; p_idx < seq_len; p_idx += 4) {
            const half2 kh0 = *reinterpret_cast<const half2*>(k_base + (size_t)p_idx * 64 + (lane << 1));
            float qk = q0 * __half2float(kh0.x) + q1 * __half2float(kh0.y);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
            const float m_new = fmaxf(m_i, qk);
            const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
            l_i = l_i * alpha + p_w; m_i = m_new;
            const half2 vh0 = *reinterpret_cast<const half2*>(v_base + (size_t)p_idx * 64 + (lane << 1));
            a0 = a0 * alpha + p_w * __half2float(vh0.x); a1 = a1 * alpha + p_w * __half2float(vh0.y);
        }
        if (lane == 0) { s_m[wid] = m_i; s_l[wid] = l_i; }
        s_acc[wid][(lane << 1) + 0] = a0; s_acc[wid][(lane << 1) + 1] = a1;
        __syncthreads();
        if (wid == 0) {
            float m_max = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
            float l_sum = 0.0f, o0 = 0, o1 = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const float ws = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
                l_sum += s_l[w] * ws;
                o0 += s_acc[w][(lane << 1) + 0] * ws; o1 += s_acc[w][(lane << 1) + 1] * ws;
            }
            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            float* op = out + ((size_t)t * n_heads + h) * 64 + (lane << 1);
            op[0] = o0 * inv_l; op[1] = o1 * inv_l;
        }
    }
}
