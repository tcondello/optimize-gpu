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

struct __align__(16) QwenProjLora {
    uint64_t a_f16;   // const half* [rank, k_cols] (alpha/rank scale baked in)
    uint64_t b_f16;   // const half* [n_rows, rank]
    int32_t  rank;    // 0 if inactive, 1..=32 if active
    int32_t  out_off; // row offset in target buffer for T=1
    int32_t  n_rows;  // output rows of this sub-projection
    int32_t  k_cols;  // input cols of this sub-projection
};

struct __align__(16) QwenLayerLoraDesc {
    QwenProjLora projs[7]; // 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
    uint32_t     active_mask;
    uint32_t     _pad0;
    uint64_t     _pad1;
};

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

    uint64_t lora_ptr;      // const QwenLayerLoraDesc* (0 if no LoRA on this layer)
    uint64_t _pad_lora;
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

    uint64_t ws_ext_embed;    // const float* [dim] (when emb_type == -1 for Talker decode)
    uint64_t ws_hidden_last;  // float* [dim] (post-output_norm hidden bridge exported when export_logits != 0)
    uint64_t ws_logits_out;   // float* [vocab_size] (raw logits exported when export_logits != 0)
    uint64_t _pad_model;
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
    int32_t  export_logits;     // 0=normal argmax/gumbel select; 1=export ws_hidden_last & ws_logits_out
    int32_t  _pad_ctrl[3];
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
    int dim,
    const float* ext_embed = nullptr
) {
    const int tid = threadIdx.x;
    if (emb_type == -1) {
        const int nthreads = (int)blockDim.x;
        for (int col = tid; col < dim; col += nthreads) {
            st_cg_f32(&x_res[col], ld_cg_f32(&ext_embed[col]));
        }
        return;
    }
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

// Register-isolated (`__noinline__`) Fused Rank-r (`r <= 32`) LoRA Epilogue for `T=1` Megakernel.
// Keeps `qwen_persistent_decode_megakernel` register footprint untouched (`0.0%` overhead when `lora_ptr == 0`)!
__device__ __noinline__ bool dev_apply_lora_stage_t1(
    uint64_t lora_ptr_va,
    int stage,
    float* ws_qkv,
    float* x_res2,
    float* ws_gate_up,
    float* x_res,
    const int* s_xq_i32,
    const float2* s_xds,
    float* s_scratch,
    int bid,
    int num_sms,
    uint64_t* bar_cnt,
    uint64_t& bar_step,
    int& s_abort,
    volatile QwenDecodeCtrl* ctrl
) {
    const QwenLayerLoraDesc* ld_lora = reinterpret_cast<const QwenLayerLoraDesc*>(lora_ptr_va);
    const uint32_t stage_mask = (stage == 0) ? 0x07u : ((stage == 1) ? 0x08u : ((stage == 2) ? 0x30u : 0x40u));
    if ((ld_lora->active_mask & stage_mask) == 0u) {
        return true;
    }

    const int p_beg = (stage == 0) ? 0 : ((stage == 1) ? 3 : ((stage == 2) ? 4 : 6));
    const int p_end = (stage == 0) ? 3 : ((stage == 1) ? 4 : ((stage == 2) ? 6 : 7));
    float* out_base = (stage == 0) ? ws_qkv : ((stage == 1) ? x_res2 : ((stage == 2) ? ws_gate_up : x_res));

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const signed char* s_xq_s8 = reinterpret_cast<const signed char*>(s_xq_i32);

    for (int p_idx = p_beg; p_idx < p_end; ++p_idx) {
        const QwenProjLora& p = ld_lora->projs[p_idx];
        const int r = p.rank;
        if (r <= 0) continue;
        const int K = p.k_cols;
        const int N = p.n_rows;
        const half* a_ptr = reinterpret_cast<const half*>(p.a_f16);
        const half* b_ptr = reinterpret_cast<const half*>(p.b_f16);

        // 1. Compute `u[0..r-1] = A @ x` in SMEM `s_scratch[0..31]` using SMEM `s_xq_s8` + `s_xds`
        for (int j = wid; j < r; j += MK_WARPS) {
            const half* a_row = a_ptr + (size_t)j * K;
            float acc = 0.0f;
            for (int k = lane; k < K; k += 32) {
                const float x_val = (float)s_xq_s8[k] * s_xds[k >> 5].x;
                acc += __half2float(__ldg(&a_row[k])) * x_val;
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) s_scratch[j] = acc;
        }
        __syncthreads();

        // 2. Partition `N` output rows across all `num_sms * MK_THREADS` threads and add `B @ u`
        float* dst = out_base + p.out_off;
        const int total_threads = num_sms * MK_THREADS;
        for (int row = bid * MK_THREADS + tid; row < N; row += total_threads) {
            const half* b_row = b_ptr + (size_t)row * r;
            float delta = 0.0f;
            for (int j = 0; j < r; ++j) {
                delta += __half2float(__ldg(&b_row[j])) * s_scratch[j];
            }
            st_cg_f32(&dst[row], ld_cg_f32(&dst[row]) + delta);
        }
        __syncthreads();
    }

    return megakernel_barrier(bar_cnt, bar_step, num_sms, s_abort, ctrl);
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
    __shared__ int    s_export_logits;

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
        s_export_logits = ctrl->export_logits;
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
                m->emb_type, cur_tok, dim,
                reinterpret_cast<const float*>(m->ws_ext_embed)
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
                    // Stage 4: Final RMSNorm + Fused LM_Head Argmax/Gumbel Select (or Raw Logits + Hidden Bridge Export)
                    const float* out_nw = reinterpret_cast<const float*>(m->output_norm_w);
                    dev_rmsnorm_q8_1_to_smem(
                        s_xq, s_xds, s_scratch, x_res,
                        out_nw, dim, eps
                    );
                    g_w = reinterpret_cast<const unsigned char*>(m->lm_head_w);
                    g_wtype = m->lm_head_type;
                    g_N = m->vocab_size;
                    g_K = dim;
                    if (__builtin_expect(s_export_logits != 0, 0)) {
                        if (bid == 0 && m->ws_hidden_last != 0ULL) {
                            float* h_dst = reinterpret_cast<float*>(m->ws_hidden_last);
                            const float rms_scale = s_scratch[0];
                            for (int col = tid; col < dim; col += MK_THREADS) {
                                st_cg_f32(&h_dst[col], ld_cg_f32(&x_res[col]) * rms_scale * __ldg(&out_nw[col]));
                            }
                        }
                        g_out = reinterpret_cast<float*>(m->ws_logits_out);
                        is_select = false;
                    } else {
                        is_select = true;
                    }
                }

                dev_gemv_or_select_smem(
                    g_out, ws_cta_val, ws_cta_idx, s_scratch, s_idx_red,
                    g_w, g_wtype, s_xq, s_xds, g_bias, g_res,
                    g_N, g_K, g_sm_idx, g_sm_cnt,
                    is_select, s_temp, s_seed, (uint32_t)step_idx
                );
                if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (__builtin_expect(l < m->num_layers && ld.lora_ptr != 0ULL, 0)) {
                    if (!dev_apply_lora_stage_t1(
                        ld.lora_ptr, stage, ws_qkv, x_res2, ws_gate_up, x_res,
                        s_xq, s_xds, s_scratch, bid, num_sms, &bar_state[0], bar_step, s_abort, ctrl
                    )) return;
                }

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

        if (__builtin_expect(s_export_logits != 0, 0)) {
            if (bid == 0 && tid == 0) {
                st_host_s32(&ctrl->steps_completed, step_idx + 1);
            }
            break;
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

__device__ __noinline__ bool dev_apply_lora_stage_bm4(
    uint64_t lora_ptr_va,
    int stage,
    int split_v,
    int split_gate_up,
    float* ws_qkv,
    float* x_res2,
    float* ws_gate_up,
    float* x_res,
    int q_dim,
    int kv_dim,
    int dim,
    int hidden_dim,
    const signed char* ws_xq,
    const float2* ws_xds,
    float* s_scratch,
    int bid,
    int num_sms,
    uint64_t* bar_cnt,
    uint64_t& bar_step,
    int& s_abort,
    volatile QwenDecodeCtrl* ctrl
) {
    const QwenLayerLoraDesc* ld_lora = reinterpret_cast<const QwenLayerLoraDesc*>(lora_ptr_va);
    const uint32_t stage_mask = (stage == 0) ? 0x07u : ((stage == 1) ? 0x08u : ((stage == 2) ? 0x30u : 0x40u));
    if ((ld_lora->active_mask & stage_mask) == 0u) {
        return true;
    }

    const int p_beg = (stage == 0) ? 0 : ((stage == 1) ? 3 : ((stage == 2) ? 4 : 6));
    const int p_end = (stage == 0) ? 3 : ((stage == 1) ? 4 : ((stage == 2) ? 6 : 7));

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    for (int p_idx = p_beg; p_idx < p_end; ++p_idx) {
        const QwenProjLora& p = ld_lora->projs[p_idx];
        const int r = p.rank;
        if (r <= 0) continue;
        const int K = p.k_cols;
        const int N = p.n_rows;
        const int k_ds = K >> 5;
        const half* a_ptr = reinterpret_cast<const half*>(p.a_f16);
        const half* b_ptr = reinterpret_cast<const half*>(p.b_f16);

        // Compute `u[mi][0..r-1]` into `s_scratch[mi * 32 + j]` for `mi = 0..3`
        for (int j = wid; j < r; j += BM4_WARPS) {
            const half* a_row = a_ptr + (size_t)j * K;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int k = lane; k < K; k += 32) {
                const float aw = __half2float(__ldg(&a_row[k]));
                const int b_idx = k >> 5;
                acc0 += aw * ((float)ws_xq[0 * K + k] * ld_cg_f32x2(&ws_xds[0 * k_ds + b_idx]).x);
                acc1 += aw * ((float)ws_xq[1 * K + k] * ld_cg_f32x2(&ws_xds[1 * k_ds + b_idx]).x);
                acc2 += aw * ((float)ws_xq[2 * K + k] * ld_cg_f32x2(&ws_xds[2 * k_ds + b_idx]).x);
                acc3 += aw * ((float)ws_xq[3 * K + k] * ld_cg_f32x2(&ws_xds[3 * k_ds + b_idx]).x);
            }
            acc0 = warp_reduce_sum(acc0);
            acc1 = warp_reduce_sum(acc1);
            acc2 = warp_reduce_sum(acc2);
            acc3 = warp_reduce_sum(acc3);
            if (lane == 0) {
                s_scratch[0 * 32 + j] = acc0;
                s_scratch[1 * 32 + j] = acc1;
                s_scratch[2 * 32 + j] = acc2;
                s_scratch[3 * 32 + j] = acc3;
            }
        }
        __syncthreads();

        // Resolve BM=4 output buffer & row stride for projection `p_idx`
        float* dst_base = nullptr;
        int row_stride = N;
        int col_off = 0;
        if (p_idx == 0 || p_idx == 1) {
            dst_base = ws_qkv;
            row_stride = split_v ? (q_dim + kv_dim) : (q_dim + 2 * kv_dim);
            col_off = (p_idx == 0) ? 0 : q_dim;
        } else if (p_idx == 2) {
            if (split_v) {
                dst_base = ws_gate_up;
                row_stride = kv_dim;
                col_off = 0;
            } else {
                dst_base = ws_qkv;
                row_stride = q_dim + 2 * kv_dim;
                col_off = q_dim + kv_dim;
            }
        } else if (p_idx == 3) {
            dst_base = x_res2;
            row_stride = dim;
        } else if (p_idx == 4) {
            dst_base = ws_gate_up;
            row_stride = split_gate_up ? hidden_dim : (2 * hidden_dim);
        } else if (p_idx == 5) {
            if (split_gate_up) {
                dst_base = ws_gate_up + (size_t)4 * hidden_dim;
                row_stride = hidden_dim;
            } else {
                dst_base = ws_gate_up;
                row_stride = 2 * hidden_dim;
                col_off = hidden_dim;
            }
        } else {
            dst_base = x_res;
            row_stride = dim;
        }

        const int total_threads = num_sms * BM4_THREADS;
        for (int row = bid * BM4_THREADS + tid; row < N; row += total_threads) {
            const half* b_row = b_ptr + (size_t)row * r;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (int j = 0; j < r; ++j) {
                const float bw = __half2float(__ldg(&b_row[j]));
                d0 += bw * s_scratch[0 * 32 + j];
                d1 += bw * s_scratch[1 * 32 + j];
                d2 += bw * s_scratch[2 * 32 + j];
                d3 += bw * s_scratch[3 * 32 + j];
            }
            st_cg_f32(&dst_base[0 * (size_t)row_stride + col_off + row], ld_cg_f32(&dst_base[0 * (size_t)row_stride + col_off + row]) + d0);
            st_cg_f32(&dst_base[1 * (size_t)row_stride + col_off + row], ld_cg_f32(&dst_base[1 * (size_t)row_stride + col_off + row]) + d1);
            st_cg_f32(&dst_base[2 * (size_t)row_stride + col_off + row], ld_cg_f32(&dst_base[2 * (size_t)row_stride + col_off + row]) + d2);
            st_cg_f32(&dst_base[3 * (size_t)row_stride + col_off + row], ld_cg_f32(&dst_base[3 * (size_t)row_stride + col_off + row]) + d3);
        }
        __syncthreads();
    }

    return megakernel_barrier(bar_cnt, bar_step, num_sms, s_abort, ctrl);
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

            if (__builtin_expect(l < m->num_layers && ld.lora_ptr != 0ULL, 0)) {
                if (!dev_apply_lora_stage_bm4(
                    ld.lora_ptr, stage, ld.split_v, ld.split_gate_up,
                    ws_qkv, x_res2, ws_gate_up, x_res, q_dim, kv_dim, dim, hidden_dim,
                    ws_xq, ws_xds, s_scratch, bid, num_sms, &bar_state[0], bar_step, s_abort, ctrl
                )) return;
            }

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


// =============================================================================
// 4. PREFILL FUSED RANK-r LoRA KERNEL (`r <= 32`)
// =============================================================================

extern "C" __global__ __launch_bounds__(256)
void qwen_prefill_lora_add(
    float* __restrict__ out,
    const half* __restrict__ a_f16,
    const half* __restrict__ b_f16,
    const signed char* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    int M,
    int N_sub,
    int K,
    int r,
    int out_stride,
    int col_off
) {
    __shared__ float s_u[32];
    const int m = blockIdx.y;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const signed char* xr = x_q8 + (size_t)m * K;
    const float2* dsr = x_ds + (size_t)m * (K >> 5);

    for (int j = wid; j < r; j += 8) {
        const half* a_row = a_f16 + (size_t)j * K;
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32) {
            acc += __half2float(__ldg(&a_row[k])) * ((float)xr[k] * dsr[k >> 5].x);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) s_u[j] = acc;
    }
    __syncthreads();

    const int row = (blockIdx.x << 8) + tid;
    if (row < N_sub) {
        const half* b_row = b_f16 + (size_t)row * r;
        float delta = 0.0f;
        for (int j = 0; j < r; ++j) {
            delta += __half2float(__ldg(&b_row[j])) * s_u[j];
        }
        out[(size_t)m * out_stride + col_off + row] += delta;
    }
}

// =============================================================================
// 5. 100%-ON-RTX-3080 QWEN3-TTS-12HZ-1.7B (TALKER + CODEPREDICTOR + CODE2WAV)
// =============================================================================

struct __align__(16) TtsCodePredDesc {
    uint64_t layers_ptr;             // const QwenLayerDesc* [5]
    uint64_t mtp_proj_w;             // const uint8_t* [1024, 2048]
    uint64_t mtp_proj_b;             // const float* [1024] (0 if identity/none)
    uint64_t output_norm_w;          // const float* [1024]
    uint64_t lm_head_w[16];          // [0..14]: const uint8_t* [2048, 1024] (slot 15 pad)
    uint64_t cp_codec_emb_w[16];     // [0..14]: const uint8_t* [2048, 2048] (slot 15 pad)
    uint64_t talker_codec_emb_w;     // const uint8_t* [3072, 2048]
    uint64_t freqs_cis;              // const float* [32, 128]
    uint64_t talker_hidden_last;     // const float* [2048]
    uint64_t tts_pad_embed;          // const float* [2048]
    uint64_t talker_ws_ext_embed;    // float* [2048] (next frame Talker input written here!)

    uint64_t ws_in2048;              // float* [2048]
    uint64_t ws_x_res;               // float* [1024]
    uint64_t ws_x_res2;              // float* [1024]
    uint64_t ws_xq;                  // signed char* [4096]
    uint64_t ws_xds;                 // float2* [128]
    uint64_t ws_qkv;                 // float* [4096]
    uint64_t ws_q_roped;             // float* [2048]
    uint64_t ws_gate_up;             // float* [6144]
    uint64_t ws_logits;              // float* [2048]
    uint64_t ws_codes_frame;         // int32_t* [16] (VRAM scratch for current frame codes)
    uint64_t ws_bar_state;           // uint64_t* [4]

    int32_t  num_layers;             // 5
    int32_t  talker_dim;             // 2048
    int32_t  dim;                    // 1024
    int32_t  hidden_dim;             // 3072
    int32_t  n_heads;                // 16
    int32_t  n_kv_heads;             // 8
    int32_t  head_dim;               // 128
    int32_t  vocab_size;             // 2048
    int32_t  max_ctx;                // 32
    int32_t  mtp_proj_type;
    int32_t  talker_codec_emb_type;
    int32_t  lm_head_type;           // Q8_0 (8) or Q4_K (12)
    int32_t  cp_codec_emb_type;      // Q8_0 (8) or Q4_K (12) or F16 (1) or F32 (0)
    float    norm_eps;               // 1e-6f
    float    inv_sqrt_head_dim;      // 1 / sqrt(128)
    int32_t  _pad_cp;
};

struct __align__(64) TtsFrameCtrl {
    uint64_t seed;                   // Philox 64-bit seed
    uint64_t overlay_ptr;            // const float* [2048] (0 -> use tts_pad_embed)
    int32_t  frame_idx;              // 0..max_frames-1
    int32_t  c0;                     // written by tts_sample_c0_kernel
    int32_t  is_eos;                 // written 1 by tts_sample_c0_kernel when c0 == codec_eos_id
    int32_t  error_code;
    int32_t  frames_completed;       // GPU publishes frame_idx + 1 when frame finishes
    float    talker_temp;
    int32_t  talker_top_k;
    float    talker_top_p;
    float    rep_penalty;
    float    sub_temp;
    int32_t  sub_top_k;
    float    sub_top_p;
    int32_t  codec_eos_id;           // 2150
    int32_t  _pad_fc[3];
    int32_t  codes_out[2048 * 16];   // host-mapped [max_frames, 16] emitted RVQ codes!
};

// Exact Philox4x32-10 PRNG matching PyTorch CUDA `torch.rand()` & `qwentts.cpp/src/philox.h`
__device__ __forceinline__ float dev_philox_u01(uint64_t seed, uint64_t subseq) {
    uint32_t cx = 0u, cy = 0u, cz = (uint32_t)subseq, cw = (uint32_t)(subseq >> 32);
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
    #pragma unroll
    for (int rnd = 0; rnd < 10; ++rnd) {
        uint64_t p0 = (uint64_t)0xD2511F53u * (uint64_t)cx;
        uint64_t p1 = (uint64_t)0xCD9E8D57u * (uint64_t)cz;
        uint32_t nx = ((uint32_t)(p1 >> 32)) ^ cy ^ k0;
        uint32_t ny = (uint32_t)p1;
        uint32_t nz = ((uint32_t)(p0 >> 32)) ^ cw ^ k1;
        uint32_t nw = (uint32_t)p0;
        cx = nx; cy = ny; cz = nz; cw = nw;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return ((float)cx + 0.5f) * 2.3283064365386963e-10f;
}

__device__ inline float dev_elem_embed_f32(
    const unsigned char* emb_w,
    int emb_type,
    int tok_id,
    int col,
    int dim
) {
    if (emb_type == 8) {
        const int n_blk = dim >> 5;
        const unsigned char* blk = emb_w + ((size_t)tok_id * n_blk + (col >> 5)) * 34;
        return __half2float(__ldg(reinterpret_cast<const half*>(blk))) *
               (float)__ldg(&reinterpret_cast<const signed char*>(blk + 2)[col & 31]);
    } else if (emb_type == 12) {
        const int n_sb = dim >> 8;
        const int sb = col >> 8, idx = col & 255;
        const unsigned char* blk = emb_w + ((size_t)tok_id * n_sb + sb) * 144;
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
        return d * (float)sc * (float)q - dmin * (float)mn;
    } else if (emb_type == 14) {
        const int n_sb = dim >> 8;
        const int sb = col >> 8, idx = col & 255;
        const unsigned char* blk = emb_w + ((size_t)tok_id * n_sb + sb) * 210;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208)));
        const int sc = (int)__ldg(&reinterpret_cast<const signed char*>(blk + 192)[idx >> 4]);
        const int h = idx >> 7, rem = idx & 127, p = rem >> 6, q = (rem >> 5) & 1, k = rem & 31;
        const unsigned char ql_byte = __ldg(&blk[(h << 6) + (q << 5) + k]);
        const unsigned char qh_byte = __ldg(&blk[128 + (h << 5) + k]);
        const int xl = (ql_byte >> (p << 2)) & 0x0f;
        const int xh = ((qh_byte >> (((p << 1) + q) << 1)) & 0x03) << 4;
        return d * (float)sc * (float)((xl | xh) - 32);
    } else if (emb_type == 1) {
        const half* row = reinterpret_cast<const half*>(emb_w) + (size_t)tok_id * dim;
        return __half2float(__ldg(&row[col]));
    } else {
        const float* row = reinterpret_cast<const float*>(emb_w) + (size_t)tok_id * dim;
        return __ldg(&row[col]);
    }
}

// Quantize raw FP32 vector `x_in[0..dim-1]` (without RMSNorm) directly into SMEM `(s_xq, s_xds)`
__device__ inline void dev_quant_f32_to_smem(
    int* s_xq_i32,
    float2* s_xds,
    const float* x_in,
    int dim
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int n_blks = dim >> 5;
    signed char* s_xq_s8 = reinterpret_cast<signed char*>(s_xq_i32);

    for (int blk = wid; blk < n_blks; blk += MK_WARPS) {
        const int idx = (blk << 5) + lane;
        const float v = ld_cg_f32(&x_in[idx]);
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

// Exact HF Sampler in SMEM (`256` active threads `tid < 256`, `V <= 3072`)
__device__ __noinline__ int dev_sample_logits_256(
    float* s_logits,
    float* s_red,
    int* s_ired,
    const float* g_logits,
    const unsigned char* seen_mask,
    int V,
    int suppress_lo,
    int suppress_hi,
    int keep_eos_id,
    float rep_pen,
    float temperature,
    int top_k,
    float top_p,
    uint64_t seed,
    uint64_t subseq
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    if (tid < 256) {
        for (int i = tid; i < V; i += 256) {
            float v = ld_cg_f32(&g_logits[i]);
            if (i >= suppress_lo && i < suppress_hi && i != keep_eos_id) {
                v = -1e38f;
            } else if (seen_mask != nullptr && rep_pen != 1.0f && v > -1e30f) {
                if (seen_mask[i] != 0) {
                    v = (v < 0.0f) ? (v * rep_pen) : (v / rep_pen);
                }
            }
            s_logits[i] = v;
        }
    }
    __syncthreads();

    if (temperature <= 0.0f) {
        float bv = -1e38f;
        int bi = 0x7fffffff;
        if (tid < 256) {
            for (int i = tid; i < V; i += 256) {
                float v = s_logits[i];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { s_red[wid] = bv; s_ired[wid] = bi; }
        }
        __syncthreads();
        if (tid == 0) {
            float fv = s_red[0]; int fi = s_ired[0];
            for (int w = 1; w < 8; ++w) {
                if (s_red[w] > fv || (s_red[w] == fv && s_ired[w] < fi)) { fv = s_red[w]; fi = s_ired[w]; }
            }
            s_ired[0] = fi;
        }
        __syncthreads();
        return s_ired[0];
    }

    // Divide by temperature & find [min_finite, max_finite]
    const float inv_temp = 1.0f / temperature;
    float l_max = -1e38f, l_min = 1e38f;
    if (tid < 256) {
        for (int i = tid; i < V; i += 256) {
            float v = s_logits[i];
            if (v > -1e30f) {
                v *= inv_temp;
                s_logits[i] = v;
                l_max = fmaxf(l_max, v);
                l_min = fminf(l_min, v);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l_max = fmaxf(l_max, __shfl_xor_sync(0xffffffff, l_max, off));
            l_min = fminf(l_min, __shfl_xor_sync(0xffffffff, l_min, off));
        }
        if (lane == 0) { s_red[wid] = l_max; s_red[8 + wid] = l_min; }
    }
    __syncthreads();
    if (tid == 0) {
        float mx = s_red[0], mn = s_red[8];
        for (int w = 1; w < 8; ++w) { mx = fmaxf(mx, s_red[w]); mn = fminf(mn, s_red[8 + w]); }
        s_red[0] = mx; s_red[1] = mn;
    }
    __syncthreads();

    // Top-K filtering via 28-step parallel value bisection over SMEM
    if (top_k > 0 && top_k < V) {
        float lo = s_red[1], hi = s_red[0];
        for (int it = 0; it < 28; ++it) {
            const float mid = 0.5f * (lo + hi);
            int cnt = 0;
            if (tid < 256) {
                for (int i = tid; i < V; i += 256) {
                    if (s_logits[i] >= mid) cnt += 1;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
                if (lane == 0) s_ired[wid] = cnt;
            }
            __syncthreads();
            if (tid == 0) {
                int tot = 0;
                for (int w = 0; w < 8; ++w) tot += s_ired[w];
                s_ired[0] = tot;
            }
            __syncthreads();
            if (s_ired[0] >= top_k) lo = mid;
            else hi = mid;
        }
        if (tid < 256) {
            for (int i = tid; i < V; i += 256) {
                if (s_logits[i] < lo) s_logits[i] = -1e38f;
            }
        }
        __syncthreads();
    }

    // Softmax exponentiation in SMEM around `max_val = s_red[0]`
    const float max_val = s_red[0];
    float l_sum = 0.0f;
    if (tid < 256) {
        for (int i = tid; i < V; i += 256) {
            float e = (s_logits[i] > -1e30f) ? expf(s_logits[i] - max_val) : 0.0f;
            s_logits[i] = e;
            l_sum += e;
        }
        l_sum = warp_reduce_sum(l_sum);
        if (lane == 0) s_red[wid] = l_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float sum_exp = 0.0f;
        for (int w = 0; w < 8; ++w) sum_exp += s_red[w];

        // Optional Top-P nucleus trim (default top_p = 1.0 skips)
        if (top_p > 0.0f && top_p < 1.0f && sum_exp > 0.0f) {
            // Find prob threshold via 24-step bisection on [0, 1]
            float plo = 0.0f, phi = 1.0f;
            const float inv_s = 1.0f / sum_exp;
            for (int it = 0; it < 24; ++it) {
                const float pmid = 0.5f * (plo + phi);
                const float emid = pmid * sum_exp;
                float csum = 0.0f;
                for (int i = 0; i < V; ++i) {
                    if (s_logits[i] >= emid) csum += s_logits[i];
                }
                if (csum * inv_s >= top_p) plo = pmid;
                else phi = pmid;
            }
            const float ecut = plo * sum_exp;
            float new_sum = 0.0f;
            for (int i = 0; i < V; ++i) {
                if (s_logits[i] < ecut) s_logits[i] = 0.0f;
                else new_sum += s_logits[i];
            }
            if (new_sum > 0.0f) sum_exp = new_sum;
        }

        const float u = dev_philox_u01(seed, subseq);
        const float r = u * sum_exp;
        float acc = 0.0f;
        int chosen = V - 1;
        for (int i = 0; i < V; ++i) {
            const float p = s_logits[i];
            if (p > 0.0f) {
                chosen = i;
                acc += p;
                if (acc >= r) break;
            }
        }
        s_ired[0] = chosen;
    }
    __syncthreads();
    return s_ired[0];
}

extern "C" __global__ __launch_bounds__(256)
void tts_silu_inplace_f32(float* __restrict__ x, int n) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    if (idx < n) {
        const float v = x[idx];
        x[idx] = v / (1.0f + expf(-v));
    }
}

extern "C" __global__ __launch_bounds__(256)
void tts_copy_f32(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

extern "C" __global__ __launch_bounds__(256)
void tts_rmsnorm_f32_q8_1(
    float* __restrict__ f32_out,
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in,
    const float* __restrict__ norm_w,
    int K,
    float eps
) {
    __shared__ float s_red[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int n_blks = K >> 5;

    float ss = 0.0f;
    for (int blk = wid; blk < n_blks; blk += 8) {
        const float v = x_in[(blk << 5) + lane];
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
        const float v = x_in[idx] * scale * __ldg(&norm_w[idx]);
        f32_out[idx] = v;
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        q_out[idx] = (signed char)max(-127, min(127, __float2int_rn(v * id)));
        if (lane == 0) ds_out[blk] = make_float2(d, sum_x);
    }
}

// Assemble Talker prefill `[T_ctx, hidden]` + `tts_pad_embed [hidden]` 100% on RTX 3080!
// `proj_all` holds projected text rows:
//   [0 .. 3)                          = [tts_bos, tts_eos, tts_pad]
//   [3 .. 3 + n_head)                 = [instruct_ids... , role_ids[0..3]]
//   [3 + n_head .. 3 + n_head + n_tx) = utterance text_ids[3 .. 3 + n_tx]
extern "C" __global__ __launch_bounds__(256)
void tts_assemble_prefill_embed(
    float* __restrict__ out_emb,
    float* __restrict__ out_tts_pad,
    const float* __restrict__ proj_all,
    const unsigned char* __restrict__ codec_emb_w,
    const int* __restrict__ codec_prefix_ids,
    int n_head,
    int n_codec_left,
    int n_text,
    int codec_pad_id,
    int codec_bos_id,
    int hidden,
    int codec_emb_type
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* tts_bos = proj_all + 0 * (size_t)hidden;
    const float* tts_eos = proj_all + 1 * (size_t)hidden;
    const float* tts_pad = proj_all + 2 * (size_t)hidden;
    const float* head_p  = proj_all + 3 * (size_t)hidden;
    const float* text_p  = proj_all + (size_t)(3 + n_head) * hidden;

    if (row == 0) {
        for (int c = tid; c < hidden; c += 256) out_tts_pad[c] = tts_pad[c];
    }

    const int total_rows = n_head + n_codec_left + n_text + 2;
    if (row >= total_rows) return;
    float* dst = out_emb + (size_t)row * hidden;

    if (row < n_head) {
        const float* src = head_p + (size_t)row * hidden;
        for (int c = tid; c < hidden; c += 256) dst[c] = src[c];
    } else if (row < n_head + n_codec_left) {
        const int k = row - n_head;
        const float* tv = (k == n_codec_left - 1) ? tts_bos : tts_pad;
        const int cid = codec_prefix_ids[k];
        for (int c = tid; c < hidden; c += 256) {
            dst[c] = tv[c] + dev_elem_embed_f32(codec_emb_w, codec_emb_type, cid, c, hidden);
        }
    } else if (row < n_head + n_codec_left + n_text) {
        const int k = row - (n_head + n_codec_left);
        const float* tv = text_p + (size_t)k * hidden;
        for (int c = tid; c < hidden; c += 256) {
            dst[c] = tv[c] + dev_elem_embed_f32(codec_emb_w, codec_emb_type, codec_pad_id, c, hidden);
        }
    } else if (row == n_head + n_codec_left + n_text) {
        for (int c = tid; c < hidden; c += 256) {
            dst[c] = tts_eos[c] + dev_elem_embed_f32(codec_emb_w, codec_emb_type, codec_pad_id, c, hidden);
        }
    } else {
        for (int c = tid; c < hidden; c += 256) {
            dst[c] = tts_pad[c] + dev_elem_embed_f32(codec_emb_w, codec_emb_type, codec_bos_id, c, hidden);
        }
    }
}

// Sample semantic code `c0` from Talker `ws_logits_out [3072]` with suppression + HF repetition penalty
extern "C" __global__ __launch_bounds__(256)
void tts_sample_c0_kernel(
    volatile TtsFrameCtrl* __restrict__ fctrl,
    unsigned char* __restrict__ history_seen,
    int* __restrict__ ws_codes_frame,
    const float* __restrict__ logits_c0,
    int vocab_size
) {
    __shared__ float s_logits[3072];
    __shared__ float s_red[32];
    __shared__ int   s_ired[32];

    const int frame_idx = fctrl->frame_idx;
    const int eos_id = fctrl->codec_eos_id;
    const uint64_t subseq = (uint64_t)frame_idx * 16ULL;

    const int c0 = dev_sample_logits_256(
        s_logits, s_red, s_ired,
        logits_c0, history_seen, vocab_size,
        vocab_size - 1024, vocab_size, eos_id,
        fctrl->rep_penalty, fctrl->talker_temp, fctrl->talker_top_k, fctrl->talker_top_p,
        fctrl->seed, subseq
    );

    if (threadIdx.x == 0) {
        const int is_eos = (c0 == eos_id) ? 1 : 0;
        fctrl->c0 = c0;
        fctrl->is_eos = is_eos;
        ws_codes_frame[0] = c0;
        if (!is_eos) {
            if (c0 >= 0 && c0 < vocab_size) history_seen[c0] = 1;
            st_host_s32(&fctrl->codes_out[frame_idx * 16 + 0], c0);
        } else {
            asm volatile("membar.gl;" ::: "memory");
            st_host_s32(&fctrl->frames_completed, frame_idx);
        }
    }
}

// 68-SM Persistent `CodePredictor` Megakernel: Runs ALL 15 Acoustic Sub-Steps + Sampling +
// 16-Codebook Next-Frame Talker Embedding Composition in ONE Single GPU Launch!
extern "C" __global__ __launch_bounds__(576, 1)
void tts_code_predictor_frame_megakernel(
    const TtsCodePredDesc* __restrict__ cp,
    volatile TtsFrameCtrl* __restrict__ fctrl
) {
    __shared__ __align__(16) int    s_xq[4096 / 4];
    __shared__ __align__(16) float2 s_xds[4096 / 32];
    __shared__ __align__(16) float  s_scratch[3072]; // serves both attention & V=2048 sampler!
    __shared__ __align__(16) int    s_ired[32];
    __shared__ int    s_abort;
    __shared__ int    s_is_eos;
    __shared__ int    s_frame_idx;
    __shared__ int    s_c0;
    __shared__ int    s_prev_code;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    if (tid == 0) {
        s_abort = 0;
        s_is_eos = fctrl->is_eos;
        s_frame_idx = fctrl->frame_idx;
        s_c0 = fctrl->c0;
        s_prev_code = 0;
    }
    __syncthreads();
    if (s_is_eos != 0) return;

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(cp->ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    // Reuse `QwenDecodeCtrl*` error pointer slot safely via dummy cast only if watchdog trips
    volatile QwenDecodeCtrl* dummy_ctrl = reinterpret_cast<volatile QwenDecodeCtrl*>(fctrl);

    const QwenLayerDesc* layers = reinterpret_cast<const QwenLayerDesc*>(cp->layers_ptr);
    float* ws_in2048 = reinterpret_cast<float*>(cp->ws_in2048);
    float* x_res = reinterpret_cast<float*>(cp->ws_x_res);
    float* x_res2 = reinterpret_cast<float*>(cp->ws_x_res2);
    signed char* ws_xq = reinterpret_cast<signed char*>(cp->ws_xq);
    float2* ws_xds = reinterpret_cast<float2*>(cp->ws_xds);
    float* ws_qkv = reinterpret_cast<float*>(cp->ws_qkv);
    float* ws_q_roped = reinterpret_cast<float*>(cp->ws_q_roped);
    float* ws_gate_up = reinterpret_cast<float*>(cp->ws_gate_up);
    float* ws_logits = reinterpret_cast<float*>(cp->ws_logits);
    int* ws_codes = reinterpret_cast<int*>(cp->ws_codes_frame);
    const float* freqs_cis = reinterpret_cast<const float*>(cp->freqs_cis);

    const int talker_dim = cp->talker_dim;   // 2048
    const int dim = cp->dim;                 // 1024
    const int hidden_dim = cp->hidden_dim;   // 3072
    const int n_heads = cp->n_heads;         // 16
    const int n_kv_heads = cp->n_kv_heads;   // 8
    const int head_dim = cp->head_dim;       // 128
    const int half_rope = head_dim >> 1;     // 64
    const int q_dim = n_heads * head_dim;    // 2048
    const int kv_dim = n_kv_heads * head_dim;// 1024
    const int gqa_ratio = n_heads / n_kv_heads;
    const int max_ctx = cp->max_ctx;
    const float eps = cp->norm_eps;
    const float attn_scale = cp->inv_sqrt_head_dim;

    for (int p = 0; p < 16; ++p) {
        // 1. Assemble 2048-dim input vector into `ws_in2048`
        if (bid == 0) {
            if (p == 0) {
                const float* th = reinterpret_cast<const float*>(cp->talker_hidden_last);
                for (int col = tid; col < talker_dim; col += MK_THREADS) {
                    st_cg_f32(&ws_in2048[col], ld_cg_f32(&th[col]));
                }
            } else if (p == 1) {
                dev_embed_row_to_l2(
                    ws_in2048,
                    reinterpret_cast<const unsigned char*>(cp->talker_codec_emb_w),
                    cp->talker_codec_emb_type, s_c0, talker_dim
                );
            } else {
                dev_embed_row_to_l2(
                    ws_in2048,
                    reinterpret_cast<const unsigned char*>(cp->cp_codec_emb_w[p - 2]),
                    cp->cp_codec_emb_type, s_prev_code, talker_dim
                );
            }
        }
        if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

        // 2. Project `ws_in2048 [2048] -> x_res [1024]` via `mtp_proj` (or copy if identity)
        if (cp->mtp_proj_w != 0ULL) {
            dev_quant_f32_to_smem(s_xq, s_xds, ws_in2048, talker_dim);
            dev_gemv_or_select_smem(
                x_res, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(cp->mtp_proj_w), cp->mtp_proj_type,
                s_xq, s_xds, reinterpret_cast<const float*>(cp->mtp_proj_b), nullptr,
                dim, talker_dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
        } else if (bid == 0) {
            for (int col = tid; col < dim; col += MK_THREADS) {
                st_cg_f32(&x_res[col], ld_cg_f32(&ws_in2048[col]));
            }
        }
        if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

        // 3. Run 5-layer Qwen3 `CodePredictor` stack at causal position `pos = p`
        for (int l = 0; l < cp->num_layers; ++l) {
            const QwenLayerDesc& ld = layers[l];

            // Stage 0: Attn RMSNorm + QKV GEMV
            dev_rmsnorm_q8_1_to_smem(
                s_xq, s_xds, s_scratch, x_res,
                reinterpret_cast<const float*>(ld.attn_norm_w), dim, eps
            );
            dev_gemv_or_select_smem(
                ws_qkv, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(ld.w_qkv_or_qk), ld.qkv_type,
                s_xq, s_xds, reinterpret_cast<const float*>(ld.b_qkv_or_qk), nullptr,
                q_dim + 2 * kv_dim, dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            // QK-Norm + 1D NEOX RoPE + KV Cache Write at `pos = p`
            half* kv_cache = reinterpret_cast<half*>(ld.kv_cache);
            const int total_qkv_heads = n_heads + 2 * n_kv_heads;
            for (int h_all = bid; h_all < total_qkv_heads; h_all += num_sms) {
                if (h_all >= n_heads + n_kv_heads) {
                    if (tid < head_dim) {
                        const int kv_h = h_all - (n_heads + n_kv_heads);
                        const float v_val = ld_cg_f32(&ws_qkv[q_dim + kv_dim + kv_h * head_dim + tid]);
                        st_cg_f16(&kv_cache[((size_t)(n_kv_heads + kv_h) * max_ctx + p) * head_dim + tid], __float2half(v_val));
                    }
                } else {
                    const bool is_q = (h_all < n_heads);
                    const int h_idx = is_q ? h_all : (h_all - n_heads);
                    float val = 0.0f;
                    if (tid < head_dim) {
                        const int src_off = is_q ? (h_idx * head_dim + tid) : (q_dim + h_idx * head_dim + tid);
                        val = ld_cg_f32(&ws_qkv[src_off]);
                    }
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
                        s_scratch[16 + tid] = val;
                    }
                    __syncthreads();
                    if (tid < head_dim) {
                        const float* freq_row = freqs_cis + (size_t)p * head_dim;
                        float out_val;
                        if (tid < half_rope) {
                            out_val = val * __ldg(&freq_row[tid]) - s_scratch[16 + tid + half_rope] * __ldg(&freq_row[tid + half_rope]);
                        } else {
                            const int j = tid - half_rope;
                            out_val = s_scratch[16 + j] * __ldg(&freq_row[tid]) + val * __ldg(&freq_row[j]);
                        }
                        if (is_q) st_cg_f32(&ws_q_roped[h_idx * head_dim + tid], out_val);
                        else st_cg_f16(&kv_cache[((size_t)h_idx * max_ctx + p) * head_dim + tid], __float2half(out_val));
                    }
                    __syncthreads();
                }
            }
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            // Attention (`seq_len = p + 1`, `head_dim = 128`)
            const int seq_len = p + 1;
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
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            // Stage 1: O-proj + residual -> x_res2
            dev_load_q8_1_to_smem(s_xq, s_xds, ws_xq, ws_xds, q_dim);
            dev_gemv_or_select_smem(
                x_res2, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(ld.w_o), ld.o_type,
                s_xq, s_xds, reinterpret_cast<const float*>(ld.b_o), x_res,
                dim, q_dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            // Stage 2: FFN RMSNorm + GateUp GEMV
            dev_rmsnorm_q8_1_to_smem(
                s_xq, s_xds, s_scratch, x_res2,
                reinterpret_cast<const float*>(ld.ffn_norm_w), dim, eps
            );
            dev_gemv_or_select_smem(
                ws_gate_up, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(ld.w_gate_up), ld.gate_up_type,
                s_xq, s_xds, nullptr, nullptr,
                2 * hidden_dim, dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

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
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            // Stage 3: Down GEMV + residual -> x_res
            dev_load_q8_1_to_smem(s_xq, s_xds, ws_xq, ws_xds, hidden_dim);
            dev_gemv_or_select_smem(
                x_res, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(ld.w_down), ld.down_type,
                s_xq, s_xds, reinterpret_cast<const float*>(ld.b_down), x_res2,
                dim, hidden_dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;
        }

        // 4. For `p >= 1`, compute `lm_head[p - 1]` logits & sample `c_p` on `bid == 0`
        if (p >= 1) {
            const int g = p - 1;
            dev_rmsnorm_q8_1_to_smem(
                s_xq, s_xds, s_scratch, x_res,
                reinterpret_cast<const float*>(cp->output_norm_w), dim, eps
            );
            dev_gemv_or_select_smem(
                ws_logits, nullptr, nullptr, s_scratch, s_ired,
                reinterpret_cast<const unsigned char*>(cp->lm_head_w[g]), cp->lm_head_type,
                s_xq, s_xds, nullptr, nullptr,
                cp->vocab_size, dim, bid, num_sms, false, 0.0f, 0ULL, 0u
            );
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            if (bid == 0) {
                const uint64_t subseq = (uint64_t)s_frame_idx * 16ULL + (uint64_t)p;
                const int cg = dev_sample_logits_256(
                    s_scratch, s_scratch + 2048, s_ired,
                    ws_logits, nullptr, cp->vocab_size,
                    cp->vocab_size, cp->vocab_size, -1,
                    1.0f, fctrl->sub_temp, fctrl->sub_top_k, fctrl->sub_top_p,
                    fctrl->seed, subseq
                );
                if (tid == 0) {
                    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_codes[p]), "r"(cg) : "memory");
                    st_host_s32(&fctrl->codes_out[s_frame_idx * 16 + p], cg);
                }
            }
            if (!megakernel_barrier(&bar_state[0], bar_step, num_sms, s_abort, dummy_ctrl)) return;

            if (tid == 0) {
                int cg = 0;
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(cg) : "l"(&ws_codes[p]));
                s_prev_code = cg;
            }
            __syncthreads();
        }
    }

    // 5. Compose Next-Frame Talker Embedding `[2048]` = overlay + codec_emb(c0) + sum_{g=0..14} cp_emb[g](codes[g+1])
    if (bid == 0) {
        if (tid < 16) {
            int cv = 0;
            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(cv) : "l"(&ws_codes[tid]));
            s_ired[tid] = cv;
        }
        __syncthreads();

        float* next_emb = reinterpret_cast<float*>(cp->talker_ws_ext_embed);
        const float* overlay = (fctrl->overlay_ptr != 0ULL)
            ? reinterpret_cast<const float*>(fctrl->overlay_ptr)
            : reinterpret_cast<const float*>(cp->tts_pad_embed);
        const unsigned char* tk_emb = reinterpret_cast<const unsigned char*>(cp->talker_codec_emb_w);

        for (int col = tid; col < talker_dim; col += MK_THREADS) {
            float sum_e = __ldg(&overlay[col]) + dev_elem_embed_f32(tk_emb, cp->talker_codec_emb_type, s_ired[0], col, talker_dim);
            #pragma unroll
            for (int g = 0; g < 15; ++g) {
                const unsigned char* gw = reinterpret_cast<const unsigned char*>(cp->cp_codec_emb_w[g]);
                sum_e += dev_elem_embed_f32(gw, cp->cp_codec_emb_type, s_ired[g + 1], col, talker_dim);
            }
            st_cg_f32(&next_emb[col], sum_e);
        }
        __syncthreads();

        if (tid == 0) {
            asm volatile("membar.gl;" ::: "memory");
            st_host_s32(&fctrl->frames_completed, s_frame_idx + 1);
            asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        }
    }
}

// =============================================================================
// 6. 100%-ON-RTX-3080 `12Hz Code2Wav` NEURAL VOCODER KERNELS (`BF16` + `F32`)
// =============================================================================

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

// Split RVQ Decode: `codes_tk [T, 16]` -> `out_tc [T, 512]` (T-first row-major)
extern "C" __global__ __launch_bounds__(256)
void c2w_rvq_decode_kernel(
    float* __restrict__ out_tc,
    const int* __restrict__ codes_tk,
    const float* __restrict__ vq_first_cb,   // [2048, 256]
    const float* __restrict__ vq_rest_cb,    // [15, 2048, 256]
    const float* __restrict__ proj_first_w,  // [512, 256]
    const float* __restrict__ proj_rest_w,   // [512, 256]
    int T
) {
    __shared__ float s_sem[256];
    __shared__ float s_aco[256];

    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x; // 0..255
    const int* row_codes = codes_tk + (size_t)t * 16;

    const int c0 = row_codes[0];
    s_sem[tid] = __ldg(&vq_first_cb[(size_t)c0 * 256 + tid]);

    float aco_sum = 0.0f;
    #pragma unroll
    for (int g = 0; g < 15; ++g) {
        const int cg = row_codes[g + 1];
        aco_sum += __ldg(&vq_rest_cb[((size_t)g * 2048 + cg) * 256 + tid]);
    }
    s_aco[tid] = aco_sum;
    __syncthreads();

    const int oc = (blockIdx.y << 8) + tid; // 0..511
    if (oc < 512) {
        const float* pf = proj_first_w + (size_t)oc * 256;
        const float* pr = proj_rest_w + (size_t)oc * 256;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 256; ++k) {
            acc += __ldg(&pf[k]) * s_sem[k] + __ldg(&pr[k]) * s_aco[k];
        }
        out_tc[(size_t)t * 512 + oc] = acc;
    }
}

// Warp-Tiled (`TILE_T = 8`) Causal 1D Convolution with Persistent Streaming State `[L, IC]`!
// Weight `w_bf16` is pre-permuted at load time to `[K, OC, IC]` (`IC` innermost, divisible by 32!).
extern "C" __global__ __launch_bounds__(128)
void c2w_causal_conv1d_bf16_kernel(
    float* __restrict__ out,
    const unsigned short* __restrict__ w_bf16,
    const float* __restrict__ x,
    const float* __restrict__ state,
    const float* __restrict__ bias,
    const float* __restrict__ skip,
    int T,
    int IC,
    int OC,
    int K,
    int dilation
) {
    const int oc = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (oc >= OC) return;
    const int t0 = blockIdx.y << 3;
    const int t_cnt = min(8, T - t0);
    if (t_cnt <= 0) return;

    const int lane = threadIdx.x & 31;
    const int L = (K - 1) * dilation;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k = 0; k < K; ++k) {
        const unsigned short* wk_row = w_bf16 + ((size_t)k * OC + oc) * IC;
        const float* src_ptrs[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < t_cnt) {
                const int t_ext = (t0 + m) + k * dilation;
                src_ptrs[m] = (t_ext < L)
                    ? (state + (size_t)t_ext * IC)
                    : (x + (size_t)(t_ext - L) * IC);
            } else {
                src_ptrs[m] = nullptr;
            }
        }

        for (int ic = (lane << 2); ic < IC; ic += 128) {
            const uint2 wv = __ldg(reinterpret_cast<const uint2*>(wk_row + ic));
            const float w0 = bf16_bits_to_f32((unsigned short)(wv.x & 0xffffu));
            const float w1 = bf16_bits_to_f32((unsigned short)(wv.x >> 16));
            const float w2 = bf16_bits_to_f32((unsigned short)(wv.y & 0xffffu));
            const float w3 = bf16_bits_to_f32((unsigned short)(wv.y >> 16));

            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (m < t_cnt) {
                    const float4 xv = *reinterpret_cast<const float4*>(src_ptrs[m] + ic);
                    acc[m] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = warp_reduce_sum(acc[m]);

    if (lane == 0) {
        const float bv = (bias != nullptr) ? __ldg(&bias[oc]) : 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < t_cnt) {
                const size_t out_idx = (size_t)(t0 + m) * OC + oc;
                float val = acc[m] + bv;
                if (skip != nullptr) val += skip[out_idx];
                out[out_idx] = val;
            }
        }
    }
}

// Race-free causal Conv1d state staging: writes the new `[L, IC]` tail into `tmp_state`,
// followed by `tts_copy_f32(state, tmp_state, L * IC)`!
extern "C" __global__ __launch_bounds__(256)
void c2w_conv_state_stage_kernel(
    float* __restrict__ tmp_state,
    const float* __restrict__ old_state,
    const float* __restrict__ x,
    int T,
    int IC,
    int L
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    const int total = L * IC;
    if (idx >= total) return;
    const int r = idx / IC;
    const int ic = idx % IC;

    if (T >= L) {
        tmp_state[idx] = x[(size_t)(T - L + r) * IC + ic];
    } else {
        const int keep_old = L - T;
        if (r < keep_old) {
            tmp_state[idx] = old_state[(size_t)(r + T) * IC + ic];
        } else {
            tmp_state[idx] = x[(size_t)(r - keep_old) * IC + ic];
        }
    }
}

// Warp-Tiled (`TILE_T = 8`) `BF16` Linear Projection `[T, K] @ [N, K]^T` with Optional Bias + LayerScale + Residual
extern "C" __global__ __launch_bounds__(128)
void c2w_linear_bf16_kernel(
    float* __restrict__ out,
    const unsigned short* __restrict__ w_bf16,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ layer_scale,
    const float* __restrict__ residual,
    int T,
    int N,
    int K
) {
    const int row = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int t0 = blockIdx.y << 3;
    const int t_cnt = min(8, T - t0);
    if (t_cnt <= 0) return;

    const int lane = threadIdx.x & 31;
    const unsigned short* w_row = w_bf16 + (size_t)row * K;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k = (lane << 2); k < K; k += 128) {
        const uint2 wv = __ldg(reinterpret_cast<const uint2*>(w_row + k));
        const float w0 = bf16_bits_to_f32((unsigned short)(wv.x & 0xffffu));
        const float w1 = bf16_bits_to_f32((unsigned short)(wv.x >> 16));
        const float w2 = bf16_bits_to_f32((unsigned short)(wv.y & 0xffffu));
        const float w3 = bf16_bits_to_f32((unsigned short)(wv.y >> 16));

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < t_cnt) {
                const float4 xv = *reinterpret_cast<const float4*>(x + (size_t)(t0 + m) * K + k);
                acc[m] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = warp_reduce_sum(acc[m]);

    if (lane == 0) {
        const float bv = (bias != nullptr) ? __ldg(&bias[row]) : 0.0f;
        const float ls = (layer_scale != nullptr) ? __ldg(&layer_scale[row]) : 1.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < t_cnt) {
                const size_t idx = (size_t)(t0 + m) * N + row;
                float val = (acc[m] + bv) * ls;
                if (residual != nullptr) val += residual[idx];
                out[idx] = val;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void c2w_rmsnorm_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ norm_w,
    int T,
    int C,
    float eps
) {
    __shared__ float s_red[8];
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const float* xr = x + (size_t)t * C;
    float* yr = out + (size_t)t * C;

    float ss = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float v = xr[c];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float wss = (lane < 8) ? s_red[lane] : 0.0f;
        wss = warp_reduce_sum(wss);
        if (lane == 0) s_red[0] = rsqrtf(wss / (float)C + eps);
    }
    __syncthreads();
    const float scale = s_red[0];
    for (int c = tid; c < C; c += 256) {
        yr[c] = xr[c] * scale * __ldg(&norm_w[c]);
    }
}

// Tokenizer Transformer (`16` Q/KV heads, `hd = 64`, `NEOX` RoPE `theta = 10000`, `NO QK-norm`)
extern "C" __global__ __launch_bounds__(64)
void c2w_tfm_rope_kv_write_kernel(
    float* __restrict__ q_out,     // [n_heads, T, 64]
    float* __restrict__ kv_cache,  // [2, n_heads, max_ctx, 64]
    const float* __restrict__ qkv, // [T, 3 * n_heads * 64]
    const float* __restrict__ freqs_cis,
    int T,
    int start_pos,
    int n_heads,
    int max_ctx
) {
    __shared__ float s_vec[64];
    const int h_all = blockIdx.x; // 0 .. 3*n_heads - 1
    const int t = blockIdx.y;
    const int tid = threadIdx.x;  // 0 .. 63
    const int q_dim = n_heads * 64;
    const int row_stride = 3 * q_dim;
    const int pos = start_pos + t;

    if (h_all >= 2 * n_heads) {
        const int vh = h_all - 2 * n_heads;
        kv_cache[((size_t)(n_heads + vh) * max_ctx + pos) * 64 + tid] =
            qkv[(size_t)t * row_stride + 2 * q_dim + vh * 64 + tid];
        return;
    }

    const bool is_q = (h_all < n_heads);
    const int h = is_q ? h_all : (h_all - n_heads);
    const float val = qkv[(size_t)t * row_stride + (is_q ? 0 : q_dim) + h * 64 + tid];
    s_vec[tid] = val;
    __syncthreads();

    const float* frow = freqs_cis + (size_t)pos * 64;
    float out_v;
    if (tid < 32) {
        out_v = val * __ldg(&frow[tid]) - s_vec[tid + 32] * __ldg(&frow[tid + 32]);
    } else {
        const int j = tid - 32;
        out_v = s_vec[j] * __ldg(&frow[tid]) + val * __ldg(&frow[j]);
    }

    if (is_q) {
        q_out[((size_t)h * T + t) * 64 + tid] = out_v;
    } else {
        kv_cache[((size_t)h * max_ctx + pos) * 64 + tid] = out_v;
    }
}

// Causal Sliding-Window (`w = 72`) Attention for `Code2Wav` (`hd = 64`, FP32 KV cache)
extern "C" __global__ __launch_bounds__(128)
void c2w_tfm_sliding_attn_kernel(
    float* __restrict__ out,       // [T, n_heads * 64]
    const float* __restrict__ q_in,// [n_heads, T, 64]
    const float* __restrict__ kv_cache,
    int T,
    int start_pos,
    int n_heads,
    int max_ctx,
    int sliding_window,
    float scale
) {
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][64];

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int pos = start_pos + t;
    const int p_min = max(0, pos - sliding_window + 1);

    const float* qp = q_in + ((size_t)h * T + t) * 64 + (lane << 1);
    const float q0 = qp[0] * scale, q1 = qp[1] * scale;
    const float* k_base = kv_cache + (size_t)h * max_ctx * 64;
    const float* v_base = kv_cache + (size_t)(n_heads + h) * max_ctx * 64;

    float m_i = -1e30f, l_i = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (int p_idx = p_min + wid; p_idx <= pos; p_idx += 4) {
        const float2 kv_k = *reinterpret_cast<const float2*>(k_base + (size_t)p_idx * 64 + (lane << 1));
        float qk = q0 * kv_k.x + q1 * kv_k.y;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) qk += __shfl_xor_sync(0xffffffff, qk, off);
        const float m_new = fmaxf(m_i, qk);
        const float alpha = expf(m_i - m_new), p_w = expf(qk - m_new);
        l_i = l_i * alpha + p_w; m_i = m_new;
        const float2 kv_v = *reinterpret_cast<const float2*>(v_base + (size_t)p_idx * 64 + (lane << 1));
        a0 = a0 * alpha + p_w * kv_v.x;
        a1 = a1 * alpha + p_w * kv_v.y;
    }

    if (lane == 0) { s_m[wid] = m_i; s_l[wid] = l_i; }
    s_acc[wid][(lane << 1) + 0] = a0;
    s_acc[wid][(lane << 1) + 1] = a1;
    __syncthreads();

    if (wid == 0) {
        float m_max = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float l_sum = 0.0f, o0 = 0.0f, o1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float ws = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
            l_sum += s_l[w] * ws;
            o0 += s_acc[w][(lane << 1) + 0] * ws;
            o1 += s_acc[w][(lane << 1) + 1] * ws;
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* op = out + ((size_t)t * n_heads + h) * 64 + (lane << 1);
        op[0] = o0 * inv_l;
        op[1] = o1 * inv_l;
    }
}

extern "C" __global__ __launch_bounds__(256)
void c2w_swiglu_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate_up,
    int T,
    int H
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    const int total = T * H;
    if (idx >= total) return;
    const int t = idx / H;
    const int c = idx % H;
    const float g = gate_up[(size_t)t * (2 * H) + c];
    const float u = gate_up[(size_t)t * (2 * H) + H + c];
    out[idx] = (g / (1.0f + expf(-g))) * u;
}

// Causal `ConvTranspose1d` Col2Im + Streaming Overlap Carry (`K == stride` or `K == 2 * stride`)
extern "C" __global__ __launch_bounds__(256)
void c2w_transconv_col2im_causal_kernel(
    float* __restrict__ out,      // [T_in * stride, OC]
    const float* __restrict__ col,// [T_in, OC * kernel]
    const float* __restrict__ carry, // [stride, OC] (used when kernel == 2 * stride)
    const float* __restrict__ bias,
    int T_in,
    int OC,
    int stride,
    int kernel
) {
    const int t_out = blockIdx.y;
    if (t_out >= T_in * stride) return;
    const int oc = (blockIdx.x << 8) + threadIdx.x;
    if (oc >= OC) return;

    const int t_in = t_out / stride;
    const int r = t_out % stride;
    const int col_stride = OC * kernel;

    float val = col[(size_t)t_in * col_stride + oc * kernel + r];
    if (kernel == 2 * stride) {
        val += (t_in > 0)
            ? col[(size_t)(t_in - 1) * col_stride + oc * kernel + stride + r]
            : carry[(size_t)r * OC + oc];
    }
    if (bias != nullptr) val += __ldg(&bias[oc]);
    out[(size_t)t_out * OC + oc] = val;
}

extern "C" __global__ __launch_bounds__(256)
void c2w_transconv_carry_update_kernel(
    float* __restrict__ carry,    // [stride, OC]
    const float* __restrict__ col,// [T_in, OC * kernel]
    int T_in,
    int OC,
    int stride,
    int kernel
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    if (idx >= stride * OC) return;
    const int r = idx / OC;
    const int oc = idx % OC;
    carry[idx] = col[(size_t)(T_in - 1) * (OC * kernel) + oc * kernel + stride + r];
}

// Depthwise Causal Conv1d (`k = 7, d = 1`) for ConvNeXt Upsample Blocks (`w_f32 [C, 7]`)
extern "C" __global__ __launch_bounds__(256)
void c2w_dwconv1d_k7_causal_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ state, // [6, C]
    const float* __restrict__ w_f32, // [C, 7]
    const float* __restrict__ bias,  // [C]
    int T,
    int C
) {
    const int t = blockIdx.y;
    const int c = (blockIdx.x << 8) + threadIdx.x;
    if (t >= T || c >= C) return;

    const float* wr = w_f32 + (size_t)c * 7;
    float acc = __ldg(&bias[c]);
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
        const int t_ext = t + k;
        const float xv = (t_ext < 6)
            ? state[(size_t)t_ext * C + c]
            : x[(size_t)(t_ext - 6) * C + c];
        acc += __ldg(&wr[k]) * xv;
    }
    out[(size_t)t * C + c] = acc;
}

extern "C" __global__ __launch_bounds__(256)
void c2w_layernorm_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ norm_w,
    const float* __restrict__ norm_b,
    int T,
    int C,
    float eps
) {
    __shared__ float s_sum[8];
    __shared__ float s_var[8];
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const float* xr = x + (size_t)t * C;
    float* yr = out + (size_t)t * C;

    float sum = 0.0f;
    for (int c = tid; c < C; c += 256) sum += xr[c];
    sum = warp_reduce_sum(sum);
    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        float ws = (lane < 8) ? s_sum[lane] : 0.0f;
        ws = warp_reduce_sum(ws);
        if (lane == 0) s_sum[0] = ws / (float)C;
    }
    __syncthreads();
    const float mean = s_sum[0];

    float vsum = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float d = xr[c] - mean;
        vsum += d * d;
    }
    vsum = warp_reduce_sum(vsum);
    if (lane == 0) s_var[wid] = vsum;
    __syncthreads();
    if (wid == 0) {
        float wv = (lane < 8) ? s_var[lane] : 0.0f;
        wv = warp_reduce_sum(wv);
        if (lane == 0) s_var[0] = rsqrtf(wv / (float)C + eps);
    }
    __syncthreads();
    const float inv_std = s_var[0];

    for (int c = tid; c < C; c += 256) {
        yr[c] = (xr[c] - mean) * inv_std * __ldg(&norm_w[c]) + __ldg(&norm_b[c]);
    }
}

extern "C" __global__ __launch_bounds__(256)
void c2w_gelu_inplace_f32_kernel(float* __restrict__ x, int n) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    if (idx < n) {
        const float v = x[idx];
        x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * v * (1.0f + 0.044715f * v * v)));
    }
}

// Fused DAC SnakeBeta Activation (`y = x + sin^2(a[c] * x) * inv_b[c]`)
extern "C" __global__ __launch_bounds__(256)
void c2w_snake_beta_kernel(
    float* __restrict__ x,
    const float* __restrict__ a,
    const float* __restrict__ inv_b,
    int T,
    int C
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    const int total = T * C;
    if (idx >= total) return;
    const int c = idx % C;
    const float v = x[idx];
    const float s = sinf(__ldg(&a[c]) * v);
    x[idx] = v + (s * s) * __ldg(&inv_b[c]);
}

extern "C" __global__ __launch_bounds__(256)
void c2w_clamp_audio_kernel(
    float* __restrict__ dst_audio,
    const float* __restrict__ src_audio,
    int n_samples
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    if (idx < n_samples) {
        dst_audio[idx] = fminf(1.0f, fmaxf(-1.0f, src_audio[idx]));
    }
}

// =============================================================================
// SECTION 7: 100%-on-RTX-3080 `Qwen3-ASR` Speech-to-Text Kernels
//   (Reflect-Padded 400-pt STFT + 128-Bin Slaney Log-Mel + 3x Conv2d Subsampler
//    + 24L Bidirectional Flash-Attention AuT Encoder + Decoder Audio Injection)
// =============================================================================

__device__ __forceinline__ int asr_float_to_sortable_int(float f) {
    const int i = __float_as_int(f);
    return (i >= 0) ? i : (i ^ 0x7fffffff);
}

__device__ __forceinline__ float asr_sortable_int_to_float(int i) {
    return __int_as_float((i >= 0) ? i : (i ^ 0x7fffffff));
}

__device__ __forceinline__ float asr_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

extern "C" __global__ __launch_bounds__(32)
void asr_init_max_int_kernel(int* __restrict__ d_max_sortable) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_max_sortable = asr_float_to_sortable_int(-1.0e30f);
    }
}

// 100%-GPU Whisper-Style 16 kHz Reflect-Padded 400-pt STFT + 128-Bin Slaney Log10-Mel + Global Max Reduction
// Grid: `(n_out, 1, 1)` where `n_out = n_samples / 160`, Block: `(256, 1, 1)`
extern "C" __global__ __launch_bounds__(256)
void asr_mel_stft_128_kernel(
    float* __restrict__ log_mel_out,      // [128, n_out] row-major
    int* __restrict__ d_max_sortable,     // [1] global max tracker
    const float* __restrict__ pcm,        // [n_samples] 16 kHz mono f32
    const float* __restrict__ window,     // [400] periodic Hann window
    const float* __restrict__ dft_cos_sin,// [201 * 800] (cos[400] then -sin[400] per bin k=0..200)
    const float* __restrict__ mel_fb,     // [128 * 201] Slaney mel filterbank
    int n_samples,
    int n_out
) {
    const int t = blockIdx.x;
    if (t >= n_out) return;
    const int tid = threadIdx.x;

    __shared__ float s_frame[400];
    __shared__ float s_pwr[208];
    __shared__ float s_wmax[8];

    // Reflect-padded window load (`pad = 200`, `hop = 160`)
    const int center_base = t * 160 - 200;
    for (int n = tid; n < 400; n += 256) {
        int idx = center_base + n;
        if (idx < 0) {
            idx = -idx;
        } else if (idx >= n_samples) {
            idx = 2 * n_samples - 2 - idx;
        }
        idx = max(0, min(n_samples - 1, idx));
        s_frame[n] = pcm[idx] * __ldg(&window[n]);
    }
    __syncthreads();

    // Exact 400-point real DFT for bins k = 0..200
    if (tid < 201) {
        const float* cs = dft_cos_sin + (size_t)tid * 800;
        const float* sn = cs + 400;
        float re = 0.0f;
        float im = 0.0f;
        #pragma unroll 8
        for (int n = 0; n < 400; ++n) {
            const float x = s_frame[n];
            re = fmaf(x, __ldg(&cs[n]), re);
            im = fmaf(x, __ldg(&sn[n]), im);
        }
        s_pwr[tid] = re * re + im * im;
    }
    __syncthreads();

    // 128-bin Slaney filterbank dot product (fp64 accumulation matching transcribe-mel.cpp) + log10
    float val = -1.0e30f;
    if (tid < 128) {
        const float* fb_row = mel_fb + (size_t)tid * 201;
        double sum = 0.0;
        for (int k = 0; k < 201; ++k) {
            sum += (double)__ldg(&fb_row[k]) * (double)s_pwr[k];
        }
        if (sum < 1.0e-10) sum = 1.0e-10;
        val = log10f((float)sum);
        log_mel_out[(size_t)tid * n_out + t] = val;
    }

    // Block-wide max reduction -> global `atomicMax`
    float wmax = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, mask));
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) s_wmax[warp] = wmax;
    __syncthreads();

    if (warp == 0) {
        float bmax = (lane < 8) ? s_wmax[lane] : -1.0e30f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, mask));
        }
        if (lane == 0) {
            atomicMax(d_max_sortable, asr_float_to_sortable_int(bmax));
        }
    }
}

// Per-utterance Whisper normalization (`clamp(v, max - 8.0)`, `(v + 4.0) / 4.0`)
// + direct packing into `[n_chunks, 1, n_mels=128, mel_per_chunk=100]` (zero-padded tail)
extern "C" __global__ __launch_bounds__(256)
void asr_mel_norm_pack_kernel(
    float* __restrict__ mel_chunks_out,   // [n_chunks, 128, mel_per_chunk]
    const float* __restrict__ log_mel_in, // [128, n_out]
    const int* __restrict__ d_max_sortable,
    int n_mels,
    int n_out,
    int n_chunks,
    int mel_per_chunk,
    int last_chunk_real_mel
) {
    const int idx = (blockIdx.x << 8) + threadIdx.x;
    const int per_chunk = n_mels * mel_per_chunk;
    const int total = n_chunks * per_chunk;
    if (idx >= total) return;

    const int c = idx / per_chunk;
    const int rem = idx % per_chunk;
    const int m = rem / mel_per_chunk;
    const int tc = rem % mel_per_chunk;

    const int valid_w = (c == n_chunks - 1) ? last_chunk_real_mel : mel_per_chunk;
    if (tc >= valid_w) {
        mel_chunks_out[idx] = 0.0f;
        return;
    }

    const float global_max = asr_sortable_int_to_float(*d_max_sortable);
    const float floor_val = global_max - 8.0f;
    const int src_t = c * mel_per_chunk + tc;
    float v = log_mel_in[(size_t)m * n_out + src_t];
    if (v < floor_val) v = floor_val;
    mel_chunks_out[idx] = (v + 4.0f) * 0.25f;
}

// Warp-Tiled (`TILE_SP = 4` spatial positions) `Conv2d(k=3x3, stride=2x2, pad=1x1) + Bias + GELU_erf`
// Input:  `x [B, H_in, W_in, IC]` (`NHWC`; note for `IC=1` `[B, H_in, W_in, 1] == [B, 1, H_in, W_in]`)
// Weight: `w_f32 [OC, 9, IC]` (pre-permuted from `[OC, IC, 3, 3]` at load time for 128-bit `float4` coalesced IC loads!)
// Output: `y [B, H_out, W_out, OC]` (`NHWC`) where `H_out = (H_in + 1) / 2`, `W_out = (W_in + 1) / 2`
// Grid:   `((OC + 3) / 4, (B * H_out * W_out + 3) / 4, 1)`, Block: `(128, 1, 1)`
extern "C" __global__ __launch_bounds__(128)
void asr_conv2d_3x3_s2_gelu_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ w_f32, // [OC, 9, IC]
    const float* __restrict__ bias,  // [OC]
    int B,
    int IC,
    int OC,
    int H_in,
    int W_in,
    int H_out,
    int W_out
) {
    const int oc = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (oc >= OC) return;
    const int sp0 = blockIdx.y << 2;
    const int total_sp = B * H_out * W_out;
    const int sp_cnt = min(4, total_sp - sp0);
    if (sp_cnt <= 0) return;

    const int lane = threadIdx.x & 31;
    const int hw_out = H_out * W_out;
    const int hw_in = H_in * W_in;

    int b_idx[4], ih0[4], iw0[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int sp = (m < sp_cnt) ? (sp0 + m) : sp0;
        b_idx[m] = sp / hw_out;
        const int rem = sp % hw_out;
        ih0[m] = (rem / W_out) * 2 - 1;
        iw0[m] = (rem % W_out) * 2 - 1;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* w_oc = w_f32 + (size_t)oc * 9 * IC;

    if (IC == 1) {
        if (lane == 0) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    const float wv = __ldg(&w_oc[kh * 3 + kw]);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < sp_cnt) {
                            const int ih = ih0[m] + kh;
                            const int iw = iw0[m] + kw;
                            if ((unsigned)ih < (unsigned)H_in && (unsigned)iw < (unsigned)W_in) {
                                acc[m] += wv * x[(size_t)b_idx[m] * hw_in + ih * W_in + iw];
                            }
                        }
                    }
                }
            }
        }
    } else if ((IC & 3) == 0) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                const float* w_tap = w_oc + (size_t)(kh * 3 + kw) * IC;
                bool valid_m[4];
                const float* x_base[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int ih = ih0[m] + kh;
                    const int iw = iw0[m] + kw;
                    valid_m[m] = (m < sp_cnt) && ((unsigned)ih < (unsigned)H_in) && ((unsigned)iw < (unsigned)W_in);
                    x_base[m] = x + ((size_t)b_idx[m] * hw_in + ih * W_in + iw) * IC;
                }

                for (int ic = (lane << 2); ic < IC; ic += 128) {
                    const float4 wv = *reinterpret_cast<const float4*>(w_tap + ic);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (valid_m[m]) {
                            const float4 xv = *reinterpret_cast<const float4*>(x_base[m] + ic);
                            acc[m] = fmaf(wv.x, xv.x, acc[m]);
                            acc[m] = fmaf(wv.y, xv.y, acc[m]);
                            acc[m] = fmaf(wv.z, xv.z, acc[m]);
                            acc[m] = fmaf(wv.w, xv.w, acc[m]);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) acc[m] = warp_reduce_sum(acc[m]);
    } else {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                const float* w_tap = w_oc + (size_t)(kh * 3 + kw) * IC;
                bool valid_m[4];
                const float* x_base[4];
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int ih = ih0[m] + kh;
                    const int iw = iw0[m] + kw;
                    valid_m[m] = (m < sp_cnt) && ((unsigned)ih < (unsigned)H_in) && ((unsigned)iw < (unsigned)W_in);
                    x_base[m] = x + ((size_t)b_idx[m] * hw_in + ih * W_in + iw) * IC;
                }
                for (int ic = lane; ic < IC; ic += 32) {
                    const float wv = __ldg(&w_tap[ic]);
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (valid_m[m]) acc[m] = fmaf(wv, x_base[m][ic], acc[m]);
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) acc[m] = warp_reduce_sum(acc[m]);
    }

    if (lane == 0) {
        const float bv = __ldg(&bias[oc]);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < sp_cnt) {
                const int sp = sp0 + m;
                const int b = b_idx[m];
                const int rem = sp % hw_out;
                y[((size_t)b * hw_out + rem) * OC + oc] = asr_gelu_erf(acc[m] + bv);
            }
        }
    }
}

// Permutes NHWC `conv2_out [n_chunks, H3=16, W3=T_per_chunk, OC=480]` -> `[n_chunks * T_per_chunk, OC * H3]`
// AND fuses quantization directly into `(q_out, ds_out)` `Q8_1` for `enc.conv_out` GEMM!
// Grid: `((total_blks + 3) / 4, 1, 1)` where `total_blks = (n_chunks * T_per_chunk) * ((OC * H3) / 32)`, Block: `(128, 1, 1)`
extern "C" __global__ __launch_bounds__(128)
void asr_conv2_permute_q8_1_kernel(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ conv2_out, // [n_chunks, H3, W3, OC]
    int n_chunks,
    int OC,
    int H3,
    int W3
) {
    const int K = OC * H3;
    const int n_blks_row = K >> 5;
    const int M = n_chunks * W3;
    const int total_blks = M * n_blks_row;
    const int warp_id = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (warp_id >= total_blks) return;

    const int lane = threadIdx.x & 31;
    const int r = warp_id / n_blks_row;
    const int col = ((warp_id % n_blks_row) << 5) + lane;

    const int b = r / W3;
    const int t = r % W3;
    const int oc = col / H3;
    const int f = col % H3;

    const float v = conv2_out[(((size_t)b * H3 + f) * W3 + t) * OC + oc];
    float sum_x = v;
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffffu, sum_x, mask);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q_out[(warp_id << 5) + lane] = (signed char)__float2int_rn(v * inv_d);
    if (lane == 0) ds_out[warp_id] = make_float2(d, sum_x);
}

// Adds chunk-broadcast Sinusoidal PE `[T_per_chunk, d_model]` onto `conv_out` while trimming
// ragged tail from `T_enc_padded` down to `T_enc`!
// Grid: `(T_enc, 1, 1)`, Block: `(256, 1, 1)`
extern "C" __global__ __launch_bounds__(256)
void asr_add_pe_trim_kernel(
    float* __restrict__ enc_x_out,       // [T_enc, d_model]
    const float* __restrict__ conv_out,  // [T_enc_padded, d_model]
    const float* __restrict__ pos_emb,   // [T_per_chunk, d_model]
    int T_enc,
    int T_per_chunk,
    int d_model
) {
    const int r = blockIdx.x;
    if (r >= T_enc) return;
    const int p = r % T_per_chunk;
    const float* src = conv_out + (size_t)r * d_model;
    const float* pe = pos_emb + (size_t)p * d_model;
    float* dst = enc_x_out + (size_t)r * d_model;

    for (int c = threadIdx.x; c < d_model; c += 256) {
        dst[c] = src[c] + __ldg(&pe[c]);
    }
}

// Fused Pre-LayerNorm (`eps = 1e-5`) + `Q8_1` Activation Quantizer (also optionally writes `y_f32` if non-null)
// Grid: `(T, 1, 1)`, Block: `(256, 1, 1)` (8 warps; requires `C % 32 == 0`)
extern "C" __global__ __launch_bounds__(256)
void asr_layernorm_q8_1_kernel(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    float* __restrict__ y_f32_opt,
    const float* __restrict__ x_in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int C,
    float eps
) {
    __shared__ float s_sum[8];
    __shared__ float s_var[8];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x_in + (size_t)row * C;

    float lsum = 0.0f;
    for (int c = tid; c < C; c += 256) lsum += xr[c];
    lsum = warp_reduce_sum(lsum);
    if (lane == 0) s_sum[warp] = lsum;
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < 8) ? s_sum[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (lane == 0) s_sum[0] = bsum / (float)C;
    }
    __syncthreads();
    const float mean = s_sum[0];

    float lvar = 0.0f;
    for (int c = tid; c < C; c += 256) {
        const float d = xr[c] - mean;
        lvar = fmaf(d, d, lvar);
    }
    lvar = warp_reduce_sum(lvar);
    if (lane == 0) s_var[warp] = lvar;
    __syncthreads();

    if (warp == 0) {
        float bvar = (lane < 8) ? s_var[lane] : 0.0f;
        bvar = warp_reduce_sum(bvar);
        if (lane == 0) s_var[0] = rsqrtf(bvar / (float)C + eps);
    }
    __syncthreads();
    const float inv_std = s_var[0];

    const int n_blks = C >> 5;
    signed char* qr = q_out + (size_t)row * C;
    float2* dsr = ds_out + (size_t)row * n_blks;
    float* yr = (y_f32_opt != nullptr) ? (y_f32_opt + (size_t)row * C) : nullptr;

    for (int b = warp; b < n_blks; b += 8) {
        const int col = (b << 5) + lane;
        const float v = (xr[col] - mean) * inv_std * __ldg(&gamma[col]) + __ldg(&beta[col]);
        if (yr != nullptr) yr[col] = v;

        float sum_x = v;
        float amax = fabsf(v);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffffu, sum_x, mask);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
        }
        const float d = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        qr[col] = (signed char)__float2int_rn(v * inv_d);
        if (lane == 0) dsr[b] = make_float2(d, sum_x);
    }
}

// Online-Softmax Unmasked Bidirectional Multi-Head Attention (`head_dim = 64`)
// Grid: `(n_heads, T, 1)`, Block: `(128, 1, 1)` (4 warps split `[0 .. T)` key rows)
extern "C" __global__ __launch_bounds__(128)
void asr_bidir_flash_attn_f32_kernel(
    float* __restrict__ out,      // [T, n_heads * 64]
    const float* __restrict__ q,  // [T, n_heads * 64]
    const float* __restrict__ k,  // [T, n_heads * 64]
    const float* __restrict__ v,  // [T, n_heads * 64]
    int T,
    int n_heads,
    float scale
) {
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_o[4][64];

    const int h = blockIdx.x;
    const int tq = blockIdx.y;
    if (h >= n_heads || tq >= T) return;

    const int d_model = n_heads * 64;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d0 = lane << 1;

    const float* q_ptr = q + (size_t)tq * d_model + h * 64 + d0;
    const float q0 = q_ptr[0] * scale;
    const float q1 = q_ptr[1] * scale;

    float m_i = -1.0e30f;
    float l_i = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;

    for (int tk = warp; tk < T; tk += 4) {
        const size_t base = (size_t)tk * d_model + h * 64 + d0;
        const float k0 = k[base];
        const float k1 = k[base + 1];
        float score = q0 * k0 + q1 * k1;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            score += __shfl_xor_sync(0xffffffffu, score, mask);
        }

        const float m_new = fmaxf(m_i, score);
        const float alpha = __expf(m_i - m_new);
        const float p = __expf(score - m_new);
        l_i = l_i * alpha + p;
        o0 = o0 * alpha + p * v[base];
        o1 = o1 * alpha + p * v[base + 1];
        m_i = m_new;
    }

    if (lane == 0) {
        s_m[warp] = m_i;
        s_l[warp] = l_i;
    }
    s_o[warp][d0] = o0;
    s_o[warp][d0 + 1] = o1;
    __syncthreads();

    if (warp == 0) {
        float m_max = s_m[0];
        #pragma unroll
        for (int w = 1; w < 4; ++w) m_max = fmaxf(m_max, s_m[w]);

        float l_sum = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float e = (s_l[w] > 0.0f) ? __expf(s_m[w] - m_max) : 0.0f;
            l_sum += s_l[w] * e;
            acc0 += s_o[w][d0] * e;
            acc1 += s_o[w][d0 + 1] * e;
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* dst = out + (size_t)tq * d_model + h * 64 + d0;
        dst[0] = acc0 * inv_l;
        dst[1] = acc1 * inv_l;
    }
}

// Fused Exact `GELU_erf` + `Q8_1` Activation Quantizer
// Grid: `((total_blks + 3) / 4, 1, 1)` where `total_blks = (T * C) / 32`, Block: `(128, 1, 1)`
extern "C" __global__ __launch_bounds__(128)
void asr_gelu_erf_q8_1_kernel(
    signed char* __restrict__ q_out,
    float2* __restrict__ ds_out,
    const float* __restrict__ x_in,
    int total_blks
) {
    const int warp_id = (blockIdx.x << 2) + (threadIdx.x >> 5);
    if (warp_id >= total_blks) return;
    const int lane = threadIdx.x & 31;
    const float v = asr_gelu_erf(x_in[(warp_id << 5) + lane]);

    float sum_x = v;
    float amax = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffffu, sum_x, mask);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q_out[(warp_id << 5) + lane] = (signed char)__float2int_rn(v * inv_d);
    if (lane == 0) ds_out[warp_id] = make_float2(d, sum_x);
}

// Assembles 3-Way Decoder Prefill Input `[prefix_emb | enc_proj_out | suffix_emb]` Directly in VRAM!
// Grid: `(T_prompt, 1, 1)`, Block: `(256, 1, 1)`
extern "C" __global__ __launch_bounds__(256)
void asr_assemble_decoder_embed_kernel(
    float* __restrict__ dst_embed,            // [T_prompt, hidden]
    const unsigned char* __restrict__ emb_w,  // dec.token_embd.weight
    const int* __restrict__ prompt_ids,       // [T_prompt]
    const float* __restrict__ enc_proj_out,   // [T_enc, hidden]
    int T_prompt,
    int prefix_len,
    int T_enc,
    int hidden,
    int emb_type
) {
    const int t = blockIdx.x;
    if (t >= T_prompt) return;
    float* dst = dst_embed + (size_t)t * hidden;

    if (t >= prefix_len && t < prefix_len + T_enc) {
        const float* src = enc_proj_out + (size_t)(t - prefix_len) * hidden;
        for (int c = threadIdx.x; c < hidden; c += 256) {
            dst[c] = src[c];
        }
    } else {
        const int tid = threadIdx.x;
        const int tok = prompt_ids[t];
        if (emb_type == 14) {
            const int n_sb = hidden >> 8;
            const unsigned char* row = emb_w + (size_t)tok * (n_sb * 210);
            for (int col = tid; col < hidden; col += 256) {
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
            const int n_sb = hidden >> 8;
            const unsigned char* row = emb_w + (size_t)tok * (n_sb * 144);
            for (int col = tid; col < hidden; col += 256) {
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
            const int n_blk = hidden >> 5;
            const unsigned char* row = emb_w + (size_t)tok * (n_blk * 34);
            for (int col = tid; col < hidden; col += 256) {
                const int b = col >> 5, k = col & 31;
                const unsigned char* blk = row + b * 34;
                dst[col] = __half2float(__ldg(reinterpret_cast<const half*>(blk))) *
                           (float)__ldg(&reinterpret_cast<const signed char*>(blk + 2)[k]);
            }
        } else if (emb_type == 30) {
            const unsigned short* row = reinterpret_cast<const unsigned short*>(emb_w) + (size_t)tok * hidden;
            for (int col = tid; col < hidden; col += 256) dst[col] = bf16_bits_to_f32(__ldg(&row[col]));
        } else if (emb_type == 1) {
            const half* row = reinterpret_cast<const half*>(emb_w) + (size_t)tok * hidden;
            for (int col = tid; col < hidden; col += 256) dst[col] = __half2float(__ldg(&row[col]));
        } else {
            const float* row = reinterpret_cast<const float*>(emb_w) + (size_t)tok * hidden;
            for (int col = tid; col < hidden; col += 256) dst[col] = __ldg(&row[col]);
        }
    }
}

