// =============================================================================
// Custom GA102 (RTX 3080, sm_86, 68 SMs) Native CUDA Kernels for
// Thinking Machines `Inkling-2-Mini` (`inkling2_loop`, 26.4B Looped Sparse MoE:
// 24 physical layers -> 48 logical passes, 128 experts top-8 SITU, ShortConv,
// Relative-Position + Sink GQA, and Depth Attention Residuals).
//
// Optimized v2:
// - Shared-memory LUTs (`s_iq2xxs_grid`, `s_ksigns_iq2xs`, `s_iq4nl`) eliminating
//   divergent `__constant__` cache warp serialization on Ampere `sm_86`.
// - Superblock-coalesced warp dot products for `IQ2_XXS` (256 weights/warp/iter,
//   fused dual `gate`+`up` pass), `Q2_0` (128 weights/warp/iter), `IQ4_NL` (64/iter),
//   `Q8_0` (128/iter), and `Q6_K` (256/iter).
// - Zero-overhead VRAM-resident `QMDv3` graph support via `InkStepCtrl` (eliminates
//   1,550 sequential Thunderbolt PCIe QMD round-trips per token).
//
// Precompiled ahead-of-time to `inkling_sm86.cubin` and embedded via `include_bytes!`
// (0 runtime Docker containers, 0 external Rust crates, 100% RTX 3080 compute).
// =============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

struct InkStepCtrl {
    int32_t  tok_id;      // offset 0
    int32_t  pos;         // offset 4
    float    temperature; // offset 8
    uint32_t step;        // offset 12
    uint64_t rng_seed;    // offset 16
    uint64_t done_seq;    // offset 24
};

// -----------------------------------------------------------------------------
// Official GGML Lookup Tables (`IQ2_XXS` & `IQ4_NL`) — staged into `__shared__`
// collaboratively at CTA startup so random 32-lane lookups are conflict-light
// instead of serializing on `__constant__` broadcast ports.
// -----------------------------------------------------------------------------

__constant__ uint64_t c_iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL
};

__constant__ uint8_t c_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

__constant__ int8_t c_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// -----------------------------------------------------------------------------
// Warp Reductions & Quantized Primitives
// -----------------------------------------------------------------------------

__device__ __forceinline__ float ink_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float ink_dequant_elem(
    const uint8_t* w_raw,
    int w_type,
    size_t row_idx,
    int k_cols,
    int col
) {
    if (w_type == 0) { // F32
        const float* row = reinterpret_cast<const float*>(w_raw) + row_idx * (size_t)k_cols;
        return __ldg(&row[col]);
    } else if (w_type == 1) { // F16
        const half* row = reinterpret_cast<const half*>(w_raw) + row_idx * (size_t)k_cols;
        return __half2float(__ldg(&row[col]));
    } else if (w_type == 30) { // BF16
        const uint16_t* row = reinterpret_cast<const uint16_t*>(w_raw) + row_idx * (size_t)k_cols;
        uint32_t bits = ((uint32_t)__ldg(&row[col])) << 16;
        return __uint_as_float(bits);
    } else if (w_type == 8) { // Q8_0 (34 B / 32)
        const int n_blk = k_cols >> 5;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_blk + (size_t)(col >> 5)) * 34ULL;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
        const int q = (int)reinterpret_cast<const int8_t*>(blk + 2)[col & 31];
        return d * (float)q;
    } else if (w_type == 42) { // Q2_0 (18 B / 64)
        const int n_blk = k_cols >> 6;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_blk + (size_t)(col >> 6)) * 18ULL;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
        const int j = col & 63;
        const uint8_t qb = __ldg(&blk[2 + (j >> 2)]);
        const int q = ((qb >> ((j & 3) << 1)) & 3) - 1;
        return d * (float)q;
    } else if (w_type == 20) { // IQ4_NL (18 B / 32)
        const int n_blk = k_cols >> 5;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_blk + (size_t)(col >> 5)) * 18ULL;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
        const int k = col & 31;
        const uint8_t qb = __ldg(&blk[2 + (k & 15)]);
        const int nib = (k < 16) ? (qb & 0x0f) : (qb >> 4);
        return d * (float)c_kvalues_iq4nl[nib];
    } else if (w_type == 16) { // IQ2_XXS (66 B / 256)
        const int n_sb = k_cols >> 8;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_sb + (size_t)(col >> 8)) * 66ULL;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk + 2);
        const int ib32 = (col >> 5) & 7;
        const uint32_t lo = (uint32_t)__ldg(&qs16[4 * ib32 + 0]) | ((uint32_t)__ldg(&qs16[4 * ib32 + 1]) << 16);
        const uint32_t hi = (uint32_t)__ldg(&qs16[4 * ib32 + 2]) | ((uint32_t)__ldg(&qs16[4 * ib32 + 3]) << 16);
        const float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
        const int idx32 = col & 31;
        const int l = idx32 >> 3;
        const int j = idx32 & 7;
        const uint8_t u = (lo >> (8 * l)) & 0xffu;
        const uint64_t g64 = c_iq2xxs_grid[u];
        const float gj = (float)((g64 >> (8 * j)) & 0xffu);
        const uint8_t signs = c_ksigns_iq2xs[(hi >> (7 * l)) & 127u];
        const float sgn = (signs & (1u << j)) ? -1.0f : 1.0f;
        return db * gj * sgn;
    } else if (w_type == 14) { // Q6_K (210 B / 256)
        const int n_sb = k_cols >> 8;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_sb + (size_t)(col >> 8)) * 210ULL;
        const int idx = col & 255;
        const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208)));
        const int sc = (int)__ldg(&reinterpret_cast<const int8_t*>(blk + 192)[idx >> 4]);
        const int h = idx >> 7, rem = idx & 127, p = rem >> 6, q = (rem >> 5) & 1, k = rem & 31;
        const uint8_t ql_byte = __ldg(&blk[(h << 6) + (q << 5) + k]);
        const uint8_t qh_byte = __ldg(&blk[128 + (h << 5) + k]);
        const int xl = (ql_byte >> (p << 2)) & 0x0f;
        const int xh = ((qh_byte >> (((p << 1) + q) << 1)) & 0x03) << 4;
        return d * (float)sc * (float)((xl | xh) - 32);
    } else if (w_type == 12) { // Q4_K (144 B / 256)
        const int n_sb = k_cols >> 8;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_sb + (size_t)(col >> 8)) * 144ULL;
        const int idx = col & 255;
        const half2 dm = __ldg(reinterpret_cast<const half2*>(blk));
        const float d = __half2float(dm.x), dmin = __half2float(dm.y);
        const uint8_t* s = blk + 4;
        const int sub = idx >> 5, k = idx & 31;
        uint32_t sc, mn;
        if (sub < 4) {
            sc = __ldg(&s[sub]) & 63u;
            mn = __ldg(&s[sub + 4]) & 63u;
        } else {
            sc = (__ldg(&s[sub + 4]) & 0x0fu) | ((__ldg(&s[sub - 4]) >> 6) << 4);
            mn = (__ldg(&s[sub + 4]) >> 4)    | ((__ldg(&s[sub])     >> 6) << 4);
        }
        const uint8_t qbyte = __ldg(&blk[16 + ((sub >> 1) << 5) + k]);
        const uint32_t q = (sub & 1) ? (qbyte >> 4) : (qbyte & 0x0fu);
        return d * (float)sc * (float)q - dmin * (float)mn;
    }
    return 0.0f;
}

__device__ __forceinline__ float ink_warp_dot_dense(
    const uint8_t* w_raw,
    int w_type,
    size_t row_idx,
    const float* s_x,
    int k_cols,
    int lane
) {
    float sum = 0.0f;
    if (w_type == 8 && (k_cols & 127) == 0) {
        // Fast coalesced Q8_0 path: 4 blocks (128 weights) per warp iteration!
        const int n_blk = k_cols >> 5;
        const uint8_t* row_ptr = w_raw + row_idx * (size_t)n_blk * 34ULL;
        const int b_sub = lane >> 3;  // 0..3
        const int sub = lane & 7;     // 0..7
        const int n_quads = n_blk >> 2;
        for (int qd = 0; qd < n_quads; ++qd) {
            const int bi = (qd << 2) + b_sub;
            const uint8_t* blk = row_ptr + (size_t)bi * 34ULL;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk + 2 + (sub << 2));
            const uint16_t u0 = __ldg(&q16[0]);
            const uint16_t u1 = __ldg(&q16[1]);
            const int8_t q0 = (int8_t)(u0 & 0xffu);
            const int8_t q1 = (int8_t)(u0 >> 8);
            const int8_t q2 = (int8_t)(u1 & 0xffu);
            const int8_t q3 = (int8_t)(u1 >> 8);
            const float* xp = s_x + (bi << 5) + (sub << 2);
            sum += d * ((float)q0 * xp[0] + (float)q1 * xp[1] + (float)q2 * xp[2] + (float)q3 * xp[3]);
        }
        return ink_warp_sum(sum);
    } else if (w_type == 8 && (k_cols & 63) == 0) {
        // Coalesced Q8_0 path for k_cols == 64 (ctrl_gain / ctrl_forget): 2 blocks (64 weights) per iter!
        const int n_blk = k_cols >> 5;
        const uint8_t* row_ptr = w_raw + row_idx * (size_t)n_blk * 34ULL;
        const int b_sub = lane >> 4;  // 0..1
        const int sub = lane & 15;    // 0..15
        const int n_pairs = n_blk >> 1;
        for (int pr = 0; pr < n_pairs; ++pr) {
            const int bi = (pr << 1) + b_sub;
            const uint8_t* blk = row_ptr + (size_t)bi * 34ULL;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
            const uint16_t u0 = __ldg(reinterpret_cast<const uint16_t*>(blk + 2 + (sub << 1)));
            const int8_t q0 = (int8_t)(u0 & 0xffu);
            const int8_t q1 = (int8_t)(u0 >> 8);
            const float* xp = s_x + (bi << 5) + (sub << 1);
            sum += d * ((float)q0 * xp[0] + (float)q1 * xp[1]);
        }
        return ink_warp_sum(sum);
    } else if (w_type == 14 && (k_cols & 255) == 0) {
        // Fast coalesced Q6_K path (`output.weight` [2304, 200058]): 1 superblock (256 weights) per iter!
        const int n_sb = k_cols >> 8;
        const uint8_t* row_ptr = w_raw + row_idx * (size_t)n_sb * 210ULL;
        const int half_lane = lane >> 4; // 0..1
        for (int sb = 0; sb < n_sb; ++sb) {
            const uint8_t* blk = row_ptr + (size_t)sb * 210ULL;
            const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk + 208)));
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            const uint8_t ql0 = __ldg(&blk[0 + lane]);
            const uint8_t ql1 = __ldg(&blk[32 + lane]);
            const uint8_t ql2 = __ldg(&blk[64 + lane]);
            const uint8_t ql3 = __ldg(&blk[96 + lane]);
            const uint8_t qh0 = __ldg(&blk[128 + lane]);
            const uint8_t qh1 = __ldg(&blk[160 + lane]);
            const float* xp = s_x + (sb << 8) + lane;

            const int v0 = (int)((ql0 & 0x0fu) | (((qh0 >> 0) & 3u) << 4)) - 32;
            const int v1 = (int)((ql1 & 0x0fu) | (((qh0 >> 2) & 3u) << 4)) - 32;
            const int v2 = (int)((ql0 >> 4)    | (((qh0 >> 4) & 3u) << 4)) - 32;
            const int v3 = (int)((ql1 >> 4)    | (((qh0 >> 6) & 3u) << 4)) - 32;
            const int v4 = (int)((ql2 & 0x0fu) | (((qh1 >> 0) & 3u) << 4)) - 32;
            const int v5 = (int)((ql3 & 0x0fu) | (((qh1 >> 2) & 3u) << 4)) - 32;
            const int v6 = (int)((ql2 >> 4)    | (((qh1 >> 4) & 3u) << 4)) - 32;
            const int v7 = (int)((ql3 >> 4)    | (((qh1 >> 6) & 3u) << 4)) - 32;

            float sb_acc = (float)(int)__ldg(&sc[0  + half_lane]) * (float)v0 * xp[0]
                         + (float)(int)__ldg(&sc[2  + half_lane]) * (float)v1 * xp[32]
                         + (float)(int)__ldg(&sc[4  + half_lane]) * (float)v2 * xp[64]
                         + (float)(int)__ldg(&sc[6  + half_lane]) * (float)v3 * xp[96]
                         + (float)(int)__ldg(&sc[8  + half_lane]) * (float)v4 * xp[128]
                         + (float)(int)__ldg(&sc[10 + half_lane]) * (float)v5 * xp[160]
                         + (float)(int)__ldg(&sc[12 + half_lane]) * (float)v6 * xp[192]
                         + (float)(int)__ldg(&sc[14 + half_lane]) * (float)v7 * xp[224];
            sum += d * sb_acc;
        }
        return ink_warp_sum(sum);
    } else if (w_type == 0) {
        const float* row = reinterpret_cast<const float*>(w_raw) + row_idx * (size_t)k_cols;
        for (int col = lane; col < k_cols; col += 32) {
            sum += __ldg(&row[col]) * s_x[col];
        }
        return ink_warp_sum(sum);
    }

    for (int col = lane; col < k_cols; col += 32) {
        sum += ink_dequant_elem(w_raw, w_type, row_idx, k_cols, col) * s_x[col];
    }
    return ink_warp_sum(sum);
}

// -----------------------------------------------------------------------------
// 1. Latch `InkStepCtrl` into GDDR6X L2 + Token Embedding Lookup + `tok_norm`
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_embed_rmsnorm_kernel(
    float* out_f32,
    InkStepCtrl* ctrl_vram_out,
    const InkStepCtrl* ctrl_host_in,
    const uint8_t* emb_w,
    const float* norm_w,
    int32_t dim,
    int32_t emb_type,
    float eps
) {
    __shared__ float s_x[2304];
    __shared__ float s_red[8];
    __shared__ int32_t s_tok_id;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (tid == 0) {
        const InkStepCtrl c = *ctrl_host_in;
        *ctrl_vram_out = c;
        s_tok_id = c.tok_id;
    }
    __syncthreads();
    const int32_t tok_id = s_tok_id;

    float ss = 0.0f;
    for (int col = tid; col < dim; col += 256) {
        float v = ink_dequant_elem(emb_w, emb_type, (size_t)tok_id, dim, col);
        s_x[col] = v;
        ss += v * v;
    }
    ss = ink_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();

    if (wid == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = ink_warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();

    const float scale = s_red[0];
    for (int col = tid; col < dim; col += 256) {
        float w = (norm_w != nullptr) ? __ldg(&norm_w[col]) : 1.0f;
        out_f32[col] = s_x[col] * scale * w;
    }
}

// -----------------------------------------------------------------------------
// 2. General RMSNorm (`[dim]`, optional `norm_w`)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_rmsnorm_kernel(
    float* out_f32,
    const float* in_f32,
    const float* norm_w,
    int32_t dim,
    float eps
) {
    __shared__ float s_x[2304];
    __shared__ float s_red[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    float ss = 0.0f;
    for (int col = tid; col < dim; col += 256) {
        float v = in_f32[col];
        s_x[col] = v;
        ss += v * v;
    }
    ss = ink_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();

    if (wid == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = ink_warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();

    const float scale = s_red[0];
    for (int col = tid; col < dim; col += 256) {
        float w = (norm_w != nullptr) ? __ldg(&norm_w[col]) : 1.0f;
        out_f32[col] = s_x[col] * scale * w;
    }
}

// -----------------------------------------------------------------------------
// 3. Depth Attention Residuals (`mix(blocks, cur, query)`, `n_blocks <= 8`, `n_queries <= 4`)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_depth_mix_kernel(
    float* out_f32,         // [n_queries, dim]
    const float* blocks_f32,// [n_blocks, dim]
    const float* cur_f32,   // [dim]
    const float* query_f32, // [n_queries, dim] (GGUF ne=[dim, n_queries])
    int32_t dim,
    int32_t n_blocks,
    int32_t n_queries,
    float eps
) {
    __shared__ float s_red[8];
    __shared__ float s_inv_rms[10];
    __shared__ float s_scores[10 * 4];
    __shared__ float s_probs[10 * 4];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int n_states = n_blocks + 1;

    // 1. Parameterless RMSNorm scale for each depth state `m in 0..n_states`
    for (int m = 0; m < n_states; ++m) {
        const float* st = (m < n_blocks) ? (blocks_f32 + (size_t)m * dim) : cur_f32;
        float ss = 0.0f;
        for (int col = tid; col < dim; col += 256) {
            float v = st[col];
            ss += v * v;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < 8) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_inv_rms[m] = rsqrtf(tot / (float)dim + eps);
        }
        __syncthreads();
    }

    // 2. Dot product `(rms_norm(state_m) . query_q) / sqrt(dim)`
    const float inv_sqrt_d = rsqrtf((float)dim);
    for (int m = 0; m < n_states; ++m) {
        const float* st = (m < n_blocks) ? (blocks_f32 + (size_t)m * dim) : cur_f32;
        const float irms = s_inv_rms[m];
        for (int q = 0; q < n_queries; ++q) {
            const float* qv = query_f32 + (size_t)q * dim;
            float dot = 0.0f;
            for (int col = tid; col < dim; col += 256) {
                dot += (st[col] * irms) * __ldg(&qv[col]);
            }
            dot = ink_warp_sum(dot);
            if (lane == 0) s_red[wid] = dot;
            __syncthreads();
            if (wid == 0) {
                float tot = (lane < 8) ? s_red[lane] : 0.0f;
                tot = ink_warp_sum(tot);
                if (lane == 0) s_scores[q * 10 + m] = tot * inv_sqrt_d;
            }
            __syncthreads();
        }
    }

    // 3. Softmax over `m in 0..n_states` per query `q`
    if (tid < n_queries) {
        const int q = tid;
        float mx = -1e30f;
        for (int m = 0; m < n_states; ++m) {
            mx = fmaxf(mx, s_scores[q * 10 + m]);
        }
        float denom = 0.0f;
        for (int m = 0; m < n_states; ++m) {
            float p = expf(s_scores[q * 10 + m] - mx);
            s_probs[q * 10 + m] = p;
            denom += p;
        }
        float inv_den = 1.0f / fmaxf(denom, 1e-20f);
        for (int m = 0; m < n_states; ++m) {
            s_probs[q * 10 + m] *= inv_den;
        }
    }
    __syncthreads();

    // 4. Weighted sum of normalized states
    for (int q = 0; q < n_queries; ++q) {
        for (int col = tid; col < dim; col += 256) {
            float acc = 0.0f;
            for (int m = 0; m < n_states; ++m) {
                const float* st = (m < n_blocks) ? (blocks_f32 + (size_t)m * dim) : cur_f32;
                acc += s_probs[q * 10 + m] * (st[col] * s_inv_rms[m]);
            }
            out_f32[(size_t)q * dim + col] = acc;
        }
    }
}

// -----------------------------------------------------------------------------
// 4. Loop Controller Input Prep (`concat(rms_norm(cur), rms_norm(msg))` -> `[2*dim]`)
//    & Gated Apply (`(1 - tanh(forget))*cur + tanh(gain)*msg`)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_loop_ctrl_prep_kernel(
    float* cat_out,         // [2 * dim]
    const float* cur_in,    // [dim]
    const float* msg_in,    // [dim]
    int32_t dim,
    float eps
) {
    __shared__ float s_red_c[8];
    __shared__ float s_red_m[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    float ssc = 0.0f, ssm = 0.0f;
    for (int col = tid; col < dim; col += 256) {
        float c = cur_in[col];
        float m = msg_in[col];
        ssc += c * c;
        ssm += m * m;
    }
    ssc = ink_warp_sum(ssc);
    ssm = ink_warp_sum(ssm);
    if (lane == 0) {
        s_red_c[wid] = ssc;
        s_red_m[wid] = ssm;
    }
    __syncthreads();

    if (wid == 0) {
        float tc = (lane < 8) ? s_red_c[lane] : 0.0f;
        float tm = (lane < 8) ? s_red_m[lane] : 0.0f;
        tc = ink_warp_sum(tc);
        tm = ink_warp_sum(tm);
        if (lane == 0) {
            s_red_c[0] = rsqrtf(tc / (float)dim + eps);
            s_red_m[0] = rsqrtf(tm / (float)dim + eps);
        }
    }
    __syncthreads();

    const float sc = s_red_c[0];
    const float sm = s_red_m[0];
    for (int col = tid; col < dim; col += 256) {
        cat_out[col]       = cur_in[col] * sc;
        cat_out[dim + col] = msg_in[col] * sm;
    }
}

extern "C" __global__ __launch_bounds__(256)
void ink_loop_ctrl_apply_kernel(
    float* out_f32,
    const float* cur_in,
    const float* msg_in,
    const float* gain_raw,
    const float* forget_raw,
    int32_t dim
) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < dim) {
        const float c = cur_in[idx];
        const float m = msg_in[idx];
        const float g = tanhf(gain_raw[idx]);
        const float f = tanhf(forget_raw[idx]);
        out_f32[idx] = (1.0f - f) * c + g * m;
    }
}

// -----------------------------------------------------------------------------
// 5. Warp-Parallel Dense 2D Quantized GEMV (`k_cols <= 4608`, `8` rows/CTA)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_gemv_dense_kernel(
    float* y_out,
    const uint8_t* w_raw,
    const float* x_in,
    int32_t k_cols,
    int32_t n_rows,
    int32_t w_type,
    float softcap
) {
    __shared__ float s_x[4608];

    const int tid = threadIdx.x;
    for (int i = tid; i < k_cols; i += 256) {
        s_x[i] = x_in[i];
    }
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5; // 0..7
    const int row = blockIdx.x * 8 + wid;
    if (row >= n_rows) return;

    float val = ink_warp_dot_dense(w_raw, w_type, (size_t)row, s_x, k_cols, lane);
    if (lane == 0) {
        if (softcap > 0.0f) {
            val = softcap * tanhf(val / softcap);
        }
        y_out[row] = val;
    }
}

// -----------------------------------------------------------------------------
// 6. Causal Short Convolution (`n_conv = 4`, recurrent `[n_channels, 3]` state,
//    optional per-head dynamic taps `dyn_in [n_heads_dyn, 4]`, residual + 1/sqrt scale)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_shortconv_kernel(
    float* out_f32,
    const float* cur_in,
    const float* conv_w,    // [n_channels, 4] (GGUF ne=[4, n_channels])
    const float* dyn_in,    // nullptr if static, else [n_heads_dyn * 4] (ne=[4, n_heads_dyn])
    float* conv_state,      // [n_channels, 3]
    int32_t n_channels,
    int32_t n_heads_dyn
) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n_channels) return;

    float* st = conv_state + (size_t)c * 3;
    const float x0 = st[0];
    const float x1 = st[1];
    const float x2 = st[2];
    const float x3 = cur_in[c];

    st[0] = x1;
    st[1] = x2;
    st[2] = x3;

    const float* w = conv_w + (size_t)c * 4;
    float sum = x0 * __ldg(&w[0]) + x1 * __ldg(&w[1]) + x2 * __ldg(&w[2]) + x3 * __ldg(&w[3]);

    float inv_scale = 0.4472135955f; // 1 / sqrt(5)
    if (dyn_in != nullptr && n_heads_dyn > 0) {
        const int ch_per_head = n_channels / n_heads_dyn;
        const int h = c / ch_per_head;
        const float* d = dyn_in + h * 4;
        // lag=0 -> x3, lag=1 -> x2, lag=2 -> x1, lag=3 -> x0
        sum += d[0] * x3 + d[1] * x2 + d[2] * x1 + d[3] * x0;
        inv_scale = 0.3333333333f; // 1 / sqrt(9)
    }

    out_f32[c] = (x3 + sum) * inv_scale;
}

// -----------------------------------------------------------------------------
// 7. Q/K/V/R L2-Clip & Head-RMSNorm + FP16 KV Cache Write (`grid = (40, 1, 1)`, `block = (128, 1, 1)`)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128)
void ink_qkv_prep_kvstore_kernel(
    float* q_inout,             // [24 * 128]
    float* r_inout,             // [24 * 16]
    const float* k_in,          // [8 * 128]
    const float* v_in,          // [8 * 128]
    const float* q_clip_w,      // [24 * 128]
    const float* rel_clip_w,    // [24 * 16]
    half* k_cache_slot,         // [kv_cap, 1024] for this layer/pass slot
    half* v_cache_slot,         // [kv_cap, 1024] for this layer/pass slot
    const InkStepCtrl* ctrl,
    int32_t swa_win,            // 1024
    int32_t is_swa,
    float qk_clip,
    float rel_clip,
    float rms_eps,
    int32_t log_floor,
    float log_alpha
) {
    __shared__ float s_red[4];

    const int b = blockIdx.x;
    const int tid = threadIdx.x; // 0..127
    const int lane = tid & 31;
    const int wid = tid >> 5;    // 0..3
    const int32_t pos = ctrl->pos;
    const int32_t cache_write_idx = is_swa ? (pos & (swa_win - 1)) : pos;

    float tau = 1.0f;
    if (!is_swa) {
        float ratio = fmaxf(1.0f, (float)(pos + 1) / (float)log_floor);
        tau = 1.0f + log_alpha * logf(ratio);
    }

    if (b < 24) {
        const int h = b;
        float qv = q_inout[h * 128 + tid] * __ldg(&q_clip_w[h * 128 + tid]);
        float ss = ink_warp_sum(qv * qv);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < 4) ? s_red[lane] : 0.0f;
            tot += __shfl_down_sync(0xffffffff, tot, 2);
            tot += __shfl_down_sync(0xffffffff, tot, 1);
            if (lane == 0) {
                float nrm = sqrtf(tot) + 1e-6f;
                float fac = fminf(1.0f, fmaxf(0.0f, qk_clip / nrm));
                s_red[0] = fac * tau;
            }
        }
        __syncthreads();
        q_inout[h * 128 + tid] = qv * s_red[0];

        if (wid == 0) {
            float rv = (lane < 16) ? (r_inout[h * 16 + lane] * __ldg(&rel_clip_w[h * 16 + lane])) : 0.0f;
            float rss = rv * rv;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                rss += __shfl_xor_sync(0xffffffff, rss, off);
            }
            float rnrm = sqrtf(rss) + 1e-6f;
            float rfac = fminf(1.0f, fmaxf(0.0f, rel_clip / rnrm)) * tau;
            if (lane < 16) {
                r_inout[h * 16 + lane] = rv * rfac;
            }
        }
    } else if (b < 32) {
        const int kv_h = b - 24;
        float kv = k_in[kv_h * 128 + tid];
        float ss = ink_warp_sum(kv * kv);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < 4) ? s_red[lane] : 0.0f;
            tot += __shfl_down_sync(0xffffffff, tot, 2);
            tot += __shfl_down_sync(0xffffffff, tot, 1);
            if (lane == 0) {
                float nrm = sqrtf(tot) + 1e-6f;
                s_red[0] = fminf(1.0f, fmaxf(0.0f, qk_clip / nrm));
            }
        }
        __syncthreads();
        k_cache_slot[(size_t)cache_write_idx * 1024 + kv_h * 128 + tid] = __float2half(kv * s_red[0]);
    } else if (b < 40) {
        const int kv_h = b - 32;
        float vv = v_in[kv_h * 128 + tid];
        float ss = ink_warp_sum(vv * vv);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < 4) ? s_red[lane] : 0.0f;
            tot += __shfl_down_sync(0xffffffff, tot, 2);
            tot += __shfl_down_sync(0xffffffff, tot, 1);
            if (lane == 0) {
                s_red[0] = rsqrtf(tot * (1.0f / 128.0f) + rms_eps);
            }
        }
        __syncthreads();
        v_cache_slot[(size_t)cache_write_idx * 1024 + kv_h * 128 + tid] = __float2half(vv * s_red[0]);
    }
}

// -----------------------------------------------------------------------------
// 8. Relative-Position + Virtual Zero-Sink GQA Attention + Output Sigmoid Gate
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128)
void ink_rel_sink_attn_gate_kernel(
    float* attn_gated_out,      // [3072]
    const float* q_in,          // [3072]
    const float* r_in,          // [24 * 16]
    const float* sinks_in,      // [24]
    const float* gate_raw_in,   // [3072]
    const float* rel_bias_w,    // [1024 * 16] (GGUF ne=[16, 1024])
    const half* k_cache_slot,   // [kv_cap, 1024]
    const half* v_cache_slot,   // [kv_cap, 1024]
    const InkStepCtrl* ctrl,
    int32_t is_swa,
    int32_t swa_win,            // 1024
    int32_t rel_extent          // 1024
) {
    __shared__ float s_q[128];
    __shared__ float s_r[16];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];

    const int h = blockIdx.x;       // 0..23
    const int kv_h = h / 3;         // 0..7
    const int tid = threadIdx.x;    // 0..127
    const int lane = tid & 31;      // 0..31
    const int wid = tid >> 5;       // 0..3
    const int32_t pos = ctrl->pos;

    s_q[tid] = q_in[h * 128 + tid];
    if (tid < 16) {
        s_r[tid] = r_in[h * 16 + tid];
    }
    const float sink_h = sinks_in[h];
    __syncthreads();

    const int start_j = is_swa ? max(0, pos - swa_win + 1) : 0;
    const float inv_sqrt_hd = 0.0883883476f; // 1 / sqrt(128)

    float m_w = (wid == 0) ? 0.0f : -1e30f;
    float l_w = (wid == 0) ? 1.0f : 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int d0 = lane * 4;
    const float q0 = s_q[d0 + 0];
    const float q1 = s_q[d0 + 1];
    const float q2 = s_q[d0 + 2];
    const float q3 = s_q[d0 + 3];

    for (int j = start_j + wid; j <= pos; j += 4) {
        const int slot_j = is_swa ? (j & (swa_win - 1)) : j;
        const half2* kj2 = reinterpret_cast<const half2*>(k_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
        const half2 k01 = __ldg(&kj2[0]);
        const half2 k23 = __ldg(&kj2[1]);
        float dot = q0 * __half2float(k01.x)
                  + q1 * __half2float(k01.y)
                  + q2 * __half2float(k23.x)
                  + q3 * __half2float(k23.y);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        }

        const int dist = pos - j;
        float r_dot = 0.0f;
        if (dist < rel_extent) {
            float rp = (lane < 16) ? (s_r[lane] * __ldg(&rel_bias_w[(size_t)dist * 16 + lane])) : 0.0f;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                rp += __shfl_xor_sync(0xffffffff, rp, off);
            }
            r_dot = __shfl_sync(0xffffffff, rp, 0);
        }

        const float score = dot * inv_sqrt_hd + (r_dot - sink_h);
        const float m_new = fmaxf(m_w, score);
        const float alpha = expf(m_w - m_new);
        const float beta  = expf(score - m_new);
        l_w = l_w * alpha + beta;
        m_w = m_new;

        const half2* vj2 = reinterpret_cast<const half2*>(v_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
        const half2 v01 = __ldg(&vj2[0]);
        const half2 v23 = __ldg(&vj2[1]);
        acc0 = acc0 * alpha + beta * __half2float(v01.x);
        acc1 = acc1 * alpha + beta * __half2float(v01.y);
        acc2 = acc2 * alpha + beta * __half2float(v23.x);
        acc3 = acc3 * alpha + beta * __half2float(v23.y);
    }

    if (lane == 0) {
        s_m[wid] = m_w;
        s_l[wid] = l_w;
    }
    s_acc[wid][d0 + 0] = acc0;
    s_acc[wid][d0 + 1] = acc1;
    s_acc[wid][d0 + 2] = acc2;
    s_acc[wid][d0 + 3] = acc3;
    __syncthreads();

    float m_max = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
    float denom = 0.0f;
    float out_d = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        float ew = (s_l[w] > 0.0f) ? expf(s_m[w] - m_max) : 0.0f;
        denom += s_l[w] * ew;
        out_d += s_acc[w][tid] * ew;
    }
    const float val = out_d / fmaxf(denom, 1e-20f);
    const float gate = 1.0f / (1.0f + expf(-gate_raw_in[h * 128 + tid]));
    attn_gated_out[h * 128 + tid] = val * gate;
}

// -----------------------------------------------------------------------------
// 9. Sparse MoE Sigmoid Top-8 Router (`128` experts -> `top_ids[8]`, `top_weights[8]`)
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128)
void ink_moe_router_topk8_kernel(
    int32_t* top_ids,           // [8]
    float* top_weights,         // [8]
    const float* router_logits, // [128]
    const float* exp_probs_b,   // [128] (optional bias for selection)
    float weights_scale         // 8.0
) {
    __shared__ float s_prob[128];
    __shared__ float s_sel[128];

    const int e = threadIdx.x; // 0..127
    const float p = 1.0f / (1.0f + expf(-router_logits[e]));
    const float b = (exp_probs_b != nullptr) ? __ldg(&exp_probs_b[e]) : 0.0f;
    s_prob[e] = p;
    s_sel[e]  = p + b;
    __syncthreads();

    if (e == 0) {
        int chosen[8];
        float w_sum = 1e-20f;
        for (int k = 0; k < 8; ++k) {
            int best_idx = 0;
            float best_val = -1e30f;
            for (int i = 0; i < 128; ++i) {
                if (s_sel[i] > best_val) {
                    best_val = s_sel[i];
                    best_idx = i;
                }
            }
            chosen[k] = best_idx;
            s_sel[best_idx] = -1e31f;
            w_sum += s_prob[best_idx];
        }
        const float inv_sum = weights_scale / w_sum;
        for (int k = 0; k < 8; ++k) {
            int id = chosen[k];
            top_ids[k]     = id;
            top_weights[k] = s_prob[id] * inv_sum;
        }
    }
}

// -----------------------------------------------------------------------------
// 10. Sparse MoE Active-8 `ffn_gate_up_exps` + `SITU` Activation
//     (`grid = (144, 8, 1)`, `block = (256, 1, 1)` = 8 warps/CTA, `n_ff = 1152`)
//     Fuses `gate` and `up` superblock passes against shared `s_x` + shared `IQ2_XXS` LUT!
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_moe_gate_up_situ_kernel(
    float* situ_out,            // [8, n_ff]
    const uint8_t* gate_up_w,   // [dim, 2*n_ff, 128]
    const float* ffn_in,        // [dim]
    const int32_t* top_ids,     // [8]
    int32_t dim,                // 2304
    int32_t n_ff,               // 1152
    int32_t w_type,             // 16 (IQ2_XXS)
    float situ_beta,            // 4.0
    float situ_lb               // 25.0
) {
    __shared__ uint64_t s_iq2xxs_grid[256];
    __shared__ uint8_t  s_ksigns_iq2xs[128];
    __shared__ float    s_x[2304];

    const int tid = threadIdx.x; // 0..255
    s_iq2xxs_grid[tid] = c_iq2xxs_grid[tid];
    if (tid < 128) {
        s_ksigns_iq2xs[tid] = c_ksigns_iq2xs[tid];
    }
    for (int i = tid; i < dim; i += 256) {
        s_x[i] = ffn_in[i];
    }
    __syncthreads();

    const int slot = blockIdx.y; // 0..7
    const int wid = tid >> 5;    // 0..7
    const int lane = tid & 31;   // 0..31
    const int ff_idx = blockIdx.x * 8 + wid;
    if (ff_idx >= n_ff) return;

    const int exp_id = top_ids[slot];
    const size_t base_row = (size_t)exp_id * (size_t)(2 * n_ff);
    const size_t gate_row = base_row + (size_t)ff_idx;
    const size_t up_row   = base_row + (size_t)(n_ff + ff_idx);

    float gate = 0.0f;
    float up   = 0.0f;

    if (w_type == 16 && (dim & 255) == 0) {
        const int n_sb = dim >> 8; // 9
        const int ib32 = lane >> 2; // 0..7
        const int l    = lane & 3;  // 0..3
        const uint8_t* g_row_ptr = gate_up_w + gate_row * (size_t)n_sb * 66ULL;
        const uint8_t* u_row_ptr = gate_up_w + up_row   * (size_t)n_sb * 66ULL;

        #pragma unroll 1
        for (int sb = 0; sb < n_sb; ++sb) {
            const float* xp = s_x + (sb << 8) + (lane << 3);
            const float x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];

            // Gate superblock `sb`
            {
                const uint8_t* blk = g_row_ptr + (size_t)sb * 66ULL;
                const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk + 2);
                const uint8_t u_grid = __ldg(&blk[2 + (ib32 << 3) + l]);
                const uint32_t hi = (uint32_t)__ldg(&qs16[(ib32 << 2) + 2])
                                  | ((uint32_t)__ldg(&qs16[(ib32 << 2) + 3]) << 16);
                const float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
                const uint64_t g64 = s_iq2xxs_grid[u_grid];
                const uint8_t signs = s_ksigns_iq2xs[(hi >> (7 * l)) & 127u];

                float sub_g = 0.0f;
                sub_g += ((signs & 0x01u) ? -1.0f : 1.0f) * (float)((g64 >>  0) & 0xffu) * x0;
                sub_g += ((signs & 0x02u) ? -1.0f : 1.0f) * (float)((g64 >>  8) & 0xffu) * x1;
                sub_g += ((signs & 0x04u) ? -1.0f : 1.0f) * (float)((g64 >> 16) & 0xffu) * x2;
                sub_g += ((signs & 0x08u) ? -1.0f : 1.0f) * (float)((g64 >> 24) & 0xffu) * x3;
                sub_g += ((signs & 0x10u) ? -1.0f : 1.0f) * (float)((g64 >> 32) & 0xffu) * x4;
                sub_g += ((signs & 0x20u) ? -1.0f : 1.0f) * (float)((g64 >> 40) & 0xffu) * x5;
                sub_g += ((signs & 0x40u) ? -1.0f : 1.0f) * (float)((g64 >> 48) & 0xffu) * x6;
                sub_g += ((signs & 0x80u) ? -1.0f : 1.0f) * (float)((g64 >> 56) & 0xffu) * x7;
                gate += db * sub_g;
            }

            // Up superblock `sb`
            {
                const uint8_t* blk = u_row_ptr + (size_t)sb * 66ULL;
                const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk + 2);
                const uint8_t u_grid = __ldg(&blk[2 + (ib32 << 3) + l]);
                const uint32_t hi = (uint32_t)__ldg(&qs16[(ib32 << 2) + 2])
                                  | ((uint32_t)__ldg(&qs16[(ib32 << 2) + 3]) << 16);
                const float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
                const uint64_t g64 = s_iq2xxs_grid[u_grid];
                const uint8_t signs = s_ksigns_iq2xs[(hi >> (7 * l)) & 127u];

                float sub_u = 0.0f;
                sub_u += ((signs & 0x01u) ? -1.0f : 1.0f) * (float)((g64 >>  0) & 0xffu) * x0;
                sub_u += ((signs & 0x02u) ? -1.0f : 1.0f) * (float)((g64 >>  8) & 0xffu) * x1;
                sub_u += ((signs & 0x04u) ? -1.0f : 1.0f) * (float)((g64 >> 16) & 0xffu) * x2;
                sub_u += ((signs & 0x08u) ? -1.0f : 1.0f) * (float)((g64 >> 24) & 0xffu) * x3;
                sub_u += ((signs & 0x10u) ? -1.0f : 1.0f) * (float)((g64 >> 32) & 0xffu) * x4;
                sub_u += ((signs & 0x20u) ? -1.0f : 1.0f) * (float)((g64 >> 40) & 0xffu) * x5;
                sub_u += ((signs & 0x40u) ? -1.0f : 1.0f) * (float)((g64 >> 48) & 0xffu) * x6;
                sub_u += ((signs & 0x80u) ? -1.0f : 1.0f) * (float)((g64 >> 56) & 0xffu) * x7;
                up += db * sub_u;
            }
        }
        gate = ink_warp_sum(gate);
        up   = ink_warp_sum(up);
    } else {
        gate = ink_warp_dot_dense(gate_up_w, w_type, gate_row, s_x, dim, lane);
        up   = ink_warp_dot_dense(gate_up_w, w_type, up_row,   s_x, dim, lane);
    }

    if (lane == 0) {
        const float act_g = situ_beta * tanhf(gate / situ_beta) * (1.0f / (1.0f + expf(-gate)));
        const float act_u = (situ_lb > 0.0f) ? (situ_lb * tanhf(up / situ_lb)) : up;
        situ_out[(size_t)slot * n_ff + ff_idx] = act_g * act_u;
    }
}

// -----------------------------------------------------------------------------
// 11. Sparse MoE Active-8 `ffn_down_exps` Weighted Accumulation
//     (`grid = (288, 1, 1)`, `block = (256, 1, 1)` = 8 warps/CTA -> `dim = 2304`)
//     Coalesced `Q2_0` (128 weights/iter) & shared-LUT `IQ4_NL` (64 weights/iter)!
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_moe_down_accum_kernel(
    float* out_f32,             // [dim]
    const uint8_t* down_w,      // [n_ff, dim, 128]
    const float* situ_in,       // [8, n_ff]
    const int32_t* top_ids,     // [8]
    const float* top_weights,   // [8]
    int32_t n_ff,               // 1152
    int32_t dim,                // 2304
    int32_t w_type              // 42 (Q2_0) or 20 (IQ4_NL)
) {
    __shared__ int8_t s_iq4nl[16];
    __shared__ float  s_act[1152];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int out_row = blockIdx.x * 8 + wid;

    if (tid < 16) {
        s_iq4nl[tid] = c_kvalues_iq4nl[tid];
    }

    float total = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        const int exp_id = top_ids[slot];
        const float w_exp = top_weights[slot];
        for (int i = tid; i < n_ff; i += 256) {
            s_act[i] = situ_in[(size_t)slot * n_ff + i];
        }
        __syncthreads();

        if (out_row < dim) {
            const size_t row_idx = (size_t)exp_id * (size_t)dim + (size_t)out_row;
            float dot = 0.0f;

            if (w_type == 42 && (n_ff & 127) == 0) {
                // Q2_0: 18 bytes / 64 weights -> 2 blocks (128 weights) per warp iteration!
                const int n_blk = n_ff >> 6; // 18
                const uint8_t* row_ptr = down_w + row_idx * (size_t)n_blk * 18ULL;
                const int b_sub = lane >> 4; // 0..1
                const int sub   = lane & 15; // 0..15
                const int n_pairs = n_blk >> 1; // 9
                #pragma unroll
                for (int pr = 0; pr < n_pairs; ++pr) {
                    const int bi = (pr << 1) + b_sub;
                    const uint8_t* blk = row_ptr + (size_t)bi * 18ULL;
                    const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                    const uint8_t qb = __ldg(&blk[2 + sub]);
                    const float* xp = s_act + (bi << 6) + (sub << 2);
                    const float acc4 = (float)((int)((qb >> 0) & 3u) - 1) * xp[0]
                                     + (float)((int)((qb >> 2) & 3u) - 1) * xp[1]
                                     + (float)((int)((qb >> 4) & 3u) - 1) * xp[2]
                                     + (float)((int)((qb >> 6) & 3u) - 1) * xp[3];
                    dot += d * acc4;
                }
                dot = ink_warp_sum(dot);
            } else if (w_type == 20 && (n_ff & 63) == 0) {
                // IQ4_NL: 18 bytes / 32 weights -> 2 blocks (64 weights) per warp iteration!
                const int n_blk = n_ff >> 5; // 36
                const uint8_t* row_ptr = down_w + row_idx * (size_t)n_blk * 18ULL;
                const int b_sub = lane >> 4; // 0..1
                const int sub   = lane & 15; // 0..15
                const int n_pairs = n_blk >> 1; // 18
                #pragma unroll
                for (int pr = 0; pr < n_pairs; ++pr) {
                    const int bi = (pr << 1) + b_sub;
                    const uint8_t* blk = row_ptr + (size_t)bi * 18ULL;
                    const float d = __half2float(__ldg(reinterpret_cast<const half*>(blk)));
                    const uint8_t qb = __ldg(&blk[2 + sub]);
                    const float* xp = s_act + (bi << 5);
                    dot += d * ((float)s_iq4nl[qb & 0x0fu] * xp[sub] + (float)s_iq4nl[qb >> 4] * xp[16 + sub]);
                }
                dot = ink_warp_sum(dot);
            } else {
                dot = ink_warp_dot_dense(down_w, w_type, row_idx, s_act, n_ff, lane);
            }

            if (lane == 0) {
                total += w_exp * dot;
            }
        }
        __syncthreads();
    }

    if (out_row < dim && lane == 0) {
        out_f32[out_row] = total;
    }
}

// -----------------------------------------------------------------------------
// 12. Vector Copy / Residual Add, Sampler & Tail Completion Signal
// -----------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256)
void ink_vec_op_kernel(
    float* dst,
    const float* a,
    const float* b, // nullptr -> pure copy `dst = a`; else `dst = a + b`
    int32_t n
) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        dst[i] = (b != nullptr) ? (a[i] + b[i]) : a[i];
    }
}

extern "C" __global__ __launch_bounds__(256)
void ink_sample_logits_kernel(
    int32_t* out_tok_host,      // host-mapped int32*
    const float* logits,        // [vocab_unpadded]
    const InkStepCtrl* ctrl,
    int32_t vocab_unpadded
) {
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];

    const int tid = threadIdx.x;
    const float temperature = ctrl->temperature;
    const uint64_t rng_seed = ctrl->rng_seed;
    const uint32_t step     = ctrl->step;

    float best_v = -1e30f;
    int32_t best_i = 0;

    for (int v = tid; v < vocab_unpadded; v += 256) {
        float score = logits[v];
        if (temperature > 1e-5f) {
            uint64_t x = rng_seed + ((uint64_t)step * 0x9E3779B97F4A7C15ULL) + ((uint64_t)v * 0xBF58476D1CE4E5B9ULL);
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27;
            x *= 0x94D049BB133111EBULL;
            x ^= x >> 31;
            uint32_t u24 = (uint32_t)(x >> 40);
            float u = fmaxf(1e-7f, fminf(1.0f - 1e-7f, (float)u24 * (1.0f / 16777216.0f)));
            float gumbel = -logf(-logf(u));
            score = (score / temperature) + gumbel;
        }
        if (score > best_v) {
            best_v = score;
            best_i = v;
        }
    }

    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid] ||
                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("st.global.wt.s32 [%0], %1;" :: "l"(out_tok_host), "r"(s_idx[0]) : "memory");
    }
}

extern "C" __global__ __launch_bounds__(32)
void ink_signal_done_kernel(
    uint64_t* done_seq_host,
    const InkStepCtrl* ctrl
) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        const uint64_t seq = ctrl->done_seq;
        asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(done_seq_host), "l"(seq) : "memory");
    }
}
