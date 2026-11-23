// =============================================================================
// Custom GA102 (RTX 3080, sm_86, 68 SMs) Persistent Megakernel & BM=8 Batched
// Speculative Verification / Prefill Suite for Thinking Machines `Inkling-2-Mini`
// (`inkling2_loop`, 26.4B Looped Sparse MoE: 24 physical layers -> 48 logical
// visits, 128 experts top-8 SITU, ShortConv, Relative-Position + Sink GQA,
// Depth Attention Residuals, and DSpark Block-8 Speculative Tap Extraction).
//
// Architecture v3 (`sm_86` Roofline Edition):
// 1. 68-SM Persistent Multi-Token Megakernel (`ink_persistent_decode_megakernel`):
//    - 1 single kernel launch per generation (`0` per-token doorbells / QMD fetches)
//    - Native Ampere `__dp4a` INT8 SIMD dot products (`Q8_1` dynamic activations)
//      across `Q8_0` (512 weights/warp/iter), `IQ2_XXS` (256/iter + 2-cycle 4-byte
//      SIMD sign flip), `Q2_0` (512/iter), `IQ4_NL` (256/iter), and `Q6_K` (512/iter)
//    - Register-fused `ShortConv` + `block_res` + `lm_head` softcap/argmax
//    - Posted PCIe live token streaming (`st.global.wt.s32`) + instant cancel
// 2. 68-SM Batched `BM <= 8` Prefill & Speculative Verify Megakernel
//    (`ink_batch_forward_bm8_megakernel`):
//    - Processes `1..8` tokens in lockstep across all 48 layer visits
//    - GPU-side Active Expert Deduplication: shared routed experts across the `B`
//      tokens are read ONCE from GDDR6X in both `ffn_gate_up_exps` & `ffn_down_exps`
//    - Per-token ShortConv state snapshots (`ws_conv_backup [8, 48, 6656, 3]`) for
//      5-microsecond exact lossless speculative rollback on draft rejection
//    - Exports exact `DSpark` (`Inkling2MTP`) hidden taps `[5, 23, 41, 48]`
//
// Precompiled ahead-of-time to `inkling_sm86.cubin` and embedded via `include_bytes!`
// (0 runtime Docker containers, 0 external Rust crates, 100% RTX 3080 compute).
// =============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

static constexpr int INK_THREADS = 512;
static constexpr int INK_WARPS   = 16; // 512 / 32 -> 1,088 concurrent warps across 68 SMs!

static constexpr int TOK_END_SAMPLING = 200006; // `<|content_model_end_sampling|>`

// -----------------------------------------------------------------------------
// Host / Device ABI Descriptors (`#[repr(C)]` 16/64-byte aligned in Rust)
// -----------------------------------------------------------------------------

struct __align__(16) InkPhysLayerDesc {
    uint64_t attn_norm;          // F32 [2304] (SWA)
    uint64_t attn_q_inp_norm;    // F32 [2304] (Full)
    uint64_t attn_k_inp_norm;    // F32 [2304] (Full)
    uint64_t attn_v_inp_norm;    // F32 [2304] (Full)
    uint64_t attn_res_query;     // F32 [2304 * (1 or 3)] (0 on layer 0)
    uint64_t wq;                 // Q8_0 [2304 -> 3072]
    uint64_t wk;                 // Q8_0 [2304 -> 1024]
    uint64_t wv;                 // Q8_0 [2304 -> 1024]
    uint64_t wo;                 // Q8_0 [3072 -> 2304]
    uint64_t wqkv_gate;          // Q8_0 [2304 -> 3072]
    uint64_t attn_rel;           // Q8_0 [2304 -> 384]
    uint64_t attn_rel_bias;      // F32  [16 * 1024]
    uint64_t attn_sink_proj;     // Q8_0 [2304 -> 24]
    uint64_t attn_q_clip;        // F32  [3072]
    uint64_t attn_rel_clip;      // F32  [384]
    uint64_t attn_k_conv;        // F32  [4 * 1024]
    uint64_t attn_v_conv;        // F32  [4 * 1024]
    uint64_t attn_k_conv_proj;   // Q8_0 [2304 -> 32] (0 on SWA)
    uint64_t attn_v_conv_proj;   // Q8_0 [2304 -> 32] (0 on SWA)
    uint64_t attn_post_conv;     // F32  [4 * 2304]
    uint64_t ffn_res_query;      // F32  [2304]
    uint64_t ffn_norm;           // F32  [2304]
    uint64_t ffn_gate_inp;       // F32  [2304 -> 128]
    uint64_t exp_probs_b;        // F32  [128]
    uint64_t ffn_gate_up_exps;   // IQ2_XXS [2304, 2304, 128]
    uint64_t ffn_down_exps;      // Q2_0 or IQ4_NL [1152, 2304, 128]
    uint64_t ffn_post_conv;      // F32  [4 * 2304]
    int32_t  is_swa;
    int32_t  qkv_type;           // 8 (Q8_0)
    int32_t  wo_type;            // 8 (Q8_0)
    int32_t  gate_up_type;       // 16 (IQ2_XXS)
    int32_t  down_type;          // 42 (Q2_0) or 20 (IQ4_NL)
    int32_t  _pad[5];
}; // 256 bytes

struct __align__(16) InkLoopCtrlDesc {
    uint64_t msg_query;          // F32  [2304 * 4]
    uint64_t ctrl_down[4];       // Q8_0 [4608 -> 64]
    uint64_t ctrl_gain[4];       // Q8_0 [64 -> 2304]
    uint64_t ctrl_forget[4];     // Q8_0 [64 -> 2304]
    uint64_t _pad[3];
}; // 128 bytes

struct __align__(16) InkModelDesc {
    uint64_t layers_ptr;         // const InkPhysLayerDesc* [24]
    uint64_t loop_ctrls_ptr;     // const InkLoopCtrlDesc* [12]
    uint64_t slot_kv_dense_ptr;  // const int32_t* [48]
    uint64_t tok_embd_va;
    uint64_t tok_norm_va;
    uint64_t output_w_va;
    uint64_t output_norm_va;
    uint64_t output_res_query_va;

    uint64_t swa_k_va;           // half* [40, 1024, 1024]
    uint64_t swa_v_va;           // half* [40, 1024, 1024]
    uint64_t full_k_va;          // half* [8, max_ctx, 1024]
    uint64_t full_v_va;          // half* [8, max_ctx, 1024]
    uint64_t conv_state_va;      // float* [48, 6656, 3]
    uint64_t conv_backup_va;     // float* [8, 48, 6656, 3] (for speculative rollback)

    // Workspace buffers in GDDR6X (sized for up to B = 8)
    uint64_t ws_cur;             // float* [8, 2304]
    uint64_t ws_inp_sa;          // float* [8, 2304]
    uint64_t ws_ffn_inp;         // float* [8, 2304]
    uint64_t ws_prelude;         // float* [8, 2304]
    uint64_t ws_bank_out;        // float* [8, 2304]
    uint64_t ws_blocks;          // float* [8, 8, 2304]
    uint64_t ws_prelude_blocks;  // float* [8, 4, 2304]
    uint64_t ws_bank;            // float* [8, 4, 2304]
    uint64_t ws_messages;        // float* [8, 4, 2304]
    uint64_t ws_mixed;           // float* [8, 3, 2304]
    uint64_t ws_inputs;          // float* [8, 3, 2304]
    uint64_t ws_norm_in;         // float* [8, 3, 2304]
    uint64_t ws_ctrl_cat;        // float* [8, 3, 4608]
    uint64_t ws_ctrl_gate;       // float* [8, 3, 64]
    uint64_t ws_ctrl_gain;       // float* [8, 3, 2304]
    uint64_t ws_ctrl_forget;     // float* [8, 3, 2304]

    uint64_t ws_xq;              // int8_t* [8, 3, 4608]
    uint64_t ws_xds;             // float2* [8, 3, 144]
    uint64_t ws_xq_moe;          // int8_t* [8, 8, 1152]
    uint64_t ws_xds_moe;         // float2* [8, 8, 36]

    uint64_t ws_q;               // float* [8, 3072]
    uint64_t ws_k_raw;           // float* [8, 1024]
    uint64_t ws_v_raw;           // float* [8, 1024]
    uint64_t ws_r;               // float* [8, 384]
    uint64_t ws_sinks;           // float* [8, 24]
    uint64_t ws_gate_raw;        // float* [8, 3072]
    uint64_t ws_dyn_k;           // float* [8, 32]
    uint64_t ws_dyn_v;           // float* [8, 32]
    uint64_t ws_attn_gated;      // float* [8, 3072]

    uint64_t ws_router_logits;   // float* [8, 128]
    uint64_t ws_top_ids;         // int32_t* [8, 8]
    uint64_t ws_top_weights;     // float* [8, 8]
    uint64_t ws_uniq_exp_id;     // int32_t* [128]
    uint64_t ws_uniq_mask;       // uint32_t* [128]
    uint64_t ws_uniq_slots;      // int32_t* [128 * 8]
    uint64_t ws_n_uniq;          // int32_t* [4]
    uint64_t ws_situ;            // float* [8, 8, 1152]

    uint64_t ws_cta_val;         // float* [8 * 128]
    uint64_t ws_cta_idx;         // int32_t* [8 * 128]
    uint64_t ws_bar_state;       // uint64_t* [4]: [0]=monotonic_cnt, [1]=step_base
    uint64_t ws_dspark_taps;     // float* [8, 4, 2304] (visits 5, 23, 41, 48)
    uint64_t ws_logits;          // float* [8, vocab_unpadded] (optional export)

    int32_t  dim;                // 2304
    int32_t  n_ff_exp;           // 1152
    int32_t  n_heads;            // 24
    int32_t  n_kv_heads;         // 8
    int32_t  head_dim;           // 128
    int32_t  n_experts;          // 128
    int32_t  n_experts_used;     // 8
    int32_t  n_phys_layers;      // 24
    int32_t  loop_start;         // 6
    int32_t  loop_end;           // 18
    int32_t  loop_passes;        // 3
    int32_t  gate_rank;          // 64
    int32_t  vocab_unpadded;     // 200058
    int32_t  swa_window;         // 1024
    int32_t  rel_len;            // 16
    int32_t  rel_extent;         // 1024
    int32_t  shortconv_len;      // 4
    int32_t  attn_res_block;     // 6
    int32_t  max_ctx;
    int32_t  tok_embd_type;      // 12 (Q4_K)
    int32_t  output_w_type;      // 14 (Q6_K)
    int32_t  log_scale_floor;    // 8192

    float    rms_eps;
    float    qk_clip;
    float    rel_clip;
    float    log_scale_alpha;
    float    situ_beta;
    float    situ_lb;
    float    expert_weights_scale;
    float    final_logit_softcap;
    int32_t  _pad_model[2];
}; // 56 * 8 + 32 * 4 = 576 bytes

struct __align__(16) InkDSparkLayerDesc {
    uint64_t dflare_norm;        // F32  [2304]
    uint64_t input_layernorm;    // F32  [2304]
    uint64_t post_attn_norm;     // F32  [2304]
    uint64_t qkv_proj;           // Aligned Split Q8_0 [5120, 2304] (q[3072] + k[1024] + v[1024])
    uint64_t o_proj;             // Aligned Split Q8_0 [2304, 3072]
    uint64_t kv_proj_ctx;        // Aligned Split Q8_0 [2048, 2304] (k_ctx[1024] + v_ctx[1024])
    uint64_t q_norm;             // F32  [128]
    uint64_t k_norm;             // F32  [128]
    uint64_t k_norm_ctx;         // F32  [128]
    uint64_t gate_up_proj;       // Aligned Split Q8_0 [18432, 2304] (gate[9216] + up[9216])
    uint64_t down_proj;          // Aligned Split Q8_0 [2304, 9216]
    int32_t  is_local;           // 1 for layers 0, 2; 0 for layer 1
    int32_t  _pad[9];
}; // 128 bytes

struct __align__(16) InkDSparkDesc {
    InkDSparkLayerDesc layers[3]; // 384 B
    float    fuse_alpha[16];      // [4, 4] row-major softmax(fuse_weight) = 64 B
    uint64_t mask_embed_va;       // F32  [2304]
    uint64_t final_norm_va;       // F32  [2304]
    uint64_t markov_dflare_va;    // F32  [2304]
    uint64_t markov_w1_bf16_va;   // uint16_t* [200058, 256] (BF16)
    uint64_t rnn_proj_w_va;       // Aligned Split Q8_0 [768, 5120]
    uint64_t rnn_proj_b_va;       // F32  [768]
    uint64_t fold_w_va;           // Aligned Split Q8_0 [200058, 256]
    uint64_t fold_b_va;           // F32  [200058]

    uint64_t draft_k_va;          // half* [3, max_ctx, 1024]
    uint64_t draft_v_va;          // half* [3, max_ctx, 1024]
    uint64_t ws_f_rnn_va;         // float* [2304]

    uint64_t ws_h_va;             // float* [8, 2304]
    uint64_t ws_qkv_va;           // float* [24 * 2048] (>= 8 * 5120)
    uint64_t ws_mlp_va;           // float* [8 * 9216]
    uint64_t ws_xq_va;            // int8_t* [8 * 9216] (>= 24 * 2304)
    uint64_t ws_xds_va;           // float2* [8 * 288]  (>= 24 * 72)
    uint64_t ws_base_logits_va;   // half*  [7 * vocab_unpadded] (L2-resident!)
    uint64_t ws_rnn_state_va;     // float* [256]
    uint64_t ws_rnn_inp_va;       // float* [5120]
    uint64_t ws_rnn_proj_va;      // float* [768]

    int32_t  max_ctx;
    int32_t  swa_window;
    float    rope_theta;
    float    rms_eps;
}; // 624 bytes

struct __align__(64) InkRunCtrl {
    int32_t  start_token;
    int32_t  start_pos;
    int32_t  max_steps;
    int32_t  eff_passes;         // 1, 2, or 3
    float    temperature;
    uint32_t step_offset;        // RNG step offset
    uint64_t rng_seed;

    int32_t  cancel_flag;        // host writes 1 to abort rollout early
    int32_t  error_code;         // >0 if watchdog ever trips
    int32_t  steps_completed;    // GPU increments after each token
    int32_t  stop_reason;        // 0=max_steps, 1=eos, 2=cancelled, 3=ctx_limit
    int32_t  export_taps;        // 1 to export DSpark taps [5, 23, 41, 48]
    int32_t  save_conv_backup;   // 1 during speculative verify to snapshot ShortConv per b
    int32_t  batch_len;          // 1..8 for bm8 / dspark
    uint32_t logits_mask;        // bitmask of batch slots `b` that run `lm_head`

    int32_t  inject_pos;         // DSpark fused context KV injection start_pos
    int32_t  inject_cnt;         // DSpark fused context KV injection count (0..8)
    int32_t  _pad_ctrl[2];

    int32_t  batch_tokens[8];    // input tokens for bm8 kernel
    int32_t  batch_out_toks[8];  // sampled/argmax output tokens per batch slot `b`
    uint64_t done_seq;           // completion sequence written at kernel exit
    uint64_t expected_seq;       // target sequence for host polling
    int32_t  tokens_out[4096];   // streamed token IDs for persistent rollout
};

// -----------------------------------------------------------------------------
// Official GGML Lookup Tables (`IQ2_XXS` & `IQ4_NL`)
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
// Ampere `sm_86` PTX Cache-Controlled Memory, Barrier, & `__dp4a` Primitives
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

__device__ __forceinline__ int2 ld_cg_s32x2(const int* p) {
    int2 v;
    asm volatile("ld.global.cg.v2.s32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ float2 ld_cg_f32x2(const float2* p) {
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
                 : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ void st_cg_s8(int8_t* p, int v) {
    asm volatile("st.global.cg.s8 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_cg_f32x2(float2* p, float2 v) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(v.x), "f"(v.y) : "memory");
}

__device__ __forceinline__ void st_host_s32(volatile int32_t* p, int32_t v) {
    asm volatile("st.global.wt.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int32_t ld_host_s32(const volatile int32_t* p) {
    int32_t v;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t load_u32_align2(const uint8_t* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return ((uint32_t)__ldg(&p16[0])) | (((uint32_t)__ldg(&p16[1])) << 16);
}

__device__ __forceinline__ float ink_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float ink_uniform_01(uint64_t seed, uint32_t step, uint32_t idx) {
    uint64_t x = seed + ((uint64_t)step * 0x9E3779B97F4A7C15ULL) + ((uint64_t)idx * 0xBF58476D1CE4E5B9ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    uint32_t u24 = (uint32_t)(x >> 40);
    return fmaxf(1e-7f, fminf(1.0f - 1e-7f, (float)u24 * (1.0f / 16777216.0f)));
}

__device__ __forceinline__ uint64_t ink_gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ bool ink_grid_barrier(
    uint64_t* bar_cnt,
    uint64_t& step,
    int num_blocks,
    int& s_abort,
    volatile InkRunCtrl* ctrl
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
                if (++spin > 50000000U) {
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

__device__ __forceinline__ void ink_init_smem_luts(
    uint64_t* s_iq2_grid,
    uint8_t* s_iq2_signs,
    uint32_t* s_sign_mask4,
    int32_t* s_q2_lut,
    uint8_t* s_iq4_u8
) {
    const int tid = threadIdx.x;
    if (tid < 256) {
        s_iq2_grid[tid] = c_iq2xxs_grid[tid];
        const int b0 = ((tid >> 0) & 3) - 1;
        const int b1 = ((tid >> 2) & 3) - 1;
        const int b2 = ((tid >> 4) & 3) - 1;
        const int b3 = ((tid >> 6) & 3) - 1;
        s_q2_lut[tid] = (int32_t)(((uint32_t)(b0 & 0xff))
                      | (((uint32_t)(b1 & 0xff)) << 8)
                      | (((uint32_t)(b2 & 0xff)) << 16)
                      | (((uint32_t)(b3 & 0xff)) << 24));
    }
    if (tid < 128) {
        s_iq2_signs[tid] = c_ksigns_iq2xs[tid];
    }
    if (tid < 16) {
        s_iq4_u8[tid] = (uint8_t)c_kvalues_iq4nl[tid];
        s_sign_mask4[tid] = ((tid & 1) ? 0x000000ffu : 0u)
                          | ((tid & 2) ? 0x0000ff00u : 0u)
                          | ((tid & 4) ? 0x00ff0000u : 0u)
                          | ((tid & 8) ? 0xff000000u : 0u);
    }
}

__device__ __forceinline__ float ink_dequant_elem_emb(
    const uint8_t* w_raw,
    int w_type,
    size_t row_idx,
    int k_cols,
    int col
) {
    if (w_type == 12) { // Q4_K (144 B / 256)
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
    } else if (w_type == 8) { // Q8_0
        const int n_blk = k_cols >> 5;
        const uint8_t* blk = w_raw + (row_idx * (size_t)n_blk + (size_t)(col >> 5)) * 34ULL;
        return __half2float(__ldg(reinterpret_cast<const half*>(blk))) *
               (float)__ldg(&reinterpret_cast<const int8_t*>(blk + 2)[col & 31]);
    } else if (w_type == 14) { // Q6_K
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
    } else if (w_type == 30) { // BF16
        const uint16_t* row = reinterpret_cast<const uint16_t*>(w_raw) + row_idx * (size_t)k_cols;
        return __uint_as_float(((uint32_t)__ldg(&row[col])) << 16);
    } else if (w_type == 1) { // F16
        const half* row = reinterpret_cast<const half*>(w_raw) + row_idx * (size_t)k_cols;
        return __half2float(__ldg(&row[col]));
    }
    const float* row = reinterpret_cast<const float*>(w_raw) + row_idx * (size_t)k_cols;
    return __ldg(&row[col]);
}

// Quantize an FP32 vector `in_f32 [K]` (with optional RMSNorm `norm_w`) into L2 `g_xq` + `g_xds`
// (and optionally also store FP32 normalized vector to `out_f32_opt`).
__device__ inline void dev_cta_rmsnorm_q8_1(
    float* out_f32_opt,
    int8_t* g_xq,
    float2* g_xds,
    float* s_red,
    const float* in_f32,
    const float* norm_w,
    int K,
    float eps,
    bool apply_rmsnorm
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_blks = K >> 5;

    float scale = 1.0f;
    if (apply_rmsnorm) {
        float ss = 0.0f;
        for (int col = tid; col < K; col += INK_THREADS) {
            float v = ld_cg_f32(&in_f32[col]);
            ss += v * v;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot / (float)K + eps);
        }
        __syncthreads();
        scale = s_red[0];
    }

    for (int blk = wid; blk < n_blks; blk += INK_WARPS) {
        const int col = (blk << 5) + lane;
        float v = ld_cg_f32(&in_f32[col]) * scale;
        if (apply_rmsnorm && norm_w != nullptr) {
            v *= __ldg(&norm_w[col]);
        }
        if (out_f32_opt != nullptr) {
            st_cg_f32(&out_f32_opt[col], v);
        }
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        st_cg_s8(&g_xq[col], max(-127, min(127, __float2int_rn(v * id))));
        if (lane == 0) st_cg_f32x2(&g_xds[blk], make_float2(d, sum_x));
    }
}

__device__ inline void dev_load_q8_1_smem(
    int* s_xq,
    float2* s_xds,
    const int8_t* g_xq,
    const float2* g_xds,
    int K
) {
    const int tid = threadIdx.x;
    const int n_int4 = K >> 4;
    const int* g_i32 = reinterpret_cast<const int*>(g_xq);
    int4* s_i4 = reinterpret_cast<int4*>(s_xq);
    for (int i = tid; i < n_int4; i += INK_THREADS) {
        s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
    }
    const int n_blks = K >> 5;
    for (int b = tid; b < n_blks; b += INK_THREADS) {
        s_xds[b] = ld_cg_f32x2(&g_xds[b]);
    }
    __syncthreads();
}

__device__ __forceinline__ float ink_hwarp_sum(float v, uint32_t hmask) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v += __shfl_down_sync(hmask, v, off);
    }
    return v;
}

// 16-lane Half-Warp Aligned Split `Q8_0` (`qs: int8[N_rows, K]` at `w_raw`, `d: half[N_rows, K/32]` at `w_raw + N_rows * K`)
// Issues 100%-coalesced 128-bit `LDG.E.128` (`int4`) weight loads (256 weights = 2 exact cache lines per iter)!
__device__ __forceinline__ float dev_hwarp_dot_q8_0_aligned(
    const uint8_t* w_raw,
    size_t n_rows_total,
    size_t row_idx,
    const int* s_xq,
    const float2* s_xds,
    int K,
    int lane16,
    uint32_t hmask
) {
    const int n_blk = K >> 5;
    const int n_i4  = K >> 4;
    const int4* qs_row = reinterpret_cast<const int4*>(w_raw + row_idx * (size_t)K);
    const half* d_row  = reinterpret_cast<const half*>(w_raw + n_rows_total * (size_t)K) + row_idx * (size_t)n_blk;
    const int4* x_i4   = reinterpret_cast<const int4*>(s_xq);

    float acc = 0.0f;
    if (n_i4 == 4) {
        if (lane16 < 4) {
            const int4 wv = __ldg(&qs_row[lane16]);
            const int4 xv = x_i4[lane16];
            const int b = lane16 >> 1;
            const float d = __half2float(__ldg(&d_row[b]));
            int sumi = __dp4a(wv.x, xv.x, 0);
            sumi = __dp4a(wv.y, xv.y, sumi);
            sumi = __dp4a(wv.z, xv.z, sumi);
            sumi = __dp4a(wv.w, xv.w, sumi);
            acc = (d * s_xds[b].x) * (float)sumi;
        }
        return ink_hwarp_sum(acc, hmask);
    }
    #pragma unroll 3
    for (int i4 = lane16; i4 < n_i4; i4 += 16) {
        const int4 wv = __ldg(&qs_row[i4]);
        const int4 xv = x_i4[i4];
        const int b = i4 >> 1;
        const float d = __half2float(__ldg(&d_row[b]));
        int sumi = __dp4a(wv.x, xv.x, 0);
        sumi = __dp4a(wv.y, xv.y, sumi);
        sumi = __dp4a(wv.z, xv.z, sumi);
        sumi = __dp4a(wv.w, xv.w, sumi);
        acc += (d * s_xds[b].x) * (float)sumi;
    }
    return ink_hwarp_sum(acc, hmask);
}

// 4-lane Eighth-Warp Aligned Split `Q8_0` for `K = 64` (`128` rows/CTA, `100%` active lanes, `512B` coalesced per warp!)
__device__ __forceinline__ float dev_4lane_dot_q8_0_k64(
    const uint8_t* w_raw,
    size_t n_rows_total,
    size_t row_idx,
    const int* s_xq,
    const float2* s_xds,
    int lane4,
    uint32_t mask4
) {
    const int4* qs_row = reinterpret_cast<const int4*>(w_raw + row_idx * 64ULL);
    const half* d_row  = reinterpret_cast<const half*>(w_raw + n_rows_total * 64ULL) + row_idx * 2ULL;
    const int4 wv = __ldg(&qs_row[lane4]);
    const int4 xv = reinterpret_cast<const int4*>(s_xq)[lane4];
    const int b = lane4 >> 1;
    const float d = __half2float(__ldg(&d_row[b]));
    int sumi = __dp4a(wv.x, xv.x, 0);
    sumi = __dp4a(wv.y, xv.y, sumi);
    sumi = __dp4a(wv.z, xv.z, sumi);
    sumi = __dp4a(wv.w, xv.w, sumi);
    float acc = (d * s_xds[b].x) * (float)sumi;
    acc += __shfl_down_sync(mask4, acc, 2);
    acc += __shfl_down_sync(mask4, acc, 1);
    return acc;
}

// K-sliced 16-lane Half-Warp Aligned Split `Q8_0` (`[i4_beg .. i4_end)`) for zero-tail `wo` (`2304 * 4 = 9216` items)
__device__ __forceinline__ float dev_hwarp_dot_q8_0_slice(
    const uint8_t* w_raw,
    size_t n_rows_total,
    size_t row_idx,
    const int* s_xq,
    const float2* s_xds,
    int K,
    int i4_beg,
    int i4_end,
    int lane16,
    uint32_t hmask
) {
    const int n_blk = K >> 5;
    const int4* qs_row = reinterpret_cast<const int4*>(w_raw + row_idx * (size_t)K);
    const half* d_row  = reinterpret_cast<const half*>(w_raw + n_rows_total * (size_t)K) + row_idx * (size_t)n_blk;
    const int4* x_i4   = reinterpret_cast<const int4*>(s_xq);

    float acc = 0.0f;
    #pragma unroll 3
    for (int i4 = i4_beg + lane16; i4 < i4_end; i4 += 16) {
        const int4 wv = __ldg(&qs_row[i4]);
        const int4 xv = x_i4[i4];
        const int b = i4 >> 1;
        const float d = __half2float(__ldg(&d_row[b]));
        int sumi = __dp4a(wv.x, xv.x, 0);
        sumi = __dp4a(wv.y, xv.y, sumi);
        sumi = __dp4a(wv.z, xv.z, sumi);
        sumi = __dp4a(wv.w, xv.w, sumi);
        acc += (d * s_xds[b].x) * (float)sumi;
    }
    return ink_hwarp_sum(acc, hmask);
}

// 32-lane Warp Aligned Split `Q8_0` (`512` weights = `4` exact cache lines per iter via `LDG.E.128`)
__device__ __forceinline__ float dev_warp_dot_q8_0_aligned(
    const uint8_t* w_raw,
    size_t n_rows_total,
    size_t row_idx,
    const int* s_xq,
    const float2* s_xds,
    int K,
    int lane
) {
    const int n_blk = K >> 5;
    const int n_i4  = K >> 4;
    const int4* qs_row = reinterpret_cast<const int4*>(w_raw + row_idx * (size_t)K);
    const half* d_row  = reinterpret_cast<const half*>(w_raw + n_rows_total * (size_t)K) + row_idx * (size_t)n_blk;
    const int4* x_i4   = reinterpret_cast<const int4*>(s_xq);

    float acc = 0.0f;
    if (n_i4 == 4) {
        if (lane < 4) {
            const int4 wv = __ldg(&qs_row[lane]);
            const int4 xv = x_i4[lane];
            const int b = lane >> 1;
            const float d = __half2float(__ldg(&d_row[b]));
            int sumi = __dp4a(wv.x, xv.x, 0);
            sumi = __dp4a(wv.y, xv.y, sumi);
            sumi = __dp4a(wv.z, xv.z, sumi);
            sumi = __dp4a(wv.w, xv.w, sumi);
            acc = (d * s_xds[b].x) * (float)sumi;
        }
        return ink_warp_sum(acc);
    }
    #pragma unroll 3
    for (int i4 = lane; i4 < n_i4; i4 += 32) {
        const int4 wv = __ldg(&qs_row[i4]);
        const int4 xv = x_i4[i4];
        const int b = i4 >> 1;
        const float d = __half2float(__ldg(&d_row[b]));
        int sumi = __dp4a(wv.x, xv.x, 0);
        sumi = __dp4a(wv.y, xv.y, sumi);
        sumi = __dp4a(wv.z, xv.z, sumi);
        sumi = __dp4a(wv.w, xv.w, sumi);
        acc += (d * s_xds[b].x) * (float)sumi;
    }
    return ink_warp_sum(acc);
}

// Single-row Aligned Split `Q6_K` x `Q8_1` `__dp4a` (`payload: 208B/sb` at `w_raw`, `d: half[tot_sb]` after payload)
// Uses 2 x 128-bit `LDG.E.128` (`uint4`) loads instead of 16 x 16-bit loads!
__device__ __forceinline__ float dev_warp_dot_q6_k_aligned(
    const uint8_t* w_raw,
    size_t n_rows_total,
    size_t row_idx,
    const int* s_xq,
    const float2* s_xds,
    int K,
    int lane
) {
    const int n_sb = K >> 8;
    const size_t tot_sb = n_rows_total * (size_t)n_sb;
    const uint8_t* w_row = w_raw + row_idx * (size_t)(n_sb * 208);
    const half* d_row = reinterpret_cast<const half*>(w_raw + tot_sb * 208ULL) + row_idx * (size_t)n_sb;
    const int sb_lane = lane >> 4, tid16 = lane & 15;
    const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
    const int ql_off = (h << 6) + (q << 5) + (half8 << 4);
    const int qh_off = 128 + (h << 5) + (half8 << 4);
    const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;

    float acc = 0.0f;
    #pragma unroll
    for (int sb = sb_lane; sb < 9; sb += 2) {
        if (sb < n_sb) {
            const uint8_t* blk = w_row + sb * 208;
            const float d_sc = __half2float(__ldg(&d_row[sb])) *
                               (float)__ldg(&reinterpret_cast<const int8_t*>(blk + 192)[tid16]);
            const uint4 ql = __ldg(reinterpret_cast<const uint4*>(blk + ql_off));
            const uint4 qh = __ldg(reinterpret_cast<const uint4*>(blk + qh_off));

            const uint32_t u6_0 = ((ql.x >> ql_sh) & 0x0f0f0f0fu) | (((qh.x >> qh_sh) & 0x03030303u) << 4);
            const uint32_t u6_1 = ((ql.y >> ql_sh) & 0x0f0f0f0fu) | (((qh.y >> qh_sh) & 0x03030303u) << 4);
            const uint32_t u6_2 = ((ql.z >> ql_sh) & 0x0f0f0f0fu) | (((qh.z >> qh_sh) & 0x03030303u) << 4);
            const uint32_t u6_3 = ((ql.w >> ql_sh) & 0x0f0f0f0fu) | (((qh.w >> qh_sh) & 0x03030303u) << 4);
            const uint32_t m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
            const uint32_t m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
            const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
            const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
            const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
            const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));

            const int x_base_int = (sb << 6) + (tid16 << 2);
            const int ds_idx = (sb << 3) + (tid16 >> 1);
            const int4 xv = *reinterpret_cast<const int4*>(s_xq + x_base_int);
            int sumi = __dp4a(s6_0, xv.x, 0);
            sumi = __dp4a(s6_1, xv.y, sumi);
            sumi = __dp4a(s6_2, xv.z, sumi);
            sumi = __dp4a(s6_3, xv.w, sumi);
            acc += (d_sc * s_xds[ds_idx].x) * (float)sumi;
        }
    }
    return ink_warp_sum(acc);
}

// Quantize an L2 FP32 vector `in_f32 [K]` (with optional RMSNorm `norm_w`) directly into SMEM `s_xq` + `s_xds`
// (`0` global memory writes, `0` grid barrier!). Optionally also writes normalized FP32 to `s_out_f32` or `g_out_f32`.
__device__ inline void dev_cta_rmsnorm_q8_1_to_smem(
    float* out_f32_opt,
    int* s_xq,
    float2* s_xds,
    float* s_red,
    const float* in_f32,
    const float* norm_w,
    int K,
    float eps,
    bool apply_rmsnorm
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_blks = K >> 5;
    int8_t* s_xq_s8 = reinterpret_cast<int8_t*>(s_xq);

    float scale = 1.0f;
    if (apply_rmsnorm) {
        float ss = 0.0f;
        for (int col = tid; col < K; col += INK_THREADS) {
            float v = ld_cg_f32(&in_f32[col]);
            ss += v * v;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot / (float)K + eps);
        }
        __syncthreads();
        scale = s_red[0];
    }

    for (int blk = wid; blk < n_blks; blk += INK_WARPS) {
        const int col = (blk << 5) + lane;
        float v = ld_cg_f32(&in_f32[col]) * scale;
        if (apply_rmsnorm && norm_w != nullptr) {
            v *= __ldg(&norm_w[col]);
        }
        if (out_f32_opt != nullptr) {
            out_f32_opt[col] = v;
        }
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        s_xq_s8[col] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
        if (lane == 0) s_xds[blk] = make_float2(d, sum_x);
    }
    __syncthreads();
}

// 1-Pass Warp-Parallel Depth Attention Residual Mix (`out [dim] = mix(blocks [n_blocks, dim], cur [dim], qv [dim])`)
// Uses mathematical identity `sum_c (st[c]*irms)*qv[c] == irms * sum_c (st[c]*qv[c])` across warps `wid = 0..n_states-1`:
// cuts `34` `__syncthreads()` + `24` global passes down to `2` `__syncthreads()` + `1` `float4` pass!
__device__ inline void dev_cta_depth_mix_single(
    float* out_f32,
    float* s_scratch, // needs >= 64 floats
    const float* blocks_f32,
    const float* cur_f32,
    const float* qv_f32,
    int dim,
    int n_blocks,
    float eps,
    bool out_is_smem = false
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_states = n_blocks + 1;
    const float inv_sqrt_d = rsqrtf((float)dim);

    float* s_irms   = s_scratch + 16;  // [16]
    float* s_scores = s_scratch + 32;  // [16]
    float* s_probs  = s_scratch + 48;  // [16]

    if (wid < n_states) {
        const float* st = (wid < n_blocks) ? (blocks_f32 + (size_t)wid * dim) : cur_f32;
        float ss = 0.0f;
        float raw_dot = 0.0f;
        for (int col = lane * 4; col < dim; col += 128) {
            const float s0 = ld_cg_f32(&st[col + 0]), s1 = ld_cg_f32(&st[col + 1]);
            const float s2 = ld_cg_f32(&st[col + 2]), s3 = ld_cg_f32(&st[col + 3]);
            const float4 q4 = __ldg(reinterpret_cast<const float4*>(qv_f32 + col));
            ss += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
            raw_dot += s0 * q4.x + s1 * q4.y + s2 * q4.z + s3 * q4.w;
        }
        ss = ink_warp_sum(ss);
        raw_dot = ink_warp_sum(raw_dot);
        if (lane == 0) {
            const float irms = rsqrtf(ss / (float)dim + eps);
            s_irms[wid] = irms;
            s_scores[wid] = (raw_dot * irms) * inv_sqrt_d;
        }
    }
    __syncthreads();

    if (wid == 0) {
        float sc = (lane < n_states) ? s_scores[lane] : -1e30f;
        float mx = sc;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        float p = (lane < n_states) ? expf(sc - mx) : 0.0f;
        float den = p;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) den += __shfl_xor_sync(0xffffffff, den, off);
        if (lane < n_states) s_probs[lane] = p / fmaxf(den, 1e-20f);
    }
    __syncthreads();

    for (int col = tid; col < dim; col += INK_THREADS) {
        float acc = 0.0f;
        for (int m = 0; m < n_states; ++m) {
            const float* st = (m < n_blocks) ? (blocks_f32 + (size_t)m * dim) : cur_f32;
            acc += s_probs[m] * (ld_cg_f32(&st[col]) * s_irms[m]);
        }
        if (out_is_smem) {
            out_f32[col] = acc;
        } else {
            st_cg_f32(&out_f32[col], acc);
        }
    }
}

// Quantize an SMEM FP32 vector `s_in [K]` (with optional RMSNorm `norm_w`, optionally updating `s_in` in-place)
// directly into SMEM `s_xq` + `s_xds` (`0` global memory touches, `0` grid barrier!).
__device__ inline void dev_smem_rmsnorm_q8_1_inplace(
    bool write_back_s_in,
    int* s_xq,
    float2* s_xds,
    float* s_red,
    float* s_in,
    const float* norm_w,
    int K,
    float eps
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int n_blks = K >> 5;
    int8_t* s_xq_s8 = reinterpret_cast<int8_t*>(s_xq);

    float ss = 0.0f;
    for (int col = tid; col < K; col += INK_THREADS) {
        float v = s_in[col];
        ss += v * v;
    }
    ss = ink_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
        tot = ink_warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / (float)K + eps);
    }
    __syncthreads();
    const float scale = s_red[0];

    for (int blk = wid; blk < n_blks; blk += INK_WARPS) {
        const int col = (blk << 5) + lane;
        float v = s_in[col] * scale * __ldg(&norm_w[col]);
        if (write_back_s_in) s_in[col] = v;
        float sum_x = v, amax = fabsf(v);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax * (1.0f / 127.0f);
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        s_xq_s8[col] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
        if (lane == 0) s_xds[blk] = make_float2(d, sum_x);
    }
    __syncthreads();
}

// =============================================================================
// 1. 68-SM Persistent Multi-Token Decode Megakernel (`B = 1`, `0`B Spills, Roofline v5)
//    - `__launch_bounds__(576, 1)` caps `ptxas` at <= 104 regs/thread (`0B` spills!)
//    - 236 full-GPU barriers eliminated per token via on-chip SMEM DepthMix+RMSNorm,
//      in-register L4 Q/R clipping, per-CTA warp top-8 router, & SMEM Situ->Q8_1!
// =============================================================================
extern "C" __global__ __launch_bounds__(576, 1)
void ink_persistent_decode_megakernel(
    const InkModelDesc* __restrict__ m_g,
    volatile InkRunCtrl* __restrict__ ctrl
) {
    __shared__ __align__(16) int              s_xq[9216 / 4];               // 9,216 B (fits 3*2304=6912 & 8*1152=9216!)
    __shared__ __align__(16) float2           s_xds[9216 / 32];             // 2,304 B
    __shared__ __align__(16) uint64_t         s_iq2_grid[256];              // 2,048 B
    __shared__ __align__(16) uint8_t          s_iq2_signs[128];             //   128 B
    __shared__ __align__(16) uint32_t         s_sign_mask4[16];             //    64 B
    __shared__ __align__(16) int32_t          s_q2_lut[256];                // 1,024 B
    __shared__ __align__(16) uint8_t          s_iq4_u8[16];                 //    16 B
    __shared__ __align__(16) float            s_scratch[2432];              // 9,728 B (64 red + 2304 vec + 64 spare)
    __shared__ __align__(16) int32_t          s_idx_red[32];
    __shared__ __align__(16) InkModelDesc     s_m;                          //   576 B
    __shared__ __align__(16) InkPhysLayerDesc s_L;                          //   256 B
    __shared__ __align__(16) InkLoopCtrlDesc  s_lc;                         //   128 B
    __shared__ __align__(16) int32_t          s_slot_kv[48];                //   192 B
    __shared__ int      s_abort;
    __shared__ int      s_cur_tok;
    __shared__ int      s_cur_pos;
    __shared__ int      s_max_steps;
    __shared__ int      s_eff_passes;
    __shared__ float    s_temp;
    __shared__ uint32_t s_step_off;
    __shared__ uint64_t s_seed;
    __shared__ int      s_export_taps;
    __shared__ int      s_done;
    __shared__ int      s_contig;
    __shared__ uint64_t s_prof[10];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const uint32_t hmask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const int hw_in_cta = (wid << 1) | (lane >> 4);
    const int total_warps = num_sms * INK_WARPS;
    const int total_hwarps = num_sms * (INK_WARPS * 2);

    ink_init_smem_luts(s_iq2_grid, s_iq2_signs, s_sign_mask4, s_q2_lut, s_iq4_u8);

    // Load `InkModelDesc` into SMEM so zero workspace pointers live in registers across barriers
    for (int i = tid; i < (int)(sizeof(InkModelDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_m)[i] = __ldg(&reinterpret_cast<const uint32_t*>(m_g)[i]);
    }
    if (tid < 10) s_prof[tid] = 0ULL;
    if (tid == 0) {
        s_abort = 0;
        s_done = 0;
        s_cur_tok = ctrl->start_token;
        s_cur_pos = ctrl->start_pos;
        s_max_steps = ctrl->max_steps;
        s_eff_passes = ctrl->eff_passes;
        s_temp = ctrl->temperature;
        s_step_off = ctrl->step_offset;
        s_seed = ctrl->rng_seed;
        s_export_taps = ctrl->export_taps;
        s_contig = ctrl->_pad_ctrl[0];
    }
    __syncthreads();
    const int global_w_ilv  = (s_contig != 0) ? (bid * INK_WARPS + wid) : (wid * num_sms + bid);
    const int global_hw_ilv = (s_contig != 0) ? (bid * (INK_WARPS * 2) + hw_in_cta) : (hw_in_cta * num_sms + bid);
    if (tid < 48) {
        s_slot_kv[tid] = __ldg(&reinterpret_cast<const int32_t*>(s_m.slot_kv_dense_ptr)[tid]);
    }
    __syncthreads();

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(s_m.ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    const int n_phys = s_m.n_phys_layers;
    const int l_start = s_m.loop_start;
    const int l_end = s_m.loop_end;
    const int eff_passes = s_eff_passes;
    const int total_visits = n_phys + (eff_passes - 1) * (l_end - l_start);

    for (int step_idx = 0; step_idx < s_max_steps; ++step_idx) {
        if (bid == 0 && tid == 0) s_prof[9] = ink_gtimer();
        const int tok_id = s_cur_tok;
        const int pos = s_cur_pos;

        // 1. Token Embedding + `tok_norm` (`CTA 0`)
        if (bid == 0) {
            const int dim = s_m.dim;
            float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);
            const uint8_t* emb_w = reinterpret_cast<const uint8_t*>(s_m.tok_embd_va);
            const float* tnorm = reinterpret_cast<const float*>(s_m.tok_norm_va);
            float ss = 0.0f;
            for (int col = tid; col < dim; col += INK_THREADS) {
                float v = ink_dequant_elem_emb(emb_w, s_m.tok_embd_type, (size_t)tok_id, dim, col);
                st_cg_f32(&ws_cur[col], v);
                ss += v * v;
            }
            ss = ink_warp_sum(ss);
            if (lane == 0) s_scratch[wid] = ss;
            __syncthreads();
            if (wid == 0) {
                float tot = (lane < INK_WARPS) ? s_scratch[lane] : 0.0f;
                tot = ink_warp_sum(tot);
                if (lane == 0) s_scratch[0] = rsqrtf(tot / (float)dim + s_m.rms_eps);
            }
            __syncthreads();
            const float sc = s_scratch[0];
            for (int col = tid; col < dim; col += INK_THREADS) {
                st_cg_f32(&ws_cur[col], ld_cg_f32(&ws_cur[col]) * sc * __ldg(&tnorm[col]));
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        int n_blocks = 0;
        int n_prelude_blocks = 0;
        int n_bank = 0;

        // 2. Unified 48-Visit Looped Schedule (`visit = 0 .. total_visits - 1`)
        for (int visit = 0; visit < total_visits; ++visit) {
            int phys_l = 0;
            int cache_slot = 0;
            bool has_loop_msgs = false;

            if (visit < l_start) {
                phys_l = visit;
                cache_slot = visit;
            } else if (visit < l_start + eff_passes * (l_end - l_start)) {
                const int core_off = visit - l_start;
                const int pass = core_off / (l_end - l_start);
                const int ip = l_start + (core_off % (l_end - l_start));

                if (core_off == 0) {
                    const int dim = s_m.dim;
                    n_prelude_blocks = n_blocks;
                    float* ws_prelude = reinterpret_cast<float*>(s_m.ws_prelude);
                    const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
                    float* ws_pre_blks = reinterpret_cast<float*>(s_m.ws_prelude_blocks);
                    const float* ws_blocks = reinterpret_cast<const float*>(s_m.ws_blocks);
                    for (int idx = bid * INK_THREADS + tid; idx < dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_prelude[idx], ld_cg_f32(&ws_cur[idx]));
                    }
                    const int pb_elems = n_prelude_blocks * dim;
                    for (int idx = bid * INK_THREADS + tid; idx < pb_elems; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_pre_blks[idx], ld_cg_f32(&ws_blocks[idx]));
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                } else if (ip == l_start && pass > 0) {
                    const int dim = s_m.dim;
                    const int bank_src_idx = n_prelude_blocks + 1;
                    n_bank = (n_blocks > bank_src_idx) ? (n_blocks - bank_src_idx) : 0;
                    float* ws_bank = reinterpret_cast<float*>(s_m.ws_bank);
                    float* ws_bank_out = reinterpret_cast<float*>(s_m.ws_bank_out);
                    float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);
                    float* ws_blocks = reinterpret_cast<float*>(s_m.ws_blocks);
                    const float* ws_prelude = reinterpret_cast<const float*>(s_m.ws_prelude);
                    const float* ws_pre_blks = reinterpret_cast<const float*>(s_m.ws_prelude_blocks);
                    const int bk_elems = n_bank * dim;
                    for (int idx = bid * INK_THREADS + tid; idx < bk_elems; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_bank[idx], ld_cg_f32(&ws_blocks[bank_src_idx * dim + idx]));
                    }
                    for (int idx = bid * INK_THREADS + tid; idx < dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_bank_out[idx], ld_cg_f32(&ws_cur[idx]));
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                    for (int idx = bid * INK_THREADS + tid; idx < dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_cur[idx], ld_cg_f32(&ws_prelude[idx]));
                    }
                    const int pb_elems = n_prelude_blocks * dim;
                    for (int idx = bid * INK_THREADS + tid; idx < pb_elems; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_blocks[idx], ld_cg_f32(&ws_pre_blks[idx]));
                    }
                    n_blocks = n_prelude_blocks;
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                }

                phys_l = ip;
                cache_slot = (pass == 0) ? ip : (n_phys + (pass - 1) * (l_end - l_start) + (ip - l_start));
                has_loop_msgs = (pass > 0);
            } else {
                const int ep_il = l_end + (visit - (l_start + eff_passes * (l_end - l_start)));
                phys_l = ep_il;
                cache_slot = ep_il;
            }

            // Load `s_L` (256 B) and `s_lc` (128 B) into SMEM in 1 instruction per lane!
            if (tid < (int)(sizeof(InkPhysLayerDesc) / 4)) {
                const uint32_t* lp = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const InkPhysLayerDesc*>(s_m.layers_ptr) + phys_l);
                reinterpret_cast<uint32_t*>(&s_L)[tid] = __ldg(&lp[tid]);
            }
            if (has_loop_msgs && tid >= 64 && tid < 64 + (int)(sizeof(InkLoopCtrlDesc) / 4)) {
                const uint32_t* lcp = reinterpret_cast<const uint32_t*>(
                    reinterpret_cast<const InkLoopCtrlDesc*>(s_m.loop_ctrls_ptr) + (phys_l - l_start));
                reinterpret_cast<uint32_t*>(&s_lc)[tid - 64] = __ldg(&lcp[tid - 64]);
            }
            __syncthreads();

            // Optional DSpark tap capture before visits 5, 23, 41
            if (s_export_taps != 0 && (visit == 5 || visit == 23 || visit == 41)) {
                const int dim = s_m.dim;
                const int tap_slot = (visit == 5) ? 0 : ((visit == 23) ? 1 : 2);
                float* dst_tap = reinterpret_cast<float*>(s_m.ws_dspark_taps) + (size_t)tap_slot * dim;
                const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
                for (int idx = bid * INK_THREADS + tid; idx < dim; idx += num_sms * INK_THREADS) {
                    st_cg_f32(&dst_tap[idx], ld_cg_f32(&ws_cur[idx]));
                }
            }

            // --- Substage L1: SMEM-Fused `depth_mix` + `rmsnorm_q8_1` (or Loop Controllers) ---
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[0] += tn - s_prof[9]; s_prof[9] = tn; }
            bool s_xq_preloaded = false;
            {
                const int dim = s_m.dim;
                const bool is_swa = (s_L.is_swa != 0);
                const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
                float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);

                if (!has_loop_msgs) {
                    if (is_swa) {
                        // 20/24 non-loop visits: 100% on-chip SMEM DepthMix + RMSNorm+Q8_1 -> ZERO grid barrier!
                        if (s_L.attn_res_query != 0) {
                            dev_cta_depth_mix_single(
                                s_scratch + 64,
                                s_scratch,
                                reinterpret_cast<const float*>(s_m.ws_blocks),
                                ws_cur,
                                reinterpret_cast<const float*>(s_L.attn_res_query),
                                dim,
                                n_blocks,
                                1e-6f,
                                true
                            );
                            __syncthreads();
                            dev_smem_rmsnorm_q8_1_inplace(
                                false,
                                s_xq,
                                s_xds,
                                s_scratch,
                                s_scratch + 64,
                                reinterpret_cast<const float*>(s_L.attn_norm),
                                dim,
                                s_m.rms_eps
                            );
                        } else {
                            dev_cta_rmsnorm_q8_1_to_smem(
                                nullptr,
                                s_xq,
                                s_xds,
                                s_scratch,
                                ws_cur,
                                reinterpret_cast<const float*>(s_L.attn_norm),
                                dim,
                                s_m.rms_eps,
                                true
                            );
                        }
                        s_xq_preloaded = true;
                    } else {
                        if (bid < 3) {
                            const float* raw_in = ws_cur;
                            if (s_L.attn_res_query != 0) {
                                const float* qv = reinterpret_cast<const float*>(s_L.attn_res_query) + (size_t)bid * dim;
                                dev_cta_depth_mix_single(
                                    ws_mixed + (size_t)bid * dim,
                                    s_scratch,
                                    reinterpret_cast<const float*>(s_m.ws_blocks),
                                    ws_cur,
                                    qv,
                                    dim,
                                    n_blocks,
                                    1e-6f
                                );
                                __syncthreads();
                                raw_in = ws_mixed + (size_t)bid * dim;
                            }
                            const float* nw = reinterpret_cast<const float*>(
                                (bid == 0) ? s_L.attn_q_inp_norm : ((bid == 1) ? s_L.attn_k_inp_norm : s_L.attn_v_inp_norm));
                            dev_cta_rmsnorm_q8_1(
                                nullptr,
                                reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)bid * 4608,
                                reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)bid * 144,
                                s_scratch,
                                raw_in,
                                nw,
                                dim,
                                s_m.rms_eps,
                                true
                            );
                        }
                        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                    }
                } else {
                    // Parallel `CTAs 0..n_q_mix-1` (`ws_mixed`) + `CTAs 4..7` (`ws_messages[0..3]`) in 1 single wave!
                    const int n_q_mix_lp = is_swa ? 1 : 3;
                    if (s_L.attn_res_query != 0 && bid < n_q_mix_lp) {
                        dev_cta_depth_mix_single(
                            ws_mixed + (size_t)bid * dim,
                            s_scratch,
                            reinterpret_cast<const float*>(s_m.ws_blocks),
                            ws_cur,
                            reinterpret_cast<const float*>(s_L.attn_res_query) + (size_t)bid * dim,
                            dim,
                            n_blocks,
                            1e-6f
                        );
                    } else if (bid >= 4 && bid < 8) {
                        const int mq = bid - 4;
                        dev_cta_depth_mix_single(
                            reinterpret_cast<float*>(s_m.ws_messages) + (size_t)mq * dim,
                            s_scratch,
                            reinterpret_cast<const float*>(s_m.ws_bank),
                            reinterpret_cast<const float*>(s_m.ws_bank_out),
                            reinterpret_cast<const float*>(s_lc.msg_query) + (size_t)mq * dim,
                            dim,
                            n_bank,
                            1e-6f
                        );
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    // Fused RMSNorm + `Q8_1` of `[raw_in, msg_j]` (`0` `ws_ctrl_cat` round-trip!) on `CTAs 0, 1, 2`
                    if (bid < 3) {
                        const int j = bid;
                        const float* raw_in = (s_L.attn_res_query != 0)
                            ? (ws_mixed + (size_t)(is_swa ? 0 : j) * dim)
                            : ws_cur;
                        const float* msg_j = reinterpret_cast<const float*>(s_m.ws_messages) + (size_t)j * dim;
                        float ssc = 0.0f, ssm = 0.0f;
                        for (int col = tid; col < dim; col += INK_THREADS) {
                            float c = ld_cg_f32(&raw_in[col]);
                            float mv = ld_cg_f32(&msg_j[col]);
                            ssc += c * c;
                            ssm += mv * mv;
                        }
                        ssc = ink_warp_sum(ssc);
                        ssm = ink_warp_sum(ssm);
                        if (lane == 0) { s_scratch[wid] = ssc; s_scratch[16 + wid] = ssm; }
                        __syncthreads();
                        if (wid == 0) {
                            float tc = (lane < INK_WARPS) ? s_scratch[lane] : 0.0f;
                            float tm = (lane < INK_WARPS) ? s_scratch[16 + lane] : 0.0f;
                            tc = ink_warp_sum(tc);
                            tm = ink_warp_sum(tm);
                            if (lane == 0) {
                                s_scratch[0] = rsqrtf(tc / (float)dim + 1e-6f);
                                s_scratch[1] = rsqrtf(tm / (float)dim + 1e-6f);
                            }
                        }
                        __syncthreads();
                        const float sc = s_scratch[0], sm = s_scratch[1];
                        int8_t* g_xq_j = reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)j * 4608;
                        float2* g_ds_j = reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)j * 144;
                        const int n_blks_half = dim >> 5;
                        for (int blk = wid; blk < n_blks_half; blk += INK_WARPS) {
                            const int col = (blk << 5) + lane;
                            const float vc = ld_cg_f32(&raw_in[col]) * sc;
                            const float vm = ld_cg_f32(&msg_j[col])  * sm;
                            float sum_c = vc, amax_c = fabsf(vc);
                            float sum_m = vm, amax_m = fabsf(vm);
                            #pragma unroll
                            for (int off = 16; off > 0; off >>= 1) {
                                sum_c += __shfl_xor_sync(0xffffffff, sum_c, off);
                                amax_c = fmaxf(amax_c, __shfl_xor_sync(0xffffffff, amax_c, off));
                                sum_m += __shfl_xor_sync(0xffffffff, sum_m, off);
                                amax_m = fmaxf(amax_m, __shfl_xor_sync(0xffffffff, amax_m, off));
                            }
                            const float dc = amax_c * (1.0f / 127.0f), idc = (amax_c > 0.0f) ? (127.0f / amax_c) : 0.0f;
                            const float dm = amax_m * (1.0f / 127.0f), idm = (amax_m > 0.0f) ? (127.0f / amax_m) : 0.0f;
                            st_cg_s8(&g_xq_j[col],       max(-127, min(127, __float2int_rn(vc * idc))));
                            st_cg_s8(&g_xq_j[dim + col], max(-127, min(127, __float2int_rn(vm * idm))));
                            if (lane == 0) {
                                st_cg_f32x2(&g_ds_j[blk],               make_float2(dc, sum_c));
                                st_cg_f32x2(&g_ds_j[n_blks_half + blk], make_float2(dm, sum_m));
                            }
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    // 4-Way K-Sliced `ctrl_down [3 * 64]` across `24` CTAs (`32` half-warps/CTA -> `4x` faster!)
                    const int gate_rank = s_m.gate_rank;
                    if (bid < 24) {
                        const int j = bid >> 3;             // 0..2
                        const int k_sl = (bid >> 1) & 3;    // 0..3
                        const int r_sub = bid & 1;          // 0..1
                        dev_load_q8_1_smem(
                            s_xq,
                            s_xds,
                            reinterpret_cast<const int8_t*>(s_m.ws_xq) + (size_t)j * 4608,
                            reinterpret_cast<const float2*>(s_m.ws_xds) + (size_t)j * 144,
                            2 * dim
                        );
                        const int r = r_sub * 32 + hw_in_cta;
                        const int slice_i4 = (2 * dim) >> 6; // 288 / 4 = 72 int4s
                        if (r < gate_rank) {
                            const float pv = dev_hwarp_dot_q8_0_slice(
                                reinterpret_cast<const uint8_t*>(s_lc.ctrl_down[j]),
                                (size_t)gate_rank,
                                (size_t)r,
                                s_xq,
                                s_xds,
                                2 * dim,
                                k_sl * slice_i4,
                                (k_sl + 1) * slice_i4,
                                lane16,
                                hmask
                            );
                            if (lane16 == 0) {
                                st_cg_f32(&reinterpret_cast<float*>(s_m.ws_ctrl_cat)[(j * 4 + k_sl) * gate_rank + r], pv);
                            }
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    // Reduce 4 K-slices of `ctrl_down` & quantize `[3 * 64 = 192]` directly into SMEM `s_xq, s_xds` (`0` barrier!)
                    if (wid < 6) {
                        const int j = wid >> 1;
                        const int r = ((wid & 1) << 5) + lane;
                        const float* p_cat = reinterpret_cast<const float*>(s_m.ws_ctrl_cat) + j * 4 * gate_rank;
                        const float v = (ld_cg_f32(&p_cat[0 * gate_rank + r]) + ld_cg_f32(&p_cat[1 * gate_rank + r]))
                                      + (ld_cg_f32(&p_cat[2 * gate_rank + r]) + ld_cg_f32(&p_cat[3 * gate_rank + r]));
                        float sum_x = v, amax = fabsf(v);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                        }
                        const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        reinterpret_cast<int8_t*>(s_xq)[wid * 32 + lane] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
                        if (lane == 0) s_xds[wid] = make_float2(d, sum_x);
                    }
                    __syncthreads();

                    // 4-Lane `K=64` Contiguous Coalesced `ctrl_gain` + `ctrl_forget` (`8,704` rows/wave = 1 single wave!)
                    {
                        const int lane4 = lane & 3;
                        const uint32_t mask4 = 0x0fu << (lane & 28);
                        for (int item = bid * 128 + (tid >> 2); item < 3 * dim; item += num_sms * 128) {
                            const int j = item / dim;
                            const int row = item - j * dim;
                            const int* sx_j = s_xq + j * (gate_rank >> 2);
                            const float2* sds_j = s_xds + j * (gate_rank >> 5);
                            const float g = dev_4lane_dot_q8_0_k64(
                                reinterpret_cast<const uint8_t*>(s_lc.ctrl_gain[j]), (size_t)dim, (size_t)row, sx_j, sds_j, lane4, mask4);
                            const float f = dev_4lane_dot_q8_0_k64(
                                reinterpret_cast<const uint8_t*>(s_lc.ctrl_forget[j]), (size_t)dim, (size_t)row, sx_j, sds_j, lane4, mask4);
                            if (lane4 == 0) {
                                const float* raw_in = (s_L.attn_res_query != 0)
                                    ? (ws_mixed + (size_t)(is_swa ? 0 : j) * dim)
                                    : ws_cur;
                                const float* msg_j = reinterpret_cast<const float*>(s_m.ws_messages) + (size_t)j * dim;
                                st_cg_f32(
                                    &reinterpret_cast<float*>(s_m.ws_inputs)[(size_t)j * dim + row],
                                    (1.0f - tanhf(f)) * ld_cg_f32(&raw_in[row]) + tanhf(g) * ld_cg_f32(&msg_j[row])
                                );
                            }
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    // 1-Pass Parallel 3-Stream SMEM RMSNorm + `Q8_1` (`warps 0..4` -> j=0, `5..9` -> j=1, `10..14` -> j=2)
                    {
                        const int n_blks_d = dim >> 5; // 72
                        if (wid < 15) {
                            const int j = wid / 5;
                            const int sub_w = wid - j * 5;
                            const float* in_j = reinterpret_cast<const float*>(s_m.ws_inputs) + (size_t)j * dim;
                            float ss = 0.0f;
                            for (int blk = sub_w; blk < n_blks_d; blk += 5) {
                                float v = ld_cg_f32(&in_j[(blk << 5) + lane]);
                                ss += v * v;
                            }
                            ss = ink_warp_sum(ss);
                            if (lane == 0) s_scratch[wid] = ss;
                        }
                        __syncthreads();
                        if (wid == 0 && lane < 3) {
                            const int base = lane * 5;
                            float tot = s_scratch[base + 0] + s_scratch[base + 1] + s_scratch[base + 2] + s_scratch[base + 3] + s_scratch[base + 4];
                            s_scratch[16 + lane] = rsqrtf(tot / (float)dim + s_m.rms_eps);
                        }
                        __syncthreads();
                        if (wid < 15) {
                            const int j = wid / 5;
                            const int sub_w = wid - j * 5;
                            const float sc = s_scratch[16 + j];
                            const float* in_j = reinterpret_cast<const float*>(s_m.ws_inputs) + (size_t)j * dim;
                            const float* nw = is_swa
                                ? reinterpret_cast<const float*>(s_L.attn_norm)
                                : reinterpret_cast<const float*>((j == 0) ? s_L.attn_q_inp_norm : ((j == 1) ? s_L.attn_k_inp_norm : s_L.attn_v_inp_norm));
                            int8_t* sx_j = reinterpret_cast<int8_t*>(s_xq + j * (dim >> 2));
                            float2* sds_j = s_xds + j * n_blks_d;
                            for (int blk = sub_w; blk < n_blks_d; blk += 5) {
                                const int col = (blk << 5) + lane;
                                const float v = ld_cg_f32(&in_j[col]) * sc * __ldg(&nw[col]);
                                float sum_x = v, amax = fabsf(v);
                                #pragma unroll
                                for (int off = 16; off > 0; off >>= 1) {
                                    sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                                }
                                const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                                sx_j[col] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
                                if (lane == 0) sds_j[blk] = make_float2(d, sum_x);
                            }
                        }
                        __syncthreads();
                    }
                    s_xq_preloaded = true;
                }
            }

            // --- Substage L2: ONE Unified Loop over ALL 8,536/8,600 Attention `Q8_0` Rows across ALL 68 SMs ---
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[1] += tn - s_prof[9]; s_prof[9] = tn; }
            {
                const int dim = s_m.dim;
                const int q_dim = s_m.n_heads * s_m.head_dim;   // 3072
                const int kv_dim = s_m.n_kv_heads * s_m.head_dim; // 1024
                const int r_dim = s_m.n_heads * s_m.rel_len;    // 384
                const bool is_swa = (s_L.is_swa != 0);
                const bool share_qkv_in = is_swa && !has_loop_msgs;
                const int n_blks = dim >> 5;

                if (!s_xq_preloaded) {
                    const int n_streams = share_qkv_in ? 1 : 3;
                    const int n_int4 = dim >> 4;
                    for (int s = 0; s < n_streams; ++s) {
                        const int* g_i32 = reinterpret_cast<const int*>(reinterpret_cast<const int8_t*>(s_m.ws_xq) + (size_t)s * 4608);
                        const float2* g_ds = reinterpret_cast<const float2*>(s_m.ws_xds) + (size_t)s * 144;
                        int4* s_i4 = reinterpret_cast<int4*>(s_xq + s * (dim >> 2));
                        float2* s_ds = s_xds + s * n_blks;
                        for (int i = tid; i < n_int4; i += INK_THREADS) s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
                        for (int b = tid; b < n_blks; b += INK_THREADS) s_ds[b] = ld_cg_f32x2(&g_ds[b]);
                    }
                    __syncthreads();
                }

                const int q_grp = q_dim + r_dim + s_m.n_heads + q_dim; // 6552
                const int k_grp = kv_dim + (is_swa ? 0 : 32);          // 1024 or 1056
                const int tot_rows = q_grp + 2 * k_grp;                // 8600 or 8664

                for (int r = global_hw_ilv; r < tot_rows; r += total_hwarps) {
                    const uint8_t* w_ptr;
                    float* dst_ptr;
                    int local_r, n_rows_m, s_idx;

                    if (r < q_grp) {
                        s_idx = 0;
                        if (r < q_dim) {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.wq);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_q);
                            local_r = r; n_rows_m = q_dim;
                        } else if (r < q_dim + r_dim) {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.attn_rel);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_r);
                            local_r = r - q_dim; n_rows_m = r_dim;
                        } else if (r < q_dim + r_dim + s_m.n_heads) {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.attn_sink_proj);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_sinks);
                            local_r = r - (q_dim + r_dim); n_rows_m = s_m.n_heads;
                        } else {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.wqkv_gate);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_gate_raw);
                            local_r = r - (q_dim + r_dim + s_m.n_heads); n_rows_m = q_dim;
                        }
                    } else if (r < q_grp + k_grp) {
                        s_idx = share_qkv_in ? 0 : 1;
                        const int kr = r - q_grp;
                        if (kr < kv_dim) {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.wk);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_k_raw);
                            local_r = kr; n_rows_m = kv_dim;
                        } else {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.attn_k_conv_proj);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_dyn_k);
                            local_r = kr - kv_dim; n_rows_m = 32;
                        }
                    } else {
                        s_idx = share_qkv_in ? 0 : 2;
                        const int vr = r - (q_grp + k_grp);
                        if (vr < kv_dim) {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.wv);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_v_raw);
                            local_r = vr; n_rows_m = kv_dim;
                        } else {
                            w_ptr = reinterpret_cast<const uint8_t*>(s_L.attn_v_conv_proj);
                            dst_ptr = reinterpret_cast<float*>(s_m.ws_dyn_v);
                            local_r = vr - kv_dim; n_rows_m = 32;
                        }
                    }

                    const float val = dev_hwarp_dot_q8_0_aligned(
                        w_ptr,
                        (size_t)n_rows_m,
                        (size_t)local_r,
                        s_xq + s_idx * (dim >> 2),
                        s_xds + s_idx * n_blks,
                        dim,
                        lane16,
                        hmask
                    );
                    if (lane16 == 0) st_cg_f32(&dst_ptr[local_r], val);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[2] += tn - s_prof[9]; s_prof[9] = tn; }

            // --- Substage L3: `K/V ShortConv` + `K-Clip / V-RMSNorm` + `FP16 KV Store` (`CTAs 0..15`, `swa_cap = 2*swa_window`) ---
            if (bid < 16) {
                const bool is_swa = (s_L.is_swa != 0);
                const int dim = s_m.dim;
                const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
                const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
                const int dense_idx = s_slot_kv[cache_slot];
                const int swa_cap = 2 * s_m.swa_window;
                const size_t kv_stride = (size_t)(is_swa ? swa_cap : s_m.max_ctx) * (size_t)kv_dim;
                half* k_cache_slot = (is_swa ? reinterpret_cast<half*>(s_m.swa_k_va) : reinterpret_cast<half*>(s_m.full_k_va)) + (size_t)dense_idx * kv_stride;
                half* v_cache_slot = (is_swa ? reinterpret_cast<half*>(s_m.swa_v_va) : reinterpret_cast<half*>(s_m.full_v_va)) + (size_t)dense_idx * kv_stride;
                const int cache_wr = is_swa ? (pos & (swa_cap - 1)) : pos;
                float* slot_conv = reinterpret_cast<float*>(s_m.conv_state_va) + (size_t)cache_slot * conv_ch_per_slot * 3ULL;

                if (bid < 8) {
                    const int kv_h = bid;
                    float* k_conv_st = slot_conv;
                    const float* ws_k_raw = reinterpret_cast<const float*>(s_m.ws_k_raw);
                    const float* ws_dyn_k = reinterpret_cast<const float*>(s_m.ws_dyn_k);
                    float kv = 0.0f;
                    if (tid < 128) {
                        const int c = kv_h * 128 + tid;
                        float* st = k_conv_st + (size_t)c * 3;
                        const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = ld_cg_f32(&ws_k_raw[c]);
                        st[0] = x1; st[1] = x2; st[2] = x3;
                        const float* cw = reinterpret_cast<const float*>(s_L.attn_k_conv) + (size_t)c * 4;
                        float csum = x0 * __ldg(&cw[0]) + x1 * __ldg(&cw[1]) + x2 * __ldg(&cw[2]) + x3 * __ldg(&cw[3]);
                        float inv_s = 0.4472135955f;
                        if (!is_swa) {
                            const float* d = ws_dyn_k + kv_h * 4;
                            csum += ld_cg_f32(&d[0]) * x3 + ld_cg_f32(&d[1]) * x2 + ld_cg_f32(&d[2]) * x1 + ld_cg_f32(&d[3]) * x0;
                            inv_s = 0.3333333333f;
                        }
                        kv = (x3 + csum) * inv_s;
                    }
                    float ss = ink_warp_sum(kv * kv);
                    if (wid < 4 && lane == 0) s_scratch[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float tot = (lane < 4) ? s_scratch[lane] : 0.0f;
                        tot += __shfl_down_sync(0xffffffff, tot, 2);
                        tot += __shfl_down_sync(0xffffffff, tot, 1);
                        if (lane == 0) s_scratch[0] = fminf(1.0f, fmaxf(0.0f, s_m.qk_clip / (sqrtf(tot) + 1e-6f)));
                    }
                    __syncthreads();
                    if (tid < 128) k_cache_slot[(size_t)cache_wr * 1024 + kv_h * 128 + tid] = __float2half(kv * s_scratch[0]);
                } else {
                    const int kv_h = bid - 8;
                    float* v_conv_st = slot_conv + (size_t)kv_dim * 3ULL;
                    const float* ws_v_raw = reinterpret_cast<const float*>(s_m.ws_v_raw);
                    const float* ws_dyn_v = reinterpret_cast<const float*>(s_m.ws_dyn_v);
                    float vv = 0.0f;
                    if (tid < 128) {
                        const int c = kv_h * 128 + tid;
                        float* st = v_conv_st + (size_t)c * 3;
                        const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = ld_cg_f32(&ws_v_raw[c]);
                        st[0] = x1; st[1] = x2; st[2] = x3;
                        const float* cw = reinterpret_cast<const float*>(s_L.attn_v_conv) + (size_t)c * 4;
                        float csum = x0 * __ldg(&cw[0]) + x1 * __ldg(&cw[1]) + x2 * __ldg(&cw[2]) + x3 * __ldg(&cw[3]);
                        float inv_s = 0.4472135955f;
                        if (!is_swa) {
                            const float* d = ws_dyn_v + kv_h * 4;
                            csum += ld_cg_f32(&d[0]) * x3 + ld_cg_f32(&d[1]) * x2 + ld_cg_f32(&d[2]) * x1 + ld_cg_f32(&d[3]) * x0;
                            inv_s = 0.3333333333f;
                        }
                        vv = (x3 + csum) * inv_s;
                    }
                    float ss = ink_warp_sum(vv * vv);
                    if (wid < 4 && lane == 0) s_scratch[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float tot = (lane < 4) ? s_scratch[lane] : 0.0f;
                        tot += __shfl_down_sync(0xffffffff, tot, 2);
                        tot += __shfl_down_sync(0xffffffff, tot, 1);
                        if (lane == 0) s_scratch[0] = rsqrtf(tot * (1.0f / 128.0f) + s_m.rms_eps);
                    }
                    __syncthreads();
                    if (tid < 128) v_cache_slot[(size_t)cache_wr * 1024 + kv_h * 128 + tid] = __float2half(vv * s_scratch[0]);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            // --- Substage L4: In-Register `Q/R` Clip + 16-Warp/Head Rel+Sink GQA Attention + Fused `Q8_1` (`CTAs 0..23`) ---
            if (bid < 24) {
                const bool is_swa = (s_L.is_swa != 0);
                const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
                const int dense_idx = s_slot_kv[cache_slot];
                const int swa_cap = 2 * s_m.swa_window;
                const size_t kv_stride = (size_t)(is_swa ? swa_cap : s_m.max_ctx) * (size_t)kv_dim;
                const half* k_cache_slot = (is_swa ? reinterpret_cast<const half*>(s_m.swa_k_va) : reinterpret_cast<const half*>(s_m.full_k_va)) + (size_t)dense_idx * kv_stride;
                const half* v_cache_slot = (is_swa ? reinterpret_cast<const half*>(s_m.swa_v_va) : reinterpret_cast<const half*>(s_m.full_v_va)) + (size_t)dense_idx * kv_stride;

                const int h = bid;
                const int kv_h = h / 3;
                float* s_acc = s_scratch;                 // [16 * 128 = 2048 floats]
                float* s_m_arr = s_scratch + 16 * 128;    // [16]
                float* s_l_arr = s_scratch + 16 * 128 + 16; // [16]
                float* s_qr  = s_scratch + 16 * 128 + 32; // [16]

                float tau = 1.0f;
                if (!is_swa) {
                    float ratio = fmaxf(1.0f, (float)(pos + 1) / (float)s_m.log_scale_floor);
                    tau = 1.0f + s_m.log_scale_alpha * logf(ratio);
                }

                const float* ws_q = reinterpret_cast<const float*>(s_m.ws_q);
                const float* ws_r = reinterpret_cast<const float*>(s_m.ws_r);
                const float* ws_sinks = reinterpret_cast<const float*>(s_m.ws_sinks);
                const float* qc_w = reinterpret_cast<const float*>(s_L.attn_q_clip) + h * 128;
                const int d0 = lane * 4;
                float q0 = ld_cg_f32(&ws_q[h * 128 + d0 + 0]) * __ldg(&qc_w[d0 + 0]);
                float q1 = ld_cg_f32(&ws_q[h * 128 + d0 + 1]) * __ldg(&qc_w[d0 + 1]);
                float q2 = ld_cg_f32(&ws_q[h * 128 + d0 + 2]) * __ldg(&qc_w[d0 + 2]);
                float q3 = ld_cg_f32(&ws_q[h * 128 + d0 + 3]) * __ldg(&qc_w[d0 + 3]);
                float qss = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) qss += __shfl_xor_sync(0xffffffff, qss, off);
                const float qfac = fminf(1.0f, fmaxf(0.0f, s_m.qk_clip / (sqrtf(qss) + 1e-6f))) * tau;
                q0 *= qfac; q1 *= qfac; q2 *= qfac; q3 *= qfac;

                if (wid == 0) {
                    const float* rc_w = reinterpret_cast<const float*>(s_L.attn_rel_clip) + h * 16;
                    float rv = (lane < 16) ? (ld_cg_f32(&ws_r[h * 16 + lane]) * __ldg(&rc_w[lane])) : 0.0f;
                    float rss = rv * rv;
                    #pragma unroll
                    for (int off = 8; off > 0; off >>= 1) rss += __shfl_xor_sync(0xffffffff, rss, off);
                    float rfac = fminf(1.0f, fmaxf(0.0f, s_m.rel_clip / (sqrtf(rss) + 1e-6f))) * tau;
                    if (lane < 16) s_qr[lane] = rv * rfac;
                }
                const float sink_h = ld_cg_f32(&ws_sinks[h]);
                __syncthreads();

                const int start_j = is_swa ? max(0, pos - s_m.swa_window + 1) : 0;
                const float inv_sqrt_hd = 0.0883883476f;
                const float* rel_bias_w = reinterpret_cast<const float*>(s_L.attn_rel_bias);

                float m_w = (wid == 0) ? 0.0f : -1e30f;
                float l_w = (wid == 0) ? 1.0f : 0.0f;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                for (int j = start_j + wid; j <= pos; j += INK_WARPS) {
                    const int slot_j = is_swa ? (j & (swa_cap - 1)) : j;
                    const half2* kj2 = reinterpret_cast<const half2*>(k_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
                    const half2 k01 = __ldg(&kj2[0]), k23 = __ldg(&kj2[1]);
                    float dot = q0 * __half2float(k01.x) + q1 * __half2float(k01.y)
                              + q2 * __half2float(k23.x) + q3 * __half2float(k23.y);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);

                    const int dist = pos - j;
                    float r_dot = 0.0f;
                    if (dist < s_m.rel_extent) {
                        float rp = (lane < 16) ? (s_qr[lane] * __ldg(&rel_bias_w[(size_t)dist * 16 + lane])) : 0.0f;
                        #pragma unroll
                        for (int off = 8; off > 0; off >>= 1) rp += __shfl_xor_sync(0xffffffff, rp, off);
                        r_dot = __shfl_sync(0xffffffff, rp, 0);
                    }

                    const float score = dot * inv_sqrt_hd + (r_dot - sink_h);
                    const float m_new = fmaxf(m_w, score);
                    const float alpha = expf(m_w - m_new);
                    const float beta  = expf(score - m_new);
                    l_w = l_w * alpha + beta;
                    m_w = m_new;

                    const half2* vj2 = reinterpret_cast<const half2*>(v_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
                    const half2 v01 = __ldg(&vj2[0]), v23 = __ldg(&vj2[1]);
                    acc0 = acc0 * alpha + beta * __half2float(v01.x);
                    acc1 = acc1 * alpha + beta * __half2float(v01.y);
                    acc2 = acc2 * alpha + beta * __half2float(v23.x);
                    acc3 = acc3 * alpha + beta * __half2float(v23.y);
                }

                if (lane == 0) { s_m_arr[wid] = m_w; s_l_arr[wid] = l_w; }
                s_acc[wid * 128 + d0 + 0] = acc0;
                s_acc[wid * 128 + d0 + 1] = acc1;
                s_acc[wid * 128 + d0 + 2] = acc2;
                s_acc[wid * 128 + d0 + 3] = acc3;
                __syncthreads();

                if (wid < 4) {
                    float m_max = -1e30f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_l_arr[w] > 0.0f) m_max = fmaxf(m_max, s_m_arr[w]);
                    }
                    float denom = 0.0f, out_d = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_l_arr[w] > 0.0f) {
                            float ew = expf(s_m_arr[w] - m_max);
                            denom += s_l_arr[w] * ew;
                            out_d += s_acc[w * 128 + tid] * ew;
                        }
                    }
                    const int col = h * 128 + tid;
                    const float gate = 1.0f / (1.0f + expf(-ld_cg_f32(&reinterpret_cast<const float*>(s_m.ws_gate_raw)[col])));
                    const float v = (out_d / fmaxf(denom, 1e-20f)) * gate;

                    float sum_x = v, amax = fabsf(v);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                    }
                    const float d = amax * (1.0f / 127.0f);
                    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    st_cg_s8(&reinterpret_cast<int8_t*>(s_m.ws_xq)[col], max(-127, min(127, __float2int_rn(v * id))));
                    if (lane == 0) st_cg_f32x2(&reinterpret_cast<float2*>(s_m.ws_xds)[h * 4 + wid], make_float2(d, sum_x));
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[3] += tn - s_prof[9]; s_prof[9] = tn; }

            // --- Substage L5: 4-Way K-Sliced `wo` (`9,216` Half-Warp Items -> Zero 4-SM Tail!) + `attn_post_conv` ---
            const bool push_blk = ((phys_l % s_m.attn_res_block) == 0);
            const int blk_write_idx = n_blocks;
            {
                const int dim = s_m.dim;
                const int q_dim = s_m.n_heads * s_m.head_dim; // 3072 -> 192 int4s = 4 * 48 int4s!
                dev_load_q8_1_smem(
                    s_xq,
                    s_xds,
                    reinterpret_cast<const int8_t*>(s_m.ws_xq),
                    reinterpret_cast<const float2*>(s_m.ws_xds),
                    q_dim
                );
                const uint8_t* wo_ptr = reinterpret_cast<const uint8_t*>(s_L.wo);
                float* ws_part = reinterpret_cast<float*>(s_m.ws_attn_gated); // [4 * 2304 = 9216 floats]
                const int slice_i4 = (q_dim >> 4) >> 2; // 48 int4s = 3 exact 16-lane iterations!

                for (int item = global_hw_ilv; item < 4 * dim; item += total_hwarps) {
                    const int k_part = item / dim;
                    const int c = item - k_part * dim;
                    const float pv = dev_hwarp_dot_q8_0_slice(
                        wo_ptr,
                        (size_t)dim,
                        (size_t)c,
                        s_xq,
                        s_xds,
                        q_dim,
                        k_part * slice_i4,
                        (k_part + 1) * slice_i4,
                        lane16,
                        hmask
                    );
                    if (lane16 == 0) st_cg_f32(&ws_part[k_part * dim + c], pv);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            {
                const int dim = s_m.dim;
                const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
                const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
                float* attn_p_st = reinterpret_cast<float*>(s_m.conv_state_va)
                                 + (size_t)cache_slot * conv_ch_per_slot * 3ULL
                                 + (size_t)(2 * kv_dim) * 3ULL;
                const float* pconv_w = reinterpret_cast<const float*>(s_L.attn_post_conv);
                const float* ws_part = reinterpret_cast<const float*>(s_m.ws_attn_gated);
                float* ws_blocks = reinterpret_cast<float*>(s_m.ws_blocks);
                float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);
                float* ws_ffn_inp = reinterpret_cast<float*>(s_m.ws_ffn_inp);

                for (int c = bid * INK_THREADS + tid; c < dim; c += num_sms * INK_THREADS) {
                    const float wo_v = (ld_cg_f32(&ws_part[0 * dim + c]) + ld_cg_f32(&ws_part[1 * dim + c]))
                                     + (ld_cg_f32(&ws_part[2 * dim + c]) + ld_cg_f32(&ws_part[3 * dim + c]));
                    float* st = attn_p_st + (size_t)c * 3;
                    const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = wo_v;
                    st[0] = x1; st[1] = x2; st[2] = x3;
                    const float* cw = pconv_w + (size_t)c * 4;
                    const float csum = x0 * __ldg(&cw[0]) + x1 * __ldg(&cw[1]) + x2 * __ldg(&cw[2]) + x3 * __ldg(&cw[3]);
                    const float attn_out = (x3 + csum) * 0.4472135955f;
                    const float isa = ld_cg_f32(&ws_cur[c]);
                    float new_cur;
                    if (push_blk) {
                        st_cg_f32(&ws_blocks[(size_t)blk_write_idx * dim + c], isa);
                        new_cur = attn_out;
                    } else {
                        new_cur = isa + attn_out;
                    }
                    st_cg_f32(&ws_cur[c], new_cur);
                    st_cg_f32(&ws_ffn_inp[c], new_cur);
                }
            }
            if (push_blk) n_blocks += 1;
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[4] += tn - s_prof[9]; s_prof[9] = tn; }

            // --- Substage L6: On-Chip `ffn_depth_mix` + Optional `loop_ctrl[3]` + SMEM `ffn_rmsnorm` + 64-CTA FP32 Router ---
            {
                const int dim = s_m.dim;
                const int gate_rank = s_m.gate_rank;
                float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);
                const float* ws_blocks = reinterpret_cast<const float*>(s_m.ws_blocks);
                const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);

                if (!has_loop_msgs) {
                    // Every CTA computes `depth_mix` + `ffn_norm` directly into SMEM `(s_scratch+64)` & `(s_xq, s_xds)` (`0` grid barrier!)
                    dev_cta_depth_mix_single(
                        s_scratch + 64, s_scratch, ws_blocks, ws_cur,
                        reinterpret_cast<const float*>(s_L.ffn_res_query), dim, n_blocks, 1e-6f, true);
                    __syncthreads();
                    dev_smem_rmsnorm_q8_1_inplace(
                        true,
                        s_xq, s_xds, s_scratch, s_scratch + 64,
                        reinterpret_cast<const float*>(s_L.ffn_norm), dim, s_m.rms_eps);
                } else {
                    if (bid == 0) {
                        dev_cta_depth_mix_single(
                            s_scratch + 64, s_scratch, ws_blocks, ws_cur,
                            reinterpret_cast<const float*>(s_L.ffn_res_query), dim, n_blocks, 1e-6f, true);
                        __syncthreads();
                        const float* msg_3 = reinterpret_cast<const float*>(s_m.ws_messages) + (size_t)3 * dim;
                        float ssc = 0.0f, ssm = 0.0f;
                        for (int col = tid; col < dim; col += INK_THREADS) {
                            float c = s_scratch[64 + col];
                            st_cg_f32(&ws_mixed[col], c);
                            float mv = ld_cg_f32(&msg_3[col]);
                            ssc += c * c;
                            ssm += mv * mv;
                        }
                        ssc = ink_warp_sum(ssc);
                        ssm = ink_warp_sum(ssm);
                        if (lane == 0) { s_scratch[wid] = ssc; s_scratch[16 + wid] = ssm; }
                        __syncthreads();
                        if (wid == 0) {
                            float tc = (lane < INK_WARPS) ? s_scratch[lane] : 0.0f;
                            float tm = (lane < INK_WARPS) ? s_scratch[16 + lane] : 0.0f;
                            tc = ink_warp_sum(tc); tm = ink_warp_sum(tm);
                            if (lane == 0) {
                                s_scratch[0] = rsqrtf(tc / (float)dim + 1e-6f);
                                s_scratch[1] = rsqrtf(tm / (float)dim + 1e-6f);
                            }
                        }
                        __syncthreads();
                        const float sc = s_scratch[0], sm = s_scratch[1];
                        int8_t* g_xq = reinterpret_cast<int8_t*>(s_m.ws_xq);
                        float2* g_ds = reinterpret_cast<float2*>(s_m.ws_xds);
                        const int n_blks_half = dim >> 5;
                        for (int blk = wid; blk < n_blks_half; blk += INK_WARPS) {
                            const int col = (blk << 5) + lane;
                            const float vc = s_scratch[64 + col] * sc;
                            const float vm = ld_cg_f32(&msg_3[col]) * sm;
                            float sum_c = vc, amax_c = fabsf(vc);
                            float sum_m = vm, amax_m = fabsf(vm);
                            #pragma unroll
                            for (int off = 16; off > 0; off >>= 1) {
                                sum_c += __shfl_xor_sync(0xffffffff, sum_c, off);
                                amax_c = fmaxf(amax_c, __shfl_xor_sync(0xffffffff, amax_c, off));
                                sum_m += __shfl_xor_sync(0xffffffff, sum_m, off);
                                amax_m = fmaxf(amax_m, __shfl_xor_sync(0xffffffff, amax_m, off));
                            }
                            const float dc = amax_c * (1.0f / 127.0f), idc = (amax_c > 0.0f) ? (127.0f / amax_c) : 0.0f;
                            const float dm = amax_m * (1.0f / 127.0f), idm = (amax_m > 0.0f) ? (127.0f / amax_m) : 0.0f;
                            st_cg_s8(&g_xq[col],       max(-127, min(127, __float2int_rn(vc * idc))));
                            st_cg_s8(&g_xq[dim + col], max(-127, min(127, __float2int_rn(vm * idm))));
                            if (lane == 0) {
                                st_cg_f32x2(&g_ds[blk],               make_float2(dc, sum_c));
                                st_cg_f32x2(&g_ds[n_blks_half + blk], make_float2(dm, sum_m));
                            }
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    if (bid < 8) {
                        const int k_sl = bid >> 1;
                        const int r_sub = bid & 1;
                        dev_load_q8_1_smem(
                            s_xq, s_xds,
                            reinterpret_cast<const int8_t*>(s_m.ws_xq),
                            reinterpret_cast<const float2*>(s_m.ws_xds),
                            2 * dim
                        );
                        const int r = r_sub * 32 + hw_in_cta;
                        const int slice_i4 = (2 * dim) >> 6;
                        if (r < gate_rank) {
                            float pv = dev_hwarp_dot_q8_0_slice(
                                reinterpret_cast<const uint8_t*>(s_lc.ctrl_down[3]),
                                (size_t)gate_rank, (size_t)r, s_xq, s_xds, 2 * dim,
                                k_sl * slice_i4, (k_sl + 1) * slice_i4, lane16, hmask);
                            if (lane16 == 0) st_cg_f32(&reinterpret_cast<float*>(s_m.ws_ctrl_cat)[k_sl * gate_rank + r], pv);
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    if (wid < 2) {
                        const int r = (wid << 5) + lane;
                        const float* p_cat = reinterpret_cast<const float*>(s_m.ws_ctrl_cat);
                        const float v = (ld_cg_f32(&p_cat[0 * gate_rank + r]) + ld_cg_f32(&p_cat[1 * gate_rank + r]))
                                      + (ld_cg_f32(&p_cat[2 * gate_rank + r]) + ld_cg_f32(&p_cat[3 * gate_rank + r]));
                        float sum_x = v, amax = fabsf(v);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                        }
                        const float d = amax * (1.0f / 127.0f), id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        reinterpret_cast<int8_t*>(s_xq)[r] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
                        if (lane == 0) s_xds[wid] = make_float2(d, sum_x);
                    }
                    __syncthreads();

                    const float* msg_3 = reinterpret_cast<const float*>(s_m.ws_messages) + (size_t)3 * dim;
                    const uint8_t* wg = reinterpret_cast<const uint8_t*>(s_lc.ctrl_gain[3]);
                    const uint8_t* wf = reinterpret_cast<const uint8_t*>(s_lc.ctrl_forget[3]);
                    float* ws_ctrl_gain = reinterpret_cast<float*>(s_m.ws_ctrl_gain);
                    {
                        const int lane4 = lane & 3;
                        const uint32_t mask4 = 0x0fu << (lane & 28);
                        for (int row = bid * 128 + (tid >> 2); row < dim; row += num_sms * 128) {
                            const float g = dev_4lane_dot_q8_0_k64(wg, (size_t)dim, (size_t)row, s_xq, s_xds, lane4, mask4);
                            const float f = dev_4lane_dot_q8_0_k64(wf, (size_t)dim, (size_t)row, s_xq, s_xds, lane4, mask4);
                            if (lane4 == 0) {
                                st_cg_f32(&ws_ctrl_gain[row], (1.0f - tanhf(f)) * ld_cg_f32(&ws_mixed[row]) + tanhf(g) * ld_cg_f32(&msg_3[row]));
                            }
                        }
                    }
                    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                    // Every CTA computes `ffn_norm` directly into SMEM `(s_scratch+64)` & `(s_xq, s_xds)` (`0` grid barrier!)
                    dev_cta_rmsnorm_q8_1_to_smem(
                        s_scratch + 64, s_xq, s_xds, s_scratch,
                        reinterpret_cast<const float*>(s_m.ws_ctrl_gain),
                        reinterpret_cast<const float*>(s_L.ffn_norm),
                        dim, s_m.rms_eps, true);
                }

                // 64-SM FP32 MoE Router reading `xs` from SMEM `s_scratch + 64`! (`s_xq, s_xds` stay intact for L7!)
                if (bid < 64) {
                    const int e_sub = wid >> 3;          // 0 or 1
                    const int sub_tid = tid & 255;       // 0..255 (8 warps per expert)
                    const int e = (bid << 1) + e_sub;    // 0..127
                    const float* rw = reinterpret_cast<const float*>(s_L.ffn_gate_inp) + (size_t)e * dim;
                    const float* xs = s_scratch + 64;
                    float sum = 0.0f;
                    for (int col = sub_tid * 4; col < dim; col += 1024) {
                        const float4 w4 = __ldg(reinterpret_cast<const float4*>(rw + col));
                        const float4 x4 = *reinterpret_cast<const float4*>(xs + col);
                        sum += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
                    }
                    sum = ink_warp_sum(sum);
                    if (lane == 0) s_scratch[wid] = sum;
                    __syncthreads();
                    if (wid == 0 && lane < 16) {
                        float v = s_scratch[lane];
                        v += __shfl_down_sync(0x0000ffffu, v, 4);
                        v += __shfl_down_sync(0x0000ffffu, v, 2);
                        v += __shfl_down_sync(0x0000ffffu, v, 1);
                        if ((lane & 7) == 0) {
                            st_cg_f32(&reinterpret_cast<float*>(s_m.ws_router_logits)[(bid << 1) + (lane >> 3)], v);
                        }
                    }
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[5] += tn - s_prof[9]; s_prof[9] = tn; }

            // --- Substage L7: Warp-0 On-Chip Top-8 (`0` barrier!) + Active-8 MoE `ffn_gate_up_exps` + `SITU` ---
            {
                const int dim = s_m.dim;
                const int n_ff = s_m.n_ff_exp;
                if (wid == 0) {
                    const float* ws_rtr = reinterpret_cast<const float*>(s_m.ws_router_logits);
                    const float* eb = reinterpret_cast<const float*>(s_L.exp_probs_b);
                    float p0 = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_rtr[lane + 0])));
                    float p1 = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_rtr[lane + 32])));
                    float p2 = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_rtr[lane + 64])));
                    float p3 = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_rtr[lane + 96])));
                    float s0 = p0 + (eb ? __ldg(&eb[lane + 0])  : 0.0f);
                    float s1 = p1 + (eb ? __ldg(&eb[lane + 32]) : 0.0f);
                    float s2 = p2 + (eb ? __ldg(&eb[lane + 64]) : 0.0f);
                    float s3 = p3 + (eb ? __ldg(&eb[lane + 96]) : 0.0f);

                    float w_sum = 1e-20f;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float bv = s0; int bi = lane;
                        if (s1 > bv) { bv = s1; bi = lane + 32; }
                        if (s2 > bv) { bv = s2; bi = lane + 64; }
                        if (s3 > bv) { bv = s3; bi = lane + 96; }
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            float ov = __shfl_xor_sync(0xffffffff, bv, off);
                            int oi = __shfl_xor_sync(0xffffffff, bi, off);
                            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                        }
                        float win_p = (bi == lane) ? p0 : ((bi == lane + 32) ? p1 : ((bi == lane + 64) ? p2 : p3));
                        win_p = __shfl_sync(0xffffffff, win_p, bi & 31);
                        if (bi == lane) s0 = -1e31f;
                        if (bi == lane + 32) s1 = -1e31f;
                        if (bi == lane + 64) s2 = -1e31f;
                        if (bi == lane + 96) s3 = -1e31f;
                        w_sum += win_p;
                        if (lane == k) {
                            s_idx_red[k] = bi;
                            s_scratch[k] = win_p;
                        }
                    }
                    if (lane < 8) {
                        s_scratch[lane] *= (s_m.expert_weights_scale / w_sum);
                    }
                }
                __syncthreads();

                const uint8_t* gu_w = reinterpret_cast<const uint8_t*>(s_L.ffn_gate_up_exps);
                const int n_sb = dim >> 8; // 9
                const size_t tot_gu_sb = (size_t)(s_m.n_experts * 2 * n_ff) * (size_t)n_sb;
                const uint2* gu_qs_u2 = reinterpret_cast<const uint2*>(gu_w);
                const half* gu_d = reinterpret_cast<const half*>(gu_w + tot_gu_sb * 64ULL);
                float* ws_situ = reinterpret_cast<float*>(s_m.ws_situ);
                const int ib32 = lane >> 2;
                const int l_sub = lane & 3;
                const int tot_gu = 8 * n_ff; // 9216

                for (int item = global_w_ilv; item < tot_gu; item += total_warps) {
                    const int slot = item / n_ff;
                    const int ff_idx = item - slot * n_ff;
                    const int exp_id = s_idx_red[slot];
                    const size_t g_row = ((size_t)exp_id * (size_t)(2 * n_ff) + (size_t)ff_idx) * 9ULL;
                    const size_t u_row = ((size_t)exp_id * (size_t)(2 * n_ff) + (size_t)(n_ff + ff_idx)) * 9ULL;

                    float gate = 0.0f, up = 0.0f;
                    #pragma unroll 9
                    for (int sb = 0; sb < 9; ++sb) {
                        const int2 xv = reinterpret_cast<const int2*>(s_xq)[(sb << 5) + lane];
                        const float dx = s_xds[(sb << 3) + ib32].x;

                        const uint2 q2_g = __ldg(&gu_qs_u2[((g_row + sb) << 3) + ib32]);
                        const uint2 q2_u = __ldg(&gu_qs_u2[((u_row + sb) << 3) + ib32]);
                        const float d_g = __half2float(__ldg(&gu_d[g_row + sb]));
                        const float d_u = __half2float(__ldg(&gu_d[u_row + sb]));

                        {
                            const uint8_t u_grid = (q2_g.x >> (8 * l_sub)) & 0xffu;
                            const uint32_t hi = q2_g.y;
                            const float db = d_g * (0.5f + (float)(hi >> 28)) * 0.25f;
                            const uint64_t g64 = s_iq2_grid[u_grid];
                            const uint8_t signs = s_iq2_signs[(hi >> (7 * l_sub)) & 127u];
                            const uint32_t m_lo = s_sign_mask4[signs & 0x0fu];
                            const uint32_t m_hi = s_sign_mask4[signs >> 4];
                            const int w_lo = (int)(((uint32_t)g64 ^ m_lo) + (m_lo & 0x01010101u));
                            const int w_hi = (int)(((uint32_t)(g64 >> 32) ^ m_hi) + (m_hi & 0x01010101u));
                            gate += (db * dx) * (float)__dp4a(w_lo, xv.x, __dp4a(w_hi, xv.y, 0));
                        }
                        {
                            const uint8_t u_grid = (q2_u.x >> (8 * l_sub)) & 0xffu;
                            const uint32_t hi = q2_u.y;
                            const float db = d_u * (0.5f + (float)(hi >> 28)) * 0.25f;
                            const uint64_t g64 = s_iq2_grid[u_grid];
                            const uint8_t signs = s_iq2_signs[(hi >> (7 * l_sub)) & 127u];
                            const uint32_t m_lo = s_sign_mask4[signs & 0x0fu];
                            const uint32_t m_hi = s_sign_mask4[signs >> 4];
                            const int w_lo = (int)(((uint32_t)g64 ^ m_lo) + (m_lo & 0x01010101u));
                            const int w_hi = (int)(((uint32_t)(g64 >> 32) ^ m_hi) + (m_hi & 0x01010101u));
                            up += (db * dx) * (float)__dp4a(w_lo, xv.x, __dp4a(w_hi, xv.y, 0));
                        }
                    }
                    gate = ink_warp_sum(gate);
                    up   = ink_warp_sum(up);
                    if (lane == 0) {
                        const float act_g = s_m.situ_beta * tanhf(gate / s_m.situ_beta) * (1.0f / (1.0f + expf(-gate)));
                        const float act_u = (s_m.situ_lb > 0.0f) ? (s_m.situ_lb * tanhf(up / s_m.situ_lb)) : up;
                        st_cg_f32(&ws_situ[item], act_g * act_u);
                    }
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[6] += tn - s_prof[9]; s_prof[9] = tn; }

            // --- Substage L8: On-Chip `ws_situ -> Q8_1` (`0` barrier!) + `(slot, out_row)` `ffn_down_exps` (`18,432` Half-Warps!) ---
            {
                const int dim = s_m.dim;
                const int n_ff = s_m.n_ff_exp;
                // `s_idx_red[0..7]` & `s_scratch[0..7]` ALREADY hold top-8 expert IDs & weights from L7!
                // Quantize L2 `ws_situ [8 * 1152 = 9,216]` directly into SMEM `s_xq, s_xds` (`0` grid barrier, `s_scratch[0..7]` untouched!):
                {
                    const float* ws_situ = reinterpret_cast<const float*>(s_m.ws_situ);
                    int8_t* s_xq_s8 = reinterpret_cast<int8_t*>(s_xq);
                    for (int blk = wid; blk < 288; blk += INK_WARPS) {
                        const int col = (blk << 5) + lane;
                        const float v = ld_cg_f32(&ws_situ[col]);
                        float sum_x = v, amax = fabsf(v);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                        }
                        const float d = amax * (1.0f / 127.0f);
                        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        s_xq_s8[col] = (int8_t)max(-127, min(127, __float2int_rn(v * id)));
                        if (lane == 0) s_xds[blk] = make_float2(d, sum_x);
                    }
                    __syncthreads();
                }

                const uint8_t* dn_w = reinterpret_cast<const uint8_t*>(s_L.ffn_down_exps);
                const int dn_type = s_L.down_type;
                const size_t n_dn_rows = (size_t)s_m.n_experts * (size_t)dim;
                float* ws_part = reinterpret_cast<float*>(s_m.ws_situ); // [8 * 2304 = 18,432 floats in L2!]

                if (dn_type == 42) {
                    // 8-Lane Quarter-Warp Planarized Zero-LUT `Q2_0` (`4,352` concurrent rows, `9` exact iters, `0` tail, `0` SMEM LUT loads!)
                    const uint32_t* qs_u32 = reinterpret_cast<const uint32_t*>(dn_w);
                    const half* d_all = reinterpret_cast<const half*>(dn_w + n_dn_rows * (18ULL * 16ULL));
                    const int lane8 = lane & 7;
                    const uint32_t qmask = 0xffu << (lane & 24);
                    const int qw_in_cta = (wid << 2) | (lane >> 3);
                    const int total_qwarps = num_sms * (INK_WARPS * 4);
                    const int global_qw = qw_in_cta * num_sms + bid;

                    for (int item = global_qw; item < 8 * dim; item += total_qwarps) {
                        const int slot = item / dim;
                        const int out_row = item - slot * dim;
                        const size_t r_blk = ((size_t)s_idx_red[slot] * (size_t)dim + (size_t)out_row) * 18ULL;
                        const size_t r_u32 = r_blk << 2;
                        const int4* sx_i4 = reinterpret_cast<const int4*>(s_xq + slot * (n_ff >> 2));
                        const float2* s_ds = s_xds + slot * (n_ff >> 5);

                        float slot_acc = 0.0f;
                        #pragma unroll 9
                        for (int k9 = 0; k9 < 9; ++k9) {
                            const int idx8 = (k9 << 3) + lane8;
                            const uint32_t P = __ldg(&qs_u32[r_u32 + idx8]);
                            const float d = __half2float(__ldg(&d_all[r_blk + (idx8 >> 2)]));
                            const float dx = s_ds[idx8 >> 1].x;
                            const int4 xv = sx_i4[idx8];
                            int s = __dp4a((int)(P & 0x03030303u), xv.x, 0);
                            s = __dp4a((int)((P >> 2) & 0x03030303u), xv.y, s);
                            s = __dp4a((int)((P >> 4) & 0x03030303u), xv.z, s);
                            s = __dp4a((int)((P >> 6) & 0x03030303u), xv.w, s);
                            int sx = __dp4a(0x01010101, xv.x, 0);
                            sx = __dp4a(0x01010101, xv.y, sx);
                            sx = __dp4a(0x01010101, xv.z, sx);
                            sx = __dp4a(0x01010101, xv.w, sx);
                            slot_acc += (d * dx) * (float)(s - sx);
                        }
                        slot_acc += __shfl_down_sync(qmask, slot_acc, 4);
                        slot_acc += __shfl_down_sync(qmask, slot_acc, 2);
                        slot_acc += __shfl_down_sync(qmask, slot_acc, 1);
                        if (lane8 == 0) st_cg_f32(&ws_part[slot * dim + out_row], s_scratch[slot] * slot_acc);
                    }
                } else {
                    // Aligned Split `IQ4_NL`: 9 exact 4-block iterations (`#pragma unroll 9`)!
                    const uint32_t* qs_u32 = reinterpret_cast<const uint32_t*>(dn_w);
                    const half* d_all = reinterpret_cast<const half*>(dn_w + n_dn_rows * (36ULL * 16ULL));
                    const int b_sub = lane16 >> 2;
                    const int sub4  = lane16 & 3;
                    for (int item = global_hw_ilv; item < 8 * dim; item += total_hwarps) {
                        const int slot = item / dim;
                        const int out_row = item - slot * dim;
                        const size_t r_base = ((size_t)s_idx_red[slot] * (size_t)dim + (size_t)out_row) * 36ULL;
                        const int* sx_i32 = s_xq + slot * (n_ff >> 2);
                        const float2* s_ds = s_xds + slot * (n_ff >> 5);
                        float slot_acc = 0.0f;
                        #pragma unroll 9
                        for (int bi = b_sub; bi < 36; bi += 4) {
                            const uint32_t qb4 = __ldg(&qs_u32[((r_base + bi) << 2) + sub4]);
                            const float d = __half2float(__ldg(&d_all[r_base + bi]));
                            const int w_lo = (int)((uint32_t)s_iq4_u8[qb4 & 0x0fu]
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 8) & 0x0fu] << 8)
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 16) & 0x0fu] << 16)
                                           | ((uint32_t)s_iq4_u8[qb4 >> 24] << 24));
                            const int w_hi = (int)((uint32_t)s_iq4_u8[(qb4 >> 4) & 0x0fu]
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 12) & 0x0fu] << 8)
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 20) & 0x0fu] << 16)
                                           | ((uint32_t)s_iq4_u8[qb4 >> 28] << 24));
                            const int x_lo = sx_i32[(bi << 3) + sub4];
                            const int x_hi = sx_i32[(bi << 3) + 4 + sub4];
                            slot_acc += (d * s_ds[bi].x) * (float)__dp4a(w_lo, x_lo, __dp4a(w_hi, x_hi, 0));
                        }
                        const float sum_hw = ink_hwarp_sum(slot_acc, hmask);
                        if (lane16 == 0) st_cg_f32(&ws_part[slot * dim + out_row], s_scratch[slot] * sum_hw);
                    }
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            {
                const int dim = s_m.dim;
                const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
                const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
                float* ffn_p_st = reinterpret_cast<float*>(s_m.conv_state_va)
                                + (size_t)cache_slot * conv_ch_per_slot * 3ULL
                                + (size_t)(2 * kv_dim + dim) * 3ULL;
                const float* fconv_w = reinterpret_cast<const float*>(s_L.ffn_post_conv);
                const float* ws_part = reinterpret_cast<const float*>(s_m.ws_situ);
                const float* ws_ffn_inp = reinterpret_cast<const float*>(s_m.ws_ffn_inp);
                float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);

                for (int out_row = bid * INK_THREADS + tid; out_row < dim; out_row += num_sms * INK_THREADS) {
                    const float moe_val =
                        ((ld_cg_f32(&ws_part[0 * dim + out_row]) + ld_cg_f32(&ws_part[1 * dim + out_row]))
                       + (ld_cg_f32(&ws_part[2 * dim + out_row]) + ld_cg_f32(&ws_part[3 * dim + out_row])))
                      + ((ld_cg_f32(&ws_part[4 * dim + out_row]) + ld_cg_f32(&ws_part[5 * dim + out_row]))
                       + (ld_cg_f32(&ws_part[6 * dim + out_row]) + ld_cg_f32(&ws_part[7 * dim + out_row])));
                    float* st = ffn_p_st + (size_t)out_row * 3;
                    const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = moe_val;
                    st[0] = x1; st[1] = x2; st[2] = x3;
                    const float* cw = fconv_w + (size_t)out_row * 4;
                    const float csum = x0 * __ldg(&cw[0]) + x1 * __ldg(&cw[1]) + x2 * __ldg(&cw[2]) + x3 * __ldg(&cw[3]);
                    const float ffn_out = (x3 + csum) * 0.4472135955f;
                    st_cg_f32(&ws_cur[out_row], ld_cg_f32(&ws_ffn_inp[out_row]) + ffn_out);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[7] += tn - s_prof[9]; s_prof[9] = tn; }
        }

        // 3. Final Output Depth Mix + `output_norm` + `Q8_1` (`CTA 0`)
        if (bid == 0) {
            const int dim = s_m.dim;
            float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);
            dev_cta_depth_mix_single(
                ws_mixed,
                s_scratch,
                reinterpret_cast<const float*>(s_m.ws_blocks),
                reinterpret_cast<const float*>(s_m.ws_cur),
                reinterpret_cast<const float*>(s_m.output_res_query_va),
                dim,
                n_blocks,
                1e-6f
            );
            __syncthreads();
            dev_cta_rmsnorm_q8_1(
                (s_export_taps != 0) ? (reinterpret_cast<float*>(s_m.ws_dspark_taps) + (size_t)3 * dim) : nullptr,
                reinterpret_cast<int8_t*>(s_m.ws_xq),
                reinterpret_cast<float2*>(s_m.ws_xds),
                s_scratch,
                ws_mixed,
                reinterpret_cast<const float*>(s_m.output_norm_va),
                dim,
                s_m.rms_eps,
                true
            );
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 4. `lm_head` (Aligned `Q6_K` `LDG.E.128` `__dp4a`) + Fused Softcap(30) + In-Register Argmax/Gumbel (`All 68 SMs`)
        {
            const int dim = s_m.dim;
            dev_load_q8_1_smem(
                s_xq, s_xds,
                reinterpret_cast<const int8_t*>(s_m.ws_xq),
                reinterpret_cast<const float2*>(s_m.ws_xds),
                dim
            );
            const uint8_t* out_w = reinterpret_cast<const uint8_t*>(s_m.output_w_va);
            const int vocab = s_m.vocab_unpadded;
            const float softcap = s_m.final_logit_softcap;
            const bool use_temp = (s_temp > 1e-5f);
            const float inv_temp = use_temp ? (1.0f / s_temp) : 1.0f;
            const uint32_t rng_step = s_step_off + (uint32_t)step_idx;

            float best_v = -1e38f;
            int32_t best_i = 0x7fffffff;

            for (int v = global_w_ilv; v < vocab; v += total_warps) {
                float val = (s_m.output_w_type == 14)
                    ? dev_warp_dot_q6_k_aligned(out_w, (size_t)vocab, (size_t)v, s_xq, s_xds, dim, lane)
                    : dev_warp_dot_q8_0_aligned(out_w, (size_t)vocab, (size_t)v, s_xq, s_xds, dim, lane);
                if (lane == 0) {
                    if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
                    float score = use_temp
                        ? (val * inv_temp - logf(-logf(ink_uniform_01(s_seed, rng_step, (uint32_t)v))))
                        : val;
                    if (score > best_v || (score == best_v && v < best_i)) {
                        best_v = score;
                        best_i = v;
                    }
                }
            }

            if (lane == 0) {
                s_scratch[wid] = best_v;
                s_idx_red[wid] = best_i;
            }
            __syncthreads();

            if (wid == 0) {
                float wv = (lane < INK_WARPS) ? s_scratch[lane] : -1e38f;
                int32_t wi = (lane < INK_WARPS) ? s_idx_red[lane] : 0x7fffffff;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, wv, off);
                    int32_t oi = __shfl_down_sync(0xffffffff, wi, off);
                    if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
                }
                if (lane == 0) {
                    st_cg_f32(&reinterpret_cast<float*>(s_m.ws_cta_val)[bid], wv);
                    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&reinterpret_cast<int32_t*>(s_m.ws_cta_idx)[bid]), "r"(wi) : "memory");
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 5. All-SM Final Winner Reduction + Zero-Doorbell Host Token Stream
        if (wid < 4) {
            const int idx_cta = tid;
            float bv = (idx_cta < num_sms) ? ld_cg_f32(&reinterpret_cast<const float*>(s_m.ws_cta_val)[idx_cta]) : -1e38f;
            int32_t bi = 0x7fffffff;
            if (idx_cta < num_sms) {
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(bi) : "l"(&reinterpret_cast<const int32_t*>(s_m.ws_cta_idx)[idx_cta]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int32_t oi = __shfl_down_sync(0xffffffff, bi, off);
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
            int32_t next_tok = s_idx_red[0];
            for (int w = 1; w < 4; ++w) {
                if (s_scratch[w] > best_v || (s_scratch[w] == best_v && s_idx_red[w] < next_tok)) {
                    best_v = s_scratch[w];
                    next_tok = s_idx_red[w];
                }
            }
            s_cur_tok = next_tok;
            s_cur_pos = pos + 1;

            const bool is_eos = (next_tok == TOK_END_SAMPLING || next_tok == 199999);
            const bool ctx_full = (pos + 1 >= s_m.max_ctx);

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
                asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&reinterpret_cast<int32_t*>(s_m.ws_cta_idx)[127]), "r"(stop_sig) : "memory");
            }
        }

        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
        if (bid == 0 && tid == 0) { uint64_t tn = ink_gtimer(); s_prof[8] += tn - s_prof[9]; s_prof[9] = tn; }
        if (tid == 0) {
            int stop_sig = 0;
            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(stop_sig) : "l"(&reinterpret_cast<const int32_t*>(s_m.ws_cta_idx)[127]));
            s_done = stop_sig;
        }
        __syncthreads();
        if (s_done != 0) break;
    }

    if (bid == 0 && tid == 0) {
        volatile uint64_t* p_out = reinterpret_cast<volatile uint64_t*>(&ctrl->tokens_out[4076]);
        for (int k = 0; k < 9; ++k) {
            asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&p_out[k]), "l"(s_prof[k]) : "memory");
        }
        asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        __threadfence_system();
        const uint64_t seq = ctrl->expected_seq;
        asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&ctrl->done_seq), "l"(seq) : "memory");
    }
}

// =============================================================================
// 2. 68-SM Batched `BM <= 8` Prefill & Speculative Verification Megakernel (v4)
//    - `__launch_bounds__(512, 1)` + SMEM `s_m/s_L/s_lc` -> 128 regs, minimal spills!
//    - Processes `B = 1..8` tokens in lockstep across all 48 layer visits
//    - GPU-side Active Expert Deduplication on `ffn_gate_up_exps` & `(u, out_row)` `ffn_down_exps`
//    - Optional `save_conv_backup` per slot `b` for 5-us speculative ShortConv rollback
// =============================================================================
extern "C" __global__ __launch_bounds__(512, 1)
void ink_batch_forward_bm8_megakernel(
    const InkModelDesc* __restrict__ m_g,
    volatile InkRunCtrl* __restrict__ ctrl
) {
    union SharedStage {
        struct {
            int    xq[8 * (3072 / 4)];   // 24,576 B
            float2 xds[8 * (3072 / 32)]; //  6,144 B
        } gemm;
        struct {
            float acc[INK_WARPS * 128];  //  8,192 B
            float m_arr[INK_WARPS];
            float l_arr[INK_WARPS];
            float r_vec[16];
        } attn;
    };
    __shared__ __align__(16) SharedStage      s_u;
    __shared__ __align__(16) uint64_t         s_iq2_grid[256];      // 2,048 B
    __shared__ __align__(16) uint8_t          s_iq2_signs[128];     //   128 B
    __shared__ __align__(16) uint32_t         s_sign_mask4[16];     //    64 B
    __shared__ __align__(16) int32_t          s_q2_lut[256];        // 1,024 B
    __shared__ __align__(16) uint8_t          s_iq4_u8[16];         //    16 B
    __shared__ __align__(16) float            s_red[128];           //   512 B
    __shared__ __align__(16) int32_t          s_idx_red[64];        //   256 B
    __shared__ __align__(16) InkModelDesc     s_m;                  //   576 B
    __shared__ __align__(16) InkPhysLayerDesc s_L;                  //   256 B
    __shared__ __align__(16) InkLoopCtrlDesc  s_lc;                 //   128 B
    __shared__ __align__(16) int32_t          s_slot_kv[48];        //   192 B
    __shared__ int      s_abort;
    __shared__ int      s_B;
    __shared__ int      s_start_pos;
    __shared__ int      s_eff_passes;
    __shared__ uint32_t s_logits_mask;
    __shared__ int      s_export_taps;
    __shared__ int      s_save_conv;
    __shared__ float    s_temp;
    __shared__ uint32_t s_step_off;
    __shared__ uint64_t s_seed;
    __shared__ int32_t  s_in_toks[8];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const uint32_t hmask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const int hw_in_cta = (wid << 1) | (lane >> 4);
    const int total_warps = num_sms * INK_WARPS;
    const int global_w_ilv = wid * num_sms + bid;
    const int total_hwarps = num_sms * (INK_WARPS * 2);
    const int global_hw_ilv = hw_in_cta * num_sms + bid;

    ink_init_smem_luts(s_iq2_grid, s_iq2_signs, s_sign_mask4, s_q2_lut, s_iq4_u8);

    for (int i = tid; i < (int)(sizeof(InkModelDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_m)[i] = __ldg(&reinterpret_cast<const uint32_t*>(m_g)[i]);
    }
    if (tid == 0) {
        s_abort = 0;
        s_B = ctrl->batch_len;
        s_start_pos = ctrl->start_pos;
        s_eff_passes = ctrl->eff_passes;
        s_logits_mask = ctrl->logits_mask;
        s_export_taps = ctrl->export_taps;
        s_save_conv = ctrl->save_conv_backup;
        s_temp = ctrl->temperature;
        s_step_off = ctrl->step_offset;
        s_seed = ctrl->rng_seed;
        for (int b = 0; b < 8; ++b) s_in_toks[b] = ctrl->batch_tokens[b];
    }
    __syncthreads();
    if (tid < 48) {
        s_slot_kv[tid] = __ldg(&reinterpret_cast<const int32_t*>(s_m.slot_kv_dense_ptr)[tid]);
    }
    __syncthreads();

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(s_m.ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    const int B = s_B;
    const int n_phys = s_m.n_phys_layers;
    const int l_start = s_m.loop_start;
    const int l_end = s_m.loop_end;
    const int eff_passes = s_eff_passes;
    const int total_visits = n_phys + (eff_passes - 1) * (l_end - l_start);

    // 1. Token Embeddings + `tok_norm` (`CTAs 0..B-1` in parallel!)
    if (bid < B) {
        const int dim = s_m.dim;
        const int b = bid;
        const int tok_id = s_in_toks[b];
        float* cur_b = reinterpret_cast<float*>(s_m.ws_cur) + (size_t)b * dim;
        const uint8_t* emb_w = reinterpret_cast<const uint8_t*>(s_m.tok_embd_va);
        const float* tnorm = reinterpret_cast<const float*>(s_m.tok_norm_va);
        float ss = 0.0f;
        for (int col = tid; col < dim; col += INK_THREADS) {
            float v = ink_dequant_elem_emb(emb_w, s_m.tok_embd_type, (size_t)tok_id, dim, col);
            st_cg_f32(&cur_b[col], v);
            ss += v * v;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot / (float)dim + s_m.rms_eps);
        }
        __syncthreads();
        const float sc = s_red[0];
        for (int col = tid; col < dim; col += INK_THREADS) {
            st_cg_f32(&cur_b[col], ld_cg_f32(&cur_b[col]) * sc * __ldg(&tnorm[col]));
        }
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    int n_blocks = 0;
    int n_prelude_blocks = 0;
    int n_bank = 0;

    auto load_q8_1_bm = [&](int j_stream, int K_cols, int b_beg, int b_cnt) {
        const int n_int4 = K_cols >> 4;
        const int n_blks = K_cols >> 5;
        const int8_t* ws_xq = reinterpret_cast<const int8_t*>(s_m.ws_xq);
        const float2* ws_xds = reinterpret_cast<const float2*>(s_m.ws_xds);
        for (int b = 0; b < b_cnt; ++b) {
            const int* g_i32 = reinterpret_cast<const int*>(ws_xq + ((size_t)j_stream * 8 + (b_beg + b)) * 4608);
            const float2* g_ds = ws_xds + ((size_t)j_stream * 8 + (b_beg + b)) * 144;
            int4* s_i4 = reinterpret_cast<int4*>(s_u.gemm.xq + b * (K_cols >> 2));
            float2* s_ds = s_u.gemm.xds + b * n_blks;
            for (int i = tid; i < n_int4; i += INK_THREADS) s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
            for (int blk = tid; blk < n_blks; blk += INK_THREADS) s_ds[blk] = ld_cg_f32x2(&g_ds[blk]);
        }
        __syncthreads();
    };

    for (int visit = 0; visit < total_visits; ++visit) {
        int phys_l = 0;
        int cache_slot = 0;
        bool has_loop_msgs = false;

        if (visit < l_start) {
            phys_l = visit;
            cache_slot = visit;
        } else if (visit < l_start + eff_passes * (l_end - l_start)) {
            const int core_off = visit - l_start;
            const int pass = core_off / (l_end - l_start);
            const int ip = l_start + (core_off % (l_end - l_start));

            if (core_off == 0) {
                const int dim = s_m.dim;
                n_prelude_blocks = n_blocks;
                float* ws_prelude = reinterpret_cast<float*>(s_m.ws_prelude);
                const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
                float* ws_pre_blks = reinterpret_cast<float*>(s_m.ws_prelude_blocks);
                const float* ws_blocks = reinterpret_cast<const float*>(s_m.ws_blocks);
                for (int idx = bid * INK_THREADS + tid; idx < B * dim; idx += num_sms * INK_THREADS) {
                    st_cg_f32(&ws_prelude[idx], ld_cg_f32(&ws_cur[idx]));
                }
                for (int b = 0; b < B; ++b) {
                    for (int idx = bid * INK_THREADS + tid; idx < n_prelude_blocks * dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_pre_blks[b * 8 * dim + idx], ld_cg_f32(&ws_blocks[b * 8 * dim + idx]));
                    }
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            } else if (ip == l_start && pass > 0) {
                const int dim = s_m.dim;
                const int bank_src_idx = n_prelude_blocks + 1;
                n_bank = (n_blocks > bank_src_idx) ? (n_blocks - bank_src_idx) : 0;
                float* ws_bank = reinterpret_cast<float*>(s_m.ws_bank);
                float* ws_bank_out = reinterpret_cast<float*>(s_m.ws_bank_out);
                float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);
                float* ws_blocks = reinterpret_cast<float*>(s_m.ws_blocks);
                const float* ws_prelude = reinterpret_cast<const float*>(s_m.ws_prelude);
                const float* ws_pre_blks = reinterpret_cast<const float*>(s_m.ws_prelude_blocks);
                for (int b = 0; b < B; ++b) {
                    for (int idx = bid * INK_THREADS + tid; idx < n_bank * dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_bank[b * 8 * dim + idx], ld_cg_f32(&ws_blocks[(b * 8 + bank_src_idx) * dim + idx]));
                    }
                }
                for (int idx = bid * INK_THREADS + tid; idx < B * dim; idx += num_sms * INK_THREADS) {
                    st_cg_f32(&ws_bank_out[idx], ld_cg_f32(&ws_cur[idx]));
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
                for (int idx = bid * INK_THREADS + tid; idx < B * dim; idx += num_sms * INK_THREADS) {
                    st_cg_f32(&ws_cur[idx], ld_cg_f32(&ws_prelude[idx]));
                }
                for (int b = 0; b < B; ++b) {
                    for (int idx = bid * INK_THREADS + tid; idx < n_prelude_blocks * dim; idx += num_sms * INK_THREADS) {
                        st_cg_f32(&ws_blocks[b * 8 * dim + idx], ld_cg_f32(&ws_pre_blks[b * 8 * dim + idx]));
                    }
                }
                n_blocks = n_prelude_blocks;
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            }

            phys_l = ip;
            cache_slot = (pass == 0) ? ip : (n_phys + (pass - 1) * (l_end - l_start) + (ip - l_start));
            has_loop_msgs = (pass > 0);
        } else {
            const int ep_il = l_end + (visit - (l_start + eff_passes * (l_end - l_start)));
            phys_l = ep_il;
            cache_slot = ep_il;
        }

        if (tid < (int)(sizeof(InkPhysLayerDesc) / 4)) {
            const uint32_t* lp = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const InkPhysLayerDesc*>(s_m.layers_ptr) + phys_l);
            reinterpret_cast<uint32_t*>(&s_L)[tid] = __ldg(&lp[tid]);
        }
        if (has_loop_msgs && tid >= 64 && tid < 64 + (int)(sizeof(InkLoopCtrlDesc) / 4)) {
            const uint32_t* lcp = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const InkLoopCtrlDesc*>(s_m.loop_ctrls_ptr) + (phys_l - l_start));
            reinterpret_cast<uint32_t*>(&s_lc)[tid - 64] = __ldg(&lcp[tid - 64]);
        }
        __syncthreads();

        // Optional DSpark tap capture (`[B, 4, dim]`)
        if (s_export_taps != 0 && (visit == 5 || visit == 23 || visit == 41)) {
            const int dim = s_m.dim;
            const int tap_slot = (visit == 5) ? 0 : ((visit == 23) ? 1 : 2);
            float* ws_taps = reinterpret_cast<float*>(s_m.ws_dspark_taps);
            const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
            for (int b = 0; b < B; ++b) {
                float* dst_tap = ws_taps + ((size_t)b * 4 + tap_slot) * dim;
                const float* src_b = ws_cur + (size_t)b * dim;
                for (int col = bid * INK_THREADS + tid; col < dim; col += num_sms * INK_THREADS) {
                    st_cg_f32(&dst_tap[col], ld_cg_f32(&src_b[col]));
                }
            }
        }

        // --- Substage L1 (BM=8): Fused `inp_sa` + `depth_mix` + `rmsnorm_q8_1` (or Loop Controllers) ---
        {
            const int dim = s_m.dim;
            const bool is_swa = (s_L.is_swa != 0);
            const int n_q_mix = is_swa ? 1 : 3;
            const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);
            float* ws_inp_sa = reinterpret_cast<float*>(s_m.ws_inp_sa);
            float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);

            for (int idx = bid * INK_THREADS + tid; idx < B * dim; idx += num_sms * INK_THREADS) {
                st_cg_f32(&ws_inp_sa[idx], ld_cg_f32(&ws_cur[idx]));
            }

            if (!has_loop_msgs) {
                if (bid < B * n_q_mix) {
                    const int b = bid / n_q_mix;
                    const int q = bid % n_q_mix;
                    const float* raw_in = ws_cur + (size_t)b * dim;
                    if (s_L.attn_res_query != 0) {
                        const float* qv = reinterpret_cast<const float*>(s_L.attn_res_query) + (size_t)q * dim;
                        dev_cta_depth_mix_single(
                            ws_mixed + ((size_t)b * 3 + q) * dim,
                            s_red,
                            reinterpret_cast<const float*>(s_m.ws_blocks) + (size_t)b * 8 * dim,
                            raw_in,
                            qv,
                            dim,
                            n_blocks,
                            1e-6f
                        );
                        __syncthreads();
                        raw_in = ws_mixed + ((size_t)b * 3 + q) * dim;
                    }
                    const float* nw = is_swa
                        ? reinterpret_cast<const float*>(s_L.attn_norm)
                        : reinterpret_cast<const float*>((q == 0) ? s_L.attn_q_inp_norm : ((q == 1) ? s_L.attn_k_inp_norm : s_L.attn_v_inp_norm));
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + ((size_t)q * 8 + b) * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + ((size_t)q * 8 + b) * 144,
                        s_red,
                        raw_in,
                        nw,
                        dim,
                        s_m.rms_eps,
                        true
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            } else {
                if (s_L.attn_res_query != 0 && bid < B * n_q_mix) {
                    const int b = bid / n_q_mix;
                    const int q = bid % n_q_mix;
                    const float* qv = reinterpret_cast<const float*>(s_L.attn_res_query) + (size_t)q * dim;
                    dev_cta_depth_mix_single(
                        ws_mixed + ((size_t)b * 3 + q) * dim,
                        s_red,
                        reinterpret_cast<const float*>(s_m.ws_blocks) + (size_t)b * 8 * dim,
                        ws_cur + (size_t)b * dim,
                        qv,
                        dim,
                        n_blocks,
                        1e-6f
                    );
                }
                if (bid >= 24 && bid < 24 + B * 4) {
                    const int m_item = bid - 24;
                    const int b = m_item >> 2;
                    const int mq = m_item & 3;
                    const float* qv = reinterpret_cast<const float*>(s_lc.msg_query) + (size_t)mq * dim;
                    dev_cta_depth_mix_single(
                        reinterpret_cast<float*>(s_m.ws_messages) + ((size_t)b * 4 + mq) * dim,
                        s_red,
                        reinterpret_cast<const float*>(s_m.ws_bank) + (size_t)b * 8 * dim,
                        reinterpret_cast<const float*>(s_m.ws_bank_out) + (size_t)b * dim,
                        qv,
                        dim,
                        n_bank,
                        1e-6f
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                const int gate_rank = s_m.gate_rank;
                if (bid < B * 3) {
                    const int b = bid / 3;
                    const int j = bid % 3;
                    const float* raw_in = (s_L.attn_res_query != 0)
                        ? (ws_mixed + ((size_t)b * 3 + (is_swa ? 0 : j)) * dim)
                        : (ws_cur + (size_t)b * dim);
                    const float* msg_j = reinterpret_cast<const float*>(s_m.ws_messages) + ((size_t)b * 4 + j) * dim;
                    float* cat_j = reinterpret_cast<float*>(s_m.ws_ctrl_cat) + ((size_t)b * 3 + j) * (2 * dim);

                    float ssc = 0.0f, ssm = 0.0f;
                    for (int col = tid; col < dim; col += INK_THREADS) {
                        float c = ld_cg_f32(&raw_in[col]), mv = ld_cg_f32(&msg_j[col]);
                        ssc += c * c; ssm += mv * mv;
                    }
                    ssc = ink_warp_sum(ssc); ssm = ink_warp_sum(ssm);
                    if (lane == 0) { s_red[wid] = ssc; s_red[16 + wid] = ssm; }
                    __syncthreads();
                    if (wid == 0) {
                        float tc = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
                        float tm = (lane < INK_WARPS) ? s_red[16 + lane] : 0.0f;
                        tc = ink_warp_sum(tc); tm = ink_warp_sum(tm);
                        if (lane == 0) {
                            s_red[0] = rsqrtf(tc / (float)dim + 1e-6f);
                            s_red[1] = rsqrtf(tm / (float)dim + 1e-6f);
                        }
                    }
                    __syncthreads();
                    const float sc = s_red[0], sm = s_red[1];
                    for (int col = tid; col < dim; col += INK_THREADS) {
                        st_cg_f32(&cat_j[col],       ld_cg_f32(&raw_in[col]) * sc);
                        st_cg_f32(&cat_j[dim + col], ld_cg_f32(&msg_j[col])  * sm);
                    }
                    __syncthreads();
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + ((size_t)j * 8 + b) * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + ((size_t)j * 8 + b) * 144,
                        s_red,
                        cat_j,
                        nullptr,
                        2 * dim,
                        1e-6f,
                        false
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                for (int j = 0; j < 3; ++j) {
                    for (int b_beg = 0; b_beg < B; b_beg += 4) {
                        const int b_cnt = min(4, B - b_beg);
                        if (bid < 4) {
                            load_q8_1_bm(j, 2 * dim, b_beg, b_cnt);
                            const int r = bid * INK_WARPS + wid;
                            if (r < gate_rank) {
                                const uint8_t* wd = reinterpret_cast<const uint8_t*>(s_lc.ctrl_down[j]);
                                for (int b = 0; b < b_cnt; ++b) {
                                    float v = dev_warp_dot_q8_0_aligned(wd, (size_t)gate_rank, (size_t)r, s_u.gemm.xq + b * (4608 / 4), s_u.gemm.xds + b * 144, 2 * dim, lane);
                                    if (lane == 0) st_cg_f32(&reinterpret_cast<float*>(s_m.ws_ctrl_gate)[((b_beg + b) * 3 + j) * gate_rank + r], v);
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (bid < B * 3) {
                    const int b = bid / 3, j = bid % 3;
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + ((size_t)j * 8 + b) * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + ((size_t)j * 8 + b) * 144,
                        s_red,
                        reinterpret_cast<const float*>(s_m.ws_ctrl_gate) + ((size_t)b * 3 + j) * gate_rank,
                        nullptr,
                        gate_rank,
                        1e-6f,
                        false
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                for (int j = 0; j < 3; ++j) {
                    load_q8_1_bm(j, gate_rank, 0, B);
                    const uint8_t* wg = reinterpret_cast<const uint8_t*>(s_lc.ctrl_gain[j]);
                    const uint8_t* wf = reinterpret_cast<const uint8_t*>(s_lc.ctrl_forget[j]);
                    for (int row = global_hw_ilv; row < dim; row += total_hwarps) {
                        for (int b = 0; b < B; ++b) {
                            const float g = dev_hwarp_dot_q8_0_aligned(wg, (size_t)dim, (size_t)row, s_u.gemm.xq + b * (gate_rank / 4), s_u.gemm.xds + b * 2, gate_rank, lane16, hmask);
                            const float f = dev_hwarp_dot_q8_0_aligned(wf, (size_t)dim, (size_t)row, s_u.gemm.xq + b * (gate_rank / 4), s_u.gemm.xds + b * 2, gate_rank, lane16, hmask);
                            if (lane16 == 0) {
                                const float* raw_in = (s_L.attn_res_query != 0)
                                    ? (ws_mixed + ((size_t)b * 3 + (is_swa ? 0 : j)) * dim)
                                    : (ws_cur + (size_t)b * dim);
                                const float* msg_j = reinterpret_cast<const float*>(s_m.ws_messages) + ((size_t)b * 4 + j) * dim;
                                st_cg_f32(
                                    &reinterpret_cast<float*>(s_m.ws_inputs)[((size_t)b * 3 + j) * dim + row],
                                    (1.0f - tanhf(f)) * ld_cg_f32(&raw_in[row]) + tanhf(g) * ld_cg_f32(&msg_j[row])
                                );
                            }
                        }
                    }
                    __syncthreads();
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (bid < B * 3) {
                    const int b = bid / 3, j = bid % 3;
                    const float* nw = is_swa
                        ? reinterpret_cast<const float*>(s_L.attn_norm)
                        : reinterpret_cast<const float*>((j == 0) ? s_L.attn_q_inp_norm : ((j == 1) ? s_L.attn_k_inp_norm : s_L.attn_v_inp_norm));
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + ((size_t)j * 8 + b) * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + ((size_t)j * 8 + b) * 144,
                        s_red,
                        reinterpret_cast<const float*>(s_m.ws_inputs) + ((size_t)b * 3 + j) * dim,
                        nw,
                        dim,
                        s_m.rms_eps,
                        true
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            }
        }

        // --- Substage L2 (BM=8): Batched Aligned `Q8_0` `LDG.E.128` Projections across ALL 2,176 Half-Warps ---
        {
            const int dim = s_m.dim;
            const int q_dim = s_m.n_heads * s_m.head_dim;
            const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
            const int r_dim = s_m.n_heads * s_m.rel_len;
            const bool is_swa = (s_L.is_swa != 0);
            const int n_blk_d = dim >> 5; // 72
            const int n_i4_d  = dim >> 4; // 144

            auto dot_q8_bm = [&](const uint8_t* w_ptr, int n_rows_m, int local_r, float* dst_base, int dst_stride) {
                const int4* qs_row = reinterpret_cast<const int4*>(w_ptr + (size_t)local_r * (size_t)dim);
                const half* d_row  = reinterpret_cast<const half*>(w_ptr + (size_t)n_rows_m * (size_t)dim) + (size_t)local_r * (size_t)n_blk_d;
                float acc[8] = {0,0,0,0,0,0,0,0};
                #pragma unroll 2
                for (int i4 = lane16; i4 < n_i4_d; i4 += 16) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < B) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (dim >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[b] += (d * s_u.gemm.xds[b * n_blk_d + blk].x) * (float)sumi;
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < B) {
                        float v = ink_hwarp_sum(acc[b], hmask);
                        if (lane16 == 0) st_cg_f32(&dst_base[(size_t)b * dst_stride + local_r], v);
                    }
                }
            };

            float* ws_q = reinterpret_cast<float*>(s_m.ws_q);
            float* ws_k_raw = reinterpret_cast<float*>(s_m.ws_k_raw);
            float* ws_v_raw = reinterpret_cast<float*>(s_m.ws_v_raw);
            float* ws_r = reinterpret_cast<float*>(s_m.ws_r);
            float* ws_sinks = reinterpret_cast<float*>(s_m.ws_sinks);
            float* ws_gate_raw = reinterpret_cast<float*>(s_m.ws_gate_raw);
            float* ws_dyn_k = reinterpret_cast<float*>(s_m.ws_dyn_k);
            float* ws_dyn_v = reinterpret_cast<float*>(s_m.ws_dyn_v);

            const bool share_qkv_in = is_swa && !has_loop_msgs;
            load_q8_1_bm(0, dim, 0, B);
            if (share_qkv_in) {
                const int tot_rows = q_dim + kv_dim + kv_dim + r_dim + s_m.n_heads + q_dim;
                for (int r = global_hw_ilv; r < tot_rows; r += total_hwarps) {
                    if (r < q_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wq), q_dim, r, ws_q, q_dim);
                    else if (r < q_dim + kv_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wk), kv_dim, r - q_dim, ws_k_raw, kv_dim);
                    else if (r < q_dim + 2 * kv_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wv), kv_dim, r - (q_dim + kv_dim), ws_v_raw, kv_dim);
                    else if (r < q_dim + 2 * kv_dim + r_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_rel), r_dim, r - (q_dim + 2 * kv_dim), ws_r, r_dim);
                    else if (r < q_dim + 2 * kv_dim + r_dim + s_m.n_heads) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_sink_proj), s_m.n_heads, r - (q_dim + 2 * kv_dim + r_dim), ws_sinks, s_m.n_heads);
                    else dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wqkv_gate), q_dim, r - (q_dim + 2 * kv_dim + r_dim + s_m.n_heads), ws_gate_raw, q_dim);
                }
            } else {
                const int q_tot = q_dim + r_dim + s_m.n_heads + q_dim;
                for (int r = global_hw_ilv; r < q_tot; r += total_hwarps) {
                    if (r < q_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wq), q_dim, r, ws_q, q_dim);
                    else if (r < q_dim + r_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_rel), r_dim, r - q_dim, ws_r, r_dim);
                    else if (r < q_dim + r_dim + s_m.n_heads) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_sink_proj), s_m.n_heads, r - (q_dim + r_dim), ws_sinks, s_m.n_heads);
                    else dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wqkv_gate), q_dim, r - (q_dim + r_dim + s_m.n_heads), ws_gate_raw, q_dim);
                }
                __syncthreads();

                const int k_tot = kv_dim + (is_swa ? 0 : 32);
                load_q8_1_bm(1, dim, 0, B);
                for (int r = global_hw_ilv; r < k_tot; r += total_hwarps) {
                    if (r < kv_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wk), kv_dim, r, ws_k_raw, kv_dim);
                    else dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_k_conv_proj), 32, r - kv_dim, ws_dyn_k, 32);
                }
                __syncthreads();

                load_q8_1_bm(2, dim, 0, B);
                for (int r = global_hw_ilv; r < k_tot; r += total_hwarps) {
                    if (r < kv_dim) dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.wv), kv_dim, r, ws_v_raw, kv_dim);
                    else dot_q8_bm(reinterpret_cast<const uint8_t*>(s_L.attn_v_conv_proj), 32, r - kv_dim, ws_dyn_v, 32);
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L3 (BM=8): Sequential Causal `K/V ShortConv` across `b=0..B-1` + Q/K/V/R Clip + KV Store ---
        if (bid < 40) {
            const int dim = s_m.dim;
            const int q_dim = s_m.n_heads * s_m.head_dim;
            const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
            const int r_dim = s_m.n_heads * s_m.rel_len;
            const bool is_swa = (s_L.is_swa != 0);
            const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
            const size_t conv_total_floats = 48ULL * (size_t)conv_ch_per_slot * 3ULL;
            const int dense_idx = s_slot_kv[cache_slot];
            const int swa_cap = 2 * s_m.swa_window;
            const size_t kv_stride = (size_t)(is_swa ? swa_cap : s_m.max_ctx) * (size_t)kv_dim;
            half* k_cache_slot = (is_swa ? reinterpret_cast<half*>(s_m.swa_k_va) : reinterpret_cast<half*>(s_m.full_k_va)) + (size_t)dense_idx * kv_stride;
            half* v_cache_slot = (is_swa ? reinterpret_cast<half*>(s_m.swa_v_va) : reinterpret_cast<half*>(s_m.full_v_va)) + (size_t)dense_idx * kv_stride;
            float* slot_conv = reinterpret_cast<float*>(s_m.conv_state_va) + (size_t)cache_slot * conv_ch_per_slot * 3ULL;

            if (bid < 24) {
                const int h = bid;
                float* ws_q = reinterpret_cast<float*>(s_m.ws_q);
                float* ws_r = reinterpret_cast<float*>(s_m.ws_r);
                const float* qc_w = reinterpret_cast<const float*>(s_L.attn_q_clip);
                const float* rc_w = reinterpret_cast<const float*>(s_L.attn_rel_clip);
                for (int b = 0; b < B; ++b) {
                    const int pos_b = s_start_pos + b;
                    float tau = 1.0f;
                    if (!is_swa) {
                        float ratio = fmaxf(1.0f, (float)(pos_b + 1) / (float)s_m.log_scale_floor);
                        tau = 1.0f + s_m.log_scale_alpha * logf(ratio);
                    }
                    float qv = (tid < 128) ? (ld_cg_f32(&ws_q[b * q_dim + h * 128 + tid]) * __ldg(&qc_w[h * 128 + tid])) : 0.0f;
                    float ss = ink_warp_sum(qv * qv);
                    if (wid < 4 && lane == 0) s_red[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float tot = (lane < 4) ? s_red[lane] : 0.0f;
                        tot += __shfl_down_sync(0xffffffff, tot, 2);
                        tot += __shfl_down_sync(0xffffffff, tot, 1);
                        if (lane == 0) s_red[0] = fminf(1.0f, fmaxf(0.0f, s_m.qk_clip / (sqrtf(tot) + 1e-6f))) * tau;
                    }
                    __syncthreads();
                    if (tid < 128) st_cg_f32(&ws_q[b * q_dim + h * 128 + tid], qv * s_red[0]);
                    if (wid == 0) {
                        float rv = (lane < 16) ? (ld_cg_f32(&ws_r[b * r_dim + h * 16 + lane]) * __ldg(&rc_w[h * 16 + lane])) : 0.0f;
                        float rss = rv * rv;
                        #pragma unroll
                        for (int off = 8; off > 0; off >>= 1) rss += __shfl_xor_sync(0xffffffff, rss, off);
                        float rfac = fminf(1.0f, fmaxf(0.0f, s_m.rel_clip / (sqrtf(rss) + 1e-6f))) * tau;
                        if (lane < 16) st_cg_f32(&ws_r[b * r_dim + h * 16 + lane], rv * rfac);
                    }
                    __syncthreads();
                }
            } else if (bid < 32) {
                const int kv_h = bid - 24;
                const int c = kv_h * 128 + tid;
                const float* ws_k_raw = reinterpret_cast<const float*>(s_m.ws_k_raw);
                const float* ws_dyn_k = reinterpret_cast<const float*>(s_m.ws_dyn_k);
                float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f;
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
                float* st = slot_conv + (size_t)c * 3;
                if (tid < 128) {
                    x0 = st[0]; x1 = st[1]; x2 = st[2];
                    const float* cw = reinterpret_cast<const float*>(s_L.attn_k_conv) + (size_t)c * 4;
                    w0 = __ldg(&cw[0]); w1 = __ldg(&cw[1]); w2 = __ldg(&cw[2]); w3 = __ldg(&cw[3]);
                }
                for (int b = 0; b < B; ++b) {
                    float kv = 0.0f;
                    if (tid < 128) {
                        const float x3 = ld_cg_f32(&ws_k_raw[b * kv_dim + c]);
                        float csum = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
                        float inv_s = 0.4472135955f;
                        if (!is_swa) {
                            const float* d = ws_dyn_k + b * 32 + kv_h * 4;
                            csum += ld_cg_f32(&d[0]) * x3 + ld_cg_f32(&d[1]) * x2 + ld_cg_f32(&d[2]) * x1 + ld_cg_f32(&d[3]) * x0;
                            inv_s = 0.3333333333f;
                        }
                        kv = (x3 + csum) * inv_s;
                        x0 = x1; x1 = x2; x2 = x3;
                        if (s_save_conv != 0) {
                            float* bk = reinterpret_cast<float*>(s_m.conv_backup_va) + (size_t)b * conv_total_floats + (size_t)cache_slot * conv_ch_per_slot * 3ULL + (size_t)c * 3ULL;
                            bk[0] = x0; bk[1] = x1; bk[2] = x2;
                        }
                    }
                    float ss = ink_warp_sum(kv * kv);
                    if (wid < 4 && lane == 0) s_red[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float tot = (lane < 4) ? s_red[lane] : 0.0f;
                        tot += __shfl_down_sync(0xffffffff, tot, 2);
                        tot += __shfl_down_sync(0xffffffff, tot, 1);
                        if (lane == 0) s_red[0] = fminf(1.0f, fmaxf(0.0f, s_m.qk_clip / (sqrtf(tot) + 1e-6f)));
                    }
                    __syncthreads();
                    if (tid < 128) {
                        const int pos_b = s_start_pos + b;
                        const int cw_b = is_swa ? (pos_b & (swa_cap - 1)) : pos_b;
                        k_cache_slot[(size_t)cw_b * 1024 + c] = __float2half(kv * s_red[0]);
                    }
                    __syncthreads();
                }
                if (tid < 128) { st[0] = x0; st[1] = x1; st[2] = x2; }
            } else {
                const int kv_h = bid - 32;
                const int c = kv_h * 128 + tid;
                const float* ws_v_raw = reinterpret_cast<const float*>(s_m.ws_v_raw);
                const float* ws_dyn_v = reinterpret_cast<const float*>(s_m.ws_dyn_v);
                float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f;
                float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
                float* st = slot_conv + (size_t)kv_dim * 3ULL + (size_t)c * 3;
                if (tid < 128) {
                    x0 = st[0]; x1 = st[1]; x2 = st[2];
                    const float* cw = reinterpret_cast<const float*>(s_L.attn_v_conv) + (size_t)c * 4;
                    w0 = __ldg(&cw[0]); w1 = __ldg(&cw[1]); w2 = __ldg(&cw[2]); w3 = __ldg(&cw[3]);
                }
                for (int b = 0; b < B; ++b) {
                    float vv = 0.0f;
                    if (tid < 128) {
                        const float x3 = ld_cg_f32(&ws_v_raw[b * kv_dim + c]);
                        float csum = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
                        float inv_s = 0.4472135955f;
                        if (!is_swa) {
                            const float* d = ws_dyn_v + b * 32 + kv_h * 4;
                            csum += ld_cg_f32(&d[0]) * x3 + ld_cg_f32(&d[1]) * x2 + ld_cg_f32(&d[2]) * x1 + ld_cg_f32(&d[3]) * x0;
                            inv_s = 0.3333333333f;
                        }
                        vv = (x3 + csum) * inv_s;
                        x0 = x1; x1 = x2; x2 = x3;
                        if (s_save_conv != 0) {
                            float* bk = reinterpret_cast<float*>(s_m.conv_backup_va) + (size_t)b * conv_total_floats + (size_t)cache_slot * conv_ch_per_slot * 3ULL + (size_t)(kv_dim + c) * 3ULL;
                            bk[0] = x0; bk[1] = x1; bk[2] = x2;
                        }
                    }
                    float ss = ink_warp_sum(vv * vv);
                    if (wid < 4 && lane == 0) s_red[wid] = ss;
                    __syncthreads();
                    if (wid == 0) {
                        float tot = (lane < 4) ? s_red[lane] : 0.0f;
                        tot += __shfl_down_sync(0xffffffff, tot, 2);
                        tot += __shfl_down_sync(0xffffffff, tot, 1);
                        if (lane == 0) s_red[0] = rsqrtf(tot * (1.0f / 128.0f) + s_m.rms_eps);
                    }
                    __syncthreads();
                    if (tid < 128) {
                        const int pos_b = s_start_pos + b;
                        const int cw_b = is_swa ? (pos_b & (swa_cap - 1)) : pos_b;
                        v_cache_slot[(size_t)cw_b * 1024 + c] = __float2half(vv * s_red[0]);
                    }
                    __syncthreads();
                }
                if (tid < 128) { st[0] = x0; st[1] = x1; st[2] = x2; }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L4 (BM=8): Causal Rel+Sink GQA Attention across `B * 24` `(b, h)` pairs + Fused `Q8_1` ---
        {
            const bool is_swa = (s_L.is_swa != 0);
            const int q_dim = s_m.n_heads * s_m.head_dim;
            const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
            const int r_dim = s_m.n_heads * s_m.rel_len;
            const int dense_idx = s_slot_kv[cache_slot];
            const int swa_cap = 2 * s_m.swa_window;
            const size_t kv_stride = (size_t)(is_swa ? swa_cap : s_m.max_ctx) * (size_t)kv_dim;
            const half* k_cache_slot = (is_swa ? reinterpret_cast<const half*>(s_m.swa_k_va) : reinterpret_cast<const half*>(s_m.full_k_va)) + (size_t)dense_idx * kv_stride;
            const half* v_cache_slot = (is_swa ? reinterpret_cast<const half*>(s_m.swa_v_va) : reinterpret_cast<const half*>(s_m.full_v_va)) + (size_t)dense_idx * kv_stride;
            const float* ws_q = reinterpret_cast<const float*>(s_m.ws_q);
            const float* ws_r = reinterpret_cast<const float*>(s_m.ws_r);
            const float* ws_sinks = reinterpret_cast<const float*>(s_m.ws_sinks);
            const float* ws_gate_raw = reinterpret_cast<const float*>(s_m.ws_gate_raw);

            for (int bh = bid; bh < B * 24; bh += num_sms) {
                const int b = bh / 24;
                const int h = bh % 24;
                const int kv_h = h / 3;
                const int pos_b = s_start_pos + b;
                const int d0 = lane * 4;

                const float q0 = ld_cg_f32(&ws_q[b * q_dim + h * 128 + d0 + 0]);
                const float q1 = ld_cg_f32(&ws_q[b * q_dim + h * 128 + d0 + 1]);
                const float q2 = ld_cg_f32(&ws_q[b * q_dim + h * 128 + d0 + 2]);
                const float q3 = ld_cg_f32(&ws_q[b * q_dim + h * 128 + d0 + 3]);
                if (tid < 16) s_u.attn.r_vec[tid] = ld_cg_f32(&ws_r[b * r_dim + h * 16 + tid]);
                const float sink_h = ld_cg_f32(&ws_sinks[b * 24 + h]);
                __syncthreads();

                const int start_j = is_swa ? max(0, pos_b - s_m.swa_window + 1) : 0;
                const float inv_sqrt_hd = 0.0883883476f;
                const float* rel_bias_w = reinterpret_cast<const float*>(s_L.attn_rel_bias);

                float m_w = (wid == 0) ? 0.0f : -1e30f;
                float l_w = (wid == 0) ? 1.0f : 0.0f;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                for (int j = start_j + wid; j <= pos_b; j += INK_WARPS) {
                    const int slot_j = is_swa ? (j & (swa_cap - 1)) : j;
                    const half2* kj2 = reinterpret_cast<const half2*>(k_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
                    const half2 k01 = __ldg(&kj2[0]), k23 = __ldg(&kj2[1]);
                    float dot = q0 * __half2float(k01.x) + q1 * __half2float(k01.y)
                              + q2 * __half2float(k23.x) + q3 * __half2float(k23.y);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);

                    const int dist = pos_b - j;
                    float r_dot = 0.0f;
                    if (dist < s_m.rel_extent) {
                        float rp = (lane < 16) ? (s_u.attn.r_vec[lane] * __ldg(&rel_bias_w[(size_t)dist * 16 + lane])) : 0.0f;
                        #pragma unroll
                        for (int off = 8; off > 0; off >>= 1) rp += __shfl_xor_sync(0xffffffff, rp, off);
                        r_dot = __shfl_sync(0xffffffff, rp, 0);
                    }

                    const float score = dot * inv_sqrt_hd + (r_dot - sink_h);
                    const float m_new = fmaxf(m_w, score);
                    const float alpha = expf(m_w - m_new);
                    const float beta  = expf(score - m_new);
                    l_w = l_w * alpha + beta;
                    m_w = m_new;

                    const half2* vj2 = reinterpret_cast<const half2*>(v_cache_slot + (size_t)slot_j * 1024 + kv_h * 128 + d0);
                    const half2 v01 = __ldg(&vj2[0]), v23 = __ldg(&vj2[1]);
                    acc0 = acc0 * alpha + beta * __half2float(v01.x);
                    acc1 = acc1 * alpha + beta * __half2float(v01.y);
                    acc2 = acc2 * alpha + beta * __half2float(v23.x);
                    acc3 = acc3 * alpha + beta * __half2float(v23.y);
                }

                if (lane == 0) { s_u.attn.m_arr[wid] = m_w; s_u.attn.l_arr[wid] = l_w; }
                s_u.attn.acc[wid * 128 + d0 + 0] = acc0;
                s_u.attn.acc[wid * 128 + d0 + 1] = acc1;
                s_u.attn.acc[wid * 128 + d0 + 2] = acc2;
                s_u.attn.acc[wid * 128 + d0 + 3] = acc3;
                __syncthreads();

                if (wid < 4) {
                    float m_max = -1e30f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_u.attn.l_arr[w] > 0.0f) m_max = fmaxf(m_max, s_u.attn.m_arr[w]);
                    }
                    float denom = 0.0f, out_d = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_u.attn.l_arr[w] > 0.0f) {
                            float ew = expf(s_u.attn.m_arr[w] - m_max);
                            denom += s_u.attn.l_arr[w] * ew;
                            out_d += s_u.attn.acc[w * 128 + tid] * ew;
                        }
                    }
                    const int col = h * 128 + tid;
                    const float gate = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_gate_raw[b * q_dim + col])));
                    const float v = (out_d / fmaxf(denom, 1e-20f)) * gate;

                    float sum_x = v, amax = fabsf(v);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                    }
                    const float d = amax * (1.0f / 127.0f);
                    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    st_cg_s8(&reinterpret_cast<int8_t*>(s_m.ws_xq)[b * 4608 + col], max(-127, min(127, __float2int_rn(v * id))));
                    if (lane == 0) st_cg_f32x2(&reinterpret_cast<float2*>(s_m.ws_xds)[b * 144 + h * 4 + wid], make_float2(d, sum_x));
                }
                __syncthreads();
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L5 (BM=8): 4-Way K-Sliced Batched `wo` (`9,216` Items -> Zero Tail!) + `attn_post_conv` ---
        const bool push_blk = ((phys_l % s_m.attn_res_block) == 0);
        const int blk_write_idx = n_blocks;
        {
            const int dim = s_m.dim;
            const int q_dim = s_m.n_heads * s_m.head_dim; // 3072
            load_q8_1_bm(0, q_dim, 0, B);
            const uint8_t* wo_ptr = reinterpret_cast<const uint8_t*>(s_L.wo);
            const int n_blk_o = q_dim >> 5; // 96
            const int slice_i4 = (q_dim >> 4) >> 2; // 48 int4s
            const half* d_wo_base = reinterpret_cast<const half*>(wo_ptr + (size_t)dim * (size_t)q_dim);
            float* ws_part = reinterpret_cast<float*>(s_m.ws_situ); // [4 * 8 * 2304 = 73,728 floats exact fit!]

            for (int item = global_hw_ilv; item < 4 * dim; item += total_hwarps) {
                const int k_part = item / dim;
                const int c = item - k_part * dim;
                const int4* qs_row = reinterpret_cast<const int4*>(wo_ptr + (size_t)c * (size_t)q_dim);
                const half* d_row  = d_wo_base + (size_t)c * (size_t)n_blk_o;
                float acc[8] = {0,0,0,0,0,0,0,0};
                const int i4_beg = k_part * slice_i4;
                const int i4_end = i4_beg + slice_i4;
                #pragma unroll 3
                for (int i4 = i4_beg + lane16; i4 < i4_end; i4 += 16) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < B) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (q_dim >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[b] += (d * s_u.gemm.xds[b * n_blk_o + blk].x) * (float)sumi;
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < B) {
                        const float v = ink_hwarp_sum(acc[b], hmask);
                        if (lane16 == 0) st_cg_f32(&ws_part[(k_part * 8 + b) * dim + c], v);
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        {
            const int dim = s_m.dim;
            const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
            const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
            const size_t conv_total_floats = 48ULL * (size_t)conv_ch_per_slot * 3ULL;
            float* attn_p_st = reinterpret_cast<float*>(s_m.conv_state_va)
                             + (size_t)cache_slot * conv_ch_per_slot * 3ULL
                             + (size_t)(2 * kv_dim) * 3ULL;
            const float* pconv_w = reinterpret_cast<const float*>(s_L.attn_post_conv);
            const float* ws_part = reinterpret_cast<const float*>(s_m.ws_situ);
            const float* ws_inp_sa = reinterpret_cast<const float*>(s_m.ws_inp_sa);
            float* ws_blocks = reinterpret_cast<float*>(s_m.ws_blocks);
            float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);
            float* ws_ffn_inp = reinterpret_cast<float*>(s_m.ws_ffn_inp);

            for (int c = bid * INK_THREADS + tid; c < dim; c += num_sms * INK_THREADS) {
                float* st = attn_p_st + (size_t)c * 3;
                float x0 = st[0], x1 = st[1], x2 = st[2];
                const float* cw = pconv_w + (size_t)c * 4;
                const float cw0 = __ldg(&cw[0]), cw1 = __ldg(&cw[1]), cw2 = __ldg(&cw[2]), cw3 = __ldg(&cw[3]);
                for (int b = 0; b < B; ++b) {
                    const float x3 = (ld_cg_f32(&ws_part[(0 * 8 + b) * dim + c]) + ld_cg_f32(&ws_part[(1 * 8 + b) * dim + c]))
                                   + (ld_cg_f32(&ws_part[(2 * 8 + b) * dim + c]) + ld_cg_f32(&ws_part[(3 * 8 + b) * dim + c]));
                    const float attn_out = (x3 + x0 * cw0 + x1 * cw1 + x2 * cw2 + x3 * cw3) * 0.4472135955f;
                    x0 = x1; x1 = x2; x2 = x3;
                    if (s_save_conv != 0) {
                        float* bk = reinterpret_cast<float*>(s_m.conv_backup_va) + (size_t)b * conv_total_floats + (size_t)cache_slot * conv_ch_per_slot * 3ULL + (size_t)(2 * kv_dim + c) * 3ULL;
                        bk[0] = x0; bk[1] = x1; bk[2] = x2;
                    }
                    const float isa = ld_cg_f32(&ws_inp_sa[b * dim + c]);
                    float new_cur;
                    if (push_blk) {
                        st_cg_f32(&ws_blocks[(b * 8 + blk_write_idx) * dim + c], isa);
                        new_cur = attn_out;
                    } else {
                        new_cur = isa + attn_out;
                    }
                    st_cg_f32(&ws_cur[b * dim + c], new_cur);
                    st_cg_f32(&ws_ffn_inp[b * dim + c], new_cur);
                }
                st[0] = x0; st[1] = x1; st[2] = x2;
            }
        }
        if (push_blk) n_blocks += 1;
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L6 (BM=8): `ffn_depth_mix` + Optional `loop_ctrl[3]` + `ffn_rmsnorm` + 64-CTA Router + Deduplication ---
        {
            const int dim = s_m.dim;
            const int gate_rank = s_m.gate_rank;
            float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);
            const float* ws_blocks = reinterpret_cast<const float*>(s_m.ws_blocks);
            const float* ws_cur = reinterpret_cast<const float*>(s_m.ws_cur);

            if (!has_loop_msgs) {
                if (bid < B) {
                    const int b = bid;
                    dev_cta_depth_mix_single(
                        ws_mixed + (size_t)b * dim,
                        s_red,
                        ws_blocks + (size_t)b * 8 * dim,
                        ws_cur + (size_t)b * dim,
                        reinterpret_cast<const float*>(s_L.ffn_res_query),
                        dim,
                        n_blocks,
                        1e-6f
                    );
                    __syncthreads();
                    dev_cta_rmsnorm_q8_1(
                        reinterpret_cast<float*>(s_m.ws_inputs) + (size_t)b * dim,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)b * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)b * 144,
                        s_red,
                        ws_mixed + (size_t)b * dim,
                        reinterpret_cast<const float*>(s_L.ffn_norm),
                        dim,
                        s_m.rms_eps,
                        true
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            } else {
                if (bid < B) {
                    const int b = bid;
                    dev_cta_depth_mix_single(
                        ws_mixed + (size_t)b * dim,
                        s_red,
                        ws_blocks + (size_t)b * 8 * dim,
                        ws_cur + (size_t)b * dim,
                        reinterpret_cast<const float*>(s_L.ffn_res_query),
                        dim,
                        n_blocks,
                        1e-6f
                    );
                    __syncthreads();
                    const float* msg_3 = reinterpret_cast<const float*>(s_m.ws_messages) + ((size_t)b * 4 + 3) * dim;
                    float* cat_b = reinterpret_cast<float*>(s_m.ws_ctrl_cat) + (size_t)b * (2 * dim);
                    float ssc = 0.0f, ssm = 0.0f;
                    for (int col = tid; col < dim; col += INK_THREADS) {
                        float c = ld_cg_f32(&ws_mixed[b * dim + col]), mv = ld_cg_f32(&msg_3[col]);
                        ssc += c * c; ssm += mv * mv;
                    }
                    ssc = ink_warp_sum(ssc); ssm = ink_warp_sum(ssm);
                    if (lane == 0) { s_red[wid] = ssc; s_red[16 + wid] = ssm; }
                    __syncthreads();
                    if (wid == 0) {
                        float tc = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
                        float tm = (lane < INK_WARPS) ? s_red[16 + lane] : 0.0f;
                        tc = ink_warp_sum(tc); tm = ink_warp_sum(tm);
                        if (lane == 0) {
                            s_red[0] = rsqrtf(tc / (float)dim + 1e-6f);
                            s_red[1] = rsqrtf(tm / (float)dim + 1e-6f);
                        }
                    }
                    __syncthreads();
                    const float sc = s_red[0], sm = s_red[1];
                    for (int col = tid; col < dim; col += INK_THREADS) {
                        st_cg_f32(&cat_b[col],       ld_cg_f32(&ws_mixed[b * dim + col]) * sc);
                        st_cg_f32(&cat_b[dim + col], ld_cg_f32(&msg_3[col]) * sm);
                    }
                    __syncthreads();
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)b * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)b * 144,
                        s_red,
                        cat_b,
                        nullptr,
                        2 * dim,
                        1e-6f,
                        false
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                for (int b_beg = 0; b_beg < B; b_beg += 4) {
                    const int b_cnt = min(4, B - b_beg);
                    if (bid < 4) {
                        load_q8_1_bm(0, 2 * dim, b_beg, b_cnt);
                        const int r = bid * INK_WARPS + wid;
                        if (r < gate_rank) {
                            const uint8_t* wd = reinterpret_cast<const uint8_t*>(s_lc.ctrl_down[3]);
                            for (int b = 0; b < b_cnt; ++b) {
                                float v = dev_warp_dot_q8_0_aligned(wd, (size_t)gate_rank, (size_t)r, s_u.gemm.xq + b * (4608 / 4), s_u.gemm.xds + b * 144, 2 * dim, lane);
                                if (lane == 0) st_cg_f32(&reinterpret_cast<float*>(s_m.ws_ctrl_gate)[(b_beg + b) * gate_rank + r], v);
                            }
                        }
                    }
                    __syncthreads();
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (bid < B) {
                    dev_cta_rmsnorm_q8_1(
                        nullptr,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)bid * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)bid * 144,
                        s_red,
                        reinterpret_cast<const float*>(s_m.ws_ctrl_gate) + (size_t)bid * gate_rank,
                        nullptr,
                        gate_rank,
                        1e-6f,
                        false
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                load_q8_1_bm(0, gate_rank, 0, B);
                const uint8_t* wg = reinterpret_cast<const uint8_t*>(s_lc.ctrl_gain[3]);
                const uint8_t* wf = reinterpret_cast<const uint8_t*>(s_lc.ctrl_forget[3]);
                float* ws_ctrl_gain = reinterpret_cast<float*>(s_m.ws_ctrl_gain);
                for (int row = global_hw_ilv; row < dim; row += total_hwarps) {
                    for (int b = 0; b < B; ++b) {
                        const float g = dev_hwarp_dot_q8_0_aligned(wg, (size_t)dim, (size_t)row, s_u.gemm.xq + b * (gate_rank / 4), s_u.gemm.xds + b * 2, gate_rank, lane16, hmask);
                        const float f = dev_hwarp_dot_q8_0_aligned(wf, (size_t)dim, (size_t)row, s_u.gemm.xq + b * (gate_rank / 4), s_u.gemm.xds + b * 2, gate_rank, lane16, hmask);
                        if (lane16 == 0) {
                            const float* msg_3 = reinterpret_cast<const float*>(s_m.ws_messages) + ((size_t)b * 4 + 3) * dim;
                            st_cg_f32(&ws_ctrl_gain[b * dim + row], (1.0f - tanhf(f)) * ld_cg_f32(&ws_mixed[b * dim + row]) + tanhf(g) * ld_cg_f32(&msg_3[row]));
                        }
                    }
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

                if (bid < B) {
                    dev_cta_rmsnorm_q8_1(
                        reinterpret_cast<float*>(s_m.ws_inputs) + (size_t)bid * dim,
                        reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)bid * 4608,
                        reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)bid * 144,
                        s_red,
                        ws_ctrl_gain + (size_t)bid * dim,
                        reinterpret_cast<const float*>(s_L.ffn_norm),
                        dim,
                        s_m.rms_eps,
                        true
                    );
                }
                if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
            }

            // 64-CTA Batched FP32 Router (`2` experts/CTA across `CTAs 0..63`)
            if (bid < 64) {
                const int e_sub = wid >> 3;
                const int sub_tid = tid & 255;
                const int e = (bid << 1) + e_sub;
                const float* rw = reinterpret_cast<const float*>(s_L.ffn_gate_inp) + (size_t)e * dim;
                const float* ws_inputs = reinterpret_cast<const float*>(s_m.ws_inputs);
                float sum[8] = {0,0,0,0,0,0,0,0};
                for (int col = sub_tid * 4; col < dim; col += 1024) {
                    const float4 w4 = __ldg(reinterpret_cast<const float4*>(rw + col));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < B) {
                            const float* xb = ws_inputs + (size_t)b * dim + col;
                            sum[b] += w4.x * ld_cg_f32(&xb[0]) + w4.y * ld_cg_f32(&xb[1])
                                    + w4.z * ld_cg_f32(&xb[2]) + w4.w * ld_cg_f32(&xb[3]);
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < B) {
                        float v = ink_warp_sum(sum[b]);
                        if (lane == 0) s_red[b * 16 + wid] = v;
                    }
                }
                __syncthreads();
                if (wid == 0 && lane < 16) {
                    for (int b = 0; b < B; ++b) {
                        float v = s_red[b * 16 + lane];
                        v += __shfl_down_sync(0x0000ffffu, v, 4);
                        v += __shfl_down_sync(0x0000ffffu, v, 2);
                        v += __shfl_down_sync(0x0000ffffu, v, 1);
                        if ((lane & 7) == 0) {
                            st_cg_f32(&reinterpret_cast<float*>(s_m.ws_router_logits)[b * 128 + (bid << 1) + (lane >> 3)], v);
                        }
                    }
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            // Top-8 Selection per `b` + GPU-Side Unique Active Expert Deduplication (`CTA 0`)
            if (bid == 0) {
                float* s_prob_bm = s_u.attn.acc;         // [8 * 128]
                float* s_sel_bm  = s_u.attn.acc + 1024;  // [8 * 128]
                int32_t* s_top_bm = reinterpret_cast<int32_t*>(s_u.attn.acc + 2048); // [64]
                const float* ws_rtr = reinterpret_cast<const float*>(s_m.ws_router_logits);
                int32_t* ws_top_ids = reinterpret_cast<int32_t*>(s_m.ws_top_ids);
                float* ws_top_w = reinterpret_cast<float*>(s_m.ws_top_weights);
                int32_t* ws_u_eid = reinterpret_cast<int32_t*>(s_m.ws_uniq_exp_id);
                uint32_t* ws_u_mask = reinterpret_cast<uint32_t*>(s_m.ws_uniq_mask);
                int32_t* ws_u_slots = reinterpret_cast<int32_t*>(s_m.ws_uniq_slots);

                for (int idx = tid; idx < B * 128; idx += INK_THREADS) {
                    const int e = idx & 127;
                    const float p = 1.0f / (1.0f + expf(-ld_cg_f32(&ws_rtr[idx])));
                    const float bias = (s_L.exp_probs_b != 0) ? __ldg(&reinterpret_cast<const float*>(s_L.exp_probs_b)[e]) : 0.0f;
                    s_prob_bm[idx] = p;
                    s_sel_bm[idx] = p + bias;
                }
                if (tid == 0) s_idx_red[0] = 0;
                __syncthreads();

                if (tid < B) {
                    const int b = tid;
                    int chosen[8];
                    float w_sum = 1e-20f;
                    for (int k = 0; k < 8; ++k) {
                        int best_i = 0;
                        float best_v = -1e30f;
                        for (int i = 0; i < 128; ++i) {
                            float sv = s_sel_bm[b * 128 + i];
                            if (sv > best_v) { best_v = sv; best_i = i; }
                        }
                        chosen[k] = best_i;
                        s_sel_bm[b * 128 + best_i] = -1e31f;
                        w_sum += s_prob_bm[b * 128 + best_i];
                    }
                    const float inv_s = s_m.expert_weights_scale / w_sum;
                    for (int k = 0; k < 8; ++k) {
                        int id = chosen[k];
                        s_top_bm[b * 8 + k] = id;
                        asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_top_ids[b * 8 + k]), "r"(id) : "memory");
                        st_cg_f32(&ws_top_w[b * 8 + k], s_prob_bm[b * 128 + id] * inv_s);
                    }
                }
                __syncthreads();

                if (tid < 128) {
                    const int e = tid;
                    uint32_t mask = 0u;
                    int slots_b[8] = {0,0,0,0,0,0,0,0};
                    for (int b = 0; b < B; ++b) {
                        for (int k = 0; k < 8; ++k) {
                            if (s_top_bm[b * 8 + k] == e) {
                                mask |= (1u << b);
                                slots_b[b] = k;
                            }
                        }
                    }
                    if (mask != 0u) {
                        const int u = atomicAdd(&s_idx_red[0], 1);
                        asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_u_eid[u]), "r"(e) : "memory");
                        asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(&ws_u_mask[u]), "r"(mask) : "memory");
                        for (int b = 0; b < B; ++b) {
                            asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_u_slots[u * 8 + b]), "r"(slots_b[b]) : "memory");
                        }
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&reinterpret_cast<int32_t*>(s_m.ws_n_uniq)[0]), "r"(s_idx_red[0]) : "memory");
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L7 (BM=8): Expert-Deduplicated `ffn_gate_up_exps` (`#pragma unroll 3` `LDG.E.64` `__dp4a`) + `SITU` ---
        {
            const int dim = s_m.dim;
            const int n_ff = s_m.n_ff_exp;
            load_q8_1_bm(0, dim, 0, B);
            int n_uniq = 0;
            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(n_uniq) : "l"(&reinterpret_cast<const int32_t*>(s_m.ws_n_uniq)[0]));

            const uint8_t* gu_w = reinterpret_cast<const uint8_t*>(s_L.ffn_gate_up_exps);
            const size_t tot_gu_sb = (size_t)(s_m.n_experts * 2 * n_ff) * 9ULL;
            const uint2* gu_qs_u2 = reinterpret_cast<const uint2*>(gu_w);
            const half* gu_d = reinterpret_cast<const half*>(gu_w + tot_gu_sb * 64ULL);
            const int32_t* ws_u_eid = reinterpret_cast<const int32_t*>(s_m.ws_uniq_exp_id);
            const uint32_t* ws_u_mask = reinterpret_cast<const uint32_t*>(s_m.ws_uniq_mask);
            const int32_t* ws_u_slots = reinterpret_cast<const int32_t*>(s_m.ws_uniq_slots);
            float* ws_situ = reinterpret_cast<float*>(s_m.ws_situ);
            const int ib32 = lane >> 2;
            const int l_sub = lane & 3;
            const int tot_gu = n_uniq * n_ff;

            for (int item = global_w_ilv; item < tot_gu; item += total_warps) {
                const int u = item / n_ff;
                const int ff_idx = item - u * n_ff;
                int exp_id; uint32_t mask;
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(exp_id) : "l"(&ws_u_eid[u]));
                asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(mask) : "l"(&ws_u_mask[u]));

                const size_t g_row = ((size_t)exp_id * (size_t)(2 * n_ff) + (size_t)ff_idx) * 9ULL;
                const size_t u_row = ((size_t)exp_id * (size_t)(2 * n_ff) + (size_t)(n_ff + ff_idx)) * 9ULL;

                uint32_t m_rem = mask;
                while (m_rem != 0u) {
                    const int b0 = __ffs(m_rem) - 1;
                    m_rem &= (m_rem - 1u);
                    const int b1 = (m_rem != 0u) ? (__ffs(m_rem) - 1) : -1;
                    if (b1 >= 0) m_rem &= (m_rem - 1u);

                    float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
                    const int2* sx0 = reinterpret_cast<const int2*>(s_u.gemm.xq + b0 * (dim >> 2));
                    const float2* ds0 = s_u.gemm.xds + b0 * 72;

                    #pragma unroll 3
                    for (int sb = 0; sb < 9; ++sb) {
                        const uint2 q2_g = __ldg(&gu_qs_u2[((g_row + sb) << 3) + ib32]);
                        const uint2 q2_u = __ldg(&gu_qs_u2[((u_row + sb) << 3) + ib32]);
                        const float d_g = __half2float(__ldg(&gu_d[g_row + sb]));
                        const float d_u = __half2float(__ldg(&gu_d[u_row + sb]));

                        int w_g_lo, w_g_hi, w_u_lo, w_u_hi;
                        float db_g, db_u;
                        {
                            const uint8_t u_grid = (q2_g.x >> (8 * l_sub)) & 0xffu;
                            const uint32_t hi = q2_g.y;
                            db_g = d_g * (0.5f + (float)(hi >> 28)) * 0.25f;
                            const uint64_t g64 = s_iq2_grid[u_grid];
                            const uint8_t signs = s_iq2_signs[(hi >> (7 * l_sub)) & 127u];
                            const uint32_t m_lo = s_sign_mask4[signs & 0x0fu], m_hi = s_sign_mask4[signs >> 4];
                            w_g_lo = (int)(((uint32_t)g64 ^ m_lo) + (m_lo & 0x01010101u));
                            w_g_hi = (int)(((uint32_t)(g64 >> 32) ^ m_hi) + (m_hi & 0x01010101u));
                        }
                        {
                            const uint8_t u_grid = (q2_u.x >> (8 * l_sub)) & 0xffu;
                            const uint32_t hi = q2_u.y;
                            db_u = d_u * (0.5f + (float)(hi >> 28)) * 0.25f;
                            const uint64_t g64 = s_iq2_grid[u_grid];
                            const uint8_t signs = s_iq2_signs[(hi >> (7 * l_sub)) & 127u];
                            const uint32_t m_lo = s_sign_mask4[signs & 0x0fu], m_hi = s_sign_mask4[signs >> 4];
                            w_u_lo = (int)(((uint32_t)g64 ^ m_lo) + (m_lo & 0x01010101u));
                            w_u_hi = (int)(((uint32_t)(g64 >> 32) ^ m_hi) + (m_hi & 0x01010101u));
                        }

                        {
                            const int2 xv = sx0[(sb << 5) + lane];
                            const float dx = ds0[(sb << 3) + ib32].x;
                            g0 += (db_g * dx) * (float)__dp4a(w_g_lo, xv.x, __dp4a(w_g_hi, xv.y, 0));
                            u0 += (db_u * dx) * (float)__dp4a(w_u_lo, xv.x, __dp4a(w_u_hi, xv.y, 0));
                        }
                        if (b1 >= 0) {
                            const int2 xv = reinterpret_cast<const int2*>(s_u.gemm.xq + b1 * (dim >> 2))[(sb << 5) + lane];
                            const float dx = s_u.gemm.xds[b1 * 72 + (sb << 3) + ib32].x;
                            g1 += (db_g * dx) * (float)__dp4a(w_g_lo, xv.x, __dp4a(w_g_hi, xv.y, 0));
                            u1 += (db_u * dx) * (float)__dp4a(w_u_lo, xv.x, __dp4a(w_u_hi, xv.y, 0));
                        }
                    }

                    g0 = ink_warp_sum(g0);
                    u0 = ink_warp_sum(u0);
                    if (lane == 0) {
                        int slot_b;
                        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(slot_b) : "l"(&ws_u_slots[u * 8 + b0]));
                        const float act_g = s_m.situ_beta * tanhf(g0 / s_m.situ_beta) * (1.0f / (1.0f + expf(-g0)));
                        const float act_u = (s_m.situ_lb > 0.0f) ? (s_m.situ_lb * tanhf(u0 / s_m.situ_lb)) : u0;
                        st_cg_f32(&ws_situ[(b0 * 8 + slot_b) * n_ff + ff_idx], act_g * act_u);
                    }
                    if (b1 >= 0) {
                        g1 = ink_warp_sum(g1);
                        u1 = ink_warp_sum(u1);
                        if (lane == 0) {
                            int slot_b;
                            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(slot_b) : "l"(&ws_u_slots[u * 8 + b1]));
                            const float act_g = s_m.situ_beta * tanhf(g1 / s_m.situ_beta) * (1.0f / (1.0f + expf(-g1)));
                            const float act_u = (s_m.situ_lb > 0.0f) ? (s_m.situ_lb * tanhf(u1 / s_m.situ_lb)) : u1;
                            st_cg_f32(&ws_situ[(b1 * 8 + slot_b) * n_ff + ff_idx], act_g * act_u);
                        }
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // Quantize `ws_situ [B * 8, 1152]` -> `ws_xq_moe` AND zero `ws_attn_gated[0 .. B * dim]` for L2 `atomicAdd`!
        {
            const int dim = s_m.dim;
            const int n_ff = s_m.n_ff_exp;
            float* moe_acc = reinterpret_cast<float*>(s_m.ws_attn_gated);
            for (int i = bid * INK_THREADS + tid; i < B * dim; i += num_sms * INK_THREADS) {
                st_cg_f32(&moe_acc[i], 0.0f);
            }
            if (bid < B * 8) {
                dev_cta_rmsnorm_q8_1(
                    nullptr,
                    reinterpret_cast<int8_t*>(s_m.ws_xq_moe) + (size_t)bid * n_ff,
                    reinterpret_cast<float2*>(s_m.ws_xds_moe) + (size_t)bid * (n_ff >> 5),
                    s_red,
                    reinterpret_cast<const float*>(s_m.ws_situ) + (size_t)bid * n_ff,
                    nullptr,
                    n_ff,
                    1e-6f,
                    false
                );
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // --- Substage L8 (BM=8): Zero-Spill `(u, out_row)` Parallel Deduplicated `ffn_down_exps` ---
        {
            const int dim = s_m.dim;
            const int n_ff = s_m.n_ff_exp;
            int n_uniq = 0;
            asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(n_uniq) : "l"(&reinterpret_cast<const int32_t*>(s_m.ws_n_uniq)[0]));
            const uint8_t* dn_w = reinterpret_cast<const uint8_t*>(s_L.ffn_down_exps);
            const int dn_type = s_L.down_type;
            const size_t n_dn_rows = (size_t)s_m.n_experts * (size_t)dim;
            const int32_t* ws_u_eid = reinterpret_cast<const int32_t*>(s_m.ws_uniq_exp_id);
            const uint32_t* ws_u_mask = reinterpret_cast<const uint32_t*>(s_m.ws_uniq_mask);
            const int32_t* ws_u_slots = reinterpret_cast<const int32_t*>(s_m.ws_uniq_slots);
            const float* ws_top_w = reinterpret_cast<const float*>(s_m.ws_top_weights);
            const int8_t* ws_xq_moe = reinterpret_cast<const int8_t*>(s_m.ws_xq_moe);
            const float2* ws_xds_moe = reinterpret_cast<const float2*>(s_m.ws_xds_moe);
            float* moe_acc = reinterpret_cast<float*>(s_m.ws_attn_gated); // [B * dim <= 73.7 KB in L2!]
            const int tot_dn = n_uniq * dim;

            for (int item = global_hw_ilv; item < tot_dn; item += total_hwarps) {
                const int u = item / dim;
                const int out_row = item - u * dim;
                int exp_id; uint32_t mask;
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(exp_id) : "l"(&ws_u_eid[u]));
                asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(mask) : "l"(&ws_u_mask[u]));
                const size_t row_idx = (size_t)exp_id * (size_t)dim + (size_t)out_row;

                uint32_t m_rem = mask;
                while (m_rem != 0u) {
                    const int b0 = __ffs(m_rem) - 1;
                    m_rem &= (m_rem - 1u);
                    const int b1 = (m_rem != 0u) ? (__ffs(m_rem) - 1) : -1;
                    if (b1 >= 0) m_rem &= (m_rem - 1u);

                    int s_b0 = 0, s_b1 = 0;
                    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(s_b0) : "l"(&ws_u_slots[u * 8 + b0]));
                    if (b1 >= 0) {
                        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(s_b1) : "l"(&ws_u_slots[u * 8 + b1]));
                    }
                    const int bs0 = b0 * 8 + s_b0;
                    const int bs1 = (b1 >= 0) ? (b1 * 8 + s_b1) : 0;

                    float acc0 = 0.0f, acc1 = 0.0f;
                    if (dn_type == 42) {
                        const uint4* qs_u4 = reinterpret_cast<const uint4*>(dn_w);
                        const half* d_all = reinterpret_cast<const half*>(dn_w + n_dn_rows * (18ULL * 16ULL));
                        const size_t r_base = row_idx * 18ULL;
                        auto dot_q2_p = [](uint32_t P, int4 xv) -> int {
                            int s = __dp4a((int)(P & 0x03030303u), xv.x, 0);
                            s = __dp4a((int)((P >> 2) & 0x03030303u), xv.y, s);
                            s = __dp4a((int)((P >> 4) & 0x03030303u), xv.z, s);
                            s = __dp4a((int)((P >> 6) & 0x03030303u), xv.w, s);
                            int sx = __dp4a(0x01010101, xv.x, 0);
                            sx = __dp4a(0x01010101, xv.y, sx);
                            sx = __dp4a(0x01010101, xv.z, sx);
                            sx = __dp4a(0x01010101, xv.w, sx);
                            return s - sx;
                        };
                        for (int bi = lane16; bi < 18; bi += 16) {
                            const uint4 qb = __ldg(&qs_u4[r_base + bi]);
                            const float d = __half2float(__ldg(&d_all[r_base + bi]));
                            {
                                const int* g_xq_i32 = reinterpret_cast<const int*>(ws_xq_moe + (size_t)bs0 * n_ff);
                                const float2* g_xds_s = ws_xds_moe + (size_t)bs0 * (n_ff >> 5);
                                const int4 xv0 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 0);
                                const int4 xv1 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 4);
                                const int4 xv2 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 8);
                                const int4 xv3 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 12);
                                const float dx0 = ld_cg_f32x2(&g_xds_s[(bi << 1) + 0]).x;
                                const float dx1 = ld_cg_f32x2(&g_xds_s[(bi << 1) + 1]).x;
                                const int s0 = dot_q2_p(qb.x, xv0) + dot_q2_p(qb.y, xv1);
                                const int s1 = dot_q2_p(qb.z, xv2) + dot_q2_p(qb.w, xv3);
                                acc0 += d * (dx0 * (float)s0 + dx1 * (float)s1);
                            }
                            if (b1 >= 0) {
                                const int* g_xq_i32 = reinterpret_cast<const int*>(ws_xq_moe + (size_t)bs1 * n_ff);
                                const float2* g_xds_s = ws_xds_moe + (size_t)bs1 * (n_ff >> 5);
                                const int4 xv0 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 0);
                                const int4 xv1 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 4);
                                const int4 xv2 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 8);
                                const int4 xv3 = ld_cg_s32x4(g_xq_i32 + (bi << 4) + 12);
                                const float dx0 = ld_cg_f32x2(&g_xds_s[(bi << 1) + 0]).x;
                                const float dx1 = ld_cg_f32x2(&g_xds_s[(bi << 1) + 1]).x;
                                const int s0 = dot_q2_p(qb.x, xv0) + dot_q2_p(qb.y, xv1);
                                const int s1 = dot_q2_p(qb.z, xv2) + dot_q2_p(qb.w, xv3);
                                acc1 += d * (dx0 * (float)s0 + dx1 * (float)s1);
                            }
                        }
                    } else {
                        const uint32_t* qs_u32 = reinterpret_cast<const uint32_t*>(dn_w);
                        const half* d_all = reinterpret_cast<const half*>(dn_w + n_dn_rows * (36ULL * 16ULL));
                        const size_t r_base = row_idx * 36ULL;
                        const int b_sub = lane16 >> 2, sub4 = lane16 & 3;
                        const int* g_xq0 = reinterpret_cast<const int*>(ws_xq_moe + (size_t)bs0 * n_ff);
                        const float2* g_ds0 = ws_xds_moe + (size_t)bs0 * (n_ff >> 5);
                        #pragma unroll 3
                        for (int bi = b_sub; bi < 36; bi += 4) {
                            const uint32_t qb4 = __ldg(&qs_u32[((r_base + bi) << 2) + sub4]);
                            const float d = __half2float(__ldg(&d_all[r_base + bi]));
                            const int w_lo = (int)((uint32_t)s_iq4_u8[qb4 & 0x0fu]
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 8) & 0x0fu] << 8)
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 16) & 0x0fu] << 16)
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 24) & 0x0fu] << 24));
                            const int w_hi = (int)((uint32_t)s_iq4_u8[(qb4 >> 4) & 0x0fu]
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 12) & 0x0fu] << 8)
                                           | ((uint32_t)s_iq4_u8[(qb4 >> 20) & 0x0fu] << 16)
                                           | ((uint32_t)s_iq4_u8[qb4 >> 28] << 24));

                            {
                                int x_lo, x_hi;
                                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(x_lo) : "l"(g_xq0 + (bi << 3) + sub4));
                                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(x_hi) : "l"(g_xq0 + (bi << 3) + 4 + sub4));
                                const float dx = ld_cg_f32x2(&g_ds0[bi]).x;
                                acc0 += (d * dx) * (float)__dp4a(w_lo, x_lo, __dp4a(w_hi, x_hi, 0));
                            }
                            if (b1 >= 0) {
                                const int* g_xq1 = reinterpret_cast<const int*>(ws_xq_moe + (size_t)bs1 * n_ff);
                                const float2* g_ds1 = ws_xds_moe + (size_t)bs1 * (n_ff >> 5);
                                int x_lo, x_hi;
                                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(x_lo) : "l"(g_xq1 + (bi << 3) + sub4));
                                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(x_hi) : "l"(g_xq1 + (bi << 3) + 4 + sub4));
                                const float dx = ld_cg_f32x2(&g_ds1[bi]).x;
                                acc1 += (d * dx) * (float)__dp4a(w_lo, x_lo, __dp4a(w_hi, x_hi, 0));
                            }
                        }
                    }

                    const float v0 = ink_hwarp_sum(acc0, hmask);
                    if (lane16 == 0) {
                        atomicAdd(&moe_acc[b0 * dim + out_row], v0 * ld_cg_f32(&ws_top_w[b0 * 8 + s_b0]));
                    }
                    if (b1 >= 0) {
                        const float v1 = ink_hwarp_sum(acc1, hmask);
                        if (lane16 == 0) {
                            atomicAdd(&moe_acc[b1 * dim + out_row], v1 * ld_cg_f32(&ws_top_w[b1 * 8 + s_b1]));
                        }
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        {
            const int dim = s_m.dim;
            const int kv_dim = s_m.n_kv_heads * s_m.head_dim;
            const int conv_ch_per_slot = 2 * kv_dim + 2 * dim;
            const size_t conv_total_floats = 48ULL * (size_t)conv_ch_per_slot * 3ULL;
            float* ffn_p_st = reinterpret_cast<float*>(s_m.conv_state_va)
                            + (size_t)cache_slot * conv_ch_per_slot * 3ULL
                            + (size_t)(2 * kv_dim + dim) * 3ULL;
            const float* fconv_w = reinterpret_cast<const float*>(s_L.ffn_post_conv);
            const float* moe_acc = reinterpret_cast<const float*>(s_m.ws_attn_gated);
            const float* ws_ffn_inp = reinterpret_cast<const float*>(s_m.ws_ffn_inp);
            float* ws_cur = reinterpret_cast<float*>(s_m.ws_cur);

            for (int out_row = bid * INK_THREADS + tid; out_row < dim; out_row += num_sms * INK_THREADS) {
                float* st = ffn_p_st + (size_t)out_row * 3;
                float x0 = st[0], x1 = st[1], x2 = st[2];
                const float* cw = fconv_w + (size_t)out_row * 4;
                const float cw0 = __ldg(&cw[0]), cw1 = __ldg(&cw[1]), cw2 = __ldg(&cw[2]), cw3 = __ldg(&cw[3]);
                for (int b = 0; b < B; ++b) {
                    const float x3 = ld_cg_f32(&moe_acc[b * dim + out_row]);
                    const float ffn_out = (x3 + x0 * cw0 + x1 * cw1 + x2 * cw2 + x3 * cw3) * 0.4472135955f;
                    x0 = x1; x1 = x2; x2 = x3;
                    if (s_save_conv != 0) {
                        float* bk = reinterpret_cast<float*>(s_m.conv_backup_va) + (size_t)b * conv_total_floats + (size_t)cache_slot * conv_ch_per_slot * 3ULL + (size_t)(2 * kv_dim + dim + out_row) * 3ULL;
                        bk[0] = x0; bk[1] = x1; bk[2] = x2;
                    }
                    st_cg_f32(&ws_cur[b * dim + out_row], ld_cg_f32(&ws_ffn_inp[b * dim + out_row]) + ffn_out);
                }
                st[0] = x0; st[1] = x1; st[2] = x2;
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
    }

    // Final Output Depth Mix + `output_norm` + `Q8_1` (`CTAs 0..B-1`)
    if ((s_logits_mask != 0u || s_export_taps != 0) && bid < B) {
        const int dim = s_m.dim;
        const int b = bid;
        if ((s_logits_mask & (1u << b)) != 0u || s_export_taps != 0) {
            float* ws_mixed = reinterpret_cast<float*>(s_m.ws_mixed);
            dev_cta_depth_mix_single(
                ws_mixed + (size_t)b * dim,
                s_red,
                reinterpret_cast<const float*>(s_m.ws_blocks) + (size_t)b * 8 * dim,
                reinterpret_cast<const float*>(s_m.ws_cur) + (size_t)b * dim,
                reinterpret_cast<const float*>(s_m.output_res_query_va),
                dim,
                n_blocks,
                1e-6f
            );
            __syncthreads();
            dev_cta_rmsnorm_q8_1(
                (s_export_taps != 0) ? (reinterpret_cast<float*>(s_m.ws_dspark_taps) + ((size_t)b * 4 + 3) * dim) : nullptr,
                reinterpret_cast<int8_t*>(s_m.ws_xq) + (size_t)b * 4608,
                reinterpret_cast<float2*>(s_m.ws_xds) + (size_t)b * 144,
                s_red,
                ws_mixed + (size_t)b * dim,
                reinterpret_cast<const float*>(s_m.output_norm_va),
                dim,
                s_m.rms_eps,
                true
            );
        }
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    // Batched Single-Read Aligned `lm_head` (`LDG.E.128` `__dp4a`) across active slots in `s_logits_mask`!
    if (s_logits_mask != 0u) {
        const int dim = s_m.dim;
        load_q8_1_bm(0, dim, 0, B);
        const uint8_t* out_w = reinterpret_cast<const uint8_t*>(s_m.output_w_va);
        const int vocab = s_m.vocab_unpadded;
        const float softcap = s_m.final_logit_softcap;
        const bool use_temp = (s_temp > 1e-5f);
        const float inv_temp = use_temp ? (1.0f / s_temp) : 1.0f;

        float best_v[8];
        int32_t best_i[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) { best_v[b] = -1e38f; best_i[b] = 0x7fffffff; }

        const int n_sb = dim >> 8; // 9
        const size_t tot_sb_out = (size_t)vocab * (size_t)n_sb;
        const half* d_out_q6 = reinterpret_cast<const half*>(out_w + tot_sb_out * 208ULL);
        const half* d_out_q8 = reinterpret_cast<const half*>(out_w + (size_t)vocab * (size_t)dim);
        const int sb_lane = lane >> 4, tid16 = lane & 15;
        const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
        const int ql_off = (h << 6) + (q << 5) + (half8 << 4);
        const int qh_off = 128 + (h << 5) + (half8 << 4);
        const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;

        for (int v = global_w_ilv; v < vocab; v += total_warps) {
            float acc[8] = {0,0,0,0,0,0,0,0};
            if (s_m.output_w_type == 14) {
                const uint8_t* w_row = out_w + (size_t)v * (size_t)(n_sb * 208);
                const half* d_row = d_out_q6 + (size_t)v * (size_t)n_sb;
                #pragma unroll
                for (int sb = sb_lane; sb < 9; sb += 2) {
                    if (sb < n_sb) {
                        const uint8_t* blk = w_row + sb * 208;
                        const float d_sc = __half2float(__ldg(&d_row[sb])) *
                                           (float)__ldg(&reinterpret_cast<const int8_t*>(blk + 192)[tid16]);
                        const uint4 ql = __ldg(reinterpret_cast<const uint4*>(blk + ql_off));
                        const uint4 qh = __ldg(reinterpret_cast<const uint4*>(blk + qh_off));

                        const uint32_t u6_0 = ((ql.x >> ql_sh) & 0x0f0f0f0fu) | (((qh.x >> qh_sh) & 0x03030303u) << 4);
                        const uint32_t u6_1 = ((ql.y >> ql_sh) & 0x0f0f0f0fu) | (((qh.y >> qh_sh) & 0x03030303u) << 4);
                        const uint32_t u6_2 = ((ql.z >> ql_sh) & 0x0f0f0f0fu) | (((qh.z >> qh_sh) & 0x03030303u) << 4);
                        const uint32_t u6_3 = ((ql.w >> ql_sh) & 0x0f0f0f0fu) | (((qh.w >> qh_sh) & 0x03030303u) << 4);
                        const uint32_t m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
                        const uint32_t m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
                        const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
                        const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
                        const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
                        const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));

                        const int x_base_int = (sb << 6) + (tid16 << 2);
                        const int ds_idx = (sb << 3) + (tid16 >> 1);
                        #pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            if (s_logits_mask & (1u << b)) {
                                const int4 xv = *reinterpret_cast<const int4*>(s_u.gemm.xq + b * (dim >> 2) + x_base_int);
                                int sumi = __dp4a(s6_0, xv.x, 0); sumi = __dp4a(s6_1, xv.y, sumi);
                                sumi = __dp4a(s6_2, xv.z, sumi); sumi = __dp4a(s6_3, xv.w, sumi);
                                acc[b] += (d_sc * s_u.gemm.xds[b * 72 + ds_idx].x) * (float)sumi;
                            }
                        }
                    }
                }
            } else {
                const int n_blk_o = dim >> 5;
                const int n_i4_o  = dim >> 4;
                const int4* qs_row = reinterpret_cast<const int4*>(out_w + (size_t)v * (size_t)dim);
                const half* d_row  = d_out_q8 + (size_t)v * (size_t)n_blk_o;
                #pragma unroll 2
                for (int i4 = lane; i4 < n_i4_o; i4 += 32) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (s_logits_mask & (1u << b)) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (dim >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[b] += (d * s_u.gemm.xds[b * n_blk_o + blk].x) * (float)sumi;
                        }
                    }
                }
            }

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (s_logits_mask & (1u << b)) {
                    float val = ink_warp_sum(acc[b]);
                    if (lane == 0) {
                        if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
                        float score = use_temp
                            ? (val * inv_temp - logf(-logf(ink_uniform_01(s_seed, s_step_off + (uint32_t)b, (uint32_t)v))))
                            : val;
                        if (score > best_v[b] || (score == best_v[b] && v < best_i[b])) {
                            best_v[b] = score;
                            best_i[b] = v;
                        }
                    }
                }
            }
        }

        float* ws_cta_val = reinterpret_cast<float*>(s_m.ws_cta_val);
        int32_t* ws_cta_idx = reinterpret_cast<int32_t*>(s_m.ws_cta_idx);
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if ((s_logits_mask & (1u << b)) == 0u) continue;
            if (lane == 0) {
                s_red[wid] = best_v[b];
                s_idx_red[wid] = best_i[b];
            }
            __syncthreads();
            if (wid == 0) {
                float wv = (lane < INK_WARPS) ? s_red[lane] : -1e38f;
                int32_t wi = (lane < INK_WARPS) ? s_idx_red[lane] : 0x7fffffff;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, wv, off);
                    int32_t oi = __shfl_down_sync(0xffffffff, wi, off);
                    if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
                }
                if (lane == 0) {
                    st_cg_f32(&ws_cta_val[b * 128 + bid], wv);
                    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&ws_cta_idx[b * 128 + bid]), "r"(wi) : "memory");
                }
            }
            __syncthreads();
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        if (bid < B && (s_logits_mask & (1u << bid)) != 0u) {
            const int b = bid;
            if (wid < 4) {
                const int idx_cta = tid;
                float bv = (idx_cta < num_sms) ? ld_cg_f32(&ws_cta_val[b * 128 + idx_cta]) : -1e38f;
                int32_t bi = 0x7fffffff;
                if (idx_cta < num_sms) {
                    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(bi) : "l"(&ws_cta_idx[b * 128 + idx_cta]));
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, bv, off);
                    int32_t oi = __shfl_down_sync(0xffffffff, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == 0) { s_red[wid] = bv; s_idx_red[wid] = bi; }
            }
            __syncthreads();
            if (tid == 0) {
                float best_val = s_red[0];
                int32_t win_tok = s_idx_red[0];
                for (int w = 1; w < 4; ++w) {
                    if (s_red[w] > best_val || (s_red[w] == best_val && s_idx_red[w] < win_tok)) {
                        best_val = s_red[w]; win_tok = s_idx_red[w];
                    }
                }
                st_host_s32(&ctrl->batch_out_toks[b], win_tok);
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
    }

    if (bid == 0 && tid == 0) {
        asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        __threadfence_system();
        const uint64_t seq = ctrl->expected_seq;
        asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&ctrl->done_seq), "l"(seq) : "memory");
    }
}

// Fast 5-microsecond VRAM-to-VRAM ShortConv rollback/commit kernel (`grid = (68, 1, 1), block = (512, 1, 1)`)
extern "C" __global__ __launch_bounds__(512)
void ink_copy_f32_vram_kernel(
    float* dst,
    const float* src,
    int32_t n_floats
) {
    const int n4 = n_floats >> 2;
    float4* d4 = reinterpret_cast<float4*>(dst);
    const float4* s4 = reinterpret_cast<const float4*>(src);
    for (int i = blockIdx.x * 512 + threadIdx.x; i < n4; i += gridDim.x * 512) {
        d4[i] = s4[i];
    }
    for (int i = (n4 << 2) + blockIdx.x * 512 + threadIdx.x; i < n_floats; i += gridDim.x * 512) {
        dst[i] = src[i];
    }
}

// =============================================================================
// 3. Official `DSpark` (`Inkling2MTP`) Block-8 Lossless Speculative Decoding Suite
//    - `ink_dspark_build_fold_kernel`: One-time 8ms GPU pre-folding of
//      `output.weight @ hidden_out.{weight, bias}` -> Aligned Split `Q8_0` `[200058, 256]`
//    - `ink_dspark_inject_kv_kernel`: Context KV + `f_rnn` injection from target taps
//    - `ink_dspark_propose_megakernel`: 68-SM fused KV-inject + 3-layer Non-Causal
//      Block-8 Transformer + Shared `Q6_K` Base Logits + 7-Step Markov RNN Head
// =============================================================================

__device__ __forceinline__ float ink_bf16_to_f32(uint16_t b) {
    return __uint_as_float(((uint32_t)b) << 16);
}

__device__ __forceinline__ half ld_cg_f16(const half* p) {
    uint16_t u;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(u) : "l"(p));
    return __ushort_as_half(u);
}

__device__ __forceinline__ void st_cg_f16(half* p, half v) {
    uint16_t u = __half_as_ushort(v);
    asm volatile("st.global.cg.u16 [%0], %1;" :: "l"(p), "h"(u) : "memory");
}

// One-time startup kernel (`grid = (68, 1, 1), block = (256, 1, 1)`):
// Computes `_fold_weight [vocab, 256]` (Aligned Split `Q8_0`) + `_fold_bias [vocab]` (`FP32`)
// directly on GA102 in ~8 ms with exact FP32 accumulation!
extern "C" __global__ __launch_bounds__(256)
void ink_dspark_build_fold_kernel(
    uint8_t* __restrict__ fold_w_out,
    float* __restrict__ fold_b_out,
    const uint8_t* __restrict__ out_w,
    int32_t out_w_type,
    int32_t vocab,
    const uint16_t* __restrict__ hout_w_bf16, // [2304, 256]
    const float* __restrict__ hout_b_f32      // [2304]
) {
    __shared__ float s_w[8][256];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int total_warps = gridDim.x * 8;
    const int global_w = blockIdx.x * 8 + wid;

    int8_t* fold_qs = reinterpret_cast<int8_t*>(fold_w_out);
    half* fold_d = reinterpret_cast<half*>(fold_w_out + (size_t)vocab * 256ULL);
    const size_t tot_sb_q6 = (size_t)vocab * 9ULL;
    const half* d_all_q6 = reinterpret_cast<const half*>(out_w + tot_sb_q6 * 208ULL);
    const half* d_all_q8 = reinterpret_cast<const half*>(out_w + (size_t)vocab * 2304ULL);

    for (int v = global_w; v < vocab; v += total_warps) {
        float acc_r[8] = {0,0,0,0,0,0,0,0};
        float bias_part = 0.0f;

        for (int sb = 0; sb < 9; ++sb) {
            if (out_w_type == 14) {
                const uint8_t* blk = out_w + ((size_t)v * 9ULL + (size_t)sb) * 208ULL;
                const float d = __half2float(__ldg(&d_all_q6[(size_t)v * 9ULL + (size_t)sb]));
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int idx = (j << 5) + lane;
                    const int sc = (int)__ldg(&reinterpret_cast<const int8_t*>(blk + 192)[idx >> 4]);
                    const int h = idx >> 7, rem = idx & 127, p = rem >> 6, q = (rem >> 5) & 1, k = rem & 31;
                    const uint8_t ql_b = __ldg(&blk[(h << 6) + (q << 5) + k]);
                    const uint8_t qh_b = __ldg(&blk[128 + (h << 5) + k]);
                    const int xl = (ql_b >> (p << 2)) & 0x0f;
                    const int xh = ((qh_b >> (((p << 1) + q) << 1)) & 0x03) << 4;
                    const float wv = d * (float)sc * (float)((xl | xh) - 32);
                    s_w[wid][idx] = wv;
                    bias_part += wv * __ldg(&hout_b_f32[sb * 256 + idx]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int idx = (j << 5) + lane;
                    const int blk_idx = sb * 8 + j;
                    const float d = __half2float(__ldg(&d_all_q8[(size_t)v * 72ULL + (size_t)blk_idx]));
                    const int8_t qs = __ldg(&reinterpret_cast<const int8_t*>(out_w)[((size_t)v * 72ULL + (size_t)blk_idx) * 32ULL + lane]);
                    const float wv = d * (float)qs;
                    s_w[wid][idx] = wv;
                    bias_part += wv * __ldg(&hout_b_f32[sb * 256 + idx]);
                }
            }
            __syncwarp();

            for (int k = 0; k < 256; ++k) {
                const float w_val = s_w[wid][k];
                const uint16_t* hr = hout_w_bf16 + (size_t)(sb * 256 + k) * 256ULL + lane;
                #pragma unroll
                for (int rb = 0; rb < 8; ++rb) {
                    acc_r[rb] += w_val * ink_bf16_to_f32(__ldg(&hr[rb * 32]));
                }
            }
            __syncwarp();
        }

        const float bias_v = ink_warp_sum(bias_part);
        if (lane == 0) fold_b_out[v] = bias_v;

        #pragma unroll
        for (int rb = 0; rb < 8; ++rb) {
            const float val = acc_r[rb];
            float amax = fabsf(val);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
            }
            const float d = amax * (1.0f / 127.0f);
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            fold_qs[(size_t)v * 256ULL + rb * 32 + lane] = (int8_t)max(-127, min(127, __float2int_rn(val * id)));
            if (lane == 0) fold_d[(size_t)v * 8ULL + rb] = __float2half(d);
        }
    }
}

// Per-head 128-dim RMSNorm + NeoX RoPE (`is_neox_style = True`, pairs `(i, i + 64)`) in 1 single warp (`0` SMEM!)
__device__ __forceinline__ void dev_warp_head128_rmsnorm_neox_rope(
    float& x0, float& x1, float& x2, float& x3, // indices `(lane, lane+32, lane+64, lane+96)`
    const float* norm_w,
    int pos,
    float rope_theta,
    float eps,
    int lane
) {
    float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    ss = ink_warp_sum(ss);
    const float inv_rms = __shfl_sync(0xffffffff, rsqrtf(ss * (1.0f / 128.0f) + eps), 0);

    x0 = x0 * inv_rms * __ldg(&norm_w[lane]);
    x1 = x1 * inv_rms * __ldg(&norm_w[lane + 32]);
    x2 = x2 * inv_rms * __ldg(&norm_w[lane + 64]);
    x3 = x3 * inv_rms * __ldg(&norm_w[lane + 96]);

    // NeoX pair A: `(lane, lane + 64)` -> `(x0, x2)`
    const float freq_a = powf(rope_theta, -((float)(2 * lane) * (1.0f / 128.0f)));
    float sa, ca;
    sincosf((float)pos * freq_a, &sa, &ca);
    const float r0 = x0 * ca - x2 * sa;
    const float r2 = x2 * ca + x0 * sa;

    // NeoX pair B: `(lane + 32, lane + 96)` -> `(x1, x3)`
    const float freq_b = powf(rope_theta, -((float)(2 * (lane + 32)) * (1.0f / 128.0f)));
    float sb, cb;
    sincosf((float)pos * freq_b, &sb, &cb);
    const float r1 = x1 * cb - x3 * sb;
    const float r3 = x3 * cb + x1 * sb;

    x0 = r0; x1 = r1; x2 = r2; x3 = r3;
}

// Device helper: inject `C = 1..8` committed target tap rows `ws_dspark_taps[0..C, 4, 2304]`
// at `[inject_pos .. inject_pos + C - 1]` into `draft_k/v [3, max_ctx, 1024]` + `ws_f_rnn [2304]`.
__device__ inline bool dev_dspark_inject_kv(
    const InkModelDesc& s_m,
    const InkDSparkDesc& s_d,
    int* s_xq_gemm,
    float2* s_xds_gemm,
    float* s_red,
    int inject_pos,
    int C,
    uint64_t* bar_state,
    uint64_t& bar_step,
    int& s_abort,
    volatile InkRunCtrl* ctrl
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const uint32_t hmask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const int hw_in_cta = (wid << 1) | (lane >> 4);
    const int total_warps = num_sms * INK_WARPS;
    const int global_w = bid * INK_WARPS + wid;
    const int total_hwarps = num_sms * (INK_WARPS * 2);
    const int global_hw_ilv = hw_in_cta * num_sms + bid;
    const int dim = 2304;

    // 1. `_mix_rmsnorm` + `Q8_1` for `(il = 0..2, c = 0..C-1)` on `CTAs 0 .. 3*C - 1`, and `f_rnn` on `CTA 24`
    const float* ws_taps = reinterpret_cast<const float*>(s_m.ws_dspark_taps);
    int8_t* d_xq = reinterpret_cast<int8_t*>(s_d.ws_xq_va);
    float2* d_xds = reinterpret_cast<float2*>(s_d.ws_xds_va);

    if (bid < 3 * C) {
        const int il = bid / C;
        const int c = bid % C;
        const float a0 = s_d.fuse_alpha[il * 4 + 0];
        const float a1 = s_d.fuse_alpha[il * 4 + 1];
        const float a2 = s_d.fuse_alpha[il * 4 + 2];
        const float a3 = s_d.fuse_alpha[il * 4 + 3];
        const float* t_base = ws_taps + (size_t)c * (4 * dim);
        const float* nw = reinterpret_cast<const float*>(s_d.layers[il].dflare_norm);

        float ss = 0.0f;
        for (int col = tid; col < dim; col += INK_THREADS) {
            const float m = a0 * ld_cg_f32(&t_base[0 * dim + col])
                          + a1 * ld_cg_f32(&t_base[1 * dim + col])
                          + a2 * ld_cg_f32(&t_base[2 * dim + col])
                          + a3 * ld_cg_f32(&t_base[3 * dim + col]);
            ss += m * m;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot * (1.0f / 2304.0f) + s_d.rms_eps);
        }
        __syncthreads();
        const float inv_rms = s_red[0];

        int8_t* g_xq = d_xq + (size_t)(il * 8 + c) * dim;
        float2* g_ds = d_xds + (size_t)(il * 8 + c) * 72;
        for (int blk = wid; blk < 72; blk += INK_WARPS) {
            const int col = (blk << 5) + lane;
            const float m = a0 * ld_cg_f32(&t_base[0 * dim + col])
                          + a1 * ld_cg_f32(&t_base[1 * dim + col])
                          + a2 * ld_cg_f32(&t_base[2 * dim + col])
                          + a3 * ld_cg_f32(&t_base[3 * dim + col]);
            const float v = m * inv_rms * __ldg(&nw[col]);
            float sum_x = v, amax = fabsf(v);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
            }
            const float d = amax * (1.0f / 127.0f);
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            st_cg_s8(&g_xq[col], max(-127, min(127, __float2int_rn(v * id))));
            if (lane == 0) st_cg_f32x2(&g_ds[blk], make_float2(d, sum_x));
        }
    } else if (bid == 24) {
        const float a0 = s_d.fuse_alpha[12];
        const float a1 = s_d.fuse_alpha[13];
        const float a2 = s_d.fuse_alpha[14];
        const float a3 = s_d.fuse_alpha[15];
        const float* t_base = ws_taps + (size_t)(C - 1) * (4 * dim);
        const float* nw = reinterpret_cast<const float*>(s_d.markov_dflare_va);
        float* f_rnn = reinterpret_cast<float*>(s_d.ws_f_rnn_va);

        float ss = 0.0f;
        for (int col = tid; col < dim; col += INK_THREADS) {
            const float m = a0 * ld_cg_f32(&t_base[0 * dim + col])
                          + a1 * ld_cg_f32(&t_base[1 * dim + col])
                          + a2 * ld_cg_f32(&t_base[2 * dim + col])
                          + a3 * ld_cg_f32(&t_base[3 * dim + col]);
            ss += m * m;
        }
        ss = ink_warp_sum(ss);
        if (lane == 0) s_red[wid] = ss;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
            tot = ink_warp_sum(tot);
            if (lane == 0) s_red[0] = rsqrtf(tot * (1.0f / 2304.0f) + s_d.rms_eps);
        }
        __syncthreads();
        const float inv_rms = s_red[0];
        for (int col = tid; col < dim; col += INK_THREADS) {
            const float m = a0 * ld_cg_f32(&t_base[0 * dim + col])
                          + a1 * ld_cg_f32(&t_base[1 * dim + col])
                          + a2 * ld_cg_f32(&t_base[2 * dim + col])
                          + a3 * ld_cg_f32(&t_base[3 * dim + col]);
            st_cg_f32(&f_rnn[col], m * inv_rms * __ldg(&nw[col]));
        }
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return false;

    // 2. `kv_proj_ctx` (`[2048, 2304]` Split `Q8_0`) across all 3 layers & `C` tokens
    float* ws_qkv = reinterpret_cast<float*>(s_d.ws_qkv_va);
    for (int il = 0; il < 3; ++il) {
        for (int c = 0; c < C; ++c) {
            const int* g_i32 = reinterpret_cast<const int*>(d_xq + (size_t)(il * 8 + c) * dim);
            const float2* g_ds = d_xds + (size_t)(il * 8 + c) * 72;
            int4* s_i4 = reinterpret_cast<int4*>(s_xq_gemm + c * (dim >> 2));
            float2* s_ds = s_xds_gemm + c * 72;
            for (int i = tid; i < (dim >> 4); i += INK_THREADS) s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
            for (int blk = tid; blk < 72; blk += INK_THREADS) s_ds[blk] = ld_cg_f32x2(&g_ds[blk]);
        }
        __syncthreads();

        const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(s_d.layers[il].kv_proj_ctx);
        const half* d_base = reinterpret_cast<const half*>(w_ptr + 2048ULL * 2304ULL);
        for (int r = global_hw_ilv; r < 2048; r += total_hwarps) {
            const int4* qs_row = reinterpret_cast<const int4*>(w_ptr + (size_t)r * 2304ULL);
            const half* d_row = d_base + (size_t)r * 72ULL;
            float acc[8] = {0,0,0,0,0,0,0,0};
            #pragma unroll 3
            for (int i4 = lane16; i4 < 144; i4 += 16) {
                const int4 wv = __ldg(&qs_row[i4]);
                const int blk = i4 >> 1;
                const float d = __half2float(__ldg(&d_row[blk]));
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    if (c < C) {
                        const int4 xv = reinterpret_cast<const int4*>(s_xq_gemm + c * (dim >> 2))[i4];
                        int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                        sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                        acc[c] += (d * s_xds_gemm[c * 72 + blk].x) * (float)sumi;
                    }
                }
            }
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (c < C) {
                    const float v = ink_hwarp_sum(acc[c], hmask);
                    if (lane16 == 0) st_cg_f32(&ws_qkv[(il * 8 + c) * 2048 + r], v);
                }
            }
        }
        __syncthreads();
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return false;

    // 3. `k_norm_ctx` + NeoX RoPE + Store `draft_k / draft_v [3, max_ctx, 1024]` (`1` warp per `(il, c, kv_h)`)
    half* draft_k = reinterpret_cast<half*>(s_d.draft_k_va);
    half* draft_v = reinterpret_cast<half*>(s_d.draft_v_va);
    const int tot_heads = 3 * C * 8;
    for (int item = global_w; item < tot_heads; item += total_warps) {
        const int kv_h = item & 7;
        const int tmp = item >> 3;
        const int c = tmp % C;
        const int il = tmp / C;
        const int pos_c = inject_pos + c;
        const float* kv_src = ws_qkv + (il * 8 + c) * 2048;
        const float* k_src = kv_src + kv_h * 128;
        const float* v_src = kv_src + 1024 + kv_h * 128;

        float x0 = ld_cg_f32(&k_src[lane]);
        float x1 = ld_cg_f32(&k_src[lane + 32]);
        float x2 = ld_cg_f32(&k_src[lane + 64]);
        float x3 = ld_cg_f32(&k_src[lane + 96]);
        dev_warp_head128_rmsnorm_neox_rope(
            x0, x1, x2, x3,
            reinterpret_cast<const float*>(s_d.layers[il].k_norm_ctx),
            pos_c, s_d.rope_theta, s_d.rms_eps, lane
        );

        const size_t dst_off = ((size_t)il * (size_t)s_d.max_ctx + (size_t)pos_c) * 1024ULL + (size_t)(kv_h * 128);
        st_cg_f16(&draft_k[dst_off + lane],      __float2half(x0));
        st_cg_f16(&draft_k[dst_off + lane + 32], __float2half(x1));
        st_cg_f16(&draft_k[dst_off + lane + 64], __float2half(x2));
        st_cg_f16(&draft_k[dst_off + lane + 96], __float2half(x3));

        st_cg_f16(&draft_v[dst_off + lane],      __float2half(ld_cg_f32(&v_src[lane])));
        st_cg_f16(&draft_v[dst_off + lane + 32], __float2half(ld_cg_f32(&v_src[lane + 32])));
        st_cg_f16(&draft_v[dst_off + lane + 64], __float2half(ld_cg_f32(&v_src[lane + 64])));
        st_cg_f16(&draft_v[dst_off + lane + 96], __float2half(ld_cg_f32(&v_src[lane + 96])));
    }
    return ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl);
}

// Standalone prefill context KV injection kernel (`grid = (68, 1, 1), block = (512, 1, 1)`)
extern "C" __global__ __launch_bounds__(512, 1)
void ink_dspark_inject_kv_kernel(
    const InkModelDesc* __restrict__ m_g,
    const InkDSparkDesc* __restrict__ d_g,
    volatile InkRunCtrl* __restrict__ ctrl
) {
    __shared__ __align__(16) int           s_xq[8 * (2304 / 4)];
    __shared__ __align__(16) float2        s_xds[8 * 72];
    __shared__ __align__(16) float         s_red[64];
    __shared__ __align__(16) InkModelDesc  s_m;
    __shared__ __align__(16) InkDSparkDesc s_d;
    __shared__ int s_abort;
    __shared__ int s_pos;
    __shared__ int s_cnt;

    const int tid = threadIdx.x;
    for (int i = tid; i < (int)(sizeof(InkModelDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_m)[i] = __ldg(&reinterpret_cast<const uint32_t*>(m_g)[i]);
    }
    for (int i = tid; i < (int)(sizeof(InkDSparkDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_d)[i] = __ldg(&reinterpret_cast<const uint32_t*>(d_g)[i]);
    }
    if (tid == 0) {
        s_abort = 0;
        s_pos = ctrl->inject_pos;
        s_cnt = ctrl->inject_cnt;
    }
    __syncthreads();

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(s_m.ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    if (s_cnt > 0) {
        if (!dev_dspark_inject_kv(s_m, s_d, s_xq, s_xds, s_red, s_pos, s_cnt, bar_state, bar_step, s_abort, ctrl)) {
            return;
        }
    }

    if (blockIdx.x == 0 && tid == 0) {
        asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        __threadfence_system();
        const uint64_t seq = ctrl->expected_seq;
        asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&ctrl->done_seq), "l"(seq) : "memory");
    }
}

// 68-SM Fused Block-8 `DSpark` (`Inkling2MTP`) Proposal Megakernel (`~2.3 ms` per 7 drafted tokens!)
extern "C" __global__ __launch_bounds__(512, 1)
void ink_dspark_propose_megakernel(
    const InkModelDesc* __restrict__ m_g,
    const InkDSparkDesc* __restrict__ d_g,
    volatile InkRunCtrl* __restrict__ ctrl
) {
    union SharedDSpark {
        struct {
            int    xq[8 * (3072 / 4)];   // 24,576 B (also fits 2 * 9216 = 18,432 B!)
            float2 xds[8 * (3072 / 32)]; //  6,144 B (also fits 2 * 288 = 4,608 B!)
        } gemm;
        struct {
            float acc[INK_WARPS * 128];
            float m_arr[INK_WARPS];
            float l_arr[INK_WARPS];
        } attn;
    };
    __shared__ __align__(16) SharedDSpark  s_u;
    __shared__ __align__(16) float         s_red[64];
    __shared__ __align__(16) int32_t       s_idx_red[32];
    __shared__ __align__(16) InkModelDesc  s_m;
    __shared__ __align__(16) InkDSparkDesc s_d;
    __shared__ int s_abort;
    __shared__ int s_anchor_tok;
    __shared__ int s_anchor_pos;
    __shared__ int s_Bd;
    __shared__ int s_gamma;
    __shared__ int s_inj_pos;
    __shared__ int s_inj_cnt;
    __shared__ int s_prev_tok;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_sms = gridDim.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const uint32_t hmask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const int hw_in_cta = (wid << 1) | (lane >> 4);
    const int total_warps = num_sms * INK_WARPS;
    const int global_w_ilv = wid * num_sms + bid;
    const int total_hwarps = num_sms * (INK_WARPS * 2);
    const int global_hw_ilv = hw_in_cta * num_sms + bid;

    for (int i = tid; i < (int)(sizeof(InkModelDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_m)[i] = __ldg(&reinterpret_cast<const uint32_t*>(m_g)[i]);
    }
    for (int i = tid; i < (int)(sizeof(InkDSparkDesc) / 4); i += INK_THREADS) {
        reinterpret_cast<uint32_t*>(&s_d)[i] = __ldg(&reinterpret_cast<const uint32_t*>(d_g)[i]);
    }
    if (tid == 0) {
        s_abort = 0;
        s_anchor_tok = ctrl->start_token;
        s_anchor_pos = ctrl->start_pos;
        s_Bd = ctrl->batch_len;
        const int req_g = ctrl->max_steps;
        s_gamma = (req_g > 0 && req_g < ctrl->batch_len) ? req_g : (ctrl->batch_len - 1);
        s_inj_pos = ctrl->inject_pos;
        s_inj_cnt = ctrl->inject_cnt;
        s_prev_tok = ctrl->start_token;
    }
    __syncthreads();

    uint64_t* bar_state = reinterpret_cast<uint64_t*>(s_m.ws_bar_state);
    uint64_t bar_step = 0;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(bar_step) : "l"(&bar_state[1]));

    // 0. Fused Context KV + `f_rnn` Injection (0 extra doorbells!)
    if (s_inj_cnt > 0) {
        if (!dev_dspark_inject_kv(s_m, s_d, s_u.gemm.xq, s_u.gemm.xds, s_red, s_inj_pos, s_inj_cnt, bar_state, bar_step, s_abort, ctrl)) {
            return;
        }
    }

    const int Bd = s_Bd;            // always 8 for exact trained non-causal block attention!
    const int gamma = s_gamma;      // 1..=7 Markov + base_logits steps!
    const int p_anchor = s_anchor_pos;
    const int dim = 2304;

    // 1. Block Embeddings: `h[0] = tok_norm(tok_embd[anchor])`, `h[1..Bd] = mask_embed`
    if (bid < Bd) {
        float* h_b = reinterpret_cast<float*>(s_d.ws_h_va) + (size_t)bid * dim;
        if (bid == 0) {
            const uint8_t* emb_w = reinterpret_cast<const uint8_t*>(s_m.tok_embd_va);
            const float* tnorm = reinterpret_cast<const float*>(s_m.tok_norm_va);
            float ss = 0.0f;
            for (int col = tid; col < dim; col += INK_THREADS) {
                float v = ink_dequant_elem_emb(emb_w, s_m.tok_embd_type, (size_t)s_anchor_tok, dim, col);
                st_cg_f32(&h_b[col], v);
                ss += v * v;
            }
            ss = ink_warp_sum(ss);
            if (lane == 0) s_red[wid] = ss;
            __syncthreads();
            if (wid == 0) {
                float tot = (lane < INK_WARPS) ? s_red[lane] : 0.0f;
                tot = ink_warp_sum(tot);
                if (lane == 0) s_red[0] = rsqrtf(tot * (1.0f / 2304.0f) + s_m.rms_eps);
            }
            __syncthreads();
            const float sc = s_red[0];
            for (int col = tid; col < dim; col += INK_THREADS) {
                st_cg_f32(&h_b[col], ld_cg_f32(&h_b[col]) * sc * __ldg(&tnorm[col]));
            }
        } else {
            const float* mask_emb = reinterpret_cast<const float*>(s_d.mask_embed_va);
            for (int col = tid; col < dim; col += INK_THREADS) {
                st_cg_f32(&h_b[col], __ldg(&mask_emb[col]));
            }
        }
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    auto load_dspark_q8_1 = [&](int K_cols, int b_beg, int b_cnt) {
        const int n_int4 = K_cols >> 4;
        const int n_blks = K_cols >> 5;
        const int8_t* d_xq = reinterpret_cast<const int8_t*>(s_d.ws_xq_va);
        const float2* d_xds = reinterpret_cast<const float2*>(s_d.ws_xds_va);
        for (int b = 0; b < b_cnt; ++b) {
            const int* g_i32 = reinterpret_cast<const int*>(d_xq + (size_t)(b_beg + b) * 9216ULL);
            const float2* g_ds = d_xds + (size_t)(b_beg + b) * 288ULL;
            int4* s_i4 = reinterpret_cast<int4*>(s_u.gemm.xq + b * (K_cols >> 2));
            float2* s_ds = s_u.gemm.xds + b * n_blks;
            for (int i = tid; i < n_int4; i += INK_THREADS) s_i4[i] = ld_cg_s32x4(g_i32 + (i << 2));
            for (int blk = tid; blk < n_blks; blk += INK_THREADS) s_ds[blk] = ld_cg_f32x2(&g_ds[blk]);
        }
        __syncthreads();
    };

    // 2. 3 Dense Qwen3-Style Non-Causal Block-Attention + SwiGLU Layers
    for (int il = 0; il < 3; ++il) {
        // 2a. `input_layernorm` + `Q8_1` (`CTAs 0..Bd-1`)
        if (bid < Bd) {
            dev_cta_rmsnorm_q8_1(
                nullptr,
                reinterpret_cast<int8_t*>(s_d.ws_xq_va) + (size_t)bid * 9216ULL,
                reinterpret_cast<float2*>(s_d.ws_xds_va) + (size_t)bid * 288ULL,
                s_red,
                reinterpret_cast<const float*>(s_d.ws_h_va) + (size_t)bid * dim,
                reinterpret_cast<const float*>(s_d.layers[il].input_layernorm),
                dim,
                s_d.rms_eps,
                true
            );
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2b. `qkv_proj [5120, 2304]` Split `Q8_0` across all 2,176 half-warps
        {
            load_dspark_q8_1(dim, 0, Bd);
            const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(s_d.layers[il].qkv_proj);
            const half* d_base = reinterpret_cast<const half*>(w_ptr + 5120ULL * 2304ULL);
            float* ws_qkv = reinterpret_cast<float*>(s_d.ws_qkv_va);
            for (int r = global_hw_ilv; r < 5120; r += total_hwarps) {
                const int4* qs_row = reinterpret_cast<const int4*>(w_ptr + (size_t)r * 2304ULL);
                const half* d_row = d_base + (size_t)r * 72ULL;
                float acc[8] = {0,0,0,0,0,0,0,0};
                #pragma unroll 3
                for (int i4 = lane16; i4 < 144; i4 += 16) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < Bd) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (dim >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[b] += (d * s_u.gemm.xds[b * 72 + blk].x) * (float)sumi;
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < Bd) {
                        const float v = ink_hwarp_sum(acc[b], hmask);
                        if (lane16 == 0) st_cg_f32(&ws_qkv[b * 5120 + r], v);
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2c. Per-Head `q_norm` (`24` heads) + `k_norm` (`8` heads) + NeoX RoPE (`1` warp per `(b, qk_head)`)
        {
            float* ws_qkv = reinterpret_cast<float*>(s_d.ws_qkv_va);
            const float* q_nw = reinterpret_cast<const float*>(s_d.layers[il].q_norm);
            const float* k_nw = reinterpret_cast<const float*>(s_d.layers[il].k_norm);
            const int tot_qk = Bd * 32; // 24 Q + 8 K per token
            for (int item = bid * INK_WARPS + wid; item < tot_qk; item += total_warps) {
                const int b = item >> 5;
                const int hk = item & 31;
                float* head_ptr = ws_qkv + b * 5120 + hk * 128;
                float x0 = ld_cg_f32(&head_ptr[lane]);
                float x1 = ld_cg_f32(&head_ptr[lane + 32]);
                float x2 = ld_cg_f32(&head_ptr[lane + 64]);
                float x3 = ld_cg_f32(&head_ptr[lane + 96]);
                dev_warp_head128_rmsnorm_neox_rope(
                    x0, x1, x2, x3,
                    (hk < 24) ? q_nw : k_nw,
                    p_anchor + b,
                    s_d.rope_theta,
                    s_d.rms_eps,
                    lane
                );
                st_cg_f32(&head_ptr[lane],      x0);
                st_cg_f32(&head_ptr[lane + 32], x1);
                st_cg_f32(&head_ptr[lane + 64], x2);
                st_cg_f32(&head_ptr[lane + 96], x3);
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2d. Non-Causal Block + Committed Context GQA Attention (`is_causal = False`) + Fused `Q8_1`
        {
            const float* ws_qkv = reinterpret_cast<const float*>(s_d.ws_qkv_va);
            const half* draft_k = reinterpret_cast<const half*>(s_d.draft_k_va) + (size_t)il * (size_t)s_d.max_ctx * 1024ULL;
            const half* draft_v = reinterpret_cast<const half*>(s_d.draft_v_va) + (size_t)il * (size_t)s_d.max_ctx * 1024ULL;
            const int start_ctx = (s_d.layers[il].is_local != 0) ? max(0, p_anchor - s_d.swa_window) : 0;
            const float inv_sqrt_hd = 0.0883883476f;

            for (int bh = bid; bh < Bd * 24; bh += num_sms) {
                const int b = bh / 24;
                const int h = bh % 24;
                const int kv_h = h / 3;
                const int d0 = lane * 4;

                const float* q_ptr = ws_qkv + b * 5120 + h * 128 + d0;
                const float q0 = ld_cg_f32(&q_ptr[0]), q1 = ld_cg_f32(&q_ptr[1]);
                const float q2 = ld_cg_f32(&q_ptr[2]), q3 = ld_cg_f32(&q_ptr[3]);

                float m_w = -1e30f;
                float l_w = 0.0f;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                // Committed context KV `[start_ctx .. p_anchor - 1]`
                for (int j = start_ctx + wid; j < p_anchor; j += INK_WARPS) {
                    const half2* kj2 = reinterpret_cast<const half2*>(draft_k + (size_t)j * 1024ULL + kv_h * 128 + d0);
                    const half2 k01 = *reinterpret_cast<const half2*>(&kj2[0]);
                    const float kx0 = __half2float(ld_cg_f16(reinterpret_cast<const half*>(&kj2[0]) + 0));
                    const float kx1 = __half2float(ld_cg_f16(reinterpret_cast<const half*>(&kj2[0]) + 1));
                    const float kx2 = __half2float(ld_cg_f16(reinterpret_cast<const half*>(&kj2[1]) + 0));
                    const float kx3 = __half2float(ld_cg_f16(reinterpret_cast<const half*>(&kj2[1]) + 1));
                    (void)k01;
                    float dot = q0 * kx0 + q1 * kx1 + q2 * kx2 + q3 * kx3;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                    const float score = dot * inv_sqrt_hd;
                    const float m_new = fmaxf(m_w, score);
                    const float alpha = expf(m_w - m_new);
                    const float beta  = expf(score - m_new);
                    l_w = l_w * alpha + beta;
                    m_w = m_new;

                    const half* vj = draft_v + (size_t)j * 1024ULL + kv_h * 128 + d0;
                    acc0 = acc0 * alpha + beta * __half2float(ld_cg_f16(&vj[0]));
                    acc1 = acc1 * alpha + beta * __half2float(ld_cg_f16(&vj[1]));
                    acc2 = acc2 * alpha + beta * __half2float(ld_cg_f16(&vj[2]));
                    acc3 = acc3 * alpha + beta * __half2float(ld_cg_f16(&vj[3]));
                }

                // Bidirectional current block KV `s = 0 .. Bd - 1` (`wid < Bd`)
                if (wid < Bd) {
                    const int s = wid;
                    const float* ks = ws_qkv + s * 5120 + 3072 + kv_h * 128 + d0;
                    const float* vs = ws_qkv + s * 5120 + 4096 + kv_h * 128 + d0;
                    float dot = q0 * ld_cg_f32(&ks[0]) + q1 * ld_cg_f32(&ks[1])
                              + q2 * ld_cg_f32(&ks[2]) + q3 * ld_cg_f32(&ks[3]);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                    const float score = dot * inv_sqrt_hd;
                    const float m_new = fmaxf(m_w, score);
                    const float alpha = expf(m_w - m_new);
                    const float beta  = expf(score - m_new);
                    l_w = l_w * alpha + beta;
                    m_w = m_new;

                    acc0 = acc0 * alpha + beta * ld_cg_f32(&vs[0]);
                    acc1 = acc1 * alpha + beta * ld_cg_f32(&vs[1]);
                    acc2 = acc2 * alpha + beta * ld_cg_f32(&vs[2]);
                    acc3 = acc3 * alpha + beta * ld_cg_f32(&vs[3]);
                }

                if (lane == 0) { s_u.attn.m_arr[wid] = m_w; s_u.attn.l_arr[wid] = l_w; }
                s_u.attn.acc[wid * 128 + d0 + 0] = acc0;
                s_u.attn.acc[wid * 128 + d0 + 1] = acc1;
                s_u.attn.acc[wid * 128 + d0 + 2] = acc2;
                s_u.attn.acc[wid * 128 + d0 + 3] = acc3;
                __syncthreads();

                if (wid < 4) {
                    float m_max = -1e30f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_u.attn.l_arr[w] > 0.0f) m_max = fmaxf(m_max, s_u.attn.m_arr[w]);
                    }
                    float denom = 0.0f, out_d = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < INK_WARPS; ++w) {
                        if (s_u.attn.l_arr[w] > 0.0f) {
                            float ew = expf(s_u.attn.m_arr[w] - m_max);
                            denom += s_u.attn.l_arr[w] * ew;
                            out_d += s_u.attn.acc[w * 128 + tid] * ew;
                        }
                    }
                    const float v = out_d / fmaxf(denom, 1e-20f);
                    const int col = h * 128 + tid;
                    float sum_x = v, amax = fabsf(v);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
                    }
                    const float d = amax * (1.0f / 127.0f);
                    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    st_cg_s8(&reinterpret_cast<int8_t*>(s_d.ws_xq_va)[b * 9216 + col], max(-127, min(127, __float2int_rn(v * id))));
                    if (lane == 0) st_cg_f32x2(&reinterpret_cast<float2*>(s_d.ws_xds_va)[b * 288 + h * 4 + wid], make_float2(d, sum_x));
                }
                __syncthreads();
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2e. 4-Way K-Sliced `o_proj [2304, 3072]` (`9,216` Half-Warps) + Residual + `post_attn_norm`
        {
            load_dspark_q8_1(3072, 0, Bd);
            const uint8_t* wo_ptr = reinterpret_cast<const uint8_t*>(s_d.layers[il].o_proj);
            const half* d_base = reinterpret_cast<const half*>(wo_ptr + 2304ULL * 3072ULL);
            float* ws_part = reinterpret_cast<float*>(s_d.ws_mlp_va); // [4 * 8 * 2304 = 73,728 floats]

            for (int item = global_hw_ilv; item < 4 * dim; item += total_hwarps) {
                const int kp = item / dim;
                const int c = item - kp * dim;
                const int4* qs_row = reinterpret_cast<const int4*>(wo_ptr + (size_t)c * 3072ULL);
                const half* d_row = d_base + (size_t)c * 96ULL;
                float acc[8] = {0,0,0,0,0,0,0,0};
                const int i4_beg = kp * 48;
                #pragma unroll 3
                for (int i4 = i4_beg + lane16; i4 < i4_beg + 48; i4 += 16) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < Bd) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (3072 >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[b] += (d * s_u.gemm.xds[b * 96 + blk].x) * (float)sumi;
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < Bd) {
                        const float v = ink_hwarp_sum(acc[b], hmask);
                        if (lane16 == 0) st_cg_f32(&ws_part[(kp * 8 + b) * dim + c], v);
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        if (bid < Bd) {
            const int b = bid;
            float* h_b = reinterpret_cast<float*>(s_d.ws_h_va) + (size_t)b * dim;
            const float* ws_part = reinterpret_cast<const float*>(s_d.ws_mlp_va);
            for (int c = tid; c < dim; c += INK_THREADS) {
                const float ov = (ld_cg_f32(&ws_part[(0 * 8 + b) * dim + c]) + ld_cg_f32(&ws_part[(1 * 8 + b) * dim + c]))
                               + (ld_cg_f32(&ws_part[(2 * 8 + b) * dim + c]) + ld_cg_f32(&ws_part[(3 * 8 + b) * dim + c]));
                st_cg_f32(&h_b[c], ld_cg_f32(&h_b[c]) + ov);
            }
            __syncthreads();
            dev_cta_rmsnorm_q8_1(
                nullptr,
                reinterpret_cast<int8_t*>(s_d.ws_xq_va) + (size_t)b * 9216ULL,
                reinterpret_cast<float2*>(s_d.ws_xds_va) + (size_t)b * 288ULL,
                s_red,
                h_b,
                reinterpret_cast<const float*>(s_d.layers[il].post_attn_norm),
                dim,
                s_d.rms_eps,
                true
            );
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2f. Fused `gate_up_proj [18432, 2304]` + `SiLU(gate) * up` (`9,216` Half-Warp Items!)
        {
            load_dspark_q8_1(dim, 0, Bd);
            const uint8_t* gu_ptr = reinterpret_cast<const uint8_t*>(s_d.layers[il].gate_up_proj);
            const half* d_base = reinterpret_cast<const half*>(gu_ptr + 18432ULL * 2304ULL);
            float* ws_mlp = reinterpret_cast<float*>(s_d.ws_mlp_va);

            for (int ff = global_hw_ilv; ff < 9216; ff += total_hwarps) {
                const int4* qs_g = reinterpret_cast<const int4*>(gu_ptr + (size_t)ff * 2304ULL);
                const int4* qs_u = reinterpret_cast<const int4*>(gu_ptr + (size_t)(9216 + ff) * 2304ULL);
                const half* d_g_row = d_base + (size_t)ff * 72ULL;
                const half* d_u_row = d_base + (size_t)(9216 + ff) * 72ULL;

                float g_acc[8] = {0,0,0,0,0,0,0,0};
                float u_acc[8] = {0,0,0,0,0,0,0,0};
                #pragma unroll 2
                for (int i4 = lane16; i4 < 144; i4 += 16) {
                    const int4 wvg = __ldg(&qs_g[i4]);
                    const int4 wvu = __ldg(&qs_u[i4]);
                    const int blk = i4 >> 1;
                    const float dg = __half2float(__ldg(&d_g_row[blk]));
                    const float du = __half2float(__ldg(&d_u_row[blk]));
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        if (b < Bd) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + b * (dim >> 2))[i4];
                            const float dx = s_u.gemm.xds[b * 72 + blk].x;
                            int sg = __dp4a(wvg.x, xv.x, 0); sg = __dp4a(wvg.y, xv.y, sg);
                            sg = __dp4a(wvg.z, xv.z, sg); sg = __dp4a(wvg.w, xv.w, sg);
                            int su = __dp4a(wvu.x, xv.x, 0); su = __dp4a(wvu.y, xv.y, su);
                            su = __dp4a(wvu.z, xv.z, su); su = __dp4a(wvu.w, xv.w, su);
                            g_acc[b] += (dg * dx) * (float)sg;
                            u_acc[b] += (du * dx) * (float)su;
                        }
                    }
                }
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    if (b < Bd) {
                        const float gv = ink_hwarp_sum(g_acc[b], hmask);
                        const float uv = ink_hwarp_sum(u_acc[b], hmask);
                        if (lane16 == 0) {
                            const float silu_g = gv / (1.0f + expf(-gv));
                            st_cg_f32(&ws_mlp[b * 9216 + ff], silu_g * uv);
                        }
                    }
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        if (bid < Bd) {
            dev_cta_rmsnorm_q8_1(
                nullptr,
                reinterpret_cast<int8_t*>(s_d.ws_xq_va) + (size_t)bid * 9216ULL,
                reinterpret_cast<float2*>(s_d.ws_xds_va) + (size_t)bid * 288ULL,
                s_red,
                reinterpret_cast<const float*>(s_d.ws_mlp_va) + (size_t)bid * 9216ULL,
                nullptr,
                9216,
                1e-6f,
                false
            );
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 2g. `down_proj [2304, 9216]` (`b_beg = 0, 2, 4, 6` so `2 * 9216 = 18.4 KB` fits SMEM!)
        for (int b_beg = 0; b_beg < Bd; b_beg += 2) {
            const int b_cnt = min(2, Bd - b_beg);
            load_dspark_q8_1(9216, b_beg, b_cnt);
            const uint8_t* dn_ptr = reinterpret_cast<const uint8_t*>(s_d.layers[il].down_proj);
            const half* d_base = reinterpret_cast<const half*>(dn_ptr + 2304ULL * 9216ULL);
            float* ws_part = reinterpret_cast<float*>(s_d.ws_qkv_va); // [4 * 2 * 2304 = 18,432 floats <= 49,152!]

            for (int item = global_hw_ilv; item < 4 * dim; item += total_hwarps) {
                const int kp = item / dim;
                const int c = item - kp * dim;
                const int4* qs_row = reinterpret_cast<const int4*>(dn_ptr + (size_t)c * 9216ULL);
                const half* d_row = d_base + (size_t)c * 288ULL;
                float acc0 = 0.0f, acc1 = 0.0f;
                const int i4_beg = kp * 144;
                #pragma unroll 3
                for (int i4 = i4_beg + lane16; i4 < i4_beg + 144; i4 += 16) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    {
                        const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq)[i4];
                        int s0 = __dp4a(wv.x, xv.x, 0); s0 = __dp4a(wv.y, xv.y, s0);
                        s0 = __dp4a(wv.z, xv.z, s0); s0 = __dp4a(wv.w, xv.w, s0);
                        acc0 += (d * s_u.gemm.xds[blk].x) * (float)s0;
                    }
                    if (b_cnt > 1) {
                        const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + (9216 >> 2))[i4];
                        int s1 = __dp4a(wv.x, xv.x, 0); s1 = __dp4a(wv.y, xv.y, s1);
                        s1 = __dp4a(wv.z, xv.z, s1); s1 = __dp4a(wv.w, xv.w, s1);
                        acc1 += (d * s_u.gemm.xds[288 + blk].x) * (float)s1;
                    }
                }
                const float v0 = ink_hwarp_sum(acc0, hmask);
                const float v1 = ink_hwarp_sum(acc1, hmask);
                if (lane16 == 0) {
                    st_cg_f32(&ws_part[(kp * 2 + 0) * dim + c], v0);
                    if (b_cnt > 1) st_cg_f32(&ws_part[(kp * 2 + 1) * dim + c], v1);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

            for (int b = 0; b < b_cnt; ++b) {
                float* h_b = reinterpret_cast<float*>(s_d.ws_h_va) + (size_t)(b_beg + b) * dim;
                for (int c = bid * INK_THREADS + tid; c < dim; c += num_sms * INK_THREADS) {
                    const float dv = (ld_cg_f32(&ws_part[(0 * 2 + b) * dim + c]) + ld_cg_f32(&ws_part[(1 * 2 + b) * dim + c]))
                                   + (ld_cg_f32(&ws_part[(2 * 2 + b) * dim + c]) + ld_cg_f32(&ws_part[(3 * 2 + b) * dim + c]));
                    st_cg_f32(&h_b[c], ld_cg_f32(&h_b[c]) + dv);
                }
            }
            if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;
        }
    }

    // 3. Final `norm` (`s = 1 .. Bd - 1`) + Single-Read Shared `output.weight` Base Logits -> L2 `half [gamma, vocab]`
    if (bid < gamma) {
        const int g = bid;
        float* h_s = reinterpret_cast<float*>(s_d.ws_h_va) + (size_t)(1 + g) * dim;
        dev_cta_rmsnorm_q8_1(
            h_s, // overwrite `ws_h[1 + g]` in-place with post-norm `d_hid` for Markov RNN!
            reinterpret_cast<int8_t*>(s_d.ws_xq_va) + (size_t)g * 9216ULL,
            reinterpret_cast<float2*>(s_d.ws_xds_va) + (size_t)g * 288ULL,
            s_red,
            h_s,
            reinterpret_cast<const float*>(s_d.final_norm_va),
            dim,
            s_d.rms_eps,
            true
        );
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    {
        load_dspark_q8_1(dim, 0, gamma);
        const uint8_t* out_w = reinterpret_cast<const uint8_t*>(s_m.output_w_va);
        const int vocab = s_m.vocab_unpadded;
        half* base_logits = reinterpret_cast<half*>(s_d.ws_base_logits_va);
        const size_t tot_sb_out = (size_t)vocab * 9ULL;
        const half* d_out_q6 = reinterpret_cast<const half*>(out_w + tot_sb_out * 208ULL);
        const half* d_out_q8 = reinterpret_cast<const half*>(out_w + (size_t)vocab * 2304ULL);
        const int sb_lane = lane >> 4, tid16 = lane & 15;
        const int h = tid16 >> 3, p = (tid16 >> 2) & 1, q = (tid16 >> 1) & 1, half8 = tid16 & 1;
        const int ql_off = (h << 6) + (q << 5) + (half8 << 4);
        const int qh_off = 128 + (h << 5) + (half8 << 4);
        const int ql_sh = p << 2, qh_sh = ((p << 1) + q) << 1;

        for (int v = global_w_ilv; v < vocab; v += total_warps) {
            float acc[7] = {0,0,0,0,0,0,0};
            if (s_m.output_w_type == 14) {
                const uint8_t* w_row = out_w + (size_t)v * (9ULL * 208ULL);
                const half* d_row = d_out_q6 + (size_t)v * 9ULL;
                #pragma unroll
                for (int sb = sb_lane; sb < 9; sb += 2) {
                    const uint8_t* blk = w_row + sb * 208;
                    const float d_sc = __half2float(__ldg(&d_row[sb])) *
                                       (float)__ldg(&reinterpret_cast<const int8_t*>(blk + 192)[tid16]);
                    const uint4 ql = __ldg(reinterpret_cast<const uint4*>(blk + ql_off));
                    const uint4 qh = __ldg(reinterpret_cast<const uint4*>(blk + qh_off));
                    const uint32_t u6_0 = ((ql.x >> ql_sh) & 0x0f0f0f0fu) | (((qh.x >> qh_sh) & 0x03030303u) << 4);
                    const uint32_t u6_1 = ((ql.y >> ql_sh) & 0x0f0f0f0fu) | (((qh.y >> qh_sh) & 0x03030303u) << 4);
                    const uint32_t u6_2 = ((ql.z >> ql_sh) & 0x0f0f0f0fu) | (((qh.z >> qh_sh) & 0x03030303u) << 4);
                    const uint32_t u6_3 = ((ql.w >> ql_sh) & 0x0f0f0f0fu) | (((qh.w >> qh_sh) & 0x03030303u) << 4);
                    const uint32_t m0b = (~u6_0) & 0x20202020u, m1b = (~u6_1) & 0x20202020u;
                    const uint32_t m2b = (~u6_2) & 0x20202020u, m3b = (~u6_3) & 0x20202020u;
                    const int s6_0 = (int)((u6_0 & 0x1f1f1f1fu) | m0b | (m0b << 1) | (m0b << 2));
                    const int s6_1 = (int)((u6_1 & 0x1f1f1f1fu) | m1b | (m1b << 1) | (m1b << 2));
                    const int s6_2 = (int)((u6_2 & 0x1f1f1f1fu) | m2b | (m2b << 1) | (m2b << 2));
                    const int s6_3 = (int)((u6_3 & 0x1f1f1f1fu) | m3b | (m3b << 1) | (m3b << 2));
                    const int x_base_int = (sb << 6) + (tid16 << 2);
                    const int ds_idx = (sb << 3) + (tid16 >> 1);
                    #pragma unroll
                    for (int g = 0; g < 7; ++g) {
                        if (g < gamma) {
                            const int4 xv = *reinterpret_cast<const int4*>(s_u.gemm.xq + g * (dim >> 2) + x_base_int);
                            int sumi = __dp4a(s6_0, xv.x, 0); sumi = __dp4a(s6_1, xv.y, sumi);
                            sumi = __dp4a(s6_2, xv.z, sumi); sumi = __dp4a(s6_3, xv.w, sumi);
                            acc[g] += (d_sc * s_u.gemm.xds[g * 72 + ds_idx].x) * (float)sumi;
                        }
                    }
                }
            } else {
                const int4* qs_row = reinterpret_cast<const int4*>(out_w + (size_t)v * 2304ULL);
                const half* d_row = d_out_q8 + (size_t)v * 72ULL;
                #pragma unroll 2
                for (int i4 = lane; i4 < 144; i4 += 32) {
                    const int4 wv = __ldg(&qs_row[i4]);
                    const int blk = i4 >> 1;
                    const float d = __half2float(__ldg(&d_row[blk]));
                    #pragma unroll
                    for (int g = 0; g < 7; ++g) {
                        if (g < gamma) {
                            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq + g * (dim >> 2))[i4];
                            int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                            sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                            acc[g] += (d * s_u.gemm.xds[g * 72 + blk].x) * (float)sumi;
                        }
                    }
                }
            }
            #pragma unroll
            for (int g = 0; g < 7; ++g) {
                if (g < gamma) {
                    const float val = ink_warp_sum(acc[g]);
                    if (lane == 0) st_cg_f16(&base_logits[(size_t)g * (size_t)vocab + (size_t)v], __float2half(val));
                }
            }
        }
    }
    if (bid == 0 && tid < 256) {
        st_cg_f32(&reinterpret_cast<float*>(s_d.ws_rnn_state_va)[tid], 0.0f);
    }
    if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

    // 4. Sequential `gamma`-Step Markov RNN Head (`g = 0 .. gamma - 1`, ~0.12 ms/step!)
    const int vocab = s_m.vocab_unpadded;
    for (int g = 0; g < gamma; ++g) {
        // 4a. Build `ws_rnn_inp [5120]` = `[state(256), markov_w1[prev](256), d_hid[g](2304), f_rnn(2304)]` + `Q8_1` (`CTA 0`)
        if (bid == 0) {
            float* rnn_inp = reinterpret_cast<float*>(s_d.ws_rnn_inp_va);
            const float* st = reinterpret_cast<const float*>(s_d.ws_rnn_state_va);
            const uint16_t* mw1 = reinterpret_cast<const uint16_t*>(s_d.markov_w1_bf16_va) + (size_t)s_prev_tok * 256ULL;
            const float* d_hid = reinterpret_cast<const float*>(s_d.ws_h_va) + (size_t)(1 + g) * dim;
            const float* f_rnn = reinterpret_cast<const float*>(s_d.ws_f_rnn_va);

            if (tid < 256) {
                st_cg_f32(&rnn_inp[tid], ld_cg_f32(&st[tid]));
                st_cg_f32(&rnn_inp[256 + tid], ink_bf16_to_f32(__ldg(&mw1[tid])));
            }
            for (int col = tid; col < dim; col += INK_THREADS) {
                st_cg_f32(&rnn_inp[512 + col], ld_cg_f32(&d_hid[col]));
                st_cg_f32(&rnn_inp[512 + dim + col], ld_cg_f32(&f_rnn[col]));
            }
            __syncthreads();
            dev_cta_rmsnorm_q8_1(
                nullptr,
                reinterpret_cast<int8_t*>(s_d.ws_xq_va),
                reinterpret_cast<float2*>(s_d.ws_xds_va),
                s_red,
                rnn_inp,
                nullptr,
                5120,
                1e-6f,
                false
            );
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 4b. `rnn_proj [768, 5120]` Split `Q8_0` across `CTAs 0..47` (`768` warps = 1 row/warp!)
        if (bid < 48) {
            dev_load_q8_1_smem(
                s_u.gemm.xq, s_u.gemm.xds,
                reinterpret_cast<const int8_t*>(s_d.ws_xq_va),
                reinterpret_cast<const float2*>(s_d.ws_xds_va),
                5120
            );
            const int r = bid * INK_WARPS + wid;
            if (r < 768) {
                const float dot_v = dev_warp_dot_q8_0_aligned(
                    reinterpret_cast<const uint8_t*>(s_d.rnn_proj_w_va),
                    768ULL, (size_t)r,
                    s_u.gemm.xq, s_u.gemm.xds,
                    5120, lane
                );
                if (lane == 0) {
                    const float b_v = __ldg(&reinterpret_cast<const float*>(s_d.rnn_proj_b_va)[r]);
                    st_cg_f32(&reinterpret_cast<float*>(s_d.ws_rnn_proj_va)[r], dot_v + b_v);
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 4c. `rnn_gates` + Quantize `latent [256]` to `Q8_1` (`CTA 0`, warps `0..7`)
        if (bid == 0 && tid < 256) {
            const float* rp = reinterpret_cast<const float*>(s_d.ws_rnn_proj_va);
            float* st = reinterpret_cast<float*>(s_d.ws_rnn_state_va);
            const float g_raw = ld_cg_f32(&rp[tid]);
            const float c_raw = ld_cg_f32(&rp[256 + tid]);
            const float o_raw = ld_cg_f32(&rp[512 + tid]);
            const float gate = 1.0f / (1.0f + expf(-g_raw));
            const float new_st = gate * ld_cg_f32(&st[tid]) + (1.0f - gate) * tanhf(c_raw);
            st_cg_f32(&st[tid], new_st);

            const float latent = tanhf(o_raw);
            float sum_x = latent, amax = fabsf(latent);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sum_x += __shfl_xor_sync(0xffffffff, sum_x, off);
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
            }
            const float d = amax * (1.0f / 127.0f);
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            st_cg_s8(&reinterpret_cast<int8_t*>(s_d.ws_xq_va)[tid], max(-127, min(127, __float2int_rn(latent * id))));
            if (lane == 0) st_cg_f32x2(&reinterpret_cast<float2*>(s_d.ws_xds_va)[wid], make_float2(d, sum_x));
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 4d. `_fold_weight [200058, 256]` (`1` single `LDG.E.128` per half-warp lane!) + L2 `base_logits` + Argmax
        {
            dev_load_q8_1_smem(
                s_u.gemm.xq, s_u.gemm.xds,
                reinterpret_cast<const int8_t*>(s_d.ws_xq_va),
                reinterpret_cast<const float2*>(s_d.ws_xds_va),
                256
            );
            const uint8_t* fw = reinterpret_cast<const uint8_t*>(s_d.fold_w_va);
            const half* fd = reinterpret_cast<const half*>(fw + (size_t)vocab * 256ULL);
            const float* fb = reinterpret_cast<const float*>(s_d.fold_b_va);
            const half* base_g = reinterpret_cast<const half*>(s_d.ws_base_logits_va) + (size_t)g * (size_t)vocab;
            const int4 xv = reinterpret_cast<const int4*>(s_u.gemm.xq)[lane16];
            const float dx = s_u.gemm.xds[lane16 >> 1].x;

            float best_v = -1e38f;
            int32_t best_i = 0x7fffffff;

            for (int v = global_hw_ilv; v < vocab; v += total_hwarps) {
                const int4 wv = __ldg(&reinterpret_cast<const int4*>(fw + (size_t)v * 256ULL)[lane16]);
                const float dw = __half2float(__ldg(&fd[(size_t)v * 8ULL + (lane16 >> 1)]));
                int sumi = __dp4a(wv.x, xv.x, 0); sumi = __dp4a(wv.y, xv.y, sumi);
                sumi = __dp4a(wv.z, xv.z, sumi); sumi = __dp4a(wv.w, xv.w, sumi);
                const float dot_hw = ink_hwarp_sum((dw * dx) * (float)sumi, hmask);
                if (lane16 == 0) {
                    const float tot = __half2float(ld_cg_f16(&base_g[v])) + __ldg(&fb[v]) + dot_hw;
                    if (tot > best_v || (tot == best_v && v < best_i)) {
                        best_v = tot;
                        best_i = v;
                    }
                }
            }

            // Reduce 32 half-warps (`hw_in_cta = 0..31`) inside CTA!
            if (lane16 == 0) {
                s_red[hw_in_cta] = best_v;
                s_idx_red[hw_in_cta] = best_i;
            }
            __syncthreads();
            if (wid == 0) {
                float wv = s_red[lane];
                int32_t wi = s_idx_red[lane];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, wv, off);
                    int32_t oi = __shfl_down_sync(0xffffffff, wi, off);
                    if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
                }
                if (lane == 0) {
                    st_cg_f32(&reinterpret_cast<float*>(s_m.ws_cta_val)[bid], wv);
                    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(&reinterpret_cast<int32_t*>(s_m.ws_cta_idx)[bid]), "r"(wi) : "memory");
                }
            }
        }
        if (!ink_grid_barrier(&bar_state[0], bar_step, num_sms, s_abort, ctrl)) return;

        // 4e. All-SM Argmax Reduction -> `s_prev_tok` & `ctrl->batch_out_toks[g]`
        if (wid < 4) {
            const int idx_cta = tid;
            float bv = (idx_cta < num_sms) ? ld_cg_f32(&reinterpret_cast<const float*>(s_m.ws_cta_val)[idx_cta]) : -1e38f;
            int32_t bi = 0x7fffffff;
            if (idx_cta < num_sms) {
                asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(bi) : "l"(&reinterpret_cast<const int32_t*>(s_m.ws_cta_idx)[idx_cta]));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int32_t oi = __shfl_down_sync(0xffffffff, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { s_red[wid] = bv; s_idx_red[wid] = bi; }
        }
        __syncthreads();
        if (tid == 0) {
            float best_val = s_red[0];
            int32_t win_tok = s_idx_red[0];
            for (int w = 1; w < 4; ++w) {
                if (s_red[w] > best_val || (s_red[w] == best_val && s_idx_red[w] < win_tok)) {
                    best_val = s_red[w]; win_tok = s_idx_red[w];
                }
            }
            s_prev_tok = win_tok;
            if (bid == 0) {
                st_host_s32(&ctrl->batch_out_toks[g], win_tok);
            }
        }
        __syncthreads();
    }

    if (bid == 0 && tid == 0) {
        asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(&bar_state[1]), "l"(bar_step) : "memory");
        __threadfence_system();
        const uint64_t seq = ctrl->expected_seq;
        asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&ctrl->done_seq), "l"(seq) : "memory");
    }
}
