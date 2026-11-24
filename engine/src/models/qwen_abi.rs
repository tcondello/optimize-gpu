//! Shared Host/Device `sm_86` Megakernel ABI Structs and Embedded `.cubin` Image.
//! Pure data definitions (`0` methods) shared by `models::qwen`, `models::drafter`, and `voice::{asr, tts}`
//! so `models` and `voice` remain strictly acyclic.

#![allow(dead_code)]

pub static EMBEDDED_CUBIN: &[u8] = include_bytes!("../../kernels/qwen_sm86_megakernel.cubin");

pub const DEFAULT_SPEC_CROSSOVER_4B: usize = 1536;
pub const DEFAULT_SPEC_CROSSOVER_8B: usize = 3072;

#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct QwenLayerDesc {
    pub attn_norm_w: u64,
    pub ffn_norm_w: u64,
    pub q_norm_w: u64,
    pub k_norm_w: u64,

    pub w_qkv_or_qk: u64,
    pub b_qkv_or_qk: u64,
    pub w_v: u64,
    pub b_v: u64,

    pub w_o: u64,
    pub b_o: u64,

    pub w_gate_up: u64,
    pub w_up_sep: u64,
    pub w_down: u64,
    pub b_down: u64,

    pub kv_cache: u64,

    pub qkv_type: i32,
    pub v_type: i32,
    pub o_type: i32,
    pub gate_up_type: i32,
    pub up_sep_type: i32,
    pub down_type: i32,
    pub split_v: i32,
    pub split_gate_up: i32,

    pub lora_ptr: u64,
    pub _pad_lora: u64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct QwenModelDesc {
    pub layers_ptr: u64,
    pub emb_w: u64,
    pub output_norm_w: u64,
    pub lm_head_w: u64,
    pub freqs_cis: u64,

    pub ws_x_res: u64,
    pub ws_x_res2: u64,
    pub ws_xq: u64,
    pub ws_xds: u64,
    pub ws_qkv: u64,
    pub ws_q_roped: u64,
    pub ws_gate_up: u64,
    pub ws_cta_val: u64,
    pub ws_cta_idx: u64,
    pub ws_bar_state: u64,

    pub num_layers: i32,
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub vocab_size: i32,
    pub max_ctx: i32,
    pub has_qk_norm: i32,
    pub emb_type: i32,
    pub lm_head_type: i32,
    pub norm_eps: f32,
    pub inv_sqrt_head_dim: f32,
    pub stop_id0: i32,
    pub stop_id1: i32,
    pub stop_id2: i32,
    pub stop_id3: i32,

    pub ws_ext_embed: u64,
    pub ws_hidden_last: u64,
    pub ws_logits_out: u64,
    pub _pad_model: u64,
}

#[repr(C, align(64))]
pub struct QwenDecodeCtrl {
    pub start_token: i32,
    pub start_pos: i32,
    pub max_steps: i32,
    pub temperature: f32,
    pub rng_seed: u64,
    pub cancel_flag: i32,
    pub error_code: i32,
    pub steps_completed: i32,
    pub stop_reason: i32,
    pub spec_tokens_in: [i32; 4],
    pub spec_tokens_out: [i32; 4],
    pub export_logits: i32,
    pub _pad_ctrl: [i32; 3],
    pub tokens_out: [i32; 4096],
}

#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub cached_prefix_tokens: usize,
    pub prefill_new_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub decode_tok_per_s: f64,
    pub prefill_tok_per_s: f64,
    pub kernel_launches: usize,
    pub spec_verify_calls: usize,
    pub spec_accepted_extra: usize,
    pub spec_crossover_pos: usize,
    pub spec_crossover_switched: bool,
    pub stop_reason: &'static str,
}
