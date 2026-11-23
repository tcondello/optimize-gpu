//! 100%-on-RTX-3080 (`GA102`, `sm_86`) Roofline Engine for Thinking Machines `Inkling-2-Mini`
//! (`inkling2_loop`: 26.4B Total / 3.4B–5.4B Active Looped Sparse MoE, 24 Physical Layers ->
//! 48 Logical Evaluations, 128 Experts Top-8 SITU, ShortConv, Relative-Position + Sink GQA,
//! and Depth Attention Residuals).
//!
//! Architecture v3:
//! - 68-SM Persistent Multi-Token `__dp4a` Megakernel (`0` per-token launches/doorbells)
//! - 68-SM Batched `BM <= 8` Expert-Deduplicated Prefill & Speculative Verify Megakernel
//! - Multi-Turn Exact KV + Recurrent `ShortConv` Prefix Caching & `5 us` VRAM Rollback
//!
//! Zero external Rust crates, zero PyTorch, zero runtime Docker containers (`AGENTS.md` compliant).

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::gguf::{
    f32_to_f16_bits, resolve_model_path, BpeTokenizer, GgufFile, GgufVal, MmapFile,
    Utf8StreamDecoder,
};
use crate::nv_defs::{get_u16, get_u64, put_u64};
use crate::nv_hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

pub static INKLING_CUBIN: &[u8] = include_bytes!("../kernels/inkling_sm86.cubin");

pub const TOK_END_SAMPLING: i32 = 200006; // <|content_model_end_sampling|>
pub const TOK_EOT: i32 = 199999;          // <|endoftext|>
pub const TOK_MSG_MODEL: i32 = 200001;    // <|message_model|>
pub const TOK_CONTENT_TEXT: i32 = 200004; // <|content_text|>
pub const TOK_CONTENT_THINK: i32 = 200008;// <|content_thinking|>
pub const TOK_END_MESSAGE: i32 = 200010;  // <|end_message|>

// -----------------------------------------------------------------------------
// Host / Device `#[repr(C)]` ABI Descriptors (Exact Match with `inkling_sm86.cu`)
// -----------------------------------------------------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct InkPhysLayerDesc {
    pub attn_norm: u64,
    pub attn_q_inp_norm: u64,
    pub attn_k_inp_norm: u64,
    pub attn_v_inp_norm: u64,
    pub attn_res_query: u64,
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub wo: u64,
    pub wqkv_gate: u64,
    pub attn_rel: u64,
    pub attn_rel_bias: u64,
    pub attn_sink_proj: u64,
    pub attn_q_clip: u64,
    pub attn_rel_clip: u64,
    pub attn_k_conv: u64,
    pub attn_v_conv: u64,
    pub attn_k_conv_proj: u64,
    pub attn_v_conv_proj: u64,
    pub attn_post_conv: u64,
    pub ffn_res_query: u64,
    pub ffn_norm: u64,
    pub ffn_gate_inp: u64,
    pub exp_probs_b: u64,
    pub ffn_gate_up_exps: u64,
    pub ffn_down_exps: u64,
    pub ffn_post_conv: u64,
    pub is_swa: i32,
    pub qkv_type: i32,
    pub wo_type: i32,
    pub gate_up_type: i32,
    pub down_type: i32,
    pub _pad: [i32; 5],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct InkLoopCtrlDesc {
    pub msg_query: u64,
    pub ctrl_down: [u64; 4],
    pub ctrl_gain: [u64; 4],
    pub ctrl_forget: [u64; 4],
    pub _pad: [u64; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct InkModelDesc {
    pub layers_ptr: u64,
    pub loop_ctrls_ptr: u64,
    pub slot_kv_dense_ptr: u64,
    pub tok_embd_va: u64,
    pub tok_norm_va: u64,
    pub output_w_va: u64,
    pub output_norm_va: u64,
    pub output_res_query_va: u64,

    pub swa_k_va: u64,
    pub swa_v_va: u64,
    pub full_k_va: u64,
    pub full_v_va: u64,
    pub conv_state_va: u64,
    pub conv_backup_va: u64,

    pub ws_cur: u64,
    pub ws_inp_sa: u64,
    pub ws_ffn_inp: u64,
    pub ws_prelude: u64,
    pub ws_bank_out: u64,
    pub ws_blocks: u64,
    pub ws_prelude_blocks: u64,
    pub ws_bank: u64,
    pub ws_messages: u64,
    pub ws_mixed: u64,
    pub ws_inputs: u64,
    pub ws_norm_in: u64,
    pub ws_ctrl_cat: u64,
    pub ws_ctrl_gate: u64,
    pub ws_ctrl_gain: u64,
    pub ws_ctrl_forget: u64,

    pub ws_xq: u64,
    pub ws_xds: u64,
    pub ws_xq_moe: u64,
    pub ws_xds_moe: u64,

    pub ws_q: u64,
    pub ws_k_raw: u64,
    pub ws_v_raw: u64,
    pub ws_r: u64,
    pub ws_sinks: u64,
    pub ws_gate_raw: u64,
    pub ws_dyn_k: u64,
    pub ws_dyn_v: u64,
    pub ws_attn_gated: u64,

    pub ws_router_logits: u64,
    pub ws_top_ids: u64,
    pub ws_top_weights: u64,
    pub ws_uniq_exp_id: u64,
    pub ws_uniq_mask: u64,
    pub ws_uniq_slots: u64,
    pub ws_n_uniq: u64,
    pub ws_situ: u64,

    pub ws_cta_val: u64,
    pub ws_cta_idx: u64,
    pub ws_bar_state: u64,
    pub ws_dspark_taps: u64,
    pub ws_logits: u64,

    pub dim: i32,
    pub n_ff_exp: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub n_experts: i32,
    pub n_experts_used: i32,
    pub n_phys_layers: i32,
    pub loop_start: i32,
    pub loop_end: i32,
    pub loop_passes: i32,
    pub gate_rank: i32,
    pub vocab_unpadded: i32,
    pub swa_window: i32,
    pub rel_len: i32,
    pub rel_extent: i32,
    pub shortconv_len: i32,
    pub attn_res_block: i32,
    pub max_ctx: i32,
    pub tok_embd_type: i32,
    pub output_w_type: i32,
    pub log_scale_floor: i32,

    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub expert_weights_scale: f32,
    pub final_logit_softcap: f32,
    pub _pad_model: [i32; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct InkDSparkLayerDesc {
    pub dflare_norm: u64,
    pub input_layernorm: u64,
    pub post_attn_norm: u64,
    pub qkv_proj: u64,
    pub o_proj: u64,
    pub kv_proj_ctx: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub k_norm_ctx: u64,
    pub gate_up_proj: u64,
    pub down_proj: u64,
    pub is_local: i32,
    pub _pad: [i32; 9],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct InkDSparkDesc {
    pub layers: [InkDSparkLayerDesc; 3],
    pub fuse_alpha: [f32; 16],
    pub mask_embed_va: u64,
    pub final_norm_va: u64,
    pub markov_dflare_va: u64,
    pub markov_w1_bf16_va: u64,
    pub rnn_proj_w_va: u64,
    pub rnn_proj_b_va: u64,
    pub fold_w_va: u64,
    pub fold_b_va: u64,

    pub draft_k_va: u64,
    pub draft_v_va: u64,
    pub ws_f_rnn_va: u64,

    pub ws_h_va: u64,
    pub ws_qkv_va: u64,
    pub ws_mlp_va: u64,
    pub ws_xq_va: u64,
    pub ws_xds_va: u64,
    pub ws_base_logits_va: u64,
    pub ws_rnn_state_va: u64,
    pub ws_rnn_inp_va: u64,
    pub ws_rnn_proj_va: u64,

    pub max_ctx: i32,
    pub swa_window: i32,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct InkRunCtrl {
    pub start_token: i32,
    pub start_pos: i32,
    pub max_steps: i32,
    pub eff_passes: i32,
    pub temperature: f32,
    pub step_offset: u32,
    pub rng_seed: u64,

    pub cancel_flag: i32,
    pub error_code: i32,
    pub steps_completed: i32,
    pub stop_reason: i32,
    pub export_taps: i32,
    pub save_conv_backup: i32,
    pub batch_len: i32,
    pub logits_mask: u32,

    pub inject_pos: i32,
    pub inject_cnt: i32,
    pub _pad_ctrl: [i32; 2],

    pub batch_tokens: [i32; 8],
    pub batch_out_toks: [i32; 8],
    pub done_seq: u64,
    pub expected_seq: u64,
    pub tokens_out: [i32; 4096],
}

const _: () = assert!(std::mem::size_of::<InkPhysLayerDesc>() == 256);
const _: () = assert!(std::mem::size_of::<InkLoopCtrlDesc>() == 128);
const _: () = assert!(std::mem::size_of::<InkModelDesc>() == 576);
const _: () = assert!(std::mem::size_of::<InkDSparkLayerDesc>() == 128);
const _: () = assert!(std::mem::size_of::<InkDSparkDesc>() == 624);
const _: () = assert!(std::mem::size_of::<InkRunCtrl>() == 16576);

pub struct InkDSparkState {
    pub mtp_path: PathBuf,
    pub weights_buf: VramBuffer,
    pub draft_kv_buf: VramBuffer,
    pub ws_buf: VramBuffer,
    pub desc_buf: VramBuffer,
    pub desc_va: u64,
    pub total_vram_mb: f64,
}

#[derive(Clone, Copy, Debug)]
struct WeightTensorGpu {
    va: u64,
    ggml_type: i32,
}

#[derive(Clone, Debug)]
pub struct InklingConfig {
    pub model_name: String,
    pub n_phys_layers: usize,
    pub dim: usize,
    pub n_ff_exp: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    pub gate_rank: usize,
    pub vocab_padded: usize,
    pub vocab_unpadded: usize,
    pub swa_window: usize,
    pub rel_len: usize,
    pub rel_extent: usize,
    pub shortconv_len: usize,
    pub attn_res_block: usize,
    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: i32,
    pub log_scale_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub expert_weights_scale: f32,
    pub final_logit_softcap: f32,
    pub max_ctx: usize,
}

#[derive(Clone, Debug)]
pub struct InklingGenStats {
    pub model_name: String,
    pub loop_passes: usize,
    pub logical_layers_per_tok: usize,
    pub prompt_tokens: usize,
    pub cached_prefix_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub ttft_ms: f64,
    pub vram_weights_mb: f64,
    pub vram_kv_conv_mb: f64,
    pub spec_verify_calls: usize,
    pub spec_accepted_extra: usize,
    pub gpu_launches: usize,
    pub stop_reason: &'static str,
    pub token_ids: Vec<i32>,
}

fn ggml_tensor_nbytes(dims: &[usize], ggml_type: i32) -> Result<usize, String> {
    let nelems: usize = dims.iter().copied().product();
    match ggml_type {
        0 => Ok(nelems * 4),
        1 | 30 => Ok(nelems * 2),
        8 => Ok((nelems / 32) * 34),
        12 => Ok((nelems / 256) * 144),
        14 => Ok((nelems / 256) * 210),
        16 => Ok((nelems / 256) * 66),
        20 => Ok((nelems / 32) * 18),
        23 => Ok((nelems / 256) * 136),
        42 => Ok((nelems / 64) * 18),
        other => Err(format!("Unsupported GGML quant type {} for Inkling-2-Mini", other)),
    }
}

fn parse_safetensors_offsets(hdr_json: &str) -> Result<HashMap<String, (usize, usize)>, String> {
    let bytes = hdr_json.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= n || bytes[i] != b'{' {
        return Err("Invalid safetensors JSON header".into());
    }
    i += 1;
    let mut map = HashMap::new();
    while i < n {
        while i < n && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= n || bytes[i] == b'}' {
            break;
        }
        if bytes[i] != b'"' {
            return Err("Expected string key in safetensors header".into());
        }
        i += 1;
        let k_start = i;
        while i < n && bytes[i] != b'"' {
            if bytes[i] == b'\\' {
                i += 1;
            }
            i += 1;
        }
        let key = hdr_json[k_start..i].to_string();
        i += 1;
        while i < n && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
            i += 1;
        }
        if i >= n || bytes[i] != b'{' {
            return Err(format!("Expected object value for key '{}'", key));
        }
        let obj_start = i;
        let mut depth = 0i32;
        let mut in_str = false;
        while i < n {
            let b = bytes[i];
            if in_str {
                if b == b'\\' {
                    i += 1;
                } else if b == b'"' {
                    in_str = false;
                }
            } else if b == b'"' {
                in_str = true;
            } else if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            i += 1;
        }
        let obj_str = &hdr_json[obj_start..i];
        if key != "__metadata__" {
            if let Some(do_idx) = obj_str.find("\"data_offsets\"") {
                let sub = &obj_str[do_idx + 14..];
                if let (Some(lb), Some(rb)) = (sub.find('['), sub.find(']')) {
                    let inner = &sub[lb + 1..rb];
                    let mut parts = inner.split(',');
                    if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                        let s = a.trim().parse::<usize>().map_err(|e| e.to_string())?;
                        let e = b.trim().parse::<usize>().map_err(|e| e.to_string())?;
                        map.insert(key, (s, e));
                    }
                }
            }
        }
    }
    Ok(map)
}

fn bf16_bytes_to_f32_vec(raw: &[u8]) -> Vec<f32> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let u = get_u16(raw, i * 2);
        out.push(f32::from_bits((u as u32) << 16));
    }
    out
}

fn bf16_bytes_to_f32_bytes(raw: &[u8]) -> Vec<u8> {
    let vals = bf16_bytes_to_f32_vec(raw);
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Quantize one or more row-stacked BF16 matrices (`[n_rows_i, k_cols]`) into
/// Aligned Split `Q8_0` (`[qs: int8[tot_rows, k_cols] | d: half[tot_rows, k_cols/32]]`).
fn bf16_slices_to_split_q8_0(slices: &[(&[u8], usize)], k_cols: usize) -> Vec<u8> {
    assert_eq!(k_cols % 32, 0);
    let n_blks_per_row = k_cols / 32;
    let tot_rows: usize = slices.iter().map(|s| s.1).sum();
    let qs_bytes = tot_rows * k_cols;
    let d_bytes = tot_rows * n_blks_per_row * 2;
    assert_eq!(qs_bytes % 16, 0);
    let mut out = vec![0u8; qs_bytes + d_bytes];
    let (qs_part, d_part) = out.split_at_mut(qs_bytes);

    let mut row_cursor = 0usize;
    for &(raw, n_rows) in slices {
        assert_eq!(raw.len(), n_rows * k_cols * 2);
        for r in 0..n_rows {
            let g_row = row_cursor + r;
            let row_raw = &raw[r * k_cols * 2..(r + 1) * k_cols * 2];
            let row_qs = &mut qs_part[g_row * k_cols..(g_row + 1) * k_cols];
            let row_d = &mut d_part[g_row * n_blks_per_row * 2..(g_row + 1) * n_blks_per_row * 2];

            for b in 0..n_blks_per_row {
                let blk_raw = &row_raw[b * 64..(b + 1) * 64];
                let mut vals = [0.0f32; 32];
                let mut amax = 0.0f32;
                for l in 0..32 {
                    let u = u16::from_le_bytes([blk_raw[l * 2], blk_raw[l * 2 + 1]]);
                    let v = f32::from_bits((u as u32) << 16);
                    vals[l] = v;
                    let av = v.abs();
                    if av > amax {
                        amax = av;
                    }
                }
                let d_f16 = f32_to_f16_bits(amax * (1.0 / 127.0));
                row_d[b * 2..b * 2 + 2].copy_from_slice(&d_f16.to_le_bytes());
                let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let dst_q = &mut row_qs[b * 32..(b + 1) * 32];
                for l in 0..32 {
                    let q = (vals[l] * id).round().clamp(-127.0, 127.0) as i8;
                    dst_q[l] = q as u8;
                }
            }
        }
        row_cursor += n_rows;
    }
    out
}

fn discover_mtp_safetensors(gguf_path: &Path) -> Option<PathBuf> {
    if std::env::var("INKLING_NO_DSPARK")
        .ok()
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        return None;
    }
    if let Ok(explicit) = std::env::var("INKLING_MTP_PATH") {
        let p = PathBuf::from(explicit);
        if p.exists() {
            return Some(p);
        }
    }
    let mut candidates = Vec::new();
    if let Some(parent) = gguf_path.parent() {
        candidates.push(parent.join("Inkling-2-Mini-mtp.safetensors"));
        candidates.push(parent.join("mtp.safetensors"));
    }
    if let Ok(home) = std::env::var("HOME") {
        let cache_dir = PathBuf::from(home).join(".cache/qwen-nv-rs/models");
        candidates.push(cache_dir.join("Inkling-2-Mini-mtp.safetensors"));
        candidates.push(cache_dir.join("mtp.safetensors"));
    }
    candidates.into_iter().find(|p| p.exists())
}

impl InkDSparkState {
    pub fn load(
        gpu: &mut NvGpu,
        module: &CubinModule,
        mtp_path: &Path,
        cfg: &InklingConfig,
        output_w_va: u64,
        output_w_type: i32,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        let mmap = MmapFile::open(mtp_path)?;
        let raw_all = mmap.as_slice();
        if raw_all.len() < 16 {
            return Err("mtp.safetensors is too small".into());
        }
        let hdr_len = get_u64(raw_all, 0) as usize;
        if 8 + hdr_len > raw_all.len() {
            return Err("Invalid mtp.safetensors header length".into());
        }
        let hdr_str = std::str::from_utf8(&raw_all[8..8 + hdr_len])
            .map_err(|e| format!("Invalid UTF-8 in mtp.safetensors header: {}", e))?;
        let offsets = parse_safetensors_offsets(hdr_str)?;
        let data_base = 8 + hdr_len;

        let get_raw = |name: &str| -> Result<&[u8], String> {
            let &(s, e) = offsets
                .get(name)
                .ok_or_else(|| format!("Missing tensor '{}' in {}", name, mtp_path.display()))?;
            if data_base + e > raw_all.len() || s > e {
                return Err(format!("Out-of-bounds tensor '{}' in {}", name, mtp_path.display()));
            }
            Ok(&raw_all[data_base + s..data_base + e])
        };

        // Compute row-wise softmax `fuse_alpha [4, 4]` from `fuse_weight [4, 4]`
        let fw_vals = bf16_bytes_to_f32_vec(get_raw("fuse_weight")?);
        if fw_vals.len() != 16 {
            return Err(format!("Expected 16 elements in fuse_weight, got {}", fw_vals.len()));
        }
        let mut fuse_alpha = [0.0f32; 16];
        for r in 0..4 {
            let row = &fw_vals[r * 4..(r + 1) * 4];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..4 {
                let ev = (row[c] - mx).exp();
                fuse_alpha[r * 4 + c] = ev;
                sum += ev;
            }
            for c in 0..4 {
                fuse_alpha[r * 4 + c] /= sum;
            }
        }

        // Convert & pack all DSpark weights into one contiguous 256B-aligned VRAM arena
        let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(48);
        let mut rel_offs: Vec<u64> = Vec::with_capacity(48);
        let mut cur_w_off = 0u64;
        let mut push_blob = |b: Vec<u8>| -> u64 {
            let o = align_up(cur_w_off, 256);
            cur_w_off = o + (b.len() as u64);
            rel_offs.push(o);
            blobs.push(b);
            o
        };

        let o_mask_embed = push_blob(bf16_bytes_to_f32_bytes(get_raw("mask_embed")?));
        let o_final_norm = push_blob(bf16_bytes_to_f32_bytes(get_raw("norm.weight")?));
        let o_markov_dflare = push_blob(bf16_bytes_to_f32_bytes(get_raw("markov_head.dflare_norm.weight")?));

        let mw1_raw = get_raw("markov_head.markov_w1.weight")?;
        let mw1_crop_bytes = cfg.vocab_unpadded * 256 * 2;
        if mw1_raw.len() < mw1_crop_bytes {
            return Err("markov_w1.weight smaller than vocab_unpadded * 256".into());
        }
        let o_markov_w1 = push_blob(mw1_raw[..mw1_crop_bytes].to_vec());
        let o_rnn_proj_w = push_blob(bf16_slices_to_split_q8_0(
            &[(get_raw("markov_head.rnn_proj.weight")?, 768)],
            5120,
        ));
        let o_rnn_proj_b = push_blob(bf16_bytes_to_f32_bytes(get_raw("markov_head.rnn_proj.bias")?));

        let mut layer_rel = [(0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64); 3];
        for il in 0..3 {
            let p = format!("layers.{}", il);
            let dflare = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!("{}.dflare_norm.weight", p))?));
            let in_ln = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!("{}.input_layernorm.weight", p))?));
            let post_ln = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!(
                "{}.post_attention_layernorm.weight",
                p
            ))?));
            let qkv = push_blob(bf16_slices_to_split_q8_0(
                &[
                    (get_raw(&format!("{}.self_attn.q_proj.weight", p))?, 3072),
                    (get_raw(&format!("{}.self_attn.k_proj.weight", p))?, 1024),
                    (get_raw(&format!("{}.self_attn.v_proj.weight", p))?, 1024),
                ],
                2304,
            ));
            let o_p = push_blob(bf16_slices_to_split_q8_0(
                &[(get_raw(&format!("{}.self_attn.o_proj.weight", p))?, 2304)],
                3072,
            ));
            let kv_ctx = push_blob(bf16_slices_to_split_q8_0(
                &[
                    (get_raw(&format!("{}.self_attn.k_proj_ctx.weight", p))?, 1024),
                    (get_raw(&format!("{}.self_attn.v_proj_ctx.weight", p))?, 1024),
                ],
                2304,
            ));
            let q_n = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!("{}.self_attn.q_norm.weight", p))?));
            let k_n = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!("{}.self_attn.k_norm.weight", p))?));
            let k_n_ctx = push_blob(bf16_bytes_to_f32_bytes(get_raw(&format!(
                "{}.self_attn.k_norm_ctx.weight",
                p
            ))?));
            let gu = push_blob(bf16_slices_to_split_q8_0(
                &[
                    (get_raw(&format!("{}.mlp.gate_proj.weight", p))?, 9216),
                    (get_raw(&format!("{}.mlp.up_proj.weight", p))?, 9216),
                ],
                2304,
            ));
            let dn = push_blob(bf16_slices_to_split_q8_0(
                &[(get_raw(&format!("{}.mlp.down_proj.weight", p))?, 2304)],
                9216,
            ));
            layer_rel[il] = (dflare, in_ln, post_ln, qkv, o_p, kv_ctx, q_n, k_n, k_n_ctx, gu, dn);
        }

        // Reserve VRAM inside `weights_buf` for GPU-folded `_fold_weight [vocab, 256]` (Split Q8_0) + `_fold_bias [vocab]` (F32)
        let fold_w_bytes = cfg.vocab_unpadded * 256 + cfg.vocab_unpadded * 8 * 2;
        let o_fold_w = align_up(cur_w_off, 256);
        cur_w_off = o_fold_w + (fold_w_bytes as u64);
        let fold_b_bytes = cfg.vocab_unpadded * 4;
        let o_fold_b = align_up(cur_w_off, 256);
        cur_w_off = o_fold_b + (fold_b_bytes as u64);

        let weights_buf = gpu.alloc_vram(cur_w_off)?;
        let wb = weights_buf.va_addr;
        const CHUNK: usize = 32 * 1024 * 1024;
        for (blob, &off) in blobs.iter().zip(rel_offs.iter()) {
            let mut p = 0usize;
            while p < blob.len() {
                let take = (blob.len() - p).min(CHUNK);
                gpu.upload_to_vram(&weights_buf, off + (p as u64), &blob[p..p + take])?;
                p += take;
            }
        }
        drop(blobs);

        // Allocate Draft Context KV Cache (`[2 (K,V), 3, max_ctx, 1024]` FP16 = 48 MiB @ ctx=4096)
        let per_kv_bytes = (3 * cfg.max_ctx * 1024 * 2) as u64;
        let draft_kv_buf = gpu.alloc_vram(2 * per_kv_bytes)?;
        let draft_k_va = draft_kv_buf.va_addr;
        let draft_v_va = draft_kv_buf.va_addr + per_kv_bytes;

        // Allocate DSpark Workspace (`~4.8 MiB` total, also hosts temporary `hidden_out` during 8ms startup fold!)
        let mut ws_off = 0u64;
        let mut alloc_dws = |nb: usize| -> u64 {
            let o = align_up(ws_off, 256);
            ws_off = o + (nb as u64);
            o
        };
        let d_f_rnn = alloc_dws(2304 * 4);
        let d_h = alloc_dws(8 * 2304 * 4);
        let d_qkv = alloc_dws(24 * 2048 * 4);
        let d_mlp = alloc_dws(8 * 9216 * 4);
        let d_xq = alloc_dws(8 * 9216);
        let d_xds = alloc_dws(8 * 288 * 8);
        let d_base_logits = alloc_dws(7 * cfg.vocab_unpadded * 2);
        let d_rnn_state = alloc_dws(256 * 4);
        let d_rnn_inp = alloc_dws(5120 * 4);
        let d_rnn_proj = alloc_dws(768 * 4);
        let ws_buf = gpu.alloc_vram(ws_off.max(2 * 1024 * 1024))?;
        let dwb = ws_buf.va_addr;

        // Pre-fold `_fold_weight` & `_fold_bias` on RTX 3080 using temporary space in `ws_buf` (`0` extra VRAM!)
        let hout_w_raw = get_raw("markov_head.hidden_out.weight")?; // [2304, 256] BF16 = 1,179,648 B
        let hout_b_f32 = bf16_bytes_to_f32_bytes(get_raw("markov_head.hidden_out.bias")?); // [2304] F32 = 9,216 B
        gpu.upload_to_vram(&ws_buf, 0, hout_w_raw)?;
        let hb_off = align_up(hout_w_raw.len() as u64, 256);
        gpu.upload_to_vram(&ws_buf, hb_off, &hout_b_f32)?;

        let mut fold_args = [0u8; 48];
        put_u64(&mut fold_args, 0, wb + o_fold_w);
        put_u64(&mut fold_args, 8, wb + o_fold_b);
        put_u64(&mut fold_args, 16, output_w_va);
        fold_args[24..28].copy_from_slice(&output_w_type.to_le_bytes());
        fold_args[28..32].copy_from_slice(&(cfg.vocab_unpadded as i32).to_le_bytes());
        put_u64(&mut fold_args, 32, dwb);
        put_u64(&mut fold_args, 40, dwb + hb_off);

        gpu.timeline_value += 1;
        let sig_val = gpu.timeline_value;
        let k_fold = &module.kernels["ink_dspark_build_fold_kernel"];
        let (qmd_va, _) = gpu.prepare_qmd(
            k_fold,
            (gpu.active_sms, 1, 1),
            (256, 1, 1),
            &fold_args,
            Some((gpu.timeline_buf.va_addr, sig_val)),
        );
        gpu.launch_qmd_chain(qmd_va, false)?;
        gpu.wait_timeline(sig_val, Duration::from_secs(15))?;

        let mut l_descs = [InkDSparkLayerDesc::default(); 3];
        for il in 0..3 {
            let (dflare, in_ln, post_ln, qkv, o_p, kv_ctx, q_n, k_n, k_n_ctx, gu, dn) = layer_rel[il];
            l_descs[il] = InkDSparkLayerDesc {
                dflare_norm: wb + dflare,
                input_layernorm: wb + in_ln,
                post_attn_norm: wb + post_ln,
                qkv_proj: wb + qkv,
                o_proj: wb + o_p,
                kv_proj_ctx: wb + kv_ctx,
                q_norm: wb + q_n,
                k_norm: wb + k_n,
                k_norm_ctx: wb + k_n_ctx,
                gate_up_proj: wb + gu,
                down_proj: wb + dn,
                is_local: if il == 0 || il == 2 { 1 } else { 0 },
                _pad: [0; 9],
            };
        }

        let d_desc = InkDSparkDesc {
            layers: l_descs,
            fuse_alpha,
            mask_embed_va: wb + o_mask_embed,
            final_norm_va: wb + o_final_norm,
            markov_dflare_va: wb + o_markov_dflare,
            markov_w1_bf16_va: wb + o_markov_w1,
            rnn_proj_w_va: wb + o_rnn_proj_w,
            rnn_proj_b_va: wb + o_rnn_proj_b,
            fold_w_va: wb + o_fold_w,
            fold_b_va: wb + o_fold_b,
            draft_k_va,
            draft_v_va,
            ws_f_rnn_va: dwb + d_f_rnn,
            ws_h_va: dwb + d_h,
            ws_qkv_va: dwb + d_qkv,
            ws_mlp_va: dwb + d_mlp,
            ws_xq_va: dwb + d_xq,
            ws_xds_va: dwb + d_xds,
            ws_base_logits_va: dwb + d_base_logits,
            ws_rnn_state_va: dwb + d_rnn_state,
            ws_rnn_inp_va: dwb + d_rnn_inp,
            ws_rnn_proj_va: dwb + d_rnn_proj,
            max_ctx: cfg.max_ctx as i32,
            swa_window: 1024,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
        };

        let desc_buf = gpu.alloc_vram(1024)?;
        let mut d_bytes = vec![0u8; std::mem::size_of::<InkDSparkDesc>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                &d_desc as *const InkDSparkDesc as *const u8,
                d_bytes.as_mut_ptr(),
                d_bytes.len(),
            );
        }
        gpu.upload_to_vram(&desc_buf, 0, &d_bytes)?;

        let total_vram_mb =
            ((weights_buf.size + draft_kv_buf.size + ws_buf.size + desc_buf.size) as f64) / (1024.0 * 1024.0);
        eprintln!(
            "[dspark-nv] Loaded official Inkling2MTP block-8 drafter ({:.1} MiB VRAM: Split-Q8_0 3L + BF16 Markov-256 + GPU-folded LM head) in {:.2}s",
            total_vram_mb,
            t0.elapsed().as_secs_f64()
        );

        let desc_va = desc_buf.va_addr;
        Ok(Self {
            mtp_path: mtp_path.to_path_buf(),
            weights_buf,
            draft_kv_buf,
            ws_buf,
            desc_buf,
            desc_va,
            total_vram_mb,
        })
    }
}

pub struct InklingEngine {
    pub gpu: NvGpu,
    pub module: CubinModule,
    pub cfg: InklingConfig,
    pub tok: BpeTokenizer,
    pub model_ref: String,
    weights_buf: VramBuffer,
    swa_k_buf: VramBuffer,
    swa_v_buf: VramBuffer,
    full_k_buf: VramBuffer,
    full_v_buf: VramBuffer,
    conv_state_buf: VramBuffer,
    conv_backup_buf: VramBuffer,
    ws_buf: VramBuffer,
    ws_bar_state_buf: VramBuffer,
    descs_vram_buf: VramBuffer,
    model_desc_va: u64,
    ctrl_host: HostMappedBuffer,
    done_seq_counter: u64,
    conv_total_floats: usize,
    pub dspark_taps_va: u64,
    pub dspark: Option<InkDSparkState>,
    pub dspark_gamma: usize,
    dspark_cache_valid: bool,
    pub cached_tokens: Vec<i32>,
    cached_passes: usize,
}

impl InklingEngine {
    pub fn is_inkling_gguf(path: &Path) -> bool {
        if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
            if fname.to_ascii_lowercase().contains("inkling") {
                return true;
            }
        }
        if let Ok(gf) = GgufFile::open(path) {
            if let Some(arch) = gf.kv.get("general.architecture").and_then(|v| v.as_str()) {
                return arch.starts_with("inkling2");
            }
        }
        false
    }

    pub fn is_inkling_ref(model_ref: &str) -> bool {
        let lower = model_ref.trim().to_ascii_lowercase();
        if lower.contains("inkling") {
            return true;
        }
        if let Ok(p) = resolve_model_path(model_ref) {
            return Self::is_inkling_gguf(&p);
        }
        false
    }

    pub fn new(model_ref: &str, max_context_req: usize) -> Result<Self, String> {
        let model_path = resolve_model_path(model_ref)?;
        let t_init = Instant::now();
        let mut gpu = NvGpu::init()?;
        eprintln!(
            "[inkling-nv] Initialized GA102 (sm_86, {} active SMs, {:.2} GB GDDR6X) in {:.2}s",
            gpu.active_sms,
            (gpu.mm.vram_size as f64) / 1e9,
            t_init.elapsed().as_secs_f64()
        );

        let t_load = Instant::now();
        let gguf = GgufFile::open(&model_path)?;
        let arch = gguf
            .kv
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != "inkling2_loop" {
            return Err(format!("Expected general.architecture='inkling2_loop', got '{}'", arch));
        }

        let tok = BpeTokenizer::from_gguf(&gguf)?;
        let get_u = |k: &str| -> Result<usize, String> {
            gguf.kv
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("Missing GGUF key {}", k))
        };
        let get_f = |k: &str, def: f32| -> f32 {
            gguf.kv.get(k).and_then(|v| v.as_f32()).unwrap_or(def)
        };

        let n_phys = get_u("inkling2_loop.block_count")?;
        let dim = get_u("inkling2_loop.embedding_length")?;
        let n_ff_exp = get_u("inkling2_loop.expert_feed_forward_length")?;
        let n_experts = get_u("inkling2_loop.expert_count")?;
        let n_experts_used = get_u("inkling2_loop.expert_used_count")?;
        let head_dim = get_u("inkling2_loop.attention.key_length")?;
        let loop_start = get_u("inkling2_loop.loop.start")?;
        let loop_end = get_u("inkling2_loop.loop.end")?;
        let loop_passes = get_u("inkling2_loop.loop.count")?;
        let gate_rank = get_u("inkling2_loop.loop.gate_rank")?;
        let vocab_padded = get_u("inkling2_loop.vocab_size")?;
        let vocab_unpadded = get_u("inkling2_loop.vocab_size_unpadded")?;
        let swa_window = get_u("inkling2_loop.attention.sliding_window")?;
        let rel_len = get_u("inkling2_loop.attention.relative_length")?;
        let rel_extent = get_u("inkling2_loop.attention.relative_extent")?;
        let shortconv_len = get_u("inkling2_loop.shortconv.l_cache")?;
        let attn_res_block = get_u("inkling2_loop.attn_res.block_size")?;

        let n_heads = match gguf.kv.get("inkling2_loop.attention.head_count") {
            Some(GgufVal::IntArr(v)) if !v.is_empty() => v[0] as usize,
            _ => 24,
        };
        let n_kv_heads = match gguf.kv.get("inkling2_loop.attention.head_count_kv") {
            Some(GgufVal::IntArr(v)) if !v.is_empty() => v[0] as usize,
            _ => 8,
        };
        let swa_pattern: Vec<bool> = match gguf.kv.get("inkling2_loop.attention.sliding_window_pattern") {
            Some(GgufVal::IntArr(v)) if v.len() == n_phys => v.iter().map(|&x| x != 0).collect(),
            _ => (0..n_phys).map(|i| (i % 6) != 5).collect(),
        };

        let cfg = InklingConfig {
            model_name: gguf
                .kv
                .get("general.name")
                .and_then(|v| v.as_str())
                .unwrap_or("Inkling-2-Mini")
                .to_string(),
            n_phys_layers: n_phys,
            dim,
            n_ff_exp,
            n_heads,
            n_kv_heads,
            head_dim,
            n_experts,
            n_experts_used,
            loop_start,
            loop_end,
            loop_passes,
            gate_rank,
            vocab_padded,
            vocab_unpadded,
            swa_window,
            rel_len,
            rel_extent,
            shortconv_len,
            attn_res_block,
            rms_eps: get_f("inkling2_loop.attention.layer_norm_rms_epsilon", 1e-6),
            qk_clip: get_f("inkling2_loop.attention.qk_clip", 22.627417),
            rel_clip: get_f("inkling2_loop.attention.relative_clip", 8.0),
            log_scale_floor: get_u("inkling2_loop.attention.log_scale_floor").unwrap_or(8192) as i32,
            log_scale_alpha: get_f("inkling2_loop.attention.log_scale_alpha", 0.1),
            situ_beta: get_f("inkling2_loop.activation.situ_beta", 4.0),
            situ_lb: get_f("inkling2_loop.activation.situ_linear_beta", 25.0),
            expert_weights_scale: get_f("inkling2_loop.expert_weights_scale", 8.0),
            final_logit_softcap: get_f("inkling2_loop.final_logit_softcapping", 30.0),
            max_ctx: max_context_req.clamp(512, 8192),
        };

        let module = gpu.load_cubin(INKLING_CUBIN)?;

        // Compute deterministic 256-byte-aligned VRAM layout for all GGUF tensors
        let mut names_sorted: Vec<String> = gguf.tensors.keys().cloned().collect();
        names_sorted.sort_by_key(|n| gguf.tensors[n].offset);

        let mut total_w_bytes: u64 = 0;
        let mut tensor_rel_off =
            std::collections::HashMap::<String, (u64, usize, i32)>::with_capacity(names_sorted.len());
        for name in &names_sorted {
            let info = &gguf.tensors[name];
            let nb = if name == "output.weight" && info.dims.len() == 2 && info.dims[1] >= vocab_unpadded {
                ggml_tensor_nbytes(&[info.dims[0], vocab_unpadded], info.ggml_type)?
            } else {
                ggml_tensor_nbytes(&info.dims, info.ggml_type)?
            };
            let off = align_up(total_w_bytes, 256);
            tensor_rel_off.insert(name.clone(), (off, nb, info.ggml_type));
            total_w_bytes = off + (nb as u64);
        }

        let weights_buf = gpu.alloc_vram(total_w_bytes)?;
        let w_base_va = weights_buf.va_addr;

        // Stream from mmap -> GDDR6X in <= 32 MiB DMA slices, losslessly splitting
        // quantized projection matrices into `[16B-aligned qs | contiguous fp16 d]` (`0` extra VRAM!)
        const CHUNK_MAX: usize = 32 * 1024 * 1024;
        let mut stage = vec![0u8; CHUNK_MAX];
        let t_dma = Instant::now();
        for name in &names_sorted {
            let info = &gguf.tensors[name];
            let (dst_off, nb, tt) = tensor_rel_off[name];
            let raw = gguf.tensor_bytes(info, nb);

            let split_spec: Option<(usize, usize, usize, usize)> = if name == "token_embd.weight" {
                None
            } else {
                match tt {
                    8 => Some((34, 32, 2, 0)),      // Q8_0: [2B d, 32B qs]
                    14 => Some((210, 208, 0, 208)), // Q6_K: [208B ql+qh+sc, 2B d]
                    16 => Some((66, 64, 2, 0)),     // IQ2_XXS: [2B d, 64B qs]
                    20 => Some((18, 16, 2, 0)),     // IQ4_NL: [2B d, 16B qs]
                    42 => Some((18, 16, 2, 0)),     // Q2_0: [2B d, 16B qs]
                    _ => None,
                }
            };

            if let Some((gguf_blk, pay_len, pay_off, d_off)) = split_spec {
                let tot_blks = nb / gguf_blk;
                assert_eq!(tot_blks * gguf_blk, nb);
                assert_eq!((tot_blks * pay_len) & 15, 0);

                let max_pay_blks = CHUNK_MAX / pay_len;
                let mut b_cur = 0usize;
                while b_cur < tot_blks {
                    let take = (tot_blks - b_cur).min(max_pay_blks);
                    if tt == 42 {
                        // Lossless planarization of `Q2_0` (`16`B = `4 x u32`) so `P >> {0,2,4,6} & 0x03030303`
                        // feeds `__dp4a` directly in registers with ZERO SMEM LUT gathers!
                        for bi in 0..take {
                            let s_off = (b_cur + bi) * gguf_blk + pay_off;
                            let d_pos = bi * 16;
                            for w in 0..4 {
                                let v = u32::from_le_bytes([
                                    raw[s_off + w * 4],
                                    raw[s_off + w * 4 + 1],
                                    raw[s_off + w * 4 + 2],
                                    raw[s_off + w * 4 + 3],
                                ]);
                                let b0 = v & 0xff;
                                let b1 = (v >> 8) & 0xff;
                                let b2 = (v >> 16) & 0xff;
                                let b3 = v >> 24;
                                let p0 = (b0 & 3) | ((b1 & 3) << 2) | ((b2 & 3) << 4) | ((b3 & 3) << 6);
                                let p1 = ((b0 >> 2) & 3) | (b1 & 0x0c) | (((b2 >> 2) & 3) << 4) | (((b3 >> 2) & 3) << 6);
                                let p2 = ((b0 >> 4) & 3) | (((b1 >> 4) & 3) << 2) | (b2 & 0x30) | (((b3 >> 4) & 3) << 6);
                                let p3 = (b0 >> 6) | ((b1 >> 6) << 2) | ((b2 >> 6) << 4) | (b3 & 0xc0);
                                let pv = p0 | (p1 << 8) | (p2 << 16) | (p3 << 24);
                                stage[d_pos + w * 4..d_pos + w * 4 + 4].copy_from_slice(&pv.to_le_bytes());
                            }
                        }
                    } else {
                        for bi in 0..take {
                            let s_off = (b_cur + bi) * gguf_blk + pay_off;
                            stage[bi * pay_len..(bi + 1) * pay_len]
                                .copy_from_slice(&raw[s_off..s_off + pay_len]);
                        }
                    }
                    gpu.upload_to_vram(
                        &weights_buf,
                        dst_off + ((b_cur * pay_len) as u64),
                        &stage[..take * pay_len],
                    )?;
                    b_cur += take;
                }

                let d_base_off = dst_off + ((tot_blks * pay_len) as u64);
                let max_d_blks = CHUNK_MAX / 2;
                b_cur = 0;
                while b_cur < tot_blks {
                    let take = (tot_blks - b_cur).min(max_d_blks);
                    for bi in 0..take {
                        let s_off = (b_cur + bi) * gguf_blk + d_off;
                        stage[bi * 2..bi * 2 + 2].copy_from_slice(&raw[s_off..s_off + 2]);
                    }
                    gpu.upload_to_vram(
                        &weights_buf,
                        d_base_off + ((b_cur * 2) as u64),
                        &stage[..take * 2],
                    )?;
                    b_cur += take;
                }
            } else {
                let mut pos = 0usize;
                while pos < nb {
                    let take = (nb - pos).min(CHUNK_MAX);
                    gpu.upload_to_vram(&weights_buf, dst_off + (pos as u64), &raw[pos..pos + take])?;
                    pos += take;
                }
            }
        }
        drop(stage);
        let dma_gbps = (total_w_bytes as f64 / 1e9) / t_dma.elapsed().as_secs_f64().max(1e-6);

        let get_wt = |name: &str| -> Result<WeightTensorGpu, String> {
            let &(off, _, tt) = tensor_rel_off
                .get(name)
                .ok_or_else(|| format!("Missing Inkling tensor '{}'", name))?;
            Ok(WeightTensorGpu {
                va: w_base_va + off,
                ggml_type: tt,
            })
        };
        let get_f32_va = |name: &str| -> Result<u64, String> {
            let wt = get_wt(name)?;
            if wt.ggml_type != 0 {
                return Err(format!("Expected F32 (0) for '{}', got {}", name, wt.ggml_type));
            }
            Ok(wt.va)
        };

        let tok_embd = get_wt("token_embd.weight")?;
        let tok_norm = get_f32_va("token_embd_norm.weight")?;
        let output_w = get_wt("output.weight")?;
        let output_norm = get_f32_va("output_norm.weight")?;
        let output_res_query = get_f32_va("output_res_query.weight")?;

        let mut layer_descs = Vec::with_capacity(n_phys);
        for i in 0..n_phys {
            let is_swa = swa_pattern[i];
            let (attn_norm, q_in_n, k_in_n, v_in_n, k_cp, v_cp) = if is_swa {
                (get_f32_va(&format!("blk.{}.attn_norm.weight", i))?, 0, 0, 0, 0, 0)
            } else {
                (
                    0,
                    get_f32_va(&format!("blk.{}.attn_q_inp_norm.weight", i))?,
                    get_f32_va(&format!("blk.{}.attn_k_inp_norm.weight", i))?,
                    get_f32_va(&format!("blk.{}.attn_v_inp_norm.weight", i))?,
                    get_wt(&format!("blk.{}.attn_k_conv_proj.weight", i))?.va,
                    get_wt(&format!("blk.{}.attn_v_conv_proj.weight", i))?.va,
                )
            };
            let attn_res_query = if i > 0 {
                get_f32_va(&format!("blk.{}.attn_res_query.weight", i))?
            } else {
                0
            };
            let wq = get_wt(&format!("blk.{}.attn_q.weight", i))?;
            let wk = get_wt(&format!("blk.{}.attn_k.weight", i))?;
            let wv = get_wt(&format!("blk.{}.attn_v.weight", i))?;
            let wo = get_wt(&format!("blk.{}.attn_output.weight", i))?;
            let wqkv_gate = get_wt(&format!("blk.{}.attn_gate.weight", i))?;
            let attn_rel = get_wt(&format!("blk.{}.attn_rel.weight", i))?;
            let attn_sink_proj = get_wt(&format!("blk.{}.attn_sink_proj.weight", i))?;
            let gu = get_wt(&format!("blk.{}.ffn_gate_up_exps.weight", i))?;
            let dn = get_wt(&format!("blk.{}.ffn_down_exps.weight", i))?;

            layer_descs.push(InkPhysLayerDesc {
                attn_norm,
                attn_q_inp_norm: q_in_n,
                attn_k_inp_norm: k_in_n,
                attn_v_inp_norm: v_in_n,
                attn_res_query,
                wq: wq.va,
                wk: wk.va,
                wv: wv.va,
                wo: wo.va,
                wqkv_gate: wqkv_gate.va,
                attn_rel: attn_rel.va,
                attn_rel_bias: get_f32_va(&format!("blk.{}.attn_rel_bias.weight", i))?,
                attn_sink_proj: attn_sink_proj.va,
                attn_q_clip: get_f32_va(&format!("blk.{}.attn_q_clip.weight", i))?,
                attn_rel_clip: get_f32_va(&format!("blk.{}.attn_rel_clip.weight", i))?,
                attn_k_conv: get_f32_va(&format!("blk.{}.attn_k_conv.weight", i))?,
                attn_v_conv: get_f32_va(&format!("blk.{}.attn_v_conv.weight", i))?,
                attn_k_conv_proj: k_cp,
                attn_v_conv_proj: v_cp,
                attn_post_conv: get_f32_va(&format!("blk.{}.attn_post_conv.weight", i))?,
                ffn_res_query: get_f32_va(&format!("blk.{}.ffn_res_query.weight", i))?,
                ffn_norm: get_f32_va(&format!("blk.{}.ffn_norm.weight", i))?,
                ffn_gate_inp: get_f32_va(&format!("blk.{}.ffn_gate_inp.weight", i))?,
                exp_probs_b: get_f32_va(&format!("blk.{}.exp_probs_b.bias", i))?,
                ffn_gate_up_exps: gu.va,
                ffn_down_exps: dn.va,
                ffn_post_conv: get_f32_va(&format!("blk.{}.ffn_post_conv.weight", i))?,
                is_swa: if is_swa { 1 } else { 0 },
                qkv_type: wq.ggml_type,
                wo_type: wo.ggml_type,
                gate_up_type: gu.ggml_type,
                down_type: dn.ggml_type,
                _pad: [0; 5],
            });
        }

        let mut loop_ctrl_descs = Vec::with_capacity(loop_end - loop_start);
        let ctrl_names = ["q", "k", "v", "ffn"];
        for il in loop_start..loop_end {
            let msg_query = get_f32_va(&format!("blk.{}.loop_msg_query.weight", il))?;
            let mut down = [0u64; 4];
            let mut gain = [0u64; 4];
            let mut forget = [0u64; 4];
            for (c, tag) in ctrl_names.iter().enumerate() {
                down[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_down.weight", il, tag))?.va;
                gain[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_gain.weight", il, tag))?.va;
                forget[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_forget.weight", il, tag))?.va;
            }
            loop_ctrl_descs.push(InkLoopCtrlDesc {
                msg_query,
                ctrl_down: down,
                ctrl_gain: gain,
                ctrl_forget: forget,
                _pad: [0; 3],
            });
        }

        // Map all logical slots (`n_phys + (loop_passes - 1)*(loop_end - loop_start)` = 48)
        let n_slots = n_phys + (loop_passes.saturating_sub(1)) * (loop_end - loop_start);
        let mut slot_kv_dense = Vec::<i32>::with_capacity(n_slots);
        let mut n_swa_slots = 0usize;
        let mut n_full_slots = 0usize;
        for slot in 0..n_slots {
            let phys = if slot < n_phys {
                slot
            } else {
                loop_start + (slot - n_phys) % (loop_end - loop_start)
            };
            if layer_descs[phys].is_swa != 0 {
                slot_kv_dense.push(n_swa_slots as i32);
                n_swa_slots += 1;
            } else {
                slot_kv_dense.push(n_full_slots as i32);
                n_full_slots += 1;
            }
        }

        let q_dim = n_heads * head_dim; // 3072
        let kv_dim = n_kv_heads * head_dim; // 1024
        let r_dim = n_heads * rel_len; // 384

        let swa_slot_bytes = (2 * cfg.swa_window * kv_dim * 2) as u64;
        let full_slot_bytes = (cfg.max_ctx * kv_dim * 2) as u64;
        let swa_k_buf = gpu.alloc_vram((n_swa_slots as u64).max(1) * swa_slot_bytes)?;
        let swa_v_buf = gpu.alloc_vram((n_swa_slots as u64).max(1) * swa_slot_bytes)?;
        let full_k_buf = gpu.alloc_vram((n_full_slots as u64).max(1) * full_slot_bytes)?;
        let full_v_buf = gpu.alloc_vram((n_full_slots as u64).max(1) * full_slot_bytes)?;

        let conv_channels_per_slot = 2 * kv_dim + 2 * dim; // 6656
        let conv_total_floats = n_slots * conv_channels_per_slot * (shortconv_len - 1);
        let conv_total_bytes = (conv_total_floats * 4) as u64;
        let conv_state_buf = gpu.alloc_vram(conv_total_bytes)?;
        gpu.zero_vram(&conv_state_buf)?;

        let conv_backup_buf = gpu.alloc_vram(8 * conv_total_bytes)?;

        let ws_bar_state_buf = gpu.alloc_vram(256)?;
        gpu.zero_vram(&ws_bar_state_buf)?;

        // Allocate BM<=8 + Persistent Decode Workspace in GDDR6X
        let mut w_off = 0u64;
        let mut alloc_ws = |n_bytes: usize| -> u64 {
            let o = align_up(w_off, 256);
            w_off = o + (n_bytes as u64);
            o
        };
        let o_cur = alloc_ws(8 * dim * 4);
        let o_inp_sa = alloc_ws(8 * dim * 4);
        let o_ffn_inp = alloc_ws(8 * dim * 4);
        let o_prelude = alloc_ws(8 * dim * 4);
        let o_bank_out = alloc_ws(8 * dim * 4);
        let o_blocks = alloc_ws(8 * 8 * dim * 4);
        let o_prelude_blocks = alloc_ws(8 * 8 * dim * 4);
        let o_bank = alloc_ws(8 * 8 * dim * 4);
        let o_messages = alloc_ws(8 * 4 * dim * 4);
        let o_mixed = alloc_ws(8 * 3 * dim * 4);
        let o_inputs = alloc_ws(8 * 3 * dim * 4);
        let o_norm_in = alloc_ws(8 * 3 * dim * 4);
        let o_ctrl_cat = alloc_ws(8 * 3 * (2 * dim) * 4);
        let o_ctrl_gate = alloc_ws(8 * 3 * gate_rank * 4);
        let o_ctrl_gain = alloc_ws(8 * 3 * dim * 4);
        let o_ctrl_forget = alloc_ws(8 * 3 * dim * 4);

        let o_xq = alloc_ws(3 * 8 * 4608);
        let o_xds = alloc_ws(3 * 8 * 144 * 8);
        let o_xq_moe = alloc_ws(8 * 8 * n_ff_exp);
        let o_xds_moe = alloc_ws(8 * 8 * (n_ff_exp / 32) * 8);

        let o_q = alloc_ws(8 * q_dim * 4);
        let o_k_raw = alloc_ws(8 * kv_dim * 4);
        let o_v_raw = alloc_ws(8 * kv_dim * 4);
        let o_r = alloc_ws(8 * r_dim * 4);
        let o_sinks = alloc_ws(8 * n_heads * 4);
        let o_gate_raw = alloc_ws(8 * q_dim * 4);
        let o_dyn_k = alloc_ws(8 * 32 * 4);
        let o_dyn_v = alloc_ws(8 * 32 * 4);
        let o_attn_gated = alloc_ws(8 * q_dim * 4);

        let o_router_logits = alloc_ws(8 * n_experts * 4);
        let o_top_ids = alloc_ws(8 * n_experts_used * 4);
        let o_top_weights = alloc_ws(8 * n_experts_used * 4);
        let o_uniq_exp_id = alloc_ws(128 * 4);
        let o_uniq_mask = alloc_ws(128 * 4);
        let o_uniq_slots = alloc_ws(128 * 8 * 4);
        let o_n_uniq = alloc_ws(16 * 4);
        let o_situ = alloc_ws(8 * 8 * n_ff_exp * 4);

        let o_cta_val = alloc_ws(8 * 128 * 4);
        let o_cta_idx = alloc_ws(8 * 128 * 4);
        let o_dspark_taps = alloc_ws(8 * 4 * dim * 4);
        let o_logits = alloc_ws(vocab_unpadded * 4);

        let ws_buf = gpu.alloc_vram(w_off)?;
        let wb = ws_buf.va_addr;

        // Pack `InkPhysLayerDesc[24]`, `InkLoopCtrlDesc[12]`, `slot_kv_dense[48]`, and `InkModelDesc` into VRAM
        let layers_bytes_len = layer_descs.len() * std::mem::size_of::<InkPhysLayerDesc>();
        let loops_off = align_up(layers_bytes_len as u64, 256) as usize;
        let loops_bytes_len = loop_ctrl_descs.len() * std::mem::size_of::<InkLoopCtrlDesc>();
        let slots_off = align_up((loops_off + loops_bytes_len) as u64, 256) as usize;
        let slots_bytes_len = slot_kv_dense.len() * 4;
        let model_off = align_up((slots_off + slots_bytes_len) as u64, 256) as usize;
        let total_desc_bytes = align_up((model_off + std::mem::size_of::<InkModelDesc>()) as u64, 256);

        let descs_vram_buf = gpu.alloc_vram(total_desc_bytes)?;
        let db = descs_vram_buf.va_addr;
        let model_desc_va = db + (model_off as u64);

        let model_desc = InkModelDesc {
            layers_ptr: db,
            loop_ctrls_ptr: db + (loops_off as u64),
            slot_kv_dense_ptr: db + (slots_off as u64),
            tok_embd_va: tok_embd.va,
            tok_norm_va: tok_norm,
            output_w_va: output_w.va,
            output_norm_va: output_norm,
            output_res_query_va: output_res_query,

            swa_k_va: swa_k_buf.va_addr,
            swa_v_va: swa_v_buf.va_addr,
            full_k_va: full_k_buf.va_addr,
            full_v_va: full_v_buf.va_addr,
            conv_state_va: conv_state_buf.va_addr,
            conv_backup_va: conv_backup_buf.va_addr,

            ws_cur: wb + o_cur,
            ws_inp_sa: wb + o_inp_sa,
            ws_ffn_inp: wb + o_ffn_inp,
            ws_prelude: wb + o_prelude,
            ws_bank_out: wb + o_bank_out,
            ws_blocks: wb + o_blocks,
            ws_prelude_blocks: wb + o_prelude_blocks,
            ws_bank: wb + o_bank,
            ws_messages: wb + o_messages,
            ws_mixed: wb + o_mixed,
            ws_inputs: wb + o_inputs,
            ws_norm_in: wb + o_norm_in,
            ws_ctrl_cat: wb + o_ctrl_cat,
            ws_ctrl_gate: wb + o_ctrl_gate,
            ws_ctrl_gain: wb + o_ctrl_gain,
            ws_ctrl_forget: wb + o_ctrl_forget,

            ws_xq: wb + o_xq,
            ws_xds: wb + o_xds,
            ws_xq_moe: wb + o_xq_moe,
            ws_xds_moe: wb + o_xds_moe,

            ws_q: wb + o_q,
            ws_k_raw: wb + o_k_raw,
            ws_v_raw: wb + o_v_raw,
            ws_r: wb + o_r,
            ws_sinks: wb + o_sinks,
            ws_gate_raw: wb + o_gate_raw,
            ws_dyn_k: wb + o_dyn_k,
            ws_dyn_v: wb + o_dyn_v,
            ws_attn_gated: wb + o_attn_gated,

            ws_router_logits: wb + o_router_logits,
            ws_top_ids: wb + o_top_ids,
            ws_top_weights: wb + o_top_weights,
            ws_uniq_exp_id: wb + o_uniq_exp_id,
            ws_uniq_mask: wb + o_uniq_mask,
            ws_uniq_slots: wb + o_uniq_slots,
            ws_n_uniq: wb + o_n_uniq,
            ws_situ: wb + o_situ,

            ws_cta_val: wb + o_cta_val,
            ws_cta_idx: wb + o_cta_idx,
            ws_bar_state: ws_bar_state_buf.va_addr,
            ws_dspark_taps: wb + o_dspark_taps,
            ws_logits: wb + o_logits,

            dim: dim as i32,
            n_ff_exp: n_ff_exp as i32,
            n_heads: n_heads as i32,
            n_kv_heads: n_kv_heads as i32,
            head_dim: head_dim as i32,
            n_experts: n_experts as i32,
            n_experts_used: n_experts_used as i32,
            n_phys_layers: n_phys as i32,
            loop_start: loop_start as i32,
            loop_end: loop_end as i32,
            loop_passes: loop_passes as i32,
            gate_rank: gate_rank as i32,
            vocab_unpadded: vocab_unpadded as i32,
            swa_window: swa_window as i32,
            rel_len: rel_len as i32,
            rel_extent: rel_extent as i32,
            shortconv_len: shortconv_len as i32,
            attn_res_block: attn_res_block as i32,
            max_ctx: cfg.max_ctx as i32,
            tok_embd_type: tok_embd.ggml_type,
            output_w_type: output_w.ggml_type,
            log_scale_floor: cfg.log_scale_floor,

            rms_eps: cfg.rms_eps,
            qk_clip: cfg.qk_clip,
            rel_clip: cfg.rel_clip,
            log_scale_alpha: cfg.log_scale_alpha,
            situ_beta: cfg.situ_beta,
            situ_lb: cfg.situ_lb,
            expert_weights_scale: cfg.expert_weights_scale,
            final_logit_softcap: cfg.final_logit_softcap,
            _pad_model: [0; 2],
        };

        let mut desc_host = vec![0u8; total_desc_bytes as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer_descs.as_ptr() as *const u8,
                desc_host.as_mut_ptr(),
                layers_bytes_len,
            );
            std::ptr::copy_nonoverlapping(
                loop_ctrl_descs.as_ptr() as *const u8,
                desc_host.as_mut_ptr().add(loops_off),
                loops_bytes_len,
            );
            std::ptr::copy_nonoverlapping(
                slot_kv_dense.as_ptr() as *const u8,
                desc_host.as_mut_ptr().add(slots_off),
                slots_bytes_len,
            );
            std::ptr::copy_nonoverlapping(
                &model_desc as *const InkModelDesc as *const u8,
                desc_host.as_mut_ptr().add(model_off),
                std::mem::size_of::<InkModelDesc>(),
            );
        }
        gpu.upload_to_vram(&descs_vram_buf, 0, &desc_host)?;

        let mut ctrl_host = gpu.alloc_host_mapped(std::mem::size_of::<InkRunCtrl>())?;
        ctrl_host.as_mut_slice().fill(0);

        let dspark_gamma = std::env::var("INKLING_GAMMA")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(7)
            .clamp(1, 7);

        let dspark = if cfg.loop_passes == 3 {
            if let Some(mtp_path) = discover_mtp_safetensors(&model_path) {
                match InkDSparkState::load(&mut gpu, &module, &mtp_path, &cfg, output_w.va, output_w.ggml_type) {
                    Ok(st) => Some(st),
                    Err(e) => {
                        eprintln!("[dspark-nv] Warning: failed to load {}: {} (falling back to B=1 persistent)", mtp_path.display(), e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let kv_mb = ((swa_k_buf.size + swa_v_buf.size + full_k_buf.size + full_v_buf.size + conv_state_buf.size)
            as f64)
            / (1024.0 * 1024.0);
        let ws_mb = ((ws_buf.size + conv_backup_buf.size) as f64) / (1024.0 * 1024.0);
        eprintln!(
            "[inkling-nv] Loaded '{}' ({:.2} GiB weights @ {:.2} GB/s DMA, {} phys -> {} logical layers, {}x{} MoE, KV+Conv={:.1} MiB, WS+Backup={:.1} MiB, DSpark={}, ctx={}) in {:.2}s",
            cfg.model_name,
            (total_w_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            dma_gbps,
            cfg.n_phys_layers,
            n_slots,
            cfg.n_experts,
            cfg.n_experts_used,
            kv_mb,
            ws_mb,
            if dspark.is_some() { format!("ON(gamma={})", dspark_gamma) } else { "OFF".to_string() },
            cfg.max_ctx,
            t_load.elapsed().as_secs_f64()
        );

        Ok(Self {
            gpu,
            module,
            cfg,
            tok,
            model_ref: model_ref.to_string(),
            weights_buf,
            swa_k_buf,
            swa_v_buf,
            full_k_buf,
            full_v_buf,
            conv_state_buf,
            conv_backup_buf,
            ws_buf,
            ws_bar_state_buf,
            descs_vram_buf,
            model_desc_va,
            ctrl_host,
            done_seq_counter: 0,
            conv_total_floats,
            dspark_taps_va: wb + o_dspark_taps,
            dspark,
            dspark_gamma,
            dspark_cache_valid: false,
            cached_tokens: Vec::new(),
            cached_passes: loop_passes,
        })
    }

    pub fn reset_cache(&mut self) -> Result<(), String> {
        self.gpu.zero_vram(&self.conv_state_buf)?;
        self.cached_tokens.clear();
        self.dspark_cache_valid = false;
        Ok(())
    }

    pub fn normalize_effort(raw_eff: &str) -> &'static str {
        match raw_eff.trim() {
            "none" | "0" | "0.0" | "off" | "false" => "0",
            "minimal" | "0.1" => "0.1",
            "low" | "0.2" => "0.2",
            "medium" | "0.5" | "0.7" => "0.7",
            "high" | "0.9" => "0.9",
            "xhigh" | "max" | "0.99" | "1" | "1.0" => "0.99",
            _ => "0.2",
        }
    }

    pub fn format_chat_prompt(
        &self,
        user_prompt: &str,
        sys_prompt: Option<&str>,
        effort_override: Option<&str>,
    ) -> Vec<i32> {
        let user_no_think = user_prompt.contains("/no_think");
        let clean_user = user_prompt.replace("/no_think", "");
        let clean_user = clean_user.trim();
        let raw_eff = effort_override.unwrap_or(if user_no_think { "0" } else { "0.2" });
        let eff = Self::normalize_effort(raw_eff);

        let mut s = String::with_capacity(clean_user.len() + 256);
        if let Some(sys) = sys_prompt.filter(|x| !x.trim().is_empty()) {
            s.push_str("<|message_system|><|content_text|>");
            s.push_str(sys.trim());
            s.push_str("<|end_message|>");
        }
        s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        s.push_str(eff);
        s.push_str("<|end_message|><|message_user|><|content_text|>");
        s.push_str(clean_user);
        s.push_str("<|end_message|>");
        if eff == "0" {
            // Instant non-thinking mode: jump straight to final `<|content_text|>` response
            s.push_str("<|message_model|><|content_text|>");
        }
        self.tok.encode(&s)
    }

    pub fn format_multiturn_chat(
        &self,
        messages: &[(String, String)],
        effort_override: Option<&str>,
    ) -> Vec<i32> {
        let last_no_think = messages
            .last()
            .map_or(false, |(r, c)| r == "user" && c.contains("/no_think"));
        let eff = Self::normalize_effort(effort_override.unwrap_or(if last_no_think { "0" } else { "0.2" }));

        let mut s = String::with_capacity(512);
        // Emit any caller system messages first
        for (role, content) in messages {
            if role == "system" && !content.trim().is_empty() {
                s.push_str("<|message_system|><|content_text|>");
                s.push_str(content.trim());
                s.push_str("<|end_message|>");
            }
        }
        s.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        s.push_str(eff);
        s.push_str("<|end_message|>");

        for (idx, (role, content)) in messages.iter().enumerate() {
            if role == "system" {
                continue;
            } else if role == "user" {
                s.push_str("<|message_user|><|content_text|>");
                if idx + 1 == messages.len() && last_no_think {
                    s.push_str(content.replace("/no_think", "").trim());
                } else {
                    s.push_str(content.trim());
                }
                s.push_str("<|end_message|>");
            } else if role == "assistant" {
                // Strip any rendered `<think>...</think>` wrapper back into Inkling native tags or clean text
                let trimmed = content.trim();
                if let (Some(ts), Some(te)) = (trimmed.find("<think>"), trimmed.find("</think>")) {
                    let think_body = trimmed[ts + 7..te].trim();
                    let text_body = trimmed[te + 8..].trim();
                    if !think_body.is_empty() {
                        s.push_str("<|message_model|><|content_thinking|>");
                        s.push_str(think_body);
                        s.push_str("<|end_message|>");
                    }
                    s.push_str("<|message_model|><|content_text|>");
                    s.push_str(text_body);
                    s.push_str("<|end_message|>");
                } else {
                    s.push_str("<|message_model|><|content_text|>");
                    s.push_str(trimmed);
                    s.push_str("<|end_message|>");
                }
            }
        }
        if eff == "0" {
            s.push_str("<|message_model|><|content_text|>");
        }
        self.tok.encode(&s)
    }

    /// Execute one batched `BM <= 8` 48-layer forward pass (`ink_batch_forward_bm8_megakernel`) across all 68 SMs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch(
        &mut self,
        tokens: &[i32],
        start_pos: usize,
        logits_mask: u32,
        export_taps: bool,
        save_conv_backup: bool,
        temperature: f32,
        rng_seed: u64,
        step_offset: u32,
        eff_passes: usize,
    ) -> Result<[i32; 8], String> {
        let b_len = tokens.len();
        assert!((1..=8).contains(&b_len), "forward_batch requires 1..=8 tokens");
        if start_pos + b_len > self.cfg.max_ctx {
            return Err(format!(
                "Batch [{}..{}] exceeds max_ctx {}",
                start_pos,
                start_pos + b_len,
                self.cfg.max_ctx
            ));
        }

        self.done_seq_counter += 1;
        let expected_seq = self.done_seq_counter;
        let ctrl_ptr = self.ctrl_host.as_ptr() as *mut InkRunCtrl;
        unsafe {
            let mut in_arr = [0i32; 8];
            in_arr[..b_len].copy_from_slice(tokens);
            (*ctrl_ptr).start_pos = start_pos as i32;
            (*ctrl_ptr).eff_passes = eff_passes.clamp(1, 3) as i32;
            (*ctrl_ptr).temperature = temperature;
            (*ctrl_ptr).step_offset = step_offset;
            (*ctrl_ptr).rng_seed = rng_seed;
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).export_taps = if export_taps { 1 } else { 0 };
            (*ctrl_ptr).save_conv_backup = if save_conv_backup { 1 } else { 0 };
            (*ctrl_ptr).batch_len = b_len as i32;
            (*ctrl_ptr).logits_mask = logits_mask;
            (*ctrl_ptr).inject_pos = 0;
            (*ctrl_ptr).inject_cnt = 0;
            (*ctrl_ptr).batch_tokens = in_arr;
            (*ctrl_ptr).batch_out_toks = [0; 8];
            (*ctrl_ptr).expected_seq = expected_seq;
        }
        fence(Ordering::SeqCst);

        let mut mk_args = [0u8; 16];
        put_u64(&mut mk_args, 0, self.model_desc_va);
        put_u64(&mut mk_args, 8, self.ctrl_host.va_addr);

        self.gpu.timeline_value += 1;
        let sig_val = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        let k_bm8 = &self.module.kernels["ink_batch_forward_bm8_megakernel"];
        let (qmd_va, _) = self.gpu.prepare_qmd(
            k_bm8,
            (self.gpu.active_sms, 1, 1),
            (512, 1, 1),
            &mk_args,
            Some((sig_va, sig_val)),
        );
        self.gpu.launch_qmd_chain(qmd_va, true)?;

        let sig_ptr = self.gpu.timeline_buf.as_ptr() as *const u64;
        let t_wait = Instant::now();
        loop {
            let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
            if cur_sig >= sig_val {
                break;
            }
            let err = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).error_code) };
            if err != 0 {
                let _ = self.gpu.zero_vram(&self.ws_bar_state_buf);
                return Err(format!("ink_batch_forward_bm8_megakernel barrier watchdog tripped at {}", err));
            }
            self.gpu.poll_gsp_faults()?;
            if t_wait.elapsed() > Duration::from_secs(30) {
                return Err("ink_batch_forward_bm8_megakernel timed out (>30s)".into());
            }
            std::hint::spin_loop();
        }

        let out = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).batch_out_toks) };
        Ok(out)
    }

    /// Restore `ShortConv` state from `conv_backup[slot_b]` (`~5 us` VRAM-to-VRAM copy) after speculative rejection.
    pub fn restore_conv_backup(&mut self, slot_b: usize) -> Result<(), String> {
        assert!(slot_b < 8);
        let src_va = self.conv_backup_buf.va_addr + ((slot_b * self.conv_total_floats * 4) as u64);
        let mut args = [0u8; 20];
        put_u64(&mut args, 0, self.conv_state_buf.va_addr);
        put_u64(&mut args, 8, src_va);
        args[16..20].copy_from_slice(&(self.conv_total_floats as i32).to_le_bytes());

        self.gpu.timeline_value += 1;
        let sig_val = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        let k_cp = &self.module.kernels["ink_copy_f32_vram_kernel"];
        let (qmd_va, _) = self.gpu.prepare_qmd(
            k_cp,
            (self.gpu.active_sms, 1, 1),
            (512, 1, 1),
            &args,
            Some((sig_va, sig_val)),
        );
        self.gpu.launch_qmd_chain(qmd_va, false)?;
        self.gpu.wait_timeline(sig_val, Duration::from_secs(5))
    }

    /// Inject `inject_cnt` (`1..=8`) target hidden tap rows from `ws_dspark_taps` into `draft_k/v [inject_pos ..]` + `ws_f_rnn`.
    pub fn dspark_inject_kv(&mut self, inject_pos: usize, inject_cnt: usize) -> Result<(), String> {
        let dspark_va = match self.dspark {
            Some(ref d) => d.desc_va,
            None => return Ok(()),
        };
        if inject_cnt == 0 {
            return Ok(());
        }
        assert!(inject_cnt <= 8);

        self.done_seq_counter += 1;
        let expected_seq = self.done_seq_counter;
        let ctrl_ptr = self.ctrl_host.as_ptr() as *mut InkRunCtrl;
        unsafe {
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).inject_pos = inject_pos as i32;
            (*ctrl_ptr).inject_cnt = inject_cnt as i32;
            (*ctrl_ptr).expected_seq = expected_seq;
        }
        fence(Ordering::SeqCst);

        let mut args = [0u8; 24];
        put_u64(&mut args, 0, self.model_desc_va);
        put_u64(&mut args, 8, dspark_va);
        put_u64(&mut args, 16, self.ctrl_host.va_addr);

        self.gpu.timeline_value += 1;
        let sig_val = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        let k_inj = &self.module.kernels["ink_dspark_inject_kv_kernel"];
        let (qmd_va, _) = self.gpu.prepare_qmd(
            k_inj,
            (self.gpu.active_sms, 1, 1),
            (512, 1, 1),
            &args,
            Some((sig_va, sig_val)),
        );
        self.gpu.launch_qmd_chain(qmd_va, true)?;

        let sig_ptr = self.gpu.timeline_buf.as_ptr() as *const u64;
        let t_wait = Instant::now();
        loop {
            let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
            if cur_sig >= sig_val {
                break;
            }
            let err = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).error_code) };
            if err != 0 {
                let _ = self.gpu.zero_vram(&self.ws_bar_state_buf);
                return Err(format!("ink_dspark_inject_kv_kernel barrier watchdog tripped at {}", err));
            }
            self.gpu.poll_gsp_faults()?;
            if t_wait.elapsed() > Duration::from_secs(10) {
                return Err("ink_dspark_inject_kv_kernel timed out (>10s)".into());
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Propose `bd_len - 1` (`gamma = 7` for `bd_len = 8`) draft tokens in one fused 68-SM launch (`~2.3 ms`),
    /// optionally folding in `inject_cnt` committed context taps from `ws_dspark_taps` (`0` extra doorbells!).
    pub fn dspark_propose(
        &mut self,
        anchor_tok: i32,
        anchor_pos: usize,
        bd_len: usize,
        req_gamma: usize,
        inject_pos: usize,
        inject_cnt: usize,
    ) -> Result<[i32; 8], String> {
        let dspark_va = self
            .dspark
            .as_ref()
            .ok_or_else(|| "DSpark drafter is not loaded".to_string())?
            .desc_va;
        assert!((2..=8).contains(&bd_len));
        assert!(inject_cnt <= 8);

        self.done_seq_counter += 1;
        let expected_seq = self.done_seq_counter;
        let ctrl_ptr = self.ctrl_host.as_ptr() as *mut InkRunCtrl;
        unsafe {
            (*ctrl_ptr).start_token = anchor_tok;
            (*ctrl_ptr).start_pos = anchor_pos as i32;
            (*ctrl_ptr).max_steps = req_gamma.clamp(1, 7) as i32;
            (*ctrl_ptr).batch_len = bd_len as i32;
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).inject_pos = inject_pos as i32;
            (*ctrl_ptr).inject_cnt = inject_cnt as i32;
            (*ctrl_ptr).batch_out_toks = [0; 8];
            (*ctrl_ptr).expected_seq = expected_seq;
        }
        fence(Ordering::SeqCst);

        let mut args = [0u8; 24];
        put_u64(&mut args, 0, self.model_desc_va);
        put_u64(&mut args, 8, dspark_va);
        put_u64(&mut args, 16, self.ctrl_host.va_addr);

        self.gpu.timeline_value += 1;
        let sig_val = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        let k_prop = &self.module.kernels["ink_dspark_propose_megakernel"];
        let (qmd_va, _) = self.gpu.prepare_qmd(
            k_prop,
            (self.gpu.active_sms, 1, 1),
            (512, 1, 1),
            &args,
            Some((sig_va, sig_val)),
        );
        self.gpu.launch_qmd_chain(qmd_va, true)?;

        let sig_ptr = self.gpu.timeline_buf.as_ptr() as *const u64;
        let t_wait = Instant::now();
        loop {
            let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
            if cur_sig >= sig_val {
                break;
            }
            let err = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).error_code) };
            if err != 0 {
                let _ = self.gpu.zero_vram(&self.ws_bar_state_buf);
                return Err(format!("ink_dspark_propose_megakernel barrier watchdog tripped at {}", err));
            }
            self.gpu.poll_gsp_faults()?;
            if t_wait.elapsed() > Duration::from_secs(15) {
                return Err("ink_dspark_propose_megakernel timed out (>15s)".into());
            }
            std::hint::spin_loop();
        }

        let out = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).batch_out_toks) };
        Ok(out)
    }

    pub fn forward_one_token(
        &mut self,
        tok_id: i32,
        pos: usize,
        compute_logits: bool,
        temperature: f32,
        rng_seed: u64,
        step: u32,
        passes_override: Option<usize>,
    ) -> Result<i32, String> {
        let eff_passes = passes_override.unwrap_or(self.cfg.loop_passes).clamp(1, 3);
        let out = self.forward_batch(
            &[tok_id],
            pos,
            if compute_logits { 1 } else { 0 },
            false,
            false,
            temperature,
            rng_seed,
            step,
            eff_passes,
        )?;
        Ok(out[0])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_cancel<F>(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        rng_seed: u64,
        passes_override: Option<usize>,
        enable_spec: bool,
        reset_kv: bool,
        external_cancel: Option<&AtomicBool>,
        raw_tokens_stream: bool,
        mut on_piece: F,
    ) -> Result<InklingGenStats, String>
    where
        F: FnMut(i32, &str) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err("Prompt tokens cannot be empty".into());
        }
        let eff_passes = passes_override
            .unwrap_or(self.cfg.loop_passes)
            .clamp(1, self.cfg.loop_passes);
        let logical_layers =
            self.cfg.n_phys_layers + (eff_passes - 1) * (self.cfg.loop_end - self.cfg.loop_start);

        let use_dspark = enable_spec
            && self.dspark.is_some()
            && eff_passes == 3
            && self.cfg.loop_passes == 3
            && temperature <= 1e-5
            && self.dspark_gamma >= 1;

        if reset_kv || self.cached_passes != eff_passes || (use_dspark && !self.dspark_cache_valid && !self.cached_tokens.is_empty()) {
            self.reset_cache()?;
            self.cached_passes = eff_passes;
        }
        if !use_dspark {
            self.dspark_cache_valid = false;
        }

        let prompt_len = prompt_tokens.len();
        if prompt_len >= self.cfg.max_ctx {
            return Err(format!(
                "Prompt length {} exceeds max_ctx {}",
                prompt_len, self.cfg.max_ctx
            ));
        }

        // Exact causal prefix reuse (valid whenever `prompt[..prompt_len-1]` extends `self.cached_tokens`)
        let prefix_target = &prompt_tokens[..prompt_len - 1];
        let mut pos = if !self.cached_tokens.is_empty()
            && prefix_target.len() >= self.cached_tokens.len()
            && &prefix_target[..self.cached_tokens.len()] == self.cached_tokens.as_slice()
        {
            self.cached_tokens.len()
        } else {
            if !self.cached_tokens.is_empty() {
                self.reset_cache()?;
            }
            0
        };
        let cached_prefix_tokens = pos;

        let t_total = Instant::now();
        let t_prefill = Instant::now();
        let mut launches = 0usize;
        let mut first_tok = 0i32;
        let mut pending_inject: (usize, usize) = (0, 0);

        // 1. Batched `BM <= 8` Prefill over `prompt_tokens[pos .. prompt_len]`
        while pos < prompt_len {
            let take = (prompt_len - pos).min(8);
            let is_final_chunk = pos + take == prompt_len;
            let logits_mask = if is_final_chunk { 1u32 << (take - 1) } else { 0u32 };
            let out = self.forward_batch(
                &prompt_tokens[pos..pos + take],
                pos,
                logits_mask,
                use_dspark,
                false,
                temperature,
                rng_seed,
                0,
                eff_passes,
            )?;
            launches += 1;
            if use_dspark {
                if !is_final_chunk {
                    self.dspark_inject_kv(pos, take)?;
                    launches += 1;
                } else {
                    pending_inject = (pos, take);
                }
            }
            if is_final_chunk {
                first_tok = out[take - 1];
            }
            pos += take;
        }

        let mut committed_seq = prompt_tokens.to_vec();
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = prefill_ms;

        let max_steps_cap = max_tokens.clamp(1, 4096);
        let mut generated = Vec::with_capacity(max_steps_cap);
        let mut decoder = Utf8StreamDecoder::new();
        let mut in_thinking = false;
        let mut stop_reason: &'static str = "length";

        let mut emit_tok =
            |tok: &BpeTokenizer, tid: i32, dec: &mut Utf8StreamDecoder, in_th: &mut bool| -> bool {
                if raw_tokens_stream {
                    let piece = dec.push(tok.token_bytes(tid));
                    return on_piece(tid, &piece);
                }
                match tid {
                    TOK_MSG_MODEL | TOK_CONTENT_TEXT | TOK_END_SAMPLING | TOK_EOT => {
                        on_piece(tid, "")
                    }
                    TOK_CONTENT_THINK => {
                        *in_th = true;
                        on_piece(tid, "<think>\n")
                    }
                    TOK_END_MESSAGE => {
                        if *in_th {
                            *in_th = false;
                            on_piece(tid, "\n</think>\n\n")
                        } else {
                            on_piece(tid, "")
                        }
                    }
                    _ => {
                        let piece = dec.push(tok.token_bytes(tid));
                        on_piece(tid, &piece)
                    }
                }
            };

        if first_tok == TOK_END_SAMPLING || first_tok == TOK_EOT {
            stop_reason = "stop";
            if use_dspark {
                if pending_inject.1 > 0 {
                    self.dspark_inject_kv(pending_inject.0, pending_inject.1)?;
                    launches += 1;
                }
                self.dspark_cache_valid = true;
            }
            self.cached_tokens = committed_seq;
            return Ok(self.finish_stats(
                eff_passes,
                logical_layers,
                prompt_len,
                cached_prefix_tokens,
                generated,
                prefill_ms,
                0.0,
                t_total.elapsed().as_secs_f64() * 1000.0,
                ttft_ms,
                0,
                0,
                launches,
                stop_reason,
            ));
        }

        generated.push(first_tok);
        let mut keep_going = emit_tok(&self.tok, first_tok, &mut decoder, &mut in_thinking);
        if external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
            keep_going = false;
        }
        if !keep_going {
            stop_reason = "cancelled";
        }

        let t_decode = Instant::now();
        let mut spec_verify_calls = 0usize;
        let mut spec_accepted_extra = 0usize;

        if use_dspark {
            // 2A. 100% Lossless Block-8 `DSpark` (`Inkling2MTP`) Speculative Decoding Loop
            let mut cur_tok = first_tok;
            let mut cur_pos = prompt_len;

            while keep_going && generated.len() < max_steps_cap && cur_pos < self.cfg.max_ctx {
                let rem_gen = max_steps_cap - generated.len();
                let rem_ctx = self.cfg.max_ctx - cur_pos;

                if rem_gen == 1 || rem_ctx == 1 {
                    if pending_inject.1 > 0 {
                        self.dspark_inject_kv(pending_inject.0, pending_inject.1)?;
                        launches += 1;
                    }
                    let out = self.forward_batch(
                        &[cur_tok],
                        cur_pos,
                        1,
                        true,
                        false,
                        0.0,
                        rng_seed,
                        0,
                        eff_passes,
                    )?;
                    launches += 1;
                    spec_verify_calls += 1;
                    committed_seq.push(cur_tok);
                    pending_inject = (cur_pos, 1);
                    cur_pos += 1;

                    let next_tok = out[0];
                    if next_tok == TOK_END_SAMPLING || next_tok == TOK_EOT {
                        stop_reason = "stop";
                    } else {
                        generated.push(next_tok);
                        let ok = emit_tok(&self.tok, next_tok, &mut decoder, &mut in_thinking);
                        if !ok || external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                            stop_reason = "cancelled";
                        } else if cur_pos >= self.cfg.max_ctx && generated.len() < max_steps_cap {
                            stop_reason = "context_limit";
                        }
                    }
                    break;
                }

                // Cap verification length by gamma, remaining context, and remaining budget!
                let mut k_ver = self.dspark_gamma.min(rem_ctx - 1).min(rem_gen - 1);
                // Propose `k_ver` draft tokens in 1 fused 68-SM launch (folding in `pending_inject` for 0 extra doorbells!)
                let drafts = self.dspark_propose(cur_tok, cur_pos, 8, k_ver, pending_inject.0, pending_inject.1)?;
                launches += 1;

                // Stop verification before any drafted EOS token!
                for j in 0..k_ver {
                    if drafts[j] == TOK_END_SAMPLING || drafts[j] == TOK_EOT {
                        k_ver = j;
                        break;
                    }
                }
                let v_len = 1 + k_ver;
                let mut verify_in = [0i32; 8];
                verify_in[0] = cur_tok;
                for j in 0..k_ver {
                    verify_in[1 + j] = drafts[j];
                }
                let logits_mask = (1u32 << v_len) - 1;
                let verify_out = self.forward_batch(
                    &verify_in[..v_len],
                    cur_pos,
                    logits_mask,
                    true,
                    v_len > 1,
                    0.0,
                    rng_seed,
                    0,
                    eff_passes,
                )?;
                launches += 1;
                spec_verify_calls += 1;

                let mut accepted_drafts = 0usize;
                for j in 0..k_ver {
                    if verify_out[j] == drafts[j] {
                        accepted_drafts += 1;
                    } else {
                        break;
                    }
                }

                let mut last_t_idx = 0usize;
                let mut stopped_here = false;
                for t_idx in 0..=accepted_drafts {
                    last_t_idx = t_idx;
                    let cand_tok = verify_out[t_idx];
                    if cand_tok == TOK_END_SAMPLING || cand_tok == TOK_EOT {
                        stop_reason = "stop";
                        keep_going = false;
                        stopped_here = true;
                        break;
                    }
                    generated.push(cand_tok);
                    let ok = emit_tok(&self.tok, cand_tok, &mut decoder, &mut in_thinking);
                    if !ok || external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                        stop_reason = "cancelled";
                        keep_going = false;
                        stopped_here = true;
                        break;
                    }
                    if generated.len() >= max_steps_cap {
                        stop_reason = "length";
                        keep_going = false;
                        stopped_here = true;
                        break;
                    }
                }

                let commit_count = last_t_idx + 1;
                committed_seq.extend_from_slice(&verify_in[..commit_count]);
                spec_accepted_extra += last_t_idx;
                if commit_count < v_len {
                    self.restore_conv_backup(commit_count - 1)?;
                    launches += 1;
                }
                pending_inject = (cur_pos, commit_count);
                cur_pos += commit_count;

                if !stopped_here {
                    cur_tok = verify_out[accepted_drafts];
                    if cur_pos >= self.cfg.max_ctx && generated.len() < max_steps_cap {
                        stop_reason = "context_limit";
                        keep_going = false;
                    }
                }
            }

            if pending_inject.1 > 0 {
                self.dspark_inject_kv(pending_inject.0, pending_inject.1)?;
                launches += 1;
            }
            self.dspark_cache_valid = true;
        } else if keep_going && max_steps_cap > 1 && prompt_len < self.cfg.max_ctx {
            // 2B. Single-Doorbell 68-SM Persistent Multi-Token Rollout (`0` per-token launches!)
            let rollout_steps = (max_steps_cap - 1).min(4070);
            let use_contig = std::env::var("INKLING_CONTIG").ok().as_deref() == Some("1");
            self.done_seq_counter += 1;
            let expected_seq = self.done_seq_counter;
            let ctrl_ptr = self.ctrl_host.as_ptr() as *mut InkRunCtrl;
            unsafe {
                (*ctrl_ptr).start_token = first_tok;
                (*ctrl_ptr).start_pos = prompt_len as i32;
                (*ctrl_ptr).max_steps = rollout_steps as i32;
                (*ctrl_ptr).eff_passes = eff_passes as i32;
                (*ctrl_ptr).temperature = temperature;
                (*ctrl_ptr).step_offset = 1;
                (*ctrl_ptr).rng_seed = rng_seed;
                (*ctrl_ptr).cancel_flag = 0;
                (*ctrl_ptr).error_code = 0;
                (*ctrl_ptr).steps_completed = 0;
                (*ctrl_ptr).stop_reason = 0;
                (*ctrl_ptr).export_taps = 0;
                (*ctrl_ptr).save_conv_backup = 0;
                (*ctrl_ptr)._pad_ctrl[0] = if use_contig { 1 } else { 0 };
                (*ctrl_ptr).expected_seq = expected_seq;
            }
            fence(Ordering::SeqCst);

            let mut mk_args = [0u8; 16];
            put_u64(&mut mk_args, 0, self.model_desc_va);
            put_u64(&mut mk_args, 8, self.ctrl_host.va_addr);

            self.gpu.timeline_value += 1;
            let sig_val = self.gpu.timeline_value;
            let sig_va = self.gpu.timeline_buf.va_addr;
            let k_mk = &self.module.kernels["ink_persistent_decode_megakernel"];
            let (qmd_va, _) = self.gpu.prepare_qmd(
                k_mk,
                (self.gpu.active_sms, 1, 1),
                (512, 1, 1),
                &mk_args,
                Some((sig_va, sig_val)),
            );
            self.gpu.launch_qmd_chain(qmd_va, true)?;
            launches += 1;

            let sig_ptr = self.gpu.timeline_buf.as_ptr() as *const u64;
            let rollout_t0 = Instant::now();
            let mut streamed = 0usize;
            let mut prev_uncommitted = first_tok;
            let mut user_aborted = false;
            let mut eos_seen = false;

            loop {
                let done_steps = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).steps_completed) } as usize;
                while streamed < done_steps {
                    let next_tok = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).tokens_out[streamed]) };
                    streamed += 1;
                    // Step `streamed - 1` evaluated `prev_uncommitted` into KV + `conv_state`!
                    committed_seq.push(prev_uncommitted);
                    prev_uncommitted = next_tok;

                    if eos_seen || user_aborted {
                        continue;
                    }
                    if next_tok == TOK_END_SAMPLING || next_tok == TOK_EOT {
                        eos_seen = true;
                        stop_reason = "stop";
                        continue;
                    }
                    generated.push(next_tok);
                    let ok = emit_tok(&self.tok, next_tok, &mut decoder, &mut in_thinking);
                    if !ok || external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                        user_aborted = true;
                        stop_reason = "cancelled";
                        unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                    }
                }

                if !user_aborted && external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                    user_aborted = true;
                    stop_reason = "cancelled";
                    unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                }

                let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
                if cur_sig >= sig_val {
                    let final_steps =
                        unsafe { std::ptr::read_volatile(&(*ctrl_ptr).steps_completed) } as usize;
                    while streamed < final_steps {
                        let next_tok =
                            unsafe { std::ptr::read_volatile(&(*ctrl_ptr).tokens_out[streamed]) };
                        streamed += 1;
                        committed_seq.push(prev_uncommitted);
                        prev_uncommitted = next_tok;

                        if eos_seen || user_aborted {
                            continue;
                        }
                        if next_tok == TOK_END_SAMPLING || next_tok == TOK_EOT {
                            eos_seen = true;
                            stop_reason = "stop";
                            continue;
                        }
                        generated.push(next_tok);
                        let ok = emit_tok(&self.tok, next_tok, &mut decoder, &mut in_thinking);
                        if !ok {
                            user_aborted = true;
                            stop_reason = "cancelled";
                        }
                    }
                    let sr = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).stop_reason) };
                    if sr == 3 && !eos_seen && !user_aborted {
                        stop_reason = "context_limit";
                    }
                    if std::env::var("INKLING_PROFILE").ok().as_deref() == Some("1") && streamed > 0 {
                        let p_base = unsafe { (&(*ctrl_ptr).tokens_out[4076] as *const i32) as *const u64 };
                        let mut ns = [0u64; 9];
                        for k in 0..9 {
                            ns[k] = unsafe { std::ptr::read_volatile(p_base.add(k)) };
                        }
                        let inv = 1.0 / ((streamed as f64) * 1e6);
                        eprintln!(
                            "[inkling-prof] {} steps (contig={}): Emb={:.3}ms | L1={:.3}ms | L2(QKV)={:.3}ms | L3+L4(Attn)={:.3}ms | L5(wo)={:.3}ms | L6(Rtr)={:.3}ms | L7(GateUp)={:.3}ms | L8(Down)={:.3}ms | LMHead={:.3}ms | Sum={:.3}ms/tok",
                            streamed,
                            use_contig,
                            ns[0] as f64 * inv,
                            ns[1] as f64 * inv,
                            ns[2] as f64 * inv,
                            ns[3] as f64 * inv,
                            ns[4] as f64 * inv,
                            ns[5] as f64 * inv,
                            ns[6] as f64 * inv,
                            ns[7] as f64 * inv,
                            ns[8] as f64 * inv,
                            ns.iter().sum::<u64>() as f64 * inv,
                        );
                    }
                    break;
                }

                let err = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).error_code) };
                if err != 0 {
                    let _ = self.gpu.zero_vram(&self.ws_bar_state_buf);
                    self.cached_tokens.clear();
                    return Err(format!(
                        "ink_persistent_decode_megakernel barrier watchdog tripped at step {}",
                        err
                    ));
                }
                self.gpu.poll_gsp_faults()?;
                if rollout_t0.elapsed() > Duration::from_secs(180) {
                    unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                    self.cached_tokens.clear();
                    return Err("ink_persistent_decode_megakernel exceeded 180s timeout".into());
                }
                std::hint::spin_loop();
            }
        }

        let tail = decoder.flush();
        if !tail.is_empty() {
            let _ = on_piece(-1, &tail);
        }

        self.cached_tokens = committed_seq;

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        Ok(self.finish_stats(
            eff_passes,
            logical_layers,
            prompt_len,
            cached_prefix_tokens,
            generated,
            prefill_ms,
            decode_ms,
            total_ms,
            ttft_ms,
            spec_verify_calls,
            spec_accepted_extra,
            launches,
            stop_reason,
        ))
    }

    pub fn generate<F>(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        rng_seed: u64,
        passes_override: Option<usize>,
        raw_tokens_stream: bool,
        mut on_piece: F,
    ) -> Result<InklingGenStats, String>
    where
        F: FnMut(i32, &str),
    {
        self.generate_with_cancel(
            prompt_tokens,
            max_tokens,
            temperature,
            rng_seed,
            passes_override,
            true,
            true,
            None,
            raw_tokens_stream,
            |tid, s| {
                on_piece(tid, s);
                true
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_stats(
        &self,
        eff_passes: usize,
        logical_layers: usize,
        prompt_tokens: usize,
        cached_prefix_tokens: usize,
        generated: Vec<i32>,
        prefill_ms: f64,
        decode_ms: f64,
        total_ms: f64,
        ttft_ms: f64,
        spec_verify_calls: usize,
        spec_accepted_extra: usize,
        gpu_launches: usize,
        stop_reason: &'static str,
    ) -> InklingGenStats {
        let dec_toks = generated.len().saturating_sub(1);
        let decode_tok_per_s = if decode_ms > 0.0 && dec_toks > 0 {
            (dec_toks as f64) / (decode_ms / 1000.0)
        } else if generated.len() == 1 {
            1000.0 / prefill_ms.max(1e-3)
        } else {
            0.0
        };
        let new_prefill_toks = prompt_tokens.saturating_sub(cached_prefix_tokens).max(1);
        let prefill_tok_per_s = (new_prefill_toks as f64) / (prefill_ms / 1000.0).max(1e-6);

        let vram_weights_mb = (self.weights_buf.size as f64) / (1024.0 * 1024.0);
        let vram_kv_conv_mb = ((self.swa_k_buf.size
            + self.swa_v_buf.size
            + self.full_k_buf.size
            + self.full_v_buf.size
            + self.conv_state_buf.size) as f64)
            / (1024.0 * 1024.0);

        InklingGenStats {
            model_name: self.cfg.model_name.clone(),
            loop_passes: eff_passes,
            logical_layers_per_tok: logical_layers,
            prompt_tokens,
            cached_prefix_tokens,
            completion_tokens: generated.len(),
            prefill_ms,
            decode_ms,
            total_ms,
            prefill_tok_per_s,
            decode_tok_per_s,
            ttft_ms,
            vram_weights_mb,
            vram_kv_conv_mb,
            spec_verify_calls,
            spec_accepted_extra,
            gpu_launches,
            stop_reason,
            token_ids: generated,
        }
    }
}
