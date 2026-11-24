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

use std::path::Path;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::dspark::{discover_mtp_safetensors, InkDSparkState, InkRunCtrl};
use crate::gguf::{resolve_model_path, BpeTokenizer, GgufFile, Utf8StreamDecoder};
use crate::hw::{put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer, VramCheckpoint};

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

pub static INKLING_CUBIN: &[u8] = include_bytes!("../../kernels/inkling_sm86.cubin");

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


const _: () = assert!(std::mem::size_of::<InkPhysLayerDesc>() == 256);
const _: () = assert!(std::mem::size_of::<InkLoopCtrlDesc>() == 128);
const _: () = assert!(std::mem::size_of::<InkModelDesc>() == 576);

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

pub struct InklingEngine {
    module: CubinModule,
    cfg: InklingConfig,
    tok: BpeTokenizer,
    model_ref: String,
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
    dspark_taps_va: u64,
    dspark: Option<InkDSparkState>,
    pre_dspark_cp: Option<VramCheckpoint>,
    dspark_gamma: usize,
    dspark_cache_valid: bool,
    cached_tokens: Vec<i32>,
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
            if let Some(arch) = gf.get_str("general.architecture") {
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

    #[inline]
    pub fn model_ref(&self) -> &str {
        &self.model_ref
    }

    #[inline]
    pub fn model_name(&self) -> &str {
        &self.cfg.model_name
    }

    #[inline]
    pub fn has_dspark(&self) -> bool {
        self.dspark.is_some()
    }

    #[inline]
    pub fn dspark_gamma(&self) -> usize {
        self.dspark_gamma
    }

    #[inline]
    pub fn set_dspark_gamma(&mut self, gamma: usize) {
        self.dspark_gamma = gamma.clamp(1, 7);
    }

    pub fn dspark_status_json(&self) -> String {
        if let Some(ref ds) = self.dspark {
            format!(
                "{{\"active\":true,\"model\":\"Inkling2MTP\",\"gamma\":{},\"vram_mb\":{:.1}}}",
                self.dspark_gamma, ds.total_vram_mb
            )
        } else {
            "{\"active\":false}".to_string()
        }
    }

    /// Reclaim `DSpark` (`~447-471 MiB` GDDR6X) back to the high-VRAM bump allocator so
    /// co-resident `Qwen3-ASR-0.6B` + `Qwen3-TTS-0.6B` + `12Hz Code2Wav` fit with ~400 MiB margin!
    pub fn reclaim_dspark_for_voice(&mut self, gpu: &mut NvGpu) -> f64 {
        if let (Some(ds), Some(cp)) = (self.dspark.take(), self.pre_dspark_cp.take()) {
            let mb = ds.total_vram_mb;
            drop(ds);
            self.dspark_cache_valid = false;
            gpu.restore_vram_checkpoint(cp);
            mb
        } else {
            0.0
        }
    }

    #[inline]
    pub fn encode_raw(&self, text: &str) -> Vec<i32> {
        self.tok.encode(text)
    }

    #[inline]
    pub fn decode_tokens(&self, ids: &[i32]) -> String {
        self.tok.decode(ids)
    }

    pub fn load(gpu: &mut NvGpu, model_ref: &str, max_context_req: usize) -> Result<Self, String> {
        let model_path = resolve_model_path(model_ref)?;
        let t_load = Instant::now();
        let gguf = GgufFile::open(&model_path)?;
        let arch = gguf.get_str("general.architecture").unwrap_or("");
        if arch != "inkling2_loop" {
            return Err(format!("Expected general.architecture='inkling2_loop', got '{}'", arch));
        }

        let tok = gguf.build_tokenizer()?;
        let get_u = |k: &str| gguf.req_usize(k);
        let get_f = |k: &str, def: f32| gguf.get_f32_or(k, def);

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

        let n_heads = gguf
            .get_i32_arr("inkling2_loop.attention.head_count")
            .and_then(|v| v.first().copied())
            .map(|v| v as usize)
            .unwrap_or(24);
        let n_kv_heads = gguf
            .get_i32_arr("inkling2_loop.attention.head_count_kv")
            .and_then(|v| v.first().copied())
            .map(|v| v as usize)
            .unwrap_or(8);
        let swa_pattern: Vec<bool> = match gguf.get_i32_arr("inkling2_loop.attention.sliding_window_pattern") {
            Some(v) if v.len() == n_phys => v.iter().map(|&x| x != 0).collect(),
            _ => (0..n_phys).map(|i| (i % 6) != 5).collect(),
        };

        let cfg = InklingConfig {
            model_name: gguf.get_str("general.name").unwrap_or("Inkling-2-Mini").to_string(),
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
        let names_sorted = gguf.tensor_names_by_offset();

        let mut total_w_bytes: u64 = 0;
        let mut tensor_rel_off =
            std::collections::HashMap::<String, (u64, usize, i32)>::with_capacity(names_sorted.len());
        for name in &names_sorted {
            let info = gguf.req_tensor_info(name)?;
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

        // Stream from mmap -> GDDR6X in coalesced 32 MiB DMA batches (`~240` DMAs instead of `~1,450`),
        // losslessly splitting quantized matrices into `[16B-aligned qs | contiguous fp16 d]` (`0` extra VRAM!)
        const CHUNK_MAX: usize = 32 * 1024 * 1024;
        struct VramStageCoalescer<'a> {
            buf: &'a VramBuffer,
            stage: Vec<u8>,
            base_off: u64,
            len: usize,
        }
        impl<'a> VramStageCoalescer<'a> {
            fn new(buf: &'a VramBuffer, cap: usize) -> Self {
                Self {
                    buf,
                    stage: vec![0u8; cap],
                    base_off: 0,
                    len: 0,
                }
            }
            fn flush(&mut self, gpu: &mut NvGpu) -> Result<(), String> {
                if self.len > 0 {
                    gpu.upload_to_vram(self.buf, self.base_off, &self.stage[..self.len])?;
                    self.base_off += self.len as u64;
                    self.len = 0;
                }
                Ok(())
            }
            fn write_at(&mut self, gpu: &mut NvGpu, off: u64, data: &[u8]) -> Result<(), String> {
                if data.is_empty() {
                    return Ok(());
                }
                if self.len == 0 {
                    self.base_off = off;
                } else {
                    let cur_end = self.base_off + (self.len as u64);
                    if off >= cur_end
                        && (off - cur_end) <= 4096
                        && self.len + ((off - cur_end) as usize) <= self.stage.len()
                    {
                        let pad = (off - cur_end) as usize;
                        self.stage[self.len..self.len + pad].fill(0);
                        self.len += pad;
                    } else if off != cur_end {
                        self.flush(gpu)?;
                        self.base_off = off;
                    }
                }
                let mut pos = 0usize;
                while pos < data.len() {
                    if self.len == self.stage.len() {
                        self.flush(gpu)?;
                    }
                    let avail = self.stage.len() - self.len;
                    let take = (data.len() - pos).min(avail);
                    self.stage[self.len..self.len + take].copy_from_slice(&data[pos..pos + take]);
                    self.len += take;
                    pos += take;
                }
                Ok(())
            }
        }

        let mut coalescer = VramStageCoalescer::new(&weights_buf, CHUNK_MAX);
        let mut scratch = vec![0u8; 4 * 1024 * 1024];
        let t_dma = Instant::now();
        for name in &names_sorted {
            let info = gguf.req_tensor_info(name)?;
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

                let max_pay_blks = scratch.len() / pay_len;
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
                                scratch[d_pos + w * 4..d_pos + w * 4 + 4].copy_from_slice(&pv.to_le_bytes());
                            }
                        }
                    } else {
                        for bi in 0..take {
                            let s_off = (b_cur + bi) * gguf_blk + pay_off;
                            scratch[bi * pay_len..(bi + 1) * pay_len]
                                .copy_from_slice(&raw[s_off..s_off + pay_len]);
                        }
                    }
                    coalescer.write_at(
                        gpu,
                        dst_off + ((b_cur * pay_len) as u64),
                        &scratch[..take * pay_len],
                    )?;
                    b_cur += take;
                }

                let d_base_off = dst_off + ((tot_blks * pay_len) as u64);
                let max_d_blks = scratch.len() / 2;
                b_cur = 0;
                while b_cur < tot_blks {
                    let take = (tot_blks - b_cur).min(max_d_blks);
                    for bi in 0..take {
                        let s_off = (b_cur + bi) * gguf_blk + d_off;
                        scratch[bi * 2..bi * 2 + 2].copy_from_slice(&raw[s_off..s_off + 2]);
                    }
                    coalescer.write_at(
                        gpu,
                        d_base_off + ((b_cur * 2) as u64),
                        &scratch[..take * 2],
                    )?;
                    b_cur += take;
                }
            } else {
                coalescer.write_at(gpu, dst_off, raw)?;
            }
        }
        coalescer.flush(gpu)?;
        drop(scratch);
        drop(coalescer);
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
        gpu.zero_vram(&ws_buf)?;
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

        let (dspark, pre_dspark_cp) = if cfg.loop_passes == 3 {
            if let Some(mtp_path) = discover_mtp_safetensors(&model_path) {
                let cp = gpu.checkpoint_vram();
                match InkDSparkState::load(
                    gpu,
                    &module,
                    &mtp_path,
                    cfg.vocab_unpadded,
                    cfg.max_ctx,
                    output_w.va,
                    output_w.ggml_type,
                ) {
                    Ok(st) => (Some(st), Some(cp)),
                    Err(e) => {
                        gpu.restore_vram_checkpoint(cp);
                        eprintln!("[dspark-nv] Warning: failed to load {}: {} (falling back to B=1 persistent)", mtp_path.display(), e);
                        (None, None)
                    }
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
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
            pre_dspark_cp,
            dspark_gamma,
            dspark_cache_valid: false,
            cached_tokens: Vec::new(),
            cached_passes: loop_passes,
        })
    }

    pub fn reset_cache(&mut self, gpu: &mut NvGpu) -> Result<(), String> {
        gpu.zero_vram(&self.conv_state_buf)?;
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
    fn forward_batch(
        &mut self,
        gpu: &mut NvGpu,
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

        let k_bm8 = self.module.kernel("ink_batch_forward_bm8_megakernel")?;
        let err = gpu.launch_kernel_watched(
            k_bm8,
            (gpu.active_sms, 1, 1),
            (512, 1, 1),
            &mk_args,
            true,
            unsafe { &(*ctrl_ptr).error_code as *const i32 },
            unsafe { &mut (*ctrl_ptr).cancel_flag as *mut i32 },
            Duration::from_secs(30),
            "ink_batch_forward_bm8_megakernel",
        )?;
        if err != 0 {
            let _ = gpu.zero_vram(&self.ws_bar_state_buf);
            return Err(format!("ink_batch_forward_bm8_megakernel barrier watchdog tripped at {}", err));
        }
        let out = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).batch_out_toks) };
        Ok(out)
    }

    /// Restore `ShortConv` state from `conv_backup[slot_b]` (`~5 us` VRAM-to-VRAM copy) after speculative rejection.
    fn restore_conv_backup(&mut self, gpu: &mut NvGpu, slot_b: usize) -> Result<(), String> {
        assert!(slot_b < 8);
        let src_va = self.conv_backup_buf.va_addr + ((slot_b * self.conv_total_floats * 4) as u64);
        let mut args = [0u8; 20];
        put_u64(&mut args, 0, self.conv_state_buf.va_addr);
        put_u64(&mut args, 8, src_va);
        args[16..20].copy_from_slice(&(self.conv_total_floats as i32).to_le_bytes());

        let k_cp = self.module.kernel("ink_copy_f32_vram_kernel")?;
        gpu.launch_kernel_sync(k_cp, (gpu.active_sms, 1, 1), (512, 1, 1), &args, false, Duration::from_secs(5))
    }

    fn dspark_inject_kv(&mut self, gpu: &mut NvGpu, inject_pos: usize, inject_cnt: usize) -> Result<(), String> {
        let Some(ref d) = self.dspark else {
            return Ok(());
        };
        if inject_cnt == 0 {
            return Ok(());
        }
        self.done_seq_counter += 1;
        d.inject_kv(
            gpu,
            &self.module,
            self.model_desc_va,
            &self.ctrl_host,
            &self.ws_bar_state_buf,
            self.done_seq_counter,
            inject_pos,
            inject_cnt,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dspark_propose(
        &mut self,
        gpu: &mut NvGpu,
        anchor_tok: i32,
        anchor_pos: usize,
        bd_len: usize,
        req_gamma: usize,
        inject_pos: usize,
        inject_cnt: usize,
    ) -> Result<[i32; 8], String> {
        let d = self.dspark.as_ref().ok_or_else(|| "DSpark drafter is not loaded".to_string())?;
        self.done_seq_counter += 1;
        d.propose(
            gpu,
            &self.module,
            self.model_desc_va,
            &self.ctrl_host,
            &self.ws_bar_state_buf,
            self.done_seq_counter,
            anchor_tok,
            anchor_pos,
            bd_len,
            req_gamma,
            inject_pos,
            inject_cnt,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_cancel<F>(
        &mut self,
        gpu: &mut NvGpu,
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
            self.reset_cache(gpu)?;
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
                self.reset_cache(gpu)?;
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
            let out = self.forward_batch(gpu, 
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
                    self.dspark_inject_kv(gpu, pos, take)?;
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
                    self.dspark_inject_kv(gpu, pending_inject.0, pending_inject.1)?;
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
                        self.dspark_inject_kv(gpu, pending_inject.0, pending_inject.1)?;
                        launches += 1;
                    }
                    let out = self.forward_batch(gpu, 
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
                let drafts = self.dspark_propose(gpu, cur_tok, cur_pos, 8, k_ver, pending_inject.0, pending_inject.1)?;
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
                let verify_out = self.forward_batch(gpu, 
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
                    self.restore_conv_backup(gpu, commit_count - 1)?;
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
                self.dspark_inject_kv(gpu, pending_inject.0, pending_inject.1)?;
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

            let k_mk = self.module.kernel("ink_persistent_decode_megakernel")?;
            let sig_val = gpu.launch_persistent_async(k_mk, (gpu.active_sms, 1, 1), (512, 1, 1), &mk_args, true)?;
            launches += 1;
            let rollout_t0 = Instant::now();
            let mut streamed = 0usize;
            let mut prev_uncommitted = first_tok;
            let mut user_aborted = false;
            let mut eos_seen = false;
            let mut iter = 0u32;

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

                if gpu.is_ticket_done(sig_val) {
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
                    let _ = gpu.zero_vram(&self.ws_bar_state_buf);
                    self.cached_tokens.clear();
                    return Err(format!(
                        "ink_persistent_decode_megakernel barrier watchdog tripped at step {}",
                        err
                    ));
                }
                gpu.poll_persistent_maintenance(&mut iter, streamed == 0 && rollout_t0.elapsed() > Duration::from_millis(2))?;
                if (iter & 0x3fff) == 0 && rollout_t0.elapsed() > Duration::from_secs(180) {
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
        gpu: &mut NvGpu,
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
            gpu,
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
