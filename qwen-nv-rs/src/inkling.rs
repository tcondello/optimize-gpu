//! 100%-on-RTX-3080 (`GA102`, `sm_86`) Native Engine for Thinking Machines `Inkling-2-Mini`
//! (`inkling2_loop`: 26.4B Total / 3.4B–5.4B Active Looped Sparse MoE, 24 Physical Layers ->
//! 48 Logical Evaluations, 128 Experts Top-8 SITU, ShortConv, Relative-Position + Sink GQA,
//! and Depth Attention Residuals).
//!
//! Zero external Rust crates, zero PyTorch, zero runtime Docker containers (`AGENTS.md` compliant).

#![allow(dead_code)]

use std::path::Path;
use std::time::{Duration, Instant};

use crate::gguf::{resolve_model_path, BpeTokenizer, GgufFile, GgufVal, Utf8StreamDecoder};
use crate::nv_defs::{put_u32, put_u64, QmdV3};
use crate::nv_hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};

#[inline]
fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

pub static INKLING_CUBIN: &[u8] = include_bytes!("../kernels/inkling_sm86.cubin");

const TOK_END_SAMPLING: i32 = 200006; // <|content_model_end_sampling|>
const TOK_EOT: i32 = 199999;          // <|endoftext|>
const TOK_MSG_MODEL: i32 = 200001;    // <|message_model|>
const TOK_CONTENT_TEXT: i32 = 200004; // <|content_text|>
const TOK_CONTENT_THINK: i32 = 200008;// <|content_thinking|>
const TOK_END_MESSAGE: i32 = 200010;  // <|end_message|>

#[derive(Clone, Copy, Debug)]
struct WeightTensorGpu {
    va: u64,
    ggml_type: i32,
}

#[derive(Clone, Debug)]
struct InklingPhysLayerGpu {
    is_swa: bool,
    // Attention projections
    wq: WeightTensorGpu,
    wk: WeightTensorGpu,
    wv: WeightTensorGpu,
    wo: WeightTensorGpu,
    wqkv_gate: WeightTensorGpu,
    attn_q_clip: u64,
    attn_sink_proj: WeightTensorGpu,
    // Relative position
    attn_rel: WeightTensorGpu,
    attn_rel_bias: u64,
    attn_rel_clip: u64,
    // Short convolutions (`[4, C]` F32)
    attn_k_conv: u64,
    attn_v_conv: u64,
    attn_post_conv: u64,
    ffn_post_conv: u64,
    // Input norms & full-attn dynamic conv projections
    attn_norm: u64,
    attn_q_inp_norm: u64,
    attn_k_inp_norm: u64,
    attn_v_inp_norm: u64,
    attn_k_conv_proj: Option<WeightTensorGpu>,
    attn_v_conv_proj: Option<WeightTensorGpu>,
    // Depth attention residual queries (`None` on layer 0)
    attn_res_query: Option<u64>,
    ffn_res_query: u64,
    // Sparse MoE (`128` experts, top-`8`)
    ffn_norm: u64,
    ffn_gate_inp: WeightTensorGpu,
    exp_probs_b: u64,
    ffn_gate_up_exps: WeightTensorGpu,
    ffn_down_exps: WeightTensorGpu,
}

#[derive(Clone, Debug)]
struct InklingLoopCtrlGpu {
    msg_query: u64,                  // [2304, 4] F32
    ctrl_down: [WeightTensorGpu; 4], // Q, K, V, FFN: [4608, 64]
    ctrl_gain: [WeightTensorGpu; 4], // [64, 2304]
    ctrl_forget: [WeightTensorGpu; 4],// [64, 2304]
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
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub ttft_ms: f64,
    pub vram_weights_mb: f64,
    pub vram_kv_conv_mb: f64,
    pub token_ids: Vec<i32>,
}

struct WorkspaceAddrs {
    cur: u64,
    inp_sa: u64,
    ffn_inp: u64,
    prelude: u64,
    bank_out: u64,
    blocks: u64,          // [8, dim]
    prelude_blocks: u64,  // [8, dim]
    bank: u64,            // [8, dim]
    messages: u64,        // [4, dim]
    mixed: u64,           // [3, dim]
    inputs: [u64; 3],     // 3 x [dim]
    norm_in: [u64; 3],    // 3 x [dim]
    ctrl_cat: u64,        // [2 * dim]
    ctrl_gate: u64,       // [64]
    ctrl_gain: u64,       // [dim]
    ctrl_forget: u64,     // [dim]
    q_buf: u64,           // [3072]
    k_raw: u64,           // [1024]
    k_buf: u64,           // [1024]
    v_raw: u64,           // [1024]
    v_buf: u64,           // [1024]
    r_buf: u64,           // [384]
    sinks_buf: u64,       // [24]
    gate_raw: u64,        // [3072]
    dyn_k: u64,           // [32]
    dyn_v: u64,           // [32]
    attn_gated: u64,      // [3072]
    wo_out: u64,          // [dim]
    attn_out: u64,        // [dim]
    ffn_mixed: u64,       // [dim]
    ffn_ctrl: u64,        // [dim]
    ffn_normed: u64,      // [dim]
    router_logits: u64,   // [128]
    top_ids: u64,         // [8] i32
    top_weights: u64,     // [8] f32
    situ_buf: u64,        // [8 * 1152]
    moe_raw: u64,         // [dim]
    ffn_out: u64,         // [dim]
    logits: u64,          // [vocab_unpadded]
    ctrl_vram: u64,       // [64 B] InkStepCtrl latched into GDDR6X L2
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

struct VramQmdChain<'a> {
    arena: &'a mut Vec<u8>,
    base_va: u64,
    module: &'a CubinModule,
    head_va: Option<u64>,
    prev_off: Option<usize>,
    pub count: usize,
}

impl<'a> VramQmdChain<'a> {
    fn new(arena: &'a mut Vec<u8>, base_va: u64, module: &'a CubinModule) -> Self {
        let aligned = (arena.len() + 255) & !255;
        arena.resize(aligned, 0);
        Self {
            arena,
            base_va,
            module,
            head_va: None,
            prev_off: None,
            count: 0,
        }
    }

    fn push(&mut self, kname: &str, grid: (u32, u32, u32), block: (u16, u16, u16), args: &[u8]) {
        let kern = &self.module.kernels[kname];
        let cbuf_aligned = ((kern.cbuf0_size as usize) + 255) & !255;
        let slot_size = (cbuf_aligned + 256 + 255) & !255;

        let cbuf_off = (self.arena.len() + 255) & !255;
        self.arena.resize(cbuf_off + slot_size, 0);
        let qmd_off = cbuf_off + cbuf_aligned;

        let cbuf_va = self.base_va + (cbuf_off as u64);
        let qmd_va = self.base_va + (qmd_off as u64);
        assert_eq!(qmd_va >> 40, 0, "VRAM QMD VA exceeds 40-bit PCAS address range");

        let mut qmd = kern.qmd_template.clone();
        qmd.set_cbuf(0, cbuf_va, kern.cbuf0_size);
        qmd.set_launch_dims(grid, block);

        self.arena[cbuf_off..cbuf_off + kern.cbuf0_template.len()].copy_from_slice(&kern.cbuf0_template);
        put_u32(&mut self.arena, cbuf_off + 0x00, block.0 as u32);
        put_u32(&mut self.arena, cbuf_off + 0x04, block.1 as u32);
        put_u32(&mut self.arena, cbuf_off + 0x08, block.2 as u32);
        put_u32(&mut self.arena, cbuf_off + 0x0c, grid.0);
        put_u32(&mut self.arena, cbuf_off + 0x10, grid.1);
        put_u32(&mut self.arena, cbuf_off + 0x14, grid.2);
        assert!(0x160 + args.len() <= kern.cbuf0_template.len(), "Kernel args exceed cbuf0_size");
        self.arena[cbuf_off + 0x160..cbuf_off + 0x160 + args.len()].copy_from_slice(args);
        self.arena[qmd_off..qmd_off + 256].copy_from_slice(&qmd.bytes);

        if let Some(prev) = self.prev_off {
            let mut prev_qmd = QmdV3 {
                bytes: self.arena[prev..prev + 256].try_into().unwrap(),
            };
            prev_qmd.set_dependent_qmd0(qmd_va);
            self.arena[prev..prev + 256].copy_from_slice(&prev_qmd.bytes);
        } else {
            self.head_va = Some(qmd_va);
        }
        self.prev_off = Some(qmd_off);
        self.count += 1;
    }

    fn finish(self) -> u64 {
        self.head_va.expect("Empty VRAM QMD chain")
    }
}

pub struct InklingEngine {
    pub gpu: NvGpu,
    pub module: CubinModule,
    pub cfg: InklingConfig,
    pub tok: BpeTokenizer,
    weights_buf: VramBuffer,
    swa_k_buf: VramBuffer,
    swa_v_buf: VramBuffer,
    full_k_buf: VramBuffer,
    full_v_buf: VramBuffer,
    conv_state_buf: VramBuffer,
    ws_buf: VramBuffer,
    graphs_vram_buf: VramBuffer,
    ctrl_host: HostMappedBuffer,
    graph_heads: [[u64; 2]; 3], // [passes - 1][compute_logits as usize] -> head QMD VA in GDDR6X
    done_seq_counter: u64,
    tok_embd: WeightTensorGpu,
    tok_norm: u64,
    output_w: WeightTensorGpu,
    output_norm: u64,
    output_res_query: u64,
    layers: Vec<InklingPhysLayerGpu>,
    loop_ctrls: Vec<InklingLoopCtrlGpu>,
    slot_kv_index: Vec<(bool, usize)>, // logical slot 0..48 -> (is_swa, dense_idx)
    ws: WorkspaceAddrs,
    first_launch: bool,
    cur_pos: usize,
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

        // Compute deterministic 256-byte-aligned VRAM layout for all tensors in GGUF
        let mut names_sorted: Vec<String> = gguf.tensors.keys().cloned().collect();
        names_sorted.sort_by_key(|n| gguf.tensors[n].offset);

        let mut total_w_bytes: u64 = 0;
        let mut tensor_rel_off = std::collections::HashMap::<String, (u64, usize, i32)>::with_capacity(names_sorted.len());
        for name in &names_sorted {
            let info = &gguf.tensors[name];
            let nb = ggml_tensor_nbytes(&info.dims, info.ggml_type)?;
            let off = align_up(total_w_bytes, 256);
            tensor_rel_off.insert(name.clone(), (off, nb, info.ggml_type));
            total_w_bytes = off + (nb as u64);
        }

        let weights_buf = gpu.alloc_vram(total_w_bytes)?;
        let w_base_va = weights_buf.va_addr;

        // Stream directly from mmap -> GDDR6X in <=32 MiB DMA slices (`0` extra host RAM)
        const CHUNK_MAX: usize = 32 * 1024 * 1024;
        let t_dma = Instant::now();
        for name in &names_sorted {
            let info = &gguf.tensors[name];
            let (dst_off, nb, _) = tensor_rel_off[name];
            let raw = gguf.tensor_bytes(info, nb);
            let mut pos = 0usize;
            while pos < nb {
                let take = (nb - pos).min(CHUNK_MAX);
                gpu.upload_to_vram(&weights_buf, dst_off + (pos as u64), &raw[pos..pos + take])?;
                pos += take;
            }
        }
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

        let mut layers = Vec::with_capacity(n_phys);
        for i in 0..n_phys {
            let is_swa = swa_pattern[i];
            let (attn_norm, q_in_n, k_in_n, v_in_n, k_cp, v_cp) = if is_swa {
                (get_f32_va(&format!("blk.{}.attn_norm.weight", i))?, 0, 0, 0, None, None)
            } else {
                (
                    0,
                    get_f32_va(&format!("blk.{}.attn_q_inp_norm.weight", i))?,
                    get_f32_va(&format!("blk.{}.attn_k_inp_norm.weight", i))?,
                    get_f32_va(&format!("blk.{}.attn_v_inp_norm.weight", i))?,
                    Some(get_wt(&format!("blk.{}.attn_k_conv_proj.weight", i))?),
                    Some(get_wt(&format!("blk.{}.attn_v_conv_proj.weight", i))?),
                )
            };
            let attn_res_query = if i > 0 {
                Some(get_f32_va(&format!("blk.{}.attn_res_query.weight", i))?)
            } else {
                None
            };
            layers.push(InklingPhysLayerGpu {
                is_swa,
                wq: get_wt(&format!("blk.{}.attn_q.weight", i))?,
                wk: get_wt(&format!("blk.{}.attn_k.weight", i))?,
                wv: get_wt(&format!("blk.{}.attn_v.weight", i))?,
                wo: get_wt(&format!("blk.{}.attn_output.weight", i))?,
                wqkv_gate: get_wt(&format!("blk.{}.attn_gate.weight", i))?,
                attn_q_clip: get_f32_va(&format!("blk.{}.attn_q_clip.weight", i))?,
                attn_sink_proj: get_wt(&format!("blk.{}.attn_sink_proj.weight", i))?,
                attn_rel: get_wt(&format!("blk.{}.attn_rel.weight", i))?,
                attn_rel_bias: get_f32_va(&format!("blk.{}.attn_rel_bias.weight", i))?,
                attn_rel_clip: get_f32_va(&format!("blk.{}.attn_rel_clip.weight", i))?,
                attn_k_conv: get_f32_va(&format!("blk.{}.attn_k_conv.weight", i))?,
                attn_v_conv: get_f32_va(&format!("blk.{}.attn_v_conv.weight", i))?,
                attn_post_conv: get_f32_va(&format!("blk.{}.attn_post_conv.weight", i))?,
                ffn_post_conv: get_f32_va(&format!("blk.{}.ffn_post_conv.weight", i))?,
                attn_norm,
                attn_q_inp_norm: q_in_n,
                attn_k_inp_norm: k_in_n,
                attn_v_inp_norm: v_in_n,
                attn_k_conv_proj: k_cp,
                attn_v_conv_proj: v_cp,
                attn_res_query,
                ffn_res_query: get_f32_va(&format!("blk.{}.ffn_res_query.weight", i))?,
                ffn_norm: get_f32_va(&format!("blk.{}.ffn_norm.weight", i))?,
                ffn_gate_inp: get_wt(&format!("blk.{}.ffn_gate_inp.weight", i))?,
                exp_probs_b: get_f32_va(&format!("blk.{}.exp_probs_b.bias", i))?,
                ffn_gate_up_exps: get_wt(&format!("blk.{}.ffn_gate_up_exps.weight", i))?,
                ffn_down_exps: get_wt(&format!("blk.{}.ffn_down_exps.weight", i))?,
            });
        }

        let mut loop_ctrls = Vec::with_capacity(loop_end - loop_start);
        let ctrl_names = ["q", "k", "v", "ffn"];
        for il in loop_start..loop_end {
            let msg_query = get_f32_va(&format!("blk.{}.loop_msg_query.weight", il))?;
            let mut down = [WeightTensorGpu { va: 0, ggml_type: 0 }; 4];
            let mut gain = [WeightTensorGpu { va: 0, ggml_type: 0 }; 4];
            let mut forget = [WeightTensorGpu { va: 0, ggml_type: 0 }; 4];
            for (c, tag) in ctrl_names.iter().enumerate() {
                down[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_down.weight", il, tag))?;
                gain[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_gain.weight", il, tag))?;
                forget[c] = get_wt(&format!("blk.{}.loop_{}_ctrl_forget.weight", il, tag))?;
            }
            loop_ctrls.push(InklingLoopCtrlGpu {
                msg_query,
                ctrl_down: down,
                ctrl_gain: gain,
                ctrl_forget: forget,
            });
        }

        // Map all logical slots (`n_phys + (loop_passes - 1)*(loop_end - loop_start)` = 48)
        let n_slots = n_phys + (loop_passes.saturating_sub(1)) * (loop_end - loop_start);
        let mut slot_kv_index = Vec::with_capacity(n_slots);
        let mut n_swa_slots = 0usize;
        let mut n_full_slots = 0usize;
        for slot in 0..n_slots {
            let phys = if slot < n_phys {
                slot
            } else {
                loop_start + (slot - n_phys) % (loop_end - loop_start)
            };
            if layers[phys].is_swa {
                slot_kv_index.push((true, n_swa_slots));
                n_swa_slots += 1;
            } else {
                slot_kv_index.push((false, n_full_slots));
                n_full_slots += 1;
            }
        }

        let kv_dim = n_kv_heads * head_dim; // 1024
        let swa_slot_bytes = (cfg.swa_window * kv_dim * 2) as u64;
        let full_slot_bytes = (cfg.max_ctx * kv_dim * 2) as u64;
        let swa_k_buf = gpu.alloc_vram((n_swa_slots as u64).max(1) * swa_slot_bytes)?;
        let swa_v_buf = gpu.alloc_vram((n_swa_slots as u64).max(1) * swa_slot_bytes)?;
        let full_k_buf = gpu.alloc_vram((n_full_slots as u64).max(1) * full_slot_bytes)?;
        let full_v_buf = gpu.alloc_vram((n_full_slots as u64).max(1) * full_slot_bytes)?;

        let conv_channels_per_slot = 2 * kv_dim + 2 * dim; // 6656
        let conv_slot_bytes = (conv_channels_per_slot * (shortconv_len - 1) * 4) as u64;
        let conv_total_bytes = (n_slots as u64) * conv_slot_bytes;
        let conv_state_buf = gpu.alloc_vram(conv_total_bytes)?;
        let zero_conv = vec![0u8; conv_total_bytes as usize];
        gpu.upload_to_vram(&conv_state_buf, 0, &zero_conv)?;

        // Workspace layout
        let mut w_off = 0u64;
        let mut alloc_ws = |n_bytes: usize| -> u64 {
            let o = align_up(w_off, 256);
            w_off = o + (n_bytes as u64);
            o
        };
        let o_cur = alloc_ws(dim * 4);
        let o_inp_sa = alloc_ws(dim * 4);
        let o_ffn_inp = alloc_ws(dim * 4);
        let o_prelude = alloc_ws(dim * 4);
        let o_bank_out = alloc_ws(dim * 4);
        let o_blocks = alloc_ws(8 * dim * 4);
        let o_prelude_blocks = alloc_ws(8 * dim * 4);
        let o_bank = alloc_ws(8 * dim * 4);
        let o_messages = alloc_ws(4 * dim * 4);
        let o_mixed = alloc_ws(3 * dim * 4);
        let o_in0 = alloc_ws(dim * 4);
        let o_in1 = alloc_ws(dim * 4);
        let o_in2 = alloc_ws(dim * 4);
        let o_nm0 = alloc_ws(dim * 4);
        let o_nm1 = alloc_ws(dim * 4);
        let o_nm2 = alloc_ws(dim * 4);
        let o_ctrl_cat = alloc_ws(2 * dim * 4);
        let o_ctrl_gate = alloc_ws(gate_rank * 4);
        let o_ctrl_gain = alloc_ws(dim * 4);
        let o_ctrl_forget = alloc_ws(dim * 4);
        let o_q = alloc_ws(n_heads * head_dim * 4);
        let o_k_raw = alloc_ws(kv_dim * 4);
        let o_k = alloc_ws(kv_dim * 4);
        let o_v_raw = alloc_ws(kv_dim * 4);
        let o_v = alloc_ws(kv_dim * 4);
        let o_r = alloc_ws(n_heads * rel_len * 4);
        let o_sinks = alloc_ws(n_heads * 4);
        let o_gate_raw = alloc_ws(n_heads * head_dim * 4);
        let o_dyn_k = alloc_ws(shortconv_len * n_kv_heads * 4);
        let o_dyn_v = alloc_ws(shortconv_len * n_kv_heads * 4);
        let o_attn_gated = alloc_ws(n_heads * head_dim * 4);
        let o_wo_out = alloc_ws(dim * 4);
        let o_attn_out = alloc_ws(dim * 4);
        let o_ffn_mixed = alloc_ws(dim * 4);
        let o_ffn_ctrl = alloc_ws(dim * 4);
        let o_ffn_normed = alloc_ws(dim * 4);
        let o_router_logits = alloc_ws(n_experts * 4);
        let o_top_ids = alloc_ws(n_experts_used * 4);
        let o_top_weights = alloc_ws(n_experts_used * 4);
        let o_situ = alloc_ws(n_experts_used * n_ff_exp * 4);
        let o_moe_raw = alloc_ws(dim * 4);
        let o_ffn_out = alloc_ws(dim * 4);
        let o_logits = alloc_ws(vocab_unpadded * 4);
        let o_ctrl_vram = alloc_ws(256);

        let ws_buf = gpu.alloc_vram(w_off)?;
        let wb = ws_buf.va_addr;
        let ws = WorkspaceAddrs {
            cur: wb + o_cur,
            inp_sa: wb + o_inp_sa,
            ffn_inp: wb + o_ffn_inp,
            prelude: wb + o_prelude,
            bank_out: wb + o_bank_out,
            blocks: wb + o_blocks,
            prelude_blocks: wb + o_prelude_blocks,
            bank: wb + o_bank,
            messages: wb + o_messages,
            mixed: wb + o_mixed,
            inputs: [wb + o_in0, wb + o_in1, wb + o_in2],
            norm_in: [wb + o_nm0, wb + o_nm1, wb + o_nm2],
            ctrl_cat: wb + o_ctrl_cat,
            ctrl_gate: wb + o_ctrl_gate,
            ctrl_gain: wb + o_ctrl_gain,
            ctrl_forget: wb + o_ctrl_forget,
            q_buf: wb + o_q,
            k_raw: wb + o_k_raw,
            k_buf: wb + o_k,
            v_raw: wb + o_v_raw,
            v_buf: wb + o_v,
            r_buf: wb + o_r,
            sinks_buf: wb + o_sinks,
            gate_raw: wb + o_gate_raw,
            dyn_k: wb + o_dyn_k,
            dyn_v: wb + o_dyn_v,
            attn_gated: wb + o_attn_gated,
            wo_out: wb + o_wo_out,
            attn_out: wb + o_attn_out,
            ffn_mixed: wb + o_ffn_mixed,
            ffn_ctrl: wb + o_ffn_ctrl,
            ffn_normed: wb + o_ffn_normed,
            router_logits: wb + o_router_logits,
            top_ids: wb + o_top_ids,
            top_weights: wb + o_top_weights,
            situ_buf: wb + o_situ,
            moe_raw: wb + o_moe_raw,
            ffn_out: wb + o_ffn_out,
            logits: wb + o_logits,
            ctrl_vram: wb + o_ctrl_vram,
        };

        // Host-mapped step control block (`[0..32]` = InkStepCtrl, `[64..68]` = sampled token, `[128..136]` = done_seq)
        let mut ctrl_host = gpu.alloc_host_mapped(256)?;
        ctrl_host.as_mut_slice().fill(0);

        // Pre-compile all 6 hardware-chained `QMDv3` graphs (`passes=1..=3` x `logits=false|true`)
        // directly into GDDR6X VRAM so runtime decode has ZERO QMD rebuilds & ZERO Thunderbolt QMD fetches!
        let graphs_cap = 8u64 << 20;
        let graphs_vram_buf = gpu.alloc_vram(graphs_cap)?;
        let mut graphs_arena = Vec::with_capacity(graphs_cap as usize);
        let mut graph_heads = [[0u64; 2]; 3];
        for p_idx in 0..3usize {
            let p_cnt = (p_idx + 1).min(cfg.loop_passes);
            for c_idx in 0..2usize {
                graph_heads[p_idx][c_idx] = Self::build_vram_graph(
                    &mut graphs_arena,
                    graphs_vram_buf.va_addr,
                    &module,
                    &cfg,
                    &layers,
                    &loop_ctrls,
                    &slot_kv_index,
                    &ws,
                    tok_embd,
                    tok_norm,
                    output_w,
                    output_norm,
                    output_res_query,
                    swa_k_buf.va_addr,
                    swa_v_buf.va_addr,
                    full_k_buf.va_addr,
                    full_v_buf.va_addr,
                    conv_state_buf.va_addr,
                    ctrl_host.va_addr,
                    p_cnt,
                    c_idx == 1,
                );
            }
        }
        assert!(
            (graphs_arena.len() as u64) <= graphs_cap,
            "VRAM QMD graph arena overflow ({} > {})",
            graphs_arena.len(),
            graphs_cap
        );
        gpu.upload_to_vram(&graphs_vram_buf, 0, &graphs_arena)?;

        let kv_mb = ((swa_k_buf.size + swa_v_buf.size + full_k_buf.size + full_v_buf.size + conv_state_buf.size) as f64)
            / (1024.0 * 1024.0);
        eprintln!(
            "[inkling-nv] Loaded '{}' ({:.2} GiB weights @ {:.2} GB/s DMA, {} phys -> {} logical layers, {}x{} MoE, KV+Conv={:.1} MiB, VRAM-QMD={:.2} MiB, ctx={}) in {:.2}s",
            cfg.model_name,
            (total_w_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
            dma_gbps,
            cfg.n_phys_layers,
            n_slots,
            cfg.n_experts,
            cfg.n_experts_used,
            kv_mb,
            (graphs_arena.len() as f64) / (1024.0 * 1024.0),
            cfg.max_ctx,
            t_load.elapsed().as_secs_f64()
        );

        Ok(Self {
            gpu,
            module,
            cfg,
            tok,
            weights_buf,
            swa_k_buf,
            swa_v_buf,
            full_k_buf,
            full_v_buf,
            conv_state_buf,
            ws_buf,
            graphs_vram_buf,
            ctrl_host,
            graph_heads,
            done_seq_counter: 0,
            tok_embd,
            tok_norm,
            output_w,
            output_norm,
            output_res_query,
            layers,
            loop_ctrls,
            slot_kv_index,
            ws,
            first_launch: true,
            cur_pos: 0,
        })
    }

    pub fn reset_cache(&mut self) -> Result<(), String> {
        let zero_conv = vec![0u8; self.conv_state_buf.size as usize];
        self.gpu.upload_to_vram(&self.conv_state_buf, 0, &zero_conv)?;
        self.cur_pos = 0;
        Ok(())
    }

    pub fn format_chat_prompt(&self, user_prompt: &str, sys_prompt: Option<&str>, effort_override: Option<&str>) -> Vec<i32> {
        let user_no_think = user_prompt.contains("/no_think");
        let clean_user = user_prompt.replace("/no_think", "");
        let clean_user = clean_user.trim();
        let raw_eff = effort_override.unwrap_or(if user_no_think { "0" } else { "0.2" }).trim();
        let eff = match raw_eff {
            "none" | "0.0" => "0",
            "minimal" => "0.1",
            "low" => "0.2",
            "medium" => "0.7",
            "high" => "0.9",
            "xhigh" | "max" => "0.99",
            other => other,
        };

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

    fn enqueue_vec_op(chain: &mut VramQmdChain, dst: u64, a: u64, b: u64, n: usize) {
        let mut args = [0u8; 28];
        put_u64(&mut args, 0, dst);
        put_u64(&mut args, 8, a);
        put_u64(&mut args, 16, b);
        put_u32(&mut args, 24, n as u32);
        let blocks = ((n as u32) + 255) / 256;
        chain.push("ink_vec_op_kernel", (blocks, 1, 1), (256, 1, 1), &args);
    }

    fn enqueue_rmsnorm(chain: &mut VramQmdChain, out: u64, inp: u64, norm_w: u64, dim: usize, eps: f32) {
        let mut args = [0u8; 32];
        put_u64(&mut args, 0, out);
        put_u64(&mut args, 8, inp);
        put_u64(&mut args, 16, norm_w);
        put_u32(&mut args, 24, dim as u32);
        put_u32(&mut args, 28, eps.to_bits());
        chain.push("ink_rmsnorm_kernel", (1, 1, 1), (256, 1, 1), &args);
    }

    fn enqueue_gemv(chain: &mut VramQmdChain, y: u64, w: WeightTensorGpu, x: u64, k_cols: usize, n_rows: usize, softcap: f32) {
        let mut args = [0u8; 40];
        put_u64(&mut args, 0, y);
        put_u64(&mut args, 8, w.va);
        put_u64(&mut args, 16, x);
        put_u32(&mut args, 24, k_cols as u32);
        put_u32(&mut args, 28, n_rows as u32);
        put_u32(&mut args, 32, w.ggml_type as u32);
        put_u32(&mut args, 36, softcap.to_bits());
        let blocks = ((n_rows as u32) + 7) / 8;
        chain.push("ink_gemv_dense_kernel", (blocks, 1, 1), (256, 1, 1), &args);
    }

    fn enqueue_depth_mix(
        chain: &mut VramQmdChain,
        out: u64,
        blocks_va: u64,
        cur: u64,
        query: u64,
        dim: usize,
        n_blocks: usize,
        n_queries: usize,
    ) {
        let mut args = [0u8; 48];
        put_u64(&mut args, 0, out);
        put_u64(&mut args, 8, blocks_va);
        put_u64(&mut args, 16, cur);
        put_u64(&mut args, 24, query);
        put_u32(&mut args, 32, dim as u32);
        put_u32(&mut args, 36, n_blocks as u32);
        put_u32(&mut args, 40, n_queries as u32);
        put_u32(&mut args, 44, 1e-6f32.to_bits());
        chain.push("ink_depth_mix_kernel", (1, 1, 1), (256, 1, 1), &args);
    }

    fn enqueue_loop_ctrl(
        chain: &mut VramQmdChain,
        ws: &WorkspaceAddrs,
        out: u64,
        cur_in: u64,
        msg_in: u64,
        loop_ctrl: &InklingLoopCtrlGpu,
        ctrl_idx: usize,
        dim: usize,
        gate_rank: usize,
    ) {
        let mut prep = [0u8; 32];
        put_u64(&mut prep, 0, ws.ctrl_cat);
        put_u64(&mut prep, 8, cur_in);
        put_u64(&mut prep, 16, msg_in);
        put_u32(&mut prep, 24, dim as u32);
        put_u32(&mut prep, 28, 1e-6f32.to_bits());
        chain.push("ink_loop_ctrl_prep_kernel", (1, 1, 1), (256, 1, 1), &prep);

        Self::enqueue_gemv(chain, ws.ctrl_gate, loop_ctrl.ctrl_down[ctrl_idx], ws.ctrl_cat, 2 * dim, gate_rank, 0.0);
        Self::enqueue_gemv(chain, ws.ctrl_gain, loop_ctrl.ctrl_gain[ctrl_idx], ws.ctrl_gate, gate_rank, dim, 0.0);
        Self::enqueue_gemv(chain, ws.ctrl_forget, loop_ctrl.ctrl_forget[ctrl_idx], ws.ctrl_gate, gate_rank, dim, 0.0);

        let mut app = [0u8; 44];
        put_u64(&mut app, 0, out);
        put_u64(&mut app, 8, cur_in);
        put_u64(&mut app, 16, msg_in);
        put_u64(&mut app, 24, ws.ctrl_gain);
        put_u64(&mut app, 32, ws.ctrl_forget);
        put_u32(&mut app, 40, dim as u32);
        let blocks = ((dim as u32) + 255) / 256;
        chain.push("ink_loop_ctrl_apply_kernel", (blocks, 1, 1), (256, 1, 1), &app);
    }

    fn enqueue_shortconv(
        chain: &mut VramQmdChain,
        out: u64,
        cur_in: u64,
        conv_w: u64,
        dyn_in: u64,
        conv_state: u64,
        n_channels: usize,
        n_heads_dyn: usize,
    ) {
        let mut args = [0u8; 48];
        put_u64(&mut args, 0, out);
        put_u64(&mut args, 8, cur_in);
        put_u64(&mut args, 16, conv_w);
        put_u64(&mut args, 24, dyn_in);
        put_u64(&mut args, 32, conv_state);
        put_u32(&mut args, 40, n_channels as u32);
        put_u32(&mut args, 44, n_heads_dyn as u32);
        let blocks = ((n_channels as u32) + 255) / 256;
        chain.push("ink_shortconv_kernel", (blocks, 1, 1), (256, 1, 1), &args);
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_layer(
        cfg: &InklingConfig,
        layers: &[InklingPhysLayerGpu],
        loop_ctrls: &[InklingLoopCtrlGpu],
        slot_kv_index: &[(bool, usize)],
        ws: &WorkspaceAddrs,
        swa_k_base: u64,
        swa_v_base: u64,
        full_k_base: u64,
        full_v_base: u64,
        conv_base: u64,
        chain: &mut VramQmdChain,
        phys_l: usize,
        ip: usize,
        cache_slot: usize,
        n_blocks: &mut usize,
        has_loop_msgs: bool,
    ) {
        let dim = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;    // 3072
        let kv_dim = cfg.n_kv_heads * cfg.head_dim; // 1024
        let r_dim = cfg.n_heads * cfg.rel_len;      // 384
        let layer = &layers[phys_l];
        let is_swa = layer.is_swa;

        let conv_channels_per_slot = 2 * kv_dim + 2 * dim; // 6656
        let slot_conv_va = conv_base + ((cache_slot * conv_channels_per_slot * 3 * 4) as u64);
        let k_conv_state = slot_conv_va;
        let v_conv_state = slot_conv_va + ((kv_dim * 3 * 4) as u64);
        let attn_post_conv_state = slot_conv_va + ((2 * kv_dim * 3 * 4) as u64);
        let ffn_post_conv_state = slot_conv_va + (((2 * kv_dim + dim) * 3 * 4) as u64);

        // Save `inpSA = cur`
        Self::enqueue_vec_op(chain, ws.inp_sa, ws.cur, 0, dim);

        // 1. Depth attention residual mix (`inputs[0..3]`)
        let raw_inputs: [u64; 3] = if let Some(arq) = layer.attn_res_query {
            let n_q = if is_swa { 1 } else { 3 };
            Self::enqueue_depth_mix(chain, ws.mixed, ws.blocks, ws.cur, arq, dim, *n_blocks, n_q);
            if is_swa {
                [ws.mixed, ws.mixed, ws.mixed]
            } else {
                [
                    ws.mixed,
                    ws.mixed + ((dim * 4) as u64),
                    ws.mixed + ((2 * dim * 4) as u64),
                ]
            }
        } else {
            [ws.cur, ws.cur, ws.cur]
        };

        // 2. Optional cross-pass loop controller on Q, K, V inputs + input RMSNorm
        let normed_inputs: [u64; 3] = if has_loop_msgs {
            let lc = &loop_ctrls[ip - cfg.loop_start];
            for j in 0..3 {
                let msg_j = ws.messages + ((j * dim * 4) as u64);
                Self::enqueue_loop_ctrl(chain, ws, ws.inputs[j], raw_inputs[j], msg_j, lc, j, dim, cfg.gate_rank);
                let nw = if is_swa {
                    layer.attn_norm
                } else if j == 0 {
                    layer.attn_q_inp_norm
                } else if j == 1 {
                    layer.attn_k_inp_norm
                } else {
                    layer.attn_v_inp_norm
                };
                Self::enqueue_rmsnorm(chain, ws.norm_in[j], ws.inputs[j], nw, dim, cfg.rms_eps);
            }
            ws.norm_in
        } else if is_swa {
            Self::enqueue_rmsnorm(chain, ws.norm_in[0], raw_inputs[0], layer.attn_norm, dim, cfg.rms_eps);
            [ws.norm_in[0], ws.norm_in[0], ws.norm_in[0]]
        } else {
            Self::enqueue_rmsnorm(chain, ws.norm_in[0], raw_inputs[0], layer.attn_q_inp_norm, dim, cfg.rms_eps);
            Self::enqueue_rmsnorm(chain, ws.norm_in[1], raw_inputs[1], layer.attn_k_inp_norm, dim, cfg.rms_eps);
            Self::enqueue_rmsnorm(chain, ws.norm_in[2], raw_inputs[2], layer.attn_v_inp_norm, dim, cfg.rms_eps);
            ws.norm_in
        };

        // 3. Self-Attention projections + K/V ShortConv
        Self::enqueue_gemv(chain, ws.q_buf, layer.wq, normed_inputs[0], dim, q_dim, 0.0);
        Self::enqueue_gemv(chain, ws.k_raw, layer.wk, normed_inputs[1], dim, kv_dim, 0.0);
        Self::enqueue_gemv(chain, ws.v_raw, layer.wv, normed_inputs[2], dim, kv_dim, 0.0);
        Self::enqueue_gemv(chain, ws.r_buf, layer.attn_rel, normed_inputs[0], dim, r_dim, 0.0);
        Self::enqueue_gemv(chain, ws.sinks_buf, layer.attn_sink_proj, normed_inputs[0], dim, cfg.n_heads, 0.0);
        Self::enqueue_gemv(chain, ws.gate_raw, layer.wqkv_gate, normed_inputs[0], dim, q_dim, 0.0);

        if is_swa {
            Self::enqueue_shortconv(chain, ws.k_buf, ws.k_raw, layer.attn_k_conv, 0, k_conv_state, kv_dim, 0);
            Self::enqueue_shortconv(chain, ws.v_buf, ws.v_raw, layer.attn_v_conv, 0, v_conv_state, kv_dim, 0);
        } else {
            let n_dyn = cfg.shortconv_len * cfg.n_kv_heads; // 32
            Self::enqueue_gemv(chain, ws.dyn_k, layer.attn_k_conv_proj.unwrap(), normed_inputs[1], dim, n_dyn, 0.0);
            Self::enqueue_gemv(chain, ws.dyn_v, layer.attn_v_conv_proj.unwrap(), normed_inputs[2], dim, n_dyn, 0.0);
            Self::enqueue_shortconv(chain, ws.k_buf, ws.k_raw, layer.attn_k_conv, ws.dyn_k, k_conv_state, kv_dim, cfg.n_kv_heads);
            Self::enqueue_shortconv(chain, ws.v_buf, ws.v_raw, layer.attn_v_conv, ws.dyn_v, v_conv_state, kv_dim, cfg.n_kv_heads);
        }

        // 4. Q/K/V/R Clip + RMSNorm + KV Cache Store (reads `pos` from GDDR6X L2 `ws.ctrl_vram`)
        let (slot_swa, dense_idx) = slot_kv_index[cache_slot];
        debug_assert_eq!(slot_swa, is_swa);
        let (k_slot_va, v_slot_va) = if is_swa {
            let stride = (cfg.swa_window * kv_dim * 2) as u64;
            (
                swa_k_base + (dense_idx as u64) * stride,
                swa_v_base + (dense_idx as u64) * stride,
            )
        } else {
            let stride = (cfg.max_ctx * kv_dim * 2) as u64;
            (
                full_k_base + (dense_idx as u64) * stride,
                full_v_base + (dense_idx as u64) * stride,
            )
        };

        {
            let mut args = [0u8; 100];
            put_u64(&mut args, 0, ws.q_buf);
            put_u64(&mut args, 8, ws.r_buf);
            put_u64(&mut args, 16, ws.k_buf);
            put_u64(&mut args, 24, ws.v_buf);
            put_u64(&mut args, 32, layer.attn_q_clip);
            put_u64(&mut args, 40, layer.attn_rel_clip);
            put_u64(&mut args, 48, k_slot_va);
            put_u64(&mut args, 56, v_slot_va);
            put_u64(&mut args, 64, ws.ctrl_vram);
            put_u32(&mut args, 72, cfg.swa_window as u32);
            put_u32(&mut args, 76, if is_swa { 1 } else { 0 });
            put_u32(&mut args, 80, cfg.qk_clip.to_bits());
            put_u32(&mut args, 84, cfg.rel_clip.to_bits());
            put_u32(&mut args, 88, cfg.rms_eps.to_bits());
            put_u32(&mut args, 92, cfg.log_scale_floor as u32);
            put_u32(&mut args, 96, cfg.log_scale_alpha.to_bits());
            chain.push("ink_qkv_prep_kvstore_kernel", (40, 1, 1), (128, 1, 1), &args);
        }

        // 5. Relative-Position + Virtual Zero-Sink GQA Attention + Sigmoid Gate + `wo` + `attn_post_conv`
        {
            let mut args = [0u8; 84];
            put_u64(&mut args, 0, ws.attn_gated);
            put_u64(&mut args, 8, ws.q_buf);
            put_u64(&mut args, 16, ws.r_buf);
            put_u64(&mut args, 24, ws.sinks_buf);
            put_u64(&mut args, 32, ws.gate_raw);
            put_u64(&mut args, 40, layer.attn_rel_bias);
            put_u64(&mut args, 48, k_slot_va);
            put_u64(&mut args, 56, v_slot_va);
            put_u64(&mut args, 64, ws.ctrl_vram);
            put_u32(&mut args, 72, if is_swa { 1 } else { 0 });
            put_u32(&mut args, 76, cfg.swa_window as u32);
            put_u32(&mut args, 80, cfg.rel_extent as u32);
            chain.push("ink_rel_sink_attn_gate_kernel", (cfg.n_heads as u32, 1, 1), (128, 1, 1), &args);
        }
        Self::enqueue_gemv(chain, ws.wo_out, layer.wo, ws.attn_gated, q_dim, dim, 0.0);
        Self::enqueue_shortconv(chain, ws.attn_out, ws.wo_out, layer.attn_post_conv, 0, attn_post_conv_state, dim, 0);

        // 6. Block Attention Residual boundary (`ip % 6 == 0`)
        if ip % cfg.attn_res_block == 0 {
            let blk_dst = ws.blocks + ((*n_blocks * dim * 4) as u64);
            Self::enqueue_vec_op(chain, blk_dst, ws.inp_sa, 0, dim);
            *n_blocks += 1;
            Self::enqueue_vec_op(chain, ws.cur, ws.attn_out, 0, dim);
        } else {
            Self::enqueue_vec_op(chain, ws.cur, ws.inp_sa, ws.attn_out, dim);
        }
        Self::enqueue_vec_op(chain, ws.ffn_inp, ws.cur, 0, dim);

        // 7. Sparse MoE (`128` experts, top-`8` Sigmoid + `SITU`)
        Self::enqueue_depth_mix(chain, ws.ffn_mixed, ws.blocks, ws.cur, layer.ffn_res_query, dim, *n_blocks, 1);
        let ffn_pre_norm = if has_loop_msgs {
            let lc = &loop_ctrls[ip - cfg.loop_start];
            let msg_ffn = ws.messages + ((3 * dim * 4) as u64);
            Self::enqueue_loop_ctrl(chain, ws, ws.ffn_ctrl, ws.ffn_mixed, msg_ffn, lc, 3, dim, cfg.gate_rank);
            ws.ffn_ctrl
        } else {
            ws.ffn_mixed
        };
        Self::enqueue_rmsnorm(chain, ws.ffn_normed, ffn_pre_norm, layer.ffn_norm, dim, cfg.rms_eps);
        Self::enqueue_gemv(chain, ws.router_logits, layer.ffn_gate_inp, ws.ffn_normed, dim, cfg.n_experts, 0.0);

        {
            let mut r_args = [0u8; 36];
            put_u64(&mut r_args, 0, ws.top_ids);
            put_u64(&mut r_args, 8, ws.top_weights);
            put_u64(&mut r_args, 16, ws.router_logits);
            put_u64(&mut r_args, 24, layer.exp_probs_b);
            put_u32(&mut r_args, 32, cfg.expert_weights_scale.to_bits());
            chain.push("ink_moe_router_topk8_kernel", (1, 1, 1), (128, 1, 1), &r_args);
        }
        {
            let mut gu_args = [0u8; 52];
            put_u64(&mut gu_args, 0, ws.situ_buf);
            put_u64(&mut gu_args, 8, layer.ffn_gate_up_exps.va);
            put_u64(&mut gu_args, 16, ws.ffn_normed);
            put_u64(&mut gu_args, 24, ws.top_ids);
            put_u32(&mut gu_args, 32, dim as u32);
            put_u32(&mut gu_args, 36, cfg.n_ff_exp as u32);
            put_u32(&mut gu_args, 40, layer.ffn_gate_up_exps.ggml_type as u32);
            put_u32(&mut gu_args, 44, cfg.situ_beta.to_bits());
            put_u32(&mut gu_args, 48, cfg.situ_lb.to_bits());
            let grid_x = ((cfg.n_ff_exp as u32) + 7) / 8;
            chain.push("ink_moe_gate_up_situ_kernel", (grid_x, cfg.n_experts_used as u32, 1), (256, 1, 1), &gu_args);
        }
        {
            let mut dn_args = [0u8; 52];
            put_u64(&mut dn_args, 0, ws.moe_raw);
            put_u64(&mut dn_args, 8, layer.ffn_down_exps.va);
            put_u64(&mut dn_args, 16, ws.situ_buf);
            put_u64(&mut dn_args, 24, ws.top_ids);
            put_u64(&mut dn_args, 32, ws.top_weights);
            put_u32(&mut dn_args, 40, cfg.n_ff_exp as u32);
            put_u32(&mut dn_args, 44, dim as u32);
            put_u32(&mut dn_args, 48, layer.ffn_down_exps.ggml_type as u32);
            let grid_x = ((dim as u32) + 7) / 8;
            chain.push("ink_moe_down_accum_kernel", (grid_x, 1, 1), (256, 1, 1), &dn_args);
        }

        Self::enqueue_shortconv(chain, ws.ffn_out, ws.moe_raw, layer.ffn_post_conv, 0, ffn_post_conv_state, dim, 0);
        Self::enqueue_vec_op(chain, ws.cur, ws.ffn_inp, ws.ffn_out, dim);
    }

    #[allow(clippy::too_many_arguments)]
    fn build_vram_graph(
        arena: &mut Vec<u8>,
        base_va: u64,
        module: &CubinModule,
        cfg: &InklingConfig,
        layers: &[InklingPhysLayerGpu],
        loop_ctrls: &[InklingLoopCtrlGpu],
        slot_kv_index: &[(bool, usize)],
        ws: &WorkspaceAddrs,
        tok_embd: WeightTensorGpu,
        tok_norm: u64,
        output_w: WeightTensorGpu,
        output_norm: u64,
        output_res_query: u64,
        swa_k_base: u64,
        swa_v_base: u64,
        full_k_base: u64,
        full_v_base: u64,
        conv_base: u64,
        ctrl_host_va: u64,
        eff_passes: usize,
        compute_logits: bool,
    ) -> u64 {
        let dim = cfg.dim;
        let start = cfg.loop_start;
        let end = cfg.loop_end;
        let n_phys = cfg.n_phys_layers;

        let mut chain = VramQmdChain::new(arena, base_va, module);

        // 1. Latch `InkStepCtrl` into GDDR6X L2 + Token embedding + `tok_norm`
        {
            let mut e_args = [0u8; 52];
            put_u64(&mut e_args, 0, ws.cur);
            put_u64(&mut e_args, 8, ws.ctrl_vram);
            put_u64(&mut e_args, 16, ctrl_host_va);
            put_u64(&mut e_args, 24, tok_embd.va);
            put_u64(&mut e_args, 32, tok_norm);
            put_u32(&mut e_args, 40, dim as u32);
            put_u32(&mut e_args, 44, tok_embd.ggml_type as u32);
            put_u32(&mut e_args, 48, cfg.rms_eps.to_bits());
            chain.push("ink_embed_rmsnorm_kernel", (1, 1, 1), (256, 1, 1), &e_args);
        }

        let mut n_blocks = 0usize;

        // 2. Prelude layers `0 .. loop_start`
        for il in 0..start {
            Self::enqueue_layer(
                cfg,
                layers,
                loop_ctrls,
                slot_kv_index,
                ws,
                swa_k_base,
                swa_v_base,
                full_k_base,
                full_v_base,
                conv_base,
                &mut chain,
                il,
                il,
                il,
                &mut n_blocks,
                false,
            );
        }

        Self::enqueue_vec_op(&mut chain, ws.prelude, ws.cur, 0, dim);
        if n_blocks > 0 {
            Self::enqueue_vec_op(&mut chain, ws.prelude_blocks, ws.blocks, 0, n_blocks * dim);
        }
        let n_prelude_blocks = n_blocks;

        // 3. Looped Core (`eff_passes` passes over `loop_start .. loop_end`)
        for pass in 0..eff_passes {
            let mut n_bank = 0usize;
            if pass > 0 {
                let bank_src_idx = n_prelude_blocks + 1;
                if n_blocks > bank_src_idx {
                    n_bank = n_blocks - bank_src_idx;
                    let src_va = ws.blocks + ((bank_src_idx * dim * 4) as u64);
                    Self::enqueue_vec_op(&mut chain, ws.bank, src_va, 0, n_bank * dim);
                }
                Self::enqueue_vec_op(&mut chain, ws.bank_out, ws.cur, 0, dim);
                Self::enqueue_vec_op(&mut chain, ws.cur, ws.prelude, 0, dim);
                if n_prelude_blocks > 0 {
                    Self::enqueue_vec_op(&mut chain, ws.blocks, ws.prelude_blocks, 0, n_prelude_blocks * dim);
                }
                n_blocks = n_prelude_blocks;
            }

            for ip in start..end {
                let cache_slot = if pass == 0 {
                    ip
                } else {
                    n_phys + (pass - 1) * (end - start) + (ip - start)
                };
                if pass > 0 {
                    let lc = &loop_ctrls[ip - start];
                    Self::enqueue_depth_mix(&mut chain, ws.messages, ws.bank, ws.bank_out, lc.msg_query, dim, n_bank, 4);
                }
                Self::enqueue_layer(
                    cfg,
                    layers,
                    loop_ctrls,
                    slot_kv_index,
                    ws,
                    swa_k_base,
                    swa_v_base,
                    full_k_base,
                    full_v_base,
                    conv_base,
                    &mut chain,
                    ip,
                    ip,
                    cache_slot,
                    &mut n_blocks,
                    pass > 0,
                );
            }
        }

        // 4. Epilogue layers `loop_end .. n_phys`
        for il in end..n_phys {
            Self::enqueue_layer(
                cfg,
                layers,
                loop_ctrls,
                slot_kv_index,
                ws,
                swa_k_base,
                swa_v_base,
                full_k_base,
                full_v_base,
                conv_base,
                &mut chain,
                il,
                il,
                il,
                &mut n_blocks,
                false,
            );
        }

        // 5. Optional Final Output Norm + `lm_head` + Sampler
        if compute_logits {
            Self::enqueue_depth_mix(&mut chain, ws.ffn_mixed, ws.blocks, ws.cur, output_res_query, dim, n_blocks, 1);
            Self::enqueue_rmsnorm(&mut chain, ws.cur, ws.ffn_mixed, output_norm, dim, cfg.rms_eps);
            Self::enqueue_gemv(&mut chain, ws.logits, output_w, ws.cur, dim, cfg.vocab_unpadded, cfg.final_logit_softcap);

            let mut s_args = [0u8; 28];
            put_u64(&mut s_args, 0, ctrl_host_va + 64);
            put_u64(&mut s_args, 8, ws.logits);
            put_u64(&mut s_args, 16, ws.ctrl_vram);
            put_u32(&mut s_args, 24, cfg.vocab_unpadded as u32);
            chain.push("ink_sample_logits_kernel", (1, 1, 1), (256, 1, 1), &s_args);
        }

        // 6. Tail completion fence + sequence writeback
        {
            let mut d_args = [0u8; 16];
            put_u64(&mut d_args, 0, ctrl_host_va + 128);
            put_u64(&mut d_args, 8, ws.ctrl_vram);
            chain.push("ink_signal_done_kernel", (1, 1, 1), (32, 1, 1), &d_args);
        }

        chain.finish()
    }

    /// Execute one complete 48-layer (`6 + 3x12 + 6`) looped forward pass on the RTX 3080
    /// by ringing `1` GPFIFO doorbell against a pre-built GDDR6X VRAM `QMDv3` graph (`0` QMD PCIe fetches!).
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
        if pos >= self.cfg.max_ctx {
            return Err(format!("Position {} exceeds max_ctx {}", pos, self.cfg.max_ctx));
        }

        let eff_passes = passes_override
            .unwrap_or(self.cfg.loop_passes)
            .clamp(1, 3);
        let head_va = self.graph_heads[eff_passes - 1][if compute_logits { 1 } else { 0 }];

        self.done_seq_counter += 1;
        let expected_seq = self.done_seq_counter;

        {
            let c = self.ctrl_host.as_mut_slice();
            put_u32(c, 0, tok_id as u32);
            put_u32(c, 4, pos as u32);
            put_u32(c, 8, temperature.to_bits());
            put_u32(c, 12, step);
            put_u64(c, 16, rng_seed);
            put_u64(c, 24, expected_seq);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let inv = self.first_launch;
        self.first_launch = false;
        self.gpu.launch_qmd_chain(head_va, inv)?;

        let done_ptr = unsafe { self.ctrl_host.as_ptr().add(128) as *const u64 };
        let t_wait = Instant::now();
        let mut spins = 0u64;
        loop {
            let cur_seq = unsafe { std::ptr::read_volatile(done_ptr) };
            if cur_seq >= expected_seq {
                break;
            }
            spins += 1;
            if (spins & 0x3fff) == 0 && t_wait.elapsed() > Duration::from_secs(15) {
                return Err(format!(
                    "Inkling VRAM QMD graph timeout waiting for seq {} (current={})",
                    expected_seq, cur_seq
                ));
            }
            std::hint::spin_loop();
        }

        if compute_logits {
            let tok_ptr = unsafe { self.ctrl_host.as_ptr().add(64) as *const i32 };
            let sampled = unsafe { std::ptr::read_volatile(tok_ptr) };
            Ok(sampled)
        } else {
            Ok(0)
        }
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
        if prompt_tokens.is_empty() {
            return Err("Prompt tokens cannot be empty".into());
        }
        self.reset_cache()?;

        let eff_passes = passes_override
            .unwrap_or(self.cfg.loop_passes)
            .clamp(1, self.cfg.loop_passes);
        let logical_layers = self.cfg.n_phys_layers + (eff_passes - 1) * (self.cfg.loop_end - self.cfg.loop_start);

        let t_total = Instant::now();
        let t_prefill = Instant::now();
        let mut next_tok = 0i32;
        for (idx, &ptok) in prompt_tokens.iter().enumerate() {
            let is_last = idx + 1 == prompt_tokens.len();
            next_tok = self.forward_one_token(ptok, idx, is_last, temperature, rng_seed, 0, Some(eff_passes))?;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = prefill_ms;

        let mut generated = Vec::with_capacity(max_tokens);
        let mut decoder = Utf8StreamDecoder::new();
        let mut in_thinking = false;

        let mut emit_tok = |tok: &BpeTokenizer, tid: i32, dec: &mut Utf8StreamDecoder, in_th: &mut bool| {
            if raw_tokens_stream {
                let piece = dec.push(tok.token_bytes(tid));
                on_piece(tid, &piece);
                return;
            }
            match tid {
                TOK_MSG_MODEL => {}
                TOK_CONTENT_THINK => {
                    *in_th = true;
                    on_piece(tid, "<think>\n");
                }
                TOK_END_MESSAGE => {
                    if *in_th {
                        *in_th = false;
                        on_piece(tid, "\n</think>\n\n");
                    }
                }
                TOK_CONTENT_TEXT => {}
                TOK_END_SAMPLING | TOK_EOT => {}
                _ => {
                    let piece = dec.push(tok.token_bytes(tid));
                    if !piece.is_empty() {
                        on_piece(tid, &piece);
                    }
                }
            }
        };

        let t_decode = Instant::now();
        let mut pos = prompt_tokens.len();
        for step in 0..max_tokens {
            if next_tok == TOK_END_SAMPLING || next_tok == TOK_EOT {
                break;
            }
            generated.push(next_tok);
            emit_tok(&self.tok, next_tok, &mut decoder, &mut in_thinking);
            if step + 1 >= max_tokens || pos >= self.cfg.max_ctx {
                break;
            }
            next_tok = self.forward_one_token(
                next_tok,
                pos,
                true,
                temperature,
                rng_seed,
                (step + 1) as u32,
                Some(eff_passes),
            )?;
            pos += 1;
        }
        let tail = decoder.flush();
        if !tail.is_empty() {
            on_piece(-1, &tail);
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let dec_toks = generated.len().saturating_sub(1);
        let decode_tok_per_s = if decode_ms > 0.0 && dec_toks > 0 {
            (dec_toks as f64) / (decode_ms / 1000.0)
        } else if generated.len() == 1 {
            1000.0 / prefill_ms.max(1e-3)
        } else {
            0.0
        };
        let prefill_tok_per_s = (prompt_tokens.len() as f64) / (prefill_ms / 1000.0).max(1e-6);

        let vram_weights_mb = (self.weights_buf.size as f64) / (1024.0 * 1024.0);
        let vram_kv_conv_mb = ((self.swa_k_buf.size
            + self.swa_v_buf.size
            + self.full_k_buf.size
            + self.full_v_buf.size
            + self.conv_state_buf.size) as f64)
            / (1024.0 * 1024.0);

        Ok(InklingGenStats {
            model_name: self.cfg.model_name.clone(),
            loop_passes: eff_passes,
            logical_layers_per_tok: logical_layers,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: generated.len(),
            prefill_ms,
            decode_ms,
            total_ms,
            prefill_tok_per_s,
            decode_tok_per_s,
            ttft_ms,
            vram_weights_mb,
            vram_kv_conv_mb,
            token_ids: generated,
        })
    }
}
