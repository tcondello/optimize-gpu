//! Native GA102 Persistent Megakernel & Speculative `BM=4` Qwen Inference Engine.
//! Embeds pre-compiled `qwen_sm86_megakernel.cubin` directly into the Rust binary (zero runtime Docker/nvcc).

use std::collections::HashMap;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use crate::asr::{AsrStats, QwenAsrEngine};
use crate::gguf::{
    decode_1d_f32, extract_quant_weight_bytes, precompute_freqs_cis_f32, resolve_model_path, BpeTokenizer, GgufFile,
    QwenModelConfig,
};
use crate::lora::{
    prepare_lora_from_path, LoraStatus, QwenLayerLoraDesc, LORA_ARENA_DEFAULT_BYTES,
};
use crate::nv_defs::{put_u32, put_u64};
use crate::nv_hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
use crate::tts::{QwenTtsEngine, TtsStats, TtsSynthParams};

pub static EMBEDDED_CUBIN: &[u8] = include_bytes!("../kernels/qwen_sm86_megakernel.cubin");

const PREFILL_MAX_CHUNK: usize = 64;

// -----------------------------------------------------------------------------
// Exact Host/Device ABI Structs Matching `kernels/qwen_sm86_megakernel.cu`
// -----------------------------------------------------------------------------

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
    pub stop_reason: &'static str,
}

// -----------------------------------------------------------------------------
// Fast Lossless N-Gram Draft Table (Prompt + Generated History Lookup)
// -----------------------------------------------------------------------------

struct NgramDraftTable {
    map4: HashMap<(i32, i32, i32, i32), [i32; 3]>,
    map3: HashMap<(i32, i32, i32), [i32; 3]>,
    indexed_len: usize,
}

impl NgramDraftTable {
    fn new() -> Self {
        Self {
            map4: HashMap::with_capacity(2048),
            map3: HashMap::with_capacity(2048),
            indexed_len: 0,
        }
    }

    /// Index all `seq[0..seq.len()]` transitions where 3 lookahead tokens exist (`i + 6 < seq.len()`).
    fn update(&mut self, seq: &[i32]) {
        if seq.len() < 7 {
            return;
        }
        let start = self.indexed_len.saturating_sub(7);
        let limit = seq.len() - 7;
        for i in start..=limit {
            let d = [seq[i + 4], seq[i + 5], seq[i + 6]];
            self.map4.insert((seq[i], seq[i + 1], seq[i + 2], seq[i + 3]), d);
            self.map3.insert((seq[i + 1], seq[i + 2], seq[i + 3]), d);
        }
        self.indexed_len = seq.len();
    }

    /// Look up a high-precision 3-token draft continuation after `seq` (`4`-gram -> `3`-gram fallback).
    #[inline]
    fn lookup(&self, seq: &[i32]) -> Option<[i32; 3]> {
        let n = seq.len();
        if n >= 4 {
            if let Some(&d) = self.map4.get(&(seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1])) {
                return Some(d);
            }
        }
        if n >= 3 {
            if let Some(&d) = self.map3.get(&(seq[n - 3], seq[n - 2], seq[n - 1])) {
                return Some(d);
            }
        }
        None
    }
}

// -----------------------------------------------------------------------------
// QwenMegakernelEngine
// -----------------------------------------------------------------------------

pub struct QwenMegakernelEngine {
    pub gpu: NvGpu,
    pub module: CubinModule,
    pub cfg: QwenModelConfig,
    pub tok: BpeTokenizer,
    pub model_ref: String,

    // Keep VRAM allocations alive for the lifetime of the engine
    _owned_vram: Vec<VramBuffer>,
    layers_host: Vec<QwenLayerDesc>,
    layers_gpu: VramBuffer,
    model_desc_gpu: VramBuffer,

    // Dynamic Rank-r (`r <= 32`) LoRA arena & co-resident 100%-GPU TTS + ASR engines
    lora_arena: Option<VramBuffer>,
    lora_descs_host: Vec<QwenLayerLoraDesc>,
    pub lora_status: LoraStatus,
    pub tts: Option<QwenTtsEngine>,
    pub asr: Option<QwenAsrEngine>,

    // Scratch workspace buffers (sized for both `PREFILL_MAX_CHUNK=64` and `BM=4` / `T=1`)
    ws_x_res: VramBuffer,
    ws_x_res2: VramBuffer,
    ws_xq: VramBuffer,
    ws_xds: VramBuffer,
    ws_qkv: VramBuffer,
    ws_v_sep: VramBuffer,
    ws_q_roped: VramBuffer,
    ws_attn_out: VramBuffer,
    ws_gate_up: VramBuffer,
    ws_bar_state: VramBuffer,
    freqs_cis_gpu: VramBuffer,
    emb_gpu_va: u64,
    emb_type: i32,

    ctrl_buf: HostMappedBuffer,
    prefill_toks_buf: HostMappedBuffer,
    cached_kv_tokens: Vec<i32>,
    rng_counter: u64,
}

impl QwenMegakernelEngine {
    pub fn load(model_ref: &str, max_context: usize) -> Result<Self, String> {
        assert_eq!(std::mem::size_of::<QwenLayerDesc>(), 176);
        assert_eq!(std::mem::size_of::<QwenModelDesc>(), 224);

        let t0 = Instant::now();
        let local_path = resolve_model_path(model_ref)?;
        let stem = local_path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen");
        let gguf = GgufFile::open(&local_path)?;
        let tok = BpeTokenizer::from_gguf(&gguf)?;
        let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_context, stem)?;

        let mut gpu = NvGpu::init()?;
        let module = gpu.load_cubin(EMBEDDED_CUBIN)?;

        let mut owned_vram = Vec::<VramBuffer>::new();
        let kv_alloc_ctx = cfg.max_ctx + PREFILL_MAX_CHUNK;

        // Helper to upload f32 slice to a packed VRAM arena or dedicated buffer
        let mut small_arena_bytes = Vec::<u8>::with_capacity(4 << 20);
        let push_f32_arena = |arena: &mut Vec<u8>, vals: &[f32]| -> u64 {
            let pad = (256 - (arena.len() % 256)) % 256;
            arena.resize(arena.len() + pad, 0);
            let off = arena.len() as u64;
            for &v in vals {
                arena.extend_from_slice(&v.to_le_bytes());
            }
            off
        };

        // Precompute RoPE `freqs_cis` `[kv_alloc_ctx, rope_dim]`
        let freqs_vec = precompute_freqs_cis_f32(cfg.rope_dim, kv_alloc_ctx, cfg.rope_theta);
        let mut freqs_bytes = Vec::with_capacity(freqs_vec.len() * 4);
        for &v in &freqs_vec {
            freqs_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let freqs_cis_gpu = gpu.alloc_vram(freqs_bytes.len() as u64)?;
        gpu.upload_to_vram(&freqs_cis_gpu, 0, &freqs_bytes)?;

        // Upload Embedding & LM Head
        let (emb_cow, emb_type, _, _) = extract_quant_weight_bytes(&gguf, "token_embd.weight")?;
        let emb_buf = gpu.alloc_vram(emb_cow.len() as u64)?;
        gpu.upload_to_vram(&emb_buf, 0, &emb_cow)?;
        let emb_gpu_va = emb_buf.va_addr;
        owned_vram.push(emb_buf);

        let (lm_head_va, lm_head_type) = if gguf.tensors.contains_key("output.weight") {
            let (lh_cow, lh_type, _, _) = extract_quant_weight_bytes(&gguf, "output.weight")?;
            let lh_buf = gpu.alloc_vram(lh_cow.len() as u64)?;
            gpu.upload_to_vram(&lh_buf, 0, &lh_cow)?;
            let va = lh_buf.va_addr;
            owned_vram.push(lh_buf);
            (va, lh_type)
        } else {
            (emb_gpu_va, emb_type)
        };

        let out_norm_vec = decode_1d_f32(&gguf, "output_norm.weight")?
            .ok_or("Missing output_norm.weight in GGUF")?;
        let out_norm_rel = push_f32_arena(&mut small_arena_bytes, &out_norm_vec);

        // Allocate ONE contiguous KV-cache slab for all layers (`num_layers * 2 * n_kv_heads * kv_alloc_ctx * head_dim * 2` bytes)
        let kv_layer_bytes = ((2 * cfg.n_kv_heads * kv_alloc_ctx * cfg.head_dim * 2) as u64 + 255) & !255;
        let kv_slab = gpu.alloc_vram(kv_layer_bytes * (cfg.num_layers as u64))?;
        let kv_slab_va = kv_slab.va_addr;
        owned_vram.push(kv_slab);

        // Pack all layer weights into large 64 MB VRAM slabs for 2MB-hugepage TLB residency & max DMA speed!
        let t_upload = Instant::now();
        let mut total_weight_bytes = emb_cow.len() as u64;

        struct LayerRel {
            desc: QwenLayerDesc,
            attn_norm_rel: u64,
            ffn_norm_rel: u64,
            q_norm_rel: Option<u64>,
            k_norm_rel: Option<u64>,
            b_qkv_rel: Option<u64>,
            b_v_rel: Option<u64>,
            b_o_rel: Option<u64>,
        }
        let mut layer_rels = Vec::with_capacity(cfg.num_layers);

        for i in 0..cfg.num_layers {
            let p = format!("blk.{}", i);
            let an = decode_1d_f32(&gguf, &format!("{}.attn_norm.weight", p))?
                .ok_or_else(|| format!("Missing {}.attn_norm.weight", p))?;
            let fnw = decode_1d_f32(&gguf, &format!("{}.ffn_norm.weight", p))?
                .ok_or_else(|| format!("Missing {}.ffn_norm.weight", p))?;
            let attn_norm_rel = push_f32_arena(&mut small_arena_bytes, &an);
            let ffn_norm_rel = push_f32_arena(&mut small_arena_bytes, &fnw);

            let q_norm_rel = if cfg.has_qk_norm {
                decode_1d_f32(&gguf, &format!("{}.attn_q_norm.weight", p))?
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v))
            } else {
                None
            };
            let k_norm_rel = if cfg.has_qk_norm {
                decode_1d_f32(&gguf, &format!("{}.attn_k_norm.weight", p))?
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v))
            } else {
                None
            };

            let (wq, tq, oq, _) = extract_quant_weight_bytes(&gguf, &format!("{}.attn_q.weight", p))?;
            let (wk, tk, ok, _) = extract_quant_weight_bytes(&gguf, &format!("{}.attn_k.weight", p))?;
            let (wv, tv, ov, _) = extract_quant_weight_bytes(&gguf, &format!("{}.attn_v.weight", p))?;
            let bq = decode_1d_f32(&gguf, &format!("{}.attn_q.bias", p))?;
            let bk = decode_1d_f32(&gguf, &format!("{}.attn_k.bias", p))?;
            let bv = decode_1d_f32(&gguf, &format!("{}.attn_v.bias", p))?;

            let fuse_biases = |parts: &[(&Option<Vec<f32>>, usize)]| -> Option<Vec<f32>> {
                if parts.iter().all(|(b, _)| b.is_none()) {
                    return None;
                }
                let mut f = Vec::new();
                for &(b_opt, len) in parts {
                    if let Some(bv) = b_opt {
                        f.extend_from_slice(bv);
                    } else {
                        f.resize(f.len() + len, 0.0);
                    }
                }
                Some(f)
            };

            let (w_qkv_va, qkv_type, b_qkv_rel, w_v_va, v_type, b_v_rel, split_v) = if tq == tk && tk == tv {
                let mut fused = Vec::with_capacity(wq.len() + wk.len() + wv.len());
                fused.extend_from_slice(&wq);
                fused.extend_from_slice(&wk);
                fused.extend_from_slice(&wv);
                let buf = gpu.alloc_vram(fused.len() as u64)?;
                gpu.upload_to_vram(&buf, 0, &fused)?;
                total_weight_bytes += fused.len() as u64;
                let va = buf.va_addr;
                owned_vram.push(buf);
                let brel = fuse_biases(&[(&bq, oq), (&bk, ok), (&bv, ov)])
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                (va, tq, brel, 0u64, 0i32, None, 0i32)
            } else if tq == tk {
                let mut fused_qk = Vec::with_capacity(wq.len() + wk.len());
                fused_qk.extend_from_slice(&wq);
                fused_qk.extend_from_slice(&wk);
                let buf_qk = gpu.alloc_vram(fused_qk.len() as u64)?;
                gpu.upload_to_vram(&buf_qk, 0, &fused_qk)?;
                let buf_v = gpu.alloc_vram(wv.len() as u64)?;
                gpu.upload_to_vram(&buf_v, 0, &wv)?;
                total_weight_bytes += (fused_qk.len() + wv.len()) as u64;
                let va_qk = buf_qk.va_addr;
                let va_v = buf_v.va_addr;
                owned_vram.push(buf_qk);
                owned_vram.push(buf_v);
                let bqk_rel = fuse_biases(&[(&bq, oq), (&bk, ok)])
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                let bv_rel = bv.map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                (va_qk, tq, bqk_rel, va_v, tv, bv_rel, 1i32)
            } else {
                return Err(format!("Layer {} has mismatched Q ({}) and K ({}) types", i, tq, tk));
            };

            let (wo, to_t, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.attn_output.weight", p))?;
            let bo = decode_1d_f32(&gguf, &format!("{}.attn_output.bias", p))?;
            let buf_o = gpu.alloc_vram(wo.len() as u64)?;
            gpu.upload_to_vram(&buf_o, 0, &wo)?;
            total_weight_bytes += wo.len() as u64;
            let w_o_va = buf_o.va_addr;
            owned_vram.push(buf_o);
            let b_o_rel = bo.map(|v| push_f32_arena(&mut small_arena_bytes, &v));

            let (wg, tg, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_gate.weight", p))?;
            let (wu, tu, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_up.weight", p))?;
            let (w_gu_va, gu_type, w_up_va, up_type, split_gu) = if tg == tu {
                let mut fused_gu = Vec::with_capacity(wg.len() + wu.len());
                fused_gu.extend_from_slice(&wg);
                fused_gu.extend_from_slice(&wu);
                let buf_gu = gpu.alloc_vram(fused_gu.len() as u64)?;
                gpu.upload_to_vram(&buf_gu, 0, &fused_gu)?;
                total_weight_bytes += fused_gu.len() as u64;
                let va = buf_gu.va_addr;
                owned_vram.push(buf_gu);
                (va, tg, 0u64, 0i32, 0i32)
            } else {
                let buf_g = gpu.alloc_vram(wg.len() as u64)?;
                gpu.upload_to_vram(&buf_g, 0, &wg)?;
                let buf_u = gpu.alloc_vram(wu.len() as u64)?;
                gpu.upload_to_vram(&buf_u, 0, &wu)?;
                total_weight_bytes += (wg.len() + wu.len()) as u64;
                let va_g = buf_g.va_addr;
                let va_u = buf_u.va_addr;
                owned_vram.push(buf_g);
                owned_vram.push(buf_u);
                (va_g, tg, va_u, tu, 1i32)
            };

            let (wd, td, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_down.weight", p))?;
            let buf_d = gpu.alloc_vram(wd.len() as u64)?;
            gpu.upload_to_vram(&buf_d, 0, &wd)?;
            total_weight_bytes += wd.len() as u64;
            let w_down_va = buf_d.va_addr;
            owned_vram.push(buf_d);

            let desc = QwenLayerDesc {
                w_qkv_or_qk: w_qkv_va,
                w_v: w_v_va,
                w_o: w_o_va,
                w_gate_up: w_gu_va,
                w_up_sep: w_up_va,
                w_down: w_down_va,
                kv_cache: kv_slab_va + (i as u64) * kv_layer_bytes,
                qkv_type,
                v_type,
                o_type: to_t,
                gate_up_type: gu_type,
                up_sep_type: up_type,
                down_type: td,
                split_v,
                split_gate_up: split_gu,
                ..Default::default()
            };

            layer_rels.push(LayerRel {
                desc,
                attn_norm_rel,
                ffn_norm_rel,
                q_norm_rel,
                k_norm_rel,
                b_qkv_rel,
                b_v_rel,
                b_o_rel,
            });
        }

        // Upload consolidated norm + bias F32 arena in one single DMA!
        let small_arena_gpu = gpu.alloc_vram(small_arena_bytes.len().max(4096) as u64)?;
        gpu.upload_to_vram(&small_arena_gpu, 0, &small_arena_bytes)?;
        let arena_va = small_arena_gpu.va_addr;
        owned_vram.push(small_arena_gpu);

        let mut layers_host = Vec::with_capacity(cfg.num_layers);
        for lr in layer_rels {
            let mut d = lr.desc;
            d.attn_norm_w = arena_va + lr.attn_norm_rel;
            d.ffn_norm_w = arena_va + lr.ffn_norm_rel;
            d.q_norm_w = lr.q_norm_rel.map_or(0, |r| arena_va + r);
            d.k_norm_w = lr.k_norm_rel.map_or(0, |r| arena_va + r);
            d.b_qkv_or_qk = lr.b_qkv_rel.map_or(0, |r| arena_va + r);
            d.b_v = lr.b_v_rel.map_or(0, |r| arena_va + r);
            d.b_o = lr.b_o_rel.map_or(0, |r| arena_va + r);
            layers_host.push(d);
        }

        let layers_bytes = unsafe {
            std::slice::from_raw_parts(
                layers_host.as_ptr() as *const u8,
                layers_host.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let layers_gpu = gpu.alloc_vram(layers_bytes.len() as u64)?;
        gpu.upload_to_vram(&layers_gpu, 0, layers_bytes)?;
        let layers_ptr_va = layers_gpu.va_addr;

        // Allocate L2/VRAM scratch buffers
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let max_k = cfg.dim.max(q_dim).max(cfg.hidden_dim);

        let ws_x_res = gpu.alloc_vram((PREFILL_MAX_CHUNK * cfg.dim * 4) as u64)?;
        let ws_x_res2 = gpu.alloc_vram((PREFILL_MAX_CHUNK * cfg.dim * 4) as u64)?;
        let ws_xq = gpu.alloc_vram((PREFILL_MAX_CHUNK * max_k) as u64)?;
        let ws_xds = gpu.alloc_vram((PREFILL_MAX_CHUNK * (max_k / 32) * 8) as u64)?;
        let ws_qkv = gpu.alloc_vram((PREFILL_MAX_CHUNK * (q_dim + 2 * kv_dim) * 4) as u64)?;
        let ws_v_sep = gpu.alloc_vram((PREFILL_MAX_CHUNK * kv_dim * 4) as u64)?;
        let ws_q_roped = gpu.alloc_vram((PREFILL_MAX_CHUNK * q_dim * 4) as u64)?;
        let ws_attn_out = gpu.alloc_vram((PREFILL_MAX_CHUNK * q_dim * 4) as u64)?;
        let ws_gate_up = gpu.alloc_vram((PREFILL_MAX_CHUNK * 2 * cfg.hidden_dim * 4) as u64)?;
        let ws_cta_val = gpu.alloc_vram(4 * 128 * 4)?;
        let ws_cta_idx = gpu.alloc_vram(4 * 128 * 4)?;
        let ws_bar_state = gpu.alloc_vram(4096)?;
        gpu.zero_vram(&ws_bar_state)?;

        let cta_val_va = ws_cta_val.va_addr;
        let cta_idx_va = ws_cta_idx.va_addr;
        owned_vram.push(ws_cta_val);
        owned_vram.push(ws_cta_idx);

        // Stop token IDs (up to 4)
        let mut sids: Vec<i32> = cfg.stop_ids.iter().copied().collect();
        sids.sort_unstable();
        while sids.len() < 4 {
            sids.push(cfg.eos_id);
        }

        let mdesc = QwenModelDesc {
            layers_ptr: layers_ptr_va,
            emb_w: emb_gpu_va,
            output_norm_w: arena_va + out_norm_rel,
            lm_head_w: lm_head_va,
            freqs_cis: freqs_cis_gpu.va_addr,
            ws_x_res: ws_x_res.va_addr,
            ws_x_res2: ws_x_res2.va_addr,
            ws_xq: ws_xq.va_addr,
            ws_xds: ws_xds.va_addr,
            ws_qkv: ws_qkv.va_addr,
            ws_q_roped: ws_q_roped.va_addr,
            ws_gate_up: ws_gate_up.va_addr,
            ws_cta_val: cta_val_va,
            ws_cta_idx: cta_idx_va,
            ws_bar_state: ws_bar_state.va_addr,
            num_layers: cfg.num_layers as i32,
            dim: cfg.dim as i32,
            hidden_dim: cfg.hidden_dim as i32,
            n_heads: cfg.n_heads as i32,
            n_kv_heads: cfg.n_kv_heads as i32,
            head_dim: cfg.head_dim as i32,
            rope_dim: cfg.rope_dim as i32,
            vocab_size: cfg.vocab_size as i32,
            max_ctx: kv_alloc_ctx as i32,
            has_qk_norm: i32::from(cfg.has_qk_norm),
            emb_type,
            lm_head_type,
            norm_eps: cfg.norm_eps,
            inv_sqrt_head_dim: 1.0 / (cfg.head_dim as f32).sqrt(),
            stop_id0: sids[0],
            stop_id1: sids[1],
            stop_id2: sids[2],
            stop_id3: sids[3],
            ws_ext_embed: 0,
            ws_hidden_last: 0,
            ws_logits_out: 0,
            _pad_model: 0,
        };
        let mdesc_bytes = unsafe {
            std::slice::from_raw_parts(&mdesc as *const QwenModelDesc as *const u8, std::mem::size_of::<QwenModelDesc>())
        };
        let model_desc_gpu = gpu.alloc_vram(4096)?;
        gpu.upload_to_vram(&model_desc_gpu, 0, mdesc_bytes)?;

        let ctrl_buf = gpu.alloc_host_mapped(std::mem::size_of::<QwenDecodeCtrl>())?;
        let prefill_toks_buf = gpu.alloc_host_mapped(PREFILL_MAX_CHUNK * 4)?;

        let dma_gbps = (total_weight_bytes as f64 / 1e9) / t_upload.elapsed().as_secs_f64();
        eprintln!(
            "[qwen-nv] Loaded '{}' ({:.2} GB weights @ {:.2} GB/s DMA, {} layers, ctx={}) in {:.2}s",
            cfg.model_name,
            total_weight_bytes as f64 / 1e9,
            dma_gbps,
            cfg.num_layers,
            cfg.max_ctx,
            t0.elapsed().as_secs_f64()
        );

        let num_layers_cfg = cfg.num_layers;

        Ok(Self {
            gpu,
            module,
            cfg,
            tok,
            model_ref: model_ref.to_string(),
            _owned_vram: owned_vram,
            layers_host,
            layers_gpu,
            model_desc_gpu,
            lora_arena: None,
            lora_descs_host: vec![QwenLayerLoraDesc::default(); num_layers_cfg],
            lora_status: LoraStatus::default(),
            tts: None,
            asr: None,
            ws_x_res,
            ws_x_res2,
            ws_xq,
            ws_xds,
            ws_qkv,
            ws_v_sep,
            ws_q_roped,
            ws_attn_out,
            ws_gate_up,
            ws_bar_state,
            freqs_cis_gpu,
            emb_gpu_va,
            emb_type,
            ctrl_buf,
            prefill_toks_buf,
            cached_kv_tokens: Vec::new(),
            rng_counter: 0x2026_1118_cafe_babe,
        })
    }

    pub fn reset_cache(&mut self) {
        self.cached_kv_tokens.clear();
    }

    // -------------------------------------------------------------------------
    // Dynamic Fused Rank-r (`r <= 32`) LoRA Hot-Swap (`< 1 ms` VRAM update!)
    // -------------------------------------------------------------------------

    pub fn load_lora(&mut self, path: &str, user_scale: f32) -> Result<LoraStatus, String> {
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let mut prep = prepare_lora_from_path(
            path,
            user_scale,
            self.cfg.num_layers,
            self.cfg.dim,
            self.cfg.hidden_dim,
            q_dim,
            kv_dim,
        )?;

        let need_bytes = (prep.arena_bytes.len() as u64).max(LORA_ARENA_DEFAULT_BYTES);
        let need_realloc = self.lora_arena.as_ref().map_or(true, |b| b.size < need_bytes);
        if need_realloc {
            self.lora_arena = Some(self.gpu.alloc_vram(need_bytes)?);
        }
        let t_swap = Instant::now();
        let arena_va = self.lora_arena.as_ref().unwrap().va_addr;

        // Patch relative `a_f16`/`b_f16` offsets into absolute GPU virtual addresses
        let desc_sz = std::mem::size_of::<QwenLayerLoraDesc>();
        for (l, ld_lora) in prep.layer_descs.iter_mut().enumerate() {
            for p in &mut ld_lora.projs {
                if p.rank > 0 {
                    p.a_f16 += arena_va;
                    p.b_f16 += arena_va;
                }
            }
            let src = unsafe {
                std::slice::from_raw_parts(ld_lora as *const QwenLayerLoraDesc as *const u8, desc_sz)
            };
            prep.arena_bytes[l * desc_sz..(l + 1) * desc_sz].copy_from_slice(src);
            self.layers_host[l].lora_ptr = if ld_lora.active_mask != 0 {
                arena_va + (l * desc_sz) as u64
            } else {
                0
            };
        }

        let arena_ref = self.lora_arena.as_ref().unwrap();
        // Copy borrow before mutably calling `self.gpu.upload_to_vram`:
        let arena_buf = VramBuffer {
            va_addr: arena_ref.va_addr,
            size: arena_ref.size,
            paddrs: arena_ref.paddrs.clone(),
        };
        self.gpu.upload_to_vram(&arena_buf, 0, &prep.arena_bytes)?;

        let layers_bytes = unsafe {
            std::slice::from_raw_parts(
                self.layers_host.as_ptr() as *const u8,
                self.layers_host.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let layers_buf = VramBuffer {
            va_addr: self.layers_gpu.va_addr,
            size: self.layers_gpu.size,
            paddrs: self.layers_gpu.paddrs.clone(),
        };
        self.gpu.upload_to_vram(&layers_buf, 0, layers_bytes)?;
        self.reset_cache();

        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        self.lora_descs_host = prep.layer_descs;
        self.lora_status = LoraStatus {
            active: true,
            path: path.to_string(),
            rank: prep.rank,
            alpha: prep.alpha,
            scale: prep.scale,
            num_adapted_tensors: prep.num_adapted_tensors,
            vram_bytes: prep.arena_bytes.len(),
            swap_ms,
        };
        Ok(self.lora_status.clone())
    }

    pub fn clear_lora(&mut self) -> Result<LoraStatus, String> {
        let t_swap = Instant::now();
        for l in 0..self.cfg.num_layers {
            self.layers_host[l].lora_ptr = 0;
            self.lora_descs_host[l] = QwenLayerLoraDesc::default();
        }
        let layers_bytes = unsafe {
            std::slice::from_raw_parts(
                self.layers_host.as_ptr() as *const u8,
                self.layers_host.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let layers_buf = VramBuffer {
            va_addr: self.layers_gpu.va_addr,
            size: self.layers_gpu.size,
            paddrs: self.layers_gpu.paddrs.clone(),
        };
        self.gpu.upload_to_vram(&layers_buf, 0, layers_bytes)?;
        self.reset_cache();
        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        self.lora_status = LoraStatus {
            active: false,
            swap_ms,
            ..Default::default()
        };
        Ok(self.lora_status.clone())
    }

    // -------------------------------------------------------------------------
    // Co-Resident 100%-on-RTX-3080 `Qwen3-TTS-12Hz-1.7B` Engine
    // -------------------------------------------------------------------------

    pub fn ensure_tts_loaded(&mut self, talker_ref: &str) -> Result<(), String> {
        let eff_ref = if talker_ref.trim().is_empty() { "1.7b" } else { talker_ref.trim() };
        if let Some(ref existing) = self.tts {
            if existing.talker_ref == eff_ref {
                return Ok(());
            }
        }
        let tts_engine = QwenTtsEngine::load(&mut self.gpu, eff_ref, None)?;
        self.tts = Some(tts_engine);
        Ok(())
    }

    pub fn synthesize_speech<F>(
        &mut self,
        talker_ref: &str,
        params: &TtsSynthParams,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        on_chunk: F,
    ) -> Result<(Vec<f32>, TtsStats), String>
    where
        F: FnMut(&[f32]) -> bool,
    {
        self.ensure_tts_loaded(talker_ref)?;
        let tts = self.tts.as_mut().unwrap();
        tts.synthesize(&mut self.gpu, &self.module, params, cancel, on_chunk)
    }

    // -------------------------------------------------------------------------
    // Co-Resident 100%-on-RTX-3080 `Qwen3-ASR-1.7B` Speech-to-Text Engine
    // -------------------------------------------------------------------------

    pub fn ensure_asr_loaded(&mut self, asr_ref: &str) -> Result<(), String> {
        let eff_ref = if asr_ref.trim().is_empty() { "1.7b" } else { asr_ref.trim() };
        if let Some(ref existing) = self.asr {
            if existing.asr_ref == eff_ref {
                return Ok(());
            }
        }
        let asr_engine = QwenAsrEngine::load(&mut self.gpu, eff_ref)?;
        self.asr = Some(asr_engine);
        Ok(())
    }

    pub fn transcribe_audio(
        &mut self,
        asr_ref: &str,
        pcm_16k: &[f32],
        language_hint: Option<&str>,
    ) -> Result<AsrStats, String> {
        self.ensure_asr_loaded(asr_ref)?;
        let asr = self.asr.as_mut().unwrap();
        asr.transcribe(&mut self.gpu, &self.module, pcm_16k, language_hint)
    }

    pub fn encode_chat(&self, messages: &[(String, String)], enable_thinking: bool) -> Vec<i32> {
        let mut ids = Vec::new();
        if let Some(bos) = self.cfg.bos_id {
            ids.push(bos);
        }
        for (role, content) in messages {
            ids.extend(self.tok.role_tokens(role));
            ids.extend(self.tok.encode(content));
            ids.extend(self.tok.end_turn_tokens(self.cfg.im_end_id));
        }
        ids.extend(self.tok.role_tokens("assistant"));
        let lower_name = self.cfg.model_name.to_ascii_lowercase();
        if !enable_thinking && lower_name.contains("qwen3") && !lower_name.contains("instruct") {
            ids.extend(self.tok.encode("<think>\n\n</think>\n\n"));
        }
        ids
    }

    fn common_prefix_len(&self, prompt_without_last: &[i32]) -> usize {
        prompt_without_last
            .iter()
            .zip(self.cached_kv_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Hardware-chained QMD prefill for `tokens_slice` starting at `start_pos` (`1` doorbell per chunk!).
    fn run_prefill_chunk(&mut self, chunk: &[i32], start_pos: usize) -> Result<(), String> {
        let t_len = chunk.len();
        assert!(t_len > 0 && t_len <= PREFILL_MAX_CHUNK);

        {
            let dst = self.prefill_toks_buf.as_mut_slice();
            for (i, &tok_id) in chunk.iter().enumerate() {
                put_u32(dst, i * 4, tok_id as u32);
            }
        }

        let k_embed = &self.module.kernels["qwen_prefill_embed"];
        let k_rms = &self.module.kernels["qwen_prefill_rmsnorm_q8_1"];
        let k_gemm = &self.module.kernels["qwen_prefill_gemm"];
        let k_lora = &self.module.kernels["qwen_prefill_lora_add"];
        let k_qknorm = &self.module.kernels["qwen_prefill_qknorm_rope_kv"];
        let k_flash = &self.module.kernels["qwen_prefill_flash_gqa"];
        let k_quant = &self.module.kernels["qwen_prefill_quant_q8_1"];
        let k_swiglu = &self.module.kernels["qwen_prefill_swiglu_q8_1"];

        let push_lora_qmd = |gpu: &mut NvGpu,
                             chain: &mut Vec<(u64, usize)>,
                             out_va: u64,
                             p: &crate::lora::QwenProjLora,
                             x_q8_va: u64,
                             x_ds_va: u64,
                             m_rows: i32,
                             out_stride: i32,
                             col_off: i32| {
            if p.rank <= 0 {
                return;
            }
            let mut a = [0u8; 64];
            put_u64(&mut a, 0, out_va);
            put_u64(&mut a, 8, p.a_f16);
            put_u64(&mut a, 16, p.b_f16);
            put_u64(&mut a, 24, x_q8_va);
            put_u64(&mut a, 32, x_ds_va);
            put_u32(&mut a, 40, m_rows as u32);
            put_u32(&mut a, 44, p.n_rows as u32);
            put_u32(&mut a, 48, p.k_cols as u32);
            put_u32(&mut a, 52, p.rank as u32);
            put_u32(&mut a, 56, out_stride as u32);
            put_u32(&mut a, 60, col_off as u32);
            chain.push(gpu.prepare_qmd(
                k_lora,
                (((p.n_rows as u32) + 255) / 256, m_rows as u32, 1),
                (256, 1, 1),
                &a,
                None,
            ));
        };

        let dim = self.cfg.dim as i32;
        let hidden_dim = self.cfg.hidden_dim as i32;
        let n_heads = self.cfg.n_heads as i32;
        let n_kv_heads = self.cfg.n_kv_heads as i32;
        let head_dim = self.cfg.head_dim as i32;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let m_i32 = t_len as i32;
        let grid_m = ((t_len as u32) + 7) / 8;
        let eps_bits = self.cfg.norm_eps.to_bits();
        let scale_bits = (1.0f32 / (self.cfg.head_dim as f32).sqrt()).to_bits();
        let max_ctx_i32 = (self.cfg.max_ctx + PREFILL_MAX_CHUNK) as i32;

        let mut chain = Vec::<(u64, usize)>::with_capacity(1 + self.cfg.num_layers * 9);

        // 1. Embed
        let mut a_emb = [0u8; 32];
        put_u64(&mut a_emb, 0, self.ws_x_res.va_addr);
        put_u64(&mut a_emb, 8, self.emb_gpu_va);
        put_u64(&mut a_emb, 16, self.prefill_toks_buf.va_addr);
        put_u32(&mut a_emb, 24, dim as u32);
        put_u32(&mut a_emb, 28, self.emb_type as u32);
        chain.push(self.gpu.prepare_qmd(k_embed, (t_len as u32, 1, 1), (256, 1, 1), &a_emb, None));

        for l in 0..self.cfg.num_layers {
            let ld = self.layers_host[l];
            let is_last_layer = l + 1 == self.cfg.num_layers;

            // Attn RMSNorm + Q8_1
            let mut a_rms1 = [0u8; 40];
            put_u64(&mut a_rms1, 0, self.ws_xq.va_addr);
            put_u64(&mut a_rms1, 8, self.ws_xds.va_addr);
            put_u64(&mut a_rms1, 16, self.ws_x_res.va_addr);
            put_u64(&mut a_rms1, 24, ld.attn_norm_w);
            put_u32(&mut a_rms1, 32, dim as u32);
            put_u32(&mut a_rms1, 36, eps_bits);
            chain.push(self.gpu.prepare_qmd(k_rms, (t_len as u32, 1, 1), (256, 1, 1), &a_rms1, None));

            // QKV or QK + V GEMM
            let qk_n = if ld.split_v == 0 { q_dim + 2 * kv_dim } else { q_dim + kv_dim };
            let mut a_qk = [0u8; 64];
            put_u64(&mut a_qk, 0, self.ws_qkv.va_addr);
            put_u64(&mut a_qk, 8, ld.w_qkv_or_qk);
            put_u64(&mut a_qk, 16, self.ws_xq.va_addr);
            put_u64(&mut a_qk, 24, self.ws_xds.va_addr);
            put_u64(&mut a_qk, 32, ld.b_qkv_or_qk);
            put_u64(&mut a_qk, 40, 0);
            put_u32(&mut a_qk, 48, m_i32 as u32);
            put_u32(&mut a_qk, 52, qk_n as u32);
            put_u32(&mut a_qk, 56, dim as u32);
            put_u32(&mut a_qk, 60, ld.qkv_type as u32);
            chain.push(self.gpu.prepare_qmd(
                k_gemm,
                (((qk_n as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_qk,
                None,
            ));

            if ld.split_v != 0 {
                let mut a_v = [0u8; 64];
                put_u64(&mut a_v, 0, self.ws_v_sep.va_addr);
                put_u64(&mut a_v, 8, ld.w_v);
                put_u64(&mut a_v, 16, self.ws_xq.va_addr);
                put_u64(&mut a_v, 24, self.ws_xds.va_addr);
                put_u64(&mut a_v, 32, ld.b_v);
                put_u64(&mut a_v, 40, 0);
                put_u32(&mut a_v, 48, m_i32 as u32);
                put_u32(&mut a_v, 52, kv_dim as u32);
                put_u32(&mut a_v, 56, dim as u32);
                put_u32(&mut a_v, 60, ld.v_type as u32);
                chain.push(self.gpu.prepare_qmd(
                    k_gemm,
                    (((kv_dim as u32) + 3) / 4, grid_m, 1),
                    (128, 1, 1),
                    &a_v,
                    None,
                ));
            }

            if self.lora_status.active && (self.lora_descs_host[l].active_mask & 0x07) != 0 {
                let ldesc = self.lora_descs_host[l];
                let qk_stride = if ld.split_v == 0 { q_dim + 2 * kv_dim } else { q_dim + kv_dim };
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_qkv.va_addr, &ldesc.projs[0], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, qk_stride, 0);
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_qkv.va_addr, &ldesc.projs[1], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, qk_stride, q_dim);
                if ld.split_v == 0 {
                    push_lora_qmd(&mut self.gpu, &mut chain, self.ws_qkv.va_addr, &ldesc.projs[2], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, qk_stride, q_dim + kv_dim);
                } else {
                    push_lora_qmd(&mut self.gpu, &mut chain, self.ws_v_sep.va_addr, &ldesc.projs[2], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, kv_dim, 0);
                }
            }

            // Fused QK-RMSNorm + RoPE + KV Cache Write
            let mut a_rope = [0u8; 96];
            put_u64(&mut a_rope, 0, self.ws_q_roped.va_addr);
            put_u64(&mut a_rope, 8, ld.kv_cache);
            put_u64(&mut a_rope, 16, self.ws_qkv.va_addr);
            put_u64(&mut a_rope, 24, if ld.split_v != 0 { self.ws_v_sep.va_addr } else { 0 });
            put_u64(&mut a_rope, 32, ld.q_norm_w);
            put_u64(&mut a_rope, 40, ld.k_norm_w);
            put_u64(&mut a_rope, 48, self.freqs_cis_gpu.va_addr);
            put_u32(&mut a_rope, 56, m_i32 as u32);
            put_u32(&mut a_rope, 60, start_pos as u32);
            put_u32(&mut a_rope, 64, n_heads as u32);
            put_u32(&mut a_rope, 68, n_kv_heads as u32);
            put_u32(&mut a_rope, 72, head_dim as u32);
            put_u32(&mut a_rope, 76, self.cfg.rope_dim as u32);
            put_u32(&mut a_rope, 80, max_ctx_i32 as u32);
            put_u32(&mut a_rope, 84, eps_bits);
            put_u32(&mut a_rope, 88, u32::from(self.cfg.has_qk_norm));
            put_u32(&mut a_rope, 92, ld.split_v as u32);
            chain.push(self.gpu.prepare_qmd(
                k_qknorm,
                ((n_heads + 2 * n_kv_heads) as u32, t_len as u32, 1),
                (256, 1, 1),
                &a_rope,
                None,
            ));

            // On the final layer of a pure KV-prefill chunk (where we only need KV populated up to `prompt_len - 1`),
            // we can even skip the final layer's `flash_gqa + o_proj + FFN` when `start_pos + t_len == prompt_len - 1`?
            // WAIT: NO! Only the LAST layer's `flash_gqa + o_proj + FFN` doesn't affect ANY KV cache (since KV of layer 35 was just written above!).
            // Skipping layer `num_layers - 1`'s post-KV compute during prefill saves 1.5% of prefill time with 100.000% mathematical equivalence!
            if is_last_layer {
                break;
            }

            // Flash-GQA
            let mut a_fl = [0u8; 52];
            put_u64(&mut a_fl, 0, self.ws_attn_out.va_addr);
            put_u64(&mut a_fl, 8, self.ws_q_roped.va_addr);
            put_u64(&mut a_fl, 16, ld.kv_cache);
            put_u32(&mut a_fl, 24, m_i32 as u32);
            put_u32(&mut a_fl, 28, start_pos as u32);
            put_u32(&mut a_fl, 32, n_heads as u32);
            put_u32(&mut a_fl, 36, n_kv_heads as u32);
            put_u32(&mut a_fl, 40, head_dim as u32);
            put_u32(&mut a_fl, 44, max_ctx_i32 as u32);
            put_u32(&mut a_fl, 48, scale_bits);
            chain.push(self.gpu.prepare_qmd(
                k_flash,
                (n_heads as u32, t_len as u32, 1),
                (128, 1, 1),
                &a_fl,
                None,
            ));

            // Quantize attn_out -> Q8_1
            let attn_blks = (t_len * (q_dim as usize)) / 32;
            let mut a_q8 = [0u8; 28];
            put_u64(&mut a_q8, 0, self.ws_xq.va_addr);
            put_u64(&mut a_q8, 8, self.ws_xds.va_addr);
            put_u64(&mut a_q8, 16, self.ws_attn_out.va_addr);
            put_u32(&mut a_q8, 24, attn_blks as u32);
            chain.push(self.gpu.prepare_qmd(
                k_quant,
                (((attn_blks as u32) + 3) / 4, 1, 1),
                (128, 1, 1),
                &a_q8,
                None,
            ));

            // Output Projection + Residual Add -> ws_x_res2
            let mut a_wo = [0u8; 64];
            put_u64(&mut a_wo, 0, self.ws_x_res2.va_addr);
            put_u64(&mut a_wo, 8, ld.w_o);
            put_u64(&mut a_wo, 16, self.ws_xq.va_addr);
            put_u64(&mut a_wo, 24, self.ws_xds.va_addr);
            put_u64(&mut a_wo, 32, ld.b_o);
            put_u64(&mut a_wo, 40, self.ws_x_res.va_addr);
            put_u32(&mut a_wo, 48, m_i32 as u32);
            put_u32(&mut a_wo, 52, dim as u32);
            put_u32(&mut a_wo, 56, q_dim as u32);
            put_u32(&mut a_wo, 60, ld.o_type as u32);
            chain.push(self.gpu.prepare_qmd(
                k_gemm,
                (((dim as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_wo,
                None,
            ));
            if self.lora_status.active && (self.lora_descs_host[l].active_mask & 0x08) != 0 {
                let p = self.lora_descs_host[l].projs[3];
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_x_res2.va_addr, &p, self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, dim, 0);
            }

            // FFN RMSNorm + Q8_1
            let mut a_rms2 = [0u8; 40];
            put_u64(&mut a_rms2, 0, self.ws_xq.va_addr);
            put_u64(&mut a_rms2, 8, self.ws_xds.va_addr);
            put_u64(&mut a_rms2, 16, self.ws_x_res2.va_addr);
            put_u64(&mut a_rms2, 24, ld.ffn_norm_w);
            put_u32(&mut a_rms2, 32, dim as u32);
            put_u32(&mut a_rms2, 36, eps_bits);
            chain.push(self.gpu.prepare_qmd(k_rms, (t_len as u32, 1, 1), (256, 1, 1), &a_rms2, None));

            // Gate+Up GEMM
            let gu_n = 2 * hidden_dim;
            let mut a_gu = [0u8; 64];
            put_u64(&mut a_gu, 0, self.ws_gate_up.va_addr);
            put_u64(&mut a_gu, 8, ld.w_gate_up);
            put_u64(&mut a_gu, 16, self.ws_xq.va_addr);
            put_u64(&mut a_gu, 24, self.ws_xds.va_addr);
            put_u64(&mut a_gu, 32, 0);
            put_u64(&mut a_gu, 40, 0);
            put_u32(&mut a_gu, 48, m_i32 as u32);
            put_u32(&mut a_gu, 52, gu_n as u32);
            put_u32(&mut a_gu, 56, dim as u32);
            put_u32(&mut a_gu, 60, ld.gate_up_type as u32);
            chain.push(self.gpu.prepare_qmd(
                k_gemm,
                (((gu_n as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_gu,
                None,
            ));
            if self.lora_status.active && (self.lora_descs_host[l].active_mask & 0x30) != 0 {
                let ldesc = self.lora_descs_host[l];
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_gate_up.va_addr, &ldesc.projs[4], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, gu_n, 0);
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_gate_up.va_addr, &ldesc.projs[5], self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, gu_n, hidden_dim);
            }

            // SwiGLU + Q8_1
            let ffn_blks = (t_len * (hidden_dim as usize)) / 32;
            let mut a_sw = [0u8; 32];
            put_u64(&mut a_sw, 0, self.ws_xq.va_addr);
            put_u64(&mut a_sw, 8, self.ws_xds.va_addr);
            put_u64(&mut a_sw, 16, self.ws_gate_up.va_addr);
            put_u32(&mut a_sw, 24, m_i32 as u32);
            put_u32(&mut a_sw, 28, hidden_dim as u32);
            chain.push(self.gpu.prepare_qmd(
                k_swiglu,
                (((ffn_blks as u32) + 3) / 4, 1, 1),
                (128, 1, 1),
                &a_sw,
                None,
            ));

            // Down GEMM + Residual Add -> ws_x_res
            let mut a_dn = [0u8; 64];
            put_u64(&mut a_dn, 0, self.ws_x_res.va_addr);
            put_u64(&mut a_dn, 8, ld.w_down);
            put_u64(&mut a_dn, 16, self.ws_xq.va_addr);
            put_u64(&mut a_dn, 24, self.ws_xds.va_addr);
            put_u64(&mut a_dn, 32, ld.b_down);
            put_u64(&mut a_dn, 40, self.ws_x_res2.va_addr);
            put_u32(&mut a_dn, 48, m_i32 as u32);
            put_u32(&mut a_dn, 52, dim as u32);
            put_u32(&mut a_dn, 56, hidden_dim as u32);
            put_u32(&mut a_dn, 60, ld.down_type as u32);
            chain.push(self.gpu.prepare_qmd(
                k_gemm,
                (((dim as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_dn,
                None,
            ));
            if self.lora_status.active && (self.lora_descs_host[l].active_mask & 0x40) != 0 {
                let p = self.lora_descs_host[l].projs[6];
                push_lora_qmd(&mut self.gpu, &mut chain, self.ws_x_res.va_addr, &p, self.ws_xq.va_addr, self.ws_xds.va_addr, m_i32, dim, 0);
            }
        }

        // Attach timeline release0 to the tail QMD and link the entire chain in hardware!
        self.gpu.timeline_value += 1;
        let target_sig = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        {
            let (_, last_off) = *chain.last().unwrap();
            let buf = self.gpu.kernargs_buf_mut_slice_at(last_off);
            let mut tail_qmd = crate::nv_defs::QmdV3 {
                bytes: buf[..256].try_into().unwrap(),
            };
            tail_qmd.set_release0(sig_va, target_sig);
            buf[..256].copy_from_slice(&tail_qmd.bytes);
        }

        for i in 0..(chain.len() - 1) {
            self.gpu.link_dependent_qmd(chain[i].1, chain[i + 1].0);
        }

        self.gpu.launch_qmd_chain(chain[0].0, true)?;
        self.gpu.wait_timeline(target_sig, Duration::from_secs(30))
    }

    /// Generate up to `max_tokens` tokens using the Persistent 68-SM Megakernel (+ optional `BM=4` lossless speculative verify).
    pub fn generate<F>(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        enable_speculative: bool,
        external_cancel: Option<&std::sync::atomic::AtomicBool>,
        mut on_token: F,
    ) -> Result<(Vec<i32>, GenerationStats), String>
    where
        F: FnMut(i32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err("prompt_tokens cannot be empty".into());
        }
        let max_steps_cap = max_tokens.clamp(1, 4096);
        let t_start = Instant::now();

        let prompt_len = prompt_tokens.len();
        let prefix_target = &prompt_tokens[..prompt_len - 1];
        let mut pos = self.common_prefix_len(prefix_target);
        let cached_prefix_tokens = pos;
        let prefill_new_tokens = (prompt_len - 1).saturating_sub(pos);

        let mut launches = 0usize;
        // 1. Chunked Hardware-Chained Prefill on `prompt_tokens[pos .. prompt_len - 1]`
        while pos < prompt_len - 1 {
            let take = (prompt_len - 1 - pos).min(PREFILL_MAX_CHUNK);
            self.run_prefill_chunk(&prompt_tokens[pos..pos + take], pos)?;
            pos += take;
            launches += 1;
        }
        let prefill_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        // Track full sequence (`prompt + generated`) for KV prefix cache & N-gram speculative drafting
        let mut full_seq = prompt_tokens.to_vec();
        let mut cur_tok = prompt_tokens[prompt_len - 1];
        let mut cur_pos = prompt_len - 1;

        let use_spec = enable_speculative && temperature <= 1e-6;
        let mut ngram = NgramDraftTable::new();
        if use_spec {
            ngram.update(&full_seq);
        }

        let mut out_tokens = Vec::with_capacity(max_steps_cap);
        let mut ttft_ms = 0.0f64;
        let mut stop_reason: &'static str = "length";
        let mut spec_verify_calls = 0usize;
        let mut spec_accepted_extra = 0usize;
        let t_decode_start = Instant::now();

        let ctrl_ptr = self.ctrl_buf.as_ptr() as *mut QwenDecodeCtrl;
        let mut mk_args = [0u8; 16];
        put_u64(&mut mk_args, 0, self.model_desc_gpu.va_addr);
        put_u64(&mut mk_args, 8, self.ctrl_buf.va_addr);
        let num_sms = std::env::var("QWEN_NV_SMS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(self.gpu.active_sms);

        while out_tokens.len() < max_steps_cap {
            if cur_pos + 1 >= self.cfg.max_ctx {
                stop_reason = "context_limit";
                break;
            }
            if external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                stop_reason = "cancelled";
                break;
            }

            let rem = max_steps_cap - out_tokens.len();

            // Path A: Lossless BM=4 N-Gram Speculative Verification when a draft hits (and rem >= 2)!
            if use_spec && rem >= 2 && cur_pos + 4 < self.cfg.max_ctx + PREFILL_MAX_CHUNK {
                if let Some(draft) = ngram.lookup(&full_seq) {
                    unsafe {
                        (*ctrl_ptr).start_pos = cur_pos as i32;
                        (*ctrl_ptr).error_code = 0;
                        (*ctrl_ptr).steps_completed = 0;
                        (*ctrl_ptr).spec_tokens_in = [cur_tok, draft[0], draft[1], draft[2]];
                        (*ctrl_ptr).spec_tokens_out = [0; 4];
                        (*ctrl_ptr).export_logits = 0;
                    }
                    fence(Ordering::SeqCst);

                    self.gpu.timeline_value += 1;
                    let sig_val = self.gpu.timeline_value;
                    let sig_va = self.gpu.timeline_buf.va_addr;
                    let k_spec = &self.module.kernels["qwen_spec_verify_bm4_megakernel"];
                    let (qmd_va, _) = self.gpu.prepare_qmd(
                        k_spec,
                        (num_sms, 1, 1),
                        (512, 1, 1),
                        &mk_args,
                        Some((sig_va, sig_val)),
                    );
                    self.gpu.launch_qmd_chain(qmd_va, true)?;
                    launches += 1;
                    spec_verify_calls += 1;
                    self.gpu.wait_timeline(sig_val, Duration::from_secs(10))?;

                    let (err, preds) = unsafe { ((*ctrl_ptr).error_code, (*ctrl_ptr).spec_tokens_out) };
                    if err != 0 {
                        let _ = self.gpu.zero_vram(&self.ws_bar_state);
                        return Err(format!("GPU BM=4 watchdog tripped at barrier step {}", err));
                    }

                    // Accept lossless greedy tokens!
                    let max_accept = rem.min(4);
                    let mut hit_stop = false;
                    for k in 0..max_accept {
                        let yk = preds[k];
                        if out_tokens.is_empty() {
                            ttft_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        cur_pos += 1;
                        cur_tok = yk;
                        full_seq.push(yk);

                        if self.cfg.stop_ids.contains(&yk) {
                            stop_reason = "stop";
                            hit_stop = true;
                            break;
                        }
                        out_tokens.push(yk);
                        if k > 0 {
                            spec_accepted_extra += 1;
                        }
                        if !on_token(yk) {
                            stop_reason = "cancelled";
                            hit_stop = true;
                            break;
                        }
                        if k < 3 && yk != draft[k] {
                            break;
                        }
                    }
                    ngram.update(&full_seq);
                    if hit_stop {
                        break;
                    }
                    continue;
                }
            }

            // Path B: Single-Launch Persistent 68-SM Megakernel for all remaining `rem` steps!
            self.rng_counter = self.rng_counter.wrapping_add(0x9e37_79b9_7f4a_7c15);
            unsafe {
                (*ctrl_ptr).start_token = cur_tok;
                (*ctrl_ptr).start_pos = cur_pos as i32;
                (*ctrl_ptr).max_steps = rem as i32;
                (*ctrl_ptr).temperature = temperature;
                (*ctrl_ptr).rng_seed = self.rng_counter;
                (*ctrl_ptr).cancel_flag = 0;
                (*ctrl_ptr).error_code = 0;
                (*ctrl_ptr).steps_completed = 0;
                (*ctrl_ptr).stop_reason = 0;
                (*ctrl_ptr).export_logits = 0;
            }
            fence(Ordering::SeqCst);

            self.gpu.timeline_value += 1;
            let sig_val = self.gpu.timeline_value;
            let sig_va = self.gpu.timeline_buf.va_addr;
            let k_mk = &self.module.kernels["qwen_persistent_decode_megakernel"];
            let (qmd_va, _) = self.gpu.prepare_qmd(
                k_mk,
                (num_sms, 1, 1),
                (512, 1, 1),
                &mk_args,
                Some((sig_va, sig_val)),
            );
            self.gpu.launch_qmd_chain(qmd_va, true)?;
            launches += 1;

            let mut streamed = 0usize;
            let mut user_aborted = false;
            let mut stop_hit = false;
            let sig_ptr = self.gpu.timeline_buf.as_ptr() as *const u64;
            let rollout_t0 = Instant::now();

            loop {
                let done_steps = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).steps_completed) } as usize;
                while streamed < done_steps {
                    let next_tok = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).tokens_out[streamed]) };
                    streamed += 1;
                    if out_tokens.is_empty() && ttft_ms == 0.0 {
                        ttft_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                    }
                    cur_pos += 1;
                    cur_tok = next_tok;
                    full_seq.push(next_tok);

                    if self.cfg.stop_ids.contains(&next_tok) {
                        stop_reason = "stop";
                        stop_hit = true;
                        unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                        break;
                    }
                    out_tokens.push(next_tok);
                    if !on_token(next_tok) || external_cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                        user_aborted = true;
                        stop_reason = "cancelled";
                        unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                        break;
                    }

                    // If speculative mode is enabled and an N-gram draft just became available with >=3 tokens left,
                    // signal the persistent megakernel to yield cleanly after the current token so `BM=4` takes over!
                    if use_spec && (max_steps_cap - out_tokens.len()) >= 3 {
                        ngram.update(&full_seq);
                        if ngram.lookup(&full_seq).is_some() {
                            unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                        }
                    }
                }

                let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
                if cur_sig >= sig_val {
                    // Drain any final token published right before QMD release0!
                    let final_steps = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).steps_completed) } as usize;
                    while streamed < final_steps && !stop_hit && !user_aborted {
                        let next_tok = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).tokens_out[streamed]) };
                        streamed += 1;
                        if out_tokens.is_empty() && ttft_ms == 0.0 {
                            ttft_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        cur_pos += 1;
                        cur_tok = next_tok;
                        full_seq.push(next_tok);
                        if self.cfg.stop_ids.contains(&next_tok) {
                            stop_reason = "stop";
                            stop_hit = true;
                            break;
                        }
                        out_tokens.push(next_tok);
                        if !on_token(next_tok) {
                            user_aborted = true;
                            stop_reason = "cancelled";
                            break;
                        }
                        if use_spec {
                            ngram.update(&full_seq);
                        }
                    }
                    break;
                }

                let err = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).error_code) };
                if err != 0 {
                    let _ = self.gpu.wait_timeline(sig_val, Duration::from_secs(1));
                    let _ = self.gpu.zero_vram(&self.ws_bar_state);
                    return Err(format!("GPU megakernel barrier watchdog tripped at step {}", err));
                }
                self.gpu.poll_gsp_faults()?;
                if rollout_t0.elapsed() > Duration::from_secs(120) {
                    unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                    self.gpu.poll_gsp_faults()?;
                    return Err("Persistent megakernel rollout exceeded 120s timeout".into());
                }
                std::hint::spin_loop();
            }

            if stop_hit || user_aborted {
                break;
            }
        }

        // Record KV-valid prefix (`full_seq[..cur_pos]`, i.e. all tokens except the last un-embedded `cur_tok`)
        self.cached_kv_tokens.clear();
        self.cached_kv_tokens.extend_from_slice(&full_seq[..cur_pos]);

        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;
        if ttft_ms == 0.0 {
            ttft_ms = total_ms;
        }
        let comp_toks = out_tokens.len();
        let decode_tps = if decode_ms > 0.0 {
            (comp_toks as f64) / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let prefill_tps = if prefill_ms > 0.0 && prefill_new_tokens > 0 {
            (prefill_new_tokens as f64) / (prefill_ms / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            prompt_tokens: prompt_len,
            cached_prefix_tokens,
            prefill_new_tokens,
            completion_tokens: comp_toks,
            prefill_ms,
            decode_ms,
            ttft_ms,
            total_ms,
            decode_tok_per_s: decode_tps,
            prefill_tok_per_s: prefill_tps,
            kernel_launches: launches,
            spec_verify_calls,
            spec_accepted_extra,
            stop_reason,
        };
        Ok((out_tokens, stats))
    }
}
