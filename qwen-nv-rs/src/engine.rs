//! Native GA102 Persistent Megakernel & Speculative `BM=4` Qwen Inference Engine.
//! Embeds pre-compiled `qwen_sm86_megakernel.cubin` directly into the Rust binary (zero runtime Docker/nvcc).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant, UNIX_EPOCH};

use crate::asr::{AsrStats, QwenAsrEngine};
use crate::drafter::{CalibratedNgramPrior, QwenNeuralDrafter};
use crate::gguf::{
    decode_1d_f32, extract_quant_weight_bytes, precompute_freqs_cis_f32, resolve_model_path, BpeTokenizer, GgufFile,
    QwenModelConfig,
};
use crate::lora::{
    prepare_lora_from_path, LoraStatus, QwenLayerLoraDesc, LORA_ARENA_DEFAULT_BYTES,
};
use crate::nv_defs::{put_u32, put_u64, QmdV3};
use crate::nv_hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
use crate::rag::{f32_to_bf16_bits, EmbedStrategy, GpuVectorDb, RagDoc, RagHit, RagSearchStats};
use crate::topic_lora::{TopicBankManifest, TopicEntry};
use crate::tts::{QwenTtsEngine, TtsStats, TtsSynthParams};

pub static EMBEDDED_CUBIN: &[u8] = include_bytes!("../kernels/qwen_sm86_megakernel.cubin");

const PREFILL_MAX_CHUNK: usize = 64;
const EMBED_MAX_POOL_ROWS: usize = 512;
const TOPIC_LORA_SLOT_BYTES: u64 = 8 << 20; // 8 MB per cached topic LoRA slot
pub const TOPIC_LORA_NUM_SLOTS: usize = 8; // 64 MB single one-time allocation (0 VRAM leaks)
const MLP_KV_HOST_BUF_BYTES: usize = 16 << 20; // 16 MB snooped host buffer (0 extra GDDR6X VRAM)

#[derive(Clone, Debug)]
pub struct TopicRouteResult {
    pub matched: bool,
    pub topic: Option<TopicEntry>,
    pub score: f32,
    pub lora_dir: String,
    pub system_prompt: String,
    pub embed_ms: f64,
    pub gpu_search_us: f64,
    pub swap_ms: f64,
    pub hits: Vec<RagHit>,
}

#[derive(Clone)]
struct CachedTopicSlot {
    path: String,
    user_scale: f32,
    fingerprint: (u64, u128),
    last_used: u64,
    descs: Vec<QwenLayerLoraDesc>,
    ptrs: Vec<u64>,
    status: LoraStatus,
}

struct MlpKvHookSpec<'a> {
    key_layers: &'a [usize],
    row_start: usize,
    n_rows: usize,
    keys_va: u64,
    final_res_va: u64,
}

fn adapter_file_fingerprint(path_str: &str) -> (u64, u128) {
    let p = Path::new(path_str);
    let st_path = if p.is_dir() {
        p.join("adapter_model.safetensors")
    } else {
        p.to_path_buf()
    };
    let Ok(md) = std::fs::metadata(&st_path) else {
        return (0, 0);
    };
    let mtime_ns = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    (md.len(), mtime_ns)
}

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

pub const DEFAULT_SPEC_CROSSOVER_4B: usize = 1536;
pub const DEFAULT_SPEC_CROSSOVER_8B: usize = 3072;

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

    #[inline]
    fn lookup_4gram_only(&self, seq: &[i32]) -> Option<[i32; 3]> {
        let n = seq.len();
        if n >= 4 {
            self.map4
                .get(&(seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1]))
                .copied()
        } else {
            None
        }
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

    // Dynamic Rank-r (`r <= 32`) LoRA arena & co-resident 100%-GPU TTS + ASR + RAG VectorDB engines
    lora_arena: Option<VramBuffer>,
    lora_descs_host: Vec<QwenLayerLoraDesc>,
    cached_lora_state: Option<(String, f32, Vec<QwenLayerLoraDesc>, Vec<u64>, LoraStatus)>,
    topic_lora_pool: Option<VramBuffer>,
    topic_lora_slots: Vec<Option<CachedTopicSlot>>,
    topic_lora_clock: u64,
    pub lora_status: LoraStatus,
    pub tts: Option<QwenTtsEngine>,
    pub asr: Option<QwenAsrEngine>,
    pub rag_db: Option<GpuVectorDb>,
    pub topic_bank: Option<(String, TopicBankManifest)>,
    pub drafter: Option<QwenNeuralDrafter>,

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
    ws_embed_pool: VramBuffer,
    ws_pool_acc: VramBuffer,
    ws_pool_norm: VramBuffer,
    zero_bf16_gpu: VramBuffer,
    freqs_cis_gpu: VramBuffer,
    output_norm_va: u64,
    unit_l2_va: u64,
    emb_gpu_va: u64,
    emb_type: i32,

    ctrl_buf: HostMappedBuffer,
    prefill_toks_buf: HostMappedBuffer,
    embed_out_buf: HostMappedBuffer,
    embed_in_buf: HostMappedBuffer,
    logits_out_buf: HostMappedBuffer,
    hooks_out_buf: HostMappedBuffer,
    mlp_kv_buf: HostMappedBuffer,
    cached_kv_tokens: Vec<i32>,
    pub spec_crossover_pos: usize,
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
        let unit_l2_vec = vec![1.0f32 / (cfg.dim as f32).sqrt(); cfg.dim];
        let unit_l2_rel = push_f32_arena(&mut small_arena_bytes, &unit_l2_vec);

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
        let ws_embed_pool = gpu.alloc_vram((EMBED_MAX_POOL_ROWS * cfg.dim * 4) as u64)?;
        let ws_pool_acc = gpu.alloc_vram(((cfg.dim * 4) as u64).max(4096))?;
        let ws_pool_norm = gpu.alloc_vram(((cfg.dim * 4) as u64).max(4096))?;
        let zero_bf16_gpu = gpu.alloc_vram(((cfg.dim * 4 * 2) as u64).max(4096))?;
        gpu.zero_vram(&zero_bf16_gpu)?;

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

        let ctrl_buf = gpu.alloc_host_mapped(std::mem::size_of::<QwenDecodeCtrl>())?;
        let prefill_toks_buf = gpu.alloc_host_mapped(PREFILL_MAX_CHUNK * 4)?;
        let embed_out_buf = gpu.alloc_host_mapped((cfg.dim * 4).max(4096))?;
        let embed_in_buf = gpu.alloc_host_mapped((cfg.dim * 4).max(4096))?;
        let logits_out_buf = gpu.alloc_host_mapped((cfg.vocab_size * 4).max(4096))?;
        let hooks_out_buf = gpu.alloc_host_mapped(((1 + 2 * cfg.num_layers) * cfg.dim * 4).max(4096))?;
        let mlp_kv_buf = gpu.alloc_host_mapped(MLP_KV_HOST_BUF_BYTES)?;

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
            ws_hidden_last: embed_out_buf.va_addr,
            ws_logits_out: logits_out_buf.va_addr,
            _pad_model: 0,
        };
        let mdesc_bytes = unsafe {
            std::slice::from_raw_parts(&mdesc as *const QwenModelDesc as *const u8, std::mem::size_of::<QwenModelDesc>())
        };
        let model_desc_gpu = gpu.alloc_vram(4096)?;
        gpu.upload_to_vram(&model_desc_gpu, 0, mdesc_bytes)?;

        let output_norm_va = arena_va + out_norm_rel;
        let unit_l2_va = arena_va + unit_l2_rel;

        let dma_gbps = (total_weight_bytes as f64 / 1e9) / t_upload.elapsed().as_secs_f64();
        eprintln!(
            "[qwen-nv] Loaded '{}' ({:.2} GB weights @ {:.2} GB/s DMA, {} layers, ctx={}, thinking_tmpl={}) in {:.2}s",
            cfg.model_name,
            total_weight_bytes as f64 / 1e9,
            dma_gbps,
            cfg.num_layers,
            cfg.max_ctx,
            cfg.thinking_template,
            t0.elapsed().as_secs_f64()
        );

        let num_layers_cfg = cfg.num_layers;
        let default_crossover = if cfg.dim > 2560 {
            DEFAULT_SPEC_CROSSOVER_8B
        } else {
            DEFAULT_SPEC_CROSSOVER_4B
        };

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
            cached_lora_state: None,
            topic_lora_pool: None,
            topic_lora_slots: vec![None; TOPIC_LORA_NUM_SLOTS],
            topic_lora_clock: 0,
            lora_status: LoraStatus::default(),
            tts: None,
            asr: None,
            rag_db: None,
            topic_bank: None,
            drafter: None,
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
            ws_embed_pool,
            ws_pool_acc,
            ws_pool_norm,
            zero_bf16_gpu,
            freqs_cis_gpu,
            output_norm_va,
            unit_l2_va,
            emb_gpu_va,
            emb_type,
            ctrl_buf,
            prefill_toks_buf,
            embed_out_buf,
            embed_in_buf,
            logits_out_buf,
            hooks_out_buf,
            mlp_kv_buf,
            cached_kv_tokens: Vec::new(),
            spec_crossover_pos: std::env::var("QWEN_NV_SPEC_CROSSOVER")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(default_crossover),
            rng_counter: 0x2026_1118_cafe_babe,
        })
    }

    pub fn reset_cache(&mut self) {
        self.cached_kv_tokens.clear();
    }

    // -------------------------------------------------------------------------
    // Dynamic Fused Rank-r (`r <= 32`) LoRA Hot-Swap (`< 1 ms` VRAM update!)
    // -------------------------------------------------------------------------

    fn upload_layers_host_to_gpu(&mut self) -> Result<(), String> {
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
        self.gpu.upload_to_vram(&layers_buf, 0, layers_bytes)
    }

    pub fn lora_cache_summary(&self) -> (usize, usize, bool) {
        let used_slots = self.topic_lora_slots.iter().filter(|s| s.is_some()).count();
        (used_slots, TOPIC_LORA_NUM_SLOTS, self.cached_lora_state.is_some())
    }

    pub fn load_lora(&mut self, path: &str, user_scale: f32) -> Result<LoraStatus, String> {
        let st_fp = adapter_file_fingerprint(path);

        // 1. Check `topic_lora_slots` LRU cache (`<= 8 MB` adapters, ~44 us pointer flip!)
        if self.topic_lora_pool.is_some() {
            if let Some(hit_idx) = self.topic_lora_slots.iter().position(|slot_opt| {
                slot_opt.as_ref().map_or(false, |s| {
                    s.path == path
                        && (s.user_scale - user_scale).abs() < 1e-6
                        && s.fingerprint == st_fp
                })
            }) {
                let t_swap = Instant::now();
                self.topic_lora_clock = self.topic_lora_clock.wrapping_add(1);
                let clk = self.topic_lora_clock;
                let (ptrs, descs, mut st) = {
                    let slot = self.topic_lora_slots[hit_idx].as_mut().unwrap();
                    slot.last_used = clk;
                    (slot.ptrs.clone(), slot.descs.clone(), slot.status.clone())
                };
                for l in 0..self.cfg.num_layers {
                    self.layers_host[l].lora_ptr = ptrs[l];
                }
                self.lora_descs_host = descs;
                self.upload_layers_host_to_gpu()?;
                self.reset_cache();
                st.active = true;
                st.swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                self.lora_status = st.clone();
                return Ok(st);
            }
        }

        // 2. Check pinned large-adapter cache (`> 8 MB`, e.g. `126 MB` Retrieval SVD LoRA)
        if self.lora_arena.is_some() {
            let cached_hit = self.cached_lora_state.as_ref().and_then(
                |(c_path, c_scale, c_descs, c_ptrs, c_meta)| {
                    if c_path == path && (*c_scale - user_scale).abs() < 1e-6 {
                        Some((c_descs.clone(), c_ptrs.clone(), c_meta.clone()))
                    } else {
                        None
                    }
                },
            );
            if let Some((c_descs, c_ptrs, mut st)) = cached_hit {
                let t_swap = Instant::now();
                for l in 0..self.cfg.num_layers {
                    self.layers_host[l].lora_ptr = c_ptrs[l];
                }
                self.lora_descs_host = c_descs;
                self.upload_layers_host_to_gpu()?;
                self.reset_cache();
                st.active = true;
                st.swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                self.lora_status = st.clone();
                return Ok(st);
            }
        }

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

        let desc_sz = std::mem::size_of::<QwenLayerLoraDesc>();
        let patch_and_set_ptrs = |layer_descs: &mut [QwenLayerLoraDesc],
                                  arena_bytes: &mut [u8],
                                  layers_host: &mut [QwenLayerDesc],
                                  arena_va: u64| {
            for (l, ld_lora) in layer_descs.iter_mut().enumerate() {
                for p in &mut ld_lora.projs {
                    if p.rank > 0 {
                        p.a_f16 += arena_va;
                        p.b_f16 += arena_va;
                    }
                }
                let src = unsafe {
                    std::slice::from_raw_parts(ld_lora as *const QwenLayerLoraDesc as *const u8, desc_sz)
                };
                arena_bytes[l * desc_sz..(l + 1) * desc_sz].copy_from_slice(src);
                layers_host[l].lora_ptr = if ld_lora.active_mask != 0 {
                    arena_va + (l * desc_sz) as u64
                } else {
                    0
                };
            }
        };

        // 3A. Small Topic LoRA (`<= 8 MB`): upload into pre-allocated `topic_lora_pool` slot without evicting `lora_arena`!
        if (prep.arena_bytes.len() as u64) <= TOPIC_LORA_SLOT_BYTES {
            if self.topic_lora_pool.is_none() {
                self.topic_lora_pool = Some(
                    self.gpu
                        .alloc_vram(TOPIC_LORA_SLOT_BYTES * (TOPIC_LORA_NUM_SLOTS as u64))?,
                );
            }
            let slot_idx = if let Some(empty_i) = self.topic_lora_slots.iter().position(|s| s.is_none()) {
                empty_i
            } else {
                self.topic_lora_slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.as_ref().map_or(0, |c| c.last_used))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };

            let t_swap = Instant::now();
            let pool_ref = self.topic_lora_pool.as_ref().unwrap();
            let slot_offset = (slot_idx as u64) * TOPIC_LORA_SLOT_BYTES;
            let arena_va = pool_ref.va_addr + slot_offset;
            let pool_buf = VramBuffer {
                va_addr: pool_ref.va_addr,
                size: pool_ref.size,
                paddrs: pool_ref.paddrs.clone(),
            };

            patch_and_set_ptrs(
                &mut prep.layer_descs,
                &mut prep.arena_bytes,
                &mut self.layers_host,
                arena_va,
            );
            self.gpu.upload_to_vram(&pool_buf, slot_offset, &prep.arena_bytes)?;
            self.upload_layers_host_to_gpu()?;
            self.reset_cache();

            let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
            let saved_ptrs: Vec<u64> = self.layers_host.iter().map(|ld| ld.lora_ptr).collect();
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
            self.topic_lora_clock = self.topic_lora_clock.wrapping_add(1);
            self.topic_lora_slots[slot_idx] = Some(CachedTopicSlot {
                path: path.to_string(),
                user_scale,
                fingerprint: st_fp,
                last_used: self.topic_lora_clock,
                descs: self.lora_descs_host.clone(),
                ptrs: saved_ptrs,
                status: self.lora_status.clone(),
            });
            return Ok(self.lora_status.clone());
        }

        // 3B. Large adapter (`> 8 MB`, e.g. `126 MB` Retrieval SVD LoRA): upload to `self.lora_arena`
        let need_bytes = (prep.arena_bytes.len() as u64).max(LORA_ARENA_DEFAULT_BYTES);
        let need_realloc = self.lora_arena.as_ref().map_or(true, |b| b.size < need_bytes);
        if need_realloc {
            self.lora_arena = Some(self.gpu.alloc_vram(need_bytes)?);
        }
        let t_swap = Instant::now();
        let arena_va = self.lora_arena.as_ref().unwrap().va_addr;
        patch_and_set_ptrs(
            &mut prep.layer_descs,
            &mut prep.arena_bytes,
            &mut self.layers_host,
            arena_va,
        );

        let arena_ref = self.lora_arena.as_ref().unwrap();
        let arena_buf = VramBuffer {
            va_addr: arena_ref.va_addr,
            size: arena_ref.size,
            paddrs: arena_ref.paddrs.clone(),
        };
        self.gpu.upload_to_vram(&arena_buf, 0, &prep.arena_bytes)?;
        self.upload_layers_host_to_gpu()?;
        self.reset_cache();

        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        let saved_ptrs: Vec<u64> = self.layers_host.iter().map(|ld| ld.lora_ptr).collect();
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
        self.cached_lora_state = Some((
            path.to_string(),
            user_scale,
            self.lora_descs_host.clone(),
            saved_ptrs,
            self.lora_status.clone(),
        ));
        Ok(self.lora_status.clone())
    }

    pub fn clear_lora(&mut self) -> Result<LoraStatus, String> {
        let t_swap = Instant::now();
        for l in 0..self.cfg.num_layers {
            self.layers_host[l].lora_ptr = 0;
            self.lora_descs_host[l] = QwenLayerLoraDesc::default();
        }
        self.upload_layers_host_to_gpu()?;
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
        let mut user_no_think = false;
        for (idx, (role, content)) in messages.iter().enumerate() {
            ids.extend(self.tok.role_tokens(role));
            if idx + 1 == messages.len() && role == "user" && content.trim_end().ends_with("/no_think") {
                user_no_think = true;
                let cleaned = content.trim_end().trim_end_matches("/no_think").trim_end();
                ids.extend(self.tok.encode(cleaned));
            } else {
                ids.extend(self.tok.encode(content));
            }
            ids.extend(self.tok.end_turn_tokens(self.cfg.im_end_id));
        }
        ids.extend(self.tok.role_tokens("assistant"));
        let lower_name = self.cfg.model_name.to_ascii_lowercase();
        if self.cfg.thinking_template {
            if enable_thinking && !user_no_think {
                ids.extend(self.tok.encode("<think>\n"));
            } else {
                ids.extend(self.tok.encode("<think>\n\n</think>\n\n"));
            }
        } else if (!enable_thinking || user_no_think) && lower_name.contains("qwen3") && !lower_name.contains("instruct") {
            ids.extend(self.tok.encode("<think>\n\n</think>\n\n"));
        }
        ids
    }

    pub fn ensure_drafter_loaded(&mut self, draft_ref: &str) -> Result<(), String> {
        let eff_ref = if draft_ref.trim().is_empty() {
            "qwen3:0.6b"
        } else {
            draft_ref.trim()
        };
        if let Some(ref existing) = self.drafter {
            if existing.model_ref == eff_ref {
                return Ok(());
            }
        }
        let d = QwenNeuralDrafter::load(&mut self.gpu, eff_ref, self.cfg.max_ctx)?;
        self.drafter = Some(d);
        Ok(())
    }

    pub fn clear_drafter(&mut self) {
        self.drafter = None;
    }

    pub fn load_drafter_prior(&mut self, prior_path: &std::path::Path) -> Result<(usize, usize), String> {
        if self.drafter.is_none() {
            self.ensure_drafter_loaded("qwen3:0.6b")?;
        }
        let prior = CalibratedNgramPrior::load_bin(prior_path)?;
        let n5 = prior.map5.len();
        let n4 = prior.map4.len();
        self.drafter.as_mut().unwrap().calibrated_prior = Some(prior);
        Ok((n5, n4))
    }

    fn common_prefix_len(&self, prompt_without_last: &[i32]) -> usize {
        prompt_without_last
            .iter()
            .zip(self.cached_kv_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Hardware-chained QMD prefill for `chunk` starting at `start_pos` (`1` doorbell per chunk!).
    /// - When `capture_norm_out_va` is `Some(dst_va)`, Layer `num_layers - 1` writes `output_norm(ws_x_res) [t_len, dim]`.
    /// - When `hook_last_token_va` is `Some(h_va)`, streams `[1 + 2*num_layers, dim]` residual snapshots of the final token
    ///   (`t_len - 1`) layer-by-layer to `h_va` in snooped host memory (`0` extra VRAM!).
    fn run_prefill_chunk(
        &mut self,
        chunk: &[i32],
        start_pos: usize,
        capture_norm_out_va: Option<u64>,
        hook_last_token_va: Option<u64>,
        mlp_kv_hook: Option<&MlpKvHookSpec>,
    ) -> Result<(), String> {
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

        let last_row_byte_off = ((t_len - 1) * (dim as usize) * 4) as u64;
        let slot_bytes = ((dim as usize) * 4) as u64;
        let k_copy_f32 = &self.module.kernels["tts_copy_f32"];
        if let Some(h_va) = hook_last_token_va {
            let mut a_cp = [0u8; 20];
            put_u64(&mut a_cp, 0, h_va);
            put_u64(&mut a_cp, 8, self.ws_x_res.va_addr + last_row_byte_off);
            put_u32(&mut a_cp, 16, dim as u32);
            chain.push(self.gpu.prepare_qmd(
                k_copy_f32,
                (((dim as u32) + 255) / 256, 1, 1),
                (256, 1, 1),
                &a_cp,
                None,
            ));
        }

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
            if is_last_layer
                && capture_norm_out_va.is_none()
                && hook_last_token_va.is_none()
                && mlp_kv_hook.is_none()
            {
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
            if let Some(h_va) = hook_last_token_va {
                let mut a_cp = [0u8; 20];
                put_u64(&mut a_cp, 0, h_va + ((2 * l + 1) as u64) * slot_bytes);
                put_u64(&mut a_cp, 8, self.ws_x_res2.va_addr + last_row_byte_off);
                put_u32(&mut a_cp, 16, dim as u32);
                chain.push(self.gpu.prepare_qmd(
                    k_copy_f32,
                    (((dim as u32) + 255) / 256, 1, 1),
                    (256, 1, 1),
                    &a_cp,
                    None,
                ));
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

            if let Some(spec) = mlp_kv_hook {
                if let Some(kl_idx) = spec.key_layers.iter().position(|&kl| kl == l) {
                    let k_c2w_swiglu = &self.module.kernels["c2w_swiglu_f32_kernel"];
                    let out_va = spec.keys_va + ((kl_idx * spec.n_rows * (hidden_dim as usize) * 4) as u64);
                    let gate_up_va = self.ws_gate_up.va_addr + ((spec.row_start * 2 * (hidden_dim as usize) * 4) as u64);
                    let tot_sw = (spec.n_rows * (hidden_dim as usize)) as u32;
                    let mut a_sw_f32 = [0u8; 24];
                    put_u64(&mut a_sw_f32, 0, out_va);
                    put_u64(&mut a_sw_f32, 8, gate_up_va);
                    put_u32(&mut a_sw_f32, 16, spec.n_rows as u32);
                    put_u32(&mut a_sw_f32, 20, hidden_dim as u32);
                    chain.push(self.gpu.prepare_qmd(
                        k_c2w_swiglu,
                        ((tot_sw + 255) / 256, 1, 1),
                        (256, 1, 1),
                        &a_sw_f32,
                        None,
                    ));
                }
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
            if is_last_layer {
                if let Some(spec) = mlp_kv_hook {
                    let n_el = (spec.n_rows * (dim as usize)) as u32;
                    let src_va = self.ws_x_res.va_addr + ((spec.row_start * (dim as usize) * 4) as u64);
                    let mut a_cp = [0u8; 20];
                    put_u64(&mut a_cp, 0, spec.final_res_va);
                    put_u64(&mut a_cp, 8, src_va);
                    put_u32(&mut a_cp, 16, n_el);
                    chain.push(self.gpu.prepare_qmd(
                        k_copy_f32,
                        ((n_el + 255) / 256, 1, 1),
                        (256, 1, 1),
                        &a_cp,
                        None,
                    ));
                }
            }
            if let Some(h_va) = hook_last_token_va {
                let mut a_cp = [0u8; 20];
                put_u64(&mut a_cp, 0, h_va + ((2 * l + 2) as u64) * slot_bytes);
                put_u64(&mut a_cp, 8, self.ws_x_res.va_addr + last_row_byte_off);
                put_u32(&mut a_cp, 16, dim as u32);
                chain.push(self.gpu.prepare_qmd(
                    k_copy_f32,
                    (((dim as u32) + 255) / 256, 1, 1),
                    (256, 1, 1),
                    &a_cp,
                    None,
                ));
            }
        }

        if let Some(norm_dst_va) = capture_norm_out_va {
            let k_c2w_rms = &self.module.kernels["c2w_rmsnorm_f32_kernel"];
            let mut a_nrm = [0u8; 36];
            put_u64(&mut a_nrm, 0, norm_dst_va);
            put_u64(&mut a_nrm, 8, self.ws_x_res.va_addr);
            put_u64(&mut a_nrm, 16, self.output_norm_va);
            put_u32(&mut a_nrm, 24, t_len as u32);
            put_u32(&mut a_nrm, 28, dim as u32);
            put_u32(&mut a_nrm, 32, eps_bits);
            chain.push(self.gpu.prepare_qmd(
                k_c2w_rms,
                (t_len as u32, 1, 1),
                (256, 1, 1),
                &a_nrm,
                None,
            ));
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
            self.run_prefill_chunk(&prompt_tokens[pos..pos + take], pos, None, None, None)?;
            pos += take;
            launches += 1;
        }
        let prefill_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        // Track full sequence (`prompt + generated`) for KV prefix cache & N-gram speculative drafting
        let mut full_seq = prompt_tokens.to_vec();
        let mut cur_tok = prompt_tokens[prompt_len - 1];
        let mut cur_pos = prompt_len - 1;

        let use_spec = enable_speculative && temperature <= 1e-6;
        let eff_crossover = if self.spec_crossover_pos == 0 {
            usize::MAX
        } else {
            self.spec_crossover_pos
        };
        let mut ngram = NgramDraftTable::new();
        if use_spec && cur_pos < eff_crossover {
            ngram.update(&full_seq);
        }

        let mut out_tokens = Vec::with_capacity(max_steps_cap);
        let mut ttft_ms = 0.0f64;
        let mut stop_reason: &'static str = "length";
        let mut spec_verify_calls = 0usize;
        let mut spec_accepted_extra = 0usize;
        let mut spec_crossover_switched = false;
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
            if use_spec && cur_pos >= eff_crossover {
                spec_crossover_switched = true;
            }

            // Path A: Lossless BM=4 Speculative Verification (active below `eff_crossover`, e.g. 1,536 tokens)!
            // Once sequence length reaches `eff_crossover`, automatically hand off to Path B's fused 68-SM persistent T=1 megakernel!
            if use_spec && cur_pos < eff_crossover && rem >= 2 && cur_pos + 4 < self.cfg.max_ctx + PREFILL_MAX_CHUNK {
                let draft_opt = if let Some(ref mut dr) = self.drafter {
                    if dr.calibrated_prior.is_some() {
                        if let Some(d4) = ngram.lookup_4gram_only(&full_seq) {
                            Some(d4)
                        } else {
                            dr.draft_k3(&mut self.gpu, &self.module, &full_seq).ok()
                        }
                    } else {
                        dr.draft_k3(&mut self.gpu, &self.module, &full_seq).ok()
                    }
                } else {
                    ngram.lookup(&full_seq)
                };
                if let Some(draft) = draft_opt {
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

                    // If speculative mode is enabled, we are still below the BM=4->T=1 crossover threshold,
                    // and an N-gram draft just became available with >=3 tokens left, signal the persistent
                    // megakernel to yield cleanly after the current token so `BM=4` takes over!
                    if use_spec && cur_pos < eff_crossover && (max_steps_cap - out_tokens.len()) >= 3 {
                        ngram.update(&full_seq);
                        if ngram.lookup(&full_seq).is_some() {
                            unsafe { std::ptr::write_volatile(&mut (*ctrl_ptr).cancel_flag, 1) };
                        }
                    } else if use_spec && cur_pos >= eff_crossover {
                        spec_crossover_switched = true;
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
            spec_crossover_pos: self.spec_crossover_pos,
            spec_crossover_switched,
            stop_reason,
        };
        Ok((out_tokens, stats))
    }

    // -------------------------------------------------------------------------
    // 100%-on-RTX-3080 Same-Model Embeddings & `.qvec` GPU VectorDB (`Phase 5`)
    // -------------------------------------------------------------------------

    fn execute_embed_post_chain(&mut self, chain: &[(u64, usize)]) -> Result<(), String> {
        self.gpu.timeline_value += 1;
        let target_sig = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        {
            let (_, last_off) = *chain.last().unwrap();
            let buf = self.gpu.kernargs_buf_mut_slice_at(last_off);
            let mut tail_qmd = QmdV3 {
                bytes: buf[..256].try_into().unwrap(),
            };
            tail_qmd.set_release0(sig_va, target_sig);
            buf[..256].copy_from_slice(&tail_qmd.bytes);
        }
        for i in 0..(chain.len() - 1) {
            self.gpu.link_dependent_qmd(chain[i].1, chain[i + 1].0);
        }
        self.gpu.launch_qmd_chain(chain[0].0, true)?;
        self.gpu.wait_timeline(target_sig, Duration::from_secs(10))
    }

    /// Extract a single `L2`-normalized `[dim]` embedding vector **100.0% on the RTX 3080**:
    /// - Runs hardware-chained prefill on `token_ids` (capturing `output_norm` on `[pool_start .. T)`),
    /// - Sums/averages pooled token rows on GPU (`c2w_linear_bf16_kernel` + `tts_copy_f32`),
    /// - `L2`-normalizes on GPU (`c2w_rmsnorm_f32_kernel` with `unit_l2_va`),
    /// - Optionally centers by `neg_mean_va` (`-mu`) and `L2`-renormalizes on GPU,
    /// - Writes the final `[dim]` `f32` vector directly into `self.embed_out_buf`.
    pub fn embed_tokens_on_gpu(
        &mut self,
        token_ids: &[i32],
        pool_start: usize,
        neg_mean_va: Option<u64>,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("embed_tokens_on_gpu: empty token_ids".into());
        }
        let total_len = token_ids.len().min(self.cfg.max_ctx);
        let toks = &token_ids[..total_len];
        let eff_pool_start = pool_start.min(total_len - 1);
        let pool_len = (total_len - eff_pool_start).min(EMBED_MAX_POOL_ROWS);
        let actual_pool_start = total_len - pool_len;
        let dim = self.cfg.dim;
        let row_bytes = (dim * 4) as u64;

        let mut pos = 0usize;
        let mut captured_rows = 0usize;
        while pos < total_len {
            let rem = total_len - pos;
            // Split chunk right at `actual_pool_start` if a chunk would straddle the boundary,
            // so `ws_embed_pool[0 .. pool_len]` is 100% contiguous!
            let take = if pos < actual_pool_start {
                (actual_pool_start - pos).min(PREFILL_MAX_CHUNK)
            } else {
                rem.min(PREFILL_MAX_CHUNK)
            };

            let cap_va = if pos >= actual_pool_start {
                let va = self.ws_embed_pool.va_addr + (captured_rows as u64) * row_bytes;
                captured_rows += take;
                Some(va)
            } else {
                None
            };
            self.run_prefill_chunk(&toks[pos..pos + take], pos, cap_va, None, None)?;
            pos += take;
        }
        self.reset_cache();

        // Build 100%-GPU pooling + optional mean-centering + L2-unit-normalization QMD chain
        let k_copy = &self.module.kernels["tts_copy_f32"];
        let k_lin_bf16 = &self.module.kernels["c2w_linear_bf16_kernel"];
        let k_rms_f32 = &self.module.kernels["c2w_rmsnorm_f32_kernel"];
        let l2_eps_bits = (1e-12f32).to_bits();

        let mut post_chain = Vec::with_capacity(captured_rows + 4);

        let norm1_src_va = if captured_rows == 1 {
            self.ws_embed_pool.va_addr
        } else {
            // Copy row 0 -> ws_pool_acc, then add rows `1 .. captured_rows` on GPU
            let mut a_cp = [0u8; 20];
            put_u64(&mut a_cp, 0, self.ws_pool_acc.va_addr);
            put_u64(&mut a_cp, 8, self.ws_embed_pool.va_addr);
            put_u32(&mut a_cp, 16, dim as u32);
            post_chain.push(self.gpu.prepare_qmd(
                k_copy,
                (((dim as u32) + 255) / 256, 1, 1),
                (256, 1, 1),
                &a_cp,
                None,
            ));
            for r in 1..captured_rows {
                let row_va = self.ws_embed_pool.va_addr + (r as u64) * row_bytes;
                let mut a_add = [0u8; 60];
                put_u64(&mut a_add, 0, self.ws_pool_acc.va_addr);
                put_u64(&mut a_add, 8, self.zero_bf16_gpu.va_addr);
                put_u64(&mut a_add, 16, self.zero_bf16_gpu.va_addr);
                put_u64(&mut a_add, 24, row_va);
                put_u64(&mut a_add, 32, 0);
                put_u64(&mut a_add, 40, self.ws_pool_acc.va_addr);
                put_u32(&mut a_add, 48, 1);
                put_u32(&mut a_add, 52, dim as u32);
                put_u32(&mut a_add, 56, 4);
                post_chain.push(self.gpu.prepare_qmd(
                    k_lin_bf16,
                    (((dim as u32) + 3) / 4, 1, 1),
                    (128, 1, 1),
                    &a_add,
                    None,
                ));
            }
            self.ws_pool_acc.va_addr
        };

        if let Some(neg_mu_va) = neg_mean_va {
            // 1. Unit L2-normalize pooled vector into `ws_pool_norm`
            let mut a_l2_1 = [0u8; 36];
            put_u64(&mut a_l2_1, 0, self.ws_pool_norm.va_addr);
            put_u64(&mut a_l2_1, 8, norm1_src_va);
            put_u64(&mut a_l2_1, 16, self.unit_l2_va);
            put_u32(&mut a_l2_1, 24, 1);
            put_u32(&mut a_l2_1, 28, dim as u32);
            put_u32(&mut a_l2_1, 32, l2_eps_bits);
            post_chain.push(self.gpu.prepare_qmd(k_rms_f32, (1, 1, 1), (256, 1, 1), &a_l2_1, None));

            // 2. Subtract corpus mean (`ws_pool_acc = ws_pool_norm + (-mu)`)
            let mut a_sub = [0u8; 60];
            put_u64(&mut a_sub, 0, self.ws_pool_acc.va_addr);
            put_u64(&mut a_sub, 8, self.zero_bf16_gpu.va_addr);
            put_u64(&mut a_sub, 16, self.zero_bf16_gpu.va_addr);
            put_u64(&mut a_sub, 24, neg_mu_va);
            put_u64(&mut a_sub, 32, 0);
            put_u64(&mut a_sub, 40, self.ws_pool_norm.va_addr);
            put_u32(&mut a_sub, 48, 1);
            put_u32(&mut a_sub, 52, dim as u32);
            put_u32(&mut a_sub, 56, 4);
            post_chain.push(self.gpu.prepare_qmd(
                k_lin_bf16,
                (((dim as u32) + 3) / 4, 1, 1),
                (128, 1, 1),
                &a_sub,
                None,
            ));

            // 3. Re-normalize centered vector onto unit sphere -> `embed_out_buf`
            let mut a_l2_2 = [0u8; 36];
            put_u64(&mut a_l2_2, 0, self.embed_out_buf.va_addr);
            put_u64(&mut a_l2_2, 8, self.ws_pool_acc.va_addr);
            put_u64(&mut a_l2_2, 16, self.unit_l2_va);
            put_u32(&mut a_l2_2, 24, 1);
            put_u32(&mut a_l2_2, 28, dim as u32);
            put_u32(&mut a_l2_2, 32, l2_eps_bits);
            post_chain.push(self.gpu.prepare_qmd(k_rms_f32, (1, 1, 1), (256, 1, 1), &a_l2_2, None));
        } else {
            let mut a_l2 = [0u8; 36];
            put_u64(&mut a_l2, 0, self.embed_out_buf.va_addr);
            put_u64(&mut a_l2, 8, norm1_src_va);
            put_u64(&mut a_l2, 16, self.unit_l2_va);
            put_u32(&mut a_l2, 24, 1);
            put_u32(&mut a_l2, 28, dim as u32);
            put_u32(&mut a_l2, 32, l2_eps_bits);
            post_chain.push(self.gpu.prepare_qmd(k_rms_f32, (1, 1, 1), (256, 1, 1), &a_l2, None));
        }

        self.execute_embed_post_chain(&post_chain)?;
        fence(Ordering::SeqCst);

        let src_ptr = self.embed_out_buf.as_ptr() as *const f32;
        let mut out_vec = vec![0.0f32; dim];
        for i in 0..dim {
            out_vec[i] = unsafe { std::ptr::read_volatile(src_ptr.add(i)) };
        }
        Ok(out_vec)
    }

    /// Center & re-normalize an already `L2`-normalized `[dim]` vector on the RTX 3080 (`u -> L2_norm(u - mu)`).
    fn center_and_renorm_on_gpu(&mut self, raw_unit: &[f32], neg_mu_va: u64) -> Result<Vec<f32>, String> {
        let dim = self.cfg.dim;
        {
            let dst = self.embed_in_buf.as_mut_slice();
            for (i, &v) in raw_unit.iter().enumerate() {
                put_u32(dst, i * 4, v.to_bits());
            }
        }
        fence(Ordering::SeqCst);

        let k_lin_bf16 = &self.module.kernels["c2w_linear_bf16_kernel"];
        let k_rms_f32 = &self.module.kernels["c2w_rmsnorm_f32_kernel"];
        let l2_eps_bits = (1e-12f32).to_bits();

        let mut chain = Vec::with_capacity(2);
        let mut a_sub = [0u8; 60];
        put_u64(&mut a_sub, 0, self.ws_pool_acc.va_addr);
        put_u64(&mut a_sub, 8, self.zero_bf16_gpu.va_addr);
        put_u64(&mut a_sub, 16, self.zero_bf16_gpu.va_addr);
        put_u64(&mut a_sub, 24, neg_mu_va);
        put_u64(&mut a_sub, 32, 0);
        put_u64(&mut a_sub, 40, self.embed_in_buf.va_addr);
        put_u32(&mut a_sub, 48, 1);
        put_u32(&mut a_sub, 52, dim as u32);
        put_u32(&mut a_sub, 56, 4);
        chain.push(self.gpu.prepare_qmd(
            k_lin_bf16,
            (((dim as u32) + 3) / 4, 1, 1),
            (128, 1, 1),
            &a_sub,
            None,
        ));

        let mut a_l2 = [0u8; 36];
        put_u64(&mut a_l2, 0, self.embed_out_buf.va_addr);
        put_u64(&mut a_l2, 8, self.ws_pool_acc.va_addr);
        put_u64(&mut a_l2, 16, self.unit_l2_va);
        put_u32(&mut a_l2, 24, 1);
        put_u32(&mut a_l2, 28, dim as u32);
        put_u32(&mut a_l2, 32, l2_eps_bits);
        chain.push(self.gpu.prepare_qmd(k_rms_f32, (1, 1, 1), (256, 1, 1), &a_l2, None));

        self.execute_embed_post_chain(&chain)?;
        fence(Ordering::SeqCst);

        let src_ptr = self.embed_out_buf.as_ptr() as *const f32;
        let mut out_vec = vec![0.0f32; dim];
        for i in 0..dim {
            out_vec[i] = unsafe { std::ptr::read_volatile(src_ptr.add(i)) };
        }
        Ok(out_vec)
    }

    pub fn embed_text(
        &mut self,
        strategy: EmbedStrategy,
        text: &str,
        is_query: bool,
        lora_path_opt: Option<&str>,
    ) -> Result<(Vec<f32>, f64), String> {
        let t0 = Instant::now();
        let prev_lora = if self.lora_status.active {
            Some((self.lora_status.path.clone(), self.lora_status.scale))
        } else {
            None
        };

        if strategy == EmbedStrategy::SvdLora {
            let lp = lora_path_opt
                .filter(|s| !s.trim().is_empty())
                .ok_or("svd_lora strategy requires a LoRA adapter path")?;
            if prev_lora.as_ref().map(|(p, _)| p.as_str()) != Some(lp) {
                self.load_lora(lp, 1.0)?;
            }
        } else if prev_lora.is_some() && matches!(strategy, EmbedStrategy::Oracle) {
            self.clear_lora()?;
        }

        let neg_mu_va = if strategy.uses_corpus_centering() {
            self.rag_db
                .as_ref()
                .filter(|db| db.strategy == strategy)
                .and_then(|db| db.neg_mean_gpu.as_ref().map(|b| b.va_addr))
        } else {
            None
        };

        let max_tok_cap = self.cfg.max_ctx.min(1024);
        let (tok_ids, pool_start) = strategy.encode_input(
            &self.tok,
            self.cfg.eos_id,
            self.cfg.bos_id,
            text,
            is_query,
            max_tok_cap,
        );
        let res = self.embed_tokens_on_gpu(&tok_ids, pool_start, neg_mu_va);

        // Restore prior LoRA state (`<50 us` if toggling back to base Thinker)
        if strategy == EmbedStrategy::SvdLora {
            match prev_lora {
                Some((old_p, old_s)) => {
                    if self.lora_status.path != old_p {
                        let _ = self.load_lora(&old_p, old_s);
                    }
                }
                None => {
                    let _ = self.clear_lora();
                }
            }
        }

        let vec_out = res?;
        Ok((vec_out, t0.elapsed().as_secs_f64() * 1000.0))
    }

    /// Build a `.qvec` GPU VectorDB from `docs` on the RTX 3080, save to `out_path`, and optionally load it co-resident!
    pub fn build_qvec_index(
        &mut self,
        docs: &[RagDoc],
        strategy: EmbedStrategy,
        lora_path: &str,
        out_path: &std::path::Path,
        load_into_vram: bool,
    ) -> Result<(f64, f64), String> {
        let t_start = Instant::now();
        let prev_lora = if self.lora_status.active {
            Some((self.lora_status.path.clone(), self.lora_status.scale))
        } else {
            None
        };
        if strategy == EmbedStrategy::SvdLora {
            if lora_path.trim().is_empty() {
                return Err("build_qvec_index with svd_lora requires --lora <ADAPTER_PATH>".into());
            }
            self.load_lora(lora_path, 1.0)?;
        } else if self.lora_status.active {
            self.clear_lora()?;
        }

        let dim = self.cfg.dim;
        let max_tok_cap = self.cfg.max_ctx.min(1024);
        let mut raw_vecs = Vec::with_capacity(docs.len());

        for (idx, d) in docs.iter().enumerate() {
            // Include article title in the indexed passage so entity disambiguation is preserved across paragraphs!
            let full_passage = format!("{}: {}", d.title, d.text);
            let (tok_ids, pool_start) = strategy.encode_input(
                &self.tok,
                self.cfg.eos_id,
                self.cfg.bos_id,
                &full_passage,
                false,
                max_tok_cap,
            );
            let v = self.embed_tokens_on_gpu(&tok_ids, pool_start, None)?;
            raw_vecs.push(v);
            if (idx + 1) % 200 == 0 || idx + 1 == docs.len() {
                let elapsed = t_start.elapsed().as_secs_f64();
                eprintln!(
                    "[qwen-nv rag] Indexed {}/{} passages ({:.1} passages/s, strategy={})...",
                    idx + 1,
                    docs.len(),
                    ((idx + 1) as f64) / elapsed.max(1e-6),
                    strategy.as_str()
                );
            }
        }

        // Restore prior LoRA state immediately after prefill pass
        match prev_lora {
            Some((old_p, old_s)) => {
                let _ = self.load_lora(&old_p, old_s);
            }
            None => {
                if self.lora_status.active {
                    let _ = self.clear_lora();
                }
            }
        }

        let (mean_opt, final_vecs) = if strategy.uses_corpus_centering() && !raw_vecs.is_empty() {
            let inv_n = 1.0f64 / (raw_vecs.len() as f64);
            let mut mean_f64 = vec![0.0f64; dim];
            for v in &raw_vecs {
                for i in 0..dim {
                    mean_f64[i] += (v[i] as f64) * inv_n;
                }
            }
            let mean_f32: Vec<f32> = mean_f64.iter().map(|&x| x as f32).collect();
            let mut neg_bytes = Vec::with_capacity(dim * 4);
            for &m in &mean_f32 {
                neg_bytes.extend_from_slice(&(-m).to_bits().to_le_bytes());
            }
            let tmp_neg_gpu = self.gpu.alloc_vram((neg_bytes.len() as u64).max(4096))?;
            self.gpu.upload_to_vram(&tmp_neg_gpu, 0, &neg_bytes)?;

            let mut centered = Vec::with_capacity(raw_vecs.len());
            for v in &raw_vecs {
                centered.push(self.center_and_renorm_on_gpu(v, tmp_neg_gpu.va_addr)?);
            }
            (Some(mean_f32), centered)
        } else {
            (None, raw_vecs)
        };

        let mut bf16_flat = Vec::with_capacity(docs.len() * dim);
        for v in &final_vecs {
            for &x in v {
                bf16_flat.push(f32_to_bf16_bits(x));
            }
        }

        GpuVectorDb::save_qvec(
            out_path,
            strategy,
            lora_path,
            dim,
            mean_opt.as_deref(),
            &bf16_flat,
            docs,
        )?;

        let total_s = t_start.elapsed().as_secs_f64();
        let docs_per_s = (docs.len() as f64) / total_s.max(1e-6);

        if load_into_vram {
            let db = GpuVectorDb::load_qvec(&mut self.gpu, out_path, dim)?;
            self.rag_db = Some(db);
        }

        Ok((total_s * 1000.0, docs_per_s))
    }

    pub fn load_rag_qvec(&mut self, path: &std::path::Path) -> Result<(), String> {
        let mut eff_path = path.to_path_buf();
        if self.cfg.dim == 4096 {
            if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
                if fname == "wikipedia_1k_best.qvec" {
                    let alt = path.with_file_name("wikipedia_1k_8b_best.qvec");
                    if alt.exists() {
                        eff_path = alt;
                    }
                }
            }
        }
        let db = GpuVectorDb::load_qvec(&mut self.gpu, &eff_path, self.cfg.dim)?;
        // Pre-warm the paired Retrieval LoRA into `self.lora_arena` (then immediately suspend to `0`)
        // so every subsequent RAG query toggles LoRA ON/OFF in `<50 us`!
        if db.strategy == EmbedStrategy::SvdLora && !db.lora_path.is_empty() {
            let was_active = self.lora_status.active;
            let prev = (self.lora_status.path.clone(), self.lora_status.scale);
            self.load_lora(&db.lora_path, 1.0)?;
            if !was_active {
                self.clear_lora()?;
            } else if prev.0 != db.lora_path {
                self.load_lora(&prev.0, prev.1)?;
            }
        }
        self.rag_db = Some(db);
        Ok(())
    }

    pub fn clear_rag_qvec(&mut self) {
        self.rag_db = None;
    }

    pub fn rag_search(&mut self, query: &str, top_k: usize) -> Result<RagSearchStats, String> {
        let t_tot = Instant::now();
        let (strat, lora_p, num_docs) = {
            let db = self
                .rag_db
                .as_ref()
                .ok_or("No RAG .qvec index currently loaded (run `qwen-nv rag load <file.qvec>` or `/rag <file.qvec>`)")?;
            (db.strategy, db.lora_path.clone(), db.docs.len())
        };

        let (_, embed_ms) = self.embed_text(
            strat,
            query,
            true,
            if lora_p.is_empty() { None } else { Some(lora_p.as_str()) },
        )?;

        // `self.embed_out_buf` already holds the exact `[1, dim]` `F32` query vector in GPU-accessible DMA memory!
        let db = self.rag_db.as_ref().unwrap();
        let (hits, gpu_search_us) =
            db.search_cosine_on_gpu(&mut self.gpu, &self.module, self.embed_out_buf.va_addr, top_k)?;

        Ok(RagSearchStats {
            hits,
            embed_ms,
            gpu_search_us,
            total_ms: t_tot.elapsed().as_secs_f64() * 1000.0,
            strategy: strat.as_str().to_string(),
            num_docs,
        })
    }

    // -------------------------------------------------------------------------
    // Topic Parametric LoRA Bank Router
    // -------------------------------------------------------------------------

    pub fn load_topic_bank(&mut self, bank_dir: &Path) -> Result<(String, usize, usize), String> {
        let manifest = TopicBankManifest::load(bank_dir)?;
        let router_qvec = manifest.resolve_router_qvec_path();
        self.load_rag_qvec(&router_qvec)?;
        let n_docs = self.rag_db.as_ref().map(|d| d.docs.len()).unwrap_or(0);
        let n_topics = manifest.topics.len();
        let canonical = manifest.bank_dir.display().to_string();
        self.topic_bank = Some((canonical.clone(), manifest));
        Ok((canonical, n_topics, n_docs))
    }

    pub fn clear_topic_bank(&mut self) -> Result<(), String> {
        self.topic_bank = None;
        if self.lora_status.active {
            let _ = self.clear_lora()?;
        }
        Ok(())
    }

    pub fn topic_route(
        &mut self,
        query: &str,
        top_k: usize,
        min_score_override: Option<f32>,
        auto_load: bool,
    ) -> Result<TopicRouteResult, String> {
        let sys_prompt = self
            .topic_bank
            .as_ref()
            .ok_or("No Topic LoRA bank currently loaded (run `qwen-nv topic load <BANK_DIR>`)")?
            .1
            .system_prompt
            .clone();

        let search_st = self.rag_search(query, top_k.max(1))?;
        let resolved = {
            let (_, ref manifest) = self.topic_bank.as_ref().unwrap();
            manifest
                .resolve_topic(&search_st.hits, min_score_override)
                .map(|(entry, sc)| (entry.clone(), manifest.resolve_topic_lora_path(entry), sc))
        };

        if let Some((entry, lora_path, score)) = resolved {
            let lora_dir = lora_path.display().to_string();
            let mut swap_ms = 0.0f64;
            if auto_load {
                let st = self.load_lora(&lora_dir, 1.0)?;
                swap_ms = st.swap_ms;
            }
            Ok(TopicRouteResult {
                matched: true,
                topic: Some(entry),
                score,
                lora_dir,
                system_prompt: sys_prompt,
                embed_ms: search_st.embed_ms,
                gpu_search_us: search_st.gpu_search_us,
                swap_ms,
                hits: search_st.hits,
            })
        } else {
            let mut swap_ms = 0.0f64;
            if auto_load && self.lora_status.active {
                let st = self.clear_lora()?;
                swap_ms = st.swap_ms;
            }
            let top_score = search_st.hits.first().map(|h| h.score).unwrap_or(0.0);
            Ok(TopicRouteResult {
                matched: false,
                topic: None,
                score: top_score,
                lora_dir: String::new(),
                system_prompt: sys_prompt,
                embed_ms: search_st.embed_ms,
                gpu_search_us: search_st.gpu_search_us,
                swap_ms,
                hits: search_st.hits,
            })
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2 & Phase 4: 100%-on-RTX-3080 Residual-Writer Hooks & Full Logit Eval
    // -------------------------------------------------------------------------

    /// Stream residual-stream activations `[1 + 2*num_layers, dim]` at the final token of `token_ids`
    /// layer-by-layer from RTX 3080 to snooped host memory (`0` extra VRAM!).
    pub fn extract_residual_hooks(&mut self, token_ids: &[i32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("extract_residual_hooks: empty token_ids".into());
        }
        let total_len = token_ids.len().min(self.cfg.max_ctx);
        let toks = &token_ids[..total_len];
        let mut pos = 0usize;
        while pos < total_len {
            let take = (total_len - pos).min(PREFILL_MAX_CHUNK);
            let is_final_chunk = pos + take == total_len;
            let hook_va = if is_final_chunk {
                Some(self.hooks_out_buf.va_addr)
            } else {
                None
            };
            self.run_prefill_chunk(&toks[pos..pos + take], pos, None, hook_va, None)?;
            pos += take;
        }
        self.reset_cache();
        fence(Ordering::SeqCst);

        let n_floats = (1 + 2 * self.cfg.num_layers) * self.cfg.dim;
        let ptr = self.hooks_out_buf.as_ptr() as *const f32;
        let mut out = Vec::with_capacity(n_floats);
        for i in 0..n_floats {
            out.push(unsafe { std::ptr::read_volatile(ptr.add(i)) });
        }
        Ok(out)
    }

    /// Stream exact FP32 post-SwiGLU MLP keys (`[pos_count, key_layers.len(), hidden_dim]`)
    /// and final-layer residual states (`[pos_count, dim]`) for token range `pos_start .. pos_start + pos_count`
    /// using existing `c2w_swiglu_f32_kernel` + `tts_copy_f32` (`0` `.cu` edits, `0` extra VRAM).
    pub fn extract_mlp_kv_hooks(
        &mut self,
        token_ids: &[i32],
        key_layers: &[usize],
        pos_start: usize,
        pos_count: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if token_ids.is_empty() || pos_count == 0 {
            return Err("extract_mlp_kv_hooks: empty token_ids or pos_count=0".into());
        }
        let max_len = token_ids.len().min(self.cfg.max_ctx);
        if pos_start + pos_count > max_len {
            return Err(format!(
                "extract_mlp_kv_hooks: slice [{}..{}) exceeds token_ids len {}",
                pos_start,
                pos_start + pos_count,
                max_len
            ));
        }
        for &l in key_layers {
            if l >= self.cfg.num_layers {
                return Err(format!("extract_mlp_kv_hooks: layer {} >= {}", l, self.cfg.num_layers));
            }
        }

        let dim = self.cfg.dim;
        let hidden_dim = self.cfg.hidden_dim;
        let n_kl = key_layers.len();
        let bytes_per_row = (n_kl * hidden_dim + dim) * 4;
        let max_capture_rows = (self.mlp_kv_buf.size / bytes_per_row)
            .min(PREFILL_MAX_CHUNK)
            .max(1);

        self.reset_cache();

        // Phase A: Prefill prefix `[0 .. pos_start)` without capture
        let mut pos = 0usize;
        while pos < pos_start {
            let take = (pos_start - pos).min(PREFILL_MAX_CHUNK);
            self.run_prefill_chunk(&token_ids[pos..pos + take], pos, None, None, None)?;
            pos += take;
        }

        // Phase B: Prefill `[pos_start .. pos_start + pos_count)` capturing SwiGLU keys & h_last
        let mut keys_out = vec![0.0f32; pos_count * n_kl * hidden_dim];
        let mut res_out = vec![0.0f32; pos_count * dim];
        let mut captured = 0usize;

        while captured < pos_count {
            let cur_pos = pos_start + captured;
            let take = (pos_count - captured).min(max_capture_rows);
            let keys_va = self.mlp_kv_buf.va_addr;
            let final_res_va = keys_va + ((n_kl * take * hidden_dim * 4) as u64);
            let spec = MlpKvHookSpec {
                key_layers,
                row_start: 0,
                n_rows: take,
                keys_va,
                final_res_va,
            };
            self.run_prefill_chunk(
                &token_ids[cur_pos..cur_pos + take],
                cur_pos,
                None,
                None,
                Some(&spec),
            )?;
            fence(Ordering::SeqCst);

            let ptr = self.mlp_kv_buf.as_ptr() as *const f32;
            for r in 0..take {
                let out_p = captured + r;
                for kl_idx in 0..n_kl {
                    let src_off = (kl_idx * take + r) * hidden_dim;
                    let dst_off = (out_p * n_kl + kl_idx) * hidden_dim;
                    for c in 0..hidden_dim {
                        keys_out[dst_off + c] = unsafe { std::ptr::read_volatile(ptr.add(src_off + c)) };
                    }
                }
            }
            let res_base = n_kl * take * hidden_dim;
            let res_dst = captured * dim;
            for idx in 0..(take * dim) {
                res_out[res_dst + idx] = unsafe { std::ptr::read_volatile(ptr.add(res_base + idx)) };
            }
            captured += take;
        }

        self.reset_cache();
        Ok((keys_out, res_out))
    }

    /// Evaluate the exact full-vocabulary `[vocab_size]` FP32 next-token logit vector at the end
    /// of `token_ids` on the RTX 3080 (used for exact `KL(P_base || P_patched)` control gates!).
    pub fn eval_next_token_logits(&mut self, token_ids: &[i32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("eval_next_token_logits: empty token_ids".into());
        }
        let total_len = token_ids.len().min(self.cfg.max_ctx);
        let toks = &token_ids[..total_len];

        let mut pos = 0usize;
        while pos < total_len - 1 {
            let take = (total_len - 1 - pos).min(PREFILL_MAX_CHUNK);
            self.run_prefill_chunk(&toks[pos..pos + take], pos, None, None, None)?;
            pos += take;
        }

        let cur_tok = toks[total_len - 1];
        let cur_pos = (total_len - 1) as i32;
        let ctrl_ptr = self.ctrl_buf.as_ptr() as *mut QwenDecodeCtrl;
        unsafe {
            (*ctrl_ptr).start_token = cur_tok;
            (*ctrl_ptr).start_pos = cur_pos;
            (*ctrl_ptr).max_steps = 1;
            (*ctrl_ptr).temperature = 0.0;
            (*ctrl_ptr).rng_seed = 1;
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).steps_completed = 0;
            (*ctrl_ptr).stop_reason = 0;
            (*ctrl_ptr).export_logits = 1;
        }
        fence(Ordering::SeqCst);

        let mut mk_args = [0u8; 16];
        put_u64(&mut mk_args, 0, self.model_desc_gpu.va_addr);
        put_u64(&mut mk_args, 8, self.ctrl_buf.va_addr);

        self.gpu.timeline_value += 1;
        let sig_val = self.gpu.timeline_value;
        let sig_va = self.gpu.timeline_buf.va_addr;
        let k_dec = &self.module.kernels["qwen_persistent_decode_megakernel"];
        let (qmd_va, _) = self.gpu.prepare_qmd(
            k_dec,
            (self.gpu.active_sms, 1, 1),
            (512, 1, 1),
            &mk_args,
            Some((sig_va, sig_val)),
        );
        self.gpu.launch_qmd_chain(qmd_va, true)?;
        self.gpu.wait_timeline(sig_val, Duration::from_secs(10))?;
        self.reset_cache();

        let err = unsafe { (*ctrl_ptr).error_code };
        if err != 0 {
            let _ = self.gpu.zero_vram(&self.ws_bar_state);
            return Err(format!("Logits export watchdog tripped at {}", err));
        }

        fence(Ordering::SeqCst);
        let ptr = self.logits_out_buf.as_ptr() as *const f32;
        let mut out = Vec::with_capacity(self.cfg.vocab_size);
        for i in 0..self.cfg.vocab_size {
            out.push(unsafe { std::ptr::read_volatile(ptr.add(i)) });
        }
        Ok(out)
    }
}
