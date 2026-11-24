//! GGUF-to-VRAM Weight & Scratch Arena Loader for `QwenMegakernelEngine`.

#![allow(dead_code)]

use std::time::Instant;

use super::qwen_abi::{
    QwenDecodeCtrl, QwenLayerDesc, QwenModelDesc, DEFAULT_SPEC_CROSSOVER_4B,
    DEFAULT_SPEC_CROSSOVER_8B, EMBEDDED_CUBIN,
};
use crate::gguf::{
    decode_1d_f32, extract_quant_weight_bytes, precompute_freqs_cis_f32, resolve_model_path,
    BpeTokenizer, GgufFile, QwenModelConfig,
};
use crate::hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};

pub(super) const PREFILL_MAX_CHUNK: usize = 64;
pub(super) const EMBED_MAX_POOL_ROWS: usize = 512;
pub(super) const MLP_KV_HOST_BUF_BYTES: usize = 16 << 20;

pub(super) struct QwenLoadedVram {
    pub module: CubinModule,
    pub cfg: QwenModelConfig,
    pub tok: BpeTokenizer,
    pub owned_vram: Vec<VramBuffer>,
    pub layers_host: Vec<QwenLayerDesc>,
    pub layers_gpu: VramBuffer,
    pub model_desc_gpu: VramBuffer,
    pub num_layers_cfg: usize,
    pub default_crossover: usize,
    pub ws_x_res: VramBuffer,
    pub ws_x_res2: VramBuffer,
    pub ws_xq: VramBuffer,
    pub ws_xds: VramBuffer,
    pub ws_qkv: VramBuffer,
    pub ws_v_sep: VramBuffer,
    pub ws_q_roped: VramBuffer,
    pub ws_attn_out: VramBuffer,
    pub ws_gate_up: VramBuffer,
    pub ws_bar_state: VramBuffer,
    pub ws_embed_pool: VramBuffer,
    pub ws_pool_acc: VramBuffer,
    pub ws_pool_norm: VramBuffer,
    pub zero_bf16_gpu: VramBuffer,
    pub freqs_cis_gpu: VramBuffer,
    pub output_norm_va: u64,
    pub unit_l2_va: u64,
    pub emb_gpu_va: u64,
    pub emb_type: i32,
    pub ctrl_buf: HostMappedBuffer,
    pub prefill_toks_buf: HostMappedBuffer,
    pub embed_out_buf: HostMappedBuffer,
    pub embed_in_buf: HostMappedBuffer,
    pub logits_out_buf: HostMappedBuffer,
    pub hooks_out_buf: HostMappedBuffer,
    pub mlp_kv_buf: HostMappedBuffer,
}

pub(super) fn load_qwen_vram(
    gpu: &mut NvGpu,
    model_ref: &str,
    max_context: usize,
) -> Result<QwenLoadedVram, String> {
        assert_eq!(std::mem::size_of::<QwenLayerDesc>(), 176);
        assert_eq!(std::mem::size_of::<QwenModelDesc>(), 224);

        let t0 = Instant::now();
        let local_path = resolve_model_path(model_ref)?;
        let stem = local_path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen");
        let gguf = GgufFile::open(&local_path)?;
        let tok = gguf.build_tokenizer()?;
        let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_context, stem)?;

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

        let (lm_head_va, lm_head_type) = if gguf.has_tensor("output.weight") {
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

        Ok(QwenLoadedVram {
            module,
            cfg,
            tok,
            owned_vram,
            layers_host,
            layers_gpu,
            model_desc_gpu,
            num_layers_cfg,
            default_crossover,
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
        })
}

use std::path::Path;
use std::time::UNIX_EPOCH;
use crate::lora::{prepare_lora_from_path, LoraStatus, QwenLayerLoraDesc, LORA_ARENA_DEFAULT_BYTES};

const TOPIC_LORA_SLOT_BYTES: u64 = 8 << 20;
pub const TOPIC_LORA_NUM_SLOTS: usize = 8;

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

pub(super) struct QwenLoraArena {
    lora_arena: Option<VramBuffer>,
    lora_descs_host: Vec<QwenLayerLoraDesc>,
    cached_lora_state: Option<(String, f32, Vec<QwenLayerLoraDesc>, Vec<u64>, LoraStatus)>,
    topic_lora_pool: Option<VramBuffer>,
    topic_lora_slots: Vec<Option<CachedTopicSlot>>,
    topic_lora_clock: u64,
    status: LoraStatus,
}

impl QwenLoraArena {
    pub(super) fn new(num_layers: usize) -> Self {
        Self {
            lora_arena: None,
            lora_descs_host: vec![QwenLayerLoraDesc::default(); num_layers],
            cached_lora_state: None,
            topic_lora_pool: None,
            topic_lora_slots: vec![None; TOPIC_LORA_NUM_SLOTS],
            topic_lora_clock: 0,
            status: LoraStatus::default(),
        }
    }

    #[inline]
    pub(super) fn is_active(&self) -> bool {
        self.status.active
    }

    #[inline]
    pub(super) fn active_path_and_scale(&self) -> Option<(&str, f32)> {
        if self.status.active {
            Some((self.status.path.as_str(), self.status.scale))
        } else {
            None
        }
    }

    pub(super) fn status_json(&self) -> String {
        let mut esc = String::with_capacity(self.status.path.len());
        for c in self.status.path.chars() {
            match c {
                '"' => esc.push_str("\\\""),
                '\\' => esc.push_str("\\\\"),
                '\n' => esc.push_str("\\n"),
                '\r' => esc.push_str("\\r"),
                '\t' => esc.push_str("\\t"),
                other => esc.push(other),
            }
        }
        format!(
            "{{\"active\":{},\"path\":\"{}\",\"rank\":{},\"alpha\":{:.2},\"scale\":{:.4},\
              \"adapted_tensors\":{},\"vram_bytes\":{},\"swap_ms\":{:.3}}}",
            self.status.active,
            esc,
            self.status.rank,
            self.status.alpha,
            self.status.scale,
            self.status.num_adapted_tensors,
            self.status.vram_bytes,
            self.status.swap_ms
        )
    }

    #[inline]
    pub(super) fn layer_desc(&self, layer: usize) -> QwenLayerLoraDesc {
        self.lora_descs_host[layer]
    }

    pub(super) fn cache_summary(&self) -> (usize, usize, bool) {
        let used_slots = self.topic_lora_slots.iter().filter(|s| s.is_some()).count();
        (used_slots, TOPIC_LORA_NUM_SLOTS, self.cached_lora_state.is_some())
    }

    fn upload_layers(gpu: &mut NvGpu, layers_gpu: &VramBuffer, layers_host: &[QwenLayerDesc]) -> Result<(), String> {
        let layers_bytes = unsafe {
            std::slice::from_raw_parts(
                layers_host.as_ptr() as *const u8,
                layers_host.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let layers_buf = VramBuffer {
            va_addr: layers_gpu.va_addr,
            size: layers_gpu.size,
            paddrs: layers_gpu.paddrs.clone(),
        };
        gpu.upload_to_vram(&layers_buf, 0, layers_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_adapter(
        &mut self,
        gpu: &mut NvGpu,
        cfg: &QwenModelConfig,
        layers_gpu: &VramBuffer,
        layers_host: &mut [QwenLayerDesc],
        path: &str,
        user_scale: f32,
    ) -> Result<LoraStatus, String> {
        let st_fp = adapter_file_fingerprint(path);

        if self.topic_lora_pool.is_some() {
            if let Some(hit_idx) = self.topic_lora_slots.iter().position(|slot_opt| {
                slot_opt.as_ref().map_or(false, |s| {
                    s.path == path && (s.user_scale - user_scale).abs() < 1e-6 && s.fingerprint == st_fp
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
                for l in 0..cfg.num_layers {
                    layers_host[l].lora_ptr = ptrs[l];
                }
                self.lora_descs_host = descs;
                Self::upload_layers(gpu, layers_gpu, layers_host)?;
                st.active = true;
                st.swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                self.status = st.clone();
                return Ok(st);
            }
        }

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
                for l in 0..cfg.num_layers {
                    layers_host[l].lora_ptr = c_ptrs[l];
                }
                self.lora_descs_host = c_descs;
                Self::upload_layers(gpu, layers_gpu, layers_host)?;
                st.active = true;
                st.swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                self.status = st.clone();
                return Ok(st);
            }
        }

        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let mut prep = prepare_lora_from_path(
            path,
            user_scale,
            cfg.num_layers,
            cfg.dim,
            cfg.hidden_dim,
            q_dim,
            kv_dim,
        )?;

        let desc_sz = std::mem::size_of::<QwenLayerLoraDesc>();
        let patch_and_set_ptrs = |layer_descs: &mut [QwenLayerLoraDesc],
                                  arena_bytes: &mut [u8],
                                  lh: &mut [QwenLayerDesc],
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
                lh[l].lora_ptr = if ld_lora.active_mask != 0 {
                    arena_va + (l * desc_sz) as u64
                } else {
                    0
                };
            }
        };

        if (prep.arena_bytes.len() as u64) <= TOPIC_LORA_SLOT_BYTES {
            if self.topic_lora_pool.is_none() {
                self.topic_lora_pool = Some(
                    gpu.alloc_vram(TOPIC_LORA_SLOT_BYTES * (TOPIC_LORA_NUM_SLOTS as u64))?,
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

            patch_and_set_ptrs(&mut prep.layer_descs, &mut prep.arena_bytes, layers_host, arena_va);
            gpu.upload_to_vram(&pool_buf, slot_offset, &prep.arena_bytes)?;
            Self::upload_layers(gpu, layers_gpu, layers_host)?;

            let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
            let saved_ptrs: Vec<u64> = layers_host.iter().map(|ld| ld.lora_ptr).collect();
            self.lora_descs_host = prep.layer_descs;
            self.status = LoraStatus {
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
                status: self.status.clone(),
            });
            return Ok(self.status.clone());
        }

        let need_bytes = (prep.arena_bytes.len() as u64).max(LORA_ARENA_DEFAULT_BYTES);
        let need_realloc = self.lora_arena.as_ref().map_or(true, |b| b.size < need_bytes);
        if need_realloc {
            self.lora_arena = Some(gpu.alloc_vram(need_bytes)?);
        }
        let t_swap = Instant::now();
        let arena_va = self.lora_arena.as_ref().unwrap().va_addr;
        patch_and_set_ptrs(&mut prep.layer_descs, &mut prep.arena_bytes, layers_host, arena_va);

        let arena_ref = self.lora_arena.as_ref().unwrap();
        let arena_buf = VramBuffer {
            va_addr: arena_ref.va_addr,
            size: arena_ref.size,
            paddrs: arena_ref.paddrs.clone(),
        };
        gpu.upload_to_vram(&arena_buf, 0, &prep.arena_bytes)?;
        Self::upload_layers(gpu, layers_gpu, layers_host)?;

        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        let saved_ptrs: Vec<u64> = layers_host.iter().map(|ld| ld.lora_ptr).collect();
        self.lora_descs_host = prep.layer_descs;
        self.status = LoraStatus {
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
            self.status.clone(),
        ));
        Ok(self.status.clone())
    }

    pub(super) fn clear(
        &mut self,
        gpu: &mut NvGpu,
        num_layers: usize,
        layers_gpu: &VramBuffer,
        layers_host: &mut [QwenLayerDesc],
    ) -> Result<LoraStatus, String> {
        let t_swap = Instant::now();
        for l in 0..num_layers {
            layers_host[l].lora_ptr = 0;
            self.lora_descs_host[l] = QwenLayerLoraDesc::default();
        }
        Self::upload_layers(gpu, layers_gpu, layers_host)?;
        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
        self.status = LoraStatus {
            active: false,
            swap_ms,
            ..Default::default()
        };
        Ok(self.status.clone())
    }
}
